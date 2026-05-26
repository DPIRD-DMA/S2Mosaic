"""SCL mask fetch helpers for bounds/AOI mosaics."""

from collections.abc import Sequence as SequenceABC
from typing import Any, List, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
import rasterio as rio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import Window, bounds as window_bounds, from_bounds

from ..geometry import (
    Bbox,
    _SCL_ADAPTIVE_BLOCK_SAVING_FRACTION,
    _target_grid,
    _window_bounds_in_target,
)
from .._types import BoundsItemLike, MaskFetch, SceneWindow
from ..helpers import get_rasterio_resampling, with_scene_retry
from ..sources import Source


def _read_warpvrt(
    href: str,
    indices: Union[int, List[int]],
    transform: Affine,
    width: int,
    height: int,
    target_crs_obj: CRS,
    rio_resampling: Any,
) -> npt.NDArray[Any]:
    """Open ``href`` and read ``indices`` through a WarpedVRT snapped to the grid."""
    with rio.open(href) as src:
        with WarpedVRT(
            src,
            crs=target_crs_obj,
            transform=transform,
            width=width,
            height=height,
            resampling=rio_resampling,
        ) as vrt:
            return vrt.read(indices)  # type: ignore[no-any-return, unused-ignore]


def _pick_overview_level(
    src_native_res: float, target_res: float, overview_factors: List[int]
) -> int:
    """GDAL OVERVIEW_LEVEL for reading at-or-finer than ``target_res``.

    Returns -1 to use native resolution; 0 = first overview, 1 = second, etc.
    Picks the highest-decimation overview whose resolution is still <= target,
    so the warp reads from the smallest source array that won't lose detail.
    """
    if target_res <= src_native_res or not overview_factors:
        return -1
    best_level = -1
    for idx, factor in sorted(enumerate(overview_factors), key=lambda item: item[1]):
        if src_native_res * factor <= target_res:
            best_level = idx
        else:
            break
    return best_level


def _read_band_at_target_window(
    href: str,
    band_idx: int,
    read_bounds: Bbox,
    target_crs_obj: CRS,
    target_width: int,
    target_height: int,
    rio_resampling: Any,
) -> npt.NDArray[Any]:
    """Read one band over ``read_bounds`` at target grid (width × height).

    Fast path when the source COG is already in ``target_crs``: a direct
    ``src.read(window, out_shape)`` lets rasterio pick the right COG overview,
    which is roughly an order of magnitude faster than asking WarpedVRT to
    warp full-resolution source data and resample to the same target grid.

    Cross-CRS path uses WarpedVRT, but first picks the appropriate source
    OVERVIEW_LEVEL so the warp reads from the closest-finer-than-target
    overview instead of full-resolution source — same byte savings as the
    fast path, just with a reprojection step on top.
    """
    target_res = (read_bounds[2] - read_bounds[0]) / target_width
    transform = Affine(
        (read_bounds[2] - read_bounds[0]) / target_width,
        0,
        read_bounds[0],
        0,
        -(read_bounds[3] - read_bounds[1]) / target_height,
        read_bounds[3],
    )
    with rio.open(href) as src:
        if src.crs == target_crs_obj:
            window = from_bounds(*read_bounds, transform=src.transform)
            return src.read(  # type: ignore[no-any-return]
                band_idx,
                window=window,
                out_shape=(target_height, target_width),
                resampling=rio_resampling,
                boundless=True,
                fill_value=0,
            )
        src_native_res = abs(src.transform.a)
        overview_factors = src.overviews(band_idx)
        overview_level = _pick_overview_level(
            src_native_res, target_res, overview_factors
        )
        warp_extras = (
            {} if overview_level < 0 else {"OVERVIEW_LEVEL": str(overview_level)}
        )
        with WarpedVRT(
            src,
            crs=target_crs_obj,
            transform=transform,
            width=target_width,
            height=target_height,
            resampling=rio_resampling,
            warp_extras=warp_extras,
        ) as vrt:
            return vrt.read(band_idx)  # type: ignore[no-any-return]


@with_scene_retry()
def _fetch_one_scl(
    item: BoundsItemLike,
    source: Source,
    bounds_target: Bbox,
    target_crs: int,
    mask_resolution: int,
    scene_window: SceneWindow,
) -> MaskFetch:
    """Fetch the scene's SCL band over its footprint within ``bounds_target``.

    Like the OCM fetcher, this reads only the scene's window in the target
    grid instead of the full bounds extent — so SCL read cost per scene
    stays bounded by the scene size regardless of how wide the bounds is.
    """
    _, _, win_w, win_h = scene_window
    read_bounds = _window_bounds_in_target(bounds_target, mask_resolution, scene_window)
    _, width, height, target_crs_obj = _target_grid(
        read_bounds, mask_resolution, target_crs
    )
    rio_resampling = get_rasterio_resampling("nearest")
    href = source.sign(item.assets[source.asset_name("SCL")].href)
    arr = _read_band_at_target_window(
        href, 1, read_bounds, target_crs_obj, width, height, rio_resampling
    )
    return MaskFetch(
        arr=arr.astype(np.uint8),
        target_window=scene_window,
        crop=(slice(0, win_h), slice(0, win_w)),
    )


@with_scene_retry()
def _fetch_one_scl_tiled(
    item: BoundsItemLike,
    source: Source,
    bounds_target: Bbox,
    target_crs: int,
    mask_resolution: int,
    width: int,
    height: int,
    tile_specs: List[Tuple[int, int, int, int]],
    scene_window: SceneWindow,
) -> MaskFetch:
    """Fetch one scene's SCL band using sparse AOI tile windows."""
    if tile_specs == [(0, 0, height, width)]:
        return _fetch_one_scl(
            item, source, bounds_target, target_crs, mask_resolution, scene_window
        )

    transform, expected_width, expected_height, target_crs_obj = _target_grid(
        bounds_target, mask_resolution, target_crs
    )
    if expected_width != width or expected_height != height:
        raise ValueError(
            "SCL tile grid does not match requested bounds grid: "
            f"expected {(expected_width, expected_height)}, got {(width, height)}"
        )

    rio_resampling = get_rasterio_resampling("nearest")
    href = source.sign(item.assets[source.asset_name("SCL")].href)
    scene_col, scene_row, scene_w, scene_h = scene_window
    scene_col_stop = scene_col + scene_w
    scene_row_stop = scene_row + scene_h
    relevant_specs: List[Tuple[int, int, int, int]] = []
    for r, c, h, w in tile_specs:
        row_start = max(r, scene_row)
        row_stop = min(r + h, scene_row_stop)
        col_start = max(c, scene_col)
        col_stop = min(c + w, scene_col_stop)
        if row_start < row_stop and col_start < col_stop:
            relevant_specs.append(
                (row_start, col_start, row_stop - row_start, col_stop - col_start)
            )

    if not relevant_specs:
        return MaskFetch(
            arr=np.zeros((scene_h, scene_w), dtype=np.uint8),
            target_window=scene_window,
            crop=(slice(0, scene_h), slice(0, scene_w)),
        )

    min_r = min(r for r, _, _, _ in relevant_specs)
    min_c = min(c for _, c, _, _ in relevant_specs)
    max_r = max(r + h for r, _, h, _ in relevant_specs)
    max_c = max(c + w for _, c, _, w in relevant_specs)
    out_h = max_r - min_r
    out_w = max_c - min_c
    out = np.zeros((out_h, out_w), dtype=np.uint8)
    with rio.open(href) as src:
        with WarpedVRT(
            src,
            crs=target_crs_obj,
            transform=transform,
            width=width,
            height=height,
            resampling=rio_resampling,
        ) as vrt:
            window_cls: Any = Window
            for r, c, h, w in relevant_specs:
                out[
                    r - min_r : r - min_r + h,
                    c - min_c : c - min_c + w,
                ] = vrt.read(1, window=window_cls(c, r, w, h))
    return MaskFetch(
        arr=out,
        target_window=(min_c, min_r, out_w, out_h),
        crop=(slice(0, out_h), slice(0, out_w)),
    )


def _source_block_count_for_scl_tiles(
    item: BoundsItemLike,
    source: Source,
    bounds_target: Bbox,
    target_crs: int,
    mask_resolution: int,
    width: int,
    height: int,
    tile_specs: List[Tuple[int, int, int, int]],
) -> int:
    """Estimate unique source COG blocks touched by output SCL tile windows."""
    transform, expected_width, expected_height, target_crs_obj = _target_grid(
        bounds_target, mask_resolution, target_crs
    )
    if expected_width != width or expected_height != height:
        raise ValueError(
            "SCL tile grid does not match requested bounds grid: "
            f"expected {(expected_width, expected_height)}, got {(width, height)}"
        )

    href = source.sign(item.assets[source.asset_name("SCL")].href)
    blocks: set[Tuple[int, int]] = set()
    window_cls: Any = Window
    with rio.open(href) as src:
        block_h, block_w = src.block_shapes[0]
        for r, c, h, w in tile_specs:
            out_bounds = window_bounds(window_cls(c, r, w, h), transform)
            src_bounds = transform_bounds(
                target_crs_obj,
                src.crs,
                *out_bounds,
                densify_pts=21,
            )
            src_window = from_bounds(*src_bounds, transform=src.transform)
            col_start = max(0, int(np.floor(src_window.col_off / block_w)))
            row_start = max(0, int(np.floor(src_window.row_off / block_h)))
            col_stop = min(
                int(np.ceil((src_window.col_off + src_window.width) / block_w)),
                int(np.ceil(src.width / block_w)),
            )
            row_stop = min(
                int(np.ceil((src_window.row_off + src_window.height) / block_h)),
                int(np.ceil(src.height / block_h)),
            )
            for block_row in range(row_start, row_stop):
                for block_col in range(col_start, col_stop):
                    blocks.add((block_row, block_col))
    return len(blocks)


def _should_use_tiled_scl_fetch(
    items: Union[BoundsItemLike, Sequence[BoundsItemLike]],
    source: Source,
    bounds_target: Bbox,
    target_crs: int,
    mask_resolution: int,
    width: int,
    height: int,
    tile_specs: List[Tuple[int, int, int, int]],
) -> bool:
    """Whether sparse SCL tile reads are likely to reduce COG block reads."""
    full_spec = [(0, 0, height, width)]
    if tile_specs == full_spec:
        return False

    sample_items = list(items) if isinstance(items, SequenceABC) else [items]
    if not sample_items:
        return False
    if len(sample_items) > 5:
        step = max(1, len(sample_items) // 5)
        sample_items = sample_items[::step][:5]

    full_blocks = 0
    tiled_blocks = 0
    for item in sample_items:
        full_blocks += _source_block_count_for_scl_tiles(
            item,
            source,
            bounds_target,
            target_crs,
            mask_resolution,
            width,
            height,
            full_spec,
        )
        tiled_blocks += _source_block_count_for_scl_tiles(
            item,
            source,
            bounds_target,
            target_crs,
            mask_resolution,
            width,
            height,
            tile_specs,
        )
    if full_blocks <= 0:
        return False
    return tiled_blocks <= full_blocks * _SCL_ADAPTIVE_BLOCK_SAVING_FRACTION
