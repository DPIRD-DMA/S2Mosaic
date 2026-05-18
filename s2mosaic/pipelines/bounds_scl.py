"""SCL mask fetch helpers for bounds/AOI mosaics."""

from typing import Any, List, Tuple, Union

import numpy as np
import numpy.typing as npt
import rasterio as rio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import Window, bounds as window_bounds, from_bounds

from ..geometry import Bbox, _SCL_ADAPTIVE_BLOCK_SAVING_FRACTION, _target_grid
from ..helpers import disk_cache, get_rasterio_resampling, with_scene_retry
from ..sources import Source
from ..stac_bounds import _BoundsItemLike


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


def _fetch_one_scl_key(
    item: _BoundsItemLike,
    source: Source,
    bounds_target: Bbox,
    target_crs: int,
    mask_resolution: int,
) -> str:
    return f"{source.name}|{item.id}|{bounds_target}|{target_crs}|{mask_resolution}"


@disk_cache("scl", key_fn=_fetch_one_scl_key)
@with_scene_retry()
def _fetch_one_scl(
    item: _BoundsItemLike,
    source: Source,
    bounds_target: Bbox,
    target_crs: int,
    mask_resolution: int,
) -> npt.NDArray[Any]:
    """Fetch one scene's SCL band as (h, w) uint8 at ``mask_resolution``."""
    transform, width, height, target_crs_obj = _target_grid(
        bounds_target, mask_resolution, target_crs
    )
    rio_resampling = get_rasterio_resampling("nearest")
    href = source.sign(item.assets[source.asset_name("SCL")].href)
    arr = _read_warpvrt(
        href, 1, transform, width, height, target_crs_obj, rio_resampling
    )
    return arr.astype(np.uint8)


@with_scene_retry()
def _fetch_one_scl_tiled(
    item: _BoundsItemLike,
    source: Source,
    bounds_target: Bbox,
    target_crs: int,
    mask_resolution: int,
    width: int,
    height: int,
    tile_specs: List[Tuple[int, int, int, int]],
) -> npt.NDArray[Any]:
    """Fetch one scene's SCL band using sparse AOI tile windows."""
    if tile_specs == [(0, 0, height, width)]:
        return _fetch_one_scl(item, source, bounds_target, target_crs, mask_resolution)

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
    out = np.zeros((height, width), dtype=np.uint8)
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
            for r, c, h, w in tile_specs:
                out[r : r + h, c : c + w] = vrt.read(1, window=window_cls(c, r, w, h))
    return out


def _source_block_count_for_scl_tiles(
    item: _BoundsItemLike,
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
    item: _BoundsItemLike,
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

    full_blocks = _source_block_count_for_scl_tiles(
        item,
        source,
        bounds_target,
        target_crs,
        mask_resolution,
        width,
        height,
        full_spec,
    )
    tiled_blocks = _source_block_count_for_scl_tiles(
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
