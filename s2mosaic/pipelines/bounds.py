"""Mosaic creation for arbitrary bounding boxes (single or multi-MGRS-tile).

Each scene's bands are fetched on-the-fly through a rasterio WarpedVRT snapped
to a common UTM grid, so scenes from MGRS tiles in different native projections
are all read into the same output frame. Bounds mode computes per-scene masks
on the target grid, keeps only scenes that can contribute pixels, then uses the
shared tile-streamed aggregation path.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import cv2
import numpy as np
import numpy.typing as npt
from pystac.item_collection import ItemCollection
from rasterio.errors import RasterioIOError
from rasterio.crs import CRS
from tqdm.auto import tqdm

from ..frequent_coverage import get_frequent_coverage_for_bbox
from ..config import (
    CLOUD_MASK_OCM,
    CLOUD_MASK_SCL,
    MOSAIC_FIRST,
    MosaicRequest,
)
from ..helpers import (
    SceneFetchError,
    define_dates,
    get_band_template,
    get_rasterio_resampling,
    pick_ocm_resolution,
    report_dropped_scenes,
    with_scene_retry,
)
from ..masking import compute_masks_from_array, compute_masks_from_scl
from ..aggregation import (
    adaptive_tile_specs_for_masks,
    run_tile_aggregation,
    write_tile_aggregation_geotiff,
)
from ..streaming import (
    iter_ordered_fetches,
)
from ..readers import (
    DEFAULT_TILE_WORKERS,
    should_prewarm_sources,
)
from ..output import (
    finalize_output,
    output_request_hash,
    output_sidecar_metadata,
    resolve_export_path,
    write_output_sidecar,
)
from ..sources import Source
from ..stac import (
    ITEM_COL,
    add_item_info,
    sort_items,
)
from ..geometry import (
    Aoi,
    Bbox,
    _OCM_BANDS,
    _expand_window_for_ocm_context,
    _grid_shape_for_bounds,
    _rasterize_aoi_mask,
    _scene_window_from_geometry,
    _scene_window_in_target,
    _snap_bounds_to_grid,
    _target_grid,
    _window_bounds_in_target,
    pick_utm_epsg,
    reproject_aoi,
    reproject_bbox,
)
from .._types import BoundsItemLike, MaskFetch, SceneWindow
from ..readers import make_bounds_tile_reader
from ..stac_bounds import _search_for_items_by_aoi, _search_for_items_by_bbox
from .bounds_scl import (
    _fetch_one_scl,
    _fetch_one_scl_tiled,
    _read_band_at_target_window,
    _should_use_tiled_scl_fetch,
)

logger = logging.getLogger(__name__)
SCL_NATIVE_RESOLUTION = 20
BOUNDS_ADAPTIVE_SCAN_PIXEL_LIMIT = 20_000 * 20_000


def _mask_resolution_for_request(request: MosaicRequest) -> int:
    """Choose the cloud-mask read resolution for bounds/AOI mode."""
    if request.cloud_mask == CLOUD_MASK_SCL:
        # Sentinel-2 L2A SCL is a 20m asset. Reading it onto a 10m bounds grid
        # quadruples mask pixels and network work without adding SCL detail.
        return max(request.resolution, SCL_NATIVE_RESOLUTION)
    return pick_ocm_resolution(request.resolution)


class _AllTrueMask:
    """Array-like boolean mask that materialises only requested windows."""

    def __init__(self, shape: Tuple[int, int]):
        self.shape = shape

    def __getitem__(self, key: Any) -> npt.NDArray[np.bool_]:
        row_key, col_key = key
        h = row_key.stop - row_key.start
        w = col_key.stop - col_key.start
        return np.ones((h, w), dtype=bool)

    def __array__(self, dtype: Optional["np.dtype[Any]"] = None) -> npt.NDArray[Any]:
        arr = np.ones(self.shape, dtype=bool)
        return arr.astype(dtype, copy=False) if dtype is not None else arr


class _ResampledBoolMask:
    """Array-like mask that resamples source mask windows on demand."""

    def __init__(
        self,
        source: npt.NDArray[Any],
        shape: Tuple[int, int],
        coverage: Optional[Any] = None,
    ):
        self._source = source
        self.shape = shape
        self._coverage = coverage

    def __getitem__(self, key: Any) -> npt.NDArray[np.bool_]:
        row_key, col_key = key
        row_start = int(row_key.start or 0)
        row_stop = int(row_key.stop)
        col_start = int(col_key.start or 0)
        col_stop = int(col_key.stop)
        dst_h = row_stop - row_start
        dst_w = col_stop - col_start
        src_h, src_w = self._source.shape
        out_h, out_w = self.shape

        src_row_start = max(0, int(np.floor(row_start * src_h / out_h)))
        src_row_stop = min(src_h, int(np.ceil(row_stop * src_h / out_h)))
        src_col_start = max(0, int(np.floor(col_start * src_w / out_w)))
        src_col_stop = min(src_w, int(np.ceil(col_stop * src_w / out_w)))
        tile = self._source[src_row_start:src_row_stop, src_col_start:src_col_stop]
        if tile.shape != (dst_h, dst_w):
            tile = cv2.resize(
                tile.astype(np.uint8),
                (dst_w, dst_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        else:
            tile = tile.astype(bool, copy=False)
        if self._coverage is not None:
            tile &= self._coverage[key]
        return tile

    def __array__(self, dtype: Optional["np.dtype[Any]"] = None) -> npt.NDArray[Any]:
        arr = self[slice(0, self.shape[0]), slice(0, self.shape[1])]
        return arr.astype(dtype, copy=False) if dtype is not None else arr


class _WindowedBoolMask:
    """Sparse bool mask: a dense block placed at (row_off, col_off) within a
    larger logical shape; pixels outside the block read as False.

    Lets per-scene combo masks store only the scene's footprint within
    bounds_target instead of the full bounds, which would be ruinous for
    wide AOIs with hundreds of scenes (e.g. 20k x 9.6k bool x 200 scenes
    ≈ 38 GB if dense). With a window, each scene holds ~one MGRS-tile-sized
    block (~3666 x 3666 at 30 m ≈ 13 MB).
    """

    def __init__(
        self,
        block: npt.NDArray[Any],
        col_off: int,
        row_off: int,
        shape: Tuple[int, int],
    ):
        self._block = np.ascontiguousarray(block, dtype=bool)
        self._col_off = col_off
        self._row_off = row_off
        self.shape = shape

    def __getitem__(self, key: Any) -> npt.NDArray[np.bool_]:
        row_key, col_key = key
        row_start = int(row_key.start or 0)
        row_stop = int(row_key.stop)
        col_start = int(col_key.start or 0)
        col_stop = int(col_key.stop)
        out = np.zeros((row_stop - row_start, col_stop - col_start), dtype=bool)
        block_h, block_w = self._block.shape
        r0 = max(row_start, self._row_off)
        r1 = min(row_stop, self._row_off + block_h)
        c0 = max(col_start, self._col_off)
        c1 = min(col_stop, self._col_off + block_w)
        if r0 < r1 and c0 < c1:
            out[r0 - row_start : r1 - row_start, c0 - col_start : c1 - col_start] = (
                self._block[
                    r0 - self._row_off : r1 - self._row_off,
                    c0 - self._col_off : c1 - self._col_off,
                ]
            )
        return out

    def __array__(self, dtype: Optional["np.dtype[Any]"] = None) -> npt.NDArray[Any]:
        h, w = self.shape
        out = np.zeros((h, w), dtype=bool)
        block_h, block_w = self._block.shape
        out[
            self._row_off : self._row_off + block_h,
            self._col_off : self._col_off + block_w,
        ] = self._block
        return out.astype(dtype, copy=False) if dtype is not None else out

    def any(self) -> bool:
        return bool(self._block.any())


@with_scene_retry()
def _fetch_one_ocm(
    item: BoundsItemLike,
    source: Source,
    bounds_target: Bbox,
    target_crs: int,
    ocm_resolution: int,
    scene_window: SceneWindow,
) -> MaskFetch:
    """Fetch R+G+NIR over the scene's footprint within ``bounds_target``.

    Reading at the scene's footprint (rather than the full bounds) is the
    difference between a per-scene tick costing one MGRS tile worth of work
    vs. the whole AOI. Pads to >=100 px so OCM has its required context, then
    returns a crop slice the caller applies to undo the padding.
    """
    expanded_window, crop = _expand_window_for_ocm_context(
        bounds_target, ocm_resolution, scene_window
    )
    read_bounds = _window_bounds_in_target(
        bounds_target, ocm_resolution, expanded_window
    )
    _, width, height, target_crs_obj = _target_grid(
        read_bounds, ocm_resolution, target_crs
    )
    rio_resampling = get_rasterio_resampling("nearest")

    def _read_band(band_name: str) -> npt.NDArray[Any]:
        href = source.sign(item.assets[source.asset_name(band_name)].href)
        try:
            return _read_band_at_target_window(
                href, 1, read_bounds, target_crs_obj, width, height, rio_resampling
            )
        except RasterioIOError as exc:
            raise RasterioIOError(
                f"OCM band {band_name} read failed for scene {item.id}"
            ) from exc

    with ThreadPoolExecutor(max_workers=len(_OCM_BANDS)) as executor:
        bands = list(executor.map(_read_band, _OCM_BANDS))
    arr = np.stack(bands, axis=0).astype(np.uint16)
    return MaskFetch(arr=arr, target_window=scene_window, crop=crop)


def _search_and_sort_bounds_items(
    *,
    bounds: Bbox,
    bounds_4326: Bbox,
    aoi_4326: Optional[Aoi],
    start_date: date,
    end_date: date,
    source: Source,
    additional_query: Optional[Dict[str, Any]],
    ignore_duplicate_items: bool,
    scene_order: str,
    scene_sort_fn: Optional[Callable[..., Any]],
) -> Tuple[ItemCollection, Any, List[BoundsItemLike]]:
    """Search bounds/AOI scenes and return sorted STAC items."""
    if aoi_4326 is not None:
        items = _search_for_items_by_aoi(
            aoi_4326=aoi_4326,
            start_date=start_date,
            end_date=end_date,
            source=source,
            additional_query=additional_query,
            ignore_duplicate_items=ignore_duplicate_items,
        )
    else:
        items = _search_for_items_by_bbox(
            bbox_4326=bounds_4326,
            start_date=start_date,
            end_date=end_date,
            source=source,
            additional_query=additional_query,
            ignore_duplicate_items=ignore_duplicate_items,
        )
    if len(items) == 0:
        raise ValueError(
            f"No scenes found for bounds {bounds} between "
            f"{start_date.isoformat()} and {end_date.isoformat()}"
        )

    items_with_orbits = add_item_info(items)
    if scene_sort_fn:
        sorted_items = scene_sort_fn(items=items_with_orbits)
    else:
        sorted_items = sort_items(items=items_with_orbits, scene_order=scene_order)
    return (
        items,
        sorted_items,
        cast(List[BoundsItemLike], sorted_items[ITEM_COL].tolist()),
    )


def _scene_window_for_item(
    item: BoundsItemLike,
    bounds_target: Bbox,
    target_crs: int,
    resolution: int,
) -> Optional[SceneWindow]:
    """Polygon-based scene window with bbox fallback for items missing geometry."""
    geometry = getattr(item, "geometry", None)
    if geometry is not None:
        window = _scene_window_from_geometry(
            geometry, bounds_target, target_crs, resolution
        )
        if window is not None:
            return window
    return _scene_window_in_target(item.bbox, bounds_target, target_crs, resolution)


def _stream_bounds_combo_masks(
    *,
    items_list: List[BoundsItemLike],
    source: Source,
    bounds_target: Bbox,
    target_crs: int,
    mask_resolution: int,
    mask_w: int,
    mask_h: int,
    coverage_mask: npt.NDArray[Any],
    cloud_mask: str,
    mosaic_method: str,
    tile_workers: Optional[int],
    ocm_batch_size: int,
    ocm_inference_dtype: str,
    scl_tile_specs: Optional[List[Tuple[int, int, int, int]]],
    show_progress: bool,
) -> Tuple[Dict[int, "_WindowedBoolMask"], List[Dict[str, str]]]:
    """Stream per-scene cloud masks and keep only masks that contribute pixels.

    Each scene's read window is clipped to its own footprint within
    ``bounds_target``, so per-scene OCM/SCL work scales with the scene
    (~one MGRS tile) rather than the full AOI extent.
    """
    n_time = len(items_list)
    logger.info(
        f"Streaming cloud mask over up to {n_time} scenes "
        f"(per-scene fetch at {mask_resolution}m, EPSG:{target_crs})"
    )

    # Pre-compute the scene-footprint window in bounds_target's grid for each
    # item. Scenes with no overlap (rare: STAC search uses bbox-in-lon/lat;
    # corner cases can still drop empty intersections) are skipped up-front so
    # we don't spend any worker time on them. Prefer item.geometry (polygon)
    # over item.bbox so the read window tracks the actual swath footprint
    # instead of its lon/lat bounding rectangle — meaningfully less nodata is
    # fed into OCM, especially for cross-UTM-zone scenes.
    scene_windows: List[Optional[SceneWindow]] = [
        _scene_window_for_item(item, bounds_target, target_crs, mask_resolution)
        for item in items_list
    ]
    valid_scene_idx = [i for i, w in enumerate(scene_windows) if w is not None]
    if len(valid_scene_idx) < n_time:
        logger.info(
            "Skipping %d/%d scenes with no footprint overlap after reprojection",
            n_time - len(valid_scene_idx),
            n_time,
        )

    kept_combo_masks: Dict[int, _WindowedBoolMask] = {}
    good_pixel_tracker = np.zeros((mask_h, mask_w), dtype=bool)
    dropped_scenes: List[Dict[str, str]] = []
    mask_progress: Optional["tqdm[Any]"] = None
    if show_progress:
        mask_progress = tqdm(
            total=n_time,
            desc=(
                f"Phase 1: streaming bands for {cloud_mask} cloud mask"
                if cloud_mask == "OCM"
                else f"Phase 1: streaming {cloud_mask} cloud masks"
            ),
            unit="scene",
        )

    # For ordered iteration we still walk through all items_list positions;
    # entries without a window are no-ops below.
    mask_fetch_iter: Optional[Iterator[Tuple[int, Union[MaskFetch, Exception]]]] = None
    phase1_workers = tile_workers if tile_workers is not None else DEFAULT_TILE_WORKERS

    def _fetch_scene(idx: int, item: BoundsItemLike) -> MaskFetch:
        window = scene_windows[idx]
        if window is None:
            raise SceneFetchError(
                f"Scene {item.id} does not intersect bounds_target after reprojection"
            )
        if cloud_mask == CLOUD_MASK_SCL:
            if scl_tile_specs is not None:
                return _fetch_one_scl_tiled(
                    item,
                    source,
                    bounds_target,
                    target_crs,
                    mask_resolution,
                    mask_w,
                    mask_h,
                    scl_tile_specs,
                    window,
                )
            return _fetch_one_scl(
                item, source, bounds_target, target_crs, mask_resolution, window
            )
        return _fetch_one_ocm(
            item, source, bounds_target, target_crs, mask_resolution, window
        )

    if cloud_mask == CLOUD_MASK_SCL:
        mask_fetch_iter = iter_ordered_fetches(
            items=items_list,
            fetch_fn=_fetch_scene,
            max_workers=phase1_workers,
        )
    elif cloud_mask == CLOUD_MASK_OCM:
        # Each OCM fetch already reads R/G/NIR in parallel. Keep scene-level
        # prefetch modest so download for the next scene overlaps inference
        # without multiplying concurrent reads too aggressively.
        mask_fetch_iter = iter_ordered_fetches(
            items=items_list,
            fetch_fn=_fetch_scene,
            max_workers=min(2, phase1_workers),
        )

    try:
        for scene_position in range(n_time):
            # FIRST mode: stop scanning once everything in coverage is filled.
            if (
                mosaic_method == MOSAIC_FIRST
                and (good_pixel_tracker | ~coverage_mask).all()
            ):
                logger.info(
                    "All in-coverage pixels filled after "
                    f"{scene_position}/{n_time} scenes — "
                    "skipping remaining cloud-mask fetches"
                )
                break

            try:
                assert mask_fetch_iter is not None
                scene_idx, mask_result = next(mask_fetch_iter)
            except StopIteration:
                break
            if isinstance(mask_result, Exception):
                if isinstance(mask_result, SceneFetchError):
                    dropped_scenes.append(
                        {
                            "id": items_list[scene_idx].id,
                            "reason": str(mask_result),
                        }
                    )
                    logger.warning(
                        f"Scene {scene_idx + 1}/{n_time} "
                        f"({items_list[scene_idx].id}): mask fetch failed, "
                        f"skipping ({mask_result})"
                    )
                    if mask_progress is not None:
                        mask_progress.update(1)
                    continue
                raise mask_result

            if cloud_mask == CLOUD_MASK_SCL:
                clear, valid = compute_masks_from_scl(mask_result.arr)
            else:
                clear, valid = compute_masks_from_array(
                    mask_result.arr,
                    batch_size=ocm_batch_size,
                    inference_dtype=ocm_inference_dtype,
                )
            clear = clear[mask_result.crop]
            valid = valid[mask_result.crop]
            if mask_progress is not None:
                mask_progress.update(1)
            combo_block = clear & valid

            col_off, row_off, win_w, win_h = mask_result.target_window
            if mosaic_method == MOSAIC_FIRST:
                tracker_slice = good_pixel_tracker[
                    row_off : row_off + win_h, col_off : col_off + win_w
                ]
                new_pixels = combo_block & ~tracker_slice
                if not new_pixels.any():
                    continue
                combo_block = new_pixels
            elif not combo_block.any():
                # All-cloud scene — no contribution to mean/percentile either.
                continue

            kept_combo_masks[scene_idx] = _WindowedBoolMask(
                combo_block, col_off, row_off, (mask_h, mask_w)
            )
            good_pixel_tracker[
                row_off : row_off + win_h, col_off : col_off + win_w
            ] |= combo_block
    finally:
        if mask_fetch_iter is not None:
            close = getattr(mask_fetch_iter, "close", None)
            if close is not None:
                close()

        if mask_progress is not None:
            # `first` mode can break the loop early. Snap the bar to total so tqdm
            # renders it as complete rather than red. Set ``n`` directly and force
            # a refresh — ``update`` honours min-interval throttling and a quick
            # ``close`` after may skip the final redraw in tqdm.notebook.
            if mask_progress.n < mask_progress.total:
                mask_progress.n = mask_progress.total
                mask_progress.refresh()
            mask_progress.close()

    if dropped_scenes:
        logger.warning(
            f"Mask phase: {len(dropped_scenes)}/{n_time} scenes failed to fetch "
            "after retries; continuing with the rest"
        )
        report_dropped_scenes(dropped_scenes, total=n_time)

    if not kept_combo_masks:
        raise RuntimeError(
            "No usable scenes — every scene was fully cloud-masked, invalid, "
            "or failed to fetch"
        )
    return kept_combo_masks, dropped_scenes


def run_bounds_pipeline(
    request: MosaicRequest,
    *,
    source: Source,
) -> Union[Tuple[npt.NDArray[Any], Dict[str, Any]], Path]:
    """Bounds/AOI-mode pipeline. Called from :func:`s2mosaic.mosaic` for non-grid AOIs.

    Searches the configured STAC source for Sentinel-2 L2A scenes intersecting
    ``request.bounds`` or ``request.aoi`` over the date window, streams per-scene
    cloud masks (skipping scenes that contribute no new pixels), then fetches user
    bands only for the kept scenes and aggregates them into a single mosaic on a
    UTM grid.

    Differs from the grid-mode pipeline in that the AOI is an arbitrary bbox
    (possibly spanning multiple MGRS tiles, possibly intersecting tiles in
    different UTM-zone projections), so each per-scene read goes through a
    rasterio WarpedVRT to land on one common grid in ``request.output_crs``.

    Args:
        request: Normalized and validated :class:`MosaicRequest`. See
            :func:`s2mosaic.mosaic` for the meaning of each field.
        source: STAC source provider (e.g. MPC, AWS) supplying the catalog,
            asset signing, and band/asset naming.

    Returns:
        ``(array, profile)`` if no export path is requested, otherwise the
        ``Path`` of the written GeoTIFF. ``array`` has shape
        ``(bands, height, width)`` and dtype ``uint8`` (visual) or ``uint16``.

    Raises:
        ValueError: If no scenes are found for the requested AOI / date window.
        RuntimeError: If scenes were found but every scene was fully
            cloud-masked, invalid, or failed to fetch.
    """
    bands = request.bands
    additional_query = request.additional_query
    assert request.start_year is not None
    assert bands is not None
    assert additional_query is not None

    bounds = request.bounds
    aoi = request.aoi
    if bounds is None and aoi is not None:
        bounds = aoi.bounds
    if bounds is None:
        raise ValueError("bounds or aoi must be provided")

    logger.info(
        f"Creating mosaic for bounds {bounds} (EPSG:{request.input_crs}) "
        f"from {request.start_year}-{request.start_month:02d}-{request.start_day:02d} "
        f"+ {request.duration_years}y {request.duration_months}m "
        f"{request.duration_days}d using {request.mosaic_method} method "
        f"with bands {bands}"
    )

    is_visual = "visual" in bands

    aoi_4326 = reproject_aoi(aoi, request.input_crs, 4326) if aoi is not None else None
    bounds_4326 = (
        aoi_4326.bounds
        if aoi_4326 is not None
        else reproject_bbox(bounds, request.input_crs, 4326)
    )
    target_crs = request.output_crs
    if target_crs is None:
        cx = (bounds_4326[0] + bounds_4326[2]) / 2
        cy = (bounds_4326[1] + bounds_4326[3]) / 2
        target_crs = pick_utm_epsg(cx, cy)
        logger.info(f"Auto-picked target CRS: EPSG:{target_crs}")

    # bounds= always fills the rectangle (or its reprojected envelope for
    # cross-CRS): no synthesised polygon, no implicit mask. Callers who want
    # the lat/lng rectangle clipped after reprojection use aoi=shapely.box(...)
    # explicitly, which sets aoi_target below and engages the AOI mask path.
    aoi_target = (
        reproject_aoi(aoi, request.input_crs, target_crs) if aoi is not None else None
    )
    bounds_target = (
        aoi_target.bounds
        if aoi_target is not None
        else reproject_bbox(bounds, request.input_crs, target_crs)
    )
    if request.snap_to_source_grid:
        bounds_target = _snap_bounds_to_grid(bounds_target, request.resolution)
        logger.info(
            "Snapped target bounds to %dm grid: %s",
            request.resolution,
            bounds_target,
        )

    start_date, end_date = define_dates(
        request.start_year,
        request.start_month,
        request.start_day,
        request.duration_years,
        request.duration_months,
        request.duration_days,
    )

    mode = "aoi" if request.aoi is not None else "bounds"
    filename_hash = output_request_hash(
        request,
        mode=mode,
        start_date=start_date,
        end_date=end_date,
        source_name=source.name,
        target_crs=target_crs,
        bounds_4326=bounds_4326,
    )
    sidecar_metadata = output_sidecar_metadata(
        request,
        mode=mode,
        filename_hash=filename_hash,
        start_date=start_date,
        end_date=end_date,
        source_name=source.name,
        target_crs=target_crs,
        bounds_4326=bounds_4326,
    )
    export_path = resolve_export_path(
        output_dir=request.output_dir,
        output_path=request.output_path,
        start_date=start_date,
        end_date=end_date,
        scene_order=request.scene_order,
        mosaic_method=request.mosaic_method,
        bands=bands,
        percentile=request.percentile,
        bounds=bounds_4326,
        aoi=aoi_4326,
        source_name=source.name,
        resolution=request.resolution,
        cloud_mask=request.cloud_mask,
        filename_hash=filename_hash,
    )
    if export_path is not None:
        if export_path.exists() and not request.overwrite:
            return export_path

    items, _, items_list = _search_and_sort_bounds_items(
        bounds=bounds,
        bounds_4326=bounds_4326,
        aoi_4326=aoi_4326,
        start_date=start_date,
        end_date=end_date,
        source=source,
        additional_query=additional_query,
        ignore_duplicate_items=request.ignore_duplicate_items,
        scene_order=request.scene_order,
        scene_sort_fn=request.scene_sort_fn,
    )

    # Mask resolution depends on provider. OCM is fastest at coarser
    # resolutions, and SCL is native 20m, so avoid upsampling SCL to 10m
    # during the network-heavy mask scan.
    mask_resolution = _mask_resolution_for_request(request)
    logger.info(f"Cloud mask provider {request.cloud_mask} at {mask_resolution}m")

    # Each per-scene fetch reads a window clipped to the scene's footprint
    # within bounds_target, so per-scene work scales with the scene (~one MGRS
    # tile) rather than the full AOI. OCM context padding happens per-window
    # inside the fetcher.
    mask_w, mask_h = _grid_shape_for_bounds(bounds_target, mask_resolution)
    n_time = len(items_list)

    # Coverage mask at mask resolution — used for skip decisions.
    if request.min_coverage_fraction is not None:
        coverage_mask_ocm = get_frequent_coverage_for_bbox(
            scenes=items,
            bounds_target=bounds_target,
            target_crs=target_crs,
            width=mask_w,
            height=mask_h,
            resolution=mask_resolution,
            min_coverage_fraction=request.min_coverage_fraction,
        )
    else:
        coverage_mask_ocm = np.ones((mask_h, mask_w), dtype=bool)
    if aoi_target is not None:
        coverage_mask_ocm &= _rasterize_aoi_mask(
            aoi_target=aoi_target,
            bounds_target=bounds_target,
            resolution=mask_resolution,
            width=mask_w,
            height=mask_h,
        )
    possible_pixel_count = int(coverage_mask_ocm.sum())
    scl_tile_specs: Optional[List[Tuple[int, int, int, int]]] = None
    if (
        request.cloud_mask == CLOUD_MASK_SCL
        and aoi_target is not None
        and request.adaptive_tiling
        and possible_pixel_count > 0
    ):
        scl_tile_specs = adaptive_tile_specs_for_masks(
            masks=[coverage_mask_ocm],
            height=mask_h,
            width=mask_w,
            max_tile_size=min(2048, max(mask_h, mask_w)),
        )
        if not _should_use_tiled_scl_fetch(
            items_list,
            source,
            bounds_target,
            target_crs,
            mask_resolution,
            mask_w,
            mask_h,
            scl_tile_specs,
        ):
            scl_tile_specs = None

    kept_combo_masks_ocm, dropped_scenes = _stream_bounds_combo_masks(
        items_list=items_list,
        source=source,
        bounds_target=bounds_target,
        target_crs=target_crs,
        mask_resolution=mask_resolution,
        mask_w=mask_w,
        mask_h=mask_h,
        coverage_mask=coverage_mask_ocm,
        cloud_mask=request.cloud_mask,
        mosaic_method=request.mosaic_method,
        tile_workers=request.tile_workers,
        ocm_batch_size=request.ocm_batch_size,
        ocm_inference_dtype=request.ocm_inference_dtype,
        scl_tile_specs=scl_tile_specs,
        show_progress=request.show_progress,
    )
    sidecar_metadata["dropped_scenes"] = dropped_scenes

    # Prepare the user-resolution masks and target grid for tile aggregation.
    kept_indices = sorted(kept_combo_masks_ocm.keys())
    kept_items = [items_list[i] for i in kept_indices]
    logger.info(f"Streaming user bands for {len(kept_items)}/{n_time} kept scenes")

    # User grid is fixed upfront from bounds_target + resolution — every
    # per-scene fetch snaps to exactly this (transform, w, h), so accumulators
    # can be sized before any data is fetched.
    user_transform, w, h, _ = _target_grid(
        bounds_target, request.resolution, target_crs
    )
    n_bands = 3 if is_visual else len(bands)

    def _to_user_mask(mask: Any, coverage: Optional[Any] = None) -> Any:
        if mask.shape == (h, w) and coverage is None:
            return mask
        return _ResampledBoolMask(mask, (h, w), coverage=coverage)

    if (
        request.min_coverage_fraction is None
        and aoi_target is None
        and coverage_mask_ocm.all()
    ):
        coverage_mask: Any = _AllTrueMask((h, w))
    else:
        coverage_mask = _to_user_mask(coverage_mask_ocm)
    if aoi_target is not None:
        aoi_user_mask = _rasterize_aoi_mask(
            aoi_target=aoi_target,
            bounds_target=bounds_target,
            resolution=request.resolution,
            width=w,
            height=h,
        )
        if isinstance(coverage_mask, _AllTrueMask):
            coverage_mask = aoi_user_mask
        else:
            coverage_mask = _ResampledBoolMask(
                coverage_mask_ocm, (h, w), coverage=aoi_user_mask
            )
    logger.info(
        "Prepared lazy user-grid masks for %d scenes (%dx%d output)",
        len(kept_indices),
        h,
        w,
    )
    combo_masks_user = {
        i: _to_user_mask(kept_combo_masks_ocm[i], coverage=coverage_mask)
        for i in kept_indices
    }
    # OCM/SCL-resolution mask dict can be released once lazy user-grid wrappers exist.
    del kept_combo_masks_ocm

    # Phase 3 — tile-streamed aggregation. Same architecture as grid_id mode:
    # per-tile workers each read tile windows via the bounds tile reader
    # (WarpedVRT-backed, or direct read from a local cached file), apply
    # the per-tile slice of the precomputed mask, and aggregate by method.
    # Peak RAM is one tile per worker rather than the full N-scene stack.
    href_template, _, _ = get_band_template(bands)
    kept_items = [items_list[i] for i in kept_indices]
    read_fn = make_bounds_tile_reader(
        items=kept_items,
        href_template=href_template,
        source=source,
        bounds_target=bounds_target,
        target_crs=target_crs,
        user_transform=user_transform,
        width=w,
        height=h,
        resolution=request.resolution,
        resampling_method=request.resampling_method,
        prewarm=should_prewarm_sources(
            request.mosaic_method,
            request.min_observations,
            request.max_observations,
        ),
    )

    masks_in_order: List[Optional[npt.NDArray[Any]]] = [
        combo_masks_user[scene_idx] for scene_idx in kept_indices
    ]
    del combo_masks_user
    # Tile small AOIs as a single tile; cap large AOIs at 2048 so each
    # tile read from PC stays one big range request. Smaller tiles trade
    # round-trip latency for worker utilisation — a bad deal when reads
    # are network-bound.
    tile_size = min(2048, max(h, w))
    logger.info(
        "Tile-streaming %d scenes x %d bands at tile=%d (%dx%d output)",
        len(kept_items),
        n_bands,
        tile_size,
        h,
        w,
    )
    profile: Dict[str, Any] = {
        "driver": "GTiff",
        "dtype": (
            np.dtype(np.uint16)
            if request.include_observation_count
            else (np.dtype(np.uint8) if is_visual else np.dtype(np.uint16))
        ),
        "width": w,
        "height": h,
        "count": n_bands + (1 if request.include_observation_count else 0),
        "crs": CRS.from_epsg(target_crs),
        "transform": user_transform,
    }
    output_coverage_mask = (
        coverage_mask if request.min_coverage_fraction is not None else None
    )

    # Adaptive sub-tile floor: largest source COG block among the user bands,
    # so each tile read fetches whole source blocks rather than fringes. On
    # AWS 10m bands this is 1024; on MPC always 512. (Falls back to
    # source.default_block_size for unmeasured assets.)
    min_tile_size = source.max_block_size_for_bands(bands)
    logger.info(
        "Adaptive tile min size: %d (source=%s, bands=%s)",
        min_tile_size,
        source.name,
        bands,
    )
    adaptive_tiling = request.adaptive_tiling
    if adaptive_tiling and h * w > BOUNDS_ADAPTIVE_SCAN_PIXEL_LIMIT:
        adaptive_tiling = False
        logger.info(
            "Disabling adaptive tiling for large bounds output (%d pixels); "
            "using fixed %d-pixel tiles to avoid pre-scanning masks",
            h * w,
            tile_size,
        )

    try:
        if export_path is not None:
            result = write_tile_aggregation_geotiff(
                export_path=export_path,
                profile=profile,
                bands=bands,
                masks=masks_in_order,
                read_fn=read_fn,
                bands_count=n_bands,
                height=h,
                width=w,
                coverage_mask=coverage_mask,
                output_coverage_mask=output_coverage_mask,
                min_observations=request.min_observations,
                max_observations=request.max_observations,
                mosaic_method=request.mosaic_method,
                percentile=request.percentile,
                tile_size=tile_size,
                tile_workers=request.tile_workers,
                out_dtype=np.dtype(np.uint8) if is_visual else np.dtype(np.uint16),
                adaptive_tiling=adaptive_tiling,
                show_progress=request.show_progress,
                min_tile_size=min_tile_size,
                include_observation_count=request.include_observation_count,
            )
            write_output_sidecar(export_path, sidecar_metadata)
            return result

        output_array = run_tile_aggregation(
            masks=masks_in_order,
            read_fn=read_fn,
            bands_count=n_bands,
            height=h,
            width=w,
            coverage_mask=coverage_mask,
            min_observations=request.min_observations,
            max_observations=request.max_observations,
            mosaic_method=request.mosaic_method,
            percentile=request.percentile,
            tile_size=tile_size,
            tile_workers=request.tile_workers,
            out_dtype=np.dtype(np.uint8) if is_visual else np.dtype(np.uint16),
            adaptive_tiling=adaptive_tiling,
            show_progress=request.show_progress,
            min_tile_size=min_tile_size,
            include_observation_count=request.include_observation_count,
        )

        return finalize_output(
            array=output_array,
            profile=profile,
            bands=bands,
            coverage_mask=output_coverage_mask,
            export_path=export_path,
            include_observation_count=request.include_observation_count,
        )
    finally:
        close = getattr(read_fn, "close", None)
        if close is not None:
            close()
