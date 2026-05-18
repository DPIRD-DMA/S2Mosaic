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
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union, cast

import cv2
import numpy as np
import numpy.typing as npt
from pystac.item_collection import ItemCollection
from rasterio.crs import CRS
from tqdm.auto import tqdm

from ..frequent_coverage import get_frequent_coverage_for_bbox
from ..config import (
    CLOUD_MASK_OCM,
    CLOUD_MASK_SCL,
    MOSAIC_FIRST,
    MOSAIC_PERCENTILE,
    MosaicRequest,
)
from ..helpers import (
    SceneFetchError,
    define_dates,
    disk_cache,
    get_band_template,
    get_rasterio_resampling,
    pick_ocm_resolution,
    with_scene_retry,
)
from ..masking import compute_masks_from_array, compute_masks_from_scl
from ..aggregation import (
    adaptive_tile_specs_for_masks,
    run_tile_aggregation,
    write_tile_aggregation_geotiff,
)
from ..cache import (
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
    _expand_bounds_for_ocm_context,
    _grid_shape_for_bounds,
    _rasterize_aoi_mask,
    _target_grid,
    pick_utm_epsg,
    reproject_aoi,
    reproject_bbox,
)
from ..readers import make_bounds_tile_reader
from ..stac_bounds import (
    _BoundsItemLike,
    _search_for_items_by_aoi,
    _search_for_items_by_bbox,
)
from .bounds_scl import (
    _fetch_one_scl,
    _fetch_one_scl_tiled,
    _read_warpvrt,
    _should_use_tiled_scl_fetch,
)

logger = logging.getLogger(__name__)


def _fetch_one_ocm_key(
    item: _BoundsItemLike,
    source: Source,
    bounds_target: Bbox,
    target_crs: int,
    ocm_resolution: int,
) -> str:
    return f"{source.name}|{item.id}|{bounds_target}|{target_crs}|{ocm_resolution}"


@disk_cache("ocm", key_fn=_fetch_one_ocm_key)
@with_scene_retry()
def _fetch_one_ocm(
    item: _BoundsItemLike,
    source: Source,
    bounds_target: Bbox,
    target_crs: int,
    ocm_resolution: int,
) -> npt.NDArray[Any]:
    """Fetch one scene's OCM bands (B04, B03, B8A) as (3, h, w) uint16.

    Reads via rasterio + WarpedVRT so the mask loop can stream one scene at a
    time and skip fetching late scenes entirely once the no_data threshold or
    first-mode coverage is met.
    """
    transform, width, height, target_crs_obj = _target_grid(
        bounds_target, ocm_resolution, target_crs
    )
    rio_resampling = get_rasterio_resampling("nearest")

    def _read_band(band_name: str) -> npt.NDArray[Any]:
        href = source.sign(item.assets[source.asset_name(band_name)].href)
        return _read_warpvrt(
            href, 1, transform, width, height, target_crs_obj, rio_resampling
        )

    with ThreadPoolExecutor(max_workers=len(_OCM_BANDS)) as executor:
        bands = list(executor.map(_read_band, _OCM_BANDS))
    return np.stack(bands, axis=0).astype(np.uint16)  # type: ignore[no-any-return, unused-ignore]


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
    sort_method: str,
    sort_function: Optional[Callable[..., Any]],
) -> Tuple[ItemCollection, Any, List[_BoundsItemLike]]:
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
    if sort_function:
        sorted_items = sort_function(items=items_with_orbits)
    else:
        sorted_items = sort_items(items=items_with_orbits, sort_method=sort_method)
    return (
        items,
        sorted_items,
        cast(List[_BoundsItemLike], sorted_items[ITEM_COL].tolist()),
    )


def _stream_bounds_combo_masks(
    *,
    items_list: List[_BoundsItemLike],
    source: Source,
    bounds_target: Bbox,
    target_crs: int,
    mask_resolution: int,
    mask_w: int,
    mask_h: int,
    coverage_mask: npt.NDArray[Any],
    cloud_mask: str,
    mosaic_method: str,
    no_data_tolerance: Optional[float],
    possible_pixel_count: int,
    tile_workers: Optional[int],
    ocm_bounds_target: Bbox,
    ocm_crop: Optional[Tuple[slice, slice]],
    ocm_batch_size: int,
    ocm_inference_dtype: str,
    scl_tile_specs: Optional[List[Tuple[int, int, int, int]]],
    show_progress: bool,
) -> Dict[int, npt.NDArray[Any]]:
    """Stream per-scene cloud masks and keep only masks that contribute pixels."""
    n_time = len(items_list)
    logger.info(
        f"Streaming cloud mask over up to {n_time} scenes "
        f"(per-scene fetch at {mask_resolution}m, EPSG:{target_crs})"
    )
    kept_combo_masks: Dict[int, npt.NDArray[Any]] = {}
    good_pixel_tracker = np.zeros((mask_h, mask_w), dtype=bool)
    n_mask_fetch_failed = 0
    mask_progress: Optional["tqdm[Any]"] = None
    if show_progress:
        mask_progress = tqdm(
            total=n_time,
            desc=f"Phase 1: streaming cloud masks ({cloud_mask})",
            unit="scene",
        )
    mask_fetch_iter: Optional[
        Iterator[Tuple[int, Union[npt.NDArray[Any], Exception]]]
    ] = None
    phase1_workers = tile_workers if tile_workers is not None else DEFAULT_TILE_WORKERS
    if cloud_mask == CLOUD_MASK_SCL:
        mask_fetch_iter = iter_ordered_fetches(
            items=items_list,
            fetch_fn=lambda _i, item: (
                _fetch_one_scl_tiled(
                    item,
                    source,
                    bounds_target,
                    target_crs,
                    mask_resolution,
                    mask_w,
                    mask_h,
                    scl_tile_specs,
                )
                if scl_tile_specs is not None
                else _fetch_one_scl(
                    item, source, bounds_target, target_crs, mask_resolution
                )
            ),
            max_workers=phase1_workers,
        )
    elif cloud_mask == CLOUD_MASK_OCM:
        # Each OCM fetch already reads R/G/NIR in parallel. Keep scene-level
        # prefetch modest so download for the next scene overlaps inference
        # without multiplying concurrent reads too aggressively.
        mask_fetch_iter = iter_ordered_fetches(
            items=items_list,
            fetch_fn=lambda _i, item: _fetch_one_ocm(
                item,
                source,
                ocm_bounds_target,
                target_crs,
                mask_resolution,
            ),
            max_workers=min(2, phase1_workers),
        )

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
            if isinstance(mask_result, Exception):
                raise mask_result
            if mask_progress is not None:
                mask_progress.update(1)
        except StopIteration:
            break
        except SceneFetchError as e:
            if mask_progress is not None:
                mask_progress.update(1)
            n_mask_fetch_failed += 1
            logger.warning(
                f"Scene {scene_idx + 1}/{n_time} "
                f"({items_list[scene_idx].id}): mask fetch failed, "
                f"skipping ({e})"
            )
            continue

        if cloud_mask == CLOUD_MASK_SCL:
            clear, valid = compute_masks_from_scl(mask_result)
        else:
            clear, valid = compute_masks_from_array(
                mask_result,
                batch_size=ocm_batch_size,
                inference_dtype=ocm_inference_dtype,
            )
            if ocm_crop is not None:
                clear = clear[ocm_crop]
                valid = valid[ocm_crop]
        combo = clear & valid

        if mosaic_method == MOSAIC_FIRST:
            new_pixels = combo & ~good_pixel_tracker
            if not new_pixels.any():
                continue
            combo = new_pixels
        elif not combo.any():
            # All-cloud scene — no contribution to mean/percentile either.
            continue

        kept_combo_masks[scene_idx] = combo
        good_pixel_tracker |= combo

        if (
            no_data_tolerance is not None
            and mosaic_method != MOSAIC_PERCENTILE
            and possible_pixel_count > 0
        ):
            completed = int((coverage_mask & good_pixel_tracker).sum())
            no_data_sum = possible_pixel_count - completed
            no_data_pct = (1 - completed / possible_pixel_count) * 100
            logger.info(
                f"Scene {scene_idx + 1}/{n_time} kept; no-data {no_data_pct:.1f}%"
            )
            if no_data_sum < possible_pixel_count * no_data_tolerance:
                logger.info(
                    f"no_data_tolerance met after {len(kept_combo_masks)} kept "
                    f"scenes ({scene_idx + 1}/{n_time} examined)"
                )
                break

    if mask_progress is not None:
        # `first` mode and no_data_tolerance can break the loop early. Snap
        # the bar to total so tqdm renders it as complete rather than red.
        remaining = mask_progress.total - mask_progress.n
        if remaining > 0:
            mask_progress.update(remaining)
        mask_progress.close()

    if n_mask_fetch_failed:
        logger.warning(
            f"Mask phase: {n_mask_fetch_failed}/{n_time} scenes failed to fetch "
            "after retries; continuing with the rest"
        )

    if not kept_combo_masks:
        raise RuntimeError(
            "No usable scenes — every scene was fully cloud-masked, invalid, "
            "or failed to fetch"
        )
    return kept_combo_masks


def run_bounds_pipeline(
    request: MosaicRequest,
    *,
    source: Source,
) -> Union[Tuple[npt.NDArray[Any], Dict[str, Any]], Path]:
    """Bounds/AOI-mode pipeline. Called from mosaic() for non-grid AOIs.

    Searches the Planetary Computer STAC for Sentinel-2 L2A scenes intersecting
    ``bounds`` or ``aoi`` over the date window, streams per-scene cloud masks (skipping
    scenes that contribute no new pixels), then fetches user bands only for the
    kept scenes and aggregates them into a single mosaic on a UTM grid.

    Differs from the grid-mode pipeline in that the AOI is an arbitrary bbox
    (possibly spanning multiple MGRS tiles, possibly intersecting tiles in
    different UTM-zone projections), so each per-scene read goes through a
    rasterio WarpedVRT to land on one common grid in ``output_crs``.

    Args:
        bounds: ``(minx, miny, maxx, maxy)`` in ``input_crs`` units. Validated
            for orientation, minimum area, and (for EPSG:4326) lon/lat range;
            very large AOIs emit a warning but are not rejected.
        aoi: Single polygon AOI in ``input_crs`` units. The output raster uses
            the polygon bounds, and pixels outside the polygon are skipped and
            written as nodata.
        input_crs: EPSG code of ``bounds``. Defaults to 4326 (lon/lat).
        start_year, start_month, start_day, duration_years, duration_months,
            duration_days: Date window for the STAC search (start inclusive,
            end exclusive).
        required_bands: Spectral bands to fetch (e.g. ``["B04", "B03", "B02"]``)
            or ``["visual"]`` for the 3-band uint8 TCI asset.
        mosaic_method, percentile: As for :func:`s2mosaic.mosaic`.
        output_dir: Directory to write the GeoTIFF using an auto-generated
            filename. Mutually exclusive with ``output_path``. If neither is
            provided, returns the array + profile instead.
        output_path: Full GeoTIFF path to write, including the filename.
            Mutually exclusive with ``output_dir``.
        overwrite: If ``False`` and the output file already exists, skip the
            mosaic and return the existing path.
        output_crs: EPSG code of the output grid. If ``None``, auto-picked as
            the UTM zone containing the bbox centroid.
        resolution: Output pixel size in metres. Reads come from the nearest
            COG overview for non-native resolutions.
        resampling_method: How source COGs are resampled to ``output_crs``
            / ``resolution`` during the WarpedVRT read.
        additional_query: Extra STAC query filters
            (e.g. ``{"eo:cloud_cover": {"lt": 50}}``).
        no_data_tolerance: Stop early once the uncovered fraction within the
            coverage mask drops below this during scene selection. Set to
            ``0.0`` (default) or ``None`` to examine every scene.
        observation_target: Optional per-tile early-stop target for
            ``mean`` and ``percentile``. When set, user-band aggregation stops
            reading later scenes for a tile once every coverable pixel has at
            least this many valid observations. This is not an output quality
            filter.
        min_coverage_fraction: Drop pixels covered by fewer than this fraction
            of overlapping scenes (scene-edge pixels). ``None`` disables.
        ignore_duplicate_items: Keep only the latest processing baseline per scene.
        sort_method, sort_function: As for :func:`s2mosaic.mosaic`.
        cloud_mask: ``"OCM"`` (deep-learning, default) or ``"SCL"`` (L2A scene
            classification layer — cheaper, less accurate).
        ocm_batch_size, ocm_inference_dtype: Only used when ``cloud_mask="OCM"``.
        tile_workers: Number of output tiles to aggregate concurrently.
            Defaults to ``min(4, os.cpu_count() or 1)``.
        adaptive_tiling: Split sparse output tiles based on the actual
            cloud-valid contribution masks. Defaults to True.
        show_progress: Show tqdm progress bars for the cloud-mask streaming
            and tile-aggregation phases. Defaults to False.

    Returns:
        ``(array, profile)`` if no export path is requested, otherwise the
        ``Path`` of the written GeoTIFF. ``array`` has shape
        ``(bands, height, width)`` and dtype ``uint8`` (visual) or ``uint16``.

    Raises:
        ValueError: If inputs fail validation or no scenes are found.
        RuntimeError: If scenes were found but every scene was fully
            cloud-masked or invalid.
    """
    required_bands = request.required_bands
    additional_query = request.additional_query
    assert required_bands is not None
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
        f"with bands {required_bands}"
    )

    is_visual = "visual" in required_bands

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

    aoi_target = (
        reproject_aoi(aoi, request.input_crs, target_crs) if aoi is not None else None
    )
    bounds_target = (
        aoi_target.bounds
        if aoi_target is not None
        else reproject_bbox(bounds, request.input_crs, target_crs)
    )

    start_date, end_date = define_dates(
        request.start_year,
        request.start_month,
        request.start_day,
        request.duration_years,
        request.duration_months,
        request.duration_days,
    )

    mode = "aoi" if aoi is not None else "bounds"
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
        sort_method=request.sort_method,
        mosaic_method=request.mosaic_method,
        required_bands=required_bands,
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
        sort_method=request.sort_method,
        sort_function=request.sort_function,
    )

    # Mask resolution depends on provider: OCM is fastest at coarser resolutions
    # so we clamp to [20, 50]; SCL is a single COG read, so we fetch at the
    # user's output resolution to avoid a resize step. Either way the streaming
    # loop produces masks at this resolution and the coverage mask is computed
    # at the same shape.
    if request.cloud_mask == CLOUD_MASK_SCL:
        mask_resolution = request.resolution
    else:
        mask_resolution = pick_ocm_resolution(request.resolution)
    logger.info(f"Cloud mask provider {request.cloud_mask} at {mask_resolution}m")

    # Each per-scene fetch via WarpedVRT snaps to exactly this transform /
    # width / height, so all scenes share the same (h, w) without us
    # materialising them. OCM gets an expanded read if needed so the model sees
    # at least 100 x 100 pixels of spatial context; predictions are cropped
    # back to the requested bounds before scene-selection logic uses them.
    mask_w, mask_h = _grid_shape_for_bounds(bounds_target, mask_resolution)
    ocm_bounds_target = bounds_target
    ocm_crop: Optional[Tuple[slice, slice]] = None
    if request.cloud_mask == CLOUD_MASK_OCM:
        ocm_bounds_target, ocm_crop = _expand_bounds_for_ocm_context(
            bounds_target, mask_resolution
        )
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
            items_list[0],
            source,
            bounds_target,
            target_crs,
            mask_resolution,
            mask_w,
            mask_h,
            scl_tile_specs,
        ):
            scl_tile_specs = None

    kept_combo_masks_ocm = _stream_bounds_combo_masks(
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
        no_data_tolerance=request.no_data_tolerance,
        possible_pixel_count=possible_pixel_count,
        tile_workers=request.tile_workers,
        ocm_bounds_target=ocm_bounds_target,
        ocm_crop=ocm_crop,
        ocm_batch_size=request.ocm_batch_size,
        ocm_inference_dtype=request.ocm_inference_dtype,
        scl_tile_specs=scl_tile_specs,
        show_progress=request.show_progress,
    )

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
    n_bands = 3 if is_visual else len(required_bands)

    def _to_user_shape(mask: npt.NDArray[Any]) -> npt.NDArray[Any]:
        if mask.shape == (h, w):
            return mask
        return cv2.resize(
            mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        ).astype(bool)

    coverage_mask = _to_user_shape(coverage_mask_ocm)
    if aoi_target is not None:
        coverage_mask &= _rasterize_aoi_mask(
            aoi_target=aoi_target,
            bounds_target=bounds_target,
            resolution=request.resolution,
            width=w,
            height=h,
        )
    combo_masks_user = {
        i: _to_user_shape(kept_combo_masks_ocm[i]) & coverage_mask for i in kept_indices
    }
    # OCM-resolution mask dict can be released once the user-resolution copies exist.
    del kept_combo_masks_ocm

    # Phase 3 — tile-streamed aggregation. Same architecture as grid_id mode:
    # per-tile workers each read tile windows via the bounds tile reader
    # (WarpedVRT-backed, or direct read from a local cached file), apply
    # the per-tile slice of the precomputed mask, and aggregate by method.
    # Peak RAM is one tile per worker rather than the full N-scene stack.
    href_template, _, _ = get_band_template(required_bands)
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
            request.mosaic_method, request.no_data_tolerance, request.observation_target
        ),
    )

    masks_in_order: List[Optional[npt.NDArray[Any]]] = [
        combo_masks_user[scene_idx] for scene_idx in kept_indices
    ]
    # Tile small AOIs as a single tile; cap large AOIs at 2048 so each
    # tile read from PC stays one big range request. Smaller tiles trade
    # round-trip latency for worker utilisation — a bad deal when reads
    # are network-bound (i.e., production with no debug cache).
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
        "dtype": np.dtype(np.uint8) if is_visual else np.dtype(np.uint16),
        "width": w,
        "height": h,
        "count": n_bands,
        "crs": CRS.from_epsg(target_crs),
        "transform": user_transform,
    }
    output_coverage_mask = (
        coverage_mask if request.min_coverage_fraction is not None else None
    )

    try:
        if export_path is not None:
            result = write_tile_aggregation_geotiff(
                export_path=export_path,
                profile=profile,
                required_bands=required_bands,
                masks=masks_in_order,
                read_fn=read_fn,
                bands_count=n_bands,
                height=h,
                width=w,
                coverage_mask=coverage_mask,
                output_coverage_mask=output_coverage_mask,
                no_data_tolerance=request.no_data_tolerance,
                observation_target=request.observation_target,
                mosaic_method=request.mosaic_method,
                percentile=request.percentile,
                tile_size=tile_size,
                tile_workers=request.tile_workers,
                out_dtype=np.dtype(np.uint8) if is_visual else np.dtype(np.uint16),
                adaptive_tiling=request.adaptive_tiling,
                show_progress=request.show_progress,
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
            no_data_tolerance=request.no_data_tolerance,
            observation_target=request.observation_target,
            mosaic_method=request.mosaic_method,
            percentile=request.percentile,
            tile_size=tile_size,
            tile_workers=request.tile_workers,
            out_dtype=np.dtype(np.uint8) if is_visual else np.dtype(np.uint16),
            adaptive_tiling=request.adaptive_tiling,
            show_progress=request.show_progress,
        )

        return finalize_output(
            array=output_array,
            profile=profile,
            required_bands=required_bands,
            coverage_mask=output_coverage_mask,
            export_path=export_path,
        )
    finally:
        close = getattr(read_fn, "close", None)
        if close is not None:
            close()
