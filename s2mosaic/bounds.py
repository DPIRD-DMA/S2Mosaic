"""Mosaic creation for arbitrary bounding boxes (single or multi-MGRS-tile).

Each scene's bands are fetched on-the-fly through a rasterio WarpedVRT snapped
to a common UTM grid, so scenes from MGRS tiles in different native projections
are all read into the same output frame. The aggregation loop runs per-scene
and accumulates in place (MEAN / FIRST) so peak memory is independent of the
scene count.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import planetary_computer
import pystac_client
import rasterio as rio
import stackstac
from pyproj import Transformer
from pystac.item_collection import ItemCollection
from pystac_client.stac_api_io import StacApiIO
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from urllib3 import Retry

from .frequent_coverage import get_frequent_coverage_for_bbox
from .helpers import (
    CLOUD_MASK_OCM,
    CLOUD_MASK_SCL,
    MOSAIC_FIRST,
    MOSAIC_MEAN,
    MOSAIC_PERCENTILE,
    SceneFetchError,
    define_dates,
    disk_cache,
    finalize_output,
    get_output_path,
    get_rasterio_resampling,
    normalize_mosaic_inputs,
    pick_ocm_resolution,
    validate_inputs,
    with_scene_retry,
)
from .masking import compute_masks_from_array, compute_masks_from_scl
from .mosaic_utils import calculate_percentile_mosaic
from .stac_utils import (
    ITEM_COL,
    add_item_info,
    filter_latest_processing_baselines,
    sort_items,
)

logger = logging.getLogger(__name__)

Bbox = Tuple[float, float, float, float]


def pick_utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the UTM zone containing (lon, lat)."""
    zone = int((lon + 180) / 6) + 1
    return (32700 if lat < 0 else 32600) + zone


def reproject_bbox(bbox: Bbox, src_epsg: int, dst_epsg: int) -> Bbox:
    """Reproject (minx, miny, maxx, maxy) between CRSes."""
    if src_epsg == dst_epsg:
        return bbox
    transformer = Transformer.from_crs(src_epsg, dst_epsg, always_xy=True)
    minx, miny, maxx, maxy = bbox
    xs, ys = transformer.transform([minx, maxx, minx, maxx], [miny, miny, maxy, maxy])
    return (min(xs), min(ys), max(xs), max(ys))


def _search_for_items_by_bbox(
    bbox_4326: Bbox,
    start_date: date,
    end_date: date,
    additional_query: Optional[Dict[str, Any]] = None,
    ignore_duplicate_items: bool = True,
) -> ItemCollection:
    """Search Sentinel-2 L2A items intersecting bbox in EPSG:4326."""
    query: Dict[str, Any] = {
        "collections": ["sentinel-2-l2a"],
        "bbox": list(bbox_4326),
        "datetime": f"{start_date.isoformat()}Z/{end_date.isoformat()}Z",
    }
    if additional_query:
        query["query"] = additional_query

    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[502, 503, 504],
        allowed_methods=None,
    )
    stac_api_io = StacApiIO(max_retries=retry)
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
        stac_io=stac_api_io,
    )
    items = catalog.search(**query).item_collection()
    logger.info(f"Found {len(items)} items for bbox {bbox_4326}")

    if ignore_duplicate_items:
        items = filter_latest_processing_baselines(items)
        logger.info(f"After dedupe, {len(items)} items remain")
    return items


def aggregate_stack(
    user_data: np.ndarray,
    combo_masks: List[np.ndarray],
    mosaic_method: str,
    percentile_value: Optional[float],
) -> np.ndarray:
    """Aggregate a fetch-then-aggregate ``(time, bands, h, w)`` stack into one mosaic.

    Pure reference implementation kept for tests and external callers. The
    bounds-mode pipeline itself runs the same math one scene at a time inside
    ``run_bounds_pipeline`` so peak memory is independent of the scene count.
    """
    n_time, _, h, _ = user_data.shape

    if mosaic_method == MOSAIC_PERCENTILE:
        scenes = [
            np.where(
                combo_masks[i][None, :, :], user_data[i].astype(np.float32), np.nan
            )
            for i in range(n_time)
        ]
        return calculate_percentile_mosaic(
            scenes, s2_scene_size=h, percentile_value=percentile_value or 50.0
        )

    if mosaic_method == MOSAIC_MEAN:
        sum_arr = np.zeros(user_data.shape[1:], dtype=np.float32)
        count = np.zeros(user_data.shape[2:], dtype=np.uint32)
        for i in range(n_time):
            mask = combo_masks[i]
            sum_arr += user_data[i].astype(np.float32) * mask[None, :, :]
            count += mask
        with np.errstate(divide="ignore", invalid="ignore"):
            mosaic = np.where(count > 0, sum_arr / count, 0)
        return mosaic.astype(np.float32)

    if mosaic_method == MOSAIC_FIRST:
        mosaic = np.zeros(user_data.shape[1:], dtype=np.float32)
        filled = np.zeros(user_data.shape[2:], dtype=bool)
        for i in range(n_time):
            mask = combo_masks[i] & ~filled
            mosaic += user_data[i].astype(np.float32) * mask[None, :, :]
            filled |= mask
        return mosaic

    raise ValueError(f"Unsupported mosaic_method: {mosaic_method}")


def _stack_compute_key(
    items_list: list,
    assets: List[str],
    bounds_target: Bbox,
    target_crs: int,
    resolution: int,
    dtype: str,
    resampling: str = "nearest",
) -> str:
    item_ids = ",".join(sorted(item.id for item in items_list))
    return (
        f"{bounds_target}|{target_crs}|{resolution}|"
        f"{','.join(assets)}|{dtype}|{resampling}|{item_ids}"
    )


@disk_cache("stack", key_fn=_stack_compute_key)
def cached_stack_compute(
    items_list: list,
    assets: List[str],
    bounds_target: Bbox,
    target_crs: int,
    resolution: int,
    dtype: str,
    resampling: str = "nearest",
) -> np.ndarray:
    """Materialise a stackstac stack as `dtype`, with optional pickle cache.

    Stackstac requires the fill_value to be outside the data dtype's range
    (no integer fits this for uint16/uint8), so we fetch as float32 with NaN
    fill, then nan_to_num + cast to the requested integer dtype here.
    """
    stack = stackstac.stack(
        items_list,
        assets=assets,
        bounds=bounds_target,
        epsg=target_crs,
        resolution=resolution,
        rescale=False,
        resampling=get_rasterio_resampling(resampling),
        sortby_date=False,  # preserve caller's order (valid_data, etc.)
    )
    arr_float = stack.compute().values
    return np.nan_to_num(arr_float, nan=0).astype(dtype)


_OCM_BANDS: Tuple[str, str, str] = ("B04", "B03", "B8A")


def _target_grid(
    bounds_target: Bbox, resolution: int, target_crs: int
) -> Tuple[Affine, int, int, CRS]:
    """Pixel grid + CRS for ``bounds_target`` at ``resolution`` in ``target_crs``."""
    minx, miny, maxx, maxy = bounds_target
    width = int(round((maxx - minx) / resolution))
    height = int(round((maxy - miny) / resolution))
    transform = Affine(resolution, 0, minx, 0, -resolution, maxy)
    target_crs_obj = CRS.from_epsg(target_crs)
    return transform, width, height, target_crs_obj


def _read_warpvrt(
    href: str,
    indices: Union[int, List[int]],
    transform: Affine,
    width: int,
    height: int,
    target_crs_obj: CRS,
    rio_resampling: Any,
) -> np.ndarray:
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
            return vrt.read(indices)


def _fetch_one_scl_key(
    item: Any,
    bounds_target: Bbox,
    target_crs: int,
    mask_resolution: int,
) -> str:
    return f"{item.id}|{bounds_target}|{target_crs}|{mask_resolution}"


@disk_cache("scl", key_fn=_fetch_one_scl_key)
@with_scene_retry()
def _fetch_one_scl(
    item: Any,
    bounds_target: Bbox,
    target_crs: int,
    mask_resolution: int,
) -> np.ndarray:
    """Fetch one scene's SCL band as (h, w) uint8 at ``mask_resolution``.

    Analog of :func:`_fetch_one_ocm` for the SCL cloud-mask provider — a
    single COG read, no DL inference. SCL is native 20m; the WarpedVRT will
    upsample if ``mask_resolution`` < 20.
    """
    transform, width, height, target_crs_obj = _target_grid(
        bounds_target, mask_resolution, target_crs
    )
    rio_resampling = get_rasterio_resampling("nearest")
    href = planetary_computer.sign(item.assets["SCL"].href)
    arr = _read_warpvrt(
        href, 1, transform, width, height, target_crs_obj, rio_resampling
    )
    return arr.astype(np.uint8)


def _fetch_one_ocm_key(
    item: Any,
    bounds_target: Bbox,
    target_crs: int,
    ocm_resolution: int,
) -> str:
    return f"{item.id}|{bounds_target}|{target_crs}|{ocm_resolution}"


@disk_cache("ocm", key_fn=_fetch_one_ocm_key)
@with_scene_retry()
def _fetch_one_ocm(
    item: Any,
    bounds_target: Bbox,
    target_crs: int,
    ocm_resolution: int,
) -> np.ndarray:
    """Fetch one scene's OCM bands (B04, B03, B8A) as (3, h, w) uint16.

    Uses rasterio + WarpedVRT directly — avoids stackstac's eager-stack
    overhead so the bounds pipeline can stream OCM scenes through the mask
    loop and discard each as it goes (and skip fetching late scenes entirely
    once the no_data threshold or first-mode coverage is met).
    """
    transform, width, height, target_crs_obj = _target_grid(
        bounds_target, ocm_resolution, target_crs
    )
    rio_resampling = get_rasterio_resampling("nearest")

    def _read_band(band_name: str) -> np.ndarray:
        href = planetary_computer.sign(item.assets[band_name].href)
        return _read_warpvrt(
            href, 1, transform, width, height, target_crs_obj, rio_resampling
        )

    with ThreadPoolExecutor(max_workers=len(_OCM_BANDS)) as executor:
        bands = list(executor.map(_read_band, _OCM_BANDS))
    return np.stack(bands, axis=0).astype(np.uint16)


def _fetch_one_user_scene_key(
    item: Any,
    assets: List[str],
    bounds_target: Bbox,
    target_crs: int,
    resolution: int,
    dtype: str,
    resampling: str,
) -> str:
    return (
        f"{item.id}|{bounds_target}|{target_crs}|{resolution}|"
        f"{','.join(assets)}|{dtype}|{resampling}"
    )


@disk_cache("user_scene", key_fn=_fetch_one_user_scene_key)
@with_scene_retry()
def _fetch_one_user_scene(
    item: Any,
    assets: List[str],
    bounds_target: Bbox,
    target_crs: int,
    resolution: int,
    dtype: str,
    resampling: str,
) -> np.ndarray:
    """Fetch one scene's spectral bands as ``(n_bands, h, w)`` of ``dtype``.

    Mirrors ``_fetch_one_ocm`` but for an arbitrary band list. Used by the
    bounds-mode streaming aggregation loop so that only one scene's worth of
    band data is in memory at a time.
    """
    transform, width, height, target_crs_obj = _target_grid(
        bounds_target, resolution, target_crs
    )
    rio_resampling = get_rasterio_resampling(resampling)

    def _read_band(asset_name: str) -> np.ndarray:
        href = planetary_computer.sign(item.assets[asset_name].href)
        return _read_warpvrt(
            href, 1, transform, width, height, target_crs_obj, rio_resampling
        )

    with ThreadPoolExecutor(max_workers=max(1, len(assets))) as executor:
        bands = list(executor.map(_read_band, assets))
    return np.stack(bands, axis=0).astype(dtype)


def _fetch_one_tci_key(
    item: Any,
    bounds_target: Bbox,
    target_crs: int,
    resolution: int,
    resampling: str,
) -> str:
    return f"{item.id}|{bounds_target}|{target_crs}|{resolution}|{resampling}|visual"


@disk_cache("tci_one", key_fn=_fetch_one_tci_key)
@with_scene_retry()
def _fetch_one_tci(
    item: Any,
    bounds_target: Bbox,
    target_crs: int,
    resolution: int,
    resampling: str,
) -> np.ndarray:
    """Fetch one scene's TCI as ``(3, h, w)`` uint8. Per-scene variant of
    ``_fetch_tci_stack`` for the streaming loop."""
    transform, width, height, target_crs_obj = _target_grid(
        bounds_target, resolution, target_crs
    )
    rio_resampling = get_rasterio_resampling(resampling)
    href = planetary_computer.sign(item.assets["visual"].href)
    arr = _read_warpvrt(
        href, [1, 2, 3], transform, width, height, target_crs_obj, rio_resampling
    )
    return arr.astype(np.uint8)


def _fetch_tci_stack_key(
    items_list: list,
    bounds_target: Bbox,
    target_crs: int,
    resolution: int,
    resampling: str,
    max_workers: int = 4,
) -> str:
    item_ids = ",".join(sorted(item.id for item in items_list))
    return f"{bounds_target}|{target_crs}|{resolution}|{resampling}|{item_ids}"


@disk_cache("tci", key_fn=_fetch_tci_stack_key)
def _fetch_tci_stack(
    items_list: list,
    bounds_target: Bbox,
    target_crs: int,
    resolution: int,
    resampling: str,
    max_workers: int = 4,
) -> np.ndarray:
    """Fetch TCI (3-band uint8 RGB) for each item, reprojected to target grid.

    TCI is a single multi-band asset; stackstac's asset->band mapping assumes
    1 band per asset, so we read each scene's TCI directly via WarpedVRT and
    stack the results — mirroring the multi-band-index pattern used in
    grid_id mode (data_reader.get_full_band).
    """
    transform, width, height, target_crs_obj = _target_grid(
        bounds_target, resolution, target_crs
    )
    rio_resampling = get_rasterio_resampling(resampling)

    def _fetch_one(item: Any) -> np.ndarray:
        href = planetary_computer.sign(item.assets["visual"].href)
        return _read_warpvrt(
            href, [1, 2, 3], transform, width, height, target_crs_obj, rio_resampling
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        arrays = list(executor.map(_fetch_one, items_list))
    return np.stack(arrays, axis=0).astype(np.uint8)


def run_bounds_pipeline(
    bounds: Bbox,
    start_year: int,
    bounds_crs: int = 4326,
    target_crs: Optional[int] = None,
    resolution: int = 10,
    start_month: int = 1,
    start_day: int = 1,
    output_dir: Optional[Union[Path, str]] = None,
    sort_method: str = "valid_data",
    sort_function: Optional[Callable] = None,
    mosaic_method: str = "mean",
    duration_years: int = 0,
    duration_months: int = 0,
    duration_days: int = 0,
    required_bands: Optional[List[str]] = None,
    no_data_threshold: Optional[float] = 0.01,
    overwrite: bool = True,
    ocm_batch_size: int = 1,
    ocm_inference_dtype: str = "bf16",
    additional_query: Optional[Dict[str, Any]] = None,
    percentile_value: Optional[float] = None,
    ignore_duplicate_items: bool = True,
    coverage_threshold_pct: Optional[float] = 0.1,
    resampling_method: str = "nearest",
    cloud_mask: str = CLOUD_MASK_OCM,
) -> Union[Tuple[np.ndarray, Dict[str, Any]], Path]:
    """Bounds-mode pipeline. Called from mosaic() when bounds is set.

    Searches the Planetary Computer STAC for Sentinel-2 L2A scenes intersecting
    ``bounds`` over the date window, streams per-scene cloud masks (skipping
    scenes that contribute no new pixels), then fetches user bands only for the
    kept scenes and aggregates them into a single mosaic on a UTM grid.

    Differs from the grid-mode pipeline in that the AOI is an arbitrary bbox
    (possibly spanning multiple MGRS tiles, possibly intersecting tiles in
    different UTM-zone projections), so all reads go through stackstac /
    WarpedVRT to land on one common grid in ``target_crs``.

    Args:
        bounds: ``(minx, miny, maxx, maxy)`` in ``bounds_crs`` units. Validated
            for orientation, size (10m–200km per side), and (for EPSG:4326) lon/lat
            range.
        start_year, start_month, start_day, duration_years, duration_months,
            duration_days: Date window for the STAC search (start inclusive,
            end exclusive).
        bounds_crs: EPSG code of ``bounds``. Defaults to 4326 (lon/lat).
        target_crs: EPSG code of the output grid. If ``None``, auto-picked as
            the UTM zone containing the bbox centroid.
        resolution: Output pixel size in metres. Stackstac will read from the
            nearest COG overview for non-native resolutions.
        sort_method, sort_function, mosaic_method, percentile_value: As for
            :func:`s2mosaic.mosaic`.
        required_bands: Spectral bands to fetch (e.g. ``["B04", "B03", "B02"]``)
            or ``["visual"]`` for the 3-band uint8 TCI asset.
        no_data_threshold: Stop the mask-streaming loop once the fraction of
            uncovered pixels (within the coverage mask) drops below this.
            Ignored for ``mosaic_method="percentile"``.
        coverage_threshold_pct: Drop pixels covered by fewer than this fraction
            of overlapping scenes (scene-edge pixels). ``None`` disables.
        ocm_batch_size, ocm_inference_dtype: Only used when ``cloud_mask="OCM"``.
        additional_query: Extra STAC query filters
            (e.g. ``{"eo:cloud_cover": {"lt": 50}}``).
        ignore_duplicate_items: Keep only the latest processing baseline per scene.
        resampling_method: How stackstac resamples source COGs to ``target_crs``
            / ``resolution``.
        cloud_mask: ``"OCM"`` (deep-learning, default) or ``"SCL"`` (L2A scene
            classification layer — cheaper, less accurate).
        output_dir: Directory to write the GeoTIFF. If ``None``, returns the
            array + profile instead.
        overwrite: If ``False`` and the output file already exists, skip the
            mosaic and return the existing path.

    Returns:
        ``(array, profile)`` if ``output_dir`` is ``None``, otherwise the
        ``Path`` of the written GeoTIFF. ``array`` has shape
        ``(bands, height, width)`` and dtype ``uint8`` (visual) or ``uint16``.

    Raises:
        ValueError: If inputs fail validation or no scenes are found.
        RuntimeError: If scenes were found but every scene was fully
            cloud-masked or invalid.
    """
    (
        required_bands,
        additional_query,
        sort_method,
        mosaic_method,
        percentile_value,
    ) = normalize_mosaic_inputs(
        required_bands=required_bands,
        additional_query=additional_query,
        sort_method=sort_method,
        sort_function=sort_function,
        mosaic_method=mosaic_method,
        percentile_value=percentile_value,
    )

    logger.info(
        f"Creating mosaic for bounds {bounds} (EPSG:{bounds_crs}) "
        f"from {start_year}-{start_month:02d}-{start_day:02d} "
        f"+ {duration_years}y {duration_months}m {duration_days}d "
        f"using {mosaic_method} method with bands {required_bands}"
    )

    is_visual = "visual" in required_bands

    validate_inputs(
        sort_method=sort_method,
        mosaic_method=mosaic_method,
        no_data_threshold=no_data_threshold,
        required_bands=required_bands,
        grid_id=None,
        percentile_value=percentile_value,
        resampling_method=resampling_method,
        bounds=bounds,
        bounds_crs=bounds_crs,
        resolution=resolution,
        cloud_mask=cloud_mask,
    )

    bounds_4326 = reproject_bbox(bounds, bounds_crs, 4326)
    if target_crs is None:
        cx = (bounds_4326[0] + bounds_4326[2]) / 2
        cy = (bounds_4326[1] + bounds_4326[3]) / 2
        target_crs = pick_utm_epsg(cx, cy)
        logger.info(f"Auto-picked target CRS: EPSG:{target_crs}")

    bounds_target = reproject_bbox(bounds, bounds_crs, target_crs)

    start_date, end_date = define_dates(
        start_year,
        start_month,
        start_day,
        duration_years,
        duration_months,
        duration_days,
    )

    export_path: Optional[Path] = None
    if output_dir:
        export_path = get_output_path(
            output_dir=output_dir,
            start_date=start_date,
            end_date=end_date,
            sort_method=sort_method,
            mosaic_method=mosaic_method,
            required_bands=required_bands,
            bounds=bounds_4326,
        )
        if export_path.exists() and not overwrite:
            return export_path

    items = _search_for_items_by_bbox(
        bbox_4326=bounds_4326,
        start_date=start_date,
        end_date=end_date,
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
    items_list = sorted_items[ITEM_COL].tolist()

    # Mask resolution depends on provider: OCM is fastest at coarser resolutions
    # so we clamp to [20, 50]; SCL is a single COG read, so we fetch at the
    # user's output resolution to avoid a resize step. Either way the streaming
    # loop produces masks at this resolution and the coverage mask is computed
    # at the same shape.
    if cloud_mask == CLOUD_MASK_SCL:
        mask_resolution = resolution
    else:
        mask_resolution = pick_ocm_resolution(resolution)
    logger.info(f"Cloud mask provider {cloud_mask} at {mask_resolution}m")

    # Each per-scene fetch via WarpedVRT snaps to exactly this transform /
    # width / height, so all scenes share the same (h, w) without us
    # materialising them.
    mask_minx, mask_miny, mask_maxx, mask_maxy = bounds_target
    mask_w = int(round((mask_maxx - mask_minx) / mask_resolution))
    mask_h = int(round((mask_maxy - mask_miny) / mask_resolution))
    n_time = len(items_list)

    # Coverage mask at mask resolution — used for skip decisions.
    if coverage_threshold_pct is not None:
        coverage_mask_ocm = get_frequent_coverage_for_bbox(
            scenes=items,
            bounds_target=bounds_target,
            target_crs=target_crs,
            width=mask_w,
            height=mask_h,
            resolution=mask_resolution,
            coverage_threshold_pct=coverage_threshold_pct,
        )
    else:
        coverage_mask_ocm = np.ones((mask_h, mask_w), dtype=bool)
    possible_pixel_count = int(coverage_mask_ocm.sum())

    # Phase 1+2: stream cloud-mask fetch + classification. One scene's mask
    # input is in memory at a time, and late scenes are skipped entirely once
    # first mode's coverage is filled or no_data_threshold is met.
    logger.info(
        f"Streaming cloud mask over up to {n_time} scenes "
        f"(per-scene fetch at {mask_resolution}m, EPSG:{target_crs})"
    )
    kept_combo_masks_ocm: Dict[int, np.ndarray] = {}
    good_pixel_tracker = np.zeros((mask_h, mask_w), dtype=bool)
    n_mask_fetch_failed = 0
    for i in range(n_time):
        # FIRST mode: stop scanning once everything in coverage is filled.
        if (
            mosaic_method == MOSAIC_FIRST
            and (good_pixel_tracker | ~coverage_mask_ocm).all()
        ):
            logger.info(
                f"All in-coverage pixels filled after {i}/{n_time} scenes — "
                "skipping remaining cloud-mask fetches"
            )
            break

        try:
            if cloud_mask == CLOUD_MASK_SCL:
                scl_scene = _fetch_one_scl(
                    items_list[i], bounds_target, target_crs, mask_resolution
                )
            else:
                ocm_scene = _fetch_one_ocm(
                    items_list[i],
                    bounds_target,
                    target_crs,
                    mask_resolution,
                )
        except SceneFetchError as e:
            n_mask_fetch_failed += 1
            logger.warning(
                f"Scene {i + 1}/{n_time} ({items_list[i].id}): mask fetch failed, "
                f"skipping ({e})"
            )
            continue

        if cloud_mask == CLOUD_MASK_SCL:
            clear, valid = compute_masks_from_scl(scl_scene)
            del scl_scene
        else:
            clear, valid = compute_masks_from_array(
                ocm_scene,
                batch_size=ocm_batch_size,
                inference_dtype=ocm_inference_dtype,
            )
            del ocm_scene
        combo = clear & valid

        if mosaic_method == MOSAIC_FIRST:
            new_pixels = combo & ~good_pixel_tracker
            if not new_pixels.any():
                continue
            combo = new_pixels
        elif not combo.any():
            # All-cloud scene — no contribution to mean/percentile either.
            continue

        kept_combo_masks_ocm[i] = combo
        good_pixel_tracker |= combo

        if (
            no_data_threshold is not None
            and mosaic_method != MOSAIC_PERCENTILE
            and possible_pixel_count > 0
        ):
            completed = int((coverage_mask_ocm & good_pixel_tracker).sum())
            no_data_sum = possible_pixel_count - completed
            no_data_pct = (1 - completed / possible_pixel_count) * 100
            logger.info(f"Scene {i + 1}/{n_time} kept; no-data {no_data_pct:.1f}%")
            if no_data_sum < possible_pixel_count * no_data_threshold:
                logger.info(
                    f"no_data_threshold met after {len(kept_combo_masks_ocm)} kept "
                    f"scenes ({i + 1}/{n_time} examined)"
                )
                break

    if n_mask_fetch_failed:
        logger.warning(
            f"Mask phase: {n_mask_fetch_failed}/{n_time} scenes failed to fetch "
            "after retries; continuing with the rest"
        )

    if not kept_combo_masks_ocm:
        raise RuntimeError(
            "No usable scenes — every scene was fully cloud-masked, invalid, "
            "or failed to fetch"
        )

    # Phase 3: stream per-scene user-band fetches and accumulate into the
    # mosaic in place. Mirrors grid mode's ``download_bands_pool`` loop so peak
    # memory is O(one scene + accumulator) rather than O(N scenes) for MEAN
    # and FIRST. PERCENTILE still buffers N scenes because exact percentiles
    # need all values per pixel — but we avoid stackstac's transient float32
    # doubling.
    kept_indices = sorted(kept_combo_masks_ocm.keys())
    kept_items = [items_list[i] for i in kept_indices]
    logger.info(f"Streaming user bands for {len(kept_items)}/{n_time} kept scenes")

    # User grid is fixed upfront from bounds_target + resolution — every
    # per-scene fetch snaps to exactly this (transform, w, h), so accumulators
    # can be sized before any data is fetched.
    user_transform, w, h, _ = _target_grid(bounds_target, resolution, target_crs)
    n_bands = 3 if is_visual else len(required_bands)

    def _to_user_shape(mask: np.ndarray) -> np.ndarray:
        if mask.shape == (h, w):
            return mask
        return cv2.resize(
            mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        ).astype(bool)

    combo_masks_user = {
        i: _to_user_shape(kept_combo_masks_ocm[i]) for i in kept_indices
    }
    coverage_mask = _to_user_shape(coverage_mask_ocm)
    # OCM-resolution mask dict can be released once the user-resolution copies exist.
    del kept_combo_masks_ocm

    scene_dtype = "uint8" if is_visual else "uint16"

    # Initialise accumulators sized to the user grid.
    sum_arr: Optional[np.ndarray] = None
    count: Optional[np.ndarray] = None
    first_mosaic: Optional[np.ndarray] = None
    filled: Optional[np.ndarray] = None
    percentile_scenes: List[np.ndarray] = []

    if mosaic_method == MOSAIC_MEAN:
        sum_arr = np.zeros((n_bands, h, w), dtype=np.float32)
        count = np.zeros((h, w), dtype=np.uint32)
    elif mosaic_method == MOSAIC_FIRST:
        first_mosaic = np.zeros((n_bands, h, w), dtype=np.float32)
        filled = np.zeros((h, w), dtype=bool)

    n_user_fetch_failed = 0
    n_user_succeeded = 0
    for loop_idx, scene_idx in enumerate(kept_indices):
        item = items_list[scene_idx]
        mask = combo_masks_user[scene_idx]

        try:
            if is_visual:
                scene = _fetch_one_tci(
                    item, bounds_target, target_crs, resolution, resampling_method
                )
            else:
                scene = _fetch_one_user_scene(
                    item,
                    list(required_bands),
                    bounds_target,
                    target_crs,
                    resolution,
                    scene_dtype,
                    resampling_method,
                )
        except SceneFetchError as e:
            n_user_fetch_failed += 1
            logger.warning(
                f"Scene {loop_idx + 1}/{len(kept_indices)} ({item.id}): user-band "
                f"fetch failed, skipping ({e})"
            )
            continue

        # Scene fetcher returns its native (h, w); resize to user grid only if
        # WarpedVRT snapped differently (shouldn't, but guard against drift).
        if scene.shape[1:] != (h, w):
            scene = np.stack(
                [
                    cv2.resize(scene[b], (w, h), interpolation=cv2.INTER_NEAREST)
                    for b in range(scene.shape[0])
                ],
                axis=0,
            )

        if mosaic_method == MOSAIC_MEAN:
            assert sum_arr is not None and count is not None
            sum_arr += scene.astype(np.float32) * mask[None, :, :]
            count += mask
        elif mosaic_method == MOSAIC_FIRST:
            assert first_mosaic is not None and filled is not None
            new_pixels = mask & ~filled
            if new_pixels.any():
                first_mosaic += scene.astype(np.float32) * new_pixels[None, :, :]
                filled |= new_pixels
        else:  # MOSAIC_PERCENTILE
            percentile_scenes.append(
                np.where(mask[None, :, :], scene.astype(np.float32), np.nan)
            )

        del scene  # free before next iteration
        n_user_succeeded += 1

        logger.info(
            f"Aggregated scene {loop_idx + 1}/{len(kept_indices)} ({mosaic_method})"
        )

    if n_user_fetch_failed:
        logger.warning(
            f"User-band phase: {n_user_fetch_failed}/{len(kept_indices)} kept "
            "scenes failed to fetch after retries"
        )
    if n_user_succeeded == 0:
        raise RuntimeError(
            f"All {len(kept_indices)} kept scenes failed to fetch during the "
            "user-band phase"
        )

    # Finalise.
    mosaic: np.ndarray
    if mosaic_method == MOSAIC_MEAN:
        assert sum_arr is not None and count is not None
        with np.errstate(divide="ignore", invalid="ignore"):
            mosaic = np.where(count > 0, sum_arr / count, 0).astype(np.float32)
    elif mosaic_method == MOSAIC_FIRST:
        assert first_mosaic is not None
        mosaic = first_mosaic
    else:
        mosaic = calculate_percentile_mosaic(
            percentile_scenes,
            s2_scene_size=h,
            percentile_value=percentile_value or 50.0,
        )

    if is_visual:
        output_array = np.clip(np.nan_to_num(mosaic, nan=0), 0, 255).astype(np.uint8)
    else:
        output_array = np.clip(np.nan_to_num(mosaic, nan=0), 0, 65535).astype(np.uint16)

    profile: Dict[str, Any] = {
        "driver": "GTiff",
        "dtype": output_array.dtype,
        "width": w,
        "height": h,
        "count": output_array.shape[0],
        "crs": CRS.from_epsg(target_crs),
        "transform": user_transform,
    }

    return finalize_output(
        array=output_array,
        profile=profile,
        required_bands=required_bands,
        coverage_mask=coverage_mask if coverage_threshold_pct is not None else None,
        export_path=export_path,
    )
