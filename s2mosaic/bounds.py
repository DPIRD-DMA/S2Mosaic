"""Mosaic creation for arbitrary bounding boxes (single or multi-MGRS-tile).

Uses stackstac to fetch and reproject scenes onto a common UTM grid in one
step, then runs the same cloud-mask + aggregation logic as the grid_id path.
"""

import hashlib
import logging
import pickle
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
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from tqdm.auto import tqdm
from urllib3 import Retry

from .frequent_coverage import get_frequent_coverage_for_bbox
from .helpers import (
    MOSAIC_FIRST,
    MOSAIC_MEAN,
    MOSAIC_PERCENTILE,
    define_dates,
    export_tif,
    get_output_path,
    get_rasterio_resampling,
    pick_ocm_resolution,
    progress_disabled,
    validate_inputs,
)
from .masking import compute_masks_from_array
from .mosaic_utils import calculate_percentile_mosaic
from .stac_utils import (
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


def search_for_items_by_bbox(
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


def _aggregate(
    user_data: np.ndarray,
    combo_masks: List[np.ndarray],
    mosaic_method: str,
    percentile_value: Optional[float],
) -> np.ndarray:
    """Aggregate (time, bands, h, w) stack into (bands, h, w) using combo_masks."""
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


def _cached_stack_compute(
    items_list: list,
    assets: List[str],
    bounds_target: Bbox,
    target_crs: int,
    resolution: int,
    dtype: str,
    debug_cache: bool,
    resampling: str = "nearest",
) -> np.ndarray:
    """Materialise a stackstac stack as `dtype`, with optional pickle cache.

    Stackstac requires the fill_value to be outside the data dtype's range
    (no integer fits this for uint16/uint8), so we fetch as float32 with NaN
    fill, then nan_to_num + cast to the requested integer dtype here.
    """
    cache_path: Optional[Path] = None
    if debug_cache:
        item_ids = ",".join(sorted(item.id for item in items_list))
        key = (
            f"{bounds_target}|{target_crs}|{resolution}|"
            f"{','.join(assets)}|{dtype}|{resampling}|{item_ids}"
        )
        digest = hashlib.md5(key.encode()).hexdigest()
        cache_path = Path("cache") / f"stack_{digest}.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                return pickle.load(f)

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
    arr = np.nan_to_num(arr_float, nan=0).astype(dtype)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(arr, f)
    return arr


def _fetch_tci_stack(
    items_list: list,
    bounds_target: Bbox,
    target_crs: int,
    resolution: int,
    resampling: str,
    debug_cache: bool,
    max_workers: int = 4,
) -> np.ndarray:
    """Fetch TCI (3-band uint8 RGB) for each item, reprojected to target grid.

    TCI is a single multi-band asset; stackstac's asset->band mapping assumes
    1 band per asset, so we read each scene's TCI directly via WarpedVRT and
    stack the results — mirroring the multi-band-index pattern used in
    grid_id mode (data_reader.get_full_band).
    """
    minx, miny, maxx, maxy = bounds_target
    width = int(round((maxx - minx) / resolution))
    height = int(round((maxy - miny) / resolution))
    transform = Affine(resolution, 0, minx, 0, -resolution, maxy)
    target_crs_obj = rio.crs.CRS.from_epsg(target_crs)
    rio_resampling = get_rasterio_resampling(resampling)

    cache_path: Optional[Path] = None
    if debug_cache:
        item_ids = ",".join(sorted(item.id for item in items_list))
        key = f"tci|{bounds_target}|{target_crs}|{resolution}|{resampling}|{item_ids}"
        digest = hashlib.md5(key.encode()).hexdigest()
        cache_path = Path("cache") / f"tci_{digest}.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as cache_f:
                return pickle.load(cache_f)

    def _fetch_one(item: Any) -> np.ndarray:
        href = planetary_computer.sign(item.assets["visual"].href)
        with rio.open(href) as src:
            with WarpedVRT(
                src,
                crs=target_crs_obj,
                transform=transform,
                width=width,
                height=height,
                resampling=rio_resampling,
            ) as vrt:
                return vrt.read([1, 2, 3])

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        arrays = list(executor.map(_fetch_one, items_list))
    arr = np.stack(arrays, axis=0).astype(np.uint8)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(arr, f)
    return arr


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
    debug_cache: bool = False,
    additional_query: Optional[Dict[str, Any]] = None,
    percentile_value: Optional[float] = None,
    ignore_duplicate_items: bool = True,
    coverage_threshold_pct: Optional[float] = 0.1,
    resampling_method: str = "nearest",
) -> Union[Tuple[np.ndarray, Dict[str, Any]], Path]:
    """Bounds-mode pipeline. Called from mosaic() when bounds is set."""
    if required_bands is None:
        required_bands = ["B04", "B03", "B02", "B08"]
    if additional_query is None:
        additional_query = {"eo:cloud_cover": {"lt": 100}}
    if sort_function:
        sort_method = "custom"

    logger.info(
        f"Creating mosaic for bounds {bounds} (EPSG:{bounds_crs}) "
        f"from {start_year}-{start_month:02d}-{start_day:02d} "
        f"+ {duration_years}y {duration_months}m {duration_days}d "
        f"using {mosaic_method} method with bands {required_bands}"
    )

    if mosaic_method == "median":
        if percentile_value is not None:
            raise ValueError(
                "percentile_value should not be set when using mosaic_method='median'."
            )
        mosaic_method = MOSAIC_PERCENTILE
        percentile_value = 50.0

    is_visual = "visual" in required_bands

    if len(bounds) != 4:
        raise ValueError("bounds must be (minx, miny, maxx, maxy)")
    minx, miny, maxx, maxy = bounds
    if minx >= maxx or miny >= maxy:
        raise ValueError(f"Invalid bounds: {bounds}")
    if resolution <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}")

    validate_inputs(
        sort_method=sort_method,
        mosaic_method=mosaic_method,
        no_data_threshold=no_data_threshold,
        required_bands=required_bands,
        grid_id=None,
        percentile_value=percentile_value,
        resampling_method=resampling_method,
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

    items = search_for_items_by_bbox(
        bbox_4326=bounds_4326,
        start_date=start_date,
        end_date=end_date,
        additional_query=additional_query,
        ignore_duplicate_items=ignore_duplicate_items,
    )
    if len(items) == 0:
        raise Exception(
            f"No scenes found for bounds {bounds} between "
            f"{start_date.isoformat()} and {end_date.isoformat()}"
        )

    items_with_orbits = add_item_info(items)
    if sort_function:
        sorted_items = sort_function(items=items_with_orbits)
    else:
        sorted_items = sort_items(items=items_with_orbits, sort_method=sort_method)
    items_list = sorted_items["item"].tolist()

    ocm_assets = ["B04", "B03", "B8A"]
    ocm_resolution = pick_ocm_resolution(resolution)
    logger.info(f"OCM resolution: {ocm_resolution}m")

    # Phase 1: fetch OCM bands for ALL scenes (small, 3 bands at 20-50m).
    # We do all skip-decision math at OCM resolution, then resize masks to
    # user_data's actual shape once we've fetched it. This avoids relying on
    # an upfront probe (different fetchers — stackstac vs the TCI WarpedVRT
    # path — can snap bounds to slightly different output shapes).
    n_time = len(items_list)
    logger.info(
        f"Fetching {n_time} scenes' OCM bands at {ocm_resolution}m (EPSG:{target_crs})"
    )
    ocm_data = _cached_stack_compute(
        items_list,
        ocm_assets,
        bounds_target,
        target_crs,
        ocm_resolution,
        "uint16",
        debug_cache,
        resampling="nearest",
    )
    _, _, ocm_h, ocm_w = ocm_data.shape

    # Coverage mask at OCM resolution — used for skip decisions.
    if coverage_threshold_pct is not None:
        coverage_mask_ocm = get_frequent_coverage_for_bbox(
            scenes=items,
            bounds_target=bounds_target,
            target_crs=target_crs,
            width=ocm_w,
            height=ocm_h,
            resolution=ocm_resolution,
            coverage_threshold_pct=coverage_threshold_pct,
        )
    else:
        coverage_mask_ocm = np.ones((ocm_h, ocm_w), dtype=bool)
    possible_pixel_count = int(coverage_mask_ocm.sum())

    # Phase 2: per-scene cloud masking with skip logic so we can avoid
    # fetching user bands for scenes that won't contribute to the mosaic.
    logger.info(f"Running cloud mask on up to {n_time} scenes")
    kept_combo_masks_ocm: Dict[int, np.ndarray] = {}
    good_pixel_tracker = np.zeros((ocm_h, ocm_w), dtype=bool)
    pbar = tqdm(
        total=n_time,
        desc="Cloud masking",
        leave=False,
        disable=progress_disabled(),
    )
    for i in range(n_time):
        # FIRST mode: stop scanning once everything in coverage is filled.
        if (
            mosaic_method == MOSAIC_FIRST
            and (good_pixel_tracker | ~coverage_mask_ocm).all()
        ):
            logger.info(
                f"All in-coverage pixels filled after {i}/{n_time} scenes — "
                "skipping remaining OCM"
            )
            break

        clear, valid = compute_masks_from_array(
            ocm_data[i],
            batch_size=ocm_batch_size,
            inference_dtype=ocm_inference_dtype,
        )
        # compute_masks_from_array returns at OCM resolution; no resize here.
        combo = clear & valid

        if mosaic_method == MOSAIC_FIRST:
            new_pixels = combo & ~good_pixel_tracker
            if not new_pixels.any():
                pbar.update(1)
                continue
            combo = new_pixels
        elif not combo.any():
            # All-cloud scene — no contribution to mean/percentile either.
            pbar.update(1)
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
            pbar.set_postfix_str(f"no-data {no_data_pct:.1f}%")
            if no_data_sum < possible_pixel_count * no_data_threshold:
                pbar.update(1)
                logger.info(
                    f"no_data_threshold met after {len(kept_combo_masks_ocm)} kept "
                    f"scenes ({i + 1}/{n_time} examined)"
                )
                break
        pbar.update(1)
    pbar.close()

    if not kept_combo_masks_ocm:
        raise Exception(
            "No usable scenes — every scene was fully cloud-masked or invalid"
        )

    # Phase 3: fetch user bands ONLY for the scenes we kept.
    kept_indices = sorted(kept_combo_masks_ocm.keys())
    kept_items = [items_list[i] for i in kept_indices]
    logger.info(f"Fetching user bands for {len(kept_items)}/{n_time} kept scenes")
    if is_visual:
        user_data = _fetch_tci_stack(
            kept_items,
            bounds_target,
            target_crs,
            resolution,
            resampling=resampling_method,
            debug_cache=debug_cache,
        )
    else:
        user_data = _cached_stack_compute(
            kept_items,
            list(required_bands),
            bounds_target,
            target_crs,
            resolution,
            "uint16",
            debug_cache,
            resampling=resampling_method,
        )

    # Resize OCM-resolution masks to user_data's actual (h, w). Whatever shape
    # the fetcher returned is the truth — stackstac and the TCI WarpedVRT path
    # snap bounds independently, so we don't assume they agree.
    _, _, h, w = user_data.shape

    def _to_user_shape(mask: np.ndarray) -> np.ndarray:
        if mask.shape == (h, w):
            return mask
        return cv2.resize(
            mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        ).astype(bool)

    combo_masks = [_to_user_shape(kept_combo_masks_ocm[i]) for i in kept_indices]
    coverage_mask = _to_user_shape(coverage_mask_ocm)

    logger.info(
        f"Aggregating {len(kept_items)} scene(s) into mosaic using "
        f"{mosaic_method} method"
    )
    mosaic = _aggregate(user_data, combo_masks, mosaic_method, percentile_value)

    if coverage_threshold_pct is not None:
        mosaic = np.where(coverage_mask[None, :, :], mosaic, 0)

    if is_visual:
        output_array = np.clip(np.nan_to_num(mosaic, nan=0), 0, 255).astype(np.uint8)
        band_descriptions = ["Red", "Green", "Blue"]
        nodata_value: Optional[int] = None
    else:
        output_array = np.clip(np.nan_to_num(mosaic, nan=0), 0, 65535).astype(np.uint16)
        band_descriptions = list(required_bands)
        nodata_value = 0

    transform = Affine(
        resolution, 0, bounds_target[0], 0, -resolution, bounds_target[3]
    )
    profile: Dict[str, Any] = {
        "driver": "GTiff",
        "dtype": output_array.dtype,
        "width": w,
        "height": h,
        "count": output_array.shape[0],
        "crs": rio.crs.CRS.from_epsg(target_crs),
        "transform": transform,
    }

    if export_path is not None:
        logger.info(f"Writing GeoTIFF to {export_path}")
        export_tif(
            array=output_array,
            profile=profile,
            export_path=export_path,
            required_bands=band_descriptions,
            nodata_value=nodata_value,
        )
        return export_path
    logger.info(
        f"Returning array shape={output_array.shape} dtype={output_array.dtype}"
    )
    return output_array, profile
