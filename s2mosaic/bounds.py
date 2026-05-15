"""Mosaic creation for arbitrary bounding boxes (single or multi-MGRS-tile).

Each scene's bands are fetched on-the-fly through a rasterio WarpedVRT snapped
to a common UTM grid, so scenes from MGRS tiles in different native projections
are all read into the same output frame. Bounds mode computes per-scene masks
on the target grid, keeps only scenes that can contribute pixels, then uses the
shared tile-streamed aggregation path from ``mosaic_core``.
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt
import planetary_computer
import pystac_client
import rasterio as rio
from pyproj import Transformer
from pystac.item_collection import ItemCollection
from pystac_client.stac_api_io import StacApiIO
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from urllib3 import Retry

from .frequent_coverage import get_frequent_coverage_for_bbox
from .helpers import (
    CLOUD_MASK_OCM,
    CLOUD_MASK_SCL,
    MOSAIC_FIRST,
    MOSAIC_PERCENTILE,
    SceneFetchError,
    define_dates,
    disk_cache,
    finalize_output,
    get_band_template,
    get_output_path,
    get_rasterio_resampling,
    normalize_mosaic_inputs,
    pick_ocm_resolution,
    validate_inputs,
    with_scene_retry,
)
from .masking import compute_masks_from_array, compute_masks_from_scl
from .mosaic_core import (
    _prewarm_sources,
    _write_tiled_copy,
    materialise_tiled_band,
    run_tile_aggregation,
    should_prewarm_sources,
)
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


_OCM_BANDS: Tuple[str, str, str] = ("B04", "B03", "B8A")
_OCM_MIN_CONTEXT_PIXELS = 100


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


def _grid_shape_for_bounds(bounds_target: Bbox, resolution: int) -> Tuple[int, int]:
    """Return (width, height), keeping tiny valid bounds at least one pixel."""
    minx, miny, maxx, maxy = bounds_target
    width = max(1, int(round((maxx - minx) / resolution)))
    height = max(1, int(round((maxy - miny) / resolution)))
    return width, height


def _expand_bounds_for_ocm_context(
    bounds_target: Bbox, resolution: int, min_pixels: int = _OCM_MIN_CONTEXT_PIXELS
) -> Tuple[Bbox, Tuple[slice, slice]]:
    """Pad OCM reads to at least ``min_pixels`` each way and return AOI crop.

    The expanded bounds stay aligned to the requested mask grid. The returned
    slices crop an expanded OCM prediction back to the originally requested
    bounds at ``resolution``.
    """
    req_w, req_h = _grid_shape_for_bounds(bounds_target, resolution)
    expanded_w = max(req_w, min_pixels)
    expanded_h = max(req_h, min_pixels)

    pad_x = expanded_w - req_w
    pad_y = expanded_h - req_h
    left = pad_x // 2
    right = pad_x - left
    top = pad_y // 2
    bottom = pad_y - top

    minx, miny, maxx, maxy = bounds_target
    expanded_bounds = (
        minx - left * resolution,
        miny - bottom * resolution,
        maxx + right * resolution,
        maxy + top * resolution,
    )
    return expanded_bounds, (slice(top, top + req_h), slice(left, left + req_w))


def _materialise_bounds_band(
    item: Any,
    asset_name: str,
    bounds_target: Bbox,
    target_crs: int,
    width: int,
    height: int,
    resolution: int,
    rio_resampling: Any,
) -> Callable[[Path], None]:
    """Build a materialiser for a bounds-mode band cache entry.

    Closes over the item, asset, target grid, and resampling. The returned
    function opens the COG via WarpedVRT (which handles reprojection to the
    user's target grid) and streams destination blocks to a tiled GeoTIFF.
    Subsequent reads of any tile window can open the local file directly —
    no WarpedVRT needed once materialised.
    """
    target_crs_obj = CRS.from_epsg(target_crs)
    minx, _, _, maxy = bounds_target
    transform = Affine(resolution, 0, minx, 0, -resolution, maxy)

    def write(tmp_path: Path) -> None:
        signed = planetary_computer.sign(item.assets[asset_name].href)
        with rio.open(signed) as src:
            with WarpedVRT(
                src,
                crs=target_crs_obj,
                transform=transform,
                width=width,
                height=height,
                resampling=rio_resampling,
            ) as vrt:
                n_bands = vrt.count
                dtype = vrt.dtypes[0]
                profile = {
                    "driver": "GTiff",
                    "count": n_bands,
                    "dtype": dtype,
                    "width": width,
                    "height": height,
                    "crs": target_crs_obj,
                    "transform": transform,
                    "tiled": True,
                    "blockxsize": 512,
                    "blockysize": 512,
                    "compress": "lzw",
                    "BIGTIFF": "IF_SAFER",
                }
                _write_tiled_copy(vrt, tmp_path, profile, rio_resampling, lambda w: w)

    return write


def make_bounds_tile_reader(
    items: List[Any],
    href_template: List[Tuple[str, int]],
    bounds_target: Bbox,
    target_crs: int,
    user_transform: Affine,
    width: int,
    height: int,
    resolution: int,
    resampling_method: str,
    prewarm: bool = True,
) -> Callable[[int, int, Tuple[int, int, int, int]], "npt.NDArray[Any]"]:
    """Build a tile-reader for bounds mode (WarpedVRT-backed reads).

    Builds lazy source resolvers for every (scene, asset). Workers open
    handles lazily per thread; if ``S2MOSAIC_DEBUG_CACHE`` is on, the local
    cache file is materialised only when a tile actually reads that source.
    PC URLs get wrapped in a ``WarpedVRT`` to reproject on read, while cached
    local files are opened directly (they're already on the target grid).

    With ``prewarm=True`` (default) the resolvers are called in parallel once
    before returning. Pass ``prewarm=False`` to keep them strictly lazy.
    """
    target_crs_obj = CRS.from_epsg(target_crs)
    rio_resampling = get_rasterio_resampling(resampling_method)
    href_band_indices = [band_idx for _, band_idx in href_template]

    sources: List[List[Callable[[], Tuple[str, bool]]]] = []
    for item in items:
        scene_sources: List[Callable[[], Tuple[str, bool]]] = []
        for asset, _ in href_template:
            cache_key = (
                f"bounds|{item.id}|{asset}|{bounds_target}|{target_crs}|"
                f"{width}|{height}|{resolution}|{resampling_method}"
            )
            signed_url = planetary_computer.sign(item.assets[asset].href)

            def source_for(
                item: Any = item,
                asset: str = asset,
                cache_key: str = cache_key,
                signed_url: str = signed_url,
            ) -> Tuple[str, bool]:
                local = materialise_tiled_band(
                    cache_key,
                    _materialise_bounds_band(
                        item,
                        asset,
                        bounds_target,
                        target_crs,
                        width,
                        height,
                        resolution,
                        rio_resampling,
                    ),
                )
                return (str(local), True) if local is not None else (signed_url, False)

            scene_sources.append(source_for)
        sources.append(scene_sources)

    if prewarm:
        # Pre-warm sources in parallel. With debug cache on this materialises
        # every (scene, asset) up front; without cache it's just parallel URL
        # signing. Avoids the serial fan-out tile workers would otherwise do
        # when each first touches a (scene, asset) inside the tile loop.
        _prewarm_sources(sources)

    vrt_local = threading.local()

    def _get_source(scene_idx: int, asset_idx: int) -> Any:
        per_thread = getattr(vrt_local, "handles", None)
        if per_thread is None:
            per_thread = {}
            vrt_local.handles = per_thread
        key = (scene_idx, asset_idx)
        entry = per_thread.get(key)
        if entry is None:
            source, is_local = sources[scene_idx][asset_idx]()
            src = rio.open(source)
            if is_local:
                handle: Any = src
            else:
                handle = WarpedVRT(
                    src,
                    crs=target_crs_obj,
                    transform=user_transform,
                    width=width,
                    height=height,
                    resampling=rio_resampling,
                )
            entry = (src, handle)
            per_thread[key] = entry
        return entry[1]

    def read_fn(
        scene_idx: int, band_idx: int, spec: Tuple[int, int, int, int]
    ) -> npt.NDArray[Any]:
        r, c, th, tw = spec
        src = _get_source(scene_idx, band_idx)
        return src.read(href_band_indices[band_idx], window=Window(c, r, tw, th))  # type: ignore[no-any-return, unused-ignore]

    return read_fn


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
) -> npt.NDArray[Any]:
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
        href = planetary_computer.sign(item.assets[band_name].href)
        return _read_warpvrt(
            href, 1, transform, width, height, target_crs_obj, rio_resampling
        )

    with ThreadPoolExecutor(max_workers=len(_OCM_BANDS)) as executor:
        bands = list(executor.map(_read_band, _OCM_BANDS))
    return np.stack(bands, axis=0).astype(np.uint16)  # type: ignore[no-any-return, unused-ignore]


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
    sort_function: Optional[Callable[..., Any]] = None,
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
) -> Union[Tuple[npt.NDArray[Any], Dict[str, Any]], Path]:
    """Bounds-mode pipeline. Called from mosaic() when bounds is set.

    Searches the Planetary Computer STAC for Sentinel-2 L2A scenes intersecting
    ``bounds`` over the date window, streams per-scene cloud masks (skipping
    scenes that contribute no new pixels), then fetches user bands only for the
    kept scenes and aggregates them into a single mosaic on a UTM grid.

    Differs from the grid-mode pipeline in that the AOI is an arbitrary bbox
    (possibly spanning multiple MGRS tiles, possibly intersecting tiles in
    different UTM-zone projections), so each per-scene read goes through a
    rasterio WarpedVRT to land on one common grid in ``target_crs``.

    Args:
        bounds: ``(minx, miny, maxx, maxy)`` in ``bounds_crs`` units. Validated
            for orientation, minimum area, and (for EPSG:4326) lon/lat range;
            very large AOIs emit a warning but are not rejected.
        start_year, start_month, start_day, duration_years, duration_months,
            duration_days: Date window for the STAC search (start inclusive,
            end exclusive).
        bounds_crs: EPSG code of ``bounds``. Defaults to 4326 (lon/lat).
        target_crs: EPSG code of the output grid. If ``None``, auto-picked as
            the UTM zone containing the bbox centroid.
        resolution: Output pixel size in metres. Reads come from the nearest
            COG overview for non-native resolutions.
        sort_method, sort_function, mosaic_method, percentile_value: As for
            :func:`s2mosaic.mosaic`.
        required_bands: Spectral bands to fetch (e.g. ``["B04", "B03", "B02"]``)
            or ``["visual"]`` for the 3-band uint8 TCI asset.
        no_data_threshold: Stop early once the uncovered fraction within the
            coverage mask drops below this. For ``percentile``, the bounds
            mask scan still examines every usable scene, but per-tile user-band
            aggregation can stop once the tile is sufficiently covered.
        coverage_threshold_pct: Drop pixels covered by fewer than this fraction
            of overlapping scenes (scene-edge pixels). ``None`` disables.
        ocm_batch_size, ocm_inference_dtype: Only used when ``cloud_mask="OCM"``.
        additional_query: Extra STAC query filters
            (e.g. ``{"eo:cloud_cover": {"lt": 50}}``).
        ignore_duplicate_items: Keep only the latest processing baseline per scene.
        resampling_method: How source COGs are resampled to ``target_crs``
            / ``resolution`` during the WarpedVRT read.
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
    # materialising them. OCM gets an expanded read if needed so the model sees
    # at least 100 x 100 pixels of spatial context; predictions are cropped
    # back to the requested bounds before scene-selection logic uses them.
    mask_w, mask_h = _grid_shape_for_bounds(bounds_target, mask_resolution)
    ocm_bounds_target = bounds_target
    ocm_crop: Optional[Tuple[slice, slice]] = None
    if cloud_mask == CLOUD_MASK_OCM:
        ocm_bounds_target, ocm_crop = _expand_bounds_for_ocm_context(
            bounds_target, mask_resolution
        )
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
    kept_combo_masks_ocm: Dict[int, npt.NDArray[Any]] = {}
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
                    ocm_bounds_target,
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

    # Prepare the user-resolution masks and target grid for tile aggregation.
    kept_indices = sorted(kept_combo_masks_ocm.keys())
    kept_items = [items_list[i] for i in kept_indices]
    logger.info(f"Streaming user bands for {len(kept_items)}/{n_time} kept scenes")

    # User grid is fixed upfront from bounds_target + resolution — every
    # per-scene fetch snaps to exactly this (transform, w, h), so accumulators
    # can be sized before any data is fetched.
    user_transform, w, h, _ = _target_grid(bounds_target, resolution, target_crs)
    n_bands = 3 if is_visual else len(required_bands)

    def _to_user_shape(mask: npt.NDArray[Any]) -> npt.NDArray[Any]:
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
        bounds_target=bounds_target,
        target_crs=target_crs,
        user_transform=user_transform,
        width=w,
        height=h,
        resolution=resolution,
        resampling_method=resampling_method,
        prewarm=should_prewarm_sources(mosaic_method, no_data_threshold),
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
    output_array = run_tile_aggregation(
        masks=masks_in_order,
        read_fn=read_fn,
        bands_count=n_bands,
        height=h,
        width=w,
        coverage_mask=coverage_mask,
        no_data_threshold=no_data_threshold,
        mosaic_method=mosaic_method,
        percentile_value=percentile_value,
        tile_size=tile_size,
        tile_workers=None,
        out_dtype=np.dtype(np.uint8) if is_visual else np.dtype(np.uint16),
    )

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
