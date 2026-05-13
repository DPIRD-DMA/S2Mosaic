import functools
import hashlib
import logging
import os
import pickle
import time
from datetime import date, datetime
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import geopandas as gpd
import numpy as np
import rasterio as rio
from dateutil.relativedelta import relativedelta
from shapely.geometry.polygon import Polygon

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Debug cache: opt-in via env var. When unset, pickle_cache and the @disk_cache
# decorator are no-ops — callers don't need to thread a flag through their
# signatures. CWD-relative because the typical entry point (notebooks, scripts)
# treats the project working directory as scratch space.
DEBUG_CACHE_DIR = Path("cache")
DEBUG_CACHE_ENV_VAR = "S2MOSAIC_DEBUG_CACHE"


def debug_cache_enabled() -> bool:
    """True if S2MOSAIC_DEBUG_CACHE is set to a truthy value."""
    return os.environ.get(DEBUG_CACHE_ENV_VAR, "").lower() in ("1", "true", "yes")


def pickle_cache(prefix: str, key: str, compute: Callable[[], T]) -> T:
    """Memoize ``compute()`` to ``DEBUG_CACHE_DIR/{prefix}_{md5(key)}.pkl``.

    No-op (just calls ``compute()``) unless ``S2MOSAIC_DEBUG_CACHE`` is set.
    """
    if not debug_cache_enabled():
        return compute()
    digest = hashlib.md5(key.encode()).hexdigest()
    path = DEBUG_CACHE_DIR / f"{prefix}_{digest}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    result = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(result, f)
    return result


def disk_cache(
    prefix: str, key_fn: Callable[..., str]
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator: wraps a function with the optional debug-cache layer.

    ``key_fn`` receives the same args/kwargs as the wrapped function and
    returns a cache-key string. The cache is gated on ``S2MOSAIC_DEBUG_CACHE``
    via :func:`pickle_cache`, so decorated functions transparently skip the
    cache machinery when the env var is unset.
    """

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            if not debug_cache_enabled():
                return fn(*args, **kwargs)
            return pickle_cache(
                prefix, key_fn(*args, **kwargs), lambda: fn(*args, **kwargs)
            )

        return wrapper

    return decorator


class SceneFetchError(Exception):
    """Raised by a per-scene COG/asset fetch after all retries are exhausted.

    Caught at the pipeline-loop level so one bad scene doesn't abort the whole
    mosaic — the loop logs and skips. Errors that aren't fetch-related (e.g.
    OCM inference failures) bypass this and propagate.
    """


def with_scene_retry(
    attempts: int = 3,
    base_delay: float = 1.0,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator: retry a per-scene fetcher with exponential backoff.

    On exhaustion, the last exception is wrapped in :class:`SceneFetchError`
    so the pipeline loop can catch fetch failures specifically without also
    swallowing inference or programming errors that arise outside the fetch.
    Backoff doubles each attempt (``base_delay``, ``2 * base_delay``, ...).
    """

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exc: Optional[BaseException] = None
            for attempt in range(attempts):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if attempt < attempts - 1:
                        delay = base_delay * (2**attempt)
                        logger.warning(
                            f"{fn.__name__} attempt {attempt + 1}/{attempts} "
                            f"failed: {e}; retrying in {delay:.1f}s"
                        )
                        time.sleep(delay)
            assert last_exc is not None
            raise SceneFetchError(
                f"{fn.__name__} failed after {attempts} attempts: {last_exc}"
            ) from last_exc

        return wrapper

    return decorator


SORT_VALID_DATA = "valid_data"
SORT_OLDEST = "oldest"
SORT_NEWEST = "newest"
SORT_CUSTOM = "custom"
MOSAIC_MEAN = "mean"
MOSAIC_FIRST = "first"
MOSAIC_PERCENTILE = "percentile"

CLOUD_MASK_OCM = "OCM"
CLOUD_MASK_SCL = "SCL"

VALID_SORT_METHODS = {SORT_VALID_DATA, SORT_OLDEST, SORT_NEWEST, SORT_CUSTOM}
VALID_MOSAIC_METHODS = {MOSAIC_MEAN, MOSAIC_FIRST, MOSAIC_PERCENTILE}
VALID_CLOUD_MASKS = {CLOUD_MASK_OCM, CLOUD_MASK_SCL}


VALID_RESAMPLING_METHODS = {
    "nearest",
    "bilinear",
    "cubic",
    "average",
    "lanczos",
}


def get_band_template(
    required_bands: List[str],
) -> Tuple[List[Tuple[str, int]], int, List[int]]:
    """Return per-band STAC asset/raster-band template + sizes.

    The result is the same shape for grid_id and bounds modes:

        * ``href_template`` — list of ``(stac_asset_name, raster_band_idx)``;
          one entry per output band.
        * ``bands_count`` — number of output bands.
        * ``href_band_indices`` — just the raster band indices, pulled out
          for the hot path.

    ``"visual"`` is the 3-band TCI asset; spectral requests are one asset
    per band, each reading raster band 1.
    """
    is_visual = "visual" in required_bands
    if is_visual:
        href_template: List[Tuple[str, int]] = [
            ("visual", 1),
            ("visual", 2),
            ("visual", 3),
        ]
        bands_count = 3
    else:
        href_template = [(band, 1) for band in required_bands]
        bands_count = len(required_bands)
    href_band_indices = [band_idx for _, band_idx in href_template]
    return href_template, bands_count, href_band_indices


def get_rasterio_resampling(method: str):
    """Map a string resampling method to a rasterio.enums.Resampling value."""
    from rasterio.enums import Resampling

    return {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "average": Resampling.average,
        "lanczos": Resampling.lanczos,
    }[method]


OCM_MIN_RESOLUTION = 20
OCM_MAX_RESOLUTION = 50

# MGRS tile is exactly 109800m on each side.
MGRS_TILE_SIZE_M = 109800


def pick_ocm_resolution(user_resolution: int) -> int:
    """Pick the OCM input resolution given the user's output resolution.

    OCM is fastest at coarser resolutions (less data to transfer / process).
    20m is the recommended default; 50m is the coarsest tested. We use the
    user's resolution where it falls in [20, 50], and clamp otherwise:
        user <= 20 → 20  (e.g. 10m output keeps OCM at 20m)
        20 < user < 50 → user
        user >= 50 → 50
    """
    return max(OCM_MIN_RESOLUTION, min(user_resolution, OCM_MAX_RESOLUTION))


@lru_cache(maxsize=1)
def _load_s2_grid() -> gpd.GeoDataFrame:
    with resources.as_file(
        resources.files("s2mosaic") / "sentinel_2_index.gpkg"
    ) as path:
        S2_grid_file = Path(path)
    assert S2_grid_file.exists(), (
        f"S2 grid file not found at {S2_grid_file}. "
        "This suggests the S2Mosaic package was not installed correctly. "
        "Please reinstall the package."
    )
    return gpd.read_file(S2_grid_file)


def get_extent_from_grid_id(grid_id: str) -> Polygon:
    try:
        all_grids = _load_s2_grid()
        grid_entry = all_grids[all_grids["Name"] == grid_id]

        return_count = grid_entry.shape[0]

        if return_count == 0:
            raise ValueError(
                f"Grid {grid_id} not found. It should be in the format '50HMH'. "
                "See https://sentiwiki.copernicus.eu/web/s2-products and "
                "https://dpird-dma.github.io/Sentinel-2-grid-explorer/"
            )
        assert return_count == 1, (
            f"Multiple entries found for grid {grid_id}. "
            "This should not happen, please check the S2 grid file."
        )

        return grid_entry.iloc[0].geometry

    except Exception as e:
        logger.error(f"Error reading grid entry: {e}")
        raise


def define_dates(
    start_year: int,
    start_month: int,
    start_day: int,
    duration_years: int,
    duration_months: int,
    duration_days: int,
) -> Tuple[date, date]:
    start_date = datetime(start_year, start_month, start_day)
    end_date = start_date + relativedelta(
        years=duration_years, months=duration_months, days=duration_days
    )
    return start_date, end_date


DEFAULT_REQUIRED_BANDS: List[str] = ["B04", "B03", "B02", "B08"]
DEFAULT_ADDITIONAL_QUERY: Dict[str, Any] = {"eo:cloud_cover": {"lt": 100}}


def normalize_mosaic_inputs(
    required_bands: Optional[List[str]],
    additional_query: Optional[Dict[str, Any]],
    sort_method: str,
    sort_function: Optional[Any],
    mosaic_method: str,
    percentile_value: Optional[float],
) -> Tuple[List[str], Dict[str, Any], str, str, Optional[float]]:
    """Apply defaults and rewrites shared by the grid and bounds pipelines.

    Returns the normalized (required_bands, additional_query, sort_method,
    mosaic_method, percentile_value) — callers should overwrite their locals
    with the returned values. ``"median"`` is rewritten to
    ``("percentile", 50.0)``; ``sort_function`` overrides ``sort_method``.
    """
    if required_bands is None:
        required_bands = list(DEFAULT_REQUIRED_BANDS)
    if additional_query is None:
        additional_query = dict(DEFAULT_ADDITIONAL_QUERY)
    if sort_function is not None:
        sort_method = SORT_CUSTOM
    if mosaic_method == "median":
        if percentile_value is not None:
            raise ValueError(
                "percentile_value should not be set when using mosaic_method='median'."
            )
        mosaic_method = MOSAIC_PERCENTILE
        percentile_value = 50.0
    return (
        required_bands,
        additional_query,
        sort_method,
        mosaic_method,
        percentile_value,
    )


BOUNDS_MIN_DIM_M = 10
BOUNDS_MAX_DIM_M = 200_000


def validate_inputs(
    sort_method: str,
    mosaic_method: str,
    no_data_threshold: Union[float, None],
    required_bands: List[str],
    grid_id: Optional[str],
    percentile_value: Optional[float],
    resampling_method: str = "nearest",
    bounds: Optional[Tuple[float, float, float, float]] = None,
    bounds_crs: Optional[int] = None,
    resolution: Optional[int] = None,
    cloud_mask: str = CLOUD_MASK_OCM,
) -> None:
    if grid_id is not None and (not grid_id.isalnum() or not grid_id.isupper()):
        raise ValueError(
            f"""Grid {grid_id} is invalid. It should be in the format '50HMH'.
            For more info on the S2 grid system visit https://sentiwiki.copernicus.eu/web/s2-products"""
        )
    if bounds is not None:
        if len(bounds) != 4:
            raise ValueError("bounds must be (minx, miny, maxx, maxy)")
        minx, miny, maxx, maxy = bounds
        if minx >= maxx or miny >= maxy:
            raise ValueError(f"Invalid bounds: {bounds}")

        # Range check for EPSG:4326 (lon/lat). A swapped (lat, lon, lat, lon)
        # tuple is caught here whenever a longitude > 90 lands in the latitude
        # slots; ambiguous near (0, 0), which we can't disambiguate without
        # more context.
        if bounds_crs == 4326:
            if not (-180 <= minx <= 180 and -180 <= maxx <= 180):
                raise ValueError(
                    f"Invalid bounds: longitude must be in [-180, 180] for "
                    f"EPSG:4326, got minx={minx}, maxx={maxx}"
                )
            if not (-90 <= miny <= 90 and -90 <= maxy <= 90):
                raise ValueError(
                    f"Invalid bounds: latitude must be in [-90, 90] for "
                    f"EPSG:4326, got miny={miny}, maxy={maxy} "
                    f"(possible lat/lon axis swap)"
                )

        # Size check. For EPSG:4326, convert degrees to approximate metres
        # using the bbox-centre latitude; otherwise treat bounds units as
        # metres (true for UTM and most projected CRSes).
        if bounds_crs == 4326:
            center_lat = (miny + maxy) / 2
            width_m = (maxx - minx) * 111_111 * np.cos(np.radians(center_lat))
            height_m = (maxy - miny) * 111_111
        else:
            width_m = maxx - minx
            height_m = maxy - miny
        if width_m < BOUNDS_MIN_DIM_M or height_m < BOUNDS_MIN_DIM_M:
            raise ValueError(
                f"Invalid bounds: width and height must each be at least "
                f"{BOUNDS_MIN_DIM_M}m, got width={width_m:.2f}m "
                f"height={height_m:.2f}m"
            )
        if width_m > BOUNDS_MAX_DIM_M or height_m > BOUNDS_MAX_DIM_M:
            raise ValueError(
                f"Invalid bounds: width and height must each be at most "
                f"{BOUNDS_MAX_DIM_M / 1000:.0f}km, got "
                f"width={width_m / 1000:.2f}km height={height_m / 1000:.2f}km"
            )
    if resolution is not None and resolution <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}")
    if cloud_mask not in VALID_CLOUD_MASKS:
        raise ValueError(
            f"Invalid cloud_mask: {cloud_mask}. Must be one of {VALID_CLOUD_MASKS}"
        )
    if sort_method not in VALID_SORT_METHODS:
        raise ValueError(
            f"Invalid sort method: {sort_method}. Must be one of {VALID_SORT_METHODS}"
        )
    if mosaic_method not in VALID_MOSAIC_METHODS:
        raise ValueError(
            f"Invalid mosaic method: {mosaic_method}. Must be of {VALID_MOSAIC_METHODS}"
        )
    if resampling_method not in VALID_RESAMPLING_METHODS:
        raise ValueError(
            f"Invalid resampling method: {resampling_method}. "
            f"Must be one of {VALID_RESAMPLING_METHODS}"
        )
    if no_data_threshold is not None:
        if not (0.0 <= no_data_threshold <= 1.0):
            raise ValueError(
                f"""No data threshold must be between 0 and 1 or None, 
                got {no_data_threshold}"""
            )
    valid_bands = [
        "AOT",
        "SCL",
        "WVP",
        "visual",
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B11",
        "B12",
    ]
    for band in required_bands:
        if band not in valid_bands:
            raise ValueError(f"Invalid band: {band}, must be one of {valid_bands}")
    if "visual" in required_bands and len(required_bands) > 1:
        raise ValueError("Cannot use visual band with other bands, must be used alone")

    if mosaic_method != MOSAIC_PERCENTILE:
        if percentile_value is not None:
            raise ValueError(
                f"""percentile_value is only valid for percentile mosaic method, 
                got {percentile_value}"""
            )

    if mosaic_method == MOSAIC_PERCENTILE:
        if percentile_value is None:
            raise ValueError(
                "percentile_value must be provided for percentile mosaic method"
            )
        if percentile_value < 0 or percentile_value > 100:
            raise ValueError(
                f"percentile_value must be between 0 and 100, got {percentile_value}"
            )


def get_output_path(
    output_dir: Union[Path, str],
    start_date: date,
    end_date: date,
    sort_method: str,
    mosaic_method: str,
    required_bands: List[str],
    grid_id: Optional[str] = None,
    bounds: Optional[tuple] = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    bands_str = "_".join(required_bands)

    if grid_id is not None:
        prefix = grid_id
    elif bounds is not None:
        prefix = (
            f"bounds_{bounds[0]:.4f}_{bounds[1]:.4f}_{bounds[2]:.4f}_{bounds[3]:.4f}"
        )
    else:
        raise ValueError("Either grid_id or bounds is required")

    return output_dir / (
        f"{prefix}_{start_date.strftime('%Y-%m-%d')}_to_"
        f"{end_date.strftime('%Y-%m-%d')}_{sort_method}_{mosaic_method}_"
        f"{bands_str}.tif"
    )


def _export_tif(
    array: np.ndarray,
    profile: Dict[str, Any],
    export_path: Path,
    required_bands: List[str],
    nodata_value: Union[int, None] = 0,
) -> None:
    profile.update(
        count=array.shape[0], dtype=array.dtype, nodata=nodata_value, compress="lzw"
    )
    with rio.open(export_path, "w", **profile) as dst:
        dst.write(array)
        dst.descriptions = required_bands


def finalize_output(
    array: np.ndarray,
    profile: Dict[str, Any],
    required_bands: List[str],
    coverage_mask: Optional[np.ndarray],
    export_path: Optional[Path],
) -> Union[Tuple[np.ndarray, Dict[str, Any]], Path]:
    """Apply coverage mask, set band names + nodata, export or return.

    Used at the end of both the grid_id and bounds pipelines so the
    coverage-mask zeroing, ``visual`` → RGB band-name remap, and
    export-vs-return decision live in one place.
    """
    if coverage_mask is not None:
        array = np.where(coverage_mask[None, :, :], array, 0)

    if "visual" in required_bands:
        band_descriptions = ["Red", "Green", "Blue"]
        nodata_value: Optional[int] = None
    else:
        band_descriptions = list(required_bands)
        nodata_value = 0

    if export_path is not None:
        logger.info(f"Writing GeoTIFF to {export_path}")
        _export_tif(
            array=array,
            profile=profile,
            export_path=export_path,
            required_bands=band_descriptions,
            nodata_value=nodata_value,
        )
        return export_path

    logger.info(f"Returning array shape={array.shape} dtype={array.dtype}")
    return array, profile
