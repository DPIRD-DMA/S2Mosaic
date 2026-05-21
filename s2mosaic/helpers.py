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
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    Optional,
    Tuple,
    TypeVar,
)

import geopandas as gpd
from dateutil.relativedelta import relativedelta
from shapely.geometry.polygon import Polygon

if TYPE_CHECKING:
    from rasterio.enums import Resampling

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
            return pickle.load(f)  # type: ignore[no-any-return, unused-ignore]
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


def _exception_chain_summary(exc: BaseException) -> str:
    """Compactly format an exception plus its Python cause/context chain."""
    parts = []
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
    return " <- ".join(parts)


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
                            f"failed: {_exception_chain_summary(e)}; "
                            f"retrying in {delay:.1f}s"
                        )
                        time.sleep(delay)
            assert last_exc is not None
            raise SceneFetchError(
                f"{fn.__name__} failed after {attempts} attempts: "
                f"{_exception_chain_summary(last_exc)}"
            ) from last_exc

        return wrapper

    return decorator


def get_band_template(
    bands: List[str],
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
    is_visual = "visual" in bands
    if is_visual:
        href_template: List[Tuple[str, int]] = [
            ("visual", 1),
            ("visual", 2),
            ("visual", 3),
        ]
        bands_count = 3
    else:
        href_template = [(band, 1) for band in bands]
        bands_count = len(bands)
    href_band_indices = [band_idx for _, band_idx in href_template]
    return href_template, bands_count, href_band_indices


def get_rasterio_resampling(method: str) -> "Resampling":
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
        s2_grid_file = Path(path)
        if not s2_grid_file.exists():
            raise RuntimeError(
                f"S2 grid file not found at {s2_grid_file}. "
                "This suggests the S2Mosaic package was not installed correctly. "
                "Please reinstall the package."
            )
        return gpd.read_file(s2_grid_file)


def get_extent_from_grid_id(grid_id: str) -> Polygon:
    all_grids = _load_s2_grid()
    grid_entry = all_grids[all_grids["Name"] == grid_id]

    return_count = grid_entry.shape[0]
    if return_count == 0:
        raise ValueError(
            f"Grid {grid_id} not found. It should be in the format '50HMH'. "
            "See https://sentiwiki.copernicus.eu/web/s2-products and "
            "https://dpird-dma.github.io/Sentinel-2-grid-explorer/"
        )
    if return_count > 1:
        raise RuntimeError(
            f"Multiple entries found for grid {grid_id}. "
            "This should not happen, please check the S2 grid file."
        )

    return grid_entry.iloc[0].geometry


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
