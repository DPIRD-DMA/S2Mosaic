"""Input normalization and validation for mosaic requests."""

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from shapely.geometry.polygon import Polygon

from .geometry import Aoi, Bbox
from .sources import SOURCE_MPC

logger = logging.getLogger(__name__)

SCENE_ORDER_VALID_DATA = "valid_data"
SCENE_ORDER_OLDEST = "oldest"
SCENE_ORDER_NEWEST = "newest"
SCENE_ORDER_CUSTOM = "custom"
MOSAIC_MEAN = "mean"
MOSAIC_FIRST = "first"
MOSAIC_PERCENTILE = "percentile"

CLOUD_MASK_OCM = "OCM"
CLOUD_MASK_SCL = "SCL"

VALID_SCENE_ORDERS = {
    SCENE_ORDER_VALID_DATA,
    SCENE_ORDER_OLDEST,
    SCENE_ORDER_NEWEST,
    SCENE_ORDER_CUSTOM,
}
VALID_MOSAIC_METHODS = {MOSAIC_MEAN, MOSAIC_FIRST, MOSAIC_PERCENTILE}
VALID_CLOUD_MASKS = {CLOUD_MASK_OCM, CLOUD_MASK_SCL}
VALID_RESAMPLING_METHODS = {
    "nearest",
    "bilinear",
    "cubic",
    "average",
    "lanczos",
}
VALID_BANDS = frozenset(
    {
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
    }
)

DEFAULT_BANDS: List[str] = ["B04", "B03", "B02", "B08"]
DEFAULT_ADDITIONAL_QUERY: Dict[str, Any] = {"eo:cloud_cover": {"lt": 100}}

BOUNDS_MIN_AREA_M2 = 100
BOUNDS_MIN_DIM_M = 10
BOUNDS_LARGE_PIXEL_WARNING_COUNT = 20_000 * 20_000


@dataclass(frozen=True)
class MosaicRequest:
    """Normalized configuration for a single mosaic request.

    Exactly one of ``grid_id``, ``bounds``, or ``aoi`` selects the spatial
    mode. ``grid_id`` mosaics a full Sentinel-2 MGRS tile; ``bounds`` and
    ``aoi`` stream intersecting scenes onto a common output grid. Call
    :meth:`normalized` before :meth:`validate` so optional public inputs such as
    ``bands`` and ``additional_query`` are expanded to concrete values.
    """

    grid_id: Optional[str] = None
    bounds: Optional[Bbox] = None
    aoi: Optional[Aoi] = None
    input_crs: int = 4326
    start_year: Optional[int] = None
    start_month: int = 1
    start_day: int = 1
    duration_years: int = 0
    duration_months: int = 0
    duration_days: int = 0
    bands: Optional[List[str]] = None
    mosaic_method: str = MOSAIC_MEAN
    percentile: Optional[float] = None
    output_dir: Optional[Union[Path, str]] = None
    output_path: Optional[Union[Path, str]] = None
    overwrite: bool = True
    output_crs: Optional[int] = None
    resolution: int = 10
    resampling_method: str = "nearest"
    additional_query: Optional[Dict[str, Any]] = None
    source: str = SOURCE_MPC
    min_observations: Optional[int] = None
    max_observations: Optional[int] = None
    min_coverage_fraction: Optional[float] = None
    ignore_duplicate_items: bool = True
    scene_order: str = SCENE_ORDER_VALID_DATA
    scene_sort_fn: Optional[Callable[..., Any]] = None
    cloud_mask: str = CLOUD_MASK_OCM
    ocm_batch_size: int = 1
    ocm_inference_dtype: str = "bf16"
    tile_workers: Optional[int] = None
    adaptive_tiling: bool = True
    show_progress: bool = False

    def normalized(self) -> "MosaicRequest":
        bands, additional_query, scene_order, mosaic_method, percentile = (
            normalize_mosaic_inputs(
                bands=self.bands,
                additional_query=self.additional_query,
                scene_order=self.scene_order,
                scene_sort_fn=self.scene_sort_fn,
                mosaic_method=self.mosaic_method,
                percentile=self.percentile,
            )
        )
        return replace(
            self,
            bands=bands,
            additional_query=additional_query,
            scene_order=scene_order,
            mosaic_method=mosaic_method,
            percentile=percentile,
        )

    def validate(self) -> None:
        if sum(x is not None for x in (self.grid_id, self.bounds, self.aoi)) != 1:
            raise ValueError("Exactly one of grid_id, bounds, or aoi must be provided")
        if self.bands is None:
            raise ValueError("MosaicRequest must be normalized before validation")
        if self.start_year is None:
            raise ValueError("start_year must be provided")
        if self.start_year <= 0:
            raise ValueError(f"start_year must be positive, got {self.start_year}")
        validate_inputs(
            scene_order=self.scene_order,
            mosaic_method=self.mosaic_method,
            bands=self.bands,
            grid_id=self.grid_id,
            percentile=self.percentile,
            resampling_method=self.resampling_method,
            bounds=self.bounds,
            aoi=self.aoi,
            input_crs=self.input_crs,
            resolution=self.resolution,
            cloud_mask=self.cloud_mask,
            min_observations=self.min_observations,
            max_observations=self.max_observations,
            tile_workers=self.tile_workers,
            adaptive_tiling=self.adaptive_tiling,
            min_coverage_fraction=self.min_coverage_fraction,
            source=self.source,
        )


def normalize_mosaic_inputs(
    bands: Optional[List[str]],
    additional_query: Optional[Dict[str, Any]],
    scene_order: str,
    scene_sort_fn: Optional[Any],
    mosaic_method: str,
    percentile: Optional[float],
) -> Tuple[List[str], Dict[str, Any], str, str, Optional[float]]:
    if bands is None:
        bands = list(DEFAULT_BANDS)
    if additional_query is None:
        additional_query = dict(DEFAULT_ADDITIONAL_QUERY)
    if scene_sort_fn is not None:
        scene_order = SCENE_ORDER_CUSTOM
    if mosaic_method == "median":
        if percentile is not None:
            raise ValueError(
                "percentile should not be set when using mosaic_method='median'."
            )
        mosaic_method = MOSAIC_PERCENTILE
        percentile = 50.0
    return (
        bands,
        additional_query,
        scene_order,
        mosaic_method,
        percentile,
    )


def validate_inputs(
    scene_order: str,
    mosaic_method: str,
    bands: List[str],
    grid_id: Optional[str],
    percentile: Optional[float],
    resampling_method: str = "nearest",
    bounds: Optional[Bbox] = None,
    input_crs: Optional[int] = None,
    resolution: Optional[int] = None,
    cloud_mask: str = CLOUD_MASK_OCM,
    min_observations: Optional[int] = None,
    max_observations: Optional[int] = None,
    tile_workers: Optional[int] = None,
    adaptive_tiling: bool = True,
    aoi: Optional[Polygon] = None,
    min_coverage_fraction: Optional[float] = None,
    source: str = SOURCE_MPC,
) -> None:
    from .sources import VALID_SOURCES

    if source not in VALID_SOURCES:
        raise ValueError(
            f"Invalid source: {source}. Must be one of {sorted(VALID_SOURCES)}"
        )
    if grid_id is not None and (
        len(grid_id) != 5 or not grid_id.isalnum() or not grid_id.isupper()
    ):
        raise ValueError(
            f"Grid {grid_id} is invalid. It should be in the format '50HMH'. "
            "For more info on the S2 grid system visit "
            "https://sentiwiki.copernicus.eu/web/s2-products"
        )
    if aoi is not None:
        if not isinstance(aoi, Polygon):
            raise ValueError("aoi must be a single shapely Polygon")
        if aoi.is_empty:
            raise ValueError("aoi must not be empty")
        if not aoi.is_valid:
            raise ValueError("aoi must be a valid Polygon")
        if aoi.area <= 0:
            raise ValueError("aoi must have a positive area")

    if resolution is not None and resolution <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}")

    bounds_to_validate = (
        bounds if bounds is not None else (aoi.bounds if aoi is not None else None)
    )
    if bounds_to_validate is not None:
        _validate_bounds(bounds_to_validate, input_crs, resolution)

    if cloud_mask not in VALID_CLOUD_MASKS:
        raise ValueError(
            f"Invalid cloud_mask: {cloud_mask}. Must be one of {VALID_CLOUD_MASKS}"
        )
    if scene_order not in VALID_SCENE_ORDERS:
        raise ValueError(
            f"Invalid scene_order: {scene_order}. Must be one of {VALID_SCENE_ORDERS}"
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
    if min_coverage_fraction is not None and not (0.0 <= min_coverage_fraction <= 1.0):
        raise ValueError(
            f"min_coverage_fraction must be between 0 and 1 or None, "
            f"got {min_coverage_fraction}"
        )
    if min_observations is not None:
        if (
            isinstance(min_observations, bool)
            or not isinstance(min_observations, int)
            or min_observations < 1
        ):
            raise ValueError(
                "min_observations must be a positive integer or None, "
                f"got {min_observations}"
            )
    if max_observations is not None:
        if (
            isinstance(max_observations, bool)
            or not isinstance(max_observations, int)
            or max_observations < 1
        ):
            raise ValueError(
                "max_observations must be a positive integer or None, "
                f"got {max_observations}"
            )
        if min_observations is not None and max_observations < min_observations:
            raise ValueError(
                f"max_observations ({max_observations}) must be >= "
                f"min_observations ({min_observations})"
            )
    if tile_workers is not None:
        if (
            isinstance(tile_workers, bool)
            or not isinstance(tile_workers, int)
            or tile_workers < 1
        ):
            raise ValueError(
                f"tile_workers must be a positive integer or None, got {tile_workers}"
            )
    if not isinstance(adaptive_tiling, bool):
        raise ValueError(f"adaptive_tiling must be a bool, got {adaptive_tiling}")
    for band in bands:
        if band not in VALID_BANDS:
            raise ValueError(
                f"Invalid band: {band}, must be one of {sorted(VALID_BANDS)}"
            )
    if "visual" in bands and len(bands) > 1:
        raise ValueError("Cannot use visual band with other bands, must be used alone")

    if mosaic_method != MOSAIC_PERCENTILE and percentile is not None:
        raise ValueError(
            f"percentile is only valid for percentile mosaic method, got {percentile}"
        )

    if mosaic_method == MOSAIC_PERCENTILE:
        if percentile is None:
            raise ValueError("percentile must be provided for percentile mosaic method")
        if percentile < 0 or percentile > 100:
            raise ValueError(f"percentile must be between 0 and 100, got {percentile}")


def _validate_bounds(
    bounds_to_validate: Bbox, input_crs: Optional[int], resolution: Optional[int]
) -> None:
    if len(bounds_to_validate) != 4:
        raise ValueError("bounds must be (minx, miny, maxx, maxy)")
    minx, miny, maxx, maxy = bounds_to_validate
    if minx >= maxx or miny >= maxy:
        raise ValueError(f"Invalid bounds: {bounds_to_validate}")

    if input_crs == 4326:
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
    area_m2 = width_m * height_m
    if area_m2 < BOUNDS_MIN_AREA_M2:
        raise ValueError(
            f"Invalid bounds: area must be at least {BOUNDS_MIN_AREA_M2} "
            f"square metres, got area={area_m2:.2f}m^2 "
            f"(width={width_m:.2f}m height={height_m:.2f}m)"
        )
    if resolution is None:
        return

    output_width_px = int(np.ceil(width_m / resolution))
    output_height_px = int(np.ceil(height_m / resolution))
    output_pixels = output_width_px * output_height_px
    if output_pixels > BOUNDS_LARGE_PIXEL_WARNING_COUNT:
        logger.warning(
            "Bounds output is larger than a 20,000 x 20,000 pixel raster; "
            "this may require "
            "substantial time, memory, and network I/O "
            "(pixels=%d width_px=%d height_px=%d resolution=%sm "
            "area=%.2fkm^2 width=%.2fkm height=%.2fkm)",
            output_pixels,
            output_width_px,
            output_height_px,
            resolution,
            area_m2 / 1_000_000,
            width_m / 1000,
            height_m / 1000,
        )
