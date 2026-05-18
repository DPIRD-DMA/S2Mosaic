"""Input normalization and validation for mosaic requests."""

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from shapely.geometry.polygon import Polygon

from .geometry import Aoi, Bbox

logger = logging.getLogger(__name__)

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

DEFAULT_REQUIRED_BANDS: List[str] = ["B04", "B03", "B02", "B08"]
DEFAULT_ADDITIONAL_QUERY: Dict[str, Any] = {"eo:cloud_cover": {"lt": 100}}

BOUNDS_MIN_AREA_M2 = 100
BOUNDS_MIN_DIM_M = 10
BOUNDS_LARGE_AREA_WARNING_M2 = 200_000 * 200_000


@dataclass(frozen=True)
class MosaicRequest:
    grid_id: Optional[str] = None
    bounds: Optional[Bbox] = None
    aoi: Optional[Aoi] = None
    input_crs: int = 4326
    start_year: int = 0
    start_month: int = 1
    start_day: int = 1
    duration_years: int = 0
    duration_months: int = 0
    duration_days: int = 0
    required_bands: Optional[List[str]] = None
    mosaic_method: str = MOSAIC_MEAN
    percentile: Optional[float] = None
    output_dir: Optional[Union[Path, str]] = None
    output_path: Optional[Union[Path, str]] = None
    overwrite: bool = True
    output_crs: Optional[int] = None
    resolution: int = 10
    resampling_method: str = "nearest"
    additional_query: Optional[Dict[str, Any]] = None
    source: str = "MPC"
    no_data_tolerance: Optional[float] = 0.0
    observation_target: Optional[int] = None
    min_coverage_fraction: Optional[float] = 0.1
    ignore_duplicate_items: bool = True
    sort_method: str = SORT_VALID_DATA
    sort_function: Optional[Callable[..., Any]] = None
    cloud_mask: str = CLOUD_MASK_OCM
    ocm_batch_size: int = 1
    ocm_inference_dtype: str = "bf16"
    tile_workers: Optional[int] = None
    adaptive_tiling: bool = True
    show_progress: bool = False

    def normalized(self) -> "MosaicRequest":
        required_bands, additional_query, sort_method, mosaic_method, percentile = (
            normalize_mosaic_inputs(
                required_bands=self.required_bands,
                additional_query=self.additional_query,
                sort_method=self.sort_method,
                sort_function=self.sort_function,
                mosaic_method=self.mosaic_method,
                percentile=self.percentile,
            )
        )
        return replace(
            self,
            required_bands=required_bands,
            additional_query=additional_query,
            sort_method=sort_method,
            mosaic_method=mosaic_method,
            percentile=percentile,
        )

    def validate(self) -> None:
        if sum(x is not None for x in (self.grid_id, self.bounds, self.aoi)) != 1:
            raise ValueError("Exactly one of grid_id, bounds, or aoi must be provided")
        if self.required_bands is None:
            raise ValueError("MosaicRequest must be normalized before validation")
        validate_inputs(
            sort_method=self.sort_method,
            mosaic_method=self.mosaic_method,
            no_data_tolerance=self.no_data_tolerance,
            required_bands=self.required_bands,
            grid_id=self.grid_id,
            percentile=self.percentile,
            resampling_method=self.resampling_method,
            bounds=self.bounds,
            aoi=self.aoi,
            input_crs=self.input_crs,
            resolution=self.resolution,
            cloud_mask=self.cloud_mask,
            observation_target=self.observation_target,
            tile_workers=self.tile_workers,
            adaptive_tiling=self.adaptive_tiling,
            min_coverage_fraction=self.min_coverage_fraction,
            source=self.source,
        )


def normalize_mosaic_inputs(
    required_bands: Optional[List[str]],
    additional_query: Optional[Dict[str, Any]],
    sort_method: str,
    sort_function: Optional[Any],
    mosaic_method: str,
    percentile: Optional[float],
) -> Tuple[List[str], Dict[str, Any], str, str, Optional[float]]:
    if required_bands is None:
        required_bands = list(DEFAULT_REQUIRED_BANDS)
    if additional_query is None:
        additional_query = dict(DEFAULT_ADDITIONAL_QUERY)
    if sort_function is not None:
        sort_method = SORT_CUSTOM
    if mosaic_method == "median":
        if percentile is not None:
            raise ValueError(
                "percentile should not be set when using mosaic_method='median'."
            )
        mosaic_method = MOSAIC_PERCENTILE
        percentile = 50.0
    return (
        required_bands,
        additional_query,
        sort_method,
        mosaic_method,
        percentile,
    )


def validate_inputs(
    sort_method: str,
    mosaic_method: str,
    no_data_tolerance: Union[float, None],
    required_bands: List[str],
    grid_id: Optional[str],
    percentile: Optional[float],
    resampling_method: str = "nearest",
    bounds: Optional[Bbox] = None,
    input_crs: Optional[int] = None,
    resolution: Optional[int] = None,
    cloud_mask: str = CLOUD_MASK_OCM,
    observation_target: Optional[int] = None,
    tile_workers: Optional[int] = None,
    adaptive_tiling: bool = True,
    aoi: Optional[Polygon] = None,
    min_coverage_fraction: Optional[float] = None,
    source: str = "MPC",
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

    bounds_to_validate = bounds if bounds is not None else aoi.bounds if aoi else None
    if bounds_to_validate is not None:
        _validate_bounds(bounds_to_validate, input_crs)

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
    if no_data_tolerance is not None and not (0.0 <= no_data_tolerance <= 1.0):
        raise ValueError(
            f"No data threshold must be between 0 and 1 or None, "
            f"got {no_data_tolerance}"
        )
    if min_coverage_fraction is not None and not (0.0 <= min_coverage_fraction <= 1.0):
        raise ValueError(
            f"min_coverage_fraction must be between 0 and 1 or None, "
            f"got {min_coverage_fraction}"
        )
    if observation_target is not None:
        if (
            isinstance(observation_target, bool)
            or not isinstance(observation_target, int)
            or observation_target < 1
        ):
            raise ValueError(
                "observation_target must be a positive integer or None, "
                f"got {observation_target}"
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
    for band in required_bands:
        if band not in VALID_BANDS:
            raise ValueError(
                f"Invalid band: {band}, must be one of {sorted(VALID_BANDS)}"
            )
    if "visual" in required_bands and len(required_bands) > 1:
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


def _validate_bounds(bounds_to_validate: Bbox, input_crs: Optional[int]) -> None:
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
    if area_m2 > BOUNDS_LARGE_AREA_WARNING_M2:
        logger.warning(
            "Bounds area is larger than 200km x 200km; this may require "
            "substantial time, memory, and network I/O "
            "(area=%.2fkm^2 width=%.2fkm height=%.2fkm)",
            area_m2 / 1_000_000,
            width_m / 1000,
            height_m / 1000,
        )
