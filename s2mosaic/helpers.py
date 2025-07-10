import logging
from datetime import date, datetime
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import rasterio as rio
from dateutil.relativedelta import relativedelta
from shapely.geometry.polygon import Polygon

logger = logging.getLogger(__name__)


SORT_VALID_DATA = "valid_data"
SORT_OLDEST = "oldest"
SORT_NEWEST = "newest"
SORT_CUSTOM = "custom"
MOSAIC_MEAN = "mean"
MOSAIC_FIRST = "first"
MOSAIC_PERCENTILE = "percentile"

VALID_SORT_METHODS = {SORT_VALID_DATA, SORT_OLDEST, SORT_NEWEST, SORT_CUSTOM}
VALID_MOSAIC_METHODS = {MOSAIC_MEAN, MOSAIC_FIRST, MOSAIC_PERCENTILE}


def format_progress(current, total, no_data_pct):
    return f"Scenes: {current}/{total} | Mosaic currently contains {no_data_pct:.2f}% no data pixels"  # noqa: E501


def get_extent_from_grid_id(grid_id: str) -> Polygon:
    with resources.as_file(
        resources.files("s2mosaic") / "sentinel_2_index.gpkg"
    ) as path:
        S2_grid_file = Path(path)

    assert S2_grid_file.exists(), (
        f"S2 grid file not found at {S2_grid_file}. "
        "This suggests the S2Mosaic package was not installed correctly. "
        "Please reinstall the package."
    )

    try:
        all_grids = gpd.read_file(S2_grid_file)
        grid_entry = all_grids[all_grids["Name"] == grid_id]

        return_count = grid_entry.shape[0]

        if return_count == 0:
            raise ValueError(
                f"""Grid {grid_id} not found. It should be in the format '50HMH'. 
                for more info on the S2 grid system visit https://sentiwiki.copernicus.eu/web/s2-products
                View a map of the S2 grid at https://dpird-dma.github.io/Sentinel-2-grid-explorer/
                File: {S2_grid_file}
                Gdf: {all_grids.head(5)}"""
            )
        assert return_count == 1, (
            f"""Multiple entries found for grid {grid_id}. 
            This should not happen, please check the S2 grid file."""
            f"File: {S2_grid_file}"
            f"Gdf: {all_grids.head(5)}"
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


def validate_inputs(
    sort_method: str,
    mosaic_method: str,
    no_data_threshold: Union[float, None],
    required_bands: List[str],
    grid_id: str,
    percentile_value: Optional[float],
) -> None:
    if not grid_id.isalnum() or not grid_id.isupper():
        raise ValueError(
            f"""Grid {grid_id} is invalid. It should be in the format '50HMH'. 
            For more info on the S2 grid system visit https://sentiwiki.copernicus.eu/web/s2-products"""
        )
    if sort_method not in VALID_SORT_METHODS:
        raise ValueError(
            f"Invalid sort method: {sort_method}. Must be one of {VALID_SORT_METHODS}"
        )
    if mosaic_method not in VALID_MOSAIC_METHODS:
        raise ValueError(
            f"Invalid mosaic method: {mosaic_method}. Must be of {VALID_MOSAIC_METHODS}"
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
    grid_id: str,
    start_date: date,
    end_date: date,
    sort_method: str,
    mosaic_method: str,
    required_bands: List[str],
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    bands_str = "_".join(required_bands)
    export_path = output_dir / (
        f"{grid_id}_{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}_{sort_method}_{mosaic_method}_{bands_str}.tif"
    )
    return export_path


def export_tif(
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
