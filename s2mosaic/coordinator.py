from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional, overload

import numpy as np
from .helpers import (
    get_extent_from_grid_id,
    validate_inputs,
    define_dates,
    get_output_path,
    search_for_items,
    add_item_info,
    sort_items,
    download_bands_pool,
    export_tif,
)


@overload
def mosaic(
    grid_id: str,
    start_year: int,
    start_month: int = 1,
    start_day: int = 1,
    output_dir: None = None,
    sort_method: str = "valid_data",
    mosaic_method: str = "mean",
    duration_years: int = 0,
    duration_months: int = 0,
    duration_days: int = 0,
    required_bands: List[str] = ["B04", "B03", "B02", "B08"],
    no_data_threshold: Optional[float] = 0.01,
    overwrite: bool = True,
    ocm_batch_size: int = 1,
    ocm_inference_dtype: str = "bf16",
    debug_cache: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]: ...


@overload
def mosaic(
    grid_id: str,
    start_year: int,
    start_month: int = 1,
    start_day: int = 1,
    output_dir: Union[str, Path] = ...,
    sort_method: str = "valid_data",
    mosaic_method: str = "mean",
    duration_years: int = 0,
    duration_months: int = 0,
    duration_days: int = 0,
    required_bands: List[str] = ["B04", "B03", "B02", "B08"],
    no_data_threshold: Optional[float] = 0.01,
    overwrite: bool = True,
    ocm_batch_size: int = 1,
    ocm_inference_dtype: str = "bf16",
    debug_cache: bool = False,
) -> Path: ...


def mosaic(
    grid_id: str,
    start_year: int,
    start_month: int = 1,
    start_day: int = 1,
    output_dir: Optional[Union[Path, str]] = None,
    sort_method: str = "valid_data",
    mosaic_method: str = "mean",
    duration_years: int = 0,
    duration_months: int = 0,
    duration_days: int = 0,
    required_bands: List[str] = ["B04", "B03", "B02", "B08"],
    no_data_threshold: Union[float, None] = 0.01,
    overwrite: bool = True,
    ocm_batch_size: int = 1,
    ocm_inference_dtype: str = "bf16",
    debug_cache: bool = False,
) -> Union[Tuple[np.ndarray, Dict[str, Any]], Path]:
    """
    Create a Sentinel-2 mosaic for a specified grid and time range.

    This function generates a mosaic from Sentinel-2 satellite imagery based on the provided
    grid ID and time range. It can either return the mosaic data and metadata or save it as
    a GeoTIFF file.

    Args:
        grid_id (str): The ID of the grid area for which to create the mosaic (e.g., "50HMH").
        start_year (int): The start year of the time range.
        start_month (int, optional): The start month of the time range. Defaults to 1 (January).
        start_day (int, optional): The start day of the time range. Defaults to 1.
        output_dir (Optional[Union[Path, str]], optional): Directory to save the output GeoTIFF.
            If None, the mosaic is not saved to disk and is returned instead. Defaults to None.
        sort_method (str, optional): Method to sort scenes. Options are "valid_data", "oldest", or "newest". Defaults to "valid_data".
        mosaic_method (str, optional): Method to create the mosaic. Options are "mean" or "first". Defaults to "mean".
        duration_years (int, optional): Duration in years to add to the start date. Defaults to 0.
        duration_months (int, optional): Duration in months to add to the start date. Defaults to 0.
        duration_days (int, optional): Duration in days to add to the start date. Defaults to 0.
        required_bands (List[str], optional): List of required spectral bands.
            Defaults to ["B04", "B03", "B02", "B08"] (Red, Green, Blue, NIR).
        no_data_threshold (float, optional): Threshold for no data values. Defaults to 0.01.
        overwrite (bool, optional): Whether to overwrite existing output files. Defaults to True.
        ocm_batch_size (int, optional): Batch size for OCM inference. Defaults to 1.
        ocm_inference_dtype (str, optional): Data type for OCM inference. Defaults to "bf16".

    Returns:
        Union[Tuple[np.ndarray, Dict[str, Any]], Path]: If output_dir is None, returns a tuple
        containing the mosaic array and metadata dictionary. If output_dir is provided,
        returns the path to the saved GeoTIFF file.

    Raises:
        Exception: If no scenes are found for the specified grid ID and time range.

    Note:
        - The function uses the STAC API to search for Sentinel-2 scenes.
        - If 'visual' is included in required_bands, it will be replaced with 'Red', 'Green', 'Blue' in the output.
        - The time range for scene selection is inclusive of the start date and exclusive of the end date.
    """
    bounds = get_extent_from_grid_id(grid_id)

    validate_inputs(sort_method, mosaic_method, no_data_threshold, required_bands)

    start_date, end_date = define_dates(
        start_year,
        start_month,
        start_day,
        duration_years,
        duration_months,
        duration_days,
    )
    if output_dir:
        export_path = get_output_path(
            grid_id=grid_id,
            start_date=start_date,
            end_date=end_date,
            sort_method=sort_method,
            mosaic_method=mosaic_method,
            required_bands=required_bands,
            output_dir=output_dir,
        )

    if output_dir:
        if export_path.exists() and not overwrite:
            return export_path

    items = search_for_items(
        bounds=bounds, grid_id=grid_id, start_date=start_date, end_date=end_date
    )

    if len(items) == 0:
        raise Exception(
            f"No scenes found for {grid_id} between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}"
        )

    items_with_orbits = add_item_info(items)

    sorted_items = sort_items(items=items_with_orbits, sort_method=sort_method)

    mosaic, profile = download_bands_pool(
        sorted_scenes=sorted_items,
        required_bands=required_bands,
        no_data_threshold=no_data_threshold,
        mosaic_method=mosaic_method,
        ocm_batch_size=ocm_batch_size,
        ocm_inference_dtype=ocm_inference_dtype,
        debug_cache=debug_cache,
    )
    if "visual" in required_bands:
        required_bands = ["Red", "Green", "Blue"]
        nodata_value = None
    else:
        nodata_value = 0

    if output_dir:
        export_tif(
            array=mosaic,
            profile=profile,
            export_path=export_path,
            required_bands=required_bands,
            nodata_value=nodata_value,
        )
        return export_path

    return mosaic, profile