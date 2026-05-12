import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, overload

import numpy as np

from .bounds import Bbox, run_bounds_pipeline
from .frequent_coverage import get_frequent_coverage
from .helpers import (
    MGRS_TILE_SIZE_M,
    define_dates,
    finalize_output,
    get_extent_from_grid_id,
    get_output_path,
    normalize_mosaic_inputs,
    validate_inputs,
)
from .mosaic_core import download_bands_pool
from .stac_utils import add_item_info, search_for_items, sort_items

logger = logging.getLogger(__name__)


@overload
def mosaic(
    grid_id: Optional[str] = ...,
    *,
    start_year: int,
    start_month: int = ...,
    start_day: int = ...,
    output_dir: None = None,
    sort_method: str = ...,
    sort_function: Optional[Callable] = ...,
    mosaic_method: str = ...,
    duration_years: int = ...,
    duration_months: int = ...,
    duration_days: int = ...,
    required_bands: Optional[List[str]] = ...,
    no_data_threshold: Union[float, None] = ...,
    overwrite: bool = ...,
    ocm_batch_size: int = ...,
    ocm_inference_dtype: str = ...,
    additional_query: Optional[Dict[str, Any]] = ...,
    percentile_value: Optional[float] = ...,
    ignore_duplicate_items: bool = ...,
    bounds: Optional[Bbox] = ...,
    bounds_crs: int = ...,
    target_crs: Optional[int] = ...,
    resolution: int = ...,
    coverage_threshold_pct: Optional[float] = ...,
    resampling_method: str = ...,
    cloud_mask: str = ...,
) -> Tuple[np.ndarray, Dict[str, Any]]: ...


@overload
def mosaic(
    grid_id: Optional[str] = ...,
    *,
    start_year: int,
    start_month: int = ...,
    start_day: int = ...,
    output_dir: Union[Path, str],
    sort_method: str = ...,
    sort_function: Optional[Callable] = ...,
    mosaic_method: str = ...,
    duration_years: int = ...,
    duration_months: int = ...,
    duration_days: int = ...,
    required_bands: Optional[List[str]] = ...,
    no_data_threshold: Union[float, None] = ...,
    overwrite: bool = ...,
    ocm_batch_size: int = ...,
    ocm_inference_dtype: str = ...,
    additional_query: Optional[Dict[str, Any]] = ...,
    percentile_value: Optional[float] = ...,
    ignore_duplicate_items: bool = ...,
    bounds: Optional[Bbox] = ...,
    bounds_crs: int = ...,
    target_crs: Optional[int] = ...,
    resolution: int = ...,
    coverage_threshold_pct: Optional[float] = ...,
    resampling_method: str = ...,
    cloud_mask: str = ...,
) -> Path: ...


def mosaic(
    grid_id: Optional[str] = None,
    *,
    start_year: int,
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
    no_data_threshold: Union[float, None] = 0.01,
    overwrite: bool = True,
    ocm_batch_size: int = 1,
    ocm_inference_dtype: str = "bf16",
    additional_query: Optional[Dict[str, Any]] = None,
    percentile_value: Optional[float] = None,
    ignore_duplicate_items: bool = True,
    bounds: Optional[Bbox] = None,
    bounds_crs: int = 4326,
    target_crs: Optional[int] = None,
    resolution: int = 10,
    coverage_threshold_pct: Optional[float] = 0.1,
    resampling_method: str = "nearest",
    cloud_mask: str = "OCM",
) -> Union[Tuple[np.ndarray, Dict[str, Any]], Path]:
    """
    Create a Sentinel-2 mosaic.

    Two modes — pass exactly one of:
        * ``grid_id`` (e.g. "50HMH"): mosaic an entire MGRS tile.
        * ``bounds`` (minx, miny, maxx, maxy): mosaic an arbitrary bounding
          box. Scenes from any intersecting MGRS tiles are streamed through
          per-scene rasterio WarpedVRT reads and aggregated on a common UTM
          grid in ``target_crs``.

    ``bounds_crs`` and ``target_crs`` only apply when ``bounds`` is set.

    Args:
        grid_id (str, optional): MGRS tile ID (e.g. "50HMH"). Mutually exclusive with bounds.
        start_year (int): The start year of the time range.
        start_month (int, optional): The start month of the time range. Defaults to 1 (January).
        start_day (int, optional): The start day of the time range. Defaults to 1.
        output_dir (Optional[Union[Path, str]], optional): Directory to save the output GeoTIFF.
            If None, the mosaic is not saved to disk and is returned instead. Defaults to None.
        sort_method (str, optional): Method to sort scenes. Options are "valid_data", "oldest", or "newest". Defaults to "valid_data".
        sort_function (Callable, optional): Custom sorting function. If provided, overrides sort_method.
        mosaic_method (str, optional): Method to create the mosaic. Options are "mean", "first", "median" or "percentile". Defaults to "mean".
        duration_years (int, optional): Duration in years to add to the start date. Defaults to 0.
        duration_months (int, optional): Duration in months to add to the start date. Defaults to 0.
        duration_days (int, optional): Duration in days to add to the start date. Defaults to 0.
        required_bands (List[str], optional): List of required spectral bands.
            Defaults to ["B04", "B03", "B02", "B08"] (Red, Green, Blue, NIR).
        no_data_threshold (float, optional): Threshold for no data values. Defaults to 0.01.
        overwrite (bool, optional): Whether to overwrite existing output files. Defaults to True.
        ocm_batch_size (int, optional): Batch size for OCM inference. Defaults to 1.
        ocm_inference_dtype (str, optional): Data type for OCM inference. Defaults to "bf16".
        additional_query (Dict[str, Any], optional): Additional query parameters for STAC API.
            Defaults to {"eo:cloud_cover": {"lt": 100}}.
        percentile_value (Optional[float], optional): If provided, calculates the specified percentile mosaic.
            must be used with `mosaic_method='percentile'`. Defaults to None, can be a value between 0 and 100.
        ignore_duplicate_items (bool, optional): Whether to remove duplicate scenes based on their IDs. Defaults to True.
        cloud_mask (str, optional): Cloud-mask provider. ``"OCM"`` (default) runs the
            OmniCloudMask deep-learning model on R+G+NIR; ``"SCL"`` reads the L2A
            Scene Classification Layer published with the scene. SCL is much cheaper
            (one COG read, no inference) but lower accuracy.

    Returns:
        Union[Tuple[np.ndarray, Dict[str, Any]], Path]: If output_dir is None, returns a tuple
        containing the mosaic array and metadata dictionary. If output_dir is provided,
        returns the path to the saved GeoTIFF file.

    Raises:
        ValueError: If inputs fail validation, or if no scenes are found for the
            specified grid_id / bounds and time range.
        RuntimeError: If scenes were found but all were fully cloud-masked or
            invalid (bounds mode only).

    Note:
        - The function uses the STAC API to search for Sentinel-2 scenes.
        - If 'visual' is included in required_bands, it will be replaced with 'Red', 'Green', 'Blue' in the output.
        - The time range for scene selection is inclusive of the start date and exclusive of the end date.
    """  # noqa: E501
    if (grid_id is None) == (bounds is None):
        raise ValueError("Exactly one of grid_id or bounds must be provided")

    if bounds is not None:
        return run_bounds_pipeline(
            bounds=bounds,
            start_year=start_year,
            bounds_crs=bounds_crs,
            target_crs=target_crs,
            resolution=resolution,
            start_month=start_month,
            start_day=start_day,
            output_dir=output_dir,
            sort_method=sort_method,
            sort_function=sort_function,
            mosaic_method=mosaic_method,
            duration_years=duration_years,
            duration_months=duration_months,
            duration_days=duration_days,
            required_bands=required_bands,
            no_data_threshold=no_data_threshold,
            overwrite=overwrite,
            ocm_batch_size=ocm_batch_size,
            ocm_inference_dtype=ocm_inference_dtype,
            additional_query=additional_query,
            percentile_value=percentile_value,
            ignore_duplicate_items=ignore_duplicate_items,
            coverage_threshold_pct=coverage_threshold_pct,
            resampling_method=resampling_method,
            cloud_mask=cloud_mask,
        )

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
        f"Creating mosaic for grid {grid_id} "
        f"from {start_year}-{start_month:02d}-{start_day:02d} "
        f"to {duration_years} years, {duration_months} months, "
        f"{duration_days} days later using {mosaic_method} method "
        f"with bands {required_bands}."
    )

    validate_inputs(
        sort_method=sort_method,
        mosaic_method=mosaic_method,
        no_data_threshold=no_data_threshold,
        required_bands=required_bands,
        grid_id=grid_id,
        percentile_value=percentile_value,
        resampling_method=resampling_method,
        cloud_mask=cloud_mask,
    )
    logger.info("All inputs validated successfully.")

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

    assert grid_id is not None  # narrowing: bounds-mode dispatch happened above
    grid_polygon = get_extent_from_grid_id(grid_id)

    logger.info(
        f"Searching for scenes in grid {grid_id} within bounds {grid_polygon} "
        f"from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}."
    )
    items = search_for_items(
        bounds=grid_polygon.buffer(-0.05),
        grid_id=grid_id,
        start_date=start_date,
        end_date=end_date,
        additional_query=additional_query,
        ignore_duplicate_items=ignore_duplicate_items,
    )
    logger.info(f"Found {len(items)} scenes for grid {grid_id}.")
    if len(items) == 0:
        raise ValueError(
            f"No scenes found for {grid_id} between {start_date.strftime('%Y-%m-%d')} "
            f"and {end_date.strftime('%Y-%m-%d')}"
        )

    target_size = int(round(MGRS_TILE_SIZE_M / resolution))

    # for scenes with only partial S2 coverage work out which pixels are covered
    if coverage_threshold_pct is None:
        coverage_mask = np.ones((target_size, target_size), dtype=bool)
    else:
        coverage_mask = get_frequent_coverage(
            scene_bounds=grid_polygon,
            scenes=items,
            coverage_threshold_pct=coverage_threshold_pct,
            resolution=resolution,
        )

    items_with_orbits = add_item_info(items)

    if not sort_function:
        sorted_items = sort_items(items=items_with_orbits, sort_method=sort_method)
    else:
        sorted_items = sort_function(items=items_with_orbits)

    logger.info(f"Sorted {len(sorted_items)} scenes using {sort_method} method.")

    mosaic, profile = download_bands_pool(
        sorted_scenes=sorted_items,
        required_bands=required_bands,
        no_data_threshold=no_data_threshold,
        mosaic_method=mosaic_method,
        ocm_batch_size=ocm_batch_size,
        ocm_inference_dtype=ocm_inference_dtype,
        coverage_mask=coverage_mask,
        percentile_value=percentile_value,
        s2_scene_size=target_size,
        resampling_method=resampling_method,
        resolution=resolution,
        cloud_mask=cloud_mask,
    )
    return finalize_output(
        array=mosaic,
        profile=profile,
        required_bands=required_bands,
        coverage_mask=coverage_mask if coverage_threshold_pct is not None else None,
        export_path=export_path if output_dir else None,
    )
