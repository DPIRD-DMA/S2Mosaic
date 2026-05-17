import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, overload

import numpy as np
import numpy.typing as npt

from .bounds import Aoi, Bbox, run_bounds_pipeline
from .frequent_coverage import get_frequent_coverage
from .helpers import (
    MGRS_TILE_SIZE_M,
    define_dates,
    finalize_output,
    get_extent_from_grid_id,
    normalize_mosaic_inputs,
    resolve_export_path,
    validate_inputs,
)
from .mosaic_core import stream_mosaic_pipeline
from .stac_utils import add_item_info, search_for_items, sort_items

logger = logging.getLogger(__name__)


@overload
def mosaic(
    *,
    grid_id: Optional[str] = ...,
    bounds: Optional[Bbox] = ...,
    aoi: Optional[Aoi] = ...,
    input_crs: int = ...,
    start_year: int,
    start_month: int = ...,
    start_day: int = ...,
    duration_years: int = ...,
    duration_months: int = ...,
    duration_days: int = ...,
    required_bands: Optional[List[str]] = ...,
    mosaic_method: str = ...,
    percentile_value: Optional[float] = ...,
    output_dir: None = None,
    output_path: None = None,
    overwrite: bool = ...,
    output_crs: Optional[int] = ...,
    resolution: int = ...,
    resampling_method: str = ...,
    additional_query: Optional[Dict[str, Any]] = ...,
    no_data_threshold: Union[float, None] = ...,
    observation_target: Optional[int] = ...,
    coverage_threshold: Optional[float] = ...,
    ignore_duplicate_items: bool = ...,
    sort_method: str = ...,
    sort_function: Optional[Callable[..., Any]] = ...,
    cloud_mask: str = ...,
    ocm_batch_size: int = ...,
    ocm_inference_dtype: str = ...,
    tile_workers: Optional[int] = ...,
    adaptive_tiling: bool = ...,
    show_progress: bool = ...,
) -> Tuple[npt.NDArray[Any], Dict[str, Any]]: ...


@overload
def mosaic(
    *,
    grid_id: Optional[str] = ...,
    bounds: Optional[Bbox] = ...,
    aoi: Optional[Aoi] = ...,
    input_crs: int = ...,
    start_year: int,
    start_month: int = ...,
    start_day: int = ...,
    duration_years: int = ...,
    duration_months: int = ...,
    duration_days: int = ...,
    required_bands: Optional[List[str]] = ...,
    mosaic_method: str = ...,
    percentile_value: Optional[float] = ...,
    output_dir: Union[Path, str],
    output_path: None = None,
    overwrite: bool = ...,
    output_crs: Optional[int] = ...,
    resolution: int = ...,
    resampling_method: str = ...,
    additional_query: Optional[Dict[str, Any]] = ...,
    no_data_threshold: Union[float, None] = ...,
    observation_target: Optional[int] = ...,
    coverage_threshold: Optional[float] = ...,
    ignore_duplicate_items: bool = ...,
    sort_method: str = ...,
    sort_function: Optional[Callable[..., Any]] = ...,
    cloud_mask: str = ...,
    ocm_batch_size: int = ...,
    ocm_inference_dtype: str = ...,
    tile_workers: Optional[int] = ...,
    adaptive_tiling: bool = ...,
    show_progress: bool = ...,
) -> Path: ...


@overload
def mosaic(
    *,
    grid_id: Optional[str] = ...,
    bounds: Optional[Bbox] = ...,
    aoi: Optional[Aoi] = ...,
    input_crs: int = ...,
    start_year: int,
    start_month: int = ...,
    start_day: int = ...,
    duration_years: int = ...,
    duration_months: int = ...,
    duration_days: int = ...,
    required_bands: Optional[List[str]] = ...,
    mosaic_method: str = ...,
    percentile_value: Optional[float] = ...,
    output_dir: None = None,
    output_path: Union[Path, str],
    overwrite: bool = ...,
    output_crs: Optional[int] = ...,
    resolution: int = ...,
    resampling_method: str = ...,
    additional_query: Optional[Dict[str, Any]] = ...,
    no_data_threshold: Union[float, None] = ...,
    observation_target: Optional[int] = ...,
    coverage_threshold: Optional[float] = ...,
    ignore_duplicate_items: bool = ...,
    sort_method: str = ...,
    sort_function: Optional[Callable[..., Any]] = ...,
    cloud_mask: str = ...,
    ocm_batch_size: int = ...,
    ocm_inference_dtype: str = ...,
    tile_workers: Optional[int] = ...,
    adaptive_tiling: bool = ...,
    show_progress: bool = ...,
) -> Path: ...


def mosaic(
    *,
    grid_id: Optional[str] = None,
    bounds: Optional[Bbox] = None,
    aoi: Optional[Aoi] = None,
    input_crs: int = 4326,
    start_year: int,
    start_month: int = 1,
    start_day: int = 1,
    duration_years: int = 0,
    duration_months: int = 0,
    duration_days: int = 0,
    required_bands: Optional[List[str]] = None,
    mosaic_method: str = "mean",
    percentile_value: Optional[float] = None,
    output_dir: Optional[Union[Path, str]] = None,
    output_path: Optional[Union[Path, str]] = None,
    overwrite: bool = True,
    output_crs: Optional[int] = None,
    resolution: int = 10,
    resampling_method: str = "nearest",
    additional_query: Optional[Dict[str, Any]] = None,
    no_data_threshold: Union[float, None] = 0.01,
    observation_target: Optional[int] = None,
    coverage_threshold: Optional[float] = 0.1,
    ignore_duplicate_items: bool = True,
    sort_method: str = "valid_data",
    sort_function: Optional[Callable[..., Any]] = None,
    cloud_mask: str = "OCM",
    ocm_batch_size: int = 1,
    ocm_inference_dtype: str = "bf16",
    tile_workers: Optional[int] = None,
    adaptive_tiling: bool = True,
    show_progress: bool = False,
) -> Union[Tuple[npt.NDArray[Any], Dict[str, Any]], Path]:
    """
    Create a Sentinel-2 mosaic.

    Three modes — pass exactly one of:
        * ``grid_id`` (e.g. "50HMH"): mosaic an entire MGRS tile.
        * ``bounds`` (minx, miny, maxx, maxy): mosaic an arbitrary bounding
          box. Scenes from any intersecting MGRS tiles are streamed through
          per-scene rasterio WarpedVRT reads and aggregated on a common UTM
          grid in ``output_crs``.
        * ``aoi``: mosaic a single polygon AOI. The output raster uses the
          polygon bounds, with pixels outside the polygon written as nodata.

    ``input_crs`` and ``output_crs`` only apply when ``bounds`` or ``aoi`` is set.

    Args:
        grid_id (str, optional): MGRS tile ID (e.g. "50HMH"). Mutually exclusive with bounds.
        bounds (Tuple[float, float, float, float], optional): Arbitrary AOI
            rectangle as ``(minx, miny, maxx, maxy)`` in ``input_crs``.
            Mutually exclusive with ``grid_id``.
        aoi (Polygon, optional): Single polygon AOI in ``input_crs``.
            Mutually exclusive with ``grid_id`` and ``bounds``.
        input_crs (int, optional): EPSG code of ``bounds`` or ``aoi``.
            Defaults to 4326. Only used in bounds/AOI mode.
        start_year (int): The start year of the time range.
        start_month (int, optional): The start month of the time range. Defaults to 1 (January).
        start_day (int, optional): The start day of the time range. Defaults to 1.
        duration_years (int, optional): Duration in years to add to the start date. Defaults to 0.
        duration_months (int, optional): Duration in months to add to the start date. Defaults to 0.
        duration_days (int, optional): Duration in days to add to the start date. Defaults to 0.
        required_bands (List[str], optional): List of required spectral bands.
            Defaults to ["B04", "B03", "B02", "B08"] (Red, Green, Blue, NIR).
        mosaic_method (str, optional): Method to create the mosaic. Options are "mean", "first", "median" or "percentile". Defaults to "mean".
        percentile_value (Optional[float], optional): Percentile to calculate
            when using ``mosaic_method="percentile"``. Must be between 0 and 100.
        output_dir (Optional[Union[Path, str]], optional): Directory to save
            the output GeoTIFF using an auto-generated filename. Mutually
            exclusive with ``output_path``. If neither is provided, the mosaic
            is returned instead. Defaults to None.
        output_path (Optional[Union[Path, str]], optional): Full GeoTIFF path
            to write, including the filename. Mutually exclusive with
            ``output_dir``. Defaults to None.
        overwrite (bool, optional): Whether to overwrite existing output files. Defaults to True.
        output_crs (int, optional): EPSG code for the output grid. In bounds
            mode, defaults to the UTM zone containing the AOI centroid. Ignored
            in grid mode.
        resolution (int, optional): Output pixel size in metres. Defaults to 10.
        resampling_method (str, optional): Rasterio resampling method used when
            reading source COGs onto the output grid. Options include "nearest",
            "bilinear", "cubic", "average", and "lanczos". Defaults to "nearest".
        additional_query (Dict[str, Any], optional): Additional query parameters for STAC API.
            Defaults to {"eo:cloud_cover": {"lt": 100}}.
        no_data_threshold (float, optional): Threshold for no data values. Defaults to 0.01.
        observation_target (int, optional): Per-tile early-stop target
            for ``mean`` and ``percentile``. When set, aggregation stops
            reading later scenes for a tile once every coverable pixel has at
            least this many valid observations. This is not an output quality
            filter. Defaults to None.
        coverage_threshold (float, optional): Drop pixels covered by fewer
            than this fraction of overlapping scenes. Set to None to disable.
            Defaults to 0.1.
        ignore_duplicate_items (bool, optional): Whether to remove duplicate scenes based on their IDs. Defaults to True.
        sort_method (str, optional): Method to sort scenes. Options are "valid_data", "oldest", or "newest". Defaults to "valid_data".
        sort_function (Callable, optional): Custom sorting function. If provided, overrides sort_method.
        cloud_mask (str, optional): Cloud-mask provider. ``"OCM"`` (default) runs the
            OmniCloudMask deep-learning model on R+G+NIR; ``"SCL"`` reads the L2A
            Scene Classification Layer published with the scene. SCL is much cheaper
            (one COG read, no inference) but lower accuracy.
        ocm_batch_size (int, optional): Batch size for OCM inference. Defaults to 1.
        ocm_inference_dtype (str, optional): Data type for OCM inference. Defaults to "bf16".
        tile_workers (int, optional): Number of output tiles to aggregate
            concurrently. Higher values can improve throughput for
            network-bound reads, but increase memory use and simultaneous
            source reads. Defaults to ``min(4, os.cpu_count() or 1)``.
        adaptive_tiling (bool, optional): Split sparse output tiles based on
            the actual cloud-valid contribution masks. Reduces wasted reads for
            irregular AOIs and sparse scene coverage. Defaults to True.
        show_progress (bool, optional): Show tqdm progress bars for the
            cloud-mask and tile-aggregation phases. Defaults to False.

    Returns:
        Union[Tuple[npt.NDArray[Any], Dict[str, Any]], Path]: If no export path
        is requested, returns the mosaic array and metadata dictionary.
        Otherwise returns the path to the saved GeoTIFF file.

    Raises:
        ValueError: If inputs fail validation, or if no scenes are found for the
            specified grid_id / bounds and time range.
        RuntimeError: If scenes were found but all were fully cloud-masked or
            invalid, or if every scene failed to fetch after retries.

    Note:
        - The function uses the STAC API to search for Sentinel-2 scenes.
        - If 'visual' is included in required_bands, it will be replaced with 'Red', 'Green', 'Blue' in the output.
        - The time range for scene selection is inclusive of the start date and exclusive of the end date.
    """  # noqa: E501
    if sum(x is not None for x in (grid_id, bounds, aoi)) != 1:
        raise ValueError("Exactly one of grid_id, bounds, or aoi must be provided")

    if bounds is not None or aoi is not None:
        return run_bounds_pipeline(
            bounds=bounds,
            aoi=aoi,
            input_crs=input_crs,
            start_year=start_year,
            start_month=start_month,
            start_day=start_day,
            duration_years=duration_years,
            duration_months=duration_months,
            duration_days=duration_days,
            required_bands=required_bands,
            mosaic_method=mosaic_method,
            percentile_value=percentile_value,
            output_dir=output_dir,
            output_path=output_path,
            overwrite=overwrite,
            output_crs=output_crs,
            resolution=resolution,
            resampling_method=resampling_method,
            additional_query=additional_query,
            no_data_threshold=no_data_threshold,
            observation_target=observation_target,
            coverage_threshold=coverage_threshold,
            ignore_duplicate_items=ignore_duplicate_items,
            sort_method=sort_method,
            sort_function=sort_function,
            cloud_mask=cloud_mask,
            ocm_batch_size=ocm_batch_size,
            ocm_inference_dtype=ocm_inference_dtype,
            tile_workers=tile_workers,
            adaptive_tiling=adaptive_tiling,
            show_progress=show_progress,
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
        observation_target=observation_target,
        tile_workers=tile_workers,
        required_bands=required_bands,
        grid_id=grid_id,
        percentile_value=percentile_value,
        resampling_method=resampling_method,
        cloud_mask=cloud_mask,
        adaptive_tiling=adaptive_tiling,
        coverage_threshold=coverage_threshold,
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
    export_path = resolve_export_path(
        output_dir=output_dir,
        output_path=output_path,
        grid_id=grid_id,
        start_date=start_date,
        end_date=end_date,
        sort_method=sort_method,
        mosaic_method=mosaic_method,
        required_bands=required_bands,
        percentile_value=percentile_value,
    )
    if export_path is not None:
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
    coverage_mask: npt.NDArray[np.bool_]
    if coverage_threshold is None:
        coverage_mask = np.ones((target_size, target_size), dtype=bool)
    else:
        coverage_mask = get_frequent_coverage(
            scene_bounds=grid_polygon,
            scenes=items,
            coverage_threshold=coverage_threshold,
            resolution=resolution,
        )

    items_with_orbits = add_item_info(items)

    if not sort_function:
        sorted_items = sort_items(items=items_with_orbits, sort_method=sort_method)
    else:
        sorted_items = sort_function(items=items_with_orbits)

    logger.info(f"Sorted {len(sorted_items)} scenes using {sort_method} method.")

    output_coverage_mask = coverage_mask if coverage_threshold is not None else None
    mosaic, profile = stream_mosaic_pipeline(
        sorted_scenes=sorted_items,
        required_bands=required_bands,
        no_data_threshold=no_data_threshold,
        observation_target=observation_target,
        export_path=export_path,
        output_coverage_mask=output_coverage_mask,
        mosaic_method=mosaic_method,
        ocm_batch_size=ocm_batch_size,
        ocm_inference_dtype=ocm_inference_dtype,
        coverage_mask=coverage_mask,
        percentile_value=percentile_value,
        s2_scene_size=target_size,
        resampling_method=resampling_method,
        resolution=resolution,
        cloud_mask=cloud_mask,
        tile_workers=tile_workers,
        adaptive_tiling=adaptive_tiling,
        show_progress=show_progress,
    )
    if export_path is not None:
        return export_path
    assert mosaic is not None
    return finalize_output(
        array=mosaic,
        profile=profile,
        required_bands=required_bands,
        coverage_mask=output_coverage_mask,
        export_path=export_path,
    )
