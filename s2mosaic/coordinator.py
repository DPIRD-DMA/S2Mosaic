import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, overload

import numpy.typing as npt

from .config import MosaicRequest
from .gdal_env import apply_gdal_network_defaults
from .geometry import Aoi, Bbox
from .pipelines.bounds import run_bounds_pipeline
from .pipelines.grid import run_grid_pipeline
from .sources import SOURCE_MPC, get_source

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
    bands: Optional[List[str]] = ...,
    mosaic_method: str = ...,
    percentile: Optional[float] = ...,
    output_dir: None = None,
    output_path: None = None,
    overwrite: bool = ...,
    output_crs: Optional[int] = ...,
    resolution: int = ...,
    resampling_method: str = ...,
    additional_query: Optional[Dict[str, Any]] = ...,
    source: str = ...,
    min_observations: Optional[int] = ...,
    max_observations: Optional[int] = ...,
    min_coverage_fraction: Optional[float] = ...,
    ignore_duplicate_items: bool = ...,
    scene_order: str = ...,
    scene_sort_fn: Optional[Callable[..., Any]] = ...,
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
    bands: Optional[List[str]] = ...,
    mosaic_method: str = ...,
    percentile: Optional[float] = ...,
    output_dir: Union[Path, str],
    output_path: None = None,
    overwrite: bool = ...,
    output_crs: Optional[int] = ...,
    resolution: int = ...,
    resampling_method: str = ...,
    additional_query: Optional[Dict[str, Any]] = ...,
    source: str = ...,
    min_observations: Optional[int] = ...,
    max_observations: Optional[int] = ...,
    min_coverage_fraction: Optional[float] = ...,
    ignore_duplicate_items: bool = ...,
    scene_order: str = ...,
    scene_sort_fn: Optional[Callable[..., Any]] = ...,
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
    bands: Optional[List[str]] = ...,
    mosaic_method: str = ...,
    percentile: Optional[float] = ...,
    output_dir: None = None,
    output_path: Union[Path, str],
    overwrite: bool = ...,
    output_crs: Optional[int] = ...,
    resolution: int = ...,
    resampling_method: str = ...,
    additional_query: Optional[Dict[str, Any]] = ...,
    source: str = ...,
    min_observations: Optional[int] = ...,
    max_observations: Optional[int] = ...,
    min_coverage_fraction: Optional[float] = ...,
    ignore_duplicate_items: bool = ...,
    scene_order: str = ...,
    scene_sort_fn: Optional[Callable[..., Any]] = ...,
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
    bands: Optional[List[str]] = None,
    mosaic_method: str = "mean",
    percentile: Optional[float] = None,
    output_dir: Optional[Union[Path, str]] = None,
    output_path: Optional[Union[Path, str]] = None,
    overwrite: bool = True,
    output_crs: Optional[int] = None,
    resolution: int = 10,
    resampling_method: str = "nearest",
    additional_query: Optional[Dict[str, Any]] = None,
    source: str = SOURCE_MPC,
    min_observations: Optional[int] = None,
    max_observations: Optional[int] = None,
    min_coverage_fraction: Optional[float] = None,
    ignore_duplicate_items: bool = True,
    scene_order: str = "valid_data",
    scene_sort_fn: Optional[Callable[..., Any]] = None,
    cloud_mask: str = "OCM",
    ocm_batch_size: int = 1,
    ocm_inference_dtype: str = "fp16",
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
        bands (List[str], optional): List of spectral bands.
            Defaults to ["B04", "B03", "B02", "B08"] (Red, Green, Blue, NIR).
        mosaic_method (str, optional): Method to create the mosaic. Options are "mean", "first", "median" or "percentile". Defaults to "mean".
        percentile (Optional[float], optional): Percentile to calculate
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
        source (str, optional): STAC imagery source. ``"MPC"`` (default) uses
            Microsoft Planetary Computer (SAS-signed URLs); ``"AWS"`` uses
            Element 84 Earth Search on AWS Open Data (public S3, no auth).
            Defaults to "MPC".
        min_observations (int, optional): Per-tile early-stop target
            for ``mean`` and ``percentile``. When set, aggregation stops
            reading later scenes for a tile once every coverable pixel has at
            least this many valid observations. This is not an output quality
            filter. Defaults to None.
        max_observations (int, optional): Per-pixel cap for ``mean`` and
            ``percentile``. Each pixel accepts at most this many valid scenes,
            in ``scene_order``; later valid scenes are dropped for that pixel.
            Combined with ``scene_order="oldest"``/``"newest"`` this biases
            the mosaic to early or late dates. Must be >= ``min_observations``
            when both are set. Ignored by ``first`` (effectively N=1).
            Defaults to None.
        min_coverage_fraction (float, optional): Drop pixels covered by fewer
            than this fraction of overlapping scenes. Set to None to disable.
            Defaults to None.
        ignore_duplicate_items (bool, optional): Whether to remove duplicate scenes based on their IDs. Defaults to True.
        scene_order (str, optional): Scene ordering. Options are "valid_data", "oldest", or "newest". Defaults to "valid_data".
        scene_sort_fn (Callable, optional): Custom sorting function. If provided, overrides scene_order.
        cloud_mask (str, optional): Cloud-mask provider. ``"OCM"`` (default) runs the
            OmniCloudMask deep-learning model on R+G+NIR; ``"SCL"`` reads the L2A
            Scene Classification Layer published with the scene. SCL is much cheaper
            (one COG read, no inference) but lower accuracy.
        ocm_batch_size (int, optional): Batch size for OCM inference. Defaults to 1.
        ocm_inference_dtype (str, optional): Data type for OCM inference.
            Defaults to "fp16" for broad GPU/MPS compatibility. Use "fp32" for
            CPU inference (most CPUs lack efficient fp16/bf16 paths), or "bf16"
            on devices that support it.
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
        - If 'visual' is included in bands, it will be replaced with 'Red', 'Green', 'Blue' in the output.
        - The time range for scene selection is inclusive of the start date and exclusive of the end date.
    """  # noqa: E501
    apply_gdal_network_defaults()

    request = MosaicRequest(
        grid_id=grid_id,
        bounds=bounds,
        aoi=aoi,
        input_crs=input_crs,
        start_year=start_year,
        start_month=start_month,
        start_day=start_day,
        duration_years=duration_years,
        duration_months=duration_months,
        duration_days=duration_days,
        bands=bands,
        mosaic_method=mosaic_method,
        percentile=percentile,
        output_dir=output_dir,
        output_path=output_path,
        overwrite=overwrite,
        output_crs=output_crs,
        resolution=resolution,
        resampling_method=resampling_method,
        additional_query=additional_query,
        source=source,
        min_observations=min_observations,
        max_observations=max_observations,
        min_coverage_fraction=min_coverage_fraction,
        ignore_duplicate_items=ignore_duplicate_items,
        scene_order=scene_order,
        scene_sort_fn=scene_sort_fn,
        cloud_mask=cloud_mask,
        ocm_batch_size=ocm_batch_size,
        ocm_inference_dtype=ocm_inference_dtype,
        tile_workers=tile_workers,
        adaptive_tiling=adaptive_tiling,
        show_progress=show_progress,
    ).normalized()
    request.validate()

    source_obj = get_source(request.source)

    if request.bounds is not None or request.aoi is not None:
        return run_bounds_pipeline(request, source=source_obj)
    return run_grid_pipeline(request, source=source_obj)
