"""Output path resolution and GeoTIFF finalisation."""

import logging
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import rasterio as rio

logger = logging.getLogger(__name__)
MOSAIC_PERCENTILE = "percentile"


def get_output_path(
    output_dir: Union[Path, str],
    start_date: date,
    end_date: date,
    sort_method: str,
    mosaic_method: str,
    required_bands: List[str],
    percentile: Optional[float] = None,
    grid_id: Optional[str] = None,
    bounds: Optional[Tuple[float, float, float, float]] = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    bands_str = "_".join(required_bands)
    method_str = mosaic_method
    if mosaic_method == MOSAIC_PERCENTILE and percentile is not None:
        percentile_str = f"{percentile:g}".replace(".", "p")
        method_str = f"{mosaic_method}_p{percentile_str}"

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
        f"{end_date.strftime('%Y-%m-%d')}_{sort_method}_{method_str}_"
        f"{bands_str}.tif"
    )


def resolve_export_path(
    output_dir: Optional[Union[Path, str]],
    output_path: Optional[Union[Path, str]],
    start_date: date,
    end_date: date,
    sort_method: str,
    mosaic_method: str,
    required_bands: List[str],
    percentile: Optional[float] = None,
    grid_id: Optional[str] = None,
    bounds: Optional[Tuple[float, float, float, float]] = None,
) -> Optional[Path]:
    """Resolve explicit or auto-generated GeoTIFF export path."""
    if output_dir is not None and output_path is not None:
        raise ValueError("Only one of output_dir or output_path can be provided")
    if output_path is not None:
        path = Path(output_path)
        if path.suffix.lower() not in {".tif", ".tiff"}:
            raise ValueError("output_path must include a .tif or .tiff filename")
        path.parent.mkdir(exist_ok=True, parents=True)
        return path
    if output_dir is None:
        return None
    return get_output_path(
        output_dir=output_dir,
        start_date=start_date,
        end_date=end_date,
        sort_method=sort_method,
        mosaic_method=mosaic_method,
        required_bands=required_bands,
        percentile=percentile,
        grid_id=grid_id,
        bounds=bounds,
    )


def _export_tif(
    array: npt.NDArray[Any],
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


def output_band_metadata(
    required_bands: List[str],
) -> Tuple[List[str], Optional[int]]:
    """Return output band descriptions and nodata value for requested bands."""
    if "visual" in required_bands:
        return ["Red", "Green", "Blue"], None
    return list(required_bands), 0


def finalize_output(
    array: npt.NDArray[Any],
    profile: Dict[str, Any],
    required_bands: List[str],
    coverage_mask: Optional[npt.NDArray[Any]],
    export_path: Optional[Path],
) -> Union[Tuple[npt.NDArray[Any], Dict[str, Any]], Path]:
    """Apply coverage mask, set band names + nodata, export or return."""
    if coverage_mask is not None:
        np.multiply(array, coverage_mask[None, :, :], out=array, casting="unsafe")

    band_descriptions, nodata_value = output_band_metadata(required_bands)

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
