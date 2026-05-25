"""Output path resolution and GeoTIFF finalisation."""

import hashlib
import json
import logging
import re
from dataclasses import fields, is_dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import rasterio as rio
from shapely.geometry.base import BaseGeometry
from shapely.geometry import mapping

logger = logging.getLogger(__name__)
MOSAIC_PERCENTILE = "percentile"
REQUEST_HASH_EXCLUDED_FIELDS = {
    "output_dir",
    "output_path",
    "overwrite",
    "show_progress",
    "tile_workers",
}


def _jsonable(value: Any, _seen: Optional[set[int]] = None) -> Any:
    if _seen is None:
        _seen = set()
    value_id = id(value)
    is_container = isinstance(value, (Mapping, tuple, list, BaseGeometry))
    if is_container:
        if value_id in _seen:
            raise ValueError("Cannot serialise recursive request metadata")
        _seen.add(value_id)
    try:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, BaseGeometry):
            return mapping(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Mapping):
            return {str(k): _jsonable(v, _seen) for k, v in sorted(value.items())}
        if isinstance(value, tuple):
            return [_jsonable(v, _seen) for v in value]
        if isinstance(value, list):
            return [_jsonable(v, _seen) for v in value]
        if callable(value):
            name = getattr(value, "__qualname__", repr(value))
            module = getattr(value, "__module__", None)
            code = getattr(value, "__code__", None)
            code_hash = ""
            if code is not None:
                code_hash = hashlib.sha256(code.co_code).hexdigest()[:10]
            qualname = f"{module}.{name}" if module else name
            return {"callable": qualname, "code_hash": code_hash}
        return value
    finally:
        if is_container:
            _seen.remove(value_id)


def _request_metadata(request: Any) -> Dict[str, Any]:
    if not is_dataclass(request):
        raise TypeError("request must be a dataclass instance")
    return {
        field.name: _jsonable(getattr(request, field.name)) for field in fields(request)
    }


def output_request_hash(
    request: Any,
    *,
    mode: str,
    start_date: date,
    end_date: date,
    source_name: str,
    target_crs: Optional[int] = None,
    bounds_4326: Optional[Tuple[float, float, float, float]] = None,
) -> str:
    """Stable short hash for output-affecting request parameters."""
    metadata = {
        field.name: _jsonable(getattr(request, field.name))
        for field in fields(request)
        if field.name not in REQUEST_HASH_EXCLUDED_FIELDS
    }
    metadata.update(
        {
            "mode": mode,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "source_name": source_name,
            "target_crs": target_crs,
            "bounds_4326": bounds_4326,
        }
    )
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:10]


def _hash_value(value: Any, length: int = 10) -> str:
    encoded = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()[:length]


def _safe_token(value: Union[float, int, str]) -> str:
    token = f"{value:.4f}" if isinstance(value, float) else str(value)
    token = token.strip()
    sign = "neg" if token.startswith("-") else ""
    if sign:
        token = token[1:]
    token = token.replace(".", "p")
    token = re.sub(r"[^A-Za-z0-9p]+", "_", token).strip("_")
    return f"{sign}{token}" if sign else token


def _method_token(mosaic_method: str, percentile: Optional[float]) -> str:
    if mosaic_method == MOSAIC_PERCENTILE and percentile is not None:
        percentile_str = f"{percentile:g}".replace(".", "p")
        return f"{mosaic_method}-p{percentile_str}"
    return mosaic_method


def output_sidecar_metadata(
    request: Any,
    *,
    mode: str,
    filename_hash: str,
    start_date: date,
    end_date: date,
    source_name: str,
    target_crs: Optional[int] = None,
    bounds_4326: Optional[Tuple[float, float, float, float]] = None,
) -> Dict[str, Any]:
    """JSON-serialisable metadata describing an exported GeoTIFF request."""
    return {
        "filename_hash": filename_hash,
        "mode": mode,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "source_name": source_name,
        "target_crs": target_crs,
        "bounds_4326": _jsonable(bounds_4326),
        "request": _request_metadata(request),
    }


def write_output_sidecar(export_path: Path, metadata: Dict[str, Any]) -> None:
    """Write a JSON metadata sidecar beside an exported GeoTIFF."""
    sidecar_path = export_path.with_suffix(".json")
    sidecar_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def get_output_path(
    output_dir: Union[Path, str],
    start_date: date,
    end_date: date,
    scene_order: str,
    mosaic_method: str,
    bands: List[str],
    percentile: Optional[float] = None,
    grid_id: Optional[str] = None,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    aoi: Optional[BaseGeometry] = None,
    source_name: Optional[str] = None,
    resolution: Optional[int] = None,
    cloud_mask: Optional[str] = None,
    filename_hash: Optional[str] = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    bands_str = "-".join(bands)
    method_str = _method_token(mosaic_method, percentile)

    if grid_id is not None:
        prefix = f"grid-{grid_id}"
    elif aoi is not None:
        prefix = f"aoi-{_hash_value(aoi)}"
    elif bounds is not None:
        coords = "_".join(_safe_token(coord) for coord in bounds)
        prefix = f"bbox-{coords}"
    else:
        raise ValueError("Either grid_id, bounds, or aoi is required")

    detail_tokens = [bands_str, method_str, f"scene-{scene_order}"]
    if resolution is not None:
        detail_tokens.append(f"{resolution}m")
    if cloud_mask is not None:
        detail_tokens.append(cloud_mask)
    if source_name is not None:
        detail_tokens.append(source_name)
    if filename_hash is not None:
        detail_tokens.append(filename_hash)
    details = "_".join(detail_tokens)

    return output_dir / (
        f"{prefix}_{start_date.strftime('%Y-%m-%d')}_to_"
        f"{end_date.strftime('%Y-%m-%d')}_{details}.tif"
    )


def resolve_export_path(
    output_dir: Optional[Union[Path, str]],
    output_path: Optional[Union[Path, str]],
    start_date: date,
    end_date: date,
    scene_order: str,
    mosaic_method: str,
    bands: List[str],
    percentile: Optional[float] = None,
    grid_id: Optional[str] = None,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    aoi: Optional[BaseGeometry] = None,
    source_name: Optional[str] = None,
    resolution: Optional[int] = None,
    cloud_mask: Optional[str] = None,
    filename_hash: Optional[str] = None,
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
        scene_order=scene_order,
        mosaic_method=mosaic_method,
        bands=bands,
        percentile=percentile,
        grid_id=grid_id,
        bounds=bounds,
        aoi=aoi,
        source_name=source_name,
        resolution=resolution,
        cloud_mask=cloud_mask,
        filename_hash=filename_hash,
    )


def _export_tif(
    array: npt.NDArray[Any],
    profile: Dict[str, Any],
    export_path: Path,
    bands: List[str],
    nodata_value: Union[int, None] = 0,
) -> None:
    write_profile = profile.copy()
    write_profile.update(
        count=array.shape[0], dtype=array.dtype, nodata=nodata_value, compress="lzw"
    )
    with rio.open(export_path, "w", **write_profile) as dst:
        dst.write(array)
        dst.descriptions = bands


def output_band_metadata(
    bands: List[str],
) -> Tuple[List[str], Optional[int]]:
    """Return output band descriptions and nodata value for requested bands."""
    if "visual" in bands:
        return ["Red", "Green", "Blue"], None
    return list(bands), 0


def finalize_output(
    array: npt.NDArray[Any],
    profile: Dict[str, Any],
    bands: List[str],
    coverage_mask: Optional[npt.NDArray[Any]],
    export_path: Optional[Path],
) -> Union[Tuple[npt.NDArray[Any], Dict[str, Any]], Path]:
    """Apply coverage mask, set band names + nodata, export or return."""
    if coverage_mask is not None:
        coverage = np.asarray(coverage_mask, dtype=bool)
        np.multiply(array, coverage[None, :, :], out=array)

    band_descriptions, nodata_value = output_band_metadata(bands)

    if export_path is not None:
        logger.info(f"Writing GeoTIFF to {export_path}")
        _export_tif(
            array=array,
            profile=profile,
            export_path=export_path,
            bands=band_descriptions,
            nodata_value=nodata_value,
        )
        return export_path

    logger.info(f"Returning array shape={array.shape} dtype={array.dtype}")
    return array, profile
