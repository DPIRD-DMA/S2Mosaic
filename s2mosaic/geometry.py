"""Geometry and target-grid helpers for bounds/AOI mosaics."""

from typing import Tuple, TypeAlias, cast

import numpy as np
import numpy.typing as npt
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely.geometry import Polygon
from shapely.ops import transform as shapely_transform

Bbox = Tuple[float, float, float, float]
Aoi: TypeAlias = Polygon


def pick_utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the UTM zone containing (lon, lat)."""
    zone = min(60, max(1, int((lon + 180) / 6) + 1))
    return (32700 if lat < 0 else 32600) + zone


def reproject_bbox(bbox: Bbox, src_epsg: int, dst_epsg: int) -> Bbox:
    """Reproject (minx, miny, maxx, maxy) between CRSes."""
    if src_epsg == dst_epsg:
        return bbox
    transformer = Transformer.from_crs(src_epsg, dst_epsg, always_xy=True)
    minx, miny, maxx, maxy = bbox
    xs, ys = transformer.transform([minx, maxx, minx, maxx], [miny, miny, maxy, maxy])
    return (min(xs), min(ys), max(xs), max(ys))


def reproject_aoi(aoi: Aoi, src_epsg: int, dst_epsg: int) -> Aoi:
    """Reproject a polygon AOI between CRSes."""
    if src_epsg == dst_epsg:
        return aoi
    transformer = Transformer.from_crs(src_epsg, dst_epsg, always_xy=True)
    reprojected = shapely_transform(transformer.transform, aoi)
    if not isinstance(reprojected, Polygon):
        raise ValueError("aoi must reproject to a single Polygon")
    return cast(Aoi, reprojected)


_OCM_BANDS: Tuple[str, str, str] = ("B04", "B03", "B8A")
_OCM_MIN_CONTEXT_PIXELS = 100
_SCL_ADAPTIVE_BLOCK_SAVING_FRACTION = 0.75


def _target_grid(
    bounds_target: Bbox, resolution: int, target_crs: int
) -> Tuple[Affine, int, int, CRS]:
    """Pixel grid + CRS for ``bounds_target`` at ``resolution`` in ``target_crs``."""
    minx, miny, maxx, maxy = bounds_target
    width, height = _grid_shape_for_bounds(bounds_target, resolution)
    transform = Affine(resolution, 0, minx, 0, -resolution, maxy)
    target_crs_obj = CRS.from_epsg(target_crs)
    return transform, width, height, target_crs_obj


def _rasterize_aoi_mask(
    aoi_target: Aoi,
    bounds_target: Bbox,
    resolution: int,
    width: int,
    height: int,
) -> npt.NDArray[np.bool_]:
    """Rasterize a polygon AOI onto a target grid."""
    minx, _, _, maxy = bounds_target
    transform = Affine(resolution, 0, minx, 0, -resolution, maxy)
    mask = rasterize(
        [(aoi_target, 1)],
        out_shape=(height, width),
        fill=0,
        dtype=np.uint8,
        transform=transform,
        all_touched=True,
    )
    return mask.astype(bool)  # type: ignore[no-any-return]


def _grid_shape_for_bounds(bounds_target: Bbox, resolution: int) -> Tuple[int, int]:
    """Return (width, height), keeping tiny valid bounds at least one pixel."""
    minx, miny, maxx, maxy = bounds_target
    width = max(1, int(round((maxx - minx) / resolution)))
    height = max(1, int(round((maxy - miny) / resolution)))
    return width, height


def _expand_bounds_for_ocm_context(
    bounds_target: Bbox, resolution: int, min_pixels: int = _OCM_MIN_CONTEXT_PIXELS
) -> Tuple[Bbox, Tuple[slice, slice]]:
    """Pad OCM reads to at least ``min_pixels`` each way and return AOI crop.

    The expanded bounds stay aligned to the requested mask grid. The returned
    slices crop an expanded OCM prediction back to the originally requested
    bounds at ``resolution``.
    """
    req_w, req_h = _grid_shape_for_bounds(bounds_target, resolution)
    expanded_w = max(req_w, min_pixels)
    expanded_h = max(req_h, min_pixels)

    pad_x = expanded_w - req_w
    pad_y = expanded_h - req_h
    left = pad_x // 2
    right = pad_x - left
    top = pad_y // 2
    bottom = pad_y - top

    minx, miny, maxx, maxy = bounds_target
    expanded_bounds = (
        minx - left * resolution,
        miny - bottom * resolution,
        maxx + right * resolution,
        maxy + top * resolution,
    )
    return expanded_bounds, (slice(top, top + req_h), slice(left, left + req_w))
