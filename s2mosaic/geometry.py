"""Geometry and target-grid helpers for bounds/AOI mosaics."""

from typing import Any, Optional, Tuple, TypeAlias, cast

import numpy as np
import numpy.typing as npt
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely.geometry import MultiPolygon, Polygon, box, shape
from shapely.ops import transform as shapely_transform

from ._types import SceneWindow

Bbox = Tuple[float, float, float, float]
Aoi: TypeAlias = Polygon


def pick_utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the UTM zone containing (lon, lat)."""
    if not -180.0 <= lon <= 180.0:
        raise ValueError(f"longitude must be in [-180, 180], got {lon}")
    if not -90.0 <= lat <= 90.0:
        raise ValueError(f"latitude must be in [-90, 90], got {lat}")
    if not -80.0 <= lat <= 84.0:
        raise ValueError(
            "Automatic output_crs selection only supports UTM latitudes "
            f"[-80, 84], got {lat}; pass output_crs explicitly for polar regions"
        )
    zone = min(60, max(1, int((lon + 180) / 6) + 1))
    return (32700 if lat < 0 else 32600) + zone


def reproject_bbox(bbox: Bbox, src_epsg: int, dst_epsg: int) -> Bbox:
    """Reproject (minx, miny, maxx, maxy) between CRSes."""
    if src_epsg == dst_epsg:
        return bbox
    transformer = Transformer.from_crs(src_epsg, dst_epsg, always_xy=True)
    return transformer.transform_bounds(*bbox, densify_pts=21)


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


def _snap_bounds_to_grid(bounds_target: Bbox, resolution: int) -> Bbox:
    """Expand ``bounds_target`` outward to whole-``resolution`` multiples.

    Anchors the output grid to integer multiples of ``resolution`` in the
    target CRS so repeat runs over the same area produce identical grids and,
    at ``resolution=10``, align with the native Sentinel-2 source grid.
    Always expands (never shrinks) the requested extent.
    """
    minx, miny, maxx, maxy = bounds_target
    minx_snap = np.floor(minx / resolution) * resolution
    miny_snap = np.floor(miny / resolution) * resolution
    maxx_snap = np.ceil(maxx / resolution) * resolution
    maxy_snap = np.ceil(maxy / resolution) * resolution
    return (
        float(minx_snap),
        float(miny_snap),
        float(maxx_snap),
        float(maxy_snap),
    )


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


def _window_from_target_bounds(
    intersection_bounds: Bbox,
    bounds_target: Bbox,
    resolution: int,
) -> Optional[SceneWindow]:
    """Snap a target-CRS bounds to bounds_target's pixel grid as a SceneWindow."""
    minx_t, _, _, maxy_t = bounds_target
    minx, miny, maxx, maxy = intersection_bounds
    if minx >= maxx or miny >= maxy:
        return None
    grid_w, grid_h = _grid_shape_for_bounds(bounds_target, resolution)
    col_off = max(0, int(np.floor((minx - minx_t) / resolution)))
    row_off = max(0, int(np.floor((maxy_t - maxy) / resolution)))
    col_end = min(grid_w, int(np.ceil((maxx - minx_t) / resolution)))
    row_end = min(grid_h, int(np.ceil((maxy_t - miny) / resolution)))
    width = col_end - col_off
    height = row_end - row_off
    if width <= 0 or height <= 0:
        return None
    return col_off, row_off, width, height


def _scene_window_in_target(
    item_bbox_4326: Bbox,
    bounds_target: Bbox,
    target_crs: int,
    resolution: int,
) -> Optional[SceneWindow]:
    """Where in the bounds_target grid a scene's footprint lands.

    Reprojects the scene's lon/lat bbox into ``target_crs``, intersects with
    ``bounds_target``, and snaps to the grid at ``resolution``. Returns
    ``(col_off, row_off, width, height)`` or None if there's no overlap.

    For wide-AOI cases prefer :func:`_scene_window_from_geometry` — the bbox
    here always circumscribes the actual scene footprint (a UTM-aligned tile
    appears as a tilted trapezoid in lon/lat, so its lon/lat bbox is strictly
    larger than the tile). The slack shows up as nodata fed into OCM.
    """
    item_t = reproject_bbox(item_bbox_4326, 4326, target_crs)
    minx_t, miny_t, maxx_t, maxy_t = bounds_target
    minx_i, miny_i, maxx_i, maxy_i = item_t
    intersection = (
        max(minx_t, minx_i),
        max(miny_t, miny_i),
        min(maxx_t, maxx_i),
        min(maxy_t, maxy_i),
    )
    return _window_from_target_bounds(intersection, bounds_target, resolution)


def _scene_window_from_geometry(
    item_geometry: Any,
    bounds_target: Bbox,
    target_crs: int,
    resolution: int,
) -> Optional[SceneWindow]:
    """Tight scene window from the scene's GeoJSON footprint polygon (in 4326).

    Reprojects the polygon vertex-by-vertex into ``target_crs``, intersects
    with ``bounds_target`` as polygons, and takes the intersection's bbox.
    Tighter than :func:`_scene_window_in_target` because the polygon traces
    the actual data footprint rather than its circumscribing rectangle —
    cuts the nodata fed into OCM, especially for cross-UTM-zone scenes.

    Accepts either a GeoJSON-like mapping (``{"type": "Polygon", ...}``) or a
    shapely geometry. Returns None on empty intersection or empty geometry.
    """
    geom = item_geometry if hasattr(item_geometry, "is_empty") else shape(item_geometry)
    if geom.is_empty:
        return None
    if target_crs != 4326:
        transformer = Transformer.from_crs(4326, target_crs, always_xy=True)
        geom = shapely_transform(transformer.transform, geom)
    if not isinstance(geom, (Polygon, MultiPolygon)):
        return None
    intersection = geom.intersection(box(*bounds_target))
    if intersection.is_empty:
        return None
    return _window_from_target_bounds(intersection.bounds, bounds_target, resolution)


def _window_bounds_in_target(
    bounds_target: Bbox,
    resolution: int,
    window: SceneWindow,
) -> Bbox:
    """World-space bbox of a (col_off, row_off, w, h) window in bounds_target."""
    minx_t, _, _, maxy_t = bounds_target
    col_off, row_off, width, height = window
    minx = minx_t + col_off * resolution
    maxy = maxy_t - row_off * resolution
    maxx = minx + width * resolution
    miny = maxy - height * resolution
    return (minx, miny, maxx, maxy)


def _expand_window_for_ocm_context(
    bounds_target: Bbox,
    resolution: int,
    window: SceneWindow,
    min_pixels: int = _OCM_MIN_CONTEXT_PIXELS,
) -> Tuple[SceneWindow, Tuple[slice, slice]]:
    """Pad a scene window centered to satisfy OCM's >=``min_pixels`` context.

    Mirrors :func:`_expand_bounds_for_ocm_context` but for per-scene windows.
    The padded window may extend outside bounds_target's grid (negative
    offsets or beyond the grid edge) — OCM gets context but the crop puts the
    inferred pixels back at the original window position. Returns the padded
    window and the crop slices that undo the padding on the OCM output.
    """
    col_off, row_off, width, height = window
    target_w = max(width, min_pixels)
    target_h = max(height, min_pixels)
    if target_w == width and target_h == height:
        return window, (slice(0, height), slice(0, width))
    pad_x = target_w - width
    pad_y = target_h - height
    left = pad_x // 2
    top = pad_y // 2
    new_col_off = col_off - left
    new_row_off = row_off - top
    crop_col = col_off - new_col_off
    crop_row = row_off - new_row_off
    return (
        (new_col_off, new_row_off, target_w, target_h),
        (slice(crop_row, crop_row + height), slice(crop_col, crop_col + width)),
    )


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
