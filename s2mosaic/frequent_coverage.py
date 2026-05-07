import logging
from typing import List, Tuple

import cv2
import geopandas as gpd
import numpy as np
from geopandas import GeoDataFrame
from pystac.item import Item
from pystac.item_collection import ItemCollection
from rasterio.enums import MergeAlg
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)


def get_coverage(scenes: List[Item]) -> gpd.GeoDataFrame:
    extents = []
    for scene in scenes:
        if scene.geometry is not None and "coordinates" in scene.geometry:
            extents.append(Polygon(scene.geometry["coordinates"][0]))

    extent_gdf = gpd.GeoDataFrame(geometry=extents, crs="EPSG:4326")
    return extent_gdf


def get_raster_coverage(
    scene_bounds: Polygon,
    coverage_gdf: GeoDataFrame,
    local_crs: int,
    resolution: int = 10,
):
    scene_gdf = gpd.GeoDataFrame(
        [scene_bounds], geometry=[scene_bounds], crs="EPSG:4326"
    ).to_crs(f"EPSG:{local_crs}")

    coverage_gdf_local = coverage_gdf.to_crs(f"EPSG:{local_crs}")

    coverage_gdf_local["geometry"] = coverage_gdf_local.make_valid()

    extent = scene_gdf.total_bounds
    x_min, _, _, y_max = extent

    # MGRS tile is exactly 109800m on each side
    tile_size_m = 109800
    side_px = int(round(tile_size_m / resolution))

    geoms_with_values = [(geom, 1) for geom in coverage_gdf_local.geometry]
    raster = rasterize(
        geoms_with_values,
        out_shape=(side_px, side_px),
        fill=0,
        dtype=np.int16,
        transform=Affine(resolution, 0, x_min, 0, -resolution, y_max),
        merge_alg=MergeAlg.add,
    )
    return raster


def get_frequent_coverage(
    scene_bounds: Polygon,
    scenes: ItemCollection,
    coverage_threshold_pct: float = 0.1,
    resolution: int = 10,
) -> np.ndarray:
    scenes_list = list(scenes)
    logger.info(f"Calculating total coverage for {len(scenes_list)} scenes")

    try:
        local_crs = scenes_list[0].properties["proj:epsg"]
    except KeyError:
        local_crs = scenes_list[0].properties["proj:code"]
        local_crs = int(local_crs.split(":")[-1])

    logger.info(f"Using local CRS: EPSG:{local_crs}")

    coverage_gdf = get_coverage(scenes_list)
    raster = get_raster_coverage(
        scene_bounds, coverage_gdf, local_crs, resolution=resolution
    )
    logger.info(f"Coverage raster shape: {raster.shape}")

    max_count = raster.max()
    logger.info(f"Max coverage count: {max_count}")

    # Any area that is covered by more than 10% of the scenes is considered covered
    dynamic_threshold = max_count * coverage_threshold_pct
    logger.info(f"Dynamic threshold: {dynamic_threshold}")

    # Threshold the raster to get a mask of the frequent data
    frequent_data_mask = raster >= dynamic_threshold

    # Expand the mask to include nearby pixels, this grows the no data areas by 4 pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    dilated = cv2.dilate((~frequent_data_mask).astype(np.uint8), kernel, iterations=4)
    return dilated == 0


def get_frequent_coverage_for_bbox(
    scenes: ItemCollection,
    bounds_target: Tuple[float, float, float, float],
    target_crs: int,
    width: int,
    height: int,
    resolution: int,
    coverage_threshold_pct: float = 0.1,
) -> np.ndarray:
    """Frequent-coverage mask for an arbitrary bbox in `target_crs`.

    Variant of get_frequent_coverage() that doesn't assume a single MGRS
    tile: caller passes explicit output shape, target CRS, and bounds in
    that CRS. Useful for the bounds-based mosaic path where scenes may
    come from multiple MGRS tiles in different UTM zones.
    """
    scenes_list = list(scenes)
    coverage_gdf = get_coverage(scenes_list).to_crs(f"EPSG:{target_crs}")
    coverage_gdf["geometry"] = coverage_gdf.make_valid()

    minx, _, _, maxy = bounds_target
    geoms_with_values = [(geom, 1) for geom in coverage_gdf.geometry]
    raster = rasterize(
        geoms_with_values,
        out_shape=(height, width),
        fill=0,
        dtype=np.int16,
        transform=Affine(resolution, 0, minx, 0, -resolution, maxy),
        merge_alg=MergeAlg.add,
    )

    max_count = raster.max()
    if max_count == 0:
        return np.zeros((height, width), dtype=bool)

    dynamic_threshold = max_count * coverage_threshold_pct
    frequent_data_mask = raster >= dynamic_threshold

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    dilated = cv2.dilate((~frequent_data_mask).astype(np.uint8), kernel, iterations=4)
    return dilated == 0
