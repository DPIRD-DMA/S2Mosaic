import logging
from typing import List

import geopandas as gpd
import numpy as np
import scipy
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
    scene_bounds: Polygon, coverage_gdf: GeoDataFrame, local_crs: int, resolution=10
):
    scene_gdf = gpd.GeoDataFrame(
        [scene_bounds], geometry=[scene_bounds], crs="EPSG:4326"
    ).to_crs(f"EPSG:{local_crs}")

    coverage_gdf_local = coverage_gdf.to_crs(f"EPSG:{local_crs}")

    coverage_gdf_local["geometry"] = coverage_gdf_local.make_valid()

    extent = scene_gdf.total_bounds
    x_min, _, _, y_max = extent

    geoms_with_values = [(geom, 1) for geom in coverage_gdf_local.geometry]
    raster = rasterize(
        geoms_with_values,
        out_shape=(10980, 10980),
        fill=0,
        dtype=np.int16,
        transform=Affine(resolution, 0, x_min, 0, -resolution, y_max),
        merge_alg=MergeAlg.add,
    )
    return raster


def get_frequent_coverage(
    scene_bounds: Polygon, scenes: ItemCollection, coverage_threshold_pct=0.1
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
    raster = get_raster_coverage(scene_bounds, coverage_gdf, local_crs)
    logger.info(f"Coverage raster shape: {raster.shape}")

    max_count = raster.max()
    logger.info(f"Max coverage count: {max_count}")

    # Any area that is covered by more than 10% of the scenes is considered covered
    dynamic_threshold = max_count * coverage_threshold_pct
    logger.info(f"Dynamic threshold: {dynamic_threshold}")

    # Threshold the raster to get a mask of the frequent data
    frequent_data_mask = raster >= dynamic_threshold

    # Expand the mask to include nearby pixels, this grows the no data areas by 4 pixels
    frequent_data_mask = ~scipy.ndimage.binary_dilation(
        ~frequent_data_mask, iterations=4
    )
    return frequent_data_mask
