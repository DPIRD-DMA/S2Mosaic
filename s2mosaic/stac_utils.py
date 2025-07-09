import logging
from datetime import date
from typing import Any, Dict, List

import pandas as pd
import pystac_client
import shapely
from pandas import DataFrame
from pystac import Item
from pystac.item_collection import ItemCollection
from pystac_client.stac_api_io import StacApiIO
from shapely.geometry.polygon import Polygon
from urllib3 import Retry

from .helpers import SORT_NEWEST, SORT_OLDEST, SORT_VALID_DATA

logger = logging.getLogger(__name__)


def add_item_info(items: ItemCollection) -> DataFrame:
    """Split items by orbit and sort by no_data"""

    items_list = []
    for item in items:
        nodata = item.properties["s2:nodata_pixel_percentage"]
        data_pct = 100 - nodata

        cloud = item.properties["s2:high_proba_clouds_percentage"]
        shadow = item.properties["s2:cloud_shadow_percentage"]
        good_data_pct = data_pct * (1 - (cloud + shadow) / 100)
        capture_date = item.datetime

        items_list.append(
            {
                "item": item,
                "orbit": item.properties["sat:relative_orbit"],
                "good_data_pct": good_data_pct,
                "datetime": capture_date,
            }
        )

    items_df = pd.DataFrame(items_list)
    return items_df


def search_for_items(
    bounds: Polygon,
    grid_id: str,
    start_date: date,
    end_date: date,
    additional_query: Dict[str, Any],
    ignore_duplicate_items: bool = True,
) -> ItemCollection:
    base_query = {"s2:mgrs_tile": {"eq": grid_id}}
    if additional_query:
        base_query.update(additional_query)

    query = {
        "collections": ["sentinel-2-l2a"],
        "intersects": shapely.to_geojson(bounds),
        "datetime": f"{start_date.isoformat()}Z/{end_date.isoformat()}Z",
        "query": base_query,
    }

    logger.info(
        f"""Searching for items in grid {grid_id} from 
        {start_date} to {end_date} with query: {query}"""
    )

    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[502, 503, 504],
        allowed_methods=None,
    )
    stac_api_io = StacApiIO(max_retries=retry)
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1", stac_io=stac_api_io
    )
    items = catalog.search(**query).item_collection()
    logger.info(f"Found {len(items)}")

    if ignore_duplicate_items:
        items = filter_latest_processing_baselines(items)
        logger.info(f"After filtering, {len(items)} items remain")

    return items


def sort_items(items: DataFrame, sort_method: str) -> DataFrame:
    # Sort the dataframe by selected method then by orbit
    if sort_method == SORT_VALID_DATA:
        items_sorted = items.sort_values("good_data_pct", ascending=False)
        orbits = items_sorted["orbit"].unique()
        orbit_groups = {
            orbit: items_sorted[items_sorted["orbit"] == orbit] for orbit in orbits
        }

        result = []

        while any(len(group) > 0 for group in orbit_groups.values()):
            for orbit in orbits:
                if len(orbit_groups[orbit]) > 0:
                    result.append(orbit_groups[orbit].iloc[0])
                    orbit_groups[orbit] = orbit_groups[orbit].iloc[1:]

        items_sorted = pd.DataFrame(result).reset_index(drop=True)

    elif sort_method == SORT_OLDEST:
        items_sorted = items.sort_values("datetime", ascending=True).reset_index(
            drop=True
        )
    elif sort_method == SORT_NEWEST:
        items_sorted = items.sort_values("datetime", ascending=False).reset_index(
            drop=True
        )
    else:
        raise Exception("Invalid sort method, must be valid_data, oldest or newest")

    return items_sorted


def filter_latest_processing_baselines(
    items: ItemCollection,
) -> ItemCollection:
    """
    Filter STAC items to keep only the latest processing
    baseline for each unique acquisition.
    """
    if len(items) == 0:
        return items

    # Group items by acquisition (same datetime + tile)
    acquisition_groups: Dict[str, List[Dict[str, Any]]] = {}

    for item in items:
        # Create unique key for this acquisition
        datetime_str: str = (
            item.datetime.strftime("%Y%m%dT%H%M%S") if item.datetime else "unknown"
        )
        tile_id: str = item.properties.get("s2:mgrs_tile", "unknown")
        acquisition_key: str = f"{datetime_str}_{tile_id}"

        # Get processing baseline from properties
        baseline_str: str = item.properties.get("s2:processing_baseline", "0.00")
        # Convert to number for comparison (e.g., '05.11' -> 5.11)
        baseline_num: float = float(baseline_str)

        if acquisition_key not in acquisition_groups:
            acquisition_groups[acquisition_key] = []

        acquisition_groups[acquisition_key].append(
            {"item": item, "baseline": baseline_str, "baseline_num": baseline_num}
        )

    # Keep only the latest baseline for each acquisition
    filtered_items: List[Item] = []
    for acquisition_key, group in acquisition_groups.items():
        if len(group) == 1:
            # No duplicates
            filtered_items.append(group[0]["item"])
        else:
            # Keep the highest baseline number
            latest = max(group, key=lambda x: x["baseline_num"])
            filtered_items.append(latest["item"])
            logger.info(
                f"Filtered {acquisition_key}: kept {latest['baseline']}, "
                f"removed {[x['baseline'] for x in group if x != latest]}"
            )

    return ItemCollection(filtered_items)
