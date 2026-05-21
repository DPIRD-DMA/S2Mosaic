import json
import logging
import re
from datetime import date
from typing import Any, Dict, List, Optional

import pandas as pd
import shapely
from pandas import DataFrame
from pystac import Item
from pystac.item_collection import ItemCollection
from pystac_client.stac_api_io import StacApiIO
from shapely.geometry.polygon import Polygon
from urllib3 import Retry

from .config import SCENE_ORDER_NEWEST, SCENE_ORDER_OLDEST, SCENE_ORDER_VALID_DATA
from .helpers import pickle_cache
from .sources import Source

logger = logging.getLogger(__name__)
STAC_READ_TIMEOUT_SECONDS = 30
STAC_RETRY_STATUS_CODES = [408, 429, 500, 502, 503, 504]

# Column names for the DataFrame produced by add_item_info().
ITEM_COL = "item"
ORBIT_COL = "orbit"
GOOD_DATA_PCT_COL = "good_data_pct"
DATETIME_COL = "datetime"

# Element 84's Earth Search v1 drops ``sat:relative_orbit`` and
# ``s2:mgrs_tile`` from item properties. The orbit number is embedded in the
# product URI (``..._R060_...``) and the tile lives in ``grid:code``
# (``MGRS-50HMH``); these helpers recover both with property fallbacks first.
_PRODUCT_URI_ORBIT_RE = re.compile(r"_R(\d+)_")


def _extract_relative_orbit(props: Dict[str, Any]) -> int:
    if "sat:relative_orbit" in props:
        return int(props["sat:relative_orbit"])
    product_uri = props.get("s2:product_uri") or props.get("s2:product_id") or ""
    m = _PRODUCT_URI_ORBIT_RE.search(product_uri)
    return int(m.group(1)) if m else 0


def _extract_mgrs_tile(props: Dict[str, Any]) -> Optional[str]:
    if "s2:mgrs_tile" in props:
        return str(props["s2:mgrs_tile"])
    grid_code = props.get("grid:code")
    if isinstance(grid_code, str) and grid_code.startswith("MGRS-"):
        return grid_code[len("MGRS-") :]
    return None


def add_item_info(items: ItemCollection) -> DataFrame:
    """Split items by orbit and sort by no_data.

    Element 84's Earth Search publishes ``s2:nodata_pixel_percentage`` and
    ``s2:high_proba_clouds_percentage`` (so the MPC code path is preserved),
    but it does *not* publish ``sat:relative_orbit`` — we recover that from
    the ``_R(\\d+)_`` token in ``s2:product_uri``.
    """

    items_list = []
    for item in items:
        props = item.properties
        nodata = props.get("s2:nodata_pixel_percentage", 0)
        data_pct = 100 - nodata

        cloud = props.get("s2:high_proba_clouds_percentage", 0)
        shadow = props.get("s2:cloud_shadow_percentage", 0)
        good_data_pct = data_pct * (1 - (cloud + shadow) / 100)
        capture_date = item.datetime

        items_list.append(
            {
                ITEM_COL: item,
                ORBIT_COL: _extract_relative_orbit(props),
                GOOD_DATA_PCT_COL: good_data_pct,
                DATETIME_COL: capture_date,
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
    source: Source,
    ignore_duplicate_items: bool = True,
) -> ItemCollection:
    base_query: Dict[str, Any] = {}
    mgrs_filter = source.mgrs_query(grid_id)
    if mgrs_filter is not None:
        base_query.update(mgrs_filter)
    if additional_query:
        base_query.update(additional_query)

    query: Dict[str, Any] = {
        "collections": [source.collection_id],
        "intersects": shapely.to_geojson(bounds),
        "datetime": (
            f"{start_date.strftime('%Y-%m-%dT00:00:00Z')}/"
            f"{end_date.strftime('%Y-%m-%dT00:00:00Z')}"
        ),
    }
    if base_query:
        query["query"] = base_query

    logger.info(
        f"""Searching for items in grid {grid_id} from
        {start_date} to {end_date} with query: {query}"""
    )

    # Cache the STAC query result so dev iteration survives PC outages and
    # avoids redundant network calls. URLs in cached items aren't signed yet
    # (signing happens at asset read time), so the cached payload is stable
    # across runs.
    cache_key = (
        f"{source.name}|{grid_id}|{start_date.isoformat()}|{end_date.isoformat()}|"
        f"{shapely.to_geojson(bounds)}|"
        f"{json.dumps(additional_query or {}, sort_keys=True, default=str)}|"
        f"dedupe={ignore_duplicate_items}"
    )

    def _do_search() -> ItemCollection:
        retry = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=STAC_RETRY_STATUS_CODES,
            allowed_methods=None,
            respect_retry_after_header=True,
        )
        stac_api_io = StacApiIO(
            max_retries=retry,
            timeout=STAC_READ_TIMEOUT_SECONDS,
        )
        catalog = source.open_catalog(stac_io=stac_api_io)
        items = catalog.search(**query).item_collection()
        logger.info(f"Found {len(items)}")
        # When the source can't filter by MGRS server-side (e.g. AWS — Earth
        # Search rejects ``query`` combined with ``intersects``), the search
        # also returns scenes from adjacent tiles that overlap the tile
        # polygon. Filter those out client-side using the per-item tile ID
        # so grid_id-mode output is restricted to the requested tile.
        if mgrs_filter is None:
            before = len(items)
            kept = [it for it in items if _extract_mgrs_tile(it.properties) == grid_id]
            items = ItemCollection(kept)
            if len(items) != before:
                logger.info(
                    "Post-filtered %d -> %d items by grid_id=%s",
                    before,
                    len(items),
                    grid_id,
                )
        if ignore_duplicate_items:
            items = filter_latest_processing_baselines(items)
            logger.info(f"After filtering, {len(items)} items remain")
        return items

    return pickle_cache("stac_search", cache_key, _do_search)


def sort_items(items: DataFrame, scene_order: str) -> DataFrame:
    # The valid_data branch round-robins by relative orbit so the early-stopped
    # mosaic blends scenes from different overpasses within a single MGRS tile.
    # In bounds mode an AOI may pull scenes from several MGRS tiles, where
    # ``sat:relative_orbit`` no longer identifies a single ground-track pass —
    # the round-robin still produces a valid sort but is no longer "balance
    # acquisitions across passes". Acceptable today; revisit if bounds-mode
    # output quality becomes a concern.
    if scene_order == SCENE_ORDER_VALID_DATA:
        items_sorted = items.sort_values(GOOD_DATA_PCT_COL, ascending=False)
        orbits = items_sorted[ORBIT_COL].unique()
        orbit_groups = {
            orbit: items_sorted[items_sorted[ORBIT_COL] == orbit] for orbit in orbits
        }

        result = []

        while any(len(group) > 0 for group in orbit_groups.values()):
            for orbit in orbits:
                if len(orbit_groups[orbit]) > 0:
                    result.append(orbit_groups[orbit].iloc[0])
                    orbit_groups[orbit] = orbit_groups[orbit].iloc[1:]

        items_sorted = pd.DataFrame(result).reset_index(drop=True)

    elif scene_order == SCENE_ORDER_OLDEST:
        items_sorted = items.sort_values(DATETIME_COL, ascending=True).reset_index(
            drop=True
        )
    elif scene_order == SCENE_ORDER_NEWEST:
        items_sorted = items.sort_values(DATETIME_COL, ascending=False).reset_index(
            drop=True
        )
    else:
        raise ValueError("Invalid scene_order, must be valid_data, oldest or newest")

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
        tile_id: str = _extract_mgrs_tile(item.properties) or "unknown"
        acquisition_key: str = f"{datetime_str}_{tile_id}"

        # Get processing baseline from properties
        baseline_str: str = item.properties.get("s2:processing_baseline", "0.00")
        # Convert to number for comparison (e.g., '05.11' -> 5.11)
        try:
            baseline_num = float(baseline_str)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid processing baseline %r for item %s; treating as 0.00",
                baseline_str,
                item.id,
            )
            baseline_str = str(baseline_str)
            baseline_num = 0.0

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
