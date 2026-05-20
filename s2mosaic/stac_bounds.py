"""STAC search helpers for Sentinel-2 mosaics."""

import logging
from datetime import date
from typing import Any, Dict, Optional, Protocol, Union

from pystac.item_collection import ItemCollection
from pystac_client.stac_api_io import StacApiIO
from shapely.geometry import mapping
from urllib3 import Retry

from .geometry import Aoi, Bbox
from .sources import Source
from .stac import STAC_READ_TIMEOUT_SECONDS, filter_latest_processing_baselines

logger = logging.getLogger(__name__)


class _AssetLike(Protocol):
    href: str


class _BoundsItemLike(Protocol):
    id: str
    assets: Any


def _search_for_items_by_bbox(
    bbox_4326: Bbox,
    start_date: date,
    end_date: date,
    source: Source,
    additional_query: Optional[Dict[str, Any]] = None,
    ignore_duplicate_items: bool = True,
) -> ItemCollection:
    """Search Sentinel-2 L2A items intersecting bbox in EPSG:4326."""
    return _search_for_items_by_geometry(
        geometry=bbox_4326,
        start_date=start_date,
        end_date=end_date,
        source=source,
        additional_query=additional_query,
        ignore_duplicate_items=ignore_duplicate_items,
    )


def _search_for_items_by_aoi(
    aoi_4326: Aoi,
    start_date: date,
    end_date: date,
    source: Source,
    additional_query: Optional[Dict[str, Any]] = None,
    ignore_duplicate_items: bool = True,
) -> ItemCollection:
    """Search Sentinel-2 L2A items intersecting a polygon in EPSG:4326."""
    return _search_for_items_by_geometry(
        geometry=aoi_4326,
        start_date=start_date,
        end_date=end_date,
        source=source,
        additional_query=additional_query,
        ignore_duplicate_items=ignore_duplicate_items,
    )


def _search_for_items_by_geometry(
    geometry: Union[Bbox, Aoi],
    start_date: date,
    end_date: date,
    source: Source,
    additional_query: Optional[Dict[str, Any]] = None,
    ignore_duplicate_items: bool = True,
) -> ItemCollection:
    """Search Sentinel-2 L2A items intersecting a bbox or polygon in EPSG:4326."""
    query: Dict[str, Any] = {
        "collections": [source.collection_id],
        "datetime": (
            f"{start_date.strftime('%Y-%m-%dT00:00:00Z')}/"
            f"{end_date.strftime('%Y-%m-%dT00:00:00Z')}"
        ),
    }
    if isinstance(geometry, tuple):
        query["bbox"] = list(geometry)
        search_label = f"bbox {geometry}"
    else:
        query["intersects"] = mapping(geometry)
        search_label = "AOI"
    if additional_query:
        query["query"] = additional_query

    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[502, 503, 504],
        allowed_methods=None,
    )
    stac_api_io = StacApiIO(
        max_retries=retry,
        timeout=STAC_READ_TIMEOUT_SECONDS,
    )
    catalog = source.open_catalog(stac_io=stac_api_io)
    items = catalog.search(**query).item_collection()
    logger.info(f"Found {len(items)} items for {search_label}")

    if ignore_duplicate_items:
        items = filter_latest_processing_baselines(items)
        logger.info(f"After dedupe, {len(items)} items remain")
    return items
