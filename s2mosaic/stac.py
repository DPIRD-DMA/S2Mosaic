import logging
import re
from datetime import date
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import pyproj
from pandas import DataFrame
from pystac import Item
from pystac.item_collection import ItemCollection
from pystac_client.stac_api_io import StacApiIO
from shapely.geometry import shape
from shapely.ops import transform as transform_geometry
from urllib3 import Retry

from .config import (
    CLOUD_MASK_OCM,
    CLOUD_MASK_SCL,
    DEFAULT_BANDS,
    SCENE_ORDER_NEWEST,
    SCENE_ORDER_OLDEST,
    SCENE_ORDER_VALID_DATA,
)
from .geometry import _OCM_BANDS
from .sources import SEARCH_QUERY, Source

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


# The sensing-start field of an ``s2:datastrip_id``; see _acquisition_key.
_DATASTRIP_SENSING_RE = re.compile(r"_S(\d{8}T\d{6})_")


def _extract_mgrs_tile(props: Dict[str, Any]) -> Optional[str]:
    if "s2:mgrs_tile" in props:
        return str(props["s2:mgrs_tile"])
    grid_code = props.get("grid:code")
    if isinstance(grid_code, str) and grid_code.startswith("MGRS-"):
        return grid_code[len("MGRS-") :]
    # DEA names each dataset by the MGRS tile it was processed on.
    region_code = props.get("odc:region_code")
    if isinstance(region_code, str):
        return region_code
    return None


def _datastrip_id(props: Dict[str, Any]) -> Any:
    # MPC and Element 84 publish ``s2:datastrip_id``; DEA carries the same
    # ESA identifier (of the L1C datastrip) as ``sentinel:datastrip_id``.
    return props.get("s2:datastrip_id", props.get("sentinel:datastrip_id"))


def _nodata_percentage(item: Any) -> float:
    """Percent of the item's tile grid that holds no data.

    MPC and Element 84 publish ``s2:nodata_pixel_percentage``. DEA doesn't,
    so it is estimated from the item's footprint against its raster grid
    (``proj:shape`` x ``proj:transform``); on 50HMH that lands within 0.6
    points of MPC's published figure for the same acquisitions. Falls back
    to 0, the previous behaviour, when neither is available.
    """
    props = item.properties
    if "s2:nodata_pixel_percentage" in props:
        return float(props["s2:nodata_pixel_percentage"])
    try:
        code = props.get("proj:epsg") or props["proj:code"]
        epsg = int(str(code).split(":")[-1])
        grid_transform = props["proj:transform"]
        rows, cols = props["proj:shape"]
        footprint = shape(item.geometry)
    except (AttributeError, KeyError, TypeError, ValueError):
        return 0.0
    tile_area = rows * cols * abs(grid_transform[0] * grid_transform[4])
    to_utm = pyproj.Transformer.from_crs(4326, epsg, always_xy=True).transform
    data_area = transform_geometry(to_utm, footprint).area
    return float(min(max(100.0 - 100.0 * data_area / tile_area, 0.0), 100.0))


def add_item_info(items: ItemCollection) -> DataFrame:
    """Split items by orbit and sort by no_data.

    Element 84's Earth Search publishes ``s2:nodata_pixel_percentage`` and
    ``s2:high_proba_clouds_percentage`` (so the MPC code path is preserved),
    but it does *not* publish ``sat:relative_orbit``; we recover that from
    the ``_R(\\d+)_`` token in ``s2:product_uri``.
    """

    items_list = []
    for item in items:
        props = item.properties
        data_pct = 100 - _nodata_percentage(item)

        # DEA publishes fmask percentages instead of Sen2Cor's.
        cloud = props.get(
            "s2:high_proba_clouds_percentage", props.get("fmask:cloud", 0)
        )
        shadow = props.get(
            "s2:cloud_shadow_percentage", props.get("fmask:cloud_shadow", 0)
        )
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


def assets_read(bands: Iterable[str], cloud_mask: str) -> List[str]:
    """Canonical band/asset names a mosaic reads from each scene."""
    mask = [CLOUD_MASK_SCL] if cloud_mask == CLOUD_MASK_SCL else list(_OCM_BANDS)
    return list(dict.fromkeys([*bands, *mask]))


def drop_unreadable_items(
    items: ItemCollection,
    source: Source,
    assets: Optional[Iterable[str]] = None,
) -> ItemCollection:
    """Drop items whose needed assets this source cannot read.

    Runs before ``filter_latest_processing_baselines``: an unreadable 05.00
    item would otherwise shadow the readable 02.xx processing of the same
    acquisition, and the scene would be lost at read time. ``assets`` are
    canonical band names; ``None`` checks what a default ``mosaic()`` call
    reads (``DEFAULT_BANDS`` plus the OCM inputs). It never checks every
    asset on the item: Earth Search items also carry ``*-jp2`` data assets
    that always point at the requester-pays archive, and nothing reads them.
    """
    # getattr: duck-typed sources (and test fakes) may not declare the field.
    prefixes = getattr(source, "readable_href_prefixes", None)
    if prefixes is None or len(items) == 0:
        return items

    needed = (
        list(assets)
        if assets is not None
        else assets_read(DEFAULT_BANDS, CLOUD_MASK_OCM)
    )
    keys = [source.asset_name(a) for a in needed]

    # With explicit assets a missing one will be read, so it is unreadable.
    # The default is only a guess at what will be read, so absent is not bad.
    strict = assets is not None

    def readable(item: Item) -> bool:
        for key in keys:
            asset = item.assets.get(key)
            if asset is None:
                if strict:
                    return False
            elif not asset.href.startswith(prefixes):
                return False
        return True

    kept = [it for it in items if readable(it)]
    if len(kept) != len(items):
        dropped = [it.id for it in items if not readable(it)]
        logger.warning(
            "Dropped %d item(s) whose assets %s cannot read without credentials: %s",
            len(dropped),
            source.name,
            dropped,
        )
    return ItemCollection(kept)


# Query-extension operators and their CQL2 equivalents.
_QUERY_TO_CQL2_OPS = {
    "eq": "=",
    "neq": "<>",
    "lt": "<",
    "lte": "<=",
    "gt": ">",
    "gte": ">=",
    "in": "in",
}


def query_to_cql2(query: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Translate a Query-extension dict into CQL2 JSON clauses.

    ``additional_query`` is documented in the Query extension's form, which
    is all MPC and Element 84 accept. A source that only takes CQL2 (DEA)
    gets the same filter translated, so one ``additional_query`` works on
    every source. Operators without a CQL2 equivalent here raise rather than
    being dropped, because a silently ignored filter changes the mosaic.
    """
    clauses: List[Dict[str, Any]] = []
    for prop, conditions in query.items():
        if not isinstance(conditions, dict):
            raise ValueError(
                f"additional_query[{prop!r}] must map operators to values, "
                f"e.g. {{'lt': 50}}; got {conditions!r}"
            )
        for op, value in conditions.items():
            if op not in _QUERY_TO_CQL2_OPS:
                raise ValueError(
                    f"additional_query operator {op!r} on {prop!r} has no CQL2 "
                    f"translation; supported: {sorted(_QUERY_TO_CQL2_OPS)}"
                )
            args: List[Any] = [{"property": prop}, value]
            clauses.append({"op": _QUERY_TO_CQL2_OPS[op], "args": args})
    return clauses


def search_params(
    source: Source,
    mgrs_filter: Optional[Dict[str, Any]],
    additional_query: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Property-filter kwargs for ``Client.search`` in the source's dialect.

    ``mgrs_filter`` comes from ``source.mgrs_query`` and is already in that
    dialect; ``additional_query`` is always in Query-extension form. Returns
    ``{}`` when there is nothing to filter on.
    """
    # getattr: duck-typed sources (and test fakes) may not declare these.
    extension = getattr(source, "search_extension", SEARCH_QUERY)
    if extension == SEARCH_QUERY:
        query: Dict[str, Any] = {}
        if mgrs_filter:
            query.update(mgrs_filter)
        if additional_query:
            query.update(additional_query)
        return {"query": query} if query else {}

    clauses = list(getattr(source, "base_filters", ()))
    if mgrs_filter:
        clauses.append(mgrs_filter)
    if additional_query:
        clauses.extend(query_to_cql2(additional_query))
    if not clauses:
        return {}
    cql2 = clauses[0] if len(clauses) == 1 else {"op": "and", "args": clauses}
    return {"filter": cql2, "filter_lang": extension}


def search_collections(source: Source) -> List[str]:
    """Collections to search; duck-typed sources may declare only one."""
    return list(getattr(source, "collections", None) or [source.collection_id])


def search_for_items(
    grid_id: str,
    start_date: date,
    end_date: date,
    additional_query: Dict[str, Any],
    source: Source,
    ignore_duplicate_items: bool = True,
    assets: Optional[Iterable[str]] = None,
) -> ItemCollection:
    mgrs_filter = source.mgrs_query(grid_id)
    if mgrs_filter is None:
        raise ValueError(
            f"Source {source.name!r} does not support MGRS tile search; "
            "grid_id mode requires a source with a server-side MGRS filter."
        )

    # Search by MGRS tile only, with no ``intersects``. Both MPC and AWS reject
    # the combination on the same query, and the per-field MGRS filter is
    # precise enough on its own (one MGRS tile id ↔ one set of items).
    query: Dict[str, Any] = {
        "collections": search_collections(source),
        "datetime": (
            f"{start_date.strftime('%Y-%m-%dT00:00:00Z')}/"
            f"{end_date.strftime('%Y-%m-%dT00:00:00Z')}"
        ),
        **search_params(source, mgrs_filter, additional_query),
    }

    logger.info(
        f"""Searching for items in grid {grid_id} from
        {start_date} to {end_date} with query: {query}"""
    )

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
    # Defensive client-side filter. Both providers should already return
    # exactly the requested tile, but if a provider ever loosens its query
    # semantics this catches the regression rather than silently mosaicking
    # in scenes from an adjacent tile.
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
    items = drop_unreadable_items(items, source, assets)
    if ignore_duplicate_items:
        items = filter_latest_processing_baselines(items)
        logger.info(f"After filtering, {len(items)} items remain")
    return items


def sort_items(items: DataFrame, scene_order: str) -> DataFrame:
    # The valid_data branch round-robins by relative orbit so the early-stopped
    # mosaic blends scenes from different overpasses within a single MGRS tile.
    # In bounds mode an AOI may pull scenes from several MGRS tiles, where
    # ``sat:relative_orbit`` no longer identifies a single ground-track pass,
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


def _acquisition_key(item: Item) -> str:
    """Identify the granule an item is a processing of.

    The datastrip sensing start, from ``s2:datastrip_id``::

        S2A_OPER_MSI_L2A_DS_ESRI_20201003T190725_S20191123T022659_N02.12
                                 └─generation──┘ └─sensing start─┘ └base┘

    Only the generation time and the baseline move when ESA reprocesses, so
    the sensing start names the granule itself. Both providers publish it and
    agree on it, which the ``datetime`` property does not:

    * Element 84 stores the granule sensing time, and the Collection-1
      reprocessing restamps it -- the 2019-03-23 acquisition of 50HMH is
      ``02:31:32`` at baseline 02.11 and ``02:27:06`` at 05.00. Keyed on
      datetime the two are different acquisitions, so this function keeps
      both and the same scene enters the stack twice.
    * Microsoft stores the *datatake* start, which every granule of that
      datatake shares. Where one tile is covered by two granules -- 50HMH on
      2019-11-23 has one at 58% nodata and another at 74% -- keying on
      datetime merges them, and since their baselines tie, ``max`` discards
      one by response order alone.

    Measured over 6 tiles x 4 years, moving to this key recovers 56 granules
    Microsoft was dropping and collapses 85 duplicates Element 84 was
    keeping; the two providers then agree to within 0.2% on the number of
    distinct granules, against 5% before.

    Falls back to the datetime key if ``s2:datastrip_id`` is absent or
    unparseable. No item in that sample lacked it, so the fallback is
    defensive rather than routine, and it warns when it fires.
    """
    tile_id: str = _extract_mgrs_tile(item.properties) or "unknown"
    datastrip_id = _datastrip_id(item.properties)
    if isinstance(datastrip_id, str):
        match = _DATASTRIP_SENSING_RE.search(datastrip_id)
        if match is not None:
            return f"{match.group(1)}_{tile_id}"
    logger.warning(
        "Item %s has no parseable s2:datastrip_id or sentinel:datastrip_id "
        "(%r); falling back to its "
        "datetime, which providers stamp inconsistently",
        item.id,
        datastrip_id,
    )
    datetime_str = (
        item.datetime.strftime("%Y%m%dT%H%M%S") if item.datetime else "unknown"
    )
    return f"{datetime_str}_{tile_id}"


def filter_latest_processing_baselines(
    items: ItemCollection,
) -> ItemCollection:
    """
    Filter STAC items to keep only the latest processing
    baseline for each unique acquisition.
    """
    if len(items) == 0:
        return items

    # Group items by the granule they are a processing of; see _acquisition_key.
    acquisition_groups: Dict[str, List[Dict[str, Any]]] = {}

    for item in items:
        acquisition_key: str = _acquisition_key(item)

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
