"""Imagery provider abstraction.

s2mosaic supports multiple STAC sources for Sentinel-2 L2A. Each ``Source``
captures the per-provider knowledge needed to search, sign, and read assets:

- ``stac_url`` — STAC API root
- ``collection_id`` — L2A collection name on this provider
- ``sign(href)`` — return a usable HTTPS URL (SAS-signed for MPC, identity
  for AWS public buckets)
- ``asset_name(canonical)`` — map s2mosaic's canonical band names
  (``B04``, ``SCL`` ...) to the provider's STAC asset key
- ``mgrs_query(grid_id)`` — build a STAC ``query`` clause that filters to a
  single MGRS tile, or ``None`` if the provider doesn't expose one (callers
  then rely on ``intersects`` alone)
- ``open_catalog(stac_io)`` — open the STAC client; provider-specific options
  (e.g. MPC's ``sign_inplace`` modifier) live here
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import pystac_client
from pystac_client.stac_api_io import StacApiIO

logger = logging.getLogger(__name__)

SOURCE_MPC = "MPC"
SOURCE_AWS = "AWS"


def _identity_sign(href: str) -> str:
    return href


def _mpc_sign(href: str) -> str:
    import planetary_computer

    return planetary_computer.sign(href)  # type: ignore[no-any-return, unused-ignore]


@dataclass(frozen=True)
class Source:
    name: str
    stac_url: str
    collection_id: str
    sign: Callable[[str], str]
    # Map of canonical band/asset name -> provider's STAC asset key.
    # Lookups fall through to the canonical name when not in the map.
    band_assets: Dict[str, str] = field(default_factory=dict)
    # Builder for a STAC ``query`` clause restricting to a single MGRS tile.
    # Returns ``None`` for providers that don't expose a single-field MGRS
    # property (callers then rely on the ``intersects`` geometry filter alone).
    _mgrs_query: Optional[Callable[[str], Dict[str, Any]]] = None

    def asset_name(self, canonical: str) -> str:
        return self.band_assets.get(canonical, canonical)

    def mgrs_query(self, grid_id: str) -> Optional[Dict[str, Any]]:
        if self._mgrs_query is None:
            return None
        return self._mgrs_query(grid_id)

    def open_catalog(self, stac_io: StacApiIO) -> pystac_client.Client:
        return pystac_client.Client.open(self.stac_url, stac_io=stac_io)


def _mpc_mgrs_query(grid_id: str) -> Dict[str, Any]:
    return {"s2:mgrs_tile": {"eq": grid_id}}


def _aws_mgrs_query(grid_id: str) -> Dict[str, Any]:
    # Element 84 Earth Search v1 splits the MGRS tile into separate fields.
    # grid_id format: ``50HMH`` -> utm_zone=50, latitude_band=H, grid_square=MH
    #
    # Element 84 returns 0 items when ``query`` is combined with
    # ``intersects``/``bbox`` (verified against live API), so we leave this
    # builder in for parity/test coverage but the AWS source itself disables
    # ``mgrs_query`` (set to ``None`` below) — grid-mode precision is then
    # restored by ``search_for_items`` post-filtering on ``grid:code``.
    utm_zone = grid_id[:-3]
    latitude_band = grid_id[-3]
    grid_square = grid_id[-2:]
    return {
        "mgrs:utm_zone": {"eq": int(utm_zone)},
        "mgrs:latitude_band": {"eq": latitude_band},
        "mgrs:grid_square": {"eq": grid_square},
    }


MPC = Source(
    name=SOURCE_MPC,
    stac_url="https://planetarycomputer.microsoft.com/api/stac/v1",
    collection_id="sentinel-2-l2a",
    sign=_mpc_sign,
    band_assets={},  # MPC uses canonical band IDs as asset keys
    _mgrs_query=_mpc_mgrs_query,
)

# Element 84 Earth Search v1. The ``sentinel-2-l2a`` collection here uses
# common-name asset keys (``red``, ``green`` ...) rather than band IDs and
# lowercases ``scl``. Public S3 backing — no signing required.
AWS = Source(
    name=SOURCE_AWS,
    stac_url="https://earth-search.aws.element84.com/v1",
    collection_id="sentinel-2-l2a",
    sign=_identity_sign,
    band_assets={
        "B01": "coastal",
        "B02": "blue",
        "B03": "green",
        "B04": "red",
        "B05": "rededge1",
        "B06": "rededge2",
        "B07": "rededge3",
        "B08": "nir",
        "B8A": "nir08",
        "B09": "nir09",
        "B11": "swir16",
        "B12": "swir22",
        "SCL": "scl",
        # "visual" is the same key on both providers.
    },
    # ``_mgrs_query=None``: see ``_aws_mgrs_query`` docstring — Earth Search
    # rejects ``query`` combined with ``intersects``. Grid-mode tile precision
    # is restored by ``search_for_items``'s post-filter on ``grid:code``.
    _mgrs_query=None,
)


_SOURCES: Dict[str, Source] = {SOURCE_MPC: MPC, SOURCE_AWS: AWS}
VALID_SOURCES = frozenset(_SOURCES)


def get_source(name: str) -> Source:
    try:
        return _SOURCES[name]
    except KeyError as e:
        raise ValueError(
            f"Unknown source {name!r}; must be one of {sorted(VALID_SOURCES)}"
        ) from e
