"""Imagery provider abstraction.

s2mosaic supports multiple STAC sources for Sentinel-2 L2A. Each ``Source``
captures the per-provider knowledge needed to search, sign, and read assets:

- ``stac_url``: STAC API root
- ``collection_id``: L2A collection name on this provider
- ``sign(href)``: return a usable HTTPS URL (SAS-signed for MPC, identity
  for AWS public buckets)
- ``asset_name(canonical)``: map s2mosaic's canonical band names
  (``B04``, ``SCL`` ...) to the provider's STAC asset key
- ``mgrs_query(grid_id)``: build a STAC ``query`` clause that filters to a
  single MGRS tile, or ``None`` if the provider doesn't expose one (callers
  then rely on ``intersects`` alone)
- ``open_catalog(stac_io)``: open the STAC client; provider-specific options
  (e.g. MPC's ``sign_inplace`` modifier) live here
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import numpy.typing as npt
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


# From processing baseline 04.00 (25 Jan 2022) ESA encodes L2A reflectance as
# ``DN = reflectance * 10000 - BOA_ADD_OFFSET`` with an offset of -1000, so
# zero reflectance reads DN 1000; earlier baselines have no offset. Only the
# spectral bands carry it: SCL, AOT, WVP and the 8-bit TCI are unaffected.
#
# Microsoft serves those DNs as-is and publishes no offset metadata, so the
# baseline is the only signal. Element 84 removes the offset itself and flags
# the item ``earthsearch:boa_offset_applied``; its ``raster:bands`` still say
# ``offset: -0.1``, which describes the original product, not its pixels.
BOA_OFFSET_MIN_BASELINE = 4.0
BOA_ADD_OFFSET = -1000
BOA_OFFSET_BANDS = frozenset(
    {
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B11",
        "B12",
    }
)


def boa_add_offset(properties: Dict[str, Any], canonical: str) -> int:
    """DN offset to add to ``canonical`` so it reads on the pre-04.00 scale.

    Returns ``BOA_ADD_OFFSET`` for a spectral band still carrying the offset,
    else 0. An item with no parseable baseline is treated as unshifted.
    """
    if canonical not in BOA_OFFSET_BANDS:
        return 0
    if properties.get("earthsearch:boa_offset_applied"):
        return 0
    try:
        baseline = float(properties.get("s2:processing_baseline", ""))
    except (TypeError, ValueError):
        return 0
    return BOA_ADD_OFFSET if baseline >= BOA_OFFSET_MIN_BASELINE else 0


def apply_boa_offset(arr: npt.NDArray[Any], offset: int) -> npt.NDArray[Any]:
    """Add a (negative) ``offset`` to valid DNs, keeping 0 as nodata.

    DN 0 is L2A's NODATA under every baseline and stays 0. DNs at or below
    ``-offset`` encode reflectance <= 0: real dark-water observations, not
    missing data. They clip to 1 so they stay valid rather than becoming
    nodata, or wrapping to ~65000 under unsigned subtraction.
    """
    if offset == 0:
        return arr
    shift = -offset
    shifted = arr.astype(np.int32) - shift
    out = np.where(arr == 0, 0, np.maximum(shifted, 1))
    return out.astype(arr.dtype)


@dataclass(frozen=True)
class Source:
    name: str
    stac_url: str
    collection_id: str
    sign: Callable[[str], str]
    # Map of canonical band/asset name -> provider's STAC asset key.
    # Lookups fall through to the canonical name when not in the map.
    band_assets: Dict[str, str] = field(default_factory=dict)
    # Per-asset internal COG block size (square). Used by the tile-aggregation
    # path to pick output tile sizes that align with source blocks so each
    # tile read fetches whole blocks rather than fringes. Unlisted assets
    # fall back to ``default_block_size``.
    asset_block_sizes: Dict[str, int] = field(default_factory=dict)
    default_block_size: int = 512
    # Builder for a STAC ``query`` clause restricting to a single MGRS tile.
    # Returns ``None`` for providers that don't expose a single-field MGRS
    # property (callers then rely on the ``intersects`` geometry filter alone).
    _mgrs_query: Optional[Callable[[str], Dict[str, Any]]] = None
    # Asset href prefixes this source can read without credentials. Items
    # whose needed assets fall outside them are dropped before de-duplication
    # (see ``stac.drop_unreadable_items``), so a readable processing of the
    # same acquisition wins instead. ``None`` disables the check.
    readable_href_prefixes: Optional[Tuple[str, ...]] = None

    def asset_name(self, canonical: str) -> str:
        return self.band_assets.get(canonical, canonical)

    def block_size(self, canonical: str) -> int:
        """Internal COG block size for ``canonical`` band on this source."""
        return self.asset_block_sizes.get(canonical, self.default_block_size)

    def max_block_size_for_bands(self, bands: Iterable[str]) -> int:
        """Largest block size across ``bands``.

        When the aggregation reads several bands per tile, the smallest
        useful output tile is the *largest* source block among those bands:
        anything smaller still fetches whole source blocks and wastes the
        fringe. Returning the max gives the tile sizer a safe lower bound.
        """
        sizes = [self.block_size(b) for b in bands]
        if not sizes:
            return self.default_block_size
        return max(sizes)

    def mgrs_query(self, grid_id: str) -> Optional[Dict[str, Any]]:
        if self._mgrs_query is None:
            return None
        return self._mgrs_query(grid_id)

    def open_catalog(self, stac_io: StacApiIO) -> pystac_client.Client:
        client = pystac_client.Client.open(self.stac_url, stac_io=stac_io)
        # Every search this package issues carries ``query`` -- the MGRS tile
        # filter in grid mode, and ``eo:cloud_cover`` everywhere. MPC's
        # landing page declares only four conformance classes and omits
        # ``item-search#query``, so pystac_client warns on every search even
        # though MPC honours the extension; verified against ``s2:mgrs_tile``
        # and ``eo:cloud_cover``, and pinned by the slow test
        # ``TestServerSideQueryFiltering``. Element 84 declares it, where this
        # is a no-op. Asserting it is narrower than filtering the warning:
        # it records what was checked and leaves every other warning intact.
        #
        # The Query extension, not the newer CQL2 Filter extension, is what
        # both providers actually implement. Neither declares
        # ``item-search#filter``, and Element 84 answers a CQL2 search with
        # HTTP 200 and an unfiltered item list, which would put scenes from
        # the wrong MGRS tiles into a grid mosaic with nothing raised.
        client.add_conforms_to("QUERY")
        return client


def _mpc_mgrs_query(grid_id: str) -> Dict[str, Any]:
    return {"s2:mgrs_tile": {"eq": grid_id}}


def _aws_mgrs_query(grid_id: str) -> Dict[str, Any]:
    # Element 84 Earth Search v1 splits the MGRS tile into separate fields.
    # grid_id format: ``50HMH`` -> utm_zone=50, latitude_band=H, grid_square=MH
    #
    # Element 84 rejects ``query`` *combined* with ``intersects``/``bbox``,
    # but accepts query-only searches. ``search_for_items`` therefore drops
    # ``intersects`` when an MGRS filter is present on this source.
    match = re.fullmatch(
        r"(?P<utm_zone>\d{1,2})(?P<latitude_band>[A-Z])"
        r"(?P<grid_square>[A-Z]{2})",
        grid_id,
    )
    if match is None:
        raise ValueError(
            f"Grid {grid_id!r} is invalid. It should be in the format '50HMH'."
        )
    utm_zone = match.group("utm_zone")
    latitude_band = match.group("latitude_band")
    grid_square = match.group("grid_square")
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
# common-name asset keys (``red``, ``green`` ...) rather than band IDs, and
# lowercases the non-spectral ones (``scl``, ``aot``, ``wvp``). Every band in
# ``VALID_BANDS`` needs an entry here except ``visual``, which is spelled the
# same on both providers: ``asset_name`` falls through to the canonical name,
# so a gap surfaces as a KeyError on the item's assets rather than an error
# anyone can read. Public S3 backing, so no signing required.
AWS = Source(
    name=SOURCE_AWS,
    stac_url="https://earth-search.aws.element84.com/v1",
    collection_id="sentinel-2-l2a",
    sign=_identity_sign,
    band_assets={
        "AOT": "aot",
        "WVP": "wvp",
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
    # Element 84's Sentinel-2 L2A COGs use 1024-pixel blocks for the
    # native-10m bands (verified for B02/B03/B04/B08/visual) and 512-pixel
    # blocks for SCL. Bands not yet measured fall back to default_block_size
    # (512), which is safe: a smaller default just means the adaptive tiler
    # is allowed to split tiles further than strictly optimal.
    asset_block_sizes={
        "B02": 1024,
        "B03": 1024,
        "B04": 1024,
        "B08": 1024,
        "visual": 1024,
        "SCL": 512,
    },
    default_block_size=512,
    _mgrs_query=_aws_mgrs_query,
    # Element 84 occasionally publishes an item before its COG conversion.
    # Its assets then point at the requester-pays JP2 archive
    # (``s3://sentinel-s2-l2a/...``), and every read fails without AWS
    # credentials. Seen on S2B_35UNT_20190914_1_L2A (baseline 05.00), whose
    # 02.13 processing has normal public COGs.
    readable_href_prefixes=("https://",),
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
