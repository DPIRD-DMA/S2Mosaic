"""Centralised GDAL env config for remote COG reads.

These settings target the Planetary Computer / AWS Open Data COG-over-HTTPS
path: enable HTTP/2 multiplexing, merge adjacent ranges, suppress directory
listing on open, and turn on VSI byte caching. They are applied via
``os.environ`` because ``rasterio.Env`` is thread-local — its context does
not propagate into the ThreadPoolExecutor workers used by the tile readers
and the mask-streaming pipeline, so the settings would have no effect on
hot-path COG reads.

User-set values are respected; we only fill in defaults that aren't already
present in the environment.
"""

from __future__ import annotations

import os


# Defaults below were picked from titiler/gdalcubes guidance for remote COG
# access. Comments describe what each setting buys us; tune values per network.
GDAL_NETWORK_DEFAULTS: dict[str, str] = {
    # HTTP/2 + multiplexing: one TCP/TLS connection carries many concurrent
    # range requests in parallel. PC and S3 both serve over HTTP/2.
    "GDAL_HTTP_VERSION": "2TLS",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_MULTIRANGE": "YES",
    # Coalesce neighbouring range requests into one — common when tiled reads
    # span adjacent COG blocks. Cheap, always-on win.
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    # Suppress the LIST on open + sidecar (.aux.xml/.ovr) probing. We always
    # open exact-path .tifs, never directories.
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff,.TIF,.TIFF",
    # VSI byte cache: keeps recently-read ranges in RAM per handle so adjacent
    # window reads don't re-fetch overlapping headers/blocks.
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "5000000",
    # Cross-handle LRU — survives close()/reopen() so retries don't refetch
    # the header.
    "CPL_VSIL_CURL_CACHE_SIZE": "200000000",
    # Bound pathological remote reads so caller retry logic can recover
    # instead of a notebook appearing stuck forever on one COG request.
    "GDAL_HTTP_CONNECTTIMEOUT": "30",
    "GDAL_HTTP_TIMEOUT": "120",
    # GDAL block cache for decoded GeoTIFF tile blocks.
    "GDAL_CACHEMAX": "256",
    # Read enough up-front to cover the COG header + IFD; saves a round trip
    # on every open.
    "GDAL_INGESTED_BYTES_AT_OPEN": "32000",
}


def apply_gdal_network_defaults() -> None:
    """Set network-tuned GDAL env vars, without overwriting any the user set.

    Called once at library import so settings are visible to every rasterio
    open across every thread (rasterio.Env is thread-local — it would not
    reach the ThreadPoolExecutor workers used in the hot path).

    Set ``S2MOSAIC_NO_GDAL_DEFAULTS=1`` to opt out (e.g. for A/B benchmarking
    against the unconfigured baseline).
    """
    if os.environ.get("S2MOSAIC_NO_GDAL_DEFAULTS", "").lower() in ("1", "true", "yes"):
        return
    for key, value in GDAL_NETWORK_DEFAULTS.items():
        os.environ.setdefault(key, value)
