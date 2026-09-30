"""Centralised GDAL env config for remote COG reads.

These settings target the Planetary Computer / AWS Open Data COG-over-HTTPS
path: enable HTTP/2 multiplexing, merge adjacent ranges, suppress directory
listing on open, and turn on VSI byte caching. They are applied via
``os.environ`` because ``rasterio.Env`` is thread-local: its context does
not propagate into the ThreadPoolExecutor workers used by the tile readers
and the mask-streaming pipeline, so the settings would have no effect on
hot-path COG reads.

User-set values are respected; we only fill in defaults that aren't already
present in the environment. Call ``apply_gdal_network_defaults`` explicitly
when a process wants these global GDAL defaults.
"""

from __future__ import annotations

import functools
import os
from contextlib import contextmanager
from typing import Callable, Iterator, Optional, TypeVar

import rasterio as rio
from rasterio.env import get_gdal_config

T = TypeVar("T")

_NON_CACHED_KEY = "CPL_VSIL_CURL_NON_CACHED"


# Defaults below were picked from titiler/gdalcubes guidance for remote COG
# access. Comments describe what each setting buys us; tune values per network.
GDAL_NETWORK_DEFAULTS: dict[str, str] = {
    # HTTP/2 + multiplexing: one TCP/TLS connection carries many concurrent
    # range requests in parallel. PC and S3 both serve over HTTP/2.
    "GDAL_HTTP_VERSION": "2TLS",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_MULTIRANGE": "YES",
    # Coalesce neighbouring range requests into one, common when tiled reads
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
    # Cross-handle LRU. Survives close()/reopen() so retries don't refetch
    # the header.
    "CPL_VSIL_CURL_CACHE_SIZE": "200000000",
    # Bound pathological remote reads so caller retry logic can recover
    # instead of a notebook appearing stuck forever on one COG request.
    "GDAL_HTTP_CONNECTTIMEOUT": "30",
    "GDAL_HTTP_TIMEOUT": "120",
    # Curl-layer retry for transient HTTP failures (429/502/503/504, connection
    # resets, TLS hiccups). Operates below the app-level _retry_open_raster
    # so a brief blob-storage blip doesn't burn through the 3-strike app retry
    # in a 3-second window.
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "1",
    # GDAL block cache for decoded GeoTIFF tile blocks.
    "GDAL_CACHEMAX": "256",
    # Read enough up-front to cover the COG header + IFD; saves a round trip
    # on every open.
    "GDAL_INGESTED_BYTES_AT_OPEN": "32000",
}


GdalEnvSnapshot = dict[str, Optional[str]]


def apply_gdal_network_defaults() -> GdalEnvSnapshot:
    """Set network-tuned GDAL env vars, without overwriting any the user set.

    These are process-wide environment variables. They are intentionally not
    applied at import time because they affect every GDAL user in the process.
    Returns the previous values for the managed keys so callers can restore
    the process environment with ``restore_gdal_network_env``.
    """
    previous = {key: os.environ.get(key) for key in GDAL_NETWORK_DEFAULTS}
    if os.environ.get("S2MOSAIC_NO_GDAL_DEFAULTS", "").lower() in ("1", "true", "yes"):
        return previous
    for key, value in GDAL_NETWORK_DEFAULTS.items():
        os.environ.setdefault(key, value)
    return previous


def restore_gdal_network_env(snapshot: GdalEnvSnapshot) -> None:
    """Restore GDAL env vars captured by ``apply_gdal_network_defaults``."""
    for key, value in snapshot.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@contextmanager
def fresh_remote_reads() -> Iterator[None]:
    """Open remote rasters without GDAL's ``/vsicurl/`` region cache.

    Use around a retry. When a range response is cut short (a flaky network
    closing the connection mid-body), GDAL keeps the short block in its
    process-wide ``/vsicurl/`` cache, and every later open of the same URL is
    served that block without a new request. A plain reopen therefore fails
    identically however many times it is retried -- libtiff reports the same
    ``got N bytes, expected M`` on each attempt. Reproduced against a local
    server that truncates responses: with stock GDAL settings and with
    ``GDAL_NETWORK_DEFAULTS``, retries issue no requests and never recover;
    under this context a retry refetches and returns identical pixels.

    ``CPL_VSIL_CURL_NON_CACHED`` is consulted when a file is opened, so a
    handle opened here stays uncached after the context exits, and the stale
    blocks are dropped for later opens too. GDAL splits the option on ``:``,
    which rules out naming a single ``https://`` URL; the ``/vsicurl/`` prefix
    covers every HTTP(S) COG.

    Where the setting applies depends on the thread that enters the context.
    ``rasterio.Env`` sets GDAL config options process-wide from the main
    thread, so a retry there (``n_workers=1``) turns the cache off for every
    thread's opens until it exits; that costs speed, not correctness. From any
    other thread the options are thread-local, so a retry on a streaming
    worker does not reach a thread pool it starts itself. Wrap the pool's
    function in ``propagate_remote_read_env`` so its opens skip the cache too.
    """
    with rio.Env(**{_NON_CACHED_KEY: "/vsicurl/"}):
        yield


def propagate_remote_read_env(fn: Callable[..., T]) -> Callable[..., T]:
    """Carry the calling thread's ``fresh_remote_reads`` into pool workers.

    Call it on the thread that submits the work, then submit the result.
    ``_fetch_one_ocm`` needs this: it retries on a streaming worker and opens
    its bands on a sub-pool. Without it, the sub-pool opens use the cache and
    every retry replays the truncated block. Returns ``fn`` unchanged when no
    uncached context is active, so first attempts add no overhead.
    """
    value = get_gdal_config(_NON_CACHED_KEY)
    if not value:
        return fn

    @functools.wraps(fn)
    def wrapped(*args: object, **kwargs: object) -> T:
        with rio.Env(**{_NON_CACHED_KEY: value}):
            return fn(*args, **kwargs)

    return wrapped
