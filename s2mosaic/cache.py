"""Debug cache and concurrent fetch helpers."""

import hashlib
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, TypeVar, Union

import numpy.typing as npt
import rasterio as rio
from rasterio.errors import RasterioIOError
from rasterio.enums import Resampling
from rasterio.windows import Window

from .helpers import DEBUG_CACHE_DIR, debug_cache_enabled

REMOTE_READ_ATTEMPTS = 3
T = TypeVar("T")


def _tiled_gtiff_cache_path(cache_key: str) -> Path:
    digest = hashlib.md5(cache_key.encode()).hexdigest()
    return DEBUG_CACHE_DIR / f"tiled_band_{digest}.tif"


# Per-key locks deduplicate concurrent materialise calls for the same
# (scene, asset). Without this, every tile-worker that needs a (scene, asset)
# for its first tile races on the same cache entry, runs the WarpedVRT
# concurrently, and writes the same output N times.
_materialise_locks: Dict[str, threading.Lock] = {}
_materialise_locks_guard = threading.Lock()


def _get_materialise_lock(cache_key: str) -> threading.Lock:
    with _materialise_locks_guard:
        lock = _materialise_locks.get(cache_key)
        if lock is None:
            lock = threading.Lock()
            _materialise_locks[cache_key] = lock
        return lock


def iter_ordered_fetches(
    items: List[Any],
    fetch_fn: Callable[[int, Any], T],
    max_workers: int,
    on_complete: Optional[Callable[[int], None]] = None,
) -> Iterator[Tuple[int, Union[T, Exception]]]:
    """Fetch items concurrently while yielding results in input order.

    The next fetch is submitted before yielding each completed result so
    caller-side processing, such as OCM inference, can overlap with downloads
    for later scenes. Exceptions are yielded in-order for the caller to handle.

    ``on_complete`` fires once per item as soon as that item's fetch finishes,
    regardless of yield order. Use it to drive a progress bar so it ticks per
    completion instead of jumping when the in-order yields catch up — the
    slowest in-flight fetch otherwise blocks all earlier-completed yields.
    """
    n_items = len(items)
    n_workers = min(max(1, max_workers), n_items)

    def _do_fetch(i: int, item: Any) -> T:
        try:
            return fetch_fn(i, item)
        finally:
            if on_complete is not None:
                on_complete(i)

    if n_workers <= 1:
        for i, item in enumerate(items):
            try:
                yield i, _do_fetch(i, item)
            except Exception as e:
                yield i, e
        return

    executor = ThreadPoolExecutor(max_workers=n_workers)
    futures: Dict[int, Future[T]] = {}
    next_submit = 0

    def _submit_next() -> None:
        nonlocal next_submit
        i = next_submit
        futures[i] = executor.submit(_do_fetch, i, items[i])
        next_submit += 1

    try:
        for _ in range(n_workers):
            _submit_next()
        for next_yield in range(n_items):
            future = futures.pop(next_yield)
            try:
                result: Union[T, Exception] = future.result()
            except Exception as e:
                result = e
            if next_submit < n_items:
                _submit_next()
            yield next_yield, result
    finally:
        for future in futures.values():
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)


def materialise_tiled_band(
    cache_key: str,
    materialiser: Callable[[Path], None],
) -> Optional[Path]:
    """If debug cache is enabled, materialise the band-as-tiled-GeoTIFF once.

    Returns the local cache path that the reader should open instead of the
    PC URL. Returns None when caching is disabled — caller falls back to the
    direct streaming read path. ``materialiser`` is mode-specific and should
    write a tiled GeoTIFF on the target grid.

    Safe to call from multiple threads with the same key: a per-key lock
    serialises the write, and the second caller sees the materialised file
    on its re-check.
    """
    if not debug_cache_enabled():
        return None
    cache_path = _tiled_gtiff_cache_path(cache_key)
    if cache_path.exists():
        return cache_path
    with _get_materialise_lock(cache_key):
        # Re-check inside the lock — another thread may have materialised
        # it while we waited.
        if cache_path.exists():
            return cache_path
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(
            f".tmp.{os.getpid()}.{threading.get_ident()}.tif"
        )
        try:
            materialiser(tmp_path)
            tmp_path.rename(cache_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
    return cache_path


def _read_with_retry(
    src: rio.DatasetReader,
    *,
    window: Window,
    out_shape: Tuple[int, int, int],
    resampling: Resampling,
    attempts: int = REMOTE_READ_ATTEMPTS,
) -> npt.NDArray[Any]:
    """Read one source window, retrying transient remote COG tile failures."""
    last_error: RasterioIOError | None = None
    for attempt in range(attempts):
        try:
            return src.read(window=window, out_shape=out_shape, resampling=resampling)  # type: ignore[no-any-return, unused-ignore]
        except RasterioIOError as exc:
            last_error = exc
            if attempt == attempts - 1:
                break
            time.sleep(0.5 * (attempt + 1))
    if last_error is None:
        raise RuntimeError("Remote read was not attempted")
    raise last_error


def _write_tiled_copy(
    src: rio.DatasetReader,
    tmp_path: Path,
    profile: Dict[str, Any],
    rio_resampling: Resampling,
    source_window_for: Callable[[Window], Window],
) -> None:
    """Write a local tiled GeoTIFF by streaming destination blocks."""
    with rio.open(tmp_path, "w", **profile) as dst:
        for _, dst_window in dst.block_windows(1):
            data = _read_with_retry(
                src,
                window=source_window_for(dst_window),
                out_shape=(
                    profile["count"],
                    int(dst_window.height),
                    int(dst_window.width),
                ),
                resampling=rio_resampling,
            )
            dst.write(data, window=dst_window)
