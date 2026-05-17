"""Grid-id mode mosaic pipeline.

Architecture:
    Phase 1 — per-scene mask compute. For SCL: range-read the SCL band and
    apply the SCL-class rules. For OCM: download R+G+NIR at OCM resolution,
    run inference, get the (clear, valid) mask. Both produce a single
    ``(s2_scene_size, s2_scene_size)`` bool combo mask per scene.

    Phase 2 — tile-streamed aggregation. The output is partitioned into
    ``tile_size``-square tiles. For each tile in parallel, the worker
    range-reads the corresponding window from each scene's user-requested
    band COGs (resampling at read time for 20m bands), applies the per-tile
    slice of the precomputed mask, and aggregates by method:

        * mean: weighted sum / valid-pixel count
        * first: take the first scene with a valid pixel at each location
        * percentile: nanquantile across all valid scenes

    Peak working set per worker is ``n_scenes_in_tile * bands * tile^2 * 4``
    bytes (a few hundred MB at typical sizes), so total RAM stays low even
    for 34+ scene full-MGRS percentile mosaics that previously needed 65 GB.

OCM inference still requires the full R+G+NIR per scene (the deep-learning
model has receptive field, so tile-by-tile inference would not match
whole-scene inference). That's confined to Phase 1; Phase 2 tile-streams
the user bands.
"""

import hashlib
import logging
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, TypeVar, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import planetary_computer
import rasterio as rio
from numba import njit
from rasterio.errors import RasterioIOError
from rasterio.enums import Resampling
from rasterio.windows import Window
from tqdm.auto import tqdm

from .helpers import (
    CLOUD_MASK_OCM,
    CLOUD_MASK_SCL,
    DEBUG_CACHE_DIR,
    MOSAIC_FIRST,
    MOSAIC_MEAN,
    MOSAIC_PERCENTILE,
    SceneFetchError,
    debug_cache_enabled,
    get_band_template,
    get_rasterio_resampling,
    output_band_metadata,
    pick_ocm_resolution,
)
from .masking import get_masks, get_scl_masks
from .stac_utils import ITEM_COL

logger = logging.getLogger(__name__)
REMOTE_READ_ATTEMPTS = 3
DEFAULT_OUTPUT_DTYPE = np.dtype(np.uint16)
DEFAULT_TILE_WORKERS = min(4, os.cpu_count() or 1)
DEFAULT_ADAPTIVE_TILE_MIN_SIZE = 512
DEFAULT_ADAPTIVE_TILE_DENSE_FRACTION = 0.75
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


def _materialise_grid_band(
    item: Any,
    asset_name: str,
    s2_scene_size: int,
    rio_resampling: Resampling,
) -> Callable[[Path], None]:
    """Build a materialiser for a grid_id-mode band cache entry.

    Returns a closure that, given an output path, downloads the full asset
    from PC and writes it as a tiled GeoTIFF on the MGRS grid at the user's
    output resolution.
    """

    def write(tmp_path: Path) -> None:
        signed = planetary_computer.sign(item.assets[asset_name].href)
        with rio.open(signed) as src:
            n_bands = src.count
            scale_x = src.width / s2_scene_size
            scale_y = src.height / s2_scene_size
            transform = src.transform * rio.Affine.scale(scale_x, scale_y)
            profile = {
                "driver": "GTiff",
                "count": n_bands,
                "dtype": src.dtypes[0],
                "width": s2_scene_size,
                "height": s2_scene_size,
                "crs": src.crs,
                "transform": transform,
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512,
                "compress": "lzw",
                "BIGTIFF": "IF_SAFER",
            }

            window_cls: Any = Window

            def source_window_for(dst_window: Window) -> Window:
                return window_cls(
                    dst_window.col_off * scale_x,
                    dst_window.row_off * scale_y,
                    dst_window.width * scale_x,
                    dst_window.height * scale_y,
                )

            _write_tiled_copy(src, tmp_path, profile, rio_resampling, source_window_for)

    return write


def _compute_one_scene_mask(
    item: Any,
    cloud_mask: str,
    ocm_batch_size: int,
    ocm_inference_dtype: str,
    ocm_resolution: int,
    max_dl_workers: int,
    s2_scene_size: int,
    resolution: int,
) -> Optional[npt.NDArray[Any]]:
    """Phase-1 worker: return the per-scene combo mask or None on fetch error."""
    try:
        if cloud_mask == CLOUD_MASK_SCL:
            clear, valid = get_scl_masks(item=item, user_resolution=resolution)
        else:
            clear, valid = get_masks(
                item=item,
                batch_size=ocm_batch_size,
                inference_dtype=ocm_inference_dtype,
                max_dl_workers=max_dl_workers,
                target_size=s2_scene_size,
                ocm_resolution=ocm_resolution,
            )
    except SceneFetchError as e:
        logger.warning("Mask fetch failed for %s, skipping (%s)", item.id, e)
        return None
    combo: npt.NDArray[Any] = (clear & valid).astype(np.bool_)
    return combo


def _build_output_profile(
    sample_href_signed: str, s2_scene_size: int
) -> Dict[str, Any]:
    """Build a rasterio profile snapped to the s2_scene_size output grid."""
    with rio.open(sample_href_signed) as src:
        profile = src.profile.copy()
        scale_x = src.width / s2_scene_size
        scale_y = src.height / s2_scene_size
        profile["transform"] = src.transform * rio.Affine.scale(scale_x, scale_y)
        profile["width"] = s2_scene_size
        profile["height"] = s2_scene_size
    return profile  # type: ignore[no-any-return, unused-ignore]


class _HandleCache:
    """Per-thread cache of open rasterio handles, lazy per (scene, band).

    rasterio's DatasetReader is not safe to share across threads — so each
    worker thread keeps its own dictionary. Handles open on first use of
    a given (scene, band) so workers that only touch a subset of scenes
    don't pay the open cost for the rest.
    """

    def __init__(self, source_resolvers: List[List[Callable[[], str]]]):
        # source_resolvers[scene_idx][band_idx]() -> local cache path or signed URL
        self._source_resolvers = source_resolvers
        self._local = threading.local()

    def get(self, scene_idx: int, band_idx: int) -> rio.DatasetReader:
        per_thread = getattr(self._local, "handles", None)
        if per_thread is None:
            per_thread = {}
            self._local.handles = per_thread
        key = (scene_idx, band_idx)
        h = per_thread.get(key)
        if h is None:
            h = rio.open(self._source_resolvers[scene_idx][band_idx]())
            per_thread[key] = h
        return h


def _read_tile_window(
    src: rio.DatasetReader,
    raster_band_idx: int,
    spec: Tuple[int, int, int, int],
    s2_scene_size: int,
    rio_resampling: Resampling,
) -> npt.NDArray[Any]:
    """Read one tile window from a COG, resampling to the target 10m grid."""
    r, c, h, w = spec
    scale_x = src.width / s2_scene_size
    scale_y = src.height / s2_scene_size
    window_cls: Any = Window
    src_window = window_cls(
        c * scale_x,
        r * scale_y,
        w * scale_x,
        h * scale_y,
    )
    return src.read(  # type: ignore[no-any-return, unused-ignore]
        raster_band_idx,
        window=src_window,
        out_shape=(h, w),
        resampling=rio_resampling,
    )


# Reader function shared by grid_id and bounds streamers.
# Signature: read_fn(scene_idx, band_idx, spec) -> ndarray of shape (h, w).
# Implementations close over their own source-handle cache (HandleCache for
# grid_id COG reads, a WarpedVRT cache for bounds).
ReaderFn = Callable[[int, int, Tuple[int, int, int, int]], npt.NDArray[Any]]


def _empty_tile(
    spec: Tuple[int, int, int, int], bands_count: int, out_dtype: "np.dtype[Any]"
) -> npt.NDArray[Any]:
    _, _, h, w = spec
    return np.zeros((bands_count, h, w), dtype=out_dtype)


def _finalise_tile(
    arr: npt.NDArray[Any], out_dtype: "np.dtype[Any]"
) -> npt.NDArray[Any]:
    """Clip + cast a tile result so workers return the pipeline's output dtype.

    Doing the cast per tile lets ``run_tile_aggregation`` allocate ``out`` as
    the final dtype, which halves the output buffer footprint for non-visual
    mosaics (uint16 instead of float32) and is essentially free overhead
    per tile (a clip + a cast).
    """
    if np.issubdtype(out_dtype, np.unsignedinteger):
        info = np.iinfo(out_dtype)
        return np.clip(arr, info.min, info.max).astype(out_dtype, copy=False)  # type: ignore[no-any-return, unused-ignore]
    return arr.astype(out_dtype, copy=False)  # type: ignore[no-any-return, unused-ignore]


@njit(cache=True)  # type: ignore[untyped-decorator]
def _nanquantile_axis0(stack: npt.NDArray[Any], q: float) -> npt.NDArray[Any]:
    """Serial NaN-skipping quantile over stack axis 0.

    ``stack`` shape is ``(scene, band, height, width)``. This is intentionally
    specialised to the tile aggregation hot path: scene counts are small, so a
    per-pixel insertion sort avoids allocations and is faster than a generic
    quantile implementation.

    This kernel deliberately avoids Numba's parallel mode. Numba's default
    ``workqueue`` threading layer is not safe to enter concurrently from
    several Python threads, and users may also call ``mosaic`` from their own
    thread pools. Tile-level concurrency supplies the parallelism instead.
    """
    n_scenes, n_bands, height, width = stack.shape
    out = np.empty((n_bands, height, width), dtype=np.float32)
    total = n_bands * height * width

    for idx in range(total):
        values = np.empty(n_scenes, dtype=np.float32)
        band = idx // (height * width)
        rem = idx - band * height * width
        row = rem // width
        col = rem - row * width

        n_valid = 0
        for scene_idx in range(n_scenes):
            value = stack[scene_idx, band, row, col]
            if not np.isnan(value):
                values[n_valid] = value
                n_valid += 1

        if n_valid == 0:
            out[band, row, col] = np.nan
        elif n_valid == 1:
            out[band, row, col] = values[0]
        else:
            for i in range(1, n_valid):
                key = values[i]
                j = i - 1
                while j >= 0 and values[j] > key:
                    values[j + 1] = values[j]
                    j -= 1
                values[j + 1] = key

            q32 = np.float32(q)
            pos = q32 * np.float32(n_valid - 1)
            lo = int(np.floor(pos))
            hi = int(np.ceil(pos))
            if lo == hi:
                out[band, row, col] = values[lo]
            else:
                frac = pos - lo
                out[band, row, col] = values[lo] + (values[hi] - values[lo]) * frac

    return out


def _warm_nanquantile_axis0() -> None:
    """Compile the Numba percentile kernel on the main thread.

    Letting the first call happen inside the worker pool can make multiple
    threads enter Numba's compilation path at once, which is fragile on macOS.
    A tiny warm call here pays the compile cost before the pool starts and keeps
    workers on the already-compiled execution path.
    """
    sample = np.array([[[[0.0]]], [[[1.0]]]], dtype=np.float32)
    _nanquantile_axis0(sample, 0.5)


def _copy_single_scene_tile(
    spec: Tuple[int, int, int, int],
    mask_tile: npt.NDArray[Any],
    read_fn: ReaderFn,
    scene_idx: int,
    bands_count: int,
    out_dtype: "np.dtype[Any]",
) -> npt.NDArray[Any]:
    """Copy one contributing scene into an output tile, zeroing masked pixels."""
    _, _, h, w = spec
    out = np.zeros((bands_count, h, w), dtype=out_dtype)
    for j in range(bands_count):
        data = read_fn(scene_idx, j, spec)
        np.copyto(out[j], data, where=mask_tile, casting="unsafe")
    return out


def _contributing_scene_indices(
    spec: Tuple[int, int, int, int],
    masks: List[Optional[npt.NDArray[Any]]],
    tile_coverage: npt.NDArray[Any],
    tile_observation_target: Optional[int],
) -> List[int]:
    """Scene indices that contribute to a tile before the observation target."""
    r, c, h, w = spec
    contributing: List[int] = []
    observation_count: Optional[npt.NDArray[Any]] = None
    if tile_observation_target is not None:
        observation_count = np.zeros((h, w), dtype=np.uint16)

    for scene_idx, m in enumerate(masks):
        if m is None:
            continue
        mask_tile = m[r : r + h, c : c + w]
        if not mask_tile.any():
            continue
        contributing.append(scene_idx)

        if observation_count is not None:
            np.add(
                observation_count,
                mask_tile & tile_coverage,
                out=observation_count,
                casting="unsafe",
            )
            if ((observation_count >= tile_observation_target) | ~tile_coverage).all():
                break

    return contributing


def tile_percentile(
    spec: Tuple[int, int, int, int],
    masks: List[Optional[npt.NDArray[Any]]],
    read_fn: ReaderFn,
    bands_count: int,
    percentile_value: float,
    coverage_mask: npt.NDArray[Any],
    tile_observation_target: Optional[int],
    out_dtype: "np.dtype[Any]",
) -> Tuple[Tuple[int, int, int, int], npt.NDArray[Any]]:
    r, c, h, w = spec
    tile_coverage = coverage_mask[r : r + h, c : c + w]
    if not tile_coverage.any():
        return spec, _empty_tile(spec, bands_count, out_dtype)

    contributing = _contributing_scene_indices(
        spec, masks, tile_coverage, tile_observation_target
    )

    if not contributing:
        return spec, _empty_tile(spec, bands_count, out_dtype)

    if len(contributing) == 1:
        scene_idx = contributing[0]
        mask = masks[scene_idx]
        assert mask is not None
        mask_tile = mask[r : r + h, c : c + w]
        return spec, _copy_single_scene_tile(
            spec, mask_tile, read_fn, scene_idx, bands_count, out_dtype
        )

    stack = np.empty((len(contributing), bands_count, h, w), dtype=np.float32)
    for k, scene_idx in enumerate(contributing):
        mask = masks[scene_idx]
        assert mask is not None
        mask_tile = mask[r : r + h, c : c + w]
        for j in range(bands_count):
            data = read_fn(scene_idx, j, spec)
            stack[k, j].fill(np.nan)
            np.copyto(stack[k, j], data, where=mask_tile, casting="unsafe")

    res = _nanquantile_axis0(stack, percentile_value / 100.0)
    res = np.nan_to_num(res, nan=0.0)
    return spec, _finalise_tile(res, out_dtype)


def tile_mean(
    spec: Tuple[int, int, int, int],
    masks: List[Optional[npt.NDArray[Any]]],
    read_fn: ReaderFn,
    bands_count: int,
    coverage_mask: npt.NDArray[Any],
    tile_observation_target: Optional[int],
    out_dtype: "np.dtype[Any]",
) -> Tuple[Tuple[int, int, int, int], npt.NDArray[Any]]:
    r, c, h, w = spec
    tile_coverage = coverage_mask[r : r + h, c : c + w]
    if not tile_coverage.any():
        return spec, _empty_tile(spec, bands_count, out_dtype)

    contributing = _contributing_scene_indices(
        spec, masks, tile_coverage, tile_observation_target
    )

    if not contributing:
        return spec, _empty_tile(spec, bands_count, out_dtype)

    if len(contributing) == 1:
        scene_idx = contributing[0]
        mask = masks[scene_idx]
        assert mask is not None
        mask_tile = mask[r : r + h, c : c + w]
        return spec, _copy_single_scene_tile(
            spec, mask_tile, read_fn, scene_idx, bands_count, out_dtype
        )

    sum_block = np.zeros((bands_count, h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.uint16)
    for scene_idx in contributing:
        mask = masks[scene_idx]
        assert mask is not None
        mask_tile = mask[r : r + h, c : c + w]
        for j in range(bands_count):
            data = read_fn(scene_idx, j, spec)
            np.add(sum_block[j], data, out=sum_block[j], where=mask_tile)
        np.add(count, mask_tile, out=count, casting="unsafe")
    result = np.divide(sum_block, count, out=np.zeros_like(sum_block), where=count != 0)
    return spec, _finalise_tile(result, out_dtype)


def tile_first(
    spec: Tuple[int, int, int, int],
    masks: List[Optional[npt.NDArray[Any]]],
    read_fn: ReaderFn,
    bands_count: int,
    coverage_mask: npt.NDArray[Any],
    no_data_threshold: Optional[float],
    out_dtype: "np.dtype[Any]",
) -> Tuple[Tuple[int, int, int, int], npt.NDArray[Any]]:
    r, c, h, w = spec
    tile_coverage = coverage_mask[r : r + h, c : c + w]
    if not tile_coverage.any():
        return spec, _empty_tile(spec, bands_count, out_dtype)
    # FIRST copies source pixels straight through, so we can accumulate
    # directly in the output dtype — no float32 working buffer needed.
    result = np.zeros((bands_count, h, w), dtype=out_dtype)
    filled = np.zeros((h, w), dtype=bool)
    for scene_idx, m in enumerate(masks):
        if m is None:
            continue
        mask_tile = m[r : r + h, c : c + w]
        new_pixels = mask_tile & ~filled
        if not new_pixels.any():
            continue
        for j in range(bands_count):
            data = read_fn(scene_idx, j, spec)
            result[j][new_pixels] = data[new_pixels]
        filled |= new_pixels
        if (filled | ~tile_coverage).all():
            break
    return spec, result


def tile_specs_for(
    height: int, width: int, tile_size: int
) -> List[Tuple[int, int, int, int]]:
    specs: List[Tuple[int, int, int, int]] = []
    for r in range(0, height, tile_size):
        for c in range(0, width, tile_size):
            h = min(tile_size, height - r)
            w = min(tile_size, width - c)
            specs.append((r, c, h, w))
    return specs


def _split_tile_size_aligned(length: int, min_tile_size: int) -> List[int]:
    """Split a sparse tile dimension on a min-tile multiple where possible."""
    if length <= min_tile_size:
        return [length]

    midpoint = length / 2
    split = round(midpoint / min_tile_size) * min_tile_size
    split = max(min_tile_size, min(split, length - min_tile_size))
    if split <= 0 or split >= length:
        return [length]
    return [split, length - split]


def adaptive_tile_specs_for_masks(
    masks: List[Optional[npt.NDArray[Any]]],
    height: int,
    width: int,
    max_tile_size: int,
    min_tile_size: int = DEFAULT_ADAPTIVE_TILE_MIN_SIZE,
    dense_fraction: float = DEFAULT_ADAPTIVE_TILE_DENSE_FRACTION,
) -> List[Tuple[int, int, int, int]]:
    """Mixed-size tile specs based on where any scene can contribute pixels."""
    specs: List[Tuple[int, int, int, int]] = []

    def contribution_fraction(r: int, c: int, h: int, w: int) -> float:
        combined = np.zeros((h, w), dtype=bool)
        for mask in masks:
            if mask is not None:
                combined |= mask[r : r + h, c : c + w]
        return float(combined.sum()) / float(h * w)

    def add_tile(r: int, c: int, h: int, w: int) -> None:
        fraction = contribution_fraction(r, c, h, w)
        if fraction == 0.0:
            return
        if fraction >= dense_fraction or (h <= min_tile_size and w <= min_tile_size):
            specs.append((r, c, h, w))
            return

        row_sizes = _split_tile_size_aligned(h, min_tile_size)
        col_sizes = _split_tile_size_aligned(w, min_tile_size)
        rr = r
        for rh in row_sizes:
            cc = c
            for cw in col_sizes:
                add_tile(rr, cc, rh, cw)
                cc += cw
            rr += rh

    for spec in tile_specs_for(height, width, max_tile_size):
        add_tile(*spec)
    return specs


def _expected_reads_upper_bound(
    masks: List[Optional[npt.NDArray[Any]]],
    specs: List[Tuple[int, int, int, int]],
    bands_count: int,
) -> int:
    """Upper bound on Phase 2 ``read_fn`` calls.

    Counts, for each tile spec, the scenes whose mask intersects that tile,
    times the number of user bands. ``first`` and ``tile_observation_target``
    can stop reading mid-tile, so the actual count may be lower — that's
    fine for the progress bar; we just won't naturally hit 100% in those
    cases and fast-forward at the end.
    """
    total = 0
    for r, c, h, w in specs:
        n_contrib = 0
        for m in masks:
            if m is None:
                continue
            if m[r : r + h, c : c + w].any():
                n_contrib += 1
        total += n_contrib * bands_count
    return total


def run_tile_aggregation(
    masks: List[Optional[npt.NDArray[Any]]],
    read_fn: ReaderFn,
    bands_count: int,
    height: int,
    width: int,
    coverage_mask: npt.NDArray[Any],
    no_data_threshold: Optional[float],
    mosaic_method: str,
    percentile_value: Optional[float],
    tile_size: int,
    tile_workers: Optional[int],
    out_dtype: "np.dtype[Any]" = DEFAULT_OUTPUT_DTYPE,
    tile_observation_target: Optional[int] = None,
    adaptive_tiling: bool = True,
    tile_specs: Optional[List[Tuple[int, int, int, int]]] = None,
    show_progress: bool = False,
) -> npt.NDArray[Any]:
    """Generic streaming aggregation. Called by both grid_id and bounds modes.

    ``out_dtype`` is the pipeline's final output dtype (``uint16`` for
    spectral, ``uint8`` for visual). Tile workers cast to it before
    returning, so the output buffer can be allocated as the final dtype —
    no intermediate float32 array the size of the whole mosaic.
    """
    out = np.zeros((bands_count, height, width), dtype=out_dtype)
    for spec, tile_data in iter_tile_aggregation(
        masks=masks,
        read_fn=read_fn,
        bands_count=bands_count,
        height=height,
        width=width,
        coverage_mask=coverage_mask,
        no_data_threshold=no_data_threshold,
        mosaic_method=mosaic_method,
        percentile_value=percentile_value,
        tile_size=tile_size,
        tile_workers=tile_workers,
        out_dtype=out_dtype,
        tile_observation_target=tile_observation_target,
        adaptive_tiling=adaptive_tiling,
        tile_specs=tile_specs,
        show_progress=show_progress,
    ):
        r, c, h, w = spec
        out[:, r : r + h, c : c + w] = tile_data
    return out


def iter_tile_aggregation(
    masks: List[Optional[npt.NDArray[Any]]],
    read_fn: ReaderFn,
    bands_count: int,
    height: int,
    width: int,
    coverage_mask: npt.NDArray[Any],
    no_data_threshold: Optional[float],
    mosaic_method: str,
    percentile_value: Optional[float],
    tile_size: int,
    tile_workers: Optional[int],
    out_dtype: "np.dtype[Any]" = DEFAULT_OUTPUT_DTYPE,
    tile_observation_target: Optional[int] = None,
    adaptive_tiling: bool = True,
    tile_specs: Optional[List[Tuple[int, int, int, int]]] = None,
    show_progress: bool = False,
) -> Iterator[Tuple[Tuple[int, int, int, int], npt.NDArray[Any]]]:
    """Yield aggregated output tiles without allocating the full mosaic."""
    if tile_specs is not None:
        specs = tile_specs
    elif adaptive_tiling:
        specs = adaptive_tile_specs_for_masks(
            masks=masks,
            height=height,
            width=width,
            max_tile_size=tile_size,
        )
    else:
        specs = tile_specs_for(height, width, tile_size)

    # Phase 2 progress is per band-read rather than per tile so the bar
    # advances smoothly. Total is the upper bound — each (scene, band) read
    # that *would* happen if no early-stop kicks in. ``first`` / observation
    # target modes may finish below 100%, which we fast-forward at the end.
    progress_bar: Optional["tqdm[Any]"] = None
    effective_read_fn = read_fn
    if show_progress:
        total_reads = _expected_reads_upper_bound(masks, specs, bands_count)
        if total_reads > 0:
            progress_bar = tqdm(
                total=total_reads,
                desc=f"Phase 2: aggregating tiles ({mosaic_method})",
                unit="read",
            )
            base_read_fn = read_fn
            _pb = progress_bar

            def _counting_read_fn(
                scene_idx: int,
                band_idx: int,
                spec: Tuple[int, int, int, int],
            ) -> npt.NDArray[Any]:
                result = base_read_fn(scene_idx, band_idx, spec)
                _pb.update(1)
                return result

            effective_read_fn = _counting_read_fn

    if mosaic_method == MOSAIC_PERCENTILE:
        _warm_nanquantile_axis0()
        pv = percentile_value if percentile_value is not None else 50.0
        n_workers = tile_workers if tile_workers is not None else DEFAULT_TILE_WORKERS

        def worker_fn(
            s: Tuple[int, int, int, int],
        ) -> Tuple[Tuple[int, int, int, int], npt.NDArray[Any]]:
            return tile_percentile(
                s,
                masks,
                effective_read_fn,
                bands_count,
                pv,
                coverage_mask,
                tile_observation_target,
                out_dtype,
            )

    elif mosaic_method == MOSAIC_MEAN:
        n_workers = tile_workers if tile_workers is not None else DEFAULT_TILE_WORKERS

        def worker_fn(
            s: Tuple[int, int, int, int],
        ) -> Tuple[Tuple[int, int, int, int], npt.NDArray[Any]]:
            return tile_mean(
                s,
                masks,
                effective_read_fn,
                bands_count,
                coverage_mask,
                tile_observation_target,
                out_dtype,
            )

    elif mosaic_method == MOSAIC_FIRST:
        n_workers = tile_workers if tile_workers is not None else DEFAULT_TILE_WORKERS

        def worker_fn(
            s: Tuple[int, int, int, int],
        ) -> Tuple[Tuple[int, int, int, int], npt.NDArray[Any]]:
            return tile_first(
                s,
                masks,
                effective_read_fn,
                bands_count,
                coverage_mask,
                no_data_threshold,
                out_dtype,
            )

    else:
        raise ValueError(f"Unknown mosaic_method: {mosaic_method}")

    completed = 0
    log_every = max(1, len(specs) // 10)
    try:
        if n_workers <= 1:
            for spec, tile_data in map(worker_fn, specs):
                completed += 1
                if completed % log_every == 0 or completed == len(specs):
                    logger.info("Phase 2: %d/%d tiles done", completed, len(specs))
                yield spec, tile_data
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                for spec, tile_data in ex.map(worker_fn, specs):
                    completed += 1
                    if completed % log_every == 0 or completed == len(specs):
                        logger.info("Phase 2: %d/%d tiles done", completed, len(specs))
                    yield spec, tile_data
    finally:
        if progress_bar is not None:
            # Early-stop modes (first, observation target) skip reads, so the
            # bar may not have reached total. Snap to total so it shows done.
            remaining = progress_bar.total - progress_bar.n
            if remaining > 0:
                progress_bar.update(remaining)
            progress_bar.close()


def write_tile_aggregation_geotiff(
    export_path: Path,
    profile: Dict[str, Any],
    required_bands: List[str],
    masks: List[Optional[npt.NDArray[Any]]],
    read_fn: ReaderFn,
    bands_count: int,
    height: int,
    width: int,
    coverage_mask: npt.NDArray[Any],
    output_coverage_mask: Optional[npt.NDArray[Any]],
    no_data_threshold: Optional[float],
    mosaic_method: str,
    percentile_value: Optional[float],
    tile_size: int,
    tile_workers: Optional[int],
    out_dtype: "np.dtype[Any]" = DEFAULT_OUTPUT_DTYPE,
    tile_observation_target: Optional[int] = None,
    adaptive_tiling: bool = True,
    tile_specs: Optional[List[Tuple[int, int, int, int]]] = None,
    show_progress: bool = False,
) -> Path:
    """Aggregate tiles and write them directly into a GeoTIFF."""
    band_descriptions, nodata_value = output_band_metadata(required_bands)
    write_profile = profile.copy()
    write_profile.update(
        driver="GTiff",
        width=width,
        height=height,
        count=bands_count,
        dtype=out_dtype,
        nodata=nodata_value,
        compress="lzw",
    )
    logger.info("Writing streamed GeoTIFF to %s", export_path)
    with rio.open(export_path, "w", **write_profile) as dst:
        dst.descriptions = band_descriptions
        for spec, tile_data in iter_tile_aggregation(
            masks=masks,
            read_fn=read_fn,
            bands_count=bands_count,
            height=height,
            width=width,
            coverage_mask=coverage_mask,
            no_data_threshold=no_data_threshold,
            mosaic_method=mosaic_method,
            percentile_value=percentile_value,
            tile_size=tile_size,
            tile_workers=tile_workers,
            out_dtype=out_dtype,
            tile_observation_target=tile_observation_target,
            adaptive_tiling=adaptive_tiling,
            tile_specs=tile_specs,
            show_progress=show_progress,
        ):
            r, c, h, w = spec
            if output_coverage_mask is not None:
                coverage_tile = output_coverage_mask[r : r + h, c : c + w]
                np.multiply(
                    tile_data,
                    coverage_tile[None, :, :],
                    out=tile_data,
                    casting="unsafe",
                )
            dst.write(tile_data, window=Window(c, r, w, h))
    return export_path


def make_grid_tile_reader(
    items: List[Any],
    href_template: List[Tuple[str, int]],
    s2_scene_size: int,
    resolution: int,
    resampling_method: str,
    prewarm: bool = True,
) -> ReaderFn:
    """Build a tile-reader for grid_id mode (direct MGRS COG reads).

    Builds lazy source resolvers for every (scene, asset). Workers in Phase 2
    open handles lazily per thread; if ``S2MOSAIC_DEBUG_CACHE`` is on, the
    cache file is materialised only when a tile actually reads that source.

    With ``prewarm=True`` (default) the resolvers are called in parallel
    once before returning, so cache materialisation happens in one parallel
    burst rather than being fanned out serially by the per-tile workers.
    Set ``prewarm=False`` to keep the resolvers strictly lazy (useful for
    tests, or for callers that expect to skip many reads via early stopping).
    """
    rio_resampling = get_rasterio_resampling(resampling_method)
    href_band_indices = [band_idx for _, band_idx in href_template]

    sources: List[List[Callable[[], str]]] = []
    for item in items:
        scene_sources: List[Callable[[], str]] = []
        for asset, _ in href_template:
            cache_key = (
                f"grid|{item.id}|{asset}|{s2_scene_size}|"
                f"{resolution}|{resampling_method}"
            )
            signed_url = planetary_computer.sign(item.assets[asset].href)

            def source_for(
                item: Any = item,
                asset: str = asset,
                cache_key: str = cache_key,
                signed_url: str = signed_url,
            ) -> str:
                local = materialise_tiled_band(
                    cache_key,
                    _materialise_grid_band(item, asset, s2_scene_size, rio_resampling),
                )
                return str(local) if local is not None else signed_url

            scene_sources.append(source_for)
        sources.append(scene_sources)
    cache = _HandleCache(sources)

    if prewarm:
        # Pre-warm the per-(scene, asset) sources in parallel. When the debug
        # cache is on this materialises every cache entry up-front (much faster
        # than serially-on-first-read inside the tile loop); when caching is off
        # it's a cheap parallel URL-sign and adds negligible overhead.
        _prewarm_sources(sources)

    def read_fn(
        scene_idx: int, band_idx: int, spec: Tuple[int, int, int, int]
    ) -> npt.NDArray[Any]:
        src = cache.get(scene_idx, band_idx)
        return _read_tile_window(
            src, href_band_indices[band_idx], spec, s2_scene_size, rio_resampling
        )

    return read_fn


def _prewarm_sources(sources: List[List[Callable[..., Any]]]) -> None:
    """Call every lazy source resolver in parallel.

    Resolvers are no-ops when the debug cache is disabled (just URL signing).
    With cache enabled they trigger materialisation — pre-warming avoids the
    serial fan-out you'd otherwise get from tile workers each lazily
    materialising the (scene, asset) entries they touch.
    """
    flat: List[Callable[..., Any]] = [
        resolver for scene in sources for resolver in scene
    ]
    if not flat:
        return
    n_workers = min(len(flat), (os.cpu_count() or 8) * 2)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        # Drain any exceptions — the lazy path would raise on first read
        # anyway, but raising eagerly gives a clearer stack.
        for _ in ex.map(lambda fn: fn(), flat):
            pass


def should_prewarm_sources(
    mosaic_method: str,
    no_data_threshold: Optional[float],
    tile_observation_target: Optional[int] = None,
) -> bool:
    """Whether to pre-materialise tile sources before aggregation.

    Prewarming improves throughput when most scene/band sources will be read
    anyway. Keep sources lazy when the aggregation is likely to skip many reads:
    ``first`` mode can stop as pixels fill, ``no_data_threshold`` can stop
    scene walks before all scenes are touched, and ``tile_observation_target``
    can cap per-tile observations.
    """
    return (
        mosaic_method != MOSAIC_FIRST
        and no_data_threshold is None
        and tile_observation_target is None
    )


def stream_mosaic_pipeline(
    sorted_scenes: pd.DataFrame,
    required_bands: List[str],
    coverage_mask: npt.NDArray[Any],
    no_data_threshold: Union[float, None],
    tile_observation_target: Optional[int] = None,
    export_path: Optional[Path] = None,
    output_coverage_mask: Optional[npt.NDArray[Any]] = None,
    mosaic_method: str = "mean",
    ocm_batch_size: int = 6,
    ocm_inference_dtype: str = "bf16",
    max_dl_workers: int = 4,
    percentile_value: Optional[float] = 50.0,
    s2_scene_size: int = 10980,
    resampling_method: str = "nearest",
    resolution: int = 10,
    cloud_mask: str = CLOUD_MASK_OCM,
    tile_size: int = 2048,
    tile_workers: Optional[int] = None,
    adaptive_tiling: bool = True,
    show_progress: bool = False,
) -> Tuple[Optional[npt.NDArray[Any]], Dict[str, Any]]:
    """Tile-streamed mosaic for grid_id mode.

    Replaces the old in-memory ``download_bands_pool`` path. Peak working
    set is per-worker (a few hundred MB), so 34-scene full-MGRS percentile
    mosaics that previously needed ~65 GB of RAM now fit in a few GB.

    ``tile_observation_target`` is an optional per-tile early-stop target for
    ``mean`` and ``percentile``: each tile walks scenes in priority order and
    stops once every coverable pixel has at least that many valid observations.
    ``first`` always stops once every coverable pixel has its first observation.
    """
    ocm_resolution = pick_ocm_resolution(resolution)
    logger.info(f"OCM resolution: {ocm_resolution}m")
    possible_pixel_count = coverage_mask.sum()
    logger.info(f"Possible pixel count: {possible_pixel_count}")

    items: List[Any] = sorted_scenes[ITEM_COL].tolist()
    n_scenes = len(items)
    is_visual = "visual" in required_bands
    href_template, bands_count, _ = get_band_template(required_bands)

    # Phase 1: compute per-scene combo masks in sorted order with bounded
    # prefetch. This keeps early-stop decisions deterministic while allowing
    # downloads for later masks to overlap current-scene processing.
    phase1_workers = tile_workers if tile_workers is not None else DEFAULT_TILE_WORKERS
    mask_workers = max_dl_workers if cloud_mask == CLOUD_MASK_OCM else phase1_workers
    logger.info(
        "Phase 1: streaming masks for %d scenes (%s, workers=%d)",
        n_scenes,
        cloud_mask,
        mask_workers,
    )
    masks: List[Optional[npt.NDArray[Any]]] = [None] * n_scenes
    good_pixel_tracker = np.zeros_like(coverage_mask, dtype=bool)
    n_mask_fetch_failed = 0

    def _fetch_mask(idx: int, item: Any) -> Optional[npt.NDArray[Any]]:
        return _compute_one_scene_mask(
            item=item,
            cloud_mask=cloud_mask,
            ocm_batch_size=ocm_batch_size,
            ocm_inference_dtype=ocm_inference_dtype,
            ocm_resolution=ocm_resolution,
            max_dl_workers=max_dl_workers,
            s2_scene_size=s2_scene_size,
            resolution=resolution,
        )

    mask_progress: Optional["tqdm[Any]"] = None
    if show_progress:
        mask_progress = tqdm(
            total=n_scenes,
            desc=f"Phase 1: streaming cloud masks ({cloud_mask})",
            unit="scene",
        )
    _pb = mask_progress

    def _on_mask_complete(_i: int) -> None:
        if _pb is not None:
            _pb.update(1)

    mask_iter: Iterator[Tuple[int, Union[Optional[npt.NDArray[Any]], Exception]]]
    mask_iter = iter_ordered_fetches(
        items=items,
        fetch_fn=_fetch_mask,
        max_workers=mask_workers,
        on_complete=_on_mask_complete,
    )

    try:
        for scene_position in range(n_scenes):
            if (
                mosaic_method == MOSAIC_FIRST
                and (good_pixel_tracker | ~coverage_mask).all()
            ):
                logger.info(
                    "All in-coverage pixels filled after %d/%d scenes — "
                    "skipping remaining cloud-mask fetches",
                    scene_position,
                    n_scenes,
                )
                break
            try:
                scene_idx, combo_result = next(mask_iter)
            except StopIteration:
                break
            if isinstance(combo_result, Exception):
                n_mask_fetch_failed += 1
                logger.warning(
                    "Mask fetch failed for %s, skipping (%s)",
                    items[scene_idx].id,
                    combo_result,
                )
                continue
            combo = combo_result
            logger.info(
                "Phase 1: scene %d/%d (%s): %s",
                scene_idx + 1,
                n_scenes,
                items[scene_idx].id,
                "ok" if combo is not None else "skipped",
            )
            if combo is None:
                n_mask_fetch_failed += 1
                continue
            if mosaic_method == MOSAIC_FIRST:
                new_pixels = combo & ~good_pixel_tracker
                if not new_pixels.any():
                    continue
                combo = new_pixels
            elif not combo.any():
                continue
            masks[scene_idx] = combo
            good_pixel_tracker |= combo

            if (
                no_data_threshold is not None
                and mosaic_method != MOSAIC_PERCENTILE
                and possible_pixel_count > 0
            ):
                completed = int((coverage_mask & good_pixel_tracker).sum())
                no_data_sum = int(possible_pixel_count) - completed
                if no_data_sum < possible_pixel_count * no_data_threshold:
                    logger.info(
                        "no_data_threshold met after %d kept scenes (%d/%d examined)",
                        sum(1 for m in masks if m is not None),
                        scene_idx + 1,
                        n_scenes,
                    )
                    break
    finally:
        if mask_progress is not None:
            remaining = mask_progress.total - mask_progress.n
            if remaining > 0:
                mask_progress.update(remaining)
            mask_progress.close()

    n_succeeded = sum(1 for m in masks if m is not None)
    if n_mask_fetch_failed:
        logger.warning(
            "Phase 1: %d/%d scenes failed mask compute",
            n_mask_fetch_failed,
            n_scenes,
        )
    if n_succeeded == 0:
        raise RuntimeError(
            f"All {n_scenes} scenes failed to fetch masks — no data to mosaic"
        )

    # Pull a sample profile for output georeferencing. Any valid scene's
    # first band will do — they all snap to the same MGRS grid.
    sample_idx = next(i for i, m in enumerate(masks) if m is not None)
    first_asset, _ = href_template[0]
    sample_href = planetary_computer.sign(items[sample_idx].assets[first_asset].href)
    last_profile = _build_output_profile(sample_href, s2_scene_size)

    read_fn = make_grid_tile_reader(
        items=items,
        href_template=href_template,
        s2_scene_size=s2_scene_size,
        resolution=resolution,
        resampling_method=resampling_method,
        prewarm=should_prewarm_sources(
            mosaic_method, no_data_threshold, tile_observation_target
        ),
    )

    logger.info(
        "Phase 2: %s aggregation (tile=%d)",
        mosaic_method,
        tile_size,
    )
    out_dtype = np.dtype(np.uint8) if is_visual else np.dtype(np.uint16)
    last_profile["dtype"] = out_dtype
    last_profile["count"] = bands_count

    if export_path is not None:
        write_tile_aggregation_geotiff(
            export_path=export_path,
            profile=last_profile,
            required_bands=required_bands,
            masks=masks,
            read_fn=read_fn,
            bands_count=bands_count,
            height=s2_scene_size,
            width=s2_scene_size,
            coverage_mask=coverage_mask,
            output_coverage_mask=output_coverage_mask,
            no_data_threshold=no_data_threshold,
            tile_observation_target=tile_observation_target,
            mosaic_method=mosaic_method,
            percentile_value=percentile_value,
            tile_size=tile_size,
            tile_workers=tile_workers,
            out_dtype=out_dtype,
            adaptive_tiling=adaptive_tiling,
            show_progress=show_progress,
        )
        return None, last_profile

    out = run_tile_aggregation(
        masks=masks,
        read_fn=read_fn,
        bands_count=bands_count,
        height=s2_scene_size,
        width=s2_scene_size,
        coverage_mask=coverage_mask,
        no_data_threshold=no_data_threshold,
        tile_observation_target=tile_observation_target,
        mosaic_method=mosaic_method,
        percentile_value=percentile_value,
        tile_size=tile_size,
        tile_workers=tile_workers,
        out_dtype=out_dtype,
        adaptive_tiling=adaptive_tiling,
        show_progress=show_progress,
    )
    return out, last_profile
