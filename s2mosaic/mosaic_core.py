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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import planetary_computer
import rasterio as rio
from rasterio.errors import RasterioIOError
from numbagg import nanquantile
from rasterio.enums import Resampling
from rasterio.windows import Window

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
    pick_ocm_resolution,
)
from .masking import get_masks, get_scl_masks
from .stac_utils import ITEM_COL

logger = logging.getLogger(__name__)
REMOTE_READ_ATTEMPTS = 3
DEFAULT_OUTPUT_DTYPE = np.dtype(np.uint16)


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
) -> np.ndarray:
    """Read one source window, retrying transient remote COG tile failures."""
    last_error: RasterioIOError | None = None
    for attempt in range(attempts):
        try:
            return src.read(window=window, out_shape=out_shape, resampling=resampling)
        except RasterioIOError as exc:
            last_error = exc
            if attempt == attempts - 1:
                break
            time.sleep(0.5 * (attempt + 1))
    raise last_error  # type: ignore[misc]


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

            def source_window_for(dst_window: Window) -> Window:
                return Window(
                    dst_window.col_off * scale_x,
                    dst_window.row_off * scale_y,
                    dst_window.width * scale_x,
                    dst_window.height * scale_y,
                )

            _write_tiled_copy(src, tmp_path, profile, rio_resampling, source_window_for)

    return write


def _pin_numba_threads(n: int) -> None:
    """Cap numba's parallel pool. Called once on the main thread to prevent
    the worker-side corruption noted in the project memory.
    """
    try:
        from numba import set_num_threads

        set_num_threads(max(1, n))
    except Exception:
        pass


def _compute_one_scene_mask(
    item: Any,
    cloud_mask: str,
    ocm_batch_size: int,
    ocm_inference_dtype: str,
    ocm_resolution: int,
    max_dl_workers: int,
    s2_scene_size: int,
    resolution: int,
) -> Optional[np.ndarray]:
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
    combo = (clear & valid).astype(np.bool_)
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
    return profile


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
) -> np.ndarray:
    """Read one tile window from a COG, resampling to the target 10m grid."""
    r, c, h, w = spec
    scale_x = src.width / s2_scene_size
    scale_y = src.height / s2_scene_size
    src_window = Window(
        col_off=c * scale_x,
        row_off=r * scale_y,
        width=w * scale_x,
        height=h * scale_y,
    )
    return src.read(
        raster_band_idx,
        window=src_window,
        out_shape=(h, w),
        resampling=rio_resampling,
    )


def _tile_threshold_met(
    tile_filled: np.ndarray,
    tile_coverage: np.ndarray,
    no_data_threshold: Optional[float],
) -> bool:
    """Per-tile coverage short-circuit, mirroring the old global threshold.

    Returns True when uncovered-possible-pixels / possible-pixels is below
    ``no_data_threshold``. ``tile_coverage`` is the slice of the global
    coverage mask for this tile (pixels that CAN have data); ``tile_filled``
    is a bool array of pixels we've already contributed data to.
    """
    if no_data_threshold is None:
        return False
    possible = int(tile_coverage.sum())
    if possible == 0:
        return True  # no work to do here anyway
    filled = int((tile_filled & tile_coverage).sum())
    no_data = possible - filled
    return no_data < possible * no_data_threshold


# Reader function shared by grid_id and bounds streamers.
# Signature: read_fn(scene_idx, band_idx, spec) -> ndarray of shape (h, w).
# Implementations close over their own source-handle cache (HandleCache for
# grid_id COG reads, a WarpedVRT cache for bounds).
ReaderFn = Callable[[int, int, Tuple[int, int, int, int]], np.ndarray]


def _empty_tile(
    spec: Tuple[int, int, int, int], bands_count: int, out_dtype: np.dtype
) -> np.ndarray:
    _, _, h, w = spec
    return np.zeros((bands_count, h, w), dtype=out_dtype)


def _finalise_tile(arr: np.ndarray, out_dtype: np.dtype) -> np.ndarray:
    """Clip + cast a tile result so workers return the pipeline's output dtype.

    Doing the cast per tile lets ``run_tile_aggregation`` allocate ``out`` as
    the final dtype, which halves the output buffer footprint for non-visual
    mosaics (uint16 instead of float32) and is essentially free overhead
    per tile (a clip + a cast).
    """
    if np.issubdtype(out_dtype, np.unsignedinteger):
        info = np.iinfo(out_dtype)
        return np.clip(arr, info.min, info.max).astype(out_dtype, copy=False)
    return arr.astype(out_dtype, copy=False)


def _copy_single_scene_tile(
    spec: Tuple[int, int, int, int],
    mask_tile: np.ndarray,
    read_fn: ReaderFn,
    scene_idx: int,
    bands_count: int,
    out_dtype: np.dtype,
) -> np.ndarray:
    """Copy one contributing scene into an output tile, zeroing masked pixels."""
    _, _, h, w = spec
    out = np.zeros((bands_count, h, w), dtype=out_dtype)
    for j in range(bands_count):
        data = read_fn(scene_idx, j, spec)
        np.copyto(out[j], data, where=mask_tile, casting="unsafe")
    return out


def tile_percentile(
    spec: Tuple[int, int, int, int],
    masks: List[Optional[np.ndarray]],
    read_fn: ReaderFn,
    bands_count: int,
    percentile_value: float,
    coverage_mask: np.ndarray,
    no_data_threshold: Optional[float],
    out_dtype: np.dtype,
) -> Tuple[Tuple[int, int, int, int], np.ndarray]:
    r, c, h, w = spec
    tile_coverage = coverage_mask[r : r + h, c : c + w]
    if not tile_coverage.any():
        return spec, _empty_tile(spec, bands_count, out_dtype)

    # Pre-count scenes that actually contribute pixels in this tile so we
    # can allocate the stack once instead of building a list + np.stack-ing.
    # mask_tile.any() is cheap (bool reduction over h*w bytes) compared to
    # an extra h*w*bands*4-byte temporary per scene.
    contributing: List[int] = []
    filled = np.zeros((h, w), dtype=bool)
    for scene_idx, m in enumerate(masks):
        if m is None:
            continue
        mask_tile = m[r : r + h, c : c + w]
        if not mask_tile.any():
            continue
        contributing.append(scene_idx)
        filled |= mask_tile
        if _tile_threshold_met(filled, tile_coverage, no_data_threshold):
            break

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

    res = nanquantile(stack, percentile_value / 100.0, axis=0)
    res = np.nan_to_num(res, nan=0.0)
    return spec, _finalise_tile(res, out_dtype)


def tile_mean(
    spec: Tuple[int, int, int, int],
    masks: List[Optional[np.ndarray]],
    read_fn: ReaderFn,
    bands_count: int,
    coverage_mask: np.ndarray,
    no_data_threshold: Optional[float],
    out_dtype: np.dtype,
) -> Tuple[Tuple[int, int, int, int], np.ndarray]:
    r, c, h, w = spec
    tile_coverage = coverage_mask[r : r + h, c : c + w]
    if not tile_coverage.any():
        return spec, _empty_tile(spec, bands_count, out_dtype)

    contributing: List[int] = []
    filled = np.zeros((h, w), dtype=bool)
    for scene_idx, m in enumerate(masks):
        if m is None:
            continue
        mask_tile = m[r : r + h, c : c + w]
        if not mask_tile.any():
            continue
        contributing.append(scene_idx)
        filled |= mask_tile
        if _tile_threshold_met(filled, tile_coverage, no_data_threshold):
            break

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
    masks: List[Optional[np.ndarray]],
    read_fn: ReaderFn,
    bands_count: int,
    coverage_mask: np.ndarray,
    no_data_threshold: Optional[float],
    out_dtype: np.dtype,
) -> Tuple[Tuple[int, int, int, int], np.ndarray]:
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
        if _tile_threshold_met(filled, tile_coverage, no_data_threshold):
            break
        if filled.all():
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


def run_tile_aggregation(
    masks: List[Optional[np.ndarray]],
    read_fn: ReaderFn,
    bands_count: int,
    height: int,
    width: int,
    coverage_mask: np.ndarray,
    no_data_threshold: Optional[float],
    mosaic_method: str,
    percentile_value: Optional[float],
    tile_size: int,
    tile_workers: Optional[int],
    out_dtype: np.dtype = DEFAULT_OUTPUT_DTYPE,
) -> np.ndarray:
    """Generic streaming aggregation. Called by both grid_id and bounds modes.

    ``out_dtype`` is the pipeline's final output dtype (``uint16`` for
    spectral, ``uint8`` for visual). Tile workers cast to it before
    returning, so the output buffer can be allocated as the final dtype —
    no intermediate float32 array the size of the whole mosaic.
    """
    out = np.zeros((bands_count, height, width), dtype=out_dtype)
    specs = tile_specs_for(height, width, tile_size)
    n_workers = tile_workers or os.cpu_count() or 8

    if mosaic_method == MOSAIC_PERCENTILE:
        _pin_numba_threads(1)
        pv = percentile_value if percentile_value is not None else 50.0

        def worker_fn(
            s: Tuple[int, int, int, int],
        ) -> Tuple[Tuple[int, int, int, int], np.ndarray]:
            return tile_percentile(
                s,
                masks,
                read_fn,
                bands_count,
                pv,
                coverage_mask,
                no_data_threshold,
                out_dtype,
            )

    elif mosaic_method == MOSAIC_MEAN:

        def worker_fn(
            s: Tuple[int, int, int, int],
        ) -> Tuple[Tuple[int, int, int, int], np.ndarray]:
            return tile_mean(
                s,
                masks,
                read_fn,
                bands_count,
                coverage_mask,
                no_data_threshold,
                out_dtype,
            )

    elif mosaic_method == MOSAIC_FIRST:

        def worker_fn(
            s: Tuple[int, int, int, int],
        ) -> Tuple[Tuple[int, int, int, int], np.ndarray]:
            return tile_first(
                s,
                masks,
                read_fn,
                bands_count,
                coverage_mask,
                no_data_threshold,
                out_dtype,
            )

    else:
        raise ValueError(f"Unknown mosaic_method: {mosaic_method}")

    completed = 0
    log_every = max(1, len(specs) // 10)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for spec, tile_data in ex.map(worker_fn, specs):
            r, c, h, w = spec
            out[:, r : r + h, c : c + w] = tile_data
            completed += 1
            if completed % log_every == 0 or completed == len(specs):
                logger.info("Phase 2: %d/%d tiles done", completed, len(specs))
    return out


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
    tests, or for callers that expect to skip many scenes via no_data_threshold).
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
    ) -> np.ndarray:
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
) -> bool:
    """Whether to pre-materialise tile sources before aggregation.

    Prewarming improves throughput when most scene/band sources will be read
    anyway. Keep sources lazy when the aggregation is likely to skip many reads:
    ``first`` mode can stop as pixels fill, and any ``no_data_threshold`` can
    stop per-tile scene walks before all scenes are touched.
    """
    return mosaic_method != MOSAIC_FIRST and no_data_threshold is None


def stream_mosaic_pipeline(
    sorted_scenes: pd.DataFrame,
    required_bands: List[str],
    coverage_mask: np.ndarray,
    no_data_threshold: Union[float, None],
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
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Tile-streamed mosaic for grid_id mode.

    Replaces the old in-memory ``download_bands_pool`` path. Peak working
    set is per-worker (a few hundred MB), so 34-scene full-MGRS percentile
    mosaics that previously needed ~65 GB of RAM now fit in a few GB.

    ``no_data_threshold`` is applied per-tile: each tile's time series
    walks scenes in priority order and stops once the tile's coverage of
    the global ``coverage_mask`` exceeds ``1 - threshold``. Different
    tiles may use different numbers of scenes — clear tiles finish after
    the first scene, cloudy tiles process more.
    """
    ocm_resolution = pick_ocm_resolution(resolution)
    logger.info(f"OCM resolution: {ocm_resolution}m")
    possible_pixel_count = coverage_mask.sum()
    logger.info(f"Possible pixel count: {possible_pixel_count}")

    items: List[Any] = sorted_scenes[ITEM_COL].tolist()
    n_scenes = len(items)
    is_visual = "visual" in required_bands
    href_template, bands_count, _ = get_band_template(required_bands)

    # Phase 1: compute per-scene combo masks.
    # OCM runs the deep-learning cloud detector per scene — keep its
    # download concurrency limited so we don't blow GPU/CPU. SCL is just
    # a band read so we can fan out wider.
    mask_workers = (
        max_dl_workers if cloud_mask == CLOUD_MASK_OCM else max(4, max_dl_workers)
    )
    logger.info(
        "Phase 1: computing masks for %d scenes (%s, workers=%d)",
        n_scenes,
        cloud_mask,
        mask_workers,
    )
    masks: List[Optional[np.ndarray]] = [None] * n_scenes

    def _worker(idx_item: Tuple[int, Any]) -> Tuple[int, Optional[np.ndarray]]:
        idx, item = idx_item
        combo = _compute_one_scene_mask(
            item=item,
            cloud_mask=cloud_mask,
            ocm_batch_size=ocm_batch_size,
            ocm_inference_dtype=ocm_inference_dtype,
            ocm_resolution=ocm_resolution,
            max_dl_workers=max_dl_workers,
            s2_scene_size=s2_scene_size,
            resolution=resolution,
        )
        logger.info(
            "Phase 1: scene %d/%d (%s): %s",
            idx + 1,
            n_scenes,
            item.id,
            "ok" if combo is not None else "skipped",
        )
        return idx, combo

    with ThreadPoolExecutor(max_workers=mask_workers) as ex:
        for idx, combo in ex.map(_worker, enumerate(items)):
            masks[idx] = combo

    n_succeeded = sum(1 for m in masks if m is not None)
    n_failed = n_scenes - n_succeeded
    if n_failed:
        logger.warning(f"Phase 1: {n_failed}/{n_scenes} scenes failed mask compute")
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
        prewarm=should_prewarm_sources(mosaic_method, no_data_threshold),
    )

    logger.info(
        "Phase 2: %s aggregation (tile=%d)",
        mosaic_method,
        tile_size,
    )
    out = run_tile_aggregation(
        masks=masks,
        read_fn=read_fn,
        bands_count=bands_count,
        height=s2_scene_size,
        width=s2_scene_size,
        coverage_mask=coverage_mask,
        no_data_threshold=no_data_threshold,
        mosaic_method=mosaic_method,
        percentile_value=percentile_value,
        tile_size=tile_size,
        tile_workers=tile_workers,
        out_dtype=np.dtype(np.uint8) if is_visual else np.dtype(np.uint16),
    )
    return out, last_profile
