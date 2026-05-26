"""Raster tile readers."""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import rasterio as rio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

from .config import CLOUD_MASK_SCL, MOSAIC_FIRST
from .geometry import Bbox
from .helpers import SceneFetchError, get_rasterio_resampling
from .masking import get_masks, get_scl_masks
from .sources import Source
from ._types import BoundsItemLike

logger = logging.getLogger(__name__)
DEFAULT_TILE_WORKERS = 8
SIGNED_URL_TTL_SECONDS = 45 * 60
REMOTE_RASTER_ATTEMPTS = 3
MAX_PREWARM_WORKERS = 16
GridSourceResolver = Callable[[bool], str]
BoundsSourceResolver = Callable[[bool], str]
RasterOpener = Callable[[bool], rio.DatasetReader]


def _lazy_signed_url(
    source: Source,
    href: str,
    ttl_seconds: int = SIGNED_URL_TTL_SECONDS,
) -> Callable[[bool], str]:
    """Return a thread-safe TTL-aware signer for a source asset URL."""
    signed_url: Optional[str] = None
    signed_at: Optional[float] = None
    lock = threading.Lock()

    def get(refresh: bool = False) -> str:
        nonlocal signed_url, signed_at
        with lock:
            now = time.monotonic()
            expired = (
                signed_at is not None
                and ttl_seconds > 0
                and now - signed_at >= ttl_seconds
            )
            if signed_url is None or refresh or expired:
                signed_url = source.sign(href)
                signed_at = now
            return signed_url

    return get


def _retry_open_raster(
    open_source: RasterOpener, *, refresh: bool
) -> rio.DatasetReader:
    """Open a remote raster with retry/backoff and optional source refresh."""
    last_error: Optional[RasterioIOError] = None
    for attempt in range(REMOTE_RASTER_ATTEMPTS):
        try:
            return open_source(refresh or attempt > 0)
        except RasterioIOError as exc:
            last_error = exc
            if attempt < REMOTE_RASTER_ATTEMPTS - 1:
                time.sleep(0.5 * (attempt + 1))
    if last_error is None:
        raise RasterioIOError("Remote raster open failed")
    raise last_error


def _compute_one_scene_mask(
    item: Any,
    source: Source,
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
            clear, valid = get_scl_masks(
                item=item, source=source, user_resolution=resolution
            )
        else:
            clear, valid = get_masks(
                item=item,
                source=source,
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

    def __init__(self, source_resolvers: List[List[GridSourceResolver]]):
        # source_resolvers[scene_idx][band_idx]() -> local cache path or signed URL
        self._source_resolvers = source_resolvers
        self._local = threading.local()
        self._handles: Dict[int, rio.DatasetReader] = {}
        self._handles_lock = threading.Lock()
        self._closed = False

    def get(self, scene_idx: int, band_idx: int) -> rio.DatasetReader:
        if self._closed:
            raise RuntimeError("Cannot read from a closed raster handle cache")
        per_thread = getattr(self._local, "handles", None)
        if per_thread is None:
            per_thread = {}
            self._local.handles = per_thread
        key = (scene_idx, band_idx)
        h = per_thread.get(key)
        if h is None:
            h = _retry_open_raster(
                lambda refresh: rio.open(
                    self._source_resolvers[scene_idx][band_idx](refresh)
                ),
                refresh=False,
            )
            per_thread[key] = h
            with self._handles_lock:
                self._handles[id(h)] = h
        return h

    def reopen(self, scene_idx: int, band_idx: int) -> rio.DatasetReader:
        if self._closed:
            raise RuntimeError("Cannot read from a closed raster handle cache")
        per_thread = getattr(self._local, "handles", None)
        if per_thread is None:
            per_thread = {}
            self._local.handles = per_thread
        key = (scene_idx, band_idx)
        old = per_thread.pop(key, None)
        if old is not None:
            with self._handles_lock:
                self._handles.pop(id(old), None)
            old.close()
        h = _retry_open_raster(
            lambda refresh: rio.open(
                self._source_resolvers[scene_idx][band_idx](refresh)
            ),
            refresh=True,
        )
        per_thread[key] = h
        with self._handles_lock:
            self._handles[id(h)] = h
        return h

    def close(self) -> None:
        with self._handles_lock:
            self._closed = True
            handles = list(self._handles.values())
            self._handles = {}
        if hasattr(self._local, "handles"):
            self._local.handles = {}
        for handle in handles:
            handle.close()


class GridTileReader:
    """Callable grid-mode tile reader with explicit raster handle cleanup."""

    def __init__(
        self,
        cache: _HandleCache,
        href_band_indices: List[int],
        s2_scene_size: int,
        rio_resampling: Resampling,
    ):
        self._cache = cache
        self._href_band_indices = href_band_indices
        self._s2_scene_size = s2_scene_size
        self._rio_resampling = rio_resampling

    def __call__(
        self, scene_idx: int, band_idx: int, spec: Tuple[int, int, int, int]
    ) -> npt.NDArray[Any]:
        last_error: Optional[RasterioIOError] = None
        for attempt in range(REMOTE_RASTER_ATTEMPTS):
            src = (
                self._cache.get(scene_idx, band_idx)
                if attempt == 0
                else self._cache.reopen(scene_idx, band_idx)
            )
            try:
                return _read_tile_window(
                    src,
                    self._href_band_indices[band_idx],
                    spec,
                    self._s2_scene_size,
                    self._rio_resampling,
                )
            except RasterioIOError as exc:
                last_error = exc
                if attempt < REMOTE_RASTER_ATTEMPTS - 1:
                    time.sleep(0.5 * (attempt + 1))
        if last_error is None:
            raise RasterioIOError("Remote raster read failed")
        raise last_error

    def close(self) -> None:
        self._cache.close()


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


def make_grid_tile_reader(
    items: List[Any],
    href_template: List[Tuple[str, int]],
    source: Source,
    s2_scene_size: int,
    resolution: int,
    resampling_method: str,
    prewarm: bool = True,
) -> GridTileReader:
    """Build a tile-reader for grid_id mode (direct MGRS COG reads).

    Builds lazy source resolvers for every (scene, asset). Workers in Phase 2
    open handles lazily per thread.

    With ``prewarm=True`` (default) the resolvers are called in parallel once
    before returning so all signed URLs are warmed in a single burst rather
    than fanned out serially by the per-tile workers. Set ``prewarm=False``
    to keep the resolvers strictly lazy (useful for tests, or for callers
    that expect to skip many reads via early stopping).
    """
    rio_resampling = get_rasterio_resampling(resampling_method)
    href_band_indices = [band_idx for _, band_idx in href_template]

    sources: List[List[GridSourceResolver]] = []
    for item in items:
        scene_sources: List[GridSourceResolver] = []
        for asset, _ in href_template:
            asset_key = source.asset_name(asset)
            get_signed_url = _lazy_signed_url(source, item.assets[asset_key].href)

            def source_for(
                refresh: bool = False,
                get_signed_url: Callable[[bool], str] = get_signed_url,
            ) -> str:
                return get_signed_url(refresh)

            scene_sources.append(source_for)
        sources.append(scene_sources)
    cache = _HandleCache(sources)

    if prewarm:
        _prewarm_sources(sources)

    return GridTileReader(cache, href_band_indices, s2_scene_size, rio_resampling)


def _prewarm_sources(sources: List[List[Callable[..., Any]]]) -> None:
    """Call every lazy source resolver in parallel to pre-sign URLs."""
    flat: List[Callable[..., Any]] = [
        resolver for scene in sources for resolver in scene
    ]
    if not flat:
        return
    n_workers = min(len(flat), MAX_PREWARM_WORKERS)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        # Drain exceptions; the lazy path still raises on first read.
        futures = [ex.submit(resolver) for resolver in flat]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                logger.warning("Prewarm source signing failed: %r", exc)


def should_prewarm_sources(
    mosaic_method: str,
    min_observations: Optional[int] = None,
    max_observations: Optional[int] = None,
) -> bool:
    """Whether to pre-sign tile sources before aggregation.

    Prewarming improves throughput when most scene/band sources will be read
    anyway. Keep sources lazy when the aggregation is likely to skip many reads:
    ``first`` mode can stop as pixels fill, and observation bounds can cap
    per-tile or per-pixel observations.
    """
    return (
        mosaic_method != MOSAIC_FIRST
        and min_observations is None
        and max_observations is None
    )


def make_bounds_tile_reader(
    items: List[BoundsItemLike],
    href_template: List[Tuple[str, int]],
    source: Source,
    bounds_target: Bbox,
    target_crs: int,
    user_transform: Affine,
    width: int,
    height: int,
    resolution: int,
    resampling_method: str,
    prewarm: bool = True,
) -> "BoundsTileReader":
    """Build a tile-reader for bounds mode (WarpedVRT-backed reads).

    Builds lazy source resolvers for every (scene, asset). Workers open
    handles lazily per thread and wrap each opened COG in a ``WarpedVRT``
    that reprojects to the user's target grid.

    With ``prewarm=True`` (default) the resolvers are called in parallel once
    before returning. Pass ``prewarm=False`` to keep them strictly lazy.
    """
    target_crs_obj = CRS.from_epsg(target_crs)
    rio_resampling = get_rasterio_resampling(resampling_method)
    href_band_indices = [band_idx for _, band_idx in href_template]

    sources: List[List[BoundsSourceResolver]] = []
    for item in items:
        scene_sources: List[BoundsSourceResolver] = []
        for asset, _ in href_template:
            asset_key = source.asset_name(asset)
            get_signed_url = _lazy_signed_url(source, item.assets[asset_key].href)

            def source_for(
                refresh: bool = False,
                get_signed_url: Callable[[bool], str] = get_signed_url,
            ) -> str:
                return get_signed_url(refresh)

            scene_sources.append(source_for)
        sources.append(scene_sources)

    if prewarm:
        _prewarm_sources(sources)

    return BoundsTileReader(
        sources=sources,
        href_band_indices=href_band_indices,
        target_crs_obj=target_crs_obj,
        user_transform=user_transform,
        width=width,
        height=height,
        rio_resampling=rio_resampling,
    )


class BoundsTileReader:
    """Callable bounds-mode tile reader with explicit raster handle cleanup."""

    def __init__(
        self,
        sources: List[List[BoundsSourceResolver]],
        href_band_indices: List[int],
        target_crs_obj: CRS,
        user_transform: Affine,
        width: int,
        height: int,
        rio_resampling: Any,
    ):
        self._sources = sources
        self._href_band_indices = href_band_indices
        self._target_crs_obj = target_crs_obj
        self._user_transform = user_transform
        self._width = width
        self._height = height
        self._rio_resampling = rio_resampling
        self._local = threading.local()
        self._entries: Dict[int, Tuple[Any, Any]] = {}
        self._entries_lock = threading.Lock()
        self._closed = False

    def _open_entry(
        self, scene_idx: int, asset_idx: int, refresh: bool
    ) -> Tuple[Any, Any]:
        if self._closed:
            raise RuntimeError("Cannot read from a closed bounds tile reader")

        def open_source(refresh_attempt: bool) -> rio.DatasetReader:
            href = self._sources[scene_idx][asset_idx](refresh_attempt)
            return rio.open(href)

        src = _retry_open_raster(open_source, refresh=refresh)
        handle = WarpedVRT(
            src,
            crs=self._target_crs_obj,
            transform=self._user_transform,
            width=self._width,
            height=self._height,
            resampling=self._rio_resampling,
        )
        return src, handle

    def _get_source(self, scene_idx: int, asset_idx: int) -> Any:
        if self._closed:
            raise RuntimeError("Cannot read from a closed bounds tile reader")
        per_thread = getattr(self._local, "handles", None)
        if per_thread is None:
            per_thread = {}
            self._local.handles = per_thread
        key = (scene_idx, asset_idx)
        entry = per_thread.get(key)
        if entry is None:
            entry = self._open_entry(scene_idx, asset_idx, refresh=False)
            per_thread[key] = entry
            with self._entries_lock:
                self._entries[id(entry)] = entry
        return entry[1]

    def _reopen_source(self, scene_idx: int, asset_idx: int) -> Any:
        if self._closed:
            raise RuntimeError("Cannot read from a closed bounds tile reader")
        per_thread = getattr(self._local, "handles", None)
        if per_thread is None:
            per_thread = {}
            self._local.handles = per_thread
        key = (scene_idx, asset_idx)
        old = per_thread.pop(key, None)
        if old is not None:
            with self._entries_lock:
                self._entries.pop(id(old), None)
            src, handle = old
            handle.close()
            src.close()
        entry = self._open_entry(scene_idx, asset_idx, refresh=True)
        per_thread[key] = entry
        with self._entries_lock:
            self._entries[id(entry)] = entry
        return entry[1]

    def __call__(
        self, scene_idx: int, band_idx: int, spec: Tuple[int, int, int, int]
    ) -> npt.NDArray[Any]:
        r, c, th, tw = spec
        window_cls: Any = Window
        window = window_cls(c, r, tw, th)
        last_error: Optional[RasterioIOError] = None
        for attempt in range(REMOTE_RASTER_ATTEMPTS):
            src = (
                self._get_source(scene_idx, band_idx)
                if attempt == 0
                else self._reopen_source(scene_idx, band_idx)
            )
            try:
                return src.read(  # type: ignore[no-any-return, unused-ignore]
                    self._href_band_indices[band_idx],
                    window=window,
                )
            except RasterioIOError as exc:
                last_error = exc
                if attempt < REMOTE_RASTER_ATTEMPTS - 1:
                    time.sleep(0.5 * (attempt + 1))
        if last_error is None:
            raise RasterioIOError("Remote raster read failed")
        raise last_error

    def close(self) -> None:
        with self._entries_lock:
            self._closed = True
            entries = list(self._entries.values())
            self._entries = {}
        if hasattr(self._local, "handles"):
            self._local.handles = {}
        for src, handle in entries:
            handle.close()
            src.close()
