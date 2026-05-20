"""Raster tile readers and local materialisation helpers."""

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

from .cache import _write_tiled_copy, materialise_tiled_band
from .config import CLOUD_MASK_SCL, MOSAIC_FIRST
from .geometry import Bbox
from .helpers import SceneFetchError, get_rasterio_resampling
from .masking import get_masks, get_scl_masks
from .sources import Source
from .stac_bounds import _BoundsItemLike

logger = logging.getLogger(__name__)
DEFAULT_TILE_WORKERS = min(4, os.cpu_count() or 1)


def _materialise_grid_band(
    item: Any,
    asset_name: str,
    source: Source,
    s2_scene_size: int,
    rio_resampling: Resampling,
) -> Callable[[Path], None]:
    """Build a materialiser for a grid_id-mode band cache entry.

    Returns a closure that, given an output path, downloads the full asset
    from the configured source and writes it as a tiled GeoTIFF on the MGRS
    grid at the user's output resolution.
    """

    def write(tmp_path: Path) -> None:
        signed = source.sign(item.assets[asset_name].href)
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

    def __init__(self, source_resolvers: List[List[Callable[[], str]]]):
        # source_resolvers[scene_idx][band_idx]() -> local cache path or signed URL
        self._source_resolvers = source_resolvers
        self._local = threading.local()
        self._handles: List[rio.DatasetReader] = []
        self._handles_lock = threading.Lock()

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
            with self._handles_lock:
                self._handles.append(h)
        return h

    def close(self) -> None:
        with self._handles_lock:
            handles = self._handles
            self._handles = []
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
        src = self._cache.get(scene_idx, band_idx)
        return _read_tile_window(
            src,
            self._href_band_indices[band_idx],
            spec,
            self._s2_scene_size,
            self._rio_resampling,
        )

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
            asset_key = source.asset_name(asset)
            cache_key = (
                f"grid|{source.name}|{item.id}|{asset_key}|{s2_scene_size}|"
                f"{resolution}|{resampling_method}"
            )
            signed_url = source.sign(item.assets[asset_key].href)

            def source_for(
                item: Any = item,
                asset_key: str = asset_key,
                cache_key: str = cache_key,
                signed_url: str = signed_url,
            ) -> str:
                local = materialise_tiled_band(
                    cache_key,
                    _materialise_grid_band(
                        item, asset_key, source, s2_scene_size, rio_resampling
                    ),
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

    return GridTileReader(cache, href_band_indices, s2_scene_size, rio_resampling)


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
        # Drain exceptions; the lazy path still raises on first read, but this
        # makes prewarm-only failures visible under DEBUG.
        futures = [ex.submit(resolver) for resolver in flat]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                logger.debug("Prewarm failure: %r", exc)


def should_prewarm_sources(
    mosaic_method: str,
    early_stop_missing_fraction: Optional[float],
    min_observations: Optional[int] = None,
) -> bool:
    """Whether to pre-materialise tile sources before aggregation.

    Prewarming improves throughput when most scene/band sources will be read
    anyway. Keep sources lazy when the aggregation is likely to skip many reads:
    ``first`` mode can stop as pixels fill, ``early_stop_missing_fraction`` can stop
    scene walks before all scenes are touched, and ``min_observations``
    can cap per-tile observations.
    """
    return (
        mosaic_method != MOSAIC_FIRST
        and (early_stop_missing_fraction is None or early_stop_missing_fraction == 0.0)
        and min_observations is None
    )


def _materialise_bounds_band(
    item: _BoundsItemLike,
    asset_name: str,
    source: Source,
    bounds_target: Bbox,
    target_crs: int,
    width: int,
    height: int,
    resolution: int,
    rio_resampling: Any,
) -> Callable[[Path], None]:
    """Build a materialiser for a bounds-mode band cache entry.

    Closes over the item, asset, target grid, and resampling. The returned
    function opens the COG via WarpedVRT (which handles reprojection to the
    user's target grid) and streams destination blocks to a tiled GeoTIFF.
    Subsequent reads of any tile window can open the local file directly —
    no WarpedVRT needed once materialised.
    """
    target_crs_obj = CRS.from_epsg(target_crs)
    minx, _, _, maxy = bounds_target
    transform = Affine(resolution, 0, minx, 0, -resolution, maxy)

    def write(tmp_path: Path) -> None:
        signed = source.sign(item.assets[asset_name].href)
        with rio.open(signed) as src:
            with WarpedVRT(
                src,
                crs=target_crs_obj,
                transform=transform,
                width=width,
                height=height,
                resampling=rio_resampling,
            ) as vrt:
                n_bands = vrt.count
                dtype = vrt.dtypes[0]
                profile = {
                    "driver": "GTiff",
                    "count": n_bands,
                    "dtype": dtype,
                    "width": width,
                    "height": height,
                    "crs": target_crs_obj,
                    "transform": transform,
                    "tiled": True,
                    "blockxsize": 512,
                    "blockysize": 512,
                    "compress": "lzw",
                    "BIGTIFF": "IF_SAFER",
                }
                _write_tiled_copy(vrt, tmp_path, profile, rio_resampling, lambda w: w)

    return write


def make_bounds_tile_reader(
    items: List[_BoundsItemLike],
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
    handles lazily per thread; if ``S2MOSAIC_DEBUG_CACHE`` is on, the local
    cache file is materialised only when a tile actually reads that source.
    PC URLs get wrapped in a ``WarpedVRT`` to reproject on read, while cached
    local files are opened directly (they're already on the target grid).

    With ``prewarm=True`` (default) the resolvers are called in parallel once
    before returning. Pass ``prewarm=False`` to keep them strictly lazy.
    """
    target_crs_obj = CRS.from_epsg(target_crs)
    rio_resampling = get_rasterio_resampling(resampling_method)
    href_band_indices = [band_idx for _, band_idx in href_template]

    sources: List[List[Callable[[], Tuple[str, bool]]]] = []
    for item in items:
        scene_sources: List[Callable[[], Tuple[str, bool]]] = []
        for asset, _ in href_template:
            asset_key = source.asset_name(asset)
            cache_key = (
                f"bounds|{source.name}|{item.id}|{asset_key}|{bounds_target}|"
                f"{target_crs}|{width}|{height}|{resolution}|{resampling_method}"
            )
            signed_url = source.sign(item.assets[asset_key].href)

            def source_for(
                item: _BoundsItemLike = item,
                asset_key: str = asset_key,
                cache_key: str = cache_key,
                signed_url: str = signed_url,
            ) -> Tuple[str, bool]:
                local = materialise_tiled_band(
                    cache_key,
                    _materialise_bounds_band(
                        item,
                        asset_key,
                        source,
                        bounds_target,
                        target_crs,
                        width,
                        height,
                        resolution,
                        rio_resampling,
                    ),
                )
                return (str(local), True) if local is not None else (signed_url, False)

            scene_sources.append(source_for)
        sources.append(scene_sources)

    if prewarm:
        # Pre-warm sources in parallel. With debug cache on this materialises
        # every (scene, asset) up front; without cache it's just parallel URL
        # signing. Avoids the serial fan-out tile workers would otherwise do
        # when each first touches a (scene, asset) inside the tile loop.
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
        sources: List[List[Callable[[], Tuple[str, bool]]]],
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
        self._entries: List[Tuple[Any, Any]] = []
        self._entries_lock = threading.Lock()

    def _get_source(self, scene_idx: int, asset_idx: int) -> Any:
        per_thread = getattr(self._local, "handles", None)
        if per_thread is None:
            per_thread = {}
            self._local.handles = per_thread
        key = (scene_idx, asset_idx)
        entry = per_thread.get(key)
        if entry is None:
            source, is_local = self._sources[scene_idx][asset_idx]()
            src = rio.open(source)
            if is_local:
                handle: Any = src
            else:
                handle = WarpedVRT(
                    src,
                    crs=self._target_crs_obj,
                    transform=self._user_transform,
                    width=self._width,
                    height=self._height,
                    resampling=self._rio_resampling,
                )
            entry = (src, handle)
            per_thread[key] = entry
            with self._entries_lock:
                self._entries.append(entry)
        return entry[1]

    def __call__(
        self, scene_idx: int, band_idx: int, spec: Tuple[int, int, int, int]
    ) -> npt.NDArray[Any]:
        r, c, th, tw = spec
        src = self._get_source(scene_idx, band_idx)
        window_cls: Any = Window
        return src.read(  # type: ignore[no-any-return, unused-ignore]
            self._href_band_indices[band_idx],
            window=window_cls(c, r, tw, th),
        )

    def close(self) -> None:
        with self._entries_lock:
            entries = self._entries
            self._entries = []
        for src, handle in entries:
            if handle is not src:
                handle.close()
            src.close()
