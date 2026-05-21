"""Tile aggregation algorithms shared by grid and bounds pipelines."""

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import rasterio as rio
from numba import njit
from rasterio.windows import Window
from tqdm.auto import tqdm

from .config import MOSAIC_FIRST, MOSAIC_MEAN, MOSAIC_PERCENTILE
from .output import output_band_metadata

logger = logging.getLogger(__name__)
DEFAULT_OUTPUT_DTYPE = np.dtype(np.uint16)
DEFAULT_TILE_WORKERS = min(4, os.cpu_count() or 1)
DEFAULT_ADAPTIVE_TILE_MIN_SIZE = 512
DEFAULT_ADAPTIVE_TILE_DENSE_FRACTION = 0.75

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
    min_observations: Optional[int],
) -> List[int]:
    """Scene indices that contribute to a tile before min_observations is met."""
    r, c, h, w = spec
    contributing: List[int] = []
    observation_count: Optional[npt.NDArray[Any]] = None
    if min_observations is not None:
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
            if ((observation_count >= min_observations) | ~tile_coverage).all():
                break

    return contributing


def tile_percentile(
    spec: Tuple[int, int, int, int],
    masks: List[Optional[npt.NDArray[Any]]],
    read_fn: ReaderFn,
    bands_count: int,
    percentile: float,
    coverage_mask: npt.NDArray[Any],
    min_observations: Optional[int],
    out_dtype: "np.dtype[Any]",
) -> Tuple[Tuple[int, int, int, int], npt.NDArray[Any]]:
    r, c, h, w = spec
    tile_coverage = coverage_mask[r : r + h, c : c + w]
    if not tile_coverage.any():
        return spec, _empty_tile(spec, bands_count, out_dtype)

    contributing = _contributing_scene_indices(
        spec, masks, tile_coverage, min_observations
    )

    if not contributing:
        return spec, _empty_tile(spec, bands_count, out_dtype)

    if len(contributing) == 1:
        scene_idx = contributing[0]
        mask = masks[scene_idx]
        if mask is None:
            raise RuntimeError(f"Missing mask for contributing scene {scene_idx}")
        mask_tile = mask[r : r + h, c : c + w]
        return spec, _copy_single_scene_tile(
            spec, mask_tile, read_fn, scene_idx, bands_count, out_dtype
        )

    stack = np.empty((len(contributing), bands_count, h, w), dtype=np.float32)
    for k, scene_idx in enumerate(contributing):
        mask = masks[scene_idx]
        if mask is None:
            raise RuntimeError(f"Missing mask for contributing scene {scene_idx}")
        mask_tile = mask[r : r + h, c : c + w]
        for j in range(bands_count):
            data = read_fn(scene_idx, j, spec)
            stack[k, j].fill(np.nan)
            np.copyto(stack[k, j], data, where=mask_tile, casting="unsafe")

    res = _nanquantile_axis0(stack, percentile / 100.0)
    res = np.nan_to_num(res, nan=0.0)
    return spec, _finalise_tile(res, out_dtype)


def tile_mean(
    spec: Tuple[int, int, int, int],
    masks: List[Optional[npt.NDArray[Any]]],
    read_fn: ReaderFn,
    bands_count: int,
    coverage_mask: npt.NDArray[Any],
    min_observations: Optional[int],
    out_dtype: "np.dtype[Any]",
) -> Tuple[Tuple[int, int, int, int], npt.NDArray[Any]]:
    r, c, h, w = spec
    tile_coverage = coverage_mask[r : r + h, c : c + w]
    if not tile_coverage.any():
        return spec, _empty_tile(spec, bands_count, out_dtype)

    contributing = _contributing_scene_indices(
        spec, masks, tile_coverage, min_observations
    )

    if not contributing:
        return spec, _empty_tile(spec, bands_count, out_dtype)

    if len(contributing) == 1:
        scene_idx = contributing[0]
        mask = masks[scene_idx]
        if mask is None:
            raise RuntimeError(f"Missing mask for contributing scene {scene_idx}")
        mask_tile = mask[r : r + h, c : c + w]
        return spec, _copy_single_scene_tile(
            spec, mask_tile, read_fn, scene_idx, bands_count, out_dtype
        )

    sum_block = np.zeros((bands_count, h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.uint16)
    for scene_idx in contributing:
        mask = masks[scene_idx]
        if mask is None:
            raise RuntimeError(f"Missing mask for contributing scene {scene_idx}")
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
    early_stop_missing_fraction: Optional[float],
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
    times the number of user bands. ``first`` and ``min_observations``
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
    early_stop_missing_fraction: Optional[float],
    mosaic_method: str,
    percentile: Optional[float],
    tile_size: int,
    tile_workers: Optional[int],
    out_dtype: "np.dtype[Any]" = DEFAULT_OUTPUT_DTYPE,
    min_observations: Optional[int] = None,
    adaptive_tiling: bool = True,
    tile_specs: Optional[List[Tuple[int, int, int, int]]] = None,
    show_progress: bool = False,
    min_tile_size: int = DEFAULT_ADAPTIVE_TILE_MIN_SIZE,
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
        early_stop_missing_fraction=early_stop_missing_fraction,
        mosaic_method=mosaic_method,
        percentile=percentile,
        tile_size=tile_size,
        tile_workers=tile_workers,
        out_dtype=out_dtype,
        min_observations=min_observations,
        adaptive_tiling=adaptive_tiling,
        tile_specs=tile_specs,
        show_progress=show_progress,
        min_tile_size=min_tile_size,
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
    early_stop_missing_fraction: Optional[float],
    mosaic_method: str,
    percentile: Optional[float],
    tile_size: int,
    tile_workers: Optional[int],
    out_dtype: "np.dtype[Any]" = DEFAULT_OUTPUT_DTYPE,
    min_observations: Optional[int] = None,
    adaptive_tiling: bool = True,
    tile_specs: Optional[List[Tuple[int, int, int, int]]] = None,
    show_progress: bool = False,
    min_tile_size: int = DEFAULT_ADAPTIVE_TILE_MIN_SIZE,
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
            min_tile_size=min_tile_size,
        )
    else:
        specs = tile_specs_for(height, width, tile_size)

    # Phase 2 progress is per band-read rather than per tile so the bar
    # advances smoothly. Total is the upper bound — each (scene, band) read
    # that *would* happen if no early-stop kicks in. ``first`` /
    # min_observations modes may finish below 100%, which we fast-forward.
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
        pv = percentile if percentile is not None else 50.0
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
                min_observations,
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
                min_observations,
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
                early_stop_missing_fraction,
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
            # Early-stop modes (first, min_observations) skip reads, so the
            # bar may not have reached total. Snap to total so it shows done.
            remaining = progress_bar.total - progress_bar.n
            if remaining > 0:
                progress_bar.update(remaining)
            progress_bar.close()


def write_tile_aggregation_geotiff(
    export_path: Path,
    profile: Dict[str, Any],
    bands: List[str],
    masks: List[Optional[npt.NDArray[Any]]],
    read_fn: ReaderFn,
    bands_count: int,
    height: int,
    width: int,
    coverage_mask: npt.NDArray[Any],
    output_coverage_mask: Optional[npt.NDArray[Any]],
    early_stop_missing_fraction: Optional[float],
    mosaic_method: str,
    percentile: Optional[float],
    tile_size: int,
    tile_workers: Optional[int],
    out_dtype: "np.dtype[Any]" = DEFAULT_OUTPUT_DTYPE,
    min_observations: Optional[int] = None,
    adaptive_tiling: bool = True,
    tile_specs: Optional[List[Tuple[int, int, int, int]]] = None,
    show_progress: bool = False,
    min_tile_size: int = DEFAULT_ADAPTIVE_TILE_MIN_SIZE,
) -> Path:
    """Aggregate tiles and write them directly into a GeoTIFF."""
    band_descriptions, nodata_value = output_band_metadata(bands)
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
    tmp_path = export_path.with_suffix(
        f".tmp.{os.getpid()}.{threading.get_ident()}{export_path.suffix}"
    )
    logger.info("Writing streamed GeoTIFF to %s via %s", export_path, tmp_path)
    try:
        with rio.open(tmp_path, "w", **write_profile) as dst:
            dst.descriptions = band_descriptions
            for spec, tile_data in iter_tile_aggregation(
                masks=masks,
                read_fn=read_fn,
                bands_count=bands_count,
                height=height,
                width=width,
                coverage_mask=coverage_mask,
                early_stop_missing_fraction=early_stop_missing_fraction,
                mosaic_method=mosaic_method,
                percentile=percentile,
                tile_size=tile_size,
                tile_workers=tile_workers,
                out_dtype=out_dtype,
                min_observations=min_observations,
                adaptive_tiling=adaptive_tiling,
                tile_specs=tile_specs,
                show_progress=show_progress,
                min_tile_size=min_tile_size,
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
                window_cls: Any = Window
                dst.write(tile_data, window=window_cls(c, r, w, h))
        tmp_path.replace(export_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return export_path
