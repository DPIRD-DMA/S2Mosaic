"""Tile aggregation algorithms shared by grid and bounds pipelines."""

import logging
import os
import tempfile
import threading
from concurrent.futures import (
    FIRST_COMPLETED,
    Executor,
    Future,
    ThreadPoolExecutor,
    wait,
)
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import rasterio as rio
from numba import njit
from rasterio.errors import RasterioIOError
from rasterio.windows import Window
from tqdm.auto import tqdm

from .config import MOSAIC_FIRST, MOSAIC_MEAN, MOSAIC_MEDOID, MOSAIC_PERCENTILE
from .output import output_band_metadata, output_valid_mask

logger = logging.getLogger(__name__)
DEFAULT_OUTPUT_DTYPE = np.dtype(np.uint16)
# Bumped from min(4, cpu_count). With band-parallel reads each tile worker
# fans out ``bands_count`` concurrent range requests on the same HTTP/2
# connection, so the work per worker is I/O-bound and benefits from a higher
# tile-worker count than CPU count would suggest. 8 was the sweet spot on a
# 400 Mbps Starlink benchmark — more workers gave diminishing returns and
# multiplied concurrent requests against the source server.
DEFAULT_TILE_WORKERS = 8
DEFAULT_BAND_READ_WORKERS = 16
DEFAULT_ADAPTIVE_TILE_MIN_SIZE = 512
DEFAULT_ADAPTIVE_TILE_DENSE_FRACTION = 0.75
MEDOID_STRIPE_HEIGHT = 128
EXPECTED_READ_EXACT_SCAN_LIMIT = 50_000

# Tile requeue policy. When a tile worker fails with RasterioIOError the
# parallel path puts the spec back at the end of the executor's queue, giving
# the failure time to clear while other tiles finish. Caps below keep a real
# systemic problem (e.g. a 404, an auth failure) from looping forever.
#
# MAX_REQUEUES_PER_TILE: how many additional attempts each individual tile may
# receive on top of the in-worker retries already done by the reader.
# MAX_TOTAL_REQUEUE_FRACTION: ceiling on requeues across the whole run as a
# fraction of total tile count. If a quarter of all tiles have already been
# requeued, the next failure surfaces immediately — the run is in trouble.
MAX_REQUEUES_PER_TILE = 2
MAX_TOTAL_REQUEUE_FRACTION = 0.25

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


def _empty_output_tile(
    spec: Tuple[int, int, int, int],
    bands_count: int,
    out_dtype: "np.dtype[Any]",
    include_observation_count: bool,
) -> npt.NDArray[Any]:
    dtype = (
        np.promote_types(out_dtype, np.dtype(np.uint16))
        if include_observation_count
        else out_dtype
    )
    count = bands_count + (1 if include_observation_count else 0)
    return _empty_tile(spec, count, dtype)


def _read_scene_bands(
    read_fn: ReaderFn,
    scene_idx: int,
    bands_count: int,
    spec: Tuple[int, int, int, int],
    band_executor: Optional[Executor] = None,
) -> List[npt.NDArray[Any]]:
    """Read all bands for one (scene, tile) concurrently.

    Per-tile band reads are network-bound. When a shared executor is supplied,
    issuing them in parallel lets HTTP/2 multiplex range requests without
    constructing a new thread pool for every scene/tile pair. For
    ``bands_count == 1`` we skip the executor overhead entirely.
    """
    if bands_count <= 1:
        return [read_fn(scene_idx, 0, spec)]
    if band_executor is None:
        return [read_fn(scene_idx, j, spec) for j in range(bands_count)]
    return list(
        band_executor.map(lambda j: read_fn(scene_idx, j, spec), range(bands_count))
    )


def _source_valid_from_bands(
    band_data: List[npt.NDArray[Any]],
) -> Optional[npt.NDArray[np.bool_]]:
    """Pixels with at least one non-zero band in a multi-band source read."""
    if len(band_data) <= 1:
        return None
    valid = np.zeros(band_data[0].shape, dtype=bool)
    for data in band_data:
        valid |= data != 0
    return valid


def _append_observation_count(
    tile_data: npt.NDArray[Any],
    count: npt.NDArray[Any],
) -> npt.NDArray[Any]:
    """Append a uint16-safe observation-count band to a tile result."""
    out_dtype = np.promote_types(tile_data.dtype, np.dtype(np.uint16))
    out = np.empty((tile_data.shape[0] + 1, *tile_data.shape[1:]), dtype=out_dtype)
    out[: tile_data.shape[0]] = tile_data.astype(out_dtype, copy=False)
    out[-1] = count.astype(out_dtype, copy=False)
    return out


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


@njit(cache=True, nogil=True)  # type: ignore[untyped-decorator]
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
    ``nogil=True`` lets those Python tile-worker threads actually run this
    kernel in parallel rather than serialising on the GIL.
    """
    n_scenes, n_bands, height, width = stack.shape
    out = np.empty((n_bands, height, width), dtype=np.float32)
    # ``values`` is hoisted out of the per-pixel loop. Numba's allocator
    # holds an internal lock per ``np.empty`` call that nogil does not
    # release — allocating once per pixel was serialising tile workers
    # on that lock. One allocation per kernel call keeps multi-thread
    # scaling clean and is also ~17% faster single-thread.
    values = np.empty(n_scenes, dtype=np.float32)
    total = n_bands * height * width

    for idx in range(total):
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


@njit(cache=True, nogil=True)  # type: ignore[untyped-decorator]
def _medoid_axis0_u16(
    stack: npt.NDArray[np.uint16],
    valid: npt.NDArray[np.bool_],
) -> Tuple[npt.NDArray[np.uint16], npt.NDArray[np.bool_]]:
    """Per-pixel medoid composite over uint16 scene stack.

    For each pixel, picks the scene whose multi-band spectrum is closest
    (squared Euclidean) to the per-band median spectrum computed across
    all valid scenes at that pixel. The result is always an actually
    observed spectrum — band relationships are preserved, which matters
    for downstream indices and classifiers — unlike per-band
    percentile/median which can return a synthetic per-band combination.

    This is the "closest to per-band median" formulation of the medoid
    common in Google Earth Engine tutorials and the gee-community
    libraries (O(S·B) per pixel). It is NOT the strict Flood 2013
    definition, which picks ``arg min_s Σᵢ d(scene_s, scene_i)`` over all
    pairs (O(S²·B) per pixel). The two often agree, but can pick different
    scenes when the cluster of observations is asymmetric.

    Inputs:
        stack: shape ``(scene, band, height, width)`` uint16. Values at
            invalid (scene, pixel) positions are ignored — they must be
            flagged via ``valid``.
        valid: shape ``(scene, height, width)`` bool. ``True`` where scene
            is a candidate (all bands present) for that pixel.

    Returns:
        out: ``(band, height, width)`` uint16 containing the chosen
            spectrum at each pixel. Pixels with no candidate are zeroed.
        out_valid: ``(height, width)`` bool, true where a candidate was
            chosen. Used by the caller to propagate the no-data mask.

    Implementation notes:
        - Stripe-blocked two-pass kernel — keeps the scene-outer scoring
          pattern but limits scratch arrays to ``MEDOID_STRIPE_HEIGHT`` rows
          at a time.
        - ``nogil=True`` so Python tile workers can run in parallel
          inside this kernel rather than serialising on the GIL.
        - ``values`` is hoisted out of the per-pixel loop. Numba's
          allocator takes an internal lock per ``np.empty`` call that
          nogil does not release; one allocation per kernel call keeps
          multi-thread scaling clean (see ``bench_percentile_nogil.py``
          for the same lesson on the quantile kernel).
        - **Doubled-target trick keeps even-count medians exact in
          integer math.** For an even count of valid scenes the true
          median is a half-integer — naively floor-dividing by 2 shifts
          the target by 0.5, which directly shifts squared distances
          and can flip the chosen scene (covered by
          ``TestMedoidAxis0U16.test_even_count_median_uses_exact_half_integer_target``).
          Instead the kernel stores the target *doubled*: even-count
          targets are ``values[mid-1] + values[mid]`` and odd-count
          targets are ``2 * values[mid]``. Pass 2 then computes
          ``diff = 2 * stack[s,b,y,x] - target[b,y,x]`` so both operands
          are at the same scale; all per-scene distances scale by the
          same factor of 4, so argmin is identical to the true-median
          ranking with no floats and no precision loss.
        - Squared-distance accumulator is int64 to absorb worst-case
          doubled-diff² × bands. Doubled diffs reach ~2·65535, squared
          ~1.7e10; summed over 13 bands ≈ 2.2e11 — well within int64.
    """
    n_scenes, n_bands, h, w = stack.shape

    out = np.zeros((n_bands, h, w), dtype=np.uint16)
    out_valid = np.zeros((h, w), dtype=np.bool_)
    values = np.empty(n_scenes, dtype=np.uint16)

    for y0 in range(0, h, MEDOID_STRIPE_HEIGHT):
        y1 = min(h, y0 + MEDOID_STRIPE_HEIGHT)
        rows = y1 - y0

        # Pass 1: per-band median target via insertion sort over the valid
        # scenes for each pixel. Targets are stored doubled, so even-count
        # half-integer medians stay exact without switching the distance
        # kernel to float.
        target = np.zeros((n_bands, rows, w), dtype=np.int32)
        for b in range(n_bands):
            for yy in range(rows):
                y = y0 + yy
                for x in range(w):
                    n_valid = 0
                    for s in range(n_scenes):
                        if valid[s, y, x]:
                            values[n_valid] = stack[s, b, y, x]
                            n_valid += 1
                    if n_valid == 0:
                        continue
                    for i in range(1, n_valid):
                        key = values[i]
                        j = i - 1
                        while j >= 0 and values[j] > key:
                            values[j + 1] = values[j]
                            j -= 1
                        values[j + 1] = key
                    mid = n_valid // 2
                    if n_valid % 2 == 1:
                        target[b, yy, x] = np.int32(2) * np.int32(values[mid])
                    else:
                        target[b, yy, x] = np.int32(values[mid - 1]) + np.int32(
                            values[mid]
                        )

        # Pass 2: running best per pixel within this stripe, iterating scenes
        # outermost for cache-friendly stack access.
        best_dist = np.full((rows, w), np.iinfo(np.int64).max, dtype=np.int64)
        best_idx = np.full((rows, w), -1, dtype=np.int32)
        for s in range(n_scenes):
            for yy in range(rows):
                y = y0 + yy
                for x in range(w):
                    if not valid[s, y, x]:
                        continue
                    d = np.int64(0)
                    for b in range(n_bands):
                        diff = (
                            np.int32(2) * np.int32(stack[s, b, y, x]) - target[b, yy, x]
                        )
                        d += np.int64(diff) * np.int64(diff)
                    if d < best_dist[yy, x]:
                        best_dist[yy, x] = d
                        best_idx[yy, x] = s

        for yy in range(rows):
            y = y0 + yy
            for x in range(w):
                s = best_idx[yy, x]
                if s >= 0:
                    out_valid[y, x] = True
                    for b in range(n_bands):
                        out[b, y, x] = stack[s, b, y, x]

    return out, out_valid


def _warm_medoid_axis0_u16() -> None:
    """Compile the medoid kernel on the main thread before workers start."""
    sample_stack = np.zeros((2, 1, 1, 1), dtype=np.uint16)
    sample_stack[0, 0, 0, 0] = 100
    sample_stack[1, 0, 0, 0] = 200
    sample_valid = np.ones((2, 1, 1), dtype=np.bool_)
    _medoid_axis0_u16(sample_stack, sample_valid)


def _copy_single_scene_tile(
    spec: Tuple[int, int, int, int],
    mask_tile: npt.NDArray[Any],
    tile_coverage: npt.NDArray[Any],
    read_fn: ReaderFn,
    scene_idx: int,
    bands_count: int,
    out_dtype: "np.dtype[Any]",
    band_executor: Optional[Executor] = None,
    include_observation_count: bool = False,
) -> npt.NDArray[Any]:
    """Copy one contributing scene into an output tile, zeroing masked pixels."""
    _, _, h, w = spec
    out = np.zeros((bands_count, h, w), dtype=out_dtype)
    pick = mask_tile & tile_coverage
    if not pick.any():
        return out
    band_data = _read_scene_bands(read_fn, scene_idx, bands_count, spec, band_executor)
    source_valid = _source_valid_from_bands(band_data)
    if source_valid is not None:
        pick = pick & source_valid
        if not pick.any():
            count = np.zeros((h, w), dtype=np.uint16)
            if include_observation_count:
                return _append_observation_count(out, count)
            return out
    for j, data in enumerate(band_data):
        np.copyto(out[j], data, where=pick, casting="unsafe")
    if include_observation_count:
        return _append_observation_count(out, pick.astype(np.uint16))
    return out


def _contributing_scene_indices(
    spec: Tuple[int, int, int, int],
    masks: List[Optional[npt.NDArray[Any]]],
    tile_coverage: npt.NDArray[Any],
    min_observations: Optional[int],
    max_observations: Optional[int],
) -> List[int]:
    """Scene indices that contribute to a tile before the early-stop fires.

    Observation counts are driven by the cloud/valid mask. Pixel value ``0`` is
    allowed as source data; unobserved output pixels are zero-filled.
    """
    r, c, h, w = spec
    contributing: List[int] = []
    observation_count: Optional[npt.NDArray[Any]] = None
    effective_target: Optional[int] = None
    if min_observations is not None or max_observations is not None:
        observation_count = np.zeros((h, w), dtype=np.uint16)
        effective_target = (
            min_observations if min_observations is not None else max_observations
        )

    for scene_idx, m in enumerate(masks):
        if m is None:
            continue
        mask_tile = m[r : r + h, c : c + w]
        if not mask_tile.any():
            continue

        if max_observations is not None and observation_count is not None:
            uncapped = observation_count < max_observations
            if not (mask_tile & uncapped).any():
                continue
        contributing.append(scene_idx)

        if observation_count is not None:
            contribution = mask_tile & tile_coverage
            if max_observations is not None:
                contribution = contribution & (observation_count < max_observations)
            np.add(
                observation_count,
                contribution,
                out=observation_count,
                casting="unsafe",
            )
            if ((observation_count >= effective_target) | ~tile_coverage).all():
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
    max_observations: Optional[int],
    out_dtype: "np.dtype[Any]",
    band_executor: Optional[Executor] = None,
    include_observation_count: bool = False,
) -> Tuple[Tuple[int, int, int, int], npt.NDArray[Any]]:
    r, c, h, w = spec
    tile_coverage = coverage_mask[r : r + h, c : c + w]
    if not tile_coverage.any():
        return spec, _empty_output_tile(
            spec, bands_count, out_dtype, include_observation_count
        )

    if bands_count > 1 and (
        min_observations is not None or max_observations is not None
    ):
        contributing = _contributing_scene_indices(
            spec, masks, tile_coverage, None, None
        )
    else:
        contributing = _contributing_scene_indices(
            spec, masks, tile_coverage, min_observations, max_observations
        )

    if not contributing:
        return spec, _empty_output_tile(
            spec, bands_count, out_dtype, include_observation_count
        )

    if len(contributing) == 1:
        scene_idx = contributing[0]
        mask = masks[scene_idx]
        if mask is None:
            raise RuntimeError(f"Missing mask for contributing scene {scene_idx}")
        mask_tile = mask[r : r + h, c : c + w]
        return spec, _copy_single_scene_tile(
            spec,
            mask_tile,
            tile_coverage,
            read_fn,
            scene_idx,
            bands_count,
            out_dtype,
            band_executor,
            include_observation_count,
        )

    stack = np.full((len(contributing), bands_count, h, w), np.nan, dtype=np.float32)
    observation_count = np.zeros((h, w), dtype=np.uint16)
    pixel_count = (
        np.zeros((h, w), dtype=np.uint16) if max_observations is not None else None
    )
    for k, scene_idx in enumerate(contributing):
        mask = masks[scene_idx]
        if mask is None:
            raise RuntimeError(f"Missing mask for contributing scene {scene_idx}")
        mask_tile = mask[r : r + h, c : c + w]
        if pixel_count is not None and max_observations is not None:
            pick = mask_tile & tile_coverage & (pixel_count < max_observations)
        else:
            pick = mask_tile & tile_coverage
        band_data = _read_scene_bands(
            read_fn, scene_idx, bands_count, spec, band_executor
        )
        source_valid = _source_valid_from_bands(band_data)
        if source_valid is not None:
            pick = pick & source_valid
            if not pick.any():
                stack[k].fill(np.nan)
                continue
        for j, data in enumerate(band_data):
            stack[k, j].fill(np.nan)
            np.copyto(stack[k, j], data, where=pick, casting="unsafe")
        np.add(observation_count, pick, out=observation_count, casting="unsafe")
        if pixel_count is not None:
            np.add(pixel_count, pick, out=pixel_count, casting="unsafe")
        if (
            bands_count > 1
            and min_observations is not None
            and max_observations is None
            and ((observation_count >= min_observations) | ~tile_coverage).all()
        ):
            break
        if (
            bands_count > 1
            and max_observations is not None
            and pixel_count is not None
            and ((pixel_count >= max_observations) | ~tile_coverage).all()
        ):
            break

    res = _nanquantile_axis0(stack, percentile / 100.0)
    res = np.nan_to_num(res, nan=0.0)
    tile = _finalise_tile(res, out_dtype)
    if include_observation_count:
        tile = _append_observation_count(tile, observation_count)
    return spec, tile


def tile_medoid(
    spec: Tuple[int, int, int, int],
    masks: List[Optional[npt.NDArray[Any]]],
    read_fn: ReaderFn,
    bands_count: int,
    coverage_mask: npt.NDArray[Any],
    min_observations: Optional[int],
    max_observations: Optional[int],
    out_dtype: "np.dtype[Any]",
    band_executor: Optional[Executor] = None,
    include_observation_count: bool = False,
) -> Tuple[Tuple[int, int, int, int], npt.NDArray[Any]]:
    r, c, h, w = spec
    tile_coverage = coverage_mask[r : r + h, c : c + w]
    if not tile_coverage.any():
        return spec, _empty_output_tile(
            spec, bands_count, out_dtype, include_observation_count
        )

    if bands_count > 1 and (
        min_observations is not None or max_observations is not None
    ):
        contributing = _contributing_scene_indices(
            spec, masks, tile_coverage, None, None
        )
    else:
        contributing = _contributing_scene_indices(
            spec, masks, tile_coverage, min_observations, max_observations
        )

    if not contributing:
        return spec, _empty_output_tile(
            spec, bands_count, out_dtype, include_observation_count
        )

    if len(contributing) == 1:
        scene_idx = contributing[0]
        mask = masks[scene_idx]
        if mask is None:
            raise RuntimeError(f"Missing mask for contributing scene {scene_idx}")
        mask_tile = mask[r : r + h, c : c + w]
        return spec, _copy_single_scene_tile(
            spec,
            mask_tile,
            tile_coverage,
            read_fn,
            scene_idx,
            bands_count,
            out_dtype,
            band_executor,
            include_observation_count,
        )

    # Stack stays uint16 — medoid picks an actual observed spectrum so it
    # needs no fractional precision. Validity carried in a separate bool
    # array instead of NaN sentinels in float32, which halves stack memory
    # and removes the per-cell NaN check from the kernel hot loop.
    stack = np.zeros((len(contributing), bands_count, h, w), dtype=np.uint16)
    valid = np.zeros((len(contributing), h, w), dtype=np.bool_)
    observation_count = np.zeros((h, w), dtype=np.uint16)
    pixel_count = (
        np.zeros((h, w), dtype=np.uint16) if max_observations is not None else None
    )
    for k, scene_idx in enumerate(contributing):
        mask = masks[scene_idx]
        if mask is None:
            raise RuntimeError(f"Missing mask for contributing scene {scene_idx}")
        mask_tile = mask[r : r + h, c : c + w]
        if pixel_count is not None and max_observations is not None:
            pick = mask_tile & tile_coverage & (pixel_count < max_observations)
        else:
            pick = mask_tile & tile_coverage
        band_data = _read_scene_bands(
            read_fn, scene_idx, bands_count, spec, band_executor
        )
        source_valid = _source_valid_from_bands(band_data)
        if source_valid is not None:
            pick = pick & source_valid
            if not pick.any():
                continue
        for j, data in enumerate(band_data):
            np.copyto(stack[k, j], data, where=pick, casting="unsafe")
        valid[k] = pick
        np.add(observation_count, pick, out=observation_count, casting="unsafe")
        if pixel_count is not None:
            np.add(pixel_count, pick, out=pixel_count, casting="unsafe")
        if (
            bands_count > 1
            and min_observations is not None
            and max_observations is None
            and ((observation_count >= min_observations) | ~tile_coverage).all()
        ):
            break
        if (
            bands_count > 1
            and max_observations is not None
            and pixel_count is not None
            and ((pixel_count >= max_observations) | ~tile_coverage).all()
        ):
            break

    res, _ = _medoid_axis0_u16(stack, valid)
    tile = _finalise_tile(res, out_dtype)
    if include_observation_count:
        tile = _append_observation_count(tile, observation_count)
    return spec, tile


def tile_mean(
    spec: Tuple[int, int, int, int],
    masks: List[Optional[npt.NDArray[Any]]],
    read_fn: ReaderFn,
    bands_count: int,
    coverage_mask: npt.NDArray[Any],
    min_observations: Optional[int],
    max_observations: Optional[int],
    out_dtype: "np.dtype[Any]",
    band_executor: Optional[Executor] = None,
    include_observation_count: bool = False,
) -> Tuple[Tuple[int, int, int, int], npt.NDArray[Any]]:
    r, c, h, w = spec
    tile_coverage = coverage_mask[r : r + h, c : c + w]
    if not tile_coverage.any():
        return spec, _empty_output_tile(
            spec, bands_count, out_dtype, include_observation_count
        )

    source_valid_can_change_observations = bands_count > 1 and (
        min_observations is not None or max_observations is not None
    )
    if source_valid_can_change_observations:
        contributing = _contributing_scene_indices(
            spec, masks, tile_coverage, None, None
        )
    else:
        contributing = _contributing_scene_indices(
            spec, masks, tile_coverage, min_observations, max_observations
        )

    if not contributing:
        return spec, _empty_output_tile(
            spec, bands_count, out_dtype, include_observation_count
        )

    if len(contributing) == 1:
        scene_idx = contributing[0]
        mask = masks[scene_idx]
        if mask is None:
            raise RuntimeError(f"Missing mask for contributing scene {scene_idx}")
        mask_tile = mask[r : r + h, c : c + w]
        return spec, _copy_single_scene_tile(
            spec,
            mask_tile,
            tile_coverage,
            read_fn,
            scene_idx,
            bands_count,
            out_dtype,
            band_executor,
            include_observation_count,
        )

    # uint32 accumulator (was float32, same byte width). Skips the per-pixel
    # uint16->float32 cast and uses integer add, which beats numpy's masked
    # add by ~3x on production-sized tiles. The multiply-then-add pattern
    # below replaces the masked add: a uint16 ``pick_u16`` of 0/1 zeros out
    # unwanted pixels before the accumulate, so the inner loop stays a
    # tight vectorised add with no per-element branch.
    sum_block = np.zeros((bands_count, h, w), dtype=np.uint32)
    count = np.zeros((h, w), dtype=np.uint16)
    pick_u16 = np.empty((h, w), dtype=np.uint16)
    for scene_idx in contributing:
        mask = masks[scene_idx]
        if mask is None:
            raise RuntimeError(f"Missing mask for contributing scene {scene_idx}")
        mask_tile = mask[r : r + h, c : c + w]
        if max_observations is not None:
            pick = mask_tile & tile_coverage & (count < max_observations)
            if not pick.any():
                continue
        else:
            pick = mask_tile & tile_coverage
        band_data = _read_scene_bands(
            read_fn, scene_idx, bands_count, spec, band_executor
        )
        source_valid = _source_valid_from_bands(band_data)
        if source_valid is not None:
            pick = pick & source_valid
            if not pick.any():
                continue
        np.copyto(pick_u16, pick.view(np.uint8), casting="unsafe")
        for j, data in enumerate(band_data):
            np.add(sum_block[j], data * pick_u16, out=sum_block[j], casting="unsafe")
        np.add(count, pick_u16, out=count, casting="unsafe")
        if (
            source_valid_can_change_observations
            and min_observations is not None
            and max_observations is None
            and ((count >= min_observations) | ~tile_coverage).all()
        ):
            break
        if (
            source_valid_can_change_observations
            and max_observations is not None
            and ((count >= max_observations) | ~tile_coverage).all()
        ):
            break
    # Integer floor-divide reuses ``sum_block`` as the quotient buffer
    # (avoids a second (bands_count, h, w) allocation). ``safe_count`` is 1
    # at unobserved pixels — sum_block is also 0 there, so 0 // 1 = 0 and
    # no explicit mask is needed.
    safe_count = np.maximum(count, np.uint16(1)).astype(np.uint32, copy=False)
    for b in range(bands_count):
        np.floor_divide(sum_block[b], safe_count, out=sum_block[b])
    tile = _finalise_tile(sum_block, out_dtype)
    if include_observation_count:
        tile = _append_observation_count(tile, count)
    return spec, tile


def tile_first(
    spec: Tuple[int, int, int, int],
    masks: List[Optional[npt.NDArray[Any]]],
    read_fn: ReaderFn,
    bands_count: int,
    coverage_mask: npt.NDArray[Any],
    out_dtype: "np.dtype[Any]",
    band_executor: Optional[Executor] = None,
    include_observation_count: bool = False,
) -> Tuple[Tuple[int, int, int, int], npt.NDArray[Any]]:
    r, c, h, w = spec
    tile_coverage = coverage_mask[r : r + h, c : c + w]
    if not tile_coverage.any():
        return spec, _empty_output_tile(
            spec, bands_count, out_dtype, include_observation_count
        )
    # FIRST copies source pixels straight through, so we can accumulate
    # directly in the output dtype — no float32 working buffer needed.
    result = np.zeros((bands_count, h, w), dtype=out_dtype)
    filled = np.zeros((h, w), dtype=bool)
    for scene_idx, m in enumerate(masks):
        if m is None:
            continue
        mask_tile = m[r : r + h, c : c + w]
        new_pixels = mask_tile & tile_coverage & ~filled
        if not new_pixels.any():
            continue
        band_data = _read_scene_bands(
            read_fn, scene_idx, bands_count, spec, band_executor
        )
        source_valid = _source_valid_from_bands(band_data)
        if source_valid is not None:
            new_pixels = new_pixels & source_valid
            if not new_pixels.any():
                continue
        for j, data in enumerate(band_data):
            result[j][new_pixels] = data[new_pixels]
        filled |= new_pixels
        if (filled | ~tile_coverage).all():
            break
    if include_observation_count:
        result = _append_observation_count(result, filled.astype(np.uint16))
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
    non_empty_masks = sum(1 for m in masks if m is not None)
    if len(specs) * non_empty_masks > EXPECTED_READ_EXACT_SCAN_LIMIT:
        return len(specs) * non_empty_masks * bands_count

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
    mosaic_method: str,
    percentile: Optional[float],
    tile_size: int,
    tile_workers: Optional[int],
    out_dtype: "np.dtype[Any]" = DEFAULT_OUTPUT_DTYPE,
    min_observations: Optional[int] = None,
    max_observations: Optional[int] = None,
    adaptive_tiling: bool = True,
    tile_specs: Optional[List[Tuple[int, int, int, int]]] = None,
    show_progress: bool = False,
    min_tile_size: int = DEFAULT_ADAPTIVE_TILE_MIN_SIZE,
    include_observation_count: bool = False,
) -> npt.NDArray[Any]:
    """Generic streaming aggregation. Called by both grid_id and bounds modes.

    ``out_dtype`` is the pipeline's final output dtype (``uint16`` for
    spectral, ``uint8`` for visual). Tile workers cast to it before
    returning, so the output buffer can be allocated as the final dtype —
    no intermediate float32 array the size of the whole mosaic.

    When ``include_observation_count`` is true, the returned array has one
    extra final band containing per-pixel valid-observation counts. The output
    dtype is promoted to at least ``uint16`` so visual RGB mosaics can carry
    counts above 255 without clipping.
    """
    output_bands_count = bands_count + (1 if include_observation_count else 0)
    output_dtype = (
        np.promote_types(out_dtype, np.dtype(np.uint16))
        if include_observation_count
        else out_dtype
    )
    out = np.zeros((output_bands_count, height, width), dtype=output_dtype)
    for spec, tile_data in iter_tile_aggregation(
        masks=masks,
        read_fn=read_fn,
        bands_count=bands_count,
        height=height,
        width=width,
        coverage_mask=coverage_mask,
        mosaic_method=mosaic_method,
        percentile=percentile,
        tile_size=tile_size,
        tile_workers=tile_workers,
        out_dtype=out_dtype,
        min_observations=min_observations,
        max_observations=max_observations,
        adaptive_tiling=adaptive_tiling,
        tile_specs=tile_specs,
        show_progress=show_progress,
        min_tile_size=min_tile_size,
        include_observation_count=include_observation_count,
    ):
        r, c, h, w = spec
        out[:, r : r + h, c : c + w] = tile_data
    return out


def _drain_with_requeue(
    *,
    specs: List[Tuple[int, int, int, int]],
    worker_fn: Callable[
        [Tuple[int, int, int, int]],
        Tuple[Tuple[int, int, int, int], npt.NDArray[Any]],
    ],
    n_workers: int,
    log_every: int,
) -> Iterator[Tuple[Tuple[int, int, int, int], npt.NDArray[Any]]]:
    """Run ``worker_fn`` for every spec in parallel, requeueing transient failures.

    Yields ``(spec, tile_data)`` in completion order. When a worker raises
    ``RasterioIOError`` the spec is re-submitted to the same executor (so it
    naturally lands at the back of the queue while other tiles fill the gap),
    bounded by :data:`MAX_REQUEUES_PER_TILE` per spec and
    :data:`MAX_TOTAL_REQUEUE_FRACTION` across the run. Any other exception
    propagates immediately.
    """
    total_specs = len(specs)
    max_total_requeues = max(1, int(total_specs * MAX_TOTAL_REQUEUE_FRACTION))
    requeue_counts: Dict[Tuple[int, int, int, int], int] = {}
    total_requeues = 0
    completed = 0

    with ThreadPoolExecutor(max_workers=n_workers) as tile_executor:
        pending: Dict[Future[Any], Tuple[int, int, int, int]] = {
            tile_executor.submit(worker_fn, s): s for s in specs
        }
        while pending:
            done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
            for fut in done:
                spec = pending.pop(fut)
                try:
                    result = fut.result()
                except RasterioIOError as exc:
                    count = requeue_counts.get(spec, 0)
                    if (
                        count < MAX_REQUEUES_PER_TILE
                        and total_requeues < max_total_requeues
                    ):
                        requeue_counts[spec] = count + 1
                        total_requeues += 1
                        logger.warning(
                            "Tile %s failed (requeue %d/%d, %d/%d total): %s",
                            spec,
                            count + 1,
                            MAX_REQUEUES_PER_TILE,
                            total_requeues,
                            max_total_requeues,
                            exc,
                        )
                        pending[tile_executor.submit(worker_fn, spec)] = spec
                        continue
                    raise
                completed += 1
                if completed % log_every == 0 or completed == total_specs:
                    logger.info("Phase 2: %d/%d tiles done", completed, total_specs)
                yield result


def iter_tile_aggregation(
    masks: List[Optional[npt.NDArray[Any]]],
    read_fn: ReaderFn,
    bands_count: int,
    height: int,
    width: int,
    coverage_mask: npt.NDArray[Any],
    mosaic_method: str,
    percentile: Optional[float],
    tile_size: int,
    tile_workers: Optional[int],
    out_dtype: "np.dtype[Any]" = DEFAULT_OUTPUT_DTYPE,
    min_observations: Optional[int] = None,
    max_observations: Optional[int] = None,
    adaptive_tiling: bool = True,
    tile_specs: Optional[List[Tuple[int, int, int, int]]] = None,
    show_progress: bool = False,
    min_tile_size: int = DEFAULT_ADAPTIVE_TILE_MIN_SIZE,
    include_observation_count: bool = False,
) -> Iterator[Tuple[Tuple[int, int, int, int], npt.NDArray[Any]]]:
    """Yield aggregated output tiles without allocating the full mosaic.

    If ``include_observation_count`` is true, each yielded tile includes one
    extra final band containing per-pixel valid-observation counts.
    """
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

    elif mosaic_method == MOSAIC_MEDOID:
        _warm_medoid_axis0_u16()

    elif mosaic_method not in (MOSAIC_MEAN, MOSAIC_FIRST):
        raise ValueError(f"Unknown mosaic_method: {mosaic_method}")

    n_workers = tile_workers if tile_workers is not None else DEFAULT_TILE_WORKERS
    band_read_workers = (
        min(DEFAULT_BAND_READ_WORKERS, max(bands_count, n_workers * bands_count))
        if bands_count > 1
        else 0
    )
    completed = 0
    log_every = max(1, len(specs) // 10)
    band_executor: Optional[Executor] = None

    if mosaic_method == MOSAIC_PERCENTILE:

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
                max_observations,
                out_dtype,
                band_executor,
                include_observation_count,
            )

    elif mosaic_method == MOSAIC_MEAN:

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
                max_observations,
                out_dtype,
                band_executor,
                include_observation_count,
            )

    elif mosaic_method == MOSAIC_MEDOID:

        def worker_fn(
            s: Tuple[int, int, int, int],
        ) -> Tuple[Tuple[int, int, int, int], npt.NDArray[Any]]:
            return tile_medoid(
                s,
                masks,
                effective_read_fn,
                bands_count,
                coverage_mask,
                min_observations,
                max_observations,
                out_dtype,
                band_executor,
                include_observation_count,
            )

    else:

        def worker_fn(
            s: Tuple[int, int, int, int],
        ) -> Tuple[Tuple[int, int, int, int], npt.NDArray[Any]]:
            return tile_first(
                s,
                masks,
                effective_read_fn,
                bands_count,
                coverage_mask,
                out_dtype,
                band_executor,
                include_observation_count,
            )

    band_executor_cm: Optional[ThreadPoolExecutor] = None
    try:
        if band_read_workers > 0:
            band_executor_cm = ThreadPoolExecutor(max_workers=band_read_workers)
            band_executor = band_executor_cm
        if n_workers <= 1:
            tile_results: Iterator[
                Tuple[Tuple[int, int, int, int], npt.NDArray[Any]]
            ] = map(worker_fn, specs)
            for spec, tile_data in tile_results:
                completed += 1
                if completed % log_every == 0 or completed == len(specs):
                    logger.info("Phase 2: %d/%d tiles done", completed, len(specs))
                yield spec, tile_data
        else:
            yield from _drain_with_requeue(
                specs=specs,
                worker_fn=worker_fn,
                n_workers=n_workers,
                log_every=log_every,
            )
    finally:
        if band_executor_cm is not None:
            band_executor_cm.shutdown(wait=True)
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
    mosaic_method: str,
    percentile: Optional[float],
    tile_size: int,
    tile_workers: Optional[int],
    out_dtype: "np.dtype[Any]" = DEFAULT_OUTPUT_DTYPE,
    min_observations: Optional[int] = None,
    max_observations: Optional[int] = None,
    adaptive_tiling: bool = True,
    tile_specs: Optional[List[Tuple[int, int, int, int]]] = None,
    show_progress: bool = False,
    min_tile_size: int = DEFAULT_ADAPTIVE_TILE_MIN_SIZE,
    include_observation_count: bool = False,
) -> Path:
    """Aggregate tiles and write them directly into a GeoTIFF.

    When ``include_observation_count`` is true, append an ``Observation count``
    band after the requested image bands. The GeoTIFF dtype is promoted to at
    least ``uint16`` so visual RGB exports can carry counts above 255.
    """
    band_descriptions, nodata_value = output_band_metadata(bands)
    if include_observation_count:
        band_descriptions = [*band_descriptions, "Observation count"]
    output_bands_count = bands_count + (1 if include_observation_count else 0)
    output_dtype = (
        np.promote_types(out_dtype, np.dtype(np.uint16))
        if include_observation_count
        else out_dtype
    )
    write_profile = profile.copy()
    write_profile.update(
        driver="GTiff",
        width=width,
        height=height,
        count=output_bands_count,
        dtype=output_dtype,
        nodata=nodata_value,
        compress="lzw",
        BIGTIFF="IF_SAFER",
    )
    tmp_file = tempfile.NamedTemporaryFile(
        delete=False,
        dir=export_path.parent,
        prefix=f".{export_path.stem}.{os.getpid()}.{threading.get_ident()}.",
        suffix=export_path.suffix,
    )
    tmp_path = Path(tmp_file.name)
    tmp_file.close()
    logger.info("Writing streamed GeoTIFF to %s via %s", export_path, tmp_path)
    try:
        with rio.Env(GDAL_TIFF_INTERNAL_MASK=True):
            with rio.open(tmp_path, "w", **write_profile) as dst:
                dst.descriptions = band_descriptions
                for spec, tile_data in iter_tile_aggregation(
                    masks=masks,
                    read_fn=read_fn,
                    bands_count=bands_count,
                    height=height,
                    width=width,
                    coverage_mask=coverage_mask,
                    mosaic_method=mosaic_method,
                    percentile=percentile,
                    tile_size=tile_size,
                    tile_workers=tile_workers,
                    out_dtype=out_dtype,
                    min_observations=min_observations,
                    max_observations=max_observations,
                    adaptive_tiling=adaptive_tiling,
                    tile_specs=tile_specs,
                    show_progress=show_progress,
                    min_tile_size=min_tile_size,
                    include_observation_count=include_observation_count,
                ):
                    r, c, h, w = spec
                    if output_coverage_mask is not None:
                        coverage_tile = output_coverage_mask[r : r + h, c : c + w]
                        np.multiply(
                            tile_data,
                            coverage_tile[None, :, :],
                            out=tile_data,
                        )
                    window_cls: Any = Window
                    window = window_cls(c, r, w, h)
                    dst.write(tile_data, window=window)
                    dst.write_mask(
                        output_valid_mask(tile_data, include_observation_count),
                        window=window,
                    )
        tmp_path.replace(export_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return export_path
