"""Benchmark medoid reducer speed and sampled RSS by implementation shape.

This is intentionally separate from the library. It compares allocation
strategies for the uint16 medoid reducer while preserving the production
constraints:

* input stack is uint16 plus a scene-level validity mask
* no Numba ``parallel=True`` / ``prange`` inside the reducer
* medoid target uses exact half-integer medians via doubled integer targets

Run:
    uv run python bench_medoid_memory.py --quick
    uv run python bench_medoid_memory.py

RSS is sampled from a child process with ``ps``. Treat it as an empirical
process-memory signal, not an exact allocator accounting.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
from numba import njit


StackU16 = np.ndarray  # (S, B, H, W) uint16
ValidMask = np.ndarray  # (S, H, W) bool
OutputU16 = np.ndarray  # (B, H, W) uint16
OutValid = np.ndarray  # (H, W) bool
Kernel = Callable[[StackU16, ValidMask], Tuple[OutputU16, OutValid]]


@njit(cache=True, nogil=True)
def _medoid_full_target_set(
    stack: StackU16, valid: ValidMask
) -> Tuple[OutputU16, OutValid]:
    """Current library shape: full target, target_set, best_dist, best_idx."""
    n_scenes, n_bands, h, w = stack.shape
    target = np.zeros((n_bands, h, w), dtype=np.int32)
    target_set = np.zeros((n_bands, h, w), dtype=np.bool_)
    values = np.empty(n_scenes, dtype=np.uint16)

    for b in range(n_bands):
        for y in range(h):
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
                    target[b, y, x] = np.int32(2) * np.int32(values[mid])
                else:
                    target[b, y, x] = np.int32(values[mid - 1]) + np.int32(values[mid])
                target_set[b, y, x] = True

    best_dist = np.full((h, w), np.iinfo(np.int64).max, dtype=np.int64)
    best_idx = np.full((h, w), -1, dtype=np.int32)
    for s in range(n_scenes):
        for y in range(h):
            for x in range(w):
                if not valid[s, y, x]:
                    continue
                d = np.int64(0)
                ok = True
                for b in range(n_bands):
                    if not target_set[b, y, x]:
                        ok = False
                        break
                    diff = np.int32(2) * np.int32(stack[s, b, y, x]) - target[b, y, x]
                    d += np.int64(diff) * np.int64(diff)
                if ok and d < best_dist[y, x]:
                    best_dist[y, x] = d
                    best_idx[y, x] = s

    out = np.zeros((n_bands, h, w), dtype=np.uint16)
    out_valid = np.zeros((h, w), dtype=np.bool_)
    for y in range(h):
        for x in range(w):
            s = best_idx[y, x]
            if s >= 0:
                out_valid[y, x] = True
                for b in range(n_bands):
                    out[b, y, x] = stack[s, b, y, x]
    return out, out_valid


@njit(cache=True, nogil=True)
def _medoid_full_no_target_set(
    stack: StackU16, valid: ValidMask
) -> Tuple[OutputU16, OutValid]:
    """Full target but no target_set array.

    With scene-level validity, a pixel with any valid candidate has a target for
    every band, so ``valid[s, y, x]`` is enough in pass 2.
    """
    n_scenes, n_bands, h, w = stack.shape
    target = np.zeros((n_bands, h, w), dtype=np.int32)
    values = np.empty(n_scenes, dtype=np.uint16)

    for b in range(n_bands):
        for y in range(h):
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
                    target[b, y, x] = np.int32(2) * np.int32(values[mid])
                else:
                    target[b, y, x] = np.int32(values[mid - 1]) + np.int32(values[mid])

    best_dist = np.full((h, w), np.iinfo(np.int64).max, dtype=np.int64)
    best_idx = np.full((h, w), -1, dtype=np.int32)
    for s in range(n_scenes):
        for y in range(h):
            for x in range(w):
                if not valid[s, y, x]:
                    continue
                d = np.int64(0)
                for b in range(n_bands):
                    diff = np.int32(2) * np.int32(stack[s, b, y, x]) - target[b, y, x]
                    d += np.int64(diff) * np.int64(diff)
                if d < best_dist[y, x]:
                    best_dist[y, x] = d
                    best_idx[y, x] = s

    out = np.zeros((n_bands, h, w), dtype=np.uint16)
    out_valid = np.zeros((h, w), dtype=np.bool_)
    for y in range(h):
        for x in range(w):
            s = best_idx[y, x]
            if s >= 0:
                out_valid[y, x] = True
                for b in range(n_bands):
                    out[b, y, x] = stack[s, b, y, x]
    return out, out_valid


@njit(cache=True, nogil=True)
def _medoid_striped(
    stack: StackU16, valid: ValidMask, stripe_h: int
) -> Tuple[OutputU16, OutValid]:
    """Stripe-blocked target/best arrays to reduce scratch RSS."""
    n_scenes, n_bands, h, w = stack.shape
    out = np.zeros((n_bands, h, w), dtype=np.uint16)
    out_valid = np.zeros((h, w), dtype=np.bool_)
    values = np.empty(n_scenes, dtype=np.uint16)

    for y0 in range(0, h, stripe_h):
        y1 = min(h, y0 + stripe_h)
        rows = y1 - y0
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


@njit(cache=True, nogil=True)
def _medoid_pixel_scratch(
    stack: StackU16, valid: ValidMask
) -> Tuple[OutputU16, OutValid]:
    """Lowest scratch-memory shape: only O(S + B) scratch."""
    n_scenes, n_bands, h, w = stack.shape
    out = np.zeros((n_bands, h, w), dtype=np.uint16)
    out_valid = np.zeros((h, w), dtype=np.bool_)
    values = np.empty(n_scenes, dtype=np.uint16)
    target = np.empty(n_bands, dtype=np.int32)

    for y in range(h):
        for x in range(w):
            any_valid = False
            for b in range(n_bands):
                n_valid = 0
                for s in range(n_scenes):
                    if valid[s, y, x]:
                        values[n_valid] = stack[s, b, y, x]
                        n_valid += 1
                if n_valid == 0:
                    target[b] = np.int32(0)
                    continue
                any_valid = True
                for i in range(1, n_valid):
                    key = values[i]
                    j = i - 1
                    while j >= 0 and values[j] > key:
                        values[j + 1] = values[j]
                        j -= 1
                    values[j + 1] = key
                mid = n_valid // 2
                if n_valid % 2 == 1:
                    target[b] = np.int32(2) * np.int32(values[mid])
                else:
                    target[b] = np.int32(values[mid - 1]) + np.int32(values[mid])

            if not any_valid:
                continue

            best_idx = -1
            best_dist = np.iinfo(np.int64).max
            for s in range(n_scenes):
                if not valid[s, y, x]:
                    continue
                d = np.int64(0)
                for b in range(n_bands):
                    diff = np.int32(2) * np.int32(stack[s, b, y, x]) - target[b]
                    d += np.int64(diff) * np.int64(diff)
                if d < best_dist:
                    best_dist = d
                    best_idx = s

            if best_idx >= 0:
                out_valid[y, x] = True
                for b in range(n_bands):
                    out[b, y, x] = stack[best_idx, b, y, x]

    return out, out_valid


def medoid_full_target_set(
    stack: StackU16, valid: ValidMask
) -> Tuple[OutputU16, OutValid]:
    return _medoid_full_target_set(stack, valid)


def medoid_full_no_target_set(
    stack: StackU16, valid: ValidMask
) -> Tuple[OutputU16, OutValid]:
    return _medoid_full_no_target_set(stack, valid)


def medoid_stripe_64(stack: StackU16, valid: ValidMask) -> Tuple[OutputU16, OutValid]:
    return _medoid_striped(stack, valid, 64)


def medoid_stripe_128(stack: StackU16, valid: ValidMask) -> Tuple[OutputU16, OutValid]:
    return _medoid_striped(stack, valid, 128)


def medoid_stripe_256(stack: StackU16, valid: ValidMask) -> Tuple[OutputU16, OutValid]:
    return _medoid_striped(stack, valid, 256)


def medoid_pixel_scratch(
    stack: StackU16, valid: ValidMask
) -> Tuple[OutputU16, OutValid]:
    return _medoid_pixel_scratch(stack, valid)


IMPLS: Dict[str, Kernel] = {
    "full_target_set": medoid_full_target_set,
    "full_no_target_set": medoid_full_no_target_set,
    "stripe_64": medoid_stripe_64,
    "stripe_128": medoid_stripe_128,
    "stripe_256": medoid_stripe_256,
    "pixel_scratch": medoid_pixel_scratch,
}


def make_stack(
    n_scenes: int,
    n_bands: int,
    h: int,
    w: int,
    valid_fraction: float,
    seed: int,
) -> Tuple[StackU16, ValidMask]:
    rng = np.random.default_rng(seed)
    stack = rng.integers(0, 12000, size=(n_scenes, n_bands, h, w), dtype=np.uint16)
    valid = rng.random((n_scenes, h, w)) < valid_fraction
    return stack, valid


def _rss_bytes() -> int:
    out = subprocess.check_output(
        ["ps", "-o", "rss=", "-p", str(os.getpid())], text=True
    )
    return int(out.strip()) * 1024


class RssSampler:
    def __init__(self, interval_s: float = 0.005) -> None:
        self.interval_s = interval_s
        self.samples: List[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(_rss_bytes())
            except Exception:
                pass
            time.sleep(self.interval_s)

    def __enter__(self) -> "RssSampler":
        self.samples.append(_rss_bytes())
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.samples.append(_rss_bytes())
        self._stop.set()
        self._thread.join()
        self.samples.append(_rss_bytes())

    @property
    def peak(self) -> int:
        return max(self.samples) if self.samples else 0


def _warm_impl(name: str, fn: Kernel, n_bands: int) -> None:
    sample_stack = np.zeros((2, n_bands, 2, 2), dtype=np.uint16)
    sample_stack[0] = 100
    sample_stack[1] = 200
    sample_valid = np.ones((2, 2, 2), dtype=np.bool_)
    fn(sample_stack, sample_valid)


def run_child(args: argparse.Namespace) -> None:
    fn = IMPLS[args.impl]
    _warm_impl(args.impl, fn, args.bands)
    stack, valid = make_stack(
        args.scenes, args.bands, args.size, args.size, args.valid_fraction, args.seed
    )
    gc.collect()
    base_rss = _rss_bytes()

    with RssSampler(interval_s=args.sample_interval) as sampler:
        t0 = time.perf_counter()
        out, out_valid = fn(stack, valid)
        seconds = time.perf_counter() - t0
        checksum = int(out.sum(dtype=np.uint64) % np.uint64(1_000_000_007))
        checksum += int(out_valid.sum())

    end_rss = _rss_bytes()
    result = {
        "impl": args.impl,
        "size": args.size,
        "scenes": args.scenes,
        "bands": args.bands,
        "stack_mb": stack.nbytes / 1e6,
        "valid_mb": valid.nbytes / 1e6,
        "seconds": seconds,
        "mpix_s": (args.size * args.size / 1e6) / seconds,
        "base_rss_mb": base_rss / 1e6,
        "peak_delta_mb": max(0, sampler.peak - base_rss) / 1e6,
        "end_delta_mb": max(0, end_rss - base_rss) / 1e6,
        "checksum": checksum,
    }
    print(json.dumps(result), flush=True)


@dataclass(frozen=True)
class Config:
    size: int
    scenes: int
    bands: int


def run_parent(args: argparse.Namespace) -> None:
    if args.quick:
        configs = [Config(512, 10, 6), Config(1024, 10, 6)]
    else:
        configs = [
            Config(512, 10, 6),
            Config(1024, 10, 6),
            Config(1024, 20, 10),
            Config(2048, 10, 6),
            Config(2048, 15, 10),
        ]

    impls = args.impls or list(IMPLS)
    header = (
        f"{'impl':<18}{'size':>10}{'S':>4}{'B':>4}{'stack':>9}"
        f"{'sec':>9}{'Mpix/s':>9}{'peakMB':>9}{'endMB':>8}"
    )
    print(header)
    print("-" * len(header))

    for cfg in configs:
        for impl in impls:
            cmd = [
                sys.executable,
                __file__,
                "--child",
                "--impl",
                impl,
                "--size",
                str(cfg.size),
                "--scenes",
                str(cfg.scenes),
                "--bands",
                str(cfg.bands),
                "--valid-fraction",
                str(args.valid_fraction),
                "--seed",
                str(args.seed),
                "--sample-interval",
                str(args.sample_interval),
            ]
            proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
            result = json.loads(proc.stdout.strip().splitlines()[-1])
            print(
                f"{impl:<18}{cfg.size:>6}x{cfg.size:<3}"
                f"{cfg.scenes:>4}{cfg.bands:>4}{result['stack_mb']:>9.0f}"
                f"{result['seconds']:>9.3f}{result['mpix_s']:>9.2f}"
                f"{result['peak_delta_mb']:>9.0f}{result['end_delta_mb']:>8.0f}",
                flush=True,
            )
        print(flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--impl", choices=sorted(IMPLS))
    parser.add_argument("--impls", nargs="+", choices=sorted(IMPLS))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--scenes", type=int, default=10)
    parser.add_argument("--bands", type=int, default=6)
    parser.add_argument("--valid-fraction", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-interval", type=float, default=0.005)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.child:
        if args.impl is None:
            raise SystemExit("--child requires --impl")
        run_child(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
