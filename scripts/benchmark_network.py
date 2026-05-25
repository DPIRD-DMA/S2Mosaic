"""Network-bottleneck benchmark for s2mosaic.

Runs a fixed bounds mosaic N times and records wall-clock time, peak memory,
and result-array fingerprint to JSONL. Use SCL masking (one COG read per
scene, no DL inference) so the measurement isolates network/I-O cost.

Usage:
    python scripts/benchmark_network.py --label baseline --runs 3
    python scripts/benchmark_network.py --label gdal-env --runs 3

Results append to scripts/bench_results.jsonl.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from s2mosaic import mosaic


HERE = Path(__file__).parent
RESULTS = HERE / "bench_results.jsonl"

# Bounds chosen for repeatability:
# - ~20km square near Perth — big enough that network I/O dominates the timing
#   instead of Numba JIT + STAC search + Python overhead (a small AOI lets
#   fixed costs swamp the network signal we're trying to measure).
# - 1-month window in winter 2023 with mixed cloud cover so multiple scenes
#   contribute to a mean mosaic (real network work, not a single-scene cache).
# - SCL masking (one COG read per scene) keeps OCM inference out of the timing
#   so the difference between runs reflects network/IO, not GPU/CPU jitter.
BENCH_BOUNDS = (115.75, -32.05, 116.00, -31.85)
BENCH_KWARGS: Dict[str, Any] = dict(
    bounds=BENCH_BOUNDS,
    start_year=2023,
    start_month=6,
    start_day=1,
    duration_months=1,
    bands=["B04", "B03", "B02", "B08"],
    mosaic_method="mean",
    cloud_mask="SCL",  # avoid OCM inference variance
    resolution=10,
    show_progress=False,
)


def peak_rss_mb() -> float:
    # On macOS ru_maxrss is bytes; on Linux it's kilobytes.
    units = 1024 * 1024 if platform.system() == "Darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / units


def array_fingerprint(arr: np.ndarray) -> str:
    """Stable hash so we can confirm tweaks don't change the output."""
    return hashlib.sha1(arr.tobytes()).hexdigest()[:16]


def single_run(label: str, run_idx: int, extra: Dict[str, Any]) -> Dict[str, Any]:
    gc.collect()
    rss_before = peak_rss_mb()
    t0 = time.perf_counter()
    array, profile = mosaic(**BENCH_KWARGS, **extra)
    t1 = time.perf_counter()
    rss_after = peak_rss_mb()
    rec = {
        "label": label,
        "run": run_idx,
        "seconds": round(t1 - t0, 2),
        "rss_mb_peak": round(rss_after, 1),
        "rss_mb_delta": round(rss_after - rss_before, 1),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "fingerprint": array_fingerprint(array),
        "transform_a": profile["transform"].a,
        "crs": str(profile["crs"]),
        "extra": extra,
    }
    del array
    gc.collect()
    return rec


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="Tag for this config")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--tile-workers",
        type=int,
        default=None,
        help="Override tile_workers for this run set",
    )
    args = parser.parse_args()

    extra: Dict[str, Any] = {}
    if args.tile_workers is not None:
        extra["tile_workers"] = args.tile_workers

    print(
        f"== Benchmark {args.label} | runs={args.runs} | extra={extra} ==",
        flush=True,
    )
    print(f"Python={sys.version.split()[0]} platform={platform.platform()}", flush=True)

    results = []
    for i in range(args.runs):
        rec = single_run(args.label, i, extra)
        results.append(rec)
        print(
            f"  run {i + 1}/{args.runs}: {rec['seconds']:.2f}s "
            f"rss={rec['rss_mb_peak']:.0f}MB fp={rec['fingerprint']}",
            flush=True,
        )
        with RESULTS.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    times = [r["seconds"] for r in results]
    fingerprints = {r["fingerprint"] for r in results}
    print(
        f"\nSummary {args.label}: "
        f"min={min(times):.2f}s median={sorted(times)[len(times) // 2]:.2f}s "
        f"max={max(times):.2f}s "
        f"mean={sum(times) / len(times):.2f}s "
        f"fingerprints={len(fingerprints)} "
        f"({'STABLE' if len(fingerprints) == 1 else 'DIFFERED'})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
