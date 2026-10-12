"""Tail-depth sweep parameterized by model, for cross-model replication.

Same design as tail_depth_sweep.py, with tails scaled in proportion to model
depth so models of different depth are measured at matching relative depths.
Env: SWEEP_MODEL, SWEEP_LAYERS, SWEEP_DATASET, SWEEP_REPEATS, SWEEP_TAG.
"""

from __future__ import annotations

from typing import Any, Optional
import json
import math
import os
import random
import statistics
import sys
import time
import torch
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from scale_bench_stats import HERE, run_one_trial, start_server
from testset_loader import load_prompts

MODEL_ID = os.environ.get("SWEEP_MODEL", "meta-llama/Llama-3.2-3B")
LAYERS = int(os.environ.get("SWEEP_LAYERS", "28"))
DATASET = os.environ.get("SWEEP_DATASET", "wildgpt4")
SESSIONS, REPEATS = 8, int(os.environ.get("SWEEP_REPEATS", "6"))
TAG = os.environ.get("SWEEP_TAG", "")
SEED = 20260909
EDGE_DEVICE = "cuda:2"
OUT = os.path.join(HERE, f"stats_tail_depth_sweep{TAG}.json")
MIN_EDGE_MEMORY_GIB = 60

CONDITIONS = {
    "fifo": dict(batch_window_ms=0),
    "batch": dict(batch_window_ms=8),
    "progressive-sc": dict(batch_window_ms=8, progressive_batching=True),
}

# Eight consecutive distinct cut layers ending at `deepest`, with tails scaled from the
# 28-layer grid so models of different depth match in relative terms.
BASE_TAILS = [20, 17, 14, 11, 8, 5, 1]
TAILS = sorted({max(1, round(tail * LAYERS / 28)) for tail in BASE_TAILS}, reverse=True)
DEEPEST = [LAYERS - tail for tail in TAILS]
POINTS = [(f"tail{LAYERS-d:02d}_deep{d}", list(range(d - 7, d + 1)), LAYERS - d) for d in DEEPEST]


def run_point(
    label: str,
    cuts: list[int],
    pool: list[str],
    idx: list[int],
    rng: random.Random,
) -> dict[str, Any]:
    """Measure every scheduler at one point, in randomized order.

    Returns:
        Per-condition mean, standard deviation and raw trials, plus the order
        the conditions ran in.
    """

    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env.pop("PROFILE_PROGRESSIVE", None)

    order = list(CONDITIONS)
    rng.shuffle(order)
    print(f"    condition order: {' -> '.join(order)}", flush=True)
    out: dict[str, Any] = {"_order": order}
    for condition in order:
        config = CONDITIONS[condition]
        server = start_server(MODEL_ID, config["batch_window_ms"], env,
                              progressive_batching=config.get("progressive_batching", False))
        trials: list[float] = []
        try:
            for repeat in range(REPEATS):
                prompts = [pool[(idx[0] + i) % len(pool)] for i in range(SESSIONS)]
                idx[0] += SESSIONS
                tok_s, _ = run_one_trial(SESSIONS, cuts, prompts, MODEL_ID, EDGE_DEVICE, "bfloat16", env,
                                         trial_tag=f"td_{label}_{condition}_r{repeat}")
                trials.append(tok_s)
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except Exception:
                server.kill()
        mean = statistics.mean(trials)
        stdev = statistics.stdev(trials) if len(trials) > 1 else 0.0
        out[condition] = {"mean": mean, "std": stdev, "trials": trials}
        print(f"[{label}] {condition}: {mean:.2f} +/- {stdev:.2f}", flush=True)
    return out


def fit(xs: list[float], ys: list[float]) -> Optional[tuple[float, float]]:
    """Return the correlation and slope of ``ys`` against ``xs``."""

    mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    spread_x = sum((x - mean_x) ** 2 for x in xs)
    denominator = math.sqrt(spread_x * sum((y - mean_y) ** 2 for y in ys))
    if not denominator:
        return None
    return covariance / denominator, covariance / spread_x


def summary(results: dict[str, Any]) -> None:
    """Print the curve measured so far, with its linear fit."""

    print("\n" + "=" * 20 + " TAIL-DEPTH CURVE " + "=" * 20, flush=True)
    print(f"{'shared tail':>12} {'cut layers':>14} {'fifo':>8} {'batch':>8} {'prog-sc':>9} {'sc_vs_batch':>12}",
          flush=True)
    points: list[tuple[int, float]] = []
    for label, cuts, tail in sorted(POINTS, key=lambda point: -point[2]):
        result = results.get(label)
        if not result:
            continue
        fifo, batch, dsb = result["fifo"]["mean"], result["batch"]["mean"], result["progressive-sc"]["mean"]
        gain = (dsb / batch - 1) * 100
        points.append((tail, gain))
        print(f"{tail:>9} lyr {f'{cuts[0]}..{cuts[-1]}':>14} {fifo:>8.2f} {batch:>8.2f} {dsb:>9.2f} "
              f"{gain:>+11.1f}%", flush=True)
    if len(points) >= 3:
        fitted = fit([float(p[0]) for p in points], [p[1] for p in points])
        if fitted:
            correlation, slope = fitted
            print(f"\ncorr(shared tail depth, benefit): r={correlation:+.3f}  n={len(points)}", flush=True)
            print(f"slope: {slope:+.2f}% benefit per extra shared-tail layer", flush=True)


def main() -> None:
    """Run every point not already present in the output file."""

    rng = random.Random(SEED)
    results = json.load(open(OUT)) if os.path.exists(OUT) else {}
    todo = [point for point in POINTS if point[0] not in results]
    rng.shuffle(todo)

    
    capacity = torch.cuda.get_device_properties(int(EDGE_DEVICE.split(":")[1])).total_memory / 2**30
    print(f"edge device {EDGE_DEVICE} = {capacity:.0f} GiB", flush=True)
    if capacity < MIN_EDGE_MEMORY_GIB:
        # The deepest points need roughly 40 GiB of edge weights.
        sys.exit("ABORT: edge device is too small; check CUDA_DEVICE_ORDER.")

    print(f"tail-depth sweep: {len(POINTS)} points, {len(todo)} to run (randomised), seed={SEED}", flush=True)
    print("order: " + ", ".join(point[0] for point in todo), flush=True)
    pool = load_prompts(DATASET, root=None, limit=len(todo) * len(CONDITIONS) * REPEATS * SESSIONS)
    print(f"Loaded {len(pool)} prompts.", flush=True)
    idx = [0]
    started = time.time()

    for position, (label, cuts, tail) in enumerate(todo, 1):
        print(f"\n{'='*16} [{position}/{len(todo)}] {label} cuts={cuts} tail={tail} {'='*16}", flush=True)
        try:
            result = run_point(label, cuts, pool, idx, rng)
        except Exception as exc:
            print(f"!! POINT FAILED {label}: {type(exc).__name__}: {exc}", flush=True)
            continue
        result["_meta"] = {"cut_layers": cuts, "shared_tail": tail, "edge_device": EDGE_DEVICE,
                           "repeats": REPEATS, "n": SESSIONS, "dataset": DATASET, "model": MODEL_ID,
                           "minutes_into_run": (time.time() - started) / 60}
        results[label] = result
        with open(OUT, "w") as handle:
            json.dump(results, handle, indent=2)
        elapsed = (time.time() - started) / position
        print(f"-- done; {len(todo)-position} left; ETA ~{elapsed*(len(todo)-position)/60:.0f} min", flush=True)
        summary(results)

    print(f"\nALL DONE in {(time.time()-started)/60:.1f} min -> {OUT}", flush=True)
    summary(results)


if __name__ == "__main__":
    main()
