"""Throughput across 19 fleet compositions at eight sessions.

Homogeneous fleets, a majority with one or two outliers, two- and three-cluster
fleets, and fully spread fleets: where depth-synchronized batching helps, where
it is neutral, and where it costs. Env: SWEEP_OUT.
"""

from __future__ import annotations

from typing import Any
import json
import os
import random
import statistics
import time

from scale_bench_stats import HERE, run_one_trial, start_server
from testset_loader import load_prompts

MODEL_ID = "meta-llama/Llama-3.2-3B"
SESSIONS, REPEATS, DATASET = 8, 6, "wildgpt4"
SEED = 20260909
EDGE_DEVICE = "cuda:1"
OUT_JSON = os.path.join(HERE, os.environ.get("SWEEP_OUT", "stats_fullshape_matrix_v2.json"))

CONDITIONS = {
    "fifo": dict(batch_window_ms=0),
    "batch": dict(batch_window_ms=8),
    "progressive-sc": dict(batch_window_ms=8, progressive_batching=True),
}

# (label, category, cut layers) for every fleet composition in the matrix.
SHAPES: list[tuple[str, str, list[int]]] = []
for depth in [2, 8, 14]:
    SHAPES.append((f"s01_homog_d{depth}", "1_homogeneous", [depth] * 8))
for outlier in [2, 4, 7, 13, 16, 19, 22, 26]:
    SHAPES.append((f"s03_majmid10_out{outlier}", "3_majority_mid", [10] * 7 + [outlier]))
for outlier in [10, 19]:
    SHAPES.append((f"s04_maj6_pair{outlier}", "4_two_outliers_same", [2] * 6 + [outlier, outlier]))
SHAPES += [
    ("s05_maj6_out10_22", "5_two_outliers_diff", [2] * 6 + [10, 22]),
    ("s06_twoclust_2_10", "6_two_clusters_close", [2] * 4 + [10] * 4),
    ("s07_twoclust_2_18", "7_two_clusters_far", [2] * 4 + [18] * 4),
    ("s08_threeclust", "8_three_clusters", [2] * 3 + [12] * 3 + [22] * 2),
    ("s09_spread_shallow", "9_spread_shallow", [2, 3, 4, 5, 6, 7, 8, 9]),
    ("s11_spread_even", "11_spread_even", [2, 4, 7, 10, 13, 16, 19, 22]),
]


def run_point(
    label: str,
    cuts: list[int],
    pool: list[str],
    idx: list[int],
    rng: random.Random,
) -> dict[str, Any]:
    """Measure every scheduler on one fleet composition, in randomized order."""

    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
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
                                         trial_tag=f"v2_{label}_{condition}_r{repeat}")
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


def summary(results: dict[str, Any]) -> None:
    """Print every measured shape and the spread of gains across them."""

    print("\n" + "=" * 26 + " SUMMARY (randomised order) " + "=" * 26, flush=True)
    print(f"{'shape':>24} {'cut layers':>30} {'fifo':>8} {'batch':>8} {'prog-sc':>8} {'sc_vs_batch':>12}",
          flush=True)
    gains: list[float] = []
    for label, _category, cuts in SHAPES:
        result = results.get(label)
        if not result:
            continue
        fifo, batch, dsb = result["fifo"]["mean"], result["batch"]["mean"], result["progressive-sc"]["mean"]
        gain = (dsb / batch - 1) * 100
        gains.append(gain)
        print(f"{label:>24} {','.join(map(str, cuts)):>30} {fifo:>8.2f} {batch:>8.2f} {dsb:>8.2f} "
              f"{gain:>+11.1f}%", flush=True)
    if gains:
        print(f"\nmean sc_vs_batch across {len(gains)} shapes: {statistics.mean(gains):+.2f}%  "
              f"(range {min(gains):+.1f}% .. {max(gains):+.1f}%)  "
              f"positive: {sum(1 for gain in gains if gain > 0)}/{len(gains)}", flush=True)


def main() -> None:
    """Run every fleet composition not already present in the output file."""

    rng = random.Random(SEED)
    results = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
    todo = [shape for shape in SHAPES if shape[0] not in results]
    rng.shuffle(todo)
    print(f"shape matrix: {len(SHAPES)} shapes, {len(todo)} to run, randomised order, seed={SEED}", flush=True)
    print("order: " + ", ".join(shape[0] for shape in todo), flush=True)

    pool = load_prompts(DATASET, root=None, limit=len(todo) * len(CONDITIONS) * REPEATS * SESSIONS)
    print(f"Loaded {len(pool)} prompts.", flush=True)
    idx = [0]
    run_started = time.time()

    for position, (label, category, cuts) in enumerate(todo, 1):
        print(f"\n{'='*18} [{position}/{len(todo)}] {label} cuts={cuts} {'='*18}", flush=True)
        point_started = time.time()
        try:
            result = run_point(label, cuts, pool, idx, rng)
        except Exception as exc:
            print(f"!! POINT FAILED {label}: {type(exc).__name__}: {exc}", flush=True)
            continue
        result["_meta"] = {"shape_category": category, "cut_layers": cuts, "edge_device": EDGE_DEVICE,
                           "repeats": REPEATS, "n": SESSIONS, "dataset": DATASET, "model": MODEL_ID,
                           "run_index": position, "started_epoch": point_started,
                           "minutes_into_run": (point_started - run_started) / 60}
        results[label] = result
        with open(OUT_JSON, "w") as handle:
            json.dump(results, handle, indent=2)
        eta = (time.time() - run_started) / position * (len(todo) - position)
        print(f"-- done; {len(todo)-position} left; ETA ~{eta/60:.0f} min", flush=True)
        summary(results)

    print(f"\nALL DONE in {(time.time()-run_started)/60:.1f} min -> {OUT_JSON}", flush=True)
    summary(results)


if __name__ == "__main__":
    main()
