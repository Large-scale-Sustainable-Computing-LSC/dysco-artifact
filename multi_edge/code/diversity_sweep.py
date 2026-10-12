"""Throughput against the number of distinct cut layers, with the tail fixed.

Every fleet keeps max(cut_layer) = 9, so the shared tail stays at 19 of 28
layers and only diversity varies, from 1 to 8. At one distinct cut layer both
schedulers are identical, so that point measures noise. Env: SWEEP_OUT.
"""

from __future__ import annotations

from typing import Any
import json
import os
import random
import statistics
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["HF_HUB_OFFLINE"] = "1"

from scale_bench_stats import HERE, run_one_trial, start_server
from testset_loader import load_prompts

MODEL_ID = "meta-llama/Llama-3.2-3B"
SESSIONS, REPEATS, DATASET = 8, 6, "wildgpt4"
EDGE_DEVICE = "cuda:2"
SEED = 20260909
SHARED_TAIL = 19
OUT = os.path.join(HERE, os.environ.get("SWEEP_OUT", "stats_diversity_sweep.json"))

# Every fleet keeps max(cut_layer) = 9, so only the number of distinct cut
# layers varies while the shared tail stays at 19 layers.
FLEETS = {
    1: [9, 9, 9, 9, 9, 9, 9, 9],
    2: [2, 2, 2, 2, 9, 9, 9, 9],
    3: [2, 2, 2, 5, 5, 5, 9, 9],
    4: [2, 2, 4, 4, 6, 6, 9, 9],
    5: [2, 2, 4, 4, 6, 7, 9, 9],
    6: [2, 2, 3, 4, 6, 7, 9, 9],
    7: [2, 2, 3, 4, 5, 6, 7, 9],
    8: [2, 3, 4, 5, 6, 7, 8, 9],
}
CONDITIONS = {
    "fifo": dict(batch_window_ms=0),
    "batch": dict(batch_window_ms=8),
    "progressive-sc": dict(batch_window_ms=8, progressive_batching=True),
}


def run_point(
    distinct: int,
    cuts: list[int],
    pool: list[str],
    idx: list[int],
    rng: random.Random,
) -> dict[str, Any]:
    """Measure every scheduler on one fleet, in randomized order.

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
    result: dict[str, Any] = {"_order": order}
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
                                         trial_tag=f"div{distinct}_{condition}_r{repeat}")
                trials.append(tok_s)
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except Exception:
                server.kill()
        mean = statistics.mean(trials)
        stdev = statistics.stdev(trials) if len(trials) > 1 else 0.0
        result[condition] = {"mean": mean, "std": stdev, "trials": trials}
        print(f"[k={distinct}] {condition}: {mean:.2f} +/- {stdev:.2f}", flush=True)
    return result


def summary(results: dict[str, Any]) -> None:
    """Print every fleet measured so far and its gain over exact-match batching."""

    print("\n" + "=" * 14 + f" DIVERSITY SWEEP (tail fixed at {SHARED_TAIL} layers) " + "=" * 14, flush=True)
    print(f"{'#distinct':>10} {'cut layers':>26} {'fifo':>8} {'batch':>8} {'prog-sc':>9} {'sc_vs_batch':>12}",
          flush=True)
    for distinct in sorted(FLEETS):
        result = results.get(str(distinct))
        if not result:
            continue
        fifo, batch, dsb = result["fifo"]["mean"], result["batch"]["mean"], result["progressive-sc"]["mean"]
        print(f"{distinct:>10} {','.join(map(str, FLEETS[distinct])):>26} {fifo:>8.2f} {batch:>8.2f} "
              f"{dsb:>9.2f} {(dsb/batch-1)*100:>+11.1f}%", flush=True)


def main() -> None:
    """Run every fleet not already present in the output file."""

    rng = random.Random(SEED)
    results = json.load(open(OUT)) if os.path.exists(OUT) else {}
    todo = [distinct for distinct in FLEETS if str(distinct) not in results]
    rng.shuffle(todo)
    print(f"diversity sweep: {len(FLEETS)} points, {len(todo)} to run (randomised), seed={SEED}", flush=True)
    print(f"order: {todo}", flush=True)
    pool = load_prompts(DATASET, root=None, limit=len(todo) * len(CONDITIONS) * REPEATS * SESSIONS)
    print(f"Loaded {len(pool)} prompts.", flush=True)
    idx = [0]
    started = time.time()

    for position, distinct in enumerate(todo, 1):
        print(f"\n{'='*12} [{position}/{len(todo)}] k={distinct} distinct  cuts={FLEETS[distinct]} {'='*12}",
              flush=True)
        try:
            result = run_point(distinct, FLEETS[distinct], pool, idx, rng)
        except Exception as exc:
            print(f"!! POINT FAILED k={distinct}: {type(exc).__name__}: {exc}", flush=True)
            continue
        result["_meta"] = {"n_distinct": distinct, "cut_layers": FLEETS[distinct], "shared_tail": SHARED_TAIL,
                           "n": SESSIONS, "repeats": REPEATS, "dataset": DATASET, "model": MODEL_ID}
        results[str(distinct)] = result
        with open(OUT, "w") as handle:
            json.dump(results, handle, indent=2)
        elapsed = (time.time() - started) / position
        print(f"-- done; {len(todo)-position} left; ETA ~{elapsed*(len(todo)-position)/60:.0f} min", flush=True)
        summary(results)

    print(f"\nALL DONE in {(time.time()-started)/60:.1f} min -> {OUT}", flush=True)
    summary(results)


if __name__ == "__main__":
    main()
