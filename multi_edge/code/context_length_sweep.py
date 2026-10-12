"""Throughput against prompt length, with the fleet held constant.

Long-reasoning prompts are truncated to each target length, so only length
changes. Fleet: eight sessions at cut layers 1 to 8, a 20-layer shared tail.
Condition and point order are randomized; completed points are resumed.
Env: SWEEP_OUT.
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

from transformers import AutoTokenizer
from scale_bench_stats import HERE, run_one_trial, start_server
from testset_loader import load_prompts

MODEL_ID = "meta-llama/Llama-3.2-3B"
SESSIONS, REPEATS = 8, 6
CUTS = [1, 2, 3, 4, 5, 6, 7, 8]
SHARED_TAIL = 20
EDGE_DEVICE = "cuda:2"
SEED = 20260909
LENGTHS = [64, 128, 256, 512, 1024, 2048, 4096]
OUT = os.path.join(HERE, os.environ.get("SWEEP_OUT", "stats_context_length_sweep.json"))

CONDITIONS = {
    "fifo": dict(batch_window_ms=0),
    "batch": dict(batch_window_ms=8),
    "progressive-sc": dict(batch_window_ms=8, progressive_batching=True),
}


def build_pool(target_tokens: int, tokenizer: Any, raw: list[str], need: int) -> list[str]:
    """Truncate prompts to ``target_tokens`` so length is the only variable."""

    pool: list[str] = []
    for prompt in raw:
        ids = tokenizer(prompt).input_ids[:target_tokens]
        if len(ids) < target_tokens * 0.9:
            continue
        pool.append(tokenizer.decode(ids, skip_special_tokens=True))
        if len(pool) >= need:
            break
    return pool


def run_point(label: str, pool: list[str], rng: random.Random) -> dict[str, Any]:
    """Measure every scheduler at one prompt length, in randomized order."""

    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env.pop("PROFILE_PROGRESSIVE", None)

    order = list(CONDITIONS)
    rng.shuffle(order)
    print(f"    condition order: {' -> '.join(order)}", flush=True)
    result: dict[str, Any] = {"_order": order}
    idx = 0
    for condition in order:
        config = CONDITIONS[condition]
        server = start_server(MODEL_ID, config["batch_window_ms"], env,
                              progressive_batching=config.get("progressive_batching", False))
        trials: list[float] = []
        try:
            for repeat in range(REPEATS):
                prompts = [pool[(idx + i) % len(pool)] for i in range(SESSIONS)]
                idx += SESSIONS
                tok_s, _ = run_one_trial(SESSIONS, CUTS, prompts, MODEL_ID, EDGE_DEVICE, "bfloat16", env,
                                         trial_tag=f"ctx_{label}_{condition}_r{repeat}")
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
        print(f"[{label}] {condition}: {mean:.2f} +/- {stdev:.2f}", flush=True)
    return result


def summary(results: dict[str, Any]) -> None:
    """Print the measured curve and the longest context with a clear benefit."""

    print("\n" + "=" * 18 + " CONTEXT-LENGTH OPERATING RANGE " + "=" * 18, flush=True)
    print(f"{'ctx tokens':>11} {'fifo':>8} {'batch':>8} {'prog-sc':>9} {'sc_vs_batch':>12}", flush=True)
    points: list[tuple[int, float]] = []
    for length in LENGTHS:
        result = results.get(str(length))
        if not result:
            continue
        fifo, batch, dsb = result["fifo"]["mean"], result["batch"]["mean"], result["progressive-sc"]["mean"]
        gain = (dsb / batch - 1) * 100
        points.append((length, gain))
        print(f"{length:>11} {fifo:>8.2f} {batch:>8.2f} {dsb:>9.2f} {gain:>+11.1f}%", flush=True)
    beneficial = [length for length, gain in points if gain > 5]
    if beneficial and len(points) > 1:
        print(f"\n>5% benefit up to {max(beneficial)} tokens; crossing toward zero beyond that.", flush=True)


def main() -> None:
    """Run every prompt length not already present in the output file."""

    rng = random.Random(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    raw = load_prompts("longreason", root=None, limit=400)
    results = json.load(open(OUT)) if os.path.exists(OUT) else {}
    todo = [length for length in LENGTHS if str(length) not in results]
    rng.shuffle(todo)
    print(f"context sweep: {len(LENGTHS)} lengths, {len(todo)} to run (randomised), seed={SEED}", flush=True)
    print(f"order: {todo}", flush=True)
    started = time.time()

    for position, length in enumerate(todo, 1):
        need = len(CONDITIONS) * REPEATS * SESSIONS
        pool = build_pool(length, tokenizer, raw, need)
        if len(pool) < SESSIONS:
            print(f"!! skipping {length}: only {len(pool)} prompts long enough", flush=True)
            continue
        print(f"\n{'='*14} [{position}/{len(todo)}] ctx={length} tokens ({len(pool)} prompts) {'='*14}",
              flush=True)
        try:
            result = run_point(str(length), pool, rng)
        except Exception as exc:
            print(f"!! POINT FAILED ctx={length}: {type(exc).__name__}: {exc}", flush=True)
            continue
        result["_meta"] = {"context_tokens": length, "cut_layers": CUTS, "shared_tail": SHARED_TAIL,
                           "edge_device": EDGE_DEVICE, "repeats": REPEATS, "n": SESSIONS,
                           "source": "longreason truncated", "model": MODEL_ID}
        results[str(length)] = result
        with open(OUT, "w") as handle:
            json.dump(results, handle, indent=2)
        elapsed = (time.time() - started) / position
        print(f"-- done; {len(todo)-position} left; ETA ~{elapsed*(len(todo)-position)/60:.0f} min", flush=True)
        summary(results)

    print(f"\nALL DONE in {(time.time()-started)/60:.1f} min -> {OUT}", flush=True)
    summary(results)


if __name__ == "__main__":
    main()
