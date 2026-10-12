"""Per-phase profile of depth-synchronized decode steps.

Runs one fleet with PROFILE_PROGRESSIVE=1, so the server reports CPU wall time
per phase to scale_stats_server.log: forward launch, cache merge, split back and
bookkeeping. Env: CUTS, DS, WINDOW_MS, EDGE_DEV.
"""

from __future__ import annotations

import os
import statistics

from scale_bench_stats import run_one_trial, start_server
from testset_loader import load_prompts

MODEL_ID = "meta-llama/Llama-3.2-3B"
SESSIONS, REPEATS = 8, 3
CUTS = [int(cut) for cut in os.environ.get("CUTS", "2,3,4,5,6,7,8,9").split(",")]
DATASET = os.environ.get("DS", "wildgpt4")
WINDOW_MS = float(os.environ.get("WINDOW_MS", "8"))
EDGE_DEVICE = os.environ.get("EDGE_DEV", "cuda:1")


def main() -> None:
    """Profile a few decode steps and report the throughput they reached."""

    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PROFILE_PROGRESSIVE"] = "1"

    print(f"dataset: {DATASET}  cuts: {CUTS}", flush=True)
    print(f"batch window: {WINDOW_MS} ms", flush=True)
    pool = load_prompts(DATASET, root=None, limit=REPEATS * SESSIONS)

    server = start_server(MODEL_ID, WINDOW_MS, env,
                          progressive_batching=True)
    trials: list[float] = []
    try:
        for repeat in range(REPEATS):
            prompts = [pool[(repeat * SESSIONS + i) % len(pool)] for i in range(SESSIONS)]
            tok_s, _ = run_one_trial(SESSIONS, CUTS, prompts, MODEL_ID, EDGE_DEVICE, "bfloat16", env,
                                     trial_tag=f"prof_r{repeat}")
            trials.append(tok_s)
            print(f"repeat {repeat+1}/{REPEATS}: {tok_s:.2f} tok/s", flush=True)
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except Exception:
            server.kill()

    print(f"\nmean {statistics.mean(trials):.2f} tok/s (the relative phase costs are the point)")
    print("profile written to scale_stats_server.log")


if __name__ == "__main__":
    main()
