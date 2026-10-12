"""Burst-arrival harness: N sessions start together, repeated per fleet size.

One server is started per scheduler; throughput is measured over the whole
burst, so prompt processing is included. The sweeps and the profiler import
start_server and run_one_trial from here. Env: GEN_TOKENS, EDGE_PYTHON.
"""

from __future__ import annotations

from typing import Any, Optional
import argparse
import json
import os
import socket
import statistics
import subprocess
import time

from testset_loader import load_prompts

HERE = os.path.dirname(os.path.abspath(__file__))
EDGE_PYTHON = os.environ.get("EDGE_PYTHON", "python3")
HOST = "127.0.0.1"
PORT = 5087
MAX_NEW_TOKENS = int(os.environ.get("GEN_TOKENS", "16"))
EDGE_COUNTS = [2, 4, 6, 8]
DATASET = "wildgpt4"


def wait_for_port(host: str, port: int, timeout_s: float = 180) -> bool:
    """Wait until ``host:port`` accepts a connection, or the timeout expires."""

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(1)
    return False


def start_server(
    model_id: str,
    batch_window_ms: float,
    env: dict[str, str],
    interleave_group_size: int = 0,
    progressive_batching: bool = False,
) -> subprocess.Popen:
    """Start one cloud server with the flags that select a scheduler."""

    log = open(os.path.join(HERE, "scale_stats_server.log"), "w")
    args = [EDGE_PYTHON, os.path.join(HERE, "shard02_multi.py"),
            "--model-id", model_id, "--host", "[::]", "--port", str(PORT),
            "--device", "cuda:0", "--dtype", "bfloat16", "--batch-window-ms", str(batch_window_ms),
            "--interleave-group-size", str(interleave_group_size)]
    if progressive_batching:
        args.append("--progressive-batching")
    proc = subprocess.Popen(args, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT)
    print(f"server pid={proc.pid} (batch_window_ms={batch_window_ms}, "
          f"interleave_group_size={interleave_group_size}, progressive_batching={progressive_batching}), "
          f"waiting for port {PORT} ...")
    if not wait_for_port(HOST, PORT):
        proc.terminate()
        raise RuntimeError("server did not come up in time; see scale_stats_server.log")
    time.sleep(2)
    return proc


def run_edge(
    session_id: str,
    cut_layer: int,
    prompt: str,
    out_path: str,
    model_id: str,
    edge_device: str,
    edge_dtype: str,
    env: dict[str, str],
) -> subprocess.Popen:
    """Launch one edge process for a single session."""

    args = [
        EDGE_PYTHON, os.path.join(HERE, "shard01_multi.py"),
        "--session-id", session_id,
        "--model-path", model_id, "--server-model-id", model_id,
        "--cut-layer", str(cut_layer),
        "--host", HOST, "--port", str(PORT),
        "--device", edge_device, "--dtype", edge_dtype,
        "--prompt", prompt, "--max-new-tokens", str(MAX_NEW_TOKENS),
        "--out", out_path,
    ]
    return subprocess.Popen(args, cwd=HERE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def run_one_trial(
    n: int,
    cut_layers: list[int],
    prompts: list[str],
    model_id: str,
    edge_device: str,
    edge_dtype: str,
    env: dict[str, str],
    trial_tag: str,
) -> tuple[float, float]:
    """Run one burst of ``n`` sessions and return its throughput and queueing delay.

    Args:
        cut_layers: Cut layer per edge slot, at least ``n`` values. Equal values
            give a homogeneous fleet; distinct values give a heterogeneous one,
            which exact-match batching cannot merge at all.
    """

    procs = []
    out_paths = []
    for i in range(n):
        session_id = f"edge{i}_{trial_tag}"
        out_path = os.path.join(HERE, f"stats_{session_id}.json")
        out_paths.append(out_path)
        procs.append(run_edge(session_id, cut_layers[i], prompts[i], out_path, model_id, edge_device,
                              edge_dtype, env))

    for proc in procs:
        stdout, _ = proc.communicate()
        if proc.returncode != 0:
            print(stdout)
            raise RuntimeError(f"edge failed in trial {trial_tag} (N={n})")

    results = [json.load(open(path)) for path in out_paths]
    for path in out_paths:
        os.remove(path)

    gen_window_s = max(r["gen_end_ts"] for r in results) - min(r["gen_start_ts"] for r in results)
    clean_tok_s = (n * MAX_NEW_TOKENS) / gen_window_s
    avg_wait_ms = sum(r["avg_gpu_queue_wait_ms"] for r in results) / len(results)
    return clean_tok_s, avg_wait_ms


def run_condition(
    label: str,
    batch_window_ms: float,
    args: argparse.Namespace,
    prompt_pool: list[str],
    interleave_group_size: int = 0,
    progressive_batching: bool = False,
) -> dict[int, list[tuple[float, float]]]:
    """Sweep every fleet size for one scheduler against a freshly started server."""

    # No CUDA_VISIBLE_DEVICES restriction: the server needs cuda:0 and the edges
    # need args.edge_device visible at the same time.
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    server = start_server(args.model_id, batch_window_ms, env,
                          interleave_group_size=interleave_group_size,
                          progressive_batching=progressive_batching)
    prompt_idx = 0
    results_by_n: dict[int, list[tuple[float, float]]] = {}
    try:
        for n in EDGE_COUNTS:
            trials: list[tuple[float, float]] = []
            for repeat in range(args.repeats):
                batch_prompts = [prompt_pool[(prompt_idx + i) % len(prompt_pool)] for i in range(n)]
                prompt_idx += n
                clean_tok_s, avg_wait_ms = run_one_trial(
                    n, args.cut_layers_list, batch_prompts, args.model_id, args.edge_device, args.edge_dtype,
                    env, trial_tag=f"{label}_N{n}_r{repeat}",
                )
                trials.append((clean_tok_s, avg_wait_ms))
                print(f"[{label}] N={n} repeat={repeat+1}/{args.repeats}: "
                      f"clean_tok_s={clean_tok_s:.2f} avg_wait_ms={avg_wait_ms:.1f}", flush=True)
            results_by_n[n] = trials
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()

    return results_by_n


def summarize(label: str, results_by_n: dict[int, list[tuple[float, float]]]) -> dict[int, dict[str, float]]:
    """Print and return mean and standard deviation per fleet size."""

    print(f"\n=== {label}: summary (mean +/- stdev over repeats) ===")
    print(f"{'N':>3}  {'repeats':>8}  {'clean_tok/s (mean+/-std)':>26}  {'avg_wait_ms (mean+/-std)':>26}")
    summary: dict[int, dict[str, float]] = {}
    for n, trials in results_by_n.items():
        throughputs = [trial[0] for trial in trials]
        waits = [trial[1] for trial in trials]
        tok_mean = statistics.mean(throughputs)
        tok_std = statistics.stdev(throughputs) if len(throughputs) > 1 else 0.0
        wait_mean = statistics.mean(waits)
        wait_std = statistics.stdev(waits) if len(waits) > 1 else 0.0
        summary[n] = {"clean_tok_s_mean": tok_mean, "clean_tok_s_std": tok_std,
                      "avg_wait_ms_mean": wait_mean, "avg_wait_ms_std": wait_std, "n_trials": len(trials)}
        print(f"{n:>3}  {len(trials):>8}  {tok_mean:>18.2f} +/- {tok_std:<5.2f}  "
              f"{wait_mean:>18.2f} +/- {wait_std:<5.2f}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for the burst sweep."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--cut-layer", type=int, default=2,
                        help="Homogeneous cut layer for every edge slot. Ignored if --cut-layers is given.")
    parser.add_argument("--cut-layers", default=None,
                        help="Comma-separated per-slot cut layers, e.g. '2,2,2,2,14,14,14,14'. "
                             "Needs max(EDGE_COUNTS) values; overrides --cut-layer.")
    parser.add_argument("--edge-device", default="cuda:1")
    parser.add_argument("--edge-dtype", default="bfloat16")
    parser.add_argument("--repeats", type=int, default=50,
                        help="Independent trials per N. Total samples = repeats * sum(EDGE_COUNTS).")
    parser.add_argument("--results-json", default=None, help="Optional path for the full raw results.")
    parser.add_argument("--conditions", default="fifo,batch,interleave",
                        help="Comma-separated subset of {fifo,batch,interleave,progressive-sc}.")
    parser.add_argument("--interleave-group-size", type=int, default=0,
                        help="Rotation size for the interleave condition; 0 resolves to "
                             "max(EDGE_COUNTS), so overflow never triggers.")
    return parser


def main() -> None:
    """Run every requested scheduler and print the comparison."""

    args = build_parser().parse_args()
    args.conditions_list = [condition.strip() for condition in args.conditions.split(",")]

    if args.cut_layers:
        args.cut_layers_list = [int(cut) for cut in args.cut_layers.split(",")]
        if len(args.cut_layers_list) < max(EDGE_COUNTS):
            raise ValueError(f"--cut-layers needs at least {max(EDGE_COUNTS)} values, "
                             f"got {len(args.cut_layers_list)}")
        print(f"Heterogeneous cut layers per edge slot: {args.cut_layers_list[:max(EDGE_COUNTS)]}")
    else:
        args.cut_layers_list = [args.cut_layer] * max(EDGE_COUNTS)

    total_needed = args.repeats * sum(EDGE_COUNTS)
    print(f"Loading {total_needed} prompts from {DATASET} "
          f"({args.repeats} repeats x sum(N)={sum(EDGE_COUNTS)}) ...")
    prompt_pool = load_prompts(DATASET, root=None, limit=total_needed)
    print(f"Loaded {len(prompt_pool)} prompts.")

    interleave_group_size = args.interleave_group_size or max(EDGE_COUNTS)
    condition_specs = {
        "fifo": ("fifo", dict(batch_window_ms=0)),
        "batch": ("batch", dict(batch_window_ms=8)),
        "interleave": ("interleave", dict(batch_window_ms=0, interleave_group_size=interleave_group_size)),
        "progressive-sc": ("progressive-sc", dict(batch_window_ms=8, progressive_batching=True)),
    }
    unknown = set(args.conditions_list) - set(condition_specs)
    if unknown:
        raise ValueError(f"--conditions has unknown entries {unknown}, expected a subset of {set(condition_specs)}")

    all_summaries: dict[str, dict[int, dict[str, float]]] = {}
    all_raw: dict[str, dict[int, list[tuple[float, float]]]] = {}
    for key in args.conditions_list:
        label, kwargs = condition_specs[key]
        print(f"\n{'='*20} {label} ({kwargs}) {'='*20}")
        results_by_n = run_condition(label, kwargs["batch_window_ms"], args, prompt_pool,
                                     interleave_group_size=kwargs.get("interleave_group_size", 0),
                                     progressive_batching=kwargs.get("progressive_batching", False))
        all_raw[label] = results_by_n
        all_summaries[label] = summarize(label, results_by_n)

    print(f"\n{'='*20} FINAL COMPARISON {'='*20}")
    print(f"{'N':>3}" + "".join(f"  {condition + ' clean_tok/s':>22}" for condition in args.conditions_list))
    for n in EDGE_COUNTS:
        row = f"{n:>3}"
        for condition in args.conditions_list:
            stats = all_summaries[condition][n]
            row += f"  {stats['clean_tok_s_mean']:>14.2f} +/- {stats['clean_tok_s_std']:<5.2f}"
        print(row)
    if len(args.conditions_list) >= 2:
        baseline = args.conditions_list[0]
        print(f"\n(improvement relative to '{baseline}')")
        for condition in args.conditions_list[1:]:
            print(f"  {condition}:")
            for n in EDGE_COUNTS:
                base = all_summaries[baseline][n]["clean_tok_s_mean"]
                other = all_summaries[condition][n]["clean_tok_s_mean"]
                print(f"    N={n}: {(other / base - 1) * 100:+.1f}%")

    if args.results_json:
        with open(args.results_json, "w") as handle:
            json.dump({"summaries": all_summaries, "raw": all_raw, "args": vars(args)}, handle, indent=2)
        print(f"\nWrote full results to {args.results_json}")


if __name__ == "__main__":
    main()
