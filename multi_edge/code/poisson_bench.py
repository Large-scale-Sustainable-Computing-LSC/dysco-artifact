"""Poisson-arrival harness: sessions arrive continuously, not in one burst.

Arrivals are served by a fixed pool of warm worker processes, since per-session
process startup would otherwise throttle the rate. The arrival rate comes from
Little's law and a measured solo duration; achieved concurrency is reported.
"""

from __future__ import annotations

from typing import Any, Optional
import argparse
import json
import os
import queue
import random
import statistics
import subprocess
import threading
import time

from testset_loader import load_prompts
from scale_bench_stats import HERE, EDGE_PYTHON, HOST, PORT, MAX_NEW_TOKENS, start_server

DATASET = "wildgpt4"
# Sessions used to measure W before the sweep; the first is discarded as the
# one-time cold-load sample.
CALIB_SAMPLES = 6


class WorkerPool:
    """A fixed-size pool of warm edge worker processes."""

    def __init__(
        self,
        size: int,
        model_id: str,
        cut_layer: int | list[int],
        edge_device: str | list[str],
        edge_dtype: str,
        env: dict[str, str],
    ) -> None:
        """Start ``size`` workers, cycling cut layers and devices across them."""
        self.workers = []
        self.free = queue.Queue()
        self.results = {}
        self.events = {}
        self.lock = threading.Lock()
        cuts = cut_layer if isinstance(cut_layer, (list, tuple)) else [cut_layer]
        self.cut_layers = [cuts[i % len(cuts)] for i in range(size)]
        devs = edge_device if isinstance(edge_device, (list, tuple)) else [edge_device]
        self.edge_devices = [devs[i % len(devs)] for i in range(size)]
        for i in range(size):
            proc = subprocess.Popen(
                [EDGE_PYTHON, os.path.join(HERE, "shard01_multi.py"), "--worker",
                 "--model-path", model_id, "--server-model-id", model_id,
                 "--cut-layer", str(self.cut_layers[i]), "--host", HOST, "--port", str(PORT),
                 "--device", self.edge_devices[i], "--dtype", edge_dtype],
                cwd=HERE, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                # Keep stderr: a worker that dies at startup, usually on CUDA out-of-
                # memory, would otherwise look like a timeout.
                stderr=open(os.path.join(HERE, "poisson_workers.err"), "a"), text=True,
            )
            self.workers.append(proc)
            self.free.put(proc)
            threading.Thread(target=self._reader, args=(proc,), daemon=True).start()

    def _reader(self, proc: subprocess.Popen) -> None:
        """Collect result lines from one worker and return it to the free pool."""

        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            result = json.loads(line)
            sid = result["session_id"]
            with self.lock:
                self.results[sid] = result
                ev = self.events.get(sid)
            if ev is not None:
                ev.set()
            self.free.put(proc)

    def submit(self, session_id: str, prompt: str, max_new_tokens: int) -> None:
        """Block until a worker is free, then dispatch the session to it."""
        with self.lock:
            self.events[session_id] = threading.Event()
        proc = self.free.get()
        proc.stdin.write(json.dumps({"session_id": session_id, "prompt": prompt,
                                      "max_new_tokens": max_new_tokens}) + "\n")
        proc.stdin.flush()

    def wait_for(self, session_id: str, timeout: float = 300) -> dict[str, Any]:
        """Wait for one session's result, or raise if the worker never answers."""

        ev = self.events[session_id]
        if not ev.wait(timeout=timeout):
            raise TimeoutError(f"worker never returned a result for session {session_id}")
        with self.lock:
            return self.results.pop(session_id)

    def shutdown(self) -> None:
        """Close every worker's input and wait for it to exit."""

        for p in self.workers:
            try:
                p.stdin.close()
            except Exception:
                pass
        for p in self.workers:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()


def measure_solo_latency(
    cut_layer: int | list[int],
    prompts: list[str],
    model_id: str,
    edge_device: str | list[str],
    edge_dtype: str,
    env: dict[str, str],
    samples: int,
) -> float:
    """Measure W for Little's law: the mean duration of a solo warm session."""
    dev = edge_device[0] if isinstance(edge_device, (list, tuple)) else edge_device
    pool = WorkerPool(1, model_id, cut_layer, dev, edge_dtype, env)
    try:
        durations = []
        for i, prompt in enumerate(prompts[:samples]):
            sid = f"calib{i}"
            pool.submit(sid, prompt, MAX_NEW_TOKENS)
            result = pool.wait_for(sid)
            durations.append(result["gen_end_ts"] - result["gen_start_ts"])
    finally:
        pool.shutdown()
    warm = durations[1:] if len(durations) > 1 else durations
    mean_w = statistics.mean(warm)
    print(f"Calibration: warm decode-phase duration W = {mean_w:.3f}s "
          f"(discarded cold sample {durations[0]:.2f}s; warm samples: {[f'{d:.2f}' for d in warm]})")
    return mean_w


def run_poisson_trial(
    pool: WorkerPool,
    target_concurrency: float,
    service_time_s: float,
    num_sessions: int,
    prompts: list[str],
    trial_tag: str,
) -> tuple[float, float, float]:
    """Fire arrivals with exponential inter-arrival times against a warm pool.

    Returns:
        Aggregate throughput, achieved average concurrency, and mean
        per-session latency.
    """
    rate = target_concurrency / service_time_s
    session_ids = [f"poisson{i}_{trial_tag}" for i in range(num_sessions)]
    submitters = []
    for i in range(num_sessions):
        if i > 0:
            time.sleep(random.expovariate(rate))
        t = threading.Thread(target=pool.submit, args=(session_ids[i], prompts[i], MAX_NEW_TOKENS))
        t.start()
        submitters.append(t)
    for t in submitters:
        t.join()

    results = [pool.wait_for(sid) for sid in session_ids]

    starts = [r["gen_start_ts"] for r in results]
    ends = [r["gen_end_ts"] for r in results]
    durations = [e - s for s, e in zip(starts, ends)]
    span = max(ends) - min(starts)
    aggregate_tok_s = (num_sessions * MAX_NEW_TOKENS) / span
    # Achieved concurrency, measured from the observed intervals. Under FIFO it can exceed
    # the target, since a session's duration includes queueing.
    achieved_concurrency = sum(durations) / span
    mean_latency_s = statistics.mean(durations)
    return aggregate_tok_s, achieved_concurrency, mean_latency_s


def run_condition(
    label: str,
    batch_window_ms: float,
    target_concurrencies: list[float],
    service_time_s: float,
    args: argparse.Namespace,
    prompt_pool: list[str],
    interleave_group_size: int = 0,
    progressive_batching: bool = False,
) -> dict[float, list[tuple[float, float, float]]]:
    """Measure one scheduler at every target concurrency."""

    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    server = start_server(args.model_id, batch_window_ms, env,
                           interleave_group_size=interleave_group_size,
                           progressive_batching=progressive_batching)
    pool = WorkerPool(args.pool_size, args.model_id, args.cut_layers_list, args.edge_devices_list,
                       args.edge_dtype, env)
    print(f"[{label}] worker pool cut layers: {pool.cut_layers}", flush=True)
    print(f"[{label}] worker pool devices:    {pool.edge_devices}", flush=True)
    prompt_idx = 0
    results_by_target = {}
    try:
        for target in target_concurrencies:
            trials = []
            for r in range(args.repeats):
                trial_prompts = [
                    prompt_pool[(prompt_idx + i) % len(prompt_pool)] for i in range(args.num_sessions)
                ]
                prompt_idx += args.num_sessions
                aggregate_tok_s, achieved_concurrency, mean_latency_s = run_poisson_trial(
                    pool, target, service_time_s, args.num_sessions, trial_prompts,
                    trial_tag=f"{label}_L{target}_r{r}",
                )
                trials.append((aggregate_tok_s, achieved_concurrency, mean_latency_s))
                print(f"[{label}] target_L={target} repeat={r+1}/{args.repeats}: "
                      f"aggregate_tok_s={aggregate_tok_s:.2f} achieved_L={achieved_concurrency:.2f} "
                      f"mean_latency_s={mean_latency_s:.3f}", flush=True)
            results_by_target[target] = trials
    finally:
        pool.shutdown()
        server.terminate()
        try:
            server.wait(timeout=10)
        except Exception:
            server.kill()

    return results_by_target


def summarize(label: str, results_by_target: dict[float, list[tuple[float, float, float]]]) -> dict[float, dict[str, float]]:
    """Print and return mean and standard deviation per target concurrency."""

    print(f"\n=== {label}: summary (mean +/- stdev over repeats) ===")
    print(f"{'target L':>9}  {'repeats':>8}  {'aggregate tok/s':>22}  {'achieved L':>16}  {'mean latency (s)':>18}")
    summary = {}
    for target, trials in results_by_target.items():
        tok_vals = [t[0] for t in trials]
        conc_vals = [t[1] for t in trials]
        lat_vals = [t[2] for t in trials]
        tok_mean = statistics.mean(tok_vals)
        tok_std = statistics.stdev(tok_vals) if len(tok_vals) > 1 else 0.0
        conc_mean = statistics.mean(conc_vals)
        conc_std = statistics.stdev(conc_vals) if len(conc_vals) > 1 else 0.0
        lat_mean = statistics.mean(lat_vals)
        lat_std = statistics.stdev(lat_vals) if len(lat_vals) > 1 else 0.0
        summary[target] = {"tok_s_mean": tok_mean, "tok_s_std": tok_std,
                            "achieved_l_mean": conc_mean, "achieved_l_std": conc_std,
                            "mean_latency_s_mean": lat_mean, "mean_latency_s_std": lat_std,
                            "n_trials": len(trials)}
        print(f"{target:>9}  {len(trials):>8}  {tok_mean:>14.2f} +/- {tok_std:<5.2f}  "
              f"{conc_mean:>8.2f} +/- {conc_std:<5.2f}  {lat_mean:>10.3f} +/- {lat_std:<5.3f}")
    return summary




def estimate_worker_bytes(
    model_id: str,
    cut_layer: int,
    dtype_bytes: int = 2,
    ctx_overhead: float = 0.6e9,
) -> float:
    """Estimate the memory one worker holds, to decide whether a pool fits."""
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(model_id)
    h, inter, V = c.hidden_size, c.intermediate_size, c.vocab_size
    nkv = getattr(c, "num_key_value_heads", c.num_attention_heads)
    hd = getattr(c, "head_dim", h // c.num_attention_heads)
    per_layer = 2 * h * h + 2 * h * nkv * hd + 3 * h * inter
    return (V * h + cut_layer * per_layer) * dtype_bytes + ctx_overhead


def autospread_devices(
    devices: list[str],
    cut_layers: list[int],
    pool_size: int,
    model_id: str,
    dtype_bytes: int,
    server_device: str = "cuda:0",
) -> list[str]:
    """Extend the device list if the pool does not fit on the devices requested.

    Returns:
        The device list, extended with further CUDA devices if needed.
    """
    if len(devices) > 1:
        return devices
    need = sum(estimate_worker_bytes(model_id, cut_layers[i % len(cut_layers)], dtype_bytes)
               for i in range(pool_size))
    # Query in a subprocess: mem_get_info would pin a CUDA context on every card in this
    # process.
    probe = ("import torch,json;"
             "print(json.dumps({f'cuda:{i}': torch.cuda.mem_get_info(i)[0] "
             "for i in range(torch.cuda.device_count())}))")
    try:
        out = subprocess.run([EDGE_PYTHON, "-c", probe], capture_output=True, text=True, timeout=180)
        free = json.loads(out.stdout.strip().splitlines()[-1])
    except Exception as exc:
        print(f"could not query free GPU memory ({exc}); leaving pool on {devices}")
        return devices
    if need <= 0.85 * free.get(devices[0], 0):
        return devices
    if server_device in devices:
        print(f"WARNING: edge workers were pointed at the cloud server's own GPU ({server_device}); "
              f"they will contend with the forward passes being measured.")
    # Never spread onto the server's GPU: workers there would contend with the calls under
    # measurement.
    extra = [d for d in sorted(free, key=lambda d: -free[d])
             if d not in devices and d != server_device]
    chosen = devices + extra
    total = sum(free.get(d, 0) for d in chosen)
    if total < need:
        print(f"WARNING: {chosen} has only {total/1e9:.1f} GB free for a pool needing "
              f"~{need/1e9:.1f} GB - reduce --pool-size or use shallower --cut-layers.")
    print(f"NOTE: pool needs ~{need/1e9:.1f} GB of edge weights but {devices[0]} has only "
          f"{free.get(devices[0],0)/1e9:.1f} GB free - spreading across {chosen} "
          f"({total/1e9:.1f} GB free in total). A heterogeneous pool is much heavier than a "
          f"homogeneous shallow-cut one; this is the same failure that killed an earlier run.")
    return chosen

def main() -> None:
    """Run every requested scheduler under Poisson arrivals and report results."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--cut-layers", default="2",
                         help="Comma-separated cut layers cycled across the pool, e.g. "
                              "'2,3,4,5,6,7,8,9'.")
    parser.add_argument("--edge-devices", default="cuda:1",
                         help="Comma-separated devices cycled across the pool, e.g. "
                              "'cuda:1,cuda:2'. Note torch orders devices fastest-first, not by "
                              "PCI id; check "
                              "torch.cuda.get_device_properties before assuming an index.")
    parser.add_argument("--edge-dtype", default="bfloat16")
    parser.add_argument("--target-concurrency", default="2,4,8",
                         help="Comma-separated target average concurrency levels (Little's law L).")
    parser.add_argument("--num-sessions", type=int, default=24,
                         help="Sessions launched per trial (one Poisson-arrival run).")
    parser.add_argument("--repeats", type=int, default=20, help="Independent trials per target concurrency.")
    parser.add_argument("--pool-size", type=int, default=16,
                         help="Warm edge worker processes, the concurrency ceiling. Keep above the "
                              "highest --target-concurrency and within edge GPU memory.")
    parser.add_argument("--conditions", default="fifo,batch,interleave",
                         help="Comma-separated subset of {fifo,batch,interleave,progressive-sc}.")
    parser.add_argument("--results-json", default=None)
    args = parser.parse_args()

    args.conditions_list = [c.strip() for c in args.conditions.split(",")]
    targets = [float(x) for x in args.target_concurrency.split(",")]

    args.edge_devices_list = [d.strip() for d in args.edge_devices.split(",")]
    args.cut_layers_list = [int(x) for x in args.cut_layers.split(",")]
    print(f"Worker pool: cut layers {args.cut_layers_list} cycled across "
          f"{args.pool_size} warm workers.")
    args.edge_devices_list = autospread_devices(
        args.edge_devices_list, args.cut_layers_list, args.pool_size, args.model_id,
        dtype_bytes=(4 if args.edge_dtype == "float32" else 2))
    # One W for the whole run, calibrated on the fleet's median cut; achieved concurrency
    # is measured and reported.
    calib_cut = sorted(args.cut_layers_list)[len(args.cut_layers_list) // 2]

    total_needed = CALIB_SAMPLES + len(args.conditions_list) * len(targets) * args.repeats * args.num_sessions
    print(f"Loading {total_needed} prompts from {DATASET} ...")
    prompt_pool = load_prompts(DATASET, root=None, limit=total_needed)
    print(f"Loaded {len(prompt_pool)} prompts.")
    if len(prompt_pool) < total_needed:
        print(f"NOTE: dataset only has {len(prompt_pool)} prompts; each condition's own sweep still draws "
              f"non-overlapping prompts (needs {len(targets) * args.repeats * args.num_sessions}), only "
              f"cross-condition reuse happens, matching scale_bench_stats.py's convention.")

    calib_env = dict(os.environ)
    calib_env["HF_HUB_OFFLINE"] = "1"
    calib_env["PYTHONUNBUFFERED"] = "1"
    calib_server = start_server(args.model_id, batch_window_ms=0, env=calib_env)
    try:
        service_time_s = measure_solo_latency(calib_cut, prompt_pool, args.model_id,
                                               args.edge_devices_list[0],
                                               args.edge_dtype, calib_env, CALIB_SAMPLES)
    finally:
        calib_server.terminate()
        try:
            calib_server.wait(timeout=10)
        except Exception:
            calib_server.kill()
    remaining_pool = prompt_pool[CALIB_SAMPLES:]

    condition_specs = {
        "fifo": ("fifo", dict(batch_window_ms=0)),
        "batch": ("batch", dict(batch_window_ms=8)),
        "interleave": ("interleave", dict(batch_window_ms=0, interleave_group_size=args.pool_size)),
        "progressive-sc": ("progressive-sc", dict(batch_window_ms=8, progressive_batching=True)),
    }
    if len(args.cut_layers_list) == 1 and any(c.startswith("progressive") for c in args.conditions_list):
        print("WARNING: a progressive condition was requested on a homogeneous fleet. "
              "Depth-synchronized batching reduces exactly to exact-match batching when every "
              "session shares a cut layer, so this measures the same schedule twice. "
              "Pass --cut-layers to make the comparison meaningful.")
    unknown = set(args.conditions_list) - set(condition_specs)
    if unknown:
        raise ValueError(f"--conditions has unknown entries {unknown}")

    all_summaries = {}
    all_raw = {}
    for key in args.conditions_list:
        label, kwargs = condition_specs[key]
        print(f"\n{'='*20} {label} ({kwargs}) {'='*20}")
        results_by_target = run_condition(label, kwargs["batch_window_ms"], targets, service_time_s, args,
                                           remaining_pool, interleave_group_size=kwargs.get("interleave_group_size", 0),
                                           progressive_batching=kwargs.get("progressive_batching", False))
        all_raw[label] = results_by_target
        all_summaries[label] = summarize(label, results_by_target)

    print(f"\n{'='*20} FINAL COMPARISON (Poisson arrivals, W={service_time_s:.3f}s, pool_size={args.pool_size}) {'='*20}")
    header = f"{'target L':>9}" + "".join(f"  {c + ' tok/s':>22}" for c in args.conditions_list)
    print(header)
    for target in targets:
        row = f"{target:>9}"
        for c in args.conditions_list:
            s = all_summaries[c][target]
            row += f"  {s['tok_s_mean']:>14.2f} +/- {s['tok_s_std']:<5.2f}"
        print(row)

    print(f"\n{'target L':>9}" + "".join(f"  {c + ' mean latency (s)':>26}" for c in args.conditions_list))
    for target in targets:
        row = f"{target:>9}"
        for c in args.conditions_list:
            s = all_summaries[c][target]
            row += f"  {s['mean_latency_s_mean']:>18.3f} +/- {s['mean_latency_s_std']:<5.3f}"
        print(row)

    if len(args.conditions_list) >= 2:
        first = args.conditions_list[0]
        print(f"\n(throughput improvement relative to '{first}')")
        for c in args.conditions_list[1:]:
            print(f"  {c}:")
            for target in targets:
                base = all_summaries[first][target]["tok_s_mean"]
                other = all_summaries[c][target]["tok_s_mean"]
                pct = (other / base - 1) * 100
                print(f"    target_L={target}: {pct:+.1f}%")

    if args.results_json:
        with open(args.results_json, "w") as f:
            json.dump({"summaries": all_summaries, "raw": all_raw, "service_time_s": service_time_s,
                       "args": vars(args)}, f, indent=2)
        print(f"\nWrote full results to {args.results_json}")


if __name__ == "__main__":
    main()
