"""Measured GPU memory cost of the cloud-side key/value caches.

Part A varies concurrent sessions, part B generation length. A background thread
polls GetMemoryStats while sessions generate and keeps the peak, so the reported
cache size is summed from real tensors. Env: GEN_TOKENS, EDGE_PYTHON.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import argparse
import os
import socket
import subprocess
import threading
import time

import grpc

import llama_multi_pb2
import llama_multi_pb2_grpc
from testset_loader import load_prompts

HERE = os.path.dirname(os.path.abspath(__file__))
EDGE_PYTHON = os.environ.get("EDGE_PYTHON", "python3")
HOST = "127.0.0.1"
PORT = 5090
GEN_TOKENS = int(os.environ.get("GEN_TOKENS", "16"))

DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-1B"
DATASET = "wildgpt4"
DEFAULT_CUT_LAYERS = [2, 4, 6, 8, 10, 12, 14, 15]
SESSION_COUNTS = [2, 4, 6, 8]
GENERATION_LENGTHS = [16, 64, 128, 256]


@dataclass
class Fleet:
    """The model, cut layers, prompts and edge settings one run uses."""

    model_id: str
    cut_layers: list[int]
    prompts: list[str]
    edge_device: str
    edge_dtype: str


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


def start_server(model_id: str, batch_window_ms: float = 0.0) -> subprocess.Popen:
    """Start the cloud server and wait until it accepts connections."""

    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    log = open(os.path.join(HERE, "mem_server.log"), "w")
    proc = subprocess.Popen(
        [EDGE_PYTHON, os.path.join(HERE, "shard02_multi.py"),
         "--model-id", model_id, "--host", "[::]", "--port", str(PORT),
         "--device", "cuda:0", "--dtype", "bfloat16", "--batch-window-ms", str(batch_window_ms)],
        cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT,
    )
    print(f"server pid={proc.pid}, waiting for port {PORT} ...")
    if not wait_for_port(HOST, PORT):
        proc.terminate()
        raise RuntimeError("server did not come up in time; see mem_server.log")
    time.sleep(2)
    return proc


def stats_stub() -> Any:
    """Open a client stub for the memory-statistics RPC."""

    channel = grpc.insecure_channel(f"{HOST}:{PORT}")
    return llama_multi_pb2_grpc.ShardServiceStub(channel)


def get_stats(stub: Any) -> Any:
    """Read the server's current memory statistics."""

    return stub.GetMemoryStats(llama_multi_pb2.MemoryStatsRequest())


class PeakMonitor:
    """Polls the memory statistics in the background and keeps the peak."""

    def __init__(self, interval_s: float = 0.12) -> None:
        self.stub = stats_stub()
        self.interval_s = interval_s
        self._stop = threading.Event()
        self.peak_allocated = 0
        self.peak_reserved = 0
        self.peak_cache = 0
        self.peak_sessions = 0
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                stats = get_stats(self.stub)
                self.peak_allocated = max(self.peak_allocated, stats.allocated_bytes)
                self.peak_reserved = max(self.peak_reserved, stats.reserved_bytes)
                self.peak_cache = max(self.peak_cache, stats.cache_bytes_exact)
                self.peak_sessions = max(self.peak_sessions, stats.active_sessions)
            except grpc.RpcError:
                pass
            time.sleep(self.interval_s)

    def start(self) -> PeakMonitor:
        """Begin polling and return self, for use in an assignment."""

        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop polling."""

        self._stop.set()
        self._thread.join(timeout=2)


def run_edge(
    session_id: str,
    cut_layer: int,
    prompt: str,
    max_new_tokens: int,
    out_path: str,
    fleet: Fleet,
    edge_threads: int = 2,
) -> subprocess.Popen:
    """Launch one edge process for a single session."""

    args = [
        EDGE_PYTHON, os.path.join(HERE, "shard01_multi.py"),
        "--session-id", session_id,
        "--model-path", fleet.model_id,
        "--server-model-id", fleet.model_id,
        "--cut-layer", str(cut_layer),
        "--host", HOST, "--port", str(PORT),
        "--device", fleet.edge_device, "--dtype", fleet.edge_dtype,
        "--prompt", prompt,
        "--max-new-tokens", str(max_new_tokens),
        "--out", out_path,
    ]
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    if edge_threads > 0 and fleet.edge_device == "cpu":
        env["OMP_NUM_THREADS"] = str(edge_threads)
        env["MKL_NUM_THREADS"] = str(edge_threads)
    return subprocess.Popen(args, cwd=HERE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def run_batch(n: int, max_new_tokens: int, tag: str, fleet: Fleet) -> PeakMonitor:
    """Run ``n`` concurrent sessions and return the peak memory seen."""

    monitor = PeakMonitor().start()
    procs, out_paths = [], []
    for i in range(n):
        session_id = f"edge{i}_{tag}"
        out_path = os.path.join(HERE, f"mem_{session_id}.json")
        out_paths.append(out_path)
        procs.append(run_edge(session_id, fleet.cut_layers[i], fleet.prompts[i], max_new_tokens,
                              out_path, fleet))

    for proc in procs:
        stdout, _ = proc.communicate()
        if proc.returncode != 0:
            print(stdout)
            raise RuntimeError(f"an edge client failed in batch {tag}")

    time.sleep(0.3)  # let the last poll land before stopping the monitor
    monitor.stop()
    for path in out_paths:
        os.remove(path)
    return monitor


def gb(value: float) -> float:
    """Convert bytes to gigabytes."""

    return value / 1e9


def mb(value: float) -> float:
    """Convert bytes to megabytes."""

    return value / 1e6


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for the memory benchmark."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--cut-layers", default=None,
                        help=f"Comma-separated cut layers, one per prompt; defaults to {DEFAULT_CUT_LAYERS}.")
    parser.add_argument("--edge-device", default="cpu",
                        help="Device each simulated edge runs on.")
    parser.add_argument("--edge-dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--batch-window-ms", type=float, default=0.0,
                        help="Decode-batching window for the server; 0 measures the unbatched profile.")
    return parser


def main() -> None:
    """Measure cache memory against session count and against generation length."""

    args = build_parser().parse_args()
    cut_layers = [int(cut) for cut in args.cut_layers.split(",")] if args.cut_layers else list(DEFAULT_CUT_LAYERS)
    prompts = load_prompts(DATASET, root=None, limit=len(cut_layers))
    assert len(cut_layers) == len(prompts), "need one cut layer per prompt"
    fleet = Fleet(model_id=args.model_id, cut_layers=cut_layers, prompts=prompts,
                  edge_device=args.edge_device, edge_dtype=args.edge_dtype)

    server = start_server(fleet.model_id, args.batch_window_ms)
    try:
        stub = stats_stub()
        baseline = get_stats(stub)
        print(f"\nbaseline weights + CUDA context: {gb(baseline.weight_bytes):.3f} GB "
              f"(model: {fleet.model_id}, cut layers: {fleet.cut_layers}, bf16, "
              f"batch_window_ms={args.batch_window_ms}, edge={fleet.edge_device}/{fleet.edge_dtype})\n")

        print(f"=== A. KV-cache memory vs number of concurrent edges ({GEN_TOKENS} new tokens each) ===")
        print(f"{'N':>3}  {'peak_alloc_GB':>13}  {'peak_reserved_GB':>16}  "
              f"{'peak_cache_MB':>13}  {'cache_%_of_weights':>18}")
        for n in SESSION_COUNTS:
            monitor = run_batch(n, max_new_tokens=GEN_TOKENS, tag=f"A_N{n}", fleet=fleet)
            pct = 100 * monitor.peak_cache / baseline.weight_bytes
            print(f"{n:>3}  {gb(monitor.peak_allocated):>13.3f}  {gb(monitor.peak_reserved):>16.3f}  "
                  f"{mb(monitor.peak_cache):>13.2f}  {pct:>17.2f}%")

        print("\n=== B. KV-cache memory vs generation length (4 concurrent edges) ===")
        print(f"{'max_new_tokens':>14}  {'peak_alloc_GB':>13}  {'peak_reserved_GB':>16}  "
              f"{'peak_cache_MB':>13}  {'cache_%_of_weights':>18}")
        for tokens in GENERATION_LENGTHS:
            monitor = run_batch(4, max_new_tokens=tokens, tag=f"B_T{tokens}", fleet=fleet)
            pct = 100 * monitor.peak_cache / baseline.weight_bytes
            print(f"{tokens:>14}  {gb(monitor.peak_allocated):>13.3f}  {gb(monitor.peak_reserved):>16.3f}  "
                  f"{mb(monitor.peak_cache):>13.2f}  {pct:>17.2f}%")

        final = get_stats(stub)
        print(f"\nafter everything closed: active_sessions={final.active_sessions}, "
              f"cache_bytes_exact={final.cache_bytes_exact} (should be back to 0)")

    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    main()
