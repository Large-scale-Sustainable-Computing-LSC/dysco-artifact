import csv
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import threading
import time
import uuid

try:
    import psutil
except ImportError:  # pragma: no cover - optional runtime dependency
    psutil = None

try:
    import torch
except ImportError:  # pragma: no cover - qwen scripts import torch before using metrics
    torch = None


TRACE_METADATA_KEY = "qwen-trace"

DTYPE_BYTES = {
    "torch.bool": 1,
    "bool": 1,
    "torch.uint8": 1,
    "uint8": 1,
    "torch.int8": 1,
    "int8": 1,
    "torch.float16": 2,
    "float16": 2,
    "torch.bfloat16": 2,
    "bfloat16": 2,
    "torch.int16": 2,
    "int16": 2,
    "torch.float32": 4,
    "float32": 4,
    "torch.int32": 4,
    "int32": 4,
    "torch.float64": 8,
    "float64": 8,
    "torch.int64": 8,
    "int64": 8,
    "torch.long": 8,
    "long": 8,
}


def now_utc_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def make_run_id(prefix="qwen9b"):
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    suffix = uuid.uuid4().hex[:8]
    return f"{prefix}_{timestamp}_{suffix}"


def make_event_id(trace):
    parts = [
        trace.get("run_id", "run"),
        trace.get("phase", "phase"),
        f"round{trace.get('warmup_round', 'na')}",
        f"cut{trace.get('cut_layer', 'na')}",
        str(trace.get("dataset", "dataset")),
        str(trace.get("record_id", "record")),
        f"sample{trace.get('sample_index', 'na')}",
        str(trace.get("stage", "stage")),
        f"tok{trace.get('generated_token_index', 'na')}",
    ]
    return "|".join(parts)


def json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as output_file:
        json.dump(data, output_file, indent=2, sort_keys=True, default=json_default)
        output_file.write("\n")


class JsonlWriter:
    def __init__(self, path, flush_each_write=False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_each_write = flush_each_write
        self._lock = threading.Lock()
        self._file = self.path.open("a")

    def write(self, row):
        with self._lock:
            self._file.write(json.dumps(row, sort_keys=True, default=json_default))
            self._file.write("\n")
            if self.flush_each_write:
                self._file.flush()

    def close(self):
        with self._lock:
            self._file.close()


class MetricsContext:
    def __init__(self, initial=None):
        self._lock = threading.Lock()
        self._context = dict(initial or {})

    def set(self, context):
        with self._lock:
            self._context = dict(context)

    def update(self, **items):
        with self._lock:
            self._context.update(items)

    def snapshot(self):
        with self._lock:
            return dict(self._context)


def trace_to_metadata(trace):
    payload = json.dumps(trace, ensure_ascii=True, separators=(",", ":"), default=json_default)
    return ((TRACE_METADATA_KEY, payload),)


def trace_from_metadata(metadata):
    for key, value in metadata:
        if key == TRACE_METADATA_KEY:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return {}
    return {}


def tensor_raw_nbytes(tensor):
    return int(tensor.numel() * tensor.element_size())


def tensor_data_serialized_bytes(tensor_data):
    return len(tensor_data.tensor_bytes)


def tensor_data_raw_nbytes(tensor_data):
    item_size = DTYPE_BYTES.get(tensor_data.dtype, 0)
    if not item_size:
        return 0
    return int(math.prod(tensor_data.shape) * item_size)


def mb(num_bytes):
    return float(num_bytes) / (1024.0 * 1024.0)


def reset_cuda_peak_memory(device=None):
    if torch is None or not torch.cuda.is_available():
        return
    try:
        torch.cuda.reset_peak_memory_stats(device)
    except Exception:
        pass


def cuda_memory_snapshot(prefix):
    row = {}
    if torch is None or not torch.cuda.is_available():
        row[f"{prefix}_cuda_available"] = False
        return row

    try:
        device_index = torch.cuda.current_device()
        device = torch.device("cuda", device_index)
        row[f"{prefix}_cuda_available"] = True
        row[f"{prefix}_cuda_device_index"] = int(device_index)
        row[f"{prefix}_cuda_device_name"] = torch.cuda.get_device_name(device_index)
        row[f"{prefix}_torch_mem_allocated_mb"] = mb(torch.cuda.memory_allocated(device))
        row[f"{prefix}_torch_mem_reserved_mb"] = mb(torch.cuda.memory_reserved(device))
        row[f"{prefix}_torch_mem_max_allocated_mb"] = mb(torch.cuda.max_memory_allocated(device))
        row[f"{prefix}_torch_mem_max_reserved_mb"] = mb(torch.cuda.max_memory_reserved(device))
        if hasattr(torch.cuda, "mem_get_info"):
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            row[f"{prefix}_cuda_mem_free_mb"] = mb(free_bytes)
            row[f"{prefix}_cuda_mem_total_mb"] = mb(total_bytes)
            row[f"{prefix}_cuda_mem_used_mb"] = mb(total_bytes - free_bytes)
    except Exception as exc:
        row[f"{prefix}_cuda_metric_error"] = str(exc)
    return row


def runtime_versions():
    packages = ("torch", "transformers", "grpcio", "protobuf", "psutil")
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None

    if torch is not None:
        versions["torch_runtime_version"] = getattr(torch, "__version__", None)
        versions["torch_cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
        versions["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            try:
                versions["cuda_device_name"] = torch.cuda.get_device_name(torch.cuda.current_device())
                versions["cuda_device_capability"] = list(torch.cuda.get_device_capability())
            except Exception:
                pass
    return versions


def host_info():
    return {
        "hostname": None,  # Do not embed machine identifiers in shared metrics.
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pid": os.getpid(),
    }


def collect_process_resources(process):
    row = {}
    if psutil is None:
        row["psutil_available"] = False
        return row

    row["psutil_available"] = True
    try:
        row["system_cpu_percent"] = psutil.cpu_percent(interval=None)
        row["system_cpu_count_logical"] = psutil.cpu_count(logical=True)
        row["system_cpu_count_physical"] = psutil.cpu_count(logical=False)
        freq = psutil.cpu_freq()
        if freq is not None:
            row["system_cpu_freq_current_mhz"] = freq.current
            row["system_cpu_freq_min_mhz"] = freq.min
            row["system_cpu_freq_max_mhz"] = freq.max
        load1, load5, load15 = os.getloadavg()
        row["system_load_1m"] = load1
        row["system_load_5m"] = load5
        row["system_load_15m"] = load15
    except Exception as exc:
        row["system_cpu_metric_error"] = str(exc)

    try:
        vm = psutil.virtual_memory()
        row["ram_total_mb"] = mb(vm.total)
        row["ram_available_mb"] = mb(vm.available)
        row["ram_used_mb"] = mb(vm.used)
        row["ram_percent"] = vm.percent
        swap = psutil.swap_memory()
        row["swap_total_mb"] = mb(swap.total)
        row["swap_used_mb"] = mb(swap.used)
        row["swap_percent"] = swap.percent
    except Exception as exc:
        row["memory_metric_error"] = str(exc)

    try:
        mem = process.memory_info()
        row["process_rss_mb"] = mb(mem.rss)
        row["process_vms_mb"] = mb(mem.vms)
        row["process_cpu_percent"] = process.cpu_percent(interval=None)
        row["process_num_threads"] = process.num_threads()
    except Exception as exc:
        row["process_metric_error"] = str(exc)

    return row


def parse_tegrastats_line(line):
    row = {"tegrastats_raw": line.strip()}

    match = re.search(r"RAM\s+(\d+)/(\d+)MB", line)
    if match:
        row["jetson_ram_used_mb"] = int(match.group(1))
        row["jetson_ram_total_mb"] = int(match.group(2))

    match = re.search(r"SWAP\s+(\d+)/(\d+)MB", line)
    if match:
        row["jetson_swap_used_mb"] = int(match.group(1))
        row["jetson_swap_total_mb"] = int(match.group(2))

    match = re.search(r"GR3D_FREQ\s+(\d+)%@?(\d+)?", line)
    if match:
        row["jetson_gr3d_util_percent"] = int(match.group(1))
        if match.group(2):
            row["jetson_gr3d_freq_mhz"] = int(match.group(2))

    match = re.search(r"EMC_FREQ\s+(\d+)%@?(\d+)?", line)
    if match:
        row["jetson_emc_util_percent"] = int(match.group(1))
        if match.group(2):
            row["jetson_emc_freq_mhz"] = int(match.group(2))

    match = re.search(r"CPU\s+\[([^\]]+)\]", line)
    if match:
        utils = []
        freqs = []
        online_cores = 0
        for entry in match.group(1).split(","):
            entry = entry.strip()
            if entry == "off":
                continue
            core_match = re.match(r"(\d+)%@(\d+)", entry)
            if not core_match:
                continue
            online_cores += 1
            utils.append(int(core_match.group(1)))
            freqs.append(int(core_match.group(2)))
        row["jetson_cpu_online_cores"] = online_cores
        if utils:
            row["jetson_cpu_mean_util_percent"] = sum(utils) / len(utils)
            row["jetson_cpu_max_util_percent"] = max(utils)
        if freqs:
            row["jetson_cpu_mean_freq_mhz"] = sum(freqs) / len(freqs)
            row["jetson_cpu_max_freq_mhz"] = max(freqs)

    for name, value in re.findall(r"([A-Za-z0-9_]+)@([0-9.]+)C", line):
        row[f"jetson_temp_{name}_c"] = float(value)

    for name, instant, average in re.findall(r"([A-Za-z0-9_]+)\s+(\d+)mW/(\d+)mW", line):
        row[f"jetson_power_{name}_mw"] = int(instant)
        row[f"jetson_power_{name}_avg_mw"] = int(average)

    return row


class TegrastatsMonitor:
    def __init__(self, interval_s):
        self.interval_ms = max(100, int(interval_s * 1000))
        self.path = shutil.which("tegrastats")
        self._lock = threading.Lock()
        self._latest = {}
        self._process = None
        self._thread = None

    def start(self):
        if not self.path:
            return
        try:
            self._process = subprocess.Popen(
                [self.path, "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            with self._lock:
                self._latest = {"tegrastats_error": str(exc)}
            return
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        if self._process is None or self._process.stdout is None:
            return
        for line in self._process.stdout:
            parsed = parse_tegrastats_line(line)
            parsed["tegrastats_timestamp_unix"] = time.time()
            with self._lock:
                self._latest = parsed

    def snapshot(self):
        with self._lock:
            if not self.path:
                return {"tegrastats_available": False}
            data = dict(self._latest)
        data["tegrastats_available"] = True
        return data

    def stop(self):
        if self._process is None:
            return
        try:
            self._process.terminate()
            self._process.wait(timeout=2)
        except Exception:
            try:
                self._process.kill()
            except Exception:
                pass


NVIDIA_SMI_FIELDS = (
    "index",
    "name",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.free",
    "memory.total",
    "power.draw",
    "power.limit",
    "temperature.gpu",
    "clocks.sm",
    "clocks.mem",
)


NVIDIA_SMI_KEYS = (
    "nvidia_smi_gpu_index",
    "nvidia_smi_gpu_name",
    "nvidia_smi_gpu_util_percent",
    "nvidia_smi_memory_util_percent",
    "nvidia_smi_memory_used_mb",
    "nvidia_smi_memory_free_mb",
    "nvidia_smi_memory_total_mb",
    "nvidia_smi_power_draw_w",
    "nvidia_smi_power_limit_w",
    "nvidia_smi_temperature_gpu_c",
    "nvidia_smi_clock_sm_mhz",
    "nvidia_smi_clock_mem_mhz",
)


def maybe_number(value):
    value = value.strip()
    if value in {"", "[Not Supported]", "N/A"}:
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def collect_nvidia_smi():
    path = shutil.which("nvidia-smi")
    if not path:
        return {"nvidia_smi_available": False}

    command = [
        path,
        f"--query-gpu={','.join(NVIDIA_SMI_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=2,
        )
    except Exception as exc:
        return {"nvidia_smi_available": True, "nvidia_smi_error": str(exc)}

    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        return {"nvidia_smi_available": True, "nvidia_smi_error": "empty output"}

    values = next(csv.reader([lines[0]], skipinitialspace=True))
    row = {"nvidia_smi_available": True}
    for key, value in zip(NVIDIA_SMI_KEYS, values):
        row[key] = maybe_number(value)
    return row


class ResourceSampler:
    def __init__(
        self,
        output_path,
        role,
        context,
        interval_s=0.5,
        include_tegrastats=False,
        include_nvidia_smi=False,
    ):
        self.output_path = Path(output_path)
        self.role = role
        self.context = context
        self.interval_s = float(interval_s)
        self.include_tegrastats = include_tegrastats
        self.include_nvidia_smi = include_nvidia_smi
        self.writer = JsonlWriter(self.output_path)
        self._stop = threading.Event()
        self._thread = None
        self._process = psutil.Process(os.getpid()) if psutil is not None else None
        if self._process is not None:
            self._process.cpu_percent(interval=None)
        self._tegrastats = TegrastatsMonitor(self.interval_s) if include_tegrastats else None

    def start(self):
        if self._tegrastats is not None:
            self._tegrastats.start()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            self.sample_once()
            self._stop.wait(self.interval_s)

    def sample_once(self):
        row = {
            "timestamp_unix": time.time(),
            "timestamp_perf": time.perf_counter(),
            "role": self.role,
        }
        row.update(self.context.snapshot())
        if self._process is not None:
            row.update(collect_process_resources(self._process))
        else:
            row["psutil_available"] = False
        row.update(cuda_memory_snapshot(self.role))
        if self.include_tegrastats and self._tegrastats is not None:
            row.update(self._tegrastats.snapshot())
        if self.include_nvidia_smi:
            row.update(collect_nvidia_smi())
        self.writer.write(row)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_s * 2))
        if self._tegrastats is not None:
            self._tegrastats.stop()
        try:
            self.sample_once()
        except Exception:
            pass
        self.writer.close()
