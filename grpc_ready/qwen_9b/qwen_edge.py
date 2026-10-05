import argparse
from collections import deque
import gc
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("USE_TORCH", "1")

import grpc
import torch
from torch import nn
from qwen_inputs import load_input_file


def patch_transformers_torch_version_for_jetson():
    torch_version = getattr(torch, "__version__", "")
    if "a0" not in torch_version or "nv" not in torch_version:
        return

    normalized_version = torch_version.split("a0", 1)[0]
    torch.__version__ = normalized_version
    if hasattr(torch, "version"):
        torch.version.__version__ = normalized_version

    import importlib.metadata as stdlib_metadata

    real_stdlib_version = stdlib_metadata.version

    def stdlib_version(distribution_name):
        if distribution_name == "torch":
            return normalized_version
        return real_stdlib_version(distribution_name)

    stdlib_metadata.version = stdlib_version

    try:
        import importlib_metadata as backport_metadata
    except ImportError:
        return

    real_backport_version = backport_metadata.version

    def backport_version(distribution_name):
        if distribution_name == "torch":
            return normalized_version
        return real_backport_version(distribution_name)

    backport_metadata.version = backport_version


patch_transformers_torch_version_for_jetson()


def patch_transformers_torch_availability_for_jetson():
    import transformers.utils.import_utils as import_utils

    if getattr(import_utils, "_torch_available", True):
        return

    import_utils._torch_available = True
    import_utils._torch_version = torch.__version__


patch_transformers_torch_availability_for_jetson()

from qwen_metrics import (
    JsonlWriter,
    MetricsContext,
    ResourceSampler,
    cuda_memory_snapshot,
    host_info,
    make_event_id,
    make_run_id,
    now_utc_iso,
    reset_cuda_peak_memory,
    runtime_versions,
    tensor_data_serialized_bytes,
    tensor_raw_nbytes,
    trace_to_metadata,
    write_json,
)


deserialize_tensor = None
pb2 = None
pb2_grpc = None
serialize_tensor = None


def ensure_grpc_modules_loaded():
    global deserialize_tensor, pb2, pb2_grpc, serialize_tensor
    if pb2 is not None:
        return

    from qwen_grpc import (
        deserialize_tensor as loaded_deserialize_tensor,
        pb2 as loaded_pb2,
        pb2_grpc as loaded_pb2_grpc,
        serialize_tensor as loaded_serialize_tensor,
    )

    deserialize_tensor = loaded_deserialize_tensor
    pb2 = loaded_pb2
    pb2_grpc = loaded_pb2_grpc
    serialize_tensor = loaded_serialize_tensor


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EDGE_PREFIX_LAYERS = 16
DEFAULT_FULL_MODEL_LAYERS = 32

MODEL_PROFILES = {
    "qwen": {
        "display_name": "Qwen3.5-9B",
        "model_id": "Qwen/Qwen3.5-9B",
        "tokenized_filename": "qwen_non_thinking.jsonl",
        "run_prefix": "qwen9b",
        "dynamic_model_type": "qwen3_5",
        "full_model_layers": 32,
    },
    "llama": {
        "display_name": "Llama-3.1-8B",
        "model_id": "meta-llama/Llama-3.1-8B",
        "tokenized_filename": "llama_tokenized.jsonl",
        "run_prefix": "llama31_8b",
        "dynamic_model_type": "llama",
        "full_model_layers": 32,
    },
}


def env_first(*names):
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def normalize_model_profile_name(name=None):
    raw_name = name or env_first("SPLIT_MODEL", "QWEN_MODEL_PROFILE", "QWEN_MODEL") or "qwen"
    normalized = raw_name.strip().lower().replace("_", "-")
    aliases = {
        "qwen": "qwen",
        "qwen3.5": "qwen",
        "qwen3.5-9b": "qwen",
        "qwen35": "qwen",
        "qwen35-9b": "qwen",
        "llama": "llama",
        "llama3": "llama",
        "llama3.1": "llama",
        "llama-3.1": "llama",
        "llama3.1-8b": "llama",
        "llama-3.1-8b": "llama",
    }
    if normalized not in aliases:
        valid = ", ".join(sorted(MODEL_PROFILES))
        raise ValueError(f"Unsupported --model {raw_name!r}; valid profiles: {valid}")
    return aliases[normalized]


def resolve_model_id(profile_name):
    if profile_name == "qwen":
        return env_first("QWEN_MODEL_ID", "SPLIT_MODEL_ID") or MODEL_PROFILES[profile_name]["model_id"]
    if profile_name == "llama":
        return env_first("LLAMA_MODEL_ID", "SPLIT_MODEL_ID") or MODEL_PROFILES[profile_name]["model_id"]
    raise AssertionError("unreachable")


def resolve_tokenized_filename_for_profile(profile_name):
    if profile_name == "qwen":
        return (
            env_first("QWEN_TOKENIZED_FILENAME", "SPLIT_TOKENIZED_FILENAME")
            or MODEL_PROFILES[profile_name]["tokenized_filename"]
        )
    if profile_name == "llama":
        return (
            env_first("LLAMA_TOKENIZED_FILENAME", "SPLIT_TOKENIZED_FILENAME")
            or MODEL_PROFILES[profile_name]["tokenized_filename"]
        )
    raise AssertionError("unreachable")


def default_local_model_dir():
    username = os.environ.get("USER") or os.environ.get("LOGNAME") or "user"
    configured_base = os.environ.get("QWEN_LOCAL_MODEL_BASE")
    if configured_base:
        return Path(configured_base).expanduser()
    return Path("/local") / username / "models"


def resolve_dyforward_dir():
    for dirname in ("dyforward", "dyForward"):
        candidate = (SCRIPT_DIR / f"../../{dirname}").resolve()
        if (candidate / "dynamic_forward.py").exists():
            return candidate
    return (SCRIPT_DIR / "../../dyforward").resolve()


def resolve_edge_model_path():
    configured_path = os.environ.get("QWEN_EDGE_MODEL_PATH")
    if configured_path:
        return configured_path

    preferred_names = (
        "qwen35_9b_edge_f16_bf16",
        "qwen35_9b_edge_first16_bf16",
        "qwen35_9b_edge_f12_bf16",
        "qwen35_9b_edge_f8_bf16",
        "qwen35_9b_edge_f6_bf16",
        "qwen35_9b_edge_first6_bf16",
        "qwen35_9b_edge",
    )
    search_roots = (default_local_model_dir(), SCRIPT_DIR.parent)
    for root in search_roots:
        for dirname in preferred_names:
            candidate = (root / dirname).resolve()
            if candidate.exists():
                return str(candidate)

    matches = sorted(
        candidate.resolve()
        for root in search_roots
        for candidate in root.glob("qwen35*_edge*")
        if any(tag in candidate.name for tag in ("f16", "first16", "f12", "f8", "f6", "first6"))
    )
    if matches:
        return str(matches[0])
    return str((default_local_model_dir() / "qwen35_9b_edge_f16_bf16").resolve())


def layer_count_from_config(model_path, default=DEFAULT_EDGE_PREFIX_LAYERS):
    try:
        with (Path(model_path).expanduser() / "config.json").open() as config_file:
            config = json.load(config_file)
        if config.get("num_hidden_layers"):
            return int(config["num_hidden_layers"])
        layer_types = config.get("layer_types")
        if layer_types:
            return len(layer_types)
    except (OSError, ValueError, TypeError):
        pass
    return default


def resolve_selected_testset_root():
    configured_path = os.environ.get("QWEN_SELECTED_TESTSET_ROOT")
    if configured_path:
        return Path(configured_path).expanduser()

    candidates = [
        SCRIPT_DIR / "../../inputs/selected_testsets",
        SCRIPT_DIR / "../../selected_testsets",
        SCRIPT_DIR / "../../testSet/selected_testsets",
    ]
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate
    return (SCRIPT_DIR / "../../inputs/selected_testsets").resolve()


def parse_runtime_args():
    parser = argparse.ArgumentParser(description="Run split inference from the edge device.")
    parser.add_argument(
        "--input-file", default=None,
        help="Prepared JSONL inputs; use all marked records instead of --level/--only-mode selection.",
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_PROFILES),
        default=None,
        help="Model profile to run. Defaults to qwen, or SPLIT_MODEL/QWEN_MODEL_PROFILE if set.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Override the Hugging Face model id sent to the cloud.",
    )
    parser.add_argument(
        "--tokenized-filename",
        default=None,
        help="Override the selected-testset tokenized filename.",
    )
    parser.add_argument(
        "--edge-full-model",
        action="store_true",
        help=(
            "Load the official full CausalLM checkpoint on edge and use its text model "
            "as the prefix runtime. This bypasses the exported edge shard path."
        ),
    )
    parser.add_argument(
        "--edge-model-path",
        default=None,
        help="Path to the exported edge prefix model. Overrides QWEN_EDGE_MODEL_PATH.",
    )
    parser.add_argument(
        "--edge-layers",
        "--active-layers",
        type=int,
        default=None,
        help=(
            "Number of loaded edge model layers to keep active. "
            "Extra loaded layers are dropped before moving the model to CUDA, "
            "and default cut layers become 1..N."
        ),
    )
    parser.add_argument(
        "--split-layers",
        default=None,
        help="Comma-separated cut layers to test. If omitted, defaults to 1..edge-layers.",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=None,
        help=(
            "Formal test-set scale. N uses 8*N wildgpt4 records and N "
            "longreason records. Valid values are 1..100, or 999 for all records. "
            "The final loaded datasets are filtered by --only-mode. "
            "Overrides QWEN_LEVEL/QWEN_TEST_LEVEL."
        ),
    )
    parser.add_argument(
        "--only-mode",
        type=int,
        choices=(0, 1, 2),
        default=None,
        help=(
            "Dataset mode: 0=short/wildgpt4 only, 1=long/longreason only, "
            "2=both. Overrides QWEN_ONLY_MODE. Default is 0."
        ),
    )
    return parser.parse_args()


DYFORWARD_DIR = resolve_dyforward_dir()
if str(DYFORWARD_DIR) not in sys.path:
    sys.path.insert(0, str(DYFORWARD_DIR))

from dynamic_forward import dynamic_forward


MODEL_PROFILE_NAME = normalize_model_profile_name()
MODEL_PROFILE = MODEL_PROFILES[MODEL_PROFILE_NAME]
EDGE_MODEL_PATH = resolve_edge_model_path()
SERVER_MODEL_ID = resolve_model_id(MODEL_PROFILE_NAME)
EDGE_FULL_MODEL = os.environ.get("QWEN_EDGE_FULL_MODEL", "0").lower() in {"1", "true", "yes"}
if MODEL_PROFILE_NAME == "llama":
    EDGE_FULL_MODEL = True
LAYER_TO_SPLIT = int(os.environ.get("QWEN_LAYER_TO_SPLIT", layer_count_from_config(EDGE_MODEL_PATH)))
DTYPE = torch.bfloat16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN_ID = os.environ.get("QWEN_RUN_ID") or make_run_id(MODEL_PROFILE["run_prefix"])
RUN_STARTED_AT = now_utc_iso()
METRICS_ROOT = Path(os.environ.get("QWEN_METRICS_ROOT", SCRIPT_DIR / "results")).expanduser()
RUN_DIR = METRICS_ROOT / RUN_ID
EDGE_METRICS_DIR = RUN_DIR / "edge_metrics"
RESOURCE_SAMPLE_INTERVAL_S = float(os.environ.get("QWEN_RESOURCE_SAMPLE_INTERVAL_S", "0.5"))

TARGET_HOST = os.environ.get("QWEN_CLOUD_HOST", "127.0.0.1")
TARGET_PORT = int(os.environ.get("QWEN_CLOUD_PORT", "5087"))

PROMPT = os.environ.get(
    "QWEN_PROMPT",
    "Explain split inference for large language models in one concise paragraph.",
)
SELECTED_TESTSET_ROOT = resolve_selected_testset_root()
INPUT_FILE = None
FILE_RECORDS = None
USE_SELECTED_TESTSET = os.environ.get("QWEN_USE_SELECTED_TESTSET", "1").lower() in {"1", "true", "yes"}
TOKENIZED_FILENAME = resolve_tokenized_filename_for_profile(MODEL_PROFILE_NAME)
DYNAMIC_MODEL_TYPE = MODEL_PROFILE["dynamic_model_type"]
TEST_LEVEL_ALL = 999
TEST_LEVEL_MAX = 100
WILDGPT4_PER_LEVEL = 8
LONGREASON_PER_LEVEL = 1
TEST_LEVEL = int(os.environ.get("QWEN_LEVEL", os.environ.get("QWEN_TEST_LEVEL", "1")))
TEST_ONLY_MODE = int(os.environ.get("QWEN_ONLY_MODE", "0"))
WILDGPT4_COUNT = None
LONGREASON_COUNT = None
WARMUP_GPT4_COUNT = int(os.environ.get("QWEN_WARMUP_GPT4_COUNT", "8"))
WARMUP_LONGREASON_COUNT = int(os.environ.get("QWEN_WARMUP_LONGREASON_COUNT", "2"))
WARMUP_ROUNDS = int(os.environ.get("QWEN_WARMUP_ROUNDS", "1"))
ENABLE_THINKING = os.environ.get("QWEN_ENABLE_THINKING", "0").lower() in {"1", "true", "yes"}
MAX_NEW_TOKENS = int(os.environ.get("QWEN_MAX_NEW_TOKENS", "64"))
WARMUP_MAX_NEW_TOKENS = int(os.environ.get("QWEN_WARMUP_MAX_NEW_TOKENS", str(min(MAX_NEW_TOKENS, 4))))
DO_SAMPLE = False
TEMPERATURE = 1.0
TOP_P = 1.0
TOP_K = 0


class EdgeMetrics:
    def __init__(self):
        EDGE_METRICS_DIR.mkdir(parents=True, exist_ok=True)
        self.started_at = RUN_STARTED_AT
        self.outputs = JsonlWriter(EDGE_METRICS_DIR / "outputs_edge.jsonl")
        self.latency = JsonlWriter(EDGE_METRICS_DIR / "latency_events_edge.jsonl")
        self.context = MetricsContext(
            {
                "run_id": RUN_ID,
                "role": "edge",
                "state": "starting",
                "phase": "setup",
                "cut_layer": None,
                "dataset": None,
                "record_id": None,
                "sample_index": None,
                "stage": None,
                "generated_token_index": None,
                "input_token_count": None,
                "context_token_length": None,
                "event_id": None,
            }
        )
        self.resource_sampler = ResourceSampler(
            EDGE_METRICS_DIR / "resource_edge.jsonl",
            role="edge",
            context=self.context,
            interval_s=RESOURCE_SAMPLE_INTERVAL_S,
            include_tegrastats=True,
            include_nvidia_smi=True,
        )

    def start(self):
        self.resource_sampler.start()

    def set_context(self, **context):
        base = {
            "run_id": RUN_ID,
            "role": "edge",
            "state": context.pop("state", "running"),
        }
        base.update(context)
        self.context.set(base)

    def write_latency(self, row):
        self.latency.write(row)

    def write_output(self, row):
        self.outputs.write(row)

    def write_manifest(self, manifest):
        write_json(EDGE_METRICS_DIR / "manifest_edge.json", manifest)

    def close(self):
        self.resource_sampler.stop()
        self.outputs.close()
        self.latency.close()


def build_manifest(
    status,
    split_layers=None,
    test_records=None,
    warmup_records=None,
    warmup_layer_split=None,
    error=None,
    finished_at=None,
):
    manifest = {
        "run_id": RUN_ID,
        "role": "edge",
        "status": status,
        "started_at": RUN_STARTED_AT,
        "finished_at": finished_at,
        "metrics_root": str(METRICS_ROOT),
        "run_dir": str(RUN_DIR),
        "metrics_dir": str(EDGE_METRICS_DIR),
        "metrics_files": {
            "outputs": str(EDGE_METRICS_DIR / "outputs_edge.jsonl"),
            "latency_events": str(EDGE_METRICS_DIR / "latency_events_edge.jsonl"),
            "resource": str(EDGE_METRICS_DIR / "resource_edge.jsonl"),
            "manifest": str(EDGE_METRICS_DIR / "manifest_edge.json"),
        },
        "host": host_info(),
        "runtime": runtime_versions(),
        "experiment": {
            "model_profile": MODEL_PROFILE_NAME,
            "model_display_name": MODEL_PROFILE["display_name"],
            "model_id": SERVER_MODEL_ID,
            "dynamic_model_type": DYNAMIC_MODEL_TYPE,
            "edge_model_source": "official_full_model_prefix" if EDGE_FULL_MODEL else "exported_prefix",
            "edge_full_model": EDGE_FULL_MODEL,
            "edge_model_path": EDGE_MODEL_PATH,
            "server_model_id": SERVER_MODEL_ID,
            "cloud_host": TARGET_HOST,
            "cloud_port": TARGET_PORT,
            "exported_edge_prefix_layers": layer_count_from_config(EDGE_MODEL_PATH),
            "active_edge_prefix_layers": LAYER_TO_SPLIT,
            "available_edge_prefix_layers": LAYER_TO_SPLIT,
            "split_layers": split_layers,
            "torch_dtype": str(DTYPE),
            "device": str(DEVICE),
            "max_new_tokens": MAX_NEW_TOKENS,
            "fixed_new_tokens": True,
            "early_stop_enabled": False,
            "do_sample": DO_SAMPLE,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
        },
        "warmup": {
            "wildgpt4_tail_count": WARMUP_GPT4_COUNT,
            "longreason_tail_count": WARMUP_LONGREASON_COUNT,
            "only_mode": TEST_ONLY_MODE,
            "only_mode_label": only_mode_label(TEST_ONLY_MODE),
            "rounds": WARMUP_ROUNDS,
            "max_new_tokens": WARMUP_MAX_NEW_TOKENS,
            "mode": "single_initial_pass",
            "cut_layer": warmup_layer_split,
        },
        "resource_sampling": {
            "interval_s": RESOURCE_SAMPLE_INTERVAL_S,
            "edge_extra_sources": ["torch.cuda", "psutil", "tegrastats", "nvidia-smi"],
        },
        "selected_testset": {
            "enabled": USE_SELECTED_TESTSET,
            "root": str(SELECTED_TESTSET_ROOT),
            "tokenized_filename": TOKENIZED_FILENAME,
            "level": TEST_LEVEL,
            "level_all_value": TEST_LEVEL_ALL,
            "level_max": TEST_LEVEL_MAX,
            "only_mode": TEST_ONLY_MODE,
            "only_mode_label": only_mode_label(TEST_ONLY_MODE),
            "wildgpt4_per_level": WILDGPT4_PER_LEVEL,
            "longreason_per_level": LONGREASON_PER_LEVEL,
            "wildgpt4_count": WILDGPT4_COUNT,
            "longreason_count": LONGREASON_COUNT,
        },
    }
    if INPUT_FILE is not None:
        manifest["selected_testset"] = {
            "enabled": True, "mode": "input_file", "input_file": str(INPUT_FILE),
        }
        manifest["warmup"] = {
            "mode": "explicit_input_phases", "rounds": WARMUP_ROUNDS,
            "max_new_tokens": WARMUP_MAX_NEW_TOKENS, "cut_layer": warmup_layer_split,
        }
    if test_records is not None:
        manifest["selected_testset"]["formal_records"] = [record_summary(record) for record in test_records]
    if warmup_records is not None:
        manifest["selected_testset"]["warmup_records"] = [record_summary(record) for record in warmup_records]
    if error is not None:
        manifest["error"] = str(error)
    return manifest


def sync_device():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def slice_layer_types(config, start, end):
    layer_types = getattr(config, "layer_types", None)
    if layer_types is not None:
        config.layer_types = list(layer_types[start:end])
    if getattr(config, "layer_types", None):
        config.num_hidden_layers = len(config.layer_types)


def apply_runtime_args(args):
    global MODEL_PROFILE_NAME, MODEL_PROFILE, SERVER_MODEL_ID, TOKENIZED_FILENAME, DYNAMIC_MODEL_TYPE
    global EDGE_MODEL_PATH, EDGE_FULL_MODEL, LAYER_TO_SPLIT, TEST_LEVEL, TEST_ONLY_MODE, WILDGPT4_COUNT, LONGREASON_COUNT
    global RUN_ID, RUN_DIR, EDGE_METRICS_DIR
    global INPUT_FILE, FILE_RECORDS

    requested_profile_name = normalize_model_profile_name(args.model) if args.model else MODEL_PROFILE_NAME
    profile_changed = requested_profile_name != MODEL_PROFILE_NAME
    if profile_changed:
        MODEL_PROFILE_NAME = requested_profile_name
        MODEL_PROFILE = MODEL_PROFILES[MODEL_PROFILE_NAME]
        DYNAMIC_MODEL_TYPE = MODEL_PROFILE["dynamic_model_type"]
        if "QWEN_RUN_ID" not in os.environ:
            RUN_ID = make_run_id(MODEL_PROFILE["run_prefix"])
            RUN_DIR = METRICS_ROOT / RUN_ID
            EDGE_METRICS_DIR = RUN_DIR / "edge_metrics"

    if args.model_id:
        SERVER_MODEL_ID = args.model_id
    elif profile_changed:
        SERVER_MODEL_ID = resolve_model_id(MODEL_PROFILE_NAME)

    if args.tokenized_filename:
        TOKENIZED_FILENAME = args.tokenized_filename
    elif profile_changed:
        TOKENIZED_FILENAME = resolve_tokenized_filename_for_profile(MODEL_PROFILE_NAME)

    INPUT_FILE = Path(args.input_file).expanduser().resolve() if args.input_file else None
    FILE_RECORDS = load_input_file(INPUT_FILE, MODEL_PROFILE_NAME) if INPUT_FILE else None

    if args.edge_model_path:
        EDGE_MODEL_PATH = str(Path(args.edge_model_path).expanduser().resolve())
    if args.edge_full_model:
        EDGE_FULL_MODEL = True
    if MODEL_PROFILE_NAME == "llama":
        EDGE_FULL_MODEL = True

    if args.level is not None:
        TEST_LEVEL = args.level
    if args.only_mode is not None:
        TEST_ONLY_MODE = args.only_mode
    validate_only_mode(TEST_ONLY_MODE)
    WILDGPT4_COUNT, LONGREASON_COUNT = counts_from_level(TEST_LEVEL, TEST_ONLY_MODE)

    exported_layer_count = layer_count_from_config(EDGE_MODEL_PATH)
    max_edge_layer_count = int(MODEL_PROFILE.get("full_model_layers", DEFAULT_FULL_MODEL_LAYERS)) if EDGE_FULL_MODEL else exported_layer_count
    if args.edge_layers is not None:
        if args.edge_layers <= 0:
            raise ValueError(f"--edge-layers must be positive, got {args.edge_layers}")
        if args.edge_layers > max_edge_layer_count:
            raise ValueError(
                f"--edge-layers={args.edge_layers} exceeds available edge layer count "
                f"{max_edge_layer_count}. full_model_mode={EDGE_FULL_MODEL}, "
                f"edge_model_path={EDGE_MODEL_PATH}"
            )
        LAYER_TO_SPLIT = args.edge_layers
    elif args.edge_model_path and "QWEN_LAYER_TO_SPLIT" not in os.environ and not EDGE_FULL_MODEL:
        LAYER_TO_SPLIT = exported_layer_count

    if LAYER_TO_SPLIT <= 0 or LAYER_TO_SPLIT > max_edge_layer_count:
        raise ValueError(
            f"Active edge layer count must be in [1, {max_edge_layer_count}], "
            f"got {LAYER_TO_SPLIT}. full_model_mode={EDGE_FULL_MODEL}, "
            f"edge_model_path={EDGE_MODEL_PATH}"
        )


def validate_only_mode(only_mode):
    if only_mode not in {0, 1, 2}:
        raise ValueError(f"--only-mode must be 0, 1, or 2, got {only_mode}")


def only_mode_label(only_mode):
    return {
        0: "short_only",
        1: "long_only",
        2: "short_and_long",
    }[only_mode]


def counts_from_level(level, only_mode):
    if level == TEST_LEVEL_ALL:
        wildgpt4_count = None
        longreason_count = None
    elif level <= 0 or level > TEST_LEVEL_MAX:
        raise ValueError(
            f"--level must be in [1, {TEST_LEVEL_MAX}] or {TEST_LEVEL_ALL} for all records, got {level}"
        )
    else:
        wildgpt4_count = WILDGPT4_PER_LEVEL * level
        longreason_count = LONGREASON_PER_LEVEL * level

    if only_mode == 0:
        return wildgpt4_count, 0
    if only_mode == 1:
        return 0, longreason_count
    if only_mode == 2:
        return wildgpt4_count, longreason_count

    validate_only_mode(only_mode)
    raise AssertionError("unreachable")


def count_label(count):
    if count == 0:
        return "0"
    return "all" if count is None else str(count)


def keep_prefix_layers(model, keep_layers):
    total_layers = len(model.layers)
    if keep_layers <= 0 or keep_layers > total_layers:
        raise ValueError(f"LAYER_TO_SPLIT must be in [1, {total_layers}], got {keep_layers}")

    model.layers = model.layers[:keep_layers]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Keep the exported prefix config intact. Qwen3.5's DynamicCache needs the
    # hybrid layer type list, even when this run uses only the first few layers.
    model.norm = nn.Identity()


def get_text_model(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model
    if hasattr(model, "layers"):
        return model
    raise AttributeError(f"Could not find text decoder layers for {MODEL_PROFILE['display_name']}.")


def load_edge_model_and_tokenizer():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if EDGE_FULL_MODEL:
        print(f"Edge model source: official full model prefix ({SERVER_MODEL_ID})")
        full_model = AutoModelForCausalLM.from_pretrained(
            SERVER_MODEL_ID,
            torch_dtype=DTYPE,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model = get_text_model(full_model)
        keep_prefix_layers(model, LAYER_TO_SPLIT)
        del full_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        tokenizer = AutoTokenizer.from_pretrained(SERVER_MODEL_ID, trust_remote_code=True)
    else:
        if MODEL_PROFILE_NAME != "qwen":
            raise ValueError(
                "Exported prefix shard mode is only configured for Qwen. "
                "Use --model llama without --edge-model-path; Llama mode loads the full model prefix."
            )
        from transformers import Qwen3_5TextModel

        print("Edge model source: exported prefix shard")
        model = Qwen3_5TextModel.from_pretrained(
            EDGE_MODEL_PATH,
            torch_dtype=DTYPE,
            trust_remote_code=True,
        )
        keep_prefix_layers(model, LAYER_TO_SPLIT)
        tokenizer = AutoTokenizer.from_pretrained(EDGE_MODEL_PATH, trust_remote_code=True)

    model.eval()
    model.to(DEVICE)
    sync_device()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def resolve_split_layers(max_layer_split, configured_layers=None, use_env=True):
    if configured_layers is None and use_env:
        configured_layers = os.environ.get("QWEN_SPLIT_LAYERS")
    if configured_layers:
        layers = [int(item.strip()) for item in configured_layers.split(",") if item.strip()]
    else:
        layers = list(range(1, max_layer_split + 1))

    invalid_layers = [layer for layer in layers if layer <= 0 or layer > max_layer_split]
    if invalid_layers:
        raise ValueError(
            f"QWEN_SPLIT_LAYERS contains invalid layers {invalid_layers}; "
            f"valid range is [1, {max_layer_split}]."
        )
    return layers


def run_edge_forward(model, input_ids, attention_mask, position_ids, past_key_values, layer_split):
    input_embeds = model.embed_tokens(input_ids)
    outputs = dynamic_forward(
        model,
        input_embeds,
        start_layer=0,
        end_layer=layer_split,
        model_type=DYNAMIC_MODEL_TYPE,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=True,
        apply_final_norm=False,
    )
    sync_device()
    return outputs


class QwenShardClient:
    def __init__(self, host, port):
        ensure_grpc_modules_loaded()
        options = [
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
            ("grpc.max_send_message_length", 100 * 1024 * 1024),
        ]
        self.channel = grpc.insecure_channel(f"{host}:{port}", options=options)
        self.stub = pb2_grpc.ShardServiceStub(self.channel)

    def init_connection(self, layer_split, trace=None):
        request = pb2.InitRequest(
            config=pb2.RuntimeConfig(
                model_id=SERVER_MODEL_ID,
                layer_split=layer_split,
                torch_dtype=str(DTYPE),
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
            )
        )
        start = time.perf_counter()
        response = self.stub.InitConnection(request, metadata=trace_to_metadata(trace or {}))
        elapsed = time.perf_counter() - start
        if response.status != 200:
            raise RuntimeError(f"Cloud init failed: {response.message}")
        return {
            "init_rpc_s": elapsed,
            "init_status": response.status,
            "init_message": response.message,
        }

    def prefill(self, hidden, attention_mask, position_ids, trace):
        hidden_raw_bytes = tensor_raw_nbytes(hidden)
        attention_mask_raw_bytes = tensor_raw_nbytes(attention_mask)
        position_ids_raw_bytes = tensor_raw_nbytes(position_ids)

        request_start = time.perf_counter()
        hidden_start = time.perf_counter()
        hidden_data = serialize_tensor(hidden)
        hidden_serialize_s = time.perf_counter() - hidden_start

        attention_start = time.perf_counter()
        attention_mask_data = serialize_tensor(attention_mask)
        attention_mask_serialize_s = time.perf_counter() - attention_start

        position_start = time.perf_counter()
        position_ids_data = serialize_tensor(position_ids)
        position_ids_serialize_s = time.perf_counter() - position_start
        request_serialize_s = time.perf_counter() - request_start

        request = pb2.PrefillRequest(
            hidden=hidden_data,
            attention_mask=attention_mask_data,
            position_ids=position_ids_data,
        )
        rpc_start = time.perf_counter()
        response = self.stub.PrefillStep(request, metadata=trace_to_metadata(trace))
        rpc_wall_s = time.perf_counter() - rpc_start

        deserialize_start = time.perf_counter()
        next_token = deserialize_tensor(response.next_token, DEVICE)
        response_deserialize_s = time.perf_counter() - deserialize_start

        timings = {
            "edge_hidden_serialize_s": hidden_serialize_s,
            "edge_attention_mask_serialize_s": attention_mask_serialize_s,
            "edge_position_ids_serialize_s": position_ids_serialize_s,
            "edge_request_serialize_s": request_serialize_s,
            "rpc_wall_s": rpc_wall_s,
            "cloud_infer_s": response.server_infer_s,
            "cloud_handler_total_s": response.server_total_s,
            "edge_response_deserialize_s": response_deserialize_s,
            "hidden_raw_bytes": hidden_raw_bytes,
            "attention_mask_raw_bytes": attention_mask_raw_bytes,
            "position_ids_raw_bytes": position_ids_raw_bytes,
            "hidden_serialized_bytes": tensor_data_serialized_bytes(hidden_data),
            "attention_mask_serialized_bytes": tensor_data_serialized_bytes(attention_mask_data),
            "position_ids_serialized_bytes": tensor_data_serialized_bytes(position_ids_data),
            "request_serialized_bytes": (
                tensor_data_serialized_bytes(hidden_data)
                + tensor_data_serialized_bytes(attention_mask_data)
                + tensor_data_serialized_bytes(position_ids_data)
            ),
            "response_serialized_bytes": tensor_data_serialized_bytes(response.next_token),
        }
        return next_token, timings

    def decode(self, hidden, trace):
        hidden_raw_bytes = tensor_raw_nbytes(hidden)

        request_start = time.perf_counter()
        hidden_data = serialize_tensor(hidden)
        request_serialize_s = time.perf_counter() - request_start

        request = pb2.DecodeRequest(hidden=hidden_data)
        rpc_start = time.perf_counter()
        response = self.stub.DecodeStep(request, metadata=trace_to_metadata(trace))
        rpc_wall_s = time.perf_counter() - rpc_start

        deserialize_start = time.perf_counter()
        next_token = deserialize_tensor(response.next_token, DEVICE)
        response_deserialize_s = time.perf_counter() - deserialize_start

        timings = {
            "edge_hidden_serialize_s": request_serialize_s,
            "edge_attention_mask_serialize_s": 0.0,
            "edge_position_ids_serialize_s": 0.0,
            "edge_request_serialize_s": request_serialize_s,
            "rpc_wall_s": rpc_wall_s,
            "cloud_infer_s": response.server_infer_s,
            "cloud_handler_total_s": response.server_total_s,
            "edge_response_deserialize_s": response_deserialize_s,
            "hidden_raw_bytes": hidden_raw_bytes,
            "attention_mask_raw_bytes": 0,
            "position_ids_raw_bytes": 0,
            "hidden_serialized_bytes": tensor_data_serialized_bytes(hidden_data),
            "attention_mask_serialized_bytes": 0,
            "position_ids_serialized_bytes": 0,
            "request_serialized_bytes": tensor_data_serialized_bytes(hidden_data),
            "response_serialized_bytes": tensor_data_serialized_bytes(response.next_token),
        }
        return next_token, timings


def build_model_inputs(tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if MODEL_PROFILE_NAME == "qwen":
        template_kwargs["enable_thinking"] = ENABLE_THINKING
    text = tokenizer.apply_chat_template(messages, **template_kwargs)
    return tokenizer(text, return_tensors="pt")


def build_model_inputs_from_ids(input_ids):
    input_ids = torch.tensor([input_ids], device=DEVICE, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, device=DEVICE, dtype=torch.long)
    position_ids = attention_mask.long().cumsum(dim=1) - 1
    return input_ids, attention_mask, position_ids


def record_summary(record):
    return {
        "dataset": record.get("dataset"),
        "record_id": record.get("record_id"),
        "source": record.get("source"),
        "source_label": record.get("source_label"),
        "source_split": record.get("source_split"),
        "source_row": record.get("source_row"),
        "source_sample_index": record.get("sample_index"),
        "token_count": len(record.get("input_ids", [])),
        "qwen_length": record.get("qwen_length"),
        "llama_length": record.get("llama_length"),
    }


def build_sample_trace(record, phase, layer_split, sample_index, warmup_round=None):
    input_token_count = len(record["input_ids"])
    return {
        "run_id": RUN_ID,
        "phase": phase,
        "warmup_round": warmup_round,
        "cut_layer": layer_split,
        "dataset": record.get("dataset"),
        "record_id": record.get("record_id"),
        "sample_index": sample_index,
        "source": record.get("source"),
        "source_label": record.get("source_label"),
        "source_split": record.get("source_split"),
        "source_row": record.get("source_row"),
        "source_sample_index": record.get("sample_index"),
        "input_token_count": input_token_count,
    }


def build_event_trace(sample_trace, stage, generated_token_index, context_token_length):
    trace = dict(sample_trace)
    trace.update(
        {
            "stage": stage,
            "generated_token_index": generated_token_index,
            "context_token_length": context_token_length,
        }
    )
    trace["event_id"] = make_event_id(trace)
    return trace


def load_tokenized_records(group_name, count, from_tail=False):
    if count == 0:
        return []

    path = SELECTED_TESTSET_ROOT / group_name / TOKENIZED_FILENAME
    load_all = count is None
    records = deque(maxlen=count) if from_tail and not load_all else []
    with path.open() as jsonl_file:
        for line in jsonl_file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["input_ids"] = [int(token_id) for token_id in record["input_ids"]]
            records.append(record)
            if not from_tail and not load_all and len(records) >= count:
                break

    records = list(records)
    if not load_all and len(records) < count:
        raise ValueError(f"Only found {len(records)} records in {path}, requested {count}.")
    return records


def load_warmup_records():
    if FILE_RECORDS is not None:
        return FILE_RECORDS[1]
    if not USE_SELECTED_TESTSET:
        return []
    if not SELECTED_TESTSET_ROOT.exists():
        return []

    records = []
    if TEST_ONLY_MODE in {0, 2} and WARMUP_GPT4_COUNT > 0:
        records.extend(load_tokenized_records("wildgpt4", WARMUP_GPT4_COUNT, from_tail=True))
    if TEST_ONLY_MODE in {1, 2} and WARMUP_LONGREASON_COUNT > 0:
        records.extend(load_tokenized_records("longreason", WARMUP_LONGREASON_COUNT, from_tail=True))
    return records


def load_selected_testset_records():
    if FILE_RECORDS is not None:
        return FILE_RECORDS[0]
    if not USE_SELECTED_TESTSET:
        return []
    if not SELECTED_TESTSET_ROOT.exists():
        return []

    records = []
    if TEST_ONLY_MODE in {0, 2}:
        records.extend(load_tokenized_records("wildgpt4", WILDGPT4_COUNT))
    if TEST_ONLY_MODE in {1, 2}:
        records.extend(load_tokenized_records("longreason", LONGREASON_COUNT))
    return records


def write_edge_latency_event(
    metrics,
    trace,
    event_start_unix,
    event_start_perf,
    event_end_unix,
    event_end_perf,
    edge_forward_s,
    rpc_timings,
    next_token,
):
    row = dict(trace)
    row.update(
        {
            "event_start_unix": event_start_unix,
            "event_end_unix": event_end_unix,
            "event_total_s": event_end_perf - event_start_perf,
            "edge_forward_s": edge_forward_s,
            "generated_token_id": int(next_token.item()),
            "timeline_serial": True,
            "edge_cloud_compute_overlap_s": 0.0,
        }
    )
    row.update(rpc_timings)
    row["rpc_transport_overhead_s"] = row["rpc_wall_s"] - row["cloud_handler_total_s"]
    row["cloud_non_infer_s"] = row["cloud_handler_total_s"] - row["cloud_infer_s"]
    row["known_compute_s"] = row["edge_forward_s"] + row["cloud_infer_s"]
    row["edge_non_forward_s"] = (
        row["edge_request_serialize_s"] + row["edge_response_deserialize_s"]
    )
    row.update(cuda_memory_snapshot("edge"))
    metrics.write_latency(row)


def run_generation_from_input_ids(
    model,
    tokenizer,
    client,
    input_ids,
    layer_split,
    max_new_tokens=None,
    metrics=None,
    sample_trace=None,
):
    if max_new_tokens is None:
        max_new_tokens = MAX_NEW_TOKENS
    max_new_tokens = max(1, int(max_new_tokens))

    input_ids, attention_mask, position_ids = build_model_inputs_from_ids(input_ids)
    generated_ids = input_ids.clone()
    generated_token_ids = []
    eos_seen = False
    past_key_values = None

    if sample_trace is None:
        sample_trace = {
            "run_id": RUN_ID,
            "phase": "manual",
            "warmup_round": None,
            "cut_layer": layer_split,
            "dataset": "manual",
            "record_id": "manual",
            "sample_index": 1,
            "input_token_count": input_ids.shape[1],
        }

    sample_start_unix = time.time()
    sample_start_perf = time.perf_counter()
    init_trace = build_event_trace(sample_trace, "init", 0, input_ids.shape[1])
    if metrics is not None:
        metrics.set_context(state=sample_trace.get("phase", "running"), **init_trace)
    init_timing = client.init_connection(layer_split, init_trace)
    generation_start_perf = time.perf_counter()

    with torch.inference_mode():
        event_trace = build_event_trace(sample_trace, "prefill", 1, input_ids.shape[1])
        if metrics is not None:
            metrics.set_context(state=sample_trace.get("phase", "running"), **event_trace)
        reset_cuda_peak_memory(DEVICE)
        sync_device()
        event_start_unix = time.time()
        event_start_perf = time.perf_counter()
        forward_start = time.perf_counter()
        outputs = run_edge_forward(
            model,
            input_ids,
            attention_mask,
            position_ids,
            None,
            layer_split,
        )
        edge_forward_s = time.perf_counter() - forward_start
        past_key_values = outputs.past_key_values
        hidden = outputs.last_hidden_state

        next_token, rpc_timings = client.prefill(hidden, attention_mask, position_ids, event_trace)
        event_end_unix = time.time()
        event_end_perf = time.perf_counter()
        if metrics is not None:
            write_edge_latency_event(
                metrics,
                event_trace,
                event_start_unix,
                event_start_perf,
                event_end_unix,
                event_end_perf,
                edge_forward_s,
                rpc_timings,
                next_token,
            )
        generated_token_ids.append(int(next_token.item()))
        if tokenizer.eos_token_id is not None and int(next_token.item()) == tokenizer.eos_token_id:
            eos_seen = True
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)

        for generated_token_index in range(2, max_new_tokens + 1):
            context_token_length = generated_ids.shape[1]
            position_ids = torch.tensor(
                [[context_token_length - 1]],
                device=DEVICE,
                dtype=torch.long,
            )
            event_trace = build_event_trace(
                sample_trace,
                "decode",
                generated_token_index,
                context_token_length,
            )
            if metrics is not None:
                metrics.set_context(state=sample_trace.get("phase", "running"), **event_trace)
            reset_cuda_peak_memory(DEVICE)
            sync_device()
            event_start_unix = time.time()
            event_start_perf = time.perf_counter()
            forward_start = time.perf_counter()
            outputs = run_edge_forward(
                model,
                next_token,
                attention_mask,
                position_ids,
                past_key_values,
                layer_split,
            )
            edge_forward_s = time.perf_counter() - forward_start
            past_key_values = outputs.past_key_values
            hidden = outputs.last_hidden_state

            next_token, rpc_timings = client.decode(hidden, event_trace)
            event_end_unix = time.time()
            event_end_perf = time.perf_counter()
            if metrics is not None:
                write_edge_latency_event(
                    metrics,
                    event_trace,
                    event_start_unix,
                    event_start_perf,
                    event_end_unix,
                    event_end_perf,
                    edge_forward_s,
                    rpc_timings,
                    next_token,
                )

            generated_token_ids.append(int(next_token.item()))
            if tokenizer.eos_token_id is not None and int(next_token.item()) == tokenizer.eos_token_id:
                eos_seen = True
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)

    sample_end_unix = time.time()
    sample_end_perf = time.perf_counter()
    generated_text = tokenizer.decode(generated_ids[0][input_ids.shape[1] :], skip_special_tokens=True)
    if metrics is not None:
        metrics.set_context(
            state=sample_trace.get("phase", "running"),
            **build_event_trace(sample_trace, "between_samples", max_new_tokens, generated_ids.shape[1]),
        )
    return {
        "output_text": generated_text,
        "output_ids": generated_token_ids,
        "generated_token_count": len(generated_token_ids),
        "eos_seen": eos_seen,
        "sample_start_unix": sample_start_unix,
        "sample_end_unix": sample_end_unix,
        "sample_total_s": sample_end_perf - sample_start_perf,
        "generation_total_s": sample_end_perf - generation_start_perf,
        **init_timing,
    }


def run_generation(model, tokenizer, client, prompt, layer_split, max_new_tokens=None, metrics=None):
    inputs = build_model_inputs(tokenizer, prompt)
    return run_generation_from_input_ids(
        model,
        tokenizer,
        client,
        inputs.input_ids[0].tolist(),
        layer_split,
        max_new_tokens=max_new_tokens,
        metrics=metrics,
    )


def run_warmup_rounds(model, tokenizer, client, warmup_records, warmup_layer_split, metrics):
    if not warmup_records or WARMUP_ROUNDS <= 0:
        return

    print(
        f"\n======== Warmup: cut layer {warmup_layer_split}, rounds={WARMUP_ROUNDS}, "
        f"max_new_tokens={WARMUP_MAX_NEW_TOKENS} ========"
    )
    for round_index in range(1, WARMUP_ROUNDS + 1):
        print(f"\n======== Warmup round {round_index}/{WARMUP_ROUNDS} ========")
        for record_index, record in enumerate(warmup_records, start=1):
            label = f"{record.get('dataset')}#{record.get('record_id')}"
            token_count = len(record["input_ids"])
            sample_trace = build_sample_trace(
                record,
                "warmup",
                warmup_layer_split,
                record_index,
                warmup_round=round_index,
            )
            result = run_generation_from_input_ids(
                model,
                tokenizer,
                client,
                record["input_ids"],
                warmup_layer_split,
                max_new_tokens=WARMUP_MAX_NEW_TOKENS,
                metrics=metrics,
                sample_trace=sample_trace,
            )
            print(
                f"Warmup {record_index}/{len(warmup_records)} {label} "
                f"tokens={token_count}: {result['sample_total_s']:.4f} s"
            )


def main():
    runtime_args = parse_runtime_args()
    apply_runtime_args(runtime_args)

    metrics = EdgeMetrics()
    metrics.start()
    split_layers = None
    test_records = None
    warmup_records = None
    warmup_layer_split = None
    status = "running"
    error = None

    try:
        print(f"Run ID: {RUN_ID}")
        print(f"Edge metrics dir: {EDGE_METRICS_DIR}")
        print(f"Loading {MODEL_PROFILE['display_name']} edge model ...")
        print(f"Model profile: {MODEL_PROFILE_NAME}")
        print(f"Edge full model mode: {EDGE_FULL_MODEL}")
        print(f"Edge model path: {EDGE_MODEL_PATH}")
        print(f"Server/full model id: {SERVER_MODEL_ID}")
        print(f"Dynamic forward backend: {DYNAMIC_MODEL_TYPE}")
        print(f"Input file: {INPUT_FILE}" if INPUT_FILE else f"Tokenized file: {TOKENIZED_FILENAME}")
        print(f"Using device: {DEVICE}, dtype: {DTYPE}, available edge prefix layers: {LAYER_TO_SPLIT}")

        metrics.set_context(
            state="load_model",
            phase="setup",
            cut_layer=None,
            dataset=None,
            record_id=None,
            sample_index=None,
            stage="load_model",
            generated_token_index=None,
            input_token_count=None,
            context_token_length=None,
            event_id=None,
        )
        model, tokenizer = load_edge_model_and_tokenizer()

        client = QwenShardClient(TARGET_HOST, TARGET_PORT)
        use_env_split_layers = runtime_args.edge_layers is None and runtime_args.split_layers is None
        split_layers = resolve_split_layers(
            LAYER_TO_SPLIT,
            configured_layers=runtime_args.split_layers,
            use_env=use_env_split_layers,
        )
        print(f"Split layers to test: {split_layers}")
        print(f"Formal generation tokens: {MAX_NEW_TOKENS} (early stop disabled)")

        test_records = load_selected_testset_records()
        if test_records:
            warmup_records = load_warmup_records()
            warmup_layer_split = max(split_layers)
            metrics.write_manifest(
                build_manifest(
                    "running",
                    split_layers,
                    test_records,
                    warmup_records,
                    warmup_layer_split=warmup_layer_split,
                )
            )
            if INPUT_FILE:
                print(f"Input file: {INPUT_FILE} (explicit formal/warmup records)")
            else:
                print(f"Selected testset root: {SELECTED_TESTSET_ROOT}")
                print(
                    f"Test level: {TEST_LEVEL}, only-mode={TEST_ONLY_MODE} "
                    f"({only_mode_label(TEST_ONLY_MODE)}), "
                    f"(wildgpt4={count_label(WILDGPT4_COUNT)}, longreason={count_label(LONGREASON_COUNT)})"
                )
            print(f"Loaded formal records: {len(test_records)}")
            print(f"Loaded warmup records: {len(warmup_records)}, cut layer {warmup_layer_split}")

            run_warmup_rounds(model, tokenizer, client, warmup_records, warmup_layer_split, metrics)

            for layer_split in split_layers:
                print(f"\n======== Formal round for split layer {layer_split} ========")
                for index, record in enumerate(test_records, start=1):
                    label = f"{record.get('dataset')}#{record.get('record_id')}"
                    token_count = len(record["input_ids"])
                    sample_trace = build_sample_trace(record, "formal", layer_split, index)
                    print(
                        f"\n======== {MODEL_PROFILE['display_name']} split inference "
                        f"sample {index}/{len(test_records)}: {label} ========"
                    )
                    print(f"Split layer: {layer_split}")
                    print(f"Input tokens: {token_count}")
                    result = run_generation_from_input_ids(
                        model,
                        tokenizer,
                        client,
                        record["input_ids"],
                        layer_split,
                        metrics=metrics,
                        sample_trace=sample_trace,
                    )
                    output_row = dict(sample_trace)
                    output_row.update(record_summary(record))
                    output_row.update(result)
                    output_row["max_new_tokens"] = MAX_NEW_TOKENS
                    output_row["fixed_new_tokens"] = True
                    output_row["early_stop_enabled"] = False
                    metrics.write_output(output_row)
                    print(f"Output: {result['output_text']}")
                    print(f"Generated exactly {MAX_NEW_TOKENS} tokens in {result['sample_total_s']:.4f} s")
        else:
            metrics.write_manifest(build_manifest("running", split_layers, [], []))
            inputs = build_model_inputs(tokenizer, PROMPT)
            prompt_record = {
                "dataset": "manual",
                "record_id": "prompt",
                "source": "QWEN_PROMPT",
                "source_label": "QWEN_PROMPT",
                "source_split": None,
                "source_row": None,
                "sample_index": 1,
                "input_ids": inputs.input_ids[0].tolist(),
            }
            for layer_split in split_layers:
                sample_trace = build_sample_trace(prompt_record, "formal", layer_split, 1)
                print(f"Thinking enabled: {ENABLE_THINKING}")
                print(f"Prompt: {PROMPT}")
                result = run_generation_from_input_ids(
                    model,
                    tokenizer,
                    client,
                    prompt_record["input_ids"],
                    layer_split,
                    metrics=metrics,
                    sample_trace=sample_trace,
                )
                output_row = dict(sample_trace)
                output_row.update(record_summary(prompt_record))
                output_row.update(result)
                output_row["prompt"] = PROMPT
                output_row["max_new_tokens"] = MAX_NEW_TOKENS
                output_row["fixed_new_tokens"] = True
                output_row["early_stop_enabled"] = False
                metrics.write_output(output_row)

                print(
                    f"\n======== {MODEL_PROFILE['display_name']} split inference "
                    f"finished: split layer {layer_split} ========"
                )
                print(f"Output: {result['output_text']}")
                print(f"Generated exactly {MAX_NEW_TOKENS} tokens in {result['sample_total_s']:.4f} s")

        status = "complete"
    except Exception as exc:
        status = "failed"
        error = exc
        raise
    finally:
        metrics.set_context(
            state=status,
            phase="finished",
            cut_layer=None,
            dataset=None,
            record_id=None,
            sample_index=None,
            stage=status,
            generated_token_index=None,
            input_token_count=None,
            context_token_length=None,
            event_id=None,
        )
        metrics.write_manifest(
            build_manifest(
                status,
                split_layers,
                test_records,
                warmup_records,
                warmup_layer_split=warmup_layer_split,
                error=error,
                finished_at=now_utc_iso(),
            )
        )
        metrics.close()


if __name__ == "__main__":
    main()
