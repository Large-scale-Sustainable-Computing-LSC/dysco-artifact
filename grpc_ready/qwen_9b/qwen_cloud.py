from concurrent import futures
import argparse
import json
import os
from pathlib import Path
import sys
import time

import grpc
import torch
from transformers import AutoModelForCausalLM

from qwen_grpc import (
    deserialize_tensor,
    parse_torch_dtype,
    pb2,
    pb2_grpc,
    sample_next_token,
    serialize_tensor,
)
from qwen_metrics import (
    JsonlWriter,
    MetricsContext,
    ResourceSampler,
    cuda_memory_snapshot,
    host_info,
    make_run_id,
    now_utc_iso,
    reset_cuda_peak_memory,
    runtime_versions,
    tensor_data_raw_nbytes,
    tensor_data_serialized_bytes,
    tensor_raw_nbytes,
    trace_from_metadata,
    write_json,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EDGE_PREFIX_LAYERS = 16
DEFAULT_FULL_MODEL_LAYERS = 32

MODEL_PROFILES = {
    "qwen": {
        "display_name": "Qwen3.5-9B",
        "model_id": "Qwen/Qwen3.5-9B",
        "run_prefix": "qwen9b",
        "dynamic_model_type": "qwen3_5",
        "full_model_layers": 32,
    },
    "llama": {
        "display_name": "Llama-3.1-8B",
        "model_id": "meta-llama/Llama-3.1-8B",
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


DYFORWARD_DIR = resolve_dyforward_dir()
if str(DYFORWARD_DIR) not in sys.path:
    sys.path.insert(0, str(DYFORWARD_DIR))

from dynamic_forward import dynamic_forward


DEFAULT_DTYPE = torch.bfloat16
SERVER_HOST = "[::]"
SERVER_PORT = int(os.environ.get("QWEN_CLOUD_PORT", "5087"))
RUN_ID = os.environ.get("QWEN_RUN_ID")
RUN_STARTED_AT = now_utc_iso()
METRICS_ROOT = Path(os.environ.get("QWEN_METRICS_ROOT", SCRIPT_DIR / "results")).expanduser()
RESOURCE_SAMPLE_INTERVAL_S = float(os.environ.get("QWEN_RESOURCE_SAMPLE_INTERVAL_S", "0.5"))

MODEL_PROFILE_NAME = normalize_model_profile_name()
MODEL_PROFILE = MODEL_PROFILES[MODEL_PROFILE_NAME]
MODEL_ID = resolve_model_id(MODEL_PROFILE_NAME)
DYNAMIC_MODEL_TYPE = MODEL_PROFILE["dynamic_model_type"]
EDGE_MODEL_PATH = resolve_edge_model_path()
LAYER_TO_SPLIT = int(os.environ.get("QWEN_LAYER_TO_SPLIT", layer_count_from_config(EDGE_MODEL_PATH)))
MAX_NEW_TOKENS = int(os.environ.get("QWEN_MAX_NEW_TOKENS", "64"))
DO_SAMPLE = False
TEMPERATURE = 1.0
TOP_P = 1.0
TOP_K = 0
FORCE_CONTIGUOUS_HIDDEN = os.environ.get("QWEN_FORCE_CONTIGUOUS_HIDDEN", "0") == "1"


def parse_args():
    parser = argparse.ArgumentParser(description="Run the split-inference cloud gRPC server.")
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_PROFILES),
        default=None,
        help="Model profile to run. Defaults to qwen, or SPLIT_MODEL/QWEN_MODEL_PROFILE if set.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Override the Hugging Face model id to preload on the cloud.",
    )
    parser.add_argument(
        "--contiguous-hidden",
        action="store_true",
        help="Force hidden tensors received from edge to be contiguous before cloud suffix forward.",
    )
    return parser.parse_args()


def maybe_make_hidden_contiguous(hidden):
    if FORCE_CONTIGUOUS_HIDDEN:
        return hidden.contiguous()
    return hidden


class CloudMetrics:
    def __init__(self, initial_run_id=None):
        self.started_at = RUN_STARTED_AT
        self.run_id = None
        self.run_dir = None
        self.cloud_dir = None
        self.latency = None
        self.resource_sampler = None
        self.context = MetricsContext(
            {
                "run_id": initial_run_id,
                "role": "cloud",
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
        if initial_run_id:
            self.ensure_run({"run_id": initial_run_id})

    def ensure_run(self, trace):
        run_id = trace.get("run_id") or self.run_id or make_run_id(
            f"{MODEL_PROFILE['run_prefix']}_cloud"
        )
        if self.run_id == run_id and self.latency is not None:
            return
        if self.resource_sampler is not None:
            self.close()

        self.run_id = run_id
        self.run_dir = METRICS_ROOT / run_id
        self.cloud_dir = self.run_dir / "cloud_metrics"
        self.cloud_dir.mkdir(parents=True, exist_ok=True)
        self.latency = JsonlWriter(self.cloud_dir / "latency_events_cloud.jsonl")
        self.context.update(run_id=run_id)
        self.resource_sampler = ResourceSampler(
            self.cloud_dir / "resource_cloud.jsonl",
            role="cloud",
            context=self.context,
            interval_s=RESOURCE_SAMPLE_INTERVAL_S,
            include_tegrastats=False,
            include_nvidia_smi=True,
        )
        self.resource_sampler.start()

    def set_context(self, **context):
        base = {
            "run_id": self.run_id or context.get("run_id"),
            "role": "cloud",
            "state": context.pop("state", "running"),
        }
        base.update(context)
        self.context.set(base)

    def write_latency(self, row):
        if self.latency is not None:
            self.latency.write(row)

    def write_manifest(self, manifest):
        if self.cloud_dir is not None:
            write_json(self.cloud_dir / "manifest_cloud.json", manifest)

    def close(self):
        if self.resource_sampler is not None:
            self.resource_sampler.stop()
            self.resource_sampler = None
        if self.latency is not None:
            self.latency.close()
            self.latency = None


def build_manifest(status, run_id, total_layers=None, model_load_s=None, error=None, finished_at=None):
    run_dir = METRICS_ROOT / run_id if run_id else None
    cloud_dir = run_dir / "cloud_metrics" if run_dir else None
    return {
        "run_id": run_id,
        "role": "cloud",
        "status": status,
        "started_at": RUN_STARTED_AT,
        "finished_at": finished_at,
        "metrics_root": str(METRICS_ROOT),
        "run_dir": str(run_dir) if run_dir else None,
        "metrics_dir": str(cloud_dir) if cloud_dir else None,
        "metrics_files": {
            "latency_events": str(cloud_dir / "latency_events_cloud.jsonl") if cloud_dir else None,
            "resource": str(cloud_dir / "resource_cloud.jsonl") if cloud_dir else None,
            "manifest": str(cloud_dir / "manifest_cloud.json") if cloud_dir else None,
        },
        "host": host_info(),
        "runtime": runtime_versions(),
        "experiment": {
            "model_profile": MODEL_PROFILE_NAME,
            "model_display_name": MODEL_PROFILE["display_name"],
            "model_id": MODEL_ID,
            "dynamic_model_type": DYNAMIC_MODEL_TYPE,
            "edge_model_path_for_layer_count": EDGE_MODEL_PATH,
            "server_host": SERVER_HOST,
            "server_port": SERVER_PORT,
            "default_layer_split": LAYER_TO_SPLIT,
            "total_layers": total_layers,
            "torch_dtype": str(DEFAULT_DTYPE),
            "max_new_tokens": MAX_NEW_TOKENS,
            "fixed_new_tokens": True,
            "early_stop_enabled": False,
            "do_sample": DO_SAMPLE,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
        },
        "timings": {
            "model_load_s": model_load_s,
        },
        "resource_sampling": {
            "interval_s": RESOURCE_SAMPLE_INTERVAL_S,
            "cloud_extra_sources": ["torch.cuda", "psutil", "nvidia-smi"],
        },
        "error": str(error) if error is not None else None,
    }


def sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


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


def run_cloud_forward(model, hidden, attention_mask, position_ids, past_key_values, layer_split):
    outputs = dynamic_forward(
        model,
        hidden,
        start_layer=layer_split,
        model_type=DYNAMIC_MODEL_TYPE,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=True,
        apply_final_norm=True,
    )
    sync_device(hidden.device)
    return outputs


def runtime_config_dict():
    return {
        "model_id": MODEL_ID,
        "layer_split": LAYER_TO_SPLIT,
        "torch_dtype": str(DEFAULT_DTYPE),
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": DO_SAMPLE,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
    }


def request_config_dict(config):
    return {
        "model_id": config.model_id,
        "layer_split": config.layer_split,
        "torch_dtype": str(parse_torch_dtype(config.torch_dtype, DEFAULT_DTYPE)),
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.do_sample,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
    }


class QwenShardServicer(pb2_grpc.ShardServiceServicer):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_config = runtime_config_dict()
        self.model = None
        self.past_key_values = None
        self.current_length = 0
        self.layer_split = LAYER_TO_SPLIT
        self.total_layers = None
        self.model_load_s = None
        self.metrics = CloudMetrics(RUN_ID)
        if RUN_ID:
            self.metrics.set_context(
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
        self._preload_model()
        if RUN_ID:
            self.metrics.write_manifest(
                build_manifest("running", RUN_ID, self.total_layers, self.model_load_s)
            )

    def _preload_model(self):
        dtype = parse_torch_dtype(self.current_config["torch_dtype"], DEFAULT_DTYPE)
        print(f"Loading cloud model: {MODEL_PROFILE['display_name']} ({MODEL_ID})")
        print(f"Model profile: {MODEL_PROFILE_NAME}")
        print(f"Dynamic forward backend: {DYNAMIC_MODEL_TYPE}")
        print(f"Using device: {self.device}, dtype: {dtype}, suffix starts at layer: {LAYER_TO_SPLIT}")

        load_start = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self.total_layers = len(get_text_model(model).layers)
        model.eval()
        model.to(self.device)
        self.model = model
        sync_device(self.device)
        self.model_load_s = time.perf_counter() - load_start
        print("Cloud model ready.")

    def InitConnection(self, request, context):
        trace = trace_from_metadata(context.invocation_metadata())
        self.metrics.ensure_run(trace)
        trace.setdefault("run_id", self.metrics.run_id)
        self.metrics.set_context(state=trace.get("phase", "init"), **trace)
        try:
            requested = request_config_dict(request.config)
            expected = dict(self.current_config)
            requested_base = dict(requested)
            expected.pop("layer_split", None)
            requested_base.pop("layer_split", None)
            if requested_base != expected:
                message = (
                    "Requested runtime config does not match preloaded cloud config. "
                    f"requested={requested}, preloaded={self.current_config}"
                )
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details(message)
                return pb2.InitResponse(status=412, message=message)

            layer_split = requested["layer_split"]
            if layer_split <= 0 or layer_split >= self.total_layers:
                message = f"Requested layer_split must be in [1, {self.total_layers - 1}], got {layer_split}"
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(message)
                return pb2.InitResponse(status=400, message=message)

            self.layer_split = layer_split
            self.past_key_values = None
            self.current_length = 0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.metrics.write_manifest(
                build_manifest("running", self.metrics.run_id, self.total_layers, self.model_load_s)
            )
            return pb2.InitResponse(status=200, message="ok")
        except Exception as exc:
            self.metrics.write_manifest(
                build_manifest(
                    "failed",
                    self.metrics.run_id,
                    self.total_layers,
                    self.model_load_s,
                    error=exc,
                    finished_at=now_utc_iso(),
                )
            )
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return pb2.InitResponse(status=500, message=str(exc))

    def _trace_from_context(self, context, fallback_stage):
        trace = trace_from_metadata(context.invocation_metadata())
        self.metrics.ensure_run(trace)
        trace.setdefault("run_id", self.metrics.run_id)
        trace.setdefault("phase", "unknown")
        trace.setdefault("cut_layer", self.layer_split)
        trace.setdefault("stage", fallback_stage)
        self.metrics.set_context(state=trace.get("phase", "running"), **trace)
        return trace

    def _write_cloud_latency_event(self, trace, row):
        output = dict(trace)
        output.update(row)
        output.update(cuda_memory_snapshot("cloud"))
        self.metrics.write_latency(output)

    def PrefillStep(self, request, context):
        trace = self._trace_from_context(context, "prefill")
        reset_cuda_peak_memory(self.device)
        start_unix = time.time()
        start_perf = time.perf_counter()

        hidden_deserialize_start = time.perf_counter()
        hidden = deserialize_tensor(request.hidden, self.device)
        hidden = maybe_make_hidden_contiguous(hidden)
        hidden_deserialize_s = time.perf_counter() - hidden_deserialize_start

        attention_deserialize_start = time.perf_counter()
        attention_mask = deserialize_tensor(request.attention_mask, self.device)
        attention_mask_deserialize_s = time.perf_counter() - attention_deserialize_start

        position_deserialize_start = time.perf_counter()
        position_ids = deserialize_tensor(request.position_ids, self.device)
        position_ids_deserialize_s = time.perf_counter() - position_deserialize_start
        cloud_input_deserialize_s = (
            hidden_deserialize_s + attention_mask_deserialize_s + position_ids_deserialize_s
        )

        with torch.inference_mode():
            sync_device(self.device)
            forward_start = time.perf_counter()
            outputs = run_cloud_forward(
                self.model,
                hidden,
                attention_mask,
                position_ids,
                None,
                self.layer_split,
            )
            cloud_forward_s = time.perf_counter() - forward_start
            self.past_key_values = outputs.past_key_values

            sync_device(self.device)
            lm_head_start = time.perf_counter()
            logits = self.model.lm_head(outputs.last_hidden_state[:, -1:, :])
            sync_device(self.device)
            cloud_lm_head_s = time.perf_counter() - lm_head_start

            sample_start = time.perf_counter()
            next_token = sample_next_token(logits[:, -1, :], DO_SAMPLE, TEMPERATURE, TOP_P, TOP_K)
            sync_device(self.device)
            cloud_sample_s = time.perf_counter() - sample_start

        serialize_start = time.perf_counter()
        next_token_data = serialize_tensor(next_token)
        cloud_output_serialize_s = time.perf_counter() - serialize_start

        self.current_length = attention_mask.shape[1] + 1
        end_unix = time.time()
        handler_total_s = time.perf_counter() - start_perf
        cloud_infer_s = cloud_forward_s + cloud_lm_head_s + cloud_sample_s

        self._write_cloud_latency_event(
            trace,
            {
                "event_start_unix": start_unix,
                "event_end_unix": end_unix,
                "cloud_handler_total_s": handler_total_s,
                "cloud_input_deserialize_s": cloud_input_deserialize_s,
                "cloud_hidden_deserialize_s": hidden_deserialize_s,
                "cloud_attention_mask_deserialize_s": attention_mask_deserialize_s,
                "cloud_position_ids_deserialize_s": position_ids_deserialize_s,
                "cloud_attention_mask_build_s": 0.0,
                "cloud_forward_s": cloud_forward_s,
                "cloud_lm_head_s": cloud_lm_head_s,
                "cloud_sample_s": cloud_sample_s,
                "cloud_output_serialize_s": cloud_output_serialize_s,
                "cloud_infer_s": cloud_infer_s,
                "cloud_non_forward_s": handler_total_s - cloud_forward_s,
                "cloud_non_infer_s": handler_total_s - cloud_infer_s,
                "generated_token_id": int(next_token.item()),
                "cloud_current_length_after": self.current_length,
                "hidden_shape": list(hidden.shape),
                "attention_mask_shape": list(attention_mask.shape),
                "position_ids_shape": list(position_ids.shape),
                "request_hidden_raw_bytes": tensor_data_raw_nbytes(request.hidden),
                "request_attention_mask_raw_bytes": tensor_data_raw_nbytes(request.attention_mask),
                "request_position_ids_raw_bytes": tensor_data_raw_nbytes(request.position_ids),
                "request_hidden_serialized_bytes": tensor_data_serialized_bytes(request.hidden),
                "request_attention_mask_serialized_bytes": tensor_data_serialized_bytes(request.attention_mask),
                "request_position_ids_serialized_bytes": tensor_data_serialized_bytes(request.position_ids),
                "request_serialized_bytes": (
                    tensor_data_serialized_bytes(request.hidden)
                    + tensor_data_serialized_bytes(request.attention_mask)
                    + tensor_data_serialized_bytes(request.position_ids)
                ),
                "hidden_raw_bytes_after_deserialize": tensor_raw_nbytes(hidden),
                "response_serialized_bytes": tensor_data_serialized_bytes(next_token_data),
            },
        )
        return pb2.PrefillResponse(
            next_token=next_token_data,
            server_infer_s=cloud_infer_s,
            server_total_s=handler_total_s,
        )

    def DecodeStep(self, request, context):
        trace = self._trace_from_context(context, "decode")
        reset_cuda_peak_memory(self.device)
        start_unix = time.time()
        start_perf = time.perf_counter()

        hidden_deserialize_start = time.perf_counter()
        hidden = deserialize_tensor(request.hidden, self.device)
        hidden = maybe_make_hidden_contiguous(hidden)
        hidden_deserialize_s = time.perf_counter() - hidden_deserialize_start

        mask_build_start = time.perf_counter()
        attention_mask = torch.ones((1, self.current_length), device=self.device, dtype=torch.long)
        position_ids = torch.tensor([[self.current_length - 1]], device=self.device, dtype=torch.long)
        sync_device(self.device)
        attention_mask_build_s = time.perf_counter() - mask_build_start

        with torch.inference_mode():
            sync_device(self.device)
            forward_start = time.perf_counter()
            outputs = run_cloud_forward(
                self.model,
                hidden,
                attention_mask,
                position_ids,
                self.past_key_values,
                self.layer_split,
            )
            cloud_forward_s = time.perf_counter() - forward_start
            self.past_key_values = outputs.past_key_values

            sync_device(self.device)
            lm_head_start = time.perf_counter()
            logits = self.model.lm_head(outputs.last_hidden_state[:, -1:, :])
            sync_device(self.device)
            cloud_lm_head_s = time.perf_counter() - lm_head_start

            sample_start = time.perf_counter()
            next_token = sample_next_token(logits[:, -1, :], DO_SAMPLE, TEMPERATURE, TOP_P, TOP_K)
            sync_device(self.device)
            cloud_sample_s = time.perf_counter() - sample_start

        serialize_start = time.perf_counter()
        next_token_data = serialize_tensor(next_token)
        cloud_output_serialize_s = time.perf_counter() - serialize_start

        self.current_length += 1
        end_unix = time.time()
        handler_total_s = time.perf_counter() - start_perf
        cloud_infer_s = cloud_forward_s + cloud_lm_head_s + cloud_sample_s

        self._write_cloud_latency_event(
            trace,
            {
                "event_start_unix": start_unix,
                "event_end_unix": end_unix,
                "cloud_handler_total_s": handler_total_s,
                "cloud_input_deserialize_s": hidden_deserialize_s,
                "cloud_hidden_deserialize_s": hidden_deserialize_s,
                "cloud_attention_mask_deserialize_s": 0.0,
                "cloud_position_ids_deserialize_s": 0.0,
                "cloud_attention_mask_build_s": attention_mask_build_s,
                "cloud_forward_s": cloud_forward_s,
                "cloud_lm_head_s": cloud_lm_head_s,
                "cloud_sample_s": cloud_sample_s,
                "cloud_output_serialize_s": cloud_output_serialize_s,
                "cloud_infer_s": cloud_infer_s,
                "cloud_non_forward_s": handler_total_s - cloud_forward_s,
                "cloud_non_infer_s": handler_total_s - cloud_infer_s,
                "generated_token_id": int(next_token.item()),
                "cloud_current_length_after": self.current_length,
                "hidden_shape": list(hidden.shape),
                "attention_mask_shape": list(attention_mask.shape),
                "position_ids_shape": list(position_ids.shape),
                "request_hidden_raw_bytes": tensor_data_raw_nbytes(request.hidden),
                "request_attention_mask_raw_bytes": 0,
                "request_position_ids_raw_bytes": 0,
                "request_hidden_serialized_bytes": tensor_data_serialized_bytes(request.hidden),
                "request_attention_mask_serialized_bytes": 0,
                "request_position_ids_serialized_bytes": 0,
                "request_serialized_bytes": tensor_data_serialized_bytes(request.hidden),
                "hidden_raw_bytes_after_deserialize": tensor_raw_nbytes(hidden),
                "response_serialized_bytes": tensor_data_serialized_bytes(next_token_data),
            },
        )
        return pb2.DecodeResponse(
            next_token=next_token_data,
            server_infer_s=cloud_infer_s,
            server_total_s=handler_total_s,
        )


def serve():
    options = [
        ("grpc.max_receive_message_length", 100 * 1024 * 1024),
        ("grpc.max_send_message_length", 100 * 1024 * 1024),
    ]
    servicer = QwenShardServicer()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1), options=options)
    pb2_grpc.add_ShardServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"{SERVER_HOST}:{SERVER_PORT}")
    server.start()
    print(f"{MODEL_PROFILE['display_name']} cloud server listening on {SERVER_HOST}:{SERVER_PORT}")
    print(f"Force contiguous hidden: {FORCE_CONTIGUOUS_HIDDEN}")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        if servicer.metrics.run_id:
            servicer.metrics.set_context(
                state="stopped",
                phase="finished",
                cut_layer=None,
                dataset=None,
                record_id=None,
                sample_index=None,
                stage="stopped",
                generated_token_index=None,
                input_token_count=None,
                context_token_length=None,
                event_id=None,
            )
            servicer.metrics.write_manifest(
                build_manifest(
                    "stopped",
                    servicer.metrics.run_id,
                    servicer.total_layers,
                    servicer.model_load_s,
                    finished_at=now_utc_iso(),
                )
            )
        servicer.metrics.close()
        server.stop(0)


def apply_args(args):
    global MODEL_PROFILE_NAME, MODEL_PROFILE, MODEL_ID, DYNAMIC_MODEL_TYPE, FORCE_CONTIGUOUS_HIDDEN

    if args.model:
        MODEL_PROFILE_NAME = normalize_model_profile_name(args.model)
        MODEL_PROFILE = MODEL_PROFILES[MODEL_PROFILE_NAME]
        DYNAMIC_MODEL_TYPE = MODEL_PROFILE["dynamic_model_type"]
        MODEL_ID = resolve_model_id(MODEL_PROFILE_NAME)

    if args.model_id:
        MODEL_ID = args.model_id

    FORCE_CONTIGUOUS_HIDDEN = FORCE_CONTIGUOUS_HIDDEN or args.contiguous_hidden


def main():
    args = parse_args()
    apply_args(args)
    serve()


if __name__ == "__main__":
    main()
