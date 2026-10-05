import argparse
import csv
from collections import defaultdict
import gc
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("USE_TORCH", "1")

import torch
from torch import nn
from qwen_inputs import load_input_file

from qwen_metrics import (
    JsonlWriter,
    collect_nvidia_smi,
    cuda_memory_snapshot,
    host_info,
    make_run_id,
    now_utc_iso,
    reset_cuda_peak_memory,
    runtime_versions,
    tensor_raw_nbytes,
    write_json,
)


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_ID = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen3.5-9B")
TOKENIZED_FILENAME = os.environ.get("QWEN_TOKENIZED_FILENAME", "qwen_non_thinking.jsonl")
SHORT_DATASET = "wildgpt4"
LONG_DATASET = "longreason"
DEFAULT_CUT_LAYERS = list(range(1, 13))
DEFAULT_MAX_NEW_TOKENS = 64
DTYPE = torch.bfloat16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DYNAMIC_FORWARD = None


def parse_cut_layers(text):
    if not text:
        return list(DEFAULT_CUT_LAYERS)
    text = text.strip()
    if "-" in text and "," not in text:
        start, end = text.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Dump Qwen hidden states on one node, then replay suffix forward from those "
            "hidden states on another node."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    dump = subparsers.add_parser("dump", help="Generate hidden states from the full model.")
    dump.add_argument("--input-file", default=None, help="Prepared Qwen JSONL; replaces testset counts.")
    dump.add_argument("--model-id", default=MODEL_ID)
    dump.add_argument("--selected-testset-root", default=os.environ.get("QWEN_SELECTED_TESTSET_ROOT"))
    dump.add_argument("--tokenized-filename", default=TOKENIZED_FILENAME)
    dump.add_argument("--output-root", default=f"/local/{os.environ.get('USER', 'user')}/qwen_remote_hidden")
    dump.add_argument("--short-count", type=int, default=16)
    dump.add_argument("--long-count", type=int, default=0)
    dump.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    dump.add_argument("--cut-layers", default="1-12")
    dump.add_argument("--allow-cpu", action="store_true", help="Allow CPU execution instead of failing.")

    replay = subparsers.add_parser("replay", help="Replay suffix forward from dumped hidden states.")
    replay.add_argument("--hidden-dir", required=True)
    replay.add_argument("--model-id", default=MODEL_ID)
    replay.add_argument("--results-root", default=str(SCRIPT_DIR / "results_remote_hidden"))
    replay.add_argument("--backend", choices=("dy", "official"), default="dy")
    replay.add_argument(
        "--dataset",
        choices=("wildgpt4", "longreason", "all"),
        default="wildgpt4",
        help="Replay only this dataset from the hidden index. Default: wildgpt4 only.",
    )
    replay.add_argument("--cut-layers", default=None)
    replay.add_argument("--warmup-per-dataset", type=int, default=0)
    replay.add_argument("--delay-base-ms", type=float, default=0.0)
    replay.add_argument("--delay-per-cut-ms", type=float, default=0.0)
    replay.add_argument(
        "--delay-sweep",
        action="store_true",
        help="Run a fixed inter-token delay sweep instead of a single delay setting.",
    )
    replay.add_argument("--delay-sweep-step-ms", type=float, default=2.5)
    replay.add_argument("--delay-sweep-max-ms", type=float, default=35.0)
    replay.add_argument(
        "--delay-values-ms",
        default=None,
        help=(
            "Comma-separated fixed decode idle gaps, for example 0,5,20. "
            "Overrides --delay-sweep and the single-delay arguments."
        ),
    )
    replay.add_argument(
        "--max-hidden-samples",
        type=int,
        default=None,
        help="Use only the first N selected hidden samples for a compact experiment.",
    )
    replay.add_argument(
        "--run-label",
        default=None,
        help="Optional label recorded in the run and metrics, such as unlocked or locked.",
    )
    replay.add_argument("--allow-cpu", action="store_true", help="Allow CPU execution instead of failing.")
    return parser.parse_args()


def sync_device():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def device_info():
    info = {
        "device": str(DEVICE),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        current = torch.cuda.current_device()
        info.update(
            {
                "cuda_current_device": int(current),
                "cuda_device_name": torch.cuda.get_device_name(current),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            }
        )
    return info


def require_cuda_unless_allowed(allow_cpu):
    info = device_info()
    print(
        "Runtime device: "
        f"device={info['device']} cuda_available={info['cuda_available']} "
        f"device_count={info['cuda_device_count']} "
        f"name={info.get('cuda_device_name')}",
        flush=True,
    )
    if DEVICE.type != "cuda" and not allow_cpu:
        raise RuntimeError(
            "CUDA is not available, so this run would use CPU. "
            "Fix the node CUDA environment or pass --allow-cpu intentionally."
        )


def resolve_selected_testset_root(explicit_root):
    if explicit_root:
        root = Path(explicit_root).expanduser()
        if root.exists():
            return root
        raise FileNotFoundError(f"Selected testset root does not exist: {root}")

    candidates = [
        (SCRIPT_DIR / "../../inputs/selected_testsets").resolve(),
        (SCRIPT_DIR / "../../selected_testsets").resolve(),
        (SCRIPT_DIR / "../../testSet/selected_testsets").resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find selected_testsets. Set QWEN_SELECTED_TESTSET_ROOT.")


def load_tokenized_records(root, dataset_name, tokenized_filename, count):
    if count <= 0:
        return []

    path = root / dataset_name / tokenized_filename
    records = []
    with path.open() as input_file:
        for line in input_file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["input_ids"] = [int(token_id) for token_id in record["input_ids"]]
            records.append(record)
            if len(records) >= count:
                break
    if len(records) < count:
        raise ValueError(f"Only found {len(records)} records in {path}; requested {count}.")
    return records


def record_summary(record, sample_index):
    return {
        "input_phase": record.get("phase"),
        "dataset": record.get("dataset"),
        "record_id": record.get("record_id"),
        "source": record.get("source"),
        "source_label": record.get("source_label"),
        "source_split": record.get("source_split"),
        "source_row": record.get("source_row"),
        "source_sample_index": record.get("sample_index"),
        "sample_index": sample_index,
        "input_token_count": len(record["input_ids"]),
        "qwen_length": record.get("qwen_length"),
        "llama_length": record.get("llama_length"),
        "qwen_thinking_mode": record.get("qwen_thinking_mode"),
    }


def build_attention_and_position(input_ids):
    input_ids = torch.tensor([input_ids], device=DEVICE, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, device=DEVICE, dtype=torch.long)
    position_ids = attention_mask.long().cumsum(dim=1) - 1
    return input_ids, attention_mask, position_ids


def sample_next_token(logits):
    return torch.argmax(logits, dim=-1, keepdim=True)


def get_text_model(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model
    raise AttributeError("Could not find text model layers.")


def load_full_model(model_id, load_tokenizer=True):
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM

    print(f"Loading model: {model_id}")
    model = Qwen3_5ForCausalLM.from_pretrained(
        model_id,
        torch_dtype=DTYPE,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True) if load_tokenizer else None
    model.eval()
    model.to(DEVICE)
    sync_device()
    return model, tokenizer


def make_capture_hooks(model, cut_layers):
    text_model = get_text_model(model)
    captured = {}
    handles = []
    wanted = set(cut_layers)
    max_cut = max(wanted)
    if max_cut > len(text_model.layers):
        raise ValueError(f"Cannot capture cut layer {max_cut}; model has {len(text_model.layers)} layers.")

    def make_hook(cut_layer):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[cut_layer] = hidden.detach().to("cpu", dtype=DTYPE).contiguous()

        return hook

    for layer_index, layer in enumerate(text_model.layers[:max_cut], start=1):
        if layer_index in wanted:
            handles.append(layer.register_forward_hook(make_hook(layer_index)))
    return captured, handles


def read_json(path):
    with Path(path).open() as input_file:
        return json.load(input_file)


def load_hidden_blob(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_dyforward_dir():
    for dirname in ("dyforward", "dyForward"):
        candidate = (SCRIPT_DIR / f"../../{dirname}").resolve()
        if (candidate / "dynamic_forward.py").exists():
            return candidate
    return (SCRIPT_DIR / "../../dyforward").resolve()


def ensure_dynamic_forward_loaded():
    global DYNAMIC_FORWARD
    if DYNAMIC_FORWARD is not None:
        return DYNAMIC_FORWARD

    dyforward_dir = resolve_dyforward_dir()
    if str(dyforward_dir) not in sys.path:
        sys.path.insert(0, str(dyforward_dir))

    from dynamic_forward import dynamic_forward

    DYNAMIC_FORWARD = dynamic_forward
    return DYNAMIC_FORWARD


def set_layer_idx_recursively(layer, new_index):
    for module in layer.modules():
        if hasattr(module, "layer_idx"):
            module.layer_idx = new_index


def crop_model_to_suffix(model, cut_layer):
    text_model = get_text_model(model)
    total_layers = len(text_model.layers)
    if cut_layer <= 0 or cut_layer >= total_layers:
        raise ValueError(f"cut_layer must be in [1, {total_layers - 1}], got {cut_layer}")

    suffix_layers = list(text_model.layers[cut_layer:])
    text_model.layers = nn.ModuleList(suffix_layers)

    layer_types = getattr(text_model.config, "layer_types", None)
    if layer_types is not None:
        text_model.config.layer_types = list(layer_types[cut_layer:])
    text_model.config.num_hidden_layers = len(suffix_layers)
    if hasattr(model, "config"):
        model.config.num_hidden_layers = len(suffix_layers)
        if layer_types is not None:
            model.config.layer_types = list(text_model.config.layer_types)

    for new_index, layer in enumerate(text_model.layers):
        set_layer_idx_recursively(layer, new_index)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return total_layers, len(suffix_layers)


def maybe_delay(stage, cut_layer, delay_base_ms, delay_per_cut_ms):
    if stage != "decode":
        return 0.0, 0.0
    scheduled_s = (delay_base_ms + delay_per_cut_ms * cut_layer) / 1000.0
    if scheduled_s <= 0.0:
        return 0.0, 0.0
    start = time.perf_counter()
    time.sleep(scheduled_s)
    return scheduled_s, time.perf_counter() - start


def format_delay_name(delay_ms):
    text = f"{delay_ms:.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def build_delay_profiles(args):
    if args.delay_values_ms:
        values = []
        seen = set()
        for item in args.delay_values_ms.split(","):
            item = item.strip()
            if not item:
                continue
            value = float(item)
            if value < 0.0:
                raise ValueError("--delay-values-ms cannot contain negative values.")
            value = round(value, 6)
            if value not in seen:
                values.append(value)
                seen.add(value)
        if not values:
            raise ValueError("--delay-values-ms did not contain any values.")
        return [
            {
                "name": f"fixed_{format_delay_name(value)}ms",
                "delay_base_ms": value,
                "delay_per_cut_ms": 0.0,
                "fixed_delay_ms": value,
            }
            for value in values
        ]

    if not args.delay_sweep:
        delay_ms = args.delay_base_ms
        per_cut_ms = args.delay_per_cut_ms
        if delay_ms == 0.0 and per_cut_ms == 0.0:
            name = "none"
        elif per_cut_ms == 0.0:
            name = f"fixed_{format_delay_name(delay_ms)}ms"
        else:
            name = f"base_{format_delay_name(delay_ms)}ms_percut_{format_delay_name(per_cut_ms)}ms"
        return [
            {
                "name": name,
                "delay_base_ms": delay_ms,
                "delay_per_cut_ms": per_cut_ms,
                "fixed_delay_ms": delay_ms if per_cut_ms == 0.0 else None,
            }
        ]

    if args.delay_sweep_step_ms <= 0.0:
        raise ValueError("--delay-sweep-step-ms must be positive.")
    if args.delay_sweep_max_ms < 0.0:
        raise ValueError("--delay-sweep-max-ms must be non-negative.")

    values = []
    current = 0.0
    epsilon = args.delay_sweep_step_ms / 1000.0
    while current <= args.delay_sweep_max_ms + epsilon:
        values.append(round(current, 6))
        current += args.delay_sweep_step_ms

    return [
        {
            "name": f"fixed_{format_delay_name(value)}ms",
            "delay_base_ms": value,
            "delay_per_cut_ms": 0.0,
            "fixed_delay_ms": value,
        }
        for value in values
    ]


def run_official_suffix_step(model, hidden_cpu, context_token_length, past_key_values):
    attention_mask = torch.ones((1, context_token_length), device=DEVICE, dtype=torch.long)
    if hidden_cpu.shape[1] == context_token_length:
        position_ids = torch.arange(context_token_length, device=DEVICE, dtype=torch.long).unsqueeze(0)
    else:
        position_ids = torch.tensor([[context_token_length - 1]], device=DEVICE, dtype=torch.long)

    sync_device()
    h2d_start = time.perf_counter()
    hidden = hidden_cpu.to(DEVICE, non_blocking=True)
    sync_device()
    hidden_to_device_s = time.perf_counter() - h2d_start

    reset_cuda_peak_memory(DEVICE)
    sync_device()
    forward_start = time.perf_counter()
    outputs = get_text_model(model)(
        inputs_embeds=hidden,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=True,
        return_dict=True,
    )
    sync_device()
    suffix_forward_s = time.perf_counter() - forward_start

    sync_device()
    lm_head_start = time.perf_counter()
    logits = model.lm_head(outputs.last_hidden_state[:, -1:, :])
    sync_device()
    lm_head_s = time.perf_counter() - lm_head_start

    sync_device()
    sample_start = time.perf_counter()
    next_token = sample_next_token(logits[:, -1, :])
    sync_device()
    sample_s = time.perf_counter() - sample_start

    return {
        "past_key_values": outputs.past_key_values,
        "generated_token_id": int(next_token.item()),
        "hidden_to_device_s": hidden_to_device_s,
        "suffix_forward_s": suffix_forward_s,
        "lm_head_s": lm_head_s,
        "sample_s": sample_s,
        "hidden_shape": list(hidden.shape),
        "hidden_raw_bytes": tensor_raw_nbytes(hidden),
    }


def run_dyforward_suffix_step(model, hidden_cpu, context_token_length, past_key_values, cut_layer):
    dynamic_forward = ensure_dynamic_forward_loaded()
    attention_mask = torch.ones((1, context_token_length), device=DEVICE, dtype=torch.long)
    if hidden_cpu.shape[1] == context_token_length:
        position_ids = torch.arange(context_token_length, device=DEVICE, dtype=torch.long).unsqueeze(0)
    else:
        position_ids = torch.tensor([[context_token_length - 1]], device=DEVICE, dtype=torch.long)

    sync_device()
    h2d_start = time.perf_counter()
    hidden = hidden_cpu.to(DEVICE, non_blocking=True)
    sync_device()
    hidden_to_device_s = time.perf_counter() - h2d_start

    reset_cuda_peak_memory(DEVICE)
    sync_device()
    forward_start = time.perf_counter()
    outputs = dynamic_forward(
        model,
        hidden,
        start_layer=cut_layer,
        model_type="qwen3_5",
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=True,
        apply_final_norm=True,
    )
    sync_device()
    suffix_forward_s = time.perf_counter() - forward_start

    sync_device()
    lm_head_start = time.perf_counter()
    logits = model.lm_head(outputs.last_hidden_state[:, -1:, :])
    sync_device()
    lm_head_s = time.perf_counter() - lm_head_start

    sync_device()
    sample_start = time.perf_counter()
    next_token = sample_next_token(logits[:, -1, :])
    sync_device()
    sample_s = time.perf_counter() - sample_start

    return {
        "past_key_values": outputs.past_key_values,
        "generated_token_id": int(next_token.item()),
        "hidden_to_device_s": hidden_to_device_s,
        "suffix_forward_s": suffix_forward_s,
        "lm_head_s": lm_head_s,
        "sample_s": sample_s,
        "hidden_shape": list(hidden.shape),
        "hidden_raw_bytes": tensor_raw_nbytes(hidden),
    }


def percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def mean(values):
    return sum(values) / len(values) if values else None


def dump_hidden(args):
    require_cuda_unless_allowed(args.allow_cpu)
    cut_layers = parse_cut_layers(args.cut_layers)
    run_id = make_run_id("qwen9b_remote_hidden_source")
    run_dir = Path(args.output_root).expanduser() / run_id
    hidden_dir = run_dir / "hidden"
    run_dir.mkdir(parents=True, exist_ok=True)
    hidden_dir.mkdir(parents=True, exist_ok=True)

    if args.input_file:
        formal, warmup = load_input_file(args.input_file, "qwen")
        records = warmup + formal
        selected_root = None
    else:
        selected_root = resolve_selected_testset_root(args.selected_testset_root)
        records = load_tokenized_records(selected_root, SHORT_DATASET, args.tokenized_filename, args.short_count)
        records.extend(load_tokenized_records(selected_root, LONG_DATASET, args.tokenized_filename, args.long_count))

    manifest_path = run_dir / "manifest_hidden.json"
    hidden_index_path = run_dir / "hidden_index.jsonl"
    generation_path = run_dir / "hidden_generation.jsonl"
    write_json(
        manifest_path,
        {
            "run_id": run_id,
            "status": "running",
            "started_at": now_utc_iso(),
            "model_id": args.model_id,
            "torch_dtype": str(DTYPE),
            "device": str(DEVICE),
            "input_file": str(Path(args.input_file).expanduser().resolve()) if args.input_file else None,
            "selected_testset_root": str(selected_root) if selected_root else None,
            "tokenized_filename": args.tokenized_filename if not args.input_file else None,
            "short_count": sum(r["input_group"] == "short" for r in records) if args.input_file else args.short_count,
            "long_count": sum(r["input_group"] == "long" for r in records) if args.input_file else args.long_count,
            "cut_layers": cut_layers,
            "captured_layer_indices": [cut - 1 for cut in cut_layers],
            "max_new_tokens": args.max_new_tokens,
            "run_dir": str(run_dir),
            "hidden_dir": str(hidden_dir),
            "metrics_files": {
                "hidden_index": str(hidden_index_path),
                "hidden_generation": str(generation_path),
            },
            "host": host_info(),
            "runtime": runtime_versions(),
            "device_info": device_info(),
        },
    )

    print(f"Hidden source run ID: {run_id}")
    print(f"Hidden output dir: {run_dir}")
    print(f"Records: {len(records)}" + (" (explicit input phases)" if args.input_file else ""))
    print(f"Cut layers: {cut_layers}")
    print(f"Max new tokens: {args.max_new_tokens}")

    index_writer = JsonlWriter(hidden_index_path, flush_each_write=True)
    generation_writer = JsonlWriter(generation_path, flush_each_write=True)
    hidden_rows = []
    model = None
    tokenizer = None
    handles = []
    try:
        model, tokenizer = load_full_model(args.model_id, load_tokenizer=True)
        text_model = get_text_model(model)
        total_layers = len(text_model.layers)
        captured, handles = make_capture_hooks(model, cut_layers)

        for sample_index, record in enumerate(records, start=1):
            sample = record_summary(record, sample_index)
            input_ids, attention_mask, position_ids = build_attention_and_position(record["input_ids"])
            generated_ids = input_ids.clone()
            generated_token_ids = []
            past_key_values = None
            steps = []
            sample_start = time.perf_counter()

            print(
                f"Dump hidden {sample_index}/{len(records)} "
                f"{sample['dataset']}#{sample['record_id']} tokens={sample['input_token_count']}"
            )

            with torch.inference_mode():
                for generated_token_index in range(1, args.max_new_tokens + 1):
                    stage = "prefill" if generated_token_index == 1 else "decode"
                    context_token_length = int(attention_mask.shape[1])
                    if stage == "decode":
                        position_ids = torch.tensor(
                            [[context_token_length - 1]], device=DEVICE, dtype=torch.long
                        )

                    captured.clear()
                    reset_cuda_peak_memory(DEVICE)
                    sync_device()
                    forward_start = time.perf_counter()
                    outputs = model(
                        input_ids=input_ids if stage == "prefill" else generated_ids[:, -1:],
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )
                    sync_device()
                    full_forward_s = time.perf_counter() - forward_start

                    missing = [cut for cut in cut_layers if cut not in captured]
                    if missing:
                        raise RuntimeError(f"Missing captured hidden states for cut layers: {missing}")

                    next_token = sample_next_token(outputs.logits[:, -1, :])
                    token_id = int(next_token.item())
                    generated_token_ids.append(token_id)
                    past_key_values = outputs.past_key_values

                    step = {
                        "stage": stage,
                        "generated_token_index": generated_token_index,
                        "context_token_length": context_token_length,
                        "full_forward_s": full_forward_s,
                        "generated_token_id": token_id,
                        "hidden_by_cut": {cut: captured[cut] for cut in cut_layers},
                    }
                    steps.append(step)

                    generation_row = {
                        "run_id": run_id,
                        "sample_index": sample_index,
                        "dataset": sample["dataset"],
                        "record_id": sample["record_id"],
                        "stage": stage,
                        "generated_token_index": generated_token_index,
                        "context_token_length": context_token_length,
                        "full_forward_s": full_forward_s,
                        "generated_token_id": token_id,
                    }
                    generation_row.update(cuda_memory_snapshot("source"))
                    generation_writer.write(generation_row)

                    generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                    attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
                    input_ids = next_token

            output_text = tokenizer.decode(
                generated_ids[0][sample["input_token_count"] :], skip_special_tokens=True
            )
            hidden_path = hidden_dir / f"sample_{sample_index:04d}_{sample['dataset']}_{sample['record_id']}.pt"
            torch.save(
                {
                    "sample": sample,
                    "source_run_id": run_id,
                    "source_host": host_info(),
                    "model_id": args.model_id,
                    "torch_dtype": str(DTYPE),
                    "total_layers": total_layers,
                    "cut_layers": cut_layers,
                    "captured_layer_indices": [cut - 1 for cut in cut_layers],
                    "max_new_tokens": args.max_new_tokens,
                    "generated_token_ids": generated_token_ids,
                    "output_text": output_text,
                    "steps": steps,
                },
                hidden_path,
            )

            sample_total_s = time.perf_counter() - sample_start
            row = {
                **sample,
                "run_id": run_id,
                "hidden_path": str(hidden_path),
                "generated_token_count": len(generated_token_ids),
                "sample_total_s": sample_total_s,
                "output_text": output_text,
            }
            row.update(cuda_memory_snapshot("source"))
            row.update(collect_nvidia_smi())
            index_writer.write(row)
            hidden_rows.append(row)

            del steps, generated_ids, attention_mask, position_ids, past_key_values
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        write_json(
            manifest_path,
            {
                "run_id": run_id,
                "status": "completed",
                "started_at": read_json(manifest_path)["started_at"],
                "finished_at": now_utc_iso(),
                "model_id": args.model_id,
                "torch_dtype": str(DTYPE),
                "device": str(DEVICE),
                "selected_testset_root": str(selected_root),
                "tokenized_filename": args.tokenized_filename,
                "short_count": args.short_count,
                "long_count": args.long_count,
                "cut_layers": cut_layers,
                "captured_layer_indices": [cut - 1 for cut in cut_layers],
                "max_new_tokens": args.max_new_tokens,
                "run_dir": str(run_dir),
                "hidden_dir": str(hidden_dir),
                "metrics_files": {
                    "hidden_index": str(hidden_index_path),
                    "hidden_generation": str(generation_path),
                },
                "host": host_info(),
                "runtime": runtime_versions(),
                "device_info": device_info(),
                "records": hidden_rows,
            },
        )
    finally:
        for handle in handles:
            handle.remove()
        index_writer.close()
        generation_writer.close()
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("Done.")
    print(f"Hidden dir: {run_dir}")
    print(f"Manifest: {manifest_path}")


def load_hidden_index(hidden_dir):
    hidden_dir = Path(hidden_dir).expanduser()
    index_path = hidden_dir / "hidden_index.jsonl"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing hidden index: {index_path}")
    rows = []
    with index_path.open() as input_file:
        for line in input_file:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            hidden_path = Path(row["hidden_path"])
            if not hidden_path.exists():
                hidden_path = hidden_dir / "hidden" / hidden_path.name
            row["hidden_path"] = str(hidden_path)
            rows.append(row)
    return rows


def summarize_replay(sample_rows, output_path):
    grouped = defaultdict(list)
    for row in sample_rows:
        grouped[(row["backend"], row["delay_profile"], row["cut_layer"], row["dataset"])].append(row)

    fieldnames = [
        "backend",
        "delay_profile",
        "cut_layer",
        "dataset",
        "sample_count",
        "fixed_delay_ms",
        "delay_base_ms",
        "delay_per_cut_ms",
        "decode_actual_delay_per_token_mean_s",
        "prefill_hidden_to_device_mean_s",
        "prefill_suffix_forward_mean_s",
        "prefill_lm_head_mean_s",
        "prefill_sample_mean_s",
        "decode_hidden_to_device_per_token_mean_s",
        "decode_suffix_forward_per_token_mean_s",
        "decode_suffix_forward_per_token_p90_s",
        "decode_lm_head_per_token_mean_s",
        "decode_sample_per_token_mean_s",
        "sample_compute_total_mean_s",
        "sample_replay_total_mean_s",
        "next_token_match_rate",
    ]
    with Path(output_path).open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        def sort_key(item):
            (_backend, _delay_profile, cut_layer, dataset), rows = item
            fixed_delay_ms = rows[0].get("fixed_delay_ms")
            if fixed_delay_ms is None:
                fixed_delay_ms = rows[0].get("delay_base_ms", 0.0)
            return (dataset, cut_layer, float(fixed_delay_ms))

        for (backend, delay_profile, cut_layer, dataset), rows in sorted(grouped.items(), key=sort_key):
            writer.writerow(
                {
                    "backend": backend,
                    "delay_profile": delay_profile,
                    "cut_layer": cut_layer,
                    "dataset": dataset,
                    "sample_count": len(rows),
                    "fixed_delay_ms": rows[0]["fixed_delay_ms"],
                    "delay_base_ms": rows[0]["delay_base_ms"],
                    "delay_per_cut_ms": rows[0]["delay_per_cut_ms"],
                    "decode_actual_delay_per_token_mean_s": mean(
                        [row["decode_actual_delay_per_token_s"] for row in rows]
                    ),
                    "prefill_hidden_to_device_mean_s": mean([row["prefill_hidden_to_device_s"] for row in rows]),
                    "prefill_suffix_forward_mean_s": mean([row["prefill_suffix_forward_s"] for row in rows]),
                    "prefill_lm_head_mean_s": mean([row["prefill_lm_head_s"] for row in rows]),
                    "prefill_sample_mean_s": mean([row["prefill_sample_s"] for row in rows]),
                    "decode_hidden_to_device_per_token_mean_s": mean(
                        [row["decode_hidden_to_device_per_token_s"] for row in rows]
                    ),
                    "decode_suffix_forward_per_token_mean_s": mean(
                        [row["decode_suffix_forward_per_token_s"] for row in rows]
                    ),
                    "decode_suffix_forward_per_token_p90_s": percentile(
                        [row["decode_suffix_forward_per_token_s"] for row in rows], 90
                    ),
                    "decode_lm_head_per_token_mean_s": mean([row["decode_lm_head_per_token_s"] for row in rows]),
                    "decode_sample_per_token_mean_s": mean([row["decode_sample_per_token_s"] for row in rows]),
                    "sample_compute_total_mean_s": mean([row["sample_compute_total_s"] for row in rows]),
                    "sample_replay_total_mean_s": mean([row["sample_replay_total_s"] for row in rows]),
                    "next_token_match_rate": mean([row["next_token_match_rate"] for row in rows]),
                }
            )


def replay_hidden(args):
    require_cuda_unless_allowed(args.allow_cpu)
    source_dir = Path(args.hidden_dir).expanduser()
    source_manifest_path = source_dir / "manifest_hidden.json"
    source_manifest = read_json(source_manifest_path)
    cut_layers = parse_cut_layers(args.cut_layers) if args.cut_layers else list(source_manifest["cut_layers"])
    delay_profiles = build_delay_profiles(args)
    hidden_index = load_hidden_index(source_dir)
    if args.dataset != "all":
        hidden_index = [row for row in hidden_index if row.get("dataset") == args.dataset]
    if args.max_hidden_samples is not None:
        if args.max_hidden_samples <= 0:
            raise ValueError("--max-hidden-samples must be positive.")
        hidden_index = hidden_index[: args.max_hidden_samples]
    if not hidden_index:
        raise ValueError(f"No hidden records selected from {source_dir} with dataset={args.dataset!r}.")
    if all(row.get("input_phase") == "warmup" for row in hidden_index):
        raise ValueError("The selected hidden samples contain only warmup records.")
    run_label = None
    if args.run_label:
        run_label = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in args.run_label.strip()
        ).strip("_")
    run_prefix = f"qwen9b_remote_hidden_replay_{args.backend}"
    if run_label:
        run_prefix += f"_{run_label}"
    run_id = make_run_id(run_prefix)
    run_dir = Path(args.results_root).expanduser() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    event_writer = JsonlWriter(run_dir / "suffix_forward_events.jsonl", flush_each_write=True)
    sample_writer = JsonlWriter(run_dir / "suffix_forward_samples.jsonl", flush_each_write=True)
    cut_setup_writer = JsonlWriter(run_dir / "cut_setup.jsonl", flush_each_write=True)
    sample_rows = []
    model = None
    dy_model = None
    status = "running"
    error = None
    started_at = now_utc_iso()

    print(f"Replay run ID: {run_id}")
    print(f"Hidden source dir: {source_dir}")
    print(f"Results dir: {run_dir}")
    print(f"Backend: {args.backend}")
    print(f"Run label: {run_label or 'none'}")
    print(f"Replay dataset: {args.dataset} ({len(hidden_index)} hidden samples)")
    print(f"Cut layers: {cut_layers}")
    print(f"Warmup per dataset per cut: {args.warmup_per_dataset}")
    print(f"Delay profiles: {len(delay_profiles)}")
    for profile in delay_profiles:
        print(
            f"  {profile['name']}: "
            f"base={profile['delay_base_ms']:.1f}ms "
            f"per_cut={profile['delay_per_cut_ms']:.1f}ms"
        )

    write_json(
        run_dir / "manifest_replay.json",
        {
            "run_id": run_id,
            "status": status,
            "started_at": started_at,
            "model_id": args.model_id,
            "backend": args.backend,
            "run_label": run_label,
            "torch_dtype": str(DTYPE),
            "device": str(DEVICE),
            "hidden_source_dir": str(source_dir),
            "hidden_source_manifest": str(source_manifest_path),
            "hidden_source_run_id": source_manifest.get("run_id"),
            "hidden_source_host": source_manifest.get("host"),
            "source_max_new_tokens": source_manifest.get("max_new_tokens"),
            "replay_dataset": args.dataset,
            "selected_hidden_sample_count": len(hidden_index),
            "max_hidden_samples": args.max_hidden_samples,
            "cut_layers": cut_layers,
            "warmup_per_dataset_per_cut": args.warmup_per_dataset,
            "delay_sweep": args.delay_sweep,
            "delay_sweep_step_ms": args.delay_sweep_step_ms,
            "delay_sweep_max_ms": args.delay_sweep_max_ms,
            "delay_profiles": delay_profiles,
            "delay_base_ms": args.delay_base_ms,
            "delay_per_cut_ms": args.delay_per_cut_ms,
            "results_dir": str(run_dir),
            "host": host_info(),
            "runtime": runtime_versions(),
            "device_info": device_info(),
        },
    )

    try:
        if args.backend == "dy":
            ensure_dynamic_forward_loaded()
            load_start = time.perf_counter()
            dy_model, _tokenizer = load_full_model(args.model_id, load_tokenizer=False)
            dy_load_s = time.perf_counter() - load_start
            total_layers = len(get_text_model(dy_model).layers)
        else:
            dy_load_s = None
            total_layers = None

        for cut_layer in cut_layers:
            if args.backend == "dy":
                model = dy_model
                if cut_layer <= 0 or cut_layer >= total_layers:
                    raise ValueError(f"cut_layer must be in [1, {total_layers - 1}], got {cut_layer}")
                suffix_layers = total_layers - cut_layer
                model_load_s = dy_load_s if cut_layer == cut_layers[0] else 0.0
                crop_s = 0.0
            else:
                load_start = time.perf_counter()
                model, _tokenizer = load_full_model(args.model_id, load_tokenizer=False)
                model_load_s = time.perf_counter() - load_start
                crop_start = time.perf_counter()
                total_layers, suffix_layers = crop_model_to_suffix(model, cut_layer)
                crop_s = time.perf_counter() - crop_start
                model.eval()

            sync_device()
            print(f"Replay cut layer {cut_layer}: suffix layers={suffix_layers}/{total_layers}")
            cut_setup_row = {
                "run_id": run_id,
                "backend": args.backend,
                "run_label": run_label,
                "cut_layer": cut_layer,
                "total_layers": total_layers,
                "suffix_layers": suffix_layers,
                "model_load_s": model_load_s,
                "crop_s": crop_s,
                "model_loaded_once": args.backend == "dy",
                "delay_base_ms": args.delay_base_ms,
                "delay_per_cut_ms": args.delay_per_cut_ms,
            }
            cut_setup_row.update(cuda_memory_snapshot("replay"))
            cut_setup_row.update(collect_nvidia_smi())
            cut_setup_writer.write(cut_setup_row)

            for delay_profile in delay_profiles:
                print(
                    f"  Delay profile {delay_profile['name']} "
                    f"(fixed={delay_profile['fixed_delay_ms']}ms)",
                    flush=True,
                )
                seen_by_dataset = defaultdict(int)
                formal_seen_by_dataset = defaultdict(int)
                for hidden_order, hidden_record in enumerate(hidden_index, start=1):
                    dataset = hidden_record["dataset"]
                    seen_by_dataset[dataset] += 1
                    group_index = seen_by_dataset[dataset]
                    input_phase = hidden_record.get("input_phase")
                    is_warmup = (input_phase == "warmup") if input_phase else (group_index <= args.warmup_per_dataset)
                    if not is_warmup:
                        formal_seen_by_dataset[dataset] += 1
                    formal_group_index = None if is_warmup else formal_seen_by_dataset[dataset]

                    load_start = time.perf_counter()
                    blob = load_hidden_blob(hidden_record["hidden_path"])
                    hidden_load_s = time.perf_counter() - load_start
                    sample = blob["sample"]
                    past_key_values = None
                    token_matches = []
                    prefill_metrics = None
                    decode_h2d = []
                    decode_forward = []
                    decode_lm_head = []
                    decode_sample = []
                    decode_actual_delay = []
                    sample_start = time.perf_counter()

                    with torch.inference_mode():
                        for step in blob["steps"]:
                            hidden_by_cut = step["hidden_by_cut"]
                            hidden_cpu = hidden_by_cut.get(cut_layer)
                            if hidden_cpu is None:
                                hidden_cpu = hidden_by_cut.get(str(cut_layer))
                            if hidden_cpu is None:
                                raise KeyError(
                                    f"Hidden cut layer {cut_layer} is not available in "
                                    f"{hidden_record['hidden_path']}"
                                )

                            scheduled_delay_s, actual_delay_s = maybe_delay(
                                step["stage"],
                                cut_layer,
                                delay_profile["delay_base_ms"],
                                delay_profile["delay_per_cut_ms"],
                            )
                            if args.backend == "dy":
                                result = run_dyforward_suffix_step(
                                    model,
                                    hidden_cpu,
                                    step["context_token_length"],
                                    past_key_values,
                                    cut_layer,
                                )
                            else:
                                result = run_official_suffix_step(
                                    model,
                                    hidden_cpu,
                                    step["context_token_length"],
                                    past_key_values,
                                )

                            past_key_values = result.pop("past_key_values")
                            next_token_match = result["generated_token_id"] == step["generated_token_id"]
                            token_matches.append(next_token_match)

                            if step["stage"] == "prefill":
                                prefill_metrics = dict(result)
                            else:
                                decode_h2d.append(result["hidden_to_device_s"])
                                decode_forward.append(result["suffix_forward_s"])
                                decode_lm_head.append(result["lm_head_s"])
                                decode_sample.append(result["sample_s"])
                                decode_actual_delay.append(actual_delay_s)

                            if not is_warmup:
                                event_row = {
                                    "run_id": run_id,
                                    "backend": args.backend,
                                    "run_label": run_label,
                                    "delay_profile": delay_profile["name"],
                                    "fixed_delay_ms": delay_profile["fixed_delay_ms"],
                                    "delay_base_ms": delay_profile["delay_base_ms"],
                                    "delay_per_cut_ms": delay_profile["delay_per_cut_ms"],
                                    "hidden_source_run_id": source_manifest.get("run_id"),
                                    "hidden_source_host": source_manifest.get("host", {}).get("hostname"),
                                    "cut_layer": cut_layer,
                                    "total_layers": total_layers,
                                    "suffix_layers": suffix_layers,
                                    "dataset": sample["dataset"],
                                    "record_id": sample["record_id"],
                                    "sample_index": sample["sample_index"],
                                    "group_index": group_index,
                                    "formal_group_index": formal_group_index,
                                    "input_token_count": sample["input_token_count"],
                                    "stage": step["stage"],
                                    "generated_token_index": step["generated_token_index"],
                                    "context_token_length": step["context_token_length"],
                                    "source_generated_token_id": step["generated_token_id"],
                                    "replay_generated_token_id": result["generated_token_id"],
                                    "next_token_match": next_token_match,
                                    "hidden_load_s": hidden_load_s,
                                    "scheduled_delay_s": scheduled_delay_s,
                                    "actual_delay_s": actual_delay_s,
                                }
                                event_row.update(result)
                                event_row.update(cuda_memory_snapshot("replay"))
                                event_writer.write(event_row)

                    sample_total_s = time.perf_counter() - sample_start
                    decode_count = len(decode_forward)
                    decode_forward_per_token_s = sum(decode_forward) / decode_count if decode_count else 0.0
                    match_rate = sum(token_matches) / len(token_matches) if token_matches else 0.0
                    if is_warmup:
                        print(
                            f"    warmup {hidden_order}/{len(hidden_index)} "
                            f"{dataset}#{sample['record_id']} cut={cut_layer} "
                            f"delay={delay_profile['fixed_delay_ms']}ms "
                            f"total={sample_total_s:.3f}s "
                            f"decode_forward={decode_forward_per_token_s * 1000:.2f}ms/token "
                            f"match={match_rate:.3f}",
                            flush=True,
                        )
                    else:
                        if decode_count == 0:
                            raise RuntimeError("No decode steps found in hidden blob.")
                        prefill_metrics = prefill_metrics or {}
                        sample_row = {
                            "run_id": run_id,
                            "backend": args.backend,
                            "run_label": run_label,
                            "delay_profile": delay_profile["name"],
                            "fixed_delay_ms": delay_profile["fixed_delay_ms"],
                            "delay_base_ms": delay_profile["delay_base_ms"],
                            "delay_per_cut_ms": delay_profile["delay_per_cut_ms"],
                            "hidden_source_run_id": source_manifest.get("run_id"),
                            "hidden_source_host": source_manifest.get("host", {}).get("hostname"),
                            "cut_layer": cut_layer,
                            "total_layers": total_layers,
                            "suffix_layers": suffix_layers,
                            "dataset": sample["dataset"],
                            "record_id": sample["record_id"],
                            "sample_index": sample["sample_index"],
                            "group_index": group_index,
                            "formal_group_index": formal_group_index,
                            "input_token_count": sample["input_token_count"],
                            "hidden_path": hidden_record["hidden_path"],
                            "hidden_load_s": hidden_load_s,
                            "generated_token_count": len(blob["steps"]),
                            "decode_token_count": decode_count,
                            "prefill_hidden_to_device_s": prefill_metrics.get("hidden_to_device_s", 0.0),
                            "prefill_suffix_forward_s": prefill_metrics.get("suffix_forward_s", 0.0),
                            "prefill_lm_head_s": prefill_metrics.get("lm_head_s", 0.0),
                            "prefill_sample_s": prefill_metrics.get("sample_s", 0.0),
                            "decode_actual_delay_total_s": sum(decode_actual_delay),
                            "decode_actual_delay_per_token_s": sum(decode_actual_delay) / decode_count,
                            "decode_hidden_to_device_total_s": sum(decode_h2d),
                            "decode_hidden_to_device_per_token_s": sum(decode_h2d) / decode_count,
                            "decode_suffix_forward_total_s": sum(decode_forward),
                            "decode_suffix_forward_per_token_s": decode_forward_per_token_s,
                            "decode_lm_head_total_s": sum(decode_lm_head),
                            "decode_lm_head_per_token_s": sum(decode_lm_head) / decode_count,
                            "decode_sample_total_s": sum(decode_sample),
                            "decode_sample_per_token_s": sum(decode_sample) / decode_count,
                            "sample_compute_total_s": (
                                prefill_metrics.get("hidden_to_device_s", 0.0)
                                + prefill_metrics.get("suffix_forward_s", 0.0)
                                + prefill_metrics.get("lm_head_s", 0.0)
                                + prefill_metrics.get("sample_s", 0.0)
                                + sum(decode_h2d)
                                + sum(decode_forward)
                                + sum(decode_lm_head)
                                + sum(decode_sample)
                            ),
                            "sample_replay_total_s": sample_total_s,
                            "next_token_match_rate": match_rate,
                        }
                        sample_row.update(cuda_memory_snapshot("replay"))
                        sample_row.update(collect_nvidia_smi())
                        sample_writer.write(sample_row)
                        sample_rows.append(sample_row)
                        print(
                            f"    sample {hidden_order}/{len(hidden_index)} "
                            f"{dataset}#{sample['record_id']} cut={cut_layer} "
                            f"delay={delay_profile['fixed_delay_ms']}ms "
                            f"total={sample_total_s:.3f}s "
                            f"decode_forward={decode_forward_per_token_s * 1000:.2f}ms/token "
                            f"match={match_rate:.3f}",
                            flush=True,
                        )

                    del blob, past_key_values
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            if args.backend == "official":
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                sync_device()

        summarize_replay(sample_rows, run_dir / "summary_by_cut.csv")
        status = "completed"
    except Exception as exc:
        status = "failed"
        error = exc
        raise
    finally:
        event_writer.close()
        sample_writer.close()
        cut_setup_writer.close()
        if dy_model is not None:
            del dy_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        write_json(
            run_dir / "manifest_replay.json",
            {
                "run_id": run_id,
                "status": status,
                "started_at": started_at,
                "finished_at": now_utc_iso(),
                "model_id": args.model_id,
                "backend": args.backend,
                "run_label": run_label,
                "torch_dtype": str(DTYPE),
                "device": str(DEVICE),
                "hidden_source_dir": str(source_dir),
                "hidden_source_manifest": str(source_manifest_path),
                "hidden_source_run_id": source_manifest.get("run_id"),
                "hidden_source_host": source_manifest.get("host"),
                "source_max_new_tokens": source_manifest.get("max_new_tokens"),
                "replay_dataset": args.dataset,
                "selected_hidden_sample_count": len(hidden_index),
                "max_hidden_samples": args.max_hidden_samples,
                "cut_layers": cut_layers,
                "warmup_per_dataset_per_cut": args.warmup_per_dataset,
                "delay_sweep": args.delay_sweep,
                "delay_sweep_step_ms": args.delay_sweep_step_ms,
                "delay_sweep_max_ms": args.delay_sweep_max_ms,
                "delay_profiles": delay_profiles,
                "delay_base_ms": args.delay_base_ms,
                "delay_per_cut_ms": args.delay_per_cut_ms,
                "results_dir": str(run_dir),
                "metrics_files": {
                    "suffix_forward_events": str(run_dir / "suffix_forward_events.jsonl"),
                    "suffix_forward_samples": str(run_dir / "suffix_forward_samples.jsonl"),
                    "cut_setup": str(run_dir / "cut_setup.jsonl"),
                    "summary_by_cut": str(run_dir / "summary_by_cut.csv"),
                },
                "host": host_info(),
                "runtime": runtime_versions(),
                "device_info": device_info(),
                "sample_count": len(sample_rows),
                "error": str(error) if error is not None else None,
            },
        )

    print("Done.")
    print(f"Metrics saved under: {run_dir}")


def main():
    args = parse_args()
    if args.command == "dump":
        dump_hidden(args)
    elif args.command == "replay":
        replay_hidden(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
