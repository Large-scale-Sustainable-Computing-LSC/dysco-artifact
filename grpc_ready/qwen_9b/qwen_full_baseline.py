#!/usr/bin/env python3
"""Single-device full-model baseline for Qwen3.5-9B or Llama-3.1-8B.

This script intentionally has fixed experiment settings so it can be copied to
an A6000 or A5000 node and run directly with:

    python qwen_full_baseline.py

Use Llama-3.1-8B with:

    python qwen_full_baseline.py --model llama

Pass --input-file to use a small prepared JSONL with explicit warmup records
instead of the fixed selected-testset counts.
"""

import argparse
import csv
from collections import defaultdict, deque
import gc
import json
import math
import os
from pathlib import Path
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from qwen_inputs import load_input_file

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
    tensor_raw_nbytes,
    write_json,
)


SCRIPT_DIR = Path(__file__).resolve().parent

MODEL_PROFILES = {
    "qwen": {
        "display_name": "Qwen3.5-9B",
        "model_id": "Qwen/Qwen3.5-9B",
        "tokenized_filename": "qwen_non_thinking.jsonl",
        "run_prefix": "qwen9b_full_baseline",
    },
    "llama": {
        "display_name": "Llama-3.1-8B",
        "model_id": "meta-llama/Llama-3.1-8B",
        "tokenized_filename": "llama_tokenized.jsonl",
        "run_prefix": "llama31_8b_full_baseline",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a single-device full-model baseline for Qwen or Llama."
    )
    parser.add_argument("--input-file", default=None, help="Prepared JSONL instead of fixed testset counts.")
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_PROFILES),
        default=os.environ.get("BASELINE_MODEL", "qwen"),
        help="Model profile to run. Defaults to qwen.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Override the Hugging Face model id for this run.",
    )
    parser.add_argument(
        "--tokenized-filename",
        default=None,
        help="Override the selected-testset tokenized filename.",
    )
    parser.add_argument(
        "--metrics-root",
        default=None,
        help="Override the output metrics root. Defaults to ./results_baseline.",
    )
    return parser.parse_args()


ARGS = parse_args()
MODEL_PROFILE = MODEL_PROFILES[ARGS.model]
INPUT_FILE = Path(ARGS.input_file).expanduser().resolve() if ARGS.input_file else None
FILE_RECORDS = load_input_file(INPUT_FILE, ARGS.model) if INPUT_FILE else None


def env_first(*names):
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


# Fixed baseline settings.
if ARGS.model_id:
    MODEL_ID = ARGS.model_id
elif ARGS.model == "qwen":
    MODEL_ID = env_first("QWEN_MODEL_ID", "BASELINE_MODEL_ID") or MODEL_PROFILE["model_id"]
else:
    MODEL_ID = env_first("LLAMA_MODEL_ID", "BASELINE_MODEL_ID") or MODEL_PROFILE["model_id"]

DTYPE = torch.bfloat16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_NEW_TOKENS = 64
WARMUP_MAX_NEW_TOKENS = 8
TEST_LEVEL = 3
WILDGPT4_COUNT = 8 * TEST_LEVEL
LONGREASON_COUNT = TEST_LEVEL
WARMUP_GPT4_COUNT = 4
WARMUP_LONGREASON_COUNT = 1
if ARGS.tokenized_filename:
    TOKENIZED_FILENAME = ARGS.tokenized_filename
elif ARGS.model == "qwen":
    TOKENIZED_FILENAME = (
        env_first("QWEN_TOKENIZED_FILENAME", "BASELINE_TOKENIZED_FILENAME")
        or MODEL_PROFILE["tokenized_filename"]
    )
else:
    TOKENIZED_FILENAME = (
        env_first("LLAMA_TOKENIZED_FILENAME", "BASELINE_TOKENIZED_FILENAME")
        or MODEL_PROFILE["tokenized_filename"]
    )
DO_SAMPLE = False
TEMPERATURE = 1.0
TOP_P = 1.0
TOP_K = 0
RESOURCE_SAMPLE_INTERVAL_S = float(os.environ.get("QWEN_RESOURCE_SAMPLE_INTERVAL_S", "0.5"))

RUN_ID = (
    os.environ.get("QWEN_RUN_ID")
    or os.environ.get("BASELINE_RUN_ID")
    or make_run_id(MODEL_PROFILE["run_prefix"])
)
RUN_STARTED_AT = now_utc_iso()
METRICS_ROOT = Path(
    ARGS.metrics_root
    or env_first("QWEN_BASELINE_METRICS_ROOT", "BASELINE_METRICS_ROOT")
    or SCRIPT_DIR / "results_baseline"
).expanduser()
RUN_DIR = METRICS_ROOT / RUN_ID
BASELINE_METRICS_DIR = RUN_DIR / "baseline_metrics"


def sync_device():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


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


SELECTED_TESTSET_ROOT = resolve_selected_testset_root()


class BaselineMetrics:
    def __init__(self):
        BASELINE_METRICS_DIR.mkdir(parents=True, exist_ok=True)
        self.outputs = JsonlWriter(BASELINE_METRICS_DIR / "outputs_baseline.jsonl")
        self.latency = JsonlWriter(BASELINE_METRICS_DIR / "latency_events_baseline.jsonl")
        self.context = MetricsContext(
            {
                "run_id": RUN_ID,
                "role": "baseline",
                "state": "starting",
                "phase": "setup",
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
            BASELINE_METRICS_DIR / "resource_baseline.jsonl",
            role="baseline",
            context=self.context,
            interval_s=RESOURCE_SAMPLE_INTERVAL_S,
            include_tegrastats=False,
            include_nvidia_smi=True,
        )

    def start(self):
        self.resource_sampler.start()

    def set_context(self, **context):
        base = {
            "run_id": RUN_ID,
            "role": "baseline",
            "state": context.pop("state", "running"),
        }
        base.update(context)
        self.context.set(base)

    def write_latency(self, row):
        self.latency.write(row)

    def write_output(self, row):
        self.outputs.write(row)

    def write_manifest(self, manifest):
        write_json(BASELINE_METRICS_DIR / "manifest_baseline.json", manifest)

    def close(self):
        self.resource_sampler.stop()
        self.outputs.close()
        self.latency.close()


def input_group_for_record(record):
    if record.get("input_group") in {"short", "long"}:
        return record["input_group"]
    dataset = str(record.get("dataset", "")).lower()
    if "long" in dataset or len(record.get("input_ids", [])) >= 1024:
        return "long"
    return "short"


def record_summary(record):
    return {
        "dataset": record.get("dataset"),
        "input_group": input_group_for_record(record),
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


def load_tokenized_records(group_name, count, from_tail=False):
    if count <= 0:
        return []

    path = SELECTED_TESTSET_ROOT / group_name / TOKENIZED_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Tokenized testset file not found: {path}")

    records = deque(maxlen=count) if from_tail else []
    with path.open() as jsonl_file:
        for line in jsonl_file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["input_ids"] = [int(token_id) for token_id in record["input_ids"]]
            records.append(record)
            if not from_tail and len(records) >= count:
                break

    records = list(records)
    if len(records) < count:
        raise ValueError(f"Only found {len(records)} records in {path}, requested {count}.")
    return records


def load_formal_records():
    if FILE_RECORDS is not None:
        return FILE_RECORDS[0]
    records = []
    records.extend(load_tokenized_records("wildgpt4", WILDGPT4_COUNT))
    records.extend(load_tokenized_records("longreason", LONGREASON_COUNT))
    return records


def load_warmup_records():
    if FILE_RECORDS is not None:
        return FILE_RECORDS[1]
    records = []
    records.extend(load_tokenized_records("wildgpt4", WARMUP_GPT4_COUNT, from_tail=True))
    records.extend(load_tokenized_records("longreason", WARMUP_LONGREASON_COUNT, from_tail=True))
    return records


def build_sample_trace(record, phase, sample_index, warmup_round=None):
    return {
        "run_id": RUN_ID,
        "phase": phase,
        "warmup_round": warmup_round,
        "baseline_policy": "single_device_full_model",
        "cut_layer": 0,
        "dataset": record.get("dataset"),
        "input_group": input_group_for_record(record),
        "record_id": record.get("record_id"),
        "sample_index": sample_index,
        "source": record.get("source"),
        "source_label": record.get("source_label"),
        "source_split": record.get("source_split"),
        "source_row": record.get("source_row"),
        "source_sample_index": record.get("sample_index"),
        "input_token_count": len(record["input_ids"]),
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


def sample_next_token(logits):
    return logits.argmax(dim=-1, keepdim=True)


def clear_cuda_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_text_model(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model
    raise AttributeError(f"Could not find text-model layers in {type(model).__name__}.")


def write_latency_event(metrics, trace, timing, next_token):
    row = dict(trace)
    row.update(timing)
    row.update(
        {
            "generated_token_id": int(next_token.item()),
            "timeline_serial": True,
            "baseline_infer_s": (
                timing["baseline_forward_s"]
                + timing["baseline_lm_head_s"]
                + timing["baseline_token_selection_s"]
            ),
            "baseline_non_forward_s": timing["event_total_s"] - timing["baseline_forward_s"],
        }
    )
    row.update(cuda_memory_snapshot("baseline"))
    metrics.write_latency(row)


def run_text_model_step(text_model, lm_head, input_ids, attention_mask, position_ids, past_key_values):
    sync_device()
    forward_start = time.perf_counter()
    outputs = text_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    sync_device()
    baseline_forward_s = time.perf_counter() - forward_start

    lm_head_start = time.perf_counter()
    logits = lm_head(outputs.last_hidden_state[:, -1:, :])
    sync_device()
    baseline_lm_head_s = time.perf_counter() - lm_head_start

    selection_start = time.perf_counter()
    next_token = sample_next_token(logits[:, -1, :])
    sync_device()
    baseline_token_selection_s = time.perf_counter() - selection_start

    return outputs, next_token, {
        "baseline_forward_s": baseline_forward_s,
        "baseline_lm_head_s": baseline_lm_head_s,
        "baseline_token_selection_s": baseline_token_selection_s,
    }


def run_generation_from_input_ids(
    text_model,
    lm_head,
    tokenizer,
    input_ids_list,
    metrics,
    sample_trace,
    max_new_tokens,
):
    clear_cuda_cache()
    input_ids = torch.tensor([input_ids_list], device=DEVICE, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, device=DEVICE, dtype=torch.long)
    position_ids = attention_mask.long().cumsum(dim=1) - 1
    generated_token_ids = []
    eos_seen = False
    past_key_values = None

    sample_start_unix = time.time()
    sample_start_perf = time.perf_counter()
    generation_start_perf = time.perf_counter()

    with torch.inference_mode():
        event_trace = build_event_trace(sample_trace, "prefill", 1, input_ids.shape[1])
        metrics.set_context(state=sample_trace.get("phase", "running"), **event_trace)
        reset_cuda_peak_memory(DEVICE)
        event_start_unix = time.time()
        event_start_perf = time.perf_counter()
        outputs, next_token, timing = run_text_model_step(
            text_model,
            lm_head,
            input_ids,
            attention_mask,
            position_ids,
            None,
        )
        event_end_unix = time.time()
        event_end_perf = time.perf_counter()
        past_key_values = outputs.past_key_values
        timing.update(
            {
                "event_start_unix": event_start_unix,
                "event_end_unix": event_end_unix,
                "event_total_s": event_end_perf - event_start_perf,
                "input_ids_shape": list(input_ids.shape),
                "attention_mask_shape": list(attention_mask.shape),
                "position_ids_shape": list(position_ids.shape),
                "input_ids_raw_bytes": tensor_raw_nbytes(input_ids),
                "attention_mask_raw_bytes": tensor_raw_nbytes(attention_mask),
                "position_ids_raw_bytes": tensor_raw_nbytes(position_ids),
            }
        )
        write_latency_event(metrics, event_trace, timing, next_token)

        generated_token_ids.append(int(next_token.item()))
        if tokenizer.eos_token_id is not None and int(next_token.item()) == tokenizer.eos_token_id:
            eos_seen = True
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)

        for generated_token_index in range(2, max_new_tokens + 1):
            context_token_length = input_ids.shape[1] + len(generated_token_ids)
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
            metrics.set_context(state=sample_trace.get("phase", "running"), **event_trace)
            reset_cuda_peak_memory(DEVICE)
            event_start_unix = time.time()
            event_start_perf = time.perf_counter()
            outputs, next_token, timing = run_text_model_step(
                text_model,
                lm_head,
                next_token,
                attention_mask,
                position_ids,
                past_key_values,
            )
            event_end_unix = time.time()
            event_end_perf = time.perf_counter()
            past_key_values = outputs.past_key_values
            timing.update(
                {
                    "event_start_unix": event_start_unix,
                    "event_end_unix": event_end_unix,
                    "event_total_s": event_end_perf - event_start_perf,
                    "input_ids_shape": list(next_token.shape),
                    "attention_mask_shape": list(attention_mask.shape),
                    "position_ids_shape": list(position_ids.shape),
                    "input_ids_raw_bytes": tensor_raw_nbytes(next_token),
                    "attention_mask_raw_bytes": tensor_raw_nbytes(attention_mask),
                    "position_ids_raw_bytes": tensor_raw_nbytes(position_ids),
                }
            )
            write_latency_event(metrics, event_trace, timing, next_token)

            generated_token_ids.append(int(next_token.item()))
            if tokenizer.eos_token_id is not None and int(next_token.item()) == tokenizer.eos_token_id:
                eos_seen = True
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)

    sample_end_unix = time.time()
    sample_end_perf = time.perf_counter()
    generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    metrics.set_context(
        state=sample_trace.get("phase", "running"),
        **build_event_trace(sample_trace, "between_samples", max_new_tokens, input_ids.shape[1] + max_new_tokens),
    )
    return {
        "status": "ok",
        "output_text": generated_text,
        "output_ids": generated_token_ids,
        "generated_token_count": len(generated_token_ids),
        "eos_seen": eos_seen,
        "sample_start_unix": sample_start_unix,
        "sample_end_unix": sample_end_unix,
        "sample_total_s": sample_end_perf - sample_start_perf,
        "generation_total_s": sample_end_perf - generation_start_perf,
    }


def is_cuda_oom(exc):
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower() and "cuda" in str(exc).lower()


def run_records(text_model, lm_head, tokenizer, records, phase, metrics, max_new_tokens, warmup_round=None):
    outputs = []
    for sample_index, record in enumerate(records, start=1):
        label = f"{record.get('dataset')}#{record.get('record_id')}"
        token_count = len(record["input_ids"])
        sample_trace = build_sample_trace(record, phase, sample_index, warmup_round=warmup_round)
        print(f"\n======== Baseline {phase} sample {sample_index}/{len(records)}: {label} ========")
        print(f"Input tokens: {token_count}, max_new_tokens={max_new_tokens}")
        try:
            result = run_generation_from_input_ids(
                text_model,
                lm_head,
                tokenizer,
                record["input_ids"],
                metrics,
                sample_trace,
                max_new_tokens=max_new_tokens,
            )
        except Exception as exc:
            clear_cuda_cache()
            output_row = dict(sample_trace)
            output_row.update(record_summary(record))
            output_row.update(
                {
                    "status": "cuda_oom" if is_cuda_oom(exc) else "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "max_new_tokens": max_new_tokens,
                    "fixed_new_tokens": True,
                    "early_stop_enabled": False,
                }
            )
            metrics.write_output(output_row)
            outputs.append(output_row)
            print(f"Failed: {type(exc).__name__}: {exc}")
            if not is_cuda_oom(exc):
                raise
            continue

        output_row = dict(sample_trace)
        output_row.update(record_summary(record))
        output_row.update(result)
        output_row["max_new_tokens"] = max_new_tokens
        output_row["fixed_new_tokens"] = True
        output_row["early_stop_enabled"] = False
        metrics.write_output(output_row)
        outputs.append(output_row)
        print(f"Generated exactly {result['generated_token_count']} tokens in {result['sample_total_s']:.4f} s")
    return outputs


def percentile(values, q):
    values = sorted(float(v) for v in values)
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * (q / 100.0)
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - position) + values[hi] * (position - lo)


def write_summary_csv(event_rows, output_rows):
    successful_samples = {
        (
            row.get("phase"),
            row.get("dataset"),
            row.get("record_id"),
            row.get("sample_index"),
        )
        for row in output_rows
        if row.get("phase") == "formal" and row.get("status", "ok") == "ok"
    }

    by_stage = defaultdict(list)
    for row in event_rows:
        if row.get("phase") != "formal":
            continue
        sample_key = (
            row.get("phase"),
            row.get("dataset"),
            row.get("record_id"),
            row.get("sample_index"),
        )
        if sample_key not in successful_samples:
            continue
        key = (row.get("input_group"), row.get("stage"))
        by_stage[key].append(row)

    summary_path = BASELINE_METRICS_DIR / "baseline_summary_by_stage.csv"
    with summary_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "input_group",
                "stage",
                "event_count",
                "mean_event_total_s",
                "p90_event_total_s",
                "mean_forward_s",
                "p90_forward_s",
                "mean_lm_head_s",
                "mean_token_selection_s",
                "mean_infer_s",
            ],
        )
        writer.writeheader()
        for (input_group, stage), rows in sorted(by_stage.items()):
            event_total = [row["event_total_s"] for row in rows]
            forward = [row["baseline_forward_s"] for row in rows]
            lm_head = [row["baseline_lm_head_s"] for row in rows]
            token_selection = [row["baseline_token_selection_s"] for row in rows]
            infer = [row["baseline_infer_s"] for row in rows]
            writer.writerow(
                {
                    "input_group": input_group,
                    "stage": stage,
                    "event_count": len(rows),
                    "mean_event_total_s": statistics.mean(event_total),
                    "p90_event_total_s": percentile(event_total, 90),
                    "mean_forward_s": statistics.mean(forward),
                    "p90_forward_s": percentile(forward, 90),
                    "mean_lm_head_s": statistics.mean(lm_head),
                    "mean_token_selection_s": statistics.mean(token_selection),
                    "mean_infer_s": statistics.mean(infer),
                }
            )

    by_sample = defaultdict(list)
    for row in output_rows:
        if row.get("phase") == "formal" and row.get("status", "ok") == "ok":
            by_sample[row.get("input_group")].append(row)

    sample_path = BASELINE_METRICS_DIR / "baseline_summary_by_sample.csv"
    with sample_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "input_group",
                "sample_count",
                "mean_sample_total_s",
                "p90_sample_total_s",
                "mean_generation_total_s",
                "p90_generation_total_s",
            ],
        )
        writer.writeheader()
        for input_group, rows in sorted(by_sample.items()):
            sample_total = [row["sample_total_s"] for row in rows]
            generation_total = [row["generation_total_s"] for row in rows]
            writer.writerow(
                {
                    "input_group": input_group,
                    "sample_count": len(rows),
                    "mean_sample_total_s": statistics.mean(sample_total),
                    "p90_sample_total_s": percentile(sample_total, 90),
                    "mean_generation_total_s": statistics.mean(generation_total),
                    "p90_generation_total_s": percentile(generation_total, 90),
                }
            )

    return summary_path, sample_path


def build_manifest(status, model_load_s=None, total_layers=None, formal_records=None, warmup_records=None, error=None):
    manifest = {
        "run_id": RUN_ID,
        "role": "baseline",
        "status": status,
        "started_at": RUN_STARTED_AT,
        "finished_at": now_utc_iso() if status in {"complete", "failed"} else None,
        "metrics_root": str(METRICS_ROOT),
        "run_dir": str(RUN_DIR),
        "metrics_dir": str(BASELINE_METRICS_DIR),
        "metrics_files": {
            "outputs": str(BASELINE_METRICS_DIR / "outputs_baseline.jsonl"),
            "latency_events": str(BASELINE_METRICS_DIR / "latency_events_baseline.jsonl"),
            "resource": str(BASELINE_METRICS_DIR / "resource_baseline.jsonl"),
            "manifest": str(BASELINE_METRICS_DIR / "manifest_baseline.json"),
            "summary_by_stage": str(BASELINE_METRICS_DIR / "baseline_summary_by_stage.csv"),
            "summary_by_sample": str(BASELINE_METRICS_DIR / "baseline_summary_by_sample.csv"),
        },
        "host": host_info(),
        "runtime": runtime_versions(),
        "experiment": {
            "baseline_policy": "single_device_full_model",
            "model_profile": ARGS.model,
            "model_display_name": MODEL_PROFILE["display_name"],
            "model_id": MODEL_ID,
            "torch_dtype": str(DTYPE),
            "device": str(DEVICE),
            "total_layers": total_layers,
            "max_new_tokens": MAX_NEW_TOKENS,
            "warmup_max_new_tokens": WARMUP_MAX_NEW_TOKENS,
            "fixed_new_tokens": True,
            "early_stop_enabled": False,
            "do_sample": DO_SAMPLE,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "test_level": TEST_LEVEL,
            "wildgpt4_count": WILDGPT4_COUNT,
            "longreason_count": LONGREASON_COUNT,
            "warmup_gpt4_tail_count": WARMUP_GPT4_COUNT,
            "warmup_longreason_tail_count": WARMUP_LONGREASON_COUNT,
        },
        "resource_sampling": {
            "interval_s": RESOURCE_SAMPLE_INTERVAL_S,
            "sources": ["torch.cuda", "psutil", "nvidia-smi"],
        },
        "selected_testset": {
            "root": str(SELECTED_TESTSET_ROOT),
            "tokenized_filename": TOKENIZED_FILENAME,
            "formal_records": [record_summary(record) for record in formal_records or []],
            "warmup_records": [record_summary(record) for record in warmup_records or []],
        },
        "error": str(error) if error is not None else None,
    }
    if INPUT_FILE:
        manifest["selected_testset"].pop("root")
        manifest["selected_testset"].pop("tokenized_filename")
        manifest["selected_testset"].update(mode="input_file", input_file=str(INPUT_FILE))
        for key in ("test_level", "wildgpt4_count", "longreason_count",
                    "warmup_gpt4_tail_count", "warmup_longreason_tail_count"):
            manifest["experiment"].pop(key)
    return manifest


def main():
    if DEVICE.type != "cuda":
        raise RuntimeError("CUDA is required for this baseline. Current device is CPU.")

    metrics = BaselineMetrics()
    metrics.start()
    model_load_s = None
    total_layers = None
    formal_records = []
    warmup_records = []
    event_rows = []
    output_rows = []
    status = "running"
    error = None

    original_write_latency = metrics.write_latency

    def capture_latency(row):
        event_rows.append(dict(row))
        original_write_latency(row)

    metrics.write_latency = capture_latency

    try:
        print(f"Run ID: {RUN_ID}")
        print(f"Baseline metrics dir: {BASELINE_METRICS_DIR}")
        print(f"Input file: {INPUT_FILE}" if INPUT_FILE else f"Selected testset root: {SELECTED_TESTSET_ROOT}")
        print(f"Model profile: {ARGS.model} ({MODEL_PROFILE['display_name']})")
        if INPUT_FILE is None:
            print(f"Tokenized file: {TOKENIZED_FILENAME}")
        print(f"Loading full model: {MODEL_ID}")
        print(f"Using device: {DEVICE}, dtype: {DTYPE}")

        metrics.set_context(
            state="load_model",
            phase="setup",
            dataset=None,
            record_id=None,
            sample_index=None,
            stage="load_model",
            generated_token_index=None,
            input_token_count=None,
            context_token_length=None,
            event_id=None,
        )
        load_start = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model.eval()
        model.to(DEVICE)
        sync_device()
        model_load_s = time.perf_counter() - load_start
        text_model = get_text_model(model)
        total_layers = len(text_model.layers)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        formal_records = load_formal_records()
        warmup_records = load_warmup_records()
        metrics.write_manifest(
            build_manifest(
                "running",
                model_load_s=model_load_s,
                total_layers=total_layers,
                formal_records=formal_records,
                warmup_records=warmup_records,
            )
        )
        print(f"Full model ready in {model_load_s:.2f} s, layers={total_layers}")
        print(
            f"Formal records: {len(formal_records)} "
            f"max_new_tokens={MAX_NEW_TOKENS}"
        )
        print(
            f"Warmup records: {len(warmup_records)} "
            f"max_new_tokens={WARMUP_MAX_NEW_TOKENS}"
        )

        output_rows.extend(
            run_records(
                text_model,
                model.lm_head,
                tokenizer,
                warmup_records,
                "warmup",
                metrics,
                max_new_tokens=WARMUP_MAX_NEW_TOKENS,
                warmup_round=1,
            )
        )
        output_rows.extend(
            run_records(
                text_model,
                model.lm_head,
                tokenizer,
                formal_records,
                "formal",
                metrics,
                max_new_tokens=MAX_NEW_TOKENS,
            )
        )
        write_summary_csv(event_rows, output_rows)
        status = "complete"
    except Exception as exc:
        status = "failed"
        error = exc
        raise
    finally:
        metrics.set_context(
            state=status,
            phase="finished",
            dataset=None,
            record_id=None,
            sample_index=None,
            stage=status,
            generated_token_index=None,
            input_token_count=None,
            context_token_length=None,
            event_id=None,
        )
        try:
            metrics.write_manifest(
                build_manifest(
                    status,
                    model_load_s=model_load_s,
                    total_layers=total_layers,
                    formal_records=formal_records,
                    warmup_records=warmup_records,
                    error=error,
                )
            )
        finally:
            metrics.close()

    print(f"\nDone. Metrics saved under: {RUN_DIR}")


if __name__ == "__main__":
    main()
