import argparse
import gc
import os
import shutil
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def default_local_base_dir():
    configured_base = os.environ.get("QWEN_LOCAL_BASE")
    if configured_base:
        return Path(configured_base).expanduser()

    username = os.environ.get("USER") or os.environ.get("LOGNAME") or "user"
    local_user_dir = Path("/local") / username
    if Path("/local").exists():
        return local_user_dir

    for env_name in ("SLURM_TMPDIR", "TMPDIR", "LOCAL_SCRATCH", "TMP"):
        base_dir = os.environ.get(env_name)
        if base_dir:
            return Path(base_dir).expanduser()
    return Path("/tmp") / username


if "HF_HOME" not in os.environ and "HF_HUB_CACHE" not in os.environ:
    os.environ["HF_HOME"] = str(default_local_base_dir() / "hf_home")


DTYPE_CHOICES = ("auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export a small text-only Qwen3.5-9B prefix model for the edge device. "
            "The output contains the token embedding, the first N text layers, "
            "the final norm, config, and tokenizer."
        )
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3.5-9B",
        help="Hugging Face model id or local Qwen3.5-9B model directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for the exported edge model. If omitted, the script writes "
            "under the node-local model directory, e.g. "
            "/local/$USER/models/qwen35_9b_edge_f16_bf16."
        ),
    )
    parser.add_argument(
        "--keep-layers",
        type=int,
        default=16,
        help="Number of initial Qwen text layers to keep for the edge model.",
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=sorted(DTYPE_CHOICES),
        help="Dtype used when loading and saving the exported edge model.",
    )
    parser.add_argument(
        "--max-shard-size",
        default="5GB",
        help="Shard size passed to save_pretrained.",
    )
    parser.add_argument(
        "--safe-serialization",
        dest="safe_serialization",
        action="store_true",
        default=True,
        help="Save as safetensors instead of PyTorch bin files.",
    )
    parser.add_argument(
        "--no-safe-serialization",
        dest="safe_serialization",
        action="store_false",
        help="Save as PyTorch bin files instead of safetensors.",
    )
    return parser.parse_args()


def resolve_dtype(dtype_name):
    import torch

    return {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }[dtype_name]


def resolve_output_dir(output_dir, model_id, keep_layers, dtype_name):
    if output_dir:
        expanded = os.path.expandvars(output_dir)
        if "$" in expanded:
            raise ValueError(
                f"Output path still contains an unresolved environment variable: {expanded}"
            )
        return Path(expanded).expanduser().resolve()

    model_name = Path(model_id.rstrip("/")).name.lower()
    if model_name in {"qwen3.5-9b", "qwen3_5_9b"}:
        dirname = f"qwen35_9b_edge_f{keep_layers}_{dtype_name}"
    else:
        dirname_model = model_name.replace(".", "_").replace("-", "_")
        dirname = f"{dirname_model}_edge_f{keep_layers}_{dtype_name}"

    return (default_local_base_dir() / "models" / dirname).resolve()


def print_disk_space(path):
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    total, used, free = shutil.disk_usage(probe)
    gib = 1024 ** 3
    print(f"Disk space at {probe}: free={free / gib:.2f} GiB, total={total / gib:.2f} GiB")


def slice_layer_types(config, keep_layers):
    layer_types = getattr(config, "layer_types", None)
    if layer_types is not None:
        config.layer_types = list(layer_types[:keep_layers])
    config.num_hidden_layers = keep_layers


def keep_prefix_layers(text_model, keep_layers):
    total_layers = len(text_model.layers)
    if keep_layers <= 0 or keep_layers > total_layers:
        raise ValueError(f"--keep-layers must be in [1, {total_layers}], got {keep_layers}")

    text_model.layers = text_model.layers[:keep_layers]
    slice_layer_types(text_model.config, keep_layers)
    return total_layers


def main():
    args = parse_args()
    dtype = resolve_dtype(args.dtype)
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM

    output_dir = resolve_output_dir(
        output_dir=args.output_dir,
        model_id=args.model_id,
        keep_layers=args.keep_layers,
        dtype_name=args.dtype,
    )

    print(f"Loading Qwen model from: {args.model_id}")
    print(f"Requested dtype: {args.dtype}")
    print(f"Exporting first {args.keep_layers} text layers")
    print(f"Output directory: {output_dir}")
    print(f"HF_HOME: {os.environ.get('HF_HOME', '<not set>')}")
    print_disk_space(output_dir)

    full_model = Qwen3_5ForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    text_model = full_model.model
    total_layers = keep_prefix_layers(text_model, args.keep_layers)

    output_dir.mkdir(parents=True, exist_ok=True)
    text_model.save_pretrained(
        output_dir,
        safe_serialization=args.safe_serialization,
        max_shard_size=args.max_shard_size,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    saved_dtype = next(text_model.parameters()).dtype
    del full_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Done.")
    print(f"Original text layers: {total_layers}")
    print(f"Saved edge layers: {args.keep_layers}")
    print(f"Saved dtype: {saved_dtype}")
    print(f"Saved model path: {output_dir}")
    print_disk_space(output_dir)


if __name__ == "__main__":
    main()
