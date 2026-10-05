"""Prepare a small Qwen or Llama input file without loading model weights."""

import argparse
import json
from pathlib import Path

from grpc_ready.qwen_9b.qwen_inputs import validate_input_records


ROOT = Path(__file__).resolve().parent
MODELS = {"qwen": "Qwen/Qwen3.5-9B", "llama": "meta-llama/Llama-3.1-8B"}


def tokenize_prompt(tokenizer, prompt, model):
    if model == "qwen":
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    return tokenizer(prompt, add_special_tokens=False)["input_ids"]


def prepare_records(prompts, tokenizer, model, model_id, revision):
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("Prompts must be a non-empty JSON array.")
    records = []
    for item in prompts:
        if not isinstance(item, dict) or not isinstance(item.get("prompt"), str) or not item["prompt"].strip():
            raise ValueError("Each prompt entry needs a non-empty prompt string.")
        ids = tokenize_prompt(tokenizer, item["prompt"], model)
        records.append({
            "record_id": item.get("record_id"),
            "dataset": item.get("dataset", "custom"),
            "source": item.get("source", "user_provided"),
            "input_group": item.get("input_group"),
            "phase": item.get("phase", "formal"),
            "model_profile": model,
            "tokenizer": model_id,
            "tokenizer_revision": revision,
            "tokenizer_commit": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
            "tokenization_mode": "chat" if model == "qwen" else "plain",
            "qwen_thinking_mode": "disabled" if model == "qwen" else None,
            "token_count": len(ids),
            "input_ids": ids,
        })
    validate_input_records(records, model)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="qwen")
    parser.add_argument("--prompts", type=Path, default=ROOT / "inputs/example_prompts.json")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model-id", default=None, help="Tokenizer model ID or local path.")
    parser.add_argument("--revision", default="main", help="Tokenizer revision; use a commit for repeatability.")
    parser.add_argument("--force", action="store_true", help="Replace an existing output file.")
    args = parser.parse_args()
    output = (args.output or ROOT / "inputs/generated" / f"{args.model}.jsonl").expanduser()
    if output.exists() and not args.force:
        parser.error(f"Output exists: {output}. Use --force to replace it.")
    prompts = json.loads(args.prompts.expanduser().read_text(encoding="utf-8"))
    from transformers import AutoTokenizer

    model_id = args.model_id or MODELS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=args.revision, use_fast=True)
    records = prepare_records(prompts, tokenizer, args.model, model_id, args.revision)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w" if args.force else "x", encoding="utf-8") as target:
        for record in records:
            target.write(json.dumps(record) + "\n")
    formal, warmup = validate_input_records(records, args.model)
    print(f"Saved {len(formal)} formal + {len(warmup)} warmup inputs to {output}")
    for record in records:
        print(f"  {record['record_id']}: {record['phase']}, {record['token_count']} tokens")


if __name__ == "__main__":
    main()
