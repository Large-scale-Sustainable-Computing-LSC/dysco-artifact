"""Rebuild the exact prompt sets used in the paper from their public sources.

The prompts themselves are not redistributed here: both source datasets carry
their own licence terms. This script reconstructs them from the two manifests
in this directory, each of which records, for every prompt actually used, the
source dataset, split and row index, plus the tokenized length that row had in
our runs.

    pip install datasets
    python3 build_prompts.py                 # writes ../selected_testsets/<group>/raw_inputs.jsonl
    python3 build_prompts.py --check-only    # verify an existing rebuild

Sources
    wildgpt4    mlabonne/WildGPT-4, split "train"
                GPT-4 conversations extracted from allenai/WildChat-1M.
                The prompt is the first human turn of the conversation.
    longreason  lz1bytedance/LongReason, split "8k"
                The prompt is the row's "prompt" field.

Verification
    Lengths in the manifests were measured with the Llama-3.1-8B and
    Qwen3.5-9B tokenizers (no special tokens). After rebuilding, the script
    re-tokenizes every prompt and compares; any mismatch is reported and the
    affected rows are listed. A mismatch means the upstream dataset changed
    after our runs, in which case the results in results/ still stand but a
    re-run would not use byte-identical prompts.
"""
import argparse, csv, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.normpath(os.path.join(HERE, "..", "selected_testsets"))

GROUPS = {
    "wildgpt4": dict(repo="mlabonne/WildGPT-4", split="train",
                     extract=lambda row: row["conversations"][0]["value"]),
    "longreason": dict(repo="lz1bytedance/LongReason", split="8k",
                       extract=lambda row: row["prompt"]),
}
TOKENIZERS = {"llama_tokens": "meta-llama/Llama-3.1-8B", "qwen_tokens": "Qwen/Qwen3.5-9B"}


def manifest(group):
    with open(os.path.join(HERE, f"prompt_manifest_{group}.csv")) as fh:
        return list(csv.DictReader(fh))


def build(group, rows):
    from datasets import load_dataset
    spec = GROUPS[group]
    print(f"[{group}] loading {spec['repo']} split={spec['split']} ...", flush=True)
    ds = load_dataset(spec["repo"], split=spec["split"])
    out_dir = os.path.join(OUT_ROOT, group)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "raw_inputs.jsonl")
    with open(out_path, "w") as fh:
        for r in rows:
            text = spec["extract"](ds[int(r["source_row"])])
            fh.write(json.dumps({
                "record_id": int(r["order"]), "dataset": group, "source": r["source_dataset"],
                "source_split": r["source_split"], "source_row": int(r["source_row"]),
                "llama_length": int(r["llama_tokens"]), "qwen_length": int(r["qwen_tokens"]),
                "input_text": text,
            }) + "\n")
    print(f"[{group}] wrote {len(rows)} prompts -> {out_path}")
    return out_path


def check(group, rows, path, tokenizer_key="llama_tokens"):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZERS[tokenizer_key])
    built = [json.loads(l) for l in open(path) if l.strip()]
    if len(built) != len(rows):
        print(f"[{group}] FAIL: {len(built)} prompts rebuilt, {len(rows)} expected")
        return False
    bad = []
    for r, b in zip(rows, built):
        n = len(tok(b["input_text"], add_special_tokens=False).input_ids)
        if n != int(r[tokenizer_key]):
            bad.append((int(r["order"]), int(r[tokenizer_key]), n))
    if bad:
        print(f"[{group}] FAIL: {len(bad)} of {len(rows)} prompts differ in length; first 5: {bad[:5]}")
        return False
    print(f"[{group}] OK: all {len(rows)} prompts match the manifest lengths")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="wildgpt4,longreason")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args()

    ok = True
    for group in args.groups.split(","):
        rows = manifest(group)
        path = os.path.join(OUT_ROOT, group, "raw_inputs.jsonl")
        if not args.check_only:
            path = build(group, rows)
        elif not os.path.exists(path):
            sys.exit(f"[{group}] nothing to check at {path}; run without --check-only first")
        if not args.no_check:
            ok &= check(group, rows, path)
    sys.exit(0 if ok else 1)
