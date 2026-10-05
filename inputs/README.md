# Input Preparation

`example_prompts.json` contains a small set of self-authored prompts for smoke
testing. It includes both formal and warmup records. These examples are not part
of the paper evaluation workload.

## Prepare Demo Inputs

From the repository root:

```bash
python prepare_inputs.py --model qwen
python prepare_inputs.py --model llama
```

Only the tokenizer is loaded. Outputs are written to:

```text
inputs/generated/qwen.jsonl
inputs/generated/llama.jsonl
```

Use `--force` to overwrite an existing output. Custom prompts can be supplied
with `--prompts`, and `--model-id` / `--revision` can be used to select a specific
tokenizer checkpoint.

Qwen preparation uses the chat template with thinking disabled. Llama preparation
uses plain text tokenization with `add_special_tokens=False`. Token IDs should not
be reused across model families.

## Input Format

The source JSON is an array of objects. The main fields are:

- `record_id`: unique identifier;
- `prompt`: input text;
- `input_group`: `short` or `long`;
- `phase`: `formal` or `warmup`;
- `dataset` and `source`: optional labels.

Prepared JSONL additionally stores the model profile, token IDs, token count, and
tokenizer metadata. At least one formal record is required.

## Using Prepared Inputs

The generated JSONL can be passed to the edge runtime or full-model baseline:

```bash
python grpc_ready/qwen_9b/qwen_edge.py \
  --model qwen \
  --edge-full-model \
  --edge-layers 1 \
  --input-file inputs/generated/qwen.jsonl
```

```bash
python grpc_ready/qwen_9b/qwen_full_baseline.py \
  --model qwen \
  --input-file inputs/generated/qwen.jsonl
```

It can also be used for Qwen hidden-state generation:

```bash
python grpc_ready/qwen_9b/qwen_remote_hidden_replay.py dump \
  --input-file inputs/generated/qwen.jsonl \
  --cut-layers 1,4 \
  --max-new-tokens 8 \
  --output-root /tmp/edge-cloud-hidden
```

## Paper Workloads

The paper experiments use larger selected workloads derived from
[WildGPT-4](https://huggingface.co/datasets/mlabonne/WildGPT-4) and
[LongReason](https://huggingface.co/datasets/lz1bytedance/LongReason).
The existing workflow expects them under `wildgpt4/` and `longreason/`
subdirectories of `QWEN_SELECTED_TESTSET_ROOT`, with model-specific prepared
files named `qwen_non_thinking.jsonl` and `llama_tokenized.jsonl`.

The full datasets and generated paper-workload files are not distributed in this
repository.
