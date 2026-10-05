# Edge-Cloud Split Inference

This repository contains code for edge-cloud LLM inference with Qwen3.5-9B and
Llama-3.1-8B. The included implementation covers single-edge, single-cloud
execution.

## Scope

Included in this repository:

- layer-wise split inference between an edge device and a cloud server;
- dynamic forward execution over selected decoder-layer ranges;
- Qwen prefix-shard export for memory-constrained edge devices;
- Qwen hidden-state dump and suffix replay with optional delay injection;
- single-device full-model baselines for Qwen and Llama;
- small self-authored inputs and input-preparation utilities.

The multi-edge / depth-synchronized batching (DSB) implementation is not included
in this repository. The `multi_edge/` directory is kept as a placeholder
and documents this limitation.

## Repository Layout

```text
.
|-- README.md
|-- requirements.txt
|-- prepare_inputs.py
|-- dyForward/
|   `-- dynamic_forward.py
|-- grpc_ready/
|   |-- llama_ready.proto
|   |-- llama_ready_pb2.py
|   |-- llama_ready_pb2_grpc.py
|   `-- qwen_9b/
|       |-- qwen_edge.py
|       |-- qwen_cloud.py
|       |-- qwen_grpc.py
|       |-- qwen_metrics.py
|       |-- qwen_inputs.py
|       |-- qwen_remote_hidden_replay.py
|       |-- export_edge_qwen.py
|       `-- qwen_full_baseline.py
|-- inputs/
|   |-- README.md
|   `-- example_prompts.json
|-- tests/
|   `-- test_inputs.py
`-- multi_edge/
    `-- README.md
```

Despite the `qwen_9b` directory name, the edge/cloud entry points and the
full-model baseline support both Qwen3.5-9B and Llama-3.1-8B. Qwen prefix export
and hidden-state replay are Qwen-specific.

## Environment

The reference Linux environment used Python 3.10 and PyTorch 2.6.0 with CUDA
12.4. A typical x86_64 setup is:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

Jetson/Orin systems require a PyTorch build compatible with the installed JetPack
version rather than the x86_64 CUDA wheel above.

Model weights are not included. The scripts use the public model identifiers
`Qwen/Qwen3.5-9B` and `meta-llama/Llama-3.1-8B` by default; model access and local
cache configuration are the user's responsibility.

## Example Inputs

The repository includes a few self-authored prompts for smoke testing. They are
not samples from the paper datasets and are not intended for performance results.
Prepare model-specific JSONL inputs with:

```bash
python prepare_inputs.py --model qwen
python prepare_inputs.py --model llama
```

The generated files are written under `inputs/generated/` and are ignored by Git.
See [`inputs/README.md`](inputs/README.md) for the format and additional options.

The paper experiments use larger selected workloads derived from WildGPT-4 and
LongReason. Those full datasets and generated tokenized files are not included in
this repository.

## Basic Usage

Run commands from the repository root.

### Edge-cloud split inference

Start the cloud process first:

```bash
python grpc_ready/qwen_9b/qwen_cloud.py --model qwen
```

Then start the edge process:

```bash
python grpc_ready/qwen_9b/qwen_edge.py \
  --model qwen \
  --edge-full-model \
  --edge-layers 1 \
  --input-file inputs/generated/qwen.jsonl
```

For two different machines, set `QWEN_CLOUD_HOST` on the edge to the reachable
cloud address. For Llama, use `--model llama` on both processes and pass the
Llama input file.

### Export a Qwen edge prefix

Instead of loading the full checkpoint on the edge, a Qwen prefix shard can be
exported once and reused:

```bash
python grpc_ready/qwen_9b/export_edge_qwen.py \
  --keep-layers 16 \
  --output-dir /path/to/qwen_edge_prefix
```

Then pass it to the edge process with:

```bash
--edge-model-path /path/to/qwen_edge_prefix
```

### Full-model baseline

```bash
python grpc_ready/qwen_9b/qwen_full_baseline.py \
  --model qwen \
  --input-file inputs/generated/qwen.jsonl
```

Use `--model llama` with `inputs/generated/llama.jsonl` for the Llama baseline.

### Qwen hidden-state dump and replay

A small example is:

```bash
python grpc_ready/qwen_9b/qwen_remote_hidden_replay.py dump \
  --input-file inputs/generated/qwen.jsonl \
  --cut-layers 1,4 \
  --max-new-tokens 8 \
  --output-root /tmp/edge-cloud-hidden
```

Use the printed hidden-state directory for replay:

```bash
python grpc_ready/qwen_9b/qwen_remote_hidden_replay.py replay \
  --hidden-dir /path/to/generated/hidden/run \
  --dataset all \
  --delay-values-ms 0,20
```

## Tests and Outputs

Input handling can be checked without downloading model weights:

```bash
python -m unittest discover -s tests -v
```

By default, runtime metrics are written under `grpc_ready/qwen_9b/results*`.
Generated inputs, logs, model weights, hidden tensors, and experiment outputs are
excluded from the repository through `.gitignore`.
