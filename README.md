# Edge-Cloud Split Inference

This repository contains the DySCo artifact for edge-cloud LLM inference:
single-edge split inference with Qwen3.5-9B and Llama-3.1-8B, and multi-edge
serving with depth-synchronized batching (DSB), evaluated with Llama-3.2-3B
and Qwen2.5-3B.

## Scope

Included in this repository:

- layer-wise split inference between an edge device and a cloud server;
- dynamic forward execution over selected decoder-layer ranges;
- Qwen prefix-shard export for memory-constrained edge devices;
- Qwen hidden-state dump and suffix replay with optional delay injection;
- single-device full-model baselines for Qwen and Llama;
- small self-authored inputs and input-preparation utilities;
- multi-edge serving with heterogeneous cut layers and depth-synchronized
  batching, with the measurement data behind the reported numbers.

Use the [single-edge setup](#single-edge) or [multi-edge / DSB setup](#multi-edge-and-dsb)
below. Each has its own environment and input preparation.

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
    |-- Containerfile
    |-- reproduce.sh
    |-- code/
    |-- data/
    |-- figures/
    `-- results/
```

## Single-Edge

Run this section's commands from the repository root. Despite the `qwen_9b`
directory name, the edge/cloud entry points and full-model baseline support both
Qwen3.5-9B and Llama-3.1-8B. Qwen prefix export and hidden-state replay are
Qwen-specific.

### Environment

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

### Example inputs

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

### Usage

#### Edge-cloud split inference

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

#### Export a Qwen edge prefix

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

#### Full-model baseline

```bash
python grpc_ready/qwen_9b/qwen_full_baseline.py \
  --model qwen \
  --input-file inputs/generated/qwen.jsonl
```

Use `--model llama` with `inputs/generated/llama.jsonl` for the Llama baseline.

#### Qwen hidden-state dump and replay

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

### Tests and outputs

Input handling can be checked without downloading model weights:

```bash
python -m unittest discover -s tests -v
```

Single-edge runtime metrics are written under `grpc_ready/qwen_9b/results*`.
Generated inputs, logs, model weights, hidden tensors, and runtime outputs are
ignored by Git.

## Multi-Edge and DSB

Code and measurement data for the multi-edge part of DySCo: the
depth-synchronized batching (DSB) scheduler and its three baselines.
Experiments use `meta-llama/Llama-3.2-3B` and `Qwen/Qwen2.5-3B`.

The multi-edge implementation uses `multi_edge/Containerfile` (CUDA 12.8,
torch 2.9.0, transformers 5.16.1, grpcio 1.83.0), separately from the root
`requirements.txt`. From the repository root, enter its directory before running
this section's commands:

```bash
cd multi_edge
```

Paths below are relative to `multi_edge/`.

### Analyze the included results

With Python 3.10 or newer, recompute the summary numbers without a GPU or model
download:

```bash
./reproduce.sh numbers
```

To redraw the figures:

```bash
python3 -m pip install matplotlib
./reproduce.sh figures
```

Analysis reads each result file from `code/` if present, otherwise from
`results/`. It writes `results/analyzed_pod128.json`; figure generation writes
PDFs to `figures/`.

DSB is implemented in `code/shard02_multi.py`. The scheduler is selected at
server start:

| Scheduler | Server flags |
| --- | --- |
| FIFO | `--batch-window-ms 0` |
| Exact-match batching | `--batch-window-ms 8` |
| Round-robin interleaving | `--batch-window-ms 0 --interleave-group-size N` |
| DSB | `--batch-window-ms 8 --progressive-batching` |

### Re-run the measurements

Use Linux with Podman and NVIDIA GPU access (`--device nvidia.com/gpu=all`).
The current scripts require at least three visible GPUs: `cuda:0` for the cloud
and `cuda:1` / `cuda:2` for edge processes. The reference setup used a 96 GB cloud
GPU and 32 GB edge GPUs. Adjust device assignments if your GPU ordering differs.

Prepare a host Python environment for prompt reconstruction and model downloads.
Before logging in, obtain access to the gated Hugging Face repositories
`meta-llama/Llama-3.2-3B` (measurement model) and `meta-llama/Llama-3.1-8B`
(tokenizer used by the prompt length check).

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install datasets transformers==5.16.1 huggingface_hub
hf auth login
python data/build_prompts.py
```

The reconstructed prompts are written to `selected_testsets/<group>/raw_inputs.jsonl`.
The measurement script mounts this directory read-only at `/app/selected_testsets`.

Measurements run with `HF_HUB_OFFLINE=1`, so download both models beforehand to
the cache that the script mounts into the container:

```bash
export HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
hf download meta-llama/Llama-3.2-3B --cache-dir "$HF_CACHE/hub"
hf download Qwen/Qwen2.5-3B --cache-dir "$HF_CACHE/hub"
podman build -t edge-cloud-dsb:latest -f Containerfile .
./reproduce.sh measure
```

For CLI details, see the [Hugging Face CLI documentation](https://huggingface.co/docs/huggingface_hub/guides/cli).

`code/chain_podman_128.sh` runs the measurement sequence with 128 generated tokens
per session and Python 3.12 inside the container. `IMG` can override the image
tag; `HF_CACHE` can override the host cache directory. New measurement files are
written to `code/`; rerun `./reproduce.sh numbers` and `./reproduce.sh figures`
to analyze them. Check the output for `FAILED` messages before using the results.

### Prompts

Prompt text is not redistributed. The manifests record dataset, split, source
row and tokenized lengths for 2,314 prompts from `mlabonne/WildGPT-4` and 632
from `lz1bytedance/LongReason` (split `8k`). `data/build_prompts.py` reconstructs
the prompts and checks their lengths with the Llama-3.1-8B tokenizer.

### Measurement scope

Throughput includes prompt processing and covers 128 generated tokens per
session. Edge and cloud processes share one server; wide-area latency is
untested, heterogeneous edge memory is emulated through cut layers, and prefill
is not batched.
