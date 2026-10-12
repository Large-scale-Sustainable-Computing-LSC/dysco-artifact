# Multi-Edge / DSB

Code and measurement data for the multi-edge part of the paper: the
depth-synchronized batching (DSB) scheduler, the three baselines it is compared
against, and every experiment reported in Section V-B-2.

This directory is self-contained. It pins its own environment in
`Containerfile` (CUDA 12.8, torch 2.9.0, transformers 5.16.1, grpcio 1.83.0),
which is newer than the repository-root `requirements.txt`, and it prepares its
own prompt sets through `data/build_prompts.py` rather than the root
`prepare_inputs.py` flow. Paths below are relative to this directory.

## Reproduce

```bash
./reproduce.sh numbers    # recompute every reported value from results/ (seconds, no GPU)
./reproduce.sh figures    # redraw the figures (needs matplotlib)
./reproduce.sh measure    # re-run all experiments (container, two GPUs, ~9 h)
```

`code/analyze_results.py` derives every number quoted in the paper from the raw
result files; nothing is transcribed by hand.

## Layout

| Path | Contents |
| --- | --- |
| `code/` | Prototype (`shard01_multi.py` edge, `shard02_multi.py` cloud, `llama_multi.proto`), the measurement scripts, and the analysis scripts |
| `results/` | Raw results of every reported experiment |
| `figures/` | The paper's multi-edge figures |
| `data/` | Prompt manifests and `build_prompts.py` |
| `Containerfile` | The environment used for all measurements, with pinned versions |

DSB is `_run_decode_group_progressive_single_checkpoint` in
`shard02_multi.py`. The scheduler is chosen at server start:

| Scheduler | Server flags |
| --- | --- |
| FIFO | `--batch-window-ms 0` |
| Exact-match batching | `--batch-window-ms 8` |
| Round-robin interleaving | `--batch-window-ms 0 --interleave-group-size N` |
| DSB | `--batch-window-ms 8 --progressive-batching` |

## Re-running the measurements

All measurements ran in the container on one server with a 96 GB GPU for the
cloud process and 32 GB GPUs for the edges:

```bash
podman build -t edge-cloud-dsb:latest -f Containerfile .
python3 data/build_prompts.py     # rebuild the prompt sets, see below
./reproduce.sh measure
```

`code/chain_podman_128.sh` runs the ten steps that produced `results/`, in
order. Variables: `GEN_TOKENS` (tokens per session, 128 in the paper),
`EDGE_PYTHON` (interpreter for spawned processes), `IMG` (image tag),
`HF_CACHE` (host model cache). Models come from Hugging Face on first use:
`meta-llama/Llama-3.2-3B` and `Qwen/Qwen2.5-3B`; the Llama checkpoints are
gated.

## Prompts

The prompt text is not redistributed, because the source datasets carry their
own licence terms. `data/` records the source dataset, split, row index and
tokenized length of every prompt used: 2,314 from `mlabonne/WildGPT-4` at 16 to
128 tokens, and 632 from `lz1bytedance/LongReason` (split `8k`) at 4,145 to
5,271 tokens. `build_prompts.py` rebuilds them and re-checks each length
against the manifest.

## Method notes

- Condition order is randomized within each measurement point, and point order
  within each sweep, so drift contributes noise rather than bias.
- A difference is significant at Welch's `|t| > 2.5`; differences below about
  six percentage points are within run-to-run variation and are not interpreted.
- Throughput covers a fixed 128 tokens per session, timed from the first
  session's start to the last token, so prompt processing is included.
- Edge and cloud share one server, so wide-area latency is untested;
  heterogeneous edge memory is emulated by assigning cut layers; prefill is not
  batched. The single-edge study is not part of this artifact.
