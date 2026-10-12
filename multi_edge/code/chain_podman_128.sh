#!/bin/bash
# Reruns every reported experiment at 128 generated tokens per session (was 16),
# inside Podman. 128 is the shortest output length the standard serving
# benchmarks use; at 16 a session's decode phase lasted ~0.23 s, so fleet
# membership turned over constantly and prefill was a large share of every
# timed window.
#
# Every output carries a _pod128 suffix. The original run also included a
# batch-window-width sweep and a cache-persistence comparison; neither is
# reported, so both are omitted here.
set -u
HERE=$(dirname "$(readlink -f "$0")"); cd "$HERE"
RUN="podman run --rm --device nvidia.com/gpu=all \
  -v "${HF_CACHE:-$HOME/.cache/huggingface}":/root/.cache/huggingface \
  -v "$HERE":/app/code -w /app/code \
  -e EDGE_PYTHON=python3.12 -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
  -e GEN_TOKENS=128"
IMG=${IMG:-edge-cloud-dsb:latest}
PY=python3.12
step() { echo; echo "########## $1 $(date -Is)"; }


step "[1/10] Poisson arrivals, heterogeneous"
$RUN $IMG $PY poisson_bench.py --cut-layers 2,3,4,5,6,7,8,9 --edge-devices cuda:1,cuda:2 \
  --conditions fifo,batch,interleave,progressive-sc --target-concurrency 2,4,8 \
  --repeats 15 --num-sessions 24 --pool-size 16 \
  --results-json stats_poisson_hetero_pod128.json || echo "[1] FAILED"

step "[2/10] burst crossover vs interleaving, block 1"
$RUN $IMG $PY scale_bench_stats.py --cut-layers 9,8,7,6,5,4,3,2 --repeats 15 \
  --conditions fifo,batch,interleave,progressive-sc \
  --results-json stats_hetero_sched_pod128_b1.json || echo "[2a] FAILED"
step "[2/10] burst crossover, block 2 (order reversed)"
$RUN $IMG $PY scale_bench_stats.py --cut-layers 9,8,7,6,5,4,3,2 --repeats 15 \
  --conditions progressive-sc,interleave,batch,fifo \
  --results-json stats_hetero_sched_pod128_b2.json || echo "[2b] FAILED"

step "[3/10] tail-depth sweep"
$RUN -e SWEEP_OUT=stats_tail_depth_sweep_pod128.json $IMG $PY tail_depth_sweep.py || echo "[3] FAILED"

step "[4/10] diversity sweep"
$RUN -e SWEEP_OUT=stats_diversity_sweep_pod128.json $IMG $PY diversity_sweep.py || echo "[4] FAILED"

step "[5/10] decode-step profile, short and long context"
for ds in wildgpt4 longreason; do
  $RUN -e DS=$ds $IMG $PY profile_progressive.py > prof_${ds}_pod128.log 2>&1 || echo "[5:$ds] FAILED"
  cp scale_stats_server.log profile_${ds}_pod128_server.log 2>/dev/null
done

step "[6/10] context-length sweep"
$RUN -e SWEEP_OUT=stats_context_length_sweep_pod128.json $IMG $PY context_length_sweep.py || echo "[6] FAILED"


step "[7/10] three-shape gradient: homogeneous cut 2"
$RUN $IMG $PY scale_bench_stats.py --cut-layer 2 --repeats 30 --conditions fifo,batch \
  --results-json stats_gradient_homog_pod128.json || echo "[7a] FAILED"
step "[7/10] three-shape gradient: two-tier 2/14"
$RUN $IMG $PY scale_bench_stats.py --cut-layers 2,2,2,2,14,14,14,14 --repeats 30 --conditions fifo,batch \
  --results-json stats_gradient_twotier_pod128.json || echo "[7b] FAILED"
step "[7/10] three-shape gradient: fully heterogeneous"
$RUN $IMG $PY scale_bench_stats.py --cut-layers 2,4,7,10,13,16,19,22 --repeats 30 --conditions fifo,batch \
  --results-json stats_gradient_fullyhetero_pod128.json || echo "[7c] FAILED"

step "[8/10] KV-cache memory, batching off and on"
for w in 0 8; do
  tag=$([ $w = 0 ] && echo off || echo on)
  $RUN $IMG $PY mem_bench.py --model-id meta-llama/Llama-3.2-3B --cut-layers 2,2,2,2,2,2,2,2 \
    --edge-device cuda:1 --edge-dtype bfloat16 --batch-window-ms $w \
    > log_mem_llama_homog_${tag}_pod128.log 2>&1 || echo "[8:$tag] FAILED"
done

step "[9/10] Qwen2.5-3B cross-model tail-depth sweep"
$RUN -e SWEEP_MODEL=Qwen/Qwen2.5-3B -e SWEEP_LAYERS=36 -e SWEEP_TAG=_qwen_pod128 \
  $IMG $PY tail_depth_sweep_gen.py || echo "[10] FAILED"

step "[10/10] fleet-shape neutrality matrix (19 configurations)"
$RUN -e SWEEP_OUT=stats_fullshape_matrix_pod128.json $IMG $PY fullshape_matrix_v2.py || echo "[11] FAILED"

step "ALL RUNS DONE"
