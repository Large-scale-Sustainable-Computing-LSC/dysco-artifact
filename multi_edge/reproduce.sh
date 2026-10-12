#!/bin/bash
# Reproduce the paper's multi-edge numbers and figures.
#
#   ./reproduce.sh numbers   recompute every reported value from results/ (seconds, no GPU)
#   ./reproduce.sh figures   redraw the paper's figures from results/      (seconds, no GPU)
#   ./reproduce.sh all       both of the above
#   ./reproduce.sh measure   re-run every experiment from scratch (GPU, ~10 h)
#
# "numbers" and "figures" need only Python; "figures" additionally needs
# matplotlib. "measure" needs the container image, two GPUs and the prompt
# sets rebuilt by data/build_prompts.py.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
FAMILY=pod128          # 128 generated tokens, the configuration reported in the paper
PY=${PYTHON:-python3}

case "${1:-all}" in
  numbers)
    "$PY" "$HERE/code/analyze_results.py" "$FAMILY"
    ;;
  figures)
    "$PY" "$HERE/code/make_paper_figures.py" "$FAMILY"
    ;;
  all)
    "$PY" "$HERE/code/analyze_results.py" "$FAMILY"
    "$PY" "$HERE/code/make_paper_figures.py" "$FAMILY"
    ;;
  measure)
    echo "Re-running all measurements. This needs the container image, two GPUs"
    echo "and the prompt sets (see data/build_prompts.py). Roughly 9 hours."
    bash "$HERE/code/chain_podman_128.sh"
    ;;
  *)
    echo "usage: $0 {numbers|figures|all|measure}" >&2; exit 2
    ;;
esac
