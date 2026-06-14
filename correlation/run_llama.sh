#!/bin/bash

set -euo pipefail

usage() {
    echo "Usage: $0 [-v] [-k] [-m] prompt_suite" >&2
    echo "  -v   pass args to llama-cli appropriate for verbose output" >&2
    echo "  -k   pass args to llama--cli appropriate for top-k sampling (t>0, top-k>0, etc)" >&2
    echo "  -m   pass args to llama-cli appripriate for min-p sampling" >&2
    exit 1
}

MODEL_DIR=/mnt/models_nvme/models/
MODEL=$MODEL_DIR/cognitivecomputations_Dolphin-Mistral-24B-Venice-Edition-Q4_K_M.gguf
N_GPU_LAYERS=24
CTX=24576

# be terse and deterministic by default; override with -v and -r
VERBOSITY_ARGS=()
DETERMINISTIC_SETTINGS=(--temp 0.0 --top-p 1.0 --top-k  1 --min-p 0.0 --typical_p 1.0)
TOPK_SETTINGS=(--temp 0.9 --top-p 1.0 --top-k 40 --min-p 0.0 --typical_p 1.0)
MINP_SETTINGS=(--temp 0.9 --top-p 1.0 --top-k  0 --min-p 0.2 --typical_p 1.0)
RANDOMNESS_ARGS=(${DETERMINISTIC_SETTINGS[@]})

while getopts "vkm" opt; do
    case "${opt}" in
        v) VERBOSITY_ARGS=(--verbose-prompt -v) ;;
        k) RANDOMNESS_ARGS=(${TOPK_SETTINGS[@]}) ;;
        m) RANDOMNESS_ARGS=(${MINP_SETTINGS[@]}) ;;
        *) usage ;;
    esac
done
shift "$((OPTIND - 1))"

if [[ $# -ne 1 ]]; then
    usage
    exit 1
else
    PROMPT_DIR=$1
fi     

SYS_BEHAVIOR_PROMPT="$(cat $PROMPT_DIR/sys_behavior.txt)"
SYS_FORMATTING_PROMPT="$(cat $PROMPT_DIR/sys_formatting.txt)"
SYS_PROMPT="$SYS_BEHAVIOR_PROMPT$SYS_FORMATTING_PROMPT"
INITIAL_PROMPT=$(cat $PROMPT_DIR/init.txt)

CMD=(
../../../llama.cpp-latest/build/bin/llama-cli
  -m "$MODEL"
  --n-gpu-layers "$N_GPU_LAYERS"
  --ctx-size "$CTX"
  --flash-attn on
  ${RANDOMNESS_ARGS[@]}
  --repeat-penalty 1.0
  --conversation
  --color on
  --system-prompt "$SYS_PROMPT" 
  --prompt "$INITIAL_PROMPT"
  --seed 1
  ${VERBOSITY_ARGS[@]}
)
echo "${CMD[@]}"
"${CMD[@]}"


