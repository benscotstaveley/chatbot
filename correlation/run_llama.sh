#!/bin/bash

MODEL_DIR=/mnt/models_nvme/models/
PROMPT_DIR=./prompt_simple
CTX=24576  # default

# be sure to only use if all the weights fit in VRAM!  and note
# it will confuse nvtop's memory reporting.
#export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
# this isn't nearly as good as the gold standard one.  it confuses physical
# locations of characters, (who is present)

# https://huggingface.co/bartowski/Qwen2.5-32B-Instruct-GGUF/resolve/main/Qwen2.5-32B-Instruct-Q4_K_M.gguf
# note to self: get rid of thought blocks for qwen
#MODEL=$MODEL_DIR/Qwen2.5-32B-Instruct-Q4_K_M.gguf
#N_GPU_LAYERS=24 ; CTX=24576

#MODEL=$MODEL_DIR/TheDrummer_Cydonia-24B-v4.1-IQ4_XS.gguf
#N_GPU_LAYERS=24 ; CTX=24576

#MODEL=$MODEL_DIR/Midnight-Miqu-70B-v1.5-Q3_K_M.gguf
#N_GPU_LAYERS=24 ; CTX=16384

MODEL=$MODEL_DIR/L3.2-Rogue-Creative-Instruct-Uncensored-Abliterated-7B-D_AU-IQ4_XS.gguf
N_GPU_LAYERS=24 ; CTX=24576

#MODEL=$MODEL_DIR/cognitivecomputations_Dolphin-Mistral-24B-Venice-Edition-Q4_K_M.gguf
#N_GPU_LAYERS=24 ; CTX=24576

#MODEL=$MODEL_DIR/qwen3-30b-a3b-arliai-rpr-v4-fast-q6_k.gguf
#N_GPU_LAYERS=20 ; CTX=24576

#MODEL=$MODEL_DIR/Qwen3-14B-Base.Q5_K_M.gguf
#N_GPU_LAYERS=999 ; CTX=8192

# i suspect the chat template is wrong for this one
#MODEL=$MODEL_DIR/mythomax-l2-13b.Q6_K.gguf
#N_GPU_LAYERS=26 ; CTX=8192

#MODEL=$MODEL_DIR/dolphin-2.2.1-mistral-7b.Q6_K.gguf
#N_GPU_LAYERS=999 ; CTX=24576

SYS_BEHAVIOR_PROMPT="$(cat $PROMPT_DIR/sys_behavior.txt)
SYS_FORMATTING_PROMPT="$(cat $PROMPT_DIR/sys_formatting.txt)
SYS_PROMPT="$SYS_BEHAVIOR_PROMPT$SYS_FORMATTING_PROMPT"
INITIAL_PROMPT=$(cat $PROMPT_DIR/init.txt)

../../../llama.cpp-latest/build/bin/llama-cli \
  -m $MODEL \
  --n-gpu-layers $N_GPU_LAYERS \
  --ctx-size $CTX \
  --flash-attn on \
  --temp 0.0 \
  --top-p 1.0 \
  --top-k 1 \
  --min-p 0.0 \
  --typical_p 1.0 \
  --repeat-penalty 1.0 \
  --conversation \
  --color on \
  --system-prompt "$SYS_PROMPT"   \
  --prompt "$INITIAL_PROMPT" \
  --seed 1 \
  "$@"

