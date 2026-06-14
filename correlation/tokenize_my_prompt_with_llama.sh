#!/bin/bash


../../../llama.cpp-latest/build/bin/llama-cli \
  -m /mnt/models_nvme/models/L3.2-Rogue-Creative-Instruct-Uncensored-Abliterated-7B-D_AU-IQ4_XS.gguf \
  --n-gpu-layers 24 \
  --ctx-size 24576 \
  --flash-attn on \
  --temp 0.0 \
  --top-p 1.0 \
  --top-k 1 \
  --min-p 0.0 \
  --typical-p 1.0 \
  --repeat-penalty 1.0 \
  --repeat-last-n 0 \
  --seed 1 \
  -no-cnv \
  --verbose-prompt \
  -v \
  -f /tmp/test_prompt.txt \
  > ./tokenize_my_prompt_with_llama.log 2>&1 
