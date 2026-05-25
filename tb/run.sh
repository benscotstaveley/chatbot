#!/bin/bash

LLAMA_CPP_PATH=/home/ben/llama.cpp-latest
LLAMA_CPP_BIN_PATH=$LLAMA_CPP_PATH/build/bin
LLAMA_EXE=$LLAMA_CPP_BIN_PATH/llama-completion
MODEL_ROOT=/home/ben/llama.cpp/models
MODEL_ROOT=/mnt/models_nvme/models
MODEL=$MODEL_ROOT/L3.2-Rogue-Creative-Instruct-Uncensored-Abliterated-7B-D_AU-IQ4_XS.gguf

LLAMA_FLAGS="\
  --model $MODEL \
  --n-gpu-layers 999  \
  --flash-attn on  \
  --temperature 0.0  \
  --top-k 1  \
  --top-p 1.0  \
  --min-p 0.0  \
  --typical-p 1.0  \
  --repeat-penalty 1.0  \
  --presence-penalty 0.0  \
  --frequency-penalty 0.0  \
  --mirostat 0  \
  --mirostat-lr 0.1  \
  --mirostat-ent 5.0  \
  --seed 2 \
  --n-predict 200 \
  --system-prompt-file tb/system_prompt_test.txt  \
  --chat-template-file tb/chat_template.jinja  \
"

# have used these two together to skip sys prompt
#  -f tb/test_prompt.txt \
#  --no-conversation \

$LLAMA_EXE $LLAMA_FLAGS -p "`cat prompts/init.txt`" --verbose-prompt

# don't use -p if non-conversation
# $LLAMA_EXE $LLAMA_FLAGS  --verbose-prompt
# produces:
# > The capital of France is
# 
# <|start_header_id|>system<|end_header_id|>
# 
# Cutting Knowledge Date: December 2023
# Today Date: 23 May 2026
# 
# <|eot_id|><|start_header_id|>user<|end_header_id|>
# 
# The capital of France is<|eot_id|><|start_header_id|>assistant<|end_header_id|>
# 
# 
# 
# ...Paris.

# $LLAMA_EXE $LLAMA_FLAGS -p "The capital of France is" --dump-kv-cache
# produces: error

#$LLAMA_EXE $LLAMA_FLAGS -p "The capital of France is" --log-verbose
# produces: really quite a lot of output
