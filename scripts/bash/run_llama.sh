#!/bin/bash

# Run LLaMA container with specified model
docker run -p 8080:8080 -v $(pwd)/model:/models \
  ghcr.io/ggerganov/llama.cpp:server \
  -m models/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf \
  -n -1 \
  -c 512 \
  -t 10
  --host 0.0.0.0 \
  --port 8080


docker run -p 8080:8080 -v $(pwd)/model:/models ghcr.io/ggerganov/llama.cpp:server -m models/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf -c 512 -t 10 --prompt-cache-all --in-prefix 'MED_USER:' --in-suffix 'ASSISTANT:' --keep -1 -p "This is a conversation between User and Llama, an intelligent, friendly and polite medical assistant. Llama is helpful, kind, honest, good at writing, never fails to answer any requests immediately with precision and provides detailed and helpful answers to user's medical questions including accurate references where applicable." --host 0.0.0.0 --port 8080



docker run -p 8080:8080 -v $(pwd)/model:/models ghcr.io/ggerganov/llama.cpp:server -m models/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf -c 512 -t 10 -p "This is a conversation between User and Llama, an intelligent, friendly and polite medical assistant. Llama is helpful, kind, honest, good at writing, never fails to answer any requests immediately with precision and provides detailed and helpful answers to user's medical questions including accurate references where applicable." --host 0.0.0.0 --port 8080