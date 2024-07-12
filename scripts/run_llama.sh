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
