#!/usr/bin/env bash

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --gpu) CUDA_DEVICE="$2"; shift ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
  shift
done

CUDA_DEVICE="${CUDA_DEVICE:-0}"

echo "Using CUDA_VISIBLE_DEVICES=${CUDA_DEVICE}"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
python3 -m vllm.entrypoints.openai.api_server \
  --model LangAGI-Lab/camel \
  --gpu-memory-utilization 0.80 \
  --host 0.0.0.0 \
  --port 8000
