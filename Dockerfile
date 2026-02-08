FROM vllm/vllm-openai:latest

# Cache location (you’ll want to mount storage here in ACA)
ENV HF_HOME=/data/hf
ENV TRANSFORMERS_CACHE=/data/hf
ENV HUGGINGFACE_HUB_CACHE=/data/hf

# Configurable at deploy time
ENV MODEL_NAME=LangAGI-Lab/camel
ENV VLLM_GPU_UTIL=0.80
ENV PORT=8000

EXPOSE 8000

# Start OpenAI-compatible server
ENTRYPOINT ["bash","-lc","python3 -m vllm.entrypoints.openai.api_server --model ${MODEL_NAME} --gpu-memory-utilization ${VLLM_GPU_UTIL} --host 0.0.0.0 --port ${PORT}"]
