# vLLM nightly + Triton 3.8 (ROCm/triton 669b31ac) with batched mfma_scaled
ARG BASE=vllm/vllm-openai-rocm:nightly-rocm100-00b6c4c565a8732c131e1f452d4a17803bb0842b
FROM ${BASE}
COPY wheels/triton-*.whl /tmp/
RUN pip install --no-deps --force-reinstall /tmp/triton-*.whl && rm /tmp/triton-*.whl
