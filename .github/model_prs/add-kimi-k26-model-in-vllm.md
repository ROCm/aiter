# Add Kimi-K2.6 Model in vLLM

- Scope: vLLM + Kimi-K2.6
- Repository: ROCm/aiter
- Model: Kimi-K2.6-MXFP4
- Image: vllm/vllm-openai-rocm:v0.23.0
- Router type: vllm-router
- RUN_AFTER_HEALTH: accuracy
- Co-author: lcskrishna <lollachaitanya@gmail.com>
- YAML change: add `Kimi-K2.6-MXFP4` to the vLLM benchmark matrix in `.github/workflows/vllm_benchmark.yaml`.
- Note: no AITER runtime code change unless follow-up integration is requested.
- Migration note: moved from the mistaken ATOM repo target to AITER.

Provided command:

```bash
IMAGE=vllm/vllm-openai-rocm:v0.23.0 MODEL_NAME=Kimi-K2.6-MXFP4 NODES=2 GPUS_PER_NODE=8 WIDE_EP_MODE=0 MORIIO_READ_MODE=0 RUN_AFTER_HEALTH=accuracy ROUTER_TYPE=vllm-router WAIT=1 SLURM_TIME_LIMIT=08:30:00 bash .buildkite/amd-disagg/run-slurm-disagg-test.sh &
```
