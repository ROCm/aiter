# Add MiniMax-M3 Model in vLLM

- Scope: vLLM + MiniMax-M3
- Repository: ROCm/aiter
- Model: MiniMax-M3-MXFP8
- Image: vllm/vllm-openai-rocm:v0.23.0
- Router type: proxy
- RUN_AFTER_HEALTH: accuracy
- Co-author: lcskrishna <lollachaitanya@gmail.com>
- YAML change: add `MiniMax-M3-MXFP8` to the vLLM benchmark matrix in `.github/workflows/vllm_benchmark.yaml`.
- Note: no AITER runtime code change unless follow-up integration is requested.
- This work was moved from the mistaken ATOM repo target to AITER.

## Provided Command

```bash
IMAGE=vllm/vllm-openai-rocm:v0.23.0 MODEL_NAME=MiniMax-M3-MXFP8 NODES=2 GPUS_PER_NODE=8 WIDE_EP_MODE=0 MORIIO_READ_MODE=0 RUN_AFTER_HEALTH=accuracy ROUTER_TYPE=proxy WAIT=1 SLURM_TIME_LIMIT=08:30:00 bash .buildkite/amd-disagg/run-slurm-disagg-test.sh &
```
