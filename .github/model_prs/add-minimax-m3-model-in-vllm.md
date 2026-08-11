# Add MiniMax-M3 Model in vLLM

- Scope: vLLM + MiniMax-M3
- Repository: ROCm/aiter
- Model: MiniMax-M3-MXFP8
- vLLM source: lcskrishna/vllm@3e90fd20a48f6b42c395224f10a7793117bf65a6
- Image: vllm/vllm-openai-rocm:v0.23.0
- Script: .buildkite/amd-disagg/run-slurm-disagg-test.sh
- Router type: proxy
- RUN_AFTER_HEALTH: accuracy
- Co-author: lcskrishna <lollachaitanya@gmail.com>
- YAML change: add `MiniMax-M3-MXFP8` to the vLLM benchmark matrix in `.github/workflows/vllm_benchmark.yaml`.
- DI workflow change: make `.github/workflows/vllm-disagg-ci-smoke-workflow.yaml` default to MiniMax and pass the model-specific disagg environment from the vLLM Buildkite config.
- Temporary PR trigger: PR #4274 runs this workflow on PR updates; other PRs can trigger it with `ci:vllm-di` or `ci:all`.
- Note: no AITER runtime code change unless follow-up integration is requested.
- This work was moved from the mistaken ATOM repo target to AITER.

## Provided Command

```bash
IMAGE=vllm/vllm-openai-rocm:v0.23.0 MODEL_NAME=MiniMax-M3-MXFP8 NODES=2 GPUS_PER_NODE=8 WIDE_EP_MODE=0 MORIIO_READ_MODE=0 RUN_AFTER_HEALTH=accuracy ROUTER_TYPE=proxy WAIT=1 SLURM_TIME_LIMIT=08:30:00 bash .buildkite/amd-disagg/run-slurm-disagg-test.sh &
```
