#!/bin/bash
# bench_1k1k.sh TAG [PORT] - vLLM random 1k in / 1k out, concurrency 1, same settings for every backend.
set -eo pipefail
TAG=${1:?tag}; PORT=${2:-8343}; N=${N:-32}
OUT=/home/vigokhal/m3_bench/results; mkdir -p $OUT
TS=$(date +%Y%m%d_%H%M%S)
export HF_HOME=/data/huggingface-cache HF_HUB_OFFLINE=1
vllm bench serve --backend openai-chat --endpoint /v1/chat/completions --port "$PORT" \
  --model minimax-m3 --tokenizer /data/models/MiniMax-M3-MXFP4 --trust-remote-code \
  --dataset-name random --random-input-len 1024 --random-output-len 1024 --ignore-eos \
  --max-concurrency 1 --num-prompts "$N" --num-warmups 2 --seed 42 --percentile-metrics ttft,tpot,itl,e2el \
  --save-result --result-dir $OUT --result-filename "${TAG}_${TS}.json" 2>&1 | tee $OUT/${TAG}_${TS}.log
