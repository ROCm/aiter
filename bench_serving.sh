#!/bin/bash
set -euo pipefail

PORT=${1:-8000}

CONCS=(16 64 128)
ISLS=(1024 8192)

for isl in "${ISLS[@]}"; do
  for conc in "${CONCS[@]}"; do
    num_prompts=$((conc * 10))
    echo "=== conc=${conc}, isl=${isl}, num_prompts=${num_prompts} ==="
    python -m atom.benchmarks.benchmark_serving \
      --model=/data/models/gpt-oss-120b/ \
      --backend=vllm \
      --base-url="http://localhost:${PORT}" \
      --dataset-name=random \
      --random-input-len="${isl}" \
      --random-output-len=1024 \
      --random-range-ratio=0.8 \
      --num-prompts="${num_prompts}" \
      --max-concurrency="${conc}" \
      --request-rate=inf \
      --ignore-eos \
      --save-result \
      --percentile-metrics="ttft,tpot,itl,e2el"
    echo ""
  done
done
