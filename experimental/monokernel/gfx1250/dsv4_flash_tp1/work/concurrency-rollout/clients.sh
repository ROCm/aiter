#!/bin/bash
set -euo pipefail
export PYTHONPATH=/app/aiter:/app/ATOM:${PYTHONPATH:-}
kind=${1:?baseline or fused}
port=${2:?HTTP port}
con=${3:?concurrency}
for round in 1 2; do
python -m atom.benchmarks.benchmark_serving --model=/data/DeepSeek-V4-Flash --backend=vllm --base-url=http://localhost:${port} --dataset-name=random --random-input-len=1024 --random-output-len=1024 --random-range-ratio=1 --num-prompts=$((con*8)) --max-concurrency=${con} --request-rate=inf --ignore-eos --seed=0 --percentile-metrics=ttft,tpot,itl,e2el --save-result --result-dir=/work/concurrency-rollout --result-filename=${kind}-c${con}-round${round}.json > /work/concurrency-rollout/${kind}-c${con}-round${round}.log 2>&1
done
