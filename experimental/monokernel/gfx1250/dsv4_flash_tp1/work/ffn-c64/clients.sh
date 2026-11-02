#!/bin/bash
set -euo pipefail
cd /work/ffn-c64
for round in 1 2; do
python -m atom.benchmarks.benchmark_serving --model=/data/DeepSeek-V4-Flash --backend=vllm --base-url=http://localhost:8000 --dataset-name=random --random-input-len=1024 --random-output-len=1024 --random-range-ratio=1 --num-prompts=256 --max-concurrency=64 --request-rate=inf --ignore-eos --seed=0 --percentile-metrics=ttft,tpot,itl,e2el --save-result --result-dir=/work/ffn-c64 --result-filename=client-round${round}.json >client-round${round}.log 2>&1
done
