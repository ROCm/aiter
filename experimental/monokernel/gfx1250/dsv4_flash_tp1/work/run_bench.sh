#!/bin/bash
set -euo pipefail
MODE=${1:?baseline or hybrid}; CONC=${2:?concurrency}; PROFILE=${3:-0}
mkdir -p /work/results /work/logs
OSL=1024; MULT=4; EXTRA=()
if [ "$PROFILE" = 1 ]; then OSL=50; MULT=1; EXTRA+=(--profile); fi
NAME=${MODE}_c${CONC}_profile${PROFILE}
curl -sf http://localhost:8000/health >/dev/null
python -m atom.benchmarks.benchmark_serving \
 --model=/data/DeepSeek-V4-Pro --backend=vllm --base-url=http://localhost:8000 \
 --dataset-name=random --random-input-len=1024 --random-output-len="$OSL" \
 --random-range-ratio=1 --num-prompts=$((CONC * MULT)) --max-concurrency="$CONC" \
 --request-rate=inf --ignore-eos --seed=0 --percentile-metrics=ttft,tpot,itl,e2el \
 --save-result --result-dir=/work/results --result-filename="$NAME.json" \
 "${EXTRA[@]}" 2>&1 | tee "/work/logs/$NAME.log"
