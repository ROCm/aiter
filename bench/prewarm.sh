#!/bin/bash
# Compile every prefill shape bench_dsv4.sh produces (k concurrent ISL=1024 requests -> M = 1024*k)
# so first-use JIT compiles do not land inside the measured runs.
MODEL_PATH="$1"
for k in $(seq 1 16); do
  python -m atom.benchmarks.benchmark_serving \
    --model="$MODEL_PATH" \
    --backend=vllm \
    --base-url="http://localhost:8000" \
    --dataset-name=random \
    --random-input-len=1024 \
    --random-output-len=4 \
    --random-range-ratio 1 \
    --num-prompts="$k" \
    --max-concurrency="$k" \
    --request-rate=inf \
    --ignore-eos \
    --num-warmups 0 \
    --disable-tqdm 2>&1 | grep -E "Successful requests|Mean TTFT" | tr -s ' ' | tr '\n' ' '
  echo " <- k=$k"
done
