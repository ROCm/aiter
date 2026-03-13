#!/bin/bash
set -euo pipefail

TYPE=${1:-launch}
MODEL_PATH=${2:-meta-llama/Meta-Llama-3-8B-Instruct}
EXTRA_ARGS=("${@:3}")


if [ "$TYPE" == "launch" ]; then
  echo ""
  echo "========== Launching ATOM server =========="
  python -m atom.entrypoints.openai_server --model "$MODEL_PATH" "${EXTRA_ARGS[@]}" &
  atom_server_pid=$!

  echo ""
  echo "========== Waiting for ATOM server to start =========="
  max_retries=30
  retry_interval=60
  for ((i=1; i<=max_retries; i++)); do
      if curl -s http://localhost:8000/v1/completions -o /dev/null; then
          echo "ATOM server is up."
          break
      fi
      echo "Waiting for ATOM server to be ready... ($i/$max_retries)"
      sleep $retry_interval
  done
  if ! curl -s http://localhost:8000/v1/completions -o /dev/null; then
      echo "ATOM server did not start after $((max_retries * retry_interval)) seconds."
      kill $atom_server_pid
      exit 1
  fi
fi

if [ "$TYPE" == "accuracy" ]; then
  echo ""
  if ! command -v lm_eval >/dev/null 2>&1; then
    echo "========== Installing lm-eval =========="
    pip install lm-eval[api]
  else
    echo "========== lm-eval already installed; skipping installation =========="
  fi

  echo ""
  echo "========== Running accuracy test =========="
  mkdir -p accuracy_test_results
  RESULT_FILENAME=accuracy_test_results/$(date +%Y%m%d%H%M%S).json
  lm_eval --model local-completions \
          --model_args model="$MODEL_PATH",base_url=http://localhost:8000/v1/completions,num_concurrent=65,max_retries=1,tokenized_requests=False \
          --tasks gsm8k \
          --num_fewshot 3 \
          --output_path "${RESULT_FILENAME}"
  echo "Accuracy test results saved to ${RESULT_FILENAME}"
  chmod -R 777 accuracy_test_results
fi

if [ "$TYPE" == "benchmark" ]; then
  echo ""
  echo "========== Cloning bench_serving =========="
  git clone https://github.com/kimbochen/bench_serving.git && chmod +x bench_serving/benchmark_serving.py
  echo "========== Running benchmark test =========="
  python bench_serving/benchmark_serving.py \
    --model=$MODEL_PATH --backend=vllm --base-url="http://localhost:8000" \
    --dataset-name=random \
    --random-input-len=$ISL --random-output-len=$OSL --random-range-ratio=$RANDOM_RANGE_RATIO \
    --max-concurrency=$CONC \
    --num-prompts=$(( $CONC * 10 )) \
    --trust-remote-code \
    --request-rate=inf --ignore-eos \
    --save-result --percentile-metrics="ttft,tpot,itl,e2el" \
    --result-dir=. --result-filename=${RESULT_FILENAME}.json
fi