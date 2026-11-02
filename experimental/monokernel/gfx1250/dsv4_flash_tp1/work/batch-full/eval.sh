#!/bin/bash
set -euo pipefail
port=${1:?port};tag=${2:?tag}
python -m lm_eval --model local-completions --model_args "model=/data/DeepSeek-V4-Flash,base_url=http://localhost:${port}/v1/completions,num_concurrent=${EVAL_CONCURRENCY:-8},max_retries=1,tokenized_requests=False,trust_remote_code=True" --tasks gsm8k --num_fewshot 3 --limit 32 --batch_size 1 --gen_kwargs max_gen_toks=512,temperature=0 --log_samples --output_path "/work/batch-full/gsm8k-${tag}"
