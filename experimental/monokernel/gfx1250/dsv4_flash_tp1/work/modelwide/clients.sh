export PYTHONPATH=/app/aiter:/app/ATOM:${PYTHONPATH:-}
#!/bin/bash
set -euo pipefail
cd /work/modelwide
kind=${1:?baseline or fused}
port=${2:?HTTP port}
for round in 1 2; do
python -m atom.benchmarks.benchmark_serving --model=/data/DeepSeek-V4-Flash --backend=vllm --base-url=http://localhost:${port} --dataset-name=random --random-input-len=1024 --random-output-len=1024 --random-range-ratio=1 --num-prompts=8 --max-concurrency=1 --request-rate=inf --ignore-eos --seed=0 --percentile-metrics=ttft,tpot,itl,e2el --save-result --result-dir=/work/modelwide --result-filename=${kind}-client-round${round}.json >${kind}-client-round${round}.log 2>&1
done
