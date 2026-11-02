#!/bin/bash
set -euo pipefail

# step 1: please pip install pandas openpyxl
# step 3: change the variables as below
# step 4: bash benchmark.sh

##################### need changed #######################
MODEL_PATH="${1:-/data/DeepSeek-V4-Pro}"
ISL_LIST=(1024)
CONC_LIST=(${2:-64})
OSL=1024
PORT=8000
##########################################################

# LOG_FILE="result_0910_tp4_a4w4_opt.txt"
LOG_FILE="result_0910_tp4_baseline.txt"
# LOG_FILE="result.txt"
# clean old file
# if [ -f "$LOG_FILE" ]; then
#     rm "$LOG_FILE"
# fi

# health check
curl -sf "http://localhost:$PORT/health" > /dev/null || {
    echo "ERROR: Server not running on port $PORT at vllm backend"
    exit 1
}

for ISL in "${ISL_LIST[@]}"; do

    for CONC in "${CONC_LIST[@]}"; do
    echo "=========================================" | tee -a "$LOG_FILE"
    echo "Start Test ISL=$ISL, OSL=$OSL, CONC=$CONC" | tee -a "$LOG_FILE"
    echo "=========================================" | tee -a "$LOG_FILE"
        
        python -m atom.benchmarks.benchmark_serving \
            --model="$MODEL_PATH" \
            --backend=vllm \
            --base-url="http://localhost:$PORT" \
            --dataset-name=random \
            --random-input-len="$ISL" \
            --random-output-len="$OSL" \
            --random-range-ratio 1 \
            --num-prompts=$(( CONC * 4 )) \
            --max-concurrency="$CONC" \
            --request-rate=inf \
            --ignore-eos \
            --percentile-metrics="ttft,tpot,itl,e2el"  2>&1 | tee -a "$LOG_FILE"
        
        echo -e "\n" | tee -a "$LOG_FILE"
    done
done

if [ -f "$LOG_FILE" ]; then
    python save_csv.py $LOG_FILE
fi

