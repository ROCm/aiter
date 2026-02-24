#!/usr/bin/env bash
# Generic vLLM serving benchmark profiler.
# Self-contained: starts server, runs benchmarks, collects metrics, outputs table + CSV.
# No external scripts required.
#
# Usage:
#   ./run_profiling.sh --model /data/MyModel [OPTIONS]
#
# Required:
#   --model PATH                  Model path or HuggingFace model name
#
# Server options:
#   --tp SIZE                     Tensor parallel size (default: auto-detect GPU count)
#   --max-model-len LEN           Max model length (default: 16384)
#   --max-num-seqs N              Max concurrent sequences (default: 1024)
#   --gpu-mem-util FRAC           GPU memory utilization (default: 0.95)
#   --host HOST                   Server host (default: localhost)
#   --port PORT                   Server port (default: 8000)
#   --server-extra-args "ARGS"    Extra args passed verbatim to `vllm serve`
#
# Benchmark options:
#   --combos "I:O ..."            Input:output token combos (default: "1024:1024 1024:8192 8192:1024")
#   --concurrencies "N ..."       Concurrency values to sweep (default: "4 8 16 32 64")
#   --prompts-multiplier N        num_prompts = N * max_concurrency (default: 8)
#   --no-warmup                   Skip the JIT warmup run (not recommended)
#
# Output options:
#   --output-dir DIR              Directory for CSV output (default: script directory)
#
# Environment variables:
#   Server-side env vars (e.g. VLLM_ROCM_USE_AITER=1) should be exported before running.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ──────────────────────── Defaults ────────────────────────

MODEL=""
TP_SIZE=""
MAX_MODEL_LEN=16384
MAX_NUM_SEQS=1024
GPU_MEM_UTIL=0.95
HOST=localhost
PORT=8000
SERVER_EXTRA_ARGS=""

INPUT_OUTPUT_COMBOS="1024:1024 1024:8192 8192:1024"
CONCURRENCIES="4 8 16 32 64"
PROMPTS_MULTIPLIER=8
DO_WARMUP=1

OUTPUT_DIR="$SCRIPT_DIR"

# ──────────────────────── Arg parsing ────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)              MODEL="$2";              shift 2 ;;
        --tp)                 TP_SIZE="$2";            shift 2 ;;
        --max-model-len)      MAX_MODEL_LEN="$2";     shift 2 ;;
        --max-num-seqs)       MAX_NUM_SEQS="$2";      shift 2 ;;
        --gpu-mem-util)       GPU_MEM_UTIL="$2";      shift 2 ;;
        --host)               HOST="$2";              shift 2 ;;
        --port)               PORT="$2";              shift 2 ;;
        --server-extra-args)  SERVER_EXTRA_ARGS="$2";  shift 2 ;;
        --combos)             INPUT_OUTPUT_COMBOS="$2"; shift 2 ;;
        --concurrencies)      CONCURRENCIES="$2";      shift 2 ;;
        --prompts-multiplier) PROMPTS_MULTIPLIER="$2"; shift 2 ;;
        --no-warmup)          DO_WARMUP=0;             shift ;;
        --output-dir)         OUTPUT_DIR="$2";         shift 2 ;;
        -h|--help)
            sed -n '2,/^$/{ s/^# \?//; p }' "$0"
            exit 0 ;;
        *)
            echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "Error: --model is required. Run with --help for usage." >&2
    exit 1
fi

# Auto-detect TP size from GPU count if not specified
if [[ -z "$TP_SIZE" ]]; then
    if command -v rocm-smi &>/dev/null; then
        TP_SIZE=$(rocm-smi --showid 2>/dev/null | grep -c "GPU\[" || echo 1)
    elif command -v nvidia-smi &>/dev/null; then
        TP_SIZE=$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)
    else
        TP_SIZE=1
    fi
    echo "Auto-detected TP size: $TP_SIZE"
fi

ulimit -c 0

echo "============================================================"
echo " vLLM Serving Profiler"
echo "============================================================"
echo " Model:          $MODEL"
echo " TP size:        $TP_SIZE"
echo " Max model len:  $MAX_MODEL_LEN"
echo " GPU mem util:   $GPU_MEM_UTIL"
echo " Server:         $HOST:$PORT"
echo " Token combos:   $INPUT_OUTPUT_COMBOS"
echo " Concurrencies:  $CONCURRENCIES"
echo " Prompts mult:   $PROMPTS_MULTIPLIER"
echo " Warmup:         $([ "$DO_WARMUP" -eq 1 ] && echo yes || echo no)"
echo " Output dir:     $OUTPUT_DIR"
if [[ -n "$SERVER_EXTRA_ARGS" ]]; then
    echo " Extra args:     $SERVER_EXTRA_ARGS"
fi
echo "============================================================"
echo ""

# ──────────────────────── Helpers ────────────────────────

clear_caches() {
    rm -rf ~/.triton/cache 2>/dev/null || true
    rm -rf ~/.cache/vllm 2>/dev/null || true
    rm -rf ~/.cache/comgr 2>/dev/null || true
    rm -rf /tmp/torchinductor_root 2>/dev/null || true
}

check_segfault() {
    local code=$1
    local log=${2:-}
    if [[ "$code" -eq 139 ]] || [[ "$code" -eq 134 ]] || [[ "$code" -eq 132 ]]; then
        echo "FATAL: Segfault or memory fault (exit code $code). Exiting."
        exit 1
    fi
    if [[ -n "$log" ]] && [[ -f "$log" ]]; then
        if grep -qE "Segmentation fault|memory access fault|core dumped" "$log" 2>/dev/null; then
            echo "FATAL: Segfault/memory fault detected in log. Exiting."
            exit 1
        fi
    fi
}

parse_client_output() {
    local out="$1"
    TTFT_MS=$(grep -E "Median TTFT \(ms\):|Mean TTFT \(ms\):" "$out" 2>/dev/null | head -1 | awk '{print $NF}' | tr -d '\r')
    TPOT_MS=$(grep -E "Median TPOT \(ms\):|Mean TPOT \(ms\):" "$out" 2>/dev/null | head -1 | awk '{print $NF}' | tr -d '\r')
    E2EL_MS=$(grep -E "Median E2EL \(ms\):|Mean E2EL \(ms\):" "$out" 2>/dev/null | head -1 | awk '{print $NF}' | tr -d '\r')
    OUTPUT_THROUGHPUT=$(grep "Output token throughput (tok/s):" "$out" 2>/dev/null | awk '{print $NF}' | tr -d '\r')
    TOTAL_THROUGHPUT=$(grep "Total token throughput (tok/s):" "$out" 2>/dev/null | awk '{print $NF}' | tr -d '\r')
}

start_server() {
    : > "$SERVER_LOG"
    # shellcheck disable=SC2086
    vllm serve "$MODEL" \
        --host "$HOST" \
        --port "$PORT" \
        --swap-space 64 \
        --trust-remote-code \
        --tensor-parallel-size "$TP_SIZE" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --no-enable-prefix-caching \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        $SERVER_EXTRA_ARGS \
        >> "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
}

wait_for_server() {
    SERVER_READY=
    for _ in $(seq 1 900); do
        if grep -q "Application startup complete." "$SERVER_LOG" 2>/dev/null; then
            SERVER_READY=1
            break
        fi
        if curl -sf -o /dev/null --connect-timeout 2 "http://${HOST}:${PORT}/v1/models" 2>/dev/null; then
            SERVER_READY=1
            break
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            wait "$SERVER_PID" || true
            check_segfault $? "$SERVER_LOG"
            echo "WARNING: Server exited before startup. Last 40 lines:"
            tail -40 "$SERVER_LOG" 2>/dev/null || true
            SERVER_READY=FAILED
            break
        fi
        sleep 2
    done
    if [[ -z "${SERVER_READY:-}" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        echo "WARNING: Timeout (30 min) waiting for server startup."
        tail -40 "$SERVER_LOG" 2>/dev/null || true
        SERVER_READY=FAILED
    fi
}

kill_server() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        check_segfault $? "$SERVER_LOG"
        SERVER_PID=""
    fi
}

run_client() {
    local input_tok=$1 output_tok=$2 max_conc=$3 log_file=$4
    local num_prompts=$((PROMPTS_MULTIPLIER * max_conc))
    : > "$log_file"
    set +e
    vllm bench serve \
        --host "$HOST" --port "$PORT" \
        --model "$MODEL" \
        --dataset-name random \
        --random-input-len "$input_tok" \
        --random-output-len "$output_tok" \
        --max-concurrency "$max_conc" \
        --num-prompts "$num_prompts" \
        --percentile-metrics ttft,tpot,itl,e2el \
        --ignore-eos \
        --trust-remote-code \
        >> "$log_file" 2>&1
    local exit_code=$?
    set -e
    check_segfault $exit_code "$log_file"
}

# ──────────────────────── Temp files & cleanup ────────────────────────

SERVER_LOG=$(mktemp)
CLIENT_LOG=$(mktemp)
CLIENT_LOG2=$(mktemp)
RESULTS_FILE=$(mktemp)

dump_results() {
    echo ""
    echo "============= RESULTS (human-readable) ============= "
    if [[ -f "$RESULTS_FILE" ]] && [[ $(wc -l < "$RESULTS_FILE") -gt 1 ]]; then
        printf "%6s %6s %4s %10s %10s %10s %12s %12s\n" \
            "in_tok" "out_tok" "conc" "TTFT_ms" "TPOT_ms" "E2EL_ms" "out_tok/s" "total_tok/s"
        printf "%6s %6s %4s %10s %10s %10s %12s %12s\n" \
            "------" "------" "----" "--------" "--------" "--------" "------------" "------------"
        tail -n +2 "$RESULTS_FILE" | while IFS=, read -r it ot conc ttft tpot e2el out_thr tot_thr; do
            printf "%6s %6s %4s %10s %10s %10s %12s %12s\n" \
                "$it" "$ot" "$conc" "$ttft" "$tpot" "$e2el" "$out_thr" "$tot_thr"
        done
        CSV_OUTPUT="$OUTPUT_DIR/profile_results_$(date +%Y%m%d_%H%M%S).csv"
        cp "$RESULTS_FILE" "$CSV_OUTPUT"
        echo ""
        echo "CSV saved to: $CSV_OUTPUT"
    else
        echo "(No completed runs to display)"
    fi
    echo ""
    echo "============= RESULTS (CSV) ============= "
    cat "$RESULTS_FILE" 2>/dev/null || true
    rm -f "$SERVER_LOG" "$CLIENT_LOG" "$CLIENT_LOG2" "$RESULTS_FILE"
}
trap dump_results EXIT

echo "input_tokens,output_tokens,max_concurrency,ttft_ms,tpot_ms,e2el_ms,output_throughput_tok_s,total_throughput_tok_s" > "$RESULTS_FILE"

# ──────────────────────── Main loop ────────────────────────

for combo in $INPUT_OUTPUT_COMBOS; do
    input_tok="${combo%%:*}"
    output_tok="${combo##*:}"
    for max_concurrency in $CONCURRENCIES; do
        echo "=== Config: input_tokens=$input_tok output_tokens=$output_tok max_concurrency=$max_concurrency ==="

        clear_caches
        start_server
        wait_for_server

        if [[ "${SERVER_READY}" == "FAILED" ]]; then
            echo "Skipping config and continuing..."
            echo "${input_tok},${output_tok},${max_concurrency},FAIL,FAIL,FAIL,FAIL,FAIL" >> "$RESULTS_FILE"
            continue
        fi

        if [[ "$DO_WARMUP" -eq 1 ]]; then
            echo "  Warmup run..."
            run_client "$input_tok" "$output_tok" "$max_concurrency" "$CLIENT_LOG"
        fi

        echo "  Measurement run..."
        run_client "$input_tok" "$output_tok" "$max_concurrency" "$CLIENT_LOG2"

        kill_server

        parse_client_output "$CLIENT_LOG2"
        echo "${input_tok},${output_tok},${max_concurrency},${TTFT_MS:-},${TPOT_MS:-},${E2EL_MS:-},${OUTPUT_THROUGHPUT:-},${TOTAL_THROUGHPUT:-}" >> "$RESULTS_FILE"
    done
done
