#!/usr/bin/env bash
# End-to-end Qwen3-Next-80B-A3B demo for the SplitK zero-init fusion.
#
# For each of the three configs (none / splitk / splitk_fused) this:
#   1. (Re)builds module_gemm_a8w8_blockscale_bpreshuffle_cktile against
#      the matching tuned CSV when the CSV changes.
#   2. Launches `atom.entrypoints.openai_server` with -tp 1 in the
#      background and waits for /health.
#   3. Runs atom.benchmarks.benchmark_serving against it.
#   4. Kills the server and saves the bench result to a per-config
#      JSON under the timestamped results directory.
#
# Usage:
#   bash op_tests/run_atom_qwen3_next_demo.sh [<results_dir>]
set -euo pipefail

AITER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ATOM_ROOT="${ATOM_ROOT:-${HOME}/dev/ATOM}"
RESULTS_DIR="${1:-${AITER_ROOT}/op_tests/zero_init_demo_results/atom_qwen3_next_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${RESULTS_DIR}"

NOSPLITK_CSV="${AITER_ROOT}/aiter/configs/zero_init_demo/robust/qwen3_next_80b_a3b_per1x128_cktile_nosplitk_gfx950.csv"
SPLITK_CSV="${AITER_ROOT}/aiter/configs/zero_init_demo/robust/qwen3_next_80b_a3b_per1x128_cktile_splitk_yz_gfx950.csv"

MODEL="${MODEL:-Qwen/Qwen3-Next-80B-A3B-Instruct-FP8}"
SERVER_PORT="${SERVER_PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-4}"
INPUT_LEN="${INPUT_LEN:-2048}"
OUTPUT_LEN="${OUTPUT_LEN:-1024}"
TP_SIZE="${TP_SIZE:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"

PYTHONPATH_BASE="${HOME}/dev/aiter:${HOME}/dev/ATOM${PYTHONPATH:+:${PYTHONPATH}}"
SERVER_BOOT_TIMEOUT="${SERVER_BOOT_TIMEOUT:-1200}"   # seconds (80B FP8 weight load)

# Workaround for a triton 3.6.0+rocm7.2.2 + gfx950 bug: the AMD backend's
# `TritonAMDGPUCanonicalizePointers` pass crashes (`PassManager::run failed`)
# on ATOM's mamba/GDN conv1d kernels (`_causal_conv1d_update_kernel` and
# `_causal_conv1d_fwd_kernel`). The pass only runs when AMD buffer-ops
# codegen is enabled; setting AMDGCN_USE_BUFFER_OPS=0 short-circuits the
# `if knobs.amd.use_buffer_ops:` block at
# /opt/venv/lib/python3.12/site-packages/triton/backends/amd/compiler.py:250
# and lets the kernels lower via plain `tt.load`/`tt.store` instead of
# `buffer_load`/`buffer_store`. The knob default is 1 (see
# /opt/venv/lib/python3.12/site-packages/triton/knobs.py:508).
#
# Risk: this disables AMD buffer-ops codegen for ALL triton kernels in the
# run, costing some perf on aiter/ATOM triton ops (e.g. fused MoE). The
# FP8 blockscale GEMM that this demo measures is CK/CKTile in C++ (not a
# triton kernel) so the zero-init fusion timings are unaffected.
export AMDGCN_USE_BUFFER_OPS="${AMDGCN_USE_BUFFER_OPS:-0}"
echo "# AMDGCN_USE_BUFFER_OPS=${AMDGCN_USE_BUFFER_OPS} (workaround for triton mamba conv1d crash)"

JIT_DIR="${AITER_ROOT}/aiter/jit"
SO_FILE="${JIT_DIR}/module_gemm_a8w8_blockscale_bpreshuffle_cktile.so"
BUILD_DIR="${JIT_DIR}/build/module_gemm_a8w8_blockscale_bpreshuffle_cktile"

LAST_CSV=""

nuke_module_if_csv_changed() {
    local new_csv="$1"
    if [[ "${new_csv}" != "${LAST_CSV}" ]]; then
        echo "# CSV changed (${LAST_CSV} -> ${new_csv}); nuking bpreshuffle CKTile .so"
        rm -f "${SO_FILE}"
        rm -rf "${BUILD_DIR}"
        LAST_CSV="${new_csv}"
    else
        echo "# CSV unchanged (${new_csv}); reusing existing .so"
    fi
}

wait_for_server() {
    local pid="$1"
    local deadline=$(( SECONDS + SERVER_BOOT_TIMEOUT ))
    while (( SECONDS < deadline )); do
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "ERROR: server PID ${pid} died during boot" >&2
            return 1
        fi
        if curl -fsS "http://${HOST}:${SERVER_PORT}/health" >/dev/null 2>&1; then
            echo "# server up after $(( SECONDS - (deadline - SERVER_BOOT_TIMEOUT) ))s"
            return 0
        fi
        sleep 5
    done
    echo "ERROR: server did not become ready within ${SERVER_BOOT_TIMEOUT}s" >&2
    return 1
}

run_config() {
    local mode="$1"
    local csv="$2"
    local result_json="${RESULTS_DIR}/${mode}.json"
    local server_log="${RESULTS_DIR}/${mode}.server.log"
    local bench_log="${RESULTS_DIR}/${mode}.bench.log"
    echo "=========================================================="
    echo "# config: mode=${mode} csv=${csv}"
    echo "# server log -> ${server_log}"
    echo "# bench log  -> ${bench_log}"
    echo "# result     -> ${result_json}"
    echo "=========================================================="

    nuke_module_if_csv_changed "${csv}"

    local server_pid
    # Launch the server in its own session/process group via setsid + exec so
    # we can kill the whole tree (the openai_server python + its mp.spawn
    # children + inductor workers) atomically with `kill -- -<pgid>`.  Without
    # this, killing just the parent leaves children re-parented to PID 1 and
    # they keep VRAM allocated, OOM'ing the next mode's weight load.
    setsid bash -c "
        cd \"${ATOM_ROOT}\"
        exec env \
            PYTHONPATH=\"${PYTHONPATH_BASE}\" \
            ATOM_BLOCKSCALE_SPLITK_MODE=\"${mode}\" \
            ATOM_BLOCKSCALE_BPRESHUFFLE_TUNED_NOSPLITK_CSV=\"${NOSPLITK_CSV}\" \
            ATOM_BLOCKSCALE_BPRESHUFFLE_TUNED_SPLITK_CSV=\"${SPLITK_CSV}\" \
            AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE=\"${csv}\" \
            python -m atom.entrypoints.openai_server \
                --model \"${MODEL}\" \
                --tensor-parallel-size \"${TP_SIZE}\" \
                --host \"${HOST}\" --server-port \"${SERVER_PORT}\" \
                --gpu-memory-utilization \"${GPU_MEM_UTIL}\" \
                >\"${server_log}\" 2>&1
    " &
    server_pid=$!
    echo "# server pid (pgid leader): ${server_pid}"
    # The pgid equals the leader PID because we used setsid.
    trap "kill -TERM -${server_pid} 2>/dev/null || true" EXIT

    if ! wait_for_server "${server_pid}"; then
        kill -TERM -"${server_pid}" 2>/dev/null || true
        sleep 3
        kill -KILL -"${server_pid}" 2>/dev/null || true
        wait "${server_pid}" 2>/dev/null || true
        trap - EXIT
        return 1
    fi

    cd "${ATOM_ROOT}"
    PYTHONPATH="${PYTHONPATH_BASE}" \
    python -m atom.benchmarks.benchmark_serving \
        --model="${MODEL}" --backend=vllm \
        --base-url="http://${HOST}:${SERVER_PORT}" \
        --dataset-name=random \
        --random-input-len="${INPUT_LEN}" \
        --random-output-len="${OUTPUT_LEN}" \
        --random-range-ratio 1.0 \
        --num-prompts="${NUM_PROMPTS}" \
        --max-concurrency="${MAX_CONCURRENCY}" \
        --request-rate=inf --ignore-eos \
        --save-result --percentile-metrics="ttft,tpot,itl,e2el" \
        --result-dir="${RESULTS_DIR}" \
        --result-filename="${mode}.json" \
        2>&1 | tee "${bench_log}"
    cd - >/dev/null

    echo "# stopping server pgid ${server_pid}"
    kill -TERM -"${server_pid}" 2>/dev/null || true
    # Give the server up to 30 s for graceful shutdown of the whole pgroup.
    for _i in $(seq 1 30); do
        if ! kill -0 -"${server_pid}" 2>/dev/null; then
            break
        fi
        sleep 1
    done
    # Force-kill anything lingering in the pgroup.  Because we launched with
    # setsid, this catches the openai_server python + all its mp.spawn /
    # inductor workers atomically without risk of matching unrelated processes.
    kill -KILL -"${server_pid}" 2>/dev/null || true
    sleep 3
    # Wait until both the port AND VRAM are actually released before the next
    # iteration tries to bind / load weights.  VRAM is the critical one: the
    # 80B FP8 weights need ~80 GB and lingering allocations will HIP-OOM.
    for _i in $(seq 1 30); do
        local port_busy=0
        local vram_used_mib=0
        if ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":${SERVER_PORT}\$"; then
            port_busy=1
        fi
        vram_used_mib=$(rocm-smi --showmeminfo vram 2>/dev/null \
            | awk '/VRAM Total Used Memory \(B\)/ {print int($NF/1024/1024)}' \
            | head -1)
        vram_used_mib=${vram_used_mib:-0}
        if (( port_busy == 0 && vram_used_mib < 2048 )); then
            break
        fi
        sleep 2
    done
    trap - EXIT
}

run_config none         "${NOSPLITK_CSV}"
run_config splitk       "${SPLITK_CSV}"
run_config splitk_fused "${SPLITK_CSV}"

echo
echo "=========================================================="
echo "# Summary across configs:"
echo "=========================================================="
python - <<PY
import json, os, glob
results_dir = "${RESULTS_DIR}"
for mode in ("none", "splitk", "splitk_fused"):
    path = os.path.join(results_dir, f"{mode}.json")
    if not os.path.exists(path):
        print(f"{mode}: (missing)")
        continue
    with open(path) as f:
        d = json.load(f)
    keys = ("request_throughput", "output_throughput", "median_ttft_ms",
            "median_tpot_ms", "median_itl_ms", "median_e2el_ms")
    line = f"{mode:>14} | "
    for k in keys:
        v = d.get(k)
        if isinstance(v, (int, float)):
            line += f"{k}={v:8.2f}  "
        else:
            line += f"{k}=n/a       "
    print(line)
PY

echo
echo "# All artifacts saved under: ${RESULTS_DIR}"
