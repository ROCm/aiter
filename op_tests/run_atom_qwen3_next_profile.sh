#!/usr/bin/env bash
# Drive ATOM's offline profiler (atom/examples/profile_offline.py) for each
# of the three SplitK zero-init demo configs (none / splitk / splitk_fused),
# capturing a torch.profiler chrome trace per mode for a small batch=1
# generation.  The companion ``parse_atom_traces.py`` then tallies fill
# kernels and per-iteration GPU time across modes -- this isolates the
# fusion impact from the openai_server / benchmark_serving harness noise
# that drowns the ~0.2 ms TPOT delta in the e2e demo.
#
# Usage:
#   bash op_tests/run_atom_qwen3_next_profile.sh [<results_dir>]
set -euo pipefail

AITER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ATOM_ROOT="${ATOM_ROOT:-${HOME}/dev/ATOM}"
RESULTS_DIR="${1:-${AITER_ROOT}/op_tests/zero_init_demo_results/atom_qwen3_next_profile_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${RESULTS_DIR}"

NOSPLITK_CSV="${AITER_ROOT}/aiter/configs/zero_init_demo/robust/qwen3_next_80b_a3b_per1x128_cktile_nosplitk_gfx950.csv"
SPLITK_CSV="${AITER_ROOT}/aiter/configs/zero_init_demo/robust/qwen3_next_80b_a3b_per1x128_cktile_splitk_yz_gfx950.csv"

MODEL="${MODEL:-Qwen/Qwen3-Next-80B-A3B-Instruct-FP8}"
TP_SIZE="${TP_SIZE:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"

# Small batch + short output keeps the chrome trace tractable
# (a single decode iteration is enough since CUDA-graph replay produces
# identical kernel sequences across iterations).
BS="${BS:-1}"
INPUT_LEN="${INPUT_LEN:-128}"
OUTPUT_LEN="${OUTPUT_LEN:-32}"

PYTHONPATH_BASE="${HOME}/dev/aiter:${HOME}/dev/ATOM${PYTHONPATH:+:${PYTHONPATH}}"

# Triton + gfx950 buffer-ops workaround (same as run_atom_qwen3_next_demo.sh).
export AMDGCN_USE_BUFFER_OPS="${AMDGCN_USE_BUFFER_OPS:-0}"

JIT_DIR="${AITER_ROOT}/aiter/jit"
SO_FILE="${JIT_DIR}/module_gemm_a8w8_blockscale_bpreshuffle_cktile.so"
BUILD_DIR="${JIT_DIR}/build/module_gemm_a8w8_blockscale_bpreshuffle_cktile"
# vLLM's torch.compile cache lives under ~/.cache/atom/torch_compile_cache.
# Its cache key is keyed off model + shapes + ops (NOT off
# ATOM_BLOCKSCALE_SPLITK_MODE), so a graph compiled in "none" mode -- where
# LinearBase.preallocate_y returns None and the fusion branch traces
# as dead code -- gets re-used unmodified in "splitk_fused" mode and
# silently keeps y_is_zeroed=False / out=None in every per_1x128 GEMM
# call.  Until that cache key is fixed upstream we nuke the cache
# between modes so each mode starts from a fresh recompile (~22 s).
ATOM_COMPILE_CACHE="${HOME}/.cache/atom/torch_compile_cache"
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

nuke_torch_compile_cache() {
    if [[ -d "${ATOM_COMPILE_CACHE}" ]]; then
        echo "# nuking torch.compile cache at ${ATOM_COMPILE_CACHE}"
        rm -rf "${ATOM_COMPILE_CACHE}"
    fi
}

run_mode() {
    local mode="$1"
    local csv="$2"
    local trace_dir="${RESULTS_DIR}/${mode}_traces"
    local stdout_log="${RESULTS_DIR}/${mode}.log"
    mkdir -p "${trace_dir}"

    echo "=========================================================="
    echo "# mode=${mode} csv=${csv}"
    echo "# trace dir -> ${trace_dir}"
    echo "# log       -> ${stdout_log}"
    echo "=========================================================="

    nuke_module_if_csv_changed "${csv}"
    nuke_torch_compile_cache

    # Launch the offline profiler in its own session/process group
    # (setsid) so we can kill the whole tree atomically -- ATOM spawns
    # ModelRunner / EngineCore / inductor workers that don't always
    # exit cleanly when the parent returns from llm.stop_profile() and
    # would otherwise hold shm + VRAM through the next mode.
    local pgleader_pid
    setsid bash -c "
        cd \"${ATOM_ROOT}\"
        exec env \
            PYTHONPATH=\"${PYTHONPATH_BASE}\" \
            ATOM_BLOCKSCALE_SPLITK_MODE=\"${mode}\" \
            ATOM_BLOCKSCALE_BPRESHUFFLE_TUNED_NOSPLITK_CSV=\"${NOSPLITK_CSV}\" \
            ATOM_BLOCKSCALE_BPRESHUFFLE_TUNED_SPLITK_CSV=\"${SPLITK_CSV}\" \
            AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE=\"${csv}\" \
            python -m atom.examples.profile_offline \
                --model \"${MODEL}\" \
                --tensor-parallel-size \"${TP_SIZE}\" \
                --gpu-memory-utilization \"${GPU_MEM_UTIL}\" \
                --bs \"${BS}\" \
                --input-length \"${INPUT_LEN}\" \
                --output-length \"${OUTPUT_LEN}\" \
                --random-input \
                --torch-profiler-dir \"${trace_dir}\" \
                >\"${stdout_log}\" 2>&1
    " &
    pgleader_pid=$!
    echo "# pgid leader pid: ${pgleader_pid}"
    trap "kill -KILL -${pgleader_pid} 2>/dev/null || true" EXIT

    # Poll for the trace file to appear -- once the profiler has flushed
    # to disk we have what we need; we don't care if the engine then
    # struggles to tear itself down (we kill it forcibly below).
    local deadline=$(( SECONDS + 1500 ))
    local trace_done=0
    while (( SECONDS < deadline )); do
        if compgen -G "${trace_dir}/rank_*/*.json*" >/dev/null 2>&1 \
           || compgen -G "${trace_dir}/*.json*" >/dev/null 2>&1; then
            # File exists; wait briefly for size to stabilize so we
            # don't grab a half-flushed trace.
            local sz1 sz2
            sz1=$(du -sb "${trace_dir}" 2>/dev/null | awk '{print $1}')
            sleep 5
            sz2=$(du -sb "${trace_dir}" 2>/dev/null | awk '{print $1}')
            if [[ "${sz1}" == "${sz2}" ]]; then
                trace_done=1
                break
            fi
        fi
        if ! kill -0 "${pgleader_pid}" 2>/dev/null; then
            # Process exited (clean or otherwise). Give the FS a beat
            # then break -- we either have the trace or we don't.
            sleep 2
            if compgen -G "${trace_dir}/rank_*/*.json*" >/dev/null 2>&1 \
               || compgen -G "${trace_dir}/*.json*" >/dev/null 2>&1; then
                trace_done=1
            fi
            break
        fi
        sleep 5
    done

    if (( trace_done )); then
        echo "# trace captured for mode=${mode}; killing pgroup"
    else
        echo "# WARNING: no trace produced for mode=${mode} within timeout" >&2
    fi

    # Force-kill the whole pgroup so subsequent modes start clean.
    kill -KILL -"${pgleader_pid}" 2>/dev/null || true
    sleep 2

    # Wait for VRAM release before the next mode starts loading weights.
    for _i in $(seq 1 30); do
        local vram_used_mib=0
        vram_used_mib=$(rocm-smi --showmeminfo vram 2>/dev/null \
            | awk '/VRAM Total Used Memory \(B\)/ {print int($NF/1024/1024)}' \
            | head -1)
        vram_used_mib=${vram_used_mib:-0}
        if (( vram_used_mib < 2048 )); then
            break
        fi
        sleep 2
    done
    trap - EXIT
}

run_mode none         "${NOSPLITK_CSV}"
run_mode splitk       "${SPLITK_CSV}"
run_mode splitk_fused "${SPLITK_CSV}"

echo
echo "=========================================================="
echo "# Parsing traces for fill / gemm counts:"
echo "=========================================================="
PYTHONPATH="${PYTHONPATH_BASE}" \
python "${AITER_ROOT}/op_tests/parse_atom_traces.py" \
    --results-dir "${RESULTS_DIR}" \
    --output-len "${OUTPUT_LEN}"

echo
echo "# Artifacts saved under: ${RESULTS_DIR}"
