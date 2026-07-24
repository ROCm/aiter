#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/data/yanguahe/code/wk_sp1/aiter}"
CONTAINER_NAME="${CONTAINER_NAME:-hyg_fyd1}"
SCRIPT_NAME="$(basename "$0")"
OUT_DIR="perf_logs/mha_three_isa_dump"

container_main() {
    local log_dir="${OUT_DIR}/logs"
    local flydsl_dump_dir="./flydsl_dump"
    local flydsl_out="${OUT_DIR}/flydsl_192x128"
    local asm_bf16_out="${OUT_DIR}/asm_bf16_mask_128x128"
    local asm_mxfp8_out="${OUT_DIR}/asm_mxfp8_128x128"

    cd "${REPO_ROOT}"
    rm -rf "${OUT_DIR}" "${flydsl_dump_dir}"
    mkdir -p "${log_dir}" "${flydsl_out}" "${asm_bf16_out}" "${asm_mxfp8_out}"

    pick_tool() {
        local tool
        for tool in "$@"; do
            if [[ -x "${tool}" ]]; then
                echo "${tool}"
                return 0
            fi
            if command -v "${tool}" >/dev/null 2>&1; then
                command -v "${tool}"
                return 0
            fi
        done
        return 1
    }

    local objdump
    objdump="$(pick_tool /opt/rocm/llvm/bin/llvm-objdump llvm-objdump roc-objdump)"

    {
        echo "date: $(date -Is)"
        echo "host: $(hostname)"
        echo "container: ${CONTAINER_NAME}"
        echo "pwd: $(pwd)"
        echo "host_git_commit: ${HOST_GIT_COMMIT:-unknown}"
        echo "python: $(command -v python || true)"
        python --version || true
        echo "objdump: ${objdump}"
        "${objdump}" --version | sed -n '1,3p' || true
    } 2>&1 | tee "${log_dir}/environment.log"

    run_case() {
        local name="$1"
        local log_file="$2"
        shift 2
        echo "Running ${name}: $*" | tee "${log_file}"
        set +e
        "$@" 2>&1 | tee -a "${log_file}"
        local status=${PIPESTATUS[0]}
        set -e
        echo "${name} exit status: ${status}" | tee -a "${log_file}"
        return "${status}"
    }

    set +e
    rm -rf "${flydsl_dump_dir}"
    run_case \
        "flydsl-bf16-192x128-causal" \
        "${log_dir}/01_flydsl_bf16_192x128.log" \
        env FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR="${flydsl_dump_dir}" \
            python op_tests/test_mha_flydsl_varlen.py \
            --causal true --return_lse true \
            -b 1 -nh 64 -sq 8192 -sk 8192 \
            --random-value false --warmup 5 --repeat 20
    flydsl_status=$?
    set -e
    if [[ -d "${flydsl_dump_dir}" ]]; then
        cp -a "${flydsl_dump_dir}/." "${flydsl_out}/"
        find "${flydsl_out}" -type f -print | sort > "${flydsl_out}/files.txt"
    fi

    set +e
    run_case \
        "asm-bf16-mask-128x128-causal" \
        "${log_dir}/02_asm_bf16_mask_128x128.log" \
        python op_tests/test_mha_flydsl_varlen.py \
            --causal true --return_lse true -d_qk_v 128,128 \
            -b 1 -nh 64 -sq 8192 -sk 8192 \
            --random-value false --warmup 5 --repeat 20
    asm_bf16_status=$?
    set -e

    set +e
    run_case \
        "asm-mxfp8-128x128-noncausal" \
        "${log_dir}/03_asm_mxfp8_128x128.log" \
        python op_tests/test_mha_flydsl_varlen.py \
            --dtype mxfp8 --causal false --return_lse true -d_qk_v 128,128 \
            -b 1 -nh 64 -sq 8192 -sk 8192 \
            --random-value false --warmup 5 --repeat 20
    asm_mxfp8_status=$?
    set -e

    local bf16_co="hsa/gfx1250/fmha_fwd_bf16_varlen/fmha_bf16_pertokenBf16_hd128_128x256_mask_varlen.co"
    local bf16_csv="hsa/gfx1250/fmha_fwd_bf16_varlen/fmha_fwd_bf16_varlen.csv"
    local mxfp8_co="hsa/gfx1250/fmha_fwd_mxfp8/fmha_fwd_mxfp8_d128.co"
    local mxfp8_csv="hsa/gfx1250/fmha_fwd_mxfp8/fmha_fwd_mxfp8.csv"

    cp "${bf16_csv}" "${asm_bf16_out}/"
    cp "${bf16_co}" "${asm_bf16_out}/"
    set +e
    "${objdump}" -d "${bf16_co}" > "${asm_bf16_out}/fmha_bf16_pertokenBf16_hd128_128x256_mask_varlen.isa.s" \
        2> "${asm_bf16_out}/objdump.stderr"
    bf16_objdump_status=$?
    set -e

    cp "${mxfp8_csv}" "${asm_mxfp8_out}/"
    cp "${mxfp8_co}" "${asm_mxfp8_out}/"
    set +e
    "${objdump}" -d "${mxfp8_co}" > "${asm_mxfp8_out}/fmha_fwd_mxfp8_d128.isa.s" \
        2> "${asm_mxfp8_out}/objdump.stderr"
    mxfp8_objdump_status=$?
    set -e

    {
        echo "flydsl_status=${flydsl_status}"
        echo "asm_bf16_status=${asm_bf16_status}"
        echo "asm_mxfp8_status=${asm_mxfp8_status}"
        echo "bf16_objdump_status=${bf16_objdump_status}"
        echo "mxfp8_objdump_status=${mxfp8_objdump_status}"
        echo
        echo "Expected kernels:"
        echo "flydsl: ${flydsl_out}/fmha_fwd_kernel_*/21_final_isa.s"
        echo "asm_bf16: ${bf16_co}"
        echo "asm_mxfp8: ${mxfp8_co}"
        echo
        echo "Loaded kernel lines:"
        grep -h "LoadKernel" "${log_dir}"/*.log || true
        echo
        echo "Collected files:"
        find "${OUT_DIR}" -type f -print | sort
    } 2>&1 | tee "${OUT_DIR}/summary.log"

    if [[ "${flydsl_status}" -ne 0 || "${asm_bf16_status}" -ne 0 || "${asm_mxfp8_status}" -ne 0 ||
          "${bf16_objdump_status}" -ne 0 || "${mxfp8_objdump_status}" -ne 0 ]]; then
        return 1
    fi
}

if [[ "${1:-}" == "--inside-container" ]]; then
    container_main
    exit $?
fi

host_git_commit="$(git -C "${REPO_ROOT}" rev-parse HEAD || true)"

set +e
docker exec -i \
    -e REPO_ROOT="${REPO_ROOT}" \
    -e CONTAINER_NAME="${CONTAINER_NAME}" \
    -e HOST_GIT_COMMIT="${host_git_commit}" \
    "${CONTAINER_NAME}" \
    bash -lc "cd '${REPO_ROOT}' && bash './${SCRIPT_NAME}' --inside-container"
run_status=$?
set -e

git -C "${REPO_ROOT}" add -f "${SCRIPT_NAME}" "${OUT_DIR}"
if ! git -C "${REPO_ROOT}" diff --cached --quiet; then
    git -C "${REPO_ROOT}" -c user.name=yanguahe -c user.email=yanguahe@amd.com \
        commit --amend --author="yanguahe <yanguahe@amd.com>" -m Update
fi
git -C "${REPO_ROOT}" push -f origin hyg_dev

exit "${run_status}"
