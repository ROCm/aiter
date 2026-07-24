#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/data/yanguahe/code/wk_sp1/aiter}"
CONTAINER_NAME="${CONTAINER_NAME:-hyg_fyd1}"
SCRIPT_NAME="$(basename "$0")"
OUT_DIR="perf_logs/mha_isa_dump"

container_main() {
    local log_dir="${OUT_DIR}/logs"
    local flydsl_out_dir="${OUT_DIR}/flydsl_kernel"
    local asm_out_dir="${OUT_DIR}/asm_kernel"
    local flydsl_dump_dir="./flydsl_dump"

    local flydsl_cmd=(
        python op_tests/test_mha_flydsl_varlen.py
        --causal true
        --return_lse true
        -b 1
        -nh 64
        -sq 8192
        -sk 8192
        --random-value false
        --warmup 5
        --repeat 20
    )

    local asm_cmd=(
        python op_tests/test_mha_flydsl_varlen.py
        --causal true
        --return_lse true
        -d_qk_v 128,128
        -b 1
        -nh 64
        -sq 8192
        -sk 8192
        --random-value false
        --warmup 5
        --repeat 20
    )

    local asm_co_name="fmha_bf16_pertokenBf16_hd128_128x256_varlen.co"
    local asm_csv="hsa/gfx1250/fmha_fwd_bf16_varlen/fmha_fwd_bf16_varlen.csv"

    cd "${REPO_ROOT}"

    rm -rf "${OUT_DIR}"
    mkdir -p "${log_dir}" "${flydsl_out_dir}" "${asm_out_dir}"

    {
        echo "date: $(date -Is)"
        echo "host: $(hostname)"
        echo "container: ${CONTAINER_NAME}"
        echo "pwd: $(pwd)"
        echo "host_git_branch: ${HOST_GIT_BRANCH:-unknown}"
        echo "host_git_commit: ${HOST_GIT_COMMIT:-unknown}"
        echo "python: $(command -v python || true)"
        python --version || true
        echo
        echo "rocm tools:"
        for tool in /opt/rocm/llvm/bin/llvm-objdump llvm-objdump roc-objdump /opt/rocm/llvm/bin/llvm-readobj llvm-readobj; do
            if [[ -x "${tool}" ]] || command -v "${tool}" >/dev/null 2>&1; then
                echo "  ${tool}: available"
            fi
        done
    } 2>&1 | tee "${log_dir}/environment.log"

    run_with_flydsl_dump() {
        local name="$1"
        local log_file="$2"
        shift 2

        rm -rf "${flydsl_dump_dir}"
        mkdir -p "${flydsl_dump_dir}"

        echo "Running ${name}: $*" | tee "${log_file}"
        set +e
        FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR="${flydsl_dump_dir}" "$@" 2>&1 | tee -a "${log_file}"
        local status=${PIPESTATUS[0]}
        set -e

        {
            echo
            echo "${name} exit status: ${status}"
            echo
            echo "Dump files under ${flydsl_dump_dir}:"
            find "${flydsl_dump_dir}" -type f -print | sort
        } 2>&1 | tee -a "${log_file}"

        return "${status}"
    }

    set +e
    run_with_flydsl_dump "flydsl-varlen-mha" "${log_dir}/flydsl_test.log" "${flydsl_cmd[@]}"
    local flydsl_status=$?
    set -e

    if [[ -d "${flydsl_dump_dir}" ]]; then
        cp -a "${flydsl_dump_dir}/." "${flydsl_out_dir}/"
        find "${flydsl_out_dir}" -type f -print | sort > "${flydsl_out_dir}/files.txt"
    fi

    set +e
    run_with_flydsl_dump "asm-varlen-mha" "${log_dir}/asm_test.log" "${asm_cmd[@]}"
    local asm_status=$?
    set -e

    if [[ -d "${flydsl_dump_dir}" ]]; then
        cp -a "${flydsl_dump_dir}/." "${asm_out_dir}/flydsl_dump_from_asm_run/"
        find "${asm_out_dir}/flydsl_dump_from_asm_run" -type f -print | sort > "${asm_out_dir}/flydsl_dump_files.txt"
    fi

    pick_objdump() {
        for tool in /opt/rocm/llvm/bin/llvm-objdump llvm-objdump roc-objdump; do
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

    pick_readobj() {
        for tool in /opt/rocm/llvm/bin/llvm-readobj llvm-readobj; do
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

    find_asm_co() {
        python - <<'PY'
import glob
import os

name = "fmha_bf16_pertokenBf16_hd128_128x256_varlen.co"
roots = [os.getcwd()]

try:
    import aiter

    aiter_file = os.path.abspath(aiter.__file__)
    roots.extend(
        [
            os.path.dirname(aiter_file),
            os.path.dirname(os.path.dirname(aiter_file)),
        ]
    )
except Exception as exc:
    print(f"# unable to import aiter while searching code object: {exc}")

seen = set()
for root in roots:
    if not root or root in seen or not os.path.isdir(root):
        continue
    seen.add(root)
    for path in glob.glob(os.path.join(root, "**", name), recursive=True):
        print(os.path.abspath(path))
PY
    }

    local asm_dump_status=0
    {
        echo "Expected ASM registry: ${asm_csv}"
        if [[ -f "${asm_csv}" ]]; then
            cp "${asm_csv}" "${asm_out_dir}/"
            echo
            echo "Registry contents:"
            cat "${asm_csv}"
        else
            echo "Missing registry: ${asm_csv}"
        fi

        echo
        echo "Searching for ${asm_co_name}:"
        find_asm_co | tee "${asm_out_dir}/co_candidates.txt"
    } 2>&1 | tee "${log_dir}/asm_code_object.log"

    local asm_co_path
    asm_co_path="$(grep -v '^#' "${asm_out_dir}/co_candidates.txt" | sed '/^[[:space:]]*$/d' | head -n 1 || true)"
    if [[ -n "${asm_co_path}" && -f "${asm_co_path}" ]]; then
        cp "${asm_co_path}" "${asm_out_dir}/${asm_co_name}"
        if objdump_tool="$(pick_objdump)"; then
            echo "Using objdump: ${objdump_tool}" | tee -a "${log_dir}/asm_code_object.log"
            set +e
            "${objdump_tool}" -d "${asm_co_path}" > "${asm_out_dir}/${asm_co_name%.co}.isa.s" 2> "${asm_out_dir}/${asm_co_name%.co}.objdump.stderr"
            asm_dump_status=$?
            set -e
            if [[ "${asm_dump_status}" -ne 0 ]]; then
                echo "objdump failed with status ${asm_dump_status}" | tee -a "${log_dir}/asm_code_object.log"
            fi
        else
            echo "No llvm-objdump/roc-objdump found; ASM .co copied but ISA was not disassembled." | tee -a "${log_dir}/asm_code_object.log"
            asm_dump_status=1
        fi

        if readobj_tool="$(pick_readobj)"; then
            "${readobj_tool}" --file-headers --notes "${asm_co_path}" > "${asm_out_dir}/${asm_co_name%.co}.readobj.txt" 2>&1 || true
        fi
    else
        echo "Unable to locate ${asm_co_name}; ASM ISA was not dumped." | tee -a "${log_dir}/asm_code_object.log"
        asm_dump_status=1
    fi

    {
        echo "flydsl_status=${flydsl_status}"
        echo "asm_status=${asm_status}"
        echo "asm_dump_status=${asm_dump_status}"
        echo
        echo "Collected files:"
        find "${OUT_DIR}" -type f -print | sort
    } 2>&1 | tee "${OUT_DIR}/summary.log"

    if [[ "${flydsl_status}" -ne 0 || "${asm_status}" -ne 0 || "${asm_dump_status}" -ne 0 ]]; then
        return 1
    fi
}

if [[ "${1:-}" == "--inside-container" ]]; then
    container_main
    exit $?
fi

host_git_branch="$(git -C "${REPO_ROOT}" branch --show-current || true)"
host_git_commit="$(git -C "${REPO_ROOT}" rev-parse HEAD || true)"

set +e
docker exec -i \
    -e REPO_ROOT="${REPO_ROOT}" \
    -e CONTAINER_NAME="${CONTAINER_NAME}" \
    -e HOST_GIT_BRANCH="${host_git_branch}" \
    -e HOST_GIT_COMMIT="${host_git_commit}" \
    "${CONTAINER_NAME}" \
    bash -lc "cd '${REPO_ROOT}' && bash './${SCRIPT_NAME}' --inside-container"
container_status=$?
set -e

git -C "${REPO_ROOT}" add -f "${SCRIPT_NAME}" "${OUT_DIR}" 2>/dev/null || git -C "${REPO_ROOT}" add -f "${SCRIPT_NAME}"
if ! git -C "${REPO_ROOT}" diff --cached --quiet; then
    git -C "${REPO_ROOT}" -c user.name=yanguahe -c user.email=yanguahe@amd.com \
        commit --amend --author="yanguahe <yanguahe@amd.com>" -m Update
fi
git -C "${REPO_ROOT}" push -f origin hyg_dev

exit "${container_status}"
