#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/data/yanguahe/code/wk_sp1/aiter}"
CONTAINER_NAME="${CONTAINER_NAME:-}"
SCRIPT_NAME="$(basename "$0")"
OUT_DIR="tdm_objdump_probe"

container_main() {
    local asm_file="${OUT_DIR}/tdm_probe.s"
    local co_file="${OUT_DIR}/tdm_probe.co"
    local objdump_file="${OUT_DIR}/tdm_probe.objdump.s"
    local summary_file="${OUT_DIR}/summary.log"

    cd "${REPO_ROOT}"

    rm -rf "${OUT_DIR}"
    mkdir -p "${OUT_DIR}"

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

    local clangxx
    local objdump
    clangxx="$(pick_tool /opt/rocm/llvm/bin/clang++ clang++)"
    objdump="$(pick_tool /opt/rocm/llvm/bin/llvm-objdump llvm-objdump)"

    cat > "${asm_file}" <<'ASM'
        .amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
        .amdhsa_code_object_version 6
        .text
        .globl  tdm_objdump_probe
        .p2align        8
        .type   tdm_objdump_probe,@function
tdm_objdump_probe:
        s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
        tensor_load_to_lds s[12:15], s[4:11]
        s_endpgm
.Lfunc_end0:
        .size   tdm_objdump_probe, .Lfunc_end0-tdm_objdump_probe
ASM

    {
        echo "date: $(date -Is)"
        echo "host: $(hostname)"
        echo "container: ${CONTAINER_NAME}"
        echo "pwd: $(pwd)"
        echo "repo_head: ${HOST_GIT_COMMIT:-unknown}"
        echo "clangxx: ${clangxx}"
        echo "objdump: ${objdump}"
        "${clangxx}" --version
        "${objdump}" --version | sed -n '1,3p'
        echo
        echo "ASM input:"
        cat "${asm_file}"
        echo
        echo "Assembling with clang++ -x assembler..."
    } 2>&1 | tee "${OUT_DIR}/build.log"

    set +e
    "${clangxx}" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -c "${asm_file}" -o "${co_file}" \
        > "${OUT_DIR}/clang_stdout.log" 2> "${OUT_DIR}/clang_stderr.log"
    local clang_status=$?
    set -e

    {
        echo "clang_status=${clang_status}"
        echo
        echo "clang stdout:"
        cat "${OUT_DIR}/clang_stdout.log"
        echo
        echo "clang stderr:"
        cat "${OUT_DIR}/clang_stderr.log"
    } 2>&1 | tee -a "${OUT_DIR}/build.log"

    local objdump_status=1
    if [[ "${clang_status}" -eq 0 && -f "${co_file}" ]]; then
        set +e
        "${objdump}" -d "${co_file}" > "${objdump_file}" 2> "${OUT_DIR}/objdump_stderr.log"
        objdump_status=$?
        set -e
    else
        : > "${objdump_file}"
        echo "skip objdump because clang failed or ${co_file} is missing" > "${OUT_DIR}/objdump_stderr.log"
    fi

    local tdm_matches=0
    if [[ -f "${objdump_file}" ]]; then
        tdm_matches="$(grep -c 'tensor_load_to_lds' "${objdump_file}" || true)"
    fi

    {
        echo "clang_status=${clang_status}"
        echo "objdump_status=${objdump_status}"
        echo "tensor_load_to_lds_matches=${tdm_matches}"
        echo
        echo "Generated files:"
        find "${OUT_DIR}" -type f -print | sort
        echo
        echo "Objdump tensor lines:"
        grep -n 'tensor_load_to_lds\|tensor_store_from_lds' "${objdump_file}" || true
    } 2>&1 | tee "${summary_file}"

    if [[ "${clang_status}" -ne 0 || "${objdump_status}" -ne 0 || "${tdm_matches}" -eq 0 ]]; then
        return 1
    fi
}

if [[ "${1:-}" == "--inside-container" ]]; then
    container_main
    exit $?
fi

if [[ -z "${CONTAINER_NAME}" ]]; then
    echo "ERROR: CONTAINER_NAME is required. Re-run with CONTAINER_NAME=<container> bash ./${SCRIPT_NAME}" >&2
    exit 2
fi

host_git_commit="$(git -C "${REPO_ROOT}" rev-parse HEAD || true)"

set +e
docker exec -i \
    -e REPO_ROOT="${REPO_ROOT}" \
    -e CONTAINER_NAME="${CONTAINER_NAME}" \
    -e HOST_GIT_COMMIT="${host_git_commit}" \
    "${CONTAINER_NAME}" \
    bash -lc "cd '${REPO_ROOT}' && bash './${SCRIPT_NAME}' --inside-container"
probe_status=$?
set -e

git -C "${REPO_ROOT}" add -f "${SCRIPT_NAME}" "${OUT_DIR}"
if ! git -C "${REPO_ROOT}" diff --cached --quiet; then
    git -C "${REPO_ROOT}" -c user.name=yanguahe -c user.email=yanguahe@amd.com \
        commit --amend --author="yanguahe <yanguahe@amd.com>" -m Update
fi
git -C "${REPO_ROOT}" push -f origin hyg_dev

exit "${probe_status}"
