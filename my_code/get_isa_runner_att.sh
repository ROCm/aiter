#!/usr/bin/env bash
set -Eeuo pipefail

workspace_dir="${workspace_dir:-/data/yanguahe/code/wk_sp1}"
REPO_ROOT="${REPO_ROOT:-${workspace_dir}/aiter}"
CONTAINER_NAME="${CONTAINER_NAME:-hyg_fyd1}"
SCRIPT_NAME="$(basename "$0")"
SCRIPT_RELATIVE_PATH="${SCRIPT_RELATIVE_PATH:-my_code/${SCRIPT_NAME}}"
TRACE_ROOT="${TRACE_ROOT:-my_code}"
HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"

usage() {
    echo "Usage: ${SCRIPT_NAME} <KERNEL_NAME> <output-dir-name> <TEST_CMD> [--all-simd] [--git] [--am]" >&2
    echo "Example: bash my_code/${SCRIPT_NAME} moe_gemm1_a8w4 isa_runner_att 'python my_code/isa_runner/tdm_adapter.py replay --which gemm1 --iters 100 --isa ./my_code/moe_gemm1_a8w4.v0.s'" >&2
    echo "  --all-simd  run four ATT captures for SIMD0, SIMD1, SIMD2, and SIMD3" >&2
    echo "  --git  only add/commit/push an existing trace directory; skip trace collection" >&2
    echo "  --am   amend the current commit; requires --git" >&2
}

validate_kernel_name() {
    local kernel_name="$1"
    local kernel_name_re='^[A-Za-z_.$][A-Za-z0-9_.$@:+-]*$'

    if [[ -z "${kernel_name}" || "${kernel_name}" == *$'\n'* ||
          "${kernel_name}" == *$'\r'* || ! "${kernel_name}" =~ ${kernel_name_re} ]]; then
        usage
        echo "KERNEL_NAME must be a non-empty C/AMDGPU-style symbol using letters, numbers, _, ., $, @, :, +, or -" >&2
        return 1
    fi
}

escape_kernel_regex() {
    local value="$1"
    value="${value//./\\.}"
    value="${value//\$/\\$}"
    value="${value//+/\\+}"
    printf '%s' "${value}"
}

validate_output_dir_name() {
    local output_dir_name="$1"
    local output_dir_name_re='^[A-Za-z0-9_][A-Za-z0-9._-]*$'

    if [[ -z "${output_dir_name}" || ! "${output_dir_name}" =~ ${output_dir_name_re} ]]; then
        usage
        echo "output-dir-name must start with a letter, number, or underscore and then use only letters, numbers, dot, underscore, or dash" >&2
        return 1
    fi
}

validate_test_cmd() {
    if [[ -z "${TEST_CMD}" || "${TEST_CMD}" == *$'\n'* || "${TEST_CMD}" == *$'\r'* ]]; then
        echo "TEST_CMD must be a non-empty single-line command" >&2
        return 1
    fi
}

container_main() {
    local kernel_name="$1"
    local output_dir_name="$2"
    local all_simd="$3"
    local output_dir="${TRACE_ROOT}/${output_dir_name}"
    local output_archive="${TRACE_ROOT}/${output_dir_name}.tar.gz"
    local work_dir="${TRACE_ROOT}/.get_isa_runner_att_${output_dir_name}_$$"
    local log_dir="${work_dir}/logs"
    local kernel_trace_dir="${work_dir}/kernel_trace"
    local thread_trace_root="${work_dir}/thread_trace"
    local summary_log="${work_dir}/summary.log"

    local -a test_cmd
    read -r -a test_cmd <<< "${TEST_CMD}"
    if [[ "${#test_cmd[@]}" -eq 0 ]]; then
        echo "Unable to parse TEST_CMD into an executable command" >&2
        return 1
    fi
    cd "${REPO_ROOT}"

    # --- gfx1250 ATT 修复注入（4 处，见 rocprofv3_att_debug/README_gfx1250_new.md）---
    # ① 采集：source rocprof_env.sh 让 LD_LIBRARY_PATH 含 /opt/rocm/lib（HSA 裸名
    #    dlopen aqlprofile），并前置 comgr_new（LLVM23，认 gfx1250 新指令，避免解码
    #    吐 .long）。② 钉死自编译 rocprofv3（带 gfx1250 修复）。③ 强制用已验证能
    #    解码 gfx1250 navi 的 0.1.5 decoder（decoder_new），绕开脚本下载的 0.1.6。
    source "${workspace_dir}/rocprof_env.sh"
    export PATH="${workspace_dir}/rocprof-install/bin:${PATH}"
    export ROCPROF_ATT_LIBRARY_PATH="${workspace_dir}/decoder_new"

    rm -rf "${work_dir}" "${output_dir}" "${output_archive}"
    mkdir -p "${log_dir}" "${kernel_trace_dir}" "${thread_trace_root}" "${TRACE_ROOT}"

    {
        echo "date: $(date -Is)"
        echo "host: $(hostname)"
        echo "container: ${CONTAINER_NAME}"
        echo "pwd: $(pwd)"
        echo "host_git_branch: ${HOST_GIT_BRANCH:-unknown}"
        echo "host_git_commit: ${HOST_GIT_COMMIT:-unknown}"
        echo "HIP_VISIBLE_DEVICES: ${HIP_VISIBLE_DEVICES:-unset}"
        echo "requested kernel: ${kernel_name}"
        echo "capture all SIMDs: ${all_simd}"
        echo "output directory name: ${output_dir_name}"
        echo "output directory: ${output_dir}"
        echo "output archive: ${output_archive}"
        echo "python: $(command -v python || true)"
        python --version || true
        echo "rocprofv3: $(command -v rocprofv3 || true)"
        rocprofv3 --version || true
        echo
        echo "test command: ${TEST_CMD}"
    } 2>&1 | tee "${log_dir}/environment.log"

    ensure_trace_decoder() {
        {
            echo "Checking rocprof trace decoder:"
            echo "ROCPROF_ATT_LIBRARY_PATH=${ROCPROF_ATT_LIBRARY_PATH:-unset}"
            ls -lah "${ROCPROF_ATT_LIBRARY_PATH:-/nonexistent}/librocprof-trace-decoder.so" || true
            ls -lah /opt/rocm/lib/librocprof-trace-decoder.so || true
        } 2>&1 | tee "${log_dir}/trace_decoder.log"

        # gfx1250: 已用 ROCPROF_ATT_LIBRARY_PATH 钉死已验证的 0.1.5 decoder，
        # 其优先级高于 /opt/rocm/lib，无需再下载 0.1.6。
        if [[ -n "${ROCPROF_ATT_LIBRARY_PATH:-}" && \
              -f "${ROCPROF_ATT_LIBRARY_PATH}/librocprof-trace-decoder.so" ]]; then
            return 0
        fi

        if [[ -f /opt/rocm/lib/librocprof-trace-decoder.so ]]; then
            return 0
        fi

        (
            echo "Installing rocprof trace decoder into /opt/rocm"
            cd /tmp
            wget -q https://github.com/ROCm/rocprof-trace-decoder/releases/download/0.1.6/rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux.sh
            chmod a+x rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux.sh
            echo -e 'y\nn' | ./rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux.sh --prefix=/opt/rocm/
            cp /opt/rocm/opt/rocm/lib/librocprof-trace-decoder.so /opt/rocm/lib/
            ls -lah /opt/rocm/lib/librocprof-trace-decoder.so
        ) 2>&1 | tee -a "${log_dir}/trace_decoder.log"
    }

    run_and_log() {
        local name="$1"
        local log_file="$2"
        local restore_errexit=0
        shift 2

        if [[ "$-" == *e* ]]; then
            restore_errexit=1
        fi
        echo "Running ${name}: $*" | tee "${log_file}"
        set +e
        "$@" 2>&1 | tee -a "${log_file}"
        local status=${PIPESTATUS[0]}
        if [[ "${restore_errexit}" -eq 1 ]]; then
            set -e
        fi
        echo "${name} exit status: ${status}" | tee -a "${log_file}"
        return "${status}"
    }

    package_outputs() {
        rm -rf "${output_dir}"
        mkdir -p "${output_dir}"

        cp -a "${log_dir}" "${output_dir}/logs"
        if [[ -d "${kernel_trace_dir}" ]]; then
            cp -a "${kernel_trace_dir}" "${output_dir}/kernel_trace"
        fi
        if [[ -d "${thread_trace_root}" ]]; then
            cp -a "${thread_trace_root}" "${output_dir}/thread_trace"
        fi
        for artifact in \
            "${summary_log}" \
            "${work_dir}"/input_*.yaml; do
            if [[ -f "${artifact}" ]]; then
                cp -a "${artifact}" "${output_dir}/"
            fi
        done

        {
            echo "Collected single-kernel trace files:"
            find "${output_dir}" -type f -print | sort
        } 2>&1 | tee "${log_dir}/att_files.log"
        cp -a "${log_dir}/att_files.log" "${output_dir}/logs/att_files.log"

        tar -C "${TRACE_ROOT}" -czf "${REPO_ROOT}/${output_archive}" "${output_dir_name}"
        chmod -R a+rwX "${output_dir}"
        chmod a+rw "${output_archive}"
        ls -lah "${output_dir}" "${output_archive}"
    }

    run_att_for_kernel() {
        local kernel_name="$1"
        local kernel_regex="$2"
        local simd_id="$3"
        local log_file="$4"
        local input_yaml
        local att_output_dir
        local status
        local att_file

        if [[ "${all_simd}" -eq 1 ]]; then
            input_yaml="${work_dir}/input_kernel_simd${simd_id}.yaml"
            att_output_dir="${thread_trace_root}/simd${simd_id}/kernel/rpf_v3"
        else
            input_yaml="${work_dir}/input_kernel.yaml"
            att_output_dir="${thread_trace_root}/kernel/rpf_v3"
        fi

        cat > "${input_yaml}" <<YAML
jobs:
 -
  kernel_include_regex: '^${kernel_regex}$'
  kernel_exclude_regex:
  kernel_iteration_range: "[1]"
  output_file: out
  output_directory: ${att_output_dir}
  output_format: [csv]
  truncate_kernels: false
  sys_trace: false
  advanced_thread_trace: true
  att_target_cu: 1
  att_shader_engine_mask: "0xf"
  att_simd_select: "${simd_id}"
  att_buffer_size: "0x10000000"
  att_library_path: ["${ROCPROF_ATT_LIBRARY_PATH}"]
YAML

        rm -rf "${att_output_dir}"
        mkdir -p "$(dirname "${att_output_dir}")"
        run_and_log \
            "advanced-thread-trace-${kernel_name}-simd${simd_id}" \
            "${log_file}" \
            rocprofv3 -i "${input_yaml}" -- \
                "${test_env[@]}" \
                "${test_cmd[@]}"
        status=$?
        if [[ "${status}" -ne 0 ]]; then
            return "${status}"
        fi

        att_file="$(find "${att_output_dir}" -type f -name '*.att' -print -quit 2>/dev/null || true)"
        if [[ -z "${att_file}" ]]; then
            echo "ATT command succeeded but produced no .att file for ${kernel_name}" | tee -a "${log_file}"
            return 1
        fi
        echo "ATT artifact found: ${att_file}" | tee -a "${log_file}"
    }

    ensure_trace_decoder

    rm -rf "${kernel_trace_dir}"
    mkdir -p "${kernel_trace_dir}"

    local -a test_env=(
        env -u FLYDSL_DUMP_DIR
        PYTORCH_ALLOC_CONF=expandable_segments:True
        GPU_ARCHS=gfx1250
        ENABLE_CK=0
        AITER_FORCE_GFX1250=1
        FLYDSL_DUMP_IR=0
    )

    # A stale cache would silently trace a previously-built tile config.
    rm -rf "${HOME}/.flydsl/cache"/* 2>/dev/null || true

    set +e
    run_and_log \
        "kernel-trace-stats" \
        "${log_dir}/01_kernel_trace_stats.log" \
        rocprofv3 --kernel-trace --stats -d "${kernel_trace_dir}" -- \
            "${test_env[@]}" \
            "${test_cmd[@]}"
    local kernel_trace_status=$?
    set -e

    local kernel_regex
    kernel_regex="$(escape_kernel_regex "${kernel_name}")"
    local -a simd_ids=(3)
    if [[ "${all_simd}" -eq 1 ]]; then
        simd_ids=(0 1 2 3)
    fi

    local att_status=99
    local simd_id
    local simd_status
    local simd_log
    local simd_status_log="${log_dir}/att_simd_status.log"
    if [[ "${kernel_trace_status}" -eq 0 ]]; then
        att_status=0
        : > "${simd_status_log}"
        for simd_id in "${simd_ids[@]}"; do
            if [[ "${all_simd}" -eq 1 ]]; then
                simd_log="${log_dir}/02_thread_trace_kernel_simd${simd_id}.log"
            else
                simd_log="${log_dir}/02_thread_trace_kernel.log"
            fi
            set +e
            run_att_for_kernel \
                "${kernel_name}" \
                "${kernel_regex}" \
                "${simd_id}" \
                "${simd_log}"
            simd_status=$?
            set -e
            echo "simd${simd_id}_att_status=${simd_status}" | tee -a "${simd_status_log}"
            if [[ "${simd_status}" -ne 0 ]]; then
                att_status="${simd_status}"
            fi
        done
    fi

    {
        echo "requested_kernel=${kernel_name}"
        echo "kernel_regex=${kernel_regex}"
        echo "kernel_name_validation=trusted_without_database_lookup"
        echo "all_simd=${all_simd}"
        echo "simd_ids=${simd_ids[*]}"
        echo "output_dir_name=${output_dir_name}"
        echo "test_command=${TEST_CMD}"
        echo "kernel_trace_status=${kernel_trace_status}"
        echo "att_status=${att_status}"
        if [[ "${kernel_trace_status}" -eq 0 ]]; then
            echo "kernel_trace_state=success"
        else
            echo "kernel_trace_state=failed"
        fi
        if [[ "${att_status}" -eq 0 ]]; then
            echo "att_state=success"
        elif [[ "${att_status}" -eq 99 ]]; then
            echo "att_state=not_run"
        else
            echo "att_state=failed"
        fi
        if [[ -f "${simd_status_log}" ]]; then
            cat "${simd_status_log}"
        fi
        echo "final_output_dir=${output_dir}"
        echo "final_archive=${output_archive}"
    } 2>&1 | tee "${summary_log}"

    package_outputs
    rm -rf "${work_dir}"

    if [[ "${kernel_trace_status}" -ne 0 || "${att_status}" -ne 0 ]]; then
        return 1
    fi
}

if [[ "${1:-}" == "--inside-container" ]]; then
    kernel_name="${2:-}"
    output_dir_name="${3:-}"
    TEST_CMD="${4:-}"
    all_simd="${5:-0}"
    if [[ "$#" -ne 5 ]]; then
        usage
        echo "--inside-container requires KERNEL_NAME, output-dir-name, TEST_CMD, and ALL_SIMD" >&2
        exit 1
    fi
    if [[ "${all_simd}" != "0" && "${all_simd}" != "1" ]]; then
        echo "ALL_SIMD must be 0 or 1" >&2
        exit 1
    fi
    validate_kernel_name "${kernel_name}"
    validate_output_dir_name "${output_dir_name}"
    validate_test_cmd
    container_main "${kernel_name}" "${output_dir_name}" "${all_simd}"
    exit $?
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ "$#" -lt 3 ]]; then
    usage
    exit 1
fi

kernel_name="$1"
output_dir_name="$2"
TEST_CMD="$3"
validate_kernel_name "${kernel_name}"
validate_output_dir_name "${output_dir_name}"
output_dir="${TRACE_ROOT}/${output_dir_name}"
output_archive="${TRACE_ROOT}/${output_dir_name}.tar.gz"
shift 3

git_mode=0
am_mode=0
all_simd=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --all-simd)
            all_simd=1
            shift
            ;;
        --git)
            git_mode=1
            shift
            ;;
        --am)
            am_mode=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if [[ "${am_mode}" -eq 1 && "${git_mode}" -ne 1 ]]; then
    usage
    echo "--am requires --git" >&2
    exit 1
fi

if [[ "${git_mode}" -eq 1 ]]; then
    export GIT_SSH_COMMAND='ssh -i /data/yanguahe/code/id_rsa.hyg -o IdentitiesOnly=yes'
    if [[ ! -d "${REPO_ROOT}/${output_dir}" ]]; then
        echo "Missing expected trace directory: ${REPO_ROOT}/${output_dir}" >&2
        exit 1
    fi

    git -C "${REPO_ROOT}" add -f -- "${output_dir}"
    if ! git -C "${REPO_ROOT}" diff --cached --quiet; then
        if [[ "${am_mode}" -eq 1 ]]; then
            git -C "${REPO_ROOT}" -c user.name=yanguahe -c user.email=yanguahe@amd.com \
                commit --amend --author="yanguahe <yanguahe@amd.com>" -m Update
        else
            git -C "${REPO_ROOT}" -c user.name=yanguahe -c user.email=yanguahe@amd.com \
                commit --author="yanguahe <yanguahe@amd.com>" -m Update
        fi
    fi
    if [[ "${am_mode}" -eq 1 ]]; then
        git -C "${REPO_ROOT}" push -f origin hyg_gfx1250_gemm
    else
        git -C "${REPO_ROOT}" push origin hyg_gfx1250_gemm
    fi
    exit 0
fi

host_git_branch="not-requested"
host_git_commit="not-requested"
validate_test_cmd
docker_env=(
    -e workspace_dir="${workspace_dir}"
    -e REPO_ROOT="${REPO_ROOT}"
    -e CONTAINER_NAME="${CONTAINER_NAME}"
    -e TRACE_ROOT="${TRACE_ROOT}"
    -e SCRIPT_RELATIVE_PATH="${SCRIPT_RELATIVE_PATH}"
    -e KERNEL_NAME="${kernel_name}"
    -e OUTPUT_DIR_NAME="${output_dir_name}"
    -e TEST_CMD="${TEST_CMD}"
    -e ALL_SIMD="${all_simd}"
    -e HOST_GIT_BRANCH="${host_git_branch}"
    -e HOST_GIT_COMMIT="${host_git_commit}"
    -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}"
)

set +e
docker exec -i "${docker_env[@]}" "${CONTAINER_NAME}" \
    bash -lc 'cd "$REPO_ROOT" && bash "./$SCRIPT_RELATIVE_PATH" --inside-container "$KERNEL_NAME" "$OUTPUT_DIR_NAME" "$TEST_CMD" "$ALL_SIMD"'
run_status=$?
set -e

echo "Trace collection complete; Git operations were not requested."
echo "Run the same command with --git appended to commit the trace directory."

exit "${run_status}"
