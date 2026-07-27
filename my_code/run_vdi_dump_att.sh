#!/usr/bin/env bash
set -Eeuo pipefail

workspace_dir="${workspace_dir:-/data/yanguahe/code/wk_sp1}"
REPO_ROOT="${REPO_ROOT:-${workspace_dir}/aiter}"
CONTAINER_NAME="${CONTAINER_NAME:-hyg_fyd1}"
SCRIPT_NAME="$(basename "$0")"
SCRIPT_RELATIVE_PATH="${SCRIPT_RELATIVE_PATH:-my_code/${SCRIPT_NAME}}"
TRACE_ROOT="${TRACE_ROOT:-my_code}"
# The command keeps one tokens=4096 shape; GEMM_TEST_CMD can be overridden by the caller.
GEMM_TEST_CMD="${GEMM_TEST_CMD:-python op_tests/test_flydsl_grouped_gemm_gfx1250.py   --scenario bench --data-format a8w4 --layout gugu   --experts 384 --tokens  4096 --topk 6 --iters 100   --model-dim 7168 --inter-dim 768 --act silu --real-gemm --no-check-aot}"

usage() {
    echo "Usage: ${SCRIPT_NAME} <output-dir-name> [--git] [--am]" >&2
    echo "  --git  only add/commit/push an existing .tar.gz; skip trace collection" >&2
    echo "  --am   amend the current commit; requires --git" >&2
}

validate_output_dir_name() {
    local output_dir_name="$1"

    if [[ -z "${output_dir_name}" || ! "${output_dir_name}" =~ ^[A-Za-z0-9._-]+$ ||
          "${output_dir_name}" == "." || "${output_dir_name}" == ".." ]]; then
        usage
        echo "output-dir-name must use only letters, numbers, dot, underscore, or dash" >&2
        return 1
    fi
}

container_main() {
    local output_dir_name="$1"
    local output_dir="${TRACE_ROOT}/${output_dir_name}"
    local output_archive="${TRACE_ROOT}/${output_dir_name}.tar.gz"
    local work_dir="${TRACE_ROOT}/.run_vdi_dump_att_${output_dir_name}_$$"
    local log_dir="${work_dir}/logs"
    local kernel_trace_dir="${work_dir}/kernel_trace"
    local thread_trace_root="${work_dir}/thread_trace"
    local selected_env="${work_dir}/selected_kernel.env"
    local selector_py="${work_dir}/select_grouped_gemm_kernels.py"
    local kernel_trace_summary="${work_dir}/kernel_trace_summary.txt"
    local summary_log="${work_dir}/summary.log"

    local -a test_cmd
    read -r -a test_cmd <<< "${GEMM_TEST_CMD}"
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
        echo "python: $(command -v python || true)"
        python --version || true
        echo "rocprofv3: $(command -v rocprofv3 || true)"
        rocprofv3 --version || true
        echo
        echo "test command: ${test_cmd[*]}"
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

    write_selector() {
        cat > "${selector_py}" <<'PY'
import glob
import os
import re
import shlex
import sqlite3
import subprocess
import sys

out_dir = sys.argv[1]
env_path = sys.argv[2]
summary_path = sys.argv[3]


def q(name):
    return '"' + name.replace('"', '""') + '"'


def table_columns(cur, table):
    cur.execute(f"PRAGMA table_info({q(table)})")
    return [row[1] for row in cur.fetchall()]


def first_existing(columns, names):
    for name in names:
        if name in columns:
            return name
    return None


def strip_flydsl_kernel_suffix(name):
    if name.endswith(".kd"):
        return name[:-3]
    return name


def demangle_kernel_name(name):
    stripped = strip_flydsl_kernel_suffix(name)
    if not stripped.startswith("_Z"):
        return stripped
    try:
        completed = subprocess.run(
            ["c++filt", stripped],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return stripped
    demangled = completed.stdout.strip()
    return demangled if completed.returncode == 0 and demangled else stripped


db_files = sorted(glob.glob(os.path.join(out_dir, "**", "*results.db"), recursive=True))
if not db_files:
    db_files = sorted(glob.glob(os.path.join(out_dir, "**", "*.db"), recursive=True))

rows = []
normalized_resources_by_name = {}
db_path = db_files[0] if db_files else ""
db_error = ""

if db_path:
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cur.fetchall()]
        dispatch_table = next((t for t in tables if "kernel_dispatch" in t.lower()), None)
        symbol_table = next((t for t in tables if "kernel_symbol" in t.lower()), None)
        if not dispatch_table or not symbol_table:
            raise RuntimeError(f"missing dispatch/symbol tables; available={tables}")

        dcols = table_columns(cur, dispatch_table)
        scols = table_columns(cur, symbol_table)
        start_col = first_existing(dcols, ["start", "start_ns", "start_timestamp"])
        end_col = first_existing(dcols, ["end", "end_ns", "end_timestamp"])
        kernel_id_col = first_existing(dcols, ["kernel_id"])
        symbol_id_col = first_existing(scols, ["id", "kernel_id"])
        kernel_name_col = first_existing(scols, ["kernel_name", "name"])
        if not all([start_col, end_col, kernel_id_col, symbol_id_col, kernel_name_col]):
            raise RuntimeError(
                "unexpected kernel trace schema: "
                f"dispatch={dcols}, symbol={scols}"
            )

        resource_cols = [
            "arch_vgpr_count",
            "accum_vgpr_count",
            "sgpr_count",
            "group_segment_size",
        ]
        resource_exprs = []
        for col in resource_cols:
            if col in scols:
                resource_exprs.append(f"MAX(s.{q(col)}) AS {q(col)}")
            else:
                resource_exprs.append(f"NULL AS {q(col)}")

        query = f"""
            SELECT
                s.{q(kernel_name_col)} AS kernel_name,
                COUNT(*) AS dispatches,
                AVG(d.{q(end_col)} - d.{q(start_col)}) AS avg_ns,
                MIN(d.{q(end_col)} - d.{q(start_col)}) AS min_ns,
                MAX(d.{q(end_col)} - d.{q(start_col)}) AS max_ns,
                {", ".join(resource_exprs)}
            FROM {q(dispatch_table)} d
            JOIN {q(symbol_table)} s
                ON d.{q(kernel_id_col)} = s.{q(symbol_id_col)}
            GROUP BY s.{q(kernel_name_col)}
            ORDER BY avg_ns DESC
        """
        cur.execute(query)
        rows = cur.fetchall()
        for row in rows:
            normalized_resources_by_name[demangle_kernel_name(row[0])] = row[5:]
        conn.close()
    except Exception as exc:
        db_error = str(exc)

expected = {
    "GEMM1": (
        "gemm_a8w4_tdm_t16x256x256_w1x4_b2_e384_"
        "afp8_outbf16_silu_bias1_qout0_qrep1_v1"
    ),
    "GEMM2": (
        "gemm_a8w4_tdm_t16x512x128_w1x4_b2_e384_"
        "afp8_outbf16_noact_bias1_qout0_qrep1_v1"
    ),
}
rows_by_name = {}
raw_names_by_name = {}
for row in rows:
    normalized = demangle_kernel_name(row[0])
    rows_by_name[normalized] = row
    raw_names_by_name[normalized] = row[0]

missing = [label for label, name in expected.items() if name not in rows_by_name]

with open(summary_path, "w", encoding="utf-8") as out:
    out.write(f"db_path={db_path or 'not_found'}\n")
    if db_error:
        out.write(f"db_error={db_error}\n")
    out.write("\ntop kernel trace rows:\n")
    out.write(
        "kernel_name,dispatches,avg_us,min_us,max_us,"
        "arch_vgpr,accum_vgpr,sgpr,lds\n"
    )
    for row in rows[:30]:
        name, dispatches, avg_ns, min_ns, max_ns, *resources = row
        out.write(
            f"{name},{dispatches},{avg_ns / 1000.0:.3f},"
            f"{min_ns / 1000.0:.3f},{max_ns / 1000.0:.3f},"
            + ",".join("" if value is None else str(value) for value in resources)
            + "\n"
        )
    out.write("\nselected grouped GEMM kernels:\n")
    for label, name in expected.items():
        out.write(f"{label.lower()}={name if name in rows_by_name else 'not_found'}\n")
        raw_name = raw_names_by_name.get(name)
        if raw_name and raw_name != name:
            out.write(f"{label.lower()}_raw={raw_name}\n")
        if name in normalized_resources_by_name:
            resource_labels = ["arch_vgpr", "accum_vgpr", "sgpr", "lds"]
            out.write(f"{label.lower()}_resources:\n")
            for resource_label, value in zip(
                resource_labels, normalized_resources_by_name[name]
            ):
                out.write(f"  {resource_label}={value}\n")

if missing:
    sys.exit(
        "Unable to find expected grouped GEMM kernel(s): " + ", ".join(missing)
    )

with open(env_path, "w", encoding="utf-8") as env:
    for label, name in expected.items():
        env.write(f"{label}_KERNEL_NAME={shlex.quote(name)}\n")
        env.write(f"{label}_KERNEL_REGEX={shlex.quote(re.escape(name))}\n")
PY
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
            "${selected_env}" \
            "${selector_py}" \
            "${kernel_trace_summary}" \
            "${summary_log}" \
            "${work_dir}"/input_*.yaml; do
            if [[ -f "${artifact}" ]]; then
                cp -a "${artifact}" "${output_dir}/"
            fi
        done

        {
            echo "Collected grouped-GEMM trace files:"
            find "${output_dir}" -type f -print | sort
        } 2>&1 | tee "${log_dir}/att_files.log"
        cp -a "${log_dir}/att_files.log" "${output_dir}/logs/att_files.log"

        tar -C "${TRACE_ROOT}" -czf "${REPO_ROOT}/${output_archive}" "${output_dir_name}"
        chmod -R a+rwX "${output_dir}"
        chmod a+rw "${output_archive}"
        ls -lah "${output_dir}" "${output_archive}"
    }

    run_att_for_kernel() {
        local label="$1"
        local kernel_name="$2"
        local kernel_regex="$3"
        local log_file="$4"
        local input_yaml="${work_dir}/input_${label}.yaml"
        local att_output_dir="${thread_trace_root}/${label}/rpf_v3"

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
  att_simd_select: "0xf"
  att_buffer_size: "0x10000000"
  att_library_path: ["${ROCPROF_ATT_LIBRARY_PATH}"]
YAML

        rm -rf "${att_output_dir}"
        mkdir -p "$(dirname "${att_output_dir}")"
        run_and_log \
            "advanced-thread-trace-${label}-${kernel_name}" \
            "${log_file}" \
            rocprofv3 -i "${input_yaml}" -- \
                "${test_env[@]}" \
                "${test_cmd[@]}"
    }

    ensure_trace_decoder

    rm -rf "${kernel_trace_dir}"
    mkdir -p "${kernel_trace_dir}"

    local -a test_env=(
        env -u FLYDSL_DUMP_DIR
        PYTORCH_ALLOC_CONF=expandable_segments:True
        GPU_ARCHS=gfx1250
        ENABLE_CK=0
        AITER_MOE_EXPERT_BALANCE=true
        AITER_LOG_MORE=0
        AITER_FORCE_GFX1250=1
        AITER_GROUPED_A8W4_TDM=1
        FLYDSL_DUMP_IR=0
    )

    set +e
    run_and_log \
        "kernel-trace-stats" \
        "${log_dir}/01_kernel_trace_stats.log" \
        rocprofv3 --kernel-trace --stats -d "${kernel_trace_dir}" -- \
            "${test_env[@]}" \
            "${test_cmd[@]}"
    local kernel_trace_status=$?
    set -e

    write_selector
    set +e
    python "${selector_py}" \
        "${kernel_trace_dir}" \
        "${selected_env}" \
        "${kernel_trace_summary}" \
        2>&1 | tee "${log_dir}/02_select_kernel.log"
    local selector_status=${PIPESTATUS[0]}
    set -e

    local gemm1_att_status=99
    local gemm2_att_status=99
    if [[ "${selector_status}" -eq 0 ]]; then
        # shellcheck disable=SC1090
        source "${selected_env}"

        set +e
        run_att_for_kernel \
            "gemm1" \
            "${GEMM1_KERNEL_NAME}" \
            "${GEMM1_KERNEL_REGEX}" \
            "${log_dir}/03_thread_trace_gemm1.log"
        gemm1_att_status=$?
        run_att_for_kernel \
            "gemm2" \
            "${GEMM2_KERNEL_NAME}" \
            "${GEMM2_KERNEL_REGEX}" \
            "${log_dir}/04_thread_trace_gemm2.log"
        gemm2_att_status=$?
        set -e
    fi

    {
        echo "kernel_trace_status=${kernel_trace_status}"
        echo "selector_status=${selector_status}"
        echo "gemm1_att_status=${gemm1_att_status}"
        echo "gemm2_att_status=${gemm2_att_status}"
        echo "gemm1_kernel=${GEMM1_KERNEL_NAME:-unknown}"
        echo "gemm2_kernel=${GEMM2_KERNEL_NAME:-unknown}"
        echo "final_output_dir=${output_dir}"
        echo "final_archive=${output_archive}"
        echo
        echo "Kernel trace summary:"
        if [[ -f "${kernel_trace_summary}" ]]; then
            cat "${kernel_trace_summary}"
        fi
    } 2>&1 | tee "${summary_log}"

    package_outputs
    rm -rf "${work_dir}"

    if [[ "${kernel_trace_status}" -ne 0 || "${selector_status}" -ne 0 ||
          "${gemm1_att_status}" -ne 0 || "${gemm2_att_status}" -ne 0 ]]; then
        return 1
    fi
}

if [[ "${1:-}" == "--inside-container" ]]; then
    output_dir_name="${2:-}"
    validate_output_dir_name "${output_dir_name}"
    container_main "${output_dir_name}"
    exit $?
fi

output_dir_name="${1:-}"
validate_output_dir_name "${output_dir_name}"
output_dir="${TRACE_ROOT}/${output_dir_name}"
output_archive="${TRACE_ROOT}/${output_dir_name}.tar.gz"
shift

git_mode=0
am_mode=0
while [[ $# -gt 0 ]]; do
    case "$1" in
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
    if [[ ! -f "${REPO_ROOT}/${output_archive}" ]]; then
        echo "Missing expected output archive: ${REPO_ROOT}/${output_archive}" >&2
        exit 1
    fi

    git -C "${REPO_ROOT}" add -f "${output_archive}"
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
docker_env=(
    -e workspace_dir="${workspace_dir}"
    -e REPO_ROOT="${REPO_ROOT}"
    -e CONTAINER_NAME="${CONTAINER_NAME}"
    -e TRACE_ROOT="${TRACE_ROOT}"
    -e SCRIPT_RELATIVE_PATH="${SCRIPT_RELATIVE_PATH}"
    -e GEMM_TEST_CMD="${GEMM_TEST_CMD}"
    -e HOST_GIT_BRANCH="${host_git_branch}"
    -e HOST_GIT_COMMIT="${host_git_commit}"
)
if [[ -n "${HIP_VISIBLE_DEVICES:-}" ]]; then
    docker_env+=(-e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}")
fi

set +e
docker exec -i "${docker_env[@]}" "${CONTAINER_NAME}" \
    bash -lc "cd '${REPO_ROOT}' && bash './${SCRIPT_RELATIVE_PATH}' --inside-container '${output_dir_name}'"
run_status=$?
set -e

echo "Trace collection complete; Git operations were not requested."
echo "Run '${SCRIPT_RELATIVE_PATH} ${output_dir_name} --git' separately to commit the archive."

exit "${run_status}"
