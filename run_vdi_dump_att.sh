#!/usr/bin/env bash
set -Eeuo pipefail

workspace_dir="${workspace_dir:-/data/yanguahe/code/wk_sp1}"
REPO_ROOT="${REPO_ROOT:-${workspace_dir}/aiter}"
CONTAINER_NAME="${CONTAINER_NAME:-hyg_fyd1}"
SCRIPT_NAME="$(basename "$0")"
PERF_ROOT="perf_logs"
MHA_TEST_CMD="${MHA_TEST_CMD:-python op_tests/test_mha_flydsl_varlen.py --causal true --return_lse true -d_qk_v 128,128 -b 1 -nh 32 -sq 8192 -sk 8192 --random-value false --warmup 5 --repeat 20}"

usage() {
    echo "Usage: ${SCRIPT_NAME} <output-dir-name>" >&2
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
    local output_dir="${PERF_ROOT}/${output_dir_name}"
    local work_dir="${TMPDIR:-/tmp}/run_vdi_dump_att_${output_dir_name}_$$"
    local log_dir="${work_dir}/logs"
    local kernel_trace_dir="${work_dir}/kernel_trace"
    local att_output_dir="${PERF_ROOT}/rpf_v3"
    local input_yaml="${work_dir}/input.yaml"
    local selected_env="${work_dir}/selected_kernel.env"
    local selector_py="${work_dir}/select_flydsl_kernel.py"
    local kernel_trace_summary="${work_dir}/kernel_trace_summary.txt"
    local summary_log="${work_dir}/summary.log"

    local -a test_cmd
    read -r -a test_cmd <<< "${MHA_TEST_CMD}"
    cd "${REPO_ROOT}"

    # --- gfx1250 ATT 修复注入（4 处，见 rocprofv3_att_debug/README_gfx1250_new.md）---
    # ① 采集：source rocprof_env.sh 让 LD_LIBRARY_PATH 含 /opt/rocm/lib（HSA 裸名
    #    dlopen aqlprofile），并前置 comgr_new（LLVM23，认 gfx1250 新指令，避免解码
    #    吐 .long）。② 钉死自编译 rocprofv3（带 gfx1250 修复）。③ 强制用已验证能
    #    解码 gfx1250 navi 的 0.1.5 decoder（decoder_new），绕开脚本下载的 0.1.6。
    source "${workspace_dir}/rocprof_env.sh"
    export PATH="${workspace_dir}/rocprof-install/bin:${PATH}"
    export ROCPROF_ATT_LIBRARY_PATH="${workspace_dir}/decoder_new"

    rm -rf "${work_dir}" "${output_dir}"
    mkdir -p "${log_dir}" "${kernel_trace_dir}" "${PERF_ROOT}"

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
        shift 2

        echo "Running ${name}: $*" | tee "${log_file}"
        set +e
        "$@" 2>&1 | tee -a "${log_file}"
        local status=${PIPESTATUS[0]}
        set -e
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

selected = None
selected_raw = None
preferred = [
    (row[0], demangle_kernel_name(row[0]))
    for row in rows
    if re.search(r"(fmha|flash.*attn|mha)", demangle_kernel_name(row[0]), re.IGNORECASE)
    and not re.search(r"(triton|aten|elementwise|copy|memset)", demangle_kernel_name(row[0]), re.IGNORECASE)
]
if preferred:
    selected_raw, selected = preferred[0]
elif rows:
    selected_raw = rows[0][0]
    selected = demangle_kernel_name(selected_raw)

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
    out.write(f"\nselected_kernel={selected or 'not_found'}\n")
    if selected_raw and selected_raw != selected:
        out.write(f"selected_raw_kernel={selected_raw}\n")
    if selected and selected in normalized_resources_by_name:
        labels = ["arch_vgpr", "accum_vgpr", "sgpr", "lds"]
        out.write("selected_resources:\n")
        for label, value in zip(labels, normalized_resources_by_name[selected]):
            out.write(f"  {label}={value}\n")

if not selected:
    sys.exit("Unable to select kernel symbol from kernel trace DB")

with open(env_path, "w", encoding="utf-8") as env:
    env.write(f"KERNEL_NAME={shlex.quote(selected)}\n")
    env.write(f"KERNEL_REGEX={shlex.quote(re.escape(selected))}\n")
PY
    }

    package_outputs() {
        rm -rf "${output_dir}"
        mkdir -p "${output_dir}"

        if [[ -d "${att_output_dir}" ]]; then
            shopt -s nullglob
            cp -a \
                "${att_output_dir}"/*.att \
                "${att_output_dir}"/ui_* \
                "${att_output_dir}"/*_agent_info.csv \
                "${att_output_dir}"/stats_ui_*.csv \
                "${output_dir}/" 2>/dev/null || true
            shopt -u nullglob
        fi

        {
            echo "Collected ATT files:"
            find "${output_dir}" -type f -print | sort
        } 2>&1 | tee "${log_dir}/att_files.log"
        ls -lah "${output_dir}"
    }

    ensure_trace_decoder

    rm -rf "${kernel_trace_dir}"
    mkdir -p "${kernel_trace_dir}"

    set +e
    run_and_log \
        "kernel-trace-stats" \
        "${log_dir}/01_kernel_trace_stats.log" \
        rocprofv3 --kernel-trace --stats -d "${kernel_trace_dir}" -- \
            env PYTORCH_ALLOC_CONF=expandable_segments:True \
                GPU_ARCHS=gfx1250 \
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

    local att_status=99
    if [[ "${selector_status}" -eq 0 ]]; then
        # shellcheck disable=SC1090
        source "${selected_env}"

        cat > "${input_yaml}" <<YAML
jobs:
 -
  kernel_include_regex: '^${KERNEL_REGEX}$'
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
YAML

        rm -rf "${att_output_dir}"
        mkdir -p "${PERF_ROOT}"
        set +e
        run_and_log \
            "advanced-thread-trace-${KERNEL_NAME}" \
            "${log_dir}/03_thread_trace.log" \
            rocprofv3 -i "${input_yaml}" -- \
                env PYTORCH_ALLOC_CONF=expandable_segments:True \
                    GPU_ARCHS=gfx1250 \
                    "${test_cmd[@]}"
        att_status=$?
        set -e
    fi

    {
        echo "kernel_trace_status=${kernel_trace_status}"
        echo "selector_status=${selector_status}"
        echo "att_status=${att_status}"
        echo "selected_kernel=${KERNEL_NAME:-unknown}"
        echo "final_output_dir=${output_dir}"
        echo
        echo "Kernel trace summary:"
        if [[ -f "${kernel_trace_summary}" ]]; then
            cat "${kernel_trace_summary}"
        fi
    } 2>&1 | tee "${summary_log}"

    package_outputs
    chmod -R a+rwX "${PERF_ROOT}"
    rm -rf "${work_dir}"

    if [[ "${kernel_trace_status}" -ne 0 || "${selector_status}" -ne 0 ||
          "${att_status}" -ne 0 ]]; then
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
output_dir="${PERF_ROOT}/${output_dir_name}"

host_git_branch="$(git -C "${REPO_ROOT}" branch --show-current || true)"
host_git_commit="$(git -C "${REPO_ROOT}" rev-parse HEAD || true)"

docker_env=(
    -e workspace_dir="${workspace_dir}"
    -e REPO_ROOT="${REPO_ROOT}"
    -e CONTAINER_NAME="${CONTAINER_NAME}"
    -e MHA_TEST_CMD="${MHA_TEST_CMD}"
    -e HOST_GIT_BRANCH="${host_git_branch}"
    -e HOST_GIT_COMMIT="${host_git_commit}"
)
if [[ -n "${HIP_VISIBLE_DEVICES:-}" ]]; then
    docker_env+=(-e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}")
fi

set +e
docker exec -i "${docker_env[@]}" "${CONTAINER_NAME}" \
    bash -lc "cd '${REPO_ROOT}' && bash './${SCRIPT_NAME}' --inside-container '${output_dir_name}'"
run_status=$?
set -e

if [[ -d "${REPO_ROOT}/${output_dir}" ]]; then
    git -C "${REPO_ROOT}" add -f "${output_dir}"
    if ! git -C "${REPO_ROOT}" diff --cached --quiet; then
        git -C "${REPO_ROOT}" -c user.name=yanguahe -c user.email=yanguahe@amd.com \
            commit --author="yanguahe <yanguahe@amd.com>" -m Update
    fi
    git -C "${REPO_ROOT}" push origin hyg_dev
else
    echo "Missing expected output directory: ${REPO_ROOT}/${output_dir}" >&2
fi

exit "${run_status}"
