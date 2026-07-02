#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/data/yanguahe/code/wk_sp1/aiter}"
CONTAINER_NAME="${CONTAINER_NAME:-hyg_fyd1}"
SCRIPT_NAME="$(basename "$0")"

OUT_DIR="perf_logs/mha_flydsl_varlen_nh32_s8192_thread_trace"
FINAL_TARBALL="${OUT_DIR}/mha_flydsl_varlen_b1_nh32_sq8192_sk8192_causal_lse_thread_trace.tar.gz"

container_main() {
    local log_dir="${OUT_DIR}/logs"
    local kernel_trace_dir="${OUT_DIR}/kernel_trace"
    local thread_trace_dir="${OUT_DIR}/thread_trace"
    local att_output_dir="${thread_trace_dir}/rpf_v3"
    local flydsl_dump_dir="./flydsl_dump"
    local collect_dir="${OUT_DIR}/mha_flydsl_varlen_b1_nh32_sq8192_sk8192_causal_lse_thread_trace"
    local input_yaml="${OUT_DIR}/input.yaml"
    local selected_env="${OUT_DIR}/selected_kernel.env"
    local selector_py="${OUT_DIR}/select_flydsl_kernel.py"

    local test_cmd=(
        python op_tests/test_mha_flydsl_varlen.py
        --causal true
        --return_lse true
        -b 1
        -nh 32
        -sq 8192
        -sk 8192
        --random-value false
        --warmup 5
        --repeat 20
    )

    cd "${REPO_ROOT}"
    rm -rf "${OUT_DIR}" "${flydsl_dump_dir}"
    mkdir -p "${log_dir}" "${kernel_trace_dir}" "${thread_trace_dir}"

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
            ls -lah /opt/rocm/lib/librocprof-trace-decoder.so || true
        } 2>&1 | tee "${log_dir}/trace_decoder.log"

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
        ) 2>&1 | tee -a "${REPO_ROOT}/${log_dir}/trace_decoder.log"
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
import sys

out_dir = sys.argv[1]
flydsl_dump_dir = sys.argv[2]
env_path = sys.argv[3]
summary_path = sys.argv[4]


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


dump_symbols = []
for llvm_ir in sorted(glob.glob(os.path.join(flydsl_dump_dir, "**", "20_llvm_ir.ll"), recursive=True)):
    try:
        with open(llvm_ir, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError:
        continue
    for match in re.finditer(r"define\s+amdgpu_kernel\s+void\s+@([A-Za-z0-9_.$]+)\(", text):
        dump_symbols.append(match.group(1))

db_files = sorted(glob.glob(os.path.join(out_dir, "**", "*results.db"), recursive=True))
if not db_files:
    db_files = sorted(glob.glob(os.path.join(out_dir, "**", "*.db"), recursive=True))

rows = []
resources_by_name = {}
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
            resources_by_name[row[0]] = row[5:]
        conn.close()
    except Exception as exc:
        db_error = str(exc)

selected = None
for symbol in dump_symbols:
    if symbol.startswith("fmha_fwd_kernel"):
        selected = symbol
        break
if not selected and dump_symbols:
    selected = dump_symbols[0]
if not selected:
    preferred = [
        row[0]
        for row in rows
        if re.search(r"(fmha|flash.*attn|mha)", row[0], re.IGNORECASE)
        and not re.search(r"(triton|aten|elementwise|copy|memset)", row[0], re.IGNORECASE)
    ]
    if preferred:
        selected = preferred[0]

with open(summary_path, "w", encoding="utf-8") as out:
    out.write(f"db_path={db_path or 'not_found'}\n")
    if db_error:
        out.write(f"db_error={db_error}\n")
    out.write("dump_symbols:\n")
    for symbol in dump_symbols:
        out.write(f"  {symbol}\n")
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
    if selected and selected in resources_by_name:
        labels = ["arch_vgpr", "accum_vgpr", "sgpr", "lds"]
        out.write("selected_resources:\n")
        for label, value in zip(labels, resources_by_name[selected]):
            out.write(f"  {label}={value}\n")

if not selected:
    sys.exit("Unable to select FlyDSL kernel symbol")

with open(env_path, "w", encoding="utf-8") as env:
    env.write(f"KERNEL_NAME={shlex.quote(selected)}\n")
    env.write(f"KERNEL_REGEX={shlex.quote(re.escape(selected))}\n")
PY
    }

    package_outputs() {
        rm -rf "${collect_dir}"
        mkdir -p "${collect_dir}/att_files"

        cp -a "${log_dir}" "${collect_dir}/logs" 2>/dev/null || true
        cp -a "${kernel_trace_dir}" "${collect_dir}/kernel_trace" 2>/dev/null || true
        cp -a "${flydsl_dump_dir}" "${collect_dir}/flydsl_dump" 2>/dev/null || true
        cp -a "${input_yaml}" "${collect_dir}/input.yaml" 2>/dev/null || true
        cp -a "${selected_env}" "${collect_dir}/selected_kernel.env" 2>/dev/null || true
        cp -a "${OUT_DIR}/kernel_trace_summary.txt" "${collect_dir}/kernel_trace_summary.txt" 2>/dev/null || true

        if [[ -d "${att_output_dir}" ]]; then
            while IFS= read -r file; do
                cp -a "${file}" "${collect_dir}/att_files/$(basename "${file}")"
            done < <(
                find "${att_output_dir}" -type f \( \
                    -name '*.att' -o \
                    -name 'ui_*' -o \
                    -name '*_agent_info.csv' -o \
                    -name 'stats_ui_*.csv' -o \
                    -name 'code.json' -o \
                    -name 'realtime.json' -o \
                    -name 'occupancy.json' -o \
                    -name 'wstates*.json' -o \
                    -name 'filenames.json' \
                \) -print | sort
            )
        fi

        {
            echo "Collected files for final artifact:"
            find "${collect_dir}" -type f -print | sort
        } 2>&1 | tee "${OUT_DIR}/artifact_contents.log"
        cp -a "${OUT_DIR}/artifact_contents.log" "${collect_dir}/artifact_contents.log"

        tar -czf "${FINAL_TARBALL}" -C "${OUT_DIR}" "$(basename "${collect_dir}")"
        ls -lah "${FINAL_TARBALL}"
    }

    ensure_trace_decoder

    # Force a fresh FlyDSL dump during the discovery pass so the ATT regex comes
    # from the generated code object rather than from a guessed kernel name.
    rm -rf ~/.flydsl/cache/ "${flydsl_dump_dir}" "${kernel_trace_dir}"
    mkdir -p "${kernel_trace_dir}" "${flydsl_dump_dir}"

    set +e
    run_and_log \
        "kernel-trace-stats" \
        "${log_dir}/01_kernel_trace_stats.log" \
        rocprofv3 --kernel-trace --stats -d "${kernel_trace_dir}" -- \
            env PYTORCH_ALLOC_CONF=expandable_segments:True \
                GPU_ARCHS=gfx1250 \
                FLYDSL_DUMP_IR=1 \
                FLYDSL_DUMP_DIR="${flydsl_dump_dir}" \
                "${test_cmd[@]}"
    local kernel_trace_status=$?
    set -e

    write_selector
    set +e
    python "${selector_py}" \
        "${kernel_trace_dir}" \
        "${flydsl_dump_dir}" \
        "${selected_env}" \
        "${OUT_DIR}/kernel_trace_summary.txt" \
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
        mkdir -p "${thread_trace_dir}"
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
        echo "final_tarball=${FINAL_TARBALL}"
        echo
        echo "Kernel trace summary:"
        if [[ -f "${OUT_DIR}/kernel_trace_summary.txt" ]]; then
            cat "${OUT_DIR}/kernel_trace_summary.txt"
        fi
    } 2>&1 | tee "${OUT_DIR}/summary.log"

    package_outputs
    chmod -R a+rwX "${OUT_DIR}"

    if [[ "${kernel_trace_status}" -ne 0 || "${selector_status}" -ne 0 || "${att_status}" -ne 0 ]]; then
        return 1
    fi
}

if [[ "${1:-}" == "--inside-container" ]]; then
    container_main
    exit $?
fi

host_git_branch="$(git -C "${REPO_ROOT}" branch --show-current || true)"
host_git_commit="$(git -C "${REPO_ROOT}" rev-parse HEAD || true)"

docker_env=(
    -e REPO_ROOT="${REPO_ROOT}"
    -e CONTAINER_NAME="${CONTAINER_NAME}"
    -e HOST_GIT_BRANCH="${host_git_branch}"
    -e HOST_GIT_COMMIT="${host_git_commit}"
)
if [[ -n "${HIP_VISIBLE_DEVICES:-}" ]]; then
    docker_env+=(-e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}")
fi

set +e
docker exec -i "${docker_env[@]}" "${CONTAINER_NAME}" \
    bash -lc "cd '${REPO_ROOT}' && bash './${SCRIPT_NAME}' --inside-container"
run_status=$?
set -e

if [[ -f "${REPO_ROOT}/${FINAL_TARBALL}" ]]; then
    git -C "${REPO_ROOT}" add -f "${FINAL_TARBALL}"
    if ! git -C "${REPO_ROOT}" diff --cached --quiet; then
        git -C "${REPO_ROOT}" -c user.name=yanguahe -c user.email=yanguahe@amd.com \
            commit --amend --author="yanguahe <yanguahe@amd.com>" -m Update
    fi
    git -C "${REPO_ROOT}" push -f origin hyg_dev
else
    echo "Missing expected artifact: ${REPO_ROOT}/${FINAL_TARBALL}" >&2
fi

exit "${run_status}"
