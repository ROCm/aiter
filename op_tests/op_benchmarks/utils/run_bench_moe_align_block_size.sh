#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run op_tests/op_benchmarks/triton/bench_moe_align_block_size.py across a sweep and MERGE results
into ONE CSV by using the benchmark's own generated CSV (via -o).

From the Python file, the loopable CLI params are:
  --model <name|all>     (NOTE: the script currently hard-codes mixtral inside model_benchmark_configs)
  -M <int>
  -block_size <int>
  -o                     (write benchmark CSV to current directory)

For each run:
  1) create a temp workdir
  2) run bench_moe_align_block_size.py with -o in that workdir
  3) merge the generated bench_*.csv into the final output CSV with extra metadata columns
  4) delete the temp workdir (unless --keep-tmp)

Usage:
  ./run_bench_moe_align_block_size.sh -o <output.csv> [--keep-tmp]

Example:
  ./run_bench_moe_align_block_size.sh -o bench_out/bench_moe_align_block_size_all.csv

Env overrides:
  AITER_REPO=/path/to/aiter  (default: pwd)
  MODEL_CONFIGS=/path/to/model_configs.json
EOF
}

OUT_FILE=""
KEEP_TMP=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -o) OUT_FILE="${2:-}"; shift 2 ;;
    --keep-tmp) KEEP_TMP=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${OUT_FILE}" ]]; then
  echo "ERROR: -o <output.csv> is required" >&2
  usage
  exit 2
fi

if [[ "${OUT_FILE}" != *.csv ]]; then
  echo "ERROR: output file must end with .csv (got: ${OUT_FILE})" >&2
  exit 2
fi

AITER_REPO="${AITER_REPO:-$(pwd)}"
BENCH="${AITER_REPO}/op_tests/op_benchmarks/triton/bench_moe_align_block_size.py"
MODEL_CONFIGS="${MODEL_CONFIGS:-${AITER_REPO}/op_tests/op_benchmarks/triton/utils/model_configs.json}"

# ----------------------------
# Sweep inputs (edit as needed)
# ----------------------------
# Note: -M is used in the python to generate benchmark configs (default 4096)
M_LIST=(128 256 512 1024 2048 4096 8192 16384)

# NOTE: bench_moe_align_block_size.py currently uses:
#   get_model_configs(..., models="mixtral")
# so --model is effectively ignored. We keep the loop minimal and set it to "mixtral"
# to avoid giving the impression that other models are swept.
MODEL_LIST=(all)

# block_size is explicitly loopable (default 128).
BLOCK_SIZE_LIST=(32 64 128 256)

common_args=(
  "-model_configs" "${MODEL_CONFIGS}"
)

mkdir -p "$(dirname "${OUT_FILE}")"

# ----------------------------
# Find the benchmark-generated CSV in a workdir
# ----------------------------
find_generated_csv() {
  local workdir="$1"

  # Try likely name first (plot_name derived from caller file)
  if [[ -f "${workdir}/bench_moe_align_block_size.csv" ]]; then
    echo "${workdir}/bench_moe_align_block_size.csv"
    return 0
  fi

  local found=""
  found="$(find "${workdir}" -maxdepth 1 -type f -name "bench_*.csv" | sort | head -n 1 || true)"
  if [[ -n "${found}" ]]; then
    echo "${found}"
    return 0
  fi

  found="$(find "${workdir}" -maxdepth 1 -type f -name "*.csv" | sort | head -n 1 || true)"
  if [[ -n "${found}" ]]; then
    echo "${found}"
    return 0
  fi

  return 1
}

# ----------------------------
# Output header management
# ----------------------------
write_header_once() {
  local bench_csv="$1"
  local bench_header
  bench_header="$(head -n 1 "${bench_csv}" | tr -d '\r')"
  printf "timestamp,sweep_model_arg,M_arg,block_size,bench_csv,%s\n" "${bench_header}" > "${OUT_FILE}"
}

append_bench_csv_body() {
  local ts="$1"
  local sweep_model_arg="$2"
  local M_arg="$3"
  local block_size="$4"
  local bench_csv="$5"

  tail -n +2 "${bench_csv}" | tr -d '\r' \
  | awk -v ts="${ts}" -v sm="${sweep_model_arg}" -v Mm="${M_arg}" -v bs="${block_size}" -v bcsv="${bench_csv##*/}" '
      BEGIN{OFS=","}
      { print ts, sm, Mm, bs, bcsv, $0 }
    ' >> "${OUT_FILE}"
}

# ----------------------------
# Progress helpers
# ----------------------------
total_runs=$(( ${#MODEL_LIST[@]} * ${#M_LIST[@]} * ${#BLOCK_SIZE_LIST[@]} ))
run_idx=0
global_start_epoch="$(date +%s)"

fmt_hms() {
  local s="$1"
  local h=$((s/3600))
  local m=$(((s%3600)/60))
  local ss=$((s%60))
  printf "%02d:%02d:%02d" "${h}" "${m}" "${ss}"
}

echo "Writing merged CSV to: ${OUT_FILE}"
echo "Bench: ${BENCH}"
echo "Model configs: ${MODEL_CONFIGS}"
echo "Models: ${MODEL_LIST[*]}  (NOTE: script hard-codes mixtral internally)"
echo "M sweep: ${M_LIST[*]}"
echo "block_size sweep: ${BLOCK_SIZE_LIST[*]}"
echo "Total runs: ${total_runs}"
echo

header_written=0

for model in "${MODEL_LIST[@]}"; do
  for M in "${M_LIST[@]}"; do
    for block_size in "${BLOCK_SIZE_LIST[@]}"; do
      ts="$(date -Iseconds)"

      cmd=(python3 "${BENCH}" --model "${model}" -M "${M}" -block_size "${block_size}" "${common_args[@]}" -o)

      run_idx=$((run_idx + 1))
      now_epoch="$(date +%s)"
      elapsed_global=$((now_epoch - global_start_epoch))
      pct=$(( run_idx * 100 / total_runs ))
      if [[ "${run_idx}" -gt 1 ]]; then
        avg_per_run=$(( elapsed_global / (run_idx - 1) ))
        remaining=$(( total_runs - run_idx + 1 ))
        eta=$(( avg_per_run * remaining ))
      else
        eta=0
      fi

      echo "================================================================================"
      echo "[${run_idx}/${total_runs}] ${pct}%  elapsed=$(fmt_hms "${elapsed_global}")  eta=$(fmt_hms "${eta}")"
      echo "RUN model=${model} M=${M} block_size=${block_size}"
      printf 'cmd: '; printf '%q ' "${cmd[@]}"; echo
      echo "--------------------------------------------------------------------------------"

      workdir="$(mktemp -d)"
      run_start_epoch="$(date +%s)"

      set +e
      ( cd "${workdir}" && "${cmd[@]}" ) > "${workdir}/stdout_stderr.log" 2>&1
      rc=$?
      set -e

      run_end_epoch="$(date +%s)"
      run_secs=$((run_end_epoch - run_start_epoch))
      echo "run_time=$(fmt_hms "${run_secs}") exit_code=${rc}"
      echo "workdir=${workdir}"

      if bench_csv="$(find_generated_csv "${workdir}")"; then
        if [[ "${header_written}" -eq 0 ]]; then
          write_header_once "${bench_csv}"
          header_written=1
        fi

        before_lines="$(wc -l < "${OUT_FILE}" | tr -d ' ')"
        append_bench_csv_body "${ts}" "${model}" "${M}" "${block_size}" "${bench_csv}"
        after_lines="$(wc -l < "${OUT_FILE}" | tr -d ' ')"
        appended=$((after_lines - before_lines))
        echo "merged_rows_from_bench_csv=${appended}"
        echo "bench_csv_used=${bench_csv}"
      else
        echo "WARN: no CSV found in workdir; nothing merged for this run." >&2
        echo "      Check ${workdir}/stdout_stderr.log" >&2
      fi

      if [[ "${KEEP_TMP}" -eq 1 ]]; then
        echo "KEEP_TMP=1 -> keeping workdir: ${workdir}"
      else
        rm -rf "${workdir}"
      fi

      echo
    done
  done
done

echo "Done. Merged results at: ${OUT_FILE}"