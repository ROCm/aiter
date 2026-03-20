#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run op_tests/op_benchmarks/triton/bench_moe.py across a sweep and MERGE results into ONE CSV
by using the benchmark's own generated CSV (via -o), instead of parsing stdout.

For each run:
  1) create a temp workdir
  2) run bench_moe.py with -o in that workdir (it writes bench_moe.csv and maybe other artifacts)
  3) merge bench_moe.csv rows into the final CSV, adding sweep metadata columns
  4) delete the temp workdir (unless --keep-tmp)

Usage:
  ./run_bench_moe.sh -o <output.csv> [--keep-tmp]

Example:
  ./run_bench_moe.sh -o bench_out/bench_moe_all.csv

Env overrides:
  AITER_REPO=/path/to/aiter  (default: pwd)
  MODEL_CONFIGS=/path/to/model_configs.json
  DTYPE=fp16
  PRINT_TIME_ONLY=0|1        (default: 0) NOTE: to guarantee Time/TFLOPS/BW, leave this 0
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

# Hard-force CSV output only
if [[ "${OUT_FILE}" != *.csv ]]; then
  echo "ERROR: output file must end with .csv (got: ${OUT_FILE})" >&2
  exit 2
fi

AITER_REPO="${AITER_REPO:-$(pwd)}"
BENCH="${AITER_REPO}/op_tests/op_benchmarks/triton/bench_moe.py"
MODEL_CONFIGS="${MODEL_CONFIGS:-${AITER_REPO}/op_tests/op_benchmarks/triton/utils/model_configs.json}"

DTYPE="${DTYPE:-fp16}"
PRINT_TIME_ONLY="${PRINT_TIME_ONLY:-0}"

# ----------------------------
# Sweep inputs ("everything" in practical terms)
# ----------------------------
M_LIST=(128 256 512 1024 2048 4096 8192 16384)
MODEL_LIST=(all)

declare -a VARIANT_TAGS VARIANT_FLAGS
VARIANT_TAGS+=("fp16_baseline");            VARIANT_FLAGS+=("")
VARIANT_TAGS+=("fp16_routed_weight");       VARIANT_FLAGS+=("-routed_weight")
VARIANT_TAGS+=("fp16_silu_fused");          VARIANT_FLAGS+=("-silu_fused")
VARIANT_TAGS+=("int8_w8a16");               VARIANT_FLAGS+=("-int8_w8a16")
VARIANT_TAGS+=("int8_w8a16_routed_weight"); VARIANT_FLAGS+=("-int8_w8a16 -routed_weight")
VARIANT_TAGS+=("fp8_w8a8_e5m2fnuz");        VARIANT_FLAGS+=("-fp8_w8a8 -fp8_type e5m2fnuz")
VARIANT_TAGS+=("fp8_w8a8_e5m2fnuz_rw");     VARIANT_FLAGS+=("-fp8_w8a8 -fp8_type e5m2fnuz -routed_weight")
VARIANT_TAGS+=("int4_w4a16_g128_zp");       VARIANT_FLAGS+=("-int4_w4a16 -group_size 128 -has_zp")
VARIANT_TAGS+=("int4_w4a16_g128_nozp");     VARIANT_FLAGS+=("-int4_w4a16 -group_size 128")

common_args=(
  "-model_configs" "${MODEL_CONFIGS}"
  "-dtype" "${DTYPE}"
)
if [[ "${PRINT_TIME_ONLY}" == "1" ]]; then
  # WARNING: bench_moe.py will only emit Time when -print_time is set.
  # You said you want Time/TFLOPS/BW, so default is 0.
  common_args+=("-print_time")
fi

mkdir -p "$(dirname "${OUT_FILE}")"

# ----------------------------
# Output header management
# ----------------------------
write_header_once() {
  local bench_csv="$1"
  if [[ ! -f "${bench_csv}" ]]; then
    echo "ERROR: expected bench CSV at ${bench_csv} but it does not exist" >&2
    return 1
  fi
  local bench_header
  bench_header="$(head -n 1 "${bench_csv}" | tr -d '\r')"

  # Prepend metadata columns to the original bench header
  printf "timestamp,sweep_model_arg,M_arg,variant_tag,flags,%s\n" "${bench_header}" > "${OUT_FILE}"
}

# Append bench CSV body (skip header), add metadata columns.
# We assume bench_moe.csv is a "simple" CSV (no embedded newlines).
append_bench_csv_body() {
  local ts="$1"
  local sweep_model_arg="$2"
  local M_arg="$3"
  local vtag="$4"
  local flags="$5"
  local bench_csv="$6"

  tail -n +2 "${bench_csv}" | tr -d '\r' \
  | awk -v ts="${ts}" -v sm="${sweep_model_arg}" -v Mm="${M_arg}" -v vtag="${vtag}" -v flags="${flags}" '
      BEGIN{OFS=","}
      # Prepend metadata columns; keep original CSV row as-is in $0
      { print ts, sm, Mm, vtag, flags, $0 }
    ' >> "${OUT_FILE}"
}

# ----------------------------
# Progress helpers
# ----------------------------
total_runs=$(( ${#MODEL_LIST[@]} * ${#M_LIST[@]} * ${#VARIANT_TAGS[@]} ))
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
echo "DTYPE: ${DTYPE}"
echo "Models: ${MODEL_LIST[*]}"
echo "M sweep: ${M_LIST[*]}"
echo "Variants: ${#VARIANT_TAGS[@]}"
echo "Total runs: ${total_runs}"
echo

header_written=0

for model in "${MODEL_LIST[@]}"; do
  for M in "${M_LIST[@]}"; do
    for i in "${!VARIANT_TAGS[@]}"; do
      vtag="${VARIANT_TAGS[$i]}"
      flags="${VARIANT_FLAGS[$i]}"
      ts="$(date -Iseconds)"

      cmd=(python3 "${BENCH}" --model "${model}" -M "${M}" "${common_args[@]}" -o)
      # shellcheck disable=SC2206
      cmd+=(${flags})

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
      echo "RUN model=${model} M=${M} variant=${vtag}"
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

      bench_csv="${workdir}/bench_moe.csv"
      if [[ ! -f "${bench_csv}" ]]; then
        echo "WARN: ${bench_csv} not found; nothing merged for this run." >&2
        echo "      Check ${workdir}/stdout_stderr.log" >&2
      else
        if [[ "${header_written}" -eq 0 ]]; then
          write_header_once "${bench_csv}"
          header_written=1
        fi

        before_lines="$(wc -l < "${OUT_FILE}" | tr -d ' ')"
        append_bench_csv_body "${ts}" "${model}" "${M}" "${vtag}" "${flags}" "${bench_csv}"
        after_lines="$(wc -l < "${OUT_FILE}" | tr -d ' ')"
        appended=$((after_lines - before_lines))
        echo "merged_rows_from_bench_csv=${appended}"
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