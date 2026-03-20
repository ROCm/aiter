#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run op_tests/op_benchmarks/triton/bench_moe_mx.py across a sweep and MERGE results into ONE CSV
by using the benchmark's own generated CSV (via -o).

This script is resilient to known Triton compile-time assertion failures (e.g. swizzle-mx variants
that require specific block-size constraints). When a run fails with CompileTimeAssertionFailure,
the script SKIPS that combination and continues.

For each run:
  1) create a temp workdir
  2) run bench_moe_mx.py with -o in that workdir (it writes a bench_*.csv)
  3) merge that CSV rows into the final CSV, adding sweep metadata columns
  4) delete the temp workdir (unless --keep-tmp)

Usage:
  ./run_bench_moe_mx.sh -o <output.csv> [--keep-tmp]

Example:
  ./run_bench_moe_mx.sh -o bench_out/bench_moe_mx_all.csv

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
BENCH="${AITER_REPO}/op_tests/op_benchmarks/triton/bench_moe_mx.py"
MODEL_CONFIGS="${MODEL_CONFIGS:-${AITER_REPO}/op_tests/op_benchmarks/triton/utils/model_configs.json}"

# ----------------------------
# Sweep inputs ("everything" in practical terms)
# ----------------------------
M_LIST=(128 256 512 1024 2048 4096 8192 16384)
MODEL_LIST=(all)
A_DTYPE_LIST=(bf16 fp16 fp8_e5m2 mxfp4_e2m1)

# Variants from flags exposed by bench_moe_mx.py
declare -a VARIANT_TAGS VARIANT_FLAGS
VARIANT_TAGS+=("baseline");                  VARIANT_FLAGS+=("")
VARIANT_TAGS+=("routed_weight");             VARIANT_FLAGS+=("--routed-weight")
VARIANT_TAGS+=("swizzle_mx");                VARIANT_FLAGS+=("--swizzle-mx")
VARIANT_TAGS+=("routed_weight_swizzle_mx");  VARIANT_FLAGS+=("--routed-weight --swizzle-mx")
VARIANT_TAGS+=("silu_fused");                VARIANT_FLAGS+=("-silu_fused")
VARIANT_TAGS+=("silu_fused_routed_weight");  VARIANT_FLAGS+=("-silu_fused --routed-weight")
VARIANT_TAGS+=("silu_fused_swizzle_mx");     VARIANT_FLAGS+=("-silu_fused --swizzle-mx")
VARIANT_TAGS+=("silu_fused_rw_swizzle");     VARIANT_FLAGS+=("-silu_fused --routed-weight --swizzle-mx")

common_args=(
  "-model_configs" "${MODEL_CONFIGS}"
)

mkdir -p "$(dirname "${OUT_FILE}")"

# Optional: record skipped combos
SKIP_FILE="${OUT_FILE%.csv}.skipped.csv"
if [[ ! -f "${SKIP_FILE}" ]]; then
  echo "timestamp,model,M,a_dtype,variant_tag,flags,reason" > "${SKIP_FILE}"
fi

find_generated_csv() {
  local workdir="$1"

  if [[ -f "${workdir}/bench_moe_mx.csv" ]]; then
    echo "${workdir}/bench_moe_mx.csv"
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

write_header_once() {
  local bench_csv="$1"
  if [[ ! -f "${bench_csv}" ]]; then
    echo "ERROR: expected bench CSV at ${bench_csv} but it does not exist" >&2
    return 1
  fi
  local bench_header
  bench_header="$(head -n 1 "${bench_csv}" | tr -d '\r')"
  printf "timestamp,sweep_model_arg,M_arg,a_dtype,variant_tag,flags,%s\n" "${bench_header}" > "${OUT_FILE}"
}

append_bench_csv_body() {
  local ts="$1"
  local sweep_model_arg="$2"
  local M_arg="$3"
  local a_dtype="$4"
  local vtag="$5"
  local flags="$6"
  local bench_csv="$7"

  tail -n +2 "${bench_csv}" | tr -d '\r' \
  | awk -v ts="${ts}" -v sm="${sweep_model_arg}" -v Mm="${M_arg}" -v ad="${a_dtype}" -v vtag="${vtag}" -v flags="${flags}" '
      BEGIN{OFS=","}
      { print ts, sm, Mm, ad, vtag, flags, $0 }
    ' >> "${OUT_FILE}"
}

fmt_hms() {
  local s="$1"
  local h=$((s/3600))
  local m=$(((s%3600)/60))
  local ss=$((s%60))
  printf "%02d:%02d:%02d" "${h}" "${m}" "${ss}"
}

# Heuristic pre-skip:
# - The failure you saw happens when SWIZZLE_MX_B is enabled and a tl.static_assert about BLOCK_SIZE_N % 128 == 0 fails.
# - We cannot know BLOCK_SIZE_N chosen by the autotuner ahead of time, but we *can* aggressively skip obviously risky cases:
#   * swizzle_mx with very small M tends to pick smaller blocks and can trigger asserts for some shapes.
#   * you can tune this heuristic; by default we do NOT skip based on M alone.
should_preskip_variant() {
  local vtag="$1"
  local M="$2"
  local a_dtype="$3"
  local model="$4"

  # Example: if you want to be more conservative, uncomment:
  # if [[ "${vtag}" == *swizzle* && "${M}" -lt 256 ]]; then
  #   return 0
  # fi

  return 1
}

is_compile_time_assert_failure() {
  local log="$1"
  grep -q "CompileTimeAssertionFailure" "${log}" \
    || grep -q "tl.static_assert" "${log}" \
    || grep -q "static_assert" "${log}"
}

total_runs=$(( ${#MODEL_LIST[@]} * ${#M_LIST[@]} * ${#A_DTYPE_LIST[@]} * ${#VARIANT_TAGS[@]} ))
run_idx=0
global_start_epoch="$(date +%s)"

echo "Writing merged CSV to: ${OUT_FILE}"
echo "Skip log: ${SKIP_FILE}"
echo "Bench: ${BENCH}"
echo "Model configs: ${MODEL_CONFIGS}"
echo "Models: ${MODEL_LIST[*]}"
echo "A dtypes: ${A_DTYPE_LIST[*]}"
echo "M sweep: ${M_LIST[*]}"
echo "Variants: ${#VARIANT_TAGS[@]}"
echo "Total runs (including skipped): ${total_runs}"
echo

header_written=0

for a_dtype in "${A_DTYPE_LIST[@]}"; do
  for model in "${MODEL_LIST[@]}"; do
    for M in "${M_LIST[@]}"; do
      for i in "${!VARIANT_TAGS[@]}"; do
        vtag="${VARIANT_TAGS[$i]}"
        flags="${VARIANT_FLAGS[$i]}"
        ts="$(date -Iseconds)"

        run_idx=$((run_idx + 1))
        now_epoch="$(date +%s)"
        elapsed_global=$((now_epoch - global_start_epoch))
        pct=$(( run_idx * 100 / total_runs ))

        if should_preskip_variant "${vtag}" "${M}" "${a_dtype}" "${model}"; then
          echo "================================================================================"
          echo "[${run_idx}/${total_runs}] ${pct}% elapsed=$(fmt_hms "${elapsed_global}")"
          echo "PRESKIP a_dtype=${a_dtype} model=${model} M=${M} variant=${vtag} (heuristic)"
          echo "${ts},${model},${M},${a_dtype},${vtag},\"${flags}\",preskip_heuristic" >> "${SKIP_FILE}"
          echo
          continue
        fi

        cmd=(python3 "${BENCH}" --model "${model}" -M "${M}" -A "${a_dtype}" "${common_args[@]}" -o)
        # shellcheck disable=SC2206
        cmd+=(${flags})

        echo "================================================================================"
        echo "[${run_idx}/${total_runs}] ${pct}% elapsed=$(fmt_hms "${elapsed_global}")"
        echo "RUN a_dtype=${a_dtype} model=${model} M=${M} variant=${vtag}"
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

        # If Triton compile-time assertion fails, skip cleanly.
        if [[ "${rc}" -ne 0 ]] && is_compile_time_assert_failure "${workdir}/stdout_stderr.log"; then
          echo "SKIP: Compile-time assertion failure (expected for some swizzle-mx combinations)."
          echo "${ts},${model},${M},${a_dtype},${vtag},\"${flags}\",CompileTimeAssertionFailure" >> "${SKIP_FILE}"

          if [[ "${KEEP_TMP}" -eq 1 ]]; then
            echo "KEEP_TMP=1 -> keeping workdir: ${workdir}"
          else
            rm -rf "${workdir}"
          fi
          echo
          continue
        fi

        # Another expected no-output case: MXFP4 unsupported
        if [[ "${rc}" -eq 0 ]] && grep -q "MXFP4 not supported" "${workdir}/stdout_stderr.log"; then
          echo "SKIP: MXFP4 not supported on this architecture (no CSV expected)."
          echo "${ts},${model},${M},${a_dtype},${vtag},\"${flags}\",MXFP4_not_supported" >> "${SKIP_FILE}"

          if [[ "${KEEP_TMP}" -eq 1 ]]; then
            echo "KEEP_TMP=1 -> keeping workdir: ${workdir}"
          else
            rm -rf "${workdir}"
          fi
          echo
          continue
        fi

        bench_csv=""
        if bench_csv="$(find_generated_csv "${workdir}")"; then
          if [[ "${header_written}" -eq 0 ]]; then
            write_header_once "${bench_csv}"
            header_written=1
          fi

          before_lines="$(wc -l < "${OUT_FILE}" | tr -d ' ')"
          append_bench_csv_body "${ts}" "${model}" "${M}" "${a_dtype}" "${vtag}" "${flags}" "${bench_csv}"
          after_lines="$(wc -l < "${OUT_FILE}" | tr -d ' ')"
          appended=$((after_lines - before_lines))
          echo "merged_rows_from_bench_csv=${appended}"
          echo "bench_csv_used=${bench_csv}"
        else
          echo "WARN: no CSV found in workdir; nothing merged for this run." >&2
          echo "      Check ${workdir}/stdout_stderr.log" >&2
          echo "${ts},${model},${M},${a_dtype},${vtag},\"${flags}\",no_csv_found" >> "${SKIP_FILE}"
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
done

echo "Done. Merged results at: ${OUT_FILE}"
echo "Skipped runs logged at: ${SKIP_FILE}"