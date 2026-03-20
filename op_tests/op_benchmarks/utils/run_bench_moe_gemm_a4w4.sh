#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run op_tests/op_benchmarks/triton/bench_moe_gemm_a4w4.py across a sweep and MERGE the benchmark's
generated logs/**.csv into ONE CSV.

IMPORTANT: This benchmark does NOT support "-o" like the perf_report benchmarks.
It writes output under logs/.../*.csv (see roofline_mlp() in the python file).

Sweepable CLI parameters in bench_moe_gemm_a4w4.py:
  --M <int...>            (optional; if omitted, benchmark uses its internal predetermined list)
  --shape <dim1 dim2>     (required; exactly 2 ints)
  --experts <total active>(required; exactly 2 ints)
  --op-regex <regex>      (optional; default ".*moe_gemm.*")
  --num-weight-inits <n>  (optional; default 1)

This script (by default) sweeps ALL of the above by looping over:
  - SHAPES_LIST (multiple dim1/dim2 pairs)
  - EXPERTS_LIST (multiple total/active pairs)
  - NUM_WEIGHT_INITS_LIST
  - two "M modes":
      * builtin M list (do not pass --M)
      * explicit M list (pass --M with M_LIST)

Usage:
  ./run_bench_moe_gemm_a4w4.sh -o <output.csv> [--keep-tmp]

Example:
  ./run_bench_moe_gemm_a4w4.sh -o bench_out/bench_moe_gemm_a4w4_merged.csv

Env overrides:
  AITER_REPO=/path/to/aiter  (default: pwd)
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
BENCH="${AITER_REPO}/op_tests/op_benchmarks/triton/bench_moe_gemm_a4w4.py"

mkdir -p "$(dirname "${OUT_FILE}")"

# ----------------------------
# Default sweep grids (edit as needed)
# ----------------------------

# --shape dim1 dim2 (Moe input dim1 and intermediate dim2)
# Provide a few common MoE-ish shapes; add your own.
SHAPES_LIST=(
  "4096 14336"
  "4096 16384"
  "6144 16384"
  "8192 28672"
)

# --experts total active
EXPERTS_LIST=(
  "8 2"
  "16 2"
  "16 4"
  "32 2"
  "32 4"
  "64 2"
  "64 4"
  "64 8"
)

# --num-weight-inits
NUM_WEIGHT_INITS_LIST=(1 3)

# op-regex (usually you keep default, but it IS sweepable)
OP_REGEX_LIST=(".*moe_gemm.*")

# Two M modes: builtin (no --M) and explicit list (pass --M ...)
M_MODE_LIST=("builtin" "explicit")

# Explicit M list (only used when M_MODE=explicit)
M_LIST=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096)

# ----------------------------
# helpers
# ----------------------------
fmt_hms() {
  local s="$1"
  local h=$((s/3600))
  local m=$(((s%3600)/60))
  local ss=$((s%60))
  printf "%02d:%02d:%02d" "${h}" "${m}" "${ss}"
}

# find output CSVs produced by the benchmark (under logs/**.csv)
find_bench_output_csvs() {
  local workdir="$1"
  find "${workdir}/logs" -type f -name "*.csv" 2>/dev/null | sort || true
}

header_written=0
write_header_once() {
  local bench_csv="$1"
  local bench_header
  bench_header="$(head -n 1 "${bench_csv}" | tr -d '\r')"
  printf "timestamp,shape_dim1,shape_dim2,experts_total,experts_active,num_weight_inits,op_regex,m_mode,M_list,%s\n" "${bench_header}" > "${OUT_FILE}"
}

append_csv_body() {
  local ts="$1"
  local dim1="$2"
  local dim2="$3"
  local total_ex="$4"
  local active_ex="$5"
  local nwi="$6"
  local op_regex="$7"
  local m_mode="$8"
  local m_list_str="$9"
  local bench_csv="${10}"

  tail -n +2 "${bench_csv}" | tr -d '\r' \
  | awk -v ts="${ts}" -v d1="${dim1}" -v d2="${dim2}" -v te="${total_ex}" -v ae="${active_ex}" \
        -v nwi="${nwi}" -v rx="${op_regex}" -v mm="${m_mode}" -v ml="${m_list_str}" '
      BEGIN{OFS=","}
      { print ts, d1, d2, te, ae, nwi, rx, mm, ml, $0 }
    ' >> "${OUT_FILE}"
}

# ----------------------------
# Progress bookkeeping
# ----------------------------
total_runs=$(( ${#SHAPES_LIST[@]} * ${#EXPERTS_LIST[@]} * ${#NUM_WEIGHT_INITS_LIST[@]} * ${#OP_REGEX_LIST[@]} * ${#M_MODE_LIST[@]} ))
run_idx=0
global_start_epoch="$(date +%s)"

echo "Writing merged CSV to: ${OUT_FILE}"
echo "Bench: ${BENCH}"
echo "Total runs: ${total_runs}"
echo

for shape in "${SHAPES_LIST[@]}"; do
  read -r dim1 dim2 <<< "${shape}"

  for ex in "${EXPERTS_LIST[@]}"; do
    read -r total_ex active_ex <<< "${ex}"

    for nwi in "${NUM_WEIGHT_INITS_LIST[@]}"; do
      for op_regex in "${OP_REGEX_LIST[@]}"; do
        for m_mode in "${M_MODE_LIST[@]}"; do
          ts="$(date -Iseconds)"
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

          # Build command
          cmd=(python3 "${BENCH}"
            --shape "${dim1}" "${dim2}"
            --experts "${total_ex}" "${active_ex}"
            --op-regex "${op_regex}"
            --num-weight-inits "${nwi}"
          )

          m_list_str=""
          if [[ "${m_mode}" == "explicit" ]]; then
            cmd+=(--M "${M_LIST[@]}")
            m_list_str="$(IFS=' '; echo "${M_LIST[*]}")"
          else
            m_list_str="(builtin)"
          fi

          echo "================================================================================"
          echo "[${run_idx}/${total_runs}] ${pct}%  elapsed=$(fmt_hms "${elapsed_global}")  eta=$(fmt_hms "${eta}")"
          echo "RUN shape=${dim1}x${dim2} experts=${total_ex}/${active_ex} num_weight_inits=${nwi} m_mode=${m_mode}"
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

          # Merge any CSVs that were generated
          mapfile -t csvs < <(find_bench_output_csvs "${workdir}")

          if [[ "${#csvs[@]}" -eq 0 ]]; then
            echo "WARN: no logs/**/*.csv found; nothing merged. Check ${workdir}/stdout_stderr.log" >&2
          else
            echo "found_csvs=${#csvs[@]}"
            for bench_csv in "${csvs[@]}"; do
              if [[ "${header_written}" -eq 0 ]]; then
                write_header_once "${bench_csv}"
                header_written=1
              fi
              append_csv_body "${ts}" "${dim1}" "${dim2}" "${total_ex}" "${active_ex}" "${nwi}" "${op_regex}" "${m_mode}" "${m_list_str}" "${bench_csv}"
              echo "merged_from=$(basename "${bench_csv}")"
            done
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
done

echo "Done. Merged results at: ${OUT_FILE}"
echo "NOTE: This script merges the benchmark-generated logs/**/*.csv (roofline/proton output), not perf_report CSV."