#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run op_tests/op_benchmarks/triton/bench_moe_routing_sigmoid_top1_fused.py across a sweep and MERGE
the benchmark's generated perf_report CSV (via -o) into ONE CSV.

This wrapper derives shapes from model_configs.json and calls --shape M N K.

Options:
  -o <output.csv>     (required)
  --keep-tmp          Keep per-run temp directories
  --show-errors       Print failing run logs (tail) to stderr
  --fail-fast         Abort on first failing run (implies you’ll see the error)

Env overrides:
  AITER_REPO=/path/to/aiter  (default: pwd)
  MODEL_CONFIGS=/path/to/model_configs.json
EOF
}

OUT_FILE=""
KEEP_TMP=0
SHOW_ERRORS=0
FAIL_FAST=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o) OUT_FILE="${2:-}"; shift 2 ;;
    --keep-tmp) KEEP_TMP=1; shift ;;
    --show-errors) SHOW_ERRORS=1; shift ;;
    --fail-fast) FAIL_FAST=1; shift ;;
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
BENCH="${AITER_REPO}/op_tests/op_benchmarks/triton/bench_moe_routing_sigmoid_top1_fused.py"
MODEL_CONFIGS="${MODEL_CONFIGS:-${AITER_REPO}/op_tests/op_benchmarks/triton/utils/model_configs.json}"

mkdir -p "$(dirname "${OUT_FILE}")"

METRIC_LIST=(time throughput bandwidth)
M_LIST=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384)

fmt_hms() {
  local s="$1"
  local h=$((s/3600))
  local m=$(((s%3600)/60))
  local ss=$((s%60))
  printf "%02d:%02d:%02d" "${h}" "${m}" "${ss}"
}

find_generated_csvs() {
  local workdir="$1"
  find "${workdir}" -maxdepth 1 -type f -name "bench_*.csv" 2>/dev/null | sort || true
}

header_written=0
write_header_once() {
  local bench_csv="$1"
  local bench_header
  bench_header="$(head -n 1 "${bench_csv}" | tr -d '\r')"
  printf "timestamp,model_family,model_variant,metric,M,N,K,bench_csv,%s\n" "${bench_header}" > "${OUT_FILE}"
}

append_bench_csv_body() {
  local ts="$1"
  local family="$2"
  local variant="$3"
  local metric="$4"
  local M="$5"
  local N="$6"
  local K="$7"
  local bench_csv="$8"

  tail -n +2 "${bench_csv}" | tr -d '\r' \
  | awk -v ts="${ts}" -v fam="${family}" -v var="${variant}" -v metric="${metric}" -v M="${M}" -v N="${N}" -v K="${K}" -v bcsv="${bench_csv##*/}" '
      BEGIN{OFS=","}
      { print ts, fam, var, metric, M, N, K, bcsv, $0 }
    ' >> "${OUT_FILE}"
}

SKIP_FILE="${OUT_FILE%.csv}.skipped.csv"
if [[ ! -f "${SKIP_FILE}" ]]; then
  echo "timestamp,model_family,model_variant,metric,M,N,K,reason,workdir" > "${SKIP_FILE}"
fi

log_skip() {
  local ts="$1"
  local family="$2"
  local variant="$3"
  local metric="$4"
  local M="$5"
  local N="$6"
  local K="$7"
  local reason="$8"
  local workdir="$9"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${ts}" "${family}" "${variant}" "${metric}" "${M}" "${N}" "${K}" "${reason}" "${workdir}" \
    >> "${SKIP_FILE}"
}

# Extract (family, variant, K, N) from model_configs.json.
mapfile -t SHAPES < <(
  MODEL_CONFIGS_PATH="${MODEL_CONFIGS}" python3 - <<'PY'
import json, os
path = os.environ["MODEL_CONFIGS_PATH"]
with open(path, "r") as f:
    data = json.load(f)

for family, variants in data.items():
    if not isinstance(variants, dict):
        continue
    for variant, cfg in variants.items():
        if not isinstance(cfg, dict):
            continue
        K = cfg.get("hidden_size")
        N = cfg.get("n_routed_experts", cfg.get("num_expert"))
        if K is None or N is None:
            continue
        print(f"{family}\t{variant}\t{int(K)}\t{int(N)}")
PY
)

if [[ "${#SHAPES[@]}" -eq 0 ]]; then
  echo "ERROR: No usable shapes found in ${MODEL_CONFIGS}" >&2
  exit 2
fi

total_runs=$(( ${#SHAPES[@]} * ${#METRIC_LIST[@]} * ${#M_LIST[@]} ))
run_idx=0
global_start_epoch="$(date +%s)"

echo "Writing merged CSV to: ${OUT_FILE}"
echo "Skipped runs logged at: ${SKIP_FILE}"
echo "Bench: ${BENCH}"
echo "Model configs: ${MODEL_CONFIGS}"
echo "Detected base shapes: ${#SHAPES[@]}"
echo "Total runs (including failures): ${total_runs}"
echo

for shape in "${SHAPES[@]}"; do
  IFS=$'\t' read -r family variant K N <<<"${shape}"

  for metric in "${METRIC_LIST[@]}"; do
    for M in "${M_LIST[@]}"; do
      ts="$(date -Iseconds)"
      run_idx=$((run_idx + 1))
      now_epoch="$(date +%s)"
      elapsed_global=$((now_epoch - global_start_epoch))
      pct=$(( run_idx * 100 / total_runs ))

      cmd=(python3 "${BENCH}" --metric "${metric}" --shape "${M}" "${N}" "${K}" -o)

      echo "================================================================================"
      echo "[${run_idx}/${total_runs}] ${pct}% elapsed=$(fmt_hms "${elapsed_global}")"
      echo "RUN family=${family} variant=${variant} metric=${metric} shape=M${M} N${N} K${K}"
      printf 'cmd: '; printf '%q ' "${cmd[@]}"; echo
      echo "--------------------------------------------------------------------------------"

      workdir="$(mktemp -d)"
      set +e
      ( cd "${workdir}" && "${cmd[@]}" ) > "${workdir}/stdout_stderr.log" 2>&1
      rc=$?
      set -e

      if [[ "${rc}" -ne 0 ]]; then
        if [[ "${SHOW_ERRORS}" -eq 1 ]]; then
          echo "ERROR: exit_code=${rc} (see below). workdir=${workdir}" >&2
          echo "----- begin stdout/stderr (last 200 lines) -----" >&2
          tail -n 200 "${workdir}/stdout_stderr.log" >&2 || true
          echo "----- end stdout/stderr -----" >&2
        else
          echo "FAIL: exit_code=${rc} (use --show-errors to print log). workdir=${workdir}"
        fi

        log_skip "${ts}" "${family}" "${variant}" "${metric}" "${M}" "${N}" "${K}" "exit_${rc}" "${workdir}"

        if [[ "${FAIL_FAST}" -eq 1 ]]; then
          echo "FAIL_FAST=1 -> aborting." >&2
          exit "${rc}"
        fi

        if [[ "${KEEP_TMP}" -eq 1 ]]; then
          echo "KEEP_TMP=1 -> keeping workdir: ${workdir}"
        else
          rm -rf "${workdir}"
        fi
        echo
        continue
      fi

      mapfile -t csvs < <(find_generated_csvs "${workdir}")
      if [[ "${#csvs[@]}" -eq 0 ]]; then
        echo "FAIL: run succeeded but no bench_*.csv found. workdir=${workdir}"
        log_skip "${ts}" "${family}" "${variant}" "${metric}" "${M}" "${N}" "${K}" "no_csv" "${workdir}"
        if [[ "${FAIL_FAST}" -eq 1 ]]; then
          exit 2
        fi
        if [[ "${KEEP_TMP}" -eq 1 ]]; then
          echo "KEEP_TMP=1 -> keeping workdir: ${workdir}"
        else
          rm -rf "${workdir}"
        fi
        echo
        continue
      fi

      for bench_csv in "${csvs[@]}"; do
        if [[ "${header_written}" -eq 0 ]]; then
          write_header_once "${bench_csv}"
          header_written=1
        fi
        append_bench_csv_body "${ts}" "${family}" "${variant}" "${metric}" "${M}" "${N}" "${K}" "${bench_csv}"
      done

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