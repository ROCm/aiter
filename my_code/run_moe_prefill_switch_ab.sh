#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

git_repo() {
  git -c safe.directory="$REPO_ROOT" "$@"
}

PYTHON_BIN="${PYTHON_BIN:-python3}"
ROUNDS="${ROUNDS:-2}"
BASELINE_COMMIT="527047231cb41a3265549cb4e3f6c0b1d2725935"
MODE=e2e-const0
mode_seen=0
CUSTOM_EXPERTS=""
CUSTOM_TOKENS=""
CUSTOM_TOPK=""
CUSTOM_MODEL_DIM=""
CUSTOM_INTER_DIM=""
CUSTOM_SHAPE_REQUESTED=0

usage() {
  cat <<'EOF'
usage: bash my_code/run_moe_prefill_switch_ab.sh [MODE] [SHAPE OPTIONS]

Modes:
  e2e-const0   Run selected shapes with --const-init 0 (default)
  e2e-random   Run selected shapes with random initialization
  e2e-both     Run random followed by const0

Shape options:
  --experts N
  --tokens N
  --topk N
  --model-dim N
  --inter-dim N

  Specify all five options to run only that shape on HEAD at script startup.
  Custom-shape mode does not run the baseline/optimized revision comparison.
  Without shape options, the script compares both built-in shapes below.

Fixed test arguments:
  --scenario bench --data-format a4w4 --act silu
  --no-bias --no-check-aot-cache

  For a custom shape, other test options such as --iters use the
  defaults from my_code/test_flydsl_grouped_gemm_gfx1250.py.

Revisions:
  baseline     527047231cb41a3265549cb4e3f6c0b1d2725935
  optimized    HEAD at script startup

  Revision comparison applies only when no custom shape is specified.

Environment:
  ROUNDS=N     Number of rounds for every data/shape/mode case (default: 2)
  PYTHON_BIN   Python executable (default: python3)
  LOG_DIR      Output directory override
EOF
}

while (($#)); do
  case "$1" in
    -h|--help|help)
      usage
      exit 0
      ;;
    e2e-const0|e2e-random|e2e-both)
      if [[ "$mode_seen" == 1 ]]; then
        printf 'Only one MODE may be specified.\n' >&2
        usage >&2
        exit 2
      fi
      MODE="$1"
      mode_seen=1
      shift
      ;;
    --experts|--tokens|--topk|--model-dim|--inter-dim)
      CUSTOM_SHAPE_REQUESTED=1
      if (($# < 2)); then
        printf 'Missing value for %s.\n' "$1" >&2
        exit 2
      fi
      case "$1" in
        --experts) CUSTOM_EXPERTS="$2" ;;
        --tokens) CUSTOM_TOKENS="$2" ;;
        --topk) CUSTOM_TOPK="$2" ;;
        --model-dim) CUSTOM_MODEL_DIM="$2" ;;
        --inter-dim) CUSTOM_INTER_DIM="$2" ;;
      esac
      shift 2
      ;;
    --experts=*|--tokens=*|--topk=*|--model-dim=*|--inter-dim=*)
      CUSTOM_SHAPE_REQUESTED=1
      value="${1#*=}"
      case "$1" in
        --experts=*) CUSTOM_EXPERTS="$value" ;;
        --tokens=*) CUSTOM_TOKENS="$value" ;;
        --topk=*) CUSTOM_TOPK="$value" ;;
        --model-dim=*) CUSTOM_MODEL_DIM="$value" ;;
        --inter-dim=*) CUSTOM_INTER_DIM="$value" ;;
      esac
      shift
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

require_positive_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    printf '%s must be a positive integer, got %q.\n' "$name" "$value" >&2
    exit 2
  fi
}

custom_count=0
for value in \
  "$CUSTOM_EXPERTS" "$CUSTOM_TOKENS" "$CUSTOM_TOPK" \
  "$CUSTOM_MODEL_DIM" "$CUSTOM_INTER_DIM"; do
  [[ -n "$value" ]] && custom_count=$((custom_count + 1))
done
if ((CUSTOM_SHAPE_REQUESTED && custom_count != 5)); then
  printf '%s\n' \
    'Custom shape requires --experts, --tokens, --topk, --model-dim, and --inter-dim.' >&2
  exit 2
fi

CUSTOM_SHAPE=0
if ((custom_count == 5)); then
  require_positive_integer --experts "$CUSTOM_EXPERTS"
  require_positive_integer --tokens "$CUSTOM_TOKENS"
  require_positive_integer --topk "$CUSTOM_TOPK"
  require_positive_integer --model-dim "$CUSTOM_MODEL_DIM"
  require_positive_integer --inter-dim "$CUSTOM_INTER_DIM"
  if ((10#$CUSTOM_TOPK > 10#$CUSTOM_EXPERTS)); then
    printf '%s\n' '--topk must not exceed --experts.' >&2
    exit 2
  fi
  CUSTOM_SHAPE=1
fi

if [[ ! "$ROUNDS" =~ ^[1-9][0-9]*$ ]]; then
  printf 'ROUNDS must be a positive integer, got %q\n' "$ROUNDS" >&2
  exit 2
fi

if [[ ! -f /.dockerenv ]]; then
  printf 'This script must be run inside the ROCm container.\n' >&2
  exit 2
fi

if ! git_repo diff --quiet || ! git_repo diff --cached --quiet; then
  printf 'Tracked changes must be committed or stashed before revision switching.\n' >&2
  exit 2
fi

OPTIMIZED_COMMIT="$(git_repo rev-parse HEAD)"
ORIGINAL_BRANCH="$(git_repo symbolic-ref --quiet --short HEAD || true)"
if ((!CUSTOM_SHAPE)) && \
  ! git_repo cat-file -e "${BASELINE_COMMIT}^{commit}" 2>/dev/null; then
  printf 'Baseline commit is unavailable: %s\n' "$BASELINE_COMMIT" >&2
  exit 2
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/moe_prefill_switch_ab_runs/$RUN_ID}"
mkdir -p "$LOG_DIR"

RESULTS_TSV="$LOG_DIR/results.tsv"
SUMMARY_MD="$LOG_DIR/summary.md"
printf 'data\tround\torder\tcase\tshape\tmode\tgit_commit\treturn_code\tgemm1_us\tgemm1_tflops\tgemm1_rw_tbps\tgemm1_ref_output_hash128\tgemm1_output_hash128\tgemm2_us\tgemm2_tflops\tgemm2_rw_tbps\tgemm2_ref_output_hash128\tgemm2_output_hash128\tmoe_e2e_us\tlogits_diff\trel_l2\tpass\tgemm1_symbol\tgemm2_symbol\tlog_file\n' \
  >"$RESULTS_TSV"

# Common environment for every baseline/optimized case.
export ENABLE_CK=0
export AITER_MOE_EXPERT_BALANCE=true
export AITER_LOG_MORE=1
export AITER_USE_GROUPED_GEMM=1
export AITER_GROUPED_DEBUG=0
export AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1

clear_legacy_optimization_env() {
  local var
  while IFS='=' read -r var _; do
    case "$var" in
      AITER_FLYDSL_GEMM1_*|AITER_FLYDSL_GEMM2_*|AITER_FLYDSL_MXFP4_CLUSTER_*|AITER_TDM_NEXT_STAGE_*)
        unset "$var"
        ;;
    esac
  done < <(env)
}

checkout_revision() {
  local revision="$1"
  if [[ "$(git_repo rev-parse HEAD)" != "$revision" ]]; then
    git_repo -c advice.detachedHead=false checkout --quiet --detach "$revision"
  fi
}

restore_original_checkout() {
  if [[ -n "$ORIGINAL_BRANCH" ]]; then
    git_repo checkout --quiet "$ORIGINAL_BRANCH"
  else
    checkout_revision "$OPTIMIZED_COMMIT"
  fi
}

cleanup() {
  local rc=$?
  trap - EXIT
  clear_legacy_optimization_env
  if ! restore_original_checkout; then
    printf 'Failed to restore the original checkout.\n' >&2
    rc=1
  fi
  exit "$rc"
}

print_case_environment() {
  local mode="$1"
  local tested_commit="$2"

  printf 'mode=%s\n' "$mode"
  printf 'git_commit=%s\n' "$tested_commit"
  printf 'ENABLE_CK=%s\n' "$ENABLE_CK"
  printf 'AITER_MOE_EXPERT_BALANCE=%s\n' "$AITER_MOE_EXPERT_BALANCE"
  printf 'AITER_LOG_MORE=%s\n' "$AITER_LOG_MORE"
  printf 'AITER_USE_GROUPED_GEMM=%s\n' "$AITER_USE_GROUPED_GEMM"
  printf 'AITER_GROUPED_DEBUG=%s\n' "$AITER_GROUPED_DEBUG"
  printf 'AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=%s\n' \
    "$AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE"
  printf 'legacy_gemm_optimization_env=cleared\n'
}

extract_precision_metrics() {
  local log_file="$1"
  "$PYTHON_BIN" - "$log_file" <<'PY'
import sys
import re
from pathlib import Path


def cells(line: str) -> list[str]:
    return [part.strip() for part in line.strip().strip("|").split("|")]


lines = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace").splitlines()
header = None
row = None
for line in lines:
    if not line.lstrip().startswith("|"):
        continue
    values = cells(line)
    if "data_format" in values and "logits_diff" in values and "pass" in values:
        header = values
        continue
    if header and values and values[0] == "a4w4" and len(values) == len(header):
        row = dict(zip(header, values))

if row is None:
    print("NA\tNA\tNA\tNA\tNA\tNA\tNA")
    raise SystemExit(0)


def rate(column: str, unit: str) -> str:
    match = re.search(
        r"([0-9][0-9,]*(?:\.[0-9]+)?)\s+" + re.escape(unit),
        row[column],
    )
    return "NA" if match is None else match.group(1).replace(",", "")


print(
    "\t".join(
        (
            row["logits_diff"],
            row["rel_l2"],
            row["pass"],
            rate("gemm1 executed", "TFLOP/s"),
            rate("gemm1 effective R+W", "TB/s"),
            rate("gemm2 executed", "TFLOP/s"),
            rate("gemm2 effective R+W", "TB/s"),
        )
    )
)
PY
}

run_case() {
  local case_name="$1"
  local shape="$2"
  local data="$3"
  local mode="$4"
  local round="$5"
  local order="$6"
  shift 6

  local tested_commit
  clear_legacy_optimization_env
  case "$mode" in
    baseline)
      tested_commit="$BASELINE_COMMIT"
      ;;
    optimized|current)
      tested_commit="$OPTIMIZED_COMMIT"
      ;;
    *)
      printf 'Unknown mode: %s\n' "$mode" >&2
      return 2
      ;;
  esac
  checkout_revision "$tested_commit"
  tested_commit="$(git_repo rev-parse HEAD)"

  local log_file="$LOG_DIR/${data}_r${round}_o${order}_${case_name}.log"
  local rc gemm1_us gemm2_us moe_e2e_us logits_diff rel_l2 pass
  local gemm1_tflops gemm1_rw_tbps gemm2_tflops gemm2_rw_tbps
  local gemm1_ref_hash gemm1_out_hash gemm2_ref_hash gemm2_out_hash
  local gemm1_symbol gemm2_symbol

  set +e
  (
    printf '\n============================================================\n'
    printf 'case: %s\n' "$case_name"
    printf 'shape: %s\n' "$shape"
    printf 'data: %s\n' "$data"
    printf 'round: %s/%s\n' "$round" "$ROUNDS"
    printf 'order: %s\n' "$order"
    printf 'started_utc: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'repository: %s\n' "$REPO_ROOT"
    printf 'git_commit: %s\n' "$tested_commit"
    printf 'command:'
    printf ' %q' "$PYTHON_BIN" "$@"
    printf '\n'
    print_case_environment "$mode" "$tested_commit"
    printf '============================================================\n\n'

    "$PYTHON_BIN" "$@"
    rc=$?

    printf '\nfinished_utc: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    exit "$rc"
  ) 2>&1 | tee "$log_file"
  rc=${PIPESTATUS[0]}
  set -e

  gemm1_us="$(sed -n 's/.*gemm1: device_time_avg=\([0-9.]*\) us.*/\1/p' "$log_file" | tail -1)"
  gemm2_us="$(sed -n 's/.*gemm2: device_time_avg=\([0-9.]*\) us.*/\1/p' "$log_file" | tail -1)"
  moe_e2e_us="$(sed -n 's/.*fused_moe end-to-end us = \([0-9.]*\).*/\1/p' "$log_file" | tail -1)"
  gemm1_symbol="$(sed -n 's/.*gemm1: device_time_avg=[0-9.]* us count=[0-9]* symbol=//p' "$log_file" | tail -1)"
  gemm2_symbol="$(sed -n 's/.*gemm2: device_time_avg=[0-9.]* us count=[0-9]* symbol=//p' "$log_file" | tail -1)"
  gemm1_ref_hash="$(sed -n 's/.*gemm1_ref_output_hash128=//p' "$log_file" | tail -1)"
  gemm1_out_hash="$(sed -n 's/.*gemm1_output_hash128=//p' "$log_file" | tail -1)"
  gemm2_ref_hash="$(sed -n 's/.*gemm2_ref_output_hash128=//p' "$log_file" | tail -1)"
  gemm2_out_hash="$(sed -n 's/.*gemm2_output_hash128=//p' "$log_file" | tail -1)"
  IFS=$'\t' read -r logits_diff rel_l2 pass \
    gemm1_tflops gemm1_rw_tbps gemm2_tflops gemm2_rw_tbps \
    < <(extract_precision_metrics "$log_file")

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$data" "$round" "$order" "$case_name" "$shape" "$mode" \
    "$tested_commit" "$rc" \
    "${gemm1_us:-NA}" "${gemm1_tflops:-NA}" "${gemm1_rw_tbps:-NA}" \
    "${gemm1_ref_hash:-NA}" "${gemm1_out_hash:-NA}" \
    "${gemm2_us:-NA}" "${gemm2_tflops:-NA}" "${gemm2_rw_tbps:-NA}" \
    "${gemm2_ref_hash:-NA}" "${gemm2_out_hash:-NA}" \
    "${moe_e2e_us:-NA}" "${logits_diff:-NA}" "${rel_l2:-NA}" "${pass:-NA}" \
    "${gemm1_symbol:-NA}" "${gemm2_symbol:-NA}" \
    "$log_file" >>"$RESULTS_TSV"

  if [[ "$rc" -ne 0 || -z "$gemm1_us" || -z "$gemm2_us" || -z "$moe_e2e_us" \
        || "$gemm1_tflops" == "NA" || "$gemm1_rw_tbps" == "NA" \
        || "$gemm2_tflops" == "NA" || "$gemm2_rw_tbps" == "NA" ]]; then
    printf 'Case failed or timing extraction failed: %s round=%s\n' \
      "$case_name" "$round" >&2
    exit 4
  fi

  if [[ "$pass" != "True" ]]; then
    printf 'Correctness check failed: %s round=%s pass=%s\n' \
      "$case_name" "$round" "$pass" >&2
    exit 4
  fi

  if [[ ! "$gemm1_ref_hash" =~ ^[0-9a-f]{32}$ \
        || ! "$gemm1_out_hash" =~ ^[0-9a-f]{32}$ \
        || ! "$gemm2_ref_hash" =~ ^[0-9a-f]{32}$ \
        || ! "$gemm2_out_hash" =~ ^[0-9a-f]{32}$ ]]; then
    printf 'Hash extraction failed: %s round=%s\n' "$case_name" "$round" >&2
    exit 4
  fi
}

COMMON_TEST_COMMAND=(
  -u
  my_code/test_flydsl_grouped_gemm_gfx1250.py
  --scenario bench
  --data-format a4w4
  --act silu
  --no-bias
  --no-check-aot-cache
)

E96_COMMAND=(
  "${COMMON_TEST_COMMAND[@]}"
  --experts 96
  --tokens 16384
  --topk 6
  --iters 20
  --model-dim 7168
  --inter-dim 3072
)

E256_COMMAND=(
  "${COMMON_TEST_COMMAND[@]}"
  --experts 256
  --tokens 16384
  --topk 8
  --iters 100
  --model-dim 7168
  --inter-dim 2048
)

if ((CUSTOM_SHAPE)); then
  CUSTOM_COMMAND=(
    "${COMMON_TEST_COMMAND[@]}"
    --experts "$CUSTOM_EXPERTS"
    --tokens "$CUSTOM_TOKENS"
    --topk "$CUSTOM_TOPK"
    --model-dim "$CUSTOM_MODEL_DIM"
    --inter-dim "$CUSTOM_INTER_DIM"
  )
  CUSTOM_SHAPE_LABEL="E${CUSTOM_EXPERTS}/T${CUSTOM_TOKENS}/topk${CUSTOM_TOPK}/M${CUSTOM_MODEL_DIM}/I${CUSTOM_INTER_DIM}"
fi

run_named_case() {
  local case_name="$1"
  local data="$2"
  local round="$3"
  local order="$4"
  local -a data_args=()

  case "$data" in
    const0)
      data_args=(--const-init 0)
      ;;
    random)
      ;;
    *)
      printf 'Unknown data mode: %s\n' "$data" >&2
      exit 2
      ;;
  esac

  case "$case_name" in
    e96_baseline)
      run_case "$case_name" "E96/T16384/topk6/I3072" "$data" baseline \
        "$round" "$order" "${E96_COMMAND[@]}" "${data_args[@]}"
      ;;
    e96_optimized)
      run_case "$case_name" "E96/T16384/topk6/I3072" "$data" optimized \
        "$round" "$order" "${E96_COMMAND[@]}" "${data_args[@]}"
      ;;
    e256_baseline)
      run_case "$case_name" "E256/T16384/topk8/I2048" "$data" baseline \
        "$round" "$order" "${E256_COMMAND[@]}" "${data_args[@]}"
      ;;
    e256_optimized)
      run_case "$case_name" "E256/T16384/topk8/I2048" "$data" optimized \
        "$round" "$order" "${E256_COMMAND[@]}" "${data_args[@]}"
      ;;
    custom_current)
      run_case "$case_name" "$CUSTOM_SHAPE_LABEL" "$data" current \
        "$round" "$order" "${CUSTOM_COMMAND[@]}" "${data_args[@]}"
      ;;
    *)
      printf 'Unknown case: %s\n' "$case_name" >&2
      exit 2
      ;;
  esac
}

write_summary() {
  "$PYTHON_BIN" - "$RESULTS_TSV" <<'PY' >"$SUMMARY_MD"
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path


path = Path(sys.argv[1])
with path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))

if not rows:
    raise SystemExit(0)

grouped = defaultdict(list)
data_order = []
shape_order = []
for row in rows:
    key = (row["data"], row["shape"], row["mode"])
    grouped[key].append(row)
    if row["data"] not in data_order:
        data_order.append(row["data"])
    if row["shape"] not in shape_order:
        shape_order.append(row["shape"])


def values(case_rows, key):
    return [float(row[key]) for row in case_rows]


def samples(vals, digits):
    return ", ".join(f"{value:.{digits}f}" for value in vals)


def gain(baseline, value):
    return (baseline - value) / baseline * 100.0


def gain_text(baseline, value):
    return "N/A" if baseline is None else f"{gain(baseline, value):+.2f}%"


def hashes(case_rows, key):
    return "<br>".join(dict.fromkeys(row[key] for row in case_rows))


print(
    "| data | shape | mode | commit | GEMM1 samples (us) | GEMM1 median us | "
    "GEMM1 vs baseline | "
    "GEMM1 TFLOP/s | GEMM1 effective R+W (TB/s) | "
    "GEMM1 ref out hash128 | GEMM1 out hash128 | "
    "GEMM2 samples (us) | GEMM2 median us | GEMM2 vs baseline | "
    "GEMM2 TFLOP/s | GEMM2 effective R+W (TB/s) | "
    "GEMM2 ref out hash128 | GEMM2 out hash128 | "
    "MoE e2e samples (us) | MoE e2e median us | MoE e2e vs baseline | "
    "pass | logits_diff | rel_l2 |"
)
print(
    "|---|---|---|---|---|---:|---:|---:|---:|---|---|"
    "---|---:|---:|---:|---:|---|---|"
    "---|---:|---:|:---:|---:|---:|"
)

for data in data_order:
    for shape in shape_order:
        baseline_rows = grouped.get((data, shape, "baseline"), [])
        if baseline_rows:
            baseline_g1 = statistics.median(values(baseline_rows, "gemm1_us"))
            baseline_g2 = statistics.median(values(baseline_rows, "gemm2_us"))
            baseline_e2e = statistics.median(
                values(baseline_rows, "moe_e2e_us")
            )
            modes = ("baseline", "optimized")
        else:
            baseline_g1 = baseline_g2 = baseline_e2e = None
            modes = ("current",)

        for mode in modes:
            case_rows = grouped.get((data, shape, mode), [])
            if not case_rows:
                continue

            g1 = values(case_rows, "gemm1_us")
            g2 = values(case_rows, "gemm2_us")
            e2e = values(case_rows, "moe_e2e_us")
            g1_med = statistics.median(g1)
            g2_med = statistics.median(g2)
            e2e_med = statistics.median(e2e)
            g1_tflops = statistics.median(values(case_rows, "gemm1_tflops"))
            g1_rw_tbps = statistics.median(values(case_rows, "gemm1_rw_tbps"))
            g2_tflops = statistics.median(values(case_rows, "gemm2_tflops"))
            g2_rw_tbps = statistics.median(values(case_rows, "gemm2_rw_tbps"))
            last = case_rows[-1]

            print(
                f"| {data} | {shape} | {mode} | {last['git_commit'][:12]} | "
                f"{samples(g1, 3)} | "
                f"{g1_med:.3f} | {gain_text(baseline_g1, g1_med)} | "
                f"{g1_tflops:.1f} | {g1_rw_tbps:.3f} | "
                f"{hashes(case_rows, 'gemm1_ref_output_hash128')} | "
                f"{hashes(case_rows, 'gemm1_output_hash128')} | "
                f"{samples(g2, 3)} | {g2_med:.3f} | "
                f"{gain_text(baseline_g2, g2_med)} | "
                f"{g2_tflops:.1f} | {g2_rw_tbps:.3f} | "
                f"{hashes(case_rows, 'gemm2_ref_output_hash128')} | "
                f"{hashes(case_rows, 'gemm2_output_hash128')} | "
                f"{samples(e2e, 2)} | {e2e_med:.2f} | "
                f"{gain_text(baseline_e2e, e2e_med)} | "
                f"{last['pass']} | {last['logits_diff']} | {last['rel_l2']} |"
            )
PY

  printf '\n==================== Summary ====================\n'
  cat "$SUMMARY_MD"
  printf '=================================================\n'
}

if ((CUSTOM_SHAPE)); then
  ODD_CASES=(custom_current)
  EVEN_CASES=(custom_current)
else
  ODD_CASES=(e96_baseline e96_optimized e256_baseline e256_optimized)
  EVEN_CASES=(e256_optimized e256_baseline e96_optimized e96_baseline)
fi

case "$MODE" in
  e2e-random)
    DATA_MODES=(random)
    ;;
  e2e-const0)
    DATA_MODES=(const0)
    ;;
  e2e-both)
    DATA_MODES=(random const0)
    ;;
esac

trap cleanup EXIT

printf 'MODE=%s\n' "$MODE"
printf 'ROUNDS=%s\n' "$ROUNDS"
if ((CUSTOM_SHAPE)); then
  printf 'Current commit: %s\n' "$OPTIMIZED_COMMIT"
  printf 'Custom shape: %s\n' "$CUSTOM_SHAPE_LABEL"
else
  printf 'Baseline commit: %s\n' "$BASELINE_COMMIT"
  printf 'Optimized commit: %s\n' "$OPTIMIZED_COMMIT"
  printf 'Shapes: E96/T16384/topk6/I3072, E256/T16384/topk8/I2048\n'
fi
printf 'Logs: %s\n' "$LOG_DIR"
printf 'Raw results: %s\n' "$RESULTS_TSV"

for data in "${DATA_MODES[@]}"; do
  for ((round = 1; round <= ROUNDS; ++round)); do
    if ((round % 2 == 1)); then
      ROUND_CASES=("${ODD_CASES[@]}")
    else
      ROUND_CASES=("${EVEN_CASES[@]}")
    fi

    order=0
    for case_name in "${ROUND_CASES[@]}"; do
      order=$((order + 1))
      run_named_case "$case_name" "$data" "$round" "$order"
    done
  done
done

clear_legacy_optimization_env

write_summary

printf '\nAll cases completed. Logs: %s\n' "$LOG_DIR"
printf 'Summary: %s\n' "$SUMMARY_MD"
