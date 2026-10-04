#!/usr/bin/env bash

set -euo pipefail

# Run inside hyg_fyd_e2e from /app/aiter:
#   ROUNDS=3 bash my_code/moe_perf_compare.sh
# ROUNDS is applied independently to each entry in SHAPE_SPECS below.
# The current branch always uses its fused GEMM1 quant pipeline.
#
# /app/aiter is the baseline tree. ROCm/main is kept in a separate host worktree
# visible through /data, so switching revisions is only a directory change and
# /app/aiter remains untouched.
if [[ ! -f /.dockerenv ]]; then
  printf 'Run this script inside the hyg_fyd_e2e container.\n' >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_REPO="${BASELINE_REPO:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
YADAI_REPO="${YADAI_REPO:-/data/yanguahe/code/wk_sp1/aiter_a4w4_prefill_v2_yadai}"
YADAI_REPO="$(readlink -f "$YADAI_REPO")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ROUNDS="${ROUNDS:-3}"
BASELINE_COMMIT_LABEL="${BASELINE_COMMIT_LABEL:-}"
BASELINE_BRANCH_LABEL="${BASELINE_BRANCH_LABEL:-}"
YADAI_COMMIT_LABEL="${YADAI_COMMIT_LABEL:-pending}"
GPU_USERS_SCRIPT="${GPU_USERS_SCRIPT:-/data/yanguahe/code/gpu_users.sh}"
GPU_IDLE_POLL_SEC="${GPU_IDLE_POLL_SEC:-6}"
GIT_ENV_FILE="${GIT_ENV_FILE:-/data/yanguahe/code/git_env}"
TARGET_BRANCH="${TARGET_BRANCH:-main}"
TARGET_URL="${TARGET_URL:-git@github.com:ROCm/aiter.git}"
TARGET_REF="refs/remotes/rocm/$TARGET_BRANCH"
IGNORE_GPU_BUSY="${IGNORE_GPU_BUSY:-0}"
RUN_MODE=both

SHAPE_SPECS=(
  "e64_t1536_k8_m7168_i2048|64|1536|8|7168|2048"
  "e64_t16384_k8_m7168_i2048|64|16384|8|7168|2048"
  "e256_t512_k8_m7168_i2048|256|512|8|7168|2048"
  "e256_t16384_k8_m7168_i2048|256|16384|8|7168|2048"
  "e96_t512_k6_m7168_i3072|96|512|6|7168|3072"
  "e96_t16384_k6_m7168_i3072|96|16384|6|7168|3072"
)
ROUND_SHAPES=()
COMMON_ARGS=()

set_round_shapes() {
  local round_id="$1"
  local shape_index

  ROUND_SHAPES=()
  if ((round_id % 2 == 0)); then
    ROUND_SHAPES=("${SHAPE_SPECS[@]}")
  else
    for ((shape_index = ${#SHAPE_SPECS[@]} - 1; shape_index >= 0; --shape_index)); do
      ROUND_SHAPES+=("${SHAPE_SPECS[shape_index]}")
    done
  fi
}

set_common_args() {
  local experts="$1" tokens="$2" topk="$3" model_dim="$4" inter_dim="$5"

  COMMON_ARGS=(
    --data-format a4w4
    --experts "$experts"
    --tokens "$tokens"
    --topk "$topk"
    --model-dim "$model_dim"
    --inter-dim "$inter_dim"
    --act silu
    --no-bias
    --no-check-aot-cache
    --warmup 5
    --iters 20
    --data-init zero
    --scale-init zero
  )
}

usage() {
  cat <<'EOF'
usage: bash my_code/moe_perf_compare.sh [--both|--curr|--base]

  --both  Run the current hyg/moe_a4w4_pr_refactor pipeline and ROCm/main
          isolated GEMM plus graph-off MoE e2e tests. This is the default.
  --curr  Run only the current hyg/moe_a4w4_pr_refactor MoE pipeline.
  --base  Run only the latest ROCm/main isolated GEMM and graph-off MoE e2e.

ROUNDS applies independently to every shape in every selected mode. Even
round IDs use forward shape order; odd round IDs use reversed shape order.
EOF
}

while (($#)); do
  case "$1" in
    --both)
      RUN_MODE=both
      ;;
    --curr)
      RUN_MODE=curr
      ;;
    --base)
      RUN_MODE=base
      ;;
    -h|--help|help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

RUN_CURRENT=0
RUN_BASE=0
case "$RUN_MODE" in
  both)
    RUN_CURRENT=1
    RUN_BASE=1
    ;;
  curr)
    RUN_CURRENT=1
    ;;
  base)
    RUN_BASE=1
    ;;
esac

if [[ ! "$ROUNDS" =~ ^[1-9][0-9]*$ ]]; then
  printf 'ROUNDS must be a positive integer, got %q\n' "$ROUNDS" >&2
  exit 2
fi

if ((RUN_CURRENT)); then
  if [[ ! -f "$BASELINE_REPO/my_code/run_moe_prefill_switch_ab.sh" ]]; then
    printf 'Missing baseline script: %s\n' \
      "$BASELINE_REPO/my_code/run_moe_prefill_switch_ab.sh" >&2
    exit 2
  fi

  if [[ -z "$BASELINE_COMMIT_LABEL" ]]; then
    BASELINE_COMMIT_LABEL="$(
      git -c safe.directory="$BASELINE_REPO" -C "$BASELINE_REPO" rev-parse HEAD
    )"
  fi
  if [[ -z "$BASELINE_BRANCH_LABEL" ]]; then
    BASELINE_BRANCH_LABEL="$(
      git -c safe.directory="$BASELINE_REPO" -C "$BASELINE_REPO" \
        symbolic-ref --quiet --short HEAD || printf 'detached'
    )"
  fi
fi

if ((RUN_BASE)) && [[ ! -d "$YADAI_REPO" ]]; then
  printf 'Missing comparison repository: %s\n' "$YADAI_REPO" >&2
  exit 2
fi

update_yadai_checkout() {
  local tracked_changes current_commit latest_commit ignore_gpu_busy_saved
  local -a git_cmd

  if [[ ! -f "$GIT_ENV_FILE" ]]; then
    printf 'Missing Git environment file: %s\n' "$GIT_ENV_FILE" >&2
    exit 2
  fi

  # Required on these gfx1250 development machines before any container-side
  # Git command (SSH key, identity and related environment setup).
  # shellcheck disable=SC1090
  ignore_gpu_busy_saved="$IGNORE_GPU_BUSY"
  source "$GIT_ENV_FILE"
  IGNORE_GPU_BUSY="$ignore_gpu_busy_saved"

  if ! command -v git >/dev/null 2>&1; then
    printf 'Git is unavailable after sourcing %s.\n' "$GIT_ENV_FILE" >&2
    exit 2
  fi

  git_cmd=(git -c safe.directory="$YADAI_REPO" -C "$YADAI_REPO")
  tracked_changes="$("${git_cmd[@]}" status --porcelain --untracked-files=no)"
  if [[ -n "$tracked_changes" ]]; then
    printf 'Comparison worktree has tracked changes; refusing to overwrite them:\n%s\n' \
      "$tracked_changes" >&2
    exit 2
  fi

  printf '\nFetching latest ROCm/%s inside the container...\n' "$TARGET_BRANCH"
  "${git_cmd[@]}" fetch "$TARGET_URL" "$TARGET_BRANCH:$TARGET_REF"
  latest_commit="$("${git_cmd[@]}" rev-parse "$TARGET_REF")"
  current_commit="$("${git_cmd[@]}" rev-parse HEAD)"
  if [[ "$current_commit" != "$latest_commit" ]]; then
    "${git_cmd[@]}" switch --detach "$latest_commit"
  fi

  tracked_changes="$("${git_cmd[@]}" status --porcelain --untracked-files=no)"
  if [[ -n "$tracked_changes" ]]; then
    printf 'Comparison worktree is not clean after update:\n%s\n' \
      "$tracked_changes" >&2
    exit 2
  fi

  YADAI_COMMIT_LABEL="$("${git_cmd[@]}" rev-parse HEAD)"
  printf 'ROCm/%s updated commit: %s\n' "$TARGET_BRANCH" "$YADAI_COMMIT_LABEL"
}

ensure_yadai_core_enum_abi() {
  local probe_code

  # Older comparison revisions do not reference Relu2 and remain compatible
  # with their older core extension. Newer revisions require the enum in both the
  # Python source and module_aiter_core.so.
  if ! grep -Fq 'ActivationType.Relu2' "$YADAI_REPO/aiter/fused_moe.py"; then
    return
  fi

  probe_code='from aiter import ActivationType; assert hasattr(ActivationType, "Relu2")'
  if (
    cd "$YADAI_REPO"
    PYTHONPATH="$YADAI_REPO${PYTHONPATH:+:$PYTHONPATH}" \
      "$PYTHON_BIN" -c "$probe_code"
  ) >/dev/null 2>&1; then
    printf 'Comparison module_aiter_core enum ABI is current.\n'
    return
  fi

  printf '\nComparison module_aiter_core is stale; rebuilding the core extension...\n'
  (
    cd "$YADAI_REPO"
    AITER_REBUILD=1 \
    AITER_META_DIR="$YADAI_REPO" \
    PYTHONPATH="$YADAI_REPO${PYTHONPATH:+:$PYTHONPATH}" \
      "$PYTHON_BIN" -c "$probe_code"
  )

  # Verify from a fresh process without AITER_REBUILD, matching the benchmark
  # processes below.  This catches a rebuild that succeeded but wrote to an
  # unexpected JIT directory.
  (
    cd "$YADAI_REPO"
    AITER_META_DIR="$YADAI_REPO" \
    PYTHONPATH="$YADAI_REPO${PYTHONPATH:+:$PYTHONPATH}" \
      "$PYTHON_BIN" -c "$probe_code"
  )
  printf 'Comparison module_aiter_core enum ABI rebuilt and verified.\n'
}

require_gpu_idle() {
  local label="$1"
  local users_output smi_output attempt busy

  if [[ "$IGNORE_GPU_BUSY" == 1 ]]; then
    printf '\n===== GPU idle check skipped by IGNORE_GPU_BUSY=1: %s =====\n' \
      "$label"
    return
  fi

  for attempt in 1 2 3 4 5; do
    busy=0
    printf '\n===== GPU idle check: %s (attempt %s/5) =====\n' \
      "$label" "$attempt"

    if [[ -f "$GPU_USERS_SCRIPT" ]]; then
      users_output="$(bash "$GPU_USERS_SCRIPT" 2>&1 || true)"
      printf '%s\n' "$users_output"
      if ! grep -Fq '当前没有进程在使用 GPU' <<<"$users_output"; then
        busy=1
      fi
    else
      users_output="$(fuser -v /dev/kfd 2>&1 || true)"
      printf '%s\n' "$users_output"
      if [[ -n "$users_output" ]]; then
        busy=1
      fi
    fi

    smi_output="$(rocm-smi --showuse --showmemuse 2>&1)"
    printf '%s\n' "$smi_output"
    if grep -Eq 'GPU use \(%\):[[:space:]]*[1-9][0-9]*' <<<"$smi_output"; then
      busy=1
    fi
    if grep -Eq 'VRAM%\):[[:space:]]*[1-9][0-9]*' <<<"$smi_output"; then
      busy=1
    fi
    if ((busy == 0)); then
      return
    fi
    if ((attempt < 5)); then
      sleep "$GPU_IDLE_POLL_SEC"
    fi
  done

  printf 'GPU/KFD remained busy; performance data would be invalid.\n' >&2
  exit 3
}

write_baseline_summary() {
  local output_file="$1"

  "$PYTHON_BIN" - \
    "$SHAPE_MANIFEST" \
    "$BASELINE_CASE_ROOT" \
    "$output_file" \
    "$BASELINE_BRANCH_LABEL" \
    "$BASELINE_COMMIT_LABEL" \
    "$BASELINE_REPO" \
    "$ROUNDS" <<'PY'
import csv
import pathlib
import statistics
import sys

(
    manifest_name,
    case_root_name,
    output_name,
    branch,
    commit,
    repo,
    rounds,
) = sys.argv[1:]

with open(manifest_name, encoding="utf-8", newline="") as src:
    shapes = list(csv.DictReader(src, delimiter="\t"))

case_root = pathlib.Path(case_root_name)
lines = [
    "# Baseline multi-shape benchmark",
    "",
    f"- branch: `{branch}`",
    f"- commit: `{commit}`",
    f"- rounds per shape: `{rounds}`",
    "- GEMM1 quant pipeline: `fused-quant`",
    "- MoE e2e timing: `testGraph=False`, `use_cuda_event=False`, "
    "`num_warmup=5`, `num_iters=20`, torch profiler "
    "`get_trace_perf(...).device_time_sum`",
    f"- repository: `{repo}`",
    "",
    "| shape | GEMM1 samples (us) | GEMM1 median (us) | standalone quant samples (us) | GEMM1 pipeline median (us) | GEMM2 samples (us) | GEMM2 median (us) | MoE e2e samples (us) | MoE median (us) | pass | max logits_diff | max rel_l2 | hashes |",
    "|---|---|---:|---|---:|---|---:|---|---:|:---:|---:|---:|:---:|",
]


def values(rows, key):
    return [float(row[key] or 0.0) for row in rows]


def samples(vals, digits=3):
    return ", ".join(f"{value:.{digits}f}" for value in vals)


for shape in shapes:
    rows = []
    for round_id in range(int(rounds)):
        result_file = (
            case_root / f"round{round_id}_{shape['shape_id']}" / "results.tsv"
        )
        if not result_file.is_file():
            raise SystemExit(f"missing baseline result: {result_file}")
        with result_file.open(encoding="utf-8", newline="") as src:
            round_rows = [
                row
                for row in csv.DictReader(src, delimiter="\t")
                if row["data"] == "const0"
            ]
        if len(round_rows) != 1:
            raise SystemExit(
                f"expected one baseline row in {result_file}, got {len(round_rows)}"
            )
        rows.extend(round_rows)
    if len(rows) != int(rounds):
        raise SystemExit(
            f"expected {rounds} baseline rows for {shape['shape_id']}, got {len(rows)}"
        )

    gemm1 = values(rows, "gemm1_us")
    quant = values(rows, "quant_us")
    gemm1_pipeline = [g + q for g, q in zip(gemm1, quant)]
    gemm2 = values(rows, "gemm2_us")
    moe = values(rows, "moe_e2e_us")
    passed = all(row["pass"] == "True" for row in rows)
    logits_diff = max(values(rows, "logits_diff"))
    rel_l2 = max(values(rows, "rel_l2"))
    hashes_match = all(
        row["gemm1_ref_output_hash128"] == row["gemm1_output_hash128"]
        and row["gemm2_ref_output_hash128"] == row["gemm2_output_hash128"]
        for row in rows
    )
    label = (
        f"E{shape['experts']}/T{shape['tokens']}/topk{shape['topk']}/"
        f"M{shape['model_dim']}/I{shape['inter_dim']}"
    )
    lines.append(
        f"| {label} | {samples(gemm1)} | {statistics.median(gemm1):.3f} | "
        f"{samples(quant)} | {statistics.median(gemm1_pipeline):.3f} | "
        f"{samples(gemm2)} | {statistics.median(gemm2):.3f} | "
        f"{samples(moe, 2)} | {statistics.median(moe):.2f} | {passed} | "
        f"{logits_diff:.4e} | {rel_l2:.4e} | {hashes_match} |"
    )

path = pathlib.Path(output_name)
path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_ROOT="${LOG_ROOT:-$SCRIPT_DIR/moe_prefill_yadai_compare_runs/$RUN_ID}"
BASELINE_LOG_DIR="$LOG_ROOT/baseline"
BASELINE_CASE_ROOT="$BASELINE_LOG_DIR/cases"
YADAI_LOG_DIR="$LOG_ROOT/rocm_main"
FAKE_GIT_DIR="$LOG_ROOT/fake_git"
SHAPE_MANIFEST="$LOG_ROOT/shapes.tsv"
mkdir -p "$BASELINE_CASE_ROOT" "$YADAI_LOG_DIR" "$FAKE_GIT_DIR"

printf 'shape_id\texperts\ttokens\ttopk\tmodel_dim\tinter_dim\n' >"$SHAPE_MANIFEST"
for shape_spec in "${SHAPE_SPECS[@]}"; do
  IFS='|' read -r shape_id experts tokens topk model_dim inter_dim \
    <<<"$shape_spec"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$shape_id" "$experts" "$tokens" "$topk" "$model_dim" "$inter_dim" \
    >>"$SHAPE_MANIFEST"
done

# The baseline custom-shape path only asks Git for HEAD and the branch name. A
# tiny shim supplies those labels without running Git inside the container.
cat >"$FAKE_GIT_DIR/git" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-c" ]]; then
  shift 2
fi

case "${1:-}" in
  rev-parse)
    if [[ "${2:-}" == HEAD ]]; then
      printf '%s\n' "$CODEX_TEST_COMMIT"
      exit 0
    fi
    ;;
  symbolic-ref)
    printf '%s\n' "$CODEX_TEST_BRANCH"
    exit 0
    ;;
esac

printf 'Unexpected test-harness git invocation:' >&2
printf ' %q' "$@" >&2
printf '\n' >&2
exit 2
EOF
chmod +x "$FAKE_GIT_DIR/git"

export ENABLE_CK=0
export AITER_MOE_EXPERT_BALANCE=true
export AITER_LOG_MORE=1
export AITER_USE_GROUPED_GEMM=1
export AITER_GROUPED_DEBUG=0
export AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1

printf 'Run mode: %s\n' "$RUN_MODE"
if ((RUN_CURRENT)); then
  printf 'Current repository: %s\n' "$BASELINE_REPO"
  printf 'Current commit label: %s\n' "$BASELINE_COMMIT_LABEL"
fi
if ((RUN_BASE)); then
  printf 'Comparison repository: %s\n' "$YADAI_REPO"
  printf 'Comparison target branch: ROCm/%s\n' "$TARGET_BRANCH"
fi
printf 'Rounds: %s\n' "$ROUNDS"
printf 'Logs: %s\n' "$LOG_ROOT"

if ((RUN_CURRENT)); then
  BASELINE_SUMMARY="$BASELINE_LOG_DIR/summary.md"
  printf '\n===== mode current: hyg/moe_a4w4_pr_refactor MoE pipeline =====\n'
  for ((round_id = 0; round_id < ROUNDS; ++round_id)); do
    set_round_shapes "$round_id"
    printf '\n===== current round %s: %s =====\n' \
      "$round_id" "${ROUND_SHAPES[*]}"
    for shape_spec in "${ROUND_SHAPES[@]}"; do
      IFS='|' read -r shape_id experts tokens topk model_dim inter_dim \
        <<<"$shape_spec"
      shape_log_dir="$BASELINE_CASE_ROOT/round${round_id}_${shape_id}"
      mkdir -p "$shape_log_dir"
      require_gpu_idle "before current round $round_id $shape_id"
      printf '\n===== current round %s shape %s =====\n' "$round_id" "$shape_id"
      (
        cd "$BASELINE_REPO"
        PATH="$FAKE_GIT_DIR:$PATH" \
        CODEX_TEST_COMMIT="$BASELINE_COMMIT_LABEL" \
        CODEX_TEST_BRANCH="$BASELINE_BRANCH_LABEL" \
        LOG_DIR="$shape_log_dir" \
        ROUNDS=1 \
          bash ./my_code/run_moe_prefill_switch_ab.sh e2e-const0 \
            --experts "$experts" \
            --tokens "$tokens" \
            --topk "$topk" \
            --model-dim "$model_dim" \
            --inter-dim "$inter_dim"
      ) 2>&1 | tee \
        "$BASELINE_LOG_DIR/round${round_id}_${shape_id}_console.log"
      require_gpu_idle "after current round $round_id $shape_id"
    done
  done
  write_baseline_summary "$BASELINE_SUMMARY"
fi

if ((RUN_BASE)); then
update_yadai_checkout
ensure_yadai_core_enum_abi
YADAI_TEST="$YADAI_REPO/op_tests/flydsl_tests/test_flydsl_grouped_gemm.py"
if [[ ! -f "$YADAI_TEST" ]]; then
  printf 'Missing comparison Python test after update: %s\n' "$YADAI_TEST" >&2
  exit 2
fi
YADAI_AITER_IMPORT="$(
  cd "$YADAI_REPO"
  PYTHONPATH="$YADAI_REPO${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" -c \
      'from pathlib import Path; import aiter; print(Path(aiter.__file__).resolve())'
)"
case "$YADAI_AITER_IMPORT" in
  "$YADAI_REPO"/*) ;;
  *)
    printf 'Comparison test would import AITER from the wrong tree: %s\n' \
      "$YADAI_AITER_IMPORT" >&2
    exit 2
    ;;
esac
printf 'Comparison Python import: %s\n' "$YADAI_AITER_IMPORT"

# Baseline e2e timing is run_perftest(testGraph=False, use_cuda_event=False,
# num_warmup=5, num_iters=20), returning torch.profiler get_trace_perf device
# time. Force and assert exactly the same contract for the comparison checkout.
YADAI_E2E_CODE="$(cat <<'PY'
import sys

import aiter.test_common as test_common
from aiter import ActivationType
from op_tests.flydsl_tests import test_flydsl_grouped_gemm as grouped_test


original_run_perftest = test_common.run_perftest


def run_perftest_baseline_compatible(*args, **kwargs):
    if kwargs.get("num_warmup") != 5 or kwargs.get("num_iters") != 20:
        raise RuntimeError(
            "comparison e2e timing must match baseline: num_warmup=5, num_iters=20"
        )
    kwargs["testGraph"] = False
    kwargs["use_cuda_event"] = False
    return original_run_perftest(*args, **kwargs)


test_common.run_perftest = run_perftest_baseline_compatible
grouped_test.set_data_format("a4w4")
experts, tokens, topk, model_dim, inter_dim = map(int, sys.argv[1:6])
metrics = grouped_test.run_moe(
    "a4w4",
    experts=experts,
    tokens=tokens,
    topk=topk,
    model_dim=model_dim,
    inter_dim=inter_dim,
    activation=ActivationType.Silu,
    use_bias=False,
    bench=True,
    kernel_bench=False,
    warmup=5,
    iters=20,
    seed=0,
    data_init="zero",
    scale_init="zero",
    check_aot_cache=False,
)
print(
    "[yadai-e2e graph=False] fused_moe end-to-end us = "
    f"{metrics['us']:.3f}",
    flush=True,
)
print(
    "[yadai-e2e graph=False] "
    f"logits_diff={metrics['logits_diff']:.4e} "
    f"rel_l2={metrics['rel_l2']:.4e} "
    f"pass={metrics['passed']}",
    flush=True,
)
print(
    "[yadai-e2e timing] backend=torch.profiler "
    "metric=get_trace_perf.device_time_sum "
    "num_warmup=5 num_iters=20 testGraph=False use_cuda_event=False",
    flush=True,
)
if not metrics["passed"]:
    raise SystemExit(4)
PY
)"
YADAI_E2E_SCRIPT="$YADAI_LOG_DIR/run_e2e_graph_false.py"
printf '%s\n' "$YADAI_E2E_CODE" >"$YADAI_E2E_SCRIPT"

printf '\n===== mode base-isolated: ROCm/%s GEMM1/GEMM2 =====\n' \
  "$TARGET_BRANCH"
for ((round_id = 0; round_id < ROUNDS; ++round_id)); do
  set_round_shapes "$round_id"
  printf '\n===== ROCm/%s kernel round %s: %s =====\n' \
    "$TARGET_BRANCH" "$round_id" "${ROUND_SHAPES[*]}"
  for shape_spec in "${ROUND_SHAPES[@]}"; do
    IFS='|' read -r shape_id experts tokens topk model_dim inter_dim \
      <<<"$shape_spec"
    set_common_args "$experts" "$tokens" "$topk" "$model_dim" "$inter_dim"
    require_gpu_idle \
      "before ROCm/$TARGET_BRANCH kernel round $round_id $shape_id"
    printf '\n===== ROCm/%s kernel round %s shape %s =====\n' \
      "$TARGET_BRANCH" "$round_id" "$shape_id"
    (
      cd "$YADAI_REPO"
      AITER_META_DIR="$YADAI_REPO" \
      PYTHONPATH="$YADAI_REPO${PYTHONPATH:+:$PYTHONPATH}" \
        "$PYTHON_BIN" -u op_tests/flydsl_tests/test_flydsl_grouped_gemm.py \
        --scenario kernel \
        "${COMMON_ARGS[@]}"
    ) 2>&1 | tee "$YADAI_LOG_DIR/${shape_id}_kernel_r${round_id}.log"
    require_gpu_idle \
      "after ROCm/$TARGET_BRANCH kernel round $round_id $shape_id"
  done
done

printf '\n===== mode base-e2e: ROCm/%s graph-off MoE e2e =====\n' \
  "$TARGET_BRANCH"
for ((round_id = 0; round_id < ROUNDS; ++round_id)); do
  set_round_shapes "$round_id"
  printf '\n===== ROCm/%s e2e round %s: %s =====\n' \
    "$TARGET_BRANCH" "$round_id" "${ROUND_SHAPES[*]}"
  for shape_spec in "${ROUND_SHAPES[@]}"; do
    IFS='|' read -r shape_id experts tokens topk model_dim inter_dim \
      <<<"$shape_spec"
    require_gpu_idle \
      "before ROCm/$TARGET_BRANCH e2e round $round_id $shape_id"
    printf '\n===== ROCm/%s e2e round %s shape %s =====\n' \
      "$TARGET_BRANCH" "$round_id" "$shape_id"
    (
      cd "$YADAI_REPO"
      AITER_META_DIR="$YADAI_REPO" \
      PYTHONPATH="$YADAI_REPO${PYTHONPATH:+:$PYTHONPATH}" \
        "$PYTHON_BIN" -u "$YADAI_E2E_SCRIPT" \
          "$experts" "$tokens" "$topk" "$model_dim" "$inter_dim"
    ) 2>&1 | tee "$YADAI_LOG_DIR/${shape_id}_e2e_r${round_id}.log"
    require_gpu_idle \
      "after ROCm/$TARGET_BRANCH e2e round $round_id $shape_id"
  done
done

YADAI_SUMMARY="$YADAI_LOG_DIR/summary.md"
"$PYTHON_BIN" - \
  "$SHAPE_MANIFEST" \
  "$YADAI_LOG_DIR" \
  "$YADAI_SUMMARY" \
  "$YADAI_COMMIT_LABEL" \
  "$TARGET_BRANCH" \
  "$YADAI_REPO" \
  "$PYTHON_BIN" \
  "$YADAI_E2E_SCRIPT" \
  "$ROUNDS" <<'PY'
import csv
import os
import pathlib
import re
import shlex
import statistics
import sys

(
    manifest_name,
    log_dir_name,
    output_name,
    commit,
    branch,
    repo,
    python_bin,
    e2e_script,
    rounds,
) = sys.argv[1:]
log_dir = pathlib.Path(log_dir_name)
with open(manifest_name, encoding="utf-8", newline="") as src:
    shapes = list(csv.DictReader(src, delimiter="\t"))


def extract(path: pathlib.Path, pattern: str) -> float:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(pattern, text)
    if match is None:
        raise SystemExit(f"missing metric in {path}: {pattern}")
    return float(match.group(1))


def extract_accuracy(path: pathlib.Path) -> tuple[float, float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"\[yadai-e2e graph=False\] logits_diff=([0-9.eE+-]+) "
        r"rel_l2=([0-9.eE+-]+) pass=True",
        text,
    )
    if match is None:
        raise SystemExit(f"missing graph=False accuracy result in {path}")
    return float(match.group(1)), float(match.group(2))


def samples(values: list[float], digits=3) -> str:
    return ", ".join(f"{value:.{digits}f}" for value in values)


lines = [
    f"# ROCm/{branch} multi-shape benchmark",
    "",
    f"- branch: `ROCm/{branch}`",
    f"- commit: `{commit}`",
    f"- rounds per shape: `{rounds}`",
    "- MoE e2e timing: `testGraph=False`, `use_cuda_event=False`, "
    "`num_warmup=5`, `num_iters=20`, torch profiler "
    "`get_trace_perf(...).device_time_sum`",
    "",
    "| shape | GEMM1 samples (us) | GEMM1 median (us) | GEMM2 samples (us) | GEMM2 median (us) | MoE e2e samples (us) | MoE median (us) | max logits_diff | max rel_l2 | pass |",
    "|---|---|---:|---|---:|---|---:|---:|---:|:---:|",
]

for shape in shapes:
    shape_id = shape["shape_id"]
    kernel_logs = [
        log_dir / f"{shape_id}_kernel_r{round_id}.log"
        for round_id in range(int(rounds))
    ]
    e2e_logs = [
        log_dir / f"{shape_id}_e2e_r{round_id}.log"
        for round_id in range(int(rounds))
    ]
    for path in kernel_logs + e2e_logs:
        if not path.is_file():
            raise SystemExit(f"missing comparison log: {path}")

    gemm1 = [
        extract(path, r"\[kernel-bench a4w4 silu\] gemm1: us = ([0-9.]+)")
        for path in kernel_logs
    ]
    gemm2 = [
        extract(path, r"\[kernel-bench a4w4 silu\] gemm2: us = ([0-9.]+)")
        for path in kernel_logs
    ]
    e2e = [
        extract(
            path,
            r"\[yadai-e2e graph=False\] fused_moe end-to-end us = ([0-9.]+)",
        )
        for path in e2e_logs
    ]
    accuracy = [extract_accuracy(path) for path in e2e_logs]
    label = (
        f"E{shape['experts']}/T{shape['tokens']}/topk{shape['topk']}/"
        f"M{shape['model_dim']}/I{shape['inter_dim']}"
    )
    lines.append(
        f"| {label} | {samples(gemm1)} | {statistics.median(gemm1):.3f} | "
        f"{samples(gemm2)} | {statistics.median(gemm2):.3f} | "
        f"{samples(e2e)} | {statistics.median(e2e):.3f} | "
        f"{max(value[0] for value in accuracy):.4e} | "
        f"{max(value[1] for value in accuracy):.4e} | True |"
    )

pythonpath = repo + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
env = {
    "ENABLE_CK": "0",
    "AITER_MOE_EXPERT_BALANCE": "true",
    "AITER_LOG_MORE": "1",
    "AITER_USE_GROUPED_GEMM": "1",
    "AITER_GROUPED_DEBUG": "0",
    "AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE": "1",
    "AITER_META_DIR": repo,
    "PYTHONPATH": pythonpath,
}
env_text = " \\\n  ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
lines.extend(
    [
        "",
        "## Reproduction command templates",
        "",
        "```bash",
        f"cd {shlex.quote(repo)}",
        f"{env_text} \\",
        f"  {shlex.join([python_bin, '-u', 'op_tests/flydsl_tests/test_flydsl_grouped_gemm.py', '--scenario', 'kernel', '--data-format', 'a4w4', '--experts', '<E>', '--tokens', '<T>', '--topk', '<TOPK>', '--model-dim', '<M>', '--inter-dim', '<I>', '--act', 'silu', '--no-bias', '--no-check-aot-cache', '--warmup', '5', '--iters', '20', '--data-init', 'zero', '--scale-init', 'zero'])}",
        "",
        f"{env_text} \\",
        f"  {shlex.join([python_bin, '-u', e2e_script, '<E>', '<T>', '<TOPK>', '<M>', '<I>'])}",
        "```",
    ]
)
path = pathlib.Path(output_name)
path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
fi

if ((RUN_CURRENT)); then
  printf '\n============== current summary ==============\n'
  cat "$BASELINE_SUMMARY"
  printf '=============================================\n'
  printf 'Current summary: %s\n' "$BASELINE_SUMMARY"
fi
if ((RUN_BASE)); then
  printf '\n============= ROCm/%s summary =============\n' "$TARGET_BRANCH"
  cat "$YADAI_SUMMARY"
  printf '================================================\n'
  printf 'ROCm/%s summary: %s\n' "$TARGET_BRANCH" "$YADAI_SUMMARY"
fi
if ((RUN_CURRENT && RUN_BASE)); then
  COMPARISON_SUMMARY="$LOG_ROOT/comparison.md"
  "$PYTHON_BIN" - \
    "$BASELINE_SUMMARY" \
    "$YADAI_SUMMARY" \
    "$COMPARISON_SUMMARY" \
    "$BASELINE_BRANCH_LABEL" \
    "$BASELINE_COMMIT_LABEL" \
    "$TARGET_BRANCH" \
    "$YADAI_COMMIT_LABEL" <<'PY'
import pathlib
import sys

(
    current_summary_name,
    base_summary_name,
    output_name,
    current_branch,
    current_commit,
    base_branch,
    base_commit,
) = sys.argv[1:]


def table_rows(path: pathlib.Path) -> dict[str, list[str]]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| E"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        rows[cells[0]] = cells
    return rows


current_rows = table_rows(pathlib.Path(current_summary_name))
base_rows = table_rows(pathlib.Path(base_summary_name))
if current_rows.keys() != base_rows.keys():
    missing_current = sorted(base_rows.keys() - current_rows.keys())
    missing_base = sorted(current_rows.keys() - base_rows.keys())
    raise SystemExit(
        "summary shape mismatch: "
        f"missing_current={missing_current}, missing_base={missing_base}"
    )


def speedup(current: float, base: float) -> float:
    return (base - current) / base * 100.0


lines = [
    "# Current vs ROCm/main performance comparison",
    "",
    f"- current: `{current_branch}@{current_commit[:12]}`",
    f"- comparison: `ROCm/{base_branch}@{base_commit[:12]}`",
    "- positive percentages mean current is faster than ROCm/main",
    "- current GEMM1 uses the pipeline median, including standalone quant when enabled",
    "",
    "| shape | current GEMM1 pipeline (us) | ROCm/main GEMM1 (us) | GEMM1 speedup | current GEMM2 (us) | ROCm/main GEMM2 (us) | GEMM2 speedup | current MoE e2e (us) | ROCm/main MoE e2e (us) | MoE speedup | current pass | ROCm/main pass |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|",
]

for shape, current in current_rows.items():
    base = base_rows[shape]
    current_gemm1 = float(current[4])
    current_gemm2 = float(current[6])
    current_moe = float(current[8])
    base_gemm1 = float(base[2])
    base_gemm2 = float(base[4])
    base_moe = float(base[6])
    lines.append(
        f"| {shape} | {current_gemm1:.3f} | {base_gemm1:.3f} | "
        f"{speedup(current_gemm1, base_gemm1):+.2f}% | "
        f"{current_gemm2:.3f} | {base_gemm2:.3f} | "
        f"{speedup(current_gemm2, base_gemm2):+.2f}% | "
        f"{current_moe:.3f} | {base_moe:.3f} | "
        f"{speedup(current_moe, base_moe):+.2f}% | "
        f"{current[9]} | {base[9]} |"
    )

path = pathlib.Path(output_name)
path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n================ comparison ================")
print(path.read_text(encoding="utf-8"), end="")
print("============================================")
print(f"Comparison summary: {path}")
PY
fi
printf 'All logs: %s\n' "$LOG_ROOT"
