#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ROUNDS="${ROUNDS:-3}"
WARMUP="${WARMUP:-5}"
VERSION="${1:-}"
SHAPES_JSON="${2:-}"

BASELINE_COMMIT="314ab7d48a077f3ccc055e7d353040c10a5f7fe5"
OPTIMIZED_COMMIT="b899eaa23ebda8985440332fe3f40570633d2d9e"
TEST_FILE="$REPO_ROOT/op_tests/flydsl_tests/test_flydsl_grouped_gemm.py"

usage() {
  cat <<'EOF'
usage: bash my_code/profile_serving_moe_shapes.sh baseline|optimized SHAPES_JSON

The repository must already be checked out at the corresponding commit.
SHAPES_JSON is produced from the serving trace and supplies phase, T, and calls.

Environment:
  ROUNDS=N    Number of rounds per shape (default: 3)
  WARMUP=N    Untimed warmup forwards per case (default: 5)
  OUT_DIR=... Output directory override
  PYTHON_BIN  Python executable (default: python3)
EOF
}

if [[ "$VERSION" != "baseline" && "$VERSION" != "optimized" ]]; then
  usage >&2
  exit 2
fi
if [[ -z "$SHAPES_JSON" || ! -f "$SHAPES_JSON" ]]; then
  printf 'Missing serving-shape JSON: %s\n' "$SHAPES_JSON" >&2
  usage >&2
  exit 2
fi
if [[ ! "$ROUNDS" =~ ^[1-9][0-9]*$ ]]; then
  printf 'ROUNDS must be a positive integer, got %q\n' "$ROUNDS" >&2
  exit 2
fi
if [[ ! "$WARMUP" =~ ^[0-9]+$ ]]; then
  printf 'WARMUP must be a non-negative integer, got %q\n' "$WARMUP" >&2
  exit 2
fi

case "$VERSION" in
  baseline)
    EXPECTED_COMMIT="$BASELINE_COMMIT"
    ;;
  optimized)
    EXPECTED_COMMIT="$OPTIMIZED_COMMIT"
    ;;
esac

cd "$REPO_ROOT"
ACTUAL_COMMIT="$(git rev-parse HEAD)"
if [[ "$ACTUAL_COMMIT" != "$EXPECTED_COMMIT" ]]; then
  printf 'Expected %s commit %s, got %s\n' \
    "$VERSION" "$EXPECTED_COMMIT" "$ACTUAL_COMMIT" >&2
  exit 2
fi
if [[ ! -f "$TEST_FILE" ]]; then
  printf 'Missing common test harness: %s\n' "$TEST_FILE" >&2
  exit 2
fi

mapfile -t SHAPES < <(
  "$PYTHON_BIN" - "$SHAPES_JSON" "$EXPECTED_COMMIT" <<'PY'
import json
import sys
from pathlib import Path


path = Path(sys.argv[1])
expected_commit = sys.argv[2]
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("aiter_commit") != expected_commit:
    raise SystemExit(
        f"shape JSON commit {payload.get('aiter_commit')!r} != {expected_commit!r}"
    )
records = []
for record in payload.get("records", []):
    phase = str(record.get("phase"))
    if phase == "eager_decode":
        phase = "decode"
    if phase not in ("decode", "prefill"):
        continue
    records.append(
        (phase, int(record["moe_tokens"]), int(record["calls"]))
    )
for phase, tokens, calls in sorted(set(records)):
    print(f"{phase}:{tokens}:{calls}")
PY
)
if [[ "${#SHAPES[@]}" -eq 0 ]]; then
  printf 'No decode/prefill records in %s\n' "$SHAPES_JSON" >&2
  exit 2
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/serving_moe_shape_profiles/${VERSION}_${RUN_ID}}"
mkdir -p "$OUT_DIR"

COMMON_ENV=(
  AITER_USE_GROUPED_GEMM=1
  AITER_GROUPED_DEBUG=0
  ENABLE_CK=0
  FLYDSL_DUMP_IR=0
  AITER_LOG_MORE=0
  AITER_FORCE_GFX1250=1
  GPU_ARCHS=gfx1250
  CU_NUM=256
  AITER_MOE_EXPERT_BALANCE=true
  AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1
)

CLEAR_OPT_ENV=(
  -u AITER_FLYDSL_MXFP4_CLUSTER_N
  -u AITER_TDM_NEXT_STAGE_PREFETCH
  -u AITER_FLYDSL_GEMM1_A_PRESHUFFLE
  -u AITER_FLYDSL_GEMM1_WAVES_PER_TENSOR_TDM
  -u AITER_FLYDSL_GEMM1_MMA_GROUP
  -u AITER_FLYDSL_GEMM1_FENCE_COVER_MMA
  -u AITER_FLYDSL_GEMM1_DISABLE_XDL_ARB_STALL
  -u AITER_FLYDSL_GEMM1_WMMA_REUSE
  -u AITER_FLYDSL_GEMM1_OVERLAP_OUTPUT_STORE
  -u AITER_FLYDSL_GEMM2_A_PRESHUFFLE
  -u AITER_FLYDSL_GEMM2_A_PRESHUFFLE_PRODUCER
  -u AITER_FLYDSL_GEMM2_A_PRESHUFFLE_RPW
  -u AITER_FLYDSL_GEMM2_A_PRESHUFFLE_PREFETCH
  -u AITER_FLYDSL_GEMM2_WAVES_PER_TENSOR_TDM
  -u AITER_FLYDSL_GEMM2_SCHEDULE_HINTS
  -u AITER_FLYDSL_GEMM2_MMA_GROUP
  -u AITER_FLYDSL_GEMM2_FENCE_COVER_MMA
  -u AITER_FLYDSL_GEMM2_OVERLAP_OUTPUT_STORE
  -u AITER_FLYDSL_GEMM2_OUTPUT_SPLIT_WM
  -u AITER_FLYDSL_GEMM2_OUTPUT_WAVE_SPLIT
)

OPT_ENV=(
  AITER_FLYDSL_GEMM1_A_PRESHUFFLE=1
  AITER_FLYDSL_GEMM1_WAVES_PER_TENSOR_TDM=2
  AITER_FLYDSL_GEMM1_MMA_GROUP=4
  AITER_FLYDSL_GEMM1_FENCE_COVER_MMA=28
  AITER_FLYDSL_GEMM1_DISABLE_XDL_ARB_STALL=0
  AITER_FLYDSL_GEMM1_WMMA_REUSE=1
  AITER_FLYDSL_GEMM1_OVERLAP_OUTPUT_STORE=1
  AITER_FLYDSL_GEMM2_A_PRESHUFFLE=1
  AITER_FLYDSL_GEMM2_A_PRESHUFFLE_PRODUCER=rowgroup
  AITER_FLYDSL_GEMM2_A_PRESHUFFLE_RPW=2
  AITER_FLYDSL_GEMM2_A_PRESHUFFLE_PREFETCH=2
  AITER_FLYDSL_GEMM2_WAVES_PER_TENSOR_TDM=2
  AITER_FLYDSL_GEMM2_SCHEDULE_HINTS=1
  AITER_FLYDSL_GEMM2_MMA_GROUP=4
  AITER_FLYDSL_GEMM2_FENCE_COVER_MMA=28
  AITER_FLYDSL_GEMM2_OVERLAP_OUTPUT_STORE=1
  AITER_FLYDSL_GEMM2_OUTPUT_SPLIT_WM=3
  AITER_FLYDSL_GEMM2_OUTPUT_WAVE_SPLIT=1
)

run_case() {
  local phase="$1"
  local tokens="$2"
  local forward_calls="$3"
  local round="$4"
  local json_file="$OUT_DIR/${phase}_t${tokens}_r${round}.json"
  local log_file="$OUT_DIR/${phase}_t${tokens}_r${round}.log"
  local -a version_env=()
  if [[ "$VERSION" == "optimized" ]]; then
    version_env=("${OPT_ENV[@]}")
  fi

  printf '\n===== version=%s phase=%s T=%s calls=%s round=%s/%s =====\n' \
    "$VERSION" "$phase" "$tokens" "$forward_calls" "$round" "$ROUNDS"

  env "${CLEAR_OPT_ENV[@]}" "${COMMON_ENV[@]}" "${version_env[@]}" \
    "$PYTHON_BIN" - \
      "$TEST_FILE" "$json_file" "$VERSION" "$ACTUAL_COMMIT" \
      "$phase" "$tokens" "$forward_calls" "$round" "$WARMUP" \
      2>&1 <<'PY' | tee "$log_file"
from __future__ import annotations

import importlib.util
import json
import sys
from collections import OrderedDict
from pathlib import Path

import torch
import torch.profiler as torch_profiler


(
    test_file,
    output_file,
    version,
    commit,
    phase,
    tokens_raw,
    calls_raw,
    round_raw,
    warmup_raw,
) = sys.argv[1:]
tokens = int(tokens_raw)
forward_calls = int(calls_raw)
round_id = int(round_raw)
warmup = int(warmup_raw)

spec = importlib.util.spec_from_file_location("serving_moe_shape_test", test_file)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load test harness: {test_file}")
test_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_module)

import aiter.test_common as test_common
from aiter import ActivationType

profile_result: dict = {}


def exact_profile_run_perftest(
    func,
    *args,
    num_iters=101,
    num_warmup=2,
    testGraph=False,
    **kwargs,
):
    del testGraph
    for _ in range(num_warmup):
        out = func(*args, **kwargs)
    torch.cuda.synchronize()

    with torch_profiler.profile(
        activities=[
            torch_profiler.ProfilerActivity.CPU,
            torch_profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for _ in range(num_iters):
            out = func(*args, **kwargs)
        torch.cuda.synchronize()

    aggregated: OrderedDict[str, dict[str, float | int | str]] = OrderedDict()
    for event in prof.events():
        if "CUDA" not in str(getattr(event, "device_type", "")):
            continue
        elapsed = float(getattr(event, "self_device_time_total", 0.0) or 0.0)
        if elapsed <= 0:
            continue
        name = str(getattr(event, "name", ""))
        record = aggregated.setdefault(
            name,
            {"name": name, "calls": 0, "total_us": 0.0},
        )
        record["calls"] += 1
        record["total_us"] += elapsed

    kernels = []
    for record in aggregated.values():
        record["total_us"] = round(float(record["total_us"]), 6)
        record["avg_us"] = round(
            float(record["total_us"]) / int(record["calls"]), 6
        )
        kernels.append(record)
    kernels.sort(key=lambda item: (-float(item["total_us"]), str(item["name"])))
    total_us = sum(float(item["total_us"]) for item in kernels)
    profile_result.update(
        {
            "profiled_forward_calls": int(num_iters),
            "warmup_forward_calls": int(num_warmup),
            "total_cuda_kernel_time_us": round(total_us, 6),
            "average_cuda_kernel_time_per_forward_us": round(
                total_us / int(num_iters), 6
            ),
            "kernels": kernels,
        }
    )
    return out, total_us / int(num_iters)


test_common.run_perftest = exact_profile_run_perftest

metrics = test_module.run_moe(
    "a4w4",
    experts=256,
    tokens=tokens,
    topk=8,
    model_dim=7168,
    inter_dim=2048,
    activation=ActivationType.Silu,
    use_bias=False,
    raise_on_fail=False,
    bench=True,
    kernel_bench=False,
    warmup=warmup,
    iters=forward_calls,
    seed=0,
    data_init="uniform",
    scale_init="auto",
    check_aot_cache=False,
)

payload = {
    "version": version,
    "commit": commit,
    "phase": phase,
    "tokens": tokens,
    "forward_calls": forward_calls,
    "round": round_id,
    "shape": {
        "data_format": "a4w4",
        "experts": 256,
        "topk": 8,
        "model_dim": 7168,
        "inter_dim": 2048,
        "activation": "silu",
        "bias": False,
        "data_init": "uniform",
        "scale_init": "auto",
    },
    "correctness": {
        "pass": bool(metrics["passed"]),
        "logits_diff": float(metrics["logits_diff"]),
        "rel_l2": float(metrics["rel_l2"]),
    },
    **profile_result,
}
Path(output_file).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(
    "[shape-profile] "
    f"phase={phase} T={tokens} calls={forward_calls} "
    f"total_cuda_us={payload['total_cuda_kernel_time_us']:.3f} "
    f"avg_cuda_us={payload['average_cuda_kernel_time_per_forward_us']:.3f} "
    f"kernels={len(payload['kernels'])} pass={payload['correctness']['pass']}",
    flush=True,
)
if not payload["correctness"]["pass"]:
    raise SystemExit(4)
PY
}

for ((round = 1; round <= ROUNDS; ++round)); do
  if ((round % 2 == 1)); then
    ROUND_SHAPES=("${SHAPES[@]}")
  else
    ROUND_SHAPES=()
    for ((i = ${#SHAPES[@]} - 1; i >= 0; --i)); do
      ROUND_SHAPES+=("${SHAPES[i]}")
    done
  fi
  for spec in "${ROUND_SHAPES[@]}"; do
    IFS=: read -r phase tokens forward_calls <<<"$spec"
    run_case "$phase" "$tokens" "$forward_calls" "$round"
  done
done

"$PYTHON_BIN" - "$OUT_DIR" "$VERSION" "$ACTUAL_COMMIT" <<'PY'
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


out_dir = Path(sys.argv[1])
version = sys.argv[2]
commit = sys.argv[3]
records = [
    json.loads(path.read_text(encoding="utf-8"))
    for path in sorted(out_dir.glob("*_t*_r*.json"))
]
grouped = defaultdict(list)
for record in records:
    grouped[(record["phase"], record["tokens"])].append(record)

cases = []
for (phase, tokens), rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
    totals = [float(row["total_cuda_kernel_time_us"]) for row in rows]
    averages = [float(row["average_cuda_kernel_time_per_forward_us"]) for row in rows]
    kernel_names = sorted({kernel["name"] for row in rows for kernel in row["kernels"]})
    cases.append(
        {
            "phase": phase,
            "tokens": tokens,
            "forward_calls": rows[0]["forward_calls"],
            "rounds": len(rows),
            "total_cuda_kernel_time_us_samples": totals,
            "total_cuda_kernel_time_us_median": statistics.median(totals),
            "average_cuda_kernel_time_per_forward_us_samples": averages,
            "average_cuda_kernel_time_per_forward_us_median": statistics.median(averages),
            "kernel_names": kernel_names,
            "kernel_names_stable_across_rounds": all(
                {kernel["name"] for kernel in row["kernels"]} == set(kernel_names)
                for row in rows
            ),
            "correctness_pass_all_rounds": all(row["correctness"]["pass"] for row in rows),
        }
    )

summary = {
    "version": version,
    "commit": commit,
    "rounds": len({record["round"] for record in records}),
    "cases": cases,
}
(out_dir / "summary.json").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
)

lines = [
    "| phase | T | forward calls | total CUDA kernel time samples (us) | median total (us) | median per-forward (us) | kernel names stable | pass |",
    "|---|---:|---:|---|---:|---:|:---:|:---:|",
]
for case in cases:
    samples = ", ".join(f"{value:.3f}" for value in case["total_cuda_kernel_time_us_samples"])
    lines.append(
        f"| {case['phase']} | {case['tokens']} | {case['forward_calls']} | "
        f"{samples} | {case['total_cuda_kernel_time_us_median']:.3f} | "
        f"{case['average_cuda_kernel_time_per_forward_us_median']:.3f} | "
        f"{case['kernel_names_stable_across_rounds']} | "
        f"{case['correctness_pass_all_rounds']} |"
    )
lines.append("")
lines.append("## Kernel names")
for case in cases:
    lines.append("")
    lines.append(f"### {case['phase']} T={case['tokens']}")
    lines.extend(f"- `{name}`" for name in case["kernel_names"])
(out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print(f"Results: {out_dir}")
PY
