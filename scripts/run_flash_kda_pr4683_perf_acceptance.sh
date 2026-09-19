#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

# One-command coverage and acceptance for the shapes/input contract in
# ROCm/aiter PR #4683's original six-row gfx950 dense table. The canonical
# Python entry point owns the matrix, correctness checks, paired statistics,
# and fail-closed performance decision; this wrapper owns clean-checkout and
# isolated-build provenance.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="$(command -v -- "${PYTHON:-python3}")"
readonly EXPECTED_SEED_CELLS=18
readonly EXPECTED_LOGICAL_CELLS=6
readonly WARMUP=20
readonly REPEAT=120
readonly BOOTSTRAP_RESAMPLES=10000
readonly MIN_NATIVE_SPEEDUP=1.03
readonly FORMAL_PLAN_SHA256=8e651f53b430ef4fe8a947de1ee0ed64687a7d8192369351b65ab71a0b4a1fab

usage() {
    cat <<'EOF'
Usage: scripts/run_flash_kda_pr4683_perf_acceptance.sh [OUTPUT_DIR]

Run the canonical native-HIP versus direct latest-main Triton dense eager matrix
on exactly one visible 256-CU gfx950 GPU. Select the GPU with
ROCR_VISIBLE_DEVICES (or HIP_VISIBLE_DEVICES) before invoking the script.

The checkout must be clean and OUTPUT_DIR must be a new directory outside the
checkout. The default is a new /tmp/flash-kda-pr4683-* directory. The fixed
matrix is B=1, T={8192,16384}, H={32,64,96}, K=V=128 crossed with seeds
42/43/44. Each seed cell requires >=1.03x native p50 speedup, >50% paired
native wins, negative upper bounds for paired p50/mean bootstrap confidence
intervals, and output rRMS <=0.04.
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    usage
    exit 0
fi
if (($# > 1)); then
    usage >&2
    exit 2
fi

cd "$REPO_ROOT"
repo_is_clean() {
    local submodule_status untracked
    git -c safe.directory="$REPO_ROOT" diff --quiet -- || return 1
    git -c safe.directory="$REPO_ROOT" diff --cached --quiet -- || return 1
    untracked="$(git -c safe.directory="$REPO_ROOT" \
        ls-files --others --exclude-standard)" || return 1
    [[ -z $untracked ]] || return 1
    submodule_status="$(git -c safe.directory="$REPO_ROOT" \
        submodule status --recursive)" || return 1
    if grep -Eq '^[-+U]' <<<"$submodule_status"; then
        return 1
    fi
    git -c safe.directory="$REPO_ROOT" submodule foreach --recursive \
        --quiet 'submodule_tree="$(git status --porcelain --untracked-files=all)" || exit $?; test -z "$submodule_tree"'
}

if ! repo_is_clean; then
    echo "ERROR: acceptance requires a clean checkout" >&2
    git -c safe.directory="$REPO_ROOT" status --short --untracked-files=all >&2
    exit 2
fi

INITIAL_HEAD="$(git -c safe.directory="$REPO_ROOT" rev-parse HEAD)"
INITIAL_TREE="$(git -c safe.directory="$REPO_ROOT" rev-parse HEAD^{tree})"
if (($# == 1)); then
    OUTPUT_DIR="$(realpath -m -- "$1")"
    case "$OUTPUT_DIR" in
        "$REPO_ROOT"|"$REPO_ROOT"/*)
            echo "ERROR: output directory must be outside the checkout" >&2
            exit 2
            ;;
    esac
    [[ ! -e $OUTPUT_DIR ]] || {
        echo "ERROR: output path already exists: $OUTPUT_DIR" >&2
        exit 2
    }
    mkdir -p "$OUTPUT_DIR"
else
    OUTPUT_PARENT="$(realpath -m -- "${TMPDIR:-/tmp}")"
    case "$OUTPUT_PARENT" in
        "$REPO_ROOT"|"$REPO_ROOT"/*)
            echo "ERROR: TMPDIR must be outside the checkout" >&2
            exit 2
            ;;
    esac
    mkdir -p "$OUTPUT_PARENT"
    OUTPUT_DIR="$(mktemp -d "$OUTPUT_PARENT/flash-kda-pr4683-XXXXXXXX")"
fi

ACCEPTANCE_STAGE=setup
acceptance_exit() {
    local exit_code=$?
    trap - EXIT
    set +e
    local final_head final_tree repository_ok
    final_head="$(git -c safe.directory="$REPO_ROOT" rev-parse HEAD 2>/dev/null)"
    final_tree="$(
        git -c safe.directory="$REPO_ROOT" rev-parse HEAD^{tree} 2>/dev/null
    )"
    repository_ok=1
    if [[ $final_head != "$INITIAL_HEAD" || $final_tree != "$INITIAL_TREE" ]] ||
        ! repo_is_clean; then
        repository_ok=0
        if ((exit_code == 0)); then
            exit_code=1
            ACCEPTANCE_STAGE=repository-changed
        fi
    fi

    if ! {
        printf 'initial_head=%s\n' "$INITIAL_HEAD"
        printf 'final_head=%s\n' "$final_head"
        printf 'initial_tree=%s\n' "$INITIAL_TREE"
        printf 'final_tree=%s\n' "$final_tree"
        printf 'clean_and_unchanged=%s\n' "$repository_ok"
    } >"$OUTPUT_DIR/repository-after.txt"; then
        if ((exit_code == 0)); then
            exit_code=1
            ACCEPTANCE_STAGE=repository-evidence-finalization
        fi
    fi

    write_acceptance_status() {
        local status=failed
        if ((exit_code == 0)); then
            status=passed
        fi
        {
            printf 'status=%s\n' "$status"
            printf 'exit_code=%s\n' "$exit_code"
            printf 'stage=%s\n' "$ACCEPTANCE_STAGE"
            printf 'git_head=%s\n' "$INITIAL_HEAD"
            printf 'git_tree=%s\n' "$INITIAL_TREE"
            printf 'logical_cells=%s\n' "$EXPECTED_LOGICAL_CELLS"
            printf 'seed_cells=%s\n' "$EXPECTED_SEED_CELLS"
            printf 'seeds=42,43,44\n'
            printf 'warmup=%s\n' "$WARMUP"
            printf 'repeat=%s\n' "$REPEAT"
            printf 'bootstrap_resamples=%s\n' "$BOOTSTRAP_RESAMPLES"
            printf 'minimum_native_speedup=%s\n' "$MIN_NATIVE_SPEEDUP"
            printf 'execution=eager\n'
            printf 'python=%s\n' "$PYTHON_BIN"
            printf 'benchmark_exit=%s\n' "${BENCHMARK_EXIT:-not-run}"
        } >"$OUTPUT_DIR/status.txt"
    }
    if ! write_acceptance_status; then
        if ((exit_code == 0)); then
            exit_code=1
            ACCEPTANCE_STAGE=status-finalization
        fi
        write_acceptance_status || true
    fi

    local candidate
    local -a artifacts=()
    for candidate in \
        status.txt repository-after.txt git-head.txt git-tree.txt \
        git-provenance.log static-self-test.json static-self-test.log \
        plan.json plan.log environment.log benchmark.log benchmark-status.txt \
        result.json partial-results.jsonl artifact-validation.log \
        jit/module_flash_kda_hip_v2.so; do
        [[ -s $OUTPUT_DIR/$candidate ]] && artifacts+=("$candidate")
    done
    if ((${#artifacts[@]} > 0)); then
        if ! (
            cd "$OUTPUT_DIR" || exit 1
            sha256sum -- "${artifacts[@]}"
        ) >"$OUTPUT_DIR/artifacts.sha256"; then
            rm -f -- "$OUTPUT_DIR/artifacts.sha256"
            if ((exit_code == 0)); then
                exit_code=1
                ACCEPTANCE_STAGE=artifact-checksum-finalization
            fi
            write_acceptance_status || true
        fi
    fi

    if ((repository_ok == 0)); then
        git -c safe.directory="$REPO_ROOT" status --short \
            --untracked-files=all >&2
    fi
    if ((exit_code == 0)); then
        echo "FlashKDA PR #4683 performance acceptance passed: $OUTPUT_DIR"
    else
        printf \
            'FlashKDA PR #4683 performance acceptance failed during %s: %s\n' \
            "$ACCEPTANCE_STAGE" "$OUTPUT_DIR" >&2
    fi
    exit "$exit_code"
}
trap acceptance_exit EXIT

printf '%s\n' "$INITIAL_HEAD" >"$OUTPUT_DIR/git-head.txt"
printf '%s\n' "$INITIAL_TREE" >"$OUTPUT_DIR/git-tree.txt"
{
    git -c safe.directory="$REPO_ROOT" show --no-patch \
        --format=fuller "$INITIAL_HEAD"
    git -c safe.directory="$REPO_ROOT" branch --show-current
    git -c safe.directory="$REPO_ROOT" remote -v
    git -c safe.directory="$REPO_ROOT" submodule status --recursive
} >"$OUTPUT_DIR/git-provenance.log"
echo "FlashKDA PR #4683 acceptance artifacts: $OUTPUT_DIR"

# Route/tuning environment must not silently change either implementation.
# Clear it before the Python process imports AITER or Triton.
while IFS='=' read -r name _; do
    case "$name" in
        AITER_*|FLASH_KDA_*|CHUNK_DELTA_ATTN_*|KDA_*|FLA_*)
            unset "$name"
            ;;
    esac
done < <(env)
unset CK_DIR HIP_KITTENS_DIR OPUS_GEN_CO_DIR PYTHONOPTIMIZE

export AITER_AOT_IMPORT=1
export AITER_FDA_USE_GLUON=0
export AITER_JIT_DIR="$OUTPUT_DIR/jit"
export AITER_META_DIR="$REPO_ROOT"
export GPU_ARCHS=gfx950
export MAX_JOBS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO_ROOT"
export TRITON_CACHE_DIR="$OUTPUT_DIR/triton-cache"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0="$REPO_ROOT"
mkdir -p "$AITER_JIT_DIR" "$TRITON_CACHE_DIR"

BENCHMARK_ENTRY=op_tests/op_benchmarks/triton/bench_flash_kda_pr4683.py

ACCEPTANCE_STAGE=static-self-test
"$PYTHON_BIN" -u "$BENCHMARK_ENTRY" \
    --static-self-test \
    --output "$OUTPUT_DIR/static-self-test.json" \
    2>&1 | tee "$OUTPUT_DIR/static-self-test.log"

ACCEPTANCE_STAGE=plan
"$PYTHON_BIN" -u "$BENCHMARK_ENTRY" \
    --print-plan \
    --output "$OUTPUT_DIR/plan.json" \
    2>&1 | tee "$OUTPUT_DIR/plan.log"

ACCEPTANCE_STAGE=environment
"$PYTHON_BIN" - <<'PY' 2>&1 | tee "$OUTPUT_DIR/environment.log"
import hashlib
import os
import platform
import shutil
import subprocess
import sys

import torch
import triton


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if torch.version.hip is None or not torch.cuda.is_available():
    raise SystemExit("a visible ROCm GPU is required")
if torch.cuda.device_count() != 1:
    raise SystemExit("select exactly one visible ROCm GPU")
properties = torch.cuda.get_device_properties(0)
arch_detail = getattr(properties, "gcnArchName", "unknown")
arch = arch_detail.split(":", 1)[0]
if arch != "gfx950" or properties.multi_processor_count != 256:
    raise SystemExit(
        "acceptance requires the 256-CU gfx950 reference target; "
        f"got {arch}/{properties.multi_processor_count} CU"
    )
print(
    f"GPU: {torch.cuda.get_device_name(0)} "
    f"({arch_detail}, {properties.multi_processor_count} CU)"
)
print(f"PyTorch: {torch.__version__}; ROCm: {torch.version.hip}; {torch.__file__}")
print(f"Triton: {triton.__version__}; {triton.__file__}")
print(
    f"Python: {sys.executable}; {platform.python_version()}; "
    f"sha256={sha256(sys.executable)}"
)
hipcc = shutil.which("hipcc")
print(f"hipcc: {hipcc}")
if hipcc is not None:
    version = subprocess.run(
        [hipcc, "--version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    print(version)
for name in (
    "AITER_AOT_IMPORT",
    "AITER_FDA_USE_GLUON",
    "AITER_JIT_DIR",
    "AITER_META_DIR",
    "CK_DIR",
    "GPU_ARCHS",
    "HIP_KITTENS_DIR",
    "MAX_JOBS",
    "OPUS_GEN_CO_DIR",
    "PYTHONHASHSEED",
    "PYTHONNOUSERSITE",
    "PYTHONOPTIMIZE",
    "PYTHONPATH",
    "TRITON_CACHE_DIR",
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
):
    print(f"{name}={os.environ.get(name)}")
PY

ACCEPTANCE_STAGE=benchmark
set +e
"$PYTHON_BIN" -u "$BENCHMARK_ENTRY" \
    --output "$OUTPUT_DIR/result.json" \
    2>&1 | tee "$OUTPUT_DIR/benchmark.log"
BENCHMARK_EXIT=${PIPESTATUS[0]}
set -e
printf 'benchmark_exit=%s\n' "$BENCHMARK_EXIT" \
    >"$OUTPUT_DIR/benchmark-status.txt"

if ((BENCHMARK_EXIT != 0)); then
    ACCEPTANCE_STAGE=benchmark-failed
    exit "$BENCHMARK_EXIT"
fi

ACCEPTANCE_STAGE=artifact-validation
for artifact in plan.json result.json partial-results.jsonl \
    jit/module_flash_kda_hip_v2.so; do
    [[ -s $OUTPUT_DIR/$artifact ]] || {
        echo "ERROR: successful benchmark did not produce $artifact" >&2
        exit 1
    }
done
"$PYTHON_BIN" - \
    "$OUTPUT_DIR" "$INITIAL_HEAD" "$INITIAL_TREE" "$FORMAL_PLAN_SHA256" \
    "$EXPECTED_LOGICAL_CELLS" "$EXPECTED_SEED_CELLS" <<'PY' \
    2>&1 | tee "$OUTPUT_DIR/artifact-validation.log"
import hashlib
import json
import math
import pathlib
import sys


root = pathlib.Path(sys.argv[1])
expected_head = sys.argv[2]
expected_tree = sys.argv[3]
expected_plan_sha256 = sys.argv[4]
expected_logical_cells = int(sys.argv[5])
expected_seed_cells = int(sys.argv[6])
with (root / "result.json").open(encoding="utf-8") as handle:
    result = json.load(handle)
with (root / "plan.json").open(encoding="utf-8") as handle:
    emitted_plan = json.load(handle)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def finite_number(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


schema = "flash-kda-pr4683-dense-gfx950-v1"
require(result.get("schema") == schema, "result schema")
require(result.get("capture_complete") is True, "capture is incomplete")
require(result.get("performance_gate_evaluated") is True, "gate not evaluated")
require(result.get("performance_gate_passed") is True, "global gate failed")

encoded_plan = json.dumps(
    emitted_plan, sort_keys=True, separators=(",", ":")
).encode()
emitted_plan_sha256 = hashlib.sha256(encoded_plan).hexdigest()
require(emitted_plan_sha256 == expected_plan_sha256, "emitted plan hash")
require(result.get("plan_sha256") == expected_plan_sha256, "result plan hash")
require(result.get("plan") == emitted_plan, "emitted and embedded plans differ")
plan = emitted_plan
require(plan.get("schema") == schema, "plan schema")
require(plan.get("logical_shapes") == expected_logical_cells,
        "plan logical-cell count")
require(plan.get("seed_cells") == expected_seed_cells, "plan seed-cell count")
require(plan.get("seeds") == [42, 43, 44], "plan seed set")
require(plan.get("warmup_rounds") == 20, "plan warmup")
require(plan.get("measured_rounds") == 120, "plan measured rounds")
require(plan.get("bootstrap_resamples") == 10000, "plan bootstrap resamples")
require(len(plan.get("cells", ())) == expected_seed_cells, "plan cells")
comparator = plan.get("comparator", {})
require(comparator.get("forced_direct_import") is True,
        "plan direct Triton import")
require(comparator.get("gluon_disabled") is True,
        "plan Triton-only route")
require(comparator.get("forced_route_environment") == {
            "AITER_FDA_USE_GLUON": "0"},
        "plan route environment")

configuration = result.get("configuration", {})
require(configuration.get("seeds") == [42, 43, 44], "configuration seeds")
require(configuration.get("warmup_rounds") == 20, "configuration warmup")
require(configuration.get("measured_rounds") == 120,
        "configuration measured rounds")
require(configuration.get("bootstrap_resamples") == 10000,
        "configuration bootstrap resamples")
require(configuration.get("minimum_native_speedup") == 1.03,
        "configuration speedup gate")
require(configuration.get("execution") == "eager", "configuration execution")
require(configuration.get("triton_forced_direct_internal_import") is True,
        "configuration Triton import")
require(configuration.get("triton_gluon_disabled") is True,
        "configuration Triton-only route")

environment = result.get("environment", {})
git = environment.get("git", {})
require(git.get("head") == expected_head, "benchmark git HEAD")
require(git.get("tree") == expected_tree, "benchmark git tree")
require(git.get("status_porcelain") == "", "benchmark dirty checkout")
require(environment.get("gpu", {}).get("arch") == "gfx950", "GPU arch")
require(environment.get("gpu", {}).get("compute_units") == 256,
        "GPU CU count")
require(environment.get("gpu", {}).get("visible_device_count") == 1,
        "visible GPU count")
require(environment.get("active_route_control_environment") == {
            "AITER_FDA_USE_GLUON": "0"},
        "route-control environment")
modules = environment.get("modules", {})
native_jit = modules.get("native_jit", {})
expected_module = (root / "jit/module_flash_kda_hip_v2.so").resolve()
require(expected_module.is_file() and expected_module.stat().st_size > 0,
        "native module is missing")
module_sha256 = sha256(expected_module)
require(native_jit.get("matches_expected_jit_path") is True,
        "native JIT expected-path marker")
require(native_jit.get("path") == str(expected_module), "native JIT path")
require(native_jit.get("sha256") == module_sha256, "native JIT SHA256")
require(native_jit.get("raw_abi_version") == 3, "native raw ABI")
require(modules.get("native_build_roots", {}).get("both_match_checkout") is True,
        "native build roots")
require(modules.get("latest_main_triton_python", {}).get(
            "matches_expected_source") is True,
        "latest-main Triton source identity")
require(modules.get("latest_main_triton_python", {}).get(
            "gluon_k1_enabled") is False,
        "latest-main Triton K1 route")
require(modules.get("latest_main_triton_python", {}).get(
            "gluon_k2_enabled") is False,
        "latest-main Triton K2 route")
require(modules.get("latest_main_triton_python", {}).get(
            "triton_only_route_verified") is True,
        "latest-main Triton-only route")

rows = result.get("results")
require(isinstance(rows, list) and len(rows) == expected_seed_cells,
        "result seed-cell count")
expected_cells = {
    (cell["logical_name"], cell["seed"]): cell for cell in plan["cells"]
}
actual_keys = [(row.get("logical_name"), row.get("seed")) for row in rows]
require(len(set(actual_keys)) == expected_seed_cells and
        set(actual_keys) == set(expected_cells),
        "missing, duplicate, or unexpected seed cell")
for row in rows:
    key = (row.get("logical_name"), row.get("seed"))
    label = f"{key[0]}/seed-{key[1]}"
    require(row.get("shape") == expected_cells[key]["shape"], f"{label}: shape")
    require(row.get("execution") == "eager", f"{label}: execution")
    require(row.get("performance_gate_passed") is True, f"{label}: cell gate")
    correctness = row.get("correctness", {})
    rrms = correctness.get("native_vs_triton_output_relative_rms")
    require(correctness.get("passed") is True and finite_number(rrms) and
            float(rrms) <= 0.04, f"{label}: correctness")
    timing = row.get("timing", {})
    speedup = timing.get("triton_over_native_speedup_from_p50")
    paired_speedup = timing.get("paired_triton_over_native_speedup_p50")
    win_fraction = timing.get("native_paired_win_fraction")
    require(timing.get("measured_rounds") == 120, f"{label}: timing rounds")
    require(finite_number(speedup) and float(speedup) >= 1.03,
            f"{label}: p50 speedup")
    require(finite_number(paired_speedup) and float(paired_speedup) >= 1.03,
            f"{label}: paired p50 speedup")
    require(finite_number(win_fraction) and float(win_fraction) > 0.5,
            f"{label}: paired win fraction")
    require(len(row.get("raw_timing_samples", ())) == 240,
            f"{label}: raw timing count")
    require(len(row.get("paired_rounds", ())) == 120,
            f"{label}: paired timing count")
    ci = row.get("paired_bootstrap", {}).get("bootstrap_95pct_ci", {})
    p50_ci = ci.get("p50_delta_us", ())
    mean_ci = ci.get("mean_delta_us", ())
    require(len(p50_ci) == 2 and finite_number(p50_ci[1]) and
            float(p50_ci[1]) < 0.0, f"{label}: p50 CI")
    require(len(mean_ci) == 2 and finite_number(mean_ci[1]) and
            float(mean_ci[1]) < 0.0, f"{label}: mean CI")

summary = result.get("cross_seed_summary", {})
require(summary.get("logical_cells") == expected_logical_cells,
        "summary logical-cell count")
require(summary.get("seed_cells") == expected_seed_cells,
        "summary seed-cell count")
require(summary.get("passed_logical_cells") == expected_logical_cells,
        "summary passed-cell count")
require(summary.get("failed_logical_cells") == [], "summary failed cells")
require(len(summary.get("cells", ())) == expected_logical_cells,
        "summary cell list")
for cell in summary["cells"]:
    label = cell.get("logical_name")
    require(cell.get("performance_gate_passed") is True,
            f"{label}: cross-seed gate")
    ci = cell.get("cross_seed_paired_bootstrap", {}).get(
        "bootstrap_95pct_ci", {}
    )
    require(float(ci.get("p50_delta_us", [0.0, 0.0])[1]) < 0.0,
            f"{label}: cross-seed p50 CI")
    require(float(ci.get("mean_delta_us", [0.0, 0.0])[1]) < 0.0,
            f"{label}: cross-seed mean CI")
gate = result.get("performance_gate", {})
require(gate.get("evaluated") is True and gate.get("passed") is True,
        "global gate")
require(gate.get("failed_logical_cells") == [], "global failed cells")

with (root / "partial-results.jsonl").open(encoding="utf-8") as handle:
    events = [json.loads(line) for line in handle if line.strip()]
require(all(event.get("schema") == schema for event in events),
        "checkpoint schema")
require(not any(event.get("event") == "run-failed" for event in events),
        "checkpoint contains a failure")
completed = [event for event in events if event.get("event") == "seed-cell-complete"]
require(len(completed) == expected_seed_cells, "checkpoint seed-cell count")
completed_keys = [
    (event.get("result", {}).get("logical_name"),
     event.get("result", {}).get("seed"))
    for event in completed
]
require(len(set(completed_keys)) == expected_seed_cells and
        set(completed_keys) == set(expected_cells),
        "checkpoint seed-cell identity")
require(sum(event.get("event") == "run-complete" for event in events) == 1,
        "checkpoint completion count")
require(events and events[-1].get("event") == "run-complete",
        "checkpoint terminal event")
terminal = events[-1]
require(terminal.get("complete") is True, "checkpoint incomplete")
require(terminal.get("capture_complete") is True, "checkpoint capture")
require(terminal.get("result_json_written") is True, "checkpoint result")
require(terminal.get("completed_seed_cells") == expected_seed_cells,
        "checkpoint completed count")
require(terminal.get("total_seed_cells") == expected_seed_cells,
        "checkpoint total count")
require(terminal.get("performance_gate_passed") is True,
        "checkpoint gate failed")
print("artifact validation: PASS (6/6 logical cells, 18/18 seed cells)")
PY

ACCEPTANCE_STAGE=repository-validation
CURRENT_HEAD="$(git -c safe.directory="$REPO_ROOT" rev-parse HEAD)"
CURRENT_TREE="$(git -c safe.directory="$REPO_ROOT" rev-parse HEAD^{tree})"
if [[ $CURRENT_HEAD != "$INITIAL_HEAD" || $CURRENT_TREE != "$INITIAL_TREE" ]] ||
    ! repo_is_clean; then
    echo "ERROR: acceptance changed HEAD or the source checkout" >&2
    exit 1
fi

ACCEPTANCE_STAGE=complete
