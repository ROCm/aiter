#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

# One-command gfx950 public-K3 performance acceptance. The benchmark logic and
# acceptance criteria remain in the canonical Python entry point; this script
# provides a clean, reproducible runner and preserves the resulting artifacts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="$(command -v -- "${PYTHON:-python3}")"
readonly WARMUP=20
readonly REPEAT=120
readonly BOOTSTRAP_RESAMPLES=10000

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    cat <<'EOF'
Usage: scripts/run_flash_kda_public_k3_perf_acceptance.sh [OUTPUT_DIR]

Run the formal public-K3 performance acceptance on exactly one visible
gfx950/256-CU GPU. Select the GPU with ROCR_VISIBLE_DEVICES (or
HIP_VISIBLE_DEVICES) before invoking the script.

The output directory must be outside the source checkout. The default is a
new /tmp/flash-kda-public-k3-* directory. The formal seed set is 42, 43, and
44. Timing and statistics are pinned to warmup 20, repeat 120, and 10,000
bootstrap resamples so this command always runs the published configuration.
EOF
    exit 0
fi
if (($# > 1)); then
    echo "Usage: scripts/run_flash_kda_public_k3_perf_acceptance.sh [OUTPUT_DIR]" >&2
    exit 2
fi

cd "$REPO_ROOT"
repo_is_clean() {
    git -c safe.directory="$REPO_ROOT" diff --quiet -- &&
        git -c safe.directory="$REPO_ROOT" diff --cached --quiet -- &&
        [[ -z $(git -c safe.directory="$REPO_ROOT" \
            ls-files --others --exclude-standard) ]] &&
        ! git -c safe.directory="$REPO_ROOT" submodule status --recursive |
            grep -Eq '^[+U]' &&
        git -c safe.directory="$REPO_ROOT" submodule foreach --recursive \
            --quiet 'test -z "$(git status --porcelain --untracked-files=all)"'
}

if ! repo_is_clean; then
    echo "ERROR: acceptance requires a clean checkout" >&2
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
    OUTPUT_DIR="$(mktemp -d "$OUTPUT_PARENT/flash-kda-public-k3-XXXXXXXX")"
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
    {
        printf 'initial_head=%s\n' "$INITIAL_HEAD"
        printf 'final_head=%s\n' "$final_head"
        printf 'initial_tree=%s\n' "$INITIAL_TREE"
        printf 'final_tree=%s\n' "$final_tree"
        printf 'clean_and_unchanged=%s\n' "$repository_ok"
    } >"$OUTPUT_DIR/repository-after.txt"

    local status=failed
    if ((exit_code == 0)); then
        status=passed
    fi
    {
        printf 'status=%s\n' "$status"
        printf 'exit_code=%s\n' "$exit_code"
        printf 'stage=%s\n' "$ACCEPTANCE_STAGE"
        printf 'git_head=%s\n' "$(git rev-parse HEAD)"
        printf 'initial_git_head=%s\n' "$INITIAL_HEAD"
        printf 'initial_git_tree=%s\n' "$INITIAL_TREE"
        printf 'warmup=%s\n' "$WARMUP"
        printf 'repeat=%s\n' "$REPEAT"
        printf 'seeds=42,43,44\n'
        printf 'bootstrap_resamples=%s\n' "$BOOTSTRAP_RESAMPLES"
        printf 'python=%s\n' "$PYTHON_BIN"
        printf 'benchmark_exit=%s\n' "${BENCHMARK_EXIT:-not-run}"
    } >"$OUTPUT_DIR/status.txt"

    local candidate
    local -a artifacts=()
    for candidate in \
        status.txt repository-after.txt git-head.txt git-tree.txt \
        static-self-test.json static-self-test.log plan.json plan.log \
        environment.log benchmark.log benchmark-status.txt result.json \
        partial-results.jsonl jit/module_aiter_core.so \
        jit/module_flash_kda_hip.so; do
        [[ -s $OUTPUT_DIR/$candidate ]] && artifacts+=("$candidate")
    done
    if ((${#artifacts[@]} > 0)); then
        (
            cd "$OUTPUT_DIR" || exit 1
            sha256sum -- "${artifacts[@]}"
        ) >"$OUTPUT_DIR/artifacts.sha256"
    fi

    if ((repository_ok == 0)); then
        git -c safe.directory="$REPO_ROOT" status --short \
            --untracked-files=all >&2
    fi
    if ((exit_code == 0)); then
        echo "FlashKDA public-K3 performance acceptance passed: $OUTPUT_DIR"
    else
        printf \
            'FlashKDA public-K3 performance acceptance failed during %s: %s\n' \
            "$ACCEPTANCE_STAGE" "$OUTPUT_DIR" >&2
    fi
    exit "$exit_code"
}
trap acceptance_exit EXIT
printf '%s\n' "$INITIAL_HEAD" >"$OUTPUT_DIR/git-head.txt"
printf '%s\n' "$INITIAL_TREE" >"$OUTPUT_DIR/git-tree.txt"
echo "FlashKDA public-K3 performance acceptance artifacts: $OUTPUT_DIR"

for prefix in AITER_ KDA_ FLASH_KDA_ CHUNK_DELTA_ATTN_ FLA_; do
    for name in $(compgen -e "$prefix" || true); do
        unset "$name"
    done
done
unset AITER_KDA_BACKEND AITER_TRITON_ONLY AITER_REBUILD
unset AITER_META_DIR CK_DIR HIP_KITTENS_DIR
export AITER_AOT_IMPORT=1
export AITER_JIT_DIR="$OUTPUT_DIR/jit"
export GPU_ARCHS=gfx950
export MAX_JOBS=1
export PYTHONDONTWRITEBYTECODE=1
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
mkdir -p "$AITER_JIT_DIR"

BENCHMARK_ENTRY=op_tests/op_benchmarks/triton/bench_flash_kda_public_k3.py

ACCEPTANCE_STAGE=static-self-test
"$PYTHON_BIN" -u "$BENCHMARK_ENTRY" \
    --static-self-test \
    --output "$OUTPUT_DIR/static-self-test.json" \
    2>&1 | tee "$OUTPUT_DIR/static-self-test.log"

ACCEPTANCE_STAGE=plan
"$PYTHON_BIN" -u "$BENCHMARK_ENTRY" \
    --print-plan \
    --seed 42 --seed 43 --seed 44 \
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
arch = getattr(properties, "gcnArchName", "unknown").split(":", 1)[0]
if arch != "gfx950" or properties.multi_processor_count != 256:
    raise SystemExit(
        "acceptance requires the 256-CU gfx950 reference target; "
        f"got {arch}/{properties.multi_processor_count} CU"
    )
print(
    f"GPU: {torch.cuda.get_device_name(0)} "
    f"({arch}, {properties.multi_processor_count} CU); "
    f"PyTorch {torch.__version__}; ROCm {torch.version.hip}"
)
print(
    f"Python: {sys.executable}; {platform.python_version()}; "
    f"sha256={sha256(sys.executable)}"
)
print(f"Triton: {triton.__version__}; {triton.__file__}")
hipcc = shutil.which("hipcc")
print(f"hipcc: {hipcc}")
if hipcc is not None:
    version = subprocess.run(
        [hipcc, "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    print(version)
print(f"AITER_JIT_DIR: {os.environ['AITER_JIT_DIR']}")
print(f"AITER_AOT_IMPORT: {os.environ.get('AITER_AOT_IMPORT')}")
print(f"TRITON_CACHE_DIR: {os.environ['TRITON_CACHE_DIR']}")
print(f"MAX_JOBS: {os.environ['MAX_JOBS']}")
PY

ACCEPTANCE_STAGE=benchmark
set +e
"$PYTHON_BIN" -u "$BENCHMARK_ENTRY" \
    --seed 42 --seed 43 --seed 44 \
    --warmup "$WARMUP" \
    --repeat "$REPEAT" \
    --bootstrap-resamples "$BOOTSTRAP_RESAMPLES" \
    --output "$OUTPUT_DIR/result.json" \
    2>&1 | tee "$OUTPUT_DIR/benchmark.log"
BENCHMARK_EXIT=${PIPESTATUS[0]}
set -e
printf 'benchmark_exit=%s\n' "$BENCHMARK_EXIT" \
    >"$OUTPUT_DIR/benchmark-status.txt"

ACCEPTANCE_STAGE=artifact-finalization
CURRENT_HEAD="$(git -c safe.directory="$REPO_ROOT" rev-parse HEAD)"
CURRENT_TREE="$(git -c safe.directory="$REPO_ROOT" rev-parse HEAD^{tree})"
REPOSITORY_OK=1
if [[ $CURRENT_HEAD != "$INITIAL_HEAD" || $CURRENT_TREE != "$INITIAL_TREE" ]] ||
    ! repo_is_clean; then
    REPOSITORY_OK=0
fi
if ((BENCHMARK_EXIT == 0)); then
    for artifact in result.json partial-results.jsonl \
        jit/module_flash_kda_hip.so; do
        [[ -s $OUTPUT_DIR/$artifact ]] || {
            echo "ERROR: successful benchmark did not produce $artifact" >&2
            exit 1
        }
    done
fi

if ((REPOSITORY_OK == 0)); then
    echo "ERROR: acceptance changed HEAD or the source checkout" >&2
    exit 1
fi
if ((BENCHMARK_EXIT != 0)); then
    ACCEPTANCE_STAGE=benchmark-failed
    exit "$BENCHMARK_EXIT"
fi
ACCEPTANCE_STAGE=complete
