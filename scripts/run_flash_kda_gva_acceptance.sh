#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

# One-command gfx950 GVA acceptance. The coverage remains in the canonical
# pytest and benchmark files; this script only provides a reproducible runner.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
WARMUP="${KDA_ACCEPTANCE_WARMUP:-10}"
REPEAT="${KDA_ACCEPTANCE_REPEAT:-50}"
SEED="${KDA_ACCEPTANCE_SEED:-42}"
MIN_SPEEDUP="${KDA_ACCEPTANCE_MIN_SPEEDUP:-1.0}"
MIN_GEOMEAN="${KDA_ACCEPTANCE_MIN_GEOMEAN:-1.05}"
MIN_WIN_FRACTION="${KDA_ACCEPTANCE_MIN_WIN_FRACTION:-0.75}"

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    cat <<'EOF'
Usage: scripts/run_flash_kda_gva_acceptance.sh [OUTPUT_DIR]

Run the FlashKDA Python API tests, native GPU correctness tests, and the
gfx950/256-CU HIP-versus-Triton performance matrix. Select one GPU with
ROCR_VISIBLE_DEVICES (or HIP_VISIBLE_DEVICES) before invoking the script.

The output directory must be outside the source checkout. The default is a
new /tmp/flash-kda-gva-* directory.
Override benchmark settings with KDA_ACCEPTANCE_WARMUP,
KDA_ACCEPTANCE_REPEAT, KDA_ACCEPTANCE_SEED, KDA_ACCEPTANCE_MIN_SPEEDUP,
KDA_ACCEPTANCE_MIN_GEOMEAN, and KDA_ACCEPTANCE_MIN_WIN_FRACTION.
EOF
    exit 0
fi
if (($# > 1)); then
    echo "Usage: scripts/run_flash_kda_gva_acceptance.sh [OUTPUT_DIR]" >&2
    exit 2
fi

cd "$REPO_ROOT"
if [[ -n $(git -c safe.directory="$REPO_ROOT" status --porcelain) ]]; then
    echo "ERROR: acceptance requires a clean checkout" >&2
    exit 2
fi

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
    OUTPUT_DIR="$(mktemp -d "$OUTPUT_PARENT/flash-kda-gva-XXXXXXXX")"
fi

ACCEPTANCE_STAGE=setup
acceptance_exit() {
    local exit_code=$?
    local status=failed
    if ((exit_code == 0)); then
        status=passed
    fi
    {
        printf 'status=%s\n' "$status"
        printf 'exit_code=%s\n' "$exit_code"
        printf 'stage=%s\n' "$ACCEPTANCE_STAGE"
        printf 'git_head=%s\n' "$(git rev-parse HEAD)"
    } >"$OUTPUT_DIR/status.txt"
    if ((exit_code == 0)); then
        echo "FlashKDA GVA acceptance passed: $OUTPUT_DIR"
    else
        echo "FlashKDA GVA acceptance failed during $ACCEPTANCE_STAGE: $OUTPUT_DIR" >&2
    fi
}
trap acceptance_exit EXIT
git rev-parse HEAD >"$OUTPUT_DIR/git-head.txt"
echo "FlashKDA GVA acceptance artifacts: $OUTPUT_DIR"

for name in $(compgen -e FLASH_KDA_ || true); do
    unset "$name"
done
for name in $(compgen -e CHUNK_DELTA_ATTN_ || true); do
    unset "$name"
done
for name in $(compgen -e FLA_ || true); do
    unset "$name"
done
unset AITER_KDA_BACKEND AITER_TRITON_ONLY AITER_REBUILD
unset AITER_META_DIR CK_DIR HIP_KITTENS_DIR
export AITER_AOT_IMPORT=1
export AITER_JIT_DIR="$OUTPUT_DIR/jit"
export GPU_ARCHS=gfx950
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0="$REPO_ROOT"
mkdir -p "$AITER_JIT_DIR"

ACCEPTANCE_STAGE=environment
"$PYTHON_BIN" - <<'PY' | tee "$OUTPUT_DIR/environment.log"
import torch

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
PY

run_logged() {
    local label=$1
    shift
    ACCEPTANCE_STAGE=$label
    echo "===== $label ====="
    "$@" 2>&1 | tee "$OUTPUT_DIR/$label.log"
}

run_logged correctness env AITER_REBUILD=1 "$PYTHON_BIN" -m pytest -q \
    op_tests/triton_tests/chunk_delta_attn/test_flash_kda_native_python_api.py \
    op_tests/triton_tests/chunk_delta_attn/test_flash_kda_native.py

BENCHMARK=(
    "$PYTHON_BIN" -u op_tests/op_benchmarks/triton/bench_flash_kda_native.py
    --backend native --backend triton --execution graph --public-k3
    --warmup "$WARMUP" --repeat "$REPEAT" --seed "$SEED" --tolerance 0.04
    --require-arch gfx950 --require-compute-units 256
    --min-speedup "$MIN_SPEEDUP"
    --min-geomean-speedup "$MIN_GEOMEAN"
    --min-paired-win-fraction "$MIN_WIN_FRACTION"
)

run_benchmark() {
    local label=$1
    shift
    run_logged "$label" "${BENCHMARK[@]}" "$@" \
        --csv "$OUTPUT_DIR/$label.csv" \
        --raw-csv "$OUTPUT_DIR/$label.raw.csv" \
        --json "$OUTPUT_DIR/$label.json"
}

# K3 TP8 GVA, including single 128..16K, batched/ragged, fresh/resume,
# actual mixed decode+prefill batches, and the direct/hybrid 1024/1025 edge.
run_benchmark gva-hq2-hv4-core \
    --suite core --heads 2 --value-heads 4
run_benchmark gva-hq2-hv8-stress \
    --case single-2k --case ragged-16k --case resume-4x4k \
    --heads 2 --value-heads 8
run_benchmark gva-hq2-hv4-mixed-production \
    --suite mixed-production --heads 2 --value-heads 4
run_benchmark gva-hq2-hv4-mixed-boundary \
    --suite mixed-boundary --heads 2 --value-heads 4

ACCEPTANCE_STAGE=artifact-verification
sha256sum "$AITER_JIT_DIR/module_flash_kda_hip.so" \
    >"$OUTPUT_DIR/module.sha256"
if [[ -n $(git -c safe.directory="$REPO_ROOT" status --porcelain) ]]; then
    echo "ERROR: acceptance modified the source checkout" >&2
    git -c safe.directory="$REPO_ROOT" status --short >&2
    exit 1
fi
ACCEPTANCE_STAGE=complete
