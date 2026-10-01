#!/usr/bin/env bash
set -Eeuo pipefail

# Reproduce same-machine comparisons for the gfx1250 E96/T16384 FlyDSL MoE
# kernels. Run this script directly inside the existing ROCm container.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
EXPECTED_BRANCH="${EXPECTED_BRANCH:-hyg/moe_a4w4_pr}"
E2E_ITERS="${AITER_REPRO_E2E_ITERS:-${E2E_ITERS:-20}}"
E2E_ROUNDS="${AITER_REPRO_E2E_ROUNDS:-${ROUNDS:-2}}"
RUN_VERIFY="${AITER_REPRO_RUN_VERIFY:-${RUN_VERIFY:-1}}"
RUN_ATT="${AITER_REPRO_RUN_ATT:-${RUN_ATT:-0}}"
ATT_VALIDATE_ONLY="${AITER_ATT_VALIDATE_ONLY:-0}"
ATT_E2E_ITERS="${AITER_ATT_E2E_ITERS:-2}"
ATT_KERNEL_ITERATION_RANGE="${AITER_ATT_KERNEL_ITERATION_RANGE:-[8]}"
ATT_SIMD_LIST="${AITER_ATT_SIMD_LIST:-0,1,2,3}"
ATT_TARGET_CU="${AITER_ATT_TARGET_CU:-1}"
ATT_SHADER_ENGINE_MASK="${AITER_ATT_SHADER_ENGINE_MASK:-0x1}"
ATT_BUFFER_SIZE="${AITER_ATT_BUFFER_SIZE:-0x10000000}"
ATT_TIMEOUT_SECONDS="${AITER_ATT_TIMEOUT_SECONDS:-300}"
ATT_DECODER_DIR="${AITER_ATT_LIBRARY_PATH:-/data/yanguahe/code/wk_sp1/decoder_new}"
ROCPROF_ENV="${AITER_ROCPROF_ENV:-/data/yanguahe/code/wk_sp1/rocprof_env.sh}"
ROCPROF_BIN_DIR="${AITER_ROCPROF_BIN_DIR:-/data/yanguahe/code/wk_sp1/rocprof-install/bin}"

MODE=e2e-const0
TARGET=gemm1
GEMM1_FOR_GEMM2="${AITER_REPRO_GEMM1_CASE:-sync_mg4_fc28_apre_exactopt}"
GEMM2_APRE_PRODUCER="${AITER_REPRO_GEMM2_APRE_PRODUCER:-rowgroup}"
GEMM2_APRE_RPW="${AITER_REPRO_GEMM2_APRE_RPW:-2}"
GEMM2_APRE_PREFETCH="${AITER_REPRO_GEMM2_APRE_PREFETCH:-2}"
mode_seen=0

usage() {
    cat <<'EOF'
usage: bash my_code/reproduce_compare.sh [MODE] [--gemm2]

Modes:
  e2e-const0   Full MoE const0 correctness and profiler benchmark (default)
  e2e-random   Full MoE random correctness and profiler benchmark
  e2e-both     Run random followed by const0
  att          Preflight and capture/decode ATT for every selected case
  att-validate Validate the ATT workload without tracing

Target selection:
  --gemm2      Benchmark the registered GEMM2 baseline and optimization cases.
               Select GEMM1 with AITER_REPRO_GEMM1_CASE; the default is
               sync_mg4_fc28_apre_exactopt.

Environment aliases:
  ROUNDS=3 RUN_VERIFY=1 RUN_ATT=0
  CASE_LIST=baseline_93665e,sync_mg4_fc28_apre_exactopt   GEMM1 cases
  CASE_LIST=baseline,apre,apre_wpt2                      GEMM2 cases
  AITER_REPRO_GEMM1_CASE=sync_mg4_fc28   Select GEMM1 in --gemm2 mode
  AITER_REPRO_GEMM2_APRE_PRODUCER=rowgroup|three_kernel
  AITER_REPRO_GEMM2_APRE_RPW=1|2|4|8     Rowgroup rows per wave
  AITER_REPRO_GEMM2_APRE_PREFETCH=1|2|4|8 Rowgroup load prefetch depth
  AITER_REPRO_EXPERT_BALANCE=false   Run non-balanced routing correctness
  AITER_ATT_SIMD_LIST=0,1,2,3
  AITER_ATT_KERNEL_ITERATION_RANGE='[8]'
EOF
}

for arg in "$@"; do
    case "$arg" in
        -h|--help|help)
            usage
            exit 0
            ;;
        --gemm2)
            TARGET=gemm2
            ;;
        e2e-const0|e2e-random|e2e-both|att|att-validate)
            if [[ "$mode_seen" == 1 ]]; then
                echo "Only one MODE may be specified." >&2
                usage >&2
                exit 2
            fi
            MODE="$arg"
            mode_seen=1
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            usage >&2
            exit 2
            ;;
    esac
done
if [[ "$MODE" == att-validate ]]; then
    ATT_VALIDATE_ONLY=1
fi

require_positive_integer() {
    local name="$1"
    local value="$2"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "$name must be a positive integer, got '$value'" >&2
        exit 2
    fi
}

require_nonnegative_integer() {
    local name="$1"
    local value="$2"
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        echo "$name must be a non-negative integer, got '$value'" >&2
        exit 2
    fi
}

require_flag() {
    local name="$1"
    local value="$2"
    if [[ "$value" != 0 && "$value" != 1 ]]; then
        echo "$name must be 0 or 1, got '$value'" >&2
        exit 2
    fi
}

require_positive_integer AITER_REPRO_E2E_ITERS "$E2E_ITERS"
require_positive_integer AITER_REPRO_E2E_ROUNDS "$E2E_ROUNDS"
require_flag AITER_REPRO_RUN_VERIFY "$RUN_VERIFY"
require_flag AITER_REPRO_RUN_ATT "$RUN_ATT"
require_flag AITER_ATT_VALIDATE_ONLY "$ATT_VALIDATE_ONLY"
require_positive_integer AITER_ATT_E2E_ITERS "$ATT_E2E_ITERS"
require_nonnegative_integer AITER_ATT_TARGET_CU "$ATT_TARGET_CU"
require_positive_integer AITER_ATT_TIMEOUT_SECONDS "$ATT_TIMEOUT_SECONDS"
if [[ ! "$ATT_KERNEL_ITERATION_RANGE" =~ ^\[[0-9]+([:-][0-9]+)?\]$ ]]; then
    echo "AITER_ATT_KERNEL_ITERATION_RANGE must look like [8], [8:8], or [8-8], got '$ATT_KERNEL_ITERATION_RANGE'" >&2
    exit 2
fi
if [[ ! "$ATT_SHADER_ENGINE_MASK" =~ ^0x[0-9A-Fa-f]+$ ]]; then
    echo "AITER_ATT_SHADER_ENGINE_MASK must be hexadecimal, got '$ATT_SHADER_ENGINE_MASK'" >&2
    exit 2
fi
if [[ ! "$ATT_BUFFER_SIZE" =~ ^0x[0-9A-Fa-f]+$ ]]; then
    echo "AITER_ATT_BUFFER_SIZE must be hexadecimal, got '$ATT_BUFFER_SIZE'" >&2
    exit 2
fi
IFS=',' read -r -a ATT_SIMDS <<<"$ATT_SIMD_LIST"
declare -A seen_att_simd=()
for att_simd in "${ATT_SIMDS[@]}"; do
    if [[ ! "$att_simd" =~ ^[0-3]$ || -n "${seen_att_simd[$att_simd]:-}" ]]; then
        echo "AITER_ATT_SIMD_LIST must contain unique comma-separated SIMD IDs 0..3, got '$ATT_SIMD_LIST'" >&2
        exit 2
    fi
    seen_att_simd[$att_simd]=1
done

if [[ ! -f /.dockerenv ]]; then
    echo "This script must be run inside the ROCm container." >&2
    exit 2
fi

cd "$REPO_ROOT"

BASELINE_KERNEL_REL="aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm_93665e.py"
BASELINE_KERNEL_SHA256="c8b02647cacf457b772e5e38f9d08999e995e6f221917a45846e32a22c8927fe"
BASELINE_KERNEL_PATH="$REPO_ROOT/$BASELINE_KERNEL_REL"
if [[ "$TARGET" == gemm1 ]]; then
    if [[ ! -f "$BASELINE_KERNEL_PATH" ]]; then
        echo "Missing baseline kernel source: $BASELINE_KERNEL_PATH" >&2
        exit 2
    fi
    actual_baseline_sha256="$(sha256sum "$BASELINE_KERNEL_PATH" | awk '{print $1}')"
    if [[ "$actual_baseline_sha256" != "$BASELINE_KERNEL_SHA256" ]]; then
        echo "Baseline kernel source checksum mismatch: $BASELINE_KERNEL_PATH" >&2
        echo "expected $BASELINE_KERNEL_SHA256, got $actual_baseline_sha256" >&2
        exit 2
    fi
fi

read_git_head_without_git() {
    local head ref ref_file
    if [[ ! -f "$REPO_ROOT/.git/HEAD" ]]; then
        printf '%s\n' unknown
        return
    fi
    IFS= read -r head <"$REPO_ROOT/.git/HEAD"
    if [[ "$head" != "ref: "* ]]; then
        printf '%s\n' "$head"
        return
    fi
    ref="${head#ref: }"
    ref_file="$REPO_ROOT/.git/$ref"
    if [[ -f "$ref_file" ]]; then
        IFS= read -r head <"$ref_file"
        printf '%s\n' "$head"
        return
    fi
    awk -v ref="$ref" '$2 == ref { print $1; found = 1; exit } END { if (!found) print "unknown" }' \
        "$REPO_ROOT/.git/packed-refs" 2>/dev/null || printf '%s\n' unknown
}

read_git_branch_without_git() {
    local head
    if [[ ! -f "$REPO_ROOT/.git/HEAD" ]]; then
        printf '%s\n' unknown
        return
    fi
    IFS= read -r head <"$REPO_ROOT/.git/HEAD"
    if [[ "$head" == "ref: refs/heads/"* ]]; then
        printf '%s\n' "${head#ref: refs/heads/}"
    else
        printf '%s\n' detached
    fi
}

branch="$(read_git_branch_without_git)"
head="$(read_git_head_without_git)"
if [[ "$branch" != "$EXPECTED_BRANCH" ]]; then
    echo "Refusing to run on branch '$branch'; expected '$EXPECTED_BRANCH'." >&2
    exit 2
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
host="$(hostname -s)"
out_rel="my_code/gemm1_cycle_105pct_20260909/runs/${host}_${timestamp}_${TARGET}_${MODE}"
out_dir="$REPO_ROOT/$out_rel"
mkdir -p "$out_dir"
cleanup_output_permissions() {
    chmod -R a+rwX "$out_dir" 2>/dev/null || true
}
trap cleanup_output_permissions EXIT

e2e_runner="$out_dir/run_flydsl_e2e_hash128.py"
cat >"$e2e_runner" <<'PY'
#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path


REPO = Path(os.environ["AITER_REPRO_REPO_ROOT"]).resolve()
TEST = REPO / "my_code" / "test_flydsl_grouped_gemm_gfx1250.py"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# The script is allowed to run only on the mapped gfx1250 systems. Pin the
# FlyDSL build-time query so correctness runs do not spawn a second rocminfo
# process behind an unrelated system-wide rocminfo/GPU probe.
from flydsl.runtime import device as flydsl_device

flydsl_device.get_rocm_arch = lambda: "gfx1250"


def replace_exact(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"expected exactly one test-source match, got {count}: {old!r}"
        )
    return source.replace(old, new, 1)


def patched_test_source() -> str:
    source = TEST.read_text(encoding="utf-8")
    # This runner is intentionally restricted to the mapped gfx1250 machines.
    # Avoid a redundant rocminfo subprocess, which can block behind unrelated
    # system probes even when correctness-only execution is otherwise allowed.
    source = replace_exact(source, "    if not is_gfx1250():", "    if False:")
    source = replace_exact(source, "        default=is_gfx1250(),", "        default=True,")
    return source


if os.environ.get("AITER_REPRO_USE_BASELINE_93665E") == "1":
    from aiter.ops.flydsl.grouped_gemm_mxfp4 import use_gemm1_baseline_93665e

    use_gemm1_baseline_93665e()

sys.argv = [str(TEST), *sys.argv[1:]]
namespace = {
    "__name__": "__main__",
    "__file__": str(TEST),
    "__package__": None,
    "__cached__": None,
}
exec(compile(patched_test_source(), str(TEST), "exec"), namespace)
PY
chmod a+rx "$e2e_runner"

declare -a CLEAR_ENV=(
    -u AITER_MOE_GEMM1_LAUNCH_BACKEND
    -u AITER_FLYDSL_GEMM1_MMA_GROUP
    -u AITER_FLYDSL_GEMM1_FENCE_COVER_MMA
    -u AITER_FLYDSL_GEMM1_SILU_HARD
    -u AITER_FLYDSL_GEMM1_SILU_RELU
    -u AITER_FLYDSL_GEMM1_A_PRESHUFFLE
    -u AITER_FLYDSL_GEMM1_WAVES_PER_TENSOR_TDM
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

declare -a COMMON_ENV=(
    AITER_USE_GROUPED_GEMM=1
    AITER_GROUPED_DEBUG=0
    ENABLE_CK=0
    "FLYDSL_DUMP_IR=${FLYDSL_DUMP_IR:-0}"
    AITER_LOG_MORE=1
    AITER_FORCE_GFX1250=1
    GPU_ARCHS=gfx1250
    CU_NUM=256
    "AITER_MOE_EXPERT_BALANCE=${AITER_REPRO_EXPERT_BALANCE:-true}"
    AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1
    AITER_FLYDSL_GEMM1_A_PRESHUFFLE=0
    "AITER_REPRO_REPO_ROOT=$REPO_ROOT"
)

declare -a TEST_SHAPE=(
    --data-format a4w4
    --experts 96
    --tokens 16384
    --topk 6
    --model-dim 7168
    --inter-dim 3072
    --act silu
    --no-bias
    --no-check-aot-cache
)

declare -a GEMM1_CASES=(
    baseline_93665e
    sync_mg4_fc8
    sync_mg2_fc12
    sync_mg4_fc28
    sync_mg4_fc28_apre
    sync_mg4_fc28_apre_exactopt
    sync_mg4_fc28_hard
    sync_mg4_fc28_relu
)

if [[ "$TARGET" == gemm2 ]]; then
    declare -a CASES=(
        baseline
        apre
        apre_wpt2
        apre_wpt2_mg4_fc28
        apre_wpt2_mg4_fc28_ostore2p
        apre_wpt2_mg4_fc28_ostore2p_ow2
    )
else
    declare -a CASES=("${GEMM1_CASES[@]}")
fi
if [[ -n "${CASE_LIST:-}" ]]; then
    IFS=',' read -r -a CASES <<<"$CASE_LIST"
fi
if [[ "$TARGET" == gemm2 ]]; then
    for name in "${CASES[@]}"; do
        case "$name" in
            baseline|apre|apre_wpt2|apre_wpt2_mg4_fc28|\
                apre_wpt2_mg4_fc28_ostore2p|\
                apre_wpt2_mg4_fc28_ostore2p_ow2)
                ;;
            *)
                echo "--gemm2 supports only registered GEMM2 cases (got '$name')." >&2
                exit 2
                ;;
        esac
    done
fi

declare -a CASE_ENV=()
CASE_BASELINE_FLAG=0
gemm1_case_env() {
    CASE_BASELINE_FLAG=0
    case "$1" in
        baseline_93665e)
            CASE_BASELINE_FLAG=1
            CASE_ENV=()
            ;;
        sync_mg4_fc8)
            CASE_ENV=(
                AITER_FLYDSL_GEMM1_MMA_GROUP=4
                AITER_FLYDSL_GEMM1_FENCE_COVER_MMA=8
                AITER_FLYDSL_GEMM1_SILU_HARD=0
                AITER_FLYDSL_GEMM1_SILU_RELU=0
            )
            ;;
        sync_mg2_fc12)
            CASE_ENV=(
                AITER_FLYDSL_GEMM1_MMA_GROUP=2
                AITER_FLYDSL_GEMM1_FENCE_COVER_MMA=12
                AITER_FLYDSL_GEMM1_SILU_HARD=0
                AITER_FLYDSL_GEMM1_SILU_RELU=0
            )
            ;;
        sync_mg4_fc28)
            CASE_ENV=(
                AITER_FLYDSL_GEMM1_MMA_GROUP=4
                AITER_FLYDSL_GEMM1_FENCE_COVER_MMA=28
                AITER_FLYDSL_GEMM1_SILU_HARD=0
                AITER_FLYDSL_GEMM1_SILU_RELU=0
            )
            ;;
        sync_mg4_fc28_apre)
            CASE_ENV=(
                AITER_FLYDSL_GEMM1_MMA_GROUP=4
                AITER_FLYDSL_GEMM1_FENCE_COVER_MMA=28
                AITER_FLYDSL_GEMM1_SILU_HARD=0
                AITER_FLYDSL_GEMM1_SILU_RELU=0
                AITER_FLYDSL_GEMM1_A_PRESHUFFLE=1
            )
            ;;
        sync_mg4_fc28_apre_exactopt)
            CASE_ENV=(
                AITER_FLYDSL_GEMM1_MMA_GROUP=4
                AITER_FLYDSL_GEMM1_FENCE_COVER_MMA=28
                AITER_FLYDSL_GEMM1_SILU_HARD=0
                AITER_FLYDSL_GEMM1_SILU_RELU=0
                AITER_FLYDSL_GEMM1_A_PRESHUFFLE=1
                AITER_FLYDSL_GEMM1_WAVES_PER_TENSOR_TDM=2
                AITER_FLYDSL_GEMM1_DISABLE_XDL_ARB_STALL=0
                AITER_FLYDSL_GEMM1_WMMA_REUSE=1
                AITER_FLYDSL_GEMM1_OVERLAP_OUTPUT_STORE=1
            )
            ;;
        sync_mg4_fc28_hard)
            CASE_ENV=(
                AITER_FLYDSL_GEMM1_MMA_GROUP=4
                AITER_FLYDSL_GEMM1_FENCE_COVER_MMA=28
                AITER_FLYDSL_GEMM1_SILU_HARD=1
                AITER_FLYDSL_GEMM1_SILU_RELU=0
            )
            ;;
        sync_mg4_fc28_relu)
            CASE_ENV=(
                AITER_FLYDSL_GEMM1_MMA_GROUP=4
                AITER_FLYDSL_GEMM1_FENCE_COVER_MMA=28
                AITER_FLYDSL_GEMM1_SILU_HARD=0
                AITER_FLYDSL_GEMM1_SILU_RELU=1
            )
            ;;
        *)
            echo "Unknown case: $1" >&2
            return 2
            ;;
    esac
}

case_env() {
    local name="$1"
    if [[ "$TARGET" == gemm1 ]]; then
        gemm1_case_env "$name"
        return
    fi

    gemm1_case_env "$GEMM1_FOR_GEMM2"
    case "$name" in
        baseline)
            CASE_ENV+=(
                AITER_FLYDSL_GEMM2_A_PRESHUFFLE=0
                AITER_FLYDSL_GEMM2_WAVES_PER_TENSOR_TDM=1
                AITER_FLYDSL_GEMM2_SCHEDULE_HINTS=0
            )
            ;;
        apre)
            CASE_ENV+=(
                AITER_FLYDSL_GEMM2_A_PRESHUFFLE=1
                AITER_FLYDSL_GEMM2_WAVES_PER_TENSOR_TDM=1
                AITER_FLYDSL_GEMM2_SCHEDULE_HINTS=0
                "AITER_FLYDSL_GEMM2_A_PRESHUFFLE_PRODUCER=$GEMM2_APRE_PRODUCER"
                "AITER_FLYDSL_GEMM2_A_PRESHUFFLE_RPW=$GEMM2_APRE_RPW"
                "AITER_FLYDSL_GEMM2_A_PRESHUFFLE_PREFETCH=$GEMM2_APRE_PREFETCH"
            )
            ;;
        apre_wpt2)
            CASE_ENV+=(
                AITER_FLYDSL_GEMM2_A_PRESHUFFLE=1
                AITER_FLYDSL_GEMM2_WAVES_PER_TENSOR_TDM=2
                AITER_FLYDSL_GEMM2_SCHEDULE_HINTS=0
                "AITER_FLYDSL_GEMM2_A_PRESHUFFLE_PRODUCER=$GEMM2_APRE_PRODUCER"
                "AITER_FLYDSL_GEMM2_A_PRESHUFFLE_RPW=$GEMM2_APRE_RPW"
                "AITER_FLYDSL_GEMM2_A_PRESHUFFLE_PREFETCH=$GEMM2_APRE_PREFETCH"
            )
            ;;
        apre_wpt2_mg4_fc28)
            CASE_ENV+=(
                AITER_FLYDSL_GEMM2_A_PRESHUFFLE=1
                AITER_FLYDSL_GEMM2_WAVES_PER_TENSOR_TDM=2
                AITER_FLYDSL_GEMM2_SCHEDULE_HINTS=1
                AITER_FLYDSL_GEMM2_MMA_GROUP=4
                AITER_FLYDSL_GEMM2_FENCE_COVER_MMA=28
                "AITER_FLYDSL_GEMM2_A_PRESHUFFLE_PRODUCER=$GEMM2_APRE_PRODUCER"
                "AITER_FLYDSL_GEMM2_A_PRESHUFFLE_RPW=$GEMM2_APRE_RPW"
                "AITER_FLYDSL_GEMM2_A_PRESHUFFLE_PREFETCH=$GEMM2_APRE_PREFETCH"
            )
            ;;
        apre_wpt2_mg4_fc28_ostore2p|\
            apre_wpt2_mg4_fc28_ostore2p_ow2)
            local output_wave_split=0
            if [[ "$name" != apre_wpt2_mg4_fc28_ostore2p ]]; then
                output_wave_split=1
            fi
            CASE_ENV+=(
                AITER_FLYDSL_GEMM2_A_PRESHUFFLE=1
                AITER_FLYDSL_GEMM2_WAVES_PER_TENSOR_TDM=2
                AITER_FLYDSL_GEMM2_SCHEDULE_HINTS=1
                AITER_FLYDSL_GEMM2_MMA_GROUP=4
                AITER_FLYDSL_GEMM2_FENCE_COVER_MMA=28
                AITER_FLYDSL_GEMM2_OVERLAP_OUTPUT_STORE=1
                "AITER_FLYDSL_GEMM2_OUTPUT_SPLIT_WM=${AITER_REPRO_GEMM2_OUTPUT_SPLIT_WM:-3}"
                "AITER_FLYDSL_GEMM2_OUTPUT_WAVE_SPLIT=$output_wave_split"
                "AITER_FLYDSL_GEMM2_A_PRESHUFFLE_PRODUCER=$GEMM2_APRE_PRODUCER"
                "AITER_FLYDSL_GEMM2_A_PRESHUFFLE_RPW=$GEMM2_APRE_RPW"
                "AITER_FLYDSL_GEMM2_A_PRESHUFFLE_PREFETCH=$GEMM2_APRE_PREFETCH"
            )
            ;;
        *)
            echo "Unknown GEMM2 case: $name" >&2
            return 2
            ;;
    esac
}

if [[ "$TARGET" == gemm2 ]]; then
    if ! gemm1_case_env "$GEMM1_FOR_GEMM2"; then
        echo "Unknown AITER_REPRO_GEMM1_CASE: $GEMM1_FOR_GEMM2" >&2
        echo "Supported GEMM1 cases: ${GEMM1_CASES[*]}" >&2
        exit 2
    fi
    if [[ "$GEMM2_APRE_PRODUCER" != rowgroup \
          && "$GEMM2_APRE_PRODUCER" != three_kernel ]]; then
        echo "AITER_REPRO_GEMM2_APRE_PRODUCER must be rowgroup or three_kernel" >&2
        exit 2
    fi
    if [[ ! "$GEMM2_APRE_RPW" =~ ^(1|2|4|8)$ ]]; then
        echo "AITER_REPRO_GEMM2_APRE_RPW must be one of 1, 2, 4, 8" >&2
        exit 2
    fi
    if [[ ! "$GEMM2_APRE_PREFETCH" =~ ^(1|2|4|8)$ ]]; then
        echo "AITER_REPRO_GEMM2_APRE_PREFETCH must be one of 1, 2, 4, 8" >&2
        exit 2
    fi
fi

declare -A seen_cases=()
for name in "${CASES[@]}"; do
    if [[ -z "$name" || -n "${seen_cases[$name]:-}" ]]; then
        echo "Case list contains an empty or duplicate case: '$name'" >&2
        exit 2
    fi
    case_env "$name"
    seen_cases[$name]=1
done

e2e_tsv="$out_dir/e2e.tsv"
att_tsv="$out_dir/att.tsv"
printf 'data\tround\torder\tcase\ttarget\treturn_code\tgemm1_us\tgemm2_us\tmoe_e2e_us\tlogits_diff\trel_l2\tpass\tmoe_output_hash128\tref_output_hash128\tgemm1_tflops\tgemm1_rw_tbps\tgemm2_tflops\tgemm2_rw_tbps\tgemm1_symbol\tgemm2_symbol\n' >"$e2e_tsv"
printf 'case\ttarget\treturn_code\tkernel\n' >"$att_tsv"

extract_precision_metrics() {
    local log="$1"
    python3 - "$log" <<'PY'
import re
import sys
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
    if "data_format" in values and "gemm1 executed" in values:
        header = values
        continue
    if header and values and values[0] == "a4w4" and len(values) == len(header):
        row = dict(zip(header, values))

if row is None:
    print("NA\tNA\tNA\tNA\tNA\tNA\tNA")
    raise SystemExit(0)


def rate(column: str, unit: str) -> str:
    match = re.search(r"([0-9][0-9,]*(?:\.[0-9]+)?)\s+" + re.escape(unit), row[column])
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

print_context() {
    echo "date_utc=$timestamp"
    echo "host=$host"
    echo "branch=$branch"
    echo "head=$head"
    echo "mode=$MODE"
    echo "target=$TARGET"
    echo "e2e_iters=$E2E_ITERS"
    echo "e2e_rounds=$E2E_ROUNDS"
    echo "run_verify=$RUN_VERIFY"
    echo "run_att=$RUN_ATT"
    echo "expert_balance=${AITER_REPRO_EXPERT_BALANCE:-true}"
    echo "att_validate_only=$ATT_VALIDATE_ONLY"
    echo "att_e2e_iters=$ATT_E2E_ITERS"
    echo "att_kernel_iteration_range=$ATT_KERNEL_ITERATION_RANGE"
    echo "att_simd_list=$ATT_SIMD_LIST"
    echo "att_target_cu=$ATT_TARGET_CU"
    echo "att_shader_engine_mask=$ATT_SHADER_ENGINE_MASK"
    echo "att_buffer_size=$ATT_BUFFER_SIZE"
    echo "att_timeout_seconds=$ATT_TIMEOUT_SECONDS"
    echo "execution=inside-container"
    echo "moe_e2e_metric=fused_moe end-to-end us"
    if [[ "$TARGET" == gemm1 ]]; then
        echo "baseline_commit=93665e8417afe1f07cb9bbe1c4902c38da8e3fa3"
    else
        echo "gemm2_baseline=current FlyDSL GEMM2"
        echo "gemm1_for_gemm2=$GEMM1_FOR_GEMM2"
        echo "gemm2_apre_producer=$GEMM2_APRE_PRODUCER"
        echo "gemm2_apre_rows_per_wave=$GEMM2_APRE_RPW"
        echo "gemm2_apre_prefetch=$GEMM2_APRE_PREFETCH"
    fi
    echo "cases=${CASES[*]}"
    echo
    echo "task source hashes:"
    sha256sum \
        my_code/reproduce_compare.sh \
        my_code/run_gemm1_baseline_93665e.py \
        aiter/ops/flydsl/grouped_gemm_mxfp4.py \
        aiter/ops/flydsl/grouped_moe_gfx1250.py \
        aiter/ops/flydsl/moe_kernels.py \
        aiter/ops/flydsl/kernels/gemm_common_gfx1250.py \
        aiter/ops/flydsl/kernels/moe_fused_route_quant_scatter.py \
        aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm_93665e.py \
        aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py \
        my_code/test_flydsl_grouped_gemm_gfx1250.py
}

run_e2e_case() {
    local data="$1"
    local round="$2"
    local order="$3"
    local name="$4"
    local log="$out_dir/e2e_${data}_r${round}_${order}_${name}.log"
    local rc gemm1 gemm2 moe_e2e moe_hash ref_hash gemm1_symbol gemm2_symbol
    local logits rel pass gemm1_tflops gemm1_rw gemm2_tflops gemm2_rw
    local primary_us primary_tflops primary_rw
    local -a const_args=()
    case_env "$name"
    if [[ "$data" == const0 ]]; then
        const_args=(--const-init 0)
    fi
    echo "===== e2e target=$TARGET case=$name data=$data round=$round order=$order ====="
    set +e
    env "${CLEAR_ENV[@]}" "${COMMON_ENV[@]}" "${CASE_ENV[@]}" \
        "AITER_REPRO_USE_BASELINE_93665E=$CASE_BASELINE_FLAG" \
        python3 -u "$e2e_runner" \
        --scenario bench "${TEST_SHAPE[@]}" --iters "$E2E_ITERS" \
        "${const_args[@]}" \
        2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    set -e

    gemm1="$(sed -n 's/.*gemm1: device_time_avg=\([0-9.]*\) us.*/\1/p' "$log" | tail -1)"
    gemm2="$(sed -n 's/.*gemm2: device_time_avg=\([0-9.]*\) us.*/\1/p' "$log" | tail -1)"
    moe_e2e="$(sed -n 's/.*fused_moe end-to-end us = \([0-9.]*\).*/\1/p' "$log" | tail -1)"
    gemm1_symbol="$(sed -n 's/.*gemm1: device_time_avg=[0-9.]* us count=[0-9]* symbol=//p' "$log" | tail -1)"
    gemm2_symbol="$(sed -n 's/.*gemm2: device_time_avg=[0-9.]* us count=[0-9]* symbol=//p' "$log" | tail -1)"
    moe_hash="$(sed -n 's/.*moe_output_hash128=//p' "$log" | tail -1)"
    ref_hash="$(sed -n 's/.*ref_output_hash128=//p' "$log" | tail -1)"
    IFS=$'\t' read -r logits rel pass gemm1_tflops gemm1_rw gemm2_tflops gemm2_rw \
        < <(extract_precision_metrics "$log")

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$data" "$round" "$order" "$name" "$TARGET" "$rc" \
        "${gemm1:-NA}" "${gemm2:-NA}" "${moe_e2e:-NA}" \
        "${logits:-NA}" "${rel:-NA}" "${pass:-NA}" \
        "${moe_hash:-NA}" "${ref_hash:-NA}" \
        "${gemm1_tflops:-NA}" "${gemm1_rw:-NA}" \
        "${gemm2_tflops:-NA}" "${gemm2_rw:-NA}" \
        "${gemm1_symbol:-NA}" "${gemm2_symbol:-NA}" \
        >>"$e2e_tsv"

    primary_us="$gemm1"
    primary_tflops="$gemm1_tflops"
    primary_rw="$gemm1_rw"
    if [[ "$TARGET" == gemm2 ]]; then
        primary_us="$gemm2"
        primary_tflops="$gemm2_tflops"
        primary_rw="$gemm2_rw"
    fi
    if [[ "$rc" -ne 0 || -z "$primary_us" || -z "$moe_e2e" \
          || "$primary_tflops" == NA || "$primary_rw" == NA ]]; then
        echo "e2e launch, timing, or effective-metric extraction failed for $name" >&2
        exit 4
    fi
    if [[ "$RUN_VERIFY" == 1 && ( "$pass" != True \
          || ! "$moe_hash" =~ ^[0-9a-f]{32}$ \
          || ! "$ref_hash" =~ ^[0-9a-f]{32}$ ) ]]; then
        echo "e2e correctness verification failed for $name" >&2
        exit 4
    fi
}

run_e2e() {
    local data="$1"
    local round order name i
    local -a order_cases=()
    for ((round = 1; round <= E2E_ROUNDS; ++round)); do
        if ((round % 2 == 1)); then
            order_cases=("${CASES[@]}")
        else
            order_cases=()
            for ((i = ${#CASES[@]} - 1; i >= 0; --i)); do
                order_cases+=("${CASES[i]}")
            done
        fi
        order=0
        for name in "${order_cases[@]}"; do
            order=$((order + 1))
            run_e2e_case "$data" "$round" "$order" "$name"
        done
    done
}

escape_kernel_regex() {
    python3 - "$1" <<'PY'
import re
import sys
print(re.escape(sys.argv[1]))
PY
}

write_att_yaml() {
    local yaml="$1"
    local output_directory="$2"
    local simd="$3"
    local kernel_regex="$4"
    cat >"$yaml" <<YAML
jobs:
 -
  kernel_include_regex: '^${kernel_regex}$'
  kernel_exclude_regex:
  kernel_iteration_range: "${ATT_KERNEL_ITERATION_RANGE}"
  output_file: out
  output_directory: ${output_directory}
  output_format: [csv]
  truncate_kernels: false
  sys_trace: false
  advanced_thread_trace: true
  att_target_cu: ${ATT_TARGET_CU}
  att_shader_engine_mask: "${ATT_SHADER_ENGINE_MASK}"
  att_simd_select: "${simd}"
  att_buffer_size: "${ATT_BUFFER_SIZE}"
  att_library_path: ["${ATT_DECODER_DIR}"]
YAML
}

run_att_e2e_command() {
    local name="$1"
    local log="$2"
    local yaml="${3:-}"
    local restore_errexit=0
    case_env "$name"
    local -a command=(
        env -u FLYDSL_DUMP_DIR
        "${CLEAR_ENV[@]}"
        PYTORCH_ALLOC_CONF=expandable_segments:True
        GPU_ARCHS=gfx1250
        AITER_FORCE_GFX1250=1
        HIP_VISIBLE_DEVICES=0
        "${COMMON_ENV[@]}"
        "${CASE_ENV[@]}"
        "AITER_REPRO_USE_BASELINE_93665E=$CASE_BASELINE_FLAG"
        python3 -u "$e2e_runner"
        --scenario bench
        "${TEST_SHAPE[@]}"
        --iters "$ATT_E2E_ITERS"
        --const-init 0
    )

    if [[ "$-" == *e* ]]; then
        restore_errexit=1
    fi
    set +e
    if [[ -z "$yaml" ]]; then
        timeout --signal=TERM --kill-after=20s "$ATT_TIMEOUT_SECONDS" \
            "${command[@]}" 2>&1 | tee "$log"
    else
        (
            set -e
            source "$ROCPROF_ENV"
            export PATH="$ROCPROF_BIN_DIR:$PATH"
            export ROCPROF_ATT_LIBRARY_PATH="$ATT_DECODER_DIR"
            timeout --signal=TERM --kill-after=20s "$ATT_TIMEOUT_SECONDS" \
                rocprofv3 -i "$yaml" -- "${command[@]}"
        ) 2>&1 | tee "$log"
    fi
    local rc=${PIPESTATUS[0]}
    if [[ "$restore_errexit" == 1 ]]; then
        set -e
    fi
    return "$rc"
}

validate_att_artifacts() {
    local capture_root="$1"
    local att_count code_count wave_count
    att_count="$(find "$capture_root" -type f -name '*.att' -size +0c | wc -l)"
    code_count="$(find "$capture_root" -type f -path '*/ui_output_agent_*/code.json' -size +0c | wc -l)"
    wave_count="$(find "$capture_root" -type f -path '*/ui_output_agent_*/se*_sm*_sl*_wv*.json' -size +0c | wc -l)"
    if [[ "$att_count" -lt 1 || "$code_count" -ne 1 || "$wave_count" -lt 1 ]]; then
        echo "incomplete ATT output under $capture_root: att=$att_count code=$code_count waves=$wave_count" >&2
        return 1
    fi
    echo "ATT artifacts verified: att=$att_count code=$code_count waves=$wave_count root=$capture_root"
}

run_att_case() {
    local name="$1"
    local case_root="$att_root/$name"
    local log rc simd capture_root yaml kernel kernel_regex
    mkdir -p "$case_root/logs" "$case_root/thread_trace"
    echo "===== ATT target=$TARGET case=$name ====="

    log="$case_root/logs/preflight_e2e.log"
    set +e
    run_att_e2e_command "$name" "$log"
    rc=$?
    set -e
    kernel="$(sed -n "s/.*${TARGET}: device_time_avg=[0-9.]* us count=[0-9]* symbol=//p" "$log" | tail -1)"
    if [[ "$rc" -eq 0 && -z "$kernel" ]]; then
        echo "ATT preflight could not identify the $TARGET kernel for $name" >&2
        rc=1
    fi

    if [[ "$rc" -eq 0 && "$ATT_VALIDATE_ONLY" != 1 ]]; then
        kernel_regex="$(escape_kernel_regex "$kernel")"
        for simd in "${ATT_SIMDS[@]}"; do
            capture_root="$case_root/thread_trace/simd${simd}/kernel/rpf_v3"
            yaml="$case_root/input_simd${simd}.yaml"
            log="$case_root/logs/capture_simd${simd}.log"
            rm -rf "$capture_root"
            mkdir -p "$capture_root"
            write_att_yaml "$yaml" "$capture_root" "$simd" "$kernel_regex"
            set +e
            run_att_e2e_command "$name" "$log" "$yaml"
            rc=$?
            set -e
            if [[ "$rc" -ne 0 ]]; then
                echo "ATT capture failed for $name SIMD$simd with return code $rc" >&2
                break
            fi
            if ! validate_att_artifacts "$capture_root" | tee -a "$log"; then
                rc=1
                break
            fi
        done
        if [[ "$rc" -eq 0 && "$ATT_SHADER_ENGINE_MASK" == 0xf ]]; then
            set +e
            python3 "$REPO_ROOT/my_code/analyze_att_capture.py" \
                --dir "$case_root" --no-plot \
                2>&1 | tee "$case_root/logs/analyze_att_capture.log"
            rc=${PIPESTATUS[0]}
            set -e
        elif [[ "$rc" -eq 0 ]]; then
            printf '%s\n' \
                "Skipping analyze_att_capture.py because ATT_SHADER_ENGINE_MASK=$ATT_SHADER_ENGINE_MASK does not capture all SEs." \
                "The decoded code.json and wave JSON files were verified for every requested SIMD." \
                | tee "$case_root/logs/analyze_att_capture.log"
        fi
    fi

    printf '%s\t%s\t%s\t%s\n' "$name" "$TARGET" "$rc" "${kernel:-NA}" >>"$att_tsv"
    if [[ "$rc" -ne 0 ]]; then
        echo "ATT failed for $name with return code $rc" >&2
        exit 5
    fi
}

run_att() {
    local name lock_file
    att_root="$out_dir/att"
    mkdir -p "$att_root"
    if [[ ! -f "$ROCPROF_ENV" ]]; then
        echo "missing rocprof environment: $ROCPROF_ENV" >&2
        exit 5
    fi
    if [[ ! -x "$ROCPROF_BIN_DIR/rocprofv3" ]]; then
        echo "missing rocprofv3: $ROCPROF_BIN_DIR/rocprofv3" >&2
        exit 5
    fi
    if [[ ! -s "$ATT_DECODER_DIR/librocprof-trace-decoder.so" ]]; then
        echo "missing pinned ATT decoder: $ATT_DECODER_DIR/librocprof-trace-decoder.so" >&2
        exit 5
    fi
    lock_file="/tmp/aiter_reproduce_${TARGET}_att_${UID}.lock"
    exec {att_lock_fd}>"$lock_file"
    if ! flock -n "$att_lock_fd"; then
        echo "another reproduce_compare.sh ATT capture owns $lock_file" >&2
        exit 5
    fi
    for name in "${CASES[@]}"; do
        run_att_case "$name"
    done
}

write_summary() {
    python3 - "$e2e_tsv" "$TARGET" "${CASES[@]}" <<'PY' >"$out_dir/summary.md"
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path


path = Path(sys.argv[1])
target = sys.argv[2]
case_order = sys.argv[3:]
with path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
if not rows:
    raise SystemExit(0)

grouped = defaultdict(list)
for row in rows:
    grouped[(row["data"], row["case"])].append(row)
data_order = list(dict.fromkeys(row["data"] for row in rows))
stage = target.upper()
stage_us_key = f"{target}_us"
tflops_key = f"{target}_tflops"
rw_key = f"{target}_rw_tbps"
preferred_baseline_case = "baseline_93665e" if target == "gemm1" else "baseline"
baseline_case = (
    preferred_baseline_case
    if preferred_baseline_case in case_order
    else case_order[0]
)
comparison = (
    "93665e"
    if baseline_case == "baseline_93665e"
    else "baseline"
    if baseline_case == "baseline"
    else f"first case ({baseline_case})"
)

print(
    f"| data | case | {stage} samples (us) | {stage} median us | "
    f"{stage} vs {comparison} | {stage} effective R+W (TB/s) | "
    f"{stage} executed (TFLOP/s) | MoE e2e samples (us) | "
    f"MoE e2e median us | MoE e2e vs {comparison} | pass | logits_diff | "
    "rel_l2 | MoE output hash128 | ref output hash128 |"
)
print("|---|---|---|---:|---:|---:|---:|---|---:|---:|:---:|---:|---:|---|---|")
for data in data_order:
    present = [name for name in case_order if (data, name) in grouped]
    if baseline_case not in present:
        raise SystemExit(f"missing baseline case {baseline_case!r} for data={data}")
    baseline_stage = statistics.median(
        float(row[stage_us_key]) for row in grouped[(data, baseline_case)]
    )
    baseline_e2e = statistics.median(
        float(row["moe_e2e_us"]) for row in grouped[(data, baseline_case)]
    )
    for name in present:
        case_rows = grouped[(data, name)]
        stage_values = [float(row[stage_us_key]) for row in case_rows]
        e2e_values = [float(row["moe_e2e_us"]) for row in case_rows]
        stage_med = statistics.median(stage_values)
        e2e_med = statistics.median(e2e_values)
        stage_gain = (baseline_stage - stage_med) / baseline_stage * 100.0
        e2e_gain = (baseline_e2e - e2e_med) / baseline_e2e * 100.0
        rw_med = statistics.median(float(row[rw_key]) for row in case_rows)
        tflops_med = statistics.median(float(row[tflops_key]) for row in case_rows)
        last = case_rows[-1]
        print(
            f"| {data} | {name} | "
            f"{', '.join(f'{value:.3f}' for value in stage_values)} | "
            f"{stage_med:.3f} | {stage_gain:+.2f}% | {rw_med:.3f} | "
            f"{tflops_med:.1f} | "
            f"{', '.join(f'{value:.2f}' for value in e2e_values)} | "
            f"{e2e_med:.2f} | {e2e_gain:+.2f}% | {last['pass']} | "
            f"{last['logits_diff']} | {last['rel_l2']} | "
            f"{last['moe_output_hash128']} | {last['ref_output_hash128']} |"
        )
PY
    if [[ -s "$out_dir/summary.md" ]]; then
        cat "$out_dir/summary.md"
    fi
}

run_all() {
    print_context | tee "$out_dir/environment.log"
    case "$MODE" in
        e2e-random)
            run_e2e random
            ;;
        e2e-const0)
            run_e2e const0
            ;;
        e2e-both)
            run_e2e random
            run_e2e const0
            ;;
        att|att-validate)
            run_att
            ;;
    esac
    if [[ "$RUN_ATT" == 1 && "$MODE" != att && "$MODE" != att-validate ]]; then
        run_att
    fi
    write_summary
}

run_all 2>&1 | tee "$out_dir/run.log"
chmod -R a+rwX "$out_dir"
echo "Results: $out_dir"
