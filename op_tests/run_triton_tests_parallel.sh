#!/usr/bin/env bash
# run_triton_tests_parallel.sh — Distribute triton test files across GPUs
#
# Usage:
#   bash op_tests/run_triton_tests_parallel.sh [OPTIONS]
#
# Options:
#   --ngpus N        Number of GPUs to use (default: auto-detect via rocm-smi)
#   --timeout N      Per-GPU pytest timeout in seconds (default: 900)
#   --junit-dir DIR  Directory for JUnit XML reports (one per GPU)
#   --test-dir DIR   Root directory containing test files (default: op_tests/triton_tests)
#   -v, --verbose    Pass -v to pytest
#   --dry-run        Print GPU assignments without running tests
#
# Exit code: 0 if all GPUs pass, 1 if any GPU has failures.

set -euo pipefail

# ────────────────────────────────────────────────────────────────────────────
# Defaults
# ────────────────────────────────────────────────────────────────────────────
NGPUS=""
TIMEOUT=900
JUNIT_DIR=""
TEST_DIR="op_tests/triton_tests"
VERBOSE=""
DRY_RUN=0

# ────────────────────────────────────────────────────────────────────────────
# Parse arguments
# ────────────────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ngpus)   NGPUS="$2"; shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        --junit-dir) JUNIT_DIR="$2"; shift 2 ;;
        --test-dir)  TEST_DIR="$2"; shift 2 ;;
        -v|--verbose) VERBOSE="-v"; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# Normalize TEST_DIR: strip trailing slash(es) so prefix stripping works
TEST_DIR="${TEST_DIR%/}"

# Validate numeric arguments
if [[ -n "$NGPUS" ]] && ! [[ "$NGPUS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --ngpus must be a positive integer, got '$NGPUS'" >&2
    exit 1
fi
if ! [[ "$TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --timeout must be a positive integer, got '$TIMEOUT'" >&2
    exit 1
fi

# ────────────────────────────────────────────────────────────────────────────
# Auto-detect GPUs
# ────────────────────────────────────────────────────────────────────────────
if [[ -z "$NGPUS" ]]; then
    if command -v rocm-smi &>/dev/null; then
        NGPUS=$(rocm-smi --showid 2>/dev/null | grep -c "^[0-9]" || true)
    fi
    if [[ -z "$NGPUS" ]] || [[ "$NGPUS" -eq 0 ]]; then
        # Fallback: count /dev/dri/renderD* devices
        NGPUS=$(ls /dev/dri/renderD* 2>/dev/null | wc -l || echo 1)
    fi
    if [[ "$NGPUS" -eq 0 ]]; then
        NGPUS=1
    fi
fi

echo "=========================================="
echo " Triton Parallel Test Runner"
echo " GPUs: $NGPUS | Timeout: ${TIMEOUT}s"
echo "=========================================="

# ────────────────────────────────────────────────────────────────────────────
# Measured per-file timings (seconds). Used for LPT bin-packing.
# Files not listed here get a default estimate of 15s.
# Timings measured on MI300X, May 2025.
# ────────────────────────────────────────────────────────────────────────────
declare -A FILE_TIMES
# XL tier (300-600s+)
FILE_TIMES[rope/test_rope.py]=600
FILE_TIMES[attention/test_pa_decode.py]=601
FILE_TIMES[test_causal_conv1d.py]=600
FILE_TIMES[test_gmm.py]=300
FILE_TIMES[test_gated_delta_rule.py]=300
FILE_TIMES[gemm/batched/test_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py]=301
# Large tier (200-300s)
FILE_TIMES[attention/test_chunked_pa_prefill.py]=295
FILE_TIMES[attention/test_pa_prefill.py]=277
FILE_TIMES[test_pa_decode_gluon.py]=276
FILE_TIMES[attention/test_mha.py]=254
# Medium tier (30-80s)
FILE_TIMES[gemm/basic/test_gemm_a8w8.py]=79
FILE_TIMES[moe/test_moe_gemm_a8w8_blockscale.py]=71
FILE_TIMES[quant/test_fused_fp8_quant.py]=49
FILE_TIMES[moe/test_moe_gemm_a8w8.py]=45
FILE_TIMES[gemm/basic/test_gemm_a8w8_blockscale.py]=40
FILE_TIMES[gemm/basic/test_gemm_a16w8_blockscale.py]=35
FILE_TIMES[moe/test_moe.py]=35
FILE_TIMES[attention/test_unified_attention.py]=30
FILE_TIMES[attention/test_prefill_attention.py]=30
FILE_TIMES[rope/test_fused_qkv_split_qk_rope.py]=30
FILE_TIMES[gemm/basic/test_gemm_a16w16.py]=25
FILE_TIMES[gemm/basic/test_gemm_a16wfp4.py]=25
FILE_TIMES[gemm/basic/test_gemm_a8wfp4.py]=25
FILE_TIMES[gemm/basic/test_gemm_afp4wfp4.py]=25
FILE_TIMES[gemm/basic/test_gemm_a8w8_per_token_scale.py]=25
FILE_TIMES[moe/test_moe_gemm_a4w4.py]=25
FILE_TIMES[moe/test_moe_gemm_a8w4.py]=25
FILE_TIMES[attention/test_mla_decode_rope.py]=20
FILE_TIMES[gemm/batched/test_batched_gemm_a8w8.py]=20
FILE_TIMES[gemm/batched/test_batched_gemm_afp4wfp4.py]=20
FILE_TIMES[gemm/batched/test_batched_gemm_a16wfp4.py]=20
FILE_TIMES[gemm/batched/test_batched_gemm_bf16.py]=20

# ────────────────────────────────────────────────────────────────────────────
# Discover test files
# ────────────────────────────────────────────────────────────────────────────
mapfile -t ALL_FILES < <(find "$TEST_DIR" -name 'test_*.py' -type f | sort)

if [[ ${#ALL_FILES[@]} -eq 0 ]]; then
    echo "ERROR: No test files found in $TEST_DIR" >&2
    exit 1
fi
echo "Found ${#ALL_FILES[@]} test files"

# ────────────────────────────────────────────────────────────────────────────
# Helper: get estimated time for a file (relative to TEST_DIR)
# ────────────────────────────────────────────────────────────────────────────
get_time() {
    local filepath="$1"
    # Strip the TEST_DIR prefix to get relative path
    local rel="${filepath#${TEST_DIR}/}"
    if [[ -n "${FILE_TIMES[$rel]+x}" ]]; then
        echo "${FILE_TIMES[$rel]}"
    else
        echo 15  # default for unknown files
    fi
}

# ────────────────────────────────────────────────────────────────────────────
# LPT (Longest Processing Time First) bin-packing
#
# 1. Sort files by estimated time (descending)
# 2. Assign each file to the GPU with the smallest current total
# ────────────────────────────────────────────────────────────────────────────

# Build array of "time filepath" pairs and sort descending
declare -a SORTED_FILES
while IFS= read -r line; do
    SORTED_FILES+=("$line")
done < <(
    for f in "${ALL_FILES[@]}"; do
        t=$(get_time "$f")
        printf '%d %s\n' "$t" "$f"
    done | sort -t' ' -k1 -rn
)

# Per-GPU assignment arrays and load counters
declare -a GPU_LOADS
declare -a GPU_FILES  # space-separated file lists per GPU
for ((g = 0; g < NGPUS; g++)); do
    GPU_LOADS[$g]=0
    GPU_FILES[$g]=""
done

for entry in "${SORTED_FILES[@]}"; do
    t="${entry%% *}"
    f="${entry#* }"

    # Find GPU with minimum load
    min_gpu=0
    min_load=${GPU_LOADS[0]}
    for ((g = 1; g < NGPUS; g++)); do
        if [[ ${GPU_LOADS[$g]} -lt $min_load ]]; then
            min_load=${GPU_LOADS[$g]}
            min_gpu=$g
        fi
    done

    GPU_LOADS[$min_gpu]=$(( ${GPU_LOADS[$min_gpu]} + t ))
    if [[ -z "${GPU_FILES[$min_gpu]}" ]]; then
        GPU_FILES[$min_gpu]="$f"
    else
        GPU_FILES[$min_gpu]+=" $f"
    fi
done

# ────────────────────────────────────────────────────────────────────────────
# Print assignment summary
# ────────────────────────────────────────────────────────────────────────────
echo ""
echo "GPU Assignments:"
echo "------------------------------------------"
for ((g = 0; g < NGPUS; g++)); do
    # Count files assigned to this GPU
    if [[ -z "${GPU_FILES[$g]}" ]]; then
        nfiles=0
    else
        nfiles=$(echo "${GPU_FILES[$g]}" | wc -w)
    fi
    echo "  GPU $g: ${nfiles} files, est. ${GPU_LOADS[$g]}s"
done
echo "------------------------------------------"
echo ""

if [[ $DRY_RUN -eq 1 ]]; then
    for ((g = 0; g < NGPUS; g++)); do
        echo "=== GPU $g (est. ${GPU_LOADS[$g]}s) ==="
        for f in ${GPU_FILES[$g]}; do
            t=$(get_time "$f")
            echo "  [${t}s] $f"
        done
        echo ""
    done
    echo "(dry run — no tests executed)"
    exit 0
fi

# ────────────────────────────────────────────────────────────────────────────
# Run tests in parallel (one pytest per GPU)
# ────────────────────────────────────────────────────────────────────────────
TMPDIR_BASE=$(mktemp -d /tmp/triton_parallel.XXXXXX)
cleanup() { rm -rf "$TMPDIR_BASE"; }
trap cleanup EXIT INT TERM
declare -a PIDS

for ((g = 0; g < NGPUS; g++)); do
    if [[ -z "${GPU_FILES[$g]}" ]]; then
        continue
    fi

    LOGFILE="${TMPDIR_BASE}/gpu${g}.log"

    # Build pytest argv as a proper array to avoid word-splitting issues
    declare -a PYTEST_ARGS=()
    [[ -n "$VERBOSE" ]] && PYTEST_ARGS+=("$VERBOSE")
    # shellcheck disable=SC2206  # GPU_FILES entries are find-produced paths (no spaces)
    PYTEST_ARGS+=(${GPU_FILES[$g]})
    if [[ -n "$JUNIT_DIR" ]]; then
        mkdir -p "$JUNIT_DIR"
        PYTEST_ARGS+=("--junitxml=${JUNIT_DIR}/triton_gpu${g}.xml")
    fi

    (
        set +e
        echo "[GPU $g] START $(date '+%H:%M:%S')"
        START_TS=$(date +%s)

        HIP_VISIBLE_DEVICES=$g timeout "$TIMEOUT" \
            pytest "${PYTEST_ARGS[@]}" 2>&1

        rc=$?
        END_TS=$(date +%s)
        ELAPSED=$(( END_TS - START_TS ))
        echo "[GPU $g] DONE  $(date '+%H:%M:%S') (${ELAPSED}s, exit=$rc)"
        exit $rc
    ) > "$LOGFILE" 2>&1 &

    PIDS[$g]=$!
    echo "Launched GPU $g (PID ${PIDS[$g]})"
done

# ────────────────────────────────────────────────────────────────────────────
# Wait for all GPU processes and collect results
# ────────────────────────────────────────────────────────────────────────────
declare -a EXIT_CODES
OVERALL_RC=0

set +e  # allow wait to return non-zero without exiting
for ((g = 0; g < NGPUS; g++)); do
    if [[ -z "${PIDS[$g]+x}" ]]; then
        EXIT_CODES[$g]=-1  # no files assigned
        continue
    fi
    wait "${PIDS[$g]}" 2>/dev/null
    EXIT_CODES[$g]=$?
    if [[ ${EXIT_CODES[$g]} -ne 0 ]]; then
        OVERALL_RC=1
    fi
done
set -e

# ────────────────────────────────────────────────────────────────────────────
# Print results summary
# ────────────────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo " Results Summary"
echo "=========================================="
for ((g = 0; g < NGPUS; g++)); do
    if [[ ${EXIT_CODES[$g]} -eq -1 ]]; then
        echo "  GPU $g: (no files assigned)"
        continue
    fi

    LOGFILE="${TMPDIR_BASE}/gpu${g}.log"
    if [[ ${EXIT_CODES[$g]} -eq 0 ]]; then
        STATUS="PASS"
    elif [[ ${EXIT_CODES[$g]} -eq 124 ]]; then
        STATUS="TIMEOUT"
    else
        STATUS="FAIL (exit=${EXIT_CODES[$g]})"
    fi

    # Extract pytest summary line if available
    SUMMARY=$(grep -E "^(FAILED|PASSED|ERROR|=)" "$LOGFILE" | tail -1 || true)
    echo "  GPU $g: $STATUS  $SUMMARY"
done
echo "=========================================="

# Print full logs for failed GPUs
for ((g = 0; g < NGPUS; g++)); do
    if [[ ${EXIT_CODES[$g]} -gt 0 ]]; then
        echo ""
        echo "vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv"
        echo " GPU $g LOGS (exit=${EXIT_CODES[$g]})"
        echo "vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv"
        cat "${TMPDIR_BASE}/gpu${g}.log"
        echo "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
    fi
done

exit $OVERALL_RC
