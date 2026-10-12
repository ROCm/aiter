#!/usr/bin/env bash
# split_tests.sh — shards tests in op_tests/triton_tests
# N shards, shards with similar total test time

# Usage:
#   bash .github/scripts/split_tests.sh --shards N [--test-dir DIR]
#
# Parameters:
#   --shards N     number of shards (required)
#   --test-type TYPE test type, default aiter
#   --select-file F  only shard test files listed in F (one per line)
#   --dry-run      only output allocation plan, do not execute
#   -v             Pytest's -v option, no effect
# Exit code: 0 on success, 1 for invalid arguments or selected paths

set -euo pipefail

SHARDS=0
TEST_TYPE="aiter"
DRY_RUN=0
SELECT_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --shards) SHARDS="$2"; shift 2 ;;
        --test-type) TEST_TYPE="$2"; shift 2 ;;
        --select-file) SELECT_FILE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -v|--verbose) shift ;; # compatibility, ignore
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ "$TEST_TYPE" == "aiter" ]]; then
    TEST_DIR="op_tests"
elif [[ "$TEST_TYPE" == "triton" ]]; then
    TEST_DIR="op_tests/triton_tests"
else
    echo "Unknown test type: $TEST_TYPE" >&2
    exit 1
fi

if ! [[ "$SHARDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Use --shards N to specify the number of shards (positive integer)" >&2
    exit 1
fi
TEST_DIR="${TEST_DIR%/}"

# ------------------------------
# scan test files in TEST_DIR
# ------------------------------
if [[ "$TEST_TYPE" == "aiter" ]]; then
    mapfile -t ALL_FILES < <(
        {
            find "$TEST_DIR" -maxdepth 1 -name 'test_*.py' -type f
            printf '%s\n' \
                "$TEST_DIR/tuning_tests/test_csv_validation.py" \
                "$TEST_DIR/tuning_tests/test_config_shape_collision.py" \
                "$TEST_DIR/tuning_tests/test_mixed_mxfp_tuning.py"
        } | LC_ALL=C sort -u
    )
elif [[ "$TEST_TYPE" == "triton" ]]; then
    mapfile -t ALL_FILES < <(find "$TEST_DIR" -name 'test_*.py' -type f | LC_ALL=C sort)
fi
if [[ ${#ALL_FILES[@]} -eq 0 ]]; then
    echo "No test files found: $TEST_DIR/test_*.py" >&2
    exit 1
fi

# Apply the optional selection before sharding.
if [[ -n "$SELECT_FILE" ]]; then
    if [[ ! -f "$SELECT_FILE" ]]; then
        echo "Selection file not found: $SELECT_FILE" >&2
        exit 1
    fi
    declare -A SELECTED=()
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ -n "$line" ]] && SELECTED["$line"]=1
    done < "$SELECT_FILE"
    FILTERED=()
    for f in "${ALL_FILES[@]}"; do
        if [[ -n "${SELECTED[$f]:-}" ]]; then
            FILTERED+=("$f")
            unset "SELECTED[$f]"
        fi
    done
    # Never silently drop a path the selector asked to run.
    if [[ ${#SELECTED[@]} -gt 0 ]]; then
        echo "Selection lists paths that are not test files under ${TEST_DIR}:" >&2
        for f in "${!SELECTED[@]}"; do echo "  ${f}" >&2; done
        exit 1
    fi
    echo "Test selection: ${#FILTERED[@]} of ${#ALL_FILES[@]} test files selected."
    if [[ ${#FILTERED[@]} -eq 0 ]]; then
        echo "Selection is empty — writing ${SHARDS} empty shard lists."
        if [[ $DRY_RUN -eq 0 ]]; then
            for ((s=0; s < SHARDS; s++)); do
                : > "${TEST_TYPE}_shard_${s}.list"
            done
        fi
        exit 0
    fi
    ALL_FILES=("${FILTERED[@]}")
fi

# ------------------------------
# FILE_TIMES (seconds), unknown files default 15
# ------------------------------
DEFAULT_TEST_TIME=15
declare -A FILE_TIMES
TIMES_FILE=".github/split-test-times/${TEST_TYPE}.tsv"

if [[ "$TEST_TYPE" == "aiter" ]]; then
    echo "Aiter test files:"
else
    echo "Triton test files:"
fi

if [[ ! -f "$TIMES_FILE" ]]; then
    echo "Split test times file not found: $TIMES_FILE" >&2
    exit 1
fi

while IFS=$'	' read -r file_path seconds || [[ -n "$file_path" ]]; do
    [[ -z "$file_path" || "$file_path" == \#* ]] && continue
    if [[ ! "$seconds" =~ ^[1-9][0-9]*$ ]]; then
        echo "Invalid split test time in $TIMES_FILE: ${file_path}<TAB>${seconds}" >&2
        exit 1
    fi
    FILE_TIMES["$file_path"]="$seconds"
done < "$TIMES_FILE"

get_time() {
    local abs="$1"
    local seconds
    # FILE_TIMES keys use full path (e.g. op_tests/test_mla.py), so look up with abs
    if [[ -n "${FILE_TIMES[$abs]+x}" ]]; then
        seconds="${FILE_TIMES[$abs]}"
    else
        seconds="${DEFAULT_TEST_TIME}"
    fi

    if [[ -n "${MEMORY_WEIGHT_FLOOR[$abs]+x}" && "$seconds" -lt "${MEMORY_WEIGHT_FLOOR[$abs]}" ]]; then
        echo "${MEMORY_WEIGHT_FLOOR[$abs]}"
    else
        echo "$seconds"
    fi
}

# Some tests have short wall time but high peak memory usage. Give them a
# scheduling weight floor so the greedy splitter avoids packing them together.
declare -A MEMORY_WEIGHT_FLOOR
if [[ "$TEST_TYPE" == "aiter" ]]; then
    MEMORY_WEIGHT_FLOOR[op_tests/test_flydsl_causal_conv1d_update.py]=300
    MEMORY_WEIGHT_FLOOR[op_tests/test_flydsl_gdr_mtp.py]=300
    MEMORY_WEIGHT_FLOOR[op_tests/test_flydsl_qk_norm_rope_quant.py]=300
    MEMORY_WEIGHT_FLOOR[op_tests/test_kvcache.py]=300
    MEMORY_WEIGHT_FLOOR[op_tests/test_mla_prefill_ps.py]=300
fi

# ------------------------------
# LPT greedy allocation: sort first then distribute
# ------------------------------
declare -a SORTED_FILES
for f in "${ALL_FILES[@]}"; do
    t=$(get_time "$f")
    SORTED_FILES+=("$t $f")
done

IFS=$'\n' SORTED_FILES=($(sort -nr <<<"${SORTED_FILES[*]}"))
unset IFS

declare -a SHARD_LOADS
declare -a SHARD_FILES

for ((i=0; i < SHARDS; i++)); do
    SHARD_LOADS[$i]=0
    SHARD_FILES[$i]=""
done

for entry in "${SORTED_FILES[@]}"; do
    t="${entry%% *}"
    f="${entry#* }"
    min_shard=0
    min_load="${SHARD_LOADS[0]}"
    for ((s=1; s < SHARDS; s++)); do
        if [[ ${SHARD_LOADS[$s]} -lt $min_load ]]; then
            min_shard=$s
            min_load=${SHARD_LOADS[$s]}
        fi
    done
    SHARD_LOADS[$min_shard]=$(( ${SHARD_LOADS[$min_shard]} + t ))
    if [[ -z "${SHARD_FILES[$min_shard]}" ]]; then
        SHARD_FILES[$min_shard]="$f"
    else
        SHARD_FILES[$min_shard]+=" $f"
    fi
done

# ------------------------------
# output allocation plan
# ------------------------------
echo "================= ${TEST_TYPE} Shard Assignment ================="
for ((s=0; s < SHARDS; s++)); do
    nfiles=0
    if [[ -n "${SHARD_FILES[$s]}" ]]; then
        nfiles=$(wc -w <<< "${SHARD_FILES[$s]}")
    fi
    echo "Shard $s: ${nfiles} files, est. ${SHARD_LOADS[$s]}s"
    for f in ${SHARD_FILES[$s]}; do
        printf "  [%4ss] %s\n" "$(get_time "$f")" "$f"
    done
    echo ""
done
echo "==========================================================="

if [[ $DRY_RUN -eq 1 ]]; then
    exit 0
fi

# output each shard's test files list to local text file
for ((s=0; s < SHARDS; s++)); do
    echo "${SHARD_FILES[$s]}" > "${TEST_TYPE}_shard_${s}.list"
done

exit 
