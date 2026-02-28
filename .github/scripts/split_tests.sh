#!/usr/bin/env bash
# split_tests.sh — shards tests in op_tests/triton_tests
# N shards, shards with similar total test time

# Usage:
#   bash .github/scripts/split_tests.sh --shards N [--test-dir DIR]
#
# Parameters:
#   --shards N     number of shards (required)
#   --test-type TYPE test type, default aiter
#   --dry-run      only output allocation plan, do not execute
#   -v             Pytest's -v option, no effect
# Exit code: always 0

set -euo pipefail

SHARDS=0
TEST_TYPE="aiter"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --shards) SHARDS="$2"; shift 2 ;;
        --test-type) TEST_TYPE="$2"; shift 2 ;;
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
    mapfile -t ALL_FILES < <(find "$TEST_DIR" -maxdepth 1 -name 'test_*.py' -type f | LC_ALL=C sort)
elif [[ "$TEST_TYPE" == "triton" ]]; then
    mapfile -t ALL_FILES < <(find "$TEST_DIR" -name 'test_*.py' -type f | LC_ALL=C sort)
fi
if [[ ${#ALL_FILES[@]} -eq 0 ]]; then
    echo "No test files found: $TEST_DIR/test_*.py" >&2
    exit 1
fi

# ------------------------------
# FILE_TIMES (seconds), unknown files default 15
# ------------------------------
declare -A FILE_TIMES
if [[ "$TEST_TYPE" == "aiter" ]]; then
    echo "Aiter test files:"
    FILE_TIMES[op_tests/test_mla.py]=1115
    FILE_TIMES[op_tests/test_fused_qk_norm_mrope_cache_quant.py]=1079
    FILE_TIMES[op_tests/test_kvcache.py]=804
    FILE_TIMES[op_tests/test_mha.py]=793
    FILE_TIMES[op_tests/test_batch_prefill.py]=596
    FILE_TIMES[op_tests/test_mha_varlen.py]=595
    FILE_TIMES[op_tests/test_mla_sparse.py]=591
    FILE_TIMES[op_tests/test_pa.py]=510
    FILE_TIMES[op_tests/test_rope.py]=408
    FILE_TIMES[op_tests/test_fused_qk_norm_rope_cache_quant.py]=401
    FILE_TIMES[op_tests/test_topk_plain.py]=320
    FILE_TIMES[op_tests/test_concat_cache_mla.py]=309
    FILE_TIMES[op_tests/test_moe_sorting.py]=215
    FILE_TIMES[op_tests/test_moe_sorting_mxfp4.py]=186
    FILE_TIMES[op_tests/test_gemm_a8w8_blockscale.py]=168
    FILE_TIMES[op_tests/test_topk_per_row.py]=123
    FILE_TIMES[op_tests/test_mla_persistent.py]=121
    FILE_TIMES[op_tests/test_pa_mtp.py]=103
    FILE_TIMES[op_tests/test_jit_dir_with_enum.py]=85
    FILE_TIMES[op_tests/test_gemm_a8w8.py]=83
    FILE_TIMES[op_tests/test_moe_ep.py]=76
    FILE_TIMES[op_tests/test_pa_ps.py]=56
    FILE_TIMES[op_tests/test_quant.py]=52
    FILE_TIMES[op_tests/test_moe.py]=51
    FILE_TIMES[op_tests/test_activation.py]=33
    FILE_TIMES[op_tests/test_mha_fp8.py]=32
    FILE_TIMES[op_tests/test_batched_gemm_a8w8.py]=31
    FILE_TIMES[op_tests/test_batched_gemm_bf16.py]=31
    FILE_TIMES[op_tests/test_mha_varlen_fp8.py]=28
    FILE_TIMES[op_tests/test_pa_ragged.py]=26
    FILE_TIMES[op_tests/test_moe_dp_share_expert.py]=24
    FILE_TIMES[op_tests/test_moe_blockscale.py]=24
    FILE_TIMES[op_tests/test_kvcache_blockscale.py]=24
    FILE_TIMES[op_tests/test_rmsnorm2d.py]=21
    FILE_TIMES[op_tests/test_sampling.py]=20
    FILE_TIMES[op_tests/test_moe_tkw1.py]=17
    FILE_TIMES[op_tests/test_sample.py]=17
    FILE_TIMES[op_tests/test_pa_ragged_experimental.py]=16
    FILE_TIMES[op_tests/test_pa_v1.py]=16
    FILE_TIMES[op_tests/test_moeTopkSoftmax.py]=16
    FILE_TIMES[op_tests/test_deepgemm.py]=15
    FILE_TIMES[op_tests/test_moe_2stage.py]=12
    FILE_TIMES[op_tests/test_layernorm2dFusedAddQuant.py]=11
    FILE_TIMES[op_tests/test_gemm_a16w16.py]=10
    FILE_TIMES[op_tests/test_rmsnorm2dFusedAddQuant.py]=9
    FILE_TIMES[op_tests/test_aiter_sigmoid.py]=7
    FILE_TIMES[op_tests/test_aiter_addInp.py]=6
    FILE_TIMES[op_tests/test_smoothquant.py]=6
    FILE_TIMES[op_tests/test_aiter_add.py]=5
    FILE_TIMES[op_tests/test_mla_prefill_ps.py]=4
    FILE_TIMES[op_tests/test_gemm_a4w4.py]=4
    FILE_TIMES[op_tests/test_indexer_k_quant_and_cache.py]=4
    FILE_TIMES[op_tests/test_groupnorm.py]=4
    FILE_TIMES[op_tests/test_topk_row_prefill.py]=3
    FILE_TIMES[op_tests/test_layernorm2d.py]=3
    FILE_TIMES[op_tests/test_moe_topk_sigmoid.py]=3
elif [[ "$TEST_TYPE" == "triton" ]]; then
    echo "Triton test files:"
    FILE_TIMES[op_tests/triton_tests/rope/test_rope.py]=600
    FILE_TIMES[op_tests/triton_tests/attention/test_pa_decode.py]=601
    FILE_TIMES[op_tests/triton_tests/test_causal_conv1d.py]=600
    FILE_TIMES[op_tests/triton_tests/test_gmm.py]=300
    FILE_TIMES[op_tests/triton_tests/test_gated_delta_rule.py]=300
    FILE_TIMES[op_tests/triton_tests/gemm/batched/test_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py]=301
    FILE_TIMES[op_tests/triton_tests/attention/test_chunked_pa_prefill.py]=295
    FILE_TIMES[op_tests/triton_tests/attention/test_pa_prefill.py]=277
    FILE_TIMES[op_tests/triton_tests/test_pa_decode_gluon.py]=276
    FILE_TIMES[op_tests/triton_tests/attention/test_mha.py]=254
    FILE_TIMES[op_tests/triton_tests/gemm/basic/test_gemm_a8w8.py]=79
    FILE_TIMES[op_tests/triton_tests/moe/test_moe_gemm_a8w8_blockscale.py]=71
    FILE_TIMES[op_tests/triton_tests/quant/test_fused_fp8_quant.py]=49
    FILE_TIMES[op_tests/triton_tests/moe/test_moe_gemm_a8w8.py]=45
    FILE_TIMES[op_tests/triton_tests/gemm/basic/test_gemm_a8w8_blockscale.py]=40
    FILE_TIMES[op_tests/triton_tests/gemm/basic/test_gemm_a16w8_blockscale.py]=35
    FILE_TIMES[op_tests/triton_tests/moe/test_moe.py]=35
    FILE_TIMES[op_tests/triton_tests/attention/test_unified_attention.py]=30
    FILE_TIMES[op_tests/triton_tests/attention/test_prefill_attention.py]=30
    FILE_TIMES[op_tests/triton_tests/rope/test_fused_qkv_split_qk_rope.py]=30
    FILE_TIMES[op_tests/triton_tests/gemm/basic/test_gemm_a16w16.py]=25
    FILE_TIMES[op_tests/triton_tests/gemm/basic/test_gemm_a16wfp4.py]=25
    FILE_TIMES[op_tests/triton_tests/gemm/basic/test_gemm_a8wfp4.py]=25
    FILE_TIMES[op_tests/triton_tests/gemm/basic/test_gemm_afp4wfp4.py]=25
    FILE_TIMES[op_tests/triton_tests/gemm/basic/test_gemm_a8w8_per_token_scale.py]=25
    FILE_TIMES[op_tests/triton_tests/moe/test_moe_gemm_a4w4.py]=25
    FILE_TIMES[op_tests/triton_tests/moe/test_moe_gemm_a8w4.py]=25
    FILE_TIMES[op_tests/triton_tests/attention/test_mla_decode_rope.py]=20
    FILE_TIMES[op_tests/triton_tests/gemm/batched/test_batched_gemm_a8w8.py]=20
    FILE_TIMES[op_tests/triton_tests/gemm/batched/test_batched_gemm_afp4wfp4.py]=20
    FILE_TIMES[op_tests/triton_tests/gemm/batched/test_batched_gemm_a16wfp4.py]=20
    FILE_TIMES[op_tests/triton_tests/gemm/batched/test_batched_gemm_bf16.py]=20
fi

get_time() {
    local abs="$1"
    local rel="${abs#${TEST_DIR}/}"
    if [[ -n "${FILE_TIMES[$rel]+x}" ]]; then
        echo "${FILE_TIMES[$rel]}"
    else
        echo 15
    fi
}

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