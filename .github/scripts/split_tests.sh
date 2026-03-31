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
    FILE_TIMES[op_tests/test_fused_qk_norm_mrope_cache_quant.py]=1463
    FILE_TIMES[op_tests/test_mla.py]=921
    FILE_TIMES[op_tests/test_mha.py]=792
    FILE_TIMES[op_tests/test_mla_persistent.py]=763
    FILE_TIMES[op_tests/test_mha_varlen.py]=678
    FILE_TIMES[op_tests/test_mla_sparse.py]=589
    FILE_TIMES[op_tests/test_batch_prefill.py]=572
    FILE_TIMES[op_tests/test_pa.py]=566
    FILE_TIMES[op_tests/test_moe_2stage.py]=525
    FILE_TIMES[op_tests/test_fused_qk_norm_rope_cache_quant.py]=476
    FILE_TIMES[op_tests/test_rope.py]=412
    FILE_TIMES[op_tests/test_gemm_a8w8_blockscale.py]=394
    FILE_TIMES[op_tests/test_topk_plain.py]=299
    FILE_TIMES[op_tests/test_concat_cache_mla.py]=225
    FILE_TIMES[op_tests/test_moe_sorting_mxfp4.py]=165
    FILE_TIMES[op_tests/test_moe_sorting.py]=128
    FILE_TIMES[op_tests/test_moe_dp_share_expert.py]=116
    FILE_TIMES[op_tests/test_topk_per_row.py]=116
    FILE_TIMES[op_tests/test_moe_tkw1.py]=104
    FILE_TIMES[op_tests/test_pa_mtp.py]=98
    FILE_TIMES[op_tests/test_gemm_a8w8.py]=77
    FILE_TIMES[op_tests/test_kvcache.py]=54
    FILE_TIMES[op_tests/test_causal_conv1d.py]=53
    FILE_TIMES[op_tests/test_quant.py]=50
    FILE_TIMES[op_tests/test_activation.py]=43
    FILE_TIMES[op_tests/test_pa_ps.py]=38
    FILE_TIMES[op_tests/test_jit_dir_with_enum.py]=36
    FILE_TIMES[op_tests/test_aiter_addInp.py]=34
    FILE_TIMES[op_tests/test_aiter_add.py]=33
    FILE_TIMES[op_tests/test_batched_gemm_bf16.py]=31
    FILE_TIMES[op_tests/test_mhc.py]=29
    FILE_TIMES[op_tests/test_batched_gemm_a8w8.py]=27
    FILE_TIMES[op_tests/test_gemm_a4w4.py]=27
    FILE_TIMES[op_tests/test_mha_fp8.py]=27
    FILE_TIMES[op_tests/test_groupnorm.py]=25
    FILE_TIMES[op_tests/test_split_gdr_update.py]=25
    FILE_TIMES[op_tests/test_kvcache_blockscale.py]=23
    FILE_TIMES[op_tests/test_pa_ragged.py]=23
    FILE_TIMES[op_tests/test_sampling.py]=22
    FILE_TIMES[op_tests/test_moe_blockscale.py]=19
    FILE_TIMES[op_tests/test_rmsnorm2d.py]=18
    FILE_TIMES[op_tests/test_mla_prefill_ps.py]=17
    FILE_TIMES[op_tests/test_pa_ragged_experimental.py]=17
    FILE_TIMES[op_tests/test_sample.py]=17
    FILE_TIMES[op_tests/test_mha_varlen_fp8.py]=16
    FILE_TIMES[op_tests/test_moeTopkSoftmax.py]=16
    FILE_TIMES[op_tests/test_pa_v1.py]=16
    FILE_TIMES[op_tests/test_fused_qk_norm.py]=12
    FILE_TIMES[op_tests/test_layernorm2dFusedAddQuant.py]=11
    FILE_TIMES[op_tests/test_gemm_a16w16.py]=9
    FILE_TIMES[op_tests/test_moe.py]=9
    FILE_TIMES[op_tests/test_moe_ep.py]=7
    FILE_TIMES[op_tests/test_smoothquant.py]=6
    FILE_TIMES[op_tests/test_rmsnorm2dFusedAddQuant.py]=5
    FILE_TIMES[op_tests/test_indexer_k_quant_and_cache.py]=4
    FILE_TIMES[op_tests/test_topk_row_prefill.py]=4
    FILE_TIMES[op_tests/test_aiter_sigmoid.py]=3
    FILE_TIMES[op_tests/test_deepgemm.py]=3
    FILE_TIMES[op_tests/test_layernorm2d.py]=3
    FILE_TIMES[op_tests/test_moe_topk_sigmoid.py]=3
elif [[ "$TEST_TYPE" == "triton" ]]; then
    echo "Triton test files:"
    FILE_TIMES[op_tests/triton_tests/attention/test_mha.py]=1484
    FILE_TIMES[op_tests/triton_tests/rope/test_rope.py]=1462
    FILE_TIMES[op_tests/triton_tests/test_pa_decode_gluon.py]=722
    FILE_TIMES[op_tests/triton_tests/test_causal_conv1d.py]=653
    FILE_TIMES[op_tests/triton_tests/attention/test_pa_decode.py]=620
    FILE_TIMES[op_tests/triton_tests/gemm/batched/test_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py]=390
    FILE_TIMES[op_tests/triton_tests/test_gmm.py]=350
    FILE_TIMES[op_tests/triton_tests/attention/test_chunked_pa_prefill.py]=339
    FILE_TIMES[op_tests/triton_tests/gemm/basic/test_gemm_a8w8.py]=338
    FILE_TIMES[op_tests/triton_tests/attention/test_pa_prefill.py]=319
    FILE_TIMES[op_tests/triton_tests/test_gated_delta_rule.py]=319
    FILE_TIMES[op_tests/triton_tests/moe/test_moe_gemm_a8w8_blockscale.py]=260
    FILE_TIMES[op_tests/triton_tests/attention/test_flash_attn_kvcache.py]=257
    FILE_TIMES[op_tests/triton_tests/attention/test_unified_attention.py]=212
    FILE_TIMES[op_tests/triton_tests/gemm/basic/test_gemm_a16w16.py]=208
    FILE_TIMES[op_tests/triton_tests/gemm/basic/test_gemm_a16w16_gated.py]=184
    FILE_TIMES[op_tests/triton_tests/gemm/feed_forward/test_ff_a16w16.py]=158
    FILE_TIMES[op_tests/triton_tests/rope/test_fused_qkv_split_qk_rope.py]=142
    FILE_TIMES[op_tests/triton_tests/moe/test_moe.py]=125
    FILE_TIMES[op_tests/triton_tests/gemm/feed_forward/test_ff_a16w16_fused.py]=99
    FILE_TIMES[op_tests/triton_tests/normalization/test_rmsnorm.py]=73
    FILE_TIMES[op_tests/triton_tests/gemm/fused/test_fused_gemm_a8w8_blockscale_a16w16.py]=72
    FILE_TIMES[op_tests/triton_tests/normalization/test_layernorm.py]=70
    FILE_TIMES[op_tests/triton_tests/fusions/test_fused_kv_cache.py]=60
    FILE_TIMES[op_tests/triton_tests/moe/test_moe_gemm_int8_smoothquant.py]=54
    FILE_TIMES[op_tests/triton_tests/quant/test_fused_fp8_quant.py]=50
    FILE_TIMES[op_tests/triton_tests/attention/test_la.py]=49
    FILE_TIMES[op_tests/triton_tests/gemm/basic/test_gemm_a8w8_per_token_scale.py]=37
    FILE_TIMES[op_tests/triton_tests/gemm/basic/test_gemm_a16w8_blockscale.py]=36
    FILE_TIMES[op_tests/triton_tests/gemm/fused/test_fused_gemm_a8w8_blockscale_mul_add.py]=36
    FILE_TIMES[op_tests/triton_tests/attention/test_unified_attention_sparse_mla.py]=32
    FILE_TIMES[op_tests/triton_tests/attention/test_la_paged.py]=30
    FILE_TIMES[op_tests/triton_tests/attention/test_mla_decode_rope.py]=27
    FILE_TIMES[op_tests/triton_tests/fusions/test_fused_bmm_rope_kv_cache.py]=25
    FILE_TIMES[op_tests/triton_tests/attention/test_fav3_sage.py]=22
    FILE_TIMES[op_tests/triton_tests/gemm/basic/test_gemm_a8w8_blockscale.py]=22
    FILE_TIMES[op_tests/triton_tests/gemm/batched/test_batched_gemm_a8w8.py]=20
    FILE_TIMES[op_tests/triton_tests/gemm/batched/test_batched_gemm_bf16.py]=20
    FILE_TIMES[op_tests/triton_tests/attention/test_extend_attention.py]=7
    FILE_TIMES[op_tests/triton_tests/fusions/test_fused_mul_add.py]=6
    FILE_TIMES[op_tests/triton_tests/normalization/test_fused_add_rmsnorm_pad.py]=6
    FILE_TIMES[op_tests/triton_tests/attention/test_prefill_attention.py]=5
    FILE_TIMES[op_tests/triton_tests/fusions/test_fused_qk_concat.py]=5
    FILE_TIMES[op_tests/triton_tests/test_topk.py]=5
    FILE_TIMES[op_tests/triton_tests/attention/test_fp8_mqa_logits.py]=4
    FILE_TIMES[op_tests/triton_tests/gemm/fused/test_fused_gemm_a8w8_blockscale_split_cat.py]=4
    FILE_TIMES[op_tests/triton_tests/quant/test_quant.py]=4
    FILE_TIMES[op_tests/triton_tests/quant/test_quant_mxfp4.py]=4
    FILE_TIMES[op_tests/triton_tests/test_gather_kv_b_proj.py]=3
    FILE_TIMES[op_tests/triton_tests/test_softmax.py]=2
    FILE_TIMES[op_tests/triton_tests/attention/test_hstu_attn.py]=1
    FILE_TIMES[op_tests/triton_tests/gemm/basic/test_gemm_a16wfp4.py]=1
    FILE_TIMES[op_tests/triton_tests/gemm/basic/test_gemm_a8wfp4.py]=1
    FILE_TIMES[op_tests/triton_tests/gemm/basic/test_gemm_afp4wfp4.py]=1
    FILE_TIMES[op_tests/triton_tests/gemm/batched/test_batched_gemm_a16wfp4.py]=1
    FILE_TIMES[op_tests/triton_tests/gemm/batched/test_batched_gemm_afp4wfp4.py]=1
    FILE_TIMES[op_tests/triton_tests/gemm/fused/test_fused_gemm_afp4wfp4_a16w16.py]=1
    FILE_TIMES[op_tests/triton_tests/gemm/fused/test_fused_gemm_afp4wfp4_mul_add.py]=1
    FILE_TIMES[op_tests/triton_tests/gemm/fused/test_fused_gemm_afp4wfp4_split_cat.py]=1
    FILE_TIMES[op_tests/triton_tests/moe/test_moe_align_block_size.py]=1
    FILE_TIMES[op_tests/triton_tests/moe/test_moe_gemm_a4w4.py]=1
    FILE_TIMES[op_tests/triton_tests/moe/test_moe_gemm_a8w4.py]=1
    FILE_TIMES[op_tests/triton_tests/moe/test_moe_gemm_a8w8.py]=1
    FILE_TIMES[op_tests/triton_tests/moe/test_moe_mx.py]=1
    FILE_TIMES[op_tests/triton_tests/moe/test_moe_routing.py]=1
    FILE_TIMES[op_tests/triton_tests/moe/test_moe_routing_sigmoid_top1_fused.py]=1
    FILE_TIMES[op_tests/triton_tests/quant/test_fused_mxfp4_quant.py]=1
    FILE_TIMES[op_tests/triton_tests/test_activation.py]=1
    FILE_TIMES[op_tests/triton_tests/triton_metadata_redirect/test_metadata_redirect.py]=1
fi

get_time() {
    local abs="$1"
    # FILE_TIMES keys use full path (e.g. op_tests/test_mla.py), so look up with abs
    if [[ -n "${FILE_TIMES[$abs]+x}" ]]; then
        echo "${FILE_TIMES[$abs]}"
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