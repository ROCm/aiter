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
    FILE_TIMES[op_tests/test_fused_qk_norm_mrope_cache_quant.py]=1478
    FILE_TIMES[op_tests/test_gemm_a8w8_blockscale.py]=1442
    FILE_TIMES[op_tests/test_mla_persistent.py]=1213
    FILE_TIMES[op_tests/test_batch_prefill.py]=1016
    FILE_TIMES[op_tests/test_mla.py]=954
    FILE_TIMES[op_tests/test_mha.py]=856
    FILE_TIMES[op_tests/test_mha_varlen.py]=756
    FILE_TIMES[op_tests/test_pa.py]=662
    FILE_TIMES[op_tests/test_mla_sparse.py]=602
    FILE_TIMES[op_tests/test_fused_qk_norm_rope_cache_quant.py]=574
    FILE_TIMES[op_tests/test_topk_plain.py]=436
    FILE_TIMES[op_tests/test_rope.py]=402
    FILE_TIMES[op_tests/test_topk_per_row.py]=362
    FILE_TIMES[op_tests/test_concat_cache_mla.py]=253
    FILE_TIMES[op_tests/test_gemm_a8w8.py]=202
    FILE_TIMES[op_tests/test_moe_dp_share_expert.py]=190
    FILE_TIMES[op_tests/test_moe_2stage.py]=171
    FILE_TIMES[op_tests/test_pa_mtp.py]=161
    FILE_TIMES[op_tests/test_moe_sorting.py]=157
    FILE_TIMES[op_tests/test_kvcache.py]=66
    FILE_TIMES[op_tests/test_aiter_add.py]=58
    FILE_TIMES[op_tests/test_pa_ps.py]=58
    FILE_TIMES[op_tests/test_quant.py]=58
    FILE_TIMES[op_tests/test_aiter_addInp.py]=57
    FILE_TIMES[op_tests/test_jit_dir_with_enum.py]=54
    FILE_TIMES[op_tests/test_activation.py]=52
    FILE_TIMES[op_tests/test_causal_conv1d.py]=47
    FILE_TIMES[op_tests/test_batched_gemm_bf16.py]=42
    FILE_TIMES[op_tests/test_mhc.py]=39
    FILE_TIMES[op_tests/test_pa_ragged.py]=38
    FILE_TIMES[op_tests/test_batched_gemm_a8w8.py]=36
    FILE_TIMES[op_tests/test_sampling.py]=30
    FILE_TIMES[op_tests/test_groupnorm.py]=28
    FILE_TIMES[op_tests/test_pa_v1.py]=28
    FILE_TIMES[op_tests/test_kvcache_blockscale.py]=26
    FILE_TIMES[op_tests/test_moeTopkSoftmax.py]=26
    FILE_TIMES[op_tests/test_moe_sorting_mxfp4.py]=26
    FILE_TIMES[op_tests/test_rmsnorm2d.py]=26
    FILE_TIMES[op_tests/test_split_gdr_update.py]=26
    FILE_TIMES[op_tests/test_moe_blockscale.py]=25
    FILE_TIMES[op_tests/test_pa_ragged_experimental.py]=24
    FILE_TIMES[op_tests/test_sample.py]=23
    FILE_TIMES[op_tests/test_gemm_a8w8_blockscale_cktile_aq_rowmajor.py]=20
    FILE_TIMES[op_tests/test_gemm_codegen.py]=20
    FILE_TIMES[op_tests/test_gated_rmsnorm_fp8_group_quant.py]=18
    FILE_TIMES[op_tests/test_fused_qk_norm.py]=17
    FILE_TIMES[op_tests/test_moe_tkw1.py]=17
    FILE_TIMES[op_tests/test_deepgemm.py]=16
    FILE_TIMES[op_tests/test_layernorm2dFusedAddQuant.py]=16
    FILE_TIMES[op_tests/test_gemm_a16w16.py]=14
    FILE_TIMES[op_tests/test_moe.py]=13
    FILE_TIMES[op_tests/test_moe_ep.py]=10
    FILE_TIMES[op_tests/test_smoothquant.py]=9
    FILE_TIMES[op_tests/test_rmsnorm2dFusedAddQuant.py]=8
    FILE_TIMES[op_tests/test_fused_qk_rmsnorm_group_quant.py]=6
    FILE_TIMES[op_tests/test_moe_topk_sigmoid.py]=6
    FILE_TIMES[op_tests/test_indexer_k_quant_and_cache.py]=5
    FILE_TIMES[op_tests/test_aiter_sigmoid.py]=4
    FILE_TIMES[op_tests/test_fused_qk_rmsnorm_per_token_quant.py]=4
    FILE_TIMES[op_tests/test_gemm_a4w4.py]=4
    FILE_TIMES[op_tests/test_layernorm2d.py]=4
    FILE_TIMES[op_tests/test_mha_fp8.py]=4
    FILE_TIMES[op_tests/test_mha_varlen_fp8.py]=4
    FILE_TIMES[op_tests/test_mla_prefill_ps.py]=4
    FILE_TIMES[op_tests/test_rotate_fp4quant.py]=4
    FILE_TIMES[op_tests/test_topk_row_prefill.py]=4
    FILE_TIMES[op_tests/test_pretune.py]=1
elif [[ "$TEST_TYPE" == "triton" ]]; then
    echo "Triton test files:"
    FILE_TIMES[op_tests/triton_tests/test_causal_conv1d.py]=773
    FILE_TIMES[op_tests/triton_tests/test_pa_decode_gluon.py]=717
    FILE_TIMES[op_tests/triton_tests/gemm/batched/test_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py]=328
    FILE_TIMES[op_tests/triton_tests/attention/test_mha_fp8.py]=297
    FILE_TIMES[op_tests/triton_tests/gemm/fused/test_fused_gemm_afp4wfp4_a16w16.py]=272
    FILE_TIMES[op_tests/triton_tests/test_gated_delta_rule.py]=258
    FILE_TIMES[op_tests/triton_tests/attention/test_mha_with_pe.py]=224
    FILE_TIMES[op_tests/triton_tests/attention/test_mha.py]=168
    FILE_TIMES[op_tests/triton_tests/gemm/basic/test_gemm_a8w8.py]=148
    FILE_TIMES[op_tests/triton_tests/attention/test_pa_decode.py]=140
    FILE_TIMES[op_tests/triton_tests/rope/test_rope.py]=136
    FILE_TIMES[op_tests/triton_tests/moe/test_moe.py]=134
    FILE_TIMES[op_tests/triton_tests/moe/test_moe_gemm_a8w8.py]=125
    FILE_TIMES[op_tests/triton_tests/gemm/fused/test_fused_gemm_afp4wfp4_mul_add.py]=118
    FILE_TIMES[op_tests/triton_tests/moe/test_moe_gemm_a8w8_blockscale.py]=115
    FILE_TIMES[op_tests/triton_tests/gemm/batched/test_batched_gemm_afp4wfp4.py]=114
    FILE_TIMES[op_tests/triton_tests/moe/test_moe_gemm_a4w4.py]=111
    FILE_TIMES[op_tests/triton_tests/fusions/test_fused_kv_cache.py]=106
    FILE_TIMES[op_tests/triton_tests/quant/test_fused_mxfp4_quant.py]=105
    FILE_TIMES[op_tests/triton_tests/gemm/basic/test_gemm_a16w16_gated.py]=98
    FILE_TIMES[op_tests/triton_tests/gemm/feed_forward/test_ff_a16w16_fused.py]=96
    FILE_TIMES[op_tests/triton_tests/gemm/fused/test_fused_gemm_a8w8_blockscale_a16w16.py]=94
    FILE_TIMES[op_tests/triton_tests/gemm/feed_forward/test_ff_a16w16.py]=92
    FILE_TIMES[op_tests/triton_tests/normalization/test_layernorm.py]=91
    FILE_TIMES[op_tests/triton_tests/attention/test_fav3_sage.py]=90
    FILE_TIMES[op_tests/triton_tests/moe/test_moe_gemm_a16w4.py]=84
    FILE_TIMES[op_tests/triton_tests/attention/test_mha_with_sink.py]=82
    FILE_TIMES[op_tests/triton_tests/moe/test_moe_routing.py]=81
    FILE_TIMES[op_tests/triton_tests/normalization/test_rmsnorm.py]=74
    FILE_TIMES[op_tests/triton_tests/test_gmm.py]=67
    FILE_TIMES[op_tests/triton_tests/attention/test_flash_attn_kvcache.py]=66
    FILE_TIMES[op_tests/triton_tests/fusions/test_fused_bmm_rope_kv_cache.py]=66
    FILE_TIMES[op_tests/triton_tests/moe/test_moe_gemm_a8w4.py]=64
    FILE_TIMES[op_tests/triton_tests/attention/test_la.py]=54
    FILE_TIMES[op_tests/triton_tests/attention/test_unified_attention.py]=52
    FILE_TIMES[op_tests/triton_tests/gemm/fused/test_fused_gemm_afp4wfp4_split_cat.py]=50
    FILE_TIMES[op_tests/triton_tests/moe/test_moe_gemm_int8_smoothquant.py]=48
    FILE_TIMES[op_tests/triton_tests/rope/test_fused_qkv_split_qk_rope.py]=48
    FILE_TIMES[op_tests/triton_tests/test_activation.py]=48
    FILE_TIMES[op_tests/triton_tests/quant/test_fused_fp8_quant.py]=46
    FILE_TIMES[op_tests/triton_tests/attention/test_la_paged.py]=40
    FILE_TIMES[op_tests/triton_tests/gemm/basic/test_gemm_a16w16.py]=32
    FILE_TIMES[op_tests/triton_tests/attention/test_chunked_pa_prefill.py]=31
    FILE_TIMES[op_tests/triton_tests/attention/test_pa_prefill.py]=31
    FILE_TIMES[op_tests/triton_tests/gemm/fused/test_fused_gemm_a8w8_blockscale_mul_add.py]=31
    FILE_TIMES[op_tests/triton_tests/attention/test_mla_decode_rope.py]=30
    FILE_TIMES[op_tests/triton_tests/attention/test_unified_attention_sparse_mla.py]=23
    FILE_TIMES[op_tests/triton_tests/gemm/basic/test_gemm_a16w8_blockscale.py]=22
    FILE_TIMES[op_tests/triton_tests/attention/test_mha_dao_ai.py]=20
    FILE_TIMES[op_tests/triton_tests/test_gather_kv_b_proj.py]=20
    FILE_TIMES[op_tests/triton_tests/gemm/basic/test_gemm_a8w8_per_token_scale.py]=16
    FILE_TIMES[op_tests/triton_tests/gemm/basic/test_gemm_afp4wfp4.py]=16
    FILE_TIMES[op_tests/triton_tests/gemm/batched/test_batched_gemm_a16wfp4.py]=16
    FILE_TIMES[op_tests/triton_tests/gemm/batched/test_batched_gemm_a8w8.py]=16
    FILE_TIMES[op_tests/triton_tests/gemm/batched/test_batched_gemm_bf16.py]=15
    FILE_TIMES[op_tests/triton_tests/test_fused_rearrange_sigmoid_gdr.py]=15
    FILE_TIMES[op_tests/triton_tests/torch_compile/test_compile_gemm_a16w16.py]=13
    FILE_TIMES[op_tests/triton_tests/moe/test_moe_mx.py]=12
    FILE_TIMES[op_tests/triton_tests/attention/test_fav3_sage_compile.py]=11
    FILE_TIMES[op_tests/triton_tests/gemm/basic/test_gemm_a8w8_blockscale.py]=11
    FILE_TIMES[op_tests/triton_tests/gemm/basic/test_gemm_a16wfp4.py]=10
    FILE_TIMES[op_tests/triton_tests/gemm/fused/test_fused_gemm_a8w8_blockscale_split_cat.py]=9
    FILE_TIMES[op_tests/triton_tests/fusions/test_fused_qk_concat.py]=8
    FILE_TIMES[op_tests/triton_tests/normalization/test_fused_add_rmsnorm_pad.py]=7
    FILE_TIMES[op_tests/triton_tests/quant/test_fused_rms_gated_fp8_group_quant.py]=7
    FILE_TIMES[op_tests/triton_tests/attention/test_extend_attention.py]=6
    FILE_TIMES[op_tests/triton_tests/attention/test_prefill_attention.py]=6
    FILE_TIMES[op_tests/triton_tests/fusions/test_fused_mul_add.py]=6
    FILE_TIMES[op_tests/triton_tests/gemm/basic/test_gemm_a8wfp4.py]=6
    FILE_TIMES[op_tests/triton_tests/test_causal_conv1d_update_single_token.py]=6
    FILE_TIMES[op_tests/triton_tests/test_topk.py]=6
    FILE_TIMES[op_tests/triton_tests/torch_compile/test_compile_moe_routing.py]=6
    FILE_TIMES[op_tests/triton_tests/torch_compile/test_compile_rope.py]=6
    FILE_TIMES[op_tests/triton_tests/attention/test_fp8_mqa_logits.py]=5
    FILE_TIMES[op_tests/triton_tests/quant/test_quant_mxfp4.py]=5
    FILE_TIMES[op_tests/triton_tests/torch_compile/test_compile_softmax.py]=5
    FILE_TIMES[op_tests/triton_tests/quant/test_quant.py]=4
    FILE_TIMES[op_tests/triton_tests/torch_compile/test_compile_activation.py]=4
    FILE_TIMES[op_tests/triton_tests/torch_compile/test_compile_fused_mul_add.py]=4
    FILE_TIMES[op_tests/triton_tests/torch_compile/test_compile_quant_per_token.py]=4
    FILE_TIMES[op_tests/triton_tests/moe/test_moe_routing_sigmoid_top1_fused.py]=3
    FILE_TIMES[op_tests/triton_tests/fusions/test_fused_routing_from_topk.py]=2
    FILE_TIMES[op_tests/triton_tests/test_softmax.py]=2
    FILE_TIMES[op_tests/triton_tests/torch_compile/test_compile_quant_per_tensor.py]=2
    FILE_TIMES[op_tests/triton_tests/torch_compile/test_compile_rmsnorm.py]=2
    FILE_TIMES[op_tests/triton_tests/attention/test_hstu_attn.py]=1
    FILE_TIMES[op_tests/triton_tests/moe/test_moe_align_block_size.py]=1
    FILE_TIMES[op_tests/triton_tests/torch_compile/test_compile_topk.py]=1
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