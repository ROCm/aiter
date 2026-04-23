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
    FILE_TIMES[op_tests/test_fused_qk_norm_mrope_cache_quant.py]=1458
    FILE_TIMES[op_tests/test_mla.py]=840
    FILE_TIMES[op_tests/test_mla_persistent.py]=813
    FILE_TIMES[op_tests/test_mha.py]=728
    FILE_TIMES[op_tests/test_pa.py]=614
    FILE_TIMES[op_tests/test_batch_prefill.py]=597
    FILE_TIMES[op_tests/test_mla_sparse.py]=583
    FILE_TIMES[op_tests/test_mha_varlen.py]=577
    FILE_TIMES[op_tests/test_fused_qk_norm_rope_cache_quant.py]=486
    FILE_TIMES[op_tests/test_topk_plain.py]=481
    FILE_TIMES[op_tests/test_rope.py]=392
    FILE_TIMES[op_tests/test_gemm_a8w8_blockscale.py]=285
    FILE_TIMES[op_tests/test_concat_cache_mla.py]=249
    FILE_TIMES[op_tests/test_moe_sorting.py]=130
    FILE_TIMES[op_tests/test_moe_dp_share_expert.py]=120
    FILE_TIMES[op_tests/test_topk_per_row.py]=116
    FILE_TIMES[op_tests/test_moe_2stage.py]=112
    FILE_TIMES[op_tests/test_pa_mtp.py]=104
    FILE_TIMES[op_tests/test_gemm_a8w8.py]=84
    FILE_TIMES[op_tests/test_causal_conv1d.py]=59
    FILE_TIMES[op_tests/test_kvcache.py]=57
    FILE_TIMES[op_tests/test_quant.py]=51
    FILE_TIMES[op_tests/test_activation.py]=44
    FILE_TIMES[op_tests/test_pa_ps.py]=42
    FILE_TIMES[op_tests/test_jit_dir_with_enum.py]=40
    FILE_TIMES[op_tests/test_aiter_add.py]=37
    FILE_TIMES[op_tests/test_mhc.py]=37
    FILE_TIMES[op_tests/test_aiter_addInp.py]=36
    FILE_TIMES[op_tests/test_batched_gemm_bf16.py]=30
    FILE_TIMES[op_tests/test_batched_gemm_a8w8.py]=28
    FILE_TIMES[op_tests/test_gemm_a4w4.py]=28
    FILE_TIMES[op_tests/test_groupnorm.py]=28
    FILE_TIMES[op_tests/test_moe_sorting_mxfp4.py]=28
    FILE_TIMES[op_tests/test_split_gdr_update.py]=28
    FILE_TIMES[op_tests/test_kvcache_blockscale.py]=24
    FILE_TIMES[op_tests/test_pa_ragged.py]=24
    FILE_TIMES[op_tests/test_sampling.py]=22
    FILE_TIMES[op_tests/test_moe_blockscale.py]=20
    FILE_TIMES[op_tests/test_pa_ragged_experimental.py]=19
    FILE_TIMES[op_tests/test_rmsnorm2d.py]=19
    FILE_TIMES[op_tests/test_mha_fp8.py]=18
    FILE_TIMES[op_tests/test_mla_prefill_ps.py]=18
    FILE_TIMES[op_tests/test_pa_v1.py]=18
    FILE_TIMES[op_tests/test_sample.py]=18
    FILE_TIMES[op_tests/test_mha_varlen_fp8.py]=17
    FILE_TIMES[op_tests/test_moeTopkSoftmax.py]=16
    FILE_TIMES[op_tests/test_gated_rmsnorm_fp8_group_quant.py]=15
    FILE_TIMES[op_tests/test_fused_qk_norm.py]=14
    FILE_TIMES[op_tests/test_gemm_codegen.py]=14
    FILE_TIMES[op_tests/test_moe_tkw1.py]=14
    FILE_TIMES[op_tests/test_layernorm2dFusedAddQuant.py]=11
    FILE_TIMES[op_tests/test_deepgemm.py]=10
    FILE_TIMES[op_tests/test_gemm_a16w16.py]=10
    FILE_TIMES[op_tests/test_moe.py]=10
    FILE_TIMES[op_tests/test_moe_ep.py]=8
    FILE_TIMES[op_tests/test_rmsnorm2dFusedAddQuant.py]=6
    FILE_TIMES[op_tests/test_smoothquant.py]=6
    FILE_TIMES[op_tests/test_fused_qk_rmsnorm_group_quant.py]=5
    FILE_TIMES[op_tests/test_aiter_sigmoid.py]=4
    FILE_TIMES[op_tests/test_indexer_k_quant_and_cache.py]=4
    FILE_TIMES[op_tests/test_mha_varlen_fp8.py]=4
    FILE_TIMES[op_tests/test_mla_prefill_ps.py]=4
    FILE_TIMES[op_tests/test_moe_topk_sigmoid.py]=4
    FILE_TIMES[op_tests/test_topk_row_prefill.py]=4
    FILE_TIMES[op_tests/test_layernorm2d.py]=3
    FILE_TIMES[op_tests/test_pretune.py]=1
elif [[ "$TEST_TYPE" == "triton" ]]; then
    echo "Triton test files:"
    FILE_TIMES[op_tests/triton_tests/attention/test_mha.py]=1452
    FILE_TIMES[op_tests/triton_tests/test_pa_decode_gluon.py]=718
    FILE_TIMES[op_tests/triton_tests/attention/test_pa_decode.py]=635
    FILE_TIMES[op_tests/triton_tests/test_causal_conv1d.py]=634
    FILE_TIMES[op_tests/triton_tests/gemm/batched/test_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py]=402
    FILE_TIMES[op_tests/triton_tests/attention/test_flash_attn_kvcache.py]=357
    FILE_TIMES[op_tests/triton_tests/attention/test_chunked_pa_prefill.py]=336
    FILE_TIMES[op_tests/triton_tests/attention/test_pa_prefill.py]=313
    FILE_TIMES[op_tests/triton_tests/test_gated_delta_rule.py]=303
    FILE_TIMES[op_tests/triton_tests/attention/test_unified_attention.py]=276
    FILE_TIMES[op_tests/triton_tests/moe/test_moe_gemm_a8w8_blockscale.py]=263
    FILE_TIMES[op_tests/triton_tests/moe/test_moe.py]=133
    FILE_TIMES[op_tests/triton_tests/rope/test_rope.py]=126
    FILE_TIMES[op_tests/triton_tests/gemm/basic/test_gemm_a16w16_gated.py]=92
    FILE_TIMES[op_tests/triton_tests/gemm/feed_forward/test_ff_a16w16.py]=84
    FILE_TIMES[op_tests/triton_tests/attention/test_mha_with_sink.py]=78
    FILE_TIMES[op_tests/triton_tests/gemm/basic/test_gemm_a8w8.py]=69
    FILE_TIMES[op_tests/triton_tests/gemm/fused/test_fused_gemm_a8w8_blockscale_a16w16.py]=69
    FILE_TIMES[op_tests/triton_tests/normalization/test_layernorm.py]=69
    FILE_TIMES[op_tests/triton_tests/normalization/test_rmsnorm.py]=66
    FILE_TIMES[op_tests/triton_tests/gemm/feed_forward/test_ff_a16w16_fused.py]=65
    FILE_TIMES[op_tests/triton_tests/test_gmm.py]=58
    FILE_TIMES[op_tests/triton_tests/fusions/test_fused_kv_cache.py]=56
    FILE_TIMES[op_tests/triton_tests/rope/test_fused_qkv_split_qk_rope.py]=51
    FILE_TIMES[op_tests/triton_tests/moe/test_moe_gemm_int8_smoothquant.py]=50
    FILE_TIMES[op_tests/triton_tests/attention/test_la.py]=48
    FILE_TIMES[op_tests/triton_tests/gemm/basic/test_gemm_a16w16.py]=40
    FILE_TIMES[op_tests/triton_tests/attention/test_la_paged.py]=32
    FILE_TIMES[op_tests/triton_tests/attention/test_unified_attention_sparse_mla.py]=31
    FILE_TIMES[op_tests/triton_tests/gemm/fused/test_fused_gemm_a8w8_blockscale_mul_add.py]=31
    FILE_TIMES[op_tests/triton_tests/attention/test_fav3_sage.py]=26
    FILE_TIMES[op_tests/triton_tests/attention/test_mla_decode_rope.py]=25
    FILE_TIMES[op_tests/triton_tests/fusions/test_fused_bmm_rope_kv_cache.py]=24
    FILE_TIMES[op_tests/triton_tests/gemm/basic/test_gemm_a16w8_blockscale.py]=20
    FILE_TIMES[op_tests/triton_tests/gemm/batched/test_batched_gemm_bf16.py]=20
    FILE_TIMES[op_tests/triton_tests/gemm/batched/test_batched_gemm_a8w8.py]=19
    FILE_TIMES[op_tests/triton_tests/gemm/basic/test_gemm_a8w8_per_token_scale.py]=17
    FILE_TIMES[op_tests/triton_tests/quant/test_fused_fp8_quant.py]=17
    FILE_TIMES[op_tests/triton_tests/test_gather_kv_b_proj.py]=16
    FILE_TIMES[op_tests/triton_tests/attention/test_extend_attention.py]=7
    FILE_TIMES[op_tests/triton_tests/fusions/test_fused_qk_concat.py]=7
    FILE_TIMES[op_tests/triton_tests/gemm/basic/test_gemm_a8w8_blockscale.py]=7
    FILE_TIMES[op_tests/triton_tests/attention/test_fp8_mqa_logits.py]=6
    FILE_TIMES[op_tests/triton_tests/attention/test_mha_dao_ai.py]=6
    FILE_TIMES[op_tests/triton_tests/fusions/test_fused_mul_add.py]=5
    FILE_TIMES[op_tests/triton_tests/normalization/test_fused_add_rmsnorm_pad.py]=5
    FILE_TIMES[op_tests/triton_tests/test_topk.py]=5
    FILE_TIMES[op_tests/triton_tests/gemm/fused/test_fused_gemm_a8w8_blockscale_split_cat.py]=4
    FILE_TIMES[op_tests/triton_tests/quant/test_quant_mxfp4.py]=4
    FILE_TIMES[op_tests/triton_tests/quant/test_quant.py]=3
    FILE_TIMES[op_tests/triton_tests/attention/test_prefill_attention.py]=2
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
    if [[ "$TEST_TYPE" == "aiter" ]] && [[ "${AITER_SKIP_PERF:-0}" == "1" ]] && [[ -n "${FAST_FILE_TIMES[$abs]+x}" ]]; then
        echo "${FAST_FILE_TIMES[$abs]}"
        return
    fi
    # FILE_TIMES keys use full path (e.g. op_tests/test_mla.py), so look up with abs
    if [[ -n "${FILE_TIMES[$abs]+x}" ]]; then
        echo "${FILE_TIMES[$abs]}"
    else
        echo 15
    fi
}

if [[ "$TEST_TYPE" == "aiter" ]]; then
    # PR fast-path timings with AITER_SKIP_PERF=1.
    declare -A FAST_FILE_TIMES
    FAST_FILE_TIMES[op_tests/test_mla_persistent.py]=635
    FAST_FILE_TIMES[op_tests/test_batch_prefill.py]=600
    FAST_FILE_TIMES[op_tests/test_mha.py]=550
    FAST_FILE_TIMES[op_tests/test_gemm_a8w8_blockscale.py]=465
    FAST_FILE_TIMES[op_tests/test_mha_varlen.py]=417
    FAST_FILE_TIMES[op_tests/test_mla.py]=195
    FAST_FILE_TIMES[op_tests/test_topk_per_row.py]=106
    FAST_FILE_TIMES[op_tests/test_topk_plain.py]=86
    FAST_FILE_TIMES[op_tests/test_jit_dir_with_enum.py]=78
    FAST_FILE_TIMES[op_tests/test_pa_mtp.py]=73
    FAST_FILE_TIMES[op_tests/test_pa.py]=59
    FAST_FILE_TIMES[op_tests/test_mla_sparse.py]=45
    FAST_FILE_TIMES[op_tests/test_groupnorm.py]=37
    FAST_FILE_TIMES[op_tests/test_causal_conv1d.py]=33
    FAST_FILE_TIMES[op_tests/test_mhc.py]=33
    FAST_FILE_TIMES[op_tests/test_mha_varlen_fp8.py]=30
    FAST_FILE_TIMES[op_tests/test_mha_fp8.py]=30
    FAST_FILE_TIMES[op_tests/test_split_gdr_update.py]=27
    FAST_FILE_TIMES[op_tests/test_batched_gemm_bf16.py]=24
    FAST_FILE_TIMES[op_tests/test_pa_ps.py]=23
    FAST_FILE_TIMES[op_tests/test_sampling.py]=22
    FAST_FILE_TIMES[op_tests/test_pa_ragged.py]=22
    FAST_FILE_TIMES[op_tests/test_gemm_a4w4.py]=20
    FAST_FILE_TIMES[op_tests/test_pa_ragged_experimental.py]=19
    FAST_FILE_TIMES[op_tests/test_batched_gemm_a8w8.py]=18
    FAST_FILE_TIMES[op_tests/test_gated_rmsnorm_fp8_group_quant.py]=16
    FAST_FILE_TIMES[op_tests/test_pa_v1.py]=16
    FAST_FILE_TIMES[op_tests/test_fused_qk_norm.py]=15
    FAST_FILE_TIMES[op_tests/test_rope.py]=13
    FAST_FILE_TIMES[op_tests/test_fused_qk_norm_rope_cache_quant.py]=10
    FAST_FILE_TIMES[op_tests/test_gemm_a16w16.py]=10
    FAST_FILE_TIMES[op_tests/test_mla_prefill_ps.py]=10
    FAST_FILE_TIMES[op_tests/test_concat_cache_mla.py]=9
    FAST_FILE_TIMES[op_tests/test_moe_2stage.py]=9
    FAST_FILE_TIMES[op_tests/test_gemm_a8w8.py]=8
    FAST_FILE_TIMES[op_tests/test_kvcache.py]=8
    FAST_FILE_TIMES[op_tests/test_fused_qk_norm_mrope_cache_quant.py]=8
    FAST_FILE_TIMES[op_tests/test_deepgemm.py]=8
    FAST_FILE_TIMES[op_tests/test_quant.py]=7
    FAST_FILE_TIMES[op_tests/test_moe_dp_share_expert.py]=7
    FAST_FILE_TIMES[op_tests/test_moe_sorting_mxfp4.py]=6
    FAST_FILE_TIMES[op_tests/test_moe_sorting.py]=6
    FAST_FILE_TIMES[op_tests/test_moe_blockscale.py]=6
    FAST_FILE_TIMES[op_tests/test_moe_tkw1.py]=6
    FAST_FILE_TIMES[op_tests/test_moe.py]=6
    FAST_FILE_TIMES[op_tests/test_fused_qk_rmsnorm_group_quant.py]=6
    FAST_FILE_TIMES[op_tests/test_moe_ep.py]=5
    FAST_FILE_TIMES[op_tests/test_aiter_addInp.py]=5
    FAST_FILE_TIMES[op_tests/test_sample.py]=5
    FAST_FILE_TIMES[op_tests/test_activation.py]=5
    FAST_FILE_TIMES[op_tests/test_kvcache_blockscale.py]=5
    FAST_FILE_TIMES[op_tests/test_aiter_add.py]=5
    FAST_FILE_TIMES[op_tests/test_smoothquant.py]=4
    FAST_FILE_TIMES[op_tests/test_topk_row_prefill.py]=4
    FAST_FILE_TIMES[op_tests/test_indexer_k_quant_and_cache.py]=4
    FAST_FILE_TIMES[op_tests/test_moe_topk_sigmoid.py]=4
    FAST_FILE_TIMES[op_tests/test_moeTopkSoftmax.py]=3
    FAST_FILE_TIMES[op_tests/test_layernorm2dFusedAddQuant.py]=3
    FAST_FILE_TIMES[op_tests/test_rmsnorm2d.py]=3
    FAST_FILE_TIMES[op_tests/test_rmsnorm2dFusedAddQuant.py]=3
    FAST_FILE_TIMES[op_tests/test_aiter_sigmoid.py]=3
    FAST_FILE_TIMES[op_tests/test_layernorm2d.py]=3
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