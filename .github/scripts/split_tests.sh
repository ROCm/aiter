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
    FILE_TIMES[op_tests/test_mla_v4_nm.py]=944
    FILE_TIMES[op_tests/test_fused_qk_norm_mrope_cache_quant.py]=936
    FILE_TIMES[op_tests/test_fused_qk_norm_rope_cache_quant.py]=906
    FILE_TIMES[op_tests/test_mla.py]=828
    FILE_TIMES[op_tests/test_mla_persistent.py]=750
    FILE_TIMES[op_tests/test_pa.py]=634
    FILE_TIMES[op_tests/test_moe_topk_gating.py]=630
    FILE_TIMES[op_tests/test_batch_prefill.py]=610
    FILE_TIMES[op_tests/test_mla_persistent_round_robin.py]=596
    FILE_TIMES[op_tests/test_moe_2stage.py]=586
    FILE_TIMES[op_tests/test_mla_sparse.py]=568
    FILE_TIMES[op_tests/test_mha.py]=560
    FILE_TIMES[op_tests/test_mha_varlen.py]=504
    FILE_TIMES[op_tests/test_rope.py]=400
    FILE_TIMES[op_tests/test_gemm_a8w8.py]=270
    FILE_TIMES[op_tests/test_topk_per_row.py]=266
    FILE_TIMES[op_tests/test_concat_cache_mla.py]=252
    FILE_TIMES[op_tests/test_gated_delta_rule.py]=212
    FILE_TIMES[op_tests/test_pa_mtp.py]=120
    FILE_TIMES[op_tests/test_moe_dp_share_expert.py]=115
    FILE_TIMES[op_tests/test_activation.py]=86
    FILE_TIMES[op_tests/test_gemm_a8w8_blockscale_cktile_aq_rowmajor.py]=84
    FILE_TIMES[op_tests/test_flydsl_qk_norm_rope_quant.py]=82
    FILE_TIMES[op_tests/test_kvcache.py]=58
    FILE_TIMES[op_tests/test_topk_plain.py]=58
    FILE_TIMES[op_tests/test_quant.py]=52
    FILE_TIMES[op_tests/test_gemm_a8w8_blockscale.py]=51
    FILE_TIMES[op_tests/test_mhc.py]=50
    FILE_TIMES[op_tests/test_jit_dir_with_enum.py]=42
    FILE_TIMES[op_tests/test_pa_sparse_prefill_opus.py]=42
    FILE_TIMES[op_tests/test_pa_ps.py]=39
    FILE_TIMES[op_tests/test_rmsnorm2d.py]=39
    FILE_TIMES[op_tests/test_causal_conv1d_update.py]=36
    FILE_TIMES[op_tests/test_flydsl_compress_attn.py]=35
    FILE_TIMES[op_tests/test_moe_sorting_mxfp4.py]=34
    FILE_TIMES[op_tests/test_moe_sorting.py]=32
    FILE_TIMES[op_tests/test_batched_gemm_bf16.py]=30
    FILE_TIMES[op_tests/test_gemm_a4w4.py]=30
    FILE_TIMES[op_tests/test_batched_gemm_a8w8.py]=28
    FILE_TIMES[op_tests/test_pa_ragged.py]=27
    FILE_TIMES[op_tests/test_gated_rmsnorm_fp8_quant.py]=24
    FILE_TIMES[op_tests/test_kvcache_blockscale.py]=24
    FILE_TIMES[op_tests/test_sampling.py]=24
    FILE_TIMES[op_tests/test_moe_blockscale.py]=22
    FILE_TIMES[op_tests/test_aiter_add.py]=20
    FILE_TIMES[op_tests/test_aiter_addInp.py]=20
    FILE_TIMES[op_tests/test_mla_prefill_ps.py]=20
    FILE_TIMES[op_tests/test_mla_reduce.py]=20
    FILE_TIMES[op_tests/test_pa_ragged_experimental.py]=19
    FILE_TIMES[op_tests/test_fused_qk_norm_rope_group_quant.py]=18
    FILE_TIMES[op_tests/test_moeTopkSoftmax.py]=18
    FILE_TIMES[op_tests/test_pa_v1.py]=18
    FILE_TIMES[op_tests/test_sample.py]=17
    FILE_TIMES[op_tests/test_gemm_a16w16.py]=16
    FILE_TIMES[op_tests/test_moe_tkw1.py]=16
    FILE_TIMES[op_tests/test_fused_qk_norm.py]=15
    FILE_TIMES[op_tests/test_mla_stage2_merge.py]=14
    FILE_TIMES[op_tests/test_moe_ep.py]=14
    FILE_TIMES[op_tests/test_layernorm2dFusedAddQuant.py]=12
    FILE_TIMES[op_tests/test_causal_conv1d_prefill_split_qkv.py]=11
    FILE_TIMES[op_tests/test_moe.py]=11
    FILE_TIMES[op_tests/test_deepgemm.py]=8
    FILE_TIMES[op_tests/test_flydsl_pa_mqa_logits_fp4.py]=8
    FILE_TIMES[op_tests/test_quant_mxfp4.py]=8
    FILE_TIMES[op_tests/test_smoothquant.py]=7
    FILE_TIMES[op_tests/test_flydsl_pa_mqa_logits_fp4_prefill.py]=6
    FILE_TIMES[op_tests/test_fused_kv_norm_rope_group_quant.py]=6
    FILE_TIMES[op_tests/test_fused_qk_norm_rope_2way_perhead.py]=6
    FILE_TIMES[op_tests/test_fused_qk_rmsnorm_group_quant.py]=6
    FILE_TIMES[op_tests/test_mha_varlen_fp8.py]=6
    FILE_TIMES[op_tests/test_rmsnorm2dFusedAddQuant.py]=6
    FILE_TIMES[op_tests/test_flydsl_grouped_gemm_gfx1250.py]=5
    FILE_TIMES[op_tests/test_fused_qknorm_idxrqknorm.py]=5
    FILE_TIMES[op_tests/test_metadata.py]=5
    FILE_TIMES[op_tests/test_moe_local_expert_ids.py]=5
    FILE_TIMES[op_tests/test_opus_a16w16_gemm.py]=5
    FILE_TIMES[op_tests/test_pa_mqa_logits_offset.py]=5
    FILE_TIMES[op_tests/test_aiter_sigmoid.py]=4
    FILE_TIMES[op_tests/test_dsv4_rotate_quant.py]=4
    FILE_TIMES[op_tests/test_fused_qk_norm_rope_1way_perhead.py]=4
    FILE_TIMES[op_tests/test_fused_qk_rmsnorm_per_token_quant.py]=4
    FILE_TIMES[op_tests/test_gemm_codegen.py]=4
    FILE_TIMES[op_tests/test_groupnorm.py]=4
    FILE_TIMES[op_tests/test_indexer_k_quant_and_cache.py]=4
    FILE_TIMES[op_tests/test_mha_fp8.py]=4
    FILE_TIMES[op_tests/test_mla_decode_gate.py]=4
    FILE_TIMES[op_tests/test_mla_v40_persistent.py]=4
    FILE_TIMES[op_tests/test_pa_block_id_truncation.py]=4
    FILE_TIMES[op_tests/test_split_gdr_update.py]=4
    FILE_TIMES[op_tests/test_topk_row_prefill.py]=4
    FILE_TIMES[op_tests/test_f4gemm.py]=3
    FILE_TIMES[op_tests/test_fmha_fwd_mxfp8_asm.py]=3
    FILE_TIMES[op_tests/test_fmha_fwd_with_sink_asm.py]=3
    FILE_TIMES[op_tests/test_fmha_fwd_with_sink_varlen_asm.py]=3
    FILE_TIMES[op_tests/test_gemm_a8w8_bpreshuffle_pad_k.py]=3
    FILE_TIMES[op_tests/test_jit_arch_guard.py]=3
    FILE_TIMES[op_tests/test_layernorm2d.py]=3
    FILE_TIMES[op_tests/test_mha_flydsl_varlen.py]=3
    FILE_TIMES[op_tests/test_mla_decode_pagesize64.py]=3
    FILE_TIMES[op_tests/test_mla_v4_kargpreld.py]=3
    FILE_TIMES[op_tests/test_mxfp8fp4gemm.py]=3
    FILE_TIMES[op_tests/test_pa_decode_bf16_asm.py]=3
    FILE_TIMES[op_tests/test_pretune.py]=1
elif [[ "$TEST_TYPE" == "triton" ]]; then
    echo "Triton test files:"
    FILE_TIMES[op_tests/triton_tests/attention/test_mha_v3.py]=1330
    FILE_TIMES[op_tests/triton_tests/test_pa_decode_gluon.py]=748
    FILE_TIMES[op_tests/triton_tests/attention/test_mha_fp8.py]=732
    FILE_TIMES[op_tests/triton_tests/conv/test_causal_conv1d.py]=714
    FILE_TIMES[op_tests/triton_tests/gemm/batched/test_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py]=382
    FILE_TIMES[op_tests/triton_tests/rope/test_fused_qkv_split_qk_rope.py]=290
    FILE_TIMES[op_tests/triton_tests/gemm/fused/test_fused_gemm_afp4wfp4_a16w16.py]=275
    FILE_TIMES[op_tests/triton_tests/attention/test_fav3_sage.py]=266
    FILE_TIMES[op_tests/triton_tests/attention/test_mha.py]=225
    FILE_TIMES[op_tests/triton_tests/attention/test_mha_dao_ai.py]=222
    FILE_TIMES[op_tests/triton_tests/moe/test_moe.py]=187
    FILE_TIMES[op_tests/triton_tests/gemm/basic/test_gemm_a8w8.py]=176
    FILE_TIMES[op_tests/triton_tests/attention/test_mla.py]=174
    FILE_TIMES[op_tests/triton_tests/gemm/batched/test_batched_gemm_afp4wfp4.py]=167
    FILE_TIMES[op_tests/triton_tests/rope/test_rope.py]=166
    FILE_TIMES[op_tests/triton_tests/attention/test_unified_attention.py]=153
    FILE_TIMES[op_tests/triton_tests/attention/test_mha_with_pe.py]=152
    FILE_TIMES[op_tests/triton_tests/fusions/test_mhc.py]=142
    FILE_TIMES[op_tests/triton_tests/attention/test_pa_decode.py]=141
    FILE_TIMES[op_tests/triton_tests/moe/test_moe_gemm_a8w8.py]=138
    FILE_TIMES[op_tests/triton_tests/moe/test_moe_routing.py]=128
    FILE_TIMES[op_tests/triton_tests/gemm/feed_forward/test_ff_a16w16_fused.py]=122
    FILE_TIMES[op_tests/triton_tests/moe/test_moe_gemm_a8w4.py]=121
    FILE_TIMES[op_tests/triton_tests/moe/test_moe_gemm_a8w8_blockscale.py]=118
    FILE_TIMES[op_tests/triton_tests/gemm/fused/test_fused_gemm_afp4wfp4_mul_add.py]=116
    FILE_TIMES[op_tests/triton_tests/attention/test_mha_with_sink.py]=110
    FILE_TIMES[op_tests/triton_tests/gemm/basic/test_gemm_afp4wfp4.py]=110
    FILE_TIMES[op_tests/triton_tests/normalization/test_rmsnorm.py]=108
    FILE_TIMES[op_tests/triton_tests/quant/test_fused_mxfp4_quant.py]=102
    FILE_TIMES[op_tests/triton_tests/test_gmm.py]=96
    FILE_TIMES[op_tests/triton_tests/conv/test_conv2d.py]=94
    FILE_TIMES[op_tests/triton_tests/gemm/fused/test_fused_gemm_a8w8_blockscale_a16w16.py]=92
    FILE_TIMES[op_tests/triton_tests/moe/test_moe_gemm_a4w4.py]=90
    FILE_TIMES[op_tests/triton_tests/gemm/basic/test_gemm_a8w8_blockscale.py]=88
    FILE_TIMES[op_tests/triton_tests/normalization/test_layernorm.py]=80
    FILE_TIMES[op_tests/triton_tests/gemm/feed_forward/test_ff_a16w16.py]=76
    FILE_TIMES[op_tests/triton_tests/fusions/test_fused_kv_cache.py]=74
    FILE_TIMES[op_tests/triton_tests/fusions/test_fused_bmm_rope_kv_cache.py]=71
    FILE_TIMES[op_tests/triton_tests/gemm/basic/test_gemm_a16w16_gated.py]=70
    FILE_TIMES[op_tests/triton_tests/moe/test_moe_gemm_a16w4.py]=70
    FILE_TIMES[op_tests/triton_tests/attention/test_la.py]=55
    FILE_TIMES[op_tests/triton_tests/test_activation.py]=53
    FILE_TIMES[op_tests/triton_tests/moe/test_moe_gemm_int8_smoothquant.py]=51
    FILE_TIMES[op_tests/triton_tests/test_gather_kv_b_proj.py]=49
    FILE_TIMES[op_tests/triton_tests/gemm/fused/test_fused_gemm_afp4wfp4_split_cat.py]=48
    FILE_TIMES[op_tests/triton_tests/fusions/test_fused_reduce_qk_norm_rope_swa_write.py]=42
    FILE_TIMES[op_tests/triton_tests/attention/test_la_paged.py]=37
    FILE_TIMES[op_tests/triton_tests/attention/test_mla_decode_rope.py]=35
    FILE_TIMES[op_tests/triton_tests/gemm/basic/test_gemm_a16w16.py]=32
    FILE_TIMES[op_tests/triton_tests/attention/test_pa_prefill.py]=30
    FILE_TIMES[op_tests/triton_tests/gemm/fused/test_fused_gemm_a8w8_blockscale_mul_add.py]=30
    FILE_TIMES[op_tests/triton_tests/attention/test_chunked_pa_prefill.py]=29
    FILE_TIMES[op_tests/triton_tests/gemm/batched/test_batched_gemm_a8w8.py]=26
    FILE_TIMES[op_tests/triton_tests/gemm/batched/test_batched_gemm_a16wfp4.py]=22
    FILE_TIMES[op_tests/triton_tests/attention/test_unified_attention_sparse_mla.py]=20
    FILE_TIMES[op_tests/triton_tests/gemm/basic/test_gemm_afp8wfp8.py]=20
    FILE_TIMES[op_tests/triton_tests/moe/test_moe_routing_herd.py]=20
    FILE_TIMES[op_tests/triton_tests/gemm/basic/test_gemm_a8w8_per_token_scale.py]=18
    FILE_TIMES[op_tests/triton_tests/quant/test_fused_fp8_quant.py]=18
    FILE_TIMES[op_tests/triton_tests/test_fused_rearrange_sigmoid_gdr.py]=18
    FILE_TIMES[op_tests/triton_tests/attention/test_pa_decode_sparse.py]=17
    FILE_TIMES[op_tests/triton_tests/attention/test_fav3_sage_compile.py]=16
    FILE_TIMES[op_tests/triton_tests/torch_compile/test_compile_gemm_a16w16.py]=15
    FILE_TIMES[op_tests/triton_tests/fusions/test_fused_clamp_act_mul.py]=14
    FILE_TIMES[op_tests/triton_tests/gemm/basic/test_gemm_a16w8_blockscale.py]=14
    FILE_TIMES[op_tests/triton_tests/gemm/fused/test_fused_gemm_a16w16_quant_x.py]=14
    FILE_TIMES[op_tests/triton_tests/moe/test_moe_mx.py]=13
    FILE_TIMES[op_tests/triton_tests/gemm/basic/test_gemm_a16wfp4.py]=10
    FILE_TIMES[op_tests/triton_tests/gemm/batched/test_batched_gemm_bf16.py]=10
    FILE_TIMES[op_tests/triton_tests/quant/test_fused_rms_gated_fp8_group_quant.py]=10
    FILE_TIMES[op_tests/triton_tests/fusions/test_fused_qk_concat.py]=9
    FILE_TIMES[op_tests/triton_tests/gemm/fused/test_fused_gemm_a8w8_blockscale_split_cat.py]=9
    FILE_TIMES[op_tests/triton_tests/quant/test_quant_mxfp8.py]=8
    FILE_TIMES[op_tests/triton_tests/attention/test_extend_attention.py]=7
    FILE_TIMES[op_tests/triton_tests/attention/test_fp8_mqa_logits.py]=7
    FILE_TIMES[op_tests/triton_tests/attention/test_prefill_attention.py]=6
    FILE_TIMES[op_tests/triton_tests/conv/test_causal_conv1d_update_single_token.py]=6
    FILE_TIMES[op_tests/triton_tests/fusions/test_fused_mul_add.py]=6
    FILE_TIMES[op_tests/triton_tests/gemm/basic/test_gemm_a8wfp4.py]=6
    FILE_TIMES[op_tests/triton_tests/normalization/test_fused_add_rmsnorm_pad.py]=6
    FILE_TIMES[op_tests/triton_tests/torch_compile/test_compile_activation.py]=6
    FILE_TIMES[op_tests/triton_tests/torch_compile/test_compile_softmax.py]=6
    FILE_TIMES[op_tests/triton_tests/test_topk.py]=5
    FILE_TIMES[op_tests/triton_tests/torch_compile/test_compile_fused_mul_add.py]=5
    FILE_TIMES[op_tests/triton_tests/quant/test_quant.py]=4
    FILE_TIMES[op_tests/triton_tests/quant/test_quant_mxfp4.py]=4
    FILE_TIMES[op_tests/triton_tests/torch_compile/test_compile_moe_routing.py]=3
    FILE_TIMES[op_tests/triton_tests/torch_compile/test_compile_rope.py]=3
    FILE_TIMES[op_tests/triton_tests/normalization/test_fused_rmsnorm_add.py]=2
    FILE_TIMES[op_tests/triton_tests/test_softmax.py]=2
    FILE_TIMES[op_tests/triton_tests/torch_compile/test_compile_constexpr_mutation.py]=2
    FILE_TIMES[op_tests/triton_tests/torch_compile/test_compile_quant_per_tensor.py]=2
    FILE_TIMES[op_tests/triton_tests/torch_compile/test_compile_quant_per_token.py]=2
    FILE_TIMES[op_tests/triton_tests/torch_compile/test_compile_rmsnorm.py]=2
    FILE_TIMES[op_tests/triton_tests/attention/test_hstu_attn.py]=1
    FILE_TIMES[op_tests/triton_tests/attention/test_pa_prefill_sparse.py]=1
    FILE_TIMES[op_tests/triton_tests/fusions/test_fused_silu_mul.py]=1
    FILE_TIMES[op_tests/triton_tests/moe/test_moe_align_block_size.py]=1
    FILE_TIMES[op_tests/triton_tests/moe/test_moe_routing_sigmoid_top1_fused.py]=1
    FILE_TIMES[op_tests/triton_tests/test_kv_cache.py]=1
    FILE_TIMES[op_tests/triton_tests/torch_compile/test_compile_topk.py]=1
    FILE_TIMES[op_tests/triton_tests/triton_metadata_redirect/test_metadata_redirect.py]=1
fi

get_time() {
    local abs="$1"
    local seconds
    # FILE_TIMES keys use full path (e.g. op_tests/test_mla.py), so look up with abs
    if [[ -n "${FILE_TIMES[$abs]+x}" ]]; then
        seconds="${FILE_TIMES[$abs]}"
    else
        seconds=15
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
