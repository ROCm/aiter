// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

// libtorch's INTERFACE_COMPILE_OPTIONS sets these, which break <hip/hip_fp4.h>.
#ifdef __HIP_NO_HALF_CONVERSIONS__
#undef __HIP_NO_HALF_CONVERSIONS__
#endif
#ifdef __HIP_NO_HALF_OPERATORS__
#undef __HIP_NO_HALF_OPERATORS__
#endif

#include "moe_mxfp4_aux.h"

#include "aiter_stream.h"

#include <string>
#include <unordered_map>

#include "mxfp4_moe_aux_lookup.h"  // codegen-emitted (forward decls + lookup macros)

using namespace aiter::mxfp4_moe::aux_dispatch;

namespace {

// ── codegen'd lookup tables (string shape-key -> extern "C" instance) ────────
const std::unordered_map<std::string, SortQuantFn>& sort_quant_lookup() {
    static const std::unordered_map<std::string, SortQuantFn> t =
        GENERATE_AUX_SORT_QUANT_LOOKUP_TABLE();
    return t;
}
const std::unordered_map<std::string, Sort3StageFn>& sort3stage_lookup() {
    static const std::unordered_map<std::string, Sort3StageFn> t =
        GENERATE_AUX_SORT3STAGE_LOOKUP_TABLE();
    return t;
}
const std::unordered_map<std::string, SortOnlyZiFn>& sort_only_zi_lookup() {
    static const std::unordered_map<std::string, SortOnlyZiFn> t =
        GENERATE_AUX_SORT_ONLY_ZI_LOOKUP_TABLE();
    return t;
}
const std::unordered_map<std::string, SortOnlyFn>& sort_only_lookup() {
    static const std::unordered_map<std::string, SortOnlyFn> t =
        GENERATE_AUX_SORT_ONLY_LOOKUP_TABLE();
    return t;
}
const std::unordered_map<std::string, QuantFn>& quant_lookup() {
    static const std::unordered_map<std::string, QuantFn> t =
        GENERATE_AUX_QUANT_LOOKUP_TABLE();
    return t;
}
const std::unordered_map<std::string, SortScalesFn>& sort_scales_lookup() {
    static const std::unordered_map<std::string, SortScalesFn> t =
        GENERATE_AUX_SORT_SCALES_LOOKUP_TABLE();
    return t;
}
const std::unordered_map<std::string, ScatterReduceFn>& scatter_reduce_lookup() {
    static const std::unordered_map<std::string, ScatterReduceFn> t =
        GENERATE_AUX_SCATTER_REDUCE_LOOKUP_TABLE();
    return t;
}
const std::unordered_map<std::string, ScatterReduceQFn>& scatter_reduce_q_lookup() {
    static const std::unordered_map<std::string, ScatterReduceQFn> t =
        GENERATE_AUX_SCATTER_REDUCE_Q_LOOKUP_TABLE();
    return t;
}

template <class Fn>
Fn aux_find(const std::unordered_map<std::string, Fn>& table,
            const std::string& key, const char* what) {
    auto it = table.find(key);
    AITER_CHECK(it != table.end(), what,
        ": no codegen'd instance for shape key '", key,
        "'. See moe_aux/codegen/gen_instances.py (enumerate_instances).");
    return it->second;
}

}  // namespace


void mxfp4_moe_sort_quant_kernel(
    aiter_tensor_t& a_input,
    aiter_tensor_t& topk_ids,
    aiter_tensor_t& topk_weight,
    aiter_tensor_t& sorted_token_ids,
    aiter_tensor_t& sorted_expert_ids,
    aiter_tensor_t& cumsum_tensor,
    aiter_tensor_t& reverse_sorted,
    aiter_tensor_t& sorted_weights,
    aiter_tensor_t& a_quant,
    aiter_tensor_t& a_scale,
    aiter_tensor_t& m_indices,
    aiter_tensor_t& bf16_zero_out,
    int64_t NE,
    int64_t TOPK,
    int64_t D_HIDDEN,
    int64_t MB)
{
    HipDeviceGuard guard(a_input.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    const int M = static_cast<int>(a_input.size(0));

    void* bf16_zero_ptr = (bf16_zero_out.numel() > 0) ? bf16_zero_out.data_ptr() : nullptr;

    const std::string key = "aux_sort_quant_NE" + std::to_string(NE)
        + "_TOPK" + std::to_string(TOPK) + "_MB" + std::to_string(MB)
        + "_H" + std::to_string(D_HIDDEN);
    aux_find(sort_quant_lookup(), key, "mxfp4_moe_sort_quant")(
        stream, M,
        a_input.data_ptr(),
        static_cast<int32_t*>(topk_ids.data_ptr()), static_cast<float*>(topk_weight.data_ptr()),
        static_cast<int32_t*>(sorted_token_ids.data_ptr()), static_cast<int32_t*>(sorted_expert_ids.data_ptr()),
        static_cast<int32_t*>(cumsum_tensor.data_ptr()), static_cast<int32_t*>(reverse_sorted.data_ptr()),
        static_cast<float*>(sorted_weights.data_ptr()),
        a_quant.data_ptr(), a_scale.data_ptr(),
        static_cast<int32_t*>(m_indices.data_ptr()),
        bf16_zero_ptr);
}


void mxfp4_moe_sort_kernel(
    aiter_tensor_t& topk_ids,
    aiter_tensor_t& topk_weight,
    aiter_tensor_t& sorted_token_ids,
    aiter_tensor_t& sorted_expert_ids,
    aiter_tensor_t& cumsum_tensor,
    aiter_tensor_t& reverse_sorted,
    aiter_tensor_t& sorted_weights,
    aiter_tensor_t& m_indices,
    aiter_tensor_t& bf16_zero_out,
    aiter_tensor_t& bf16_zero_workspace,
    aiter_tensor_t& sort3stage_ws,
    int64_t M_logical,
    int64_t NE,
    int64_t TOPK,
    int64_t D_HIDDEN,
    int64_t D_INTER,
    int64_t MB,
    int64_t prologue)
{
    (void)D_INTER;
    HipDeviceGuard guard(topk_ids.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    const int M = static_cast<int>(M_logical);

    void* bf16_zero_ptr = (bf16_zero_out.numel() > 0) ? bf16_zero_out.data_ptr() : nullptr;
    void* bf16_zero_ws_ptr = nullptr;
    long long workspace_bytes = 0;
    if (bf16_zero_workspace.numel() > 0) {
        bf16_zero_ws_ptr = bf16_zero_workspace.data_ptr();
        workspace_bytes  = static_cast<long long>(bf16_zero_workspace.numel())
                         * static_cast<long long>(bf16_zero_workspace.element_size());
    }

    if (prologue == 1 /* threestage */) {
        // Caller-provided int32 scratch (was torch::empty here before de-torch):
        // block_offsets[NE*kSplitSortCtas] followed by real_counts[NE].
        AITER_CHECK(sort3stage_ws.numel() >= (size_t)(NE * kSplitSortCtas + NE),
                    "mxfp4_moe_sort (threestage): sort3stage_ws too small");
        int32_t* block_offsets = static_cast<int32_t*>(sort3stage_ws.data_ptr());
        int32_t* real_counts   = block_offsets + NE * kSplitSortCtas;

        const std::string key = "aux_sort3s_NE" + std::to_string(NE)
            + "_TOPK" + std::to_string(TOPK) + "_MB" + std::to_string(MB);
        aux_find(sort3stage_lookup(), key, "mxfp4_moe_sort (threestage)")(
            stream, M,
            static_cast<int32_t*>(topk_ids.data_ptr()), static_cast<float*>(topk_weight.data_ptr()),
            static_cast<int32_t*>(sorted_token_ids.data_ptr()), static_cast<int32_t*>(sorted_expert_ids.data_ptr()),
            static_cast<int32_t*>(cumsum_tensor.data_ptr()), static_cast<int32_t*>(reverse_sorted.data_ptr()),
            static_cast<float*>(sorted_weights.data_ptr()),
            static_cast<int32_t*>(m_indices.data_ptr()),
            block_offsets, real_counts);
        return;
    }

    // prologue == 0 (inline_quant): with bf16_zero_out → multi-CTA overlap zero-init
    // with sort; otherwise single-CTA sort only.
    if (bf16_zero_ptr != nullptr) {
        const std::string key = "aux_sortzi_NE" + std::to_string(NE)
            + "_TOPK" + std::to_string(TOPK) + "_MB" + std::to_string(MB)
            + "_H" + std::to_string(D_HIDDEN);
        aux_find(sort_only_zi_lookup(), key, "mxfp4_moe_sort (inline_quant+zero_init)")(
            stream, M,
            static_cast<int32_t*>(topk_ids.data_ptr()), static_cast<float*>(topk_weight.data_ptr()),
            static_cast<int32_t*>(sorted_token_ids.data_ptr()), static_cast<int32_t*>(sorted_expert_ids.data_ptr()),
            static_cast<int32_t*>(cumsum_tensor.data_ptr()), static_cast<int32_t*>(reverse_sorted.data_ptr()),
            static_cast<float*>(sorted_weights.data_ptr()),
            static_cast<int32_t*>(m_indices.data_ptr()),
            bf16_zero_ptr, bf16_zero_ws_ptr, workspace_bytes);
    } else {
        const std::string key = "aux_sortonly_NE" + std::to_string(NE)
            + "_TOPK" + std::to_string(TOPK) + "_MB" + std::to_string(MB)
            + "_H" + std::to_string(D_HIDDEN);
        aux_find(sort_only_lookup(), key, "mxfp4_moe_sort (inline_quant)")(
            stream, M,
            static_cast<int32_t*>(topk_ids.data_ptr()), static_cast<float*>(topk_weight.data_ptr()),
            static_cast<int32_t*>(sorted_token_ids.data_ptr()), static_cast<int32_t*>(sorted_expert_ids.data_ptr()),
            static_cast<int32_t*>(cumsum_tensor.data_ptr()), static_cast<int32_t*>(reverse_sorted.data_ptr()),
            static_cast<float*>(sorted_weights.data_ptr()),
            static_cast<int32_t*>(m_indices.data_ptr()));
    }
}


void mxfp4_moe_quant_kernel(
    aiter_tensor_t& a_input,
    aiter_tensor_t& a_quant,
    aiter_tensor_t& a_scale,
    aiter_tensor_t& bf16_zero_out,
    int64_t NE,
    int64_t TOPK,
    int64_t D_HIDDEN,
    int64_t MB)
{
    HipDeviceGuard guard(a_input.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    const int M = static_cast<int>(a_input.size(0));

    void* bf16_zero_ptr = (bf16_zero_out.numel() > 0) ? bf16_zero_out.data_ptr() : nullptr;

    const std::string key = "aux_quant_NE" + std::to_string(NE)
        + "_TOPK" + std::to_string(TOPK) + "_MB" + std::to_string(MB)
        + "_H" + std::to_string(D_HIDDEN);
    aux_find(quant_lookup(), key, "mxfp4_moe_quant")(
        stream, M,
        a_input.data_ptr(), a_quant.data_ptr(), a_scale.data_ptr(),
        bf16_zero_ptr);
}


void mxfp4_moe_sort_scales_kernel(
    aiter_tensor_t& a_scale,
    aiter_tensor_t& sorted_token_ids,
    aiter_tensor_t& cumsum_tensor,
    aiter_tensor_t& a_scale_sorted_shuffled,
    int64_t NE,
    int64_t TOPK,
    int64_t D_HIDDEN,
    int64_t MB,
    int64_t max_sorted)
{
    HipDeviceGuard guard(a_scale.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    const int M = static_cast<int>(a_scale.size(0));
    (void)TOPK;

    // sort_scales requires BM ≥ 32 (MN_PACK=2 layout); clamp at BM=16 caller.
    const int64_t BM_clamped = (MB < 32) ? 32 : MB;

    const std::string key = "aux_sortscales_BM" + std::to_string(BM_clamped)
        + "_NE" + std::to_string(NE)
        + "_H" + std::to_string(D_HIDDEN);
    aux_find(sort_scales_lookup(), key, "mxfp4_moe_sort_scales")(
        stream, M, static_cast<int>(max_sorted),
        a_scale.data_ptr(), static_cast<int32_t*>(sorted_token_ids.data_ptr()),
        static_cast<int32_t*>(cumsum_tensor.data_ptr()),
        a_scale_sorted_shuffled.data_ptr());
}


void mxfp4_moe_scatter_reduce_kernel(
    aiter_tensor_t& flat_out,
    aiter_tensor_t& reverse_sorted,
    aiter_tensor_t& sorted_weights,
    aiter_tensor_t& out,
    int64_t NE,
    int64_t TOPK,
    int64_t D_HIDDEN,
    int64_t MB)
{
    (void)NE;
    HipDeviceGuard guard(flat_out.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    const int M = static_cast<int>(out.size(0));

    // nt_hints on only at BM=128: large M is DRAM-bound, smaller M fits L2.
    const int nt = (MB >= 128) ? 1 : 0;

    const std::string key = "aux_scatter_H" + std::to_string(D_HIDDEN)
        + "_TOPK" + std::to_string(TOPK) + "_NT" + std::to_string(nt);
    aux_find(scatter_reduce_lookup(), key, "mxfp4_moe_scatter_reduce")(
        stream, M,
        flat_out.data_ptr(),
        static_cast<int32_t*>(reverse_sorted.data_ptr()),
        static_cast<float*>(sorted_weights.data_ptr()),
        out.data_ptr());
}


// MXFP4-input scatter_reduce: flat_out staged as packed fp4 + e8m0 block scales.
void mxfp4_moe_scatter_reduce_q_kernel(
    aiter_tensor_t& flat_out_q,
    aiter_tensor_t& flat_out_scale,
    aiter_tensor_t& reverse_sorted,
    aiter_tensor_t& sorted_weights,
    aiter_tensor_t& out,
    int64_t NE,
    int64_t TOPK,
    int64_t D_HIDDEN,
    int64_t MB)
{
    (void)NE;
    HipDeviceGuard guard(flat_out_q.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    const int M = static_cast<int>(out.size(0));

    const int nt = (MB >= 128) ? 1 : 0;

    const std::string key = "aux_scatterq_H" + std::to_string(D_HIDDEN)
        + "_TOPK" + std::to_string(TOPK) + "_NT" + std::to_string(nt);
    aux_find(scatter_reduce_q_lookup(), key, "mxfp4_moe_scatter_reduce_q")(
        stream, M,
        flat_out_q.data_ptr(), flat_out_scale.data_ptr(),
        static_cast<int32_t*>(reverse_sorted.data_ptr()),
        static_cast<float*>(sorted_weights.data_ptr()),
        out.data_ptr());
}
