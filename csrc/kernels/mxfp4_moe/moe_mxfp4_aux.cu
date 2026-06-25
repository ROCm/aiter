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

#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>

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
    TORCH_CHECK(it != table.end(), what,
        ": no codegen'd instance for shape key '", key,
        "'. See moe_aux/codegen/gen_instances.py (enumerate_instances).");
    return it->second;
}

}  // namespace


void mxfp4_moe_sort_quant_kernel(
    torch::Tensor& a_input,
    torch::Tensor& topk_ids,
    torch::Tensor& topk_weight,
    torch::Tensor& sorted_token_ids,
    torch::Tensor& sorted_expert_ids,
    torch::Tensor& cumsum_tensor,
    torch::Tensor& reverse_sorted,
    torch::Tensor& sorted_weights,
    torch::Tensor& a_quant,
    torch::Tensor& a_scale,
    torch::Tensor& m_indices,
    torch::Tensor& bf16_zero_out,
    int64_t NE,
    int64_t TOPK,
    int64_t D_HIDDEN,
    int64_t MB)
{
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(device_of(a_input));
    const hipStream_t stream = at::hip::getCurrentHIPStream();
    const int M = static_cast<int>(a_input.size(0));

    void* bf16_zero_ptr = (bf16_zero_out.numel() > 0) ? bf16_zero_out.data_ptr() : nullptr;

    const std::string key = "aux_sort_quant_NE" + std::to_string(NE)
        + "_TOPK" + std::to_string(TOPK) + "_MB" + std::to_string(MB)
        + "_H" + std::to_string(D_HIDDEN);
    aux_find(sort_quant_lookup(), key, "mxfp4_moe_sort_quant")(
        stream, M,
        a_input.data_ptr(),
        topk_ids.data_ptr<int32_t>(), topk_weight.data_ptr<float>(),
        sorted_token_ids.data_ptr<int32_t>(), sorted_expert_ids.data_ptr<int32_t>(),
        cumsum_tensor.data_ptr<int32_t>(), reverse_sorted.data_ptr<int32_t>(),
        sorted_weights.data_ptr<float>(),
        a_quant.data_ptr(), a_scale.data_ptr(),
        m_indices.data_ptr<int32_t>(),
        bf16_zero_ptr);
}


void mxfp4_moe_sort_kernel(
    torch::Tensor& topk_ids,
    torch::Tensor& topk_weight,
    torch::Tensor& sorted_token_ids,
    torch::Tensor& sorted_expert_ids,
    torch::Tensor& cumsum_tensor,
    torch::Tensor& reverse_sorted,
    torch::Tensor& sorted_weights,
    torch::Tensor& m_indices,
    torch::Tensor& bf16_zero_out,
    torch::Tensor& bf16_zero_workspace,
    int64_t M_logical,
    int64_t NE,
    int64_t TOPK,
    int64_t D_HIDDEN,
    int64_t D_INTER,
    int64_t MB,
    int64_t prologue)
{
    (void)D_INTER;
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(device_of(topk_ids));
    const hipStream_t stream = at::hip::getCurrentHIPStream();
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
        auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(topk_ids.device());
        auto scratch  = torch::empty({(int64_t)NE * kSplitSortCtas + NE}, opts_i32);
        int32_t* block_offsets = scratch.data_ptr<int32_t>();
        int32_t* real_counts   = block_offsets + NE * kSplitSortCtas;

        const std::string key = "aux_sort3s_NE" + std::to_string(NE)
            + "_TOPK" + std::to_string(TOPK) + "_MB" + std::to_string(MB);
        aux_find(sort3stage_lookup(), key, "mxfp4_moe_sort (threestage)")(
            stream, M,
            topk_ids.data_ptr<int32_t>(), topk_weight.data_ptr<float>(),
            sorted_token_ids.data_ptr<int32_t>(), sorted_expert_ids.data_ptr<int32_t>(),
            cumsum_tensor.data_ptr<int32_t>(), reverse_sorted.data_ptr<int32_t>(),
            sorted_weights.data_ptr<float>(),
            m_indices.data_ptr<int32_t>(),
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
            topk_ids.data_ptr<int32_t>(), topk_weight.data_ptr<float>(),
            sorted_token_ids.data_ptr<int32_t>(), sorted_expert_ids.data_ptr<int32_t>(),
            cumsum_tensor.data_ptr<int32_t>(), reverse_sorted.data_ptr<int32_t>(),
            sorted_weights.data_ptr<float>(),
            m_indices.data_ptr<int32_t>(),
            bf16_zero_ptr, bf16_zero_ws_ptr, workspace_bytes);
    } else {
        const std::string key = "aux_sortonly_NE" + std::to_string(NE)
            + "_TOPK" + std::to_string(TOPK) + "_MB" + std::to_string(MB)
            + "_H" + std::to_string(D_HIDDEN);
        aux_find(sort_only_lookup(), key, "mxfp4_moe_sort (inline_quant)")(
            stream, M,
            topk_ids.data_ptr<int32_t>(), topk_weight.data_ptr<float>(),
            sorted_token_ids.data_ptr<int32_t>(), sorted_expert_ids.data_ptr<int32_t>(),
            cumsum_tensor.data_ptr<int32_t>(), reverse_sorted.data_ptr<int32_t>(),
            sorted_weights.data_ptr<float>(),
            m_indices.data_ptr<int32_t>());
    }
}


void mxfp4_moe_quant_kernel(
    torch::Tensor& a_input,
    torch::Tensor& a_quant,
    torch::Tensor& a_scale,
    torch::Tensor& bf16_zero_out,
    int64_t NE,
    int64_t TOPK,
    int64_t D_HIDDEN,
    int64_t MB)
{
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(device_of(a_input));
    const hipStream_t stream = at::hip::getCurrentHIPStream();
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
    torch::Tensor& a_scale,
    torch::Tensor& sorted_token_ids,
    torch::Tensor& cumsum_tensor,
    torch::Tensor& a_scale_sorted_shuffled,
    int64_t NE,
    int64_t TOPK,
    int64_t D_HIDDEN,
    int64_t D_INTER,
    int64_t MB,
    int64_t max_sorted)
{
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(device_of(a_scale));
    const hipStream_t stream = at::hip::getCurrentHIPStream();
    const int M = static_cast<int>(a_scale.size(0));
    (void)TOPK;

    // sort_scales requires BM ≥ 32 (MN_PACK=2 layout); clamp at BM=16 caller.
    const int64_t BM_clamped = (MB < 32) ? 32 : MB;

    const std::string key = "aux_sortscales_BM" + std::to_string(BM_clamped)
        + "_NE" + std::to_string(NE) + "_E" + std::to_string(D_INTER)
        + "_H" + std::to_string(D_HIDDEN);
    aux_find(sort_scales_lookup(), key, "mxfp4_moe_sort_scales")(
        stream, M, static_cast<int>(max_sorted),
        a_scale.data_ptr(), sorted_token_ids.data_ptr<int32_t>(),
        cumsum_tensor.data_ptr<int32_t>(),
        a_scale_sorted_shuffled.data_ptr());
}


void mxfp4_moe_scatter_reduce_kernel(
    torch::Tensor& flat_out,
    torch::Tensor& reverse_sorted,
    torch::Tensor& sorted_weights,
    torch::Tensor& out,
    int64_t NE,
    int64_t TOPK,
    int64_t D_HIDDEN,
    int64_t MB)
{
    (void)NE;
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(device_of(flat_out));
    const hipStream_t stream = at::hip::getCurrentHIPStream();
    const int M = static_cast<int>(out.size(0));

    // nt_hints on only at BM=128: large M is DRAM-bound, smaller M fits L2.
    const int nt = (MB >= 128) ? 1 : 0;

    const std::string key = "aux_scatter_H" + std::to_string(D_HIDDEN)
        + "_TOPK" + std::to_string(TOPK) + "_NT" + std::to_string(nt);
    aux_find(scatter_reduce_lookup(), key, "mxfp4_moe_scatter_reduce")(
        stream, M,
        flat_out.data_ptr(),
        reverse_sorted.data_ptr<int32_t>(),
        sorted_weights.data_ptr<float>(),
        out.data_ptr());
}


// MXFP4-input scatter_reduce: flat_out staged as packed fp4 + e8m0 block scales.
void mxfp4_moe_scatter_reduce_q_kernel(
    torch::Tensor& flat_out_q,
    torch::Tensor& flat_out_scale,
    torch::Tensor& reverse_sorted,
    torch::Tensor& sorted_weights,
    torch::Tensor& out,
    int64_t NE,
    int64_t TOPK,
    int64_t D_HIDDEN,
    int64_t MB)
{
    (void)NE;
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(device_of(flat_out_q));
    const hipStream_t stream = at::hip::getCurrentHIPStream();
    const int M = static_cast<int>(out.size(0));

    const int nt = (MB >= 128) ? 1 : 0;

    const std::string key = "aux_scatterq_H" + std::to_string(D_HIDDEN)
        + "_TOPK" + std::to_string(TOPK) + "_NT" + std::to_string(nt);
    aux_find(scatter_reduce_q_lookup(), key, "mxfp4_moe_scatter_reduce_q")(
        stream, M,
        flat_out_q.data_ptr(), flat_out_scale.data_ptr(),
        reverse_sorted.data_ptr<int32_t>(),
        sorted_weights.data_ptr<float>(),
        out.data_ptr());
}
