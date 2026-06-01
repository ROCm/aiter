// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#ifdef __HIP_NO_HALF_CONVERSIONS__
#undef __HIP_NO_HALF_CONVERSIONS__
#endif
#ifdef __HIP_NO_HALF_OPERATORS__
#undef __HIP_NO_HALF_OPERATORS__
#endif

#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <hip/amd_detail/amd_hip_bf16.h>
#include <torch/all.h>

#include <string>
#include <unordered_map>

#include "mxfp4_moe_gemm_lookup.h"  // codegen-emitted
#include "gemm2_a4w4.cuh"           // launch_nonatomic_mxfp4 (direct, not codegen)

using namespace aiter::mxfp4_moe::dispatch;

namespace {

const std::unordered_map<std::string, Gemm1CshuffleFn>& g1_cshuffle_lookup() {
    static const std::unordered_map<std::string, Gemm1CshuffleFn> table =
        GENERATE_G1_CSHUFFLE_LOOKUP_TABLE();
    return table;
}
const std::unordered_map<std::string, Gemm2AtomicFn>& g2_atomic_lookup() {
    static const std::unordered_map<std::string, Gemm2AtomicFn> table =
        GENERATE_G2_ATOMIC_LOOKUP_TABLE();
    return table;
}
const std::unordered_map<std::string, Gemm2NonatomicFn>& g2_nonatomic_lookup() {
    static const std::unordered_map<std::string, Gemm2NonatomicFn> table =
        GENERATE_G2_NONATOMIC_LOOKUP_TABLE();
    return table;
}

}  // namespace

// ── gemm1 (cshuffle) ───────────────────────────────────────────────────────
void mxfp4_moe_gemm1_a4w4_kernel(
    torch::Tensor& cumsum_tensor,
    torch::Tensor& a_quant,
    torch::Tensor& a_scale_sorted_shuffled,
    torch::Tensor& w12_shuffled_quant,
    torch::Tensor& w12_shuffled_scale,
    torch::Tensor& sorted_expert_ids,
    torch::Tensor& m_indices,
    torch::Tensor& inter_sorted_quant,
    torch::Tensor& inter_sorted_shuffled_scale,
    torch::Tensor& hidden_states,
    const std::string& kernelName)
{
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(device_of(a_quant));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    const auto& table = g1_cshuffle_lookup();
    auto it = table.find(kernelName);
    TORCH_CHECK(it != table.end(),
        "mxfp4_moe_gemm1_a4w4 kernel not found: '", kernelName,
        "'. See gen_instances.py (enumerate_g1_instances) for the supported set.");

    it->second(
        stream,
        a_quant.data_ptr(),
        a_scale_sorted_shuffled.data_ptr(),
        w12_shuffled_quant.data_ptr(),
        w12_shuffled_scale.data_ptr(),
        sorted_expert_ids.data_ptr<int32_t>(),
        cumsum_tensor.data_ptr<int32_t>(),
        m_indices.data_ptr<int32_t>(),
        static_cast<int>(a_quant.size(0)),
        inter_sorted_quant.data_ptr(),
        inter_sorted_shuffled_scale.data_ptr(),
        hidden_states.data_ptr());
}

// ── gemm2 (atomic or nonatomic, runtime-selected by kernelName) ────────────
void mxfp4_moe_gemm2_a4w4_kernel(
    torch::Tensor& cumsum_tensor,
    torch::Tensor& inter_sorted_quant,
    torch::Tensor& inter_sorted_shuffled_scale,
    torch::Tensor& w3_shuffled_quant,
    torch::Tensor& w3_shuffled_scale,
    torch::Tensor& sorted_token_ids,
    torch::Tensor& sorted_expert_ids,
    torch::Tensor& sorted_weights,
    torch::Tensor& flat_out,
    int64_t M_logical,
    int64_t max_sorted,
    const std::string& kernelName)
{
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(device_of(inter_sorted_quant));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    // Try nonatomic first (BM=128 only); fall back to atomic (BM ∈ {16,32,64}).
    {
        const auto& table = g2_nonatomic_lookup();
        auto it = table.find(kernelName);
        if (it != table.end()) {
            it->second(
                stream,
                inter_sorted_quant.data_ptr(),
                inter_sorted_shuffled_scale.data_ptr(),
                w3_shuffled_quant.data_ptr(),
                w3_shuffled_scale.data_ptr(),
                sorted_expert_ids.data_ptr<int32_t>(),
                cumsum_tensor.data_ptr<int32_t>(),
                static_cast<int>(max_sorted),
                flat_out.data_ptr());
            return;
        }
    }
    {
        const auto& table = g2_atomic_lookup();
        auto it = table.find(kernelName);
        if (it != table.end()) {
            it->second(
                stream,
                inter_sorted_quant.data_ptr(),
                inter_sorted_shuffled_scale.data_ptr(),
                w3_shuffled_quant.data_ptr(),
                w3_shuffled_scale.data_ptr(),
                sorted_expert_ids.data_ptr<int32_t>(),
                cumsum_tensor.data_ptr<int32_t>(),
                sorted_token_ids.data_ptr<int32_t>(),
                sorted_weights.data_ptr<float>(),
                static_cast<int>(M_logical),
                flat_out.data_ptr());
            return;
        }
    }
    TORCH_CHECK(false,
        "mxfp4_moe_gemm2_a4w4 kernel not found: '", kernelName,
        "'. See gen_instances.py (enumerate_g2_instances) for the supported set.");
}


// gemm2 nonatomic that writes MXFP4 (packed fp4 + e8m0 scale) instead of bf16,
// consumed by mxfp4_moe_scatter_reduce_q. Direct launch (Kimi shape only).
void mxfp4_moe_gemm2_a4w4_mxfp4out_kernel(
    torch::Tensor& cumsum_tensor,
    torch::Tensor& inter_sorted_quant,
    torch::Tensor& inter_sorted_shuffled_scale,
    torch::Tensor& w3_shuffled_quant,
    torch::Tensor& w3_shuffled_scale,
    torch::Tensor& sorted_expert_ids,
    torch::Tensor& flat_out_q,
    torch::Tensor& flat_out_scale,
    int64_t NE,
    int64_t D_HIDDEN,
    int64_t D_INTER,
    int64_t max_sorted)
{
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(device_of(inter_sorted_quant));
    const hipStream_t stream = at::hip::getCurrentHIPStream();
    // MUST match gen_instances.py's MAX_M (drives the A_q/A_scale buffer bound;
    // too-small clips large-prefill tail rows to 0 → wrong output).
    constexpr int MAX_M = 655360;

#define LAUNCH_G2_MXFP4(NE_, K_, N_)                                                              \
    aiter::mxfp4_moe::gemm2::launch_nonatomic_mxfp4<MAX_M, NE_, K_, N_>(                          \
        stream,                                                                                  \
        inter_sorted_quant.data_ptr(), inter_sorted_shuffled_scale.data_ptr(),                   \
        w3_shuffled_quant.data_ptr(), w3_shuffled_scale.data_ptr(),                              \
        sorted_expert_ids.data_ptr<int32_t>(), cumsum_tensor.data_ptr<int32_t>(),                \
        static_cast<int>(max_sorted),                                                            \
        flat_out_q.data_ptr(), flat_out_scale.data_ptr())

    if (D_HIDDEN == 7168 && D_INTER == 512) {
        if (NE == 385) { LAUNCH_G2_MXFP4(385, 512, 7168); return; }
        if (NE == 257) { LAUNCH_G2_MXFP4(257, 512, 7168); return; }
    }
    TORCH_CHECK(false, "mxfp4_moe_gemm2_a4w4_mxfp4out: unsupported (NE=", NE,
        " D_HIDDEN=", D_HIDDEN, " D_INTER=", D_INTER, ")");
#undef LAUNCH_G2_MXFP4
}
