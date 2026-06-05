// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

// libtorch's INTERFACE_COMPILE_OPTIONS sets these, which break <hip/hip_fp4.h>.
#ifdef __HIP_NO_HALF_CONVERSIONS__
#undef __HIP_NO_HALF_CONVERSIONS__
#endif
#ifdef __HIP_NO_HALF_OPERATORS__
#undef __HIP_NO_HALF_OPERATORS__
#endif

#include "mxfp4_moe.h"

// __hip_bfloat16 must be visible before the .cuh impls.
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <hip/amd_detail/amd_hip_bf16.h>

#include "moe_aux/moe_3stage_sort.cuh"
#include "moe_aux/moe_scatter_reduce.cuh"
#include "moe_aux/moe_sort_quant.cuh"
#include "moe_aux/moe_sort_scales.cuh"

namespace {

constexpr int kNCtasSort            = 512;
constexpr int kThreadsSort          = 1024;
constexpr int kNCtasScales          = 512;
constexpr int kThreadsScales        = 1024;
constexpr int kThreadsScatterReduce = 128;
constexpr int kColsPerThread        = 8;
constexpr int kColsPerThreadQ       = 8;  // mxfp4-input reduce: 8 fp4 = one u32 load (max threads/MLP)
constexpr int kThreadsScatterReduceQ = 128;  // mxfp4-input reduce CTA size (bigger → larger fp4 burst/row)

constexpr int kSplitSortCtas        = 16;
constexpr int kInlineQuantZeroInitCtas = 128;

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
    torch::Tensor& masked_m,
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

    __hip_bfloat16* bf16_zero_ptr = (bf16_zero_out.numel() > 0)
        ? reinterpret_cast<__hip_bfloat16*>(bf16_zero_out.data_ptr())
        : nullptr;

#define LAUNCH(NE_, TOPK_, MB_, D_HIDDEN_)                                                     \
    aiter::mxfp4_moe::moe_sort_quant::launch<NE_, TOPK_, MB_, D_HIDDEN_,                       \
                                              kNCtasSort, kThreadsSort>(                       \
        stream, M,                                                                             \
        reinterpret_cast<const __hip_bfloat16*>(a_input.data_ptr()),                           \
        topk_ids.data_ptr<int32_t>(), topk_weight.data_ptr<float>(),                           \
        sorted_token_ids.data_ptr<int32_t>(), sorted_expert_ids.data_ptr<int32_t>(),           \
        cumsum_tensor.data_ptr<int32_t>(), reverse_sorted.data_ptr<int32_t>(),                 \
        sorted_weights.data_ptr<float>(),                                                      \
        reinterpret_cast<uint8_t*>(a_quant.data_ptr()),                                        \
        reinterpret_cast<uint8_t*>(a_scale.data_ptr()),                                        \
        masked_m.data_ptr<int32_t>(), m_indices.data_ptr<int32_t>(),                           \
        bf16_zero_ptr)

    if (TOPK == 9 && D_HIDDEN == 7168) {
        if (NE == 385) {
            if (MB == 32) { LAUNCH(385, 9, 32, 7168); return; }
        }
        if (NE == 257) {
            if (MB == 32) { LAUNCH(257, 9, 32, 7168); return; }
        }
    }
    TORCH_CHECK(false,
        "mxfp4_moe_sort_quant: unsupported (NE=", NE,
        " TOPK=", TOPK, " D_HIDDEN=", D_HIDDEN, " MB=", MB, ")");
#undef LAUNCH
}


void mxfp4_moe_sort_kernel(
    torch::Tensor& topk_ids,
    torch::Tensor& topk_weight,
    torch::Tensor& sorted_token_ids,
    torch::Tensor& sorted_expert_ids,
    torch::Tensor& cumsum_tensor,
    torch::Tensor& reverse_sorted,
    torch::Tensor& sorted_weights,
    torch::Tensor& masked_m,
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

    __hip_bfloat16* bf16_zero_ptr = (bf16_zero_out.numel() > 0)
        ? reinterpret_cast<__hip_bfloat16*>(bf16_zero_out.data_ptr())
        : nullptr;
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

#define LAUNCH_3S(NE_, TOPK_, MB_)                                                             \
        aiter::mxfp4_moe::moe_3stage_sort::launch<NE_, TOPK_, MB_,                             \
                                                   kSplitSortCtas, kThreadsSort>(              \
            stream, M,                                                                         \
            topk_ids.data_ptr<int32_t>(), topk_weight.data_ptr<float>(),                       \
            sorted_token_ids.data_ptr<int32_t>(), sorted_expert_ids.data_ptr<int32_t>(),       \
            cumsum_tensor.data_ptr<int32_t>(), reverse_sorted.data_ptr<int32_t>(),             \
            sorted_weights.data_ptr<float>(),                                                  \
            masked_m.data_ptr<int32_t>(), m_indices.data_ptr<int32_t>(),                       \
            block_offsets, real_counts)

        if (TOPK == 9) {
            if (NE == 385) {
                if (MB == 32)  { LAUNCH_3S(385, 9, 32);  return; }
                if (MB == 128) { LAUNCH_3S(385, 9, 128); return; }
            }
            if (NE == 257) {
                if (MB == 32)  { LAUNCH_3S(257, 9, 32);  return; }
                if (MB == 128) { LAUNCH_3S(257, 9, 128); return; }
            }
        }
        TORCH_CHECK(false,
            "mxfp4_moe_sort (threestage): unsupported (NE=", NE,
            " TOPK=", TOPK, " MB=", MB, ")");
#undef LAUNCH_3S
    }

    // prologue == 0 (inline_quant): with bf16_zero_out → multi-CTA overlap zero-init
    // with sort; otherwise single-CTA sort only.
    if (bf16_zero_ptr != nullptr) {
#define LAUNCH_IQ_ZI(NE_, TOPK_, MB_, D_HIDDEN_)                                                 \
        aiter::mxfp4_moe::moe_sort_quant::launch_sort_only_with_zero_init<                       \
                NE_, TOPK_, MB_, D_HIDDEN_, kInlineQuantZeroInitCtas, kThreadsSort>(             \
            stream, M,                                                                           \
            topk_ids.data_ptr<int32_t>(), topk_weight.data_ptr<float>(),                         \
            sorted_token_ids.data_ptr<int32_t>(), sorted_expert_ids.data_ptr<int32_t>(),         \
            cumsum_tensor.data_ptr<int32_t>(), reverse_sorted.data_ptr<int32_t>(),               \
            sorted_weights.data_ptr<float>(),                                                    \
            masked_m.data_ptr<int32_t>(), m_indices.data_ptr<int32_t>(),                         \
            bf16_zero_ptr, bf16_zero_ws_ptr, workspace_bytes)

        if (TOPK == 9 && D_HIDDEN == 7168) {
            if (NE == 385) {
                if (MB == 16) { LAUNCH_IQ_ZI(385, 9, 16, 7168); return; }
            }
            if (NE == 257) {
                if (MB == 16) { LAUNCH_IQ_ZI(257, 9, 16, 7168); return; }
            }
        }
        TORCH_CHECK(false,
            "mxfp4_moe_sort (inline_quant+zero_init): unsupported (NE=", NE,
            " TOPK=", TOPK, " D_HIDDEN=", D_HIDDEN, " MB=", MB, ")");
#undef LAUNCH_IQ_ZI
    } else {
#define LAUNCH_IQ(NE_, TOPK_, MB_, D_HIDDEN_)                                                   \
        aiter::mxfp4_moe::moe_sort_quant::launch_sort_only<                                     \
                NE_, TOPK_, MB_, D_HIDDEN_, kThreadsSort>(                                      \
            stream, M,                                                                          \
            topk_ids.data_ptr<int32_t>(), topk_weight.data_ptr<float>(),                        \
            sorted_token_ids.data_ptr<int32_t>(), sorted_expert_ids.data_ptr<int32_t>(),        \
            cumsum_tensor.data_ptr<int32_t>(), reverse_sorted.data_ptr<int32_t>(),              \
            sorted_weights.data_ptr<float>(),                                                   \
            masked_m.data_ptr<int32_t>(), m_indices.data_ptr<int32_t>())

        if (TOPK == 9 && D_HIDDEN == 7168) {
            if (NE == 385) {
                if (MB == 16) { LAUNCH_IQ(385, 9, 16, 7168); return; }
            }
            if (NE == 257) {
                if (MB == 16) { LAUNCH_IQ(257, 9, 16, 7168); return; }
            }
        }
        TORCH_CHECK(false,
            "mxfp4_moe_sort (inline_quant): unsupported (NE=", NE,
            " TOPK=", TOPK, " D_HIDDEN=", D_HIDDEN, " MB=", MB, ")");
#undef LAUNCH_IQ
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

    __hip_bfloat16* bf16_zero_ptr = (bf16_zero_out.numel() > 0)
        ? reinterpret_cast<__hip_bfloat16*>(bf16_zero_out.data_ptr())
        : nullptr;

#define LAUNCH(NE_, TOPK_, MB_, D_HIDDEN_)                                                      \
    aiter::mxfp4_moe::moe_sort_quant::launch_quant<                                             \
            NE_, TOPK_, MB_, D_HIDDEN_, kNCtasSort, kThreadsSort>(                              \
        stream, M,                                                                              \
        reinterpret_cast<const __hip_bfloat16*>(a_input.data_ptr()),                            \
        reinterpret_cast<uint8_t*>(a_quant.data_ptr()),                                         \
        reinterpret_cast<uint8_t*>(a_scale.data_ptr()),                                         \
        bf16_zero_ptr)

    if (TOPK == 9 && D_HIDDEN == 7168) {
        if (NE == 385) {
            if (MB == 32)  { LAUNCH(385, 9, 32,  7168); return; }
            if (MB == 128) { LAUNCH(385, 9, 128, 7168); return; }
        }
        if (NE == 257) {
            if (MB == 32)  { LAUNCH(257, 9, 32,  7168); return; }
            if (MB == 128) { LAUNCH(257, 9, 128, 7168); return; }
        }
    }
    TORCH_CHECK(false,
        "mxfp4_moe_quant: unsupported (NE=", NE,
        " TOPK=", TOPK, " D_HIDDEN=", D_HIDDEN, " MB=", MB, ")");
#undef LAUNCH
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
    constexpr int kBK = 256;

#define LAUNCH(BM_, NE_, D_INTER_, D_HIDDEN_, BK_)                                              \
    aiter::mxfp4_moe::moe_sort_scales::launch<                                                   \
            BM_, NE_, D_INTER_, D_HIDDEN_, BK_, kNCtasScales, kThreadsScales>(                  \
        stream, M, static_cast<int>(max_sorted),                                                 \
        a_scale.data_ptr<uint8_t>(), sorted_token_ids.data_ptr<int32_t>(),                       \
        cumsum_tensor.data_ptr<int32_t>(),                                                       \
        a_scale_sorted_shuffled.data_ptr<uint8_t>())

    if (D_HIDDEN == 7168 && D_INTER == 512) {
        if (NE == 385) {
            if (BM_clamped == 32)  { LAUNCH(32,  385, 512, 7168, kBK); return; }
            if (BM_clamped == 128) { LAUNCH(128, 385, 512, 7168, kBK); return; }
        }
        if (NE == 257) {
            if (BM_clamped == 32)  { LAUNCH(32,  257, 512, 7168, kBK); return; }
            if (BM_clamped == 128) { LAUNCH(128, 257, 512, 7168, kBK); return; }
        }
    }
    TORCH_CHECK(false,
        "mxfp4_moe_sort_scales: unsupported (NE=", NE,
        " D_HIDDEN=", D_HIDDEN, " D_INTER=", D_INTER,
        " MB=", MB, " → BM_clamped=", BM_clamped, ")");
#undef LAUNCH
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
    const bool nt_hints = (MB >= 128);

#define LAUNCH(D_HIDDEN_, TOPK_, NT_)                                                            \
    aiter::mxfp4_moe::moe_scatter_reduce::launch<                                                \
            D_HIDDEN_, TOPK_, kThreadsScatterReduce, kColsPerThread, NT_>(                       \
        stream, M,                                                                               \
        reinterpret_cast<const __hip_bfloat16*>(flat_out.data_ptr()),                            \
        reverse_sorted.data_ptr<int32_t>(),                                                      \
        sorted_weights.data_ptr<float>(),                                                        \
        reinterpret_cast<__hip_bfloat16*>(out.data_ptr()))

    if (D_HIDDEN == 7168 && TOPK == 9) {
        if (nt_hints) { LAUNCH(7168, 9, true);  return; }
        else          { LAUNCH(7168, 9, false); return; }
    }
    TORCH_CHECK(false,
        "mxfp4_moe_scatter_reduce: unsupported (TOPK=", TOPK,
        " D_HIDDEN=", D_HIDDEN, ")");
#undef LAUNCH
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
    const bool nt_hints = (MB >= 128);

#define LAUNCH_Q(D_HIDDEN_, TOPK_, NT_)                                                           \
    aiter::mxfp4_moe::moe_scatter_reduce::launch_mxfp4<                                           \
            D_HIDDEN_, TOPK_, kThreadsScatterReduceQ, kColsPerThreadQ, NT_>(                      \
        stream, M,                                                                               \
        reinterpret_cast<const uint8_t*>(flat_out_q.data_ptr()),                                  \
        reinterpret_cast<const uint8_t*>(flat_out_scale.data_ptr()),                              \
        reverse_sorted.data_ptr<int32_t>(),                                                       \
        sorted_weights.data_ptr<float>(),                                                         \
        reinterpret_cast<__hip_bfloat16*>(out.data_ptr()))

    if (D_HIDDEN == 7168 && TOPK == 9) {
        if (nt_hints) { LAUNCH_Q(7168, 9, true);  return; }
        else          { LAUNCH_Q(7168, 9, false); return; }
    }
    TORCH_CHECK(false,
        "mxfp4_moe_scatter_reduce_q: unsupported (TOPK=", TOPK,
        " D_HIDDEN=", D_HIDDEN, ")");
#undef LAUNCH_Q
}
