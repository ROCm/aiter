// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// gfx1250 MXFP4/MXFP8 x MXFP4 MoE down GEMM + weighted BF16 atomic reduction.
#include "asm_moe_gfx1250_common.h"
#include "aiter_ctypes_error.h"
#include "asm_moe_stage2_configs.hpp"
#include <cstddef>

namespace
{
struct __attribute__((packed)) KernelArgs
{
    void* output;               // 0x00
    void* activation;           // 0x08
    void* weight;               // 0x10
    void* num_valid_ids;         // 0x18
    void* activation_scale;     // 0x20
    void* weight_scale;         // 0x28
    void* sorted_token_ids;     // 0x30
    void* sorted_expert_ids;    // 0x38
    void* sorted_weights;       // 0x40
    unsigned int hidden_size;   // 0x48
    unsigned int inter_dim;     // 0x4c
    unsigned int num_tokens;    // 0x50
    unsigned int num_experts;   // 0x54
    unsigned int topk;          // 0x58
};
static_assert(offsetof(KernelArgs, sorted_weights) == 64);
static_assert(offsetof(KernelArgs, hidden_size) == 72);
static_assert(sizeof(KernelArgs) == 92); // 23 preloaded dwords
} // namespace

AITER_CTYPES_ERROR_DEF

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    moe_stage2_gfx1250_asm,
    (aiter_tensor_t* A, aiter_tensor_t* W, aiter_tensor_t* AScale,
     aiter_tensor_t* WScale, aiter_tensor_t* sorted_token_ids,
     aiter_tensor_t* sorted_expert_ids, aiter_tensor_t* num_valid_ids,
     aiter_tensor_t* sorted_weights, aiter_tensor_t* out, int64_t block_m,
     const char* kernelName, hipStream_t stream),
    (A, W, AScale, WScale, sorted_token_ids, sorted_expert_ids, num_valid_ids,
     sorted_weights, out, block_m, kernelName, stream))
{
    using namespace aiter::moe_gfx1250;
    AITER_CHECK(A != nullptr, "A must not be null");
    const int device = A->device_id;
    check_tensor(A, "A", 3, device);
    check_tensor(W, "W", 3, device);
    check_tensor(out, "out", 2, device);
    AITER_CHECK(is_fp4(W->dtype()), "W must contain packed MXFP4 bytes");
    AITER_CHECK(is_fp4(A->dtype()) || A->dtype() == AITER_DTYPE_fp8,
                "A must be MXFP4 or MXFP8 E4M3");
    const bool fp8 = A->dtype() == AITER_DTYPE_fp8;
    const int experts = dimension(W->size(0), "num_experts");
    const int n = dimension(W->size(1), "hidden_size");
    const int k = dimension(W->size(2) * 2, "inter_dim");
    const int topk = dimension(A->size(1), "topk");
    check_shape(A, {A->size(0), topk, fp8 ? k : k / 2}, "A");
    check_routing(sorted_token_ids, sorted_expert_ids, num_valid_ids,
                  A->size(0), topk, block_m, device);
    check_scales(AScale, WScale, experts, n, k, sorted_token_ids->numel(), device);
    check_tensor(sorted_weights, "sorted_weights", 1, device);
    AITER_CHECK(sorted_weights->dtype() == AITER_DTYPE_fp32 &&
                    sorted_weights->numel() >= sorted_token_ids->numel(),
                "sorted_weights must be FP32 with one value per sorted token");
    AITER_CHECK(out->dtype() == AITER_DTYPE_bf16, "stage2 output must be BF16");
    check_shape(out, {A->size(0), n}, "out");

    const HipDeviceGuard guard(device);
    const auto& cfg = select_config(cfg_moe_stage2, fp8 ? "mxfp8" : "mxfp4",
                                    "bf16", block_m, n, k, kernelName);
    // Atomic output MUST be cleared on every call, including graph replay and
    // a caller-provided output; otherwise previous results accumulate into it.
    if(out->numel() != 0)
        HIP_CALL(hipMemsetAsync(out->ptr, 0, out->numel() * out->element_size(), stream));
    if(A->size(0) == 0 || sorted_token_ids->numel() == 0)
        return;

    KernelArgs args{out->ptr, A->ptr, W->ptr, num_valid_ids->ptr, AScale->ptr,
                    WScale->ptr, sorted_token_ids->ptr, sorted_expert_ids->ptr,
                    sorted_weights->ptr, static_cast<unsigned>(n),
                    static_cast<unsigned>(k), static_cast<unsigned>(A->size(0)),
                    static_cast<unsigned>(experts), static_cast<unsigned>(topk)};
    launch(cfg, args, sizeof(args), stream);
}
