// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// gfx1250 MXFP4/MXFP8 x MXFP4 MoE gate/up GEMM + SwiGLU (limit=10).
#include "asm_moe_gfx1250_common.h"
#include "aiter_ctypes_error.h"
#include "asm_moe_stage1_configs.hpp"
#include <cstddef>

namespace
{
// preload ABI: BF16 uses the first 84 bytes (21 dwords). Fused MXFP4
// output appends an unaligned pointer at 0x54, outside the preload window.
struct __attribute__((packed)) KernelArgs
{
    void* output;               // 0x00
    void* activation;           // 0x08
    void* gate_up;              // 0x10
    void* num_valid_ids;         // 0x18
    void* activation_scale;     // 0x20
    void* gate_up_scale;        // 0x28
    void* sorted_token_ids;     // 0x30
    void* sorted_expert_ids;    // 0x38
    unsigned int hidden_size;   // 0x40
    unsigned int inter_dim;     // 0x44
    unsigned int num_tokens;    // 0x48
    unsigned int num_experts;   // 0x4c
    unsigned int topk;          // 0x50
    void* output_scale;         // 0x54
};
static_assert(offsetof(KernelArgs, hidden_size) == 64);
static_assert(offsetof(KernelArgs, output_scale) == 84);
static_assert(sizeof(KernelArgs) == 92);
} // namespace

AITER_CTYPES_ERROR_DEF

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    moe_stage1_gfx1250_asm,
    (aiter_tensor_t* A, aiter_tensor_t* W, aiter_tensor_t* AScale,
     aiter_tensor_t* WScale, aiter_tensor_t* sorted_token_ids,
     aiter_tensor_t* sorted_expert_ids, aiter_tensor_t* num_valid_ids,
     aiter_tensor_t* out, aiter_tensor_t* out_scale, int64_t topk, int64_t block_m,
     const char* kernelName, hipStream_t stream),
    (A, W, AScale, WScale, sorted_token_ids, sorted_expert_ids, num_valid_ids,
     out, out_scale, topk, block_m, kernelName, stream))
{
    using namespace aiter::moe_gfx1250;
    AITER_CHECK(A != nullptr, "A must not be null");
    const int device = A->device_id;
    check_tensor(A, "A", 2, device);
    check_tensor(W, "W", 3, device);
    check_tensor(out, "out", 3, device);
    AITER_CHECK(is_fp4(W->dtype()), "W must contain packed MXFP4 bytes");
    AITER_CHECK(is_fp4(A->dtype()) || A->dtype() == AITER_DTYPE_fp8,
                "A must be MXFP4 or MXFP8 E4M3");
    const bool fp8 = A->dtype() == AITER_DTYPE_fp8;
    const int experts = dimension(W->size(0), "num_experts");
    const int n = dimension(W->size(1), "2*inter_dim");
    const int k = dimension(W->size(2) * 2, "hidden_size");
    AITER_CHECK(n % 32 == 0, "2*inter_dim must be divisible by 32");
    const int inter = n / 2;
    check_shape(A, {A->size(0), fp8 ? k : k / 2}, "A");
    check_routing(sorted_token_ids, sorted_expert_ids, num_valid_ids,
                  A->size(0), topk, block_m, device);
    check_scales(AScale, WScale, experts, n, k, sorted_token_ids->numel(), device);

    const bool fq = is_fp4(out->dtype());
    AITER_CHECK(fq || out->dtype() == AITER_DTYPE_bf16,
                "stage1 output must be BF16 or packed MXFP4");
    check_shape(out, {A->size(0), topk, fq ? inter / 2 : inter}, "out");
    if(fq)
    {
        AITER_CHECK(fp8, "fused MXFP4 output requires MXFP8 input");
        AITER_CHECK(inter % 128 == 0, "fused MXFP4 output requires inter_dim%128==0");
        check_tensor(out_scale, "out_scale", 2, device);
        AITER_CHECK(is_scale(out_scale->dtype()), "out_scale must contain E8M0 bytes");
        AITER_CHECK(out_scale->size(1) == inter / 32 &&
                        static_cast<size_t>(out_scale->size(0)) >= scale_rows(sorted_token_ids->numel()),
                    "out_scale must have shape [at least round_up(sorted capacity,32), inter_dim/32]");
    }
    else
        AITER_CHECK(out_scale == nullptr, "BF16 stage1 must not receive out_scale");

    const HipDeviceGuard guard(device);
    const auto& cfg = select_config(cfg_moe_stage1, fp8 ? "mxfp8" : "mxfp4",
                                    fq ? "mxfp4" : "bf16", block_m, n, k, kernelName);
    // Keep unrouted token/slot pairs zero, including when reusing an output.
    if(out->numel() != 0)
        HIP_CALL(hipMemsetAsync(out->ptr, 0, out->numel() * out->element_size(), stream));
    if(fq && out_scale->numel() != 0)
        HIP_CALL(hipMemsetAsync(out_scale->ptr, 0x7f, out_scale->numel(), stream));
    if(A->size(0) == 0 || sorted_token_ids->numel() == 0)
        return;

    KernelArgs args{out->ptr, A->ptr, W->ptr, num_valid_ids->ptr, AScale->ptr,
                    WScale->ptr, sorted_token_ids->ptr, sorted_expert_ids->ptr,
                    static_cast<unsigned>(k), static_cast<unsigned>(inter),
                    static_cast<unsigned>(A->size(0)), static_cast<unsigned>(experts),
                    static_cast<unsigned>(topk), fq ? out_scale->ptr : nullptr};
    launch(cfg, args, fq ? sizeof(args) : offsetof(KernelArgs, output_scale), stream);
}
