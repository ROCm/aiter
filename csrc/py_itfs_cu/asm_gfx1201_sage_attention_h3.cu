// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx1201 SageAttention H3 forward from the hand-scheduled code object in hsa/gfx1201/sage_attn_h3.
// It consumes gfx1201_sage_prepare_h3 output and, unlike the bm512 core, takes Q as raw BF16 and
// quantizes it per 32-row block inside the kernel, so no q_int8/q_scale is passed.
#include "aiter_hip_common.h"
#include "aiter_tensor.h"

#include <cstddef>

#include "aiter_ctypes_error.h"
#include "asm_sage_attn_h3_configs.hpp"

struct __attribute__((packed)) KernelArgs
{
    const void* q_bf16;
    const void* k_int8;
    const void* v_fp8;
    void* out;
    const void* unused;
    const void* k_scale;
    const void* v_scale;
    int64_t batch;
    int64_t seq_len_q;
    int64_t seq_len_kv;
    int64_t num_heads_q;
    int64_t num_heads_kv;
    int64_t q_stride_b, q_stride_s, q_stride_h;
    int64_t k_stride_b, k_stride_s, k_stride_h;
    int64_t v_stride_b, v_stride_d, v_stride_h;
    int64_t o_stride_b, o_stride_s, o_stride_h;
    int64_t reserved0;
    int64_t reserved1;
    int64_t k_scale_stride_b;
    int64_t k_scale_stride_h;
    int32_t mode;
    float softmax_scale;
    bool causal;
    uint8_t padding[3]; // the code object's kernarg segment is 236 B, the fields end at 233
};
static_assert(sizeof(KernelArgs) == 236, "kernarg layout of the code object");
static_assert(offsetof(KernelArgs, mode) == 224 && offsetof(KernelArgs, softmax_scale) == 228 &&
                  offsetof(KernelArgs, causal) == 232,
              "scalar kernarg offsets of the code object");

static constexpr int kBlockM    = 256; // query rows per workgroup
static constexpr int kBlockSize = 512; // 8 waves x wave64

AITER_CTYPES_ERROR_DEF

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    gfx1201_sage_attention_h3_fwd_asm,
    (aiter_tensor_t * query,
     aiter_tensor_t* k_int8,
     aiter_tensor_t* k_scale,
     aiter_tensor_t* v_fp8,
     aiter_tensor_t* v_scale,
     aiter_tensor_t* out,
     float softmax_scale,
     hipStream_t stream),
    (query, k_int8, k_scale, v_fp8, v_scale, out, softmax_scale, stream))
{
    const HipDeviceGuard device_guard(query->device_id);
    const int head_dim            = (int)query->size(3);
    const std::string arch_id     = get_gpu_arch();
    const sage_attn_h3Config* cfg = nullptr;
    for(const auto& el : cfg_sage_attn_h3)
        if(el.first.find(arch_id) == 0 && el.second.hdim == head_dim)
            cfg = &el.second;
    AITER_CHECK(
        cfg != nullptr, __func__, ": no code object for arch ", arch_id, " head dim ", head_dim);

    static SynchronizedCache<std::string_view, AiterAsmKernel> kernels;
    AiterAsmKernel& kernel = kernels.get_or_create(cfg->knl_name, [&]() {
        return AiterAsmKernel(cfg->knl_name.c_str(), cfg->co_name.c_str());
    });

    const int64_t batch   = query->size(0);
    const int64_t seq_len = query->size(1);
    const int64_t heads   = query->size(2);
    AITER_CHECK(k_int8->size(0) == batch && k_int8->size(1) == seq_len &&
                    k_int8->size(2) == heads,
                __func__,
                ": Q and K must have the same batch, sequence length and head count");

    KernelArgs args;
    args.q_bf16  = query->ptr;
    args.k_int8  = k_int8->ptr;
    args.v_fp8   = v_fp8->ptr;
    args.out     = out->ptr;
    args.unused  = nullptr;
    args.k_scale = k_scale->ptr;
    args.v_scale = v_scale->ptr;

    args.batch        = batch;
    args.seq_len_q    = seq_len;
    args.seq_len_kv   = k_int8->size(1);
    args.num_heads_q  = heads;
    args.num_heads_kv = heads;

    args.q_stride_b = query->stride(0);
    args.q_stride_s = query->stride(1);
    args.q_stride_h = query->stride(2);
    args.k_stride_b = k_int8->stride(0);
    args.k_stride_s = k_int8->stride(1);
    args.k_stride_h = k_int8->stride(2);
    // V is stored transposed as [B, H, D, S_pad], so the kernel wants batch, row then head.
    args.v_stride_b = v_fp8->stride(0);
    args.v_stride_d = v_fp8->stride(2);
    args.v_stride_h = v_fp8->stride(1);
    args.o_stride_b = out->stride(0);
    args.o_stride_s = out->stride(1);
    args.o_stride_h = out->stride(2);

    args.reserved0        = 0;
    args.reserved1        = 0;
    args.k_scale_stride_b = k_scale->stride(0);
    args.k_scale_stride_h = k_scale->stride(1);

    args.mode          = 0;
    args.softmax_scale = softmax_scale;
    args.causal        = false;

    size_t arg_size = sizeof(args);
    const int64_t grid_x = (seq_len + kBlockM - 1) / kBlockM;
    AITER_CHECK(grid_x * heads * batch < (int64_t(1) << 31), __func__, ": grid too large");
    kernel.launch_kernel(
        {&args, &arg_size, (int)grid_x, (int)heads, (int)batch, kBlockSize, 1, 1, stream});
}
