// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// OPUS-based sparse paged prefill attention.
// Hosts launcher + dtype dispatch on top of the device kernel template in
// `pa_sparse_prefill_opus.h` (single-header, IMPL-guarded).

#define PA_SPARSE_PREFILL_OPUS_IMPL
#include "pa_sparse_prefill_opus.h"

#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "aiter_tensor.h"

void pa_sparse_prefill_opus_fwd(aiter_tensor_t& q,
                                aiter_tensor_t& unified_kv,
                                aiter_tensor_t& kv_indices_prefix,
                                aiter_tensor_t& kv_indptr_prefix,
                                aiter_tensor_t& kv,
                                aiter_tensor_t& kv_indices_extend,
                                aiter_tensor_t& kv_indptr_extend,
                                aiter_tensor_t& attn_sink,
                                aiter_tensor_t& out,
                                float softmax_scale)
{
    // ---- Shape / dtype validation -----------------------------------------
    AITER_CHECK(q.dim() == 3, "q must be 3-D [N, H, D], got ndim=", q.dim());
    AITER_CHECK(unified_kv.dim() == 2,
                "unified_kv must be 2-D [total_pages, D], got ndim=",
                unified_kv.dim());
    AITER_CHECK(kv.dim() == 2,
                "kv must be 2-D [total_tokens, D], got ndim=",
                kv.dim());
    AITER_CHECK(out.dim() == 3, "out must be 3-D [N, H, D], got ndim=", out.dim());
    AITER_CHECK(attn_sink.dim() == 1, "attn_sink must be 1-D [H]");

    AITER_CHECK(q.dtype() == kv.dtype() && q.dtype() == unified_kv.dtype() &&
                    q.dtype() == out.dtype(),
                "q/unified_kv/kv/out must share dtype");
    AITER_CHECK(q.dtype() == AITER_DTYPE_bf16 || q.dtype() == AITER_DTYPE_fp16,
                "Only bf16/fp16 are supported");
    AITER_CHECK(attn_sink.dtype() == AITER_DTYPE_fp32, "attn_sink must be fp32");

    AITER_CHECK(kv_indptr_prefix.dtype() == AITER_DTYPE_i32, "kv_indptr_prefix must be int32");
    AITER_CHECK(kv_indices_prefix.dtype() == AITER_DTYPE_i32, "kv_indices_prefix must be int32");
    AITER_CHECK(kv_indptr_extend.dtype() == AITER_DTYPE_i32, "kv_indptr_extend must be int32");
    AITER_CHECK(kv_indices_extend.dtype() == AITER_DTYPE_i32, "kv_indices_extend must be int32");

    const int N = static_cast<int>(q.size(0));
    const int H = static_cast<int>(q.size(1));
    const int D = static_cast<int>(q.size(2));
    AITER_CHECK(D == 512,
                "Only D=512 is compiled for pa_sparse_prefill_opus_fwd, got D=", D);
    AITER_CHECK(unified_kv.size(1) == D, "unified_kv last dim must equal q last dim (D=512)");
    AITER_CHECK(kv.size(1) == D, "kv last dim must equal q last dim (D=512)");
    AITER_CHECK(out.size(0) == N && out.size(1) == H && out.size(2) == D,
                "out shape must match q [N, H, D]");
    AITER_CHECK(attn_sink.size(0) == H, "attn_sink length must equal H");
    AITER_CHECK(kv_indptr_prefix.size(0) == N + 1,
                "kv_indptr_prefix length must be N+1");
    AITER_CHECK(kv_indptr_extend.size(0) == N + 1,
                "kv_indptr_extend length must be N+1");

    // Row-major contiguous strides are required for Q/UnifiedKV/KV/O along D.
    AITER_CHECK(q.stride(2) == 1 && unified_kv.stride(1) == 1 && kv.stride(1) == 1 &&
                    out.stride(2) == 1,
                "Q/UnifiedKV/KV/O must be contiguous along the head-dim D");

    // Kernel reads these 1-D buffers via raw pointer arithmetic; stride must be 1.
    AITER_CHECK(kv_indices_prefix.is_contiguous() && kv_indptr_prefix.is_contiguous() &&
                    kv_indices_extend.is_contiguous() && kv_indptr_extend.is_contiguous() &&
                    attn_sink.is_contiguous(),
                "kv_indices/kv_indptr (prefix+extend) and attn_sink must be contiguous");

    const int total_pages  = static_cast<int>(unified_kv.size(0));
    const int total_tokens = static_cast<int>(kv.size(0));

    if (N == 0) return;

    // ---- Build kernel args -----------------------------------------------
    pa_sparse_prefill_kargs kargs{};
    kargs.q_ptr             = q.data_ptr();
    kargs.unified_kv_ptr    = unified_kv.data_ptr();
    kargs.kv_ptr            = kv.data_ptr();
    kargs.attn_sink_ptr     = attn_sink.data_ptr();
    kargs.out_ptr           = out.data_ptr();
    kargs.kv_indptr_prefix  = reinterpret_cast<const int*>(kv_indptr_prefix.data_ptr());
    kargs.kv_indices_prefix = reinterpret_cast<const int*>(kv_indices_prefix.data_ptr());
    kargs.kv_indptr_extend  = reinterpret_cast<const int*>(kv_indptr_extend.data_ptr());
    kargs.kv_indices_extend = reinterpret_cast<const int*>(kv_indices_extend.data_ptr());
    kargs.N                 = N;
    kargs.H                 = H;
    kargs.D                 = D;
    kargs.total_pages       = total_pages;
    kargs.total_tokens      = total_tokens;
    // The kernel assumes the standard row-major layout for [N, H, D] with the
    // head dim contiguous; we already enforced stride(D) == 1 above.
    kargs.stride_qo_n       = static_cast<int>(q.stride(0));
    kargs.stride_qo_h       = static_cast<int>(q.stride(1));
    kargs.stride_kv_page    = static_cast<int>(unified_kv.stride(0));
    AITER_CHECK(kargs.stride_kv_page == static_cast<int>(kv.stride(0)),
                "unified_kv and kv must share row stride along the D dim");
    kargs.softmax_scale     = softmax_scale;

    // ---- Launch ----------------------------------------------------------
    HipDeviceGuard guard(q.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

#define LAUNCH_PA_PREFILL(KERNEL, TRAITS, KV_TILE, NUM_WARPS)                        \
    do {                                                                             \
        auto launch = [&](auto dtype_tag) {                                          \
            using Traits = TRAITS<16, KV_TILE, 512, NUM_WARPS, decltype(dtype_tag)>; \
            const int num_h_blocks = ceil_div(H, Traits::Q_TILE_SIZE * Traits::T_M); \
            dim3 grid(N, num_h_blocks, 1);                                           \
            dim3 block(Traits::BLOCK_SIZE);                                          \
            KERNEL<Traits><<<grid, block, 0, stream>>>(kargs);                       \
            HIP_CALL_LAUNCH(hipGetLastError());                                      \
        };                                                                           \
        if(q.dtype() == AITER_DTYPE_bf16)                                            \
            launch(bf16_t{});                                                        \
        else                                                                         \
            launch(fp16_t{});                                                        \
    } while(0)

    // 16mx8_32nx1 (T_M=NUM_WARPS) for H > 32; 16mx1_16nx4 (T_M=1) for H <= 32.
    if(H <= 32)
        LAUNCH_PA_PREFILL(pa_prefill_16mx1_16nx4_kernel, pa_prefill_16mx1_16nx4_traits, 64, 4);
    else
        LAUNCH_PA_PREFILL(pa_prefill_16mx8_32nx1_kernel, pa_prefill_16mx8_32nx1_traits, 32, 8);

#undef LAUNCH_PA_PREFILL
}

void pa_sparse_prefill_fp8_opus_fwd(aiter_tensor_t& q_nope,
                                    aiter_tensor_t& q_rope,
                                    aiter_tensor_t& unified_kv_nope,
                                    aiter_tensor_t& unified_kv_rope,
                                    aiter_tensor_t& kv_indices_prefix,
                                    aiter_tensor_t& kv_indptr_prefix,
                                    aiter_tensor_t& kv_nope,
                                    aiter_tensor_t& kv_rope,
                                    aiter_tensor_t& kv_indices_extend,
                                    aiter_tensor_t& kv_indptr_extend,
                                    aiter_tensor_t& attn_sink,
                                    aiter_tensor_t& out,
                                    float softmax_scale)
{
    // Single compiled configuration: split NoPE fp8 (448 + 14 E8M0 scales + pad
    // = 512 fp8 slots/row) and RoPE bf16 (64), D_HEAD = 512.
    using Traits = pa_16mx1_16nx4_fp8_traits<16, 64, 4, fp8_t, bf16_t, bf16_t>;
    constexpr int D_NOPE_PADDED = Traits::D_NOPE_PADDED_SIZE; // 512
    constexpr int D_ROPE        = Traits::D_ROPE_SIZE;        // 64
    constexpr int D_HEAD        = Traits::D_HEAD_SIZE;        // 512

    // ---- Shape / dtype validation -----------------------------------------
    AITER_CHECK(q_nope.dim() == 3, "q_nope must be 3-D [N, H, 512], got ndim=", q_nope.dim());
    AITER_CHECK(q_rope.dim() == 3, "q_rope must be 3-D [N, H, 64], got ndim=", q_rope.dim());
    AITER_CHECK(unified_kv_nope.dim() == 2,
                "unified_kv_nope must be 2-D [total_pages, 512], got ndim=", unified_kv_nope.dim());
    AITER_CHECK(unified_kv_rope.dim() == 2,
                "unified_kv_rope must be 2-D [total_pages, 64], got ndim=", unified_kv_rope.dim());
    AITER_CHECK(kv_nope.dim() == 2,
                "kv_nope must be 2-D [total_tokens, 512], got ndim=", kv_nope.dim());
    AITER_CHECK(kv_rope.dim() == 2,
                "kv_rope must be 2-D [total_tokens, 64], got ndim=", kv_rope.dim());
    AITER_CHECK(out.dim() == 3, "out must be 3-D [N, H, 512], got ndim=", out.dim());
    AITER_CHECK(attn_sink.dim() == 1, "attn_sink must be 1-D [H]");

    AITER_CHECK(q_nope.dtype() == AITER_DTYPE_fp8 && unified_kv_nope.dtype() == AITER_DTYPE_fp8 &&
                    kv_nope.dtype() == AITER_DTYPE_fp8,
                "q_nope/unified_kv_nope/kv_nope must be fp8");
    AITER_CHECK(q_rope.dtype() == AITER_DTYPE_bf16 && unified_kv_rope.dtype() == AITER_DTYPE_bf16 &&
                    kv_rope.dtype() == AITER_DTYPE_bf16,
                "q_rope/unified_kv_rope/kv_rope must be bf16");
    AITER_CHECK(out.dtype() == AITER_DTYPE_bf16, "out must be bf16");
    AITER_CHECK(attn_sink.dtype() == AITER_DTYPE_fp32, "attn_sink must be fp32");

    AITER_CHECK(kv_indptr_prefix.dtype() == AITER_DTYPE_i32, "kv_indptr_prefix must be int32");
    AITER_CHECK(kv_indices_prefix.dtype() == AITER_DTYPE_i32, "kv_indices_prefix must be int32");
    AITER_CHECK(kv_indptr_extend.dtype() == AITER_DTYPE_i32, "kv_indptr_extend must be int32");
    AITER_CHECK(kv_indices_extend.dtype() == AITER_DTYPE_i32, "kv_indices_extend must be int32");

    const int N = static_cast<int>(q_nope.size(0));
    const int H = static_cast<int>(q_nope.size(1));

    AITER_CHECK(q_nope.size(2) == D_NOPE_PADDED, "q_nope last dim must be 512 (NoPE padded + scales)");
    AITER_CHECK(q_rope.size(0) == N && q_rope.size(1) == H && q_rope.size(2) == D_ROPE,
                "q_rope shape must be [N, H, 64]");
    AITER_CHECK(unified_kv_nope.size(1) == D_NOPE_PADDED, "unified_kv_nope last dim must be 512");
    AITER_CHECK(unified_kv_rope.size(1) == D_ROPE, "unified_kv_rope last dim must be 64");
    AITER_CHECK(kv_nope.size(1) == D_NOPE_PADDED, "kv_nope last dim must be 512");
    AITER_CHECK(kv_rope.size(1) == D_ROPE, "kv_rope last dim must be 64");
    AITER_CHECK(unified_kv_nope.size(0) == unified_kv_rope.size(0),
                "unified_kv_nope and unified_kv_rope must share total_pages");
    AITER_CHECK(kv_nope.size(0) == kv_rope.size(0),
                "kv_nope and kv_rope must share total_tokens");
    AITER_CHECK(out.size(0) == N && out.size(1) == H && out.size(2) == D_HEAD,
                "out shape must be [N, H, 512]");
    AITER_CHECK(attn_sink.size(0) == H, "attn_sink length must equal H");
    AITER_CHECK(kv_indptr_prefix.size(0) == N + 1, "kv_indptr_prefix length must be N+1");
    AITER_CHECK(kv_indptr_extend.size(0) == N + 1, "kv_indptr_extend length must be N+1");

    // The kernel indexes consecutive query heads within a tile by D_NOPE_PADDED /
    // D_ROPE; Q/KV NoPE/RoPE rows must therefore be densely packed.
    AITER_CHECK(q_nope.stride(2) == 1 && q_nope.stride(1) == D_NOPE_PADDED,
                "q_nope must be contiguous with row stride 512");
    AITER_CHECK(q_rope.stride(2) == 1 && q_rope.stride(1) == D_ROPE,
                "q_rope must be contiguous with row stride 64");
    AITER_CHECK(unified_kv_nope.stride(1) == 1 && kv_nope.stride(1) == 1,
                "kv_nope/unified_kv_nope must be contiguous along the head-dim");
    AITER_CHECK(unified_kv_rope.stride(1) == 1 && kv_rope.stride(1) == 1,
                "kv_rope/unified_kv_rope must be contiguous along the head-dim");
    AITER_CHECK(out.stride(2) == 1, "out must be contiguous along the head-dim");

    AITER_CHECK(kv_indices_prefix.is_contiguous() && kv_indptr_prefix.is_contiguous() &&
                    kv_indices_extend.is_contiguous() && kv_indptr_extend.is_contiguous() &&
                    attn_sink.is_contiguous(),
                "kv_indices/kv_indptr (prefix+extend) and attn_sink must be contiguous");

    const int total_pages  = static_cast<int>(unified_kv_nope.size(0));
    const int total_tokens = static_cast<int>(kv_nope.size(0));

    if(N == 0)
        return;

    const int stride_kv_nope_page = static_cast<int>(unified_kv_nope.stride(0));
    const int stride_kv_rope_page = static_cast<int>(unified_kv_rope.stride(0));
    AITER_CHECK(stride_kv_nope_page == static_cast<int>(kv_nope.stride(0)),
                "unified_kv_nope and kv_nope must share row stride");
    AITER_CHECK(stride_kv_rope_page == static_cast<int>(kv_rope.stride(0)),
                "unified_kv_rope and kv_rope must share row stride");

    // ---- Build kernel args -----------------------------------------------
    pa_fp8_kargs kargs{};
    kargs.q_nope_ptr          = q_nope.data_ptr();
    kargs.q_rope_ptr          = q_rope.data_ptr();
    kargs.unified_kv_nope_ptr = unified_kv_nope.data_ptr();
    kargs.unified_kv_rope_ptr = unified_kv_rope.data_ptr();
    kargs.kv_nope_ptr         = kv_nope.data_ptr();
    kargs.kv_rope_ptr         = kv_rope.data_ptr();
    kargs.attn_sink_ptr       = attn_sink.data_ptr();
    kargs.out_ptr             = out.data_ptr();
    kargs.kv_indptr_prefix    = reinterpret_cast<const int*>(kv_indptr_prefix.data_ptr());
    kargs.kv_indices_prefix   = reinterpret_cast<const int*>(kv_indices_prefix.data_ptr());
    kargs.kv_indptr_extend    = reinterpret_cast<const int*>(kv_indptr_extend.data_ptr());
    kargs.kv_indices_extend   = reinterpret_cast<const int*>(kv_indices_extend.data_ptr());
    kargs.N                   = N;
    kargs.H                   = H;
    kargs.total_pages         = total_pages;
    kargs.total_tokens        = total_tokens;
    kargs.stride_q_nope_n     = static_cast<int>(q_nope.stride(0));
    kargs.stride_q_nope_h     = static_cast<int>(q_nope.stride(1));
    kargs.stride_q_rope_n     = static_cast<int>(q_rope.stride(0));
    kargs.stride_q_rope_h     = static_cast<int>(q_rope.stride(1));
    kargs.stride_o_n          = static_cast<int>(out.stride(0));
    kargs.stride_o_h          = static_cast<int>(out.stride(1));
    kargs.stride_kv_nope_page = stride_kv_nope_page;
    kargs.stride_kv_rope_page = stride_kv_rope_page;
    kargs.softmax_scale       = softmax_scale;

    // ---- Launch ----------------------------------------------------------
    HipDeviceGuard guard(q_nope.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

#define LAUNCH_PA_PREFILL_FP8(KERNEL, TRAITS, KV_TILE, NUM_WARPS)                  \
    do {                                                                          \
        using KTraits = TRAITS<16, KV_TILE, NUM_WARPS, fp8_t, bf16_t, bf16_t>;    \
        const int num_h_blocks = ceil_div(H, KTraits::Q_TILE_SIZE * KTraits::T_M);\
        dim3 grid(N, num_h_blocks, 1);                                            \
        dim3 block(KTraits::BLOCK_SIZE);                                          \
        KERNEL<KTraits><<<grid, block, 0, stream>>>(kargs);                       \
        HIP_CALL_LAUNCH(hipGetLastError());                                       \
    } while(0)

    // 16mx8_32nx1 (T_M=NUM_WARPS) for H > 32; 16mx1_16nx4 (T_M=1) for H <= 32.
    if(H <= 32)
        LAUNCH_PA_PREFILL_FP8(pa_prefill_16mx1_16nx4_fp8_kernel, pa_16mx1_16nx4_fp8_traits, 64, 4);
    else
        LAUNCH_PA_PREFILL_FP8(pa_prefill_16mx8_32nx1_fp8_kernel, pa_16mx8_32nx1_fp8_traits, 32, 8);

#undef LAUNCH_PA_PREFILL_FP8
}
