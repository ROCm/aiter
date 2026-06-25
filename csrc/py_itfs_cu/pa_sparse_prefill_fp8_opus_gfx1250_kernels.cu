// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx1250-ONLY split-precision (NoPE fp8 / RoPE bf16) sparse paged prefill
// attention launcher. Validation + kargs fill + launch derived from the gfx950
// base launcher pa_sparse_prefill_fp8_opus_fwd.

#define PA_SPARSE_PREFILL_FP8_OPUS_GFX1250_IMPL
#include "pa_sparse_prefill_fp8_opus_gfx1250.h"

#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "aiter_tensor.h"

void pa_sparse_prefill_fp8_opus_gfx1250_fwd(aiter_tensor_t& q_nope,
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
    using Traits = pa_fp8_gfx1250_traits<16, 32, 1, fp8_t, bf16_t, bf16_t>;
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
    pa_fp8_gfx1250_kargs kargs{};
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

    // gfx1250 warp size = 32; query at runtime for robustness.
    hipDeviceProp_t prop{};
    HIP_CALL(hipGetDeviceProperties(&prop, q_nope.device_id));
    const int warp_size = prop.warpSize;  // 32 on gfx1250

    const int num_h_blocks = ceil_div_g(H, Traits::Q_TILE_SIZE * Traits::T_M);
    dim3 grid(N, num_h_blocks, 1);
    dim3 block(Traits::NUM_WARPS * warp_size);
    pa_prefill_fp8_gfx1250_kernel<Traits><<<grid, block, 0, stream>>>(kargs);
    HIP_CALL_LAUNCH(hipGetLastError());
}
