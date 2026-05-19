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

    using TraitsBF16 = pa_sparse_prefill_traits<16, 32, 512, 8, bf16_t>;
    using TraitsFP16 = pa_sparse_prefill_traits<16, 32, 512, 8, fp16_t>;

    auto launch = [&](auto traits_tag) {
        using Traits           = decltype(traits_tag);
        const int num_h_tiles  = ceil_div(H, Traits::Q_TILE_SIZE);
        const int num_h_blocks = ceil_div(num_h_tiles, Traits::NUM_WARPS);
        dim3 grid(N, num_h_blocks, 1);
        dim3 block(Traits::BLOCK_SIZE);
        pa_prefill_kernel<Traits><<<grid, block, 0, stream>>>(kargs);
        HIP_CALL_LAUNCH(hipGetLastError());
    };

    if(q.dtype() == AITER_DTYPE_bf16)
    {
        launch(TraitsBF16{});
    }
    else
    {
        launch(TraitsFP16{});
    }
}
