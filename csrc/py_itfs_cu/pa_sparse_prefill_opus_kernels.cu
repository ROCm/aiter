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
                                float softmax_scale,
                                std::optional<aiter_tensor_t> kv_scales)
{
    const bool use_fp8_prefix = kv_scales.has_value();
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

    // q/kv/out always share the compute dtype (bf16/fp16). unified_kv matches
    // that on the bf16 path, but is fp8 when kv_scales is provided (dequant in-kernel).
    AITER_CHECK(q.dtype() == kv.dtype() && q.dtype() == out.dtype(),
                "q/kv/out must share dtype");
    AITER_CHECK(q.dtype() == AITER_DTYPE_bf16 || q.dtype() == AITER_DTYPE_fp16,
                "Only bf16/fp16 are supported");
    if(use_fp8_prefix)
        AITER_CHECK(unified_kv.dtype() == AITER_DTYPE_fp8,
                    "fp8 prefix path: unified_kv must be fp8 (provide kv_scales only for fp8 unified_kv)");
    else
        AITER_CHECK(unified_kv.dtype() == q.dtype(),
                    "bf16 path: unified_kv must share dtype with q");
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

    // fp8 prefix block scales (null/0 on the bf16 path).
    if(use_fp8_prefix)
    {
        AITER_CHECK(kv_scales->dim() == 2,
                    "kv_scales must be 2-D [total_pages, num_groups], got ndim=", kv_scales->dim());
        AITER_CHECK(kv_scales->dtype() == AITER_DTYPE_fp32, "kv_scales must be fp32");
        AITER_CHECK(kv_scales->size(0) == total_pages,
                    "kv_scales rows must equal unified_kv total_pages");
        AITER_CHECK(kv_scales->stride(1) == 1, "kv_scales groups must be contiguous (stride(1)==1)");
        kargs.kv_scales_ptr   = kv_scales->data_ptr();
        kargs.stride_kv_scale = static_cast<int>(kv_scales->stride(0));
        kargs.num_groups      = static_cast<int>(kv_scales->size(1));
    }
    else
    {
        kargs.kv_scales_ptr   = nullptr;
        kargs.stride_kv_scale = 0;
        kargs.num_groups      = 0;
    }

    // ---- Launch ----------------------------------------------------------
    HipDeviceGuard guard(q.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

#define LAUNCH_PA_PREFILL(KERNEL, TRAITS, KV_TILE, NUM_WARPS, PREFIX_FP8)            \
    do {                                                                             \
        auto launch = [&](auto dtype_tag) {                                          \
            using Traits = TRAITS<16, KV_TILE, 512, NUM_WARPS, decltype(dtype_tag)>; \
            const int num_h_blocks = ceil_div(H, Traits::Q_TILE_SIZE * Traits::T_M); \
            dim3 grid(N, num_h_blocks, 1);                                           \
            dim3 block(Traits::BLOCK_SIZE);                                          \
            KERNEL<Traits, PREFIX_FP8><<<grid, block, 0, stream>>>(kargs);           \
            HIP_CALL_LAUNCH(hipGetLastError());                                      \
        };                                                                           \
        if(q.dtype() == AITER_DTYPE_bf16)                                            \
            launch(bf16_t{});                                                        \
        else                                                                         \
            launch(fp16_t{});                                                        \
    } while(0)

    // 16mx8_32nx1 (T_M=NUM_WARPS) for H > 32; 16mx1_16nx4 (T_M=1) for H <= 32.
    // Both variants support the fp8 prefix: H<=32 uses the pipelined 16mx1 fp8
    // path; H>32 (e.g. DP attention carrying all heads per rank) dequants the
    // prefix via the 16mx8 le2_tiles accumulator.
    if(H <= 32)
    {
        if(use_fp8_prefix)
            LAUNCH_PA_PREFILL(pa_prefill_16mx1_16nx4_kernel, pa_prefill_16mx1_16nx4_traits, 64, 4, true);
        else
            LAUNCH_PA_PREFILL(pa_prefill_16mx1_16nx4_kernel, pa_prefill_16mx1_16nx4_traits, 64, 4, false);
    }
    else
    {
        if(use_fp8_prefix)
            LAUNCH_PA_PREFILL(pa_prefill_16mx8_32nx1_kernel, pa_prefill_16mx8_32nx1_traits, 32, 8, true);
        else
            LAUNCH_PA_PREFILL(pa_prefill_16mx8_32nx1_kernel, pa_prefill_16mx8_32nx1_traits, 32, 8, false);
    }

#undef LAUNCH_PA_PREFILL
}
