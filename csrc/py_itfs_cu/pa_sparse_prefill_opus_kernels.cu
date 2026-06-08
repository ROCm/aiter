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

// =============================================================================
// FP8 (DeepSeek-V4 / asm-v4 layout) entry: dequant q/unified_kv/kv into bf16
// scratch via the standalone dequant kernel, then run the bf16 attention.
// =============================================================================
void pa_sparse_prefill_opus_fp8_fwd(aiter_tensor_t& q_nope,
                                    aiter_tensor_t& q_rope,
                                    aiter_tensor_t& q_scale,
                                    aiter_tensor_t& unified_kv_nope,
                                    aiter_tensor_t& unified_kv_rope,
                                    aiter_tensor_t& unified_kv_scale,
                                    aiter_tensor_t& kv_nope,
                                    aiter_tensor_t& kv_rope,
                                    aiter_tensor_t& kv_scale,
                                    aiter_tensor_t& kv_indices_prefix,
                                    aiter_tensor_t& kv_indptr_prefix,
                                    aiter_tensor_t& kv_indices_extend,
                                    aiter_tensor_t& kv_indptr_extend,
                                    aiter_tensor_t& attn_sink,
                                    aiter_tensor_t& q_bf16,
                                    aiter_tensor_t& unified_kv_bf16,
                                    aiter_tensor_t& kv_bf16,
                                    aiter_tensor_t& out,
                                    float softmax_scale)
{
    constexpr int D_NOPE = 448;
    constexpr int D_ROPE = 64;
    constexpr int D_FULL = 512;
    constexpr int NUM_TILES = 7;

    // ---- dtype validation -------------------------------------------------
    auto check_nope = [&](aiter_tensor_t& t, const char* nm) {
        AITER_CHECK(t.dtype() == AITER_DTYPE_fp8, nm, " must be fp8 (e4m3)");
        AITER_CHECK(t.is_contiguous(), nm, " must be contiguous");
        AITER_CHECK(t.size(t.dim() - 1) == D_NOPE, nm, " last dim must be 448");
    };
    auto check_rope = [&](aiter_tensor_t& t, const char* nm) {
        AITER_CHECK(t.dtype() == AITER_DTYPE_bf16, nm, " must be bf16");
        AITER_CHECK(t.is_contiguous(), nm, " must be contiguous");
        AITER_CHECK(t.size(t.dim() - 1) == D_ROPE, nm, " last dim must be 64");
    };
    auto check_scale = [&](aiter_tensor_t& t, const char* nm) {
        AITER_CHECK(t.dtype() == AITER_DTYPE_fp32, nm, " must be fp32");
        AITER_CHECK(t.is_contiguous(), nm, " must be contiguous");
        AITER_CHECK(t.size(t.dim() - 1) == NUM_TILES, nm, " last dim must be 7");
    };
    auto check_bf16_scratch = [&](aiter_tensor_t& t, const char* nm) {
        AITER_CHECK(t.dtype() == AITER_DTYPE_bf16, nm, " scratch must be bf16");
        AITER_CHECK(t.is_contiguous(), nm, " scratch must be contiguous");
        AITER_CHECK(t.size(t.dim() - 1) == D_FULL, nm, " scratch last dim must be 512");
    };

    check_nope(q_nope, "q_nope");       check_rope(q_rope, "q_rope");       check_scale(q_scale, "q_scale");
    check_nope(unified_kv_nope, "unified_kv_nope"); check_rope(unified_kv_rope, "unified_kv_rope"); check_scale(unified_kv_scale, "unified_kv_scale");
    check_nope(kv_nope, "kv_nope");     check_rope(kv_rope, "kv_rope");     check_scale(kv_scale, "kv_scale");
    check_bf16_scratch(q_bf16, "q_bf16"); check_bf16_scratch(unified_kv_bf16, "unified_kv_bf16"); check_bf16_scratch(kv_bf16, "kv_bf16");

    HipDeviceGuard guard(q_nope.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    auto rows_of = [&](aiter_tensor_t& t) {
        long n = 1;
        for(int i = 0; i < t.dim() - 1; ++i) n *= t.size(i);
        return static_cast<int>(n);
    };

    auto launch_dequant = [&](aiter_tensor_t& nope, aiter_tensor_t& rope,
                              aiter_tensor_t& scale, aiter_tensor_t& dst, int rows) {
        if(rows == 0) return;
        pa_v4_dequant_kargs dk{};
        dk.nope_ptr     = nope.data_ptr();
        dk.rope_ptr     = rope.data_ptr();
        dk.scale_ptr    = reinterpret_cast<const float*>(scale.data_ptr());
        dk.out_ptr      = dst.data_ptr();
        dk.rows         = rows;
        dk.stride_nope  = D_NOPE;
        dk.stride_rope  = D_ROPE;
        dk.stride_scale = NUM_TILES;
        dk.stride_out   = D_FULL;
        dim3 grid(rows);
        dim3 block(128);
        pa_v4_fp8_dequant_kernel<<<grid, block, 0, stream>>>(dk);
        HIP_CALL_LAUNCH(hipGetLastError());
    };

    launch_dequant(q_nope, q_rope, q_scale, q_bf16, rows_of(q_nope));
    launch_dequant(unified_kv_nope, unified_kv_rope, unified_kv_scale, unified_kv_bf16, rows_of(unified_kv_nope));
    launch_dequant(kv_nope, kv_rope, kv_scale, kv_bf16, rows_of(kv_nope));

    // Run the existing, proven bf16 attention on the dequantized scratch.
    pa_sparse_prefill_opus_fwd(q_bf16,
                               unified_kv_bf16,
                               kv_indices_prefix,
                               kv_indptr_prefix,
                               kv_bf16,
                               kv_indices_extend,
                               kv_indptr_extend,
                               attn_sink,
                               out,
                               softmax_scale);
}

// =============================================================================
// FUSED fp8 entry: single-warp kernel reading fp8 KV directly (no bf16 scratch).
// =============================================================================
void pa_sparse_prefill_opus_fp8_fused_fwd(aiter_tensor_t& q_nope,
                                          aiter_tensor_t& q_rope,
                                          aiter_tensor_t& q_scale,
                                          aiter_tensor_t& unified_kv_nope,
                                          aiter_tensor_t& unified_kv_rope,
                                          aiter_tensor_t& unified_kv_scale,
                                          aiter_tensor_t& kv_nope,
                                          aiter_tensor_t& kv_rope,
                                          aiter_tensor_t& kv_scale,
                                          aiter_tensor_t& kv_indices_prefix,
                                          aiter_tensor_t& kv_indptr_prefix,
                                          aiter_tensor_t& kv_indices_extend,
                                          aiter_tensor_t& kv_indptr_extend,
                                          aiter_tensor_t& attn_sink,
                                          aiter_tensor_t& out,
                                          float softmax_scale)
{
    const int N = static_cast<int>(q_nope.size(0));
    const int H = static_cast<int>(q_nope.size(1));
    AITER_CHECK(q_nope.dtype() == AITER_DTYPE_fp8, "q_nope must be fp8");
    AITER_CHECK(q_rope.dtype() == AITER_DTYPE_bf16, "q_rope must be bf16");
    AITER_CHECK(out.dtype() == AITER_DTYPE_bf16, "out must be bf16");
    AITER_CHECK(H % 16 == 0, "H must be a multiple of 16 for the fused fp8 kernel, got H=", H);
    AITER_CHECK(q_nope.size(2) == 448, "q_nope last dim must be 448");
    if (N == 0) return;

    pa_sparse_prefill_fp8_kargs kargs{};
    kargs.q_nope    = q_nope.data_ptr();
    kargs.q_rope    = q_rope.data_ptr();
    kargs.q_scale   = reinterpret_cast<const float*>(q_scale.data_ptr());
    kargs.ukv_nope  = unified_kv_nope.data_ptr();
    kargs.ukv_rope  = unified_kv_rope.data_ptr();
    kargs.ukv_scale = reinterpret_cast<const float*>(unified_kv_scale.data_ptr());
    kargs.kv_nope   = kv_nope.data_ptr();
    kargs.kv_rope   = kv_rope.data_ptr();
    kargs.kv_scale  = reinterpret_cast<const float*>(kv_scale.data_ptr());
    kargs.attn_sink = attn_sink.data_ptr();
    kargs.out       = out.data_ptr();
    kargs.kv_indptr_prefix  = reinterpret_cast<const int*>(kv_indptr_prefix.data_ptr());
    kargs.kv_indices_prefix = reinterpret_cast<const int*>(kv_indices_prefix.data_ptr());
    kargs.kv_indptr_extend  = reinterpret_cast<const int*>(kv_indptr_extend.data_ptr());
    kargs.kv_indices_extend = reinterpret_cast<const int*>(kv_indices_extend.data_ptr());
    kargs.N = N;
    kargs.H = H;
    kargs.total_pages  = static_cast<int>(unified_kv_nope.size(0));
    kargs.total_tokens = static_cast<int>(kv_nope.size(0));
    kargs.softmax_scale = softmax_scale;

    HipDeviceGuard guard(q_nope.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    // v4 layout: 448 nope + 64 rope, e8m0 scale per 64-elem tile, KV tile 32,
    // up to MAX_WARPS=4 independent head-tiles/block.
    using Traits = pa_prefill_fp8_traits<16, 32, 448, 64, 64, 4>;
    // Pack up to MAX_WARPS head-tiles per block (more resident waves to hide
    // memory latency). nwarp must divide H/16 so every warp maps to a valid head.
    const int htiles = H / Traits::QTILE;
    int nwarp = 1;
    for (int nw = Traits::MAX_WARPS; nw >= 1; --nw) {
        if (htiles % nw == 0) { nwarp = nw; break; }
    }
    dim3 grid(N, htiles / nwarp, 1);
    dim3 block(nwarp * Traits::WARP_SIZE);
    pa_prefill_fp8_fused_kernel<Traits><<<grid, block, 0, stream>>>(kargs);
    HIP_CALL_LAUNCH(hipGetLastError());
}
