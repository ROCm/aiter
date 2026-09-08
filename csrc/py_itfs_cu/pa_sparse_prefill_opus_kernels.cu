// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// OPUS-based sparse paged prefill attention, one launcher pair per target:
//
//   gfx950  -- kernels compiled from the device templates in
//              `pa_sparse_prefill_opus.h` (single-header, IMPL-guarded).
//   gfx1250 -- kernels loaded from the prebuilt code objects in
//              `hsa/gfx1250/mla_v4_opus/`, one per precision.

#define PA_SPARSE_PREFILL_OPUS_IMPL
#include "pa_sparse_prefill_opus.h"

#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "aiter_tensor.h"

#include <cstddef>

void pa_sparse_prefill_gfx950_opus_fwd(aiter_tensor_t& q,
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
                "Only D=512 is compiled for pa_sparse_prefill_gfx950_opus_fwd, got D=", D);
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

void pa_sparse_prefill_fp8_gfx950_opus_fwd(aiter_tensor_t& q_nope,
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

// ============================================================================
// gfx1250: prebuilt code object
// ============================================================================

// The code objects' kernel arguments are a field-for-field match of
// pa_sparse_prefill_kargs and pa_fp8_kargs above, so those are reused verbatim
// as the kernarg buffers.
//
// The catch is that nothing in this repo rebuilds the code objects: an edit made
// for the gfx950 path would silently corrupt the gfx1250 launch. Pin every field
// so such an edit fails the build instead. Sizes match the
// `.kernarg_segment_size` reported by `llvm-readelf --notes <code object>`.
#define PA_GFX1250_CO_ABI(struct_, field_, offset_)                               \
    static_assert(offsetof(struct_, field_) == (offset_),                         \
                  #struct_ "::" #field_ " moved; rebuild the gfx1250 code objects")

static_assert(sizeof(pa_sparse_prefill_kargs) == 112,
              "pa_sparse_prefill_kargs resized; rebuild the gfx1250 code objects");
PA_GFX1250_CO_ABI(pa_sparse_prefill_kargs, q_ptr, 0);
PA_GFX1250_CO_ABI(pa_sparse_prefill_kargs, unified_kv_ptr, 8);
PA_GFX1250_CO_ABI(pa_sparse_prefill_kargs, kv_ptr, 16);
PA_GFX1250_CO_ABI(pa_sparse_prefill_kargs, attn_sink_ptr, 24);
PA_GFX1250_CO_ABI(pa_sparse_prefill_kargs, out_ptr, 32);
PA_GFX1250_CO_ABI(pa_sparse_prefill_kargs, kv_indptr_prefix, 40);
PA_GFX1250_CO_ABI(pa_sparse_prefill_kargs, kv_indices_prefix, 48);
PA_GFX1250_CO_ABI(pa_sparse_prefill_kargs, kv_indptr_extend, 56);
PA_GFX1250_CO_ABI(pa_sparse_prefill_kargs, kv_indices_extend, 64);
PA_GFX1250_CO_ABI(pa_sparse_prefill_kargs, N, 72);
PA_GFX1250_CO_ABI(pa_sparse_prefill_kargs, H, 76);
PA_GFX1250_CO_ABI(pa_sparse_prefill_kargs, D, 80);
PA_GFX1250_CO_ABI(pa_sparse_prefill_kargs, total_pages, 84);
PA_GFX1250_CO_ABI(pa_sparse_prefill_kargs, total_tokens, 88);
PA_GFX1250_CO_ABI(pa_sparse_prefill_kargs, stride_qo_n, 92);
PA_GFX1250_CO_ABI(pa_sparse_prefill_kargs, stride_qo_h, 96);
PA_GFX1250_CO_ABI(pa_sparse_prefill_kargs, stride_kv_page, 100);
PA_GFX1250_CO_ABI(pa_sparse_prefill_kargs, softmax_scale, 104);

static_assert(sizeof(pa_fp8_kargs) == 152,
              "pa_fp8_kargs resized; rebuild the gfx1250 code objects");
PA_GFX1250_CO_ABI(pa_fp8_kargs, q_nope_ptr, 0);
PA_GFX1250_CO_ABI(pa_fp8_kargs, q_rope_ptr, 8);
PA_GFX1250_CO_ABI(pa_fp8_kargs, unified_kv_nope_ptr, 16);
PA_GFX1250_CO_ABI(pa_fp8_kargs, unified_kv_rope_ptr, 24);
PA_GFX1250_CO_ABI(pa_fp8_kargs, kv_nope_ptr, 32);
PA_GFX1250_CO_ABI(pa_fp8_kargs, kv_rope_ptr, 40);
PA_GFX1250_CO_ABI(pa_fp8_kargs, attn_sink_ptr, 48);
PA_GFX1250_CO_ABI(pa_fp8_kargs, out_ptr, 56);
PA_GFX1250_CO_ABI(pa_fp8_kargs, kv_indptr_prefix, 64);
PA_GFX1250_CO_ABI(pa_fp8_kargs, kv_indices_prefix, 72);
PA_GFX1250_CO_ABI(pa_fp8_kargs, kv_indptr_extend, 80);
PA_GFX1250_CO_ABI(pa_fp8_kargs, kv_indices_extend, 88);
PA_GFX1250_CO_ABI(pa_fp8_kargs, N, 96);
PA_GFX1250_CO_ABI(pa_fp8_kargs, H, 100);
PA_GFX1250_CO_ABI(pa_fp8_kargs, total_pages, 104);
PA_GFX1250_CO_ABI(pa_fp8_kargs, total_tokens, 108);
PA_GFX1250_CO_ABI(pa_fp8_kargs, stride_q_nope_n, 112);
PA_GFX1250_CO_ABI(pa_fp8_kargs, stride_q_nope_h, 116);
PA_GFX1250_CO_ABI(pa_fp8_kargs, stride_q_rope_n, 120);
PA_GFX1250_CO_ABI(pa_fp8_kargs, stride_q_rope_h, 124);
PA_GFX1250_CO_ABI(pa_fp8_kargs, stride_o_n, 128);
PA_GFX1250_CO_ABI(pa_fp8_kargs, stride_o_h, 132);
PA_GFX1250_CO_ABI(pa_fp8_kargs, stride_kv_nope_page, 136);
PA_GFX1250_CO_ABI(pa_fp8_kargs, stride_kv_rope_page, 140);
PA_GFX1250_CO_ABI(pa_fp8_kargs, softmax_scale, 144);

#undef PA_GFX1250_CO_ABI

namespace {

// Launch geometry, mirroring pa_16mx4_64nx1_traits<16, 64, 512, 4, CY, ...>:
// one workgroup covers one query token and Q_TILE_SIZE * T_M query heads.
constexpr int kGfx1250QTileSize   = 16;
constexpr int kGfx1250NumWarps    = 4;
constexpr int kGfx1250WarpSize    = 32; // wave32 on gfx1250
constexpr int kGfx1250BlockSize   = kGfx1250NumWarps * kGfx1250WarpSize;
constexpr int kGfx1250HeadsPerBlk = kGfx1250QTileSize * kGfx1250NumWarps;
constexpr int kGfx1250MaxClusterY = 2;

// fp8 split-precision head layout.
constexpr int kGfx1250DNopePadded = 512;
constexpr int kGfx1250DRope       = 64;
constexpr int kGfx1250DHead       = 512;

// Symbol names match each code object's file stem, as elsewhere under hsa/.
constexpr const char* kGfx1250A16W16Co =
    "mla_v4_opus/mla_prefill_a16w16_16mx4_64nx1.co";
constexpr const char* kGfx1250A8W8Co = "mla_v4_opus/mla_prefill_a8w8_16mx4_64nx1.co";

// aiter's prebuilt-code-object loader. Its name comes from the hand-written
// assembly kernels it was introduced for; these code objects are compiler
// output, but the load-and-launch-by-symbol path is the same.
using PrebuiltKernel = AiterAsmKernel;

// Largest cluster width that evenly divides grid.y. A cluster that does not
// divide grid.y would gather through a masked peer set and silently corrupt the
// tail workgroup, so the kernel only ever sees an exact divisor.
int gfx1250_pick_cluster_y(int num_h_blocks)
{
    for(int c = kGfx1250MaxClusterY; c > 1; c >>= 1)
        if(num_h_blocks % c == 0)
            return c;
    return 1;
}

} // namespace

void pa_sparse_prefill_gfx1250_opus_fwd(aiter_tensor_t& q,
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
    AITER_CHECK(kv.dim() == 2, "kv must be 2-D [total_tokens, D], got ndim=", kv.dim());
    AITER_CHECK(out.dim() == 3, "out must be 3-D [N, H, D], got ndim=", out.dim());
    AITER_CHECK(attn_sink.dim() == 1, "attn_sink must be 1-D [H]");

    AITER_CHECK(q.dtype() == kv.dtype() && q.dtype() == unified_kv.dtype() &&
                    q.dtype() == out.dtype(),
                "q/unified_kv/kv/out must share dtype");
    // Only the bf16 traits are instantiated in the gfx1250 code object.
    AITER_CHECK(q.dtype() == AITER_DTYPE_bf16,
                "the gfx1250 code object only provides the bf16 variant");
    AITER_CHECK(attn_sink.dtype() == AITER_DTYPE_fp32, "attn_sink must be fp32");

    AITER_CHECK(kv_indptr_prefix.dtype() == AITER_DTYPE_i32, "kv_indptr_prefix must be int32");
    AITER_CHECK(kv_indices_prefix.dtype() == AITER_DTYPE_i32, "kv_indices_prefix must be int32");
    AITER_CHECK(kv_indptr_extend.dtype() == AITER_DTYPE_i32, "kv_indptr_extend must be int32");
    AITER_CHECK(kv_indices_extend.dtype() == AITER_DTYPE_i32, "kv_indices_extend must be int32");

    const int N = static_cast<int>(q.size(0));
    const int H = static_cast<int>(q.size(1));
    const int D = static_cast<int>(q.size(2));
    AITER_CHECK(D == 512, "Only D=512 is built into the gfx1250 code object, got D=", D);
    AITER_CHECK(unified_kv.size(1) == D, "unified_kv last dim must equal q last dim (D=512)");
    AITER_CHECK(kv.size(1) == D, "kv last dim must equal q last dim (D=512)");
    AITER_CHECK(out.size(0) == N && out.size(1) == H && out.size(2) == D,
                "out shape must match q [N, H, D]");
    AITER_CHECK(attn_sink.size(0) == H, "attn_sink length must equal H");
    AITER_CHECK(kv_indptr_prefix.size(0) == N + 1, "kv_indptr_prefix length must be N+1");
    AITER_CHECK(kv_indptr_extend.size(0) == N + 1, "kv_indptr_extend length must be N+1");

    AITER_CHECK(q.stride(2) == 1 && unified_kv.stride(1) == 1 && kv.stride(1) == 1 &&
                    out.stride(2) == 1,
                "Q/UnifiedKV/KV/O must be contiguous along the head-dim D");
    AITER_CHECK(kv_indices_prefix.is_contiguous() && kv_indptr_prefix.is_contiguous() &&
                    kv_indices_extend.is_contiguous() && kv_indptr_extend.is_contiguous() &&
                    attn_sink.is_contiguous(),
                "kv_indices/kv_indptr (prefix+extend) and attn_sink must be contiguous");

    if(N == 0)
        return;

    // ---- Build kernel args -----------------------------------------------
    pa_sparse_prefill_kargs args{};
    args.q_ptr             = q.data_ptr();
    args.unified_kv_ptr    = unified_kv.data_ptr();
    args.kv_ptr            = kv.data_ptr();
    args.attn_sink_ptr     = attn_sink.data_ptr();
    args.out_ptr           = out.data_ptr();
    args.kv_indptr_prefix  = reinterpret_cast<const int*>(kv_indptr_prefix.data_ptr());
    args.kv_indices_prefix = reinterpret_cast<const int*>(kv_indices_prefix.data_ptr());
    args.kv_indptr_extend  = reinterpret_cast<const int*>(kv_indptr_extend.data_ptr());
    args.kv_indices_extend = reinterpret_cast<const int*>(kv_indices_extend.data_ptr());
    args.N                 = N;
    args.H                 = H;
    args.D                 = D;
    args.total_pages       = static_cast<int>(unified_kv.size(0));
    args.total_tokens      = static_cast<int>(kv.size(0));
    args.stride_qo_n       = static_cast<int>(q.stride(0));
    args.stride_qo_h       = static_cast<int>(q.stride(1));
    args.stride_kv_page    = static_cast<int>(unified_kv.stride(0));
    AITER_CHECK(args.stride_kv_page == static_cast<int>(kv.stride(0)),
                "unified_kv and kv must share row stride along the D dim");
    args.softmax_scale = softmax_scale;

    // ---- Launch ----------------------------------------------------------
    HipDeviceGuard guard(q.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    size_t arg_size        = sizeof(args);
    const int num_h_blocks = ceil_div(H, kGfx1250HeadsPerBlk);
    const int cluster_y    = gfx1250_pick_cluster_y(num_h_blocks);

    if(cluster_y == 2)
    {
        static PrebuiltKernel impl("mla_prefill_a16w16_16mx4_64nx1_cy2", kGfx1250A16W16Co);
        impl.launch_kernel(
            {&args, &arg_size, N, num_h_blocks, 1, kGfx1250BlockSize, 1, 1, stream, 1, 2, 1});
    }
    else
    {
        static PrebuiltKernel impl("mla_prefill_a16w16_16mx4_64nx1_cy1", kGfx1250A16W16Co);
        impl.launch_kernel(
            {&args, &arg_size, N, num_h_blocks, 1, kGfx1250BlockSize, 1, 1, stream});
    }
}

void pa_sparse_prefill_fp8_gfx1250_opus_fwd(aiter_tensor_t& q_nope,
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
    // ---- Shape / dtype validation -----------------------------------------
    AITER_CHECK(q_nope.dim() == 3, "q_nope must be 3-D [N, H, 512], got ndim=", q_nope.dim());
    AITER_CHECK(q_rope.dim() == 3, "q_rope must be 3-D [N, H, 64], got ndim=", q_rope.dim());
    AITER_CHECK(unified_kv_nope.dim() == 2,
                "unified_kv_nope must be 2-D [total_pages, 512], got ndim=",
                unified_kv_nope.dim());
    AITER_CHECK(unified_kv_rope.dim() == 2,
                "unified_kv_rope must be 2-D [total_pages, 64], got ndim=",
                unified_kv_rope.dim());
    AITER_CHECK(kv_nope.dim() == 2,
                "kv_nope must be 2-D [total_tokens, 512], got ndim=",
                kv_nope.dim());
    AITER_CHECK(kv_rope.dim() == 2,
                "kv_rope must be 2-D [total_tokens, 64], got ndim=",
                kv_rope.dim());
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

    AITER_CHECK(q_nope.size(2) == kGfx1250DNopePadded,
                "q_nope last dim must be 512 (NoPE padded + scales)");
    AITER_CHECK(q_rope.size(0) == N && q_rope.size(1) == H && q_rope.size(2) == kGfx1250DRope,
                "q_rope shape must be [N, H, 64]");
    AITER_CHECK(unified_kv_nope.size(1) == kGfx1250DNopePadded,
                "unified_kv_nope last dim must be 512");
    AITER_CHECK(unified_kv_rope.size(1) == kGfx1250DRope, "unified_kv_rope last dim must be 64");
    AITER_CHECK(kv_nope.size(1) == kGfx1250DNopePadded, "kv_nope last dim must be 512");
    AITER_CHECK(kv_rope.size(1) == kGfx1250DRope, "kv_rope last dim must be 64");
    AITER_CHECK(unified_kv_nope.size(0) == unified_kv_rope.size(0),
                "unified_kv_nope and unified_kv_rope must share total_pages");
    AITER_CHECK(kv_nope.size(0) == kv_rope.size(0),
                "kv_nope and kv_rope must share total_tokens");
    AITER_CHECK(out.size(0) == N && out.size(1) == H && out.size(2) == kGfx1250DHead,
                "out shape must be [N, H, 512]");
    AITER_CHECK(attn_sink.size(0) == H, "attn_sink length must equal H");
    AITER_CHECK(kv_indptr_prefix.size(0) == N + 1, "kv_indptr_prefix length must be N+1");
    AITER_CHECK(kv_indptr_extend.size(0) == N + 1, "kv_indptr_extend length must be N+1");

    AITER_CHECK(q_nope.stride(2) == 1 && q_nope.stride(1) == kGfx1250DNopePadded,
                "q_nope must be contiguous with row stride 512");
    AITER_CHECK(q_rope.stride(2) == 1 && q_rope.stride(1) == kGfx1250DRope,
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

    if(N == 0)
        return;

    const int stride_kv_nope_page = static_cast<int>(unified_kv_nope.stride(0));
    const int stride_kv_rope_page = static_cast<int>(unified_kv_rope.stride(0));
    AITER_CHECK(stride_kv_nope_page == static_cast<int>(kv_nope.stride(0)),
                "unified_kv_nope and kv_nope must share row stride");
    AITER_CHECK(stride_kv_rope_page == static_cast<int>(kv_rope.stride(0)),
                "unified_kv_rope and kv_rope must share row stride");

    // ---- Build kernel args -----------------------------------------------
    pa_fp8_kargs args{};
    args.q_nope_ptr          = q_nope.data_ptr();
    args.q_rope_ptr          = q_rope.data_ptr();
    args.unified_kv_nope_ptr = unified_kv_nope.data_ptr();
    args.unified_kv_rope_ptr = unified_kv_rope.data_ptr();
    args.kv_nope_ptr         = kv_nope.data_ptr();
    args.kv_rope_ptr         = kv_rope.data_ptr();
    args.attn_sink_ptr       = attn_sink.data_ptr();
    args.out_ptr             = out.data_ptr();
    args.kv_indptr_prefix    = reinterpret_cast<const int*>(kv_indptr_prefix.data_ptr());
    args.kv_indices_prefix   = reinterpret_cast<const int*>(kv_indices_prefix.data_ptr());
    args.kv_indptr_extend    = reinterpret_cast<const int*>(kv_indptr_extend.data_ptr());
    args.kv_indices_extend   = reinterpret_cast<const int*>(kv_indices_extend.data_ptr());
    args.N                   = N;
    args.H                   = H;
    args.total_pages         = static_cast<int>(unified_kv_nope.size(0));
    args.total_tokens        = static_cast<int>(kv_nope.size(0));
    args.stride_q_nope_n     = static_cast<int>(q_nope.stride(0));
    args.stride_q_nope_h     = static_cast<int>(q_nope.stride(1));
    args.stride_q_rope_n     = static_cast<int>(q_rope.stride(0));
    args.stride_q_rope_h     = static_cast<int>(q_rope.stride(1));
    args.stride_o_n          = static_cast<int>(out.stride(0));
    args.stride_o_h          = static_cast<int>(out.stride(1));
    args.stride_kv_nope_page = stride_kv_nope_page;
    args.stride_kv_rope_page = stride_kv_rope_page;
    args.softmax_scale       = softmax_scale;

    // ---- Launch ----------------------------------------------------------
    HipDeviceGuard guard(q_nope.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    size_t arg_size        = sizeof(args);
    const int num_h_blocks = ceil_div(H, kGfx1250HeadsPerBlk);
    const int cluster_y    = gfx1250_pick_cluster_y(num_h_blocks);

    if(cluster_y == 2)
    {
        static PrebuiltKernel impl("mla_prefill_a8w8_16mx4_64nx1_cy2", kGfx1250A8W8Co);
        impl.launch_kernel(
            {&args, &arg_size, N, num_h_blocks, 1, kGfx1250BlockSize, 1, 1, stream, 1, 2, 1});
    }
    else
    {
        static PrebuiltKernel impl("mla_prefill_a8w8_16mx4_64nx1_cy1", kGfx1250A8W8Co);
        impl.launch_kernel(
            {&args, &arg_size, N, num_h_blocks, 1, kGfx1250BlockSize, 1, 1, stream});
    }
}
