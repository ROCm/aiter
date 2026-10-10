// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Stage 1 of the opus MLA decode kernels. These launch ONLY the decode kernel:
// work partitioning comes from aiter's metadata (work_indptr / work_info_set)
// and the split-KV merge is left to mla_reduce_v1. Per-split partials go to
// logits/attn_lse (== aiter split_output/split_lse); work items that own a whole
// request (partial_slot < 0) write the final output directly. gfx950 only.
//
// Three variants, all in this translation unit:
//   opus_mla_decode_fwd        bf16 Q/KV, merged d=576. Prebuilt code objects
//                              under hsa/<arch>/mla_opus/, because they need a
//                              toolchain aiter does not ship.
//   opus_mla_decode_mxfp8_fwd  fp8 NoPE with per-block E8M0 scales + bf16 RoPE,
//                              so Q and KV are three tensors each.
//   opus_mla_decode_fp8_fwd    fp8 NoPE and RoPE merged into one d=576 buffer,
//                              with per-tensor scalar descales.
//
// Bound to Python through the torch-free ctypes C ABI (aiter_tensor_t* +
// trailing hipStream_t), so this .so carries no pybind/libtorch dependency.

#include "aiter_tensor.h"

#include "aiter_ctypes_error.h"

#include "aiter_hip_common.h"
#include "opus/mla_decode_fp8_16mx1_32nx4.hpp"
#include "opus/mla_decode_kargs.h"
#include "opus/mla_decode_mxfp8_16mx8_32nx1.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <string>

AITER_CTYPES_ERROR_DEF

namespace {

// Fixed properties of the prebuilt code objects.
struct OpusDecodeVariant
{
    const char* kernel_name;
    const char* co_path;
    int heads_per_block;
    int block_size;
};

constexpr OpusDecodeVariant kA16W16_16mx4{"opus_mla_decode_a16w16_16mx4_64nx1_kernel",
                                       "mla_opus/opus_mla_decode_a16w16_16mx4_64nx1.co",
                                       64,
                                       256};
constexpr OpusDecodeVariant kA16W16_32mx1{"opus_mla_decode_a16w16_32mx1_16nx4_kernel",
                                       "mla_opus/opus_mla_decode_a16w16_32mx1_16nx4.co",
                                       32,
                                       256};
constexpr OpusDecodeVariant kA16W16_32mx4{"opus_mla_decode_a16w16_32mx4_32nx1_kernel",
                                       "mla_opus/opus_mla_decode_a16w16_32mx4_32nx1.co",
                                       128,
                                       256};
constexpr OpusDecodeVariant kA16W16_32mx3{"opus_mla_decode_a16w16_32mx3_32nx1_kernel",
                                       "mla_opus/opus_mla_decode_a16w16_32mx3_32nx1.co",
                                       96,
                                       256};
// fp8 32mx4 / 64nx1 (csrc/kernels/mla/opus/mla_decode_fp8_32mx4_64nx1.hpp): 4 waves x 32 packed
// query rows, i.e. the 128-row work items aiter's metadata packs by default for fp8. Prebuilt
// only: its register map (Q, half of O and the K/V ring in AGPRs, the other half of O and both
// score buffers in VGPRs) needs the pin toolchain, and a stock ROCm clang forces every MFMA into
// AGPR form, which cannot fit. One entry per specialization.
constexpr const char* kFp8_32mx4_co = "mla_opus/opus_mla_decode_fp8_32mx4_64nx1.co";
// fp8 16mx4 / 64nx1 (mla_decode_fp8_16mx4_64nx1.hpp, the HIP port of the SP3 16mx4_64nx1
// shape), prebuilt the same way (mla_decode_fp8_16mx4_64nx1_co.hip): a stock hipcc drops its
// register pins. 64 packed query rows per workgroup, so a work item must not carry more than
// 64 rows -- hand it a 128-row item and every wave but the first reads the wrong head --
// which the metadata guarantees for exactly the shapes opus_fp8_route sends here (64 rows or
// fewer per request, and kPackedQoLenPerWg 64 for nhead 32 x 3).
constexpr const char* kFp8_16mx4_co = "mla_opus/opus_mla_decode_fp8_16mx4_64nx1.co";
constexpr int kFp8_co_block         = 256;

constexpr int kHeadDimQk = 576;
constexpr int kHeadDimVo = 512;

using MxFp8Traits = opus_mla_decode_mxfp8_16mx8_32nx1_traits<16, 32, 8, fp8_t, bf16_t, bf16_t>;

// --- fp8 (16, 1): 4 waves of 32 KV tokens, Q shared through LDS ---------------------
// The row-tiled builds give every wave its own query rows, so they need nhead * max_seqlen_q
// rows to fill a workgroup. At nhead 16 with one query token there are only 16 rows to go
// round, so this build tiles the KV tokens across the waves instead: four waves of 32
// tokens, ping-pong over two LDS slots. AITER_MLA_OPUS_32NX4_SLOTS_1=1 selects the
// single-slot floor build, which has no DMA overlap but fits two blocks per CU.
//
// page_size 1 only -- the rope DMA deals a 16-token line, so the within-page token offset
// would be per-lane above 1. The 16nx8 geometry this replaced is the only fp8 build that
// ever addressed a real block table; it still sits in opus/ but nothing routes to it, and
// the python gate now keeps page_size > 1 on the asm path.
template <bool CAUSAL, bool LARGE_KV = false>
using OpusTraits16mx1x32nx4C =
    opus_mla_decode_fp8_16mx1_32nx4_traits<16, 32, 4, fp8_t, fp8_t, bf16_t, CAUSAL, LARGE_KV, 2>;
template <bool CAUSAL, bool LARGE_KV = false>
using OpusTraits16mx1x32nx4S1C =
    opus_mla_decode_fp8_16mx1_32nx4_traits<16, 32, 4, fp8_t, fp8_t, bf16_t, CAUSAL, LARGE_KV, 1>;
inline bool opus_env_flag(const char* name)
{
    const char* v = std::getenv(name);
    return v != nullptr && v[0] == '1' && v[1] == '\0';
}

// Shape routing for the merged-buffer fp8 path, mirroring the asm persistent fp8 heuristic
// (csrc/py_itfs_cu/asm_mla.cu): what asm runs on its qh32 / qseqlen4 kernel (128 packed rows
// per workgroup) goes to 32mx4, what it runs on qh16 / qseqlen4 (64 rows) to 16mx4. nhead
// here is nhead per KV head (MLA has one), after mla.py's fold of the non-native head counts.
// aiter/mla.py keeps the same table so a shape with no build stays on the asm path.
enum class OpusFp8Route
{
    k32mx4,
    k16mx4,
    k16mx1,
    kNone,
};

inline OpusFp8Route opus_fp8_route(int nhead, int max_seqlen_q)
{
    const int q = max_seqlen_q;
    if((nhead == 32 && q >= 4) || (nhead == 64 && q >= 2) || nhead == 128 ||
       (nhead == 96 && q <= 6))
        return OpusFp8Route::k32mx4;
    if((nhead == 16 && (q == 3 || q == 4)) || (nhead == 32 && (q == 2 || q == 3)) ||
       (nhead == 64 && q == 1))
        return OpusFp8Route::k16mx4;
    if(nhead == 16 && q == 1)
        return OpusFp8Route::k16mx1;
    return OpusFp8Route::kNone;
}

struct OpusLaunch16mx1x32nx4
{
    template <class Traits>
    static void
    launch(int num_workers, hipStream_t stream, const opus_mla_decode_fp8_kargs& kargs)
    {
        opus_mla_decode_fp8_16mx1_32nx4_kernel<Traits>
            <<<dim3(num_workers, 1, 1), dim3(Traits::BLOCK_SIZE), 0, stream>>>(kargs);
    }
};

// causal x large_kv fan-out for the traits-templated in-tree build (16mx1).
template <template <bool, bool> class TraitsC, class Launcher>
void launch_opus_16mx1(bool causal,
                       int max_seqlen_q,
                       bool large_kv,
                       int num_workers,
                       hipStream_t stream,
                       const opus_mla_decode_fp8_kargs& kargs)
{
    const bool use_causal = causal && max_seqlen_q > 1;
    auto launch           = [&](auto traits) {
        Launcher::template launch<decltype(traits)>(num_workers, stream, kargs);
    };
    if(use_causal && large_kv)
        launch(TraitsC<true, true>{});
    else if(use_causal)
        launch(TraitsC<true, false>{});
    else if(large_kv)
        launch(TraitsC<false, true>{});
    else
        launch(TraitsC<false, false>{});
}

} // namespace

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    opus_mla_decode_fwd,
    (aiter_tensor_t * q,             // [total_q, H, 576] bf16
     aiter_tensor_t* kv,             // [num_page, 1, 1, 576] bf16, page_size == 1
     aiter_tensor_t* qo_indptr,      // [B+1]
     aiter_tensor_t* kv_indptr,      // [B+1]
     aiter_tensor_t* kv_indices,     // [num_page_used]
     aiter_tensor_t* work_indptr,    // metadata
     aiter_tensor_t* work_info_set,  // metadata
     int page_size,
     float softmax_scale,
     aiter_tensor_t* logits,         // split_output [num_partials, 1, H, 512] fp32
     aiter_tensor_t* attn_lse,       // split_lse    [num_partials, 1, H, 1]   fp32
     aiter_tensor_t* out,            // final [total_q, H, 512] bf16
     aiter_tensor_t* final_lse,      // [total_q, H] fp32 (nullable)
     hipStream_t stream),
    (q,
     kv,
     qo_indptr,
     kv_indptr,
     kv_indices,
     work_indptr,
     work_info_set,
     page_size,
     softmax_scale,
     logits,
     attn_lse,
     out,
     final_lse,
     stream))
{
    const std::string gfx = get_gpu_arch();
    AITER_CHECK(
        gfx == "gfx950", __func__, ": unsupported GPU arch '", gfx, "' (supported: gfx950).");
    AITER_CHECK(page_size == 1, __func__, ": only page_size == 1 is supported.");
    AITER_CHECK(q->size(-1) == kHeadDimQk, __func__, ": q last dim must be ", kHeadDimQk, ".");
    AITER_CHECK(kv->size(-1) == kHeadDimQk, __func__, ": kv last dim must be ", kHeadDimQk, ".");
    AITER_CHECK(out->size(-1) == kHeadDimVo, __func__, ": out last dim must be ", kHeadDimVo, ".");

    const int H           = q->size(1);
    const int total_tokens = kv->size(0);
    const int num_workers = work_indptr->size(0) - 1;

    const OpusDecodeVariant& variant = (H <= kA16W16_32mx1.heads_per_block) ? kA16W16_32mx1
                                       : (H % kA16W16_32mx4.heads_per_block == 0) ? kA16W16_32mx4
                                       : (H % kA16W16_32mx3.heads_per_block == 0) ? kA16W16_32mx3
                                                                                  : kA16W16_16mx4;
    const int num_h_blocks = (H + variant.heads_per_block - 1) / variant.heads_per_block;

    const HipDeviceGuard device_guard(q->device_id);

    opus_mla_decode_kargs kargs{};
    kargs.q_ptr         = q->data_ptr();
    kargs.kv_ptr        = kv->data_ptr();
    kargs.out_ptr   = out->data_ptr();
    kargs.lse_ptr   = (final_lse && final_lse->numel() > 0) ? final_lse->data_ptr() : nullptr;
    kargs.o_accum   = logits->data_ptr();
    kargs.lse_accum = attn_lse->data_ptr();
    kargs.q_indptr      = static_cast<const int*>(qo_indptr->data_ptr());
    kargs.kv_indptr     = static_cast<const int*>(kv_indptr->data_ptr());
    kargs.kv_indices    = static_cast<const int*>(kv_indices->data_ptr());
    kargs.work_indptr   = static_cast<const int*>(work_indptr->data_ptr());
    kargs.work_info_set =
        static_cast<const opus_mla_decode_work_info*>(work_info_set->data_ptr());
    kargs.H             = H;
    kargs.total_tokens  = total_tokens;
    kargs.softmax_scale = softmax_scale;
    kargs.stride_q_b     = H * kHeadDimQk;
    kargs.stride_q_h     = kHeadDimQk;
    kargs.stride_o_b     = H * kHeadDimVo;
    kargs.stride_o_h     = kHeadDimVo;
    kargs.stride_kv_page = kHeadDimQk;

    size_t arg_size = sizeof(kargs);
    auto launch     = [&](AiterAsmKernel& impl) {
        impl.launch_kernel({&kargs, &arg_size, num_workers, num_h_blocks, 1,
                            variant.block_size, 1, 1, stream});
    };
    if(&variant == &kA16W16_32mx1)
    {
        static AiterAsmKernel impl(kA16W16_32mx1.kernel_name, kA16W16_32mx1.co_path);
        launch(impl);
    }
    else if(&variant == &kA16W16_32mx4)
    {
        static AiterAsmKernel impl(kA16W16_32mx4.kernel_name, kA16W16_32mx4.co_path);
        launch(impl);
    }
    else if(&variant == &kA16W16_32mx3)
    {
        static AiterAsmKernel impl(kA16W16_32mx3.kernel_name, kA16W16_32mx3.co_path);
        launch(impl);
    }
    else
    {
        static AiterAsmKernel impl(kA16W16_16mx4.kernel_name, kA16W16_16mx4.co_path);
        launch(impl);
    }
}

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    opus_mla_decode_mxfp8_fwd,
    (aiter_tensor_t * q_nope,   // [total_q, H, 512]         fp8
     aiter_tensor_t* q_scale,   // [total_q, H, D_SCALE]     uint8 (E8M0)
     aiter_tensor_t* q_rope,    // [total_q, H, 64]          bf16
     aiter_tensor_t* kv_nope,   // [total_tokens, 512]       fp8
     aiter_tensor_t* kv_scale,  // [total_tokens, D_SCALE]   uint8 (E8M0)
     aiter_tensor_t* kv_rope,   // [total_tokens, 64]        bf16
     aiter_tensor_t* qo_indptr,
     aiter_tensor_t* kv_indptr,
     aiter_tensor_t* kv_indices,
     aiter_tensor_t* work_indptr,
     aiter_tensor_t* work_info_set,
     int page_size,
     float softmax_scale,
     aiter_tensor_t* logits,
     aiter_tensor_t* attn_lse,
     aiter_tensor_t* out,
     aiter_tensor_t* final_lse,
     hipStream_t stream),
    (q_nope,
     q_scale,
     q_rope,
     kv_nope,
     kv_scale,
     kv_rope,
     qo_indptr,
     kv_indptr,
     kv_indices,
     work_indptr,
     work_info_set,
     page_size,
     softmax_scale,
     logits,
     attn_lse,
     out,
     final_lse,
     stream))
{
    using T = MxFp8Traits;
    const std::string gfx = get_gpu_arch();
    AITER_CHECK(
        gfx == "gfx950", __func__, ": unsupported GPU arch '", gfx, "' (supported: gfx950).");
    AITER_CHECK(page_size == 1, __func__, ": only page_size == 1 is supported.");
    AITER_CHECK(q_nope->dtype() == AITER_DTYPE_fp8 && kv_nope->dtype() == AITER_DTYPE_fp8,
                __func__,
                ": q_nope/kv_nope must be fp8.");
    AITER_CHECK(q_rope->dtype() == AITER_DTYPE_bf16 && kv_rope->dtype() == AITER_DTYPE_bf16,
                __func__,
                ": q_rope/kv_rope must be bf16.");
    // The kernel reads the scales as bit_cast<float>(e8m0 << 23); fp32 factors must be
    // converted on the host first.
    AITER_CHECK(q_scale->dtype() == AITER_DTYPE_u8 && kv_scale->dtype() == AITER_DTYPE_u8,
                __func__,
                ": q_scale/kv_scale must be E8M0 uint8.");
    AITER_CHECK(kv_scale->size(-1) == T::D_SCALE_SIZE,
                __func__,
                ": kv_scale last dim must be ",
                T::D_SCALE_SIZE,
                ".");

    const int H           = q_nope->size(1);
    const int total_tokens = kv_nope->size(0);
    const int num_workers = work_indptr->size(0) - 1;

    const HipDeviceGuard device_guard(q_nope->device_id);

    opus_mla_decode_mxfp8_kargs kargs{};
    kargs.q_nope_ptr    = q_nope->data_ptr();
    kargs.q_scale_ptr   = q_scale->data_ptr();
    kargs.q_rope_ptr    = q_rope->data_ptr();
    kargs.kv_nope_ptr   = kv_nope->data_ptr();
    kargs.kv_scale_ptr  = kv_scale->data_ptr();
    kargs.kv_rope_ptr   = kv_rope->data_ptr();
    kargs.out_ptr   = out->data_ptr();
    kargs.lse_ptr   = (final_lse && final_lse->numel() > 0) ? final_lse->data_ptr() : nullptr;
    kargs.o_accum   = logits->data_ptr();
    kargs.lse_accum = attn_lse->data_ptr();
    kargs.q_indptr      = static_cast<const int*>(qo_indptr->data_ptr());
    kargs.kv_indptr     = static_cast<const int*>(kv_indptr->data_ptr());
    kargs.kv_indices    = static_cast<const int*>(kv_indices->data_ptr());
    kargs.work_indptr   = static_cast<const int*>(work_indptr->data_ptr());
    kargs.work_info_set =
        static_cast<const opus_mla_decode_work_info*>(work_info_set->data_ptr());
    kargs.H             = H;
    kargs.total_tokens  = total_tokens;
    kargs.softmax_scale = softmax_scale;
    kargs.stride_q_nope_b      = H * T::D_NOPE_SIZE;
    kargs.stride_q_nope_h      = T::D_NOPE_SIZE;
    kargs.stride_q_scale_b     = H * T::D_SCALE_SIZE;
    kargs.stride_q_scale_h     = T::D_SCALE_SIZE;
    kargs.stride_q_rope_b      = H * T::D_ROPE_SIZE;
    kargs.stride_q_rope_h      = T::D_ROPE_SIZE;
    kargs.stride_o_b           = H * T::D_NOPE_SIZE;
    kargs.stride_o_h           = T::D_NOPE_SIZE;
    kargs.stride_kv_nope_page  = T::D_NOPE_SIZE;
    kargs.stride_kv_scale_page = T::D_SCALE_SIZE;
    kargs.stride_kv_rope_page  = T::D_ROPE_SIZE;

    opus_mla_decode_mxfp8_16mx8_32nx1_kernel<T>
        <<<dim3(num_workers, 1, 1), dim3(T::BLOCK_SIZE), 0, stream>>>(kargs);
}

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    opus_mla_decode_fp8_fwd,
    (aiter_tensor_t * q,                // [B, H, 576]           fp8 (merged nope+rope)
     aiter_tensor_t* kv,                // [total_tokens, 576]   fp8 (merged nope+rope)
     aiter_tensor_t* qo_indptr,         // [B+1]
     aiter_tensor_t* kv_indptr,         // [B+1]
     aiter_tensor_t* kv_indices,        // [total_tokens]
     aiter_tensor_t* kv_last_page_lens, // [B] -- unused, page_size is 1
     aiter_tensor_t* work_indptr,       // metadata
     aiter_tensor_t* work_info_set,     // metadata
     int max_seqlen_q,
     int page_size,
     int nhead_kv, // unused, kept for API parity
     float softmax_scale,
     aiter_tensor_t* logits,    // aiter split_output [num_partials,1,H,512] fp32
     aiter_tensor_t* attn_lse,  // aiter split_lse    [num_partials,1,H,1]   fp32
     aiter_tensor_t* out,       // final [B, H, 512] bf16
     aiter_tensor_t* final_lse, // [B, H] fp32 (nullable)
     aiter_tensor_t* q_scale,   // float[1] per-tensor descale
     aiter_tensor_t* kv_scale,  // float[1] per-tensor descale
     int causal,                // apply the causal mask across the query tokens
     hipStream_t stream),
    (q,
     kv,
     qo_indptr,
     kv_indptr,
     kv_indices,
     kv_last_page_lens,
     work_indptr,
     work_info_set,
     max_seqlen_q,
     page_size,
     nhead_kv,
     softmax_scale,
     logits,
     attn_lse,
     out,
     final_lse,
     q_scale,
     kv_scale,
     causal,
     stream))
{
    const std::string gfx = get_gpu_arch();
    AITER_CHECK(
        gfx == "gfx950", __func__, ": unsupported GPU arch '", gfx, "' (supported: gfx950).");
    // No build here addresses a block table any more: the 16mx1 shape routes to 32nx4,
    // whose rope DMA deals a 16-token line and so cannot carry a per-lane within-page
    // offset. page_size > 1 belongs to the asm path, and the python gate keeps it there.
    AITER_CHECK(page_size == 1, __func__, ": only page_size == 1 is supported, got ", page_size);
    const OpusFp8Route route = opus_fp8_route(q->size(1), max_seqlen_q);
    AITER_CHECK(route != OpusFp8Route::kNone,
                __func__,
                ": no opus fp8 build for nhead ",
                q->size(1),
                " x max_seqlen_q ",
                max_seqlen_q,
                " (the python gate keeps such shapes on the asm path)");
    AITER_CHECK(q->size(-1) == kHeadDimQk,
                __func__,
                ": q last dim must be ",
                kHeadDimQk,
                " (merged nope+rope), got ",
                q->size(-1));
    AITER_CHECK(kv->size(-1) == kHeadDimQk,
                __func__,
                ": kv last dim must be ",
                kHeadDimQk,
                " (merged nope+rope), got ",
                kv->size(-1));
    AITER_CHECK(q_scale != nullptr && q_scale->dtype() == AITER_DTYPE_fp32 && q_scale->numel() >= 1,
                __func__,
                ": q_scale must be a float scalar tensor");
    AITER_CHECK(kv_scale != nullptr && kv_scale->dtype() == AITER_DTYPE_fp32 &&
                    kv_scale->numel() >= 1,
                __func__,
                ": kv_scale must be a float scalar tensor");

    const int H = q->size(1);
    // kv is [total_tokens, D] at page_size 1 and [num_page, page_size, 1, D] above it, so
    // dim 0 counts pages either way and the token count is that times the page size.
    const int total_tokens = static_cast<int>(kv->size(0)) * page_size;
    const int num_workers  = work_indptr->size(0) - 1;

    const HipDeviceGuard device_guard(q->device_id);

    opus_mla_decode_fp8_kargs kargs{};
    kargs.q_buffer_ptr  = q->data_ptr();
    kargs.q_scale_ptr   = q_scale->data_ptr();
    kargs.kv_buffer_ptr = kv->data_ptr();
    kargs.kv_scale_ptr  = kv_scale->data_ptr();
    // Only a paged build would read it, and there is none left; null so nothing can quietly
    // start depending on it.
    kargs.kv_last_page_lens = nullptr;
    kargs.out_ptr   = out->data_ptr();
    kargs.lse_ptr   = (final_lse && final_lse->numel() > 0) ? final_lse->data_ptr() : nullptr;
    kargs.o_accum   = logits->data_ptr();
    kargs.lse_accum = attn_lse->data_ptr();
    kargs.q_indptr      = static_cast<const int*>(qo_indptr->data_ptr());
    kargs.kv_indptr     = static_cast<const int*>(kv_indptr->data_ptr());
    kargs.kv_indices    = static_cast<const int*>(kv_indices->data_ptr());
    kargs.work_indptr   = static_cast<const int*>(work_indptr->data_ptr());
    kargs.work_info_set =
        static_cast<const opus_mla_decode_work_info*>(work_info_set->data_ptr());
    kargs.H             = H;
    kargs.total_tokens  = total_tokens;
    kargs.softmax_scale = softmax_scale;

    // Merged d=576 buffer: one row per (token, head); rope is the +D_NOPE slice.
    kargs.stride_q_b     = H * kHeadDimQk;
    kargs.stride_q_h     = kHeadDimQk;
    kargs.stride_o_b     = H * kHeadDimVo;
    kargs.stride_o_h     = kHeadDimVo;
    kargs.stride_kv_page = kHeadDimQk;

    // A buffer descriptor's num_records is 32 bits, so it cannot span a KV cache of 4 GiB
    // or more; past that the bound wraps and every load beyond it silently returns zero.
    // Unlike the contiguous fmha case there is no way to rebase the descriptor per tile
    // here: page_size is 1, so one KV tile's 32 tokens sit at unrelated, per-lane offsets
    // while a descriptor base is wave-uniform. The large path addresses KV with a flat
    // 64-bit pointer (global_load_lds) instead, which costs ~1-2%, hence the gate.
    const int64_t kv_bytes = static_cast<int64_t>(total_tokens) *
                             static_cast<int64_t>(kargs.stride_kv_page) *
                             static_cast<int64_t>(sizeof(fp8_t));
    const bool large_kv = kv_bytes >= (int64_t{1} << 32);

    // The prebuilt builds: entry <name>_{nc,causal}[_large] of one code object each, every
    // entry loaded once on first use.
    const bool use_causal = causal && max_seqlen_q > 1;
    size_t arg_size       = sizeof(kargs);
    auto launch_co        = [&](AiterAsmKernel& impl) {
        impl.launch_kernel({&kargs, &arg_size, num_workers, 1, 1, kFp8_co_block, 1, 1, stream});
    };
#define OPUS_FP8_CO_ENTRY(name, entry, co)                                             \
    do                                                                                 \
    {                                                                                  \
        static AiterAsmKernel impl("opus_mla_decode_fp8_" name "_" entry, co);         \
        launch_co(impl);                                                               \
    } while(0)
#define OPUS_FP8_CO_LAUNCH(name, co)                                                   \
    do                                                                                 \
    {                                                                                  \
        static bool traced = false; /* first call only: the test loops the kernel */  \
        if(!traced && opus_env_flag("AITER_MLA_OPUS_TRACE") && (traced = true))        \
            fprintf(stderr,                                                            \
                    "[opus_mla_decode_fp8] " name " co causal=%d large_kv=%d\n",       \
                    use_causal,                                                        \
                    large_kv);                                                         \
        if(large_kv && use_causal)                                                     \
            OPUS_FP8_CO_ENTRY(name, "causal_large", co);                               \
        else if(large_kv)                                                              \
            OPUS_FP8_CO_ENTRY(name, "nc_large", co);                                   \
        else if(use_causal)                                                            \
            OPUS_FP8_CO_ENTRY(name, "causal", co);                                     \
        else                                                                           \
            OPUS_FP8_CO_ENTRY(name, "nc", co);                                         \
    } while(0)

    // 128 packed query rows per workgroup on 4 waves.
    if(route == OpusFp8Route::k32mx4)
    {
        OPUS_FP8_CO_LAUNCH("32mx4_64nx1", kFp8_32mx4_co);
        return;
    }

    // 64 packed query rows per workgroup on 4 waves. The pinned code object measured 2% faster
    // than the same source through aiter's stock JIT, which drops the pins.
    if(route == OpusFp8Route::k16mx4)
    {
        OPUS_FP8_CO_LAUNCH("16mx4_64nx1", kFp8_16mx4_co);
        return;
    }
#undef OPUS_FP8_CO_LAUNCH
#undef OPUS_FP8_CO_ENTRY

    // route == k16mx1. The floor build, for A/B only: one slot, so a tile's whole gmem latency
    // sits in the open. Measured 13 to 15% slower than the ring on the memory-bound shapes.
    static const bool want_1slot = opus_env_flag("AITER_MLA_OPUS_32NX4_SLOTS_1");
    if(want_1slot)
        launch_opus_16mx1<OpusTraits16mx1x32nx4S1C, OpusLaunch16mx1x32nx4>(
            causal, max_seqlen_q, large_kv, num_workers, stream, kargs);
    else
        launch_opus_16mx1<OpusTraits16mx1x32nx4C, OpusLaunch16mx1x32nx4>(
            causal, max_seqlen_q, large_kv, num_workers, stream, kargs);
}
