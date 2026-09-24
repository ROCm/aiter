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
#include "opus/mla_decode_fp8_16mx4_64nx1.hpp"
#include "opus/mla_decode_fp8_16mx8_32nx1.hpp"
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
// The one fp8 code object: the same 16mx4_64nx1 kernel that is compiled in-tree below, but
// built in OpFoundry (opus_attn/dsa_v32/gfx950, `make co`) with the amdgpu-pin-op-dst
// toolchain, which is the only way its register pins take effect -- a stock hipcc reports
// [[clang::amdgpu_pin_agpr]] as an unknown attribute and drops it. Pinning Q to a0 takes the
// build from 250 VGPR + 136 AGPR to 256 + 44. AITER_MLA_OPUS_16MX4_CO=1 picks it over the
// in-tree build so the two can be compared directly; both want metadata packed at 64 rows.
constexpr OpusDecodeVariant kFp8_16mx4{"opus_mla_decode_fp8_16mx4_64nx1_kernel",
                                       "mla_opus/opus_mla_decode_fp8_16mx4_64nx1.co",
                                       64,
                                       256};
// fp8 32mx4 / 64nx1 (csrc/kernels/mla/opus/mla_decode_fp8_32mx4_64nx1.hpp): 4 waves x 32 packed
// query rows, i.e. the same 128-row work items the 16mx8 build consumes, so it reuses that
// metadata as is. Prebuilt only: its register map (Q, half of O and the K/V ring in AGPRs, the
// other half of O and both score buffers in VGPRs) needs the pin toolchain, and a stock ROCm
// clang forces every MFMA into AGPR form, which cannot fit. One entry per specialization.
constexpr const char* kFp8_32mx4_co = "mla_opus/opus_mla_decode_fp8_32mx4_64nx1.co";
constexpr int kFp8_32mx4_rows       = 128;
constexpr int kFp8_32mx4_block      = 256;

constexpr int kHeadDimQk = 576;
constexpr int kHeadDimVo = 512;

using MxFp8Traits = opus_mla_decode_mxfp8_16mx8_32nx1_traits<16, 32, 8, fp8_t, bf16_t, bf16_t>;

// Causal is a compile-time specialization, so the caller's `causal` flag picks
// between two builds rather than steering a branch inside one. A request with a
// single query token needs no diagonal either way -- it then sits at the end of
// the KV run and masks nothing -- so max_seqlen_q == 1 keeps the build that only
// masks out-of-bounds columns even when causal is asked for.
template <bool CAUSAL, bool LARGE_KV = false>
using OpusTraitsC =
    opus_mla_decode_fp8_16mx8_32nx1_traits<16, 32, 8, fp8_t, fp8_t, bf16_t, CAUSAL, LARGE_KV>;
using OpusTraits = OpusTraitsC<false>;

// --- fp8 (16, 1): 4 waves of 32 KV tokens, Q shared through LDS ---------------------
// The 16mx8 kernel above gives every wave its own 16 query rows, so it needs nhead *
// max_seqlen_q >= 128 to fill a workgroup. At nhead 16 with one query token there are only
// 16 rows to go round, so this build tiles the KV tokens across the waves instead: four
// waves of 32 tokens, ping-pong over two LDS slots. AITER_MLA_OPUS_32NX4_SLOTS_1=1 selects
// the single-slot floor build, which has no DMA overlap but fits two blocks per CU.
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

// --- fp8 (16, 4): 4 waves of 16 query rows, 64 KV tokens per tile -------------------
// The HIP port of the SP3 16mx4_64nx1 shape. It fills a workgroup with 64 packed query rows
// where 16mx8 needs 128, so it is the build for nhead * max_seqlen_q == 64.
//
// Env-gated rather than routed by shape, because a work item here covers T_M * W_M = 64 rows
// and the metadata has to have been generated with that packing (kPackedQoLenPerWg 64) --
// hand it 128-row items and every wave but the first reads the wrong head.
template <bool CAUSAL, bool LARGE_KV = false>
using OpusTraits16mx4x64nx1C =
    opus_mla_decode_fp8_16mx4_64nx1_traits<16, 64, 4, fp8_t, fp8_t, bf16_t, CAUSAL, LARGE_KV, 3>;

struct OpusLaunch16mx4x64nx1
{
    template <class Traits>
    static void launch(int num_workers, hipStream_t stream, const opus_mla_decode_fp8_kargs& kargs)
    {
        opus_mla_decode_fp8_16mx4_64nx1_kernel<Traits>
            <<<dim3(num_workers, 1, 1), dim3(Traits::BLOCK_SIZE), 0, stream>>>(kargs);
    }
};

inline bool opus_env_flag(const char* name)
{
    const char* v = std::getenv(name);
    return v != nullptr && v[0] == '1' && v[1] == '\0';
}

inline bool opus_use_16mx1_kernel(int nhead, int max_seqlen_q)
{
    return nhead == 16 && max_seqlen_q == 1;
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

// causal x large_kv fan-out, shared by every traits-templated build (16mx1 and 16mx4).
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
    using T               = OpusTraits;
    const std::string gfx = get_gpu_arch();
    AITER_CHECK(
        gfx == "gfx950", __func__, ": unsupported GPU arch '", gfx, "' (supported: gfx950).");
    // No build here addresses a block table any more: the 16mx1 shape routes to 32nx4,
    // whose rope DMA deals a 16-token line and so cannot carry a per-lane within-page
    // offset. page_size > 1 belongs to the asm path, and the python gate keeps it there.
    const bool use_16mx1 = opus_use_16mx1_kernel(q->size(1), max_seqlen_q);
    AITER_CHECK(page_size == 1, __func__, ": only page_size == 1 is supported, got ", page_size);
    AITER_CHECK(q->size(-1) == T::D_HEAD_SIZE,
                __func__,
                ": q last dim must be ",
                T::D_HEAD_SIZE,
                " (merged nope+rope), got ",
                q->size(-1));
    AITER_CHECK(kv->size(-1) == T::D_HEAD_SIZE,
                __func__,
                ": kv last dim must be ",
                T::D_HEAD_SIZE,
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
    kargs.stride_q_b     = H * T::D_HEAD_SIZE;
    kargs.stride_q_h     = T::D_HEAD_SIZE;
    kargs.stride_o_b     = H * T::D_NOPE_SIZE;
    kargs.stride_o_h     = T::D_NOPE_SIZE;
    kargs.stride_kv_page = T::D_HEAD_SIZE;

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

    // 128 packed query rows per workgroup on 4 waves (32mx4 / 64nx1, prebuilt). Opt-in for now.
    if(opus_env_flag("AITER_MLA_OPUS_32MX4") && (H * max_seqlen_q) % kFp8_32mx4_rows == 0)
    {
        const bool use_causal = causal && max_seqlen_q > 1;
        // Split-DMA builds release a KV slot a chunk at a time, at the price of a second
        // barrier per tile; that pays once a worker has a long run of tiles (measured on
        // nhead*qlen 128: +2..4% from ~32 tiles per worker, -1.5..3% below). The KV token count
        // over the tile size and the worker count gives that run length. Large-KV builds are
        // whole-tile only. AITER_MLA_OPUS_32MX4_SPLIT=0/1 overrides the heuristic.
        const int64_t tiles_per_worker =
            static_cast<int64_t>(kv_indices->numel()) / 64 / std::max(num_workers, 1);
        const char* split_env = std::getenv("AITER_MLA_OPUS_32MX4_SPLIT");
        const bool split = split_env != nullptr ? (split_env[0] == '1') : tiles_per_worker >= 32;
        static bool traced = false; // first call only: the test loops the kernel
        if(!traced && opus_env_flag("AITER_MLA_OPUS_TRACE") && (traced = true))
            fprintf(stderr,
                    "[opus_mla_decode_fp8] 32mx4_64nx1 co causal=%d large_kv=%d split=%d "
                    "(tiles/worker %ld)\n",
                    use_causal,
                    large_kv,
                    split && !large_kv,
                    static_cast<long>(tiles_per_worker));
        size_t arg_size = sizeof(kargs);
        auto launch     = [&](AiterAsmKernel& impl) {
            impl.launch_kernel({&kargs, &arg_size, num_workers, 1, 1, kFp8_32mx4_block, 1, 1,
                                stream});
        };
#define MLA32MX4_LAUNCH(entry)                                                        \
    do                                                                                \
    {                                                                                 \
        static AiterAsmKernel impl("opus_mla_decode_fp8_32mx4_64nx1_" entry, kFp8_32mx4_co); \
        launch(impl);                                                                 \
    } while(0)
        if(large_kv)
        {
            if(use_causal)
                MLA32MX4_LAUNCH("causal_large");
            else
                MLA32MX4_LAUNCH("nc_large");
        }
        else if(use_causal)
        {
            if(split)
                MLA32MX4_LAUNCH("causal");
            else
                MLA32MX4_LAUNCH("causal_ns");
        }
        else
        {
            if(split)
                MLA32MX4_LAUNCH("nc");
            else
                MLA32MX4_LAUNCH("nc_ns");
        }
#undef MLA32MX4_LAUNCH
        return;
    }

    // 64 packed query rows per workgroup. Opt-in: the metadata must have been generated with
    // the matching packing, which the shape check below only makes plausible, not certain.
    if(opus_env_flag("AITER_MLA_OPUS_16MX4") && H * max_seqlen_q == 64)
    {
        // The prebuilt, register-pinned build of this same kernel. Only the non-causal,
        // small-KV instantiation is in the code object, so anything else stays in-tree.
        if(opus_env_flag("AITER_MLA_OPUS_16MX4_CO") && !(causal && max_seqlen_q > 1) &&
           !large_kv)
        {
            size_t arg_size = sizeof(kargs);
            static AiterAsmKernel impl(kFp8_16mx4.kernel_name, kFp8_16mx4.co_path);
            impl.launch_kernel({&kargs, &arg_size, num_workers, 1, 1,
                                kFp8_16mx4.block_size, 1, 1, stream});
            return;
        }
        launch_opus_16mx1<OpusTraits16mx4x64nx1C, OpusLaunch16mx4x64nx1>(
            causal, max_seqlen_q, large_kv, num_workers, stream, kargs);
        return;
    }

    if(use_16mx1)
    {
        // The floor build, for A/B only: one slot, so a tile's whole gmem latency sits in
        // the open. Measured 13 to 15% slower than the ring on the memory-bound shapes.
        static const bool want_1slot = opus_env_flag("AITER_MLA_OPUS_32NX4_SLOTS_1");

        if(want_1slot)
            launch_opus_16mx1<OpusTraits16mx1x32nx4S1C, OpusLaunch16mx1x32nx4>(
                causal, max_seqlen_q, large_kv, num_workers, stream, kargs);
        else
            launch_opus_16mx1<OpusTraits16mx1x32nx4C, OpusLaunch16mx1x32nx4>(
                causal, max_seqlen_q, large_kv, num_workers, stream, kargs);
        return;
    }

    auto launch = [&](auto traits) {
        opus_mla_decode_fp8_16mx8_32nx1_kernel<decltype(traits)>
            <<<dim3(num_workers, 1, 1), dim3(T::BLOCK_SIZE), 0, stream>>>(kargs);
    };
    if(causal && max_seqlen_q > 1)
    {
        if(large_kv)
            launch(OpusTraitsC<true, true>{});
        else
            launch(OpusTraitsC<true, false>{});
    }
    else
    {
        if(large_kv)
            launch(OpusTraitsC<false, true>{});
        else
            launch(OpusTraitsC<false, false>{});
    }
}
