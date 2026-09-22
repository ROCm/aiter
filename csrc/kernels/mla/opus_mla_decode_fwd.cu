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
#include "opus/mla_decode_fp8_16mx8_32nx1.hpp"
#include "opus/mla_decode_kargs.h"
#include "opus/mla_decode_mxfp8_16mx8_32nx1.hpp"

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
    AITER_CHECK(page_size == 1, __func__, ": only page_size==1 supported, got ", page_size);
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

    const int H            = q->size(1);
    const int total_tokens = kv->size(0);
    const int num_workers  = work_indptr->size(0) - 1;

    const HipDeviceGuard device_guard(q->device_id);

    opus_mla_decode_fp8_kargs kargs{};
    kargs.q_buffer_ptr  = q->data_ptr();
    kargs.q_scale_ptr   = q_scale->data_ptr();
    kargs.kv_buffer_ptr = kv->data_ptr();
    kargs.kv_scale_ptr  = kv_scale->data_ptr();
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
