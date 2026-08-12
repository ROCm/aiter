// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// aiter stage1 of the opus merged-buffer MLA decode kernel
// (mla_decode_fwd_16mx8_32nx1_fp8fp8_ps_opus.hpp). Same integration shape as
// ds32_decode_fwd.cu / hk_decode_fwd.cu: this launches ONLY the decode kernel,
// reusing aiter's metadata (work_indptr / work_info_set) and reduce
// (mla_reduce_v1). Per-split partials go to logits/attn_lse (== aiter
// split_output/split_lse), or, for no-split work items (partial_qo_loc < 0),
// straight to the final output. gfx950 only.
//
// Differences vs. dsa_v32 (ds32_decode_fwd.cu):
//   * q / kv are a *single* contiguous d = 576 fp8 tensor (nope [0,512) + rope
//     [512,576)) -- no separate nope/rope tensors.
//   * q_scale / kv_scale are per-tensor scalar float descales (a single float
//     each), not per-block E8M0 uint8.
//
// Bound to Python through the torch-free ctypes C ABI (aiter_tensor_t* +
// trailing hipStream_t), like mla_decode_stage1_asm_fwd in asm_mla.cu, so this
// .so carries no pybind/libtorch dependency.

#include "aiter_tensor.h"

// Must follow aiter_tensor.h: aiter_safe_call names aiter_detail::g_aiter_can_throw,
// declared by aiter_hip_common.h. The blank line above stops clang-format from
// sorting the two into one block and swapping them.
#include "aiter_ctypes_error.h"

#include "opus/mla_decode_fwd_16mx8_32nx1_fp8fp8_ps_opus.hpp"

#include <hip/hip_runtime.h>
#include <string>

// Per-.so TLS error storage + aiter_get_last_error / aiter_clear_last_error
// exports, so AITER_CHECK failures below surface as a Python RuntimeError
// instead of aborting the worker process.
AITER_CTYPES_ERROR_DEF

// Causal is a compile-time specialization, so the caller's `causal` flag picks
// between two builds rather than steering a branch inside one. A request with a
// single query token needs no diagonal either way -- it then sits at the end of
// the KV run and masks nothing -- so max_seqlen_q == 1 keeps the build that only
// masks out-of-bounds columns even when causal is asked for.
template <bool CAUSAL, bool LARGE_KV = false>
using OpusTraitsC =
    mla_16mx8_32nx1_fp8fp8_ps_traits<16, 32, 8, fp8_t, fp8_t, bf16_t, CAUSAL, LARGE_KV>;
using OpusTraits = OpusTraitsC<false>;

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    mla_decode_fwd_opus_stage1,
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

    mla_kargs kargs{};
    kargs.q_buffer_ptr  = q->data_ptr();
    kargs.q_scale_ptr   = q_scale->data_ptr();
    kargs.kv_buffer_ptr = kv->data_ptr();
    kargs.kv_scale_ptr  = kv_scale->data_ptr();
    kargs.o_accum       = logits->data_ptr();   // aiter split_output
    kargs.lse_accum     = attn_lse->data_ptr(); // aiter split_lse
    kargs.out_ptr       = out->data_ptr();
    kargs.lse_ptr       = (final_lse && final_lse->numel() > 0) ? final_lse->data_ptr() : nullptr;
    kargs.q_indptr      = static_cast<const int*>(qo_indptr->data_ptr());
    kargs.kv_indptr     = static_cast<const int*>(kv_indptr->data_ptr());
    kargs.kv_indices    = static_cast<const int*>(kv_indices->data_ptr());
    kargs.work_indptr   = static_cast<const int*>(work_indptr->data_ptr());
    kargs.work_info_set = static_cast<const int*>(work_info_set->data_ptr());
    kargs.H             = H;
    kargs.total_tokens  = total_tokens;

    // Merged d=576 buffer: one row per (token, head); rope is the +D_NOPE slice.
    kargs.stride_q_b     = H * T::D_HEAD_SIZE;
    kargs.stride_q_h     = T::D_HEAD_SIZE;
    kargs.stride_o_b     = H * T::D_NOPE_SIZE;
    kargs.stride_o_h     = T::D_NOPE_SIZE;
    kargs.stride_kv_page = T::D_HEAD_SIZE;
    kargs.softmax_scale  = softmax_scale;

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
        mla_decode_fwd_16mx8_32nx1_fp8fp8_opus_kernel<decltype(traits)>
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
