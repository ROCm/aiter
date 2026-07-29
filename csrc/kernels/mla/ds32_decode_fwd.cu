// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// aiter stage1 of the DSA v3.2 (OpFoundry opus_attn/dsa_v32) MLA decode kernel.
// Reuses aiter's metadata (work_indptr / work_info_set) and reduce (mla_reduce_v1):
// this launches ONLY the decode kernel, which writes per-split partial outputs to
// logits/attn_lse (== aiter split_output/split_lse) or, for no-split work items
// (partial_qo_loc < 0), directly to the final output. gfx950 only.

#include <hip/hip_runtime.h>

#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "ds32/dsa_v32_splitkv.hpp"
#include "mla.h"
#include "mla_opus.h"

using Ds32Traits = dsa_v32_16mx8_32nx1_fp8_traits<16, 32, 8, fp8_t, bf16_t, bf16_t>;

// q_nope  : [B, H, D_NOPE]         fp8
// q_scale : [B, H, D_SCALE]        uint8 (E8M0)
// q_rope  : [B, H, D_ROPE]         bf16
// kv_nope : [total_tokens, D_NOPE] fp8
// kv_scale: [total_tokens, D_SCALE] uint8
// kv_rope : [total_tokens, D_ROPE]  bf16
// logits  : [num_partials, 1, H, D_NOPE] fp32  (aiter split_output)
// attn_lse: [num_partials, 1, H, 1]      fp32  (aiter split_lse)
// o       : [B, H, D_NOPE] bf16 (final, for no-split work items)
void mla_decode_stage1_opus_fwd_ds32(aiter_tensor_t& q_nope,
                                     aiter_tensor_t& q_rope,
                                     aiter_tensor_t& kv_nope,
                                     aiter_tensor_t& kv_rope,
                                     const aiter_tensor_t& qo_indptr,
                                     const aiter_tensor_t& kv_indptr,
                                     const aiter_tensor_t& kv_indices,
                                     const aiter_tensor_t& kv_last_page_lens,
                                     const aiter_tensor_t& work_indptr,
                                     const aiter_tensor_t& work_info_set,
                                     const int max_seqlen_q,
                                     const int page_size,
                                     const int nhead_kv,
                                     const double softmax_scale,
                                     aiter_tensor_t& logits,
                                     aiter_tensor_t& attn_lse,
                                     aiter_tensor_t& o,
                                     aiter_tensor_t& final_lse,
                                     aiter_tensor_t& q_scale,
                                     aiter_tensor_t& kv_scale)
{
    using T = Ds32Traits;
    const std::string gfx = get_gpu_arch();
    AITER_CHECK(gfx == "gfx950", "mla_decode_stage1_opus_fwd_ds32: unsupported GPU arch '", gfx,
                "' (supported: gfx950).");
    AITER_CHECK(page_size == 1, "mla_decode_stage1_opus_fwd_ds32: only page_size==1 supported, got ",
                page_size);
    AITER_CHECK(q_nope.dtype() == AITER_DTYPE_fp8 && kv_nope.dtype() == AITER_DTYPE_fp8,
                "mla_decode_stage1_opus_fwd_ds32: q_nope/kv_nope must be fp8");
    AITER_CHECK(q_rope.dtype() == AITER_DTYPE_bf16 && kv_rope.dtype() == AITER_DTYPE_bf16,
                "mla_decode_stage1_opus_fwd_ds32: q_rope/kv_rope must be bf16");

    // Scales must be E8M0 exponent bytes (uint8); the kernel reads them via
    // bit_cast<float>(e8m0 << 23). fp32 scale factors are NOT accepted here --
    // convert on the host first (e.g. mla.py _ds32_to_e8m0).
    AITER_CHECK(q_scale.dtype() == AITER_DTYPE_u8,
                "mla_decode_stage1_opus_fwd_ds32: q_scale must be E8M0 uint8, got ",
                AiterDtype_to_str(q_scale.dtype()));
    AITER_CHECK(kv_scale.dtype() == AITER_DTYPE_u8,
                "mla_decode_stage1_opus_fwd_ds32: kv_scale must be E8M0 uint8, got ",
                AiterDtype_to_str(kv_scale.dtype()));
    AITER_CHECK(kv_scale.size(-1) == T::D_SCALE_SIZE,
                "mla_decode_stage1_opus_fwd_ds32: kv_scale last dim (scale_dim) must be ",
                T::D_SCALE_SIZE, ", got ", kv_scale.size(-1));
    
    const int B            = q_nope.size(0);
    const int H            = q_nope.size(1);
    const int total_tokens = kv_nope.size(0);
    const int num_workers  = work_indptr.size(0) - 1;

    dsa_kargs kargs{};
    kargs.q_nope_ptr   = q_nope.data_ptr();
    kargs.q_scale_ptr  = q_scale.data_ptr();
    kargs.q_rope_ptr   = q_rope.data_ptr();
    kargs.kv_nope_ptr  = kv_nope.data_ptr();
    kargs.kv_scale_ptr = kv_scale.data_ptr();
    kargs.kv_rope_ptr  = kv_rope.data_ptr();
    kargs.o_accum      = logits.data_ptr();    // aiter split_output
    kargs.lse_accum    = attn_lse.data_ptr();  // aiter split_lse
    kargs.out_ptr      = o.data_ptr();
    kargs.lse_ptr      = final_lse.numel() > 0 ? final_lse.data_ptr() : nullptr;
    kargs.q_indptr     = reinterpret_cast<int*>(qo_indptr.data_ptr());
    kargs.kv_indptr    = reinterpret_cast<int*>(kv_indptr.data_ptr());
    kargs.kv_indices   = reinterpret_cast<int*>(kv_indices.data_ptr());
    kargs.work_indptr  = reinterpret_cast<int*>(work_indptr.data_ptr());
    kargs.work_info_set = reinterpret_cast<int*>(work_info_set.data_ptr());
    kargs.B            = B;
    kargs.H            = H;
    kargs.total_tokens = total_tokens;

    kargs.stride_q_nope_b     = H * T::D_NOPE_SIZE;
    kargs.stride_q_nope_h     = T::D_NOPE_SIZE;
    kargs.stride_q_scale_b    = H * T::D_SCALE_SIZE;
    kargs.stride_q_scale_h    = T::D_SCALE_SIZE;
    kargs.stride_q_rope_b     = H * T::D_ROPE_SIZE;
    kargs.stride_q_rope_h     = T::D_ROPE_SIZE;
    kargs.stride_o_b          = H * T::D_NOPE_SIZE;
    kargs.stride_o_h          = T::D_NOPE_SIZE;
    kargs.stride_kv_nope_page = T::D_NOPE_SIZE;
    kargs.stride_kv_scale_page = T::D_SCALE_SIZE;
    kargs.stride_kv_rope_page = T::D_ROPE_SIZE;
    kargs.softmax_scale       = static_cast<float>(softmax_scale);

    const HipDeviceGuard device_guard(o.device_id);
    hipStream_t stream = aiter::getCurrentHIPStream();
    dsa_v32_decode_16mx8_32nx1_fp8_kernel<T>
        <<<dim3(num_workers, 1, 1), dim3(T::BLOCK_SIZE), 0, stream>>>(kargs);
}
