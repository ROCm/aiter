// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// aiter stage1 of the DSA v3.2 (OpFoundry opus_attn/dsa_v32) MLA decode kernel.
// Reuses aiter's metadata (work_indptr / work_info_set) and reduce (mla_reduce_v1):
// this launches ONLY the decode kernel, which writes per-split partial outputs to
// logits/attn_lse (== aiter split_output/split_lse) or, for no-split work items
// (partial_qo_loc < 0), directly to the final output. gfx950 only.

#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>

#include "ds32/dsa_v32_splitkv.hpp"
#include "mla.h"
#include "aiter_hip_common.h"

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
void mla_decode_stage1_opus_fwd_ds32(torch::Tensor& q_nope,
                                     torch::Tensor& q_rope,
                                     torch::Tensor& kv_nope,
                                     torch::Tensor& kv_rope,
                                     const torch::Tensor& qo_indptr,
                                     const torch::Tensor& kv_indptr,
                                     const torch::Tensor& kv_indices,
                                     const torch::Tensor& kv_last_page_lens,
                                     const torch::Tensor& work_indptr,
                                     const torch::Tensor& work_info_set,
                                     const int max_seqlen_q,
                                     const int page_size,
                                     const int nhead_kv,
                                     const double softmax_scale,
                                     torch::Tensor& logits,
                                     torch::Tensor& attn_lse,
                                     torch::Tensor& o,
                                     torch::Tensor& final_lse,
                                     torch::Tensor& q_scale,
                                     torch::Tensor& kv_scale)
{
    using T = Ds32Traits;
    const std::string gfx = get_gpu_arch();
    TORCH_CHECK(gfx == "gfx950", "mla_decode_stage1_opus_fwd_ds32: unsupported GPU arch '", gfx,
                "' (supported: gfx950).");
    TORCH_CHECK(page_size == 1, "mla_decode_stage1_opus_fwd_ds32: only page_size==1 supported, got ",
                page_size);
    // Scales must be E8M0 exponent bytes (uint8); the kernel reads them via
    // bit_cast<float>(e8m0 << 23). fp32 scale factors are NOT accepted here --
    // convert on the host first (e.g. mla.py _ds32_to_e8m0).
    TORCH_CHECK(q_scale.scalar_type() == at::kByte,
                "mla_decode_stage1_opus_fwd_ds32: q_scale must be E8M0 uint8, got ",
                q_scale.scalar_type());
    TORCH_CHECK(kv_scale.scalar_type() == at::kByte,
                "mla_decode_stage1_opus_fwd_ds32: kv_scale must be E8M0 uint8, got ",
                kv_scale.scalar_type());

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
    kargs.q_indptr     = qo_indptr.data_ptr<int>();
    kargs.kv_indptr    = kv_indptr.data_ptr<int>();
    kargs.kv_indices   = kv_indices.data_ptr<int>();
    kargs.work_indptr  = work_indptr.data_ptr<int>();
    kargs.work_info_set = work_info_set.data_ptr<int>();
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

    auto stream = at::cuda::getCurrentHIPStream().stream();
    dsa_v32_decode_16mx8_32nx1_fp8_kernel<T>
        <<<dim3(num_workers, 1, 1), dim3(T::BLOCK_SIZE), 0, stream>>>(kargs);
}
