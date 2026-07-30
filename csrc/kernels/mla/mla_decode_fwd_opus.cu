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

#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>

#include "ds32/mla_decode_fwd_16mx8_32nx1_fp8fp8_ps_opus.hpp"
#include "mla.h"
#include "aiter_hip_common.h"

// Causal is a compile-time specialization: a request only needs the causal
// diagonal when it carries more than one query token, so a pure decode launch
// (max_seqlen_q == 1) picks the build that masks out-of-bounds columns only.
template <bool CAUSAL>
using OpusTraitsC = mla_16mx8_32nx1_fp8fp8_ps_traits<16, 32, 8, fp8_t, fp8_t, bf16_t, CAUSAL>;
using OpusTraits  = OpusTraitsC<false>;

// q       : [B, H, D_HEAD]           fp8 (merged nope+rope, d = 576)
// kv      : [total_tokens, D_HEAD]   fp8 (merged nope+rope, d = 576)
// q_scale : float[1]                 per-tensor descale (s_descale_q)
// kv_scale: float[1]                 per-tensor descale (s_descale_k)
// logits  : [num_partials, 1, H, D_NOPE] fp32  (aiter split_output)
// attn_lse: [num_partials, 1, H, 1]      fp32  (aiter split_lse)
// o       : [B, H, D_NOPE] bf16 (final, for no-split work items)
void mla_decode_fwd_opus_stage1(torch::Tensor& q,
                                torch::Tensor& kv,
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
    using T               = OpusTraits;
    const std::string gfx = get_gpu_arch();
    TORCH_CHECK(gfx == "gfx950",
                "mla_decode_fwd_opus_stage1: unsupported GPU arch '",
                gfx,
                "' (supported: gfx950).");
    TORCH_CHECK(page_size == 1,
                "mla_decode_fwd_opus_stage1: only page_size==1 supported, got ",
                page_size);
    TORCH_CHECK(q.size(-1) == T::D_HEAD_SIZE,
                "mla_decode_fwd_opus_stage1: q last dim must be ",
                T::D_HEAD_SIZE,
                " (merged nope+rope), got ",
                q.size(-1));
    TORCH_CHECK(kv.size(-1) == T::D_HEAD_SIZE,
                "mla_decode_fwd_opus_stage1: kv last dim must be ",
                T::D_HEAD_SIZE,
                " (merged nope+rope), got ",
                kv.size(-1));
    TORCH_CHECK(q_scale.scalar_type() == at::kFloat && q_scale.numel() >= 1,
                "mla_decode_fwd_opus_stage1: q_scale must be a float scalar tensor");
    TORCH_CHECK(kv_scale.scalar_type() == at::kFloat && kv_scale.numel() >= 1,
                "mla_decode_fwd_opus_stage1: kv_scale must be a float scalar tensor");

    const int B            = q.size(0);
    const int H            = q.size(1);
    const int total_tokens = kv.size(0);
    const int num_workers  = work_indptr.size(0) - 1;

    mla_kargs kargs{};
    kargs.q_buffer_ptr  = q.data_ptr();
    kargs.q_scale_ptr   = q_scale.data_ptr();
    kargs.kv_buffer_ptr = kv.data_ptr();
    kargs.kv_scale_ptr  = kv_scale.data_ptr();
    kargs.o_accum       = logits.data_ptr();   // aiter split_output
    kargs.lse_accum     = attn_lse.data_ptr(); // aiter split_lse
    kargs.out_ptr       = o.data_ptr();
    kargs.lse_ptr       = final_lse.numel() > 0 ? final_lse.data_ptr() : nullptr;
    kargs.q_indptr      = qo_indptr.data_ptr<int>();
    kargs.kv_indptr     = kv_indptr.data_ptr<int>();
    kargs.kv_indices    = kv_indices.data_ptr<int>();
    kargs.work_indptr   = work_indptr.data_ptr<int>();
    kargs.work_info_set = work_info_set.data_ptr<int>();
    kargs.B             = B;
    kargs.H             = H;
    kargs.total_tokens  = total_tokens;

    // Merged d=576 buffer: one row per (token, head); rope is the +D_NOPE slice.
    kargs.stride_q_b     = H * T::D_HEAD_SIZE;
    kargs.stride_q_h     = T::D_HEAD_SIZE;
    kargs.stride_o_b     = H * T::D_NOPE_SIZE;
    kargs.stride_o_h     = T::D_NOPE_SIZE;
    kargs.stride_kv_page = T::D_HEAD_SIZE;
    kargs.softmax_scale  = static_cast<float>(softmax_scale);

    auto stream = at::cuda::getCurrentHIPStream().stream();
    if(max_seqlen_q > 1)
    {
        mla_decode_fwd_16mx8_32nx1_fp8fp8_opus_kernel<OpusTraitsC<true>>
            <<<dim3(num_workers, 1, 1), dim3(T::BLOCK_SIZE), 0, stream>>>(kargs);
    }
    else
    {
        mla_decode_fwd_16mx8_32nx1_fp8fp8_opus_kernel<OpusTraitsC<false>>
            <<<dim3(num_workers, 1, 1), dim3(T::BLOCK_SIZE), 0, stream>>>(kargs);
    }
}
