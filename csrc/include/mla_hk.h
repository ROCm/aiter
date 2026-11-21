// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

// Torch-free entry points for the HipKittens MLA decode kernels.
//   v3.2: csrc/kernels/mla/hk_v32_decode_fwd.cu
//   v4.0: csrc/kernels/mla/hk_v40_decode_fwd.cu

#include "aiter_tensor.h"
#include <optional>

void hk_mla_v32_decode_fwd(aiter_tensor_t& query,
                           aiter_tensor_t& kv_buffer,
                           const aiter_tensor_t& qo_indptr,
                           const aiter_tensor_t& kv_indptr,
                           const aiter_tensor_t& kv_page_indices,
                           const aiter_tensor_t& kv_last_page_lens,
                           const aiter_tensor_t& work_indptr,
                           const aiter_tensor_t& work_info_set,
                           const int max_seqlen_q,
                           const float softmax_scale,
                           aiter_tensor_t& split_output,
                           aiter_tensor_t& split_lse,
                           aiter_tensor_t& final_output);

// V4.0 MLA decode entry. Q/KV are split:
//   query        : [total_q, nhead, V4_DIM_QK_PACKED=512]  FP8  (NOPE 448
//                  + duplicated E8M0 scale 14 + unused trailing pad 50 per
//                  token; the kernel never reads bytes [462, 512))
//   query_rope   : [total_q, nhead, V4_DIM_ROPE=64]        BF16
//   kv_buffer    : [num_page, page_size, 1, 512]           FP8  (same packing)
//   kv_buffer_rope: [num_page, page_size, 1, 64]           BF16
// The 14 scale bytes are one E8M0 per 32-elem sub-tile (448/32), duplicated
// in pairs (scale[2i]==scale[2i+1]) since the quant tile is 64 elements.
// Constraints: (max_seqlen_q * nhead) in {128 (m16x8), 64 (m16x4)}, FP8 NOPE,
// BF16 ROPE.
void hk_mla_v40_decode_fwd(aiter_tensor_t& query,
                           aiter_tensor_t& query_rope,
                           aiter_tensor_t& kv_buffer,
                           aiter_tensor_t& kv_buffer_rope,
                           const aiter_tensor_t& qo_indptr,
                           const aiter_tensor_t& kv_page_indices,
                           const aiter_tensor_t& kv_last_page_lens,
                           const aiter_tensor_t& work_indptr,
                           const aiter_tensor_t& work_info_set,
                           const int max_seqlen_q,
                           const float softmax_scale,
                           aiter_tensor_t& split_output,
                           aiter_tensor_t& split_lse,
                           aiter_tensor_t& final_output,
                           std::optional<aiter_tensor_t> attn_sink);

// gfx950 MLA target verify (QLEN 8, 12 heads, latent 576 / value 512, FP8 e4m3
// KV, page size 1): csrc/kernels/mla/hk_mla_verify_fwd.cu.
//   q          : bf16 [bs*8, 12, 576] (last dim contiguous)
//   kv         : fp8 [N, ..., 576] (last dim contiguous)
//   kv_indptr  : int32 [bs+1], kv_indices: int32
//   kv_scale   : fp32 [1]
//   o_part     : bf16 [bs, nsplit, 96, 512], lse_part: fp32 [bs, nsplit, 96]
//                (base-2 LSE); unused when nsplit == 1
//   out        : bf16 [bs*8, 12, 512], written directly when nsplit == 1
// hk_mla_verify_reduce merges o_part / lse_part into out (nsplit <= 256).
void hk_mla_verify_fwd(const aiter_tensor_t& q,
                       const aiter_tensor_t& kv,
                       const aiter_tensor_t& kv_indptr,
                       const aiter_tensor_t& kv_indices,
                       const aiter_tensor_t& kv_scale,
                       aiter_tensor_t& o_part,
                       aiter_tensor_t& lse_part,
                       aiter_tensor_t& out,
                       const int nsplit,
                       const int min_chunk,
                       const float sm_scale_log2,
                       const int part8_min);

void hk_mla_verify_reduce(const aiter_tensor_t& o_part,
                          const aiter_tensor_t& lse_part,
                          const aiter_tensor_t& kv_indptr,
                          const aiter_tensor_t& kv_scale,
                          aiter_tensor_t& out,
                          const int nsplit,
                          const int min_chunk,
                          const int part8_min);
