#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Standalone ASM entry for the qkptph/vph (PER_TOKEN_HEAD) FP8 causal paged
// batch-prefill kernel. This path is fully decoupled from the CK batch_prefill
// module: it depends only on torch, AiterAsmKernel and the fmha_v3_fwd codegen
// registry. See csrc/py_itfs_cu/asm_mha_batch_prefill.cu.
//
// The tensor layouts (5D vec_k_col_v K, col-major V, per-token-head descales and
// the SGLang 1D page table) mirror the AITERKER-112 mha_batch_prefill interface,
// which is used purely as a reference here.

#include <torch/extension.h>

namespace aiter {
namespace torch_itfs {

at::Tensor
mha_batch_prefill_asm(const at::Tensor& q,                   // [total_q, hq, d] fp8
                      const at::Tensor& k,                   // [num_pages, hk, d/x, page, x] fp8
                      const at::Tensor& v,                   // [num_pages, hk, d, page] fp8 (col-major)
                      const at::Tensor& cu_seqlens_q,        // [b+1] int32 (QTP)
                      const at::Tensor& kv_indptr,           // [b+1] int32 (LTP)
                      const at::Tensor& kv_page_indices,     // [num_pages] int32 (LTD)
                      const at::Tensor& seqlens_kvcache,     // [b] int32 per-batch KV token len
                      at::Tensor& out,                       // [total_q, hq, dv] bf16
                      const at::Tensor& q_descale_per_token, // [total_q, hq] f32
                      const at::Tensor& k_descale_per_token, // [num_pages, page, hk] f32
                      const at::Tensor& v_descale_per_head,  // [hk] f32
                      int batch,
                      int num_heads,
                      int num_heads_k,
                      int head_size_q,
                      int head_size_v,
                      int page_block_size,
                      int num_total_pages,
                      int max_seqlen_q,
                      float softmax_scale,
                      std::optional<const at::Tensor> p_scale = std::nullopt);

} // namespace torch_itfs
} // namespace aiter
