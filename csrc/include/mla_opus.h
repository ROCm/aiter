// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

// Torch-free entry point for the DSA v3.2 (OpFoundry opus_attn/dsa_v32) MLA
// decode kernel. Follows the aiter MLA convention (see mla_hk.h): the pybind
// layer receives aiter_tensor_t (not torch::Tensor) so the generated docstring
// is parseable by aiter/jit/core.py check_args, and torch.Tensor arguments are
// converted host-side via torch_to_aiter_pybind (compile_ops develop=True).

#include "aiter_enum.h"
#include "aiter_tensor.h"

void mla_decode_stage1_opus_fwd_ds32(
    aiter_tensor_t& q_nope,  // [B, H, D_NOPE]          fp8
    aiter_tensor_t& q_rope,  // [B, H, D_ROPE]          bf16
    aiter_tensor_t& kv_nope, // [total_tokens, D_NOPE]  fp8
    aiter_tensor_t& kv_rope, // [total_tokens, D_ROPE]  bf16
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
    aiter_tensor_t& logits,   // aiter split_output [num_partials,1,H,D_NOPE] fp32
    aiter_tensor_t& attn_lse, // aiter split_lse    [num_partials,1,H,1]      fp32
    aiter_tensor_t& out,      // final [B, H, D_NOPE] bf16
    aiter_tensor_t& final_lse,
    aiter_tensor_t& q_scale,   // [B, H, D_SCALE]         uint8 (E8M0)
    aiter_tensor_t& kv_scale); // [total_tokens, D_SCALE] uint8
