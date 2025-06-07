#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

std::vector<at::Tensor>
mha_batch_decode(at::Tensor& q,               // [total_q, hq, d]
                 const at::Tensor& k,         // [total_k, hk, d]
                 const at::Tensor& v,         // [total_k, hk, d]
                 const at::Tensor& kv_indptr, // [b+1]
                 const at::Tensor& kv_page_indices,
                 float softmax_scale,
                 float logits_soft_cap,
                 bool zero_tensors,
                 bool return_softmax_lse,
                 std::optional<at::Tensor> out_,               // [total_q, hq, d]
                 std::optional<const at::Tensor> alibi_slopes_ // [hq] or [b, hq]
);