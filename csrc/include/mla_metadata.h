// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

// Metadata entry points. The kernel TUs under csrc/kernels/mla/metadata/ still
// use heavy torch host-side APIs (TensorOptions, from_blob, std::vector<Tensor>
// returns, .to(CPU)), so this header keeps <torch/all.h>.

#include <torch/all.h>

void get_mla_metadata_v1(const torch::Tensor& seqlens_qo_indptr,
                         const torch::Tensor& seqlens_kv_indptr,
                         const torch::Tensor& kv_last_page_lens,
                         const int32_t num_heads_per_head_k,
                         const int32_t num_heads_k,
                         const bool is_causal,
                         torch::Tensor& work_metadata_ptrs,
                         torch::Tensor& work_indptr,
                         torch::Tensor& work_info,
                         torch::Tensor& reduce_indptr,
                         torch::Tensor& reduce_final_map,
                         torch::Tensor& reduce_partial_map,
                         const int32_t page_size,
                         const int32_t kv_granularity,
                         const int32_t max_seqlen_qo,
                         const int32_t uni_seqlen_qo,
                         const bool fast_mode,
                         const int32_t topk,
                         const int32_t max_split_per_batch,
                         const bool intra_batch_mode,
                         const std::optional<at::ScalarType> dtype_q,
                         const std::optional<at::ScalarType> dtype_kv);

std::vector<torch::Tensor> get_mla_metadata_v1_no_redundant(const torch::Tensor& seqlens_qo_indptr,
                                                            const torch::Tensor& seqlens_kv_indptr,
                                                            const int32_t num_heads_per_head_k,
                                                            const int32_t num_heads_k,
                                                            const bool is_causal,
                                                            const int32_t kv_granularity);
