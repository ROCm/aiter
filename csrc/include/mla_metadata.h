// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "aiter_enum.h"
#include "aiter_tensor.h"
#include <cstdint>
#include <optional>

void get_mla_metadata_v1(const aiter_tensor_t& seqlens_qo_indptr,
                         const aiter_tensor_t& seqlens_kv_indptr,
                         const aiter_tensor_t& kv_last_page_lens,
                         const int32_t num_heads_per_head_k,
                         const int32_t num_heads_k,
                         const bool is_causal,
                         aiter_tensor_t& work_metadata_ptrs,
                         aiter_tensor_t& work_indptr,
                         aiter_tensor_t& work_info,
                         aiter_tensor_t& reduce_indptr,
                         aiter_tensor_t& reduce_final_map,
                         aiter_tensor_t& reduce_partial_map,
                         const int32_t page_size,
                         const int32_t kv_granularity,
                         const int32_t max_seqlen_qo,
                         const int32_t uni_seqlen_qo,
                         const bool fast_mode,
                         const int32_t topk,
                         const int32_t max_split_per_batch,
                         const bool intra_batch_mode,
                         const bool is_cp_round_robin             = false,
                         const int64_t mla_version                = 0,
                         const std::optional<int64_t> dtype_q_nope  = std::nullopt,
                         const std::optional<int64_t> dtype_q_rope  = std::nullopt,
                         const std::optional<int64_t> dtype_kv_nope = std::nullopt,
                         const std::optional<int64_t> dtype_kv_rope = std::nullopt);
