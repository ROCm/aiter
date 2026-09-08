// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

// Torch-free entry point for the MLA reduce kernel. The kernel TU
// (csrc/kernels/mla/reduce.cu) uses aiter_tensor_t directly; the Python
// wrapper marshals torch.Tensor -> aiter_tensor_t via the develop=True path
// in aiter/jit/core.py.

#include "aiter_tensor.h"
#include <optional>

void mla_reduce_v1(const aiter_tensor_t& partial_output,
                   const aiter_tensor_t& partial_lse,
                   const aiter_tensor_t& reduce_indptr,
                   std::optional<aiter_tensor_t> reduce_final_map,
                   const aiter_tensor_t& reduce_partial_map,
                   const int32_t max_seqlen_q,
                   const int32_t num_kv_splits,
                   aiter_tensor_t& final_output,
                   std::optional<aiter_tensor_t> final_lse = std::nullopt);
