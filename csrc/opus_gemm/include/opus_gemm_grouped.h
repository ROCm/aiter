// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

// gfx950 bf16 grouped GEMM entries (opus_gemm_grouped.cu). aiter_tensor_t keeps this header torch-free.
#include "aiter_tensor.h"
#include <optional>

// M-grouped GEMM (DeepGEMM m_grouped interface); group = grouped_layout [M] or masked_m [G];
// workspace: int32 [2 G] (contiguous).
void opus_gemm_a16w16_grouped_launch(aiter_tensor_t& XQ,
                                     aiter_tensor_t& WQ,
                                     aiter_tensor_t& Y,
                                     aiter_tensor_t& group,
                                     aiter_tensor_t& workspace,
                                     int kid,
                                     bool masked);

// K-grouped GEMM (DeepGEMM k_grouped interface): Y[g] (+)= XQ[rows of g]^T WQ[rows of g]; grouped_layout:
// int32 [G], the rows (K) of each group.
void opus_gemm_a16w16_k_grouped_launch(aiter_tensor_t& XQ,
                                       aiter_tensor_t& WQ,
                                       aiter_tensor_t& Y,
                                       aiter_tensor_t& grouped_layout,
                                       std::optional<aiter_tensor_t> C,
                                       aiter_tensor_t& workspace,
                                       int kid,
                                       int seg_len);
