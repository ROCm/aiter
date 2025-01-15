#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

void rotary_embedding(torch::Tensor &positions, torch::Tensor &query,
                      torch::Tensor &key, int64_t head_size,
                      torch::Tensor &cos_sin_cache, bool is_neox);

void batched_rotary_embedding(torch::Tensor &positions, torch::Tensor &query,
                              torch::Tensor &key, int64_t head_size,
                              torch::Tensor &cos_sin_cache, bool is_neox,
                              int64_t rot_dim,
                              torch::Tensor &cos_sin_cache_offsets);