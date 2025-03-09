// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <torch/extension.h>

void build_tree_kernel_efficient(at::Tensor parent_list, at::Tensor selected_index, at::Tensor verified_seq_len,
                                 at::Tensor tree_mask, at::Tensor positions, at::Tensor retrive_index,
                                 at::Tensor retrive_next_token, at::Tensor retrive_next_sibling, int64_t topk,
                                 int64_t depth, int64_t draft_token_num);

void build_tree_kernel(at::Tensor parent_list, at::Tensor selected_index, at::Tensor verified_seq_len,
                       at::Tensor tree_mask, at::Tensor positions, at::Tensor retrive_index, int64_t topk,
                       int64_t depth, int64_t draft_token_num);