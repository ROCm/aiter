// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.


#pragma once

#include <torch/extension.h>

void tree_speculative_sampling_target_only(at::Tensor predicts, at::Tensor accept_index,
                                           at::Tensor accept_token_num,  // mutable
                                           at::Tensor candidates, at::Tensor retrive_index,
                                           at::Tensor retrive_next_token, at::Tensor retrive_next_sibling,
                                           at::Tensor uniform_samples, at::Tensor target_probs, at::Tensor draft_probs,
                                           bool deterministic = true);