#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

void paged_mqa_logits_metadata(const torch::Tensor& context_lens,
                               const torch::Tensor& schedule_metadata,
                               const int& batch_size,
                               const int& block_kv,
                               const int& num_sms);
