// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <torch/extension.h>

namespace aiter {

void quantize_fp4_e8m0_per_channel_kblock_hip(
    torch::Tensor& packed,
    torch::Tensor& scale_byte,
    const torch::Tensor& v,
    int64_t kblock_size);

} // namespace aiter
