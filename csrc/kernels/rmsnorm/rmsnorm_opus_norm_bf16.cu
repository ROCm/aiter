// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Plain norm launcher for bf16 input (be + generic kernels). Own translation unit.
#include "rmsnorm_opus_norm.hpp"

namespace aiter {
OPUS_NORM_DEFINE(opus_norm_bf16, bf16_t)
} // namespace aiter
