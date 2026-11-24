// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Plain norm launcher for fp32 input (be + generic kernels). Own translation unit.
#include "rmsnorm_opus_norm.hpp"

namespace aiter {
OPUS_NORM_DEFINE(opus_norm_fp32, fp32_t)
} // namespace aiter
