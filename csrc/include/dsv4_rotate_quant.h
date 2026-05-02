// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_tensor.h"
#include <cstdint>

namespace aiter {

void rotate_activation_fp4quant_inplace(aiter_tensor_t& out,
                                        const aiter_tensor_t& input,
                                        int32_t group_size = 32);

} // namespace aiter
