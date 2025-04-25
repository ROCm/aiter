#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

enum class ActivationType : int
{
    No = -1,
    Silu = 0,
    Gelu
};
enum class QuantType : int
{
    No,
    per_Tensor,
    per_Token,
    per_1x128,
    per_128x128,
};
