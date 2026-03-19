#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

enum class ActivationType : int
{
    No     = -1,
    Silu   = 0,
    Gelu   = 1,
    Swiglu = 2,
};
enum class QuantType : int
{
    No,
    per_Tensor,
    per_Token,
    per_1x32,
    per_1x128,
    per_128x128,
    per_256x128,
    per_1024x128,
};
