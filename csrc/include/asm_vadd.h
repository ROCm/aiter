#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

// Elementwise C = A + B for 2-D contiguous fp32 tensors (row-major), using HSACO vadd_kernel.
void vadd_asm(torch::Tensor& a, torch::Tensor& b, torch::Tensor& c);
