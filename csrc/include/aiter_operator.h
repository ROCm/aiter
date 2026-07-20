#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_tensor.h"

bool aiter_add(aiter_tensor_t &input, aiter_tensor_t &other, aiter_tensor_t &output);
bool aiter_mul(aiter_tensor_t &input, aiter_tensor_t &other, aiter_tensor_t &output);
bool aiter_sub(aiter_tensor_t &input, aiter_tensor_t &other, aiter_tensor_t &output);
bool aiter_div(aiter_tensor_t &input, aiter_tensor_t &other, aiter_tensor_t &output);

bool aiter_add_(aiter_tensor_t &input, aiter_tensor_t &other);
bool aiter_mul_(aiter_tensor_t &input, aiter_tensor_t &other);
bool aiter_sub_(aiter_tensor_t &input, aiter_tensor_t &other);
bool aiter_div_(aiter_tensor_t &input, aiter_tensor_t &other);
