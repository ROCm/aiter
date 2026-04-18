#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_tensor.h"

void aiter_sigmoid(const aiter_tensor_t& input, const aiter_tensor_t& output);
void aiter_tanh(const aiter_tensor_t& input, const aiter_tensor_t& output);
