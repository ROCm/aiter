#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_tensor.h"

AiterTensor aiter_sigmoid(const aiter_tensor_t &input);
AiterTensor aiter_tanh(const aiter_tensor_t &input);
