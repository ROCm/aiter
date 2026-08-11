#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_tensor.h"

// Elementwise unary activations. The output is written into a caller-provided
// tensor (allocated Python-side) instead of being returned, so this header
// stays torch-free (see aiter/ops/aiter_operator.py for the public wrappers,
// which also handle the non-tile-friendly fallback via torch).
void aiter_sigmoid(aiter_tensor_t& out, aiter_tensor_t& input);
void aiter_tanh(aiter_tensor_t& out, aiter_tensor_t& input);
