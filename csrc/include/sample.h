// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_tensor.h"
#include <cstdint>

namespace aiter {

void greedy_sample(aiter_tensor_t& out, aiter_tensor_t& input);

void random_sample_outer_exponential(aiter_tensor_t& out,
                                     aiter_tensor_t& input,
                                     aiter_tensor_t& exponentials,
                                     aiter_tensor_t& temperatures,
                                     float eps = 1e-10);

// RNG is seeded from a host-provided philox (seed, offset) pair instead of an
// at::Generator: the Python wrapper reads/advances the torch generator and
// passes the two values down (see aiter/ops/sample.py::_philox_seed_offset).
void random_sample(aiter_tensor_t& out,
                   aiter_tensor_t& input,
                   aiter_tensor_t& temperatures,
                   float lambd            = 1.0,
                   int64_t philox_seed    = 0,
                   int64_t philox_offset  = 0,
                   float eps              = 1e-10);

void mixed_sample_outer_exponential(aiter_tensor_t& out,
                                    aiter_tensor_t& input,
                                    aiter_tensor_t& exponentials,
                                    aiter_tensor_t& temperatures,
                                    float eps = 1e-10);

void mixed_sample(aiter_tensor_t& out,
                  aiter_tensor_t& input,
                  aiter_tensor_t& temperatures,
                  float lambd            = 1.0,
                  int64_t philox_seed    = 0,
                  int64_t philox_offset  = 0,
                  float eps              = 1e-10);

void exponential(aiter_tensor_t& out,
                 float lambd            = 1.0,
                 int64_t philox_seed    = 0,
                 int64_t philox_offset  = 0,
                 float eps              = 1e-10);
} // namespace aiter
