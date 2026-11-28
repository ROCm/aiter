// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
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

// The token picks of a sampled speculative-verification step: one row per
// verification row (target_idx) and per request (bonus_idx), each read in place
// from `input`. A row at or below greedy_eps takes topk_select(.., 1,
// tie="low")'s answer, every other row mixed_sample_outer_exponential's pick on
// its own noise row. One launch, or with split_rows two (each row spread over
// 16 workgroups, then a finishing pass; `workspace` holds rows * 4096 int32), with
// bitwise the same picks. See aiter/ops/sample.py::spec_verify_pick.
void spec_verify_pick(aiter_tensor_t& out_t,
                      aiter_tensor_t& out_b,
                      aiter_tensor_t& input,
                      aiter_tensor_t& expo_t,
                      aiter_tensor_t& expo_b,
                      aiter_tensor_t& temps,
                      aiter_tensor_t& target_idx,
                      aiter_tensor_t& bonus_idx,
                      aiter_tensor_t& cu_draft,
                      aiter_tensor_t& workspace,
                      bool split_rows  = false,
                      float eps        = 1e-10,
                      float greedy_eps = 1e-10);
} // namespace aiter
