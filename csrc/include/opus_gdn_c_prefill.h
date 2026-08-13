// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <torch/extension.h>

// Dense gfx942 C-input GDN prefill.
//
// c_mode:
//   0 = conservative measured auto policy
//   1 = CF, fused recurrence and output
//   2 = CS, split recurrence followed by shared dense K6
void opus_gdn_c_prefill_fwd(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor g,
    torch::Tensor beta,
    torch::Tensor o,
    float scale,
    torch::Tensor initial_state,
    torch::Tensor final_state,
    bool has_initial_state,
    bool output_final_state,
    int c_mode,
    bool use_env_overrides = true);
