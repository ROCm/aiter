// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_tensor.h"

// key/value: [B, S, H, 128] bf16. key_mean: [B, H, 128] bf16 workspace (K row mean,
// per batch/head/channel). key_out: [B, S, H, 128] int8 (no row padding), key_scale:
// [B, H, ceil(S / 64)] fp32 (one per 64-row block, after mean subtraction). value_out:
// [B, H, 128, S rounded up to 16] fp8 e4m3, value_scale: [B, H, 128] fp32 (one per channel).
void gfx1201_sage_prepare_h3_hip(const aiter_tensor_t& key,
                                 const aiter_tensor_t& value,
                                 aiter_tensor_t& key_mean,
                                 aiter_tensor_t& key_out,
                                 aiter_tensor_t& key_scale,
                                 aiter_tensor_t& value_out,
                                 aiter_tensor_t& value_scale);
