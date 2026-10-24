// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_tensor.h"
#include <cstdint>
#include <optional>

// query/key/value: [B, S, H, 128] bf16. query_out/key_out: [B, S_pad, H, 128] int8,
// value_out: [B, H, 128, S_pad] fp8 e4m3, query_scale/key_scale: [B, H, S_pad / 32] fp32,
// value_scale: [B, H, 128] fp32, maximum: [B, H * 128] int32 workspace.
// The norm weights ([128] bf16) and cosine/sine ([S, rope_dim] fp32) are all given or all absent.
void gfx1201_sage_prepare_hip(const aiter_tensor_t& query,
                              const aiter_tensor_t& key,
                              const aiter_tensor_t& value,
                              std::optional<aiter_tensor_t> query_weight,
                              std::optional<aiter_tensor_t> key_weight,
                              std::optional<aiter_tensor_t> cosine,
                              std::optional<aiter_tensor_t> sine,
                              aiter_tensor_t& maximum,
                              aiter_tensor_t& query_out,
                              aiter_tensor_t& key_out,
                              aiter_tensor_t& value_out,
                              aiter_tensor_t& query_scale,
                              aiter_tensor_t& key_scale,
                              aiter_tensor_t& value_scale,
                              int64_t rope_dim,
                              double eps,
                              double sm_scale);
