// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

namespace aiter {

// The floor a dynamic quantizer seeds each group's absolute maximum with, so an
// all-zero group (graph warmup, padded rows) still derives a finite, nonzero
// scale.
//
// One value for every kernel: a fused quantizer and the reference one then
// agree bit for bit on those groups. It is also ATOM's `AMAX_FLOOR`
// (atom/model_ops/blockscale_kernels/quantization.py), which the DeepSeek-V4.1
// path feeds the Triton quantizers as `amax_floor`.
//
// 1e-10 keeps every scale form in range: an e8m0 exponent of -41 against a
// floor of -127, and an fp32 inverse scale of about 4.5e12 at an fp8 max of 448.
inline constexpr float kAmaxFloor = 1e-10f;

} // namespace aiter
