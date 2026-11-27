// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_tensor.h"

namespace aiter {

// DeepSeek-V4.1 shared-expert segment fusion: the shared expert's
// clamped SiLU-mul and the MXFP8 quant of its result in one launch.
//
//   input [T, 2*d] bf16 ([gate | up])
//   out   [T, d]    e4m3
//   scale [T, d/32] e8m0 bytes, row-major
//
// Bit for bit `silu_and_mul(tmp, input, limit)` (module_activation) followed by
// a group-32 MXFP8 quant of the BF16 `tmp` (ATOM `quantize_fp8`, Mega-MHC's
// MXFP8 epilogue): the activation result is rounded to BF16 first and the
// quant reads those BF16 bits. gfx950 only.
void silu_mul_mxfp8(const aiter_tensor_t& out,
                    const aiter_tensor_t& scale,
                    const aiter_tensor_t& input,
                    float limit = 0.0f);

} // namespace aiter
