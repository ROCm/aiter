#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include "aiter_tensor.h"

namespace aiter {
namespace torch_itfs {

// Apply normalized Walsh-Hadamard rotation to contiguous hd128 rows. When `mean` is a valid
// (batch, heads, 128) fp32 tensor its per-(batch, head, channel) values are subtracted first,
// fusing K-smoothing into the rotation pass; pass an empty tensor to skip it.
void rotate_activation_hd128(aiter_tensor_t& out,
                             const aiter_tensor_t& input,
                             const aiter_tensor_t& mean,
                             aiter_tensor_t& partial_amax);

// Rotate hd128 rows and emit token-major MX data plus one E8M0 scale per 32 values. `mean` is
// the optional K mean documented above; pass an empty tensor for Q.
void rotate_activation_mxfp8_quant(aiter_tensor_t& out,
                                   aiter_tensor_t& scale,
                                   const aiter_tensor_t& input,
                                   double multiplier,
                                   const aiter_tensor_t& mean);

void rotate_activation_mxfp6_quant(aiter_tensor_t& out,
                                   aiter_tensor_t& scale,
                                   const aiter_tensor_t& input,
                                   double multiplier);

// The K variants write the coalesced tile layouts consumed directly by MHA v4 code objects.
void rotate_activation_mxfp6_quant_k(aiter_tensor_t& out,
                                     aiter_tensor_t& scale,
                                     const aiter_tensor_t& input,
                                     const aiter_tensor_t& mean);

void quantize_v_mxfp6_fp6_p(aiter_tensor_t& out,
                            aiter_tensor_t& scale,
                            const aiter_tensor_t& input);

void rotate_activation_mxfp4_quant(aiter_tensor_t& out,
                                   aiter_tensor_t& scale,
                                   const aiter_tensor_t& input,
                                   double multiplier);

void rotate_activation_mxfp4_quant_k(aiter_tensor_t& out,
                                     aiter_tensor_t& scale,
                                     const aiter_tensor_t& input,
                                     const aiter_tensor_t& mean);

void quantize_v_mxfp4_fp6_p(aiter_tensor_t& out,
                            aiter_tensor_t& scale,
                            const aiter_tensor_t& input);

void quantize_v_mxfp4(aiter_tensor_t& out,
                      aiter_tensor_t& scale,
                      const aiter_tensor_t& input);

} // namespace torch_itfs
} // namespace aiter
