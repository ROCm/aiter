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
// `head_split` divides the head axis into (heads / head_split, head_split) and places the inner
// index inside the tile axis. hd128 passes 1; hd256 runs as 2h half-heads and passes 2 so the two
// halves of a head land adjacent per tile, which is the layout its code object reads.
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
                                     const aiter_tensor_t& mean,
                                     int64_t head_split = 1);

void quantize_v_mxfp4_fp6_p(aiter_tensor_t& out,
                            aiter_tensor_t& scale,
                            const aiter_tensor_t& input,
                            int64_t head_split = 1);

void quantize_v_mxfp4(aiter_tensor_t& out,
                      aiter_tensor_t& scale,
                      const aiter_tensor_t& input,
                      int64_t head_split = 1);

} // namespace torch_itfs
} // namespace aiter
