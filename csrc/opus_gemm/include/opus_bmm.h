// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_tensor.h"

// Opus BMM public C++ API. These frontends use BMM/grouped layouts (for example
// DSV4 wo_a) while reusing the shared opus GEMM backend kernels.

// mmajor fp8 block-scale BMM (zero-copy DSV4 wo_a): O/Y are [M, batch, *],
// wo_a/w_scale batch-major. Y dtype in {fp32, bf16}.
void opus_bmm_a8w8_scale_mmajor(aiter_tensor_t& O,
                                aiter_tensor_t& wo_a,
                                aiter_tensor_t& Y,
                                aiter_tensor_t& x_scale,
                                aiter_tensor_t& w_scale);

// Same BMM layout as opus_bmm_a8w8_scale_mmajor, but consumes E8M0 (uint8)
// scales and uses gfx950 native scaled MFMA.
void opus_bmm_a8w8_mxscale_mmajor(aiter_tensor_t& O,
                                  aiter_tensor_t& wo_a,
                                  aiter_tensor_t& Y,
                                  aiter_tensor_t& x_scale,
                                  aiter_tensor_t& w_scale,
                                  int kernelId);
void opus_bmm_a8w8_mxscale_splitk_mmajor(aiter_tensor_t& O,
                                         aiter_tensor_t& wo_a,
                                         aiter_tensor_t& Y,
                                         aiter_tensor_t& x_scale,
                                         aiter_tensor_t& w_scale,
                                         int splitK);

// fp8 block-scale UNIFORM (Route B fp8, 4-wave full-tile, direct store) BMM.
// Y dtype in {fp32, bf16}; kernelId selects tile 700=128x128, 701=256x128.
// Batch-major (O/wo_a/Y = [batch,M,K]/[batch,N,K]/[batch,M,N]) and mmajor
// (O/Y = [M,batch,*]) surfaces.
void opus_bmm_a8w8_uniform_scale(aiter_tensor_t& O,
                            aiter_tensor_t& wo_a,
                            aiter_tensor_t& Y,
                            aiter_tensor_t& x_scale,
                            aiter_tensor_t& w_scale,
                            int kernelId);
void opus_bmm_a8w8_uniform_scale_mmajor(aiter_tensor_t& O,
                                   aiter_tensor_t& wo_a,
                                   aiter_tensor_t& Y,
                                   aiter_tensor_t& x_scale,
                                   aiter_tensor_t& w_scale,
                                   int kernelId);
