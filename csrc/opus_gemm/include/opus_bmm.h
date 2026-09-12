// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_tensor.h"

// Opus BMM public C++ API. These frontends use BMM/grouped layouts (for example
// DSV4 wo_a) while reusing the shared opus GEMM backend kernels.

// fp8 e8m0 mxscale (block-scale) BMM (zero-copy DSV4 wo_a): O/Y are [M, batch,
// *], wo_a/w_scale batch-major. Y dtype in {fp32, bf16}. dim0=M, dim1=batch (K
// contiguous); the batch axis memory position is otherwise free (see host
// stride checks). kid-dispatched; driven by bmm_a8w8_mxscale_opus (Python).
void opus_bmm_a8w8_mxscale(aiter_tensor_t& O,
                           aiter_tensor_t& wo_a,
                           aiter_tensor_t& Y,
                           aiter_tensor_t& x_scale,
                           aiter_tensor_t& w_scale,
                           int splitK,
                           int kernelId);

// gfx1250 (MI450) fp8 e8m0 mxscale BMM with a PRESHUFFLED B. Same tensor
// contract as opus_bmm_a8w8_mxscale above, except wo_a is the output of
// shuffle_weight(w, layout=(16,16)) rather than a row-major [batch,N,K] weight.
// splitK must be 1 (the ws+reduce path is not wired yet); kernelId is reserved
// for the codegen kid lookup and is currently ignored.
void opus_bmm_a8w8_mxscale_bpreshuffle(aiter_tensor_t& O,
                                       aiter_tensor_t& wo_a,
                                       aiter_tensor_t& Y,
                                       aiter_tensor_t& x_scale,
                                       aiter_tensor_t& w_scale,
                                       int splitK,
                                       int kernelId);

// CLUSTER-LAUNCH, FUSED SPLIT-K sibling of the above. Same tensor contract for
// O / wo_a / Y / x_scale / w_scale (wo_a is still the shuffle_weight(w,(16,16))
// buffer), plus two tensors the split-K epilogue needs:
//
//   ws    partial workspace. Required when splitK > 1, ignored when splitK == 1.
//         Size it with opus_bmm_a8w8_mxscale_bpreshuffle_clusterclaunch_ws_numel
//         below and allocate it yourself (torch.empty) -- this call never
//         allocates, because the buffer is reused across a decode loop and a
//         per-launch hipMalloc would dominate the kernel it feeds.
//   bias  optional bf16 [N] or [batch, N], folded once by the last split.
//
// EMPTY MEANS ABSENT for both: pass a 0-element tensor rather than juggling an
// optional through the binding.
//
// mClusterWg folds that many ADJACENT M-TILES into one cluster so they can
// TDM-multicast the shared B block. It only pays when there is more than one
// M tile: at decode shapes (m <= 16 < B_M) ceil(M/B_M) is 1, every extra peer
// is an out-of-range workgroup, and mClusterWg=1 is the right answer.
//
// splitK and mClusterWg are CLUSTER DIMENSIONS and therefore compile-time; this
// entry point switches over the instantiated set and throws on anything else,
// rather than silently rounding to a neighbour.
void opus_bmm_a8w8_mxscale_bpreshuffle_clusterclaunch(aiter_tensor_t& O,
                                                      aiter_tensor_t& wo_a,
                                                      aiter_tensor_t& Y,
                                                      aiter_tensor_t& x_scale,
                                                      aiter_tensor_t& w_scale,
                                                      aiter_tensor_t& ws,
                                                      aiter_tensor_t& bias,
                                                      int splitK,
                                                      int mClusterWg,
                                                      int kernelId);

// Elements the `ws` tensor must hold for a given (m, n, batch, splitK, kernelId).
// Returns 0 when splitK <= 1. The element type is the one the kernel will use
// for partials, which follows Y's dtype (bf16 Y -> bf16 partials, fp32 Y ->
// fp32), so allocate ws with Y's dtype -- or any dtype whose total BYTE count is
// at least numel * Y.element_size(), which is what the launcher checks.
size_t opus_bmm_a8w8_mxscale_bpreshuffle_clusterclaunch_ws_numel(int m,
                                                                 int n,
                                                                 int batch,
                                                                 int splitK,
                                                                 int kernelId);
