// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Opus PA decode kernel — unified template (gfx942/gfx950).
//
// Fuses paged-attention decode into one block-per-kv-head kernel:
//   1. prologue — load Q, init accumulators / softmax state
//   2. core_loop(cl_p) — per wave-group tile: GEMM0 (Q@K) -> fuse/softmax/quant -> GEMM1 (P@V)
//   3. tail — finalize partial tiles
//   4. R_div_L + write_out — normalize and store BF16 output
//
// Grid: (num_kv_heads, batch, 1)  Block: (BLOCK_THREADS = 256, 4 warps)
// MFMA: fp8 16×16×32 on gfx942/gfx950.
//
// Build flags (module_pa_opus): PA_MFMA_MAIN_PATH=1 PA_SP3_MFMA_GEMM1=1 PA_USE_SP3_PI=1
#pragma once

#include <hip/hip_runtime.h>

#include "opus/opus.hpp"
#include "opus_pa/pa_decode_defs.h"

// Consolidated kernel body (all helpers + core loop live in this single header).
#include "opus_pa/pa_decode_body.hpp"

// ===========================================================================
// Global kernel entry (GDN-style: defined in this header, instantiated in .cu)
// ===========================================================================
template<typename Traits = pa_default_traits>
__global__ void __launch_bounds__(Traits::BLOCK_THREADS, 1)
pa_decode_kernel(pa_decode_kargs kargs) {
    pa_decode_kernel_body<Traits>(kargs);
}
