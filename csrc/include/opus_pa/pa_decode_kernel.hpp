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
// Build flags (module_pa_opus): PA_MFMA_MAIN_PATH=1 PA_MFMA_GEMM0=1 PA_MFMA_GEMM1=1 PA_GEMM1_VGATHER=1
#pragma once

#include <hip/hip_runtime.h>

#include "opus/opus.hpp"
#include "opus_pa/pa_decode_defs.h"

// ===========================================================================
// MFMA fp8 16×16×32 helpers (local to this kernel entry).
// ===========================================================================
namespace pa_opus {

#if defined(__gfx942__) || defined(__gfx950__)
using mfma_acc4 = float __attribute__((ext_vector_type(4)));

__device__ inline mfma_acc4 mfma_f32_16x16x32_fp8(uint64_t a_packed,
                                                  uint64_t b_packed,
                                                  mfma_acc4 c,
                                                  bool init) {
    const long a = __builtin_bit_cast(long, a_packed);
    const long b = __builtin_bit_cast(long, b_packed);
    if (init) {
        mfma_acc4 zero{};
        return __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b, zero, 0, 0, 0);
    }
    return __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b, c, 0, 0, 0);
}

__device__ inline uint64_t pack_u64(uint32_t lo, uint32_t hi) {
    return static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
}
#endif

__device__ __forceinline__ int lane_id() { return threadIdx.x % 64; }
__device__ __forceinline__ int wave_id() { return threadIdx.x / 64; }

} // namespace pa_opus

// MFMA structural body (self-contained under opus_pa/kernels/).
#include "opus_pa/kernels/pa_decode_body.hpp"

// ===========================================================================
// Global kernel entry (GDN-style: defined in this header, instantiated in .cu)
// ===========================================================================
template<typename Traits = pa_default_traits>
__global__ void __launch_bounds__(Traits::BLOCK_THREADS, 1)
pa_decode_kernel(pa_decode_kargs kargs) {
    pa_decode_kernel_body<Traits>(kargs);
}
