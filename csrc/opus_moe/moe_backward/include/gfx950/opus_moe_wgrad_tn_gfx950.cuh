// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Full-TN grouped wgrad (BF16->FP32 accumulate, BF16 store): dW[e]=dy_e^T@a_e,
// contracting over routes, reading dy/a in their NATURAL compact [routes,feat]
// layout (NO transpose, NO padding). Raw 32x32x16 bf16 MFMA, verified operand
// layout (validated by the raw MFMA bring-up):
//   A[m,k]: lane->m=lane%32, pack p->k=(lane/32)*8+p
//   B[k,n]: lane->n=lane%32, pack p->k=(lane/32)*8+p
//   C[m,n]: lane->n=lane%32, c[e]->m=(e/4)*8+(e%4)+(lane/32)*4
// DIRECT global loads, NO LDS, NO __syncthreads -- the LDS variants stalled at
// MfmaUtil 3.5% on the per-iter barrier; removing it gave ~2x. 128x128 block:
// each wave a 64x64 register tile (2x2 mfma) reusing each loaded dy/a route-slice
// across 2 subtiles (halves global traffic) AND giving 4 independent accumulators
// (ILP to fill mfma latency). Pointer-increment addressing; ragged K masked in a
// hoisted remainder. bf16 store (matches triton ptgmm). P,Q mult 64.
#pragma once

#include <hip/hip_runtime.h>

typedef __bf16 opus_bf16x8 __attribute__((ext_vector_type(8)));
typedef float opus_f32x16 __attribute__((ext_vector_type(16)));

#define OPUS_WGTN_BM 128
#define OPUS_WGTN_BN 128

// dy [M,P] bf16, a [M,Q] bf16 (compact, expert-grouped), offs [E+1] i32,
// dW [E,P,Q] bf16. grid(ceil(Q/128), ceil(P/128), E), block 256 (4 waves).
__global__ void opus_moe_wgrad_tn_kernel(const __bf16* __restrict__ dy,
                                         const __bf16* __restrict__ a,
                                         const int32_t* __restrict__ offs,
                                         __bf16* __restrict__ dW, int P, int Q)
{
    const int lane = threadIdx.x % 64, warp = threadIdx.x / 64;
    const int e = blockIdx.z, wm = warp / 2, wn = warp % 2;
    const int m0 = blockIdx.y * OPUS_WGTN_BM + wm * 64;   // this wave's 64-row base
    const int n0 = blockIdx.x * OPUS_WGTN_BN + wn * 64;
    const int r0 = offs[e], r1 = offs[e + 1], nroute = r1 - r0;
    const int lane_k0 = (lane / 32) * 8;
    const int lm = lane % 32;
    const int64_t P64 = P, Q64 = Q;
    // Clamp the load column into range (branchless): edge lanes (m/n >= P/Q, only
    // when P/Q not mult 128) then read a valid-but-wrong column; their outputs map
    // to m>=P / n>=Q rows which the write masks drop, so vc for valid outputs is
    // unaffected. Keeps the hot loop fully unmasked (per-load ternary was ~4x
    // slower). The last-route ragged tail keeps its route-mask ternary.
    const int mc0 = (m0 + lm     < P) ? m0 + lm     : P - 1;
    const int mc1 = (m0 + 32 + lm < P) ? m0 + 32 + lm : P - 1;
    const int nc0 = (n0 + lm     < Q) ? n0 + lm     : Q - 1;
    const int nc1 = (n0 + 32 + lm < Q) ? n0 + 32 + lm : Q - 1;

    opus_f32x16 vc[2][2];
    for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++)
        for (int k = 0; k < 16; k++) vc[i][j][k] = 0.0f;

    const int KF = nroute / 16;
    const __bf16* pdy[2] = { dy + static_cast<int64_t>(r0 + lane_k0) * P64 + mc0,
                             dy + static_cast<int64_t>(r0 + lane_k0) * P64 + mc1 };
    const __bf16* pa[2]  = { a  + static_cast<int64_t>(r0 + lane_k0) * Q64 + nc0,
                             a  + static_cast<int64_t>(r0 + lane_k0) * Q64 + nc1 };

    for (int kt = 0; kt < KF; kt++)
    {
        opus_bf16x8 va[2], vb[2];
        const __bf16* qa0 = pdy[0]; const __bf16* qa1 = pdy[1];
        const __bf16* qb0 = pa[0];  const __bf16* qb1 = pa[1];
#pragma unroll
        for (int pk = 0; pk < 8; pk++) {
            va[0][pk] = *qa0; qa0 += P64;  va[1][pk] = *qa1; qa1 += P64;
            vb[0][pk] = *qb0; qb0 += Q64;  vb[1][pk] = *qb1; qb1 += Q64;
        }
#pragma unroll
        for (int sm = 0; sm < 2; sm++)
            for (int sn = 0; sn < 2; sn++)
                vc[sm][sn] = __builtin_amdgcn_mfma_f32_32x32x16_bf16(va[sm], vb[sn], vc[sm][sn], 0, 0, 0);
        pdy[0] += 16 * P64; pdy[1] += 16 * P64; pa[0] += 16 * Q64; pa[1] += 16 * Q64;
    }
    const int rem = nroute - KF * 16;
    if (rem > 0) {
        opus_bf16x8 va[2], vb[2];
        const __bf16* qa0 = pdy[0]; const __bf16* qa1 = pdy[1];
        const __bf16* qb0 = pa[0];  const __bf16* qb1 = pa[1];
#pragma unroll
        for (int pk = 0; pk < 8; pk++) {
            const bool ok = (lane_k0 + pk) < rem;
            va[0][pk] = ok ? *qa0 : (__bf16)0; qa0 += P64;  va[1][pk] = ok ? *qa1 : (__bf16)0; qa1 += P64;
            vb[0][pk] = ok ? *qb0 : (__bf16)0; qb0 += Q64;  vb[1][pk] = ok ? *qb1 : (__bf16)0; qb1 += Q64;
        }
#pragma unroll
        for (int sm = 0; sm < 2; sm++)
            for (int sn = 0; sn < 2; sn++)
                vc[sm][sn] = __builtin_amdgcn_mfma_f32_32x32x16_bf16(va[sm], vb[sn], vc[sm][sn], 0, 0, 0);
    }

    const int64_t dW_e = static_cast<int64_t>(e) * P64 * Q64;
#pragma unroll
    for (int sm = 0; sm < 2; sm++) {
        const int m_base = m0 + sm * 32 + (lane / 32) * 4;
#pragma unroll
        for (int sn = 0; sn < 2; sn++) {
            const int nc = n0 + sn * 32 + lm;
            if (nc >= Q) continue;
#pragma unroll
            for (int e2 = 0; e2 < 16; e2++) {
                const int p = m_base + (e2 / 4) * 8 + (e2 % 4);
                if (p < P)
                    dW[dW_e + static_cast<int64_t>(p) * Q64 + nc] = (__bf16)vc[sm][sn][e2];
            }
        }
    }
}

inline void opus_moe_wgrad_tn_launch_gfx950(const __bf16* dy, const __bf16* a,
                                            const int32_t* offs, __bf16* dW,
                                            int E, int P, int Q, hipStream_t stream) {
    dim3 grid((Q + OPUS_WGTN_BN - 1) / OPUS_WGTN_BN,
              (P + OPUS_WGTN_BM - 1) / OPUS_WGTN_BM, E);
    dim3 block(256);
    opus_moe_wgrad_tn_kernel<<<grid, block, 0, stream>>>(dy, a, offs, dW, P, Q);
}
