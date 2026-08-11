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
// The direct kernel is retained as a comparison/fallback.  The active kernel
// stages natural route-major operands in double-buffered LDS and uses gfx950's
// ds_read_b64_tr_b16 to produce MFMA fragments without eight dependent scalar
// global loads and VALU inserts per operand. Each thread vector-loads one
// contiguous bf16x8 route slice, then writes it to typed LDS. 128x128 block:
// each wave a 64x64 register tile (2x2 mfma) reusing each loaded dy/a route-slice
// across 2 subtiles (halves global traffic) AND giving 4 independent accumulators
// (ILP to fill mfma latency). Pointer-increment addressing; ragged K masked in a
// hoisted remainder. bf16 store (matches triton ptgmm). P,Q mult 64.
#pragma once

#include <hip/hip_runtime.h>
#include "opus/opus.hpp"

typedef __bf16 opus_bf16x8 __attribute__((ext_vector_type(8)));
typedef float opus_f32x16 __attribute__((ext_vector_type(16)));

#define OPUS_WGTN_BM 128
#define OPUS_WGTN_BN 128
#define OPUS_WGTN_BK 16
#define OPUS_WGTN_BLOCK 256

using opus::operator""_I;

// Layout used by ds_read_b64_tr_b16.  A row-major [K=16,N=32] LDS patch is
// returned in the eight-BF16-per-lane fragment consumed by m32n32k16 MFMA.
template<int kNPerWarp = 32, int kKPerWarp = 16, int Vec = 4>
__device__ inline auto opus_wgtn_make_tr_layout(int lane_id, int stride)
{
    constexpr int lane_per_grp = 16;
    constexpr int lane_lo = 4;
    constexpr int lane_hi = lane_per_grp / lane_lo;
    constexpr int num_grps = opus::get_warp_size() / lane_per_grp;
    constexpr int grp_n = kNPerWarp / (lane_lo * Vec);
    constexpr int grp_k = num_grps / grp_n;
    constexpr auto shape = opus::make_tuple(
        opus::number<grp_k>{},
        opus::number<kKPerWarp / (lane_hi * grp_k)>{},
        opus::number<lane_hi>{},
        opus::number<grp_n>{},
        opus::number<lane_lo>{},
        opus::number<Vec>{});
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));
    const int grp_id = lane_id / lane_per_grp;
    const int lane_in_grp = lane_id % lane_per_grp;
    return opus::make_layout<Vec>(
        shape,
        opus::unfold_x_stride(dim, shape, opus::make_tuple(stride, 1_I)),
        opus::unfold_p_coord(dim, opus::make_tuple(
            grp_id / grp_n,
            lane_in_grp / lane_lo,
            grp_id % grp_n,
            lane_in_grp % lane_lo)));
}

__device__ inline opus_bf16x8 opus_wgtn_materialize_fragment(opus_bf16x8 x)
{
    // tr_load is issued as two b64 reads. Force their results into one
    // contiguous four-VGPR range before the native m32n32k16 BF16 instruction;
    // clang otherwise occasionally keeps the two pairs discontiguous.
    const opus::i32x4_t src = __builtin_bit_cast(opus::i32x4_t, x);
    opus::i32x4_t dst;
#pragma unroll
    for(int i = 0; i < 4; ++i)
        asm volatile("v_mov_b32 %0, %1\n" : "=v"(dst[i]) : "v"(src[i]));
    return __builtin_bit_cast(opus_bf16x8, dst);
}

// dy [M,P] bf16, a [M,Q] bf16 (compact, expert-grouped), offs [E+1] i32,
// dW [E,P,Q] bf16. grid(ceil(Q/128), ceil(P/128), E), block 256 (4 waves).
__global__ void opus_moe_wgrad_tn_direct_kernel(const __bf16* __restrict__ dy,
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

__global__ void opus_moe_wgrad_tn_lds_tr_kernel(const __bf16* __restrict__ dy,
                                                const __bf16* __restrict__ a,
                                                const int32_t* __restrict__ offs,
                                                __bf16* __restrict__ dW, int P, int Q)
{
    __shared__ __align__(16) opus::bf16_t As[2][OPUS_WGTN_BK][OPUS_WGTN_BM];
    __shared__ __align__(16) opus::bf16_t Bs[2][OPUS_WGTN_BK][OPUS_WGTN_BN];

    const int tid = threadIdx.x;
    const int lane = tid % 64;
    const int warp = tid / 64;
    const int wm = warp / 2;
    const int wn = warp % 2;
    const int e = blockIdx.z;
    const int m0 = blockIdx.y * OPUS_WGTN_BM;
    const int n0 = blockIdx.x * OPUS_WGTN_BN;
    const int r0 = offs[e];
    const int r1 = offs[e + 1];
    const int nroute = r1 - r0;
    const int num_k_stages = (nroute + OPUS_WGTN_BK - 1) / OPUS_WGTN_BK;

    // One bf16x8 per thread covers a complete 16x128 tile collectively.
    const int load_k = tid / 16;
    const int load_f = (tid % 16) * 8;
    auto load_stage = [&](int stage, int buf) {
        const int route_base = r0 + stage * OPUS_WGTN_BK;
        auto as = opus::make_smem(&As[buf][0][0]);
        auto bs = opus::make_smem(&Bs[buf][0][0]);
#pragma unroll
        for(int kpack = 0; kpack < OPUS_WGTN_BK / 16; ++kpack)
        {
            const int tile_k = kpack * 16 + load_k;
            const int route = route_base + tile_k;
            opus_bf16x8 av = {}, bv = {};
            if(route < r1)
            {
                const int mf = m0 + load_f;
                const int nf = n0 + load_f;
                if(mf + 7 < P)
                    av = *reinterpret_cast<const opus_bf16x8*>(
                        dy + static_cast<int64_t>(route) * P + mf);
                else
                {
#pragma unroll
                    for(int i = 0; i < 8; ++i)
                        av[i] = mf + i < P
                                    ? dy[static_cast<int64_t>(route) * P + mf + i]
                                    : (__bf16)0;
                }
                if(nf + 7 < Q)
                    bv = *reinterpret_cast<const opus_bf16x8*>(
                        a + static_cast<int64_t>(route) * Q + nf);
                else
                {
#pragma unroll
                    for(int i = 0; i < 8; ++i)
                        bv[i] = nf + i < Q
                                    ? a[static_cast<int64_t>(route) * Q + nf + i]
                                    : (__bf16)0;
                }
            }
            as.template store<8>(av, tile_k * OPUS_WGTN_BM + load_f);
            bs.template store<8>(bv, tile_k * OPUS_WGTN_BN + load_f);
        }
    };

    opus_f32x16 vc[2][2];
#pragma unroll
    for(int sm = 0; sm < 2; ++sm)
#pragma unroll
        for(int sn = 0; sn < 2; ++sn)
#pragma unroll
            for(int i = 0; i < 16; ++i)
                vc[sm][sn][i] = 0.0f;

    if(num_k_stages > 0)
        load_stage(0, 0);
    // A workgroup barrier orders waves, but does not by itself guarantee that
    // each wave's ds_write payload has retired before another wave issues the
    // transpose read.
    opus::s_waitcnt_lgkmcnt(opus::number<0>{});
    __syncthreads();

    const auto tr_layout = opus_wgtn_make_tr_layout(lane, OPUS_WGTN_BM);
    for(int stage = 0; stage < num_k_stages; ++stage)
    {
        const int cur = stage & 1;
        if(stage + 1 < num_k_stages)
            load_stage(stage + 1, cur ^ 1);

#pragma unroll
        for(int kpack = 0; kpack < OPUS_WGTN_BK / 16; ++kpack)
        {
            opus_bf16x8 va[2], vb[2];
#pragma unroll
            for(int sm = 0; sm < 2; ++sm)
            {
                auto smem = opus::make_smem(
                    &As[cur][kpack * 16][wm * 64 + sm * 32]);
                va[sm] = __builtin_bit_cast(
                    opus_bf16x8, opus::tr_load<4>(smem, tr_layout));
            }
#pragma unroll
            for(int sn = 0; sn < 2; ++sn)
            {
                auto smem = opus::make_smem(
                    &Bs[cur][kpack * 16][wn * 64 + sn * 32]);
                vb[sn] = __builtin_bit_cast(
                    opus_bf16x8, opus::tr_load<4>(smem, tr_layout));
            }
            opus::s_waitcnt_lgkmcnt(opus::number<0>{});
#pragma unroll
            for(int sm = 0; sm < 2; ++sm)
                va[sm] = opus_wgtn_materialize_fragment(va[sm]);
#pragma unroll
            for(int sn = 0; sn < 2; ++sn)
                vb[sn] = opus_wgtn_materialize_fragment(vb[sn]);
#pragma unroll
            for(int sm = 0; sm < 2; ++sm)
#pragma unroll
                for(int sn = 0; sn < 2; ++sn)
                    vc[sm][sn] = __builtin_amdgcn_mfma_f32_32x32x16_bf16(
                        va[sm], vb[sn], vc[sm][sn], 0, 0, 0);
        }
        __syncthreads();
    }

    const int lm = lane % 32;
    const int64_t dW_e = static_cast<int64_t>(e) * P * Q;
#pragma unroll
    for(int sm = 0; sm < 2; ++sm)
    {
        const int m_base = m0 + wm * 64 + sm * 32 + (lane / 32) * 4;
#pragma unroll
        for(int sn = 0; sn < 2; ++sn)
        {
            const int nc = n0 + wn * 64 + sn * 32 + lm;
            if(nc >= Q)
                continue;
#pragma unroll
            for(int i = 0; i < 16; ++i)
            {
                const int p = m_base + (i / 4) * 8 + (i % 4);
                if(p < P)
                    dW[dW_e + static_cast<int64_t>(p) * Q + nc] = (__bf16)vc[sm][sn][i];
            }
        }
    }
}

inline void opus_moe_wgrad_tn_launch_gfx950(const __bf16* dy, const __bf16* a,
                                            const int32_t* offs, __bf16* dW,
                                            int E, int P, int Q, hipStream_t stream) {
    dim3 grid((Q + OPUS_WGTN_BN - 1) / OPUS_WGTN_BN,
              (P + OPUS_WGTN_BM - 1) / OPUS_WGTN_BM, E);
    dim3 block(OPUS_WGTN_BLOCK);
    opus_moe_wgrad_tn_lds_tr_kernel<<<grid, block, 0, stream>>>(dy, a, offs, dW, P, Q);
}
