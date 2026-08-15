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
// contiguous bf16x8 route slice, then writes it to typed LDS. The general
// fallback is a 128x128, four-wave block. 256-aligned P/Q use a 256x256,
// eight-wave block which halves operand traffic per output while preserving the
// same fragment mapping. Pointer-increment addressing; ragged K is zero padded.
// bf16 store (matches triton ptgmm).
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

// Minimal fixed-register wrappers for the aligned 8-wave kernel. Keeping these
// local avoids making the whole backward module depend on an external template
// library merely to name physical VGPR/AGPR operands in four ISA instructions.
#define OPUS_WGTN_CLOBBER_V8(A, B, C, D, E, F, G, H) \
    asm volatile("" ::: "v" #A, "v" #B, "v" #C, "v" #D, \
                         "v" #E, "v" #F, "v" #G, "v" #H)
#define OPUS_WGTN_CLOBBER_A8(A, B, C, D, E, F, G, H) \
    asm volatile("" ::: "a" #A, "a" #B, "a" #C, "a" #D, \
                         "a" #E, "a" #F, "a" #G, "a" #H)

__device__ inline void opus_wgtn_clobber_fixed_regs()
{
    OPUS_WGTN_CLOBBER_V8(96, 97, 98, 99, 100, 101, 102, 103);
    OPUS_WGTN_CLOBBER_V8(104, 105, 106, 107, 108, 109, 110, 111);
    OPUS_WGTN_CLOBBER_V8(112, 113, 114, 115, 116, 117, 118, 119);
    OPUS_WGTN_CLOBBER_V8(120, 121, 122, 123, 124, 125, 126, 127);
    OPUS_WGTN_CLOBBER_A8(0, 1, 2, 3, 4, 5, 6, 7);
    OPUS_WGTN_CLOBBER_A8(8, 9, 10, 11, 12, 13, 14, 15);
    OPUS_WGTN_CLOBBER_A8(16, 17, 18, 19, 20, 21, 22, 23);
    OPUS_WGTN_CLOBBER_A8(24, 25, 26, 27, 28, 29, 30, 31);
    OPUS_WGTN_CLOBBER_A8(32, 33, 34, 35, 36, 37, 38, 39);
    OPUS_WGTN_CLOBBER_A8(40, 41, 42, 43, 44, 45, 46, 47);
    OPUS_WGTN_CLOBBER_A8(48, 49, 50, 51, 52, 53, 54, 55);
    OPUS_WGTN_CLOBBER_A8(56, 57, 58, 59, 60, 61, 62, 63);
    OPUS_WGTN_CLOBBER_A8(64, 65, 66, 67, 68, 69, 70, 71);
    OPUS_WGTN_CLOBBER_A8(72, 73, 74, 75, 76, 77, 78, 79);
    OPUS_WGTN_CLOBBER_A8(80, 81, 82, 83, 84, 85, 86, 87);
    OPUS_WGTN_CLOBBER_A8(88, 89, 90, 91, 92, 93, 94, 95);
    OPUS_WGTN_CLOBBER_A8(96, 97, 98, 99, 100, 101, 102, 103);
    OPUS_WGTN_CLOBBER_A8(104, 105, 106, 107, 108, 109, 110, 111);
    OPUS_WGTN_CLOBBER_A8(112, 113, 114, 115, 116, 117, 118, 119);
    OPUS_WGTN_CLOBBER_A8(120, 121, 122, 123, 124, 125, 126, 127);
}

#undef OPUS_WGTN_CLOBBER_V8
#undef OPUS_WGTN_CLOBBER_A8

template<int GPR_START>
__device__ inline void opus_wgtn_ds_read_b64_tr_b16(uint32_t smem_ptr)
{
    asm volatile("ds_read_b64_tr_b16 v[%0:%1], %2 offset:0"
                 :
                 : "n"(GPR_START), "n"(GPR_START + 1), "v"(smem_ptr)
                 : "memory");
}

template<int GPR_START, int BYTE_OFFSET>
__device__ inline void opus_wgtn_ds_read_b64_tr_b16_offset(uint32_t smem_ptr)
{
    asm volatile("ds_read_b64_tr_b16 v[%0:%1], %2 offset:%3"
                 :
                 : "n"(GPR_START), "n"(GPR_START + 1), "v"(smem_ptr),
                   "n"(BYTE_OFFSET)
                 : "memory");
}

template<int GPR_START, int BYTE_OFFSET, int SECOND_OFFSET>
__device__ inline void opus_wgtn_fixed_tr_fragment_offset(uint32_t smem_ptr)
{
    opus_wgtn_ds_read_b64_tr_b16_offset<GPR_START, BYTE_OFFSET>(smem_ptr);
    opus_wgtn_ds_read_b64_tr_b16_offset<GPR_START + 2,
                                        BYTE_OFFSET + SECOND_OFFSET>(smem_ptr);
}

template<int A, int B, int D>
__device__ inline void opus_wgtn_mfma_zero()
{
    asm volatile("v_mfma_f32_32x32x16_bf16 a[%0:%1], "
                 "v[%2:%3], v[%4:%5], 0"
                 :
                 : "n"(D - 256), "n"(D + 15 - 256),
                   "n"(A), "n"(A + 3), "n"(B), "n"(B + 3));
}

template<int A, int B, int C, int D>
__device__ inline void opus_wgtn_mfma_accum()
{
    asm volatile("v_mfma_f32_32x32x16_bf16 a[%0:%1], "
                 "v[%2:%3], v[%4:%5], a[%6:%7]"
                 :
                 : "n"(D - 256), "n"(D + 15 - 256),
                   "n"(A), "n"(A + 3), "n"(B), "n"(B + 3),
                   "n"(C - 256), "n"(C + 15 - 256));
}

template<int GPR>
__device__ inline uint32_t opus_wgtn_read_acc()
{
    uint32_t value;
    asm volatile("v_accvgpr_read_b32 %0, a[%1]"
                 : "=v"(value)
                 : "n"(GPR - 256));
    return value;
}

template<int GPR_START, typename Smem, typename Layout>
__device__ inline void opus_wgtn_fixed_tr_fragment(Smem& src, const Layout& layout)
{
    auto offsets = opus::layout_to_offsets<4>(layout);
    const uint32_t addr0 = static_cast<uint32_t>(
        reinterpret_cast<__UINTPTR_TYPE__>(src.ptr + offsets[0] * sizeof(__bf16)));
    const uint32_t addr1 = static_cast<uint32_t>(
        reinterpret_cast<__UINTPTR_TYPE__>(src.ptr + offsets[1] * sizeof(__bf16)));
    opus_wgtn_ds_read_b64_tr_b16<GPR_START>(addr0);
    opus_wgtn_ds_read_b64_tr_b16<GPR_START + 2>(addr1);
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
    auto load_global = [&](int stage, opus_bf16x8& av, opus_bf16x8& bv) {
        const int route_base = r0 + stage * OPUS_WGTN_BK;
        const int route = route_base + load_k;
        av = {};
        bv = {};
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
    };
    auto store_stage = [&](int buf, const opus_bf16x8& av, const opus_bf16x8& bv) {
        auto as = opus::make_smem(&As[buf][0][0]);
        auto bs = opus::make_smem(&Bs[buf][0][0]);
        as.template store<8>(av, load_k * OPUS_WGTN_BM + load_f);
        bs.template store<8>(bv, load_k * OPUS_WGTN_BN + load_f);
    };
    auto load_stage = [&](int stage, int buf) {
        opus_bf16x8 av, bv;
        load_global(stage, av, bv);
        store_stage(buf, av, bv);
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
        const bool has_next = stage + 1 < num_k_stages;
        opus_bf16x8 next_av, next_bv;
        if(has_next)
            load_global(stage + 1, next_av, next_bv);

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
        if(has_next)
        {
            store_stage(cur ^ 1, next_av, next_bv);
            opus::s_waitcnt_lgkmcnt(opus::number<0>{});
            __syncthreads();
        }
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

// Eight-wave, 256x256 output tile.  Each wave owns a 128x64 tile expressed as
// 4x2 independent m32n32k16 MFMAs.  This deliberately reuses the exact
// transpose-read/MFMA mapping of the verified 128x128 kernel above while
// halving operand traffic per output element.
template<int FIXED_P = 0,
         int FIXED_Q = 0,
         int FIXED_ROUTES = 0,
         bool INTERLEAVE_EXPERTS = false>
__global__ __launch_bounds__(512, 1)
__attribute__((amdgpu_num_vgpr(128)))
void opus_moe_wgrad_tn_8wave_kernel(const __bf16* __restrict__ dy,
                                    const __bf16* __restrict__ a,
                                    const int32_t* __restrict__ offs,
                                    __bf16* __restrict__ dW, int P_arg, int Q_arg)
{
    opus_wgtn_clobber_fixed_regs();

    constexpr int BM = 256;
    constexpr int BN = 256;
    constexpr int BK = 64;
    // The natural-route target specializations use a direct VMEM->LDS copy.
    // A buffer_load_dwordx4_lds always writes lane i at M0+i*16 bytes, so one
    // full wave naturally covers two consecutive 256-BF16 rows.  Store those
    // rows as a pair and insert one 64-byte bank-phase shift per pair.  This
    // keeps all 64 lanes active (unlike the earlier half-wave padded copy),
    // removes the explicit ds_write_b128 staging pass, and still gives the
    // transpose reader a regular physical stride every two logical rows.
    constexpr bool PAIR_DIRECT_TO_LDS =
        FIXED_ROUTES == 0 && FIXED_P == 2048 &&
        (FIXED_Q == 1024 || FIXED_Q == 2048);
    // A 256-bf16 row is a 512-byte multiple of the LDS bank period, so every
    // transpose read aliases the same bank phase.  Shift successive K rows by
    // 64 bytes; this cuts the measured ds_read bank conflicts without changing
    // the global-load or MFMA fragment layouts.
    constexpr int LDS_PAD = 32;
    constexpr int LD = BM + LDS_PAD;
    constexpr int PAIR_LDS_PAD = 32;
    constexpr int PAIR_LD = 2 * BM + PAIR_LDS_PAD;
    constexpr int LDS_ROWS = PAIR_DIRECT_TO_LDS ? BK / 2 : BK;
    constexpr int LDS_STRIDE = PAIR_DIRECT_TO_LDS ? PAIR_LD : LD;
    const int P = FIXED_P > 0 ? FIXED_P : P_arg;
    const int Q = FIXED_Q > 0 ? FIXED_Q : Q_arg;
    __shared__ __align__(16) opus::bf16_t As[2][LDS_ROWS][LDS_STRIDE];
    __shared__ __align__(16) opus::bf16_t Bs[2][LDS_ROWS][LDS_STRIDE];

    const int tid = threadIdx.x;
    const int lane = tid % 64;
    const int warp = tid / 64;
    // Keep the wave id in an SGPR for Direct-to-LDS destination arithmetic.
    // Otherwise clang rebuilds the wave-uniform LDS pointer in VGPRs and emits
    // a v_readfirstlane before every buffer_load_lds instruction.
    const int scalar_warp = __builtin_amdgcn_readfirstlane(warp);
    const int wm = warp / 4;
    const int wn = warp % 4;
    int e = blockIdx.z;
    int tile_m = blockIdx.y;
    int tile_n = blockIdx.x;
    if constexpr(FIXED_P > 0 && FIXED_Q > 0)
    {
        // Keep two B panels shared while advancing all eight M tiles. For the
        // target ragged dW1, interleave four experts inside this tile order so
        // route-count variation does not leave a long final CTA wave. Other
        // exact paths retain the original expert-major 3-D mapping.
        constexpr int GROUP_M = 8;
        constexpr int GROUP_N = 2;
        const int tiles_n = FIXED_Q / BN;
        int linear;
        if constexpr(INTERLEAVE_EXPERTS)
        {
            static_assert(FIXED_Q == 2048 && FIXED_ROUTES == 0);
            constexpr int TILES_M = FIXED_P / BM;
            constexpr int TILES_PER_EXPERT = TILES_M * (FIXED_Q / BN);
            constexpr int EXPERT_GROUP = 4;
            // Keep the two N panels of one M tile adjacent before rotating to
            // the next expert. This retains four-expert tail balancing while
            // restoring short-range reuse of the shared A panel.
            constexpr int TILES_PER_TURN = 2;
            const int schedule = blockIdx.x;
            const int expert_group =
                schedule / (TILES_PER_EXPERT * EXPERT_GROUP);
            const int group_work =
                schedule % (TILES_PER_EXPERT * EXPERT_GROUP);
            const int turn = group_work / (EXPERT_GROUP * TILES_PER_TURN);
            const int in_turn = group_work % (EXPERT_GROUP * TILES_PER_TURN);
            e = expert_group * EXPERT_GROUP + in_turn / TILES_PER_TURN;
            linear = turn * TILES_PER_TURN + in_turn % TILES_PER_TURN;
        }
        else
            linear = blockIdx.y * tiles_n + blockIdx.x;
        const int group_id = linear / (GROUP_M * GROUP_N);
        const int in_group = linear % (GROUP_M * GROUP_N);
        const int groups_n = tiles_n / GROUP_N;
        tile_m = (group_id / groups_n) * GROUP_M + in_group / GROUP_N;
        tile_n = (group_id % groups_n) * GROUP_N + in_group % GROUP_N;
    }
    const int m0 = tile_m * BM;
    const int n0 = tile_n * BN;
    const int r0 = FIXED_ROUTES > 0 ? e * FIXED_ROUTES : offs[e];
    const int r1 = FIXED_ROUTES > 0 ? r0 + FIXED_ROUTES : offs[e + 1];
    const int nroute = r1 - r0;
    const int num_k_stages = (nroute + BK - 1) / BK;

    // Use a 64-route stage to halve workgroup barriers, but keep only one
    // 32-route half-stage in registers.  The same load registers are reused for
    // the second half after the first half has been written to the next buffer.
    // This preserves the BK=32 register footprint while the two current-stage
    // MFMA halves hide the corresponding global prefetches.
    // 512 threads x two bf16x8 vectors cover one 32x256 half-stage.
    const int load_k = tid / 16;
    const int load_f = (tid % 16) * 16;
    struct load_regs {
        opus_bf16x8 a0, a1, b0, b1;
    };
    const unsigned int dy_resource_bytes = PAIR_DIRECT_TO_LDS && nroute > 0
        ? static_cast<unsigned int>(
              (static_cast<int64_t>(nroute) * P - m0) * sizeof(__bf16))
        : 0xffffffffu;
    const unsigned int a_resource_bytes = PAIR_DIRECT_TO_LDS && nroute > 0
        ? static_cast<unsigned int>(
              (static_cast<int64_t>(nroute) * Q - n0) * sizeof(__bf16))
        : 0xffffffffu;
    auto g_dy = opus::make_gmem(
        reinterpret_cast<const opus::bf16_t*>(dy) +
        static_cast<int64_t>(r0) * P + m0,
        dy_resource_bytes);
    auto g_a = opus::make_gmem(
        reinterpret_cast<const opus::bf16_t*>(a) +
        static_cast<int64_t>(r0) * Q + n0,
        a_resource_bytes);
    auto load_global_valid = [&](int stage, int half, load_regs& x) {
        const int route = r0 + stage * BK + half * 32 + load_k;
        const int mf = m0 + load_f;
        const int nf = n0 + load_f;
        if constexpr(FIXED_P > 0 && FIXED_Q > 0)
        {
            const int local_route = stage * BK + half * 32 + load_k;
            x.a0 = __builtin_bit_cast(
                opus_bf16x8,
                g_dy.template load<8>(local_route * P + load_f));
            x.a1 = __builtin_bit_cast(
                opus_bf16x8,
                g_dy.template load<8>(local_route * P + load_f + 8));
            x.b0 = __builtin_bit_cast(
                opus_bf16x8,
                g_a.template load<8>(local_route * Q + load_f));
            x.b1 = __builtin_bit_cast(
                opus_bf16x8,
                g_a.template load<8>(local_route * Q + load_f + 8));
        }
        else
        {
            x.a0 = *reinterpret_cast<const opus_bf16x8*>(
                dy + static_cast<int64_t>(route) * P + mf);
            x.a1 = *reinterpret_cast<const opus_bf16x8*>(
                dy + static_cast<int64_t>(route) * P + mf + 8);
            x.b0 = *reinterpret_cast<const opus_bf16x8*>(
                a + static_cast<int64_t>(route) * Q + nf);
            x.b1 = *reinterpret_cast<const opus_bf16x8*>(
                a + static_cast<int64_t>(route) * Q + nf + 8);
        }
    };
    auto load_global_tail = [&](int stage, int half, load_regs& x) {
        const int route = r0 + stage * BK + half * 32 + load_k;
        x.a0 = {}; x.a1 = {}; x.b0 = {}; x.b1 = {};
        if(route < r1)
            load_global_valid(stage, half, x);
    };
    auto store_stage = [&](int buf, int half, const load_regs& x) {
        if constexpr(!PAIR_DIRECT_TO_LDS)
        {
            auto as = opus::make_smem(&As[buf][0][0]);
            auto bs = opus::make_smem(&Bs[buf][0][0]);
            const int os = (half * 32 + load_k) * LD + load_f;
            as.template store<8>(x.a0, os);
            as.template store<8>(x.a1, os + 8);
            bs.template store<8>(x.b0, os);
            bs.template store<8>(x.b1, os + 8);
        }
    };

    // Each wave owns four consecutive row pairs so its Direct-to-LDS writes
    // stay in one contiguous LDS region. Issue all four A copies before the
    // four B copies: keeping one LDS destination base live lets clang retain
    // M0 instead of switching it A/B for every row pair. The MUBUF immediate
    // advances both VMEM and LDS, hence the matching subtraction from vaddr.
    auto async_load_stage = [&](int stage, int buf) {
        if constexpr(PAIR_DIRECT_TO_LDS)
        {
            const int lane_row = lane >> 5;
            const int feature = (lane & 31) * 8;
            void* a_dst = reinterpret_cast<void*>(
                &As[buf][scalar_warp * 4][0]);
            void* b_dst = reinterpret_cast<void*>(
                &Bs[buf][scalar_warp * 4][0]);
            auto copy_a_pair = [&](auto pair_tag) {
                constexpr int PAIR = decltype(pair_tag)::value;
                constexpr int OFF = PAIR * PAIR_LD * sizeof(__bf16);
                static_assert(OFF < 4096);
                const int row = warp * 8 + PAIR * 2 + lane_row;
                g_dy.template async_load<8, OFF>(
                    a_dst,
                    (stage * BK + row) * P + feature -
                        OFF / sizeof(__bf16));
            };
            auto copy_b_pair = [&](auto pair_tag) {
                constexpr int PAIR = decltype(pair_tag)::value;
                constexpr int OFF = PAIR * PAIR_LD * sizeof(__bf16);
                static_assert(OFF < 4096);
                const int row = warp * 8 + PAIR * 2 + lane_row;
                g_a.template async_load<8, OFF>(
                    b_dst,
                    (stage * BK + row) * Q + feature -
                        OFF / sizeof(__bf16));
            };
            opus::static_for<4>(copy_a_pair);
            opus::static_for<4>(copy_b_pair);
        }
    };

    const int full_k_stages = nroute / BK;
    const bool has_tail_stage = full_k_stages < num_k_stages;
    // Dynamic routing uses unmasked loads for every complete 64-route stage;
    // only its final partial stage pays zero-fill and per-lane bounds checks.
    // Keep the original inline loop for FIXED_ROUTES so its compile-time path
    // and balanced-route scheduling remain unchanged.
    if(num_k_stages > 0)
    {
        if constexpr(PAIR_DIRECT_TO_LDS)
            async_load_stage(0, 0);
        else
        {
            load_regs first;
            if(full_k_stages > 0)
                load_global_valid(0, 0, first);
            else
                load_global_tail(0, 0, first);
            store_stage(0, 0, first);
            if(full_k_stages > 0)
                load_global_valid(0, 1, first);
            else
                load_global_tail(0, 1, first);
            store_stage(0, 1, first);
        }
    }
    if constexpr(PAIR_DIRECT_TO_LDS)
        opus::s_waitcnt_vmcnt(opus::number<0>{});
    opus::s_waitcnt_lgkmcnt(opus::number<0>{});
    __syncthreads();

    const auto tr_layout = opus_wgtn_make_tr_layout(
        lane, PAIR_DIRECT_TO_LDS ? BM : LD);
    const auto tr_offsets = opus::layout_to_offsets<4>(tr_layout);
    auto tr_base = [&](auto* ptr, int feature_base) {
        int physical_offset;
        if constexpr(PAIR_DIRECT_TO_LDS)
        {
            const int logical_offset = tr_offsets[0];
            const int logical_row = logical_offset / BM;
            const int logical_col = logical_offset % BM;
            physical_offset =
                (logical_row / 2) * PAIR_LD +
                (logical_row & 1) * BM + logical_col + feature_base;
        }
        else
        {
            physical_offset = tr_offsets[0] + feature_base;
        }
        auto smem = opus::make_smem(ptr + physical_offset);
        return static_cast<uint32_t>(reinterpret_cast<__UINTPTR_TYPE__>(
            smem.ptr));
    };
    const uint32_t a_base[2] = {
        tr_base(&As[0][0][0], wm * 128),
        tr_base(&As[1][0][0], wm * 128)};
    const uint32_t b_base[2] = {
        tr_base(&Bs[0][0][0], wn * 64),
        tr_base(&Bs[1][0][0], wn * 64)};
    constexpr int KPACK_BYTES = PAIR_DIRECT_TO_LDS
        ? 8 * PAIR_LD * sizeof(__bf16)
        : 16 * LD * sizeof(__bf16);
    constexpr int SUBTILE_BYTES = 32 * sizeof(__bf16);
    constexpr int TR_SECOND_BYTES = PAIR_DIRECT_TO_LDS
        ? 2 * PAIR_LD * sizeof(__bf16)
        : 4 * LD * sizeof(__bf16);
    if constexpr(FIXED_ROUTES > 0)
    {
        for(int stage = 0; stage < num_k_stages; ++stage)
        {
            const int cur = stage & 1;
            const bool has_next = stage + 1 < num_k_stages;
            load_regs next;
            if(has_next)
                load_global_valid(stage + 1, 0, next);

            __builtin_amdgcn_s_setprio(1);
            opus::static_for<4>([&](auto kpack) {
                constexpr int b_reg = 104 + (kpack.value & 1) * 8;
                if constexpr(kpack.value == 0)
                {
                    opus::static_for<2>([&](auto sn) {
                        opus_wgtn_fixed_tr_fragment_offset<
                            b_reg + sn.value * 4,
                            sn.value * SUBTILE_BYTES,
                            TR_SECOND_BYTES>(b_base[cur]);
                    });
                    opus_wgtn_fixed_tr_fragment_offset<
                        96, 0, TR_SECOND_BYTES>(a_base[cur]);
                    opus::s_waitcnt_lgkmcnt(opus::number<0>{});
                }
                opus::static_for<4>([&](auto sm) {
                    constexpr int a_reg = 96 + (sm.value & 1) * 4;
                    if constexpr(sm.value + 1 < 4)
                    {
                        opus_wgtn_fixed_tr_fragment_offset<
                            96 + ((sm.value + 1) & 1) * 4,
                            kpack.value * KPACK_BYTES +
                                (sm.value + 1) * SUBTILE_BYTES,
                            TR_SECOND_BYTES>(a_base[cur]);
                    }
                    if constexpr(sm.value == 2 && kpack.value + 1 < 4)
                    {
                        constexpr int next_b_reg =
                            104 + ((kpack.value + 1) & 1) * 8;
                        opus::static_for<2>([&](auto sn) {
                            opus_wgtn_fixed_tr_fragment_offset<
                                next_b_reg + sn.value * 4,
                                (kpack.value + 1) * KPACK_BYTES +
                                    sn.value * SUBTILE_BYTES,
                                TR_SECOND_BYTES>(b_base[cur]);
                        });
                    }
                    if constexpr(sm.value == 3 && kpack.value + 1 < 4)
                    {
                        opus_wgtn_fixed_tr_fragment_offset<
                            96, (kpack.value + 1) * KPACK_BYTES,
                            TR_SECOND_BYTES>(a_base[cur]);
                    }
                    if(stage == 0 && kpack.value == 0)
                    {
                        opus::static_for<2>([&](auto sn) {
                            constexpr int c =
                                256 + (sm.value * 2 + sn.value) * 16;
                            opus_wgtn_mfma_zero<
                                a_reg, b_reg + sn.value * 4, c>();
                        });
                    }
                    else
                    {
                        opus::static_for<2>([&](auto sn) {
                            constexpr int c =
                                256 + (sm.value * 2 + sn.value) * 16;
                            opus_wgtn_mfma_accum<
                                a_reg, b_reg + sn.value * 4, c, c>();
                        });
                    }
                    if constexpr(sm.value == 2 && kpack.value + 1 < 4)
                        opus::s_waitcnt_lgkmcnt(opus::number<4>{});
                    else if constexpr(sm.value + 1 < 4 || kpack.value + 1 < 4)
                        opus::s_waitcnt_lgkmcnt(opus::number<0>{});
                });
                if(has_next && kpack.value == 1)
                {
                    __builtin_amdgcn_s_setprio(0);
                    store_stage(cur ^ 1, 0, next);
                    load_global_valid(stage + 1, 1, next);
                    __builtin_amdgcn_s_setprio(1);
                }
            });
            __builtin_amdgcn_s_setprio(0);

            if(has_next)
            {
                store_stage(cur ^ 1, 1, next);
                opus::s_waitcnt_lgkmcnt(opus::number<0>{});
                __syncthreads();
            }
        }
    }
    else
    {
      auto run_stage = [&](auto has_next_tag,
                           auto masked_next_tag,
                           auto num_kpacks_tag,
                           int stage) {
        constexpr bool HAS_NEXT = decltype(has_next_tag)::value != 0;
        constexpr bool MASKED_NEXT = decltype(masked_next_tag)::value != 0;
        constexpr int NUM_KPACKS = decltype(num_kpacks_tag)::value;
        const int cur = stage & 1;
        load_regs next;
        if constexpr(HAS_NEXT)
        {
            if constexpr(PAIR_DIRECT_TO_LDS)
                async_load_stage(stage + 1, cur ^ 1);
            else if constexpr(MASKED_NEXT)
                load_global_tail(stage + 1, 0, next);
            else
                load_global_valid(stage + 1, 0, next);
        }

        __builtin_amdgcn_s_setprio(1);
        opus::static_for<NUM_KPACKS>([&](auto kpack) {
            constexpr int b_reg = 112 + (kpack.value & 1) * 8;
            // Four A slots let the preceding pack prefetch every fragment of
            // this pack after consuming the corresponding old slot. Only the
            // first pack of a stage needs a local prologue.
            if constexpr(kpack.value == 0)
            {
                opus::static_for<2>([&](auto sn) {
                    opus_wgtn_fixed_tr_fragment_offset<
                        b_reg + sn.value * 4,
                        sn.value * SUBTILE_BYTES,
                        TR_SECOND_BYTES>(b_base[cur]);
                });
                opus_wgtn_fixed_tr_fragment_offset<
                    96, 0, TR_SECOND_BYTES>(a_base[cur]);
                opus_wgtn_fixed_tr_fragment_offset<
                    100, SUBTILE_BYTES, TR_SECOND_BYTES>(a_base[cur]);
                opus_wgtn_fixed_tr_fragment_offset<
                    104, 2 * SUBTILE_BYTES, TR_SECOND_BYTES>(a_base[cur]);
                opus::s_waitcnt_lgkmcnt(opus::number<0>{});
                opus_wgtn_fixed_tr_fragment_offset<
                    108, 3 * SUBTILE_BYTES, TR_SECOND_BYTES>(a_base[cur]);
            }
            opus::static_for<4>([&](auto sm) {
                constexpr int a_reg =
                    96 + ((kpack.value + sm.value) & 3) * 4;
                if constexpr(kpack.value > 0 && sm.value == 1)
                {
                    if constexpr(kpack.value + 1 < NUM_KPACKS)
                        // Current A1 is the oldest of eight outstanding reads.
                        opus::s_waitcnt_lgkmcnt(opus::number<6>{});
                    else
                        opus::s_waitcnt_lgkmcnt(opus::number<2>{});
                }
                if constexpr(kpack.value > 0 && sm.value == 2)
                {
                    if constexpr(kpack.value + 1 < NUM_KPACKS)
                        // Current A2 is now the oldest of ten outstanding reads.
                        opus::s_waitcnt_lgkmcnt(opus::number<8>{});
                    else
                        opus::s_waitcnt_lgkmcnt(opus::number<0>{});
                }
                if constexpr(kpack.value == 0 && sm.value == 3)
                {
                    if constexpr(kpack.value + 1 < NUM_KPACKS)
                        // The stage prologue issued current A3 immediately
                        // before sm=0; ten next-pack reads stay queued.
                        opus::s_waitcnt_lgkmcnt(opus::number<10>{});
                    else
                        opus::s_waitcnt_lgkmcnt(opus::number<0>{});
                }
                if(stage == 0 && kpack.value == 0)
                {
                    opus::static_for<2>([&](auto sn) {
                        constexpr int c = 256 + (sm.value * 2 + sn.value) * 16;
                        opus_wgtn_mfma_zero<
                            a_reg, b_reg + sn.value * 4, c>();
                    });
                }
                else
                {
                    opus::static_for<2>([&](auto sn) {
                        constexpr int c = 256 + (sm.value * 2 + sn.value) * 16;
                        opus_wgtn_mfma_accum<
                            a_reg, b_reg + sn.value * 4, c, c>();
                    });
                }
                if constexpr(kpack.value + 1 < NUM_KPACKS)
                {
                    constexpr int next_b_reg =
                        112 + ((kpack.value + 1) & 1) * 8;
                    if constexpr(sm.value < 2)
                        opus_wgtn_fixed_tr_fragment_offset<
                            next_b_reg + sm.value * 4,
                            (kpack.value + 1) * KPACK_BYTES +
                                sm.value * SUBTILE_BYTES,
                            TR_SECOND_BYTES>(b_base[cur]);
                    constexpr int next_a_slot =
                        (kpack.value + sm.value) & 3;
                    constexpr int next_a_subtile = (sm.value + 3) & 3;
                    opus_wgtn_fixed_tr_fragment_offset<
                        96 + next_a_slot * 4,
                        (kpack.value + 1) * KPACK_BYTES +
                            next_a_subtile * SUBTILE_BYTES,
                        TR_SECOND_BYTES>(a_base[cur]);
                    if constexpr(sm.value == 3)
                        // B/A0/A3 are ready for the next sm=0; A1/A2 may
                        // continue in flight across the pack boundary.
                        opus::s_waitcnt_lgkmcnt(opus::number<4>{});
                }
            });
            if constexpr(HAS_NEXT && kpack.value == 1 &&
                         !PAIR_DIRECT_TO_LDS)
            {
                __builtin_amdgcn_s_setprio(0);
                store_stage(cur ^ 1, 0, next);
                if constexpr(MASKED_NEXT)
                    load_global_tail(stage + 1, 1, next);
                else
                    load_global_valid(stage + 1, 1, next);
                __builtin_amdgcn_s_setprio(1);
            }
        });
        __builtin_amdgcn_s_setprio(0);

        if constexpr(HAS_NEXT)
        {
            if constexpr(PAIR_DIRECT_TO_LDS)
                opus::s_waitcnt_vmcnt(opus::number<0>{});
            else
                store_stage(cur ^ 1, 1, next);
            opus::s_waitcnt_lgkmcnt(opus::number<0>{});
            __syncthreads();
        }
    };

    int dynamic_stage_start = 0;
    constexpr int COMMON_FULL_STAGES = 59;
    if constexpr(FIXED_P == 2048 &&
                 (FIXED_Q == 1024 || FIXED_Q == 2048))
    {
        // The target natural route has at least 3784 rows per expert.  Compile
        // its common 59-stage prefix with a fixed loop bound, then enter the
        // dynamic loop only for the final 1--9 stages.  Stage 59 is prefetched
        // through the masked loader because it is a tail for the shortest
        // experts and a complete stage for the rest.
        if(nroute > COMMON_FULL_STAGES * BK)
        {
#pragma unroll 4
            for(int stage = 0; stage < COMMON_FULL_STAGES - 1; ++stage)
                run_stage(
                    opus::number<1>{},
                    opus::number<0>{},
                    opus::number<4>{},
                    stage);
            run_stage(
                opus::number<1>{},
                opus::number<1>{},
                opus::number<4>{},
                COMMON_FULL_STAGES - 1);
            dynamic_stage_start = COMMON_FULL_STAGES;
        }
    }
    auto run_dynamic_suffix = [&](auto count_tag) {
        constexpr int COUNT = decltype(count_tag)::value;
        opus::static_for<COUNT>([&](auto i) {
            constexpr int stage = COMMON_FULL_STAGES + i.value;
            if constexpr(i.value + 1 < COUNT)
                run_stage(
                    opus::number<1>{},
                    opus::number<0>{},
                    opus::number<4>{},
                    stage);
            else if(has_tail_stage)
                run_stage(
                    opus::number<1>{},
                    opus::number<1>{},
                    opus::number<4>{},
                    stage);
            else
                run_stage(
                    opus::number<0>{},
                    opus::number<0>{},
                    opus::number<4>{},
                    stage);
        });
    };
    if(dynamic_stage_start == COMMON_FULL_STAGES)
    {
        const int residual_stages = full_k_stages - COMMON_FULL_STAGES;
        switch(residual_stages)
        {
        case 0: run_dynamic_suffix(opus::number<0>{}); break;
        case 1: run_dynamic_suffix(opus::number<1>{}); break;
        case 2: run_dynamic_suffix(opus::number<2>{}); break;
        case 3: run_dynamic_suffix(opus::number<3>{}); break;
        case 4: run_dynamic_suffix(opus::number<4>{}); break;
        case 5: run_dynamic_suffix(opus::number<5>{}); break;
        case 6: run_dynamic_suffix(opus::number<6>{}); break;
        case 7: run_dynamic_suffix(opus::number<7>{}); break;
        default: run_dynamic_suffix(opus::number<8>{}); break;
        }
    }
    else for(int stage = dynamic_stage_start; stage < full_k_stages; ++stage)
    {
        if(stage + 1 < full_k_stages)
            run_stage(
                opus::number<1>{},
                opus::number<0>{},
                opus::number<4>{},
                stage);
        else if(has_tail_stage)
            run_stage(
                opus::number<1>{},
                opus::number<1>{},
                opus::number<4>{},
                stage);
        else
            run_stage(
                opus::number<0>{},
                opus::number<0>{},
                opus::number<4>{},
                stage);
    }
      if(has_tail_stage)
      {
          const int tail_kpacks = (nroute - full_k_stages * BK + 15) / 16;
          if(tail_kpacks == 1)
              run_stage(
                  opus::number<0>{}, opus::number<0>{}, opus::number<1>{},
                  full_k_stages);
          else if(tail_kpacks == 2)
              run_stage(
                  opus::number<0>{}, opus::number<0>{}, opus::number<2>{},
                  full_k_stages);
          else if(tail_kpacks == 3)
              run_stage(
                  opus::number<0>{}, opus::number<0>{}, opus::number<3>{},
                  full_k_stages);
          else
              run_stage(
                  opus::number<0>{}, opus::number<0>{}, opus::number<4>{},
                  full_k_stages);
      }
    }

    const int lm = lane % 32;
    // The specialized aligned shapes occupy at most 2^28 BF16 elements, so a
    // 32-bit element offset covers the complete output.  Keep dW in a buffer
    // resource and avoid rebuilding a 64-bit flat address for every scalar
    // accumulator store.
    auto g_dW = opus::make_gmem(reinterpret_cast<opus::bf16_t*>(dW));
    const int dW_e = e * P * Q;
    opus::static_for<4>([&](auto sm) {
        const int p_base = m0 + wm * 128 + sm.value * 32 + (lane / 32) * 4;
        opus::static_for<2>([&](auto sn) {
            const int q = n0 + wn * 64 + sn.value * 32 + lm;
            opus::static_for<16>([&](auto i) {
                constexpr int c = 256 + (sm.value * 2 + sn.value) * 16 + i.value;
                const int p = p_base + (i.value / 4) * 8 + (i.value % 4);
                const uint32_t bits = opus_wgtn_read_acc<c>();
                const float value = __builtin_bit_cast(float, bits);
                g_dW.store(opus::fp32_to_bf16(value), dW_e + p * Q + q);
            });
        });
    });
}

inline void opus_moe_wgrad_tn_launch_gfx950(const __bf16* dy, const __bf16* a,
                                            const int32_t* offs, __bf16* dW,
                                            int E, int P, int Q, int uniform_m,
                                            hipStream_t stream) {
    if(P % 256 == 0 && Q % 256 == 0)
    {
        const bool interleave_experts =
            E == 64 && P == 2048 && Q == 2048 && uniform_m != 4096;
        dim3 grid = interleave_experts
            ? dim3(E * (P / 256) * (Q / 256), 1, 1)
            : dim3(Q / 256, P / 256, E);
        if(uniform_m == 4096 && P == 2048 && Q == 1024)
            opus_moe_wgrad_tn_8wave_kernel<2048, 1024, 4096>
                <<<grid, 512, 0, stream>>>(dy, a, offs, dW, P, Q);
        else if(uniform_m == 4096 && P == 2048 && Q == 2048)
            opus_moe_wgrad_tn_8wave_kernel<2048, 2048, 4096>
                <<<grid, 512, 0, stream>>>(dy, a, offs, dW, P, Q);
        else if(P == 2048 && Q == 1024)
            opus_moe_wgrad_tn_8wave_kernel<2048, 1024, 0>
                <<<grid, 512, 0, stream>>>(dy, a, offs, dW, P, Q);
        else if(P == 2048 && Q == 2048)
        {
            if(interleave_experts)
                opus_moe_wgrad_tn_8wave_kernel<2048, 2048, 0, true>
                    <<<grid, 512, 0, stream>>>(dy, a, offs, dW, P, Q);
            else
                opus_moe_wgrad_tn_8wave_kernel<2048, 2048, 0>
                    <<<grid, 512, 0, stream>>>(dy, a, offs, dW, P, Q);
        }
        else
            opus_moe_wgrad_tn_8wave_kernel<><<<grid, 512, 0, stream>>>(
                dy, a, offs, dW, P, Q);
        return;
    }
    dim3 grid((Q + OPUS_WGTN_BN - 1) / OPUS_WGTN_BN,
              (P + OPUS_WGTN_BM - 1) / OPUS_WGTN_BM, E);
    dim3 block(OPUS_WGTN_BLOCK);
    opus_moe_wgrad_tn_lds_tr_kernel<<<grid, block, 0, stream>>>(dy, a, offs, dW, P, Q);
}
