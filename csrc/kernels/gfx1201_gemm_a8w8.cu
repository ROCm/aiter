// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx1201 (RDNA4) FP8 GEMM with preshuffled weights.
// Tile 128 x 256 per 8-wave workgroup, each wave owns 64 rows x 64 columns. The weights are
// preshuffled offline into WMMA A-operand order and streamed straight into VGPRs (no LDS);
// activations go through a double-buffered LDS tile. Workgroups sweep GROUP_M row blocks per
// column block so live tiles share X and W in the L2 / MALL.
#include <hip/hip_runtime.h>
#include <stdint.h>

#include <stdexcept>

#include "aiter_stream.h"
#include "gfx1201_gemm_a8w8.h"

namespace {

typedef int v2i __attribute__((ext_vector_type(2)));
typedef float v8f __attribute__((ext_vector_type(8)));

constexpr int BM = 128, BN = 256, BK = 64, WN = 2, MT = 8 / WN, LDS_STRIDE = BK + 16;
constexpr int XV = BK / 32, WV = 4 * WN, GROUP_M = 16, THREADS = 256;

// 32-bit unsigned offset from a uniform base keeps loads in the saddr + voffset form.
__device__ inline uint4 ld16(const uint8_t* base, int off)
{
    return *reinterpret_cast<const uint4*>(base + (size_t)(unsigned)off);
}

// The kernel is only launched on gfx1201; other targets of a multi-arch build compile a stub.
__device__ inline v8f wmma_fp8(v2i a, v2i b, v8f c)
{
#if defined(__gfx1201__) || defined(__gfx12__)
    return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
#else
    (void)a;
    (void)b;
    return c;
#endif
}

__device__ inline uint32_t bf16_rne(float f)
{
    const uint32_t u = __float_as_uint(f);
    return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
}

__global__ void __launch_bounds__(THREADS)
    gfx1201_gemm_a8w8_bpreshuffle_kernel(const uint8_t* __restrict__ X,
                                         const uint8_t* __restrict__ Wp,
                                         uint16_t* __restrict__ Y,
                                         int M,
                                         int N,
                                         int K,
                                         const float* __restrict__ xs,
                                         const float* __restrict__ ws,
                                         int xs_stride,
                                         int ws_stride)
{
    __shared__ __attribute__((aligned(16))) uint8_t lds[2][BM * LDS_STRIDE];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, h = lane >> 4, r = lane & 15;
    const int nm = (M + BM - 1) / BM, nn = N / BN, pid = blockIdx.x;
    const int per = GROUP_M * nn, first = (pid / per) * GROUP_M, gsz = min(nm - first, GROUP_M);
    const int m0 = (first + (pid % per) % gsz) * BM, n0 = ((pid % per) / gsz) * BN;
    const int KT = K / BK;

    // Activations: thread owns row tid / 2, 32 bytes at half tid & 1; tail rows reread row M - 1.
    const int arow = tid >> 1, ahalf = tid & 1;
    const int xoff = min(m0 + arow, M - 1) * K + (BK / 2) * ahalf;
    const int aoff = arow * LDS_STRIDE + (BK / 2) * ahalf;
    // Weights preshuffled as [N / 32][K / 64][lane 32][64 bytes].
    const int wn_i = wave % (8 / WN), wm_i = wave / (8 / WN), row0 = wm_i * MT * 16;
    const int woff      = (((n0 / 32 + WN * wn_i) * KT) * 32 + lane) * 64;
    const int wstride_n = KT * 32 * 64;
    auto wofs           = [&](int j) { return (j / 4) * wstride_n + 16 * (j % 4); };

    v8f acc[2 * WN][MT];
#pragma unroll
    for(int t = 0; t < 2 * WN; ++t)
#pragma unroll
        for(int i = 0; i < MT; ++i)
            acc[t][i] = v8f{0, 0, 0, 0, 0, 0, 0, 0};

    uint4 xa[XV], wb[WV], wn[WV];
#pragma unroll
    for(int i = 0; i < XV; ++i)
        xa[i] = ld16(X, xoff + 16 * i);
#pragma unroll
    for(int j = 0; j < WV; ++j)
        wb[j] = ld16(Wp, woff + wofs(j));
#pragma unroll
    for(int i = 0; i < XV; ++i)
        reinterpret_cast<uint4*>(&lds[0][aoff])[i] = xa[i];
    __syncthreads();

    auto step = [&](int kt, uint4(&wc)[WV], uint4(&wx)[WV]) __attribute__((always_inline))
    {
        const bool more = kt + 1 < KT;
        if(more)
        {
            const int xo = xoff + (kt + 1) * BK, wo = woff + (kt + 1) * 2048;
#pragma unroll
            for(int i = 0; i < XV; ++i)
                xa[i] = ld16(X, xo + 16 * i);
#pragma unroll
            for(int j = 0; j < WV; ++j)
                wx[j] = ld16(Wp, wo + wofs(j));
        }
        const uint8_t* abuf = lds[kt & 1];
        v2i bw[2 * WN][4];
#pragma unroll
        for(int g = 0; g < WN; ++g)
#pragma unroll
            for(int t = 0; t < 2; ++t)
            {
                const uint4 w0 = wc[g * 4 + 2 * t], w1 = wc[g * 4 + 2 * t + 1];
                bw[2 * g + t][0] = v2i{(int)w0.x, (int)w0.y};
                bw[2 * g + t][1] = v2i{(int)w0.z, (int)w0.w};
                bw[2 * g + t][2] = v2i{(int)w1.x, (int)w1.y};
                bw[2 * g + t][3] = v2i{(int)w1.z, (int)w1.w};
            }
#pragma unroll
        for(int mt = 0; mt < MT; ++mt)
        {
            const uint4* ap =
                reinterpret_cast<const uint4*>(abuf + (row0 + mt * 16 + r) * LDS_STRIDE + 32 * h);
            const uint4 a0 = ap[0], a1 = ap[1];
            const v2i ax[4] = {v2i{(int)a0.x, (int)a0.y},
                               v2i{(int)a0.z, (int)a0.w},
                               v2i{(int)a1.x, (int)a1.y},
                               v2i{(int)a1.z, (int)a1.w}};
#pragma unroll
            for(int t = 0; t < 2 * WN; ++t)
#pragma unroll
                for(int s = 0; s < 4; ++s)
                    acc[t][mt] = wmma_fp8(bw[t][s], ax[s], acc[t][mt]);
        }
        if(more)
        {
            uint4* d = reinterpret_cast<uint4*>(&lds[(kt + 1) & 1][aoff]);
#pragma unroll
            for(int i = 0; i < XV; ++i)
                d[i] = xa[i];
        }
        __syncthreads();
    };
    // K is a multiple of 2 * BK: two steps per iteration swap the weight register buffers.
    for(int kt = 0; kt < KT; kt += 2)
    {
        step(kt, wb, wn);
        step(kt + 1, wn, wb);
    }

    // Column n of acc[2g + t][mt][e] is n0 + (WN * wn_i + g) * 32 + 16 * h + 8 * t + e.
    float wsc[WN][16];
#pragma unroll
    for(int g = 0; g < WN; ++g)
    {
        if(ws_stride)
        {
            const float4* q =
                reinterpret_cast<const float4*>(ws + n0 + (WN * wn_i + g) * 32 + 16 * h);
#pragma unroll
            for(int c = 0; c < 4; ++c)
            {
                const float4 v    = q[c];
                wsc[g][4 * c]     = v.x;
                wsc[g][4 * c + 1] = v.y;
                wsc[g][4 * c + 2] = v.z;
                wsc[g][4 * c + 3] = v.w;
            }
        }
        else
        {
#pragma unroll
            for(int c = 0; c < 16; ++c)
                wsc[g][c] = ws[0];
        }
    }
#pragma unroll
    for(int mt = 0; mt < MT; ++mt)
    {
        const int m = m0 + row0 + mt * 16 + r;
        if(m >= M)
            continue;
        const float xscale = xs[m * xs_stride];
#pragma unroll
        for(int g = 0; g < WN; ++g)
        {
            uint32_t p[8];
#pragma unroll
            for(int t = 0; t < 2; ++t)
#pragma unroll
                for(int e = 0; e < 8; e += 2)
                    p[t * 4 + e / 2] =
                        bf16_rne(acc[2 * g + t][mt][e] * (xscale * wsc[g][8 * t + e])) |
                        (bf16_rne(acc[2 * g + t][mt][e + 1] * (xscale * wsc[g][8 * t + e + 1]))
                         << 16);
            uint4* dst =
                reinterpret_cast<uint4*>(Y + (size_t)m * N + n0 + (WN * wn_i + g) * 32 + 16 * h);
            dst[0] = uint4{p[0], p[1], p[2], p[3]};
            dst[1] = uint4{p[4], p[5], p[6], p[7]};
        }
    }
}

} // namespace

void gfx1201_gemm_a8w8_bpreshuffle_hip(const aiter_tensor_t& XQ,
                                       const aiter_tensor_t& WQ,
                                       const aiter_tensor_t& x_scale,
                                       const aiter_tensor_t& w_scale,
                                       aiter_tensor_t& Y)
{
    if(XQ.ndim != 2 || WQ.ndim != 2 || Y.ndim != 2)
        throw std::invalid_argument("gfx1201_gemm_a8w8_bpreshuffle: XQ, WQ and Y must be 2-D");
    if(XQ.dtype() != AITER_DTYPE_fp8 || WQ.dtype() != AITER_DTYPE_fp8 ||
       Y.dtype() != AITER_DTYPE_bf16 || x_scale.dtype() != AITER_DTYPE_fp32 ||
       w_scale.dtype() != AITER_DTYPE_fp32)
        throw std::invalid_argument(
            "gfx1201_gemm_a8w8_bpreshuffle: expects fp8 XQ/WQ, fp32 scales and bf16 Y");
    if(!XQ.is_contiguous() || !WQ.is_contiguous() || !Y.is_contiguous() ||
       !x_scale.is_contiguous() || !w_scale.is_contiguous())
        throw std::invalid_argument("gfx1201_gemm_a8w8_bpreshuffle: tensors must be contiguous");
    const int64_t M = XQ.size(0), K = XQ.size(1), N = WQ.size(0);
    if(WQ.size(1) != K || Y.size(0) != M || Y.size(1) != N)
        throw std::invalid_argument("gfx1201_gemm_a8w8_bpreshuffle: shape mismatch");
    if(M < 1 || N % BN != 0 || K % (2 * BK) != 0)
        throw std::invalid_argument(
            "gfx1201_gemm_a8w8_bpreshuffle: needs N % 256 == 0 and K % 128 == 0");
    if(M * K >= (int64_t(1) << 31) || N * K >= (int64_t(1) << 31))
        throw std::invalid_argument(
            "gfx1201_gemm_a8w8_bpreshuffle: M * K and N * K must be below 2^31");
    const int64_t xn = (int64_t)x_scale.numel(), wnumel = (int64_t)w_scale.numel();
    if(wnumel != 1 && ((uintptr_t)w_scale.data_ptr() & 15) != 0)
        throw std::invalid_argument(
            "gfx1201_gemm_a8w8_bpreshuffle: per-channel w_scale must be 16-byte aligned");
    if((xn != 1 && xn != M) || (wnumel != 1 && wnumel != N))
        throw std::invalid_argument(
            "gfx1201_gemm_a8w8_bpreshuffle: x_scale must have M or 1 elements, w_scale N or 1");
    const int blocks = (int)(((M + BM - 1) / BM) * (N / BN));
    hipLaunchKernelGGL(gfx1201_gemm_a8w8_bpreshuffle_kernel,
                       dim3(blocks),
                       dim3(THREADS),
                       0,
                       aiter::getCurrentHIPStream(),
                       (const uint8_t*)XQ.data_ptr(),
                       (const uint8_t*)WQ.data_ptr(),
                       (uint16_t*)Y.data_ptr(),
                       (int)M,
                       (int)N,
                       (int)K,
                       (const float*)x_scale.data_ptr(),
                       (const float*)w_scale.data_ptr(),
                       xn == 1 ? 0 : 1,
                       wnumel == 1 ? 0 : 1);
}
