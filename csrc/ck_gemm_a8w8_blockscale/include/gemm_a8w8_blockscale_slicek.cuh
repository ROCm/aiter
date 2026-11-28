#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

// Deterministic small-M a8w8 blockscale GEMM for gfx950 with intra-workgroup
// slice-K: each workgroup owns one (16 * MRep) x (16 * NRep) output tile and
// its NWaves waves take the 128-wide K blocks w, w + NWaves, ... Their FP32
// partials are added in LDS in wave order, so no output atomics race. With
// KSplit = 2 each K half is rounded to EDataType before one atomic add into a
// zeroed output: r(r(a) + r(b)) is the same in either arrival order.

#ifdef USE_ROCM

#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <torch/extension.h>

#include "ck/ck.hpp"
#include "ck/host_utility/hip_check_error.hpp"
#include "ck/utility/type_convert.hpp"

#include <algorithm>

#include "gemm_dispatch_utils.h"

namespace aiter_slicek {

using i32x8_t = int __attribute__((ext_vector_type(8)));
using f32x4_t = float __attribute__((ext_vector_type(4)));

template <typename EDataType>
__device__ inline void atomic_add_pair(EDataType* p, float a, float b)
{
    if constexpr(std::is_same_v<EDataType, ck::bhalf_t>)
    {
        using bf16x2_t = short __attribute__((ext_vector_type(2)));
        bf16x2_t v{ck::bit_cast<short>(ck::type_convert<ck::bhalf_t>(a)),
                   ck::bit_cast<short>(ck::type_convert<ck::bhalf_t>(b))};
        __builtin_amdgcn_global_atomic_fadd_v2bf16(reinterpret_cast<bf16x2_t*>(p), v);
    }
    else
    {
        using f16x2_t = _Float16 __attribute__((ext_vector_type(2)));
        f16x2_t v{static_cast<_Float16>(a), static_cast<_Float16>(b)};
        __builtin_amdgcn_global_atomic_fadd_v2f16(reinterpret_cast<f16x2_t*>(p), v);
    }
}

template <int MRep, int NRep, int NWaves, int Unroll, int KSplit, typename EDataType>
__global__ __launch_bounds__(NWaves * 64) void kernel(const uint8_t* __restrict__ A,
                                                      const uint8_t* __restrict__ B,
                                                      const float* __restrict__ xs,
                                                      const float* __restrict__ ws,
                                                      EDataType* __restrict__ Y,
                                                      int M,
                                                      int K,
                                                      int lda,
                                                      int ldb,
                                                      int ldy,
                                                      int ldxs,
                                                      int ldws)
{
#if defined(__gfx950__)
    constexpr int MT = 16 * MRep;
    constexpr int NT = 16 * NRep;
    // x scales of the M tile as [KB][MT] during the main loop, then wave partials.
    extern __shared__ float smem[];
    const int lane = threadIdx.x & 63;
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
    const int n0   = blockIdx.x * NT;
    const int m0   = blockIdx.y * MT;
    const int KB   = K >> 7;
    const int KBs  = (KB + KSplit - 1) / KSplit;
    const int kbb  = KSplit == 1 ? 0 : blockIdx.z * KBs;
    const int kbe  = KSplit == 1 ? KB : min(KB, kbb + KBs);
    // 16x16x128 MFMA operand layout: lane holds row lane % 16, K bytes [32 * (lane / 16), +32).
    const int kq  = (lane >> 4) * 32;
    const int r16 = lane & 15;

    const uint8_t* bptr[NRep];
#pragma unroll
    for(int j = 0; j < NRep; j++)
        bptr[j] = B + (int64_t)(n0 + j * 16 + r16) * ldb + kq;
    const uint8_t* aptr[MRep];
    bool avalid[MRep];
#pragma unroll
    for(int i = 0; i < MRep; i++)
    {
        const int m = m0 + i * 16 + r16;
        avalid[i]   = m < M;
        aptr[i]     = A + (int64_t)(avalid[i] ? m : 0) * lda + kq;
    }
    const float* wsp[NRep];
#pragma unroll
    for(int j = 0; j < NRep; j++)
        wsp[j] = ws + (int64_t)((n0 + j * 16) >> 7) * ldws;

    f32x4_t acc[MRep][NRep];
#pragma unroll
    for(int i = 0; i < MRep; i++)
#pragma unroll
        for(int j = 0; j < NRep; j++)
            acc[i][j] = f32x4_t{0.f, 0.f, 0.f, 0.f};

    i32x8_t bbuf[Unroll][NRep];
    i32x8_t abuf[Unroll][MRep];
    auto load = [&](int kb0) {
#pragma unroll
        for(int u = 0; u < Unroll; u++)
        {
            const int kb = kb0 + u * NWaves;
            if(kb < kbe)
            {
#pragma unroll
                for(int j = 0; j < NRep; j++)
                    bbuf[u][j] = *reinterpret_cast<const i32x8_t*>(bptr[j] + (kb << 7));
#pragma unroll
                for(int i = 0; i < MRep; i++)
                    abuf[u][i] = avalid[i] ? *reinterpret_cast<const i32x8_t*>(aptr[i] + (kb << 7))
                                           : i32x8_t{0, 0, 0, 0, 0, 0, 0, 0};
            }
        }
    };
    auto compute = [&](int kb0) {
#pragma unroll
        for(int u = 0; u < Unroll; u++)
        {
            const int kb = kb0 + u * NWaves;
            if(kb < kbe)
            {
                f32x4_t sx[MRep];
#pragma unroll
                for(int i = 0; i < MRep; i++)
                    sx[i] = *reinterpret_cast<const f32x4_t*>(
                        &smem[kb * MT + i * 16 + (lane >> 4) * 4]);
#pragma unroll
                for(int j = 0; j < NRep; j++)
                {
                    const float sw = wsp[j][kb];
#pragma unroll
                    for(int i = 0; i < MRep; i++)
                    {
                        // fp8 x fp8 (format 0) with unit MX scales.
                        const f32x4_t c = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                            abuf[u][i], bbuf[u][j], f32x4_t{0.f, 0.f, 0.f, 0.f}, 0, 0, 0, 0, 0, 0);
                        acc[i][j] += c * (sx[i] * sw);
                    }
                }
            }
        }
    };

    int kb0 = kbb + wave;
    load(kb0);
    // Stage x scales behind the first operand loads.
    for(int e = threadIdx.x; e < MT * KB; e += NWaves * 64)
    {
        const int m       = e / KB;
        const int kb      = e - m * KB;
        smem[kb * MT + m] = (m0 + m < M) ? xs[(int64_t)(m0 + m) * ldxs + kb] : 0.f;
    }
    __syncthreads();
    while(true)
    {
        compute(kb0);
        kb0 += NWaves * Unroll;
        if(kb0 >= kbe)
            break;
        load(kb0);
    }
    __syncthreads();

    // C layout: lane holds column lane % 16 of rows 4 * (lane / 16) + v.
    constexpr int E = MRep * NRep * 4 * 64;
#pragma unroll
    for(int i = 0; i < MRep; i++)
#pragma unroll
        for(int j = 0; j < NRep; j++)
#pragma unroll
            for(int v = 0; v < 4; v++)
                smem[wave * E + ((i * NRep + j) * 4 + v) * 64 + lane] = acc[i][j][v];
    __syncthreads();
    auto out_index = [&](int e, int& m) {
        const int l  = e & 63;
        const int v  = (e >> 6) & 3;
        const int ij = e >> 8;
        m            = m0 + (ij / NRep) * 16 + (l >> 4) * 4 + v;
        return (int64_t)m * ldy + n0 + (ij % NRep) * 16 + (l & 15);
    };
    if constexpr(KSplit == 1)
    {
        for(int e = threadIdx.x; e < E; e += NWaves * 64)
        {
            float s = smem[e];
#pragma unroll
            for(int w = 1; w < NWaves; w++)
                s += smem[w * E + e];
            int m;
            const int64_t idx = out_index(e, m);
            if(m < M)
                Y[idx] = ck::type_convert<EDataType>(s);
        }
    }
    else
    {
        static_assert(KSplit == 2, "more than two K partials would race in the output");
        for(int e = threadIdx.x * 2; e < E; e += NWaves * 128)
        {
            float s0 = smem[e], s1 = smem[e + 1];
#pragma unroll
            for(int w = 1; w < NWaves; w++)
            {
                s0 += smem[w * E + e];
                s1 += smem[w * E + e + 1];
            }
            int m;
            const int64_t idx = out_index(e, m);
            if(m < M)
                atomic_add_pair(Y + idx, s0, s1);
        }
    }
#endif
}

} // namespace aiter_slicek

template <typename DDataType,
          typename EDataType,
          int MRep,
          int NRep,
          int NWaves,
          int Unroll,
          int KSplit>
torch::Tensor gemm_a8w8_blockscale_slicek_impl(torch::Tensor& XQ,
                                               torch::Tensor& WQ,
                                               torch::Tensor& x_scale,
                                               torch::Tensor& w_scale,
                                               torch::Tensor& Y,
                                               int KBatch)
{
    static_assert(std::is_same_v<DDataType, float>);
    const int M = XQ.size(0);
    const int N = WQ.size(0);
    const int K = XQ.size(1);
    TORCH_CHECK(KBatch == 1, "slice-K blockscale kernels split K internally; splitK must be 0");
    TORCH_CHECK(K % 128 == 0 && N % (16 * NRep) == 0,
                "slice-K blockscale kernel needs K % 128 == 0 and N % ",
                16 * NRep,
                " == 0, got N=",
                N,
                " K=",
                K);
    TORCH_CHECK(XQ.stride(-1) == 1 && WQ.stride(-1) == 1 && Y.stride(-1) == 1 &&
                    x_scale.stride(-1) == 1 && w_scale.stride(-1) == 1,
                "slice-K blockscale kernel needs contiguous rows");
    TORCH_CHECK(XQ.stride(0) % 16 == 0 && WQ.stride(0) % 16 == 0 &&
                    reinterpret_cast<uintptr_t>(XQ.data_ptr()) % 16 == 0 &&
                    reinterpret_cast<uintptr_t>(WQ.data_ptr()) % 16 == 0,
                "slice-K blockscale kernel needs 16-byte aligned rows");
    TORCH_CHECK(x_scale.size(0) >= M && x_scale.size(1) >= K / 128 &&
                    w_scale.size(0) >= (N + 127) / 128 && w_scale.size(1) >= K / 128,
                "slice-K blockscale kernel: scale shapes do not cover the GEMM");
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(XQ));
    TORCH_CHECK(get_device_gfx() == "gfx950", "slice-K blockscale kernels need gfx950");
    if(M == 0)
        return Y;

    const hipStream_t stream = at::hip::getCurrentHIPStream();
    static_assert(NWaves * MRep * NRep * 256 * sizeof(float) <= 64 * 1024,
                  "wave partials must fit in 64 KB of LDS");
    const size_t lds = sizeof(float) * std::max(16 * MRep * (K / 128), NWaves * MRep * NRep * 256);
    TORCH_CHECK(lds <= 64 * 1024, "slice-K blockscale kernel: K=", K, " is too large");
    if constexpr(KSplit > 1)
    {
        HIP_CHECK_ERROR(hipMemset2DAsync(
            Y.data_ptr(), Y.stride(0) * Y.element_size(), 0, N * Y.element_size(), M, stream));
    }
    const dim3 grid(N / (16 * NRep), (M + 16 * MRep - 1) / (16 * MRep), KSplit);
    hipLaunchKernelGGL((aiter_slicek::kernel<MRep, NRep, NWaves, Unroll, KSplit, EDataType>),
                       grid,
                       dim3(NWaves * 64),
                       lds,
                       stream,
                       reinterpret_cast<const uint8_t*>(XQ.data_ptr()),
                       reinterpret_cast<const uint8_t*>(WQ.data_ptr()),
                       x_scale.data_ptr<float>(),
                       w_scale.data_ptr<float>(),
                       reinterpret_cast<EDataType*>(Y.data_ptr()),
                       M,
                       K,
                       static_cast<int>(XQ.stride(0)),
                       static_cast<int>(WQ.stride(0)),
                       static_cast<int>(Y.stride(0)),
                       static_cast<int>(x_scale.stride(0)),
                       static_cast<int>(w_scale.stride(0)));
    HIP_CHECK_ERROR(hipGetLastError());
    return Y;
}

#endif // USE_ROCM
