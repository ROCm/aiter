// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

#include "aiter_tensor.h"
#include "aiter_stream.h"
#include "opus_moe_bwd.h"
#include "gfx950/opus_moe_dgrad_mfma_gfx950.cuh"
#include "gfx950/opus_moe_dgrad_swiglu_gfx950.cuh"
#include "gfx950/opus_moe_wgrad_mfma_gfx950.cuh"
#include "gfx950/opus_moe_wgrad_tn_gfx950.cuh"

// Naive K-grouped wgrad: one thread per output element (e, p, q), looping over
// the expert's routes as the contraction. Correctness-first (M1); MFMA/LDS
// tiling comes later.
__global__ void opus_moe_wgrad_bf16_naive_kernel(const hip_bfloat16* __restrict__ dy, // [M,P]
                                                 const hip_bfloat16* __restrict__ a,  // [M,Q]
                                                 const int32_t* __restrict__ offs,    // [E+1]
                                                 float* __restrict__ dW,              // [E,P,Q]
                                                 int P,
                                                 int Q)
{
    const int e = blockIdx.z;
    const int p = blockIdx.y * blockDim.y + threadIdx.y; // over P (dy cols)
    const int q = blockIdx.x * blockDim.x + threadIdx.x; // over Q (a cols)
    if(p >= P || q >= Q)
        return;

    const int start = offs[e];
    const int end   = offs[e + 1];
    float acc       = 0.0f;
    for(int m = start; m < end; ++m)
    {
        acc += static_cast<float>(dy[static_cast<int64_t>(m) * P + p]) *
               static_cast<float>(a[static_cast<int64_t>(m) * Q + q]);
    }
    dW[(static_cast<int64_t>(e) * P + p) * Q + q] = acc;
}

void opus_moe_wgrad_bf16(aiter_tensor_t& dy,
                         aiter_tensor_t& a,
                         aiter_tensor_t& expert_offsets,
                         aiter_tensor_t& dW)
{
    const int P = static_cast<int>(dy.size(1));
    const int Q = static_cast<int>(a.size(1));
    const int E = static_cast<int>(dW.size(0));

    dim3 block(16, 16, 1);
    dim3 grid((Q + 15) / 16, (P + 15) / 16, E);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    opus_moe_wgrad_bf16_naive_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const hip_bfloat16*>(dy.data_ptr()),
        reinterpret_cast<const hip_bfloat16*>(a.data_ptr()),
        reinterpret_cast<const int32_t*>(expert_offsets.data_ptr()),
        reinterpret_cast<float*>(dW.data_ptr()),
        P,
        Q);
}

// Naive M-grouped dgrad: one thread per output (m, n), looking up row m's expert
// and contracting over K. Correctness-first (M2); MFMA/Wᵀ tiling comes later.
__global__ void opus_moe_dgrad_bf16_naive_kernel(const hip_bfloat16* __restrict__ dy,   // [M,K]
                                                 const hip_bfloat16* __restrict__ w,    // [E,K,N]
                                                 const int32_t* __restrict__ row_expert, // [M]
                                                 hip_bfloat16* __restrict__ dh,         // [M,N]
                                                 int M,
                                                 int K,
                                                 int N)
{
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if(m >= M || n >= N)
        return;

    const int e = row_expert[m];
    const int64_t w_base = (static_cast<int64_t>(e) * K) * N + n;
    const int64_t dy_base = static_cast<int64_t>(m) * K;
    float acc = 0.0f;
    for(int k = 0; k < K; ++k)
    {
        acc += static_cast<float>(dy[dy_base + k]) *
               static_cast<float>(w[w_base + static_cast<int64_t>(k) * N]);
    }
    dh[static_cast<int64_t>(m) * N + n] = hip_bfloat16(acc);
}

void opus_moe_dgrad_bf16(aiter_tensor_t& dy,
                         aiter_tensor_t& w,
                         aiter_tensor_t& row_expert,
                         aiter_tensor_t& dh)
{
    const int M = static_cast<int>(dy.size(0));
    const int K = static_cast<int>(dy.size(1));
    const int N = static_cast<int>(dh.size(1));

    dim3 block(16, 16, 1);
    dim3 grid((N + 15) / 16, (M + 15) / 16, 1);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    opus_moe_dgrad_bf16_naive_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const hip_bfloat16*>(dy.data_ptr()),
        reinterpret_cast<const hip_bfloat16*>(w.data_ptr()),
        reinterpret_cast<const int32_t*>(row_expert.data_ptr()),
        reinterpret_cast<hip_bfloat16*>(dh.data_ptr()),
        M,
        K,
        N);
}

// Fused opus-MFMA grouped dgrad (BF16). COMPACT (unpadded) layout: dy/dh stay
// expert-grouped compact [M,*]; the per-block tables map each B_M tile to its
// expert + compact row range (built cheaply from offs, no operand padding).
//   dy [M,K] bf16, w [E,N,K] bf16, sorted_expert_ids/block_m_start/block_m_end
//   [num_blocks] i32, dh [M,N] bf16.  dh[m,:] = dy[m,:] @ w[expert(block)].
void opus_moe_dgrad_mfma_bf16(aiter_tensor_t& dy,
                              aiter_tensor_t& w,
                              aiter_tensor_t& sorted_expert_ids,
                              aiter_tensor_t& block_m_start,
                              aiter_tensor_t& block_m_end,
                              aiter_tensor_t& dh)
{
    opus_moe_dgrad_mfma_kargs k;
    k.ptr_a             = dy.data_ptr();
    k.ptr_b             = w.data_ptr();
    k.ptr_c             = dh.data_ptr();
    k.sorted_expert_ids = reinterpret_cast<const int32_t*>(sorted_expert_ids.data_ptr());
    k.block_m_start     = reinterpret_cast<const int32_t*>(block_m_start.data_ptr());
    k.block_m_end       = reinterpret_cast<const int32_t*>(block_m_end.data_ptr());
    k.num_blocks        = static_cast<int>(sorted_expert_ids.size(0));
    k.m                 = static_cast<int>(dy.size(0));
    k.n                 = static_cast<int>(dh.size(1));
    k.k                 = static_cast<int>(dy.size(1));
    k.stride_a          = static_cast<int>(dy.stride(0));
    k.stride_b          = static_cast<int>(w.stride(1));
    k.stride_c          = static_cast<int>(dh.stride(0));
    k.stride_b_expert   = static_cast<int64_t>(w.stride(0));
    opus_moe_dgrad_mfma_launch_gfx950(k, aiter::getCurrentHIPStream());
}

void opus_moe_dgrad_swiglu_bf16(aiter_tensor_t& dy,
                                aiter_tensor_t& w,
                                aiter_tensor_t& act_input,
                                aiter_tensor_t& d_act_input,
                                int uniform_m)
{
    const int E = static_cast<int>(w.size(0));
    const int N = static_cast<int>(w.size(1));
    const int K = static_cast<int>(w.size(2));
    AITER_CHECK(uniform_m > 0, "uniform_m must be positive");
    AITER_CHECK(static_cast<int>(dy.size(0)) == E * uniform_m,
                "dy rows must equal E * uniform_m");
    AITER_CHECK(static_cast<int>(dy.size(1)) == K, "dy K must match weight K");
    AITER_CHECK(static_cast<int>(act_input.size(0)) == E * uniform_m,
                "act_input rows must equal E * uniform_m");
    AITER_CHECK(static_cast<int>(act_input.size(1)) == 2 * N,
                "act_input width must equal 2 * N");
    AITER_CHECK(static_cast<int>(d_act_input.size(0)) == E * uniform_m &&
                    static_cast<int>(d_act_input.size(1)) == 2 * N,
                "d_act_input must have shape [E * uniform_m, 2 * N]");
    AITER_CHECK(K >= 128 && K % 64 == 0,
                "fused mono dgrad requires K >= 128 and divisible by 64");
    AITER_CHECK(N % 256 == 0, "fused mono dgrad requires N divisible by 256");

    opus_moe_dgrad_swiglu_kargs k{};
    k.ptr_a = dy.data_ptr();
    k.ptr_b = w.data_ptr();
    k.ptr_act_input = act_input.data_ptr();
    k.ptr_dact = d_act_input.data_ptr();
    k.m = uniform_m;
    k.n = N;
    k.k = K;
    k.batch = E;
    k.stride_a = static_cast<int>(dy.stride(0));
    k.stride_b = static_cast<int>(w.stride(1));
    k.stride_act_input = static_cast<int>(act_input.stride(0));
    k.stride_dact = static_cast<int>(d_act_input.stride(0));
    k.stride_a_batch = uniform_m * k.stride_a;
    k.stride_b_batch = static_cast<int>(w.stride(0));
    k.stride_act_input_batch = uniform_m * k.stride_act_input;
    k.stride_dact_batch = uniform_m * k.stride_dact;
    opus_moe_dgrad_swiglu_launch_gfx950(k, aiter::getCurrentHIPStream());
}

void opus_moe_dgrad_swiglu_dscore_bf16(aiter_tensor_t& dy,
                                       aiter_tensor_t& w,
                                       aiter_tensor_t& act_input,
                                       aiter_tensor_t& d_act_input,
                                       aiter_tensor_t& dscore_partials,
                                       int uniform_m)
{
    const int E = static_cast<int>(w.size(0));
    const int N = static_cast<int>(w.size(1));
    const int K = static_cast<int>(w.size(2));
    AITER_CHECK(uniform_m > 0, "uniform_m must be positive");
    AITER_CHECK(static_cast<int>(dy.size(0)) == E * uniform_m &&
                    static_cast<int>(dy.size(1)) == K,
                "dy must have shape [E * uniform_m, K]");
    AITER_CHECK(static_cast<int>(act_input.size(0)) == E * uniform_m &&
                    static_cast<int>(act_input.size(1)) == 2 * N,
                "act_input must have shape [E * uniform_m, 2 * N]");
    AITER_CHECK(static_cast<int>(d_act_input.size(0)) == E * uniform_m &&
                    static_cast<int>(d_act_input.size(1)) == 2 * N,
                "d_act_input must have shape [E * uniform_m, 2 * N]");
    AITER_CHECK(static_cast<int>(dscore_partials.size(0)) == E * uniform_m &&
                    static_cast<int>(dscore_partials.size(1)) == N / 256,
                "dscore_partials must have shape [E * uniform_m, N / 256]");
    AITER_CHECK(K >= 128 && K % 64 == 0 && N % 256 == 0,
                "fused dscore path requires K%64==0 and N%256==0");

    opus_moe_dgrad_swiglu_kargs k{};
    k.ptr_a = dy.data_ptr();
    k.ptr_b = w.data_ptr();
    k.ptr_act_input = act_input.data_ptr();
    k.ptr_dact = d_act_input.data_ptr();
    k.ptr_dscore_partials = dscore_partials.data_ptr();
    k.m = uniform_m;
    k.n = N;
    k.k = K;
    k.batch = E;
    k.stride_a = static_cast<int>(dy.stride(0));
    k.stride_b = static_cast<int>(w.stride(1));
    k.stride_act_input = static_cast<int>(act_input.stride(0));
    k.stride_dact = static_cast<int>(d_act_input.stride(0));
    k.stride_a_batch = uniform_m * k.stride_a;
    k.stride_b_batch = static_cast<int>(w.stride(0));
    k.stride_act_input_batch = uniform_m * k.stride_act_input;
    k.stride_dact_batch = uniform_m * k.stride_dact;
    k.stride_dscore = static_cast<int>(dscore_partials.stride(0));
    opus_moe_dgrad_swiglu_dscore_launch_gfx950(
        k, aiter::getCurrentHIPStream());
}

void opus_moe_dgrad_swiglu_dscore_ragged_bf16(
    aiter_tensor_t& dy,
    aiter_tensor_t& w,
    aiter_tensor_t& act_input,
    aiter_tensor_t& d_act_input,
    aiter_tensor_t& dscore_partials,
    aiter_tensor_t& expert_offsets,
    aiter_tensor_t& tile_offsets,
    int num_tiles,
    int max_m)
{
    const int E = static_cast<int>(w.size(0));
    const int N = static_cast<int>(w.size(1));
    const int K = static_cast<int>(w.size(2));
    const int M = static_cast<int>(dy.size(0));
    AITER_CHECK(max_m > 0, "max_m must be positive");
    AITER_CHECK(static_cast<int>(expert_offsets.size(0)) == E + 1,
                "expert_offsets must have E + 1 elements");
    AITER_CHECK(static_cast<int>(tile_offsets.size(0)) == E + 1,
                "tile_offsets must have E + 1 elements");
    AITER_CHECK(num_tiles > 0, "num_tiles must be positive");
    AITER_CHECK(static_cast<int>(dy.size(1)) == K, "dy K must match weight K");
    AITER_CHECK(static_cast<int>(act_input.size(0)) == M &&
                    static_cast<int>(act_input.size(1)) == 2 * N,
                "act_input must have shape [M, 2 * N]");
    AITER_CHECK(static_cast<int>(d_act_input.size(0)) == M &&
                    static_cast<int>(d_act_input.size(1)) == 2 * N,
                "d_act_input must have shape [M, 2 * N]");
    AITER_CHECK(static_cast<int>(dscore_partials.size(0)) == M &&
                    static_cast<int>(dscore_partials.size(1)) == N / 256,
                "dscore_partials must have shape [M, N / 256]");
    AITER_CHECK(K >= 128 && K % 64 == 0 && N % 256 == 0,
                "ragged fused dscore requires K%64==0 and N%256==0");

    opus_moe_dgrad_swiglu_kargs k{};
    k.ptr_a = dy.data_ptr();
    k.ptr_b = w.data_ptr();
    k.ptr_act_input = act_input.data_ptr();
    k.ptr_dact = d_act_input.data_ptr();
    k.ptr_dscore_partials = dscore_partials.data_ptr();
    k.expert_offsets = reinterpret_cast<const int32_t*>(expert_offsets.data_ptr());
    k.tile_offsets = reinterpret_cast<const int32_t*>(tile_offsets.data_ptr());
    k.m = max_m;
    k.n = N;
    k.k = K;
    k.batch = E;
    k.stride_a = static_cast<int>(dy.stride(0));
    k.stride_b = static_cast<int>(w.stride(1));
    k.stride_act_input = static_cast<int>(act_input.stride(0));
    k.stride_dact = static_cast<int>(d_act_input.stride(0));
    k.stride_b_batch = static_cast<int>(w.stride(0));
    k.stride_dscore = static_cast<int>(dscore_partials.stride(0));
    k.ragged = 1;
    k.compact_tiles = 1;
    k.num_tiles = num_tiles;
    opus_moe_dgrad_swiglu_dscore_ragged_launch_gfx950(
        k, aiter::getCurrentHIPStream());
}

void opus_moe_dgrad_mono_ragged_bf16(aiter_tensor_t& dy,
                                     aiter_tensor_t& w,
                                     aiter_tensor_t& out,
                                     aiter_tensor_t& expert_offsets,
                                     aiter_tensor_t& tile_offsets,
                                     int num_tiles,
                                     int max_m)
{
    const int E = static_cast<int>(w.size(0));
    const int N = static_cast<int>(w.size(1));
    const int K = static_cast<int>(w.size(2));
    const int M = static_cast<int>(dy.size(0));
    AITER_CHECK(max_m > 0, "max_m must be positive");
    AITER_CHECK(static_cast<int>(expert_offsets.size(0)) == E + 1,
                "expert_offsets must have E + 1 elements");
    AITER_CHECK(static_cast<int>(tile_offsets.size(0)) == E + 1,
                "tile_offsets must have E + 1 elements");
    AITER_CHECK(num_tiles > 0, "num_tiles must be positive");
    AITER_CHECK(static_cast<int>(dy.size(1)) == K, "dy K must match weight K");
    AITER_CHECK(static_cast<int>(out.size(0)) == M &&
                    static_cast<int>(out.size(1)) == N,
                "out must have shape [M, N]");
    AITER_CHECK(K >= 128 && K % 64 == 0 && N % 256 == 0,
                "ragged mono dgrad requires K%64==0 and N%256==0");

    opus_moe_dgrad_swiglu_kargs k{};
    k.ptr_a = dy.data_ptr();
    k.ptr_b = w.data_ptr();
    k.ptr_act_input = out.data_ptr();
    k.ptr_dact = out.data_ptr();
    k.expert_offsets = reinterpret_cast<const int32_t*>(expert_offsets.data_ptr());
    k.tile_offsets = reinterpret_cast<const int32_t*>(tile_offsets.data_ptr());
    k.m = max_m;
    k.n = N;
    k.k = K;
    k.batch = E;
    k.stride_a = static_cast<int>(dy.stride(0));
    k.stride_b = static_cast<int>(w.stride(1));
    k.stride_act_input = static_cast<int>(out.stride(0));
    k.stride_dact = static_cast<int>(out.stride(0));
    k.stride_b_batch = static_cast<int>(w.stride(0));
    k.ragged = 1;
    k.compact_tiles = 1;
    k.num_tiles = num_tiles;
    opus_moe_dgrad_plain_ragged_launch_gfx950(
        k, aiter::getCurrentHIPStream());
}

// Fused opus-MFMA grouped wgrad (BF16->FP32). Feature-major transposed +
// route-padded inputs (build_padded_transposed). dW[e]=dyT_e @ aT_e^T.
void opus_moe_wgrad_mfma_bf16(aiter_tensor_t& dyT,       // [P, Mp] bf16
                              aiter_tensor_t& aT,        // [Q, Mp] bf16
                              aiter_tensor_t& pad_offs,  // [E+1] i32
                              aiter_tensor_t& dW)        // [E, P, Q] fp32
{
    opus_moe_wgrad_mfma_kargs k;
    k.ptr_dyT  = dyT.data_ptr();
    k.ptr_aT   = aT.data_ptr();
    k.ptr_dW   = dW.data_ptr();
    k.pad_offs = reinterpret_cast<const int32_t*>(pad_offs.data_ptr());
    k.P        = static_cast<int>(dyT.size(0));
    k.Q        = static_cast<int>(aT.size(0));
    k.Mp       = static_cast<int>(dyT.size(1));
    const int num_experts = static_cast<int>(dW.size(0));
    opus_moe_wgrad_mfma_launch_gfx950(k, num_experts, aiter::getCurrentHIPStream());
}

// Fused compact->feature-major pad+transpose. Writes dst[F,Mp] fully in one
// pass with COALESCED writes along Mp: thread (f,col) writes dst[f,col] =
// src[col_to_m[col], f], or 0 when col_to_m[col] < 0 (padding). Reads of src
// are strided (L2-friendly); no pre-zero needed. Replaces torch's
// zeros+scatter+transpose (3-4 passes) for wgrad operands.
__global__ void opus_moe_transpose_pad_bf16_kernel(const hip_bfloat16* __restrict__ src, // [M,F]
                                                   const int32_t* __restrict__ col_to_m,  // [Mp] compact row or -1
                                                   hip_bfloat16* __restrict__ dst,        // [F,Mp]
                                                   int F,
                                                   int Mp)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(col >= Mp)
        return;
    const int f = blockIdx.y;
    const int m = col_to_m[col];
    hip_bfloat16 v;
    if(m >= 0)
        v = src[static_cast<int64_t>(m) * F + f];
    else
        v = hip_bfloat16{};
    dst[static_cast<int64_t>(f) * Mp + col] = v;
}

// Full-TN grouped wgrad: dW[e]=dy_e^T@a_e from natural compact dy/a (no
// transpose/padding). dy [M,P] bf16, a [M,Q] bf16, offs [E+1] i32, dW [E,P,Q] fp32.
void opus_moe_wgrad_tn_bf16(aiter_tensor_t& dy,
                            aiter_tensor_t& a,
                            aiter_tensor_t& offs,
                            aiter_tensor_t& dW,
                            int uniform_m)
{
    const int E = static_cast<int>(dW.size(0));
    const int P = static_cast<int>(dW.size(1));
    const int Q = static_cast<int>(dW.size(2));
    opus_moe_wgrad_tn_launch_gfx950(
        reinterpret_cast<const __bf16*>(dy.data_ptr()),
        reinterpret_cast<const __bf16*>(a.data_ptr()),
        reinterpret_cast<const int32_t*>(offs.data_ptr()),
        reinterpret_cast<__bf16*>(dW.data_ptr()),
        E, P, Q, uniform_m, aiter::getCurrentHIPStream());
}

void opus_moe_transpose_pad_bf16(aiter_tensor_t& src,      // [M,F] bf16 compact
                                 aiter_tensor_t& col_to_m, // [Mp] i32 compact row of each padded col (-1=pad)
                                 aiter_tensor_t& dst)       // [F,Mp] bf16 (fully written)
{
    const int F  = static_cast<int>(src.size(1));
    const int Mp = static_cast<int>(dst.size(1));
    if(Mp == 0 || F == 0)
        return;
    constexpr int BLK = 256;
    dim3 grid((Mp + BLK - 1) / BLK, F);
    dim3 block(BLK);
    opus_moe_transpose_pad_bf16_kernel<<<grid, block, 0, aiter::getCurrentHIPStream()>>>(
        reinterpret_cast<const hip_bfloat16*>(src.data_ptr()),
        reinterpret_cast<const int32_t*>(col_to_m.data_ptr()),
        reinterpret_cast<hip_bfloat16*>(dst.data_ptr()),
        F, Mp);
}

__device__ __forceinline__ float opus_sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

// g1u1 activation Jacobian: (dh, gate, up) -> (dgate, dup). Single source of
// truth shared by the scalar and vectorized kernels; mirrors the triton
// _act_bwd_kernel / golden _apply_activation.
__device__ __forceinline__ void
opus_act_jacobian(int ACT, float dhv, float gate, float up, float LIMIT, float ALPHA, float& dg, float& du)
{
    if(ACT == 3) // SiTUv2 beta=lb=1
    {
        const float tg = tanhf(gate), su = opus_sigmoidf(gate), tu = tanhf(up);
        dg = dhv * tu * ((1.0f - tg * tg) * su + tg * su * (1.0f - su));
        du = dhv * (tg * su) * (1.0f - tu * tu);
    }
    else if(ACT == 2) // Swiglu +1 bias
    {
        const float g = fminf(gate, LIMIT);
        const float u = fmaxf(fminf(up, LIMIT), -LIMIT);
        const float s = opus_sigmoidf(ALPHA * g), f = g * s;
        dg = dhv * (u + 1.0f) * (s + g * s * (1.0f - s) * ALPHA) * (gate <= LIMIT ? 1.0f : 0.0f);
        du = dhv * f * ((up >= -LIMIT && up <= LIMIT) ? 1.0f : 0.0f);
    }
    else if(ACT == 1) // Gelu exact
    {
        const float phi_c = 0.5f * (1.0f + erff(gate * 0.7071067811865476f));
        const float pdf   = expf(-0.5f * gate * gate) * 0.3989422804014327f;
        dg = dhv * up * (phi_c + gate * pdf);
        du = dhv * (gate * phi_c);
    }
    else // Silu
    {
        const float sig = opus_sigmoidf(gate), silu = gate * sig;
        dg = dhv * up * (sig + gate * sig * (1.0f - sig));
        du = dhv * silu;
    }
}

// Scalar fallback: one thread per (m, i). Used when I % 8 != 0.
__global__ void opus_moe_act_bwd_bf16_kernel(const hip_bfloat16* __restrict__ dh,
                                             const hip_bfloat16* __restrict__ ai,
                                             hip_bfloat16* __restrict__ dai,
                                             int M,
                                             int I,
                                             int ACT,
                                             float LIMIT,
                                             float ALPHA)
{
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(m >= M || i >= I)
        return;
    const int64_t ai_row = static_cast<int64_t>(m) * (2 * I);
    float dg, du;
    opus_act_jacobian(ACT, static_cast<float>(dh[static_cast<int64_t>(m) * I + i]),
                      static_cast<float>(ai[ai_row + i]), static_cast<float>(ai[ai_row + I + i]),
                      LIMIT, ALPHA, dg, du);
    dai[ai_row + i]     = hip_bfloat16(dg);
    dai[ai_row + I + i] = hip_bfloat16(du);
}

// Vectorized: one thread per 8 contiguous i of one row m. bf16x8 loads/stores
// (16B coalesced) raise the memory-bound act-bwd from ~26% to near-peak HBM BW.
// Requires I % 8 == 0 (rows are contiguous, i mult 8 => 16B-aligned).
__global__ void opus_moe_act_bwd_bf16_vec8_kernel(const __bf16* __restrict__ dh,
                                                  const __bf16* __restrict__ ai,
                                                  __bf16* __restrict__ dai,
                                                  int M,
                                                  int I,
                                                  int ACT,
                                                  float LIMIT,
                                                  float ALPHA)
{
    const int ipb          = I / 8;
    const int64_t nblk     = static_cast<int64_t>(M) * ipb;
    const int64_t tid      = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(tid >= nblk)
        return;
    const int m          = static_cast<int>(tid / ipb);
    const int i          = static_cast<int>(tid % ipb) * 8;
    const int64_t ai_row = static_cast<int64_t>(m) * (2 * I);
    opus_bf16x8 dhv = *reinterpret_cast<const opus_bf16x8*>(dh + static_cast<int64_t>(m) * I + i);
    opus_bf16x8 g8  = *reinterpret_cast<const opus_bf16x8*>(ai + ai_row + i);
    opus_bf16x8 u8  = *reinterpret_cast<const opus_bf16x8*>(ai + ai_row + I + i);
    opus_bf16x8 dg8, du8;
#pragma unroll
    for(int e = 0; e < 8; ++e)
    {
        float dg, du;
        opus_act_jacobian(ACT, static_cast<float>(dhv[e]), static_cast<float>(g8[e]),
                          static_cast<float>(u8[e]), LIMIT, ALPHA, dg, du);
        dg8[e] = static_cast<__bf16>(dg);
        du8[e] = static_cast<__bf16>(du);
    }
    *reinterpret_cast<opus_bf16x8*>(dai + ai_row + i)     = dg8;
    *reinterpret_cast<opus_bf16x8*>(dai + ai_row + I + i) = du8;
}

void opus_moe_act_bwd_bf16(aiter_tensor_t& dh,
                           aiter_tensor_t& act_input,
                           aiter_tensor_t& d_act_input,
                           int act,
                           float swiglu_limit)
{
    const int M = static_cast<int>(dh.size(0));
    const int I = static_cast<int>(dh.size(1));
    const float limit = swiglu_limit > 0.0f ? swiglu_limit : 7.0f;
    const hipStream_t stream = aiter::getCurrentHIPStream();

    if(I % 8 == 0)
    {
        const int64_t nblk = static_cast<int64_t>(M) * (I / 8);
        constexpr int BLK  = 256;
        const int grid     = static_cast<int>((nblk + BLK - 1) / BLK);
        opus_moe_act_bwd_bf16_vec8_kernel<<<grid, BLK, 0, stream>>>(
            reinterpret_cast<const __bf16*>(dh.data_ptr()),
            reinterpret_cast<const __bf16*>(act_input.data_ptr()),
            reinterpret_cast<__bf16*>(d_act_input.data_ptr()),
            M, I, act, limit, 1.702f);
        return;
    }

    dim3 block(16, 16, 1);
    dim3 grid((I + 15) / 16, (M + 15) / 16, 1);
    opus_moe_act_bwd_bf16_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const hip_bfloat16*>(dh.data_ptr()),
        reinterpret_cast<const hip_bfloat16*>(act_input.data_ptr()),
        reinterpret_cast<hip_bfloat16*>(d_act_input.data_ptr()),
        M, I, act, limit, 1.702f);
}

// Combine backward (M5 R6): per route m (block-per-route), dy[m,:]=p[m]*dout[t,:]
// (t=gather[m], token->route broadcast) and dp[m]=<dout[t,:], y[m,:]> (block
// reduce over H). Replaces the torch gather+scale+dot in the opus backward.
__global__ void opus_moe_combine_bwd_bf16_kernel(const hip_bfloat16* __restrict__ dout,  // [T,H]
                                                 const int32_t* __restrict__ gather,     // [M]
                                                 const float* __restrict__ p,            // [M]
                                                 const hip_bfloat16* __restrict__ y,     // [M,H]
                                                 hip_bfloat16* __restrict__ dy,          // [M,H]
                                                 float* __restrict__ dp,                 // [M]
                                                 int H)
{
    const int m = blockIdx.x;
    const int t = gather[m];
    const hip_bfloat16* drow  = dout + static_cast<int64_t>(t) * H;
    const hip_bfloat16* yrow  = y + static_cast<int64_t>(m) * H;
    hip_bfloat16* dyrow       = dy + static_cast<int64_t>(m) * H;
    const float pv            = p[m];
    float acc                 = 0.f;
    for(int hh = threadIdx.x; hh < H; hh += blockDim.x)
    {
        const float d = static_cast<float>(drow[hh]);
        dyrow[hh]     = static_cast<hip_bfloat16>(d * pv);
        acc += d * static_cast<float>(yrow[hh]);
    }
    __shared__ float sm[256];
    sm[threadIdx.x] = acc;
    __syncthreads();
    for(int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if(threadIdx.x < s)
            sm[threadIdx.x] += sm[threadIdx.x + s];
        __syncthreads();
    }
    if(threadIdx.x == 0)
        dp[m] = sm[0];
}

// Vectorized fast path: one 64-thread wave/block per route.
// Each lane streams bf16x8 chunks and the route-score dot product stays within
// the wave, eliminating the shared-memory reduction and its block barriers.
__global__ void opus_moe_combine_bwd_bf16_wave_kernel(
    const __bf16* __restrict__ dout,  // [T,H]
    const int32_t* __restrict__ gather,
    const float* __restrict__ p,
    const __bf16* __restrict__ y,     // [M,H]
    __bf16* __restrict__ dy,          // [M,H]
    float* __restrict__ dp,
    int M,
    int H)
{
    constexpr int WAVE = 64;
    constexpr int WAVES_PER_BLOCK = 1;
    const int lane = threadIdx.x % WAVE;
    const int wave = threadIdx.x / WAVE;
    const int m = blockIdx.x * WAVES_PER_BLOCK + wave;
    if(m >= M)
        return;

    const int t = gather[m];
    const __bf16* drow = dout + static_cast<int64_t>(t) * H;
    const __bf16* yrow = y + static_cast<int64_t>(m) * H;
    __bf16* dyrow = dy + static_cast<int64_t>(m) * H;
    const float pv = p[m];
    float acc = 0.0f;
    for(int h = lane * 8; h < H; h += WAVE * 8)
    {
        const opus_bf16x8 dv = *reinterpret_cast<const opus_bf16x8*>(drow + h);
        const opus_bf16x8 yv = *reinterpret_cast<const opus_bf16x8*>(yrow + h);
        opus_bf16x8 dyv;
#pragma unroll
        for(int i = 0; i < 8; ++i)
        {
            const float d = static_cast<float>(dv[i]);
            dyv[i] = static_cast<__bf16>(d * pv);
            acc = fmaf(d, static_cast<float>(yv[i]), acc);
        }
        *reinterpret_cast<opus_bf16x8*>(dyrow + h) = dyv;
    }
#pragma unroll
    for(int offset = WAVE / 2; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset, WAVE);
    if(lane == 0)
        dp[m] = acc;
}

void opus_moe_combine_bwd_bf16(aiter_tensor_t& dout,   // [T,H] bf16
                               aiter_tensor_t& gather, // [M] i32
                               aiter_tensor_t& p,      // [M] fp32
                               aiter_tensor_t& y,      // [M,H] bf16
                               aiter_tensor_t& dy,     // [M,H] bf16 out
                               aiter_tensor_t& dp)     // [M] fp32 out
{
    const int M = static_cast<int>(dy.size(0));
    const int H = static_cast<int>(dy.size(1));
    if(M == 0)
        return;
    if(H % 8 == 0)
    {
        constexpr int WAVES_PER_BLOCK = 1;
        constexpr int BLOCK = WAVES_PER_BLOCK * 64;
        const int grid = (M + WAVES_PER_BLOCK - 1) / WAVES_PER_BLOCK;
        opus_moe_combine_bwd_bf16_wave_kernel<<<
            dim3(grid), dim3(BLOCK), 0, aiter::getCurrentHIPStream()>>>(
            reinterpret_cast<const __bf16*>(dout.data_ptr()),
            reinterpret_cast<const int32_t*>(gather.data_ptr()),
            reinterpret_cast<const float*>(p.data_ptr()),
            reinterpret_cast<const __bf16*>(y.data_ptr()),
            reinterpret_cast<__bf16*>(dy.data_ptr()),
            reinterpret_cast<float*>(dp.data_ptr()),
            M, H);
        return;
    }
    opus_moe_combine_bwd_bf16_kernel<<<dim3(M), dim3(256), 0, aiter::getCurrentHIPStream()>>>(
        reinterpret_cast<const hip_bfloat16*>(dout.data_ptr()),
        reinterpret_cast<const int32_t*>(gather.data_ptr()),
        reinterpret_cast<const float*>(p.data_ptr()),
        reinterpret_cast<const hip_bfloat16*>(y.data_ptr()),
        reinterpret_cast<hip_bfloat16*>(dy.data_ptr()),
        reinterpret_cast<float*>(dp.data_ptr()),
        H);
}

// Sonic parity specialization: one 512-thread block owns a token and all its
// eight routes. Each wave retains the low-register route-major dot/scale loop,
// while the CTA stages dout[t,:] once in LDS for all eight waves to reuse.
__global__ __launch_bounds__(512, 1)
void opus_moe_combine_bwd_token8_h2048_bf16_kernel(
    const __bf16* __restrict__ dout,          // [T,2048]
    const int32_t* __restrict__ token_routes, // [T,8]
    const float* __restrict__ p,              // [M]
    const __bf16* __restrict__ y,             // [M,2048]
    __bf16* __restrict__ dy,                  // [M,2048]
    float* __restrict__ dp)                   // [M]
{
    constexpr int H = 2048;
    constexpr int TOPK = 8;
    constexpr int VEC = 8;
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 63;
    const int wave = tid >> 6;

    __shared__ __align__(16) __bf16 dout_shared[H];
    __shared__ int32_t routes[TOPK];
    __shared__ float scores[TOPK];
    if(tid < TOPK)
    {
        const int32_t route = token_routes[static_cast<int64_t>(t) * TOPK + tid];
        routes[tid] = route;
        scores[tid] = p[route];
    }
    if(tid < H / VEC)
    {
        const int h = tid * VEC;
        *reinterpret_cast<opus_bf16x8*>(dout_shared + h) =
            *reinterpret_cast<const opus_bf16x8*>(
                dout + static_cast<int64_t>(t) * H + h);
    }
    __syncthreads();

    const int32_t route = routes[wave];
    const float score = scores[wave];
    float acc = 0.0f;
    for(int h = lane * VEC; h < H; h += 64 * VEC)
    {
        const int64_t offset = static_cast<int64_t>(route) * H + h;
        const opus_bf16x8 dv =
            *reinterpret_cast<const opus_bf16x8*>(dout_shared + h);
        const opus_bf16x8 yv = *reinterpret_cast<const opus_bf16x8*>(y + offset);
        opus_bf16x8 dyv;
#pragma unroll
        for(int i = 0; i < VEC; ++i)
        {
            const float d = static_cast<float>(dv[i]);
            dyv[i] = static_cast<__bf16>(d * score);
            acc = fmaf(d, static_cast<float>(yv[i]), acc);
        }
        *reinterpret_cast<opus_bf16x8*>(dy + offset) = dyv;
    }

#pragma unroll
    for(int delta = 32; delta > 0; delta >>= 1)
        acc += __shfl_down(acc, delta, 64);
    if(lane == 0)
        dp[route] = acc;
}

void opus_moe_combine_bwd_token8_h2048_bf16(
    aiter_tensor_t& dout,
    aiter_tensor_t& token_routes,
    aiter_tensor_t& p,
    aiter_tensor_t& y,
    aiter_tensor_t& dy,
    aiter_tensor_t& dp)
{
    const int T = static_cast<int>(dout.size(0));
    if(T == 0)
        return;
    opus_moe_combine_bwd_token8_h2048_bf16_kernel<<<
        dim3(T), dim3(512), 0, aiter::getCurrentHIPStream()>>>(
        reinterpret_cast<const __bf16*>(dout.data_ptr()),
        reinterpret_cast<const int32_t*>(token_routes.data_ptr()),
        reinterpret_cast<const float*>(p.data_ptr()),
        reinterpret_cast<const __bf16*>(y.data_ptr()),
        reinterpret_cast<__bf16*>(dy.data_ptr()),
        reinterpret_cast<float*>(dp.data_ptr()));
}

// Stage dy only; dscore is reconstructed from the fused stage-2 epilogue.
// Four tokens share one CTA and each wave writes two routes, halving the grid
// relative to a dedicated wave for every route.
__global__ __launch_bounds__(512, 2)
void opus_moe_combine_scale_token8_h2048_bf16_kernel(
    const __bf16* __restrict__ dout,
    const int32_t* __restrict__ token_routes,
    const float* __restrict__ p,
    __bf16* __restrict__ dy,
    int T)
{
    constexpr int H = 2048;
    constexpr int TOPK = 8;
    constexpr int VEC = 8;
    constexpr int THREADS_PER_TOKEN = 256;
    constexpr int TOKENS_PER_BLOCK = 2;
    constexpr int WAVES_PER_TOKEN = THREADS_PER_TOKEN / 64;
    const int token_in_block = threadIdx.x / THREADS_PER_TOKEN;
    const int tid = threadIdx.x % THREADS_PER_TOKEN;
    const int t = blockIdx.x * TOKENS_PER_BLOCK + token_in_block;
    const bool valid_token = t < T;
    const int lane = tid & 63;
    const int wave = tid >> 6;
    __shared__ __align__(16) __bf16 dout_shared[TOKENS_PER_BLOCK][H];
    __shared__ int32_t routes[TOKENS_PER_BLOCK][TOPK];
    __shared__ float scores[TOKENS_PER_BLOCK][TOPK];
    if(valid_token && tid < TOPK)
    {
        const int32_t route = token_routes[static_cast<int64_t>(t) * TOPK + tid];
        routes[token_in_block][tid] = route;
        scores[token_in_block][tid] = p[route];
    }
    if(valid_token)
    {
        const int h = tid * VEC;
        *reinterpret_cast<opus_bf16x8*>(dout_shared[token_in_block] + h) =
            *reinterpret_cast<const opus_bf16x8*>(
                dout + static_cast<int64_t>(t) * H + h);
    }
    __syncthreads();
    if(!valid_token)
        return;
#pragma unroll
    for(int rank = wave; rank < TOPK; rank += WAVES_PER_TOKEN)
    {
        const int32_t route = routes[token_in_block][rank];
        const float score = scores[token_in_block][rank];
        for(int h = lane * VEC; h < H; h += 64 * VEC)
        {
            const int64_t offset = static_cast<int64_t>(route) * H + h;
            const opus_bf16x8 dv = *reinterpret_cast<const opus_bf16x8*>(
                dout_shared[token_in_block] + h);
            opus_bf16x8 dyv;
#pragma unroll
            for(int i = 0; i < VEC; ++i)
                dyv[i] = static_cast<__bf16>(static_cast<float>(dv[i]) * score);
            *reinterpret_cast<opus_bf16x8*>(dy + offset) = dyv;
        }
    }
}

void opus_moe_combine_scale_token8_h2048_bf16(
    aiter_tensor_t& dout,
    aiter_tensor_t& token_routes,
    aiter_tensor_t& p,
    aiter_tensor_t& dy)
{
    const int T = static_cast<int>(dout.size(0));
    if(T == 0)
        return;
    constexpr int TOKENS_PER_BLOCK = 2;
    opus_moe_combine_scale_token8_h2048_bf16_kernel<<<
        dim3((T + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK),
        dim3(TOKENS_PER_BLOCK * 256), 0, aiter::getCurrentHIPStream()>>>(
        reinterpret_cast<const __bf16*>(dout.data_ptr()),
        reinterpret_cast<const int32_t*>(token_routes.data_ptr()),
        reinterpret_cast<const float*>(p.data_ptr()),
        reinterpret_cast<__bf16*>(dy.data_ptr()),
        T);
}

__global__ __launch_bounds__(64, 1)
void opus_moe_dscore_router_bwd_token8_e64_bf16_kernel(
    const int32_t* __restrict__ token_routes,
    const float* __restrict__ p,
    const float* __restrict__ partials,
    const int64_t* __restrict__ order,
    const int64_t* __restrict__ topk_ids,
    __bf16* __restrict__ dlogits)
{
    constexpr int TOPK = 8;
    constexpr int E = 64;
    constexpr int PARTIALS = 4;
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float scaled_dp_rank[TOPK];
    __shared__ float score_rank[TOPK];
    if(tid < E)
        dlogits[static_cast<int64_t>(t) * E + tid] = static_cast<__bf16>(0.0f);
    if(tid < TOPK)
    {
        const int32_t route = token_routes[static_cast<int64_t>(t) * TOPK + tid];
        float scaled_dp = 0.0f;
#pragma unroll
        for(int i = 0; i < PARTIALS; ++i)
            scaled_dp += partials[static_cast<int64_t>(route) * PARTIALS + i];
        const float score = p[route];
        const int rank = static_cast<int>(order[route] & (TOPK - 1));
        scaled_dp_rank[rank] = scaled_dp;
        score_rank[rank] = score;
    }
    __syncthreads();
    if(tid == 0)
    {
        float dot = 0.0f;
#pragma unroll
        for(int k = 0; k < TOPK; ++k)
            dot += scaled_dp_rank[k];
#pragma unroll
        for(int k = 0; k < TOPK; ++k)
        {
            const int64_t expert = topk_ids[static_cast<int64_t>(t) * TOPK + k];
            dlogits[static_cast<int64_t>(t) * E + expert] =
                static_cast<__bf16>(scaled_dp_rank[k] - score_rank[k] * dot);
        }
    }
}

void opus_moe_dscore_router_bwd_token8_e64_bf16(
    aiter_tensor_t& token_routes,
    aiter_tensor_t& p,
    aiter_tensor_t& partials,
    aiter_tensor_t& order,
    aiter_tensor_t& topk_ids,
    aiter_tensor_t& dlogits)
{
    const int T = static_cast<int>(token_routes.size(0));
    if(T == 0)
        return;
    opus_moe_dscore_router_bwd_token8_e64_bf16_kernel<<<
        dim3(T), dim3(64), 0, aiter::getCurrentHIPStream()>>>(
        reinterpret_cast<const int32_t*>(token_routes.data_ptr()),
        reinterpret_cast<const float*>(p.data_ptr()),
        reinterpret_cast<const float*>(partials.data_ptr()),
        reinterpret_cast<const int64_t*>(order.data_ptr()),
        reinterpret_cast<const int64_t*>(topk_ids.data_ptr()),
        reinterpret_cast<__bf16*>(dlogits.data_ptr()));
}

// Exact parity path: consume each route's dot product inside the token CTA and
// write the softmax Jacobian directly. This removes the compact->token scatter,
// the standalone router kernel, and the dlogits memset from the timed backward.
__global__ __launch_bounds__(512, 1)
void opus_moe_combine_router_bwd_token8_h2048_e64_bf16_kernel(
    const __bf16* __restrict__ dout,          // [T,2048]
    const int32_t* __restrict__ token_routes, // [T,8]
    const float* __restrict__ p,              // [M], compact route order
    const __bf16* __restrict__ y,             // [M,2048]
    const int64_t* __restrict__ order,        // [M], compact -> token-major slot
    const int64_t* __restrict__ topk_ids,     // [T,8]
    __bf16* __restrict__ dy,                  // [M,2048]
    __bf16* __restrict__ dlogits)             // [T,64]
{
    constexpr int H = 2048;
    constexpr int TOPK = 8;
    constexpr int E = 64;
    constexpr int VEC = 8;
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 63;
    const int wave = tid >> 6;

    __shared__ __align__(16) __bf16 dout_shared[H];
    __shared__ int32_t routes[TOPK];
    __shared__ float scores[TOPK];
    __shared__ float dp_rank[TOPK];
    __shared__ float score_rank[TOPK];

    if(tid < E)
        dlogits[static_cast<int64_t>(t) * E + tid] = static_cast<__bf16>(0.0f);
    if(tid < TOPK)
    {
        const int32_t route = token_routes[static_cast<int64_t>(t) * TOPK + tid];
        routes[tid] = route;
        scores[tid] = p[route];
    }
    if(tid < H / VEC)
    {
        const int h = tid * VEC;
        *reinterpret_cast<opus_bf16x8*>(dout_shared + h) =
            *reinterpret_cast<const opus_bf16x8*>(
                dout + static_cast<int64_t>(t) * H + h);
    }
    __syncthreads();

    const int32_t route = routes[wave];
    const float score = scores[wave];
    float acc = 0.0f;
    for(int h = lane * VEC; h < H; h += 64 * VEC)
    {
        const int64_t offset = static_cast<int64_t>(route) * H + h;
        const opus_bf16x8 dv =
            *reinterpret_cast<const opus_bf16x8*>(dout_shared + h);
        const opus_bf16x8 yv = *reinterpret_cast<const opus_bf16x8*>(y + offset);
        opus_bf16x8 dyv;
#pragma unroll
        for(int i = 0; i < VEC; ++i)
        {
            const float d = static_cast<float>(dv[i]);
            dyv[i] = static_cast<__bf16>(d * score);
            acc = fmaf(d, static_cast<float>(yv[i]), acc);
        }
        *reinterpret_cast<opus_bf16x8*>(dy + offset) = dyv;
    }

#pragma unroll
    for(int delta = 32; delta > 0; delta >>= 1)
        acc += __shfl_down(acc, delta, 64);
    if(lane == 0)
    {
        const int rank = static_cast<int>(order[route] & (TOPK - 1));
        dp_rank[rank] = acc;
        score_rank[rank] = score;
    }
    __syncthreads();

    if(tid == 0)
    {
        float dot = 0.0f;
#pragma unroll
        for(int k = 0; k < TOPK; ++k)
            dot = fmaf(dp_rank[k], score_rank[k], dot);
#pragma unroll
        for(int k = 0; k < TOPK; ++k)
        {
            const int64_t expert = topk_ids[static_cast<int64_t>(t) * TOPK + k];
            dlogits[static_cast<int64_t>(t) * E + expert] =
                static_cast<__bf16>(score_rank[k] * (dp_rank[k] - dot));
        }
    }
}

void opus_moe_combine_router_bwd_token8_h2048_e64_bf16(
    aiter_tensor_t& dout,
    aiter_tensor_t& token_routes,
    aiter_tensor_t& p,
    aiter_tensor_t& y,
    aiter_tensor_t& order,
    aiter_tensor_t& topk_ids,
    aiter_tensor_t& dy,
    aiter_tensor_t& dlogits)
{
    const int T = static_cast<int>(dout.size(0));
    if(T == 0)
        return;
    opus_moe_combine_router_bwd_token8_h2048_e64_bf16_kernel<<<
        dim3(T), dim3(512), 0, aiter::getCurrentHIPStream()>>>(
        reinterpret_cast<const __bf16*>(dout.data_ptr()),
        reinterpret_cast<const int32_t*>(token_routes.data_ptr()),
        reinterpret_cast<const float*>(p.data_ptr()),
        reinterpret_cast<const __bf16*>(y.data_ptr()),
        reinterpret_cast<const int64_t*>(order.data_ptr()),
        reinterpret_cast<const int64_t*>(topk_ids.data_ptr()),
        reinterpret_cast<__bf16*>(dy.data_ptr()),
        reinterpret_cast<__bf16*>(dlogits.data_ptr()));
}

// dx scatter-add (M5 R6): sum each route's dA back to its token via atomicAdd
// (topk routes -> one token). dst must be pre-zeroed. Replaces torch index_add_.
__global__ void opus_moe_scatter_add_bf16_kernel(const hip_bfloat16* __restrict__ src, // [M,H]
                                                 const int32_t* __restrict__ gather,   // [M]
                                                 float* __restrict__ dst,              // [T,H] fp32
                                                 int H)
{
    const int m = blockIdx.x;
    const int t = gather[m];
    const hip_bfloat16* srow = src + static_cast<int64_t>(m) * H;
    float* drow              = dst + static_cast<int64_t>(t) * H;
    for(int hh = threadIdx.x; hh < H; hh += blockDim.x)
        atomicAdd(&drow[hh], static_cast<float>(srow[hh]));
}

void opus_moe_scatter_add_bf16(aiter_tensor_t& src,    // [M,H] bf16
                               aiter_tensor_t& gather, // [M] i32
                               aiter_tensor_t& dst)    // [T,H] fp32 pre-zeroed
{
    const int M = static_cast<int>(src.size(0));
    const int H = static_cast<int>(src.size(1));
    if(M == 0)
        return;
    opus_moe_scatter_add_bf16_kernel<<<dim3(M), dim3(256), 0, aiter::getCurrentHIPStream()>>>(
        reinterpret_cast<const hip_bfloat16*>(src.data_ptr()),
        reinterpret_cast<const int32_t*>(gather.data_ptr()),
        reinterpret_cast<float*>(dst.data_ptr()),
        H);
}

// Deterministic dx gather-sum (replaces the atomic scatter for fixed top-k):
// Two tokens per block; an independent 256-thread half-block sums each token's
// topk route rows of src -> dst[t]. No atomics (each dst[t] is written by one
// half-block), no pre-zero. bf16x8 vectorized; fp32 accumulate.
// token_routes[t,k] is the compact route index of token t's k-th selection
// (built once from routing). Requires H % 8 == 0.
__global__ void opus_moe_gather_sum_bf16_kernel(const __bf16* __restrict__ src,           // [M,H]
                                                const int32_t* __restrict__ token_routes, // [T,topk]
                                                __bf16* __restrict__ dst,                 // [T,H]
                                                int T,
                                                int H,
                                                int topk)
{
    constexpr int THREADS_PER_TOKEN = 256;
    constexpr int TOKENS_PER_BLOCK = 2;
    const int token_in_block = threadIdx.x / THREADS_PER_TOKEN;
    const int token_thread = threadIdx.x % THREADS_PER_TOKEN;
    const int t          = blockIdx.x * TOKENS_PER_BLOCK + token_in_block;
    if(t >= T)
        return;
    const int32_t* tr    = token_routes + static_cast<int64_t>(t) * topk;
    const int64_t dstrow = static_cast<int64_t>(t) * H;
    for(int h = token_thread * 8; h < H; h += THREADS_PER_TOKEN * 8)
    {
        float acc[8];
#pragma unroll
        for(int e = 0; e < 8; ++e)
            acc[e] = 0.0f;
        for(int k = 0; k < topk; ++k)
        {
            opus_bf16x8 v = *reinterpret_cast<const opus_bf16x8*>(src + static_cast<int64_t>(tr[k]) * H + h);
#pragma unroll
            for(int e = 0; e < 8; ++e)
                acc[e] += static_cast<float>(v[e]);
        }
#pragma unroll
        for(int e = 0; e < 8; ++e)
            dst[dstrow + h + e] = static_cast<__bf16>(acc[e]);
    }
}

void opus_moe_gather_sum_bf16(aiter_tensor_t& src,          // [M,H] bf16
                              aiter_tensor_t& token_routes, // [T,topk] i32
                              aiter_tensor_t& dst)          // [T,H] bf16
{
    const int T    = static_cast<int>(dst.size(0));
    const int H    = static_cast<int>(dst.size(1));
    const int topk = static_cast<int>(token_routes.size(1));
    if(T == 0)
        return;
    constexpr int TOKENS_PER_BLOCK = 2;
    opus_moe_gather_sum_bf16_kernel<<<
        dim3((T + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK),
        dim3(TOKENS_PER_BLOCK * 256), 0, aiter::getCurrentHIPStream()>>>(
        reinterpret_cast<const __bf16*>(src.data_ptr()),
        reinterpret_cast<const int32_t*>(token_routes.data_ptr()),
        reinterpret_cast<__bf16*>(dst.data_ptr()),
        T, H, topk);
}

// Exact Sonic-parity tail: fuse the token's dscore softmax-JVP into the dx
// gather CTA.  Only the first wave of each 256-thread token partition performs
// router work; no workgroup barrier is added to the bandwidth-critical gather.
__global__ __launch_bounds__(512, 1)
void opus_moe_gather_sum_dscore_router_token8_h2048_e64_bf16_kernel(
    const __bf16* __restrict__ src,
    const int32_t* __restrict__ token_routes,
    const float* __restrict__ route_scores,
    const float* __restrict__ partials,
    const int64_t* __restrict__ order,
    const int64_t* __restrict__ topk_ids,
    __bf16* __restrict__ dst,
    __bf16* __restrict__ dlogits,
    int T)
{
    constexpr int H = 2048;
    constexpr int TOPK = 8;
    constexpr int E = 64;
    constexpr int PARTIALS = 4;
    constexpr int THREADS_PER_TOKEN = 256;
    constexpr int TOKENS_PER_BLOCK = 2;
    const int token_in_block = threadIdx.x / THREADS_PER_TOKEN;
    const int token_thread = threadIdx.x % THREADS_PER_TOKEN;
    const int t = blockIdx.x * TOKENS_PER_BLOCK + token_in_block;
    if(t >= T)
        return;
    const int32_t* tr =
        token_routes + static_cast<int64_t>(t) * TOPK;

    if(token_thread < E)
        dlogits[static_cast<int64_t>(t) * E + token_thread] =
            static_cast<__bf16>(0.0f);
    if(token_thread < 64)
    {
        const int lane = token_thread;
        float scaled_dp = 0.0f;
        float score = 0.0f;
        int rank = 0;
        if(lane < TOPK)
        {
            const int32_t route = tr[lane];
#pragma unroll
            for(int i = 0; i < PARTIALS; ++i)
                scaled_dp +=
                    partials[static_cast<int64_t>(route) * PARTIALS + i];
            score = route_scores[route];
            rank = static_cast<int>(order[route] & (TOPK - 1));
        }
        // Preserve the standalone kernel's deterministic rank-ordered sum.
        float dot = 0.0f;
#pragma unroll
        for(int k = 0; k < TOPK; ++k)
            dot += __shfl(scaled_dp, k, TOPK);
        if(lane < TOPK)
        {
            const int64_t expert =
                topk_ids[static_cast<int64_t>(t) * TOPK + rank];
            dlogits[static_cast<int64_t>(t) * E + expert] =
                static_cast<__bf16>(scaled_dp - score * dot);
        }
    }

    const int64_t dstrow = static_cast<int64_t>(t) * H;
    for(int h = token_thread * 8; h < H;
        h += THREADS_PER_TOKEN * 8)
    {
        float acc[8] = {};
#pragma unroll
        for(int k = 0; k < TOPK; ++k)
        {
            const opus_bf16x8 v =
                *reinterpret_cast<const opus_bf16x8*>(
                    src + static_cast<int64_t>(tr[k]) * H + h);
#pragma unroll
            for(int i = 0; i < 8; ++i)
                acc[i] += static_cast<float>(v[i]);
        }
#pragma unroll
        for(int i = 0; i < 8; ++i)
            dst[dstrow + h + i] = static_cast<__bf16>(acc[i]);
    }
}

void opus_moe_gather_sum_dscore_router_token8_h2048_e64_bf16(
    aiter_tensor_t& src,
    aiter_tensor_t& token_routes,
    aiter_tensor_t& route_scores,
    aiter_tensor_t& partials,
    aiter_tensor_t& order,
    aiter_tensor_t& topk_ids,
    aiter_tensor_t& dst,
    aiter_tensor_t& dlogits)
{
    const int T = static_cast<int>(dst.size(0));
    constexpr int TOKENS_PER_BLOCK = 2;
    opus_moe_gather_sum_dscore_router_token8_h2048_e64_bf16_kernel<<<
        dim3((T + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK),
        dim3(512), 0, aiter::getCurrentHIPStream()>>>(
        reinterpret_cast<const __bf16*>(src.data_ptr()),
        reinterpret_cast<const int32_t*>(token_routes.data_ptr()),
        reinterpret_cast<const float*>(route_scores.data_ptr()),
        reinterpret_cast<const float*>(partials.data_ptr()),
        reinterpret_cast<const int64_t*>(order.data_ptr()),
        reinterpret_cast<const int64_t*>(topk_ids.data_ptr()),
        reinterpret_cast<__bf16*>(dst.data_ptr()),
        reinterpret_cast<__bf16*>(dlogits.data_ptr()),
        T);
}

__global__ __launch_bounds__(512, 1)
void opus_moe_gather_sum_dscore_router_dx_token8_h2048_e64_bf16_kernel(
    const __bf16* __restrict__ src,
    const int32_t* __restrict__ token_routes,
    const float* __restrict__ route_scores,
    const float* __restrict__ partials,
    const int64_t* __restrict__ order,
    const int64_t* __restrict__ topk_ids,
    const __bf16* __restrict__ router_w,
    __bf16* __restrict__ dst,
    __bf16* __restrict__ dlogits,
    int T)
{
    constexpr int H = 2048;
    constexpr int TOPK = 8;
    constexpr int E = 64;
    constexpr int PARTIALS = 4;
    constexpr int THREADS_PER_TOKEN = 256;
    constexpr int TOKENS_PER_BLOCK = 2;
    constexpr int WAVES_PER_TOKEN = 4;
    const int token_in_block = threadIdx.x / THREADS_PER_TOKEN;
    const int token_thread = threadIdx.x % THREADS_PER_TOKEN;
    const int wave_in_token = token_thread / warpSize;
    const int lane = token_thread % warpSize;
    const int t = blockIdx.x * TOKENS_PER_BLOCK + token_in_block;
    if(t >= T)
        return;
    const int32_t* tr =
        token_routes + static_cast<int64_t>(t) * TOPK;
    __shared__ float selected_grad_by_wave
        [TOKENS_PER_BLOCK][WAVES_PER_TOKEN][TOPK];
    if(wave_in_token == 0 && lane < E)
        dlogits[static_cast<int64_t>(t) * E + token_thread] =
            static_cast<__bf16>(0.0f);
    float selected_grad = 0.0f;
    int rank = 0;
    if(lane < TOPK)
    {
        float scaled_dp = 0.0f;
        const int32_t route = tr[lane];
#pragma unroll
        for(int i = 0; i < PARTIALS; ++i)
            scaled_dp +=
                partials[static_cast<int64_t>(route) * PARTIALS + i];
        rank = static_cast<int>(order[route] & (TOPK - 1));
        float dot = 0.0f;
#pragma unroll
        for(int k = 0; k < TOPK; ++k)
            dot += __shfl(scaled_dp, k, TOPK);
        selected_grad = scaled_dp - route_scores[route] * dot;
        const __bf16 grad_bf16 = static_cast<__bf16>(selected_grad);
        selected_grad_by_wave[token_in_block][wave_in_token][rank] =
            static_cast<float>(grad_bf16);
        if(wave_in_token == 0)
        {
            const int64_t expert =
                topk_ids[static_cast<int64_t>(t) * TOPK + rank];
            dlogits[static_cast<int64_t>(t) * E + expert] = grad_bf16;
        }
    }
    __syncwarp();

    const int64_t dstrow = static_cast<int64_t>(t) * H;
    for(int h = token_thread * 8; h < H;
        h += THREADS_PER_TOKEN * 8)
    {
        float acc[8] = {};
#pragma unroll
        for(int k = 0; k < TOPK; ++k)
        {
            const opus_bf16x8 v =
                *reinterpret_cast<const opus_bf16x8*>(
                    src + static_cast<int64_t>(tr[k]) * H + h);
#pragma unroll
            for(int i = 0; i < 8; ++i)
                acc[i] += static_cast<float>(v[i]);
        }
#pragma unroll
        for(int k = 0; k < TOPK; ++k)
        {
            const int64_t expert =
                topk_ids[static_cast<int64_t>(t) * TOPK + k];
            const opus_bf16x8 v =
                *reinterpret_cast<const opus_bf16x8*>(
                    router_w + expert * H + h);
            // Each wave recomputes and rank-orders the eight selected
            // gradients in its own tiny LDS slot.  The producer and consumers
            // are in the same lockstep wave, so no whole-CTA barrier is needed.
            const float grad =
                selected_grad_by_wave[token_in_block][wave_in_token][k];
#pragma unroll
            for(int i = 0; i < 8; ++i)
                acc[i] = fmaf(grad, static_cast<float>(v[i]), acc[i]);
        }
#pragma unroll
        for(int i = 0; i < 8; ++i)
            dst[dstrow + h + i] = static_cast<__bf16>(acc[i]);
    }
}

void opus_moe_gather_sum_dscore_router_dx_token8_h2048_e64_bf16(
    aiter_tensor_t& src,
    aiter_tensor_t& token_routes,
    aiter_tensor_t& route_scores,
    aiter_tensor_t& partials,
    aiter_tensor_t& order,
    aiter_tensor_t& topk_ids,
    aiter_tensor_t& router_w,
    aiter_tensor_t& dst,
    aiter_tensor_t& dlogits)
{
    const int T = static_cast<int>(dst.size(0));
    constexpr int TOKENS_PER_BLOCK = 2;
    opus_moe_gather_sum_dscore_router_dx_token8_h2048_e64_bf16_kernel<<<
        dim3((T + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK),
        dim3(512), 0, aiter::getCurrentHIPStream()>>>(
        reinterpret_cast<const __bf16*>(src.data_ptr()),
        reinterpret_cast<const int32_t*>(token_routes.data_ptr()),
        reinterpret_cast<const float*>(route_scores.data_ptr()),
        reinterpret_cast<const float*>(partials.data_ptr()),
        reinterpret_cast<const int64_t*>(order.data_ptr()),
        reinterpret_cast<const int64_t*>(topk_ids.data_ptr()),
        reinterpret_cast<const __bf16*>(router_w.data_ptr()),
        reinterpret_cast<__bf16*>(dst.data_ptr()),
        reinterpret_cast<__bf16*>(dlogits.data_ptr()),
        T);
}

// Router backward (M5 R7): softmax-over-topk Jacobian. Per token t (one thread):
// s = Σ_k dp[t,k]·pw[t,k]; dlogits[t, ids[t,k]] = pw[t,k]·(dp[t,k]-s). The topk
// ids are distinct per token so writes don't collide (dlogits pre-zeroed).
__global__ void opus_moe_router_bwd_bf16_kernel(const float* __restrict__ dp,       // [T,topk]
                                                const float* __restrict__ pw,         // [T,topk]
                                                const int32_t* __restrict__ ids,     // [T,topk]
                                                __bf16* __restrict__ dlogits,        // [T,E]
                                                int T,
                                                int topk,
                                                int E)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if(t >= T)
        return;
    const float* dpt        = dp + static_cast<int64_t>(t) * topk;
    const float* pwt         = pw + static_cast<int64_t>(t) * topk;
    const int32_t* idt      = ids + static_cast<int64_t>(t) * topk;
    float s = 0.f;
    for(int k = 0; k < topk; ++k)
        s += dpt[k] * pwt[k];
    __bf16* drow = dlogits + static_cast<int64_t>(t) * E;
    for(int k = 0; k < topk; ++k)
        drow[idt[k]] = static_cast<__bf16>(pwt[k] * (dpt[k] - s));
}

void opus_moe_router_bwd_bf16(aiter_tensor_t& dp,       // [T,topk] fp32
                              aiter_tensor_t& topk_w,   // [T,topk] fp32
                              aiter_tensor_t& topk_ids, // [T,topk] i32
                              aiter_tensor_t& dlogits)  // [T,E] bf16 pre-zeroed
{
    const int T    = static_cast<int>(dp.size(0));
    const int topk = static_cast<int>(dp.size(1));
    const int E    = static_cast<int>(dlogits.size(1));
    if(T == 0)
        return;
    constexpr int BLK = 256;
    opus_moe_router_bwd_bf16_kernel<<<dim3((T + BLK - 1) / BLK), dim3(BLK), 0,
                                      aiter::getCurrentHIPStream()>>>(
        reinterpret_cast<const float*>(dp.data_ptr()),
        reinterpret_cast<const float*>(topk_w.data_ptr()),
        reinterpret_cast<const int32_t*>(topk_ids.data_ptr()),
        reinterpret_cast<__bf16*>(dlogits.data_ptr()),
        T, topk, E);
}

// Router backward, sigmoid scoring (DeepSeek/Kimi). Per token t (one thread):
// s_k=sigmoid(logits[t,ids[k]]); renorm w_k=s_k/Z (Z=Σs); dot=Σ dp_k·w_k;
// dlogits[t,ids[k]]=(dp_k-dot)/Z·s_k(1-s_k). No renorm: dlogits=dp_k·s_k(1-s_k).
// Non-selected experts are zero (sigmoid is per-expert; dlogits pre-zeroed).
__global__ void opus_moe_router_bwd_sigmoid_kernel(const float* __restrict__ dp,     // [T,topk]
                                                   const float* __restrict__ logits, // [T,E]
                                                   const int32_t* __restrict__ ids,  // [T,topk]
                                                   float* __restrict__ dlogits,      // [T,E]
                                                   int T,
                                                   int topk,
                                                   int E,
                                                   int renorm)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if(t >= T)
        return;
    const float* dpt   = dp + static_cast<int64_t>(t) * topk;
    const int32_t* idt = ids + static_cast<int64_t>(t) * topk;
    const float* lrow  = logits + static_cast<int64_t>(t) * E;
    float* drow        = dlogits + static_cast<int64_t>(t) * E;
    if(!renorm)
    {
        for(int k = 0; k < topk; ++k)
        {
            float s        = 1.f / (1.f + expf(-lrow[idt[k]]));
            drow[idt[k]]   = dpt[k] * s * (1.f - s);
        }
        return;
    }
    float Z = 0.f;
    for(int k = 0; k < topk; ++k)
        Z += 1.f / (1.f + expf(-lrow[idt[k]]));
    float dot = 0.f;
    for(int k = 0; k < topk; ++k)
    {
        float s = 1.f / (1.f + expf(-lrow[idt[k]]));
        dot += dpt[k] * (s / Z);
    }
    for(int k = 0; k < topk; ++k)
    {
        float s      = 1.f / (1.f + expf(-lrow[idt[k]]));
        float ds     = (dpt[k] - dot) / Z;
        drow[idt[k]] = ds * s * (1.f - s);
    }
}

void opus_moe_router_bwd_sigmoid_bf16(aiter_tensor_t& dp,       // [T,topk] fp32
                                      aiter_tensor_t& logits,   // [T,E] fp32
                                      aiter_tensor_t& topk_ids, // [T,topk] i32
                                      aiter_tensor_t& dlogits,  // [T,E] fp32 pre-zeroed
                                      int renorm)
{
    const int T    = static_cast<int>(dp.size(0));
    const int topk = static_cast<int>(dp.size(1));
    const int E    = static_cast<int>(dlogits.size(1));
    if(T == 0)
        return;
    constexpr int BLK = 256;
    opus_moe_router_bwd_sigmoid_kernel<<<dim3((T + BLK - 1) / BLK), dim3(BLK), 0,
                                         aiter::getCurrentHIPStream()>>>(
        reinterpret_cast<const float*>(dp.data_ptr()),
        reinterpret_cast<const float*>(logits.data_ptr()),
        reinterpret_cast<const int32_t*>(topk_ids.data_ptr()),
        reinterpret_cast<float*>(dlogits.data_ptr()),
        T, topk, E, renorm);
}
