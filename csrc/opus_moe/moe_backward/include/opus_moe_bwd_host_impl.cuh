// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

#include "aiter_tensor.h"
#include "aiter_stream.h"
#include "opus_moe_bwd.h"
#include "gfx950/opus_moe_dgrad_mfma_gfx950.cuh"
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
                            aiter_tensor_t& dW)
{
    const int E = static_cast<int>(dW.size(0));
    const int P = static_cast<int>(dW.size(1));
    const int Q = static_cast<int>(dW.size(2));
    opus_moe_wgrad_tn_launch_gfx950(
        reinterpret_cast<const __bf16*>(dy.data_ptr()),
        reinterpret_cast<const __bf16*>(a.data_ptr()),
        reinterpret_cast<const int32_t*>(offs.data_ptr()),
        reinterpret_cast<__bf16*>(dW.data_ptr()),
        E, P, Q, aiter::getCurrentHIPStream());
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
    opus_moe_combine_bwd_bf16_kernel<<<dim3(M), dim3(256), 0, aiter::getCurrentHIPStream()>>>(
        reinterpret_cast<const hip_bfloat16*>(dout.data_ptr()),
        reinterpret_cast<const int32_t*>(gather.data_ptr()),
        reinterpret_cast<const float*>(p.data_ptr()),
        reinterpret_cast<const hip_bfloat16*>(y.data_ptr()),
        reinterpret_cast<hip_bfloat16*>(dy.data_ptr()),
        reinterpret_cast<float*>(dp.data_ptr()),
        H);
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
// one block per token t sums its topk route rows of src -> dst[t]. No atomics
// (each dst[t] written by exactly one block), no pre-zero. bf16x8 vectorized;
// fp32 accumulate. token_routes[t,k] = compact route index of token t's k-th
// selection (built once from routing). Requires H % 8 == 0.
__global__ void opus_moe_gather_sum_bf16_kernel(const __bf16* __restrict__ src,           // [M,H]
                                                const int32_t* __restrict__ token_routes, // [T,topk]
                                                __bf16* __restrict__ dst,                 // [T,H]
                                                int H,
                                                int topk)
{
    const int t          = blockIdx.x;
    const int32_t* tr    = token_routes + static_cast<int64_t>(t) * topk;
    const int64_t dstrow = static_cast<int64_t>(t) * H;
    for(int h = threadIdx.x * 8; h < H; h += blockDim.x * 8)
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
    opus_moe_gather_sum_bf16_kernel<<<dim3(T), dim3(256), 0, aiter::getCurrentHIPStream()>>>(
        reinterpret_cast<const __bf16*>(src.data_ptr()),
        reinterpret_cast<const int32_t*>(token_routes.data_ptr()),
        reinterpret_cast<__bf16*>(dst.data_ptr()),
        H, topk);
}

// Router backward (M5 R7): softmax-over-topk Jacobian. Per token t (one thread):
// s = Σ_k dp[t,k]·pw[t,k]; dlogits[t, ids[t,k]] = pw[t,k]·(dp[t,k]-s). The topk
// ids are distinct per token so writes don't collide (dlogits pre-zeroed).
__global__ void opus_moe_router_bwd_bf16_kernel(const float* __restrict__ dp,       // [T,topk]
                                                const float* __restrict__ pw,         // [T,topk]
                                                const int32_t* __restrict__ ids,     // [T,topk]
                                                float* __restrict__ dlogits,         // [T,E]
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
    float* drow = dlogits + static_cast<int64_t>(t) * E;
    for(int k = 0; k < topk; ++k)
        drow[idt[k]] = pwt[k] * (dpt[k] - s);
}

void opus_moe_router_bwd_bf16(aiter_tensor_t& dp,       // [T,topk] fp32
                              aiter_tensor_t& topk_w,   // [T,topk] fp32
                              aiter_tensor_t& topk_ids, // [T,topk] i32
                              aiter_tensor_t& dlogits)  // [T,E] fp32 pre-zeroed
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
        reinterpret_cast<float*>(dlogits.data_ptr()),
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
