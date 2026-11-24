// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

// MiniMax-M3.1 TP1 MoE sort (128 experts, top-4, 128-row blocks) and moe_g2_w8's tile list in one
// launch: one WG per expert builds the histogram from the ids itself (no grid sync), then its rows.
#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "m3_tp1_prefill.h"

namespace aiter {
namespace {
constexpr int SE    = 128; // experts
constexpr int STOPK = 4;   // routes per token
constexpr int SBM   = 128; // sort block rows
constexpr int SNT   = 512; // threads per WG
constexpr int SNW   = SNT / 64;

// Token t = it * SNT + tid; its slots naming expert e form a 4-bit mask. The first is the route (a
// repeated id counts once); the others read -1 in reverse and get no route row (left unwritten).
template <int MAX_IT>
__global__ void __launch_bounds__(SNT) moe_sort_kernel(const int* __restrict__ topk_ids,
                                                       const float* __restrict__ topk_w,
                                                       const int* __restrict__ local_tokens,
                                                       int M,
                                                       int* __restrict__ sorted_ids,
                                                       float* __restrict__ sorted_w,
                                                       int* __restrict__ sorted_eids,
                                                       int* __restrict__ num_valid,
                                                       int* __restrict__ m_indices,
                                                       int* __restrict__ reverse,
                                                       int4* __restrict__ tiles,
                                                       int* __restrict__ ctl)
{
    __shared__ int s_cnt[SE];
    __shared__ int s_blk[SE];
    __shared__ int s_tile[SE];
    __shared__ int s_itw[MAX_IT][SNW];
    __shared__ int s_wtot[2][2];
    const int tid = threadIdx.x, lane = tid & 63, wave = tid >> 6;
    const int e = blockIdx.x;
    const int T = local_tokens != nullptr ? min(max(local_tokens[0], 0), M) : M;
    // rows past the local tokens read -1, as moe_sorting leaves them
    for(int i = T * STOPK + e * SNT + tid; i < M * STOPK; i += SE * SNT)
        reverse[i] = -1;
    if(tid < SE)
        s_cnt[tid] = 0;
    __syncthreads();
    // 1. histogram (each token once per expert) and this expert's slot masks, 4 bits per token
    const int4* ids4 = reinterpret_cast<const int4*>(topk_ids);
    uint32_t bits[MAX_IT / 8];
#pragma unroll
    for(int w = 0; w < MAX_IT / 8; ++w)
        bits[w] = 0;
    constexpr int NB = MAX_IT < 32 ? MAX_IT : 32;
#pragma unroll
    for(int ib = 0; ib < MAX_IT; ib += NB)
    {
        int4 v[NB];
#pragma unroll
        for(int j = 0; j < NB; ++j)
            v[j] = ids4[max(min((ib + j) * SNT + tid, T - 1), 0)];
#pragma unroll
        for(int j = 0; j < NB; ++j)
        {
            const int it  = ib + j;
            uint32_t mask = 0u;
            if(it * SNT + tid < T)
            {
                const int4 x = v[j];
                if((unsigned)x.x < SE)
                    atomicAdd(&s_cnt[x.x], 1);
                if((unsigned)x.y < SE && x.y != x.x)
                    atomicAdd(&s_cnt[x.y], 1);
                if((unsigned)x.z < SE && x.z != x.x && x.z != x.y)
                    atomicAdd(&s_cnt[x.z], 1);
                if((unsigned)x.w < SE && x.w != x.x && x.w != x.y && x.w != x.z)
                    atomicAdd(&s_cnt[x.w], 1);
                mask = (x.x == e ? 1u : 0u) | (x.y == e ? 2u : 0u) | (x.z == e ? 4u : 0u) |
                       (x.w == e ? 8u : 0u);
            }
            bits[it >> 3] |= mask << (4 * (it & 7));
        }
    }
    float wsel[MAX_IT];
#pragma unroll
    for(int it = 0; it < MAX_IT; ++it)
    {
        const uint32_t mask = (bits[it >> 3] >> (4 * (it & 7))) & 0xFu;
        wsel[it]            = mask ? topk_w[(it * SNT + tid) * STOPK + __builtin_ctz(mask)] : 0.0f;
        const unsigned long long b = __ballot(mask != 0u);
        if(lane == 0)
            s_itw[it][wave] = __popcll(b);
    }
    __syncthreads();
    // 2. block and tile offsets: exclusive prefix over the experts (no block for an empty one)
    if(tid < SE)
    {
        const int nbx = (s_cnt[tid] + SBM - 1) / SBM;
        int ib = nbx, itl = (nbx + 1) / 2;
#pragma unroll
        for(int off = 1; off < 64; off <<= 1)
        {
            const int yb = __shfl_up(ib, off, 64);
            const int yt = __shfl_up(itl, off, 64);
            if(lane >= off)
            {
                ib += yb;
                itl += yt;
            }
        }
        if(lane == 63)
        {
            s_wtot[wave][0] = ib;
            s_wtot[wave][1] = itl;
        }
        s_blk[tid]  = ib - nbx;
        s_tile[tid] = itl - (nbx + 1) / 2;
    }
    __syncthreads();
    if(tid >= 64 && tid < SE)
    {
        s_blk[tid] += s_wtot[0][0];
        s_tile[tid] += s_wtot[0][1];
    }
    __syncthreads();
    if(e == 0 && tid <= 8)
    {
        if(tid == 0)
        {
            num_valid[0] = (s_wtot[0][0] + s_wtot[1][0]) * SBM;
            num_valid[1] = T;
            ctl[0]       = s_wtot[0][1] + s_wtot[1][1];
        }
        else
            ctl[tid] = 0; // moe_g2_w8's work queues
    }
    const int cnt = s_cnt[e];
    if(cnt == 0)
        return;
    const int nb    = (cnt + SBM - 1) / SBM;
    const int b0    = s_blk[e];
    const int start = b0 * SBM;
    for(int j = tid; j < nb; j += SNT)
        sorted_eids[b0 + j] = e;
    // 3. this expert's routes in token order
    int run = 0;
#pragma unroll
    for(int it = 0; it < MAX_IT; ++it)
    {
        const uint32_t mask        = (bits[it >> 3] >> (4 * (it & 7))) & 0xFu;
        const unsigned long long b = __ballot(mask != 0u);
        int before = 0, tot = 0;
#pragma unroll
        for(int w = 0; w < SNW; ++w)
        {
            const int c = s_itw[it][w];
            before += w < wave ? c : 0;
            tot += c;
        }
        if(mask != 0u)
        {
            const int t    = it * SNT + tid;
            const int slot = __builtin_ctz(mask);
            const int p    = start + run + before + __popcll(b & ((1ull << lane) - 1ull));
            sorted_ids[p]  = (int)(((uint32_t)t & 0x00ffffffu) | ((uint32_t)slot << 24));
            sorted_w[p]    = wsel[it];
            m_indices[p]   = t;
            reverse[t * STOPK + slot] = p;
            for(uint32_t rest = mask & (mask - 1u); rest != 0u; rest &= rest - 1u)
                reverse[t * STOPK + __builtin_ctz(rest)] = -1;
        }
        run += tot;
    }
    // padding of the expert's last block, as moe_sorting writes it
    for(int i = cnt + tid; i < nb * SBM; i += SNT)
    {
        sorted_ids[start + i] = (int)(((uint32_t)M & 0x00ffffffu) | ((uint32_t)STOPK << 24));
        sorted_w[start + i]   = 0.0f;
        m_indices[start + i]  = T;
    }
    // 4. moe_g2_w8's tiles: the expert's blocks paired two by two
    for(int j = 2 * tid; j < nb; j += 2 * SNT)
    {
        const int v0             = min(SBM, cnt - j * SBM);
        const int v1             = j + 1 < nb ? min(SBM, cnt - (j + 1) * SBM) : 0;
        tiles[s_tile[e] + j / 2] = make_int4((b0 + j) * SBM, v0 + v1, e, 0);
    }
}

void check_i32(const aiter_tensor_t& t, size_t numel, const char* name)
{
    AITER_CHECK(t.dtype() == AITER_DTYPE_i32 && t.is_contiguous() && t.numel() >= numel,
                "m3_tp1_moe_sort: ",
                name,
                " must be a contiguous int32 tensor of at least ",
                numel,
                " elements");
}
} // namespace

void m3_tp1_moe_sort(const aiter_tensor_t& topk_ids,
                     const aiter_tensor_t& topk_weights,
                     std::optional<aiter_tensor_t> num_local_tokens,
                     aiter_tensor_t& sorted_ids,
                     aiter_tensor_t& sorted_weights,
                     aiter_tensor_t& sorted_expert_ids,
                     aiter_tensor_t& num_valid_ids,
                     aiter_tensor_t& m_indices,
                     aiter_tensor_t& reverse_sorted,
                     aiter_tensor_t& tiles,
                     aiter_tensor_t& ctl)
{
    aiter_detail::AiterThrowGuard throw_guard;
    AITER_CHECK(topk_ids.dtype() == AITER_DTYPE_i32 && topk_ids.is_contiguous() &&
                    topk_ids.dim() == 2 && topk_ids.size(1) == STOPK,
                "m3_tp1_moe_sort: topk_ids must be a contiguous int32 [M, 4] tensor");
    const int64_t M = topk_ids.size(0);
    AITER_CHECK(M <= 64 * SNT, "m3_tp1_moe_sort: at most ", 64 * SNT, " tokens, got ", M);
    AITER_CHECK(topk_weights.dtype() == AITER_DTYPE_fp32 && topk_weights.is_contiguous() &&
                    topk_weights.numel() == M * STOPK,
                "m3_tp1_moe_sort: topk_weights must be a contiguous fp32 [M, 4] tensor");
    // every route plus 127 padding rows per expert, rounded to blocks (moe_sorting's size)
    const size_t rows = (M * STOPK + SE * SBM - STOPK + SBM - 1) / SBM * SBM;
    check_i32(sorted_ids, rows, "sorted_ids");
    AITER_CHECK(sorted_weights.dtype() == AITER_DTYPE_fp32 && sorted_weights.is_contiguous() &&
                    sorted_weights.numel() >= rows,
                "m3_tp1_moe_sort: sorted_weights must be a contiguous fp32 tensor of ",
                rows,
                " rows");
    check_i32(sorted_expert_ids, rows / SBM, "sorted_expert_ids");
    check_i32(num_valid_ids, 2, "num_valid_ids");
    check_i32(m_indices, rows, "m_indices");
    check_i32(reverse_sorted, M * STOPK, "reverse_sorted");
    check_i32(tiles, rows / SBM * 4, "tiles");
    check_i32(ctl, 9, "ctl");
    if(num_local_tokens.has_value())
        check_i32(*num_local_tokens, 1, "num_local_tokens");
    if(M == 0)
        return;

    const HipDeviceGuard guard(topk_ids.device_id);
    const hipStream_t stream = getCurrentHIPStream();
    const int* nlt           = num_local_tokens.has_value()
                                   ? static_cast<const int*>(num_local_tokens->data_ptr())
                                   : nullptr;
#define M3_MOE_SORT_LAUNCH(IT)                                                                   \
    moe_sort_kernel<IT>                                                                          \
        <<<dim3(SE), dim3(SNT), 0, stream>>>(static_cast<const int*>(topk_ids.data_ptr()),       \
                                             static_cast<const float*>(topk_weights.data_ptr()), \
                                             nlt,                                                \
                                             static_cast<int>(M),                                \
                                             static_cast<int*>(sorted_ids.data_ptr()),           \
                                             static_cast<float*>(sorted_weights.data_ptr()),     \
                                             static_cast<int*>(sorted_expert_ids.data_ptr()),    \
                                             static_cast<int*>(num_valid_ids.data_ptr()),        \
                                             static_cast<int*>(m_indices.data_ptr()),            \
                                             static_cast<int*>(reverse_sorted.data_ptr()),       \
                                             static_cast<int4*>(tiles.data_ptr()),               \
                                             static_cast<int*>(ctl.data_ptr()))
    if(M <= 8 * SNT)
        M3_MOE_SORT_LAUNCH(8);
    else if(M <= 16 * SNT)
        M3_MOE_SORT_LAUNCH(16);
    else if(M <= 32 * SNT)
        M3_MOE_SORT_LAUNCH(32);
    else
        M3_MOE_SORT_LAUNCH(64);
#undef M3_MOE_SORT_LAUNCH
    HIP_CALL(hipGetLastError());
}

} // namespace aiter
