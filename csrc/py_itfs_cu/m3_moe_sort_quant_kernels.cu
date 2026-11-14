// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// MiniMax-M3.1 FHMoE decode (H 6144, 128 routed + 1 shared expert, top-4 + shared column, block_m 32, n <= 256):
// opus moe_sorting (incl. moe_buf zero) + fused_dynamic_mx_quant_moe_sort (MXFP4 / MXFP8) in ONE launch.
// Byte-identical to that pair on every row they define (sorted ids / weights / expert ids, num_valid_ids,
// payload, the scale bytes of every real route row); pad sorted rows' scale bytes are left unwritten.
// Every CTA sorts all routes itself, in LDS, deterministically (token order inside an expert), so no CTA
// waits for another: a grid-wide hand-off costs more than the launch it saves on MI355X.
#include "aiter_hip_common.h"
#include "aiter_opus_plus.h"
#include "aiter_stream.h"
#include "aiter_tensor.h"
#include "moe_sorting_opus.h"
#include "mx_quant_utils.h"
#include "quant_amax_floor.h"
#include <hip/hip_runtime.h>

namespace aiter::m3_sort_quant {
constexpr int E = 128, KS = 5, EP = E + 1, H = 6144, NG = H / 32, BM = 32;
constexpr int MAXN = 256, W = MAXN / 64, NT = 1024, NW = NT / 64;

__device__ __forceinline__ int rup(int c) { return (c + BM - 1) & ~(BM - 1); }
__device__ __forceinline__ int mock_id(int t, int k) { return (t & 0x00ffffff) | ((k & 0xff) << 24); }
__device__ __forceinline__ int rank_below(const unsigned long long* m, int t)
{
    int r = 0;
    for(int i = 0; i < (t >> 6); i++) r += __popcll(m[i]);
    return r + __popcll(m[t >> 6] & ((1ull << (t & 63)) - 1ull));
}

template <bool FP8>
__global__ void __launch_bounds__(NT) kernel(const __hip_bfloat16* __restrict__ x,
                                             const int* __restrict__ topk_ids,
                                             const float* __restrict__ topk_w,
                                             int* __restrict__ sorted_ids,
                                             float* __restrict__ sorted_w,
                                             int* __restrict__ sorted_eids,
                                             int* __restrict__ num_valid,
                                             __hip_bfloat16* __restrict__ moe_buf,
                                             uint8_t* __restrict__ xq,
                                             uint8_t* __restrict__ xscale,
                                             int n,
                                             const int* __restrict__ num_valid_rows)
{
    using DTYPE_O     = std::conditional_t<FP8, opus::fp8_t, opus::fp4_t>;
    using DTYPE_STORE = std::conditional_t<FP8, opus::fp8_t, uint8_t>;
    __shared__ int s_e[MAXN * KS];
    __shared__ float s_w[MAXN * KS];
    __shared__ unsigned long long s_mask[EP][W];
    __shared__ int s_off[EP + 1];
    __shared__ int s_cnt[EP];

    const int tid = threadIdx.x, lane = tid & 63, wave = tid >> 6;
    const int T = num_valid_rows ? min(n, *num_valid_rows) : n; // real rows (graph pad rows follow)

    for(int i = tid; i < EP * W; i += NT) (&s_mask[0][0])[i] = 0ull;
    if(moe_buf != nullptr)
    {
        int4* o = reinterpret_cast<int4*>(moe_buf);
        for(int i = blockIdx.x * NT + tid; i < n * H * 2 / 16; i += gridDim.x * NT) o[i] = int4{0, 0, 0, 0};
    }

    // Quant (QS's helpers and rounding): one wave = one third of a token's row, one 32-group per lane.
    // Every row < n, pad rows too (their payload is scratch nobody reads); scale bytes are placed after the sort.
    const int wi = blockIdx.x * NW + wave;
    const int qt = wi / 3, qg = (wi % 3) * 64 + lane;
    uint8_t sbyte = 0;
    if(wi < n * 3)
    {
        using vec_i = opus::vector_t<opus::bf16_t, 32>;
        using vec_f = opus::vector_t<float, 32>;
        vec_i vin   = *reinterpret_cast<const vec_i*>(x + (size_t)qt * H + qg * 32);
        vec_f vf;
        float amax = aiter::kAmaxFloor;
#pragma unroll
        for(int j = 0; j < 32; j++)
        {
            vf[j] = static_cast<float>(vin[j]);
            amax  = max(amax, abs(vf[j]));
        }
        float sc;
        if constexpr(FP8)
            sc = aiter::fp_f32_to_e8m0_scale<aiter::kDefaultMxScaleRoundMode, aiter::MxDtype::FP8_E4M3>(amax);
        else
            sc = aiter::fp4_f32_to_e8m0_scale(amax);
        sbyte                     = (__builtin_bit_cast(uint32_t, sc) >> 23) & 0xFF;
        const float inv           = FP8 ? __builtin_amdgcn_rcpf(sc) : sc;
        constexpr int row_bytes   = FP8 ? H : H / 2, vo = FP8 ? 32 : 16;
        auto buf = opus::make_gmem<DTYPE_STORE>(reinterpret_cast<DTYPE_STORE*>(xq + (size_t)qt * row_bytes), row_bytes);
        aiter::store_vector<DTYPE_STORE, float, 32, RT, false, WARP_SIZE, 1, DTYPE_O>(buf, vf, qg * vo, inv);
    }

    // Sort (opus layout): expert-major, token order inside an expert, empty experts skipped, blocks padded to BM.
    for(int i = tid; i < T * KS; i += NT)
    {
        s_e[i] = topk_ids[i];
        s_w[i] = topk_w[i];
    }
    __syncthreads();
    for(int i = tid; i < T * KS; i += NT)
    {
        const int e = s_e[i], t = i / KS;
        if(e >= 0 && e < EP) atomicOr(&s_mask[e][t >> 6], 1ull << (t & 63));
    }
    __syncthreads();
    if(wave == 0)
    { // per-expert counts padded to BM, exclusive scan: experts 2l, 2l+1 on lane l, expert 128 on lane 63
        int c[2], p[2];
#pragma unroll
        for(int j = 0; j < 2; j++)
        {
            const int e = 2 * lane + j;
            c[j]        = 0;
            for(int i = 0; i < W; i++) c[j] += __popcll(s_mask[e][i]);
            p[j]     = rup(c[j]);
            s_cnt[e] = c[j];
        }
        const int v = p[0] + p[1];
        int incl    = v;
#pragma unroll
        for(int d = 1; d < 64; d <<= 1)
        {
            const int u = __shfl_up(incl, d);
            if(lane >= d) incl += u;
        }
        s_off[2 * lane]     = incl - v;
        s_off[2 * lane + 1] = incl - v + p[0];
        if(lane == 63)
        {
            int c128 = 0;
            for(int i = 0; i < W; i++) c128 += __popcll(s_mask[E][i]);
            s_cnt[E]     = c128;
            s_off[E]     = incl;
            s_off[E + 1] = incl + rup(c128);
        }
    }
    __syncthreads();

    // pad slots and block -> expert ids, one wave per expert
    for(int e = blockIdx.x + gridDim.x * wave; e < EP; e += gridDim.x * NW)
    {
        const int s = s_off[e], c = s_cnt[e], end = s_off[e + 1];
        for(int p = s + c + lane; p < end; p += 64)
        {
            sorted_ids[p] = mock_id(T, KS);
            sorted_w[p]   = 0.0f;
        }
        for(int b = s / BM + lane; b < end / BM; b += 64) sorted_eids[b] = e;
    }
    // a real token's scale byte to its 5 sorted rows, and their sorted ids / weights
    if(wi < T * 3)
    {
#pragma unroll
        for(int k = 0; k < KS; k++)
        {
            const int e   = s_e[qt * KS + k];
            const int pos = s_off[e] + rank_below(s_mask[e], qt);
            xscale[aiter::mx_scale_shuffle_idx(NG, pos, qg)] = sbyte;
            if(qg == k)
            {
                sorted_ids[pos] = mock_id(qt, k);
                sorted_w[pos]   = s_w[qt * KS + k];
            }
        }
    }
    if(blockIdx.x == 0 && tid == 0)
    {
        num_valid[0] = s_off[E + 1];
        num_valid[1] = T;
    }
}
} // namespace aiter::m3_sort_quant

void m3_moe_sort_quant_fwd(aiter_tensor_t& hidden_states,
                           aiter_tensor_t& topk_ids,
                           aiter_tensor_t& topk_weights,
                           aiter_tensor_t& sorted_token_ids,
                           aiter_tensor_t& sorted_weights,
                           aiter_tensor_t& sorted_expert_ids,
                           aiter_tensor_t& num_valid_ids,
                           aiter_tensor_t& moe_buf,
                           aiter_tensor_t& activation_quant,
                           aiter_tensor_t& activation_scale,
                           std::optional<aiter_tensor_t> num_local_tokens)
{
    using namespace aiter::m3_sort_quant;
    const int n    = static_cast<int>(hidden_states.size(0));
    const bool fp8 = activation_quant.dtype() == AITER_DTYPE_fp8;
    // the 129-expert sort writes at most this many sorted rows (moe_sorting's allocation)
    const int64_t max_sorted = (int64_t(n) * KS + EP * BM - KS + BM - 1) / BM * BM;
    AITER_CHECK(hidden_states.dtype() == AITER_DTYPE_bf16 && hidden_states.dim() == 2 &&
                    hidden_states.size(1) == H && n >= 1 && n <= MAXN && topk_ids.dtype() == AITER_DTYPE_i32 &&
                    topk_ids.numel() == int64_t(n) * KS && topk_weights.dtype() == AITER_DTYPE_fp32 &&
                    topk_weights.numel() == int64_t(n) * KS && hidden_states.is_contiguous() &&
                    topk_ids.is_contiguous() && topk_weights.is_contiguous(),
                __func__,
                ": expected contiguous bf16 x [n<=256, 6144], int32 / fp32 top-k [n, 5]");
    AITER_CHECK((fp8 || activation_quant.dtype() == AITER_DTYPE_fp4x2) &&
                    activation_quant.numel() == int64_t(n) * (fp8 ? H : H / 2) &&
                    activation_scale.numel() >= (max_sorted + 31) / 32 * 32 * NG &&
                    sorted_token_ids.numel() >= max_sorted && sorted_weights.numel() >= max_sorted &&
                    sorted_expert_ids.numel() >= max_sorted / BM && num_valid_ids.numel() >= 2 &&
                    (moe_buf.numel() == 0 || (moe_buf.dtype() == AITER_DTYPE_bf16 && moe_buf.numel() == int64_t(n) * H)),
                __func__,
                ": incompatible output tensors");
    HipDeviceGuard device_guard(hidden_states.device_id);
    const int grid = std::min(64, (3 * n + NW - 1) / NW); // one 64-group third of a token per wave
    auto launch    = [&](auto kern) {
        kern<<<grid, NT, 0, aiter::getCurrentHIPStream()>>>(
            static_cast<const __hip_bfloat16*>(hidden_states.data_ptr()),
            static_cast<const int*>(topk_ids.data_ptr()),
            static_cast<const float*>(topk_weights.data_ptr()),
            static_cast<int*>(sorted_token_ids.data_ptr()),
            static_cast<float*>(sorted_weights.data_ptr()),
            static_cast<int*>(sorted_expert_ids.data_ptr()),
            static_cast<int*>(num_valid_ids.data_ptr()),
            moe_buf.numel() ? static_cast<__hip_bfloat16*>(moe_buf.data_ptr()) : nullptr,
            static_cast<uint8_t*>(activation_quant.data_ptr()),
            static_cast<uint8_t*>(activation_scale.data_ptr()),
            n,
            num_local_tokens ? static_cast<const int*>(num_local_tokens->data_ptr()) : nullptr);
    };
    if(fp8)
        launch(kernel<true>);
    else
        launch(kernel<false>);
    HIP_CALL(hipGetLastError());
}
