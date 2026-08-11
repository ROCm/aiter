// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "../../opus_moe_backward_common.cuh"
#include "opus_moe_backward_traits_gfx950.cuh"

#include "aiter_hip_common.h"
#include "opus/opus.hpp"

#include <cstdint>
#include <hip/hip_runtime.h>

namespace opus_moe_backward::gfx950
{

#ifdef __HIP_DEVICE_COMPILE__

inline __device__ uint32_t down_bwd_cvt_pk_bf16_f32(float lo, float hi)
{
    uint32_t packed;
    asm volatile("v_cvt_pk_bf16_f32 %0, %1, %2"
                 : "=v"(packed)
                 : "v"(lo), "v"(hi));
    return packed;
}

inline __device__ void down_bwd_permlane32_swap(uint32_t& lo, uint32_t& hi)
{
    asm volatile("v_permlane32_swap_b32 %0, %1" : "+v"(lo), "+v"(hi));
}

using down_bwd_f32x2 = float __attribute__((ext_vector_type(2)));
using down_bwd_f32x8 = float __attribute__((ext_vector_type(8)));
using down_bwd_u32x4 = uint32_t __attribute__((ext_vector_type(4)));

inline __device__ down_bwd_f32x8
down_bwd_load_bf16x8(const hip_bfloat16* values)
{
    const down_bwd_u32x4 packed =
        *reinterpret_cast<const down_bwd_u32x4*>(values);
    return down_bwd_f32x8{
        __builtin_bit_cast(float, packed[0] << 16),
        __builtin_bit_cast(float, packed[0] & 0xffff0000u),
        __builtin_bit_cast(float, packed[1] << 16),
        __builtin_bit_cast(float, packed[1] & 0xffff0000u),
        __builtin_bit_cast(float, packed[2] << 16),
        __builtin_bit_cast(float, packed[2] & 0xffff0000u),
        __builtin_bit_cast(float, packed[3] << 16),
        __builtin_bit_cast(float, packed[3] & 0xffff0000u)};
}

// Read a row-major [K,N] LDS tile with gfx950's transpose-load instruction and
// return the exact register order consumed by the swap_ab BF16 MFMA adaptor.
//
// A transpose read operates on four independent 16-lane groups.  For a
// W_N=32 wave, adjacent groups select the two 16-column halves while the high
// group bit selects one of two eight-row K bands.  Two issues per group then
// provide the eight BF16 values/lane consumed by each 32x32x16 MFMA operand.
// Keeping the issue displacement compile-time lets the Opus smem wrapper
// lower every access to ds_read_b64_tr_b16 with an immediate.
template<typename T, typename Mma>
inline __device__ auto down_bwd_load_rb_tr(opus::smem<typename T::D_B>& s_b,
                                           int lane_id,
                                           int wave_id_n)
{
    constexpr int lane_per_group = 16;
    constexpr int lane_lo = 4;
    constexpr int lane_hi = lane_per_group / lane_lo;
    constexpr int num_groups = opus::get_warp_size() / lane_per_group;
    constexpr int groups_n = T::W_N / (lane_lo * T::VEC_TR_B);
    constexpr int groups_k = num_groups / groups_n;
    constexpr int k_issues_per_group =
        T::W_K / (lane_hi * groups_k);

    static_assert(T::W_N == 32 && T::W_K == 16,
                  "transpose-read layout is specialized for the BF16 32x32x16 wave");
    static_assert(T::VEC_TR_B == 4,
                  "ds_read_b64_tr_b16 returns four BF16 values per issue");
    static_assert(groups_n * groups_k == num_groups);
    static_assert(T::W_K == lane_hi * groups_k * k_issues_per_group);

    const int group_id = lane_id / lane_per_group;
    const int lane_in_group = lane_id % lane_per_group;
    const int base_k =
        (group_id / groups_n) * lane_hi * k_issues_per_group +
        lane_in_group / lane_lo;
    const int base_n =
        wave_id_n * T::W_N +
        (group_id % groups_n) * lane_lo * T::VEC_TR_B +
        (lane_in_group % lane_lo) * T::VEC_TR_B;
    constexpr int row_bytes = T::SMEM_B_ROW_BYTES;
    constexpr int rows_per_group = T::SMEM_B_GROUP_ROWS;
    constexpr int half_k = T::W_K;
    constexpr int half_stage_bytes =
        (half_k / rows_per_group) * T::SMEM_B_GROUP_BYTES;
    const int k_in_half = base_k % half_k;
    const int base_bytes =
        (base_k / half_k) * half_stage_bytes +
        (k_in_half % rows_per_group) * T::SMEM_B_GROUP_BYTES +
        (k_in_half / rows_per_group) * row_bytes +
        base_n * sizeof(typename T::D_B);

    typename Mma::vtype_b result;
    opus::static_for<T::E_N>([&](auto en) {
        opus::static_for<T::E_K>([&](auto ek) {
            opus::static_for<k_issues_per_group>([&](auto ki) {
                constexpr int issue =
                    (en.value * T::E_K + ek.value) * k_issues_per_group +
                    ki.value;
                constexpr int byte_offset =
                    ek.value * half_stage_bytes + ki.value * row_bytes +
                    en.value * T::W_N * T::T_N * sizeof(typename T::D_B);
                auto values = s_b.template _tr_load<T::VEC_TR_B, byte_offset>(
                    base_bytes);
                for(int j = 0; j < T::VEC_TR_B; ++j)
                    result[issue * T::VEC_TR_B + j] = values[j];
            });
        });
    });
    return result;
}

__device__ __forceinline__ float down_bwd_sigmoid(float value)
{
    constexpr float kNegLog2E = -1.4426950408889634f;
    const float exponential = __builtin_amdgcn_exp2f(value * kNegLog2E);
    return __builtin_amdgcn_rcpf(1.0f + exponential);
}

#endif // __HIP_DEVICE_COMPILE__

template<typename Traits,
         int FixedD = 0,
         int FixedI = 0,
         int FixedTopK = 0>
__device__ __forceinline__ void
down_bwd_process_tile_gfx950(DownBwdKargs kargs)
{
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    using namespace opus;
    using opus::operator""_I;
    using T = opus::remove_cvref_t<Traits>;
    using D_A = typename T::D_A;
    using D_B = typename T::D_B;
    using D_ACC = typename T::D_ACC;

    constexpr int BN = T::B_N;
    constexpr int BK = T::B_K;
    constexpr int ROUTE_M = T::ROUTE_M;
    constexpr int CTA_M = ROUTE_M;
    constexpr bool fixed_shape = FixedD > 0;
    static_assert((FixedD == 0 && FixedI == 0 && FixedTopK == 0) ||
                  (FixedD > 0 && FixedI > 0 && FixedTopK > 0));
    static_assert(!fixed_shape || (FixedD % BK == 0 && FixedI % BN == 0));
    const int model_dim = fixed_shape ? FixedD : kargs.model_dim;
    const int inter_dim = fixed_shape ? FixedI : kargs.inter_dim;
    const int topk = fixed_shape ? FixedTopK : kargs.route.topk;
    const int64_t stride_do_t =
        fixed_shape ? FixedD : kargs.stride_do_t;
    const int64_t stride_z_r =
        fixed_shape ? 2 * FixedI : kargs.stride_z_r;
    const int64_t stride_w2_e =
        fixed_shape ? FixedD * FixedI : kargs.stride_w2_e;
    const int64_t stride_w2_d =
        fixed_shape ? FixedI : kargs.stride_w2_d;
    const int64_t stride_score_t =
        fixed_shape ? FixedTopK : kargs.stride_score_t;
    const int64_t stride_dz_r =
        fixed_shape ? 2 * FixedI : kargs.stride_dz_r;
    const int64_t stride_a_scaled_r =
        fixed_shape ? FixedI : kargs.stride_a_scaled_r;
    const int64_t stride_ds_t =
        fixed_shape ? FixedTopK : kargs.stride_ds_t;
    const int64_t stride_ds_workspace_r =
        fixed_shape ? FixedI / BN : kargs.stride_ds_workspace_r;
    constexpr int smem_a_elems = CTA_M * BK;
    constexpr int smem_b_elems = T::SMEM_B_BYTES / sizeof(D_B);
    constexpr int ds_values = ROUTE_M * T::T_N;
    constexpr int ds_bytes = ds_values * sizeof(float);
    constexpr int smem_a_bytes = smem_a_elems * sizeof(D_A);
    constexpr int smem_b_bytes = smem_b_elems * sizeof(D_B);
    constexpr int gemm_buffer_bytes = smem_a_bytes + smem_b_bytes;
    constexpr int gemm_smem_bytes = 2 * gemm_buffer_bytes;
    constexpr int tile_storage_bytes =
        gemm_smem_bytes > ds_bytes ? gemm_smem_bytes : ds_bytes;

    const int part = static_cast<int>(blockIdx.y);
    const int route_tile = static_cast<int>(blockIdx.x);
    const int route_base = route_tile * ROUTE_M;
    const int valid_rows = kargs.route.num_valid_ids[0];
    if(route_base >= valid_rows)
        return;

    const int expert_id = kargs.route.sorted_expert_ids[route_tile];
    if(expert_id < 0 || expert_id >= kargs.route.num_experts)
        return;

    const int tid = static_cast<int>(thread_id_x());
    const int lane_id = tid % get_warp_size();
    const int wave_id = __builtin_amdgcn_readfirstlane(tid / get_warp_size());
    const int wave_id_m = wave_id / T::T_N;
    const int wave_id_n = wave_id % T::T_N;
    const int col_base = part * BN;

    // GEMM operands die before the fused epilogue starts, so the FP32 dScore
    // scratch aliases the same LDS allocation.
    __shared__ __align__(16) char tile_storage[tile_storage_bytes];
    __shared__ int32_t smem_packed_route[CTA_M];

    if(tid < CTA_M)
    {
        const int sorted_row = route_base + tid;
        const bool row_in_range = sorted_row < valid_rows;
        const int32_t packed =
            row_in_range ? kargs.route.sorted_token_ids[sorted_row] : 0;
        bool valid;
        if constexpr(T::ROUTE_LAYOUT == RouteLayout::CompactRouteMajor)
        {
            const auto decoded = decode_sorted_route<T::ROUTE_LAYOUT>(
                kargs.route, packed, row_in_range);
            valid = decoded.valid;
        }
        else
        {
            const int token = packed_token_id(packed);
            const int slot = packed_topk_slot(packed);
            valid = row_in_range && token < kargs.route.token_num &&
                    slot < topk;
        }
        smem_packed_route[tid] = valid ? packed : -1;
    }
    __syncthreads();

    const D_A* d_out = reinterpret_cast<const D_A*>(kargs.d_out);
    const D_B* w2 = reinterpret_cast<const D_B*>(kargs.w2);
    const unsigned int d_out_bytes = static_cast<unsigned int>(
        static_cast<unsigned long long>(kargs.route.token_num) *
        static_cast<unsigned long long>(stride_do_t) * sizeof(D_A));
    auto g_a = make_gmem(d_out, d_out_bytes);

    const int64_t w2_expert_base =
        static_cast<int64_t>(expert_id) * stride_w2_e;
    const unsigned int w2_bytes = static_cast<unsigned int>(
        ((static_cast<unsigned long long>(model_dim - 1) *
              static_cast<unsigned long long>(stride_w2_d) +
          static_cast<unsigned long long>(inter_dim)) *
         sizeof(D_B)));
    auto g_b = make_gmem(w2 + w2_expert_base, w2_bytes);

    auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
        seq<T::E_M, T::E_N, T::E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});
    auto p_coord_a = opus::make_tuple(
        wave_id_m, lane_id % mma.grpm_a, 0, lane_id / mma.grpm_a);
    auto u_ra = partition_layout_a<T::VEC_A>(
        mma, opus::make_tuple(BK, 1_I), p_coord_a);
    typename decltype(mma)::vtype_c v_c;
    clear(v_c);

    constexpr int a_vectors = CTA_M * BK / T::VEC_A;
    static_assert(a_vectors > 0 && a_vectors <= T::BLOCK_SIZE);

    int a_loader_base = 0;
    if(tid < a_vectors)
    {
        const int local_m = tid / (BK / T::VEC_A);
        const int32_t packed = smem_packed_route[local_m];
        int token;
        bool valid;
        if constexpr(T::ROUTE_LAYOUT == RouteLayout::CompactRouteMajor)
        {
            const auto decoded = decode_sorted_route<T::ROUTE_LAYOUT>(
                kargs.route, packed, true);
            token = decoded.token;
            valid = decoded.valid;
        }
        else
        {
            token = packed_token_id(packed);
            const int slot = packed_topk_slot(packed);
            valid = token < kargs.route.token_num && slot < topk;
        }
        a_loader_base =
            valid ? static_cast<int32_t>(static_cast<int64_t>(token) *
                                         stride_do_t)
                  : static_cast<int32_t>(
                        static_cast<int64_t>(kargs.route.token_num) *
                        stride_do_t);
    }

    {
        // Two K=32 stages occupy the same 20 KiB payload as one K=64 tile.
        // Direct buffer_load_*_lds keeps the next stage in flight while the
        // current one feeds two 32x32x16 MFMAs.
        auto issue_tile = [&](int buffer, int tile_k) {
            auto stage_smem = make_smem(
                reinterpret_cast<char*>(tile_storage + buffer * gemm_buffer_bytes));
            OPUS_LDS_ADDR char* stage_lds = stage_smem.ptr;
            OPUS_LDS_ADDR D_A* a_lds =
                reinterpret_cast<OPUS_LDS_ADDR D_A*>(stage_lds);
            OPUS_LDS_ADDR D_B* b_lds =
                reinterpret_cast<OPUS_LDS_ADDR D_B*>(stage_lds + smem_a_bytes);

            if constexpr(a_vectors == T::BLOCK_SIZE)
            {
                const int local_k = (tid % (BK / T::VEC_A)) * T::VEC_A;
                // buffer_load_*_lds supplies the lane*vector stride in hardware;
                // keeping the destination wave-uniform avoids a per-issue
                // v_readfirstlane/m0 reconstruction in the 32-stage target loop.
                OPUS_LDS_ADDR D_A* a_wave_dst =
                    a_lds + wave_id * opus::get_warp_size() * T::VEC_A;
                g_a.template _async_load<T::VEC_A>(
                    reinterpret_cast<OPUS_LDS_ADDR void*>(a_wave_dst),
                    (a_loader_base + tile_k * BK + local_k) * sizeof(D_A),
                    0,
                    opus::number<0>{},
                    opus::number<T::CACHECTL_A>{});
            }
            else if(tid < a_vectors)
            {
                const int local_k = (tid % (BK / T::VEC_A)) * T::VEC_A;
                OPUS_LDS_ADDR D_A* a_wave_dst =
                    a_lds + wave_id * opus::get_warp_size() * T::VEC_A;
                g_a.template _async_load<T::VEC_A>(
                    reinterpret_cast<OPUS_LDS_ADDR void*>(a_wave_dst),
                    (a_loader_base + tile_k * BK + local_k) * sizeof(D_A),
                    0,
                    opus::number<0>{},
                    opus::number<T::CACHECTL_A>{});
            }

            constexpr int vectors_per_group =
                T::SMEM_B_GROUP_DATA_BYTES / (T::VEC_B * sizeof(D_B));
            static_assert(vectors_per_group == opus::get_warp_size());
            static_assert(T::SMEM_B_GROUPS == 4 * T::E_K);
            constexpr int mfma_stage_groups =
                T::W_K / T::SMEM_B_GROUP_ROWS;
            constexpr int mfma_stage_bytes =
                mfma_stage_groups * T::SMEM_B_GROUP_BYTES;
            constexpr int mfma_stage_rows = T::W_K;
            const int b_local_k =
                wave_id +
                (lane_id / (BN / T::VEC_B)) * T::SMEM_B_GROUP_ROWS;
            const int b_local_n = (lane_id % (BN / T::VEC_B)) * T::VEC_B;
            const int b_global_offset =
                (tile_k * BK + b_local_k) * stride_w2_d + col_base +
                b_local_n;
            OPUS_LDS_ADDR char* b_wave_dst =
                reinterpret_cast<OPUS_LDS_ADDR char*>(b_lds) +
                wave_id * T::SMEM_B_GROUP_BYTES;
            opus::static_for<T::E_K>([&](auto ek) {
                g_b.template _async_load<T::VEC_B>(
                    reinterpret_cast<OPUS_LDS_ADDR void*>(
                        b_wave_dst + ek.value * mfma_stage_bytes),
                    (b_global_offset +
                     ek.value * mfma_stage_rows * stride_w2_d) *
                        sizeof(D_B),
                    0,
                    opus::number<0>{},
                    opus::number<T::CACHECTL_B>{});
            });
        };

        const int loops = model_dim / BK;
        issue_tile(0, 0);
        s_waitcnt_vmcnt(0_I);
        __syncthreads();

        for(int tile_k = 0; tile_k < loops; ++tile_k)
        {
            const int buffer = tile_k & 1;
            const bool has_next = tile_k + 1 < loops;
            if(has_next)
                issue_tile(buffer ^ 1, tile_k + 1);

            auto s_b_current = make_smem(reinterpret_cast<D_B*>(
                tile_storage + buffer * gemm_buffer_bytes + smem_a_bytes));
            auto v_b = down_bwd_load_rb_tr<T, decltype(mma)>(
                s_b_current, lane_id, wave_id_n);
            auto s_a = make_smem(reinterpret_cast<D_A*>(
                tile_storage + buffer * gemm_buffer_bytes));
            auto v_a = s_a.template load<T::VEC_A>(u_ra);

            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_setprio(1);
            v_c = mma(v_a, v_b, v_c);
            __builtin_amdgcn_s_setprio(0);

            if(has_next)
            {
                s_waitcnt_vmcnt(0_I);
                __syncthreads();
            }
        }
    }
    __syncthreads();

    auto s_ds = make_smem(reinterpret_cast<float*>(tile_storage));
    OPUS_LDS_ADDR float* smem_ds =
        reinterpret_cast<OPUS_LDS_ADDR float*>(s_ds.ptr);

    auto store_epilogue = [&](auto& accum) {
        static_assert(ROUTE_M == 32 && BN == 128);
        static_assert(T::T_M == 1 && T::T_N == 4);
        static_assert(T::E_M == 1 && T::E_N == 1 && T::VEC_C == 4);
        using u32x4 = uint32_t __attribute__((ext_vector_type(4)));

        // Preserve FP32 precision through the fused activation backward.
        // Each swap exchanges one four-column fragment with lane +/- 32,
        // leaving two groups of eight contiguous columns.  Use the native
        // two-in/two-out instruction instead of the Clang builtin: the
        // latter can coalesce its second result with the first when the
        // inputs come directly from an FP32 MFMA accumulator.
#pragma unroll
        for(int group = 0; group < 2; ++group)
        {
#pragma unroll
            for(int i = 0; i < 4; ++i)
            {
                const int lo = group * 8 + i;
                const int hi = lo + 4;
                uint32_t lo_bits = __builtin_bit_cast(
                    uint32_t, static_cast<float>(accum[lo]));
                uint32_t hi_bits = __builtin_bit_cast(
                    uint32_t, static_cast<float>(accum[hi]));
                down_bwd_permlane32_swap(lo_bits, hi_bits);
                accum[lo] = __builtin_bit_cast(float, lo_bits);
                accum[hi] = __builtin_bit_cast(float, hi_bits);
            }
        }

        const int local_m = lane_id % 32;
        const int sorted_row = route_base + local_m;
        const bool row_in_range = sorted_row < valid_rows;
        const int32_t packed = smem_packed_route[local_m];
        int token;
        int slot;
        int logical_route;
        bool valid;
        if constexpr(T::ROUTE_LAYOUT == RouteLayout::CompactRouteMajor)
        {
            const auto decoded = decode_sorted_route<T::ROUTE_LAYOUT>(
                kargs.route, packed, row_in_range);
            token = decoded.token;
            slot = decoded.slot;
            logical_route = decoded.logical;
            valid = decoded.valid;
        }
        else
        {
            token = packed_token_id(packed);
            slot = packed_topk_slot(packed);
            valid = token < kargs.route.token_num && slot < topk;
            logical_route =
                static_cast<int>(logical_route_id(token, slot, topk));
        }
        float score = 0.0f;
        if(valid)
        {
            if constexpr(T::ROUTE_LAYOUT ==
                         RouteLayout::CompactRouteMajor)
                score = kargs.scores[logical_route];
            else
                score = kargs.scores[
                    static_cast<int64_t>(token) * stride_score_t + slot];
        }
        const down_bwd_f32x2 score2{score, score};
        const down_bwd_f32x2 one{1.0f, 1.0f};
        const int local_n0 = wave_id_n * 32 + (lane_id / 32) * 8;

        // Fetch both non-adjacent eight-column fragments before starting
        // the activation math.  Four independent 128-bit loads replace
        // eight serial 64-bit loads and overlap their global latency.
        down_bwd_f32x8 z_gate_prefetch[2];
        down_bwd_f32x8 z_up_prefetch[2];
        if(valid)
        {
#pragma unroll
            for(int group = 0; group < 2; ++group)
            {
                const int col = col_base + local_n0 + group * 16;
                const int64_t z_base =
                    static_cast<int64_t>(sorted_row) * stride_z_r +
                    col;
                z_gate_prefetch[group] =
                    down_bwd_load_bf16x8(kargs.z + z_base);
                z_up_prefetch[group] = down_bwd_load_bf16x8(
                    kargs.z + z_base + inter_dim);
            }
        }

        float ds_wave_partial = 0.0f;
#pragma unroll
        for(int group = 0; group < 2; ++group)
        {
            const int local_n = local_n0 + group * 16;
            const int col = col_base + local_n;
            u32x4 d_gate_store{};
            u32x4 d_up_store{};
            u32x4 scaled_store{};
            float ds_partial = 0.0f;

            if(valid)
            {
#pragma unroll
                for(int pair = 0; pair < 4; ++pair)
                {
                    const int elem = pair * 2;
                    const int acc_idx = group * 8 + elem;
                    const down_bwd_f32x2 z_gate_pair{
                        z_gate_prefetch[group][elem],
                        z_gate_prefetch[group][elem + 1]};
                    const down_bwd_f32x2 z_up_pair{
                        z_up_prefetch[group][elem],
                        z_up_prefetch[group][elem + 1]};
                    const down_bwd_f32x2 sigmoid{
                        down_bwd_sigmoid(z_gate_pair[0]),
                        down_bwd_sigmoid(z_gate_pair[1])};
                    const down_bwd_f32x2 silu = z_gate_pair * sigmoid;
                    const down_bwd_f32x2 activation = silu * z_up_pair;
                    const down_bwd_f32x2 g{
                        static_cast<float>(accum[acc_idx]),
                        static_cast<float>(accum[acc_idx + 1])};
                    const down_bwd_f32x2 q = score2 * g;
                    const down_bwd_f32x2 d_gate =
                        q * z_up_pair * sigmoid *
                        (one + z_gate_pair * (one - sigmoid));
                    const down_bwd_f32x2 d_up = q * silu;
                    const down_bwd_f32x2 scaled = score2 * activation;
                    d_gate_store[pair] = down_bwd_cvt_pk_bf16_f32(
                        d_gate[0], d_gate[1]);
                    d_up_store[pair] =
                        down_bwd_cvt_pk_bf16_f32(d_up[0], d_up[1]);
                    scaled_store[pair] = down_bwd_cvt_pk_bf16_f32(
                        scaled[0], scaled[1]);
                    ds_partial += g[0] * activation[0] +
                                  g[1] * activation[1];
                }
            }

            if(row_in_range && col + 8 <= inter_dim)
            {
                const int64_t dz_base =
                    static_cast<int64_t>(sorted_row) * stride_dz_r +
                    col;
                *reinterpret_cast<u32x4*>(kargs.d_z + dz_base) =
                    d_gate_store;
                *reinterpret_cast<u32x4*>(
                    kargs.d_z + dz_base + inter_dim) = d_up_store;
                const int64_t a_scaled_base =
                    static_cast<int64_t>(sorted_row) * stride_a_scaled_r +
                    col;
                *reinterpret_cast<u32x4*>(
                    kargs.a_scaled + a_scaled_base) = scaled_store;
            }
            ds_wave_partial += ds_partial;
        }
        ds_wave_partial += __shfl_xor(ds_wave_partial, 32, 64);
        if(lane_id < 32)
            smem_ds[wave_id_n * ROUTE_M + local_m] = ds_wave_partial;
    };

    store_epilogue(v_c);
    __syncthreads();

    static_assert(T::T_N == 4);
    if(tid < ROUTE_M)
    {
        float partial = smem_ds[tid];
#pragma unroll
        for(int wave = 1; wave < T::T_N; ++wave)
            partial += smem_ds[wave * ROUTE_M + tid];
        const int32_t packed = smem_packed_route[tid];
        if constexpr(T::ROUTE_LAYOUT ==
                     RouteLayout::CompactRouteMajor)
        {
            const auto decoded = decode_sorted_route<T::ROUTE_LAYOUT>(
                kargs.route, packed, true);
            if(decoded.valid)
            {
                const int logical_route = decoded.logical;
                if(kargs.d_scores_parts == 1)
                    kargs.d_scores[logical_route] = partial;
                else
                    kargs.d_scores_workspace[
                        static_cast<int64_t>(logical_route) *
                            stride_ds_workspace_r +
                        part] = partial;
            }
        }
        else
        {
            const int token = packed_token_id(packed);
            const int slot = packed_topk_slot(packed);
            if(token < kargs.route.token_num && slot < topk)
            {
                const int logical_route = static_cast<int>(
                    logical_route_id(token, slot, topk));
                if(kargs.d_scores_parts == 1)
                {
                    kargs.d_scores[static_cast<int64_t>(token) *
                                       stride_ds_t +
                                   slot] = partial;
                }
                else
                {
                    kargs.d_scores_workspace[
                        static_cast<int64_t>(logical_route) *
                            stride_ds_workspace_r +
                        part] = partial;
                }
            }
        }
    }
#else
    (void)kargs;
#endif // __gfx950__
#else
    (void)kargs;
#endif // __HIP_DEVICE_COMPILE__
}

// Match the forward stage1 structure: the global entry owns no scheduling or
// math.  All tile work lives in the flat process_tile routine above.
template<typename Traits,
         int FixedD = 0,
         int FixedI = 0,
         int FixedTopK = 0>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::MIN_BLOCKS_PER_CU)
void down_bwd_kernel_gfx950(DownBwdKargs kargs)
{
    down_bwd_process_tile_gfx950<Traits, FixedD, FixedI, FixedTopK>(kargs);
}

template<RouteLayout Layout, int Parts>
__global__ void down_bwd_dscore_finalize_gfx950(DownBwdKargs kargs)
{
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    const int logical_route =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
        static_cast<int>(threadIdx.x);
    const int route_count = logical_route_count<Layout>(kargs.route);
    if(logical_route >= route_count)
        return;
    float value = 0.0f;
    if constexpr(Parts > 0)
    {
        const int workspace_base = logical_route * Parts;
        opus::static_for<Parts>([&](auto part) {
            value += kargs.d_scores_workspace[workspace_base + part.value];
        });
    }
    else
    {
        for(int part = 0; part < kargs.d_scores_parts; ++part)
            value += kargs.d_scores_workspace[
                static_cast<int64_t>(logical_route) *
                    kargs.stride_ds_workspace_r +
                part];
    }
    if constexpr(Layout == RouteLayout::CompactRouteMajor || Parts > 0)
        kargs.d_scores[logical_route] = value;
    else
    {
        const int token = logical_route / kargs.route.topk;
        const int slot = logical_route - token * kargs.route.topk;
        kargs.d_scores[
            static_cast<int64_t>(token) * kargs.stride_ds_t + slot] = value;
    }
#else
    (void)kargs;
#endif
#else
    (void)kargs;
#endif
}

// The launcher owns the dScore partial reduction so K1 remains one public
// family even when I spans more than one BN tile.
template<typename Traits>
inline void down_bwd_launch_gfx950(const DownBwdKargs& kargs, hipStream_t stream)
{
    using T = opus::remove_cvref_t<Traits>;
    AITER_CHECK(kargs.route.sort_block_m == T::ROUTE_M,
                "down_bwd: sort block_m must equal kernel route BM=",
                T::ROUTE_M,
                ", got ",
                kargs.route.sort_block_m);
    AITER_CHECK(kargs.model_dim % T::B_K == 0,
                "down_bwd: first gfx950 instance requires model_dim divisible by ",
                T::B_K,
                ", got ",
                kargs.model_dim);
    AITER_CHECK(kargs.inter_dim % T::B_N == 0,
                "down_bwd: first gfx950 instance requires inter_dim divisible by ",
                T::B_N,
                ", got ",
                kargs.inter_dim);
    const int expected_parts = (kargs.inter_dim + T::B_N - 1) / T::B_N;
    AITER_CHECK(kargs.d_scores_parts == expected_parts,
                "down_bwd: d_scores_parts mismatch, expected ",
                expected_parts,
                ", got ",
                kargs.d_scores_parts);

    const dim3 block(T::BLOCK_SIZE);
    const dim3 grid(
        static_cast<unsigned int>(kargs.route.sorted_block_capacity),
        static_cast<unsigned int>(expected_parts));
    constexpr int target_d = 2048;
    constexpr int target_i = 384;
    constexpr int target_topk = 4;
    const bool use_target_shape =
        T::ROUTE_LAYOUT != RouteLayout::CompactRouteMajor &&
        kargs.model_dim == target_d && kargs.inter_dim == target_i &&
        kargs.route.topk == target_topk &&
        kargs.stride_do_t == target_d &&
        kargs.stride_z_r == 2 * target_i &&
        kargs.stride_w2_e == target_d * target_i &&
        kargs.stride_w2_d == target_i &&
        kargs.stride_score_t == target_topk &&
        kargs.stride_dz_r == 2 * target_i &&
        kargs.stride_a_scaled_r == target_i &&
        kargs.stride_ds_t == target_topk &&
        kargs.stride_ds_workspace_r == target_i / T::B_N;
    if(use_target_shape)
    {
        hipLaunchKernelGGL(
            (down_bwd_kernel_gfx950<T, target_d, target_i, target_topk>),
            grid,
            block,
            0,
            stream,
            kargs);
    }
    else
    {
        hipLaunchKernelGGL(
            (down_bwd_kernel_gfx950<T>),
            grid,
            block,
            0,
            stream,
            kargs);
    }

    if(expected_parts > 1)
    {
        if(expected_parts <= 4)
        {
            AITER_CHECK(kargs.stride_ds_workspace_r == expected_parts,
                        "down_bwd: specialized dScore finalize requires a "
                        "contiguous workspace");
            constexpr int expected_ds_stride =
                T::ROUTE_LAYOUT == RouteLayout::CompactRouteMajor ? 1 : -1;
            AITER_CHECK(
                kargs.stride_ds_t ==
                    (expected_ds_stride > 0 ? expected_ds_stride
                                            : kargs.route.topk),
                        "down_bwd: specialized dScore finalize requires a "
                        "contiguous d_scores tensor");
        }
        constexpr int finalize_threads = 256;
        const int route_count =
            logical_route_count<T::ROUTE_LAYOUT>(kargs.route);
        // Compact routing permits R=0.  K1 still materializes its sorted
        // scratch outputs, but there is no dScore row to finalize and HIP
        // does not permit a zero-sized launch grid.
        if(route_count == 0)
            return;
        const dim3 finalize_grid(
            static_cast<unsigned int>((route_count + finalize_threads - 1) /
                                      finalize_threads));
        switch(expected_parts)
        {
        case 2:
            hipLaunchKernelGGL(
                (down_bwd_dscore_finalize_gfx950<T::ROUTE_LAYOUT, 2>),
                finalize_grid,
                dim3(finalize_threads),
                0,
                stream,
                kargs);
            break;
        case 3:
            hipLaunchKernelGGL(
                (down_bwd_dscore_finalize_gfx950<T::ROUTE_LAYOUT, 3>),
                finalize_grid,
                dim3(finalize_threads),
                0,
                stream,
                kargs);
            break;
        case 4:
            hipLaunchKernelGGL(
                (down_bwd_dscore_finalize_gfx950<T::ROUTE_LAYOUT, 4>),
                finalize_grid,
                dim3(finalize_threads),
                0,
                stream,
                kargs);
            break;
        default:
            hipLaunchKernelGGL(
                (down_bwd_dscore_finalize_gfx950<T::ROUTE_LAYOUT, 0>),
                finalize_grid,
                dim3(finalize_threads),
                0,
                stream,
                kargs);
            break;
        }
    }
}

} // namespace opus_moe_backward::gfx950
