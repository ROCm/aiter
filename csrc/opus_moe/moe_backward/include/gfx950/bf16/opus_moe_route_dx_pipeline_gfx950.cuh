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

inline __device__ uint32_t route_dx_cvt_pk_bf16_f32(float lo, float hi)
{
    uint32_t packed;
    asm volatile("v_cvt_pk_bf16_f32 %0, %1, %2"
                 : "=v"(packed)
                 : "v"(lo), "v"(hi));
    return packed;
}

template<typename T, typename Mma>
inline __device__ auto route_dx_load_rb_tr(
    opus::smem<typename T::D_B>& s_b,
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

    static_assert(T::W_N == 32 && T::W_K == 16);
    static_assert(T::VEC_TR_B == 4);
    static_assert(groups_n * groups_k == num_groups);

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
    typename Mma::vtype_b result;
    opus::static_for<T::E_N>([&](auto en) {
        opus::static_for<T::E_K>([&](auto ek) {
            opus::static_for<k_issues_per_group>([&](auto ki) {
                constexpr int issue =
                    (en.value * T::E_K + ek.value) * k_issues_per_group +
                    ki.value;
                const int k_in_half = base_k % half_k;
                const int base_bytes =
                    (base_k / half_k) * half_stage_bytes +
                    (k_in_half % rows_per_group) * T::SMEM_B_GROUP_BYTES +
                    (k_in_half / rows_per_group) * row_bytes +
                    base_n * sizeof(typename T::D_B);
                constexpr int byte_offset =
                    ek.value * half_stage_bytes + ki.value * row_bytes +
                    en.value * T::W_N * T::T_N * sizeof(typename T::D_B);
                auto values = s_b.template _tr_load<T::VEC_TR_B,
                                                     byte_offset>(base_bytes);
#pragma unroll
                for(int j = 0; j < T::VEC_TR_B; ++j)
                    result[issue * T::VEC_TR_B + j] = values[j];
            });
        });
    });
    return result;
}

#endif

template<typename Traits,
         int FixedD = 0,
         int FixedI = 0,
         int FixedTopK = 0,
         int RouteTiles = 1>
__device__ __forceinline__ void
route_dx_process_tile_gfx950(RouteDxKargs kargs)
{
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    using namespace opus;
    using opus::operator""_I;
    using T = opus::remove_cvref_t<Traits>;
    using D_A = typename T::D_A;
    using D_B = typename T::D_B;
    using D_ACC = typename T::D_ACC;

    constexpr int BM = T::B_M;
    constexpr int BN = T::B_N;
    constexpr int BK = T::B_K;
    constexpr int CTA_M = RouteTiles * BM;
    constexpr bool fixed_shape = FixedD > 0;
    static_assert(RouteTiles == 1 || RouteTiles == 2);
    static_assert((FixedD == 0 && FixedI == 0 && FixedTopK == 0) ||
                  (FixedD > 0 && FixedI > 0 && FixedTopK > 0));
    static_assert(!fixed_shape || (FixedD % BN == 0 && 2 * FixedI % BK == 0));
    const int model_dim = fixed_shape ? FixedD : kargs.model_dim;
    const int inter_dim = fixed_shape ? FixedI : kargs.inter_dim;
    const int topk = fixed_shape ? FixedTopK : kargs.route.topk;
    const int64_t stride_dz_r =
        fixed_shape ? 2 * FixedI : kargs.stride_dz_r;
    const int64_t stride_w1_e =
        fixed_shape ? 2 * FixedI * FixedD : kargs.stride_w1_e;
    const int64_t stride_w1_i =
        fixed_shape ? FixedD : kargs.stride_w1_i;
    const int64_t stride_dx_route_r =
        fixed_shape ? FixedD : kargs.stride_dx_route_r;
    constexpr int smem_a_bytes = CTA_M * BK * sizeof(D_A);
    constexpr int smem_b_bytes = T::SMEM_B_BYTES;
    constexpr int stage_bytes = smem_a_bytes + smem_b_bytes;

    const int col_base = static_cast<int>(blockIdx.y) * BN;
    const int valid_rows = kargs.route.num_valid_ids[0];
    const int route_tile = static_cast<int>(blockIdx.x);
    const int route_base = route_tile * BM;
    if(route_base >= valid_rows)
        return;

    const int expert_id = kargs.route.sorted_expert_ids[route_tile];
    if(expert_id < 0 || expert_id >= kargs.route.num_experts)
        return;
    int expert_end_row = valid_rows;
    if constexpr(RouteTiles == 2)
    {
        if(kargs.route.expert_offsets == nullptr)
            return;
        const int expert_first_tile =
            kargs.route.expert_offsets[expert_id] / BM;
        if((route_tile - expert_first_tile) & 1)
            return;
        // An expert may own an odd number of padded BM tiles.  In that case
        // this CTA still computes two accumulator groups, but the second one
        // must not publish rows belonging to the following expert.  Keeping
        // the tail masked avoids a separate single-M launch and preserves the
        // shared W1 load for every complete pair.
        expert_end_row = kargs.route.expert_offsets[expert_id + 1];
    }
    const int tid = static_cast<int>(thread_id_x());
    const int lane_id = tid % get_warp_size();
    const int wave_id = __builtin_amdgcn_readfirstlane(tid / get_warp_size());
    const int wave_id_m = wave_id / T::T_N;
    const int wave_id_n = wave_id % T::T_N;

    __shared__ __align__(16) char tile_storage[2 * stage_bytes];
    // K2 consumes expert-sorted rows but writes the logical token/slot route
    // order.  K3 can then reduce four adjacent rows without reverse_sorted.
    __shared__ int32_t smem_route_row[CTA_M];

    const D_A* d_z = reinterpret_cast<const D_A*>(kargs.d_z);
    const D_B* w1 = reinterpret_cast<const D_B*>(kargs.w1);
    const unsigned int d_z_bytes = static_cast<unsigned int>(
        static_cast<unsigned long long>(kargs.route.sorted_capacity) *
        static_cast<unsigned long long>(stride_dz_r) * sizeof(D_A));
    auto g_a = make_gmem(d_z, d_z_bytes);

    const int64_t w1_expert_base =
        static_cast<int64_t>(expert_id) * stride_w1_e;
    const int gate_up_dim = 2 * inter_dim;
    const unsigned int w1_bytes = static_cast<unsigned int>(
        ((static_cast<unsigned long long>(gate_up_dim - 1) *
              static_cast<unsigned long long>(stride_w1_i) +
          static_cast<unsigned long long>(model_dim)) *
         sizeof(D_B)));
    auto g_b = make_gmem(w1 + w1_expert_base, w1_bytes);

    auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
        seq<T::E_M, T::E_N, T::E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});
    auto p_coord_a = opus::make_tuple(
        wave_id_m, lane_id % mma.grpm_a, 0, lane_id / mma.grpm_a);
    auto u_ra = partition_layout_a<T::VEC_A>(
        mma, opus::make_tuple(BK, 1_I), p_coord_a);
    typename decltype(mma)::vtype_c v_c[RouteTiles];
    opus::static_for<RouteTiles>([&](auto tile) { clear(v_c[tile.value]); });

    constexpr int a_vectors = CTA_M * BK / T::VEC_A;
    static_assert(a_vectors > 0);
    int a_loader_base = 0;
    if(tid < a_vectors)
    {
        const int local_m = tid / (BK / T::VEC_A);
        const int sorted_row = route_base + local_m;
        // num_valid_ids is block_m padded, so every row of an active tile is
        // allocated.  Padding dZ may contain arbitrary BF16 values, but its
        // route is discarded by the epilogue; masking the GEMM load only adds
        // a metadata dependency to the hot prologue.
        a_loader_base = static_cast<int32_t>(
            static_cast<int64_t>(sorted_row) * stride_dz_r);
    }

    auto issue_tile = [&](int buffer, int tile_k) {
        OPUS_LDS_ADDR char* stage_lds =
            make_smem(tile_storage + buffer * stage_bytes).ptr;
        OPUS_LDS_ADDR D_A* a_lds =
            reinterpret_cast<OPUS_LDS_ADDR D_A*>(stage_lds);
        OPUS_LDS_ADDR D_B* b_lds =
            reinterpret_cast<OPUS_LDS_ADDR D_B*>(stage_lds + smem_a_bytes);

        if constexpr(a_vectors <= T::BLOCK_SIZE)
        {
            if constexpr(a_vectors == T::BLOCK_SIZE)
            {
                const int local_k =
                    (tid % (BK / T::VEC_A)) * T::VEC_A;
                // A gfx950 buffer_load_*_lds deposits lane L at
                // m0 + L * sizeof(vector).  Pass the wave base explicitly;
                // including the lane term here only makes LLVM reconstruct
                // the same uniform m0 with v_readfirstlane for every issue.
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
                const int local_k =
                    (tid % (BK / T::VEC_A)) * T::VEC_A;
                OPUS_LDS_ADDR D_A* a_wave_dst =
                    a_lds + wave_id * opus::get_warp_size() * T::VEC_A;
                g_a.template _async_load<T::VEC_A>(
                    reinterpret_cast<OPUS_LDS_ADDR void*>(a_wave_dst),
                    (a_loader_base + tile_k * BK + local_k) * sizeof(D_A),
                    0,
                    opus::number<0>{},
                    opus::number<T::CACHECTL_A>{});
            }
        }
        else
        {
            static_assert(a_vectors % T::BLOCK_SIZE == 0);
            constexpr int a_loads_per_thread =
                a_vectors / T::BLOCK_SIZE;
            opus::static_for<a_loads_per_thread>([&](auto load) {
                const int a_vector =
                    load.value * T::BLOCK_SIZE + tid;
                const int a_element = a_vector * T::VEC_A;
                const int local_m = a_element / BK;
                const int local_k = a_element % BK;
                const int sorted_row = route_base + local_m;
                const int source_base = static_cast<int32_t>(
                    static_cast<int64_t>(sorted_row) * stride_dz_r);
                OPUS_LDS_ADDR D_A* a_wave_dst =
                    a_lds +
                    (load.value * T::BLOCK_SIZE +
                     wave_id * opus::get_warp_size()) *
                        T::VEC_A;
                g_a.template _async_load<T::VEC_A>(
                    reinterpret_cast<OPUS_LDS_ADDR void*>(a_wave_dst),
                    (source_base + tile_k * BK + local_k) * sizeof(D_A),
                    0,
                    opus::number<0>{},
                    opus::number<T::CACHECTL_A>{});
            });
        }

        constexpr int vectors_per_group =
            T::SMEM_B_GROUP_DATA_BYTES / (T::VEC_B * sizeof(D_B));
        static_assert(vectors_per_group == opus::get_warp_size());
        constexpr int mfma_stage_groups =
            T::W_K / T::SMEM_B_GROUP_ROWS;
        constexpr int mfma_stage_bytes =
            mfma_stage_groups * T::SMEM_B_GROUP_BYTES;
        constexpr int mfma_stage_rows = T::W_K;
        const int b_local_k =
            wave_id +
            (lane_id / (BN / T::VEC_B)) * T::SMEM_B_GROUP_ROWS;
        const int b_local_n =
            (lane_id % (BN / T::VEC_B)) * T::VEC_B;
        const int b_global_offset =
            (tile_k * BK + b_local_k) * stride_w1_i + col_base +
            b_local_n;
        OPUS_LDS_ADDR char* b_wave_dst =
            reinterpret_cast<OPUS_LDS_ADDR char*>(b_lds) +
            wave_id * T::SMEM_B_GROUP_BYTES;
        opus::static_for<T::E_K>([&](auto stage) {
            constexpr int stage_offset =
                stage.value * mfma_stage_bytes;
            g_b.template _async_load<T::VEC_B>(
                reinterpret_cast<OPUS_LDS_ADDR void*>(
                    b_wave_dst + stage_offset),
                (b_global_offset +
                 stage.value * mfma_stage_rows * stride_w1_i) *
                    sizeof(D_B),
                0,
                opus::number<0>{},
                opus::number<T::CACHECTL_B>{});
        });
    };

    const int loops = gate_up_dim / BK;
    issue_tile(0, 0);
    if(tid < CTA_M)
    {
        const int sorted_row = route_base + tid;
        const int32_t packed =
            sorted_row < expert_end_row &&
                    sorted_row < kargs.route.sorted_capacity
                ? kargs.route.sorted_token_ids[sorted_row]
                : static_cast<int32_t>(kPackedTokenMask);
        if constexpr(T::ROUTE_LAYOUT == RouteLayout::CompactRouteMajor)
        {
            const auto decoded = decode_sorted_route<T::ROUTE_LAYOUT>(
                kargs.route,
                packed,
                sorted_row < expert_end_row &&
                    sorted_row < kargs.route.sorted_capacity);
            smem_route_row[tid] = decoded.valid ? decoded.logical : -1;
        }
        else
        {
            const int token = packed_token_id(packed);
            const int slot = packed_topk_slot(packed);
            smem_route_row[tid] =
                token < kargs.route.token_num && slot < topk
                    ? token * topk + slot
                    : -1;
        }
    }
    s_waitcnt_vmcnt(0_I);
    __syncthreads();

#pragma clang loop unroll_count(4)
    for(int tile_k = 0; tile_k < loops; ++tile_k)
    {
        const int buffer = tile_k & 1;
        const bool has_next = tile_k + 1 < loops;
        if(has_next)
            issue_tile(buffer ^ 1, tile_k + 1);

        auto s_b_current = make_smem(reinterpret_cast<D_B*>(
            tile_storage + buffer * stage_bytes + smem_a_bytes));
        auto v_b = route_dx_load_rb_tr<T, decltype(mma)>(
            s_b_current, lane_id, wave_id_n);
        opus::static_for<RouteTiles>([&](auto route_group) {
            auto s_a_current = make_smem(reinterpret_cast<D_A*>(
                tile_storage + buffer * stage_bytes +
                route_group.value * BM * BK * sizeof(D_A)));
            auto v_a = s_a_current.template load<T::VEC_A>(u_ra);
            v_c[route_group.value] =
                mma(v_a, v_b, v_c[route_group.value]);
        });

        if(has_next)
        {
            s_waitcnt_vmcnt(0_I);
            __syncthreads();
        }
    }
    static_assert(BM == 32 && BN == 128 && T::T_M == 1 && T::T_N == 4);
    static_assert(T::E_M == 1 && T::E_N == 1 && T::VEC_C == 4);
    using u32x4 = uint32_t __attribute__((ext_vector_type(4)));

    // A swap-ab MFMA lane owns four adjacent columns in each of four C
    // fragments.  The matching lane in the other wave half owns the next
    // four columns.  Pair the two halves exactly as gfx950 Triton does:
    // eight packed BF16 dwords -> four permlane swaps -> two dwordx4 stores.
    const int local_m = lane_id % 32;
    const int local_n0 = wave_id_n * 32 + (lane_id / 32) * 8;
    const int local_n1 = local_n0 + 16;
    opus::static_for<RouteTiles>([&](auto route_group) {
        uint32_t packed[8];
#pragma unroll
        for(int i = 0; i < 8; ++i)
            packed[i] = route_dx_cvt_pk_bf16_f32(
                static_cast<float>(v_c[route_group.value][2 * i]),
                static_cast<float>(v_c[route_group.value][2 * i + 1]));

        auto swap02 = __builtin_amdgcn_permlane32_swap(
            packed[0], packed[2], false, true);
        auto swap13 = __builtin_amdgcn_permlane32_swap(
            packed[1], packed[3], false, true);
        auto swap46 = __builtin_amdgcn_permlane32_swap(
            packed[4], packed[6], false, true);
        auto swap57 = __builtin_amdgcn_permlane32_swap(
            packed[5], packed[7], false, true);
        packed[0] = swap02[0];
        packed[2] = swap02[1];
        packed[1] = swap13[0];
        packed[3] = swap13[1];
        packed[4] = swap46[0];
        packed[6] = swap46[1];
        packed[5] = swap57[0];
        packed[7] = swap57[1];

        const int route_row =
            smem_route_row[route_group.value * BM + local_m];
        // smem_route_row is -1 for both the tail of the final sorted tile
        // and token/slot sentinels.  D is an exact multiple of BN.
        if(route_row >= 0)
        {
            const int64_t out_row =
                static_cast<int64_t>(route_row) * stride_dx_route_r +
                col_base;
            *reinterpret_cast<u32x4*>(
                kargs.d_x_route + out_row + local_n0) =
                u32x4{packed[0], packed[1], packed[2], packed[3]};
            *reinterpret_cast<u32x4*>(
                kargs.d_x_route + out_row + local_n1) =
                u32x4{packed[4], packed[5], packed[6], packed[7]};
        }
    });
#else
    (void)kargs;
#endif
#else
    (void)kargs;
#endif
}

template<typename Traits,
         int FixedD = 0,
         int FixedI = 0,
         int FixedTopK = 0,
         int RouteTiles = 1>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::MIN_BLOCKS_PER_CU)
void route_dx_kernel_gfx950(RouteDxKargs kargs)
{
    route_dx_process_tile_gfx950<
        Traits, FixedD, FixedI, FixedTopK, RouteTiles>(kargs);
}

template<typename Traits>
inline void route_dx_launch_gfx950(const RouteDxKargs& kargs,
                                   hipStream_t stream)
{
    using T = opus::remove_cvref_t<Traits>;
    AITER_CHECK(kargs.route.sort_block_m == T::B_M,
                "route_dx: sort block_m must equal ",
                T::B_M);
    AITER_CHECK(2 * kargs.inter_dim % T::B_K == 0,
                "route_dx: 2I must be divisible by ",
                T::B_K);
    AITER_CHECK(kargs.model_dim % T::B_N == 0,
                "route_dx: D must be divisible by ",
                T::B_N);
    const dim3 grid(
        static_cast<unsigned int>(kargs.route.sorted_block_capacity),
        static_cast<unsigned int>(kargs.model_dim / T::B_N));
    constexpr int target_d = 2048;
    constexpr int target_i = 384;
    constexpr int target_topk = 4;
    const bool use_target_shape =
        T::ROUTE_LAYOUT != RouteLayout::CompactRouteMajor &&
        kargs.model_dim == target_d && kargs.inter_dim == target_i &&
        kargs.route.topk == target_topk &&
        kargs.stride_dz_r == 2 * target_i &&
        kargs.stride_w1_e == 2 * target_i * target_d &&
        kargs.stride_w1_i == target_d &&
        kargs.stride_dx_route_r == target_d;
    if(use_target_shape)
    {
        if(kargs.route.expert_offsets != nullptr)
        {
            hipLaunchKernelGGL(
                (route_dx_kernel_gfx950<
                    T, target_d, target_i, target_topk, 2>),
                grid,
                dim3(T::BLOCK_SIZE),
                0,
                stream,
                kargs);
        }
        else
        {
            hipLaunchKernelGGL(
                (route_dx_kernel_gfx950<
                    T, target_d, target_i, target_topk, 1>),
                grid,
                dim3(T::BLOCK_SIZE),
                0,
                stream,
                kargs);
        }
    }
    else if constexpr(T::ROUTE_LAYOUT != RouteLayout::CompactRouteMajor)
    {
        if(kargs.route.expert_offsets != nullptr)
            hipLaunchKernelGGL((route_dx_kernel_gfx950<T, 0, 0, 0, 2>),
                               grid,
                               dim3(T::BLOCK_SIZE),
                               0,
                               stream,
                               kargs);
        else
            hipLaunchKernelGGL((route_dx_kernel_gfx950<T>),
                               grid,
                               dim3(T::BLOCK_SIZE),
                               0,
                               stream,
                               kargs);
    }
    else
    {
        hipLaunchKernelGGL((route_dx_kernel_gfx950<T>),
                           grid,
                           dim3(T::BLOCK_SIZE),
                           0,
                           stream,
                           kargs);
    }
}

} // namespace opus_moe_backward::gfx950
