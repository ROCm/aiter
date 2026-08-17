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

inline __device__ down_bwd_u32x4
down_bwd_load_bf16x8_packed(const hip_bfloat16* values)
{
    return *reinterpret_cast<const down_bwd_u32x4*>(values);
}

inline __device__ down_bwd_f32x8
down_bwd_unpack_bf16x8(const down_bwd_u32x4& packed)
{
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

// Read four independently swizzled [K32,N64] slabs.  Within a slab two
// native N32 fragments differ by one 64-byte XOR phase; consecutive K16
// fragments differ by 2048 bytes.  This is the same physical encoding used
// by K4's split-B path, adapted here to the BN256 down-backward operand.
template<typename T, typename Mma>
inline __device__ auto
down_bwd_load_rb_tr_split_n64(opus::smem<typename T::D_B>& s_b,
                              int lane_id,
                              int wave_id_n)
{
    static_assert(T::B_N == 256 && T::B_K == 32);
    static_assert(T::T_N == 4 && T::E_N == 2 && T::E_K == 2);
    static_assert(T::W_N == 32 && T::W_K == 16);
    static_assert(T::VEC_TR_B == 4);

    const int lane_mix = lane_id & 0x2c;
    const int read_base =
        (lane_mix << 5) |
        (((lane_id << 1) & 0x70) ^ ((lane_id << 3) & 0x18));

    typename Mma::vtype_b result;
    opus::static_for<T::E_N>([&](auto en) {
        opus::static_for<T::E_K>([&](auto ek) {
            const int native_n = en.value * T::T_N + wave_id_n;
            int base = read_base + (native_n / 2) * 4096;
            base ^= (native_n % 2) * 0x40;
            constexpr int k_byte_offset = ek.value * 2048;
            auto lo = s_b.template _tr_load<T::VEC_TR_B, k_byte_offset>(base);
            auto hi = s_b.template _tr_load<T::VEC_TR_B, k_byte_offset>(
                base ^ 0x220);
            constexpr int fragment = en.value * T::E_K + ek.value;
#pragma unroll
            for(int j = 0; j < T::VEC_TR_B; ++j)
            {
                result[fragment * 2 * T::VEC_TR_B + j] = lo[j];
                result[fragment * 2 * T::VEC_TR_B + T::VEC_TR_B + j] =
                    hi[j];
            }
        });
    });
    return result;
}

// Locally permute each row's four K vectors with a self-inverse XOR.  Both the
// direct-to-LDS writer and MFMA reader need only one shift/mask/xor, the tile
// footprint is unchanged, and source-lane reordering stays within each row.
// Keeping the native fragment addressing explicit also gives the compiler a
// shorter LDS dependency chain than the generic layout expression.
template<typename T, typename Mma, typename RouteGroup>
inline __device__ auto
down_bwd_load_ra_swizzled_xor(opus::smem<typename T::D_A>& s_a,
                              int lane_id,
                              RouteGroup route_group)
{
    static_assert(T::B_K == 32 && T::W_M == 32 && T::W_K == 16);
    static_assert(T::E_M == 1 && T::E_K == 2 && T::VEC_A == 8);

    constexpr int vectors_per_row = T::B_K / T::VEC_A;
    const int logical_row = lane_id % T::W_M;
    const int lane_k_vector = lane_id / T::W_M;

    typename Mma::vtype_a result;
    opus::static_for<T::E_K>([&](auto ek) {
        const int logical_k_vector =
            lane_k_vector + ek.value * (T::W_K / T::VEC_A);
        const int logical_vector =
            logical_row * vectors_per_row + logical_k_vector;
        constexpr int group_vector_base =
            route_group.value * T::ROUTE_M * vectors_per_row;
        const int global_logical_vector =
            group_vector_base + logical_vector;
        const int physical_vector =
            (global_logical_vector ^
             ((global_logical_vector >> T::A_XOR_SHIFT) &
              T::A_XOR_MASK)) -
            group_vector_base;
        auto values = s_a.template load<T::VEC_A>(
            physical_vector * T::VEC_A);
#pragma unroll
        for(int j = 0; j < T::VEC_A; ++j)
            result[ek.value * T::VEC_A + j] = values[j];
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
         int FixedTopK = 0,
         int RouteTiles = 1>
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
    constexpr int CTA_M = ROUTE_M * RouteTiles;
    constexpr bool batch_sigmoid = []() constexpr {
        if constexpr(requires { T::BATCH_SIGMOID; })
            return T::BATCH_SIGMOID;
        return false;
    }();
    constexpr bool stage_z_in_lds = []() constexpr {
        if constexpr(requires { T::STAGE_Z_IN_LDS; })
            return T::STAGE_Z_IN_LDS;
        return false;
    }();
    constexpr int z_lds_pair_pad = []() constexpr {
        if constexpr(requires { T::Z_LDS_PAIR_PAD; })
            return T::Z_LDS_PAIR_PAD;
        return 0;
    }();
    constexpr bool defer_z_lds_wait = []() constexpr {
        if constexpr(requires { T::DEFER_Z_LDS_WAIT; })
            return T::DEFER_Z_LDS_WAIT;
        return false;
    }();
    constexpr bool request_pipeline_z_route_groups = []() constexpr {
        if constexpr(requires { T::PIPELINE_Z_ROUTE_GROUPS; })
            return T::PIPELINE_Z_ROUTE_GROUPS;
        return false;
    }();
    constexpr bool pipeline_z_route_groups =
        request_pipeline_z_route_groups && RouteTiles > 1;
    constexpr int cachectl_z = []() constexpr {
        if constexpr(requires { T::CACHECTL_Z; })
            return T::CACHECTL_Z;
        return 0;
    }();
    constexpr bool issue_b_first = []() constexpr {
        if constexpr(requires { T::ISSUE_B_FIRST; })
            return T::ISSUE_B_FIRST;
        return false;
    }();
    constexpr int prefetch_a_tiles = []() constexpr {
        if constexpr(requires { T::PREFETCH_A_TILES; })
            return T::PREFETCH_A_TILES;
        return 0;
    }();
    constexpr bool write_a_scaled = []() constexpr {
        if constexpr(requires { T::WRITE_A_SCALED; })
            return T::WRITE_A_SCALED;
        return true;
    }();
    constexpr bool blocked_dz_g2 = []() constexpr {
        if constexpr(requires { T::BLOCKED_DZ_G2; })
            return T::BLOCKED_DZ_G2;
        return false;
    }();
    constexpr bool split_b_n64 = []() constexpr {
        if constexpr(requires { T::SPLIT_B_N64_SWIZZLE; })
            return T::SPLIT_B_N64_SWIZZLE;
        return false;
    }();
    constexpr int a_xor_shift = []() constexpr {
        if constexpr(requires { T::A_XOR_SHIFT; })
            return T::A_XOR_SHIFT;
        return 0;
    }();
    constexpr int a_xor_mask = []() constexpr {
        if constexpr(requires { T::A_XOR_MASK; })
            return T::A_XOR_MASK;
        return 0;
    }();
    constexpr bool swizzle_a_xor = a_xor_mask != 0;
    static_assert(!split_b_n64 ||
                      (BN == 256 && BK == 32 && T::BLOCK_SIZE == 256),
                  "split-B down backward requires the four-wave BN256 tile");
    static_assert(!swizzle_a_xor ||
                      (BK == 32 && T::W_M == 32 && T::W_K == 16 &&
                       T::E_M == 1 && T::E_K == 2 && T::VEC_A == 8),
                  "A XOR layout requires the native BF16 K32 fragment");
    static_assert(!swizzle_a_xor ||
                      (a_xor_shift >= 2 && a_xor_shift <= 6 &&
                       a_xor_mask > 0 && a_xor_mask <= 3),
                  "A XOR layout requires shift 2..6 and a two-bit mask");
    static_assert(!blocked_dz_g2 ||
                      (ROUTE_M == 32 && T::VEC_C == 4),
                  "blocked dZ requires M32 and eight-BF16 stores");
    static_assert(RouteTiles > 0);
    static_assert(CTA_M <= T::BLOCK_SIZE,
                  "each predecoded route row requires one workgroup thread");
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
    constexpr int ds_values = CTA_M * T::T_N;
    constexpr int ds_bytes = ds_values * sizeof(float);
    constexpr int z_lds_pair_elems = 4 * BN + z_lds_pair_pad;
    constexpr int z_lds_bytes =
        stage_z_in_lds
            ? (ROUTE_M / 2) * z_lds_pair_elems * sizeof(D_A)
            : 0;
    constexpr int z_lds_stages = pipeline_z_route_groups ? 2 : 1;
    constexpr int z_lds_total_bytes = z_lds_stages * z_lds_bytes;
    constexpr int epilogue_storage_bytes = z_lds_total_bytes + ds_bytes;
    constexpr int smem_a_bytes = smem_a_elems * sizeof(D_A);
    constexpr int smem_b_bytes = smem_b_elems * sizeof(D_B);
    constexpr int gemm_buffer_bytes = smem_a_bytes + smem_b_bytes;
    constexpr int gemm_smem_bytes = 2 * gemm_buffer_bytes;
    constexpr int tile_storage_bytes =
        gemm_smem_bytes > epilogue_storage_bytes
            ? gemm_smem_bytes
            : epilogue_storage_bytes;
    static_assert(!stage_z_in_lds || z_lds_pair_pad > 0);
    static_assert(!request_pipeline_z_route_groups ||
                      (stage_z_in_lds && defer_z_lds_wait),
                  "route-group Z pipeline requires deferred staged Z");

    const int parts = inter_dim / BN;
    const int linear_block = static_cast<int>(blockIdx.x);
    int part;
    int route_tile;
    int expert_id = -1;
    int expert_end_row = kargs.route.num_valid_ids[0];
    constexpr bool compact_group_grid =
        T::COMPACT_ROUTE_GROUP_GRID && RouteTiles > 1;
    constexpr bool sparse_compact_owner = []() constexpr {
        if constexpr(requires { T::SPARSE_COMPACT_OWNER; })
            return T::SPARSE_COMPACT_OWNER && compact_group_grid;
        return false;
    }();
    if constexpr(compact_group_grid)
    {
        static_assert(T::ROUTE_COHORT_TILES % RouteTiles == 0);
        constexpr int cohort = T::ROUTE_COHORT_TILES / RouteTiles;
        const int blocks_per_cohort = cohort * parts;
        const int cohort_id = linear_block / blocks_per_cohort;
        const int within_cohort = linear_block % blocks_per_cohort;
        part = within_cohort / cohort;
        const int compact_group =
            cohort_id * cohort + within_cohort % cohort;

        if constexpr(sparse_compact_owner)
        {
            int lo = 0;
            int hi = kargs.route.num_experts;
            while(lo < hi)
            {
                const int mid = (lo + hi) / 2;
                const int first_tile =
                    kargs.route.expert_offsets[mid] / ROUTE_M;
                const int first_group = first_tile / RouteTiles + mid;
                if(first_group <= compact_group)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            expert_id = lo - 1;
            if(expert_id >= 0)
            {
                const int first_row =
                    kargs.route.expert_offsets[expert_id];
                expert_end_row =
                    kargs.route.expert_offsets[expert_id + 1];
                const int first_tile = first_row / ROUTE_M;
                const int expert_tiles =
                    (expert_end_row - first_row) / ROUTE_M;
                const int local_group =
                    compact_group - (first_tile / RouteTiles + expert_id);
                const int expert_groups =
                    (expert_tiles + RouteTiles - 1) / RouteTiles;
                if(local_group < 0 || local_group >= expert_groups)
                    expert_id = -1;
                else
                    route_tile = first_tile + local_group * RouteTiles;
            }
        }
        else
        {
            int group_prefix = 0;
            for(int expert = 0; expert < kargs.route.num_experts; ++expert)
            {
                const int first_row = kargs.route.expert_offsets[expert];
                const int end_row = kargs.route.expert_offsets[expert + 1];
                const int expert_tiles = (end_row - first_row) / ROUTE_M;
                const int expert_groups =
                    (expert_tiles + RouteTiles - 1) / RouteTiles;
                if(compact_group < group_prefix + expert_groups)
                {
                    expert_id = expert;
                    route_tile = first_row / ROUTE_M +
                                 (compact_group - group_prefix) * RouteTiles;
                    expert_end_row = end_row;
                    break;
                }
                group_prefix += expert_groups;
            }
        }
        if(expert_id < 0)
            return;
    }
    else if constexpr(T::ROUTE_COHORT_TILES > 0)
    {
        constexpr int cohort = T::ROUTE_COHORT_TILES;
        const int blocks_per_cohort = cohort * parts;
        const int cohort_id = linear_block / blocks_per_cohort;
        const int within_cohort = linear_block % blocks_per_cohort;
        part = within_cohort / cohort;
        route_tile = cohort_id * cohort + within_cohort % cohort;
    }
    else
    {
        part = static_cast<int>(blockIdx.y);
        route_tile = linear_block;
    }
    const int route_base = route_tile * ROUTE_M;
    const int valid_rows = kargs.route.num_valid_ids[0];
    if(route_base >= valid_rows)
        return;

    if constexpr(!compact_group_grid)
    {
        expert_id = kargs.route.sorted_expert_ids[route_tile];
        if(expert_id < 0 || expert_id >= kargs.route.num_experts)
            return;
        if constexpr(RouteTiles > 1)
        {
            if(kargs.route.expert_offsets == nullptr)
                return;
            const int expert_first_tile =
                kargs.route.expert_offsets[expert_id] / ROUTE_M;
            if((route_tile - expert_first_tile) % RouteTiles != 0)
                return;
            // A final group may contain fewer than RouteTiles sorter blocks.
            // Keep the shared W2/MFMA schedule uniform, but mask rows belonging
            // to the next expert in the gather and fused epilogue.
            expert_end_row = kargs.route.expert_offsets[expert_id + 1];
        }
    }

    const int tid = static_cast<int>(thread_id_x());
    const int lane_id = tid % get_warp_size();
    const int wave_id = __builtin_amdgcn_readfirstlane(tid / get_warp_size());
    const int wave_id_m = wave_id / T::T_N;
    const int wave_id_n = wave_id % T::T_N;
    const int col_base = part * BN;

    // GEMM operands die before the fused epilogue starts, so the FP32 dScore
    // scratch aliases the same LDS allocation.
    __shared__ __align__(16) char tile_storage[tile_storage_bytes];
    __shared__ int32_t
        smem_packed_route[T::PREDECODE_ROUTE_METADATA ? 1 : CTA_M];
    __shared__ int32_t
        smem_route_token[T::PREDECODE_ROUTE_METADATA ? CTA_M : 1];
    __shared__ int32_t
        smem_logical_route[T::PREDECODE_ROUTE_METADATA ? CTA_M : 1];
    __shared__ float
        smem_route_score[T::PREDECODE_ROUTE_METADATA ? CTA_M : 1];

    if(tid < CTA_M)
    {
        const int sorted_row = route_base + tid;
        const bool row_in_range =
            sorted_row < valid_rows && sorted_row < expert_end_row;
        const int32_t packed =
            row_in_range ? kargs.route.sorted_token_ids[sorted_row] : 0;
        if constexpr(T::PREDECODE_ROUTE_METADATA)
        {
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
                valid = row_in_range && token < kargs.route.token_num &&
                        slot < topk;
                logical_route = valid ? token * topk + slot : -1;
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
            smem_route_token[tid] =
                valid ? token : kargs.route.token_num;
            smem_logical_route[tid] = valid ? logical_route : -1;
            smem_route_score[tid] = score;
        }
        else
        {
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
    }
    __syncthreads();

    const D_A* d_out = reinterpret_cast<const D_A*>(kargs.d_out);
    const D_B* w2 = reinterpret_cast<const D_B*>(kargs.w2);
    const unsigned int d_out_bytes = static_cast<unsigned int>(
        static_cast<unsigned long long>(kargs.route.token_num) *
        static_cast<unsigned long long>(stride_do_t) * sizeof(D_A));
    auto g_a = make_gmem(d_out, d_out_bytes);

    const D_A* z = reinterpret_cast<const D_A*>(kargs.z);
    const unsigned int z_bytes = static_cast<unsigned int>(
        static_cast<unsigned long long>(kargs.route.sorted_capacity) *
        static_cast<unsigned long long>(stride_z_r) * sizeof(D_A));
    auto g_z = make_gmem(z, z_bytes);

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
    typename decltype(mma)::vtype_c v_c[RouteTiles];
    opus::static_for<RouteTiles>([&](auto tile) {
        clear(v_c[tile.value]);
    });

    constexpr int a_vectors = CTA_M * BK / T::VEC_A;
    static_assert(a_vectors > 0);

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

            auto issue_b = [&]() __attribute__((always_inline)) {
                if constexpr(split_b_n64)
                {
                    const int loader_tid = tid ^ ((tid >> 4) & 7);
                    const int local_k = loader_tid / 8;
                    const int local_n = (loader_tid % 8) * T::VEC_B;
                    opus::static_for<4>([&](auto slab) {
                        OPUS_LDS_ADDR D_B* b_slab_dst =
                            b_lds + slab.value * 64 * BK +
                            wave_id * opus::get_warp_size() * T::VEC_B;
                        const int b_global_offset =
                            (tile_k * BK + local_k) * stride_w2_d +
                            col_base + slab.value * 64 + local_n;
                        g_b.template _async_load<T::VEC_B>(
                            reinterpret_cast<OPUS_LDS_ADDR void*>(b_slab_dst),
                            b_global_offset * sizeof(D_B),
                            0,
                            opus::number<0>{},
                            opus::number<T::CACHECTL_B>{});
                    });
                }
                else
                {
                    constexpr int vectors_per_group =
                        T::SMEM_B_GROUP_DATA_BYTES /
                        (T::VEC_B * sizeof(D_B));
                    static_assert(vectors_per_group % opus::get_warp_size() ==
                                  0);
                    constexpr int b_loads_per_wave =
                        vectors_per_group / opus::get_warp_size();
                    constexpr int n_vectors = BN / T::VEC_B;
                    static_assert(opus::get_warp_size() % n_vectors == 0);
                    constexpr int source_rows_per_issue =
                        opus::get_warp_size() / n_vectors;
                    static_assert(source_rows_per_issue * b_loads_per_wave ==
                                  T::SMEM_B_GROUP_ROWS);
                    static_assert(T::SMEM_B_GROUPS == 4 * T::E_K);
                    constexpr int mfma_stage_groups =
                        T::W_K / T::SMEM_B_GROUP_ROWS;
                    constexpr int mfma_stage_bytes =
                        mfma_stage_groups * T::SMEM_B_GROUP_BYTES;
                    constexpr int mfma_stage_rows = T::W_K;
                    const int b_local_n =
                        (lane_id % n_vectors) * T::VEC_B;
                    OPUS_LDS_ADDR char* b_wave_dst =
                        reinterpret_cast<OPUS_LDS_ADDR char*>(b_lds) +
                        wave_id * T::SMEM_B_GROUP_BYTES;
                    opus::static_for<T::E_K>([&](auto ek) {
                        constexpr int stage_offset =
                            ek.value * mfma_stage_bytes;
                        opus::static_for<b_loads_per_wave>([&](auto load) {
                            constexpr int load_byte_offset =
                                load.value * opus::get_warp_size() * T::VEC_B *
                                sizeof(D_B);
                            const int b_local_k =
                                wave_id +
                                ((lane_id / n_vectors) +
                                 load.value * source_rows_per_issue) *
                                    T::T_N;
                            const int b_global_offset =
                                (tile_k * BK + b_local_k +
                                 ek.value * mfma_stage_rows) *
                                    stride_w2_d +
                                col_base + b_local_n;
                            g_b.template _async_load<T::VEC_B>(
                                reinterpret_cast<OPUS_LDS_ADDR void*>(
                                    b_wave_dst + stage_offset +
                                    load_byte_offset),
                                b_global_offset * sizeof(D_B),
                                0,
                                opus::number<0>{},
                                opus::number<T::CACHECTL_B>{});
                        });
                    });
                }
            };
            if constexpr(issue_b_first)
                issue_b();

            constexpr int a_loads_per_thread =
                (a_vectors + T::BLOCK_SIZE - 1) / T::BLOCK_SIZE;
            opus::static_for<a_loads_per_thread>([&](auto load) {
                const int a_vector = load.value * T::BLOCK_SIZE + tid;
                if(a_vector < a_vectors)
                {
                    int local_m;
                    int local_k;
                    if constexpr(swizzle_a_xor)
                    {
                        const int logical_vector =
                            a_vector ^
                            ((a_vector >> a_xor_shift) & a_xor_mask);
                        local_m = logical_vector >> 2;
                        local_k =
                            (logical_vector & 3) * T::VEC_A;
                    }
                    else
                    {
                        const int a_element = a_vector * T::VEC_A;
                        local_m = a_element / BK;
                        local_k = a_element % BK;
                    }
                    int token;
                    bool valid;
                    if constexpr(T::PREDECODE_ROUTE_METADATA)
                    {
                        token = smem_route_token[local_m];
                        valid = token < kargs.route.token_num;
                    }
                    else if constexpr(T::ROUTE_LAYOUT ==
                                      RouteLayout::CompactRouteMajor)
                    {
                        const int32_t packed = smem_packed_route[local_m];
                        const auto decoded =
                            decode_sorted_route<T::ROUTE_LAYOUT>(
                                kargs.route, packed, true);
                        token = decoded.token;
                        valid = decoded.valid;
                    }
                    else
                    {
                        const int32_t packed = smem_packed_route[local_m];
                        token = packed_token_id(packed);
                        const int slot = packed_topk_slot(packed);
                        valid = token < kargs.route.token_num && slot < topk;
                    }
                    const int source_base =
                        valid
                            ? static_cast<int32_t>(
                                  static_cast<int64_t>(token) * stride_do_t)
                            : static_cast<int32_t>(
                                  static_cast<int64_t>(
                                      kargs.route.token_num) *
                                  stride_do_t);
                    // buffer_load_*_lds supplies the lane-vector offset.
                    // Each load group owns one contiguous BLOCK_SIZE-vector
                    // region, so the destination remains wave-uniform even
                    // when RouteTiles makes A larger than the workgroup.
                    OPUS_LDS_ADDR D_A* a_wave_dst =
                        a_lds +
                        (load.value * T::BLOCK_SIZE +
                         wave_id * opus::get_warp_size()) *
                            T::VEC_A;
                    g_a.template _async_load<T::VEC_A>(
                        reinterpret_cast<OPUS_LDS_ADDR void*>(a_wave_dst),
                        (source_base + tile_k * BK + local_k) *
                            sizeof(D_A),
                        0,
                        opus::number<0>{},
                        opus::number<T::CACHECTL_A>{});
                }
            });

            if constexpr(!issue_b_first)
                issue_b();
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
            auto v_b = [&]() {
                if constexpr(split_b_n64)
                    return down_bwd_load_rb_tr_split_n64<T, decltype(mma)>(
                        s_b_current, lane_id, wave_id_n);
                else
                    return down_bwd_load_rb_tr<T, decltype(mma)>(
                        s_b_current, lane_id, wave_id_n);
            }();

            auto load_route_a = [&](auto route_group) {
                auto s_a = make_smem(reinterpret_cast<D_A*>(
                    tile_storage + buffer * gemm_buffer_bytes +
                    route_group.value * ROUTE_M * BK * sizeof(D_A)));
                if constexpr(swizzle_a_xor)
                    return down_bwd_load_ra_swizzled_xor<T, decltype(mma)>(
                        s_a, lane_id, route_group);
                else
                    return s_a.template load<T::VEC_A>(u_ra);
            };
            if constexpr(prefetch_a_tiles == 3 && RouteTiles >= 3)
            {
                auto v_a0 = load_route_a(opus::number<0>{});
                auto v_a1 = load_route_a(opus::number<1>{});
                auto v_a2 = load_route_a(opus::number<2>{});
                s_waitcnt_lgkmcnt(0_I);
                __builtin_amdgcn_s_setprio(1);
                v_c[0] = mma(v_a0, v_b, v_c[0]);
                v_c[1] = mma(v_a1, v_b, v_c[1]);
                v_c[2] = mma(v_a2, v_b, v_c[2]);
                opus::static_for<RouteTiles - 3>([&](auto route_group) {
                    constexpr int group = route_group.value + 3;
                    auto next_a = load_route_a(opus::number<group>{});
                    v_c[group] = mma(next_a, v_b, v_c[group]);
                });
            }
            else
            {
                s_waitcnt_lgkmcnt(0_I);
                __builtin_amdgcn_s_setprio(1);
                opus::static_for<RouteTiles>([&](auto route_group) {
                    auto v_a = load_route_a(route_group);
                    v_c[route_group.value] =
                        mma(v_a, v_b, v_c[route_group.value]);
                });
            }
            __builtin_amdgcn_s_setprio(0);

            if(has_next)
            {
                s_waitcnt_vmcnt(0_I);
                __syncthreads();
            }
        }
    }
    __syncthreads();

    auto s_z = make_smem(reinterpret_cast<D_A*>(tile_storage));
    auto s_ds =
        make_smem(reinterpret_cast<float*>(tile_storage + z_lds_total_bytes));
    OPUS_LDS_ADDR float* smem_ds =
        reinterpret_cast<OPUS_LDS_ADDR float*>(s_ds.ptr);

    auto stage_z = [&](int route_group, auto& s_z_tile) {
        if constexpr(stage_z_in_lds)
        {
            constexpr int z_async_vec = T::VEC_A;
            constexpr int row_pairs = ROUTE_M / 2;
            constexpr int pairs_per_wave = row_pairs / T::T_N;
            static_assert(T::T_N * opus::get_warp_size() == T::BLOCK_SIZE);
            static_assert(BN == 32 * z_async_vec);
            static_assert(row_pairs % T::T_N == 0);
            opus::static_for<2>([&](auto half) {
                opus::static_for<pairs_per_wave>([&](auto pair_group) {
                    const int local_pair =
                        pair_group.value * T::T_N + wave_id_n;
                    const int row_in_pair = lane_id / 32;
                    const int local_row = 2 * local_pair + row_in_pair;
                    const int local_m = route_group * ROUTE_M + local_row;
                    const int sorted_row = route_base + local_m;
                    const bool valid_row = sorted_row < valid_rows &&
                                           sorted_row < expert_end_row;
                    const int lds_offset =
                        local_pair * z_lds_pair_elems + half.value * 2 * BN;
                    const int source_row = valid_row
                                               ? sorted_row
                                               : kargs.route.sorted_capacity;
                    const int64_t global_offset =
                        static_cast<int64_t>(source_row) * stride_z_r +
                        half.value * inter_dim + col_base +
                        (lane_id % 32) * z_async_vec;
                    g_z.template async_load<z_async_vec>(
                        reinterpret_cast<void*>(
                            reinterpret_cast<OPUS_LDS_ADDR D_A*>(s_z_tile.ptr) +
                            lds_offset),
                        static_cast<int>(global_offset),
                        0,
                        opus::number<0>{},
                        opus::number<cachectl_z>{});
                });
            });
        }
    };

    auto store_epilogue = [&](auto& accum,
                              int route_group,
                              auto& s_z_tile) {
        static_assert(ROUTE_M == 32);
        static_assert(BN == T::T_N * T::W_N * T::E_N);
        static_assert(T::T_M == 1 && T::T_N == 4);
        static_assert(T::E_M == 1 && T::E_N >= 1 && T::VEC_C == 4);
        using u32x4 = uint32_t __attribute__((ext_vector_type(4)));

        // Preserve FP32 precision through the fused activation backward.
        // Each swap exchanges one four-column fragment with lane +/- 32,
        // leaving two groups of eight contiguous columns.  Use the native
        // two-in/two-out instruction instead of the Clang builtin: the
        // latter can coalesce its second result with the first when the
        // inputs come directly from an FP32 MFMA accumulator.
#pragma unroll
        for(int repeat = 0; repeat < T::E_N; ++repeat)
        {
#pragma unroll
            for(int group = 0; group < 2; ++group)
            {
#pragma unroll
                for(int i = 0; i < 4; ++i)
                {
                    const int lo = repeat * 16 + group * 8 + i;
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
        }

        const int local_m = route_group * ROUTE_M + lane_id % ROUTE_M;
        const int sorted_row = route_base + local_m;
        const bool row_in_range =
            sorted_row < valid_rows && sorted_row < expert_end_row;
        int token;
        int slot;
        int logical_route;
        bool valid;
        if constexpr(T::PREDECODE_ROUTE_METADATA)
        {
            token = smem_route_token[local_m];
            logical_route = smem_logical_route[local_m];
            valid = logical_route >= 0;
            slot = valid ? logical_route - token * topk : 0;
        }
        else if constexpr(T::ROUTE_LAYOUT == RouteLayout::CompactRouteMajor)
        {
            const int32_t packed = smem_packed_route[local_m];
            const auto decoded = decode_sorted_route<T::ROUTE_LAYOUT>(
                kargs.route, packed, row_in_range);
            token = decoded.token;
            slot = decoded.slot;
            logical_route = decoded.logical;
            valid = decoded.valid;
        }
        else
        {
            const int32_t packed = smem_packed_route[local_m];
            token = packed_token_id(packed);
            slot = packed_topk_slot(packed);
            valid = token < kargs.route.token_num && slot < topk;
            logical_route =
                static_cast<int>(logical_route_id(token, slot, topk));
        }
        float score;
        if constexpr(T::PREDECODE_ROUTE_METADATA)
            score = smem_route_score[local_m];
        else
        {
            score = 0.0f;
            if(valid)
            {
                if constexpr(T::ROUTE_LAYOUT ==
                             RouteLayout::CompactRouteMajor)
                    score = kargs.scores[logical_route];
                else
                    score = kargs.scores[
                        static_cast<int64_t>(token) * stride_score_t + slot];
            }
        }
        if constexpr(stage_z_in_lds && defer_z_lds_wait &&
                     !pipeline_z_route_groups)
        {
            // Z is deposited directly into LDS and does not extend any VGPR
            // live range.  Hide its VMEM latency under the accumulator lane
            // permutation and route/score decode above, synchronizing only
            // immediately before the first LDS Z read below.
            s_waitcnt_vmcnt(0_I);
            s_waitcnt_lgkmcnt(0_I);
            __syncthreads();
        }
        const down_bwd_f32x2 score2{score, score};
        const down_bwd_f32x2 one{1.0f, 1.0f};
        int64_t blocked_dz_row_base = 0;
        int64_t blocked_dz_up_delta = 0;
        if constexpr(blocked_dz_g2)
        {
            constexpr int route_tile = 32;
            constexpr int column_tile = 32;
            constexpr int row_group = 2;
            constexpr int vector = 8;
            const int row_in_tile = sorted_row & (route_tile - 1);
            const int tile_row = sorted_row - row_in_tile;
            blocked_dz_row_base =
                static_cast<int64_t>(tile_row) * stride_dz_r +
                (row_in_tile / row_group) * row_group * column_tile +
                (row_in_tile % row_group) * vector;
            // This BN256 family requires I to be divisible by 256.  Gate and
            // up therefore have identical within-tile coordinates and differ
            // only by a whole number of G2 column tiles.
            blocked_dz_up_delta =
                static_cast<int64_t>(inter_dim) * route_tile;
        }
        float ds_wave_partial = 0.0f;
#pragma unroll
        for(int repeat = 0; repeat < T::E_N; ++repeat)
        {
            const int local_n0 =
                repeat * T::T_N * T::W_N + wave_id_n * T::W_N +
                (lane_id / 32) * 8;

            // Keep all four 128-bit Z loads in flight, but retain their BF16
            // payload in packed dwords.  Expanding only the group currently
            // consumed by the activation epilogue halves the prefetch live
            // range and avoids private-segment spills without serializing the
            // global loads.
            down_bwd_u32x4 z_gate_prefetch[2];
            down_bwd_u32x4 z_up_prefetch[2];

            if(valid)
            {
#pragma unroll
                for(int group = 0; group < 2; ++group)
                {
                    const int col = col_base + local_n0 + group * 16;
                    if constexpr(stage_z_in_lds)
                    {
                        const int z_local_row = local_m % ROUTE_M;
                        const int pair_base =
                            (z_local_row / 2) * z_lds_pair_elems;
                        const int row_in_pair = z_local_row % 2;
                        z_gate_prefetch[group] = __builtin_bit_cast(
                            down_bwd_u32x4,
                            s_z_tile.template load<8>(pair_base +
                                                       row_in_pair * BN +
                                                       local_n0 + group * 16));
                        z_up_prefetch[group] = __builtin_bit_cast(
                            down_bwd_u32x4,
                            s_z_tile.template load<8>(pair_base + 2 * BN +
                                                       row_in_pair * BN +
                                                       local_n0 + group * 16));
                    }
                    else
                    {
                        const int64_t z_base =
                            static_cast<int64_t>(sorted_row) * stride_z_r + col;
                        z_gate_prefetch[group] =
                            down_bwd_load_bf16x8_packed(kargs.z + z_base);
                        z_up_prefetch[group] = down_bwd_load_bf16x8_packed(
                            kargs.z + z_base + inter_dim);
                    }
                }
            }

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
                    const down_bwd_f32x8 z_gate =
                        down_bwd_unpack_bf16x8(z_gate_prefetch[group]);
                    const down_bwd_f32x8 z_up =
                        down_bwd_unpack_bf16x8(z_up_prefetch[group]);
                    down_bwd_f32x8 sigmoid_values;
                    if constexpr(batch_sigmoid)
                    {
#pragma unroll
                        for(int elem = 0; elem < 8; ++elem)
                            sigmoid_values[elem] =
                                down_bwd_sigmoid(z_gate[elem]);
                    }
#pragma unroll
                    for(int pair = 0; pair < 4; ++pair)
                    {
                        const int elem = pair * 2;
                        const int acc_idx =
                            repeat * 16 + group * 8 + elem;
                        const down_bwd_f32x2 z_gate_pair{
                            z_gate[elem], z_gate[elem + 1]};
                        const down_bwd_f32x2 z_up_pair{
                            z_up[elem], z_up[elem + 1]};
                        down_bwd_f32x2 sigmoid;
                        if constexpr(batch_sigmoid)
                            sigmoid = down_bwd_f32x2{
                                sigmoid_values[elem],
                                sigmoid_values[elem + 1]};
                        else
                            sigmoid = down_bwd_f32x2{
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
                        d_gate_store[pair] = down_bwd_cvt_pk_bf16_f32(
                            d_gate[0], d_gate[1]);
                        d_up_store[pair] = down_bwd_cvt_pk_bf16_f32(
                            d_up[0], d_up[1]);
                        if constexpr(write_a_scaled)
                        {
                            const down_bwd_f32x2 scaled = score2 * activation;
                            scaled_store[pair] = down_bwd_cvt_pk_bf16_f32(
                                scaled[0], scaled[1]);
                        }
                        ds_partial += g[0] * activation[0] +
                                      g[1] * activation[1];
                    }
                }

                if(row_in_range && col + 8 <= inter_dim)
                {
                    if constexpr(blocked_dz_g2)
                    {
                        constexpr int column_tile = 32;
                        constexpr int row_group = 2;
                        constexpr int vector = 8;
                        const int col_in_tile = col & (column_tile - 1);
                        const int64_t gate_base =
                            blocked_dz_row_base +
                            static_cast<int64_t>(col - col_in_tile) *
                                column_tile +
                            (col_in_tile / vector) * row_group * vector;
                        const int64_t up_base =
                            gate_base + blocked_dz_up_delta;
                        *reinterpret_cast<u32x4*>(kargs.d_z + gate_base) =
                            d_gate_store;
                        *reinterpret_cast<u32x4*>(kargs.d_z + up_base) =
                            d_up_store;
                    }
                    else
                    {
                        const int64_t dz_base =
                            static_cast<int64_t>(sorted_row) * stride_dz_r + col;
                        *reinterpret_cast<u32x4*>(kargs.d_z + dz_base) =
                            d_gate_store;
                        *reinterpret_cast<u32x4*>(
                            kargs.d_z + dz_base + inter_dim) = d_up_store;
                    }
                    if constexpr(write_a_scaled)
                    {
                        const int64_t a_scaled_base =
                            static_cast<int64_t>(sorted_row) *
                                stride_a_scaled_r +
                            col;
                        *reinterpret_cast<u32x4*>(
                            kargs.a_scaled + a_scaled_base) = scaled_store;
                    }
                }
                ds_wave_partial += ds_partial;
            }
        }
        ds_wave_partial += __shfl_xor(ds_wave_partial, 32, 64);
        if(lane_id < 32)
            smem_ds[wave_id_n * CTA_M + local_m] = ds_wave_partial;
    };

    if constexpr(pipeline_z_route_groups)
    {
        auto s_z_next = make_smem(
            reinterpret_cast<D_A*>(tile_storage + z_lds_bytes));
        stage_z(0, s_z);
        s_waitcnt_vmcnt(0_I);
        s_waitcnt_lgkmcnt(0_I);
        __syncthreads();
        stage_z(1, s_z_next);

        opus::static_for<RouteTiles>([&](auto route_group) {
            if constexpr(route_group.value % 2 == 0)
                store_epilogue(
                    v_c[route_group.value], route_group.value, s_z);
            else
                store_epilogue(
                    v_c[route_group.value], route_group.value, s_z_next);

            if constexpr(route_group.value + 1 < RouteTiles)
            {
                s_waitcnt_vmcnt(0_I);
                s_waitcnt_lgkmcnt(0_I);
                __syncthreads();
                if constexpr(route_group.value + 2 < RouteTiles)
                {
                    if constexpr(route_group.value % 2 == 0)
                        stage_z(route_group.value + 2, s_z);
                    else
                        stage_z(route_group.value + 2, s_z_next);
                }
            }
        });
    }
    else
    {
        opus::static_for<RouteTiles>([&](auto route_group) {
            if constexpr(stage_z_in_lds)
            {
                if constexpr(route_group.value > 0)
                    __syncthreads();
                stage_z(route_group.value, s_z);
                if constexpr(!defer_z_lds_wait)
                {
                    s_waitcnt_vmcnt(0_I);
                    s_waitcnt_lgkmcnt(0_I);
                    __syncthreads();
                }
            }
            store_epilogue(v_c[route_group.value], route_group.value, s_z);
        });
    }
    __syncthreads();

    static_assert(T::T_N == 4);
    if(tid < CTA_M)
    {
        float partial = smem_ds[tid];
#pragma unroll
        for(int wave = 1; wave < T::T_N; ++wave)
            partial += smem_ds[wave * CTA_M + tid];
        if constexpr(T::PREDECODE_ROUTE_METADATA)
        {
            const int logical_route = smem_logical_route[tid];
            if(logical_route >= 0)
            {
                if(kargs.d_scores_parts == 1)
                    kargs.d_scores[logical_route] = partial;
                else
                    kargs.d_scores_workspace[
                        static_cast<int64_t>(logical_route) *
                            stride_ds_workspace_r +
                        part] = partial;
            }
        }
        else if constexpr(T::ROUTE_LAYOUT ==
                          RouteLayout::CompactRouteMajor)
        {
            const int32_t packed = smem_packed_route[tid];
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
            const int32_t packed = smem_packed_route[tid];
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
         int FixedTopK = 0,
         int RouteTiles = 1>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::MIN_BLOCKS_PER_CU)
void down_bwd_kernel_gfx950(DownBwdKargs kargs)
{
    down_bwd_process_tile_gfx950<
        Traits, FixedD, FixedI, FixedTopK, RouteTiles>(kargs);
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
    dim3 grid;
    constexpr int route_tiles = T::ROUTE_M_TILES;
    if constexpr(T::COMPACT_ROUTE_GROUP_GRID)
    {
        if(kargs.route.expert_offsets != nullptr)
        {
            static_assert(T::ROUTE_COHORT_TILES % route_tiles == 0);
            constexpr int cohort = T::ROUTE_COHORT_TILES / route_tiles;
            const int group_capacity =
                (kargs.route.sorted_block_capacity + route_tiles - 1) /
                    route_tiles +
                kargs.route.num_experts;
            const int padded_groups =
                ((group_capacity + cohort - 1) / cohort) * cohort;
            grid = dim3(static_cast<unsigned int>(padded_groups) *
                        static_cast<unsigned int>(expected_parts));
        }
        else
        {
            constexpr int cohort = T::ROUTE_COHORT_TILES;
            const int padded_route_tiles =
                ((kargs.route.sorted_block_capacity + cohort - 1) / cohort) *
                cohort;
            grid = dim3(static_cast<unsigned int>(padded_route_tiles) *
                        static_cast<unsigned int>(expected_parts));
        }
    }
    else if constexpr(T::ROUTE_COHORT_TILES > 0)
    {
        constexpr int cohort = T::ROUTE_COHORT_TILES;
        const int padded_route_tiles =
            ((kargs.route.sorted_block_capacity + cohort - 1) / cohort) *
            cohort;
        grid = dim3(static_cast<unsigned int>(padded_route_tiles) *
                    static_cast<unsigned int>(expected_parts));
    }
    else
    {
        grid = dim3(
            static_cast<unsigned int>(kargs.route.sorted_block_capacity),
            static_cast<unsigned int>(expected_parts));
    }
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
    bool launched_target_shape = false;
    if constexpr(target_i % T::B_N == 0)
    {
        if(use_target_shape)
        {
            if(kargs.route.expert_offsets != nullptr)
                hipLaunchKernelGGL(
                    (down_bwd_kernel_gfx950<
                        T, target_d, target_i, target_topk, route_tiles>),
                    grid,
                    block,
                    0,
                    stream,
                    kargs);
            else
                hipLaunchKernelGGL(
                    (down_bwd_kernel_gfx950<
                        T, target_d, target_i, target_topk, 1>),
                    grid,
                    block,
                    0,
                    stream,
                    kargs);
            launched_target_shape = true;
        }
    }
    if(!launched_target_shape)
    {
        if constexpr(T::ROUTE_LAYOUT != RouteLayout::CompactRouteMajor)
        {
            if(kargs.route.expert_offsets != nullptr)
                hipLaunchKernelGGL(
                    (down_bwd_kernel_gfx950<T, 0, 0, 0, route_tiles>),
                    grid,
                    block,
                    0,
                    stream,
                    kargs);
            else
                hipLaunchKernelGGL(
                    (down_bwd_kernel_gfx950<T>),
                    grid,
                    block,
                    0,
                    stream,
                    kargs);
        }
        else
            hipLaunchKernelGGL((down_bwd_kernel_gfx950<T>),
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
