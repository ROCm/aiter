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

inline __device__ uint32_t weight_bwd_cvt_pk_bf16_f32(float lo, float hi)
{
    uint32_t packed;
    asm volatile("v_cvt_pk_bf16_f32 %0, %1, %2"
                 : "=v"(packed)
                 : "v"(lo), "v"(hi));
    return packed;
}

template<RouteLayout Layout, bool GatherToken>
inline __device__ int weight_bwd_source_row(const RouteMetadata& route,
                                             int sorted_row,
                                             bool in_range)
{
    if(!in_range)
        return GatherToken ? route.token_num : route.sorted_capacity;
    if constexpr(Layout == RouteLayout::CompactRouteMajor)
    {
        const int32_t encoded = route.sorted_token_ids[sorted_row];
        const auto decoded = decode_sorted_route<Layout>(
            route, encoded, in_range);
        if(!decoded.valid)
            return GatherToken ? route.token_num : route.sorted_capacity;
        return GatherToken ? decoded.token : sorted_row;
    }
    else
    {
        const int32_t packed = route.sorted_token_ids[sorted_row];
        const int token = packed_token_id(packed);
        const int slot = packed_topk_slot(packed);
        if(token >= route.token_num || slot >= route.topk)
            return GatherToken ? route.token_num : route.sorted_capacity;
        return GatherToken ? token : sorted_row;
    }
}

#endif

struct WeightBwdKernelArgs
{
    const hip_bfloat16* __restrict__ a;
    const hip_bfloat16* __restrict__ b;
    RouteMetadata route;
    hip_bfloat16* __restrict__ c;
    int output_m;
    int output_n;
    int64_t stride_a_row;
    int64_t stride_b_row;
    int64_t stride_c_expert;
    int64_t stride_c_m;
};

#ifdef __HIP_DEVICE_COMPILE__

// Repack each native 32x32 MFMA result from four-column lane fragments into
// eight contiguous BF16 values.  This is the same permlane32 layout transform
// used by the production Opus attention epilogue and Triton's gfx950 lowering.
template<typename T, typename Accum>
inline __device__ void weight_bwd_store_wide(Accum& accum,
                                              const WeightBwdKernelArgs& kargs,
                                              int expert,
                                              int output_m_base,
                                              int output_n_base,
                                              int wave_id_m,
                                              int wave_id_n,
                                              int lane_id)
{
    static_assert(T::VEC_C == 4);
    static_assert(T::W_M == 32 && T::W_N == 32);
    using u32x4 = uint32_t __attribute__((ext_vector_type(4)));

    opus::static_for<T::E_M>([&](auto em) {
        opus::static_for<T::E_N>([&](auto en) {
            opus::static_for<2>([&](auto sub) {
                constexpr int native_tile = em.value * T::E_N + en.value;
                constexpr int accum_base = native_tile * 16 + sub.value * 8;
                uint32_t packed[4];
#pragma unroll
                for(int i = 0; i < 4; ++i)
                    packed[i] = weight_bwd_cvt_pk_bf16_f32(
                        static_cast<float>(accum[accum_base + 2 * i]),
                        static_cast<float>(accum[accum_base + 2 * i + 1]));
                auto swap02 = __builtin_amdgcn_permlane32_swap(
                    packed[0], packed[2], false, true);
                auto swap13 = __builtin_amdgcn_permlane32_swap(
                    packed[1], packed[3], false, true);
                packed[0] = swap02[0];
                packed[2] = swap02[1];
                packed[1] = swap13[0];
                packed[3] = swap13[1];

                const int output_m =
                    output_m_base +
                    (em.value * T::T_M + wave_id_m) * T::W_M +
                    lane_id % T::W_M;
                const int output_n =
                    output_n_base +
                    (en.value * T::T_N + wave_id_n) * T::W_N +
                    (lane_id / T::W_M) * 8 + sub.value * 16;
                if(output_m < kargs.output_m && output_n + 8 <= kargs.output_n)
                {
                    const int64_t c_offset =
                        static_cast<int64_t>(expert) * kargs.stride_c_expert +
                        static_cast<int64_t>(output_m) * kargs.stride_c_m +
                        output_n;
                    *reinterpret_cast<u32x4*>(kargs.c + c_offset) =
                        u32x4{packed[0], packed[1], packed[2], packed[3]};
                }
            });
        });
    });
}

#endif


// Compact gfx950 K32 dW1 path.  The 64x32 and 32x128 logical operands occupy
// 4 KiB and 8 KiB respectively.  Keep both swizzled tiles resident so their
// stores and transpose reads share one pair of barriers per reduction tile.
// The first shared encoding is
// {vec=8, perPhase=2, maxPhase=8}; the second is
// {vec=8, perPhase=1, maxPhase=16}.
template<typename Traits>
__device__ __forceinline__ void
weight_bwd_k32_process_tile_gfx950(WeightBwdKernelArgs kargs)
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
    constexpr int VEC = T::VEC_A;
    static_assert(BM == 64 && BN == 128 && BK == 32);
    static_assert(T::BLOCK_SIZE == 256 && VEC == 8);
    static_assert(T::VEC_B == VEC && T::VEC_TR_B == 4);
    const int expert = static_cast<int>(blockIdx.x);
    const int output_m_base = static_cast<int>(blockIdx.y) * BM;
    const int output_n_base = static_cast<int>(blockIdx.z) * BN;
    const int route_start = kargs.route.expert_offsets[expert];
    const int route_end = kargs.route.expert_offsets[expert + 1];
    const int route_count = route_end - route_start;
    const int tid = static_cast<int>(thread_id_x());
    const int lane_id = tid % get_warp_size();
    const int wave_id = __builtin_amdgcn_readfirstlane(tid / get_warp_size());
    const int wave_id_m = wave_id / T::T_N;
    const int wave_id_n = wave_id % T::T_N;

    constexpr int smem_a_bytes = BM * BK * sizeof(D_A);
    constexpr int smem_b_bytes = BN * BK * sizeof(D_B);
    __shared__ __align__(16) char tile_storage[smem_a_bytes + smem_b_bytes];
    auto s_a = make_smem(reinterpret_cast<D_A*>(tile_storage));
    auto s_b = make_smem(
        reinterpret_cast<D_B*>(tile_storage + smem_a_bytes));

    const D_A* a = reinterpret_cast<const D_A*>(kargs.a);
    const D_B* b = reinterpret_cast<const D_B*>(kargs.b);
    const int a_rows = kargs.route.sorted_capacity;
    const int b_rows = kargs.route.token_num;
    const unsigned int a_bytes = static_cast<unsigned int>(
        static_cast<unsigned long long>(a_rows) *
        static_cast<unsigned long long>(kargs.stride_a_row) * sizeof(D_A));
    const unsigned int b_bytes = static_cast<unsigned int>(
        static_cast<unsigned long long>(b_rows) *
        static_cast<unsigned long long>(kargs.stride_b_row) * sizeof(D_B));
    auto g_a = make_gmem(a, a_bytes);
    auto g_b = make_gmem(b, b_bytes);

    auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
        seq<T::E_M, T::E_N, T::E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});
    typename decltype(mma)::vtype_c v_c;
    clear(v_c);

    const int a_local_k = tid / 8;
    const int a_local_m = (tid % 8) * VEC;
    const int b_local_k0 = tid / 16;
    const int b_local_k1 = b_local_k0 + 16;
    const int b_local_n = (tid % 16) * VEC;
    const int a_store_addr = (tid << 4) ^ (tid & 0x70);
    const int b_store_addr = (tid << 4) ^ (tid & 0xf0);

    const int lane_mix = tid & 0x2c;
    const int a_read_base =
        (lane_mix << 5) |
        (((tid << 1) & 0x70) ^ ((tid << 3) & 0x18));
    const int b_read_base =
        ((lane_mix << 2) ^ ((tid << 1) & 0x20) ^
         ((lane_mix << 6) | ((tid << 3) & 0x18))) ^
        (tid & 0xc0);
    // Expert offsets are padded to the sorter block size, which is BK=32 for
    // this kernel, so every reduction tile has exactly 32 addressable rows.
    const int loops = route_count / BK;

    for(int tile_k = 0; tile_k < loops; ++tile_k)
    {
        const int a_sorted_row = route_start + tile_k * BK + a_local_k;
        const int b_sorted_row0 = route_start + tile_k * BK + b_local_k0;
        const int b_sorted_row1 = route_start + tile_k * BK + b_local_k1;
        int token_a;
        int token_b0;
        int token_b1;
        if constexpr(T::ROUTE_LAYOUT == RouteLayout::CompactRouteMajor)
        {
            token_a = weight_bwd_source_row<T::ROUTE_LAYOUT, true>(
                kargs.route, a_sorted_row, true);
            token_b0 = weight_bwd_source_row<T::ROUTE_LAYOUT, true>(
                kargs.route, b_sorted_row0, true);
            token_b1 = weight_bwd_source_row<T::ROUTE_LAYOUT, true>(
                kargs.route, b_sorted_row1, true);
        }
        else
        {
            token_a =
                kargs.route.sorted_token_ids[a_sorted_row] & kPackedTokenMask;
            token_b0 = kargs.route.sorted_token_ids[b_sorted_row0] &
                       kPackedTokenMask;
            token_b1 = kargs.route.sorted_token_ids[b_sorted_row1] &
                       kPackedTokenMask;
        }
        const int source_a = token_a < kargs.route.token_num
                                 ? a_sorted_row
                                 : kargs.route.sorted_capacity;
        const int source_b0 = token_b0 < kargs.route.token_num
                                  ? token_b0
                                  : kargs.route.token_num;
        const int source_b1 = token_b1 < kargs.route.token_num
                                  ? token_b1
                                  : kargs.route.token_num;

        auto a0 = g_a.template load<VEC>(
            source_a * kargs.stride_a_row + output_m_base + a_local_m,
            0,
            opus::number<T::CACHECTL_A>{});
        auto b0 = g_b.template load<VEC>(
            source_b0 * kargs.stride_b_row + output_n_base + b_local_n,
            0,
            opus::number<T::CACHECTL_B>{});
        auto b1 = g_b.template load<VEC>(
            source_b1 * kargs.stride_b_row + output_n_base + b_local_n,
            0,
            opus::number<T::CACHECTL_B>{});

        __builtin_amdgcn_s_barrier();
        s_a.template _store<VEC>(a0, a_store_addr);
        s_b.template _store<VEC>(b0, b_store_addr);
        s_b.template _store<VEC>(b1, b_store_addr + 4096);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();

        typename decltype(mma)::vtype_a v_a;
        opus::static_for<T::E_M>([&](auto em) {
            opus::static_for<T::E_K>([&](auto ek) {
                constexpr int byte_offset = ek.value * 2048;
                const int base = a_read_base ^ (em.value * 0x40);
                auto lo = s_a.template _tr_load<T::VEC_TR_B, byte_offset>(
                    base);
                auto hi = s_a.template _tr_load<T::VEC_TR_B, byte_offset>(
                    base ^ 0x220);
                constexpr int fragment = em.value * T::E_K + ek.value;
#pragma unroll
                for(int j = 0; j < T::VEC_TR_B; ++j)
                {
                    v_a[fragment * 2 * T::VEC_TR_B + j] = lo[j];
                    v_a[fragment * 2 * T::VEC_TR_B + T::VEC_TR_B + j] =
                        hi[j];
                }
            });
        });
        typename decltype(mma)::vtype_b v_b;
        opus::static_for<T::E_K>([&](auto ek) {
            constexpr int byte_offset = ek.value * 4096;
            auto lo = s_b.template _tr_load<T::VEC_TR_B, byte_offset>(
                b_read_base);
            auto hi = s_b.template _tr_load<T::VEC_TR_B, byte_offset>(
                b_read_base ^ 0x440);
#pragma unroll
            for(int j = 0; j < T::VEC_TR_B; ++j)
            {
                v_b[ek.value * 2 * T::VEC_TR_B + j] = lo[j];
                v_b[ek.value * 2 * T::VEC_TR_B + T::VEC_TR_B + j] = hi[j];
            }
        });
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c = mma(v_a, v_b, v_c);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_sched_barrier(0);
    }
    __syncthreads();

    weight_bwd_store_wide<T>(v_c,
                             kargs,
                             expert,
                             output_m_base,
                             output_n_base,
                             wave_id_m,
                             wave_id_n,
                             lane_id);
#else
    (void)kargs;
#endif
#else
    (void)kargs;
#endif
}

template<typename Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::MIN_BLOCKS_PER_CU)
void weight_bwd_swizzled_k32_kernel_gfx950(WeightBwdKernelArgs kargs)
{
    weight_bwd_k32_process_tile_gfx950<Traits>(kargs);
}

// Compact gfx950 K64 path.  Both logical operands arrive route-major, so each
// wave stores contiguous 8-BF16 vectors into the same swizzled 64x64 LDS tile
// and ds_read_b64_tr_b16 materializes the MFMA fragments.  Reusing one 8 KiB
// tile avoids both the scalar transpose stores and the padded double buffer.
// The swizzle is the Opus form of {vec=8, perPhase=2, maxPhase=8}.
template<typename Traits>
__device__ __forceinline__ void
weight_bwd_k64_process_tile_gfx950(WeightBwdKernelArgs kargs)
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
    constexpr int VEC = T::VEC_A;
    static_assert(BM == 64 && BN == 64 && BK == 64);
    static_assert(T::BLOCK_SIZE == 256 && VEC == 8);
    static_assert(T::VEC_B == VEC && T::VEC_TR_B == 4);
    const int expert = static_cast<int>(blockIdx.x);
    const int output_m_base = static_cast<int>(blockIdx.y) * BM;
    const int output_n_base = static_cast<int>(blockIdx.z) * BN;
    const int route_start = kargs.route.expert_offsets[expert];
    const int route_end = kargs.route.expert_offsets[expert + 1];
    const int route_count = route_end - route_start;
    const int tid = static_cast<int>(thread_id_x());
    const int lane_id = tid % get_warp_size();
    const int wave_id = __builtin_amdgcn_readfirstlane(tid / get_warp_size());
    const int wave_id_m = wave_id / T::T_N;
    const int wave_id_n = wave_id % T::T_N;

    __shared__ __align__(16) char tile_storage[BK * BN * sizeof(D_A)];
    auto s_tile = make_smem(reinterpret_cast<D_A*>(tile_storage));

    const D_A* a = reinterpret_cast<const D_A*>(kargs.a);
    const D_B* b = reinterpret_cast<const D_B*>(kargs.b);
    const int a_rows = kargs.route.token_num;
    const int b_rows = kargs.route.sorted_capacity;
    const unsigned int a_bytes = static_cast<unsigned int>(
        static_cast<unsigned long long>(a_rows) *
        static_cast<unsigned long long>(kargs.stride_a_row) * sizeof(D_A));
    const unsigned int b_bytes = static_cast<unsigned int>(
        static_cast<unsigned long long>(b_rows) *
        static_cast<unsigned long long>(kargs.stride_b_row) * sizeof(D_B));
    auto g_a = make_gmem(a, a_bytes);
    auto g_b = make_gmem(b, b_bytes);

    auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
        seq<T::E_M, T::E_N, T::E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});
    typename decltype(mma)::vtype_c v_c;
    clear(v_c);

    // Eight lanes cover one K row; every lane owns eight contiguous columns.
    const int local_k0 = tid / 8;
    const int local_k1 = local_k0 + 32;
    const int local_vec = (tid % 8) * VEC;
    // The two logical shared encodings have the same vector-store address.
    const int store_addr_bytes = (tid << 4) ^ (tid & 0x70);

    // Operand-specific transpose-read bases emitted by the two swizzled
    // shared encodings.  Wave bits select the 32-row M/N subtile.
    const int common_hi = (tid << 5) & 0x580;
    const int common_lo = (tid << 3) & 0x18;
    const int a_read_base =
        (((tid << 1) & 0x70) ^ ((tid & 0x80) >> 1)) ^ common_hi ^ common_lo;
    const int b_read_base =
        (common_hi | (((tid << 1) & 0x70) ^ common_lo)) ^ (tid & 0x40);
    const int full_loops = route_count / BK;

    // Expert offsets are padded to the sorting block (32 rows), whereas this
    // kernel reduces 64 rows at a time.  Keep the common full-tile path free
    // of range and top-k-slot checks; a possible final 32-row tile retains the
    // general decoder.  Padded rows inside a full tile are still rejected by
    // the 24-bit token sentinel written by the sorter.
    auto run_tile = [&](int tile_k, auto full_tile_tag)
        __attribute__((always_inline)) {
        constexpr bool FullTile = decltype(full_tile_tag)::value;
        const int sorted_row0 = route_start + tile_k * BK + local_k0;
        const int sorted_row1 = route_start + tile_k * BK + local_k1;
        int gathered0;
        int gathered1;
        int sorted0;
        int sorted1;
        if constexpr(FullTile)
        {
            if constexpr(T::ROUTE_LAYOUT == RouteLayout::CompactRouteMajor)
            {
                gathered0 = weight_bwd_source_row<T::ROUTE_LAYOUT, true>(
                    kargs.route, sorted_row0, true);
                gathered1 = weight_bwd_source_row<T::ROUTE_LAYOUT, true>(
                    kargs.route, sorted_row1, true);
            }
            else
            {
                gathered0 = packed_token_id(
                    kargs.route.sorted_token_ids[sorted_row0]);
                gathered1 = packed_token_id(
                    kargs.route.sorted_token_ids[sorted_row1]);
            }
            const bool valid0 = gathered0 < kargs.route.token_num;
            const bool valid1 = gathered1 < kargs.route.token_num;
            sorted0 = valid0 ? sorted_row0 : kargs.route.sorted_capacity;
            sorted1 = valid1 ? sorted_row1 : kargs.route.sorted_capacity;
            gathered0 = valid0 ? gathered0 : kargs.route.token_num;
            gathered1 = valid1 ? gathered1 : kargs.route.token_num;
        }
        else
        {
            const bool in_range0 = sorted_row0 < route_end;
            const bool in_range1 = sorted_row1 < route_end;
            gathered0 = weight_bwd_source_row<T::ROUTE_LAYOUT, true>(
                kargs.route, sorted_row0, in_range0);
            gathered1 = weight_bwd_source_row<T::ROUTE_LAYOUT, true>(
                kargs.route, sorted_row1, in_range1);
            sorted0 = gathered0 < kargs.route.token_num
                          ? sorted_row0
                          : kargs.route.sorted_capacity;
            sorted1 = gathered1 < kargs.route.token_num
                          ? sorted_row1
                          : kargs.route.sorted_capacity;
        }
        const int source_a0 = gathered0;
        const int source_a1 = gathered1;
        const int source_b0 = sorted0;
        const int source_b1 = sorted1;

        auto a0 = g_a.template load<VEC>(
            source_a0 * kargs.stride_a_row + output_m_base + local_vec,
            0,
            opus::number<T::CACHECTL_A>{});
        auto a1 = g_a.template load<VEC>(
            source_a1 * kargs.stride_a_row + output_m_base + local_vec,
            0,
            opus::number<T::CACHECTL_A>{});
        auto b0 = g_b.template load<VEC>(
            source_b0 * kargs.stride_b_row + output_n_base + local_vec,
            0,
            opus::number<T::CACHECTL_B>{});
        auto b1 = g_b.template load<VEC>(
            source_b1 * kargs.stride_b_row + output_n_base + local_vec,
            0,
            opus::number<T::CACHECTL_B>{});

        // Close the prior tile's shared-memory lifetime while allowing its
        // MFMA to overlap this tile's global loads.
        __builtin_amdgcn_s_barrier();
        s_tile.template _store<VEC>(a0, store_addr_bytes);
        s_tile.template _store<VEC>(a1, store_addr_bytes + 4096);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();

        typename decltype(mma)::vtype_a v_a;
        opus::static_for<T::E_K>([&](auto ek) {
            constexpr int byte_offset = ek.value * 2048;
            auto lo = s_tile.template _tr_load<T::VEC_TR_B, byte_offset>(
                a_read_base);
            auto hi = s_tile.template _tr_load<T::VEC_TR_B, byte_offset>(
                a_read_base ^ 0x220);
#pragma unroll
            for(int j = 0; j < T::VEC_TR_B; ++j)
            {
                v_a[ek.value * 2 * T::VEC_TR_B + j] = lo[j];
                v_a[ek.value * 2 * T::VEC_TR_B + T::VEC_TR_B + j] = hi[j];
            }
        });
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();

        s_tile.template _store<VEC>(b0, store_addr_bytes);
        s_tile.template _store<VEC>(b1, store_addr_bytes + 4096);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();

        typename decltype(mma)::vtype_b v_b;
        opus::static_for<T::E_K>([&](auto ek) {
            constexpr int byte_offset = ek.value * 2048;
            auto lo = s_tile.template _tr_load<T::VEC_TR_B, byte_offset>(
                b_read_base);
            auto hi = s_tile.template _tr_load<T::VEC_TR_B, byte_offset>(
                b_read_base ^ 0x220);
#pragma unroll
            for(int j = 0; j < T::VEC_TR_B; ++j)
            {
                v_b[ek.value * 2 * T::VEC_TR_B + j] = lo[j];
                v_b[ek.value * 2 * T::VEC_TR_B + T::VEC_TR_B + j] = hi[j];
            }
        });
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c = mma(v_a, v_b, v_c);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_sched_barrier(0);
    };

    for(int tile_k = 0; tile_k < full_loops; ++tile_k)
        run_tile(tile_k, opus::number<1>{});
    if(full_loops * BK < route_count)
        run_tile(full_loops, opus::number<0>{});
    __syncthreads();

    weight_bwd_store_wide<T>(v_c,
                             kargs,
                             expert,
                             output_m_base,
                             output_n_base,
                             wave_id_m,
                             wave_id_n,
                             lane_id);
#else
    (void)kargs;
#endif
#else
    (void)kargs;
#endif
}

template<typename Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::MIN_BLOCKS_PER_CU)
void weight_bwd_swizzled_kernel_gfx950(WeightBwdKernelArgs kargs)
{
    weight_bwd_k64_process_tile_gfx950<Traits>(kargs);
}

template<typename Traits>
inline void dw1_launch_gfx950(const Dw1Kargs& kargs, hipStream_t stream)
{
    using T = opus::remove_cvref_t<Traits>;
    static_assert(T::B_K == 32, "dw1 registers the K32 kernel");
    AITER_CHECK(kargs.split_k == 1,
                "dw1: first Opus instance requires split_k=1");
    AITER_CHECK(2 * kargs.inter_dim % T::B_M == 0 &&
                    kargs.model_dim % T::B_N == 0,
                "dw1: first instance requires 2I%32==0 and D%128==0");
    WeightBwdKernelArgs args{};
    args.a = kargs.d_z;
    args.b = kargs.x;
    args.route = kargs.route;
    args.c = kargs.d_w1;
    args.output_m = 2 * kargs.inter_dim;
    args.output_n = kargs.model_dim;
    args.stride_a_row = kargs.stride_dz_r;
    args.stride_b_row = kargs.stride_x_t;
    args.stride_c_expert = kargs.stride_dw1_e;
    args.stride_c_m = kargs.stride_dw1_i;
    const dim3 grid(
        static_cast<unsigned int>(kargs.route.num_experts),
        static_cast<unsigned int>(args.output_m / T::B_M),
        static_cast<unsigned int>(args.output_n / T::B_N));
    hipLaunchKernelGGL((weight_bwd_swizzled_k32_kernel_gfx950<T>),
                       grid,
                       dim3(T::BLOCK_SIZE),
                       0,
                       stream,
                       args);
}

template<typename Traits>
inline void dw2_launch_gfx950(const Dw2Kargs& kargs, hipStream_t stream)
{
    using T = opus::remove_cvref_t<Traits>;
    static_assert(T::B_K == 64, "dw2 registers the K64 kernel");
    AITER_CHECK(kargs.split_k == 1,
                "dw2: first Opus instance requires split_k=1");
    AITER_CHECK(kargs.model_dim % T::B_M == 0 &&
                    kargs.inter_dim % T::B_N == 0,
                "dw2: first instance requires D%32==0 and I%128==0");
    WeightBwdKernelArgs args{};
    args.a = kargs.d_out;
    args.b = kargs.a_scaled;
    args.route = kargs.route;
    args.c = kargs.d_w2;
    args.output_m = kargs.model_dim;
    args.output_n = kargs.inter_dim;
    args.stride_a_row = kargs.stride_do_t;
    args.stride_b_row = kargs.stride_a_scaled_r;
    args.stride_c_expert = kargs.stride_dw2_e;
    args.stride_c_m = kargs.stride_dw2_d;
    const dim3 grid(
        static_cast<unsigned int>(kargs.route.num_experts),
        static_cast<unsigned int>(args.output_m / T::B_M),
        static_cast<unsigned int>(args.output_n / T::B_N));
    hipLaunchKernelGGL((weight_bwd_swizzled_kernel_gfx950<T>),
                       grid,
                       dim3(T::BLOCK_SIZE),
                       0,
                       stream,
                       args);
}

} // namespace opus_moe_backward::gfx950
