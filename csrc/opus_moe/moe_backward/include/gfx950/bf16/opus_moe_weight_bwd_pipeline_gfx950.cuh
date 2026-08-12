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


// Compact gfx950 K32 dW1 path.  A 64x32 dZ slab occupies 4 KiB and a
// 32x128 gathered-X tile occupies 8 KiB.  BM128 keeps two independently
// swizzled dZ slabs beside one X tile; the wave4 BM256 path encodes X as two
// independently swizzled 32x64 slabs to reduce transpose-read conflicts.
// Both four-wave candidates increase reuse without changing threads.
// Keep the operands resident so their stores and transpose reads share one
// pair of barriers per reduction tile.
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
    constexpr bool split_b_n64 = []() constexpr {
        if constexpr(requires { T::SPLIT_B_N64_SWIZZLE; })
            return T::SPLIT_B_N64_SWIZZLE;
        return false;
    }();
    constexpr bool prefetch_reduction_a = []() constexpr {
        if constexpr(requires { T::PREFETCH_REDUCTION_A; })
            return T::PREFETCH_REDUCTION_A;
        return false;
    }();
    static_assert((BM == 64 || BM == 128 || BM == 256) &&
                  (BN == 128 || BN == 256) && BK == 32);
    static_assert((T::BLOCK_SIZE == 256 || T::BLOCK_SIZE == 512) && VEC == 8);
    static_assert(T::BLOCK_SIZE == 512 ? (BM == 256 && BN == 128)
                                      : (BM <= 128 ||
                                         (BM == 256 && BN == 128)));
    static_assert(T::VEC_B == VEC && T::VEC_TR_B == 4);
    constexpr int a_m_slabs = BM / 64;
    constexpr int b_n_slabs = BN / 128;
    static_assert(a_m_slabs * b_n_slabs <= 4,
                  "K4 supports at most four 64x128 operand slab pairs");
    static_assert(T::E_M * T::T_M == BM / T::W_M);
    static_assert(T::E_N * T::T_N == BN / T::W_N);
    static_assert((BM == 64 && BN == 128) || T::DIRECT_GMEM_TO_LDS,
                  "wide K4 tiles require direct GMEM-to-LDS loads");
    static_assert(!split_b_n64 ||
                      (BN == 128 && T::BLOCK_SIZE == 256),
                  "split-B K4 requires a four-wave BN128 tile");
    int expert;
    int output_m_base;
    int output_n_base;
    if constexpr(T::EXPERT_COHORT > 0)
    {
        // Keep a bounded cohort of experts resident in the dispatch stream.
        // M tiles vary first so gathered X rows are reused immediately; the
        // complete per-expert dZ/X working set is then reused across N tiles
        // while it is still in L2.  Interleaving a few experts retains load
        // balance when routing counts differ.
        constexpr int cohort = T::EXPERT_COHORT;
        const int m_tiles = kargs.output_m / BM;
        const int n_tiles = kargs.output_n / BN;
        const int tiles_per_expert = m_tiles * n_tiles;
        const int blocks_per_cohort = cohort * tiles_per_expert;
        const int linear_block = static_cast<int>(blockIdx.x);
        const int cohort_id = linear_block / blocks_per_cohort;
        const int within_cohort = linear_block % blocks_per_cohort;
        const int output_tile = within_cohort / cohort;
        const int scheduled_expert =
            cohort_id * cohort + within_cohort % cohort;
        if(scheduled_expert >= kargs.route.num_experts)
            return;
        if constexpr(requires { T::REVERSE_EXPERT_ORDER; })
        {
            static_assert(T::REVERSE_EXPERT_ORDER);
            expert = kargs.route.num_experts - 1 - scheduled_expert;
        }
        else
        {
            expert = scheduled_expert;
        }
        output_m_base = (output_tile % m_tiles) * BM;
        output_n_base = (output_tile / m_tiles) * BN;
    }
    else
    {
        expert = static_cast<int>(blockIdx.x);
        output_m_base = static_cast<int>(blockIdx.y) * BM;
        output_n_base = static_cast<int>(blockIdx.z) * BN;
    }
    const int route_start = kargs.route.expert_offsets[expert];
    const int route_end = kargs.route.expert_offsets[expert + 1];
    const int route_count = route_end - route_start;
    const int tid = static_cast<int>(thread_id_x());
    if constexpr(T::EMPTY_M_TILES_PER_CTA > 1)
    {
        if(route_count == 0)
        {
            // Coalesce a bounded number of adjacent empty-expert M tiles into
            // one zeroing CTA.  The factor balances dispatch overhead against
            // memory-level parallelism in the store stream.
            constexpr int group = T::EMPTY_M_TILES_PER_CTA;
            const int output_m_tile = output_m_base / BM;
            if(output_m_tile % group != 0)
                return;
            using u32x4 = uint32_t __attribute__((ext_vector_type(4)));
            constexpr int store_values = 8;
            const u32x4 zeros{0, 0, 0, 0};
            const int vectors_per_row = BN / store_values;
            const int rows_left = kargs.output_m - output_m_base;
            const int rows = rows_left < group * BM ? rows_left : group * BM;
            const int total_vectors = rows * vectors_per_row;
            for(int vector = tid; vector < total_vectors;
                vector += T::BLOCK_SIZE)
            {
                const int row = output_m_base + vector / vectors_per_row;
                const int col = (vector % vectors_per_row) * store_values;
                const int64_t c_offset =
                    static_cast<int64_t>(expert) * kargs.stride_c_expert +
                    static_cast<int64_t>(row) * kargs.stride_c_m +
                    output_n_base + col;
                *reinterpret_cast<u32x4*>(kargs.c + c_offset) = zeros;
            }
            return;
        }
    }
    const int lane_id = tid % get_warp_size();
    const int wave_id = __builtin_amdgcn_readfirstlane(tid / get_warp_size());
    const int wave_id_m = wave_id / T::T_N;
    const int wave_id_n = wave_id % T::T_N;

    constexpr int smem_a_bytes = BM * BK * sizeof(D_A);
    constexpr int smem_b_bytes = BN * BK * sizeof(D_B);
    constexpr int smem_stage_bytes = smem_a_bytes + smem_b_bytes;
    constexpr int smem_stages = T::DOUBLE_BUFFER ? 2 : 1;
    __shared__ __align__(16) char tile_storage[smem_stages * smem_stage_bytes];
    auto s_a = make_smem(reinterpret_cast<D_A*>(tile_storage));
    auto s_b =
        make_smem(reinterpret_cast<D_B*>(tile_storage + smem_a_bytes));

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
    auto mma_k_fragment = make_tiled_mma<D_A, D_B, D_ACC>(
        seq<T::E_M, T::E_N, 1>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});
    typename decltype(mma)::vtype_c v_c;
    clear(v_c);

    // buffer_load_*_lds deposits wave lane L at base + L*16.  Invert the
    // existing shared swizzles when selecting each lane's global source so a
    // contiguous hardware deposit reconstructs exactly the same physical LDS
    // tile consumed by the transpose reads below.
    constexpr bool wide_workgroup = T::BLOCK_SIZE == 512;
    const int a_loader_linear_tid =
        wide_workgroup ? (wave_id % 4) * opus::get_warp_size() + lane_id
                       : tid;
    const int a_loader_tid =
        T::DIRECT_GMEM_TO_LDS
            ? a_loader_linear_tid ^ ((a_loader_linear_tid >> 4) & 7)
            : a_loader_linear_tid;
    const int b_loader_tid =
        T::DIRECT_GMEM_TO_LDS ? tid ^ ((tid >> 4) & 15) : tid;
    const int a_local_k = a_loader_tid / 8;
    const int a_local_m = (a_loader_tid % 8) * VEC;
    const int b_local_k0 = b_loader_tid / 16;
    const int b_local_k1 = wide_workgroup ? b_local_k0 : b_local_k0 + 16;
    const int b_local_n = (b_loader_tid % 16) * VEC;
    const int b_split_local_n = a_local_m;
    const int a_store_addr = (tid << 4) ^ (tid & 0x70);
    const int b_store_addr = (tid << 4) ^ (tid & 0xf0);

    const int lane_mix = tid & 0x2c;
    const int a_read_base =
        (lane_mix << 5) |
        (((tid << 1) & 0x70) ^ ((tid << 3) & 0x18));
    const int b_read_lane_base =
        ((lane_mix << 2) ^ ((tid << 1) & 0x20) ^
         ((lane_mix << 6) | ((tid << 3) & 0x18)));
    const int b_read_base = b_read_lane_base ^ (tid & 0xc0);
    // Expert offsets are padded to the sorter block size, which is BK=32 for
    // this kernel, so every reduction tile has exactly 32 addressable rows.
    const int loops = route_count / BK;

    struct TileSources
    {
        int a;
        int b0;
        int b1;
        int b_split;
    };
    auto tile_sources = [&](int tile_k) __attribute__((always_inline)) {
        const int a_sorted_row = route_start + tile_k * BK + a_local_k;
        const int b_sorted_row0 = route_start + tile_k * BK + b_local_k0;
        const int b_sorted_row1 = route_start + tile_k * BK + b_local_k1;
        int token_a;
        int token_b0 = 0;
        int token_b1 = 0;
        if constexpr(T::ROUTE_LAYOUT == RouteLayout::CompactRouteMajor)
        {
            token_a = weight_bwd_source_row<T::ROUTE_LAYOUT, true>(
                kargs.route, a_sorted_row, true);
            if constexpr(!split_b_n64)
            {
                token_b0 = weight_bwd_source_row<T::ROUTE_LAYOUT, true>(
                    kargs.route, b_sorted_row0, true);
                token_b1 = weight_bwd_source_row<T::ROUTE_LAYOUT, true>(
                    kargs.route, b_sorted_row1, true);
            }
        }
        else
        {
            token_a =
                kargs.route.sorted_token_ids[a_sorted_row] & kPackedTokenMask;
            if constexpr(!split_b_n64)
            {
                token_b0 = kargs.route.sorted_token_ids[b_sorted_row0] &
                           kPackedTokenMask;
                token_b1 = kargs.route.sorted_token_ids[b_sorted_row1] &
                           kPackedTokenMask;
            }
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
        const int source_b_split = token_a < kargs.route.token_num
                                       ? token_a
                                       : kargs.route.token_num;
        return TileSources{source_a, source_b0, source_b1, source_b_split};
    };

    auto issue_direct_tile = [&](const TileSources& sources,
                                 auto& s_a_tile,
                                 auto& s_b_tile)
        __attribute__((always_inline)) {
        OPUS_LDS_ADDR D_B* b_wave_dst =
            reinterpret_cast<OPUS_LDS_ADDR D_B*>(s_b_tile.ptr) +
            wave_id * opus::get_warp_size() * VEC;
        if constexpr(wide_workgroup)
        {
            // Eight waves form two four-wave loader groups.  Each group owns
            // one slab in each half of BM256, so every lane issues exactly
            // two dZ vectors and all four 64-row slabs are covered once.
            opus::static_for<2>([&](auto half) {
                const int slab = wave_id / 4 + half.value * 2;
                OPUS_LDS_ADDR D_A* a_wave_dst =
                    reinterpret_cast<OPUS_LDS_ADDR D_A*>(s_a_tile.ptr) +
                    slab * 64 * BK +
                    (wave_id % 4) * opus::get_warp_size() * VEC;
                g_a.template _async_load<VEC>(
                    reinterpret_cast<OPUS_LDS_ADDR void*>(a_wave_dst),
                    (sources.a * kargs.stride_a_row + output_m_base +
                     slab * 64 + a_local_m) *
                        sizeof(D_A),
                    0,
                    opus::number<0>{},
                    opus::number<T::CACHECTL_A>{});
            });
        }
        else
        {
            opus::static_for<a_m_slabs>([&](auto slab) {
                OPUS_LDS_ADDR D_A* a_wave_dst =
                    reinterpret_cast<OPUS_LDS_ADDR D_A*>(s_a_tile.ptr) +
                    slab.value * 64 * BK +
                    wave_id * opus::get_warp_size() * VEC;
                g_a.template _async_load<VEC>(
                    reinterpret_cast<OPUS_LDS_ADDR void*>(a_wave_dst),
                    (sources.a * kargs.stride_a_row + output_m_base +
                     slab.value * 64 + a_local_m) *
                        sizeof(D_A),
                    0,
                    opus::number<0>{},
                    opus::number<T::CACHECTL_A>{});
            });
        }
        if constexpr(split_b_n64)
        {
            opus::static_for<2>([&](auto slab) {
                OPUS_LDS_ADDR D_B* b_slab_dst =
                    b_wave_dst + slab.value * 64 * BK;
                const int source_n =
                    output_n_base + slab.value * 64 + b_split_local_n;
                g_b.template _async_load<VEC>(
                    reinterpret_cast<OPUS_LDS_ADDR void*>(b_slab_dst),
                    (sources.b_split * kargs.stride_b_row + source_n) *
                        sizeof(D_B),
                    0,
                    opus::number<0>{},
                    opus::number<T::CACHECTL_B>{});
            });
        }
        else
        {
            opus::static_for<b_n_slabs>([&](auto slab) {
                OPUS_LDS_ADDR D_B* b_slab_dst =
                    b_wave_dst + slab.value * 128 * BK;
                const int source_n =
                    output_n_base + slab.value * 128 + b_local_n;
                g_b.template _async_load<VEC>(
                    reinterpret_cast<OPUS_LDS_ADDR void*>(b_slab_dst),
                    (sources.b0 * kargs.stride_b_row + source_n) *
                        sizeof(D_B),
                    0,
                    opus::number<0>{},
                    opus::number<T::CACHECTL_B>{});
                if constexpr(!wide_workgroup)
                    g_b.template _async_load<VEC>(
                        reinterpret_cast<OPUS_LDS_ADDR void*>(
                            reinterpret_cast<OPUS_LDS_ADDR char*>(
                                b_slab_dst) +
                            4096),
                        (sources.b1 * kargs.stride_b_row + source_n) *
                            sizeof(D_B),
                        0,
                        opus::number<0>{},
                        opus::number<T::CACHECTL_B>{});
            });
        }
    };

    auto compute_tile = [&](auto& s_a_tile, auto& s_b_tile)
        __attribute__((always_inline)) {
        if constexpr(T::PIPELINE_REDUCTION_FRAGMENTS)
        {
            static_assert(T::E_K == 2);
            static_assert(std::is_same_v<
                          typename decltype(mma)::vtype_c,
                          typename decltype(mma_k_fragment)::vtype_c>);
            auto load_a_fragment = [&](auto ek, auto& v_a)
                __attribute__((always_inline)) {
                opus::static_for<T::E_M>([&](auto em) {
                    constexpr int k_byte_offset = ek.value * 2048;
                    int base;
                    if constexpr(T::T_M == 1)
                    {
                        constexpr int slab = em.value / 2;
                        constexpr int em_in_slab = em.value % 2;
                        base = a_read_base + slab * 4096;
                        base ^= em_in_slab * 0x40;
                    }
                    else
                    {
                        const int native_m =
                            em.value * T::T_M + wave_id_m;
                        base = a_read_base + (native_m / 2) * 4096;
                        base ^= (native_m % 2) * 0x40;
                    }
                    auto lo = s_a_tile.template _tr_load<T::VEC_TR_B,
                                                          k_byte_offset>(base);
                    auto hi = s_a_tile.template _tr_load<T::VEC_TR_B,
                                                          k_byte_offset>(
                        base ^ 0x220);
#pragma unroll
                    for(int j = 0; j < T::VEC_TR_B; ++j)
                    {
                        v_a[em.value * 2 * T::VEC_TR_B + j] = lo[j];
                        v_a[em.value * 2 * T::VEC_TR_B +
                            T::VEC_TR_B + j] = hi[j];
                    }
                });
            };
            auto load_b_fragment = [&](auto ek, auto& v_b)
                __attribute__((always_inline)) {
                opus::static_for<T::E_N>([&](auto en) {
                    constexpr int k_byte_offset =
                        ek.value * (split_b_n64 ? 2048 : 4096);
                    int base;
                    if constexpr(split_b_n64)
                    {
                        const int native_n =
                            en.value * T::T_N + wave_id_n;
                        base = a_read_base + (native_n / 2) * 4096;
                        base ^= (native_n % 2) * 0x40;
                    }
                    else if constexpr(T::T_N == 4)
                    {
                        base = b_read_base + en.value * 8192;
                    }
                    else
                    {
                        const int native_n =
                            en.value * T::T_N + wave_id_n;
                        base = b_read_lane_base + (native_n / 4) * 8192;
                        base ^= (native_n % 4) * 0x40;
                    }
                    auto lo = s_b_tile.template _tr_load<T::VEC_TR_B,
                                                          k_byte_offset>(base);
                    auto hi = s_b_tile.template _tr_load<T::VEC_TR_B,
                                                          k_byte_offset>(
                        base ^
                        (split_b_n64 ? 0x220 : 0x440));
#pragma unroll
                    for(int j = 0; j < T::VEC_TR_B; ++j)
                    {
                        v_b[en.value * 2 * T::VEC_TR_B + j] = lo[j];
                        v_b[en.value * 2 * T::VEC_TR_B +
                            T::VEC_TR_B + j] = hi[j];
                    }
                });
            };
            auto load_fragment = [&](auto ek, auto& v_a, auto& v_b)
                __attribute__((always_inline)) {
                load_a_fragment(ek, v_a);
                load_b_fragment(ek, v_b);
            };
            if constexpr(prefetch_reduction_a)
            {
                typename decltype(mma_k_fragment)::vtype_a v_a0;
                typename decltype(mma_k_fragment)::vtype_b v_b0;
                typename decltype(mma_k_fragment)::vtype_a v_a1;
                typename decltype(mma_k_fragment)::vtype_b v_b1;
                load_fragment(opus::number<0>{}, v_a0, v_b0);
                s_waitcnt_lgkmcnt(0_I);
                load_a_fragment(opus::number<1>{}, v_a1);
                __builtin_amdgcn_s_setprio(1);
                v_c = mma_k_fragment(v_a0, v_b0, v_c);
                load_b_fragment(opus::number<1>{}, v_b1);
                s_waitcnt_lgkmcnt(0_I);
                v_c = mma_k_fragment(v_a1, v_b1, v_c);
                __builtin_amdgcn_s_setprio(0);
            }
            else
            {
                opus::static_for<T::E_K>([&](auto ek) {
                    typename decltype(mma_k_fragment)::vtype_a v_a;
                    typename decltype(mma_k_fragment)::vtype_b v_b;
                    load_fragment(ek, v_a, v_b);
                    s_waitcnt_lgkmcnt(0_I);
                    __builtin_amdgcn_s_setprio(1);
                    v_c = mma_k_fragment(v_a, v_b, v_c);
                    __builtin_amdgcn_s_setprio(0);
                });
            }
            __builtin_amdgcn_sched_barrier(0);
            return;
        }

        typename decltype(mma)::vtype_a v_a;
        opus::static_for<T::E_M>([&](auto em) {
            opus::static_for<T::E_K>([&](auto ek) {
                constexpr int k_byte_offset = ek.value * 2048;
                int base;
                if constexpr(T::T_M == 1)
                {
                    constexpr int slab = em.value / 2;
                    constexpr int em_in_slab = em.value % 2;
                    base = a_read_base + slab * 4096;
                    base ^= em_in_slab * 0x40;
                }
                else
                {
                    const int native_m = em.value * T::T_M + wave_id_m;
                    base = a_read_base + (native_m / 2) * 4096;
                    base ^= (native_m % 2) * 0x40;
                }
                auto lo = s_a_tile.template _tr_load<T::VEC_TR_B,
                                                      k_byte_offset>(base);
                auto hi = s_a_tile.template _tr_load<T::VEC_TR_B,
                                                      k_byte_offset>(
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
        opus::static_for<T::E_N>([&](auto en) {
            opus::static_for<T::E_K>([&](auto ek) {
                constexpr int k_byte_offset =
                    ek.value * (split_b_n64 ? 2048 : 4096);
                int base;
                if constexpr(split_b_n64)
                {
                    const int native_n =
                        en.value * T::T_N + wave_id_n;
                    base = a_read_base + (native_n / 2) * 4096;
                    base ^= (native_n % 2) * 0x40;
                }
                else if constexpr(T::T_N == 4)
                {
                    base = b_read_base + en.value * 8192;
                }
                else
                {
                    const int native_n = en.value * T::T_N + wave_id_n;
                    base = b_read_lane_base + (native_n / 4) * 8192;
                    base ^= (native_n % 4) * 0x40;
                }
                auto lo = s_b_tile.template _tr_load<T::VEC_TR_B,
                                                      k_byte_offset>(base);
                auto hi = s_b_tile.template _tr_load<T::VEC_TR_B,
                                                      k_byte_offset>(
                    base ^
                    (split_b_n64 ? 0x220 : 0x440));
                constexpr int fragment = en.value * T::E_K + ek.value;
#pragma unroll
                for(int j = 0; j < T::VEC_TR_B; ++j)
                {
                    v_b[fragment * 2 * T::VEC_TR_B + j] = lo[j];
                    v_b[fragment * 2 * T::VEC_TR_B + T::VEC_TR_B + j] = hi[j];
                }
            });
        });
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c = mma(v_a, v_b, v_c);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_sched_barrier(0);
    };

    if constexpr(T::DOUBLE_BUFFER)
    {
        static_assert(T::DIRECT_GMEM_TO_LDS,
                      "double-buffered K4 requires direct GMEM-to-LDS");
        auto s_a_next = make_smem(
            reinterpret_cast<D_A*>(tile_storage + smem_stage_bytes));
        auto s_b_next = make_smem(reinterpret_cast<D_B*>(
            tile_storage + smem_stage_bytes + smem_a_bytes));
        if(loops > 0)
        {
            const auto first = tile_sources(0);
            issue_direct_tile(first, s_a, s_b);
            s_waitcnt_vmcnt(0_I);
            __builtin_amdgcn_s_barrier();

            int tile_k = 0;
            for(; tile_k + 1 < loops; tile_k += 2)
            {
                const auto odd = tile_sources(tile_k + 1);
                issue_direct_tile(odd, s_a_next, s_b_next);
                compute_tile(s_a, s_b);
                s_waitcnt_vmcnt(0_I);
                __builtin_amdgcn_s_barrier();

                if(tile_k + 2 < loops)
                {
                    const auto next_even = tile_sources(tile_k + 2);
                    issue_direct_tile(next_even, s_a, s_b);
                }
                compute_tile(s_a_next, s_b_next);
                if(tile_k + 2 < loops)
                {
                    s_waitcnt_vmcnt(0_I);
                    __builtin_amdgcn_s_barrier();
                }
            }
            if(tile_k < loops)
                compute_tile(s_a, s_b);
        }
    }
    else
    {
        for(int tile_k = 0; tile_k < loops; ++tile_k)
        {
            const auto sources = tile_sources(tile_k);

            if constexpr(T::DIRECT_GMEM_TO_LDS)
            {
                // Close the prior tile before VMEM starts overwriting this
                // single shared stage.  gfx950 adds the lane-vector
                // displacement to the wave-uniform destination base.
                __builtin_amdgcn_s_barrier();
                issue_direct_tile(sources, s_a, s_b);
                s_waitcnt_vmcnt(0_I);
                __builtin_amdgcn_s_barrier();
            }
            else
            {
                auto a0 = g_a.template load<VEC>(
                    sources.a * kargs.stride_a_row + output_m_base + a_local_m,
                    0,
                    opus::number<T::CACHECTL_A>{});
                auto b0 = g_b.template load<VEC>(
                    sources.b0 * kargs.stride_b_row + output_n_base + b_local_n,
                    0,
                    opus::number<T::CACHECTL_B>{});
                auto b1 = g_b.template load<VEC>(
                    sources.b1 * kargs.stride_b_row + output_n_base + b_local_n,
                    0,
                    opus::number<T::CACHECTL_B>{});

                __builtin_amdgcn_s_barrier();
                s_a.template _store<VEC>(a0, a_store_addr);
                s_b.template _store<VEC>(b0, b_store_addr);
                s_b.template _store<VEC>(b1, b_store_addr + 4096);
                s_waitcnt_lgkmcnt(0_I);
                __builtin_amdgcn_s_barrier();
            }
            compute_tile(s_a, s_b);
        }
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
    static_assert((BM == 64 || BM == 128 || BM == 256) &&
                  (BN == 64 || BN == 128) && BK == 64);
    static_assert((T::BLOCK_SIZE == 256 || T::BLOCK_SIZE == 512) && VEC == 8);
    static_assert(T::VEC_B == VEC && T::VEC_TR_B == 4);
    constexpr int a_m_slabs = BM / 64;
    constexpr int b_n_slabs = BN / 64;
    static_assert(a_m_slabs * b_n_slabs <= 8,
                  "K5 supports at most eight 64x64 operand slab pairs");
    static_assert(T::E_M * T::T_M == BM / T::W_M);
    static_assert(T::E_N * T::T_N == BN / T::W_N);
    static_assert((BM == 64 && BN == 64) || T::DUAL_OPERAND_LDS,
                  "wide K5 tiles require dual-operand LDS");
    int expert;
    int output_m_base;
    int output_n_base;
    if constexpr(T::EXPERT_COHORT > 0)
    {
        constexpr int cohort = T::EXPERT_COHORT;
        const int m_tiles = kargs.output_m / BM;
        const int n_tiles = kargs.output_n / BN;
        const int tiles_per_expert = m_tiles * n_tiles;
        const int blocks_per_cohort = cohort * tiles_per_expert;
        const int linear_block = static_cast<int>(blockIdx.x);
        const int cohort_id = linear_block / blocks_per_cohort;
        const int within_cohort = linear_block % blocks_per_cohort;
        const int output_tile = within_cohort / cohort;
        expert = cohort_id * cohort + within_cohort % cohort;
        if(expert >= kargs.route.num_experts)
            return;
        output_m_base = (output_tile % m_tiles) * BM;
        output_n_base = (output_tile / m_tiles) * BN;
    }
    else
    {
        expert = static_cast<int>(blockIdx.x);
        output_m_base = static_cast<int>(blockIdx.y) * BM;
        output_n_base = static_cast<int>(blockIdx.z) * BN;
    }
    const int route_start = kargs.route.expert_offsets[expert];
    const int route_end = kargs.route.expert_offsets[expert + 1];
    const int route_count = route_end - route_start;
    if constexpr(requires { T::MIN_ROUTE_COUNT; })
    {
        if(route_count < T::MIN_ROUTE_COUNT)
            return;
    }
    if constexpr(requires { T::MAX_ROUTE_COUNT; })
    {
        if(route_count > T::MAX_ROUTE_COUNT)
            return;
    }
    if constexpr(requires { T::EXCLUDED_MIN_ROUTE_COUNT;
                            T::EXCLUDED_MAX_ROUTE_COUNT; })
    {
        if(route_count >= T::EXCLUDED_MIN_ROUTE_COUNT &&
           route_count <= T::EXCLUDED_MAX_ROUTE_COUNT)
            return;
    }
    const int tid = static_cast<int>(thread_id_x());
    const int lane_id = tid % get_warp_size();
    const int wave_id = __builtin_amdgcn_readfirstlane(tid / get_warp_size());
    const int wave_id_m = wave_id / T::T_N;
    const int wave_id_n = wave_id % T::T_N;

    constexpr int a_tile_bytes = BK * BM * sizeof(D_A);
    constexpr int b_tile_bytes = BK * BN * sizeof(D_B);
    constexpr int tile_storage_bytes =
        T::DUAL_OPERAND_LDS
            ? a_tile_bytes + b_tile_bytes
            : (a_tile_bytes > b_tile_bytes ? a_tile_bytes : b_tile_bytes);
    __shared__ __align__(16) char tile_storage[tile_storage_bytes];
    auto s_tile = make_smem(reinterpret_cast<D_A*>(tile_storage));
    auto s_tile_b = make_smem(reinterpret_cast<D_B*>(
        tile_storage + (T::DUAL_OPERAND_LDS ? a_tile_bytes : 0)));

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
    auto async_load_b = [&](auto lds_dst, int byte_offset)
        __attribute__((always_inline)) {
        if constexpr(requires { T::RUNTIME_SHORT_ROUTE_CACHE_B; })
        {
            static_assert(T::RUNTIME_SHORT_ROUTE_CACHE_B);
            if(route_count <= T::SHORT_ROUTE_CACHE_B_MAX_ROUTES)
                g_b.template _async_load<VEC>(
                    lds_dst,
                    byte_offset,
                    0,
                    opus::number<0>{},
                    opus::number<T::SHORT_ROUTE_CACHECTL_B>{});
            else
                g_b.template _async_load<VEC>(lds_dst,
                                              byte_offset,
                                              0,
                                              opus::number<0>{},
                                              opus::number<T::CACHECTL_B>{});
        }
        else
            g_b.template _async_load<VEC>(lds_dst,
                                          byte_offset,
                                          0,
                                          opus::number<0>{},
                                          opus::number<T::CACHECTL_B>{});
    };

    auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
        seq<T::E_M, T::E_N, T::E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});
    auto mma_k_fragment = make_tiled_mma<D_A, D_B, D_ACC>(
        seq<T::E_M, T::E_N, 1>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});
    typename decltype(mma)::vtype_c v_c;
    clear(v_c);

    // Eight lanes cover one K row; every lane owns eight contiguous columns.
    const int loader_linear_tid =
        T::BLOCK_SIZE == 512 ? tid & 255 : tid;
    const int loader_tid =
        T::DIRECT_GMEM_TO_LDS
            ? loader_linear_tid ^ ((loader_linear_tid >> 4) & 7)
            : loader_linear_tid;
    const int local_k0 = loader_tid / 8;
    const int local_k1 = local_k0 + 32;
    const int local_vec = (loader_tid % 8) * VEC;
    // The two logical shared encodings have the same vector-store address.
    const int store_addr_bytes = (tid << 4) ^ (tid & 0x70);

    // Operand-specific transpose-read bases emitted by the two swizzled
    // shared encodings.  Wave bits select the 32-row M/N subtile.
    const int common_hi = (tid << 5) & 0x580;
    const int common_lo = (tid << 3) & 0x18;
    const int a_read_lane_base =
        ((tid << 1) & 0x70) ^ common_hi ^ common_lo;
    const int b_read_lane_base =
        common_hi | (((tid << 1) & 0x70) ^ common_lo);
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

        if constexpr(T::DUAL_OPERAND_LDS)
        {
            static_assert(T::DIRECT_GMEM_TO_LDS,
                          "dual operand LDS requires direct GMEM-to-LDS");
            // Close the prior tile before overwriting either operand.  The
            // two independent LDS destinations let all four K-half loads run
            // under one VMEM wait and one producer/consumer barrier.
            __builtin_amdgcn_s_barrier();
            if constexpr(T::BLOCK_SIZE == 512)
            {
                const int load_slab = wave_id / 4;
                const int load_wave = wave_id % 4;
                OPUS_LDS_ADDR char* a_slab_dst =
                    reinterpret_cast<OPUS_LDS_ADDR char*>(s_tile.ptr) +
                    load_slab * 64 * BK * sizeof(D_A) +
                    load_wave * opus::get_warp_size() * VEC * sizeof(D_A);
                OPUS_LDS_ADDR char* b_slab_dst =
                    reinterpret_cast<OPUS_LDS_ADDR char*>(s_tile_b.ptr) +
                    load_slab * 64 * BK * sizeof(D_B) +
                    load_wave * opus::get_warp_size() * VEC * sizeof(D_B);
                const int source_m =
                    output_m_base + load_slab * 64 + local_vec;
                const int source_n =
                    output_n_base + load_slab * 64 + local_vec;
                g_a.template _async_load<VEC>(
                    reinterpret_cast<OPUS_LDS_ADDR void*>(a_slab_dst),
                    (source_a0 * kargs.stride_a_row + source_m) * sizeof(D_A),
                    0,
                    opus::number<0>{},
                    opus::number<T::CACHECTL_A>{});
                g_a.template _async_load<VEC>(
                    reinterpret_cast<OPUS_LDS_ADDR void*>(a_slab_dst + 4096),
                    (source_a1 * kargs.stride_a_row + source_m) * sizeof(D_A),
                    0,
                    opus::number<0>{},
                    opus::number<T::CACHECTL_A>{});
                async_load_b(
                    reinterpret_cast<OPUS_LDS_ADDR void*>(b_slab_dst),
                    (source_b0 * kargs.stride_b_row + source_n) * sizeof(D_B));
                async_load_b(
                    reinterpret_cast<OPUS_LDS_ADDR void*>(b_slab_dst + 4096),
                    (source_b1 * kargs.stride_b_row + source_n) * sizeof(D_B));
            }
            else
            {
                OPUS_LDS_ADDR D_A* a_wave_dst =
                    reinterpret_cast<OPUS_LDS_ADDR D_A*>(s_tile.ptr) +
                    wave_id * opus::get_warp_size() * VEC;
                OPUS_LDS_ADDR D_B* b_wave_dst =
                    reinterpret_cast<OPUS_LDS_ADDR D_B*>(s_tile_b.ptr) +
                    wave_id * opus::get_warp_size() * VEC;
                opus::static_for<a_m_slabs>([&](auto slab) {
                    OPUS_LDS_ADDR char* a_slab_dst =
                        reinterpret_cast<OPUS_LDS_ADDR char*>(a_wave_dst) +
                        slab.value * 64 * BK * sizeof(D_A);
                    const int source_m =
                        output_m_base + slab.value * 64 + local_vec;
                    g_a.template _async_load<VEC>(
                        reinterpret_cast<OPUS_LDS_ADDR void*>(a_slab_dst),
                        (source_a0 * kargs.stride_a_row + source_m) *
                            sizeof(D_A),
                        0,
                        opus::number<0>{},
                        opus::number<T::CACHECTL_A>{});
                    g_a.template _async_load<VEC>(
                        reinterpret_cast<OPUS_LDS_ADDR void*>(a_slab_dst +
                                                              4096),
                        (source_a1 * kargs.stride_a_row + source_m) *
                            sizeof(D_A),
                        0,
                        opus::number<0>{},
                        opus::number<T::CACHECTL_A>{});
                });
                opus::static_for<b_n_slabs>([&](auto slab) {
                    OPUS_LDS_ADDR char* b_slab_dst =
                        reinterpret_cast<OPUS_LDS_ADDR char*>(b_wave_dst) +
                        slab.value * 64 * BK * sizeof(D_B);
                    const int source_n =
                        output_n_base + slab.value * 64 + local_vec;
                    async_load_b(
                        reinterpret_cast<OPUS_LDS_ADDR void*>(b_slab_dst),
                        (source_b0 * kargs.stride_b_row + source_n) *
                            sizeof(D_B));
                    async_load_b(
                        reinterpret_cast<OPUS_LDS_ADDR void*>(b_slab_dst +
                                                              4096),
                        (source_b1 * kargs.stride_b_row + source_n) *
                            sizeof(D_B));
                });
            }
            s_waitcnt_vmcnt(0_I);
            __builtin_amdgcn_s_barrier();
        }
        else if constexpr(T::DIRECT_GMEM_TO_LDS)
        {
            // Invert the {vec=8, perPhase=2, maxPhase=8} shared swizzle in
            // the global vector owner.  gfx950 then deposits every 16-byte
            // lane vector directly into the physical LDS layout consumed by
            // the transpose reads below.
            __builtin_amdgcn_s_barrier();
            OPUS_LDS_ADDR D_A* wave_dst =
                reinterpret_cast<OPUS_LDS_ADDR D_A*>(s_tile.ptr) +
                wave_id * opus::get_warp_size() * VEC;
            g_a.template _async_load<VEC>(
                reinterpret_cast<OPUS_LDS_ADDR void*>(wave_dst),
                (source_a0 * kargs.stride_a_row + output_m_base + local_vec) *
                    sizeof(D_A),
                0,
                opus::number<0>{},
                opus::number<T::CACHECTL_A>{});
            g_a.template _async_load<VEC>(
                reinterpret_cast<OPUS_LDS_ADDR void*>(
                    reinterpret_cast<OPUS_LDS_ADDR char*>(wave_dst) + 4096),
                (source_a1 * kargs.stride_a_row + output_m_base + local_vec) *
                    sizeof(D_A),
                0,
                opus::number<0>{},
                opus::number<T::CACHECTL_A>{});
            s_waitcnt_vmcnt(0_I);
            __builtin_amdgcn_s_barrier();
        }
        else
        {
            auto a0 = g_a.template load<VEC>(
                source_a0 * kargs.stride_a_row + output_m_base + local_vec,
                0,
                opus::number<T::CACHECTL_A>{});
            auto a1 = g_a.template load<VEC>(
                source_a1 * kargs.stride_a_row + output_m_base + local_vec,
                0,
                opus::number<T::CACHECTL_A>{});
            __builtin_amdgcn_s_barrier();
            s_tile.template _store<VEC>(a0, store_addr_bytes);
            s_tile.template _store<VEC>(a1, store_addr_bytes + 4096);
            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_barrier();
        }

        if constexpr(requires { T::PIPELINE_REDUCTION_FRAGMENTS; })
        {
            static_assert(T::PIPELINE_REDUCTION_FRAGMENTS);
            static_assert(T::DUAL_OPERAND_LDS && T::E_K == 4);
            static_assert(std::is_same_v<
                          typename decltype(mma)::vtype_c,
                          typename decltype(mma_k_fragment)::vtype_c>);
            opus::static_for<T::E_K>([&](auto ek) {
                typename decltype(mma_k_fragment)::vtype_a v_a_fragment;
                opus::static_for<T::E_M>([&](auto em) {
                    const int native_m = em.value * T::T_M + wave_id_m;
                    const int base =
                        (a_read_lane_base +
                         (native_m / 2) * 64 * BK * sizeof(D_A)) ^
                        ((native_m % 2) * 0x40);
                    constexpr int byte_offset = ek.value * 2048;
                    auto lo =
                        s_tile.template _tr_load<T::VEC_TR_B, byte_offset>(
                            base);
                    auto hi =
                        s_tile.template _tr_load<T::VEC_TR_B, byte_offset>(
                            base ^ 0x220);
#pragma unroll
                    for(int j = 0; j < T::VEC_TR_B; ++j)
                    {
                        v_a_fragment[em.value * 2 * T::VEC_TR_B + j] =
                            lo[j];
                        v_a_fragment[em.value * 2 * T::VEC_TR_B +
                                     T::VEC_TR_B + j] = hi[j];
                    }
                });
                typename decltype(mma_k_fragment)::vtype_b v_b_fragment;
                opus::static_for<T::E_N>([&](auto en) {
                    const int native_n = en.value * T::T_N + wave_id_n;
                    const int base =
                        (b_read_lane_base +
                         (native_n / 2) * 64 * BK * sizeof(D_B)) ^
                        ((native_n % 2) * 0x40);
                    constexpr int byte_offset = ek.value * 2048;
                    auto lo =
                        s_tile_b.template _tr_load<T::VEC_TR_B, byte_offset>(
                            base);
                    auto hi =
                        s_tile_b.template _tr_load<T::VEC_TR_B, byte_offset>(
                            base ^ 0x220);
#pragma unroll
                    for(int j = 0; j < T::VEC_TR_B; ++j)
                    {
                        v_b_fragment[en.value * 2 * T::VEC_TR_B + j] =
                            lo[j];
                        v_b_fragment[en.value * 2 * T::VEC_TR_B +
                                     T::VEC_TR_B + j] = hi[j];
                    }
                });
                s_waitcnt_lgkmcnt(0_I);
                __builtin_amdgcn_s_setprio(1);
                v_c = mma_k_fragment(v_a_fragment, v_b_fragment, v_c);
                __builtin_amdgcn_s_setprio(0);
            });
            __builtin_amdgcn_sched_barrier(0);
            return;
        }

        typename decltype(mma)::vtype_a v_a;
        opus::static_for<T::E_M>([&](auto em) {
            opus::static_for<T::E_K>([&](auto ek) {
                const int native_m = em.value * T::T_M + wave_id_m;
                const int base =
                    (a_read_lane_base +
                     (native_m / 2) * 64 * BK * sizeof(D_A)) ^
                    ((native_m % 2) * 0x40);
                constexpr int byte_offset = ek.value * 2048;
                auto lo = s_tile.template _tr_load<T::VEC_TR_B, byte_offset>(
                    base);
                auto hi = s_tile.template _tr_load<T::VEC_TR_B, byte_offset>(
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
        if constexpr(!T::DUAL_OPERAND_LDS)
        {
            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_barrier();
        }

        if constexpr(T::DUAL_OPERAND_LDS)
        {
            // Both operands were produced together above.  Keeping this
            // branch empty is what removes the overwrite barrier pair.
        }
        else if constexpr(T::DIRECT_GMEM_TO_LDS)
        {
            OPUS_LDS_ADDR D_B* wave_dst =
                reinterpret_cast<OPUS_LDS_ADDR D_B*>(s_tile.ptr) +
                wave_id * opus::get_warp_size() * VEC;
            async_load_b(
                reinterpret_cast<OPUS_LDS_ADDR void*>(wave_dst),
                (source_b0 * kargs.stride_b_row + output_n_base + local_vec) *
                    sizeof(D_B));
            async_load_b(
                reinterpret_cast<OPUS_LDS_ADDR void*>(
                    reinterpret_cast<OPUS_LDS_ADDR char*>(wave_dst) + 4096),
                (source_b1 * kargs.stride_b_row + output_n_base + local_vec) *
                    sizeof(D_B));
            s_waitcnt_vmcnt(0_I);
            __builtin_amdgcn_s_barrier();
        }
        else
        {
            auto b0 = g_b.template load<VEC>(
                source_b0 * kargs.stride_b_row + output_n_base + local_vec,
                0,
                opus::number<T::CACHECTL_B>{});
            auto b1 = g_b.template load<VEC>(
                source_b1 * kargs.stride_b_row + output_n_base + local_vec,
                0,
                opus::number<T::CACHECTL_B>{});
            s_tile.template _store<VEC>(b0, store_addr_bytes);
            s_tile.template _store<VEC>(b1, store_addr_bytes + 4096);
            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_barrier();
        }

        typename decltype(mma)::vtype_b v_b;
        opus::static_for<T::E_N>([&](auto en) {
            opus::static_for<T::E_K>([&](auto ek) {
                const int native_n = en.value * T::T_N + wave_id_n;
                const int base =
                    (b_read_lane_base +
                     (native_n / 2) * 64 * BK * sizeof(D_B)) ^
                    ((native_n % 2) * 0x40);
                constexpr int byte_offset = ek.value * 2048;
                auto lo = s_tile_b.template _tr_load<T::VEC_TR_B, byte_offset>(
                    base);
                auto hi = s_tile_b.template _tr_load<T::VEC_TR_B, byte_offset>(
                    base ^ 0x220);
                constexpr int fragment = en.value * T::E_K + ek.value;
#pragma unroll
                for(int j = 0; j < T::VEC_TR_B; ++j)
                {
                    v_b[fragment * 2 * T::VEC_TR_B + j] = lo[j];
                    v_b[fragment * 2 * T::VEC_TR_B + T::VEC_TR_B + j] =
                        hi[j];
                }
            });
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
    dim3 grid;
    if constexpr(T::EXPERT_COHORT > 0)
    {
        constexpr int cohort = T::EXPERT_COHORT;
        const int padded_experts =
            ((kargs.route.num_experts + cohort - 1) / cohort) * cohort;
        const int output_tiles = (args.output_m / T::B_M) *
                                 (args.output_n / T::B_N);
        grid = dim3(static_cast<unsigned int>(padded_experts * output_tiles));
    }
    else
    {
        grid = dim3(
            static_cast<unsigned int>(kargs.route.num_experts),
            static_cast<unsigned int>(args.output_m / T::B_M),
            static_cast<unsigned int>(args.output_n / T::B_N));
    }
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
    if constexpr(requires { T::ADAPTIVE_BM256_ROUTE_SPLIT; })
    {
        static_assert(T::ADAPTIVE_BM256_ROUTE_SPLIT);
        dw2_launch_gfx950<
            Dw2Bf16Gfx950Bm256Bn128Bk64SwizzledCohort4OutsideMidRoutes>(
            kargs, stream);
        dw2_launch_gfx950<
            Dw2Bf16Gfx950Bm128Bn128Bk64SwizzledWave2x2MidRoutes>(
            kargs, stream);
        return;
    }
    if constexpr(requires { T::ADAPTIVE_ROUTE_SPLIT; })
    {
        static_assert(T::ADAPTIVE_ROUTE_SPLIT);
        dw2_launch_gfx950<
            Dw2Bf16Gfx950Bm128Bn128Bk64SwizzledRouteLe30720>(kargs, stream);
        dw2_launch_gfx950<
            Dw2Bf16Gfx950Bm128Bn128Bk64SwizzledRouteGt30720>(kargs, stream);
        return;
    }
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
    dim3 grid;
    if constexpr(T::EXPERT_COHORT > 0)
    {
        constexpr int cohort = T::EXPERT_COHORT;
        const int padded_experts =
            ((kargs.route.num_experts + cohort - 1) / cohort) * cohort;
        const int output_tiles = (args.output_m / T::B_M) *
                                 (args.output_n / T::B_N);
        grid = dim3(static_cast<unsigned int>(padded_experts * output_tiles));
    }
    else
    {
        grid = dim3(
            static_cast<unsigned int>(kargs.route.num_experts),
            static_cast<unsigned int>(args.output_m / T::B_M),
            static_cast<unsigned int>(args.output_n / T::B_N));
    }
    hipLaunchKernelGGL((weight_bwd_swizzled_kernel_gfx950<T>),
                       grid,
                       dim3(T::BLOCK_SIZE),
                       0,
                       stream,
                       args);
}

} // namespace opus_moe_backward::gfx950
