// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// Raw FP8 prefetch for full-FP8 decode. Q remains head-major.
#pragma once
#include "cache_layout.hpp"
#include "cache_pipeline.hpp"

template <bool Packed, int PageSize = 0>
struct PackedDecodeSource64
{
    static constexpr bool is_packed = Packed;
    const void* pool;
    const void* indices;
    int length;
    int64_t extent; // bytes for packed, rows for BF16
    Fp8CacheLayout cache{};

    __device__ __forceinline__ int64_t row(int i) const
    {
        if(i < 0 || i >= length)
            return -1;
        if constexpr(Packed)
            return cache.row<PageSize>(indices, i);
        else
            return static_cast<const int*>(indices)[i];
    }
    __device__ __forceinline__ bool valid_row(int64_t r) const
    {
        if constexpr(Packed)
            return cache.valid<PageSize>(r, extent);
        else
            return r >= 0 && r < extent;
    }
    __device__ __forceinline__ bool valid(int i) const { return valid_row(row(i)); }

    // The public byte-address contract guarantees 4-byte row alignment.
    // Explicit scalar fields let LLVM form wide loads without assuming 16-byte alignment.
    struct Words2
    {
        uint32_t x, y;
    };
    struct Words4
    {
        uint32_t x, y, z, w;
    };
    struct RawTile
    {
        opus::vector_t<uint32_t, 2> bytes[8];
        opus::vector_t<uint32_t, 2> scales; // Eight lane-specific scale codes.
        bool live;
    };
    __device__ __forceinline__ RawTile prefetch(int tile, int wave, int lane) const
    {
        // Recompute lane-derived addresses in this stage instead of keeping
        // them live across QK/PV. Restore the range lost by the opaque asm.
        asm volatile("" : "+v"(lane));
        lane &= 63;
        RawTile tile_data{};
        const int64_t r = row(tile * 32 + (lane / 8) * 4 + (wave % 4));
        tile_data.live  = valid_row(r);
        if(tile_data.live)
        {
            const auto* src      = cache.data<PageSize>(pool, r);
            const auto* scales   = reinterpret_cast<const Words4*>(cache.scale<PageSize>(pool, r));
            const unsigned shift = ((lane % 8) / 4) * 8;
            auto pair_codes      = [shift](uint32_t word) {
                return ((word >> shift) & 255u) | (((word >> (shift + 16)) & 255u) << 8);
            };
            tile_data.scales = {pair_codes(scales->x) | (pair_codes(scales->y) << 16),
                                pair_codes(scales->z) | (pair_codes(scales->w) << 16)};
            opus::static_for<8>([&](auto j) {
                const auto* values =
                    reinterpret_cast<const Words2*>(src + int(j) * 64 + (lane % 8) * 8);
                tile_data.bytes[int(j)] = {values->x, values->y};
            });
        }
        return tile_data;
    }
    template <class Pointer>
    __device__ __forceinline__ void
    commit(const RawTile& raw, Pointer dst, int wave, int lane) const
    {
        // Recompute lane-derived addresses in this stage instead of keeping
        // them live across QK/PV. Restore the range lost by the opaque asm.
        asm volatile("" : "+v"(lane));
        lane &= 63;
        auto* lds = reinterpret_cast<OPUS_LDS_ADDR bf16_t*>(reinterpret_cast<uintptr_t>(dst));
        if((lane & 7) == 0)
            reinterpret_cast<OPUS_LDS_ADDR uint32_t*>(lds + wave * 528 + 512)[lane / 8] = raw.live;
        opus::static_for<8>([&](auto j) {
            opus::vector_t<bf16_t, 8> values{};
            if(raw.live)
            {
                unsigned code = (raw.scales[int(j) / 4] >> ((int(j) % 4) * 8)) & 255u;
                float scale   = code == 0     ? 0x1p-127f
                                : code == 255 ? __builtin_nanf("")
                                              : __uint_as_float(code << 23);
                opus::static_for<2>([&](auto half) {
                    // Materialize the scalar: clang 22 bit-casting a vector
                    // element lvalue directly repeated word 0 for word 1.
                    uint32_t word = raw.bytes[int(j)][int(half)];
                    auto fp8      = __builtin_bit_cast(opus::fp8x4_t, word);
                    auto x        = opus::cast<float>(fp8);
                    opus::static_for<4>([&](auto k) {
                        values[int(half) * 4 + int(k)] = static_cast<bf16_t>(x[int(k)] * scale);
                    });
                });
            }
            *reinterpret_cast<OPUS_LDS_ADDR opus::vector_t<bf16_t, 8>*>(
                lds + (int(j) * 4 + wave) * 528 + lane * 8) = values;
        });
    }
    template <class T, class G, class S, class Pointer>
    __device__ __forceinline__ void
    load_tile(int tile, Pointer dst, const G& ug, const S& us, int wave, int lane) const
    {
        const int i     = tile * 32 + (lane / 8) * 4 + (wave % 4);
        const int64_t r = row(i);
        const bool live = valid_row(r);
        auto* flag_lds  = reinterpret_cast<OPUS_LDS_ADDR bf16_t*>(reinterpret_cast<uintptr_t>(dst));
        if(wave < 4 && (lane & 7) == 0)
            reinterpret_cast<OPUS_LDS_ADDR uint32_t*>(flag_lds + wave * 528 + 512)[lane / 8] = live;
        if constexpr(!Packed)
        {
            const bf16_t* src = live ? static_cast<const bf16_t*>(pool) + r * 512 : fused_zero_kv;
            global_load<T::VEC_KV>(src, dst, ug, us);
        }
        else
        {
            const auto go   = opus::layout_to_offsets<8>(ug);
            const auto so   = opus::layout_to_offsets<8>(us);
            constexpr int N = opus::layout_load_traits<G, 8>::r_elem.value;
            auto* lds = reinterpret_cast<OPUS_LDS_ADDR bf16_t*>(reinterpret_cast<uintptr_t>(dst));
#pragma unroll 4
            for(int j = 0; j < N; ++j)
            {
                opus::vector_t<bf16_t, 8> values{};
                if(live)
                {
                    const uint8_t* src  = cache.data<PageSize>(pool, r);
                    const int d         = (j * 64 + (lane % 8) * 8);
                    const unsigned code = cache.scale<PageSize>(pool, r)[d / 32];
                    const float scale   = code == 0     ? 0x1p-127f
                                          : code == 255 ? __builtin_nanf("")
                                                        : __uint_as_float(code << 23);
                    opus::static_for<2>([&](auto half) {
                        // Rows and every vector are 4-byte aligned in ATOM's pool.
                        const auto bits =
                            *reinterpret_cast<const opus::fp8x4_t*>(src + d + int(half) * 4);
                        const auto decoded = opus::cast<float>(bits);
                        opus::static_for<4>([&](auto k) {
                            values[int(half) * 4 + int(k)] =
                                static_cast<bf16_t>(decoded[int(k)] * scale);
                        });
                    });
                }
                // GLOBAL_LOAD_LDS implicitly adds lane*8; explicit LDS stores must too.
                *reinterpret_cast<OPUS_LDS_ADDR opus::vector_t<bf16_t, 8>*>(
                    lds + ((j * 4 + wave) * 528) + lane * 8) = values;
            }
        }
    }
};

namespace opus_mla_v4_prefill_a16w16_16mx8_32nx1 {
template <class Traits, class Source>
__device__ void packed_decode_accum64(
    const Source& source,
    int valid_kv_len,
    int num_kv_tiles,
    char* smem_kv_buf,
    opus::vector_t<typename Traits::D_ATTN,
                   Traits::Q_TILE_SIZE * Traits::D_TILE_SIZE / Traits::WARP_SIZE>& v_q,
    opus::vector_t<typename Traits::D_ACC, 8> (&v_o)[16],
    typename Traits::D_ACC& actual_max,
    typename Traits::D_ACC& m_row,
    typename Traits::D_ACC& l_row,
    typename Traits::D_ACC temperature_scale)
{
    using namespace opus;
    using T      = opus::remove_cvref_t<Traits>;
    using D_ATTN = typename T::D_ATTN;
    using D_ACC  = typename T::D_ACC;

    int lane_id = thread_id_x() % T::WARP_SIZE;
    asm volatile("" : "+v"(lane_id)); // break CSE
    lane_id &= T::WARP_SIZE - 1;      // Restore range information after the opaque asm.
    const int warp_id =
        __builtin_amdgcn_readfirstlane((thread_id_x() / T::WARP_SIZE) % T::NUM_WARPS);

    // CSR bounds and int64 packed byte addresses are owned by Source.

    auto s_kv = make_smem(reinterpret_cast<D_ATTN*>(smem_kv_buf));

    auto mma0 =
        make_tiled_mma<D_ATTN, D_ATTN, D_ACC>(seq<T::GEMM0_E_M, T::GEMM0_E_N, T::GEMM0_E_K>{},
                                              seq<T::T_M, T::T_N, T::T_K>{},
                                              seq<T::W_M, T::W_N, T::W_K>{},
                                              mfma_adaptor_swap_ab{});
    auto mma1 =
        make_tiled_mma<D_ATTN, D_ATTN, D_ACC>(seq<T::GEMM1_E_M, T::GEMM1_E_N, T::GEMM1_E_K>{},
                                              seq<T::T_M, T::T_N, T::T_K>{},
                                              seq<T::W_M, T::W_N, T::W_K>{},
                                              mfma_adaptor_swap_ab{});

    auto u_gkv        = make_layout_gkv<T>(warp_id, lane_id);
    auto u_skv        = make_layout_skv<T>(warp_id);
    auto u_rk         = make_layout_rk<T>(lane_id);
    auto u_rv         = make_layout_rv<T>(lane_id);
    auto u_kv_indices = make_layout_kv_indices<T>(warp_id, lane_id);

    typename decltype(mma0)::vtype_b v_k[2];
    typename decltype(mma0)::vtype_c v_s;
    typename decltype(mma1)::vtype_a v_p;
    typename decltype(mma1)::vtype_b v_v[2];
    auto v_q_slices =
        reinterpret_cast<vector_t<D_ATTN, T::Q_TILE_SIZE * T::SLICE_D / T::WARP_SIZE>*>(&v_q);
    auto* v_o_slices = v_o;

    constexpr index_t s_len = vector_traits<typename decltype(mma0)::vtype_c>::size();

    auto skv_slice = [](auto slice_idx) {
        constexpr int s = decltype(slice_idx)::value;
        return number<(s / 2) * T::smem_n_rpt*(T::smem_linear_wave + T::smem_padding_32B) +
                      (s % 2) * T::SLICE_D>{};
    };

    int qk_tile      = 0;
    auto mask_scores = [&](auto& s) {
        // Materialize validity addresses only when scores are masked.
        int mask_lane = lane_id;
        asm volatile("" : "+v"(mask_lane));
        mask_lane &= 63;
        static_for<2>([&](auto n) {
            int first = int(n) * 16 + (mask_lane / 16) * 4;
            static_for<4>([&](auto j) {
                int k          = first + int(j);
                auto* flag_lds = reinterpret_cast<OPUS_LDS_ADDR const D_ATTN*>(
                    reinterpret_cast<uintptr_t>(s_kv.ptr));
                if(!reinterpret_cast<OPUS_LDS_ADDR const uint32_t*>(flag_lds + (k % 4) * 528 +
                                                                    512)[k / 4])
                    s[int(n) * 4 + int(j)] = -INFINITY;
            });
        });
        ++qk_tile;
    };
    auto compute_qk = [&](auto& s, const auto& q, auto& k) {
        clear(s);
        static_for<T::NUM_D_SLICES>([&](auto i) {
            constexpr int idx  = i.value;
            constexpr int slot = idx & 1;
            s                  = mma0(q[idx], k[slot], s);
            if constexpr(idx + 2 < T::NUM_D_SLICES)
            {
                k[slot] = load<T::VEC_KV>(s_kv, u_rk + skv_slice(number<idx + 2>{}));
                s_waitcnt_lgkmcnt(number<T::k_ds_read_insts>{});
            }
            else if constexpr(idx + 1 < T::NUM_D_SLICES)
            {
                s_waitcnt_lgkmcnt(0_I);
            }
        });
        mask_scores(s);
    };

    auto compute_pv = [&](const auto& p, auto& v, auto& o, int kv_tile_idx) {
        static_for<T::NUM_D_SLICES - 2>([&](auto i) {
            constexpr int idx  = i.value;
            constexpr int slot = idx & 1;
            o[idx]             = mma1(p, v[slot], o[idx]);
            v[slot]            = tr_load<T::VEC_TR_V>(s_kv, u_rv + skv_slice(number<idx + 2>{}));
            s_waitcnt_lgkmcnt(number<T::v_ds_read_insts>{});
            // Partial waits rely on LDS issue order and on which MFMA consumes
            // the completed reads. Allow VALU across this boundary, but keep
            // LDS, MFMA and SALU (including s_waitcnt) ordered. Allowing MFMA
            // or SALU here makes a later slice consume still-pending reads.
            // See experiments/20261014/PV_SCHEDULING.md for ISA and regressions.
            __builtin_amdgcn_sched_barrier(VALU_MASK);
        });
        o[T::NUM_D_SLICES - 2] = mma1(p, v[(T::NUM_D_SLICES - 2) & 1], o[T::NUM_D_SLICES - 2]);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_sched_barrier(0);
        o[T::NUM_D_SLICES - 1] = mma1(p, v[(T::NUM_D_SLICES - 1) & 1], o[T::NUM_D_SLICES - 1]);
    };

    const opus::u32_t neg_inf_v =
        std::bit_cast<opus::u32_t>(-opus::numeric_limits<D_ACC>::infinity());

    auto other_kv = make_smem(reinterpret_cast<D_ATTN*>(smem_kv_buf) + T::smem_kv_tile_elems);
    source.template load_tile<T>(0, s_kv.ptr, u_gkv, u_skv, warp_id, lane_id);
    s_waitcnt_vmcnt(0_I);
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    for(int tile_idx = 0; tile_idx < num_kv_tiles; ++tile_idx)
    {
        typename Source::RawTile next_raw;
        if constexpr(Source::is_packed)
        {
            if(tile_idx + 1 < num_kv_tiles)
                next_raw = source.prefetch(tile_idx + 1, warp_id, lane_id);
        }
        __builtin_amdgcn_sched_barrier(0);

        v_k[0] = load<T::VEC_KV>(s_kv, u_rk);
        v_k[1] = load<T::VEC_KV>(s_kv, u_rk + skv_slice(1_I));
        s_waitcnt_lgkmcnt(number<T::k_ds_read_insts>{});
        compute_qk(v_s, v_q_slices, v_k);
        if(tile_idx + 1 < num_kv_tiles)
        {
            if constexpr(Source::is_packed)
                source.commit(next_raw, other_kv.ptr, warp_id, lane_id);
            else
                source.template load_tile<T>(
                    tile_idx + 1, other_kv.ptr, u_gkv, u_skv, warp_id, lane_id);
        }

        D_ACC row_max = track_max<T>(v_s, temperature_scale, actual_max);
        bool below =
            (__builtin_amdgcn_ballot_w64(row_max - m_row <= 8.0f) == __builtin_amdgcn_read_exec());
        row_max         = below ? m_row : max(m_row, row_max);
        D_ACC rescale_m = __builtin_amdgcn_exp2f(m_row - row_max);
        m_row           = row_max;
        attn_row_scale_sub<T>(v_s, temperature_scale, row_max);
        attn_exp2_slice<T, 0, s_len>(v_s);
        l_row *= rescale_m;
        l_row += attn_row_sum<T>(v_s);
        v_p = cast<D_ATTN>(v_s);
        if(!below)
            scale_output_tile<T>(v_o, rescale_m);

        v_v[0] = tr_load<T::VEC_TR_V>(s_kv, u_rv);
        v_v[1] = tr_load<T::VEC_TR_V>(s_kv, u_rv + skv_slice(1_I));
        s_waitcnt_lgkmcnt(number<T::v_ds_read_insts>{});
        __builtin_amdgcn_sched_barrier(0);
        compute_pv(v_p, v_v, v_o_slices, tile_idx);
        s_waitcnt_vmcnt(0_I);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        std::swap(s_kv, other_kv);
    }
}

} // namespace opus_mla_v4_prefill_a16w16_16mx8_32nx1

template <bool Packed, bool BoundedDecode, int PageSize = 0>
struct PackedDecodeSource128
{
    static constexpr bool is_packed = Packed;
    const void* pool;
    const void* indices;
    int length;
    int64_t extent; // bytes for packed, rows for BF16
    Fp8CacheLayout cache{};

    __device__ __forceinline__ int64_t row(int i) const
    {
        if(i < 0 || i >= length)
            return -1;
        if constexpr(Packed)
            return cache.row<PageSize>(indices, i);
        else
            return static_cast<const int*>(indices)[i];
    }
    __device__ __forceinline__ bool valid_row(int64_t r) const
    {
        if constexpr(Packed)
            return cache.valid<PageSize>(r, extent);
        else
            return r >= 0 && r < extent;
    }
    __device__ __forceinline__ bool valid(int i) const { return valid_row(row(i)); }

    // The public byte-address contract guarantees 4-byte row alignment.
    // Explicit scalar fields let LLVM form wide loads without assuming 16-byte alignment.
    struct Words2
    {
        uint32_t x, y;
    };
    struct Words4
    {
        uint32_t x, y, z, w;
    };
    struct RawTile
    {
        opus::vector_t<uint32_t, 2> bytes[4];
        uint32_t scales; // Four group codes consumed by this lane, one per j.
        bool live;
    };
    __device__ __forceinline__ RawTile prefetch(int tile, int wave, int lane) const
    {
        // Materialize lane addresses in the load stage, not across QK/PV.
        asm volatile("" : "+v"(lane));
        lane &= 63;
        RawTile tile_data{};
        const int64_t r = row(tile * 32 + (lane / 8) * 4 + (wave % 4));
        tile_data.live  = valid_row(r);
        if(tile_data.live)
        {
            const auto* src    = cache.data<PageSize>(pool, r);
            const auto* scales = reinterpret_cast<const Words4*>(cache.scale<PageSize>(pool, r));
            // Each lane uses one scale from each four-group chunk.
            // Select before QK/PV so only one scale word remains live.
            const unsigned shift = ((wave / 4) * 2 + (lane % 8) / 4) * 8;
            tile_data.scales =
                ((scales->x >> shift) & 255u) | (((scales->y >> shift) & 255u) << 8) |
                (((scales->z >> shift) & 255u) << 16) | (((scales->w >> shift) & 255u) << 24);
            opus::static_for<4>([&](auto j) {
                const auto* values = reinterpret_cast<const Words2*>(
                    src + int(j) * 128 + (wave / 4) * 64 + (lane % 8) * 8);
                tile_data.bytes[int(j)] = {values->x, values->y};
            });
        }
        return tile_data;
    }
    template <class Pointer>
    __device__ __forceinline__ void
    commit(const RawTile& raw, Pointer dst, int wave, int lane) const
    {
        // Keep decode/store addresses local to the commit stage.
        asm volatile("" : "+v"(lane));
        lane &= 63;
        auto* lds = reinterpret_cast<OPUS_LDS_ADDR bf16_t*>(reinterpret_cast<uintptr_t>(dst));
        if(wave < 4 && (lane & 7) == 0)
            reinterpret_cast<OPUS_LDS_ADDR uint32_t*>(lds + wave * 528 + 512)[lane / 8] = raw.live;
        opus::static_for<4>([&](auto j) {
            opus::vector_t<bf16_t, 8> values{};
            if(raw.live)
            {
                unsigned code = (raw.scales >> (int(j) * 8)) & 255u;
                float scale   = code == 0     ? 0x1p-127f
                                : code == 255 ? __builtin_nanf("")
                                              : __uint_as_float(code << 23);
                opus::static_for<2>([&](auto half) {
                    // Materialize the scalar: clang 22 bit-casting a vector
                    // element lvalue directly repeated word 0 for word 1.
                    uint32_t word = raw.bytes[int(j)][int(half)];
                    auto fp8      = __builtin_bit_cast(opus::fp8x4_t, word);
                    auto x        = opus::cast<float>(fp8);
                    opus::static_for<4>([&](auto k) {
                        values[int(half) * 4 + int(k)] = static_cast<bf16_t>(x[int(k)] * scale);
                    });
                });
            }
            *reinterpret_cast<OPUS_LDS_ADDR opus::vector_t<bf16_t, 8>*>(
                lds + (int(j) * 8 + wave) * 528 + lane * 8) = values;
            __builtin_amdgcn_sched_barrier(0);
        });
    }
    template <class T, class G, class S, class Pointer>
    __device__ __forceinline__ void
    load_tile(int tile, Pointer dst, const G& ug, const S& us, int wave, int lane) const
    {
        const int i     = tile * 32 + (lane / 8) * 4 + (wave % 4);
        const int64_t r = row(i);
        const bool live = valid_row(r);
        auto* flags     = reinterpret_cast<OPUS_LDS_ADDR bf16_t*>(reinterpret_cast<uintptr_t>(dst));
        if(wave < 4 && (lane & 7) == 0)
            reinterpret_cast<OPUS_LDS_ADDR uint32_t*>(flags + wave * 528 + 512)[lane / 8] = live;
        if constexpr(!Packed)
        {
            const bf16_t* src = live ? static_cast<const bf16_t*>(pool) + r * 512 : fused_zero_kv;
            global_load<T::VEC_KV>(src, dst, ug, us);
        }
        else if constexpr(BoundedDecode)
        {
            constexpr int N = opus::layout_load_traits<G, 8>::r_elem.value;
            static_assert(T::NUM_WARPS == 8 && T::KV_TILE_SIZE == 32 && N == 4);
            auto* lds = reinterpret_cast<OPUS_LDS_ADDR bf16_t*>(reinterpret_cast<uintptr_t>(dst));
// Keep one decoded vector live at a time. With the H128 PV-tail
// load schedule, larger unroll factors cost registers and latency.
#pragma unroll 1
            for(int j = 0; j < N; ++j)
            {
                opus::vector_t<bf16_t, 8> values{};
                if(live)
                {
                    const uint8_t* src  = cache.data<PageSize>(pool, r);
                    const int d         = j * 128 + (wave / 4) * 64 + (lane % 8) * 8;
                    const unsigned code = cache.scale<PageSize>(pool, r)[d / 32];
                    const float scale   = code == 0     ? 0x1p-127f
                                          : code == 255 ? __builtin_nanf("")
                                                        : __uint_as_float(code << 23);
                    opus::static_for<2>([&](auto half) {
                        // Rows and every vector are 4-byte aligned in ATOM's pool.
                        const auto bits =
                            *reinterpret_cast<const opus::fp8x4_t*>(src + d + int(half) * 4);
                        const auto decoded = opus::cast<float>(bits);
                        opus::static_for<4>([&](auto k) {
                            values[int(half) * 4 + int(k)] =
                                static_cast<bf16_t>(decoded[int(k)] * scale);
                        });
                    });
                }
                // GLOBAL_LOAD_LDS implicitly adds lane*8; explicit LDS stores must too.
                *reinterpret_cast<OPUS_LDS_ADDR opus::vector_t<bf16_t, 8>*>(
                    lds + (j * 8 + wave) * 528 + lane * 8) = values;
                __builtin_amdgcn_sched_barrier(0);
            }
        }
        else
        {
            const auto go   = opus::layout_to_offsets<8>(ug);
            const auto so   = opus::layout_to_offsets<8>(us);
            constexpr int N = opus::layout_load_traits<G, 8>::r_elem.value;
            auto* lds = reinterpret_cast<OPUS_LDS_ADDR bf16_t*>(reinterpret_cast<uintptr_t>(dst));
            opus::static_for<N>([&](auto j) {
                opus::vector_t<bf16_t, 8> values{};
                if(live)
                {
                    const uint8_t* src  = cache.data<PageSize>(pool, r);
                    const int d         = go[j.value];
                    const unsigned code = cache.scale<PageSize>(pool, r)[d / 32];
                    const float scale   = code == 0     ? 0x1p-127f
                                          : code == 255 ? __builtin_nanf("")
                                                        : __uint_as_float(code << 23);
                    opus::static_for<2>([&](auto half) {
                        // Rows and every vector are 4-byte aligned in ATOM's pool.
                        const auto bits =
                            *reinterpret_cast<const opus::fp8x4_t*>(src + d + int(half) * 4);
                        const auto decoded = opus::cast<float>(bits);
                        opus::static_for<4>([&](auto k) {
                            values[int(half) * 4 + int(k)] =
                                static_cast<bf16_t>(decoded[int(k)] * scale);
                        });
                    });
                }
                // GLOBAL_LOAD_LDS implicitly adds lane*8; explicit LDS stores must too.
                *reinterpret_cast<OPUS_LDS_ADDR opus::vector_t<bf16_t, 8>*>(lds + so[j.value] +
                                                                            lane * 8) = values;
                __builtin_amdgcn_sched_barrier(0);
            });
        }
    }
};

namespace opus_mla_v4_prefill_a16w16_16mx8_32nx1 {
template <class Traits, bool LateLoad, class Source>
__device__ void packed_decode_accum128(
    const Source& source,
    int valid_kv_len,
    int num_kv_tiles,
    char* smem_kv_buf,
    opus::vector_t<typename Traits::D_ATTN,
                   Traits::Q_TILE_SIZE * Traits::D_TILE_SIZE / Traits::WARP_SIZE>& v_q,
    opus::vector_t<typename Traits::D_ACC, 8> (&v_o)[16],
    typename Traits::D_ACC& actual_max,
    typename Traits::D_ACC& m_row,
    typename Traits::D_ACC& l_row,
    typename Traits::D_ACC temperature_scale)
{
    using namespace opus;
    using T      = opus::remove_cvref_t<Traits>;
    using D_ATTN = typename T::D_ATTN;
    using D_ACC  = typename T::D_ACC;

    int lane_id = thread_id_x() % T::WARP_SIZE;
    asm volatile("" : "+v"(lane_id)); // break CSE
    lane_id &= 63;                    // Restore the lane range hidden by the opaque asm.
    const int warp_id =
        __builtin_amdgcn_readfirstlane((thread_id_x() / T::WARP_SIZE) % T::NUM_WARPS);

    // CSR bounds and int64 packed byte addresses are owned by Source.

    auto s_kv = make_smem(reinterpret_cast<D_ATTN*>(smem_kv_buf));

    auto mma0 =
        make_tiled_mma<D_ATTN, D_ATTN, D_ACC>(seq<T::GEMM0_E_M, T::GEMM0_E_N, T::GEMM0_E_K>{},
                                              seq<T::T_M, T::T_N, T::T_K>{},
                                              seq<T::W_M, T::W_N, T::W_K>{},
                                              mfma_adaptor_swap_ab{});
    auto mma1 =
        make_tiled_mma<D_ATTN, D_ATTN, D_ACC>(seq<T::GEMM1_E_M, T::GEMM1_E_N, T::GEMM1_E_K>{},
                                              seq<T::T_M, T::T_N, T::T_K>{},
                                              seq<T::W_M, T::W_N, T::W_K>{},
                                              mfma_adaptor_swap_ab{});

    auto u_gkv        = make_layout_gkv<T>(warp_id, lane_id);
    auto u_skv        = make_layout_skv<T>(warp_id);
    auto u_rk         = make_layout_rk<T>(lane_id);
    auto u_rv         = make_layout_rv<T>(lane_id);
    auto u_kv_indices = make_layout_kv_indices<T>(warp_id, lane_id);

    typename decltype(mma0)::vtype_b v_k[2];
    typename decltype(mma0)::vtype_c v_s;
    typename decltype(mma1)::vtype_a v_p;
    typename decltype(mma1)::vtype_b v_v[2];
    auto v_q_slices =
        reinterpret_cast<vector_t<D_ATTN, T::Q_TILE_SIZE * T::SLICE_D / T::WARP_SIZE>*>(&v_q);
    auto* v_o_slices = v_o;

    constexpr index_t s_len = vector_traits<typename decltype(mma0)::vtype_c>::size();

    auto skv_slice = [](auto slice_idx) {
        constexpr int s = decltype(slice_idx)::value;
        return number<(s / 2) * T::smem_n_rpt*(T::smem_linear_wave + T::smem_padding_32B) +
                      (s % 2) * T::SLICE_D>{};
    };

    int qk_tile      = 0;
    auto mask_scores = [&](auto& s) {
        // Materialize validity addresses only when scores are masked.
        int mask_lane = lane_id;
        asm volatile("" : "+v"(mask_lane));
        mask_lane &= 63;
        static_for<2>([&](auto n) {
            int first = int(n) * 16 + (mask_lane / 16) * 4;
            static_for<4>([&](auto j) {
                int k          = first + int(j);
                auto* flag_lds = reinterpret_cast<OPUS_LDS_ADDR const D_ATTN*>(
                    reinterpret_cast<uintptr_t>(s_kv.ptr));
                if(!reinterpret_cast<OPUS_LDS_ADDR const uint32_t*>(flag_lds + (k % 4) * 528 +
                                                                    512)[k / 4])
                    s[int(n) * 4 + int(j)] = -INFINITY;
            });
        });
        ++qk_tile;
    };
    auto compute_qk = [&](auto& s, const auto& q, auto& k) {
        clear(s);
        static_for<T::NUM_D_SLICES>([&](auto i) {
            constexpr int idx  = i.value;
            constexpr int slot = idx & 1;
            s                  = mma0(q[idx], k[slot], s);
            if constexpr(idx + 2 < T::NUM_D_SLICES)
            {
                k[slot] = load<T::VEC_KV>(s_kv, u_rk + skv_slice(number<idx + 2>{}));
                s_waitcnt_lgkmcnt(number<T::k_ds_read_insts>{});
            }
            else if constexpr(idx + 1 < T::NUM_D_SLICES)
            {
                s_waitcnt_lgkmcnt(0_I);
            }
        });
        mask_scores(s);
    };

    auto compute_pv = [&](const auto& p, auto& v, auto& o, int kv_tile_idx) {
        static_for<T::NUM_D_SLICES - 2>([&](auto i) {
            constexpr int idx  = i.value;
            constexpr int slot = idx & 1;
            o[idx]             = mma1(p, v[slot], o[idx]);
            v[slot]            = tr_load<T::VEC_TR_V>(s_kv, u_rv + skv_slice(number<idx + 2>{}));
            s_waitcnt_lgkmcnt(number<T::v_ds_read_insts>{});
            // Partial waits rely on LDS issue order and on which MFMA consumes
            // the completed reads. Allow VALU across this boundary, but keep
            // LDS, MFMA and SALU (including s_waitcnt) ordered. Allowing MFMA
            // or SALU here makes a later slice consume still-pending reads.
            // See experiments/20261014/PV_SCHEDULING.md for ISA and regressions.
            __builtin_amdgcn_sched_barrier(VALU_MASK);
        });
        o[T::NUM_D_SLICES - 2] = mma1(p, v[(T::NUM_D_SLICES - 2) & 1], o[T::NUM_D_SLICES - 2]);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_sched_barrier(0);
        o[T::NUM_D_SLICES - 1] = mma1(p, v[(T::NUM_D_SLICES - 1) & 1], o[T::NUM_D_SLICES - 1]);
    };

    const opus::u32_t neg_inf_v =
        std::bit_cast<opus::u32_t>(-opus::numeric_limits<D_ACC>::infinity());

    auto other_kv = make_smem(reinterpret_cast<D_ATTN*>(smem_kv_buf) + T::smem_kv_tile_elems);
    source.template load_tile<T>(0, s_kv.ptr, u_gkv, u_skv, warp_id, lane_id);
    s_waitcnt_vmcnt(0_I);
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    for(int tile_idx = 0; tile_idx < num_kv_tiles; ++tile_idx)
    {
        typename Source::RawTile next_raw;
        if constexpr(Source::is_packed)
        {
            if(tile_idx + 1 < num_kv_tiles)
                next_raw = source.prefetch(tile_idx + 1, warp_id, lane_id);
        }
        __builtin_amdgcn_sched_barrier(0);

        v_k[0] = load<T::VEC_KV>(s_kv, u_rk);
        v_k[1] = load<T::VEC_KV>(s_kv, u_rk + skv_slice(1_I));
        s_waitcnt_lgkmcnt(number<T::k_ds_read_insts>{});
        compute_qk(v_s, v_q_slices, v_k);
        if constexpr(!Source::is_packed)
        {
            if(tile_idx + 1 < num_kv_tiles)
                source.template load_tile<T>(
                    tile_idx + 1, other_kv.ptr, u_gkv, u_skv, warp_id, lane_id);
        }

        D_ACC row_max = track_max<T>(v_s, temperature_scale, actual_max);
        bool below =
            (__builtin_amdgcn_ballot_w64(row_max - m_row <= 8.0f) == __builtin_amdgcn_read_exec());
        row_max         = below ? m_row : max(m_row, row_max);
        D_ACC rescale_m = __builtin_amdgcn_exp2f(m_row - row_max);
        m_row           = row_max;
        attn_row_scale_sub<T>(v_s, temperature_scale, row_max);
        attn_exp2_slice<T, 0, s_len>(v_s);
        l_row *= rescale_m;
        l_row += attn_row_sum<T>(v_s);
        v_p = cast<D_ATTN>(v_s);
        if(!below)
            scale_output_tile<T>(v_o, rescale_m);

        v_v[0] = tr_load<T::VEC_TR_V>(s_kv, u_rv);
        v_v[1] = tr_load<T::VEC_TR_V>(s_kv, u_rv + skv_slice(1_I));
        s_waitcnt_lgkmcnt(number<T::v_ds_read_insts>{});
        __builtin_amdgcn_sched_barrier(0);
        compute_pv(v_p, v_v, v_o_slices, tile_idx);
        if(tile_idx + 1 < num_kv_tiles)
        {
            if constexpr(Source::is_packed)
                source.commit(next_raw, other_kv.ptr, warp_id, lane_id);
        }

        s_waitcnt_vmcnt(0_I);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        std::swap(s_kv, other_kv);
    }
}

} // namespace opus_mla_v4_prefill_a16w16_16mx8_32nx1
