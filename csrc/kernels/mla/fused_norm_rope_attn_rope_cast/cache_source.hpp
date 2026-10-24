// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include "cache_layout.hpp"
#include "pipeline_common.hpp"
#include <cstdint>

// FP8 sources support legacy tagged addresses and independent paged data/scales.
// Extend ids address BF16 rows.
template <bool Packed, int PageSize = 0>
struct PackedCacheSource64
{
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

template <bool Packed, bool BoundedDecode, int PageSize = 0>
struct PackedCacheSource128
{
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

    template <class T, class G, class S, class Pointer>
    __device__ __forceinline__ void
    load_tile(int tile, Pointer dst, const G& ug, const S& us, int wave, int lane) const
    {
        const int i     = tile * 32 + (lane / 8) * 4 + (wave % 4);
        const int64_t r = row(i);
        const bool live = valid_row(r);
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

// Small-head path: staged FP8 prefetch and shared validity flags.
template <bool Packed, int PageSize = 0>
struct PackedCacheSourceSmall
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
        opus::vector_t<uint32_t, 4> scales;
        bool live;
    };
    __device__ __forceinline__ RawTile prefetch(int tile, int wave, int lane) const
    {
        RawTile tile_data{};
        const int64_t r = row(tile * 32 + (lane / 8) * 4 + (wave % 4));
        tile_data.live  = valid_row(r);
        if(tile_data.live)
        {
            const auto* src    = cache.data<PageSize>(pool, r);
            const auto* scales = reinterpret_cast<const Words4*>(cache.scale<PageSize>(pool, r));
            tile_data.scales   = {scales->x, scales->y, scales->z, scales->w};
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
        auto* lds = reinterpret_cast<OPUS_LDS_ADDR bf16_t*>(reinterpret_cast<uintptr_t>(dst));
        if((lane & 7) == 0)
            reinterpret_cast<OPUS_LDS_ADDR uint32_t*>(lds + wave * 528 + 512)[lane / 8] = raw.live;
        const auto scales = __builtin_bit_cast(opus::vector_t<uint8_t, 16>, raw.scales);
        opus::static_for<8>([&](auto j) {
            opus::vector_t<bf16_t, 8> values{};
            if(raw.live)
            {
                unsigned code = scales[int(j) * 2 + (lane % 8) / 4];
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
        // Eight live bits per wave fit the first panel's 32-byte LDS padding.
        // Store once while the address is in registers; every head reuses it.
        auto* flag_lds = reinterpret_cast<OPUS_LDS_ADDR bf16_t*>(reinterpret_cast<uintptr_t>(dst));
        if((lane & 7) == 0)
            reinterpret_cast<OPUS_LDS_ADDR uint32_t*>(flag_lds + (wave % 4) * 528 + 512)[lane / 8] =
                live;
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
            });
        }
    }
};
