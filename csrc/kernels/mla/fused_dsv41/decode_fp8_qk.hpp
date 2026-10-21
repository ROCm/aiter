// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include "decode_prefetch.hpp"
#include <type_traits>

// MFMA consumes two 16-byte pieces per lane. Group32 scales are supplied
// by lanes in group order, while values are interleaved across pairs of lanes.
// Quantize a complete group locally, then permute only its packed FP8 words.
__device__ __forceinline__ opus::vector_t<opus::fp8_t, 32>
pack_fp8_qk_group(opus::vector_t<opus::fp8_t, 32> values, int lane)
{
    auto words = __builtin_bit_cast(opus::vector_t<uint32_t, 8>, values);
    opus::vector_t<uint32_t, 8> packed;
    const int group = lane / 16;
    opus::static_for<2>([&](auto half) {
        const int src = lane % 16 + (group / 2 + int(half) * 2) * 16;
        opus::static_for<4>([&](auto j) {
            const uint32_t lo = __builtin_amdgcn_ds_bpermute(src * 4, words[int(j)]);
            const uint32_t hi = __builtin_amdgcn_ds_bpermute(src * 4, words[int(j) + 4]);
            packed[int(half) * 4 + int(j)] = (group & 1) ? hi : lo;
        });
    });
    return __builtin_bit_cast(opus::vector_t<opus::fp8_t, 32>, packed);
}

// Two FP8 K slots carry values, group32 scales and 32 validity flags each.
// A single BF16 V slot is materialized from the current K slot before PV.
// CTA barriers bracket V use so another wave cannot overwrite live PV inputs.
template <int PageSize>
struct Fp8QkSource64 : PackedDecodeSource64<true, PageSize>
{
    using Base = PackedDecodeSource64<true, PageSize>;
    // Preserve int32 paged indices until the next iteration consumes them.
    // Extending an async int32 load immediately would force a VMEM wait before QK.
    using Index = std::conditional_t<(PageSize != 0), int32_t, int64_t>;
    __device__ __forceinline__ Index prefetch_index(int i) const
    {
        if(i < 0 || i >= this->length)
            return Index{-1};
        return static_cast<const Index*>(this->indices)[i];
    }
    using typename Base::Words4;
    struct RawTile
    {
        opus::vector_t<uint32_t, 4> bytes[4];
        opus::vector_t<uint32_t, 4> scales;
        bool live;
    };
    __device__ __forceinline__ RawTile prefetch_row(int64_t r, int wave, int lane) const
    {
        asm volatile("" : "+v"(lane));
        lane &= 63;
        RawTile tile_data;
        tile_data.live = this->valid_row(r);
        // A device-constant zero row keeps invalid/tail loads defined without
        // merging zero-filled vector PHIs before QK. commit_raw supplies scale 127.
        const auto* zero = reinterpret_cast<const uint8_t*>(fused_zero_kv);
        const auto* src =
            tile_data.live ? this->cache.template data<PageSize>(this->pool, r) : zero;
        const auto* scales = reinterpret_cast<const Words4*>(
            tile_data.live ? this->cache.template scale<PageSize>(this->pool, r) : zero);
        // Keep the original words live across QK; packing scales here serializes
        // the prefetch on their VMEM completion.
        tile_data.scales = {scales->x, scales->y, scales->z, scales->w};
        // Four 16-byte reads preserve the public 4-byte row-alignment contract:
        // Words4 has scalar uint32_t fields, rather than a 16-byte-aligned pointer.
        opus::static_for<4>([&](auto j) {
            const auto* values =
                reinterpret_cast<const Words4*>(src + int(j) * 128 + (lane % 8) * 16);
            tile_data.bytes[int(j)] = {values->x, values->y, values->z, values->w};
        });
        return tile_data;
    }
    __device__ __forceinline__ RawTile prefetch(int tile, int wave, int lane) const
    {
        return prefetch_row(this->row(tile * 32 + (lane / 8) * 4 + wave), wave, lane);
    }
    template <class RawPointer>
    __device__ __forceinline__ void
    commit_raw(const RawTile& raw, RawPointer raw_dst, int wave, int lane) const
    {
        auto* bytes =
            reinterpret_cast<OPUS_LDS_ADDR uint8_t*>(reinterpret_cast<uintptr_t>(raw_dst));
        const int row = (lane / 8) * 4 + wave;
        opus::static_for<4>([&](auto j) {
            const int d = int(j) * 128 + (lane % 8) * 16;
            *reinterpret_cast<OPUS_LDS_ADDR opus::vector_t<uint32_t, 4>*>(bytes + row * 528 + d) =
                raw.bytes[int(j)];
        });
        if((lane & 7) == 0)
        {
            *reinterpret_cast<OPUS_LDS_ADDR uint32_t*>(bytes + 32 * 528 + row * 4) = raw.live;
            const opus::vector_t<uint32_t, 4> codes =
                raw.live
                    ? raw.scales
                    : opus::vector_t<uint32_t, 4>{0x7f7f7f7f, 0x7f7f7f7f, 0x7f7f7f7f, 0x7f7f7f7f};
            *reinterpret_cast<OPUS_LDS_ADDR opus::vector_t<uint32_t, 4>*>(bytes + row * 528 + 512) =
                codes;
        }
    }
    template <class Pointer, class RawPointer>
    __device__ __forceinline__ void
    dequant_current(RawPointer raw_src, Pointer dst, int wave, int lane) const
    {
        auto* bytes =
            reinterpret_cast<OPUS_LDS_ADDR const uint8_t*>(reinterpret_cast<uintptr_t>(raw_src));
        const int row = (lane / 8) * 4 + wave;
        // Invalid rows were committed as zero bytes with scale 127, so they
        // dequantize to zero without a divergent validity branch. Load/convert
        // each 16-value piece directly: rebuilding RawTile prolongs register
        // lifetimes and crosses the measured residency threshold.
        auto* lds = reinterpret_cast<OPUS_LDS_ADDR bf16_t*>(reinterpret_cast<uintptr_t>(dst));
        opus::static_for<4>([&](auto j) {
            const int d = int(j) * 128 + (lane % 8) * 16;
            auto bits   = *reinterpret_cast<OPUS_LDS_ADDR const opus::vector_t<uint32_t, 4>*>(
                bytes + row * 528 + d);
            float scale = __uint_as_float(uint32_t(bytes[row * 528 + 512 + d / 32]) << 23);
            opus::vector_t<bf16_t, 16> values;
            opus::static_for<4>([&](auto half) {
                uint32_t word = bits[int(half)];
                auto lo       = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(word, scale, false);
                auto hi       = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(word, scale, true);
                auto* pairs   = reinterpret_cast<opus::vector_t<bf16_t, 2>*>(&values);
                pairs[int(half) * 2]     = __builtin_bit_cast(opus::vector_t<bf16_t, 2>, lo);
                pairs[int(half) * 2 + 1] = __builtin_bit_cast(opus::vector_t<bf16_t, 2>, hi);
            });
            auto* dst = lds + ((int(j) * 2 + (lane % 8) / 4) * 4 + wave) * 528 + (lane / 8) * 64 +
                        (lane % 4) * 16;
            const auto lo = opus::slice(values, opus::number<0>{}, opus::number<8>{});
            const auto hi = opus::slice(values, opus::number<8>{}, opus::number<16>{});
            // The two lane groups write opposite halves first to reduce LDS bank
            // overlap. Both stores retain exactly the original BF16 V layout.
            const bool flip = (lane & 4) != 0;
            *reinterpret_cast<OPUS_LDS_ADDR opus::vector_t<bf16_t, 8>*>(dst + (flip ? 8 : 0)) =
                flip ? hi : lo;
            *reinterpret_cast<OPUS_LDS_ADDR opus::vector_t<bf16_t, 8>*>(dst + (flip ? 0 : 8)) =
                flip ? lo : hi;
        });
    }
};

namespace opus_mla_v4_prefill_a16w16_16mx8_32nx1 {
template <class Traits, bool LateCommit = false, class Source>
__device__ void fp8_qk_decode_accum64(const Source& source,
                                      int valid_kv_len,
                                      int num_kv_tiles,
                                      char* smem_kv_buf,
                                      const opus::vector_t<opus::fp8_t, 32> (&v_q)[4],
                                      const int (&q_scales)[4],
                                      uint8_t* raw_lds,
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

    auto u_gkv = make_layout_gkv<T>(warp_id, lane_id);
    auto u_skv = make_layout_skv<T>(warp_id);
    auto u_rv  = make_layout_rv<T>(lane_id);

    typename decltype(mma0)::vtype_c v_s;
    typename decltype(mma1)::vtype_a v_p;
    typename decltype(mma1)::vtype_b v_v[3];
    auto* v_o_slices = v_o;

    constexpr index_t s_len = vector_traits<typename decltype(mma0)::vtype_c>::size();

    auto skv_slice = [](auto slice_idx) {
        constexpr int s = decltype(slice_idx)::value;
        return number<(s / 2) * T::smem_n_rpt*(T::smem_linear_wave + T::smem_padding_32B) +
                      (s % 2) * T::SLICE_D>{};
    };

    auto* raw_cur  = reinterpret_cast<OPUS_LDS_ADDR uint8_t*>(reinterpret_cast<uintptr_t>(raw_lds));
    auto* raw_next = raw_cur + 32 * 528 + 128;
    auto mask_scores = [&](auto& s) {
        // Materialize validity addresses only when scores are masked.
        int mask_lane = lane_id;
        asm volatile("" : "+v"(mask_lane));
        mask_lane &= 63;
        static_for<2>([&](auto n) {
            int first = int(n) * 16 + (mask_lane / 16) * 4;
            static_for<4>([&](auto j) {
                int k = first + int(j);
                if(!reinterpret_cast<OPUS_LDS_ADDR const uint32_t*>(raw_cur + 32 * 528)[k])
                    s[int(n) * 4 + int(j)] = -INFINITY;
            });
        });
    };
    // K is A and Q is B: lane%16 owns a head; lane/16 owns four scores.
    // Data uses two 16-byte pieces; scale lanes retain contiguous group32 order.

    auto compute_qk = [&](auto& s) {
        vector_t<float, 4> score[2]{};
        // Two K operand slots overlap the next LDS read with the current MFMA.
        // Each operand issues two ds_read_b128 and one scale-byte read. Keep
        // partial waits and scheduling fences paired; the compiler may insert
        // a stronger wait when it reorders the initial scale loads.
        vector_t<fp8_t, 32> keys[2];
        int scales[2];
        auto load_k = [&](auto index) {
            constexpr int i = int(index), slot = i % 2, step = i / 2, n = i % 2;
            const int row = n * 16 + lane_id % 16;
            static_for<2>([&](auto half) {
                const int kd = step * 128 + (lane_id / 16) * 16 + int(half) * 64;
                // Explicit DS reads keep LLVM from strengthening the pipelined
                // partial waits below. Immediate offsets avoid per-load VALU work.
                vector_t<uint32_t, 4> words;
                const uint32_t addr = uint32_t(reinterpret_cast<uintptr_t>(
                    raw_cur + (lane_id % 16) * 528 + (lane_id / 16) * 16));
                asm volatile("ds_read_b128 %0, %1 offset:%2"
                             : "=v"(words)
                             : "v"(addr), "n"(n * 16 * 528 + step * 128 + int(half) * 64)
                             : "memory");
                auto part = __builtin_bit_cast(vector_t<fp8_t, 16>, words);
                set_slice(
                    keys[slot], part, number<int(half) * 16>{}, number<int(half) * 16 + 16>{});
            });
            const uint32_t scale_addr = uint32_t(
                reinterpret_cast<uintptr_t>(raw_cur + (lane_id % 16) * 528 + lane_id / 16));
            uint32_t code;
            asm volatile("ds_read_u8 %0, %1 offset:%2"
                         : "=v"(code)
                         : "v"(scale_addr), "n"(n * 16 * 528 + 512 + step * 4)
                         : "memory");
            scales[slot] = code;
        };
        auto issue_qk = [&](auto index) {
            constexpr int i = int(index), slot = i % 2, step = i / 2, n = i % 2;
            auto kw  = __builtin_bit_cast(vector_t<uint32_t, 8>, keys[slot]);
            auto qw  = __builtin_bit_cast(vector_t<uint32_t, 8>, v_q[step]);
            auto acc = score[n];
            int ks = scales[slot], qs = q_scales[step];
            asm volatile("v_mfma_scale_f32_16x16x128_f8f6f4 %0, %1, %2, %0, %3, %4"
                         : "+v"(acc)
                         : "v"(kw), "v"(qw), "v"(ks), "v"(qs));
            score[n] = acc;
        };
        load_k(0_I);
        load_k(1_I);
        s_waitcnt_lgkmcnt(3_I);
        __builtin_amdgcn_sched_barrier(0);
        static_for<6>([&](auto i) {
            issue_qk(i);
            load_k(number<int(i) + 2>{});
            s_waitcnt_lgkmcnt(3_I);
            __builtin_amdgcn_sched_barrier(0);
        });
        issue_qk(6_I);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_sched_barrier(0);
        issue_qk(7_I);
        static_for<2>([&](auto n) {
            set_slice(s, score[int(n)], number<int(n) * 4>{}, number<int(n) * 4 + 4>{});
        });
        mask_scores(s);
    };

    // Explicit VGPR constraints retain O across QK/PV. The generic MFMA
    // builtin moved 128 O accumulators to/from AGPRs on each iteration with
    // ROCm 7.2.4. That compiler ignores amdgpu_pin_vgpr/agpr attributes.
    // Operand order matches mfma_adaptor_swap_ab: V is A and P is B.
    auto pv_vgpr = [](const auto& p, const auto& v, auto o) {
        static_assert(sizeof(p) == 16 && sizeof(v) == 32 && sizeof(o) == 32);
        auto pw = __builtin_bit_cast(vector_t<uint32_t, 4>, p);
        static_for<2>([&](auto n) {
            auto vw  = __builtin_bit_cast(vector_t<uint32_t, 4>,
                                         slice(v, number<int(n) * 8>{}, number<int(n) * 8 + 8>{}));
            auto acc = slice(o, number<int(n) * 4>{}, number<int(n) * 4 + 4>{});
            asm volatile("v_mfma_f32_16x16x32_bf16 %0, %1, %2, %0" : "+v"(acc) : "v"(vw), "v"(pw));
            set_slice(o, acc, number<int(n) * 4>{}, number<int(n) * 4 + 4>{});
        });
        return o;
    };
    // Keep three V operands in flight. Partial waits count the outstanding
    // LDS reads; the scheduling fences preserve their order relative to MFMA.
    auto compute_pv = [&](const auto& p, auto& v, auto& o, int kv_tile_idx) {
        static_for<T::NUM_D_SLICES - 3>([&](auto i) {
            constexpr int idx = int(i), slot = idx % 3;
            o[idx]  = pv_vgpr(p, v[slot], o[idx]);
            v[slot] = tr_load<T::VEC_TR_V>(s_kv, u_rv + skv_slice(number<idx + 3>{}));
            s_waitcnt_lgkmcnt(number<(3 - 1) * T::v_ds_read_insts>{});
            __builtin_amdgcn_sched_barrier(VALU_MASK);
        });
        static_for<3>([&](auto j) {
            constexpr int idx = T::NUM_D_SLICES - 3 + int(j);
            o[idx]            = pv_vgpr(p, v[idx % 3], o[idx]);
            if constexpr(int(j) + 1 < 3)
            {
                s_waitcnt_lgkmcnt(number<(3 - 2 - int(j)) * T::v_ds_read_insts>{});
                __builtin_amdgcn_sched_barrier(0);
            }
        });
    };

    if constexpr(Source::is_packed)
        source.commit_raw(source.prefetch(0, warp_id, lane_id), raw_cur, warp_id, lane_id);
    else
        source.template load_tile<T>(0, s_kv.ptr, u_gkv, u_skv, warp_id, lane_id);
    s_waitcnt_vmcnt(0_I);
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    auto next_row = source.prefetch_index(32 + (lane_id / 8) * 4 + warp_id);
    for(int tile_idx = 0; tile_idx < num_kv_tiles; ++tile_idx)
    {
        typename Source::RawTile next_raw;
        if constexpr(Source::is_packed)
        {
            auto current_row = next_row;
            asm volatile("" : "+v"(current_row));
            next_raw = source.prefetch_row(current_row, warp_id, lane_id);
            next_row = source.prefetch_index((tile_idx + 2) * 32 + (lane_id / 8) * 4 + warp_id);
        }
        __builtin_amdgcn_sched_barrier(0);

        compute_qk(v_s);
        // Keep next-tile VMEM waits after QK, so QK covers raw prefetch
        // latency. This is a compiler fence, not a CTA barrier.
        __builtin_amdgcn_sched_barrier(0);
        if constexpr(!LateCommit)
        {
            if(tile_idx + 1 < num_kv_tiles)
            {
                if constexpr(Source::is_packed)
                    source.commit_raw(next_raw, raw_next, warp_id, lane_id);
                else
                    source.template load_tile<T>(
                        tile_idx + 1, s_kv.ptr, u_gkv, u_skv, warp_id, lane_id);
            }
        }

        // Reuse the sole V slot only after the previous iteration's barrier.
        source.dequant_current(raw_cur, s_kv.ptr, warp_id, lane_id);

        D_ACC row_max = track_max<T>(v_s, temperature_scale, actual_max);
        bool below =
            (__builtin_amdgcn_ballot_w64(row_max - m_row <= 8.0f) == __builtin_amdgcn_read_exec());
        row_max         = below ? m_row : max(m_row, row_max);
        D_ACC rescale_m = __builtin_amdgcn_exp2f(m_row - row_max);
        m_row           = row_max;
        // temperature_scale already includes log2(e); row_max is in base-2 units.
        // Explicit FMA contracts the scale/subtract despite -ffp-contract=off.
        // One v_fma_f32 plus one v_exp_f32 per score; FMA changes FP32 rounding.
        static_for<s_len>([&](auto i) {
            v_s[int(i)] =
                __builtin_amdgcn_exp2f(__builtin_fmaf(v_s[int(i)], temperature_scale, -row_max));
        });
        l_row *= rescale_m;
        l_row += attn_row_sum<T>(v_s);
        v_p = cast<D_ATTN>(v_s);
        if(!below)
            scale_output_tile<T>(v_o, rescale_m);

        if constexpr(LateCommit)
        {
            if(tile_idx + 1 < num_kv_tiles)
            {
                if constexpr(Source::is_packed)
                    source.commit_raw(next_raw, raw_next, warp_id, lane_id);
                else
                    source.template load_tile<T>(
                        tile_idx + 1, s_kv.ptr, u_gkv, u_skv, warp_id, lane_id);
            }
        }

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier(); // All waves have finished writing V.
        static_for<3>(
            [&](auto j) { v_v[int(j)] = tr_load<T::VEC_TR_V>(s_kv, u_rv + skv_slice(j)); });
        s_waitcnt_lgkmcnt(number<(3 - 1) * T::v_ds_read_insts>{});
        __builtin_amdgcn_sched_barrier(0);
        compute_pv(v_p, v_v, v_o_slices, tile_idx);
        s_waitcnt_vmcnt(0_I);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier(); // All PV reads finish before V is reused.

        std::swap(raw_cur, raw_next);
    }
}

} // namespace opus_mla_v4_prefill_a16w16_16mx8_32nx1
