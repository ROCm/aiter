// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include "decode_prefetch.hpp"

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
    template <class RawPointer>
    __device__ __forceinline__ void commit_raw(
        const typename Base::RawTile& raw, RawPointer raw_dst, int wave, int lane) const
    {
        auto* bytes = reinterpret_cast<OPUS_LDS_ADDR uint8_t*>(reinterpret_cast<uintptr_t>(raw_dst));
        const int row = (lane / 8) * 4 + wave;
        if((lane & 7) == 0)
            *reinterpret_cast<OPUS_LDS_ADDR uint32_t*>(bytes + 32 * 528 + row * 4) = raw.live;
        opus::static_for<8>([&](auto j) {
            const int d = int(j) * 64 + (lane % 8) * 8;
            *reinterpret_cast<OPUS_LDS_ADDR opus::vector_t<uint32_t, 2>*>(bytes + row * 528 + d) = raw.bytes[int(j)];
        });
        if((lane & 3) == 0)
            opus::static_for<8>([&](auto j) {
                const int d = int(j) * 64 + (lane % 8) * 8;
                bytes[row * 528 + 512 + d / 32] = raw.live ? (raw.scales[int(j) / 4] >> ((int(j) % 4) * 8)) & 255u : 127;
            });
    }
    template <class Pointer, class RawPointer>
    __device__ __forceinline__ void dequant_current(
        RawPointer raw_src, Pointer dst, int wave, int lane) const
    {
        auto* bytes = reinterpret_cast<OPUS_LDS_ADDR const uint8_t*>(reinterpret_cast<uintptr_t>(raw_src));
        const int row = (lane / 8) * 4 + wave;
        // Invalid rows were committed as zero bytes with scale 127, so they
        // dequantize to zero without a divergent validity branch. Load/convert
        // each 8-value piece directly: rebuilding RawTile prolongs register
        // lifetimes and crosses the measured residency threshold.
        auto* lds = reinterpret_cast<OPUS_LDS_ADDR bf16_t*>(reinterpret_cast<uintptr_t>(dst));
        opus::static_for<8>([&](auto j) {
            const int d = int(j) * 64 + (lane % 8) * 8;
            auto bits = *reinterpret_cast<OPUS_LDS_ADDR const opus::vector_t<uint32_t, 2>*>(bytes + row * 528 + d);
            float scale = __uint_as_float(uint32_t(bytes[row * 528 + 512 + d / 32]) << 23);
            opus::vector_t<bf16_t, 8> values;
            opus::static_for<2>([&](auto half) {
                uint32_t word = bits[int(half)];
                auto lo = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(word, scale, false);
                auto hi = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(word, scale, true);
                auto* pairs = reinterpret_cast<opus::vector_t<bf16_t, 2>*>(&values);
                pairs[int(half) * 2] = __builtin_bit_cast(opus::vector_t<bf16_t, 2>, lo);
                pairs[int(half) * 2 + 1] = __builtin_bit_cast(opus::vector_t<bf16_t, 2>, hi);
            });
            *reinterpret_cast<OPUS_LDS_ADDR opus::vector_t<bf16_t, 8>*>(lds + (int(j) * 4 + wave) * 528 + lane * 8) = values;
        });
    }
};

namespace opus_mla_v4_prefill_a16w16_16mx8_32nx1 {
template <class Traits, class Source>
__device__ void fp8_qk_decode_accum64(
    const Source& source,
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

    auto u_gkv        = make_layout_gkv<T>(warp_id, lane_id);
    auto u_skv        = make_layout_skv<T>(warp_id);
    auto u_rv         = make_layout_rv<T>(lane_id);

    typename decltype(mma0)::vtype_c v_s;
    typename decltype(mma1)::vtype_a v_p;
    typename decltype(mma1)::vtype_b v_v[2];
    auto* v_o_slices = v_o;

    constexpr index_t s_len = vector_traits<typename decltype(mma0)::vtype_c>::size();

    auto skv_slice = [](auto slice_idx) {
        constexpr int s = decltype(slice_idx)::value;
        return number<(s / 2) * T::smem_n_rpt*(T::smem_linear_wave + T::smem_padding_32B) +
                      (s % 2) * T::SLICE_D>{};
    };

    auto* raw_cur = reinterpret_cast<OPUS_LDS_ADDR uint8_t*>(reinterpret_cast<uintptr_t>(raw_lds));
    auto* raw_next = raw_cur + 32 * 528 + 128;
    auto mask_scores = [&](auto& s) {
        // Materialize validity addresses only when scores are masked.
        int mask_lane = lane_id;
        asm volatile("" : "+v"(mask_lane));
        mask_lane &= 63;
        static_for<2>([&](auto n) {
            int first = int(n) * 16 + (mask_lane / 16) * 4;
            static_for<4>([&](auto j) {
                int k          = first + int(j);
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
                auto part = *reinterpret_cast<OPUS_LDS_ADDR const vector_t<fp8_t, 16>*>(raw_cur + row * 528 + kd);
                set_slice(keys[slot], part, number<int(half) * 16>{}, number<int(half) * 16 + 16>{});
            });
            scales[slot] = raw_cur[row * 528 + 512 + step * 4 + lane_id / 16];
        };
        auto issue_qk = [&](auto index) {
            constexpr int i = int(index), slot = i % 2, step = i / 2, n = i % 2;
            auto kw = __builtin_bit_cast(vector_t<uint32_t, 8>, keys[slot]);
            auto qw = __builtin_bit_cast(vector_t<uint32_t, 8>, v_q[step]);
            auto acc = score[n];
            int ks = scales[slot], qs = q_scales[step];
            asm volatile("v_mfma_scale_f32_16x16x128_f8f6f4 %0, %1, %2, %0, %3, %4"
                         : "+v"(acc) : "v"(kw), "v"(qw), "v"(ks), "v"(qs));
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
            auto vw = __builtin_bit_cast(vector_t<uint32_t, 4>,
                slice(v, number<int(n) * 8>{}, number<int(n) * 8 + 8>{}));
            auto acc = slice(o, number<int(n) * 4>{}, number<int(n) * 4 + 4>{});
            asm volatile("v_mfma_f32_16x16x32_bf16 %0, %1, %2, %0"
                         : "+v"(acc) : "v"(vw), "v"(pw));
            set_slice(o, acc, number<int(n) * 4>{}, number<int(n) * 4 + 4>{});
        });
        return o;
    };
    auto compute_pv = [&](const auto& p, auto& v, auto& o, int kv_tile_idx) {
        static_for<T::NUM_D_SLICES - 2>([&](auto i) {
            constexpr int idx  = i.value;
            constexpr int slot = idx & 1;
            o[idx]             = pv_vgpr(p, v[slot], o[idx]);
            v[slot]            = tr_load<T::VEC_TR_V>(s_kv, u_rv + skv_slice(number<idx + 2>{}));
            s_waitcnt_lgkmcnt(number<T::v_ds_read_insts>{});
            // Partial waits rely on LDS issue order and on which MFMA consumes
            // the completed reads. Allow VALU across this boundary, but keep
            // LDS, MFMA and SALU (including s_waitcnt) ordered. Allowing MFMA
            // or SALU here makes a later slice consume still-pending reads.
            // See experiments/20261014/PV_SCHEDULING.md for ISA and regressions.
            __builtin_amdgcn_sched_barrier(VALU_MASK);
        });
        o[T::NUM_D_SLICES - 2] = pv_vgpr(p, v[(T::NUM_D_SLICES - 2) & 1], o[T::NUM_D_SLICES - 2]);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_sched_barrier(0);
        o[T::NUM_D_SLICES - 1] = pv_vgpr(p, v[(T::NUM_D_SLICES - 1) & 1], o[T::NUM_D_SLICES - 1]);
    };


    if constexpr(Source::is_packed)
        source.commit_raw(source.prefetch(0, warp_id, lane_id), raw_cur, warp_id, lane_id);
    else
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

        compute_qk(v_s);
        // Keep next-tile VMEM waits after QK, so QK covers raw prefetch
        // latency. This is a compiler fence, not a CTA barrier.
        __builtin_amdgcn_sched_barrier(0);
        if(tile_idx + 1 < num_kv_tiles)
        {
            if constexpr(Source::is_packed)
                source.commit_raw(next_raw, raw_next, warp_id, lane_id);
            else
                source.template load_tile<T>(
                    tile_idx + 1, s_kv.ptr, u_gkv, u_skv, warp_id, lane_id);
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
            v_s[int(i)] = __builtin_amdgcn_exp2f(__builtin_fmaf(v_s[int(i)], temperature_scale, -row_max));
        });
        l_row *= rescale_m;
        l_row += attn_row_sum<T>(v_s);
        v_p = cast<D_ATTN>(v_s);
        if(!below)
            scale_output_tile<T>(v_o, rescale_m);

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier(); // All waves have finished writing V.
        v_v[0] = tr_load<T::VEC_TR_V>(s_kv, u_rv);
        v_v[1] = tr_load<T::VEC_TR_V>(s_kv, u_rv + skv_slice(1_I));
        s_waitcnt_lgkmcnt(number<T::v_ds_read_insts>{});
        __builtin_amdgcn_sched_barrier(0);
        compute_pv(v_p, v_v, v_o_slices, tile_idx);
        s_waitcnt_vmcnt(0_I);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier(); // All PV reads finish before V is reused.

        std::swap(raw_cur, raw_next);
    }
}

} // namespace
