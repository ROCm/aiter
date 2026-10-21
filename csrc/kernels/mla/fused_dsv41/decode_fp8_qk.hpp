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

// Preserve the stored FP8 K bytes for QK while producing BF16 V for PV.
// Separate ping-pong slots avoid a cross-wave read/write race without another barrier.
template <int PageSize>
struct Fp8QkSource64 : PackedDecodeSource64<true, PageSize>
{
    using Base = PackedDecodeSource64<true, PageSize>;
    template <class Pointer, class RawPointer>
    __device__ __forceinline__ void commit_both(
        const typename Base::RawTile& raw, Pointer dst, RawPointer raw_dst, int wave, int lane) const
    {
        this->commit(raw, dst, wave, lane);
        auto* bytes = reinterpret_cast<OPUS_LDS_ADDR uint8_t*>(reinterpret_cast<uintptr_t>(raw_dst));
        const int row = (lane / 8) * 4 + wave;
        opus::static_for<8>([&](auto j) {
            const int d = int(j) * 64 + (lane % 8) * 8;
            *reinterpret_cast<OPUS_LDS_ADDR opus::vector_t<uint32_t, 2>*>(bytes + row * 528 + d) = raw.bytes[int(j)];
            if((lane & 3) == 0)
                bytes[row * 528 + 512 + d / 32] = raw.live ? (raw.scales[int(j) / 4] >> ((int(j) % 4) * 8)) & 255u : 127;
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
    };
    // K is A and Q is B: lane%16 owns a head; lane/16 owns four scores.
    // Data uses two 16-byte pieces; scale lanes retain contiguous group32 order.
    auto* raw_cur = reinterpret_cast<OPUS_LDS_ADDR uint8_t*>(reinterpret_cast<uintptr_t>(raw_lds));
    auto* raw_next = raw_cur + 32 * 528;
    auto compute_qk = [&](auto& s) {
        auto mfma = make_mfma<fp8_t, fp8_t, float>(16_I, 16_I, 128_I);
        vector_t<float, 4> score[2]{};
        static_for<4>([&](auto step) {
            static_for<2>([&](auto n) {
                const int row = int(n) * 16 + lane_id % 16;
                const int d = int(step) * 128 + (lane_id / 16) * 32;
                vector_t<fp8_t, 32> k;
                static_for<2>([&](auto half) {
                    const int kd = int(step) * 128 + (lane_id / 16) * 16 + int(half) * 64;
                    auto part = *reinterpret_cast<OPUS_LDS_ADDR const vector_t<fp8_t, 16>*>(
                        raw_cur + row * 528 + kd);
                    set_slice(k, part, number<int(half) * 16>{}, number<int(half) * 16 + 16>{});
                });
                const int scale = raw_cur[row * 528 + 512 + d / 32];
                score[int(n)] = mfma(k, v_q[int(step)], score[int(n)], scale, q_scales[int(step)]);
            });
        });
        static_for<2>([&](auto n) {
            set_slice(s, score[int(n)], number<int(n) * 4>{}, number<int(n) * 4 + 4>{});
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

    auto other_kv = make_smem(reinterpret_cast<D_ATTN*>(smem_kv_buf) + T::smem_kv_tile_elems);
    if constexpr(Source::is_packed)
        source.commit_both(source.prefetch(0, warp_id, lane_id), s_kv.ptr, raw_cur, warp_id, lane_id);
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
        // Keep next-tile VMEM waits and dequantization after QK, so QK covers
        // the raw prefetch latency. This is a compiler fence, not a CTA barrier.
        __builtin_amdgcn_sched_barrier(0);
        if(tile_idx + 1 < num_kv_tiles)
        {
            if constexpr(Source::is_packed)
                source.commit_both(next_raw, other_kv.ptr, raw_next, warp_id, lane_id);
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
        std::swap(raw_cur, raw_next);
    }
}

} // namespace
