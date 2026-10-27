// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// What gfx950's one-row and qshare kernels share: the scheduling hints, the Q / scale / weight
// layouts and loads, the head reduction and the relu. Device pass on gfx950 only.
#pragma once
#include <opus/opus.hpp>
#include <bit>   // std::bit_cast, required by permlane_head_reduce2

#include "mqa_logits_mxfp4_gfx950_assign.cuh"

// __launch_bounds__ min waves/SIMD; raising it only constrains the register allocator.
#ifndef OPUS_LOGITS_MIN_WAVES
#define OPUS_LOGITS_MIN_WAVES 2
#endif

// relu is IEEE-754-2019 `maximum`, so a NaN score (an E8M0 0xFF) propagates instead of being
// selected away as 0. Needs -fno-finite-math-only.
#define MQA_LOGITS_GFX950_RELU(x) __builtin_elementwise_maximum((x), 0.0f)

namespace opus_logits::gfx950 {

using opus::operator""_I;

// ─── scheduling helpers ─────────────────────────────────────────────────────
constexpr int VALU_MASK = 0x02;   // v_max / v_add / v_pk_fma ... (no MFMA/TRANS/mem)
constexpr int MFMA_MASK = 0x08;   // v_mfma_*

template<int Pairs, int OTHER_MASK, int CNT, int Group>
__device__ inline void sched_mfma_pairs() {
    opus::static_for<Pairs>([&](auto) {
        __builtin_amdgcn_sched_group_barrier(MFMA_MASK, 1, Group);
        __builtin_amdgcn_sched_group_barrier(OTHER_MASK, CNT, Group);
    });
}

// Two independent lane^32 reductions; the scheduler overlaps the swaps and their hazards.
// std::bit_cast is required: __builtin_bit_cast here miscompiles the swap to `v + v` (invisible
// under uniform data). The builtin returns both swapped registers: r[0] + r[1] = v[L] + v[L^32].
__device__ inline void permlane_head_reduce2(float& o0, float& o1, float v0, float v1) {
    opus::vector_t<opus::u32_t, 2> r0 = __builtin_amdgcn_permlane32_swap(
        std::bit_cast<opus::u32_t>(v0), std::bit_cast<opus::u32_t>(v0), false, false);
    opus::vector_t<opus::u32_t, 2> r1 = __builtin_amdgcn_permlane32_swap(
        std::bit_cast<opus::u32_t>(v1), std::bit_cast<opus::u32_t>(v1), false, false);
    o0 = std::bit_cast<float>(r0[0]) + std::bit_cast<float>(r0[1]);
    o1 = std::bit_cast<float>(r1[0]) + std::bit_cast<float>(r1[1]);
}

// Q: natural [T][head][D/2]; lane L reads 16 B at
// head(mi*MFMA_M + L%32)*64 + (kt*K_CHUNKS + L/32)*16. Un-coalesced by design: Q is loaded
// once in the prologue, so the cost is amortized and the layout stays natural.
template<class T>
__device__ inline auto make_layout_q(int lane_mod_32, int lane_div_32) {
    constexpr int Q_ROW_NAT = T::HEAD_DIM * T::ELEM_BITS / 8;   // 64 B per natural head row
    constexpr auto shape = opus::make_tuple(
        opus::number<T::M_TILES>{},        // m-tile           (y, outer)
        opus::number<T::K_TILES>{},        // k-tile           (y, outer)
        opus::number<T::MFMA_M>{},         // head row in tile (p: lane % 32)
        opus::number<T::K_CHUNKS>{},       // 32-K block g     (p: lane / 32)
        opus::number<T::KV_GRP_BYTES>{});  // 16 B             (y, contiguous)
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout(
        shape,
        opus::unfold_x_stride(dim, shape, opus::make_tuple(
            opus::number<T::MFMA_M * Q_ROW_NAT>{},          // m-tile: 32 heads * 64 B
            opus::number<T::K_CHUNKS * T::KV_GRP_BYTES>{},  // k-tile: 2 blocks * 16 B
            opus::number<Q_ROW_NAT>{},                      // head row: 64 B
            opus::number<1>{})),                            // (g, byte) base=1 -> g=16, byte=1
        opus::unfold_p_coord(dim, opus::make_tuple(lane_mod_32, lane_div_32)));
}

// ─── Scale word (q_scale / kv_scale): one dword at index == lane ────────────
// The 4 bytes pack (kt, mi) for A / (kt, nt) for B; MFMA op_sel picks the byte.
template<class T>
__device__ inline auto make_layout_scale(int lane_id) {
    constexpr auto shape = opus::make_tuple(
        opus::number<T::WARP_SIZE>{},   // lane         (p)
        opus::number<1>{});             // single dword (y)
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));
    return opus::make_layout(
        shape,
        opus::unfold_x_stride(dim, shape, opus::make_tuple(
            opus::number<1>{},
            opus::number<0>{})),
        opus::unfold_p_coord(dim, opus::make_tuple(lane_id)));
}

// Weights: natural [T, H] bf16. Lane group lg = lane/32 gathers C-fragment heads at
// lg*PACK_C + mi*MFMA_M + rept*(PACK_C*GRPM_C) + pack: MT*REPT_C = 8 loads of PACK_C bf16.
template<class T>
__device__ inline auto make_layout_w(int lane_div_32) {
    constexpr auto shape = opus::make_tuple(
        opus::number<T::GRPM_C>{},     // lane group lg  (p)
        opus::number<T::M_TILES>{},    // m-tile         (y, outer)
        opus::number<T::REPT_C>{},     // rept = r/PACK_C(y, outer)
        opus::number<T::PACK_C>{});    // pack = r%PACK_C(y, contiguous)
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}));
    return opus::make_layout(
        shape,
        opus::unfold_x_stride(dim, shape, opus::make_tuple(
            opus::number<T::PACK_C>{},              // lg:   4
            opus::number<T::MFMA_M>{},              // mi:   32 heads
            opus::number<T::PACK_C * T::GRPM_C>{},  // rept: 8
            opus::number<1>{})),                    // pack: 1
        opus::unfold_p_coord(dim, opus::make_tuple(lane_div_32)));
}

// MT * KT 16 B loads -> the low half of each (mi, kt) A operand.
template<class T, class G, class U, class A>
__device__ inline void load_q(G& g_q, const U& u_q, A (&q_a)[T::M_TILES][T::K_TILES]) {
    auto vq = opus::load<T::KV_GRP_BYTES>(g_q, u_q);
    auto* q16 = reinterpret_cast<opus::vector_t<uint8_t, T::KV_GRP_BYTES>*>(&vq);
    opus::static_for<T::M_TILES>([&](auto mic) {
        constexpr int mi = mic.value;
        opus::static_for<T::K_TILES>([&](auto ktc) {
            constexpr int kt = ktc.value;
            opus::set_slice(q_a[mi][kt], q16[mi * T::K_TILES + kt],
                            opus::number<0>{}, opus::number<T::KV_GRP_BYTES>{});
        });
    });
}

// MT * (C_FRAG / PACK_C) issues, already in C-fragment order, times weight_scale.
template<class T, class G, class U>
__device__ inline void load_weights(G& g_w, const U& u_w,
                                    opus::vector_t<float, T::C_FRAG> (&w_pl)[T::M_TILES],
                                    float weight_scale) {
    constexpr int PACK_W = T::PACK_C;
    auto v_w = opus::load<PACK_W>(g_w, u_w);
    auto* wp = reinterpret_cast<opus::vector_t<typename T::D_WEIGHT, PACK_W>*>(&v_w);
    opus::static_for<T::M_TILES>([&](auto mic) {
        constexpr int mi = mic.value;
        opus::static_for<T::C_FRAG / PACK_W>([&](auto cc) {
            constexpr int c = cc.value;
            opus::static_for<PACK_W>([&](auto jc) {
                constexpr int j = jc.value;
                w_pl[mi][c * PACK_W + j] =
                    (float)wp[mi * (T::C_FRAG / PACK_W) + c][j] * weight_scale;
            });
        });
    });
}

// Weighted head sum of a tile's two n-tiles, packed two heads at a time, then the cross-lane-
// group finish: out[nt] is key nt * 32 + L % 32's logit in both lane groups.
template<class T, class S>
__device__ inline void head_reduce(const opus::vector_t<float, T::C_FRAG> (&w_pl)[T::M_TILES],
                                   S& s0, S& s1, opus::vector_t<float, T::N_TILES>& out) {
    constexpr int NTPW = T::N_TILES, EC = T::C_FRAG;
    static_assert(NTPW == 2, "permlane_head_reduce2 assumes NTPW == 2");
    S* sp[NTPW] = { &s0, &s1 };
    opus::vector_t<float, 2> acc[NTPW];
    opus::static_for<NTPW>([&](auto ntc) { acc[ntc.value] = opus::vector_t<float, 2>{0.0f, 0.0f}; });
    opus::static_for<T::M_TILES>([&](auto mic) {
        constexpr int mi = mic.value;
        opus::static_for<EC / 2>([&](auto pc) {
            constexpr int p = pc.value;
            opus::vector_t<float, 2> w{ w_pl[mi][2 * p], w_pl[mi][2 * p + 1] };
            opus::static_for<NTPW>([&](auto ntc) {
                constexpr int nt = ntc.value;
                S& sn = *sp[nt];
                opus::vector_t<float, 2> s{ sn[mi * EC + 2 * p], sn[mi * EC + 2 * p + 1] };
                acc[nt] = s * w + acc[nt];
            });
        });
    });
    // Both reductions together so the permlane chains fill each other's hazard slots.
    float ts0 = acc[0][0] + acc[0][1];
    float ts1 = acc[1][0] + acc[1][1];
    float o0, o1;
    permlane_head_reduce2(o0, o1, ts0, ts1);
    out[0] = o0;
    out[1] = o1;
}

}  // namespace opus_logits::gfx950
