// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Split-K epilogues of the gfx1250 MLA v4.1 prefill kernels.

#pragma once

#include <opus/opus.hpp>
#include "mla_v41_prefill_traits.h"

namespace opus_mla_v41_split {

// A cluster reads the partials back within its SE.
template<class T, bool CLUSTER, int ROWS>
OPUS_D void store_partial_o(opus::u32_t lds, const opus_mla_prefill_split_args& split, int token, int split_idx, int H, int head0) {
    using namespace opus;
    constexpr auto scope = CLUSTER ? tdm_traits::scope::se : tdm_traits::scope::dev;
    using po_window = tdm<float, seq<T::PO_ROW_LDS_ELEMS, ROWS>,
                          tdm_traits::cache<tdm_traits::make_cache_policy(tdm_traits::store_temporal_hint::regular, scope)>>;
    make_tdm<po_window>(lds, split.partial_o + opus_mla_prefill_split_row(split, token, split_idx, H, 0) * T::D_HEAD_SIZE,
                        (u32_t)T::D_HEAD_SIZE, (u32_t)H, (u64_t)T::D_HEAD_SIZE, 0u, (u32_t)head0).async_store();
}

// Split i of a cluster folds heads [i * HEADS / S, +HEADS / S) from LDS, so it stores none of them, nor any if empty.
template<class T, bool CLUSTER>
OPUS_D bool skips_store(int split_idx, int num_splits, int num_valid, int row0, int rows) {
    if constexpr (!CLUSTER) return false;
    const int hs = T::T_M * T::Q_TILE_SIZE / num_splits;
    return split_idx >= num_valid || (row0 >= split_idx * hs && row0 + rows <= (split_idx + 1) * hs);
}

// Slot q is split (own + q) % S, slot 0 from LDS. Bounded buffers read empty splits and rows past heads as 0, so the
// loads issue at once.
template<class T, int S, int HS>
OPUS_D void fold(const opus_mla_prefill_split_args& split, const opus_mla_v41_prefill_kargs& kargs, int token,
                 int head_lo, int heads, int num_valid, int own, const char* own_rows, float* w_lds) {
    using namespace opus;
    using D_OUT = typename T::D_OUT;
    using f32x4 = vector_t<float, 4>;
    constexpr int D = T::D_HEAD_SIZE, GPR = T::WARP_SIZE / S, REGS = (HS + GPR - 1) / GPR;
    static_assert(D == 4 * T::BLOCK_SIZE && (S & (S - 1)) == 0 && HS * S % 4 == 0);
    constexpr float LOG2_E = 1.44269504089f;
    if (heads <= 0) return;
    const int tid = thread_id_x(), lane = tid % T::WARP_SIZE;
    auto g_o = make_gmem(reinterpret_cast<D_OUT*>(kargs.out_ptr) + (int64_t)token * kargs.stride_o_n + (int64_t)head_lo * kargs.stride_o_h,
                         (u32_t)(((heads - 1) * kargs.stride_o_h + D) * sizeof(D_OUT)));
    auto store_o = [&](int j, const f32x4& v) { store<4>(g_o, cast<D_OUT>(v), tid * 4, j * kargs.stride_o_h); };
    if (num_valid == 0) {
        static_for<HS>([&](auto j) { store_o(j.value, f32x4{}); });
        return;
    }

    const auto* sink_ptr = reinterpret_cast<const float*>(kargs.attn_sink_ptr);
    const int lq = lane % S, lj = lane / S;
    float m[REGS], l[REGS], sink[REGS];
    bool valid[REGS];
    static_for<REGS>([&](auto r) {
        const int j = r.value * GPR + lj, a = (own + lq) % S, jc = j < heads ? j : 0;
        valid[r.value] = a < num_valid && j < heads;
        const int64_t row = opus_mla_prefill_split_row(split, token, valid[r.value] ? a : 0, kargs.H, head_lo + jc);
        m[r.value]    = split.partial_max[row];
        l[r.value]    = split.partial_sum[row];
        sink[r.value] = (sink_ptr ? sink_ptr : split.partial_max)[head_lo + jc];
    });

    f32x4 x[HS * S];
    static_for<S - 1>([&](auto q1) {
        constexpr int q = q1.value + 1;
        const int a = (own + q) % S;
        auto g = make_gmem(split.partial_o + opus_mla_prefill_split_row(split, token, a, kargs.H, head_lo) * D,
                           (u32_t)(a < num_valid ? heads * D * sizeof(float) : 0));
        static_for<HS>([&](auto j) { x[j.value * S + q] = load<4>(g, tid * 4 + j.value * D); });
    });
    auto s_own = make_smem(reinterpret_cast<float*>(const_cast<char*>(own_rows)));
    static_for<HS>([&](auto j) { x[j.value * S] = load<4>(s_own, j.value * T::PO_ROW_LDS_ELEMS + tid * 4); });

    static_for<REGS>([&](auto r) {
        const float mv = valid[r.value] ? m[r.value] : -FLT_MAX;
        float m_max = mv;
#pragma unroll
        for (int k = 1; k < S; k <<= 1) m_max = max(m_max, shfl(m_max, lane ^ k));
        const float scale = mv > -FLT_MAX ? __builtin_amdgcn_exp2f((mv - m_max) * LOG2_E) * l[r.value] : 0.0f;
        float sum = scale;
#pragma unroll
        for (int k = 1; k < S; k <<= 1) sum += shfl(sum, lane ^ k);
        sum += sink_ptr && m_max > -FLT_MAX ? __builtin_amdgcn_exp2f((sink[r.value] - m_max) * LOG2_E) : 0.0f;
        const int j = r.value * GPR + lj;
        if (j < HS) w_lds[j * S + lq] = scale / (sum > 0.0f ? sum : 1.0f);
    });
    // No barrier: each wave reads back only its own weights.
    auto s_w = make_smem(w_lds);
    f32x4 w[HS * S / 4];
    static_for<HS * S / 4>([&](auto k) { w[k.value] = load<4>(s_w, k.value * 4); });
    static_for<HS>([&](auto j) {
        f32x4 acc{};
        static_for<S>([&](auto q) {
            constexpr int i = j.value * S + q.value;
            acc += w[i / 4][i % 4] * x[i];
        });
        store_o(j.value, acc);
    });
}

template<class T>
OPUS_D void cluster_fold(const opus_mla_prefill_split_args& split, const opus_mla_v41_prefill_kargs& kargs, const char* smem,
                         int token, int split_idx, int num_valid, int h_block_start) {
    using namespace opus;
    constexpr int HEADS = T::T_M * T::Q_TILE_SIZE;
    s_wait_tensorcnt(0_I);
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "cluster");
    __builtin_amdgcn_s_barrier();
    const int warp_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
    // The cluster barrier counts workgroups.
    if (warp_id == 0) __builtin_amdgcn_s_barrier_signal(-3);
    __builtin_amdgcn_s_barrier_wait(-3);
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "cluster");

    auto run = [&](auto s) {
        constexpr int S = decltype(s)::value, HS = HEADS / S;
        const int head_lo = h_block_start + split_idx * HS;
        fold<T, S, HS>(split, kargs, token, head_lo, min(HS, kargs.H - head_lo), num_valid, split_idx,
                       smem + split_idx * HS * T::PO_ROW_LDS_BYTES,
                       reinterpret_cast<float*>(const_cast<char*>(smem) + HEADS * T::PO_ROW_LDS_BYTES) + warp_id * HEADS);
    };
    switch (split.num_splits) {
        case 2:  run(number<2>{});  break;
        case 4:  run(number<4>{});  break;
        case 8:  run(number<8>{});  break;
        case 16: run(number<16>{}); break;
        default: break;
    }
}

} // namespace opus_mla_v41_split
