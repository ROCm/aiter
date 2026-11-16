// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Fused FP8 MQA logits + top-k (gfx950): threshold, select and exact fallback.
// Tau uses 32-bit order keys of the fp32 sample logits. The fallback recompute is
// bit-identical to v35wes: plain MFMA, scalar even/odd fma chains.
//
//   tau  (tau_kernel, one wave per row): e = K * ns / nvalid samples expected above the K-th logit,
//        target = ceil(e + 5 sqrt(e)) + 8, tau <= the target-th largest sample; -inf when nvalid <= CAP or
//        ns < target (every valid key is a candidate).
//   select (select_kernel, one wave per row): exact top-K over the fp32 candidates (up to 48 x 64 in registers:
//        relative radix; else 4 x 8-bit radix re-reading global); a row whose segment overflowed or that has fewer
//        than min(K, nvalid) candidates is appended to the fallback list.
//   fallback (fb_compute + fb_select, fallback_kernel): flagged rows only; the exact logits of the whole row are
//        recomputed into scratch, then a radix select.
// Output = column - ks (unordered), -1 padding; a row with nvalid <= K gets 0..nvalid-1 in order.
#pragma once
#include "fmt_common.cuh"

namespace mqa_topk {

// ------------------------------------------------------------------------------------------------- wave helpers
// wave-level: bin b of hist[256] with above(b) < target <= above(b) + h[b]
// inclusive prefix sum over the wave with DPP (row_shr 1/2/4/8 inside each 16-lane row, then row_bcast 15 / 31 across
// rows; gfx9). VALU only - replaces a chain of 6 dependent ds_bpermute (__shfl_up) round trips.
__device__ __forceinline__ int wave_incl_scan_dpp(int x) {
    x += __builtin_amdgcn_update_dpp(0, x, 0x111, 0xf, 0xf, false);   // row_shr:1
    x += __builtin_amdgcn_update_dpp(0, x, 0x112, 0xf, 0xf, false);   // row_shr:2
    x += __builtin_amdgcn_update_dpp(0, x, 0x114, 0xf, 0xf, false);   // row_shr:4
    x += __builtin_amdgcn_update_dpp(0, x, 0x118, 0xf, 0xf, false);   // row_shr:8
    x += __builtin_amdgcn_update_dpp(0, x, 0x142, 0xa, 0xf, false);   // row_bcast:15 into rows 1, 3
    x += __builtin_amdgcn_update_dpp(0, x, 0x143, 0xc, 0xf, false);   // row_bcast:31 into rows 2, 3
    return x;
}
// bin of the target-th largest key in a 256-bin histogram (lane l owns bins 4l..4l+3); fab = keys in higher bins,
// fcnt = keys in the bin. Two versions, chosen per caller:
//  DPP = false: shuffle scan (ds_bpermute) - the LDS pipe runs in parallel with the VALU, faster in the
//    high-occupancy tau / select kernels (tau phase 70.5 vs 97.0 us on 576 x 250000);
//  DPP = true (default): DPP scan + readlane broadcasts + one ds_read_b128 - fewer dependent round trips (the
//    fallback's dense select). hist 16-byte aligned.
template <bool DPP = true>
__device__ __forceinline__ void wave_find256(const uint32_t* hist, int target, int lane, int& fbin, int& fab, int& fcnt) {
    const int lo = lane * 4;
    if constexpr (!DPP) {
        int s4 = 0;
        for (int b = lo; b < lo + 4; ++b) s4 += hist[b];
        const int incl = wave_incl_scan(s4, lane);
        int cum = __shfl(incl, WARP - 1) - incl, fb = -1, fa = 0, fc = 0;
        for (int b = lo + 3; b >= lo; --b) { const int h = hist[b]; if (cum < target && cum + h >= target) { fb = b; fa = cum; fc = h; } cum += h; }
        const uint64_t bm = __ballot(fb >= 0);
        const int src = bm ? __builtin_ctzll(bm) : 0;
        fbin = __shfl(fb, src); fab = __shfl(fa, src); fcnt = __shfl(fc, src);
    } else {
        const u32x4 h4 = *reinterpret_cast<const u32x4*>(&hist[lo]);
        const int incl = wave_incl_scan_dpp((int)(h4[0] + h4[1] + h4[2] + h4[3]));
        int cum = __builtin_amdgcn_readlane(incl, WARP - 1) - incl, fb = -1, fa = 0, fc = 0;
        #pragma unroll
        for (int k = 3; k >= 0; --k) {
            const int h = (int)h4[k];
            if (cum < target && cum + h >= target) { fb = lo + k; fa = cum; fc = h; }
            cum += h;
        }
        const uint64_t bm = __ballot(fb >= 0);
        const int src = bm ? __builtin_ctzll(bm) : 0;
        fbin = __builtin_amdgcn_readlane(fb, src); fab = __builtin_amdgcn_readlane(fa, src); fcnt = __builtin_amdgcn_readlane(fc, src);
    }
}

// radix over 32-bit keys delivered by each(f) (f(key, col) called per element, non-uniformly) -> key T of rank K
template <int NRD, bool DPP = true, typename Each>
__device__ __forceinline__ void wave_radix(Each&& each, uint32_t* hist, int K, int lane, uint32_t& T, int& above, int& cnt_eq) {
    uint32_t prefix = 0; above = 0; cnt_eq = 0;
    #pragma unroll
    for (int r = 0; r < NRD; ++r) {
        const int sh = 8 * (NRD - 1 - r);
        for (int b = lane; b < 256; b += WARP) hist[b] = 0;
        mem_fence_wave();
        const uint32_t pf = prefix;
        each([&](uint32_t k, int) { if (r == 0 || (uint32_t)((uint64_t)k >> (sh + 8)) == pf) atomicAdd(&hist[(k >> sh) & 255], 1u); });
        mem_fence_wave();
        int fb, fa, fc;
        wave_find256<DPP>(hist, K - above, lane, fb, fa, fc);
        prefix = (prefix << 8) | (uint32_t)fb; above += fa; cnt_eq = fc;
        mem_fence_wave();
    }
    T = prefix;
}
// wave sum of a per-lane int with DPP (VALU only, lane 63 ends with the total; gfx9 row_bcast), as a scalar
__device__ __forceinline__ int wave_sum_dpp(int x) {
    x += __builtin_amdgcn_update_dpp(0, x, 0x111, 0xf, 0xf, false);   // row_shr:1
    x += __builtin_amdgcn_update_dpp(0, x, 0x112, 0xf, 0xf, false);   // row_shr:2
    x += __builtin_amdgcn_update_dpp(0, x, 0x114, 0xf, 0xf, false);   // row_shr:4
    x += __builtin_amdgcn_update_dpp(0, x, 0x118, 0xf, 0xf, false);   // row_shr:8  -> lane 15 of each row = row sum
    x += __builtin_amdgcn_update_dpp(0, x, 0x142, 0xa, 0xf, false);   // row_bcast:15 into rows 1, 3
    x += __builtin_amdgcn_update_dpp(0, x, 0x143, 0xc, 0xf, false);   // row_bcast:31 into rows 2, 3
    return __builtin_amdgcn_readlane(x, 63);
}
// count of keys >= x over the wave (keys in registers rk[0..R), 0 = empty slot never counted since x >= 1).
// Per-lane VALU count (v_cmp + v_addc on VCC: VALU->VALU forwarding) + one DPP reduction. A ballot + s_bcnt1 per
// register serializes every compare through VCC -> SALU transfers (measured ~10x slower in the tau search).
template <int R>
__device__ __forceinline__ int wave_count_ge(const uint32_t (&rk)[R], uint32_t x) {
    int c = 0;
    #pragma unroll
    for (int j = 0; j < R; ++j) c += (rk[j] >= x) ? 1 : 0;
    return wave_sum_dpp(c);
}
// emission over a uniform stream: each_u(g) must call g(valid, key, col) with ALL lanes, same trip count
template <typename EachU>
__device__ __forceinline__ void wave_emit(EachU&& each_u, uint32_t T, int above, int K, int ks, int lane, int* __restrict__ orow) {
    const int need = K - above;
    int pdef = 0, ptie = 0;
    each_u([&](bool v, uint32_t k, int col) {
        const bool gt = v && k > T, eq = v && k == T;
        const uint64_t mg = __ballot(gt), me = __ballot(eq);
        const int og = pdef + (int)__builtin_amdgcn_mbcnt_hi((uint32_t)(mg >> 32), __builtin_amdgcn_mbcnt_lo((uint32_t)mg, 0));
        const int oe = ptie + (int)__builtin_amdgcn_mbcnt_hi((uint32_t)(me >> 32), __builtin_amdgcn_mbcnt_lo((uint32_t)me, 0));
        if (gt) orow[og] = col - ks;
        if (eq && oe < need) orow[above + oe] = col - ks;
        pdef += __popcll(mg); ptie += __popcll(me);
    });
}

// wave min / max of a per-lane u32 with DPP (gfx9 row_bcast), result as a scalar
__device__ __forceinline__ uint32_t wave_max_u32(uint32_t x) {
    x = max(x, (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0x111, 0xf, 0xf, false));
    x = max(x, (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0x112, 0xf, 0xf, false));
    x = max(x, (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0x114, 0xf, 0xf, false));
    x = max(x, (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0x118, 0xf, 0xf, false));
    x = max(x, (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0x142, 0xa, 0xf, false));
    x = max(x, (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0x143, 0xc, 0xf, false));
    return (uint32_t)__builtin_amdgcn_readlane((int)x, 63);
}
__device__ __forceinline__ uint32_t wave_min_u32(uint32_t x) { return ~wave_max_u32(~x); }

// Relative two-pass radix threshold of one row (one wave): keys of the valid samples in rk (0 = empty slot).
// Pass 1 histograms the 8 bits just below the common prefix of [kmin, kmax], pass 2 the next 8 bits inside the bin
// holding the target-th largest key, so the bucket resolution is ~(kmax - kmin) / 2^16 whatever the score offset.
// Returns the lowest float of the final bucket (<= the target-th largest sample): conservative.
template <int R, bool DPP = true>
__device__ __forceinline__ float wave_tau_relradix(const uint32_t (&rk)[R], int target, uint32_t* hist, int lane) {
    uint32_t vmax = 0u, vmin = 0xFFFFFFFFu;
    #pragma unroll
    for (int j = 0; j < R; ++j) { vmax = max(vmax, rk[j]); vmin = min(vmin, rk[j] ? rk[j] : 0xFFFFFFFFu); }
    const uint32_t kmax = wave_max_u32(vmax), kmin = wave_min_u32(vmin);
    const uint32_t diff = kmax ^ kmin;
    if (diff == 0u) return __builtin_bit_cast(float, okey32_inv(kmax));          // every sample equal
    const int hb = 31 - __builtin_clz(diff);                                      // highest differing bit
    const int sh1 = max(hb - 7, 0), sh2 = max(sh1 - 8, 0);
    const uint32_t pre1 = (sh1 + 8 >= 32) ? 0u : (kmax >> (sh1 + 8));             // common prefix above pass-1 bits
    // pass 1
    for (int b = lane; b < 256; b += WARP) hist[b] = 0u;
    mem_fence_wave();
    #pragma unroll
    for (int j = 0; j < R; ++j) if (rk[j]) atomicAdd(&hist[(rk[j] >> sh1) & 255u], 1u);
    mem_fence_wave();
    int b1, a1, c1;
    wave_find256<DPP>(hist, target, lane, b1, a1, c1);
    uint32_t T = (pre1 << 8 | (uint32_t)b1) << sh1;
    if (sh1 > 0) {
        // pass 2: keys whose bits above sh1 equal T's
        mem_fence_wave();
        for (int b = lane; b < 256; b += WARP) hist[b] = 0u;
        mem_fence_wave();
        const uint32_t hiT = T >> sh1;
        const int w2 = sh1 - sh2;                                                 // 1..8 bits
        #pragma unroll
        for (int j = 0; j < R; ++j)
            if (rk[j] && (rk[j] >> sh1) == hiT) atomicAdd(&hist[(rk[j] >> sh2) & ((1u << w2) - 1u)], 1u);
        mem_fence_wave();
        int b2, a2, c2;
        wave_find256<DPP>(hist, target - a1, lane, b2, a2, c2);
        T |= (uint32_t)b2 << sh2;
    }
    return __builtin_bit_cast(float, okey32_inv(T));
}

// tau of a row with many samples (ns > 32 * 64, any ns; one wave): every lane keeps the T largest keys of its samples
// i = lane + 64 j (branch-free insertion network, loads issued B at a time), then the relative radix runs on the 64 T
// kept keys. The kept keys are a SUBSET of the samples, so their target-th largest is <= the samples' target-th
// largest: tau stays a valid lower bound (it is lower - more candidates, still exact - only when one lane held more
// than T of the top-`target` samples). Replaces the 4-pass radix, which re-reads global with one dependent load per
// step. The caller checks target <= 16 T.
template <int T, int B, bool DPP = true, typename LD>
__device__ __forceinline__ float wave_tau_stream(LD& ld, int ns, int target, uint32_t* hist, int lane) {
    uint32_t top[T];
    #pragma unroll
    for (int t = 0; t < T; ++t) top[t] = 0u;
    #pragma unroll 1
    for (int c0 = 0; c0 < ns; c0 += B * WARP) {   // wave-uniform trip count
        uint32_t v[B];
        #pragma unroll
        for (int j = 0; j < B; ++j) {               // unconditional loads (clamped index), masked after
            const int c = c0 + j * WARP + lane;
            const float x = ld(min(c, ns - 1));
            v[j] = c < ns ? okey32(__builtin_bit_cast(uint32_t, x)) : 0u;
        }
        #pragma unroll
        for (int j = 0; j < B; ++j) {
            uint32_t x = v[j];
            #pragma unroll
            for (int t = 0; t < T - 1; ++t) { const uint32_t h = max(top[t], x); x = min(top[t], x); top[t] = h; }
            top[T - 1] = max(top[T - 1], x);
        }
    }
    return wave_tau_relradix<T, DPP>(top, target, hist, lane);
}
constexpr int TAU_STREAM_T = 8, TAU_STREAM_B = 16;

// Exact K-th largest of the keys rk (0 = empty slot, nempty of them over the wave) by a RELATIVE radix: 8-bit
// LDS-histogram passes starting just below the common prefix of the row's min / max key; stops as soon as the boundary
// bin holds exactly the keys still needed (then every key >= the bin's lower edge is selected). Returns T / above /
// cnt_eq in wave_emit's convention: emit keys > T, then keys == T until K. Typically 2-3 passes instead of a 32-step
// bitwise search.
template <int R, bool DPP = true>
__device__ __forceinline__ void wave_select_relradix(const uint32_t (&rk)[R], int K, int nempty, uint32_t* hist, int lane,
                                                     uint32_t& T, int& above, int& cnt_eq) {
    uint32_t vmax = 0u, vmin = 0xFFFFFFFFu;
    #pragma unroll
    for (int j = 0; j < R; ++j) { vmax = max(vmax, rk[j]); vmin = min(vmin, rk[j] ? rk[j] : 0xFFFFFFFFu); }
    const uint32_t kmax = wave_max_u32(vmax), kmin = wave_min_u32(vmin);
    above = 0;
    if (kmax == kmin) { T = kmax; cnt_eq = wave_count_ge(rk, kmax); return; }
    // pass covers bits [sh, top); bits >= top are fixed (common prefix, then previous passes) and must match
    int top = 32 - __builtin_clz(kmax ^ kmin);                 // highest differing bit + 1
    int sh = max(top - 8, 0);
    uint32_t pref = (top >= 32) ? 0u : ((kmax >> top) << top);
    int need = K;
    #pragma unroll 1
    while (true) {
        for (int b = lane; b < 256; b += WARP) hist[b] = 0u;
        mem_fence_wave();
        const uint32_t hp = (top >= 32) ? 0u : (pref >> top);
        const uint32_t wmask = (1u << (top - sh)) - 1u;
        // Every slot is counted and the empty ones (key 0, in bin 0 whenever the prefix is 0) are taken back out after:
        // a per-slot k != 0 test is loop-invariant, and its R hoisted lane masks spilled SGPRs.
        #pragma unroll
        for (int j = 0; j < R; ++j) {
            const uint32_t k = rk[j];
            if (((top >= 32) ? 0u : (k >> top)) == hp) atomicAdd(&hist[(k >> sh) & wmask], 1u);
        }
        if (hp == 0u && lane == 0) atomicSub(&hist[0], (uint32_t)nempty);
        mem_fence_wave();
        int b, ab, cb;
        wave_find256<DPP>(hist, need, lane, b, ab, cb);
        pref |= (uint32_t)b << sh;
        above += ab; need -= ab;
        mem_fence_wave();
        if (cb == need && pref != 0u) {   // the whole boundary bin is selected: keys >= pref (low bits zero)
            T = pref - 1u; cnt_eq = 0; above += need; return;
        }
        if (sh == 0) { T = pref; cnt_eq = cb; return; }
        top = sh;
        sh = max(sh - 8, 0);
    }
}

// ------------------------------------------------------------------------------------------------- tau
// Threshold of one row from fp32 sample logits srow[a, b) (sample j = key j*S). One wave. Returns -inf when every
// valid key must be a candidate. `hist` = 256 u32 of LDS owned by this wave. Up to 32 x 64 samples sit in registers
// (the register count follows the sample count, wave-uniform branch); longer rows stream through wave_tau_stream, or,
// for a large target rank, run the 4-pass radix re-reading global.
template <int R, bool COHERENT, bool DPP = true>
__device__ __forceinline__ float wave_tau_regs(const float* __restrict__ srow, int a, int b, int target, uint32_t* hist, int lane) {
    uint32_t rk[R];
    if constexpr (COHERENT) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    #pragma unroll
    for (int j = 0; j < R; ++j) {               // unconditional loads (clamped index), masked after: no branch + wait per load
        const int c = a + j * WARP + lane;
        const float v = srow[min(c, b - 1)];
        rk[j] = c < b ? okey32(__builtin_bit_cast(uint32_t, v)) : 0u;
    }
    return wave_tau_relradix<R, DPP>(rk, target, hist, lane);
}
template <bool COHERENT, bool DPP = true>
__device__ __forceinline__ float wave_tau(const float* __restrict__ srow, int a, int b, int nvalid, int K, int CAP,
                                          uint32_t* hist, int lane) {
    const int ns = max(b - a, 0);
    const float e = (float)K * (float)ns / (float)max(nvalid, 1);
    const int target = (int)ceilf(e + 5.f * sqrtf(e)) + 8;
    if (nvalid <= CAP || ns < target) return -__builtin_huge_valf();
    if (ns <= 8 * WARP) return wave_tau_regs<8, COHERENT, DPP>(srow, a, b, target, hist, lane);
    if (ns <= 16 * WARP) return wave_tau_regs<16, COHERENT, DPP>(srow, a, b, target, hist, lane);
    if (ns <= 32 * WARP) return wave_tau_regs<32, COHERENT, DPP>(srow, a, b, target, hist, lane);
    if constexpr (COHERENT) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    if (target <= 16 * TAU_STREAM_T) {
        auto ld = [&](int i) -> float { return srow[a + i]; };
        return wave_tau_stream<TAU_STREAM_T, TAU_STREAM_B, DPP>(ld, ns, target, hist, lane);
    }
    constexpr int RT = 32;                          // first RT * 64 samples held in VGPRs (one read), rest re-read
    uint32_t rk[RT];
    #pragma unroll
    for (int j = 0; j < RT; ++j) { const int c = a + j * WARP + lane; const float v = srow[min(c, b - 1)]; rk[j] = c < b ? okey32(__builtin_bit_cast(uint32_t, v)) : 0u; }
    auto each = [&](auto&& f) {
        #pragma unroll
        for (int j = 0; j < RT; ++j) if (rk[j]) f(rk[j], 0);
        for (int c0 = a + RT * WARP; c0 < b; c0 += WARP) {     // uniform trip count, unconditional load
            const int c = c0 + lane; const float v = srow[min(c, b - 1)];
            if (c < b) f(okey32(__builtin_bit_cast(uint32_t, v)), c);
        }
    };
    uint32_t T; int above, ce;
    wave_radix<4, DPP>(each, hist, target, lane, T, above, ce);
    return __builtin_bit_cast(float, okey32_inv(T));
}

// tau kernel: fp32 sample logits ls[M, Ns] of keys j*S; 4 rows (waves) per block.
// Also resets the fallback counter of this call (select appends, fallback kernels read).
__global__ __launch_bounds__(256) void tau_kernel(const float* __restrict__ ls, int Ns, int S, const int* __restrict__ ks_p,
                                                  const int* __restrict__ ke_p, int M, int N, int K, int CAP,
                                                  float* __restrict__ tau, int* __restrict__ fbc) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *fbc = 0;
    __shared__ uint32_t hist_all[4][256];
    const int lane = threadIdx.x % WARP, w = threadIdx.x / WARP, m = blockIdx.x * 4 + w;
    if (m >= M) return;
    const int ksm = max(ks_p[m], 0), kem = min(ke_p[m], N);
    const int nvalid = kem - ksm;
    const int a = MQA_TOPK_CDIV(ksm, S), b = kem > 0 ? min(MQA_TOPK_CDIV(kem, S), Ns) : 0;
    const float t = wave_tau<false, false>(ls + (int64_t)m * Ns, a, b, nvalid, K, CAP, hist_all[w], lane);   // shuffle find256: high occupancy
    if (lane == 0) tau[m] = t;
}

// ------------------------------------------------------------------------------------------------- select
// One row with nvalid > K: candidates i < cnt[s * M + m] of split s at cand[cand_index(s, m, i)]. Returns false if the
// row must go to the fallback: a split overflowed (cnt > CAP), or the row has fewer than K candidates.
template <int RC, bool COHERENT, bool DPP = true>
__device__ __forceinline__ bool wave_select_row(const unsigned long long* __restrict__ cand, const int* __restrict__ cnt,
                                                int nsplit, int CAP, int M, int m, int ks, int nvalid, int K,
                                                int* __restrict__ orow, uint32_t* hist, int lane) {
    const int Mp = M + (M & 1);
    if constexpr (COHERENT) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    int c_l = cnt[(int64_t)min(lane, nsplit - 1) * M + m];
    if (lane >= nsplit) c_l = 0;
    const bool fail_split = __ballot(c_l > CAP) != 0;
    c_l = min(c_l, CAP);
    const int c_incl = wave_incl_scan(c_l, lane);
    const int C = __shfl(c_incl, WARP - 1);
    if (fail_split || C < min(K, nvalid)) return false;
    int pe_s[16];                                   // segment prefix ends, read with all lanes active (uniform)
    #pragma unroll
    for (int t = 0; t < 16; ++t) pe_s[t] = __shfl(c_incl, t);
    auto addr = [&](int i) -> int64_t {
        int s = 0, base = 0;
        #pragma unroll
        for (int t = 0; t < 16; ++t) if (t < nsplit && i >= pe_s[t]) { s = t + 1; base = pe_s[t]; }
        return cand_index(s, m, i - base, Mp, CAP);
    };
    // unconditional loads of a clamped index (masked after): no exec branch + vmcnt(0) per load
    auto ldc = [&](int i) -> unsigned long long {
        const int ic = min(i, C - 1);
        const int64_t o = nsplit == 1 ? cand_index(0, m, ic, Mp, CAP) : addr(ic);
        return cand[o];
    };
    auto keyf = [&](unsigned long long d) -> uint32_t { return okey32((uint32_t)d); };
    // Register slot j holds a candidate iff rk[j] != 0: okey32 maps only the NaN pattern 0xFFFFFFFF to 0, and the scan
    // never emits a NaN. Everything below tests rk, not the slot index j * WARP + lane: index tests made the compiler
    // keep RC indices and RC masks live, which took the RC = 48 path past 256 VGPRs (spills, one wave per SIMD).
    uint32_t rk[RC]; int rc[RC];
    const int rem = C - lane;
    #pragma unroll
    for (int j = 0; j < RC; ++j) {
        const unsigned long long d = ldc(j * WARP + lane);
        rk[j] = j * WARP < rem ? keyf(d) : 0u; rc[j] = (int)(d >> 32);
    }
    auto each = [&](auto&& f) {
        #pragma unroll
        for (int j = 0; j < RC; ++j) if (rk[j]) f(rk[j], rc[j]);
        for (int i0 = RC * WARP; i0 < C; i0 += WARP) { const int i = i0 + lane; const unsigned long long d = ldc(i); const uint32_t k = keyf(d); if (i < C && k) f(k, (int)(d >> 32)); }
    };
    uint32_t T; int above, ce;
    if (C <= RC * WARP) wave_select_relradix<RC, DPP>(rk, K, RC * WARP - C, hist, lane, T, above, ce);
    else wave_radix<4, DPP>(each, hist, K, lane, T, above, ce);
    wave_emit([&](auto&& g) {
        // No validity flag needed: an empty slot's key 0 is never > T, and equals T only when T = 0, which leaves no
        // key to emit at T (T = 0 only comes from a boundary bin with every key selected).
        #pragma unroll
        for (int j = 0; j < RC; ++j) if (j * WARP < C) g(true, rk[j], rc[j]);
        for (int i0 = RC * WARP; i0 < C; i0 += WARP) {
            const int i = i0 + lane; const bool v = i < C;
            const unsigned long long d = ldc(i);
            g(v, keyf(d), (int)(d >> 32));
        }
    }, T, above, K, ks, lane, orow);
    return true;
}

// RC from the candidate count of the row (wave-uniform): fewer registers to compare in every search step
template <bool COHERENT, bool DPP = true>
__device__ __forceinline__ bool wave_select_row_auto(const unsigned long long* __restrict__ cand, const int* __restrict__ cnt,
                                                     int nsplit, int CAP, int M, int m, int ks, int nvalid, int K,
                                                     int* __restrict__ orow, uint32_t* hist, int lane) {
    if constexpr (COHERENT) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    int c_l = cnt[(int64_t)min(lane, nsplit - 1) * M + m];
    if (lane >= nsplit) c_l = 0;
    int C = min(c_l, CAP);
    #pragma unroll
    for (int d = 32; d >= 1; d >>= 1) C += __shfl_xor(C, d);
    if (C <= 16 * WARP) return wave_select_row<16, COHERENT, DPP>(cand, cnt, nsplit, CAP, M, m, ks, nvalid, K, orow, hist, lane);
    if (C <= 24 * WARP) return wave_select_row<24, COHERENT, DPP>(cand, cnt, nsplit, CAP, M, m, ks, nvalid, K, orow, hist, lane);
    if (C <= 32 * WARP) return wave_select_row<32, COHERENT, DPP>(cand, cnt, nsplit, CAP, M, m, ks, nvalid, K, orow, hist, lane);
    return wave_select_row<48, COHERENT, DPP>(cand, cnt, nsplit, CAP, M, m, ks, nvalid, K, orow, hist, lane);
}

// One wave per row; a row that cannot prove its top-K from its candidates is appended to flag[] (count *fbc).
__global__ __launch_bounds__(256) void select_kernel(const unsigned long long* __restrict__ cand, const int* __restrict__ cnt, int nsplit, int CAP,
                                                     const int* __restrict__ ks_p, const int* __restrict__ ke_p, int M, int N, int K,
                                                     int* __restrict__ out, int* __restrict__ flag, int* __restrict__ fbc) {
    __shared__ uint32_t hist_all[4][256];
    const int lane = threadIdx.x % WARP, w = threadIdx.x / WARP, m = blockIdx.x * 4 + w;
    if (m >= M) return;
    const int ks = max(ks_p[m], 0), nvalid = min(ke_p[m], N) - ks;
    int* orow = out + (int64_t)m * K;
    if (nvalid <= K) {   // every key, in order, then -1
        for (int i = lane; i < K; i += WARP) orow[i] = i < nvalid ? i : -1;
        return;
    }
    if (!wave_select_row_auto<false, false>(cand, cnt, nsplit, CAP, M, m, ks, nvalid, K, orow, hist_all[w], lane))
        if (lane == 0) flag[atomicAdd(fbc, 1)] = m;
}

// ------------------------------------------------------------------------------------------------- exact recompute
// Exact v35wes logit of (row m, token tok) for the 32 tokens held by lanes l % 32 (all 64 lanes take part); result
// valid in lanes 0..31. Same operand layout, plain MFMA, scalar even/odd fma chains, permlane pairing (hset 0 half +
// hset 1 half) and kv_scale multiply as mqa_v35wes::kernel -> bit-identical.
struct QW { f8x32 q0, q1; f32x2 w2[8]; };
__device__ __forceinline__ QW load_qw(int m, const fp8* __restrict__ Q, const float* __restrict__ W) {
    const int l = threadIdx.x % WARP, nA = l % 32, dhalf = l / 32, hset = l / 32;
    const int64_t qb = ((int64_t)m * NUM_HEADS + nA) * HEAD_SIZE;
    QW r;
    r.q0 = pack(*reinterpret_cast<const f8x16*>(&Q[qb + dhalf * 16]), *reinterpret_cast<const f8x16*>(&Q[qb + (2 + dhalf) * 16]));
    r.q1 = pack(*reinterpret_cast<const f8x16*>(&Q[qb + (4 + dhalf) * 16]), *reinterpret_cast<const f8x16*>(&Q[qb + (6 + dhalf) * 16]));
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const f32x4 wv = *reinterpret_cast<const f32x4*>(&W[(int64_t)m * NUM_HEADS + 8 * j + 4 * hset]);
        r.w2[2 * j] = f32x2{wv[0], wv[1]};
        r.w2[2 * j + 1] = f32x2{wv[2], wv[3]};
    }
    return r;
}
__device__ __forceinline__ float recompute_wes(const QW& qw, int tok, const fp8* __restrict__ Kc, const float* __restrict__ kv_scale) {
    const int l = threadIdx.x % WARP, dhalf = l / 32;
    const int64_t kb = (int64_t)tok * HEAD_SIZE;
    const f8x32 b0 = pack(*reinterpret_cast<const f8x16*>(&Kc[kb + dhalf * 16]), *reinterpret_cast<const f8x16*>(&Kc[kb + (2 + dhalf) * 16]));
    const f8x32 b1 = pack(*reinterpret_cast<const f8x16*>(&Kc[kb + (4 + dhalf) * 16]), *reinterpret_cast<const f8x16*>(&Kc[kb + (6 + dhalf) * 16]));
    const float scale = kv_scale[tok];
    VecOut z = {};
    VecOut c = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(qw.q0, b0, z, 0, 0, 0, 0, 0, 0);
    c = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(qw.q1, b1, c, 0, 0, 0, 0, 0, 0);
    float ae = 0.f, ao = 0.f;
    #pragma unroll
    for (int v = 0; v < 8; ++v) {
        const float ye = __builtin_fmaxf(c[2 * v], 0.f), yo = __builtin_fmaxf(c[2 * v + 1], 0.f);
        if (v == 0) { ae = ye * qw.w2[0][0]; ao = yo * qw.w2[0][1]; }
        else { ae = __builtin_fmaf(ye, qw.w2[v][0], ae); ao = __builtin_fmaf(yo, qw.w2[v][1], ao); }
    }
    float p0 = ae + ao, p1 = p0;
    swap32_reduce2<2>(p0, p1);      // lanes 0-31: (own hset-0 half, hset-1 half of lane + 32)
    return (p0 + p1) * scale;
}

// exact select of one row from a dense fp32 scratch row srow[ks, ke) (one wave)
template <bool COHERENT>
__device__ __forceinline__ void wave_select_dense(const float* __restrict__ srow, int ks, int ke, int K, int* __restrict__ orow,
                                                  uint32_t* hist, int lane) {
    if (ke - ks <= K) { for (int i = lane; i < K; i += WARP) orow[i] = i < ke - ks ? i : -1; return; }
    if constexpr (COHERENT) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    auto ld = [&](int c) -> uint32_t { return okey32(__builtin_bit_cast(uint32_t, srow[min(c, ke - 1)])); };
    auto each = [&](auto&& f) { for (int c0 = ks; c0 < ke; c0 += WARP) { const int c = c0 + lane; const uint32_t k = ld(c); if (c < ke) f(k, c); } };
    uint32_t T; int above, ce;
    wave_radix<4>(each, hist, K, lane, T, above, ce);
    wave_emit([&](auto&& g) {
        for (int c0 = ks; c0 < ke; c0 += WARP) { const int c = c0 + lane; const bool v = c < ke; const uint32_t k = ld(c); g(v, v ? k : 0u, c); }
    }, T, above, K, ks, lane, orow);
}

// ------------------------------------------------------------------------------------------------- fallback (3 kernels)
// The first FBR flagged rows (in list order) are recomputed by every wave of the grid in parallel (fb_compute) and
// selected one wave per row (fb_select); further flagged rows fall back to one wave per row (fallback_kernel).
constexpr int FBR = 64;
__global__ __launch_bounds__(256) void fb_compute(const int* __restrict__ flag, const int* __restrict__ fbc, const int* __restrict__ ks_p, const int* __restrict__ ke_p,
                                                  int M, int N, float* __restrict__ scratch, const fp8* __restrict__ Q, const fp8* __restrict__ Kc,
                                                  const float* __restrict__ kv_scale, const float* __restrict__ W) {
    const int lane = threadIdx.x % WARP, gw = blockIdx.x * 4 + threadIdx.x / WARP, G = gridDim.x * 4;
    const int nf = min(*fbc, FBR);
    for (int ord = 0; ord < nf; ++ord) {
        const int m = flag[ord];
        const int ks = max(ks_p[m], 0), ke = min(ke_p[m], N);
        const int nb = (ke - ks + 31) / 32;
        if (gw >= nb) continue;
        const QW qw = load_qw(m, Q, W);
        for (int b = gw; b < nb; b += G) {
            const int c0 = ks + b * 32, tok = min(c0 + lane % 32, ke - 1);
            const float v = recompute_wes(qw, tok, Kc, kv_scale);
            if (lane < 32 && c0 + lane < ke) scratch[(int64_t)ord * N + c0 + lane] = v;
        }
    }
}
__global__ __launch_bounds__(256) void fb_select(const int* __restrict__ flag, const int* __restrict__ fbc, const int* __restrict__ ks_p, const int* __restrict__ ke_p,
                                                 int M, int N, int K, int* __restrict__ out, const float* __restrict__ scratch, int* __restrict__ nfb) {
    __shared__ uint32_t hist_all[4][256];
    const int lane = threadIdx.x % WARP, w = threadIdx.x / WARP, ord = blockIdx.x * 4 + w;
    if (ord >= min(*fbc, FBR)) return;
    const int m = flag[ord];
    if (nfb && lane == 0) atomicAdd(nfb, 1);
    const int ks = max(ks_p[m], 0), ke = min(ke_p[m], N);
    wave_select_dense<false>(scratch + (int64_t)ord * N, ks, ke, K, out + (int64_t)m * K, hist_all[w], lane);
}
// beyond FBR flagged rows: one wave per row, own scratch row
__global__ __launch_bounds__(256) void fallback_kernel(const int* __restrict__ flag, const int* __restrict__ fbc, const int* __restrict__ ks_p,
                                                       const int* __restrict__ ke_p, int M, int N, int K, int* __restrict__ out, float* __restrict__ scratch,
                                                       const fp8* __restrict__ Q, const fp8* __restrict__ Kc, const float* __restrict__ kv_scale,
                                                       const float* __restrict__ W, int* __restrict__ nfb) {
    __shared__ uint32_t hist_all[4][256];
    const int lane = threadIdx.x % WARP, w = threadIdx.x / WARP, gw = blockIdx.x * 4 + w;
    uint32_t* hist = hist_all[w];
    float* srow = scratch + (int64_t)(FBR + gw) * N;
    const int nf = *fbc;
    for (int ord = FBR + gw; ord < nf; ord += gridDim.x * 4) {
        const int m = flag[ord];
        if (nfb && lane == 0) atomicAdd(nfb, 1);
        const int ks = max(ks_p[m], 0), ke = min(ke_p[m], N);
        const QW qw = load_qw(m, Q, W);
        for (int c0 = ks; c0 < ke; c0 += 32) {
            const int tok = min(c0 + lane % 32, ke - 1);
            const float v = recompute_wes(qw, tok, Kc, kv_scale);
            if (lane < 32 && c0 + lane < ke) srow[c0 + lane] = v;
        }
        __threadfence_block();
        wave_select_dense<true>(srow, ks, ke, K, out + (int64_t)m * K, hist, lane);
    }
}

}  // namespace mqa_topk
