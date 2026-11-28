// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Arch-neutral selection of mqa_topk_bf16 (wave64): the per-row threshold from the samples (tau)
// and the exact top-k of a row's candidates, stored in ascending index order.
//
// The order is (score desc, index asc): a candidate is the 64-bit key (order_key(score), ~index),
// unique, so the top-k is one set whatever order the candidates come in. Radix selection on the
// score key relative to the row's smallest (8-bit digits from the span's top bit: the digits
// spread over the scores the row has, not over the float's exponent), then, if a score is shared
// across the cut, on the index. The selected set is written in index order through a bitmap of
// the window, SEL_SEG keys at a time.
//
// A source hands each thread its candidates: for_each(f) calls f(key, index, value) on them.
#pragma once
#include "mqa_topk_bf16_common.cuh"

namespace opus_mqa_topk {

constexpr int SEL_BLOCK = 256;
constexpr int SEL_WAVES = SEL_BLOCK / 64;
constexpr int SEL_SEG   = 131072;                 // keys a bitmap pass covers
constexpr int SEL_WORDS = SEL_SEG / 32;
constexpr int SEL_WPT   = SEL_WORDS / SEL_BLOCK;  // bitmap words a thread counts, at most
constexpr int SEL_DIGIT = 11;                     // score digits: two passes resolve most rows
constexpr int SEL_BINS  = 1 << SEL_DIGIT;
constexpr int SEL_BPT   = SEL_BINS / SEL_BLOCK;   // bins a thread scans

struct select_smem {
    unsigned hist[2][SEL_BINS];   // radix passes alternate: a pass zeroes the next one's
    unsigned bitmap[SEL_WORDS];
    int tbase[SEL_BLOCK];
    int wsum[SEL_WAVES];
    unsigned red[2][SEL_WAVES];
    unsigned bc[4];
};

// -- block helpers (SEL_BLOCK threads) ------------------------------------------------------------
__device__ inline int block_excl_scan(int v, select_smem& sm, int tid, int& total) {
    const int incl = wave_incl_scan(v);
    const int wave = tid / 64;
    if ((tid & 63) == 63) sm.wsum[wave] = incl;
    __syncthreads();
    int before = 0;
    total      = 0;
    for (int x = 0; x < SEL_WAVES; ++x) {
        const int s = sm.wsum[x];
        before += x < wave ? s : 0;
        total += s;
    }
    __syncthreads();
    return before + incl - v;
}

__device__ inline void block_minmax(unsigned& mn, unsigned& mx, select_smem& sm, int tid) {
    mn = wave_min(mn);
    mx = wave_max(mx);
    if ((tid & 63) == 0) {
        sm.red[0][tid / 64] = mn;
        sm.red[1][tid / 64] = mx;
    }
    __syncthreads();
    mn = sm.red[0][0];
    mx = sm.red[1][0];
    for (int x = 1; x < SEL_WAVES; ++x) {
        mn = sm.red[0][x] < mn ? sm.red[0][x] : mn;
        mx = sm.red[1][x] > mx ? sm.red[1][x] : mx;
    }
}

// The selection: rel = key - kmin; selected iff (rel >> sh) > (P >> sh), or == and (all || index
// <= last).
struct select_cut {
    unsigned kmin, P;
    int sh;
    bool all;
    int last;
    __device__ bool operator()(unsigned key, int idx) const {
        const unsigned a = (key - kmin) >> sh, b = P >> sh;
        return a > b || (a == b && (all || idx <= last));
    }
};

// The digit d (thread d of 256 holds hist[d]) at which `need` is reached counting from the top
// (TOP) or the bottom; returns (d, the count past it, hist[d]) to every thread.
template<bool TOP>
__device__ inline void find_digit(select_smem& sm, const unsigned* h, int tid, int need,
                                  unsigned& d, int& before, int& bucket) {
    const int v  = (int)h[TOP ? 255 - tid : tid];
    int total    = 0;
    const int bf = block_excl_scan(v, sm, tid, total);
    if (bf < need && need <= bf + v) {
        sm.bc[0] = TOP ? 255 - tid : tid;
        sm.bc[1] = bf;
        sm.bc[2] = v;
    }
    __syncthreads();
    d      = sm.bc[0];
    before = (int)sm.bc[1];
    bucket = (int)sm.bc[2];
    __syncthreads();
}

// The same over SEL_BINS bins, SEL_BPT a thread.
template<bool TOP>
__device__ inline void find_digit_wide(select_smem& sm, const unsigned* h, int tid, int need,
                                       unsigned& d, int& before, int& bucket) {
    const int b0 = TOP ? SEL_BINS - 1 - SEL_BPT * tid : SEL_BPT * tid;
    int c[SEL_BPT], v = 0;
#pragma unroll
    for (int x = 0; x < SEL_BPT; ++x) {
        c[x] = (int)h[TOP ? b0 - x : b0 + x];
        v += c[x];
    }
    int total = 0;
    int bf    = block_excl_scan(v, sm, tid, total);
    if (bf < need && need <= bf + v) {
#pragma unroll
        for (int x = 0; x < SEL_BPT; ++x) {
            if (need <= bf + c[x]) {
                sm.bc[0] = TOP ? b0 - x : b0 + x;
                sm.bc[1] = bf;
                sm.bc[2] = c[x];
                break;
            }
            bf += c[x];
        }
    }
    __syncthreads();
    d      = sm.bc[0];
    before = (int)sm.bc[1];
    bucket = (int)sm.bc[2];
    __syncthreads();
}

// The top W of more than W candidates.
template<class Src>
__device__ select_cut radix_select(const Src& src, int W, select_smem& sm, int tid) {
    unsigned mn = ~0u, mx = 0u;
    src.for_each([&](unsigned key, int, float) {
        mn = key < mn ? key : mn;
        mx = key > mx ? key : mx;
    });
    block_minmax(mn, mx, sm, tid);
    const unsigned span = mx - mn;
    int hi_sh  = span ? 32 - __builtin_clz(span) : 0;   // bits at and above hi_sh: resolved
    unsigned P = 0;
    int need   = W;
    int cur    = 0;   // the histogram this pass fills; the other is zeroed meanwhile
    for (int x = tid; x < SEL_BINS; x += SEL_BLOCK) {
        sm.hist[0][x] = 0;
        sm.hist[1][x] = 0;
    }
    __syncthreads();
    // Score digits, top down.
    while (true) {
        const int sh = hi_sh > SEL_DIGIT ? hi_sh - SEL_DIGIT : 0;
        unsigned* h  = sm.hist[cur];
        src.for_each([&](unsigned key, int, float) {
            const unsigned r = key - mn;
            if (hi_sh >= 32 || (r >> hi_sh) == (P >> hi_sh))
                atomicAdd(&h[(r >> sh) & (SEL_BINS - 1u)], 1u);
        });
        __syncthreads();
        for (int x = tid; x < SEL_BINS; x += SEL_BLOCK) sm.hist[cur ^ 1][x] = 0;
        unsigned d;
        int above, bucket;
        find_digit_wide<true>(sm, h, tid, need, d, above, bucket);
        cur ^= 1;
        P |= d << sh;
        need -= above;
        if (bucket == need) return {mn, P, sh, true, 0};
        if (sh == 0) break;
        hi_sh = sh;
    }
    // A score shared across the cut: the `need` smallest indices of the candidates on it.
    unsigned I = 0;
    for (int sh = 16; sh >= 0; sh -= 8) {
        unsigned* h = sm.hist[cur];
        src.for_each([&](unsigned key, int idx, float) {
            const unsigned x = (unsigned)idx;
            if (key - mn == P && (sh == 16 || (x >> (sh + 8)) == (I >> (sh + 8))))
                atomicAdd(&h[(x >> sh) & 255u], 1u);
        });
        __syncthreads();
        sm.hist[cur ^ 1][tid] = 0;
        unsigned d;
        int below, bucket;
        find_digit<false>(sm, h, tid, need, d, below, bucket);
        cur ^= 1;
        I |= d << sh;
        need -= below;
        if (bucket == need) return {mn, P, 0, false, (int)(I | ((1u << sh) - 1))};
    }
    return {mn, P, 0, false, (int)I};
}

// Writes the selected candidates of window [lo, hi) to out_i / out_v (or null) in index order.
// A pass covers up to SEL_SEG keys, its bitmap only as many words as the window needs.
template<class Src, class Sel>
__device__ void write_ascending(const Src& src, const Sel& sel, int lo, int hi, int* out_i,
                                float* out_v, select_smem& sm, int tid) {
    int base = 0;
    for (int seg = lo; seg < hi; seg += SEL_SEG) {
        const int len   = hi - seg < SEL_SEG ? hi - seg : SEL_SEG;
        const int words = (len + 31) / 32;
        const int wpt   = (words + SEL_BLOCK - 1) / SEL_BLOCK;   // words a thread counts
        for (int x = tid; x < words; x += SEL_BLOCK) sm.bitmap[x] = 0;
        __syncthreads();
        src.for_each([&](unsigned key, int idx, float) {
            const unsigned x = (unsigned)(idx - seg);
            if (x < (unsigned)len && sel(key, idx)) atomicOr(&sm.bitmap[x >> 5], 1u << (x & 31));
        });
        __syncthreads();
        int c = 0;
        for (int x = tid * wpt; x < tid * wpt + wpt && x < words; ++x)
            c += __builtin_popcount(sm.bitmap[x]);
        int total     = 0;
        sm.tbase[tid] = block_excl_scan(c, sm, tid, total);
        __syncthreads();
        src.for_each([&](unsigned key, int idx, float val) {
            const unsigned x = (unsigned)(idx - seg);
            if (x < (unsigned)len && sel(key, idx)) {
                const int w = x >> 5, t = w / wpt;
                int pos     = base + sm.tbase[t];
                for (int y = t * wpt; y < w; ++y) pos += __builtin_popcount(sm.bitmap[y]);
                pos += __builtin_popcount(sm.bitmap[w] & ((1u << (x & 31)) - 1u));
                out_i[pos] = idx;
                if (out_v) out_v[pos] = val;
            }
        });
        base += total;
        __syncthreads();
    }
}

// The row's output from its C candidates in window [lo, hi) (C >= min(n, W); the whole window
// when n <= W): the top W, or all n then -1.
template<class Src>
__device__ void select_row(const Src& src, int C, int W, int width, window wd, int* out_i,
                           float* out_v, select_smem& sm, int tid) {
    const int n = wd.hi - wd.lo;
    if (n <= W) {
        src.for_each([&](unsigned, int idx, float val) {
            out_i[idx - wd.lo] = idx;
            if (out_v) out_v[idx - wd.lo] = val;
        });
        for (int p = n + tid; p < width; p += SEL_BLOCK) {
            out_i[p] = -1;
            if (out_v) out_v[p] = -__builtin_inff();
        }
        return;
    }
    (void)C;
    const select_cut cut = radix_select(src, W, sm, tid);
    write_ascending(src, cut, wd.lo, wd.hi, out_i, out_v, sm, tid);
}

// A row's candidates in registers, CPT a thread: thread t holds candidates t + 256 x.
template<int CPT>
struct reg_source {
    unsigned key[CPT];
    int idx[CPT];   // -1: no candidate
    template<class F>
    __device__ void for_each(F&& f) const {
#pragma unroll
        for (int x = 0; x < CPT; ++x)
            if (idx[x] >= 0) f(key[x], idx[x], order_value(key[x]));
    }
};

// A row's dense scores, index lo + i at i, n of them.
struct dense_source {
    const float* s;
    int lo, n, tid;
    template<class F>
    __device__ void for_each(F&& f) const {
        for (int i = tid; i < n; i += SEL_BLOCK) {
            const unsigned key = order_key(s[i]);
            f(key, lo + i, order_value(key));
        }
    }
};

// -- tau: a wave a row ----------------------------------------------------------------------------
// The m-th largest of a row's ns samples, their order keys staged in keys[] (wave-private).
// Wave-private 256-bin histogram.
__device__ inline unsigned wave_kth_largest(const unsigned* keys, int ns, int m, unsigned* hist,
                                            int lane) {
    unsigned mn = ~0u, mx = 0u;
    for (int j = lane; j < ns; j += 64) {
        const unsigned key = keys[j];
        mn = key < mn ? key : mn;
        mx = key > mx ? key : mx;
    }
    mn = wave_min(mn);
    mx = wave_max(mx);
    const unsigned span = mx - mn;
    int hi_sh  = span ? 32 - __builtin_clz(span) : 0;
    unsigned P = 0;
    int need   = m;
    while (true) {
        const int sh = hi_sh > 8 ? hi_sh - 8 : 0;
        for (int b = lane; b < 256; b += 64) hist[b] = 0;
        __builtin_amdgcn_wave_barrier();
        for (int j = lane; j < ns; j += 64) {
            const unsigned r = keys[j] - mn;
            if (hi_sh >= 32 || (r >> hi_sh) == (P >> hi_sh)) atomicAdd(&hist[(r >> sh) & 255u], 1u);
        }
        __builtin_amdgcn_wave_barrier();
        // lane l: digits 255 - 4 l .. 252 - 4 l, top down
        const int b0 = 255 - 4 * lane;
        int c[4];
        for (int x = 0; x < 4; ++x) c[x] = (int)hist[b0 - x];
        const int v     = c[0] + c[1] + c[2] + c[3];
        const int incl  = wave_incl_scan(v);
        const int above = incl - v;
        const unsigned long long hit = __builtin_amdgcn_ballot_w64(above < need && need <= incl);
        const int L = __builtin_ctzll(hit);
        int d = 0, a = above, bucket = 0;
        for (int x = 0; x < 4; ++x) {
            if (need <= a + c[x]) {
                d      = b0 - x;
                bucket = c[x];
                break;
            }
            a += c[x];
        }
        d      = __builtin_amdgcn_readlane(d, L);
        a      = __builtin_amdgcn_readlane(a, L);
        bucket = __builtin_amdgcn_readlane(bucket, L);
        P |= (unsigned)d << sh;
        need -= a;
        if (bucket == need || sh == 0) break;
        hi_sh = sh;
    }
    return mn + P;
}

}  // namespace opus_mqa_topk
