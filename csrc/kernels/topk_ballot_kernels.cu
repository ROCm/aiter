// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

// Ballot radix top-k for one float row per query, decode layout.
// The k largest scores win, ties keep the smallest indices, and the indices
// come out ascending: the same answer as top_k_per_row_decode(stable=True).
//
// A workgroup of 512 threads holds its keys in registers and reads them once.
// Every count is a compare and a popcount of a ballot; there is no histogram
// and no LDS atomic on a key.
//   1. Each lane keeps four sub-maxima of its keys. Each wave bounds the
//      ceil(k / waves)-th largest of its 256 sub-maxima from below; they are
//      distinct keys, so every wave holds that many keys at or above its bound
//      and the smallest bound over the waves is at or below the kth key.
//   2. Each wave stages its keys at or above the bound in its own LDS region,
//      in index order, in the same pass that counts them. If a region
//      overflows, the bound is raised a bit at a time over all the keys.
//   3. The kth key is found a bit at a time over the staged keys: each wave
//      popcounts its keys at or above the trial threshold and the wave counts
//      meet in LDS. Once at most kFinish keys are left undecided, wave 0
//      finishes the bits alone, with no barrier per bit.
//   4. Each wave writes its winners in index order, so nothing is sorted.
// When the kth key is resolved and still more keys tie with it than the
// regions hold, the winners are written straight from the registers.
//
// Unsplit, one workgroup does all of this for its row. Split, each of `parts`
// workgroups runs steps 1 and 2 on one slice of the row and writes what it
// staged to a workspace; the last of the row to finish merges the slices,
// repeating steps 2 to 4 on that union. A short row leaves most CUs idle, so
// splitting it divides the steps that scale with its length.
//
// A wave-uniform instruction costs every wave its turn, so the scalar work of
// each round is paid once per wave. Fewer, register-heavier waves keep that
// cost down.
//
// Standalone: no aiter module builds this file. Build it with
//   hipcc -O3 --offload-arch=gfx942 -std=c++17 -shared -fPIC -o topk_ballot.so topk_ballot_kernels.cu
// and call the extern "C" launchers at the end of the file.
// op_tests/op_benchmarks/hip/bench_topk_ballot.py checks and times it.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

#ifdef TOPK_TRACE
// Phase stamps from thread 0 of workgroup 0, in 100 MHz ticks.
__device__ long long g_trace[32];
#define STAMP(i) \
    do { \
        if (blockIdx.x == 0 && threadIdx.x == 0) g_trace[i] = __builtin_amdgcn_s_memrealtime(); \
    } while (0)
#define NOTE(i, v) \
    do { \
        if (blockIdx.x == 0 && threadIdx.x == 0) g_trace[i] = (v); \
    } while (0)

extern "C" int topk_trace_read(long long* host) {
    return static_cast<int>(hipMemcpyFromSymbol(host, HIP_SYMBOL(g_trace), sizeof(g_trace)));
}
extern "C" int topk_trace_clear() {
    long long z[32] = {};
    return static_cast<int>(hipMemcpyToSymbol(HIP_SYMBOL(g_trace), z, sizeof(z)));
}
#else
#define STAMP(i)
#define NOTE(i, v)
#endif

constexpr int kWave = 64;
constexpr int kGroup = 4 * kWave;   // floats one float4 load covers across a wave
constexpr int kCap = 4096;          // keys staged in LDS, over all regions
constexpr int kSub = 4;             // sub-maxima per lane for the bound
constexpr int kBoundBits = 10;      // bits of the bound resolved below its known prefix
constexpr int kFinish = 4 * kWave;  // undecided keys wave 0 takes over

// fp32_to_sortable_bits from topk_fp32_key.h: a larger float is a larger key,
// and a NaN ranks by its sign. Key 0 marks an empty slot; it is the key of the
// one NaN 0xffffffff, which is therefore never selected. Every other value,
// that NaN's neighbours included, ranks as top_k_per_row_decode ranks it.
constexpr unsigned kEmptyBits = 0xffffffffu;

__device__ __forceinline__ unsigned key_of_bits(unsigned u) {
    return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}

__device__ __forceinline__ int lane_id() { return threadIdx.x & (kWave - 1); }

__device__ __forceinline__ int uniform(int v) { return __builtin_amdgcn_readfirstlane(v); }
__device__ __forceinline__ unsigned uniform(unsigned v) {
    return static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(v)));
}

__device__ __forceinline__ int wave_id() { return uniform(static_cast<int>(threadIdx.x) / kWave); }

// The same uniform value, hidden from the optimizer. A loop that compares
// every key against `opaque(x)` recomputes the compares instead of keeping one
// 64-bit mask per key alive from an earlier loop against x, which runs the
// workgroup out of SGPRs and spills.
__device__ __forceinline__ unsigned opaque(unsigned v) {
    v = uniform(v);
    asm volatile("" : "+s"(v));
    return v;
}

// A per-lane value the optimizer cannot move: indices computed from it inside
// a loop are recomputed there rather than hoisted out and held, one register
// per key, across the loop.
__device__ __forceinline__ int pinned(int v) {
    asm volatile("" : "+v"(v));
    return v;
}

__device__ __forceinline__ int ballot_count(bool p) { return __popcll(__ballot(p)); }

// Set lanes below this one.
__device__ __forceinline__ int rank_in_wave(unsigned long long m) {
    return __builtin_amdgcn_mbcnt_hi(static_cast<unsigned>(m >> 32),
                                     __builtin_amdgcn_mbcnt_lo(static_cast<unsigned>(m), 0u));
}

template <int Ctrl>
__device__ __forceinline__ unsigned dpp(unsigned v) {
    return static_cast<unsigned>(
        __builtin_amdgcn_update_dpp(0, static_cast<int>(v), Ctrl, 0xf, 0xf, false));
}

// Reduce each row of 16 lanes with DPP. Every lane must be active.
template <typename Op>
__device__ __forceinline__ unsigned row_reduce(unsigned v, Op op) {
    v = op(v, dpp<0xb1>(v));   // quad_perm [1,0,3,2]
    v = op(v, dpp<0x4e>(v));   // quad_perm [2,3,0,1]
    v = op(v, dpp<0x141>(v));  // row_half_mirror
    v = op(v, dpp<0x140>(v));  // row_mirror
    return v;
}

template <typename Op>
__device__ __forceinline__ unsigned wave_reduce(unsigned v, Op op) {
    v = row_reduce(v, op);
    const unsigned a = __builtin_amdgcn_readlane(v, 0), b = __builtin_amdgcn_readlane(v, 16);
    const unsigned c = __builtin_amdgcn_readlane(v, 32), d = __builtin_amdgcn_readlane(v, 48);
    return op(op(a, b), op(c, d));
}

struct Add {
    __device__ unsigned operator()(unsigned a, unsigned b) const { return a + b; }
};
struct Max {
    __device__ unsigned operator()(unsigned a, unsigned b) const { return a > b ? a : b; }
};
struct Min {
    __device__ unsigned operator()(unsigned a, unsigned b) const { return a < b ? a : b; }
};

__device__ __forceinline__ int wave_sum(int v) {
    return static_cast<int>(wave_reduce(static_cast<unsigned>(v), Add{}));
}

// The bits above the highest one where lo and hi differ, and that bit.
__device__ __forceinline__ unsigned common_prefix(unsigned lo, unsigned hi, int& top) {
    top = 31 - __clz(lo ^ hi);
    return lo & ~((2u << top) - 1u);
}

// One int per wave in LDS. Lanes 0..Waves-1 read them and row 0 reduces them:
// a full-wave read of the same few bytes still moves 64 copies through LDS,
// from every wave.
template <int Waves>
struct WaveSlots {
    static_assert(Waves <= 16);
    int* slot;

    __device__ __forceinline__ void put(int v) const {
        if (lane_id() == 0) slot[wave_id()] = v;
    }
    __device__ __forceinline__ unsigned get(unsigned none) const {
        const int lane = lane_id();
        return lane < Waves ? static_cast<unsigned>(slot[lane]) : none;
    }
    __device__ __forceinline__ int sum() const {
        return static_cast<int>(__builtin_amdgcn_readlane(row_reduce(get(0u), Add{}), 0));
    }
    // Over the waves ahead of this one.
    __device__ __forceinline__ int sum_before() const {
        const unsigned v = lane_id() < wave_id() ? get(0u) : 0u;
        return static_cast<int>(__builtin_amdgcn_readlane(row_reduce(v, Add{}), 0));
    }
    __device__ __forceinline__ unsigned max() const {
        return __builtin_amdgcn_readlane(row_reduce(get(0u), Max{}), 0);
    }
    __device__ __forceinline__ unsigned min() const {
        return __builtin_amdgcn_readlane(row_reduce(get(~0u), Min{}), 0);
    }
};

// Four consecutive keys of one lane, by value so they stay in registers.
struct Key4 {
    unsigned v0, v1, v2, v3;
    __device__ __forceinline__ unsigned operator[](int c) const {
        return c == 0 ? v0 : c == 1 ? v1 : c == 2 ? v2 : v3;
    }
    __device__ __forceinline__ unsigned max() const { return ::max(::max(v0, v1), ::max(v2, v3)); }
};

// A lower bound on the r-th largest of the wave's N * 64 values: the largest t
// with at least r of them >= t, exact for kBoundBits bits below the prefix the
// wave's smallest and largest value share, and zero below that.
template <int N>
__device__ __forceinline__ unsigned wave_rank_floor(const unsigned (&v)[N], int r) {
    unsigned lo = v[0], hi = v[0];
#pragma unroll
    for (int i = 1; i < N; ++i) lo = min(lo, v[i]), hi = max(hi, v[i]);
    lo = wave_reduce(lo, Min{});
    hi = wave_reduce(hi, Max{});
    if (lo == hi) return lo;
    int b;
    unsigned t = common_prefix(lo, hi, b);
    const int stop = max(b - kBoundBits, -1);
    for (; b > stop; --b) {
        const unsigned cand = t | (1u << b);
        int c = 0;
#pragma unroll
        for (int i = 0; i < N; ++i) c += ballot_count(v[i] >= cand);
        if (c >= r) t = cand;
    }
    return t;
}

constexpr int kMaxParts = 8;       // workgroups one row may be split across
constexpr int kMinParts = 4;       // fewer save less than the merge costs
constexpr int kSplitMin = 75000;   // a shorter row is faster unsplit

// A workgroup's staged keys: one region per wave, each followed by one load's
// worth of slack so a store past the region lands there and not in the next.
template <int Waves>
struct Regions {
    static constexpr int Region = kCap / Waves;
    static constexpr int Stride = Region + kGroup;
    static constexpr int Slots = Region / kWave;
};

enum SlotId { kLoSlot, kHiSlot, kNSlot, kGtSlot, kEqSlot, kCntSlot, kPartSlot = kCntSlot + 2, kSlots };

// The workgroup's LDS, by pointer.
struct Lds {
    unsigned* key;
    int* idx;
    unsigned* fin;
    int* fin_n;
    unsigned* t;
    int* slot;  // kSlots rows of 16 ints
};

template <int Waves>
__device__ __forceinline__ WaveSlots<Waves> slot_of(Lds l, int id) {
    return {l.slot + 16 * id};
}

// What the slices of one split row share while they run. All zero between
// launches; the merging workgroup clears it.
struct RowState {
    int done;               // slices finished
    unsigned floor_k;       // the best floor_k published so far
    unsigned not_part_min;  // ~(smallest floor_part published), so zero means none
    int published;          // slices that have published their floors
};

// What one slice of a split row hands the merge.
struct SliceHead {
    int count;            // entries written
    unsigned floor_k;     // at least k of the slice's keys are >= this; 0 when none is known
    unsigned floor_part;  // at least ceil(k / parts) of them are >= this
    int unused;
};

// The kth key and its winners, from keys staged in the regions. Wave w's
// region holds `mine` keys, in index order, all >= bound, and waves are in
// index order. On entry t carries the bits above b: at least k staged keys
// are >= t and hi_n (< k) are >= t + 2^(b+1). sink(pos, i) writes winner i at
// output position pos.
template <int Waves, typename Sink>
__device__ __forceinline__ void search_and_emit(Lds l, unsigned t, int b, unsigned bound,
                                                int staged, int hi_n, int mine, int parity,
                                                int k, Sink sink) {
    using R = Regions<Waves>;
    const int lane = lane_id();
    const int wave = wave_id();
    int lo_n = staged;

    // This wave's staged keys: slot e of lane l is region entry e * 64 + l, so
    // slot, lane is index order. Entries past the wave's count hold 0, which
    // no threshold reaches.
    unsigned ck[R::Slots];
#pragma unroll
    for (int e = 0; e < R::Slots; ++e) {
        const int p = e * kWave + lane;
        ck[e] = p < mine ? l.key[wave * R::Stride + p] : 0u;
    }

    // Bits over every wave, until few enough keys are undecided.
    while (b >= 0 && lo_n != k && lo_n - hi_n > kFinish) {
        const unsigned cand = t | (1u << b);
        --b;
        if (cand <= bound) {  // counts every staged key, which is >= k
            t = cand;
            continue;
        }
        int c = 0;
#pragma unroll
        for (int e = 0; e < R::Slots; ++e) c += ballot_count(ck[e] >= cand);
        // Two slots: a wave that runs ahead writes the other one, and it cannot
        // come back to this one before every wave has passed the next barrier.
        const WaveSlots<Waves> slot = slot_of<Waves>(l, kCntSlot + parity);
        parity ^= 1;
        slot.put(c);
        __syncthreads();
        const int total = slot.sum();
        if (total >= k) {
            t = cand;
            lo_n = total;
        } else {
            hi_n = total;
        }
    }
    STAMP(4);

    if (b >= 0 && lo_n != k) {
        // The undecided keys, [t, t + 2^(b+1)), go to wave 0, which resolves
        // the rest alone: it needs the (k - hi_n)-th largest of them.
        // Empty slots hold 0, and t can still be 0 here.
        const unsigned from = max(t, 1u);
        const unsigned span_m1 = b >= 31 ? ~0u : (2u << b) - 1u;
        unsigned long long m[R::Slots];
        int n = 0;
#pragma unroll
        for (int e = 0; e < R::Slots; ++e) {
            m[e] = __ballot(ck[e] >= from && ck[e] - t <= span_m1);
            n += __popcll(m[e]);
        }
        if (n) {
            int at = 0;
            if (lane == 0) at = atomicAdd(l.fin_n, n);
            at = __builtin_amdgcn_readlane(at, 0);
#pragma unroll
            for (int e = 0; e < R::Slots; ++e) {
                if ((m[e] >> lane) & 1) l.fin[at + rank_in_wave(m[e])] = ck[e];
                at += __popcll(m[e]);
            }
        }
        __syncthreads();
        if (wave == 0) {
            const int undecided = lo_n - hi_n;
            unsigned fk[kFinish / kWave];
#pragma unroll
            for (int e = 0; e < kFinish / kWave; ++e) {
                const int p = e * kWave + lane;
                fk[e] = p < undecided ? l.fin[p] : 0u;
            }
            const int want = k - hi_n;
            for (; b >= 0; --b) {
                const unsigned cand = t | (1u << b);
                int c = 0;
#pragma unroll
                for (int e = 0; e < kFinish / kWave; ++e) c += ballot_count(fk[e] >= cand);
                if (c >= want) {
                    t = cand;
                    if (c == want) break;
                }
            }
            if (lane == 0) *l.t = t;
        }
        __syncthreads();
        t = uniform(*l.t);
    }
    STAMP(5);

    // Write the staged keys > t, then the earliest == t until k are out. A t
    // below the bound means exactly k keys are staged, all of them winners;
    // the bound is a threshold that says so without matching empty slots.
    const WaveSlots<Waves> gt_slot = slot_of<Waves>(l, kGtSlot);
    const WaveSlots<Waves> eq_slot = slot_of<Waves>(l, kEqSlot);
    const unsigned at = opaque(max(t, bound));
    int gt = 0, eq = 0;
#pragma unroll
    for (int e = 0; e < R::Slots; ++e) {
        gt += ballot_count(ck[e] > at);
        eq += ballot_count(ck[e] == at);
    }
    gt_slot.put(gt);
    eq_slot.put(eq);
    __syncthreads();
    int gt_base = gt_slot.sum_before(), eq_base = eq_slot.sum_before();
    const int need = k - gt_slot.sum();
    const int gt_end = gt_base + gt;
    const unsigned at2 = opaque(at);
#pragma unroll
    for (int e = 0; e < R::Slots; ++e) {
        if (gt_base == gt_end && eq_base >= need) break;
        const unsigned long long mg = __ballot(ck[e] > at2), me = __ballot(ck[e] == at2);
        const int gp = gt_base + rank_in_wave(mg), ep = eq_base + rank_in_wave(me);
        int pos = -1;
        if (ck[e] > at2) pos = gp + min(ep, need);
        else if (ck[e] == at2 && ep < need) pos = gp + ep;
        if (pos >= 0) sink(pos, l.idx[wave * R::Stride + e * kWave + lane]);
        gt_base += __popcll(mg);
        eq_base += __popcll(me);
    }
    STAMP(6);
}

// Steps 1 and 2 for keys [s0, s1) of the row. Unsplit, the selection then
// finishes here. Split, the slice instead writes what it staged, or its exact
// winners when too many keys tie to stage, to `ent`, and its floors to `head`.
template <int Threads, int Vec, bool WriteValues, bool Split>
__device__ __forceinline__ void select_slice(const float* __restrict__ rowp, int s0, int s1, int k,
                                             int parts, Lds l, int* out, float* vout,
                                             SliceHead* head, uint2* ent, RowState* rs) {
    constexpr int Waves = Threads / kWave;
    using R = Regions<Waves>;
    constexpr int Region = R::Region;
    constexpr int Stride = R::Stride;
    const int lane = lane_id();
    const int wave = wave_id();
    const WaveSlots<Waves> lo_slot = slot_of<Waves>(l, kLoSlot);
    const WaveSlots<Waves> hi_slot = slot_of<Waves>(l, kHiSlot);
    const WaveSlots<Waves> n_slot = slot_of<Waves>(l, kNSlot);
    const WaveSlots<Waves> part_slot = slot_of<Waves>(l, kPartSlot);

    STAMP(0);
    // Wave w owns [s0 + w * chunk, s0 + w * chunk + chunk), a multiple of 4
    // wide so float4 loads stay aligned; lane l reads float4 j at
    // s0 + w * chunk + j * kGroup + 4 * l. The slice's last (s1 - s0) % 4 floats
    // are group Vec of the last wave, one per lane. Wave, group, lane, slot is
    // index order.
    //
    // A load that is not issued leaves kEmptyBits behind, so the keys need no
    // mask: a mask per group, kept until the loads land, is a 64-bit SGPR pair
    // per group and runs the workgroup out of SGPRs.
    constexpr int Groups = Vec + 1;
    const int len = s1 - s0;
    const int chunk = ((len + Waves - 1) / Waves + 3) & ~3;
    const int base = s0 + wave * chunk + lane * 4;
    const int whole = s0 + (len & ~3);
    const int end = min(s0 + wave * chunk + chunk, whole);
    // The tail belongs to the one wave whose range holds its first float; a
    // short slice leaves later waves with ranges wholly past its end.
    const int tail =
        lane < len - (len & ~3) && chunk > 0 && (len & ~3) / chunk == wave ? whole + lane : -1;
    Key4 key[Groups];
#pragma unroll
    for (int j = 0; j < Vec; ++j) {
        const int i = base + j * kGroup;
        uint4 v = make_uint4(kEmptyBits, kEmptyBits, kEmptyBits, kEmptyBits);
        if (i < end) v = *reinterpret_cast<const uint4*>(rowp + i);
        key[j] = {key_of_bits(v.x), key_of_bits(v.y), key_of_bits(v.z), key_of_bits(v.w)};
    }
    {
        unsigned v = kEmptyBits;
        if (tail >= 0) v = __float_as_uint(rowp[tail]);
        key[Vec] = {key_of_bits(v), 0u, 0u, 0u};
    }

    {
        unsigned sub[kSub] = {};
#pragma unroll
        for (int j = 0; j < Groups; ++j) sub[j % kSub] = max(sub[j % kSub], key[j].max());
        unsigned top = sub[0];
#pragma unroll
        for (int i = 1; i < kSub; ++i) top = max(top, sub[i]);
        lo_slot.put(static_cast<int>(wave_rank_floor(sub, (k + Waves - 1) / Waves)));
        hi_slot.put(static_cast<int>(wave_reduce(top, Max{})));
        if constexpr (Split) {
            const int r = (k + Waves * parts - 1) / (Waves * parts);
            part_slot.put(static_cast<int>(wave_rank_floor(sub, r)));
        }
    }
    STAMP(1);
    __syncthreads();
    STAMP(2);
    // A real key is never 0, so a bound of 0 is no bound and 1 is the same one.
    unsigned bound = max(lo_slot.min(), 1u);
    const unsigned hi = hi_slot.max();
    const unsigned floor_part = Split ? part_slot.min() : 0u;
    if constexpr (Split) {
        // Publish the floors now, so the other slices can filter by them
        // before they write. Only a bound above 1 is known to have k keys.
        if (threadIdx.x == 0) {
            if (bound > 1u) atomicMax(&rs->floor_k, bound);
            atomicMax(&rs->not_part_min, ~floor_part);
            atomicAdd(&rs->published, 1);
        }
    }

    // The kth key is in [bound, hi]. t carries the bits above b; lo_n keys
    // (>= k) are >= t, and hi_n keys (< k) are >= t + 2^(b+1).
    int b = -1;
    unsigned t = bound;
    if (bound != hi) t = common_prefix(bound, hi, b);
    int hi_n = 0;
    int parity = 0;

    // Stage the keys >= bound, each wave in its own region in index order, and
    // count them, including any past the region.
    int staged, mine;
    bool overflow;
    for (;;) {
        const unsigned at_least = opaque(bound);
        const int region = wave * Stride;
        const int here = pinned(base), there = pinned(tail);
        auto at = [&](int j, int c) { return j < Vec ? here + j * kGroup + c : there; };
        int pos = 0;
#pragma unroll
        for (int j = 0; j < Groups; ++j) {
            unsigned long long m[4];
#pragma unroll
            for (int c = 0; c < 4; ++c) m[c] = __ballot(key[j][c] >= at_least);
            const unsigned long long any = m[0] | m[1] | m[2] | m[3];
            // Most loads hold no key >= the bound. Skip them with a branch:
            // without one the stores below run for every key. An overflowing
            // region is restaged or not used, so it stops here too; a break
            // would cost the unrolling that keeps the keys in registers.
            if (__builtin_expect(any != 0 && pos <= Region, 0)) {
                int p = region + pos + rank_in_wave(m[0]) + rank_in_wave(m[1]) +
                        rank_in_wave(m[2]) + rank_in_wave(m[3]);
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    if (key[j][c] >= at_least) {
                        l.key[p] = key[j][c];
                        l.idx[p] = at(j, c);
                        ++p;
                    }
                }
                pos += __popcll(m[0]) + __popcll(m[1]) + __popcll(m[2]) + __popcll(m[3]);
            }
        }
        mine = pos;
        n_slot.put(pos);
        __syncthreads();
        staged = n_slot.sum();
        unsigned most = n_slot.max();
        overflow = most > static_cast<unsigned>(Region);
        if (!overflow || b < 0) break;

        // A region overflowed: raise the bound a bit at a time, counting over
        // the whole slice, until every wave's keys at or above it fit.
        while (most > static_cast<unsigned>(Region) && b >= 0) {
            const unsigned cand = t | (1u << b);
            --b;
            if (cand <= bound) {  // counts every staged key, which is >= k
                t = cand;
                continue;
            }
            int n = 0;
#pragma unroll
            for (int j = 0; j < Groups; ++j)
#pragma unroll
                for (int c = 0; c < 4; ++c) n += key[j][c] >= cand;
            const WaveSlots<Waves> slot = slot_of<Waves>(l, kCntSlot + parity);
            parity ^= 1;
            slot.put(wave_sum(n));
            __syncthreads();
            const int total = slot.sum();
            if (total >= k) {
                t = cand;
                bound = cand;
                staged = total;
                most = slot.max();
            } else {
                hi_n = total;
            }
        }
        if (most > static_cast<unsigned>(Region)) break;  // b < 0: t is the kth key
        __syncthreads();  // n_slot is read before it is rewritten
    }
    STAMP(3);
    NOTE(20, staged);

    if (overflow) {
        // Too many keys tie with the kth to stage: write the winners from the
        // registers, in index order.
        const WaveSlots<Waves> gt_slot = slot_of<Waves>(l, kGtSlot);
        const WaveSlots<Waves> eq_slot = slot_of<Waves>(l, kEqSlot);
        int gl = 0, el = 0;
#pragma unroll
        for (int j = 0; j < Groups; ++j)
#pragma unroll
            for (int c = 0; c < 4; ++c) gl += key[j][c] > t, el += key[j][c] == t;
        const int gt = wave_sum(gl), eq = wave_sum(el);
        gt_slot.put(gt);
        eq_slot.put(eq);
        STAMP(7);
        __syncthreads();
        STAMP(8);
        int gt_base = gt_slot.sum_before(), eq_base = eq_slot.sum_before();
        const int need = k - gt_slot.sum();
        const int gt_end = gt_base + gt;
        const unsigned at = opaque(t);
        const int here = pinned(base), there = pinned(tail);
#pragma unroll
        for (int j = 0; j < Groups; ++j) {
            if (gt_base == gt_end && eq_base >= need) break;
            unsigned long long mg[4], me[4];
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                mg[c] = __ballot(key[j][c] > at);
                me[c] = __ballot(key[j][c] == at);
            }
            if (!(mg[0] | mg[1] | mg[2] | mg[3] | me[0] | me[1] | me[2] | me[3])) continue;
            int gp = gt_base, ep = eq_base;
#pragma unroll
            for (int c = 0; c < 4; ++c) gp += rank_in_wave(mg[c]), ep += rank_in_wave(me[c]);
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                int pos = -1;
                if (key[j][c] > at) {
                    pos = gp + min(ep, need);
                    ++gp;
                } else if (key[j][c] == at) {
                    if (ep < need) pos = gp + ep;
                    ++ep;
                }
                if (pos >= 0) {
                    const int i = j < Vec ? here + j * kGroup + c : there;
                    if constexpr (Split) {
                        ent[pos] = make_uint2(key[j][c], static_cast<unsigned>(i));
                    } else {
                        out[pos] = i;
                        if constexpr (WriteValues) vout[pos] = rowp[i];
                    }
                }
            }
#pragma unroll
            for (int c = 0; c < 4; ++c) gt_base += __popcll(mg[c]), eq_base += __popcll(me[c]);
        }
        STAMP(9);
        if constexpr (Split) {
            if (threadIdx.x == 0) *head = {k, t, floor_part, 0};
        }
        return;
    }

    if constexpr (Split) {
        // Hand the merge the staged keys at or above the best floor the row
        // has published: every slice's floor_k is a floor for the row, and so
        // is the smallest floor_part once every slice has published one. The
        // merge recomputes both from the heads, at least as high as this.
        if (threadIdx.x == 0) {
            // Independent loads, so they are in flight together.
            const unsigned fk =
                __hip_atomic_load(&rs->floor_k, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            const unsigned np =
                __hip_atomic_load(&rs->not_part_min, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            const int published =
                __hip_atomic_load(&rs->published, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            *l.t = published == parts ? max(fk, ~np) : fk;
        }
        __syncthreads();
        const unsigned keep = opaque(max(*l.t, bound));
        unsigned k_e[R::Slots];
        int n = 0;
#pragma unroll
        for (int e = 0; e < R::Slots; ++e) {
            const int p = e * kWave + lane;
            k_e[e] = p < mine ? l.key[wave * Stride + p] : 0u;
            n += ballot_count(k_e[e] >= keep);
        }
        const WaveSlots<Waves> gt_slot = slot_of<Waves>(l, kGtSlot);
        gt_slot.put(n);
        __syncthreads();
        int at = gt_slot.sum_before();
        const int written = gt_slot.sum();
#pragma unroll
        for (int e = 0; e < R::Slots; ++e) {
            const unsigned long long m = __ballot(k_e[e] >= keep);
            if (k_e[e] >= keep)
                ent[at + rank_in_wave(m)] = make_uint2(
                    k_e[e], static_cast<unsigned>(l.idx[wave * Stride + e * kWave + lane]));
            at += __popcll(m);
        }
        if (threadIdx.x == 0) *head = {written, staged >= k ? bound : 0u, floor_part, 0};
        return;
    }

    search_and_emit<Waves>(l, t, b, bound, staged, hi_n, mine, parity, k,
                           [out, vout, rowp](int pos, int i) {
                               out[pos] = i;
                               if constexpr (WriteValues) vout[pos] = rowp[i];
                           });
}

// The merge of a split row, with wave w taking slice w. Slices are in index
// order and each lists its entries in index order, so wave order is still
// index order and the regions can be searched as an unsplit row's are.
//
// The entries hold every winner: a staged slice lists all its keys >= a floor
// at or below the kth key, and a slice with too many ties lists its own k
// winners, which hold every row winner it has. Each slice's floor_k is a floor
// for the row, and so is the smallest floor_part, since every slice has
// ceil(k / parts) keys at or above it.
template <int Threads, bool WriteValues>
__device__ __forceinline__ void merge_slices(const SliceHead* heads, const uint2* ent, int parts,
                                             int k, Lds l, const float* rowp, int* out,
                                             float* vout) {
    constexpr int Waves = Threads / kWave;
    static_assert(kMaxParts <= Waves && kMaxParts <= 16, "a wave per slice, heads in row 0");
    using R = Regions<Waves>;
    constexpr int Region = R::Region;
    constexpr int Pre = 4;  // chunks of 64 entries a wave loads before it knows its count
    const int lane = lane_id();
    const int wave = wave_id();
    const WaveSlots<Waves> hi_slot = slot_of<Waves>(l, kHiSlot);
    const WaveSlots<Waves> n_slot = slot_of<Waves>(l, kNSlot);
    const uint2* mine_ent = ent + static_cast<size_t>(wave) * kCap;
    const bool has_slice = wave < parts;

    // One round trip for everything the staging pass needs: lane p reads slice
    // p's head, and the wave starts on its own slice's entries. The acquire in
    // the caller invalidated the caches these go through.
    uint2 pre[Pre];
#pragma unroll
    for (int s = 0; s < Pre; ++s) pre[s] = has_slice ? mine_ent[s * kWave + lane] : make_uint2(0u, 0u);
    uint4 h = make_uint4(0u, 0u, ~0u, 0u);
    if (lane < parts) h = reinterpret_cast<const uint4*>(heads)[lane];
    const unsigned floor_k = __builtin_amdgcn_readlane(row_reduce(h.y, Max{}), 0);
    const unsigned floor_part = __builtin_amdgcn_readlane(row_reduce(h.z, Min{}), 0);
    const int count = has_slice ? static_cast<int>(__builtin_amdgcn_readlane(h.x, wave)) : 0;
    unsigned bound = max(max(floor_k, floor_part), 1u);

    // f(entry) for this wave's entries in order, past the prefetched ones;
    // positions past the count read as key 0. f returns false to stop.
    auto visit_from = [&](int from, auto&& f) {
        for (int c = from; c < count; c += Pre * kWave) {
            uint2 e[Pre];
#pragma unroll
            for (int s = 0; s < Pre; ++s) {
                const int u = c + s * kWave + lane;
                e[s] = u < count ? mine_ent[u] : make_uint2(0u, 0u);
            }
#pragma unroll
            for (int s = 0; s < Pre; ++s)
                if (!f(e[s])) return false;
        }
        return true;
    };
#pragma unroll
    for (int s = 0; s < Pre; ++s)
        if (s * kWave + lane >= count) pre[s] = make_uint2(0u, 0u);
    auto visit = [&](auto&& f) {
#pragma unroll
        for (int s = 0; s < Pre; ++s)
            if (!f(pre[s])) return;
        visit_from(Pre * kWave, f);
    };

    unsigned t = bound;
    int b = -1, hi_n = 0, parity = 0, staged = 0, mine = 0;
    bool overflow = false, first = true;
    for (;;) {
        const unsigned at_least = opaque(bound);
        const int region = wave * R::Stride;
        int pos = 0;
        unsigned mx = 0;
        visit([&](uint2 e) {
            const bool in = e.x >= at_least;
            const unsigned long long m = __ballot(in);
            if (m && pos <= Region && in) {
                const int p = region + pos + rank_in_wave(m);
                l.key[p] = e.x;
                l.idx[p] = static_cast<int>(e.y);
            }
            pos += __popcll(m);
            mx = max(mx, in ? e.x : 0u);
            return true;
        });
        mine = pos;
        n_slot.put(pos);
        if (first) hi_slot.put(static_cast<int>(wave_reduce(mx, Max{})));
        __syncthreads();
        staged = n_slot.sum();
        unsigned most = n_slot.max();
        if (first) {
            first = false;
            const unsigned hi = hi_slot.max();
            if (bound != hi) t = common_prefix(bound, hi, b);
        }
        overflow = most > static_cast<unsigned>(Region);
        if (!overflow || b < 0) break;

        while (most > static_cast<unsigned>(Region) && b >= 0) {
            const unsigned cand = t | (1u << b);
            --b;
            if (cand <= bound) {
                t = cand;
                continue;
            }
            int c = 0;
            visit([&](uint2 e) {
                c += e.x >= cand;
                return true;
            });
            const WaveSlots<Waves> slot = slot_of<Waves>(l, kCntSlot + parity);
            parity ^= 1;
            slot.put(wave_sum(c));
            __syncthreads();
            const int total = slot.sum();
            if (total >= k) {
                t = cand;
                bound = cand;
                staged = total;
                most = slot.max();
            } else {
                hi_n = total;
            }
        }
        if (most > static_cast<unsigned>(Region)) break;
        __syncthreads();
    }

    if (overflow) {
        // Too many entries tie with the kth to stage: write the winners straight
        // from the entries.
        const WaveSlots<Waves> gt_slot = slot_of<Waves>(l, kGtSlot);
        const WaveSlots<Waves> eq_slot = slot_of<Waves>(l, kEqSlot);
        const unsigned at = opaque(t);
        int gl = 0, el = 0;
        visit([&](uint2 e) {
            gl += e.x > at;
            el += e.x == at;
            return true;
        });
        const int gt = wave_sum(gl), eq = wave_sum(el);
        gt_slot.put(gt);
        eq_slot.put(eq);
        __syncthreads();
        int gt_base = gt_slot.sum_before(), eq_base = eq_slot.sum_before();
        const int need = k - gt_slot.sum();
        const int gt_end = gt_base + gt;
        if (gt_base == gt_end && eq_base >= need) return;
        const unsigned at2 = opaque(at);
        visit([&](uint2 e) {
            const unsigned long long mg = __ballot(e.x > at2), me = __ballot(e.x == at2);
            const int gp = gt_base + rank_in_wave(mg), ep = eq_base + rank_in_wave(me);
            int pos = -1;
            if (e.x > at2) pos = gp + min(ep, need);
            else if (e.x == at2 && ep < need) pos = gp + ep;
            if (pos >= 0) {
                out[pos] = static_cast<int>(e.y);
                if constexpr (WriteValues) vout[pos] = rowp[e.y];
            }
            gt_base += __popcll(mg);
            eq_base += __popcll(me);
            return !(gt_base == gt_end && eq_base >= need);
        });
        return;
    }

    search_and_emit<Waves>(l, t, b, bound, staged, hi_n, mine, parity, k,
                           [out, vout, rowp](int pos, int i) {
                               out[pos] = i;
                               if constexpr (WriteValues) vout[pos] = rowp[i];
                           });
}

__device__ __forceinline__ int row_length(const int* seq_lens, int row, int next_n, int stride) {
    const int len = seq_lens[row / next_n] - next_n + row % next_n + 1;
    return min(max(len, 0), stride);
}

// A row of at most k keys: all of them, then -1 (and -inf).
template <bool WriteValues>
__device__ __forceinline__ void short_row(int len, int k, const float* rowp, int* out, float* vout) {
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        out[i] = i < len ? i : -1;
        if constexpr (WriteValues) vout[i] = i < len ? rowp[i] : -INFINITY;
    }
}

#define TOPK_DECLARE_LDS(Waves)                                                              \
    __shared__ __attribute__((aligned(16))) unsigned s_key[(Waves) * Regions<Waves>::Stride]; \
    __shared__ __attribute__((aligned(16))) int s_idx[(Waves) * Regions<Waves>::Stride];      \
    __shared__ unsigned s_fin[kFinish];                                                       \
    __shared__ int s_slot[kSlots * 16];                                                       \
    __shared__ int s_fin_n;                                                                   \
    __shared__ unsigned s_t;                                                                  \
    if (threadIdx.x == 0) s_fin_n = 0;                                                        \
    const Lds lds{s_key, s_idx, s_fin, &s_fin_n, &s_t, s_slot}

// One workgroup per row.
template <int Threads, int Vec, bool WriteValues>
__global__ void __launch_bounds__(Threads, 1)
    topk_ballot_kernel(const float* __restrict__ logits, const int* __restrict__ seq_lens,
                       int* __restrict__ indices, float* __restrict__ values, int stride,
                       int next_n, int k) {
    constexpr int Waves = Threads / kWave;
    const int row = blockIdx.x;
    const int len = row_length(seq_lens, row, next_n, stride);
    const float* rowp = logits + static_cast<size_t>(row) * stride;
    int* out = indices + static_cast<size_t>(row) * k;
    float* vout = WriteValues ? values + static_cast<size_t>(row) * k : nullptr;
    if (len <= k) {
        short_row<WriteValues>(len, k, rowp, out, vout);
        return;
    }
    TOPK_DECLARE_LDS(Waves);
    select_slice<Threads, Vec, WriteValues, false>(rowp, 0, len, k, 1, lds, out, vout, nullptr,
                                                   nullptr, nullptr);
}

// `parts` workgroups per row, each on one slice; the last of a row to finish
// merges them, so no workgroup waits for another. states[row] is back to zero
// when the kernel ends.
template <int Threads, int Vec, bool WriteValues>
__global__ void __launch_bounds__(Threads, 1)
    topk_ballot_split_kernel(const float* __restrict__ logits, const int* __restrict__ seq_lens,
                             int* __restrict__ indices, float* __restrict__ values, int stride,
                             int next_n, int k, int parts, RowState* __restrict__ states,
                             SliceHead* __restrict__ heads, uint2* __restrict__ entries) {
    constexpr int Waves = Threads / kWave;
    const int row = blockIdx.x / parts;
    const int part = blockIdx.x % parts;
    const int len = row_length(seq_lens, row, next_n, stride);
    const float* rowp = logits + static_cast<size_t>(row) * stride;
    int* out = indices + static_cast<size_t>(row) * k;
    float* vout = WriteValues ? values + static_cast<size_t>(row) * k : nullptr;
    if (len <= k) {
        if (part == 0) short_row<WriteValues>(len, k, rowp, out, vout);
        return;
    }
    TOPK_DECLARE_LDS(Waves);
    __shared__ int s_last;

    // A multiple of 4 wide, so every slice starts float4-aligned.
    const int slice = ((len + parts - 1) / parts + 3) & ~3;
    const int s0 = min(part * slice, len);
    const int s1 = min(s0 + slice, len);
    SliceHead* row_heads = heads + static_cast<size_t>(row) * parts;
    uint2* row_ent = entries + static_cast<size_t>(row) * parts * kCap;
    RowState* rs = states + row;
    select_slice<Threads, Vec, WriteValues, true>(rowp, s0, s1, k, parts, lds, out, vout,
                                                  row_heads + part,
                                                  row_ent + static_cast<size_t>(part) * kCap, rs);

    // Every wave's stores reach L2, then one device-scope release makes them
    // visible to other XCDs before the slice counts as done. That release
    // writes the XCD's L2 back, so one per workgroup, not one per wave.
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
    __syncthreads();
    if (threadIdx.x == 0) {
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
        s_last = atomicAdd(&rs->done, 1) == parts - 1;
        if (s_last) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    }
    __syncthreads();
    if (!s_last) return;
    merge_slices<Threads, WriteValues>(row_heads, row_ent, parts, k, lds, rowp, out, vout);
    if (threadIdx.x == 0) *rs = {0, 0u, 0u, 0};
}

constexpr int kThreads = 512;
constexpr int kWavesPerGroup = kThreads / kWave;

// float4 loads per lane that cover `width` keys.
static int vec_for(int width) {
    const int chunk = ((width + kWavesPerGroup - 1) / kWavesPerGroup + 3) & ~3;
    return (chunk + kGroup - 1) / kGroup;
}

// Calls launch(integral_constant<int, Vec>) for the smallest instance, up to
// MaxVec, that holds `width` keys; false when none does.
template <int MaxVec, typename Launch>
static bool dispatch_vec(int width, Launch&& launch) {
    const int vec = vec_for(width);
    if (vec <= 4) launch(std::integral_constant<int, 4>{});
    else if (vec <= 8) launch(std::integral_constant<int, 8>{});
    else if (vec <= 16) launch(std::integral_constant<int, 16>{});
    else if (vec <= 25) launch(std::integral_constant<int, 25>{});
    else if (MaxVec >= 49 && vec <= 49) launch(std::integral_constant<int, MaxVec >= 49 ? 49 : 25>{});
    else return false;
    return true;
}

static int check_shape(const float* logits, int stride, int k) {
    if (k < 1 || (k + kWavesPerGroup - 1) / kWavesPerGroup > kSub * kWave || k > kCap) return 1;
    if (stride % 4 || reinterpret_cast<uintptr_t>(logits) % 16) return 2;
    return 0;
}

static size_t align256(size_t n) { return (n + 255) & ~static_cast<size_t>(255); }

// Return codes for both launchers: 0 on launch; 1 for k; 2 for a row pitch
// that is not a multiple of 4 floats; 3 for a row or slice too wide to hold in
// registers; 4 for parts or a missing workspace.
extern "C" int topk_ballot_launch(const float* logits, const int* seq_lens, int* indices,
                                  float* values, int rows, int stride, int next_n, int k,
                                  hipStream_t stream) {
    if (const int rc = check_shape(logits, stride, k)) return rc;
    const bool ok = dispatch_vec<49>(stride, [&](auto vec) {
        constexpr int Vec = decltype(vec)::value;
        if (values)
            topk_ballot_kernel<kThreads, Vec, true><<<rows, kThreads, 0, stream>>>(
                logits, seq_lens, indices, values, stride, next_n, k);
        else
            topk_ballot_kernel<kThreads, Vec, false><<<rows, kThreads, 0, stream>>>(
                logits, seq_lens, indices, values, stride, next_n, k);
    });
    if (!ok) return 3;
    return static_cast<int>(hipGetLastError());
}

// Workgroups per row for the split launcher; 1 means the unsplit launcher is
// the faster one. The merge costs about as much as an unsplit row of 50k keys,
// so only rows of kSplitMin keys or more split. A split workgroup takes a CU
// to itself (its LDS does), so rows * parts stays within the CUs: past them,
// the second round of workgroups waits for the first.
// Measured on MI308X, k = 512, at 1 to 16 rows.
extern "C" int topk_ballot_parts(int rows, int stride) {
    static const int cus = [] {
        int dev = 0, n = 0;
        (void)hipGetDevice(&dev);
        (void)hipDeviceGetAttribute(&n, hipDeviceAttributeMultiprocessorCount, dev);
        return n > 0 ? n : 1;
    }();
    if (stride < kSplitMin) return 1;
    int parts = kMaxParts;
    while (parts >= kMinParts && rows * parts > cus) parts /= 2;
    return parts >= kMinParts ? parts : 1;
}

// Bytes of workspace the split launcher needs for `rows` rows of `parts`
// slices. Zero it once before first use; each launch leaves it ready for the
// next. Launches that may run at the same time need separate workspaces.
extern "C" long long topk_ballot_workspace_bytes(int rows, int parts) {
    return static_cast<long long>(align256(sizeof(RowState) * rows) +
                                  align256(sizeof(SliceHead) * rows * parts) +
                                  sizeof(uint2) * static_cast<size_t>(rows) * parts * kCap);
}

// The same selection with each row split over `parts` workgroups. parts <= 1
// runs the unsplit kernel and needs no workspace.
extern "C" int topk_ballot_split_launch(const float* logits, const int* seq_lens, int* indices,
                                        float* values, int rows, int stride, int next_n, int k,
                                        int parts, void* workspace, hipStream_t stream) {
    if (parts <= 1)
        return topk_ballot_launch(logits, seq_lens, indices, values, rows, stride, next_n, k,
                                  stream);
    if (parts > kMaxParts || !workspace) return 4;
    if (const int rc = check_shape(logits, stride, k)) return rc;
    char* ws = static_cast<char*>(workspace);
    RowState* states = reinterpret_cast<RowState*>(ws);
    ws += align256(sizeof(RowState) * rows);
    SliceHead* heads = reinterpret_cast<SliceHead*>(ws);
    ws += align256(sizeof(SliceHead) * rows * parts);
    uint2* entries = reinterpret_cast<uint2*>(ws);
    const int slice = ((stride + parts - 1) / parts + 3) & ~3;
    // A slice of more than 25 loads per lane would spill; more parts avoid it.
    const bool ok = dispatch_vec<25>(slice, [&](auto vec) {
        constexpr int Vec = decltype(vec)::value;
        if (values)
            topk_ballot_split_kernel<kThreads, Vec, true><<<rows * parts, kThreads, 0, stream>>>(
                logits, seq_lens, indices, values, stride, next_n, k, parts, states, heads,
                entries);
        else
            topk_ballot_split_kernel<kThreads, Vec, false><<<rows * parts, kThreads, 0, stream>>>(
                logits, seq_lens, indices, values, stride, next_n, k, parts, states, heads,
                entries);
    });
    if (!ok) return 3;
    return static_cast<int>(hipGetLastError());
}
