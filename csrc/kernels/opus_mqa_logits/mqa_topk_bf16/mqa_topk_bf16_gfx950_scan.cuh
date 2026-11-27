// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx950 scoring: a wave scores 4 query rows against 16-key tiles, v_mfma_f32_16x16x32_bf16, the
// 4 rows x 4 heads of a head block the MFMA's 16 A rows. Device pass on gfx950 only.
//
// Lane L = 16 g + i (i = L % 16, g = L / 16):
//   A (head block hb, k-step ks)  Q[row i / 4][head 4 hb + i % 4][32 ks + 8 g .. + 7]
//   B (k-step ks)                 K[key i][32 ks + 8 g .. + 7]
//   C (head block hb)             rows 4 g + 0..3 = row g, heads 4 hb + 0..3; key i
// so over the 8 head blocks a lane holds all 32 heads of row g for key i, and its score is one
// chain of fmas in head order, the same in every stage: relu(c) * w, then fma for heads 1..31.
// A row's 32 weights sit in two registers, lane 16 g + i holding heads i and 16 + i of row g,
// read by the fma's DPP row_newbcast.
//
// Two walks over the tiles:
//   walk_lds  a workgroup's waves share the keys: 64-key stages copied into an LDS ring by
//             buffer_load_lds (gathered, for the sample stage's keys 32 apart), one barrier a
//             stage. A tile's MFMAs run a pair of head blocks at a time around the heads of the
//             pair before, so the VALU never waits on the MFMA it reads. logits, sample, emit.
//   walk      a wave loads its own tiles straight into a ring of registers; no LDS, no barrier.
//             The fallback, whose waves split one row group's tiles.
#pragma once
#include <opus/opus.hpp>

#include "mqa_topk_bf16_common.cuh"

namespace opus_mqa_topk::gfx950 {

using bf16_t = opus::bf16_t;
using bf16x8 = opus::vector_t<bf16_t, 8>;
using f32x4  = opus::vector_t<float, 4>;
using i32x2  = opus::vector_t<int, 2>;
using Mma    = opus::mfma<bf16_t, bf16_t, float, 16, 16, 32>;

constexpr int WAVE = 64;
constexpr int R    = 4;    // rows a wave
constexpr int HB   = 8;    // head blocks of 4 heads
constexpr int KS   = 4;    // k-steps of 32
constexpr int TILE = 16;   // keys a tile
constexpr int BUF  = 3;    // walk: K tiles in registers, the one scored and two in flight
static_assert(HB * 4 == HEADS && KS * 32 == HEAD_DIM, "the tile covers 32 heads x 128 dims");

constexpr int KEY_BYTES   = HEAD_DIM * 2;             // 256
constexpr int STAGE_KEYS  = 64;                       // walk_lds: keys a stage
constexpr int STAGE_TILES = STAGE_KEYS / TILE;        // 4
constexpr int STAGE_BYTES = STAGE_KEYS * KEY_BYTES;   // 16 KB
constexpr int DMA_BYTES   = WAVE * 16;                // one buffer_load_lds of 16 B a lane
constexpr unsigned OOB    = 0x80000000u;              // a buffer byte offset past any range

// acc += dpp(w, lane Q of the 16-lane row) * v. asm: there is no builtin for a DPP fma (the DPP
// combine leaves a v_mov_dpp a head), and the fixed form keeps the chain's order. Its DPP
// operand is written once, before any walk, so it needs no VALU -> DPP wait states.
template<int Q>
__device__ inline void fmac_bcast(float& acc, float w, float v) {
    asm("v_fmac_f32_dpp %0, %1, %2 row_newbcast:%3 row_mask:0xf bank_mask:0xf"
        : "+v"(acc)
        : "v"(w), "v"(v), "i"(Q));
}
__device__ inline float mul_bcast0(float w, float v) {
    float r;
    asm("v_mul_f32_dpp %0, %1, %2 row_newbcast:0 row_mask:0xf bank_mask:0xf"
        : "=v"(r)
        : "v"(w), "v"(v));
    return r;
}

// Lane 16 g + Q's value to every lane of row g.
template<int Q>
__device__ inline int row_bcast(int v) {
    return __builtin_amdgcn_update_dpp(0, v, 0x150 + Q, 0xF, 0xF, false);
}

__device__ inline void fence() { __builtin_amdgcn_sched_barrier(0); }

// Four ints, read by compile-time index. A small array a lambda indexes stays an alloca, which
// costs a scratch store a use and, worse, a vmem op the walk's vmcnt keeps do not count.
struct int4_regs {
    int v0, v1, v2, v3;
    template<int I>
    __device__ int at() const {
        if constexpr (I == 0) return v0;
        else if constexpr (I == 1) return v1;
        else if constexpr (I == 2) return v2;
        else return v3;
    }
};

// -- a wave's 4 rows ------------------------------------------------------------------------------
struct wave_rows {
    bf16x8 q[HB][KS];
    float  w_lo, w_hi;   // row g's weights of heads i and 16 + i
    float  floor;        // 0 (relu) or -inf
    int    lo, hi;       // row g's window
    int    ulo, uhi;     // the union of the rows' windows (ulo >= uhi: empty)

    // rows[s] (s < 4) are the wave's rows: row0 + s, or (LISTED) any rows. A row past num_rows
    // scores zeros in an empty window.
    template<bool LISTED = false>
    __device__ void load(const mqa_topk_bf16_kargs& k, const int (&rows)[R], int lane) {
        const int i = lane % 16, g = lane / 16;
        const int a_off = (i % 4) * HEAD_DIM + 8 * g;
        if constexpr (LISTED) {
            const bf16x8* p = reinterpret_cast<const bf16x8*>(
                reinterpret_cast<const bf16_t*>(k.ptr_q) +
                (size_t)rows[i / 4] * (HEADS * HEAD_DIM) + a_off);
            opus::static_for<HB>([&](auto hbc) {
                opus::static_for<KS>([&](auto ksc) {
                    constexpr int hb = hbc.value, ks = ksc.value;
                    q[hb][ks] = p[(hb * 4 * HEAD_DIM + ks * 32) / 8];
                });
            });
        } else {
            const int row0  = rows[0];
            const int left  = k.num_rows - row0;
            // readfirstlane: the clamp lowers to v_med3, a VGPR, and a V# from it is a waterfall
            const int valid = __builtin_amdgcn_readfirstlane(left < 0 ? 0 : left < R ? left : R);
            auto g_q = opus::make_gmem(reinterpret_cast<const bf16_t*>(k.ptr_q) +
                                           (size_t)(valid ? row0 : 0) * (HEADS * HEAD_DIM),
                                       (unsigned)valid * (HEADS * HEAD_DIM * 2u));
            const int off = (i / 4) * (HEADS * HEAD_DIM) + a_off;
            opus::static_for<HB>([&](auto hbc) {
                opus::static_for<KS>([&](auto ksc) {
                    constexpr int hb = hbc.value, ks = ksc.value;
                    q[hb][ks] = g_q.template load<8>(off + hb * 4 * HEAD_DIM + ks * 32);
                });
            });
        }
        // row0 + g, not rows[g]: a lane-indexed array is memory, and then even rows[0] (the Q
        // V#'s base) reads as divergent.
        const int rc  = LISTED ? rows[g] : rows[0] + g;
        const bool vc = rc < k.num_rows;
        w_lo  = vc ? k.ptr_weights[(size_t)rc * HEADS + i] : 0.f;
        w_hi  = vc ? k.ptr_weights[(size_t)rc * HEADS + 16 + i] : 0.f;
        floor = (k.flags & MQA_TOPK_RELU) ? 0.0f : -__builtin_inff();
        const window wd = vc ? clamp_window(k.ptr_row_starts[rc], k.ptr_row_ends[rc], k.num_keys)
                             : window{0, 0};
        lo  = wd.lo;
        hi  = wd.hi;
        ulo = k.num_keys;
        uhi = 0;
        opus::static_for<R>([&](auto sc) {
            const int l = __builtin_amdgcn_readlane(lo, sc.value * 16);
            const int h = __builtin_amdgcn_readlane(hi, sc.value * 16);
            if (l < h) {
                ulo = l < ulo ? l : ulo;
                uhi = h > uhi ? h : uhi;
            }
        });
    }

    // MFMA J (= ks * 8 + hb) of a tile into acc.
    template<int J>
    __device__ void mfma(f32x4 (&acc)[HB], const bf16x8 (&b)[KS]) const {
        constexpr int ks = J / HB, hb = J % HB;
        Mma mma;
        acc[hb] = mma(q[hb][ks], b[ks], ks == 0 ? f32x4{0.f, 0.f, 0.f, 0.f} : acc[hb]);
    }
    // Head H of the chain over acc.
    template<int H>
    __device__ void head(float& s, const f32x4 (&acc)[HB]) const {
        const float v = __builtin_fmaxf(acc[H / 4][H % 4], floor);
        if constexpr (H == 0)
            s = mul_bcast0(w_lo, v);
        else
            fmac_bcast<H % 16>(s, H < 16 ? w_lo : w_hi, v);
    }

    // Row g's score of key i of the tile in b.
    __device__ float score(const bf16x8 (&b)[KS]) const {
        f32x4 acc[HB];
        opus::static_for<HB * KS>([&](auto jc) { mfma<jc.value>(acc, b); });
        float s = 0.f;
        opus::static_for<HEADS>([&](auto hc) { head<hc.value>(s, acc); });
        return s;
    }

    // Tile u's MFMAs into acc, a pair of head blocks at a time, the pair's two chains interleaved
    // (no MFMA waits on the one before it), around the heads of the pair before: blocks 6, 7 of
    // tile u - 1 (completing its score s_prev) under pair 0, then blocks 0 .. 5 of tile u (into
    // s_next) under pairs 1 .. 3. A head reads a chain at least two MFMAs old; blocks 6, 7 are read
    // before tile u + 1 overwrites them, so one accumulator set is enough. FIRST: no tile before.
    template<bool FIRST = false>
    __device__ float step(f32x4 (&acc)[HB], const bf16x8 (&b)[KS], float s_prev,
                          float& s_next) const {
        float s = 0.f;
        opus::static_for<HB / 2>([&](auto pc) {
            constexpr int p = pc.value;
            // heads [j0, j1) of the eight before this pair
            auto heads = [&](auto j0, auto j1) {
                opus::static_for<decltype(j1)::value - decltype(j0)::value>([&](auto jc) {
                    constexpr int j = decltype(j0)::value + jc.value;
                    if constexpr (p == 0) {
                        if constexpr (!FIRST) head<8 * (HB / 2 - 1) + j>(s_prev, acc);
                    } else {
                        head<8 * (p - 1) + j>(s, acc);
                    }
                });
            };
            // MFMA m of the pair: k-step m / 2 of block 2 p + m % 2
            auto mf = [&](auto mc) {
                constexpr int m = decltype(mc)::value;
                fence();
                mfma<(m / 2) * HB + 2 * p + m % 2>(acc, b);
            };
            mf(opus::number<0>{});
            mf(opus::number<1>{});
            fence();
            heads(opus::number<0>{}, opus::number<2>{});
            opus::static_for<6>([&](auto mc) {
                mf(opus::number<2 + mc.value>{});
                fence();
                heads(opus::number<2 + mc.value>{}, opus::number<3 + mc.value>{});
            });
        });
        fence();
        s_next = s;
        return s_prev;
    }
};

// -- walk: a wave's own tiles through registers (the fallback) -----------------------------------
// Tiles ti = 0 .. nt - 1, tile ti's 16 key rows at element offsets key_off(ti) (per lane: row i),
// the ring BUF deep. sink(ti, u, s) takes each score, group_end(u_count) closes each group of BUF
// tiles (u_count of them real). Unrolled by BUF so every ring index is a compile-time constant.
template<class KeyOff, class Sink, class GroupEnd>
__device__ inline void walk(const wave_rows& w, opus::gmem<bf16_t> g_k, int nt, int lane,
                            KeyOff key_off, Sink sink, GroupEnd group_end) {
    if (nt <= 0) return;
    const int dim_off = 8 * (lane / 16);
    bf16x8 b[BUF][KS];
    auto issue = [&](auto uc, int ti) {
        const int t   = ti < nt ? ti : nt - 1;   // past the walk: its last tile again
        const int off = key_off(t) + dim_off;
        opus::static_for<KS>([&](auto ksc) {
            constexpr int ks = ksc.value;
            b[decltype(uc)::value][ks] = g_k.template load<8>(off + ks * 32);
        });
    };
    opus::static_for<BUF - 1>([&](auto uc) { issue(uc, uc.value); });
    // Whole groups with no branch inside, so the loads in flight are the same on every path and
    // each tile waits for its own loads only.
    int t0 = 0;
    for (; t0 + BUF <= nt; t0 += BUF) {
        opus::static_for<BUF>([&](auto uc) {
            constexpr int u = uc.value;
            issue(opus::number<(u + BUF - 1) % BUF>{}, t0 + u + BUF - 1);
            sink(t0 + u, u, w.score(b[u]));
        });
        group_end(BUF);
    }
    // The last BUF - 1 tiles at most, already in flight.
    if (t0 < nt) {
        opus::static_for<BUF - 1>([&](auto uc) {
            constexpr int u = uc.value;
            if (t0 + u < nt) sink(t0 + u, u, w.score(b[u]));
        });
        group_end(nt - t0);
    }
}

// -- walk_lds: a workgroup's shared stages --------------------------------------------------------
// The workgroup walks tiles 0 .. nt - 1 (keys k_base + 16 t ..) through a ring of STAGES 64-key
// stages in LDS. A stage's image swizzles each key's sixteen 16 B chunks by the key's low 4 bits,
// so a tile's ds_read_b128 hits distinct 16 B bank groups in each 8-lane phase; buffer_load_lds
// writes 16 B a lane in lane order, so the swizzle is on the source side. Each wave copies
// 16 / WAVES KB of a stage.
//
// Tile t runs as: read B(t + 2) | MFMAs of t + 1 around the heads of t | sink(t). The first read
// of stage c + 1 (at tile 4 c + 2) waits for its copies (vmcnt: a wave its own, in issue order)
// and a barrier; the barrier also frees stage c's slot (its last read, B(4 c + 3), was a tile
// earlier) for stage c + STAGES. The loop is unrolled over the ring, so every LDS address is a
// lane's offset plus an immediate.
//
// The sinks keep the vmem queue static: sink issues SINK_OPS vmem ops a tile whatever its data
// (dropped ones go out of range, not around a branch), stage_end STAGE_OPS a stage. The walk runs
// whole stages: tiles past nt score stale keys, which the sinks drop by key.
// GATHER: the walk's key x is row key_of(x) of g_k (the sample stage), else row x.
struct contiguous_keys {
    __device__ int operator()(int x) const { return x; }
};
// The ring: one __shared__ array a slot. A ds_read and a copy into a different slot then point at
// different objects, which is what lets the compiler see they do not alias; through one array (or
// an address made from an integer) it waits for every copy in flight before each read.
template<int STAGES>
struct lds_ring;
template<>
struct lds_ring<4> {
    char* s0;
    char* s1;
    char* s2;
    char* s3;
    template<int SLOT>
    __device__ char* at() const {
        if constexpr (SLOT == 0) return s0;
        else if constexpr (SLOT == 1) return s1;
        else if constexpr (SLOT == 2) return s2;
        else return s3;
    }
};
#define MQA_TOPK_LDS_RING(name)                                                                  \
    __shared__ __attribute__((aligned(1024))) char name##0[STAGE_BYTES];                         \
    __shared__ __attribute__((aligned(1024))) char name##1[STAGE_BYTES];                         \
    __shared__ __attribute__((aligned(1024))) char name##2[STAGE_BYTES];                         \
    __shared__ __attribute__((aligned(1024))) char name##3[STAGE_BYTES];                         \
    const lds_ring<4> name{name##0, name##1, name##2, name##3}

template<int WAVES, int STAGES, int SINK_OPS, int STAGE_OPS, bool GATHER = false,
         class KeyOf = contiguous_keys, class Sink, class StageEnd>
__device__ inline void walk_lds(const wave_rows& w, const lds_ring<STAGES>& ring,
                                opus::gmem<uint8_t> g_k, int nt, int lane, int wave, Sink sink,
                                StageEnd stage_end, KeyOf key_of = {}) {
    static_assert(STAGE_BYTES % (WAVES * DMA_BYTES) == 0, "the waves split a stage's copy");
    static_assert(STAGES >= 3 && (STAGES & (STAGES - 1)) == 0, "a power-of-two ring of 3+");
    constexpr int DPW       = STAGE_BYTES / (WAVES * DMA_BYTES);   // copies a wave a stage
    constexpr int PER_STAGE = STAGE_TILES * SINK_OPS + STAGE_OPS;
    const int nst = (nt + STAGE_TILES - 1) / STAGE_TILES;
    const int i = lane % 16, g = lane / 16;

    // The copy of stage c (past the walk: its last stage again) into slot SLOT = c % STAGES.
    static_assert(DPW <= 4, "a wave's copies of a stage, at most four");
    auto src_of = [&](int j) {
        const int d     = wave * DPW + j;                           // the stage's d-th KB
        const int key   = d * (DMA_BYTES / KEY_BYTES) + lane / 16;  // 4 keys a KB
        const int chunk = (lane % 16) ^ (key % 16);
        return key * KEY_BYTES + chunk * 16;
    };
    const int4_regs dma_src{src_of(0), DPW > 1 ? src_of(1) : 0, DPW > 2 ? src_of(2) : 0,
                            DPW > 3 ? src_of(3) : 0};
    auto issue = [&](int c, auto slotc) {
        const int cs = c < nst ? c : nst - 1;
        char* slot   = ring.template at<decltype(slotc)::value>();
        opus::static_for<DPW>([&](auto jc) {
            constexpr int j = jc.value;
            if constexpr (GATHER) {
                // the lane's key, gathered: its row's chunk at the swizzled place
                const int x     = cs * STAGE_KEYS + (wave * DPW + j) * (DMA_BYTES / KEY_BYTES) +
                              lane / 16;
                const int chunk = (lane % 16) ^ (x % 16);
                g_k.template async_load<16>(slot + (wave * DPW + j) * DMA_BYTES,
                                            key_of(x) * KEY_BYTES + chunk * 16, 0);
            } else {
                g_k.template async_load<16>(slot + (wave * DPW + j) * DMA_BYTES,
                                            dma_src.template at<j>(), cs * STAGE_BYTES);
            }
        });
    };
    // B of tile TT of the stage in slot SLOT: k-step ks's 16 B of key i, chunk 4 ks + g swizzled
    // by i.
    static_assert(KS == 4, "four k-steps, four lane offsets");
    auto b_of = [&](int ks) { return i * KEY_BYTES + (((4 * ks + g) ^ i) * 16); };
    const int4_regs b_off{b_of(0), b_of(1), b_of(2), b_of(3)};
    auto read_b = [&](bf16x8 (&b)[KS], auto slotc, auto ttc) {
        const char* slot = ring.template at<decltype(slotc)::value>() +
                           decltype(ttc)::value * (TILE * KEY_BYTES);
        opus::static_for<KS>([&](auto ksc) {
            b[ksc.value] =
                *reinterpret_cast<const bf16x8*>(slot + b_off.template at<ksc.value>());
        });
    };
    auto barrier = [&]() {
        fence();
        __builtin_amdgcn_s_barrier();
        fence();
    };

    bf16x8 b[2][KS];
    f32x4 acc[HB];
    float part;   // the score of the tile whose MFMAs are in acc, up to its last head block
    // The prologue: every slot's copy in flight, stage 0 landed, B(0), B(1), tile 0's MFMAs.
    opus::static_for<STAGES>([&](auto cc) { issue(cc.value, cc); });
    opus::s_waitcnt_vmcnt(opus::number<(STAGES - 1) * DPW>{});
    barrier();
    read_b(b[0], opus::number<0>{}, opus::number<0>{});
    read_b(b[1], opus::number<0>{}, opus::number<1>{});
    w.template step<true>(acc, b[0], 0.f, part);

    for (int c0 = 0; c0 < nst; c0 += STAGES) {
        opus::static_for<STAGES>([&](auto sc) {
            constexpr int slot = sc.value;
            const int c        = c0 + slot;
            if (c >= nst) return;   // workgroup-uniform; then so is every later stage
            opus::static_for<STAGE_TILES>([&](auto ttc) {
                constexpr int tt = ttc.value, cur = tt % 2, nxt = 1 - cur;
                const int t = c * STAGE_TILES + tt;
                fence();
                if constexpr (tt == STAGE_TILES - 2) {
                    // Stage c + 1 landed for every wave; slot free for stage c + STAGES. Behind
                    // stage c + 1's copy: the prologue's later copies and c stages of sinks, or
                    // (once the loop issues them) STAGES - 2 copies and STAGES - 1 stages.
                    opus::s_waitcnt_lgkmcnt(opus::number<0>{});
                    if (slot < STAGES - 1 && c0 == 0)
                        opus::s_waitcnt_vmcnt(opus::number<(STAGES - 2) * DPW + slot * PER_STAGE +
                                                           (STAGE_TILES - 2) * SINK_OPS>{});
                    else
                        opus::s_waitcnt_vmcnt(
                            opus::number<(STAGES - 2) * DPW + (STAGES - 1) * PER_STAGE>{});
                    barrier();
                    issue(c + STAGES, sc);
                }
                // B(t + 2) into the registers B(t) left
                read_b(b[cur], opus::number<(slot + (tt + 2) / STAGE_TILES) % STAGES>{},
                       opus::number<(tt + 2) % STAGE_TILES>{});
                const float s = w.step(acc, b[nxt], part, part);
                sink(t, ttc, s);
                if constexpr (tt == STAGE_TILES - 1) stage_end(c, sc);
            });
        });
    }
    // Copies past the walk are still in flight: s_endpgm waits for them before the LDS goes.
}

// -- the item kernels -----------------------------------------------------------------------------
// A workgroup is WAVES waves of R rows; item = chunk * row_blocks + row block (chunk-major, so the
// workgroups resident at once share a chunk of K in L2). LOGITS chunks are chunk_keys keys,
// SAMPLE chunks sample_chunk samples (their keys gathered); the workgroup walks the union of its
// rows' windows within the chunk through walk_lds.
template<class T, int STAGE>
__global__ __launch_bounds__(T::BLOCK, 2) void scan_kernel(mqa_topk_bf16_kargs k) {
    constexpr int ROWS_WG = T::WAVES * R;
    const int lane  = lane_id();
    const int wave  = __builtin_amdgcn_readfirstlane(opus::thread_id_x() / WAVE);
    const int keys  = k.num_keys;
    const int nrb   = (k.num_rows + ROWS_WG - 1) / ROWS_WG;
    const int chunk = opus::block_id_x() / nrb;
    const int rb0   = (opus::block_id_x() - chunk * nrb) * ROWS_WG;
    const int row0  = rb0 + wave * R;
    const int i = lane % 16, g = lane / 16;
    const int rows[R] = {row0, row0 + 1, row0 + 2, row0 + 3};
    const int left       = k.num_rows - row0;
    const int valid_rows = __builtin_amdgcn_readfirstlane(left < 0 ? 0 : left < R ? left : R);

    {
        // Each wave loads its own rows; the workgroup walks their union.
        static_assert(T::STAGES == 4, "the ring is four slots");
        MQA_TOPK_LDS_RING(lds);
        __shared__ int s_union[T::WAVES][2];
        if (rb0 >= k.num_rows) return;   // workgroup-uniform
        // The union of the workgroup's windows first: an item outside it loads no Q.
        {
            const int r = rb0 + (int)opus::thread_id_x();
            window wd{keys, 0};
            if (opus::thread_id_x() < ROWS_WG && r < k.num_rows) {
                wd = clamp_window(k.ptr_row_starts[r], k.ptr_row_ends[r], keys);
                if (wd.lo >= wd.hi) wd = window{keys, 0};
            }
            const unsigned lo = wave_min((unsigned)wd.lo), hi = wave_max((unsigned)wd.hi);
            if (lane == 0) {
                s_union[wave][0] = (int)lo;
                s_union[wave][1] = (int)hi;
            }
        }
        __syncthreads();
        int ulo = keys, uhi = 0;
        opus::static_for<T::WAVES>([&](auto wc) {
            const int l = s_union[wc.value][0], h = s_union[wc.value][1];
            if (l < h) {
                ulo = l < ulo ? l : ulo;
                uhi = h > uhi ? h : uhi;
            }
        });
        // Uniform, but read from LDS: said so, or every V# built from it is a waterfall loop.
        ulo = __builtin_amdgcn_readfirstlane(ulo);
        uhi = __builtin_amdgcn_readfirstlane(uhi);
        if constexpr (STAGE == STAGE_SAMPLE) {
            // Samples [s0, s1) of the chunk, those of the rows' windows: the keys gathered.
            const int ns = num_samples(keys);
            const int s0 = chunk * k.sample_chunk;
            const int s1 = s0 + k.sample_chunk < ns ? s0 + k.sample_chunk : ns;
            if (ulo >= uhi) return;   // workgroup-uniform
            const int jl = sample_at_or_after(ulo, keys), jh = sample_at_or_after(uhi, keys);
            const int jb = (jl > s0 ? jl : s0) & ~(TILE - 1);
            const int je = jh < s1 ? jh : s1;
            if (je <= jb) return;
            const int nt = (je - jb + TILE - 1) / TILE;
            wave_rows w;
            w.load(k, rows, lane);
            const mqa_topk_ws ws = ws_layout(k.num_rows, keys, k.cap_row, k.fb_slots);
            auto g_s = opus::make_gmem(
                reinterpret_cast<float*>(static_cast<char*>(k.ptr_workspace) + ws.samples) +
                    (size_t)(valid_rows ? row0 : 0) * ns,
                (unsigned)(valid_rows * ns) * 4u);
            auto g_k = opus::make_gmem(reinterpret_cast<const uint8_t*>(k.ptr_k),
                                       (unsigned)keys * KEY_BYTES);
            walk_lds<T::WAVES, T::STAGES, 1, 0, true>(
                w, lds, g_k, nt, lane, wave,
                [&](int t, auto, float s) {
                    const int j = jb + t * TILE + i;
                    g_s.template store<1>(s, j < je ? g * ns + j : (int)(OOB / 4));
                },
                [](int, auto) {},
                [&](int x) {
                    const int j = jb + x;
                    return sample_key(j < ns ? j : ns - 1, keys);
                });
            return;
        }
        const int c0 = chunk * k.chunk_keys;
        const int c1 = c0 + k.chunk_keys < keys ? c0 + k.chunk_keys : keys;
        const int kb = ulo < uhi ? ((ulo > c0 ? ulo : c0) & ~(TILE - 1)) : c1;
        const int ke = ulo < uhi ? (uhi < c1 ? uhi : c1) : c1;
        const int nt = ke > kb ? (ke - kb + TILE - 1) / TILE : 0;
        wave_rows w;
        w.load(k, rows, lane);
        auto g_k = opus::make_gmem(reinterpret_cast<const uint8_t*>(k.ptr_k) +
                                       (size_t)kb * KEY_BYTES,
                                   (unsigned)(keys - kb) * KEY_BYTES);

        if constexpr (STAGE == STAGE_LOGITS) {
            // Every key of [c0, c1) of the wave's rows: -inf outside the walk and the windows.
            auto g_out = opus::make_gmem(k.ptr_values + (size_t)(valid_rows ? row0 : 0) * keys,
                                         (unsigned)(valid_rows * keys) * 4u);
            const int row_off = g * keys;
            constexpr float NEG = -__builtin_inff();
            // kb may lie past c1
            const int pe = kb < c1 ? kb : c1;
            for (int t = c0; t < pe; t += TILE)
                g_out.template store<1>(NEG, t + i < pe ? row_off + t + i : (int)(OOB / 4));
            for (int t = kb + nt * TILE; t < c1; t += TILE)
                g_out.template store<1>(NEG, t + i < c1 ? row_off + t + i : (int)(OOB / 4));
            if (nt == 0) return;   // workgroup-uniform
            walk_lds<T::WAVES, T::STAGES, 1, 0>(
                w, lds, g_k, nt, lane, wave,
                [&](int t, auto, float s) {
                    const int key = kb + t * TILE + i;
                    const float v = key >= w.lo && key < w.hi ? s : NEG;
                    g_out.template store<1>(v, key < c1 ? row_off + key : (int)(OOB / 4));
                },
                [](int, auto) {});
        }
    }
}

// -- emit: an item a workgroup, like logits -------------------------------------------------------
// Item = chunk * row_blocks + row block (chunk-major: the workgroups resident at once share a
// chunk of K in L2). An item outside its rows' windows loads no Q.
template<class T>
__global__ __launch_bounds__(T::BLOCK, 2) void emit_kernel(mqa_topk_bf16_kargs k) {
    constexpr int ROWS_WG = T::WAVES * R;
    static_assert(T::STAGES == 4, "the ring is four slots");
    MQA_TOPK_LDS_RING(lds);
    __shared__ int s_union[T::WAVES][2];
    const int lane = lane_id();
    const int wave = __builtin_amdgcn_readfirstlane(opus::thread_id_x() / WAVE);
    const int keys = k.num_keys;
    const int i = lane % 16, g = lane / 16;
    const mqa_topk_ws ws = ws_layout(k.num_rows, keys, k.cap_row, k.fb_slots);
    char* base = static_cast<char*>(k.ptr_workspace);
    const int nrb   = (k.num_rows + ROWS_WG - 1) / ROWS_WG;
    const int chunk = opus::block_id_x() / nrb;
    const int rb0   = (opus::block_id_x() - chunk * nrb) * ROWS_WG;
    {
        const int r = rb0 + (int)opus::thread_id_x();
        window wd{keys, 0};
        if (opus::thread_id_x() < ROWS_WG && r < k.num_rows) {
            wd = clamp_window(k.ptr_row_starts[r], k.ptr_row_ends[r], keys);
            if (wd.lo >= wd.hi) wd = window{keys, 0};
        }
        const unsigned lo = wave_min((unsigned)wd.lo), hi = wave_max((unsigned)wd.hi);
        if (lane == 0) {
            s_union[wave][0] = (int)lo;
            s_union[wave][1] = (int)hi;
        }
    }
    __syncthreads();
    int ulo = keys, uhi = 0;
    opus::static_for<T::WAVES>([&](auto wc) {
        const int l = s_union[wc.value][0], h = s_union[wc.value][1];
        if (l < h) {
            ulo = l < ulo ? l : ulo;
            uhi = h > uhi ? h : uhi;
        }
    });
    // Uniform, but read from LDS: said so, or every V# built from it is a waterfall loop.
    ulo = __builtin_amdgcn_readfirstlane(ulo);
    uhi = __builtin_amdgcn_readfirstlane(uhi);
    const int c0 = chunk * k.chunk_keys;
    const int c1 = c0 + k.chunk_keys < keys ? c0 + k.chunk_keys : keys;
    const int kb0 = (ulo > c0 ? ulo : c0) & ~(TILE - 1);
    const int ke0 = uhi < c1 ? uhi : c1;
    if (ulo >= uhi || ke0 <= kb0) return;   // workgroup-uniform

    auto segment = [&](int row0, int kb, int ke) {
        const int rows[R]    = {row0, row0 + 1, row0 + 2, row0 + 3};
        const int left       = k.num_rows - row0;
        const int valid_rows = __builtin_amdgcn_readfirstlane(left < 0 ? 0 : left < R ? left : R);
        wave_rows w;
        w.load(k, rows, lane);
        const int nt = (ke - kb + TILE - 1) / TILE;
        auto g_k = opus::make_gmem(reinterpret_cast<const uint8_t*>(k.ptr_k) +
                                       (size_t)kb * KEY_BYTES,
                                   (unsigned)(keys - kb) * KEY_BYTES);
            const int rc = row0 + g;
            const float tau = rc < k.num_rows ? reinterpret_cast<const float*>(base + ws.tau)[rc]
                                              : __builtin_inff();
            const int cap = k.cap_row;
            const int r0  = valid_rows ? row0 : 0;
            // The V#s are made where they are used: one captured by a lambda stays an alloca, which
            // the backend promotes to LDS, and a V# loaded from LDS is a waterfall loop.
            int* const cnt_base  = reinterpret_cast<int*>(base + ws.counts) + r0;
            int* const pool_base = reinterpret_cast<int*>(base + ws.cands) + (size_t)r0 * cap * 2;
            // A stage's candidates are counted and reserved with one atomic a row, and stored two
            // stages later, when the atomic is long back (pending sets by the stage's parity). A
            // tile's candidates are a lane mask (bool: an SGPR pair); a key is in the row's window
            // (clipped to the walk) iff (unsigned)(key - lo) < n.
            const int whi   = w.hi < ke ? w.hi : ke;
            const unsigned wn = whi > w.lo ? (unsigned)(whi - w.lo) : 0u;
            const int d0    = kb + i - w.lo;
            // A pending stage: its scores, candidates and reservation. Two, picked at compile time:
            // indexed arrays here stay allocas (a dead store a stage, and a vmem op the walk's
            // vmcnt keeps do not count).
            struct pending {
                float s[STAGE_TILES];
                bool hit[STAGE_TILES];
                int excl, base, key0;
            };
            pending pend0{}, pend1{};
            pend0.key0 = pend1.key0 = kb;
            auto pick = [&](auto pc) -> pending& {
                if constexpr (decltype(pc)::value == 0) return pend0;
                else return pend1;
            };
            float sc[STAGE_TILES];
            bool hit[STAGE_TILES];
            auto flush = [&](auto pc) {
                pending& pd = pick(pc);
                auto g_pool = opus::make_gmem(pool_base, (unsigned)valid_rows * cap * 8u);
                int slot = row_bcast<0>(pd.base) + pd.excl;
                opus::static_for<STAGE_TILES>([&](auto uc) {
                    constexpr int u = uc.value;
                    const bool put  = pd.hit[u];
                    const i32x2 cand{pd.key0 + u * TILE + i, __builtin_bit_cast(int, pd.s[u])};
                    g_pool.template store<2>(cand, put && slot < cap ? (g * cap + slot) * 2
                                                                      : (int)(OOB / 4));
                    slot += put ? 1 : 0;
                });
            };
            walk_lds<T::WAVES, T::STAGES, 0, STAGE_TILES + 1>(
                w, lds, g_k, nt, lane, wave,
                [&](int t, auto ttc, float s) {
                    constexpr int u = decltype(ttc)::value;
                    sc[u]  = s;
                    hit[u] = (unsigned)(d0 + t * TILE) < wn && s >= tau;
                },
                [&](int c, auto slotc) {
                    constexpr int p = decltype(slotc)::value % 2;
                    flush(opus::number<p>{});   // stage c - 2's
                    pending& pd = pick(opus::number<p>{});
                    int m = 0;
                    opus::static_for<STAGE_TILES>([&](auto uc) { m += hit[uc.value] ? 1 : 0; });
                    const int incl = row16_incl_scan(m);
                    const int tot  = row_bcast<15>(incl);
                    // the builtin on a fresh V#: opus's i32 atomic_add copies its V# through memory
                    pd.base = __builtin_amdgcn_raw_ptr_buffer_atomic_add_i32(
                        tot, opus::make_buffer_rsrc(cnt_base, (unsigned)valid_rows * 4u),
                        i == 0 ? g * 4 : (int)OOB, 0, 0);
                    pd.excl = incl - m;
                    pd.key0 = kb + c * STAGE_KEYS;
                    opus::static_for<STAGE_TILES>([&](auto uc) {
                        pd.s[uc.value]   = sc[uc.value];
                        pd.hit[uc.value] = hit[uc.value];
                    });
                });
            // the last two stages' (an empty set stores nothing)
            flush(opus::number<0>{});
            flush(opus::number<1>{});
    };

    segment(rb0 + wave * R, kb0, ke0);
}

}  // namespace opus_mqa_topk::gfx950
