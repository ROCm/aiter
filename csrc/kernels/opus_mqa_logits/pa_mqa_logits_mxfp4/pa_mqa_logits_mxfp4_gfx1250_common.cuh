// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// What gfx1250's n_heads = 64 and n_heads = 32 kernels share: the device helpers and where a CTA's
// work comes from. Device pass on gfx1250 only.
#pragma once
#include <opus/opus.hpp>
#include <bit>   // std::bit_cast: __builtin_bit_cast miscompiles the permlane swap to v + v

#include "pa_mqa_logits_mxfp4_common.cuh"

namespace opus_logits::gfx1250 {

// -- helpers --------------------------------------------------------------------------------------

// relu as max_num, which VOPD issues two at a time where the IEEE `maximum` is one VOP3 an
// element. max_num drops NaN, and NaN logits are a contract (the scale pool's 0xFF rows; top-k
// ranks NaN first), so the kernels put the NaN back from the scales: a logit is NaN exactly when
// its key's or its row's E8M0 scales hold a 0xFF byte -- exactly when the WMMA's scores are NaN.
// The TU must be built with -fno-honor-nans: without it the compiler quiets the WMMA's output
// before every max_num and gains nothing.
__device__ inline float relu_num(float x) { return __builtin_fmaxf(x, 0.0f); }

// The cross-group finish for TWO n-tiles at once. v_permlane16_swap trades vdst's high row
// (lanes 16-31) for vsrc's low row (lanes 0-15), so swapping a with b leaves (a.lo, b.lo) and
// (a.hi, b.hi), and their sum holds n-tile a's 16 logits in lanes 0-15 and b's in lanes 16-31:
// one swap and one add for two n-tiles, and every lane a logit to store. Both halves of the
// builtin's result are used (r[0] alone computes v + v).
__device__ inline float pair_head_reduce(float a, float b) {
    opus::vector_t<opus::u32_t, 2> r = __builtin_amdgcn_permlane16_swap(
        std::bit_cast<opus::u32_t>(a), std::bit_cast<opus::u32_t>(b), false, false);
    return std::bit_cast<float>(r[0]) + std::bit_cast<float>(r[1]);
}

// A head sum's last add, (even heads, odd heads) -> one value, as one pk_add of the pair with
// its swap (op_sel) behind an empty asm: as a scalar add the backend packs two accumulators'
// adds into one pk_add and moves the halves into place for it. The asm is a VGPR operand, so
// only for kernels at or under 256 VGPRs (above it the VGPR-MSB mode makes VGPR asm unsafe).
template<bool ASM>
__device__ inline float pair_sum(opus::vector_t<float, 2> s) {
    if constexpr (ASM) {
        opus::vector_t<float, 2> d = s + __builtin_shufflevector(s, s, 1, 0);
        asm volatile("" : "+v"(d));
        return d[0];
    } else {
        return s[0] + s[1];
    }
}

// Does a dword of four E8M0 scales hold a 0xFF (NaN) byte? Each byte's low 7 bits plus one
// carries into its bit 7 only at 0x7F, and never into the next byte; with the byte's own bit 7,
// that is exactly 0xFF. Integer, because -fno-honor-nans folds an FP NaN test to false.
__device__ inline bool e8m0x4_has_nan(unsigned x) {
    return (((x & 0x7f7f7f7fu) + 0x01010101u) & x & 0x80808080u) != 0;
}

// s_setprio 3..0 by the remaining quarter of `chunk_max` tiles: a wave with more of its work
// left asks for the SIMD first, so the CTAs that started late are not starved into a
// latency-bound tail by the hardware's age order.
__device__ inline void set_prio_by_progress(int remaining, int chunk_max) {
    const int rem4 = remaining * 4;   // uniform
    if      (rem4 > 3 * chunk_max) __builtin_amdgcn_s_setprio(3);
    else if (rem4 > 2 * chunk_max) __builtin_amdgcn_s_setprio(2);
    else if (rem4 > chunk_max)     __builtin_amdgcn_s_setprio(1);
    else                           __builtin_amdgcn_s_setprio(0);
}

// -- where a CTA's work comes from ----------------------------------------------------------------
// The kernels' template SCHED:
//
//   SCHED_TABLE    a record a CTA slot (the builder's, pa_mqa_logits_mxfp4.cu);
//   SCHED_FREE     no records: every CTA works out its own slot of the same kind of split the
//                  builder emits; every CTA reads every sequence's bounds (decode sizes);
//   SCHED_PREFILL  no records: a group is cut into a static `prefill_parts` pieces and
//                  slot / parts is the group.
// A group is up to QPB consecutive rows of one sequence, and its window is its LAST row's
// (windows do not decrease along a sequence). Every CTA computes each of its tiles the same way
// whatever cut it, so the three modes write bit-identical logits.

struct mqa_work {
    int row_id;        // the group's first row
    int batch_id;      // its sequence: the block-table row
    int group_rows;    // rows in the group, 1..QPB
    int chunk_start;   // first 64-key tile, absolute within the sequence's walk
    int chunk_count;   // tiles; > 0 for a CTA with work
    int chunk_max;     // the wave priority's scale: Free, the longest chunk any CTA may hold;
                       // Table, the record's chunk_max (TABLE_CHUNK_MAX); else chunk_count
};

// Inclusive running sum over a wave32: DPP row_shr 1/2/4/8 inside each 16-lane row (a lane whose
// source is outside its row adds 0: `old` is 0, bound_ctrl off), then the low row's total added to
// the high row through readlane -- gfx1250 has no row_bcast15/31.
__device__ inline int wave32_incl_scan(int v, int lane) {
    v += __builtin_amdgcn_update_dpp(0, v, 0x111, 0xF, 0xF, false);
    v += __builtin_amdgcn_update_dpp(0, v, 0x112, 0xF, 0xF, false);
    v += __builtin_amdgcn_update_dpp(0, v, 0x114, 0xF, 0xF, false);
    v += __builtin_amdgcn_update_dpp(0, v, 0x118, 0xF, 0xF, false);
    const int low = __builtin_amdgcn_readlane(v, 15);
    return lane >= 16 ? v + low : v;
}

// Every value that leaves here is wave-uniform (readlane / readfirstlane), so the kernel's
// pin_sgpr on them holds.
template<int QPB, int TK, bool GATHER, int SCHED, bool TABLE_CHUNK_MAX = false>
__device__ inline bool mqa_assign(const mqa_logits_mxfp4_kargs& k, int lane, mqa_work& wk) {
    constexpr int WAVE = 32;
    const int slot = opus::block_id_x();

    if constexpr (SCHED == SCHED_TABLE) {
        // The whole record in one round, before the branch (the pins), and no early return: its
        // bool phi split the prologue, and the record's tail and the kargs the first tile needs
        // went out one dependent round apart.
        const opus_mqa_cta_record rec = k.ptr_cta_info[slot];
        wk = {rec.row_id, rec.batch_id, rec.group_rows, rec.chunk_start, rec.chunk_count,
              TABLE_CHUNK_MAX ? rec.chunk_max : rec.chunk_count};
        pin_sgpr(wk.row_id); pin_sgpr(wk.batch_id); pin_sgpr(wk.group_rows);
        pin_sgpr(wk.chunk_start); pin_sgpr(wk.chunk_count); pin_sgpr(wk.chunk_max);
        return wk.chunk_count > 0 && wk.row_id < k.num_rows;
    } else {
        static_assert(!(GATHER && SCHED == SCHED_FREE),
                      "candidate rows are each a group: Free's every-CTA-reads-every-sequence "
                      "does not pay there (Prefill does)");
        const int  num_seqs = k.num_seqs;
        const int* qsl      = k.ptr_query_start_loc;
        // Tiles of group j of a sequence (first row r0, n rows): its last row's window.
        auto group_tiles = [&](int r0, int n, int j) {
            const int last = r0 + ((j + 1) * QPB < n ? (j + 1) * QPB : n) - 1;
            const int e = k.ptr_row_ends[last];
            return e > 0 ? (e + TK - 1) / TK : 0;
        };
        auto groups_of = [](int n) { return (n + QPB - 1) / QPB; };
        auto set = [&](int b, int r0, int n, int j, int start, int count) {
            wk.batch_id    = __builtin_amdgcn_readfirstlane(b);
            wk.row_id      = __builtin_amdgcn_readfirstlane(r0 + j * QPB);
            wk.group_rows  = __builtin_amdgcn_readfirstlane(n - j * QPB < QPB ? n - j * QPB : QPB);
            wk.chunk_start = __builtin_amdgcn_readfirstlane(start);
            wk.chunk_count = __builtin_amdgcn_readfirstlane(count);
            wk.chunk_max   = wk.chunk_count;
        };

        if constexpr (SCHED == SCHED_PREFILL) {
            const int parts = k.prefill_parts;
            const int gs = slot / parts, part = slot - gs * parts;   // once a CTA, scalar
            if constexpr (GATHER) {
                // A row is its own group (group g is row g) and normally its own sequence: one
                // round trip checks that (qsl[g] == g, qsl[g + 1] == g + 1) beside the row's
                // window. Otherwise -- empty sequences, rows past them, several rows a sequence --
                // the row's sequence is searched for. No scan: a candidate grid runs to tens of
                // thousands of CTAs over as many rows.
                static_assert(QPB == 1, "a candidate row is its own group");
                const int g = gs;
                if (g >= k.num_rows) return false;   // the host's grid: a group a row, x parts
                const int gi = g < num_seqs ? g : num_seqs - 1;
                const int q0 = qsl[gi], q1 = qsl[gi + 1];
                int e = k.ptr_row_ends[g];
                // Landed before the branch, so the three loads share one round trip: otherwise the
                // window's load sinks past the branch, behind the wait for the bounds.
                pin_sgpr(e);
                int b = g;
                if (!(gi == g && q0 == g && q1 == g + 1)) {   // uniform
                    if (g >= qsl[num_seqs]) return false;
                    int lo = 0, hi = num_seqs;
                    while (lo < hi) {
                        const int mid = (lo + hi + 1) >> 1;
                        if (qsl[mid] <= g) lo = mid; else hi = mid - 1;
                    }
                    b = lo;
                }
                const int tiles = __builtin_amdgcn_readfirstlane(e > 0 ? (e + TK - 1) / TK : 0);
                const int start = part_point(part, tiles, parts),
                          count = part_point(part + 1, tiles, parts) - start;
                if (count <= 0) return false;
                set(b, g, 1, 0, start, count);
                return true;
            }
            // Sequences in order, each sequence's groups last first: a sequence's windows grow
            // with the row, so its longest pieces start first and the short ones fill in at the
            // end; across sequences nothing orders the sizes.
            int carry = 0;
            for (int base = 0; base < num_seqs; base += WAVE) {
                const int b = base + lane;
                int r0 = 0, n = 0;
                if (b < num_seqs) { r0 = qsl[b]; n = qsl[b + 1] - r0; }
                const int gb = groups_of(n), incl = wave32_incl_scan(gb, lane), excl = incl - gb;
                const unsigned hit = __builtin_amdgcn_ballot_w32(
                    b < num_seqs && carry + excl <= gs && gs < carry + incl);
                if (hit) {
                    const int l  = __builtin_ctz(hit);
                    const int sr = __builtin_amdgcn_readlane(r0, l);
                    const int sn = __builtin_amdgcn_readlane(n, l);
                    const int j0 = gs - carry - __builtin_amdgcn_readlane(excl, l);
                    const int j  = groups_of(sn) - 1 - j0;
                    const int tiles = __builtin_amdgcn_readfirstlane(group_tiles(sr, sn, j));
                    const int start = part_point(part, tiles, parts);
                    const int count = part_point(part + 1, tiles, parts) - start;
                    if (count <= 0) return false;
                    set(base + l, sr, sn, j, start, count);
                    return true;
                }
                carry += __builtin_amdgcn_readlane(incl, WAVE - 1);
            }
            return false;
        } else {   // SCHED_FREE
            // The first CHUNKS * 32 sequences are read once, up front: their bounds and their first
            // two groups' tile counts (decode has one or two groups a sequence), so the whole
            // assignment is two round trips. Groups past the second and sequences past the cache
            // are re-read where they are needed. Every load of a round is issued before any is
            // used -- indices clamped into range, the lanes past the batch masked after -- or the
            // rounds go serial.
            constexpr int CHUNKS = 8;
            int c_r0[CHUNKS], c_n[CHUNKS], c_t0[CHUNKS], c_t1[CHUNKS];
            int c_qa[CHUNKS], c_qb[CHUNKS], c_e0[CHUNKS], c_e1[CHUNKS];
            opus::static_for<CHUNKS>([&](auto cc) {   // round 1: the sequences' bounds
                constexpr int c = cc.value;
                c_qa[c] = 0; c_qb[c] = 0;
                if (c * WAVE < num_seqs) {   // uniform
                    const int b = min(c * WAVE + lane, num_seqs - 1);
                    c_qa[c] = qsl[b]; c_qb[c] = qsl[b + 1];
                }
            });
            opus::static_for<CHUNKS>([&](auto cc) {   // round 2: the first two groups' window ends
                constexpr int c = cc.value;
                const bool live = c * WAVE + lane < num_seqs;
                c_r0[c] = live ? c_qa[c] : 0;
                c_n[c]  = live ? c_qb[c] - c_qa[c] : 0;
                c_e0[c] = 0; c_e1[c] = 0;
                if (c * WAVE < num_seqs) {   // uniform; a lane with no rows reads row 0, harmlessly
                    const int n = c_n[c];
                    const int l0 = c_r0[c] + (n < QPB ? n : QPB) - 1;
                    const int l1 = c_r0[c] + (n < 2 * QPB ? n : 2 * QPB) - 1;
                    c_e0[c] = k.ptr_row_ends[l0 > 0 ? l0 : 0];
                    c_e1[c] = k.ptr_row_ends[l1 > 0 ? l1 : 0];
                }
            });
            opus::static_for<CHUNKS>([&](auto cc) {
                constexpr int c = cc.value;
                const int e0 = c_e0[c], e1 = c_e1[c];
                c_t0[c] = c_n[c] > 0   && e0 > 0 ? (e0 + TK - 1) / TK : 0;
                c_t1[c] = c_n[c] > QPB && e1 > 0 ? (e1 + TK - 1) / TK : 0;
            });
            // This wave's 32 sequences from `base` on: bounds and first two groups' tiles.
            auto load_seqs = [&](int base, int& r0, int& n, int& t0, int& t1) {
                const int c = base / WAVE;
                if (c < CHUNKS) {
                    r0 = c_r0[0]; n = c_n[0]; t0 = c_t0[0]; t1 = c_t1[0];
                    opus::static_for<CHUNKS>([&](auto i) {
                        if (i.value == c) {
                            r0 = c_r0[i.value]; n = c_n[i.value];
                            t0 = c_t0[i.value]; t1 = c_t1[i.value];
                        }
                    });
                } else {
                    const int b = base + lane;
                    r0 = 0; n = 0;
                    if (b < num_seqs) { r0 = qsl[b]; n = qsl[b + 1] - r0; }
                    t0 = n > 0 ? group_tiles(r0, n, 0) : 0;
                    t1 = n > QPB ? group_tiles(r0, n, 1) : 0;
                }
            };
            // pass 1: total tiles, and the groups that have any
            int part = 0, ngroups = 0;
            for (int base = 0; base < num_seqs; base += WAVE) {
                int r0, n, t0, t1;
                load_seqs(base, r0, n, t0, t1);
                part += t0 + t1;
                ngroups += (t0 > 0) + (t1 > 0);
                for (int j = 2; j < groups_of(n); ++j) {
                    const int t = group_tiles(r0, n, j);
                    part += t;
                    ngroups += t > 0;
                }
            }
            const int total = __builtin_amdgcn_readlane(wave32_incl_scan(part, lane), WAVE - 1);
            const int G     = __builtin_amdgcn_readlane(wave32_incl_scan(ngroups, lane), WAVE - 1);
            // Each group's last chunk rounds up, so ceil(total / target) cuts up to G chunks past
            // the target -- past the resident CTAs, a second round of a few CTAs the kernel then
            // waits for. Cutting for target - G bounds the chunks by the target; with G a large
            // share of it, a group a CTA is the best there is anyway.
            const int tgt   = k.cta_target > 2 * G ? k.cta_target - G : k.cta_target;
            const int safe  = total > 0 ? ceil_div_rcp(total, tgt, 1.0f / (float)tgt) : 1;
            const float rcp = 1.0f / (float)safe;
            // pass 2: each group takes ceil(tiles / safe) slots, sequences and groups in order
            int carry = 0;
            for (int base = 0; base < num_seqs; base += WAVE) {
                int r0, n, t0, t1;
                load_seqs(base, r0, n, t0, t1);
                int s = ceil_div_rcp(t0, safe, rcp) + ceil_div_rcp(t1, safe, rcp);
                for (int j = 2; j < groups_of(n); ++j)
                    s += ceil_div_rcp(group_tiles(r0, n, j), safe, rcp);
                const int incl = wave32_incl_scan(s, lane), excl = incl - s;
                const unsigned hit =
                    __builtin_amdgcn_ballot_w32(carry + excl <= slot && slot < carry + incl);
                if (hit) {
                    const int l  = __builtin_ctz(hit);
                    const int sr = __builtin_amdgcn_readlane(r0, l);
                    const int sn = __builtin_amdgcn_readlane(n, l);
                    int o = slot - carry - __builtin_amdgcn_readlane(excl, l);
                    const int g0 = __builtin_amdgcn_readlane(t0, l);
                    const int g1 = __builtin_amdgcn_readlane(t1, l);
                    // a uniform walk of the sequence's groups
                    for (int j = 0; j < groups_of(sn); ++j) {
                        const int t  = j == 0   ? g0
                                       : j == 1 ? g1
                                       : __builtin_amdgcn_readfirstlane(group_tiles(sr, sn, j));
                        const int nc = ceil_div_rcp(t, safe, rcp);
                        if (o < nc) {
                            const int start = o * safe;
                            set(base + l, sr, sn, j, start, t - start < safe ? t - start : safe);
                            wk.chunk_max = __builtin_amdgcn_readfirstlane(safe);
                            return true;
                        }
                        o -= nc;
                    }
                    return false;
                }
                carry += __builtin_amdgcn_readlane(incl, WAVE - 1);
            }
            return false;   // a surplus slot
        }
    }
}

}  // namespace opus_logits::gfx1250
