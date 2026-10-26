// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Where a gfx1250 MQA-logits CTA's work comes from (the kernels' template SCHED), shared by the
// H = 64 and H = 32 kernels. Device pass on gfx1250 only.
//
//   SCHED_TABLE    a record a CTA slot (the builder's, mqa_logits_mxfp4_sched.cuh);
//   SCHED_FREE     no records: every CTA works out its own slot of the same kind of split the
//                  builder emits; every CTA reads every sequence's bounds (decode sizes);
//   SCHED_PREFILL  no records: a group is cut into a static `prefill_parts` pieces and
//                  slot / parts is the group.
// A group is up to QPB consecutive rows of one sequence, and its window is its LAST row's
// (windows do not decrease along a sequence). Every CTA computes each of its tiles the same way
// whatever cut it, so the three modes write bit-identical logits.
#pragma once
#include "mqa_logits_mxfp4_gfx1250_common.cuh"

namespace opus_logits::gfx1250 {

struct mqa_work {
    int row_id;        // the group's first row
    int batch_id;      // its sequence: the block-table row
    int group_rows;    // rows in the group, 1..QPB
    int chunk_start;   // first 64-key tile, absolute within the sequence's walk
    int chunk_count;   // tiles; > 0 for a CTA with work
    int chunk_max;     // the wave priority's scale: Free, the longest chunk any CTA may hold;
                       // Table, the record's (TABLE_PAD: its `_pad`); else chunk_count
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

// ceil(n / d) for 0 <= n, 1 <= d, n + d < 2^22, with rcp = 1 / d.
__device__ inline int ceil_div_rcp(int n, int d, float rcp) {
    const int m = n + d - 1;
    int q = (int)((float)m * rcp);
    q = q * d > m ? q - 1 : q;
    return (q + 1) * d <= m ? q + 1 : q;
}

// Every value that leaves here is wave-uniform (readlane / readfirstlane), so the kernel's
// pin_sgpr on them holds.
template<int QPB, int TK, bool GATHER, int SCHED, bool TABLE_PAD = false>
__device__ inline bool mqa_assign(const mqa_logits_mxfp4_kargs& k, int lane, mqa_work& wk) {
    constexpr int WAVE = 32;
    const int slot = opus::block_id_x();

    if constexpr (SCHED == SCHED_TABLE) {
        const opus_mqa_cta_record rec = k.ptr_cta_info[slot];
        if (rec.chunk_count <= 0 || rec.row_id >= k.num_rows) return false;
        wk = {rec.row_id, rec.batch_id, rec.group_rows, rec.chunk_start, rec.chunk_count,
              TABLE_PAD ? rec._pad : rec.chunk_count};
        return true;
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
            // the host's grid is this static bound times parts
            const int groups = (k.num_rows + num_seqs * (QPB - 1)) / QPB;
            const int gs = slot / parts, part = slot - gs * parts;   // once a CTA, scalar
            if constexpr (GATHER) {
                // A row is its own group (group g is row g) and normally its own sequence: one
                // round trip checks that (qsl[g] == g, qsl[g + 1] == g + 1) beside the row's
                // window. Otherwise -- empty sequences, rows past them, several rows a sequence --
                // the row's sequence is searched for. No scan: a candidate grid runs to tens of
                // thousands of CTAs over as many rows.
                const int g = gs;
                if (g >= groups) return false;
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
                const int start = part * tiles / parts, count = (part + 1) * tiles / parts - start;
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
                const unsigned hit = __builtin_amdgcn_ballot_w32(b < num_seqs && carry + excl <= gs && gs < carry + incl);
                if (hit) {
                    const int l  = __builtin_ctz(hit);
                    const int sr = __builtin_amdgcn_readlane(r0, l), sn = __builtin_amdgcn_readlane(n, l);
                    const int j0 = gs - carry - __builtin_amdgcn_readlane(excl, l);
                    const int j  = groups_of(sn) - 1 - j0;
                    const int tiles = __builtin_amdgcn_readfirstlane(group_tiles(sr, sn, j));
                    const int start = part * tiles / parts, count = (part + 1) * tiles / parts - start;
                    if (count <= 0) return false;
                    set(base + l, sr, sn, j, start, count);
                    return true;
                }
                carry += __builtin_amdgcn_readlane(incl, WAVE - 1);
            }
            (void)groups;
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
                        if (i.value == c) { r0 = c_r0[i.value]; n = c_n[i.value]; t0 = c_t0[i.value]; t1 = c_t1[i.value]; }
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
                for (int j = 2; j < groups_of(n); ++j) s += ceil_div_rcp(group_tiles(r0, n, j), safe, rcp);
                const int incl = wave32_incl_scan(s, lane), excl = incl - s;
                const unsigned hit = __builtin_amdgcn_ballot_w32(carry + excl <= slot && slot < carry + incl);
                if (hit) {
                    const int l  = __builtin_ctz(hit);
                    const int sr = __builtin_amdgcn_readlane(r0, l), sn = __builtin_amdgcn_readlane(n, l);
                    int o = slot - carry - __builtin_amdgcn_readlane(excl, l);
                    const int g0 = __builtin_amdgcn_readlane(t0, l), g1 = __builtin_amdgcn_readlane(t1, l);
                    for (int j = 0; j < groups_of(sn); ++j) {   // uniform walk of the sequence's groups
                        const int t  = j == 0 ? g0 : j == 1 ? g1 : __builtin_amdgcn_readfirstlane(group_tiles(sr, sn, j));
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
