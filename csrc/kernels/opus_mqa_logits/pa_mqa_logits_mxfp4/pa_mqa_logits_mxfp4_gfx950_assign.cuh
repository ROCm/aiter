// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Where a gfx950 MQA-logits CTA's work comes from (the kernels' template SCHED), shared by the
// one-row and qshare kernels. Device pass on gfx950 only. gfx1250's mqa_assign
// (pa_mqa_logits_mxfp4_gfx1250_assign.cuh) on wave64:
//
//   SCHED_TABLE    a record a CTA slot (the builder's, pa_mqa_logits_mxfp4_sched.cuh);
//   SCHED_FREE     no records: every CTA works out its own slot of the same kind of split the
//                  builder emits; every CTA reads every sequence's bounds (decode sizes);
//   SCHED_PREFILL  no records: a group is cut into a static `prefill_parts` pieces and
//                  slot / parts is the group.
// A group is up to QPB consecutive rows of one sequence, and its window is its LAST row's
// (windows do not decrease along a sequence). Every CTA computes each of its tiles the same way
// whatever cut it, so the three modes write bit-identical logits.
#pragma once
#include <opus/opus.hpp>

namespace opus_logits::gfx950 {

// Keep `v` in an SGPR. Must be the "+s" write form: the read-only form is a memory clobber
// that demotes s_loads to global_load + v_readfirstlane. Losing the pin costs occupancy silently.
template<class X>
__device__ inline void pin_sgpr(X& v) { asm("" : "+s"(v)); }

// What mqa_assign reads of the kargs. A kernel loads it with the rest of its arguments, ahead of
// its pins: read after them, these go out one dependent scalar round later, and the record a
// round after that (the one-row kernel, Table, cold: +0.05-0.4 us a call).
struct mqa_sched_args {
    const opus_mqa_cta_record* cta_info;
    const int* query_start_loc;
    const int* row_ends;
    int num_rows;
    int num_seqs;
    int cta_target;
    int prefill_parts;

    __device__ explicit mqa_sched_args(const mqa_logits_mxfp4_kargs& k)
        : cta_info(k.ptr_cta_info), query_start_loc(k.ptr_query_start_loc),
          row_ends(k.ptr_row_ends), num_rows(k.num_rows), num_seqs(k.num_seqs),
          cta_target(k.cta_target), prefill_parts(k.prefill_parts) {}
};

struct mqa_work {
    int row_id;        // the group's first row
    int batch_id;      // its sequence: the block-table row
    int group_rows;    // rows in the group, 1..QPB
    int chunk_start;   // first KV tile, absolute within the sequence's walk
    int chunk_count;   // tiles; > 0 for a CTA with work
    int local_end;     // the group's window end, its last row's row_ends
};

// Inclusive running sum over a wave64: row_shr 1/2/4/8 inside each 16-lane row, then row_bcast15
// into rows 1 and 3 and row_bcast31 into rows 2 and 3. `old` is 0 and bound_ctrl off, so a lane
// whose source is out of reach adds 0. FlyDSL's row-group kernel does the same ladder.
__device__ inline int wave64_incl_scan(int v) {
    v += __builtin_amdgcn_update_dpp(0, v, 0x111, 0xF, 0xF, false);
    v += __builtin_amdgcn_update_dpp(0, v, 0x112, 0xF, 0xF, false);
    v += __builtin_amdgcn_update_dpp(0, v, 0x114, 0xF, 0xF, false);
    v += __builtin_amdgcn_update_dpp(0, v, 0x118, 0xF, 0xF, false);
    v += __builtin_amdgcn_update_dpp(0, v, 0x142, 0xA, 0xF, false);
    v += __builtin_amdgcn_update_dpp(0, v, 0x143, 0xC, 0xF, false);
    return v;
}

// ceil(n / d) for 0 <= n, 1 <= d, n + d < 2^22, with rcp = 1 / d.
__device__ inline int ceil_div_rcp(int n, int d, float rcp) {
    const int m = n + d - 1;
    int q = (int)((float)m * rcp);
    q = q * d > m ? q - 1 : q;
    return (q + 1) * d <= m ? q + 1 : q;
}

// Every value that leaves here is wave-uniform (readlane / readfirstlane / the record), so the
// kernel's pin_sgpr on them holds.
template<int QPB, int TK, int SCHED>
__device__ inline bool mqa_assign(const mqa_sched_args& k, int lane, mqa_work& wk) {
    constexpr int WAVE = 64;
    const int slot = opus::block_id_x();

    if constexpr (SCHED == SCHED_TABLE) {
        // The whole assignment in one s_load_dwordx8 off a blockIdx-uniform address. A surplus
        // slot, or a row past num_rows (a mismatched record drops rows instead of reading past q
        // and writing past out), returns before anything is pinned: CTA-uniform, so it cannot
        // deadlock a barrier.
        const opus_mqa_cta_record rec = k.cta_info[slot];
        if (rec.chunk_count <= 0 || rec.row_id >= k.num_rows) return false;
        wk = {rec.row_id, rec.batch_id, rec.group_rows, rec.chunk_start, rec.chunk_count,
              rec.local_end};
        pin_sgpr(wk.batch_id); pin_sgpr(wk.local_end);
        return true;
    } else {
        const int  num_seqs = k.num_seqs;
        const int* qsl      = k.query_start_loc;
        auto last_row = [](int r0, int n, int j) {
            return r0 + ((j + 1) * QPB < n ? (j + 1) * QPB : n) - 1;
        };
        auto tiles_of = [](int e) { return e > 0 ? (e + TK - 1) / TK : 0; };
        auto groups_of = [](int n) { return (n + QPB - 1) / QPB; };
        auto set = [&](int b, int r0, int n, int j, int start, int count, int e) {
            wk.batch_id    = __builtin_amdgcn_readfirstlane(b);
            wk.row_id      = __builtin_amdgcn_readfirstlane(r0 + j * QPB);
            wk.group_rows  = __builtin_amdgcn_readfirstlane(n - j * QPB < QPB ? n - j * QPB : QPB);
            wk.chunk_start = __builtin_amdgcn_readfirstlane(start);
            wk.chunk_count = __builtin_amdgcn_readfirstlane(count);
            wk.local_end   = __builtin_amdgcn_readfirstlane(e);
        };

        if constexpr (SCHED == SCHED_PREFILL) {
            // Sequences in order, each sequence's groups last first: a sequence's windows grow
            // with the row, so its longest pieces start first and the short ones fill in at the
            // end; across sequences nothing orders the sizes.
            const int parts = k.prefill_parts;
            const int gs = slot / parts, part = slot - gs * parts;   // once a CTA, scalar
            int carry = 0;
            for (int base = 0; base < num_seqs; base += WAVE) {
                const int b = base + lane;
                int r0 = 0, n = 0;
                if (b < num_seqs) { r0 = qsl[b]; n = qsl[b + 1] - r0; }
                const int gb = groups_of(n), incl = wave64_incl_scan(gb), excl = incl - gb;
                const unsigned long long hit = __builtin_amdgcn_ballot_w64(
                    b < num_seqs && carry + excl <= gs && gs < carry + incl);
                if (hit) {
                    const int l  = __builtin_ctzll(hit);
                    const int sr = __builtin_amdgcn_readlane(r0, l), sn = __builtin_amdgcn_readlane(n, l);
                    const int j0 = gs - carry - __builtin_amdgcn_readlane(excl, l);
                    const int j  = groups_of(sn) - 1 - j0;
                    const int e  = __builtin_amdgcn_readfirstlane(k.row_ends[last_row(sr, sn, j)]);
                    const int tiles = tiles_of(e);
                    const int start = part * tiles / parts, count = (part + 1) * tiles / parts - start;
                    if (count <= 0) return false;
                    set(base + l, sr, sn, j, start, count, e);
                    return true;
                }
                carry += __builtin_amdgcn_readlane(incl, WAVE - 1);
            }
            return false;
        } else {   // SCHED_FREE
            // The first CHUNKS * 64 sequences are read once, up front: their bounds, then the window
            // ends of their first GC groups -- a decode bucket's (max_qlen <= 8) at the qlen each
            // instance is picked for -- so the whole assignment is two round trips. Sequences past
            // the cache, and groups past GC, are re-read where they are needed. Every load of a
            // round is issued before any is used -- indices clamped into range, the lanes past the
            // batch masked after -- or the rounds go serial; the cached chunks are walked unrolled,
            // not indexed through a select chain. The one-row kernel runs MTP 1 (and prefill) only:
            // caching its 8 rows cost every CTA 8x the loads and cuts (+5 us at bs 256).
            constexpr int CHUNKS = 4;
            constexpr int GC     = QPB == 1 ? 2 : (8 + QPB - 1) / QPB;
            int c_r0[CHUNKS], c_n[CHUNKS], c_e[CHUNKS][GC];
            int c_qa[CHUNKS], c_qb[CHUNKS];
            opus::static_for<CHUNKS>([&](auto cc) {   // round 1: the sequences' bounds
                constexpr int c = cc.value;
                c_qa[c] = 0; c_qb[c] = 0;
                if (c * WAVE < num_seqs) {   // uniform
                    const int b = min(c * WAVE + lane, num_seqs - 1);
                    c_qa[c] = qsl[b]; c_qb[c] = qsl[b + 1];
                }
            });
            // A lane's first GC groups' window ends (0 past its rows), every load issued first.
            auto group_ends = [&](int r0, int n, int (&e)[GC]) {
                int v[GC];
                opus::static_for<GC>([&](auto jc) {
                    const int l = last_row(r0, n, jc.value);
                    v[jc.value] = k.row_ends[l > 0 ? l : 0];   // a lane with no rows reads row 0
                });
                opus::static_for<GC>([&](auto jc) { e[jc.value] = n > jc.value * QPB ? v[jc.value] : 0; });
            };
            opus::static_for<CHUNKS>([&](auto cc) {   // round 2: their groups' window ends
                constexpr int c = cc.value;
                const bool live = c * WAVE + lane < num_seqs;
                c_r0[c] = live ? c_qa[c] : 0;
                c_n[c]  = live ? c_qb[c] - c_qa[c] : 0;
                opus::static_for<GC>([&](auto jc) { c_e[c][jc.value] = 0; });
                if (c * WAVE < num_seqs) group_ends(c_r0[c], c_n[c], c_e[c]);   // uniform
            });
            // visit(base, r0, n, ends) on each 64 sequences in order until it returns true.
            auto for_chunks = [&](auto&& visit) {
                bool done = false;
                opus::static_for<CHUNKS>([&](auto cc) {
                    constexpr int c = cc.value;
                    if (!done && c * WAVE < num_seqs) done = visit(c * WAVE, c_r0[c], c_n[c], c_e[c]);
                });
                for (int base = CHUNKS * WAVE; !done && base < num_seqs; base += WAVE) {
                    const int b = base + lane;
                    int r0 = 0, n = 0, e[GC];
                    if (b < num_seqs) { r0 = qsl[b]; n = qsl[b + 1] - r0; }
                    group_ends(r0, n, e);
                    done = visit(base, r0, n, e);
                }
            };
            // pass 1: total tiles, and the groups that have any
            int part = 0, ngroups = 0;
            for_chunks([&](int, int r0, int n, const int (&e)[GC]) {
                opus::static_for<GC>([&](auto jc) {
                    const int t = tiles_of(e[jc.value]);
                    part += t;
                    ngroups += t > 0;
                });
                for (int j = GC; j < groups_of(n); ++j) {
                    const int t = tiles_of(k.row_ends[last_row(r0, n, j)]);
                    part += t;
                    ngroups += t > 0;
                }
                return false;
            });
            const int total = __builtin_amdgcn_readlane(wave64_incl_scan(part), WAVE - 1);
            const int G     = __builtin_amdgcn_readlane(wave64_incl_scan(ngroups), WAVE - 1);
            // Each group's last chunk rounds up, so ceil(total / target) cuts up to G chunks past
            // the target -- past the resident CTAs, a second round of a few CTAs the kernel then
            // waits for. Cutting for target - G bounds the chunks by the target; with G a large
            // share of it, a group a CTA is the best there is anyway.
            const int tgt   = k.cta_target > 2 * G ? k.cta_target - G : k.cta_target;
            const int safe  = total > 0 ? ceil_div_rcp(total, tgt, 1.0f / (float)tgt) : 1;
            const float rcp = 1.0f / (float)safe;
            // pass 2: each group takes ceil(tiles / safe) slots, sequences and groups in order
            int carry = 0;
            bool found = false;
            for_chunks([&](int base, int r0, int n, const int (&e)[GC]) {
                int s = 0;
                opus::static_for<GC>([&](auto jc) { s += ceil_div_rcp(tiles_of(e[jc.value]), safe, rcp); });
                for (int j = GC; j < groups_of(n); ++j)
                    s += ceil_div_rcp(tiles_of(k.row_ends[last_row(r0, n, j)]), safe, rcp);
                const int incl = wave64_incl_scan(s), excl = incl - s;
                const unsigned long long hit = __builtin_amdgcn_ballot_w64(
                    carry + excl <= slot && slot < carry + incl);
                if (!hit) {
                    carry += __builtin_amdgcn_readlane(incl, WAVE - 1);
                    return false;
                }
                // a uniform walk of the hit sequence's groups for this slot's chunk
                const int l  = __builtin_ctzll(hit);
                const int sr = __builtin_amdgcn_readlane(r0, l), sn = __builtin_amdgcn_readlane(n, l);
                int o = slot - carry - __builtin_amdgcn_readlane(excl, l);
                auto take = [&](int j, int e_j) {
                    const int t  = tiles_of(e_j);
                    const int nc = ceil_div_rcp(t, safe, rcp);
                    if (o < nc) {
                        const int start = o * safe;
                        set(base + l, sr, sn, j, start, t - start < safe ? t - start : safe, e_j);
                        found = true;
                    }
                    o -= nc;
                };
                opus::static_for<GC>([&](auto jc) {
                    if (!found && jc.value < groups_of(sn))
                        take(jc.value, __builtin_amdgcn_readlane(e[jc.value], l));
                });
                for (int j = GC; !found && j < groups_of(sn); ++j)
                    take(j, __builtin_amdgcn_readfirstlane(k.row_ends[last_row(sr, sn, j)]));
                return true;
            });
            return found;   // false: a surplus slot
        }
    }
}

}  // namespace opus_logits::gfx950
