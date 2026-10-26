// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// The n_heads = 32 (DeepSeek-V4.1) kernel on gfx1250, full layers and candidate layers. Device
// pass on gfx1250 only (mqa_logits_mxfp4_gfx1250.cu). opus-ops opus_logits/gfx1250/docs/
// V41_DEV_PLAN_DRAFT.md has the measurements behind each choice.
//
//   * R rows a wave. A tile's keys reach HBM -> LDS once per CTA (TDM, one piece a wave) and
//     LDS -> VGPR once per wave; every B fragment a wave reads feeds R WMMAs (at H = 32 one WMMA
//     32x16x128 is all heads of one row against 16 keys, so the rows ARE the reuse).
//   * A tile's 4R accumulators are pipelined by pairs -- two WMMAs, then the previous pair's two
//     epilogues -- and the finish pairs n-tiles 2u / 2u+1 of a row into one permlane16_swap and
//     one 32-lane store.
//   * Full layers: a table entry names a run of 64-key tiles (2^tpb_shift of them), so tile t of
//     a sequence is plane row (bt[t >> tpb_shift] << tpb_shift | t & mask) * 64, one TDM copy.
//     Candidate layers (GATHER): an entry is one 8-row page and a tile is one TDM gather of 8.
//   * Output [rows, out_stride], or flat: row r's logits at out_offsets[r].
//
// Where a CTA's work comes from (template SCHED):
//   SCHED_TABLE    a record a CTA slot (the builder's);
//   SCHED_FREE     no records: every CTA works out its own slot of the same kind of split the
//                  builder emits; every CTA reads every sequence's bounds (decode sizes);
//   SCHED_PREFILL  no records: a group is cut into a static `prefill_parts` pieces and
//                  slot / parts is the group.
// A group is up to ROWS_PER_CTA consecutive rows of one sequence, and its window is its LAST
// row's (windows do not decrease along a sequence).
#pragma once
#include "mqa_logits_mxfp4_gfx1250_common.cuh"

namespace opus_logits::gfx1250 {

struct v41_work {
    int row_id;        // the group's first row
    int batch_id;      // its sequence: the block-table row
    int group_rows;    // rows in the group, 1..ROWS_PER_CTA
    int chunk_start;   // first 64-key tile, absolute within the sequence's walk
    int chunk_count;   // tiles; > 0 for a CTA with work
    int chunk_max;     // Free: the longest chunk any CTA may hold (the priority's scale);
                       // else chunk_count
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
__device__ inline int v41_ceil_div(int n, int d, float rcp) {
    const int m = n + d - 1;
    int q = (int)((float)m * rcp);
    q = q * d > m ? q - 1 : q;
    return (q + 1) * d <= m ? q + 1 : q;
}

// Every value that leaves here is wave-uniform (readlane / readfirstlane), so the kernel's
// pin_sgpr on them holds.
template<class T, int SCHED>
__device__ inline bool v41_assign(const mqa_logits_mxfp4_kargs& k, int lane, v41_work& wk) {
    constexpr int QPB  = T::ROWS_PER_CTA;
    constexpr int TK   = T::TILE_KEYS;
    constexpr int WAVE = T::WAVE_SIZE;
    const int slot = opus::block_id_x();

    if constexpr (SCHED == SCHED_TABLE) {
        const opus_mqa_cta_record rec = k.ptr_cta_info[slot];
        if (rec.chunk_count <= 0 || rec.row_id >= k.num_rows) return false;
        wk = {rec.row_id, rec.batch_id, rec.group_rows, rec.chunk_start, rec.chunk_count, rec.chunk_count};
        return true;
    } else {
        static_assert(!(T::GATHER && SCHED == SCHED_FREE),
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
            if constexpr (T::GATHER) {
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
            const int safe  = total > 0 ? v41_ceil_div(total, tgt, 1.0f / (float)tgt) : 1;
            const float rcp = 1.0f / (float)safe;
            // pass 2: each group takes ceil(tiles / safe) slots, sequences and groups in order
            int carry = 0;
            for (int base = 0; base < num_seqs; base += WAVE) {
                int r0, n, t0, t1;
                load_seqs(base, r0, n, t0, t1);
                int s = v41_ceil_div(t0, safe, rcp) + v41_ceil_div(t1, safe, rcp);
                for (int j = 2; j < groups_of(n); ++j) s += v41_ceil_div(group_tiles(r0, n, j), safe, rcp);
                const int incl = wave32_incl_scan(s, lane), excl = incl - s;
                const unsigned hit = __builtin_amdgcn_ballot_w32(carry + excl <= slot && slot < carry + incl);
                if (hit) {
                    const int l  = __builtin_ctz(hit);
                    const int sr = __builtin_amdgcn_readlane(r0, l), sn = __builtin_amdgcn_readlane(n, l);
                    int o = slot - carry - __builtin_amdgcn_readlane(excl, l);
                    const int g0 = __builtin_amdgcn_readlane(t0, l), g1 = __builtin_amdgcn_readlane(t1, l);
                    for (int j = 0; j < groups_of(sn); ++j) {   // uniform walk of the sequence's groups
                        const int t  = j == 0 ? g0 : j == 1 ? g1 : __builtin_amdgcn_readfirstlane(group_tiles(sr, sn, j));
                        const int nc = v41_ceil_div(t, safe, rcp);
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

template<class T, int SCHED>
__global__ __launch_bounds__(T::BLOCK_SIZE, 1)
void v41_kernel(mqa_logits_mxfp4_kargs k) {
    using D_BYTE   = opus::u8_t;
    using D_SCALE  = typename T::D_SCALE;
    using D_WEIGHT = typename T::D_WEIGHT;
    using Mma = opus::wmma<opus::fp4_t, opus::fp4_t, float, T::MMA_M, T::MMA_N, T::MMA_K, T::WAVE_SIZE>;
    Mma mma;
    constexpr int R      = T::ROWS_PER_WAVE;
    constexpr int NT     = T::N_TILES;            // 4
    constexpr int EC     = T::C_FRAG;             // 16
    constexpr int WAVE   = T::WAVE_SIZE;
    constexpr int PACK_W = T::PACK_C;             // 8 bf16 = one 16 B load
    constexpr int DW     = (int)sizeof(D_SCALE);
    constexpr int GRP_DW = T::KV_GRP_BYTES / DW;
    constexpr int TK     = T::TILE_KEYS;
    constexpr int STAGES = T::STAGES;
    static_assert(T::M_TILES == 1 && T::PAGES_PER_TILE == 1 && NT == 4,
                  "V4.1 is H = 32 (one m-tile) with a 64-key tile that is one contiguous run");
    static_assert(T::TDM_ISSUE_WAVES == T::NUM_WAVES, "every wave issues one piece of the tile");

    __shared__ __attribute__((aligned(16))) D_BYTE smem_buf[T::LDS_BYTES_V41];

    const int tid     = opus::thread_id_x();
    const int wave    = __builtin_amdgcn_readfirstlane(tid / WAVE);
    const int lane    = tid % WAVE;
    const int lane_m0 = lane % T::GRPN_C;
    const int lane_g  = lane / T::GRPN_C;

    int num_rows = k.num_rows;
    float weight_scale = k.weight_scale;
    int tpb_shift = k.tpb_shift;
    pin_sgpr(num_rows); pin_sgpr(weight_scale); pin_sgpr(tpb_shift);
    // This CTA's group and tile range: a schedule record, or worked out here.
    v41_work wk;
    if (!v41_assign<T, SCHED>(k, lane, wk)) return;             // CTA-uniform
    const int chunk_start = wk.chunk_start;                     // absolute tile index
    const int tile_count  = wk.chunk_count;
    const int chunk_max   = wk.chunk_max;                       // the priority's scale
    int batch_id = wk.batch_id;
    pin_sgpr(batch_id);

    // ---- the tile's 64 rows: block table -> plane row, TDM into the LDS ring ----
    // The table through the constant address space so its words come from s_load.
    using bt_scalar_ptr = const int __attribute__((address_space(4)))*;
    auto bt_c = reinterpret_cast<bt_scalar_ptr>(
        reinterpret_cast<__UINTPTR_TYPE__>(k.ptr_block_tables + (size_t)batch_id * k.bt_stride));
    const int tpb_mask = (1 << tpb_shift) - 1;
    // What tile t's issue reads from the table: its entry, or (candidate rows) its 8 page ids --
    // one s_load_dwordx8, the row's table is wave-uniform.
    constexpr int BTW = T::GATHER ? T::TILE_PAGES : 1;
    auto bt_words = [&](int t, int (&wds)[BTW]) {
        if constexpr (T::GATHER)
            opus::static_for<BTW>([&](auto i) { wds[i.value] = bt_c[(chunk_start + t) * T::TILE_PAGES + i.value]; });
        else
            wds[0] = bt_c[(chunk_start + t) >> tpb_shift];
    };

    // CU scope, not opus's device default (which defeats the L2). Legal only because KV and
    // scales are read-only for the launch: nothing may write the KV cache during the launch.
    using fp4x2 = opus::array<opus::fp4_t, 2>;
    constexpr int TDM_CACHE = opus::tdm_traits::make_cache_policy(
        opus::tdm_traits::load_temporal_hint::regular, opus::tdm_traits::scope::cu);
    using Cache  = opus::tdm_traits::cache<TDM_CACHE>;
    using KvWin  = opus::tdm<fp4x2, opus::seq<T::TOKEN_BYTES, T::TDM_ROWS>,
                             opus::tdm_traits::padding<fp4x2, T::LDS_PAD_INTERVAL, T::LDS_PAD_AMOUNT>, Cache>;
    using KvsWin = opus::tdm<opus::u8_t, opus::seq<T::SCALE_BYTES_PER_DWORD, T::TDM_ROWS>, Cache>;
    constexpr opus::u32_t TDM_EXTENT = 1u << 30;   // rows; the address moves, the extent only clamps
    const int tdm_piece    = wave;                 // one piece of the 64-row tile a wave
    const int kv_lds_base  = tdm_piece * T::LDS_PIECE_BYTES;
    const int kvs_lds_base = T::lds_scale_byte(tdm_piece * T::TDM_ROWS);
    auto kv_win = opus::make_tdm<KvWin>(
        (opus::u32_t)kv_lds_base, reinterpret_cast<const D_BYTE*>(k.ptr_kv),
        (opus::u32_t)T::TOKEN_BYTES, TDM_EXTENT, (opus::u64_t)T::TOKEN_BYTES, 0u, 0u);
    auto kvs_win = opus::make_tdm<KvsWin>(
        (opus::u32_t)kvs_lds_base, reinterpret_cast<const D_BYTE*>(k.ptr_kv_scale),
        (opus::u32_t)T::SCALE_BYTES_PER_DWORD, TDM_EXTENT, (opus::u64_t)T::SCALE_BYTES_PER_DWORD, 0u, 0u);
    // Candidate layers: the planes as [num_pages][512 B] / [num_pages][32 B], one gather of the
    // tile's 8 page ids each. The pad runs on across gathered rows, so the LDS image is the
    // contiguous tile's byte for byte, and the ids need not be sorted.
    constexpr int PAGE_VAL_BYTES = T::PAGE_ROWS * T::TOKEN_BYTES;                 // 512
    constexpr int PAGE_SCL_BYTES = T::PAGE_ROWS * T::SCALE_BYTES_PER_DWORD;       // 32
    using KvgWin  = opus::tdm<fp4x2, opus::seq<PAGE_VAL_BYTES, T::TILE_PAGES>,
                              opus::tdm_traits::padding<fp4x2, T::LDS_PAD_INTERVAL, T::LDS_PAD_AMOUNT>,
                              opus::tdm_traits::gather<32>, Cache>;
    using KvsgWin = opus::tdm<opus::u8_t, opus::seq<PAGE_SCL_BYTES, T::TILE_PAGES>,
                              opus::tdm_traits::gather<32>, Cache>;
    auto kvg_win = opus::make_tdm<KvgWin>(
        (opus::u32_t)kv_lds_base, reinterpret_cast<const D_BYTE*>(k.ptr_kv),
        (opus::u32_t)PAGE_VAL_BYTES, TDM_EXTENT, (opus::u64_t)PAGE_VAL_BYTES, 0u, 0u);
    auto kvsg_win = opus::make_tdm<KvsgWin>(
        (opus::u32_t)kvs_lds_base, reinterpret_cast<const D_BYTE*>(k.ptr_kv_scale),
        (opus::u32_t)PAGE_SCL_BYTES, TDM_EXTENT, (opus::u64_t)PAGE_SCL_BYTES, 0u, 0u);
    // Each window's D# at origin 0, built once; a tile's issue patches only what changes -- the
    // global address (or, gathering, the page ids) and the LDS stage.
    const typename KvWin::descriptor  kv_d0  = kv_win.make_descriptor();
    const typename KvsWin::descriptor kvs_d0 = kvs_win.make_descriptor();
    typename KvgWin::descriptor  kvg_d0  = kvg_win.make_descriptor();
    typename KvsgWin::descriptor kvsg_d0 = kvsg_win.make_descriptor();
    kvg_d0.set_tile_dim1(T::TILE_PAGES);    // gather mode: tile_dim1 = valid indices
    kvsg_d0.set_tile_dim1(T::TILE_PAGES);
    const opus::u64_t kv_g  = reinterpret_cast<opus::u64_t>(k.ptr_kv);
    const opus::u64_t kvs_g = reinterpret_cast<opus::u64_t>(k.ptr_kv_scale);
    // Tile t of the chunk into ring slot `slot`. Nothing is issued past the chunk -- the wait below
    // drains instead -- so a short chunk does not fetch the ring's depth of duplicate tiles.
    // SCALE_TDM = 0 reads the tile's scales per lane into a register ring instead of a second TDM
    // op, which halves the tensor ops a tile costs a wave and so allows a deeper ring.
    int kvs_ring[STAGES][NT / 2];
    auto issue_tile = [&](int t, auto slot_c, const int (&wds)[BTW]) {
        constexpr int slot = decltype(slot_c)::value;
        if (t >= tile_count) return;
        const opus::u32_t stage = (opus::u32_t)(slot * T::LDS_STAGE_BYTES);
        if constexpr (T::GATHER) {
            int ids[T::TILE_PAGES];
            opus::static_for<T::TILE_PAGES>([&](auto i) { ids[i.value] = wds[i.value]; });
            auto gd = kvg_d0;
            gd.set_lds_addr((opus::u32_t)kv_lds_base + stage);
            gd.groups23.group2 = opus::impl::tdm_sgpr4{ids[0], ids[1], ids[2], ids[3]};
            gd.groups23.group3 = opus::impl::tdm_sgpr4{ids[4], ids[5], ids[6], ids[7]};
            KvgWin::async_load(gd);
            if constexpr (T::SCALE_TDM) {
                auto gs = kvsg_d0;
                gs.set_lds_addr((opus::u32_t)kvs_lds_base + stage);
                gs.groups23.group2 = opus::impl::tdm_sgpr4{ids[0], ids[1], ids[2], ids[3]};
                gs.groups23.group3 = opus::impl::tdm_sgpr4{ids[4], ids[5], ids[6], ids[7]};
                KvsgWin::async_load(gs);
            } else {
                // lane L, half u: key 32u + L, in page (32u + L) / 8 of the tile -- a select over
                // the eight scalar ids, then the key's dword within that page's 32 B
                opus::static_for<NT / 2>([&](auto uc) {
                    constexpr int u = uc.value;
                    const int key = u * 2 * T::MMA_N + lane;
                    int pid = ids[0];
                    opus::static_for<T::TILE_PAGES>([&](auto i) { pid = (key >> 3) == i.value ? ids[i.value] : pid; });
                    kvs_ring[slot][u] = *reinterpret_cast<const int*>(
                        reinterpret_cast<const D_BYTE*>(k.ptr_kv_scale) +
                        (size_t)(opus::u32_t)pid * PAGE_SCL_BYTES + (key & 7) * T::SCALE_BYTES_PER_DWORD);
                });
            }
            return;
        }
        const opus::u32_t row0 =
            (((opus::u32_t)wds[0] << tpb_shift) + (opus::u32_t)((chunk_start + t) & tpb_mask)) * TK;
        const opus::u32_t origin = row0 + tdm_piece * T::TDM_ROWS;
        {
            auto d = kv_d0;
            d.set_lds_addr((opus::u32_t)kv_lds_base + stage);
            d.set_global_addr(kv_g + (opus::u64_t)origin * T::TOKEN_BYTES);
            KvWin::async_load(d);
        }
        if constexpr (T::SCALE_TDM) {
            auto d = kvs_d0;
            d.set_lds_addr((opus::u32_t)kvs_lds_base + stage);
            d.set_global_addr(kvs_g + (opus::u64_t)origin * T::SCALE_BYTES_PER_DWORD);
            KvsWin::async_load(d);
        } else {
            // lane L, half u: key 32u + L of the tile, i.e. n-tiles 2u (low lanes) and 2u+1 (high)
            opus::static_for<NT / 2>([&](auto uc) {
                constexpr int u = uc.value;
                kvs_ring[slot][u] = *reinterpret_cast<const int*>(
                    reinterpret_cast<const D_BYTE*>(k.ptr_kv_scale) +
                    ((size_t)(row0 + u * 2 * T::MMA_N + lane)) * T::SCALE_BYTES_PER_DWORD);
            });
        }
    };

    auto s_dw = opus::make_smem(reinterpret_cast<D_SCALE*>(smem_buf));
    const int lds_lane_dw = T::lds_byte(lane_m0, lane_g) / DW;
    auto read_kv = [&](typename Mma::vtype_b (&kv)[NT], int (&kvs)[NT / 2], auto slot_c) {
        constexpr int slot = decltype(slot_c)::value;
        const int stage_dw = slot * T::LDS_STAGE_BYTES / DW;
        opus::static_for<NT / 2>([&](auto uc) {
            constexpr int u = uc.value;
            if constexpr (T::SCALE_TDM)
                kvs[u] = opus::load<1>(s_dw, stage_dw + T::lds_scale_byte(u * 2 * T::MMA_N) / DW + lane)[0];
            else
                kvs[u] = kvs_ring[slot][u];
        });
        opus::static_for<NT>([&](auto ntc) {
            constexpr int nt   = ntc.value;
            constexpr int base = nt * T::LDS_NTILE_BYTES / DW;
            opus::static_for<T::B_VGPR_GROUPS>([&](auto vc) {
                constexpr int v   = vc.value;
                constexpr int imm = v * 2 * T::LDS_BLOCK_BYTES / DW;
                auto x = opus::load<GRP_DW>(s_dw, stage_dw + base + imm + lds_lane_dw);
                opus::set_slice(kv[nt], x, opus::number<v * GRP_DW>{}, opus::number<(v + 1) * GRP_DW>{});
            });
        });
    };

    // relu x weight summed over the lane's 16 heads, as TWO independent pk_fma chains joined by
    // one pk_add: one chain of 8 dependent steps stalls at these occupancies.
    // Under the NaN restore, few rows: a NaN key (its scales hold a 0xFF byte) makes every head of
    // its column NaN, so one element a lane through the IEEE maximum carries it to the logit, and
    // no per-tile test of the key scales is needed. asm, or -fno-honor-nans lets the compiler pick
    // max_num; a VGPR operand, so only at R <= 2 (well under 256 VGPRs).
    constexpr bool NAN_ELEM = MQA_LOGITS_NAN_RESTORE && R <= 2;
    auto head_sum = [&](const typename Mma::vtype_c& c, const opus::vector_t<float, EC>& wr) {
        opus::vector_t<float, 2> s2[2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
        opus::static_for<EC / 2>([&](auto pc) {
            constexpr int p = pc.value;
            opus::vector_t<float, 2> wv{wr[2 * p], wr[2 * p + 1]};
            opus::vector_t<float, 2> rv{MQA_LOGITS_RELU(c[2 * p]), MQA_LOGITS_RELU(c[2 * p + 1])};
            if constexpr (NAN_ELEM && p == 0) {
                float x = c[0];
                asm volatile("v_maximum_f32 %0, %0, 0" : "+v"(x));
                rv[0] = x;
            }
            s2[p % 2] = rv * wv + s2[p % 2];
        });
        return s2[0] + s2[1];   // (even heads, odd heads): the pair finish adds the halves
    };
    constexpr int LEAD = STAGES - 1;   // tiles in flight ahead of the one being computed
    opus::static_for<LEAD>([&](auto sc) {
        if (sc.value < tile_count) { int wds[BTW]; bt_words(sc.value, wds); issue_tile(sc.value, sc, wds); }
    });
    int pf[BTW];   // the words of the next tile to issue, read one tile ahead
    if (LEAD < tile_count) bt_words(LEAD, pf);
    // ---- this wave's R rows: Q, its scales, the weights, and an out V# a row ----
    // Read after the first tiles' TDM is issued, so the two start-up latencies overlap.
    // A slot past the group's rows runs with its first row's operands and a zero-record V#, so it
    // stores nothing and the CTA's barrier count stays uniform.
    typename Mma::vtype_a q_a[R];
    int q_sc[R];
    opus::vector_t<float, EC> w[R];
    float* out_ptr[R];          // each row's window: base and byte count, uniform (SGPR)
    unsigned out_bytes[R];
    unsigned q_nan_rows = 0;    // bit r = row r's q scales hold a NaN (uniform)
    opus::static_for<R>([&](auto rc) {
        constexpr int r = rc.value;
        const int  slot = wave * R + r;
        const bool ok   = slot < wk.group_rows && wk.row_id + slot < num_rows;
        int row = wk.row_id + (ok ? slot : 0);
        pin_sgpr(row);
        int end = ok ? k.ptr_row_ends[row] : 0;
        end = __builtin_amdgcn_readfirstlane(end);

        auto g_q = opus::make_gmem(reinterpret_cast<const D_SCALE*>(
            reinterpret_cast<const D_BYTE*>(k.ptr_q) + (size_t)row * T::Q_ROW_BYTES));
        opus::static_for<T::A_VGPR_GROUPS>([&](auto vc) {
            constexpr int v = vc.value;
            const int head  = lane_m0 + (v >= 2 ? 16 : 0);
            const int block = lane_g + (v % 2 ? 2 : 0);
            auto x = opus::load<GRP_DW>(g_q, T::q_byte(head, block) / DW);
            opus::set_slice(q_a[r], x, opus::number<v * GRP_DW>{}, opus::number<(v + 1) * GRP_DW>{});
        });
        auto g_qs = opus::make_gmem(reinterpret_cast<const D_SCALE*>(k.ptr_q_scale) + (size_t)row * T::N_HEADS);
        q_sc[r] = opus::load<1>(g_qs, lane)[0];
        auto g_w = opus::make_gmem(reinterpret_cast<const D_WEIGHT*>(k.ptr_weights) + (size_t)row * T::W_ROW_ELEMS);
        opus::static_for<T::REPT_C>([&](auto hc) {
            constexpr int h = hc.value;
            auto wv = opus::load<PACK_W>(g_w, (h ? 16 : 0) + lane_g * PACK_W);
            opus::static_for<PACK_W>([&](auto jc) {
                w[r][h * PACK_W + jc.value] = (float)wv[jc.value] * weight_scale;
            });
        });
        const long long base = k.ptr_out_offsets ? k.ptr_out_offsets[row] : (long long)row * k.out_stride;
        out_ptr[r]   = k.ptr_out + base;
        out_bytes[r] = (unsigned)(end > 0 ? end : 0) * (unsigned)sizeof(float);
    });

#if MQA_LOGITS_NAN_RESTORE
    // Lane h holds head h's four block scales: a row's logits are all NaN if any is 0xFF. Tested
    // after every prologue load is out -- tested per row as it loaded, the branch made each row's
    // loads wait for the previous row's scales.
    opus::static_for<R>([&](auto rc) {
        if (__builtin_amdgcn_ballot_w32(e8m0x4_has_nan((unsigned)q_sc[rc.value]))) q_nan_rows |= 1u << rc.value;
    });
#endif
    (void)q_nan_rows;
    typename Mma::vtype_b kv_b[NT];
    int kvs[NT / 2];
    // Unrolled by the ring depth so every slot index is a compile-time constant (a register ring
    // cannot be indexed at run time without going to scratch).
    for (int tb = 0; tb < tile_count; tb += STAGES) {
        opus::static_for<STAGES>([&](auto sc) {
            constexpr int S = sc.value;
            const int t = tb + S;
            if (t >= tile_count) return;   // CTA-uniform
            // Free and candidate rows (one-round-ish launches) only: in multi-round ones (full-layer
            // Table / Prefill) the age order is what retires CTAs early for the next round. Remaining
            // tiles against the launch's longest chunk, not this CTA's own: with ragged contexts the
            // split leaves many short tail chunks, which by their own count would start at top
            // priority and starve the long chunks that set the end.
            if constexpr (SCHED == SCHED_FREE || T::GATHER) set_prio_by_progress(tile_count - t, chunk_max);
            // Tile t has landed once at most the tiles issued after it are outstanding. Near the
            // end fewer were issued, and the constant would no longer cover t: drain instead.
            if (tile_count - 1 - t >= LEAD - 1) opus::s_wait_tensorcnt<T::TENSORCNT_KEEP_V41>();
            else                                opus::s_wait_tensorcnt<0>();
            cta_sync();
            issue_tile(t + LEAD, opus::number<(S + LEAD) % STAGES>{}, pf);
            if (t + LEAD + 1 < tile_count) bt_words(t + LEAD + 1, pf);
            read_kv(kv_b, kvs, sc);
            __builtin_amdgcn_sched_barrier(0);

            // Accumulator j of the tile is n-tile j / R of row j % R. Its partial (the lane's head
            // group) waits for the pair partner, n-tile nt ^ 1, and the odd one finishes the pair:
            // row r's keys 32u + lane, in out[r][u].
            float out[R][NT / 2];
            // VGPR asm only under 256 VGPRs: R <= 4 is
            constexpr bool FIN_ASM = R <= 4;
            opus::vector_t<float, 2> part[R][NT];
            auto finish = [&](auto jc, opus::vector_t<float, 2> v) {
                constexpr int j = decltype(jc)::value, nt = j / R, r = j % R;
                part[r][nt] = v;
                if constexpr (nt % 2 == 1)
                    out[r][nt / 2] = pair_head_reduce(pair_sum<FIN_ASM>(part[r][nt - 1]), pair_sum<FIN_ASM>(part[r][nt]));
            };
            constexpr int A = NT * R;
            auto wmma_j = [&](auto jc) {
                constexpr int j = decltype(jc)::value, nt = j / R, r = j % R;
                return mma(q_a[r], kv_b[nt], q_sc[r], kvs[nt / 2], opus::number<0>{}, opus::number<nt % 2>{});
            };
            auto epilogue_j = [&](auto jc, const typename Mma::vtype_c& c) {
                constexpr int r = decltype(jc)::value % R;
                finish(jc, head_sum(c, w[r]));
            };
            // Pipelined by pairs: pair m's two WMMAs, then pair m-1's two epilogues, each step
            // fenced so the WMMAs lead. A relu then never sits right behind the WMMA it reads, and
            // the two epilogues' dependent chains interleave, which one epilogue alone cannot do.
            typename Mma::vtype_c c0 = wmma_j(opus::number<0>{}), c1 = wmma_j(opus::number<1>{});
            opus::static_for<A / 2 - 1>([&](auto mc) {
                constexpr int m = mc.value + 1;
                __builtin_amdgcn_sched_barrier(0);
                typename Mma::vtype_c n0 = wmma_j(opus::number<2 * m>{}), n1 = wmma_j(opus::number<2 * m + 1>{});
                __builtin_amdgcn_sched_barrier(0);
                epilogue_j(opus::number<2 * m - 2>{}, c0);
                epilogue_j(opus::number<2 * m - 1>{}, c1);
                c0 = n0;
                c1 = n1;
            });
            __builtin_amdgcn_sched_barrier(0);
            epilogue_j(opus::number<A - 2>{}, c0);
            epilogue_j(opus::number<A - 1>{}, c1);
            // The stores go after the tile's compute, not as each pair finishes: a store's data
            // registers are the accumulator's, the next WMMA rewrites them, and that waits on the
            // store reading them out (s_wait_xcnt) -- behind the HBM write queue.
            const int tile_off = (chunk_start + t) * TK + lane;
#if MQA_LOGITS_NAN_RESTORE
            {   // NaN back where max_num dropped it: lane L of out[r][u] is key 32u + L, whose scale
                // dword is kvs[u] in the same lane. With few rows the selects are cheaper than a
                // branch -- a uniform branch makes the wave wait for the mask to reach the SALU
                // every tile; with many rows, only a sequence's first compressed row (or a NaN q
                // row) takes the branch.
                bool kff[NT / 2];
                opus::static_for<NT / 2>([&](auto uc) {
                    kff[uc.value] = NAN_ELEM ? false : e8m0x4_has_nan((unsigned)kvs[uc.value]);
                });
                auto restore = [&]() {
                    opus::static_for<R>([&](auto rc) {
                        opus::static_for<NT / 2>([&](auto uc) {
                            const bool nan = kff[uc.value] || ((q_nan_rows >> rc.value) & 1u);
                            out[rc.value][uc.value] = std::bit_cast<float>(
                                nan ? 0x7fc00000u : std::bit_cast<unsigned>(out[rc.value][uc.value]));
                        });
                    });
                };
                if constexpr (NAN_ELEM) {
                    if (q_nan_rows) restore();   // the keys' NaN rides the relu; only a NaN q row is left
                } else if constexpr (R <= 2) {
                    restore();
                } else {
                    unsigned kany = q_nan_rows;
                    opus::static_for<NT / 2>([&](auto uc) { kany |= __builtin_amdgcn_ballot_w32(kff[uc.value]); });
                    if (kany) restore();
                }
            }
#endif
            __builtin_amdgcn_sched_barrier(0);
            opus::static_for<R>([&](auto rc) {
                constexpr int r = rc.value;
                auto g_out = opus::make_gmem(out_ptr[r], out_bytes[r]);   // window-sized V#: masks the tail
                opus::static_for<NT / 2>([&](auto uc) {
                    constexpr int u = uc.value;
                    g_out.template store<1>(out[r][u], tile_off + u * 2 * T::MMA_N);
                });
            });
            __builtin_amdgcn_sched_barrier(0);
        });
    }
}

}  // namespace opus_logits::gfx1250
