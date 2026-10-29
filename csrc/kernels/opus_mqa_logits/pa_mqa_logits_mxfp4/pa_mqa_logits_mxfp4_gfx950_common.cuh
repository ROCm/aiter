// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// What gfx950's one-row and qshare kernels share: where a CTA's work comes from and the tile math
// (the Q / scale / weight layouts and loads, the tile's GEMM, relu and head reduction, the
// scheduling hints). Device pass on gfx950 only.
#pragma once
#include <opus/opus.hpp>
#include <bit>   // std::bit_cast, required by permlane_head_reduce2

#include "pa_mqa_logits_mxfp4_common.cuh"

namespace opus_logits::gfx950 {

// -- where a CTA's work comes from ----------------------------------------------------------------
// The kernels' template SCHED: gfx1250's mqa_assign (pa_mqa_logits_mxfp4_gfx1250_common.cuh) on
// wave64.
//
//   SCHED_TABLE    a record a CTA slot (the builder's, pa_mqa_logits_mxfp4.cu);
//   SCHED_FREE     no records: every CTA works out its own slot of the same kind of split the
//                  builder emits; every CTA reads every sequence's bounds (decode sizes);
//   SCHED_PREFILL  no records: a group is cut into a static `prefill_parts` pieces and
//                  slot / parts is the group.
// A group is up to QPB consecutive rows of one sequence, and its window is its LAST row's
// (windows do not decrease along a sequence). Every CTA computes each of its tiles the same way
// whatever cut it, so the three modes write bit-identical logits.

// What mqa_assign reads of the kargs. A kernel loads it with the rest of its arguments, ahead of
// its pins: read after them, these go out one dependent scalar round later, and the record a
// round after that.
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
// whose source is out of reach adds 0.
__device__ inline int wave64_incl_scan(int v) {
    v += __builtin_amdgcn_update_dpp(0, v, 0x111, 0xF, 0xF, false);
    v += __builtin_amdgcn_update_dpp(0, v, 0x112, 0xF, 0xF, false);
    v += __builtin_amdgcn_update_dpp(0, v, 0x114, 0xF, 0xF, false);
    v += __builtin_amdgcn_update_dpp(0, v, 0x118, 0xF, 0xF, false);
    v += __builtin_amdgcn_update_dpp(0, v, 0x142, 0xA, 0xF, false);
    v += __builtin_amdgcn_update_dpp(0, v, 0x143, 0xC, 0xF, false);
    return v;
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
                    const int sr = __builtin_amdgcn_readlane(r0, l);
                    const int sn = __builtin_amdgcn_readlane(n, l);
                    const int j0 = gs - carry - __builtin_amdgcn_readlane(excl, l);
                    const int j  = groups_of(sn) - 1 - j0;
                    const int e  = __builtin_amdgcn_readfirstlane(k.row_ends[last_row(sr, sn, j)]);
                    const int tiles = tiles_of(e);
                    const int start = part_point(part, tiles, parts);
                    const int count = part_point(part + 1, tiles, parts) - start;
                    if (count <= 0) return false;
                    set(base + l, sr, sn, j, start, count, e);
                    return true;
                }
                carry += __builtin_amdgcn_readlane(incl, WAVE - 1);
            }
            return false;
        } else {   // SCHED_FREE
            // The first CHUNKS * 64 sequences are read once, up front: their bounds, then the
            // window ends of their first GC groups -- a decode bucket's (max_qlen <= 8) at the qlen
            // each instance is picked for -- so the whole assignment is two round trips. Sequences
            // past the cache, and groups past GC, are re-read where they are needed. Every load of
            // a round is issued before any is used -- indices clamped into range, the lanes past
            // the batch masked after -- or the rounds go serial; the cached chunks are walked
            // unrolled, not indexed through a select chain. The one-row kernel runs MTP 1 only, so
            // it caches two groups: every cached group costs each CTA its loads and cuts.
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
                opus::static_for<GC>([&](auto jc) {
                    e[jc.value] = n > jc.value * QPB ? v[jc.value] : 0;
                });
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
                    if (!done && c * WAVE < num_seqs)
                        done = visit(c * WAVE, c_r0[c], c_n[c], c_e[c]);
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
                opus::static_for<GC>([&](auto jc) {
                    s += ceil_div_rcp(tiles_of(e[jc.value]), safe, rcp);
                });
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
                const int sr = __builtin_amdgcn_readlane(r0, l);
                const int sn = __builtin_amdgcn_readlane(n, l);
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

// -- the tile math --------------------------------------------------------------------------------

// __launch_bounds__'s minimum waves a SIMD: raising it only constrains the register allocator.
constexpr int MIN_WAVES_PER_EU = 2;

// VALU ops hinted into each MFMA's shadow.
constexpr int MFMA_VALU_COEXEC = 8;

// IEEE-754-2019 `maximum`, so a NaN score (an E8M0 0xFF) propagates instead of being selected
// away as 0. Needs -fno-finite-math-only.
__device__ inline float relu(float x) { return __builtin_elementwise_maximum(x, 0.0f); }

// Scheduling hints.
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
// head(mi * MMA_M + L % 32) * 64 + (kt * K_CHUNKS + L / 32) * 16. Un-coalesced: Q is loaded once
// in the prologue, so the layout stays natural.
template<class T>
__device__ inline auto make_layout_q(int lane_m0, int lane_g) {
    constexpr int Q_ROW_NAT = T::HEAD_DIM * T::ELEM_BITS / 8;   // 64 B per natural head row
    constexpr auto shape = opus::make_tuple(
        opus::number<T::M_TILES>{},        // m-tile           (y, outer)
        opus::number<T::K_TILES>{},        // k-tile           (y, outer)
        opus::number<T::MMA_M>{},          // head row in tile (p: lane % 32)
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
            opus::number<T::MMA_M * Q_ROW_NAT>{},           // m-tile: 32 heads * 64 B
            opus::number<T::K_CHUNKS * T::KV_GRP_BYTES>{},  // k-tile: 2 blocks * 16 B
            opus::number<Q_ROW_NAT>{},                      // head row: 64 B
            opus::number<1>{})),                            // (g, byte) base=1 -> g=16, byte=1
        opus::unfold_p_coord(dim, opus::make_tuple(lane_m0, lane_g)));
}

// q_scale / kv_scale: one dword at index == lane.
template<class T>
__device__ inline auto make_layout_scale(int lane) {
    constexpr auto shape = opus::make_tuple(
        opus::number<T::WAVE_SIZE>{},   // lane         (p)
        opus::number<1>{});             // single dword (y)
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));
    return opus::make_layout(
        shape,
        opus::unfold_x_stride(dim, shape, opus::make_tuple(
            opus::number<1>{},
            opus::number<0>{})),
        opus::unfold_p_coord(dim, opus::make_tuple(lane)));
}

// Weights: natural [T, H] bf16. Lane group lg = L / 32 gathers C-fragment heads at
// lg * PACK_C + mi * MMA_M + rept * (PACK_C * GRPM_C) + pack: MT * REPT_C loads of PACK_C bf16.
template<class T>
__device__ inline auto make_layout_w(int lane_g) {
    constexpr auto shape = opus::make_tuple(
        opus::number<T::GRPM_C>{},     // lane group lg   (p)
        opus::number<T::M_TILES>{},    // m-tile          (y, outer)
        opus::number<T::REPT_C>{},     // rept            (y, outer)
        opus::number<T::PACK_C>{});    // pack            (y, contiguous)
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}));
    return opus::make_layout(
        shape,
        opus::unfold_x_stride(dim, shape, opus::make_tuple(
            opus::number<T::PACK_C>{},              // lg:   4
            opus::number<T::MMA_M>{},               // mi:   32 heads
            opus::number<T::PACK_C * T::GRPM_C>{},  // rept: 8
            opus::number<1>{})),                    // pack: 1
        opus::unfold_p_coord(dim, opus::make_tuple(lane_g)));
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

// One n-tile of a tile: MT m-tiles, each accumulated over the KT outer K loop. op_sel picks the
// scale byte for (kt, mi) on A and (kt, nt) on B out of the single dword each lane holds.
template<class T, int NT_IDX, class Mma, class A, class B, class S>
__device__ inline void gemm_ntile(Mma& mma, const A (&q_a)[T::M_TILES][T::K_TILES], int q_scale,
                                  S& accs, const B (&kv)[T::K_TILES], int kv_scale) {
    constexpr int MT = T::M_TILES, NT = T::N_TILES, KT = T::K_TILES, EC = T::C_FRAG;
    opus::static_for<MT>([&](auto mic) {
        constexpr int mi = mic.value;
        typename Mma::vtype_c acc{};
        opus::static_for<KT>([&](auto ktc) {
            constexpr int kt = ktc.value;
            acc = mma(q_a[mi][kt], kv[kt], acc, q_scale, kv_scale,
                      opus::number<kt * MT + mi>{}, opus::number<kt * NT + NT_IDX>{});
        });
        opus::set_slice(accs, acc, opus::number<mi * EC>{}, opus::number<mi * EC + EC>{});
    });
}

template<class T, class S>
__device__ inline void relu_ntile(S& accs) {
    opus::static_for<T::M_TILES * T::C_FRAG>([&](auto ic) {
        constexpr int i = ic.value;
        accs[i] = relu(accs[i]);
    });
}

// Weighted head sum of a tile's two n-tiles, packed two heads at a time, then the cross-lane-
// group finish: out[nt] is key nt * 32 + L % 32's logit in both lane groups.
template<class T, class S>
__device__ inline void head_reduce(const opus::vector_t<float, T::C_FRAG> (&w_pl)[T::M_TILES],
                                   S& s0, S& s1, opus::vector_t<float, T::N_TILES>& out) {
    constexpr int NT = T::N_TILES, EC = T::C_FRAG;
    S* sp[NT] = {&s0, &s1};
    opus::vector_t<float, 2> acc[NT];
    opus::static_for<NT>([&](auto ntc) { acc[ntc.value] = opus::vector_t<float, 2>{0.0f, 0.0f}; });
    opus::static_for<T::M_TILES>([&](auto mic) {
        constexpr int mi = mic.value;
        opus::static_for<EC / 2>([&](auto pc) {
            constexpr int p = pc.value;
            opus::vector_t<float, 2> w{w_pl[mi][2 * p], w_pl[mi][2 * p + 1]};
            opus::static_for<NT>([&](auto ntc) {
                constexpr int nt = ntc.value;
                S& sn = *sp[nt];
                opus::vector_t<float, 2> s{sn[mi * EC + 2 * p], sn[mi * EC + 2 * p + 1]};
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

// One phase's MFMA work: tile t + 1's GEMMs (kv_p into acc_p) interleaved with tile t's relu
// (acc_c), then tile t's head reduction into out.
template<class T, class Mma, class A, class B, class S>
__device__ inline void compute_tile(Mma& mma, const A (&q_a)[T::M_TILES][T::K_TILES], int q_scale,
                                    const opus::vector_t<float, T::C_FRAG> (&w_pl)[T::M_TILES],
                                    S (&acc_c)[T::N_TILES], S (&acc_p)[T::N_TILES],
                                    const B (&kv_p)[T::N_TILES][T::K_TILES], int kvs_p,
                                    opus::vector_t<float, T::N_TILES>& out) {
    constexpr int MMA_PAIRS = T::M_TILES * T::K_TILES;
    gemm_ntile<T, 0>(mma, q_a, q_scale, acc_p[0], kv_p[0], kvs_p);
    relu_ntile<T>(acc_c[0]);
    sched_mfma_pairs<MMA_PAIRS, VALU_MASK, MFMA_VALU_COEXEC, 1>();
    __builtin_amdgcn_sched_barrier(0);
    gemm_ntile<T, 1>(mma, q_a, q_scale, acc_p[1], kv_p[1], kvs_p);
    relu_ntile<T>(acc_c[1]);
    sched_mfma_pairs<MMA_PAIRS, VALU_MASK, MFMA_VALU_COEXEC, 2>();
    __builtin_amdgcn_sched_barrier(0);
    head_reduce<T>(w_pl, acc_c[0], acc_c[1], out);
}

}  // namespace opus_logits::gfx950
