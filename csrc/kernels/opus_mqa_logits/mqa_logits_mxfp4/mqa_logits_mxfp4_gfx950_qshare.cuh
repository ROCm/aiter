// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx950 wave64 (MFMA 32x32x64) qshare kernel: QPB query rows of one sequence a CTA, one a wave,
// each wave over the whole 64-key tile, the tile fetched once into LDS and read by all of them
// (opus-ops opus_logits/gfx950 FP4_HANDOFF §19, §26). Below the CTA it is the one-row kernel's wave:
// same MFMA shape, tiling (MT = 2, NTPW = 2, KT = 2), whole-tile lead, head reduction and store.
// What is new, and where to look when it misbehaves:
//
//   1. WAVES SPLIT QUERY ROWS, NOT KV. Wave q_slot owns row row_id + q_slot; every wave reads the
//      same page, so the block_tables read is CTA-uniform and an s_load.
//   2. THE KV TILE COMES FROM LDS. The issuing waves (traits) fill a stage with buffer_load_lds,
//      16 B a lane; four ds_read_b128 a lane drain it into the MFMA B operands. The LDS image is
//      the page byte for byte: lane L reads stage + kt*2048 + nt*512 + (L/32)*1024 + (L%32)*16,
//      whose bank unit (addr >> 4) % 16 is L % 16, conflict-free in each of a wave64
//      ds_read_b128's four 16-lane phases. No swizzle or pad can improve that; both can break it.
//   3. ONE s_barrier A PHASE. It publishes the tile the pipe landed AND separates the previous
//      phase's ds_read from the copy that recycles its stage (the traits derive the depth).
//
// The contract (the caller's, mqa_logits_mxfp4.h): a group's windows do not decrease along its
// rows, so the loop runs the union -- the last row's window, CTA-uniform -- and each wave masks
// its stores by its own row's. A slot past a short group runs the loop with an empty window and
// stores nothing. NOTHING IN THE LOOP MAY DEPEND ON q_slot but which copies a wave issues: every
// wave must take the same trips and barriers, or the CTA deadlocks.
#pragma once
#include "mqa_logits_mxfp4_gfx950_common.cuh"

namespace opus_logits::gfx950 {

// The CTA rendezvous, fenced so the scheduler cannot drag LDS traffic across it. A bare s_barrier
// orders execution, not memory: the caller's s_waitcnt gives it something to publish.
__device__ inline void cta_sync() {
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);
}

template<class T, int SCHED>
__global__ __launch_bounds__(T::BLOCK_SIZE, OPUS_LOGITS_MIN_WAVES)
void qshare_kernel(mqa_logits_mxfp4_kargs kargs) {
    using D_DATA   = opus::fp4_t;              // MFMA A/B element (E2M1); opus picks format code 4
    using D_BYTE   = uint8_t;                  // gmem/LDS scalar: fp4 is addressed in bytes
    using D_WEIGHT = typename T::D_WEIGHT;
    using D_SCALE  = typename T::D_SCALE;
    using D_ACC    = typename T::D_ACC;
    using D_OUT    = typename T::D_OUT;

    constexpr int WARP    = T::WARP_SIZE;          // 64
    constexpr int NTPW    = T::N_TILES;            // 2  (a wave covers the WHOLE tile)
    constexpr int MT      = T::M_TILES;            // 2
    constexpr int KT      = T::K_TILES;            // 2  (the outer K loop)
    constexpr int QPB     = T::Q_PER_BLOCK;        // query rows == waves a CTA
    constexpr int STAGES  = T::LDS_STAGES;
    constexpr int RLEAD   = T::READ_LEAD;          // 2
    constexpr int ALEAD   = T::ASYNC_LEAD;         // STAGES + RLEAD - 1
    constexpr int block_k = T::KV_TILE_SIZE;       // 64: one page
    constexpr int MFMA_VALU_COEXEC = 8;

    __shared__ char smem_buf[T::LDS_BYTES];

    // ---- kernel arguments: one scalar round trip ----
    const void* p_q        = kargs.ptr_q;
    const void* p_q_scale  = kargs.ptr_q_scale;
    const void* p_kv       = kargs.ptr_kv;
    const void* p_kv_scale = kargs.ptr_kv_scale;
    const int*  p_bt       = kargs.ptr_block_tables;
    const void* p_weights  = kargs.ptr_weights;
    float*      p_out      = kargs.ptr_out;
    int   stride_out   = kargs.out_stride;
    float weight_scale = kargs.weight_scale;
    int   max_blk      = kargs.bt_stride;
    mqa_sched_args sa(kargs);
    pin_sgpr(sa.num_rows);
    pin_sgpr(p_q); pin_sgpr(p_q_scale); pin_sgpr(p_kv); pin_sgpr(p_kv_scale);
    pin_sgpr(p_bt); pin_sgpr(p_weights); pin_sgpr(p_out);
    pin_sgpr(stride_out); pin_sgpr(weight_scale); pin_sgpr(max_blk);

    const int tid = opus::thread_id_x();
    // readfirstlane, not a plain tid >> 6: the wave index is wave-uniform in fact but divergent to
    // LLVM, and it feeds two things that must be scalar -- the LDS destination of buffer_load_lds
    // (M0) and this wave's row base pointers.
    const int q_slot      = __builtin_amdgcn_readfirstlane(tid >> 6);
    const int lane_id     = tid & (WARP - 1);
    const int lane_mod_32 = lane_id & (T::MFMA_N - 1);   // N col / M row within the tile
    const int lane_div_32 = lane_id >> 5;                // 32-K block within the k-tile

    // ---- per-CTA assignment ----
    // The group's union window (wk.local_end) only fed the tile range; this wave's own window
    // feeds nothing but its store mask. Every early return is CTA-uniform.
    mqa_work wk;
    if (!mqa_assign<QPB, block_k, SCHED>(sa, lane_id, wk)) return;
    const int  chunk_start = wk.chunk_start;
    const int  tile_count  = wk.chunk_count;
    const int  batch_id    = wk.batch_id;
    // Clamped, not branched: q_slot and group_rows are scalar, so the select is s_cselect and the
    // row's reads stay s_load.
    const bool slot_ok     = q_slot < wk.group_rows;
    const int  row_id      = wk.row_id + (slot_ok ? q_slot : 0);
    const int  row_end     = sa.row_ends[row_id];
    const int  own_end     = slot_ok ? row_end : 0;

    // ---- GEMM object (scaled fp4 32x32x64); waves do not split the tile ----
    auto mma = opus::make_tiled_mma<D_DATA, D_DATA, D_ACC>(
        opus::seq<MT, NTPW, KT>{},
        opus::seq<1, T::NUM_WARPS, 1>{},
        opus::seq<T::MFMA_M, T::MFMA_N, T::MFMA_K>{});
    using Mma = typename decltype(mma)::MMA;
    Mma mma_op;
    static_assert(sizeof(typename Mma::vtype_a) == 2 * T::A_BYTES_PER_LANE,
                  "expected a 256-bit MFMA operand with the fp4 payload in its low half");
    static_assert(Mma::elem_c == T::C_FRAG, "traits C_FRAG must match the MFMA accumulator");

    // ---- buffer resources (row folded into base pointers; row_id is wave-uniform) ----
    const D_BYTE*   q_base  = reinterpret_cast<const D_BYTE*>(p_q) + (size_t)row_id * T::Q_ROW_BYTES;
    const D_SCALE*  qs_base = reinterpret_cast<const D_SCALE*>(p_q_scale) + (size_t)row_id * WARP;
    const D_WEIGHT* w_base  = reinterpret_cast<const D_WEIGHT*>(p_weights) + (size_t)row_id * T::W_ROW_ELEMS;
    const int*      bt_base = p_bt + (size_t)batch_id * max_blk;
    // This wave's window, not the union: the out V# spans exactly it, so the hardware's range
    // check is the whole store mask, and an empty window (a slot past the group) stores nothing.
    D_OUT*          out_base  = p_out + (size_t)row_id * stride_out;
    const unsigned  out_bytes = (unsigned)(own_end > 0 ? own_end : 0) * (unsigned)sizeof(D_OUT);

    auto g_q   = opus::make_gmem(q_base);
    auto g_qs  = opus::make_gmem(qs_base);
    auto g_kv  = opus::make_gmem(reinterpret_cast<const D_BYTE*>(p_kv));
    auto g_kvs = opus::make_gmem(reinterpret_cast<const D_SCALE*>(p_kv_scale));
    auto g_w   = opus::make_gmem(w_base);
    auto g_out = opus::make_gmem(out_base, out_bytes);
    auto s_kv  = opus::make_smem(reinterpret_cast<D_BYTE*>(smem_buf));
    // A dword view of the same LDS for the staged scales (offsets in dwords).
    auto s_kvs = opus::make_smem(reinterpret_cast<D_SCALE*>(smem_buf));

    auto u_q = make_layout_q<T>(lane_mod_32, lane_div_32);
    typename Mma::vtype_a q_a[MT][KT]{};
    auto u_qs = make_layout_scale<T>(lane_id);
    int q_scale;
    auto u_w = make_layout_w<T>(lane_div_32);
    opus::vector_t<float, T::C_FRAG> w_pl[MT];

    // Only lane group 0 stores: after the head reduction lanes L and L^32 hold the same logit,
    // so the partner group is pushed out of bounds -- by a large negative constant, never a
    // multiple of the window, which could land back inside it.
    const int lane_bias = (lane_div_32 != 0) ? -(chunk_start + tile_count) * block_k : 0;

    // block_tables through the CONSTANT address space, so the page id is an s_load: it is the
    // copy's soffset (buffer_load_lds takes no divergent scalar operand: anything else wraps
    // every copy in a readfirstlane waterfall), and on lgkmcnt it is covered by the wait the
    // phase already makes for its ds_reads (a vmem page id would drain the pipe every phase).
    // A plain dereference lands on flat_load: the loop's stores to `out` defeat the no-clobber
    // proof. chunk_start is folded into the base and the index is unsigned, which keeps the
    // address arithmetic off the SALU.
    using bt_scalar_ptr = const int __attribute__((address_space(4)))*;
    auto bt_c = reinterpret_cast<bt_scalar_ptr>(
        reinterpret_cast<__UINTPTR_TYPE__>(bt_base + chunk_start));
    auto load_page_id = [&](unsigned tile_idx) { return bt_c[tile_idx]; };
    // Unsigned result, signed compare (an unsigned compare costs 8 more SALU in the loop).
    auto clamp_tile = [&](int t) { return (unsigned)(t < tile_count ? t : tile_count - 1); };

    constexpr int EC = Mma::elem_c;   // 16: per-(mi,nt) C fragment length
    using kv_nt_t = typename Mma::vtype_b;

    // ── the LDS pipe ──
    auto stage_byte = [](int tile) { return (tile & (STAGES - 1)) * T::LDS_TILE_BYTES; };
    // Each wave's scale image of a stage, after the data region.
    auto scale_stage_byte = [&](int tile) {
        return T::LDS_DATA_BYTES + (tile & (STAGES - 1)) * T::LDS_SCALE_BYTES
             + q_slot * T::SCALE_PAGE_BYTES;
    };
    const int async_lane_off = lane_id * T::ASYNC_VEC_BYTES;          // gmem, a lane
    const int async_wave_off = q_slot * T::ASYNC_WAVE_BYTES;          // 1024 B a wave
    // Fill one stage from one page: the page copy (issuing waves), then this wave's scale copy,
    // adjacent in the vmem queue so one wait retires both.
    auto issue_async = [&](int tile, int page_id) {
        if (T::ISSUE_WAVES == T::NUM_WAVES || q_slot < T::ISSUE_WAVES) {
            D_BYTE* dst = reinterpret_cast<D_BYTE*>(smem_buf) + stage_byte(tile) + async_wave_off;
            opus::static_for<T::ASYNC_ISSUES>([&](auto ic) {
                constexpr int chunk = ic.value * T::ISSUE_WAVES * T::ASYNC_WAVE_BYTES;
                opus::async_load<T::ASYNC_VEC_BYTES>(
                    g_kv, reinterpret_cast<void*>(dst + chunk), async_lane_off,
                    page_id * T::stride_kv_block + async_wave_off + chunk);
            });
        }
        D_BYTE* sdst = reinterpret_cast<D_BYTE*>(smem_buf) + scale_stage_byte(tile);
        opus::async_load<1>(g_kvs, reinterpret_cast<void*>(sdst), lane_id,
                            page_id * (T::stride_kvs_block / (int)sizeof(D_SCALE)));
    };

    // Drain one stage into the four MFMA B operands: one address VGPR, four immediates.
    const int lds_lane_off = lane_div_32 * T::stride_kv_g + lane_mod_32 * T::stride_kv_m;
    auto read_kv_scale_lds = [&](int tile) {
        return opus::load<1>(s_kvs, scale_stage_byte(tile) / (int)sizeof(D_SCALE) + lane_id)[0];
    };
    auto read_kv_lds = [&](kv_nt_t (&kv)[NTPW][KT], int stage_off) {
        opus::static_for<NTPW>([&](auto ntc) {
            constexpr int nt = ntc.value;
            opus::static_for<KT>([&](auto ktc) {
                constexpr int kt = ktc.value;
                constexpr int imm = kt * T::stride_kv_ktile + nt * T::stride_kv_ntile;
                auto v = opus::load<T::KV_GRP_BYTES>(s_kv, stage_off + lds_lane_off + imm);
                opus::set_slice(kv[nt][kt], v, opus::number<0>{}, opus::number<T::KV_GRP_BYTES>{});
            });
        });
    };

    using sfrag = opus::vector_t<float, MT * EC>;
    auto gemm_nt = [&](sfrag& accs, kv_nt_t (&kvnt)[KT], int kvs_w, auto ntc) {
        constexpr int nt = ntc.value;
        opus::static_for<MT>([&](auto mic) {
            constexpr int mi = mic.value;
            typename Mma::vtype_c acc{};
            opus::static_for<KT>([&](auto ktc) {
                constexpr int kt = ktc.value;
                acc = mma_op(q_a[mi][kt], kvnt[kt], acc, q_scale, kvs_w,
                             opus::number<kt * MT + mi>{}, opus::number<kt * NTPW + nt>{});
            });
            opus::set_slice(accs, acc, opus::number<mi * EC>{}, opus::number<mi * EC + EC>{});
        });
    };
    auto relu_nt = [&](sfrag& accs) {
        opus::static_for<MT * EC>([&](auto ic) {
            constexpr int i = ic.value;
            accs[i] = MQA_LOGITS_GFX950_RELU(accs[i]);
        });
    };
    auto reduce_all = [&](sfrag& s0, sfrag& s1, opus::vector_t<float, NTPW>& out) {
        head_reduce<T>(w_pl, s0, s1, out);
    };
    auto do_store = [&](opus::vector_t<float, NTPW>& out, int t) {
        const int tile_off = (chunk_start + t) * block_k + lane_bias;
        opus::static_for<NTPW>([&](auto ntc) {
            constexpr int nt = ntc.value;
            g_out.template store<1>(out[nt], tile_off + nt * T::MFMA_N + lane_mod_32);
        });
    };
    auto fenced_store = [&](opus::vector_t<float, NTPW>& out_v, int t) {
        __builtin_amdgcn_sched_barrier(0);
        do_store(out_v, t);
        __builtin_amdgcn_sched_barrier(0);
    };

    // Zero-initialized: fp4 fills only the low 16 B of each 256-bit operand.
    kv_nt_t kvA[NTPW][KT]{}, kvB[NTPW][KT]{};
    int     kvsA, kvsB;
    sfrag   accA[NTPW], accB[NTPW];   // both token-tile fragments per buffer (the whole-tile lead)
    opus::vector_t<float, NTPW> outA, outB;
    int     pg_a;                     // the page id of tile t + ALEAD at entry to phase t

    // ── prologue ──
    // Q / q_scale / weights go out FIRST, ahead of any copy, which is what makes the waits below
    // exact: vmcnt retires in issue order.
    load_q<T>(g_q, u_q, q_a);
    q_scale = opus::load<1>(g_qs, u_qs)[0];
    load_weights<T>(g_w, u_w, w_pl, weight_scale);

    // Prime the pipe: tiles 0..STAGES-1 into stages 0..STAGES-1, page ids gathered first so the
    // copies issue back to back.
    int pg_pre[STAGES];
    opus::static_for<STAGES>([&](auto sc) { pg_pre[sc.value] = load_page_id(clamp_tile(sc.value)); });
    opus::static_for<STAGES>([&](auto sc) { issue_async(sc.value, pg_pre[sc.value]); });

    // Tile 0 has landed for this wave once its copies have retired; the barrier makes it true
    // for every wave's view of the page.
    opus::s_waitcnt_vmcnt(opus::number<T::PROLOGUE_KEEP0>{});
    cta_sync();
    kvsA = read_kv_scale_lds(0);
    read_kv_lds(kvA, 0);
    // Whole-tile lead: both token-tile fragments of tile 0 into accA, so phase 0 opens by
    // recycling kvA.
    gemm_nt(accA[0], kvA[0], kvsA, opus::number<0>{});
    gemm_nt(accA[1], kvA[1], kvsA, opus::number<1>{});

    // Stage 0 is free once every wave's tile-0 ds_read has landed; the same barrier publishes
    // tile 1.
    opus::s_waitcnt_lgkmcnt(opus::number<0>{});
    opus::s_waitcnt_vmcnt(opus::number<T::PROLOGUE_KEEP1>{});
    cta_sync();
    issue_async(STAGES, load_page_id(clamp_tile(STAGES)));   // -> the stage tile 0 vacated
    kvsB = read_kv_scale_lds(1);
    read_kv_lds(kvB, 1 * T::LDS_TILE_BYTES);
    pg_a = load_page_id(clamp_tile(ALEAD));
    __builtin_amdgcn_sched_barrier(0);

    // ── one phase = one KV tile ──
    // At entry acc_c holds both token-tile fragments of tile t (made last phase), so kv_c is free
    // and refilled at the top from tile t + RLEAD; kv_p (tile t + 1) feeds the gemms for the next.
    auto compute_phase = [&](kv_nt_t (&kv_c)[NTPW][KT], int& kvs_c, sfrag (&acc_c)[NTPW],
                             kv_nt_t (&kv_p)[NTPW][KT], int& kvs_p, sfrag (&acc_p)[NTPW],
                             int t, opus::vector_t<float, NTPW>& out_cur, auto keep_vmcnt) {
        __builtin_amdgcn_sched_barrier(0);
        // lgkmcnt: the previous phase's ds_reads must be off the LDS before the copy below
        // recycles their stage. vmcnt: tile t + RLEAD must be resident before anyone reads it.
        opus::s_waitcnt_lgkmcnt(opus::number<0>{});
        opus::s_waitcnt_vmcnt(keep_vmcnt);
        cta_sync();

        issue_async(t + ALEAD, pg_a);
        kvs_c = read_kv_scale_lds(t + RLEAD);
        read_kv_lds(kv_c, stage_byte(t + RLEAD));
        pg_a = load_page_id(clamp_tile(t + ALEAD + 1));
        __builtin_amdgcn_sched_barrier(0);

        gemm_nt(acc_p[0], kv_p[0], kvs_p, opus::number<0>{});
        relu_nt(acc_c[0]);
        sched_mfma_pairs<MT * KT, VALU_MASK, MFMA_VALU_COEXEC, 1>();
        __builtin_amdgcn_sched_barrier(0);
        gemm_nt(acc_p[1], kv_p[1], kvs_p, opus::number<1>{});
        relu_nt(acc_c[1]);
        sched_mfma_pairs<MT * KT, VALU_MASK, MFMA_VALU_COEXEC, 2>();
        __builtin_amdgcn_sched_barrier(0);
        reduce_all(acc_c[0], acc_c[1], out_cur);
    };

    // Every branch below is on tile_count, CTA-uniform: all waves run the same phases and hit the
    // same barriers. Do not make any of it depend on q_slot.
    compute_phase(kvA, kvsA, accA, kvB, kvsB, accB, 0, outA, opus::number<T::KEEP_PEELED>{});
    fenced_store(outA, 0);
    if (tile_count >= 2) {
        compute_phase(kvB, kvsB, accB, kvA, kvsA, accA, 1, outB, opus::number<T::KEEP_PEELED>{});
        fenced_store(outB, 1);
    }

    int t = 2;
    for (; t + 1 < tile_count; t += 2) {
        compute_phase(kvA, kvsA, accA, kvB, kvsB, accB, t, outA, opus::number<T::KEEP_STEADY>{});
        fenced_store(outA, t);
        compute_phase(kvB, kvsB, accB, kvA, kvsA, accA, t + 1, outB, opus::number<T::KEEP_STEADY>{});
        fenced_store(outB, t + 1);
    }
    if (t < tile_count) {   // the last tile
        compute_phase(kvA, kvsA, accA, kvB, kvsB, accB, t, outA, opus::number<T::KEEP_STEADY>{});
        fenced_store(outA, t);
    }
    // The last phases prefetched tiles past the walk and leave them in flight: s_endpgm waits for
    // every counter (CDNA4 ISA, S_ENDPGM) before the CTA's LDS is released.
}

}  // namespace opus_logits::gfx950
