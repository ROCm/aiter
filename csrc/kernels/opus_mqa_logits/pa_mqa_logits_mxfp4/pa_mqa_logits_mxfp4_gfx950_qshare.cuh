// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx950 wave64 (MFMA 32x32x64) qshare kernel: QPB query rows of one sequence a CTA, one a wave,
// each wave over the whole 64-key tile, the tile fetched once into LDS and read by all of them.
// A wave computes exactly as the one-row kernel's does. What differs:
//
//   1. Waves split query rows, not KV. Wave w owns row row_id + w; every wave reads the same
//      page, so the block_tables read is CTA-uniform and an s_load.
//   2. The KV tile comes from LDS. The issuing waves (traits) fill a stage with buffer_load_lds,
//      16 B a lane; four ds_read_b128 a lane drain it into the MFMA B operands. The LDS image is
//      the page byte for byte: lane L reads stage + kt * 2048 + nt * 512 + (L / 32) * 1024 +
//      (L % 32) * 16, whose bank unit (addr >> 4) % 16 is L % 16, conflict-free in each of a
//      wave64 ds_read_b128's four 16-lane phases.
//   3. One s_barrier a phase. It publishes the tile the pipe landed and separates the previous
//      phase's ds_read from the copy that recycles its stage (the traits derive the depth).
//
// A group's windows do not decrease along its rows (pa_mqa_logits_mxfp4.h), so the loop runs the
// union -- the last row's window, CTA-uniform -- and each wave masks its stores by its own row's.
// A wave past a short group runs the loop with an empty window and stores nothing. Nothing in the
// loop may depend on the wave index but which copies a wave issues: every wave must take the same
// trips and barriers, or the CTA deadlocks.
#pragma once
#include "pa_mqa_logits_mxfp4_gfx950_common.cuh"

namespace opus_logits::gfx950 {

template<class T, int SCHED>
__global__ __launch_bounds__(T::BLOCK_SIZE, MIN_WAVES_PER_EU)
void qshare_kernel(mqa_logits_mxfp4_kargs k) {
    using D_DATA   = opus::fp4_t;   // MFMA A/B element (E2M1)
    using D_BYTE   = uint8_t;       // fp4 is addressed in bytes
    using D_WEIGHT = typename T::D_WEIGHT;
    using D_SCALE  = typename T::D_SCALE;
    using D_ACC    = typename T::D_ACC;
    using D_OUT    = typename T::D_OUT;

    constexpr int WAVE   = T::WAVE_SIZE;
    constexpr int MT     = T::M_TILES;
    constexpr int NT     = T::N_TILES;
    constexpr int KT     = T::K_TILES;
    constexpr int TK     = T::KV_TILE_SIZE;
    constexpr int EC     = T::C_FRAG;
    constexpr int QPB    = T::Q_PER_BLOCK;
    constexpr int STAGES = T::STAGES;
    constexpr int RLEAD  = T::READ_LEAD;
    constexpr int ALEAD  = T::ASYNC_LEAD;

    __shared__ char smem_buf[T::LDS_BYTES];

    // One scalar round trip, pinned ahead of the early exits.
    const void* p_q        = k.ptr_q;
    const void* p_q_scale  = k.ptr_q_scale;
    const void* p_kv       = k.ptr_kv;
    const void* p_kv_scale = k.ptr_kv_scale;
    const int*  p_bt       = k.ptr_block_tables;
    const void* p_weights  = k.ptr_weights;
    float*      p_out      = k.ptr_out;
    int   stride_out   = k.out_stride;
    float weight_scale = k.weight_scale;
    int   bt_stride    = k.bt_stride;
    mqa_sched_args sa(k);
    pin_sgpr(sa.num_rows);
    pin_sgpr(p_q); pin_sgpr(p_q_scale); pin_sgpr(p_kv); pin_sgpr(p_kv_scale);
    pin_sgpr(p_bt); pin_sgpr(p_weights); pin_sgpr(p_out);
    pin_sgpr(stride_out); pin_sgpr(weight_scale); pin_sgpr(bt_stride);

    const int tid = opus::thread_id_x();
    // readfirstlane, not a plain tid >> 6: the wave index is wave-uniform but divergent to LLVM,
    // and it feeds two things that must be scalar -- the LDS destination of buffer_load_lds (M0)
    // and this wave's row base pointers.
    const int wave    = __builtin_amdgcn_readfirstlane(tid >> 6);
    const int lane    = tid & (WAVE - 1);
    const int lane_m0 = lane & (T::MMA_N - 1);   // n column / m row within the tile
    const int lane_g  = lane >> 5;               // 32-K block within the k-tile

    // The group's union window (wk.local_end) sets the tile range; this wave's own window only
    // masks its stores. Every early return is CTA-uniform.
    mqa_work wk;
    if (!mqa_assign<QPB, TK, SCHED>(sa, lane, wk)) return;
    const int  chunk_start = wk.chunk_start;
    const int  tile_count  = wk.chunk_count;
    const int  batch_id    = wk.batch_id;
    // Clamped, not branched: wave and group_rows are scalar, so the select is s_cselect and the
    // row's reads stay s_load.
    const bool slot_ok     = wave < wk.group_rows;
    const int  row_id      = wk.row_id + (slot_ok ? wave : 0);
    const int  row_end     = sa.row_ends[row_id];
    const int  own_end     = slot_ok ? row_end : 0;

    auto mma = opus::make_tiled_mma<D_DATA, D_DATA, D_ACC>(
        opus::seq<MT, NT, KT>{}, opus::seq<1, 1, 1>{},
        opus::seq<T::MMA_M, T::MMA_N, T::MMA_K>{});
    using Mma = typename decltype(mma)::MMA;
    Mma mma_op;
    static_assert(sizeof(typename Mma::vtype_a) == 2 * T::A_BYTES_PER_LANE,
                  "expected a 256-bit MFMA operand with the fp4 payload in its low half");
    static_assert(Mma::elem_c == EC, "traits C_FRAG must match the MFMA accumulator");

    // The row folded into the base pointers; row_id is wave-uniform.
    const D_BYTE* q_base = reinterpret_cast<const D_BYTE*>(p_q) + (size_t)row_id * T::Q_ROW_BYTES;
    const D_SCALE* qs_base = reinterpret_cast<const D_SCALE*>(p_q_scale) + (size_t)row_id * WAVE;
    const D_WEIGHT* w_base =
        reinterpret_cast<const D_WEIGHT*>(p_weights) + (size_t)row_id * T::W_ROW_ELEMS;
    const int* bt_base = p_bt + (size_t)batch_id * bt_stride;
    // This wave's window, not the union: the out V# spans exactly it, so the hardware's range
    // check is the whole store mask, and an empty window (a wave past the group) stores nothing.
    D_OUT* out_base = p_out + (size_t)row_id * stride_out;
    const unsigned out_bytes = (unsigned)(own_end > 0 ? own_end : 0) * (unsigned)sizeof(D_OUT);

    auto g_q   = opus::make_gmem(q_base);
    auto g_qs  = opus::make_gmem(qs_base);
    auto g_kv  = opus::make_gmem(reinterpret_cast<const D_BYTE*>(p_kv));
    auto g_kvs = opus::make_gmem(reinterpret_cast<const D_SCALE*>(p_kv_scale));
    auto g_w   = opus::make_gmem(w_base);
    auto g_out = opus::make_gmem(out_base, out_bytes);
    auto s_kv  = opus::make_smem(reinterpret_cast<D_BYTE*>(smem_buf));
    // A dword view of the same LDS for the staged scales (offsets in dwords).
    auto s_kvs = opus::make_smem(reinterpret_cast<D_SCALE*>(smem_buf));

    auto u_q = make_layout_q<T>(lane_m0, lane_g);
    typename Mma::vtype_a q_a[MT][KT]{};
    auto u_qs = make_layout_scale<T>(lane);
    int q_scale;
    auto u_w = make_layout_w<T>(lane_g);
    opus::vector_t<float, EC> w_pl[MT];

    // Only lane group 0 stores: after the head reduction lanes L and L^32 hold the same logit,
    // so the partner group is pushed out of bounds -- by a large negative constant, never a
    // multiple of the window, which could land back inside it.
    const int lane_bias = (lane_g != 0) ? -(chunk_start + tile_count) * TK : 0;

    // block_tables through the constant address space, so the page id is an s_load: it is the
    // copy's soffset (buffer_load_lds takes no divergent scalar operand), and on lgkmcnt it is
    // covered by the wait the phase already makes for its ds_reads. A plain dereference lands on
    // flat_load: the loop's stores to `out` defeat the no-clobber proof. chunk_start is folded
    // into the base and the index is unsigned, which keeps the address arithmetic off the SALU.
    using bt_scalar_ptr = const int __attribute__((address_space(4)))*;
    auto bt_c = reinterpret_cast<bt_scalar_ptr>(
        reinterpret_cast<__UINTPTR_TYPE__>(bt_base + chunk_start));
    auto load_page_id = [&](unsigned tile_idx) { return bt_c[tile_idx]; };
    // Unsigned result, signed compare (an unsigned compare costs more SALU in the loop).
    auto clamp_tile = [&](int t) { return (unsigned)(t < tile_count ? t : tile_count - 1); };

    using kv_nt_t = typename Mma::vtype_b;
    using sfrag   = opus::vector_t<float, MT * EC>;

    // The LDS pipe.
    auto stage_byte = [](int tile) { return (tile & (STAGES - 1)) * T::LDS_TILE_BYTES; };
    // Each wave's scale image of a stage, after the data region.
    auto scale_stage_byte = [&](int tile) {
        return T::LDS_DATA_BYTES + (tile & (STAGES - 1)) * T::LDS_SCALE_BYTES +
               wave * T::SCALE_PAGE_BYTES;
    };
    const int async_lane_off = lane * T::ASYNC_VEC_BYTES;
    const int async_wave_off = wave * T::ASYNC_WAVE_BYTES;
    // Fill one stage from one page: the page copy (issuing waves), then this wave's scale copy,
    // adjacent in the vmem queue so one wait retires both.
    auto issue_async = [&](int tile, int page_id) {
        if (T::ISSUE_WAVES == T::NUM_WAVES || wave < T::ISSUE_WAVES) {
            D_BYTE* dst = reinterpret_cast<D_BYTE*>(smem_buf) + stage_byte(tile) + async_wave_off;
            opus::static_for<T::ASYNC_ISSUES>([&](auto ic) {
                constexpr int chunk = ic.value * T::ISSUE_WAVES * T::ASYNC_WAVE_BYTES;
                opus::async_load<T::ASYNC_VEC_BYTES>(
                    g_kv, reinterpret_cast<void*>(dst + chunk), async_lane_off,
                    kv_page_offset<T>(page_id) + async_wave_off + chunk);
            });
        }
        D_BYTE* sdst = reinterpret_cast<D_BYTE*>(smem_buf) + scale_stage_byte(tile);
        opus::async_load<1>(g_kvs, reinterpret_cast<void*>(sdst), lane,
                            page_id * (T::KVS_PAGE_BYTES / (int)sizeof(D_SCALE)));
    };

    // Drain one stage into the four MFMA B operands: one address VGPR, four immediates.
    const int lds_lane_off = lane_g * T::STRIDE_KV_G + lane_m0 * T::STRIDE_KV_M;
    auto read_kv_scale_lds = [&](int tile) {
        return opus::load<1>(s_kvs, scale_stage_byte(tile) / (int)sizeof(D_SCALE) + lane)[0];
    };
    auto read_kv_lds = [&](kv_nt_t (&kv)[NT][KT], int stage_off) {
        opus::static_for<NT>([&](auto ntc) {
            constexpr int nt = ntc.value;
            opus::static_for<KT>([&](auto ktc) {
                constexpr int kt  = ktc.value;
                constexpr int imm = kt * T::STRIDE_KV_KTILE + nt * T::STRIDE_KV_NTILE;
                auto v = opus::load<T::KV_GRP_BYTES>(s_kv, stage_off + lds_lane_off + imm);
                opus::set_slice(kv[nt][kt], v, opus::number<0>{}, opus::number<T::KV_GRP_BYTES>{});
            });
        });
    };

    // Lanes L and L^32 hold the same logit after the head reduction; lane group 1 is biased out of
    // bounds. Fenced so the stores stay out of the MFMA groups' interleave.
    auto store = [&](opus::vector_t<float, NT>& out, int t) {
        __builtin_amdgcn_sched_barrier(0);
        const int tile_off = (chunk_start + t) * TK + lane_bias;
        opus::static_for<NT>([&](auto ntc) {
            constexpr int nt = ntc.value;
            g_out.template store<1>(out[nt], tile_off + nt * T::MMA_N + lane_m0);
        });
        __builtin_amdgcn_sched_barrier(0);
    };

    // Zero-initialized: fp4 fills only the low 16 B of each 256-bit operand.
    kv_nt_t kvA[NT][KT]{}, kvB[NT][KT]{};
    int     kvsA, kvsB;
    sfrag   accA[NT], accB[NT];
    opus::vector_t<float, NT> outA, outB;
    int     pg_a;   // the page id of tile t + ALEAD at entry to phase t

    // The prologue.
    // Q / q_scale / weights go out first, ahead of any copy, which is what makes the waits below
    // exact: vmcnt retires in issue order.
    load_q<T>(g_q, u_q, q_a);
    q_scale = opus::load<1>(g_qs, u_qs)[0];
    load_weights<T>(g_w, u_w, w_pl, weight_scale);

    // Prime the pipe: tiles 0 .. STAGES - 1 into stages 0 .. STAGES - 1, page ids gathered first
    // so the copies issue back to back.
    int pg_pre[STAGES];
    opus::static_for<STAGES>([&](auto sc) {
        pg_pre[sc.value] = load_page_id(clamp_tile(sc.value));
    });
    opus::static_for<STAGES>([&](auto sc) { issue_async(sc.value, pg_pre[sc.value]); });

    // Tile 0 has landed for this wave once its copies have retired; the barrier makes it true
    // for every wave's view of the page.
    opus::s_waitcnt_vmcnt(opus::number<T::PROLOGUE_KEEP0>{});
    cta_sync();
    kvsA = read_kv_scale_lds(0);
    read_kv_lds(kvA, 0);
    // Whole-tile lead: both n-tiles of tile 0 into accA, so phase 0 opens by recycling kvA.
    gemm_ntile<T, 0>(mma_op, q_a, q_scale, accA[0], kvA[0], kvsA);
    gemm_ntile<T, 1>(mma_op, q_a, q_scale, accA[1], kvA[1], kvsA);

    // Stage 0 is free once every wave's tile-0 ds_read has landed; the same barrier publishes
    // tile 1.
    opus::s_waitcnt_lgkmcnt(opus::number<0>{});
    opus::s_waitcnt_vmcnt(opus::number<T::PROLOGUE_KEEP1>{});
    cta_sync();
    issue_async(STAGES, load_page_id(clamp_tile(STAGES)));   // into the stage tile 0 vacated
    kvsB = read_kv_scale_lds(1);
    read_kv_lds(kvB, 1 * T::LDS_TILE_BYTES);
    pg_a = load_page_id(clamp_tile(ALEAD));
    __builtin_amdgcn_sched_barrier(0);

    // One phase = one KV tile. At entry acc_c holds both n-tiles of tile t (made last phase), so
    // kv_c is free and refilled at the top from tile t + RLEAD; kv_p (tile t + 1) feeds the GEMMs.
    auto compute_phase = [&](kv_nt_t (&kv_c)[NT][KT], int& kvs_c, sfrag (&acc_c)[NT],
                             kv_nt_t (&kv_p)[NT][KT], int& kvs_p, sfrag (&acc_p)[NT],
                             int t, opus::vector_t<float, NT>& out, auto keep_vmcnt) {
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
        compute_tile<T>(mma_op, q_a, q_scale, w_pl, acc_c, acc_p, kv_p, kvs_p, out);
    };

    // Every branch below is on tile_count, CTA-uniform: all waves run the same phases and hit the
    // same barriers.
    compute_phase(kvA, kvsA, accA, kvB, kvsB, accB, 0, outA, opus::number<T::KEEP_PEELED>{});
    store(outA, 0);
    if (tile_count >= 2) {
        compute_phase(kvB, kvsB, accB, kvA, kvsA, accA, 1, outB, opus::number<T::KEEP_PEELED>{});
        store(outB, 1);
    }

    int t = 2;
    constexpr auto KEEP = opus::number<T::KEEP_STEADY>{};
    for (; t + 1 < tile_count; t += 2) {
        compute_phase(kvA, kvsA, accA, kvB, kvsB, accB, t, outA, KEEP);
        store(outA, t);
        compute_phase(kvB, kvsB, accB, kvA, kvsA, accA, t + 1, outB, KEEP);
        store(outB, t + 1);
    }
    if (t < tile_count) {
        compute_phase(kvA, kvsA, accA, kvB, kvsB, accB, t, outA, KEEP);
        store(outA, t);
    }
    // The last phases prefetched tiles past the walk and leave them in flight: s_endpgm waits for
    // every counter before the CTA's LDS is released.
}

}  // namespace opus_logits::gfx950
