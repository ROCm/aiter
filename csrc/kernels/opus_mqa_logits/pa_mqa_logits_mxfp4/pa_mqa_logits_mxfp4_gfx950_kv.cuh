// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx950 wave64 (MFMA 32x32x64) one-row kernel: one query row a CTA, one wave over each 64-key
// tile, the KV read straight from global into a register double buffer. Device pass on gfx950
// only (pa_mqa_logits_mxfp4_gfx950.cu).
#pragma once
#include "pa_mqa_logits_mxfp4_gfx950_common.cuh"

namespace opus_logits::gfx950 {

// KV partition, byte strides. kv_cache is [pages][kt][g][nt][m][16 B]; lane L reads
// K[(kt * K_CHUNKS + g) * 32 : +32] of key nt * MMA_N + m, g = L / 32, m = L % 32.
template<class T>
__device__ inline auto make_layout_kv(int lane_m0, int lane_g) {
    constexpr auto shape = opus::make_tuple(
        opus::number<T::MMA_N>{},          // key m in tile (p: lane % 32)
        opus::number<T::K_CHUNKS>{},       // 32-K block g  (p: lane / 32)
        opus::number<T::KV_GRP_BYTES>{});  // 16 B = 32 fp4 (y, contiguous)
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));
    return opus::make_layout(
        shape,
        opus::unfold_x_stride(dim, shape, opus::make_tuple(
            opus::number<T::STRIDE_KV_M>{},   // key: consecutive keys are 16 B apart
            opus::number<T::STRIDE_KV_G>{},   // block g: one 32-K block across the page
            opus::number<1>{})),              // bytes
        opus::unfold_p_coord(dim, opus::make_tuple(lane_m0, lane_g)));
}

// block_tables: entry wave + i, one int a wave. The wave index is the layout's p coordinate, so
// the page-id address is computed per lane (VALU), off the SALU chain the prologue waits on.
template<class T>
__device__ inline auto make_layout_bt(int wave) {
    constexpr auto shape = opus::make_tuple(
        opus::number<T::NUM_WAVES>{},  // wave   (p)
        opus::number<1>{});            // one int (y)
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));
    return opus::make_layout(
        shape,
        opus::unfold_x_stride(dim, shape, opus::make_tuple(
            opus::number<1>{},
            opus::number<0>{})),
        opus::unfold_p_coord(dim, opus::make_tuple(wave)));
}

// Grid: Table one CTA a record; Free groups + cta_target; Prefill groups x prefill_parts
// (pa_mqa_logits_mxfp4.py sizes them). A group is one row here.
template<class T, int SCHED>
__global__ __launch_bounds__(T::BLOCK_SIZE, MIN_WAVES_PER_EU)
void kv_kernel(mqa_logits_mxfp4_kargs k) {
    using D_DATA   = opus::fp4_t;   // MFMA A/B element (E2M1)
    using D_BYTE   = uint8_t;       // fp4 is addressed in bytes
    using D_WEIGHT = typename T::D_WEIGHT;
    using D_SCALE  = typename T::D_SCALE;
    using D_ACC    = typename T::D_ACC;
    using D_OUT    = typename T::D_OUT;

    constexpr int WAVE = T::WAVE_SIZE;
    constexpr int MT   = T::M_TILES;
    constexpr int NT   = T::N_TILES;
    constexpr int KT   = T::K_TILES;
    constexpr int TK   = T::KV_TILE_SIZE;
    constexpr int EC   = T::C_FRAG;

    // Read and pinned ahead of the early exits, so the loads stay one scalar round trip and
    // cannot sink past them.
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

    const int tid     = opus::thread_id_x();
    const int wave    = tid >> 6;
    const int lane    = tid & (WAVE - 1);
    const int lane_m0 = lane & (T::MMA_N - 1);   // n column / m row within the tile
    const int lane_g  = lane >> 5;               // 32-K block within the k-tile

    // CTA-uniform early return. The window is a prefix [0, local_end).
    mqa_work wk;
    if (!mqa_assign<1, TK, SCHED>(sa, lane, wk)) return;
    const int row_id      = wk.row_id;
    const int batch_id    = wk.batch_id;
    const int chunk_start = wk.chunk_start;
    const int tile_count  = wk.chunk_count;
    const int local_end   = wk.local_end;

    auto mma = opus::make_tiled_mma<D_DATA, D_DATA, D_ACC>(
        opus::seq<MT, NT, KT>{}, opus::seq<1, 1, 1>{},
        opus::seq<T::MMA_M, T::MMA_N, T::MMA_K>{});
    using Mma = typename decltype(mma)::MMA;
    Mma mma_op;
    static_assert(sizeof(typename Mma::vtype_a) == 2 * T::A_BYTES_PER_LANE,
                  "expected a 256-bit MFMA operand with the fp4 payload in its low half");
    static_assert(Mma::elem_c == EC, "traits C_FRAG must match the MFMA accumulator");

    // The row folded into the base pointers to keep voffsets small.
    const D_BYTE* q_base = reinterpret_cast<const D_BYTE*>(p_q) + (size_t)row_id * T::Q_ROW_BYTES;
    const D_SCALE* qs_base = reinterpret_cast<const D_SCALE*>(p_q_scale) + (size_t)row_id * WAVE;
    const D_WEIGHT* w_base =
        reinterpret_cast<const D_WEIGHT*>(p_weights) + (size_t)row_id * T::W_ROW_ELEMS;
    const int* bt_base = p_bt + (size_t)batch_id * bt_stride;
    // The out V# spans exactly the window: the hardware's range check is the whole store mask.
    D_OUT* out_base = p_out + (size_t)row_id * stride_out;
    const unsigned out_bytes = (unsigned)(local_end > 0 ? local_end : 0) * (unsigned)sizeof(D_OUT);

    auto g_q   = opus::make_gmem(q_base);
    auto g_qs  = opus::make_gmem(qs_base);
    auto g_kv  = opus::make_gmem(reinterpret_cast<const D_BYTE*>(p_kv));
    auto g_kvs = opus::make_gmem(reinterpret_cast<const D_SCALE*>(p_kv_scale));
    auto g_bt  = opus::make_gmem(bt_base);
    auto g_w   = opus::make_gmem(w_base);
    auto g_out = opus::make_gmem(out_base, out_bytes);

    // q_a[mi][kt]: the 16 B payload in the low half of a 256-bit operand, zeroed so the ignored
    // high half is never undefined.
    auto u_q = make_layout_q<T>(lane_m0, lane_g);
    typename Mma::vtype_a q_a[MT][KT]{};
    auto u_qs = make_layout_scale<T>(lane);
    int q_scale;
    auto u_w = make_layout_w<T>(lane_g);
    opus::vector_t<float, EC> w_pl[MT];
    auto u_kv  = make_layout_kv<T>(lane_m0, lane_g);
    auto u_kvs = make_layout_scale<T>(lane);
    auto u_bt  = make_layout_bt<T>(wave);
    const int key_lane = wave * (NT * T::MMA_N) + lane_m0;   // this lane's key in a tile

    // Lanes L and L^32 hold the same logit; lane group 1 is biased out of bounds and masked. A
    // large negative constant, never a multiple of the window: that could land back inside it.
    const int lane_bias = (lane_g != 0) ? -(chunk_start + tile_count) * TK : 0;

    auto load_page_id = [&](int tile_idx) {
        return opus::load<1>(g_bt, u_bt + (chunk_start + tile_idx))[0];
    };
    auto load_kv_scale = [&](int page_id) {
        return opus::load<1>(
            g_kvs, u_kvs + page_id * (T::KVS_PAGE_BYTES / (int)sizeof(D_SCALE)))[0];
    };

    using kv_nt_t = typename Mma::vtype_b;
    using sfrag   = opus::vector_t<float, MT * EC>;

    auto clamp_tile = [&](int t) { return t < tile_count ? t : tile_count - 1; };
    // NT * KT independent 16 B loads, each covering two 512 B runs.
    auto issue_kv = [&](kv_nt_t (&kv)[NT][KT], int& kvs, int page_id) {
        kvs = load_kv_scale(page_id);
        opus::static_for<NT>([&](auto ntc) {
            constexpr int nt = ntc.value;
            opus::static_for<KT>([&](auto ktc) {
                constexpr int kt = ktc.value;
                auto v = opus::load<T::KV_GRP_BYTES>(
                    g_kv, u_kv + kv_page_offset<T>(page_id) + kt * T::STRIDE_KV_KTILE +
                              nt * T::STRIDE_KV_NTILE);
                opus::set_slice(kv[nt][kt], v, opus::number<0>{}, opus::number<T::KV_GRP_BYTES>{});
            });
        });
    };

    // One phase: refill the buffer tile t freed with tile t + 2, then tile t + 1's GEMMs, tile t's
    // relu and head reduction.
    auto compute_phase = [&](kv_nt_t (&kv_c)[NT][KT], int& kvs_c, sfrag (&acc_c)[NT],
                             kv_nt_t (&kv_p)[NT][KT], int& kvs_p, sfrag (&acc_p)[NT],
                             int& pg_c, int pf_tile, opus::vector_t<float, NT>& out) {
        __builtin_amdgcn_sched_barrier(0);
        issue_kv(kv_c, kvs_c, pg_c);
        pg_c = load_page_id(clamp_tile(pf_tile + 2));
        __builtin_amdgcn_sched_barrier(0);
        compute_tile<T>(mma_op, q_a, q_scale, w_pl, acc_c, acc_p, kv_p, kvs_p, out);
    };
    // Lanes L and L^32 hold the same logit after the head reduction; lane group 1 is biased out of
    // bounds. Fenced so the stores stay out of the MFMA groups' interleave.
    auto store = [&](opus::vector_t<float, NT>& out, int t) {
        __builtin_amdgcn_sched_barrier(0);
        const int tile_off = (chunk_start + t) * TK + lane_bias;
        opus::static_for<NT>([&](auto ntc) {
            constexpr int nt = ntc.value;
            g_out.template store<1>(out[nt], tile_off + nt * T::MMA_N + key_lane);
        });
        __builtin_amdgcn_sched_barrier(0);
    };

    // Zero-initialized: fp4 fills only the low 16 B of each 256-bit operand.
    kv_nt_t kvA[NT][KT]{}, kvB[NT][KT]{};
    int     kvsA, kvsB;
    sfrag   accA[NT], accB[NT];
    int     pgA, pgB;
    opus::vector_t<float, NT> outA, outB;

    // Prologue: Q, q_scale and weights straight from global, then KV tiles 0 and 1.
    load_q<T>(g_q, u_q, q_a);
    q_scale = opus::load<1>(g_qs, u_qs)[0];
    load_weights<T>(g_w, u_w, w_pl, weight_scale);

    // Whole-tile lead: tile 0 is fully computed into accA so phase 0 can reload kvA at once;
    // tile 1 is staged in kvB for that phase's GEMMs.
    issue_kv(kvA, kvsA, load_page_id(clamp_tile(0)));
    issue_kv(kvB, kvsB, load_page_id(clamp_tile(1)));
    pgA = load_page_id(clamp_tile(2));
    gemm_ntile<T, 0>(mma_op, q_a, q_scale, accA[0], kvA[0], kvsA);
    gemm_ntile<T, 1>(mma_op, q_a, q_scale, accA[1], kvA[1], kvsA);
    pgB = load_page_id(clamp_tile(3));
    __builtin_amdgcn_sched_barrier(0);

    compute_phase(kvA, kvsA, accA, kvB, kvsB, accB, pgA, 2, outA);
    __builtin_amdgcn_sched_barrier(0);

    // Each phase stores its own result: stores lagged a phase tighten the loop-head s_waitcnt
    // (their WAR with the next MFMA) beyond what the KV needs.
    store(outA, 0);
    if (tile_count >= 2) {
        compute_phase(kvB, kvsB, accB, kvA, kvsA, accA, pgB, 3, outB);
        store(outB, 1);
    }

    int t = 2;
    for (; t + 1 < tile_count; t += 2) {
        compute_phase(kvA, kvsA, accA, kvB, kvsB, accB, pgA, t + 2, outA);
        store(outA, t);
        compute_phase(kvB, kvsB, accB, kvA, kvsA, accA, pgB, t + 3, outB);
        store(outB, t + 1);
    }
    if (t < tile_count) {
        compute_phase(kvA, kvsA, accA, kvB, kvsB, accB, pgA, t + 2, outA);
        store(outA, t);
    }
}

}  // namespace opus_logits::gfx950
