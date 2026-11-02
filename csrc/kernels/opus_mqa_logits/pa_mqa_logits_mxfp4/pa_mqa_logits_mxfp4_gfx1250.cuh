#pragma once
// gfx1250 wave32/TDM (WMMA 32x16x128) device body for the MXFP4 MQA-logits kernel.
// Included by pa_mqa_logits_mxfp4_opus.h under PA_MQA_LOGITS_MXFP4_IMPL; NOT standalone.

// -ffinite-math-only (implied by -ffast-math) folds OPUS_LOGITS_RELU's NaN-propagating IEEE
// maximum into a NaN-swallowing select; enforce -fno-finite-math-only here.
#if defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__
#error "build this TU with -fno-finite-math-only: -ffinite-math-only folds OPUS_LOGITS_RELU's \
IEEE maximum back to a compare-and-select, which SWALLOWS a NaN E8M0 scale instead of \
propagating it. See optCompilerConfig.json's flags_extra_hip for this module."
#endif

// Device pass on gfx1250 gets the real kernel; every other pass gets an empty stub so the
// launcher's __device_stub__ resolves.
// Both halves of the guard are load-bearing: the host pass reports warp size 64 and would
// silently build the wave64 fragment layout with every byte count still matching.
#if !defined(__HIP_DEVICE_COMPILE__) || !defined(__gfx1250__)
namespace opus_logits::gfx1250 {
template <class T, mqa_logits_sched SCHED = mqa_logits_sched::Table>
__global__ void pa_mqa_logits_mxfp4_kernel(opus_mqa_logits_kargs)
{
}
} // namespace opus_logits::gfx1250
#else
#include <opus/opus.hpp>
#include <bit>   // std::bit_cast -- see permlane_head_reduce for why not __builtin_bit_cast

namespace opus_logits::gfx1250 {

// "+s" on purpose: the read-only form is a memory clobber and demotes indexed s_loads. Never
// pin a pointer: the asm divergence spreads to every derived address and costs occupancy.
template<class X>
__device__ inline void pin_sgpr(X& v) { asm("" : "+s"(v)); }

// Fenced so the scheduler cannot move LDS traffic across it; callers owe the preceding wait.
__device__ inline void cta_sync() {
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);
}

// Sum across the two C lane groups (partner at lane distance 16).
// Both returned lanes are needed: r[0] alone computes v + v.
// std::bit_cast is required: __builtin_bit_cast miscompiles this swap to v + v, which uniform
// test data cannot detect.
__device__ inline float permlane_head_reduce(float v) {
    opus::vector_t<opus::u32_t, 2> r = __builtin_amdgcn_permlane16_swap(
        std::bit_cast<opus::u32_t>(v), std::bit_cast<opus::u32_t>(v), false, false);
    return std::bit_cast<float>(r[0]) + std::bit_cast<float>(r[1]);
}

// Grid: one CTA per SCHEDULE SLOT (grid.x == kargs.num_ctas).
template<class T, mqa_logits_sched SCHED = mqa_logits_sched::Table>
__global__ __launch_bounds__(T::BLOCK_SIZE, T::WAVES_PER_EU)
void pa_mqa_logits_mxfp4_kernel(opus_mqa_logits_kargs kargs) {
    static_assert(SCHED == mqa_logits_sched::Table,
                  "this target compiles the schedule-table mapping and nothing else. Decode needs "
                  "no separate mapping: a decode table is one whose tiles carry a zero start.");

    using D_BYTE  = opus::u8_t;
    using D_ACC   = typename T::D_ACC;
    using D_SCALE = typename T::D_SCALE;
    using D_OUT   = typename T::D_OUT;
    using D_WEIGHT = typename T::D_WEIGHT;
    using Mma = opus::wmma<opus::fp4_t, opus::fp4_t, D_ACC,
                          T::MMA_M, T::MMA_N, T::MMA_K, T::WAVE_SIZE>;
    Mma mma_op;
    static_assert(sizeof(typename Mma::vtype_a) == T::A_BYTES_PER_LANE);
    static_assert(sizeof(typename Mma::vtype_b) == T::B_BYTES_PER_LANE);
    static_assert(Mma::elem_c == T::C_FRAG);

    constexpr int WAVE   = T::WAVE_SIZE;
    constexpr int MT     = T::M_TILES;
    constexpr int NT     = T::N_TILES;
    constexpr int EC     = T::C_FRAG;
    constexpr int PAGES   = T::PAGES_PER_TILE;
    constexpr int PACK_W  = T::PACK_C;              // 8 bf16 = 16 B, one load per C run

    __shared__ __attribute__((aligned(16))) D_BYTE smem_buf[T::LDS_BYTES];

    const void* p_q        = kargs.ptr_q;
    const void* p_q_scale  = kargs.ptr_q_scale;
    const void* p_kv       = kargs.ptr_kv;
    const void* p_kv_scale  = kargs.ptr_kv_scale;
    const int*  p_bt        = kargs.ptr_block_tables;
    const void* p_weights   = kargs.ptr_weights;
    D_OUT*      p_out       = kargs.ptr_out;
    int   stride_out   = kargs.stride_out_row;
    float weight_scale = kargs.weight_scale;
    int   max_blk      = kargs.max_blocks_per_seq;
    int   num_rows     = kargs.num_rows;
    pin_sgpr(stride_out); pin_sgpr(weight_scale); pin_sgpr(max_blk); pin_sgpr(num_rows);
    const opus_mqa_cta_record* p_cta_info = kargs.ptr_cta_info;
    const int* p_local_starts = kargs.ptr_local_starts;
    const int* p_local_ends   = kargs.ptr_local_ends;

    const int tid = opus::thread_id_x();
    // readfirstlane: q_slot is wave-uniform but divergent to LLVM, and it feeds the row base
    // pointers and TDM issue, which must stay scalar to avoid waterfalls.
    const int q_slot  = __builtin_amdgcn_readfirstlane(tid / WAVE);
    const int lane    = tid % WAVE;
    const int lane_m0 = lane % T::GRPN_C;    // L % 16: the n column / the m row within a tile
    const int lane_g  = lane / T::GRPN_C;    // L / 16: the K-block half, and the C m-group

    // local_* is the CTA-uniform group union and alone may set chunk_start / tile_count (the
    // barrier trip count). own_* is this wave's row and only sets the store mask. A wave past
    // the end of a short group must not return (barrier deadlock); it runs with own_end = 0 so
    // every store is dropped.
    constexpr int kv_tile_size = T::KV_TILE_SIZE;
    const opus_mqa_cta_record rec = p_cta_info[opus::block_id_x()];
    // Two clauses against num_rows. This return is CTA-uniform (rec is blockIdx-uniform) and
    // covers empty/surplus slots; it also makes the clamp below safe, since a masked-off wave
    // falls back to rec.row_id. The row bound (unchecked device data from cu_seq_q) buys a
    // failure mode, not correctness: bad input drops rows instead of reading/writing out of range.
    if (rec.chunk_count <= 0 || rec.row_id >= num_rows) return;
    // The per-wave clause folds into the store mask instead of returning (see above).
    const bool slot_ok    = q_slot < rec.group_rows && rec.row_id + q_slot < num_rows;
    // Select, not branch: both operands are scalar, so this is s_cselect and bases stay scalar.
    const int row_id      = rec.row_id + (slot_ok ? q_slot : 0);
    int       batch_id    = rec.batch_id;
    const int own_start   = p_local_starts ? p_local_starts[row_id] : rec.local_start;
    const int own_end     = slot_ok ? p_local_ends[row_id] : 0;
    const int chunk_start = rec.chunk_start;         // ABSOLUTE tile index, not an offset
    const int tile_count  = rec.chunk_count;
    pin_sgpr(batch_id);

    const D_BYTE*   q_base  = reinterpret_cast<const D_BYTE*>(p_q) + (size_t)row_id * T::Q_ROW_BYTES;
    const D_SCALE*  qs_base = reinterpret_cast<const D_SCALE*>(p_q_scale) + (size_t)row_id * T::N_HEADS;
    const D_WEIGHT* w_base  = reinterpret_cast<const D_WEIGHT*>(p_weights) + (size_t)row_id * T::W_ROW_ELEMS;
    const int*      bt_base = p_bt + (size_t)batch_id * max_blk;

    // own_start folded into the base and num_records = window length, so the buffer bound
    // check masks both ends (a negative offset is huge unsigned). Per-wave window, not group's.
    D_OUT*         out_base  = p_out + (size_t)row_id * stride_out + own_start;
    const unsigned out_bytes =
        (unsigned)(own_end > own_start ? own_end - own_start : 0) * (unsigned)sizeof(D_OUT);

    static_assert(T::LDS_STAGE_BYTES % 16 == 0 && T::LDS_PAGE_BYTES % 16 == 0
                      && T::TOKEN_BYTES % 16 == 0 && T::KV_GRP_BYTES % 16 == 0,
                  "a dword view needs every offset 4 B aligned; these are all 16 B aligned");
    constexpr int DW = (int)sizeof(D_SCALE);   // 4: the byte -> dword divisor, named so it reads
    auto g_q   = opus::make_gmem(reinterpret_cast<const D_SCALE*>(q_base));
    auto g_qs  = opus::make_gmem(qs_base);
    auto g_w   = opus::make_gmem(w_base);
    auto g_out = opus::make_gmem(out_base, out_bytes);
    auto s_dw  = opus::make_smem(reinterpret_cast<D_SCALE*>(smem_buf));

    typename Mma::vtype_a q_a[MT]{};
    int                   q_scale[MT]{};
    opus::vector_t<float, EC> w_pl[MT];
    // Two named accumulators; do not reintroduce acc[2][MT][NT]: a runtime subscript puts the
    // whole accumulator in scratch (invisible to .vgpr_spill_count). The phase loop is unrolled
    // by two so the choice stays compile-time.
    typename Mma::vtype_c accA[MT][NT], accB[MT][NT];
    float outA[NT], outB[NT];

    // block_tables via the constant address space so page ids come from s_load (TDM descriptor
    // fields live in SGPRs). A plain load cannot be proven NOCLOBBER past the store to out.
    using bt_scalar_ptr = const int __attribute__((address_space(4)))*;
    auto bt_c = reinterpret_cast<bt_scalar_ptr>(
        reinterpret_cast<__UINTPTR_TYPE__>(bt_base + chunk_start * PAGES));
    auto page_id = [&](int t, int p) { return bt_c[t * PAGES + p]; };

    // CU scope, not opus's device default (which defeats the L2). Legal only because KV and
    // scales are read-only for the launch: nothing may write the KV cache during the launch.
    using fp4x2 = opus::array<opus::fp4_t, 2>;
    constexpr int TDM_CACHE = opus::tdm_traits::make_cache_policy(
        opus::tdm_traits::load_temporal_hint::regular, opus::tdm_traits::scope::cu);
    using Cache = opus::tdm_traits::cache<TDM_CACHE>;
    using KvWin = opus::tdm<fp4x2, opus::seq<T::TOKEN_BYTES, T::TDM_ROWS>,
                            opus::tdm_traits::padding<fp4x2, T::LDS_PAD_INTERVAL, T::LDS_PAD_AMOUNT>,
                            Cache>;
    using KvsWin = opus::tdm<opus::u8_t, opus::seq<T::SCALE_BYTES_PER_DWORD, T::TDM_ROWS>, Cache>;

    // Constant: the window is walked, tensor_dim only clamps, and any large extent is equivalent.
    constexpr opus::u32_t TDM_EXTENT = 1u << 30;   // rows; 2^30 tokens, and no u32 overflow

    // The ABI lacks num_blocks, so TDM's out-of-range clamp is unusable; an over-issued tile
    // re-fetches the last real tile instead, which nobody reads.
    auto clamp_tile = [&](int t) { return t < tile_count ? t : tile_count - 1; };

    const int tdm_page  = q_slot / T::TDM_PAGE_SPLIT;
    const int tdm_piece = q_slot % T::TDM_PAGE_SPLIT;
    static_assert(T::TDM_ISSUE_WAVES == T::NUM_WAVES,
                  "the branch-free form needs one issuing wave per piece; with fewer issuers the "
                  "kernel needs the `q_slot < TDM_ISSUE_WAVES` guard this replaced");

    // Built once and walked (only the origin moves). The LDS base is this wave's fixed piece in
    // a stage; the stage offset goes to async_load(). Uses the write-stream map, not lds_byte,
    // because TDM inserts the pad as it writes.
    const int kv_lds_base  = tdm_page * T::LDS_PAGE_BYTES + tdm_piece * T::LDS_PIECE_BYTES;
    const int kvs_lds_base = T::lds_scale_byte(tdm_page * T::PAGE_SIZE + tdm_piece * T::TDM_ROWS);
    auto kv_win = opus::make_tdm<KvWin>(
        (opus::u32_t)kv_lds_base, reinterpret_cast<const D_BYTE*>(p_kv),
        /*shape0*/ (opus::u32_t)T::TOKEN_BYTES, /*shape1*/ TDM_EXTENT,
        /*dim0_stride*/ (opus::u64_t)T::TOKEN_BYTES, /*origin0*/ 0u, /*origin1*/ 0u);
    auto kvs_win = opus::make_tdm<KvsWin>(
        (opus::u32_t)kvs_lds_base, reinterpret_cast<const D_BYTE*>(p_kv_scale),
        (opus::u32_t)T::SCALE_BYTES_PER_DWORD, TDM_EXTENT,
        (opus::u64_t)T::SCALE_BYTES_PER_DWORD, 0u, 0u);
    int tdm_origin = 0;
    auto issue_tile = [&](int t) {
        const int ct     = clamp_tile(t);
        const int pid    = page_id(ct, tdm_page);
        const int origin = pid * T::PAGE_SIZE + tdm_piece * T::TDM_ROWS;
        kv_win.move(opus::number<0>{}, origin - tdm_origin);
        kvs_win.move(opus::number<0>{}, origin - tdm_origin);
        tdm_origin = origin;
        const opus::u32_t stage = (opus::u32_t)T::lds_stage_byte(ct);
        kv_win.async_load(stage);
        kvs_win.async_load(stage);
    };

    constexpr int GRP_DW = T::KV_GRP_BYTES / 4;              // 4 dwords = one 32-element block
    const int lds_lane_dw = T::lds_byte(lane_m0, lane_g) / DW;
    auto read_kv_lds = [&](typename Mma::vtype_b (&kv)[NT], int stage_dw) {
        opus::static_for<NT>([&](auto ntc) {
            constexpr int nt   = ntc.value;
            constexpr int page = nt / T::TILES_PER_PAGE;
            constexpr int base = (nt % T::TILES_PER_PAGE) * T::LDS_NTILE_BYTES / 4;
            opus::static_for<T::B_VGPR_GROUPS>([&](auto vc) {
                constexpr int v = vc.value;
                // Second half is block g+2; the step differs between kv_cache layouts, so take
                // it from the traits, never hardcode it.
                constexpr int imm = v * 2 * T::LDS_BLOCK_BYTES / 4;
                auto x = opus::load<GRP_DW>(
                    s_dw, stage_dw + T::lds_page_byte(page) / 4 + base + imm + lds_lane_dw);
                opus::set_slice(kv[nt], x, opus::number<v * GRP_DW>{},
                                opus::number<(v + 1) * GRP_DW>{});
            });
        });
    };
    auto read_kv_scale_lds = [&](int (&kvs)[NT / 2], int stage_dw) {
        opus::static_for<NT / 2>([&](auto uc) {
            constexpr int u = uc.value;
            kvs[u] = opus::load<1>(s_dw, stage_dw + T::lds_scale_byte(u * 2 * T::MMA_N) / DW + lane)[0];
        });
    };

    auto gemm_tile = [&](typename Mma::vtype_c (&a)[MT][NT],
                         typename Mma::vtype_b (&kv)[NT], int (&kvs)[NT / 2]) {
        opus::static_for<NT>([&](auto ntc) {
            constexpr int nt = ntc.value;
            // a_scale_sel = 0 (M = 32 fills all lanes); b_scale_sel = n-tile parity, so one
            // scale dword serves two n-tiles.
            constexpr int bsel = nt % 2;
            opus::static_for<MT>([&](auto mic) {
                constexpr int mi = mic.value;
                a[mi][nt] = mma_op(q_a[mi], kv[nt], q_scale[mi], kvs[nt / 2],
                                   opus::number<0>{}, opus::number<bsel>{});
            });
        });
    };

    auto reduce_tile = [&](typename Mma::vtype_c (&a)[MT][NT], float (&out)[NT]) {
        opus::static_for<NT>([&](auto ntc) {
            constexpr int nt = ntc.value;
            opus::vector_t<float, 2> s2{0.0f, 0.0f};
            opus::static_for<MT>([&](auto mic) {
                constexpr int mi = mic.value;
                opus::static_for<EC / 2>([&](auto pc) {
                    constexpr int p = pc.value;
                    opus::vector_t<float, 2> w{ w_pl[mi][2 * p], w_pl[mi][2 * p + 1] };
                    // Packed two heads at a time so the sum vectorizes to v_pk_fma_f32.
                    opus::vector_t<float, 2> r{ OPUS_LOGITS_RELU(a[mi][nt][2 * p]),
                                                OPUS_LOGITS_RELU(a[mi][nt][2 * p + 1]) };
                    s2 = r * w + s2;
                });
            });
            out[nt] = permlane_head_reduce(s2[0] + s2[1]);
        });
    };

    // Only lane group 0 holds a useful logit after the butterfly; lane_bias pushes group 1's
    // stores out of range instead of branching.
    const int lane_bias = (lane_g != 0) ? -(chunk_start + tile_count) * kv_tile_size : 0;
    auto do_store = [&](float (&out)[NT], int t) {
        const int tile_off = (chunk_start + t) * kv_tile_size - own_start + lane_bias;
        opus::static_for<NT>([&](auto ntc) {
            constexpr int nt = ntc.value;
            g_out.template store<1>(out[nt], tile_off + nt * T::MMA_N + lane_m0);
        });
    };
    auto fenced_store = [&](float (&out)[NT], int t) {
        __builtin_amdgcn_sched_barrier(0);
        do_store(out, t);
        __builtin_amdgcn_sched_barrier(0);
    };

    opus::static_for<MT>([&](auto mic) {
        constexpr int mi = mic.value;
        opus::static_for<T::A_VGPR_GROUPS>([&](auto vc) {
            constexpr int v = vc.value;
            const int head  = mi * T::MMA_M + lane_m0 + (v >= 2 ? 16 : 0);
            const int block = lane_g + (v % 2 ? 2 : 0);
            auto x = opus::load<GRP_DW>(g_q, T::q_byte(head, block) / DW);
            opus::set_slice(q_a[mi], x, opus::number<v * GRP_DW>{},
                            opus::number<(v + 1) * GRP_DW>{});
        });
    });
    opus::static_for<MT>([&](auto mic) {
        constexpr int mi = mic.value;
        q_scale[mi] = opus::load<1>(g_qs, mi * T::MMA_M + lane)[0];
    });
    opus::static_for<MT>([&](auto mic) {
        constexpr int mi = mic.value;
        opus::static_for<T::REPT_C>([&](auto rc) {
            constexpr int r = rc.value;
            const int h0 = mi * T::MMA_M + (r ? 16 : 0) + lane_g * PACK_W;
            auto wv = opus::load<PACK_W>(g_w, h0);   // 8 bf16 = 16 B
            opus::static_for<PACK_W>([&](auto jc) {
                constexpr int j = jc.value;
                w_pl[mi][r * PACK_W + j] = (float)wv[j] * weight_scale;
            });
        });
    });

    opus::static_for<T::TILES_IN_FLIGHT>([&](auto sc) { issue_tile(sc.value); });

    typename Mma::vtype_b kv_b[NT]{};
    int                   kvs_w[NT / 2]{};

    // One phase = one KV tile. The TENSORCNT wait retires this wave's loads of tile t; the
    // barrier publishes it CTA-wide and orders the previous phase's LDS reads before the issue
    // that recycles their stage. acc_cur's WMMAs and acc_prev's reduce are independent (the
    // ping-pong); do not put a sched_barrier between them.
    auto phase = [&](auto& acc_cur, auto& acc_prev, float (&out_prev)[NT], int t, bool has_prev) {
        opus::s_wait_tensorcnt<T::TENSORCNT_KEEP>();
        cta_sync();
        issue_tile(t + T::ISSUE_LEAD);
        const int stage_dw = T::lds_stage_byte(t) / DW;
        read_kv_scale_lds(kvs_w, stage_dw);
        read_kv_lds(kv_b, stage_dw);
        __builtin_amdgcn_sched_barrier(0);
        gemm_tile(acc_cur, kv_b, kvs_w);
        if (has_prev) reduce_tile(acc_prev, out_prev);
    };

    // Branch only on CTA-uniform tile_count; a per-row condition would deadlock the barrier.
    phase(accA, accB, outB, 0, false);
    int t = 1;
    for (; t + 1 < tile_count; t += 2) {
        phase(accB, accA, outA, t,     true);  fenced_store(outA, t - 1);
        phase(accA, accB, outB, t + 1, true);  fenced_store(outB, t);
    }
    if (t < tile_count) {   // t == tile_count - 1, and odd
        phase(accB, accA, outA, t, true);      fenced_store(outA, t - 1);
        reduce_tile(accB, outB);               fenced_store(outB, t);
    } else {                // tile_count - 1 was even, so accA holds it
        reduce_tile(accA, outA);               fenced_store(outA, tile_count - 1);
    }
}

} // namespace opus_logits::gfx1250
#endif
