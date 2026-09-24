#pragma once
// gfx1250 wave32/TDM (WMMA 32x16x128) device body for the MXFP4 MQA-logits kernel.
// Included by pa_mqa_logits_mxfp4_opus.h under PA_MQA_LOGITS_MXFP4_IMPL; NOT standalone.

// ── gfx1250 device body (real under __gfx1250__, host/other-arch stub otherwise) ─────
// A check rather than a comment, because a JSON build config cannot carry the reason:
// `-ffast-math` implies `-ffinite-math-only`, which folds OPUS_LOGITS_RELU's IEEE maximum back
// to a NaN-swallowing select. See that macro's definition for what that costs.
#if defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__
#error "build this TU with -fno-finite-math-only: -ffinite-math-only folds OPUS_LOGITS_RELU's \
IEEE maximum back to a compare-and-select, which SWALLOWS a NaN E8M0 scale instead of \
propagating it. See optCompilerConfig.json's flags_extra_hip for this module."
#endif

// The device pass on gfx1250 gets the real kernel; every other pass gets an empty stub so the
// launcher's `__device_stub__` reference still resolves. BOTH HALVES OF THE GUARD ARE
// LOAD-BEARING: `opus::get_warp_size()` answers 64 in the host pass, which would silently build
// the wave64 fragment layout with every byte count still matching.
#if !defined(__HIP_DEVICE_COMPILE__) || !defined(__gfx1250__)
namespace opus_logits {
namespace qshare {
template <class T, mqa_logits_sched SCHED = mqa_logits_sched::Table>
__global__ void mqa_logits_mxfp4_32x16x128_qshare_kernel(opus_mqa_logits_kargs)
{
}
} // namespace qshare
} // namespace opus_logits
#else
#include <opus/opus.hpp>
#include <bit>   // std::bit_cast -- see permlane_head_reduce for why not __builtin_bit_cast

namespace opus_logits {
namespace qshare {

// The write form ("+s") on purpose: the read-only one is a memory clobber and demotes the
// indexed s_loads. Do NOT pin a pointer -- that propagates inline-asm divergence into every
// address derived from it and costs occupancy.
template<class X>
__device__ inline void pin_sgpr(X& v) { asm("" : "+s"(v)); }

// Fenced on both sides so the scheduler cannot drag LDS traffic across it; callers owe the wait
// that gives the barrier something to publish. Lowers to TWO instructions on gfx1250.
__device__ inline void cta_sync() {
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);
}

// Sum a per-lane value across the two C lane groups, whose partner sits at lane distance 16.
// TWO WAYS TO GET THIS WRONG, both silent: the builtin returns (self, partner), so taking only
// [0] computes v + v; and `std::bit_cast` is load-bearing -- `__builtin_bit_cast` miscompiled
// this swap to v + v, which uniform test data cannot distinguish from correct.
__device__ inline float permlane_head_reduce(float v) {
    opus::vector_t<opus::u32_t, 2> r = __builtin_amdgcn_permlane16_swap(
        std::bit_cast<opus::u32_t>(v), std::bit_cast<opus::u32_t>(v), false, false);
    return std::bit_cast<float>(r[0]) + std::bit_cast<float>(r[1]);
}

// Grid: one CTA per SCHEDULE SLOT (grid.x == kargs.num_ctas).
template<class T, mqa_logits_sched SCHED = mqa_logits_sched::Table>
__global__ __launch_bounds__(T::BLOCK_SIZE, T::WAVES_PER_EU)
void mqa_logits_mxfp4_32x16x128_qshare_kernel(opus_mqa_logits_kargs kargs) {
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
    // The kernel's only use of num_rows -- it was validation-only until the row_id bound below.
    int   num_rows     = kargs.num_rows;
    pin_sgpr(stride_out); pin_sgpr(weight_scale); pin_sgpr(max_blk); pin_sgpr(num_rows);
    const opus_mqa_cta_record* p_cta_info = kargs.ptr_cta_info;
    const int* p_local_starts = kargs.ptr_local_starts;
    const int* p_local_ends   = kargs.ptr_local_ends;

    const int tid = opus::thread_id_x();
    // readfirstlane, not a plain shift: the slot is wave-uniform in fact but divergent to LLVM,
    // and it feeds this wave's row base pointers and the TDM issue -- both must stay scalar or
    // every derived address goes through a waterfall.
    const int q_slot  = __builtin_amdgcn_readfirstlane(tid / WAVE);
    const int lane    = tid % WAVE;
    const int lane_m0 = lane % T::GRPN_C;    // L % 16: the n column / the m row within a tile
    const int lane_g  = lane / T::GRPN_C;    // L / 16: the K-block half, and the C m-group

    // TWO WINDOWS COME OUT OF THIS BLOCK AND THEY MUST NOT BE CONFUSED. `local_*` is the GROUP
    // UNION and is CTA-uniform: it, and only it, may reach chunk_start / tile_count, because
    // those set the barrier count. `own_*` is THIS WAVE'S row and reaches nothing but the store
    // mask and the out descriptor's size. A wave past the end of a short group does NOT return;
    // it runs with own_end = 0, so its out descriptor has zero records and every store it makes
    // is dropped. That is the whole of "short groups are masked, not padded".
    constexpr int kv_tile_size = T::KV_TILE_SIZE;
    // The whole assignment in one `s_load_dwordx8` off a blockIdx-uniform address. The
    // surplus-slot return is CTA-uniform, so it cannot deadlock the phase barrier.
    const opus_mqa_cta_record rec = p_cta_info[opus::block_id_x()];
    // TWO clauses against num_rows, and the split is forced: an out-of-range wave may NOT
    // return, because a non-uniform early return leaves the four waves with unequal phase
    // barrier counts and DEADLOCKS. This one is CTA-uniform -- rec comes off a blockIdx-uniform
    // address -- and it is also what makes the clamp below safe, since a masked-off wave falls
    // back to rec.row_id and still reads p_local_starts[row_id] unconditionally.
    //
    // The row space is the caller's and nothing else bounds it: cu_seq_q -> cu_tiles ->
    // rec.row_id is a chain the launcher cannot check, because cu_seq_q[batch] is device data
    // and reading it host-side is the sync this design exists to avoid. This buys a FAILURE
    // MODE, not correctness -- a caller whose cu_seq_q reaches past q is wrong either way --
    // but the failure is now dropped rows instead of reads past q and writes past out.
    if (rec.chunk_count <= 0 || rec.row_id >= num_rows) return;
    // The per-wave half folds in HERE rather than returning: the wave runs to the end with
    // own_end = 0, so its out descriptor holds zero records and every store it makes is
    // dropped -- the same path short groups already take.
    const bool slot_ok    = q_slot < rec.group_rows && rec.row_id + q_slot < num_rows;
    // Clamped, not branched: q_slot is readfirstlane'd and group_rows comes off the record, so
    // the select is s_cselect and the row's base addresses below stay scalar.
    const int row_id      = rec.row_id + (slot_ok ? q_slot : 0);
    int       batch_id    = rec.batch_id;
    // The null-starts branch is on a kargs pointer, so it is uniform across the whole grid.
    const int own_start   = p_local_starts ? p_local_starts[row_id] : rec.local_start;
    const int own_end     = slot_ok ? p_local_ends[row_id] : 0;
    const int chunk_start = rec.chunk_start;         // ABSOLUTE tile index, not an offset
    const int tile_count  = rec.chunk_count;
    pin_sgpr(batch_id);

    const D_BYTE*   q_base  = reinterpret_cast<const D_BYTE*>(p_q) + (size_t)row_id * T::Q_ROW_BYTES;
    const D_SCALE*  qs_base = reinterpret_cast<const D_SCALE*>(p_q_scale) + (size_t)row_id * T::N_HEADS;
    const D_WEIGHT* w_base  = reinterpret_cast<const D_WEIGHT*>(p_weights) + (size_t)row_id * T::W_ROW_ELEMS;
    const int*      bt_base = p_bt + (size_t)batch_id * max_blk;

    // own_start folded into the base and num_records set to the window LENGTH, so BOTH bounds
    // are the hardware's: a token below the window gets a negative offset, which as an unsigned
    // byte offset is huge and fails the same test that catches one past the end. THIS WAVE'S
    // row throughout -- a wave whose window is shorter than the group's must not write the
    // difference.
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
    // The two result matrices, as TWO NAMED VARIABLES and not `acc[2][MT][NT]`. That is not
    // style: a runtime subscript on the set puts the whole 512-VGPR accumulator in SCRATCH
    // before the register allocator ever sees it -- 2112 B/lane, and `.vgpr_spill_count` still
    // reads 0 because nothing was spilled, it was never in registers. The phase loop is
    // unrolled by two so the set stays a compile-time choice.
    typename Mma::vtype_c accA[MT][NT], accB[MT][NT];
    float outA[NT], outB[NT];

    // block_tables through the CONSTANT address space so it lowers to s_load_dword, which is
    // load-bearing: a TDM descriptor's fields are all uniform and live in SGPRs, so a page id in
    // a VGPR would put a readfirstlane waterfall around every issue. Neither opus::load nor a
    // plain dereference gets there -- the pass also has to prove the load is NOCLOBBER, and the
    // loop's store to `out` defeats that.
    using bt_scalar_ptr = const int __attribute__((address_space(4)))*;
    auto bt_c = reinterpret_cast<bt_scalar_ptr>(
        reinterpret_cast<__UINTPTR_TYPE__>(bt_base + chunk_start * PAGES));
    auto page_id = [&](int t, int p) { return bt_c[t * PAGES + p]; };

    // CU SCOPE, NOT opus's DEVICE default: device scope defeats the L2 and costs between 1.55x
    // and 7x. It is legal only because the KV cache and its scales are written by the indexer
    // BEFORE the launch and are read-only for its duration -- SO NOTHING MAY WRITE THE KV CACHE
    // CONCURRENTLY WITH IT, which is a condition on the caller and not on this kernel.
    using fp4x2 = opus::array<opus::fp4_t, 2>;
    constexpr int TDM_CACHE = opus::tdm_traits::make_cache_policy(
        opus::tdm_traits::load_temporal_hint::regular, opus::tdm_traits::scope::cu);
    using Cache = opus::tdm_traits::cache<TDM_CACHE>;
    using KvWin = opus::tdm<fp4x2, opus::seq<T::TOKEN_BYTES, T::TDM_ROWS>,
                            opus::tdm_traits::padding<fp4x2, T::LDS_PAD_INTERVAL, T::LDS_PAD_AMOUNT>,
                            Cache>;
    using KvsWin = opus::tdm<opus::u8_t, opus::seq<T::SCALE_BYTES_PER_DWORD, T::TDM_ROWS>, Cache>;

    // A CONSTANT because the window is built once and WALKED: `tensor_dim` only CLAMPS, and the
    // DMA moves `tile_dim1` rows regardless, so any extent past the largest legal origin is equal.
    constexpr opus::u32_t TDM_EXTENT = 1u << 30;   // rows; 2^30 tokens, and no u32 overflow

    // WHY THERE IS STILL A CLAMP. TDM does clamp an out-of-range ORIGIN to a zero-extent DMA,
    // but exploiting it needs the cache's true page count and THE ABI DOES NOT CARRY
    // `num_blocks`. So an over-issued tile re-fetches the last real page instead, which nobody
    // reads. Cheaper than a guarded address chain even so: the clamp lands on one origin.
    auto clamp_tile = [&](int t) { return t < tile_count ? t : tile_count - 1; };

    const int tdm_page  = q_slot / T::TDM_PAGE_SPLIT;
    const int tdm_piece = q_slot % T::TDM_PAGE_SPLIT;
    static_assert(T::TDM_ISSUE_WAVES == T::NUM_WAVES,
                  "the branch-free form needs one issuing wave per piece; with fewer issuers the "
                  "kernel needs the `q_slot < TDM_ISSUE_WAVES` guard this replaced");

    // BUILT ONCE AND WALKED: rebuilding per phase rewrites extent, stride and base when only
    // the origin changes. The LDS write point is deliberately NOT window state, so the base here
    // is this wave's fixed piece within a stage and the per-tile offset rides in async_load()'s
    // argument -- and it is the WRITE stream's map, not `lds_byte` of a token, because TDM inserts
    // the pad as it writes.
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
                // The fragment's second half is block g+2, and the step comes FROM THE TRAITS
                // because it is not the same in the two kv_cache layouts: 16 B when a token's
                // blocks are adjacent, 1024 when the block index sits outside the token.
                // Hardcoding either builds fine on one branch and reads the wrong 16 B on the
                // other.
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
            // a_scale_sel is 0 always: at M = 32 the 32 lanes x 4 B exactly fill the m rows
            // and the field has nothing left to select. b_scale_sel is the n-tile's parity,
            // which is what lets one dword of scale serve two n-tiles.
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

    // Only lane group 0 carries a useful logit after the symmetric butterfly, so group 1's stores
    // are pushed out of range by `lane_bias` instead of being branched around.
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

    // The empty-assignment return sits at the record read, where `chunk_count <= 0` covers both the
    // surplus slot and the empty window BEFORE any TDM descriptor is built.

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

    // One phase = one KV tile. The wait retires this wave's share of tile t's loads and the
    // barrier then does TWO duties: makes the tile visible CTA-wide, and separates the previous
    // phase's ds_loads from the issue that recycles the stage they freed.
    //
    // `acc_cur` takes this tile's WMMAs while `acc_prev` is reduced. They are separate locals with
    // no aliasing, so the halves have no dependency -- that is the ping-pong, and a sched_barrier
    // between them would serialise exactly what it buys.
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

    // Every branch below is on tile_count, which is CTA-uniform, so all four waves take the same
    // one -- a per-row condition here would deadlock them on the phase barrier.
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

} // namespace qshare
} // namespace opus_logits
#endif
