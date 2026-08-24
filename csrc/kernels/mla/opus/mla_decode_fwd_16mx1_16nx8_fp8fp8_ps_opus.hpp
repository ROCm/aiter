#pragma once

// MLA decode forward on gfx950: fp8 Q x fp8 KV, 16mx8 / 32nx1, persistent scheduling.
//   GEMM0 (Q*K^T): d = 576 = 512 nope + 64 rope      GEMM1 (P*V): d_v = 512
//
// Adapted from dsa_v32_splitkv.hpp; three departures from it drive most of the rest:
//   (A) COMBINED d=576 fp8 BUFFER. q_buffer / kv_buffer are one contiguous fp8 tensor,
//       row-major with row stride D_HEAD_SIZE = 576. There is no separate rope tensor:
//       "nope" is d in [0, 512), "rope" is d in [512, 576) reached by a +D_NOPE_SIZE base
//       offset on the same pointer. rope is fp8 here (bf16 in dsa_v32).
//   (B) PER-TENSOR SCALAR DESCALE, no mxfp8 micro-scaling. q_scale_ptr / kv_scale_ptr are
//       single floats whose product scales the scores, so the 16x16x128 f8f6f4 MFMA takes
//       block scale `0` -- the literal, not 0_I (a number<0> makes the operand poison and
//       DCEs the body down to .vgpr_count 4) and not the E8M0 identity 127 (arithmetically
//       right, but only 0 selects the bare 8-byte form without v_mfma_ld_scale_b32).
//   (C) NO V DEQUANT, NO V LDS. V is the fp8 nope of K, transpose-read out of the K-nope
//       LDS by ds_read_b64_tr_b8 straight into the PV MFMA operand; descale_k is applied
//       once, on the output.
//
// Software pipeline: 4 LDS slots, one phase per KV tile, 4 stages, exactly ONE s_barrier
// per phase (stage0). Phases are unrolled in pairs so the two score buffers alternate as
// compile-time v_s[0]/v_s[1], never a runtime index. Softmax is split head/tail so its
// VALU rides in a neighbouring MFMA's shadow. Per phase, tile t:
//   stage0 [mem]     s_waitcnt_vmcnt + barrier (publishes tile t+1); ds_read K(t); fetch
//                    the page index for t+3, giving that gmem load a full phase of slack
//   stage1 [compute] gemm0 QK(t) [12 MFMA]  || softmax-tail(t-1) [4 EXP + ~18 VALU]
//   stage2 [mem]     tr_load V(t-1); mask S(t) before stage3 folds it into the softmax
//   stage3 [compute] gemm1 PV(t-1) [32 MFMA] || softmax-head(t) + the tile t+2 prefetch,
//                    chopped into per-d-slice chunks that ride the MFMA shadows
// slot_of(t) = (t - tile_begin) & 3; four slots is the minimum for the distance-2
// prefetch, since t-1 (PV still pending, and V reads out of that K slot), t (QK now),
// t+1 (landed) and t+2 (in flight) are all resident at once. The prologue primes slots
// 0..2 and runs a partial phase for tile_begin; the epilogue drains the last tail + PV.
//
// WHERE THE TIME GOES AT THE SHAPES THIS KERNEL IS DISPATCHED FOR: not here. At
// b=256 c=8192 page_size=1 the kernel runs at 228 us and moves 1.35 GB of DRAM read traffic
// doing it -- 5.9 TB/s against gfx950's ~6.3 TB/s coalesced roof, i.e. ~94% of the machine.
// The hand-written asm decode kernel (AITER_MLA_USE_OPUS=0) lands at 227.95 us on the
// identical shape, which is the cheapest confirmation available that this is a wall and not
// an implementation. Per phase a wave issues 9 MFMA, 288 MAI cycles, into a phase thousands
// of cycles long: the MAI pipe is a few per cent busy, not the 69% the 16mx8 kernel sees.
//
// The only lever is DRAM traffic, and rocprofv3 (see memprobe.sh) says where it goes. Every
// EA read is 64 B -- TCC_EA0_RDREQ_32B and TCC_BUBBLE are both zero -- and TCC counters come
// back from half the instances, so double them:
//
//               64B sectors per 576B row      L2 hit rate      DRAM read      time
//   page_size 1           10.08                  1.25%          1.352 GB     228.8 us
//   page_size 2            9.05                  7.11%          1.215 GB     216.5 us
//   minimum                9.00
//
// So page_size 1 fetches exactly one wasted 64 B sector per token row, and that is the whole
// 11%. 576 is not a multiple of 128, so a row's tail shares a 128 B line with the NEXT token
// row -- which at page_size 1 sits somewhere unrelated in the cache and is read by an
// unrelated request at an unrelated time. The working set is 1.2 GB against a 256 MB last
// level, and TCC_HIT/TCC_REQ = 1.25% says the sharing is never recovered. At page_size 2 the
// two rows are one page, the same workgroup reads them together, the hit rate goes to 7.1%
// and the extra sector disappears -- the traffic is then exactly the 9 sectors the data is.
//
// Nothing in the kernel can reach this: it is set by the page table the caller hands in.
// Use page_size >= 2. It is not a micro-optimisation, it removes a structural 11%, and 4
// adds ~0.5% over 2 because there is nothing left to recover.
//
// Both sides are at the memory system's knee either way. TCC_EA0_RDREQ_LEVEL / RDREQ puts
// average EA read latency at ~1840 cycles, well above unloaded, and
// TCC_EA0_RDREQ_DRAM_CREDIT_STALL falls 1.83x for that 11% traffic cut -- a superlinear
// response to backing off is what saturation looks like.
//
// Instruction interleaving in particular was tried and does nothing; do not redo it. The
// obvious holes are real -- stage3's four PV MFMA have an empty shadow in the steady state
// (the mask sits behind a scalar branch only the last tile takes), and stage1's prefetch is
// ~27 issue-bound instructions with no MFMA in reach -- but every way of filling them is a
// regression, measured best-of-3 at b=32/128/256 and at page_size 1 and 4:
//   * KV prefetch into stage3's PV shadow: +1.4%. Into the QK shadow above softmax: +10%.
//     Neither is a scheduling failure. `nxt_page` is a gmem load, so consuming it forces a
//     vmcnt(0) that also drains the PREVIOUS phase's prefetch -- issue point to drain point
//     is what sets that prefetch's latency window, and at the end of stage1 it is a full
//     phase. Anywhere else it is shorter, and the KV stream cannot afford it.
//   * Reusing the phase's own barriers for the two cross-wave softmax exchanges (push the
//     row max from stage3, pull the row sum in stage2), which removes two of the seven
//     barriers per phase and merges stage1 into one scheduling region: 4-6% slower on every
//     shape. A barrier here is not the cost it looks like -- the wave waiting at one has a
//     SIMD partner that is computing -- and the exchanges are part of what keeps the two
//     wave groups' STAGGER skew from collapsing.
//   * sched_group_barrier hints (MFMA/EXP/VALU, the fmha hd192 sched_mfma_exp_valu recipe)
//     on the QK and PV regions, at several MFMA/VALU budget splits: 0 to -0.4%, i.e. noise
//     to slightly worse, with VGPR and spill unchanged. They do move instructions -- the
//     disassembly changes -- there is just nothing to gain by moving them.
//
// The one thing that did help, ~0.4%, is issuing gemm0 AFTER stage1's entry barrier rather
// than before: MFMA issued ahead of a rendezvous the wave is about to wait at buys nothing.
//
// STAGGER IS LIVE HERE and it is worth 1.0-1.5%, unlike in the 16mx8 kernel where the same
// skew measured 4-6% slower. Keeping it means every cross-wave hazard costs TWO barriers,
// not one. The two wave groups run one barrier apart -- group B takes an extra barrier in
// the prologue and group A an extra one in each stage1, so barrier instance k pairs A's
// k-th barrier with B's k-th while B sits one program point behind -- and a single barrier
// between a producer and a consumer therefore leaves them in the same instance window. This
// is not theoretical: the prologue's slot-0 prefetch had exactly one, and it silently
// corrupted ~46% of the output for any tile range of 2 or more (see the fix site below).
// Anything new that writes LDS one group reads must be checked by pairing the two barrier
// sequences, not by eyeballing that "there is a barrier in between".
//
// The paragraph below is the 16mx8 kernel's ATT analysis, kept because the register and
// opcode conclusions carry over; the occupancy numbers do not.
//
// Where the time goes (ATT, b=256 c=8192): the MAI pipe is the binding resource at ~69%
// occupancy and its idle time is diffuse -- ~30 gaps of ~25 cycles per phase, not one
// hole -- so there is no single big win left here. Two corollaries worth not relearning.
// PV spends 512 of the 832 MAI cycles per wave per phase on the HALF-RATE
// v_mfma_f32_16x16x32_fp8_fp8; the full-rate f8f6f4 unit needs K = 128 tokens, i.e.
// 4-tile softmax blocking at ~+30 VGPR, which does not fit at occupancy 2. And the
// barrier is not a cost: of its 487-cycle arrival spread, 394 is between the two waves of
// one SIMD (one computes while the other waits) and only ~46 cycles per SIMD are dead.

#include "mla_fp8fp8_def.h"

#if !defined(__HIP_DEVICE_COMPILE__) || !defined(__gfx950__)

template <class Traits>
__global__ void mla_decode_fwd_16mx1_16nx8_fp8fp8_opus_kernel(mla_kargs)
{
}

#else

#include "mla_global_load.hpp"
#include <bit>
#include <cstdint>
#include <opus/opus.hpp>

// Route the kernel through mla_decode_fwd_simple instead of mla_decode_fwd_pipelined: one
// KV tile at a time through a single LDS slot, every stage drained and barriered, nothing
// in flight across a tile boundary. It exists to measure the pipeline against a floor, so
// it deliberately keeps everything else identical -- same layouts, same MMA shapes, same
// smem_bytes() -- and changes only the overlap.
#ifndef MLA_OPUS_16MX1_SIMPLE
#define MLA_OPUS_16MX1_SIMPLE 0
#endif

using opus::operator""_I;

namespace mla_decode_fwd_16mx1_16nx8_fp8fp8 {

// Moves a wave-uniform float into an SGPR. gfx950 has no scalar float ALU, so anything
// computed from uniform inputs still lands in a VGPR and stays live there. The bit_cast is
// required, not cosmetic: __builtin_amdgcn_readfirstlane takes an int, so handing it a float
// converts the value instead of moving it, silently truncating it towards zero.
__device__ inline float readfirstlane_f32(float v)
{
    return std::bit_cast<float>(__builtin_amdgcn_readfirstlane(std::bit_cast<int>(v)));
}

namespace sched_masks {
constexpr int MFMA               = 0x08;
constexpr int VALU               = 0x02;
constexpr int DS_READ            = 0x100;
constexpr int EXP                = 0x400;
constexpr int KEEP_DS_READ_ORDER = 0x67F;
} // namespace sched_masks

// Interleave hint for a GEMM1 region: RPT times "one MFMA, then EXP_CNT exps and VALU_CNT
// VALU". Emitted as the region's last statement, before the closing fence. The MFMA of one
// mma1 call accumulate into different registers and so stall on nothing, which means the
// scheduler will not move the softmax work into their shadow unless it is asked to -- this is
// the asking. Groups it cannot fill are dropped, so over-requesting is free.
//
// Interleaving GEMM1 by preference is only safe because V is read a whole stage earlier: the
// region itself holds no tr_load, whose inline asm the DS_READ mask cannot see and for which
// SIInsertWaitcnts emits no wait. Let one of those drift past an MFMA and it reads stale LDS.
template <int RPT, int EXP_CNT, int VALU_CNT, int ID>
__device__ inline void sched_pv()
{
    opus::static_for<RPT>([&](auto) {
        __builtin_amdgcn_sched_group_barrier(sched_masks::MFMA, 1, ID);
        if constexpr(EXP_CNT > 0)
            __builtin_amdgcn_sched_group_barrier(sched_masks::EXP, EXP_CNT, ID);
        if constexpr(VALU_CNT > 0)
            __builtin_amdgcn_sched_group_barrier(sched_masks::VALU, VALU_CNT, ID);
    });
}

// --- Q gmem->LDS->register (Q rounds through LDS, unlike the 16mx8 kernel) ---
//
// 16mx1 means the Q_TILE_SIZE rows are ONE tile that every wave multiplies against its own
// KV_TILE_SIZE tokens, so Q cannot stay in registers the way it does when each wave owns a
// distinct row block: it is DMA'd to LDS once and all NUM_WARPS waves read the whole tile
// back. LDS is NUM_WARPS blocks of smem_linear_wave_q + smem_padding_32B, each block one
// wave's DMA verbatim, i.e. one W_K line per threads_d lanes:
//
//   row = waves_m * (l / threads_d % smem_n_per_wave_q) + w % waves_m
//   d   = (w / waves_m) * smem_d_per_wave_q + l / (threads_d * smem_n_per_wave_q) * W_K
//       + l % threads_d * VEC_Q
//
// The row/d split of the waves is what makes the rest fall out: waves_m waves cover the
// rows and the remaining waves_d cover d, so every W_K line of every row lands in exactly
// one block at a lane-linear offset -- the only placement buffer_load_lds can produce (see
// make_layout_sq), and one that reads back without bank conflicts (make_layout_rq).

// Q nope, d in [0, D_NOPE_SIZE) of the combined buffer: row seed = D_HEAD_SIZE (= 576, the
// combined row stride), d seed = 1. The tile is exactly BLOCK_SIZE * VEC_Q fp8, so this is
// one dwordx4 per thread with no y iteration beyond the vector itself.
template <class T>
__device__ inline auto make_layout_gq_nope(int warp_id, int lane_id)
{
    constexpr int threads_d = T::W_K / T::VEC_Q;              // lanes covering one W_K line
    constexpr int waves_d   = T::NUM_WARPS / T::smem_n_rpt_q; // waves spanning d

    constexpr auto gq_shape = opus::make_tuple(opus::number<T::smem_n_per_wave_q>{},
                                               opus::number<T::smem_n_rpt_q>{},
                                               opus::number<waves_d>{},
                                               opus::number<T::smem_d_per_wave_q / T::W_K>{},
                                               opus::number<threads_d>{},
                                               opus::number<T::VEC_Q>{});

    constexpr auto gq_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        gq_shape,
        opus::unfold_x_stride(gq_dim, gq_shape, opus::tuple{opus::number<T::D_HEAD_SIZE>{}, 1_I}),
        opus::unfold_p_coord(gq_dim,
                             opus::tuple{(lane_id / threads_d) % T::smem_n_per_wave_q,
                                         warp_id % T::smem_n_rpt_q,
                                         warp_id / T::smem_n_rpt_q,
                                         lane_id / (threads_d * T::smem_n_per_wave_q),
                                         lane_id % threads_d}));
}

// Q LDS destination, wave-uniform by necessity: buffer_load_lds takes no per-lane LDS
// address, the hardware writes lane l to dst + l * VEC_Q. That fixed placement is not a
// constraint being worked around here -- it is exactly the layout make_layout_gq was chosen
// to produce -- so the only free parameter left is which block a wave writes.
template <class T>
__device__ inline auto make_layout_sq_nope(int warp_id)
{
    constexpr int wave_block = T::smem_linear_wave_q + T::smem_padding_32B;

    constexpr auto sq_shape =
        opus::make_tuple(opus::number<T::NUM_WARPS>{}, opus::number<T::VEC_Q>{});

    constexpr auto sq_dim =
        opus::make_tuple(opus::make_tuple(opus::p_dim{}), opus::make_tuple(opus::y_dim{}));

    return opus::make_layout(
        sq_shape,
        opus::unfold_x_stride(sq_dim, sq_shape, opus::tuple{opus::number<wave_block>{}, 1_I}),
        opus::unfold_p_coord(sq_dim, opus::tuple{warp_id}));
}

// Q LDS->register, the 16x16x128 MFMA operand: lane l holds row l % W_M and, per e_k step,
// K = kk * (WARP_SIZE / W_M) * VEC_Q + l / W_M * VEC_Q + [0, VEC_Q) for kk in [0, 2).
// Reaching a given (row, d) is a matter of undoing make_layout_gq's placement, which is why
// e_k splits into two y-dims: its low half moves d by W_K, i.e. smem_n_per_wave_q lines
// inside a block, while its high half crosses into the block waves_m away that owns the next
// smem_d_per_wave_q of d. The y odometer still runs e_k, kk, vector, so the register vector
// is the same GEMM0_NOPE_E_K slices of 32 fp8 the MFMA wants.
//
// Every wave reads the whole tile, and the read is bank-conflict-free: inside one
// ds_read_b128 phase the 16 lanes' row and d-group terms (264 and 32 dwords) tile all 64
// banks exactly once.
template <class T>
__device__ inline auto make_layout_rq_nope(int lane_id)
{
    constexpr int waves_d    = T::NUM_WARPS / T::smem_n_rpt_q;
    constexpr int wave_block = T::smem_linear_wave_q + T::smem_padding_32B;

    constexpr auto rq_shape = opus::make_tuple(
        opus::number<waves_d>{},                                   // e_k high 2
        opus::number<T::smem_n_rpt_q>{},                           // row % T::smem_n_rpt_q
        opus::number<T::smem_d_per_wave_q / T::W_K>{},             // e_k low
        opus::number<T::smem_n_per_wave_q>{},                      // row / T::smem_n_rpt_q
        opus::number<T::W_M * T::W_K / T::WARP_SIZE / T::VEC_Q>{}, // kk, the operand halves
        opus::number<T::WARP_SIZE / T::W_M>{},                     // lane's d-group in W_K
        opus::number<T::VEC_Q>{});

    constexpr auto rq_dim =
        opus::make_tuple(opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
                         opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
                         opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    const int lane_m = lane_id % T::W_M;

    return opus::make_layout(
        rq_shape,
        opus::unfold_x_stride(
            rq_dim, rq_shape, opus::tuple{opus::number<wave_block>{}, opus::number<T::W_K>{}, 1_I}),
        opus::unfold_p_coord(
            rq_dim,
            opus::tuple{lane_m % T::smem_n_rpt_q, lane_m / T::smem_n_rpt_q, lane_id / T::W_M}));
}

// Q rope, d in [D_NOPE_SIZE, D_HEAD_SIZE); the caller offsets the gmem base by +D_NOPE_SIZE.
// This one skips LDS entirely: D_ROPE_SIZE is exactly (WARP_SIZE / W_M) * VEC_Q, so the
// MFMA operand's kk = 0 half is one dwordx4 per lane and the kk = 1 half is the zero padding
// that lets rope share nope's 16x16x128 MFMA -- a cleared register tile, not memory. Staging
// it through LDS would buy nothing (every wave needs all 1024 fp8, i.e. the same few cache
// lines) and could not use a DMA anyway: buffer_load_lds' fixed lane * VEC placement makes
// the rows contiguous, and 4 rows of 64 then land 256 B apart, which is 0 mod 64 banks.
//
// The caller owns the zeroing, and it must do the same for K rope: 0 * NaN is NaN, so an
// unwritten padding half on either operand poisons every score.
template <class T>
__device__ inline auto make_layout_gq_rope(int lane_id)
{
    constexpr auto gq_shape = opus::make_tuple(
        opus::number<T::W_M>{}, opus::number<T::WARP_SIZE / T::W_M>{}, opus::number<T::VEC_Q>{});

    constexpr auto gq_dim = opus::make_tuple(opus::make_tuple(opus::p_dim{}),
                                             opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        gq_shape,
        opus::unfold_x_stride(gq_dim, gq_shape, opus::tuple{opus::number<T::D_HEAD_SIZE>{}, 1_I}),
        opus::unfold_p_coord(gq_dim, opus::tuple{lane_id % T::W_M, lane_id / T::W_M}));
}

// --- KV gmem->LDS->register ---
//
// One GEMM0 step consumes KV_TILE_SIZE * NUM_WARPS tokens of the combined d = D_HEAD_SIZE
// row. Cut along d into smem_d_rpt_kv chunks of smem_d_per_wave_kv, one chunk of
// smem_n_per_wave_kv tokens is exactly one wave's DMA (smem_linear_wave_kv), so the tile is
// smem_d_rpt_kv * smem_n_rpt_kv blocks of smem_linear_wave_kv + smem_padding_32B and
// smem_d_rpt_kv is both the d repeat and the number of buffer_load_lds per thread. Because
// the buffer is combined and D_HEAD_SIZE is a whole number of chunks, rope is just the last
// chunk: no separate tensor, layout or waitcnt budget for it.
//
// Inside a block the placement is buffer_load_lds': lane l writes l * VEC_KV, i.e. it owns
// block row l / threads_d and d-group l % threads_d. The one free choice is which token each
// row holds, and it is not the obvious one. The blocks are grouped in runs of
// blk_grp = smem_n_rpt_kv / 2, each run taking one grp_toks = blk_grp * smem_n_per_wave_kv
// stretch of the tile, tokens are dealt round-robin across a run, and the slot a token gets
// inside its block is the block row rotated by one bit:
//
//   token = grp_toks * (b / blk_grp) + blk_grp * m + b % blk_grp      (block b, slot m)
//   m     = row / 2 + (row % 2) * SWZ_TOK_BIT                        (row = l / threads_d)
//
// A reading wave's W_N consecutive tokens then spread over blk_grp blocks x W_N / blk_grp
// rows, and those two strides are 8 and 32 banks -- one block is 264 dwords thanks to the
// 32 B pad, and the reader steps two rows at a time -- so one ds_read_b128 phase tiles all
// 64 banks exactly once (make_layout_rkv). Dealing the tokens in contiguous runs, or keeping
// the slot equal to the row, costs a 2-way conflict on every K read.

// Distributes a tile's KV_TILE_SIZE * NUM_WARPS tokens across the BLOCK_SIZE threads, one
// each, so every thread computes the page/token base offset of the token it DMAs -- the deal
// above, with the token axis of the load folded into that offset. The caller adds the tile's
// own tile_idx * KV_TILE_SIZE * NUM_WARPS.
template <class T>
__device__ inline auto make_layout_kv_indices(int warp_id, int lane_id)
{
    constexpr int threads_d = T::smem_d_per_wave_kv / T::VEC_KV; // lanes covering one chunk
    constexpr int blk_grp   = T::smem_n_rpt_kv / 2;              // blocks sharing a token run

    // Above PAGE_SIZE 1 this layout addresses a block table, so its offset must be the
    // token's page (token / PAGE_SIZE) rather than the token. The deal makes that a matter
    // of shrinking one extent: the token index comes out
    //   tok = 64 * (warp_id / blk_grp) + 32 * (row % 2) + 4 * (row / 2) + warp_id % blk_grp
    // (strides derived from the shape below; verified against the built layout), and only
    // the last term is finer than blk_grp = 4. Dividing that extent and its coordinate by
    // PAGE_SIZE halves every derived stride with it, which is exactly tok / PAGE_SIZE --
    // exact because the other three strides are all multiples of 4. The leftover
    // tok % PAGE_SIZE is warp_id % PAGE_SIZE, wave-uniform, and the caller folds it into
    // the page offset. PAGE_SIZE > 4 would make that residue per-lane; hence the cap.
    static_assert(blk_grp % T::PAGE_SIZE == 0, "the token deal must split on PAGE_SIZE");

    constexpr auto kv_indices_shape =
        opus::make_tuple(opus::number<T::smem_n_rpt_kv / blk_grp>{},             // block group
                         opus::number<T::smem_n_per_wave_kv / T::SWZ_TOK_BIT>{}, // row % 2
                         opus::number<T::SWZ_TOK_BIT>{},                         // row / 2
                         opus::number<blk_grp / T::PAGE_SIZE>{},                 // block in group
                         1_I);

    constexpr auto kv_indices_dim = opus::make_tuple(opus::make_tuple(
        opus::p_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    const int row = lane_id / threads_d;

    return opus::make_layout(
        kv_indices_shape,
        opus::unfold_x_stride(kv_indices_dim, kv_indices_shape, opus::tuple{1_I}),
        opus::unfold_p_coord(
            kv_indices_dim,
            opus::tuple{warp_id / blk_grp, row % 2, row / 2, (warp_id % blk_grp) / T::PAGE_SIZE}));
}

// KV global source, d in [0, D_HEAD_SIZE) of the combined buffer. The token dimension is
// folded into the per-thread page offset (make_layout_kv_indices), so this layout is only
// the d walk: smem_d_rpt_kv chunks, one buffer_load_lds each, seed smem_d_per_wave_kv.
//
// The d-group each lane fetches is where the V-read bank swizzle has to live: buffer_load_lds
// takes no per-lane LDS address, so the slot a lane fills is fixed and the only way to put a
// different d-group in it is to fetch a different one. Rows in the upper half of a block
// (T::SWZ_TOK_BIT) therefore hold their d-groups XORed by one, which is what the V transpose
// read will ask those slots for; make_layout_rkv XORs back.
template <typename T>
__device__ inline auto make_layout_gkv(int lane_id)
{
    constexpr int threads_d = T::smem_d_per_wave_kv / T::VEC_KV;

    constexpr auto gkv_shape = opus::make_tuple(
        opus::number<T::smem_d_rpt_kv>{}, opus::number<threads_d>{}, opus::number<T::VEC_KV>{});

    constexpr auto gkv_dim = opus::make_tuple(opus::make_tuple(opus::y_dim{}),
                                              opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    const int row   = lane_id / threads_d;
    const int d_grp = (lane_id % threads_d) ^ ((row & T::SWZ_TOK_BIT) ? 1 : 0);

    return opus::make_layout(
        gkv_shape,
        opus::unfold_x_stride(
            gkv_dim, gkv_shape, opus::tuple{opus::number<T::smem_d_per_wave_kv>{}, 1_I}),
        opus::unfold_p_coord(gkv_dim, opus::tuple{d_grp}));
}

// KV LDS destination, wave-uniform because buffer_load_lds' dst is: the wave picks its block
// within each chunk's smem_n_rpt_kv blocks, the hardware does the rest.
template <typename T>
__device__ inline auto make_layout_skv(int warp_id)
{
    constexpr auto skv_shape = opus::make_tuple(opus::number<T::smem_d_rpt_kv>{},
                                                opus::number<T::smem_n_rpt_kv>{},
                                                opus::number<T::VEC_KV>{});

    constexpr auto skv_dim = opus::make_tuple(opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
                                              opus::make_tuple(opus::y_dim{}));

    return opus::make_layout(
        skv_shape,
        opus::unfold_x_stride(
            skv_dim,
            skv_shape,
            opus::tuple{opus::number<T::smem_linear_wave_kv + T::smem_padding_32B>{}, 1_I}),
        opus::unfold_p_coord(skv_dim, opus::tuple{warp_id}));
}

// KV LDS->register, the K side of GEMM0's 16x16x128 MFMA: lane l holds token l % W_N of this
// wave's KV_TILE_SIZE and, per e_k step, d = kk * smem_d_per_wave_kv + l / W_N * VEC_KV +
// [0, VEC_KV) for kk in [0, W_K / smem_d_per_wave_kv). Reaching that is inverting the deal
// above; since a wave's tokens are W_N consecutive ones, the inverse splits into warp bits
// and token bits that never mix, so it is one product layout with no compile-time variant:
//
//   block = blk_grp * (warp_id / blk_grp) + token % blk_grp
//   row   = SWZ_TOK_BIT * (warp_id % 2) + 2 * (token / blk_grp) + (warp_id % blk_grp) / 2
//
// The gmem-side swizzle collapses with it: row >= SWZ_TOK_BIT holds for a whole wave at a
// time, so the XOR is just warp_id % 2 on the lane's d-group.
//
// One MFMA operand half is exactly one d chunk -- smem_d_per_wave_kv is (WARP_SIZE / W_N) *
// VEC_KV -- so the y odometer needs no separate kk, and rope needs no read of its own: chunk
// e_k * (W_K / smem_d_per_wave_kv) + kk lands in the right half of the right k-step by itself,
// so walking all smem_d_rpt_kv chunks brings the whole combined row in as the GEMM0_E_K k-steps
// the MFMA wants, contiguous up to the padding half the caller clears (make_layout_gq_rope).
template <typename T>
__device__ inline auto make_layout_rk(int warp_id, int lane_id)
{
    constexpr int blk_grp    = T::smem_n_rpt_kv / 2;
    constexpr int wave_block = T::smem_linear_wave_kv + T::smem_padding_32B;

    constexpr auto rk_shape =
        opus::make_tuple(opus::number<T::smem_d_rpt_kv>{},                       // d chunk
                         opus::number<T::smem_n_rpt_kv / blk_grp>{},             // block group
                         opus::number<blk_grp>{},                                // block in group
                         opus::number<T::smem_n_per_wave_kv / T::SWZ_TOK_BIT>{}, // row high bit
                         opus::number<T::W_N / blk_grp>{},                       // row mid bits
                         opus::number<blk_grp / 2>{},                            // row low bit
                         opus::number<T::WARP_SIZE / T::W_N>{},                  // lane's d-group
                         opus::number<T::VEC_KV>{});

    constexpr auto rk_dim =
        opus::make_tuple(opus::make_tuple(opus::y_dim{}),
                         opus::make_tuple(opus::p_dim{}, opus::p_dim{}),
                         opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
                         opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    const int lane_n = lane_id % T::W_N;

    return opus::make_layout(
        rk_shape,
        opus::unfold_x_stride(rk_dim,
                              rk_shape,
                              opus::tuple{opus::number<T::smem_n_rpt_kv * wave_block>{},
                                          opus::number<wave_block>{},
                                          opus::number<T::smem_d_per_wave_kv>{},
                                          1_I}),
        opus::unfold_p_coord(rk_dim,
                             opus::tuple{warp_id / blk_grp,
                                         lane_n % blk_grp,
                                         warp_id % 2,
                                         lane_n / blk_grp,
                                         (warp_id % blk_grp) / 2,
                                         (lane_id / T::W_N) ^ (warp_id % 2)}));
}

// --- P register->LDS->register ---
//
// GEMM1 wants P as its A operand, Q_TILE_SIZE query rows by the tile's whole KV_TILE_SIZE *
// NUM_WARPS tokens, but GEMM0 leaves a wave holding only its own KV_TILE_SIZE columns of it:
// lane l has query row l % W_M and tokens (l / W_M) * VEC_WRITE_P + [0, VEC_WRITE_P), the C
// pack (see attn_mask_kv_tile). So P round-trips LDS, one block per wave, each written by its
// owner and then read whole by all NUM_WARPS.
//
// A block is that wave's W_M rows of KV_TILE_SIZE tokens, tokens contiguous within a row: the C
// pack is then one dword of one row, so the write is a single ds_write_b32, and a reader takes a
// row's two 8 B halves with one ds_read_b64 each.
//
// Both sides are bank-conflict free, which takes three swizzles, all measured on gfx950 (the LDS
// there is 32 banks x 4 B at 32 lanes per cycle for b32, and 32 banks x 8 B at 32 lanes per cycle
// for b64 -- so a b64 read phase spans two blocks, while a write phase stays inside one):
//
//  - the two 8 B halves of a row are swapped for the upper W_M / 2 rows. This is what fixes the
//    write: a phase's banks are {4r, 4r + 1} otherwise, and 4r wraps every 8 rows, so r and
//    r + 8 collide 2-way while half the banks idle. With the swap the upper rows contribute
//    {4r + 2, 4r + 3} instead and the 32 lanes tile the 32 banks. No row permutation can do this,
//    the set being {4 * perm(r) + pack} either way.
//  - rows are permuted onto 16 * (P_SWZ_ROW_MUL * r % W_M), which is what keeps the read free
//    once the halves move: a read phase needs its 32 slots distinct across both the 16 rows and
//    the two blocks.
//  - the block pitch is 128 mod 256, so the two blocks of a read phase sit 16 slots apart. That
//    is forced: the slot sets of one block must be disjoint from themselves shifted by
//    pitch / 8, and with rows permuted onto the 16 B grid only a shift of 16 admits a solution.
//
// Costs 2 more DS instructions per tile than the dense pitch-256 layout would (4 b64 against 2
// b128, same 8 cycles either way), and each read has to be fenced apart or the load/store
// optimiser pairs them into ds_read2_b64, which reads 16 B per lane out of one 8 B slot pair and
// conflicts 2-way. In exchange the write drops from 4 cycles to 2 and the workgroup's whole P
// round trip from 12 cycles and 16 conflicts per wave-tile to 10 and none.

// Where query row r sits inside a block, and whether its two halves are swapped.
template <typename T>
__device__ inline int p_row_perm(int r)
{
    return (T::P_SWZ_ROW_MUL * r) % T::W_M;
}

template <typename T>
__device__ inline int p_half_swz(int r)
{
    return r >= T::W_M / 2 ? 1 : 0;
}

// P LDS destination: the wave's own block, one VEC_WRITE_P dword per lane.
template <typename T>
__device__ inline auto make_layout_sp(int warp_id, int lane_id)
{
    static_assert((T::WARP_SIZE / T::W_M) * T::VEC_WRITE_P == T::KV_TILE_SIZE,
                  "the C packs of one lane group must tile a block row");
    static_assert(T::W_M == 16 && T::P_SWZ_ROW_MUL % 2 == 1, "row_off must stay a permutation");

    constexpr auto sp_shape = opus::make_tuple(opus::number<T::NUM_WARPS>{},          // block
                                               opus::number<T::W_M>{},                // query row
                                               opus::number<T::WARP_SIZE / T::W_M>{}, // C pack
                                               opus::number<T::VEC_WRITE_P>{});

    constexpr auto sp_dim =
        opus::make_tuple(opus::make_tuple(opus::p_dim{}), // 8, 16, 4, 4
                         opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    const int row = lane_id % T::W_M;
    // the pack XOR is the row's half swap: 2 packs are one 8 B half
    return opus::make_layout(
        sp_shape,
        opus::unfold_x_stride(sp_dim, sp_shape, opus::tuple{opus::number<T::smem_p_pitch>{}, 1_I}),
        opus::unfold_p_coord(sp_dim,
                             opus::tuple{warp_id,
                                         p_row_perm<T>(row),
                                         (lane_id / T::W_M) ^ (2 * p_half_swz<T>(row))}));
}

// P LDS->register, the A side of GEMM1's 16x16x128 MFMA: lane l holds query row l % W_M and
// tokens kk * (W_K / 2) + l / W_M * KV_TILE_SIZE + [0, KV_TILE_SIZE) over both operand halves.
// One layout covers the whole 16 x W_K tile: kk and the row half are y-iters; the half's byte
// offset 8 * (half ^ swz(row)) is folded into a lane-dependent half stride (+/- VEC_READ_P)
// and a +VEC_READ_P base on upper rows, the same trick as make_layout_rv's en_lo swizzle.
template <typename T>
__device__ inline auto make_layout_rp(int lane_id)
{
    constexpr int blks   = T::WARP_SIZE / T::W_M; // blocks a lane covers per operand half
    constexpr int rounds = T::NUM_WARPS / blks;   // kk halves of the operand
    static_assert(rounds * blks * T::KV_TILE_SIZE == T::W_K, "P must fill one MFMA k step");
    static_assert(2 * T::VEC_READ_P == T::KV_TILE_SIZE, "a row is two halves");

    constexpr auto rp_shape = opus::make_tuple(opus::number<rounds>{},         // kk
                                               opus::number<blks>{},           // block in round
                                               opus::number<T::W_M>{},         // query row slot
                                               opus::number<2>{},              // 8 B half
                                               opus::number<T::VEC_READ_P>{}); // tokens

    constexpr auto rp_dim = opus::make_tuple(opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
                                             opus::make_tuple(opus::p_dim{}),
                                             opus::make_tuple(opus::y_dim{}, opus::y_dim{}));

    const int row         = lane_id % T::W_M;
    const int half_swz    = p_half_swz<T>(row);
    const int half_stride = half_swz ? -T::VEC_READ_P : T::VEC_READ_P;

    constexpr int s_kk = blks * T::smem_p_pitch;

    auto u = opus::make_layout(
        rp_shape,
        opus::make_tuple(opus::number<s_kk>{},
                         opus::number<T::smem_p_pitch>{},
                         opus::number<2 * T::VEC_READ_P>{},
                         half_stride,
                         1_I),
        opus::unfold_p_coord(rp_dim, opus::tuple{lane_id / T::W_M, p_row_perm<T>(row)}));
    if(half_swz)
        u += T::VEC_READ_P;
    return u;
}

// --- V LDS->register transpose read (ds_read_b64_tr_b8) ---

template <class T>
__device__ inline auto make_layout_rv(int warp_id, int lane_id)
{
    constexpr int halves     = T::W_N / T::VEC_TR_V; // 8 B halves of a d-tile, i.e. lanes per token
    constexpr int grp_toks   = T::W_N / halves;      // 8, tokens one instruction brings in
    constexpr int blk_grp    = T::smem_n_rpt_kv / 2; // 4, blocks sharing a token run
    constexpr int wave_block = T::smem_linear_wave_kv + T::smem_padding_32B;

    static_assert(T::SLICE_D == T::smem_d_per_wave_kv && T::NUM_D_SLICES == T::SLICE_D / T::VEC_KV,
                  "a wave's GEMM1 d must be one KV d chunk, one d-tile per d-group");
    static_assert(T::NUM_D_SLICES == 4, "d-tile index splits into two y-dims of 2");
    static_assert(T::W_N == T::smem_n_per_wave_kv && grp_toks == 2 * blk_grp &&
                      T::SWZ_TOK_BIT == T::smem_n_per_wave_kv / 2 && T::W_N == 16,
                  "the row decomposition below is the token deal's own factorisation");

    // One layout covers the wave's whole GEMM1 B operand: W_K tokens x W_N d x NUM_D_SLICES
    // d-tiles. warp_id picks this wave's d chunk [0, SLICE_D); en_hi/en_lo y-iters come
    // before kk so tr_load fills v_v e_n-major. The DMA swizzle EN ^ (grp % 2) is folded
    // into a lane-dependent en_lo stride (+/- VEC_KV) and a +VEC_KV base on odd groups.
    constexpr auto rv_shape = opus::make_tuple(opus::number<T::smem_d_rpt_kv>{}, // d chunk
                                               opus::number<2>{},                // en_hi
                                               opus::number<2>{},                // en_lo
                                               opus::number<T::smem_n_rpt_kv / blk_grp>{}, // kk
                                               opus::number<blk_grp>{},      // block in run
                                               opus::number<2>{},            // row SWZ_TOK_BIT
                                               opus::number<2>{},            // row j / grp_toks
                                               opus::number<2>{},            // row t / blk_grp
                                               opus::number<2>{},            // row deal rot
                                               opus::number<halves>{},       // d-tile 8 B half
                                               opus::number<T::VEC_TR_V>{}); // transpose read

    constexpr auto rv_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    const int grp          = lane_id / T::W_N;
    const int tok          = (lane_id % T::W_N) / halves;
    const int en_lo_stride = (grp & 1) ? -T::VEC_KV : T::VEC_KV;

    constexpr int s_chunk = T::smem_n_rpt_kv * wave_block;
    constexpr int s_kk    = blk_grp * wave_block;
    constexpr int s_en_hi = 2 * T::VEC_KV;

    auto u = opus::make_layout(
        rv_shape,
        opus::make_tuple(opus::number<s_chunk>{},
                         opus::number<s_en_hi>{},
                         en_lo_stride,
                         opus::number<s_kk>{},
                         opus::number<wave_block>{},
                         opus::number<T::smem_d_per_wave_kv * 8>{},
                         opus::number<T::smem_d_per_wave_kv * 4>{},
                         opus::number<T::smem_d_per_wave_kv * 2>{},
                         opus::number<T::smem_d_per_wave_kv>{},
                         opus::number<T::VEC_TR_V>{},
                         1_I),
        opus::unfold_p_coord(
            rv_dim,
            opus::tuple{
                warp_id, tok % blk_grp, grp % 2, tok / blk_grp, grp / 2, lane_id % halves}));
    if(grp & 1)
        u += T::VEC_KV;
    return u;
}

// O register->gmem store. stride_o_h is a parameter because the same layout serves both
// destinations: the real output uses kargs.stride_o_h, while the split-KV partial writes
// a densely packed D_NOPE_SIZE-strided o_accum.
//
// 16mx1 splits O along d, not along the rows: GEMM1 contracts over every wave's tokens, so
// all NUM_WARPS waves carry the same Q_TILE_SIZE rows and warp_id picks which SLICE_D of
// the output d this one owns -- GEMM1_E_N tiles of W_M x W_N, i.e. 16 x 16 x 4 values,
// T_N * GEMM1_E_N * W_N == D_NOPE_SIZE across the block. Hence warp_id sits on the d side
// here (in 16mx8 it sits on the row side, and a wave spans all of d instead).
template <class T>
__device__ inline auto make_layout_o(int warp_id, int lane_id, int stride_o_h)
{
    static_assert(T::T_N * T::GEMM1_E_N * T::W_N == T::D_NOPE_SIZE,
                  "the waves' d slices must tile the output d exactly once");

    constexpr auto o_block_shape =
        opus::make_tuple(opus::number<T::GEMM1_E_M>{}, // e_m
                         opus::number<T::W_M>{},       // query row
                         opus::number<T::T_N>{},       // wave's d slice
                         opus::number<T::GEMM1_E_N>{}, // d-tile in it
                         opus::number<T::W_M * T::W_N / T::WARP_SIZE / T::VEC_O>{},
                         opus::number<T::WARP_SIZE / T::W_M>{}, // lane's d group
                         opus::number<T::VEC_O>{});

    constexpr auto o_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(
            opus::p_dim{}, opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        o_block_shape,
        opus::unfold_x_stride(o_block_dim, o_block_shape, opus::tuple{stride_o_h, 1_I}),
        opus::unfold_p_coord(o_block_dim,
                             opus::tuple{lane_id % T::W_M, warp_id, lane_id / T::W_M}));
}

// --- softmax / scaling helpers ---
// Two reductions per row, not one. Within a wave, W_M = 16 against 64-wide waves puts a row
// across four lane groups, so it takes permlane32_swap followed by permlane16_swap (the
// d192 FMHA kernel, at W_M = 32, gets away with the 32-swap alone). Across waves, 16mx1
// gives each wave only KV_TILE_SIZE of the NUM_WARPS * KV_TILE_SIZE tokens a GEMM0 step
// consumes, so what that leaves is a *partial* row max -- and GEMM1 contracts over all of
// the tokens, so every wave's O accumulator needs the merged one. The waves swap through
// s_m: lane writes its row's slot `row * T_N + warp_id`, then reads the whole T_N-wide row
// back and folds it, so all NUM_WARPS waves leave holding the same value.
//
// The caller merges the raw (unscaled) max; the temperature is applied to the single scalar
// afterwards, which commutes with max because it is positive.
template <typename T, typename V, typename S>
__device__ inline typename T::D_ACC attn_row_max(const V& v_s, S& s_m, int warp_id, int lane_id)
{
    using D_ACC                   = typename T::D_ACC;
    constexpr opus::index_t s_len = opus::vector_traits<V>::size();
    D_ACC row_max                 = opus::numeric_limits<D_ACC>::lowest();
    opus::static_for<s_len>([&](auto i) { row_max = max(row_max, v_s[i.value]); });

    opus::vector_t<opus::u32_t, 2> res32 = __builtin_amdgcn_permlane32_swap(
        std::bit_cast<opus::u32_t>(row_max), std::bit_cast<opus::u32_t>(row_max), false, true);
    row_max = max(std::bit_cast<float>(res32.x), std::bit_cast<float>(res32.y));
    opus::vector_t<opus::u32_t, 2> res16 = __builtin_amdgcn_permlane16_swap(
        std::bit_cast<opus::u32_t>(row_max), std::bit_cast<opus::u32_t>(row_max), false, true);
    row_max = max(std::bit_cast<float>(res16.x), std::bit_cast<float>(res16.y));

    const int row = lane_id % T::W_M;
    opus::store(s_m, row_max, row * T::T_N + warp_id);
    opus::s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    auto max_warps = opus::load<T::T_N>(s_m, row * T::T_N);
    opus::static_for<T::T_N>([&](auto i) { row_max = max(row_max, max_warps[i.value]); });
    return row_max;
}

// Fused `v_s * scale - row_max`, one v_fma per element. The caller reduces the *raw*
// scores and scales that single scalar instead (scale > 0, so max commutes with it):
// pre-scaling the tile would cost a whole extra pass of muls -- the subtract fuses the
// scale back in anyway -- and would put them on the critical path into the reduction.
template <typename T, typename V>
__device__ inline void
attn_scale_sub_row(V& v_s, typename T::D_ACC scale, typename T::D_ACC row_max)
{
    constexpr opus::index_t s_len = opus::vector_traits<V>::size();
    opus::static_for<s_len>(
        [&](auto i) { v_s[i.value] = __builtin_fmaf(v_s[i.value], scale, -row_max); });
}

template <typename T, opus::index_t Offset, opus::index_t Count, typename V>
__device__ inline void attn_exp2_slice(V& v_s)
{
    opus::static_for<Count>([&](auto i) {
        constexpr opus::index_t idx = Offset + i.value;
        v_s[idx]                    = __builtin_amdgcn_exp2f(v_s[idx]);
    });
}

// Balanced tree, not the `row_sum += v_s[i]` chain the loop shape suggests: float addition
// does not reassociate, so the chain compiles to s_len dependent v_add_f32 back to back --
// the trace measured those eight at 52 cycles, all of it on the critical path into the
// permlane swaps. The tree is 3 deep instead of 8 for the same instruction count. It sums
// in a different order and therefore rounds differently, which is harmless here: every
// term is a positive exp2 result.
//
// Then the same cross-wave merge attn_row_max does, through s_l: the l_row that normalises
// O has to cover every wave's tokens, since GEMM1 contracted over all of them.
template <typename T, typename V, typename S>
__device__ inline typename T::D_ACC attn_row_sum(const V& v_s, S& s_l, int warp_id, int lane_id)
{
    using D_ACC                   = typename T::D_ACC;
    constexpr opus::index_t s_len = opus::vector_traits<V>::size();
    static_assert(s_len > 0 && (s_len & (s_len - 1)) == 0, "row sum tree wants a power of two");
    D_ACC part[s_len];
    opus::static_for<s_len>([&](auto i) { part[i.value] = v_s[i.value]; });
    opus::static_for<s_len>([&](auto lvl) {
        constexpr opus::index_t half = s_len >> (lvl.value + 1);
        if constexpr(half >= 1)
        {
            opus::static_for<half>([&](auto i) { part[i.value] += part[i.value + half]; });
        }
    });
    D_ACC row_sum = part[0];

    opus::vector_t<opus::u32_t, 2> res32 = __builtin_amdgcn_permlane32_swap(
        std::bit_cast<opus::u32_t>(row_sum), std::bit_cast<opus::u32_t>(row_sum), false, true);
    row_sum = std::bit_cast<float>(res32.x) + std::bit_cast<float>(res32.y);
    opus::vector_t<opus::u32_t, 2> res16 = __builtin_amdgcn_permlane16_swap(
        std::bit_cast<opus::u32_t>(row_sum), std::bit_cast<opus::u32_t>(row_sum), false, true);
    row_sum = std::bit_cast<float>(res16.x) + std::bit_cast<float>(res16.y);

    const int row = lane_id % T::W_M;
    opus::store(s_l, row_sum, row * T::T_N + warp_id);
    opus::s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    auto sum_warps = opus::load<T::T_N>(s_l, row * T::T_N);
    row_sum        = D_ACC(0.0f);
    opus::static_for<T::T_N>([&](auto i) { row_sum += sum_warps[i.value]; });
    return row_sum;
}

template <typename T, typename V>
__device__ inline void scale_output_tile(V& v_o, typename T::D_ACC scale)
{
    constexpr opus::index_t o_len = opus::vector_traits<V>::size();
    opus::static_for<o_len>([&](auto i) { v_o[i.value] *= scale; });
}

// Pin the O accumulator as a scheduling/materialization fence, chunked into 8-lane groups
// so each `"+v"` operand can be allocated (a single one on the whole 128-VGPR v_o cannot).
template <typename V>
__device__ inline void pin_output_tile(V& v_o)
{
    using chunk_t = opus::vector_t<float, 8>;
    constexpr int num_chunks =
        opus::vector_traits<V>::size() / opus::vector_traits<chunk_t>::size();
    static_assert(opus::vector_traits<V>::size() % opus::vector_traits<chunk_t>::size() == 0);
    auto* chunks = reinterpret_cast<chunk_t*>(&v_o);
#pragma unroll
    for(int i = 0; i < num_chunks; i++)
    {
        asm volatile("" : "+v"(chunks[i])::);
    }
}

// --- score masking (out-of-range KV columns and, for CAUSAL, the diagonal) ---
template <int THR_X, int THR_Y>
__device__ inline void attn_mask_vec2_imm(opus::u32_t rel_vgpr,
                                          opus::u32_t neg_inf_vgpr,
                                          opus::u32_t& x_ref,
                                          opus::u32_t& y_ref)
{
    uint64_t x_mask, y_mask;
    asm volatile("v_cmp_lt_i32_e64 %0, %6, %7\n\t"
                 "v_cmp_lt_i32_e64 %1, %6, %9\n\t"
                 "v_cndmask_b32_e64 %2, %4, %8, %0\n\t"
                 "v_cndmask_b32_e64 %3, %5, %8, %1\n\t"
                 : "=s"(x_mask), "=s"(y_mask), "=v"(x_ref), "=v"(y_ref)
                 : "v"(x_ref), "v"(y_ref), "v"(rel_vgpr), "n"(THR_X), "v"(neg_inf_vgpr), "n"(THR_Y)
                 : "vcc");
}

// Last KV position the diagonal lets this wave's query rows attend to. Loop-invariant and
// costs an integer division, so it is evaluated once on entry (the pipelined body inlines
// this expression); the cheap valid_kv_len bound stays at the point of use instead, where
// it need not be kept live across the pipeline.
template <typename T>
__device__ inline int causal_kv_bound(int causal_diagonal, int nhead, int warp_id)
{
    return (warp_id * T::W_M) / nhead + causal_diagonal;
}

// Masks every score column past `last_valid_kv_pos` to -inf. `kv_base_pos` is the absolute
// KV position of this wave's first score column, which the caller has to supply because a
// tile's KV_TILE_SIZE * NUM_WARPS tokens are dealt out KV_TILE_SIZE contiguous ones per
// wave -- the score tile a wave holds is its own slice of the tile, not the whole of it.
template <typename T, typename V>
__device__ inline void
attn_mask_kv_tile(V& v_s, int last_valid_kv_pos, int kv_base_pos, opus::u32_t neg_inf_v)
{
    using D_ACC    = typename T::D_ACC;
    using D_ACC_X2 = opus::vector_t<D_ACC, 2>;
    using U32_X2   = opus::vector_t<opus::u32_t, 2>;

    constexpr int elems_per_wave_tile = (T::W_M * T::W_N) / T::WARP_SIZE;
    constexpr int c_pack              = 4;
    constexpr int c_rept              = elems_per_wave_tile / c_pack;
    constexpr int c_rept_stride       = (T::WARP_SIZE / T::W_M) * c_pack;

    const int k_start_pos = kv_base_pos;
    int lane_id           = opus::thread_id_x() % T::WARP_SIZE;
    asm volatile("" : "+v"(lane_id));
    const int lane_group = lane_id / T::W_M;

    opus::static_for<T::GEMM0_E_N>([&](auto i_n) {
        constexpr int base_idx = i_n.value * elems_per_wave_tile;
        const int k_pos        = k_start_pos + i_n.value * T::W_N + lane_group * c_pack;
        const opus::u32_t rel  = static_cast<opus::u32_t>(last_valid_kv_pos - k_pos);

        opus::static_for<c_rept>([&](auto i_rept) {
            constexpr int rept_base_idx = base_idx + i_rept.value * c_pack;
            constexpr int thr_base      = i_rept.value * c_rept_stride;
            opus::static_for<c_pack / 2>([&](auto i_pair) {
                constexpr int idx   = rept_base_idx + i_pair.value * 2;
                constexpr int thr_x = thr_base + i_pair.value * 2;
                constexpr int thr_y = thr_x + 1;

                auto pair_acc     = opus::slice(v_s, opus::number<idx>{}, opus::number<idx + 2>{});
                auto pair_bits    = __builtin_bit_cast(U32_X2, pair_acc);
                opus::u32_t x_ref = pair_bits[0];
                opus::u32_t y_ref = pair_bits[1];
                attn_mask_vec2_imm<thr_x, thr_y>(rel, neg_inf_v, x_ref, y_ref);
                pair_bits[0] = x_ref;
                pair_bits[1] = y_ref;
                opus::set_slice(v_s,
                                __builtin_bit_cast(D_ACC_X2, pair_bits),
                                opus::number<idx>{},
                                opus::number<idx + 2>{});
            });
        });
    });
}

// --- Pipelined KV-tile loop for one work item (see the stage map at the top) ---
// Q, the O accumulator and the online-softmax state (m_row / l_row) are owned by the
// caller and passed by reference, so a split-KV request can run several tile ranges into
// the same accumulator.
template <class Traits, bool STAGGER, class VO>
__device__ __attribute__((always_inline)) void
mla_decode_fwd_pipelined(mla_kargs kargs,
                         int kv_ind_ptr_s,
                         int valid_kv_len,
                         int tile_begin,
                         int tile_end,
                         char* smem_buffer,
                         int q_len_ptr_s,
                         int q_len,
                         VO& v_o,
                         typename Traits::D_ACC& m_row,
                         typename Traits::D_ACC& l_row,
                         float temperature_scale,
                         int causal_diagonal)
{
    using namespace opus;
    using T     = opus::remove_cvref_t<Traits>;
    using D_Q   = typename T::D_Q;
    using D_K   = typename T::D_K;
    using D_V   = typename T::D_V;
    using D_ACC = typename T::D_ACC;

    int lane_id = thread_id_x() % T::WARP_SIZE;
    asm volatile("" : "+v"(lane_id));
    const int warp_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);

    int diag_kv_bound = 0;
    if constexpr(T::CAUSAL)
    {
        // Every wave carries the same Q_TILE_SIZE rows (16mx1 tiles N, not M), and the
        // dispatch only routes nhead == Q_TILE_SIZE with one query token per request here,
        // so the whole tile sits on a single diagonal -- no per-wave row offset.
        diag_kv_bound = causal_diagonal;
    }

    const int q_gmem_offset = q_len_ptr_s * kargs.stride_q_b;
    auto g_q_nope = make_gmem(reinterpret_cast<const D_Q*>(kargs.q_buffer_ptr) + q_gmem_offset,
                              q_len * kargs.stride_q_b * sizeof(D_Q));
    auto g_q_rope =
        make_gmem(reinterpret_cast<const D_Q*>(kargs.q_buffer_ptr) + q_gmem_offset + T::D_NOPE_SIZE,
                  q_len * kargs.stride_q_b * sizeof(D_Q));

    const D_K* kv_base = reinterpret_cast<const D_K*>(kargs.kv_buffer_ptr);
    // kv_indices always stays on a descriptor -- it is int-indexed and never large. It holds
    // one entry per page, and valid_kv_len is in tokens, so the bound is the page count; the
    // last page may be partial, hence ceil.
    auto g_kv_indices = make_gmem(kargs.kv_indices + kv_ind_ptr_s,
                                  ceil_div(valid_kv_len, T::PAGE_SIZE) * sizeof(int));

    // Q sits in front of the KV ring and outside the slot rotation: it is written once per
    // work item and read by every wave for the whole tile range. smem_q_bytes is a multiple
    // of 128, so the KV blocks behind it keep the 32 B alignment make_layout_rv's swizzle
    // needs.
    auto s_q          = make_smem(reinterpret_cast<D_Q*>(smem_buffer));
    smem<D_K> s_kv[2] = {make_smem(reinterpret_cast<D_K*>(smem_buffer) + T::smem_q_padding_bytes),
                         make_smem(reinterpret_cast<D_K*>(smem_buffer) + T::smem_q_padding_bytes +
                                   T::smem_kv_bytes)};
    // P costs no LDS of its own: it aliases Q, which is dead once the prologue has read it into
    // v_q, with barriers between that read and the first P write covering the cross-wave WAR.
    // No slots either -- a tile's scores are written and read back inside one phase.
    auto s_p = make_smem(reinterpret_cast<D_K*>(smem_buffer));
    // Cross-wave softmax exchange, behind P in that same dead Q region: one D_ACC per
    // (query row, wave) for the max and, right after it, one for the sum.
    auto s_m = make_smem(reinterpret_cast<D_ACC*>(smem_buffer + T::smem_ml_offset_bytes));
    auto s_l = make_smem(reinterpret_cast<D_ACC*>(smem_buffer + T::smem_ml_offset_bytes) +
                         T::smem_ml_elems);

    auto u_gq_nope = make_layout_gq_nope<T>(warp_id, lane_id);
    auto u_sq_nope = make_layout_sq_nope<T>(warp_id);
    auto u_rq_nope = make_layout_rq_nope<T>(lane_id);
    auto u_gq_rope = make_layout_gq_rope<T>(lane_id);

    auto u_kv_indices = make_layout_kv_indices<T>(warp_id, lane_id);
    auto u_gkv        = make_layout_gkv<T>(lane_id);
    auto u_skv        = make_layout_skv<T>(warp_id);
    auto u_rk         = make_layout_rk<T>(warp_id, lane_id);

    auto u_sp = make_layout_sp<T>(warp_id, lane_id);
    auto u_rp = make_layout_rp<T>(lane_id);

    // Whole GEMM1 B operand in one layout: warp_id owns [0, SLICE_D) and the four W_N d-tiles
    // are y-iters folded from the DMA swizzle (make_layout_rv).
    auto u_rv = make_layout_rv<T>(warp_id, lane_id);

    // Under LARGE_KV the K handle is a bare 64-bit pointer for global_load_lds, resolved down
    // to this lane's slot in the tile; otherwise it is a buffer descriptor, whose 32-bit
    // num_records caps the cache at 4 GiB and which carries the lane offset in the layout.
    auto kv_handle = [&](const D_K* base, const auto& u_g, auto vec) {
        if constexpr(T::LARGE_KV)
            return global_load_base<decltype(vec)::value>(base, u_g);
        else
            return make_gmem(base,
                             static_cast<unsigned>(static_cast<size_t>(kargs.total_tokens) *
                                                   kargs.stride_kv_page * sizeof(D_K)));
    };
    auto g_kv = kv_handle(kv_base, u_gkv, number<T::VEC_KV>{});

    auto mma0 = make_tiled_mma<D_K, D_Q, D_ACC>(seq<T::GEMM0_E_M, T::GEMM0_E_N, T::GEMM0_E_K>{},
                                                seq<1_I, 1_I, 1_I>{},
                                                seq<T::W_M, T::W_N, T::W_K>{},
                                                mfma_adaptor_swap_ab{});
    auto mma1 = make_tiled_mma<D_K, D_K, D_ACC>(seq<T::GEMM1_E_M, T::GEMM1_E_N, T::GEMM1_E_K>{},
                                                seq<T::T_M, T::T_N, T::T_K>{},
                                                seq<T::W_M, T::W_N, T::W_K>{},
                                                mfma_adaptor_swap_ab{});
    typename decltype(mma0)::vtype_a v_q;
    typename decltype(mma0)::vtype_b v_k;
    typename decltype(mma0)::vtype_c v_s[2];
    typename decltype(mma1)::vtype_a v_p;
    typename decltype(mma1)::vtype_b v_v[2];

    // Both GEMM0 operands are cut the same way, the MFMA being square in M and N: the nope
    // k-steps, then the last k-step, whose low half is rope and whose high half is the d that
    // pads D_HEAD_SIZE out to a whole W_K and must read as zero. K comes in as one read of the
    // combined row; Q needs two pieces, its rope skipping LDS.
    constexpr int qk_nope_len = T::D_NOPE_SIZE * T::W_M / T::WARP_SIZE;         // 128
    constexpr int qk_rope_end = T::D_HEAD_SIZE * T::W_M / T::WARP_SIZE;         // 144
    constexpr int qk_pad_end  = T::D_HEAD_SIZE_PADDING * T::W_M / T::WARP_SIZE; // 160

    constexpr index_t s_len = vector_traits<typename decltype(mma0)::vtype_c>::size();

    // Online softmax: skip the O rescale entirely while every lane's new row max is within
    // this much of the running one, so exp2(m_row - row_max) stays well inside fp32 range.
    // Decided by a ballot, so the whole wave takes the same branch.
    constexpr D_ACC RESCALE_THRESHOLD = 8.0f;
    D_ACC rescale_m                   = 1.0f;
    D_ACC row_max;
    bool below_thresh, all_below;

    // u_kv_indices offsets in pages, so the tile stride has to be in pages too. Exact:
    // KV_TILE_SIZE * NUM_WARPS is 128 and PAGE_SIZE divides 4.
    static_assert((T::KV_TILE_SIZE * T::NUM_WARPS) % T::PAGE_SIZE == 0,
                  "a KV tile must hold a whole number of pages");
    constexpr int kv_tile_pages = T::KV_TILE_SIZE * T::NUM_WARPS / T::PAGE_SIZE;

    auto load_kv_page = [&](int tile_idx) {
        return load(g_kv_indices, u_kv_indices, tile_idx * kv_tile_pages)[0];
    };

    // Which token of its page this wave is fetching. Wave-uniform by construction of the
    // token deal (see make_layout_kv_indices), so it stays in an SGPR; folds to a constant
    // 0 at PAGE_SIZE 1, which is what keeps that path byte-identical to before paging.
    const int tok_in_page = warp_id % T::PAGE_SIZE;

    // A page is PAGE_SIZE contiguous token rows, so the physical row of the token this
    // lane wants is page_idx * PAGE_SIZE + tok_in_page. stride_kv_page is the per-token
    // row stride, deliberately not the paged tensor's dim-0 stride.
    auto kv_page_offset = [&](int page_idx) {
        if constexpr(T::LARGE_KV)
            return (static_cast<int64_t>(page_idx) * T::PAGE_SIZE + tok_in_page) *
                   kargs.stride_kv_page;
        else
            return static_cast<int>((static_cast<unsigned>(page_idx) * T::PAGE_SIZE + tok_in_page) *
                                    static_cast<unsigned>(kargs.stride_kv_page));
    };

    auto async_load_kv = [&](auto* s_kv_ptr, int kv_page) {
        if constexpr(T::LARGE_KV)
        {
            global_load<T::VEC_KV>(g_kv + kv_page_offset(kv_page), s_kv_ptr, u_gkv, u_skv);
        }
        else
        {
            async_load<T::VEC_KV>(g_kv, s_kv_ptr, u_gkv + kv_page_offset(kv_page), u_skv);
        }
    };

    const u32_t neg_inf_v = std::bit_cast<u32_t>(-numeric_limits<D_ACC>::infinity());

    // Only the tiles that can actually contain invalid columns pay for a mask: the last
    // partial tile of the request always, and for CAUSAL also the tile holding the
    // diagonal, which is the range's last one since a decode query attends to its whole
    // prefix. `bound` is the tighter of the two limits; the diagonal one is per-warp
    // (readfirstlane'd warp_id) and the rest workgroup-uniform, so it all stays in SGPRs
    // and the branch is scalar.
    // A tile is KV_TILE_SIZE * NUM_WARPS tokens and this wave owns the KV_TILE_SIZE
    // contiguous ones at wave_kv_base inside it (make_layout_rk's deal comes out to
    // warp_id * KV_TILE_SIZE once the DMA's block/row shuffle is composed with the read).
    constexpr int kv_tile_tokens = T::KV_TILE_SIZE * T::NUM_WARPS;
    const int wave_kv_base       = warp_id * T::KV_TILE_SIZE;
    auto mask_oob_scores         = [&](auto& s, int tile_idx) {
        bool masked = (tile_idx + 1) * kv_tile_tokens > valid_kv_len;
        if constexpr(T::CAUSAL)
        {
            masked = masked || (tile_idx == tile_end - 1);
        }
        if(masked)
        {
            int bound = valid_kv_len - 1;
            if constexpr(T::CAUSAL)
            {
                bound = diag_kv_bound < bound ? diag_kv_bound : bound;
            }
            attn_mask_kv_tile<T>(s, bound, tile_idx * kv_tile_tokens + wave_kv_base, neg_inf_v);
        }
    };

    auto softmax_tile = [&](auto& vs) {
        row_max      = temperature_scale * attn_row_max<T>(vs, s_m, warp_id, lane_id);
        below_thresh = ((row_max - m_row) <= RESCALE_THRESHOLD);
        all_below    = (__builtin_amdgcn_ballot_w64(below_thresh) == __builtin_amdgcn_read_exec());
        row_max      = all_below ? m_row : max(m_row, row_max);
        attn_scale_sub_row<T>(vs, temperature_scale, row_max);
        if(!all_below)
        {
            rescale_m = __builtin_amdgcn_exp2f(m_row - row_max);
            l_row *= rescale_m;
            m_row = row_max;
            scale_output_tile<T>(v_o, rescale_m);
        }
        attn_exp2_slice<T, 0, s_len>(vs);
        asm volatile("" : "+v"(vs)::);
        l_row += attn_row_sum<T>(vs, s_l, warp_id, lane_id);
    };

    auto stage_end = [&]() {
        __builtin_amdgcn_sched_barrier(0);
        // __builtin_amdgcn_s_barrier();
        // __builtin_amdgcn_sched_barrier(0);
    };

    auto stage_end_barrier = [&]() {
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    };

    // P back out of LDS as GEMM1's whole 16 x W_K A operand: one load over u_rp walks all
    // kk halves and row halves. Fence after so the load/store optimiser cannot pair the
    // ds_read_b64 issues into ds_read2_b64 and reintroduce 2-way bank conflicts.
    auto load_p = [&](auto& dst) {
        dst = load<T::VEC_READ_P>(s_p, u_rp);
        __builtin_amdgcn_sched_barrier(sched_masks::KEEP_DS_READ_ORDER);
    };

    // async_load q and kv
    set_slice(v_q, vector_t<D_Q, T::VEC_Q>{0}, number<qk_rope_end>{}, number<qk_pad_end>{});
    set_slice(v_k, vector_t<D_K, T::VEC_KV>{0}, number<qk_rope_end>{}, number<qk_pad_end>{});

    async_load<T::VEC_Q>(g_q_nope, s_q.ptr, u_gq_nope, u_sq_nope);
    async_load_kv(s_kv[0].ptr, load_kv_page(tile_begin));
    s_waitcnt_vmcnt(
        number<T::kv_buffer_load_insts>{}); // slot 0 is read below, so drain it outright
    stage_end_barrier();

    // load q from smem
    set_slice(v_q, load<T::VEC_Q>(s_q, u_rq_nope), 0_I, number<qk_nope_len>{});
    set_slice(
        v_q, load<T::VEC_Q>(g_q_rope, u_gq_rope), number<qk_nope_len>{}, number<qk_rope_end>{});

    async_load_kv(s_kv[1].ptr, load_kv_page(min(tile_begin + 1, tile_end - 1)));
    s_waitcnt_lgkmcnt(0_I);
    s_waitcnt_vmcnt(number<T::kv_buffer_load_insts>{});
    stage_end_barrier();

    // if constexpr(STAGGER)
    // {
    //     stage_end();
    // }
    // Page index the first phase prefetches (tile_begin + 2, the one after the two the
    // prologue has already staged). Indices past the end read as 0 through the kv_indices
    // descriptor and land in a slot nobody reads.
    int cur_page = load_kv_page(min(tile_begin + 2, tile_end - 1));
    int nxt_page = cur_page;

    set_slice(v_k, load<T::VEC_KV>(s_kv[0], u_rk), 0_I, number<qk_rope_end>{});
    v_v[0] = tr_load<T::VEC_TR_V>(s_kv[0], u_rv);
    s_waitcnt_lgkmcnt(0_I);
    // Land slot 1's prefetch here so the barrier below publishes it. Left implicit it would
    // be drained by the vmcnt(0) that consuming cur_page forces -- and the WAR fix moves
    // that use two barriers later, past the point the other wave group reads slot 1. Costs
    // nothing where it stands: the reads above are already in flight.
    s_waitcnt_vmcnt(0_I);
    stage_end_barrier();

    // Scores, mask, row max: the head of tile_begin's softmax, which no phase runs since a
    // phase's stage3 only heads its own tile. m_row starts at -inf and O and l at zero, so
    // the rescale the head would do is a no-op and drops out. The tail rides the first
    // phase's QK, like every other tile's.
    v_s[0] = mma0(v_q, v_k, 0, 0);
    clear(v_o);
    mask_oob_scores(v_s[0], tile_begin);
    asm volatile("" : "+v"(v_s[0])::);
    stage_end();

    async_load_kv(s_kv[0].ptr, cur_page);
    s_waitcnt_vmcnt(number<T::kv_buffer_load_insts>{});
    stage_end_barrier();

    auto run_phase = [&](auto& vs_cur, auto& vs_prev, int cur_slot, int prev_slot, int t) {
        // stage0 [mem]: load k.
        set_slice(v_k, load<T::VEC_KV>(s_kv[cur_slot], u_rk), 0_I, number<qk_rope_end>{});
        v_v[cur_slot] = tr_load<T::VEC_TR_V>(s_kv[cur_slot], u_rv);
        nxt_page      = load_kv_page(t + 2);
        s_waitcnt_lgkmcnt(number<T::v_ds_read_insts>{});
        stage_end();

        // stage1 [compute]: gemm0 QK(t) [12 MFMA] || softmax-tail(t-1), the whole exp side of
        // it -- scale-sub, both halves of the exp, and the row sum. QK is 12 MFMA against
        // PV's 4, and it neither reads nor writes tile t-1's scores, so this is the shadow
        // worth filling. m_row already covers tile t-1: its head ran in the last stage3.
        __builtin_amdgcn_s_setprio(1);
        vs_cur = mma0(v_q, v_k, 0, 0);
        // if constexpr(!STAGGER)
        // {
        //     stage_end();
        // }
        softmax_tile(vs_prev);
        auto p_prev = cast<D_K>(vs_prev);
        store<T::VEC_WRITE_P>(s_p, p_prev, u_sp);
        async_load_kv(s_kv[cur_slot].ptr, nxt_page);
        s_waitcnt_lgkmcnt(0_I);
        s_waitcnt_vmcnt(number<T::kv_buffer_load_insts>{});
        stage_end_barrier();

        // stage2 [mem]: V(t-1); mask S(t) before stage4's softmax head folds it in.
        // if constexpr(STAGGER)
        // {
        //     stage_end();
        // }
        load_p(v_p);
        s_waitcnt_lgkmcnt(0_I);
        stage_end();

        // stage3 [compute]: gemm1 PV(t-1) [4 MFMA] || softmax-head(t) -- the row max of the
        // scores stage2 has just masked, and the rebase of O and l that follows from it.
        // This is the only part of the softmax that can sit here at all: it reads tile t,
        // which only becomes readable in stage2, and everything after it needs its result.
        // Leaving PV bare would waste 4 MFMA outright.
        __builtin_amdgcn_s_setprio(1);
        v_o = mma1(v_p, v_v[prev_slot], v_o, 0, 0);
        mask_oob_scores(vs_cur, t);
        __builtin_amdgcn_s_setprio(0);
        stage_end();
        // nxt_page was the tile just issued into cur_slot; unused after that -- next phase
        // recomputes its own from t+2.
    };

    // --- Main loop: tiles tile_begin+1 .. tile_end-1, two phases unrolled per iteration ---
    // Full pairs run unconditionally, so the hot loop carries no inner branch and the two
    // score buffers alternate as compile-time indices; the single leftover phase (present
    // only when the tile count is even) is peeled out below.
    int t = tile_begin + 1;
    for(; t + 1 < tile_end; t += 2)
    {
        __builtin_amdgcn_sched_barrier(0);
        // ping: gemm0+head(t) -> v_s[1], tail+gemm1(t-1) from v_s[0]
        run_phase(v_s[1], v_s[0], 1, 0, t);
        __builtin_amdgcn_sched_barrier(0);
        // pong: gemm0+head(t+1) -> v_s[0], tail+gemm1(t) from v_s[1]
        run_phase(v_s[0], v_s[1], 0, 1, t + 1);
        __builtin_amdgcn_sched_barrier(0);
    }
    __builtin_amdgcn_sched_barrier(0);
    if(t < tile_end) // even tile count: one unpaired phase left
    {
        __builtin_amdgcn_sched_barrier(0);
        run_phase(v_s[1], v_s[0], 1, 0, t);
        __builtin_amdgcn_sched_barrier(0);
    }

    // --- Epilogue: softmax-tail + gemm1 of the last tile ---
    // Phase t writes v_s[1] for odd (t - tile_begin) and v_s[0] for even, so the last
    // tile's scores sit in a buffer chosen by the tile count's parity. Its LDS slot and
    // its mask were already handled by the phase (or by the prologue, for a one-tile
    // request).
    //
    // stage0 [compute]: the last tile's softmax-tail -- its head ran in the last phase's
    // stage3 (or in the prologue, for a one-tile request), but no phase runs its tail, since
    // a phase only tails the tile before its own. There is no QK left to hide it under.
    // Only this part is under the parity branch; keeping the V read and the PV outside it
    // is what keeps v_o off scratch.
    auto epilogue_tail = [&](auto& vs_last, int last_slot) {
        // if constexpr(!STAGGER)
        // {
        //     stage_end();
        // }
        softmax_tile(vs_last);
        store<T::VEC_WRITE_P>(s_p, cast<D_K>(vs_last), u_sp);
        s_waitcnt_lgkmcnt(0_I);
        stage_end_barrier();
        load_p(v_p);
        s_waitcnt_lgkmcnt(0_I);

        v_o = mma1(v_p, v_v[last_slot], v_o, 0, 0);
        __builtin_amdgcn_sched_barrier(0);
    };

    // Last V lands in v_v[cur_slot] of the last phase (or v_v[0] from the prologue when
    // there is only one tile). Phases alternate cur=1,0,1,... so an even tile count
    // (odd phase count) leaves it in v_v[1], an odd tile count in v_v[0]. Same parity
    // picks the score buffer the last phase wrote.
    if(((tile_end - tile_begin) & 1) == 0)
        epilogue_tail(v_s[1], 1);
    else
        epilogue_tail(v_s[0], 0);
}

// --- One work item: load Q, run the tile range, normalize and store O (+ LSE) ---
// work_info_set is 8 ints per item, produced by the metadata kernel. A negative `slot`
// means this item owns the whole request and writes the real output; otherwise it is one
// split-KV partial and writes o_accum / lse_accum for the reduce kernel to merge.
template <class Traits, bool STAGGER>
__device__ __attribute__((always_inline)) void
mla_decode_fwd_one_req(mla_kargs kargs, int w, char* smem_buffer, float temperature_scale)
{
    using namespace opus;
    using T     = opus::remove_cvref_t<Traits>;
    using D_ACC = typename T::D_ACC;
    using D_OUT = typename T::D_OUT;

    int lane_id = thread_id_x() % T::WARP_SIZE;
    asm volatile("" : "+v"(lane_id));
    const int warp_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);

    const int* work_item                 = kargs.work_info_set + w * 8;
    [[maybe_unused]] const int batch_idx = work_item[0];
    const int slot                       = work_item[1];
    const int q_len_ptr_s                = work_item[2];
    const int q_len_ptr_e                = work_item[3];
    const int kv_ind_ptr_s               = work_item[4];
    const int kv_ind_ptr_e               = work_item[5];
    [[maybe_unused]] const int kv_offset = work_item[6];

    const int q_len = q_len_ptr_e - q_len_ptr_s;

    // work_info's kv_start / kv_end are PAGE indices (the metadata emits them that way for
    // every page size), while everything downstream -- tiling, masking, the causal diagonal
    // -- counts tokens. At PAGE_SIZE 1 the two coincide and this all folds away. Above it,
    // the batch's last page is partial, and kv_offset == 0 is the metadata's marker for
    // "this item ends at the batch tail", i.e. the only case where that page is in range.
    // One scalar load, not one per use: both the token count and the causal diagonal convert
    // a page count to tokens the same way, and loading it twice cost 2 spilled SGPRs.
    int last_page_len = T::PAGE_SIZE;
    if constexpr(T::PAGE_SIZE > 1)
    {
        last_page_len = __builtin_amdgcn_readfirstlane(kargs.kv_last_page_lens[batch_idx]);
    }
    // pages -> tokens for a run that ends at the batch tail, where the last page is partial.
    auto pages_to_tokens = [&](int pages) {
        if constexpr(T::PAGE_SIZE == 1)
            return pages;
        else
            return (pages - 1) * T::PAGE_SIZE + last_page_len;
    };

    const int kv_pages = kv_ind_ptr_e - kv_ind_ptr_s;
    // kv_offset != 0 means the metadata handed this item an interior run, so every one of
    // its pages is full; only the tail item can see the partial one.
    const int valid_kv_len =
        (T::PAGE_SIZE == 1 || kv_offset != 0) ? kv_pages * T::PAGE_SIZE : pages_to_tokens(kv_pages);
    const int num_kv_tiles = ceil_div(valid_kv_len, T::KV_TILE_SIZE * T::NUM_WARPS);
    if(num_kv_tiles == 0)
        return;

    // Only the causal specialization needs the diagonal, so the two indptr scalar loads it
    // costs disappear entirely from the decode-only build. kv_indptr is in pages like
    // kv_ind_ptr_s, and the run it measures always ends at the batch tail.
    int causal_diagonal = 0;
    if constexpr(T::CAUSAL)
    {
        causal_diagonal =
            q_len_ptr_s +
            pages_to_tokens(__builtin_amdgcn_readfirstlane(kargs.kv_indptr[batch_idx + 1]) -
                            kv_ind_ptr_s) -
            __builtin_amdgcn_readfirstlane(kargs.q_indptr[batch_idx + 1]);
    }

    // Per-tensor descale folded into the two places it can be a single scalar multiply:
    // QK's descale_q*descale_k rides the softmax temperature, and V's descale_k is applied
    // once on the finished O below (the PV MFMA itself consumes raw fp8).
    const float descale_q = readfirstlane_f32(reinterpret_cast<const float*>(kargs.q_scale_ptr)[0]);
    const float descale_k =
        readfirstlane_f32(reinterpret_cast<const float*>(kargs.kv_scale_ptr)[0]);
    const float qk_scale = readfirstlane_f32(temperature_scale * descale_q * descale_k);

    vector_t<D_ACC, T::Q_TILE_SIZE * T::D_NOPE_SIZE / (T::T_N * T::WARP_SIZE)> v_o;
    D_ACC m_row = opus::numeric_limits<D_ACC>::lowest();
    D_ACC l_row = 0.0f;
    mla_decode_fwd_pipelined<Traits, STAGGER>(kargs,
                                              kv_ind_ptr_s,
                                              valid_kv_len,
                                              0,
                                              num_kv_tiles,
                                              smem_buffer,
                                              q_len_ptr_s,
                                              q_len,
                                              v_o,
                                              m_row,
                                              l_row,
                                              qk_scale,
                                              causal_diagonal);

    // Softmax normalisation and the V descale in one multiply. l_row == 0 means every
    // score was masked, so O must be 0 rather than NaN.
    D_ACC o_scale = (l_row > D_ACC(0.0f)) ? (descale_k / l_row) : D_ACC(0.0f);
    scale_output_tile<T>(v_o, o_scale);
    pin_output_tile(v_o);

    if(slot < 0)
    {
        const int o_gmem_offset = q_len_ptr_s * kargs.stride_o_b;
        auto g_o                = make_gmem(reinterpret_cast<D_OUT*>(kargs.out_ptr) + o_gmem_offset,
                             q_len * kargs.stride_o_b * sizeof(D_OUT));
        auto u_o                = make_layout_o<T>(warp_id, lane_id, kargs.stride_o_h);
        auto v_o_out            = cast<D_OUT>(v_o);
        store<T::VEC_O>(g_o, v_o_out, u_o);
        // One LSE per query row, and after the cross-wave merge every wave holds the same
        // m_row / l_row for a row, so one wave writes the whole Q_TILE_SIZE.
        // lse_ptr is null when the caller did not ask for LSE; lse_accum in the split-KV
        // branch is always allocated, so only this side needs the guard.
        if(kargs.lse_ptr != nullptr && warp_id == 0 && lane_id < T::W_M)
        {
            const int lse_offset = q_len_ptr_s * kargs.H;
            auto g_lse           = make_gmem(reinterpret_cast<D_ACC*>(kargs.lse_ptr) + lse_offset,
                                   q_len * kargs.H * sizeof(D_ACC));
            constexpr float INV_LOG2_E = 0.69314718055994531f; // 1 / LOG2_E == ln(2)
            const D_ACC lse = (l_row > D_ACC(0.0f)) ? ((m_row + log2f(l_row)) * INV_LOG2_E)
                                                    : opus::numeric_limits<D_ACC>::lowest();
            g_lse.store(lse, lane_id);
        }
    }
    if(slot >= 0)
    {
        const int oa_offset = slot * kargs.stride_o_b;
        auto g_oa           = make_gmem(reinterpret_cast<D_ACC*>(kargs.o_accum) + oa_offset,
                              q_len * kargs.stride_o_b * sizeof(D_ACC));
        auto u_oa           = make_layout_o<T>(warp_id, lane_id, T::D_NOPE_SIZE);
        store<T::VEC_O>(g_oa, v_o, u_oa);

        if(warp_id == 0 && lane_id < T::W_M)
        {
            const int lse_offset = slot * kargs.H;
            auto g_lse           = make_gmem(reinterpret_cast<D_ACC*>(kargs.lse_accum) + lse_offset,
                                   q_len * kargs.H * sizeof(D_ACC));
            constexpr float INV_LOG2_E = 0.69314718055994531f; // 1 / LOG2_E == ln(2)
            const D_ACC lse = (l_row > D_ACC(0.0f)) ? ((m_row + log2f(l_row)) * INV_LOG2_E)
                                                    : opus::numeric_limits<D_ACC>::lowest();
            g_lse.store(lse, lane_id);
        }
    }
}

} // namespace mla_decode_fwd_16mx1_16nx8_fp8fp8

// Persistent entry point: the grid is sized to the machine, not to the problem, and each
// block drains the work items the metadata kernel assigned it through work_indptr. The
// occupancy-2 launch bound is what caps the whole kernel at 256 VGPR.
template <class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE,
                             2) void mla_decode_fwd_16mx1_16nx8_fp8fp8_opus_kernel(mla_kargs kargs)
{
    using namespace opus;
    using namespace mla_decode_fwd_16mx1_16nx8_fp8fp8;
    using T = opus::remove_cvref_t<Traits>;

    const int work_id = block_id_x();

    // 2 KV slots, and 2 is the ceiling, not a choice: a slot is a whole 128-token tile at
    // d = 576, i.e. 74.25 KB, and two of them plus Q already come to 157.5 KB of gfx950's
    // 160. Nor can the tile shrink to buy a third -- GEMM1 contracts a wave's tokens over
    // W_K = 128, which pins KV_TILE_SIZE * NUM_WARPS at exactly 128. What stands in for the
    // missing slots is v_v: V is pulled into registers in its own phase, so a slot's life
    // is one phase and tile t+2 can be issued into it right after. The alignment is
    // load-bearing:
    // make_layout_rv folds the bank swizzle's XOR into a +/- SWZ_D_BYTES stride, which is
    // only equivalent to the XOR while bit 4 of the block's base address is zero.
    __shared__ __align__(128) char smem_buffer[T::smem_bytes()];

    const int work_idx_start = kargs.work_indptr[work_id];
    const int work_idx_end   = kargs.work_indptr[work_id + 1];
    if(work_idx_start >= work_idx_end)
        return;

    constexpr float LOG2_E        = 1.44269504089f;
    const float temperature_scale = readfirstlane_f32(kargs.softmax_scale * LOG2_E);
    const int warp_id = __builtin_amdgcn_readfirstlane(opus::thread_id_x() / Traits::WARP_SIZE);
    for(int w = work_idx_start; w < work_idx_end; ++w)
    {
        // __builtin_amdgcn_sched_barrier(0);
        // __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        mla_decode_fwd_one_req<Traits, false>(kargs, w, smem_buffer, temperature_scale);

        // if(warp_id / 4)
        //     mla_decode_fwd_one_req<Traits, true>(kargs, w, smem_buffer, temperature_scale);
        // __builtin_amdgcn_sched_barrier(0);
        // if(!(warp_id / 4))
        //     mla_decode_fwd_one_req<Traits, false>(kargs, w, smem_buffer, temperature_scale);
    }
}

#endif // !__HIP_DEVICE_COMPILE__ || !__gfx950__
