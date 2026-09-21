#pragma once

// MLA decode forward on gfx950: fp8 Q x fp8 KV, 16mx1 / 32nx4, persistent scheduling.
//   GEMM0 (Q*K^T): d = 576 = 512 nope + 64 rope      GEMM1 (P*V): d_v = 512
//
// A cross of the two kernels beside it; their file comments are the reference for anything
// not repeated here.
//   * From 16mx1_16nx8 comes the M side: all NUM_WARPS waves share ONE Q_TILE_SIZE-row tile,
//     so Q is DMA'd to LDS once per work item and read back whole by every wave, P round-trips
//     LDS (a wave scores only its own tokens but GEMM1 contracts all of them), the row max is
//     merged across waves every tile and the row sum once at the end, and warp_id picks a
//     slice of the output d rather than a block of rows.
//   * From 16mx8_32nx1 comes the KV memory pattern: a wave owns KV_TILE_SIZE = 32 tokens, and
//     those sit in LDS in 16mx8's image -- nope as smem_n_rpt line blocks of smem_n_per_wave
//     tokens x 128 d, rope in its own region -- so make_layout_rk_nope and make_layout_rv are
//     that kernel's verbatim with a token-group base added, and GEMM0 is its 12 MFMA
//     (2 e_n x 4 nope e_k of 16x16x128, plus 2 rope e_k of 16x16x32).
//
// A KV tile is KV_TILE_SIZE * NUM_WARPS = 128 tokens in NUM_WARPS groups of 32; wave g fills
// group g by itself (16 nope + 2 rope buffer_load_lds) and scores it, and GEMM1 then walks all
// four groups as its four k-steps, reading each group's P block and V out of that group's K.
//
// The default build is the two-slot ring: tile t+1's DMA is issued from inside tile t's GEMM0,
// chunk by chunk in its MFMA shadows, and lands in the other slot while tile t is scored. The
// loop at the bottom carries the barrier and vmcnt rules, and the three restructurings that
// were tried against it and lost. AITER_MLA_OPUS_32NX4_SLOTS_1=1 builds the one-slot floor
// instead -- one tile at a time, every stage drained and barriered, so the whole gmem latency
// of every tile sits in the open. That is 12.8 to 14.9% slower on the memory-bound shapes and
// exists to bound what the ring is worth.
//
// WHAT IT COSTS (gfx950, from the compiled metadata; the spread is the causal x large_kv
// fan-out):
//
//                  total VGPR (arch + acc)    SGPR    spill / scratch     LDS     blocks/CU
//   two slots          208 (176 + 32)        99-106       0 / 0        155136 B       1
//   one slot           152 (152 +  0)        97-106       0 / 0         79360 B       2
//
// Neither is register bound. Two slots put a block at 155 KB of gfx950's 160, so it is one
// block per CU: four waves on four SIMDs, one apiece, and a lone wave owns all 512 registers.
// The one-slot build fits two blocks and so two waves per SIMD, where the budget is 256 --
// still twice what it uses. Q aliasing the last KV slot (smem_q_offset in the traits) is what
// keeps that build under 80 KB and therefore at two blocks.
//
// WHERE THE CEILING IS: at b=256 c=8192 page_size=1 the kernel runs at 226.4 us and moves
// 1.352 GB of DRAM read traffic doing it -- 6.0 TB/s against gfx950's ~6.3 TB/s coalesced
// roof, i.e. ~95% of the machine. (rocprofv3: TCC_EA0_RDREQ doubled, since the counters come
// back from half the instances, times 64 B, TCC_EA0_RDREQ_32B being zero so every EA read is
// a full sector. TCC_HIT / TCC_REQ is 6.3%.) The hand-written asm decode kernel
// (AITER_MLA_USE_OPUS=0) lands at 227.95 us on the identical shape, which is the cheapest
// confirmation available that this is a wall and not an implementation.
//
// That traffic is 10.07 sectors per 576 B token row against the 9 the data is, and the extra
// one belongs to the page table rather than to the kernel: 576 is not a multiple of 128, so a
// row's tail shares a line with the next token row, which at page_size 1 sits somewhere
// unrelated and is read by an unrelated request at an unrelated time. The 16nx8 header works
// this through and shows it disappearing at page_size 2 -- but this geometry cannot use
// page_size > 1 at all, its rope DMA dealing a whole 16-token line so the within-page token
// offset would have to be per-lane. Here the 11% is structural, and the kernel is already
// within 5% of what is left after it.

#include "mla_decode_traits.h"

#if !defined(__HIP_DEVICE_COMPILE__) || !defined(__gfx950__)

template <class Traits>
__global__ void opus_mla_decode_fp8_16mx1_32nx4_kernel(opus_mla_decode_fp8_kargs)
{
}

#else

#include "mla_global_load.hpp"
#include <bit>
#include <cstdint>
#include <opus/opus.hpp>

using opus::operator""_I;

namespace opus_mla_decode_fp8_16mx1_32nx4 {

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

// --- Q gmem->LDS->register (Q rounds through LDS, as in 16mx1_16nx8) ---
//
// The LDS image is 16mx1's: smem_n_rpt_q * smem_d_rpt_q blocks of smem_q_block, each one
// wave's buffer_load_lds verbatim, block index d_chunk * smem_n_rpt_q + warp_id. With four
// waves instead of eight, the d half that 16mx1 gets from warp_id / smem_n_rpt_q becomes a
// y-iteration here, so a wave issues two DMAs and owns two blocks; the bytes land in exactly
// the same places, which is why the read layout below is unchanged from 16mx1's.
//
//   row = smem_n_rpt_q * (l / threads_d % smem_n_per_wave_q) + warp_id
//   d   = d_chunk * smem_d_per_wave_q + l / (threads_d * smem_n_per_wave_q) * W_K_NOPE
//       + l % threads_d * VEC_Q_NOPE
template <class T>
__device__ inline auto make_layout_gq_nope(int warp_id, int lane_id)
{
    constexpr int threads_d = T::W_K_NOPE / T::VEC_Q_NOPE; // lanes covering one W_K line

    constexpr auto gq_shape =
        opus::make_tuple(opus::number<T::smem_n_per_wave_q>{},
                         opus::number<T::smem_n_rpt_q>{},
                         opus::number<T::smem_d_rpt_q>{},                    // d chunk, y
                         opus::number<T::smem_d_per_wave_q / T::W_K_NOPE>{}, // W_K line in it
                         opus::number<threads_d>{},
                         opus::number<T::VEC_Q_NOPE>{});

    constexpr auto gq_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        gq_shape,
        opus::unfold_x_stride(gq_dim, gq_shape, opus::tuple{opus::number<T::D_HEAD_SIZE>{}, 1_I}),
        opus::unfold_p_coord(gq_dim,
                             opus::tuple{(lane_id / threads_d) % T::smem_n_per_wave_q,
                                         warp_id,
                                         lane_id / (threads_d * T::smem_n_per_wave_q),
                                         lane_id % threads_d}));
}

// Q LDS destination, wave-uniform by necessity: buffer_load_lds takes no per-lane LDS
// address, the hardware writes lane l to dst + l * VEC. The only free parameter is which
// block a wave's two issues land in.
template <class T>
__device__ inline auto make_layout_sq_nope(int warp_id)
{
    constexpr auto sq_shape = opus::make_tuple(opus::number<T::smem_d_rpt_q>{},
                                               opus::number<T::smem_n_rpt_q>{},
                                               opus::number<T::VEC_Q_NOPE>{});

    constexpr auto sq_dim = opus::make_tuple(opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
                                             opus::make_tuple(opus::y_dim{}));

    return opus::make_layout(
        sq_shape,
        opus::unfold_x_stride(sq_dim, sq_shape, opus::tuple{opus::number<T::smem_q_block>{}, 1_I}),
        opus::unfold_p_coord(sq_dim, opus::tuple{warp_id}));
}

// Q LDS->register, the B operand of GEMM0's four 16x16x128 nope MFMA: lane l holds row
// l % W_M and, in e_k step E, d = E * W_K_NOPE + kk * (WARP_SIZE / W_M) * VEC + l / W_M * VEC
// for kk in [0, 2). Undoing the DMA's placement is what splits E into two y-dims: its low
// half moves d by W_K_NOPE inside a block, its high half crosses to the block that owns the
// next smem_d_per_wave_q of d. The y odometer runs E high, E low, kk, vector, so the register
// vector is the four 32-fp8 slices the MFMA wants, in order.
//
// Every wave reads the whole tile and the read is bank-conflict-free: inside one ds_read_b128
// phase the 16 lanes' row and d-group terms (264 and 32 dwords) tile all 64 banks once.
template <class T>
__device__ inline auto make_layout_rq_nope(int lane_id)
{
    constexpr auto rq_shape =
        opus::make_tuple(opus::number<T::smem_d_rpt_q>{},                    // e_k high
                         opus::number<T::smem_n_rpt_q>{},                    // row % smem_n_rpt_q
                         opus::number<T::smem_d_per_wave_q / T::W_K_NOPE>{}, // e_k low
                         opus::number<T::smem_n_per_wave_q>{},               // row / smem_n_rpt_q
                         opus::number<T::W_M * T::W_K_NOPE / T::WARP_SIZE / T::VEC_Q_NOPE>{},
                         opus::number<T::WARP_SIZE / T::W_M>{}, // lane's d-group in W_K
                         opus::number<T::VEC_Q_NOPE>{});

    constexpr auto rq_dim =
        opus::make_tuple(opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
                         opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
                         opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    const int lane_m = lane_id % T::W_M;

    return opus::make_layout(
        rq_shape,
        opus::unfold_x_stride(
            rq_dim,
            rq_shape,
            opus::tuple{opus::number<T::smem_q_block>{}, opus::number<T::W_K_NOPE>{}, 1_I}),
        opus::unfold_p_coord(
            rq_dim,
            opus::tuple{lane_m % T::smem_n_rpt_q, lane_m / T::smem_n_rpt_q, lane_id / T::W_M}));
}

// Q rope, d in [D_NOPE_SIZE, D_HEAD_SIZE); the caller offsets the gmem base by +D_NOPE_SIZE.
// This skips LDS: the B operand of the two 16x16x32 rope MFMA is one dwordx2 per lane per
// e_k, and every wave wants the same Q_TILE_SIZE x 64 fp8, i.e. the same few cache lines.
// Same layout as 16mx8's make_layout_q_rope minus its warp_id row block -- here the waves
// share the rows.
template <class T>
__device__ inline auto make_layout_gq_rope(int lane_id)
{
    constexpr auto gq_shape = opus::make_tuple(opus::number<T::W_M>{},
                                               opus::number<T::GEMM0_ROPE_E_K>{},
                                               opus::number<T::WARP_SIZE / T::W_M>{},
                                               opus::number<T::VEC_Q_ROPE>{});

    constexpr auto gq_dim =
        opus::make_tuple(opus::make_tuple(opus::p_dim{}),
                         opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        gq_shape,
        opus::unfold_x_stride(gq_dim, gq_shape, opus::tuple{opus::number<T::D_HEAD_SIZE>{}, 1_I}),
        opus::unfold_p_coord(gq_dim, opus::tuple{lane_id % T::W_M, lane_id / T::W_M}));
}

// --- KV paged-index fetch ---
//
// Wave g fills and scores token group g, i.e. tile tokens [g * KV_TILE_SIZE, +KV_TILE_SIZE).
// Inside the group the nope DMA deals 16mx8's token order: line n holds the tokens congruent
// to n mod smem_n_rpt, token smem_n_rpt * row + n at block row row = lane / threads_d. The
// lines' page indices are that many consecutive kv_indices entries -- one vector load.
template <class T>
__device__ inline auto make_layout_kv_indices_nope(int warp_id, int lane_id)
{
    constexpr int threads_d = T::D_128B_NOPE_SIZE / T::VEC_KV_NOPE; // 8

    constexpr auto shape = opus::make_tuple(opus::number<T::GROUPS_PER_TILE>{},
                                            opus::number<T::smem_n_per_wave>{},
                                            opus::number<T::nope_lines_per_wave>{});

    constexpr auto dim =
        opus::make_tuple(opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(shape,
                             opus::unfold_x_stride(dim, shape, opus::tuple{1_I}),
                             opus::unfold_p_coord(dim, opus::tuple{warp_id, lane_id / threads_d}));
}

// The rope DMA deals a whole 16-token line per instruction, token lane / threads_d of line m,
// so a thread needs one page index per line and they are rope_toks_per_line apart.
template <class T>
__device__ inline auto make_layout_kv_indices_rope(int warp_id, int lane_id)
{
    constexpr int threads_d = T::D_ROPE_SIZE / T::VEC_KV_ROPE_LD; // 4 lanes per token

    constexpr auto shape = opus::make_tuple(opus::number<T::GROUPS_PER_TILE>{},
                                            opus::number<T::rope_lines_per_wave>{},
                                            opus::number<T::rope_toks_per_line>{},
                                            1_I);

    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(shape,
                             opus::unfold_x_stride(dim, shape, opus::tuple{1_I}),
                             opus::unfold_p_coord(dim, opus::tuple{warp_id, lane_id / threads_d}));
}

// --- K gmem->LDS (async buffer_load_lds) ---
//
// K nope, d in [0, D_NOPE_SIZE) of the combined buffer. The token is folded into the
// per-thread page offset, so this layout is only the d walk: smem_d_rpt_nope chunks of
// D_128B_NOPE_SIZE, one buffer_load_lds each.
//
// The V-read bank swizzle has to live on the SOURCE side, because buffer_load_lds takes no
// per-lane LDS address: lane l owns block row l / threads_d and the only free choice left is
// which d-group it fetches. Rows in the upper half of a line (T::SWZ_TOK_BIT) therefore hold
// their d-groups XORed by one, which is what the V transpose read asks those slots for.
template <class T>
__device__ inline auto make_layout_gkv_nope(int lane_id)
{
    constexpr int threads_d = T::D_128B_NOPE_SIZE / T::VEC_KV_NOPE;

    constexpr auto shape = opus::make_tuple(opus::number<T::smem_d_rpt_nope>{},
                                            opus::number<threads_d>{},
                                            opus::number<T::VEC_KV_NOPE>{});

    constexpr auto dim = opus::make_tuple(opus::make_tuple(opus::y_dim{}),
                                          opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    const int tok_in_line = lane_id / threads_d;
    const int d_grp       = (lane_id % threads_d) ^ ((tok_in_line & T::SWZ_TOK_BIT) ? 1 : 0);

    return opus::make_layout(
        shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{opus::number<T::D_128B_NOPE_SIZE>{}, 1_I}),
        opus::unfold_p_coord(dim, opus::tuple{d_grp}));
}

// K nope LDS destination. Wave-uniform: the caller adds the group base and the line, this
// walks the smem_d_rpt_nope chunks, which are smem_n_rpt line blocks apart.
template <class T>
__device__ inline auto make_layout_skv_nope()
{
    constexpr auto shape =
        opus::make_tuple(opus::number<T::smem_d_rpt_nope>{}, opus::number<T::VEC_KV_NOPE>{});

    // Every dim is a y-dim, but the coord still has to be spelled out: a layout without one
    // has no issue space at all and would collapse to a single load.
    constexpr auto dim =
        opus::make_tuple(opus::make_tuple(opus::y_dim{}), opus::make_tuple(opus::y_dim{}));

    return opus::make_layout(
        shape,
        opus::make_tuple(opus::number<T::smem_n_rpt * T::smem_nope_line>{}, 1_I),
        opus::unfold_p_coord(dim, opus::tuple<>{}));
}

// K rope, d in [D_NOPE_SIZE, D_HEAD_SIZE); the caller offsets the gmem base by +D_NOPE_SIZE.
// One instruction is one 16-token line: lane l fetches token l / threads_d and 16 B of d, and
// the LDS slot it lands in is l * VEC_KV_ROPE_LD.
//
// The d-group it fetches is XORed with the token's own quarter of the line, which is what
// makes the b64 rope read conflict-free. Without it the read's 8B slot is 8 * token + half,
// and 8 * token mod 32 collapses 16 tokens onto 4 banks; with it the slot becomes
// 8 * t + 2 * (c ^ (t / 4)) + half, and the four tokens sharing a bank group get four
// different d-group rotations, so the 32 lanes of a phase tile all 32 banks.
template <class T>
__device__ inline auto make_layout_gkv_rope(int lane_id)
{
    constexpr int threads_d = T::D_ROPE_SIZE / T::VEC_KV_ROPE_LD; // 4

    constexpr auto shape =
        opus::make_tuple(opus::number<threads_d>{}, opus::number<T::VEC_KV_ROPE_LD>{});

    constexpr auto dim = opus::make_tuple(opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    const int tok_in_line = lane_id / threads_d;
    const int d_grp       = (lane_id % threads_d) ^ (tok_in_line / threads_d);

    return opus::make_layout(shape,
                             opus::unfold_x_stride(dim, shape, opus::tuple{1_I}),
                             opus::unfold_p_coord(dim, opus::tuple{d_grp}));
}

template <class T>
__device__ inline auto make_layout_skv_rope()
{
    constexpr auto dim = opus::make_tuple(opus::make_tuple(opus::y_dim{}));
    return opus::make_layout(opus::make_tuple(opus::number<T::VEC_KV_ROPE_LD>{}),
                             opus::make_tuple(1_I),
                             opus::unfold_p_coord(dim, opus::tuple<>{}));
}

// --- K LDS->register ---
//
// K nope, the A operand of the 16x16x128 f8f6f4 MFMA, from 16mx8 unchanged. EN is the token
// tile (0 -> tokens 0..15 of the group, 1 -> 16..31); it is a template parameter rather than
// a y-dim because only the second one sits on block rows that carry the swizzled d-groups, so
// this tile's lane->d-group map is XORed to match. The caller adds the group base, the e_k
// chunk, and for EN = 1 the tile's own smem_n_rpt * D_128B_NOPE_SIZE.
template <class T, int EN>
__device__ inline auto make_layout_rk_nope(int lane_id)
{
    constexpr auto shape =
        opus::make_tuple(opus::number<T::smem_n_rpt>{},
                         opus::number<T::W_N / T::smem_n_rpt>{},
                         opus::number<T::W_N * T::W_K_NOPE / T::WARP_SIZE / T::VEC_KV_NOPE>{},
                         opus::number<T::WARP_SIZE / T::W_N>{},
                         opus::number<T::VEC_KV_NOPE>{});

    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    const int lane_n = lane_id % T::W_N;

    return opus::make_layout(
        shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{opus::number<T::smem_nope_line>{}, 1_I}),
        opus::unfold_p_coord(
            dim,
            opus::tuple{lane_n % T::smem_n_rpt, lane_n / T::smem_n_rpt, (lane_id / T::W_N) ^ EN}));
}

// K rope, the A operand of the plain fp8 16x16x32 MFMA, one e_k step per layout. A rope line
// is one e_n token tile, so lane l wants token l % W_N of line EN and d = EK * W_K_ROPE +
// (l / W_N) * VEC_KV_ROPE. Undoing the DMA's d-group XOR makes the whole byte offset a single
// per-lane value, hence a bare linear layout over the read vector.
template <class T, int EK>
__device__ inline auto make_layout_rk_rope(int lane_id)
{
    constexpr int halves_per_grp = T::VEC_KV_NOPE / T::VEC_KV_ROPE;    // 2 reads per 16B d-group
    constexpr int d_grps_per_tok = T::D_ROPE_SIZE / T::VEC_KV_ROPE_LD; // 4, and the DMA's
                                                                       // lanes per token
    constexpr int ek_d_grp = T::W_K_ROPE / T::VEC_KV_NOPE;             // 2 d-groups per k step

    const int tok = lane_id % T::W_N; // token in the line
    const int grp = lane_id / T::W_N; // which VEC_KV_ROPE of the k step this lane wants
    // Undo the DMA's d-group rotation: it stored group c of token t in slot c ^ (t / 4).
    const int d_c = (EK * ek_d_grp + grp / halves_per_grp) ^ (tok / d_grps_per_tok);
    const int off =
        tok * T::D_ROPE_SIZE + d_c * T::VEC_KV_NOPE + (grp % halves_per_grp) * T::VEC_KV_ROPE;

    constexpr auto shape = opus::make_tuple(1_I, opus::number<T::VEC_KV_ROPE>{});
    constexpr auto dim =
        opus::make_tuple(opus::make_tuple(opus::p_dim{}), opus::make_tuple(opus::y_dim{}));

    return opus::make_layout(
        shape, opus::make_tuple(1_I, 1_I), opus::unfold_p_coord(dim, opus::tuple{off}));
}

// --- V LDS->register transpose read (ds_read_b64_tr_b8), from 16mx8 unchanged ---
//
// See that kernel for the derivation; the only thing this file has to keep true is the
// alignment it assumes. EN is the W_N d-tile inside the SLICE_D slice, and the swizzle XOR
// collapses to a compile-time +/- SWZ_D_BYTES only while bit 4 of the instruction's address
// comes from EN alone -- which holds because every base the caller adds (the token group, the
// wave's d_rpt block, the slice) is a multiple of 32.
template <class T, int EN>
__device__ inline auto make_layout_rv(int lane_id)
{
    constexpr int lane_per_grp = 16;                     // ds_read_b64_tr_b8 group
    constexpr int lane_lo      = T::W_N / T::VEC_TR_V;   // W_N halves per 8x8 (2)
    constexpr int lane_hi      = lane_per_grp / lane_lo; // 8
    constexpr int hi_lo        = T::smem_n_rpt;          // lane_hi % n_rpt -> line (4)
    constexpr int hi_hi        = lane_hi / hi_lo;        // lane_hi / n_rpt -> token (2)

    constexpr int swz = ((EN * T::W_N) & T::SWZ_D_BYTES) ? -T::SWZ_D_BYTES : T::SWZ_D_BYTES;

    constexpr auto shape = opus::make_tuple(opus::number<T::smem_n_rpt>{},
                                            opus::number<hi_lo>{},
                                            opus::number<hi_hi>{},
                                            opus::number<lane_lo>{},
                                            opus::number<T::VEC_TR_V>{});

    constexpr auto dim = opus::make_tuple(opus::make_tuple(opus::p_dim{}),
                                          opus::make_tuple(opus::p_dim{}),
                                          opus::make_tuple(opus::p_dim{}),
                                          opus::make_tuple(opus::p_dim{}),
                                          opus::make_tuple(opus::y_dim{}));

    const int grp_id      = lane_id / lane_per_grp;
    const int lane_in_grp = lane_id % lane_per_grp;
    const int lh          = lane_in_grp / lane_lo;

    return opus::make_layout(
        shape,
        opus::unfold_x_stride(dim,
                              shape,
                              opus::tuple{opus::number<T::D_128B_NOPE_SIZE>{},
                                          opus::number<T::smem_nope_line>{},
                                          opus::number<T::smem_n_rpt * T::D_128B_NOPE_SIZE + swz>{},
                                          opus::number<T::VEC_TR_V>{},
                                          1_I}),
        opus::unfold_p_coord(dim,
                             opus::tuple{grp_id, lh % hi_lo, lh / hi_lo, lane_in_grp % lane_lo}));
}

// --- P register->LDS->register ---
//
// GEMM1 wants P as its A operand over the whole KV_TILE_TOKENS, but GEMM0 leaves a wave
// holding only its own group's KV_TILE_SIZE columns, so P round-trips LDS: one block per
// wave, written by its owner and read by all NUM_WARPS as the four k-steps.
//
// Both sides are the same access, which is what makes them both conflict-free. Within a
// block row (one query row) a score sits at position 8 * (lane / W_M) + 4 * e_n + pack
// element. That is at once the C register order of one lane -- a lane's eight scores are the
// e_n 0 pack then the e_n 1 pack, so the write is ONE ds_write_b64 -- and the k order the
// MFMA A operand wants, since element j of lane l stands for k = 8 * (l / W_M) + j. So both
// the write and the read are "16 rows x four 8B halves, one half per lane group".
//
// Bank-wise: the 8B slot is P_ROW_PITCH / 8 * row + half. A pitch of 6 slots makes the row
// term hit all 16 even slots exactly once, and XORing the half index with 1 on the upper 8
// rows keeps the pair {6r, 6r + 1} split across parities, so the 32 lanes of a phase tile all
// 32 banks. (The XOR costs nothing: it permutes which lane takes which half.)
template <class T>
__device__ inline int p_half_swz(int row)
{
    return row >= T::W_M / 2 ? 1 : 0;
}

template <class T>
__device__ inline int p_row_offset(int lane_id)
{
    const int row = lane_id % T::W_M;
    return row * T::P_ROW_PITCH + ((lane_id / T::W_M) ^ p_half_swz<T>(row)) * T::VEC_READ_P;
}

// P LDS destination: the lane's whole 8B half of its own group's block, one ds_write_b64.
template <class T>
__device__ inline auto make_layout_sp(int warp_id, int lane_id)
{
    static_assert((T::WARP_SIZE / T::W_M) * T::VEC_READ_P == T::KV_TILE_SIZE,
                  "the lane groups' 8B halves must tile a block row exactly once");
    static_assert(T::VEC_WRITE_P == T::VEC_READ_P, "a lane writes the half it will read back");

    constexpr auto dim = opus::make_tuple(opus::make_tuple(opus::y_dim{}));
    auto u             = opus::make_layout(opus::make_tuple(opus::number<T::VEC_WRITE_P>{}),
                               opus::make_tuple(1_I),
                               opus::unfold_p_coord(dim, opus::tuple<>{}));
    u += warp_id * T::smem_p_pitch + p_row_offset<T>(lane_id);
    return u;
}

// P LDS->register: one ds_read_b64 per k-step, k-step kk being token group kk.
template <class T>
__device__ inline auto make_layout_rp(int lane_id)
{
    constexpr auto dim =
        opus::make_tuple(opus::make_tuple(opus::y_dim{}), opus::make_tuple(opus::y_dim{}));
    auto u = opus::make_layout(
        opus::make_tuple(opus::number<T::GEMM1_K_STEPS>{}, opus::number<T::VEC_READ_P>{}),
        opus::make_tuple(opus::number<T::smem_p_pitch>{}, 1_I),
        opus::unfold_p_coord(dim, opus::tuple<>{}));
    u += p_row_offset<T>(lane_id);
    return u;
}

// O register->gmem store. stride_o_h is a parameter because the same layout serves both
// destinations: the real output uses kargs.stride_o_h, while the split-KV partial writes a
// densely packed D_NOPE_SIZE-strided o_accum.
//
// warp_id sits on the d side: every wave carries the same Q_TILE_SIZE rows and owns
// WAVE_D = T_N-th of the output d, as NUM_D_SLICES slices of GEMM1_E_N tiles of W_M x W_N.
template <class T>
__device__ inline auto make_layout_o(int warp_id, int lane_id, int stride_o_h)
{
    static_assert(T::T_N * T::NUM_D_SLICES * T::GEMM1_E_N * T::W_N == T::D_NOPE_SIZE,
                  "the waves' d slices must tile the output d exactly once");

    constexpr auto shape =
        opus::make_tuple(opus::number<T::GEMM1_E_M>{}, // e_m
                         opus::number<T::W_M>{},       // query row
                         opus::number<T::T_N>{},       // wave's d slice
                         opus::number<T::NUM_D_SLICES>{},
                         opus::number<T::GEMM1_E_N>{},
                         opus::number<T::W_M * T::W_N / T::WARP_SIZE / T::VEC_O>{},
                         opus::number<T::WARP_SIZE / T::W_M>{}, // lane's d group
                         opus::number<T::VEC_O>{});

    constexpr auto dim = opus::make_tuple(opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
                                          opus::make_tuple(opus::p_dim{},
                                                           opus::y_dim{},
                                                           opus::y_dim{},
                                                           opus::y_dim{},
                                                           opus::p_dim{},
                                                           opus::y_dim{}));

    return opus::make_layout(
        shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{stride_o_h, 1_I}),
        opus::unfold_p_coord(dim, opus::tuple{lane_id % T::W_M, warp_id, lane_id / T::W_M}));
}

// --- softmax / scaling helpers (16mx1's: the reductions are partial per wave) ---
//
// Two reductions per row inside a wave (W_M = 16 against 64-wide waves puts a row across four
// lane groups), then one across waves: a wave scores only KV_TILE_SIZE of the tile's
// KV_TILE_TOKENS, so its row max is partial and GEMM1 needs the merged one. The waves swap
// through s_m: lane writes slot row * T_N + warp_id, reads the whole T_N-wide row back and
// folds it, so all waves leave holding the same value.
template <class T, class V, class S>
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

// Fused `v_s * scale - row_max`, one v_fma per element. The caller reduces the *raw* scores
// and scales that single scalar instead (scale > 0, so max commutes with it).
template <class T, class V>
__device__ inline void
attn_scale_sub_row(V& v_s, typename T::D_ACC scale, typename T::D_ACC row_max)
{
    constexpr opus::index_t s_len = opus::vector_traits<V>::size();
    opus::static_for<s_len>(
        [&](auto i) { v_s[i.value] = __builtin_fmaf(v_s[i.value], scale, -row_max); });
}

template <class T, opus::index_t Offset, opus::index_t Count, class V>
__device__ inline void attn_exp2_slice(V& v_s)
{
    opus::static_for<Count>([&](auto i) {
        constexpr opus::index_t idx = Offset + i.value;
        v_s[idx]                    = __builtin_amdgcn_exp2f(v_s[idx]);
    });
}

// Balanced tree, not the `row_sum += v_s[i]` chain the loop shape suggests: float addition
// does not reassociate, so the chain compiles to s_len dependent v_add_f32 back to back. It
// sums in a different order and therefore rounds differently, which is harmless here: every
// term is a positive exp2 result. Wave-local only -- see merge_row_sum.
template <class T, class V>
__device__ inline typename T::D_ACC attn_row_sum_local(const V& v_s)
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
    return row_sum;
}

// Cross-wave sum merge, run ONCE for the whole tile range rather than once per tile. The max
// has to be merged every tile -- the next exp2 is taken against it -- but the sum does not:
// every wave's running l_row is already expressed against the same m_row, so partial sums
// stay additive and nothing reads l_row before the normalisation at the end.
template <class T, class S>
__device__ inline typename T::D_ACC
merge_row_sum(typename T::D_ACC l_row, S& s_l, int warp_id, int lane_id)
{
    using D_ACC   = typename T::D_ACC;
    const int row = lane_id % T::W_M;
    opus::store(s_l, l_row, row * T::T_N + warp_id);
    opus::s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    auto sum_warps = opus::load<T::T_N>(s_l, row * T::T_N);
    D_ACC total    = D_ACC(0.0f);
    opus::static_for<T::T_N>([&](auto i) { total += sum_warps[i.value]; });
    return total;
}

template <class T, class V>
__device__ inline void scale_output_tile(V& v_o, typename T::D_ACC scale)
{
    constexpr opus::index_t o_len = opus::vector_traits<V>::size();
    opus::static_for<o_len>([&](auto i) { v_o[i.value] *= scale; });
}

// Pin the O accumulator as a scheduling/materialization fence, chunked into 8-lane groups so
// each `"+v"` operand can be allocated.
template <class V>
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

// Masks every score column past `last_valid_kv_pos` to -inf. `kv_base_pos` is the absolute KV
// position of this wave's first score column: a tile's tokens are dealt KV_TILE_SIZE
// contiguous ones per wave, so the score tile a wave holds is its own group, not the tile.
template <class T, class V>
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

    int lane_id = opus::thread_id_x() % T::WARP_SIZE;
    asm volatile("" : "+v"(lane_id));
    const int lane_group = lane_id / T::W_M;

    opus::static_for<T::GEMM0_E_N>([&](auto i_n) {
        constexpr int base_idx = i_n.value * elems_per_wave_tile;
        const int k_pos        = kv_base_pos + i_n.value * T::W_N + lane_group * c_pack;
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

// --- Non-pipelined KV-tile loop for one work item ---
// Q, the O accumulator and the online-softmax state (m_row / l_row) are owned by the caller
// and passed by reference, so a split-KV request can run several tile ranges into the same
// accumulator.
template <class Traits, class VO>
__device__ __attribute__((always_inline)) void
mla_decode_fwd_simple(opus_mla_decode_fp8_kargs kargs,
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
    // kv_indices always stays on a descriptor -- it is int-indexed and never large. Entries
    // past the end read as 0 and land on a token whose scores are masked anyway.
    auto g_kv_indices = make_gmem(kargs.kv_indices + kv_ind_ptr_s, valid_kv_len * sizeof(int));

    // Q sits at the front of the KV region and is dead once every wave has read it into
    // registers; the barrier after that read is what lets the first KV DMA overwrite it.
    auto s_q  = make_smem(reinterpret_cast<D_Q*>(smem_buffer + T::smem_q_offset));
    auto s_kv = make_smem(reinterpret_cast<D_K*>(smem_buffer + T::smem_kv_offset));
    auto s_p  = make_smem(reinterpret_cast<D_K*>(smem_buffer + T::smem_p_offset));
    auto s_m  = make_smem(reinterpret_cast<D_ACC*>(smem_buffer + T::smem_ml_offset));
    auto s_l =
        make_smem(reinterpret_cast<D_ACC*>(smem_buffer + T::smem_ml_offset) + T::smem_ml_elems);

    auto u_gq_nope = make_layout_gq_nope<T>(warp_id, lane_id);
    auto u_sq_nope = make_layout_sq_nope<T>(warp_id);
    auto u_rq_nope = make_layout_rq_nope<T>(lane_id);
    auto u_gq_rope = make_layout_gq_rope<T>(lane_id);

    auto u_kv_idx_nope = make_layout_kv_indices_nope<T>(warp_id, lane_id);
    auto u_kv_idx_rope = make_layout_kv_indices_rope<T>(warp_id, lane_id);
    auto u_gkv_nope    = make_layout_gkv_nope<T>(lane_id);
    auto u_skv_nope    = make_layout_skv_nope<T>();
    auto u_gkv_rope    = make_layout_gkv_rope<T>(lane_id);
    auto u_skv_rope    = make_layout_skv_rope<T>();

    auto u_rk_nope0 = make_layout_rk_nope<T, 0>(lane_id);
    auto u_rk_nope1 = make_layout_rk_nope<T, 1>(lane_id);
    auto u_rk_rope0 = make_layout_rk_rope<T, 0>(lane_id);
    auto u_rk_rope1 = make_layout_rk_rope<T, 1>(lane_id);
    auto u_rv0      = make_layout_rv<T, 0>(lane_id);
    auto u_rv1      = make_layout_rv<T, 1>(lane_id);

    auto u_sp = make_layout_sp<T>(warp_id, lane_id);
    auto u_rp = make_layout_rp<T>(lane_id);

    // Element offsets into a KV slot. nope groups come first, then the rope ones.
    const int kv_nope_group = warp_id * static_cast<int>(T::smem_kv_nope_group_bytes);
    const int kv_rope_group = static_cast<int>(T::smem_kv_nope_bytes) +
                              warp_id * static_cast<int>(T::smem_kv_rope_group_bytes);
    // Which slot the tile being computed lives in. A runtime LDS byte offset, which is free:
    // only a runtime index into a REGISTER array would cost anything (it lands in scratch).
    int kv_slot_off = 0;
    // This wave's GEMM1 d slice is nope d_rpt block warp_id: WAVE_D == D_128B_NOPE_SIZE.
    static_assert(T::WAVE_D == T::D_128B_NOPE_SIZE, "a wave's output d must be one nope chunk");
    const int v_wave_base = warp_id * T::smem_n_rpt * T::smem_nope_line;

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
    auto g_kv_nope = kv_handle(kv_base, u_gkv_nope, number<T::VEC_KV_NOPE>{});
    auto g_kv_rope = kv_handle(kv_base + T::D_NOPE_SIZE, u_gkv_rope, number<T::VEC_KV_ROPE_LD>{});

    auto mfma0_nope =
        make_mfma<D_K, D_Q, D_ACC>(number<T::W_M>{}, number<T::W_N>{}, number<T::W_K_NOPE>{});
    auto mma0_rope = make_tiled_mma<D_K, D_Q, D_ACC>(seq<T::GEMM0_E_M, T::GEMM0_E_N, 1_I>{},
                                                     seq<1_I, 1_I, 1_I>{},
                                                     seq<T::W_M, T::W_N, T::W_K_ROPE>{},
                                                     mfma_adaptor_swap_ab{});
    auto mma1 = make_tiled_mma<D_K, D_K, D_ACC>(seq<T::GEMM1_E_M, T::GEMM1_E_N, T::GEMM1_E_K>{},
                                                seq<T::T_M, T::T_N, T::T_K>{},
                                                seq<T::W_M, T::W_N, T::W_K_ROPE>{},
                                                mfma_adaptor_swap_ab{});

    using k_nope_tile_t = vector_t<D_K, T::W_N * T::W_K_NOPE / T::WARP_SIZE>;
    using k_rope_tile_t = vector_t<D_K, T::W_N * T::W_K_ROPE / T::WARP_SIZE>;
    using v_tile_t      = vector_t<D_K, T::W_N * T::W_K_ROPE / T::WARP_SIZE>;
    using s_tile_t      = vector_t<D_ACC, T::W_M * T::W_N / T::WARP_SIZE>;

    vector_t<D_K, T::GEMM0_E_N * T::W_N * T::W_K_NOPE / T::WARP_SIZE> v_k_nope[2];
    vector_t<D_K, T::GEMM0_E_N * T::W_N * T::W_K_ROPE / T::WARP_SIZE> v_k_rope[2];
    typename decltype(mma0_rope)::vtype_c v_s;
    vector_t<D_K, T::GEMM1_K_STEPS * T::W_M * T::W_K_ROPE / T::WARP_SIZE> v_p;
    typename decltype(mma1)::vtype_b v_v[2];

    auto v_p_slices = reinterpret_cast<vector_t<D_K, T::W_M * T::W_K_ROPE / T::WARP_SIZE>*>(&v_p);
    auto v_o_slices =
        reinterpret_cast<vector_t<D_ACC, T::Q_TILE_SIZE * T::SLICE_D / T::WARP_SIZE>*>(&v_o);

    constexpr index_t s_len = vector_traits<typename decltype(mma0_rope)::vtype_c>::size();

    // Online softmax: skip the O rescale entirely while every lane's new row max is within
    // this much of the running one, so exp2(m_row - row_max) stays well inside fp32 range.
    constexpr D_ACC RESCALE_THRESHOLD = 8.0f;
    D_ACC rescale_m                   = 1.0f;

    // --- KV tile staging -------------------------------------------------------------
    auto kv_page_offset = [&](int token_idx) {
        if constexpr(T::LARGE_KV)
            return static_cast<int64_t>(token_idx) * kargs.stride_kv_page;
        else
            return static_cast<int>(static_cast<unsigned>(token_idx) *
                                    static_cast<unsigned>(kargs.stride_kv_page));
    };

    auto async_load_kv_nope = [&](int slot_off, int page, auto line) {
        const int s_off = slot_off + kv_nope_group + decltype(line)::value * T::smem_nope_line;
        if constexpr(T::LARGE_KV)
            global_load<T::VEC_KV_NOPE>(
                g_kv_nope + kv_page_offset(page), s_kv.ptr, u_gkv_nope, u_skv_nope + s_off);
        else
            async_load<T::VEC_KV_NOPE>(
                g_kv_nope, s_kv.ptr, u_gkv_nope + kv_page_offset(page), u_skv_nope + s_off);
    };
    auto async_load_kv_rope = [&](int slot_off, int page, auto line) {
        const int s_off = slot_off + kv_rope_group + decltype(line)::value * T::smem_rope_line;
        if constexpr(T::LARGE_KV)
            global_load<T::VEC_KV_ROPE_LD>(
                g_kv_rope + kv_page_offset(page), s_kv.ptr, u_gkv_rope, u_skv_rope + s_off);
        else
            async_load<T::VEC_KV_ROPE_LD>(
                g_kv_rope, s_kv.ptr, u_gkv_rope + kv_page_offset(page), u_skv_rope + s_off);
    };

    // Page indices of one tile: the four nope lines are four consecutive kv_indices entries
    // (one dwordx4), the two rope lines are rope_toks_per_line apart (two dwords). Indices
    // past the end of the request read as 0 through the descriptor and land in a slot whose
    // scores are masked, so the tile index needs no clamp.
    auto load_pages_nope = [&](int tile_idx) {
        return load<T::nope_lines_per_wave>(
            g_kv_indices, u_kv_idx_nope, tile_idx * T::KV_TILE_TOKENS);
    };
    auto load_pages_rope = [&](int tile_idx) {
        return load<1>(g_kv_indices, u_kv_idx_rope, tile_idx * T::KV_TILE_TOKENS);
    };
    auto async_load_kv = [&](int slot_off, const auto& pages, const auto& pages_rope) {
        static_for<T::nope_lines_per_wave>(
            [&](auto n) { async_load_kv_nope(slot_off, pages[n.value], n); });
        static_for<T::rope_lines_per_wave>(
            [&](auto m) { async_load_kv_rope(slot_off, pages_rope[m.value], m); });
    };

    auto stage_end_barrier = [&]() {
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    };

    // --- GEMM0 -----------------------------------------------------------------------
    // Both reads are issued per e_n tile so each carries its own compile-time swizzle.
    constexpr auto k_en_off = number<T::smem_n_rpt * T::D_128B_NOPE_SIZE>{};
    auto sk_nope_slice      = [](auto ek) {
        return number<decltype(ek)::value * T::smem_n_rpt * T::smem_nope_line>{};
    };

    auto load_k_nope = [&](auto& dst, auto slice) {
        const int base = kv_slot_off + kv_nope_group;
        auto* tile     = reinterpret_cast<k_nope_tile_t*>(&dst);
        tile[0]        = load<T::VEC_KV_NOPE>(s_kv, u_rk_nope0 + base + slice);
        tile[1]        = load<T::VEC_KV_NOPE>(s_kv, u_rk_nope1 + base + slice + k_en_off);
    };
    // The rope reads are fenced apart so the load/store optimiser cannot pair them into a
    // ds_read2_b64, which would read 16 B per lane out of one slot pair and conflict 2-way.
    auto load_k_rope = [&](auto& dst, auto ek) {
        auto* tile      = reinterpret_cast<k_rope_tile_t*>(&dst);
        const auto& u_r = [&]() -> const auto& {
            if constexpr(decltype(ek)::value == 0)
                return u_rk_rope0;
            else
                return u_rk_rope1;
        }();
        // A rope line is one e_n tile, so the two tiles are the group's two lines.
        const int base = kv_slot_off + kv_rope_group;
        tile[0]        = load<T::VEC_KV_ROPE>(s_kv, u_r + base);
        __builtin_amdgcn_sched_barrier(sched_masks::KEEP_DS_READ_ORDER);
        tile[1] = load<T::VEC_KV_ROPE>(s_kv, u_r + base + T::smem_rope_line);
    };

    // 4 e_k steps of 2 MFMA, each prefetching the K tile two steps ahead. `co` is emitted
    // after every MFMA, in its shadow: 8 chunks, which is what the KV prefetch is chopped
    // into (see prefetch_chunk). Hard fences around it because what it carries is
    // buffer_load_lds -- inline asm the scheduler must not be allowed to regroup back into
    // one block, which is the whole point of chopping it up.
    auto compute_qk_nope = [&](auto& s, auto& q, auto&& co) {
        clear(s);
        static_for<T::GEMM0_NOPE_E_K>([&](auto ek) {
            constexpr int idx  = ek.value;
            constexpr int slot = idx & 1;
            auto s_tile        = reinterpret_cast<s_tile_t*>(&s);
            auto k_tile        = reinterpret_cast<k_nope_tile_t*>(&v_k_nope[slot]);
            // The trailing 0,0 are the f8f6f4 block scales: per-tensor descale rides the
            // softmax temperature instead, and only a literal 0 selects the bare 8-byte form.
            s_tile[0] = mfma0_nope(k_tile[0], q[idx], s_tile[0], 0, 0);
            __builtin_amdgcn_sched_barrier(0);
            co(number<2 * idx>{});
            __builtin_amdgcn_sched_barrier(0);
            s_tile[1] = mfma0_nope(k_tile[1], q[idx], s_tile[1], 0, 0);
            __builtin_amdgcn_sched_barrier(0);
            co(number<2 * idx + 1>{});
            __builtin_amdgcn_sched_barrier(0);
            if constexpr(idx + 2 < T::GEMM0_NOPE_E_K)
            {
                load_k_nope(v_k_nope[slot], sk_nope_slice(number<idx + 2>{}));
                s_waitcnt_lgkmcnt(number<T::k_nope_ds_read_insts>{});
            }
            else if constexpr(idx + 1 < T::GEMM0_NOPE_E_K)
            {
                s_waitcnt_lgkmcnt(0_I);
            }
        });
    };
    auto no_co           = [](auto) {};
    auto compute_qk_rope = [&](auto& s, auto& q) {
        load_k_rope(v_k_rope[0], 0_I);
        __builtin_amdgcn_sched_barrier(sched_masks::KEEP_DS_READ_ORDER);
        load_k_rope(v_k_rope[1], 1_I);
        s_waitcnt_lgkmcnt(0_I);
        s = mma0_rope(q[0], v_k_rope[0], s);
        s = mma0_rope(q[1], v_k_rope[1], s);
    };

    // --- GEMM1 -----------------------------------------------------------------------
    // One flat walk of GEMM1_K_STEPS token groups x NUM_D_SLICES d slices, each step one mma1
    // (2 MFMA) plus the V read feeding the step two ahead. The step index is compile-time, so
    // the two v_v slots and the k-step's P slice never become runtime register indices.
    auto v_step_off = [&](auto step) {
        constexpr int s  = decltype(step)::value;
        constexpr int kk = s / T::NUM_D_SLICES;
        constexpr int ds = s % T::NUM_D_SLICES;
        return number<kk* static_cast<int>(T::smem_kv_nope_group_bytes) + ds * T::SLICE_D>{};
    };
    constexpr auto v_en_off = number<T::W_N>{};
    auto load_v             = [&](auto& dst, auto step) {
        const int base = kv_slot_off + v_wave_base;
        auto* half     = reinterpret_cast<v_tile_t*>(&dst);
        half[0]        = tr_load<T::VEC_TR_V>(s_kv, u_rv0 + base + v_step_off(step));
        half[1]        = tr_load<T::VEC_TR_V>(s_kv, u_rv1 + base + v_step_off(step) + v_en_off);
    };

    auto compute_pv = [&]() {
        load_v(v_v[0], 0_I);
        load_v(v_v[1], 1_I);
        s_waitcnt_lgkmcnt(number<T::v_ds_read_insts>{});
        // Load-bearing. tr_load is inline asm the compiler does not model, so this wait is
        // the only thing publishing step 0's V -- and without the fence the scheduler hoists
        // the first MFMA of step 0 above it (it sees no dependence), which reads a stale
        // register and puts a NaN in the e_n 0 half of d slice 0 of every wave. The fences
        // inside the loop below already pin every later step.
        __builtin_amdgcn_sched_barrier(0);
        static_for<T::PV_STEPS>([&](auto i) {
            constexpr int idx  = i.value;
            constexpr int slot = idx & 1;
            constexpr int kk   = idx / T::NUM_D_SLICES;
            constexpr int ds   = idx % T::NUM_D_SLICES;
            v_o_slices[ds]     = mma1(v_p_slices[kk], v_v[slot], v_o_slices[ds]);
            if constexpr(idx + 2 < T::PV_STEPS)
            {
                load_v(v_v[slot], number<idx + 2>{});
                s_waitcnt_lgkmcnt(number<T::v_ds_read_insts>{});
            }
            else if constexpr(idx + 1 < T::PV_STEPS)
            {
                s_waitcnt_lgkmcnt(0_I);
            }
            __builtin_amdgcn_sched_barrier(0);
        });
    };

    // --- masking / softmax -----------------------------------------------------------
    const u32_t neg_inf_v = std::bit_cast<u32_t>(-numeric_limits<D_ACC>::infinity());

    const int wave_kv_base = warp_id * T::KV_TILE_SIZE;
    auto mask_oob_scores   = [&](auto& s, int tile_idx) {
        bool masked = (tile_idx + 1) * T::KV_TILE_TOKENS > valid_kv_len;
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
            attn_mask_kv_tile<T>(s, bound, tile_idx * T::KV_TILE_TOKENS + wave_kv_base, neg_inf_v);
        }
    };

    auto softmax_tile = [&](auto& vs) {
        D_ACC row_max     = temperature_scale * attn_row_max<T>(vs, s_m, warp_id, lane_id);
        bool below_thresh = ((row_max - m_row) <= RESCALE_THRESHOLD);
        bool all_below =
            (__builtin_amdgcn_ballot_w64(below_thresh) == __builtin_amdgcn_read_exec());
        row_max = all_below ? m_row : max(m_row, row_max);
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
        l_row += attn_row_sum_local<T>(vs);
    };

    // --- Q staging ------------------------------------------------------------------
    // Q goes into the LAST slot, so at KV_SLOTS = 2 tile_begin's KV DMA (slot 0) can be put
    // on the wire alongside it and the two round trips overlap. The page indices are loaded
    // first so the KV DMA's address arithmetic does not have to wait on Q's traffic.
    auto pages      = load_pages_nope(tile_begin);
    auto pages_rope = load_pages_rope(tile_begin);
    async_load<T::VEC_Q_NOPE>(g_q_nope, s_q.ptr, u_gq_nope, u_sq_nope);
    auto v_q_rope = load<T::VEC_Q_ROPE>(g_q_rope, u_gq_rope);
    if constexpr(T::KV_SLOTS == 2)
    {
        async_load_kv(0, pages, pages_rope);
        // Next tile's indices, so the first iteration's prefetch waits on nothing.
        pages      = load_pages_nope(tile_begin + 1);
        pages_rope = load_pages_rope(tile_begin + 1);
    }
    // Everything except the KV tile and the indices just issued -- i.e. Q has landed. At one
    // slot there is no KV in flight (it would be writing the region Q is still in), so the
    // budget is just 0.
    constexpr int q_wait = T::KV_SLOTS == 2 ? T::kv_buffer_load_insts + T::kv_idx_load_insts : 0;
    s_waitcnt_vmcnt(number<q_wait>{});
    stage_end_barrier();

    auto v_q_nope = load<T::VEC_Q_NOPE>(s_q, u_rq_nope);
    s_waitcnt_lgkmcnt(0_I);
    auto v_q_nope_slices =
        reinterpret_cast<vector_t<D_Q, T::W_M * T::W_K_NOPE / T::WARP_SIZE>*>(&v_q_nope);
    auto v_q_rope_slices =
        reinterpret_cast<vector_t<D_Q, T::W_M * T::W_K_ROPE / T::WARP_SIZE>*>(&v_q_rope);

    // The whole body of one tile, out of whichever slot kv_slot_off points at. `prefetch_off`
    // is the slot tile t+1 goes into, or -1 for the last tile of the range, which must not
    // prefetch at all -- it would pull a whole tile of KV nobody reads.
    auto compute_tile = [&](int t, int prefetch_off) {
        // The tile t+1 prefetch, chopped into one chunk per QK MFMA. As a straight-line block
        // it is 18 buffer_load_lds plus 3 index loads with no MFMA anywhere near them, and
        // every LDS-DMA needs m0 rewritten with an s_nop behind it, so it is almost pure
        // issue latency that nothing covers -- the ISA showed all 18 back to back. Riding the
        // QK MFMA puts each piece in the shadow of the MFMA that just issued, and delays the
        // prefetch only by the QK region it now sits inside, not by the half tile that moving
        // it behind the softmax costs (measured: +5 to +8% on the memory-bound shapes).
        auto prefetch_chunk = [&](auto chunk) {
            constexpr int k    = decltype(chunk)::value;
            constexpr int nope = T::nope_lines_per_wave;
            constexpr int rope = T::rope_lines_per_wave;
            if constexpr(k < nope)
                async_load_kv_nope(prefetch_off, pages[k], number<k>{});
            else if constexpr(k < nope + rope)
                async_load_kv_rope(prefetch_off, pages_rope[k - nope], number<k - nope>{});
            // The indices for t+2 go last: the chunks above still need this tile's.
            else if constexpr(k == nope + rope)
                pages = load_pages_nope(t + 2);
            else if constexpr(k == nope + rope + 1)
                pages_rope = load_pages_rope(t + 2);
        };

        // --- GEMM0 on this wave's own token group, with the prefetch in its MFMA shadows.
        load_k_nope(v_k_nope[0], sk_nope_slice(0_I));
        load_k_nope(v_k_nope[1], sk_nope_slice(1_I));
        s_waitcnt_lgkmcnt(number<T::k_nope_ds_read_insts>{});
        __builtin_amdgcn_s_setprio(1);
        // One scalar branch for the whole region rather than one per chunk.
        if(prefetch_off >= 0)
            compute_qk_nope(v_s, v_q_nope_slices, prefetch_chunk);
        else
            compute_qk_nope(v_s, v_q_nope_slices, no_co);
        compute_qk_rope(v_s, v_q_rope_slices);
        __builtin_amdgcn_s_setprio(0);
        mask_oob_scores(v_s, t);

        // --- softmax; attn_row_max carries the one cross-wave rendezvous of the tile.
        softmax_tile(v_s);

        // --- P out and back in: a lane's eight scores are one ds_write_b64 into its own
        // block, and each of the four k-steps is one ds_read_b64 out of one wave's block.
        store<T::VEC_WRITE_P>(s_p, cast<D_K>(v_s), u_sp);
        s_waitcnt_lgkmcnt(0_I);
        stage_end_barrier();
        v_p = load<T::VEC_READ_P>(s_p, u_rp);
        __builtin_amdgcn_sched_barrier(sched_masks::KEEP_DS_READ_ORDER);
        s_waitcnt_lgkmcnt(0_I);

        // --- GEMM1 over all four token groups.
        __builtin_amdgcn_s_setprio(1);
        compute_pv();
        __builtin_amdgcn_s_setprio(0);
        // tr_load is inline asm the compiler does not model, so this wait is the only thing
        // keeping the last V reads in front of the barrier that frees the slot.
        s_waitcnt_lgkmcnt(0_I);
    };

    if constexpr(T::KV_SLOTS == 1)
    {
        // Floor build: one slot, so the tile's whole gmem latency sits in the open.
        stage_end_barrier(); // Q read done: the slot it aliases may be overwritten
        for(int t = tile_begin; t < tile_end; ++t)
        {
            async_load_kv(0, pages, pages_rope);
            pages      = load_pages_nope(t + 1);
            pages_rope = load_pages_rope(t + 1);
            s_waitcnt_vmcnt(number<T::kv_idx_load_insts>{});
            stage_end_barrier();
            // One slot: the tile being computed is the only one there is, so the prefetch
            // above already did the work and the QK region carries nothing.
            compute_tile(t, -1);
            stage_end_barrier();
        }
    }
    else
    {
        for(int t = tile_begin; t < tile_end; ++t)
        {
            s_waitcnt_vmcnt(number<T::kv_idx_load_insts>{});
            stage_end_barrier();

            const int nxt_slot_off = kv_slot_off ^ static_cast<int>(T::smem_kv_slot_bytes);
            // The prefetch itself is issued from inside the QK region, chunk by chunk; this
            // only picks the slot, or opts out on the range's last tile.
            compute_tile(t, t + 1 < tile_end ? nxt_slot_off : -1);
            kv_slot_off = nxt_slot_off;
        }
    }

    // The one cross-wave sum exchange for the whole range; until here l_row is this wave's
    // own tokens only.
    l_row = merge_row_sum<T>(l_row, s_l, warp_id, lane_id);
}

// --- One work item: load Q, run the tile range, normalize and store O (+ LSE) ---
// work_info_set is 8 ints per item, produced by the metadata kernel. A negative `slot` means
// this item owns the whole request and writes the real output; otherwise it is one split-KV
// partial and writes o_accum / lse_accum for the reduce kernel to merge.
template <class Traits>
__device__ __attribute__((always_inline)) void mla_decode_fwd_one_req(
    opus_mla_decode_fp8_kargs kargs, int w, char* smem_buffer, float temperature_scale)
{
    using namespace opus;
    using T     = opus::remove_cvref_t<Traits>;
    using D_ACC = typename T::D_ACC;
    using D_OUT = typename T::D_OUT;

    int lane_id = thread_id_x() % T::WARP_SIZE;
    asm volatile("" : "+v"(lane_id));
    const int warp_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);

    const opus_mla_decode_work_info work_item = kargs.work_info_set[w];
    [[maybe_unused]] const int batch_idx      = work_item.batch_idx;
    const int slot                            = work_item.partial_slot;
    const int q_len_ptr_s                     = work_item.qo_start;
    const int q_len_ptr_e                     = work_item.qo_end;
    const int kv_ind_ptr_s                    = work_item.kv_start;
    const int kv_ind_ptr_e                    = work_item.kv_end;

    const int q_len        = q_len_ptr_e - q_len_ptr_s;
    const int valid_kv_len = kv_ind_ptr_e - kv_ind_ptr_s;
    const int num_kv_tiles = ceil_div(valid_kv_len, T::KV_TILE_TOKENS);
    if(num_kv_tiles == 0)
        return;

    // Only the causal specialization needs the diagonal, so the two indptr scalar loads it
    // costs disappear entirely from the decode-only build.
    int causal_diagonal = 0;
    if constexpr(T::CAUSAL)
    {
        causal_diagonal = q_len_ptr_s - kv_ind_ptr_s +
                          __builtin_amdgcn_readfirstlane(kargs.kv_indptr[batch_idx + 1]) -
                          __builtin_amdgcn_readfirstlane(kargs.q_indptr[batch_idx + 1]);
    }

    // Per-tensor descale folded into the two places it can be a single scalar multiply: QK's
    // descale_q*descale_k rides the softmax temperature, and V's descale_k is applied once on
    // the finished O below (the PV MFMA itself consumes raw fp8).
    const float descale_q = readfirstlane_f32(reinterpret_cast<const float*>(kargs.q_scale_ptr)[0]);
    const float descale_k =
        readfirstlane_f32(reinterpret_cast<const float*>(kargs.kv_scale_ptr)[0]);
    const float qk_scale = readfirstlane_f32(temperature_scale * descale_q * descale_k);

    vector_t<D_ACC, T::Q_TILE_SIZE * T::WAVE_D / T::WARP_SIZE> v_o;
    clear(v_o);
    D_ACC m_row = opus::numeric_limits<D_ACC>::lowest();
    D_ACC l_row = 0.0f;
    mla_decode_fwd_simple<Traits>(kargs,
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

    // Softmax normalisation and the V descale in one multiply. l_row == 0 means every score
    // was masked, so O must be 0 rather than NaN.
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
        // m_row / l_row for a row, so one wave writes the whole Q_TILE_SIZE. lse_ptr is null
        // when the caller did not ask for LSE; lse_accum is always allocated.
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

} // namespace opus_mla_decode_fp8_16mx1_32nx4

// Persistent entry point: the grid is sized to the machine, not to the problem, and each block
// drains the work items the metadata kernel assigned it through work_indptr.
template <class Traits>
__global__
__launch_bounds__(Traits::BLOCK_SIZE,
                  2) void opus_mla_decode_fp8_16mx1_32nx4_kernel(opus_mla_decode_fp8_kargs kargs)
{
    using namespace opus;
    using namespace opus_mla_decode_fp8_16mx1_32nx4;
    using T = opus::remove_cvref_t<Traits>;

    const int work_id = block_id_x();

    // One KV slot (128 tokens at d = 576, 74 KB with the line padding) plus P and the softmax
    // scratch: 77.5 KB, so two blocks fit a CU and the four waves of a block see a partner on
    // every SIMD. The alignment is load-bearing: make_layout_rv folds the bank swizzle's XOR
    // into a +/- SWZ_D_BYTES stride, which is only equivalent while bit 4 of the block's base
    // address is zero.
    __shared__ __align__(128) char smem_buffer[T::smem_bytes()];

    const int work_idx_start = kargs.work_indptr[work_id];
    const int work_idx_end   = kargs.work_indptr[work_id + 1];
    if(work_idx_start >= work_idx_end)
        return;

    constexpr float LOG2_E        = 1.44269504089f;
    const float temperature_scale = readfirstlane_f32(kargs.softmax_scale * LOG2_E);
    for(int w = work_idx_start; w < work_idx_end; ++w)
    {
        __builtin_amdgcn_sched_barrier(0);
        // Q aliases the KV region, so the next item's Q DMA must not pass the last item's
        // reads of it. merge_row_sum's barrier is the last one inside the item and covers
        // everything but the store of O, which is gmem.
        __builtin_amdgcn_s_barrier();
        mla_decode_fwd_one_req<Traits>(kargs, w, smem_buffer, temperature_scale);
    }
}

#endif // !__HIP_DEVICE_COMPILE__ || !__gfx950__
