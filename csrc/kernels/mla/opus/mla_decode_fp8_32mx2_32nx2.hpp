#pragma once

// MLA decode forward on gfx950: fp8 Q x fp8 KV, 32mx2 / 32nx2, persistent scheduling.
//   GEMM0 (Q*K^T): d = 576 (nine 64-deep k-steps; the LDS row is 640)   GEMM1 (P*V): d_v = 512
//
// The mappings follow design_32mx2_32nx2.xlsx (Q_LOAD, K_LOAD, V_LOAD, V_P);
// /home/memin/m32x2/model.py is the closed-form model of all of them, checked against every
// labelled cell of the sheets.
//
//   (1) TWO WAVE PAIRS. Pair p = w / 2 owns packed query rows 32p .. 32p+31 (64 per work item).
//       Inside a pair, half h = w % 2 scores tokens 32h .. 32h+31 of each 64-token tile
//       (GEMM0, A = K, B = Q, one 32x32x64 f8f6f4 per k-step: 9 MFMA) and accumulates output
//       d 256h .. 256h+255 against all 64 tokens (GEMM1, A = V, B = P: 8 MFMA).
//
//   (2) KV (K_LOAD, V_LOAD): the 32mx4 image. A slot is 5 chunks of 128 d; a chunk 8 blocks
//       of 8 token rows x 128 B + 32 B pad, token t in block 4 (t / 32) + t % 4, row (t % 32)
//       / 4; rows 4..7 hold their 16 B d-groups XORed by one, applied on the DMA source and
//       undone by the K read (conflict-free b128) -- the V transpose read is conflict-free
//       either way. Half h's K is DMA pass h, i.e. blocks 4h .. 4h+3.
//
//   (3) Q (Q_LOAD) is staged once per work item through the same image in the third slot,
//       query row r in place of token r, and read with the K read: pair p's rows are pass p.
//
//   (4) SOFTMAX ACROSS THE PAIR (V_P). Each wave's S is [32 tokens][32 q]. The row max is
//       merged through LDS before any exponent; the row sum stays per wave, because both
//       halves apply the same rescale factors, and is merged once at the end. P: the PV B
//       operand byte b of lane l is token 32 (b / 16) + 16 (l / 32) + b % 16 -- its low 16 B
//       are half 0's tokens and its high 16 B half 1's, each exactly what the 32mx4 re-deal
//       (cast + two v_permlane32_swap) makes of one 32-token S tile. So every wave writes its
//       16 B half to LDS and reads back both halves of its pair.
//
// Registers (512 per lane at one wave per SIMD): Q 72 AGPR; K and V one shared ring of 9
// operands (72 AGPR) holding a whole tile's K, then a whole tile's V; O 8 tiles (128 VGPR)
// and the two 16-register score buffers in VGPR. The map holds only under the
// amdgpu-pin-op-dst toolchain, so aiter loads this kernel as a prebuilt code object.
//
// Software pipeline, one phase per tile, 3 KV slots. Phase t:
//   stage0 [mem]     vmcnt (DMA(t) landed) + barrier B0 (publishes K(t) and the pair's row
//                    maxima of S(t-1)); page indices of t+2; partner max; all of K(t)
//   stage1 [compute] gemm0 QK(t) [9 MFMA] || softmax tail(t-1) -> P(t-1) half to LDS
//                    || V(t-1) into the ring entries GEMM0 frees
//   stage2 [mem]     mask S(t); O rescale (rare); barrier B1 (P(t-1) published; every wave
//                    holds V(t-1) in registers, so slot t-1 is free); read P(t-1)
//   stage3 [compute] gemm1 PV(t-1) [8 MFMA] || DMA(t+2) into slot t-1 || row max of S(t)
// DMA(t+2) is consumed at B0 of phase t+2: one and a half phases of gmem latency in flight.

#include "mla_decode_traits.h"

#if !defined(__HIP_DEVICE_COMPILE__) || !defined(__gfx950__)

template <class Traits>
__global__ void opus_mla_decode_fp8_32mx2_32nx2_kernel(opus_mla_decode_kargs)
{
}

#else

#include <bit>
#include <cstddef>
#include <cstdint>

#include "mla_global_load.hpp"
#include <opus/opus.hpp>

using opus::operator""_I;

namespace mla_decode_fwd_32mx2_32nx2_fp8fp8 {

// Register pins, live only under the amdgpu-pin-op-dst toolchain. A pin binds a definition.
#if defined(__has_cpp_attribute)
#if __has_cpp_attribute(clang::amdgpu_pin_agpr) && __has_cpp_attribute(clang::amdgpu_pin_vgpr)
#define MLA32MX2_PIN_AGPR(n) [[clang::amdgpu_pin_agpr(n)]]
#define MLA32MX2_PIN_VGPR(n) [[clang::amdgpu_pin_vgpr(n)]]
#define MLA32MX2_HAS_PIN 1
#endif
#endif
#ifndef MLA32MX2_PIN_AGPR
#define MLA32MX2_PIN_AGPR(n)
#define MLA32MX2_PIN_VGPR(n)
#endif

// The register map. AGPR: Q, then the K/V ring. VGPR: O at the top, the two score buffers
// under it; v0 .. S_VGPR_BASE-1 are the compiler's (P, addresses, softmax temporaries).
template <class T>
constexpr int Q_AGPR_PER_STEP = T::OP_REGS; // 8
template <class T>
constexpr int KV_AGPR_BASE = T::GEMM0_E_K * Q_AGPR_PER_STEP<T>; // 72
template <class T>
constexpr int K_AGPR_PER_HALF = T::VEC_KV * sizeof(typename T::D_K) / 4; // 4
template <class T>
constexpr int O_VGPR_BASE = 256 - T::GEMM1_E_M * T::O_TILE_REGS; // 128
template <class T>
constexpr int S_VGPR_PER_BUF = T::O_TILE_REGS; // 16
template <class T>
constexpr int S_VGPR_BASE = O_VGPR_BASE<T> - 2 * S_VGPR_PER_BUF<T>; // 96

template <class T>
constexpr bool check_register_map()
{
    static_assert(KV_AGPR_BASE<T> + T::KV_ENTRIES * T::OP_REGS <= 256, "K/V ring past a255");
    static_assert(T::K_DEPTH == T::GEMM0_E_K && T::V_DEPTH == T::GEMM1_E_M,
                  "the phase map assumes a whole tile of K, then of V, in the ring");
    return true;
}

__device__ inline float readfirstlane_f32(float v)
{
    return std::bit_cast<float>(__builtin_amdgcn_readfirstlane(std::bit_cast<int>(v)));
}

// buffer_load ... lds from inline asm: M0 = the wave-uniform LDS destination, the hardware
// adds lane * BYTES. Asm so that the compiler does not know LDS-DMA is in flight (it would
// answer with vmcnt(0) in front of LDS reads). s_nop: an M0 write needs one wait state.
template <int BYTES>
__device__ inline void
buffer_load_lds_asm(const opus::vector_t<int, 4>& rsrc, opus::u32_t voff, opus::u32_t m0_val)
{
    static_assert(BYTES == 16 || BYTES == 4);
    if constexpr(BYTES == 16)
        asm volatile(
            "s_mov_b32 m0, %0\n\ts_nop 0\n\tbuffer_load_dwordx4 %1, %2, 0 offen lds" ::"s"(m0_val),
            "v"(voff),
            "s"(rsrc)
            : "memory");
    else
        asm volatile(
            "s_mov_b32 m0, %0\n\ts_nop 0\n\tbuffer_load_dword %1, %2, 0 offen lds" ::"s"(m0_val),
            "v"(voff),
            "s"(rsrc)
            : "memory");
}

template <opus::index_t VEC, class D, class LayoutG, class LayoutS>
__device__ inline void buffer_load_lds(const opus::vector_t<int, 4>& rsrc,
                                       const opus::smem<D>& s,
                                       const LayoutG& u_gmem,
                                       const LayoutS& u_smem)
{
    constexpr int size = static_cast<int>(sizeof(D));
    static_assert(opus::layout_load_traits<LayoutG, VEC>::r_elem.value == 1, "one issue per call");
    const auto v_os = static_cast<opus::u32_t>(opus::layout_to_offsets<VEC>(u_gmem)[0] * size);
    const auto s_os = static_cast<opus::u32_t>(opus::layout_to_offsets<VEC>(u_smem)[0] * size);
    const auto lds  = static_cast<opus::u32_t>(reinterpret_cast<__UINTPTR_TYPE__>(s.ptr));
    buffer_load_lds_asm<VEC * size>(rsrc, v_os, lds + s_os);
}

// LDS reads at a lane offset (the layout folded to one int, elements) plus a compile-time
// one. The constant goes onto the pointer after the lane offset, so it folds into the read's
// immediate and the lane offset stays one live VGPR.
template <opus::index_t VEC, int IMM, class D>
__device__ inline auto lds_load(const opus::smem<D>& s, int lane_off, opus::number<IMM>)
{
    using v_t          = opus::vector_t<D, VEC>;
    constexpr int size = static_cast<int>(sizeof(D));
    auto* lane         = s.ptr + lane_off * size;
    return *reinterpret_cast<OPUS_LDS_ADDR v_t*>(lane + IMM * size);
}

// One ds_read_b64_tr_b8 through the builtin (an asm result would be believed ready at once).
template <opus::index_t VEC, int IMM, class D>
__device__ inline auto tr_load_b8(const opus::smem<D>& s, int lane_off, opus::number<IMM>)
{
    static_assert(VEC * sizeof(D) == 8, "ds_read_b64_tr_b8 reads 8 bytes");
    using v2i_t        = opus::vector_t<int, 2>;
    constexpr int size = static_cast<int>(sizeof(D));
    auto* lane         = s.ptr + lane_off * size;
    auto* addr         = reinterpret_cast<OPUS_LDS_ADDR v2i_t*>(lane + IMM * size);
    return __builtin_bit_cast(opus::vector_t<D, VEC>, __builtin_amdgcn_ds_read_tr8_b64_v2i32(addr));
}

// ============================================================================================
// Layouts
// ============================================================================================

// --- KV / Q gmem->LDS (K_LOAD; Q_LOAD's LDS image) ---
//
// Pass p of wave w: lane l carries row 32 p + 4 (l / 8) + w (a token of the tile, or a query
// row of the work item) and lands at M0 + 16 l, i.e. row l / 8 of the wave's block, slot
// l % 8. The row each lane DMAs in each pass.
template <class T>
__device__ inline auto make_layout_kv_indices(int warp_id, int lane_id)
{
    constexpr auto shape = opus::make_tuple(opus::number<T::KV_PASSES>{},           // pass [y]
                                            opus::number<T::smem_rows_per_block>{}, // l / 8
                                            opus::number<T::NUM_WARPS>{},           // wave
                                            1_I);
    constexpr auto dim   = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{1_I}),
        opus::unfold_p_coord(dim, opus::tuple{lane_id / T::kv_threads_d, warp_id}));
}

// Global source, the d walk of one chunk (the row rides the per-lane offset). The lane at
// slot s of row r fetches d-group s ^ (r >= SWZ_ROW): the sheet's swizzle, on the source side
// because buffer_load_lds fixes where a lane lands. DG_MASK = 3 is the rope chunk, whose
// four pad groups re-read the four real ones.
template <class T, int DG_MASK>
__device__ inline auto make_layout_gkv(int lane_id)
{
    constexpr auto shape =
        opus::make_tuple(opus::number<T::kv_threads_d>{}, opus::number<T::VEC_KV>{});
    constexpr auto dim = opus::make_tuple(opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    const int row = lane_id / T::kv_threads_d;
    const int dg  = ((lane_id % T::kv_threads_d) ^ (row / T::SWZ_ROW)) & DG_MASK;

    return opus::make_layout(shape,
                             opus::unfold_x_stride(dim, shape, opus::tuple{1_I}),
                             opus::unfold_p_coord(dim, opus::tuple{dg}));
}

// LDS destination, wave-uniform (it feeds M0): block `warp_id` of a chunk. The pass
// (NUM_WARPS blocks) and the chunk are caller offsets.
template <class T>
__device__ inline auto make_layout_skv(int warp_id)
{
    constexpr auto shape =
        opus::make_tuple(opus::number<T::NUM_WARPS>{}, opus::number<T::VEC_KV>{});
    constexpr auto dim = opus::make_tuple(opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(shape,
                             opus::make_tuple(opus::number<T::smem_kv_block>{}, 1_I),
                             opus::unfold_p_coord(dim, opus::tuple{warp_id}));
}

// --- K / Q LDS->register: the A operand of GEMM0 (K, M = token) and its B operand (Q, N = q
//     row) have the same lane layout. Lane l holds row m = l % 32 of a 32-row group (K: the
//     wave's token half; Q: its pair) and, of k-step ks, the 16 B at d-group 4 (ks % 2) + 2 pp
//     + l / 32 of chunk ks / 2. Inverting the deal: block m % 4, row m / 4, slot d-group ^
//     (m >= 16); the d-group's only odd term is l / 32, so the XOR lands on it alone. The group
//     (4 blocks) and (ks, pp) are caller offsets.
//
// Banks: b128 is served 16 lanes at a time as lanes 8i .. 8i+7 with 8i+16 .. 8i+23. The first
// eight span blocks 0..3 x rows 0..1 (block pitch 264 dwords = 8 mod 64, row 32): 8 distinct
// 4-bank groups; the second eight are rows + 4 with the slot XORed, i.e. the other 8.
template <class T>
__device__ inline auto make_layout_rk(int lane_id)
{
    constexpr int blocks = T::NUM_WARPS;
    constexpr auto shape = opus::make_tuple(opus::number<blocks>{},                // m % 4
                                            opus::number<T::W_M / blocks>{},       // m / 4
                                            opus::number<T::WARP_SIZE / T::W_M>{}, // l / 32
                                            opus::number<T::VEC_KV>{});
    constexpr auto dim   = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    const int m   = lane_id % T::W_M;
    const int h   = lane_id / T::W_M;
    const int row = m / blocks;

    return opus::make_layout(
        shape,
        opus::make_tuple(opus::number<T::smem_kv_block>{},
                         opus::number<T::smem_row_kv>{},
                         opus::number<T::VEC_KV>{},
                         1_I),
        opus::unfold_p_coord(dim, opus::tuple{m % blocks, row, h ^ (row / T::SWZ_ROW)}));
}

// --- V LDS->register transpose read (ds_read_b64_tr_b8), V_LOAD ---
//
// In each 16-lane group lane 2i + hh supplies 8 B (token i of the group's run, d-half hh) and
// gets back d = l % 16 for all 8 tokens. Instruction q (0..3) of d-tile dt fills bytes
// 8q .. 8q+7, i.e. tokens 32 (q / 2) + 16 h + 8 (q % 2) + i, for lane l = 32 h + 16 dsub + 2 i
// + hh supplying d 32 dt + 16 dsub + 8 hh. The deal puts that token at
//
//   block 4 (q / 2) + i % 4,   row 4 h + 2 (q % 2) + i / 4,   slot 2 (dt % 4) + (dsub ^ h)
//
// -- the XOR is the swizzle, row >= 4 being exactly h. dt is the global d-tile: the wave's
// half (two chunks) is a runtime offset, (dt % 8, q) compile-time ones.
// Banks (b64: 32 lanes, h fixed): 8 (i % 4) + 32 (i / 4) + 4 slot + 2 hh covers all 64 banks.
template <class T>
__device__ inline auto make_layout_rv(int lane_id)
{
    constexpr int blocks = T::NUM_WARPS;
    constexpr auto shape = opus::make_tuple(opus::number<blocks>{},                // i % 4
                                            opus::number<T::WARP_SIZE / T::W_M>{}, // h
                                            opus::number<2>{},                     // i / 4
                                            opus::number<2>{},                     // dsub ^ h
                                            opus::number<2>{},                     // hh
                                            opus::number<T::VEC_TR_V>{});
    constexpr auto dim   = opus::make_tuple(opus::make_tuple(
        opus::p_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    const int h    = lane_id / T::W_M;
    const int dsub = (lane_id % T::W_M) / (T::W_M / 2);
    const int j    = lane_id % (T::W_M / 2);
    const int i = j / 2, hh = j % 2;

    return opus::make_layout(
        shape,
        opus::make_tuple(opus::number<T::smem_kv_block>{},
                         opus::number<T::SWZ_ROW * T::smem_row_kv>{},
                         opus::number<T::smem_row_kv>{},
                         opus::number<T::VEC_KV>{},
                         opus::number<T::VEC_TR_V>{},
                         1_I),
        opus::unfold_p_coord(dim, opus::tuple{i % blocks, h, i / blocks, dsub ^ h, hh}));
}

// --- P exchange (V_P) ---
//
// A wave's P half -- 16 B per lane, the re-dealt cast of its 32-token S tile -- goes to
// [pair][half][lane], 16 B per lane in lane order. That is conflict-free for both the b128
// write and the two b128 reads, measured: a slot XOR of (l & 16) >> 1, meant for b128 phases
// of lanes 8i.. with 8i+16.., cost 0.26 M conflict cycles (5% of LDS time) at b128 c4096 and
// plain lane order 0. The b128 phases on gfx950 evidently pair lane chunk c with c ^ 3 (the
// only regular pairing consistent with these two results and with the K read's swizzle
// being required), under which lane order covers all 64 banks.

// Write side: the wave's own half.
template <class T>
__device__ inline auto make_layout_sp(int warp_id, int lane_id)
{
    constexpr auto shape = opus::make_tuple(opus::number<T::NUM_WARPS>{}, // pair, half
                                            opus::number<T::WARP_SIZE>{}, // lane
                                            opus::number<T::VEC_P>{});
    constexpr auto dim =
        opus::make_tuple(opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(shape,
                             opus::unfold_x_stride(dim, shape, opus::tuple{1_I}),
                             opus::unfold_p_coord(dim, opus::tuple{warp_id, lane_id}));
}

// Read side: both halves of the pair, half 0 first -- the B operand's byte order.
template <class T>
__device__ inline auto make_layout_rp(int pair, int lane_id)
{
    constexpr auto shape = opus::make_tuple(opus::number<T::PAIRS>{},     // pair
                                            opus::number<T::HALVES>{},    // half [y]
                                            opus::number<T::WARP_SIZE>{}, // lane
                                            opus::number<T::VEC_P>{});
    constexpr auto dim   = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(shape,
                             opus::unfold_x_stride(dim, shape, opus::tuple{1_I}),
                             opus::unfold_p_coord(dim, opus::tuple{pair, lane_id}));
}

// --- row exchange: one float per lane per wave (both lanes of a query row hold the same
//     value). Written at the own wave, read at the partner's (wave ^ 1). ---
template <class T>
__device__ inline auto make_layout_srow(int wave, int lane_id)
{
    constexpr auto shape =
        opus::make_tuple(opus::number<T::NUM_WARPS>{}, opus::number<T::WARP_SIZE>{}, 1_I);
    constexpr auto dim =
        opus::make_tuple(opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(shape,
                             opus::unfold_x_stride(dim, shape, opus::tuple{1_I}),
                             opus::unfold_p_coord(dim, opus::tuple{wave, lane_id}));
}

// --- O register->gmem store, one d-tile (the caller adds the wave's half and 32 dt). PV's C
//     is [d][q]: lane l holds packed row 32 p + l % 32 and d = 8 (e / 4) + 4 (l / 32) + e % 4.
//     stride_o_h is a parameter so the same layout serves the output and the o_accum. ---
template <class T>
__device__ inline auto make_layout_o(int pair, int lane_id, int stride_o_h)
{
    constexpr auto shape = opus::make_tuple(opus::number<T::PAIRS>{},
                                            opus::number<T::W_N>{},
                                            opus::number<T::O_TILE_REGS / T::VEC_O>{}, // e / 4
                                            opus::number<T::WARP_SIZE / T::W_N>{},     // l / 32
                                            opus::number<T::VEC_O>{});
    constexpr auto dim =
        opus::make_tuple(opus::make_tuple(opus::p_dim{}, opus::p_dim{}),
                         opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{stride_o_h, 1_I}),
        opus::unfold_p_coord(dim, opus::tuple{pair, lane_id % T::W_N, lane_id / T::W_N}));
}

// P half (V_P): cast(S) re-dealt. S dword a holds tokens 8 a + 4 h + 0..3 of the wave's 32
// (h = l / 32); the operand half wants dword c to hold 16 h + 4 c + 0..3. Lane l keeps its own
// dwords 0 and 2 and trades for the partner's, which v_permlane32_swap(x, y) does in one go.
template <class T, class VS>
__device__ inline opus::vector_t<opus::u32_t, 4> build_p_half(const VS& vs)
{
    using u32x4_t   = opus::vector_t<opus::u32_t, 4>;
    const u32x4_t d = __builtin_bit_cast(u32x4_t, opus::cast<typename T::D_K>(vs));
    const auto s0   = __builtin_amdgcn_permlane32_swap(d[0], d[2], false, false);
    const auto s1   = __builtin_amdgcn_permlane32_swap(d[1], d[3], false, false);
    return u32x4_t{s0[0], s0[1], s1[0], s1[1]};
}

// ============================================================================================
// Softmax / scaling helpers. A query row is lane l % 32 in both halves of a wave.
// ============================================================================================
__device__ inline float row_combine_max(float x)
{
    const auto r = __builtin_amdgcn_permlane32_swap(
        std::bit_cast<opus::u32_t>(x), std::bit_cast<opus::u32_t>(x), false, true);
    return max(std::bit_cast<float>(r[0]), std::bit_cast<float>(r[1]));
}

__device__ inline float row_combine_sum(float x)
{
    const auto r = __builtin_amdgcn_permlane32_swap(
        std::bit_cast<opus::u32_t>(x), std::bit_cast<opus::u32_t>(x), false, true);
    return std::bit_cast<float>(r[0]) + std::bit_cast<float>(r[1]);
}

template <typename T, opus::index_t Offset, opus::index_t Count, typename V>
__device__ inline void attn_exp2_slice(V& v_s)
{
    opus::static_for<Count>([&](auto i) {
        constexpr opus::index_t idx = Offset + i.value;
        v_s[idx]                    = __builtin_amdgcn_exp2f(v_s[idx]);
    });
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
                 : "=s"(x_mask), "=s"(y_mask), "=&v"(x_ref), "=&v"(y_ref)
                 : "v"(x_ref), "v"(y_ref), "v"(rel_vgpr), "n"(THR_X), "v"(neg_inf_vgpr), "n"(THR_Y)
                 : "vcc");
}

// Masks every score column past the lane's `last_valid_kv_pos` to -inf. Element e of lane l
// is token 32 half + 8 (e / 4) + 4 (l / 32) + e % 4 of the tile, the MFMA's own C layout.
template <typename T, typename V>
__device__ inline void attn_mask_kv_tile(
    V& v_s, int last_valid_kv_pos, int kv_tile_idx, int half, opus::u32_t neg_inf_v)
{
    using D_ACC_X2 = opus::vector_t<typename T::D_ACC, 2>;
    using U32_X2   = opus::vector_t<opus::u32_t, 2>;

    int lane_id = opus::thread_id_x() % T::WARP_SIZE;
    asm volatile("" : "+v"(lane_id));
    const int k_pos = kv_tile_idx * T::KV_TILE_SIZE + half * T::W_M + (lane_id / T::W_N) * 4;
    const opus::u32_t rel = static_cast<opus::u32_t>(last_valid_kv_pos - k_pos);

    opus::static_for<T::O_TILE_REGS / 4>([&](auto a) {
        opus::static_for<2>([&](auto pr) {
            constexpr int idx   = a.value * 4 + pr.value * 2;
            constexpr int thr_x = a.value * 8 + pr.value * 2;
            constexpr int thr_y = thr_x + 1;

            auto pair_bits = __builtin_bit_cast(
                U32_X2, opus::slice(v_s, opus::number<idx>{}, opus::number<idx + 2>{}));
            opus::u32_t x_ref = pair_bits[0];
            opus::u32_t y_ref = pair_bits[1];
            attn_mask_vec2_imm<thr_x, thr_y>(rel, neg_inf_v, x_ref, y_ref);
            pair_bits[0] = x_ref;
            pair_bits[1] = y_ref;
            opus::set_slice(
                v_s, __builtin_bit_cast(D_ACC_X2, pair_bits), opus::number<idx>{}, opus::number<idx + 2>{});
        });
    });
}

// Register-class anchors: an empty asm with a tied operand is a use and a def of the value in
// that class. Free functions, because an asm operand naming a captured local does not compile
// inside a generic (static_for) lambda.
template <class X>
__device__ inline void anchor_agpr(X& x)
{
    static_assert(sizeof(X) % 4 == 0);
    using w_t = opus::vector_t<int, sizeof(X) / 4>;
    asm volatile("" : "+a"(*reinterpret_cast<w_t*>(&x))::);
}
template <class X>
__device__ inline void anchor_vgpr(X& x)
{
    static_assert(sizeof(X) % 4 == 0);
    using w_t = opus::vector_t<int, sizeof(X) / 4>;
    asm volatile("" : "+v"(*reinterpret_cast<w_t*>(&x))::);
}

template <class S>
__device__ inline opus::u32_t lds_addr(const S& s)
{
    return static_cast<opus::u32_t>(reinterpret_cast<__UINTPTR_TYPE__>(s.ptr));
}

template <class X, class S>
__device__ inline opus::smem<X> smem_at(const S& base, size_t off)
{
    opus::smem<X> s{nullptr};
    s.ptr = base.ptr + off;
    return s;
}

// ============================================================================================
// One work item: Q staged into registers, then the pipelined KV-tile loop (stage map at the
// top). O and the online-softmax state are the caller's.
// ============================================================================================
template <class Traits, class OT>
__device__ __attribute__((always_inline)) void
mla_decode_fwd_pipelined(opus_mla_decode_kargs kargs,
                         int q_len_ptr_s,
                         int q_len,
                         int kv_ind_ptr_s,
                         int valid_kv_len,
                         int tile_begin,
                         int tile_end,
                         char* smem_buffer,
                         OT (&v_o_tiles)[Traits::GEMM1_E_M],
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
    const int pair    = warp_id / T::HALVES;
    const int half    = warp_id % T::HALVES;

    // The last column each lane's row may see, and the wave's smallest such column (its first
    // row), which decides whether a tile needs the mask at all.
    int mask_bound = valid_kv_len - 1;
    int lane_bound = mask_bound;
    if constexpr(T::CAUSAL)
    {
        const int wave_diag = (pair * T::Q_TILE_SIZE) / kargs.H + causal_diagonal;
        const int lane_diag =
            (pair * T::Q_TILE_SIZE + lane_id % T::W_N) / kargs.H + causal_diagonal;
        mask_bound = wave_diag < mask_bound ? wave_diag : mask_bound;
        lane_bound = lane_diag < lane_bound ? lane_diag : lane_bound;
    }

    const D_K* kv_base = reinterpret_cast<const D_K*>(kargs.kv_ptr);
    auto g_kv_indices  = make_gmem(kargs.kv_indices + kv_ind_ptr_s, valid_kv_len * sizeof(int));
    auto g_q           = make_gmem(reinterpret_cast<const D_Q*>(kargs.q_ptr) +
                             q_len_ptr_s * kargs.stride_q_b,
                         static_cast<unsigned>(q_len * kargs.stride_q_b * sizeof(D_Q)));

    const auto lds              = make_smem(smem_buffer);
    smem<D_K> s_kv[T::KV_SLOTS] = {smem_at<D_K>(lds, 0),
                                   smem_at<D_K>(lds, T::smem_kv_slot_bytes),
                                   smem_at<D_K>(lds, 2 * T::smem_kv_slot_bytes)};
    auto s_p                    = smem_at<D_K>(lds, T::smem_p_offset);
    auto s_row                  = smem_at<D_ACC>(lds, T::smem_row_offset);
    auto s_kv_indices           = smem_at<int>(lds, T::smem_idx_offset);

    constexpr int rope_chunk = T::smem_d_rpt_kv - 1;
    constexpr int rope_g     = rope_chunk * T::smem_row_kv;
    auto u_kv_indices        = make_layout_kv_indices<T>(warp_id, lane_id);
    auto u_gkv               = make_layout_gkv<T, T::kv_threads_d - 1>(lane_id);
    auto u_gkv_r             = make_layout_gkv<T, T::kv_threads_d / 2 - 1>(lane_id);
    auto u_skv               = make_layout_skv<T>(warp_id);
    // The register-side lane offsets, folded once into one int each (the 32-row group and the
    // wave's d half included): read through the multi-coordinate layouts, every read re-adds
    // the coordinates and keeps them all live.
    const int rk_lane = layout_to_offsets<T::VEC_KV>(make_layout_rk<T>(lane_id))[0];
    const int u_rk    = rk_lane + half * T::NUM_WARPS * T::smem_kv_block;
    const int u_rq    = rk_lane + pair * T::NUM_WARPS * T::smem_kv_block;
    const int u_rv    = layout_to_offsets<T::VEC_TR_V>(make_layout_rv<T>(lane_id))[0] +
                     half * (T::WAVE_D / T::smem_row_kv) * T::smem_kv_chunk;
    const auto u_sp       = make_layout_sp<T>(warp_id, lane_id);
    const auto u_rp       = make_layout_rp<T>(pair, lane_id);
    const auto u_row_own  = make_layout_srow<T>(warp_id, lane_id);
    const auto u_row_peer = make_layout_srow<T>(warp_id ^ 1, lane_id);

    auto kv_handle = [&](auto u, auto d0) {
        if constexpr(T::LARGE_KV)
            return global_load_base<T::VEC_KV>(kv_base + decltype(d0)::value, u);
        else
            return make_gmem(kv_base,
                             static_cast<unsigned>(static_cast<size_t>(kargs.total_tokens) *
                                                   kargs.stride_kv_page * sizeof(D_K)));
    };
    auto g_kv                    = kv_handle(u_gkv, 0_I);
    [[maybe_unused]] auto g_kv_r = kv_handle(u_gkv_r, number<rope_g>{});

    // Both GEMMs: 32x32x64 f8f6f4. The trailing 0,0 are the block scales and must be literal
    // zeros. GEMM0 is (K, Q) so C is [token][q]; GEMM1 is (V, P) so C is [d][q].
    auto mfma0 = make_mfma<D_K, D_Q, D_ACC>(number<T::W_M>{}, number<T::W_N>{}, number<T::W_K>{});
    auto mfma1 = make_mfma<D_K, D_K, D_ACC>(number<T::W_M>{}, number<T::W_N>{}, number<T::W_K>{});

    using k_half_t  = vector_t<D_K, T::VEC_KV>;
    using op_t      = vector_t<D_K, T::W_M * T::W_K / T::WARP_SIZE>;   // 32, one operand
    using s_tile_t  = vector_t<D_ACC, T::W_M * T::W_N / T::WARP_SIZE>; // 16
    using v_piece_t = vector_t<D_K, T::VEC_TR_V>;
    static_assert(check_register_map<T>());

    // The K/V ring: entry e holds K k-step e during QK, then V d-tile e.
    op_t v_kv[T::KV_ENTRIES];
    // Scores for two tiles: tile t's QK runs alongside tile t-1's softmax tail.
    s_tile_t v_s[2];
    auto clear_s = [](auto& a, auto sbuf) {
        MLA32MX2_PIN_VGPR(S_VGPR_BASE<T> + decltype(sbuf)::value * S_VGPR_PER_BUF<T>)
        a = s_tile_t{0};
    };
    op_t v_p;
    op_t v_q[T::GEMM0_E_K];

    auto pin_q = [&]() { static_for<T::GEMM0_E_K>([&](auto i) { anchor_agpr(v_q[i.value]); }); };
    auto pin_o = [&]() { static_for<T::GEMM1_E_M>([&](auto i) { anchor_vgpr(v_o_tiles[i.value]); }); };

    // Inside a slot: k-step ks is chunk ks / 2 and 64 d on for odd ks, the operand's second
    // pass 32 d on. PV d-tile dt (of the wave's half) is chunk dt / 4 plus 2 slots per tile;
    // instruction q is one pass per q / 2 and two rows per q % 2.
    auto k_off = [](auto ks, auto pp) {
        constexpr int s = decltype(ks)::value;
        return number<(s / 2) * T::smem_kv_chunk +
                      ((s % 2) * (T::W_K / T::VEC_KV) + decltype(pp)::value * 2) * T::VEC_KV>{};
    };
    auto v_off = [](auto dt, auto q) {
        constexpr int d  = decltype(dt)::value;
        constexpr int qq = decltype(q)::value;
        constexpr int tc = T::v_tiles_per_chunk;
        return number<(d / tc) * T::smem_kv_chunk + (qq / 2) * T::NUM_WARPS * T::smem_kv_block +
                      (qq % 2) * 2 * T::smem_row_kv + (d % tc) * 2 * T::VEC_KV>{};
    };

    // One operand (k-step ks) of the 32-row group `u` reads: two ds_read_b128.
    auto load_op = [&](op_t& dst, smem<D_K> s, int u, auto ks, auto agpr_base) {
        auto* h = reinterpret_cast<k_half_t*>(&dst);
        static_for<2>([&](auto pp) {
            MLA32MX2_PIN_AGPR(decltype(agpr_base)::value + pp.value * K_AGPR_PER_HALF<T>)
            h[pp.value] = lds_load<T::VEC_KV>(s, u, k_off(ks, pp));
        });
    };
    auto load_k_all = [&](smem<D_K> kv) {
        static_for<T::K_DEPTH>([&](auto ks) {
            load_op(v_kv[ks.value], kv, u_rk, ks, number<KV_AGPR_BASE<T> + ks.value * T::OP_REGS>{});
        });
    };

    // V d-tile dt of the tile in slot `kv`: four transpose reads into ring entry dt.
    auto load_v = [&](smem<D_K> kv, auto dt) {
        constexpr int base = KV_AGPR_BASE<T> + decltype(dt)::value * T::OP_REGS;
        auto* piece        = reinterpret_cast<v_piece_t*>(&v_kv[dt.value]);
        static_for<T::v_tile_ds_read_insts>([&](auto q) {
            MLA32MX2_PIN_AGPR(base + q.value * (T::VEC_TR_V / 4))
            piece[q.value] = tr_load_b8<T::VEC_TR_V>(kv, u_rv, v_off(dt, q));
        });
    };

    // GEMM0: 9 k-steps of one MFMA, K step ks in ring entry ks. `per_step(ks)` rides the
    // shadow of step ks. The MFMA is a pure call, so IR passes sink the whole chain below the
    // per-step work and sched_barrier, a backend fence, cannot undo it: the V reads end up
    // ahead of the first MFMA, which then waits out ~28 LDS reads (lgkmcnt saturates at 15)
    // instead of its own two. The anchor on S ties each MFMA to its place among the volatile
    // asm, the memory fence keeps the V reads behind it.
    auto compute_qk = [&](auto& s, auto sbuf, auto&& per_step) {
        constexpr int sbase = S_VGPR_BASE<T> + decltype(sbuf)::value * S_VGPR_PER_BUF<T>;
        static_for<T::GEMM0_E_K>([&](auto ks) {
            anchor_agpr(v_kv[ks.value]);
            MLA32MX2_PIN_VGPR(sbase)
            s = mfma0(v_kv[ks.value], v_q[ks.value], s, 0, 0);
            anchor_vgpr(s);
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("" ::: "memory");
            per_step(ks);
        });
    };

    // GEMM1 out of the ring: one MFMA per d-tile; `co(dt)` rides the shadow of MFMA dt. The O
    // anchor holds each MFMA among the DMA issues, as the S anchor does in GEMM0.
    auto compute_pv = [&](auto&& co) {
        static_for<T::GEMM1_E_M>([&](auto dt) {
            constexpr int d = dt.value;
            anchor_agpr(v_kv[d]);
            MLA32MX2_PIN_VGPR(O_VGPR_BASE<T> + d * T::O_TILE_REGS)
            v_o_tiles[d] = mfma1(v_kv[d], v_p, v_o_tiles[d], 0, 0);
            anchor_vgpr(v_o_tiles[d]);
            __builtin_amdgcn_sched_barrier(0);
            co(dt);
        });
    };
    auto no_co = [](auto) {};

    constexpr index_t s_len = T::O_TILE_REGS; // 16

    // Online softmax: skip the O rescale entirely while every lane's new row max is within this
    // much of the running one. Decided by a ballot over identical values in both waves of a
    // pair (the merged max), so the pair always takes the same branch.
    // It must be 0, i.e. rescale whenever the max moves: with the exact running max each
    // row's largest P is exp2(0) = 1.0, which fp8 holds exactly, while any lag makes the
    // dominant terms inexact. Against an fp64 reference on b128 c100 the max error is 0.021
    // at 0 (the asm kernel's), 0.037 at 2 and 0.059 at 8.
    constexpr D_ACC RESCALE_THRESHOLD = 0.0f;
    D_ACC rescale_m                   = 1.0f;
    bool all_below                    = true;
    D_ACC own_max; // raw row max of the last headed tile, this wave's 32 tokens
    D_ACC peer_max;
    D_ACC row_sum;

    // The softmax tail of the previous tile, one chunk per GEMM0 k-step:
    //   0  merged max, rescale decision, m / l update   1,2  scale-subtract (8 each)
    //   3,4  exp2 (8 each)                               5  row-sum tree
    //   6  sum across the lane halves, l update; P cast, re-deal and store to LDS
    auto tail_chunk = [&](auto& vs, auto ks) {
        constexpr int k = decltype(ks)::value;
        if constexpr(k == 0)
        {
            D_ACC row_max = temperature_scale * max(own_max, peer_max);
            all_below     = __builtin_amdgcn_ballot_w64((row_max - m_row) <= RESCALE_THRESHOLD) ==
                        __builtin_amdgcn_read_exec();
            row_max   = all_below ? m_row : max(m_row, row_max);
            rescale_m = __builtin_amdgcn_exp2f(m_row - row_max);
            l_row *= rescale_m;
            m_row = row_max;
        }
        else if constexpr(k == 1 || k == 2)
        {
            static_for<8>([&](auto i) {
                constexpr int e = (k - 1) * 8 + i.value;
                vs[e]           = __builtin_fmaf(vs[e], temperature_scale, -m_row);
            });
        }
        else if constexpr(k == 3 || k == 4)
        {
            attn_exp2_slice<T, (k - 3) * 8, 8>(vs);
        }
        else if constexpr(k == 5)
        {
            D_ACC part[8];
            static_for<8>([&](auto i) { part[i.value] = vs[i.value] + vs[8 + i.value]; });
            static_for<4>([&](auto i) { part[i.value] += part[4 + i.value]; });
            static_for<2>([&](auto i) { part[i.value] += part[2 + i.value]; });
            row_sum = part[0] + part[1];
        }
        else if constexpr(k == 6)
        {
            l_row += row_combine_sum(row_sum);
            const auto ph = build_p_half<T>(vs);
            store<T::VEC_P>(s_p, __builtin_bit_cast(vector_t<D_K, T::VEC_P>, ph), u_sp);
        }
        // Redefine what the chunk wrote in place, so it stays in its MFMA shadow: pure VALU
        // otherwise sinks to the P store and runs as one block.
        using g8_t = vector_t<D_ACC, 8>;
        auto* grp  = reinterpret_cast<g8_t*>(&vs);
        if constexpr(k == 1 || k == 2)
            anchor_vgpr(grp[k - 1]);
        if constexpr(k == 3 || k == 4)
            anchor_vgpr(grp[k - 3]);
    };
    constexpr int TAIL_CHUNKS = 7;
    static_assert(TAIL_CHUNKS <= T::GEMM0_E_K, "the tail must fit the GEMM0 shadows");

    // O is at the max of tile t-2 until the tail of t-1 moved m: bring it along before PV(t-1).
    auto rescale_o = [&]() {
        if(!all_below)
        {
            static_for<T::GEMM1_E_M>([&](auto i) {
                MLA32MX2_PIN_VGPR(O_VGPR_BASE<T> + i.value * T::O_TILE_REGS)
                v_o_tiles[i.value] = v_o_tiles[i.value] * rescale_m;
            });
        }
    };

    // The softmax head of tile t: this wave's raw row max, published for the partner.
    auto head_chunk = [&](auto& vs, auto dt) {
        constexpr int k = decltype(dt)::value;
        if constexpr(k == 0)
        {
            own_max = vs[0];
            static_for<s_len / 2 - 1>([&](auto i) { own_max = max(own_max, vs[i.value + 1]); });
        }
        else if constexpr(k == 1)
        {
            static_for<s_len / 2>([&](auto i) { own_max = max(own_max, vs[s_len / 2 + i.value]); });
            own_max = row_combine_max(own_max);
            store<1>(s_row, vector_t<D_ACC, 1>{own_max}, u_row_own);
        }
    };

    // Page indices go through LDS: one buffer_load_dword lds per wave per tile puts the tile's
    // 64 indices in the wave's own ring entry, four tiles ahead, and the two a lane needs come
    // back as plain ds_reads once the phase's vmcnt has covered the DMA. Every lane's token is
    // clamped into the request, so a tile past the end repeats the last token's page.
    const auto idx_tok = layout_to_offsets<1>(u_kv_indices); // row of each pass
    auto idx_slot      = [&](int tile_idx) {
        return (((tile_idx - tile_begin) & (T::IDX_RING - 1)) * T::NUM_WARPS + warp_id) *
               static_cast<int>(T::smem_idx_tile);
    };
    vector_t<int, 4> idx_rsrc;
    __builtin_memcpy(&idx_rsrc, &g_kv_indices.cached_rsrc, sizeof(idx_rsrc));
    auto async_load_kv_indices = [&](int tile_idx) {
        const int tok = min(tile_idx * T::KV_TILE_SIZE + lane_id, valid_kv_len - 1);
        buffer_load_lds_asm<sizeof(int)>(idx_rsrc,
                                         static_cast<u32_t>(tok) * sizeof(int),
                                         lds_addr(s_kv_indices) +
                                             static_cast<u32_t>(idx_slot(tile_idx)));
    };
    auto load_kv_pages = [&](int tile_idx) {
        vector_t<int, T::KV_PASSES> pg;
        static_for<T::KV_PASSES>([&](auto p) {
            pg[p.value] = s_kv_indices.template _load<1>(
                idx_slot(tile_idx) + idx_tok[p.value] * static_cast<int>(sizeof(int)))[0];
        });
        return pg;
    };
    auto kv_page_offset = [&](int token_idx) {
        if constexpr(T::LARGE_KV)
            return static_cast<int64_t>(token_idx) * kargs.stride_kv_page;
        else
            return static_cast<int>(static_cast<unsigned>(token_idx) *
                                    static_cast<unsigned>(kargs.stride_kv_page));
    };
    using poff_t     = decltype(kv_page_offset(0));
    auto dma_offsets = [&](const auto& pages) {
        vector_t<poff_t, T::KV_PASSES> po;
        static_for<T::KV_PASSES>([&](auto p) { po[p.value] = kv_page_offset(pages[p.value]); });
        return po;
    };

    // KV gmem->LDS, one instruction at a time: DMA i of a tile is pass i / 5, chunk i % 5 (the
    // last chunk being rope + pad). A tile past the range still gets its DMA, so every phase
    // issues the same count and the vmcnt budget is a constant.
    [[maybe_unused]] vector_t<int, 4> kv_rsrc;
    if constexpr(!T::LARGE_KV)
        __builtin_memcpy(&kv_rsrc, &g_kv.cached_rsrc, sizeof(kv_rsrc));
    auto async_load_kv_chunk = [&](smem<D_K> kv, const auto& po, auto i) {
        constexpr int p     = decltype(i)::value / T::smem_d_rpt_kv;
        constexpr int c     = decltype(i)::value % T::smem_d_rpt_kv;
        constexpr int s_off = c * T::smem_kv_chunk + p * T::NUM_WARPS * T::smem_kv_block;
        if constexpr(T::LARGE_KV)
        {
            if constexpr(c < rope_chunk)
                global_load<T::VEC_KV>(
                    g_kv + (po[p] + c * T::smem_row_kv), kv.ptr, u_gkv, u_skv + s_off);
            else
                global_load<T::VEC_KV>(g_kv_r + po[p], kv.ptr, u_gkv_r, u_skv + s_off);
        }
        else
        {
            if constexpr(c < rope_chunk)
                buffer_load_lds<T::VEC_KV>(
                    kv_rsrc, kv, u_gkv + (po[p] + c * T::smem_row_kv), u_skv + s_off);
            else
                buffer_load_lds<T::VEC_KV>(kv_rsrc, kv, u_gkv_r + (po[p] + rope_g), u_skv + s_off);
        }
    };
    auto async_load_kv = [&](smem<D_K> kv, const auto& pages) {
        const auto po = dma_offsets(pages);
        static_for<T::kv_buffer_load_insts>([&](auto i) { async_load_kv_chunk(kv, po, i); });
    };

    // Q gmem->LDS: the KV DMA with query rows for tokens (contiguous, stride_q_h apart) and the
    // work item's Q as the buffer. Rows past q_len * H read zeros.
    vector_t<int, 4> q_rsrc;
    __builtin_memcpy(&q_rsrc, &g_q.cached_rsrc, sizeof(q_rsrc));
    auto async_load_q = [&](smem<D_K> s) {
        static_for<T::q_buffer_load_insts>([&](auto i) {
            constexpr int p     = decltype(i)::value / T::smem_d_rpt_kv;
            constexpr int c     = decltype(i)::value % T::smem_d_rpt_kv;
            constexpr int s_off = c * T::smem_kv_chunk + p * T::NUM_WARPS * T::smem_kv_block;
            const int row_off   = idx_tok[p] * kargs.stride_q_h;
            if constexpr(c < rope_chunk)
                buffer_load_lds<T::VEC_Q>(
                    q_rsrc, s, u_gkv + (row_off + c * T::smem_row_kv), u_skv + s_off);
            else
                buffer_load_lds<T::VEC_Q>(q_rsrc, s, u_gkv_r + (row_off + rope_g), u_skv + s_off);
        });
    };

    const u32_t neg_inf_v = std::bit_cast<u32_t>(-numeric_limits<D_ACC>::infinity());
    auto mask_oob_scores  = [&](auto& s, int tile_idx) {
        if((tile_idx + 1) * T::KV_TILE_SIZE - 1 > mask_bound)
            attn_mask_kv_tile<T>(s, lane_bound, tile_idx, half, neg_inf_v);
    };

    auto stage_end = [&]() { __builtin_amdgcn_sched_barrier(0); };
    // A workgroup barrier over LDS. The asm memory clobbers keep the compiler from moving an
    // LDS access across it; a real fence would also drain vmcnt, i.e. the KV DMA. s_barrier
    // does not wait for a wave's own LDS traffic, so every caller has an lgkmcnt(0) in front.
    auto lds_barrier = [&]() {
        asm volatile("" ::: "memory");
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("" ::: "memory");
    };
    auto read_peer_max = [&]() { peer_max = load<1>(s_row, u_row_peer)[0]; };
    auto read_p        = [&]() { v_p = __builtin_bit_cast(op_t, load<T::VEC_P>(s_p, u_rp)); };

    // --- Prologue ---
    // The previous work item may still have reads of its slots and its last (unused) DMA in
    // flight; nothing below may touch LDS before every wave is past both.
    __builtin_amdgcn_s_waitcnt(0);
    lds_barrier();
    pin_o();

    // Index DMAs of tiles tile_begin .. +3, the Q image into the third slot, tiles tile_begin
    // and tile_begin+1 into slots 0 and 1.
    static_for<T::IDX_RING>([&](auto i) { async_load_kv_indices(tile_begin + i.value); });
    __builtin_amdgcn_s_waitcnt(0);
    asm volatile("" ::: "memory");
    const auto pages_a = load_kv_pages(tile_begin);
    const auto pages_b = load_kv_pages(tile_begin + 1);
    const auto pages_c = load_kv_pages(tile_begin + 2);
    async_load_q(s_kv[T::Q_SLOT]);
    async_load_kv(s_kv[0], pages_a);
    async_load_kv(s_kv[1], pages_b);
    s_waitcnt_vmcnt(number<T::kv_buffer_load_insts>{}); // Q and tile_begin landed
    lds_barrier();
    static_for<T::GEMM0_E_K>([&](auto ks) {
        load_op(v_q[ks.value], s_kv[T::Q_SLOT], u_rq, ks, number<ks.value * Q_AGPR_PER_STEP<T>>{});
    });
    s_waitcnt_lgkmcnt(0_I);
    pin_q();
    lds_barrier(); // every wave is done with the Q image: the slot is tile_begin+2's now
    // What the phase of tile_begin would have issued, so the first loop phase's vmcnt holds.
    async_load_kv(s_kv[T::Q_SLOT], pages_c);
    async_load_kv_indices(tile_begin + T::IDX_RING);

    // Partial phase for tile_begin: QK and the row max. No tail, no PV.
    load_k_all(s_kv[0]);
    stage_end();
    clear_s(v_s[0], 0_I);
    compute_qk(v_s[0], 0_I, [](auto) {});
    mask_oob_scores(v_s[0], tile_begin);
    head_chunk(v_s[0], 0_I);
    head_chunk(v_s[0], 1_I);
    s_waitcnt_lgkmcnt(0_I);
    anchor_vgpr(v_s[0]);
    stage_end();

    // kv_prev: slot t-1 (its V read this phase, then DMA(t+2) into it), kv_cur: slot t.
    auto run_phase = [&](auto& vs_cur, auto& vs_prev, smem<D_K> kv_prev, smem<D_K> kv_cur, int t,
                         auto sbuf) {
        // stage0 [mem]: wait out DMA(t) and the index DMA of t+2, leaving DMA(t+1) and the
        // index DMA of t+3 in flight. B0 publishes K(t) and the pair's row maxima of S(t-1).
        s_waitcnt_vmcnt(number<T::kv_buffer_load_insts + T::kv_index_load_insts>{});
        lds_barrier();
        pin_q();
        read_peer_max();
        const auto pages = load_kv_pages(t + 2);
        load_k_all(kv_cur);
        stage_end();

        // stage1 [compute]: gemm0 QK(t) [9 MFMA] || softmax tail(t-1) -> P half to LDS
        // || V(t-1) d-tile ks into the ring entry step ks has just finished with.
        __builtin_amdgcn_s_setprio(1);
        clear_s(vs_cur, sbuf);
        compute_qk(vs_cur, sbuf, [&](auto ks) {
            if constexpr(ks.value < TAIL_CHUNKS)
                tail_chunk(vs_prev, ks);
            if constexpr(ks.value < T::V_DEPTH)
                load_v(kv_prev, ks);
        });
        __builtin_amdgcn_s_setprio(0);
        stage_end();

        // stage2 [mem]: mask S(t); bring O to the new max; B1 publishes P(t-1), and every wave
        // has V(t-1) home, so slot t-1 is free for DMA(t+2).
        mask_oob_scores(vs_cur, t);
        rescale_o();
        const auto po = dma_offsets(pages);
        s_waitcnt_lgkmcnt(0_I);
        lds_barrier();
        read_p();
        stage_end();

        // stage3 [compute]: gemm1 PV(t-1) [8 MFMA] || DMA(t+2) into slot t-1, then the index
        // DMA of t+4 || the row max of S(t).
        __builtin_amdgcn_s_setprio(1);
        pin_o();
        compute_pv([&](auto dt) {
            constexpr int d     = dt.value;
            constexpr int n     = T::kv_buffer_load_insts;
            constexpr int first = d * n / T::GEMM1_E_M;
            constexpr int last  = (d + 1) * n / T::GEMM1_E_M;
            static_for<last - first>(
                [&](auto j) { async_load_kv_chunk(kv_prev, po, number<first + j.value>{}); });
            if constexpr(d == T::GEMM1_E_M - 1)
                async_load_kv_indices(t + T::IDX_RING);
            head_chunk(vs_cur, dt);
        });
        // The max store must land before B0 publishes it; the P read before B0 lets the
        // partner overwrite P.
        s_waitcnt_lgkmcnt(0_I);
        pin_o();
        anchor_vgpr(vs_cur);
        __builtin_amdgcn_s_setprio(0);
        stage_end();
    };

    // --- Main loop: tiles tile_begin+1 .. tile_end-1, two phases unrolled per iteration so the
    //     two score buffers alternate as compile-time indices. The slot handles rotate
    //     (t-1, t, t+1) -> (t, t+1, t+2) each phase.
    smem<D_K> kv_prev = s_kv[0], kv_cur = s_kv[1], kv_next = s_kv[2];
    auto rotate = [&]() {
        const smem<D_K> kv = kv_prev;
        kv_prev            = kv_cur;
        kv_cur             = kv_next;
        kv_next            = kv;
    };
    // An odd phase count peels one phase up front and moves its scores into v_s[0], so the
    // loop always starts and ends there and the epilogue needs no parity.
    int t = tile_begin + 1;
    if(((tile_end - 1 - tile_begin) & 1) == 1)
    {
        __builtin_amdgcn_sched_barrier(0);
        run_phase(v_s[1], v_s[0], kv_prev, kv_cur, t, 1_I);
        rotate();
        MLA32MX2_PIN_VGPR(S_VGPR_BASE<T>)
        v_s[0] = v_s[1];
        ++t;
        __builtin_amdgcn_sched_barrier(0);
    }
    for(; t < tile_end; t += 2)
    {
        __builtin_amdgcn_sched_barrier(0);
        run_phase(v_s[1], v_s[0], kv_prev, kv_cur, t, 1_I);
        rotate();
        __builtin_amdgcn_sched_barrier(0);
        run_phase(v_s[0], v_s[1], kv_prev, kv_cur, t + 1, 0_I);
        rotate();
        __builtin_amdgcn_sched_barrier(0);
    }
    __builtin_amdgcn_sched_barrier(0);

    // --- Epilogue: softmax tail + gemm1 of the last tile, which sits in kv_prev and v_s[0].
    //     The barriers stand in for the B0 / B1 a next phase would have had.
    lds_barrier();
    read_peer_max();
    static_for<TAIL_CHUNKS>([&](auto k) { tail_chunk(v_s[0], k); });
    static_for<T::V_DEPTH>([&](auto dt) { load_v(kv_prev, dt); });
    rescale_o();
    stage_end();
    s_waitcnt_lgkmcnt(0_I);
    lds_barrier();
    read_p();
    pin_o();
    compute_pv(no_co);
    pin_o();
    __builtin_amdgcn_sched_barrier(0);
}

// --- One work item: run the tile range, normalize and store O (+ LSE) ---
// A negative `slot` means this item owns the whole request and writes the real output;
// otherwise it is one split-KV partial and writes o_accum / lse_accum for the reduce kernel.
template <class Traits>
__device__ __attribute__((always_inline)) void mla_decode_fwd_one_req(
    opus_mla_decode_kargs kargs, int w, char* smem_buffer, float temperature_scale)
{
    using namespace opus;
    using T     = opus::remove_cvref_t<Traits>;
    using D_ACC = typename T::D_ACC;
    using D_OUT = typename T::D_OUT;

    int lane_id = thread_id_x() % T::WARP_SIZE;
    asm volatile("" : "+v"(lane_id));
    const int warp_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
    const int pair    = warp_id / T::HALVES;
    const int half    = warp_id % T::HALVES;

    const opus_mla_decode_work_info work_item = kargs.work_info_set[w];
    [[maybe_unused]] const int batch_idx      = work_item.batch_idx;
    const int slot                            = work_item.partial_slot;
    const int q_len_ptr_s                     = work_item.qo_start;
    const int q_len_ptr_e                     = work_item.qo_end;
    const int kv_ind_ptr_s                    = work_item.kv_start;
    const int kv_ind_ptr_e                    = work_item.kv_end;

    const int q_len        = q_len_ptr_e - q_len_ptr_s;
    const int valid_kv_len = kv_ind_ptr_e - kv_ind_ptr_s;
    const int num_kv_tiles = ceil_div(valid_kv_len, T::KV_TILE_SIZE);
    if(num_kv_tiles == 0)
        return;

    int causal_diagonal = 0;
    if constexpr(T::CAUSAL)
    {
        causal_diagonal = q_len_ptr_s - kv_ind_ptr_s +
                          __builtin_amdgcn_readfirstlane(kargs.kv_indptr[batch_idx + 1]) -
                          __builtin_amdgcn_readfirstlane(kargs.q_indptr[batch_idx + 1]);
    }

    // Per-tensor descale: QK's descale_q * descale_k rides the softmax temperature, V's
    // descale_k is applied once on the finished O.
    const float descale_q = readfirstlane_f32(reinterpret_cast<const float*>(kargs.q_scale_ptr)[0]);
    const float descale_k =
        readfirstlane_f32(reinterpret_cast<const float*>(kargs.kv_scale_ptr)[0]);
    const float qk_scale = readfirstlane_f32(temperature_scale * descale_q * descale_k);

    using o_tile_t = vector_t<D_ACC, T::O_TILE_REGS>;
    o_tile_t v_o_tiles[T::GEMM1_E_M];
    static_for<T::GEMM1_E_M>([&](auto i) {
        MLA32MX2_PIN_VGPR(O_VGPR_BASE<T> + i.value * T::O_TILE_REGS)
        v_o_tiles[i.value] = o_tile_t{0};
    });
    D_ACC m_row = opus::numeric_limits<D_ACC>::lowest();
    D_ACC l_row = 0.0f;
    mla_decode_fwd_pipelined<Traits>(kargs,
                                     q_len_ptr_s,
                                     q_len,
                                     kv_ind_ptr_s,
                                     valid_kv_len,
                                     0,
                                     num_kv_tiles,
                                     smem_buffer,
                                     v_o_tiles,
                                     m_row,
                                     l_row,
                                     qk_scale,
                                     causal_diagonal);

    // The pair's row sum: each half summed its own tokens under the same rescale factors.
    // The epilogue's last barrier has every partner read of the row region behind it.
    auto s_row = smem_at<D_ACC>(make_smem(smem_buffer), T::smem_row_offset);
    store<1>(s_row, vector_t<D_ACC, 1>{l_row}, make_layout_srow<T>(warp_id, lane_id));
    s_waitcnt_lgkmcnt(0_I);
    asm volatile("" ::: "memory");
    __builtin_amdgcn_s_barrier();
    asm volatile("" ::: "memory");
    l_row += load<1>(s_row, make_layout_srow<T>(warp_id ^ 1, lane_id))[0];

    // Softmax normalisation and the V descale in one multiply. l == 0 means every score was
    // masked, so O must be 0 rather than NaN.
    const D_ACC o_scale        = (l_row > D_ACC(0.0f)) ? (descale_k / l_row) : D_ACC(0.0f);
    constexpr float INV_LOG2_E = 0.69314718055994531f; // ln(2)
    const D_ACC lse            = (l_row > D_ACC(0.0f)) ? ((m_row + log2f(l_row)) * INV_LOG2_E)
                                                       : opus::numeric_limits<D_ACC>::lowest();
    const bool is_final     = slot < 0;
    const int o_gmem_offset = q_len_ptr_s * kargs.stride_o_b;
    // partial_slot is negative on the final path; keep the unused o_accum address in bounds.
    const int oa_offset     = is_final ? 0 : slot * kargs.stride_o_b;
    auto g_o                = make_gmem(reinterpret_cast<D_OUT*>(kargs.out_ptr) + o_gmem_offset,
                         q_len * kargs.stride_o_b * sizeof(D_OUT));
    auto g_oa               = make_gmem(reinterpret_cast<D_ACC*>(kargs.o_accum) + oa_offset,
                          q_len * kargs.stride_o_b * sizeof(D_ACC));
    auto u_o                = make_layout_o<T>(pair, lane_id, kargs.stride_o_h) + half * T::WAVE_D;
    auto u_oa               = make_layout_o<T>(pair, lane_id, T::D_NOPE_SIZE) + half * T::WAVE_D;
    static_for<T::GEMM1_E_M>([&](auto dt) {
        const o_tile_t v = v_o_tiles[dt.value] * o_scale;
        if(is_final)
            store<T::VEC_O>(g_o, cast<D_OUT>(v), u_o + dt.value * T::W_M);
        else
            store<T::VEC_O>(g_oa, v, u_oa + dt.value * T::W_M);
    });
    // lse_ptr is null when the caller did not ask for LSE; lse_accum in the split-KV case is
    // always allocated. Half 0 of each pair stores its 32 rows.
    D_ACC* lse_base = nullptr;
    if(!is_final)
        lse_base = reinterpret_cast<D_ACC*>(kargs.lse_accum) + slot * kargs.H;
    else if(kargs.lse_ptr != nullptr)
        lse_base = reinterpret_cast<D_ACC*>(kargs.lse_ptr) + q_len_ptr_s * kargs.H;
    if(lse_base != nullptr && half == 0 && lane_id < T::W_N)
    {
        auto g_lse = make_gmem(lse_base, q_len * kargs.H * sizeof(D_ACC));
        g_lse.store(lse, pair * T::Q_TILE_SIZE + lane_id);
    }
}

// Persistent body: each block drains the work items the metadata kernel assigned it through
// work_indptr. One block per CU: 3 KV slots are 124 KB of LDS, which is also what gives one
// wave per SIMD the whole 512-register file (Q 72 + K/V 72 AGPR, O 128 + S 32 VGPR).
template <class Traits>
__device__ __attribute__((always_inline)) void
mla_decode_fwd_persistent(opus_mla_decode_kargs kargs)
{
    using namespace opus;
    using T = opus::remove_cvref_t<Traits>;

    const int work_id = block_id_x();

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
        mla_decode_fwd_one_req<Traits>(kargs, w, smem_buffer, temperature_scale);
    }
}

} // namespace mla_decode_fwd_32mx2_32nx2_fp8fp8

template <class Traits>
__global__
__launch_bounds__(Traits::BLOCK_SIZE,
                  1) void opus_mla_decode_fp8_32mx2_32nx2_kernel(opus_mla_decode_kargs kargs)
{
    mla_decode_fwd_32mx2_32nx2_fp8fp8::mla_decode_fwd_persistent<Traits>(kargs);
}

#endif // !__HIP_DEVICE_COMPILE__ || !__gfx950__
