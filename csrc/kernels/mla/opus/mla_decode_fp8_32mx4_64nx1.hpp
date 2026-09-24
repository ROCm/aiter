#pragma once

// MLA decode forward on gfx950: fp8 Q x fp8 KV, 32mx4 / 64nx1, persistent scheduling.
//   GEMM0 (Q*K^T): d = 576 (nine 64-deep k-steps; the LDS row is 640)   GEMM1 (P*V): d_v = 512
//
// HIP port of the SP3 kernel MLA_A8W8_QH32_1TG_4W_32mx4_64nx1_PS. The mappings follow
// design_32mx4_64nx1.xlsx (Q_LOAD, K_LOAD, V_LOAD, V_P); /home/memin/tmp/m32/model.py is the
// closed-form model of all of them, checked against every labelled cell of the sheets.
//
//   (1) ONE DECOMPOSITION. Wave w owns packed query rows 32w .. 32w+31 against the whole
//       64-token tile (GEMM0) and the whole 512-d output (GEMM1), both on the full-rate
//       32x32x64 f8f6f4: 18 + 16 MFMA per tile. Softmax state and P never leave the wave.
//
//   (2) KV (K_LOAD). The 16mx4 image: a slot is 5 chunks of 128 d; a chunk is 8 blocks of 8
//       token rows x 128 B + 32 B pad, block 4 * (t / 32) + t % 4, row (t % 32) / 4. Rows 4..7
//       of a block carry their 16 B d-groups XORed by one (the DMA lane fetches the swapped
//       group), which the K read undoes; it takes that read from 2-way to conflict-free.
//       The last chunk's pad half re-reads the token's rope (same cache lines).
//
//   (3) V (V_LOAD) is the fp8 nope of K, transpose-read with ds_read_b64_tr_b8 in natural
//       token order: lane l of d-tile dt holds d = 32 dt + l % 32 and, byte b, token
//       32 (b / 16) + 16 (l / 32) + b % 16. descale_k is applied once, on the finished O.
//
//   (4) P (V_P). The QK C tile has lane l holding tokens 8 (e / 4) + 4 (l / 32) + e % 4, the PV
//       B operand wants the natural order above, so cast(S) is re-dealt between lanes l and
//       l + 32 by two v_permlane32_swap per 32-token tile.
//
// Registers (512 per lane at one wave per SIMD): Q 72 AGPR, O 7 tiles in AGPR (112) + 9 in
// VGPR (144), S two 32-register ping-pong buffers in VGPR, K and V one shared 32-register ring
// in AGPR (2 GEMM0 k-steps / 4 PV d-tiles). The ring cannot be VGPR: O's half and both score
// buffers leave ~48 VGPRs, which P, addresses and the softmax need; a deeper ring (4 k-steps /
// 8 d-tiles) spills. The map holds only under the amdgpu-pin-op-dst toolchain -- see the
// kernel entry at the end -- so aiter loads this kernel as a prebuilt code object.
//
// Software pipeline, one phase per tile, 3 KV slots. Phase t:
//   stage0 [mem]     vmcnt (DMA(t) landed) + barrier (publishes K(t); every wave is done
//                    with PV(t-2), so slot t+1 is free); page indices of t+1; K head of t
//   stage1 [compute] gemm0 QK(t) [18 MFMA, DMA(t+1) one per k-step]
//                    || softmax tail(t-1) -> P(t-1), a chunk per k-step
//                    || V head of t-1 in the last two k-steps, into the K entries they free
//   stage2 [mem]     mask S(t) (scalar branch, last tiles only)
//   stage3 [compute] gemm1 PV(t-1) [16 MFMA, V ring refilled behind each]
//                    || softmax head(t), chopped into per-d-tile chunks; then O rescale
// With SPLIT_DMA (traits) stage1 carries only chunks 2, 3 of DMA(t+1), and a second barrier
// halfway through stage3 -- slot t-1's chunks 0, 1, 4 read out -- sends chunks 0, 1, 4 of
// DMA(t+2) a phase early. Measured with ATT (b128 c8192): the MFMA runs of both GEMMs are
// saturated and ~26% of a phase was vmcnt on the KV DMA, so what is left is bytes in flight,
// not issue order; at ~5 TB/s the kernel sits near the scattered-row DRAM ceiling.

#include "mla_decode_traits.h"

#if !defined(__HIP_DEVICE_COMPILE__) || !defined(__gfx950__)

template <class Traits>
__global__ void opus_mla_decode_fp8_32mx4_64nx1_kernel(opus_mla_decode_fp8_kargs)
{
}

#else

#include <bit>
#include <cstddef>
#include <cstdint>

#include "mla_global_load.hpp"
#include <opus/opus.hpp>

using opus::operator""_I;

namespace mla_decode_fwd_32mx4_64nx1_fp8fp8 {

// Register pins. Only the amdgpu-pin-op-dst toolchain knows the attributes, and only there do
// the macros expand to them, so the .co build and a stock compile share one source. A pin
// binds a *definition*: every value below is pinned where it is produced.
#if defined(__has_cpp_attribute)
#if __has_cpp_attribute(clang::amdgpu_pin_agpr) && __has_cpp_attribute(clang::amdgpu_pin_vgpr)
#define MLA32MX4_PIN_AGPR(n) [[clang::amdgpu_pin_agpr(n)]]
#define MLA32MX4_PIN_VGPR(n) [[clang::amdgpu_pin_vgpr(n)]]
#define MLA32MX4_HAS_PIN 1
#endif
#endif
#ifndef MLA32MX4_PIN_AGPR
#define MLA32MX4_PIN_AGPR(n)
#define MLA32MX4_PIN_VGPR(n)
#endif

// The register map. AGPR: Q, the AGPR half of O, the K/V ring. VGPR: the other half of O at the
// top, the two score buffers under it; v0 .. S_VGPR_BASE-1 are the compiler's (P, addresses,
// softmax temporaries).
template <class T>
constexpr int Q_AGPR_PER_STEP = T::W_N * T::W_K * sizeof(typename T::D_Q) / T::WARP_SIZE / 4; // 8
template <class T>
constexpr int O_AGPR_BASE = T::GEMM0_E_K * Q_AGPR_PER_STEP<T>; // 72
template <class T>
constexpr int KV_AGPR_BASE = O_AGPR_BASE<T> + T::O_AGPR_TILES * T::O_TILE_REGS; // 184
template <class T>
constexpr int K_AGPR_PER_HALF = T::VEC_KV * sizeof(typename T::D_K) / 4; // 4
template <class T>
constexpr int K_AGPR_PER_STEP = T::GEMM0_E_N * 2 * K_AGPR_PER_HALF<T>; // 16
template <class T>
constexpr int V_AGPR_PER_TILE = T::W_M * T::W_K * sizeof(typename T::D_K) / T::WARP_SIZE / 4; // 8
template <class T>
constexpr int O_VGPR_BASE = 256 - T::O_VGPR_TILES * T::O_TILE_REGS; // 112
template <class T>
constexpr int S_VGPR_PER_BUF = T::GEMM0_E_N * T::O_TILE_REGS; // 32
template <class T>
constexpr int S_VGPR_BASE = O_VGPR_BASE<T> - 2 * S_VGPR_PER_BUF<T>; // 48

template <class T>
constexpr bool check_register_map()
{
    static_assert(KV_AGPR_BASE<T> + T::K_DEPTH * K_AGPR_PER_STEP<T> <= 256, "K ring past a255");
    static_assert(KV_AGPR_BASE<T> + T::V_DEPTH * V_AGPR_PER_TILE<T> <= 256, "V ring past a255");
    return true;
}

// Moves a wave-uniform float into an SGPR. The bit_cast is required, not cosmetic:
// __builtin_amdgcn_readfirstlane takes an int, so handing it a float converts the value.
__device__ inline float readfirstlane_f32(float v)
{
    return std::bit_cast<float>(__builtin_amdgcn_readfirstlane(std::bit_cast<int>(v)));
}

// buffer_load ... lds from inline asm: M0 = the wave-uniform LDS destination, the hardware
// adds lane * BYTES. Asm so that the compiler does not know LDS-DMA is in flight: it keeps
// alias books on LDS-DMA, cannot tell the KV slots apart, and answers with a vmcnt(0) in
// front of LDS reads. s_nop: an M0 write needs one wait state before an LDS-DMA.
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

// ============================================================================================
// Layouts
// ============================================================================================

// --- Q gmem->register (Q_LOAD). The B operand of GEMM0: lane l holds packed row 32w + l % 32
//     and, of k-step ks, the 16 B at d = 64 ks + 32 pp + 16 (l / 32) for pp in [0, 2). The rope
//     is simply k-step 8 of the combined 576-d row. Register order: ks, pp, byte. ---
template <class T>
__device__ inline auto make_layout_q(int warp_id, int lane_id)
{
    constexpr auto shape = opus::make_tuple(opus::number<T::T_M>{},
                                            opus::number<T::W_N>{},
                                            opus::number<T::GEMM0_E_K>{},          // ks
                                            opus::number<T::W_K / T::W_K_HALF>{},  // pp
                                            opus::number<T::WARP_SIZE / T::W_N>{}, // l / 32
                                            opus::number<T::VEC_Q>{});
    constexpr auto dim   = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{opus::number<T::D_HEAD_SIZE>{}, 1_I}),
        opus::unfold_p_coord(dim, opus::tuple{warp_id, lane_id % T::W_N, lane_id / T::W_N}));
}

// --- KV gmem->LDS (K_LOAD) ---
//
// Pass p of wave w: lane l carries token 32 p + 4 (l / 8) + w and lands at M0 + 16 l, i.e. row
// l / 8 of the wave's block, slot l % 8. The token each lane DMAs in each pass (the caller adds
// tile_idx * KV_TILE_SIZE).
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

// KV global source, the d walk of one chunk (the token rides the per-lane page offset). The
// lane at slot s of row r fetches d-group s ^ (r >= SWZ_ROW): the sheet's swizzle, applied on
// the source side because buffer_load_lds fixes where each lane lands. DG_MASK = 3 is the
// rope chunk, whose four pad groups fold back onto the four real ones (and the XOR stays
// inside them).
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

// KV LDS destination, wave-uniform because buffer_load_lds' dst is: block `warp_id` of a
// chunk. The pass (NUM_WARPS blocks) and the chunk are caller offsets.
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

// K LDS->register, the A operand of GEMM0 (K_LOAD register block): lane l holds token
// 32 nt + m (m = l % 32) and, of k-step ks, the 16 B at d-group 4 (ks % 2) + 2 pp + l / 32 of
// chunk ks / 2. Inverting the deal: block m % 4 (+4 for nt), row m / 4, slot d-group ^ (m >= 16)
// -- and since the d-group's only odd term is l / 32, the XOR lands on that coordinate alone.
// (nt, ks, pp) are caller offsets (k_off).
//
// Banks: b128 is served 16 lanes at a time as lanes 8i .. 8i+7 with 8i+16 .. 8i+23. The first
// eight span blocks 0..3 x rows 0..1 (block pitch 264 dwords = 8 mod 64, row 32): 8 distinct
// 4-bank groups; the second eight are the same rows + 4 with their slot XORed, i.e. the other
// 8. Without the swizzle both halves hit the same 32 banks.
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
// -- the XOR is the swizzle, row >= 4 being exactly h. (dt, q) are constants (v_off).
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

// O register->gmem store, one d-tile (the caller adds 32 dt). PV's C is [d][q]: lane l holds
// packed row 32w + l % 32 and d = 8 (e / 4) + 4 (l / 32) + e % 4. stride_o_h is a parameter
// because the same layout serves the real output and the D_NOPE_SIZE-strided o_accum.
template <class T>
__device__ inline auto make_layout_o(int warp_id, int lane_id, int stride_o_h)
{
    constexpr auto shape = opus::make_tuple(opus::number<T::T_M>{},
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
        opus::unfold_p_coord(dim, opus::tuple{warp_id, lane_id % T::W_N, lane_id / T::W_N}));
}

// P (V_P): cast(S) re-dealt into the PV B operand. S dword a of token tile nt holds tokens
// 32 nt + 8 a + 4 h + 0..3 (h = l / 32); the operand wants dword c of that tile to hold
// 32 nt + 16 h + 4 c + 0..3. Lane l keeps its own dwords 0 and 2 (h = 0) and trades for the
// partner's, which v_permlane32_swap(x, y) does in one go: it exchanges x's upper half with
// y's lower half.
// One token tile's quarter of the operand: dwords 4 nt .. 4 nt + 3. Split out so the softmax tail
// can build P a token tile at a time, each in its own MFMA shadow.
template <class T, int NT, class VS>
__device__ inline opus::vector_t<opus::u32_t, 4> build_p_tile(const VS& vs)
{
    using u32x4_t   = opus::vector_t<opus::u32_t, 4>;
    const u32x4_t d = __builtin_bit_cast(
        u32x4_t,
        opus::cast<typename T::D_K>(opus::slice(
            vs, opus::number<NT * T::O_TILE_REGS>{}, opus::number<(NT + 1) * T::O_TILE_REGS>{})));
    const auto s0 = __builtin_amdgcn_permlane32_swap(d[0], d[2], false, false);
    const auto s1 = __builtin_amdgcn_permlane32_swap(d[1], d[3], false, false);
    return u32x4_t{s0[0], s0[1], s1[0], s1[1]};
}

// ============================================================================================
// Softmax / scaling helpers. A query row is lane l % 32 in both halves of the wave, so every
// cross-lane reduction is one permlane32_swap.
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

template <typename T, typename V>
__device__ inline typename T::D_ACC attn_row_max(const V& v_s)
{
    constexpr opus::index_t s_len = opus::vector_traits<V>::size();
    typename T::D_ACC m           = v_s[0];
    opus::static_for<s_len - 1>([&](auto i) { m = max(m, v_s[i.value + 1]); });
    return row_combine_max(m);
}

template <typename T, opus::index_t Offset, opus::index_t Count, typename V>
__device__ inline void attn_exp2_slice(V& v_s)
{
    opus::static_for<Count>([&](auto i) {
        constexpr opus::index_t idx = Offset + i.value;
        v_s[idx]                    = __builtin_amdgcn_exp2f(v_s[idx]);
    });
}

// The online-softmax rescale of one AGPR-resident O tile, spelled out on the AGPRs so the
// allocator keeps a single register class for it (a plain VALU multiply makes it home the tile
// in VGPRs). The caller owes an s_nop for the VALU-write -> MFMA-srcC hazard, which the hazard
// recognizer does not see through inline asm.
template <typename Tile>
__device__ inline Tile scale_tile_agpr(Tile t, float s)
{
    constexpr int n = opus::vector_traits<Tile>::size();
    static_assert(n % 4 == 0);
    float e[n];
#pragma unroll
    for(int i = 0; i < n; i++)
        e[i] = t[i];
#pragma unroll
    for(int i = 0; i < n; i += 4)
    {
        float t0, t1, t2, t3;
        asm volatile("v_accvgpr_read_b32 %4, %0\n\t"
                     "v_accvgpr_read_b32 %5, %1\n\t"
                     "v_accvgpr_read_b32 %6, %2\n\t"
                     "v_accvgpr_read_b32 %7, %3\n\t"
                     "v_mul_f32 %4, %4, %8\n\t"
                     "v_mul_f32 %5, %5, %8\n\t"
                     "v_mul_f32 %6, %6, %8\n\t"
                     "v_mul_f32 %7, %7, %8\n\t"
                     "v_accvgpr_write_b32 %0, %4\n\t"
                     "v_accvgpr_write_b32 %1, %5\n\t"
                     "v_accvgpr_write_b32 %2, %6\n\t"
                     "v_accvgpr_write_b32 %3, %7"
                     : "+a"(e[i]),
                       "+a"(e[i + 1]),
                       "+a"(e[i + 2]),
                       "+a"(e[i + 3]),
                       "=&v"(t0),
                       "=&v"(t1),
                       "=&v"(t2),
                       "=&v"(t3)
                     : "v"(s));
    }
#pragma unroll
    for(int i = 0; i < n; i++)
        t[i] = e[i];
    return t;
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

// Masks every score column past the lane's `last_valid_kv_pos` to -inf. Element (nt, e) of lane
// l is token 32 nt + 8 (e / 4) + 4 (l / 32) + e % 4 of the tile, the MFMA's own C layout.
template <typename T, typename V>
__device__ inline void
attn_mask_kv_tile(V& v_s, int last_valid_kv_pos, int kv_tile_idx, opus::u32_t neg_inf_v)
{
    using D_ACC_X2 = opus::vector_t<typename T::D_ACC, 2>;
    using U32_X2   = opus::vector_t<opus::u32_t, 2>;

    int lane_id = opus::thread_id_x() % T::WARP_SIZE;
    asm volatile("" : "+v"(lane_id));
    const int k_pos       = kv_tile_idx * T::KV_TILE_SIZE + (lane_id / T::W_N) * 4;
    const opus::u32_t rel = static_cast<opus::u32_t>(last_valid_kv_pos - k_pos);

    opus::static_for<T::GEMM0_E_N>([&](auto nt) {
        opus::static_for<T::O_TILE_REGS / 4>([&](auto a) {
            opus::static_for<2>([&](auto pr) {
                constexpr int idx   = nt.value * T::O_TILE_REGS + a.value * 4 + pr.value * 2;
                constexpr int thr_x = nt.value * T::W_M + a.value * 8 + pr.value * 2;
                constexpr int thr_y = thr_x + 1;

                auto pair_bits = __builtin_bit_cast(
                    U32_X2, opus::slice(v_s, opus::number<idx>{}, opus::number<idx + 2>{}));
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

// Register-class anchors: an empty asm with a tied operand is a use and a def of the value in
// that class, which the allocator honours by homing the value there. Free functions rather
// than lambdas, because an asm operand naming a captured local does not compile inside a
// generic (static_for) lambda. "a"/"v" take no _BitInt vector, hence the int view.
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

// A handle of element type X at byte `off` of the LDS image `base`, by pointer arithmetic in
// the LDS address space (make_smem on a generic pointer costs every handle a null check).
template <class X, class S>
__device__ inline opus::smem<X> smem_at(const S& base, size_t off)
{
    opus::smem<X> s{nullptr};
    s.ptr = base.ptr + off;
    return s;
}

// ============================================================================================
// Pipelined KV-tile loop for one work item (see the stage map at the top). Q, the O tiles and
// the online-softmax state are owned by the caller.
// ============================================================================================
template <class Traits, class VQ, class OT>
__device__ __attribute__((always_inline)) void
mla_decode_fwd_pipelined(opus_mla_decode_fp8_kargs kargs,
                         int kv_ind_ptr_s,
                         int valid_kv_len,
                         int tile_begin,
                         int tile_end,
                         char* smem_buffer,
                         VQ& v_q,
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

    // The last column each lane's row may see, and the wave's smallest such column (its first
    // row), which decides whether a tile needs the mask at all.
    int mask_bound = valid_kv_len - 1;
    int lane_bound = mask_bound;
    if constexpr(T::CAUSAL)
    {
        const int wave_diag = (warp_id * T::Q_TILE_SIZE) / kargs.H + causal_diagonal;
        const int lane_diag =
            (warp_id * T::Q_TILE_SIZE + lane_id % T::W_N) / kargs.H + causal_diagonal;
        mask_bound = wave_diag < mask_bound ? wave_diag : mask_bound;
        lane_bound = lane_diag < lane_bound ? lane_diag : lane_bound;
    }

    const D_K* kv_base = reinterpret_cast<const D_K*>(kargs.kv_buffer_ptr);
    auto g_kv_indices  = make_gmem(kargs.kv_indices + kv_ind_ptr_s, valid_kv_len * sizeof(int));

    // KV_SLOTS slots of the K image. They rotate every phase as the handles kv_prev / kv_cur /
    // kv_next below; s_kv[] is only ever indexed by a constant.
    const auto lds              = make_smem(smem_buffer);
    smem<D_K> s_kv[T::KV_SLOTS] = {smem_at<D_K>(lds, 0),
                                   smem_at<D_K>(lds, T::smem_kv_slot_bytes),
                                   smem_at<D_K>(lds, 2 * T::smem_kv_slot_bytes)};
    auto s_kv_indices           = smem_at<int>(lds, T::smem_idx_offset);

    constexpr int rope_chunk = T::smem_d_rpt_kv - 1;
    constexpr int rope_g     = rope_chunk * T::smem_row_kv;
    auto u_kv_indices        = make_layout_kv_indices<T>(warp_id, lane_id);
    auto u_gkv               = make_layout_gkv<T, T::kv_threads_d - 1>(lane_id);
    auto u_gkv_r             = make_layout_gkv<T, T::kv_threads_d / 2 - 1>(lane_id);
    auto u_skv               = make_layout_skv<T>(warp_id);
    auto u_rk                = make_layout_rk<T>(lane_id);
    auto u_rv                = make_layout_rv<T>(lane_id);

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
    // zeros -- a number<0> makes the operand poison and DCEs the body, and only 0 selects the
    // bare form without v_mfma_ld_scale_b32. GEMM0 is (K, Q) so C is [token][q]; GEMM1 is
    // (V, P) so C is [d][q]: one query row per lane either way.
    auto mfma0 = make_mfma<D_K, D_Q, D_ACC>(number<T::W_M>{}, number<T::W_N>{}, number<T::W_K>{});
    auto mfma1 = make_mfma<D_K, D_K, D_ACC>(number<T::W_M>{}, number<T::W_N>{}, number<T::W_K>{});

    using k_half_t  = vector_t<D_K, T::VEC_KV>;
    using op_t      = vector_t<D_K, T::W_M * T::W_K / T::WARP_SIZE>;   // 32, one operand
    using s_tile_t  = vector_t<D_ACC, T::W_M * T::W_N / T::WARP_SIZE>; // 16
    using v_piece_t = vector_t<D_K, T::VEC_TR_V>;

    // K ring: K_DEPTH k-steps, each both token tiles x both operand halves. V ring: V_DEPTH
    // d-tiles. Never live at once, so the allocator can fold them onto one set of registers.
    opus::array<k_half_t, T::GEMM0_E_N * 2> v_k[T::K_DEPTH];
    op_t v_v[T::V_DEPTH];
    // Scores for two tiles, so tile t's softmax head runs alongside PV(t-1) and its tail
    // alongside QK(t+1). Indexed only by unrolled compile-time constants.
    using s_flat_t = vector_t<D_ACC, T::GEMM0_E_N * T::O_TILE_REGS>;
    opus::array<s_tile_t, T::GEMM0_E_N> v_s[2];
    auto sflat   = [](auto& a) -> s_flat_t& { return *reinterpret_cast<s_flat_t*>(&a); };
    auto clear_s = [](auto& a, auto sbuf) {
        static_for<T::GEMM0_E_N>([&](auto i) {
            MLA32MX4_PIN_VGPR(S_VGPR_BASE<T> + decltype(sbuf)::value * S_VGPR_PER_BUF<T> +
                              i.value * T::O_TILE_REGS)
            a[i.value] = s_tile_t{0};
        });
    };
    op_t v_p;
    static_assert(check_register_map<T>());

    op_t v_q_steps[T::GEMM0_E_K];
    static_for<T::GEMM0_E_K>([&](auto i) {
        MLA32MX4_PIN_AGPR(i.value * Q_AGPR_PER_STEP<T>)
        v_q_steps[i.value] = reinterpret_cast<const op_t*>(&v_q)[i.value];
    });

    // Register classes: Q, the first O_AGPR_TILES O tiles and the K/V ring in AGPRs; the rest
    // of O, S and P in VGPRs. With O's VGPR half (144) and both score buffers (64) the VGPR
    // file has no room for the ring, and the AGPRs have exactly the 72 Q and O leave over.
    auto pin_q = [&]() {
        static_for<T::GEMM0_E_K>([&](auto i) { anchor_agpr(v_q_steps[i.value]); });
    };
    auto pin_o = [&]() {
        static_for<T::GEMM1_E_M>([&](auto i) {
            if constexpr(i.value < T::O_AGPR_TILES)
                anchor_agpr(v_o_tiles[i.value]);
            else
                anchor_vgpr(v_o_tiles[i.value]);
        });
    };
    auto pin_s = [&](auto& a) { anchor_vgpr(a); };

    // Inside a slot: GEMM0 k-step ks is chunk ks / 2 and 64 d on for odd ks; token tile nt is
    // one DMA pass (NUM_WARPS blocks); the operand's second pass is 32 d on. PV d-tile dt is
    // chunk dt / 4 plus 2 slots per tile; instruction q is one pass per q / 2 and two rows per
    // q % 2.
    auto k_off = [](auto nt, auto ks, auto pp) {
        constexpr int s = decltype(ks)::value;
        return number<(s / 2) * T::smem_kv_chunk +
                      decltype(nt)::value * T::NUM_WARPS * T::smem_kv_block +
                      ((s % 2) * (T::W_K / T::VEC_KV) + decltype(pp)::value * 2) * T::VEC_KV>{};
    };
    auto v_off = [](auto dt, auto q) {
        constexpr int d               = decltype(dt)::value;
        constexpr int qq              = decltype(q)::value;
        constexpr int tiles_per_chunk = T::smem_row_kv / T::W_M; // 4
        return number<(d / tiles_per_chunk) * T::smem_kv_chunk +
                      (qq / 2) * T::NUM_WARPS * T::smem_kv_block + (qq % 2) * 2 * T::smem_row_kv +
                      (d % tiles_per_chunk) * 2 * T::VEC_KV>{};
    };

    // One GEMM0 k-step's K, both token tiles: four ds_read_b128. The lane's offset is summed
    // once, out here, so each read is one base VGPR plus an immediate.
    const int k_lane = layout_to_offsets<T::VEC_KV>(u_rk)[0];
    auto load_k_step = [&](auto& dst, smem<D_K> kv, auto ks) {
        constexpr int base =
            KV_AGPR_BASE<T> + (decltype(ks)::value % T::K_DEPTH) * K_AGPR_PER_STEP<T>;
        static_for<T::GEMM0_E_N>([&](auto nt) {
            static_for<2>([&](auto pp) {
                constexpr int h = nt.value * 2 + pp.value;
                MLA32MX4_PIN_AGPR(base + h * K_AGPR_PER_HALF<T>)
                dst[h] = kv.template _load<T::VEC_KV>(k_lane + decltype(k_off(nt, ks, pp))::value);
            });
        });
    };
    auto load_k_head = [&](smem<D_K> kv) {
        static_for<T::K_DEPTH>([&](auto j) { load_k_step(v_k[j.value], kv, j); });
    };

    // GEMM0: GEMM0_E_K steps of GEMM0_E_N MFMA (alternating accumulators), step ks in ring entry
    // ks % K_DEPTH, refilled with step ks + K_DEPTH behind its MFMA. `per_step(ks)` rides the
    // shadow of step ks: a KV DMA and a chunk of the previous tile's softmax tail. A fence right
    // behind each step's MFMA: left free, the scheduler hoists a refill above the MFMA still
    // reading its entry, and the ring turns into K_DEPTH + 1 live buffers. Nothing else is
    // fenced, so the next step's MFMA may issue ahead of this step's chunk (~2% at c16384 over
    // fencing the chunk in too).
    auto compute_qk = [&](auto& s, smem<D_K> kv, auto sbuf, auto&& per_step) {
        constexpr int sbase = S_VGPR_BASE<T> + decltype(sbuf)::value * S_VGPR_PER_BUF<T>;
        static_for<T::GEMM0_E_K>([&](auto ks) {
            constexpr int buf = ks.value % T::K_DEPTH;
            auto* k_op        = reinterpret_cast<op_t*>(&v_k[buf][0]);
            anchor_agpr(v_k[buf]); // as for V: the load pin alone is dropped
            static_for<T::GEMM0_E_N>([&](auto nt) {
                MLA32MX4_PIN_VGPR(sbase + nt.value * T::O_TILE_REGS)
                s[nt.value] = mfma0(k_op[nt.value], v_q_steps[ks.value], s[nt.value], 0, 0);
            });
            __builtin_amdgcn_sched_barrier(0);
            if constexpr(ks.value + T::K_DEPTH < T::GEMM0_E_K)
                load_k_step(v_k[buf], kv, number<ks.value + T::K_DEPTH>{});
            per_step(ks);
        });
    };

    // V d-tile dt of the tile in slot `kv`: four transpose reads into ring entry dt % V_DEPTH.
    // The builtin, not opus's inline-asm tr_load: V outlives its read by several MFMA, and an
    // asm result is one the compiler believes ready at once.
    using v2i_t      = vector_t<int, 2>;
    const int v_lane = layout_to_offsets<T::VEC_TR_V>(u_rv)[0];
    auto load_v      = [&](smem<D_K> kv, auto dt) {
        constexpr int buf  = decltype(dt)::value % T::V_DEPTH;
        constexpr int base = KV_AGPR_BASE<T> + buf * V_AGPR_PER_TILE<T>;
        auto* piece        = reinterpret_cast<v_piece_t*>(&v_v[buf]);
        auto* lane_base    = kv.ptr + v_lane;
        static_for<T::v_tile_ds_read_insts>([&](auto q) {
            constexpr int off = decltype(v_off(dt, q))::value;
            MLA32MX4_PIN_AGPR(base + q.value * (T::VEC_TR_V / 4))
            piece[q.value] =
                __builtin_bit_cast(v_piece_t,
                                   __builtin_amdgcn_ds_read_tr8_b64_v2i32(
                                       reinterpret_cast<OPUS_LDS_ADDR v2i_t*>(lane_base + off)));
        });
    };
    auto load_v_head = [&](smem<D_K> kv) {
        static_for<T::V_DEPTH>([&](auto dt) { load_v(kv, dt); });
    };
    // PV out of the V ring: one MFMA per d-tile, whose ring entry is refilled with d-tile
    // dt + V_DEPTH right behind it. `co(dt)` rides the shadow of MFMA dt. Fenced behind each MFMA
    // for the same reason as GEMM0.
    auto compute_pv = [&](smem<D_K> kv, auto&& co) {
        static_for<T::GEMM1_E_M>([&](auto dt) {
            constexpr int d = dt.value;
            // The tr8 pin alone is dropped (the MFMA takes either file); a use in the AGPR
            // class right here is what keeps V out of the VGPRs O and S need.
            anchor_agpr(v_v[d % T::V_DEPTH]);
            if constexpr(d < T::O_AGPR_TILES)
            {
                MLA32MX4_PIN_AGPR(O_AGPR_BASE<T> + d * T::O_TILE_REGS)
                v_o_tiles[d] = mfma1(v_v[d % T::V_DEPTH], v_p, v_o_tiles[d], 0, 0);
            }
            else
            {
                MLA32MX4_PIN_VGPR(O_VGPR_BASE<T> + (d - T::O_AGPR_TILES) * T::O_TILE_REGS)
                v_o_tiles[d] = mfma1(v_v[d % T::V_DEPTH], v_p, v_o_tiles[d], 0, 0);
            }
            __builtin_amdgcn_sched_barrier(0);
            if constexpr(d + T::V_DEPTH < T::GEMM1_E_M)
                load_v(kv, number<d + T::V_DEPTH>{});
            co(dt);
        });
    };
    auto no_co = [](auto) {};

    constexpr index_t s_len      = T::GEMM0_E_N * T::O_TILE_REGS; // 32
    constexpr index_t s_half_len = s_len / 2;

    // Online softmax: skip the O rescale entirely while every lane's new row max is within this
    // much of the running one. Decided by a ballot, so the whole wave takes the same branch.
    constexpr D_ACC RESCALE_THRESHOLD = 8.0f;
    D_ACC rescale_m                   = 1.0f;
    D_ACC row_max;
    bool all_below = true;

    // The softmax tail of the previous tile, ending in its P, as one chunk per GEMM0 k-step. The
    // first half (token tile 0) was exponentiated by that tile's head already, so its P quarter
    // and its row-sum partials need not wait for the exps:
    //   0-3  exp2 of the second half, four per chunk; plus
    //   0    first-half partial sums (8 adds)        1  P of token tile 0
    //   2    first-half tree 8 -> 2                  4  second-half partial sums (8 adds)
    //   5    P of token tile 1                       6  second-half tree, combine, l update
    // No chunk tops ~20 VALU, i.e. each fits a GEMM0 step's two-MFMA (~128 cycle) shadow; the
    // row sum is a balanced tree since float adds do not reassociate.
    D_ACC part_a[s_half_len / 2], part_b[s_half_len / 2];
    vector_t<u32_t, 8> p_dw;
    auto tail_chunk = [&](auto& vs_arr, auto ks) {
        constexpr int k = decltype(ks)::value;
        constexpr int q = s_half_len / 2; // 8
        auto& vs        = sflat(vs_arr);
        if constexpr(k < 4)
            attn_exp2_slice<T, s_half_len + k * 4, 4>(vs);
        if constexpr(k == 0)
            static_for<q>([&](auto i) { part_a[i.value] = vs[i.value] + vs[q + i.value]; });
        else if constexpr(k == 1)
        {
            const auto p0 = build_p_tile<T, 0>(vs);
            static_for<4>([&](auto i) { p_dw[i.value] = p0[i.value]; });
        }
        else if constexpr(k == 2)
        {
            static_for<q / 2>([&](auto i) { part_a[i.value] += part_a[q / 2 + i.value]; });
            static_for<q / 4>([&](auto i) { part_a[i.value] += part_a[q / 4 + i.value]; });
        }
        else if constexpr(k == 4)
            static_for<q>([&](auto i) {
                part_b[i.value] = vs[s_half_len + i.value] + vs[s_half_len + q + i.value];
            });
        else if constexpr(k == 5)
        {
            const auto p1 = build_p_tile<T, 1>(vs);
            static_for<4>([&](auto i) { p_dw[4 + i.value] = p1[i.value]; });
            v_p = __builtin_bit_cast(op_t, p_dw);
        }
        else if constexpr(k == 6)
        {
            static_for<__builtin_ctz(q)>([&](auto lvl) {
                constexpr int half = q >> (lvl.value + 1);
                static_for<half>([&](auto i) { part_b[i.value] += part_b[i.value + half]; });
            });
            l_row += row_combine_sum((part_a[0] + part_a[1]) + part_b[0]);
        }
    };
    constexpr int TAIL_CHUNKS = 7;
    static_assert(TAIL_CHUNKS <= T::GEMM0_E_K, "the tail must fit the GEMM0 shadows");
    auto softmax_tail = [&](auto& vs_arr) {
        static_for<TAIL_CHUNKS>([&](auto k) { tail_chunk(vs_arr, k); });
    };

    // Only a wave some of whose rows moved their max pays for the 256 multiplies.
    auto rescale_o = [&]() {
        if(!all_below)
        {
            static_for<T::GEMM1_E_M>([&](auto i) {
                constexpr int d = i.value;
                if constexpr(d < T::O_AGPR_TILES)
                {
                    // Measured: a plain (pinned) multiply here costs 57 VGPR of spill.
                    MLA32MX4_PIN_AGPR(O_AGPR_BASE<T> + d * T::O_TILE_REGS)
                    v_o_tiles[d] = scale_tile_agpr(v_o_tiles[d], rescale_m);
                }
                else
                {
                    MLA32MX4_PIN_VGPR(O_VGPR_BASE<T> + (d - T::O_AGPR_TILES) * T::O_TILE_REGS)
                    v_o_tiles[d] = v_o_tiles[d] * rescale_m;
                }
            });
            asm volatile("s_nop 4" ::);
        }
    };

    // Page indices go through LDS: one buffer_load_dword lds per wave per tile puts the tile's
    // 64 indices in the wave's own ring entry, three tiles ahead, and the two a lane needs come
    // back as plain ds_reads once the phase's vmcnt has covered the DMA. As an ordinary buffer
    // load its wait would be the compiler's, which does not count the asm LDS-DMA and so would
    // drain the KV DMA along with it. Every lane's token is clamped into the request, so no
    // index DMA is ever out of range and a tile past the end repeats the last token's page.
    const auto idx_tok = layout_to_offsets<1>(u_kv_indices); // token of each pass, in tile
    auto idx_slot      = [&](int tile_idx) {                 // byte offset of the ring entry
        return (((tile_idx - tile_begin) & (T::IDX_RING - 1)) * T::NUM_WARPS + warp_id) *
               static_cast<int>(T::smem_idx_tile);
    };
    vector_t<int, 4> idx_rsrc;
    __builtin_memcpy(&idx_rsrc, &g_kv_indices.cached_rsrc, sizeof(idx_rsrc));
    auto issue_idx_dma = [&](int tile_idx) {
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
    // The 32-bit form wraps at 4 GiB, exactly the descriptor's own reach, so it stays exact
    // wherever the small path is legal -- but it has to wrap in unsigned arithmetic.
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

    // KV gmem->LDS, one instruction at a time: DMA i of a tile is pass i / smem_d_rpt_kv,
    // chunk i % smem_d_rpt_kv (the last chunk being rope + pad). One call is one
    // buffer_load_lds (m0 rewritten, s_nop, load), so the phase can spread them over GEMM0's
    // MFMA. A tile past the range still gets its DMA, so every phase issues the same count and
    // the vmcnt budget is a constant.
    [[maybe_unused]] vector_t<int, 4> kv_rsrc;
    [[maybe_unused]] u32_t gkv_lane = 0, gkv_r_lane = 0;
    if constexpr(!T::LARGE_KV)
    {
        __builtin_memcpy(&kv_rsrc, &g_kv.cached_rsrc, sizeof(kv_rsrc));
        gkv_lane   = static_cast<u32_t>(layout_to_offsets<T::VEC_KV>(u_gkv)[0]);
        gkv_r_lane = static_cast<u32_t>(layout_to_offsets<T::VEC_KV>(u_gkv_r)[0]);
    }
    const u32_t skv_wave = static_cast<u32_t>(layout_to_offsets<T::VEC_KV>(u_skv)[0]);
    auto issue_dma       = [&](smem<D_K> kv, const auto& po, auto i) {
        constexpr int p     = decltype(i)::value / T::smem_d_rpt_kv;
        constexpr int c     = decltype(i)::value % T::smem_d_rpt_kv;
        constexpr int s_off = c * T::smem_kv_chunk + p * T::NUM_WARPS * T::smem_kv_block;
        const auto poff     = po[p];
        if constexpr(T::LARGE_KV)
        {
            if constexpr(c < rope_chunk)
                global_load<T::VEC_KV>(
                    g_kv + (poff + c * T::smem_row_kv), kv.ptr, u_gkv, u_skv + s_off);
            else
                global_load<T::VEC_KV>(g_kv_r + poff, kv.ptr, u_gkv_r, u_skv + s_off);
        }
        else
        {
            const u32_t voff = (c < rope_chunk)
                                         ? gkv_lane + static_cast<u32_t>(poff + c * T::smem_row_kv)
                                         : gkv_r_lane + static_cast<u32_t>(poff + rope_g);
            buffer_load_lds_asm<T::VEC_KV>(
                kv_rsrc, voff, lds_addr(kv) + skv_wave + static_cast<u32_t>(s_off));
        }
    };
    auto async_load_kv = [&](smem<D_K> kv, const auto& pages) {
        const auto po = dma_offsets(pages);
        static_for<T::kv_buffer_load_insts>([&](auto i) { issue_dma(kv, po, i); });
    };

    const u32_t neg_inf_v = std::bit_cast<u32_t>(-numeric_limits<D_ACC>::infinity());
    // Only the tiles that can contain invalid columns pay for a mask: those whose last column
    // lies past the wave's smallest bound. mask_bound is wave-uniform, so the test is scalar.
    auto mask_oob_scores = [&](auto& s, int tile_idx) {
        if((tile_idx + 1) * T::KV_TILE_SIZE - 1 > mask_bound)
            attn_mask_kv_tile<T>(sflat(s), lane_bound, tile_idx, neg_inf_v);
    };

    auto stage_end = [&]() { __builtin_amdgcn_sched_barrier(0); };
    // A workgroup barrier over LDS. The asm memory clobbers keep the compiler from moving an
    // LDS access across it; a real fence would too, but at workgroup scope it also drains
    // vmcnt, i.e. the KV DMA that is supposed to stay in flight. s_barrier does not wait for a
    // wave's own LDS traffic, so every caller has an lgkmcnt(0) in front of it.
    auto lds_barrier = [&]() {
        asm volatile("" ::: "memory");
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("" ::: "memory");
    };

    // --- Prologue ---
    // The previous work item may still have V reads of its slots and its last (unused) DMA in
    // flight; nothing below may touch a slot before every wave is past both.
    __builtin_amdgcn_s_waitcnt(0);
    lds_barrier();
    pin_q();
    pin_o();

    // SPLIT_DMA: a slot is released a chunk at a time rather than whole. PV(t-1) walks V in d
    // order, so chunks 0 and 1 (d-tiles 0..7) are read out halfway through it, and chunk 4
    // (rope + pad) is GEMM0's only and free once QK(t-1) is done. A second barrier halfway
    // through PV(t-1) therefore lets the EARLY part of tile t+2 -- chunks 0, 1, 4 of both
    // passes, 60% of its bytes -- go into slot t-1 a phase ahead; only the LATE part (chunks
    // 2, 3) waits for the next phase's first barrier. With one-phase-ahead DMA the kernel sat
    // ~26% of its time on vmcnt; this keeps more of each tile in flight.
    // Not under LARGE_KV: its 64-bit per-lane DMA addresses do not fit next to PV's live set
    // (measured: 100 VGPR of spill), so that build keeps the whole tile in one issue.
    constexpr bool SPLIT_DMA = T::SPLIT_DMA && !T::LARGE_KV;
    constexpr int EARLY_DMAS = SPLIT_DMA ? 3 * T::KV_PASSES : 0;     // 6
    constexpr int LATE_DMAS  = T::kv_buffer_load_insts - EARLY_DMAS; // 4 (10 unsplit)
    // DMA index i = pass * smem_d_rpt_kv + chunk (see issue_dma).
    auto early_dma_idx = [](auto j) {
        constexpr int jj = decltype(j)::value, c = jj % 3;
        return number<(jj / 3) * T::smem_d_rpt_kv + (c < 2 ? c : T::smem_d_rpt_kv - 1)>{};
    };
    auto late_dma_idx = [](auto j) {
        constexpr int jj = decltype(j)::value;
        if constexpr(SPLIT_DMA)
            return number<(jj / 2) * T::smem_d_rpt_kv + 2 + jj % 2>{};
        else
            return number<jj>{};
    };
    // PV step after which the mid barrier sits: every V read of chunks 0 and 1 (d-tiles 0..7)
    // has been issued V_DEPTH steps before it.
    constexpr int MID_STEP = 2 * (T::smem_row_kv / T::W_M) - 1; // 7

    // Tiles tile_begin and tile_begin+1 whole into slots 0 and 1, then the index DMA of
    // tile_begin+3 and the EARLY part of tile_begin+2 into slot 2 -- what a phase for
    // tile_begin would have issued -- so the first loop phase's vmcnt budget holds.
    issue_idx_dma(tile_begin);
    issue_idx_dma(tile_begin + 1);
    issue_idx_dma(tile_begin + 2);
    __builtin_amdgcn_s_waitcnt(0);
    asm volatile("" ::: "memory");
    const auto pages_a = load_kv_pages(tile_begin);
    const auto pages_b = load_kv_pages(tile_begin + 1);
    async_load_kv(s_kv[0], pages_a);
    async_load_kv(s_kv[1], pages_b);
    issue_idx_dma(tile_begin + 3);
    if constexpr(SPLIT_DMA)
    {
        const auto po_c = dma_offsets(load_kv_pages(tile_begin + 2));
        static_for<EARLY_DMAS>([&](auto j) { issue_dma(s_kv[2], po_c, early_dma_idx(j)); });
    }
    s_waitcnt_vmcnt(number<T::kv_buffer_load_insts + T::kv_index_load_insts + EARLY_DMAS>{});
    lds_barrier();

    // Partial phase for tile_begin: QK and the softmax head. There is no previous tile to run a
    // tail or a PV for, and since O and l start at zero the head needs no rescale either.
    load_k_head(s_kv[0]);
    stage_end();
    clear_s(v_s[0], 0_I);
    compute_qk(v_s[0], s_kv[0], 0_I, [](auto) {});
    mask_oob_scores(v_s[0], tile_begin);
    m_row = max(m_row, temperature_scale * attn_row_max<T>(sflat(v_s[0])));
    static_for<s_len>([&](auto i) {
        sflat(v_s[0])[i.value] = __builtin_fmaf(sflat(v_s[0])[i.value], temperature_scale, -m_row);
    });
    attn_exp2_slice<T, 0, s_half_len>(sflat(v_s[0]));
    pin_s(v_s[0]);
    stage_end();

    static_assert(T::K_DEPTH == 2 && T::V_DEPTH % T::K_DEPTH == 0,
                  "the early V head assumes the last two GEMM0 steps free the two K entries");

    // kv_prev: slot t-1 (PV this phase), kv_cur: slot t (QK), kv_next: slot t+1 (DMA'd now).
    auto run_phase = [&](auto& vs_cur_arr,
                         auto& vs_prev_arr,
                         smem<D_K> kv_prev,
                         smem<D_K> kv_cur,
                         smem<D_K> kv_next,
                         int t,
                         auto sbuf) {
        auto& vs_cur = sflat(vs_cur_arr);
        // stage0 [mem]: wait out DMA(t) -- its LATE part is the newest of it -- leaving the
        // EARLY part of t+2 (unsplit: the index DMA of t+2) in flight; the barrier then publishes
        // K(t) and tells every wave that PV(t-2) is done with the slot the LATE part of t+1 is
        // about to overwrite.
        s_waitcnt_vmcnt(number < SPLIT_DMA ? EARLY_DMAS : T::kv_index_load_insts > {});
        lds_barrier();
        pin_q();
        const auto po = dma_offsets(load_kv_pages(t + 1));
        // The EARLY part's offsets are fetched only at the mid barrier (see pv_chunk): held from
        // here they are live across the whole phase, which the LARGE_KV build (64-bit
        // offsets) measured as 102 VGPR of spill.
        [[maybe_unused]] opus::remove_cvref_t<decltype(po)> po2{};
        load_k_head(kv_cur);
        stage_end();

        // stage1 [compute]: gemm0 QK(t) [18 MFMA, one DMA of t+1 per k-step] || softmax
        // tail(t-1) -> P(t-1) || V head of t-1 (last two k-steps).
        __builtin_amdgcn_s_setprio(1);
        clear_s(vs_cur_arr, sbuf);
        compute_qk(vs_cur_arr, kv_cur, sbuf, [&](auto ks) {
            // The index DMA of t+3 first (split: it feeds next phase's EARLY part and must beat
            // this phase's LATE DMA home), then the LATE part of t+1, one per k-step.
            if constexpr(ks.value == 0 && SPLIT_DMA)
                issue_idx_dma(t + 3);
            if constexpr(ks.value < LATE_DMAS && ks.value < T::GEMM0_E_K - 1)
                issue_dma(kv_next, po, late_dma_idx(ks));
            if constexpr(ks.value == T::GEMM0_E_K - 1)
            {
                static_for<(LATE_DMAS >= T::GEMM0_E_K ? LATE_DMAS - T::GEMM0_E_K + 1 : 0)>(
                    [&](auto j) {
                        issue_dma(kv_next, po, late_dma_idx(number<T::GEMM0_E_K - 1 + j.value>{}));
                    });
                if constexpr(!SPLIT_DMA)
                    issue_idx_dma(t + 3);
            }
            if constexpr(ks.value < TAIL_CHUNKS)
                tail_chunk(vs_prev_arr, ks);
            // The V head of t-1 rides the last two GEMM0 steps, each half into the K ring entry
            // that step has just finished with (ring entry dt of V overlays K entry dt / 2).
            // Issued after GEMM0 instead, the first PV MFMA sat ~240 cycles on its V.
            if constexpr(ks.value >= T::GEMM0_E_K - 2)
            {
                constexpr int kbuf = ks.value % T::K_DEPTH;
                static_for<T::V_DEPTH / T::K_DEPTH>([&](auto j) {
                    load_v(kv_prev, number<kbuf*(T::V_DEPTH / T::K_DEPTH) + j.value>{});
                });
            }
        });
        stage_end();

        // stage2 [mem]: mask S(t) before stage3 folds it into the softmax.
        mask_oob_scores(vs_cur_arr, t);
        stage_end();

        // stage3 [compute]: gemm1 PV(t-1) [16 MFMA] with tile t's softmax head chopped into
        // per-d-tile chunks. The head reads only vs_cur, so nothing in it depends on the PV
        // chain. Chunk k after PV d-tile k:
        //   0,1  local row max, permlane32     2  rescale decision, m / l update
        //   3-6  fused scale-subtract          4-11  exp2 of the first half, two per chunk
        D_ACC rmx;
        auto head_chunk = [&](auto dt) {
            constexpr int k = decltype(dt)::value;
            if constexpr(k == 0)
            {
                rmx = vs_cur[0];
                static_for<s_half_len - 1>([&](auto i) { rmx = max(rmx, vs_cur[i.value + 1]); });
            }
            else if constexpr(k == 1)
            {
                static_for<s_len - s_half_len>(
                    [&](auto i) { rmx = max(rmx, vs_cur[s_half_len + i.value]); });
                rmx = row_combine_max(rmx);
            }
            else if constexpr(k == 2)
            {
                row_max   = temperature_scale * rmx;
                all_below = __builtin_amdgcn_ballot_w64((row_max - m_row) <= RESCALE_THRESHOLD) ==
                            __builtin_amdgcn_read_exec();
                row_max   = all_below ? m_row : max(m_row, row_max);
                rescale_m = __builtin_amdgcn_exp2f(m_row - row_max);
                l_row *= rescale_m;
                m_row = row_max;
            }
            if constexpr(k >= 3 && k < 3 + s_len / 8)
            {
                static_for<8>([&](auto i) {
                    constexpr int e = (k - 3) * 8 + i.value;
                    vs_cur[e]       = __builtin_fmaf(vs_cur[e], temperature_scale, -row_max);
                });
            }
            if constexpr(k >= 4 && k < 4 + s_half_len / 2)
            {
                attn_exp2_slice<T, (k - 4) * 2, 2>(vs_cur);
            }
            // Redefine what this chunk wrote in place. Nothing reads it before the next phase,
            // so without a use here the whole subtract + exp sinks past the rescale branch
            // and runs as a serial block after the last PV MFMA; sched_barrier cannot stop
            // that, it only fences the scheduler.
            using g8_t = vector_t<D_ACC, 8>;
            auto* grp  = reinterpret_cast<g8_t*>(&vs_cur);
            if constexpr(k >= 3 && k < 3 + s_len / 8)
                anchor_vgpr(grp[k - 3]);
            if constexpr(k >= 4 && k < 4 + s_half_len / 2 && (k - 4) / 4 != k - 3)
                anchor_vgpr(grp[(k - 4) / 4]);
        };
        static_assert(4 + s_half_len / 2 <= T::GEMM1_E_M, "the head must fit the PV shadows");
        // SPLIT_DMA's mid barrier: once this wave's V reads of chunks 0 and 1 are home (they
        // are older than the V_DEPTH d-tiles still in flight, hence the non-zero count), the
        // barrier tells every wave that slot t-1's chunks 0, 1, 4 are free, and the EARLY part
        // of t+2 goes in, one DMA per remaining PV step.
        auto pv_chunk = [&](auto dt) {
            head_chunk(dt);
            if constexpr(SPLIT_DMA)
            {
                constexpr int d = decltype(dt)::value;
                static_assert(MID_STEP + EARLY_DMAS < T::GEMM1_E_M);
                if constexpr(d == MID_STEP)
                {
                    // The page reads come after the V reads, so the count below covers them
                    // too; they are this wave's own ring entry, landed by stage0's vmcnt.
                    po2 = dma_offsets(load_kv_pages(t + 2));
                    s_waitcnt_lgkmcnt(number<(T::V_DEPTH - 1) * T::v_tile_ds_read_insts>{});
                    lds_barrier();
                }
                if constexpr(d > MID_STEP && d - MID_STEP - 1 < EARLY_DMAS)
                    issue_dma(kv_prev, po2, early_dma_idx(number<d - MID_STEP - 1>{}));
            }
        };
        __builtin_amdgcn_s_setprio(1);
        pin_o();
        compute_pv(kv_prev, pv_chunk);
        // O now holds PV(t-1) at the old max; move it to the new one before PV(t) lands.
        rescale_o();
        // What lets the next barrier release slot t-1: without it a wave could pass it with V
        // reads of that slot in flight while another wave's DMA overwrites it.
        s_waitcnt_lgkmcnt(0_I);
        pin_o();
        pin_s(vs_cur_arr);
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
    // loop always starts and ends there and the epilogue needs no parity. Peeled behind the
    // loop instead, as the obvious shape is, the unpaired phase measured 64 VGPR of spill: the
    // loop-skip edge, the remainder and the epilogue merge with both score buffers live.
    int t = tile_begin + 1;
    if(((tile_end - 1 - tile_begin) & 1) == 1)
    {
        __builtin_amdgcn_sched_barrier(0);
        run_phase(v_s[1], v_s[0], kv_prev, kv_cur, kv_next, t, 1_I);
        rotate();
        static_for<T::GEMM0_E_N>([&](auto i) {
            MLA32MX4_PIN_VGPR(S_VGPR_BASE<T> + i.value * T::O_TILE_REGS)
            v_s[0][i.value] = v_s[1][i.value];
        });
        ++t;
        __builtin_amdgcn_sched_barrier(0);
    }
    for(; t < tile_end; t += 2)
    {
        __builtin_amdgcn_sched_barrier(0);
        run_phase(v_s[1], v_s[0], kv_prev, kv_cur, kv_next, t, 1_I);
        rotate();
        __builtin_amdgcn_sched_barrier(0);
        run_phase(v_s[0], v_s[1], kv_prev, kv_cur, kv_next, t + 1, 0_I);
        rotate();
        __builtin_amdgcn_sched_barrier(0);
    }
    __builtin_amdgcn_sched_barrier(0);

    // --- Epilogue: softmax tail + gemm1 of the last tile, which sits in kv_prev and v_s[0].
    softmax_tail(v_s[0]);
    stage_end();
    load_v_head(kv_prev);
    stage_end();
    pin_o();
    compute_pv(kv_prev, no_co);
    pin_o();
    __builtin_amdgcn_sched_barrier(0);
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
    using D_Q   = typename T::D_Q;
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

    // Per-tensor descale folded into the two places it can be a single scalar multiply: QK's
    // descale_q * descale_k rides the softmax temperature, and V's descale_k is applied once on
    // the finished O (the PV MFMA itself consumes raw fp8).
    const float descale_q = readfirstlane_f32(reinterpret_cast<const float*>(kargs.q_scale_ptr)[0]);
    const float descale_k =
        readfirstlane_f32(reinterpret_cast<const float*>(kargs.kv_scale_ptr)[0]);
    const float qk_scale = readfirstlane_f32(temperature_scale * descale_q * descale_k);

    const int q_gmem_offset = q_len_ptr_s * kargs.stride_q_b;
    auto g_q = make_gmem(reinterpret_cast<const D_Q*>(kargs.q_buffer_ptr) + q_gmem_offset,
                         q_len * kargs.stride_q_b * sizeof(D_Q));
    auto v_q = load<T::VEC_Q>(g_q, make_layout_q<T>(warp_id, lane_id));

    using o_tile_t = vector_t<D_ACC, T::O_TILE_REGS>;
    o_tile_t v_o_tiles[T::GEMM1_E_M];
    static_for<T::GEMM1_E_M>([&](auto i) {
        constexpr int d = i.value;
        if constexpr(d < T::O_AGPR_TILES)
        {
            MLA32MX4_PIN_AGPR(O_AGPR_BASE<T> + d * T::O_TILE_REGS)
            v_o_tiles[d] = o_tile_t{0};
        }
        else
        {
            MLA32MX4_PIN_VGPR(O_VGPR_BASE<T> + (d - T::O_AGPR_TILES) * T::O_TILE_REGS)
            v_o_tiles[d] = o_tile_t{0};
        }
    });
    D_ACC m_row = opus::numeric_limits<D_ACC>::lowest();
    D_ACC l_row = 0.0f;
    mla_decode_fwd_pipelined<Traits>(kargs,
                                     kv_ind_ptr_s,
                                     valid_kv_len,
                                     0,
                                     num_kv_tiles,
                                     smem_buffer,
                                     v_q,
                                     v_o_tiles,
                                     m_row,
                                     l_row,
                                     qk_scale,
                                     causal_diagonal);

    // Softmax normalisation and the V descale in one multiply. l == 0 means every score was
    // masked, so O must be 0 rather than NaN. O goes out a d-tile at a time: the whole of it
    // is 256 registers, more than one file holds.
    const D_ACC o_scale        = (l_row > D_ACC(0.0f)) ? (descale_k / l_row) : D_ACC(0.0f);
    constexpr float INV_LOG2_E = 0.69314718055994531f; // ln(2)
    const D_ACC lse            = (l_row > D_ACC(0.0f)) ? ((m_row + log2f(l_row)) * INV_LOG2_E)
                                                       : opus::numeric_limits<D_ACC>::lowest();
    // The final-vs-partial choice is taken per d-tile rather than once around two whole stores:
    // split into two arms, all 256 O registers stay live across both and the allocator spills.
    const bool is_final     = slot < 0;
    const int o_gmem_offset = q_len_ptr_s * kargs.stride_o_b;
    const int oa_offset     = slot * kargs.stride_o_b;
    auto g_o                = make_gmem(reinterpret_cast<D_OUT*>(kargs.out_ptr) + o_gmem_offset,
                         q_len * kargs.stride_o_b * sizeof(D_OUT));
    auto g_oa               = make_gmem(reinterpret_cast<D_ACC*>(kargs.o_accum) + oa_offset,
                          q_len * kargs.stride_o_b * sizeof(D_ACC));
    auto u_o                = make_layout_o<T>(warp_id, lane_id, kargs.stride_o_h);
    auto u_oa               = make_layout_o<T>(warp_id, lane_id, T::D_NOPE_SIZE);
    static_for<T::GEMM1_E_M>([&](auto dt) {
        const o_tile_t v = v_o_tiles[dt.value] * o_scale;
        if(is_final)
            store<T::VEC_O>(g_o, cast<D_OUT>(v), u_o + dt.value * T::W_M);
        else
            store<T::VEC_O>(g_oa, v, u_oa + dt.value * T::W_M);
    });
    // lse_ptr is null when the caller did not ask for LSE; lse_accum in the split-KV case is
    // always allocated, so only the final side needs the guard.
    D_ACC* lse_base = is_final ? reinterpret_cast<D_ACC*>(kargs.lse_ptr) + q_len_ptr_s * kargs.H
                               : reinterpret_cast<D_ACC*>(kargs.lse_accum) + slot * kargs.H;
    if((!is_final || kargs.lse_ptr != nullptr) && lane_id < T::W_N)
    {
        auto g_lse = make_gmem(lse_base, q_len * kargs.H * sizeof(D_ACC));
        g_lse.store(lse, warp_id * T::Q_TILE_SIZE + lane_id);
    }
}

// Persistent body: the grid is sized to the machine, not to the problem, and each block drains
// the work items the metadata kernel assigned it through work_indptr. A __device__ function so
// the prebuilt code object can wrap it in the extern "C" entry aiter looks up by name (a
// template __global__ only puts a mangled symbol in the object).
//
// One block per CU: 3 KV slots are 124 KB of LDS. That is also what makes the register budget
// work -- one wave per SIMD gets the whole 512-register file (Q 72 + O 256 + S 64 + KV ring).
template <class Traits>
__device__ __attribute__((always_inline)) void
mla_decode_fwd_persistent(opus_mla_decode_fp8_kargs kargs)
{
    using namespace opus;
    using T = opus::remove_cvref_t<Traits>;

    const int work_id = block_id_x();

    // 128 B alignment so every block base is a whole number of LDS lines; the layouts assume
    // the region's own base contributes nothing to the bank pattern.
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

} // namespace mla_decode_fwd_32mx4_64nx1_fp8fp8

// The register map above only holds with the amdgpu-pin-op-dst toolchain, which is how the
// prebuilt code object is built (see mla_decode_fp8_32mx4_64nx1_co.hip). A stock ROCm 7.1 clang
// selects every MFMA in AGPR form at 512 registers per lane, and O (256) plus the two score
// buffers (64) do not fit 256 AGPRs: that build spills heavily and is kept for reference only.
template <class Traits>
__global__
__launch_bounds__(Traits::BLOCK_SIZE,
                  1) void opus_mla_decode_fp8_32mx4_64nx1_kernel(opus_mla_decode_fp8_kargs kargs)
{
    mla_decode_fwd_32mx4_64nx1_fp8fp8::mla_decode_fwd_persistent<Traits>(kargs);
}

#endif // !__HIP_DEVICE_COMPILE__ || !__gfx950__
