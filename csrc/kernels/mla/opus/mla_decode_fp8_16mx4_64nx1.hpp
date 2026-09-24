#pragma once

// MLA decode forward on gfx950: fp8 Q x fp8 KV, 16mx4 / 64nx1, persistent scheduling.
//   GEMM0 (Q*K^T): d = 576 padded to 640     GEMM1 (P*V): d_v = 512
//
// HIP port of the SP3 kernel MLA_A8W8_QH16_1TG_4W_16mx4_64nx1_PS. The LDS mappings follow
// design_16mx4_64nx1.xlsx (K_LOAD, V_LOAD, V_P); /home/memin/design/model_16mx4.py is the
// closed-form model of all of them, checked against itself and the sheets.
//
//   (1) TWO WAVE DECOMPOSITIONS. GEMM0: wave w owns query rows 16w .. 16w+15 against the
//       whole 64-token tile (16x16x128 f8f6f4, "16m x 4"), so the online softmax is
//       wave-private. GEMM1: wave w owns rows 32*(w/2) .. +31 and output d 256*(w%2) .. +255
//       on the full-rate 32x32x64 f8f6f4 -- 8 MFMA x 64 cycles per tile, against the 64 x 16
//       of the half-rate 16x16x32 fp8 the 16-row decomposition would need. A 32-row operand
//       spans two GEMM0 waves, so P goes through LDS, and with it each row's rescale factor:
//       the wave holding an O row is not always the one holding that row's running max.
//
//   (2) KV (K_LOAD). A slot is 5 chunks of 128 d; a chunk is 8 blocks of 8 token rows of
//       128 B plus 32 B of padding, each block one wave's 1 KB buffer_load_lds. Token t sits in
//       block 4 * (t / 32) + t % 4, row (t % 32) / 4, its 16 B d-groups in order along the
//       row. Per wave per tile: 5 chunks x 2 token passes = 10 DMA.
//       The d = 640 pad half of chunk 4 is filled from the same token's rope (lanes that would
//       fetch d 576..639 fetch d 512..575 again -- same cache lines, no extra HBM traffic;
//       d 576..639 would be the NEXT token's row, i.e. another line) and cleared on the K read.
//
//   (3) V (V_LOAD) is the fp8 nope of K, transpose-read straight out of the K image with
//       ds_read_b64_tr_b8 in the 32x32x64 operand order (lane l: d = l % 32, token
//       32 r + 16 (l / 32) + i). descale_k is applied once, on the finished O.
//
//   (4) P (V_P) is the 16x16 C output cast to fp8, stored row-major [64 q][64 tokens] with the
//       16 B groups XOR-swizzled by (q / 4) % 4, and read back as the 32x32x64 B operand.
//
// Software pipeline, one phase per tile, KV_SLOTS = 3, two barriers per phase, distance-2
// prefetch (SP3's scheme: V(t) is in registers by the end of phase t, so its slot is free for
// t+3 right after the next phase's first barrier). Phase t:
//   stage0 [mem]     vmcnt(11) (DMA(t) and t+2's index DMA landed; DMA(t+1) stays in
//                    flight) + B1 (publishes K(t); every wave is done with slot t-1); ds_read
//                    t+2's page indices and K(t)
//   stage1 [compute] gemm0 QK(t) [20 MFMA, one DMA of t+2 into t-1's slot per MFMA pair]
//                    || softmax tail(t-1) + ds_write P(t-1), scale(t-1); index DMA of t+4
//   stage2 [mem]     mask S(t); B2 (publishes P(t-1)); read P(t-1), scale
//   stage3 [compute] rescale O if any row asks; gemm1 PV(t-1) [8 MFMA] out of the V
//                    registers, refilling each d-tile with V(t) from slot t right behind its
//                    MFMA || softmax head(t), chopped into per-d-tile chunks
//

#include "mla_decode_traits.h"

#if !defined(__HIP_DEVICE_COMPILE__) || !defined(__gfx950__)

template <class Traits>
__global__ void opus_mla_decode_fp8_16mx4_64nx1_kernel(opus_mla_decode_fp8_kargs)
{
}

#else

#include "mla_global_load.hpp"
#include <bit>
#include <cstdint>
#include <opus/opus.hpp>
#include <type_traits>

using opus::operator""_I;

namespace mla_decode_fwd_16mx4_64nx1_fp8fp8 {

// Register pins. Only the amdgpu-pin-op-dst toolchain knows the attributes, and only there do
// the macros expand to them (stock clang would warn and drop them anyway), so aiter's in-tree
// build and OpFoundry's .co compile one source. A pin binds a *definition*, hence the array
// types for K and S below: a store through a reinterpret_cast of a flat vector is not a
// definition the allocator honours.
#if defined(__has_cpp_attribute)
#if __has_cpp_attribute(clang::amdgpu_pin_agpr) && __has_cpp_attribute(clang::amdgpu_pin_vgpr)
#define MLA16MX4_PIN_AGPR(n) [[clang::amdgpu_pin_agpr(n)]]
#define MLA16MX4_PIN_VGPR(n) [[clang::amdgpu_pin_vgpr(n)]]
#define MLA16MX4_HAS_PIN 1
#endif
#endif
#ifndef MLA16MX4_PIN_AGPR
#define MLA16MX4_PIN_AGPR(n)
#define MLA16MX4_PIN_VGPR(n)
#endif
#ifndef MLA16MX4_DMA_ASYNC_LOAD
#define MLA16MX4_DMA_ASYNC_LOAD 0
#endif
// GEMM0 reads K K_DEPTH steps ahead, one register buffer per step in flight. Four (the 32
// spare AGPRs) measured no faster than two at b64 c16384 / c1024, JIT and .co alike.
template <class T>
constexpr int K_DEPTH = 2;
template <class T>
constexpr int K_STEPS = T::EN_GROUPS * T::GEMM0_E_K; // 10
// K, the GEMM0 A operand, in AGPRs a224..a255: K_DEPTH buffers x EN_GROUP e_n tiles x 2
// operand halves x 4 registers. S, the GEMM0 accumulators, in VGPRs v224..v255: 2 buffers x
// GEMM0_E_N tiles x 4 registers -- the softmax works on S with VALU, so an AGPR-form
// accumulator costs an accvgpr_read of every element each phase.
template <class T>
constexpr int K_AGPR_PER_HALF = T::VEC_KV * sizeof(typename T::D_K) / 4; // 4
template <class T>
constexpr int K_HALVES = T::EN_GROUP * (T::W_K / T::W_K_HALF); // 4 per buffer
template <class T>
constexpr int K_AGPR_BASE = 256 - K_DEPTH<T> * K_HALVES<T> * K_AGPR_PER_HALF<T>; // 224

// ds_reads of GEMM0 step j: two operand halves per e_n tile, one on the last k-step (its
// second half is the 640-padding, cleared in registers).
template <class T>
constexpr int k_step_reads(int j)
{
    return (j % T::GEMM0_E_K == T::GEMM0_E_K - 1) ? T::EN_GROUP : 2 * T::EN_GROUP;
}
// LDS reads issued after K(k)'s by the end of step s (s < k): what an lgkmcnt may leave in
// flight and still guarantee K(k). K(0 .. K_DEPTH-1) go out up front; then step i issues X
// reads of its own (per_step) when i < XN, and prefetches K(i + K_DEPTH). Capped at the
// counter's 4 bits -- a smaller count only waits for more.
template <class T, int X, int XN>
constexpr int k_reads_after(int k, int s)
{
    constexpr int D = K_DEPTH<T>;
    int n = 0, first = k - D + 1;
    if(k < D)
    {
        for(int j = k + 1; j < D; ++j)
            n += k_step_reads<T>(j);
        first = 0;
    }
    for(int i = first; i <= s; ++i)
        n += (i < XN ? X : 0) + (i + D < K_STEPS<T> ? k_step_reads<T>(i + D) : 0);
    return n < 15 ? n : 15;
}
template <class T>
constexpr int S_VGPR_PER_TILE = T::W_M * T::W_N / T::WARP_SIZE; // 4
template <class T>
constexpr int S_VGPR_BASE = 256 - 2 * T::GEMM0_E_N * S_VGPR_PER_TILE<T>; // 224
// Q, the GEMM0 B operand, in VGPRs v184..v223 below S. Left free the allocator puts it in
// AGPRs, where with O (128) and V (64) it squeezes K out and O ends up rotating through the
// K registers every PV.
template <class T>
constexpr int Q_VGPR_PER_STEP = T::W_M * T::W_K * sizeof(typename T::D_Q) / T::WARP_SIZE / 4; // 8
template <class T>
constexpr int Q_VGPR_BASE = S_VGPR_BASE<T> - T::GEMM0_E_K * Q_VGPR_PER_STEP<T>; // 184
// O, the PV accumulators, in AGPRs a0..a127 a d-tile after another. Left to the allocator the
// PV MFMA writes a tile somewhere new and it is moved back, 16 accvgpr_mov a tile.
template <class T>
constexpr int O_AGPR_PER_TILE = T::W_M_PV * T::W_N_PV / T::WARP_SIZE; // 16

// Moves a wave-uniform float into an SGPR. The bit_cast is required, not cosmetic:
// __builtin_amdgcn_readfirstlane takes an int, so handing it a float converts the value
// instead of moving it.
__device__ inline float readfirstlane_f32(float v)
{
    return std::bit_cast<float>(__builtin_amdgcn_readfirstlane(std::bit_cast<int>(v)));
}

namespace sched_masks {
constexpr int MFMA      = 0x08;
constexpr int VALU      = 0x02;
constexpr int DS_READ   = 0x100;
constexpr int DS_WRITE  = 0x200;
constexpr int VMEM_READ = 0x020;
constexpr int EXP       = 0x400;
} // namespace sched_masks

// A scheduler preference for GEMM0, not a fence: the K reads go through load<>, i.e. real
// ds_read that SIInsertWaitcnts re-waits after any reorder, so the solver must stay free to
// decline. It has to be, in fact -- a spill here does not merely cost time, it corrupts
// results, because scratch VMEM traffic breaks the hand-written vmcnt budget guarding the
// KV DMA. The VMEM groups spread the next-but-one tile's DMA (and the page-index fetch
// behind it) one per MFMA pair, so their issue stalls sit in MFMA shadows.
template <int Rpt, int G, int NVmem>
__device__ inline void sched_compute_qk()
{
    using namespace sched_masks;
    opus::static_for<Rpt>([&](auto i) {
        __builtin_amdgcn_sched_group_barrier(MFMA, 1, G);
        if constexpr(i.value < Rpt / 2)
        {
            __builtin_amdgcn_sched_group_barrier(DS_READ, 2, G);
        }
        if constexpr(i.value % 2 == 1 && i.value / 2 < NVmem)
        {
            __builtin_amdgcn_sched_group_barrier(VMEM_READ, 1, G);
        }
        if constexpr(i.value == Rpt - 1 && Rpt / 2 < NVmem)
        {
            __builtin_amdgcn_sched_group_barrier(VMEM_READ, NVmem - Rpt / 2, G);
        }
        __builtin_amdgcn_sched_group_barrier(EXP, 1, G);
        __builtin_amdgcn_sched_group_barrier(VALU, 2, G);
        if constexpr(i.value >= Rpt / 2 && i.value < Rpt / 2 + 5)
        {
            __builtin_amdgcn_sched_group_barrier(DS_WRITE, 1, G);
        }
    });
}

// buffer_load ... lds from inline asm: M0 = the wave-uniform LDS destination, the hardware
// adds lane * BYTES. Asm so that the compiler does not know LDS-DMA is in flight: it keeps
// alias books on LDS-DMA, cannot tell the KV slots apart, and answers with a vmcnt(0) in
// front of LDS reads -- measured in front of every V refill, draining the DMA of t+2 a
// stage after it was issued. The instruction offset is left at 0 because it would shift the
// LDS end of the copy too. s_nop: an M0 write needs one wait state before an LDS-DMA.
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

// --- Q gmem->register (Q stays in registers for the whole request) ---
//
// The B operand of the 16x16x128 f8f6f4 is 32 B per lane, and those 32 B are NOT 32
// contiguous k: the instruction runs two 64-deep passes, so they are the 16 B at
// k = 64*p + 16*(lane / W_M) for p in [0, 2). That is the whole reason both this layout and
// the K read below put the lane's d-group on a stride of VEC and a separate "half" dim on a
// stride of 64.
//
// Wave w owns query rows (heads) 16w .. 16w+15, i.e. the 64 packed rows of the block.
template <class T>
__device__ inline auto make_layout_q_nope(int warp_id, int lane_id)
{
    constexpr auto q_shape =
        opus::make_tuple(opus::number<T::GEMM0_E_M>{},
                         opus::number<T::T_M>{},
                         opus::number<T::W_M>{},
                         opus::number<T::D_NOPE_SIZE / T::W_K>{}, // 4 whole k-steps
                         opus::number<T::W_K / T::W_K_HALF>{},    // 2 operand halves
                         opus::number<T::WARP_SIZE / T::W_M>{},   // lane's d-group
                         opus::number<T::VEC_Q>{});

    constexpr auto q_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        q_shape,
        opus::unfold_x_stride(q_dim, q_shape, opus::tuple{opus::number<T::D_HEAD_SIZE>{}, 1_I}),
        opus::unfold_p_coord(q_dim, opus::tuple{warp_id, lane_id % T::W_M, lane_id / T::W_M}));
}

// Q rope, d in [D_NOPE_SIZE, D_HEAD_SIZE); the caller offsets the gmem base by +D_NOPE_SIZE.
// It is the LOW half of k-step 4 -- D_ROPE_SIZE is exactly (WARP_SIZE / W_M) * VEC_Q, one
// dwordx4 per lane -- and the high half is the 640-padding, which the caller zeroes. The
// zero matters on both operands: 0 * NaN is NaN, and an fp8 e4m3 0x7F is a NaN, so the K
// side's pad half is cleared too rather than trusted to be finite.
template <class T>
__device__ inline auto make_layout_q_rope(int warp_id, int lane_id)
{
    constexpr auto q_shape = opus::make_tuple(opus::number<T::GEMM0_E_M>{},
                                              opus::number<T::T_M>{},
                                              opus::number<T::W_M>{},
                                              opus::number<T::WARP_SIZE / T::W_M>{},
                                              opus::number<T::VEC_Q>{});

    constexpr auto q_dim =
        opus::make_tuple(opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
                         opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        q_shape,
        opus::unfold_x_stride(q_dim, q_shape, opus::tuple{opus::number<T::D_HEAD_SIZE>{}, 1_I}),
        opus::unfold_p_coord(q_dim, opus::tuple{warp_id, lane_id % T::W_M, lane_id / T::W_M}));
}

// --- KV gmem->LDS (K_LOAD) ---
//
// Pass p of wave w: lane l carries token 32 p + 4 (l / 8) + w and, of every chunk, the d-group
// l % 8. The hardware places the lane at M0 + 16 l, i.e. row l / 8 of the wave's block,
// position l % 8. No swizzle: neither reader needs one, their lanes already spread over the
// banks through the block pitch (see the bank notes on make_layout_rk / make_layout_rv);
// measured the same conflicts (none) and time with and without an XOR by row bit 2.

// The token each lane DMAs in each pass; the caller adds tile_idx * KV_TILE_SIZE.
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

// KV global source, the d walk only (the token rides the per-lane page offset): NCHUNK
// chunks of 128 d, the lane's d-group. DG_MASK = 3 is the rope chunk, where the four pad
// d-groups fold back onto the four real ones.
template <class T, int NCHUNK, int DG_MASK>
__device__ inline auto make_layout_gkv(int lane_id)
{
    constexpr auto shape = opus::make_tuple(
        opus::number<NCHUNK>{}, opus::number<T::kv_threads_d>{}, opus::number<T::VEC_KV>{});
    constexpr auto dim = opus::make_tuple(opus::make_tuple(opus::y_dim{}),
                                          opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    const int dg = (lane_id % T::kv_threads_d) & DG_MASK;

    return opus::make_layout(
        shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{opus::number<T::smem_row_kv>{}, 1_I}),
        opus::unfold_p_coord(dim, opus::tuple{dg}));
}

// KV LDS destination, wave-uniform because buffer_load_lds' dst is: block `warp_id` of each
// of NCHUNK chunks (the pass adds NUM_WARPS blocks, as a caller offset).
template <class T, int NCHUNK>
__device__ inline auto make_layout_skv(int warp_id)
{
    constexpr auto shape = opus::make_tuple(
        opus::number<NCHUNK>{}, opus::number<T::NUM_WARPS>{}, opus::number<T::VEC_KV>{});
    constexpr auto dim =
        opus::make_tuple(opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        shape,
        opus::make_tuple(opus::number<T::smem_kv_chunk>{}, opus::number<T::smem_kv_block>{}, 1_I),
        opus::unfold_p_coord(dim, opus::tuple{warp_id}));
}

// K LDS->register, the A operand of the 16x16x128 f8f6f4: lane l holds token 16 en + i
// (i = l % 16) and the 16 B at d-group l / 16 (+4 for the operand's second pass). Inverting
// the deal: block i % 4 (+4 for en / 2), row i / 4 (+4 for en % 2), position l / 16 -- the
// e_n tile is a caller offset (k_off).
//
// Banks: an 8-lane group spans blocks 0..3 x rows 0..1, i.e. 8 x (block pitch 264 dwords
// = 8 mod 64) + 32 x row: 8 distinct 4-bank groups, the same signature the previous K read
// measured conflict-free.
template <class T>
__device__ inline auto make_layout_rk(int lane_id)
{
    constexpr int blocks = T::NUM_WARPS; // i % 4
    constexpr auto shape = opus::make_tuple(opus::number<blocks>{},
                                            opus::number<T::W_N / blocks>{},       // i / 4
                                            opus::number<T::WARP_SIZE / T::W_N>{}, // d-group
                                            opus::number<T::VEC_KV>{});
    constexpr auto dim   = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    const int i = lane_id % T::W_N;

    return opus::make_layout(
        shape,
        opus::make_tuple(opus::number<T::smem_kv_block>{},
                         opus::number<T::smem_row_kv>{},
                         opus::number<T::VEC_KV>{},
                         1_I),
        opus::unfold_p_coord(dim, opus::tuple{i % blocks, i / blocks, lane_id / T::W_N}));
}

// --- V LDS->register transpose read (ds_read_b64_tr_b8), V_LOAD ---
//
// The 32x32x64 operand wants lane l to hold d = l % 32 and, byte e, token
// 32 (e / 16) + 16 (l / 32) + e % 16. One ds_read_b64_tr_b8 fills 8 of those bytes for one
// d-tile: in each 16-lane group lane 2i + h supplies the 8 B at (token i of the group's run,
// d-half h) and gets back d = l % 16 for all 8 tokens. Instruction (r, ih) of d-tile dt has
// lane l = 32 g + 16 dsub + 2 i + h supply token 32 r + 16 g + 8 ih + i, d 32 dt + 16 dsub
// + 8 h, which the deal puts at
//
//   block 4 r + i % 4,   row 4 g + 2 ih + i / 4,   position 2 (dt % 4) + dsub
//
// -- (r, ih, dt) are constants, the rest the lane's own terms.
// Banks (b64 = 32 lanes / cycle on the 8 B bank pair, g fixed within one): 4 (i % 4)
// + 16 (i / 4) + 2 pos + h covers the 32 pairs exactly once.
template <class T>
__device__ inline auto make_layout_rv(int lane_id)
{
    constexpr int blocks = T::NUM_WARPS;
    constexpr auto shape = opus::make_tuple(opus::number<blocks>{},                   // i % 4
                                            opus::number<T::WARP_SIZE / T::W_N_PV>{}, // g
                                            opus::number<2>{},                        // i / 4
                                            opus::number<2>{},                        // dsub
                                            opus::number<2>{},                        // h
                                            opus::number<T::VEC_TR_V>{});
    constexpr auto dim   = opus::make_tuple(opus::make_tuple(
        opus::p_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    const int g    = lane_id / T::W_N_PV;
    const int dsub = (lane_id % T::W_N_PV) / (T::W_N_PV / 2);
    const int m    = lane_id % (T::W_N_PV / 2);
    const int i = m / 2, h = m % 2;

    return opus::make_layout(
        shape,
        opus::make_tuple(opus::number<T::smem_kv_block>{},
                         opus::number<T::smem_rows_per_16tok * T::smem_row_kv>{},
                         opus::number<T::smem_row_kv>{},
                         opus::number<T::VEC_KV>{},
                         opus::number<T::VEC_TR_V>{},
                         1_I),
        opus::unfold_p_coord(dim, opus::tuple{i % blocks, g, i / blocks, dsub, h}));
}

// O register->gmem store. PV is swap_ab, so its C is [d][q]: lane l holds query row
// 32 h + l % 32 and, of each d-tile, d = 8 (e / 4) + 4 (l / 32) + e % 4. stride_o_h is a
// parameter because the same layout serves the real output and the D_NOPE_SIZE-strided
// split-KV o_accum.
template <class T>
__device__ inline auto make_layout_o(int pv_h, int pv_dh, int lane_id, int stride_o_h)
{
    constexpr auto o_shape =
        opus::make_tuple(opus::number<T::NUM_WARPS / T::PV_WAVES_D>{}, // h
                         opus::number<T::W_N_PV>{},                    // q
                         opus::number<T::PV_WAVES_D>{},                // dh
                         opus::number<T::GEMM1_E_N>{},                 // dt
                         opus::number<T::W_M_PV * T::W_N_PV / T::WARP_SIZE / T::VEC_O>{},
                         opus::number<T::WARP_SIZE / T::W_N_PV>{},
                         opus::number<T::VEC_O>{});

    constexpr auto o_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(
            opus::p_dim{}, opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        o_shape,
        opus::unfold_x_stride(o_dim, o_shape, opus::tuple{stride_o_h, 1_I}),
        opus::unfold_p_coord(o_dim,
                             opus::tuple{pv_h, lane_id % T::W_N_PV, pv_dh, lane_id / T::W_N_PV}));
}

// --- softmax / scaling helpers (W_M = 16 on 64-wide waves: a row spans four lane groups,
//     so both reductions need permlane32_swap followed by permlane16_swap) ---
template <typename T, typename V>
__device__ inline typename T::D_ACC attn_row_max(const V& v_s)
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
    return max(std::bit_cast<float>(res16.x), std::bit_cast<float>(res16.y));
}

// Fused `v_s * scale - row_max`. The caller reduces the RAW scores and scales that single
// scalar instead (scale > 0, so max commutes with it).
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
// does not reassociate, so the chain compiles to s_len dependent v_add_f32 back to back.
template <typename T, typename V>
__device__ inline typename T::D_ACC attn_row_sum(const V& v_s)
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
    return std::bit_cast<float>(res16.x) + std::bit_cast<float>(res16.y);
}

template <typename T, typename V>
__device__ inline void scale_output_tile(V& v_o, typename T::D_ACC scale)
{
    constexpr opus::index_t o_len = opus::vector_traits<V>::size();
    opus::static_for<o_len>([&](auto i) { v_o[i.value] *= scale; });
}

// Pin the O accumulator as a scheduling/materialization fence, chunked into 8-lane groups so
// each "+v" operand can be allocated (a single one on the whole 128-VGPR v_o cannot).
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

// The online-softmax rescale of one AGPR-resident O tile. The PV MFMA is selected in its
// AGPR form (512 registers in play), and a plain VALU multiply on O makes the allocator home
// O in VGPRs -- measured: 16 accvgpr_read behind every PV MFMA, each waiting out the whole
// 64-cycle MFMA, or with O split per tile, all 128 reads hoisted out of the (rare) rescale
// branch into every phase. Spelled out on the AGPRs, O has one register class and the copies
// stay inside the branch. The caller owes an s_nop for the VALU-write -> MFMA-srcC hazard,
// which the hazard recognizer does not see through inline asm.
template <typename Tile>
__device__ inline void scale_tile_agpr(Tile& t, float s)
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

// Masks every score column past `last_valid_kv_pos` to -inf. v_s element (e_n, reg) at lane l
// is token e_n * W_N + 4 * (l / W_M) + reg, the MFMA's own C layout.
template <typename T, typename V>
__device__ inline void
attn_mask_kv_tile(V& v_s, int last_valid_kv_pos, int kv_tile_idx, opus::u32_t neg_inf_v)
{
    using D_ACC    = typename T::D_ACC;
    using D_ACC_X2 = opus::vector_t<D_ACC, 2>;
    using U32_X2   = opus::vector_t<opus::u32_t, 2>;

    constexpr int elems_per_wave_tile = (T::W_M * T::W_N) / T::WARP_SIZE;
    constexpr int c_pack              = 4;
    constexpr int c_rept              = elems_per_wave_tile / c_pack;
    constexpr int c_rept_stride       = (T::WARP_SIZE / T::W_M) * c_pack;

    const int k_start_pos = kv_tile_idx * T::KV_TILE_SIZE;
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

// Per-lane byte offsets of the P exchange (inside the P region) and of the two per-row float
// arrays (inside the scale or the row-sum region, same shape). GEMM0 side: row
// q = 16 w + l % 16, lane group g = l / 16 (also the float copy it writes). PV side: row
// q = 32 h + l % 32, 16 B group 2 r + l / 32. P's 16 B groups are XOR-swizzled by
// f(q) = (q / 4) % 4, which for both sides is a pure lane term; the region base being a
// multiple of 64 B, XORing the offset is XORing the address.
template <class T>
struct lds_lane_addrs
{
    int p_wr; // XOR with 16 * e_n for e_n tile e_n
    int p_rd; // the operand's first 16 B; the second is p_rd ^ 32
    int row_wr;
    int row_rd;

    __device__ lds_lane_addrs(int warp_id, int lane_id, int pv_h)
    {
        const int q0 = T::Q_TILE_SIZE * warp_id + lane_id % T::W_M;
        const int g0 = lane_id / T::W_M;
        const int q1 = T::PV_ROWS * pv_h + lane_id % T::W_N_PV;
        const int g1 = lane_id / T::W_N_PV;
        auto swz     = [](int q) { return (q / 4) % 4; };
        p_wr         = q0 * T::smem_p_row + 16 * swz(q0) + T::VEC_P * g0;
        p_rd         = q1 * T::smem_p_row + 16 * (g1 ^ swz(q1));
        row_wr       = (g0 * T::ROW_COPY_PITCH + q0) * static_cast<int>(sizeof(float));
        row_rd       = q1 * static_cast<int>(sizeof(float));
    }
};

template <class S>
__device__ inline opus::u32_t lds_addr(const S& s)
{
    return static_cast<opus::u32_t>(reinterpret_cast<__UINTPTR_TYPE__>(s.ptr));
}

// A handle of element type X at byte `off` of the LDS image `base`, by pointer arithmetic in
// the LDS address space. make_smem on generic pointers instead costs every handle its own
// generic-to-LDS null check, which the compiler rematerializes inside the loop together with
// an address add per access (measured: +36 instructions per two phases).
template <class X, class S>
__device__ inline opus::smem<X> smem_at(const S& base, size_t off)
{
    opus::smem<X> s{nullptr};
    s.ptr = base.ptr + off;
    return s;
}

// --- Pipelined KV-tile loop for one work item (see the stage map at the top) ---
// Q, the O accumulator and the online-softmax state are owned by the caller and passed by
// reference. m_row / l_row are the GEMM0 wave's rows (q = 16 w + l % 16); v_o is the PV
// wave's tile (q = 32 h + l % 32).
template <class Traits, class VQ, class VO>
__device__ __attribute__((always_inline)) void
mla_decode_fwd_pipelined(opus_mla_decode_fp8_kargs kargs,
                         int kv_ind_ptr_s,
                         int valid_kv_len,
                         int tile_begin,
                         int tile_end,
                         char* smem_buffer,
                         VQ& v_q,
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
    const int pv_h    = warp_id / T::PV_WAVES_D;
    const int pv_dh   = warp_id % T::PV_WAVES_D;
    int diag_kv_bound = 0;
    if constexpr(T::CAUSAL)
    {
        diag_kv_bound = (warp_id * T::W_M) / kargs.H + causal_diagonal;
    }

    const D_K* kv_base = reinterpret_cast<const D_K*>(kargs.kv_buffer_ptr);
    auto g_kv_indices  = make_gmem(kargs.kv_indices + kv_ind_ptr_s, valid_kv_len * sizeof(int));

    // KV_SLOTS slots of the K image (V is transpose-read out of them too). The slots rotate
    // every phase as the handles kv_prev / kv_cur below, and s_kv[] is only ever indexed by a
    // constant: a runtime index puts the array in scratch, and scratch VMEM traffic breaks
    // the hand-written vmcnt budget.
    static_assert(T::KV_SLOTS == 3);
    const auto lds              = make_smem(smem_buffer);
    smem<D_K> s_kv[T::KV_SLOTS] = {smem_at<D_K>(lds, 0),
                                   smem_at<D_K>(lds, T::smem_kv_slot_bytes),
                                   smem_at<D_K>(lds, 2 * T::smem_kv_slot_bytes)};
    // P(t-1) from the GEMM0 waves to the PV waves, then each row's rescale factor.
    auto s_p     = smem_at<D_K>(lds, T::smem_p_offset);
    auto s_scale = smem_at<D_ACC>(lds, T::smem_scale_offset);
    // The page-index ring: IDX_RING tiles x NUM_WARPS private copies of a tile's 64 indices.
    auto s_kv_indices = smem_at<int>(lds, T::smem_idx_offset);
    const lds_lane_addrs<T> la(warp_id, lane_id, pv_h);
    const int p_rd1 = la.p_rd ^ 32;

    constexpr int rope_chunk = T::smem_d_rpt_kv - 1;
    auto u_kv_indices        = make_layout_kv_indices<T>(warp_id, lane_id);
    auto u_gkv1              = make_layout_gkv<T, 1, T::kv_threads_d - 1>(lane_id);
    auto u_gkv_r             = make_layout_gkv<T, 1, T::kv_threads_d / 2 - 1>(lane_id);
    auto u_skv1              = make_layout_skv<T, 1>(warp_id);
    auto u_rk                = make_layout_rk<T>(lane_id);
    auto u_rv                = make_layout_rv<T>(lane_id);
    // The PV wave's d-half is whole chunks of the K image: a wave-uniform base.
    const int v_wave_off = pv_dh * (T::PV_D / T::smem_row_kv) * T::smem_kv_chunk;

    // Under LARGE_KV the K handle is a bare 64-bit pointer for global_load_lds, resolved down
    // to this lane's slot; otherwise it is a buffer descriptor, whose 32-bit num_records caps
    // the cache at 4 GiB and which carries the lane offset in the layout.
    constexpr int rope_g = rope_chunk * T::smem_row_kv;
    auto kv_handle       = [&](auto u, auto d0) {
        if constexpr(T::LARGE_KV)
            return global_load_base<T::VEC_KV>(kv_base + decltype(d0)::value, u);
        else
            return make_gmem(kv_base,
                             static_cast<unsigned>(static_cast<size_t>(kargs.total_tokens) *
                                                   kargs.stride_kv_page * sizeof(D_K)));
    };
    auto g_kv                   = kv_handle(u_gkv1, 0_I);
    [[maybe_unused]] auto g_kv1 = g_kv;
    auto g_kv_r                 = kv_handle(u_gkv_r, number<rope_g>{});

    // GEMM0: bare 16x16x128 f8f6f4. The trailing 0,0 are the block scales and must be
    // literal zeros -- a number<0> makes the operand poison and DCEs the body down to
    // .vgpr_count 4, and only 0 selects the bare form without v_mfma_ld_scale_b32.
    auto mfma0 = make_mfma<D_K, D_Q, D_ACC>(number<T::W_M>{}, number<T::W_N>{}, number<T::W_K>{});
    // GEMM1: 32x32x64 f8f6f4, called as (V, P) so the HW A operand is V (M = d) and C comes
    // out [d][q]: one query row per lane, hence one rescale factor per lane.
    auto mfma1 =
        make_mfma<D_K, D_K, D_ACC>(number<T::W_M_PV>{}, number<T::W_N_PV>{}, number<T::W_K_PV>{});

    using k_half_t  = vector_t<D_K, T::VEC_KV>;
    using k_step_t  = vector_t<D_K, T::W_N * T::W_K / T::WARP_SIZE>;         // 32, one operand
    using q_step_t  = vector_t<D_Q, T::W_M * T::W_K / T::WARP_SIZE>;         // 32
    using s_tile_t  = vector_t<D_ACC, T::W_M * T::W_N / T::WARP_SIZE>;       // 4
    using pv_op_t   = vector_t<D_K, T::W_M_PV * T::W_K_PV / T::WARP_SIZE>;   // 32
    using o_tile_t  = vector_t<D_ACC, T::W_M_PV * T::W_N_PV / T::WARP_SIZE>; // 16
    using p_piece_t = vector_t<D_K, T::VEC_P>;
    using v_piece_t = vector_t<D_K, T::VEC_TR_V>;

    // K for the EN_GROUP e_n tiles of one k-step, K_DEPTH buffers over steps, as operand
    // halves so each ds_read is a definition the AGPR pin can bind to.
    opus::array<k_half_t, K_HALVES<T>> v_k[K_DEPTH<T>];
    // Scores for two tiles, so tile t's softmax runs alongside tile t-1's PV, as MFMA tiles so
    // each GEMM0 MFMA is a definition the VGPR pin can bind to; the softmax sees them flat.
    // Indexed only by unrolled compile-time constants -- a runtime index sinks the array to
    // scratch.
    using s_flat_t = vector_t<D_ACC, T::GEMM0_E_N * T::W_M * T::W_N / T::WARP_SIZE>;
    opus::array<s_tile_t, T::GEMM0_E_N> v_s[2];
    auto sflat   = [](auto& a) -> s_flat_t& { return *reinterpret_cast<s_flat_t*>(&a); };
    auto clear_s = [](auto& a) {
        static_for<T::GEMM0_E_N>([&](auto i) { a[i.value] = s_tile_t{0}; });
    };
    pv_op_t v_p;
    // The PV wave's whole V for one tile, a d-tile per entry, in AGPRs: V(t-1) at the start
    // of phase t, replaced d-tile by d-tile with V(t) as PV(t-1) consumes it.
    pv_op_t v_v[T::GEMM1_E_N];
    D_ACC p_scale = 1.0f;

    opus::array<q_step_t, T::GEMM0_E_K> v_q_steps;
    static_for<T::GEMM0_E_K>([&](auto i) {
        MLA16MX4_PIN_VGPR(Q_VGPR_BASE<T> + i.value * Q_VGPR_PER_STEP<T>)
        v_q_steps[i.value] = reinterpret_cast<const q_step_t*>(&v_q)[i.value];
    });
    // O as GEMM1_E_N separate MFMA C tiles for the whole loop, not views into the caller's
    // flat vector: the allocator handles eight 16-register values far better than slices of
    // one 128-register one. Anchored in AGPRs at every phase boundary.
    o_tile_t v_o_tiles[T::GEMM1_E_N];
    static_for<T::GEMM1_E_N>([&](auto i) {
        constexpr int n    = vector_traits<o_tile_t>::size();
        v_o_tiles[i.value] = slice(v_o, number<i.value * n>{}, number<(i.value + 1) * n>{});
    });
    auto pin_o = [&]() {
#pragma unroll
        for(int i = 0; i < T::GEMM1_E_N; i++)
            asm volatile("" : "+a"(v_o_tiles[i])::);
    };
    // Always behind an lgkmcnt(0) covering the refills, so a copy the allocator might make to
    // meet it copies arrived data. Without the anchors at the end of the prologue and of
    // stage 3 the allocator shuttles V between register files every phase (measured: +64
    // accvgpr reads and writes per PV, ~9 us at b64 c16384).
    auto pin_v = [&]() {
        using w_t = vector_t<int, sizeof(pv_op_t) / sizeof(int)>; // "a" takes no _BitInt vector
        auto* w   = reinterpret_cast<w_t*>(&v_v[0]);
#pragma unroll
        for(int i = 0; i < T::GEMM1_E_N; i++)
            asm volatile("" : "+a"(w[i])::);
    };
    // The register pin on the K loads is only a hint, and the allocator drops it: the MFMA
    // takes either file, so the COPY into the pinned AGPR folds away and the ds_read stays in
    // a VGPR. An asm operand is a use the compiler waits for, so forcing AGPRs right before
    // the MFMA moves arrived data at worst. Pin toolchain only: without S pinned to VGPRs the
    // scores take the AGPRs instead and this costs ~1.4 us (b64 c16384).
    auto pin_k = [&]([[maybe_unused]] auto& k) {
#ifdef MLA16MX4_HAS_PIN
        using w_t = vector_t<int, sizeof(k_half_t) / sizeof(int)>;
        auto* w   = reinterpret_cast<w_t*>(&k[0]);
#pragma unroll
        for(int i = 0; i < K_HALVES<T>; i++)
            asm volatile("" : "+a"(w[i])::);
#endif
    };

    // Inside a slot: GEMM0 k-step ek is chunk ek; e_n tile en is 32 (en / 2) tokens = one
    // pass (NUM_WARPS blocks) plus 16 (en % 2) = smem_rows_per_16tok rows; the operand's
    // second pass is 64 d on. PV d-tile dt is chunk dt / 4 plus 32 d; token run (r, ih) is
    // one pass per r and two rows per ih.
    auto k_off = [](auto en, auto ek, auto pp) {
        constexpr int e = decltype(en)::value;
        return number<decltype(ek)::value * T::smem_kv_chunk +
                      (e / 2) * T::NUM_WARPS * T::smem_kv_block +
                      (e % 2) * T::smem_rows_per_16tok * T::smem_row_kv +
                      decltype(pp)::value * T::W_K_HALF>{};
    };
    auto v_off = [](auto dt, auto r, auto ih) {
        constexpr int d            = decltype(dt)::value;
        constexpr int tiles_per_ch = T::smem_row_kv / T::W_M_PV; // 4
        return number<(d / tiles_per_ch) * T::smem_kv_chunk +
                      decltype(r)::value * T::NUM_WARPS * T::smem_kv_block +
                      decltype(ih)::value * 2 * T::smem_row_kv + (d % tiles_per_ch) * T::W_M_PV>{};
    };

    // One GEMM0 step: the EN_GROUP e_n tiles' operands for k-step EK. The last step's second
    // pass is the 640-padding, cleared rather than read -- an fp8 0x7F there is a NaN and
    // 0 * NaN poisons the whole score row.
    // `buf` is which K buffer dst is: a compile-time value so the pins below can be.
    // The lane's offset is summed once, out here: added to a rotating slot handle, the layout's
    // three lane terms are otherwise re-added on every step.
    const int k_lane = layout_to_offsets<T::VEC_KV>(u_rk)[0];
    auto load_k_step = [&](auto& dst, smem<D_K> kv, auto grp, auto ek, auto buf) {
        constexpr int EK  = decltype(ek)::value;
        constexpr int GRP = decltype(grp)::value;
        constexpr int base =
            K_AGPR_BASE<T> + decltype(buf)::value * K_HALVES<T> * K_AGPR_PER_HALF<T>;
        static_for<T::EN_GROUP>([&](auto e) {
            constexpr auto en = number<GRP * T::EN_GROUP + e.value>{};
            MLA16MX4_PIN_AGPR(base + (e.value * 2) * K_AGPR_PER_HALF<T>)
            dst[e.value * 2] =
                kv.template _load<T::VEC_KV>(k_lane + decltype(k_off(en, ek, 0_I))::value);
            if constexpr(EK + 1 < T::GEMM0_E_K)
            {
                MLA16MX4_PIN_AGPR(base + (e.value * 2 + 1) * K_AGPR_PER_HALF<T>)
                dst[e.value * 2 + 1] =
                    kv.template _load<T::VEC_KV>(k_lane + decltype(k_off(en, ek, 1_I))::value);
            }
            else
            {
                // An assignment rather than clear() so this definition carries a pin too.
                MLA16MX4_PIN_AGPR(base + (e.value * 2 + 1) * K_AGPR_PER_HALF<T>)
                dst[e.value * 2 + 1] = k_half_t{};
            }
        });
    };

    // K(0 .. K_DEPTH-1) of slot `kv`, waiting only for K(0). Anything read from LDS just
    // before stays covered by the wait.
    static_assert(K_DEPTH<T> < K_STEPS<T>);
    auto load_k_head = [&](smem<D_K> kv) {
        static_for<K_DEPTH<T>>([&](auto j) {
            load_k_step(v_k[j.value],
                        kv,
                        number<j.value / T::GEMM0_E_K>{},
                        number<j.value % T::GEMM0_E_K>{},
                        j);
        });
        s_waitcnt_lgkmcnt(number<k_reads_after<T, 0, 0>(0, -1)>{});
    };

    // GEMM0: EN_GROUPS x GEMM0_E_K steps of EN_GROUP MFMA each, step s in buffer s % K_DEPTH,
    // which it refills with step s + K_DEPTH once its MFMA are issued.
    // `per_step(step)` runs after each step's MFMA pair: the phase hangs one KV DMA on each.
    // `lgkm_extra` is how many LDS reads per_step issues on each of the first `extra_steps`
    // steps; the waits leave them in flight like the prefetches, K_DEPTH - 1 steps each.
    auto compute_qk =
        [&](auto& s, smem<D_K> kv, auto&& per_step, auto sbuf, auto lgkm_extra, auto extra_steps) {
            constexpr int X  = decltype(lgkm_extra)::value;
            constexpr int XN = decltype(extra_steps)::value;
            constexpr int sbase =
                S_VGPR_BASE<T> + decltype(sbuf)::value * T::GEMM0_E_N * S_VGPR_PER_TILE<T>;
            static_for<T::EN_GROUPS>([&](auto grp) {
                static_for<T::GEMM0_E_K>([&](auto ek) {
                    constexpr int step = grp.value * T::GEMM0_E_K + ek.value;
                    constexpr int slot = step % K_DEPTH<T>;
                    auto* k_step       = reinterpret_cast<k_step_t*>(&v_k[slot][0]);
                    pin_k(v_k[slot]);
                    static_for<T::EN_GROUP>([&](auto e) {
                        constexpr int en = grp.value * T::EN_GROUP + e.value;
                        MLA16MX4_PIN_VGPR(sbase + en * S_VGPR_PER_TILE<T>)
                        s[en] = mfma0(k_step[e.value], v_q_steps[ek.value], s[en], 0, 0);
                    });
                    per_step(number<step>{});
                    constexpr int next = step + K_DEPTH<T>;
                    if constexpr(next < K_STEPS<T>)
                        load_k_step(v_k[slot],
                                    kv,
                                    number<next / T::GEMM0_E_K>{},
                                    number<next % T::GEMM0_E_K>{},
                                    number<slot>{});
                    if constexpr(step + 1 < K_STEPS<T>)
                        s_waitcnt_lgkmcnt(number<k_reads_after<T, X, XN>(step + 1, step)>{});
                });
            });
        };

    // V d-tile dt of the tile in slot `kv`: four transpose reads into v_v[dt]. The lane's base
    // address, the wave's d-half included, is loop-invariant; the slot base is a scalar and
    // the (dt, r, ih) step folds into the instruction's offset field.
    //
    // The builtin, not opus's inline-asm tr_load: V outlives its read by most of a phase,
    // and an asm result is one the compiler believes ready at once -- measured, it copied
    // tr_load destinations (a scratch AGPR moved into v_v two instructions after the read,
    // tied "+a" operands included, and V moved wholesale at a branch merge) before the data
    // had arrived. As a builtin the read is on the compiler's lgkmcnt books like any LDS
    // load; the DMAs being asm is what keeps it from also waiting out the KV DMA here.
    using v2i_t      = vector_t<int, 2>;
    const int v_lane = layout_to_offsets<T::VEC_TR_V>(u_rv)[0] + v_wave_off;
    auto load_v      = [&](smem<D_K> kv, auto dt) {
        auto* piece = reinterpret_cast<v_piece_t*>(&v_v[decltype(dt)::value]);
        auto* base  = kv.ptr + v_lane;
        static_for<2>([&](auto r) {
            static_for<2>([&](auto ih) {
                constexpr int off = decltype(v_off(dt, r, ih))::value;
                piece[r.value * 2 + ih.value] =
                    __builtin_bit_cast(v_piece_t,
                                       __builtin_amdgcn_ds_read_tr8_b64_v2i32(
                                           reinterpret_cast<OPUS_LDS_ADDR v2i_t*>(base + off)));
            });
        });
    };
    // PV(t-1) out of registers: one MFMA per d-tile. With REFILL each d-tile's registers are
    // refilled with V(t) from slot `kv` one MFMA later (so the MFMA that reads them has long
    // issued), the last after the loop. `co(i)` rides the shadow of MFMA i.
    auto compute_pv = [&](auto& o, smem<D_K> kv, auto&& co, auto refill) {
        constexpr bool REFILL = decltype(refill)::value;
        static_for<T::GEMM1_E_N>([&](auto i) {
            constexpr int idx = i.value;
            MLA16MX4_PIN_AGPR(idx * O_AGPR_PER_TILE<T>)
            o[idx] = mfma1(v_v[idx], v_p, o[idx], 0, 0);
            if constexpr(REFILL && idx > 0)
                load_v(kv, number<idx - 1>{});
            __builtin_amdgcn_sched_barrier(0);
            co(i);
            __builtin_amdgcn_sched_barrier(0);
        });
        if constexpr(REFILL)
            load_v(kv, number<T::GEMM1_E_N - 1>{});
    };
    auto no_co = [](auto) {};

    constexpr index_t s_len      = T::GEMM0_E_N * T::W_M * T::W_N / T::WARP_SIZE; // 16
    constexpr index_t s_half_len = s_len / 2;
    constexpr int QK_MFMA_CNT    = T::GEMM0_E_N * T::GEMM0_E_K; // 20

    // Online softmax: skip the O rescale entirely while every lane's new row max is within
    // this much of the running one. Decided by a ballot per GEMM0 wave; the factor it
    // implies (exactly 1.0 when skipped) is what goes to the PV waves.
    constexpr D_ACC RESCALE_THRESHOLD = 8.0f;
    D_ACC rescale_m                   = 1.0f;
    D_ACC row_max;
    bool below_thresh, all_below;

    // The softmax tail of the previous tile, ending in P and its rows' rescale factor going
    // to LDS for the PV waves.
    auto tail_and_publish = [&](auto& vs_arr) {
        auto& vs = sflat(vs_arr);
        attn_exp2_slice<T, s_half_len, s_half_len>(vs);
        l_row += attn_row_sum<T>(vs);
        auto p8     = cast<D_K>(vs);
        auto* piece = reinterpret_cast<const p_piece_t*>(&p8);
        static_for<T::GEMM0_E_N>([&](auto en) {
            s_p.template _store<T::VEC_P>(piece[en.value], la.p_wr ^ (16 * en.value));
        });
        s_scale.template _store<1>(vector_t<D_ACC, 1>{rescale_m}, la.row_wr);
    };
    auto read_p = [&]() {
        auto* half = reinterpret_cast<k_half_t*>(&v_p);
        half[0]    = s_p.template _load<T::VEC_KV>(la.p_rd);
        half[1]    = s_p.template _load<T::VEC_KV>(p_rd1);
        p_scale    = s_scale.template _load<1>(la.row_rd)[0];
    };
    // Only a wave with a row whose factor is not exactly 1 pays for the 128 multiplies.
    auto rescale_o = [&]() {
        if(__builtin_amdgcn_ballot_w64(p_scale != 1.0f) != 0)
        {
#pragma unroll
            for(int i = 0; i < T::GEMM1_E_N; i++)
                scale_tile_agpr(v_o_tiles[i], p_scale);
            asm volatile("s_nop 4" ::);
        }
    };

    // Page indices go through LDS as well: one buffer_load_dword lds per wave per tile puts
    // the tile's 64 indices in the wave's own ring entry, four tiles ahead, and the two a
    // lane needs come back with plain ds_reads once the phase's vmcnt has covered the DMA.
    // Both obvious alternatives were measured broken or slow: as an ordinary buffer load the
    // result is waited for by the compiler, which does not count LDS-DMA on vmcnt and so
    // drains the in-flight KV DMA with it (vmcnt(3) before every phase's first DMA); as an
    // inline-asm load the compiler takes the result as ready at once and copied it -- and
    // handed its VGPR to an MFMA accumulator -- before the load had returned.
    // Every lane's offset is clamped into the request, so no index DMA is ever out of range
    // and a tile past the end simply repeats the last token's page.
    const auto idx_tok = layout_to_offsets<1>(u_kv_indices); // token of each pass, in tile
    auto idx_slot      = [&](int tile_idx) {                 // byte offset of the wave's ring entry
        return (((tile_idx - tile_begin) & (T::IDX_RING - 1)) * T::NUM_WARPS + warp_id) *
               static_cast<int>(T::smem_idx_tile);
    };
    vector_t<int, 4> idx_rsrc;
    __builtin_memcpy(&idx_rsrc, &g_kv_indices.cached_rsrc, sizeof(idx_rsrc));
    auto issue_idx_dma = [&](int tile_idx) {
        const int tok = min(tile_idx * T::KV_TILE_SIZE + lane_id, valid_kv_len - 1);
#if MLA16MX4_DMA_ASYNC_LOAD
        g_kv_indices.template async_load<1>(smem_buffer + T::smem_idx_offset + idx_slot(tile_idx),
                                            tok);
#else
        buffer_load_lds_asm<sizeof(int)>(idx_rsrc,
                                         static_cast<u32_t>(tok) * sizeof(int),
                                         lds_addr(s_kv_indices) +
                                             static_cast<u32_t>(idx_slot(tile_idx)));
#endif
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

    // KV gmem->LDS, no register round trip, one instruction at a time: DMA i of a tile is
    // pass i / smem_d_rpt_kv, chunk i % smem_d_rpt_kv (the last chunk being rope + pad). One
    // call is one buffer_load_lds (m0 rewritten, s_nop, load), so the phase can spread them
    // over GEMM0's MFMA: issued back to back they stall at issue behind a saturated memory
    // pipe -- measured ~190 cycles each by thread trace -- with nothing in the MAI pipe.
    //
    // A tile past the range still gets its DMA, so every phase issues the same count and
    // the vmcnt budget is a constant; its clamped page indices point at the last token's row
    // -- cache-hot, real, harmless traffic into the free slot.
    //
    // gfx950 counts buffer_load_lds on vmcnt. Should global_load_lds (LARGE_KV) count on
    // lgkmcnt instead, as the SP3 source assumes, the result stays correct -- every DMA is
    // followed by an lgkmcnt(0) (GEMM0's last step, stage2) before its slot is read -- and
    // only the overlap suffers.
    using poff_t     = decltype(kv_page_offset(0));
    auto dma_offsets = [&](const auto& pages) {
        vector_t<poff_t, T::KV_PASSES> po;
        static_for<T::KV_PASSES>([&](auto p) { po[p.value] = kv_page_offset(pages[p.value]); });
        return po;
    };
    [[maybe_unused]] vector_t<int, 4> kv_rsrc;
    [[maybe_unused]] u32_t gkv_lane = 0, gkv_r_lane = 0;
    if constexpr(!T::LARGE_KV)
    {
        __builtin_memcpy(&kv_rsrc, &g_kv.cached_rsrc, sizeof(kv_rsrc));
        gkv_lane   = static_cast<u32_t>(layout_to_offsets<T::VEC_KV>(u_gkv1)[0]);
        gkv_r_lane = static_cast<u32_t>(layout_to_offsets<T::VEC_KV>(u_gkv_r)[0]);
    }
    const u32_t skv_wave = static_cast<u32_t>(warp_id * T::smem_kv_block);
    auto issue_dma       = [&](smem<D_K> kv, const auto& po, auto i) {
        constexpr int p        = decltype(i)::value / T::smem_d_rpt_kv;
        constexpr int c        = decltype(i)::value % T::smem_d_rpt_kv;
        constexpr int pass_off = p * T::NUM_WARPS * T::smem_kv_block;
        constexpr int s_off    = c * T::smem_kv_chunk + pass_off;
        const auto poff        = po[p];
        if constexpr(T::LARGE_KV)
        {
            if constexpr(c < rope_chunk)
                global_load<T::VEC_KV>(
                    g_kv1 + (poff + c * T::smem_row_kv), kv.ptr, u_gkv1, u_skv1 + s_off);
            else
                global_load<T::VEC_KV>(g_kv_r + poff, kv.ptr, u_gkv_r, u_skv1 + s_off);
        }
        else
        {
#if MLA16MX4_DMA_ASYNC_LOAD
            if constexpr(c < rope_chunk)
                async_load<T::VEC_KV>(
                    g_kv, kv.ptr, u_gkv1 + (poff + c * T::smem_row_kv), u_skv1 + s_off);
            else
                async_load<T::VEC_KV>(g_kv, kv.ptr, u_gkv_r + (poff + rope_g), u_skv1 + s_off);
#else
            const u32_t voff = (c < rope_chunk)
                                   ? gkv_lane + static_cast<u32_t>(poff + c * T::smem_row_kv)
                                   : gkv_r_lane + static_cast<u32_t>(poff + rope_g);
            buffer_load_lds_asm<T::VEC_KV>(
                kv_rsrc, voff, lds_addr(kv) + skv_wave + static_cast<u32_t>(s_off));
#endif
        }
    };
    auto async_load_kv = [&](smem<D_K> kv, const auto& pages) {
        const auto po = dma_offsets(pages);
        static_for<T::kv_buffer_load_insts>([&](auto i) { issue_dma(kv, po, i); });
    };

    const u32_t neg_inf_v = std::bit_cast<u32_t>(-numeric_limits<D_ACC>::infinity());

    // Only the tiles that can contain invalid columns pay for a mask: those whose last
    // column lies past `bound`, the tighter of the range end and (CAUSAL) this wave's
    // diagonal. The diagonal is per wave -- its 16 rows are one query token for every
    // nhead >= 16 -- and wave-uniform, so the test is scalar. It is NOT enough to mask the
    // range's last tile: the diagonal reaches qlen - 1 columns back from the range end, into
    // the previous tile whenever the last one holds fewer columns than that (measured: ctx
    // 65 / 129 / 193 at qlen 4 with one split).
    int mask_bound = valid_kv_len - 1;
    if constexpr(T::CAUSAL)
    {
        mask_bound = diag_kv_bound < mask_bound ? diag_kv_bound : mask_bound;
    }
    auto mask_oob_scores = [&](auto& s, int tile_idx) {
        if((tile_idx + 1) * T::KV_TILE_SIZE - 1 > mask_bound)
            attn_mask_kv_tile<T>(sflat(s), mask_bound, tile_idx, neg_inf_v);
    };

    auto stage_end = [&]() { __builtin_amdgcn_sched_barrier(0); };
    // A workgroup barrier over LDS. The asm memory clobbers keep the compiler from moving an
    // LDS access across it (s_barrier alone is not a memory operation to the IR); a real
    // fence would do that too, but at workgroup scope it also drains vmcnt, i.e. the KV DMA
    // that is supposed to stay in flight. s_barrier does not wait for a wave's own LDS
    // traffic, so every caller has an lgkmcnt(0) in front of it (stage3's end for B1).
    auto lds_barrier = [&]() {
        asm volatile("" ::: "memory");
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("" ::: "memory");
    };

    pin_o();

    // --- Prologue: tile_begin into slot 0 (drained); tile_begin+1 and +2 in flight into
    //     slots 1 and 2, each followed by the index DMA of the tile two after it -- the
    //     steady-state issue pattern; then QK, softmax head and the whole V of tile_begin ---
    issue_idx_dma(tile_begin);
    issue_idx_dma(tile_begin + 1);
    issue_idx_dma(tile_begin + 2);
    __builtin_amdgcn_s_waitcnt(0);
    asm volatile("" ::: "memory");
    const auto pages_a = load_kv_pages(tile_begin);
    const auto pages_b = load_kv_pages(tile_begin + 1);
    const auto pages_c = load_kv_pages(tile_begin + 2);
    async_load_kv(s_kv[0], pages_a);
    __builtin_amdgcn_s_waitcnt(0);
    lds_barrier();
    async_load_kv(s_kv[1], pages_b);
    issue_idx_dma(tile_begin + 3);
    async_load_kv(s_kv[2], pages_c);
    issue_idx_dma(tile_begin + 4);
    __builtin_amdgcn_sched_barrier(0);

    load_k_head(s_kv[0]);
    stage_end();

    // V(tile_begin) for the first PV rides the shadow of this GEMM0: d-tile s on step s.
    static_assert(T::GEMM1_E_N <= K_STEPS<T>, "one V d-tile per GEMM0 step at most");
    clear_s(v_s[0]);
    compute_qk(
        v_s[0],
        s_kv[0],
        [&](auto step) {
            if constexpr(decltype(step)::value < T::GEMM1_E_N)
                load_v(s_kv[0], step);
        },
        0_I,
        number<T::v_ds_read_insts>{},
        number<T::GEMM1_E_N>{});
    mask_oob_scores(v_s[0], tile_begin);
    m_row = max(m_row, temperature_scale * attn_row_max<T>(sflat(v_s[0])));
    attn_scale_sub_row<T>(sflat(v_s[0]), temperature_scale, m_row);
    attn_exp2_slice<T, 0, s_half_len>(sflat(v_s[0]));
    asm volatile("" : "+v"(sflat(v_s[0]))::);
    rescale_m = 1.0f;
    s_waitcnt_lgkmcnt(0_I);
    pin_v();
    stage_end();

    // kv_prev is slot t-1 (free once B1 is passed, so DMA(t+2) goes there), kv_cur slot t.
    auto run_phase = [&](auto& vs_cur_arr,
                         auto& vs_prev,
                         smem<D_K> kv_prev,
                         smem<D_K> kv_cur,
                         int t,
                         auto sbuf) {
        auto& vs_cur = sflat(vs_cur_arr);
        // stage0 [mem]: wait out DMA(t) and the index DMA of t+2, leaving DMA(t+1) and the
        // index DMA of t+3 in flight; B1 then publishes K(t) and tells every wave that V(t-1)
        // -- read into registers last phase -- no longer needs its slot, so t+2 goes there
        // during this phase's GEMM0. The page reads go first so the lgkmcnt below covers them.
        s_waitcnt_vmcnt(number<T::kv_buffer_load_insts + T::kv_index_load_insts>{});
        lds_barrier();
        const auto pages = load_kv_pages(t + 2);
        load_k_head(kv_cur);
        const auto po = dma_offsets(pages);
        stage_end();

        // stage1 [compute]: gemm0 QK(t) [20 MFMA, one DMA of t+2 per MFMA pair] || softmax
        // tail(t-1) and its P / scale stores.
        __builtin_amdgcn_s_setprio(1);
        clear_s(vs_cur_arr);
        compute_qk(
            vs_cur_arr, kv_cur, [&](auto step) { issue_dma(kv_prev, po, step); }, sbuf, 0_I, 0_I);
        issue_idx_dma(t + 4);
        tail_and_publish(vs_prev);
        sched_compute_qk<QK_MFMA_CNT, 1, T::kv_buffer_load_insts + T::kv_index_load_insts>();
        stage_end();

        // stage2 [mem]: mask S(t); B2 publishes P(t-1); read it and its scale.
        mask_oob_scores(vs_cur_arr, t);
        s_waitcnt_lgkmcnt(0_I);
        lds_barrier();
        read_p();
        s_waitcnt_lgkmcnt(0_I);
        stage_end();

        // stage3 [compute]: gemm1 PV(t-1) [8 MFMA x 64 cycles] out of registers, refilling
        // them with V(t) from slot t, with tile t's softmax head chopped into per-d-tile
        // chunks. The head reads only vs_cur, so nothing in it depends on the PV chain.
        // Chunk k after PV d-tile k:
        //   0,1  local row max, permlane32     2  permlane16, rescale decision, m / l update
        //   3,4  fused scale-subtract          4-6  exp2 of the first half
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
                vector_t<u32_t, 2> r = __builtin_amdgcn_permlane32_swap(
                    std::bit_cast<u32_t>(rmx), std::bit_cast<u32_t>(rmx), false, true);
                rmx = max(std::bit_cast<float>(r.x), std::bit_cast<float>(r.y));
            }
            else if constexpr(k == 2)
            {
                vector_t<u32_t, 2> r = __builtin_amdgcn_permlane16_swap(
                    std::bit_cast<u32_t>(rmx), std::bit_cast<u32_t>(rmx), false, true);
                row_max =
                    temperature_scale * max(std::bit_cast<float>(r.x), std::bit_cast<float>(r.y));
                below_thresh = ((row_max - m_row) <= RESCALE_THRESHOLD);
                all_below =
                    (__builtin_amdgcn_ballot_w64(below_thresh) == __builtin_amdgcn_read_exec());
                row_max   = all_below ? m_row : max(m_row, row_max);
                rescale_m = __builtin_amdgcn_exp2f(m_row - row_max);
                l_row *= rescale_m;
                m_row = row_max;
            }
            else if constexpr(k == 3)
            {
                static_for<s_half_len>([&](auto i) {
                    vs_cur[i.value] = __builtin_fmaf(vs_cur[i.value], temperature_scale, -row_max);
                });
            }
            else if constexpr(k == 4)
            {
                static_for<s_len - s_half_len>([&](auto i) {
                    constexpr int e = s_half_len + i.value;
                    vs_cur[e]       = __builtin_fmaf(vs_cur[e], temperature_scale, -row_max);
                });
                attn_exp2_slice<T, 0, 2>(vs_cur);
            }
            else if constexpr(k == 5)
            {
                attn_exp2_slice<T, 2, 3>(vs_cur);
            }
            else if constexpr(k == 6)
            {
                attn_exp2_slice<T, 5, s_half_len - 5>(vs_cur);
            }
        };
        static_assert(T::GEMM1_E_N >= 7, "the softmax head must fit in the PV shadow chunks");
        __builtin_amdgcn_s_setprio(1);
        rescale_o();
        pin_o();
        pin_v();
        compute_pv(v_o_tiles, kv_cur, head_chunk, std::true_type{});
        // What lets the next B1 release slot t: without it a wave can pass B1 with V refills
        // of the slot still in flight while another wave's DMA overwrites it.
        s_waitcnt_lgkmcnt(0_I);
        pin_o();
        pin_v();
        asm volatile("" : "+v"(vs_cur)::);
        __builtin_amdgcn_s_setprio(0);
        stage_end();
    };

    // --- Main loop: tiles tile_begin+1 .. tile_end-1, two phases unrolled per iteration so
    //     the two score buffers alternate as compile-time indices. The slot handles rotate
    //     (t-1, t, t+1) -> (t, t+1, t+2) each phase.
    smem<D_K> kv_prev = s_kv[0], kv_cur = s_kv[1], kv_next = s_kv[2];
    auto rotate = [&]() {
        const smem<D_K> kv = kv_prev;
        kv_prev            = kv_cur;
        kv_cur             = kv_next;
        kv_next            = kv;
    };
    int t = tile_begin + 1;
    for(; t + 1 < tile_end; t += 2)
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
    if(t < tile_end) // even tile count: one unpaired phase left
    {
        __builtin_amdgcn_sched_barrier(0);
        run_phase(v_s[1], v_s[0], kv_prev, kv_cur, t, 1_I);
        rotate();
        __builtin_amdgcn_sched_barrier(0);
    }

    // --- Epilogue: softmax tail + gemm1 of the last tile, whose V is already in v_v ---
    // The leading barrier stands in for the B1 a next phase would have had: every wave must be
    // done reading the last phase's P before it is overwritten. Keeping the V read and the PV
    // outside the parity branch keeps the 128-register v_o off scratch.
    lds_barrier();
    if(((tile_end - 1 - tile_begin) & 1) == 1)
        tail_and_publish(v_s[1]);
    else
        tail_and_publish(v_s[0]);
    stage_end();

    s_waitcnt_lgkmcnt(0_I);
    lds_barrier();
    read_p();
    s_waitcnt_lgkmcnt(0_I);
    stage_end();

    rescale_o();
    pin_o();
    pin_v();
    compute_pv(v_o_tiles, s_kv[0], no_co, std::false_type{});
    pin_o();
    __builtin_amdgcn_sched_barrier(0);
    static_for<T::GEMM1_E_N>([&](auto i) {
        constexpr int n = vector_traits<o_tile_t>::size();
        set_slice(v_o, v_o_tiles[i.value], number<i.value * n>{}, number<(i.value + 1) * n>{});
    });
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
    const int pv_h    = warp_id / T::PV_WAVES_D;
    const int pv_dh   = warp_id % T::PV_WAVES_D;

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

    // Only the causal specialization needs the diagonal.
    int causal_diagonal = 0;
    if constexpr(T::CAUSAL)
    {
        causal_diagonal = q_len_ptr_s - kv_ind_ptr_s +
                          __builtin_amdgcn_readfirstlane(kargs.kv_indptr[batch_idx + 1]) -
                          __builtin_amdgcn_readfirstlane(kargs.q_indptr[batch_idx + 1]);
    }

    // Per-tensor descale folded into the two places it can be a single scalar multiply: QK's
    // descale_q * descale_k rides the softmax temperature, and V's descale_k is applied once
    // on the finished O (the PV MFMA itself consumes raw fp8).
    const float descale_q = readfirstlane_f32(reinterpret_cast<const float*>(kargs.q_scale_ptr)[0]);
    const float descale_k =
        readfirstlane_f32(reinterpret_cast<const float*>(kargs.kv_scale_ptr)[0]);
    const float qk_scale = readfirstlane_f32(temperature_scale * descale_q * descale_k);

    const int q_gmem_offset = q_len_ptr_s * kargs.stride_q_b;
    auto g_q = make_gmem(reinterpret_cast<const D_Q*>(kargs.q_buffer_ptr) + q_gmem_offset,
                         q_len * kargs.stride_q_b * sizeof(D_Q));
    auto g_q_rope =
        make_gmem(reinterpret_cast<const D_Q*>(kargs.q_buffer_ptr) + q_gmem_offset + T::D_NOPE_SIZE,
                  q_len * kargs.stride_q_b * sizeof(D_Q));

    // Q as the five 32 B k-step operands: four from nope, then rope in the low half of the
    // fifth and the 640-padding zeroed in its high half.
    vector_t<D_Q, T::GEMM0_E_K * T::W_M * T::W_K / T::WARP_SIZE> v_q;
    clear(v_q);
    {
        auto v_q_nope = load<T::VEC_Q>(g_q, make_layout_q_nope<T>(warp_id, lane_id));
        auto v_q_rope = load<T::VEC_Q>(g_q_rope, make_layout_q_rope<T>(warp_id, lane_id));
        using q_vec_t = vector_t<D_Q, T::VEC_Q>;
        auto* dst     = reinterpret_cast<q_vec_t*>(&v_q);
        auto* nope    = reinterpret_cast<q_vec_t*>(&v_q_nope);
        constexpr int nope_halves = T::D_NOPE_SIZE / T::W_K_HALF; // 8
        static_for<nope_halves>([&](auto i) { dst[i.value] = nope[i.value]; });
        dst[nope_halves] = v_q_rope;
    }

    vector_t<D_ACC, T::PV_ROWS * T::PV_D / T::WARP_SIZE> v_o;
    clear(v_o);
    D_ACC m_row = opus::numeric_limits<D_ACC>::lowest();
    D_ACC l_row = 0.0f;
    mla_decode_fwd_pipelined<Traits>(kargs,
                                     kv_ind_ptr_s,
                                     valid_kv_len,
                                     0,
                                     num_kv_tiles,
                                     smem_buffer,
                                     v_q,
                                     v_o,
                                     m_row,
                                     l_row,
                                     qk_scale,
                                     causal_diagonal);

    // The row sums live with the GEMM0 waves; the O rows with the PV waves. One exchange per
    // work item, which is also the barrier that frees every KV slot for the next item.
    auto s_lsum = smem_at<D_ACC>(make_smem(smem_buffer), T::smem_lsum_offset);
    const lds_lane_addrs<T> la(warp_id, lane_id, pv_h);
    s_lsum.template _store<1>(vector_t<D_ACC, 1>{l_row}, la.row_wr);
    asm volatile("" ::: "memory");
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    asm volatile("" ::: "memory");
    const D_ACC l_pv = s_lsum.template _load<1>(la.row_rd)[0];

    // Softmax normalisation and the V descale in one multiply. l == 0 means every score was
    // masked, so O must be 0 rather than NaN.
    D_ACC o_scale = (l_pv > D_ACC(0.0f)) ? (descale_k / l_pv) : D_ACC(0.0f);
    scale_output_tile<T>(v_o, o_scale);
    pin_output_tile(v_o);

    if(slot < 0)
    {
        const int o_gmem_offset = q_len_ptr_s * kargs.stride_o_b;
        auto g_o                = make_gmem(reinterpret_cast<D_OUT*>(kargs.out_ptr) + o_gmem_offset,
                             q_len * kargs.stride_o_b * sizeof(D_OUT));
        auto u_o                = make_layout_o<T>(pv_h, pv_dh, lane_id, kargs.stride_o_h);
        auto v_o_out            = cast<D_OUT>(v_o);
        store<T::VEC_O>(g_o, v_o_out, u_o);
        // lse_ptr is null when the caller did not ask for LSE; lse_accum in the split-KV
        // branch is always allocated, so only this side needs the guard.
        if(kargs.lse_ptr != nullptr && lane_id < T::W_M)
        {
            const int lse_offset = q_len_ptr_s * kargs.H;
            auto g_lse           = make_gmem(reinterpret_cast<D_ACC*>(kargs.lse_ptr) + lse_offset,
                                   q_len * kargs.H * sizeof(D_ACC));
            constexpr float INV_LOG2_E = 0.69314718055994531f; // ln(2)
            const D_ACC lse = (l_row > D_ACC(0.0f)) ? ((m_row + log2f(l_row)) * INV_LOG2_E)
                                                    : opus::numeric_limits<D_ACC>::lowest();
            g_lse.store(lse, warp_id * T::Q_TILE_SIZE + lane_id);
        }
    }
    if(slot >= 0)
    {
        const int oa_offset = slot * kargs.stride_o_b;
        auto g_oa           = make_gmem(reinterpret_cast<D_ACC*>(kargs.o_accum) + oa_offset,
                              q_len * kargs.stride_o_b * sizeof(D_ACC));
        auto u_oa           = make_layout_o<T>(pv_h, pv_dh, lane_id, T::D_NOPE_SIZE);
        store<T::VEC_O>(g_oa, v_o, u_oa);

        if(lane_id < T::W_M)
        {
            const int lse_offset = slot * kargs.H;
            auto g_lse           = make_gmem(reinterpret_cast<D_ACC*>(kargs.lse_accum) + lse_offset,
                                   q_len * kargs.H * sizeof(D_ACC));
            constexpr float INV_LOG2_E = 0.69314718055994531f;
            const D_ACC lse = (l_row > D_ACC(0.0f)) ? ((m_row + log2f(l_row)) * INV_LOG2_E)
                                                    : opus::numeric_limits<D_ACC>::lowest();
            g_lse.store(lse, warp_id * T::Q_TILE_SIZE + lane_id);
        }
    }
}

} // namespace mla_decode_fwd_16mx4_64nx1_fp8fp8

// Persistent entry point: the grid is sized to the machine, not to the problem, and each
// block drains the work items the metadata kernel assigned it through work_indptr.
//
// One block per CU: 3 KV slots plus the P exchange are 130 KB. That is also what makes the
// register budget work -- one wave per SIMD gets the whole 512-register file, and this shape
// needs more than 256 (v_o 128, v_q 40, v_s 2x16, v_k 2x16, v_v 64, v_p 8).
template <class Traits>
__global__
__launch_bounds__(Traits::BLOCK_SIZE,
                  1) void opus_mla_decode_fp8_16mx4_64nx1_kernel(opus_mla_decode_fp8_kargs kargs)
{
    using namespace opus;
    using namespace mla_decode_fwd_16mx4_64nx1_fp8fp8;
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

#endif // !__HIP_DEVICE_COMPILE__ || !__gfx950__
