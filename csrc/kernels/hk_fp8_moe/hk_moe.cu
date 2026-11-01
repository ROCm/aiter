// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx950 block-FP8 MoE. Tensor layouts and API: aiter/ops/hk_fp8_moe.py.
#include "aiter_hip_common.h"

#include "aiter_ctypes_error.h"

#include <type_traits>

#include "kittens.cuh"

AITER_CTYPES_ERROR_DEF

using namespace kittens;

// Shuffled-weight buffer and per-wave LDS base.
struct precomputed_addresses
{
    i32x4 srsrc;
    uintptr_t lds_base;
};

// Pinned registers per wave: a0:a255 accumulators, v128:v255 operands, v0:v127 temporaries.
// Accumulator (I,J) starts at a[(2*I+J)*64]; I/J select 128-row/column blocks.
// ART indices >= 256 address AGPRs. amdgpu_num_vgpr(128) does not protect the pinned
// ranges with all AGPRs occupied: verify compiler output with test_hk_fp8_moe_codegen.py.
// Weights are MFMA operand A. Subtile (m,n) starts at base+(n*4+m)*4;
// lane l holds C[n*16+l%16][m*16+4*(l/16) .. +3].

namespace pin {
using namespace kittens;

constexpr int PA0 = 128, PA1 = 160, PB0 = 192, PB1 = 224;

template <int BASE>
using op_ranges =
    ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<BASE, BASE + 31>>, 8>;
template <int ACC>
using acc_ranges = ducks::art::transpose_2d<
    ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<256 + ACC, 319 + ACC>>, 4>,
    4,
    4>;

// Four FP8 subtiles in v[BASE..BASE+31].
template <int BASE>
using OpTile = art<fp8e4m3, 64, 128, ducks::rt_layout::row, rt_16x128_s, op_ranges<BASE>>;
// FP32 accumulators in a[ACC..ACC+63].
template <int ACC>
using AccTile = art<float, 64, 64, ducks::rt_layout::col, rt_16x16_s, acc_ranges<ACC>>;

__device__ __forceinline__ void clobber_all()
{
    ducks::art::clobber<op_ranges<PA0>>();
    ducks::art::clobber<op_ranges<PA1>>();
    ducks::art::clobber<op_ranges<PB0>>();
    ducks::art::clobber<op_ranges<PB1>>();
    ducks::art::clobber<acc_ranges<0>>();
    ducks::art::clobber<acc_ranges<64>>();
    ducks::art::clobber<acc_ranges<128>>();
    ducks::art::clobber<acc_ranges<192>>();
}

// MFMA needs two wait states after AGPR writes.
template <int ACC>
__device__ __forceinline__ void zero_acc()
{
    AccTile<ACC> c;
    zero(c);
}
__device__ __forceinline__ void zero_acc_all()
{
    AccTile<0> c00;
    AccTile<64> c01;
    AccTile<128> c10;
    AccTile<192> c11;
    zero(c00);
    zero(c01);
    zero(c10);
    zero(c11);
    asm volatile("s_nop 2" ::: "memory");
}

// Transfer accumulator subtile (n,m) to/from four floats.
template <int ACC, int n, int m>
__device__ __forceinline__ void acc_rd4(float (&x)[4])
{
    using range = ducks::art::get_nth_range_t<typename AccTile<ACC>::register_ranges, m * 4 + n>;
    x[0]        = macros::v_mov_b32_p2up<range::lo + 0, float>();
    x[1]        = macros::v_mov_b32_p2up<range::lo + 1, float>();
    x[2]        = macros::v_mov_b32_p2up<range::lo + 2, float>();
    x[3]        = macros::v_mov_b32_p2up<range::lo + 3, float>();
}
template <int ACC, int n, int m>
__device__ __forceinline__ void acc_wr4(const float (&x)[4])
{
    using range = ducks::art::get_nth_range_t<typename AccTile<ACC>::register_ranges, m * 4 + n>;
    macros::v_mov_b32_up2p<range::lo + 0>(x[0]);
    macros::v_mov_b32_up2p<range::lo + 1>(x[1]);
    macros::v_mov_b32_up2p<range::lo + 2>(x[2]);
    macros::v_mov_b32_up2p<range::lo + 3>(x[3]);
}

// Pair adjacent subtiles for 16-byte BF16 stores. Rows >= nv use out-of-range offsets:
// the buffer drops their writes, but the instructions still count toward vmcnt.
template <int ACC>
__device__ __forceinline__ void store_acc_rows(const i32x4 rsrc,
                                               const int ldc,
                                               const int row0,
                                               const int col0,
                                               const int q,
                                               const int lr,
                                               const int nv)
{
    [&]<int... Is>(std::integer_sequence<int, Is...>) {
        (
            [&]<int I>() {
                constexpr int n = I / 2, m = (I % 2) * 2;
                float x[4], y[4];
                acc_rd4<ACC, n, m>(x);
                acc_rd4<ACC, n, m + 1>(y);
                uint32_t X01, X23, Y01, Y23;
                asm("v_cvt_pk_bf16_f32 %0, %1, %2" : "=v"(X01) : "v"(x[0]), "v"(x[1]));
                asm("v_cvt_pk_bf16_f32 %0, %1, %2" : "=v"(X23) : "v"(x[2]), "v"(x[3]));
                asm("v_cvt_pk_bf16_f32 %0, %1, %2" : "=v"(Y01) : "v"(y[0]), "v"(y[1]));
                asm("v_cvt_pk_bf16_f32 %0, %1, %2" : "=v"(Y23) : "v"(y[2]), "v"(y[3]));
                auto s0            = __builtin_amdgcn_permlane16_swap(X01, Y01, false, false);
                auto s1            = __builtin_amdgcn_permlane16_swap(X23, Y23, false, false);
                const int col      = col0 + ((q & 1) ? (m + 1) * 16 + 4 * (q - 1) : m * 16 + 4 * q);
                const int row      = row0 + n * 16 + lr;
                const uint32_t off = row < nv ? (uint32_t)(row * ldc + col) * 2u : 0xffffffffu;
                const __uint128_t v = (__uint128_t)s0[0] | ((__uint128_t)s1[0] << 32) |
                                      ((__uint128_t)s0[1] << 64) | ((__uint128_t)s1[1] << 96);
                llvm_amdgcn_raw_buffer_store_b128(
                    v, rsrc, off, 0, static_cast<uint32_t>(coherency::cache_all));
            }.template operator()<Is>(),
            ...);
    }(std::make_integer_sequence<int, 8>{});
}

} // namespace pin

// Tile loads: four 16-byte transfers per thread fill a 128x128 FP8 LDS tile.
// Sources are contiguous subtiles, shuffled weights, or gathered token rows.

constexpr int NUM_WARPS = 4;
using G                 = kittens::group<NUM_WARPS>;

namespace pin {

template <typename ST, typename GL>
struct plain_tile_load
{
    ST& dst;
    const GL& src;
    coord<ST> idx;
    __device__ __forceinline__ void prepare() {}
    template <int i>
    __device__ __forceinline__ void issue()
    {
        static_assert(ST::rows == 128 && ST::cols == 128 && i >= 0 && i < 4);
        auto chunk = kittens::subtile_inplace<32, 128>(dst, {i, 0});
        G::load(chunk, src, {idx.b, idx.d, idx.r * 4 + i, idx.c});
    }
};

// shuffle_weight(w, (16,16)): [N/16][K/32][2][16 rows][16 bytes].
template <int K>
__device__ __forceinline__ uint32_t shuf_off(int row, int col)
{
    return ((row / 16) * (K / 32) + col / 32) * 512 + ((col / 16) % 2) * 256 + (row % 16) * 16 +
           (col % 16);
}
template <int K, typename ST, typename T>
__device__ __forceinline__ precomputed_addresses
precompute_addresses_shuf(ST& dst, const T* W, int rb, int kk)
{
    constexpr int bytes_per_warp = ST::underlying_subtile_bytes_per_thread * kittens::WARP_THREADS;
    const T* p                   = W + (size_t)rb * ST::rows * K + (size_t)kk * ST::cols * 16;
    return {make_srsrc((void*)p, ST::rows * K),
            reinterpret_cast<uintptr_t>(&dst.data[0]) + (warpid() * bytes_per_warp)};
}
template <int i, int K, typename ST>
__device__ inline void load_one_shuf(ST& dst, precomputed_addresses addresses)
{
    using T                        = typename ST::dtype;
    constexpr int num_warps        = NUM_WARPS;
    constexpr int bytes_per_thread = ST::underlying_subtile_bytes_per_thread;
    constexpr int bytes_per_warp   = bytes_per_thread * kittens::WARP_THREADS;
    const int lane_byte_offset     = (kittens::laneid() * bytes_per_thread) +
                                 (kittens::warpid() * bytes_per_warp) +
                                 (i * num_warps * bytes_per_warp);
    const int subtile_id               = lane_byte_offset / ST::underlying_subtile_bytes;
    const int subtile_row              = subtile_id / ST::underlying_subtiles_per_row;
    const int subtile_col              = subtile_id % ST::underlying_subtiles_per_row;
    const int subtile_lane_byte_offset = lane_byte_offset % ST::underlying_subtile_bytes;
    const int row     = subtile_lane_byte_offset / ST::underlying_subtile_row_bytes;
    const int col     = (subtile_lane_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T);
    const uint32_t sw = dst.swizzle({row, col});
    const int grow =
        (sw / ST::underlying_subtile_row_bytes) + subtile_row * ST::underlying_subtile_rows;
    const int gcol = (sw % ST::underlying_subtile_row_bytes) / sizeof(T) +
                     subtile_col * ST::underlying_subtile_cols;
    as3_uint32_ptr lds_ptr =
        (as3_uint32_ptr)(addresses.lds_base + (i * num_warps * bytes_per_warp));
    llvm_amdgcn_raw_buffer_load_lds(addresses.srsrc,
                                    lds_ptr,
                                    bytes_per_thread,
                                    shuf_off<K>(grow, gcol) * sizeof(T),
                                    0,
                                    0,
                                    static_cast<int>(coherency::cache_all));
}
template <int K, typename ST>
struct shuf_tile_load
{
    ST& dst;
    const fp8e4m3* w;
    int rb, kk;
    precomputed_addresses addr;
    __device__ __forceinline__ void prepare()
    {
        addr = precompute_addresses_shuf<K>(dst, w, rb, kk);
    }
    template <int i>
    __device__ __forceinline__ void issue()
    {
        load_one_shuf<i, K>(dst, addr);
    }
};
template <int K, typename ST>
__device__ __forceinline__ void load_tile_shuf(ST& dst, const fp8e4m3* w, int rb, int kk)
{
    shuf_tile_load<K, ST> ld{dst, w, rb, kk};
    ld.prepare();
    ld.template issue<0>();
    ld.template issue<1>();
    ld.template issue<2>();
    ld.template issue<3>();
}

// srsrc spans the source; off[i] is a row/column byte offset, soff the K-step offset.
struct gather_src
{
    i32x4 srsrc;
    uint32_t off[4];
    uint32_t soff;
};
template <int i, typename ST>
__device__ inline void load_one_gather(ST& dst, const gather_src& gs)
{
    constexpr int bytes_per_thread = ST::underlying_subtile_bytes_per_thread;
    constexpr int bytes_per_warp   = bytes_per_thread * kittens::WARP_THREADS;
    const uintptr_t lds_base =
        reinterpret_cast<uintptr_t>(&dst.data[0]) + (kittens::warpid() * bytes_per_warp);
    as3_uint32_ptr lds_ptr = (as3_uint32_ptr)(lds_base + (i * NUM_WARPS * bytes_per_warp));
    llvm_amdgcn_raw_buffer_load_lds(gs.srsrc,
                                    lds_ptr,
                                    bytes_per_thread,
                                    gs.off[i],
                                    gs.soff,
                                    0,
                                    static_cast<int>(coherency::cache_all));
}
template <typename ST>
struct gather_tile_load
{
    ST& dst;
    const gather_src& gs;
    __device__ __forceinline__ void prepare() {}
    template <int i>
    __device__ __forceinline__ void issue()
    {
        load_one_gather<i>(dst, gs);
    }
};
template <typename ST>
__device__ __forceinline__ void load_tile_gather(ST& dst, const gather_src& gs)
{
    gather_tile_load<ST> ld{dst, gs};
    ld.template issue<0>();
    ld.template issue<1>();
    ld.template issue<2>();
    ld.template issue<3>();
}

// Quarter K-step: 16 MFMAs interleaved with 4 global->LDS loads and 8 LDS reads.
// AR/BR are activation/weight operands, DST the next operand. Scales use bytes JSEL of pb, n of pa.
#define PIN_SMMA(n, m)                                                \
    do                                                                \
    {                                                                 \
        if constexpr(!SKIP)                                           \
            mma_ABt_scaled<m, n, JSEL, n>(acc, wt, act, acc, pb, pa); \
    } while(0)
// SKIP preserves loads but omits MFMAs for padding rows.
template <int JSEL, int ACC, int AR, int BR, int DST, bool SKIP = false, typename LOAD, typename ST>
__device__ __forceinline__ void
do_interleaved_cluster(LOAD ld, const ST& src, const uint32_t pa, const uint32_t pb)
{
    AccTile<ACC> acc;
    OpTile<AR> act;
    OpTile<BR> wt;
    OpTile<DST> dst;
    __builtin_amdgcn_sched_barrier(0);
    PIN_SMMA(0, 0);
    __builtin_amdgcn_sched_barrier(0);

    ld.prepare();

    __builtin_amdgcn_sched_barrier(0);
    PIN_SMMA(0, 1);
    __builtin_amdgcn_sched_barrier(0);

    const auto swizzled_offsets = get_address(dst, src);

    ld.template issue<0>();
    load<0, 0, 0>(dst, src, swizzled_offsets[0]);

    __builtin_amdgcn_sched_barrier(0);
    PIN_SMMA(0, 2);
    __builtin_amdgcn_sched_barrier(0);

    load<0, 0, 1>(dst, src, swizzled_offsets[1]);

    __builtin_amdgcn_sched_barrier(0);
    PIN_SMMA(0, 3);
    __builtin_amdgcn_sched_barrier(0);

    ld.template issue<1>();
    load<1, 0, 0>(dst, src, swizzled_offsets[0]);
    __builtin_amdgcn_sched_barrier(0);
    PIN_SMMA(1, 0);
    PIN_SMMA(1, 1);
    __builtin_amdgcn_sched_barrier(0);

    load<1, 0, 1>(dst, src, swizzled_offsets[1]);
    __builtin_amdgcn_sched_barrier(0);
    PIN_SMMA(1, 2);
    PIN_SMMA(1, 3);
    __builtin_amdgcn_sched_barrier(0);

    ld.template issue<2>();
    load<2, 0, 0>(dst, src, swizzled_offsets[0]);
    __builtin_amdgcn_sched_barrier(0);
    PIN_SMMA(2, 0);
    PIN_SMMA(2, 1);
    __builtin_amdgcn_sched_barrier(0);

    load<2, 0, 1>(dst, src, swizzled_offsets[1]);
    __builtin_amdgcn_sched_barrier(0);
    PIN_SMMA(2, 2);
    PIN_SMMA(2, 3);
    __builtin_amdgcn_sched_barrier(0);

    ld.template issue<3>();
    load<3, 0, 0>(dst, src, swizzled_offsets[0]);
    __builtin_amdgcn_sched_barrier(0);
    PIN_SMMA(3, 0);
    PIN_SMMA(3, 1);
    __builtin_amdgcn_sched_barrier(0);

    load<3, 0, 1>(dst, src, swizzled_offsets[1]);
    __builtin_amdgcn_sched_barrier(0);
    PIN_SMMA(3, 2);
    PIN_SMMA(3, 3);
    __builtin_amdgcn_sched_barrier(0);
}
#undef PIN_SMMA

} // namespace pin

// Persistent grouped GEMM: 256x256 tiles, four waves, expert rows padded to 256.
// tile_expert[n_mtiles] holds the used-tile count. Workgroups advance by gridDim.x;
// each tile's last two K-steps prefetch the next tile (the final prefetch is unused).
// Stage 1 gathers via AOFF, then applies SwiGLU and FP8 quantization into act_out/act_pa.
// Stage 2 reads contiguous activations and writes unweighted BF16 Cout[rows][NTOT].
// W[E][NTOT][K] uses shuffle_weight(w, (16,16)). Packed activation scales:
// PA[K/128][n_mtiles][2][2][16]. SB[E][NTOT/128][K/128] stores FP32 power-of-two
// weight scales; V_MFMA_SCALE consumes their exponent bytes.

constexpr int MOE_ROWS_MAX = 1 << 21; // Layout extent; only strides are used.

namespace pin {
__device__ __forceinline__ float lane_max16(float m)
{ // Max across four 16-lane rows.
    auto s = __builtin_amdgcn_permlane16_swap(__float_as_uint(m), __float_as_uint(m), false, false);
    m      = fmaxf(__uint_as_float(s[0]), __uint_as_float(s[1]));
    auto t = __builtin_amdgcn_permlane32_swap(__float_as_uint(m), __float_as_uint(m), false, false);
    return fmaxf(__uint_as_float(t[0]), __uint_as_float(t[1]));
}

// Pass 1 overwrites gate accumulators with SwiGLU and records row maxima in LDS.
// Pass 2 quantizes and packs scales: byte n of [nb0][br][I][wm][lr] describes
// row I*128 + wm*64 + n*16 + lr.
template <int NTOT>
__device__ __forceinline__ void epilogue_silu_quant(const int br,
                                                    const int nb0,
                                                    const int wm,
                                                    const int wn,
                                                    const int q,
                                                    const int lr,
                                                    const float L,
                                                    fp8e4m3* __restrict__ act_out,
                                                    uint32_t* __restrict__ act_pa,
                                                    const int n_mtiles)
{
    constexpr int WARPS_COL = 2, BLOCK_SIZE_ROW = 256, ACT_N = NTOT / 2;
    __shared__ float amax_s[WARPS_COL][BLOCK_SIZE_ROW];
    // Clamped SwiGLU on gate c[I][0] and up c[I][1], subtile (n,m).
    auto act4 = [&]<int I, int n, int m>(float(&o)[4]) {
        float gg[4], uu[4];
        acc_rd4<I * 128, n, m>(gg);
        acc_rd4<I * 128 + 64, n, m>(uu);
#pragma unroll
        for(int r = 0; r < 4; r++)
        {
            float x = gg[r], u = uu[r];
            if(L > 0.f)
            {
                x = fminf(x, L);
                u = __builtin_amdgcn_fmed3f(u, -L, L);
            }
            o[r] = x *
                   __builtin_amdgcn_rcpf(1.f + __builtin_amdgcn_exp2f(-1.4426950408889634f * x)) *
                   u;
        }
    };
    [&]<int... Is>(std::integer_sequence<int, Is...>) {
        (
            [&]<int IN>() {
                constexpr int I = IN / 4, n = IN % 4;
                float mx = 0.f;
                [&]<int... Ms>(std::integer_sequence<int, Ms...>) {
                    (
                        [&]<int m>() {
                            float o[4];
                            act4.template operator()<I, n, m>(o);
                            for(int r = 0; r < 4; r++)
                                mx = fmaxf(mx, fabsf(o[r]));
                            acc_wr4<I * 128, n, m>(o);
                        }.template operator()<Ms>(),
                        ...);
                }(std::make_integer_sequence<int, 4>{});
                mx = lane_max16(mx);
                if(q == 0)
                    amax_s[wn][I * 128 + wm * 64 + n * 16 + lr] = mx;
                __builtin_amdgcn_sched_barrier(0);
            }.template operator()<Is>(),
            ...);
    }(std::make_integer_sequence<int, 8>{});
    asm volatile("s_waitcnt lgkmcnt(0)\n s_barrier" ::: "memory");
    // Pair subtiles with permlane16_swap for 8-byte stores.
    const int colw = nb0 * 128 + wn * 64;
    uint32_t pw    = 0;
    [&]<int... Is>(std::integer_sequence<int, Is...>) {
        (
            [&]<int IN>() {
                constexpr int I = IN / 4, n = IN % 4;
                const int rb    = I * 128 + wm * 64 + n * 16 + lr;
                const float sc  = exp2f(ceilf(
                                log2f(fmaxf(fmaxf(amax_s[0][rb], amax_s[1][rb]), 1e-10f) / 448.f))),
                            inv = 1.f / sc;
                const size_t r  = (size_t)br * BLOCK_SIZE_ROW + rb;
                uint32_t d[4];
                [&]<int... Ms>(std::integer_sequence<int, Ms...>) {
                    (
                        [&]<int m>() {
                            float o[4];
                            acc_rd4<I * 128, n, m>(o);
                            int w =
                                __builtin_amdgcn_cvt_pk_fp8_f32(o[0] * inv, o[1] * inv, 0, false);
                            w    = __builtin_amdgcn_cvt_pk_fp8_f32(o[2] * inv, o[3] * inv, w, true);
                            d[m] = (uint32_t)w;
                        }.template operator()<Ms>(),
                        ...);
                }(std::make_integer_sequence<int, 4>{});
#pragma unroll
                for(int mp = 0; mp < 2; mp++)
                {
                    const int m   = mp * 2;
                    auto sw       = __builtin_amdgcn_permlane16_swap(d[m], d[m + 1], false, false);
                    const int col = colw + ((q & 1) ? (m + 1) * 16 + 4 * (q - 1) : m * 16 + 4 * q);
                    uint2 v;
                    v.x                                                  = sw[0];
                    v.y                                                  = sw[1];
                    *reinterpret_cast<uint2*>(act_out + r * ACT_N + col) = v;
                }
                pw |= ((__float_as_uint(sc) >> 23) & 0xffu) << (8 * n);
                if constexpr(n == 3)
                {
                    if(wn == 0 && q == 0)
                        act_pa[(((size_t)nb0 * n_mtiles + br) * 2 + I) * 32 + wm * 16 + lr] = pw;
                    pw = 0;
                }
                __builtin_amdgcn_sched_barrier(0);
            }.template operator()<Is>(),
            ...);
    }(std::make_integer_sequence<int, 8>{});
}
} // namespace pin

// HALF_SKIP omits MFMAs for an empty second half; use only for small expert batches.
template <int NTOT, int K, bool GATE_UP, bool HALF_SKIP = false>
__global__ __launch_bounds__(256, 1) __attribute__((amdgpu_num_vgpr(128))) void moe_gemm_kernel(
    const kittens::gl<fp8e4m3, 1, 1, MOE_ROWS_MAX, K> A,
    const kittens::gl<fp8e4m3, 1, 1, NTOT, K> W0,
    bf16* __restrict__ Cout,
    const uint32_t* __restrict__ PA,
    const float* __restrict__ SB,
    const int* __restrict__ tile_expert,
    const int n_mtiles,
    fp8e4m3* __restrict__ act_out,
    uint32_t* __restrict__ act_pa,
    const float swiglu_limit,
    const uint32_t* __restrict__ AOFF,
    const int* __restrict__ ecnt,
    const int* __restrict__ eoff)
{
    using namespace pin;
    constexpr int WARPS_COL = 2, WARPS_ROW = 2, BLOCK_SIZE_ROW = 256, BLOCK_SIZE_COL = 256,
                  BLOCK_K = 128, k_step = BLOCK_K;
    constexpr int blocks_col = NTOT / BLOCK_SIZE_COL;
    constexpr int k_iters    = K / BLOCK_K;
    static_assert(k_iters >= 4, "pipeline needs >= 4 K-steps per tile");
    // vmcnt: 16 tile loads + 1 scale load; tile entry also counts prior epilogue stores.
    constexpr int VS     = 17;
    constexpr int NSTORE = GATE_UP ? 16 : 32;
    constexpr int VW     = VS + NSTORE;
    static_assert(VW <= 63, "vmcnt field");
    using GLA = kittens::gl<fp8e4m3, 1, 1, MOE_ROWS_MAX, K>;

    using ST_A = st_fp8e4m3<BLOCK_SIZE_ROW / 2, BLOCK_K, st_16x128_s>;
    using ST_B = st_fp8e4m3<BLOCK_SIZE_COL / 2, BLOCK_K, st_16x128_s>;

    __shared__ ST_A As[2][2];
    __shared__ ST_B Bs[2][2];
    // Packed A scales: one direct-to-LDS load per wave per K-step.
    __shared__ uint32_t sring[4][NUM_WARPS][64];

    clobber_all();

    typedef const __attribute__((address_space(4))) int* cint_p;
    typedef const __attribute__((address_space(4))) uint32_t* cu32_p;
    const cint_p TE  = (cint_p)tile_expert; // scalar loads
    const cu32_p SBc = (cu32_p)SB;
    const int total  = TE[n_mtiles] * blocks_col;
    const int G      = gridDim.x;
    if((int)blockIdx.x >= total)
        return;

    // Map work to (row, column); groups of eight reuse a weight column.
    auto map = [&](int s, int& r, int& c) {
        const int b  = s % G;
        const int wg = (s - b + G <= total && G % 8 == 0) ? s - b + (b % 8) * (G / 8) + b / 8 : s;
        r            = wg / blocks_col;
        c            = wg % blocks_col;
    };
    // Stage 1 pairs gate block c with up block c + NTOT/256.
    auto nb_of = [](int c, int j) { return GATE_UP ? c + j * (NTOT / 256) : 2 * c + j; };

    int warp_m          = (warpid() / WARPS_COL);
    int warp_n          = (warpid() % WARPS_COL);
    constexpr int KB    = K / 128;
    const int NBM       = n_mtiles;
    const i32x4 pa_rsrc = make_srsrc((void*)PA, (uint32_t)((size_t)KB * NBM * 2 * 32 * 4));
    auto issue_ps       = [&](int gg, int kk, int row) {
        as3_uint32_ptr lds =
            (as3_uint32_ptr)(reinterpret_cast<uintptr_t>(&sring[gg & 3][warpid()][0]));
        llvm_amdgcn_raw_buffer_load_lds(pa_rsrc,
                                        lds,
                                        4,
                                        (uint32_t)laneid() * 4,
                                        (uint32_t)((kk * NBM + row) * 2 * 32 * 4),
                                        0,
                                        static_cast<int>(coherency::cache_all));
    };
    auto pa_of = [&](int gg, int I) {
        return sring[gg & 3][warpid()][I * 32 + warp_m * 16 + laneid() % 16];
    };
    // Pack weight-block exponents into bytes 0 and 1.
    auto pb_raw = [&](int e, int c, int kk) {
        const uint32_t b0 = SBc[((size_t)e * (NTOT / 128) + nb_of(c, 0)) * KB + kk],
                       b1 = SBc[((size_t)e * (NTOT / 128) + nb_of(c, 1)) * KB + kk];
        return ((b0 >> 23) & 0xffu) | (((b1 >> 23) & 0xffu) << 8);
    };
    auto to_vgpr = [](uint32_t v) {
        asm volatile("" : "+v"(v));
        return v;
    };
    fp8e4m3* const wbase = W0.raw_ptr;
    auto wptr            = [&](int e) { return wbase + (size_t)e * NTOT * K; };

    int curr = 0, next = 1, g = 0;
    int s = blockIdx.x, br, bc;
    map(s, br, bc);
    int ex = TE[br];
    // Skip empty-half MFMAs, retaining loads for unchanged vmcnt accounting.
    auto half_empty = [&](int r, int e) {
        return HALF_SKIP &&
               ((cint_p)ecnt)[e] - (r * BLOCK_SIZE_ROW - ((cint_p)eoff)[e]) <= BLOCK_SIZE_ROW / 2;
    };
    bool half1_empty = half_empty(br, ex);
    int nbr_ = br, nbc = bc, nex = ex;
    bool has_next = false;

    // Stage-1 gather offsets; prefetch the next tile's offsets four K-steps early.
    uint32_t aoc[8], aon[8];
    const i32x4 q_rsrc = make_srsrc((void*)A.raw_ptr, 0x7ffffff0u);
    auto load_aoff     = [&](int row_tile, uint32_t(&o)[8]) {
        const uint4* p =
            reinterpret_cast<const uint4*>(AOFF + ((size_t)row_tile * 256 + threadIdx.x) * 8);
        const uint4 x = p[0], y = p[1];
        o[0] = x.x;
        o[1] = x.y;
        o[2] = x.z;
        o[3] = x.w;
        o[4] = y.x;
        o[5] = y.y;
        o[6] = y.z;
        o[7] = y.w;
    };
    auto gsrc = [&](const uint32_t(&o)[8], int h, int kk) {
        gather_src gs;
        gs.srsrc = q_rsrc;
        gs.soff  = kk * 128;
#pragma unroll
        for(int i = 0; i < 4; i++)
            gs.off[i] = o[h * 4 + i];
        return gs;
    };
    // Load 128-row half h at K-step kk.
    auto load_a_tile = [&](ST_A& dst, int h, int r, int kk) {
        if constexpr(GATE_UP)
            load_tile_gather(dst, gsrc(aoc, h, kk));
        else
            G::load(dst, A, {0, 0, r * WARPS_ROW + h, kk});
    };
    if constexpr(GATE_UP)
        load_aoff(br, aoc);

    { // prologue (first tile only)
        const fp8e4m3* const wb = wptr(ex);
        __builtin_amdgcn_sched_barrier(0);
        issue_ps(0, 0, br);
        issue_ps(1, 1, br);
        load_a_tile(As[curr][0], 0, br, 0);
        load_tile_shuf<K>(Bs[curr][0], wb, nb_of(bc, 0), 0);
        load_tile_shuf<K>(Bs[curr][1], wb, nb_of(bc, 1), 0);
        load_a_tile(As[curr][1], 1, br, 0);

        zero_acc_all();

        load_a_tile(As[next][0], 0, br, 1);
        load_tile_shuf<K>(Bs[next][0], wb, nb_of(bc, 0), 1);
        load_tile_shuf<K>(Bs[next][1], wb, nb_of(bc, 1), 1);
        load_a_tile(As[next][1], 1, br, 1);
        issue_ps(2, 2, br);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(29)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        auto a_subtile_0 = kittens::subtile_inplace<BLOCK_SIZE_ROW / 2 / WARPS_ROW, k_step>(
            As[curr][0], {warp_m, 0});
        OpTile<PA0> a0_tile;
        load(a0_tile, a_subtile_0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(25)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        auto b_subtile_0 = kittens::subtile_inplace<BLOCK_SIZE_COL / 2 / WARPS_COL, k_step>(
            Bs[curr][0], {warp_n, 0});
        OpTile<PB0> b0_tile;
        load(b0_tile, b_subtile_0);
    }

    uint32_t pbr = pb_raw(ex, bc, 0);

    // Prefetch K-step k+2 across tile boundaries. Stage 2 overlaps prior output stores
    // with the next tile's first step, storing each accumulator before reusing it.
    i32x4 p_crs{};
    int p_nv = 0, p_col0 = 0, p_col1 = 0;
    auto epi_part = [&]<int ACC>() {
        if constexpr(!GATE_UP)
        {
            int wm = warpid() / WARPS_COL, l = laneid();
            asm volatile("" : "+v"(wm), "+v"(l));
            const int row0 = wm * 64 + (ACC >= 128 ? 128 : 0), col = (ACC % 128) ? p_col1 : p_col0;
            store_acc_rows<ACC>(p_crs, NTOT, row0, col, l / 16, l % 16, p_nv);
            zero_acc<ACC>();
            asm volatile("s_nop 2" ::: "memory");
            __builtin_amdgcn_sched_barrier(0);
        }
    };
    auto step = [&]<int VT, int VM, bool EPI>(int k) {
        const bool w1 = k + 1 >= k_iters, w2 = k + 2 >= k_iters;
        const uint32_t nbr = pb_raw(w1 ? nex : ex, w1 ? nbc : bc, w1 ? k + 1 - k_iters : k + 1);
        const int r2 = w2 ? nbr_ : br, c2 = w2 ? nbc : bc, k2 = w2 ? k + 2 - k_iters : k + 2;
        const fp8e4m3* const wb2 = wptr(w2 ? nex : ex);
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(%0)" ::"n"(VT));
        const uint32_t pa0 = pa_of(g, 0), pa1 = pa_of(g, 1);
        asm volatile("s_waitcnt lgkmcnt(0)");
        const uint32_t pb = to_vgpr(pbr);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        auto bs_subtile_1 = kittens::subtile_inplace<BLOCK_SIZE_COL / 2 / WARPS_COL, k_step>(
            Bs[curr][1], {warp_n, 0});
        gather_src ga0, ga1;
        if constexpr(GATE_UP)
        {
            // Prevent offset calculations from being hoisted across K-steps.
            asm volatile(""
                         : "+v"(aoc[0]),
                           "+v"(aoc[1]),
                           "+v"(aoc[2]),
                           "+v"(aoc[3]),
                           "+v"(aoc[4]),
                           "+v"(aoc[5]),
                           "+v"(aoc[6]),
                           "+v"(aoc[7]));
            if(k == k_iters - 4)
                load_aoff(nbr_, aon);
            ga0 = w2 ? gsrc(aon, 0, k2) : gsrc(aoc, 0, k2);
            ga1 = w2 ? gsrc(aon, 1, k2) : gsrc(aoc, 1, k2);
        }
        auto a_load = [&](ST_A& dst, int h, const gather_src& ga) {
            if constexpr(GATE_UP)
                return gather_tile_load<ST_A>{dst, ga};
            else
                return plain_tile_load<ST_A, GLA>{dst, A, {0, 0, r2 * WARPS_ROW + h, k2}};
        };
        if constexpr(EPI)
            epi_part.template operator()<0>();
        do_interleaved_cluster<0, 0, PA0, PB0, PB1>(
            shuf_tile_load<K, ST_B>{Bs[curr][0], wb2, nb_of(c2, 0), k2}, bs_subtile_1, pa0, pb);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto a_subtile_1 = kittens::subtile_inplace<BLOCK_SIZE_ROW / 2 / WARPS_ROW, k_step>(
            As[curr][1], {warp_m, 0});
        if constexpr(EPI)
            epi_part.template operator()<64>();
        do_interleaved_cluster<1, 64, PA0, PB1, PA1>(
            a_load(As[curr][0], 0, ga0), a_subtile_1, pa0, pb);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(%0)" ::"n"(VM));
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto a_subtile_0 = kittens::subtile_inplace<BLOCK_SIZE_ROW / 2 / WARPS_ROW, k_step>(
            As[next][0], {warp_m, 0});
        if constexpr(EPI)
            epi_part.template operator()<128>();
        {
            const auto ldb1 = shuf_tile_load<K, ST_B>{Bs[curr][1], wb2, nb_of(c2, 1), k2};
            if(HALF_SKIP && half1_empty)
                do_interleaved_cluster<0, 128, PA1, PB0, PA0, HALF_SKIP>(
                    ldb1, a_subtile_0, pa1, pb);
            else
                do_interleaved_cluster<0, 128, PA1, PB0, PA0>(ldb1, a_subtile_0, pa1, pb);
        }

        auto b_subtile_0 = kittens::subtile_inplace<BLOCK_SIZE_COL / 2 / WARPS_COL, k_step>(
            Bs[next][0], {warp_n, 0});
        if constexpr(EPI)
            epi_part.template operator()<192>();
        if(HALF_SKIP && half1_empty)
            do_interleaved_cluster<1, 192, PA1, PB1, PB0, HALF_SKIP>(
                a_load(As[curr][1], 1, ga1), b_subtile_0, pa1, pb);
        else
            do_interleaved_cluster<1, 192, PA1, PB1, PB0>(
                a_load(As[curr][1], 1, ga1), b_subtile_0, pa1, pb);
        if(k + 3 < k_iters)
            issue_ps(g + 3, k + 3, br);
        else
            issue_ps(g + 3, k + 3 - k_iters, nbr_);
        pbr = nbr;
        curr ^= 1;
        next ^= 1;
        ++g;
    };

    bool first = true;
    for(;;)
    {
        const int sn = s + G;
        has_next     = sn < total;
        nbr_         = br;
        nbc          = bc;
        nex          = ex; // last tile: dead prefetches of this same tile
        if(has_next)
        {
            map(sn, nbr_, nbc);
            nex = TE[nbr_];
        }

        if(first)
        {
            step.template operator()<VS, VS, false>(0);
            step.template operator()<VS, VS, false>(1);
        }
        else if constexpr(GATE_UP)
        {
            step.template operator()<VW, VW, false>(0);
            step.template operator()<VW, VS, false>(1);
        }
        else
        {
            // Step 0: no prior stores at the first wait; 16 at the second.
            step.template operator()<VS, VS + NSTORE / 2, true>(0);
            step.template operator()<VW, VS, false>(1);
        }
#pragma clang loop unroll(full)
        for(int k = 2; k < k_iters; ++k)
            step.template operator()<VS, VS, false>(k);

        {
            __builtin_amdgcn_sched_barrier(0);
            int wm = warpid() / WARPS_COL, wn = warpid() % WARPS_COL, l = laneid();
            asm volatile("" : "+v"(wm), "+v"(wn), "+v"(l));
            const int q = l / 16, lr = l % 16;
            asm volatile("s_nop 7\n s_nop 7\n s_nop 7" ::
                             : "memory"); // XDL write -> accvgpr read: >= 20 wait states
            if constexpr(GATE_UP)
            {
                epilogue_silu_quant<NTOT>(
                    br, nb_of(bc, 0), wm, wn, q, lr, swiglu_limit, act_out, act_pa, n_mtiles);
            }
            else if(has_next)
            {
                // deferred to the next tile's first K-step (epi_part)
                p_nv   = ((cint_p)ecnt)[ex] - (br * BLOCK_SIZE_ROW - ((cint_p)eoff)[ex]);
                p_crs  = make_srsrc(Cout + (size_t)br * BLOCK_SIZE_ROW * NTOT,
                                   BLOCK_SIZE_ROW * NTOT * 2);
                p_col0 = nb_of(bc, 0) * 128 + wn * 64;
                p_col1 = nb_of(bc, 1) * 128 + wn * 64;
            }
            else
            {
                // Exclude padding rows from output stores.
                const int nv    = ((cint_p)ecnt)[ex] - (br * BLOCK_SIZE_ROW - ((cint_p)eoff)[ex]);
                const i32x4 crs = make_srsrc(Cout + (size_t)br * BLOCK_SIZE_ROW * NTOT,
                                             BLOCK_SIZE_ROW * NTOT * 2);
                const int row0  = wm * 64;
                store_acc_rows<0>(crs, NTOT, row0, nb_of(bc, 0) * 128 + wn * 64, q, lr, nv);
                store_acc_rows<64>(crs, NTOT, row0, nb_of(bc, 1) * 128 + wn * 64, q, lr, nv);
                store_acc_rows<128>(crs, NTOT, row0 + 128, nb_of(bc, 0) * 128 + wn * 64, q, lr, nv);
                store_acc_rows<192>(crs, NTOT, row0 + 128, nb_of(bc, 1) * 128 + wn * 64, q, lr, nv);
            }
            __builtin_amdgcn_sched_barrier(0);
        }
        if(!has_next)
            break;
        if constexpr(GATE_UP)
            zero_acc_all();
        if constexpr(GATE_UP)
        {
#pragma unroll
            for(int i = 0; i < 8; i++)
                aoc[i] = aon[i];
        }
        br          = nbr_;
        bc          = nbc;
        ex          = nex;
        s           = sn;
        first       = false;
        half1_empty = half_empty(br, ex);
    }
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory"); // Drain unused LDS prefetches before exit.
}

// Routing: expert e owns [off[e], off[e]+roundup(cnt[e],TM)).
// tok[r]=-1 marks padding; pos[t*S+j] maps a selection to its sorted row.
// Invalid expert IDs map to expert 0 with zero weight.

constexpr int SORT_EMAX = 1024, SORT_PER_BLOCK = 4096;
constexpr int SORT_THREADS = 1024, SORT_WAVES = SORT_THREADS / 64;
// 304 experts keep striped LDS below 40 KiB: four allocations per CU.
constexpr int SORT_STRIPED_EMAX = 304;
static_assert(SORT_PER_BLOCK % SORT_THREADS == 0 && SORT_PER_BLOCK < 65536);
typedef unsigned u32x4 __attribute__((ext_vector_type(4)));

// Lane-striped histogram; two 16-bit counts per word cannot overflow within a block.
__device__ __forceinline__ void
moe_count_block(int blk, const int* __restrict__ tk_e, int n, int E, int* __restrict__ cnt)
{
    if(E > SORT_STRIPED_EMAX)
    {
        // Larger expert sets exceed the LDS budget.
        const int b = blk * SORT_PER_BLOCK;
        for(int i = b + threadIdx.x; i < min(n, b + SORT_PER_BLOCK); i += blockDim.x)
        {
            const int e = tk_e[i];
            atomicAdd(&cnt[(unsigned)e < (unsigned)E ? e : 0], 1);
        }
        return;
    }
    __shared__ unsigned hist[SORT_STRIPED_EMAX / 2][64];
    __shared__ unsigned totals[SORT_STRIPED_EMAX];
    const int pairs = (E + 1) / 2, lane = threadIdx.x % 64, wave = threadIdx.x / 64;
    for(int i = threadIdx.x; i < pairs * 64; i += blockDim.x)
        (&hist[0][0])[i] = 0;
    __syncthreads();
    const int b = blk * SORT_PER_BLOCK;
    for(int i = b + threadIdx.x; i < min(n, b + SORT_PER_BLOCK); i += blockDim.x)
    {
        const int raw = tk_e[i], e = (unsigned)raw < (unsigned)E ? raw : 0;
        atomicAdd(&hist[e / 2][lane], 1u << ((e & 1) * 16));
    }
    __syncthreads();
    for(int p = wave; p < pairs; p += SORT_WAVES)
    {
        unsigned counts = hist[p][lane];
#pragma unroll
        for(int d = 1; d < 64; d <<= 1)
            counts += __shfl_xor(counts, d);
        if(lane == 0)
        {
            totals[p * 2] = counts & 0xffffu;
            if(p * 2 + 1 < E)
                totals[p * 2 + 1] = counts >> 16;
        }
    }
    __syncthreads();
    for(int e = threadIdx.x; e < E; e += blockDim.x)
        if(totals[e])
            atomicAdd(&cnt[e], totals[e]);
}

// One thread per expert: assign padded offsets and tile IDs, initialize padding.
// tile_e[max_tiles] stores the used-tile count; unused tile IDs are -1.
template <int TM>
__global__ void moe_scan(const int* __restrict__ cnt,
                         int E,
                         int* __restrict__ off,
                         int* __restrict__ cursor,
                         int* __restrict__ tile_e,
                         int max_tiles,
                         int* __restrict__ tok,
                         float* __restrict__ w)
{
    __shared__ int s_off[SORT_EMAX + 1];
    const int e = threadIdx.x;
    const int c = e < E ? cnt[e] : 0, pc = (c + TM - 1) / TM * TM;
    s_off[e + 1] = pc;
    if(e == 0)
        s_off[0] = 0;
    __syncthreads();
    for(int d = 1; d < SORT_EMAX; d <<= 1)
    { // inclusive scan of s_off[1..]
        const int v = e >= d ? s_off[e + 1 - d] : 0;
        __syncthreads();
        s_off[e + 1] += v;
        __syncthreads();
    }
    if(e < E)
    {
        off[e]    = s_off[e];
        cursor[e] = s_off[e];
    }
    if(e == 0)
        tile_e[max_tiles] = s_off[E] / TM;
    __syncthreads();
    // Wave i pads experts i, i+16, ...; one row per lane.
    for(int x = threadIdx.x / 64; x < E; x += SORT_EMAX / 64)
    {
        const int cx = cnt[x], end = s_off[x + 1];
        for(int r = s_off[x] + cx + threadIdx.x % 64; r < end; r += 64)
        {
            tok[r] = -1;
            w[r]   = 0.f;
        }
        // Fill this expert's tile IDs directly.
        for(int t = s_off[x] / TM + threadIdx.x % 64; t < end / TM; t += 64)
            tile_e[t] = x;
    }
    for(int t = threadIdx.x; t < max_tiles; t += blockDim.x)
    {
        if(t >= s_off[E] / TM)
            tile_e[t] = -1;
    }
}

__global__ void moe_scatter(const int* __restrict__ tk_e,
                            const float* __restrict__ tk_w,
                            int n,
                            int E,
                            int slots,
                            int* __restrict__ cursor,
                            int* __restrict__ tok,
                            float* __restrict__ w,
                            int* __restrict__ pos)
{
    if(E > SORT_STRIPED_EMAX)
    {
        const int b = blockIdx.x * SORT_PER_BLOCK;
        for(int i = b + threadIdx.x; i < min(n, b + SORT_PER_BLOCK); i += blockDim.x)
        {
            const int e   = tk_e[i];
            const bool ok = (unsigned)e < (unsigned)E;
            const int r   = atomicAdd(&cursor[ok ? e : 0], 1);
            tok[r]        = i / slots;
            w[r]          = ok ? tk_w[i] : 0.f;
            pos[i]        = r;
        }
        return;
    }
    constexpr int PER_T = SORT_PER_BLOCK / SORT_THREADS;
    // Keep packed counts separate from the full offsets to avoid batch barriers.
    __shared__ unsigned striped[SORT_STRIPED_EMAX][64];
    __shared__ unsigned hist[SORT_STRIPED_EMAX / 2][64];
    __shared__ unsigned totals[SORT_STRIPED_EMAX];
    const int pairs = (E + 1) / 2, lane = threadIdx.x % 64, wave = threadIdx.x / 64;
    for(int i = threadIdx.x; i < pairs * 64; i += blockDim.x)
        (&hist[0][0])[i] = 0;
    __syncthreads();
    const int b = blockIdx.x * SORT_PER_BLOCK;
    int local[PER_T];
#pragma unroll
    for(int j = 0; j < PER_T; ++j)
    {
        const int i   = b + j * SORT_THREADS + threadIdx.x;
        const int raw = i < n ? tk_e[i] : 0;
        const int e = (unsigned)raw < (unsigned)E ? raw : 0, shift = (e & 1) * 16;
        const unsigned old = i < n ? atomicAdd(&hist[e / 2][lane], 1u << shift) : 0;
        local[j]           = (old >> shift) & 0xffffu;
    }
    __syncthreads();
    // Per-expert lane prefixes; packed halves cannot borrow or carry.
    for(int p = wave; p < pairs; p += SORT_WAVES)
    {
        const unsigned own = hist[p][lane];
        unsigned inclusive = own;
#pragma unroll
        for(int d = 1; d < 64; d <<= 1)
        {
            const unsigned prev = __shfl_up(inclusive, d);
            if(lane >= d)
                inclusive += prev;
        }
        const unsigned sum = __shfl(inclusive, 63), prefix = inclusive - own;
        if(lane == 0)
        {
            totals[p * 2] = sum & 0xffffu;
            if(p * 2 + 1 < E)
                totals[p * 2 + 1] = sum >> 16;
        }
        striped[p * 2][lane] = prefix & 0xffffu;
        if(p * 2 + 1 < E)
            striped[p * 2 + 1][lane] = prefix >> 16;
    }
    __syncthreads();
    // Distribute expert reservations across lanes.
    for(int e = threadIdx.x; e < E; e += blockDim.x)
    {
        const unsigned count = totals[e];
        totals[e]            = count ? atomicAdd(&cursor[e], count) : 0;
    }
    __syncthreads();
    // Broadcast bases into lane-striped offsets for bank-conflict-free lookups.
    for(int e = wave; e < E; e += SORT_WAVES)
        striped[e][lane] += totals[e];
    __syncthreads();
#pragma unroll
    for(int j = 0; j < PER_T; ++j)
    {
        const int i = b + j * SORT_THREADS + threadIdx.x;
        if(i >= n)
            continue;
        const int raw = tk_e[i];
        const bool ok = (unsigned)raw < (unsigned)E;
        const int r   = striped[ok ? raw : 0][lane] + local[j];
        tok[r]        = i / slots;
        w[r]          = ok ? tk_w[i] : 0.f;
        pos[i]        = r;
    }
}

// Weighted sum of each token's S expert outputs.
template <int D, int S>
__global__ void moe_reduce_w(const bf16* __restrict__ cw,
                             const int* __restrict__ pos,
                             const float* __restrict__ row_w,
                             bf16* __restrict__ out)
{
    const int t = blockIdx.x, c = threadIdx.x * 8;
    float acc[8] = {0};
#pragma unroll
    for(int j = 0; j < S; j++)
    {
        const int r   = pos[t * S + j];
        const float w = row_w[r];
        // One-use expert outputs: non-temporal loads.
        const u32x4 vv =
            __builtin_nontemporal_load(reinterpret_cast<const u32x4*>(cw + (size_t)r * D + c));
        const uint4 v = {vv.x, vv.y, vv.z, vv.w};
        const bf16* b = reinterpret_cast<const bf16*>(&v);
#pragma unroll
        for(int i = 0; i < 8; i++)
            acc[i] += w * float(b[i]);
    }
    uint4 o;
    bf16* ob = reinterpret_cast<bf16*>(&o);
#pragma unroll
    for(int i = 0; i < 8; i++)
        ob[i] = bf16(acc[i]);
    *reinterpret_cast<uint4*>(out + (size_t)t * D + c) = o;
}

// Activation quantization and gather tables.

// Power-of-two 1x128 FP8 quantization: q[T][D], scales s[D/128][T].
// One wave per token, 16 elements per lane, eight lanes per scale block.
template <int D>
__device__ __forceinline__ void quant_tok16_block(
    int blk, const bf16* __restrict__ x, int T, fp8e4m3* __restrict__ q, float* __restrict__ s)
{
    const int t = blk * (blockDim.x / 64) + threadIdx.x / 64, lane = threadIdx.x % 64;
    if(t >= T)
        return;
#pragma unroll
    for(int it = 0; it < D / 1024; it++)
    {
        const int c = it * 1024 + lane * 16, kb = c / 128;
        const uint4* xs = reinterpret_cast<const uint4*>(x + (size_t)t * D + c);
        const uint4 v0 = xs[0], v1 = xs[1];
        const bf16 *b0 = reinterpret_cast<const bf16*>(&v0),
                   *b1 = reinterpret_cast<const bf16*>(&v1);
        float f[16], mx = 0.f;
#pragma unroll
        for(int i = 0; i < 8; i++)
        {
            f[i]     = float(b0[i]);
            f[8 + i] = float(b1[i]);
        }
#pragma unroll
        for(int i = 0; i < 16; i++)
            mx = fmaxf(mx, fabsf(f[i]));
#pragma unroll
        for(int o = 1; o < 8; o <<= 1)
            mx = fmaxf(mx, __shfl_xor(mx, o));
        const float sc = exp2f(ceilf(log2f(fmaxf(mx, 1e-10f) / 448.f))), inv = 1.f / sc;
        uint32_t w[4];
#pragma unroll
        for(int i = 0; i < 4; i++)
        {
            int p = __builtin_amdgcn_cvt_pk_fp8_f32(f[4 * i] * inv, f[4 * i + 1] * inv, 0, false);
            w[i]  = (uint32_t)__builtin_amdgcn_cvt_pk_fp8_f32(
                f[4 * i + 2] * inv, f[4 * i + 3] * inv, p, true);
        }
        *reinterpret_cast<uint4*>(q + (size_t)t * D + c) = {w[0], w[1], w[2], w[3]};
        if((lane & 7) == 0)
            s[(size_t)kb * T + t] = sc;
    }
}

// Fuse independent counting and quantization grids; finish counting before moe_scan.
template <int D>
__global__ void count_and_quant(const int* __restrict__ tk_e,
                                int n,
                                int E,
                                int* __restrict__ cnt,
                                int count_blocks,
                                const bf16* __restrict__ x,
                                int T,
                                fp8e4m3* __restrict__ q,
                                float* __restrict__ s)
{
    if((int)blockIdx.x < count_blocks)
        moe_count_block(blockIdx.x, tk_e, n, E, cnt);
    else
        quant_tok16_block<D>(blockIdx.x - count_blocks, x, T, q, s);
}

namespace pin {
// Same coordinates as load_one_shuf; sharing this helper changes the GEMM schedule.
struct tile_rc
{
    int row, col;
};
template <typename ST>
__device__ __forceinline__ tile_rc load_coord(int i)
{
    using T                        = typename ST::dtype;
    constexpr int bytes_per_thread = ST::underlying_subtile_bytes_per_thread;
    constexpr int bytes_per_warp   = bytes_per_thread * kittens::WARP_THREADS;
    const int lane_byte_offset     = (kittens::laneid() * bytes_per_thread) +
                                 (kittens::warpid() * bytes_per_warp) +
                                 (i * NUM_WARPS * bytes_per_warp);
    const int subtile_id               = lane_byte_offset / ST::underlying_subtile_bytes;
    const int subtile_row              = subtile_id / ST::underlying_subtiles_per_row;
    const int subtile_col              = subtile_id % ST::underlying_subtiles_per_row;
    const int subtile_lane_byte_offset = lane_byte_offset % ST::underlying_subtile_bytes;
    const int row     = subtile_lane_byte_offset / ST::underlying_subtile_row_bytes;
    const int col     = (subtile_lane_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T);
    const uint32_t sw = ST::swizzle({row, col});
    const int grow =
        (sw / ST::underlying_subtile_row_bytes) + subtile_row * ST::underlying_subtile_rows;
    const int gcol = (sw % ST::underlying_subtile_row_bytes) / sizeof(T) +
                     subtile_col * ST::underlying_subtile_cols;
    return {grow, gcol};
}

} // namespace pin

// Stage-1 tables, one block per tile. aoff[tile][thread][h*4+i] stores token*K+column;
// the K-step offset is added by the buffer instruction. Padding gathers token 0.
// pa[kb][tile][I][wm][l] byte n is the scale exponent for row
// tile*256+I*128+wm*64+n*16+l. Padding rows use unit scale.
template <int K, int KB>
__global__ void build_gather_tables(const int* __restrict__ tok,
                                    const int* __restrict__ tile_e,
                                    int max_tiles,
                                    const float* __restrict__ s,
                                    int T,
                                    uint32_t* __restrict__ aoff,
                                    uint32_t* __restrict__ pa)
{
    using ST_A     = st_fp8e4m3<128, 128, st_16x128_s>;
    const int tile = blockIdx.x;
    // Unused tiles have uninitialized token IDs; do not read them.
    if(tile >= tile_e[max_tiles])
        return;
#pragma unroll
    for(int u = 0; u < KB * 64 / 256; u++)
    {
        const int j = u * 256 + threadIdx.x, kb = j / 64, l = j % 16, wm = (j / 16) % 2,
                  I = (j / 32) % 2;
        uint32_t w  = 0;
#pragma unroll
        for(int n = 0; n < 4; n++)
        {
            const int t = tok[tile * 256 + I * 128 + wm * 64 + n * 16 + l];
            w |= ((unsigned)t >= (unsigned)T
                      ? 127u
                      : ((__float_as_uint(s[(size_t)kb * T + t]) >> 23) & 0xffu))
                 << (8 * n);
        }
        pa[((size_t)kb * max_tiles + tile) * 64 + j % 64] = w;
    }
    uint32_t o[8];
#pragma unroll
    for(int i = 0; i < 4; i++)
    {
        const pin::tile_rc rc = pin::load_coord<ST_A>(i);
#pragma unroll
        for(int h = 0; h < 2; h++)
        {
            const int t  = tok[tile * 256 + h * 128 + rc.row];
            o[h * 4 + i] = (uint32_t)(t < 0 ? 0 : t) * K + rc.col;
        }
    }
    uint4* p = reinterpret_cast<uint4*>(aoff + ((size_t)tile * 256 + threadIdx.x) * 8);
    p[0]     = {o[0], o[1], o[2], o[3]};
    p[1]     = {o[4], o[5], o[6], o[7]};
}

// Host launch and workspace.

namespace {
constexpr int D = 4096, I = 1024, KB1 = D / 128, KB2 = I / 128, TM = 256;
inline size_t align256(size_t x) { return (x + 255) & ~size_t(255); }
// At most TM-1 padding rows per expert.
inline int rowmax(int T, int S, int E)
{
    return (int)((((size_t)T * S + (size_t)E * (TM - 1)) + 255) / 256 * 256);
}

struct Workspace
{
    fp8e4m3* q;           // [T][D] token-major quantized x
    float* sa1;           // [KB1][T] its pow2 scales
    uint32_t* pa1;        // stage-1 packed A scales
    uint32_t* aoff;       // stage-1 gather offsets [tiles][256][8]
    fp8e4m3* act;         // [R][I] quantized activations (sorted rows)
    uint32_t* pa2;        // stage-2 packed A scales
    bf16* cw;             // [R][D] unweighted expert outputs (sorted rows)
    float* w;             // [R] router weight per sorted row
    int* tok;             // [R] token per sorted row (-1 = padding)
    int* tile_e;          // [tiles + 1] expert per row tile, used-tile count
    int* pos;             // [T * S] sorted row of each selection
    int *cnt, *off, *cur; // [E] routing counters
};
// Return workspace bytes; nullptr queries size only.
size_t carve(char* base, int T, int S, int E, Workspace* ws)
{
    const size_t R = rowmax(T, S, E), NT = R / TM;
    size_t o  = 0;
    auto take = [&](size_t bytes) {
        char* p = base ? base + o : nullptr;
        o += align256(bytes);
        return p;
    };
    Workspace w;
    w.q      = (fp8e4m3*)take((size_t)T * D);
    w.sa1    = (float*)take((size_t)KB1 * T * 4);
    w.pa1    = (uint32_t*)take((size_t)KB1 * NT * 64 * 4);
    w.aoff   = (uint32_t*)take(NT * 256 * 8 * 4);
    w.act    = (fp8e4m3*)take(R * I);
    w.pa2    = (uint32_t*)take((size_t)KB2 * NT * 64 * 4);
    w.cw     = (bf16*)take(R * D * 2);
    w.w      = (float*)take(R * 4);
    w.tok    = (int*)take(R * 4);
    w.tile_e = (int*)take((NT + 1) * 4);
    w.pos    = (int*)take((size_t)T * S * 4);
    w.cnt    = (int*)take(E * 4);
    w.off    = (int*)take(E * 4);
    w.cur    = (int*)take(E * 4);
    if(ws)
        *ws = w;
    return o;
}
// Query the current device on every call.
int num_cus()
{
    int n = 0, dev = 0;
    hipGetDevice(&dev);
    hipDeviceGetAttribute(&n, hipDeviceAttributeMultiprocessorCount, dev);
    return n;
}

template <int S>
void forward(const bf16* x,
             bf16* out,
             int T,
             const int* tk_e,
             const float* tk_w,
             const fp8e4m3* w13,
             const float* s13,
             const fp8e4m3* w2,
             const float* s2,
             int E,
             float swiglu_limit,
             const Workspace& ws,
             hipStream_t st)
{
    const int R = rowmax(T, S, E), NT = R / TM, n = T * S, ncu = num_cus();
    kittens::gl<fp8e4m3, 1, 1, MOE_ROWS_MAX, D> A1(ws.q, nullptr, nullptr, nullptr, nullptr);
    kittens::gl<fp8e4m3, 1, 1, 2 * I, D> W1((fp8e4m3*)w13, nullptr, nullptr, nullptr, nullptr);
    kittens::gl<fp8e4m3, 1, 1, MOE_ROWS_MAX, I> A2(ws.act, nullptr, nullptr, nullptr, nullptr);
    kittens::gl<fp8e4m3, 1, 1, D, I> W2((fp8e4m3*)w2, nullptr, nullptr, nullptr, nullptr);
    // routing
    hipMemsetAsync(ws.cnt, 0, E * 4, st);
    const int count_blocks = (n + SORT_PER_BLOCK - 1) / SORT_PER_BLOCK;
    count_and_quant<D><<<count_blocks + (T + SORT_WAVES - 1) / SORT_WAVES, SORT_THREADS, 0, st>>>(
        tk_e, n, E, ws.cnt, count_blocks, x, T, ws.q, ws.sa1);
    moe_scan<TM><<<1, SORT_EMAX, 0, st>>>(ws.cnt, E, ws.off, ws.cur, ws.tile_e, NT, ws.tok, ws.w);
    moe_scatter<<<count_blocks, SORT_THREADS, 0, st>>>(
        tk_e, tk_w, n, E, S, ws.cur, ws.tok, ws.w, ws.pos);
    // Stage 1: gate/up, clamped SwiGLU, FP8 requantization.
    build_gather_tables<D, KB1>
        <<<NT, 256, 0, st>>>(ws.tok, ws.tile_e, NT, ws.sa1, T, ws.aoff, ws.pa1);
    // Half-skip when routed experts average <=144 rows.
    const bool half_skip = (long)T * 8 <= 144L * (E - (S == 9 ? 1 : 0));
    auto gemm1 =
        half_skip ? moe_gemm_kernel<2 * I, D, true, true> : moe_gemm_kernel<2 * I, D, true, false>;
    gemm1<<<std::min(ncu, NT * (2 * I / 256)), 256, 0, st>>>(A1,
                                                             W1,
                                                             nullptr,
                                                             ws.pa1,
                                                             s13,
                                                             ws.tile_e,
                                                             NT,
                                                             ws.act,
                                                             ws.pa2,
                                                             swiglu_limit,
                                                             ws.aoff,
                                                             ws.cnt,
                                                             ws.off);
    // Stage 2: down projection and weighted combine.
    auto gemm2 =
        half_skip ? moe_gemm_kernel<D, I, false, true> : moe_gemm_kernel<D, I, false, false>;
    gemm2<<<std::min(ncu, NT * (D / 256)), 256, 0, st>>>(
        A2, W2, ws.cw, ws.pa2, s2, ws.tile_e, NT, nullptr, nullptr, 0.f, nullptr, ws.cnt, ws.off);
    moe_reduce_w<D, S><<<T, 512, 0, st>>>(ws.cw, ws.pos, ws.w, out);
    const hipError_t error = hipGetLastError();
    AITER_CHECK(error == hipSuccess, "hk_fp8_moe: ", hipGetErrorString(error));
}
} // namespace

// Use 256-byte units to fit workspace sizes in the int return.
AITER_CTYPES_DEFINE_ENTRYPOINT(
    aiter_hk_fp8_moe_workspace_units,
    (int64_t num_tokens, int64_t topk, int64_t num_experts, hipStream_t stream),
    (num_tokens, topk, num_experts, stream))
{
    return static_cast<int>((carve(nullptr,
                                   static_cast<int>(num_tokens),
                                   static_cast<int>(topk),
                                   static_cast<int>(num_experts),
                                   nullptr) +
                             255) /
                            256);
}

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(aiter_hk_fp8_moe_fwd,
                                    (size_t x,
                                     size_t out,
                                     int64_t num_tokens,
                                     size_t topk_ids,
                                     size_t topk_weights,
                                     int64_t topk,
                                     size_t w13,
                                     size_t w13_scale,
                                     size_t w2,
                                     size_t w2_scale,
                                     int64_t num_experts,
                                     float swiglu_limit,
                                     size_t workspace,
                                     int64_t workspace_bytes,
                                     hipStream_t stream),
                                    (x,
                                     out,
                                     num_tokens,
                                     topk_ids,
                                     topk_weights,
                                     topk,
                                     w13,
                                     w13_scale,
                                     w2,
                                     w2_scale,
                                     num_experts,
                                     swiglu_limit,
                                     workspace,
                                     workspace_bytes,
                                     stream))
{
    AITER_CHECK(num_tokens >= 0 && num_tokens <= 0x7ffffff0 / D,
                "hk_fp8_moe: token count exceeds the gather address range");
    AITER_CHECK(topk == 8 || topk == 9, "hk_fp8_moe: top-k must be 8 or 9");
    AITER_CHECK(num_experts >= 1 && num_experts <= SORT_EMAX,
                "hk_fp8_moe: expert count must be in [1, 1024]");
    if(num_tokens == 0)
        return;
    const int T = num_tokens, S = topk, E = num_experts;
    Workspace ws;
    AITER_CHECK(workspace_bytes >= 0 && carve(reinterpret_cast<char*>(workspace), T, S, E, &ws) <=
                                            static_cast<size_t>(workspace_bytes),
                "hk_fp8_moe: workspace too small");
    auto run = topk == 8 ? forward<8> : forward<9>;
    run(reinterpret_cast<const bf16*>(x),
        reinterpret_cast<bf16*>(out),
        T,
        reinterpret_cast<const int*>(topk_ids),
        reinterpret_cast<const float*>(topk_weights),
        reinterpret_cast<const fp8e4m3*>(w13),
        reinterpret_cast<const float*>(w13_scale),
        reinterpret_cast<const fp8e4m3*>(w2),
        reinterpret_cast<const float*>(w2_scale),
        E,
        swiglu_limit,
        ws,
        stream);
}
