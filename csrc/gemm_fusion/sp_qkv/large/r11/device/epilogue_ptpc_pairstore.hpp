// SPDX-License-Identifier: MIT
// LG-QSDMA-r1 epilogue: PTPC scale (reference order of PR695-M4 ptpc_field_epilogue.hpp::ptpc_cast)
// THEN the reference quad-subtile pair store (permlane16_swap + shfl), into the destination slab.
// Accumulator register order per sub-tile (swap_ab, same mma as the reference):
//   c[em][en][r] (32 floats, index (em*E_N+en)*4+r): for lane = 16*q + l16
//     row  m = em*32 + wave_m*16 + l16        (within half tile)
//     col  n = (en*4 + wave_n)*16 + 4*q + r   (within half tile; "band" = en*4+wave_n)
// DEPENDENCY (flagged for Evidence): this register order is the swap_ab C partition the BF16 reference's
// store_c already relies on; G-F1/G-F2 falsifiers exist to catch any mismatch.
#pragma once
#include <opus/opus.hpp>
#include "layout/quad_fp8_traits.hpp"
#include "layout/quad_fp8_layouts.hpp"
#include "lg_comm_abi_v2.h"

namespace lgq {
static constexpr int kFS = static_cast<int>(LG_COMM_F);   // r5x: slab row stride (2560); kF (=2304) remains the dense Q|K|V field-map width
// Exactly one BF16 rounding: bf16_rne( fp32( fp32(acc*Ws) * As ) ). Separate statements: no FMA contraction.
inline __device__ opus::bf16_t ptpc_cast(float acc, float wscale, float ascale) {
    const float sw = acc * wscale;
    const float sa = sw * ascale;
    return opus::fp32_to_bf16(sa);
}

// RTC-safe 16-B float vector (no HIP runtime float4 under -D__HIPCC_RTC__); moved with __builtin_memcpy.
struct alignas(16) F32x4 { float v[4]; };

// Per-lane scales for one half tile (loaded once after the K loop).
struct HalfScales {
    F32x4 w[2];       // w[en] = Wscale[raw_n0 + (en*4+wave_n)*16 + 4*q .. +3]
};

// v_c: one sub-tile accumulator (vtype_c, 32 floats). hm/hn: half-tile indices.
// as: Ascale for the lane's rows, as[em] = Ascale[row0 + hm*128 + em*32 + wave_m*16 + l16].
// c_slab: pointer to S_d (this half tile's destination slab base, [M][F]); packed_col0 = half-tile
// packed column origin within F (ht_packed(hv)); row0 = tile row origin; M for bounds is guaranteed by caller.
template<typename VC>
inline __device__ void store_c_ptpc(const VC& v_c, const HalfScales& ws, const float (&as)[Q::E_M],
                                    opus::bf16_t* c_slab, int row_base_hm /* absolute row of half tile row 0 */,
                                    int packed_col0, int lane_id, int wave_id_m, int wave_id_n, bool bypass = false) {   // r5 L2a: bypass = 16 B SC0|SC1 write-through store (staging slabs only)
    using namespace opus;
    // 1) scale + single RNE cast, packed into u32 pairs in the reference chunk layout:
    //    chunk c = em ; p[0..3] = {(en0 r0,r1), (en0 r2,r3), (en1 r0,r1), (en1 r2,r3)}
    uint32_t pk[Q::E_M][4];
    static_for<Q::E_M>([&](auto cem) {
        constexpr int em = cem.value;
        const float a = as[em];
        static_for<Q::E_N>([&](auto cen) {
            constexpr int en = cen.value;
            const F32x4 w4 = ws.w[en];
            struct { float x, y, z, w; } w = {w4.v[0], w4.v[1], w4.v[2], w4.v[3]};
            const float x0 = v_c[(em * Q::E_N + en) * 4 + 0];
            const float x1 = v_c[(em * Q::E_N + en) * 4 + 1];
            const float x2 = v_c[(em * Q::E_N + en) * 4 + 2];
            const float x3 = v_c[(em * Q::E_N + en) * 4 + 3];
            const bf16_t b0 = ptpc_cast(x0, w.x, a), b1 = ptpc_cast(x1, w.y, a);
            const bf16_t b2 = ptpc_cast(x2, w.z, a), b3 = ptpc_cast(x3, w.w, a);
            pk[em][en * 2 + 0] = static_cast<uint32_t>(__builtin_bit_cast(unsigned short, b0)) |
                                 (static_cast<uint32_t>(__builtin_bit_cast(unsigned short, b1)) << 16);
            pk[em][en * 2 + 1] = static_cast<uint32_t>(__builtin_bit_cast(unsigned short, b2)) |
                                 (static_cast<uint32_t>(__builtin_bit_cast(unsigned short, b3)) << 16);
        });
    });
    // 2) reference pair store (dtype-agnostic: moves packed 32-bit BF16 pairs)
    // r1d: plain 64-bit arithmetic (the former make_layout<-1>(make_tuple(1<<30,kF),..) could overflow a signed
    // int extent product -> UB -> compiler may delete the path). Slab row-major [M][F], F = kF.
    static_for<Q::E_M>([&](auto cem) {
        constexpr int em = cem.value;
        uint32_t* p = pk[em];
        auto r0 = __builtin_amdgcn_permlane16_swap(p[0], p[2], false, true);
        auto r1 = __builtin_amdgcn_permlane16_swap(p[1], p[3], false, true);
        p[0] = r0[0]; p[2] = r0[1];
        p[1] = r1[0]; p[3] = r1[1];

        const int lane = lane_id;
        const int half_col_block = lane / 32;                // 0: cols 0..15 of the wave band pair, 1: +64
        const int lane_in_half = lane - half_col_block * 32;
        const int row_in_16 = lane_in_half / 2;
        const int vec_pair = lane_in_half - row_in_16 * 2;
        const int src_lane = row_in_16 + half_col_block * 16 + vec_pair * 32;
        i32x4_t raw;
        raw[0] = __builtin_bit_cast(i32_t, shfl(p[0], src_lane));
        raw[1] = __builtin_bit_cast(i32_t, shfl(p[1], src_lane));
        raw[2] = __builtin_bit_cast(i32_t, shfl(p[2], src_lane));
        raw[3] = __builtin_bit_cast(i32_t, shfl(p[3], src_lane));
        const int row_in_half = em * (Q::T_M * Q::W_M) + wave_id_m * Q::W_M + row_in_16;
        const int col_vec = half_col_block * 8 + wave_id_n * 2 + vec_pair;   // 8-col unit within the half tile
        const long long off = static_cast<long long>(row_base_hm + row_in_half) * kFS + (packed_col0 + col_vec * Q::VEC_C);
        // 16-B aligned: packed_col0 % 128 == 0, VEC_C = 8 bf16, F*2 % 16 == 0
#if LG_L2A_BYPASS
        if (bypass) {
            // r5 L2a: 16 B global_store_dwordx4 with sc0 sc1 (system-coherent write-through on gfx940+/gfx950): no dirty line is left in L2, so the SDMA source needs no
            // L2 writeback drain. The asm is invisible to the compiler's vmcnt tracking: gemm_chunk_complete drains with an explicit asm vmcnt(0) (W-1) in every wave.
            // Byte content is identical to the cached store. `memory` clobber keeps ordering against surrounding accesses.
            // L2a-H (review lg-r5-l2b-review-r1): CDNA4 ISA p28 hazard table: after FLAT/GLOBAL_STORE_X4 a VALU that overwrites the store-data VGPRs needs 2 wait states
            // (1 for non-VALU writers). The hazard recognizer cannot see this asm, so the required wait states are INSIDE the same asm statement: s_nop 1 = 2 wait states.
            asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1\n\ts_nop 1" :: "v"(c_slab + off), "v"(raw) : "memory");
        } else
#endif
        __builtin_memcpy(c_slab + off, &raw, 16);
    });
}

// r5x: IK owner store (see store_c_ptpc; one cast, nslab stores)
template<typename VC>
inline __device__ void store_c_ptpc_multi(const VC& v_c, const HalfScales& ws, const float (&as)[Q::E_M],
                                    opus::bf16_t* const (&slabs)[4], int nslab, int row_base_hm /* absolute row of half tile row 0 */,
                                    int packed_col0, int lane_id, int wave_id_m, int wave_id_n) {   // r5x IK: identical scale/cast/pack as store_c_ptpc, then the SAME 16 B stored to nslab destination slabs (IK computed + rounded ONCE)
    using namespace opus;
    // 1) scale + single RNE cast, packed into u32 pairs in the reference chunk layout:
    //    chunk c = em ; p[0..3] = {(en0 r0,r1), (en0 r2,r3), (en1 r0,r1), (en1 r2,r3)}
    uint32_t pk[Q::E_M][4];
    static_for<Q::E_M>([&](auto cem) {
        constexpr int em = cem.value;
        const float a = as[em];
        static_for<Q::E_N>([&](auto cen) {
            constexpr int en = cen.value;
            const F32x4 w4 = ws.w[en];
            struct { float x, y, z, w; } w = {w4.v[0], w4.v[1], w4.v[2], w4.v[3]};
            const float x0 = v_c[(em * Q::E_N + en) * 4 + 0];
            const float x1 = v_c[(em * Q::E_N + en) * 4 + 1];
            const float x2 = v_c[(em * Q::E_N + en) * 4 + 2];
            const float x3 = v_c[(em * Q::E_N + en) * 4 + 3];
            const bf16_t b0 = ptpc_cast(x0, w.x, a), b1 = ptpc_cast(x1, w.y, a);
            const bf16_t b2 = ptpc_cast(x2, w.z, a), b3 = ptpc_cast(x3, w.w, a);
            pk[em][en * 2 + 0] = static_cast<uint32_t>(__builtin_bit_cast(unsigned short, b0)) |
                                 (static_cast<uint32_t>(__builtin_bit_cast(unsigned short, b1)) << 16);
            pk[em][en * 2 + 1] = static_cast<uint32_t>(__builtin_bit_cast(unsigned short, b2)) |
                                 (static_cast<uint32_t>(__builtin_bit_cast(unsigned short, b3)) << 16);
        });
    });
    // 2) reference pair store (dtype-agnostic: moves packed 32-bit BF16 pairs)
    // r1d: plain 64-bit arithmetic (the former make_layout<-1>(make_tuple(1<<30,kF),..) could overflow a signed
    // int extent product -> UB -> compiler may delete the path). Slab row-major [M][F], F = kF.
    static_for<Q::E_M>([&](auto cem) {
        constexpr int em = cem.value;
        uint32_t* p = pk[em];
        auto r0 = __builtin_amdgcn_permlane16_swap(p[0], p[2], false, true);
        auto r1 = __builtin_amdgcn_permlane16_swap(p[1], p[3], false, true);
        p[0] = r0[0]; p[2] = r0[1];
        p[1] = r1[0]; p[3] = r1[1];

        const int lane = lane_id;
        const int half_col_block = lane / 32;                // 0: cols 0..15 of the wave band pair, 1: +64
        const int lane_in_half = lane - half_col_block * 32;
        const int row_in_16 = lane_in_half / 2;
        const int vec_pair = lane_in_half - row_in_16 * 2;
        const int src_lane = row_in_16 + half_col_block * 16 + vec_pair * 32;
        i32x4_t raw;
        raw[0] = __builtin_bit_cast(i32_t, shfl(p[0], src_lane));
        raw[1] = __builtin_bit_cast(i32_t, shfl(p[1], src_lane));
        raw[2] = __builtin_bit_cast(i32_t, shfl(p[2], src_lane));
        raw[3] = __builtin_bit_cast(i32_t, shfl(p[3], src_lane));
        const int row_in_half = em * (Q::T_M * Q::W_M) + wave_id_m * Q::W_M + row_in_16;
        const int col_vec = half_col_block * 8 + wave_id_n * 2 + vec_pair;   // 8-col unit within the half tile
        const long long off = static_cast<long long>(row_base_hm + row_in_half) * kFS + (packed_col0 + col_vec * Q::VEC_C);
        // 16-B aligned: packed_col0 % 128 == 0, VEC_C = 8 bf16, F*2 % 16 == 0
for (int s = 0; s < nslab; ++s) __builtin_memcpy(slabs[s] + off, &raw, 16);   // plain cached stores (staging slabs are read by the SDMA after the one WBL2 release; L2a is not part of r5x)
    });
}

// Diagnostic epilogue (LG_DIAG_STAGE 1/2): plain scalar bf16 stores of ALL 32 accumulators per lane at the swap_ab
// C coordinates (no permlane/shfl/pair store), so every MFMA result stays live. scaled=false: bf16(acc) (stage 1);
// scaled=true: ptpc_cast (stage 2).
template<typename VC>
inline __device__ void store_c_plain(const VC& v_c, const HalfScales& ws, const float (&as)[Q::E_M], bool scaled,
                                     opus::bf16_t* c_slab, int row_base_hm, int packed_col0,
                                     int lane_id, int wave_id_m, int wave_id_n) {
    using namespace opus;
    const int q = lane_id / 16, l16 = lane_id % 16;
    static_for<Q::E_M>([&](auto cem) {
        constexpr int em = cem.value;
        static_for<Q::E_N>([&](auto cen) {
            constexpr int en = cen.value;
            const long long row = row_base_hm + em * (Q::T_M * Q::W_M) + wave_id_m * Q::W_M + l16;
            const long long col0 = packed_col0 + (en * Q::T_N + wave_id_n) * 16 + 4 * q;
            static_for<4>([&](auto cr) {
                constexpr int r = cr.value;
                const float x = v_c[(em * Q::E_N + en) * 4 + r];
                bf16_t b = scaled ? ptpc_cast(x, ws.w[en].v[r], as[em]) : fp32_to_bf16(x);
                c_slab[row * kFS + col0 + r] = b;
            });
        });
    });
}
} // namespace lgq
