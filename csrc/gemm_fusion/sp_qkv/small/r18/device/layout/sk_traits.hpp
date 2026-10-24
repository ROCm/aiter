// SPDX-License-Identifier: MIT
// SM-SK2-RAN-r2 split-K GEMM geometry + Opus layouts. UNCOMPILED.
// r2 vs r1: multi-stage LDS ring (G1), direct global->LDS via gmem::async_load = buffer_load..lds (G2), W staged in PHYSICAL
// shuffle order with 1 KiB contiguous per wave instruction (G3), dead A rows neither loaded nor staged (G4).
#pragma once
#include <opus/opus.hpp>
#include "sm_abi_v2.h"

namespace sks {
// ---- geometry ----
constexpr int BM = 32, BN = 64, BK = 128;
constexpr int THREADS = 256, PRODUCER_WAVES = 2, CONSUMER_WAVES = 2;
constexpr int WM = 16, WN = 16, WK = 128;            // mfma 16x16x128 fp8
constexpr int TM = 2, TN = 1, TK = 1;                // consumer wave grid
constexpr int EM = BM / (WM * TM), EN = BN / (WN * TN), EK = BK / (WK * TK);   // 1,4,1
constexpr int SK = static_cast<int>(SM_SK), WG_PER_CU = 2;          // r14: from the ABI (SM_SK_CFG 2|3)
constexpr int KBLOCKS = 48;                          // K=6144 / 128
constexpr int A_STAGE_BYTES = BM * BK;               // 4096  row-major [32][128]
constexpr int B_STAGE_BYTES = BN * BK;               // 8192  4 bands x 2048 B, physical shuffle order
constexpr int STAGE_BYTES = A_STAGE_BYTES + B_STAGE_BYTES;   // 12288
constexpr int PFK = 6;                               // ring depth; PFK-1 = 5 K-blocks in flight per WG
constexpr int LDS_BYTES = PFK * STAGE_BYTES;         // 73728; x WG_PER_CU=2 = 147456 <= 163840
static_assert(EM == 1 && EN == 4 && EK == 1);
static_assert(LDS_BYTES * WG_PER_CU <= 163840);
static_assert(PFK >= 3);
constexpr int B_INSTR_PER_WAVE = 4;                  // 8 x 1 KiB per stage / 2 producer waves
constexpr int A_INSTR_M32 = 2, A_INSTR_M16 = 1;      // per producer wave: 1 KiB (8 rows x 128 B) each; total A rows 32 / 16

// ---- contiguous K128-block split ownership (first SK-1 splits get ceil(48/SK)) ----
OPUS_D int kb_per_split() { return (KBLOCKS + SK - 1) / SK; }
OPUS_D int kb_begin(int s) { return s * kb_per_split(); }
OPUS_D int kb_end(int s) { const int e = (s + 1) * kb_per_split(); return (s == SK - 1 || e > KBLOCKS) ? KBLOCKS : e; }

// ---- gmem source views (element == byte for fp8) ----
// W physical (shuffle16x16, same storage as preshuffled_weight_view strides 16K,512,256,16,1): one n16 band x K128 block
// is 2048 contiguous bytes = (half 2, lane 64, byte 16). Each half is ONE wave instruction (1 KiB contiguous).
OPUS_D auto w_lane_view() { return opus::make_layout<-1>(opus::make_tuple(2, 64, 16), opus::make_tuple(1024, 16, 1)); }
// Byte offset of (band n16 index, K128 block) origin inside W: n16*16*K + kb*2048  (K = full row length)
OPUS_D int w_block_origin(int n16, int kb, int K) { return n16 * 16 * K + kb * 2048; }
// A source: lane -> (row-in-8, 16 B chunk): 8 rows x 128 B = 1 KiB per instruction.
OPUS_D auto a_lane_view(int strideA) { return opus::make_layout<-1>(opus::make_tuple(8, 8, 16), opus::make_tuple(strideA, 16, 1)); }

// ---- LDS fragment views (byte offsets inside one stage) ----
// A: row-major [32][128]; lane fragment = (row, seg 2 (stride 64 B), g 4 (stride 16 B)); one lane reads 16 B at seg 0 and seg 1.
OPUS_D auto a_frag_view() { return opus::make_layout<-1>(opus::make_tuple(32, 2, 4), opus::make_tuple(BK, 64, 16)); }
// B (physical order): (band nb 4, seg 2, g/2 2, g%2 2, n16 16): strides (2048,1024,512,256,16). Same K permutation as A.
OPUS_D auto b_frag_view() { return opus::make_layout<-1>(opus::make_tuple(4, 2, 2, 2, 16), opus::make_tuple(2048, 1024, 512, 256, 16)); }

// ---- MFMA lane decode (C row-major (lane/16, lane%16)) ----
template<class Shape> OPUS_D auto lane_coord(const Shape& shape, int lane) {
    return opus::flat_to_coords(lane, opus::make_index_seq<opus::size<Shape>()>{}, shape);
}
OPUS_D auto consumer_wave_coord(int cw) {
    return opus::flat_to_coords(cw, opus::make_index_seq<3>{},
        opus::make_tuple(opus::number<TM>{}, opus::number<TN>{}, opus::number<TK>{}));
}
template<class MMA> OPUS_D auto accumulator_register_view(const MMA& mma) {
    using Base = typename MMA::MMA;
    return opus::make_layout<-1>(mma.y_shape_c(),
        opus::make_tuple(opus::number<MMA::expd_n * Base::elem_c>{},
                         opus::number<Base::pack_c>{}, opus::number<1>{},
                         opus::number<Base::elem_c>{}));
}
// Private partial buffer P[SK][BM][N] fp32
OPUS_D auto partial_view(int N) { return opus::make_layout<-1>(opus::make_tuple(SK, BM, N), opus::make_tuple(BM * N, N, 1)); }
} // namespace sks
