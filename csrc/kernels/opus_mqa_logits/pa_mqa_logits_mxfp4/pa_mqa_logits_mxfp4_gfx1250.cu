// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx1250's MXFP4 MQA-logits instances: their traits and registry entries. The kernels are in
// pa_mqa_logits_mxfp4_gfx1250_h64.cuh and pa_mqa_logits_mxfp4_gfx1250_h32.cuh.
//
// The only translation unit built with -fno-honor-nans (optCompilerConfig.json's per-source
// flags): the kernels' relu is max_num with the NaN put back from the scales
// (pa_mqa_logits_mxfp4_gfx1250_common.cuh), and the flag cannot be scoped to the relu. Rerun the
// op test's NaN / Inf rows after every compiler upgrade: the flag lets the compiler assume no NaN.
//
// Also built with `-mllvm -enable-post-misched=1`, overriding aiter's default of 0: the kernels'
// WMMA / VALU interleave depends on it.
#include "pa_mqa_logits_mxfp4.h"
#include <opus/dtypes.hpp>

namespace opus_logits::gfx1250 {

// -- traits ---------------------------------------------------------------------------------------
// Host-safe: the host pass sizes the launch from them. Natural layouts: q [T, H, D/2],
// q_scale [T, H, 4], kv [pages, PAGE, D/2], kv_scale [pages, PAGE, 4]; each scale is the plain
// E8M0 byte of one 32-element K block.

// What both kernels share: a 64-key tile of one contiguous 64-row run of the plane, split into
// one TDM piece a wave, through a ring of STAGES LDS stages. Q_PER_BLOCK is the CTA's wave count;
// SCALE_TDM = 1 copies the key scales by a second TDM op, 0 loads them per lane.
template<int Q_PER_BLOCK_, int N_HEADS_, int STAGES_, int SCALE_TDM_>
struct mqa_logits_traits {
    static constexpr int KV_TILE_SIZE = 64;
    static constexpr int PAGE_SIZE    = 64;   // rows of one TDM run
    static constexpr int HEAD_DIM     = 128;
    static constexpr int N_HEADS      = N_HEADS_;

    // Constant, not opus::get_warp_size(): that returns 64 in the host pass, silently building
    // the wave64 fragment layout.
    static constexpr int WAVE_SIZE   = 32;
    static constexpr int Q_PER_BLOCK = Q_PER_BLOCK_;
    static constexpr int NUM_WAVES   = Q_PER_BLOCK;
    static constexpr int BLOCK_SIZE  = NUM_WAVES * WAVE_SIZE;

    using D_WEIGHT = opus::dtypes::bf16;   // per-head weights
    using D_ACC    = float;
    using D_OUT    = float;
    using D_SCALE  = int;   // E8M0 blockscale, packed as one int32 dword per lane (BX32)

    static constexpr int MMA_M = 32;   // heads
    static constexpr int MMA_N = 16;   // tokens
    static constexpr int MMA_K = 128;  // head_dim

    static constexpr int ELEM_BITS   = 4;
    static constexpr int SCALE_BLOCK = 32;

    static constexpr int M_TILES  = N_HEADS / MMA_M;      // 2 at H = 64, 1 at H = 32
    static constexpr int K_CHUNKS = MMA_K / SCALE_BLOCK;  // 4

    static constexpr int N_TILES        = KV_TILE_SIZE / MMA_N;      // 4
    static constexpr int PAGES_PER_TILE = KV_TILE_SIZE / PAGE_SIZE;  // 1

    static constexpr int C_FRAG = MMA_M * MMA_N / WAVE_SIZE;   // 16 floats per lane per m-tile
    static constexpr int GRPN_C = MMA_N;                       // 16: n == lane % 16
    static constexpr int GRPM_C = WAVE_SIZE / GRPN_C;          // 2:  M spread over 2 lane groups
    static constexpr int PACK_C = C_FRAG / 2;                  // 8:  c[0..7] is m = base + i
    static constexpr int REPT_C = C_FRAG / PACK_C;             // 2:  and c[8..15] is base + 16
    static constexpr int HEADS_PER_LANE = M_TILES * C_FRAG;

    static constexpr int KV_GRP_ELEMS = SCALE_BLOCK;                     // 32 fp4 per block
    static constexpr int KV_GRP_BYTES = KV_GRP_ELEMS * ELEM_BITS / 8;    // 16 B == one ds_load_b128
    static constexpr int A_BYTES_PER_LANE = MMA_M * MMA_K / WAVE_SIZE * ELEM_BITS / 8;  // 64
    static constexpr int B_BYTES_PER_LANE = MMA_N * MMA_K / WAVE_SIZE * ELEM_BITS / 8;  // 32
    static constexpr int A_VGPR_GROUPS = A_BYTES_PER_LANE / KV_GRP_BYTES;   // 4
    static constexpr int B_VGPR_GROUPS = B_BYTES_PER_LANE / KV_GRP_BYTES;   // 2

    static constexpr int SCALE_BYTES_PER_DWORD = K_CHUNKS;   // 4
    static_assert(SCALE_BYTES_PER_DWORD == 4,
                  "a BX32 scale operand is one dword; byte b must be block b, nothing left over");

    static constexpr int TOKEN_BYTES    = HEAD_DIM * ELEM_BITS / 8;    // 64 == one token's fp4 row
    static constexpr int Q_ROW_BYTES    = N_HEADS * TOKEN_BYTES;
    static constexpr int KV_PAGE_BYTES  = PAGE_SIZE * TOKEN_BYTES;     // 4096
    static constexpr int W_ROW_ELEMS    = N_HEADS;                     // natural [T, H] bf16

    // -- the ABI, as byte offsets within one row / one page --
    static constexpr int q_byte(int head, int block) {
        return head * TOKEN_BYTES + block * KV_GRP_BYTES;
    }
    static constexpr int kv_byte(int token, int block) {
        return token * TOKEN_BYTES + block * KV_GRP_BYTES;
    }

    // -- the WMMA fragment maps: (lane, VGPR group / register) -> (row or column, K block) --
    static constexpr int lane_m0(int lane) { return lane % GRPN_C; }   // L % 16
    static constexpr int lane_g (int lane) { return lane / GRPN_C; }   // L / 16

    static constexpr int frag_a_row  (int lane, int v) { return lane_m0(lane) + (v >= 2 ? 16 : 0); }
    static constexpr int frag_a_block(int lane, int v) { return lane_g(lane) + (v % 2 ? 2 : 0); }
    static constexpr int frag_b_col  (int lane)        { return lane_m0(lane); }
    static constexpr int frag_b_block(int lane, int v) { return lane_g(lane) + (v ? 2 : 0); }
    static constexpr int frag_c_n    (int lane)        { return lane_m0(lane); }
    static constexpr int frag_c_m    (int lane, int reg) {
        return (reg < PACK_C ? 0 : 16) + lane_g(lane) * PACK_C + reg % PACK_C;
    }
    static constexpr bool frag_a_is_bijection() {
        bool seen[MMA_M][K_CHUNKS] = {};
        for (int lane = 0; lane < WAVE_SIZE; ++lane)
            for (int v = 0; v < A_VGPR_GROUPS; ++v) {
                const int r = frag_a_row(lane, v), b = frag_a_block(lane, v);
                if (r < 0 || r >= MMA_M || b < 0 || b >= K_CHUNKS || seen[r][b]) return false;
                seen[r][b] = true;
            }
        for (int r = 0; r < MMA_M; ++r)
            for (int b = 0; b < K_CHUNKS; ++b) if (!seen[r][b]) return false;
        return true;
    }
    static constexpr bool frag_b_is_bijection() {
        bool seen[MMA_N][K_CHUNKS] = {};
        for (int lane = 0; lane < WAVE_SIZE; ++lane)
            for (int v = 0; v < B_VGPR_GROUPS; ++v) {
                const int n = frag_b_col(lane), b = frag_b_block(lane, v);
                if (n < 0 || n >= MMA_N || b < 0 || b >= K_CHUNKS || seen[n][b]) return false;
                seen[n][b] = true;
            }
        for (int n = 0; n < MMA_N; ++n)
            for (int b = 0; b < K_CHUNKS; ++b) if (!seen[n][b]) return false;
        return true;
    }
    static constexpr bool frag_c_is_bijection() {
        bool seen[MMA_M][MMA_N] = {};
        for (int lane = 0; lane < WAVE_SIZE; ++lane)
            for (int reg = 0; reg < C_FRAG; ++reg) {
                const int m = frag_c_m(lane, reg), n = frag_c_n(lane);
                if (m < 0 || m >= MMA_M || n < 0 || n >= MMA_N || seen[m][n]) return false;
                seen[m][n] = true;
            }
        for (int m = 0; m < MMA_M; ++m)
            for (int n = 0; n < MMA_N; ++n) if (!seen[m][n]) return false;
        return true;
    }
    static_assert(frag_a_is_bijection(),
                  "A's (lane, VGPR group) -> (row, block) map is not a bijection");
    static_assert(frag_b_is_bijection(),
                  "B's (lane, VGPR group) -> (col, block) map is not a bijection");
    static_assert(frag_c_is_bijection(), "C's (lane, reg) -> (m, n) map is not a bijection");

    // -- LDS tile image. TDM copies the run verbatim and WMMA fixes the lane->token map, so the
    // padding is the only way to break the bank conflict. --
    static constexpr int LDS_PAD_INTERVAL = 128;
    static constexpr int LDS_PAD_AMOUNT   =  16;

    static constexpr int lds_expand(int page_byte) {
        return page_byte + (page_byte / LDS_PAD_INTERVAL) * LDS_PAD_AMOUNT;
    }
    static constexpr int lds_byte(int token_in_page, int block) {
        return lds_expand(kv_byte(token_in_page, block));
    }
    static constexpr int LDS_PAGE_BYTES  = lds_expand(KV_PAGE_BYTES);   // 4608
    static constexpr int LDS_TILE_BYTES  = PAGES_PER_TILE * LDS_PAGE_BYTES;
    static constexpr int LDS_SCALE_BYTES = KV_TILE_SIZE * SCALE_BYTES_PER_DWORD;   // 256
    static constexpr int LDS_STAGE_BYTES = LDS_TILE_BYTES + LDS_SCALE_BYTES;
    static constexpr int lds_scale_byte(int token_in_tile) {
        return LDS_TILE_BYTES + token_in_tile * SCALE_BYTES_PER_DWORD;
    }

    static constexpr int LDS_NTILE_BYTES = lds_byte(MMA_N, 0) - lds_byte(0, 0);   // 1152
    static constexpr bool lds_ntile_step_is_uniform() {
        for (int t = 0; t + MMA_N <= PAGE_SIZE - MMA_N; ++t)
            for (int b = 0; b < K_CHUNKS; ++b)
                if (lds_byte(t + MMA_N, b) - lds_byte(t, b) != LDS_NTILE_BYTES) return false;
        return true;
    }
    static_assert(lds_ntile_step_is_uniform(),
                  "the n-tile step in LDS is not lane-independent, so the B reads cannot share an "
                  "address register. The pad interval must divide MMA_N * TOKEN_BYTES.");

    static constexpr int LDS_BLOCK_BYTES = lds_byte(0, 1) - lds_byte(0, 0);   // 16
    static constexpr bool lds_block_step_is_uniform() {
        for (int t = 0; t < PAGE_SIZE; ++t)
            for (int b = 0; b + 1 < K_CHUNKS; ++b)
                if (lds_byte(t, b + 1) - lds_byte(t, b) != LDS_BLOCK_BYTES) return false;
        return true;
    }
    static_assert(lds_block_step_is_uniform(),
                  "the K-block step in LDS is not uniform (the pad cuts a token in half), so the B "
                  "reads' block offset cannot be an immediate");

    static constexpr int lds_bank_unit(int lane, int v) {
        return (lds_byte(frag_b_col(lane), frag_b_block(lane, v)) >> 4) % 8;
    }
    static constexpr bool lds_conflict_free() {
        for (int v = 0; v < B_VGPR_GROUPS; ++v)
            for (int base = 0; base < WAVE_SIZE; base += 8) {
                int seen = 0;
                for (int i = 0; i < 8; ++i) seen |= 1 << lds_bank_unit(base + i, v);
                if (seen != 0xFF) return false;
            }
        return true;
    }
    static constexpr bool lds_conflict_free_unpadded_control() {
        for (int base = 0; base < WAVE_SIZE; base += 8) {
            int seen = 0;
            for (int i = 0; i < 8; ++i) {
                const int lane = base + i;
                const int linear =
                    frag_b_col(lane) * TOKEN_BYTES + frag_b_block(lane, 0) * KV_GRP_BYTES;
                seen |= 1 << ((linear >> 4) % 8);
            }
            if (seen != 0xFF) return false;
        }
        return true;
    }
    static_assert(!lds_conflict_free_unpadded_control(),
                  "the unpadded natural layout is supposed to CONFLICT; if it does not, this "
                  "checker cannot detect a conflict and the assert below proves nothing");
    static_assert(lds_conflict_free(),
                  "the LDS padding policy leaves a bank conflict: the pad of one read vector per "
                  "128 B is what makes 8 lanes tile all 32 banks under the 64 B token stride");

    // -- TDM: each issuing wave loads one contiguous piece of the run (keeps the loop
    // branch-free). The issuing waves are the largest power of two <= NUM_WAVES, so a 64-row
    // page splits evenly: all of them at 1 / 2 / 4 / 8 waves, the first 4 of 6. --
    static constexpr int TDM_ISSUE_WAVES =
        NUM_WAVES >= 8 ? 8 : NUM_WAVES >= 4 ? 4 : NUM_WAVES >= 2 ? 2 : 1;
    static constexpr int TDM_PAGE_SPLIT        = TDM_ISSUE_WAVES / PAGES_PER_TILE;
    static constexpr int TDM_INFLIGHT_PER_WAVE = 3;   // hardware, per opus.hpp
    static constexpr int TDM_ROWS        = PAGE_SIZE / TDM_PAGE_SPLIT;
    static constexpr int TDM_PIECE_BYTES = KV_PAGE_BYTES / TDM_PAGE_SPLIT;
    static constexpr int LDS_PIECE_BYTES = lds_expand(TDM_PIECE_BYTES);
    static_assert(TDM_ISSUE_WAVES % PAGES_PER_TILE == 0,
                  "the issuing waves must divide into whole runs, or a wave straddles two");
    static_assert(PAGE_SIZE % TDM_PAGE_SPLIT == 0 && KV_PAGE_BYTES % TDM_PAGE_SPLIT == 0,
                  "a run must split into equal pieces");
    static_assert(LDS_PIECE_BYTES * TDM_PAGE_SPLIT == LDS_PAGE_BYTES,
                  "the pieces must tile the padded LDS run exactly; otherwise they overlap by the "
                  "rounding -- silently, because each DMA on its own is still in range");

    // -- the ring: tile t's TDM ops have landed once at most the later tiles' are in flight --
    static constexpr int  STAGES         = STAGES_;
    static constexpr bool SCALE_TDM      = SCALE_TDM_ != 0;
    static constexpr int  TDM_OPS_A_TILE = SCALE_TDM ? 2 : 1;   // per issuing wave
    static constexpr int  TENSORCNT_KEEP = (STAGES - 2) * TDM_OPS_A_TILE;
    static constexpr int  LDS_BYTES      = STAGES * LDS_STAGE_BYTES;
    static_assert(STAGES >= 2, "need at least a double buffer");
    static_assert((STAGES - 1) * TDM_OPS_A_TILE <= TDM_INFLIGHT_PER_WAVE,
                  "the ring asks for more TDM ops in flight than a wave may have (3)");

    static_assert(HEAD_DIM == MMA_K, "K = 128 is the whole head_dim in one instruction");
    static_assert(N_HEADS % MMA_M == 0 && M_TILES >= 1, "N_HEADS must be a multiple of 32");
    static_assert(KV_TILE_SIZE % PAGE_SIZE == 0 && PAGE_SIZE % MMA_N == 0,
                  "a KV tile is whole runs of whole n-tiles");
    static_assert(GRPM_C == 2, "the head reduction emits exactly one permlane16_swap");
    static_assert(HEADS_PER_LANE * GRPM_C == N_HEADS,
                  "a lane and its swap partner must together hold every head");
    static_assert(A_BYTES_PER_LANE == 64 && B_BYTES_PER_LANE == 32,
                  "the dedicated f4 instruction takes i32x16 / i32x8 and uses all of both");
    static_assert(Q_PER_BLOCK >= 1, "Q_PER_BLOCK must be positive");
};

// The n_heads = 64 kernel (V4): QPB rows a CTA, one a wave.
template<int QPB_, int STAGES_, int SCALE_TDM_>
struct h64_traits : mqa_logits_traits<QPB_, 64, STAGES_, SCALE_TDM_> {};

// The n_heads = 32 kernel (V4.1): W waves a CTA, R rows a wave. GATHER_ = 1 is the candidate
// layers' kernel: every row its own table of 8-row page ids, a 64-key tile one TDM gather of its
// 8 pages; rows share no keys, so one row a wave and one wave a CTA.
template<int R_, int W_, int STAGES_, int SCALE_TDM_, int GATHER_>
struct h32_traits : mqa_logits_traits<W_, 32, STAGES_, SCALE_TDM_> {
    using base = mqa_logits_traits<W_, 32, STAGES_, SCALE_TDM_>;
    static constexpr int  ROWS_PER_WAVE = R_;
    static constexpr int  ROWS_PER_CTA  = W_ * R_;
    static constexpr bool GATHER        = GATHER_ != 0;
    static constexpr int  PAGE_ROWS     = 8;   // the candidate plane's page
    static constexpr int  TILE_PAGES    = base::KV_TILE_SIZE / PAGE_ROWS;   // gather<32>'s list
    static_assert(!GATHER || (R_ == 1 && W_ == 1),
                  "a candidate row shares its table with no row: one row a wave, one wave a CTA");
};

}  // namespace opus_logits::gfx1250

// The device pass on gfx1250 gets the kernels; every other pass gets empty stubs so the launch
// symbols resolve. Both halves of the guard are load-bearing: the host pass reports warp size 64.
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1250__)
#include "pa_mqa_logits_mxfp4_gfx1250_h64.cuh"
#include "pa_mqa_logits_mxfp4_gfx1250_h32.cuh"
#else
namespace opus_logits::gfx1250 {
template<class T, int SCHED> __global__ void h64_kernel(mqa_logits_mxfp4_kargs) {}
template<class T, int SCHED> __global__ void h32_kernel(mqa_logits_mxfp4_kargs) {}
}  // namespace opus_logits::gfx1250
#endif

namespace opus_logits::gfx1250 {
namespace {

using h64_r1w4s2T  = h64_traits<4, 2, 1>;         // V4 prefill; decode, 3-4 and 7-8 rows a sequence
using h64_r1w1s2T  = h64_traits<1, 2, 1>;         // V4 decode, 1 row a sequence
using h64_r1w2s2T  = h64_traits<2, 2, 1>;         // V4 decode, 2 rows a sequence
using h64_r1w6s2T  = h64_traits<6, 2, 1>;         // V4 decode, 5-6 rows (4 waves issue the TDM)
using h32_r1w1s2L  = h32_traits<1, 1, 2, 0, 0>;   // V4.1 full-layer decode, 1 row a sequence
using h32_r1w2s2T  = h32_traits<1, 2, 2, 1, 0>;   // ... 2 rows
using h32_r3w1s2T  = h32_traits<3, 1, 2, 1, 0>;   // ... 3 rows
using h32_r2w2s2T  = h32_traits<2, 2, 2, 1, 0>;   // ... 4 rows
using h32_r3w2s2T  = h32_traits<3, 2, 2, 1, 0>;   // ... 5-6 rows
using h32_r4w2s2T  = h32_traits<4, 2, 2, 1, 0>;   // ... 7-8 rows; V4.1 full-layer prefill
using h32g_r1w1s4L = h32_traits<1, 1, 4, 0, 1>;   // V4.1 candidate-layer decode
using h32g_r1w1s3L = h32_traits<1, 1, 3, 0, 1>;   // V4.1 candidate-layer prefill

// A full-layer table entry may be any power-of-two run of 64-key tiles.
constexpr int ENTRY_ANY_RUN = 1 << 30;

template<class T, class K>
constexpr mqa_logits_instance entry(int rows_per_wave, int gather, int entry_rows_min,
                                    int entry_rows_max, int flat_out) {
    return {.arch           = "gfx1250",
            .heads          = T::N_HEADS,
            .rows_per_wave  = rows_per_wave,
            .waves          = T::NUM_WAVES,
            .stages         = T::STAGES,
            .scale_tdm      = T::SCALE_TDM,
            .gather         = gather,
            .entry_rows_min = entry_rows_min,
            .entry_rows_max = entry_rows_max,
            .tile_keys      = T::KV_TILE_SIZE,
            .modes          = K::modes,
            .flat_out       = flat_out,
            .launch         = K::launch,
            .occupancy      = K::occupancy};
}

// A table entry is one 64-key tile: 64 rows (page_size * pages_per_block).
template<class T>
constexpr mqa_logits_instance h64_entry() {
    using K = instance_kernels<T::BLOCK_SIZE, h64_kernel<T, SCHED_TABLE>,
                               h64_kernel<T, SCHED_FREE>, h64_kernel<T, SCHED_PREFILL>>;
    return entry<T, K>(1, 0, T::KV_TILE_SIZE, T::KV_TILE_SIZE, 0);
}

// Free is not compiled for the candidate kernel (mqa_assign says why).
template<class T>
constexpr mqa_logits_kernel h32_free_kernel() {
    if constexpr (T::GATHER) return nullptr;
    else return h32_kernel<T, SCHED_FREE>;
}

// Full layers: an entry is a run of 2^tpb_shift tiles. Candidate layers: an 8-row page.
template<class T>
constexpr mqa_logits_instance h32_entry() {
    using K = instance_kernels<T::BLOCK_SIZE, h32_kernel<T, SCHED_TABLE>, h32_free_kernel<T>(),
                               h32_kernel<T, SCHED_PREFILL>>;
    return entry<T, K>(T::ROWS_PER_WAVE, T::GATHER, T::GATHER ? T::PAGE_ROWS : T::KV_TILE_SIZE,
                       T::GATHER ? T::PAGE_ROWS : ENTRY_ANY_RUN, 1);
}

}  // namespace
}  // namespace opus_logits::gfx1250

namespace opus_logits {

const mqa_logits_instance* mqa_logits_gfx1250_instances(int* count) {
    using namespace gfx1250;
    static const mqa_logits_instance table[] = {
        h64_entry<h64_r1w4s2T>(),
        h64_entry<h64_r1w1s2T>(),
        h64_entry<h64_r1w2s2T>(),
        h64_entry<h64_r1w6s2T>(),
        h32_entry<h32_r1w1s2L>(),
        h32_entry<h32_r1w2s2T>(),
        h32_entry<h32_r3w1s2T>(),
        h32_entry<h32_r2w2s2T>(),
        h32_entry<h32_r3w2s2T>(),
        h32_entry<h32_r4w2s2T>(),
        h32_entry<h32g_r1w1s4L>(),
        h32_entry<h32g_r1w1s3L>(),
    };
    *count = static_cast<int>(sizeof(table) / sizeof(table[0]));
    return table;
}

}  // namespace opus_logits
