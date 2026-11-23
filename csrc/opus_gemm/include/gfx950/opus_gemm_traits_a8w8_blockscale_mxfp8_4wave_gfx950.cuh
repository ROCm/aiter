// SPDX-License-Identifier: Apache-2.0
#pragma once

// gfx950 MXFP8 1x128 blockscale GEMM in the NT / NN / TT / TN directions with plain
// (not shuffled) A and B, as DeepGEMM. FP8 A / B, E8M0 scales 1x128 for A and 128x128
// for B, BF16 C. Both operands share the copy / LDS / read layouts: N plays M for B.
//   A_T: A stored [K, M] (M-contiguous, TR8 reads), else [M, K] (K-contiguous, plain reads)
//   B_T: B stored [K, N] (N-contiguous, TR8 reads), else [N, K] (K-contiguous, plain reads)
//        NT = (A_T, B_T) = (0, 0), NN = (0, 1), TT = (1, 0), TN = (1, 1)
//
// A tile is built from 128x128 C quadrants: H_M = B_M / 128 = 2 and H_N = B_N / 128
// quadrants (1 or 2). Inside a quadrant the 2x2 waves each own E_M x E_N = 4x4
// 16x16 MFMA fragments, so the LDS slot layouts, TR8 pairing / swizzles and scale
// packing of a quadrant do not depend on the tile size.
//   GroupM: 0 keeps the 2x2 raster; 4 / 8 / 16 walk GroupM M tiles per N column, as NT's gm.
template<bool A_T_, bool B_T_, int BM = 256, int BN = 256, bool PadM = false, int GroupM = 0>
struct opus_gemm_mxfp8_4wave_tile_traits {
    static_assert(BM == 256 && (BN == 128 || BN == 256),
                  "MXFP8 direction tiles are 256x256 or 256x128");
    static_assert(GroupM == 0 || GroupM == 4 || GroupM == 8 || GroupM == 16);
    static constexpr bool A_T = A_T_, B_T = B_T_;

    static constexpr int BLOCK_SIZE = 256;
    static constexpr int WARP_SIZE = 64;
    static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
    static constexpr int B_M = BM, B_N = BN, B_K = 128;
    // PAD_M: M only needs to be a multiple of 16. A's whole copy offset then goes in the
    // bounds-checked VGPR offset (the buffer bounds check ignores SOFFSET) so A past M
    // reads zero; that costs ~0.8% at aligned shapes, hence the separate variant.
    // C is always bounded to the tile's valid rows (free), so padded rows are not written.
    static constexpr bool PAD_M = PadM;
    // HALF_B_* is the quadrant edge; H_* counts quadrants per tile side.
    static constexpr int HALF_B_M = 128, HALF_B_N = 128;
    static constexpr int H_M = B_M / HALF_B_M, H_N = B_N / HALF_B_N;
    static constexpr int T_M = 2, T_N = 2, T_K = 1;
    static constexpr int E_M = 4, E_N = 4, E_K = 1;
    static constexpr int W_M = 16, W_N = 16, W_K = 128;
    // Global-to-LDS copies use 16 FP8 bytes per lane for both operands.
    static constexpr int VEC_A = 16, VEC_B = 16, VEC_C = 4;
    static constexpr int GROUP_M = 1, GROUP_N = 128, GROUP_K = 128;
    static constexpr int VEC_SF = 4;
    static constexpr int VEC_SCALE_A = 16;
    static constexpr int VEC_SCALE_SF = 4;
    static constexpr int VEC_SCALE_SF_PAIR = 8;
    // One B scale per 128 N columns, i.e. per N quadrant.
    static constexpr int SCALE_N_HALVES = B_N / GROUP_N;
    static constexpr int SCALE_PANEL_K_CAPACITY = 64;
    static constexpr int GROUPED_M = GroupM;
    // A lane loads VEC_SCALE_A contiguous M scales, so B_M / VEC_SCALE_A lanes cover a K column.
    static constexpr int SFA_LANES_PER_K_COLUMN = B_M / VEC_SCALE_A;
    static constexpr int SFA_K_COLUMNS_PER_WAVE = WARP_SIZE / SFA_LANES_PER_K_COLUMN;
    static constexpr int SFA_K_COLUMNS_PER_PASS = NUM_WAVES * SFA_K_COLUMNS_PER_WAVE;
    static constexpr int SFA_PASSES_PER_CACHE_PANEL = SCALE_PANEL_K_CAPACITY / SFA_K_COLUMNS_PER_PASS;
    static constexpr int SFA_PANEL_PITCH = B_M;
    static constexpr int SFA_PANEL_BYTES = SFA_PANEL_PITCH * SCALE_PANEL_K_CAPACITY;
    static constexpr int SFB_PANEL_BYTES = SCALE_N_HALVES * SCALE_PANEL_K_CAPACITY * VEC_SF;
    // One packed A / B scale dword per quadrant row / column: bytes read per K tile.
    static constexpr int SFA_READ_BYTES = H_M * VEC_SF;
    static constexpr int SFB_READ_BYTES = H_N * VEC_SF;

    // A wave slot is one 16-byte-per-lane LDS-DMA pass (1024 bytes) plus padding.
    static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
    static constexpr int smem_padding = 32;
    // K-contiguous ([M, K] / [N, K]) slots hold 8 rows of 128 bytes, so their rows share
    // banks. With a 64-byte pad and chunk c of row r stored at c ^ (r % 4), the plain
    // ds_read_b128 operand reads are free of bank conflicts (SQ_LDS_BANK_CONFLICT = 0 in
    // a microbenchmark of the reader's addresses; 32 bytes of pad or c ^ (r / 2) are not).
    static constexpr int smem_padding_k = 64;
    static constexpr int SMEM_SLOT_K = smem_linear_wave + smem_padding_k;
    // Rows of B_K bytes per wave slot (M rows for a [M,K] slot).
    static constexpr int smem_sub = smem_linear_wave / B_K;
    // Slots per quadrant operand: 128 x B_K bytes / 1024.
    static constexpr int smem_m_rep = HALF_B_M * B_K / smem_linear_wave;
    static constexpr int smem_n_rep = HALF_B_N * B_K / smem_linear_wave;
    // Global-to-LDS copies (one 16-byte LDS-DMA load per lane) per K tile.
    static constexpr int A_COPIES = B_M * B_K / (BLOCK_SIZE * VEC_A);
    static constexpr int B_COPIES = B_N * B_K / (BLOCK_SIZE * VEC_B);
    static constexpr int COPIES_PER_QUADRANT = HALF_B_M * B_K / (BLOCK_SIZE * VEC_A);
    static_assert(NUM_WAVES == T_M * T_N * T_K && NUM_WAVES == 4);
    static_assert(HALF_B_M == E_M * T_M * W_M && HALF_B_N == E_N * T_N * W_N);

    // LDS reads: TR8 returns 8 FP8 bytes per lane, a plain read 16.
    static constexpr int VEC_LDS_A = A_T ? 8 : 16;
    static constexpr int VEC_LDS_B = B_T ? 8 : 16;
    // Each quadrant occupies sixteen 1024-byte wave slots, with 32-byte padding.
    static constexpr int SMEM_HALF_A_ELEMS = smem_m_rep * (A_T ? smem_linear_wave + smem_padding : SMEM_SLOT_K);
    static constexpr int SMEM_HALF_B_ELEMS = smem_n_rep * (B_T ? smem_linear_wave + smem_padding : SMEM_SLOT_K);
    static_assert(HALF_B_N == HALF_B_M && VEC_B == VEC_A && E_N == E_M && W_N == W_M && T_N == T_M,
                  "B reuses the A copy / LDS / read layouts");
    // Two LDS stages of H_M A quadrants and H_N B quadrants.
    static constexpr int SMEM_A_ELEMS = SMEM_HALF_A_ELEMS * 2 * H_M;
    static constexpr int SMEM_B_ELEMS = SMEM_HALF_B_ELEMS * 2 * H_N;
    static constexpr int LDS_BYTES = SMEM_A_ELEMS + SMEM_B_ELEMS + SFA_PANEL_BYTES + SFB_PANEL_BYTES;
    static_assert(LDS_BYTES <= 163840);
};

template<int BM = 256, int BN = 256, bool PadM = false, int GroupM = 0>
using opus_gemm_mxfp8_nt_4wave_tile_traits = opus_gemm_mxfp8_4wave_tile_traits<false, false, BM, BN, PadM, GroupM>;
template<int BM = 256, int BN = 256, bool PadM = false, int GroupM = 0>
using opus_gemm_mxfp8_nn_4wave_tile_traits = opus_gemm_mxfp8_4wave_tile_traits<false, true, BM, BN, PadM, GroupM>;
template<int BM = 256, int BN = 256, bool PadM = false, int GroupM = 0>
using opus_gemm_mxfp8_tt_4wave_tile_traits = opus_gemm_mxfp8_4wave_tile_traits<true, false, BM, BN, PadM, GroupM>;
template<int BM = 256, int BN = 256, bool PadM = false, int GroupM = 0>
using opus_gemm_mxfp8_tn_4wave_tile_traits = opus_gemm_mxfp8_4wave_tile_traits<true, true, BM, BN, PadM, GroupM>;

static_assert(opus_gemm_mxfp8_nt_4wave_tile_traits<256, 256>::LDS_BYTES == 156160);
static_assert(opus_gemm_mxfp8_nn_4wave_tile_traits<256, 256>::LDS_BYTES == 154112);
static_assert(opus_gemm_mxfp8_tn_4wave_tile_traits<256, 256>::LDS_BYTES == 152064);
