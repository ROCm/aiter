// SPDX-License-Identifier: MIT
// LG-QSDMA-r1 FP8 quad-subtile traits: large-amend-r1.md §2.2 table as constexpr + static_asserts.
#pragma once
namespace lgq {
constexpr int ceil_div_c(int a, int b) { return (a + b - 1) / b; }
struct QuadFp8Traits {
    static constexpr int BLOCK_SIZE = 512, WAVE = 64;
    static constexpr int B_M = 256, B_N = 256, B_K = 128;
    static constexpr int T_M = 2, T_N = 4, T_K = 1;
    static constexpr int W_M = 16, W_N = 16, W_K = 128;   // MFMA 16x16x128 FP8
    static constexpr int ESZ = 1;                          // sizeof(fp8)
    static constexpr int VEC_A = 16 / ESZ, VEC_B = 16 / ESZ, VEC_C = 8;  // C vec = 8 bf16 = 16 B
    static constexpr int HALF_B_M = B_M / 2, HALF_B_N = B_N / 2;
    static constexpr int E_M = HALF_B_M / (W_M * T_M);     // 4
    static constexpr int E_N = HALF_B_N / (W_N * T_N);     // 2
    static constexpr int E_K = B_K / (W_K * T_K);          // 1 (the two 64-K segments live inside one MFMA operand)
    static constexpr int SEG = 2;                          // 16-B K segments per lane operand, 64 K apart
    static constexpr int smem_linear_wave = WAVE * 16 / ESZ;             // 1024 elements == bytes
    static constexpr int smem_sub = smem_linear_wave / B_K;              // 8 rows per 1024B chunk
    static constexpr int smem_m_rep = HALF_B_M / smem_sub;               // 16
    static constexpr int smem_n_rep = HALF_B_N / smem_sub;               // 16 (B: 8 n16-bands x 2 K-segs)
    static constexpr int smem_padding = 2 * 16 / ESZ;                    // 32 B
    static constexpr int chunk_stride = smem_linear_wave + smem_padding; // 1056
    static constexpr int a_buffer_load_insts = HALF_B_M * B_K / (BLOCK_SIZE * VEC_A);  // 2
    static constexpr int b_buffer_load_insts = HALF_B_N * B_K / (BLOCK_SIZE * VEC_B);  // 2
    static constexpr int a_ds_read_insts = (E_M * SEG);                                // 8
    static constexpr int b_ds_read_insts = (E_N * SEG);                                // 4
    static constexpr int smem_a_bytes = smem_m_rep * chunk_stride * ESZ;               // 16896
    static constexpr int smem_b_bytes = smem_n_rep * chunk_stride * ESZ;               // 16896
    static constexpr int LDS_BYTES = 4 * smem_a_bytes + 4 * smem_b_bytes;              // 135168
    static constexpr int ACC_PER_LANE = 4 * E_M * E_N * 4;                             // 128
    static constexpr int MFMA_PER_WAVE_PER_KTILE = 4 * E_M * E_N * E_K;                // 32
    static constexpr int BAND_BYTES_PER_KTILE = 16 * B_K;                              // 2048 B per n16 band per K-tile
};
using Q = QuadFp8Traits;
static_assert(Q::BLOCK_SIZE / Q::WAVE == Q::T_M * Q::T_N * Q::T_K, "8 waves");
static_assert(Q::E_M == 4 && Q::E_N == 2 && Q::E_K == 1);
static_assert(Q::smem_sub == 8 && Q::smem_m_rep == 16 && Q::smem_n_rep == 16 && Q::smem_padding == 32);
static_assert(Q::a_buffer_load_insts == 2 && Q::b_buffer_load_insts == 2, "loads 2/2");
static_assert(Q::a_ds_read_insts == 8 && Q::b_ds_read_insts == 4, "ds_reads 8/4");
static_assert(Q::LDS_BYTES == 135168, "LDS 135168");
static_assert(Q::ACC_PER_LANE == 128 && Q::MFMA_PER_WAVE_PER_KTILE == 32);
static_assert((6144 / Q::B_K) == 48 && (6144 / Q::B_K) % 2 == 0, "K loops even");
static_assert(Q::BAND_BYTES_PER_KTILE == 2048);
} // namespace lgq
