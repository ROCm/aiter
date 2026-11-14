#pragma once

#include <cstddef>
#include <type_traits>

#include <opus/dtypes.hpp>

#include "../mla_prefill_kargs.h"

using bf16_t = opus::dtypes::bf16;
// 8-bit float storage types, aliased to match opus's dtype registration.
using fp8_t  = _BitInt(8);
using bf8_t  = unsigned _BitInt(8);

// SPLIT (every variant): the split-K kernel, see mla_prefill_split.h; without it the
// kernel's arguments and code are the unsplit ones.

// bf16 32mx4_32nx1 (H > 32): 4 waves x 32 heads, KV_TILE=32, 32x32 MFMA, prebuilt code object
// (build_co.sh). QK is split by head, PV by dv. One LDS tile serves as K (row reads) and V
// (tr-reads): bricks of 8 rows x 128 B + 64 B per 128 B d-chunk, brick w holding rows 4r + w, 16 B
// pieces XOR-swizzled by r % 4.
template <int Q_TILE_SIZE_  = 32,
          int KV_TILE_SIZE_ = 32,
          int D_TILE_SIZE_  = 512,
          int NUM_WARPS_    = 4,
          typename D_ATTN_  = bf16_t,
          bool SPLIT_       = false>
struct opus_mla_v4_prefill_a16w16_32mx4_32nx1_traits
{
    static constexpr int Q_TILE_SIZE  = Q_TILE_SIZE_;
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;
    static constexpr int D_TILE_SIZE  = D_TILE_SIZE_;
    static constexpr int D_HEAD_SIZE  = D_TILE_SIZE;
    static constexpr int NUM_WARPS    = NUM_WARPS_;
    static constexpr bool SPLIT       = SPLIT_;
    using kargs_type = std::conditional_t<SPLIT_,
                                          opus_mla_prefill_split_kargs<opus_mla_v4_prefill_kargs>,
                                          opus_mla_v4_prefill_kargs>;

    static constexpr int WARP_SIZE  = 64;
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;

    using D_ATTN = D_ATTN_;
    using D_OUT  = D_ATTN_;
    using D_ACC  = float;

    static constexpr int T_M = NUM_WARPS;
    static constexpr int T_N = 1;
    static constexpr int T_K = 1;

    // QK: S^T = K . Q^T, PV: O^T = V^T . P^T
    static constexpr int W_M = 32;
    static constexpr int W_N = 32;
    static constexpr int W_K = 16;

    static constexpr int GEMM0_E_K = D_TILE_SIZE / W_K;
    static constexpr int GEMM1_E_N = D_HEAD_SIZE / W_N;
    static constexpr int GEMM1_E_K = KV_TILE_SIZE / W_K;
    static_assert(Q_TILE_SIZE == W_M && KV_TILE_SIZE == W_N);

    static constexpr int VEC_KV   = 16 / sizeof(D_ATTN);
    static constexpr int VEC_TR_V = 4;

    static constexpr int smem_brick    = 1088;
    static constexpr int brick_rows    = 8;
    static constexpr int smem_n_rpt    = KV_TILE_SIZE / brick_rows;
    static constexpr int smem_d_rpt    = D_TILE_SIZE * sizeof(D_ATTN) / 128;
    static constexpr int smem_chunk    = smem_n_rpt * smem_brick;
    static constexpr int smem_kv_bytes = smem_d_rpt * smem_chunk;
    static constexpr int KV_SLOTS = 4; // DMA 2 tiles ahead; a tile's V reads outlive its stage
    // P and rescale factors exchanged between waves, double-buffered; output scales
    static constexpr int smem_x_bytes =
        (2 * (GEMM1_E_K * WARP_SIZE * 16 + WARP_SIZE * 4) + WARP_SIZE * 4) * NUM_WARPS;
    static_assert(smem_n_rpt == NUM_WARPS, "one brick per wave in each d-chunk");

    static constexpr size_t smem_size_bytes() { return KV_SLOTS * smem_kv_bytes + smem_x_bytes; }
};

// Compile-time tile/MFMA configuration for the 16mx1_16nx4 variant (T_M=1,
// T_N=NUM_WARPS). Used when H <= 32. KV_TILE=64, NUM_WARPS=4, BLOCK_SIZE=256.
template <int Q_TILE_SIZE_  = 16,
          int KV_TILE_SIZE_ = 64,
          int D_TILE_SIZE_  = 512,
          int NUM_WARPS_    = 4,
          typename D_ATTN_  = bf16_t,
          bool SPLIT_       = false>
struct opus_mla_v4_prefill_a16w16_16mx1_16nx4_traits
{
    static constexpr int Q_TILE_SIZE  = Q_TILE_SIZE_;
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;
    static constexpr int D_TILE_SIZE  = D_TILE_SIZE_;
    static constexpr int NUM_WARPS    = NUM_WARPS_;
    static constexpr bool SPLIT       = SPLIT_;
    using kargs_type = std::conditional_t<SPLIT_,
                                          opus_mla_prefill_split_kargs<opus_mla_v4_prefill_kargs>,
                                          opus_mla_v4_prefill_kargs>;

    static constexpr int WARP_SIZE  = 64; // AMD wavefront size
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;

    using D_ATTN = D_ATTN_;
    using D_ACC  = float;

    static constexpr int T_M = 1;         // waves along M
    static constexpr int T_N = NUM_WARPS; // waves along N
    static constexpr int T_K = 1;         // waves along K

    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 32;

    static constexpr int SLICE_D      = 32;
    static constexpr int NUM_D_SLICES = D_TILE_SIZE / SLICE_D;
    static_assert(D_TILE_SIZE % SLICE_D == 0);

    static constexpr int GEMM0_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM0_E_N = KV_TILE_SIZE / (W_N * T_N);
    static constexpr int GEMM0_E_K = D_TILE_SIZE / W_K;

    static constexpr int GEMM1_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM1_E_N = D_TILE_SIZE / (W_N * T_N);
    static constexpr int GEMM1_E_K = KV_TILE_SIZE / W_K;

    static constexpr int VEC_Q    = 8;
    static constexpr int VEC_KV   = 8;
    static constexpr int VEC_P    = 4;
    static constexpr int VEC_TR_V = 4;
    static constexpr int VEC_O    = 4;

    static constexpr int D_128B_SIZE = 128 / sizeof(D_ATTN);
    static_assert(VEC_KV == 16 / sizeof(D_ATTN));
    static constexpr int smem_linear_wave   = WARP_SIZE * 16 / sizeof(D_ATTN);
    static constexpr int smem_n_per_wave    = smem_linear_wave / D_128B_SIZE;
    static constexpr int smem_n_rpt         = KV_TILE_SIZE / smem_n_per_wave;
    static constexpr int smem_d_rpt         = D_TILE_SIZE / D_128B_SIZE;
    static constexpr int smem_padding_32B   = 32 / sizeof(D_ATTN);
    static constexpr int smem_kv_tile_elems = smem_n_rpt * smem_d_rpt * (smem_linear_wave + smem_padding_32B);

    static constexpr int kv_buffer_load_insts = (KV_TILE_SIZE * D_TILE_SIZE) / (BLOCK_SIZE * VEC_KV);
    static constexpr int k_ds_read_insts = (GEMM0_E_N * GEMM0_E_K * W_N * W_K) / (WARP_SIZE * VEC_KV);
    static constexpr int v_ds_read_insts = (GEMM1_E_N * GEMM1_E_K * W_N * W_K) / (WARP_SIZE * VEC_TR_V);

    // Shared memory: the KV tile, the waves' row maxima and their fp32 P.
    static constexpr size_t smem_size_bytes()
    {
        return smem_kv_tile_elems * sizeof(D_ATTN)
             + T_N * W_M * sizeof(D_ACC)
             + T_N * W_M * W_N * sizeof(D_ACC);
    }
};

// fp8 NoPE + bf16 RoPE 32mx4_32nx1 (H > 32), laid out as the bf16 variant. K: bricks of 8 rows x
// 128 B + 32 B, 16 B pieces XOR-swizzled by bit 1 of the row; V dequantized to bf16 1088 B rows.
template <int Q_TILE_SIZE_  = 32,
          int KV_TILE_SIZE_ = 32,
          int NUM_WARPS_    = 4,
          typename D_NOPE_  = fp8_t,
          typename D_ROPE_  = bf16_t,
          typename D_OUT_   = bf16_t,
          bool SPLIT_       = false>
struct opus_mla_v4_prefill_a8w8_32mx4_32nx1_traits
{
    static constexpr int Q_TILE_SIZE  = Q_TILE_SIZE_;
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;
    static constexpr int NUM_WARPS    = NUM_WARPS_;
    static constexpr bool SPLIT       = SPLIT_;
    using kargs_type =
        std::conditional_t<SPLIT_,
                           opus_mla_prefill_split_kargs<opus_mla_v4_prefill_fp8_kargs>,
                           opus_mla_v4_prefill_fp8_kargs>;

    static constexpr int WARP_SIZE  = 64;
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;

    static constexpr int D_NOPE_SIZE        = 448;
    static constexpr int D_NOPE_PADDED_SIZE = 512;
    static constexpr int D_ROPE_SIZE        = 64;
    static constexpr int D_HEAD_SIZE        = D_NOPE_SIZE + D_ROPE_SIZE;
    static constexpr int D_SCALE_BLOCK      = 32;

    using D_NOPE = D_NOPE_;
    using D_ROPE = D_ROPE_;
    using D_ATTN = D_NOPE_;
    using D_OUT  = D_OUT_;
    using D_ACC  = float;

    static constexpr int T_M = NUM_WARPS;
    static constexpr int T_N = 1;
    static constexpr int T_K = 1;

    // QK: S^T = K . Q^T, fp8 32x32x64 (scaled) NoPE + bf16 32x32x16 RoPE; PV: bf16 32x32x16
    static constexpr int W_M      = 32;
    static constexpr int W_N      = 32;
    static constexpr int W_K_NOPE = 64;
    static constexpr int W_K_ROPE = 16;
    static constexpr int W_K_PV   = 16;

    static constexpr int GEMM0_NOPE_E_K = D_NOPE_SIZE / W_K_NOPE;
    static constexpr int GEMM0_ROPE_E_K = D_ROPE_SIZE / W_K_ROPE;
    static constexpr int GEMM1_E_N      = D_HEAD_SIZE / W_N;
    static constexpr int GEMM1_E_K      = KV_TILE_SIZE / W_K_PV;
    static_assert(Q_TILE_SIZE == W_M && KV_TILE_SIZE == W_N);

    static constexpr int VEC_KV_NOPE = 16;
    static constexpr int VEC_KV_ROPE = 8;
    static constexpr int VEC_TR_V    = 4;
    static constexpr int VEC_DEQ     = 8;

    static constexpr int smem_brick        = 1056;
    static constexpr int brick_rows        = 8;
    static constexpr int smem_n_rpt        = KV_TILE_SIZE / brick_rows;
    static constexpr int smem_d_rpt_nope   = D_NOPE_PADDED_SIZE / 128;
    static constexpr int smem_k_nope_bytes = smem_d_rpt_nope * smem_n_rpt * smem_brick;
    static constexpr int smem_k_rope_bytes = smem_n_rpt * smem_brick;
    static constexpr int smem_k_bytes      = smem_k_nope_bytes + smem_k_rope_bytes;
    static constexpr int smem_v_row        = D_HEAD_SIZE + 64 / sizeof(D_ROPE);
    static constexpr int smem_v_bytes      = KV_TILE_SIZE * smem_v_row * sizeof(D_ROPE);
    static constexpr int K_SLOTS           = 3; // the DMA runs K_SLOTS - 1 tiles ahead
    static constexpr int V_SLOTS           = 1; // V(t) is in registers before V(t + 1) is written
    // P, rescale factors and output scales exchanged between waves
    static constexpr int smem_x_bytes =
        NUM_WARPS * (GEMM1_E_K * WARP_SIZE * 16 + 2 * WARP_SIZE * 4);
    static_assert(smem_n_rpt == NUM_WARPS, "one brick per wave in each d-chunk");

    static constexpr size_t smem_size_bytes()
    {
        return K_SLOTS * smem_k_bytes + V_SLOTS * smem_v_bytes + smem_x_bytes;
    }
};

// Compile-time tile/MFMA configuration for the split-precision 16mx1_16nx4 fp8
// variant (T_M=1, T_N=NUM_WARPS). KV_TILE=64, NUM_WARPS=4, BLOCK_SIZE=256.
template <int Q_TILE_SIZE_  = 16,
          int KV_TILE_SIZE_ = 64,
          int NUM_WARPS_    = 4,
          typename D_NOPE_  = fp8_t,
          typename D_ROPE_  = bf16_t,
          typename D_OUT_   = bf16_t,
          bool SPLIT_       = false>
struct opus_mla_v4_prefill_a8w8_16mx1_16nx4_traits
{
    static constexpr int Q_TILE_SIZE  = Q_TILE_SIZE_;
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;
    static constexpr int NUM_WARPS    = NUM_WARPS_;
    static constexpr bool SPLIT       = SPLIT_;
    using kargs_type =
        std::conditional_t<SPLIT_,
                           opus_mla_prefill_split_kargs<opus_mla_v4_prefill_fp8_kargs>,
                           opus_mla_v4_prefill_fp8_kargs>;

    static constexpr int WARP_SIZE  = 64; // AMD wavefront size
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;

    // Packed DSA hdim split
    static constexpr int D_NOPE_SIZE        = 448; // NoPE fp8 elements
    static constexpr int D_NOPE_PADDED_SIZE = 512; // NoPE padded to multiple of 128
    static constexpr int D_ROPE_SIZE        = 64;  // RoPE bf16 elements
    static constexpr int D_HEAD_SIZE        = D_NOPE_SIZE + D_ROPE_SIZE; // 512

    // Data types: NoPE fp8 + RoPE bf16; accumulation fp32.
    using D_NOPE = D_NOPE_;
    using D_ROPE = D_ROPE_;
    using D_ATTN = D_NOPE_;
    using D_OUT  = D_OUT_;
    using D_ACC  = float;

    // MFMA wave layout (identical to the bf16 16mx1_16nx4 variant)
    static constexpr int T_M = 1;         // waves along M
    static constexpr int T_N = NUM_WARPS; // waves along N
    static constexpr int T_K = 1;         // waves along K

    // MFMA base tile: NoPE uses fp8 16x16x128 (scaled f8f6f4 on gfx950);
    // RoPE (bf16 QK^T) and PV (bf16) use 16x16x32.
    static constexpr int W_M      = 16;
    static constexpr int W_N      = 16;
    static constexpr int W_K_NOPE = 128;
    static constexpr int W_K_ROPE = 32;

    // GEMM0: S = Q @ K^T
    static constexpr int GEMM0_E_M      = Q_TILE_SIZE / W_M;
    static constexpr int GEMM0_E_N      = KV_TILE_SIZE / (W_N * T_N);
    static constexpr int GEMM0_NOPE_E_K = D_NOPE_PADDED_SIZE / W_K_NOPE;
    static constexpr int GEMM0_ROPE_E_K = D_ROPE_SIZE / W_K_ROPE;

    // GEMM1: O = P @ V
    static constexpr int GEMM1_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM1_E_N = D_HEAD_SIZE / (W_N * T_N);
    static constexpr int GEMM1_E_K = KV_TILE_SIZE / W_K_ROPE;

    // Vector lengths for global load/store
    static constexpr int VEC_Q_NOPE  = 16;
    static constexpr int VEC_Q_ROPE  = 8;
    static constexpr int VEC_KV_NOPE = 16;
    static constexpr int VEC_KV_ROPE = 8;
    static constexpr int VEC_P    = 4;
    static constexpr int VEC_TR_V = 4;
    static constexpr int VEC_O    = 4;

    // Per-token row stride for the bf16 KV tile staged in LDS.
    static constexpr int SMEM_KV_PAD = 8;
    static constexpr int SMEM_KV_ROW = D_HEAD_SIZE + SMEM_KV_PAD;

    // Shared memory: the KV tile, the waves' row maxima and their fp32 P.
    static constexpr size_t smem_size_bytes()
    {
        return KV_TILE_SIZE * SMEM_KV_ROW * sizeof(D_ROPE)
             + T_N * W_M * sizeof(D_ACC)
             + T_N * W_M * W_N * sizeof(D_ACC);
    }
};

__host__ __device__ inline int ceil_div(int a, int b) { return (a + b - 1) / b; }
