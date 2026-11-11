#pragma once

#include <cstddef>
#include <type_traits>

#include <opus/dtypes.hpp>

#include "../mla_prefill_kargs.h"

using bf16_t = opus::dtypes::bf16;
using fp16_t = opus::dtypes::fp16;
// 8-bit float storage types, aliased to match opus's dtype registration.
using fp8_t  = _BitInt(8);
using bf8_t  = unsigned _BitInt(8);

// Compile-time tile/MFMA configuration for the 16mx8_32nx1 variant (T_M=NUM_WARPS,
// T_N=1). Used when H > 32. KV_TILE=32, NUM_WARPS=8, BLOCK_SIZE=512.
template <int Q_TILE_SIZE_  = 16,
          int KV_TILE_SIZE_ = 32,
          int D_TILE_SIZE_  = 512,
          int NUM_WARPS_    = 8,
          typename D_ATTN_  = bf16_t,
          bool SPLIT_       = false>
struct opus_mla_v4_prefill_a16w16_16mx8_32nx1_traits
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

    static constexpr int T_M = NUM_WARPS;
    static constexpr int T_N = 1;
    static constexpr int T_K = 1;

    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 32;

    static constexpr int SLICE_D      = 32;
    static constexpr int NUM_D_SLICES = D_TILE_SIZE / SLICE_D;
    static_assert(D_TILE_SIZE % SLICE_D == 0);

    static constexpr int GEMM0_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM0_E_N = KV_TILE_SIZE / W_N;
    static constexpr int GEMM0_E_K = SLICE_D / W_K;

    static constexpr int GEMM1_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM1_E_N = SLICE_D / W_N;
    static constexpr int GEMM1_E_K = KV_TILE_SIZE / W_K;

    static constexpr int VEC_Q    = 8;
    static constexpr int VEC_KV   = 8;
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

    static constexpr size_t smem_size_bytes()
    {
        return 4 * smem_kv_tile_elems * sizeof(D_ATTN);
    }
};

// Compile-time tile/MFMA configuration for the 16mx1_16nx4 variant (T_M=1,
// T_N=NUM_WARPS). Used when H <= 32. KV_TILE=64, NUM_WARPS=4, BLOCK_SIZE=256.
// SPLIT (all four variants): the split-K kernel, see mla_prefill_split.h; without it the
// kernel's arguments and code are the unsplit ones.
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

    // The unsplit path exchanges P in fp32; split-K retains D_ATTN.
    static constexpr size_t smem_size_bytes()
    {
        return smem_kv_tile_elems * sizeof(D_ATTN)
             + 2 * T_N * W_M * sizeof(D_ACC)
             + T_N * W_M * W_N * (SPLIT ? sizeof(D_ATTN) : sizeof(D_ACC));
    }
};

// Compile-time tile/MFMA configuration for the split-precision 16mx8_32nx1 fp8
// variant (T_M=NUM_WARPS, T_N=1). KV_TILE=32, NUM_WARPS=8, BLOCK_SIZE=512.
template <int Q_TILE_SIZE_  = 16,
          int KV_TILE_SIZE_ = 32,
          int NUM_WARPS_    = 8,
          typename D_NOPE_  = fp8_t,
          typename D_ROPE_  = bf16_t,
          typename D_OUT_   = bf16_t,
          bool SPLIT_       = false>
struct opus_mla_v4_prefill_a8w8_16mx8_32nx1_traits
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

    // MFMA wave layout (identical to the bf16 16mx8_32nx1 variant)
    static constexpr int T_M = NUM_WARPS; // waves along M
    static constexpr int T_N = 1;         // waves along N
    static constexpr int T_K = 1;         // waves along K

    // MFMA base tile: NoPE uses fp8 16x16x128 (scaled f8f6f4 on gfx950);
    // RoPE (bf16 QK^T) and PV (bf16) use 16x16x32.
    static constexpr int W_M      = 16;
    static constexpr int W_N      = 16;
    static constexpr int W_K_NOPE = 128;
    static constexpr int W_K_ROPE = 32;

    static constexpr int SLICE_D      = 32;
    static constexpr int NUM_D_SLICES = D_HEAD_SIZE / SLICE_D;

    static constexpr int GEMM0_E_M      = Q_TILE_SIZE / W_M;
    static constexpr int GEMM0_E_N      = KV_TILE_SIZE / W_N;
    static constexpr int GEMM0_NOPE_E_K = D_NOPE_PADDED_SIZE / W_K_NOPE;
    static constexpr int GEMM0_ROPE_E_K = D_ROPE_SIZE / W_K_ROPE;

    static constexpr int GEMM1_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM1_E_N = SLICE_D / W_N;
    static constexpr int GEMM1_E_K = KV_TILE_SIZE / W_K_ROPE;

    static constexpr int VEC_Q_NOPE  = 16;
    static constexpr int VEC_Q_ROPE  = 8;
    static constexpr int VEC_KV_NOPE = 16;
    static constexpr int VEC_KV_ROPE = 8;
    static constexpr int VEC_TR_V = 4;
    static constexpr int VEC_O    = 4;

    static constexpr int D_128B_NOPE_SIZE      = 128 / sizeof(D_NOPE);
    static constexpr int dwordx4_size          = 16;
    static constexpr int smem_linear_wave_nope = WARP_SIZE * dwordx4_size / sizeof(D_NOPE);
    static constexpr int smem_n_per_wave       = 8;
    static constexpr int smem_n_rpt            = KV_TILE_SIZE / smem_n_per_wave;
    static constexpr int smem_d_rpt_nope       = D_NOPE_PADDED_SIZE / D_128B_NOPE_SIZE;
    static constexpr int smem_padding_32B_nope = 32 / sizeof(D_NOPE);
    static constexpr size_t smem_k_nope_bytes = smem_n_rpt * smem_d_rpt_nope * (smem_linear_wave_nope + smem_padding_32B_nope) * sizeof(D_NOPE);

    static constexpr int D_128B_ROPE_SIZE      = 128 / sizeof(D_ROPE);
    static constexpr int smem_linear_wave_rope = WARP_SIZE * dwordx4_size / sizeof(D_ROPE);
    static constexpr int smem_d_rpt_rope       = D_ROPE_SIZE / D_128B_ROPE_SIZE;
    static constexpr int smem_padding_32B_rope = 32 / sizeof(D_ROPE);
    static constexpr size_t smem_k_rope_bytes = smem_n_rpt * smem_d_rpt_rope * (smem_linear_wave_rope + smem_padding_32B_rope) * sizeof(D_ROPE);

    static constexpr int smem_d_rpt_head = D_HEAD_SIZE / D_128B_ROPE_SIZE;
    static constexpr size_t smem_v_bytes = smem_n_rpt * smem_d_rpt_head * (smem_linear_wave_rope + smem_padding_32B_rope) * sizeof(D_ROPE);
    static constexpr size_t smem_v_nope_bytes = smem_n_rpt * (smem_d_rpt_head - smem_d_rpt_rope) * (smem_linear_wave_rope + smem_padding_32B_rope) * sizeof(D_ROPE);

    static constexpr size_t smem_kv_bytes()
    {
        return (smem_k_nope_bytes + smem_k_rope_bytes) > smem_v_bytes
                   ? (smem_k_nope_bytes + smem_k_rope_bytes)
                   : smem_v_bytes;
    }
    static constexpr size_t smem_size_bytes()
    {
        return 4 * smem_kv_bytes();
    }

    
    static constexpr int kv_buffer_load_insts = (KV_TILE_SIZE * D_NOPE_PADDED_SIZE) / (BLOCK_SIZE * VEC_KV_NOPE)
                                              + (KV_TILE_SIZE * D_ROPE_SIZE) / (BLOCK_SIZE / 2 * VEC_KV_ROPE);
    static constexpr int k_nope_ds_read_insts = (GEMM0_E_N * W_N * W_K_NOPE) / (WARP_SIZE * VEC_KV_NOPE);
    static constexpr int k_rope_ds_read_insts = (GEMM0_E_N * W_N * W_K_ROPE) / (WARP_SIZE * VEC_KV_ROPE);
    static constexpr int v_ds_read_insts = (GEMM1_E_N * GEMM1_E_K * W_N * W_K_ROPE) / (WARP_SIZE * VEC_TR_V);
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
    static constexpr int VEC_O    = 4;

#if defined(OPUS_MLA_V4_FP8_PV) || defined(OPUS_MLA_V4_DMA)
    // V and P are staged in LDS as flat fp8 and P*V runs on the fp8 MFMA.
    using D_V  = D_NOPE_;  // LDS V tile
    using D_P  = D_NOPE_;  // cross-wave P exchange
    using D_PV = D_NOPE_;  // P*V MFMA operands
    // P <= 1 after the online-softmax max subtraction, so one constant exponent
    // packs it into e4m3's range. Any gain up to 2^8 measures identical (a
    // power-of-two scale only moves the underflow floor); 2^2 leaves headroom
    // under the 448 limit should the running max ever be stale. Folded into
    // the exp2 argument, so l_row carries it too and it cancels in O/l -- only
    // the sink term has to be scaled to match.
    static constexpr int P_PACK_LOG2 = 2;
#elif defined(OPUS_MLA_V4_FP8_V_ONLY)
    // Isolation build: fp8 V tile read back with ds_read_b64_tr_b8, converted
    // to bf16 immediately after the read so P and the P*V MFMA stay exactly as
    // they are. Exercises the new transposed-read layout and nothing else.
    using D_V  = D_NOPE_;
    using D_P  = D_ROPE_;
    using D_PV = D_ROPE_;
    static constexpr int P_PACK_LOG2 = 0;
#else
    using D_V  = D_ROPE_;
    using D_P  = D_ROPE_;
    using D_PV = D_ROPE_;
    static constexpr int P_PACK_LOG2 = 0;
#endif
    // DS_READ_B64_TR moves 8 bytes/lane whatever the element width, so an fp8
    // tile yields 8 elements per instruction instead of 4. LANE_LO_V halves to
    // keep make_layout_rv's grp_n = W_N / (LANE_LO_V * VEC_TR_V) equal to 1.
    static constexpr bool V_IS_FP8 = (sizeof(D_V) == 1);
    static constexpr int VEC_TR_V  = V_IS_FP8 ? 8 : 4;
    static constexpr int LANE_LO_V = V_IS_FP8 ? 2 : 4;
    // Flat exponent the per-32-element MX block scales are collapsed onto for V.
    // Tunable: the only knob trading V's dynamic range against underflow.
    static constexpr int V_SCALE_LOG2 = 0;

    // Per-token row stride for the KV tile staged in LDS, in D_V elements.
    static constexpr int SMEM_KV_PAD = 8;
    static constexpr int SMEM_KV_ROW = D_HEAD_SIZE + SMEM_KV_PAD;

#if defined(OPUS_MLA_V4_DMA)
    // DMA-staged tile. global_load_lds writes one wave-uniform, lane-contiguous
    // 64 x 16 B = 1024 B block per instruction (M0 + lane*16), so the tile is a
    // sequence of such chunks, each holding 8 rows x 128 B of one D slice, plus
    // 32 B of padding so successive chunks start on different banks.
    static constexpr int DMA_CHUNK_STRIDE = WARP_SIZE * 16 + 32;            // 1056
    static constexpr int DMA_ROWS_PER_CHUNK = 8;
    static constexpr int DMA_D_SLICES = D_NOPE_PADDED_SIZE / 128;           // 4
    static constexpr int DMA_HALVES = KV_TILE_SIZE / (DMA_ROWS_PER_CHUNK * NUM_WARPS); // 2
    static constexpr int SMEM_KV_BYTES = DMA_HALVES * DMA_D_SLICES * NUM_WARPS * DMA_CHUNK_STRIDE;
    // Tile n+1 streams into one buffer while tile n is read from the other.
    // 2 x 33 KB keeps two workgroups per CU inside 160 KB.
    static constexpr int DMA_BUFFERS = 2;
#else
    static constexpr int SMEM_KV_BYTES = KV_TILE_SIZE * SMEM_KV_ROW * sizeof(D_V);
    static constexpr int DMA_BUFFERS = 1;
#endif

    // Shared memory: kernel uses three static buffers (KV tiles, m/l, P).
    static constexpr size_t smem_size_bytes()
    {
        return DMA_BUFFERS * SMEM_KV_BYTES
             + 2 * T_N * W_M * sizeof(D_ACC)
             + T_N * W_M * W_N * sizeof(D_P);
    }
};

__host__ __device__ inline int ceil_div(int a, int b) { return (a + b - 1) / b; }
