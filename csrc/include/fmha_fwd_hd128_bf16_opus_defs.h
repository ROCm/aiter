// Shared types and constants between device kernel and host code
#pragma once

using bf16_t = __bf16;

// Kernel arguments for GQA attention
struct opus_gqa_kargs {
    const void* __restrict__ ptr_q;  // [B, N, H, D]
    const void* __restrict__ ptr_k;  // [B, N, H_KV, D]
    const void* __restrict__ ptr_v;  // [B, N, H_KV, D]
    void* __restrict__ ptr_o;        // [B, N, H, D]
    int B;
    int N;
    int H;
    int H_KV;
    int D;
    int stride_q_b;
    int stride_q_n;
    int stride_q_h;
    int stride_kv_b;
    int stride_kv_n;
    int stride_kv_h;
    float softmax_scale;  // QK^T scale (host passes 1/sqrt(D) by default)
};

// Configuration traits for GQA kernel (tile sizes, data types, vector lengths, MFMA config).
// D_TILE_SIZE selects between two kernel families:
//   D=128: MFMA 32x32x16 bf16, K/V loaded in one shot (no D slicing)
//   D=512: MFMA 16x16x32 bf16, D iterated in SLICE_D=32 chunks
template<int Q_TILE_SIZE_ = 16,
        int KV_TILE_SIZE_ = 32,
        int D_TILE_SIZE_ = 512,
        int NUM_WARPS_ = 8,
        bool CAUSAL_ = false>
struct opus_gqa_traits {
    static_assert(D_TILE_SIZE_ == 128 || D_TILE_SIZE_ == 512,
                  "opus_gqa_traits supports D_TILE_SIZE 128 or 512");

    static constexpr int Q_TILE_SIZE = Q_TILE_SIZE_;
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;
    static constexpr int D_TILE_SIZE = D_TILE_SIZE_;
    static constexpr int NUM_WARPS = NUM_WARPS_;
    static constexpr bool CAUSAL = CAUSAL_;

    static constexpr int WARP_SIZE = 64; // AMD wavefront size
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;

    // Data types: Q/K/V/O share one bf16 type; accumulation fp32
    using D_ATTN = bf16_t;
    using D_ACC  = float;

    // MFMA wave layout
    static constexpr int T_M = NUM_WARPS; // waves along M
    static constexpr int T_N = 1;         // waves along N
    static constexpr int T_K = 1;         // waves along K

    // MFMA base tile depends on D
    //   D=128: bf16 32x32x16
    //   D=512: bf16 16x16x32
    static constexpr int W_M = (D_TILE_SIZE == 128) ? 32 : 16;
    static constexpr int W_N = (D_TILE_SIZE == 128) ? 32 : 16;
    static constexpr int W_K = (D_TILE_SIZE == 128) ? 16 : 32;

    // D slicing: only D=512 iterates D in chunks; D=128 covers full D in one MMA
    static constexpr int SLICE_D = (D_TILE_SIZE == 512) ? 32 : D_TILE_SIZE;
    static constexpr int NUM_D_SLICES = D_TILE_SIZE / SLICE_D;
    static_assert(D_TILE_SIZE % SLICE_D == 0);

    // GEMM0: S[Q_TILE x KV_TILE] = Q[Q_TILE x SLICE_D] @ K^T[SLICE_D x KV_TILE]
    static constexpr int GEMM0_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM0_E_N = KV_TILE_SIZE / W_N;
    static constexpr int GEMM0_E_K = SLICE_D / W_K;

    // GEMM1: O[Q_TILE x SLICE_D] = P[Q_TILE x KV_TILE] @ V[KV_TILE x SLICE_D]
    static constexpr int GEMM1_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM1_E_N = SLICE_D / W_N;
    static constexpr int GEMM1_E_K = KV_TILE_SIZE / W_K;

    // Vector lengths for global load/store
    static constexpr int VEC_Q    = 8;
    static constexpr int VEC_KV   = 8;
    static constexpr int VEC_TR_V = 4;
    static constexpr int VEC_O    = 4;

    // Minimal compact pixels for async copy for one wave
    static constexpr int D_128B_SIZE = 128 / sizeof(D_ATTN);
    static_assert(VEC_KV == 16 / sizeof(D_ATTN));
    static constexpr int smem_linear_wave = WARP_SIZE * 16 / sizeof(D_ATTN);
    static constexpr int smem_n_per_wave = smem_linear_wave / D_128B_SIZE;
    static constexpr int smem_n_rpt = KV_TILE_SIZE / smem_n_per_wave;
    static constexpr int smem_d_rpt = D_TILE_SIZE / D_128B_SIZE;

    static constexpr int smem_padding_16B = 16 / sizeof(D_ATTN);
    static constexpr int smem_padding_32B = 32 / sizeof(D_ATTN);
    static constexpr int smem_padding_64B = 64 / sizeof(D_ATTN);

    // K/V smem padding differs across the two kernel families:
    //   D=128: K uses 16B padding, V uses 64B padding
    //   D=512: K and V both use 32B padding
    static constexpr int smem_k_padding = (D_TILE_SIZE == 128) ? smem_padding_16B : smem_padding_32B;
    static constexpr int smem_v_padding = (D_TILE_SIZE == 128) ? smem_padding_64B : smem_padding_32B;

    static constexpr int smem_k_tile_elems = smem_n_rpt * smem_d_rpt * (smem_linear_wave + smem_k_padding);
    static constexpr int smem_v_tile_elems = smem_n_rpt * smem_d_rpt * (smem_linear_wave + smem_v_padding);
    static constexpr int smem_buffer_elems = smem_k_tile_elems + smem_v_tile_elems;

    static constexpr int k_buffer_load_insts = (KV_TILE_SIZE * D_TILE_SIZE) / (BLOCK_SIZE * VEC_KV);
    static constexpr int v_buffer_load_insts = (KV_TILE_SIZE * D_TILE_SIZE) / (BLOCK_SIZE * VEC_KV);
    static constexpr int k_ds_read_insts = (GEMM0_E_N * GEMM0_E_K * W_N * W_K) / (WARP_SIZE * VEC_KV);
    static constexpr int v_ds_read_insts = (GEMM1_E_N * GEMM1_E_K * W_N * W_K) / (WARP_SIZE * VEC_TR_V);

    static constexpr size_t smem_size_bytes() {
        return 2 * smem_buffer_elems * sizeof(D_ATTN);
    }
};

__host__ __device__ inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}
