#pragma once

#include <algorithm>

using bf16_t = __bf16;
using fp16_t = __fp16;
using fp8_t  = _BitInt(8);
using bf8_t  = unsigned _BitInt(8);

static constexpr int DSA_V32_NUM_CU         = 256;
static constexpr int DSA_V32_FIXED_OVERHEAD = 5;

struct alignas(16) DsaSchedMeta
{
    int begin_req_idx;
    int end_req_idx;
    int begin_tile_idx;
    int end_tile_idx;
    int begin_split_idx;
    int _pad[3];
};

struct dsa_kargs
{
    const void* __restrict__ q_nope_ptr;
    const void* __restrict__ q_scale_ptr;
    const void* __restrict__ q_rope_ptr;
    const void* __restrict__ kv_nope_ptr;
    const void* __restrict__ kv_scale_ptr;
    const void* __restrict__ kv_rope_ptr;
    void* __restrict__ o_accum;
    void* __restrict__ lse_accum;
    void* __restrict__ out_ptr;
    void* __restrict__ lse_ptr;
    const int* __restrict__ q_indptr;
    const int* __restrict__ kv_indptr;
    const int* __restrict__ kv_indices;

    const int* __restrict__ work_indptr;
    const int* __restrict__ work_info_set;

    int B;
    int H;
    int total_tokens;
    int stride_q_nope_b;
    int stride_q_nope_h;
    int stride_q_scale_b;
    int stride_q_scale_h;
    int stride_q_rope_b;
    int stride_q_rope_h;
    int stride_o_b;
    int stride_o_h;
    int stride_kv_nope_page;
    int stride_kv_scale_page;
    int stride_kv_rope_page;
    float softmax_scale;
};

template <int Q_TILE_SIZE_  = 16,
          int KV_TILE_SIZE_ = 32,
          int NUM_WARPS_    = 8,
          typename D_NOPE_  = fp8_t,
          typename D_ROPE_  = bf16_t,
          typename D_OUT_   = bf16_t>
struct dsa_v32_16mx8_32nx1_fp8_traits
{
    static constexpr int Q_TILE_SIZE  = Q_TILE_SIZE_;
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;
    static constexpr int NUM_WARPS    = NUM_WARPS_;

    static constexpr int WARP_SIZE  = 64;
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;

    static constexpr int D_NOPE_SIZE         = 512;
    static constexpr int D_ROPE_SIZE         = 64;
    static constexpr int D_HEAD_SIZE         = D_NOPE_SIZE + D_ROPE_SIZE;
    static constexpr int D_SCALE_SIZE        = D_NOPE_SIZE / 32;
    static constexpr int D_SCALE_PADDED_SIZE = 32;

    using D_NOPE = D_NOPE_;
    using D_ROPE = D_ROPE_;
    using D_OUT  = D_OUT_;
    using D_ACC  = float;

    static constexpr int T_M = NUM_WARPS;
    static constexpr int T_N = 1;
    static constexpr int T_K = 1;

    static constexpr int W_M      = 16;
    static constexpr int W_N      = 16;
    static constexpr int W_K_NOPE = 128;
    static constexpr int W_K_ROPE = 32;

    static constexpr int SLICE_D      = 32;
    static constexpr int NUM_D_SLICES = D_NOPE_SIZE / SLICE_D;

    static constexpr int GEMM0_E_M      = Q_TILE_SIZE / W_M;
    static constexpr int GEMM0_E_N      = KV_TILE_SIZE / W_N;
    static constexpr int GEMM0_NOPE_E_K = D_NOPE_SIZE / W_K_NOPE;
    static constexpr int GEMM0_ROPE_E_K = D_ROPE_SIZE / W_K_ROPE;

    static constexpr int GEMM1_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM1_E_N = SLICE_D / W_N;
    static constexpr int GEMM1_E_K = KV_TILE_SIZE / W_K_ROPE;

    static constexpr int VEC_Q_NOPE  = 16;
    static constexpr int VEC_Q_ROPE  = 8;
    static constexpr int VEC_KV_NOPE = 16;
    static constexpr int VEC_KV_ROPE = 8;
    static constexpr int VEC_TR_V    = 4;
    static constexpr int VEC_O       = 4;

    static constexpr int D_128B_NOPE_SIZE      = 128 / sizeof(D_NOPE);
    static constexpr int dwordx4_size          = 16;
    static constexpr int smem_linear_wave_nope = WARP_SIZE * dwordx4_size / sizeof(D_NOPE);
    static constexpr int smem_n_per_wave       = 8;
    static constexpr int smem_n_rpt            = KV_TILE_SIZE / smem_n_per_wave;
    static constexpr int smem_d_rpt_nope       = D_NOPE_SIZE / D_128B_NOPE_SIZE;
    static constexpr int smem_padding_32B_nope = 32 / sizeof(D_NOPE);
    static constexpr size_t smem_k_nope_bytes  = smem_n_rpt * smem_d_rpt_nope *
                                                (smem_linear_wave_nope + smem_padding_32B_nope) *
                                                sizeof(D_NOPE);

    static constexpr int D_128B_ROPE_SIZE      = 128 / sizeof(D_ROPE);
    static constexpr int smem_linear_wave_rope = WARP_SIZE * dwordx4_size / sizeof(D_ROPE);
    static constexpr int smem_d_rpt_rope       = D_ROPE_SIZE / D_128B_ROPE_SIZE;
    static constexpr int smem_padding_32B_rope = 32 / sizeof(D_ROPE);
    static constexpr size_t smem_k_rope_bytes  = smem_n_rpt * smem_d_rpt_rope *
                                                (smem_linear_wave_rope + smem_padding_32B_rope) *
                                                sizeof(D_ROPE);

    static constexpr int smem_v_padding = 32 / sizeof(D_ROPE);
    static constexpr size_t smem_v_bytes =
        KV_TILE_SIZE * (D_NOPE_SIZE + smem_v_padding) * sizeof(D_ROPE);

    static constexpr int smem_mxscl_padding = 4 / sizeof(D_NOPE);
    static constexpr size_t smem_mxscl_bytes =
        smem_n_rpt * (D_SCALE_PADDED_SIZE * smem_n_per_wave + smem_mxscl_padding) * sizeof(D_NOPE);

    static constexpr size_t smem_kv_bytes()
    {
        return std::max(smem_k_nope_bytes + smem_k_rope_bytes, smem_v_bytes);
    }

    static constexpr int kv_buffer_load_insts =
        (KV_TILE_SIZE * D_NOPE_SIZE) / (BLOCK_SIZE * VEC_KV_NOPE) // nope = 2
        + 1; // rope = 1 for warp_id < 4 or mxscl = 1 for warp_id >= 4
    static constexpr int k_nope_ds_read_insts =
        (GEMM0_E_N * W_N * W_K_NOPE) / (WARP_SIZE * VEC_KV_NOPE);
    static constexpr int k_rope_ds_read_insts =
        (GEMM0_E_N * W_N * W_K_ROPE) / (WARP_SIZE * VEC_KV_ROPE);
    static constexpr int v_ds_read_insts =
        (GEMM1_E_N * GEMM1_E_K * W_N * W_K_ROPE) / (WARP_SIZE * VEC_TR_V);
};

__host__ __device__ inline int ceil_div(int a, int b) { return (a + b - 1) / b; }
