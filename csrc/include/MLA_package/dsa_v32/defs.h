#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

using bf16_t = __bf16;
using fp8_t  = _BitInt(8);
using bf8_t  = unsigned _BitInt(8);

inline fp8_t float_to_fp8_e4m3(float f) {
    const uint32_t bits = __builtin_bit_cast(uint32_t, f);
    const uint8_t  sign = static_cast<uint8_t>((bits >> 24) & 0x80u);
    const uint32_t mag  = bits & 0x7FFFFFFFu;

    if (mag >= 0x7F800000u) return __builtin_bit_cast(fp8_t, static_cast<uint8_t>(sign | 0x7Fu));
    if (mag >= 0x43E00000u) return __builtin_bit_cast(fp8_t, static_cast<uint8_t>(sign | 0x7Eu));

    const int exp = static_cast<int>(mag >> 23) - 127;
    uint8_t payload;
    uint32_t dropped, halfway;
    if (exp >= -6) {
        const uint32_t mant = mag & 0x7FFFFFu;
        payload = static_cast<uint8_t>(((exp + 7) << 3) | (mant >> 20));
        dropped = mant & 0xFFFFFu;
        halfway = 0x80000u;
    } else {
        const int shift = -exp - 6;
        if (shift > 11) return __builtin_bit_cast(fp8_t, sign);
        const uint32_t mant  = (mag & 0x7FFFFFu) | 0x800000u;
        const uint32_t width = 20 + shift;
        payload = static_cast<uint8_t>(mant >> width);
        dropped = mant & ((1u << width) - 1);
        halfway = 1u << (width - 1);
    }
    if (dropped > halfway || (dropped == halfway && (payload & 1))) payload++;
    return __builtin_bit_cast(fp8_t, static_cast<uint8_t>(sign | payload));
}

inline float fp8_e4m3_to_float(fp8_t v) {
    const uint8_t  bits = __builtin_bit_cast(uint8_t, v);
    const uint32_t sign = static_cast<uint32_t>(bits & 0x80u) << 24;
    const uint32_t exp  = (bits >> 3) & 0x0Fu;
    const uint32_t mant = bits & 0x07u;

    if (exp == 0x0Fu && mant == 0x07u) return __builtin_bit_cast(float, sign | 0x7FC00000u);
    if (exp == 0) {
        const float m = static_cast<float>(mant) * 0x1p-9f;
        return (bits & 0x80u) ? -m : m;
    }
    return __builtin_bit_cast(float, sign | ((exp + 120) << 23) | (mant << 20));
}

static constexpr float MLA_DECODE_LN_2 = 0.69314718055994531f;
static constexpr float MLA_DECODE_LOG2_E = 1.4426950408889634f;

static constexpr int MLA_DECODE_NUM_CU = 256;
static constexpr int MLA_DECODE_KV_GRANULARITY = 16;
static constexpr int MLA_DECODE_FIXED_OVERHEAD = 16;
static constexpr int MLA_DECODE_PACKED_QO_LEN_PER_WG = 128;
static constexpr int MLA_DECODE_MAX_SPLIT_PER_BATCH = 32;
static constexpr float MLA_DECODE_SPLIT_COEF = 1.2f;

struct opus_mla_decode_work_info {
    int batch_idx;
    int partial_slot;
    int qo_start;
    int qo_end;
    int kv_start;
    int kv_end;
    int kv_offset;
    int _pad;
};

struct opus_mla_decode_mxfp8_kargs {
    const void* __restrict__ q_nope_ptr;
    const void* __restrict__ q_scale_ptr;
    const void* __restrict__ q_rope_ptr;
    const void* __restrict__ kv_nope_ptr;
    const void* __restrict__ kv_scale_ptr;
    const void* __restrict__ kv_rope_ptr;
    void* __restrict__ out_ptr;
    void* __restrict__ lse_ptr;
    void* __restrict__ o_accum;
    void* __restrict__ lse_accum;

    const int* __restrict__ q_indptr;
    const int* __restrict__ kv_indptr;
    const int* __restrict__ kv_indices;
    const int* __restrict__ work_indptr;
    const opus_mla_decode_work_info* __restrict__ work_info_set;

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

struct opus_mla_decode_kargs {
    const void* __restrict__ q_ptr;
    const void* __restrict__ kv_ptr;
    void* __restrict__ out_ptr;
    void* __restrict__ lse_ptr;
    void* __restrict__ o_accum;
    void* __restrict__ lse_accum;

    const int* __restrict__ q_indptr;
    const int* __restrict__ kv_indptr;
    const int* __restrict__ kv_indices;
    const int* __restrict__ work_indptr;
    const opus_mla_decode_work_info* __restrict__ work_info_set;

    int H;
    int total_tokens;
    int stride_q_b;
    int stride_q_h;
    int stride_o_b;
    int stride_o_h;
    int stride_kv_page;
    float softmax_scale;
};

// Harness-only: in production aiter owns both stages, so these are not ABI.
struct opus_mla_decode_metadata_kargs {
    const int* __restrict__ qo_indptr;
    const int* __restrict__ kv_indptr;

    int* __restrict__ work_indptr;
    opus_mla_decode_work_info* __restrict__ work_info_set;
    int* __restrict__ reduce_indptr;
    int* __restrict__ reduce_final_map;
    int* __restrict__ reduce_partial_map;

    int B;
    int H;
    int num_cu;
    int num_splits;
    int uni_seqlen_qo;
    int kv_granularity;
    int fixed_overhead;
    int tail_done_threshold;
    int reduce_indptr_size;
    int auto_split;
    int is_causal;
};

struct opus_mla_decode_reduce_kargs {
    const void* __restrict__ o_accum;
    const void* __restrict__ lse_accum;
    void* __restrict__ out_ptr;
    void* __restrict__ lse_ptr;

    const int* __restrict__ reduce_indptr;
    const int* __restrict__ reduce_final_map;
    const int* __restrict__ reduce_partial_map;

    int H;
    int stride_o_b;
    int stride_o_h;
};

// Frozen: the prebuilt code objects are compiled against these layouts.
static_assert(sizeof(opus_mla_decode_work_info) == 32);
static_assert(sizeof(opus_mla_decode_kargs) == 120);
static_assert(offsetof(opus_mla_decode_kargs, work_info_set) == 80);
static_assert(offsetof(opus_mla_decode_kargs, softmax_scale) == 116);
static_assert(sizeof(opus_mla_decode_mxfp8_kargs) == 176);
static_assert(offsetof(opus_mla_decode_mxfp8_kargs, work_info_set) == 112);
static_assert(offsetof(opus_mla_decode_mxfp8_kargs, softmax_scale) == 172);

template<int Q_TILE_SIZE_ = 16,
         int KV_TILE_SIZE_ = 32,
         int NUM_WARPS_ = 8,
         typename D_NOPE_ = fp8_t,
         typename D_ROPE_ = bf16_t,
         typename D_OUT_ = bf16_t>
struct opus_mla_decode_mxfp8_16mx8_32nx1_traits {
    static constexpr int Q_TILE_SIZE = Q_TILE_SIZE_;
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;
    static constexpr int NUM_WARPS = NUM_WARPS_;

    static constexpr int WARP_SIZE = 64;
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;

    static constexpr int D_NOPE_SIZE = 512;
    static constexpr int D_ROPE_SIZE = 64;
    static constexpr int D_HEAD_SIZE = D_NOPE_SIZE + D_ROPE_SIZE;
    static constexpr int D_SCALE_SIZE = D_NOPE_SIZE / 32;
    static constexpr int D_SCALE_PADDED_SIZE = 32;

    static constexpr int D_QK_SIZE = D_HEAD_SIZE;
    static constexpr int D_VO_SIZE  = D_NOPE_SIZE;

    using D_NOPE = D_NOPE_;
    using D_ROPE = D_ROPE_;
    using D_OUT  = D_OUT_;
    using D_ACC  = float;

    static constexpr int T_M = NUM_WARPS;
    static constexpr int T_N = 1;
    static constexpr int T_K = 1;

    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K_NOPE = 128;
    static constexpr int W_K_ROPE = 32;

    static constexpr int SLICE_D = 32;
    static constexpr int NUM_D_SLICES = D_NOPE_SIZE / SLICE_D;

    static constexpr int GEMM0_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM0_E_N = KV_TILE_SIZE / W_N;
    static constexpr int GEMM0_NOPE_E_K = D_NOPE_SIZE / W_K_NOPE;
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

    static constexpr int D_128B_NOPE_SIZE = 128 / sizeof(D_NOPE);
    static constexpr int dwordx4_size = 16;
    static constexpr int smem_linear_wave_nope = WARP_SIZE * dwordx4_size / sizeof(D_NOPE);
    static constexpr int smem_n_per_wave = 8;
    static constexpr int smem_n_rpt = KV_TILE_SIZE / smem_n_per_wave;
    static constexpr int smem_d_rpt_nope = D_NOPE_SIZE / D_128B_NOPE_SIZE;
    static constexpr int smem_padding_32B_nope = 32 / sizeof(D_NOPE);
    static constexpr size_t smem_k_nope_bytes = smem_n_rpt * smem_d_rpt_nope * (smem_linear_wave_nope + smem_padding_32B_nope) * sizeof(D_NOPE);

    static constexpr int D_128B_ROPE_SIZE = 128 / sizeof(D_ROPE);
    static constexpr int smem_linear_wave_rope = WARP_SIZE * dwordx4_size / sizeof(D_ROPE);
    static constexpr int smem_d_rpt_rope = D_ROPE_SIZE / D_128B_ROPE_SIZE;
    static constexpr int smem_padding_32B_rope = 32 / sizeof(D_ROPE);
    static constexpr size_t smem_k_rope_bytes = smem_n_rpt * smem_d_rpt_rope * (smem_linear_wave_rope + smem_padding_32B_rope) * sizeof(D_ROPE);

    static constexpr int smem_v_padding = 32 / sizeof(D_ROPE);
    static constexpr size_t smem_v_bytes = KV_TILE_SIZE * (D_NOPE_SIZE + smem_v_padding) * sizeof(D_ROPE);

    static constexpr int smem_mxscl_padding = 4 / sizeof(D_NOPE);
    static constexpr size_t smem_mxscl_bytes = smem_n_rpt * (D_SCALE_PADDED_SIZE * smem_n_per_wave + smem_mxscl_padding) * sizeof(D_NOPE);

    static constexpr size_t smem_kv_bytes() {
        return std::max(smem_k_nope_bytes + smem_k_rope_bytes, smem_v_bytes);
    }

    static constexpr int kv_buffer_load_insts = (KV_TILE_SIZE * D_NOPE_SIZE) / (BLOCK_SIZE * VEC_KV_NOPE)  // nope = 2
                                                + 1; // rope = 1 for warp_id < 4 or mxscl = 1 for warp_id >= 4
    static constexpr int k_nope_ds_read_insts = (GEMM0_E_N * W_N * W_K_NOPE) / (WARP_SIZE * VEC_KV_NOPE);
    static constexpr int k_rope_ds_read_insts = (GEMM0_E_N * W_N * W_K_ROPE) / (WARP_SIZE * VEC_KV_ROPE);
    static constexpr int v_ds_read_insts = (GEMM1_E_N * GEMM1_E_K * W_N * W_K_ROPE) / (WARP_SIZE * VEC_TR_V);
};

template<int Q_TILE_SIZE_ = 16,
         int KV_TILE_SIZE_ = 64,
         int NUM_WARPS_ = 4,
         typename D_ATTN_ = bf16_t,
         typename D_OUT_ = bf16_t>
struct opus_mla_decode_a16w16_16mx4_64nx1_traits {
    static constexpr int Q_TILE_SIZE = Q_TILE_SIZE_;
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;
    static constexpr int NUM_WARPS = NUM_WARPS_;

    static constexpr int WARP_SIZE = 64;
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;

    static constexpr int D_QK_SIZE = 576;
    static constexpr int D_VO_SIZE  = 512;

    using D_ATTN = D_ATTN_;
    using D_OUT  = D_OUT_;
    using D_ACC  = float;

    static constexpr int T_M = NUM_WARPS;
    static constexpr int T_N = 1;
    static constexpr int T_K = 1;

    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 32;

    static constexpr int GEMM0_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM0_E_N = KV_TILE_SIZE / (W_N * T_N);
    static constexpr int GEMM0_E_K = D_QK_SIZE / W_K;

    static constexpr int SLICE_D = 32;
    static constexpr int NUM_D_SLICES = D_VO_SIZE / SLICE_D;

    static constexpr int GEMM1_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM1_E_N = SLICE_D / W_N;
    static constexpr int GEMM1_E_K = KV_TILE_SIZE / W_K;

    static constexpr int VEC_Q = 8;
    static constexpr int VEC_KV = 8;
    static constexpr int VEC_TR_V = 4;
    static constexpr int VEC_O = 4;

    static constexpr int dwordx4_size = 16;
    static constexpr int D_128B_SIZE = 128 / sizeof(D_ATTN);
    static constexpr int smem_linear_wave = WARP_SIZE * dwordx4_size / sizeof(D_ATTN);
    static constexpr int smem_n_per_wave = smem_linear_wave / D_128B_SIZE;
    static constexpr int smem_n_rpt = KV_TILE_SIZE / smem_n_per_wave;
    static constexpr int smem_d_rpt = D_QK_SIZE / D_128B_SIZE;
    static constexpr int smem_d_rpt_v = D_VO_SIZE / D_128B_SIZE;
    static constexpr int smem_padding_32B = 32 / sizeof(D_ATTN);
    static constexpr int smem_brick = smem_linear_wave + smem_padding_32B;

    static constexpr int smem_n_sub_tile = smem_n_per_wave * NUM_WARPS;
    static constexpr int smem_n_sub_tile_rpt = KV_TILE_SIZE / smem_n_sub_tile;
    static constexpr int kv_async_load_insts = smem_n_sub_tile_rpt * smem_d_rpt;

    static constexpr size_t smem_kv_bytes = (size_t)smem_n_rpt * smem_d_rpt * smem_brick * sizeof(D_ATTN);

    static constexpr int NUM_KV_BUFS = 2;
    static constexpr int smem_slot_elems = (int)(smem_kv_bytes / sizeof(D_ATTN));
    static constexpr size_t smem_bytes() { return NUM_KV_BUFS * smem_kv_bytes; }
};

template<int Q_TILE_SIZE_ = 32,
         int KV_TILE_SIZE_ = 64,
         int NUM_WARPS_ = 4,
         typename D_ATTN_ = bf16_t,
         typename D_OUT_ = bf16_t>
struct opus_mla_decode_a16w16_32mx1_16nx4_traits {
    static constexpr int Q_TILE_SIZE = Q_TILE_SIZE_;
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;
    static constexpr int NUM_WARPS = NUM_WARPS_;

    static constexpr int WARP_SIZE = 64;
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;

    static constexpr int D_QK_SIZE = 576;
    static constexpr int D_VO_SIZE  = 512;

    using D_ATTN = D_ATTN_;
    using D_OUT  = D_OUT_;
    using D_ACC  = float;

    static constexpr int T_M = 1;
    static constexpr int T_N = NUM_WARPS;
    static constexpr int T_K = 1;

    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 32;

    static constexpr int GEMM0_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM0_E_N = KV_TILE_SIZE / (W_N * T_N);
    static constexpr int GEMM0_E_K = D_QK_SIZE / W_K;

    static constexpr int GEMM1_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM1_E_N = D_VO_SIZE / (W_N * T_N);
    static constexpr int GEMM1_E_K = KV_TILE_SIZE / W_K;

    static constexpr int VEC_Q = 8;
    static constexpr int VEC_KV = 8;
    static constexpr int VEC_TR_V = 4;
    static constexpr int VEC_O = 4;

    static constexpr int dwordx4_size = 16;
    static constexpr int D_128B_SIZE = 128 / sizeof(D_ATTN);
    static constexpr int smem_linear_wave = WARP_SIZE * dwordx4_size / sizeof(D_ATTN);
    static constexpr int smem_n_per_wave = smem_linear_wave / D_128B_SIZE;
    static constexpr int smem_n_rpt = KV_TILE_SIZE / smem_n_per_wave;
    static constexpr int smem_d_rpt = D_QK_SIZE / D_128B_SIZE;
    static constexpr int smem_d_rpt_v = D_VO_SIZE / D_128B_SIZE;
    static constexpr int smem_padding_32B = 32 / sizeof(D_ATTN);
    static constexpr int smem_brick = smem_linear_wave + smem_padding_32B;

    static constexpr int smem_n_sub_tile = smem_n_per_wave * NUM_WARPS;
    static constexpr int smem_n_sub_tile_rpt = KV_TILE_SIZE / smem_n_sub_tile;
    static constexpr int kv_async_load_insts = smem_n_sub_tile_rpt * smem_d_rpt;

    static constexpr size_t smem_kv_bytes = (size_t)smem_n_rpt * smem_d_rpt * smem_brick * sizeof(D_ATTN);

    static constexpr int NUM_KV_BUFS = 2;
    static constexpr int smem_slot_elems = (int)(smem_kv_bytes / sizeof(D_ATTN));

    static constexpr int ML_ELEMS = GEMM0_E_M;
    static constexpr int ML_SLOT_ELEMS = GEMM0_E_M * W_M * T_N;
    static constexpr int P_SLOT_ELEMS = Q_TILE_SIZE * KV_TILE_SIZE;
    static constexpr size_t smem_ml_bytes = 2 * (size_t)ML_SLOT_ELEMS * sizeof(D_ACC);
    static constexpr size_t smem_p_bytes  = 2 * (size_t)P_SLOT_ELEMS * sizeof(D_ATTN);

    static constexpr size_t smem_bytes() {
        return NUM_KV_BUFS * smem_kv_bytes + smem_ml_bytes + smem_p_bytes;
    }
};

template<int Q_TILE_SIZE_, int KV_TILE_SIZE_, int NUM_WARPS_, int NUM_COMPUTE_WARPS_,
         typename D_ATTN_, typename D_OUT_>
struct opus_mla_decode_a16w16_32mxt_32nx1_traits_base {
    static constexpr int Q_TILE_SIZE = Q_TILE_SIZE_;
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;
    static constexpr int NUM_WARPS = NUM_WARPS_;
    static constexpr int NUM_COMPUTE_WARPS = NUM_COMPUTE_WARPS_;

    static constexpr int WARP_SIZE = 64;
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;

    static constexpr int D_QK_SIZE = 576;
    static constexpr int D_VO_SIZE  = 512;

    using D_ATTN = D_ATTN_;
    using D_OUT  = D_OUT_;
    using D_ACC  = float;

    static constexpr int T_M = NUM_COMPUTE_WARPS;
    static constexpr int T_N = 1;
    static constexpr int T_K = 1;

    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 32;

    static constexpr int GEMM0_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM0_E_N = KV_TILE_SIZE / (W_N * T_N);
    static constexpr int GEMM0_E_K = D_QK_SIZE / W_K;

    static constexpr int GEMM1_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM1_E_N = D_VO_SIZE / (W_N * T_N);
    static constexpr int GEMM1_E_K = KV_TILE_SIZE / W_K;

    static constexpr int VEC_Q = 8;
    static constexpr int VEC_KV = 8;
    static constexpr int VEC_TR_V = 4;
    static constexpr int VEC_O = 4;

    static constexpr int dwordx4_size = 16;
    static constexpr int D_128B_SIZE = 128 / sizeof(D_ATTN);
    static constexpr int smem_linear_wave = WARP_SIZE * dwordx4_size / sizeof(D_ATTN);
    static constexpr int smem_n_per_wave = smem_linear_wave / D_128B_SIZE;
    static constexpr int smem_n_rpt = KV_TILE_SIZE / smem_n_per_wave;
    static constexpr int smem_d_rpt = D_QK_SIZE / D_128B_SIZE;
    static constexpr int smem_d_rpt_v = D_VO_SIZE / D_128B_SIZE;
    static constexpr int smem_padding_32B = 32 / sizeof(D_ATTN);
    static constexpr int smem_brick = smem_linear_wave + smem_padding_32B;

    static constexpr int smem_n_sub_tile = smem_n_per_wave * NUM_WARPS;
    static constexpr int smem_n_sub_tile_rpt = KV_TILE_SIZE / smem_n_sub_tile;
    static constexpr int kv_async_load_insts = smem_n_sub_tile_rpt * smem_d_rpt;

    static constexpr size_t smem_kv_bytes = (size_t)smem_n_rpt * smem_d_rpt * smem_brick * sizeof(D_ATTN);

    static constexpr int NUM_KV_BUFS = 4;
    static constexpr int smem_slot_elems = (int)(smem_kv_bytes / sizeof(D_ATTN));

    static constexpr int ML_ELEMS = GEMM0_E_M;

    static constexpr size_t smem_bytes() { return NUM_KV_BUFS * smem_kv_bytes; }
};

template<int Q_TILE_SIZE_ = 32, int KV_TILE_SIZE_ = 32, int NUM_WARPS_ = 4,
         typename D_ATTN_ = bf16_t, typename D_OUT_ = bf16_t>
struct opus_mla_decode_a16w16_32mx4_32nx1_traits
    : opus_mla_decode_a16w16_32mxt_32nx1_traits_base<
          Q_TILE_SIZE_, KV_TILE_SIZE_, NUM_WARPS_, NUM_WARPS_, D_ATTN_, D_OUT_> {};

template<int Q_TILE_SIZE_ = 32, int KV_TILE_SIZE_ = 32, int NUM_WARPS_ = 4,
         typename D_ATTN_ = bf16_t, typename D_OUT_ = bf16_t>
struct opus_mla_decode_a16w16_32mx3_32nx1_traits
    : opus_mla_decode_a16w16_32mxt_32nx1_traits_base<
          Q_TILE_SIZE_, KV_TILE_SIZE_, NUM_WARPS_, NUM_WARPS_ - 1, D_ATTN_, D_OUT_> {};

template<typename T>
__host__ __device__ inline T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

__host__ __device__ inline int mla_decode_num_qo_tiles(int seqlen_qo, int H) {
    if (seqlen_qo * H <= MLA_DECODE_PACKED_QO_LEN_PER_WG) return 1;
    if (H * 2 > MLA_DECODE_PACKED_QO_LEN_PER_WG) return seqlen_qo;
    return ceil_div(seqlen_qo * H, MLA_DECODE_PACKED_QO_LEN_PER_WG);
}
