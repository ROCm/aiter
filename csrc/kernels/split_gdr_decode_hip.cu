// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <torch/extension.h>

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

#include "split_gdr_decode_hip.h"

namespace aiter {

__device__ __forceinline__ float hip_softplus(
    float x, float beta, float inv_beta, float threshold) {
    const float beta_x = beta * x;
    return (beta_x <= threshold) ? inv_beta * logf(1.0f + expf(beta_x)) : x;
}

__device__ __forceinline__ float hip_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ void load_bf16x2(
    float* __restrict__ smem,
    const __hip_bfloat16* __restrict__ src,
    int base,
    int K,
    int BK) {
    if (base + 1 < K) {
        const uint32_t packed = *reinterpret_cast<const uint32_t*>(src + base);
        const __hip_bfloat16* v = reinterpret_cast<const __hip_bfloat16*>(&packed);
        smem[base] = static_cast<float>(v[0]);
        smem[base + 1] = static_cast<float>(v[1]);
    } else if (base < K) {
        smem[base] = static_cast<float>(src[base]);
        if (base + 1 < BK) {
            smem[base + 1] = 0.0f;
        }
    } else {
        if (base < BK) {
            smem[base] = 0.0f;
        }
        if (base + 1 < BK) {
            smem[base + 1] = 0.0f;
        }
    }
}

template <int CHUNK>
__device__ __forceinline__ int lds_padded_idx(int idx) {
    return idx + idx / CHUNK;
}

template <int CHUNK>
__device__ __forceinline__ void load_bf16x2_padded_fast(
    float* __restrict__ smem,
    const __hip_bfloat16* __restrict__ src,
    int base) {
    const int p0 = lds_padded_idx<CHUNK>(base);
    const int p1 = lds_padded_idx<CHUNK>(base + 1);
    const uint32_t packed = *reinterpret_cast<const uint32_t*>(src + base);
    const __hip_bfloat16* v = reinterpret_cast<const __hip_bfloat16*>(&packed);
    smem[p0] = static_cast<float>(v[0]);
    smem[p1] = static_cast<float>(v[1]);
}

template <int BK, int BV, bool USE_INITIAL_STATE, bool USE_QK_L2NORM>
__global__ __launch_bounds__(BV) void fused_split_gdr_update_kernel_ksplit4_db(
    const __hip_bfloat16* __restrict__ mixed_qkv,
    const float* __restrict__ A_log,
    const __hip_bfloat16* __restrict__ a,
    const __hip_bfloat16* __restrict__ dt_bias,
    float softplus_beta,
    float softplus_threshold,
    const __hip_bfloat16* __restrict__ b_gate,
    __hip_bfloat16* __restrict__ o,
    float* __restrict__ h0_source,
    const int32_t* __restrict__ h0_indices,
    int T,
    int key_dim,
    int value_dim,
    int64_t stride_x_batch,
    int64_t stride_x_dim,
    int64_t stride_x_seq,
    int64_t stride_o_batch,
    int64_t stride_o_seq,
    int64_t stride_o_head,
    int64_t stride_o_dim,
    int B,
    int H,
    int HV,
    int K,
    int V_dim,
    float scale) {
    static_assert(BK % 4 == 0, "BK must be divisible by 4");
    static_assert(BK >= 16, "BK must be >= 16 for K_SPLIT=4");

    constexpr int BK_QTR = BK / 4;
    constexpr int BK4_QTR = BK_QTR / 4;
    constexpr int BV_OUT = BV / 4;
    constexpr int SMEM_PAD = (BK_QTR == 32) ? 1 : 0;
    constexpr int SMEM_STRIDE = BK_QTR + SMEM_PAD;
    constexpr int SMEM_SIZE = SMEM_STRIDE * 4;

    const int i_v = blockIdx.y;
    const int i_nh = blockIdx.z;
    const int i_n = i_nh / HV;
    const int i_hv = i_nh % HV;
    const int i_h = i_hv / (HV / H);
    const int lane = threadIdx.x;

    const int v_idx = lane & (BV_OUT - 1);
    const int k_split = lane >> 4;
    const int v_col = i_v * BV_OUT + v_idx;
    const int k_start_logical = k_split * BK_QTR;
    const int k_start = k_split * SMEM_STRIDE;

    __shared__ float smem_k[SMEM_SIZE];
    __shared__ float smem_q[SMEM_SIZE];

    const float a_log_val = A_log[i_hv];
    const float dt_bias_val = static_cast<float>(dt_bias[i_hv]);

    float4 h_vec[BK4_QTR];
#pragma unroll
    for (int i = 0; i < BK4_QTR; i++) {
        h_vec[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    if constexpr (USE_INITIAL_STATE) {
        const int32_t idx = h0_indices[i_n];
        if (idx >= 0) {
            const int K4 = K / 4;
            const float4* h0_base = reinterpret_cast<const float4*>(
                h0_source + static_cast<int64_t>(idx) * HV * K4 * V_dim * 4 +
                static_cast<int64_t>(i_hv) * K4 * V_dim * 4);
            const int kg_start = k_start_logical / 4;
#pragma unroll
            for (int i = 0; i < BK4_QTR; i++) {
                h_vec[i] = h0_base[(kg_start + i) * V_dim + v_col];
            }
        }
    }

    const float neg_exp_A_log = -expf(a_log_val);
    const float inv_softplus_beta = 1.0f / softplus_beta;

    const int q_dim_off = i_h * K;
    const int k_dim_off = key_dim + i_h * K;
    const int v_dim_off = 2 * key_dim + i_hv * V_dim;

    const __hip_bfloat16* x_base =
        mixed_qkv + static_cast<int64_t>(i_n) * stride_x_batch;
    const __hip_bfloat16* p_a = a + static_cast<int64_t>(i_n) * T * HV + i_hv;
    const __hip_bfloat16* p_b =
        b_gate + static_cast<int64_t>(i_n) * T * HV + i_hv;
    __hip_bfloat16* p_o = o + static_cast<int64_t>(i_n) * stride_o_batch +
                          static_cast<int64_t>(i_hv) * stride_o_head;

    const bool use_vec2 = (stride_x_dim == 1);

    for (int t = 0; t < T; t++) {
        const __hip_bfloat16* x_t = x_base + static_cast<int64_t>(t) * stride_x_seq;

        if (use_vec2) {
            const int base = 2 * lane;
            if constexpr (SMEM_PAD) {
                load_bf16x2_padded_fast<BK_QTR>(smem_k, x_t + k_dim_off, base);
                load_bf16x2_padded_fast<BK_QTR>(smem_q, x_t + q_dim_off, base);
            } else {
                load_bf16x2(smem_k, x_t + k_dim_off, base, K, BK);
                load_bf16x2(smem_q, x_t + q_dim_off, base, K, BK);
            }
        } else {
            for (int i = lane; i < BK; i += BV) {
                const float q_val = static_cast<float>(
                    x_t[static_cast<int64_t>(q_dim_off + i) * stride_x_dim]);
                const float k_val = static_cast<float>(
                    x_t[static_cast<int64_t>(k_dim_off + i) * stride_x_dim]);
                if constexpr (SMEM_PAD) {
                    smem_q[i + i / BK_QTR] = q_val;
                    smem_k[i + i / BK_QTR] = k_val;
                } else {
                    smem_q[i] = q_val;
                    smem_k[i] = k_val;
                }
            }
        }

        const __hip_bfloat16 v_raw =
            x_t[static_cast<int64_t>(v_dim_off + v_col) * stride_x_dim];
        const __hip_bfloat16 a_raw = p_a[static_cast<int64_t>(t) * HV];
        const __hip_bfloat16 b_raw = p_b[static_cast<int64_t>(t) * HV];

        __syncthreads();

        float4 k_cache[BK4_QTR];
        float k_inv_norm = 1.0f;
        float dot_partial;
        {
            float dp = 0.0f;
            float k_sq = 0.0f;

            float pf0 = smem_k[k_start], pf1 = smem_k[k_start + 1];
            float pf2 = smem_k[k_start + 2], pf3 = smem_k[k_start + 3];

#pragma unroll
            for (int i = 0; i < BK4_QTR; i++) {
                const float s0 = pf0, s1 = pf1, s2 = pf2, s3 = pf3;

                if (i + 1 < BK4_QTR) {
                    const int k0_next = k_start + (i + 1) * 4;
                    pf0 = smem_k[k0_next];
                    pf1 = smem_k[k0_next + 1];
                    pf2 = smem_k[k0_next + 2];
                    pf3 = smem_k[k0_next + 3];
                }

                k_cache[i] = make_float4(s0, s1, s2, s3);
                dp += h_vec[i].x * s0 + h_vec[i].y * s1 + h_vec[i].z * s2 +
                      h_vec[i].w * s3;
                if constexpr (USE_QK_L2NORM) {
                    k_sq += s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3;
                }
            }
            dp += __shfl_xor(dp, 16, 64);
            dp += __shfl_xor(dp, 32, 64);
            if constexpr (USE_QK_L2NORM) {
                k_sq += __shfl_xor(k_sq, 16, 64);
                k_sq += __shfl_xor(k_sq, 32, 64);
                k_inv_norm = rsqrtf(k_sq + 1e-6f);
            }
            dot_partial = dp;
        }

        float v_local = static_cast<float>(v_raw);
        const float a_t = static_cast<float>(a_raw);
        const float b_t = static_cast<float>(b_raw);

        const float sp = hip_softplus(
            a_t + dt_bias_val, softplus_beta, inv_softplus_beta, softplus_threshold);
        const float g = neg_exp_A_log * sp;
        const float exp_g = expf(g);
        const float beta = hip_sigmoid(b_t);

        v_local -= dot_partial * exp_g * k_inv_norm;
        v_local *= beta;

        {
            const float kv = k_inv_norm * v_local;
#pragma unroll
            for (int i = 0; i < BK4_QTR; i++) {
                h_vec[i].x = h_vec[i].x * exp_g + k_cache[i].x * kv;
                h_vec[i].y = h_vec[i].y * exp_g + k_cache[i].y * kv;
                h_vec[i].z = h_vec[i].z * exp_g + k_cache[i].z * kv;
                h_vec[i].w = h_vec[i].w * exp_g + k_cache[i].w * kv;
            }
        }

        {
            float out_partial = 0.0f;
            float q_sq = 0.0f;

            float qf0 = smem_q[k_start], qf1 = smem_q[k_start + 1];
            float qf2 = smem_q[k_start + 2], qf3 = smem_q[k_start + 3];

#pragma unroll
            for (int i = 0; i < BK4_QTR; i++) {
                const float s0 = qf0, s1 = qf1, s2 = qf2, s3 = qf3;

                if (i + 1 < BK4_QTR) {
                    const int k0_next = k_start + (i + 1) * 4;
                    qf0 = smem_q[k0_next];
                    qf1 = smem_q[k0_next + 1];
                    qf2 = smem_q[k0_next + 2];
                    qf3 = smem_q[k0_next + 3];
                }

                out_partial += h_vec[i].x * s0 + h_vec[i].y * s1 + h_vec[i].z * s2 +
                               h_vec[i].w * s3;
                if constexpr (USE_QK_L2NORM) {
                    q_sq += s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3;
                }
            }
            out_partial += __shfl_xor(out_partial, 16, 64);
            out_partial += __shfl_xor(out_partial, 32, 64);
            float q_inv_norm = 1.0f;
            if constexpr (USE_QK_L2NORM) {
                q_sq += __shfl_xor(q_sq, 16, 64);
                q_sq += __shfl_xor(q_sq, 32, 64);
                q_inv_norm = rsqrtf(q_sq + 1e-6f);
            }
            const float o_local = out_partial * q_inv_norm * scale;

            if (k_split == 0) {
                p_o[static_cast<int64_t>(t) * stride_o_seq +
                    static_cast<int64_t>(v_col) * stride_o_dim] =
                    static_cast<__hip_bfloat16>(o_local);
            }
        }
    }

    if constexpr (USE_INITIAL_STATE) {
        const int32_t idx = h0_indices[i_n];
        if (idx >= 0) {
            const int K4 = K / 4;
            float4* h0_base = reinterpret_cast<float4*>(
                h0_source + static_cast<int64_t>(idx) * HV * K4 * V_dim * 4 +
                static_cast<int64_t>(i_hv) * K4 * V_dim * 4);
            const int kg_start = k_start_logical / 4;
#pragma unroll
            for (int i = 0; i < BK4_QTR; i++) {
                h0_base[(kg_start + i) * V_dim + v_col] = h_vec[i];
            }
        }
    }
}

#define LAUNCH_KS(BK_CT, USE_INIT, USE_L2)                                           \
    hipLaunchKernelGGL((fused_split_gdr_update_kernel_ksplit4_db<BK_CT, BV_VAL,      \
                                                                 USE_INIT, USE_L2>),  \
        dim3(grid),                                                                    \
        dim3(block),                                                                   \
        0,                                                                             \
        stream,                                                                        \
        reinterpret_cast<const __hip_bfloat16*>(mixed_qkv.data_ptr()),                \
        A_log.data_ptr<float>(),                                                       \
        reinterpret_cast<const __hip_bfloat16*>(a.data_ptr()),                        \
        reinterpret_cast<const __hip_bfloat16*>(dt_bias.data_ptr()),                  \
        softplus_beta,                                                                 \
        softplus_threshold,                                                            \
        reinterpret_cast<const __hip_bfloat16*>(b_gate.data_ptr()),                   \
        reinterpret_cast<__hip_bfloat16*>(o.data_ptr()),                              \
        use_initial_state ? initial_state_source.data_ptr<float>() : nullptr,         \
        initial_state_indices_ptr.data_ptr<int32_t>(),                                \
        T,                                                                             \
        key_dim,                                                                       \
        value_dim,                                                                     \
        stride_x_batch,                                                                \
        stride_x_dim,                                                                  \
        stride_x_seq,                                                                  \
        stride_o_batch,                                                                \
        stride_o_seq,                                                                  \
        stride_o_head,                                                                 \
        stride_o_dim,                                                                  \
        B,                                                                             \
        H,                                                                             \
        HV,                                                                            \
        K,                                                                             \
        V,                                                                             \
        scale)

#define DISPATCH_KS_BOOL(BK_CT)                                      \
    if (use_initial_state && use_qk_l2norm_in_kernel) {              \
        LAUNCH_KS(BK_CT, true, true);                                \
    } else if (use_initial_state && !use_qk_l2norm_in_kernel) {      \
        LAUNCH_KS(BK_CT, true, false);                               \
    } else if (!use_initial_state && use_qk_l2norm_in_kernel) {      \
        LAUNCH_KS(BK_CT, false, true);                               \
    } else {                                                         \
        LAUNCH_KS(BK_CT, false, false);                              \
    }

torch::Tensor fused_split_gdr_update(
    torch::Tensor mixed_qkv,
    torch::Tensor A_log,
    torch::Tensor a,
    torch::Tensor dt_bias,
    torch::Tensor b_gate,
    torch::Tensor initial_state_source,
    torch::Tensor initial_state_indices,
    int key_dim,
    int value_dim,
    int num_heads_qk,
    int num_heads_v,
    int head_dim,
    float softplus_beta,
    float softplus_threshold,
    float scale,
    bool use_qk_l2norm_in_kernel,
    c10::optional<torch::Tensor> output) {
    TORCH_CHECK(mixed_qkv.is_cuda(), "mixed_qkv must be CUDA/HIP tensor");
    TORCH_CHECK(A_log.is_cuda(), "A_log must be CUDA/HIP tensor");
    TORCH_CHECK(a.is_cuda(), "a must be CUDA/HIP tensor");
    TORCH_CHECK(dt_bias.is_cuda(), "dt_bias must be CUDA/HIP tensor");
    TORCH_CHECK(b_gate.is_cuda(), "b_gate must be CUDA/HIP tensor");
    TORCH_CHECK(mixed_qkv.scalar_type() == torch::kBFloat16, "mixed_qkv must be bfloat16");
    TORCH_CHECK(A_log.scalar_type() == torch::kFloat32, "A_log must be float32");
    TORCH_CHECK(a.scalar_type() == torch::kBFloat16, "a must be bfloat16");
    TORCH_CHECK(dt_bias.scalar_type() == torch::kBFloat16, "dt_bias must be bfloat16");
    TORCH_CHECK(b_gate.scalar_type() == torch::kBFloat16, "b_gate must be bfloat16");
    TORCH_CHECK(mixed_qkv.dim() == 3, "mixed_qkv must be 3-D (B, dim, T)");

    const int B = mixed_qkv.size(0);
    const int dim = mixed_qkv.size(1);
    const int T = mixed_qkv.size(2);
    const int H = num_heads_qk;
    const int HV = num_heads_v;
    const int K = head_dim;
    const int V = head_dim;

    TORCH_CHECK(H > 0 && HV > 0, "num_heads_qk/num_heads_v must be > 0");
    TORCH_CHECK(HV >= H, "num_heads_v must be >= num_heads_qk");
    TORCH_CHECK(HV % H == 0, "num_heads_v must be divisible by num_heads_qk");
    TORCH_CHECK(dim == 2 * key_dim + value_dim, "mixed_qkv dim mismatch");
    TORCH_CHECK(K % 4 == 0, "head_dim must be divisible by 4");
    TORCH_CHECK(A_log.numel() == HV, "A_log shape mismatch");
    TORCH_CHECK(dt_bias.numel() == HV, "dt_bias shape mismatch");
    TORCH_CHECK(a.numel() == static_cast<int64_t>(B) * T * HV, "a shape mismatch");
    TORCH_CHECK(
        b_gate.numel() == static_cast<int64_t>(B) * T * HV,
        "b_gate shape mismatch");

    bool use_initial_state = initial_state_source.defined() && initial_state_source.numel() > 0;
    if (use_initial_state) {
        TORCH_CHECK(initial_state_source.is_cuda(), "initial_state_source must be CUDA/HIP tensor");
        TORCH_CHECK(initial_state_source.scalar_type() == torch::kFloat32, "initial_state_source must be float32");
        TORCH_CHECK(initial_state_source.dim() == 5, "initial_state_source must be 5-D swizzled tensor");
        TORCH_CHECK(initial_state_source.size(1) == HV, "initial_state_source HV mismatch");
        TORCH_CHECK(initial_state_source.size(2) * 4 == K, "initial_state_source K/4 mismatch");
        TORCH_CHECK(initial_state_source.size(3) == V, "initial_state_source V mismatch");
        TORCH_CHECK(initial_state_source.size(4) == 4, "initial_state_source last dim must be 4");
        TORCH_CHECK(
            initial_state_indices.defined() && initial_state_indices.numel() == B,
            "initial_state_indices must be provided with shape [B]");
        TORCH_CHECK(initial_state_indices.is_cuda(), "initial_state_indices must be CUDA/HIP tensor");
        TORCH_CHECK(initial_state_indices.scalar_type() == torch::kInt32, "initial_state_indices must be int32");
    }

    if (scale <= 0.0f) {
        scale = 1.0f / std::sqrt(static_cast<float>(K));
    }

    torch::Tensor o;
    if (output.has_value()) {
        o = output.value();
        TORCH_CHECK(o.is_cuda(), "output must be CUDA/HIP tensor");
        TORCH_CHECK(o.scalar_type() == torch::kBFloat16, "output must be bfloat16");
        TORCH_CHECK(
            o.size(0) == B && o.size(1) == T && o.size(2) == HV && o.size(3) == V,
            "output shape mismatch");
    } else {
        o = torch::empty({B, T, HV, V}, mixed_qkv.options());
    }

    torch::Tensor initial_state_indices_ptr = initial_state_indices;
    if (!initial_state_indices_ptr.defined() || initial_state_indices_ptr.numel() == 0) {
        initial_state_indices_ptr = torch::zeros({B}, mixed_qkv.options().dtype(torch::kInt32));
    }

    int bk_runtime = 1;
    while (bk_runtime < K) {
        bk_runtime <<= 1;
    }
    TORCH_CHECK((K + bk_runtime - 1) / bk_runtime == 1, "NK > 1 unsupported");

    constexpr int BV_VAL = 64;
    constexpr int BV_OUT = BV_VAL / 4;
    dim3 grid(1, (V + BV_OUT - 1) / BV_OUT, B * HV);
    dim3 block(BV_VAL);

    const int64_t stride_x_batch = mixed_qkv.stride(0);
    const int64_t stride_x_dim = mixed_qkv.stride(1);
    const int64_t stride_x_seq = mixed_qkv.stride(2);
    const int64_t stride_o_batch = o.stride(0);
    const int64_t stride_o_seq = o.stride(1);
    const int64_t stride_o_head = o.stride(2);
    const int64_t stride_o_dim = o.stride(3);
    auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();

    if (bk_runtime == 128) {
        DISPATCH_KS_BOOL(128);
    } else if (bk_runtime == 64) {
        DISPATCH_KS_BOOL(64);
    } else if (bk_runtime == 256) {
        DISPATCH_KS_BOOL(256);
    } else if (bk_runtime == 32) {
        DISPATCH_KS_BOOL(32);
    } else {
        TORCH_CHECK(false, "Unsupported BK: ", bk_runtime);
    }

    return o;
}

#undef DISPATCH_KS_BOOL
#undef LAUNCH_KS

} // namespace aiter
