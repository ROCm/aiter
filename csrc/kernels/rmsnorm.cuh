#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_cooperative_groups.h>
#include <hip/hip_cooperative_groups.h>
#include "../include/mhc_layer.h"

namespace cg = cooperative_groups;

namespace mhc {

template<int BLOCK_SIZE, bool OUTPUT_RMS = false>
__global__ void rmsnorm_kernel(__hip_bfloat16* __restrict__ out, float* __restrict__ rms_out,
                               const __hip_bfloat16* __restrict__ inp, const __hip_bfloat16* __restrict__ weight,
                               int N, int C, float eps) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int idx = blockIdx.x;
    if (idx >= N)
        return;

    const __hip_bfloat16* x = inp + idx * C;
    __hip_bfloat16* o = out + idx * C;

    extern __shared__ float shared[];
    float* s_sum_sq = shared;

    float thread_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < C; i += BLOCK_SIZE) {
        float val = (float)x[i];
        thread_sum_sq += val * val;
    }

    float warp_sum = cg::reduce(warp, thread_sum_sq, cg::plus<float>());

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int num_warps = BLOCK_SIZE / 32;

    if (lane_id == 0) {
        s_sum_sq[warp_id] = warp_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? s_sum_sq[lane_id] : 0.0f;
        float block_sum = cg::reduce(warp, val, cg::plus<float>());

        if (lane_id == 0) {
            float rms = sqrtf(block_sum / (float)C + eps);
            float rms_inv = 1.0f / rms;
            s_sum_sq[0] = rms_inv;
            if constexpr (OUTPUT_RMS) {
                rms_out[idx] = rms;
            }
        }
    }
    __syncthreads();

    float rms_inv = s_sum_sq[0];

    for (int i = threadIdx.x; i < C; i += BLOCK_SIZE) {
        float val = (float)x[i];
        float w = (float)weight[i];
        o[i] = (__hip_bfloat16)(val * rms_inv * w);
    }
}

template<int BLOCK_SIZE, bool OUTPUT_RMS = false>
__global__ void rmsnorm_kernel_vectorized(__hip_bfloat16* __restrict__ out, float* __restrict__ rms_out,
                                          const __hip_bfloat16* __restrict__ inp,
                                          const __hip_bfloat16* __restrict__ weight, int N, int C,
                                          float eps) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int idx = blockIdx.x;
    if (idx >= N)
        return;

    const __hip_bfloat16* x = inp + idx * C;
    __hip_bfloat16* o = out + idx * C;

    extern __shared__ float shared[];
    float* s_sum_sq = shared;

    constexpr int VEC_SIZE = 8;
    int C_vec = C / VEC_SIZE;

    float thread_sum_sq = 0.0f;

    using vec_t = float4;
    const vec_t* x_vec = reinterpret_cast<const vec_t*>(x);

    for (int i = threadIdx.x; i < C_vec; i += BLOCK_SIZE) {
        vec_t v = x_vec[i];
        __hip_bfloat162* bf_v = reinterpret_cast<__hip_bfloat162*>(&v);

#pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 f = __bfloat1622float2(bf_v[j]);
            thread_sum_sq += f.x * f.x + f.y * f.y;
        }
    }

    int remainder_start = C_vec * VEC_SIZE;
    for (int i = remainder_start + threadIdx.x; i < C; i += BLOCK_SIZE) {
        float val = (float)x[i];
        thread_sum_sq += val * val;
    }

    float warp_sum = cg::reduce(warp, thread_sum_sq, cg::plus<float>());

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int num_warps = BLOCK_SIZE / 32;

    if (lane_id == 0) {
        s_sum_sq[warp_id] = warp_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? s_sum_sq[lane_id] : 0.0f;
        float block_sum = cg::reduce(warp, val, cg::plus<float>());

        if (lane_id == 0) {
            float rms = sqrtf(block_sum / (float)C + eps);
            float rms_inv = 1.0f / rms;
            s_sum_sq[0] = rms_inv;
            if constexpr (OUTPUT_RMS) {
                rms_out[idx] = rms;
            }
        }
    }
    __syncthreads();

    float rms_inv = s_sum_sq[0];

    vec_t* o_vec = reinterpret_cast<vec_t*>(o);
    const vec_t* w_vec = reinterpret_cast<const vec_t*>(weight);

    for (int i = threadIdx.x; i < C_vec; i += BLOCK_SIZE) {
        vec_t xv = x_vec[i];
        vec_t wv = w_vec[i];

        __hip_bfloat162* bf_x = reinterpret_cast<__hip_bfloat162*>(&xv);
        __hip_bfloat162* bf_w = reinterpret_cast<__hip_bfloat162*>(&wv);

        vec_t ov;
        __hip_bfloat162* bf_o = reinterpret_cast<__hip_bfloat162*>(&ov);

#pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 xf = __bfloat1622float2(bf_x[j]);
            float2 wf = __bfloat1622float2(bf_w[j]);
            float2 of;
            of.x = xf.x * rms_inv * wf.x;
            of.y = xf.y * rms_inv * wf.y;
            bf_o[j] = __float22bfloat162_rn(of);
        }

        o_vec[i] = ov;
    }

    for (int i = remainder_start + threadIdx.x; i < C; i += BLOCK_SIZE) {
        float val = (float)x[i];
        float w = (float)weight[i];
        o[i] = (__hip_bfloat16)(val * rms_inv * w);
    }
}

inline void rmsnorm_forward(__hip_bfloat16* out, const __hip_bfloat16* inp, const __hip_bfloat16* weight, int N, int C,
                            float eps, hipStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 512;
    int num_warps = BLOCK_SIZE / 32;
    size_t shared_mem = num_warps * sizeof(float);

    dim3 grid(N);
    dim3 block(BLOCK_SIZE);

    if (C % 8 == 0 && C >= 64) {
        rmsnorm_kernel_vectorized<BLOCK_SIZE, false>
            <<<grid, block, shared_mem, stream>>>(out, nullptr, inp, weight, N, C, eps);
    } else {
        rmsnorm_kernel<BLOCK_SIZE, false>
            <<<grid, block, shared_mem, stream>>>(out, nullptr, inp, weight, N, C, eps);
    }
}

inline void rmsnorm_forward_with_rms(__hip_bfloat16* out, float* rms_out, const __hip_bfloat16* inp,
                                     const __hip_bfloat16* weight, int N, int C, float eps,
                                     hipStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 512;
    int num_warps = BLOCK_SIZE / 32;
    size_t shared_mem = num_warps * sizeof(float);

    dim3 grid(N);
    dim3 block(BLOCK_SIZE);

    if (C % 8 == 0 && C >= 64) {
        rmsnorm_kernel_vectorized<BLOCK_SIZE, true>
            <<<grid, block, shared_mem, stream>>>(out, rms_out, inp, weight, N, C, eps);
    } else {
        rmsnorm_kernel<BLOCK_SIZE, true>
            <<<grid, block, shared_mem, stream>>>(out, rms_out, inp, weight, N, C, eps);
    }
}
template<int BLOCK_SIZE>
__global__ void rmsnorm_backward_kernel(float* __restrict__ d_inp, float* __restrict__ d_weight,
                                        const float* __restrict__ grad,
                                        const __hip_bfloat16* __restrict__ inp,
                                        const __hip_bfloat16* __restrict__ weight,
                                        const float* __restrict__ rms, int N, int C) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int idx = blockIdx.x;
    if (idx >= N)
        return;

    const __hip_bfloat16* x = inp + idx * C;
    const float* g = grad + idx * C;
    float* dx = d_inp + idx * C;
    float r = rms[idx];
    float r_inv = 1.0f / r;

    extern __shared__ float shared[];
    float* s_reduce = shared;

    float thread_dot = 0.0f;
    for (int i = threadIdx.x; i < C; i += BLOCK_SIZE) {
        float g_val = g[i];
        float w_val = (float)weight[i];
        float x_val = (float)x[i];
        thread_dot += g_val * w_val * x_val;
    }

    float warp_dot = cg::reduce(warp, thread_dot, cg::plus<float>());

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int num_warps = BLOCK_SIZE / 32;

    if (lane_id == 0) {
        s_reduce[warp_id] = warp_dot;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? s_reduce[lane_id] : 0.0f;
        float block_dot = cg::reduce(warp, val, cg::plus<float>());
        if (lane_id == 0) {
            s_reduce[0] = block_dot;
        }
    }
    __syncthreads();

    float dot_sum = s_reduce[0];
    float correction = dot_sum / ((float)C * r * r);

    for (int i = threadIdx.x; i < C; i += BLOCK_SIZE) {
        float g_val = g[i];
        float w_val = (float)weight[i];
        float x_val = (float)x[i];

        dx[i] = (g_val * w_val * r_inv) - (x_val * correction * r_inv);

        atomicAdd(&d_weight[i], g_val * x_val * r_inv);
    }
}

inline void rmsnorm_backward(float* d_inp, float* d_weight, const float* grad, const __hip_bfloat16* inp,
                             const __hip_bfloat16* weight, const float* rms, int N, int C,
                             hipStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 512;
    int num_warps = BLOCK_SIZE / 32;
    size_t shared_mem = num_warps * sizeof(float);

    rmsnorm_backward_kernel<BLOCK_SIZE>
        <<<N, BLOCK_SIZE, shared_mem, stream>>>(d_inp, d_weight, grad, inp, weight, rms, N, C);
}

} // namespace mhc
