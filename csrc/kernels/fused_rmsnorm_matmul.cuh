#pragma once

#include <hip/hip_cooperative_groups.h>
#include <hip/hip_cooperative_groups.h>
#include <hipblaslt/hipblaslt.h>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include "../include/mhc_types.h"

namespace cg = cooperative_groups;

namespace mhc {

template<int BLOCK_SIZE>
__global__ void compute_rms_kernel(float* __restrict__ rms_out, const floatX* __restrict__ inp,
                                   int N, int C, float eps) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int idx = blockIdx.x;
    if (idx >= N)
        return;

    const floatX* x = inp + idx * C;

    extern __shared__ float shared[];

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
        shared[warp_id] = warp_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
        float block_sum = cg::reduce(warp, val, cg::plus<float>());

        if (lane_id == 0) {
            float rms = sqrtf(block_sum / (float)C + eps);
            rms_out[idx] = rms;
        }
    }
}

template<int BLOCK_SIZE>
__global__ void compute_rms_kernel_vectorized(float* __restrict__ rms_out,
                                              const floatX* __restrict__ inp, int N, int C,
                                              float eps) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int idx = blockIdx.x;
    if (idx >= N)
        return;

    const floatX* x = inp + idx * C;

    extern __shared__ float shared[];

    constexpr int VEC_SIZE = 8;
    int C_vec = C / VEC_SIZE;

    float thread_sum_sq = 0.0f;

    using vec_t = float4;
    const vec_t* x_vec = reinterpret_cast<const vec_t*>(x);

    for (int i = threadIdx.x; i < C_vec; i += BLOCK_SIZE) {
        vec_t v = x_vec[i];
        nv_bfloat162* bf_v = reinterpret_cast<nv_bfloat162*>(&v);

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
        shared[warp_id] = warp_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
        float block_sum = cg::reduce(warp, val, cg::plus<float>());

        if (lane_id == 0) {
            float rms = sqrtf(block_sum / (float)C + eps);
            rms_out[idx] = rms;
        }
    }
}

inline void compute_rms(float* rms_out, const floatX* inp, int N, int C, float eps,
                        hipStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 512;
    int num_warps = BLOCK_SIZE / 32;
    size_t shared_mem = num_warps * sizeof(float);

#ifdef MHC_ENABLE_PDL
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t config = {};
    config.numAttrs = 1;
    config.attrs = attrs;
    config.blockDim = {BLOCK_SIZE, 1, 1};
    config.gridDim = {(unsigned int)N, 1, 1};
    config.dynamicSmemBytes = shared_mem;
    config.stream = stream;

    if (C % 8 == 0 && C >= 64) {
        cudaLaunchKernelEx(&config, compute_rms_kernel_vectorized<BLOCK_SIZE>, rms_out, inp, N, C,
                           eps);
    } else {
        cudaLaunchKernelEx(&config, compute_rms_kernel<BLOCK_SIZE>, rms_out, inp, N, C, eps);
    }
#else
    if (C % 8 == 0 && C >= 64) {
        compute_rms_kernel_vectorized<BLOCK_SIZE>
            <<<N, BLOCK_SIZE, shared_mem, stream>>>(rms_out, inp, N, C, eps);
    } else {
        compute_rms_kernel<BLOCK_SIZE>
            <<<N, BLOCK_SIZE, shared_mem, stream>>>(rms_out, inp, N, C, eps);
    }
#endif
}

template<int BLOCK_SIZE>
__global__ void divide_by_rms_kernel(float* __restrict__ out, const float* __restrict__ rms, int M,
                                     int N) {
    int row = blockIdx.x;
    if (row >= M)
        return;

    float r_inv = 1.0f / rms[row];
    float* out_row = out + row * N;

    for (int i = threadIdx.x; i < N; i += BLOCK_SIZE) {
        out_row[i] *= r_inv;
    }
}

template<int BLOCK_SIZE>
__global__ void divide_by_rms_kernel_vectorized(float* __restrict__ out,
                                                const float* __restrict__ rms, int M, int N) {
    int row = blockIdx.x;
    if (row >= M)
        return;

    float r_inv = 1.0f / rms[row];
    float* out_row = out + row * N;

    constexpr int VEC_SIZE = 4;
    int N_vec = N / VEC_SIZE;

    float4* out_vec = reinterpret_cast<float4*>(out_row);

    for (int i = threadIdx.x; i < N_vec; i += BLOCK_SIZE) {
        float4 v = out_vec[i];
        v.x *= r_inv;
        v.y *= r_inv;
        v.z *= r_inv;
        v.w *= r_inv;
        out_vec[i] = v;
    }

    int remainder_start = N_vec * VEC_SIZE;
    for (int i = remainder_start + threadIdx.x; i < N; i += BLOCK_SIZE) {
        out_row[i] *= r_inv;
    }
}

inline void divide_by_rms(float* out, const float* rms, int M, int N,
                          hipStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;

#ifdef MHC_ENABLE_PDL
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t config = {};
    config.numAttrs = 1;
    config.attrs = attrs;
    config.blockDim = {BLOCK_SIZE, 1, 1};
    config.gridDim = {(unsigned int)M, 1, 1};
    config.dynamicSmemBytes = 0;
    config.stream = stream;

    if (N % 4 == 0 && N >= 16) {
        cudaLaunchKernelEx(&config, divide_by_rms_kernel_vectorized<BLOCK_SIZE>, out, rms, M, N);
    } else {
        cudaLaunchKernelEx(&config, divide_by_rms_kernel<BLOCK_SIZE>, out, rms, M, N);
    }
#else
    if (N % 4 == 0 && N >= 16) {
        divide_by_rms_kernel_vectorized<BLOCK_SIZE><<<M, BLOCK_SIZE, 0, stream>>>(out, rms, M, N);
    } else {
        divide_by_rms_kernel<BLOCK_SIZE><<<M, BLOCK_SIZE, 0, stream>>>(out, rms, M, N);
    }
#endif
}

struct MatmulDescriptors {
    hipblasLtHandle_t handle;
    hipblasLtMatmulDesc_t matmul_desc;
    hipblasLtMatrixLayout_t A_desc;
    hipblasLtMatrixLayout_t B_desc;
    hipblasLtMatrixLayout_t C_desc;
    hipblasLtMatmulPreference_t preference;
    hipblasLtMatmulHeuristicResult_t heuristic;
    void* workspace;
    size_t workspace_size;
};

inline void init_matmul_descriptors(MatmulDescriptors& desc, int M, int N, int K,
                                    size_t workspace_size = 32 * 1024 * 1024) {
    CHECK_COND(hipblasLtCreate(&desc.handle) == HIPBLAS_STATUS_SUCCESS);

    hipblasComputeType_t compute_type = HIPBLAS_COMPUTE_32F;
    hipDataType ab_type = HIP_R_16BF;
    hipDataType c_type = HIP_R_32F;
    hipDataType scale_type = HIP_R_32F;

    CHECK_COND(hipblasLtMatmulDescCreate(&desc.matmul_desc, compute_type, scale_type) == HIPBLAS_STATUS_SUCCESS);

    hipblasOperation_t trans_a = HIPBLAS_OP_T;
    hipblasOperation_t trans_b = HIPBLAS_OP_N;
    CHECK_COND(hipblasLtMatmulDescSetAttribute(desc.matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA,
                                                &trans_a, sizeof(trans_a)) == HIPBLAS_STATUS_SUCCESS);
    CHECK_COND(hipblasLtMatmulDescSetAttribute(desc.matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB,
                                                &trans_b, sizeof(trans_b)) == HIPBLAS_STATUS_SUCCESS);

    // Use column-major layouts and swap operands:
    // op(A) = B_row (N x K) via A_desc (K x N) with transA = T
    // op(B) = A_row^T (K x M) via B_desc (K x M) with transB = N
    // C_desc is (N x M) column-major, which matches row-major (M x N) layout.
    CHECK_COND(hipblasLtMatrixLayoutCreate(&desc.A_desc, ab_type, K, N, K) == HIPBLAS_STATUS_SUCCESS);
    CHECK_COND(hipblasLtMatrixLayoutCreate(&desc.B_desc, ab_type, K, M, K) == HIPBLAS_STATUS_SUCCESS);
    CHECK_COND(hipblasLtMatrixLayoutCreate(&desc.C_desc, c_type, N, M, N) == HIPBLAS_STATUS_SUCCESS);

    CHECK_COND(hipblasLtMatmulPreferenceCreate(&desc.preference) == HIPBLAS_STATUS_SUCCESS);
    CHECK_COND(hipblasLtMatmulPreferenceSetAttribute(desc.preference,
                                                      HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                      &workspace_size, sizeof(workspace_size)) == HIPBLAS_STATUS_SUCCESS);

    int returned_results = 0;
    CHECK_COND(hipblasLtMatmulAlgoGetHeuristic(
        desc.handle, desc.matmul_desc, desc.A_desc, desc.B_desc, desc.C_desc, desc.C_desc,
        desc.preference, 1, &desc.heuristic, &returned_results) == HIPBLAS_STATUS_SUCCESS);

    if (returned_results == 0) {
        fprintf(stderr, "No cuBLASLt algorithm found for row-major matmul\n");
        exit(EXIT_FAILURE);
    }

    desc.workspace_size = workspace_size;
    HIP_CALL(hipMalloc(&desc.workspace, workspace_size));
}

inline void destroy_matmul_descriptors(MatmulDescriptors& desc) {
    hipblasLtMatmulPreferenceDestroy(desc.preference);
    hipblasLtMatrixLayoutDestroy(desc.A_desc);
    hipblasLtMatrixLayoutDestroy(desc.B_desc);
    hipblasLtMatrixLayoutDestroy(desc.C_desc);
    hipblasLtMatmulDescDestroy(desc.matmul_desc);
    hipblasLtDestroy(desc.handle);
    hipFree(desc.workspace);
}

inline void matmul_forward(MatmulDescriptors& desc, float* out, const floatX* A, const floatX* B,
                           float alpha, float beta, hipStream_t stream = nullptr) {
    CHECK_COND(hipblasLtMatmul(desc.handle, desc.matmul_desc, &alpha, A, desc.A_desc, B,
                                desc.B_desc, &beta, out, desc.C_desc, out, desc.C_desc,
                                &desc.heuristic.algo, desc.workspace, desc.workspace_size, stream) == HIPBLAS_STATUS_SUCCESS);
}

struct FusedRMSNormMatmul {
    MatmulDescriptors matmul_desc;
    float* rms_buffer;
    int M, N, K;
    float eps;
    bool initialized;

    FusedRMSNormMatmul() : rms_buffer(nullptr), initialized(false) {}

    void init(int m, int n, int k, float epsilon = 1e-5f) {
        M = m;
        N = n;
        K = k;
        eps = epsilon;

        init_matmul_descriptors(matmul_desc, M, N, K);
        HIP_CALL(hipMalloc(&rms_buffer, M * sizeof(float)));
        initialized = true;
    }

    void destroy() {
        if (initialized) {
            destroy_matmul_descriptors(matmul_desc);
            hipFree(rms_buffer);
            initialized = false;
        }
    }

    void forward(float* out, const floatX* inp, const floatX* proj_weight,
                 hipStream_t stream = nullptr) {
        compute_rms(rms_buffer, inp, M, K, eps, stream);
        matmul_forward(matmul_desc, out, proj_weight, inp, 1.0f, 0.0f, stream);
        divide_by_rms(out, rms_buffer, M, N, stream);
    }

    float* get_rms_values() { return rms_buffer; }
};

template<int BLOCK_SIZE>
__global__ void compute_rms_pdl_kernel(float* __restrict__ rms_out, const floatX* __restrict__ inp,
                                       int N, int C, float eps) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int idx = blockIdx.x;
    if (idx >= N)
        return;

    const floatX* x = inp + idx * C;

    extern __shared__ float shared[];

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
        shared[warp_id] = warp_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
        float block_sum = cg::reduce(warp, val, cg::plus<float>());

        if (lane_id == 0) {
            float rms = sqrtf(block_sum / (float)C + eps);
            rms_out[idx] = rms;
        }
    }

#if __CUDA_ARCH__ >= 900
    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        cudaTriggerProgrammaticLaunchCompletion();
    }
#endif
}

inline void compute_rms_pdl(float* rms_out, const floatX* inp, int N, int C, float eps,
                            hipStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 512;
    int num_warps = BLOCK_SIZE / 32;
    size_t shared_mem = num_warps * sizeof(float);

#ifdef MHC_ENABLE_PDL
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(N);
    config.blockDim = dim3(BLOCK_SIZE);
    config.dynamicSmemBytes = shared_mem;
    config.stream = stream;
    config.attrs = attrs;
    config.numAttrs = 1;

    HIP_CALL(
        cudaLaunchKernelEx(&config, compute_rms_pdl_kernel<BLOCK_SIZE>, rms_out, inp, N, C, eps));
#else
    compute_rms_pdl_kernel<BLOCK_SIZE>
        <<<N, BLOCK_SIZE, shared_mem, stream>>>(rms_out, inp, N, C, eps);
#endif
}

struct FusedRMSNormMatmulPDL {
    MatmulDescriptors matmul_desc;
    float* rms_buffer;
    int M, N, K;
    float eps;
    bool initialized;

    FusedRMSNormMatmulPDL() : rms_buffer(nullptr), initialized(false) {}

    void init(int m, int n, int k, float epsilon = 1e-5f) {
        M = m;
        N = n;
        K = k;
        eps = epsilon;

        init_matmul_descriptors(matmul_desc, M, N, K);
        HIP_CALL(hipMalloc(&rms_buffer, M * sizeof(float)));
        initialized = true;
    }

    void destroy() {
        if (initialized) {
            destroy_matmul_descriptors(matmul_desc);
            hipFree(rms_buffer);
            initialized = false;
        }
    }

    void forward(float* out, const floatX* inp, const floatX* proj_weight,
                 hipStream_t stream = nullptr) {
        compute_rms_pdl(rms_buffer, inp, M, K, eps, stream);
        matmul_forward(matmul_desc, out, proj_weight, inp, 1.0f, 0.0f, stream);
        divide_by_rms(out, rms_buffer, M, N, stream);
    }

    float* get_rms_values() { return rms_buffer; }
};

template<int BLOCK_SIZE>
__global__ void bf16_to_fp32_kernel(float* __restrict__ out, const floatX* __restrict__ inp,
                                    int total) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= total)
        return;
    out[idx] = (float)inp[idx];
}

template<int BLOCK_SIZE>
__global__ void scale_grad_by_rms_kernel(float* __restrict__ grad_scaled,
                                         const float* __restrict__ grad,
                                         const float* __restrict__ rms, int M, int N) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int total = M * N;
    if (idx >= total)
        return;

    int row = idx / N;
    float r_inv = 1.0f / rms[row];
    grad_scaled[idx] = grad[idx] * r_inv;
}

template<int BLOCK_SIZE>
__global__ void rms_correction_kernel(float* __restrict__ dx, const float* __restrict__ K_buf,
                                      const floatX* __restrict__ x, const float* __restrict__ rms,
                                      int M, int K_dim) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int row = blockIdx.x;
    if (row >= M)
        return;

    extern __shared__ float shared[];

    const floatX* x_row = x + row * K_dim;
    const float* K_row = K_buf + row * K_dim;
    float* dx_row = dx + row * K_dim;
    float r = rms[row];

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int num_warps = BLOCK_SIZE / 32;

    float thread_dot = 0.0f;
    for (int i = threadIdx.x; i < K_dim; i += BLOCK_SIZE) {
        float xi = (float)x_row[i];
        thread_dot += K_row[i] * xi;
    }

    float warp_dot = cg::reduce(warp, thread_dot, cg::plus<float>());
    if (lane_id == 0) {
        shared[warp_id] = warp_dot;
    }
    __syncthreads();

    float K_dot_x = 0.0f;
    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
        K_dot_x = cg::reduce(warp, val, cg::plus<float>());
        if (lane_id == 0) {
            shared[0] = K_dot_x;
        }
    }
    __syncthreads();
    K_dot_x = shared[0];

    float correction_scale = K_dot_x / ((float)K_dim * r * r);

    for (int i = threadIdx.x; i < K_dim; i += BLOCK_SIZE) {
        float xi = (float)x_row[i];
        dx_row[i] = K_row[i] - correction_scale * xi;
    }
}

struct FusedRMSNormMatmulBackward {
    hipblasLtHandle_t handle;
    hipblasLtMatmulDesc_t dW_matmul_desc;
    hipblasLtMatrixLayout_t dW_grad_desc;
    hipblasLtMatrixLayout_t dW_x_desc;
    hipblasLtMatrixLayout_t dW_out_desc;
    hipblasLtMatmulPreference_t dW_pref;
    hipblasLtMatmulHeuristicResult_t dW_heuristic;

    hipblasLtMatmulDesc_t dx_matmul_desc;
    hipblasLtMatrixLayout_t dx_grad_desc;
    hipblasLtMatrixLayout_t dx_W_desc;
    hipblasLtMatrixLayout_t dx_out_desc;
    hipblasLtMatmulPreference_t dx_pref;
    hipblasLtMatmulHeuristicResult_t dx_heuristic;

    void* workspace;
    size_t workspace_size;
    float* grad_scaled_buffer;
    float* K_buffer;
    float* x_fp32_buffer;
    float* W_fp32_buffer;

    int M, N, K;
    bool initialized;

    FusedRMSNormMatmulBackward()
        : workspace(nullptr), grad_scaled_buffer(nullptr), K_buffer(nullptr),
          x_fp32_buffer(nullptr), W_fp32_buffer(nullptr), initialized(false) {}

    void init(int m, int n, int k, float epsilon = 1e-5f) {
        M = m;
        N = n;
        K = k;

        workspace_size = 32 * 1024 * 1024;
        HIP_CALL(hipMalloc(&workspace, workspace_size));
        HIP_CALL(hipMalloc(&grad_scaled_buffer, M * N * sizeof(float)));
        HIP_CALL(hipMalloc(&K_buffer, M * K * sizeof(float)));
        HIP_CALL(hipMalloc(&x_fp32_buffer, M * K * sizeof(float)));
        HIP_CALL(hipMalloc(&W_fp32_buffer, N * K * sizeof(float)));

        CHECK_COND(hipblasLtCreate(&handle) == HIPBLAS_STATUS_SUCCESS);

        CHECK_COND(hipblasLtMatmulDescCreate(&dW_matmul_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F) == HIPBLAS_STATUS_SUCCESS);
        hipblasOperation_t trans_a = HIPBLAS_OP_T;
        hipblasOperation_t trans_b = HIPBLAS_OP_N;
        CHECK_COND(hipblasLtMatmulDescSetAttribute(dW_matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA,
                                                    &trans_a, sizeof(trans_a)) == HIPBLAS_STATUS_SUCCESS);
        CHECK_COND(hipblasLtMatmulDescSetAttribute(dW_matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB,
                                                    &trans_b, sizeof(trans_b)) == HIPBLAS_STATUS_SUCCESS);

        CHECK_COND(hipblasLtMatrixLayoutCreate(&dW_grad_desc, HIP_R_32F, M, N, M) == HIPBLAS_STATUS_SUCCESS);
        CHECK_COND(hipblasLtMatrixLayoutCreate(&dW_x_desc, HIP_R_32F, M, K, M) == HIPBLAS_STATUS_SUCCESS);
        CHECK_COND(hipblasLtMatrixLayoutCreate(&dW_out_desc, HIP_R_32F, N, K, N) == HIPBLAS_STATUS_SUCCESS);

        CHECK_COND(hipblasLtMatmulPreferenceCreate(&dW_pref) == HIPBLAS_STATUS_SUCCESS);
        CHECK_COND(hipblasLtMatmulPreferenceSetAttribute(dW_pref,
                                                          HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                          &workspace_size, sizeof(workspace_size)) == HIPBLAS_STATUS_SUCCESS);

        int returned = 0;
        CHECK_COND(hipblasLtMatmulAlgoGetHeuristic(handle, dW_matmul_desc, dW_grad_desc, dW_x_desc,
                                                    dW_out_desc, dW_out_desc, dW_pref, 1,
                                                    &dW_heuristic, &returned) == HIPBLAS_STATUS_SUCCESS);
        if (returned == 0) {
            fprintf(stderr, "No cuBLASLt algorithm found for dW backward matmul\n");
            exit(EXIT_FAILURE);
        }

        CHECK_COND(hipblasLtMatmulDescCreate(&dx_matmul_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F) == HIPBLAS_STATUS_SUCCESS);
        trans_a = HIPBLAS_OP_N;
        trans_b = HIPBLAS_OP_N;
        CHECK_COND(hipblasLtMatmulDescSetAttribute(dx_matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA,
                                                    &trans_a, sizeof(trans_a)) == HIPBLAS_STATUS_SUCCESS);
        CHECK_COND(hipblasLtMatmulDescSetAttribute(dx_matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB,
                                                    &trans_b, sizeof(trans_b)) == HIPBLAS_STATUS_SUCCESS);

        CHECK_COND(hipblasLtMatrixLayoutCreate(&dx_grad_desc, HIP_R_32F, M, N, M) == HIPBLAS_STATUS_SUCCESS);
        CHECK_COND(hipblasLtMatrixLayoutCreate(&dx_W_desc, HIP_R_32F, N, K, N) == HIPBLAS_STATUS_SUCCESS);
        CHECK_COND(hipblasLtMatrixLayoutCreate(&dx_out_desc, HIP_R_32F, M, K, M) == HIPBLAS_STATUS_SUCCESS);

        CHECK_COND(hipblasLtMatmulPreferenceCreate(&dx_pref) == HIPBLAS_STATUS_SUCCESS);
        CHECK_COND(hipblasLtMatmulPreferenceSetAttribute(dx_pref,
                                                          HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                          &workspace_size, sizeof(workspace_size)) == HIPBLAS_STATUS_SUCCESS);

        returned = 0;
        CHECK_COND(hipblasLtMatmulAlgoGetHeuristic(handle, dx_matmul_desc, dx_grad_desc, dx_W_desc,
                                                    dx_out_desc, dx_out_desc, dx_pref, 1,
                                                    &dx_heuristic, &returned) == HIPBLAS_STATUS_SUCCESS);
        if (returned == 0) {
            fprintf(stderr, "No cuBLASLt algorithm found for dx backward matmul\n");
            exit(EXIT_FAILURE);
        }

        initialized = true;
    }

    void destroy() {
        if (initialized) {
            hipblasLtMatmulPreferenceDestroy(dW_pref);
            hipblasLtMatrixLayoutDestroy(dW_grad_desc);
            hipblasLtMatrixLayoutDestroy(dW_x_desc);
            hipblasLtMatrixLayoutDestroy(dW_out_desc);
            hipblasLtMatmulDescDestroy(dW_matmul_desc);

            hipblasLtMatmulPreferenceDestroy(dx_pref);
            hipblasLtMatrixLayoutDestroy(dx_grad_desc);
            hipblasLtMatrixLayoutDestroy(dx_W_desc);
            hipblasLtMatrixLayoutDestroy(dx_out_desc);
            hipblasLtMatmulDescDestroy(dx_matmul_desc);

            hipblasLtDestroy(handle);
            hipFree(workspace);
            hipFree(grad_scaled_buffer);
            hipFree(K_buffer);
            hipFree(x_fp32_buffer);
            hipFree(W_fp32_buffer);
            initialized = false;
        }
    }

    void backward(float* dW, float* dx_out, const float* grad_output, const floatX* x,
                  const floatX* weight, const float* rms, hipStream_t stream = nullptr) {
        constexpr int BLOCK_SIZE = 256;

        int total_x = M * K;
        int num_blocks_x = (total_x + BLOCK_SIZE - 1) / BLOCK_SIZE;
        bf16_to_fp32_kernel<BLOCK_SIZE>
            <<<num_blocks_x, BLOCK_SIZE, 0, stream>>>(x_fp32_buffer, x, total_x);

        int total_w = N * K;
        int num_blocks_w = (total_w + BLOCK_SIZE - 1) / BLOCK_SIZE;
        bf16_to_fp32_kernel<BLOCK_SIZE>
            <<<num_blocks_w, BLOCK_SIZE, 0, stream>>>(W_fp32_buffer, weight, total_w);

        int total = M * N;
        int num_blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
        scale_grad_by_rms_kernel<BLOCK_SIZE>
            <<<num_blocks, BLOCK_SIZE, 0, stream>>>(grad_scaled_buffer, grad_output, rms, M, N);

        float alpha = 1.0f, beta = 0.0f;
        CHECK_COND(hipblasLtMatmul(handle, dW_matmul_desc, &alpha, grad_scaled_buffer,
                                    dW_grad_desc, x_fp32_buffer, dW_x_desc, &beta, dW, dW_out_desc,
                                    dW, dW_out_desc, &dW_heuristic.algo, workspace, workspace_size,
                                    stream) == HIPBLAS_STATUS_SUCCESS);

        CHECK_COND(hipblasLtMatmul(handle, dx_matmul_desc, &alpha, grad_scaled_buffer,
                                    dx_grad_desc, W_fp32_buffer, dx_W_desc, &beta, K_buffer,
                                    dx_out_desc, K_buffer, dx_out_desc, &dx_heuristic.algo,
                                    workspace, workspace_size, stream) == HIPBLAS_STATUS_SUCCESS);

        int num_warps = BLOCK_SIZE / 32;
        size_t shared_mem = num_warps * sizeof(float);
        rms_correction_kernel<BLOCK_SIZE>
            <<<M, BLOCK_SIZE, shared_mem, stream>>>(dx_out, K_buffer, x, rms, M, K);
    }
};

} // namespace mhc
