// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#define CHECK_HIP(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "[HIP ERROR] " << #call << " failed: " << hipGetErrorString(err) << std::endl; \
            return torch::Tensor(); \
        } \
    } while (0)

void print_tensor_shape(const char* name, const torch::Tensor& tensor) {
    auto sizes = tensor.sizes();
    printf("%s shape: [", name);
    for (size_t i = 0; i < sizes.size(); ++i) {
        printf("%ld", sizes[i]);
        if (i != sizes.size() - 1) printf(", ");
    }
    printf("]\n");
}

__global__ void mla_decode_hip_kernel(
    const __hip_bfloat16* __restrict__ q,
    const __hip_bfloat16* __restrict__ kv,
    __hip_bfloat16* __restrict__ out,
    const int* __restrict__ kv_indptr,
    const int* __restrict__ kv_indices,
    int D, int H, int B,
    int S,
    int max_tile
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;
    int idx = b * H + h;

    const __hip_bfloat16* q_ptr = q + idx * D;
    __hip_bfloat16* out_ptr = out + idx * D;

    int start = kv_indptr[b];
    int end = kv_indptr[b + 1];
    int len = end - start;

    extern __shared__ float shared_mem[];
    float* scores = shared_mem;
    float* acc = shared_mem + max_tile;
    float* max_buf = shared_mem + max_tile + D;

    for (int d = tid; d < D; d += blockDim.x)
        acc[d] = 0.0f;
    __syncthreads();

    float local_max = -1e9f;
    for (int tile_start = 0; tile_start < len; tile_start += max_tile) {
        int tile_len = min(max_tile, len - tile_start);
        for (int i = tid; i < tile_len; i += blockDim.x) {
            int global_idx = start + tile_start + i;
            int kv_page = kv_indices[global_idx];
            int slot_id = global_idx % S;
            int flat_idx = ((kv_page * S + slot_id) * H + h) * D;
            const __hip_bfloat16* k_ptr = kv + flat_idx;

            float dot = 0.0f;
            for (int d = 0; d < D; ++d)
                dot += __bfloat162float(q_ptr[d]) * __bfloat162float(k_ptr[d]);
            scores[i] = dot;
            local_max = fmaxf(local_max, dot);
        }
        __syncthreads();
    }

    if (tid < blockDim.x)
        max_buf[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            max_buf[tid] = fmaxf(max_buf[tid], max_buf[tid + stride]);
        __syncthreads();
    }
    float e_max = max_buf[0];

    __shared__ float e_sum_shared;
    if (tid == 0) e_sum_shared = 0.0f;
    __syncthreads();

    for (int tile_start = 0; tile_start < len; tile_start += max_tile) {
        int tile_len = min(max_tile, len - tile_start);
        for (int i = tid; i < tile_len; i += blockDim.x) {
            int global_idx = start + tile_start + i;
            int kv_page = kv_indices[global_idx];
            int slot_id = global_idx % S;
            int flat_idx = ((kv_page * S + slot_id) * H + h) * D;
            const __hip_bfloat16* k_ptr = kv + flat_idx;
            const __hip_bfloat16* v_ptr = k_ptr;

            float dot = 0.0f;
            for (int d = 0; d < D; ++d)
                dot += __bfloat162float(q_ptr[d]) * __bfloat162float(k_ptr[d]);
            float p = expf(dot - e_max);
            atomicAdd(&e_sum_shared, p);

            for (int d = 0; d < D; ++d) {
                float weighted = p * __bfloat162float(v_ptr[d]);
                atomicAdd(&acc[d], weighted);
            }
        }
        __syncthreads();
    }

    float e_sum = e_sum_shared;
    for (int d = tid; d < D; d += blockDim.x) {
        float result = acc[d] / (e_sum + 1e-6f);
        out_ptr[d] = __float2bfloat16(result);
    }
}

torch::Tensor mla_decode_fwd_hip(torch::Tensor &Q, torch::Tensor &KV,
    int D_v,
    torch::Tensor &kv_indptr, torch::Tensor &kv_page_indices, torch::Tensor &kv_last_page_lens,
    float softmax_scale)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(Q));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int B = Q.size(0);
    int H = Q.size(1);
    int D = Q.size(2);
    int P = KV.size(0);
    int S = KV.size(1);

    TORCH_CHECK(Q.dtype() == torch::kBFloat16, "Q must be bfloat16");
    TORCH_CHECK(KV.dtype() == torch::kBFloat16, "KV must be bfloat16");

    torch::Tensor O = torch::zeros({B, H, D_v}, Q.options());

    // printf("===================================\n");
    // print_tensor_shape("Q", Q);
    // print_tensor_shape("KV", KV);
    // print_tensor_shape("O", O);
    // print_tensor_shape("kv_indptr", kv_indptr);
    // print_tensor_shape("kv_page_indices", kv_page_indices);
    // print_tensor_shape("kv_last_page_lens", kv_last_page_lens);

    const __hip_bfloat16* q_ptr = reinterpret_cast<const __hip_bfloat16*>(Q.data_ptr<at::BFloat16>());
    const __hip_bfloat16* kv_ptr = reinterpret_cast<const __hip_bfloat16*>(KV.data_ptr<at::BFloat16>());
    __hip_bfloat16* out_ptr = reinterpret_cast<__hip_bfloat16*>(O.data_ptr<at::BFloat16>());

    const int* indptr_ptr = kv_indptr.data_ptr<int>();
    const int* indices_ptr = kv_page_indices.data_ptr<int>();

    int tile_max = std::min(P * S, 1024);
    size_t shared_mem = (tile_max + D + tile_max) * sizeof(float);

    dim3 grid(B, H);
    dim3 block(tile_max);

    // hipDeviceProp_t props;
    // hipGetDeviceProperties(&props, 0);  // device 0
    // std::cout << "Device name: " << props.name << std::endl;
    // std::cout << "Max threads per block: " << props.maxThreadsPerBlock << std::endl;
    // std::cout << "Shared memory per block: " << props.sharedMemPerBlock << " bytes" << std::endl;
    // std::cout << "Max block dim (x): " << props.maxThreadsDim[0] << std::endl;
    // std::cout << "Max grid dim (x): " << props.maxGridSize[0] << std::endl;


    // std::cout << "Launching kernel with: B=" << B << ", H=" << H << ", D=" << D
    // << ", P=" << P << ", S=" << S << ", blockDim.x=" << std::max(P*S, D)
    // << ", shared=" << shared_mem << " bytes\n";

    mla_decode_hip_kernel<<<grid, block, shared_mem, stream>>>(
        q_ptr, kv_ptr, out_ptr,
        indptr_ptr, indices_ptr,
        D, H, B, S, tile_max);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());

    // for (int i = 0; i < 8; ++i)
    //     std::cout << "[host check] O[0][0][" << i << "] = " << accessor[0][0][i] << std::endl;

    return O;
}
