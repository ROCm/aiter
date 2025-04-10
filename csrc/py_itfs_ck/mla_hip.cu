// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

//#define MLA_HIP_DEBUG_LOG

#ifdef MLA_HIP_DEBUG_LOG
#define DEBUG_PRINTF(fmt, ...) \
    printf(fmt, ##__VA_ARGS__)
#else
#define DEBUG_PRINTF(fmt, ...)
#endif


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
    DEBUG_PRINTF("%s shape: [", name);
    for (size_t i = 0; i < sizes.size(); ++i) {
        DEBUG_PRINTF("%ld", sizes[i]);
        if (i != sizes.size() - 1) printf(", ");
    }
    DEBUG_PRINTF("]\n");
}

__device__ inline void atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;

    do {
        assumed = old;
        float f_assumed = __int_as_float(assumed);
        float f_old = fmaxf(f_assumed, val);
        old = atomicCAS(address_as_int, assumed, __float_as_int(f_old));
    } while (assumed != old);
}

__global__ void mla_decode_hip_kernel(
    const __hip_bfloat16* __restrict__ q,     // [B, H, D]
    const __hip_bfloat16* __restrict__ kv,    // [P * S, H, D]
    __hip_bfloat16* __restrict__ out,         // [B, H, D_v]
    const int* __restrict__ kv_indptr,        // [B+1]
    const int* __restrict__ kv_indices,       // [#used_kv]
    int D, int D_v, int H, int B, int S,
    int max_tile, float softmax_scale
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int d = threadIdx.x;
    if (d >= D) return;

    int idx = b * H + h;
    const __hip_bfloat16* q_ptr = q + idx * D;
    __hip_bfloat16* out_ptr = out + idx * D_v;

    int start = kv_indptr[b];
    int end = kv_indptr[b + 1];
    int len = end - start;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        DEBUG_PRINTF("(%d,%d,%d) D=%d H=%d B=%d S=%d max_tile=%d idx=%d start=%d end=%d len=%d\n", b, h, d, D, H, B, S, max_tile, idx, start, end, len);
    }

    if (threadIdx.x == 0 && blockIdx.x == 1) {
        DEBUG_PRINTF("(%d,%d,%d) D=%d H=%d B=%d S=%d max_tile=%d idx=%d start=%d end=%d len=%d\n", b, h, d, D, H, B, S, max_tile, idx, start, end, len);
    }

    if (threadIdx.x == 0 && blockIdx.y == 0) { // h == 0
        if (blockIdx.x == 0 || blockIdx.x == 1) { // b == 0 or 1
            int b = blockIdx.x;
            int h = blockIdx.y;
            int d = threadIdx.x;
            int idx = b * H + h;
            const __hip_bfloat16* q_ptr = q + idx * D;
            const __hip_bfloat16* k_ptr = kv;
    
            for (int i = 0; i < 8; ++i) {
                DEBUG_PRINTF("(%d,%d,%d) q[%d] %.4f\n", b, h, d, i, __bfloat162float(q_ptr[i]));
            }
            DEBUG_PRINTF("\n");
        }
    }

    extern __shared__ float shared_mem[];
    float* dot_buf = shared_mem;              // [max_tile]
    float* reduce_buf = dot_buf + max_tile;   // [D]

    float acc = 0.0f;
    float p_sum = 0.0f;

    float local_max = -1e9f;

    for (int tile_start = 0; tile_start < len; tile_start += max_tile) {
        int tile_len = min(max_tile, len - tile_start);
        for (int i = 0; i < tile_len; ++i) {
            int global_idx = start + tile_start + i;
            int page = kv_indices[global_idx];
            int flat_idx = ((page * S) * H + h) * D;

            if (threadIdx.x == 0 && blockIdx.x == 0) {
                DEBUG_PRINTF("(%d,%d,%d) global_idx=%d page=%d flat_idx=%d\n", b, h, d, global_idx, page, flat_idx);
            }

            if (threadIdx.x == 0 && blockIdx.x == 1) {
                DEBUG_PRINTF("(%d,%d,%d) global_idx=%d page=%d flat_idx=%d\n", b, h, d, global_idx, page, flat_idx);
            }

            const __hip_bfloat16* k_ptr = kv + flat_idx;

            if (threadIdx.x == 0 && blockIdx.x == 0) {
                for (int i = 0; i < 8; ++i) {
                    DEBUG_PRINTF("(%d,%d,%d) k: kv[%d] %.4f\n", b, h, d, i, __bfloat162float(k_ptr[i]));
                }
            }

            float dot = __bfloat162float(q_ptr[d]) * __bfloat162float(k_ptr[d]);
            reduce_buf[d] = dot;
            __syncthreads();

            if (d == 0) {
                float total_dot = 0.0f;
                for (int j = 0; j < D; ++j)
                    total_dot += reduce_buf[j];
                dot_buf[tile_start + i] = total_dot;
            }
            __syncthreads();

            if (d == 0)
                local_max = fmaxf(local_max, dot_buf[tile_start + i]);
        }
        __syncthreads();
    }

    __shared__ float e_max;
    if (d == 0) e_max = local_max;
    __syncthreads();

    if(d < max_tile)
        DEBUG_PRINTF("(%d,%d,%d) dot_buf[%d] = %.5f\n", b, h, d, d, dot_buf[d]);

    if (threadIdx.x == 0 && blockIdx.y == 0) // h == 0
        if (blockIdx.x == 0 || blockIdx.x == 1) // b == 0 or 1
        DEBUG_PRINTF("(%d,%d,%d) e_max=%.5f\n", b, h, d, e_max);


    for (int tile_start = 0; tile_start < len; tile_start += max_tile) {
        int tile_len = min(max_tile, len - tile_start);
        for (int i = 0; i < tile_len; ++i) {
            int global_idx = start + tile_start + i;
            int page = kv_indices[global_idx];
            int flat_idx = ((page * S) * H + h) * D;

            if (threadIdx.x == 0 && blockIdx.x == 0) {
                DEBUG_PRINTF("(%d,%d,%d) global_idx=%d page=%d flat_idx=%d\n", b, h, d, global_idx, page, flat_idx);
            }

            if (threadIdx.x == 0 && blockIdx.x == 1) {
                DEBUG_PRINTF("(%d,%d,%d) global_idx=%d page=%d flat_idx=%d\n", b, h, d, global_idx, page, flat_idx);
            }

            const __hip_bfloat16* v_ptr = kv + flat_idx;

            if (threadIdx.x == 0 && blockIdx.x == 0) {
                for (int i = 0; i < 8; ++i) {
                    DEBUG_PRINTF("(%d,%d,%d) v: kv[%d] %.4f\n", b, h, d, i, __bfloat162float(v_ptr[i]));
                }
            }

            float p = expf((dot_buf[tile_start + i] - e_max) * softmax_scale);
            float val = __bfloat162float(v_ptr[d]);
            acc += p * val;
            p_sum += p;

            DEBUG_PRINTF("(%d,%d,%d) tile_start=%d, i=%d, global_idx=%d, p=%.5f, val=%.5f, acc=%.5f\n", b, h, d, tile_start, i, global_idx, p, val, acc);
        }
    }

    if (d < D_v) {
        out_ptr[d] = __float2bfloat16(acc / (p_sum + 1e-6f));
    }

    if(d < 8)
        DEBUG_PRINTF("[B%d] out_ptr address = %p (idx=%d), final acc = %.5f, p_sum = %.5f, out_ptr[%d]=%.5f\n", b, (void*)out_ptr, idx, acc, p_sum, d, __bfloat162float(out_ptr[d]));
}

torch::Tensor mla_decode_fwd_hip(torch::Tensor &Q, torch::Tensor &KV,
    torch::Tensor &kv_indptr, torch::Tensor &kv_page_indices, torch::Tensor &kv_last_page_lens,
    int D_v, float softmax_scale)
{
    TORCH_CHECK(Q.dtype() == torch::kBFloat16, "Q must be bfloat16");
    TORCH_CHECK(KV.dtype() == torch::kBFloat16, "KV must be bfloat16");
 
    const at::cuda::OptionalCUDAGuard device_guard(device_of(Q));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int B = Q.size(0);
    int H = Q.size(1);
    int D = Q.size(2);
    int P = KV.size(0);
    int S = KV.size(1);

    torch::Tensor O = torch::zeros({B, H, D_v}, Q.options());

#ifdef MLA_HIP_DEBUG_LOG
    printf("===================================\n");
    print_tensor_shape("Q", Q);
    print_tensor_shape("KV", KV);
    print_tensor_shape("O", O);
    print_tensor_shape("kv_indptr", kv_indptr);
    print_tensor_shape("kv_page_indices", kv_page_indices);
    print_tensor_shape("kv_last_page_lens", kv_last_page_lens);
    
    std::cout << "B=" << B << ", H=" << H << ", D=" << D << ", P=" << P << ", S=" << S << ", softmax_scale=" << softmax_scale << std::endl;
#endif

    const __hip_bfloat16* q_ptr = reinterpret_cast<const __hip_bfloat16*>(Q.data_ptr<at::BFloat16>());
    const __hip_bfloat16* kv_ptr = reinterpret_cast<const __hip_bfloat16*>(KV.data_ptr<at::BFloat16>());
    __hip_bfloat16* out_ptr = reinterpret_cast<__hip_bfloat16*>(O.data_ptr<at::BFloat16>());

    const int* indptr_ptr = kv_indptr.data_ptr<int>();
    const int* indices_ptr = kv_page_indices.data_ptr<int>();

    constexpr int MAX_TILE_LEN = 8192;
    size_t shared_mem = (MAX_TILE_LEN + D * 2) * sizeof(float);

    dim3 grid(B, H);
    dim3 block(D);

    // hipDeviceProp_t props;
    // hipGetDeviceProperties(&props, 0);  // device 0
    // std::cout << "Device name: " << props.name << std::endl;
    // std::cout << "Max threads per block: " << props.maxThreadsPerBlock << std::endl;
    // std::cout << "Shared memory per block: " << props.sharedMemPerBlock << " bytes" << std::endl;
    // std::cout << "Max block dim (x): " << props.maxThreadsDim[0] << std::endl;
    // std::cout << "Max grid dim (x): " << props.maxGridSize[0] << std::endl;

#ifdef MLA_HIP_DEBUG_LOG
    std::cout << "blockDim.x=" << block.x << ", MAX_TILE_LEN=" << MAX_TILE_LEN << ", shared=" << shared_mem << " bytes\n";
#endif

    mla_decode_hip_kernel<<<grid, block, shared_mem, stream>>>(
        q_ptr, kv_ptr, out_ptr,
        indptr_ptr, indices_ptr,
        D, D_v, H, B, S, MAX_TILE_LEN, softmax_scale);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());

#ifdef MLA_HIP_DEBUG_LOG
    auto* raw_ptr = O.data_ptr<at::BFloat16>();
    std::cout << "O.data_ptr(base) = " << static_cast<void*>(raw_ptr) << std::endl;
    std::cout << "O shape: " << O.sizes() << ", strides: " << O.strides() << std::endl;
    std::cout << "O.is_contiguous(): " << O.is_contiguous() << std::endl;

    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < 8; ++i) {
            int idx = (b * H + 0) * D_v + i;
            float val = static_cast<float>(raw_ptr[idx]);
            void* addr = static_cast<void*>(&raw_ptr[idx]);
            std::cout << "[host] O[" << b << "][0][" << i << "] @ " << addr << " = " << val << std::endl;
        }
    }
    
    auto out_cpu = O.cpu();
    auto accessor = out_cpu.accessor<at::BFloat16, 3>();

    for (int b = 0; b < B; b++)
        for (int i = 0; i < 8; ++i) {
            float val = static_cast<float>(accessor[b][0][i]);
            std::cout << "[host check] O[" << b << "][0][" << i << "] = " << val << std::endl;
        }
#endif

    return O;
}
