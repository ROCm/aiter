// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "dispatch_utils.h"
#include "hip_compat.h"
#include "hip_reduce.h"
#include "py_itfs_common.h"

#include "quant_utils.cuh"
#include "vec_convert.h"

#include <algorithm>
#include <cassert>
#include <map>
#include <vector>

#include <hip/hip_bf16.h>

template <typename T, typename F>
__device__ constexpr T block_reduce(T val, F reduce_f)
{
    __shared__ T smem[256];
    T wave_local = wave_reduce(val, reduce_f);
    T v_local    = cross_wave_reduce(wave_local, reduce_f, smem);
    return v_local;
}

namespace aiter {

void swap_blocks(torch::Tensor& src, torch::Tensor& dst, const torch::Tensor& block_mapping)
{
    torch::Device src_device = src.device();
    torch::Device dst_device = dst.device();
    cudaMemcpyKind memcpy_type;
    if(src_device.is_cuda() && dst_device.is_cuda())
    {
        TORCH_CHECK(src_device.index() == dst_device.index(),
                    "src and dst must be on the same GPU");
        memcpy_type = cudaMemcpyDeviceToDevice;
    }
    else if(src_device.is_cuda() && dst_device.is_cpu())
    {
        memcpy_type = cudaMemcpyDeviceToHost;
    }
    else if(src_device.is_cpu() && dst_device.is_cuda())
    {
        memcpy_type = cudaMemcpyHostToDevice;
    }
    else
    {
        TORCH_CHECK(false, "Invalid device combination");
    }

    // NOTE(youkaichao): keep in mind that `block_mapping` should be
    // a cpu tensor, otherwise every `item` call will require a gpu-cpu
    // synchronization.
    TORCH_CHECK(block_mapping.device().is_cpu(), "block_mapping must be on CPU");

    char* src_ptr = static_cast<char*>(src.data_ptr());
    char* dst_ptr = static_cast<char*>(dst.data_ptr());

    const int64_t block_size_in_bytes = src.element_size() * src[0].numel();
    const at::cuda::OptionalCUDAGuard device_guard(src_device.is_cuda() ? src_device : dst_device);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // NOTE(woosuk): This can be slow if the number of blocks is large.
    const int64_t num_blocks = block_mapping.size(0);
    for(size_t i = 0; i < num_blocks; i++)
    {
        int64_t src_block_number = block_mapping[i][0].item<int64_t>();
        int64_t dst_block_number = block_mapping[i][1].item<int64_t>();
        int64_t src_offset       = src_block_number * block_size_in_bytes;
        int64_t dst_offset       = dst_block_number * block_size_in_bytes;
        cudaMemcpyAsync(
            dst_ptr + dst_offset, src_ptr + src_offset, block_size_in_bytes, memcpy_type, stream);
    }
}

} // namespace aiter

namespace aiter {

// Grid: (num_layers, num_pairs)
template <typename scalar_t>
__global__ void copy_blocks_kernel(int64_t* key_cache_ptrs,
                                   int64_t* value_cache_ptrs,
                                   const int64_t* __restrict__ block_mapping,
                                   const int numel_per_block)
{
    const int layer_idx = blockIdx.x;
    const int pair_idx  = blockIdx.y;

    scalar_t* key_cache      = reinterpret_cast<scalar_t*>(key_cache_ptrs[layer_idx]);
    scalar_t* value_cache    = reinterpret_cast<scalar_t*>(value_cache_ptrs[layer_idx]);
    int64_t src_block_number = block_mapping[2 * pair_idx];
    int64_t dst_block_number = block_mapping[2 * pair_idx + 1];

    const int64_t src_block_offset = src_block_number * numel_per_block;
    const int64_t dst_block_offset = dst_block_number * numel_per_block;
    for(int i = threadIdx.x; i < numel_per_block; i += blockDim.x)
    {
        int64_t src_offset    = src_block_offset + i;
        int64_t dst_offset    = dst_block_offset + i;
        key_cache[dst_offset] = key_cache[src_offset];
    }
    for(int i = threadIdx.x; i < numel_per_block; i += blockDim.x)
    {
        int64_t src_offset      = src_block_offset + i;
        int64_t dst_offset      = dst_block_offset + i;
        value_cache[dst_offset] = value_cache[src_offset];
    }
}

} // namespace aiter

namespace aiter {

// Note: the key_caches and value_caches vectors are constant but
// not the Tensors they contain. The vectors need to be const refs
// in order to satisfy pytorch's C++ operator registration code.
void copy_blocks(std::vector<torch::Tensor> const& key_caches,
                 std::vector<torch::Tensor> const& value_caches,
                 const torch::Tensor& block_mapping)
{
    int num_layers = key_caches.size();
    TORCH_CHECK(num_layers == value_caches.size());
    if(num_layers == 0)
    {
        return;
    }
    torch::Device cache_device = key_caches[0].device();
    TORCH_CHECK(cache_device.is_cuda());

    // Create data structures for the kernel.
    // Create an array of pointers to the key and value caches.
    int64_t key_cache_ptrs[num_layers];
    int64_t value_cache_ptrs[num_layers];
    for(int layer_idx = 0; layer_idx < num_layers; ++layer_idx)
    {
        key_cache_ptrs[layer_idx]   = reinterpret_cast<int64_t>(key_caches[layer_idx].data_ptr());
        value_cache_ptrs[layer_idx] = reinterpret_cast<int64_t>(value_caches[layer_idx].data_ptr());
    }

    // block_mapping is a 2D tensor with shape (num_pairs, 2).
    int num_pairs = block_mapping.size(0);

    // Move the data structures to the GPU.
    // NOTE: This synchronizes the CPU and GPU.
    torch::Tensor key_cache_ptrs_tensor =
        torch::from_blob(key_cache_ptrs, {num_layers}, torch::kInt64).to(cache_device);
    torch::Tensor value_cache_ptrs_tensor =
        torch::from_blob(value_cache_ptrs, {num_layers}, torch::kInt64).to(cache_device);

    // Launch the kernel.
    const int numel_per_block = key_caches[0][0].numel();
    dim3 grid(num_layers, num_pairs);
    dim3 block(std::min(1024, numel_per_block));
    const at::cuda::OptionalCUDAGuard device_guard(cache_device);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    VLLM_DISPATCH_FLOATING_AND_BYTE_TYPES(key_caches[0].scalar_type(), "copy_blocks_kernel", ([&] {
                                              aiter::copy_blocks_kernel<scalar_t>
                                                  <<<grid, block, 0, stream>>>(
                                                      key_cache_ptrs_tensor.data_ptr<int64_t>(),
                                                      value_cache_ptrs_tensor.data_ptr<int64_t>(),
                                                      block_mapping.data_ptr<int64_t>(),
                                                      numel_per_block);
                                          }));
}

} // namespace aiter

namespace aiter {

template <typename scalar_t,
          typename cache_t,
          vllm::Fp8KVCacheDataType kv_dt,
          bool asmLayout          = false,
          typename slot_mapping_t = int64_t>
__global__ void
reshape_and_cache_kernel(const scalar_t* __restrict__ key,   // [num_tokens, num_heads, head_size]
                         const scalar_t* __restrict__ value, // [num_tokens, num_heads, head_size]
                         cache_t* __restrict__ key_cache,    // [num_blocks, num_heads, head_size/x,
                                                             // block_size, x]
                         cache_t* __restrict__ value_cache,  // [num_blocks, num_heads, head_size,
                                                             // block_size]
                         const slot_mapping_t* __restrict__ slot_mapping, // [num_tokens]
                         const int key_stride,
                         const int value_stride,
                         const int num_heads,
                         const int head_size,
                         const int block_size,
                         const int x,
                         const float* k_scale,
                         const float* v_scale)
{
    const int64_t token_idx       = blockIdx.x;
    const slot_mapping_t slot_idx = slot_mapping[token_idx];
    if(slot_idx < 0)
    {
        // Padding token that should be ignored.
        return;
    }

    const int64_t block_idx    = static_cast<int64_t>(slot_idx) / block_size;
    const int64_t block_offset = static_cast<int64_t>(slot_idx) % block_size;

    const int n                 = num_heads * head_size;
    const float inverted_kscale = 1 / (*k_scale);
    const float inverted_vscale = 1 / (*v_scale);
    for(int i = threadIdx.x; i < n; i += blockDim.x)
    {
        const int64_t src_key_idx   = token_idx * key_stride + i;
        const int64_t src_value_idx = token_idx * value_stride + i;

        const int head_idx    = i / head_size;
        const int head_offset = i % head_size;
        const int x_idx       = head_offset / x;
        const int x_offset    = head_offset % x;

        const int64_t tgt_key_idx = block_idx * num_heads * (head_size / x) * block_size * x +
                                    head_idx * (head_size / x) * block_size * x +
                                    x_idx * block_size * x + block_offset * x + x_offset;
        int64_t tgt_value_idx;
        if constexpr(asmLayout)
        { //[num_blocks, num_heads, block_size/X, head_size, X]
            const int x_idx_v    = block_offset / x;
            const int x_offset_v = block_offset % x;
            tgt_value_idx        = block_idx * num_heads * head_size * block_size +
                            head_idx * head_size * block_size + x_idx_v * head_size * x +
                            head_offset * x + x_offset_v;
        }
        else
        { //[num_blocks, num_heads, head_size, block_size]
            tgt_value_idx = block_idx * num_heads * head_size * block_size +
                            head_idx * head_size * block_size + head_offset * block_size +
                            block_offset;
        }
        scalar_t tgt_key   = key[src_key_idx];
        scalar_t tgt_value = value[src_value_idx];
        if constexpr(kv_dt == vllm::Fp8KVCacheDataType::kAuto)
        {
            key_cache[tgt_key_idx]     = tgt_key;
            value_cache[tgt_value_idx] = tgt_value;
        }
        else
        {
            key_cache[tgt_key_idx] = ck_tile::type_convert<cache_t>(
                ck_tile::type_convert<float>(tgt_key) * inverted_kscale);
            value_cache[tgt_value_idx] = ck_tile::type_convert<cache_t>(
                ck_tile::type_convert<float>(tgt_value) * inverted_vscale);
        }
    }
}

template <typename scalar_t, typename cache_t, vllm::Fp8KVCacheDataType kv_dt>
__global__ void reshape_and_cache_flash_kernel(
    const scalar_t* __restrict__ key,         // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,       // [num_tokens, num_heads, head_size]
    cache_t* __restrict__ key_cache,          // [num_blocks, block_size, num_heads,
                                              // head_size]
    cache_t* __restrict__ value_cache,        // [num_blocks, block_size, num_heads,
                                              // head_size]
    const int64_t* __restrict__ slot_mapping, // [num_tokens]
    const int block_stride,
    const int key_stride,
    const int value_stride,
    const int num_heads,
    const int head_size,
    const int block_size,
    const float* k_scale,
    const float* v_scale)
{
    const int64_t token_idx = blockIdx.x;
    const int64_t slot_idx  = slot_mapping[token_idx];
    // NOTE: slot_idx can be -1 if the token is padded
    if(slot_idx < 0)
    {
        return;
    }
    const int64_t block_idx     = slot_idx / block_size;
    const int64_t block_offset  = slot_idx % block_size;
    const int n                 = num_heads * head_size;
    const float inverted_kscale = 1 / (*k_scale);
    const float inverted_vscale = 1 / (*v_scale);
    for(int i = threadIdx.x; i < n; i += blockDim.x)
    {
        const int64_t src_key_idx       = token_idx * key_stride + i;
        const int64_t src_value_idx     = token_idx * value_stride + i;
        const int head_idx              = i / head_size;
        const int head_offset           = i % head_size;
        const int64_t tgt_key_value_idx = block_idx * block_stride +
                                          block_offset * num_heads * head_size +
                                          head_idx * head_size + head_offset;
        scalar_t tgt_key   = key[src_key_idx];
        scalar_t tgt_value = value[src_value_idx];
        if constexpr(kv_dt == vllm::Fp8KVCacheDataType::kAuto)
        {
            key_cache[tgt_key_value_idx]   = tgt_key;
            value_cache[tgt_key_value_idx] = tgt_value;
        }
        else
        {
            key_cache[tgt_key_value_idx] = ck_tile::type_convert<cache_t>(
                ck_tile::type_convert<float>(tgt_key) * inverted_kscale);
            value_cache[tgt_key_value_idx] = ck_tile::type_convert<cache_t>(
                ck_tile::type_convert<float>(tgt_value) * inverted_vscale);
        }
    }
}

namespace impl {

__device__ float abs(float x)
{
    union
    {
        float f32;
        uint32_t u32;
    } y;
    y.f32 = x;
    y.u32 = y.u32 & 0x7fffffff;
    return y.f32;
};
} // namespace impl

// TODO: this is for kv pertoken quant
template <typename scalar_t,
          typename cache_t,
          typename dequant_scale_t,
          bool asmLayout = false,
          int wg_size    = 256>
__global__ void reshape_and_cache_with_per_token_quant_kernel(
    const scalar_t* __restrict__ key,   // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value, // [num_tokens, num_heads, head_size]
    cache_t* __restrict__ key_cache,    // [num_blocks, num_heads, head_size/x, block_size, x]
    cache_t* __restrict__ value_cache,  // [num_blocks, num_heads, head_size, block_size]
    dequant_scale_t* __restrict__ k_dequant_scales, // [num_heads, max_kv_tokens]
    dequant_scale_t* __restrict__ v_dequant_scales, // [num_heads, max_kv_tokens]
    const int64_t* __restrict__ slot_mapping,       // [num_tokens]
    const int key_stride,
    const int value_stride,
    const int num_heads,
    const int head_size,
    const int block_size,
    const int x,
    const int num_tokens,
    const int max_kv_tokens)
{
    float dtypeMax              = ck_tile::type_convert<float>(ck_tile::numeric<cache_t>::max());
    const int32_t tokens_per_wg = wg_size / warpSize;

    // every wave compute one token, one head, all the headim
    int wave_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;

    const int64_t token_idx = static_cast<int64_t>(blockIdx.x * tokens_per_wg + wave_id);
    const int32_t head_idx  = blockIdx.y;
    const int64_t slot_idx  = slot_mapping[token_idx];

    if(token_idx >= num_tokens || slot_idx < 0)
    {
        // Padding token that should be ignored.
        return;
    }

    const int64_t block_idx    = slot_idx / block_size;
    const int64_t block_offset = slot_idx % block_size;

    auto f_absmax_f32 = [](float v_0_, float v_1_) {
        return __builtin_fmaxf(impl::abs(v_0_), impl::abs(v_1_));
    };
    auto f_max_f32 = [](float v_0_, float v_1_) { return __builtin_fmaxf(v_0_, v_1_); };

    constexpr int local_dim_elems = 8;

    float k_local_dim[local_dim_elems]{0}; // up to 64*8 = 512 hdim
    float v_local_dim[local_dim_elems]{0}; // up to 64*8 = 512 hdim
#pragma unroll
    for(int i_d = 0; i_d < local_dim_elems; i_d++)
    {
        int current_d           = lane_id + i_d * warpSize;
        const int64_t src_k_idx = token_idx * key_stride + head_idx * head_size + current_d;
        const int64_t src_v_idx = token_idx * value_stride + head_idx * head_size + current_d;
        if(current_d < head_size)
        {
            k_local_dim[i_d] = ck_tile::type_convert<float>(key[src_k_idx]);
            v_local_dim[i_d] = ck_tile::type_convert<float>(value[src_v_idx]);
        }
    }

    // smoot-quant
    float k_local_max = [&]() {
        float max_ = k_local_dim[0];
#pragma unroll
        for(int i_d = 1; i_d < local_dim_elems; i_d++)
        {
            max_ = f_absmax_f32(max_, k_local_dim[i_d]);
        }
        return max_;
    }();

    float k_max = wave_reduce(k_local_max, f_max_f32);

    float v_local_max = [&]() {
        float max_ = v_local_dim[0];
#pragma unroll
        for(int i_d = 1; i_d < local_dim_elems; i_d++)
        {
            max_ = f_absmax_f32(max_, v_local_dim[i_d]);
        }
        return max_;
    }();
    float v_max = wave_reduce(v_local_max, f_max_f32);

    float k_token_scale = k_max / dtypeMax;
    float v_token_scale = v_max / dtypeMax;

#pragma unroll
    for(int i_d = 0; i_d < local_dim_elems; i_d++)
    {
        k_local_dim[i_d] = k_local_dim[i_d] / k_token_scale;
        v_local_dim[i_d] = v_local_dim[i_d] / v_token_scale;
    }

    // store the scale
    int scale_idx;
    if constexpr(asmLayout)
    {
        // [num_blocks, num_heads, block_size]
        scale_idx = block_size * num_heads * block_idx + block_size * head_idx + block_offset;
        k_dequant_scales[scale_idx] = k_token_scale;
        v_dequant_scales[scale_idx] = v_token_scale;
    }
    else
    {
        scale_idx                   = head_idx * max_kv_tokens + slot_idx;
        k_dequant_scales[scale_idx] = k_token_scale;
        v_dequant_scales[scale_idx] = v_token_scale;
    }

    // now let's store out
#pragma unroll
    for(int i = 0; i < local_dim_elems; i++)
    {
        // const int head_idx = i / head_size;
        // const int head_offset = i % head_size;
        int i_d = lane_id + i * warpSize;
        if(i_d >= head_size)
        {
            break;
        }
        const int x_idx    = i_d / x;
        const int x_offset = i_d % x;

        const int64_t tgt_key_idx = block_idx * num_heads * (head_size / x) * block_size * x +
                                    head_idx * (head_size / x) * block_size * x +
                                    x_idx * block_size * x + block_offset * x + x_offset;
        int64_t tgt_value_idx;
        if constexpr(asmLayout)
        { //[num_blocks, num_heads, block_size/X, head_size, X]
            const int x_idx_v    = block_offset / x;
            const int x_offset_v = block_offset % x;
            tgt_value_idx        = block_idx * num_heads * head_size * block_size +
                            head_idx * head_size * block_size + x_idx_v * head_size * x + i_d * x +
                            x_offset_v;
        }
        else
        { //[num_blocks, num_heads, head_size, block_size]
            tgt_value_idx = block_idx * num_heads * head_size * block_size +
                            head_idx * head_size * block_size + i_d * block_size + block_offset;
        }
        key_cache[tgt_key_idx]     = ck_tile::type_convert<cache_t>(k_local_dim[i]);
        value_cache[tgt_value_idx] = ck_tile::type_convert<cache_t>(v_local_dim[i]);
    }
}

// TODO: this is for kv pertoken quant
template <typename scalar_t,
          typename cache_t,
          typename dequant_scale_t,
          bool asmLayout = false,
          int wg_size    = 256>
__global__ void reshape_and_cache_with_block_quant_kernel(
    const scalar_t* __restrict__ key,   // [batch_size, seq_len, num_heads, head_size]
    const scalar_t* __restrict__ value, // [batch_size, seq_len, num_heads, head_size]
    cache_t* __restrict__ key_cache,    // [num_blocks, num_heads, head_size/x, block_size, x]
    cache_t* __restrict__ value_cache,  // [num_blocks, num_heads, head_size, block_size]
    dequant_scale_t* __restrict__ k_dequant_scales, // [num_heads, num_blocks]
    dequant_scale_t* __restrict__ v_dequant_scales, // [num_heads, num_blocks]
    const int64_t* __restrict__ slot_mapping,       // [num_tokens]
    const int key_stride,
    const int value_stride,
    const int num_heads,
    const int num_blocks,
    const int head_size,
    const int block_size,
    const int x,
    const int num_tokens,
    const int seq_len)
{
    float dtypeMax          = ck_tile::type_convert<float>(ck_tile::numeric<cache_t>::max());
    int64_t first_token_idx = blockIdx.x * seq_len + blockIdx.y * block_size;
    int64_t slot_idx;
    int64_t block_idx;
    int64_t block_offset;
    if(blockIdx.y * block_size >= seq_len)
    {
        int64_t preTg_block_idx = slot_mapping[first_token_idx - block_size] / block_size;
        first_token_idx         = blockIdx.x * seq_len + seq_len - 1;
        slot_idx                = slot_mapping[first_token_idx];
        block_idx               = slot_idx / block_size;
        if(preTg_block_idx == block_idx)
        {
            return;
        }
        block_offset = slot_idx % block_size;
    }
    else
    {
        slot_idx     = slot_mapping[first_token_idx];
        block_idx    = slot_idx / block_size;
        block_offset = slot_idx % block_size;
    }

    if(slot_idx < 0)
    {
        // Padding token that should be ignored.
        return;
    }
    const int32_t head_idx = blockIdx.z;

    // fix first_token_idx to real block first_token_idx
    if(blockIdx.y > 0 && block_offset > 0)
    {
        __shared__ int64_t idx_smem[2];
        if(threadIdx.x < block_size)
        {
            int64_t token_idx  = first_token_idx - (threadIdx.x + 1);
            int64_t block_idx1 = slot_mapping[token_idx] / block_size;
            int64_t slot_idx2  = slot_mapping[token_idx + 1];
            int64_t block_idx2 = slot_idx2 / block_size;
            if(block_idx1 != block_idx2 && block_idx2 == block_idx)
            {
                idx_smem[0] = token_idx + 1;
                idx_smem[1] = slot_idx2;
            }
        }
        __syncthreads();
        first_token_idx = idx_smem[0];
        slot_idx        = idx_smem[1];
    }

    block_offset = slot_idx % block_size;

    int tokens_in_block = 0;
    if(first_token_idx + threadIdx.x < num_tokens)
    {
        tokens_in_block = slot_mapping[first_token_idx + threadIdx.x] / block_size;
        tokens_in_block = tokens_in_block == block_idx ? 1 : 0;
    }
    int numtokens_in_block = block_reduce(tokens_in_block, [](float a, float b) { return a + b; });

    auto f_absmax_f32 = [](float v_0_, float v_1_) {
        return __builtin_fmaxf(impl::abs(v_0_), impl::abs(v_1_));
    };
    auto f_max_f32 = [](float v_0_, float v_1_) { return __builtin_fmaxf(v_0_, v_1_); };

    float k_max_val = 1e-6;
    float v_max_val = 1e-6;
#pragma unroll
    for(int id = 0; id < numtokens_in_block * head_size; id += blockDim.x)
    {
        if((id + threadIdx.x) < numtokens_in_block * head_size)
        {
            int64_t token_idx = (id + threadIdx.x) / head_size + first_token_idx;
            int current_d     = (id + threadIdx.x) % head_size;

            const int64_t src_k_idx = token_idx * key_stride + head_idx * head_size + current_d;
            const int64_t src_v_idx = token_idx * value_stride + head_idx * head_size + current_d;

            k_max_val = f_absmax_f32(k_max_val, ck_tile::type_convert<float>(key[src_k_idx]));
            v_max_val = f_absmax_f32(v_max_val, ck_tile::type_convert<float>(value[src_v_idx]));
        }
    }

    k_max_val = block_reduce(k_max_val, f_max_f32);
    v_max_val = block_reduce(v_max_val, f_max_f32);

    float k_block_scale = k_max_val / dtypeMax;
    float v_block_scale = v_max_val / dtypeMax;

    int64_t scale_idx;
    if constexpr(asmLayout)
    {
        scale_idx = block_idx * num_heads + head_idx;
    }
    else
    {
        scale_idx = head_idx * num_blocks + block_idx;
    }

    if(block_offset > 0)
    {
        float k_block_scale_global = k_dequant_scales[scale_idx];
        float v_block_scale_global = v_dequant_scales[scale_idx];

        if(k_block_scale_global < k_block_scale)
        {
            int64_t tgt_value_idx =
                block_idx * num_heads * head_size * block_size + head_idx * head_size * block_size;
#pragma unroll
            for(int id = 0; id < block_offset * head_size; id += blockDim.x)
            {
                if(id + threadIdx.x < block_offset * head_size)
                {
                    int block_offset_local = (id + threadIdx.x) / head_size;
                    int x_idx              = (id + threadIdx.x) % head_size / x;
                    int x_offset           = (id + threadIdx.x) % x;
                    int64_t cache_idx =
                        tgt_value_idx + x_idx * block_size * x + block_offset_local * x + x_offset;
                    float tmp            = ck_tile::type_convert<float>(key_cache[cache_idx]);
                    tmp                  = tmp * k_block_scale_global / k_block_scale;
                    key_cache[cache_idx] = ck_tile::type_convert<cache_t>(tmp);
                }
            }
            k_dequant_scales[scale_idx] = k_block_scale;
        }
        else
        {
            k_block_scale = k_block_scale_global;
        }

        if(v_block_scale_global < v_block_scale)
        {
            int64_t tgt_value_idx =
                block_idx * num_heads * head_size * block_size + head_idx * head_size * block_size;
#pragma unroll
            for(int id = 0; id < block_offset * head_size; id += blockDim.x)
            {
                if(id + threadIdx.x < block_offset * head_size)
                {
                    int64_t cache_idx;
                    if constexpr(asmLayout)
                    {
                        int block_offset_local      = (id + threadIdx.x) / head_size;
                        int head_offset             = (id + threadIdx.x) % head_size;
                        int block_offset_local_divX = block_offset_local / x;
                        int x_idx                   = block_offset_local % x;
                        cache_idx = tgt_value_idx + block_offset_local_divX * head_size * x +
                                    head_offset * x + x_idx;
                    }
                    else
                    {
                        int block_offset_local = (id + threadIdx.x) / head_size;
                        int head_offset        = (id + threadIdx.x) % head_size;
                        cache_idx = tgt_value_idx + head_offset * block_size + block_offset_local;
                    }
                    float tmp              = ck_tile::type_convert<float>(value_cache[cache_idx]);
                    tmp                    = tmp * v_block_scale_global / v_block_scale;
                    value_cache[cache_idx] = ck_tile::type_convert<cache_t>(tmp);
                }
            }
            v_dequant_scales[scale_idx] = v_block_scale;
        }
        else
        {
            v_block_scale = v_block_scale_global;
        }
    }
    else
    {
        k_dequant_scales[scale_idx] = k_block_scale;
        v_dequant_scales[scale_idx] = v_block_scale;
    }
    k_block_scale = 1 / k_block_scale;
    v_block_scale = 1 / v_block_scale;

    // now let's store out
    for(int id = 0; id < numtokens_in_block * head_size; id += blockDim.x)
    {
        if((id + threadIdx.x) < numtokens_in_block * head_size)
        {
            int token_idx          = (id + threadIdx.x) / head_size + first_token_idx;
            int current_d          = (id + threadIdx.x) % head_size;
            int block_offset_local = token_idx - first_token_idx + block_offset;

            const int64_t src_k_idx = token_idx * key_stride + head_idx * head_size + current_d;
            const int64_t src_v_idx = token_idx * value_stride + head_idx * head_size + current_d;
            float tmp_k             = ck_tile::type_convert<float>(key[src_k_idx]) * k_block_scale;
            float tmp_v = ck_tile::type_convert<float>(value[src_v_idx]) * v_block_scale;

            const int x_idx    = current_d / x;
            const int x_offset = current_d % x;
            //[num_blocks, num_heads, head_size/X, block_size, X]
            const int64_t tgt_key_idx = block_idx * num_heads * head_size * block_size +
                                        head_idx * head_size * block_size + x_idx * block_size * x +
                                        block_offset_local * x + x_offset;

            int64_t tgt_value_idx;
            if constexpr(asmLayout)
            { //[num_blocks, num_heads, block_size/X, head_size, X]
                const int x_idx    = block_offset_local / x;
                const int x_offset = block_offset_local % x;
                tgt_value_idx      = block_idx * num_heads * head_size * block_size +
                                head_idx * head_size * block_size + x_idx * head_size * x +
                                current_d * x + x_offset;
            }
            else
            { //[num_blocks, num_heads, head_size, block_size]
                tgt_value_idx = block_idx * num_heads * head_size * block_size +
                                head_idx * head_size * block_size + current_d * block_size +
                                block_offset_local;
            }
            key_cache[tgt_key_idx]     = ck_tile::type_convert<cache_t>(tmp_k);
            value_cache[tgt_value_idx] = ck_tile::type_convert<cache_t>(tmp_v);
        }
    }
}

// TODO: this is for kv block quant for asm pa
template <typename scalar_t,
          typename cache_t,
          typename dequant_scale_t,
          bool asmLayout = false,
          int wg_size    = 256>
__global__ void reshape_and_cache_with_block_quant_kernel_for_asmpa(
    const scalar_t* __restrict__ key,   // [batch_size, seq_len, num_heads, head_size]
    const scalar_t* __restrict__ value, // [batch_size, seq_len, num_heads, head_size]
    cache_t* __restrict__ key_cache,    // [num_blocks, num_heads, head_size/x, block_size:16, x]
    cache_t* __restrict__ value_cache,  // [num_blocks, num_heads, head_size, block_size:16]
    dequant_scale_t* __restrict__ k_dequant_scales, // [num_heads,
                                                    // num_blocks/(ori_block_size/block_size:16)]
    dequant_scale_t* __restrict__ v_dequant_scales, // [num_heads,
                                                    // num_blocks/(ori_block_size/block_size:16)]
    const int64_t* __restrict__ slot_mapping,       // [num_tokens]
    const int key_stride,
    const int value_stride,
    const int num_heads,
    const int num_blocks,
    const int head_size,
    const int block_size,
    const int x,
    const int num_tokens,
    const int seq_len,
    const int ori_block_size)
{
    float dtypeMax          = ck_tile::type_convert<float>(ck_tile::numeric<cache_t>::max());
    int64_t first_token_idx = blockIdx.x * seq_len + blockIdx.y * ori_block_size;
    int64_t slot_idx;
    int64_t block_idx;
    int64_t block_offset;
    if(blockIdx.y * ori_block_size >= seq_len)
    {
        int64_t preTg_block_idx = slot_mapping[first_token_idx - ori_block_size] / ori_block_size;
        first_token_idx         = blockIdx.x * seq_len + seq_len - 1;
        slot_idx                = slot_mapping[first_token_idx];
        block_idx               = slot_idx / ori_block_size;
        if(preTg_block_idx == block_idx)
        {
            return;
        }
        block_offset = slot_idx % ori_block_size;
    }
    else
    {
        slot_idx     = slot_mapping[first_token_idx];
        block_idx    = slot_idx / ori_block_size;
        block_offset = slot_idx % ori_block_size;
    }

    if(slot_idx < 0)
    {
        // Padding token that should be ignored.
        return;
    }
    const int32_t head_idx = blockIdx.z;

    // fix first_token_idx to real block first_token_idx
    if(blockIdx.y > 0 && block_offset > 0)
    {
        __shared__ int64_t idx_smem[2];
        if(threadIdx.x < ori_block_size)
        {
            int64_t token_idx  = first_token_idx - (threadIdx.x + 1);
            int64_t block_idx1 = slot_mapping[token_idx] / ori_block_size;
            int64_t slot_idx2  = slot_mapping[token_idx + 1];
            int64_t block_idx2 = slot_idx2 / ori_block_size;
            if(block_idx1 != block_idx2 && block_idx2 == block_idx)
            {
                idx_smem[0] = token_idx + 1;
                idx_smem[1] = slot_idx2;
            }
        }
        __syncthreads();
        first_token_idx = idx_smem[0];
        slot_idx        = idx_smem[1];
    }

    block_offset = slot_idx % ori_block_size;

    int tokens_in_block = 0;
    if(first_token_idx + threadIdx.x < num_tokens)
    {
        tokens_in_block = slot_mapping[first_token_idx + threadIdx.x] / ori_block_size;
        tokens_in_block = tokens_in_block == block_idx ? 1 : 0;
    }
    int numtokens_in_block = block_reduce(tokens_in_block, [](float a, float b) { return a + b; });

    auto f_absmax_f32 = [](float v_0_, float v_1_) {
        return __builtin_fmaxf(impl::abs(v_0_), impl::abs(v_1_));
    };
    auto f_max_f32 = [](float v_0_, float v_1_) { return __builtin_fmaxf(v_0_, v_1_); };

    float k_max_val = 1e-6;
    float v_max_val = 1e-6;
#pragma unroll
    for(int id = 0; id < numtokens_in_block * head_size; id += blockDim.x)
    {
        if((id + threadIdx.x) < numtokens_in_block * head_size)
        {
            int64_t token_idx = (id + threadIdx.x) / head_size + first_token_idx;
            int current_d     = (id + threadIdx.x) % head_size;

            const int64_t src_k_idx = token_idx * key_stride + head_idx * head_size + current_d;
            const int64_t src_v_idx = token_idx * value_stride + head_idx * head_size + current_d;

            k_max_val = f_absmax_f32(k_max_val, ck_tile::type_convert<float>(key[src_k_idx]));
            v_max_val = f_absmax_f32(v_max_val, ck_tile::type_convert<float>(value[src_v_idx]));
        }
    }

    k_max_val = block_reduce(k_max_val, f_max_f32);
    v_max_val = block_reduce(v_max_val, f_max_f32);

    float k_block_scale = k_max_val / dtypeMax;
    float v_block_scale = v_max_val / dtypeMax;

    int64_t scale_idx;
    if constexpr(asmLayout)
    {
        scale_idx = block_idx * num_heads + head_idx;
    }
    else
    {
        scale_idx = head_idx * num_blocks / (ori_block_size / block_size) + block_idx;
    }

    if(block_offset > 0)
    {
        float k_block_scale_global = k_dequant_scales[scale_idx];
        float v_block_scale_global = v_dequant_scales[scale_idx];

        if(k_block_scale_global < k_block_scale)
        {
            int64_t tgt_key_idx = block_idx * num_heads * head_size * ori_block_size +
                                  head_idx * head_size * block_size;
#pragma unroll
            for(int id = 0; id < block_offset * head_size; id += blockDim.x)
            {
                if(id + threadIdx.x < block_offset * head_size)
                {
                    int block_offset_local = (id + threadIdx.x) / head_size;
                    int cur_block_id       = block_offset_local / block_size;
                    block_offset_local     = block_offset_local % block_size;
                    int x_idx              = (id + threadIdx.x) % head_size / x;
                    int x_offset           = (id + threadIdx.x) % x;
                    int64_t cache_idx      = tgt_key_idx +
                                        cur_block_id * num_heads * head_size * block_size +
                                        x_idx * block_size * x + block_offset_local * x + x_offset;
                    float tmp            = ck_tile::type_convert<float>(key_cache[cache_idx]);
                    tmp                  = tmp * k_block_scale_global / k_block_scale;
                    key_cache[cache_idx] = ck_tile::type_convert<cache_t>(tmp);
                }
            }
            k_dequant_scales[scale_idx] = k_block_scale;
        }
        else
        {
            k_block_scale = k_block_scale_global;
        }

        if(v_block_scale_global < v_block_scale)
        {
            int64_t tgt_value_idx = block_idx * num_heads * head_size * ori_block_size +
                                    head_idx * head_size * block_size;
#pragma unroll
            for(int id = 0; id < block_offset * head_size; id += blockDim.x)
            {
                if(id + threadIdx.x < block_offset * head_size)
                {
                    int64_t cache_idx;
                    int block_offset_local = (id + threadIdx.x) / head_size;
                    int cur_block_id       = block_offset_local / block_size;
                    block_offset_local     = block_offset_local % block_size;
                    if constexpr(asmLayout)
                    {
                        int head_offset             = (id + threadIdx.x) % head_size;
                        int block_offset_local_divX = block_offset_local / x;
                        int x_idx                   = block_offset_local % x;
                        cache_idx =
                            tgt_value_idx + cur_block_id * num_heads * head_size * block_size +
                            block_offset_local_divX * head_size * x + head_offset * x + x_idx;
                    }
                    else
                    {
                        int head_offset = (id + threadIdx.x) % head_size;
                        cache_idx       = tgt_value_idx +
                                    cur_block_id * num_heads * head_size * block_size +
                                    head_offset * block_size + block_offset_local;
                    }
                    float tmp              = ck_tile::type_convert<float>(value_cache[cache_idx]);
                    tmp                    = tmp * v_block_scale_global / v_block_scale;
                    value_cache[cache_idx] = ck_tile::type_convert<cache_t>(tmp);
                }
            }
            v_dequant_scales[scale_idx] = v_block_scale;
        }
        else
        {
            v_block_scale = v_block_scale_global;
        }
    }
    else
    {
        k_dequant_scales[scale_idx] = k_block_scale;
        v_dequant_scales[scale_idx] = v_block_scale;
    }
    k_block_scale = 1 / k_block_scale;
    v_block_scale = 1 / v_block_scale;

    // now let's store out
    block_idx = block_idx * (ori_block_size / block_size);
    for(int id = 0; id < numtokens_in_block * head_size; id += blockDim.x)
    {
        if((id + threadIdx.x) < numtokens_in_block * head_size)
        {
            int token_idx           = (id + threadIdx.x) / head_size + first_token_idx;
            int current_d           = (id + threadIdx.x) % head_size;
            int block_offset_local  = token_idx - first_token_idx + block_offset;
            int64_t block_idx_local = block_offset_local / block_size + block_idx;
            block_offset_local      = block_offset_local % block_size;

            const int64_t src_k_idx = token_idx * key_stride + head_idx * head_size + current_d;
            const int64_t src_v_idx = token_idx * value_stride + head_idx * head_size + current_d;
            float tmp_k             = ck_tile::type_convert<float>(key[src_k_idx]) * k_block_scale;
            float tmp_v = ck_tile::type_convert<float>(value[src_v_idx]) * v_block_scale;

            const int x_idx    = current_d / x;
            const int x_offset = current_d % x;
            //[num_blocks, num_heads, head_size/X, block_size, X]
            const int64_t tgt_key_idx = block_idx_local * num_heads * head_size * block_size +
                                        head_idx * head_size * block_size + x_idx * block_size * x +
                                        block_offset_local * x + x_offset;

            int64_t tgt_value_idx;
            if constexpr(asmLayout)
            { //[num_blocks, num_heads, block_size/X, head_size, X]
                const int x_idx    = block_offset_local / x;
                const int x_offset = block_offset_local % x;
                tgt_value_idx      = block_idx_local * num_heads * head_size * block_size +
                                head_idx * head_size * block_size + x_idx * head_size * x +
                                current_d * x + x_offset;
            }
            else
            { //[num_blocks, num_heads, head_size, block_size]
                tgt_value_idx = block_idx_local * num_heads * head_size * block_size +
                                head_idx * head_size * block_size + current_d * block_size +
                                block_offset_local;
            }
            // printf("tgt_key_idx%d, src_k_idx: %d, tmp_k:%f, k_block_scale:%f\n",tgt_key_idx,
            // src_k_idx, tmp_k, k_block_scale);
            key_cache[tgt_key_idx]     = ck_tile::type_convert<cache_t>(tmp_k);
            value_cache[tgt_value_idx] = ck_tile::type_convert<cache_t>(tmp_v);
        }
    }
}
} // namespace aiter

// KV_T is the stored data type of kv-cache.
// CACHE_T is the data type of key and value tensors.
// KV_DTYPE is the real data type of kv-cache.
#define CALL_RESHAPE_AND_CACHE(KV_T, CACHE_T, KV_DTYPE)                                          \
    aiter::reshape_and_cache_kernel<KV_T, CACHE_T, KV_DTYPE>                                     \
        <<<grid, block, 0, stream>>>(reinterpret_cast<KV_T*>(key.data_ptr()),                    \
                                     reinterpret_cast<KV_T*>(value.data_ptr()),                  \
                                     reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),           \
                                     reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),         \
                                     slot_mapping.data_ptr<int64_t>(),                           \
                                     key_stride,                                                 \
                                     value_stride,                                               \
                                     num_heads,                                                  \
                                     head_size,                                                  \
                                     block_size,                                                 \
                                     x,                                                          \
                                     k_scale.has_value() ? k_scale->data_ptr<float>() : nullptr, \
                                     v_scale.has_value() ? v_scale->data_ptr<float>() : nullptr);

#define CALL_RESHAPE_AND_CACHE_ASM(KV_T, CACHE_T, KV_DTYPE)                                      \
    aiter::reshape_and_cache_kernel<KV_T, CACHE_T, KV_DTYPE, true>                               \
        <<<grid, block, 0, stream>>>(reinterpret_cast<KV_T*>(key.data_ptr()),                    \
                                     reinterpret_cast<KV_T*>(value.data_ptr()),                  \
                                     reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),           \
                                     reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),         \
                                     slot_mapping.data_ptr<int64_t>(),                           \
                                     key_stride,                                                 \
                                     value_stride,                                               \
                                     num_heads,                                                  \
                                     head_size,                                                  \
                                     block_size,                                                 \
                                     x,                                                          \
                                     k_scale.has_value() ? k_scale->data_ptr<float>() : nullptr, \
                                     v_scale.has_value() ? v_scale->data_ptr<float>() : nullptr);

namespace aiter {

void reshape_and_cache(
    torch::Tensor& key,          // [num_tokens, num_heads, head_size]
    torch::Tensor& value,        // [num_tokens, num_heads, head_size]
    torch::Tensor& key_cache,    // [num_blocks, num_heads, head_size/x, block_size, x]
    torch::Tensor& value_cache,  // [num_blocks, num_heads, head_size, block_size]
    torch::Tensor& slot_mapping, // [num_tokens]
    const std::string& kv_cache_dtype,
    std::optional<torch::Tensor> k_scale,
    std::optional<torch::Tensor> v_scale,
    const bool asm_layout)
{
    int num_tokens = key.size(0);
    int num_heads  = key.size(1);
    int head_size  = key.size(2);
    int block_size = key_cache.size(3);
    int x          = key_cache.size(4);

    int key_stride   = key.stride(0);
    int value_stride = value.stride(0);

    dim3 grid(num_tokens);
    dim3 block(std::min(num_heads * head_size, 512));
    const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if(asm_layout)
    {
        DISPATCH_BY_KV_CACHE_DTYPE(key.dtype(), kv_cache_dtype, CALL_RESHAPE_AND_CACHE_ASM)
    }
    else
    {
        DISPATCH_BY_KV_CACHE_DTYPE(key.dtype(), kv_cache_dtype, CALL_RESHAPE_AND_CACHE)
    }
}

} // namespace aiter

// KV_T is the stored data type of kv-cache.
// CACHE_T is the data type of key and value tensors.
// KV_DTYPE is the real data type of kv-cache.
#define CALL_RESHAPE_AND_CACHE_FLASH(KV_T, CACHE_T, KV_DTYPE)                            \
    aiter::reshape_and_cache_flash_kernel<KV_T, CACHE_T, KV_DTYPE>                       \
        <<<grid, block, 0, stream>>>(reinterpret_cast<KV_T*>(key.data_ptr()),            \
                                     reinterpret_cast<KV_T*>(value.data_ptr()),          \
                                     reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),   \
                                     reinterpret_cast<CACHE_T*>(value_cache.data_ptr()), \
                                     slot_mapping.data_ptr<int64_t>(),                   \
                                     block_stride,                                       \
                                     key_stride,                                         \
                                     value_stride,                                       \
                                     num_heads,                                          \
                                     head_size,                                          \
                                     block_size,                                         \
                                     k_scale.data_ptr<float>(),                          \
                                     v_scale.data_ptr<float>());

namespace aiter {

void reshape_and_cache_flash(
    torch::Tensor& key,          // [num_tokens, num_heads, head_size]
    torch::Tensor& value,        // [num_tokens, num_heads, head_size]
    torch::Tensor& key_cache,    // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor& value_cache,  // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor& slot_mapping, // [num_tokens]
    const std::string& kv_cache_dtype,
    torch::Tensor& k_scale,
    torch::Tensor& v_scale)
{
    int num_tokens = key.size(0);
    int num_heads  = key.size(1);
    int head_size  = key.size(2);
    int block_size = key_cache.size(1);

    int key_stride   = key.stride(0);
    int value_stride = value.stride(0);
    int block_stride = key_cache.stride(0);
    TORCH_CHECK(key_cache.stride(0) == value_cache.stride(0));

    dim3 grid(num_tokens);
    dim3 block(std::min(num_heads * head_size, 512));
    const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    DISPATCH_BY_KV_CACHE_DTYPE(key.dtype(), kv_cache_dtype, CALL_RESHAPE_AND_CACHE_FLASH);
}
} // namespace aiter

// KV_T is the stored data type of kv-cache.
// CACHE_T is the data type of key and value tensors.
// KV_DTYPE is the real data type of kv-cache.
#define CALL_RESHAPE_AND_CACHE_WITH_PERTOKEN_QUANT(KV_T, CACHE_T, dequant_scale_t)                 \
    if(asm_layout)                                                                                 \
    {                                                                                              \
        aiter::reshape_and_cache_with_per_token_quant_kernel<KV_T, CACHE_T, dequant_scale_t, true> \
            <<<grid, block, 0, stream>>>(                                                          \
                reinterpret_cast<KV_T*>(key.data_ptr()),                                           \
                reinterpret_cast<KV_T*>(value.data_ptr()),                                         \
                reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),                                  \
                reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),                                \
                reinterpret_cast<dequant_scale_t*>(k_dequant_scales.data_ptr()),                   \
                reinterpret_cast<dequant_scale_t*>(v_dequant_scales.data_ptr()),                   \
                slot_mapping.data_ptr<int64_t>(),                                                  \
                key_stride,                                                                        \
                value_stride,                                                                      \
                num_heads,                                                                         \
                head_size,                                                                         \
                block_size,                                                                        \
                x,                                                                                 \
                num_tokens,                                                                        \
                max_kv_tokens);                                                                    \
    }                                                                                              \
    else                                                                                           \
    {                                                                                              \
        aiter::reshape_and_cache_with_per_token_quant_kernel<KV_T, CACHE_T, dequant_scale_t>       \
            <<<grid, block, 0, stream>>>(                                                          \
                reinterpret_cast<KV_T*>(key.data_ptr()),                                           \
                reinterpret_cast<KV_T*>(value.data_ptr()),                                         \
                reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),                                  \
                reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),                                \
                reinterpret_cast<dequant_scale_t*>(k_dequant_scales.data_ptr()),                   \
                reinterpret_cast<dequant_scale_t*>(v_dequant_scales.data_ptr()),                   \
                slot_mapping.data_ptr<int64_t>(),                                                  \
                key_stride,                                                                        \
                value_stride,                                                                      \
                num_heads,                                                                         \
                head_size,                                                                         \
                block_size,                                                                        \
                x,                                                                                 \
                num_tokens,                                                                        \
                max_kv_tokens);                                                                    \
    }

#define CALL_RESHAPE_AND_CACHE_WITH_BLOCK_QUANT(KV_T, CACHE_T, dequant_scale_t)                \
    if(asm_layout)                                                                             \
    {                                                                                          \
        aiter::reshape_and_cache_with_block_quant_kernel<KV_T, CACHE_T, dequant_scale_t, true> \
            <<<grid, block, 0, stream>>>(                                                      \
                reinterpret_cast<KV_T*>(key.data_ptr()),                                       \
                reinterpret_cast<KV_T*>(value.data_ptr()),                                     \
                reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),                              \
                reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),                            \
                reinterpret_cast<dequant_scale_t*>(k_dequant_scales.data_ptr()),               \
                reinterpret_cast<dequant_scale_t*>(v_dequant_scales.data_ptr()),               \
                slot_mapping.data_ptr<int64_t>(),                                              \
                key_stride,                                                                    \
                value_stride,                                                                  \
                num_heads,                                                                     \
                num_blocks,                                                                    \
                head_size,                                                                     \
                block_size,                                                                    \
                x,                                                                             \
                num_tokens,                                                                    \
                seq_len);                                                                      \
    }                                                                                          \
    else                                                                                       \
    {                                                                                          \
        aiter::reshape_and_cache_with_block_quant_kernel<KV_T, CACHE_T, dequant_scale_t>       \
            <<<grid, block, 0, stream>>>(                                                      \
                reinterpret_cast<KV_T*>(key.data_ptr()),                                       \
                reinterpret_cast<KV_T*>(value.data_ptr()),                                     \
                reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),                              \
                reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),                            \
                reinterpret_cast<dequant_scale_t*>(k_dequant_scales.data_ptr()),               \
                reinterpret_cast<dequant_scale_t*>(v_dequant_scales.data_ptr()),               \
                slot_mapping.data_ptr<int64_t>(),                                              \
                key_stride,                                                                    \
                value_stride,                                                                  \
                num_heads,                                                                     \
                num_blocks,                                                                    \
                head_size,                                                                     \
                block_size,                                                                    \
                x,                                                                             \
                num_tokens,                                                                    \
                seq_len);                                                                      \
    }

#define CALL_RESHAPE_AND_CACHE_WITH_BLOCK_QUANT_FOR_ASMPA(KV_T, CACHE_T, dequant_scale_t)          \
    if(asm_layout)                                                                                 \
    {                                                                                              \
        aiter::reshape_and_cache_with_block_quant_kernel_for_asmpa<KV_T,                           \
                                                                   CACHE_T,                        \
                                                                   dequant_scale_t,                \
                                                                   true>                           \
            <<<grid, block, 0, stream>>>(                                                          \
                reinterpret_cast<KV_T*>(key.data_ptr()),                                           \
                reinterpret_cast<KV_T*>(value.data_ptr()),                                         \
                reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),                                  \
                reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),                                \
                reinterpret_cast<dequant_scale_t*>(k_dequant_scales.data_ptr()),                   \
                reinterpret_cast<dequant_scale_t*>(v_dequant_scales.data_ptr()),                   \
                slot_mapping.data_ptr<int64_t>(),                                                  \
                key_stride,                                                                        \
                value_stride,                                                                      \
                num_heads,                                                                         \
                num_blocks,                                                                        \
                head_size,                                                                         \
                block_size,                                                                        \
                x,                                                                                 \
                num_tokens,                                                                        \
                seq_len,                                                                           \
                ori_block_size);                                                                   \
    }                                                                                              \
    else                                                                                           \
    {                                                                                              \
        aiter::reshape_and_cache_with_block_quant_kernel_for_asmpa<KV_T, CACHE_T, dequant_scale_t> \
            <<<grid, block, 0, stream>>>(                                                          \
                reinterpret_cast<KV_T*>(key.data_ptr()),                                           \
                reinterpret_cast<KV_T*>(value.data_ptr()),                                         \
                reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),                                  \
                reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),                                \
                reinterpret_cast<dequant_scale_t*>(k_dequant_scales.data_ptr()),                   \
                reinterpret_cast<dequant_scale_t*>(v_dequant_scales.data_ptr()),                   \
                slot_mapping.data_ptr<int64_t>(),                                                  \
                key_stride,                                                                        \
                value_stride,                                                                      \
                num_heads,                                                                         \
                num_blocks,                                                                        \
                head_size,                                                                         \
                block_size,                                                                        \
                x,                                                                                 \
                num_tokens,                                                                        \
                seq_len,                                                                           \
                ori_block_size);                                                                   \
    }

namespace aiter {

void reshape_and_cache_with_pertoken_quant(
    torch::Tensor& key,              // [num_tokens, num_heads, head_size]
    torch::Tensor& value,            // [num_tokens, num_heads, head_size]
    torch::Tensor& key_cache,        // [num_blocks, num_heads, head_size/x, block_size, x]
    torch::Tensor& value_cache,      // [num_blocks, num_heads, head_size, block_size]
    torch::Tensor& k_dequant_scales, // [num_heads, max_kv_tokens]
    torch::Tensor& v_dequant_scales, // [num_heads, max_kv_tokens]
    torch::Tensor& slot_mapping,     // [num_tokens]
    const bool asm_layout)
{
    int num_tokens    = key.size(0);
    int num_heads     = key.size(1);
    int head_size     = key.size(2);
    int block_size    = key_cache.size(3);
    int x             = key_cache.size(4);
    int max_kv_tokens = k_dequant_scales.size(1);

    int key_stride   = key.stride(0);
    int value_stride = value.stride(0);

    dim3 grid((num_tokens + 3) / 4, num_heads);
    dim3 block(256);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    using dequant_scale_t = float; // should align with k_dequant_scales/v_dequant_scales dtype

    float dtypeMax;
    if(key_cache.dtype() == torch_fp8)
    {
        if(key.dtype() == at::ScalarType::Float)
        {
            CALL_RESHAPE_AND_CACHE_WITH_PERTOKEN_QUANT(float, ck_tile::fp8_t, dequant_scale_t);
        }
        else if(key.dtype() == at::ScalarType::Half)
        {
            CALL_RESHAPE_AND_CACHE_WITH_PERTOKEN_QUANT(
                ck_tile::fp16_t, ck_tile::fp8_t, dequant_scale_t);
        }
        else if(key.dtype() == at::ScalarType::BFloat16)
        {
            CALL_RESHAPE_AND_CACHE_WITH_PERTOKEN_QUANT(
                ck_tile::bf16_t, ck_tile::fp8_t, dequant_scale_t);
        }
        else
        {
            TORCH_CHECK(false, "Unsupported input type of kv: ", key.dtype());
        }
    }
    else if(key_cache.dtype() == at::ScalarType::Char)
    {
        if(key.dtype() == at::ScalarType::Float)
        {
            CALL_RESHAPE_AND_CACHE_WITH_PERTOKEN_QUANT(float, int8_t, dequant_scale_t);
        }
        else if(key.dtype() == at::ScalarType::Half)
        {
            CALL_RESHAPE_AND_CACHE_WITH_PERTOKEN_QUANT(ck_tile::fp16_t, int8_t, dequant_scale_t);
        }
        else if(key.dtype() == at::ScalarType::BFloat16)
        {
            CALL_RESHAPE_AND_CACHE_WITH_PERTOKEN_QUANT(ck_tile::bf16_t, int8_t, dequant_scale_t);
        }
        else
        {
            TORCH_CHECK(false,
                        "Unsupported input type of kv: ",
                        key.dtype(),
                        " kv cache: ",
                        key_cache.dtype());
        }
    }
    else
    {
        TORCH_CHECK(false, "Unsupported data type of kv cache: ", key_cache.dtype());
    }
}

void reshape_and_cache_with_block_quant(
    torch::Tensor& key,              // [batch_size, seq_len, num_heads, head_size]
    torch::Tensor& value,            // [batch_size, seq_len, num_heads, head_size]
    torch::Tensor& key_cache,        // [num_blocks, num_heads, head_size/x, block_size, x]
    torch::Tensor& value_cache,      // [num_blocks, num_heads, head_size, block_size]
    torch::Tensor& k_dequant_scales, // [num_heads, num_blocks]
    torch::Tensor& v_dequant_scales, // [num_heads, num_blocks]
    torch::Tensor& slot_mapping,     // [num_tokens]
    const bool asm_layout)
{
    int batch_size = key.size(0);
    int seq_len    = key.size(1);
    int num_heads  = key.size(2);
    int head_size  = key.size(3);
    int num_blocks = key_cache.size(0);
    int block_size = key_cache.size(3);
    int x          = key_cache.size(4);
    int num_tokens = batch_size * seq_len;

    int key_stride   = key.stride(0) / seq_len;
    int value_stride = value.stride(0) / seq_len;
    int blockDimx    = (block_size + 255) / 256 * 256;

    dim3 grid(batch_size, (seq_len + block_size - 1) / block_size + 1, num_heads);
    dim3 block(blockDimx);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    using dequant_scale_t = float; // should align with k_dequant_scales/v_dequant_scales dtype

    float dtypeMax;
    if(key_cache.dtype() == torch_fp8)
    {
        if(key.dtype() == at::ScalarType::Float)
        {
            CALL_RESHAPE_AND_CACHE_WITH_BLOCK_QUANT(float, ck_tile::fp8_t, dequant_scale_t);
        }
        else if(key.dtype() == at::ScalarType::Half)
        {
            CALL_RESHAPE_AND_CACHE_WITH_BLOCK_QUANT(
                ck_tile::fp16_t, ck_tile::fp8_t, dequant_scale_t);
        }
        else if(key.dtype() == at::ScalarType::BFloat16)
        {
            CALL_RESHAPE_AND_CACHE_WITH_BLOCK_QUANT(
                ck_tile::bf16_t, ck_tile::fp8_t, dequant_scale_t);
        }
        else
        {
            TORCH_CHECK(false, "Unsupported input type of kv: ", key.dtype());
        }
    }
    else if(key_cache.dtype() == at::ScalarType::Char)
    {
        if(key.dtype() == at::ScalarType::Float)
        {
            CALL_RESHAPE_AND_CACHE_WITH_BLOCK_QUANT(float, ck_tile::int8_t, dequant_scale_t);
        }
        else if(key.dtype() == at::ScalarType::Half)
        {
            CALL_RESHAPE_AND_CACHE_WITH_BLOCK_QUANT(
                ck_tile::fp16_t, ck_tile::int8_t, dequant_scale_t);
        }
        else if(key.dtype() == at::ScalarType::BFloat16)
        {
            CALL_RESHAPE_AND_CACHE_WITH_BLOCK_QUANT(
                ck_tile::bf16_t, ck_tile::int8_t, dequant_scale_t);
        }
        else
        {
            TORCH_CHECK(false,
                        "Unsupported input type of kv: ",
                        key.dtype(),
                        " kv cache: ",
                        key_cache.dtype());
        }
    }
    else
    {
        TORCH_CHECK(false, "Unsupported data type of kv cache: ", key_cache.dtype());
    }
}

void reshape_and_cache_with_block_quant_for_asm_pa(
    torch::Tensor& key,              // [batch_size, seq_len, num_heads, head_size]
    torch::Tensor& value,            // [batch_size, seq_len, num_heads, head_size]
    torch::Tensor& key_cache,        // [num_blocks, num_heads, head_size/x, block_size:16, x]
    torch::Tensor& value_cache,      // [num_blocks, num_heads, head_size, block_size:16]
    torch::Tensor& k_dequant_scales, // [num_heads, num_blocks/(ori_block_size/block_size:16)]
    torch::Tensor& v_dequant_scales, // [num_heads, num_blocks/(ori_block_size/block_size:16)]
    torch::Tensor& slot_mapping,     // [num_tokens]
    const bool asm_layout,
    const int ori_block_size = 128)
{
    TORCH_CHECK(
        key.dim() == 4 && value.dim() == 4,
        "key/value must be a 4D tensor with shape [batch_size, seq_len, num_heads, head_size]");
    TORCH_CHECK(ori_block_size == 128 || ori_block_size == 256,
                "ori_block_size only support 128/256");

    int batch_size   = key.size(0);
    int seq_len      = key.size(1);
    int num_heads    = key.size(2);
    int head_size    = key.size(3);
    int num_blocks   = key_cache.size(0);
    int block_size   = key_cache.size(3);
    int x            = key_cache.size(4);
    int num_tokens   = batch_size * seq_len;
    int key_stride   = key.stride(0) / seq_len;
    int value_stride = value.stride(0) / seq_len;

    int blockDimx = (ori_block_size + 255) / 256 * 256;
    dim3 grid(batch_size, (seq_len + ori_block_size - 1) / ori_block_size + 1, num_heads);
    dim3 block(blockDimx);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    using dequant_scale_t = float; // should align with k_dequant_scales/v_dequant_scales dtype

    if(key_cache.dtype() == torch_fp8)
    {
        if(key.dtype() == at::ScalarType::Float)
        {
            CALL_RESHAPE_AND_CACHE_WITH_BLOCK_QUANT_FOR_ASMPA(
                float, ck_tile::fp8_t, dequant_scale_t);
        }
        else if(key.dtype() == at::ScalarType::Half)
        {
            CALL_RESHAPE_AND_CACHE_WITH_BLOCK_QUANT_FOR_ASMPA(
                ck_tile::fp16_t, ck_tile::fp8_t, dequant_scale_t);
        }
        else if(key.dtype() == at::ScalarType::BFloat16)
        {
            CALL_RESHAPE_AND_CACHE_WITH_BLOCK_QUANT_FOR_ASMPA(
                ck_tile::bf16_t, ck_tile::fp8_t, dequant_scale_t);
        }
        else
        {
            TORCH_CHECK(false, "Unsupported input type of kv: ", key.dtype());
        }
    }
    else if(key_cache.dtype() == at::ScalarType::Char)
    {
        if(key.dtype() == at::ScalarType::Float)
        {
            CALL_RESHAPE_AND_CACHE_WITH_BLOCK_QUANT_FOR_ASMPA(
                float, ck_tile::int8_t, dequant_scale_t);
        }
        else if(key.dtype() == at::ScalarType::Half)
        {
            CALL_RESHAPE_AND_CACHE_WITH_BLOCK_QUANT_FOR_ASMPA(
                ck_tile::fp16_t, ck_tile::int8_t, dequant_scale_t);
        }
        else if(key.dtype() == at::ScalarType::BFloat16)
        {
            CALL_RESHAPE_AND_CACHE_WITH_BLOCK_QUANT_FOR_ASMPA(
                ck_tile::bf16_t, ck_tile::int8_t, dequant_scale_t);
        }
        else
        {
            TORCH_CHECK(false,
                        "Unsupported input type of kv: ",
                        key.dtype(),
                        " kv cache: ",
                        key_cache.dtype());
        }
    }
    else
    {
        TORCH_CHECK(false, "Unsupported data type of kv cache: ", key_cache.dtype());
    }
}
} // namespace aiter

