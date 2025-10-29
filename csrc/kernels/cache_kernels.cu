// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
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

namespace aiter {

void swap_blocks(torch::Tensor& src, torch::Tensor& dst, const torch::Tensor& block_mapping)
{
    torch::Device src_device = src.device();
    torch::Device dst_device = dst.device();
    hipMemcpyKind memcpy_type;
    if(src_device.is_cuda() && dst_device.is_cuda())
    {
        TORCH_CHECK(src_device.index() == dst_device.index(),
                    "src and dst must be on the same GPU");
        memcpy_type = hipMemcpyDeviceToDevice;
    }
    else if(src_device.is_cuda() && dst_device.is_cpu())
    {
        memcpy_type = hipMemcpyDeviceToHost;
    }
    else if(src_device.is_cpu() && dst_device.is_cuda())
    {
        memcpy_type = hipMemcpyHostToDevice;
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
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(
        src_device.is_cuda() ? src_device : dst_device);
    const hipStream_t stream = at::hip::getCurrentHIPStream();
    // NOTE(woosuk): This can be slow if the number of blocks is large.
    const int64_t num_blocks = block_mapping.size(0);
    for(size_t i = 0; i < num_blocks; i++)
    {
        int64_t src_block_number = block_mapping[i][0].item<int64_t>();
        int64_t dst_block_number = block_mapping[i][1].item<int64_t>();
        int64_t src_offset       = src_block_number * block_size_in_bytes;
        int64_t dst_offset       = dst_block_number * block_size_in_bytes;
        hipMemcpyAsync(
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
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(cache_device);
    const hipStream_t stream = at::hip::getCurrentHIPStream();
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
    const float inverted_kscale = k_scale == nullptr ? 1.0f : 1 / (*k_scale);
    const float inverted_vscale = v_scale == nullptr ? 1.0f : 1 / (*v_scale);
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
          int wg_size    = 64>
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

    float k_token_scale          = k_max / dtypeMax;
    float v_token_scale          = v_max / dtypeMax;
    float k_token_scale_inverted = 1.0 / k_token_scale;
    float v_token_scale_inverted = 1.0 / v_token_scale;

#pragma unroll
    for(int i_d = 0; i_d < local_dim_elems; i_d++)
    {
        k_local_dim[i_d] = k_local_dim[i_d] * k_token_scale_inverted;
        v_local_dim[i_d] = v_local_dim[i_d] * v_token_scale_inverted;
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
    auto sum               = [](float a, float b) { return a + b; };
    int numtokens_in_block = block_reduce<int, decltype(sum), wg_size, true>(tokens_in_block, sum);

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

    k_max_val = block_reduce<float, decltype(f_max_f32), wg_size, true>(k_max_val, f_max_f32);
    v_max_val = block_reduce<float, decltype(f_max_f32), wg_size, true>(v_max_val, f_max_f32);

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
    auto sum = [](float a, float b) { return a + b; };
    int numtokens_in_block =
        block_reduce<float, decltype(sum), wg_size, true>(tokens_in_block, sum);

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

    k_max_val = block_reduce<float, decltype(f_max_f32), wg_size, true>(k_max_val, f_max_f32);
    v_max_val = block_reduce<float, decltype(f_max_f32), wg_size, true>(v_max_val, f_max_f32);

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
template <typename scalar_t, typename cache_t, vllm::Fp8KVCacheDataType kv_dt>
__global__ void concat_and_cache_mla_kernel(
    const scalar_t* __restrict__ kv_c,        // [num_tokens, kv_lora_rank]
    const scalar_t* __restrict__ k_pe,        // [num_tokens, pe_dim]
    cache_t* __restrict__ kv_cache,           // [num_blocks, block_size, (kv_lora_rank
                                              // + pe_dim)]
    const int64_t* __restrict__ slot_mapping, // [num_tokens]
    const int block_stride,                   //
    const int entry_stride,                   //
    const int kv_c_stride,                    //
    const int k_pe_stride,                    //
    const int kv_lora_rank,                   //
    const int pe_dim,                         //
    const int block_size,                     //
    const float* scale                        //
)
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
    const float inverted_kscale = 1.0f / *scale;
    auto copy                   = [&](const scalar_t* __restrict__ src,
                    cache_t* __restrict__ dst,
                    int src_stride,
                    int dst_stride,
                    int size,
                    int offset) {
        for(int i = threadIdx.x; i < size; i += blockDim.x)
        {
            const int64_t src_idx = token_idx * src_stride + i;
            const int64_t dst_idx =
                block_idx * block_stride + block_offset * entry_stride + i + offset;
            if constexpr(kv_dt == vllm::Fp8KVCacheDataType::kAuto)
            {
                dst[dst_idx] = src[src_idx];
            }
            else
            {
                dst[dst_idx] = ck_tile::type_convert<cache_t>(
                    ck_tile::type_convert<float>(src[src_idx]) * inverted_kscale);
            }
        }
    };
    copy(kv_c, kv_cache, kv_c_stride, block_stride, kv_lora_rank, 0);
    copy(k_pe, kv_cache, k_pe_stride, block_stride, pe_dim, kv_lora_rank);
}

template <typename scalar_t, typename cache_t, vllm::Fp8KVCacheDataType kv_dt>
__global__ void concat_and_cache_mla_opt_kernel(
    const scalar_t* __restrict__ kv_c,        // [num_tokens, kv_lora_rank]
    const scalar_t* __restrict__ k_pe,        // [num_tokens, pe_dim]
    cache_t* __restrict__ kv_cache,           // [num_blocks, block_size, (kv_lora_rank
                                              // + pe_dim)]
    const int64_t* __restrict__ slot_mapping, // [num_tokens]
    const int block_stride,                   //
    const int entry_stride,                   //
    const int kv_c_stride,                    //
    const int k_pe_stride,                    //
    const int kv_lora_rank,                   //
    const int pe_dim,                         //
    const int block_size,                     //
    const float* scale                        //
)
{
    const int64_t token_idx = blockIdx.x;
    const int64_t slot_idx  = slot_mapping[token_idx];
    // NOTE: slot_idx can be -1 if the token is padded
    if(slot_idx < 0)
    {
        return;
    }
    const int64_t block_idx             = slot_idx / block_size;
    const int64_t block_offset          = slot_idx % block_size;
    const float inverted_kscale         = 1.0f / *scale;
    static constexpr int32_t vec_size_i = std::is_same_v<scalar_t, float> ? 4 : 8;
    static constexpr int32_t vec_size_o = vec_size_i;
    using vec_i                         = ck_tile::vec_t<scalar_t, vec_size_i>;
    static constexpr int32_t ooba_i     = 4 / sizeof(scalar_t);
    static constexpr int32_t ooba_o     = 4 / sizeof(cache_t);
    auto out_offset                     = block_idx * block_stride + block_offset * entry_stride;
    auto copy                           = [&](const scalar_t* __restrict__ src,
                    cache_t* __restrict__ dst,
                    int src_stride,
                    int dst_stride,
                    int size,
                    int offset) {
        const int32_t oob_i = (size + ooba_i - 1) / ooba_i * ooba_i;
        const int32_t oob_o = (size + ooba_o - 1) / ooba_o * ooba_o;
        auto const* ptr_i   = reinterpret_cast<scalar_t const*>(src + token_idx * src_stride);
        auto* ptr_o         = reinterpret_cast<cache_t*>(dst + out_offset + offset);
        auto buffer_i =
            ck_tile::make_buffer_view<ck_tile::address_space_enum::global>(ptr_i, oob_i);
        buffer_i.init_raw();
        auto buffer_o =
            ck_tile::make_buffer_view<ck_tile::address_space_enum::global>(ptr_o, oob_o);
        buffer_o.init_raw();

        // double load core loop start
        const int32_t num_vecs = (size + vec_size_i - 1) / vec_size_i;
        vec_i vec_nxt;
        vec_i vec_cur;

        size_t vec_idx    = threadIdx.x;
        size_t vec_stride = blockDim.x;
        if(vec_idx < num_vecs)
        {
            vec_cur = buffer_i.template get<vec_i>(vec_idx * vec_size_i, 0, true);
        }
        for(vec_idx += vec_stride; vec_idx < num_vecs; vec_idx += vec_stride)
        {
            vec_nxt = buffer_i.template get<vec_i>(vec_idx * vec_size_i, 0, true);
            if constexpr(kv_dt == vllm::Fp8KVCacheDataType::kAuto)
            {
                buffer_o.template set((vec_idx - vec_stride) * vec_size_o,
                                      0,
                                      true,
                                      vec_cur.template get_as<cache_t>());
            }
            else
            {
                buffer_o.template set(
                    (vec_idx - vec_stride) * vec_size_o,
                    0,
                    true,
                    ck_tile::vec_convert<cache_t, scalar_t, vec_size_i>(vec_cur, inverted_kscale)
                        .template get_as<cache_t>());
            }
            vec_cur = vec_nxt;
        }
        if(vec_idx - vec_stride < num_vecs)
        {
            if constexpr(kv_dt == vllm::Fp8KVCacheDataType::kAuto)
            {
                buffer_o.template set((vec_idx - vec_stride) * vec_size_o,
                                      0,
                                      true,
                                      vec_cur.template get_as<cache_t>());
            }
            else
            {
                buffer_o.template set(
                    (vec_idx - vec_stride) * vec_size_o,
                    0,
                    true,
                    ck_tile::vec_convert<cache_t, scalar_t, vec_size_i>(vec_cur, inverted_kscale)
                        .template get_as<cache_t>());
            }
        }
    };

    copy(kv_c, kv_cache, kv_c_stride, block_stride, kv_lora_rank, 0);
    copy(k_pe, kv_cache, k_pe_stride, block_stride, pe_dim, kv_lora_rank);
}

template <typename scalar_t, typename cache_t, vllm::Fp8KVCacheDataType kv_dt>
__global__ void indexer_k_quant_and_cache_kernel(
    const scalar_t* __restrict__ k,           // [num_tokens, head_dim]
    cache_t* __restrict__ kv_cache,           // [num_blocks, block_size, cache_stride]
    const int64_t* __restrict__ slot_mapping, // [num_tokens]
    const int head_dim,                       // dimension of each head
    const int quant_block_size,               // quantization block size
    const int cache_block_size,               // cache block size
    const int cache_stride,                   // stride for each token in kv_cache
    const bool use_ue8m0                      // use ue8m0 scale format
)
{
    constexpr int VEC_SIZE  = 4;
    const int64_t token_idx = blockIdx.x;
    const int64_t head_dim_idx =
        (blockIdx.y * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x) * VEC_SIZE;
    const int64_t slot_idx     = slot_mapping[token_idx];
    const int64_t block_idx    = slot_idx / cache_block_size;
    const int64_t block_offset = slot_idx % cache_block_size;
    using vec_i                = ck_tile::vec_t<scalar_t, VEC_SIZE>;
    using vec_o                = ck_tile::vec_t<cache_t, VEC_SIZE>;

    // NOTE: slot_idx can be -1 if the token is padded
    if(slot_idx < 0 || (head_dim_idx >= head_dim))
    {
        return;
    }

    vec_i k_val =
        (reinterpret_cast<const vec_i*>(k))[(token_idx * head_dim + head_dim_idx) / VEC_SIZE];

    // float amax = 0.0f;
    // for(int i = 0; i < VEC_SIZE; i++)
    // {
    //     amax = fmaxf(amax, fabsf(float(k_val[i])));
    // }

    // // Reduced amax
    // for(int mask = 16; mask > 0; mask /= 2)
    // {
    //     amax = fmaxf(amax, __shfl_xor_sync(uint64_t(-1), amax, mask));
    // }
    // float scale = fmaxf(amax, 1e-4) / ck_tile::type_convert<float>(ck_tile::numeric<cache_t>::max());
    // if(use_ue8m0)
    // {
    //     scale = exp2f(ceilf(log2f(scale)));
    // }

    // const int64_t dst_offset =
    //     block_idx * cache_block_size * cache_stride + block_offset * head_dim + head_dim_idx;

    //   for (int i = 0; i < VEC_SIZE; i++) {
    //     kv_cache[dst_offset + i] = ck_tile::type_convert<cache_t>(k_val[i] / scale);
        // kv_cache[dst_offset + i] =
        //     fp8::scaled_convert<cache_t, scalar_t, kv_dt>(k_val_ptr[i], scale);
    //   }
    // if(threadIdx.x == 0)
    // {
    //     const int64_t dst_scale_idx =
    //         block_idx * cache_block_size * cache_stride + cache_block_size * head_dim +
    //         (block_offset * head_dim + head_dim_idx) * 4 / quant_block_size;
    //     reinterpret_cast<float*>(kv_cache)[dst_scale_idx / 4] = scale;
    // }
    // scale = 1.0f / scale;
    // vec_o* kv_cache_vec = reinterpret_cast<vec_o*>(kv_cache + dst_offset);
    // *kv_cache_vec = ck_tile::vec_convert<cache_t, scalar_t, VEC_SIZE>(k_val, scale);
}

template <int BLOCK_X_SIZE, int BLOCK_Y_SIZE>
__global__ void cp_gather_indexer_k_quant_cache_kernel(
    const char* __restrict__ kv_cache,   // [num_blocks, block_size,
                                         // cache_stride]
    char* __restrict__ dst_k,            // [num_tokens, head_dim]
    char* __restrict__ dst_scale,        // [num_tokens, head_dim / quant_block_size *
                                         // 4]
    const int* __restrict__ block_table, // [batch_size, num_blocks]
    const int* __restrict__ cu_seq_lens, // [batch_size + 1]
    const int batch_size,                // batch size
    const int64_t token_stride,          // stride for each token in dst_k
    const int64_t head_dim,              // dimension of each head
    const int64_t block_stride,          // stride for each block in kv_cache
    const int64_t cache_token_stride,    // stride for each token in kv_cache
    const int64_t cache_block_size,      // num_tokens for each block in kv_cache
    const int num_blocks,                // number of blocks
    const int num_tokens,                // number of tokens
    const int quant_block_size           // quantization block size
)
{
    constexpr int VEC_SIZE = sizeof(float4) / sizeof(char);
    const int token_idx    = blockIdx.x * blockDim.y + threadIdx.y;
    const int head_idx     = (blockIdx.y * blockDim.x + threadIdx.x) * VEC_SIZE;
    // Find batch index within a block
    __shared__ int batch_idx[BLOCK_Y_SIZE];
    for(int iter = 0; iter < (batch_size - BLOCK_X_SIZE + 1) / BLOCK_X_SIZE; iter++)
    {
        int tid = iter * blockDim.x + threadIdx.x;
        if(tid < batch_size)
        {
            const int seq_start = cu_seq_lens[tid];
            const int seq_end   = cu_seq_lens[tid + 1];
            if(token_idx >= seq_start && token_idx < seq_end)
            {
                batch_idx[threadIdx.y] = tid;
            }
        }
    }

    if(head_idx >= head_dim || token_idx >= num_tokens)
    {
        return;
    }
    const int inbatch_seq_idx = token_idx - cu_seq_lens[batch_idx[threadIdx.y]];
    const int block_idx =
        block_table[batch_idx[threadIdx.y] * num_blocks + inbatch_seq_idx / cache_block_size];
    const int64_t src_block_offset     = block_idx * block_stride;
    const int64_t cache_inblock_offset = (inbatch_seq_idx % cache_block_size) * head_dim + head_idx;
    const int64_t src_inblock_offset   = src_block_offset + cache_inblock_offset;
    const int64_t dst_inblock_offset   = token_idx * token_stride + head_idx;

    reinterpret_cast<float4*>(dst_k)[dst_inblock_offset / VEC_SIZE] =
        reinterpret_cast<const float4*>(kv_cache)[src_inblock_offset / VEC_SIZE];
    ;
    if(threadIdx.x == 0)
    {
        const int64_t src_scale_offset = src_block_offset + cache_block_size * head_dim +
                                         cache_inblock_offset * 4 / quant_block_size;
        reinterpret_cast<float*>(dst_scale)[dst_inblock_offset / quant_block_size] =
            reinterpret_cast<const float*>(kv_cache)[src_scale_offset / 4];
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
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(key));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

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
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(key));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

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

// KV_T is the data type of key and value tensors.
// CACHE_T is the stored data type of kv-cache.
// KV_DTYPE is the real data type of kv-cache.
#define CALL_CONCAT_AND_CACHE_MLA(KV_T, CACHE_T, KV_DTYPE)                            \
    aiter::concat_and_cache_mla_kernel<KV_T, CACHE_T, KV_DTYPE>                       \
        <<<grid, block, 0, stream>>>(reinterpret_cast<KV_T*>(kv_c.data_ptr()),        \
                                     reinterpret_cast<KV_T*>(k_pe.data_ptr()),        \
                                     reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()), \
                                     slot_mapping.data_ptr<int64_t>(),                \
                                     block_stride,                                    \
                                     entry_stride,                                    \
                                     kv_c_stride,                                     \
                                     k_pe_stride,                                     \
                                     kv_lora_rank,                                    \
                                     pe_dim,                                          \
                                     block_size,                                      \
                                     reinterpret_cast<const float*>(scale.data_ptr()));

#define CALL_CONCAT_AND_CACHE_MLA_OPT(KV_T, CACHE_T, KV_DTYPE)                        \
    aiter::concat_and_cache_mla_opt_kernel<KV_T, CACHE_T, KV_DTYPE>                   \
        <<<grid, block, 0, stream>>>(reinterpret_cast<KV_T*>(kv_c.data_ptr()),        \
                                     reinterpret_cast<KV_T*>(k_pe.data_ptr()),        \
                                     reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()), \
                                     slot_mapping.data_ptr<int64_t>(),                \
                                     block_stride,                                    \
                                     entry_stride,                                    \
                                     kv_c_stride,                                     \
                                     k_pe_stride,                                     \
                                     kv_lora_rank,                                    \
                                     pe_dim,                                          \
                                     block_size,                                      \
                                     reinterpret_cast<const float*>(scale.data_ptr()));

// Macro to dispatch the kernel based on the data type.
#define CALL_INDEXER_K_QUANT_AND_CACHE(KV_T, CACHE_T, KV_DTYPE)                       \
    aiter::indexer_k_quant_and_cache_kernel<KV_T, CACHE_T, KV_DTYPE>                  \
        <<<grid, block, 0, stream>>>(reinterpret_cast<KV_T*>(k.data_ptr()),           \
                                     reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()), \
                                     slot_mapping.data_ptr<int64_t>(),                \
                                     head_dim,                                        \
                                     quant_block_size,                                \
                                     cache_block_size,                                \
                                     cache_stride,                                    \
                                     use_ue8m0);

// Macro to dispatch the kernel based on the data amount.
#define CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(BLOCK_Y_SIZE)          \
    aiter::cp_gather_indexer_k_quant_cache_kernel<8, BLOCK_Y_SIZE>  \
        <<<dim3((num_tokens + BLOCK_Y_SIZE - 1) / BLOCK_Y_SIZE,     \
                (head_dim + 8 * vec_size - 1) / (8 * vec_size)),    \
           dim3(8, BLOCK_Y_SIZE),                                   \
           0,                                                       \
           stream>>>(reinterpret_cast<char*>(kv_cache.data_ptr()),  \
                     reinterpret_cast<char*>(dst_k.data_ptr()),     \
                     reinterpret_cast<char*>(dst_scale.data_ptr()), \
                     block_table.data_ptr<int32_t>(),               \
                     cu_seq_lens.data_ptr<int32_t>(),               \
                     batch_size,                                    \
                     dst_k.stride(0),                               \
                     dst_k.size(1),                                 \
                     kv_cache.stride(0),                            \
                     kv_cache.stride(1),                            \
                     kv_cache.size(1),                              \
                     block_table.size(1),                           \
                     num_tokens,                                    \
                     quant_block_size);

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
    TORCH_CHECK(head_size <= 512, __func__, " Unsupported head_size: ", head_size);

    int key_stride   = key.stride(0);
    int value_stride = value.stride(0);

    dim3 grid(num_tokens, num_heads);
    dim3 block(64);
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(key));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

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
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(key));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

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
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(key));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

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

void concat_and_cache_mla(torch::Tensor& kv_c,         // [num_tokens, kv_lora_rank]
                          torch::Tensor& k_pe,         // [num_tokens, pe_dim]
                          torch::Tensor& kv_cache,     // [num_blocks, block_size, (kv_lora_rank +
                                                       // pe_dim)]
                          torch::Tensor& slot_mapping, // [num_tokens] or [num_actual_tokens]
                          const std::string& kv_cache_dtype,
                          torch::Tensor& scale)
{
    int num_tokens   = slot_mapping.size(0);
    int kv_lora_rank = kv_c.size(1);
    int pe_dim       = k_pe.size(1);
    int block_size   = kv_cache.size(1);

    TORCH_CHECK(kv_cache.size(2) == kv_lora_rank + pe_dim);
    int kv_c_stride  = kv_c.stride(0);
    int k_pe_stride  = k_pe.stride(0);
    int block_stride = kv_cache.stride(0);
    int entry_stride = kv_cache.stride(1);
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(kv_c));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    if((pe_dim & 0x7) == 0 && (kv_lora_rank & 0x7) == 0)
    {
        dim3 grid(num_tokens);
        dim3 block(std::min(kv_lora_rank, 1024) / 8);
        DISPATCH_BY_KV_CACHE_DTYPE(kv_c.dtype(), kv_cache_dtype, CALL_CONCAT_AND_CACHE_MLA_OPT);
    }
    else
    {
        dim3 grid(num_tokens);
        dim3 block(std::min(kv_lora_rank, 512));
        DISPATCH_BY_KV_CACHE_DTYPE(kv_c.dtype(), kv_cache_dtype, CALL_CONCAT_AND_CACHE_MLA);
    }
}

// copy from vllm: https://github.com/vllm-project/vllm/blob/main/csrc/cache_kernels.cu
void indexer_k_quant_and_cache(torch::Tensor& k,        // [num_tokens, head_dim]
                               torch::Tensor& kv_cache, // [num_blocks, block_size, cache_stride]
                               torch::Tensor& slot_mapping, // [num_tokens]
                               int64_t quant_block_size,    // quantization block size
                               const std::string& scale_fmt)
{
    int num_tokens       = k.size(0);
    int head_dim         = k.size(1);
    int cache_block_size = kv_cache.size(1);
    int cache_stride     = kv_cache.size(2);
    bool use_ue8m0       = scale_fmt == "ue8m0";

    TORCH_CHECK(k.device() == kv_cache.device(), "k and kv_cache must be on the same device");
    TORCH_CHECK(k.device() == slot_mapping.device(),
                "k and slot_mapping must be on the same device");
    TORCH_CHECK(head_dim % quant_block_size == 0, "head_dim must be divisible by quant_block_size");

    constexpr int vec_size = 4;
    dim3 grid(num_tokens,
              (head_dim + quant_block_size * vec_size - 1) / (quant_block_size * vec_size));
    dim3 block(32, vec_size);
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(k));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    DISPATCH_BY_KV_CACHE_DTYPE(k.dtype(), "fp8_e4m3", CALL_INDEXER_K_QUANT_AND_CACHE);
}

// copy from vllm: https://github.com/vllm-project/vllm/blob/main/csrc/cache_kernels.cu
void cp_gather_indexer_k_quant_cache(
    const torch::Tensor& kv_cache,    // [num_blocks, block_size, cache_stride]
    torch::Tensor& dst_k,             // [num_tokens, head_dim]
    torch::Tensor& dst_scale,         // [num_tokens, head_dim / quant_block_size * 4]
    const torch::Tensor& block_table, // [batch_size, num_blocks]
    const torch::Tensor& cu_seq_lens  // [batch_size + 1]
)
{
    int batch_size       = block_table.size(0);
    int num_tokens       = dst_k.size(0);
    int head_dim         = dst_k.size(1);
    int quant_block_size = head_dim * 4 / dst_scale.size(1);

    TORCH_CHECK(kv_cache.device() == dst_k.device(),
                "kv_cache and dst_k must be on the same device");
    TORCH_CHECK(kv_cache.device() == dst_scale.device(),
                "kv_cache and dst_scale must be on the same device");
    TORCH_CHECK(kv_cache.device() == block_table.device(),
                "kv_cache and block_table must be on the same device");
    TORCH_CHECK(kv_cache.device() == cu_seq_lens.device(),
                "kv_cache and cu_seq_lens must be on the same device");
    TORCH_CHECK(head_dim % quant_block_size == 0, "head_dim must be divisible by quant_block_size");

    constexpr int vec_size = 16;
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(kv_cache));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    if(num_tokens < 32)
    {
        CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(1);
    }
    else if(num_tokens < 64)
    {
        CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(2);
    }
    else if(num_tokens < 128)
    {
        CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(4);
    }
    else if(num_tokens < 256)
    {
        CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(8);
    }
    else if(num_tokens < 512)
    {
        CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(16);
    }
    else
    {
        CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(32);
    }
}

} // namespace aiter
