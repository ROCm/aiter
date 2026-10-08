// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

// This translation unit is torch-free: define AITER_NO_TORCH_TYPES before any
// aiter header so aiter_opus_plus.h does not pull in the c10 half/bfloat16
// headers. The kernels use aiter::hip2opus + the _rmTorch dispatch macros, never
// the t2opus<c10::*> specializations, so nothing here needs torch/ATen/c10.
#define AITER_NO_TORCH_TYPES
#include "aiter_hip_common.h"
#include "aiter_dispatch.h"
#include "aiter_stream.h"
#include "cache.h"
#include "hip_reduce.h"

#include "attention_dtypes.h"
#include "opus/opus.hpp"
#include "aiter_opus_plus.h"

#include <algorithm>
#include <cassert>
#include <map>
#include <vector>

#include <hip/hip_bf16.h>

namespace aiter {

// slot_mapping arrives as int64 from the torch stacks -- vLLM allocates it as
// torch.int64 and SGLang's out_cache_loc is likewise int64 -- and as int32 from
// JAX front ends, which emit 32-bit integers unless jax_enable_x64 is set at
// startup, process-wide. The cache kernels therefore have to accept both.
//
// The pointer is passed untyped with a width flag rather than templating the
// kernels on the index type. Templating would double every instantiation
// produced by DISPATCH_BY_KV_CACHE_DTYPE_OPUS_rmTorch, and this translation unit
// is a long compile already. The runtime cost is nil: each kernel reads exactly
// one slot per thread block, and the flag is a kernel argument and therefore
// uniform across the grid, so it compiles to a scalar branch rather than a
// divergent one.
enum class SlotIndexWidth : int
{
    kInt32 = 0,
    kInt64 = 1,
};

__device__ __forceinline__ int64_t load_slot_index(const void* __restrict__ slot_mapping,
                                                   int64_t token_idx,
                                                   SlotIndexWidth width)
{
    return width == SlotIndexWidth::kInt64
               ? static_cast<const int64_t*>(slot_mapping)[token_idx]
               : static_cast<int64_t>(
                     static_cast<const int32_t*>(slot_mapping)[token_idx]);
}

inline SlotIndexWidth slot_index_width(const aiter_tensor_t& slot_mapping)
{
    AITER_CHECK(slot_mapping.dtype() == AITER_DTYPE_i32 ||
                    slot_mapping.dtype() == AITER_DTYPE_i64,
                "slot_mapping must be int32 or int64, got ",
                AiterDtype_to_str(slot_mapping.dtype()));
    return slot_mapping.dtype() == AITER_DTYPE_i32 ? SlotIndexWidth::kInt32
                                                   : SlotIndexWidth::kInt64;
}

void swap_blocks(aiter_tensor_t& src, aiter_tensor_t& dst, const aiter_tensor_t& block_mapping)
{
    bool src_is_gpu = src.is_gpu();
    bool dst_is_gpu = dst.is_gpu();
    hipMemcpyKind memcpy_type;
    if(src_is_gpu && dst_is_gpu)
    {
        AITER_CHECK(src.device_id == dst.device_id,
                    "src and dst must be on the same GPU");
        memcpy_type = hipMemcpyDeviceToDevice;
    }
    else if(src_is_gpu && !dst_is_gpu)
    {
        memcpy_type = hipMemcpyDeviceToHost;
    }
    else if(!src_is_gpu && dst_is_gpu)
    {
        memcpy_type = hipMemcpyHostToDevice;
    }
    else
    {
        AITER_CHECK(false, "Invalid device combination");
    }

    AITER_CHECK(block_mapping.is_cpu(), "block_mapping must be on CPU");

    char* src_ptr = static_cast<char*>(src.data_ptr());
    char* dst_ptr = static_cast<char*>(dst.data_ptr());

    const int64_t block_size_in_bytes = src.element_size() * (src.numel() / src.size(0));
    int guard_device = src_is_gpu ? src.device_id : dst.device_id;
    HipDeviceGuard device_guard(guard_device);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    const int64_t num_blocks = block_mapping.size(0);
    int64_t* mapping_ptr = static_cast<int64_t*>(block_mapping.data_ptr());
    for(size_t i = 0; i < num_blocks; i++)
    {
        int64_t src_block_number = mapping_ptr[i * 2];
        int64_t dst_block_number = mapping_ptr[i * 2 + 1];
        int64_t src_offset       = src_block_number * block_size_in_bytes;
        int64_t dst_offset       = dst_block_number * block_size_in_bytes;
        HIP_CALL(hipMemcpyAsync(
            dst_ptr + dst_offset, src_ptr + src_offset, block_size_in_bytes, memcpy_type, stream));
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

void copy_blocks(std::vector<aiter_tensor_t> const& key_caches,
                 std::vector<aiter_tensor_t> const& value_caches,
                 const aiter_tensor_t& block_mapping)
{
    int num_layers = key_caches.size();
    AITER_CHECK(num_layers == (int)value_caches.size());
    if(num_layers == 0)
    {
        return;
    }
    AITER_CHECK(key_caches[0].is_gpu(), "cache must be on GPU");
    int cache_device_id = key_caches[0].device_id;

    int64_t key_cache_ptrs[num_layers];
    int64_t value_cache_ptrs[num_layers];
    for(int layer_idx = 0; layer_idx < num_layers; ++layer_idx)
    {
        key_cache_ptrs[layer_idx]   = reinterpret_cast<int64_t>(key_caches[layer_idx].data_ptr());
        value_cache_ptrs[layer_idx] = reinterpret_cast<int64_t>(value_caches[layer_idx].data_ptr());
    }

    int num_pairs = block_mapping.size(0);

    HipDeviceGuard device_guard(cache_device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    int64_t* d_key_ptrs;
    int64_t* d_value_ptrs;
    HIP_CALL(hipMalloc(&d_key_ptrs, num_layers * sizeof(int64_t)));
    HIP_CALL(hipMalloc(&d_value_ptrs, num_layers * sizeof(int64_t)));
    HIP_CALL(hipMemcpyAsync(d_key_ptrs, key_cache_ptrs,
        num_layers * sizeof(int64_t), hipMemcpyHostToDevice, stream));
    HIP_CALL(hipMemcpyAsync(d_value_ptrs, value_cache_ptrs,
        num_layers * sizeof(int64_t), hipMemcpyHostToDevice, stream));

    const int numel_per_block = key_caches[0].numel() / key_caches[0].size(0);
    dim3 grid(num_layers, num_pairs);
    dim3 block(std::min(1024, numel_per_block));
    VLLM_DISPATCH_FLOATING_AND_BYTE_TYPES_rmTorch(key_caches[0].dtype(), "copy_blocks_kernel", ([&] {
                                              aiter::copy_blocks_kernel<scalar_t>
                                                  <<<grid, block, 0, stream>>>(
                                                      d_key_ptrs,
                                                      d_value_ptrs,
                                                      reinterpret_cast<int64_t*>(block_mapping.data_ptr()),
                                                      numel_per_block);
                                          }));
    HIP_CALL(hipStreamSynchronize(stream));
    HIP_CALL(hipFree(d_key_ptrs));
    HIP_CALL(hipFree(d_value_ptrs));
}

} // namespace aiter

namespace aiter {

template <typename scalar_t,
          typename cache_t,
          vllm::Fp8KVCacheDataType kv_dt,
          bool asmLayout = false>
__global__ void
reshape_and_cache_kernel(const scalar_t* __restrict__ key,   // [num_tokens, num_heads, head_size]
                         const scalar_t* __restrict__ value, // [num_tokens, num_heads, head_size]
                         cache_t* __restrict__ key_cache,    // [num_blocks, num_heads, head_size/x,
                                                             // block_size, x]
                         cache_t* __restrict__ value_cache,  // [num_blocks, num_heads, head_size,
                                                             // block_size]
                         const void* __restrict__ slot_mapping, // [num_tokens], int32 or int64
                         const SlotIndexWidth slot_width,
                         const int key_stride,
                         const int value_stride,
                         const int num_heads,
                         const int head_size,
                         const int block_size,
                         const int x,
                         const float* k_scale,
                         const float* v_scale)
{
    const int64_t token_idx = blockIdx.x;
    const int64_t slot_idx  = load_slot_index(slot_mapping, token_idx, slot_width);
    if(slot_idx < 0)
    {
        // Padding token that should be ignored.
        return;
    }

    const int64_t block_idx    = slot_idx / block_size;
    const int64_t block_offset = slot_idx % block_size;

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
            key_cache[tgt_key_idx] = opus::cast<cache_t>(
                static_cast<float>(tgt_key) * inverted_kscale);
            value_cache[tgt_value_idx] = opus::cast<cache_t>(
                static_cast<float>(tgt_value) * inverted_vscale);
        }
    }
}

template <typename scalar_t, typename cache_t, vllm::Fp8KVCacheDataType kv_dt>
__global__ void reshape_and_cache_flash_kernel(
    const scalar_t* __restrict__ key,         // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,       // [num_tokens, num_heads, head_size]
    cache_t* __restrict__ key_cache,          // [num_blocks, block_size, num_heads,
                                              // head_size]
    cache_t* __restrict__ value_cache,     // [num_blocks, block_size, num_heads,
                                           // head_size]
    const void* __restrict__ slot_mapping, // [num_tokens], int32 or int64
    const SlotIndexWidth slot_width,
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
    const int64_t slot_idx  = load_slot_index(slot_mapping, token_idx, slot_width);
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
            key_cache[tgt_key_value_idx] = opus::cast<cache_t>(
                static_cast<float>(tgt_key) * inverted_kscale);
            value_cache[tgt_key_value_idx] = opus::cast<cache_t>(
                static_cast<float>(tgt_value) * inverted_vscale);
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
          int wg_size_    = -1>
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
    float dtypeMax              = static_cast<float>(opus::finfo<cache_t>::max());
    constexpr int wg_size = wg_size_ == -1 ? WARP_SIZE : wg_size_;
    const int32_t tokens_per_wg = wg_size / WARP_SIZE;

    // every wave compute one token, one head, all the headim
    int wave_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

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
        int current_d           = lane_id + i_d * WARP_SIZE;
        const int64_t src_k_idx = token_idx * key_stride + head_idx * head_size + current_d;
        const int64_t src_v_idx = token_idx * value_stride + head_idx * head_size + current_d;
        if(current_d < head_size)
        {
            k_local_dim[i_d] = static_cast<float>(key[src_k_idx]);
            v_local_dim[i_d] = static_cast<float>(value[src_v_idx]);
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

    constexpr float k_pertoken_quant_scale_eps = 1e-12f;
    float k_token_scale = k_max / dtypeMax;
    float v_token_scale = v_max / dtypeMax;
    float k_token_scale_inverted = 1.0f / fmaxf(k_token_scale, k_pertoken_quant_scale_eps);
    float v_token_scale_inverted = 1.0f / fmaxf(v_token_scale, k_pertoken_quant_scale_eps);

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
        int i_d = lane_id + i * WARP_SIZE;
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
        key_cache[tgt_key_idx]     = opus::cast<cache_t>(k_local_dim[i]);
        value_cache[tgt_value_idx] = opus::cast<cache_t>(v_local_dim[i]);
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
    float dtypeMax          = static_cast<float>(opus::finfo<cache_t>::max());
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

            k_max_val = f_absmax_f32(k_max_val, static_cast<float>(key[src_k_idx]));
            v_max_val = f_absmax_f32(v_max_val, static_cast<float>(value[src_v_idx]));
        }
    }

    k_max_val = block_reduce<float, decltype(f_max_f32), wg_size, true>(k_max_val, f_max_f32);
    // block_reduce's only barrier sits after its cross-wave smem write, so
    // two rounds of one instantiation must be separated by the caller.
    __syncthreads();
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
                    float tmp            = static_cast<float>(key_cache[cache_idx]);
                    tmp                  = tmp * k_block_scale_global / k_block_scale;
                    key_cache[cache_idx] = opus::cast<cache_t>(tmp);
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
                    float tmp              = static_cast<float>(value_cache[cache_idx]);
                    tmp                    = tmp * v_block_scale_global / v_block_scale;
                    value_cache[cache_idx] = opus::cast<cache_t>(tmp);
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
            float tmp_k             = static_cast<float>(key[src_k_idx]) * k_block_scale;
            float tmp_v = static_cast<float>(value[src_v_idx]) * v_block_scale;

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
            key_cache[tgt_key_idx]     = opus::cast<cache_t>(tmp_k);
            value_cache[tgt_value_idx] = opus::cast<cache_t>(tmp_v);
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
    float dtypeMax          = static_cast<float>(opus::finfo<cache_t>::max());
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

            k_max_val = f_absmax_f32(k_max_val, static_cast<float>(key[src_k_idx]));
            v_max_val = f_absmax_f32(v_max_val, static_cast<float>(value[src_v_idx]));
        }
    }

    k_max_val = block_reduce<float, decltype(f_max_f32), wg_size, true>(k_max_val, f_max_f32);
    // block_reduce's only barrier sits after its cross-wave smem write, so
    // two rounds of one instantiation must be separated by the caller.
    __syncthreads();
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
                    float tmp            = static_cast<float>(key_cache[cache_idx]);
                    tmp                  = tmp * k_block_scale_global / k_block_scale;
                    key_cache[cache_idx] = opus::cast<cache_t>(tmp);
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
                    float tmp              = static_cast<float>(value_cache[cache_idx]);
                    tmp                    = tmp * v_block_scale_global / v_block_scale;
                    value_cache[cache_idx] = opus::cast<cache_t>(tmp);
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
            float tmp_k             = static_cast<float>(key[src_k_idx]) * k_block_scale;
            float tmp_v = static_cast<float>(value[src_v_idx]) * v_block_scale;

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
            key_cache[tgt_key_idx]     = opus::cast<cache_t>(tmp_k);
            value_cache[tgt_value_idx] = opus::cast<cache_t>(tmp_v);
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
                dst[dst_idx] = opus::cast<cache_t>(
                    static_cast<float>(src[src_idx]) * inverted_kscale);
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
    using vec_i                         = opus::vector_t<scalar_t, vec_size_i>;
    static constexpr int32_t ooba_i     = 4 / sizeof(scalar_t);
    static constexpr int32_t ooba_o     = 4 / sizeof(cache_t);
    auto out_offset                     = block_idx * block_stride + block_offset * entry_stride;

    const int32_t oob_i = (kv_lora_rank + ooba_i - 1) / ooba_i * ooba_i;
    auto const* ptr_i   = reinterpret_cast<scalar_t const*>(kv_c + token_idx * kv_c_stride);
    // auto buffer_i =
    //     ck_tile::make_buffer_view<ck_tile::address_space_enum::global>(ptr_i, oob_i);
    // buffer_i.init_raw();
    auto buffer_i = opus::make_gmem<scalar_t>(ptr_i, oob_i * sizeof(scalar_t));

    const int32_t pe_oob_i = (pe_dim + ooba_i - 1) / ooba_i * ooba_i;
    auto const* pe_ptr_i   = reinterpret_cast<scalar_t const*>(k_pe + token_idx * k_pe_stride);
    // auto pe_buffer_i =
    //     ck_tile::make_buffer_view<ck_tile::address_space_enum::global>(pe_ptr_i, pe_oob_i);
    // pe_buffer_i.init_raw();
    auto pe_buffer_i = opus::make_gmem<scalar_t>(pe_ptr_i, pe_oob_i * sizeof(scalar_t));
    const int32_t pe_num_vecs = (pe_dim + vec_size_i - 1) / vec_size_i;
    vec_i pe_vec_nxt;
    vec_i pe_vec_cur;
    size_t vec_idx    = threadIdx.x;
    size_t vec_stride = blockDim.x;
    // double load core loop start
    const int32_t num_vecs = (kv_lora_rank + vec_size_i - 1) / vec_size_i;
    vec_i vec_nxt;
    vec_i vec_cur;
    // vec_cur = buffer_i.template get<vec_i>(vec_idx * vec_size_i, 0, true);
    vec_cur = buffer_i.template load<vec_size_i>(vec_idx * vec_size_i);
    if(vec_idx < pe_num_vecs)
    {
        // pe_vec_cur = pe_buffer_i.template get<vec_i>(vec_idx * vec_size_i, 0, true);
        pe_vec_cur = pe_buffer_i.template load<vec_size_i>(vec_idx * vec_size_i);
    }
    const int32_t oob_o = (kv_lora_rank + ooba_o - 1) / ooba_o * ooba_o;
    auto* ptr_o         = reinterpret_cast<cache_t*>(kv_cache + out_offset);
    // auto buffer_o =
    //     ck_tile::make_buffer_view<ck_tile::address_space_enum::global>(ptr_o, oob_o);
    // buffer_o.init_raw();
    auto buffer_o = opus::make_gmem<cache_t>(ptr_o, oob_o * sizeof(cache_t));
    const int32_t pe_oob_o = (pe_dim + ooba_o - 1) / ooba_o * ooba_o;
    auto* pe_ptr_o         = reinterpret_cast<cache_t*>(kv_cache + out_offset + kv_lora_rank);
    // auto pe_buffer_o =
    //     ck_tile::make_buffer_view<ck_tile::address_space_enum::global>(pe_ptr_o, pe_oob_o);
    // pe_buffer_o.init_raw();
    auto pe_buffer_o = opus::make_gmem<cache_t>(pe_ptr_o, pe_oob_o * sizeof(cache_t));
    for(vec_idx += vec_stride; vec_idx < num_vecs; vec_idx += vec_stride)
    {
        // vec_nxt = buffer_i.template get<vec_i>(vec_idx * vec_size_i, 0, true);
        vec_nxt = buffer_i.template load<vec_size_i>(vec_idx * vec_size_i);
        if constexpr(kv_dt == vllm::Fp8KVCacheDataType::kAuto)
        {
            store_vector<cache_t, scalar_t, vec_size_i, RT, false, WARP_SIZE, 1, cache_t>(buffer_o, vec_cur, (vec_idx - vec_stride) * vec_size_o);
        }
        else
        {
            store_vector<cache_t, scalar_t, vec_size_i, RT, false, WARP_SIZE, 1, cache_t>(buffer_o, vec_cur, (vec_idx - vec_stride) * vec_size_o, inverted_kscale);
        }
        vec_cur = vec_nxt;
    }
    if(threadIdx.x < pe_num_vecs) {
      if constexpr(kv_dt == vllm::Fp8KVCacheDataType::kAuto)
      {
          store_vector<cache_t, scalar_t, vec_size_i, RT, false, WARP_SIZE, 1, cache_t>(pe_buffer_o, pe_vec_cur, threadIdx.x * vec_size_o);
      }
      else
      {
          store_vector<cache_t, scalar_t, vec_size_i, RT, false, WARP_SIZE, 1, cache_t>(pe_buffer_o, pe_vec_cur, threadIdx.x * vec_size_o, inverted_kscale);
      }
    }
    if(vec_idx - vec_stride < num_vecs)
    {
        if constexpr(kv_dt == vllm::Fp8KVCacheDataType::kAuto)
        {
            store_vector<cache_t, scalar_t, vec_size_i, RT, false, WARP_SIZE, 1, cache_t>(buffer_o, vec_cur, (vec_idx - vec_stride) * vec_size_o);
        }
        else
        {
            store_vector<cache_t, scalar_t, vec_size_i, RT, false, WARP_SIZE, 1, cache_t>(buffer_o, vec_cur, (vec_idx - vec_stride) * vec_size_o, inverted_kscale);
        }
    }

}

// ============================================================================
// Segmented paged KV cache write (no RoPE): concat kv_c (nope) + k_pe into a
// flat block layout that matches fused_qk_rope_concat_and_cache_mla_seg:
//   block: [page_size x kv_lora (nope)][page_size x pe], token-major.
//     nope: block_idx*block_stride + block_offset*kv_lora_rank + i
//     pe:   block_idx*block_stride + page_size*kv_lora_rank + block_offset*pe_dim + i
// ============================================================================
template <typename scalar_t, typename cache_t, vllm::Fp8KVCacheDataType kv_dt>
__global__ void concat_and_cache_mla_seg_kernel(
    const scalar_t* __restrict__ kv_c,        // [num_tokens, kv_lora_rank]
    const scalar_t* __restrict__ k_pe,        // [num_tokens, pe_dim]
    cache_t* __restrict__ kv_cache,           // [num_blocks, block_stride] flat
    const int64_t* __restrict__ slot_mapping, // [num_tokens]
    const int block_stride,                   //
    const int kv_c_stride,                    //
    const int k_pe_stride,                    //
    const int kv_lora_rank,                   //
    const int pe_dim,                         //
    const int page_size,                      //
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
    const int64_t block_idx     = slot_idx / page_size;
    const int64_t block_offset  = slot_idx % page_size;
    const float inverted_kscale = 1.0f / *scale;
    const int64_t nope_base     = block_idx * block_stride + block_offset * kv_lora_rank;
    const int64_t pe_base =
        block_idx * block_stride + (int64_t)page_size * kv_lora_rank + block_offset * pe_dim;

    auto copy = [&](const scalar_t* __restrict__ src,
                    int src_stride,
                    int size,
                    int64_t dst_base) {
        for(int i = threadIdx.x; i < size; i += blockDim.x)
        {
            const scalar_t v = src[token_idx * src_stride + i];
            if constexpr(kv_dt == vllm::Fp8KVCacheDataType::kAuto)
            {
                kv_cache[dst_base + i] = v;
            }
            else
            {
                kv_cache[dst_base + i] =
                    opus::cast<cache_t>(static_cast<float>(v) * inverted_kscale);
            }
        }
    };
    copy(kv_c, kv_c_stride, kv_lora_rank, nope_base);
    copy(k_pe, k_pe_stride, pe_dim, pe_base);
}

// Vectorized variant (kv_lora_rank & pe_dim divisible by VEC): 128-bit
// loads/stores for 16-bit inputs. Plain global vector access, portable.
template <typename scalar_t, typename cache_t, vllm::Fp8KVCacheDataType kv_dt, int VEC>
__global__ void concat_and_cache_mla_seg_opt_kernel(
    const scalar_t* __restrict__ kv_c,        // [num_tokens, kv_lora_rank]
    const scalar_t* __restrict__ k_pe,        // [num_tokens, pe_dim]
    cache_t* __restrict__ kv_cache,           // [num_blocks, block_stride] flat
    const int64_t* __restrict__ slot_mapping, // [num_tokens]
    const int block_stride,                   //
    const int kv_c_stride,                    //
    const int k_pe_stride,                    //
    const int kv_lora_rank,                   //
    const int pe_dim,                         //
    const int page_size,                      //
    const float* scale                        //
)
{
    using in_vec_t  = opus::vector_t<scalar_t, VEC>;
    using out_vec_t = opus::vector_t<cache_t, VEC>;

    const int64_t token_idx = blockIdx.x;
    const int64_t slot_idx  = slot_mapping[token_idx];
    if(slot_idx < 0)
    {
        return;
    }
    const int64_t block_idx     = slot_idx / page_size;
    const int64_t block_offset  = slot_idx % page_size;
    const float inverted_kscale = 1.0f / *scale;
    const int64_t nope_base     = block_idx * block_stride + block_offset * kv_lora_rank;
    const int64_t pe_base =
        block_idx * block_stride + (int64_t)page_size * kv_lora_rank + block_offset * pe_dim;

    auto copy = [&](const scalar_t* __restrict__ src, int src_stride, int size, int64_t dst_base) {
        const int num_vec        = size / VEC;
        const in_vec_t* src_v    = reinterpret_cast<const in_vec_t*>(src + token_idx * src_stride);
        out_vec_t*      dst_v    = reinterpret_cast<out_vec_t*>(kv_cache + dst_base);
        for(int v = threadIdx.x; v < num_vec; v += blockDim.x)
        {
            const in_vec_t vin = src_v[v];
            if constexpr(kv_dt == vllm::Fp8KVCacheDataType::kAuto)
            {
                out_vec_t vout;
                for(int j = 0; j < VEC; ++j)
                    vout[j] = static_cast<cache_t>(vin[j]);
                dst_v[v] = vout;
            }
            else
            {
                dst_v[v] = aiter::scaled_cast<cache_t>(vin, inverted_kscale);
            }
        }
    };
    copy(kv_c, kv_c_stride, kv_lora_rank, nope_base);
    copy(k_pe, k_pe_stride, pe_dim, pe_base);
}

// The layout fused_qk_rope_concat_and_cache_mla_seg writes (kv_lora 512, pe 64,
// page 64), TPB tokens per block. Same data movement as the kernel above; what
// changes is when the loads go out.
//
// There, a block is one token, so a lane has a single 16 B load in flight, and
// that load waits behind slot_mapping[token] and a 64-bit divide by the
// runtime page_size -- neither of which the source address depends on. Here
// every source load issues first, from token_idx alone, and the slot lookup
// (now a shift, the page being a compile-time 64) overlaps it. A block takes
// TPB tokens so a lane carries TPB independent nope loads; each token's pe is
// spread over its own PE_CHUNKS lanes rather than stacked on lanes
// 0..PE_CHUNKS-1, so a lane holds one token's pe however large TPB is.
template <typename scalar_t, typename cache_t, vllm::Fp8KVCacheDataType kv_dt, int TPB>
__global__ void concat_and_cache_mla_seg_512_kernel(
    const scalar_t* __restrict__ kv_c,        // [num_tokens, 512]
    const scalar_t* __restrict__ k_pe,        // [num_tokens, 64]
    cache_t* __restrict__ kv_cache,           // [num_blocks, block_stride] flat
    const int64_t* __restrict__ slot_mapping, // [num_tokens]
    const int num_tokens,
    const int64_t block_stride,
    const int64_t kv_c_stride,
    const int64_t k_pe_stride,
    const float* scale)
{
    constexpr int KV_LORA   = 512;
    constexpr int PE_DIM    = 64;
    constexpr int PAGE_SIZE = 64;
    constexpr int VEC       = 8;
    constexpr int NUM_VEC   = KV_LORA / VEC; // == blockDim.x
    constexpr int PE_CHUNKS = PE_DIM / VEC;
    constexpr int PE_TOKENS_PER_ROUND = NUM_VEC / PE_CHUNKS;
    static_assert(TPB <= PE_TOKENS_PER_ROUND, "a token's pe needs its own PE_CHUNKS lanes");
    using in_vec_t  = opus::vector_t<scalar_t, VEC>;
    using out_vec_t = opus::vector_t<cache_t, VEC>;

    const int64_t tok0          = static_cast<int64_t>(blockIdx.x) * TPB;
    const float inverted_kscale = 1.0f / *scale;

    auto convert = [&](const in_vec_t& vin) -> out_vec_t {
        if constexpr(kv_dt == vllm::Fp8KVCacheDataType::kAuto)
        {
            out_vec_t vout;
#pragma unroll
            for(int j = 0; j < VEC; ++j)
                vout[j] = static_cast<cache_t>(vin[j]);
            return vout;
        }
        else
        {
            return aiter::scaled_cast<cache_t>(vin, inverted_kscale);
        }
    };

    // ---- issue: every source load, then the slots ----
    in_vec_t nope_in[TPB];
#pragma unroll
    for(int k = 0; k < TPB; ++k)
    {
        const int64_t t = tok0 + k;
        if(t < num_tokens)
            nope_in[k] = reinterpret_cast<const in_vec_t*>(kv_c + t * kv_c_stride)[threadIdx.x];
    }
    const int pe_tok   = threadIdx.x / PE_CHUNKS;
    const int pe_chunk = threadIdx.x % PE_CHUNKS;
    const int64_t pe_t = tok0 + pe_tok;
    const bool pe_lane = pe_tok < TPB && pe_t < num_tokens;
    in_vec_t pe_in{};
    int64_t pe_slot = -1;
    if(pe_lane)
    {
        pe_in   = reinterpret_cast<const in_vec_t*>(k_pe + pe_t * k_pe_stride)[pe_chunk];
        pe_slot = slot_mapping[pe_t];
    }

    int64_t slot[TPB];
#pragma unroll
    for(int k = 0; k < TPB; ++k)
    {
        const int64_t t = tok0 + k;
        slot[k]         = t < num_tokens ? slot_mapping[t] : -1;
    }

    // ---- emit ----
#pragma unroll
    for(int k = 0; k < TPB; ++k)
    {
        // Uniform over the block: slot and t come from blockIdx and k.
        if(slot[k] < 0)
            continue;
        const int64_t base =
            (slot[k] / PAGE_SIZE) * block_stride + (slot[k] % PAGE_SIZE) * KV_LORA;
        reinterpret_cast<out_vec_t*>(kv_cache + base)[threadIdx.x] = convert(nope_in[k]);
    }
    if(pe_slot >= 0)
    {
        const int64_t base = (pe_slot / PAGE_SIZE) * block_stride +
                             static_cast<int64_t>(PAGE_SIZE) * KV_LORA +
                             (pe_slot % PAGE_SIZE) * PE_DIM;
        reinterpret_cast<out_vec_t*>(kv_cache + base)[pe_chunk] = convert(pe_in);
    }
}

template <typename scalar_t,
          typename cache_t,
          vllm::Fp8KVCacheDataType kv_dt,
          int BLOCK_X_SIZE,
          int BLOCK_Y_SIZE,
          int VEC_SIZE>
__global__ void indexer_k_quant_and_cache_kernel(
    const scalar_t* __restrict__ k,           // [num_tokens, head_dim]
    cache_t* __restrict__ kv_cache,           // [num_blocks, block_size, cache_stride]
    const int64_t* __restrict__ slot_mapping, // [num_tokens]
    const int num_tokens,
    const int head_dim,         // dimension of each head
    const int quant_block_size, // quantization block size
    const int cache_block_size, // cache block size
    const int cache_stride,     // stride for each token in kv_cache
    const bool use_ue8m0,       // use ue8m0 scale format
    const bool preshuffle       // use MFMA 16x16 preshuffled layout
)
{
    const int quant_block_per_head = head_dim / quant_block_size;
    const int64_t token_idx = (blockIdx.x * BLOCK_Y_SIZE + threadIdx.y) / quant_block_per_head;
    if(token_idx >= num_tokens)
        return;
    const int64_t slot_idx = slot_mapping[token_idx];
    const int head_dim_idx =
        (blockIdx.x * BLOCK_Y_SIZE + threadIdx.y) % quant_block_per_head * quant_block_size +
        threadIdx.x * VEC_SIZE;
    const int64_t block_idx    = slot_idx / cache_block_size;
    const int64_t block_offset = slot_idx % cache_block_size;
    using vec_i                = opus::vector_t<scalar_t, VEC_SIZE>;
    using vec_o                = opus::vector_t<cache_t, VEC_SIZE>;

    // NOTE: slot_idx can be -1 if the token is padded
    if(slot_idx < 0 || (head_dim_idx >= head_dim))
    {
        return;
    }

    vec_i k_val =
        (reinterpret_cast<const vec_i*>(k))[(token_idx * head_dim + head_dim_idx) / VEC_SIZE];
    float amax = 0.0f;
    if constexpr(VEC_SIZE % 2 == 0)
    {
        for(int i = 0; i < VEC_SIZE; i += 2)
        {
            asm volatile("v_max3_f32 %0, %1, %2, %3\n"
                         : "=v"(amax)
                         : "v"(amax),
                           "v"(fabsf(static_cast<float>(k_val[i]))),
                           "v"(fabsf(static_cast<float>(k_val[i + 1]))));
        }
    }
    else
    {
        for(int i = 0; i < VEC_SIZE; i++)
        {
            amax = fmaxf(amax, fabsf(static_cast<float>(k_val[i])));
        }
    }

    // Reduced amax
    amax = multithread_reduce(amax, fmaxf, BLOCK_X_SIZE);

    float scale =
        fmaxf(amax, 1e-4) / static_cast<float>(opus::finfo<cache_t>::max());
    if(use_ue8m0)
    {
        scale = exp2f(ceilf(log2f(scale)));
    }

    int64_t dst_offset;
    if(preshuffle)
    {
        // Preshuffled layout for MFMA 16x16 tile.
        // Works for any cache_block_size and head_dim that are multiples of 16.
        // A paged block is split into (cache_block_size / 16) token groups; each group
        // contains (head_dim / 16) contiguous 16x16 tiles laid out row-major within tile.
        constexpr int TILE       = 16;
        const int token_tile_id  = block_offset / TILE;
        const int token_in_tile  = block_offset % TILE;
        const int col_tile_id    = head_dim_idx / TILE;
        const int col_in_tile    = head_dim_idx % TILE;
        dst_offset = block_idx * cache_block_size * cache_stride
                   + token_tile_id * (TILE * head_dim)
                   + col_tile_id   * (TILE * TILE)
                   + token_in_tile * TILE
                   + col_in_tile;
    }
    else
    {
        dst_offset =
            block_idx * cache_block_size * cache_stride + block_offset * head_dim + head_dim_idx;
    }

    if(threadIdx.x == 0)
    {
        // Scale layout is unchanged regardless of preshuffle
        const int64_t dst_scale_idx =
            block_idx * cache_block_size * cache_stride + cache_block_size * head_dim +
            (block_offset * head_dim + head_dim_idx) * 4 / quant_block_size;
        reinterpret_cast<float*>(kv_cache)[dst_scale_idx / 4] = scale;
    }
    scale               = 1.0f / scale;
    vec_o* kv_cache_vec = reinterpret_cast<vec_o*>(kv_cache + dst_offset);
    *kv_cache_vec       = aiter::scaled_cast<cache_t>(k_val, scale);
}

// FP4 (e2m1 + e8m0) output layout for the FlyDSL paged MQA-logits indexer kernels,
// defined in aiter/ops/flydsl/kernels/mqa_logits/pa_mqa_logits_fp4.py. The same cache
// is written from the DSv4 path by csrc/kernels/dsv4_rotate_quant.cu, so the offsets
// below must track its kv_fp4_preshuffle_offset / kv_scale_preshuffle_offset helpers.
constexpr int INDEXER_FP4_GROUP_SIZE = 32;

// ceil(log2(amax / 6)) + 127, the e8m0 exponent that maps amax into e2m1 range.
__device__ __forceinline__ uint8_t indexer_fp4_scale_e8m0(const float amax)
{
    constexpr float fp4_max     = static_cast<float>(opus::finfo<opus::fp4_t>::max());
    constexpr float inv_fp4_max = 1.0f / fp4_max;
    constexpr float eps_amax    = fp4_max * __builtin_bit_cast(float, 0x00800000u);

    const uint32_t bits = __builtin_bit_cast(uint32_t, fmaxf(amax, eps_amax) * inv_fp4_max);
    uint8_t exponent    = (bits >> 23) & 0xFF;
    if(exponent == 0xFF)
    {
        return exponent;
    }
    if(bits & 0x7FFFFF)
    {
        exponent += 1;
    }
    return exponent;
}

__device__ __forceinline__ float indexer_fp4_scale_from_e8m0(const uint8_t scale_e8m0)
{
    return __builtin_bit_cast(float, static_cast<uint32_t>(scale_e8m0) << 23);
}

// Per page: dense [kv_block_size, k_tiles, 4, 16] permuted to [k_tiles, 4, kv_block_size, 16].
__device__ __forceinline__ int64_t indexer_fp4_kv_data_offset(const int64_t block_idx,
                                                              const int pos_in_block,
                                                              const int packed_byte_idx,
                                                              const int k_tiles,
                                                              const int kv_block_size)
{
    constexpr int bytes_per_k_tile = 4 * 16;
    const int k_tile               = packed_byte_idx / bytes_per_k_tile;
    const int rem                  = packed_byte_idx % bytes_per_k_tile;
    const int group4               = rem / 16;
    const int sub16                = rem % 16;
    return ((block_idx * k_tiles + k_tile) * 4 + group4) *
               static_cast<int64_t>(kv_block_size) * 16 +
           static_cast<int64_t>(pos_in_block) * 16 + sub16;
}

// Token axis interleaved so the reader's packed-dword load covers one token's N-tiles.
__device__ __forceinline__ int64_t indexer_fp4_kv_scale_offset(const int64_t block_idx,
                                                               const int pos_in_block,
                                                               const int scale_group_idx,
                                                               const int k_tiles,
                                                               const int kv_block_size)
{
    const int k_tile          = scale_group_idx / 4;
    const int group4          = scale_group_idx % 4;
    const int tiles_per_block = kv_block_size / 16;
    const int sflat = (pos_in_block % 16) * tiles_per_block + (pos_in_block / 16);
    return ((block_idx * k_tiles + k_tile) * 4 + group4) *
               static_cast<int64_t>(kv_block_size) +
           sflat;
}

// Heads split into (m_tile, m_inner = 16) with m_tile innermost and padded to a
// multiple of 4, so the reader can dword-load four m_tiles at once.
__device__ __forceinline__ void indexer_fp4_store_q_scale(uint8_t* __restrict__ scale,
                                                          const uint8_t scale_e8m0,
                                                          const int64_t token_idx,
                                                          const int head_idx,
                                                          const int n_heads,
                                                          const int group_idx,
                                                          const int groups_per_row)
{
    const int m_tiles        = n_heads >> 4;
    const int m_tiles_padded = (m_tiles + 3) & ~3;
    const int m_tile         = head_idx >> 4;
    const int m_inner        = head_idx & 15;

    const int64_t tile_base =
        ((token_idx * groups_per_row + group_idx) * 16 + m_inner) * m_tiles_padded;
    scale[tile_base + m_tile] = scale_e8m0;

    if(m_tiles_padded != m_tiles && m_tile == m_tiles - 1)
    {
        for(int pad_tile = m_tiles; pad_tile < m_tiles_padded; ++pad_tile)
        {
            scale[tile_base + pad_tile] = 0;
        }
    }
}

__device__ __forceinline__ uint8_t indexer_fp4_pack_pair(const float val,
                                                         const float partner,
                                                         const float scale)
{
    const opus::fp32x2_t pair = {val, partner};
    return __builtin_bit_cast(uint8_t, scaled_cast<opus::fp4_t>(pair, scale));
}

template <bool FP4_OUT, typename cache_t>
using indexer_qk_out_t = std::conditional_t<FP4_OUT, uint8_t, cache_t>;

template <bool FP4_OUT, typename scalar_t>
using indexer_weights_out_t = std::conditional_t<FP4_OUT, scalar_t, float>;

// One wave per block: the Q phase gives each lane VEC elements of a q row, the K
// phase spans the whole wave. The dispatch also instantiates 4-byte scalars, so
// per-lane element counts have to follow sizeof(scalar_t).
constexpr int INDEXER_VEC_BYTES_WIDE = 16;
// Measured crossover of the two instantiations.
constexpr int INDEXER_NARROW_MAX_TOKENS = 256;

// WARP_SIZE is a device-pass constant; a constant-evaluated read of it always
// answers 64, whatever the target. It must therefore never reach a template
// argument or a const host variable, or a wave32 build registers a host stub
// under a kernel name its device binary does not define. The kernel is templated
// on NARROW alone and both passes recompute the shape from the wave they see.
__host__ __device__ constexpr int
indexer_vec_elems(bool narrow, int threads, int head_dim, int elem_bytes)
{
    return narrow ? head_dim / threads : INDEXER_VEC_BYTES_WIDE / elem_bytes;
}

__host__ __device__ constexpr int
indexer_heads_per_block(int threads, int head_dim, int vec_elems)
{
    return threads * vec_elems / head_dim;
}

// Folds the per-element wave results in element order, so the caller's
// dim-to-lane mapping fixes the summation order. The K layernorm must keep the
// strided split dim = lane + e*THREADS, which reproduces the 128-thread kernel's
// wave-0 / wave-1 pairing; a contiguous split changes FP4 output.
template <typename T, typename F, int THREADS, int ELEMS>
__device__ __forceinline__ T indexer_lane_reduce(const T (&v)[ELEMS], F reduce_op)
{
    T acc = wave_reduce<T, F, THREADS, true>(v[0], reduce_op);
#pragma unroll
    for(int e = 1; e < ELEMS; ++e)
    {
        acc = reduce_op(acc, wave_reduce<T, F, THREADS, true>(v[e], reduce_op));
    }
    return acc;
}

template <typename T, int N>
struct alignas(sizeof(T) * N) indexer_vec
{
    T v[N];
};

template <typename scalar_t,
          typename cache_t,
          vllm::Fp8KVCacheDataType kv_dt,
          int HEAD_DIM,
          int ROPE_DIM,
          bool FP4_OUT,
          bool NARROW>
__global__ void indexer_qk_rope_quant_and_cache_kernel(
    const scalar_t* __restrict__ q,           // [num_tokens, n_heads, head_dim]
    indexer_qk_out_t<FP4_OUT, cache_t>* __restrict__ q_out,
    const scalar_t* __restrict__ weights,     // [num_tokens, n_heads]
    indexer_weights_out_t<FP4_OUT, scalar_t>* __restrict__ weights_out,
    const scalar_t* __restrict__ k,           // [num_tokens, head_dim]
    indexer_qk_out_t<FP4_OUT, cache_t>* __restrict__ kv_cache,
    const int64_t* __restrict__ slot_mapping, // [num_tokens]
    // fp32 to match caller's LN params (bf16 cast drops precision).
    const float* __restrict__ norm_weight,    // [head_dim]
    const float* __restrict__ norm_bias,      // [head_dim]
    const int64_t* __restrict__ positions,    // [num_tokens]
    const scalar_t* __restrict__ cos_cache,   // [max_position, ..., rope_dim / 2]
    const scalar_t* __restrict__ sin_cache,   // [max_position, ..., rope_dim / 2]
    uint8_t* __restrict__ q_scale_out,        // FP4 only
    uint8_t* __restrict__ kv_cache_scale,     // FP4 only
    const int num_tokens,
    const int n_heads,
    const int quant_block_size,
    const int cache_block_size,
    const int cache_stride,
    const int64_t q_stride_t,
    const int64_t q_stride_h,
    const int64_t q_stride_d,
    const int64_t q_out_stride_t,
    const int64_t q_out_stride_h,
    const int64_t q_out_stride_d,
    const int64_t weights_stride_t,
    const int64_t weights_stride_h,
    const int64_t weights_out_stride_t,
    const int64_t weights_out_stride_h,
    const int64_t k_stride_t,
    const int64_t k_stride_d,
    const int64_t cos_stride0,
    const int64_t sin_stride0,
    const float epsilon,
    const float weights_scale,
    const bool use_ue8m0,
    const bool preshuffle,
    const bool is_neox,
    const int max_position,
    const bool compute_all_q_rope)
{
    static_assert(HEAD_DIM == 128, "Indexer fused qk cache currently supports head_dim=128");
    static_assert(ROPE_DIM == 64, "Indexer fused qk cache currently supports rope_dim=64");
    constexpr int THREADS = WARP_SIZE;
    static_assert(HEAD_DIM % THREADS == 0,
                  "every reduction here is wave-level, so a block is exactly one wave");
    static_assert((ROPE_DIM / 2) % THREADS == 0 || THREADS % (ROPE_DIM / 2) == 0,
                  "K splits as dim = lane + e * THREADS, so dim ^ (ROPE_DIM / 2) has to fall "
                  "wholly on the lane axis or wholly on the element axis");
    constexpr int ELEMS = HEAD_DIM / THREADS;

    constexpr int VEC =
        indexer_vec_elems(NARROW, THREADS, HEAD_DIM, static_cast<int>(sizeof(scalar_t)));
    constexpr int LANES_PER_HEAD = HEAD_DIM / VEC;
    // Only the narrow instantiation keeps shared memory: its 1-64 token band is
    // latency-bound, where a lane exchange costs more than a shared round trip.
    constexpr bool SHARED_ROPE    = NARROW;
    constexpr int HEADS_PER_BLOCK = indexer_heads_per_block(THREADS, HEAD_DIM, VEC);
    static_assert(HEAD_DIM % VEC == 0 && THREADS % LANES_PER_HEAD == 0,
                  "the wave must split evenly into whole head rows");
    static_assert(HEADS_PER_BLOCK >= 1, "a wave must cover at least one head row");
    static_assert(ROPE_DIM % VEC == 0, "a lane must sit wholly on one side of ROPE_DIM");
    static_assert(!SHARED_ROPE || HEADS_PER_BLOCK == 1,
                  "the shared path barriers under active_q, block-uniform only at one head per block");
    static_assert(INDEXER_FP4_GROUP_SIZE % VEC == 0,
                  "an FP4 group must be a whole number of lanes");

    const int64_t token_idx = blockIdx.x;
    const int lane          = threadIdx.x;
    const int head_slot     = lane / LANES_PER_HEAD;
    const int vec_id        = lane % LANES_PER_HEAD;
    const int dim0          = vec_id * VEC;
    const int head_idx      = blockIdx.y * HEADS_PER_BLOCK + head_slot;
    if(token_idx >= num_tokens)
        return;

    const int64_t slot_idx = slot_mapping[token_idx];
    if(!compute_all_q_rope && slot_idx < 0)
        return;
    int64_t pos = positions[token_idx];
    if(slot_idx < 0)
        pos = pos < 0 ? 0 : (pos >= max_position ? max_position - 1 : pos);
    const scalar_t* cos_ptr = cos_cache + pos * cos_stride0;
    const scalar_t* sin_ptr = sin_cache + pos * sin_stride0;

    auto max_func = [](float a, float b) { return fmaxf(a, b); };
    // Uniform across a head's lane group, so the group-wide reductions still see
    // every lane when a tail block overhangs n_heads.
    const bool active_q = head_idx < n_heads;
    float q_val[VEC];
    if(active_q)
    {
        const scalar_t* q_row = q + token_idx * q_stride_t + head_idx * q_stride_h;
        const auto qv = *reinterpret_cast<const indexer_vec<scalar_t, VEC>*>(q_row + dim0);
#pragma unroll
        for(int j = 0; j < VEC; ++j)
        {
            q_val[j] = static_cast<float>(qv.v[j]);
        }

        // Partners are gathered before any rotation, so the rotation runs in place.
        constexpr int ROPE_HALF_LANES = (ROPE_DIM / 2) / VEC;
        constexpr int HALF            = ROPE_DIM / 2;
        float pair[VEC];
        if constexpr(SHARED_ROPE)
        {
            __shared__ float q_sh[HEADS_PER_BLOCK][HEAD_DIM];
#pragma unroll
            for(int j = 0; j < VEC; ++j)
            {
                q_sh[head_slot][dim0 + j] = q_val[j];
            }
            __syncthreads();
#pragma unroll
            for(int j = 0; j < VEC; ++j)
            {
                pair[j] = q_sh[head_slot][(dim0 + j) ^ (is_neox ? HALF : 1)];
            }
        }
        else
        {
#pragma unroll
            for(int j = 0; j < VEC; ++j)
            {
                pair[j] = is_neox ? __shfl_xor(q_val[j], ROPE_HALF_LANES) : q_val[j ^ 1];
            }
        }

        if(dim0 < ROPE_DIM)
        {
#pragma unroll
            for(int j = 0; j < VEC; ++j)
            {
                const int dim     = dim0 + j;
                const bool lead   = is_neox ? (dim < HALF) : (dim % 2 == 0);
                const int cos_idx = is_neox ? (dim < HALF ? dim : dim - HALF) : dim / 2;
                const float cos_v = static_cast<float>(cos_ptr[cos_idx]);
                const float sin_v = static_cast<float>(sin_ptr[cos_idx]);
                q_val[j] = lead ? (q_val[j] * cos_v - pair[j] * sin_v)
                                : (q_val[j] * cos_v + pair[j] * sin_v);
                // Match the separate RoPE path, which materializes q_pe before FP8 quant.
                q_val[j] = static_cast<float>(static_cast<scalar_t>(q_val[j]));
            }
        }

        float lane_amax = fabsf(q_val[0]);
#pragma unroll
        for(int j = 1; j < VEC; ++j)
        {
            lane_amax = max_func(lane_amax, fabsf(q_val[j]));
        }

        if constexpr(FP4_OUT)
        {
            constexpr int LANES_PER_GROUP = INDEXER_FP4_GROUP_SIZE / VEC;
            static_assert(LANES_PER_HEAD % LANES_PER_GROUP == 0,
                          "an FP4 group must not span two head rows");
            const float q_amax = multithread_reduce<float, decltype(max_func)>(
                lane_amax, max_func, LANES_PER_GROUP);
            const uint8_t q_scale_e8m0 = indexer_fp4_scale_e8m0(q_amax);
            const float q_scale_f      = indexer_fp4_scale_from_e8m0(q_scale_e8m0);

            uint8_t* q_dst = q_out + token_idx * q_out_stride_t +
                             head_idx * q_out_stride_h + (dim0 / 2) * q_out_stride_d;
#pragma unroll
            for(int j = 0; j < VEC / 2; ++j)
            {
                q_dst[j * q_out_stride_d] =
                    indexer_fp4_pack_pair(q_val[2 * j], q_val[2 * j + 1], q_scale_f);
            }
            if(vec_id % LANES_PER_GROUP == 0)
            {
                indexer_fp4_store_q_scale(q_scale_out,
                                          q_scale_e8m0,
                                          token_idx,
                                          head_idx,
                                          n_heads,
                                          dim0 / INDEXER_FP4_GROUP_SIZE,
                                          HEAD_DIM / INDEXER_FP4_GROUP_SIZE);
            }
            if(vec_id == 0)
            {
                weights_out[token_idx * weights_out_stride_t +
                            head_idx * weights_out_stride_h] =
                    weights[token_idx * weights_stride_t + head_idx * weights_stride_h];
            }
        }
        else
        {
            const float q_amax = multithread_reduce<float, decltype(max_func)>(
                lane_amax, max_func, LANES_PER_HEAD);

            const float q_fp8_max     = static_cast<float>(opus::finfo<cache_t>::max());
            const float q_inv_fp8_max = 1.0f / q_fp8_max;
            float q_scale             = fmaxf(q_amax, 1e-10f) * q_inv_fp8_max;
            if(use_ue8m0)
            {
                q_scale = exp2f(ceilf(log2f(q_scale)));
            }
            const float q_inv_scale = 1.0f / q_scale;
            cache_t* q_dst = q_out + token_idx * q_out_stride_t +
                             head_idx * q_out_stride_h + dim0 * q_out_stride_d;
#pragma unroll
            for(int j = 0; j < VEC; ++j)
            {
                q_dst[j * q_out_stride_d] = opus::cast<cache_t>(q_val[j] * q_inv_scale);
            }
            if(vec_id == 0)
            {
                const float w = static_cast<float>(
                    weights[token_idx * weights_stride_t + head_idx * weights_stride_h]);
                weights_out[token_idx * weights_out_stride_t +
                            head_idx * weights_out_stride_h] =
                    w * q_scale * weights_scale;
            }
        }
    }

    if(blockIdx.y != 0 || slot_idx < 0)
        return;

    const scalar_t* k_row = k + token_idx * k_stride_t;

    float x[ELEMS];
#pragma unroll
    for(int e = 0; e < ELEMS; ++e)
    {
        x[e] = static_cast<float>(k_row[(lane + e * THREADS) * k_stride_d]);
    }
    auto sum_func = [](float a, float b) { return a + b; };
    float k_val[ELEMS];
    {
        // Leave contraction at the default; forcing it off here moves fp8
        // kv_cache away from the pre-rewrite kernel, not toward it.
        const float sum =
            indexer_lane_reduce<float, decltype(sum_func), THREADS, ELEMS>(x, sum_func);
        const float mean = sum / static_cast<float>(HEAD_DIM);

        float centered[ELEMS], sq[ELEMS];
#pragma unroll
        for(int e = 0; e < ELEMS; ++e)
        {
            centered[e] = x[e] - mean;
            sq[e]       = centered[e] * centered[e];
        }
        const float ss =
            indexer_lane_reduce<float, decltype(sum_func), THREADS, ELEMS>(sq, sum_func);
        const float inv_std = rsqrtf(ss / static_cast<float>(HEAD_DIM) + epsilon);

#pragma unroll
        for(int e = 0; e < ELEMS; ++e)
        {
            const int dim = lane + e * THREADS;
            k_val[e]      = centered[e] * inv_std * norm_weight[dim] + norm_bias[dim];
            k_val[e]      = static_cast<float>(static_cast<scalar_t>(k_val[e]));
        }
    }
    constexpr int K_HALF = ROPE_DIM / 2;
    float k_pair[ELEMS];
    if constexpr(SHARED_ROPE)
    {
        __shared__ float normed[HEAD_DIM];
#pragma unroll
        for(int e = 0; e < ELEMS; ++e)
        {
            normed[lane + e * THREADS] = k_val[e];
        }
        __syncthreads();
#pragma unroll
        for(int e = 0; e < ELEMS; ++e)
        {
            k_pair[e] = normed[(lane + e * THREADS) ^ (is_neox ? K_HALF : 1)];
        }
    }
    else if constexpr(K_HALF >= THREADS)
    {
#pragma unroll
        for(int e = 0; e < ELEMS; ++e)
        {
            k_pair[e] = is_neox ? k_val[e ^ (K_HALF / THREADS)] : __shfl_xor(k_val[e], 1);
        }
    }
    else
    {
#pragma unroll
        for(int e = 0; e < ELEMS; ++e)
        {
            k_pair[e] = __shfl_xor(k_val[e], is_neox ? K_HALF : 1);
        }
    }

#pragma unroll
    for(int e = 0; e < ELEMS; ++e)
    {
        const int dim = lane + e * THREADS;
        if(dim < ROPE_DIM)
        {
            const bool lead   = is_neox ? (dim < K_HALF) : (dim % 2 == 0);
            const int cos_idx = is_neox ? (dim < K_HALF ? dim : dim - K_HALF) : dim / 2;
            const float cos_v = static_cast<float>(cos_ptr[cos_idx]);
            const float sin_v = static_cast<float>(sin_ptr[cos_idx]);
            k_val[e] = lead ? (k_val[e] * cos_v - k_pair[e] * sin_v)
                            : (k_val[e] * cos_v + k_pair[e] * sin_v);
            k_val[e] = static_cast<float>(static_cast<scalar_t>(k_val[e]));
        }
    }

    if constexpr(FP4_OUT)
    {
        // cache_block_size carries kv_cache.size(3) in FP4 mode.
        constexpr int K_TILES = HEAD_DIM / 128;

        const int64_t fp4_block_idx = slot_idx / cache_block_size;
        const int fp4_pos_in_block  = static_cast<int>(slot_idx % cache_block_size);

#pragma unroll
        for(int e = 0; e < ELEMS; ++e)
        {
            const int dim         = lane + e * THREADS;
            const float kv_val    = k_val[e];
            const int group_idx   = dim / INDEXER_FP4_GROUP_SIZE;
            const float k_amax    = multithread_reduce<float, decltype(max_func)>(
                fabsf(kv_val), max_func, INDEXER_FP4_GROUP_SIZE);
            const uint8_t k_scale_e8m0 = indexer_fp4_scale_e8m0(k_amax);

            const float kv_partner = __shfl_down(kv_val, 1);
            if((dim & 1) == 0)
            {
                kv_cache[indexer_fp4_kv_data_offset(
                    fp4_block_idx, fp4_pos_in_block, dim / 2, K_TILES, cache_block_size)] =
                    indexer_fp4_pack_pair(
                        kv_val, kv_partner, indexer_fp4_scale_from_e8m0(k_scale_e8m0));
            }
            if(dim % INDEXER_FP4_GROUP_SIZE == 0)
            {
                kv_cache_scale[indexer_fp4_kv_scale_offset(
                    fp4_block_idx, fp4_pos_in_block, group_idx, K_TILES, cache_block_size)] =
                    k_scale_e8m0;
            }
        }
        return;
    }

    float k_abs[ELEMS];
#pragma unroll
    for(int e = 0; e < ELEMS; ++e)
    {
        k_abs[e] = fabsf(k_val[e]);
    }
    const float k_amax =
        indexer_lane_reduce<float, decltype(max_func), THREADS, ELEMS>(k_abs, max_func);

    const float q_fp8_max = static_cast<float>(opus::finfo<cache_t>::max());
    float k_scale = fmaxf(k_amax, 1e-4f) / q_fp8_max;
    if(use_ue8m0)
    {
        k_scale = exp2f(ceilf(log2f(k_scale)));
    }

    const int64_t block_idx    = slot_idx / cache_block_size;
    const int64_t block_offset = slot_idx % cache_block_size;

    if(lane == 0)
    {
        const int64_t dst_scale_idx =
            block_idx * cache_block_size * cache_stride + cache_block_size * HEAD_DIM +
            block_offset * HEAD_DIM * 4 / quant_block_size;
        reinterpret_cast<float*>(kv_cache)[dst_scale_idx / 4] = k_scale;
    }

    const float k_inv_scale = 1.0f / k_scale;
#pragma unroll
    for(int e = 0; e < ELEMS; ++e)
    {
        const int dim = lane + e * THREADS;
        int64_t dst_offset;
        if(preshuffle)
        {
            constexpr int TILE       = 16;
            const int token_tile_id  = block_offset / TILE;
            const int token_in_tile  = block_offset % TILE;
            const int col_tile_id    = dim / TILE;
            const int col_in_tile    = dim % TILE;
            dst_offset = block_idx * cache_block_size * cache_stride
                       + token_tile_id * (TILE * HEAD_DIM)
                       + col_tile_id   * (TILE * TILE)
                       + token_in_tile * TILE
                       + col_in_tile;
        }
        else
        {
            dst_offset =
                block_idx * cache_block_size * cache_stride + block_offset * HEAD_DIM + dim;
        }
        kv_cache[dst_offset] = opus::cast<cache_t>(k_val[e] * k_inv_scale);
    }
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
    const int quant_block_size,          // quantization block size
    const bool preshuffle                // source uses MFMA 16x16 preshuffled layout
)
{
    constexpr int VEC_SIZE = sizeof(float4) / sizeof(char);
    const int token_idx    = blockIdx.x * BLOCK_Y_SIZE + threadIdx.y;
    const int head_idx     = (blockIdx.y * BLOCK_X_SIZE + threadIdx.x) * VEC_SIZE;
    // Find batch index within a block
    __shared__ int batch_idx[BLOCK_Y_SIZE];
    for(int iter = 0; iter < (batch_size + BLOCK_X_SIZE - 1) / BLOCK_X_SIZE; iter++)
    {
        int tid = iter * BLOCK_X_SIZE + threadIdx.x;
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
    const int64_t block_offset         = inbatch_seq_idx % cache_block_size;
    const int64_t dst_inblock_offset   = token_idx * token_stride + head_idx;

    int64_t src_inblock_offset;
    if(preshuffle)
    {
        // Preshuffled layout: reverse the MFMA 16x16 tile mapping.
        // Works for any cache_block_size and head_dim that are multiples of 16.
        constexpr int TILE       = 16;
        const int token_tile_id  = block_offset / TILE;
        const int token_in_tile  = block_offset % TILE;
        const int col_tile_id    = head_idx / TILE;
        const int col_in_tile    = head_idx % TILE;
        src_inblock_offset = src_block_offset
                           + token_tile_id * (TILE * head_dim)
                           + col_tile_id   * (TILE * TILE)
                           + token_in_tile * TILE
                           + col_in_tile;
    }
    else
    {
        src_inblock_offset = src_block_offset + block_offset * head_dim + head_idx;
    }

    // Inference engines like ATOM and vLLM allocate head_size+4 bytes for each block in kv_cache to
    // store cache and scales. In models like DSv3.2 and GLM5, this gives 128+4=132 bytes per block,
    // which is not divisible by VEC_SIZE=16. Therefore, use byte addressing to advance through the
    // block before casting to dwordx4 for read/write.
    *reinterpret_cast<float4*>(dst_k + dst_inblock_offset) =
        *reinterpret_cast<const float4*>(kv_cache + src_inblock_offset);
    if(threadIdx.x == 0)
    {
        // Scale layout is unchanged regardless of preshuffle
        const int64_t cache_inblock_offset = block_offset * head_dim + head_idx;
        const int64_t src_scale_offset = src_block_offset + cache_block_size * head_dim +
                                         cache_inblock_offset * 4 / quant_block_size;
        *reinterpret_cast<float*>(dst_scale + dst_inblock_offset * 4 / quant_block_size) =
            *reinterpret_cast<const float*>(kv_cache + src_scale_offset);
    }
}

template <typename scalar_t, typename cache_t, bool IS_NEOX>
  inline __device__ void apply_token_rotary_embedding(
      const scalar_t *__restrict__ arr_in,
      cache_t *__restrict__ arr_out,  const scalar_t *__restrict__ cos_ptr,
      const scalar_t *__restrict__ sin_ptr, const float inv_scale,
      int rot_offset, int embed_dim)
  {
    int x_index, y_index;
    scalar_t cos, sin;
    if constexpr (IS_NEOX)
    {
      // GPT-NeoX style rotary embedding.
      x_index = rot_offset;
      y_index = embed_dim + rot_offset;
      cos = *(cos_ptr + x_index);
      sin = *(sin_ptr + x_index);
    }
    else
    {
      // GPT-J style rotary embedding.
      x_index = 2 * rot_offset;
      y_index = 2 * rot_offset + 1;
      cos = *(cos_ptr + x_index / 2);
      sin = *(sin_ptr + x_index / 2);
    }

    const scalar_t x = arr_in[x_index];
    const scalar_t y = arr_in[y_index];

    float f32_x = static_cast<float>(x);
    float f32_y = static_cast<float>(y);
    float f32_cos = static_cast<float>(cos);
    float f32_sin = static_cast<float>(sin);
    if constexpr (std::is_same_v<cache_t, opus::fp8_t>) {
        arr_out[x_index] = opus::cast<cache_t>(
                (f32_x * f32_cos - f32_y * f32_sin) * inv_scale);
        arr_out[y_index] = opus::cast<cache_t>(
                (f32_y * f32_cos + f32_x * f32_sin) * inv_scale);
    } else {
        arr_out[x_index] = opus::cast<cache_t>((f32_x * f32_cos - f32_y * f32_sin));
        arr_out[y_index] = opus::cast<cache_t>((f32_y * f32_cos + f32_x * f32_sin));
    }

  }

  template <typename scalar_t, typename cache_t, typename query_t, bool IS_NEOX, bool is_nope_first>
   __device__ void apply_rotary_embedding(
      const scalar_t *__restrict__ q_pe, // [batch_size, seq_len, num_heads,
                                    // head_size] or [num_tokens, num_heads,
                                    // head_size]
      const scalar_t *__restrict__ k_pe,   // [batch_size, seq_len, num_kv_heads,
                                    // head_size] or [num_tokens, num_kv_heads,
                                    // head_size]
      cache_t * __restrict__ kv_cache,
      query_t * __restrict__ q_out,
      const scalar_t *cos_ptr, const scalar_t *sin_ptr,
      const float inv_kscale,
      const float inv_qscale,                      //
      const int head_size, const int num_heads,
      const int num_kv_heads, const int rot_dim, const int token_idx,
      const int64_t q_pe_stride_0, const int64_t q_pe_stride_1, const int64_t key_stride,
      const int64_t q_out_stride_0, const int64_t q_out_stride_1, const int64_t kv_cache_offset)
  {
    const int embed_dim = rot_dim / 2;
    // const scalar_t *cos_ptr = cache_ptr;
    // const scalar_t *sin_ptr = cache_ptr + embed_dim;

    const int nq = num_heads * embed_dim;
    if constexpr (is_nope_first)
    {
      q_out += head_size - rot_dim;
      kv_cache += head_size - rot_dim;
    }

    for (int i = threadIdx.x; i < nq; i += blockDim.x)
    {
      const int head_idx = i / embed_dim;
      const int64_t token_head_in = token_idx * q_pe_stride_0 + head_idx * q_pe_stride_1;
      const int64_t token_head = token_idx * q_out_stride_0 + head_idx * q_out_stride_1;
      const int rot_offset = i % embed_dim;
      // to opt -> vec
      apply_token_rotary_embedding<scalar_t, query_t, IS_NEOX>(
          q_pe + token_head_in, q_out + token_head, cos_ptr, sin_ptr, inv_qscale, rot_offset, embed_dim);
    }
    const int nk = num_kv_heads * embed_dim;
    for (int i = threadIdx.x; i < nk; i += blockDim.x) 
    {
      const int head_idx = i / embed_dim;
      const int64_t token_head_in = token_idx * key_stride + head_idx * embed_dim;
      const int64_t token_head = kv_cache_offset;
      const int rot_offset = i % embed_dim;
      apply_token_rotary_embedding<scalar_t, cache_t, IS_NEOX>(
          k_pe + token_head_in, kv_cache + token_head, cos_ptr, sin_ptr, inv_kscale, rot_offset, embed_dim);
    }
  }
 template <typename scalar_t, typename cache_t, typename query_t, vllm::Fp8KVCacheDataType kv_dt, 
         vllm::Fp8KVCacheDataType q_dt, bool is_neox, bool is_nope_first=true, int32_t vec_size=4>
inline __device__ void fuse_qk_rope_concat_and_cache_mla_per_head_kernel_impl(
    const scalar_t* __restrict__ q_nope,  // [num_tokens, num_heads, kv_lora_rank]
    const scalar_t* __restrict__ q_pe,  // [num_tokens, num_heads, pe_dim]
    const scalar_t* __restrict__ kv_c,  // [num_tokens, kv_lora_rank]
    const scalar_t* __restrict__ k_pe,  // [num_tokens, pe_dim]
    cache_t* __restrict__ kv_cache,  // [num_blocks, block_size, (qk_lora_rank
                                     // + pe_dim)]
    query_t* __restrict__ q_out,  // [num_tokens, num_heads, kv_lora_rank + pe_dim]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int64_t* __restrict__ positions,     // [num_tokens]
    const scalar_t *__restrict__ cos_cache,        // [max_position, rot_dim //2]
    const scalar_t *__restrict__ sin_cache,        // [max_position, rot_dim //2]
    const int block_stride,                    //
    const int entry_stride,                    //
    const int q_nope_stride_0, const int q_nope_stride_1,                   //
    const int q_pe_stride_0, const int q_pe_stride_1,                     //
    const int q_out_stride_0, const int q_out_stride_1,                    //
    const int num_heads,                       //
    const int kv_c_stride,                     //
    const int k_pe_stride,                     //
    const int kv_lora_rank,                    //
    const int pe_dim,                          // 64
    const int block_size,                      //
    const float* k_scale,                         //
    const float* q_scale,
    const int max_position,
    const bool compute_all_q_rope
) {
  const int64_t token_idx = blockIdx.x / num_heads; //num_heads
  const int64_t head_idx = blockIdx.x % num_heads;

  const int64_t slot_idx = slot_mapping[token_idx];
  int64_t pos = positions[token_idx];

  // compute_all_q_rope == false (non-DCP default): restore the original early-return
  // so padded / cudagraph tokens (slot_idx < 0) skip Q RoPE + q_out entirely.
  // compute_all_q_rope == true (DCP): every rank needs all queries after the head
  // all-gather, so compute Q RoPE unconditionally; only the KV-cache writes
  // below are guarded by (slot_idx >= 0). Clamp pos defensively: padded tokens
  // may carry a stale position that would index cos/sin out of bounds.
  if (!compute_all_q_rope && slot_idx < 0) {
    return;
  }
  pos = pos < 0 ? 0 : (pos >= max_position ? max_position - 1 : pos);
  int64_t cos_sin_cache_offset = pos * 32;
  const scalar_t *cos_ptr = cos_cache + cos_sin_cache_offset;
  const scalar_t *sin_ptr = sin_cache + cos_sin_cache_offset;
  const int64_t block_idx = slot_idx;// / block_size;
  const int64_t block_offset = 0;//slot_idx % block_size;

  const int64_t head_size = kv_lora_rank + 64;
  // rotary emmbedding
  //concat
  static constexpr int32_t ooba_i = 4 / sizeof(scalar_t);
  static constexpr int32_t ooba_o = 4 / sizeof(cache_t);
  const int32_t oob_i             = (kv_lora_rank + ooba_i - 1) / ooba_i * ooba_i;
  const int32_t oob_o             = (kv_lora_rank + ooba_o - 1) / ooba_o * ooba_o;
  // Auto-adjust vec_size based on scalar_t size to avoid exceeding 16-byte limit
  // float (4 bytes): max vec_size=4 (16 bytes), half/bf16 (2 bytes): max vec_size=8 (16 bytes)
  static constexpr int32_t max_vec_size = (sizeof(scalar_t) == 4) ? 4 : vec_size;
  static constexpr int32_t vec_size_i = max_vec_size;
  static constexpr int32_t vec_size_o = vec_size_i;
  using opus_vec_i = opus::vector_t<scalar_t, vec_size_i>;
  using opus_vec_o = opus::vector_t<cache_t, vec_size_o>;
  using opus_vec_q = opus::vector_t<query_t, vec_size_o>;

  float inv_qscale = 1.0f;
  if constexpr (kv_dt != vllm::Fp8KVCacheDataType::kAuto) {
      inv_qscale = 1.0f / *q_scale;
  }
  static constexpr int32_t q_ooba_o = 4 / sizeof(query_t);
  auto const* q_ptr_i               = reinterpret_cast<scalar_t const*>(q_nope + token_idx * q_nope_stride_0 + head_idx * q_nope_stride_1);
  auto* q_ptr_o                     = reinterpret_cast<query_t*>(q_out + token_idx * q_out_stride_0 + head_idx * q_out_stride_1);
  // Use opus::make_gmem instead of ck_tile::make_buffer_view
  auto buffer_i = opus::make_gmem<scalar_t>(q_ptr_i, oob_i * sizeof(scalar_t));
  auto buffer_o = opus::make_gmem<query_t>(q_ptr_o, oob_o * sizeof(query_t));
  opus_vec_i vec_cur;  // Use opus_vec_i directly, no need to cast on load
  size_t vec_idx    = threadIdx.x;
  vec_cur = buffer_i.template load<vec_size_i>(vec_idx * vec_size_i);
  const int embed_dim = 32;
  const int nq =  embed_dim;
  q_out += head_size - pe_dim;

  scalar_t cos, sin;
  scalar_t x, y;
  int x_index, y_index;
  if(threadIdx.x < nq)
  {
    // GPT-NeoX style rotary embedding. 
    if constexpr (is_neox)
    {
      // GPT-NeoX style rotary embedding.
      x_index = threadIdx.x;
      y_index = embed_dim + threadIdx.x;
      cos = cos_ptr[x_index];//*(cos_ptr + x_index);
      sin = sin_ptr[x_index];//*(sin_ptr + x_index);
    }
    else
    {
      // GPT-J style rotary embedding.
      x_index = 2 * threadIdx.x;
      y_index = 2 * threadIdx.x + 1;
      cos = cos_ptr[x_index/2];//*(cos_ptr + x_index / 2);
      sin = sin_ptr[x_index/2];//*(sin_ptr + x_index / 2);
    }
    const int64_t token_head_in = token_idx * q_pe_stride_0 + head_idx * q_pe_stride_1;
    const int rot_offset = threadIdx.x;
    const scalar_t * q_pe_rot = q_pe + token_head_in;
    x = q_pe_rot[x_index];
    y = q_pe_rot[y_index];
  }
  if constexpr (q_dt == vllm::Fp8KVCacheDataType::kAuto) {
    buffer_o.template store<vec_size_o, opus_vec_i>(vec_cur, vec_idx * vec_size_o);
  } else {
    opus_vec_q vec_converted = aiter::scaled_cast<query_t>(vec_cur, inv_qscale);
    buffer_o.template store<vec_size_o, opus_vec_q>(vec_converted, vec_idx * vec_size_o);
  }
  float fp32_cos = static_cast<float>(cos);
  float fp32_sin = static_cast<float>(sin);
  if (head_idx == 0 && slot_idx >= 0) {
    auto const* ptr_i               = reinterpret_cast<scalar_t const*>(kv_c + token_idx * kv_c_stride);

    // Use opus::make_gmem for kv_c input
    auto kv_buffer_i = opus::make_gmem<scalar_t>(ptr_i, oob_i * sizeof(scalar_t));
    vec_cur = kv_buffer_i.template load<vec_size_i>(vec_idx * vec_size_i);

    float inv_kscale = 1.0f;
    if constexpr (kv_dt != vllm::Fp8KVCacheDataType::kAuto) {
      inv_kscale = 1.0f / *k_scale;
    }
    const int64_t token_head_in = token_idx * k_pe_stride;

     scalar_t k_x, k_y;
    if (threadIdx.x < 32)
    {
      const int rot_offset = threadIdx.x;
      const scalar_t* k_pe_rot =  k_pe + token_head_in;

      k_x = k_pe_rot[x_index];
      k_y = k_pe_rot[y_index];
    }
    const int64_t kv_cache_offset = block_idx * block_stride + block_offset * entry_stride;
    auto* ptr_o                     = reinterpret_cast<cache_t*>(kv_cache + kv_cache_offset);
    // Use opus::make_gmem for kv_cache output
    auto kv_buffer_o = opus::make_gmem<cache_t>(ptr_o, oob_o * sizeof(cache_t));
    if constexpr (kv_dt == vllm::Fp8KVCacheDataType::kAuto) {
        kv_buffer_o.template store<vec_size_o, opus_vec_i>(vec_cur, vec_idx * vec_size_o);
    } else {
        opus_vec_o vec_converted = aiter::scaled_cast<cache_t>(vec_cur, inv_kscale);
        kv_buffer_o.template store<vec_size_o, opus_vec_o>(vec_converted, vec_idx * vec_size_o);
    }

    float fp32_k_x = static_cast<float>(k_x);
    float fp32_k_y = static_cast<float>(k_y);

    if (threadIdx.x < 32)
    {
        kv_cache += kv_lora_rank;
        const int64_t token_head = kv_cache_offset;
        cache_t* kv_cache_rot = kv_cache + token_head;
        if constexpr (std::is_same_v<cache_t, opus::fp8_t>) {
          kv_cache_rot[x_index] = opus::cast<opus::fp8_t>(
               (fp32_k_x * fp32_cos - fp32_k_y * fp32_sin) * inv_kscale);
          kv_cache_rot[y_index] = opus::cast<opus::fp8_t>(
               (fp32_k_y * fp32_cos + fp32_k_x * fp32_sin) * inv_kscale);
        } else {
          kv_cache_rot[x_index] = static_cast<cache_t>((fp32_k_x * fp32_cos - fp32_k_y * fp32_sin));
          kv_cache_rot[y_index] = static_cast<cache_t>((fp32_k_y * fp32_cos + fp32_k_x * fp32_sin));
        }
    }
  }
  if (threadIdx.x < 32)
  {
    const int64_t token_head = token_idx * q_out_stride_0 + head_idx * q_out_stride_1;
    query_t * q_out_rot = q_out + token_head;
    float f32_x = static_cast<float>(x);
    float f32_y = static_cast<float>(y);
    if constexpr (std::is_same_v<query_t, opus::fp8_t>) {
        q_out_rot[x_index] = opus::cast<opus::fp8_t>((f32_x * fp32_cos - f32_y * fp32_sin) * inv_qscale);
        q_out_rot[y_index] = opus::cast<opus::fp8_t>((f32_y * fp32_cos + f32_x * fp32_sin) * inv_qscale);
    } else {
        q_out_rot[x_index] = static_cast<query_t>((f32_x * fp32_cos - f32_y * fp32_sin));
        q_out_rot[y_index] = static_cast<query_t>((f32_y * fp32_cos + f32_x * fp32_sin));
    }
  }

}
 template <typename scalar_t, typename cache_t, typename query_t, vllm::Fp8KVCacheDataType kv_dt, vllm::Fp8KVCacheDataType q_dt, int32_t vec_size=4>
__global__ void fuse_qk_rope_concat_and_cache_mla_per_head_kernel(
    const scalar_t* __restrict__ q_nope,  // [num_tokens, num_heads, kv_lora_rank]
    const scalar_t* __restrict__ q_pe,  // [num_tokens, num_heads, pe_dim]
    const scalar_t* __restrict__ kv_c,  // [num_tokens, kv_lora_rank]
    const scalar_t* __restrict__ k_pe,  // [num_tokens, pe_dim]
    cache_t* __restrict__ kv_cache,  // [num_blocks, block_size, (qk_lora_rank
                                     // + pe_dim)]
    query_t* __restrict__ q_out,  // [num_tokens, num_heads, kv_lora_rank + pe_dim]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int64_t* __restrict__ positions,     // [num_tokens]
    const scalar_t *__restrict__ cos_cache,        // [max_position, rot_dim //2]
    const scalar_t *__restrict__ sin_cache,        // [max_position, rot_dim //2]
    const int block_stride, const int entry_stride,                    //
    const int q_nope_stride_0, const int q_nope_stride_1,                   //
    const int q_pe_stride_0, const int q_pe_stride_1,                     //
    const int q_out_stride_0, const int q_out_stride_1,                    //
    const int num_heads,                       //
    const int kv_c_stride, const int k_pe_stride,                     //
    const int kv_lora_rank, const int pe_dim,                          // 64
    const int block_size,                      //
    const float* k_scale, const float* q_scale,
    bool is_neox, bool is_nope_first, const int max_position,
    const bool compute_all_q_rope
) {
  if (is_neox) {
    fuse_qk_rope_concat_and_cache_mla_per_head_kernel_impl<scalar_t, cache_t, query_t, kv_dt, q_dt, true, true, vec_size>(q_nope, q_pe, kv_c, k_pe, kv_cache, q_out, slot_mapping,
                                                  positions, cos_cache, sin_cache,block_stride, entry_stride, q_nope_stride_0, q_nope_stride_1, q_pe_stride_0, q_pe_stride_1,
                                                  q_out_stride_0, q_out_stride_1, num_heads, kv_c_stride, k_pe_stride, kv_lora_rank, pe_dim, block_size, k_scale, q_scale, max_position, compute_all_q_rope);
  } else {
    fuse_qk_rope_concat_and_cache_mla_per_head_kernel_impl<scalar_t, cache_t, query_t, kv_dt, q_dt, false, true, vec_size>(q_nope, q_pe, kv_c, k_pe, kv_cache, q_out, slot_mapping,
                                                  positions, cos_cache, sin_cache,block_stride, entry_stride, q_nope_stride_0, q_nope_stride_1, q_pe_stride_0, q_pe_stride_1,
                                                  q_out_stride_0, q_out_stride_1, num_heads, kv_c_stride, k_pe_stride, kv_lora_rank, pe_dim, block_size, k_scale, q_scale, max_position, compute_all_q_rope);
  }

}

// ============================================================================
// DeepSeek V3.1 MLA: fused QK RoPE(pe only) + static FP8 per-tensor quant +
// segmented paged KV cache write. No RMSNorm (q/k are already post-projection).
//
// q: nope quantized directly, pe RoPE'd then quantized.
// k: nope quantized directly, pe RoPE'd then quantized.
//
//   q_nope [T, H, KV_LORA]   q_pe [T, H, PE_DIM]
//   kv_c   [T, KV_LORA]      k_pe [T, PE_DIM]      (num_kv_heads == 1)
//   cos_cache / sin_cache [max_pos, PE_DIM/2]      (same dtype as input)
//   q_scale / k_scale [1] fp32                     (static per-tensor)
//
//   q_out [T, H, Q_OUT_DIM] fp8  -> [0:KV_LORA]=quant nope,
//                                   [KV_LORA:KV_LORA+PE_DIM]=quant rope,
//                                   [KV_LORA+PE_DIM:Q_OUT_DIM] left untouched (pad).
//   kv_cache flat per block:  [PAGE_SIZE*KV_LORA nope][PAGE_SIZE*PE_DIM rope] fp8
//     nope: block_idx*block_stride + block_offset*KV_LORA + d
//     rope: block_idx*block_stride + PAGE_SIZE*KV_LORA + block_offset*PE_DIM + d
//
// Launch: grid = T*H, block = KV_LORA/VEC threads. One block per (token, head)
// handles that head's q; when head_idx==0 the same block also handles the
// token's k (kv=1). The nope segment is written with VEC-wide vectorized
// loads/stores (128-bit for 16-bit inputs); the pe segment is RoPE'd per
// element (PE_DIM threads).
// ============================================================================
// Which heads-per-block instantiations stage Q through TDM on gfx1250: bit HPT
// set means that HPT does. Read by the kernel and by the host's dynamic-LDS
// sizing, so the two cannot disagree. Not HPT=1: that path serves the small
// launches sitting on the latency floor, where the tensor issue-and-wait costs
// more than the direct loads it replaces (+2..+5%, H=16..128 at T<=192 and
// H=3 T=1536). HPT=2/4/8 gain 3/11/6%.
#ifndef AITER_MLA_SEG_TDM_HPT_MASK
#define AITER_MLA_SEG_TDM_HPT_MASK 0x114
#endif

//
// PAGE_SIZE == 0 selects the plain per-entry layout of
// fused_qk_rope_concat_and_cache_mla instead (runtime block_size/entry_stride,
// entry = [nope | pe], or [pe | nope] when !NOPE_FIRST; q_out rows likewise).
// An output whose type equals the input's is stored unscaled, as kAuto.
template <typename scalar_t, typename cache_t, typename query_t, int KV_LORA, int PE_DIM,
          int PAGE_SIZE, bool IS_NEOX, bool NOPE_FIRST, int VEC, int HPT,
          bool COMPUTE_ALL_Q_ROPE = false, bool K_EARLY = (HPT <= 2)>
__global__ void fused_qk_rope_concat_and_cache_mla_seg_kernel(
    const scalar_t* __restrict__ q_nope,    // [T, H, KV_LORA]
    const scalar_t* __restrict__ q_pe,      // [T, H, PE_DIM]
    const scalar_t* __restrict__ kv_c,      // [T, KV_LORA]
    const scalar_t* __restrict__ k_pe,      // [T, PE_DIM]
    cache_t* __restrict__ kv_cache,         // flat [num_blocks, block_stride]
    query_t* __restrict__ q_out,            // [T, H, Q_OUT_DIM]
    const int64_t* __restrict__ slot_mapping,
    const int64_t* __restrict__ positions,
    const scalar_t* __restrict__ cos_cache, // [max_pos, PE_DIM/2]
    const scalar_t* __restrict__ sin_cache, // [max_pos, PE_DIM/2]
    const float* __restrict__ q_scale,
    const float* __restrict__ k_scale,
    const int num_heads,
    const int64_t q_nope_stride_0, const int64_t q_nope_stride_1,
    const int64_t q_pe_stride_0,   const int64_t q_pe_stride_1,
    const int64_t q_out_stride_0,  const int64_t q_out_stride_1,
    const int64_t kv_c_stride,
    const int64_t k_pe_stride,
    const int64_t cos_stride0,
    const int64_t sin_stride0,
    const int64_t block_stride,
    const int max_position,
    const int64_t entry_stride = 0,
    const int block_size = PAGE_SIZE,
    const int num_tokens = 0,
    const uint32_t tok_magic = 0, // blockIdx.x / num_tokens, see mla_rope_fast_div
    const int tok_shift = 0,
    const uint32_t bs_magic = 0, // slot / block_size
    const int bs_shift = 0)
{
    constexpr int HALF    = PE_DIM / 2;
    constexpr int NUM_VEC = KV_LORA / VEC; // nope vectors per row
    constexpr int NOPE_OFF = NOPE_FIRST ? 0 : PE_DIM;
    constexpr int PE_OFF   = NOPE_FIRST ? KV_LORA : 0;
    static_assert(PAGE_SIZE == 0 || NOPE_FIRST, "the segmented layout is nope-first");
    using in_vec_t  = opus::vector_t<scalar_t, VEC>;
    using out_vec_t = opus::vector_t<cache_t, VEC>;
    using q_vec_t   = opus::vector_t<query_t, VEC>;
    // A template parameter: tested at run time it cost ~3% at eight heads per
    // block (H=128 T=1536), even when false.
    constexpr bool compute_all_q_rope = COMPUTE_ALL_Q_ROPE;
    static_assert(PAGE_SIZE == 0 || !COMPUTE_ALL_Q_ROPE);
    auto cvt_vec = [](const in_vec_t& v, float inv_scale, auto* out) {
        using out_t = std::remove_cv_t<std::remove_pointer_t<decltype(out)>>;
        if constexpr(std::is_same_v<out_t, scalar_t>)
            return v;
        else
            return aiter::scaled_cast<out_t>(v, inv_scale);
    };

    // Two-dimensional grid so the coordinates arrive directly. A flat grid has
    // to divide and modulo by num_heads, which is a runtime value, and
    // token_idx heads every address chain in the block -- so that divide sits
    // at the front of the critical path, once per block, across T*H blocks.
    // x is the head so that consecutive blocks stay within one token and keep
    // sharing its slot, position and cos/sin row, which is the order the flat
    // grid already produced.
    //
    // The per-entry layout (PAGE_SIZE == 0) runs a flat grid, head-group
    // major: block = group * T + token. Consecutive blocks round-robin over
    // the XCDs, and token-major put every K-writing (group 0) block on one
    // XCD whenever H / HPT is a multiple of 8 (H=16 T=1536 7.50 -> 6.62 us,
    // H=128 T=192 7.14 -> 6.25 us). The grid stays one-dimensional because a
    // tall grid dispatches slower (T=1 H=16: 3.34 us as 1 x 16, 3.22 as
    // 16 x 1), and the divide by T is a host-built magic multiply, kept off
    // the front of every block's address chain.
    int64_t token_idx;
    int     head_base;
    if constexpr(PAGE_SIZE == 0)
    {
        const uint32_t bid   = blockIdx.x;
        const uint32_t group = (__umulhi(bid, tok_magic) + bid) >> tok_shift;
        token_idx            = bid - group * static_cast<uint32_t>(num_tokens);
        head_base            = group * HPT;
    }
    else
    {
        token_idx = blockIdx.y;
        head_base = blockIdx.x * HPT;
    }

    // The slot only gates the writes. Every Q load is addressed from
    // token/head alone and stays in bounds for a padded token too, so on the
    // one-head-per-block path -- on its latency floor -- the early return waits
    // until they are issued: taken first, it held every Q load behind the
    // slot_mapping round trip (H=16 T=192 -8.3%, H=32 T=192 -7.3%). With more
    // heads per block the op is bandwidth-bound, and deferring it measured ~1%
    // slower there, so that path returns up front as before.
    const int64_t slot_idx = slot_mapping[token_idx];
    // compute_all_q_rope (DCP) keeps a padded token's Q; only K is skipped.
    if constexpr(HPT > 1)
    {
        if(slot_idx < 0 && !compute_all_q_rope)
            return; // uniform across the block (same token)
    }

    // cos/sin are read at <buf>[<row> + i]; <end> is an offset past the buffer.
    // A padded token reaches this on the one-head path and may carry a stale
    // position, so there the descriptors span the whole cache with the row in
    // the offset: the hardware range check then covers the row too, and any
    // position reads in bounds or not at all. Clamping it instead cost that
    // path 3-5% (H=16/32 T=192), and gating it on the slot would put the slot
    // round trip back in front of the cos/sin loads. With more heads per block
    // a padded token has already returned, and a per-row descriptor keeps the
    // row offset scalar (~2% at H=128 T=1536).
    int64_t pos = positions[token_idx];
    // A padded token whose Q is kept may carry a stale position: clamp it.
    if(compute_all_q_rope)
        pos = pos < 0 ? 0 : (pos >= max_position ? max_position - 1 : pos);
    const scalar_t* cos_base = cos_cache;
    const scalar_t* sin_base = sin_cache;
    int cos_row = 0, sin_row = 0, cos_end = HALF, sin_end = HALF;
    if constexpr(HPT == 1)
    {
        cos_row = static_cast<int>(pos * cos_stride0);
        sin_row = static_cast<int>(pos * sin_stride0);
        cos_end = static_cast<int>(max_position * cos_stride0);
        sin_end = static_cast<int>(max_position * sin_stride0);
    }
    else
    {
        cos_base += pos * cos_stride0;
        sin_base += pos * sin_stride0;
    }
    auto cos_buf = opus::make_gmem<scalar_t>(cos_base, cos_end * sizeof(scalar_t));
    auto sin_buf = opus::make_gmem<scalar_t>(sin_base, sin_end * sizeof(scalar_t));

    // ---- helper: the whole pe segment, VEC elements per thread ----
    //
    // The segment is walked a chunk at a time rather than an element at a
    // time. Taken per element it costs four narrow loads and a one-byte store
    // for every 2 bytes of input, against one wide load per 8 elements on the
    // nope path; the segment is an eighth of Q's bytes and was measuring ~30%
    // of the kernel.
    //
    // Both pairings keep a chunk's operands inside one load. NEOX pairs across
    // HALF, a multiple of VEC, so a chunk's partner is a whole chunk and its
    // cos/sin indices are the chunk's own folded into the low half. GPT-J
    // pairs adjacent elements, so a chunk already holds both halves of every
    // pair and only cos/sin narrows, to VEC/2.
    constexpr int PE_CHUNKS = PE_DIM / VEC;
    static_assert(PE_DIM % VEC == 0 && HALF % VEC == 0);
    using half_vec_t = opus::vector_t<scalar_t, VEC / 2>;
    using pe_cs_t    = std::conditional_t<IS_NEOX, in_vec_t, half_vec_t>;

    // Issue and consume are split so the caller can put the segment's loads in
    // front of the nope pass. Every operand address is known as soon as the
    // token is -- pe from the row, cos/sin from positions[token] -- so nothing
    // here has to wait for nope, and the four loads then overlap it instead of
    // starting a fresh dependency chain after it.
    // With more than one head per block the NEOX partner chunk is taken
    // across lanes (see rope_pe_emit); with one, the op is on its latency
    // floor and a second parallel load beats waiting on x to exchange it
    // (H=32 T=192: +0.7% exchanged, against -6..-8.5% at HPT > 1).
    constexpr bool PE_EXCHANGE = IS_NEOX && HPT > 1;
    struct PeOperands
    {
        in_vec_t x, y;
        pe_cs_t c, s;
    };

    // `active` masks like rope_pe_emit's: an inactive lane reads past
    // num_records, which returns zero without touching the cache -- no branch,
    // and no extra requests either (clamping to a live row instead cost the
    // one-head path eight times its pe loads).
    //
    // pe_ptr is the wave's first head row and row_off (elements) the lane's
    // own row within the `extent` elements after it. The buffer descriptor has
    // to stay wave-uniform: built from a lane-derived head, it landed in VGPRs
    // and the compiler wrapped every access in a readfirstlane waterfall loop.
    auto rope_pe_issue = [&](const scalar_t* pe_ptr, int row_off, int extent, int chunk,
                             bool active) -> PeOperands {
        PeOperands o{};
        const int d0 = chunk * VEC;
        auto pe_buf  = opus::make_gmem<scalar_t>(pe_ptr, extent * sizeof(scalar_t));
        o.x = pe_buf.template load<VEC>(active ? row_off + d0 : extent);
        if constexpr(IS_NEOX)
        {
            const int cidx = d0 < HALF ? d0 : d0 - HALF;
            if constexpr(!PE_EXCHANGE)
                o.y = pe_buf.template load<VEC>(active ? row_off + (d0 ^ HALF) : extent);
            o.c = cos_buf.template load<VEC>(active ? cos_row + cidx : cos_end);
            o.s = sin_buf.template load<VEC>(active ? sin_row + cidx : sin_end);
        }
        else
        {
            o.c = cos_buf.template load<VEC / 2>(active ? cos_row + d0 / 2 : cos_end);
            o.s = sin_buf.template load<VEC / 2>(active ? sin_row + d0 / 2 : sin_end);
        }
        return o;
    };
    // Same row_off / extent convention as rope_pe_issue.

    // `active` masks the store without touching EXEC: an inactive lane writes
    // at PE_DIM, past the buffer's num_records, which the hardware drops. A
    // divergent `if` here put s_and_saveexec right behind the nope store, and
    // EXEC being an operand of that in-flight store it had to drain the whole
    // address queue first -- s_wait_xcnt, 18% of the wave at H=32 T=192 in ATT.
    // The rotated, converted chunk. `exchange` (a constant at every call)
    // takes the NEOX partner across lanes rather than from o.y.
    auto rope_pe_calc = [&](const PeOperands& o, auto* out_tag, float inv_scale, int chunk,
                            bool exchange) {
        using o_t = std::remove_pointer_t<decltype(out_tag)>;
        using o_vec_t = opus::vector_t<o_t, VEC>;
        auto cvt = [](float r, float s) -> o_t {
            if constexpr(std::is_same_v<o_t, opus::fp8_t>)
                return opus::cast<o_t>(r * s);
            else
                return static_cast<o_t>(r);
        };
        const int d0 = chunk * VEC;
        o_vec_t vout;
        if constexpr(IS_NEOX)
        {
            // The partner chunk is the one lane chunk^HALF/VEC already loaded
            // -- the same 128 B row -- so take it across lanes instead of
            // reading it again. Loaded twice, it went to GL2 as a second
            // 128 B request per head (34% of read requests at H=128, against
            // 4 x 256 B for the nope row). Done here rather than at issue so
            // the exchange does not force a wait on x before the nope loads go
            // out. Partners share a PE_CHUNKS-lane group on every caller.
            in_vec_t y = o.y;
            if(PE_EXCHANGE && exchange)
            {
                constexpr int PARTNER = HALF / VEC;
                using u32x4 = uint32_t __attribute__((ext_vector_type(4)));
                static_assert(sizeof(in_vec_t) == sizeof(u32x4));
                const u32x4 xw = __builtin_bit_cast(u32x4, o.x);
                u32x4 yw;
#pragma unroll
                for(int w = 0; w < 4; ++w)
                    yw[w] = static_cast<uint32_t>(
                        __shfl_xor(static_cast<int>(xw[w]), PARTNER, PE_CHUNKS));
                y = __builtin_bit_cast(in_vec_t, yw);
            }
            const bool low   = d0 < HALF;
#pragma unroll
            for(int i = 0; i < VEC; ++i)
            {
                const float xv = static_cast<float>(o.x[i]);
                const float yv = static_cast<float>(y[i]);
                const float cv = static_cast<float>(o.c[i]);
                const float sv = static_cast<float>(o.s[i]);
                const float r  = low ? (xv * cv - yv * sv) : (xv * cv + yv * sv);
                vout[i]        = cvt(r, inv_scale);
            }
        }
        else
        {
#pragma unroll
            for(int k = 0; k < VEC / 2; ++k)
            {
                const float a  = static_cast<float>(o.x[2 * k]);
                const float b  = static_cast<float>(o.x[2 * k + 1]);
                const float cv = static_cast<float>(o.c[k]);
                const float sv = static_cast<float>(o.s[k]);
                vout[2 * k]     = cvt(a * cv - b * sv, inv_scale);
                vout[2 * k + 1] = cvt(b * cv + a * sv, inv_scale);
            }
        }
        return vout;
    };
    auto rope_pe_emit = [&](const PeOperands& o, auto* out_ptr, int row_off, int extent,
                            float inv_scale, int chunk, bool active) {
        using o_t     = std::remove_pointer_t<decltype(out_ptr)>;
        using o_vec_t = opus::vector_t<o_t, VEC>;
        auto out_buf  = opus::make_gmem<o_t>(out_ptr, extent * sizeof(o_t));
        out_buf.template store<VEC, o_vec_t>(rope_pe_calc(o, out_ptr, inv_scale, chunk, true),
                                             active ? row_off + chunk * VEC : extent);
    };

    auto rope_pe_seg = [&](const scalar_t* pe_ptr, cache_t* out_ptr, float inv_scale) {
        if(threadIdx.x < PE_CHUNKS)
            rope_pe_emit(rope_pe_issue(pe_ptr, 0, PE_DIM, threadIdx.x, true), out_ptr, 0,
                         PE_DIM, inv_scale, threadIdx.x, true);
    };

    // K_EARLY: K's loads go out with Q's, so the K block pays one memory
    // round trip rather than a second one after its Q stores. That wins while
    // the op is latency-bound (H=16 T=192 4.24 -> 3.47 us, H=128 T=32 4.15 ->
    // 3.52 us); once it is bandwidth-bound the registers they hold cost more
    // (+5..8% at eight heads, +10% at two heads with 64 rows per CU).
    in_vec_t k_nope_in{};
    PeOperands k_pe_op{};
    if(K_EARLY && head_base == 0) // uniform
    {
        auto in_buf = opus::make_gmem<scalar_t>(kv_c + token_idx * kv_c_stride,
                                                KV_LORA * sizeof(scalar_t));
        k_nope_in   = in_buf.template load<VEC>(threadIdx.x * VEC);
        k_pe_op     = rope_pe_issue(k_pe + token_idx * k_pe_stride, 0, PE_DIM,
                                    threadIdx.x % PE_CHUNKS, threadIdx.x < PE_CHUNKS);
    }

    // ================= Q (HPT heads per block) =================
    //
    // The block is one thread per nope vector, so a thread carries a single
    // 16 B load per head. With one head per block that is all it ever has in
    // flight, which is far short of what it takes to cover memory latency on
    // this part: a plain bf16->fp8 cast of the same footprint runs ~1.7x
    // faster. So each block takes HPT heads of one token and issues every
    // head's nope and pe loads before consuming any -- HPT independent 16 B
    // requests per thread instead of one. The head index comes from blockIdx
    // and the unrolled loop counter, so it stays scalar; widening the block
    // with more threads instead made it thread-derived, pushed the address
    // chain into VGPRs and measured slower at every width.
#if defined(__gfx1250__)
    // gfx1250: stage Q through the tensor engine, within the three tensor ops
    // a wave can hold in flight. A wave consumes only what it staged, so there
    // is no block barrier, just the tensor wait. The host sizes the dynamic
    // LDS to match (two waves x WAVE_ELEMS, the same for both layouts at
    // eight heads).
    //
    // Per-entry layout, eight heads, contiguous q_out rows: each wave owns
    // four whole heads -- a nope tile (512 x 4) and a pe tile (64 x 4) --
    // converts them in place into q_out's row layout, and stores from there
    // in address order, nine contiguous 256 B runs. Stored per head (nope by
    // column, pe by chunk), the 576-element rows split their 128 B lines
    // across stores and waves, and GL2 sent 3.6x the write requests and 1.8x
    // the reads to memory (H=128 T=4096 68.3 -> 59.8 us, T=1536 26.4 -> 24.7
    // us).
    if constexpr(((AITER_MLA_SEG_TDM_HPT_MASK >> HPT) & 1) != 0)
    {
        if(PAGE_SIZE == 0 && HPT == 8 && q_out_stride_1 == KV_LORA + PE_DIM)
        {
            if constexpr(PAGE_SIZE == 0 && HPT == 8)
            {
                static_assert(sizeof(query_t) <= sizeof(scalar_t));
                constexpr int W          = 32;
                constexpr int ROWS       = HPT / 2;
                constexpr int ROW        = KV_LORA + PE_DIM;
                constexpr int WAVE_ELEMS = ROWS * ROW;
                constexpr int ROW_CH     = ROW / VEC; // output chunks per head
                static_assert(NUM_VEC == 2 * W && ROWS * PE_CHUNKS == W &&
                              ROWS * ROW_CH % W == 0);
                extern __shared__ char mla_seg_lds[];
                using NopeWin = opus::tdm<scalar_t, opus::seq<KV_LORA, ROWS>>;
                using PeWin   = opus::tdm<scalar_t, opus::seq<PE_DIM, ROWS>>;
#if defined(__HIP_DEVICE_COMPILE__)
                const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / W);
#else
                const int wave = threadIdx.x / W;
#endif
                const int wave_head = head_base + wave * ROWS;
                const opus::u32_t lds_base =
                    static_cast<opus::u32_t>(reinterpret_cast<__UINTPTR_TYPE__>(mla_seg_lds));
                const opus::u32_t wave_off = static_cast<opus::u32_t>(wave * WAVE_ELEMS);

                auto wn =
                    opus::make_tdm<NopeWin>(lds_base, q_nope + token_idx * q_nope_stride_0,
                                            KV_LORA, num_heads, q_nope_stride_1, 0, wave_head);
                wn.async_load(wave_off);
                auto wp = opus::make_tdm<PeWin>(lds_base, q_pe + token_idx * q_pe_stride_0, PE_DIM,
                                                num_heads, q_pe_stride_1, 0, wave_head);
                wp.async_load(wave_off + ROWS * KV_LORA);

                // cos/sin go the ordinary way, issued while the tiles are in flight.
                const int lane     = threadIdx.x % W;
                const int pe_r     = lane / PE_CHUNKS;
                const int pe_chunk = lane % PE_CHUNKS;
                const int d0       = pe_chunk * VEC;
                PeOperands pe_op{};
                if constexpr(IS_NEOX)
                {
                    const int cidx = d0 < HALF ? d0 : d0 - HALF;
                    pe_op.c        = cos_buf.template load<VEC>(cos_row + cidx);
                    pe_op.s        = sin_buf.template load<VEC>(sin_row + cidx);
                }
                else
                {
                    pe_op.c = cos_buf.template load<VEC / 2>(cos_row + d0 / 2);
                    pe_op.s = sin_buf.template load<VEC / 2>(sin_row + d0 / 2);
                }
                const float inv_qscale = 1.0f / (*q_scale);
                opus::s_wait_tensorcnt<0>();

                const __UINTPTR_TYPE__ lds =
                    reinterpret_cast<__UINTPTR_TYPE__>(mla_seg_lds) + wave_off * sizeof(scalar_t);
                auto lds_in = [&](int elem) -> in_vec_t {
                    return *reinterpret_cast<const OPUS_LDS_ADDR in_vec_t*>(
                        lds + elem * sizeof(scalar_t));
                };
                auto lds_out = [&](int elem) -> OPUS_LDS_ADDR q_vec_t* {
                    return reinterpret_cast<OPUS_LDS_ADDR q_vec_t*>(lds + elem * sizeof(query_t));
                };
                // Lane chunk i * W + lane of the nope tile: head i / 2, a whole row
                // per two chunks since NUM_VEC == 2 * W.
                in_vec_t nope_in[2 * ROWS];
#pragma unroll
                for(int i = 0; i < 2 * ROWS; ++i)
                    nope_in[i] = lds_in((i / 2) * KV_LORA + ((i % 2) * W + lane) * VEC);
                pe_op.x = lds_in(ROWS * KV_LORA + pe_r * PE_DIM + d0);
                if constexpr(IS_NEOX)
                    pe_op.y = lds_in(ROWS * KV_LORA + pe_r * PE_DIM + (d0 ^ HALF));
                // Every staged read precedes the first in-place write.
#pragma unroll
                for(int i = 0; i < 2 * ROWS; ++i)
                    *lds_out((i / 2) * ROW + NOPE_OFF + ((i % 2) * W + lane) * VEC) =
                        cvt_vec(nope_in[i], inv_qscale, q_out);
                *lds_out(pe_r * ROW + PE_OFF + d0) =
                    rope_pe_calc(pe_op, q_out, inv_qscale, pe_chunk, false);

                // The rows are contiguous, so chunk j of the wave's heads, at LDS
                // element j * VEC, goes to the same offset in q_out.
                auto out_buf =
                    opus::make_gmem<query_t>(q_out + token_idx * q_out_stride_0 + wave_head * ROW,
                                             ROWS * ROW * sizeof(query_t));
#pragma unroll
                for(int i = 0; i < ROWS * ROW_CH / W; ++i)
                    out_buf.template store<VEC, q_vec_t>(*lds_out((i * W + lane) * VEC),
                                                         (i * W + lane) * VEC);
            }
        }
        // Otherwise each wave owns half the nope columns of every head plus up
        // to four heads' pe rows, as two tiles (256 x HPT and 64 x PE_ROWS).
        // Two heads a wave are too few for the in-order store to pay (H=16
        // T=1536, H=128 T=192: +4..5%), and padded rows break the address
        // order, leaving the staging pure latency (768-wide rows, H=128
        // T=4096: 72.1 us against 68.9 per head). The segmented layout's
        // q_out is padded, and even the untaken branch cost it 1.5..3%.
        else
        {
            constexpr int W          = 32;
            constexpr int COLS       = KV_LORA / 2;
            // pe lanes are threadIdx.x / PE_CHUNKS, so a wave's 32 lanes carry at
            // most four heads' pe: all of them in wave 0 for HPT <= 4, four each
            // for HPT == 8.
            constexpr int PE_ROWS    = HPT < 4 ? HPT : 4;
            constexpr int WAVE_ELEMS = HPT * COLS + PE_ROWS * PE_DIM;
            static_assert(NUM_VEC == 2 * W && COLS == W * VEC && 4 * PE_CHUNKS == W);
            extern __shared__ char mla_seg_lds[];
            using NopeWin = opus::tdm<scalar_t, opus::seq<COLS, HPT>>;
            using PeWin   = opus::tdm<scalar_t, opus::seq<PE_DIM, PE_ROWS>>;
#if defined(__HIP_DEVICE_COMPILE__)
            const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / W);
#else
            const int wave = threadIdx.x / W;
#endif
            const opus::u32_t lds_base =
                static_cast<opus::u32_t>(reinterpret_cast<__UINTPTR_TYPE__>(mla_seg_lds));
            const opus::u32_t wave_off = static_cast<opus::u32_t>(wave * WAVE_ELEMS);

            auto wn = opus::make_tdm<NopeWin>(lds_base, q_nope + token_idx * q_nope_stride_0,
                                              KV_LORA, num_heads, q_nope_stride_1,
                                              wave * COLS, head_base);
            wn.async_load(wave_off);
            const bool wave_has_pe = wave * PE_ROWS < HPT;
            if(wave_has_pe)
            {
                auto wp = opus::make_tdm<PeWin>(lds_base, q_pe + token_idx * q_pe_stride_0,
                                                PE_DIM, num_heads, q_pe_stride_1,
                                                0, head_base + wave * PE_ROWS);
                wp.async_load(wave_off + HPT * COLS);
            }

            // cos/sin go the ordinary way, issued while the tiles are in flight.
            const int pe_row   = (threadIdx.x % W) / PE_CHUNKS;
            const int pe_chunk = threadIdx.x % PE_CHUNKS;
            const int d0       = pe_chunk * VEC;
            PeOperands pe_op{};
            if constexpr(IS_NEOX)
            {
                const int cidx = d0 < HALF ? d0 : d0 - HALF;
                pe_op.c        = cos_buf.template load<VEC>(cos_row + cidx);
                pe_op.s        = sin_buf.template load<VEC>(sin_row + cidx);
            }
            else
            {
                pe_op.c = cos_buf.template load<VEC / 2>(cos_row + d0 / 2);
                pe_op.s = sin_buf.template load<VEC / 2>(sin_row + d0 / 2);
            }
            const float inv_qscale = 1.0f / (*q_scale);
            opus::s_wait_tensorcnt<0>();
            // One head per block defers the padded-token return until the loads
            // are out (see slot_idx); it must also wait for them, or the block
            // could retire with a tensor DMA still writing its LDS.
            if constexpr(HPT == 1)
            {
                if(slot_idx < 0 && !compute_all_q_rope)
                    return;
            }

            const __UINTPTR_TYPE__ lds = reinterpret_cast<__UINTPTR_TYPE__>(mla_seg_lds);
            auto lds_vec = [&](int elem) -> in_vec_t {
                return *reinterpret_cast<const OPUS_LDS_ADDR in_vec_t*>(
                    lds + static_cast<__UINTPTR_TYPE__>(wave_off + elem) * sizeof(scalar_t));
            };
            const int col = (threadIdx.x % W) * VEC;
            // Every head is converted before any is stored. Converted and stored
            // one at a time, each head's store data landed in the registers the
            // next head's was about to use, and the wave sat in s_wait_xcnt behind
            // every store until the memory pipe had read it out.
            q_vec_t nope_out[HPT];
#pragma unroll
            for(int k = 0; k < HPT; ++k)
                nope_out[k] = cvt_vec(lds_vec(k * COLS + col), inv_qscale, q_out);
            query_t* q_out_tok = q_out + token_idx * q_out_stride_0;
#pragma unroll
            for(int k = 0; k < HPT; ++k)
            {
                auto out_buf = opus::make_gmem<query_t>(
                    q_out_tok + (head_base + k) * q_out_stride_1 + NOPE_OFF,
                    KV_LORA * sizeof(query_t));
                out_buf.template store<VEC, q_vec_t>(nope_out[k], wave * COLS + col);
            }
            {
                const bool pe_active = wave * PE_ROWS + pe_row < HPT;
                const int  pe_r      = pe_active ? pe_row : 0; // keep the LDS read in range
                pe_op.x = lds_vec(HPT * COLS + pe_r * PE_DIM + d0);
                // Without the lane exchange rope_pe_emit reads the NEOX partner
                // from o.y; here it is the same staged row, HALF elements over.
                if constexpr(IS_NEOX && !PE_EXCHANGE)
                    pe_op.y = lds_vec(HPT * COLS + pe_r * PE_DIM + (d0 ^ HALF));
                rope_pe_emit(pe_op,
                             q_out_tok + (head_base + wave * PE_ROWS) * q_out_stride_1 + PE_OFF,
                             static_cast<int>(pe_r * q_out_stride_1),
                             static_cast<int>((PE_ROWS - 1) * q_out_stride_1) + PE_DIM,
                             inv_qscale, pe_chunk, pe_active);
            }
        }
    }
    else
#endif
    {
        const float inv_qscale = 1.0f / (*q_scale);

        // pe is spread across the block, one head per PE_CHUNKS lanes, rather
        // than every head's pe queued on lanes 0..PE_CHUNKS-1. A head's pe
        // operands are four VEC-wide vectors, and VGPRs are allocated per
        // wave however few lanes use them, so stacking HPT heads on the same
        // lanes costs 16*HPT registers for the whole wave -- the reason HPT=8
        // collapsed. Spread, each lane holds at most one head's pe whatever
        // HPT is. pe_head is lane-derived, which is fine: pe addressing was
        // per-lane already, and the nope path keeps its scalar head index.
        static_assert(HPT * PE_CHUNKS <= NUM_VEC, "pe lanes must fit in the block");
        const int pe_head  = threadIdx.x / PE_CHUNKS;
        const int pe_chunk = threadIdx.x % PE_CHUNKS;
        const bool pe_lane = pe_head < HPT;
        // Lanes past the last head are masked by offset, not by a branch: no
        // EXEC write near in-flight memory ops.
        const int pe_h     = pe_lane ? pe_head : 0;
        // A wave holding no pe lane at all skips the segment outright. The
        // test is on the wave's first lane, so it is uniform and compiles to a
        // scalar branch -- it moves SCC, not EXEC, and costs no drain. Without
        // it the one-head path's second wave ran the whole pe segment masked
        // off, which cost more than the drain saved once the grid reached
        // ~6K blocks (+2.6% at H=32 T=192).
#if defined(__HIP_DEVICE_COMPILE__)
        const bool wave_pe =
            __builtin_amdgcn_readfirstlane(static_cast<int>(threadIdx.x)) < HPT * PE_CHUNKS;
#else
        const bool wave_pe = true;
#endif
        PeOperands pe_op{};
        if(wave_pe)
            pe_op = rope_pe_issue(q_pe + token_idx * q_pe_stride_0 + head_base * q_pe_stride_1,
                                  static_cast<int>(pe_h * q_pe_stride_1),
                                  static_cast<int>((HPT - 1) * q_pe_stride_1) + PE_DIM,
                                  pe_chunk, pe_lane);

        in_vec_t nope_in[HPT];
#pragma unroll
        for(int k = 0; k < HPT; ++k)
        {
            const int h = head_base + k;
            auto in_buf = opus::make_gmem<scalar_t>(
                q_nope + token_idx * q_nope_stride_0 + h * q_nope_stride_1,
                KV_LORA * sizeof(scalar_t));
            nope_in[k] = in_buf.template load<VEC>(threadIdx.x * VEC);
        }
        // A padded token (HPT > 1 returned up front) is dropped the same way,
        // by an out-of-bounds offset: a return here is a branch the compiler
        // may sink the nope load past, which ATT showed it doing.
        const bool q_live =
            HPT > 1 || slot_idx >= 0 || compute_all_q_rope; // uniform over the block
#pragma unroll
        for(int k = 0; k < HPT; ++k)
        {
            const int h        = head_base + k;
            query_t* q_out_row =
                q_out + token_idx * q_out_stride_0 + h * q_out_stride_1 + NOPE_OFF;
            auto out_buf = opus::make_gmem<query_t>(q_out_row, KV_LORA * sizeof(query_t));
            out_buf.template store<VEC, q_vec_t>(
                cvt_vec(nope_in[k], inv_qscale, q_out),
                q_live ? static_cast<int>(threadIdx.x) * VEC : KV_LORA);
        }
        if(wave_pe)
            rope_pe_emit(pe_op,
                         q_out + token_idx * q_out_stride_0 + head_base * q_out_stride_1 +
                             PE_OFF,
                         static_cast<int>(pe_h * q_out_stride_1),
                         static_cast<int>((HPT - 1) * q_out_stride_1) + PE_DIM,
                         inv_qscale, pe_chunk, pe_lane && q_live);
    }

    // ================= K (only head 0, kv=1) =================
    // A scalar branch (SCC, not EXEC), so no WAR drain: the one-head direct
    // path no longer returns early for a padded token.
    if(head_base == 0 && slot_idx >= 0)
    {
        const float inv_kscale = 1.0f / (*k_scale);
        int64_t nope_base, rope_base;
        if constexpr(PAGE_SIZE > 0)
        {
            const int64_t block_idx    = slot_idx / PAGE_SIZE;
            const int64_t block_offset = slot_idx % PAGE_SIZE;
            nope_base = block_idx * block_stride + block_offset * KV_LORA;
            rope_base =
                block_idx * block_stride + (int64_t)PAGE_SIZE * KV_LORA + block_offset * PE_DIM;
        }
        else
        {
            int64_t block_idx, block_offset;
            if(slot_idx < (int64_t(1) << 31))
            {
                const uint32_t s = static_cast<uint32_t>(slot_idx);
                const uint32_t b = (__umulhi(s, bs_magic) + s) >> bs_shift;
                block_idx        = b;
                block_offset     = s - b * static_cast<uint32_t>(block_size);
            }
            else
            {
                block_idx    = slot_idx / block_size;
                block_offset = slot_idx % block_size;
            }
            const int64_t entry = block_idx * block_stride + block_offset * entry_stride;
            nope_base = entry + NOPE_OFF;
            rope_base = entry + PE_OFF;
        }
        auto out_buf = opus::make_gmem<cache_t>(kv_cache + nope_base, KV_LORA * sizeof(cache_t));
        if constexpr(K_EARLY)
        {
            out_buf.template store<VEC, out_vec_t>(cvt_vec(k_nope_in, inv_kscale, kv_cache),
                                                   threadIdx.x * VEC);
            rope_pe_emit(k_pe_op, kv_cache + rope_base, 0, PE_DIM, inv_kscale,
                         threadIdx.x % PE_CHUNKS, threadIdx.x < PE_CHUNKS);
        }
        else
        {
            // nope: vectorized static quant.
            const scalar_t* kv_c_row = kv_c + token_idx * kv_c_stride;
            auto in_buf = opus::make_gmem<scalar_t>(kv_c_row, KV_LORA * sizeof(scalar_t));
            in_vec_t vin = in_buf.template load<VEC>(threadIdx.x * VEC);
            out_buf.template store<VEC, out_vec_t>(cvt_vec(vin, inv_kscale, kv_cache),
                                                   threadIdx.x * VEC);

            // pe: RoPE then static quant.
            const scalar_t* k_pe_row = k_pe + token_idx * k_pe_stride;
            rope_pe_seg(k_pe_row, kv_cache + rope_base, inv_kscale);
        }
    }
}

  template <typename scalar_t, typename cache_t, typename query_t, bool IS_NEOX, bool is_nope_first>
  inline __device__ void rotary_embedding_kernel(
      const int64_t *__restrict__ positions,      // [batch_size, seq_len] or
                                                  // [num_tokens]
      const scalar_t *__restrict__ query,               // [batch_size, seq_len, num_heads,
                                                  // head_size] or [num_tokens, num_heads,
                                                  // head_size]
      const scalar_t *__restrict__ key,                 // [batch_size, seq_len, num_kv_heads,
                                                  // head_size] or [num_tokens, num_kv_heads,
                                                  // head_size]
      cache_t * __restrict__ kv_cache,
      query_t * __restrict__ q_out,
      const scalar_t *__restrict__ cos_cache,        // [max_position, rot_dim //2]
      const scalar_t *__restrict__ sin_cache,        // [max_position, rot_dim //2]
      const float inv_kscale,
      const float inv_qscale,                       //
      const int rot_dim, const int64_t q_pe_stride_0, const int64_t q_pe_stride_1, const int64_t key_stride,
      const int64_t q_out_stride_0, const int64_t q_out_stride_1, const int64_t kv_cache_offset,
      const int num_heads, const int num_kv_heads, const int head_size)
  {
    // Each thread block is responsible for one token.
    const int token_idx = blockIdx.x;
    int64_t pos = positions[token_idx];

    int64_t cos_sin_cache_offset = pos * rot_dim / 2;

    const scalar_t *cos_ptr = cos_cache + cos_sin_cache_offset;
    const scalar_t *sin_ptr = sin_cache + cos_sin_cache_offset;

    apply_rotary_embedding<scalar_t, cache_t, query_t, IS_NEOX, is_nope_first>(
        query, key, kv_cache, q_out, cos_ptr, sin_ptr, inv_kscale, inv_qscale,
        head_size, num_heads, num_kv_heads, rot_dim,
        token_idx, q_pe_stride_0, q_pe_stride_1, key_stride, q_out_stride_0, q_out_stride_1, kv_cache_offset);
  }
   
    template <typename scalar_t, typename cache_t, typename query_t, vllm::Fp8KVCacheDataType kv_dt, vllm::Fp8KVCacheDataType q_dt, bool is_neox, bool is_nope_first>
    __device__ void fuse_qk_rope_concat_and_cache_mla_kernel_opt(
        const scalar_t* __restrict__ q_nope,  // [num_tokens, num_heads, kv_lora_rank]
        const scalar_t* __restrict__ q_pe,    // [num_tokens, num_heads, pe_dim]
        const scalar_t* __restrict__ kv_c,    // [num_tokens, kv_lora_rank]
        const scalar_t* __restrict__ k_pe,    // [num_tokens, pe_dim]
        cache_t* __restrict__ kv_cache,       // [num_blocks, block_size, (qk_lora_rank + pe_dim)]
        query_t* __restrict__ q_out,          // [num_tokens, num_heads, kv_lora_rank + pe_dim]
        const int64_t* __restrict__ slot_mapping,  // [num_tokens]
        const int64_t* __restrict__ positions,     // [num_tokens]
        const scalar_t *__restrict__ cos_cache,        // [max_position, rot_dim //2]
        const scalar_t *__restrict__ sin_cache,        // [max_position, rot_dim //2]
        const int block_stride, const int entry_stride,
        const int q_nope_stride_0, const int q_nope_stride_1, 
        const int q_pe_stride_0,const int q_pe_stride_1,
        const int q_out_stride_0, const int q_out_stride_1, 
        const int num_heads,
        const int kv_c_stride, const int k_pe_stride,
        const int kv_lora_rank, const int pe_dim,
        const int block_size,
        const float* scale, const float* q_scale,
        const int max_position,
        const bool compute_all_q_rope
    ) {
      const int64_t token_idx = blockIdx.x;
      const int64_t slot_idx = slot_mapping[token_idx];
      // compute_all_q_rope == false (non-DCP default): restore the original
      // early-return so padded / cudagraph tokens (slot_idx < 0) skip Q RoPE +
      // q_out entirely. compute_all_q_rope == true (DCP): every rank needs all
      // queries after the head all-gather, so keep Q RoPE unconditional; only
      // the KV-cache writes are guarded by write_kv.
      if (!compute_all_q_rope && slot_idx < 0) {
        return;
      }
      const bool write_kv = (slot_idx >= 0);
      //concat
      const int64_t block_idx = slot_idx / block_size;
      const int64_t block_offset = slot_idx % block_size;
      const float inverted_kscale = 1.0f / *scale;
      const int64_t kv_cache_offset = block_idx * block_stride + block_offset * entry_stride;
      static constexpr int32_t vec_size_i = std::is_same_v<scalar_t, float> ? 4 : 8;
      static constexpr int32_t vec_size_o = vec_size_i;
      using opus_vec_i = opus::vector_t<scalar_t, vec_size_i>;
      using opus_vec_o = opus::vector_t<cache_t, vec_size_o>;
      static constexpr int32_t ooba_i = 4 / sizeof(scalar_t);
      static constexpr int32_t ooba_o = 4 / sizeof(cache_t);
      auto out_offset = block_idx * block_stride + block_offset * entry_stride;
      const int64_t qH_per_kH = num_heads;
      const int64_t kv_lora_dim = 512; // extend: = kv_lora_rank
      const int32_t oob_i             = (kv_lora_dim + ooba_i - 1) / ooba_i * ooba_i;
      const int32_t oob_o             = (kv_lora_dim + ooba_o - 1) / ooba_o * ooba_o;
      int32_t nope_offset = 0;
      if constexpr (!is_nope_first) {
        nope_offset = pe_dim;
      }
      auto const* ptr_i               = reinterpret_cast<scalar_t const*>(kv_c + token_idx * kv_c_stride);
      auto* ptr_o                     = reinterpret_cast<cache_t*>(kv_cache + out_offset + nope_offset);
      
      // FIX: oob_i is in elements, but make_gmem expects size in BYTES
      auto buffer_i = opus::make_gmem<scalar_t>(ptr_i, oob_i * sizeof(scalar_t));
      auto buffer_o = opus::make_gmem<cache_t>(ptr_o, oob_o * sizeof(cache_t));
      // Simple load and store for kv_lora_dim data
      const int32_t k_num_vecs       = (kv_lora_dim + vec_size_i - 1) / vec_size_i;
      opus_vec_i k_vec_cur;
      size_t vec_idx    = threadIdx.x;
      size_t vec_stride = 256;//blockDim.x;

      const float inverted_qscale = 1.0f / *q_scale;
      const int64_t head_size = kv_lora_dim + pe_dim;
      int64_t size = num_heads * kv_lora_dim;
      static constexpr int32_t q_ooba_o = 4 / sizeof(query_t);
      const int64_t q_input_span        = (num_heads - 1) * static_cast<int64_t>(q_nope_stride_1) + kv_lora_dim;
      const int64_t q_oob_i             = (q_input_span + ooba_i - 1) / ooba_i * ooba_i;
      const int32_t q_oob_o             = (num_heads * head_size + q_ooba_o - 1) / q_ooba_o * q_ooba_o;
      auto const* q_ptr_i               = reinterpret_cast<scalar_t const*>(q_nope + token_idx * q_nope_stride_0);

      auto* q_ptr_o                     = reinterpret_cast<query_t*>(q_out + q_out_stride_0 * token_idx);
      // Use opus::make_gmem instead of ck_tile::make_buffer_view, size in BYTES
      auto q_buffer_i = opus::make_gmem<scalar_t>(q_ptr_i, q_oob_i * sizeof(scalar_t));
      auto q_buffer_o = opus::make_gmem<query_t>(q_ptr_o, q_oob_o * sizeof(query_t));
      const int32_t num_vecs       = (size + vec_size_i - 1) / vec_size_i;
      size_t q_vec_idx    = threadIdx.x;
      using opus_vec_q = opus::vector_t<query_t, vec_size_o>;
      opus_vec_i vec_nxt;
      opus_vec_i vec_cur;
      const size_t kv_lora_vec = kv_lora_dim / vec_size_o;
      size_t q_head_idx = q_vec_idx / kv_lora_vec;
      size_t q_vec_dst_idx = q_vec_idx % kv_lora_vec;
      vec_cur = q_buffer_i.template load<vec_size_i>(
          q_head_idx * q_nope_stride_1 + q_vec_dst_idx * vec_size_i);
      
      // Load and store k vector (only threads < k_num_vecs need to work).
      // DCP: KV write only on the owning rank (write_kv).
      if (write_kv && vec_idx < k_num_vecs)
      {
        k_vec_cur = buffer_i.template load<vec_size_i>(vec_idx * vec_size_i);
        if constexpr (kv_dt == vllm::Fp8KVCacheDataType::kAuto) {
          buffer_o.template store<vec_size_o, opus_vec_o>(k_vec_cur, vec_idx * vec_size_o);
        } else {
          opus_vec_o vec_converted = aiter::scaled_cast<cache_t>(k_vec_cur, inverted_kscale);
          buffer_o.template store<vec_size_o, opus_vec_o>(vec_converted, vec_idx * vec_size_o);
        }
      }
      int64_t pos = positions[token_idx];
      // Clamp pos defensively: padded / cudagraph tokens (slot_idx < 0) may
      // carry a stale position that would index cos/sin caches out of bounds.
      pos = pos < 0 ? 0 : (pos >= max_position ? max_position - 1 : pos);
      int64_t cos_sin_cache_offset = pos * pe_dim / 2;

      const scalar_t *cos_ptr = cos_cache + cos_sin_cache_offset;
      const scalar_t *sin_ptr = sin_cache + cos_sin_cache_offset;

      const int embed_dim = 32;
      const int nq = num_heads * embed_dim;
      if constexpr (is_nope_first)
      {
        q_out += head_size - pe_dim;
      }
      for (; q_vec_idx < nq; q_vec_idx += vec_stride)
      {
          const size_t next_q_vec_idx = q_vec_idx + vec_stride;
          const size_t next_head_idx = next_q_vec_idx / kv_lora_vec;
          const size_t next_vec_dst_idx = next_q_vec_idx % kv_lora_vec;
          vec_nxt = q_buffer_i.template load<vec_size_i>(
              next_head_idx * q_nope_stride_1 + next_vec_dst_idx * vec_size_i);
          size_t cur_idx = q_vec_idx;
          size_t head_idx = cur_idx / kv_lora_vec;
          size_t vec_dst_idx = cur_idx % kv_lora_vec;
          const int rot_offset = cur_idx % embed_dim;
          int x_index, y_index;
          scalar_t cos, sin;
          // GPT-NeoX style rotary embedding.
          if constexpr (is_neox)
          {
            // GPT-NeoX style rotary embedding.
            x_index = rot_offset;
            y_index = embed_dim + rot_offset;
            cos = *(cos_ptr + x_index);
            sin = *(sin_ptr + x_index);
          }
          else
          {
            // GPT-J style rotary embedding.
            x_index = 2 * rot_offset;
            y_index = 2 * rot_offset + 1;
            cos = *(cos_ptr + x_index / 2);
            sin = *(sin_ptr + x_index / 2);
          }

          const int r_head_idx = q_vec_idx / embed_dim;
          const int64_t token_head_in = token_idx * q_pe_stride_0 + r_head_idx * q_pe_stride_1;
          const scalar_t* q_pe_in = q_pe + token_head_in;
          const scalar_t x = q_pe_in[x_index];
          const scalar_t y = q_pe_in[y_index];

          if constexpr (q_dt == vllm::Fp8KVCacheDataType::kAuto) {
              q_buffer_o.template store<vec_size_o, opus_vec_q>(vec_cur, (head_idx * q_out_stride_1) + vec_dst_idx * vec_size_o + nope_offset);
          } else {
              opus_vec_q vec_q_converted = aiter::scaled_cast<query_t>(vec_cur, inverted_qscale);
              q_buffer_o.template store<vec_size_o, opus_vec_q>(vec_q_converted, (head_idx * q_out_stride_1) + vec_dst_idx * vec_size_o + nope_offset);
          }
          vec_cur = vec_nxt;
          const int64_t token_head = token_idx * q_out_stride_0 + r_head_idx * q_out_stride_1;
          query_t* q_out_rope = q_out + token_head;
          float f32_x = static_cast<float>(x);
          float f32_y = static_cast<float>(y);
          float f32_cos = static_cast<float>(cos);
          float f32_sin = static_cast<float>(sin);
          if constexpr (std::is_same_v<query_t, opus::fp8_t>) {
              q_out_rope[x_index] = opus::cast<opus::fp8_t>(
                      (f32_x * f32_cos - f32_y * f32_sin) * inverted_qscale);
              q_out_rope[y_index] = opus::cast<opus::fp8_t>(
                      (f32_y * f32_cos + f32_x * f32_sin) * inverted_qscale);
          } else {
              q_out_rope[x_index] = static_cast<query_t>(f32_x * f32_cos - f32_y * f32_sin);
              q_out_rope[y_index] = static_cast<query_t>(f32_y * f32_cos + f32_x * f32_sin);
          }
      }

      for (q_vec_idx += vec_stride; q_vec_idx < num_vecs; q_vec_idx += vec_stride)
      {
          const size_t next_head_idx = q_vec_idx / kv_lora_vec;
          const size_t next_vec_dst_idx = q_vec_idx % kv_lora_vec;
          vec_nxt = q_buffer_i.template load<vec_size_i>(
              next_head_idx * q_nope_stride_1 + next_vec_dst_idx * vec_size_i);
          size_t head_idx = (q_vec_idx - vec_stride)  / kv_lora_vec;
          size_t vec_dst_idx = (q_vec_idx - vec_stride) % kv_lora_vec;
          if constexpr (q_dt == vllm::Fp8KVCacheDataType::kAuto) {
              q_buffer_o.template store<vec_size_o, opus_vec_q>(vec_cur, (head_idx * q_out_stride_1) + vec_dst_idx * vec_size_o + nope_offset);
          } else {
              opus_vec_q vec_q_converted = aiter::scaled_cast<query_t>(vec_cur, inverted_qscale);
              q_buffer_o.template store<vec_size_o, opus_vec_q>(vec_q_converted, (head_idx * q_out_stride_1) + vec_dst_idx * vec_size_o + nope_offset);
          }
          vec_cur = vec_nxt;
      }
      if (q_vec_idx - vec_stride < num_vecs)
      {
          size_t head_idx = (q_vec_idx - vec_stride) / kv_lora_vec;
          size_t vec_dst_idx = (q_vec_idx - vec_stride) % kv_lora_vec;
          if constexpr (q_dt == vllm::Fp8KVCacheDataType::kAuto) {
              q_buffer_o.template store<vec_size_o, opus_vec_q>(vec_cur, (head_idx * q_out_stride_1) + vec_dst_idx * vec_size_o + nope_offset);
          } else {
              opus_vec_q vec_q_converted = aiter::scaled_cast<query_t>(vec_cur, inverted_qscale);
              q_buffer_o.template store<vec_size_o, opus_vec_q>(vec_q_converted, (head_idx * q_out_stride_1) + vec_dst_idx * vec_size_o + nope_offset);
          }
      }
    // apply rotary (k_pe RoPE write into KV cache; DCP: owning rank only)
    const int nk =  embed_dim;
    if (write_kv && threadIdx.x < nk)
    {
      if constexpr (is_nope_first)
      {
        kv_cache += head_size - pe_dim;
      }
      //const int head_idx = i / embed_dim;
      const int64_t token_head_in = token_idx * k_pe_stride;// + head_idx * embed_dim;
      const int64_t token_head = kv_cache_offset;
      const int rot_offset = threadIdx.x;// % embed_dim;
      apply_token_rotary_embedding<scalar_t, cache_t, is_neox>(
          k_pe + token_head_in, kv_cache + token_head, cos_ptr, sin_ptr, inverted_kscale, rot_offset, embed_dim);
    }
    }

    template <typename scalar_t, typename cache_t, typename query_t, vllm::Fp8KVCacheDataType kv_dt, vllm::Fp8KVCacheDataType q_dt>
    __global__ void fuse_qk_rope_concat_and_cache_mla_kernel(
        const scalar_t* __restrict__ q_nope,  // [num_tokens, num_heads, kv_lora_rank]
        const scalar_t* __restrict__ q_pe,    // [num_tokens, num_heads, pe_dim]
        const scalar_t* __restrict__ kv_c,    // [num_tokens, kv_lora_rank]
        const scalar_t* __restrict__ k_pe,    // [num_tokens, pe_dim]
        cache_t* __restrict__ kv_cache,       // [num_blocks, block_size, (qk_lora_rank + pe_dim)]
        query_t* __restrict__ q_out,          // [num_tokens, num_heads, kv_lora_rank + pe_dim]
        const int64_t* __restrict__ slot_mapping,  // [num_tokens]
        const int64_t* __restrict__ positions,     // [num_tokens]
        const scalar_t *__restrict__ cos_cache,        // [max_position, rot_dim //2]
        const scalar_t *__restrict__ sin_cache,        // [max_position, rot_dim //2]
        const int block_stride, const int entry_stride,
        const int q_nope_stride_0, const int q_nope_stride_1,
        const int q_pe_stride_0, const int q_pe_stride_1,
        const int q_out_stride_0, const int q_out_stride_1, 
        const int num_heads,
        const int kv_c_stride, const int k_pe_stride,
        const int kv_lora_rank, const int pe_dim,
        const int block_size,
        const float* scale, const float* q_scale,
        bool is_neox, bool is_nope_first, const int max_position,
        const bool compute_all_q_rope
    ) {
      if (is_neox && is_nope_first) {
        fuse_qk_rope_concat_and_cache_mla_kernel_opt<scalar_t,cache_t,query_t, kv_dt, q_dt, true, true>(q_nope, q_pe, kv_c, k_pe, kv_cache, q_out, slot_mapping, positions,
                                           cos_cache, sin_cache, block_stride, entry_stride, q_nope_stride_0, q_nope_stride_1, q_pe_stride_0, q_pe_stride_1, q_out_stride_0,
                                           q_out_stride_1, num_heads, kv_c_stride, k_pe_stride, kv_lora_rank, pe_dim, block_size, scale, q_scale, max_position, compute_all_q_rope);
      } else if (is_neox && !is_nope_first) {
        fuse_qk_rope_concat_and_cache_mla_kernel_opt<scalar_t,cache_t,query_t, kv_dt, q_dt, true, false>(q_nope, q_pe, kv_c, k_pe, kv_cache, q_out, slot_mapping, positions,
                                            cos_cache, sin_cache, block_stride, entry_stride, q_nope_stride_0, q_nope_stride_1, q_pe_stride_0, q_pe_stride_1, q_out_stride_0,
                                            q_out_stride_1, num_heads, kv_c_stride, k_pe_stride, kv_lora_rank, pe_dim, block_size, scale, q_scale, max_position, compute_all_q_rope);
      } else if (!is_neox && is_nope_first) {
        fuse_qk_rope_concat_and_cache_mla_kernel_opt<scalar_t,cache_t,query_t, kv_dt, q_dt, false, true>(q_nope, q_pe, kv_c, k_pe, kv_cache, q_out, slot_mapping, positions,
                                            cos_cache, sin_cache, block_stride, entry_stride, q_nope_stride_0, q_nope_stride_1, q_pe_stride_0, q_pe_stride_1, q_out_stride_0,
                                            q_out_stride_1, num_heads, kv_c_stride, k_pe_stride, kv_lora_rank, pe_dim, block_size, scale, q_scale, max_position, compute_all_q_rope);
      } else {
        fuse_qk_rope_concat_and_cache_mla_kernel_opt<scalar_t,cache_t,query_t, kv_dt, q_dt, false, false>(q_nope, q_pe, kv_c, k_pe, kv_cache, q_out, slot_mapping, positions,
                                            cos_cache, sin_cache, block_stride, entry_stride, q_nope_stride_0, q_nope_stride_1, q_pe_stride_0, q_pe_stride_1, q_out_stride_0,
                                            q_out_stride_1, num_heads, kv_c_stride, k_pe_stride, kv_lora_rank, pe_dim, block_size, scale, q_scale, max_position, compute_all_q_rope);
      }
    }

    template <typename scalar_t, typename cache_t, typename query_t, vllm::Fp8KVCacheDataType kv_dt, vllm::Fp8KVCacheDataType q_dt, bool is_neox, bool is_nope_first>
    __device__ void fuse_qk_rope_concat_and_cache_mla_kernel_prefill_opt(
        const scalar_t* __restrict__ q_nope,  // [num_tokens, num_heads, kv_lora_rank]
        const scalar_t* __restrict__ q_pe,    // [num_tokens, num_heads, pe_dim]
        const scalar_t* __restrict__ kv_c,    // [num_tokens, kv_lora_rank]
        const scalar_t* __restrict__ k_pe,    // [num_tokens, pe_dim]
        cache_t* __restrict__ kv_cache,       // [num_blocks, block_size, (qk_lora_rank + pe_dim)]
        query_t* __restrict__ q_out,          // [num_tokens, num_heads, kv_lora_rank + pe_dim]
        const int64_t* __restrict__ slot_mapping,  // [num_tokens]
        const int64_t* __restrict__ positions,     // [num_tokens]
        const scalar_t *__restrict__ cos_cache,        // [max_position, rot_dim //2]
        const scalar_t *__restrict__ sin_cache,        // [max_position, rot_dim //2]
        const int block_stride, const int entry_stride, const int kv_cache_stride_h,
        const int q_nope_stride_0, const int q_nope_stride_1, 
        const int q_pe_stride_0,const int q_pe_stride_1,
        const int q_out_stride_0, const int q_out_stride_1, 
        const int num_heads, const int num_kv_heads,
        const int kv_c_stride_0, const int kv_c_stride_1, 
        const int k_pe_stride_0, const int k_pe_stride_1,
        const int kv_lora_rank, const int pe_dim,
        const int block_size,
        const float* scale, const float* q_scale
    ) {
      const int64_t token_idx = blockIdx.x;
      const int64_t slot_idx = slot_mapping[token_idx];
      // NOTE: slot_idx can be -1 if the token is padded
      if (slot_idx < 0) {
        return;
      }
      //concat
      const int64_t block_idx = slot_idx / block_size;
      const int64_t block_offset = slot_idx % block_size;
      const float inverted_kscale = 1.0f / *scale;
      const int64_t kv_cache_offset = block_idx * block_stride + block_offset * entry_stride;
      static constexpr int32_t vec_size_i = std::is_same_v<scalar_t, float> ? 4 : 8;
      static constexpr int32_t vec_size_o = vec_size_i;
      using opus_vec_i = opus::vector_t<scalar_t, vec_size_i>;
      using opus_vec_o = opus::vector_t<cache_t, vec_size_o>;
      static constexpr int32_t ooba_i = 4 / sizeof(scalar_t);
      static constexpr int32_t ooba_o = 4 / sizeof(cache_t);
      const int64_t qH_per_kH = num_heads;
      const int64_t kv_lora_dim = 512; // extend: = kv_lora_rank
      const int64_t head_size = kv_lora_dim + pe_dim;
      const int32_t oob_i             = (kv_lora_dim * num_kv_heads + ooba_i - 1) / ooba_i * ooba_i;
      const int32_t oob_o             = (head_size * num_kv_heads + ooba_o - 1) / ooba_o * ooba_o;
      int32_t nope_offset = 0;
      if constexpr (!is_nope_first) {
        nope_offset = pe_dim;
      }
      auto const* ptr_i               = reinterpret_cast<scalar_t const*>(kv_c + token_idx * kv_c_stride_0);
      auto* ptr_o                     = reinterpret_cast<cache_t*>(kv_cache + kv_cache_offset + nope_offset);
      
      // FIX: oob_i is in elements, but make_gmem expects size in BYTES
      auto buffer_i = opus::make_gmem<scalar_t>(ptr_i, oob_i * sizeof(scalar_t));
      auto buffer_o = opus::make_gmem<cache_t>(ptr_o, oob_o * sizeof(cache_t));
      
      const float inverted_qscale = 1.0f / *q_scale;
      const int32_t size = num_heads * kv_lora_dim;
      static constexpr int32_t q_ooba_o = 4 / sizeof(query_t);
      const int64_t q_input_span = (num_heads - 1) * static_cast<int64_t>(q_nope_stride_1) + kv_lora_dim;
      const int64_t q_oob_i = (q_input_span + ooba_i - 1) / ooba_i * ooba_i;
      const int32_t q_oob_o = (num_heads * head_size + q_ooba_o - 1) / q_ooba_o * q_ooba_o;
      
      // Use opus::make_gmem for Q buffers, size in BYTES
      auto q_buffer_i = opus::make_gmem<scalar_t>(q_nope + token_idx * q_nope_stride_0, q_oob_i * sizeof(scalar_t));
      auto q_buffer_o = opus::make_gmem<query_t>(q_out + q_out_stride_0 * token_idx + nope_offset, q_oob_o * sizeof(query_t));
      
      const int32_t num_vecs = (size + vec_size_i - 1) / vec_size_i;
      const int32_t num_kv_vecs = (kv_lora_dim * num_kv_heads + vec_size_i - 1) / vec_size_i;
      const uint32_t kv_lora_vec = 64;//kv_lora_dim / vec_size_o;
      
      using opus_vec_q = opus::vector_t<query_t, vec_size_o>;
      // Reduced vector registers: only use two vectors total (reuse for both Q and K)
      opus_vec_i vec_cur, vec_nxt;
      uint32_t vec_idx = threadIdx.x;
      constexpr uint32_t vec_stride = 256;
      
      // Prepare RoPE cos/sin pointers (needed for both Q and K RoPE)
      const int32_t cos_sin_cache_offset = positions[token_idx] * (pe_dim >> 1);
      const scalar_t *cos_ptr = cos_cache + cos_sin_cache_offset;
      const scalar_t *sin_ptr = sin_cache + cos_sin_cache_offset;
      constexpr int32_t embed_dim = 32;
      
      // Phase 1: Process Q and K nope together for first num_kv_heads
      // Calculate head indices once and maintain them as loop variables
      uint32_t head_idx = vec_idx / kv_lora_vec;
      uint32_t in_head_idx = vec_idx % kv_lora_vec;
      bool has_data = (vec_idx < num_kv_vecs);
      
      // Load first vectors if thread is in range
      if (has_data) {
        // Calculate offset considering stride(1) for non-contiguous tensors
        uint32_t kv_c_offset = head_idx * kv_c_stride_1 + in_head_idx * vec_size_i;
        size_t q_nope_offset = head_idx * static_cast<size_t>(q_nope_stride_1) + in_head_idx * vec_size_i;
        vec_cur = buffer_i.template load<vec_size_i>(kv_c_offset);  // K data in vec_cur initially
        vec_nxt = q_buffer_i.template load<vec_size_i>(q_nope_offset); // Q data in vec_nxt
      }
      
      // Double buffering loop: process Q and K nope together
      for (uint32_t next_idx = vec_idx + vec_stride; next_idx < num_kv_vecs; next_idx += vec_stride)
      {
        // Calculate store offsets using current head_idx and in_head_idx
        uint32_t store_offset = head_idx * kv_cache_stride_h + in_head_idx * vec_size_o;
        uint32_t q_store_offset = head_idx * q_out_stride_1 + in_head_idx * vec_size_o;
        
        // Calculate next indices for prefetch
        uint32_t next_head_idx = next_idx / kv_lora_vec;
        uint32_t next_in_head_idx = next_idx % kv_lora_vec;
        uint32_t next_kv_c_offset = next_head_idx * kv_c_stride_1 + next_in_head_idx * vec_size_i;
        size_t next_q_nope_offset = next_head_idx * static_cast<size_t>(q_nope_stride_1) + next_in_head_idx * vec_size_i;
        opus_vec_i k_nxt = buffer_i.template load<vec_size_i>(next_kv_c_offset);
        opus_vec_i q_nxt = q_buffer_i.template load<vec_size_i>(next_q_nope_offset);
        
        // Store K data (vec_cur holds K)
        if constexpr (kv_dt == vllm::Fp8KVCacheDataType::kAuto) {
          buffer_o.template store<vec_size_o>(vec_cur, store_offset);
        } else {
          buffer_o.template store<vec_size_o>(aiter::scaled_cast<cache_t>(vec_cur, inverted_kscale), store_offset);
        }
        
        // Store Q data (vec_nxt holds Q)
        if constexpr (q_dt == vllm::Fp8KVCacheDataType::kAuto) {
          q_buffer_o.template store<vec_size_o>(vec_nxt, q_store_offset);
        } else {
          q_buffer_o.template store<vec_size_o>(aiter::scaled_cast<query_t>(vec_nxt, inverted_qscale), q_store_offset);
        }
        
        // Swap: next K goes to vec_cur, next Q goes to vec_nxt
        vec_cur = k_nxt;
        vec_nxt = q_nxt;
        
        // Update loop variables for next iteration
        vec_idx = next_idx;
        head_idx = next_head_idx;
        in_head_idx = next_in_head_idx;
      }
      
      // Store last vectors if we loaded data (use maintained head_idx and in_head_idx)
      if (has_data) {
        // Store last K
        if constexpr (kv_dt == vllm::Fp8KVCacheDataType::kAuto) {
          buffer_o.template store<vec_size_o>(vec_cur, head_idx * kv_cache_stride_h + in_head_idx * vec_size_o);
        } else {
          buffer_o.template store<vec_size_o>(aiter::scaled_cast<cache_t>(vec_cur, inverted_kscale), 
              head_idx * kv_cache_stride_h + in_head_idx * vec_size_o);
        }
        
        // Store last Q
        if constexpr (q_dt == vllm::Fp8KVCacheDataType::kAuto) {
          q_buffer_o.template store<vec_size_o>(vec_nxt, head_idx * q_out_stride_1 + in_head_idx * vec_size_o);
        } else {
          q_buffer_o.template store<vec_size_o>(aiter::scaled_cast<query_t>(vec_nxt, inverted_qscale), 
              head_idx * q_out_stride_1 + in_head_idx * vec_size_o);
        }
      }
      
      // Phase 2: Process remaining Q nope only
      // Start from the next vector after num_kv_vecs
      uint32_t q_vec_idx = num_kv_vecs + threadIdx.x;
      uint32_t q_head_idx = q_vec_idx / kv_lora_vec;
      uint32_t q_in_head_idx = q_vec_idx % kv_lora_vec;
      
      // Load first Q vector if in range (num_heads > num_kv_heads case)
      if (q_vec_idx < num_vecs) {
        size_t q_nope_offset = q_head_idx * static_cast<size_t>(q_nope_stride_1) + q_in_head_idx * vec_size_i;
        vec_cur = q_buffer_i.template load<vec_size_i>(q_nope_offset);
      }
      
      // Double buffering loop for remaining Q
      for (uint32_t next_q_idx = q_vec_idx + vec_stride; next_q_idx < num_vecs; next_q_idx += vec_stride)
      {
        // Calculate next indices for prefetch
        uint32_t next_head_idx = next_q_idx / kv_lora_vec;
        uint32_t next_in_head_idx = next_q_idx % kv_lora_vec;
        size_t next_q_nope_offset = next_head_idx * static_cast<size_t>(q_nope_stride_1) + next_in_head_idx * vec_size_i;
        vec_nxt = q_buffer_i.template load<vec_size_i>(next_q_nope_offset);
        
        // Calculate store offset using current q_head_idx and q_in_head_idx
        uint32_t store_offset = q_head_idx * q_out_stride_1 + q_in_head_idx * vec_size_o;
        
        if constexpr (q_dt == vllm::Fp8KVCacheDataType::kAuto) {
          q_buffer_o.template store<vec_size_o>(vec_cur, store_offset);
        } else {
          q_buffer_o.template store<vec_size_o>(aiter::scaled_cast<query_t>(vec_cur, inverted_qscale), store_offset);
        }
        
        // Update loop variables
        vec_cur = vec_nxt;
        q_vec_idx = next_q_idx;
        q_head_idx = next_head_idx;
        q_in_head_idx = next_in_head_idx;
      }
      
      // Store last Q vector if loaded (use maintained q_head_idx and q_in_head_idx)
      if (q_vec_idx < num_vecs) {
        uint32_t store_offset = q_head_idx * q_out_stride_1 + q_in_head_idx * vec_size_o;
        if constexpr (q_dt == vllm::Fp8KVCacheDataType::kAuto) {
          q_buffer_o.template store<vec_size_o>(vec_cur, store_offset);
        } else {
          q_buffer_o.template store<vec_size_o>(aiter::scaled_cast<query_t>(vec_cur, inverted_qscale), store_offset);
        }
      }
    
      // ============ RoPE Phase ============
      // Adjust base pointers for RoPE region
      if constexpr (is_nope_first) {
        q_out += head_size - pe_dim;
        kv_cache += head_size - pe_dim;
      }
      
      const int32_t nq = num_heads * embed_dim;
      const int32_t nk = num_kv_heads * embed_dim;
      const int32_t token_q_base = token_idx * q_pe_stride_0;
      const int32_t token_k_base = token_idx * k_pe_stride_0;
      const int32_t token_q_out_base = token_idx * q_out_stride_0;
      
      // Phase 1: Process Q and K RoPE together
      for (uint32_t r_idx = threadIdx.x; r_idx < nk; r_idx += vec_stride)
      {
          const uint32_t rope_head = r_idx / embed_dim;
          const uint32_t rot_off = r_idx % embed_dim;
          
          // Calculate cos/sin indices and load once
          uint32_t x_idx, y_idx;
          float f32_cos, f32_sin;
          if constexpr (is_neox) {
            x_idx = rot_off;
            y_idx = embed_dim + rot_off;
            f32_cos = static_cast<float>(*(cos_ptr + x_idx));
            f32_sin = static_cast<float>(*(sin_ptr + x_idx));
          } else {
            x_idx = rot_off << 1;  // *2
            y_idx = x_idx + 1;
            f32_cos = static_cast<float>(*(cos_ptr + (x_idx >> 1)));
            f32_sin = static_cast<float>(*(sin_ptr + (x_idx >> 1)));
          }
          
          // K RoPE: load, compute, store
          const scalar_t* k_in = k_pe + token_k_base + rope_head * k_pe_stride_1;
          cache_t* k_out = kv_cache + kv_cache_offset + rope_head * kv_cache_stride_h;
          
          float kx = static_cast<float>(k_in[x_idx]);
          float ky = static_cast<float>(k_in[y_idx]);
          float k_rot_x = kx * f32_cos - ky * f32_sin;
          float k_rot_y = ky * f32_cos + kx * f32_sin;
          
          if constexpr (std::is_same_v<cache_t, opus::fp8_t>) {
            k_out[x_idx] = opus::cast<opus::fp8_t>(k_rot_x * inverted_kscale);
            k_out[y_idx] = opus::cast<opus::fp8_t>(k_rot_y * inverted_kscale);
          } else {
            k_out[x_idx] = static_cast<cache_t>(k_rot_x);
            k_out[y_idx] = static_cast<cache_t>(k_rot_y);
          }
          
          // Q RoPE: load, compute, store (same head index in this range)
          const scalar_t* q_in = q_pe + token_q_base + rope_head * q_pe_stride_1;
          query_t* q_out_ptr = q_out + token_q_out_base + rope_head * q_out_stride_1;
          
          float qx = static_cast<float>(q_in[x_idx]);
          float qy = static_cast<float>(q_in[y_idx]);
          float q_rot_x = qx * f32_cos - qy * f32_sin;
          float q_rot_y = qy * f32_cos + qx * f32_sin;
          
          if constexpr (std::is_same_v<query_t, opus::fp8_t>) {
            q_out_ptr[x_idx] = opus::cast<opus::fp8_t>(q_rot_x * inverted_qscale);
            q_out_ptr[y_idx] = opus::cast<opus::fp8_t>(q_rot_y * inverted_qscale);
          } else {
            q_out_ptr[x_idx] = static_cast<query_t>(q_rot_x);
            q_out_ptr[y_idx] = static_cast<query_t>(q_rot_y);
          }
      }
      
      // Phase 2: Process remaining Q RoPE only (if num_heads > num_kv_heads)
      for (uint32_t r_idx = nk + threadIdx.x; r_idx < nq; r_idx += vec_stride)
      {
          const uint32_t rope_head = r_idx / embed_dim;
          const uint32_t rot_off = r_idx % embed_dim;
          
          // Calculate cos/sin indices and load
          uint32_t x_idx, y_idx;
          float f32_cos, f32_sin;
          if constexpr (is_neox) {
            x_idx = rot_off;
            y_idx = embed_dim + rot_off;
            f32_cos = static_cast<float>(*(cos_ptr + x_idx));
            f32_sin = static_cast<float>(*(sin_ptr + x_idx));
          } else {
            x_idx = rot_off << 1;
            y_idx = x_idx + 1;
            f32_cos = static_cast<float>(*(cos_ptr + (x_idx >> 1)));
            f32_sin = static_cast<float>(*(sin_ptr + (x_idx >> 1)));
          }
          
          // Q RoPE: load, compute, store
          const scalar_t* q_in = q_pe + token_q_base + rope_head * q_pe_stride_1;
          query_t* q_out_ptr = q_out + token_q_out_base + rope_head * q_out_stride_1;
          
          float qx = static_cast<float>(q_in[x_idx]);
          float qy = static_cast<float>(q_in[y_idx]);
          
          if constexpr (std::is_same_v<query_t, opus::fp8_t>) {
            q_out_ptr[x_idx] = opus::cast<opus::fp8_t>((qx * f32_cos - qy * f32_sin) * inverted_qscale);
            q_out_ptr[y_idx] = opus::cast<opus::fp8_t>((qy * f32_cos + qx * f32_sin) * inverted_qscale);
          } else {
            q_out_ptr[x_idx] = static_cast<query_t>(qx * f32_cos - qy * f32_sin);
            q_out_ptr[y_idx] = static_cast<query_t>(qy * f32_cos + qx * f32_sin);
          }
      }
    }

    // General version with kv_lora_dim and embed_dim as parameters
    template <typename scalar_t, typename cache_t, typename query_t, vllm::Fp8KVCacheDataType kv_dt, vllm::Fp8KVCacheDataType q_dt, bool is_neox, bool is_nope_first>
    __device__ void fuse_qk_rope_concat_and_cache_mla_kernel_general_kernel(
        const scalar_t* __restrict__ q_nope,  // [num_tokens, num_heads, kv_lora_rank]
        const scalar_t* __restrict__ q_pe,    // [num_tokens, num_heads, pe_dim]
        const scalar_t* __restrict__ kv_c,    // [num_tokens, kv_lora_rank]
        const scalar_t* __restrict__ k_pe,    // [num_tokens, pe_dim]
        cache_t* __restrict__ kv_cache,       // [num_blocks, block_size, (qk_lora_rank + pe_dim)]
        query_t* __restrict__ q_out,          // [num_tokens, num_heads, kv_lora_rank + pe_dim]
        const int64_t* __restrict__ slot_mapping,  // [num_tokens]
        const int64_t* __restrict__ positions,     // [num_tokens]
        const scalar_t *__restrict__ cos_cache,        // [max_position, rot_dim //2]
        const scalar_t *__restrict__ sin_cache,        // [max_position, rot_dim //2]
        const int block_stride, const int entry_stride, const int kv_cache_stride_h,
        const int q_nope_stride_0, const int q_nope_stride_1, 
        const int q_pe_stride_0,const int q_pe_stride_1,
        const int q_out_stride_0, const int q_out_stride_1, 
        const int num_heads, const int num_kv_heads,
        const int kv_c_stride_0, const int kv_c_stride_1, 
        const int k_pe_stride_0, const int k_pe_stride_1,
        const int kv_lora_rank, const int pe_dim,
        const int block_size,
        const float* scale, const float* q_scale
    ) {
      const int64_t token_idx = blockIdx.x;
      const int64_t slot_idx = slot_mapping[token_idx];
      // NOTE: slot_idx can be -1 if the token is padded
      if (slot_idx < 0) {
        return;
      }
      //concat
      const int64_t block_idx = slot_idx / block_size;
      const int64_t block_offset = slot_idx % block_size;
      const float inverted_kscale = 1.0f / *scale;
      const int64_t kv_cache_offset = block_idx * block_stride + block_offset * entry_stride;
      
      static constexpr int32_t vec_size_i = std::is_same_v<scalar_t, float> ? 4 : 8;
      static constexpr int32_t vec_size_o = vec_size_i;
      using opus_vec_i = opus::vector_t<scalar_t, vec_size_i>;
      using opus_vec_o = opus::vector_t<cache_t, vec_size_o>;
      static constexpr int32_t ooba_i = 4 / sizeof(scalar_t);
      static constexpr int32_t ooba_o = 4 / sizeof(cache_t);
      
      // Use kv_lora_rank parameter instead of hardcoded 512
      const int64_t kv_lora_dim = kv_lora_rank;
      const int64_t head_size = kv_lora_dim + pe_dim;
      const int32_t oob_i = (kv_lora_dim * num_kv_heads + ooba_i - 1) / ooba_i * ooba_i;
      const int32_t oob_o = (head_size * num_kv_heads + ooba_o - 1) / ooba_o * ooba_o;
      
      int32_t nope_offset = 0;
      if constexpr (!is_nope_first) {
        nope_offset = pe_dim;
      }
      
      auto const* ptr_i = reinterpret_cast<scalar_t const*>(kv_c + token_idx * kv_c_stride_0);
      auto* ptr_o = reinterpret_cast<cache_t*>(kv_cache + kv_cache_offset + nope_offset);
      
      auto buffer_i = opus::make_gmem<scalar_t>(ptr_i, oob_i * sizeof(scalar_t));
      auto buffer_o = opus::make_gmem<cache_t>(ptr_o, oob_o * sizeof(cache_t));
      
      const float inverted_qscale = 1.0f / *q_scale;
      const int32_t size = num_heads * kv_lora_dim;
      static constexpr int32_t q_ooba_o = 4 / sizeof(query_t);
      const int64_t q_input_span = (num_heads - 1) * static_cast<int64_t>(q_nope_stride_1) + kv_lora_dim;
      const int64_t q_oob_i = (q_input_span + ooba_i - 1) / ooba_i * ooba_i;
      const int32_t q_oob_o = (num_heads * head_size + q_ooba_o - 1) / q_ooba_o * q_ooba_o;
      
      auto q_buffer_i = opus::make_gmem<scalar_t>(q_nope + token_idx * q_nope_stride_0, q_oob_i * sizeof(scalar_t));
      auto q_buffer_o = opus::make_gmem<query_t>(q_out + q_out_stride_0 * token_idx + nope_offset, q_oob_o * sizeof(query_t));
      
      const int32_t num_vecs = (size + vec_size_i - 1) / vec_size_i;
      const int32_t num_kv_vecs = (kv_lora_dim * num_kv_heads + vec_size_i - 1) / vec_size_i;
      const uint32_t kv_lora_vec = (kv_lora_dim + vec_size_o - 1) / vec_size_o;
      
      using opus_vec_q = opus::vector_t<query_t, vec_size_o>;
      opus_vec_i vec_cur, vec_nxt;
      uint32_t vec_idx = threadIdx.x;
      const uint32_t vec_stride = blockDim.x;  // Use actual block size instead of hardcoded 256
      
      // Prepare RoPE cos/sin pointers
      const int32_t cos_sin_cache_offset = positions[token_idx] * (pe_dim >> 1);
      const scalar_t *cos_ptr = cos_cache + cos_sin_cache_offset;
      const scalar_t *sin_ptr = sin_cache + cos_sin_cache_offset;
      
      // Use rope_dim parameter instead of hardcoded embed_dim
      const int32_t embed_dim = pe_dim / 2;  // rope_dim = 64 means embed_dim = 32
      
      // ============ Nope Phase ============
      // Phase 1: Process Q and K nope together for first num_kv_heads
      // Calculate head indices once and maintain them as loop variables
      uint32_t head_idx = vec_idx / kv_lora_vec;
      uint32_t in_head_idx = vec_idx % kv_lora_vec;
      bool has_data = (vec_idx < num_kv_vecs);
      
      if (has_data) {
        // Calculate offset considering stride(1) for non-contiguous tensors
        uint32_t kv_c_offset = head_idx * kv_c_stride_1 + in_head_idx * vec_size_i;
        size_t q_nope_offset = head_idx * static_cast<size_t>(q_nope_stride_1) + in_head_idx * vec_size_i;
        vec_cur = buffer_i.template load<vec_size_i>(kv_c_offset);  // K data
        vec_nxt = q_buffer_i.template load<vec_size_i>(q_nope_offset); // Q data
      }
      
      // Double buffering loop: process Q and K nope together
      for (uint32_t next_idx = vec_idx + vec_stride; next_idx < num_kv_vecs; next_idx += vec_stride)
      {
        // Calculate store offsets using current head_idx and in_head_idx
        uint32_t store_offset = head_idx * kv_cache_stride_h + in_head_idx * vec_size_o;
        uint32_t q_store_offset = head_idx * q_out_stride_1 + in_head_idx * vec_size_o;
        
        // Calculate next indices for prefetch
        uint32_t next_head_idx = next_idx / kv_lora_vec;
        uint32_t next_in_head_idx = next_idx % kv_lora_vec;
        uint32_t next_kv_c_offset = next_head_idx * kv_c_stride_1 + next_in_head_idx * vec_size_i;
        size_t next_q_nope_offset = next_head_idx * static_cast<size_t>(q_nope_stride_1) + next_in_head_idx * vec_size_i;
        opus_vec_i k_nxt = buffer_i.template load<vec_size_i>(next_kv_c_offset);
        opus_vec_i q_nxt = q_buffer_i.template load<vec_size_i>(next_q_nope_offset);
        
        // Store K data
        if constexpr (kv_dt == vllm::Fp8KVCacheDataType::kAuto) {
          buffer_o.template store<vec_size_o>(vec_cur, store_offset);
        } else {
          buffer_o.template store<vec_size_o>(aiter::scaled_cast<cache_t>(vec_cur, inverted_kscale), store_offset);
        }
        
        // Store Q data
        if constexpr (q_dt == vllm::Fp8KVCacheDataType::kAuto) {
          q_buffer_o.template store<vec_size_o>(vec_nxt, q_store_offset);
        } else {
          q_buffer_o.template store<vec_size_o>(aiter::scaled_cast<query_t>(vec_nxt, inverted_qscale), q_store_offset);
        }
        
        // Update loop variables
        vec_cur = k_nxt;
        vec_nxt = q_nxt;
        vec_idx = next_idx;
        head_idx = next_head_idx;
        in_head_idx = next_in_head_idx;
      }
      
      // Store last vectors (use maintained head_idx and in_head_idx)
      if (has_data) {
        if constexpr (kv_dt == vllm::Fp8KVCacheDataType::kAuto) {
          buffer_o.template store<vec_size_o>(vec_cur, head_idx * kv_cache_stride_h + in_head_idx * vec_size_o);
        } else {
          buffer_o.template store<vec_size_o>(aiter::scaled_cast<cache_t>(vec_cur, inverted_kscale), 
              head_idx * kv_cache_stride_h + in_head_idx * vec_size_o);
        }
        
        if constexpr (q_dt == vllm::Fp8KVCacheDataType::kAuto) {
          q_buffer_o.template store<vec_size_o>(vec_nxt, head_idx * q_out_stride_1 + in_head_idx * vec_size_o);
        } else {
          q_buffer_o.template store<vec_size_o>(aiter::scaled_cast<query_t>(vec_nxt, inverted_qscale), 
              head_idx * q_out_stride_1 + in_head_idx * vec_size_o);
        }
      }
      
      // Phase 2: Process remaining Q nope only (when num_heads > num_kv_heads)
      uint32_t q_vec_idx = num_kv_vecs + threadIdx.x;
      uint32_t q_head_idx = q_vec_idx / kv_lora_vec;
      uint32_t q_in_head_idx = q_vec_idx % kv_lora_vec;
      
      if (q_vec_idx < num_vecs) {
        size_t q_nope_offset = q_head_idx * static_cast<size_t>(q_nope_stride_1) + q_in_head_idx * vec_size_i;
        vec_cur = q_buffer_i.template load<vec_size_i>(q_nope_offset);
      }
      
      for (uint32_t next_q_idx = q_vec_idx + vec_stride; next_q_idx < num_vecs; next_q_idx += vec_stride)
      {
        // Calculate next indices for prefetch
        uint32_t next_head_idx = next_q_idx / kv_lora_vec;
        uint32_t next_in_head_idx = next_q_idx % kv_lora_vec;
        size_t next_q_nope_offset = next_head_idx * static_cast<size_t>(q_nope_stride_1) + next_in_head_idx * vec_size_i;
        vec_nxt = q_buffer_i.template load<vec_size_i>(next_q_nope_offset);
        
        // Calculate store offset using current q_head_idx and q_in_head_idx
        uint32_t store_offset = q_head_idx * q_out_stride_1 + q_in_head_idx * vec_size_o;
        
        if constexpr (q_dt == vllm::Fp8KVCacheDataType::kAuto) {
          q_buffer_o.template store<vec_size_o>(vec_cur, store_offset);
        } else {
          q_buffer_o.template store<vec_size_o>(aiter::scaled_cast<query_t>(vec_cur, inverted_qscale), store_offset);
        }
        
        // Update loop variables
        vec_cur = vec_nxt;
        q_vec_idx = next_q_idx;
        q_head_idx = next_head_idx;
        q_in_head_idx = next_in_head_idx;
      }
      
      // Store last Q vector if loaded (use maintained q_head_idx and q_in_head_idx)
      if (q_vec_idx < num_vecs) {
        uint32_t store_offset = q_head_idx * q_out_stride_1 + q_in_head_idx * vec_size_o;
        if constexpr (q_dt == vllm::Fp8KVCacheDataType::kAuto) {
          q_buffer_o.template store<vec_size_o>(vec_cur, store_offset);
        } else {
          q_buffer_o.template store<vec_size_o>(aiter::scaled_cast<query_t>(vec_cur, inverted_qscale), store_offset);
        }
      }
    
      // ============ RoPE Phase ============
      // Adjust base pointers for RoPE region
      if constexpr (is_nope_first) {
        q_out += head_size - pe_dim;
        kv_cache += head_size - pe_dim;
      }
      
      const int32_t nq = num_heads * embed_dim;
      const int32_t nk = num_kv_heads * embed_dim;
      const int32_t token_q_base = token_idx * q_pe_stride_0;
      const int32_t token_k_base = token_idx * k_pe_stride_0;
      const int32_t token_q_out_base = token_idx * q_out_stride_0;
      
      // Phase 1: Process Q and K RoPE together
      for (uint32_t r_idx = threadIdx.x; r_idx < nk; r_idx += vec_stride)
      {
          const uint32_t rope_head = r_idx / embed_dim;
          const uint32_t rot_off = r_idx % embed_dim;
          
          // Calculate cos/sin indices
          uint32_t x_idx, y_idx;
          float f32_cos, f32_sin;
          if constexpr (is_neox) {
            x_idx = rot_off;
            y_idx = embed_dim + rot_off;
            f32_cos = static_cast<float>(*(cos_ptr + x_idx));
            f32_sin = static_cast<float>(*(sin_ptr + x_idx));
          } else {
            x_idx = rot_off << 1;
            y_idx = x_idx + 1;
            f32_cos = static_cast<float>(*(cos_ptr + (x_idx >> 1)));
            f32_sin = static_cast<float>(*(sin_ptr + (x_idx >> 1)));
          }
          
          // K RoPE
          const scalar_t* k_in = k_pe + token_k_base + rope_head * k_pe_stride_1;
          cache_t* k_out = kv_cache + kv_cache_offset + rope_head * kv_cache_stride_h;
          
          float kx = static_cast<float>(k_in[x_idx]);
          float ky = static_cast<float>(k_in[y_idx]);
          float k_rot_x = kx * f32_cos - ky * f32_sin;
          float k_rot_y = ky * f32_cos + kx * f32_sin;
          
          if constexpr (std::is_same_v<cache_t, opus::fp8_t>) {
            k_out[x_idx] = opus::cast<opus::fp8_t>(k_rot_x * inverted_kscale);
            k_out[y_idx] = opus::cast<opus::fp8_t>(k_rot_y * inverted_kscale);
          } else {
            k_out[x_idx] = static_cast<cache_t>(k_rot_x);
            k_out[y_idx] = static_cast<cache_t>(k_rot_y);
          }
          
          // Q RoPE
          const scalar_t* q_in = q_pe + token_q_base + rope_head * q_pe_stride_1;
          query_t* q_out_ptr = q_out + token_q_out_base + rope_head * q_out_stride_1;
          
          float qx = static_cast<float>(q_in[x_idx]);
          float qy = static_cast<float>(q_in[y_idx]);
          float q_rot_x = qx * f32_cos - qy * f32_sin;
          float q_rot_y = qy * f32_cos + qx * f32_sin;
          
          if constexpr (std::is_same_v<query_t, opus::fp8_t>) {
            q_out_ptr[x_idx] = opus::cast<opus::fp8_t>(q_rot_x * inverted_qscale);
            q_out_ptr[y_idx] = opus::cast<opus::fp8_t>(q_rot_y * inverted_qscale);
          } else {
            q_out_ptr[x_idx] = static_cast<query_t>(q_rot_x);
            q_out_ptr[y_idx] = static_cast<query_t>(q_rot_y);
          }
      }
      
      // Phase 2: Process remaining Q RoPE only (when num_heads > num_kv_heads)
      for (uint32_t r_idx = nk + threadIdx.x; r_idx < nq; r_idx += vec_stride)
      {
          const uint32_t rope_head = r_idx / embed_dim;
          const uint32_t rot_off = r_idx % embed_dim;
          
          // Calculate cos/sin indices
          uint32_t x_idx, y_idx;
          float f32_cos, f32_sin;
          if constexpr (is_neox) {
            x_idx = rot_off;
            y_idx = embed_dim + rot_off;
            f32_cos = static_cast<float>(*(cos_ptr + x_idx));
            f32_sin = static_cast<float>(*(sin_ptr + x_idx));
          } else {
            x_idx = rot_off << 1;
            y_idx = x_idx + 1;
            f32_cos = static_cast<float>(*(cos_ptr + (x_idx >> 1)));
            f32_sin = static_cast<float>(*(sin_ptr + (x_idx >> 1)));
          }
          
          // Q RoPE
          const scalar_t* q_in = q_pe + token_q_base + rope_head * q_pe_stride_1;
          query_t* q_out_ptr = q_out + token_q_out_base + rope_head * q_out_stride_1;
          
          float qx = static_cast<float>(q_in[x_idx]);
          float qy = static_cast<float>(q_in[y_idx]);
          
          if constexpr (std::is_same_v<query_t, opus::fp8_t>) {
            q_out_ptr[x_idx] = opus::cast<opus::fp8_t>((qx * f32_cos - qy * f32_sin) * inverted_qscale);
            q_out_ptr[y_idx] = opus::cast<opus::fp8_t>((qy * f32_cos + qx * f32_sin) * inverted_qscale);
          } else {
            q_out_ptr[x_idx] = static_cast<query_t>(qx * f32_cos - qy * f32_sin);
            q_out_ptr[y_idx] = static_cast<query_t>(qy * f32_cos + qx * f32_sin);
          }
      }
    }
    template <typename scalar_t, typename cache_t, typename query_t, vllm::Fp8KVCacheDataType kv_dt, vllm::Fp8KVCacheDataType q_dt>
    __global__ void fuse_qk_rope_concat_and_cache_mla_kernel_prefill(
        const scalar_t* __restrict__ q_nope,  // [num_tokens, num_heads, kv_lora_rank]
        const scalar_t* __restrict__ q_pe,    // [num_tokens, num_heads, pe_dim]
        const scalar_t* __restrict__ kv_c,    // [num_tokens, num_kv_heads, kv_lora_rank]
        const scalar_t* __restrict__ k_pe,    // [num_tokens, num_kv_heads, pe_dim]
        cache_t* __restrict__ kv_cache,       // [num_blocks, block_size, num_kv_heads, (qk_lora_rank + pe_dim)]
        query_t* __restrict__ q_out,          // [num_tokens, num_heads, num_kv_heads, kv_lora_rank + pe_dim]
        const int64_t* __restrict__ slot_mapping,  // [num_tokens]
        const int64_t* __restrict__ positions,     // [num_tokens]
        const scalar_t *__restrict__ cos_cache,        // [max_position, rot_dim //2]
        const scalar_t *__restrict__ sin_cache,        // [max_position, rot_dim //2]
        const int block_stride, const int entry_stride, const int kv_cache_stride_h,
        const int q_nope_stride_0, const int q_nope_stride_1,
        const int q_pe_stride_0, const int q_pe_stride_1,
        const int q_out_stride_0, const int q_out_stride_1, 
        const int num_heads, const int num_kv_heads,
        const int kv_c_stride_0, const int kv_c_stride_1, 
        const int k_pe_stride_0, const int k_pe_stride_1,
        const int kv_lora_rank, const int pe_dim,
        const int block_size,
        const float* scale, const float* q_scale,
        bool is_neox, bool is_nope_first
    ) {
      if (is_neox && is_nope_first) {
        fuse_qk_rope_concat_and_cache_mla_kernel_prefill_opt<scalar_t,cache_t,query_t, kv_dt, q_dt, true, true>(q_nope, q_pe, kv_c, k_pe, kv_cache, q_out, slot_mapping, positions, 
                                           cos_cache, sin_cache, block_stride, entry_stride, kv_cache_stride_h, q_nope_stride_0, q_nope_stride_1, q_pe_stride_0, q_pe_stride_1, q_out_stride_0,
                                           q_out_stride_1, num_heads, num_kv_heads, kv_c_stride_0, kv_c_stride_1, k_pe_stride_0, k_pe_stride_1, kv_lora_rank, pe_dim, block_size, scale, q_scale);
      } else if (is_neox && !is_nope_first) {
        fuse_qk_rope_concat_and_cache_mla_kernel_prefill_opt<scalar_t,cache_t,query_t, kv_dt, q_dt, true, false>(q_nope, q_pe, kv_c, k_pe, kv_cache, q_out, slot_mapping, positions, 
                                            cos_cache, sin_cache, block_stride, entry_stride, kv_cache_stride_h, q_nope_stride_0, q_nope_stride_1, q_pe_stride_0, q_pe_stride_1, q_out_stride_0, 
                                            q_out_stride_1, num_heads, num_kv_heads, kv_c_stride_0, kv_c_stride_1, k_pe_stride_0, k_pe_stride_1, kv_lora_rank, pe_dim, block_size, scale, q_scale);
      } else if (!is_neox && is_nope_first) {
        fuse_qk_rope_concat_and_cache_mla_kernel_prefill_opt<scalar_t,cache_t,query_t, kv_dt, q_dt, false, true>(q_nope, q_pe, kv_c, k_pe, kv_cache, q_out, slot_mapping, positions, 
                                            cos_cache, sin_cache, block_stride, entry_stride, kv_cache_stride_h, q_nope_stride_0, q_nope_stride_1, q_pe_stride_0, q_pe_stride_1, q_out_stride_0, 
                                            q_out_stride_1, num_heads, num_kv_heads, kv_c_stride_0, kv_c_stride_1, k_pe_stride_0, k_pe_stride_1, kv_lora_rank, pe_dim, block_size, scale, q_scale);
      } else {
        fuse_qk_rope_concat_and_cache_mla_kernel_prefill_opt<scalar_t,cache_t,query_t, kv_dt, q_dt, false, false>(q_nope, q_pe, kv_c, k_pe, kv_cache, q_out, slot_mapping, positions, 
                                            cos_cache, sin_cache, block_stride, entry_stride, kv_cache_stride_h, q_nope_stride_0, q_nope_stride_1, q_pe_stride_0, q_pe_stride_1, q_out_stride_0, 
                                            q_out_stride_1, num_heads, num_kv_heads, kv_c_stride_0, kv_c_stride_1, k_pe_stride_0, k_pe_stride_1, kv_lora_rank, pe_dim, block_size, scale, q_scale);
      }
    }

    // General version kernel wrapper with rope_dim parameter
    template <typename scalar_t, typename cache_t, typename query_t, vllm::Fp8KVCacheDataType kv_dt, vllm::Fp8KVCacheDataType q_dt>
    __global__ void fuse_qk_rope_concat_and_cache_mla_kernel_general(
        const scalar_t* __restrict__ q_nope,  // [num_tokens, num_heads, kv_lora_rank]
        const scalar_t* __restrict__ q_pe,    // [num_tokens, num_heads, pe_dim]
        const scalar_t* __restrict__ kv_c,    // [num_tokens, num_kv_heads, kv_lora_rank]
        const scalar_t* __restrict__ k_pe,    // [num_tokens, num_kv_heads, pe_dim]
        cache_t* __restrict__ kv_cache,       // [num_blocks, block_size, num_kv_heads, (qk_lora_rank + pe_dim)]
        query_t* __restrict__ q_out,          // [num_tokens, num_heads, num_kv_heads, kv_lora_rank + pe_dim]
        const int64_t* __restrict__ slot_mapping,  // [num_tokens]
        const int64_t* __restrict__ positions,     // [num_tokens]
        const scalar_t *__restrict__ cos_cache,        // [max_position, rot_dim //2]
        const scalar_t *__restrict__ sin_cache,        // [max_position, rot_dim //2]
        const int block_stride, const int entry_stride, const int kv_cache_stride_h,
        const int q_nope_stride_0, const int q_nope_stride_1,
        const int q_pe_stride_0, const int q_pe_stride_1,
        const int q_out_stride_0, const int q_out_stride_1, 
        const int num_heads, const int num_kv_heads,
        const int kv_c_stride_0, const int kv_c_stride_1, 
        const int k_pe_stride_0, const int k_pe_stride_1,
        const int kv_lora_rank, const int pe_dim,
        const int block_size,
        const float* scale, const float* q_scale,
        bool is_neox, bool is_nope_first
    ) {
      if (is_neox && is_nope_first) {
        fuse_qk_rope_concat_and_cache_mla_kernel_general_kernel<scalar_t,cache_t,query_t, kv_dt, q_dt, true, true>(q_nope, q_pe, kv_c, k_pe, kv_cache, q_out, slot_mapping, positions, 
                                           cos_cache, sin_cache, block_stride, entry_stride, kv_cache_stride_h, q_nope_stride_0, q_nope_stride_1, q_pe_stride_0, q_pe_stride_1, q_out_stride_0,
                                           q_out_stride_1, num_heads, num_kv_heads, kv_c_stride_0, kv_c_stride_1, k_pe_stride_0, k_pe_stride_1, kv_lora_rank, pe_dim, block_size, scale, q_scale);
      } else if (is_neox && !is_nope_first) {
        fuse_qk_rope_concat_and_cache_mla_kernel_general_kernel<scalar_t,cache_t,query_t, kv_dt, q_dt, true, false>(q_nope, q_pe, kv_c, k_pe, kv_cache, q_out, slot_mapping, positions, 
                                            cos_cache, sin_cache, block_stride, entry_stride, kv_cache_stride_h, q_nope_stride_0, q_nope_stride_1, q_pe_stride_0, q_pe_stride_1, q_out_stride_0, 
                                            q_out_stride_1, num_heads, num_kv_heads, kv_c_stride_0, kv_c_stride_1, k_pe_stride_0, k_pe_stride_1, kv_lora_rank, pe_dim, block_size, scale, q_scale);
      } else if (!is_neox && is_nope_first) {
        fuse_qk_rope_concat_and_cache_mla_kernel_general_kernel<scalar_t,cache_t,query_t, kv_dt, q_dt, false, true>(q_nope, q_pe, kv_c, k_pe, kv_cache, q_out, slot_mapping, positions, 
                                            cos_cache, sin_cache, block_stride, entry_stride, kv_cache_stride_h, q_nope_stride_0, q_nope_stride_1, q_pe_stride_0, q_pe_stride_1, q_out_stride_0, 
                                            q_out_stride_1, num_heads, num_kv_heads, kv_c_stride_0, kv_c_stride_1, k_pe_stride_0, k_pe_stride_1, kv_lora_rank, pe_dim, block_size, scale, q_scale);
      } else {
        fuse_qk_rope_concat_and_cache_mla_kernel_general_kernel<scalar_t,cache_t,query_t, kv_dt, q_dt, false, false>(q_nope, q_pe, kv_c, k_pe, kv_cache, q_out, slot_mapping, positions, 
                                            cos_cache, sin_cache, block_stride, entry_stride, kv_cache_stride_h, q_nope_stride_0, q_nope_stride_1, q_pe_stride_0, q_pe_stride_1, q_out_stride_0, 
                                            q_out_stride_1, num_heads, num_kv_heads, kv_c_stride_0, kv_c_stride_1, k_pe_stride_0, k_pe_stride_1, kv_lora_rank, pe_dim, block_size, scale, q_scale);
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
                                     slot_mapping.data_ptr(),                                    \
                                     slot_width,                                                 \
                                     key_stride,                                                 \
                                     value_stride,                                               \
                                     num_heads,                                                  \
                                     head_size,                                                  \
                                     block_size,                                                 \
                                     x,                                                          \
                                     k_scale.has_value() ? reinterpret_cast<float*>(k_scale->data_ptr()) : nullptr, \
                                     v_scale.has_value() ? reinterpret_cast<float*>(v_scale->data_ptr()) : nullptr);

#define CALL_RESHAPE_AND_CACHE_ASM(KV_T, CACHE_T, KV_DTYPE)                                      \
    aiter::reshape_and_cache_kernel<KV_T, CACHE_T, KV_DTYPE, true>                               \
        <<<grid, block, 0, stream>>>(reinterpret_cast<KV_T*>(key.data_ptr()),                    \
                                     reinterpret_cast<KV_T*>(value.data_ptr()),                  \
                                     reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),           \
                                     reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),         \
                                     slot_mapping.data_ptr(),                                    \
                                     slot_width,                                                 \
                                     key_stride,                                                 \
                                     value_stride,                                               \
                                     num_heads,                                                  \
                                     head_size,                                                  \
                                     block_size,                                                 \
                                     x,                                                          \
                                     k_scale.has_value() ? reinterpret_cast<float*>(k_scale->data_ptr()) : nullptr, \
                                     v_scale.has_value() ? reinterpret_cast<float*>(v_scale->data_ptr()) : nullptr);

namespace aiter {

void reshape_and_cache(
    aiter_tensor_t& key,          // [num_tokens, num_heads, head_size]
    aiter_tensor_t& value,        // [num_tokens, num_heads, head_size]
    aiter_tensor_t& key_cache,    // [num_blocks, num_heads, head_size/x, block_size, x]
    aiter_tensor_t& value_cache,  // [num_blocks, num_heads, head_size, block_size]
    aiter_tensor_t& slot_mapping, // [num_tokens]
    const std::string& kv_cache_dtype,
    std::optional<aiter_tensor_t> k_scale,
    std::optional<aiter_tensor_t> v_scale,
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
    HipDeviceGuard device_guard(key.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    const SlotIndexWidth slot_width = slot_index_width(slot_mapping);

    if(asm_layout)
    {
        DISPATCH_BY_KV_CACHE_DTYPE_OPUS_rmTorch(key.dtype(), kv_cache_dtype, CALL_RESHAPE_AND_CACHE_ASM)
    }
    else
    {
        DISPATCH_BY_KV_CACHE_DTYPE_OPUS_rmTorch(key.dtype(), kv_cache_dtype, CALL_RESHAPE_AND_CACHE)
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
                                     slot_mapping.data_ptr(),                            \
                                     slot_width,                                         \
                                     block_stride,                                       \
                                     key_stride,                                         \
                                     value_stride,                                       \
                                     num_heads,                                          \
                                     head_size,                                          \
                                     block_size,                                         \
                                     reinterpret_cast<float*>(k_scale.data_ptr()),                          \
                                     reinterpret_cast<float*>(v_scale.data_ptr()));

namespace aiter {

void reshape_and_cache_flash(
    aiter_tensor_t& key,          // [num_tokens, num_heads, head_size]
    aiter_tensor_t& value,        // [num_tokens, num_heads, head_size]
    aiter_tensor_t& key_cache,    // [num_blocks, block_size, num_heads, head_size]
    aiter_tensor_t& value_cache,  // [num_blocks, block_size, num_heads, head_size]
    aiter_tensor_t& slot_mapping, // [num_tokens]
    const std::string& kv_cache_dtype,
    aiter_tensor_t& k_scale,
    aiter_tensor_t& v_scale)
{
    int num_tokens = key.size(0);
    int num_heads  = key.size(1);
    int head_size  = key.size(2);
    int block_size = key_cache.size(1);

    int key_stride   = key.stride(0);
    int value_stride = value.stride(0);
    int block_stride = key_cache.stride(0);
    AITER_CHECK(key_cache.stride(0) == value_cache.stride(0));

    dim3 grid(num_tokens);
    dim3 block(std::min(num_heads * head_size, 512));
    HipDeviceGuard device_guard(key.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    const SlotIndexWidth slot_width = slot_index_width(slot_mapping);

    DISPATCH_BY_KV_CACHE_DTYPE_OPUS_rmTorch(key.dtype(), kv_cache_dtype, CALL_RESHAPE_AND_CACHE_FLASH);
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
                reinterpret_cast<int64_t*>(slot_mapping.data_ptr()),                                                  \
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
                reinterpret_cast<int64_t*>(slot_mapping.data_ptr()),                                                  \
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
                reinterpret_cast<int64_t*>(slot_mapping.data_ptr()),                                              \
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
                reinterpret_cast<int64_t*>(slot_mapping.data_ptr()),                                              \
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
                reinterpret_cast<int64_t*>(slot_mapping.data_ptr()),                                                  \
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
                reinterpret_cast<int64_t*>(slot_mapping.data_ptr()),                                                  \
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
                                     reinterpret_cast<int64_t*>(slot_mapping.data_ptr()),                \
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
                                     reinterpret_cast<int64_t*>(slot_mapping.data_ptr()),                \
                                     block_stride,                                    \
                                     entry_stride,                                    \
                                     kv_c_stride,                                     \
                                     k_pe_stride,                                     \
                                     kv_lora_rank,                                    \
                                     pe_dim,                                          \
                                     block_size,                                      \
                                     reinterpret_cast<const float*>(scale.data_ptr()));

#define CALL_CONCAT_AND_CACHE_MLA_SEG(KV_T, CACHE_T, KV_DTYPE)                        \
    aiter::concat_and_cache_mla_seg_kernel<KV_T, CACHE_T, KV_DTYPE>                   \
        <<<grid, block, 0, stream>>>(reinterpret_cast<KV_T*>(kv_c.data_ptr()),        \
                                     reinterpret_cast<KV_T*>(k_pe.data_ptr()),        \
                                     reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()), \
                                     reinterpret_cast<int64_t*>(slot_mapping.data_ptr()),                \
                                     block_stride,                                    \
                                     kv_c_stride,                                     \
                                     k_pe_stride,                                     \
                                     kv_lora_rank,                                    \
                                     pe_dim,                                          \
                                     page_size,                                       \
                                     reinterpret_cast<const float*>(scale.data_ptr()));

#define CALL_CONCAT_AND_CACHE_MLA_SEG_OPT(KV_T, CACHE_T, KV_DTYPE)                    \
    aiter::concat_and_cache_mla_seg_opt_kernel<KV_T, CACHE_T, KV_DTYPE, 8>            \
        <<<grid, block, 0, stream>>>(reinterpret_cast<KV_T*>(kv_c.data_ptr()),        \
                                     reinterpret_cast<KV_T*>(k_pe.data_ptr()),        \
                                     reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()), \
                                     reinterpret_cast<int64_t*>(slot_mapping.data_ptr()),                \
                                     block_stride,                                    \
                                     kv_c_stride,                                     \
                                     k_pe_stride,                                     \
                                     kv_lora_rank,                                    \
                                     pe_dim,                                          \
                                     page_size,                                       \
                                     reinterpret_cast<const float*>(scale.data_ptr()));

// 16-bit inputs only (the host never selects this for fp32), so fp32 is not
// instantiated at all.
#define CALL_CONCAT_AND_CACHE_MLA_SEG_512(KV_T, CACHE_T, KV_DTYPE)                    \
    if constexpr(sizeof(KV_T) == 2)                                                   \
    {                                                                                 \
        switch(tpb)                                                                   \
        {                                                                             \
        case 8: CALL_CONCAT_AND_CACHE_MLA_SEG_512_TPB(KV_T, CACHE_T, KV_DTYPE, 8); break; \
        case 4: CALL_CONCAT_AND_CACHE_MLA_SEG_512_TPB(KV_T, CACHE_T, KV_DTYPE, 4); break; \
        default: CALL_CONCAT_AND_CACHE_MLA_SEG_512_TPB(KV_T, CACHE_T, KV_DTYPE, 1); break;\
        }                                                                             \
    }

#define CALL_CONCAT_AND_CACHE_MLA_SEG_512_TPB(KV_T, CACHE_T, KV_DTYPE, TPB_)          \
    aiter::concat_and_cache_mla_seg_512_kernel<KV_T, CACHE_T, KV_DTYPE, TPB_>         \
        <<<dim3((num_tokens + TPB_ - 1) / TPB_), dim3(64), 0, stream>>>(              \
            reinterpret_cast<KV_T*>(kv_c.data_ptr()),                                 \
            reinterpret_cast<KV_T*>(k_pe.data_ptr()),                                 \
            reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()),                          \
            reinterpret_cast<int64_t*>(slot_mapping.data_ptr()),                      \
            num_tokens,                                                               \
            (int64_t)block_stride,                                                    \
            (int64_t)kv_c_stride,                                                     \
            (int64_t)k_pe_stride,                                                     \
            reinterpret_cast<const float*>(scale.data_ptr()));

// Macro to dispatch the kernel based on the data type.
#define CALL_INDEXER_K_QUANT_AND_CACHE(KV_T, CACHE_T, KV_DTYPE)                                   \
    aiter::indexer_k_quant_and_cache_kernel<KV_T,                                                 \
                                            CACHE_T,                                              \
                                            KV_DTYPE,                                             \
                                            blockDimx,                                            \
                                            blockDimy,                                            \
                                            vec_size>                                             \
        <<<grid, block, 0, stream>>>(reinterpret_cast<KV_T*>(k.data_ptr()),                       \
                                     reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()),             \
                                     reinterpret_cast<int64_t*>(slot_mapping.data_ptr()),                            \
                                     num_tokens,                                                  \
                                     head_dim,                                                    \
                                     quant_block_size,                                            \
                                     cache_block_size,                                            \
                                     cache_stride,                                                \
                                     use_ue8m0,                                                   \
                                     do_preshuffle);

#define INDEXER_QK_LAUNCH_ONE(KV_T, CACHE_T, KV_DTYPE, FP4_OUT, Q_OUT_T, W_OUT_T, NARROW)         \
    aiter::indexer_qk_rope_quant_and_cache_kernel<KV_T,                                           \
                                                 CACHE_T,                                         \
                                                 KV_DTYPE,                                        \
                                                 128,                                             \
                                                 64,                                              \
                                                 FP4_OUT,                                         \
                                                 NARROW>                                          \
        <<<dim3(num_tokens, (n_heads + heads_per_block - 1) / heads_per_block),                   \
           block,                                                                                 \
           0,                                                                                     \
           stream>>>(reinterpret_cast<KV_T*>(q.data_ptr()),                                       \
                                     reinterpret_cast<Q_OUT_T*>(q_out.data_ptr()),                \
                                     reinterpret_cast<KV_T*>(weights.data_ptr()),                  \
                                     reinterpret_cast<W_OUT_T*>(weights_out.data_ptr()),           \
                                     reinterpret_cast<KV_T*>(k.data_ptr()),                       \
                                     reinterpret_cast<Q_OUT_T*>(kv_cache.data_ptr()),             \
                                     reinterpret_cast<int64_t*>(slot_mapping.data_ptr()),          \
                                     reinterpret_cast<float*>(norm_weight.data_ptr()),             \
                                     reinterpret_cast<float*>(norm_bias.data_ptr()),               \
                                     reinterpret_cast<int64_t*>(positions.data_ptr()),             \
                                     reinterpret_cast<KV_T*>(cos_cache.data_ptr()),                \
                                     reinterpret_cast<KV_T*>(sin_cache.data_ptr()),                \
                                     q_scale_out_ptr,                                             \
                                     kv_cache_scale_ptr,                                          \
                                     num_tokens,                                                  \
                                     n_heads,                                                     \
                                     quant_block_size,                                            \
                                     cache_block_size,                                            \
                                     cache_stride,                                                \
                                     q.stride(0),                                                 \
                                     q.stride(1),                                                 \
                                     q.stride(2),                                                 \
                                     q_out.stride(0),                                             \
                                     q_out.stride(1),                                             \
                                     q_out.stride(2),                                             \
                                     weights.stride(0),                                           \
                                     weights.stride(1),                                           \
                                     weights_out.stride(0),                                       \
                                     weights_out.stride(1),                                       \
                                     k.stride(0),                                                 \
                                     k.stride(1),                                                 \
                                     cos_cache.stride(0),                                         \
                                     sin_cache.stride(0),                                         \
                                     epsilon,                                                     \
                                     weights_scale,                                               \
                                     use_ue8m0,                                                   \
                                     do_preshuffle,                                               \
                                     is_neox,                                                      \
                                     max_position,                                                \
                                     compute_all_q_rope);

#define CALL_INDEXER_QK_ROPE_QUANT_AND_CACHE_IMPL(                                                \
    KV_T, CACHE_T, KV_DTYPE, FP4_OUT, Q_OUT_T, W_OUT_T)                                           \
    if(narrow)                                                                                    \
    {                                                                                             \
        INDEXER_QK_LAUNCH_ONE(KV_T, CACHE_T, KV_DTYPE, FP4_OUT, Q_OUT_T, W_OUT_T, true)           \
    }                                                                                             \
    else                                                                                          \
    {                                                                                             \
        INDEXER_QK_LAUNCH_ONE(KV_T, CACHE_T, KV_DTYPE, FP4_OUT, Q_OUT_T, W_OUT_T, false)          \
    }

#define CALL_INDEXER_QK_ROPE_QUANT_AND_CACHE(KV_T, CACHE_T, KV_DTYPE)                             \
    CALL_INDEXER_QK_ROPE_QUANT_AND_CACHE_IMPL(KV_T, CACHE_T, KV_DTYPE, false, CACHE_T, float)

#define CALL_INDEXER_QK_ROPE_QUANT_AND_CACHE_FP4(KV_T, CACHE_T, KV_DTYPE)                         \
    CALL_INDEXER_QK_ROPE_QUANT_AND_CACHE_IMPL(KV_T, CACHE_T, KV_DTYPE, true, uint8_t, KV_T)

#define CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(BLOCK_Y_SIZE)          \
    aiter::cp_gather_indexer_k_quant_cache_kernel<8, BLOCK_Y_SIZE>  \
        <<<dim3((num_tokens + BLOCK_Y_SIZE - 1) / BLOCK_Y_SIZE,     \
                (head_dim + 8 * vec_size - 1) / (8 * vec_size)),    \
           dim3(8, BLOCK_Y_SIZE),                                   \
           0,                                                       \
           stream>>>(reinterpret_cast<char*>(kv_cache.data_ptr()),  \
                     reinterpret_cast<char*>(dst_k.data_ptr()),     \
                     reinterpret_cast<char*>(dst_scale.data_ptr()), \
                     reinterpret_cast<int32_t*>(block_table.data_ptr()),               \
                     reinterpret_cast<int32_t*>(cu_seq_lens.data_ptr()),               \
                     batch_size,                                    \
                     dst_k.stride(0),                               \
                     dst_k.size(1),                                 \
                     kv_cache.stride(0),                            \
                     kv_cache.stride(1),                            \
                     kv_cache.size(1),                              \
                     block_table.size(1),                           \
                     num_tokens,                                    \
                     quant_block_size,                              \
                     do_preshuffle);

#define CALL_FUSED_QK_ROPE_CONCAT_AND_CACHE_MLA_OPT(KV_T, CACHE_T, QUERY_T, KV_DTYPE, Q_DTYPE, VEC_SIZE)   \
 aiter::fuse_qk_rope_concat_and_cache_mla_per_head_kernel<KV_T, CACHE_T, QUERY_T, KV_DTYPE, Q_DTYPE, VEC_SIZE>      \
       <<<grid, block, 0, stream>>>(                                                             \
         reinterpret_cast<KV_T*>(q_nope.data_ptr()),                                             \
         reinterpret_cast<KV_T*>(q_pe.data_ptr()),                                               \
         reinterpret_cast<KV_T*>(kv_c.data_ptr()),                                               \
         reinterpret_cast<KV_T*>(k_pe.data_ptr()),                                               \
         reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()),                                        \
         reinterpret_cast<QUERY_T*>(q_out.data_ptr()),                                           \
         reinterpret_cast<int64_t*>(slot_mapping.data_ptr()),                                                       \
         reinterpret_cast<int64_t*>(positions.data_ptr()),                                                          \
         reinterpret_cast<KV_T*>(cos_cache.data_ptr()),                                          \
         reinterpret_cast<KV_T*>(sin_cache.data_ptr()),                                          \
         block_stride, entry_stride,                                                             \
         q_nope_stride_0, q_nope_stride_1, q_pe_stride_0, q_pe_stride_1,                         \
         q_out_stride_0, q_out_stride_1, num_heads,                                              \
         kv_c_stride, k_pe_stride, kv_lora_rank, pe_dim, block_size,                             \
         reinterpret_cast<const float*>(k_scale.data_ptr()),                                     \
         reinterpret_cast<const float*>(q_scale.data_ptr()),                                     \
         is_neox, is_nope_first, max_position, compute_all_q_rope);
#define CALL_FUSED_QK_ROPE_CONCAT_AND_CACHE_MLA(KV_T, CACHE_T, QUERY_T, KV_DTYPE, Q_DTYPE)   \
 aiter::fuse_qk_rope_concat_and_cache_mla_kernel<KV_T, CACHE_T, QUERY_T, KV_DTYPE, Q_DTYPE>      \
       <<<grid, block, 0, stream>>>(                                                             \
         reinterpret_cast<KV_T*>(q_nope.data_ptr()),                                             \
         reinterpret_cast<KV_T*>(q_pe.data_ptr()),                                               \
         reinterpret_cast<KV_T*>(kv_c.data_ptr()),                                               \
         reinterpret_cast<KV_T*>(k_pe.data_ptr()),                                               \
         reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()),                                        \
         reinterpret_cast<QUERY_T*>(q_out.data_ptr()),                                           \
         reinterpret_cast<int64_t*>(slot_mapping.data_ptr()),                                                       \
         reinterpret_cast<int64_t*>(positions.data_ptr()),                                                          \
         reinterpret_cast<KV_T*>(cos_cache.data_ptr()),                                          \
         reinterpret_cast<KV_T*>(sin_cache.data_ptr()),                                          \
         block_stride, entry_stride,                                                             \
         q_nope_stride_0, q_nope_stride_1, q_pe_stride_0, q_pe_stride_1,                         \
         q_out_stride_0, q_out_stride_1, num_heads,                                              \
         kv_c_stride, k_pe_stride, kv_lora_rank, pe_dim, block_size,                             \
         reinterpret_cast<const float*>(k_scale.data_ptr()),                                     \
         reinterpret_cast<const float*>(q_scale.data_ptr()),                                     \
         is_neox, is_nope_first, max_position, compute_all_q_rope);
#define CALL_PREFILL_FUSED_QK_ROPE_CONCAT_AND_CACHE_MLA(KV_T, CACHE_T, QUERY_T, KV_DTYPE, Q_DTYPE)   \
         aiter::fuse_qk_rope_concat_and_cache_mla_kernel_prefill<KV_T, CACHE_T, QUERY_T, KV_DTYPE, Q_DTYPE>      \
               <<<grid, block, 0, stream>>>(                                                             \
                 reinterpret_cast<KV_T*>(q_nope.data_ptr()),                                             \
                 reinterpret_cast<KV_T*>(q_pe.data_ptr()),                                               \
                 reinterpret_cast<KV_T*>(kv_c.data_ptr()),                                               \
                 reinterpret_cast<KV_T*>(k_pe.data_ptr()),                                               \
                 reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()),                                        \
                 reinterpret_cast<QUERY_T*>(q_out.data_ptr()),                                           \
                 reinterpret_cast<int64_t*>(slot_mapping.data_ptr()),                                                       \
                 reinterpret_cast<int64_t*>(positions.data_ptr()),                                                          \
                 reinterpret_cast<KV_T*>(cos_cache.data_ptr()),                                          \
                 reinterpret_cast<KV_T*>(sin_cache.data_ptr()),                                          \
                 block_stride, entry_stride, kv_cache_stride_h,                                             \
                 q_nope_stride_0, q_nope_stride_1, q_pe_stride_0, q_pe_stride_1,                         \
                 q_out_stride_0, q_out_stride_1, num_heads, num_kv_heads,                                \
                 kv_c_stride_0, kv_c_stride_1, k_pe_stride_0, k_pe_stride_1,                             \
                 kv_lora_rank, pe_dim, block_size,                                                       \
                 reinterpret_cast<const float*>(k_scale.data_ptr()),                                     \
                 reinterpret_cast<const float*>(q_scale.data_ptr()),                                     \
                 is_neox, is_nope_first);
#define CALL_FUSED_QK_ROPE_CONCAT_AND_CACHE_MLA_GENERAL(KV_T, CACHE_T, QUERY_T, KV_DTYPE, Q_DTYPE)   \
                 aiter::fuse_qk_rope_concat_and_cache_mla_kernel_general<KV_T, CACHE_T, QUERY_T, KV_DTYPE, Q_DTYPE>      \
                       <<<grid, block, 0, stream>>>(                                                             \
                         reinterpret_cast<KV_T*>(q_nope.data_ptr()),                                             \
                         reinterpret_cast<KV_T*>(q_pe.data_ptr()),                                               \
                         reinterpret_cast<KV_T*>(kv_c.data_ptr()),                                               \
                         reinterpret_cast<KV_T*>(k_pe.data_ptr()),                                               \
                         reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()),                                        \
                         reinterpret_cast<QUERY_T*>(q_out.data_ptr()),                                           \
                         reinterpret_cast<int64_t*>(slot_mapping.data_ptr()),                                                       \
                         reinterpret_cast<int64_t*>(positions.data_ptr()),                                                          \
                         reinterpret_cast<KV_T*>(cos_cache.data_ptr()),                                          \
                         reinterpret_cast<KV_T*>(sin_cache.data_ptr()),                                          \
                         block_stride, entry_stride, kv_cache_stride_h,                                             \
                         q_nope_stride_0, q_nope_stride_1, q_pe_stride_0, q_pe_stride_1,                         \
                         q_out_stride_0, q_out_stride_1, num_heads, num_kv_heads,                                \
                         kv_c_stride_0, kv_c_stride_1, k_pe_stride_0, k_pe_stride_1,                             \
                         kv_lora_rank, pe_dim, block_size,                                                       \
                         reinterpret_cast<const float*>(k_scale.data_ptr()),                                     \
                         reinterpret_cast<const float*>(q_scale.data_ptr()),                                     \
                         is_neox, is_nope_first);
namespace aiter {

void reshape_and_cache_with_pertoken_quant(
    aiter_tensor_t& key,              // [num_tokens, num_heads, head_size]
    aiter_tensor_t& value,            // [num_tokens, num_heads, head_size]
    aiter_tensor_t& key_cache,        // [num_blocks, num_heads, head_size/x, block_size, x]
    aiter_tensor_t& value_cache,      // [num_blocks, num_heads, head_size, block_size]
    aiter_tensor_t& k_dequant_scales, // [num_heads, max_kv_tokens]
    aiter_tensor_t& v_dequant_scales, // [num_heads, max_kv_tokens]
    aiter_tensor_t& slot_mapping,     // [num_tokens]
    const bool asm_layout)
{
    int num_tokens    = key.size(0);
    int num_heads     = key.size(1);
    int head_size     = key.size(2);
    int block_size    = key_cache.size(3);
    int x             = key_cache.size(4);
    int max_kv_tokens = k_dequant_scales.size(1);
    AITER_CHECK(head_size <= 512, __func__, " Unsupported head_size: ", head_size);

    int key_stride   = key.stride(0);
    int value_stride = value.stride(0);

    dim3 grid(num_tokens, num_heads);
    dim3 block(WARP_SIZE);
    HipDeviceGuard device_guard(key.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    using dequant_scale_t = float; // should align with k_dequant_scales/v_dequant_scales dtype

    float dtypeMax;
    if(key_cache.dtype() == AITER_DTYPE_fp8)
    {
        if(key.dtype() == AITER_DTYPE_fp32)
        {
            CALL_RESHAPE_AND_CACHE_WITH_PERTOKEN_QUANT(float, opus::fp8_t, dequant_scale_t);
        }
        else if(key.dtype() == AITER_DTYPE_fp16)
        {
            CALL_RESHAPE_AND_CACHE_WITH_PERTOKEN_QUANT(
                opus::fp16_t, opus::fp8_t, dequant_scale_t);
        }
        else if(key.dtype() == AITER_DTYPE_bf16)
        {
            CALL_RESHAPE_AND_CACHE_WITH_PERTOKEN_QUANT(
                opus::bf16_t, opus::fp8_t, dequant_scale_t);
        }
        else
        {
            AITER_CHECK(false, "Unsupported input type of kv: ", key.dtype());
        }
    }
    else if(key_cache.dtype() == AITER_DTYPE_i8)
    {
        if(key.dtype() == AITER_DTYPE_fp32)
        {
            CALL_RESHAPE_AND_CACHE_WITH_PERTOKEN_QUANT(float, opus::i8_t, dequant_scale_t);
        }
        else if(key.dtype() == AITER_DTYPE_fp16)
        {
            CALL_RESHAPE_AND_CACHE_WITH_PERTOKEN_QUANT(opus::fp16_t, opus::i8_t, dequant_scale_t);
        }
        else if(key.dtype() == AITER_DTYPE_bf16)
        {
            CALL_RESHAPE_AND_CACHE_WITH_PERTOKEN_QUANT(opus::bf16_t, opus::i8_t, dequant_scale_t);
        }
        else
        {
            AITER_CHECK(false,
                        "Unsupported input type of kv: ",
                        key.dtype(),
                        " kv cache: ",
                        key_cache.dtype());
        }
    }
    else
    {
        AITER_CHECK(false, "Unsupported data type of kv cache: ", key_cache.dtype());
    }
}

void reshape_and_cache_with_block_quant(
    aiter_tensor_t& key,              // [batch_size, seq_len, num_heads, head_size]
    aiter_tensor_t& value,            // [batch_size, seq_len, num_heads, head_size]
    aiter_tensor_t& key_cache,        // [num_blocks, num_heads, head_size/x, block_size, x]
    aiter_tensor_t& value_cache,      // [num_blocks, num_heads, head_size, block_size]
    aiter_tensor_t& k_dequant_scales, // [num_heads, num_blocks]
    aiter_tensor_t& v_dequant_scales, // [num_heads, num_blocks]
    aiter_tensor_t& slot_mapping,     // [num_tokens]
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
    HipDeviceGuard device_guard(key.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    using dequant_scale_t = float; // should align with k_dequant_scales/v_dequant_scales dtype

    float dtypeMax;
    if(key_cache.dtype() == AITER_DTYPE_fp8)
    {
        if(key.dtype() == AITER_DTYPE_fp32)
        {
            CALL_RESHAPE_AND_CACHE_WITH_BLOCK_QUANT(float, opus::fp8_t, dequant_scale_t);
        }
        else if(key.dtype() == AITER_DTYPE_fp16)
        {
            CALL_RESHAPE_AND_CACHE_WITH_BLOCK_QUANT(
                opus::fp16_t, opus::fp8_t, dequant_scale_t);
        }
        else if(key.dtype() == AITER_DTYPE_bf16)
        {
            CALL_RESHAPE_AND_CACHE_WITH_BLOCK_QUANT(
                opus::bf16_t, opus::fp8_t, dequant_scale_t);
        }
        else
        {
            AITER_CHECK(false, "Unsupported input type of kv: ", key.dtype());
        }
    }
    else if(key_cache.dtype() == AITER_DTYPE_i8)
    {
        if(key.dtype() == AITER_DTYPE_fp32)
        {
            CALL_RESHAPE_AND_CACHE_WITH_BLOCK_QUANT(float, opus::i8_t, dequant_scale_t);
        }
        else if(key.dtype() == AITER_DTYPE_fp16)
        {
            CALL_RESHAPE_AND_CACHE_WITH_BLOCK_QUANT(
                opus::fp16_t, opus::i8_t, dequant_scale_t);
        }
        else if(key.dtype() == AITER_DTYPE_bf16)
        {
            CALL_RESHAPE_AND_CACHE_WITH_BLOCK_QUANT(
                opus::bf16_t, opus::i8_t, dequant_scale_t);
        }
        else
        {
            AITER_CHECK(false,
                        "Unsupported input type of kv: ",
                        key.dtype(),
                        " kv cache: ",
                        key_cache.dtype());
        }
    }
    else
    {
        AITER_CHECK(false, "Unsupported data type of kv cache: ", key_cache.dtype());
    }
}

void reshape_and_cache_with_block_quant_for_asm_pa(
    aiter_tensor_t& key,              // [batch_size, seq_len, num_heads, head_size]
    aiter_tensor_t& value,            // [batch_size, seq_len, num_heads, head_size]
    aiter_tensor_t& key_cache,        // [num_blocks, num_heads, head_size/x, block_size:16, x]
    aiter_tensor_t& value_cache,      // [num_blocks, num_heads, head_size, block_size:16]
    aiter_tensor_t& k_dequant_scales, // [num_heads, num_blocks/(ori_block_size/block_size:16)]
    aiter_tensor_t& v_dequant_scales, // [num_heads, num_blocks/(ori_block_size/block_size:16)]
    aiter_tensor_t& slot_mapping,     // [num_tokens]
    const bool asm_layout,
    const int ori_block_size)
{
    AITER_CHECK(
        key.dim() == 4 && value.dim() == 4,
        "key/value must be a 4D tensor with shape [batch_size, seq_len, num_heads, head_size]");
    AITER_CHECK(ori_block_size == 128 || ori_block_size == 256,
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
    HipDeviceGuard device_guard(key.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    using dequant_scale_t = float; // should align with k_dequant_scales/v_dequant_scales dtype

    if(key_cache.dtype() == AITER_DTYPE_fp8)
    {
        if(key.dtype() == AITER_DTYPE_fp32)
        {
            CALL_RESHAPE_AND_CACHE_WITH_BLOCK_QUANT_FOR_ASMPA(
                float, opus::fp8_t, dequant_scale_t);
        }
        else if(key.dtype() == AITER_DTYPE_fp16)
        {
            CALL_RESHAPE_AND_CACHE_WITH_BLOCK_QUANT_FOR_ASMPA(
                opus::fp16_t, opus::fp8_t, dequant_scale_t);
        }
        else if(key.dtype() == AITER_DTYPE_bf16)
        {
            CALL_RESHAPE_AND_CACHE_WITH_BLOCK_QUANT_FOR_ASMPA(
                opus::bf16_t, opus::fp8_t, dequant_scale_t);
        }
        else
        {
            AITER_CHECK(false, "Unsupported input type of kv: ", key.dtype());
        }
    }
    else if(key_cache.dtype() == AITER_DTYPE_i8)
    {
        if(key.dtype() == AITER_DTYPE_fp32)
        {
            CALL_RESHAPE_AND_CACHE_WITH_BLOCK_QUANT_FOR_ASMPA(
                float, opus::i8_t, dequant_scale_t);
        }
        else if(key.dtype() == AITER_DTYPE_fp16)
        {
            CALL_RESHAPE_AND_CACHE_WITH_BLOCK_QUANT_FOR_ASMPA(
                opus::fp16_t, opus::i8_t, dequant_scale_t);
        }
        else if(key.dtype() == AITER_DTYPE_bf16)
        {
            CALL_RESHAPE_AND_CACHE_WITH_BLOCK_QUANT_FOR_ASMPA(
                opus::bf16_t, opus::i8_t, dequant_scale_t);
        }
        else
        {
            AITER_CHECK(false,
                        "Unsupported input type of kv: ",
                        key.dtype(),
                        " kv cache: ",
                        key_cache.dtype());
        }
    }
    else
    {
        AITER_CHECK(false, "Unsupported data type of kv cache: ", key_cache.dtype());
    }
}

void concat_and_cache_mla(aiter_tensor_t& kv_c,         // [num_tokens, kv_lora_rank]
                          aiter_tensor_t& k_pe,         // [num_tokens, pe_dim]
                          aiter_tensor_t& kv_cache,     // [num_blocks, block_size, (kv_lora_rank +
                                                       // pe_dim)]
                          aiter_tensor_t& slot_mapping, // [num_tokens] or [num_actual_tokens]
                          const std::string& kv_cache_dtype,
                          aiter_tensor_t& scale)
{
    int num_tokens   = slot_mapping.size(0);
    int kv_lora_rank = kv_c.size(1);
    int pe_dim       = k_pe.size(1);
    int block_size   = kv_cache.size(1);

    AITER_CHECK(kv_cache.size(2) == kv_lora_rank + pe_dim);
    int kv_c_stride  = kv_c.stride(0);
    int k_pe_stride  = k_pe.stride(0);
    int block_stride = kv_cache.stride(0);
    int entry_stride = kv_cache.stride(1);
    HipDeviceGuard device_guard(kv_c.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    if((pe_dim & 0x7) == 0 && (kv_lora_rank & 0x7) == 0)
    {
        dim3 grid(num_tokens);
        dim3 block(std::min(kv_lora_rank, 1024) / 8);
        DISPATCH_BY_KV_CACHE_DTYPE_OPUS_rmTorch(kv_c.dtype(), kv_cache_dtype, CALL_CONCAT_AND_CACHE_MLA_OPT);
    }
    else
    {
        dim3 grid(num_tokens);
        dim3 block(std::min(kv_lora_rank, 512));
        DISPATCH_BY_KV_CACHE_DTYPE_OPUS_rmTorch(kv_c.dtype(), kv_cache_dtype, CALL_CONCAT_AND_CACHE_MLA);
    }
}

// Same as concat_and_cache_mla but writes the segmented block layout used by
// fused_qk_rope_concat_and_cache_mla_seg: kv_cache is flat
// [num_blocks, page_size*(kv_lora_rank + pe_dim)], nope segment then pe segment.
void concat_and_cache_mla_seg(aiter_tensor_t& kv_c,         // [num_tokens, kv_lora_rank]
                              aiter_tensor_t& k_pe,         // [num_tokens, pe_dim]
                              aiter_tensor_t& kv_cache,     // [num_blocks, page_size*(kv_lora+pe)]
                              aiter_tensor_t& slot_mapping, // [num_tokens]
                              const std::string& kv_cache_dtype,
                              aiter_tensor_t& scale)
{
    int num_tokens   = slot_mapping.size(0);
    int kv_lora_rank = kv_c.size(-1);
    int pe_dim       = k_pe.size(-1);
    int kv_c_stride  = kv_c.stride(0);
    int k_pe_stride  = k_pe.stride(0);
    int block_stride = kv_cache.stride(0);

    const int entry = kv_lora_rank + pe_dim;
    AITER_CHECK(block_stride % entry == 0,
                "kv_cache block stride must be a multiple of kv_lora_rank + pe_dim");
    int page_size = block_stride / entry;
    AITER_CHECK(kv_c.stride(-1) == 1 && k_pe.stride(-1) == 1,
                "kv_c/k_pe must be contiguous in last dim");

    HipDeviceGuard device_guard(kv_c.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    // The fused seg op's own layout gets a dedicated kernel (see its comment);
    // any other rank/pe/page keeps the generic vectorized one. 16-bit inputs
    // only, since the dedicated kernel moves 8 elements as one 16 B load.
    const bool seg_512 = kv_lora_rank == 512 && pe_dim == 64 && page_size == 64 &&
                         kv_c.element_size() == 2 && num_tokens > 0;
    if(seg_512)
    {
        // Tokens per block: a token is only ~1.7 KB, so below ~48 per CU the
        // op sits on its launch floor and one token per block is the shortest
        // path; more loads in flight pay from there, and eight from ~96 per CU
        // (256 CUs: T=8192 1->4.41 us 8->4.78, T=16384 4->5.30 8->5.48,
        // T=65536 4->11.65 8->11.02).
        const int64_t per_cu = num_tokens / static_cast<int64_t>(get_num_cu_func());
        const int tpb        = per_cu >= 96 ? 8 : (per_cu >= 48 ? 4 : 1);
        DISPATCH_BY_KV_CACHE_DTYPE_OPUS_rmTorch(
            kv_c.dtype(), kv_cache_dtype, CALL_CONCAT_AND_CACHE_MLA_SEG_512);
    }
    else if((pe_dim & 0x7) == 0 && (kv_lora_rank & 0x7) == 0)
    {
        dim3 grid(num_tokens);
        dim3 block(std::min(kv_lora_rank / 8, 1024));
        DISPATCH_BY_KV_CACHE_DTYPE_OPUS_rmTorch(
            kv_c.dtype(), kv_cache_dtype, CALL_CONCAT_AND_CACHE_MLA_SEG_OPT);
    }
    else
    {
        dim3 grid(num_tokens);
        dim3 block(std::min(kv_lora_rank, 512));
        DISPATCH_BY_KV_CACHE_DTYPE_OPUS_rmTorch(
            kv_c.dtype(), kv_cache_dtype, CALL_CONCAT_AND_CACHE_MLA_SEG);
    }
}

// copy from vllm: https://github.com/vllm-project/vllm/blob/main/csrc/cache_kernels.cu
void indexer_k_quant_and_cache(aiter_tensor_t& k,        // [num_tokens, head_dim]
                               aiter_tensor_t& kv_cache, // [num_blocks, block_size, cache_stride]
                               aiter_tensor_t& slot_mapping, // [num_tokens]
                               int64_t quant_block_size,    // quantization block size
                               const std::string& scale_fmt,
                               bool preshuffle)
{
    int num_tokens       = std::min(k.size(0), slot_mapping.size(0));
    int head_dim         = k.size(1);
    int cache_block_size = kv_cache.size(1);
    int cache_stride     = kv_cache.size(2);
    bool use_ue8m0       = scale_fmt == "ue8m0";
    bool do_preshuffle   = preshuffle;

    AITER_CHECK(k.device_id == kv_cache.device_id, "k and kv_cache must be on the same device");
    AITER_CHECK(k.device_id == slot_mapping.device_id,
                "k and slot_mapping must be on the same device");

    AITER_CHECK(head_dim % quant_block_size == 0, "head_dim must be divisible by quant_block_size");
    if(preshuffle)
    {
        AITER_CHECK(cache_block_size % 16 == 0,
                    "preshuffle requires cache_block_size to be a multiple of 16, got ",
                    cache_block_size);
        AITER_CHECK(head_dim % 16 == 0,
                    "preshuffle requires head_dim to be a multiple of 16, got ",
                    head_dim);
    }

    int quant_blocks    = num_tokens * head_dim / quant_block_size;
    const int vec_size  = 16;
    const int blockDimx = 8;
    const int blockDimy = opus::get_warp_size() / blockDimx;
    dim3 grid((quant_blocks + blockDimy - 1) / (blockDimy));
    dim3 block(blockDimx, blockDimy);
    HipDeviceGuard device_guard(k.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    DISPATCH_BY_KV_CACHE_DTYPE_OPUS_rmTorch(k.dtype(), "fp8_e4m3", CALL_INDEXER_K_QUANT_AND_CACHE);
}

void indexer_qk_rope_quant_and_cache(
    aiter_tensor_t& q,            // [num_tokens, n_heads, head_dim]
    aiter_tensor_t& q_out,        // [num_tokens, n_heads, head_dim]
    aiter_tensor_t& weights,      // [num_tokens, n_heads]
    aiter_tensor_t& weights_out,  // [num_tokens, n_heads]
    aiter_tensor_t& k,            // [num_tokens, head_dim]
    aiter_tensor_t& kv_cache,     // [num_blocks, block_size, cache_stride]
    aiter_tensor_t& slot_mapping, // [num_tokens]
    aiter_tensor_t& norm_weight,  // [head_dim]
    aiter_tensor_t& norm_bias,    // [head_dim]
    aiter_tensor_t& positions,    // [num_tokens]
    aiter_tensor_t& cos_cache,    // [max_position, ..., rope_dim / 2]
    aiter_tensor_t& sin_cache,    // [max_position, ..., rope_dim / 2]
    double epsilon,
    int64_t quant_block_size,
    const std::string& scale_fmt,
    double weights_scale,
    bool preshuffle,
    bool is_neox,
    bool compute_all_q_rope,
    std::optional<aiter_tensor_t> q_scale_out,
    std::optional<aiter_tensor_t> kv_cache_scale)
{
    const bool fp4_out = q_scale_out.has_value() || kv_cache_scale.has_value();
    AITER_CHECK(!fp4_out || (q_scale_out.has_value() && kv_cache_scale.has_value()),
                "fp4 output needs both q_scale_out and kv_cache_scale");
    AITER_CHECK(!fp4_out || kv_cache.dim() == 5,
                "fp4 kv_cache must be [num_blocks, k_tiles, 4, kv_block_size, 16], got dim ",
                kv_cache.dim());

    int num_tokens       = std::min(k.size(0), slot_mapping.size(0));
    int head_dim         = k.size(1);
    int n_heads          = q.size(1);
    int rope_dim         = cos_cache.size(-1) * 2;
    int cache_block_size = fp4_out ? kv_cache.size(3) : kv_cache.size(1);
    int cache_stride     = fp4_out ? 0 : kv_cache.size(2);
    int max_position     = cos_cache.size(0);
    bool use_ue8m0       = scale_fmt == "ue8m0";
    bool do_preshuffle   = preshuffle;

    AITER_CHECK(q.device_id == k.device_id, "q and k must be on the same device");
    AITER_CHECK(q.device_id == q_out.device_id, "q and q_out must be on the same device");
    AITER_CHECK(q.device_id == weights.device_id, "q and weights must be on the same device");
    AITER_CHECK(q.device_id == weights_out.device_id,
                "q and weights_out must be on the same device");
    AITER_CHECK(q.device_id == kv_cache.device_id, "q and kv_cache must be on the same device");
    AITER_CHECK(q.device_id == slot_mapping.device_id,
                "q and slot_mapping must be on the same device");
    AITER_CHECK(q.device_id == norm_weight.device_id,
                "q and norm_weight must be on the same device");
    AITER_CHECK(q.device_id == norm_bias.device_id, "q and norm_bias must be on the same device");
    AITER_CHECK(q.device_id == positions.device_id, "q and positions must be on the same device");
    AITER_CHECK(q.device_id == cos_cache.device_id, "q and cos_cache must be on the same device");
    AITER_CHECK(q.device_id == sin_cache.device_id, "q and sin_cache must be on the same device");
    AITER_CHECK(q.dim() == 3, "q must be [num_tokens, n_heads, head_dim]");
    AITER_CHECK(q_out.dim() == 3, "q_out must be [num_tokens, n_heads, head_dim]");
    AITER_CHECK(k.dim() == 2, "k must be [num_tokens, head_dim]");
    AITER_CHECK(weights.dim() == 2, "weights must be [num_tokens, n_heads]");
    AITER_CHECK(weights_out.dim() == 2, "weights_out must be [num_tokens, n_heads]");
    AITER_CHECK(cos_cache.dim() == 2, "cos_cache must be [max_position, rope_dim / 2]");
    AITER_CHECK(sin_cache.dim() == 2, "sin_cache must be [max_position, rope_dim / 2]");
    AITER_CHECK(q.size(0) >= num_tokens, "q must cover all indexed tokens");
    AITER_CHECK(!compute_all_q_rope || q.size(0) == num_tokens,
                "compute_all_q_rope requires q to have exactly num_tokens rows");
    AITER_CHECK(q.size(2) == head_dim, "q head_dim must match k head_dim");
    AITER_CHECK(positions.size(0) >= num_tokens, "positions must cover all indexed tokens");
    AITER_CHECK(q_out.size(0) >= num_tokens && q_out.size(1) == n_heads &&
                    q_out.size(2) == (fp4_out ? head_dim / 2 : head_dim),
                "q_out must cover all indexed tokens");
    AITER_CHECK(weights.size(0) >= num_tokens && weights.size(1) == n_heads,
                "weights must cover all indexed tokens");
    AITER_CHECK(weights_out.size(0) >= num_tokens && weights_out.size(1) == n_heads,
                "weights_out must cover all indexed tokens");
    AITER_CHECK(cos_cache.size(0) == sin_cache.size(0) &&
                    cos_cache.size(1) == sin_cache.size(1),
                "cos_cache and sin_cache shapes must match");
    AITER_CHECK(max_position > 0, "cos_cache and sin_cache must not be empty");
    AITER_CHECK(cos_cache.stride(1) == 1 && sin_cache.stride(1) == 1,
                "cos_cache and sin_cache last dimension must be contiguous");
    AITER_CHECK(head_dim == 128, "indexer fused qk cache only supports head_dim=128");
    AITER_CHECK(rope_dim == 64, "indexer fused qk cache only supports rope_dim=64");
    AITER_CHECK(quant_block_size == (fp4_out ? INDEXER_FP4_GROUP_SIZE : head_dim),
                fp4_out ? "fp4 indexer fused qk cache only supports quant_block_size == 32"
                        : "indexer fused qk cache only supports quant_block_size == head_dim");
    AITER_CHECK(k.dtype() == q.dtype(), "k dtype must match q dtype");
    AITER_CHECK(weights.dtype() == q.dtype(), "weights dtype must match q dtype");
    AITER_CHECK(norm_weight.dtype() == AITER_DTYPE_fp32, "norm_weight dtype must be fp32");
    AITER_CHECK(norm_bias.dtype() == AITER_DTYPE_fp32, "norm_bias dtype must be fp32");
    AITER_CHECK(cos_cache.dtype() == q.dtype(), "cos_cache dtype must match q dtype");
    AITER_CHECK(sin_cache.dtype() == q.dtype(), "sin_cache dtype must match q dtype");
    AITER_CHECK(norm_weight.size(0) == head_dim, "norm_weight size must match head_dim");
    AITER_CHECK(norm_bias.size(0) == head_dim, "norm_bias size must match head_dim");
    AITER_CHECK(norm_weight.dim() == 1, "norm_weight must be 1D");
    AITER_CHECK(norm_bias.dim() == 1, "norm_bias must be 1D");
    AITER_CHECK(norm_weight.is_contiguous(), "norm_weight must be contiguous");
    AITER_CHECK(norm_bias.is_contiguous(), "norm_bias must be contiguous");

    uint8_t* q_scale_out_ptr   = nullptr;
    uint8_t* kv_cache_scale_ptr = nullptr;
    if(fp4_out)
    {
        aiter_tensor_t& qs = q_scale_out.value();
        aiter_tensor_t& ks = kv_cache_scale.value();
        const int k_tiles  = head_dim / 128;
        const int qs_pad   = ((n_heads / 16) + 3) & ~3;

        // opus.hpp packs fp32 -> fp4 with a gfx950 instruction and compiles that call to a
        // zero store everywhere else, so an unchecked fp4 request would silently write zeros.
        const std::string arch = get_gpu_arch();
        AITER_CHECK(arch == "gfx950", "fp4 output requires gfx950, got ", arch);
        AITER_CHECK(!preshuffle,
                    "fp4 output always writes the pa_mqa_logits_fp4 preshuffled layout; "
                    "the fp8-only preshuffle flag must be left unset");
        AITER_CHECK(scale_fmt == "ue8m0", "fp4 output requires scale_fmt=\"ue8m0\", got ", scale_fmt);
        AITER_CHECK(n_heads % 16 == 0,
                    "fp4 output requires n_heads to be a multiple of 16, got ",
                    n_heads);
        AITER_CHECK(cache_block_size == 64,
                    "fp4 output only supports kv_block_size=64 (kv_cache.size(3)), got ",
                    cache_block_size);
        AITER_CHECK(kv_cache.size(1) == k_tiles && kv_cache.size(2) == 4 &&
                        kv_cache.size(4) == 16,
                    "fp4 kv_cache must be [num_blocks, ",
                    k_tiles,
                    ", 4, 64, 16]");
        AITER_CHECK(ks.dim() == 4 && ks.size(0) == kv_cache.size(0) && ks.size(1) == k_tiles &&
                        ks.size(2) == 4 && ks.size(3) == cache_block_size,
                    "fp4 kv_cache_scale must be [num_blocks, ",
                    k_tiles,
                    ", 4, 64]");
        AITER_CHECK(qs.dim() == 5 && qs.size(0) >= num_tokens && qs.size(1) == k_tiles &&
                        qs.size(2) == 4 && qs.size(3) == 16 && qs.size(4) == qs_pad,
                    "fp4 q_scale_out must be [num_tokens, ",
                    k_tiles,
                    ", 4, 16, ",
                    qs_pad,
                    "]");
        AITER_CHECK(q.device_id == qs.device_id, "q and q_scale_out must be on the same device");
        AITER_CHECK(q.device_id == ks.device_id,
                    "q and kv_cache_scale must be on the same device");
        AITER_CHECK(q_out.dtype() == AITER_DTYPE_u8 || q_out.dtype() == AITER_DTYPE_fp4x2,
                    "fp4 q_out dtype must be u8 or fp4x2");
        AITER_CHECK(kv_cache.dtype() == AITER_DTYPE_u8 || kv_cache.dtype() == AITER_DTYPE_fp4x2,
                    "fp4 kv_cache dtype must be u8 or fp4x2");
        AITER_CHECK(qs.dtype() == AITER_DTYPE_u8 || qs.dtype() == AITER_DTYPE_fp8_e8m0,
                    "fp4 q_scale_out dtype must be u8 or fp8_e8m0");
        AITER_CHECK(ks.dtype() == AITER_DTYPE_u8 || ks.dtype() == AITER_DTYPE_fp8_e8m0,
                    "fp4 kv_cache_scale dtype must be u8 or fp8_e8m0");
        AITER_CHECK(weights_out.dtype() == q.dtype(),
                    "fp4 weights_out dtype must match q dtype");
        AITER_CHECK(kv_cache.is_contiguous(), "fp4 kv_cache must be contiguous");
        AITER_CHECK(ks.is_contiguous(), "fp4 kv_cache_scale must be contiguous");
        AITER_CHECK(qs.is_contiguous(), "fp4 q_scale_out must be contiguous");

        q_scale_out_ptr    = reinterpret_cast<uint8_t*>(qs.data_ptr());
        kv_cache_scale_ptr = reinterpret_cast<uint8_t*>(ks.data_ptr());
    }
    else
    {
        AITER_CHECK(q_out.dtype() == AITER_DTYPE_fp8, "q_out dtype must be fp8");
        AITER_CHECK(weights_out.dtype() == AITER_DTYPE_fp32, "weights_out dtype must be fp32");
        if(preshuffle)
        {
            AITER_CHECK(cache_block_size % 16 == 0,
                        "preshuffle requires cache_block_size to be a multiple of 16, got ",
                        cache_block_size);
            AITER_CHECK(head_dim % 16 == 0,
                        "preshuffle requires head_dim to be a multiple of 16, got ",
                        head_dim);
        }
    }

    // q is the only vector-accessed tensor, and the launch picks the vector width
    // from num_tokens, so the alignment it needs follows that choice. The address
    // is q + token * stride0 + head * stride1 + dim0, and dim0 is already a
    // multiple of the width, so both strides have to be too.
    const int64_t q_elem_bytes = q.dtype() == AITER_DTYPE_fp32 ? 4 : 2;
    int threads                = WARP_SIZE;
    const bool narrow          = num_tokens <= aiter::INDEXER_NARROW_MAX_TOKENS;
    const int64_t q_vec_elems  = aiter::indexer_vec_elems(
        narrow, threads, static_cast<int>(head_dim), static_cast<int>(q_elem_bytes));
    AITER_CHECK(q.stride(2) == 1, "q must be contiguous along head_dim");
    AITER_CHECK(q.stride(0) % q_vec_elems == 0 && q.stride(1) % q_vec_elems == 0,
                "q strides must be multiples of the vector width ",
                q_vec_elems,
                ", got ",
                q.stride(0),
                " and ",
                q.stride(1));
    AITER_CHECK(reinterpret_cast<uintptr_t>(q.data_ptr()) % (q_vec_elems * q_elem_bytes) == 0,
                "q must be aligned to ",
                q_vec_elems * q_elem_bytes,
                " bytes");

    // grid.y counts head groups, whose size follows the vector width.
    const int heads_per_block = aiter::indexer_heads_per_block(
        threads, static_cast<int>(head_dim), static_cast<int>(q_vec_elems));
    dim3 block(threads);
    HipDeviceGuard device_guard(q.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    float eps = static_cast<float>(epsilon);
    float w_scale = static_cast<float>(weights_scale);

    if(fp4_out)
    {
        DISPATCH_BY_KV_CACHE_DTYPE_OPUS_rmTorch(k.dtype(),
                                                "fp8_e4m3",
                                                CALL_INDEXER_QK_ROPE_QUANT_AND_CACHE_FP4);
    }
    else
    {
        DISPATCH_BY_KV_CACHE_DTYPE_OPUS_rmTorch(k.dtype(),
                                                "fp8_e4m3",
                                                CALL_INDEXER_QK_ROPE_QUANT_AND_CACHE);
    }
}

// copy from vllm: https://github.com/vllm-project/vllm/blob/main/csrc/cache_kernels.cu
void cp_gather_indexer_k_quant_cache(
    const aiter_tensor_t& kv_cache,    // [num_blocks, block_size, cache_stride]
    aiter_tensor_t& dst_k,             // [num_tokens, head_dim]
    aiter_tensor_t& dst_scale,         // [num_tokens, head_dim / quant_block_size] float
    const aiter_tensor_t& block_table, // [batch_size, num_blocks]
    const aiter_tensor_t& cu_seq_lens,  // [batch_size + 1]
    bool preshuffle)
{
    int batch_size       = block_table.size(0);
    int num_tokens       = dst_k.size(0);
    int head_dim         = dst_k.size(1);
    int quant_block_size = head_dim / (dst_scale.size(1) * dst_scale.element_size() / 4);
    bool do_preshuffle   = preshuffle;

    AITER_CHECK(kv_cache.device_id == dst_k.device_id,
                "kv_cache and dst_k must be on the same device");
    AITER_CHECK(kv_cache.device_id == dst_scale.device_id,
                "kv_cache and dst_scale must be on the same device");
    AITER_CHECK(kv_cache.device_id == block_table.device_id,
                "kv_cache and block_table must be on the same device");
    AITER_CHECK(kv_cache.device_id == cu_seq_lens.device_id,
                "kv_cache and cu_seq_lens must be on the same device");
    AITER_CHECK(head_dim % quant_block_size == 0, "head_dim must be divisible by quant_block_size");
    if(preshuffle)
    {
        int cache_block_size = kv_cache.size(1);
        AITER_CHECK(cache_block_size % 16 == 0,
                    "preshuffle requires cache_block_size to be a multiple of 16, got ",
                    cache_block_size);
        AITER_CHECK(head_dim % 16 == 0,
                    "preshuffle requires head_dim to be a multiple of 16, got ",
                    head_dim);
    }

    constexpr int vec_size = 16;
    HipDeviceGuard device_guard(kv_cache.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

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

// Heads per block for fused_qk_rope_concat_and_cache_mla_seg_kernel. Only
// worth it once there is enough work to be bandwidth-bound: below ~32
// (token, head) rows per CU the op sits on its latency floor, and there the
// shortest per-block path wins -- H=32 T=192 runs 6.34 us at one head per
// block and 6.75 us at eight. Above it, take the largest power of two up to 8
// that divides num_heads, so every block is full; 8 is also the most heads
// whose pe the block's lanes can hold (8 x PE_CHUNKS). H=128 T=768: 35.3 us at
// 1, 20.0 us at 8.
static int mla_rope_heads_per_block(int64_t num_tokens, int num_heads)
{
    if(num_tokens * num_heads < 32 * static_cast<int64_t>(get_num_cu_func()))
        return 1;
    for(int h = 8; h >= 2; h >>= 1)
        if(num_heads % h == 0)
            return h;
    return 1;
}

// Dynamic LDS of the gfx1250 TDM Q staging (see the kernel): two waves, each
// hpt x 256 nope plus min(hpt, 4) x 64 pe elements.
static size_t mla_rope_tdm_lds_bytes(int hpt, size_t elem_size)
{
    static const bool tdm_arch = get_gpu_arch() == "gfx1250";
    if(!tdm_arch || ((AITER_MLA_SEG_TDM_HPT_MASK >> hpt) & 1) == 0)
        return 0;
    return size_t(2) * (hpt * 256 + std::min(hpt, 4) * 64) * elem_size;
}

// Heads per block and K_EARLY on the flat head-group-major grid of the
// per-entry layout, fit on gfx1250 (H=16/32/128, T=1..4096) against the
// (token, head) rows per CU: one head with K early up to 32, two with K
// early up to 48, two up to 80, four up to 160, eight beyond.
static int mla_rope_entry_heads_per_block(int64_t num_tokens, int num_heads, bool& k_early)
{
    const int64_t per_cu = num_tokens * num_heads / get_num_cu_func();
    int hpt              = per_cu <= 48 ? (per_cu <= 32 ? 1 : 2) : per_cu <= 80 ? 2
                         : per_cu <= 160 ? 4 : 8;
    while(num_heads % hpt != 0)
        hpt >>= 1;
    k_early = hpt == 1 || per_cu <= 48;
    return hpt;
}

// n / d == (umulhi(n, magic) + n) >> shift for every n < 2^31.
static void mla_rope_fast_div(uint32_t d, uint32_t& magic, int& shift)
{
    shift = 0;
    while((uint64_t(1) << shift) < d)
        ++shift;
    magic = static_cast<uint32_t>(((uint64_t(1) << 32) * ((uint64_t(1) << shift) - d)) / d + 1);
}

struct MlaRopeHptArgs
{
    void *q_nope, *q_pe, *kv_c, *k_pe, *kv_cache, *q_out;
    const int64_t *slot_mapping, *positions;
    const void *cos_cache, *sin_cache;
    const float *q_scale, *k_scale;
    int num_tokens, num_heads;
    int64_t q_nope_stride_0, q_nope_stride_1, q_pe_stride_0, q_pe_stride_1;
    int64_t q_out_stride_0, q_out_stride_1, kv_c_stride, k_pe_stride;
    int64_t cos_stride0, sin_stride0, block_stride, entry_stride;
    int max_position, block_size;
    bool is_neox, is_nope_first, compute_all_q_rope;
};

// fused_qk_rope_concat_and_cache_mla on the head-grouped kernel, plain
// per-entry cache layout (PAGE_SIZE = 0).
template <typename scalar_t, typename cache_t, typename query_t>
static void launch_mla_rope_hpt(const MlaRopeHptArgs& a, hipStream_t stream)
{
    if constexpr(std::is_same_v<scalar_t, float>)
    {
        AITER_CHECK(false, "fp32 input is not supported by the head-grouped MLA kernel");
    }
    else
    {
        constexpr int KV_LORA = 512, PE_DIM = 64, VEC = 8;
        bool k_early;
        const int hpt         = mla_rope_entry_heads_per_block(a.num_tokens, a.num_heads, k_early);
        const size_t lds      = mla_rope_tdm_lds_bytes(hpt, sizeof(scalar_t));
        uint32_t tok_magic;
        int tok_shift;
        mla_rope_fast_div(static_cast<uint32_t>(a.num_tokens), tok_magic, tok_shift);
        uint32_t bs_magic;
        int bs_shift;
        mla_rope_fast_div(static_cast<uint32_t>(a.block_size), bs_magic, bs_shift);
        const dim3 grid(static_cast<unsigned>(int64_t(a.num_heads / hpt) * a.num_tokens));
        const dim3 block(KV_LORA / VEC);
#define LAUNCH_MLA_ROPE_HPT_C(NEOX, NF, HPT_, CAQR, KE)                                        \
    aiter::fused_qk_rope_concat_and_cache_mla_seg_kernel<scalar_t, cache_t, query_t, KV_LORA,  \
                                                         PE_DIM, 0, NEOX, NF, VEC, HPT_, CAQR, \
                                                         KE>                                   \
        <<<grid, block, lds, stream>>>(                                                        \
            static_cast<const scalar_t*>(a.q_nope), static_cast<const scalar_t*>(a.q_pe),      \
            static_cast<const scalar_t*>(a.kv_c), static_cast<const scalar_t*>(a.k_pe),        \
            static_cast<cache_t*>(a.kv_cache), static_cast<query_t*>(a.q_out),                 \
            a.slot_mapping, a.positions, static_cast<const scalar_t*>(a.cos_cache),            \
            static_cast<const scalar_t*>(a.sin_cache), a.q_scale, a.k_scale, a.num_heads,      \
            a.q_nope_stride_0, a.q_nope_stride_1, a.q_pe_stride_0, a.q_pe_stride_1,            \
            a.q_out_stride_0, a.q_out_stride_1, a.kv_c_stride, a.k_pe_stride, a.cos_stride0,   \
            a.sin_stride0, a.block_stride, a.max_position, a.entry_stride, a.block_size,       \
            a.num_tokens, tok_magic, tok_shift, bs_magic, bs_shift)
#define LAUNCH_MLA_ROPE_HPT(NEOX, NF, HPT_, KE)                  \
    do                                                           \
    {                                                            \
        if(a.compute_all_q_rope)                                 \
            LAUNCH_MLA_ROPE_HPT_C(NEOX, NF, HPT_, true, KE);     \
        else                                                     \
            LAUNCH_MLA_ROPE_HPT_C(NEOX, NF, HPT_, false, KE);    \
    } while(0)
#define LAUNCH_MLA_ROPE_HPT_H(NEOX, NF)                                  \
    switch(hpt)                                                          \
    {                                                                    \
    case 8: LAUNCH_MLA_ROPE_HPT(NEOX, NF, 8, false); break;              \
    case 4: LAUNCH_MLA_ROPE_HPT(NEOX, NF, 4, false); break;              \
    case 2:                                                              \
        if(k_early)                                                      \
            LAUNCH_MLA_ROPE_HPT(NEOX, NF, 2, true);                      \
        else                                                             \
            LAUNCH_MLA_ROPE_HPT(NEOX, NF, 2, false);                     \
        break;                                                           \
    default: LAUNCH_MLA_ROPE_HPT(NEOX, NF, 1, true); break;              \
    }
        if(a.is_neox && a.is_nope_first)
            LAUNCH_MLA_ROPE_HPT_H(true, true)
        else if(a.is_neox)
            LAUNCH_MLA_ROPE_HPT_H(true, false)
        else if(a.is_nope_first)
            LAUNCH_MLA_ROPE_HPT_H(false, true)
        else
            LAUNCH_MLA_ROPE_HPT_H(false, false)
#undef LAUNCH_MLA_ROPE_HPT_H
#undef LAUNCH_MLA_ROPE_HPT
#undef LAUNCH_MLA_ROPE_HPT_C
    }
}

void fused_qk_rope_concat_and_cache_mla(
    aiter_tensor_t& q_nope,        // [num_tokens, num_heads, qk_lora_rank]
    aiter_tensor_t& q_pe,          // [num_tokens, num_heads, pe_dim]
    aiter_tensor_t& kv_c,          // [num_tokens, k_num_heads, kv_lora_rank] or [num_tokens, kv_lora_rank]
    aiter_tensor_t& k_pe,          // [num_tokens, k_num_heads, pe_dim] or [num_tokens, pe_dim]
    aiter_tensor_t& kv_cache,      // [num_blocks, block_size, (kv_lora_rank +
                                  // pe_dim)] or [num_blocks, block_size, k_num_heads, kv_lora_rank + pe_dim)]
    aiter_tensor_t& q_out,        // [num_tokens, num_heads, qk_lora_rank+pe_dim]
    aiter_tensor_t& slot_mapping,  // [num_tokens] or [num_actual_tokens]
    aiter_tensor_t& k_scale,   // scale for k
    aiter_tensor_t& q_scale,   // scale for q
    aiter_tensor_t& positions, // [num_tokens]
    aiter_tensor_t &cos_cache, // [max_position, rot_dim//2]
    aiter_tensor_t &sin_cache, // [max_position, rot_dim//2]
    bool is_neox, bool is_nope_first,
    bool compute_all_q_rope
) {
  int num_tokens = slot_mapping.size(0);
  int kv_lora_rank = kv_c.size(-1);
  int pe_dim = k_pe.size(-1);
  int block_size = kv_cache.size(1);
  int num_heads = q_nope.size(1);
  int qk_lora_rank = q_nope.size(-1);
  int rot_dim = cos_cache.size(-1) * 2;
  int num_blocks = (num_tokens + block_size - 1) / block_size;
  int num_actual_tokens = slot_mapping.size(0);
  int num_slots = slot_mapping.size(0);
  // Upper bound for the defensive pos clamp inside the decode kernels (padded /
  // cudagraph tokens with slot_idx < 0 may carry a stale position).
  const int max_position = cos_cache.size(0);

  AITER_CHECK(q_nope.dim() == q_pe.dim());
  AITER_CHECK(q_nope.size(1) == q_pe.size(1));

  AITER_CHECK(q_out.size(2) == qk_lora_rank + pe_dim);
  AITER_CHECK(kv_lora_rank == qk_lora_rank, "kv_lora_rank and qk_lora_rank must be the same");
  int kv_c_stride = kv_c.stride(0);
  int k_pe_stride = k_pe.stride(0);
  int q_nope_stride_0 = q_nope.stride(0);
  int q_pe_stride_0 = q_pe.stride(0);
  int q_out_stride_0 = q_out.stride(0);
  int q_nope_stride_1 = q_nope.stride(1);
  int q_pe_stride_1 = q_pe.stride(1);
  int q_out_stride_1 = q_out.stride(1);
  int block_stride = kv_cache.stride(0);
  int entry_stride = kv_cache.stride(1);
  HipDeviceGuard device_guard(kv_c.device_id);
  // device_guard1 for q_out removed (same device as prior guard)
  const hipStream_t stream = aiter::getCurrentHIPStream();

  std::string q_out_type = "auto";
  std::string kv_cache_dtype = "auto";
  if (kv_cache.dtype() == AITER_DTYPE_fp32 ||
             kv_cache.dtype() == AITER_DTYPE_fp16  ||
             kv_cache.dtype() == AITER_DTYPE_bf16) {
    kv_cache_dtype = "auto";
  } else if(kv_cache.dtype() == AITER_DTYPE_fp8 || 
              kv_cache.dtype() == AITER_DTYPE_fp8 || 
              kv_cache.dtype() ==AITER_DTYPE_fp8) {
    kv_cache_dtype = "fp8";
  } else{
    AITER_CHECK(false, "kv cache data type is not supported");
  }
  if (q_out.dtype() == kv_cache.dtype()) {
    q_out_type = kv_cache_dtype;
  } else if (q_out.dtype() == AITER_DTYPE_fp32 ||
             q_out.dtype() == AITER_DTYPE_fp16  ||
             q_out.dtype() == AITER_DTYPE_bf16) {
    q_out_type = "auto";
  } else if(q_out.dtype() == AITER_DTYPE_fp8 || 
            q_out.dtype() == AITER_DTYPE_fp8 || 
            q_out.dtype() ==AITER_DTYPE_fp8) {
    q_out_type = "fp8";
  } else{
    AITER_CHECK(false, "kv cache data type is not supported");
  }
  if (kv_cache_dtype == "auto" && q_out_type == "fp8") {
    AITER_CHECK(false, "kv cache data type is auto and q_out data type is fp8, which is not supported");
  }
  AITER_CHECK(kv_c.stride(-1) == 1, "kv_c stride(-1) must be equal to 1");
  AITER_CHECK(k_pe.stride(-1) == 1, "k_pe stride(-1) must be equal to 1");

  // DeepSeek MLA decode (kv_lora 512, rope 64, one kv head, 16-bit input):
  // the head-grouped kernel shared with the segmented cache path. It loads
  // and stores 16 B / VEC-element vectors and addresses Q with int32 offsets
  // per token.
  {
    const bool one_kv_head = (kv_c.dim() == 2 && k_pe.dim() == 2) ||
                             (kv_c.dim() == 3 && k_pe.dim() == 3 && kv_c.size(1) == 1 &&
                              k_pe.size(1) == 1);
    const auto in_dt = kv_c.dtype();
    auto al16 = [](const aiter_tensor_t& t) {
      return reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0;
    };
    const int64_t vec_strides[] = {q_nope.stride(0), q_nope.stride(1), q_pe.stride(0),
                                   q_pe.stride(1),   q_out.stride(0),  q_out.stride(1),
                                   kv_c.stride(0),   k_pe.stride(0),   cos_cache.stride(0),
                                   sin_cache.stride(0), kv_cache.stride(0), kv_cache.stride(1)};
    bool strides_ok = true;
    for (int64_t s : vec_strides)
      strides_ok = strides_ok && s % 8 == 0;
    const int64_t cs_span = cos_cache.size(0) * std::max(cos_cache.stride(0), sin_cache.stride(0)) *
                            cos_cache.element_size();
    const int64_t q_span = (num_heads - 1) * std::max({q_nope.stride(1), q_pe.stride(1),
                                                       q_out.stride(1)}) + kv_lora_rank + pe_dim;
    const bool use_hpt =
        one_kv_head && kv_lora_rank == 512 &&
        qk_lora_rank == 512 && pe_dim == 64 && rot_dim == 64 && q_pe.size(-1) == 64 &&
        (in_dt == AITER_DTYPE_bf16 || in_dt == AITER_DTYPE_fp16) && q_nope.dtype() == in_dt &&
        q_pe.dtype() == in_dt && k_pe.dtype() == in_dt && cos_cache.dtype() == in_dt &&
        sin_cache.dtype() == in_dt && q_nope.stride(-1) == 1 && q_pe.stride(-1) == 1 &&
        q_out.stride(-1) == 1 && sin_cache.size(0) == cos_cache.size(0) &&
        strides_ok &&
        al16(q_nope) && al16(q_pe) && al16(kv_c) && al16(k_pe) && al16(q_out) && al16(kv_cache) &&
        al16(cos_cache) && al16(sin_cache) && num_tokens > 0 &&
        static_cast<int64_t>(num_tokens) * num_heads < (int64_t(1) << 31) &&
        max_position > 0 && cs_span <= std::numeric_limits<int32_t>::max() &&
        q_span * 2 <= std::numeric_limits<int32_t>::max();
    if (use_hpt) {
      MlaRopeHptArgs a{q_nope.data_ptr(), q_pe.data_ptr(), kv_c.data_ptr(), k_pe.data_ptr(),
                       kv_cache.data_ptr(), q_out.data_ptr(),
                       reinterpret_cast<const int64_t*>(slot_mapping.data_ptr()),
                       reinterpret_cast<const int64_t*>(positions.data_ptr()),
                       cos_cache.data_ptr(), sin_cache.data_ptr(),
                       reinterpret_cast<const float*>(q_scale.data_ptr()),
                       reinterpret_cast<const float*>(k_scale.data_ptr()),
                       num_tokens, num_heads,
                       q_nope.stride(0), q_nope.stride(1), q_pe.stride(0), q_pe.stride(1),
                       q_out.stride(0), q_out.stride(1), kv_c.stride(0), k_pe.stride(0),
                       cos_cache.stride(0), sin_cache.stride(0),
                       kv_cache.stride(0), kv_cache.stride(1),
                       max_position, block_size,
                       is_neox, is_nope_first, compute_all_q_rope};
#define CALL_MLA_ROPE_HPT(KV_T, CACHE_T, QUERY_T, KV_DTYPE, Q_DTYPE) \
      launch_mla_rope_hpt<KV_T, CACHE_T, QUERY_T>(a, stream);
      DISPATCH_BY_KV_CACHE_QUERY_DTYPE_OPUS_rmTorch(in_dt, kv_cache_dtype, q_out_type,
                                                    CALL_MLA_ROPE_HPT);
#undef CALL_MLA_ROPE_HPT
      return;
    }
  }
  // ============================================================================
  // Kernel Dispatch Logic
  // ============================================================================
  
  // Configuration constants for kernel selection
  constexpr int64_t OPTIMIZED_KV_LORA_RANK = 512;
  constexpr int64_t OPTIMIZED_ROT_DIM = 64;
  constexpr int64_t OPTIMIZED_BLOCK_SIZE = 256;
  constexpr int64_t MAX_TOKENS_PER_HEAD = 256;
  constexpr int64_t MAX_BLOCK_THREADS = 2048;
  constexpr int64_t MIN_SIZE_FOR_OPT = 2048;
  
  // Determine if this is a decode or prefill scenario
  const bool is_decode = (kv_c.dim() == 2 && k_pe.dim() == 2);
  
  // For decode with single kv_head (dim=3 with size=1), check stride continuity
  // Stride(1) should equal size(2) for contiguous storage
  const bool kv_c_contiguous = (kv_c.dim() == 3 && kv_c.size(1) == 1) ? 
                                (kv_c.stride(1) == kv_c.size(2)) : true;
  const bool k_pe_contiguous = (k_pe.dim() == 3 && k_pe.size(1) == 1) ? 
                                (k_pe.stride(1) == k_pe.size(2)) : true;
  
  const bool is_decode_single_kv_head = (kv_c.dim() == 3 && k_pe.dim() == 3 && 
                                          kv_c.size(1) == 1 && k_pe.size(1) == 1 &&
                                          kv_c_contiguous && k_pe_contiguous);
  
  const bool is_prefill_gqa = (kv_c.dim() == 3 && k_pe.dim() == 3 &&
                               kv_c.size(1) > 1);

  // compute_all_q_rope (DCP: RoPE all queries incl. padded slot=-1) is
  // implemented ONLY in the per-head (option 1) and opt (option 2) decode
  // kernels. Every other dispatch target -- the general decode kernel (option
  // 3) and the prefill kernels -- still hard-returns on slot_idx < 0 and would
  // silently drop padded-token q_out. Reject any config that would land there
  // instead of computing wrong results. Conditions mirror the dispatch below.
  if (compute_all_q_rope) {
    const bool on_decode_path = is_decode || is_decode_single_kv_head;
    const bool hits_per_head =
        is_nope_first && kv_lora_rank <= OPTIMIZED_KV_LORA_RANK &&
        block_size == 1 && rot_dim == OPTIMIZED_ROT_DIM &&
        num_tokens < MAX_TOKENS_PER_HEAD;
    const bool hits_opt =
        rot_dim == OPTIMIZED_ROT_DIM &&
        kv_lora_rank * num_heads >= MIN_SIZE_FOR_OPT &&
        kv_lora_rank == OPTIMIZED_KV_LORA_RANK;
    AITER_CHECK(on_decode_path && (hits_per_head || hits_opt),
                "compute_all_q_rope is only supported by the per-head/opt MLA decode "
                "kernels: decode path with kv_lora_rank=512 and rot_dim=64, and either "
                "num_tokens<256 (per-head) or kv_lora_rank*num_heads>=2048 (opt). "
                "Other dispatch targets (general decode / prefill) do not honor the flag.");
  }
  // ============================================================================
  // DECODE PATH (per-token processing)
  // ============================================================================
  if (is_decode || is_decode_single_kv_head) {
    
    // Option 1: Per-head kernel for small batches with standard config
    // Best for: low latency, small batch decode
    const bool use_per_head_kernel = (
      is_nope_first && 
      kv_lora_rank <= OPTIMIZED_KV_LORA_RANK && 
      block_size == 1 &&
      rot_dim == OPTIMIZED_ROT_DIM && 
      num_tokens < MAX_TOKENS_PER_HEAD
    );
    
    if (use_per_head_kernel) {
      // Launch one block per (token, head) pair
      dim3 grid(num_tokens * num_heads);
      
      // Determine vec_size: float must use 4, half/bfloat16 can use 8 for large tensors
      const bool is_float = (kv_c.dtype() == AITER_DTYPE_fp32);
      const bool use_vec4 = is_float || (kv_lora_rank >= 64 && kv_lora_rank <= 128);
      
      if (use_vec4) {
        constexpr int vec_size = 4;
        dim3 block(std::min<int64_t>(kv_lora_rank, OPTIMIZED_KV_LORA_RANK) / vec_size);
        #define CALL_OPT_VEC4(KV_T, CACHE_T, QUERY_T, KV_DTYPE, Q_DTYPE) \
          CALL_FUSED_QK_ROPE_CONCAT_AND_CACHE_MLA_OPT(KV_T, CACHE_T, QUERY_T, KV_DTYPE, Q_DTYPE, 4)
        DISPATCH_BY_KV_CACHE_QUERY_DTYPE_OPUS_rmTorch(kv_c.dtype(), kv_cache_dtype, q_out_type, CALL_OPT_VEC4);
        #undef CALL_OPT_VEC4
      } else {
        // Only half/bfloat16 with kv_lora_rank > 128 use vec_size=8
        constexpr int vec_size = 8;
        dim3 block(std::min<int64_t>(kv_lora_rank, OPTIMIZED_KV_LORA_RANK) / vec_size);
        #define CALL_OPT_VEC8(KV_T, CACHE_T, QUERY_T, KV_DTYPE, Q_DTYPE) \
          CALL_FUSED_QK_ROPE_CONCAT_AND_CACHE_MLA_OPT(KV_T, CACHE_T, QUERY_T, KV_DTYPE, Q_DTYPE, 8)
        DISPATCH_BY_KV_CACHE_QUERY_DTYPE_OPUS_rmTorch(kv_c.dtype(), kv_cache_dtype, q_out_type, CALL_OPT_VEC8);
        #undef CALL_OPT_VEC8
      }
    }
    // Option 2: Optimized decode kernel for standard config with large workload
    // Best for: high throughput, standard DeepSeek config
    else if (rot_dim == OPTIMIZED_ROT_DIM && 
             kv_lora_rank * num_heads >= MIN_SIZE_FOR_OPT && 
             kv_lora_rank == OPTIMIZED_KV_LORA_RANK) {
      // Launch one block per token, process all heads together
      dim3 grid(num_tokens);
      dim3 block(OPTIMIZED_BLOCK_SIZE);
      
      DISPATCH_BY_KV_CACHE_QUERY_DTYPE_OPUS_rmTorch(kv_c.dtype(), kv_cache_dtype, q_out_type,
                                        CALL_FUSED_QK_ROPE_CONCAT_AND_CACHE_MLA);
    }
    // Option 3: General decode kernel for arbitrary configs
    // Best for: custom models, variable dimensions
    else {
      // For decode path, we need to set up GQA-style strides even for single kv_head
      // Treat as if kv_c and k_pe have an extra dimension with size 1
      const int kv_c_stride_0 = kv_c.stride(0);
      const int kv_c_stride_1 = (kv_c.dim() == 3) ? kv_c.stride(1) : 0;  // 0 for dim=2
      const int k_pe_stride_0 = k_pe.stride(0);
      const int k_pe_stride_1 = (k_pe.dim() == 3) ? k_pe.stride(1) : 0;  // 0 for dim=2
      const int num_kv_heads = (kv_c.dim() == 3) ? kv_c.size(1) : 1;     // 1 for dim=2
      const int kv_cache_stride_h = (kv_cache.dim() >= 3) ? kv_cache.stride(2) : (kv_lora_rank + pe_dim);
      // Dynamic block size based on workload
      dim3 grid(num_tokens);
      dim3 block(std::min<int64_t>(kv_lora_rank * num_heads, MAX_BLOCK_THREADS) / 8);
      
      DISPATCH_BY_KV_CACHE_QUERY_DTYPE_OPUS_rmTorch(kv_c.dtype(), kv_cache_dtype, q_out_type,
                                        CALL_FUSED_QK_ROPE_CONCAT_AND_CACHE_MLA_GENERAL);
    }
  }
  
  // ============================================================================
  // PREFILL PATH (batched processing with GQA)
  // ============================================================================
  else if (is_prefill_gqa) {
    // Extract GQA-specific strides
    const int kv_c_stride_0 = kv_c.stride(0);
    const int kv_c_stride_1 = kv_c.stride(1);
    const int k_pe_stride_0 = k_pe.stride(0);
    const int k_pe_stride_1 = k_pe.stride(1);
    const int num_kv_heads = kv_c.size(1);
    AITER_CHECK(num_kv_heads <= num_heads, "num_kv_heads must be less than or equal to num_heads");
    const int kv_cache_stride_h = kv_cache.stride(2);
    
    // Option 1: Optimized prefill kernel for standard config
    // Best for: DeepSeek-V2/V3 prefill phase
    const bool use_optimized_prefill = (
      rot_dim == OPTIMIZED_ROT_DIM && 
      kv_lora_rank == OPTIMIZED_KV_LORA_RANK
    );
    
    if (use_optimized_prefill) {
      dim3 grid(num_tokens);
      dim3 block(OPTIMIZED_BLOCK_SIZE);
      DISPATCH_BY_KV_CACHE_QUERY_DTYPE_OPUS_rmTorch(kv_c.dtype(), kv_cache_dtype, q_out_type,
                                        CALL_PREFILL_FUSED_QK_ROPE_CONCAT_AND_CACHE_MLA);
    }
    // Option 2: General prefill kernel for arbitrary configs
    // Best for: custom models, variable dimensions, different GQA ratios
    else {
      dim3 grid(num_tokens);

      dim3 block(std::min<int64_t>(kv_lora_rank * num_heads, MAX_BLOCK_THREADS) / 8);
      DISPATCH_BY_KV_CACHE_QUERY_DTYPE_OPUS_rmTorch(kv_c.dtype(), kv_cache_dtype, q_out_type,
                                        CALL_FUSED_QK_ROPE_CONCAT_AND_CACHE_MLA_GENERAL);
    }
  }
  else {
    AITER_CHECK(false,
                "Unsupported tensor dimensions: kv_c.dim()=", kv_c.dim(),
                ", k_pe.dim()=", k_pe.dim(),
                ". Expected either decode (dim=2) or prefill with GQA (dim=3).");
  }
}

// ============================================================================
// DeepSeek V3.1 MLA: fused QK RoPE + static FP8 quant + segmented paged KV
// cache write (no RMSNorm). See kernel comment for layout.
// ============================================================================
void fused_qk_rope_concat_and_cache_mla_seg(
    aiter_tensor_t& q_nope,       // [T, H, kv_lora_rank]
    aiter_tensor_t& q_pe,         // [T, H, pe_dim]
    aiter_tensor_t& kv_c,         // [T, kv_lora_rank]
    aiter_tensor_t& k_pe,         // [T, pe_dim]
    aiter_tensor_t& kv_cache,     // [num_blocks, page_size*kv_lora + page_size*pe] flat
    aiter_tensor_t& q_out,        // [T, H, q_out_dim] (>= kv_lora+pe; tail untouched)
    aiter_tensor_t& slot_mapping, // [T]
    aiter_tensor_t& k_scale,      // [1] fp32
    aiter_tensor_t& q_scale,      // [1] fp32
    aiter_tensor_t& positions,    // [T]
    aiter_tensor_t& cos_cache,    // [max_pos, pe_dim/2]
    aiter_tensor_t& sin_cache,    // [max_pos, pe_dim/2]
    bool is_neox,
    bool is_nope_first)
{
    AITER_CHECK(is_nope_first, "is_nope_first=false is not supported yet");

    constexpr int KV_LORA   = 512;
    constexpr int PE_DIM    = 64;
    constexpr int PAGE_SIZE = 64;

    const int num_tokens = slot_mapping.size(0);
    const int num_heads  = q_nope.size(1);

    AITER_CHECK(kv_c.size(-1) == KV_LORA, "kv_c last dim must be ", KV_LORA);
    AITER_CHECK(q_nope.size(-1) == KV_LORA, "q_nope last dim must be ", KV_LORA);
    AITER_CHECK(q_pe.size(-1) == PE_DIM && k_pe.size(-1) == PE_DIM,
                "q_pe/k_pe last dim must be ", PE_DIM);
    AITER_CHECK(q_out.size(-1) >= KV_LORA + PE_DIM,
                "q_out last dim must be >= ", KV_LORA + PE_DIM);
    AITER_CHECK(cos_cache.size(-1) == PE_DIM / 2 && sin_cache.size(-1) == PE_DIM / 2,
                "cos/sin cache last dim must be ", PE_DIM / 2);
    AITER_CHECK(q_out.dtype() == AITER_DTYPE_fp8, "q_out must be fp8");
    AITER_CHECK(kv_cache.dtype() == AITER_DTYPE_fp8, "kv_cache must be fp8");
    AITER_CHECK(q_scale.dtype() == AITER_DTYPE_fp32 && k_scale.dtype() == AITER_DTYPE_fp32,
                "q_scale/k_scale must be fp32");
    AITER_CHECK(kv_c.stride(-1) == 1 && k_pe.stride(-1) == 1,
                "kv_c/k_pe must be contiguous in last dim");

    const int64_t q_nope_stride_0 = q_nope.stride(0);
    const int64_t q_nope_stride_1 = q_nope.stride(1);
    const int64_t q_pe_stride_0   = q_pe.stride(0);
    const int64_t q_pe_stride_1   = q_pe.stride(1);
    const int64_t q_out_stride_0  = q_out.stride(0);
    const int64_t q_out_stride_1  = q_out.stride(1);
    const int64_t kv_c_stride     = kv_c.stride(0);
    const int64_t k_pe_stride     = k_pe.stride(0);
    const int64_t cos_stride0     = cos_cache.stride(0);
    const int64_t sin_stride0     = sin_cache.stride(0);
    const int64_t block_stride    = kv_cache.stride(0);
    const int max_position        = std::min(cos_cache.size(0), sin_cache.size(0));
    AITER_CHECK(max_position > 0, "cos_cache and sin_cache must not be empty");
    // The kernel reads cos/sin through one buffer descriptor per cache.
    AITER_CHECK(max_position * std::max(cos_stride0, sin_stride0) * cos_cache.element_size() <=
                    std::numeric_limits<int32_t>::max(),
                "cos/sin cache too large for a buffer descriptor");

    HipDeviceGuard device_guard(kv_c.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    // 16-bit inputs (fp16/bf16) use 128-bit vectorized loads/stores (VEC=8).
    constexpr int VEC = 8;
    static_assert(KV_LORA % VEC == 0, "KV_LORA must be divisible by VEC");

    const int hpt              = mla_rope_heads_per_block(num_tokens, num_heads);
    const size_t tdm_lds_bytes = mla_rope_tdm_lds_bytes(hpt, q_nope.element_size());

    // The kernel indexes nope vectors by threadIdx.x, one per thread.
    dim3 grid((unsigned)(num_heads / hpt), (unsigned)num_tokens);
    dim3 block(KV_LORA / VEC);

#define LAUNCH_MLA_NORM_ROPE(SCALAR_T, NEOX, HPT_)                                        \
    aiter::fused_qk_rope_concat_and_cache_mla_seg_kernel<SCALAR_T, opus::fp8_t,           \
                                                         opus::fp8_t, KV_LORA, PE_DIM,    \
                                                         PAGE_SIZE, NEOX, true, VEC,      \
                                                         HPT_>                            \
        <<<grid, block, tdm_lds_bytes, stream>>>(                                         \
            reinterpret_cast<SCALAR_T*>(q_nope.data_ptr()),                               \
            reinterpret_cast<SCALAR_T*>(q_pe.data_ptr()),                                 \
            reinterpret_cast<SCALAR_T*>(kv_c.data_ptr()),                                 \
            reinterpret_cast<SCALAR_T*>(k_pe.data_ptr()),                                 \
            reinterpret_cast<opus::fp8_t*>(kv_cache.data_ptr()),                          \
            reinterpret_cast<opus::fp8_t*>(q_out.data_ptr()),                             \
            reinterpret_cast<int64_t*>(slot_mapping.data_ptr()),                          \
            reinterpret_cast<int64_t*>(positions.data_ptr()),                             \
            reinterpret_cast<SCALAR_T*>(cos_cache.data_ptr()),                            \
            reinterpret_cast<SCALAR_T*>(sin_cache.data_ptr()),                            \
            reinterpret_cast<float*>(q_scale.data_ptr()),                                 \
            reinterpret_cast<float*>(k_scale.data_ptr()),                                 \
            num_heads,                                                                    \
            q_nope_stride_0, q_nope_stride_1,                                             \
            q_pe_stride_0, q_pe_stride_1,                                                 \
            q_out_stride_0, q_out_stride_1,                                               \
            kv_c_stride, k_pe_stride,                                                     \
            cos_stride0, sin_stride0,                                                     \
            block_stride, max_position);

#define LAUNCH_MLA_NORM_ROPE_HPT(SCALAR_T, NEOX)                                          \
    switch(hpt)                                                                           \
    {                                                                                     \
    case 8: LAUNCH_MLA_NORM_ROPE(SCALAR_T, NEOX, 8); break;                               \
    case 4: LAUNCH_MLA_NORM_ROPE(SCALAR_T, NEOX, 4); break;                               \
    case 2: LAUNCH_MLA_NORM_ROPE(SCALAR_T, NEOX, 2); break;                               \
    default: LAUNCH_MLA_NORM_ROPE(SCALAR_T, NEOX, 1); break;                              \
    }

    AITER_DISPATCH_FLOATING16_TYPES_rmTorch(
        q_nope.dtype(), "fused_qk_rope_concat_and_cache_mla_seg", [&] {
            using input_dtype = typename aiter::hip2opus<scalar_t>::type;
            if(is_neox)
            {
                LAUNCH_MLA_NORM_ROPE_HPT(input_dtype, true);
            }
            else
            {
                LAUNCH_MLA_NORM_ROPE_HPT(input_dtype, false);
            }
        });
#undef LAUNCH_MLA_NORM_ROPE_HPT
#undef LAUNCH_MLA_NORM_ROPE
}

} // namespace aiter
