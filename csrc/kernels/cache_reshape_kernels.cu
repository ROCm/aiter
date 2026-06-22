// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Dedicated, CK-free translation unit for the paged KV-cache *write* ops
// (reshape_and_cache / reshape_and_cache_flash).
//
// These two kernels are byte-for-byte identical to the ones in
// cache_kernels.cu; they are split into their own module
// ("module_cache_reshape") so the KV-cache write builds and runs on any
// architecture -- including gfx1201 / gfx1200 (RDNA4) -- without dragging in
// the CK/ck_tile-heavy and CDNA-flavored kernels (MLA concat, DeepSeek
// indexer, inline GCN asm) that also live in cache_kernels.cu and force that
// TU onto the CK-only prebuilt path. The kernel logic here uses only the
// opus:: dtype/dispatch layer (RDNA4-ready) and the ck_tile_shim fallback in
// aiter_hip_common.h, so it compiles with or without composable_kernel.

#include "aiter_dispatch.h"
#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "cache.h"

#include "aiter_opus_plus.h"
#include "attention_dtypes.h"
#include "opus/opus.hpp"

#include <algorithm>
#include <optional>
#include <string>

#include <hip/hip_bf16.h>

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
            key_cache[tgt_key_idx] =
                opus::cast<cache_t>(static_cast<float>(tgt_key) * inverted_kscale);
            value_cache[tgt_value_idx] =
                opus::cast<cache_t>(static_cast<float>(tgt_value) * inverted_vscale);
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
        scalar_t tgt_key                = key[src_key_idx];
        scalar_t tgt_value              = value[src_value_idx];
        if constexpr(kv_dt == vllm::Fp8KVCacheDataType::kAuto)
        {
            key_cache[tgt_key_value_idx]   = tgt_key;
            value_cache[tgt_key_value_idx] = tgt_value;
        }
        else
        {
            key_cache[tgt_key_value_idx] =
                opus::cast<cache_t>(static_cast<float>(tgt_key) * inverted_kscale);
            value_cache[tgt_key_value_idx] =
                opus::cast<cache_t>(static_cast<float>(tgt_value) * inverted_vscale);
        }
    }
}

} // namespace aiter

// KV_T is the stored data type of kv-cache.
// CACHE_T is the data type of key and value tensors.
// KV_DTYPE is the real data type of kv-cache.
#define CALL_RESHAPE_AND_CACHE(KV_T, CACHE_T, KV_DTYPE)                                   \
    aiter::reshape_and_cache_kernel<KV_T, CACHE_T, KV_DTYPE><<<grid, block, 0, stream>>>( \
        reinterpret_cast<KV_T*>(key.data_ptr()),                                          \
        reinterpret_cast<KV_T*>(value.data_ptr()),                                        \
        reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),                                 \
        reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),                               \
        reinterpret_cast<int64_t*>(slot_mapping.data_ptr()),                              \
        key_stride,                                                                       \
        value_stride,                                                                     \
        num_heads,                                                                        \
        head_size,                                                                        \
        block_size,                                                                       \
        x,                                                                                \
        k_scale.has_value() ? reinterpret_cast<float*>(k_scale->data_ptr()) : nullptr,    \
        v_scale.has_value() ? reinterpret_cast<float*>(v_scale->data_ptr()) : nullptr);

#define CALL_RESHAPE_AND_CACHE_ASM(KV_T, CACHE_T, KV_DTYPE)                                     \
    aiter::reshape_and_cache_kernel<KV_T, CACHE_T, KV_DTYPE, true><<<grid, block, 0, stream>>>( \
        reinterpret_cast<KV_T*>(key.data_ptr()),                                                \
        reinterpret_cast<KV_T*>(value.data_ptr()),                                              \
        reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),                                       \
        reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),                                     \
        reinterpret_cast<int64_t*>(slot_mapping.data_ptr()),                                    \
        key_stride,                                                                             \
        value_stride,                                                                           \
        num_heads,                                                                              \
        head_size,                                                                              \
        block_size,                                                                             \
        x,                                                                                      \
        k_scale.has_value() ? reinterpret_cast<float*>(k_scale->data_ptr()) : nullptr,          \
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

    if(asm_layout)
    {
        DISPATCH_BY_KV_CACHE_DTYPE_OPUS_rmTorch(
            key.dtype(), kv_cache_dtype, CALL_RESHAPE_AND_CACHE_ASM)
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
#define CALL_RESHAPE_AND_CACHE_FLASH(KV_T, CACHE_T, KV_DTYPE)                             \
    aiter::reshape_and_cache_flash_kernel<KV_T, CACHE_T, KV_DTYPE>                        \
        <<<grid, block, 0, stream>>>(reinterpret_cast<KV_T*>(key.data_ptr()),             \
                                     reinterpret_cast<KV_T*>(value.data_ptr()),           \
                                     reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),    \
                                     reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),  \
                                     reinterpret_cast<int64_t*>(slot_mapping.data_ptr()), \
                                     block_stride,                                        \
                                     key_stride,                                          \
                                     value_stride,                                        \
                                     num_heads,                                           \
                                     head_size,                                           \
                                     block_size,                                          \
                                     reinterpret_cast<float*>(k_scale.data_ptr()),        \
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

    DISPATCH_BY_KV_CACHE_DTYPE_OPUS_rmTorch(
        key.dtype(), kv_cache_dtype, CALL_RESHAPE_AND_CACHE_FLASH);
}
} // namespace aiter
