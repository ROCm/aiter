// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Horizontally fused Q/K RMSNorm + partial NeoX RoPE, with optional
// sparse-layer KV/index-cache insert. This ports the AME/vLLM operator into
// aiter's pybind/JIT extension style.

#include "fused_qknorm_idxrqknorm.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "aiter_dispatch.h"
#include "aiter_stream.h"
#include "opus/opus.hpp"
#include "quant_utils.cuh"

namespace aiter {
namespace fused_qknorm_idxrqknorm_ops {

constexpr int kHeadDim = 128;
constexpr int kNumLanes = 32;
constexpr int kElemsPerLane = kHeadDim / kNumLanes;

__device__ __forceinline__ float warpReduceSum(float val)
{
#pragma unroll
    for(int mask = 16; mask > 0; mask >>= 1)
    {
        val += __shfl_xor(val, mask, 32);
    }
    return val;
}

template <typename scalar_t>
__device__ __forceinline__ void loadElems(const scalar_t* __restrict__ src,
                                          float (&elems)[kElemsPerLane])
{
#pragma unroll
    for(int i = 0; i < kElemsPerLane; ++i)
    {
        elems[i] = static_cast<float>(src[i]);
    }
}

template <typename scalar_t>
__device__ __forceinline__ void storeElems(scalar_t* __restrict__ dst,
                                           const float (&elems)[kElemsPerLane])
{
#pragma unroll
    for(int i = 0; i < kElemsPerLane; ++i)
    {
        dst[i] = static_cast<scalar_t>(elems[i]);
    }
}

template <typename scalar_t>
__device__ __forceinline__ void copyRawElems(const scalar_t* __restrict__ src,
                                             scalar_t* __restrict__ dst)
{
    *reinterpret_cast<uint2*>(dst) = *reinterpret_cast<const uint2*>(src);
}

template <typename scalar_t, typename cache_t, vllm::Fp8KVCacheDataType kv_dt>
__device__ __forceinline__ void storeCacheElems(cache_t* __restrict__ dst,
                                                const float (&elems)[kElemsPerLane],
                                                float scale)
{
#pragma unroll
    for(int i = 0; i < kElemsPerLane; ++i)
    {
        if constexpr(kv_dt == vllm::Fp8KVCacheDataType::kAuto)
        {
            dst[i] = static_cast<cache_t>(elems[i]);
        }
        else
        {
            const scalar_t rounded = static_cast<scalar_t>(elems[i]);
            dst[i] = opus::cast<cache_t>(static_cast<float>(rounded) / scale);
        }
    }
}

template <typename scalar_t, typename cache_t, vllm::Fp8KVCacheDataType kv_dt>
__device__ __forceinline__ void copyOrQuantizeCacheElems(const scalar_t* __restrict__ src,
                                                         cache_t* __restrict__ dst,
                                                         float scale)
{
    if constexpr(kv_dt == vllm::Fp8KVCacheDataType::kAuto)
    {
        copyRawElems(src, dst);
    }
    else
    {
        float elems[kElemsPerLane];
        loadElems(src, elems);
        storeCacheElems<scalar_t, cache_t, kv_dt>(dst, elems, scale);
    }
}

template <typename scalar_t>
__device__ __forceinline__ void normAndRope(
    float (&elems)[kElemsPerLane],
    int lane_id,
    float eps,
    const scalar_t* __restrict__ weight,
    bool do_rope,
    int rotary_dim,
    const scalar_t* __restrict__ cos_ptr,
    bool apply_norm)
{
    if(apply_norm)
    {
        float sumsq = 0.0f;
#pragma unroll
        for(int i = 0; i < kElemsPerLane; ++i)
        {
            sumsq += elems[i] * elems[i];
        }
        sumsq = warpReduceSum(sumsq);
        const float rms_rcp = rsqrtf(sumsq / static_cast<float>(kHeadDim) + eps);
#pragma unroll
        for(int i = 0; i < kElemsPerLane; ++i)
        {
            const int dim = lane_id * kElemsPerLane + i;
            elems[i] = elems[i] * rms_rcp * (1.0f + static_cast<float>(weight[dim]));
        }
    }

    if(do_rope)
    {
        const int half = rotary_dim / 2;
        const int dim0 = lane_id * kElemsPerLane;
        const bool in_rope = dim0 < rotary_dim;

        if(in_rope)
        {
            const bool first_half = dim0 < half;
            const int i_base = first_half ? dim0 : dim0 - half;
            const int partner_lane = first_half ? (dim0 + half) / kElemsPerLane
                                                : (dim0 - half) / kElemsPerLane;
            const scalar_t* sin_ptr = cos_ptr + half;
#pragma unroll
            for(int i = 0; i < kElemsPerLane; ++i)
            {
                const float c = static_cast<float>(cos_ptr[i_base + i]);
                const float s = static_cast<float>(sin_ptr[i_base + i]);
                const float partner = __shfl(elems[i], partner_lane, 32);
                elems[i] = first_half ? elems[i] * c - partner * s
                                      : elems[i] * c + partner * s;
            }
        }
    }
}

template <typename scalar_t,
          typename cache_t,
          vllm::Fp8KVCacheDataType kv_dt,
          bool kIsSparse,
          bool kInsertKV>
__global__ void fusedQKNormIdxrQKNormKernel(
    scalar_t* __restrict__ qkv,
    scalar_t* __restrict__ q_out,
    scalar_t* __restrict__ index_q_out,
    const scalar_t* __restrict__ q_norm_w,
    const scalar_t* __restrict__ k_norm_w,
    const scalar_t* __restrict__ iq_norm_w,
    const scalar_t* __restrict__ ik_norm_w,
    const scalar_t* __restrict__ cos_sin_cache,
    const int64_t* __restrict__ positions,
    const int64_t* __restrict__ slot_mapping,
    const int64_t* __restrict__ index_slot_mapping,
    cache_t* __restrict__ kv_cache,
    scalar_t* __restrict__ index_cache,
    const float* __restrict__ k_scale,
    const float* __restrict__ v_scale,
    float eps,
    int rotary_dim,
    int num_tokens,
    int nq,
    int nkv,
    int niq,
    int block_size,
    int64_t kv_s_block,
    int64_t kv_s_kv,
    int64_t kv_s_token,
    int64_t kv_s_head)
{
    const int warps_per_block = blockDim.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int global_warp_idx = blockIdx.x * warps_per_block + (threadIdx.x / 32);

    const int v_slots = kInsertKV ? nkv : 0;
    const int idx_slots = kIsSparse ? niq + 1 : 0;
    const int slots_per_token = nq + nkv + v_slots + idx_slots;

    const int token_idx = global_warp_idx / slots_per_token;
    const int slot = global_warp_idx % slots_per_token;
    if(token_idx >= num_tokens)
    {
        return;
    }

    const int k_begin = nq;
    const int v_begin = nq + nkv;
    const int iq_begin = nq + nkv + v_slots;
    const int ik_slot = iq_begin + niq;

    const bool is_q = slot < k_begin;
    const bool is_k = slot >= k_begin && slot < v_begin;
    bool is_v = false;
    if constexpr(kInsertKV)
    {
        is_v = slot >= v_begin && slot < v_begin + nkv;
    }
    bool is_iq = false;
    bool is_ik = false;
    if constexpr(kIsSparse)
    {
        is_iq = slot >= iq_begin && slot < ik_slot;
        is_ik = slot == ik_slot;
    }

    const int dim_base = lane_id * kElemsPerLane;
    const int qkv_row = (nq + 2 * nkv + (kIsSparse ? (niq + 1) : 0)) * kHeadDim;

    scalar_t* row_ptr = nullptr;
    const scalar_t* norm_w = nullptr;
    bool do_rope = true;
    int head = 0;

    if(is_q)
    {
        row_ptr = qkv + static_cast<int64_t>(token_idx) * qkv_row + slot * kHeadDim;
        norm_w = q_norm_w;
    }
    else if(is_k)
    {
        head = slot - k_begin;
        row_ptr = qkv + static_cast<int64_t>(token_idx) * qkv_row + slot * kHeadDim;
        norm_w = k_norm_w;
    }
    else if(is_v)
    {
        head = slot - v_begin;
        row_ptr = qkv + static_cast<int64_t>(token_idx) * qkv_row + slot * kHeadDim;
        do_rope = false;
    }
    else if(is_iq)
    {
        const int ih = slot - iq_begin;
        row_ptr = qkv + static_cast<int64_t>(token_idx) * qkv_row + (nq + 2 * nkv + ih) * kHeadDim;
        norm_w = iq_norm_w;
    }
    else
    {
        row_ptr = qkv + static_cast<int64_t>(token_idx) * qkv_row + (nq + 2 * nkv + niq) * kHeadDim;
        norm_w = ik_norm_w;
    }

    scalar_t* store_ptr = row_ptr;
    if(is_q && q_out != nullptr)
    {
        store_ptr = q_out + static_cast<int64_t>(token_idx) * nq * kHeadDim + slot * kHeadDim;
    }
    else if(is_iq && index_q_out != nullptr)
    {
        store_ptr = index_q_out + static_cast<int64_t>(token_idx) * niq * kHeadDim +
                    (slot - iq_begin) * kHeadDim;
    }

    int64_t mapped_slot = -1;
    if constexpr(kInsertKV)
    {
        mapped_slot = is_ik ? index_slot_mapping[token_idx]
                            : ((is_k || is_v) ? slot_mapping[token_idx] : -1);
    }

    if constexpr(kInsertKV)
    {
        if(is_v)
        {
            if(mapped_slot >= 0)
            {
                const int64_t b = mapped_slot / block_size;
                const int64_t t = mapped_slot % block_size;
                const int64_t off = b * kv_s_block + kv_s_kv + t * kv_s_token + head * kv_s_head;
                const float v_scale_val =
                    (kv_dt == vllm::Fp8KVCacheDataType::kAuto) ? 1.0f : *v_scale;
                copyOrQuantizeCacheElems<scalar_t, cache_t, kv_dt>(
                    row_ptr + dim_base, kv_cache + off + dim_base, v_scale_val);
            }
            return;
        }
    }

    float elems[kElemsPerLane];
    loadElems(row_ptr + dim_base, elems);
    const int64_t pos = positions[token_idx];
    const scalar_t* cos_ptr = cos_sin_cache + pos * rotary_dim;
    normAndRope(elems, lane_id, eps, norm_w, do_rope, rotary_dim, cos_ptr, norm_w != nullptr);

    if constexpr(kInsertKV)
    {
        if(is_q || is_iq)
        {
            storeElems(store_ptr + dim_base, elems);
        }

        if(mapped_slot >= 0)
        {
            if(is_ik)
            {
                storeElems(index_cache + mapped_slot * kHeadDim + dim_base, elems);
            }
            else if(is_k)
            {
                const int64_t b = mapped_slot / block_size;
                const int64_t t = mapped_slot % block_size;
                const int64_t off = b * kv_s_block + t * kv_s_token + head * kv_s_head;
                const float k_scale_val =
                    (kv_dt == vllm::Fp8KVCacheDataType::kAuto) ? 1.0f : *k_scale;
                storeCacheElems<scalar_t, cache_t, kv_dt>(
                    kv_cache + off + dim_base, elems, k_scale_val);
            }
        }
    }
    else
    {
        storeElems(store_ptr + dim_base, elems);
    }
}

template <typename scalar_t, typename cache_t, vllm::Fp8KVCacheDataType kv_dt>
void launchFusedQKNormIdxrQKNorm(
    scalar_t* qkv,
    scalar_t* q_out,
    scalar_t* index_q_out,
    const scalar_t* q_norm_w,
    const scalar_t* k_norm_w,
    const scalar_t* iq_norm_w,
    const scalar_t* ik_norm_w,
    const scalar_t* cos_sin_cache,
    const int64_t* positions,
    const int64_t* slot_mapping,
    const int64_t* index_slot_mapping,
    cache_t* kv_cache,
    scalar_t* index_cache,
    const float* k_scale,
    const float* v_scale,
    float eps,
    int rotary_dim,
    int num_tokens,
    int nq,
    int nkv,
    int niq,
    int block_size,
    int64_t kv_s_block,
    int64_t kv_s_kv,
    int64_t kv_s_token,
    int64_t kv_s_head,
    bool has_index,
    bool insert_kv,
    hipStream_t stream)
{
    const int v_slots = insert_kv ? nkv : 0;
    const int idx_slots = has_index ? niq + 1 : 0;
    const int slots_per_token = nq + nkv + v_slots + idx_slots;
    constexpr int kBlockSize = 256;
    constexpr int kWarpsPerBlock = kBlockSize / 32;
    const int64_t total_warps = static_cast<int64_t>(num_tokens) * slots_per_token;
    const int grid = static_cast<int>((total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock);
    if(grid == 0)
    {
        return;
    }

#define LAUNCH(IS_SPARSE, INSERT)                                                           \
    fusedQKNormIdxrQKNormKernel<scalar_t, cache_t, kv_dt, IS_SPARSE, INSERT>      \
        <<<grid, kBlockSize, 0, stream>>>(qkv,                                               \
                                          q_out,                                             \
                                          index_q_out,                                       \
                                          q_norm_w,                                          \
                                          k_norm_w,                                          \
                                          iq_norm_w,                                         \
                                          ik_norm_w,                                         \
                                          cos_sin_cache,                                     \
                                          positions,                                         \
                                          slot_mapping,                                      \
                                          index_slot_mapping,                                \
                                          kv_cache,                                          \
                                          index_cache,                                       \
                                          k_scale,                                           \
                                          v_scale,                                           \
                                          eps,                                               \
                                          rotary_dim,                                        \
                                          num_tokens,                                        \
                                          nq,                                                \
                                          nkv,                                               \
                                          niq,                                               \
                                          block_size,                                        \
                                          kv_s_block,                                        \
                                          kv_s_kv,                                           \
                                          kv_s_token,                                        \
                                          kv_s_head)

    if(has_index)
    {
        if(insert_kv)
        {
            LAUNCH(true, true);
        }
        else
        {
            LAUNCH(true, false);
        }
    }
    else
    {
        if(insert_kv)
        {
            LAUNCH(false, true);
        }
        else
        {
            LAUNCH(false, false);
        }
    }
#undef LAUNCH
}

} // namespace fused_qknorm_idxrqknorm_ops

static void fused_qknorm_idxrqknorm_impl(
    aiter_tensor_t& qkv,
    const aiter_tensor_t& q_norm_weight,
    const aiter_tensor_t& k_norm_weight,
    const aiter_tensor_t& cos_sin_cache,
    const aiter_tensor_t& positions,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t rotary_dim,
    double eps,
    std::optional<aiter_tensor_t> index_q_norm_weight,
    std::optional<aiter_tensor_t> index_k_norm_weight,
    int64_t num_index_heads,
    std::optional<aiter_tensor_t> slot_mapping,
    std::optional<aiter_tensor_t> kv_cache,
    std::optional<aiter_tensor_t> index_cache,
    int64_t block_size,
    std::optional<aiter_tensor_t> q_out,
    std::optional<aiter_tensor_t> index_q_out,
    std::optional<aiter_tensor_t> index_slot_mapping,
    const std::string& kv_cache_dtype,
    std::optional<aiter_tensor_t> k_scale,
    std::optional<aiter_tensor_t> v_scale)
{
    using namespace fused_qknorm_idxrqknorm_ops;

    AITER_CHECK(qkv.is_gpu() && qkv.is_contiguous(), "qkv must be contiguous CUDA");
    AITER_CHECK(positions.is_gpu() && positions.is_contiguous() &&
                    positions.dtype() == AITER_DTYPE_i64,
                "positions must be contiguous int64 CUDA");
    AITER_CHECK(cos_sin_cache.is_gpu() && cos_sin_cache.is_contiguous(),
                "cos_sin_cache must be contiguous CUDA");
    AITER_CHECK(cos_sin_cache.dtype() == qkv.dtype(),
                "cos_sin_cache dtype must match qkv");
    AITER_CHECK(cos_sin_cache.dim() == 2 && cos_sin_cache.size(1) == rotary_dim,
                "cos_sin_cache shape must be [max_pos, rotary_dim]");
    AITER_CHECK(q_norm_weight.is_gpu() && k_norm_weight.is_gpu() &&
                    q_norm_weight.is_contiguous() && k_norm_weight.is_contiguous(),
                "q/k norm weights must be contiguous CUDA tensors");
    AITER_CHECK(q_norm_weight.dtype() == qkv.dtype() &&
                    k_norm_weight.dtype() == qkv.dtype(),
                "q/k norm weight dtype must match qkv");
    AITER_CHECK(q_norm_weight.numel() == kHeadDim && k_norm_weight.numel() == kHeadDim,
                "q/k norm weight must have 128 elements");
    AITER_CHECK(rotary_dim > 0 && rotary_dim % 8 == 0 && rotary_dim <= kHeadDim,
                "rotary_dim must be a positive multiple of 8 and <= 128");

    const int num_tokens = static_cast<int>(qkv.size(0));
    const int nq = static_cast<int>(num_heads);
    const int nkv = static_cast<int>(num_kv_heads);
    const int niq = static_cast<int>(num_index_heads);
    AITER_CHECK(nq > 0 && nkv > 0 && niq >= 0, "num_heads/num_kv_heads must be positive");
    const bool has_index = niq > 0;
    const bool insert_kv = kv_cache.has_value();
    const bool fp8_kv_cache =
        insert_kv && kv_cache_dtype.rfind("fp8", 0) == 0;
    const int expected_row = (nq + 2 * nkv + (has_index ? niq + 1 : 0)) * kHeadDim;
    AITER_CHECK(positions.dim() == 1 && positions.size(0) >= num_tokens,
                "positions must be 1D with at least num_tokens elements");
    AITER_CHECK(qkv.dim() == 2 && qkv.size(1) == expected_row,
                "qkv must be [num_tokens, (num_heads + 2*num_kv_heads"
                " + optional index heads) * 128]");
    if(has_index)
    {
        AITER_CHECK(index_q_norm_weight.has_value() && index_k_norm_weight.has_value(),
                    "index branch requires both index norm weights");
        AITER_CHECK(index_q_norm_weight->is_gpu() && index_k_norm_weight->is_gpu() &&
                        index_q_norm_weight->is_contiguous() &&
                        index_k_norm_weight->is_contiguous(),
                    "index norm weights must be contiguous CUDA tensors");
        AITER_CHECK(index_q_norm_weight->dtype() == qkv.dtype() &&
                        index_k_norm_weight->dtype() == qkv.dtype(),
                    "index norm weights dtype must match qkv");
        AITER_CHECK(index_q_norm_weight->numel() == kHeadDim &&
                        index_k_norm_weight->numel() == kHeadDim,
                    "index norm weights must have 128 elements");
    }

    int64_t kv_s_block = 0;
    int64_t kv_s_kv = 0;
    int64_t kv_s_token = 0;
    int64_t kv_s_head = 0;
    if(insert_kv)
    {
        AITER_CHECK(block_size > 0, "block_size must be positive in insert mode");
        AITER_CHECK(slot_mapping.has_value() && slot_mapping->is_gpu() &&
                        slot_mapping->is_contiguous() &&
                        slot_mapping->dtype() == AITER_DTYPE_i64,
                    "insert mode requires contiguous int64 CUDA slot_mapping");
        AITER_CHECK(slot_mapping->dim() == 1 && slot_mapping->size(0) >= num_tokens,
                    "slot_mapping must be 1D with at least num_tokens elements");
        if(has_index && !index_slot_mapping.has_value())
        {
            index_slot_mapping = slot_mapping;
        }
        if(has_index)
        {
            AITER_CHECK(index_slot_mapping.has_value() && index_slot_mapping->is_gpu() &&
                            index_slot_mapping->is_contiguous() &&
                            index_slot_mapping->dtype() == AITER_DTYPE_i64,
                        "sparse insert mode requires contiguous int64 CUDA index_slot_mapping");
            AITER_CHECK(index_slot_mapping->dim() == 1 &&
                            index_slot_mapping->size(0) >= num_tokens,
                        "index_slot_mapping must be 1D with at least num_tokens elements");
            AITER_CHECK(index_cache.has_value() && index_cache->is_gpu() &&
                            index_cache->is_contiguous() &&
                            index_cache->dtype() == qkv.dtype(),
                        "sparse insert mode requires contiguous matching index_cache");
        }
        AITER_CHECK(kv_cache->is_gpu(), "kv_cache must be CUDA");
        if(fp8_kv_cache)
        {
            AITER_CHECK(kv_cache_dtype == "fp8" || kv_cache_dtype == "fp8_e4m3",
                        "fused_qknorm_idxrqknorm fp8 cache insert supports fp8_e4m3 only");
            AITER_CHECK(kv_cache->dtype() == AITER_DTYPE_fp8 ||
                            kv_cache->dtype() == AITER_DTYPE_u8,
                        "fp8 kv_cache must use float8_e4m3 or uint8 storage");
            AITER_CHECK(k_scale.has_value() && v_scale.has_value() &&
                            k_scale->is_gpu() && v_scale->is_gpu() &&
                            k_scale->is_contiguous() && v_scale->is_contiguous() &&
                            k_scale->dtype() == AITER_DTYPE_fp32 &&
                            v_scale->dtype() == AITER_DTYPE_fp32,
                        "fp8 insert requires contiguous float32 CUDA k_scale/v_scale");
            AITER_CHECK(k_scale->numel() >= 1 && v_scale->numel() >= 1,
                        "k_scale/v_scale must contain at least one element");
        }
        else
        {
            AITER_CHECK(kv_cache_dtype == "auto",
                        "non-fp8 fused_qknorm_idxrqknorm expects kv_cache_dtype='auto'");
            AITER_CHECK(kv_cache->dtype() == qkv.dtype(),
                        "kv_cache dtype must match qkv for non-fp8 cache");
        }
        AITER_CHECK(kv_cache->dim() == 5 && kv_cache->size(1) == 2 &&
                        kv_cache->size(2) == block_size && kv_cache->size(3) == nkv &&
                        kv_cache->size(4) == kHeadDim && kv_cache->stride(4) == 1,
                    "kv_cache must be [num_blocks, 2, block_size, num_kv_heads, 128] "
                    "with contiguous head_dim");
        if(has_index)
        {
            AITER_CHECK(index_cache->numel() >= kv_cache->size(0) * block_size * kHeadDim,
                        "index_cache must contain at least num_blocks * block_size * 128 elements");
        }
        kv_s_block = kv_cache->stride(0);
        kv_s_kv = kv_cache->stride(1);
        kv_s_token = kv_cache->stride(2);
        kv_s_head = kv_cache->stride(3);
    }

    if(q_out.has_value())
    {
        AITER_CHECK(q_out->is_gpu() && q_out->is_contiguous() &&
                        q_out->dtype() == qkv.dtype(),
                    "q_out must be contiguous CUDA and match qkv dtype");
        AITER_CHECK(q_out->numel() == static_cast<int64_t>(num_tokens) * nq * kHeadDim,
                    "q_out must have num_tokens * num_heads * 128 elements");
    }
    if(index_q_out.has_value())
    {
        AITER_CHECK(has_index, "index_q_out requires num_index_heads > 0");
        AITER_CHECK(index_q_out->is_gpu() && index_q_out->is_contiguous() &&
                        index_q_out->dtype() == qkv.dtype(),
                    "index_q_out must be contiguous CUDA and match qkv dtype");
        AITER_CHECK(index_q_out->numel() == static_cast<int64_t>(num_tokens) * niq * kHeadDim,
                    "index_q_out must have num_tokens * num_index_heads * 128 elements");
    }

    HipDeviceGuard device_guard(qkv.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    AITER_DISPATCH_FLOATING16_TYPES_rmTorch(qkv.dtype(), "fused_qknorm_idxrqknorm", [&] {
        using T = scalar_t;
        if(fp8_kv_cache)
        {
            launchFusedQKNormIdxrQKNorm<T, opus::fp8_t, vllm::Fp8KVCacheDataType::kFp8E4M3>(
                reinterpret_cast<T*>(qkv.data_ptr()),
                q_out.has_value() ? reinterpret_cast<T*>(q_out->data_ptr()) : nullptr,
                index_q_out.has_value() ? reinterpret_cast<T*>(index_q_out->data_ptr())
                                        : nullptr,
                reinterpret_cast<const T*>(q_norm_weight.data_ptr()),
                reinterpret_cast<const T*>(k_norm_weight.data_ptr()),
                has_index ? reinterpret_cast<const T*>(index_q_norm_weight->data_ptr())
                          : nullptr,
                has_index ? reinterpret_cast<const T*>(index_k_norm_weight->data_ptr())
                          : nullptr,
                reinterpret_cast<const T*>(cos_sin_cache.data_ptr()),
                reinterpret_cast<int64_t*>(positions.data_ptr()),
                insert_kv ? reinterpret_cast<int64_t*>(slot_mapping->data_ptr()) : nullptr,
                (insert_kv && has_index) ? reinterpret_cast<int64_t*>(index_slot_mapping->data_ptr()) : nullptr,
                insert_kv ? reinterpret_cast<opus::fp8_t*>(kv_cache->data_ptr()) : nullptr,
                (insert_kv && has_index) ? reinterpret_cast<T*>(index_cache->data_ptr())
                                         : nullptr,
                insert_kv ? reinterpret_cast<float*>(k_scale->data_ptr()) : nullptr,
                insert_kv ? reinterpret_cast<float*>(v_scale->data_ptr()) : nullptr,
                static_cast<float>(eps),
                static_cast<int>(rotary_dim),
                num_tokens,
                nq,
                nkv,
                niq,
                static_cast<int>(block_size),
                kv_s_block,
                kv_s_kv,
                kv_s_token,
                kv_s_head,
                has_index,
                insert_kv,
                stream);
        }
        else
        {
            launchFusedQKNormIdxrQKNorm<T, T, vllm::Fp8KVCacheDataType::kAuto>(
                reinterpret_cast<T*>(qkv.data_ptr()),
                q_out.has_value() ? reinterpret_cast<T*>(q_out->data_ptr()) : nullptr,
                index_q_out.has_value() ? reinterpret_cast<T*>(index_q_out->data_ptr())
                                        : nullptr,
                reinterpret_cast<const T*>(q_norm_weight.data_ptr()),
                reinterpret_cast<const T*>(k_norm_weight.data_ptr()),
                has_index ? reinterpret_cast<const T*>(index_q_norm_weight->data_ptr())
                          : nullptr,
                has_index ? reinterpret_cast<const T*>(index_k_norm_weight->data_ptr())
                          : nullptr,
                reinterpret_cast<const T*>(cos_sin_cache.data_ptr()),
                reinterpret_cast<int64_t*>(positions.data_ptr()),
                insert_kv ? reinterpret_cast<int64_t*>(slot_mapping->data_ptr()) : nullptr,
                (insert_kv && has_index) ? reinterpret_cast<int64_t*>(index_slot_mapping->data_ptr()) : nullptr,
                insert_kv ? reinterpret_cast<T*>(kv_cache->data_ptr()) : nullptr,
                (insert_kv && has_index) ? reinterpret_cast<T*>(index_cache->data_ptr())
                                         : nullptr,
                nullptr,
                nullptr,
                static_cast<float>(eps),
                static_cast<int>(rotary_dim),
                num_tokens,
                nq,
                nkv,
                niq,
                static_cast<int>(block_size),
                kv_s_block,
                kv_s_kv,
                kv_s_token,
                kv_s_head,
                has_index,
                insert_kv,
                stream);
        }
    });
}

void fused_qknorm_idxrqknorm(
    aiter_tensor_t& qkv,
    const aiter_tensor_t& q_norm_weight,
    const aiter_tensor_t& k_norm_weight,
    const aiter_tensor_t& cos_sin_cache,
    const aiter_tensor_t& positions,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t rotary_dim,
    double eps,
    std::optional<aiter_tensor_t> index_q_norm_weight,
    std::optional<aiter_tensor_t> index_k_norm_weight,
    int64_t num_index_heads,
    std::optional<aiter_tensor_t> slot_mapping,
    std::optional<aiter_tensor_t> kv_cache,
    std::optional<aiter_tensor_t> index_cache,
    int64_t block_size,
    std::optional<aiter_tensor_t> q_out,
    std::optional<aiter_tensor_t> index_q_out,
    std::optional<aiter_tensor_t> index_slot_mapping)
{
    fused_qknorm_idxrqknorm_impl(qkv,
                                                q_norm_weight,
                                                k_norm_weight,
                                                cos_sin_cache,
                                                positions,
                                                num_heads,
                                                num_kv_heads,
                                                rotary_dim,
                                                eps,
                                                index_q_norm_weight,
                                                index_k_norm_weight,
                                                num_index_heads,
                                                slot_mapping,
                                                kv_cache,
                                                index_cache,
                                                block_size,
                                                q_out,
                                                index_q_out,
                                                index_slot_mapping,
                                                "auto",
                                                std::nullopt,
                                                std::nullopt);
}

void fused_qknorm_idxrqknorm_fp8(
    aiter_tensor_t& qkv,
    const aiter_tensor_t& q_norm_weight,
    const aiter_tensor_t& k_norm_weight,
    const aiter_tensor_t& cos_sin_cache,
    const aiter_tensor_t& positions,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t rotary_dim,
    double eps,
    const aiter_tensor_t& index_q_norm_weight,
    const aiter_tensor_t& index_k_norm_weight,
    int64_t num_index_heads,
    const aiter_tensor_t& slot_mapping,
    aiter_tensor_t& kv_cache,
    aiter_tensor_t& index_cache,
    int64_t block_size,
    aiter_tensor_t& q_out,
    aiter_tensor_t& index_q_out,
    const aiter_tensor_t& index_slot_mapping,
    const std::string& kv_cache_dtype,
    const aiter_tensor_t& k_scale,
    const aiter_tensor_t& v_scale)
{
    fused_qknorm_idxrqknorm_impl(qkv,
                                                q_norm_weight,
                                                k_norm_weight,
                                                cos_sin_cache,
                                                positions,
                                                num_heads,
                                                num_kv_heads,
                                                rotary_dim,
                                                eps,
                                                index_q_norm_weight,
                                                index_k_norm_weight,
                                                num_index_heads,
                                                slot_mapping,
                                                kv_cache,
                                                index_cache,
                                                block_size,
                                                q_out,
                                                index_q_out,
                                                index_slot_mapping,
                                                kv_cache_dtype,
                                                k_scale,
                                                v_scale);
}

} // namespace aiter
