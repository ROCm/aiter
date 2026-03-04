/*
 * Copyright (C) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cmath>
#include <type_traits>

#include "quant_utils.cuh"
#include "rope/rope_common.h"
#include "vec_convert.h"
#include <torch/cuda.h>

#define CHECK_TYPE(x, st) \
    TORCH_CHECK(          \
        x.scalar_type() == st, #x " dtype is ", x.scalar_type(), ", while ", st, " is expected")
#define CHECK_TH_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_TH_CUDA(x);  \
    CHECK_CONTIGUOUS(x)

namespace {

using mrope_utils::vec_t;

template <typename Func, typename T>
__inline__ __device__ T warpReduceSum(Func func, T val)
{
#pragma unroll
    for(int mask = 16; mask > 0; mask >>= 1)
        val = func(val, __shfl_xor(val, mask, 32));
    return val;
}

template <typename T>
inline __device__ __host__ T divUp(T m, T n)
{
    return (m + n - 1) / n;
}

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

// Adopted and changed from vllm
// https://github.com/vllm-project/vllm/blob/main/csrc/fused_qknorm_rope_kernel.cu

// Perform per-head QK Norm,  RoPE in a single kernel.
// scalar_t: data type of QKV and RMSNorm weights
// kv_cache_scalar_t: data type of kv cache
// head_dim: the dimension of each head
// interleave: interleave=!is_neox.
// num_kv_heads: number of kv heads for kv cache
// kv_dt: data type of kv cache for quantization
template <typename scalar_t,
          typename kv_cache_scalar_t,
          int head_dim,
          bool interleave,
          int num_kv_heads,
          vllm::Fp8KVCacheDataType kv_dt>
__global__ void fusedQKNormRopeQuantCacheShuffleKernel(
    scalar_t* qkv_void,            // Combined QKV tensor
    int const num_heads_q,         // Number of query heads
    int const num_heads_k,         // Number of key heads
    int const num_heads_v,         // Number of value heads
    float const eps,               // Epsilon for RMS normalization
    scalar_t const* q_weight,      // RMSNorm weights for query
    scalar_t const* k_weight,      // RMSNorm weights for key
    scalar_t const* cos_sin_cache, // Pre-computed cos/sin cache
    int64_t const* position_ids,   // Position IDs for RoPE
    kv_cache_scalar_t*
        k_cache, // Key cache [num_blocks, num_kv_heads, head_size // x, block_size, x]
    kv_cache_scalar_t*
        v_cache,           // Value cache [num_blocks, num_kv_heads, block_size/X, head_size, X]
    int64_t* slot_mapping, // Slot mapping
    float* k_scale,        // Key scale for quantized key cache [num_blocks, block_size]
    float* v_scale,        // Value scale for quantized value cache [num_blocks, block_size]
    int const num_tokens,  // Number of tokens
    int const page_size,   // Page size for kv cache
    int x                  // kv cache tiling size
)
{

    int const warpsPerBlock = blockDim.x / 32;
    int const warpId        = threadIdx.x / 32;
    int const laneId        = threadIdx.x % 32;

    int const globalWarpIdx = blockIdx.x * warpsPerBlock + warpId;

    int const num_heads    = num_heads_q + num_heads_k + num_heads_v;
    int const tokenIdx     = globalWarpIdx / num_heads;
    int const localHeadIdx = globalWarpIdx % num_heads;
    if(tokenIdx >= num_tokens)
        return;
    bool const isQ                  = localHeadIdx < num_heads_q;
    bool const isK                  = (localHeadIdx < num_heads_q + num_heads_k) & !isQ;
    bool const isV                  = !isQ & !isK;
    int const headIdx               = isV   ? localHeadIdx - num_heads_q - num_heads_k
                                      : isK ? localHeadIdx - num_heads_q
                                            : localHeadIdx;
    constexpr int numElemsPerThread = head_dim / 32;
    scalar_t elements[numElemsPerThread];
    constexpr int best_vec_size = sizeof(float4) / sizeof(scalar_t);
    constexpr int vec_size      = std::min(best_vec_size, numElemsPerThread);
    constexpr int load_loop_cnt = numElemsPerThread / vec_size;
    using ltype                 = ::vec_t<scalar_t, vec_size>;
    const float inverted_kscale = k_scale == nullptr ? 1.0f : 1 / (*k_scale);
    const float inverted_vscale = v_scale == nullptr ? 1.0f : 1 / (*v_scale);

#pragma unroll
    // Load data first, suppose have no tail since we check the head_dim is multiple of 32 before
    // kernel launch
    for(int i = 0; i < load_loop_cnt; i += 1)
    {
        int64_t offsetWarp = (tokenIdx * num_heads * head_dim + localHeadIdx * head_dim +
                              laneId * numElemsPerThread) /
                             vec_size;
        reinterpret_cast<ltype*>(elements)[i] = reinterpret_cast<ltype*>(qkv_void)[offsetWarp + i];
    }

    // If qk, we adopt RMSNorm + RoPE, so we need to compute sum of squares.
    if(!isV)
    {

        // Compute norm squares
        float sumOfSquares = 0.0f;
#pragma unroll
        for(int i = 0; i < numElemsPerThread; i++)
        {
            sumOfSquares += static_cast<float>(elements[i]) * static_cast<float>(elements[i]);
        }
        auto sum_func = [](float a, float b) { return a + b; };
        sumOfSquares  = warpReduceSum(sum_func, sumOfSquares);
        float rms_rcp = rsqrtf(sumOfSquares / static_cast<float>(head_dim) + eps);

        // Normalize elements
#pragma unroll
        for(int i = 0; i < numElemsPerThread; i++)
        {
            int dim      = laneId * numElemsPerThread + i;
            float weight = isQ ? float(q_weight[dim]) : float(k_weight[dim]);
            elements[i]  = static_cast<scalar_t>(elements[i] * rms_rcp * weight);
        }

        // Apply RoPE to normalized elements

        int64_t pos_id = position_ids[tokenIdx];

        // Calculate cache pointer for this position - similar to
        // pos_encoding_kernels.cu
        scalar_t const* cache_ptr = cos_sin_cache + pos_id * head_dim;
        int const embed_dim       = head_dim / 2;
        scalar_t const* cos_ptr   = cache_ptr;
        scalar_t const* sin_ptr   = cache_ptr + embed_dim;

        if constexpr(interleave)
        {
            // Perform interleaving. Use pre-computed cos/sin values.
#pragma unroll
            for(int i = 0; i < numElemsPerThread / 2; ++i)
            {
                int const idx0 = 2 * i;
                int const idx1 = 2 * i + 1;

                float const val0 = elements[idx0];
                float const val1 = elements[idx1];

                int const dim_idx  = laneId * numElemsPerThread + idx0;
                int const half_dim = dim_idx / 2;
                float cos_val      = static_cast<float>(cos_ptr[half_dim]);
                float sin_val      = static_cast<float>(sin_ptr[half_dim]);

                elements[idx0] = static_cast<scalar_t>(val0 * cos_val - val1 * sin_val);
                elements[idx1] = static_cast<scalar_t>(val0 * sin_val + val1 * cos_val);
            }
        }
        else
        {
            scalar_t elements2[numElemsPerThread]; // Additional buffer required for RoPE.
            // Before data exchange with in warp, we need to sync.
            __syncwarp();
            // Get the data from the other half of the warp. Use pre-computed cos/sin
            // values.
#pragma unroll
            for(int i = 0; i < numElemsPerThread; i++)
            {
                elements2[i] = static_cast<scalar_t>(__shfl_xor(float(elements[i]), 16, 32));
                if(laneId < 16)
                {
                    elements2[i] = -elements2[i];
                }

                int dim_idx  = laneId * numElemsPerThread + i;
                dim_idx      = (dim_idx * 2) % head_dim;
                int half_dim = dim_idx / 2;
                // Use pre-computed cos/sin from cache
                float cos_val = cos_ptr[half_dim];
                float sin_val = sin_ptr[half_dim];

                elements[i] = static_cast<scalar_t>(elements[i] * cos_val + elements2[i] * sin_val);
            }
            __syncwarp();
        }
#pragma unroll
        for(int i = 0; i < load_loop_cnt; i += 1)
        {
            int64_t offsetWarp = (tokenIdx * num_heads * head_dim + localHeadIdx * head_dim +
                                  laneId * numElemsPerThread) /
                                 vec_size;
            reinterpret_cast<ltype*>(qkv_void)[offsetWarp + i] =
                reinterpret_cast<ltype*>(elements)[i];
        }
    }

    if(isQ)
    {
        // For Q, we are done.
        return;
    }

    // cache the kv into kv cache and quant if required
    int64_t slot_id = slot_mapping[tokenIdx];
    if(slot_id < 0)
    {
        // invalid slot, skip
        return;
    }
    int64_t block_idx    = slot_id / page_size;
    int64_t block_offset = slot_id % page_size;
    __shared__ float shared_max[num_kv_heads];
    float dtype_max = ck_tile::type_convert<float>(ck_tile::numeric<kv_cache_scalar_t>::max());
    float warp_max  = elements[0];

    // If quantization is required, compute the max abs value across the head_dim * num_heads
    if constexpr(kv_dt != vllm::Fp8KVCacheDataType::kAuto)
    {
        auto f_absmax_f32 = [](float v_0_, float v_1_) {
            return __builtin_fmaxf(abs(v_0_), abs(v_1_));
        };
#pragma unroll
        for(int i = 1; i < numElemsPerThread; i++)
        {
            warp_max = f_absmax_f32(warp_max, elements[i]);
        }
        warp_max = warpReduceSum(f_absmax_f32, warp_max);
    }
    if(isK)
    {
        float k_scale_val = 1.0f;
        if constexpr(kv_dt != vllm::Fp8KVCacheDataType::kAuto)
        {
            k_scale_val = warp_max / dtype_max;
            int64_t scale_offset =
                block_idx * page_size * num_kv_heads + headIdx * page_size + block_offset;
            k_scale[scale_offset] = k_scale_val;
        }
        int64_t cache_offset = block_idx * page_size * num_heads_k * head_dim +
                               headIdx * head_dim * page_size + block_offset * x;

#pragma unroll
        for(int i = 0; i < numElemsPerThread; i++)
        {
            int64_t offset = cache_offset + (laneId * numElemsPerThread + i) / x * page_size * x +
                             (laneId * numElemsPerThread + i) % x;
            if constexpr(kv_dt == vllm::Fp8KVCacheDataType::kAuto)
            {
                k_cache[offset] = elements[i];
            }
            else
            {
                k_cache[offset] =
                    ck_tile::type_convert<kv_cache_scalar_t>(float(elements[i]) / k_scale_val);
            }
        }
    }
    else
    {
        float v_scale_val = 1.0f;
        if constexpr(kv_dt != vllm::Fp8KVCacheDataType::kAuto)
        {
            v_scale_val = warp_max / dtype_max;
            int64_t scale_offset =
                block_idx * page_size * num_kv_heads + headIdx * page_size + block_offset;
            v_scale[scale_offset] = v_scale_val;
        }
        int64_t cache_offset = block_idx * page_size * num_heads_v * head_dim +
                               headIdx * head_dim * page_size + block_offset / x * head_dim * x +
                               block_offset % x;

        // no vectorized store for v cache since its not contiguous on head_dim
#pragma unroll
        for(int i = 0; i < numElemsPerThread; i++)
        {
            int64_t offset = cache_offset + (laneId * numElemsPerThread + i) * x;
            if constexpr(kv_dt == vllm::Fp8KVCacheDataType::kAuto)
            {
                v_cache[offset] = elements[i];
            }
            else
            {
                v_cache[offset] =
                    ck_tile::type_convert<kv_cache_scalar_t>(float(elements[i]) / v_scale_val);
            }
        }
    }
}

#define DISPATCH_KV_HEAD(num_kv_heads, ...)                             \
    if(num_kv_heads == 1)                                               \
    {                                                                   \
        constexpr int NUM_KV_HEADS = 1;                                 \
        __VA_ARGS__                                                     \
    }                                                                   \
    else if(num_kv_heads == 2)                                          \
    {                                                                   \
        constexpr int NUM_KV_HEADS = 2;                                 \
        __VA_ARGS__                                                     \
    }                                                                   \
    else if(num_kv_heads == 4)                                          \
    {                                                                   \
        constexpr int NUM_KV_HEADS = 4;                                 \
        __VA_ARGS__                                                     \
    }                                                                   \
    else if(num_kv_heads == 8)                                          \
    {                                                                   \
        constexpr int NUM_KV_HEADS = 8;                                 \
        __VA_ARGS__                                                     \
    }                                                                   \
    else if(num_kv_heads == 16)                                         \
    {                                                                   \
        constexpr int NUM_KV_HEADS = 16;                                \
        __VA_ARGS__                                                     \
    }                                                                   \
    else if(num_kv_heads == 32)                                         \
    {                                                                   \
        constexpr int NUM_KV_HEADS = 32;                                \
        __VA_ARGS__                                                     \
    }                                                                   \
    else                                                                \
    {                                                                   \
        TORCH_CHECK(false, "Unsupported num_kv_heads: ", num_kv_heads); \
    }

#define DISPATCH_INTERLEAVE(interleave, INTERLEAVE, ...) \
    if(interleave)                                       \
    {                                                    \
        const bool INTERLEAVE = true;                    \
        DISPATCH_KV_HEAD(num_heads_k, __VA_ARGS__)       \
    }                                                    \
    else                                                 \
    {                                                    \
        const bool INTERLEAVE = false;                   \
        DISPATCH_KV_HEAD(num_heads_k, __VA_ARGS__)       \
    }

template <typename scalar_t, typename kv_cache_scalar_t, vllm::Fp8KVCacheDataType kv_dt>
void launchFusedQKNormRopeQuantCacheShuffle(scalar_t* qkv,
                                            int const num_tokens,
                                            int const num_heads_q,
                                            int const num_heads_k,
                                            int const num_heads_v,
                                            int const head_dim,
                                            float const eps,
                                            scalar_t const* q_weight,
                                            scalar_t const* k_weight,
                                            scalar_t const* cos_sin_cache,
                                            bool const interleave,
                                            int64_t const* position_ids,
                                            kv_cache_scalar_t* k_cache,
                                            kv_cache_scalar_t* v_cache,
                                            int64_t* slot_mapping,
                                            float* k_scale,
                                            float* v_scale,
                                            int page_size,
                                            int x,
                                            hipStream_t stream)
{
    // make sure no thread is wasted, adopt 64 here
    constexpr int blockSize      = 64;
    constexpr int warp_per_block = blockSize / 32;
    int const gridSize =
        (num_tokens * (num_heads_q + num_heads_k + num_heads_v) + 1) / warp_per_block;

    dim3 gridDim(gridSize);
    dim3 blockDim(blockSize);

    switch(head_dim)
    {
    case 64:
        DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
            fusedQKNormRopeQuantCacheShuffleKernel<scalar_t,
                                                   kv_cache_scalar_t,
                                                   64,
                                                   INTERLEAVE,
                                                   NUM_KV_HEADS,
                                                   kv_dt>
                <<<gridDim, blockDim, 0, stream>>>(qkv,
                                                   num_heads_q,
                                                   num_heads_k,
                                                   num_heads_v,
                                                   eps,
                                                   q_weight,
                                                   k_weight,
                                                   cos_sin_cache,
                                                   position_ids,
                                                   k_cache,
                                                   v_cache,
                                                   slot_mapping,
                                                   k_scale,
                                                   v_scale,
                                                   num_tokens,
                                                   page_size,
                                                   x);
        });
        break;
    case 128:
        DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
            fusedQKNormRopeQuantCacheShuffleKernel<scalar_t,
                                                   kv_cache_scalar_t,
                                                   128,
                                                   INTERLEAVE,
                                                   NUM_KV_HEADS,
                                                   kv_dt>
                <<<gridDim, blockDim, 0, stream>>>(qkv,
                                                   num_heads_q,
                                                   num_heads_k,
                                                   num_heads_v,
                                                   eps,
                                                   q_weight,
                                                   k_weight,
                                                   cos_sin_cache,
                                                   position_ids,
                                                   k_cache,
                                                   v_cache,
                                                   slot_mapping,
                                                   k_scale,
                                                   v_scale,
                                                   num_tokens,
                                                   page_size,
                                                   x);
        });
        break;
    case 256:
        DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
            fusedQKNormRopeQuantCacheShuffleKernel<scalar_t,
                                                   kv_cache_scalar_t,
                                                   256,
                                                   INTERLEAVE,
                                                   NUM_KV_HEADS,
                                                   kv_dt>
                <<<gridDim, blockDim, 0, stream>>>(qkv,
                                                   num_heads_q,
                                                   num_heads_k,
                                                   num_heads_v,
                                                   eps,
                                                   q_weight,
                                                   k_weight,
                                                   cos_sin_cache,
                                                   position_ids,
                                                   k_cache,
                                                   v_cache,
                                                   slot_mapping,
                                                   k_scale,
                                                   v_scale,
                                                   num_tokens,
                                                   page_size,
                                                   x);
        });
        break;
    default: TORCH_CHECK(false, "Unsupported head dimension for fusedQKNormRope: ", head_dim);
    }
}

} // namespace

#define CALL_QK_NORM_ROPE_CACHE_QUANT(SRC_T, CACHE_T, KV_DTYPE)       \
    launchFusedQKNormRopeQuantCacheShuffle<SRC_T, CACHE_T, KV_DTYPE>( \
        reinterpret_cast<SRC_T*>(qkv.data_ptr()),                     \
        num_tokens,                                                   \
        num_heads_q,                                                  \
        num_heads_k,                                                  \
        num_heads_v,                                                  \
        head_dim,                                                     \
        eps,                                                          \
        reinterpret_cast<SRC_T*>(q_weight.data_ptr()),                \
        reinterpret_cast<SRC_T*>(k_weight.data_ptr()),                \
        reinterpret_cast<SRC_T*>(cos_sin_cache.data_ptr()),           \
        !is_neox,                                                     \
        position_ids.data_ptr<int64_t>(),                             \
        reinterpret_cast<CACHE_T*>(k_cache.data_ptr()),               \
        reinterpret_cast<CACHE_T*>(v_cache.data_ptr()),               \
        slot_mapping.data_ptr<int64_t>(),                             \
        k_scale.has_value() ? k_scale->data_ptr<float>() : nullptr,   \
        v_scale.has_value() ? v_scale->data_ptr<float>() : nullptr,   \
        page_size,                                                    \
        x,                                                            \
        stream);

template <typename T, int HEAD_SIZE, bool IS_NEOX>
__global__ void fused_rope_rms_2way_kernel(const T* q0_,
                                           const T* k0_,
                                           const T* q1_,
                                           const T* k1_,
                                           const T* w_q0,
                                           const T* w_k0,
                                           const T* w_q1,
                                           const T* w_k1,
                                           const T* cos_sin0,
                                           const T* cos_sin1,
                                           int num_tokens0,
                                           int num_tokens1,
                                           int num_heads_q,
                                           int num_heads_k,
                                           float eps,
                                           int total_warps,
                                           T* out_q01_,
                                           T* out_k01_)
{
    using mrope_utils::WARP_SIZE;
    constexpr int VEC_SIZE        = HEAD_SIZE / WARP_SIZE;
    constexpr int HALF_HEAD_SIZE  = HEAD_SIZE / 2;
    const int warp_id             = threadIdx.x / WARP_SIZE;
    const int num_warps_per_block = blockDim.x / WARP_SIZE;
    const int global_warp_id      = blockIdx.x * num_warps_per_block + warp_id;
    if(global_warp_id >= total_warps)
    {
        return;
    }
    // batch_size, num_tokens, num_heads, head_size
    int batch_id = blockIdx.y;
    auto q0      = q0_ + batch_id * num_tokens0 * num_heads_q * HEAD_SIZE;
    auto k0      = k0_ + batch_id * num_tokens0 * num_heads_k * HEAD_SIZE;
    auto q1      = q1_ + batch_id * num_tokens1 * num_heads_q * HEAD_SIZE;
    auto k1      = k1_ + batch_id * num_tokens1 * num_heads_k * HEAD_SIZE;
    auto out_q01 = out_q01_ + batch_id * (num_tokens0 + num_tokens1) * num_heads_q * HEAD_SIZE;
    auto out_k01 = out_k01_ + batch_id * (num_tokens0 + num_tokens1) * num_heads_k * HEAD_SIZE;
    int warp_offset_q0 = 0;
    int warp_offset_k0 = num_tokens0 * num_heads_q;
    int warp_offset_q1 = num_tokens0 * (num_heads_q + num_heads_k);
    int warp_offset_k1 = num_tokens0 * (num_heads_q + num_heads_k) + num_tokens1 * num_heads_q;

    bool is_q0 = global_warp_id < warp_offset_k0;
    bool is_k0 = !is_q0 && global_warp_id < warp_offset_q1;
    bool is_q1 = !is_q0 && !is_k0 && global_warp_id < warp_offset_k1;
    bool is_k1 = !is_q0 && !is_k0 && !is_q1;

    int access_id_in_head = (threadIdx.x % WARP_SIZE) * VEC_SIZE;
    int neighbor_offset =
        access_id_in_head < HALF_HEAD_SIZE ? HALF_HEAD_SIZE / VEC_SIZE : -HALF_HEAD_SIZE / VEC_SIZE;

    int token_id;
    int specialized_warp_id;
    int head_id_in_token;
    int data_offset;

    vec_t<T, VEC_SIZE> w_vec, x_vec, cos_sin_vec, cos_vec, sin_vec;

    if(is_q0)
    {
        specialized_warp_id = global_warp_id - warp_offset_q0;
        token_id            = specialized_warp_id / num_heads_q;
        head_id_in_token    = specialized_warp_id % num_heads_q;
        data_offset         = (token_id * num_heads_q + head_id_in_token) * HEAD_SIZE;
        w_vec.load(w_q0 + access_id_in_head);
        x_vec.load(q0 + data_offset + access_id_in_head);
        if constexpr(IS_NEOX)
        {
            cos_sin_vec.load(&cos_sin0[token_id * HEAD_SIZE + access_id_in_head]);
        }
        else
        {
            cos_vec.load(&cos_sin0[token_id * HEAD_SIZE + access_id_in_head / 2]);
            sin_vec.load(&cos_sin0[token_id * HEAD_SIZE + access_id_in_head / 2 + HALF_HEAD_SIZE]);
        }
    }
    else if(is_k0)
    {
        specialized_warp_id = global_warp_id - warp_offset_k0;
        token_id            = specialized_warp_id / num_heads_k;
        head_id_in_token    = specialized_warp_id % num_heads_k;
        data_offset         = (token_id * num_heads_k + head_id_in_token) * HEAD_SIZE;
        w_vec.load(w_k0 + access_id_in_head);
        x_vec.load(k0 + data_offset + access_id_in_head);
        if constexpr(IS_NEOX)
        {
            cos_sin_vec.load(&cos_sin0[token_id * HEAD_SIZE + access_id_in_head]);
        }
        else
        {
            cos_vec.load(&cos_sin0[token_id * HEAD_SIZE + access_id_in_head / 2]);
            sin_vec.load(&cos_sin0[token_id * HEAD_SIZE + access_id_in_head / 2 + HALF_HEAD_SIZE]);
        }
    }
    else if(is_q1)
    {
        specialized_warp_id = global_warp_id - warp_offset_q1;
        token_id            = specialized_warp_id / num_heads_q;
        head_id_in_token    = specialized_warp_id % num_heads_q;
        data_offset         = (token_id * num_heads_q + head_id_in_token) * HEAD_SIZE;
        w_vec.load(w_q1 + access_id_in_head);
        x_vec.load(q1 + data_offset + access_id_in_head);
        if constexpr(IS_NEOX)
        {
            cos_sin_vec.load(&cos_sin1[token_id * HEAD_SIZE + access_id_in_head]);
        }
        else
        {
            cos_vec.load(&cos_sin1[token_id * HEAD_SIZE + access_id_in_head / 2]);
            sin_vec.load(&cos_sin1[token_id * HEAD_SIZE + access_id_in_head / 2 + HALF_HEAD_SIZE]);
        }
    }
    else
    {
        specialized_warp_id = global_warp_id - warp_offset_k1;
        token_id            = specialized_warp_id / num_heads_k;
        head_id_in_token    = specialized_warp_id % num_heads_k;
        data_offset         = (token_id * num_heads_k + head_id_in_token) * HEAD_SIZE;
        w_vec.load(w_k1 + access_id_in_head);
        x_vec.load(k1 + data_offset + access_id_in_head);
        if constexpr(IS_NEOX)
        {
            cos_sin_vec.load(&cos_sin1[token_id * HEAD_SIZE + access_id_in_head]);
        }
        else
        {
            cos_vec.load(&cos_sin1[token_id * HEAD_SIZE + access_id_in_head / 2]);
            sin_vec.load(&cos_sin1[token_id * HEAD_SIZE + access_id_in_head / 2 + HALF_HEAD_SIZE]);
        }
    }

    mrope_utils::warp_rms_norm_<T, VEC_SIZE>(x_vec, w_vec, HEAD_SIZE, eps);
    vec_t<T, VEC_SIZE> out_vec;

    if constexpr(IS_NEOX)
    {
        auto nb_cos_sin_vec = mrope_utils::warp_shfl_sync_vec<T, VEC_SIZE>(
            cos_sin_vec, threadIdx.x + neighbor_offset);
        auto nb_x_vec =
            mrope_utils::warp_shfl_sync_vec<T, VEC_SIZE>(x_vec, threadIdx.x + neighbor_offset);
        if(neighbor_offset > 0)
        {
#pragma unroll
            for(int i = 0; i < VEC_SIZE; ++i)
            {
                out_vec[i] = (float)x_vec[i] * (float)cos_sin_vec[i] -
                             (float)nb_x_vec[i] * (float)nb_cos_sin_vec[i]; // x0 * cos - x1 * sin
            }
        }
        else
        {
#pragma unroll
            for(int i = 0; i < VEC_SIZE; ++i)
            {
                out_vec[i] = (float)x_vec[i] * (float)nb_cos_sin_vec[i] +
                             (float)nb_x_vec[i] * (float)cos_sin_vec[i]; // x1 * cos + x0 * sin
            }
        }
    }
    else
    {
#pragma unroll
        for(int i = 0; i < VEC_SIZE / 2; ++i)
        {
            out_vec[2 * i + 0] = (float)x_vec[2 * i + 0] * (float)cos_vec[i] -
                                 (float)x_vec[2 * i + 1] * (float)sin_vec[i];
            out_vec[2 * i + 1] = (float)x_vec[2 * i + 1] * (float)cos_vec[i] +
                                 (float)x_vec[2 * i + 0] * (float)sin_vec[i];
        }
    }

    if(is_q0)
    {
        out_vec.store(out_q01 + (token_id * num_heads_q + head_id_in_token) * HEAD_SIZE +
                      access_id_in_head);
    }
    else if(is_k0)
    {
        out_vec.store(out_k01 + (token_id * num_heads_k + head_id_in_token) * HEAD_SIZE +
                      access_id_in_head);
    }
    else if(is_q1)
    {
        out_vec.store(out_q01 +
                      ((num_tokens0 + token_id) * num_heads_q + head_id_in_token) * HEAD_SIZE +
                      access_id_in_head);
    }
    else
    {
        out_vec.store(out_k01 +
                      ((num_tokens0 + token_id) * num_heads_k + head_id_in_token) * HEAD_SIZE +
                      access_id_in_head);
    }
}

template <typename T>
void fused_rope_rms_2way(const T* q0,
                         const T* k0,
                         const T* q1,
                         const T* k1,
                         const T* w_q0,
                         const T* w_k0,
                         const T* w_q1,
                         const T* w_k1,
                         const T* cos_sin0,
                         const T* cos_sin1,
                         int64_t batch_size,
                         int64_t num_tokens0,
                         int64_t num_tokens1,
                         int64_t num_heads_q,
                         int64_t num_heads_k,
                         int64_t head_size,
                         bool is_interleaved,
                         double eps,
                         T* out_q01,
                         T* out_k01,
                         hipStream_t stream)
{
    using mrope_utils::WARP_SIZE;
    TORCH_CHECK(head_size == 64 || head_size == 128 || head_size == 256);
    constexpr int block_size = 256;
    auto total_warps         = (num_tokens0 + num_tokens1) * (num_heads_q + num_heads_k);
    auto num_warps_per_block = block_size / WARP_SIZE;
    dim3 threadsPerBlock(block_size);
    dim3 numBlocks((total_warps + num_warps_per_block - 1) / num_warps_per_block, batch_size);
#define DISPATCH_NEOX(HEAD_SIZE)                                     \
    if(!is_interleaved)                                              \
    {                                                                \
        fused_rope_rms_2way_kernel<T, HEAD_SIZE, true>               \
            <<<numBlocks, threadsPerBlock, 0, stream>>>(q0,          \
                                                        k0,          \
                                                        q1,          \
                                                        k1,          \
                                                        w_q0,        \
                                                        w_k0,        \
                                                        w_q1,        \
                                                        w_k1,        \
                                                        cos_sin0,    \
                                                        cos_sin1,    \
                                                        num_tokens0, \
                                                        num_tokens1, \
                                                        num_heads_q, \
                                                        num_heads_k, \
                                                        eps,         \
                                                        total_warps, \
                                                        out_q01,     \
                                                        out_k01);    \
    }                                                                \
    else                                                             \
    {                                                                \
        fused_rope_rms_2way_kernel<T, HEAD_SIZE, false>              \
            <<<numBlocks, threadsPerBlock, 0, stream>>>(q0,          \
                                                        k0,          \
                                                        q1,          \
                                                        k1,          \
                                                        w_q0,        \
                                                        w_k0,        \
                                                        w_q1,        \
                                                        w_k1,        \
                                                        cos_sin0,    \
                                                        cos_sin1,    \
                                                        num_tokens0, \
                                                        num_tokens1, \
                                                        num_heads_q, \
                                                        num_heads_k, \
                                                        eps,         \
                                                        total_warps, \
                                                        out_q01,     \
                                                        out_k01);    \
    }
    switch(head_size)
    {
    case 64: DISPATCH_NEOX(64) break;
    case 128: DISPATCH_NEOX(128) break;
    case 256: DISPATCH_NEOX(256) break;
    }

#undef DISPATCH_NEOX
}

namespace aiter {

void fused_qk_norm_rope_cache_quant_shuffle(
    at::Tensor& qkv,                   // Combined QKV tensor [num_tokens,
                                       // (num_heads_q+num_heads_k+num_heads_v)*head_dim]
    int64_t num_heads_q,               // Number of query heads
    int64_t num_heads_k,               // Number of key heads
    int64_t num_heads_v,               // Number of value heads
    int64_t head_dim,                  // Dimension per head
    double eps,                        // Epsilon for RMS normalization
    at::Tensor& q_weight,              // RMSNorm weights for query [head_dim]
    at::Tensor& k_weight,              // RMSNorm weights for key [head_dim]
    at::Tensor& cos_sin_cache,         // Cos/sin cache [max_position, head_dim]
    bool is_neox,                      // Whether RoPE is applied in Neox style
    at::Tensor& position_ids,          // Position IDs for RoPE [num_tokens]
    at::Tensor& k_cache,               // k cache
    at::Tensor& v_cache,               // v cache
    at::Tensor& slot_mapping,          // slot mapping
    const std::string& kv_cache_dtype, // kv cache data type
    std::optional<at::Tensor> k_scale, // k scale tensor for quantized k cache
    std::optional<at::Tensor> v_scale  // v scale tensor for quantized v cache
)
{
    // Input validation
    CHECK_INPUT(qkv);
    CHECK_INPUT(position_ids);
    CHECK_INPUT(q_weight);
    CHECK_INPUT(k_weight);
    CHECK_INPUT(cos_sin_cache);
    CHECK_TYPE(position_ids, torch::kInt64);

    TORCH_CHECK(qkv.dim() == 2,
                "QKV tensor must be 2D: [num_tokens, "
                "(num_heads_q+num_heads_k+num_heads_v)*head_dim]");
    TORCH_CHECK(position_ids.dim() == 1, "Position IDs must be 1D: [num_tokens]");
    TORCH_CHECK(q_weight.dim() == 1, "Query weights must be 1D: [head_dim]");
    TORCH_CHECK(k_weight.dim() == 1, "Key weights must be 1D: [head_dim]");
    TORCH_CHECK(cos_sin_cache.dim() == 2, "Cos/sin cache must be 2D: [max_position, head_dim]");
    TORCH_CHECK(q_weight.size(0) == head_dim, "Query weights size must match head dimension");
    TORCH_CHECK(k_weight.size(0) == head_dim, "Key weights size must match head dimension");
    TORCH_CHECK(cos_sin_cache.size(1) == head_dim, "Cos/sin cache dimension must match head_dim");
    TORCH_CHECK(qkv.scalar_type() == q_weight.scalar_type() &&
                    qkv.scalar_type() == k_weight.scalar_type(),
                "qkv, q_weight and k_weight must have the same dtype");
    TORCH_CHECK(head_dim % 32 == 0,
                "Head dimension must be multiple of 32 for fused QK Norm RoPE kernel");
    TORCH_CHECK(
        num_heads_k <= 32,
        "Number of key heads must be less than or equal to 32 for fused QK Norm RoPE kernel");

    int64_t num_tokens = qkv.size(0);
    int64_t page_size  = v_cache.size(-1);
    int64_t x          = k_cache.size(-1);
    TORCH_CHECK(position_ids.size(0) == num_tokens,
                "Number of tokens in position_ids must match QKV");

    int64_t total_heads = num_heads_q + num_heads_k + num_heads_v;
    TORCH_CHECK(qkv.size(1) == total_heads * head_dim,
                "QKV tensor size must match total number of heads and head dimension");

    auto stream = at::hip::getCurrentHIPStream(qkv.get_device());

    DISPATCH_BY_KV_CACHE_DTYPE(qkv.scalar_type(), kv_cache_dtype, CALL_QK_NORM_ROPE_CACHE_QUANT);
}

template <typename T>
struct KernelElementType
{
    using type = T;
};

template <>
struct KernelElementType<c10::Half>
{
    using type = __half;
};

template <>
struct KernelElementType<c10::BFloat16>
{
    using type = hip_bfloat16;
};

void fused_qk_norm_rope_cache_pts_quant_shuffle(at::Tensor& qkv,
                                                at::Tensor& qw,
                                                at::Tensor& kw,
                                                at::Tensor& cos_sin,
                                                at::Tensor& positions,
                                                int64_t num_tokens,
                                                int64_t num_heads_q,
                                                int64_t num_heads_k,
                                                int64_t num_heads_v,
                                                int64_t head_size,
                                                bool is_neox_style,
                                                double eps,
                                                at::Tensor& q_out,
                                                at::Tensor& k_cache,
                                                at::Tensor& v_cache,
                                                at::Tensor& slot_mapping,
                                                at::Tensor& per_tensor_k_scale,
                                                at::Tensor& per_tensor_v_scale,
                                                std::optional<at::Tensor> k_out,
                                                std::optional<at::Tensor> v_out,
                                                bool return_kv,
                                                bool use_shuffle_layout,
                                                int64_t block_size,
                                                int64_t x)
{
    TORCH_CHECK(qkv.is_contiguous() && qw.is_contiguous() && kw.is_contiguous() &&
                cos_sin.is_contiguous());
    TORCH_CHECK(k_cache.is_contiguous() && v_cache.is_contiguous() && slot_mapping.is_contiguous());
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(qkv));
    auto stream         = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
    auto pos_strides    = positions.strides();
    auto kv_cache_dtype = k_cache.scalar_type();
    auto qkv_dtype      = qkv.scalar_type();
    TORCH_CHECK(pos_strides.size() == 1);
    float per_tensor_k_scale_ = per_tensor_k_scale.item<float>();
    float per_tensor_v_scale_ = per_tensor_v_scale.item<float>();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf, qkv_dtype, "fused_qk_norm_rope_cache_pts_quant_shuffle", [&] {
            using T = KernelElementType<scalar_t>::type;
            if(kv_cache_dtype == qkv_dtype)
            {
                T* k_out_ptr = (return_kv && k_out.has_value())
                                   ? (T*)k_out.value().data_ptr<scalar_t>()
                                   : nullptr;
                T* v_out_ptr = (return_kv && v_out.has_value())
                                   ? (T*)v_out.value().data_ptr<scalar_t>()
                                   : nullptr;
                mrope_utils::fused_rope_rms_set_kv<T, T>((T*)qkv.data_ptr<scalar_t>(),
                                                         (T*)qw.data_ptr<scalar_t>(),
                                                         (T*)kw.data_ptr<scalar_t>(),
                                                         (T*)cos_sin.data_ptr<scalar_t>(),
                                                         positions.data_ptr<int64_t>(),
                                                         0,
                                                         pos_strides[0],
                                                         num_tokens,
                                                         num_heads_q,
                                                         num_heads_k,
                                                         num_heads_v,
                                                         head_size,
                                                         is_neox_style,
                                                         eps,
                                                         (T*)q_out.data_ptr<scalar_t>(),
                                                         (T*)k_cache.data_ptr<scalar_t>(),
                                                         (T*)v_cache.data_ptr<scalar_t>(),
                                                         slot_mapping.data_ptr<int64_t>(),
                                                         stream,
                                                         per_tensor_k_scale_,
                                                         per_tensor_v_scale_,
                                                         k_out_ptr,
                                                         v_out_ptr,
                                                         use_shuffle_layout,
                                                         block_size,
                                                         x);
            }
            else
            {
                // Check if kv_cache_dtype is fp8e4m3fnuz or fp8e4m3fn
                if(kv_cache_dtype == at::ScalarType::Float8_e4m3fnuz)
                {
                    mrope_utils::fp8e4m3fnuz* k_out_fp8_ptr =
                        (return_kv && k_out.has_value())
                            ? (mrope_utils::fp8e4m3fnuz*)k_out.value().data_ptr()
                            : nullptr;
                    mrope_utils::fp8e4m3fnuz* v_out_fp8_ptr =
                        (return_kv && v_out.has_value())
                            ? (mrope_utils::fp8e4m3fnuz*)v_out.value().data_ptr()
                            : nullptr;
                    mrope_utils::fused_rope_rms_set_kv<T, mrope_utils::fp8e4m3fnuz>(
                        (T*)qkv.data_ptr<scalar_t>(),
                        (T*)qw.data_ptr<scalar_t>(),
                        (T*)kw.data_ptr<scalar_t>(),
                        (T*)cos_sin.data_ptr<scalar_t>(),
                        positions.data_ptr<int64_t>(),
                        0,
                        pos_strides[0],
                        num_tokens,
                        num_heads_q,
                        num_heads_k,
                        num_heads_v,
                        head_size,
                        is_neox_style,
                        eps,
                        (T*)q_out.data_ptr<scalar_t>(),
                        (mrope_utils::fp8e4m3fnuz*)k_cache.data_ptr(),
                        (mrope_utils::fp8e4m3fnuz*)v_cache.data_ptr(),
                        slot_mapping.data_ptr<int64_t>(),
                        stream,
                        per_tensor_k_scale_,
                        per_tensor_v_scale_,
                        k_out_fp8_ptr,
                        v_out_fp8_ptr,
                        use_shuffle_layout,
                        block_size,
                        x);
                }
                else if(kv_cache_dtype == at::ScalarType::Float8_e4m3fn)
                {
                    mrope_utils::fp8e4m3fn* k_out_fp8_ptr =
                        (return_kv && k_out.has_value())
                            ? (mrope_utils::fp8e4m3fn*)k_out.value().data_ptr()
                            : nullptr;
                    mrope_utils::fp8e4m3fn* v_out_fp8_ptr =
                        (return_kv && v_out.has_value())
                            ? (mrope_utils::fp8e4m3fn*)v_out.value().data_ptr()
                            : nullptr;
                    mrope_utils::fused_rope_rms_set_kv<T, mrope_utils::fp8e4m3fn>(
                        (T*)qkv.data_ptr<scalar_t>(),
                        (T*)qw.data_ptr<scalar_t>(),
                        (T*)kw.data_ptr<scalar_t>(),
                        (T*)cos_sin.data_ptr<scalar_t>(),
                        positions.data_ptr<int64_t>(),
                        0,
                        pos_strides[0],
                        num_tokens,
                        num_heads_q,
                        num_heads_k,
                        num_heads_v,
                        head_size,
                        is_neox_style,
                        eps,
                        (T*)q_out.data_ptr<scalar_t>(),
                        (mrope_utils::fp8e4m3fn*)k_cache.data_ptr(),
                        (mrope_utils::fp8e4m3fn*)v_cache.data_ptr(),
                        slot_mapping.data_ptr<int64_t>(),
                        stream,
                        per_tensor_k_scale_,
                        per_tensor_v_scale_,
                        k_out_fp8_ptr,
                        v_out_fp8_ptr,
                        use_shuffle_layout,
                        block_size,
                        x);
                }
                else
                {
                    TORCH_CHECK(false, "Unsupported KV cache dtype: ", kv_cache_dtype);
                }
            }
        });
}

void fused_qk_norm_rope_2way(at::Tensor& q0,
                             at::Tensor& k0,
                             at::Tensor& q1,
                             at::Tensor& k1,
                             at::Tensor& w_q0,
                             at::Tensor& w_k0,
                             at::Tensor& w_q1,
                             at::Tensor& w_k1,
                             at::Tensor& cos_sin0,
                             at::Tensor& cos_sin1,
                             int64_t batch_size,
                             int64_t num_tokens0,
                             int64_t num_tokens1,
                             int64_t num_heads_q,
                             int64_t num_heads_k,
                             int64_t head_size,
                             bool is_interleaved,
                             double eps,
                             at::Tensor& out_q01,
                             at::Tensor& out_k01)
{
    TORCH_CHECK(q0.is_contiguous() && k0.is_contiguous() && q1.is_contiguous() &&
                k1.is_contiguous());
    TORCH_CHECK(w_q0.is_contiguous() && w_k0.is_contiguous() && w_q1.is_contiguous() &&
                w_k1.is_contiguous());
    TORCH_CHECK(cos_sin0.is_contiguous() && cos_sin1.is_contiguous());
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(q0));
    auto stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf, q0.scalar_type(), "fused_qk_norm_rope_2way", [&] {
            using T = KernelElementType<scalar_t>::type;
            fused_rope_rms_2way<T>((T*)q0.data_ptr<scalar_t>(),
                                   (T*)k0.data_ptr<scalar_t>(),
                                   (T*)q1.data_ptr<scalar_t>(),
                                   (T*)k1.data_ptr<scalar_t>(),
                                   (T*)w_q0.data_ptr<scalar_t>(),
                                   (T*)w_k0.data_ptr<scalar_t>(),
                                   (T*)w_q1.data_ptr<scalar_t>(),
                                   (T*)w_k1.data_ptr<scalar_t>(),
                                   (T*)cos_sin0.data_ptr<scalar_t>(),
                                   (T*)cos_sin1.data_ptr<scalar_t>(),
                                   batch_size,
                                   num_tokens0,
                                   num_tokens1,
                                   num_heads_q,
                                   num_heads_k,
                                   head_size,
                                   is_interleaved,
                                   eps,
                                   (T*)out_q01.data_ptr<scalar_t>(),
                                   (T*)out_k01.data_ptr<scalar_t>(),
                                   stream);
        });
}

} // namespace aiter
