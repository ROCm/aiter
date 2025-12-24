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
template <typename T, int vec_size>
struct alignas(sizeof(T) * vec_size) vec_t
{
    T data[vec_size];
    __device__ __forceinline__ T& operator[](int i) { return data[i]; }
    __device__ __forceinline__ T const& operator[](int i) const { return data[i]; }
    __device__ __forceinline__ void load(const T* ptr)
    {
        *this = *reinterpret_cast<vec_t<T, vec_size>*>(const_cast<T*>(ptr));
    }
    __device__ __forceinline__ void loop_load(const T* ptr)
    {
#pragma unroll
        for(int i = 0; i < vec_size; ++i)
        {
            data[i] = ptr[i];
        }
    }
    __device__ __forceinline__ void store(T* ptr)
    {
        *reinterpret_cast<vec_t<T, vec_size>*>(ptr) = *this;
    }
    __device__ __forceinline__ void loop_store(T* ptr)
    {
#pragma unroll
        for(int i = 0; i < vec_size; ++i)
        {
            ptr[i] = data[i];
        }
    }
    __device__ __forceinline__ void nontemporal_load(const T* ptr)
    {
        constexpr int ITERS = vec_size * sizeof(T) / sizeof(uint32_t);
#pragma unroll
        for(int i = 0; i < ITERS; ++i)
        {
            reinterpret_cast<uint32_t*>(&data)[i] = __builtin_nontemporal_load((uint32_t*)ptr + i);
        }
    }
    __device__ __forceinline__ void nontemporal_store(T* ptr)
    {
        constexpr int ITERS = vec_size * sizeof(T) / sizeof(uint32_t);
#pragma unroll
        for(int i = 0; i < ITERS; ++i)
        {
            __builtin_nontemporal_store(reinterpret_cast<uint32_t*>(&data)[i], (uint32_t*)ptr + i);
        }
    }
    __device__ __forceinline__ void fill(T val)
    {
#pragma unroll
        for(int i = 0; i < vec_size; ++i)
        {
            data[i] = val;
        }
    }
};

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
} // namespace aiter