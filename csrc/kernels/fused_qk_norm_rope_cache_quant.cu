/*
 * Copyright (C) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
#include <cstdlib>
#include <type_traits>

#include "aiter_dispatch.h"
#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "fused_qk_norm_rope_cache_quant.h"
#include "hip_reduce.h"
#include "opus/opus.hpp"
#include "aiter_opus_plus.h"
#include "quant_utils.cuh"
#include "mx_quant_utils.h"  // fp_f32_to_e8m0_scale (MX RoundUp E8M0 block scale)
#include "rope/rope_common.h"
#include "vec_convert.h"

#define CHECK_TYPE(x, st)                     \
    AITER_CHECK(x.dtype() == st,              \
                #x " dtype is ",              \
                AiterDtype_to_str(x.dtype()), \
                ", while ",                   \
                AiterDtype_to_str(st),        \
                " is expected")
#define CHECK_TH_CUDA(x) AITER_CHECK(x.is_gpu(), #x " must be a GPU tensor")
#define CHECK_CONTIGUOUS(x) AITER_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_TH_CUDA(x);  \
    CHECK_CONTIGUOUS(x)

// Like is_contiguous() but ignoring dim 0 (permits an interleaved block stride, e.g. vLLM unbind(1)).
static inline bool is_contiguous_from_dim1(const aiter_tensor_t& t)
{
    if(t.numel() == 0)
        return true;
    int64_t expected = 1;
    for(int d = t.dim() - 1; d >= 1; --d)
    {
        if(t.size(d) != 1 && t.stride(d) != expected)
            return false;
        expected *= t.size(d);
    }
    return true;
}

namespace aiter {
/** Map q/k/v tensor strides to logical [token, head, dim] element strides (PyTorch strides are in
 * elements). */
struct ActivationStrides3D
{
    int64_t st;
    int64_t sh;
    int64_t sd;
};

inline ActivationStrides3D
activation_strides_logical_3d(const aiter_tensor_t& t, int64_t num_heads, int64_t head_dim)
{
    if(t.dim() == 2)
    {
        AITER_CHECK(t.size(1) == num_heads * head_dim,
                    "activation dim 1 must be num_heads * head_dim (got ",
                    t.size(1),
                    " vs ",
                    num_heads * head_dim,
                    ")");
        return {t.stride(0), head_dim * t.stride(1), t.stride(1)};
    }
    AITER_CHECK(t.dim() == 3, "q/k/v must be 2D [T, H*D] or 3D [T, H, D], got dim ", t.dim());
    AITER_CHECK(t.size(1) == num_heads && t.size(2) == head_dim,
                "q/k/v 3D shape must be [T, num_heads, head_dim]");
    return {t.stride(0), t.stride(1), t.stride(2)};
}
} // namespace aiter

namespace {
using mrope_utils::vec_t;

// Minimum absmax used when computing FP8 KV scales to avoid division by zero when
// activations are all zero (e.g. CUDA graph warmup, invalid slots, or padding).
static constexpr float kFp8KvQuantAbsmaxFloorF32 = 1e-8f;

// HW-native fp8 e4m3 element dtype, selected by the compile target (same idiom as
// quant_kernels.cu): gfx942 ships e4m3fnuz (max_pos=240), gfx950+ ships OCP e4m3fn
// (max_pos=448). Used as the MX dtype tag for the e8m0 block-scale helpers. Keyed on the
// arch macro rather than an ad-hoc finfo<>::max() threshold, matching opus::finfo<fp8_t>.
static constexpr aiter::MxDtype kHwFp8E4m3Dtype =
#if defined(__gfx942__)
    aiter::MxDtype::FP8_E4M3_FNUZ;
#else
    aiter::MxDtype::FP8_E4M3;
#endif

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
    scalar_t* q_act, // [num_tokens, num_heads_q * head_dim]
    scalar_t* k_act, // [num_tokens, num_heads_k * head_dim]
    scalar_t* v_act, // [num_tokens, num_heads_v * head_dim]
    int64_t const q_st,
    int64_t const q_sh,
    int64_t const q_sd,
    int64_t const k_st,
    int64_t const k_sh,
    int64_t const k_sd,
    int64_t const v_st,
    int64_t const v_sh,
    int64_t const v_sd,
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
    int x,                 // kv cache tiling size
    int const rotary_dim   // Rotary dimension (concatenated cos+sin width); <= head_dim
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

    int64_t const act_st     = isQ ? q_st : (isK ? k_st : v_st);
    int64_t const act_sh     = isQ ? q_sh : (isK ? k_sh : v_sh);
    int64_t const act_sd     = isQ ? q_sd : (isK ? k_sd : v_sd);
    scalar_t* const act_base = isQ ? q_act : (isK ? k_act : v_act);

    // Load data first, suppose have no tail since we check the head_dim is multiple of 32 before
    // kernel launch
    if(act_sd == 1)
    {
        int64_t const base_elems = (int64_t)tokenIdx * act_st + (int64_t)headIdx * act_sh +
                                   (int64_t)(laneId * numElemsPerThread);
#pragma unroll
        for(int i = 0; i < load_loop_cnt; i += 1)
        {
            reinterpret_cast<ltype*>(elements)[i] =
                *reinterpret_cast<ltype const*>(act_base + base_elems + i * vec_size);
        }
    }
    else
    {
#pragma unroll
        for(int j = 0; j < numElemsPerThread; j++)
        {
            int64_t const off = (int64_t)tokenIdx * act_st + (int64_t)headIdx * act_sh +
                                (int64_t)(laneId * numElemsPerThread + j) * act_sd;
            elements[j] = act_base[off];
        }
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

        int64_t pos_id            = position_ids[tokenIdx];
        int const embed_dim       = rotary_dim / 2;
        scalar_t const* cache_ptr = cos_sin_cache + pos_id * rotary_dim;
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
                int const dim0 = laneId * numElemsPerThread + idx0;

                if(dim0 + 1 < rotary_dim)
                {
                    float const val0  = elements[idx0];
                    float const val1  = elements[idx1];
                    int const half_dim = dim0 / 2;
                    float cos_val     = static_cast<float>(cos_ptr[half_dim]);
                    float sin_val     = static_cast<float>(sin_ptr[half_dim]);

                    elements[idx0] = static_cast<scalar_t>(val0 * cos_val - val1 * sin_val);
                    elements[idx1] = static_cast<scalar_t>(val0 * sin_val + val1 * cos_val);
                }
            }
        }
        else
        {
            scalar_t elements2[numElemsPerThread]; // Additional buffer required for RoPE.
            // Before data exchange within warp, we need to sync.
            __syncwarp();
            int const partner_lane_delta = embed_dim / numElemsPerThread;
            // Get the data from the other half of the warp. Use pre-computed cos/sin values.
#pragma unroll
            for(int i = 0; i < numElemsPerThread; i++)
            {
                int const dim_idx = laneId * numElemsPerThread + i;
                if(dim_idx < rotary_dim)
                {
                    elements2[i] = static_cast<scalar_t>(
                        __shfl_xor(float(elements[i]), partner_lane_delta, 32));
                    if(dim_idx < embed_dim)
                    {
                        elements2[i] = -elements2[i];
                    }

                    int const half_dim = dim_idx % embed_dim;
                    float cos_val      = static_cast<float>(cos_ptr[half_dim]);
                    float sin_val      = static_cast<float>(sin_ptr[half_dim]);

                    elements[i] = static_cast<scalar_t>(
                        elements[i] * cos_val + elements2[i] * sin_val);
                }
            }
            __syncwarp();
        }
        int64_t const qk_st    = isQ ? q_st : k_st;
        int64_t const qk_sh    = isQ ? q_sh : k_sh;
        int64_t const qk_sd    = isQ ? q_sd : k_sd;
        scalar_t* const qk_dst = isQ ? q_act : k_act;
        if(qk_sd == 1)
        {
            int64_t const base_elems = (int64_t)tokenIdx * qk_st + (int64_t)headIdx * qk_sh +
                                       (int64_t)(laneId * numElemsPerThread);
#pragma unroll
            for(int i = 0; i < load_loop_cnt; i += 1)
            {
                *reinterpret_cast<ltype*>(qk_dst + base_elems + i * vec_size) =
                    reinterpret_cast<ltype*>(elements)[i];
            }
        }
        else
        {
#pragma unroll
            for(int j = 0; j < numElemsPerThread; j++)
            {
                int64_t const off = (int64_t)tokenIdx * qk_st + (int64_t)headIdx * qk_sh +
                                    (int64_t)(laneId * numElemsPerThread + j) * qk_sd;
                qk_dst[off] = elements[j];
            }
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
            float const warp_max_safe = fmaxf(warp_max, kFp8KvQuantAbsmaxFloorF32);
            k_scale_val               = warp_max_safe / dtype_max;
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
            float const warp_max_safe = fmaxf(warp_max, kFp8KvQuantAbsmaxFloorF32);
            v_scale_val               = warp_max_safe / dtype_max;
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

template <typename scalar_t,
          typename kv_cache_scalar_t,
          int head_dim,
          bool interleave,
          int X,
          int wg_size = 64,
          vllm::Fp8KVCacheDataType kv_dt>
__global__ void fusedQKNormRopeBlockQuantCacheShuffleKernel(
    scalar_t* qkv_void, // Combined QKV tensor [num_tokens, (num_heads_q+num_heads_k+num_heads_v),
                        // head_dim]
    int const num_heads_q,         // Number of query heads
    int const num_heads_k,         // Number of key heads
    int const num_heads_v,         // Number of value heads
    float const eps,               // Epsilon for RMS normalization
    scalar_t const* q_weight,      // RMSNorm weights for query
    scalar_t const* k_weight,      // RMSNorm weights for key
    scalar_t const* cos_sin_cache, // Pre-computed cos/sin cache
    int64_t const* position_ids,   // Position IDs for RoPE
    kv_cache_scalar_t*
        k_cache, // Key cache [num_blocks, num_heads_k, head_size // X, block_size, X]
    kv_cache_scalar_t*
        v_cache,           // Value cache [num_blocks, num_heads_v, block_size // X, head_size, X]
    int64_t* slot_mapping, // Slot mapping
    int64_t const*
        cu_q_len,   // Cu Q len tensor [0, batch0_seq_len, batch0_seq_len + batch1_seq_len, ...]
    float* k_scale, // Key scale for quantized key cache [num_blocks, num_heads_k]
    float* v_scale, // Value scale for quantized value cache [num_blocks, num_heads_v]
    int const num_tokens, // Number of tokens
    int const page_size,  // Page size for kv cache
    int const batch_size, // Batch size
    int const
        blocks_per_batch // Uniform blocks per batch (>0: division mapping, 0: prefix-sum fallback)
)
{
    int const num_heads      = num_heads_q + num_heads_k + num_heads_v;
    int const localHeadIdx   = blockIdx.z;
    int const page_size_log2 = __builtin_ctz(page_size);
    int const page_mask      = page_size - 1;

    int batch_id   = -1;
    int cum_blocks = 0;
    if(gridDim.x > 1)
    {
        // Decode fast path: batch_id = blockIdx.x, no overhead
        batch_id = blockIdx.x;
    }
    else if(blocks_per_batch > 0)
    {
        // Uniform allocation: simple integer division, no shared memory / syncthreads.
        // Used when max_tokens_per_batch is known (prefill, mixed, etc.)
        batch_id = (int)blockIdx.y / blocks_per_batch;
        if(batch_id >= batch_size)
            return;
        cum_blocks = batch_id * blocks_per_batch;
    }
    else
    {
        // Fallback: batch_size <= 1 or max_tokens_per_batch unknown
        batch_id   = 0;
        cum_blocks = 0;
    }
    if(batch_id < 0)
        return;
    int block_within_batch = (int)blockIdx.y - cum_blocks;

    int64_t batch_start_idx = cu_q_len[batch_id];
    int64_t batch_end_idx   = cu_q_len[batch_id + 1];
    int64_t first_token_idx = batch_start_idx + block_within_batch * page_size;
    int64_t slot_idx;
    int64_t block_idx;
    int64_t block_offset;
    // ============================================================================
    // BOUNDARY HANDLING: Similar to cache_kernels.cu lines 504-521
    // Handle case where GPU block extends beyond current batch's sequence length
    // Ensure one wave group only processes one cache block (page)
    // ============================================================================
    if(first_token_idx >= batch_end_idx)
    {
        // This is the extra block for this batch (boundary handler)
        // Check if we need to process remaining tokens from a different cache page
        // Get the previous GPU block's first token
        int64_t prev_first_token_idx = batch_start_idx + (block_within_batch - 1) * page_size;
        if(prev_first_token_idx < batch_start_idx || prev_first_token_idx >= batch_end_idx)
        {
            return;
        }
        int64_t prev_slot_idx   = slot_mapping[prev_first_token_idx];
        int64_t preTg_block_idx = prev_slot_idx >> page_size_log2;
        int64_t last_token_idx  = batch_end_idx - 1;
        slot_idx                = slot_mapping[last_token_idx];
        block_idx               = slot_idx >> page_size_log2;
        if(preTg_block_idx == block_idx)
        {
            return;
        }
        block_offset = slot_idx & page_mask;
    }
    else
    {
        slot_idx     = slot_mapping[first_token_idx];
        block_idx    = slot_idx >> page_size_log2;
        block_offset = slot_idx & page_mask;
    }
    if(slot_idx < 0)
    {
        return;
    }
    if(first_token_idx > batch_start_idx && block_offset > 0)
    {
        __shared__ int64_t idx_smem[2];
        if(threadIdx.x < page_size)
        {
            int64_t token_idx = first_token_idx - (threadIdx.x + 1);
            if(token_idx >= batch_start_idx && token_idx < batch_end_idx)
            {
                int64_t block_idx1 = slot_mapping[token_idx] >> page_size_log2;
                int64_t slot_idx2  = slot_mapping[token_idx + 1];
                int64_t block_idx2 = slot_idx2 >> page_size_log2;
                if(block_idx1 != block_idx2 && block_idx2 == block_idx)
                {
                    idx_smem[0] = token_idx + 1;
                    idx_smem[1] = slot_idx2;
                }
            }
        }
        __syncthreads();
        first_token_idx = idx_smem[0];
        slot_idx        = idx_smem[1];
        // block_idx unchanged: idx_smem search guarantees same page (block_idx2 == block_idx)
        block_offset = slot_idx & page_mask;
    }
    // Each token should compute its own slot_id and block_offset
    int64_t actual_slot_id      = -1;
    int64_t actual_block_offset = 0;
    int64_t actual_block_idx    = -1;
    // Calculate the num_tokens that are in the same cache block (page)
    int tokens_in_block = 0;
    if(first_token_idx + threadIdx.x < batch_end_idx)
    {
        actual_slot_id = slot_mapping[first_token_idx + threadIdx.x];
        if(actual_slot_id >= 0)
        {
            actual_block_idx    = actual_slot_id >> page_size_log2;
            actual_block_offset = actual_slot_id & page_mask;
            tokens_in_block     = (actual_block_idx == block_idx) ? 1 : 0;
        }
    }
    auto sum               = [](float a, float b) { return a + b; };
    int numtokens_in_block = 0;
    numtokens_in_block =
        block_reduce<float, decltype(sum), wg_size, true>(static_cast<float>(tokens_in_block), sum);
    // Calculate tokenIdx for current thread
    int tokenIdx                    = first_token_idx + threadIdx.x;
    bool const isQ                  = localHeadIdx < num_heads_q;
    bool const isK                  = (localHeadIdx < num_heads_q + num_heads_k) & !isQ;
    bool const isV                  = !isQ & !isK;
    int const headIdx               = isV   ? localHeadIdx - num_heads_q - num_heads_k
                                      : isK ? localHeadIdx - num_heads_q
                                            : localHeadIdx;
    constexpr int numElemsPerThread = head_dim;
    constexpr int best_vec_size     = sizeof(float4) / sizeof(scalar_t);
    constexpr int vec_size          = std::min(best_vec_size, numElemsPerThread);
    constexpr int load_loop_cnt     = numElemsPerThread / vec_size;
    using ltype                     = ::vec_t<scalar_t, vec_size>;
    using kv_cache_ltype            = ::vec_t<kv_cache_scalar_t, vec_size>;
    ltype elements;
    ltype next_elements;
    float block_max         = 0.0f;
    auto cur_element_offset = head_dim * threadIdx.x;
    auto f_absmax_f32       = [](float v_0_, float v_1_) {
        return __builtin_fmaxf(abs(v_0_), abs(v_1_));
    };
    // V: only valid tokens; Q/K: ALL threads must participate (avoids __syncthreads deadlock in
    // block_reduce)
    if(isV)
    {
        int64_t total_elements = numtokens_in_block * head_dim;
        for(int idx = threadIdx.x; idx < total_elements / vec_size; idx += blockDim.x)
        {
            int token_idx = first_token_idx + idx * vec_size / head_dim;
            int64_t offsetWarp =
                (token_idx * num_heads * head_dim + localHeadIdx * head_dim) / vec_size;
            int vec_slot = idx % (head_dim / vec_size);
            elements     = reinterpret_cast<ltype*>(qkv_void)[offsetWarp + vec_slot];
#pragma unroll
            for(int j = 0; j < vec_size; j++)
            {
                block_max = f_absmax_f32(block_max, static_cast<float>(elements[j]));
            }
        }
    }
    else
    {
        constexpr int64_t head_thread = head_dim / vec_size;
        int64_t total_elements        = numtokens_in_block * head_dim;
        auto sum_op                   = [](float a, float b) { return a + b; };
        if constexpr(interleave)
        {
            for(int idx = threadIdx.x; idx < total_elements / vec_size; idx += blockDim.x)
            {
                int token_local = idx / head_thread;
                int vec_slot    = idx % head_thread;
                int token_idx   = first_token_idx + token_local;
                if(token_idx >= batch_end_idx)
                    continue;
                int64_t offsetWarp =
                    (token_idx * num_heads * head_dim + localHeadIdx * head_dim) / vec_size;
                elements = reinterpret_cast<ltype*>(qkv_void)[offsetWarp + vec_slot];
                ltype weights;
                scalar_t const* weight_ptr = isQ ? q_weight : k_weight;
                weights                    = reinterpret_cast<const ltype*>(weight_ptr)[vec_slot];
                float partial_sum          = 0.0f;
#pragma unroll
                for(int j = 0; j < vec_size; j++)
                    partial_sum +=
                        static_cast<float>(elements[j]) * static_cast<float>(elements[j]);
                float sumOfSquares =
                    wave_reduce<float, decltype(sum_op), head_thread, true>(partial_sum, sum_op);
                float rms_rcp  = rsqrtf(sumOfSquares / static_cast<float>(head_dim) + eps);
                int64_t pos_id = position_ids[token_idx];
                scalar_t const* cache_ptr = cos_sin_cache + pos_id * head_dim;
                scalar_t const* cos_ptr   = cache_ptr;
                scalar_t const* sin_ptr   = cache_ptr + head_dim / 2;
                int const base_idx        = vec_slot * vec_size;

                using cos_sin_ltype = ::vec_t<scalar_t, vec_size / 2>;
                cos_sin_ltype cos;
                cos = reinterpret_cast<const cos_sin_ltype*>(cos_ptr)[vec_slot];
                cos_sin_ltype sin;
                sin = reinterpret_cast<const cos_sin_ltype*>(sin_ptr)[vec_slot];
#pragma unroll
                for(int k = 0; k < vec_size; k += 2)
                {
                    int const local0   = base_idx + k;
                    int const local1   = base_idx + k + 1;
                    float weight0      = static_cast<float>(weights[k]);
                    float weight1      = static_cast<float>(weights[k + 1]);
                    int const half_dim = local0 / 2;
                    float cos_val      = static_cast<float>(cos[k / 2]);
                    float sin_val      = static_cast<float>(sin[k / 2]);
                    float const val0   = static_cast<float>(elements[k]) * rms_rcp * weight0;
                    float const val1   = static_cast<float>(elements[k + 1]) * rms_rcp * weight1;
                    elements[k]        = static_cast<scalar_t>(val0 * cos_val - val1 * sin_val);
                    elements[k + 1]    = static_cast<scalar_t>(val0 * sin_val + val1 * cos_val);
                    block_max          = f_absmax_f32(block_max, elements[k]);
                    block_max          = f_absmax_f32(block_max, elements[k + 1]);
                }
                reinterpret_cast<ltype*>(qkv_void)[offsetWarp + vec_slot] = elements;
            }
        }
        else
        {
            constexpr int64_t head_thread_half = head_dim / vec_size / 2;
            for(int idx = threadIdx.x; idx < total_elements / vec_size; idx += blockDim.x)
            {
                int token_local = idx / head_thread;
                int vec_slot    = idx % head_thread;
                int token_idx   = first_token_idx + token_local;
                if(token_idx >= batch_end_idx)
                    continue;
                if(vec_slot >= head_thread_half)
                    continue;
                int pair_slot = vec_slot + head_thread_half;
                int64_t offsetWarp =
                    (token_idx * num_heads * head_dim + localHeadIdx * head_dim) / vec_size;
                elements      = reinterpret_cast<ltype*>(qkv_void)[offsetWarp + vec_slot];
                next_elements = reinterpret_cast<ltype*>(qkv_void)[offsetWarp + pair_slot];
                ltype weights0, weights1;
                scalar_t const* weight_ptr = isQ ? q_weight : k_weight;
                weights0                   = reinterpret_cast<const ltype*>(weight_ptr)[vec_slot];
                weights1                   = reinterpret_cast<const ltype*>(weight_ptr)[pair_slot];
                int64_t pos_id             = position_ids[token_idx];
                scalar_t const* cache_ptr  = cos_sin_cache + pos_id * head_dim;
                scalar_t const* cos_ptr    = cache_ptr;
                scalar_t const* sin_ptr    = cache_ptr + head_dim / 2;
                float partial_sum          = 0.0f;
#pragma unroll
                for(int j = 0; j < vec_size; j++)
                    partial_sum +=
                        static_cast<float>(elements[j]) * static_cast<float>(elements[j]) +
                        static_cast<float>(next_elements[j]) * static_cast<float>(next_elements[j]);
                float sumOfSquares = wave_reduce<float, decltype(sum_op), head_thread_half, true>(
                    partial_sum, sum_op);
                float rms_rcp       = rsqrtf(sumOfSquares / static_cast<float>(head_dim) + eps);
                using cos_sin_ltype = ::vec_t<scalar_t, vec_size>;
                cos_sin_ltype cos;
                cos = reinterpret_cast<const cos_sin_ltype*>(cos_ptr)[vec_slot];
                cos_sin_ltype sin;
                sin = reinterpret_cast<const cos_sin_ltype*>(sin_ptr)[vec_slot];
#pragma unroll
                for(int j = 0; j < vec_size; j++)
                {
                    int const idx0   = vec_slot * vec_size + j;
                    int const idx1   = pair_slot * vec_size + j;
                    float weight0    = static_cast<float>(weights0[j]);
                    float weight1    = static_cast<float>(weights1[j]);
                    float cos_val    = static_cast<float>(cos[j]);
                    float sin_val    = static_cast<float>(sin[j]);
                    float const val0 = static_cast<float>(elements[j]) * rms_rcp * weight0;
                    float const val1 = static_cast<float>(next_elements[j]) * rms_rcp * weight1;
                    float out0       = val0 * cos_val - val1 * sin_val;
                    float out1       = val1 * cos_val + val0 * sin_val;
                    block_max        = f_absmax_f32(block_max, out0);
                    block_max        = f_absmax_f32(block_max, out1);
                    elements[j]      = static_cast<scalar_t>(out0);
                    next_elements[j] = static_cast<scalar_t>(out1);
                }
                reinterpret_cast<ltype*>(qkv_void)[offsetWarp + vec_slot]  = elements;
                reinterpret_cast<ltype*>(qkv_void)[offsetWarp + pair_slot] = next_elements;
            }
        }
        // store q
    }
    if(isQ)
    {
        // For Q, we are done.
        return;
    }
    float dtype_max = opus::cast<float>(opus::finfo<opus::fp8_t>::max());
    auto f_max_f32  = [](float v_0_, float v_1_) { return __builtin_fmaxf(v_0_, v_1_); };
    if(kv_dt != vllm::Fp8KVCacheDataType::kAuto)
    {
        block_max = block_reduce<float, decltype(f_max_f32), wg_size, true>(block_max, f_max_f32);
    }
    if(isK)
    {
        float k_scale_val   = 1.0f;
        float inv_scale_val = 1.0f;
        if constexpr(kv_dt != vllm::Fp8KVCacheDataType::kAuto)
        {
            float const block_max_safe = fmaxf(block_max, kFp8KvQuantAbsmaxFloorF32);
            k_scale_val                = block_max_safe / dtype_max;
            inv_scale_val              = dtype_max / block_max_safe;
            int64_t scale_offset       = block_idx * num_heads_k + headIdx;
            if(block_offset > 0)
            {
                float k_scale_global = k_scale[scale_offset];
                if(k_scale_global < k_scale_val)
                {
                    // k_cache layout: [num_blocks, num_heads_k, head_size//X, page_size, X]
                    // TODO(stride-aware): this assumes a contiguous block stride
                    // (block_idx * page_size * num_heads_k * head_dim). Mirror the
                    // runtime per-block-stride fix from the pts shuffle path when this
                    // kernel must support non-contiguous (e.g. vLLM unbind(1)) caches.
                    int64_t cache_base = block_idx * page_size * num_heads_k * head_dim +
                                         headIdx * head_dim * page_size;
                    float rescale            = k_scale_global * inv_scale_val;
                    constexpr int num_hc     = head_dim / X;
                    constexpr int vecs_per_x = X / vec_size;
                    for(int hc = 0; hc < num_hc; hc++)
                    {
                        int64_t hc_base = cache_base + hc * page_size * X;
                        for(int xo = 0; xo < vecs_per_x; xo++)
                        {
                            for(int tok = threadIdx.x; tok < block_offset; tok += blockDim.x)
                            {
                                int64_t addr = hc_base + tok * X + xo * vec_size;
                                kv_cache_ltype data =
                                    *reinterpret_cast<kv_cache_ltype*>(&k_cache[addr]);
#pragma unroll
                                for(int j = 0; j < vec_size; j++)
                                {
                                    data[j] = opus::cast<kv_cache_scalar_t>(
                                        opus::cast<float>(data[j]) * rescale);
                                }
                                *reinterpret_cast<kv_cache_ltype*>(&k_cache[addr]) = data;
                            }
                        }
                    }
                    k_scale[scale_offset] = k_scale_val;
                }
                else
                {
                    k_scale_val   = k_scale_global;
                    inv_scale_val = 1.0f / fmaxf(k_scale_global, kFp8KvQuantAbsmaxFloorF32);
                }
            }
            else
            {
                k_scale[scale_offset] = k_scale_val;
            }
        }
        int64_t cache_offset =
            block_idx * page_size * num_heads_k * head_dim + headIdx * head_dim * page_size;
        int64_t total_elements = numtokens_in_block * head_dim;
        for(int64_t idx = threadIdx.x; idx < total_elements / vec_size; idx += blockDim.x)
        {
            int token_idx          = first_token_idx + idx * vec_size / head_dim;
            int head_offset        = (idx * vec_size) % head_dim;
            int block_offset_local = (token_idx - first_token_idx + block_offset) & page_mask;
            int64_t offsetWarp =
                (token_idx * num_heads * head_dim + localHeadIdx * head_dim) / vec_size;
            elements = reinterpret_cast<ltype*>(qkv_void)[offsetWarp + head_offset / vec_size];
            int64_t vec_offset = cache_offset + (head_offset / X) * page_size * X +
                                 block_offset_local * X + head_offset % X;
            if constexpr(kv_dt == vllm::Fp8KVCacheDataType::kAuto)
            {
                *reinterpret_cast<ltype*>(k_cache + vec_offset) = elements;
            }
            else
            {
                kv_cache_ltype out_vec;
                for(int j = 0; j < vec_size; j++)
                {
                    out_vec[j] = opus::cast<kv_cache_scalar_t>(float(elements[j]) * inv_scale_val);
                }
                *reinterpret_cast<kv_cache_ltype*>(k_cache + vec_offset) = out_vec;
            }
        }
    }
    else
    {
        float v_scale_val   = 1.0f;
        float inv_scale_val = 1.0f;
        if constexpr(kv_dt != vllm::Fp8KVCacheDataType::kAuto)
        {
            float const block_max_safe = fmaxf(block_max, kFp8KvQuantAbsmaxFloorF32);
            v_scale_val                = block_max_safe / dtype_max;
            inv_scale_val              = dtype_max / block_max_safe;
            int64_t scale_offset       = block_idx * num_heads_k + headIdx;
            if(block_offset > 0)
            {
                float v_scale_global = v_scale[scale_offset];
                if(v_scale_global < v_scale_val)
                {
                    // v_cache layout: [num_blocks, num_heads_k, page_size//X, head_size, X]
                    int64_t cache_base = block_idx * page_size * num_heads_v * head_dim +
                                         headIdx * head_dim * page_size;
                    float rescale             = v_scale_global * inv_scale_val;
                    constexpr int vecs_per_bh = (X / vec_size) * head_dim;
                    int n_full_blocks         = block_offset / X;
                    int full_vecs             = n_full_blocks * vecs_per_bh;
                    for(int idx = threadIdx.x; idx < full_vecs; idx += blockDim.x)
                    {
                        kv_cache_ltype data = *reinterpret_cast<kv_cache_ltype*>(
                            v_cache + cache_base + idx * vec_size);
#pragma unroll
                        for(int j = 0; j < vec_size; j++)
                        {
                            data[j] =
                                opus::cast<kv_cache_scalar_t>(opus::cast<float>(data[j]) * rescale);
                        }
                        *reinterpret_cast<kv_cache_ltype*>(v_cache + cache_base + idx * vec_size) =
                            data;
                    }
                    if((block_offset % X) != 0)
                    {
                        int last_block_divX = (block_offset - 1) / X;
                        int last_x_idx      = (block_offset - 1) % X;
                        int last_full_vec   = (last_x_idx + 1) / vec_size;
                        int partial_vecs    = last_full_vec * head_dim;
                        for(int idx = threadIdx.x; idx < partial_vecs; idx += blockDim.x)
                        {
                            int head_offset = idx / last_full_vec;
                            int vec_chunk   = idx % last_full_vec;
                            int64_t vec_off = cache_base + last_block_divX * head_dim * X +
                                              head_offset * X + vec_chunk * vec_size;
                            kv_cache_ltype data =
                                *reinterpret_cast<kv_cache_ltype*>(&v_cache[vec_off]);
#pragma unroll
                            for(int j = 0; j < vec_size; j++)
                            {
                                data[j] = opus::cast<kv_cache_scalar_t>(opus::cast<float>(data[j]) *
                                                                        rescale);
                            }
                            *reinterpret_cast<kv_cache_ltype*>(&v_cache[vec_off]) = data;
                        }
                        int tail_count = (last_x_idx - last_full_vec * vec_size + 1) * head_dim;
                        for(int idx = threadIdx.x; idx < tail_count; idx += blockDim.x)
                        {
                            int head_offset = idx % head_dim;
                            int x_idx       = last_full_vec * vec_size + idx / head_dim;
                            int64_t v_base  = cache_base + last_block_divX * head_dim * X +
                                             head_offset * X + x_idx;
                            v_cache[v_base] = opus::cast<kv_cache_scalar_t>(
                                opus::cast<float>(v_cache[v_base]) * rescale);
                        }
                    }
                    v_scale[scale_offset] = v_scale_val;
                }
                else
                {
                    v_scale_val   = v_scale_global;
                    inv_scale_val = 1.0f / fmaxf(v_scale_global, kFp8KvQuantAbsmaxFloorF32);
                }
            }
            else
            {
                v_scale[scale_offset] = v_scale_val;
            }
        }
        int64_t cache_offset =
            block_idx * page_size * num_heads_v * head_dim + headIdx * head_dim * page_size;
        int64_t total_elements = numtokens_in_block * head_dim;
        for(int64_t idx = threadIdx.x; idx < total_elements / vec_size; idx += blockDim.x)
        {
            int token_idx          = first_token_idx + idx * vec_size / head_dim;
            int head_offset        = (idx * vec_size) % head_dim;
            int block_offset_local = (token_idx - first_token_idx + block_offset) & page_mask;
            int64_t v_base         = cache_offset + (block_offset_local / X) * head_dim * X +
                             head_offset * X + block_offset_local % X;
            int64_t offsetWarp =
                (token_idx * num_heads * head_dim + localHeadIdx * head_dim) / vec_size;
            elements = reinterpret_cast<ltype*>(qkv_void)[offsetWarp + head_offset / vec_size];
#pragma unroll
            for(int j = 0; j < vec_size; j++)
            {
                int64_t offset = v_base + j * X;
                if constexpr(kv_dt == vllm::Fp8KVCacheDataType::kAuto)
                {
                    v_cache[offset] = elements[j];
                }
                else
                {
                    v_cache[offset] =
                        opus::cast<kv_cache_scalar_t>(float(elements[j]) * inv_scale_val);
                }
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
        AITER_CHECK(false, "Unsupported num_kv_heads: ", num_kv_heads); \
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
void launchFusedQKNormRopeQuantCacheShuffle(scalar_t* q_act,
                                            scalar_t* k_act,
                                            scalar_t* v_act,
                                            int64_t const q_st,
                                            int64_t const q_sh,
                                            int64_t const q_sd,
                                            int64_t const k_st,
                                            int64_t const k_sh,
                                            int64_t const k_sd,
                                            int64_t const v_st,
                                            int64_t const v_sh,
                                            int64_t const v_sd,
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
                                            int const rotary_dim,
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
                <<<gridDim, blockDim, 0, stream>>>(q_act,
                                                   k_act,
                                                   v_act,
                                                   q_st,
                                                   q_sh,
                                                   q_sd,
                                                   k_st,
                                                   k_sh,
                                                   k_sd,
                                                   v_st,
                                                   v_sh,
                                                   v_sd,
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
                                                   x,
                                                   rotary_dim);
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
                <<<gridDim, blockDim, 0, stream>>>(q_act,
                                                   k_act,
                                                   v_act,
                                                   q_st,
                                                   q_sh,
                                                   q_sd,
                                                   k_st,
                                                   k_sh,
                                                   k_sd,
                                                   v_st,
                                                   v_sh,
                                                   v_sd,
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
                                                   x,
                                                   rotary_dim);
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
                <<<gridDim, blockDim, 0, stream>>>(q_act,
                                                   k_act,
                                                   v_act,
                                                   q_st,
                                                   q_sh,
                                                   q_sd,
                                                   k_st,
                                                   k_sh,
                                                   k_sd,
                                                   v_st,
                                                   v_sh,
                                                   v_sd,
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
                                                   x,
                                                   rotary_dim);
        });
        break;
    default: AITER_CHECK(false, "Unsupported head dimension for fusedQKNormRope: ", head_dim);
    }
}
template <typename scalar_t, typename kv_cache_scalar_t, vllm::Fp8KVCacheDataType kv_dt>
void launchFusedQKNormRopeBlockQuantCacheShuffle(scalar_t* qkv,
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
                                                 int64_t const* cu_q_len,
                                                 float* k_scale,
                                                 float* v_scale,
                                                 int page_size,
                                                 int x,
                                                 int batch_size,
                                                 int max_tokens_per_batch,
                                                 hipStream_t stream)
{
    int blockSize = page_size < 64 ? 64 : page_size;

    // Three batch-mapping modes, chosen at launch time:
    //
    // Mode A: best when max_tpb < page_size (gridSizeY small, each batch few Y-blocks)
    // Mode B: best when max_tpb known but large (no prefix-sum, simple division)
    // Mode C: only when max_tpb unknown AND avg >= page_size
    int max_tpb           = max_tokens_per_batch > 0
                                ? max_tokens_per_batch
                                : (batch_size > 0 ? (num_tokens + batch_size - 1) / batch_size : num_tokens);
    int gridSizeY_decode  = (max_tpb + page_size - 1) / page_size + 1;
    int gridSizeY_general = (num_tokens + page_size - 1) / page_size + 2 * batch_size;

    int gridSizeY;
    int gridDimX;
    int blocks_per_batch_param = 0; // 0 = not using uniform division

    if(batch_size > 1 && max_tpb < page_size)
    {
        // Mode A: decode fast path — batch_id = blockIdx.x
        gridDimX  = batch_size;
        gridSizeY = gridSizeY_decode;
    }
    else if(batch_size > 1)
    {
        // Mode B: uniform division — batch_id = blockIdx.y / blocks_per_batch
        // When max_tokens_per_batch provided: use actual max (exact).
        // When max_tokens_per_batch=0: use num_tokens as conservative upper bound
        // (safe for any distribution; may over-allocate Y-blocks for small batches).
        gridDimX               = 1;
        blocks_per_batch_param = max_tokens_per_batch > 0
                                     ? gridSizeY_decode
                                     : ((num_tokens + page_size - 1) / page_size + 1);
        gridSizeY              = batch_size * blocks_per_batch_param;
    }
    else
    {
        // batch_size <= 1: single batch, batch_id = 0
        gridDimX  = 1;
        gridSizeY = (num_tokens + page_size - 1) / page_size + 1;
    }

    dim3 gridDim(gridDimX, gridSizeY, num_heads_q + num_heads_k + num_heads_v);
    dim3 blockDim(blockSize);

#define DISPATCH_X_VALUE(x_val, ...)                  \
    if(x_val == 16)                                   \
    {                                                 \
        constexpr int X_VAL = 16;                     \
        __VA_ARGS__                                   \
    }                                                 \
    else if(x_val == 8)                               \
    {                                                 \
        constexpr int X_VAL = 8;                      \
        __VA_ARGS__                                   \
    }                                                 \
    else if(x_val == 4)                               \
    {                                                 \
        constexpr int X_VAL = 4;                      \
        __VA_ARGS__                                   \
    }                                                 \
    else                                              \
    {                                                 \
        AITER_CHECK(false, "Unsupported x: ", x_val); \
    }

#define DISPATCH_INTERLEAVE_BQ(interleave, ...) \
    if(interleave)                              \
    {                                           \
        const bool INTERLEAVE = true;           \
        __VA_ARGS__                             \
    }                                           \
    else                                        \
    {                                           \
        const bool INTERLEAVE = false;          \
        __VA_ARGS__                             \
    }

#define LAUNCH_BLOCK_QUANT_ARGS                                                                  \
    num_heads_q, num_heads_k, num_heads_v, eps, q_weight, k_weight, cos_sin_cache, position_ids, \
        k_cache, v_cache, slot_mapping, cu_q_len, k_scale, v_scale, num_tokens, page_size,       \
        batch_size, blocks_per_batch_param

#define LAUNCH_BLOCK_QUANT_KERNEL(HEAD_DIM, WG_SIZE)                              \
    DISPATCH_INTERLEAVE_BQ(interleave, {                                          \
        DISPATCH_X_VALUE(x, {                                                     \
            fusedQKNormRopeBlockQuantCacheShuffleKernel<scalar_t,                 \
                                                        kv_cache_scalar_t,        \
                                                        HEAD_DIM,                 \
                                                        INTERLEAVE,               \
                                                        X_VAL,                    \
                                                        WG_SIZE,                  \
                                                        kv_dt>                    \
                <<<gridDim, blockDim, 0, stream>>>(qkv, LAUNCH_BLOCK_QUANT_ARGS); \
        });                                                                       \
    });

#define DISPATCH_BLOCK_SIZE(HEAD_DIM)                             \
    if(blockSize == 64)                                           \
    {                                                             \
        LAUNCH_BLOCK_QUANT_KERNEL(HEAD_DIM, 64)                   \
    }                                                             \
    else if(blockSize == 128)                                     \
    {                                                             \
        LAUNCH_BLOCK_QUANT_KERNEL(HEAD_DIM, 128)                  \
    }                                                             \
    else if(blockSize == 256)                                     \
    {                                                             \
        LAUNCH_BLOCK_QUANT_KERNEL(HEAD_DIM, 256)                  \
    }                                                             \
    else                                                          \
    {                                                             \
        AITER_CHECK(false, "Unsupported blockSize: ", blockSize); \
    }

    switch(head_dim)
    {
    case 64: DISPATCH_BLOCK_SIZE(64); break;
    case 128: DISPATCH_BLOCK_SIZE(128); break;
    case 256: DISPATCH_BLOCK_SIZE(256); break;

#undef LAUNCH_BLOCK_QUANT_KERNEL
#undef DISPATCH_BLOCK_SIZE
#undef DISPATCH_X_VALUE
#undef DISPATCH_INTERLEAVE_BQ
    default: AITER_CHECK(false, "Unsupported head dimension for fusedQKNormRope: ", head_dim);
    }
}
} // namespace
#define CALL_QK_NORM_ROPE_CACHE_BLOCK_QUANT(SRC_T, CACHE_T, KV_DTYPE)                  \
    launchFusedQKNormRopeBlockQuantCacheShuffle<SRC_T, CACHE_T, KV_DTYPE>(             \
        reinterpret_cast<SRC_T*>(qkv.data_ptr()),                                      \
        num_tokens,                                                                    \
        num_heads_q,                                                                   \
        num_heads_k,                                                                   \
        num_heads_v,                                                                   \
        head_dim,                                                                      \
        eps,                                                                           \
        reinterpret_cast<SRC_T*>(q_weight.data_ptr()),                                 \
        reinterpret_cast<SRC_T*>(k_weight.data_ptr()),                                 \
        reinterpret_cast<SRC_T*>(cos_sin_cache.data_ptr()),                            \
        !is_neox,                                                                      \
        reinterpret_cast<int64_t*>(position_ids.data_ptr()),                           \
        reinterpret_cast<CACHE_T*>(k_cache.data_ptr()),                                \
        reinterpret_cast<CACHE_T*>(v_cache.data_ptr()),                                \
        reinterpret_cast<int64_t*>(slot_mapping.data_ptr()),                           \
        reinterpret_cast<int64_t*>(cu_q_len.data_ptr()),                               \
        k_scale.has_value() ? reinterpret_cast<float*>(k_scale->data_ptr()) : nullptr, \
        v_scale.has_value() ? reinterpret_cast<float*>(v_scale->data_ptr()) : nullptr, \
        page_size,                                                                     \
        x,                                                                             \
        batch_size,                                                                    \
        max_tokens_per_batch,                                                          \
        stream);

#define CALL_QK_NORM_ROPE_CACHE_QUANT(SRC_T, CACHE_T, KV_DTYPE)                        \
    launchFusedQKNormRopeQuantCacheShuffle<SRC_T, CACHE_T, KV_DTYPE>(                  \
        reinterpret_cast<SRC_T*>(q_t.data_ptr()),                                       \
        reinterpret_cast<SRC_T*>(k_t.data_ptr()),                                       \
        reinterpret_cast<SRC_T*>(v_t.data_ptr()),                                       \
        q_stride_token,                                                                \
        q_stride_head,                                                                 \
        q_stride_dim,                                                                  \
        k_stride_token,                                                                \
        k_stride_head,                                                                 \
        k_stride_dim,                                                                  \
        v_stride_token,                                                                \
        v_stride_head,                                                                 \
        v_stride_dim,                                                                  \
        num_tokens,                                                                    \
        num_heads_q,                                                                   \
        num_heads_k,                                                                   \
        num_heads_v,                                                                   \
        head_dim,                                                                      \
        eps,                                                                           \
        reinterpret_cast<SRC_T*>(q_weight.data_ptr()),                                 \
        reinterpret_cast<SRC_T*>(k_weight.data_ptr()),                                 \
        reinterpret_cast<SRC_T*>(cos_sin_cache.data_ptr()),                            \
        !is_neox,                                                                      \
        reinterpret_cast<int64_t*>(position_ids.data_ptr()),                           \
        reinterpret_cast<CACHE_T*>(k_cache.data_ptr()),                                \
        reinterpret_cast<CACHE_T*>(v_cache.data_ptr()),                                \
        reinterpret_cast<int64_t*>(slot_mapping.data_ptr()),                           \
        k_scale.has_value() ? reinterpret_cast<float*>(k_scale->data_ptr()) : nullptr, \
        v_scale.has_value() ? reinterpret_cast<float*>(v_scale->data_ptr()) : nullptr, \
        page_size,                                                                     \
        x,                                                                             \
        rotary_dim_,                                                                   \
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
    constexpr int PAIR_VEC_SIZE   = VEC_SIZE / 2;
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

    vec_t<T, VEC_SIZE> w_vec, x_vec, cos_sin_vec;
    vec_t<T, PAIR_VEC_SIZE> cos_vec, sin_vec;

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
        for(int i = 0; i < PAIR_VEC_SIZE; ++i)
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
    AITER_CHECK(head_size == 64 || head_size == 128 || head_size == 256);
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

template <typename T, int HEAD_SIZE, bool IS_NEOX>
__global__ void fused_rope_rms_1way_kernel(const T* q_,
                                           const T* k_,
                                           const T* w_q,
                                           const T* w_k,
                                           const float* cos_sin,
                                           int num_tokens,
                                           int num_heads_q,
                                           int num_heads_k,
                                           float eps,
                                           int total_warps,
                                           T* out_q_,
                                           T* out_k_)
{
    using mrope_utils::WARP_SIZE;
    constexpr int VEC_SIZE        = HEAD_SIZE / WARP_SIZE;
    constexpr int HALF_HEAD_SIZE  = HEAD_SIZE / 2;
    // NEOX neighbor in lane space: lane k swaps with lane (k ^ NEIGHBOR_XOR).
    // For all supported HEAD_SIZE in {64, 128, 256}, NEIGHBOR_XOR = 16 (= half of WARP_SIZE).
    constexpr int NEIGHBOR_XOR    = HALF_HEAD_SIZE / VEC_SIZE;
    const int warp_id             = threadIdx.x / WARP_SIZE;
    const int num_warps_per_block = blockDim.x / WARP_SIZE;
    const int global_warp_id      = blockIdx.x * num_warps_per_block + warp_id;
    if(global_warp_id >= total_warps)
    {
        return;
    }
    // batch_size, num_tokens, num_heads, head_size
    int batch_id = blockIdx.y;
    auto q       = q_ + batch_id * num_tokens * num_heads_q * HEAD_SIZE;
    auto k       = k_ + batch_id * num_tokens * num_heads_k * HEAD_SIZE;
    auto out_q   = out_q_ + batch_id * num_tokens * num_heads_q * HEAD_SIZE;
    auto out_k   = out_k_ + batch_id * num_tokens * num_heads_k * HEAD_SIZE;

    int warp_offset_k = num_tokens * num_heads_q;
    bool is_q         = global_warp_id < warp_offset_k;

    int access_id_in_head = (threadIdx.x % WARP_SIZE) * VEC_SIZE;
    bool is_lower_half    = access_id_in_head < HALF_HEAD_SIZE;

    int token_id;
    int specialized_warp_id;
    int head_id_in_token;
    int data_offset;

    vec_t<T, VEC_SIZE> w_vec, x_vec;
    // cos_sin is fp32 per the diffusers reference (qwen-image-edit
    // _apply_rope_complex passes complex freqs in fp32, so the underlying
    // cos/sin pairs carry full fp32 precision). Loading as fp32 keeps the
    // input precision unchanged through the rope multiply.
    vec_t<float, VEC_SIZE> cos_sin_vec, cos_vec, sin_vec;

    if(is_q)
    {
        specialized_warp_id = global_warp_id;
        token_id            = specialized_warp_id / num_heads_q;
        head_id_in_token    = specialized_warp_id % num_heads_q;
        data_offset         = (token_id * num_heads_q + head_id_in_token) * HEAD_SIZE;
        w_vec.load(w_q + access_id_in_head);
        x_vec.load(q + data_offset + access_id_in_head);
    }
    else
    {
        specialized_warp_id = global_warp_id - warp_offset_k;
        token_id            = specialized_warp_id / num_heads_k;
        head_id_in_token    = specialized_warp_id % num_heads_k;
        data_offset         = (token_id * num_heads_k + head_id_in_token) * HEAD_SIZE;
        w_vec.load(w_k + access_id_in_head);
        x_vec.load(k + data_offset + access_id_in_head);
    }

    if constexpr(IS_NEOX)
    {
        cos_sin_vec.load(&cos_sin[token_id * HEAD_SIZE + access_id_in_head]);
    }
    else
    {
        // Interleaved mode only consumes VEC_SIZE/2 cos/sin per lane (one per pair).
        // Use scalar loads of exactly VEC_SIZE/2 elements to avoid the OOB read at
        // the buffer tail when access_id_in_head/2 + HALF_HEAD_SIZE + VEC_SIZE-1
        // would read past the last token's row.
#pragma unroll
        for(int i = 0; i < VEC_SIZE / 2; ++i)
        {
            cos_vec[i] = cos_sin[token_id * HEAD_SIZE + access_id_in_head / 2 + i];
            sin_vec[i] =
                cos_sin[token_id * HEAD_SIZE + access_id_in_head / 2 + HALF_HEAD_SIZE + i];
        }
    }

    // ===========================================================
    // Inline RMSNorm (vs the shared mrope_utils::warp_rms_norm_)
    // ===========================================================
    // Cache the FP32 reads of x_vec[i] in v[] so the writeback loop doesn't
    // re-read bf16 from x_vec (would be redundant v_lshlrev_b32 conversions),
    // then pack the result via pack_f32_to_vec_t (10 instr per bf16x2 pair vs
    // the compiler default ~26 instr — see f32x2_to_bf16x2_rne in
    // rope_common.h). Bit-exact RNE equivalent to warp_rms_norm_ — only
    // difference is NaN payload normalisation (canonical 0x7fff bf16 NaN).
    // To match diffusers RMSNorm semantics, reuse the same scratch for a
    // 2-stage writeback:
    //   n = x * rsqrt(...)
    //   n = round_T(n * gamma_T) after x_vec has been packed back to T
    {
        float v[VEC_SIZE];
        float acc = 0.f;
#pragma unroll
        for(int i = 0; i < VEC_SIZE; ++i)
        {
            v[i] = (float)x_vec[i];
            acc += v[i] * v[i];
        }
        acc         = mrope_utils::block_utils::warp_reduce_sum<float>(acc);
        float s_val = rsqrtf(acc / (float)HEAD_SIZE + eps);

        float n[VEC_SIZE];
#pragma unroll
        for(int i = 0; i < VEC_SIZE; ++i)
        {
            n[i] = v[i] * s_val;
        }
        mrope_utils::pack_f32_to_vec_t(x_vec, n);

#pragma unroll
        for(int i = 0; i < VEC_SIZE; ++i)
        {
            n[i] = (float)x_vec[i] * (float)w_vec[i];
        }
        mrope_utils::pack_f32_to_vec_t(x_vec, n);
    }

    vec_t<T, VEC_SIZE> out_vec;

    if constexpr(IS_NEOX)
    {
        // ds_swizzle XOR-by-NEIGHBOR_XOR — replaces the prior runtime `lane + neighbor_offset`
        // path that lowered to ds_bpermute_b32. Same semantics as `__shfl(v, lane ^ NEIGHBOR_XOR, 32)`.
        auto nb_cos_sin_vec = mrope_utils::warp_shfl_xor_sync_vec<float, VEC_SIZE>(
            cos_sin_vec, opus::number<NEIGHBOR_XOR>{});
        auto nb_x_vec = mrope_utils::warp_shfl_xor_sync_vec<T, VEC_SIZE>(
            x_vec, opus::number<NEIGHBOR_XOR>{});

        // Replace the divergent `if(is_lower_half){}else{}` (which made the
        // compiler emit two copies of the RoPE math AND the FP32→bf16 cvt
        // sequence with s_and_saveexec / s_xor / s_or EXEC mask flips between
        // them) with a per-lane v_cndmask select. Both expressions are
        // evaluated in the SAME FP32 op order as the original divergent code
        // (mul + mul + sub for lower, mul + mul + add for upper) — bit-exact
        // equivalent. Then a single pack_f32_to_vec_t cvt path is reused for
        // every lane.
        float out_f32[VEC_SIZE];
#pragma unroll
        for(int i = 0; i < VEC_SIZE; ++i)
        {
            const float c   = (float)cos_sin_vec[i];
            const float nc  = (float)nb_cos_sin_vec[i];
            const float x0  = (float)x_vec[i];
            const float nx0 = (float)nb_x_vec[i];

            const float lower = x0 * c - nx0 * nc;  // matches old lower branch
            const float upper = x0 * nc + nx0 * c;  // matches old upper branch
            out_f32[i]        = is_lower_half ? lower : upper;
        }
        mrope_utils::pack_f32_to_vec_t(out_vec, out_f32);
    }
    else
    {
        // Stage RoPE results in FP32 then pack via pack_f32_to_vec_t for the
        // same conversion-instruction-count win as the RMSNorm writeback.
        float out_f32[VEC_SIZE];
#pragma unroll
        for(int i = 0; i < VEC_SIZE / 2; ++i)
        {
            out_f32[2 * i + 0] = (float)x_vec[2 * i + 0] * (float)cos_vec[i] -
                                 (float)x_vec[2 * i + 1] * (float)sin_vec[i];
            out_f32[2 * i + 1] = (float)x_vec[2 * i + 1] * (float)cos_vec[i] +
                                 (float)x_vec[2 * i + 0] * (float)sin_vec[i];
        }
        mrope_utils::pack_f32_to_vec_t(out_vec, out_f32);
    }

    if(is_q)
    {
        out_vec.store(out_q + (token_id * num_heads_q + head_id_in_token) * HEAD_SIZE +
                      access_id_in_head);
    }
    else
    {
        out_vec.store(out_k + (token_id * num_heads_k + head_id_in_token) * HEAD_SIZE +
                      access_id_in_head);
    }
}

// quad kernel: a single warp processes 4 heads (= 2 same-token head_pairs)
// for one q-or-k side. Built up from two ideas that compose:
//
//   (1) PAIR PACKING (half-warp layout)
//       The single-head 1way kernel uses VEC_SIZE = HEAD_SIZE / WARP_SIZE
//       elements per lane. For HEAD_SIZE = 128 and bf16 that is 4 elements
//       = 8 bytes/lane → the compiler emits global_load_dwordx2 (8B), which
//       is half the peak per-lane VMEM bandwidth on gfx942.
//
//       Inside this kernel we carve the warp into TWO half-warps and assign
//       each half to one head:
//
//           half_warp_idx = (lane >> 4)   ∈ {0, 1}     ← which head
//           lane_in_half  = lane & 15      ∈ [0, 16)    ← position-in-head
//           VEC_PAIR      = HEAD_SIZE / 16 = 8 bf16     ← bytes/lane × 2
//
//       Each lane now owns 16 bytes of work (a "pair" of heads, with the
//       upper/lower half-warp providing each one). The compiler emits a
//       single global_load_dwordx4 per (token, head_pair).
//
//       Knock-on wins from the pair grouping:
//         * cos_sin depends only on the token — both heads share it, so we
//           load it ONCE per pair instead of twice.
//         * w_q / w_k depend only on the head index modulo HEAD_SIZE —
//           identical for the two heads, again a single shared load.
//         * RMSNorm reduce becomes a 16-lane butterfly (helper
//           half_warp_reduce_sum() in rope_common.h skips the XOR-by-16
//           step so the two halves reduce independently).
//
//   (2) BUNDLED VMEM ISSUE (multiple outstanding loads, same token)
//       NOTE on naming: this is NOT classical software-pipelined double
//       buffering — there is no `prefetch t0; for i: prefetch ti; compute
//       t(i-1)` loop. There is no loop at all. Each warp processes ONE
//       (token, head_quad) tile and exits. What we do is just batch
//       multiple HBM round-trips so they fly in parallel; the win is from
//       load↔load overlap, not load↔compute overlap.
//
//       Step (1) gave us "1 warp = 1 head_pair" with 4 VMEM ops per pair
//       (q/k load + w + cos_sin + store). At single in-flight load per
//       warp the kernel is VMEM-latency-bound on MI300X: load → first-use
//       distance is hundreds of cycles and one outstanding load can't fill
//       that.
//
//       So this kernel doubles the work per warp to TWO head_pairs of the
//       SAME token and bundles ALL their input loads in the prologue:
//
//           prologue (no loop, all issued back-to-back):
//             global_load_dwordx4 x_pair_0   [pair 0 q/k, heads 4p+0,4p+1]
//             global_load_dwordx4 x_pair_1   [pair 1 q/k, heads 4p+2,4p+3,
//                                             +offset 2*HEAD_SIZE]
//             global_load_dwordx4 w_vec      [shared by both pairs]
//             global_load_dwordx4 cos_sin    [shared by both pairs, same token]
//
//       The compiler sinks each consumer behind a decreasing waitcnt
//       (vmcnt(3) → vmcnt(2) → ... → vmcnt(0)) so all 4 HBM round-trips
//       are in flight simultaneously — total wait is max() of the four,
//       not sum(). One load's latency is hidden behind ANOTHER LOAD's
//       latency, not behind compute.
//
//       (Cross-loop producer-consumer pipelining — the "real" double
//       buffer that interleaves prefetch ti with compute t(i-1) — needs a
//       loop. We tried it via TPW=2 (1 warp = 2 tokens) and it didn't
//       help: the kernel is already at ~50-60% HBM peak BW with ~10
//       waves/SIMD, and cross-warp occupancy is already hiding the
//       load latency that a loop-level pipeline would have to fight for.)
//
//       Same-token bundling adds two more wins on top of (1):
//         * w_vec and cos_sin are now loaded ONCE for ALL FOUR heads, not
//           once per pair (so 2× more reuse than pair packing alone).
//         * The NEOX cos_sin shuffle (warp_shfl_xor_sync_vec) only needs
//           to run once per warp; both pair-0 and pair-1 RoPE rotations
//           reuse the same shuffled cos_sin_vec / nb_cos_sin_vec.
//
// Per-warp VMEM cost (4 heads of work):
//     2× dwordx4 q/k load + 1× dwordx4 w + 1× dwordx4 cos_sin
//   + 2× dwordx4 store
//   = 6 VMEM ops per 4 heads → 1.5 ops/head
//   (vs single-head fallback kernel: 4 ops/head)
//
// Numerical envelope:
//   Each (token, head_pair) is computed with the identical math as the
//   single-head kernel — only the cross-lane reduce tree changes (16-lane
//   butterfly instead of 32-lane). With non-associative FP32 the rounded
//   result drifts by at most 1 mantissa ULP, mapping to 0..1 bf16 ULP on
//   ≤ 0.0003% of output elements (verified by sweep against the single-head
//   path). The end-to-end magnitude bound stays inside atol=0.05 vs PyTorch
//   reference, identical envelope to both the single-head 1way kernel and
//   the existing 2way kernel — i.e. no model-accuracy impact.
//
// Constraint: num_heads_q % 4 == 0 && num_heads_k % 4 == 0. The dispatcher
// falls back to the single-head fused_rope_rms_1way_kernel for any other
// shape; that path is untouched and produces bitwise-identical output to
// the pre-quad-kernel baseline.
//
// QUAD_Q_CT / QUAD_K_CT (compile-time):
//   When > 0 the kernel uses them as constexpr divisors so the compiler
//   folds `spec / quad_q` / `spec % quad_q` into a magic-number multiply
//   (5 VALU ops: mul_hi + lshr + mul_lo + sub) instead of the runtime
//   signed integer-divide expansion (~30 ops including v_rcp_iflag_f32).
//   Pass 0 to keep the runtime path. Selected by the host dispatcher
//   based on the actual num_heads_q / num_heads_k.
//
//   Empirical impact at T=8192, HEAD_SIZE=128, bf16 on MI308X: 3-5% faster
//   per-warp than the runtime path (kernel is dominated by VMEM latency,
//   not int-div). VGPR usage and occupancy are identical.
template <typename T,
          int HEAD_SIZE,
          bool IS_NEOX,
          int QUAD_Q_CT = 0,
          int QUAD_K_CT = 0>
__global__ void fused_rope_rms_1way_quad_kernel(const T* q_,
                                                   const T* k_,
                                                   const T* w_q,
                                                   const T* w_k,
                                                   const float* cos_sin,
                                                   int num_tokens,
                                                   int num_heads_q,
                                                   int num_heads_k,
                                                   float eps,
                                                   int total_warps_quad,
                                                   T* out_q_,
                                                   T* out_k_)
{
    using mrope_utils::WARP_SIZE;
    constexpr int LANES_PER_HEAD    = WARP_SIZE / 2; // 16
    constexpr int VEC_PAIR          = HEAD_SIZE / LANES_PER_HEAD;
    constexpr int HALF_HEAD_SIZE    = HEAD_SIZE / 2;
    constexpr int NEIGHBOR_XOR_PAIR = HALF_HEAD_SIZE / VEC_PAIR;
    static_assert(NEIGHBOR_XOR_PAIR == 8,
                  "quad kernel requires NEIGHBOR_XOR_PAIR == 8 (XOR within half-warp)");

    const int warp_id             = threadIdx.x / WARP_SIZE;
    const int num_warps_per_block = blockDim.x / WARP_SIZE;
    const int global_warp_id      = blockIdx.x * num_warps_per_block + warp_id;

    // ---------- branch hoist (uniform Q/K split, scalar cmp) ----------
    // Block layout is 256 threads = 4 physical waves × 2 logical warps each
    // (WARP_SIZE here is 32, half of the 64-lane physical wave). Within a
    // physical wave the two halves have consecutive global_warp_id values
    // X and X+1, so `is_q = global_warp_id < warp_q_end` is uniform across
    // the full 64-lane wave iff `warp_q_end = T*QUAD_Q_CT` is even — which
    // is guaranteed when QUAD_Q_CT is even. Same logic for total_warps_quad
    // = T*(QUAD_Q_CT + QUAD_K_CT). For our deployed instances Q,K ∈ {4,6,8}
    // both are even, so we can `readfirstlane` the warp_id and let the
    // compiler emit `s_cmp + s_cbranch` instead of the per-lane
    // `v_cmp + s_and_saveexec + s_xor + s_cbranch_execz` sequence (saves a
    // few cycles + EXEC-mask thrash on every wave). For odd QUAD_*_CT (e.g.
    // H=12 → QUAD=3) the boundary may cut a wave, so we keep the original
    // divergent path. `spec` below stays per-lane (it differs between the
    // two logical warps of a physical wave by design).
    constexpr bool kBranchUniform =
        (QUAD_Q_CT > 0) && (QUAD_Q_CT % 2 == 0) &&
        (QUAD_K_CT > 0) && (QUAD_K_CT % 2 == 0);
    const int branch_warp_id = kBranchUniform
                                   ? __builtin_amdgcn_readfirstlane(global_warp_id)
                                   : global_warp_id;
    if(branch_warp_id >= total_warps_quad)
    {
        return;
    }

    const int batch_id = blockIdx.y;
    auto q             = q_ + batch_id * num_tokens * num_heads_q * HEAD_SIZE;
    auto k             = k_ + batch_id * num_tokens * num_heads_k * HEAD_SIZE;
    auto out_q         = out_q_ + batch_id * num_tokens * num_heads_q * HEAD_SIZE;
    auto out_k         = out_k_ + batch_id * num_tokens * num_heads_k * HEAD_SIZE;

    // "quad" count per token = how many groups-of-4-heads each token contributes.
    // When QUAD_Q_CT / QUAD_K_CT are non-zero compile-time constants, the
    // div/mod below becomes a constant-divisor magic-multiply (~3 VALU ops);
    // otherwise the compiler emits the full runtime int-div sequence
    // (~30 ops, sat on the critical path before any VMEM can issue).
    const int quad_q     = (QUAD_Q_CT > 0) ? QUAD_Q_CT : (num_heads_q / 4);
    const int quad_k     = (QUAD_K_CT > 0) ? QUAD_K_CT : (num_heads_k / 4);
    const int warp_q_end = num_tokens * quad_q;
    const bool is_q      = branch_warp_id < warp_q_end;

    const int lane_full        = threadIdx.x % WARP_SIZE; // 0..31
    const int lane_in_half     = lane_full & (LANES_PER_HEAD - 1); // 0..15
    const int access_id_in_head = lane_in_half * VEC_PAIR;
    const bool is_lower_half   = access_id_in_head < HALF_HEAD_SIZE;

    int token_id;
    int quad_idx_in_token;
    if(is_q)
    {
        const int spec = global_warp_id;
        if constexpr(QUAD_Q_CT > 0)
        {
            token_id          = spec / QUAD_Q_CT;
            quad_idx_in_token = spec % QUAD_Q_CT;
        }
        else
        {
            token_id          = spec / quad_q;
            quad_idx_in_token = spec % quad_q;
        }
    }
    else
    {
        const int spec = global_warp_id - warp_q_end;
        if constexpr(QUAD_K_CT > 0)
        {
            token_id          = spec / QUAD_K_CT;
            quad_idx_in_token = spec % QUAD_K_CT;
        }
        else
        {
            token_id          = spec / quad_k;
            quad_idx_in_token = spec % quad_k;
        }
    }

    // ===========================================================
    // PROLOGUE: issue all 4 input loads concurrently
    //   - x_pair_0: q/k for pair 0 (heads 4q+0, 4q+1)
    //   - x_pair_1: q/k for pair 1 (heads 4q+2, 4q+3) — offset = +2 * HEAD_SIZE
    //   - w_vec   : RMSNorm gamma (head-independent, shared across pairs)
    //   - cos_sin : token-only (shared across pairs since same token)
    // ===========================================================
    const int head0_in_token = 4 * quad_idx_in_token;

    vec_t<T, VEC_PAIR> x_pair_0, x_pair_1;
    if(is_q)
    {
        const int64_t base_off =
            (static_cast<int64_t>(token_id) * num_heads_q + head0_in_token) * HEAD_SIZE;
        x_pair_0.load(q + base_off + lane_full * VEC_PAIR);
        x_pair_1.load(q + base_off + 2 * HEAD_SIZE + lane_full * VEC_PAIR);
    }
    else
    {
        const int64_t base_off =
            (static_cast<int64_t>(token_id) * num_heads_k + head0_in_token) * HEAD_SIZE;
        x_pair_0.load(k + base_off + lane_full * VEC_PAIR);
        x_pair_1.load(k + base_off + 2 * HEAD_SIZE + lane_full * VEC_PAIR);
    }

    vec_t<T, VEC_PAIR> w_vec;
    if(is_q)
    {
        w_vec.load(w_q + access_id_in_head);
    }
    else
    {
        w_vec.load(w_k + access_id_in_head);
    }

    // cos_sin is fp32 per the diffusers reference — see comment in
    // fused_rope_rms_1way_kernel for the rationale.
    vec_t<float, VEC_PAIR> cos_sin_vec;
    vec_t<float, VEC_PAIR / 2> cos_vec_pair, sin_vec_pair;
    if constexpr(IS_NEOX)
    {
        cos_sin_vec.load(cos_sin + token_id * HEAD_SIZE + access_id_in_head);
    }
    else
    {
#pragma unroll
        for(int i = 0; i < VEC_PAIR / 2; ++i)
        {
            cos_vec_pair[i] =
                cos_sin[token_id * HEAD_SIZE + access_id_in_head / 2 + i];
            sin_vec_pair[i] =
                cos_sin[token_id * HEAD_SIZE + access_id_in_head / 2 + HALF_HEAD_SIZE + i];
        }
    }

    // ===========================================================
    // RMSNorm × 2 (one reduce per pair, both use shared w_vec)
    // ===========================================================
    {
        // Cache FP32 reads so the writeback loop doesn't re-read BF16 from
        // x_pair_0/1 (would be redundant lshl b32 conversions). Same RNE on
        // writeback via pack_f32_to_vec_t (10 instr per bf16x2 pair vs the
        // compiler default 26 instr — see f32x2_to_bf16x2_rne in rope_common.h).
        // To match diffusers RMSNorm semantics, reuse the same scratch for a
        // 2-stage writeback: first pack x * rsqrt(...), then multiply the
        // packed low-precision values by gamma and pack once more.
        float v0[VEC_PAIR], v1[VEC_PAIR];
        float acc0 = 0.f, acc1 = 0.f;
#pragma unroll
        for(int i = 0; i < VEC_PAIR; ++i)
        {
            v0[i] = (float)x_pair_0[i];
            v1[i] = (float)x_pair_1[i];
            acc0 += v0[i] * v0[i];
            acc1 += v1[i] * v1[i];
        }
        acc0         = mrope_utils::block_utils::half_warp_reduce_sum<float>(acc0);
        acc1         = mrope_utils::block_utils::half_warp_reduce_sum<float>(acc1);
        float s_val0 = rsqrtf(acc0 / (float)HEAD_SIZE + eps);
        float s_val1 = rsqrtf(acc1 / (float)HEAD_SIZE + eps);

        float n0[VEC_PAIR], n1[VEC_PAIR];
#pragma unroll
        for(int i = 0; i < VEC_PAIR; ++i)
        {
            n0[i] = v0[i] * s_val0;
            n1[i] = v1[i] * s_val1;
        }
        mrope_utils::pack_f32_to_vec_t(x_pair_0, n0);
        mrope_utils::pack_f32_to_vec_t(x_pair_1, n1);

#pragma unroll
        for(int i = 0; i < VEC_PAIR; ++i)
        {
            n0[i] = (float)x_pair_0[i] * (float)w_vec[i];
            n1[i] = (float)x_pair_1[i] * (float)w_vec[i];
        }
        mrope_utils::pack_f32_to_vec_t(x_pair_0, n0);
        mrope_utils::pack_f32_to_vec_t(x_pair_1, n1);
    }

    // ===========================================================
    // RoPE × 2 (cos_sin shuffle SHARED between pair 0 and pair 1)
    // ===========================================================
    vec_t<T, VEC_PAIR> out_pair_0, out_pair_1;
    if constexpr(IS_NEOX)
    {
        // Single shuffle of cos_sin reused by both pairs.
        auto nb_cos_sin_vec = mrope_utils::warp_shfl_xor_sync_vec<float, VEC_PAIR>(
            cos_sin_vec, opus::number<NEIGHBOR_XOR_PAIR>{});
        // Per-pair x shuffles.
        auto nb_x_pair_0 = mrope_utils::warp_shfl_xor_sync_vec<T, VEC_PAIR>(
            x_pair_0, opus::number<NEIGHBOR_XOR_PAIR>{});
        auto nb_x_pair_1 = mrope_utils::warp_shfl_xor_sync_vec<T, VEC_PAIR>(
            x_pair_1, opus::number<NEIGHBOR_XOR_PAIR>{});

        // Replace divergent `if(is_lower_half){}else{}` (which forced the
        // compiler to emit two copies of the rope math AND of the FP32→BF16
        // cvt sequence, with s_and_saveexec / s_xor / s_or EXEC mask
        // switches between them) with a per-lane cndmask select. Both
        // expressions are evaluated in the SAME FP32 op order as the
        // original divergent code (mul + mul + sub for lower, mul + mul +
        // add for upper), then cndmask picks the right one — bit-exact
        // equivalent for every lane, single cvt path per output.
        // FP32 results are staged then packed via pack_f32_to_vec_t which
        // for bfloat16 lowers to v_cmp_u_f32 + v_bfe_u32 + v_add3_u32 +
        // v_cndmask + v_and_or_b32 (10 instr per bf16x2 pair vs 26 for the
        // default scalar __hip_bfloat16(float) ctor expansion).
        float out0_f32[VEC_PAIR], out1_f32[VEC_PAIR];
#pragma unroll
        for(int i = 0; i < VEC_PAIR; ++i)
        {
            const float c   = (float)cos_sin_vec[i];
            const float nc  = (float)nb_cos_sin_vec[i];
            const float x0  = (float)x_pair_0[i];
            const float x1  = (float)x_pair_1[i];
            const float nx0 = (float)nb_x_pair_0[i];
            const float nx1 = (float)nb_x_pair_1[i];

            const float lower0 = x0 * c - nx0 * nc;   // matches old lower branch
            const float upper0 = x0 * nc + nx0 * c;   // matches old upper branch
            const float lower1 = x1 * c - nx1 * nc;
            const float upper1 = x1 * nc + nx1 * c;

            out0_f32[i] = is_lower_half ? lower0 : upper0;
            out1_f32[i] = is_lower_half ? lower1 : upper1;
        }
        mrope_utils::pack_f32_to_vec_t(out_pair_0, out0_f32);
        mrope_utils::pack_f32_to_vec_t(out_pair_1, out1_f32);
    }
    else
    {
        float out0_f32[VEC_PAIR], out1_f32[VEC_PAIR];
#pragma unroll
        for(int i = 0; i < VEC_PAIR / 2; ++i)
        {
            out0_f32[2 * i + 0] =
                (float)x_pair_0[2 * i + 0] * (float)cos_vec_pair[i] -
                (float)x_pair_0[2 * i + 1] * (float)sin_vec_pair[i];
            out0_f32[2 * i + 1] =
                (float)x_pair_0[2 * i + 1] * (float)cos_vec_pair[i] +
                (float)x_pair_0[2 * i + 0] * (float)sin_vec_pair[i];
            out1_f32[2 * i + 0] =
                (float)x_pair_1[2 * i + 0] * (float)cos_vec_pair[i] -
                (float)x_pair_1[2 * i + 1] * (float)sin_vec_pair[i];
            out1_f32[2 * i + 1] =
                (float)x_pair_1[2 * i + 1] * (float)cos_vec_pair[i] +
                (float)x_pair_1[2 * i + 0] * (float)sin_vec_pair[i];
        }
        mrope_utils::pack_f32_to_vec_t(out_pair_0, out0_f32);
        mrope_utils::pack_f32_to_vec_t(out_pair_1, out1_f32);
    }

    // ===========================================================
    // Stores: 2 × dwordx4
    // ===========================================================
    if(is_q)
    {
        const int64_t base_off =
            (static_cast<int64_t>(token_id) * num_heads_q + head0_in_token) * HEAD_SIZE;
        out_pair_0.store(out_q + base_off + lane_full * VEC_PAIR);
        out_pair_1.store(out_q + base_off + 2 * HEAD_SIZE + lane_full * VEC_PAIR);
    }
    else
    {
        const int64_t base_off =
            (static_cast<int64_t>(token_id) * num_heads_k + head0_in_token) * HEAD_SIZE;
        out_pair_0.store(out_k + base_off + lane_full * VEC_PAIR);
        out_pair_1.store(out_k + base_off + 2 * HEAD_SIZE + lane_full * VEC_PAIR);
    }
}

template <typename T>
void fused_rope_rms_1way(const T* q,
                         const T* k,
                         const T* w_q,
                         const T* w_k,
                         const float* cos_sin,
                         int64_t batch_size,
                         int64_t num_tokens,
                         int64_t num_heads_q,
                         int64_t num_heads_k,
                         int64_t head_size,
                         bool is_interleaved,
                         double eps,
                         T* out_q,
                         T* out_k,
                         hipStream_t stream)
{
    using mrope_utils::WARP_SIZE;
    AITER_CHECK(head_size == 64 || head_size == 128 || head_size == 256);
    constexpr int block_size = 256;
    auto num_warps_per_block = block_size / WARP_SIZE;
    dim3 threadsPerBlock(block_size);

    // Quad fast path: 1 warp processes 4 heads (2 adjacent head_pairs of the
    // same token, half-warp layout, all input loads bundled into the prologue
    // so 4 HBM round-trips overlap). See the kernel-side comment on
    // fused_rope_rms_1way_quad_kernel for the full derivation; requires
    // num_heads_q % 4 == 0 && num_heads_k % 4 == 0.
    const bool can_quad = (num_heads_q % 4 == 0) && (num_heads_k % 4 == 0);
    if(can_quad)
    {
        auto total_warps_quad = num_tokens * ((num_heads_q + num_heads_k) / 4);
        dim3 numBlocks(
            (total_warps_quad + num_warps_per_block - 1) / num_warps_per_block, batch_size);
        // Inner macro: pick (IS_NEOX, QUAD_Q_CT, QUAD_K_CT) and launch.
        // QUAD_Q_CT/QUAD_K_CT = 0 means runtime division; > 0 makes the
        // div/mod inside the kernel a constant-divisor magic-multiply.
#define DISPATCH_NEOX_QUAD_CT(HEAD_SIZE, QQ, QK)                            \
    if(!is_interleaved)                                                     \
    {                                                                       \
        fused_rope_rms_1way_quad_kernel<T, HEAD_SIZE, true, QQ, QK>         \
            <<<numBlocks, threadsPerBlock, 0, stream>>>(q,                  \
                                                        k,                  \
                                                        w_q,                \
                                                        w_k,                \
                                                        cos_sin,            \
                                                        num_tokens,         \
                                                        num_heads_q,        \
                                                        num_heads_k,        \
                                                        eps,                \
                                                        total_warps_quad,   \
                                                        out_q,              \
                                                        out_k);             \
    }                                                                       \
    else                                                                    \
    {                                                                       \
        fused_rope_rms_1way_quad_kernel<T, HEAD_SIZE, false, QQ, QK>        \
            <<<numBlocks, threadsPerBlock, 0, stream>>>(q,                  \
                                                        k,                  \
                                                        w_q,                \
                                                        w_k,                \
                                                        cos_sin,            \
                                                        num_tokens,         \
                                                        num_heads_q,        \
                                                        num_heads_k,        \
                                                        eps,                \
                                                        total_warps_quad,   \
                                                        out_q,              \
                                                        out_k);             \
    }
        // Outer macro: route common (num_heads_q, num_heads_k) shapes to the
        // compile-time-divisor specialization, default to runtime path.
        // Specialized list intentionally short — each adds ~one .so MB after
        // template expansion across (T, HEAD_SIZE, IS_NEOX) → ~24 instances.
#define DISPATCH_NEOX_QUAD(HEAD_SIZE)                              \
    if(num_heads_q == 24 && num_heads_k == 24)                     \
    {                                                              \
        DISPATCH_NEOX_QUAD_CT(HEAD_SIZE, 6, 6)                     \
    }                                                              \
    else if(num_heads_q == 32 && num_heads_k == 32)                \
    {                                                              \
        DISPATCH_NEOX_QUAD_CT(HEAD_SIZE, 8, 8)                     \
    }                                                              \
    else if(num_heads_q == 16 && num_heads_k == 16)                \
    {                                                              \
        DISPATCH_NEOX_QUAD_CT(HEAD_SIZE, 4, 4)                     \
    }                                                              \
    else                                                           \
    {                                                              \
        DISPATCH_NEOX_QUAD_CT(HEAD_SIZE, 0, 0)                     \
    }
        switch(head_size)
        {
        case 64: DISPATCH_NEOX_QUAD(64) break;
        case 128: DISPATCH_NEOX_QUAD(128) break;
        case 256: DISPATCH_NEOX_QUAD(256) break;
        }
#undef DISPATCH_NEOX_QUAD
#undef DISPATCH_NEOX_QUAD_CT
        return;
    }

    // Fallback: num_heads_q or num_heads_k is not divisible by 4. Use the
    // single-head-per-warp kernel — slower but works for any shape.
    auto total_warps = num_tokens * (num_heads_q + num_heads_k);
    dim3 numBlocks((total_warps + num_warps_per_block - 1) / num_warps_per_block, batch_size);
#define DISPATCH_NEOX(HEAD_SIZE)                                    \
    if(!is_interleaved)                                             \
    {                                                               \
        fused_rope_rms_1way_kernel<T, HEAD_SIZE, true>              \
            <<<numBlocks, threadsPerBlock, 0, stream>>>(q,          \
                                                        k,          \
                                                        w_q,        \
                                                        w_k,        \
                                                        cos_sin,    \
                                                        num_tokens, \
                                                        num_heads_q,\
                                                        num_heads_k,\
                                                        eps,        \
                                                        total_warps,\
                                                        out_q,      \
                                                        out_k);     \
    }                                                               \
    else                                                            \
    {                                                               \
        fused_rope_rms_1way_kernel<T, HEAD_SIZE, false>             \
            <<<numBlocks, threadsPerBlock, 0, stream>>>(q,          \
                                                        k,          \
                                                        w_q,        \
                                                        w_k,        \
                                                        cos_sin,    \
                                                        num_tokens, \
                                                        num_heads_q,\
                                                        num_heads_k,\
                                                        eps,        \
                                                        total_warps,\
                                                        out_q,      \
                                                        out_k);     \
    }
    switch(head_size)
    {
    case 64: DISPATCH_NEOX(64) break;
    case 128: DISPATCH_NEOX(128) break;
    case 256: DISPATCH_NEOX(256) break;
    }

#undef DISPATCH_NEOX
}

template <typename T, int HEAD_SIZE, bool IS_NEOX>
__global__ void fused_rope_rms_2way_amax_kernel(const T* q0_,
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
                                                T* out_k01_,
                                                float* q_partial_amax,
                                                float* k_partial_amax)
{
    using mrope_utils::WARP_SIZE;
    constexpr int VEC_SIZE        = HEAD_SIZE / WARP_SIZE;
    constexpr int PAIR_VEC_SIZE   = VEC_SIZE / 2;
    constexpr int HALF_HEAD_SIZE  = HEAD_SIZE / 2;
    const int warp_id             = threadIdx.x / WARP_SIZE;
    const int lane_id             = threadIdx.x % WARP_SIZE;
    const int num_warps_per_block = blockDim.x / WARP_SIZE;
    const int global_warp_id      = blockIdx.x * num_warps_per_block + warp_id;
    if(global_warp_id >= total_warps)
    {
        return;
    }

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

    int access_id_in_head = lane_id * VEC_SIZE;
    int neighbor_offset =
        access_id_in_head < HALF_HEAD_SIZE ? HALF_HEAD_SIZE / VEC_SIZE : -HALF_HEAD_SIZE / VEC_SIZE;

    int token_id;
    int specialized_warp_id;
    int head_id_in_token;
    int data_offset;

    vec_t<T, VEC_SIZE> w_vec, x_vec, cos_sin_vec;
    vec_t<T, PAIR_VEC_SIZE> cos_vec, sin_vec;

    if(is_q0)
    {
        specialized_warp_id = global_warp_id - warp_offset_q0;
        token_id            = specialized_warp_id / num_heads_q;
        head_id_in_token    = specialized_warp_id % num_heads_q;
        data_offset         = (token_id * num_heads_q + head_id_in_token) * HEAD_SIZE;
        w_vec.load(w_q0 + access_id_in_head);
        x_vec.load(q0 + data_offset + access_id_in_head);
        if constexpr(IS_NEOX)
            cos_sin_vec.load(&cos_sin0[token_id * HEAD_SIZE + access_id_in_head]);
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
            cos_sin_vec.load(&cos_sin0[token_id * HEAD_SIZE + access_id_in_head]);
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
            cos_sin_vec.load(&cos_sin1[token_id * HEAD_SIZE + access_id_in_head]);
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
            cos_sin_vec.load(&cos_sin1[token_id * HEAD_SIZE + access_id_in_head]);
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
                out_vec[i] = (float)x_vec[i] * (float)cos_sin_vec[i] -
                             (float)nb_x_vec[i] * (float)nb_cos_sin_vec[i];
        }
        else
        {
#pragma unroll
            for(int i = 0; i < VEC_SIZE; ++i)
                out_vec[i] = (float)x_vec[i] * (float)nb_cos_sin_vec[i] +
                             (float)nb_x_vec[i] * (float)cos_sin_vec[i];
        }
    }
    else
    {
#pragma unroll
        for(int i = 0; i < PAIR_VEC_SIZE; ++i)
        {
            out_vec[2 * i + 0] = (float)x_vec[2 * i + 0] * (float)cos_vec[i] -
                                 (float)x_vec[2 * i + 1] * (float)sin_vec[i];
            out_vec[2 * i + 1] = (float)x_vec[2 * i + 1] * (float)cos_vec[i] +
                                 (float)x_vec[2 * i + 0] * (float)sin_vec[i];
        }
    }

    float local_max = 0.0f;
#pragma unroll
    for(int i = 0; i < VEC_SIZE; ++i)
        local_max = fmaxf(local_max, fabsf((float)out_vec[i]));
#pragma unroll
    for(int mask = 16; mask > 0; mask >>= 1)
        local_max = fmaxf(local_max, __shfl_xor(local_max, mask, WARP_SIZE));
    if(lane_id == 0)
    {
        if(is_q0 || is_q1)
        {
            q_partial_amax[blockIdx.y * total_warps + global_warp_id] = local_max;
            k_partial_amax[blockIdx.y * total_warps + global_warp_id] = 0.0f;
        }
        else
        {
            q_partial_amax[blockIdx.y * total_warps + global_warp_id] = 0.0f;
            k_partial_amax[blockIdx.y * total_warps + global_warp_id] = local_max;
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
// Per-head scale reduction for the 2way layout. q_partial_amax / k_partial_amax
// are produced by fused_rope_rms_2way_amax_kernel: shape [batch, total_warps],
// where total_warps lays out 4 contiguous segments
//   q0 [num_tokens0 * num_heads_q],
//   k0 [num_tokens0 * num_heads_k],
//   q1 [num_tokens1 * num_heads_q],
//   k1 [num_tokens1 * num_heads_k].
// Slots that do not match a side are written as 0, so we only read the segments
// for the requested side.
__global__ void qk_partial_amax_to_perhead_scale_kernel(
    const float* q_partial_amax,
    const float* k_partial_amax,
    int num_tokens0,
    int num_tokens1,
    int num_heads_q,
    int num_heads_k,
    int total_warps,
    float* q_scale, // [batch, num_heads_q]
    float* k_scale) // [batch, num_heads_k]
{
    int b = blockIdx.y;
    int head_packed = blockIdx.x; // 0 .. num_heads_q + num_heads_k - 1
    bool is_q = head_packed < num_heads_q;
    int h = is_q ? head_packed : head_packed - num_heads_q;
    int H = is_q ? num_heads_q : num_heads_k;

    int warp_offset_seg0 = is_q ? 0 : num_tokens0 * num_heads_q;
    int warp_offset_seg1 = is_q
        ? num_tokens0 * (num_heads_q + num_heads_k)
        : num_tokens0 * (num_heads_q + num_heads_k) + num_tokens1 * num_heads_q;

    const float* base = (is_q ? q_partial_amax : k_partial_amax) + (int64_t)b * total_warps;

    float local = 0.0f;
    for(int t = threadIdx.x; t < num_tokens0; t += blockDim.x)
        local = fmaxf(local, base[warp_offset_seg0 + t * H + h]);
    for(int t = threadIdx.x; t < num_tokens1; t += blockDim.x)
        local = fmaxf(local, base[warp_offset_seg1 + t * H + h]);

    __shared__ float sm[256];
    sm[threadIdx.x] = local;
    __syncthreads();
#pragma unroll
    for(int s = 128; s > 0; s >>= 1)
    {
        if(threadIdx.x < s)
            sm[threadIdx.x] = fmaxf(sm[threadIdx.x], sm[threadIdx.x + s]);
        __syncthreads();
    }
    if(threadIdx.x == 0)
    {
        constexpr float fp8_max = 240.0f;
        float* out             = is_q ? q_scale : k_scale;
        out[b * H + h]         = fmaxf(sm[0], 1e-8f) / fp8_max;
    }
}

// FP8 static quant where each (batch, head) carries its own scale.
// Input/output shape: [batch, num_tokens, num_heads, head_size].
template <typename T>
__global__ void static_fp8_quant_perhead_kernel(mrope_utils::fp8e4m3fnuz* out,
                                                const T* input,
                                                const float* scale, // [batch, num_heads]
                                                int batch_size,
                                                int num_tokens,
                                                int num_heads,
                                                int head_size)
{
    int64_t idx    = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    int64_t numel  = (int64_t)batch_size * num_tokens * num_heads * head_size;
    for(int64_t i = idx; i < numel; i += stride)
    {
        int64_t tmp = i / head_size;
        int h       = tmp % num_heads;
        tmp /= num_heads;
        int b         = tmp / num_tokens;
        float inv     = 1.0f / scale[b * num_heads + h];
        out[i] = mrope_utils::fp8e4m3fnuz(static_cast<float>(input[i]) * inv);
    }
}

template <typename scalar_t, int PackSize>
__device__ __forceinline__ vec_t<scalar_t, PackSize>
minimax_apply_neox_rope_pack(vec_t<scalar_t, PackSize> x,
                             scalar_t const* __restrict__ cos_sin_cache,
                             int64_t const* __restrict__ position_ids,
                             int token_idx,
                             int access_id_in_head,
                             int rotary_dim)
{
    int const embed_dim       = rotary_dim / 2;
    int64_t const pos_id      = position_ids[token_idx];
    scalar_t const* cache_ptr = cos_sin_cache + pos_id * rotary_dim;
    scalar_t const* cos_ptr   = cache_ptr;
    scalar_t const* sin_ptr   = cache_ptr + embed_dim;
    int const cos_base =
        access_id_in_head < embed_dim ? access_id_in_head : access_id_in_head - embed_dim;
    int const partner_delta = embed_dim / PackSize;

    vec_t<scalar_t, PackSize> y;
#pragma unroll
    for(int i = 0; i < PackSize; ++i)
    {
        int const dim = access_id_in_head + i;
        if(dim < rotary_dim)
        {
            float const self = static_cast<float>(x[i]);
            float const peer = __shfl_xor(self, partner_delta, WARP_SIZE);
            float const c    = static_cast<float>(cos_ptr[cos_base + i]);
            float const s    = static_cast<float>(sin_ptr[cos_base + i]);
            y[i]             = dim < embed_dim ? static_cast<scalar_t>(self * c - peer * s)
                                               : static_cast<scalar_t>(self * c + peer * s);
        }
        else
        {
            y[i] = x[i];
        }
    }
    return y;
}

template <typename scalar_t, int PackSize>
__device__ __forceinline__ vec_t<scalar_t, PackSize>
minimax_apply_gptj_rope_pack(vec_t<scalar_t, PackSize> x,
                             scalar_t const* __restrict__ cos_sin_cache,
                             int64_t const* __restrict__ position_ids,
                             int token_idx,
                             int access_id_in_head,
                             int rotary_dim)
{
    int const embed_dim       = rotary_dim / 2;
    int64_t const pos_id      = position_ids[token_idx];
    scalar_t const* cache_ptr = cos_sin_cache + pos_id * rotary_dim;
    scalar_t const* cos_ptr   = cache_ptr;
    scalar_t const* sin_ptr   = cache_ptr + embed_dim;

    vec_t<scalar_t, PackSize> y = x;
#pragma unroll
    for(int i = 0; i < PackSize; i += 2)
    {
        int const dim = access_id_in_head + i;
        if(dim + 1 < rotary_dim)
        {
            float const x0 = static_cast<float>(x[i]);
            float const x1 = static_cast<float>(x[i + 1]);
            float const c  = static_cast<float>(cos_ptr[dim / 2]);
            float const s  = static_cast<float>(sin_ptr[dim / 2]);
            y[i]           = static_cast<scalar_t>(x0 * c - x1 * s);
            y[i + 1]       = static_cast<scalar_t>(x1 * c + x0 * s);
        }
    }
    return y;
}

template <typename scalar_t, typename weight_t, int BlockSize>
__global__ void minimax_qk_norm_rope_kernel(
    scalar_t const* __restrict__ qkv,
    weight_t const* __restrict__ q_weight,
    weight_t const* __restrict__ k_weight,
    scalar_t const* __restrict__ cos_sin_cache,
    int64_t const* __restrict__ position_ids,
    int const num_heads_q,
    int const num_heads_k,
    int const head_dim,
    int const rotary_dim,
    float const eps,
    bool const is_neox,
    scalar_t* __restrict__ q_out,
    scalar_t* __restrict__ k_out,
    scalar_t* __restrict__ v_out)
{
    constexpr int PackSize = 16 / sizeof(scalar_t);
    using pack_t           = vec_t<scalar_t, PackSize>;
    using weight_pack_t    = vec_t<weight_t, PackSize>;

    int const token_idx = blockIdx.x;
    int const tid       = threadIdx.x;
    int const q_size    = num_heads_q * head_dim;
    int const kv_size   = num_heads_k * head_dim;
    int const hidden_dim = q_size + 2 * kv_size;
    int const q_packs    = q_size / PackSize;
    int const k_packs    = kv_size / PackSize;
    int const qk_packs   = q_packs + k_packs;
    int const all_packs  = hidden_dim / PackSize;
    int const pack_start = tid * PackSize;
    int64_t const qkv_row_stride = static_cast<int64_t>(hidden_dim);

    scalar_t const* q_ptr = qkv + static_cast<int64_t>(token_idx) * qkv_row_stride;
    scalar_t const* k_ptr = q_ptr + q_size;
    scalar_t const* v_ptr = k_ptr + kv_size;

    pack_t x{};
    if(tid < all_packs)
    {
        x = *reinterpret_cast<pack_t const*>(q_ptr + pack_start);
    }
    float acc[PackSize];
#pragma unroll
    for(int i = 0; i < PackSize; ++i)
    {
        acc[i] = static_cast<float>(x[i]);
    }

    float q_square_sum = 0.0f;
    float k_square_sum = 0.0f;
    if(tid < qk_packs)
    {
#pragma unroll
        for(int i = 0; i < PackSize; ++i)
        {
            if(tid < q_packs)
            {
                q_square_sum += acc[i] * acc[i];
            }
            else
            {
                k_square_sum += acc[i] * acc[i];
            }
        }
    }
    auto sum_op = [](float a, float b) { return a + b; };
    q_square_sum = block_reduce<float, decltype(sum_op), BlockSize, true>(q_square_sum, sum_op);
    __syncthreads();
    k_square_sum = block_reduce<float, decltype(sum_op), BlockSize, true>(k_square_sum, sum_op);
    float const q_rstd = rsqrtf(q_square_sum / static_cast<float>(q_size) + eps);
    float const k_rstd = rsqrtf(k_square_sum / static_cast<float>(kv_size) + eps);

    if(tid < q_packs)
    {
        weight_pack_t w = *reinterpret_cast<weight_pack_t const*>(q_weight + pack_start);
#pragma unroll
        for(int i = 0; i < PackSize; ++i)
        {
            x[i] = static_cast<scalar_t>(acc[i] * q_rstd * static_cast<float>(w[i]));
        }
        int const access_id_in_head = pack_start % head_dim;
        x = is_neox ? minimax_apply_neox_rope_pack<scalar_t, PackSize>(
                          x, cos_sin_cache, position_ids, token_idx, access_id_in_head, rotary_dim)
                    : minimax_apply_gptj_rope_pack<scalar_t, PackSize>(
                          x, cos_sin_cache, position_ids, token_idx, access_id_in_head, rotary_dim);
        *reinterpret_cast<pack_t*>(q_out + static_cast<int64_t>(token_idx) * q_size + pack_start) =
            x;
    }
    else if(tid < qk_packs)
    {
        int const k_pack_id = tid - q_packs;
        int const k_start   = k_pack_id * PackSize;
        weight_pack_t w     = *reinterpret_cast<weight_pack_t const*>(k_weight + k_start);
#pragma unroll
        for(int i = 0; i < PackSize; ++i)
        {
            x[i] = static_cast<scalar_t>(acc[i] * k_rstd * static_cast<float>(w[i]));
        }
        int const access_id_in_head = k_start % head_dim;
        x = is_neox ? minimax_apply_neox_rope_pack<scalar_t, PackSize>(
                          x, cos_sin_cache, position_ids, token_idx, access_id_in_head, rotary_dim)
                    : minimax_apply_gptj_rope_pack<scalar_t, PackSize>(
                          x, cos_sin_cache, position_ids, token_idx, access_id_in_head, rotary_dim);
        *reinterpret_cast<pack_t*>(k_out + static_cast<int64_t>(token_idx) * kv_size + k_start) =
            x;
    }
    else if(tid < all_packs)
    {
        int const v_pack_id = tid - qk_packs;
        int const v_start   = v_pack_id * PackSize;
        *reinterpret_cast<pack_t*>(v_out + static_cast<int64_t>(token_idx) * kv_size + v_start) =
            x;
    }
}

namespace aiter {

void minimax_qk_norm_rope(aiter_tensor_t& qkv,
                              aiter_tensor_t& q_weight,
                              aiter_tensor_t& k_weight,
                              aiter_tensor_t& cos_sin_cache,
                              aiter_tensor_t& position_ids,
                              int64_t num_heads_q,
                              int64_t num_heads_k,
                              int64_t head_dim,
                              int64_t rotary_dim,
                              double eps,
                              bool is_neox,
                              aiter_tensor_t& q_out,
                              aiter_tensor_t& k_out,
                              aiter_tensor_t& v_out)
{
    CHECK_INPUT(qkv);
    CHECK_INPUT(q_weight);
    CHECK_INPUT(k_weight);
    CHECK_INPUT(cos_sin_cache);
    CHECK_INPUT(position_ids);
    CHECK_INPUT(q_out);
    CHECK_INPUT(k_out);
    CHECK_INPUT(v_out);
    CHECK_TYPE(position_ids, AITER_DTYPE_i64);

    AITER_CHECK(qkv.dim() == 2, "qkv must be 2D [num_tokens, q_size + 2 * kv_size]");
    AITER_CHECK(q_weight.dim() == 1, "q_weight must be 1D [num_heads_q * head_dim]");
    AITER_CHECK(k_weight.dim() == 1, "k_weight must be 1D [num_heads_k * head_dim]");
    AITER_CHECK(cos_sin_cache.dim() == 2,
                "cos_sin_cache must be 2D [max_position, rotary_dim]");
    AITER_CHECK(position_ids.dim() == 1, "position_ids must be 1D [num_tokens]");
    AITER_CHECK(q_out.dim() == 2 && k_out.dim() == 2 && v_out.dim() == 2,
                "q_out, k_out and v_out must be 2D");
    AITER_CHECK(num_heads_q > 0 && num_heads_k > 0 && head_dim > 0,
                "num_heads_q, num_heads_k and head_dim must be positive");
    AITER_CHECK(rotary_dim > 0 && rotary_dim <= head_dim && rotary_dim % 2 == 0,
                "rotary_dim must be positive, even and <= head_dim");

    int64_t const num_tokens = qkv.size(0);
    int64_t const q_size     = num_heads_q * head_dim;
    int64_t const kv_size    = num_heads_k * head_dim;
    AITER_CHECK(qkv.size(1) == q_size + 2 * kv_size,
                "qkv dim 1 must equal q_size + 2 * kv_size");
    AITER_CHECK(q_weight.size(0) == q_size,
                "q_weight size must equal num_heads_q * head_dim for MiniMax TP1");
    AITER_CHECK(k_weight.size(0) == kv_size,
                "k_weight size must equal num_heads_k * head_dim for MiniMax TP1");
    AITER_CHECK(cos_sin_cache.size(1) == rotary_dim,
                "cos_sin_cache dim 1 must equal rotary_dim");
    AITER_CHECK(position_ids.size(0) == num_tokens,
                "position_ids size must match qkv num_tokens");
    AITER_CHECK(q_out.size(0) == num_tokens && q_out.size(1) == q_size,
                "q_out must be [num_tokens, num_heads_q * head_dim]");
    AITER_CHECK(k_out.size(0) == num_tokens && k_out.size(1) == kv_size,
                "k_out must be [num_tokens, num_heads_k * head_dim]");
    AITER_CHECK(v_out.size(0) == num_tokens && v_out.size(1) == kv_size,
                "v_out must be [num_tokens, num_heads_k * head_dim]");
    AITER_CHECK(q_weight.dtype() == k_weight.dtype(),
                "q_weight and k_weight must have the same dtype");
    AITER_CHECK(q_weight.dtype() == qkv.dtype() || q_weight.dtype() == AITER_DTYPE_fp32,
                "MiniMax TP1 fused kernel supports q/k weights in qkv dtype or fp32");
    AITER_CHECK(qkv.dtype() == cos_sin_cache.dtype() && qkv.dtype() == q_out.dtype() &&
                    qkv.dtype() == k_out.dtype() && qkv.dtype() == v_out.dtype(),
                "qkv, cos_sin_cache and outputs must have the same dtype");

    HipDeviceGuard device_guard(qkv.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    constexpr int block_size = 1024;

    VLLM_DISPATCH_FLOATING_TYPES_rmTorch(qkv.dtype(), "minimax_qk_norm_rope", [&] {
        constexpr int pack_size = 16 / sizeof(scalar_t);
        AITER_CHECK(q_size % pack_size == 0 && kv_size % pack_size == 0 &&
                        head_dim % pack_size == 0,
                    "MiniMax TP1 fused kernel requires q/k/head dims to be 16B-pack aligned");
        AITER_CHECK((rotary_dim / 2) % pack_size == 0,
                    "MiniMax TP1 fused kernel requires rotary_dim / 2 to be pack aligned");
        AITER_CHECK((q_size + 2 * kv_size) / pack_size <= block_size,
                    "MiniMax TP1 fused kernel supports at most 1024 packs per token");
        dim3 grid(num_tokens);
        dim3 block(block_size);
        if(q_weight.dtype() == AITER_DTYPE_fp32)
        {
            minimax_qk_norm_rope_kernel<scalar_t, float, block_size>
                <<<grid, block, 0, stream>>>(
                    reinterpret_cast<scalar_t const*>(qkv.data_ptr()),
                    reinterpret_cast<float const*>(q_weight.data_ptr()),
                    reinterpret_cast<float const*>(k_weight.data_ptr()),
                    reinterpret_cast<scalar_t const*>(cos_sin_cache.data_ptr()),
                    reinterpret_cast<int64_t const*>(position_ids.data_ptr()),
                    static_cast<int>(num_heads_q),
                    static_cast<int>(num_heads_k),
                    static_cast<int>(head_dim),
                    static_cast<int>(rotary_dim),
                    static_cast<float>(eps),
                    is_neox,
                    reinterpret_cast<scalar_t*>(q_out.data_ptr()),
                    reinterpret_cast<scalar_t*>(k_out.data_ptr()),
                    reinterpret_cast<scalar_t*>(v_out.data_ptr()));
        }
        else
        {
            minimax_qk_norm_rope_kernel<scalar_t, scalar_t, block_size>
                <<<grid, block, 0, stream>>>(
                    reinterpret_cast<scalar_t const*>(qkv.data_ptr()),
                    reinterpret_cast<scalar_t const*>(q_weight.data_ptr()),
                    reinterpret_cast<scalar_t const*>(k_weight.data_ptr()),
                    reinterpret_cast<scalar_t const*>(cos_sin_cache.data_ptr()),
                    reinterpret_cast<int64_t const*>(position_ids.data_ptr()),
                    static_cast<int>(num_heads_q),
                    static_cast<int>(num_heads_k),
                    static_cast<int>(head_dim),
                    static_cast<int>(rotary_dim),
                    static_cast<float>(eps),
                    is_neox,
                    reinterpret_cast<scalar_t*>(q_out.data_ptr()),
                    reinterpret_cast<scalar_t*>(k_out.data_ptr()),
                    reinterpret_cast<scalar_t*>(v_out.data_ptr()));
        }
    });
}

void fused_qk_norm_rope_cache_quant_shuffle(
    aiter_tensor_t& q,
    aiter_tensor_t& k,
    aiter_tensor_t& v,
    int64_t num_heads_q,               // Number of query heads
    int64_t num_heads_k,               // Number of key heads
    int64_t num_heads_v,               // Number of value heads
    int64_t head_dim,                  // Dimension per head
    double eps,                        // Epsilon for RMS normalization
    aiter_tensor_t& q_weight,          // RMSNorm weights for query [head_dim]
    aiter_tensor_t& k_weight,          // RMSNorm weights for key [head_dim]
    aiter_tensor_t& cos_sin_cache,     // Cos/sin cache [max_position, rotary_dim]
    bool is_neox,                      // Whether RoPE is applied in Neox style
    aiter_tensor_t& position_ids,      // Position IDs for RoPE [num_tokens]
    aiter_tensor_t& k_cache,           // [num_blocks, num_kv_heads, head_dim//x, page_size, x]
    aiter_tensor_t& v_cache,           // 4D [num_blocks, num_heads_v, head_dim, page_size] or 5D shuffle
                                       // [num_blocks, num_heads_v, page_size//x, head_dim, x]
    aiter_tensor_t& slot_mapping,      // slot mapping
    const std::string& kv_cache_dtype, // kv cache data type
    std::optional<aiter_tensor_t> k_scale, // k scale tensor for quantized k cache
    std::optional<aiter_tensor_t> v_scale  // v scale tensor for quantized v cache
)
{
    CHECK_INPUT(position_ids);
    CHECK_INPUT(q_weight);
    CHECK_INPUT(k_weight);
    CHECK_INPUT(cos_sin_cache);
    CHECK_INPUT(k_cache);
    CHECK_INPUT(v_cache);
    CHECK_INPUT(slot_mapping);
    CHECK_TH_CUDA(q);
    CHECK_TH_CUDA(k);
    CHECK_TH_CUDA(v);
    CHECK_TYPE(position_ids, AITER_DTYPE_i64);
    CHECK_TYPE(slot_mapping, AITER_DTYPE_i64);

    AITER_CHECK(position_ids.dim() == 1, "Position IDs must be 1D: [num_tokens]");
    AITER_CHECK(q_weight.dim() == 1, "Query weights must be 1D: [head_dim]");
    AITER_CHECK(k_weight.dim() == 1, "Key weights must be 1D: [head_dim]");
    AITER_CHECK(cos_sin_cache.dim() == 2, "Cos/sin cache must be 2D: [max_position, rotary_dim]");
    AITER_CHECK(q_weight.size(0) == head_dim, "Query weights size must match head dimension");
    AITER_CHECK(k_weight.size(0) == head_dim, "Key weights size must match head dimension");
    int64_t const rotary_dim_ = cos_sin_cache.size(1);
    AITER_CHECK(rotary_dim_ > 0, "rotary_dim must be positive");
    AITER_CHECK(rotary_dim_ <= head_dim,
                "rotary_dim (",
                rotary_dim_,
                ") must be <= head_dim (",
                head_dim,
                ")");
    AITER_CHECK(rotary_dim_ % 2 == 0, "rotary_dim must be even");
    if(is_neox)
    {
        int64_t const num_elems_per_thread = head_dim / 32;
        AITER_CHECK((rotary_dim_ / 2) % num_elems_per_thread == 0,
                    "For NeoX-style partial rotary, rotary_dim/2 (",
                    rotary_dim_ / 2,
                    ") must be divisible by head_dim/32 (",
                    num_elems_per_thread,
                    ")");
    }
    AITER_CHECK(head_dim % 32 == 0,
                "Head dimension must be multiple of 32 for fused QK Norm RoPE kernel");
    AITER_CHECK(
        num_heads_k <= 32,
        "Number of key heads must be less than or equal to 32 for fused QK Norm RoPE kernel");

    int64_t num_tokens = 0;
    AiterDtype act_dtype{};

    int64_t q_stride_token = 0, q_stride_head = 0, q_stride_dim = 0;
    int64_t k_stride_token = 0, k_stride_head = 0, k_stride_dim = 0;
    int64_t v_stride_token = 0, v_stride_head = 0, v_stride_dim = 0;
    aiter_tensor_t const& q_t = q;
    aiter_tensor_t const& k_t = k;
    aiter_tensor_t const& v_t = v;

    AITER_CHECK((q_t.dim() == 2 || q_t.dim() == 3) && (k_t.dim() == 2 || k_t.dim() == 3) &&
                    (v_t.dim() == 2 || v_t.dim() == 3),
                "q, k, v must be 2D [num_tokens, num_heads * head_dim] or 3D [num_tokens, "
                "num_heads, head_dim]");
    num_tokens = q_t.size(0);
    AITER_CHECK(k_t.size(0) == num_tokens && v_t.size(0) == num_tokens,
                "q, k, v must share the same num_tokens");
    if(q_t.dim() == 2)
    {
        AITER_CHECK(q_t.size(1) == num_heads_q * head_dim, "q dim 1 must be num_heads_q * head_dim");
    }
    else
    {
        AITER_CHECK(q_t.size(1) == num_heads_q && q_t.size(2) == head_dim,
                    "q 3D shape must be [num_tokens, num_heads_q, head_dim]");
    }
    if(k_t.dim() == 2)
    {
        AITER_CHECK(k_t.size(1) == num_heads_k * head_dim, "k dim 1 must be num_heads_k * head_dim");
    }
    else
    {
        AITER_CHECK(k_t.size(1) == num_heads_k && k_t.size(2) == head_dim,
                    "k 3D shape must be [num_tokens, num_heads_k, head_dim]");
    }
    if(v_t.dim() == 2)
    {
        AITER_CHECK(v_t.size(1) == num_heads_v * head_dim, "v dim 1 must be num_heads_v * head_dim");
    }
    else
    {
        AITER_CHECK(v_t.size(1) == num_heads_v && v_t.size(2) == head_dim,
                    "v 3D shape must be [num_tokens, num_heads_v, head_dim]");
    }
    AITER_CHECK(q_t.dtype() == k_t.dtype() && q_t.dtype() == v_t.dtype(),
                "q, k, v must share the same dtype");
    AITER_CHECK(q_t.dtype() == q_weight.dtype() && q_t.dtype() == k_weight.dtype(),
                "q/k/v must match q_weight/k_weight dtype");
    act_dtype                    = q_t.dtype();
    ActivationStrides3D const sq = activation_strides_logical_3d(q_t, num_heads_q, head_dim);
    ActivationStrides3D const sk = activation_strides_logical_3d(k_t, num_heads_k, head_dim);
    ActivationStrides3D const sv = activation_strides_logical_3d(v_t, num_heads_v, head_dim);
    q_stride_token               = sq.st;
    q_stride_head                = sq.sh;
    q_stride_dim                 = sq.sd;
    k_stride_token               = sk.st;
    k_stride_head                = sk.sh;
    k_stride_dim                 = sk.sd;
    v_stride_token               = sv.st;
    v_stride_head                = sv.sh;
    v_stride_dim                 = sv.sd;

    AITER_CHECK(position_ids.size(0) == num_tokens,
                "Number of tokens in position_ids must match activations");

    AITER_CHECK(
        k_cache.dim() == 5,
        "k_cache must be 5D [num_blocks, num_kv_heads, head_dim//x, page_size, x], got dim ",
        k_cache.dim());
    int64_t x           = k_cache.size(-1);
    int64_t page_size_k = k_cache.size(-2);
    AITER_CHECK(x > 0 && head_dim % x == 0,
                "head_dim (",
                head_dim,
                ") must be divisible by k_cache x (",
                x,
                ")");
    AITER_CHECK(k_cache.size(2) == head_dim / x,
                "k_cache dim 2 must equal head_dim//x, got ",
                k_cache.size(2),
                " expected ",
                head_dim / x);
    AITER_CHECK(k_cache.size(1) == num_heads_k,
                "k_cache dim 1 must equal num_heads_k, got ",
                k_cache.size(1));

    int64_t page_size;
    if(v_cache.dim() == 5)
    {
        // Shuffle layout: [num_blocks, num_heads_v, page_size//x, head_dim, x]
        AITER_CHECK(v_cache.size(0) == k_cache.size(0),
                    "v_cache and k_cache num_blocks must match");
        AITER_CHECK(v_cache.size(1) == num_heads_v,
                    "v_cache dim 1 must equal num_heads_v, got ",
                    v_cache.size(1));
        AITER_CHECK(v_cache.size(-1) == x && v_cache.size(-2) == head_dim,
                    "v_cache trailing dims must be [head_dim, x], got [",
                    v_cache.size(-2),
                    ", ",
                    v_cache.size(-1),
                    "]");
        AITER_CHECK(v_cache.size(-3) * x == page_size_k,
                    "v_cache shuffle: size(-3)*x must equal k_cache page_size; got ",
                    v_cache.size(-3),
                    "*",
                    x,
                    " vs ",
                    page_size_k);
        page_size = page_size_k;
    }
    else if(v_cache.dim() == 4)
    {
        // [num_blocks, num_heads_v, head_dim, page_size]
        AITER_CHECK(v_cache.size(0) == k_cache.size(0),
                    "v_cache and k_cache num_blocks must match");
        AITER_CHECK(v_cache.size(1) == num_heads_v,
                    "v_cache dim 1 must equal num_heads_v, got ",
                    v_cache.size(1));
        AITER_CHECK(v_cache.size(2) == head_dim,
                    "v_cache dim 2 must equal head_dim, got ",
                    v_cache.size(2));
        page_size = v_cache.size(-1);
        AITER_CHECK(page_size == page_size_k,
                    "v_cache page_size (last dim) must match k_cache page_size; got ",
                    page_size,
                    " vs ",
                    page_size_k);
        AITER_CHECK(page_size % x == 0,
                    "page_size must be divisible by x for V cache layout; got page_size=",
                    page_size,
                    " x=",
                    x);
    }
    else
    {
        AITER_CHECK(
            false,
            "v_cache must be 4D [num_blocks, num_heads_v, head_dim, page_size] or 5D shuffle "
            "[num_blocks, num_heads_v, page_size//x, head_dim, x], got dim ",
            v_cache.dim());
    }

    const int stream_device = q_t.device_id;
    HipDeviceGuard device_guard(stream_device);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    DISPATCH_BY_KV_CACHE_DTYPE_rmTorch(act_dtype, kv_cache_dtype, CALL_QK_NORM_ROPE_CACHE_QUANT);
}

void fused_qk_norm_rope_cache_pts_quant_shuffle(aiter_tensor_t& qkv,
                                                aiter_tensor_t& qw,
                                                aiter_tensor_t& kw,
                                                aiter_tensor_t& cos_sin,
                                                aiter_tensor_t& positions,
                                                int64_t num_tokens,
                                                int64_t num_heads_q,
                                                int64_t num_heads_k,
                                                int64_t num_heads_v,
                                                int64_t head_size,
                                                bool is_neox_style,
                                                double eps,
                                                aiter_tensor_t& q_out,
                                                aiter_tensor_t& k_cache,
                                                aiter_tensor_t& v_cache,
                                                aiter_tensor_t& slot_mapping,
                                                aiter_tensor_t& per_tensor_k_scale,
                                                aiter_tensor_t& per_tensor_v_scale,
                                                std::optional<aiter_tensor_t> k_out,
                                                std::optional<aiter_tensor_t> v_out,
                                                bool return_kv,
                                                bool use_shuffle_layout,
                                                int64_t block_size,
                                                int64_t x,
                                                int64_t rotary_dim)
{
    AITER_CHECK(qkv.is_contiguous() && qw.is_contiguous() && kw.is_contiguous() &&
                cos_sin.is_contiguous());
    AITER_CHECK(slot_mapping.is_contiguous());
    if(!(k_cache.is_contiguous() && v_cache.is_contiguous()))
    {
        // Non-contiguous block dim (e.g. vLLM [num_blocks, 2, ...] after unbind(1)) is OK as long as each block is internally contiguous.
        AITER_CHECK(is_contiguous_from_dim1(k_cache) && is_contiguous_from_dim1(v_cache),
                    "k_cache/v_cache must be contiguous within a block (dims >= 1)");
    }
    HipDeviceGuard device_guard(qkv.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    auto kv_cache_dtype      = k_cache.dtype();
    auto qkv_dtype           = qkv.dtype();
    AITER_CHECK(positions.dim() == 1, "positions must be 1D");
    float per_tensor_k_scale_ = *reinterpret_cast<float*>(per_tensor_k_scale.data_ptr());
    float per_tensor_v_scale_ = *reinterpret_cast<float*>(per_tensor_v_scale.data_ptr());
    // Per-block (dim-0) stride: == num_heads_k*HEAD_SIZE*block_size when contiguous (old formula); larger for an interleaved [num_blocks, 2, ...] cache.
    int64_t k_cache_block_stride = k_cache.stride(0);
    int64_t v_cache_block_stride = v_cache.stride(0);
    VLLM_DISPATCH_FLOATING_TYPES_rmTorch(
        qkv_dtype, "fused_qk_norm_rope_cache_pts_quant_shuffle", [&] {
            using T = scalar_t;
            if(kv_cache_dtype == qkv_dtype)
            {
                T* k_out_ptr = (return_kv && k_out.has_value())
                                   ? reinterpret_cast<T*>(k_out.value().data_ptr())
                                   : nullptr;
                T* v_out_ptr = (return_kv && v_out.has_value())
                                   ? reinterpret_cast<T*>(v_out.value().data_ptr())
                                   : nullptr;
                mrope_utils::fused_rope_rms_set_kv<T, T>(
                    reinterpret_cast<T*>(qkv.data_ptr()),
                    reinterpret_cast<T*>(qw.data_ptr()),
                    reinterpret_cast<T*>(kw.data_ptr()),
                    reinterpret_cast<T*>(cos_sin.data_ptr()),
                    reinterpret_cast<int64_t*>(positions.data_ptr()),
                    0,
                    positions.stride(0),
                    num_tokens,
                    num_heads_q,
                    num_heads_k,
                    num_heads_v,
                    head_size,
                    is_neox_style,
                    eps,
                    reinterpret_cast<T*>(q_out.data_ptr()),
                    reinterpret_cast<T*>(k_cache.data_ptr()),
                    reinterpret_cast<T*>(v_cache.data_ptr()),
                    reinterpret_cast<int64_t*>(slot_mapping.data_ptr()),
                    stream,
                    per_tensor_k_scale_,
                    per_tensor_v_scale_,
                    k_out_ptr,
                    v_out_ptr,
                    use_shuffle_layout,
                    block_size,
                    x,
                    rotary_dim,
                    k_cache_block_stride,
                    v_cache_block_stride);
            }
            else
            {
                if(kv_cache_dtype == AITER_DTYPE_fp8)
                {
                    if(is_fp8_ocp_arch())
                    {
                        mrope_utils::fp8e4m3fn* k_out_fp8_ptr =
                            (return_kv && k_out.has_value())
                                ? reinterpret_cast<mrope_utils::fp8e4m3fn*>(
                                      k_out.value().data_ptr())
                                : nullptr;
                        mrope_utils::fp8e4m3fn* v_out_fp8_ptr =
                            (return_kv && v_out.has_value())
                                ? reinterpret_cast<mrope_utils::fp8e4m3fn*>(
                                      v_out.value().data_ptr())
                                : nullptr;
                        mrope_utils::fused_rope_rms_set_kv<T, mrope_utils::fp8e4m3fn>(
                            reinterpret_cast<T*>(qkv.data_ptr()),
                            reinterpret_cast<T*>(qw.data_ptr()),
                            reinterpret_cast<T*>(kw.data_ptr()),
                            reinterpret_cast<T*>(cos_sin.data_ptr()),
                            reinterpret_cast<int64_t*>(positions.data_ptr()),
                            0,
                            positions.stride(0),
                            num_tokens,
                            num_heads_q,
                            num_heads_k,
                            num_heads_v,
                            head_size,
                            is_neox_style,
                            eps,
                            reinterpret_cast<T*>(q_out.data_ptr()),
                            reinterpret_cast<mrope_utils::fp8e4m3fn*>(k_cache.data_ptr()),
                            reinterpret_cast<mrope_utils::fp8e4m3fn*>(v_cache.data_ptr()),
                            reinterpret_cast<int64_t*>(slot_mapping.data_ptr()),
                            stream,
                            per_tensor_k_scale_,
                            per_tensor_v_scale_,
                            k_out_fp8_ptr,
                            v_out_fp8_ptr,
                            use_shuffle_layout,
                            block_size,
                            x,
                            rotary_dim,
                            k_cache_block_stride,
                            v_cache_block_stride);
                    }
                    else
                    {
                        mrope_utils::fp8e4m3fnuz* k_out_fp8_ptr =
                            (return_kv && k_out.has_value())
                                ? reinterpret_cast<mrope_utils::fp8e4m3fnuz*>(
                                      k_out.value().data_ptr())
                                : nullptr;
                        mrope_utils::fp8e4m3fnuz* v_out_fp8_ptr =
                            (return_kv && v_out.has_value())
                                ? reinterpret_cast<mrope_utils::fp8e4m3fnuz*>(
                                      v_out.value().data_ptr())
                                : nullptr;
                        mrope_utils::fused_rope_rms_set_kv<T, mrope_utils::fp8e4m3fnuz>(
                            reinterpret_cast<T*>(qkv.data_ptr()),
                            reinterpret_cast<T*>(qw.data_ptr()),
                            reinterpret_cast<T*>(kw.data_ptr()),
                            reinterpret_cast<T*>(cos_sin.data_ptr()),
                            reinterpret_cast<int64_t*>(positions.data_ptr()),
                            0,
                            positions.stride(0),
                            num_tokens,
                            num_heads_q,
                            num_heads_k,
                            num_heads_v,
                            head_size,
                            is_neox_style,
                            eps,
                            reinterpret_cast<T*>(q_out.data_ptr()),
                            reinterpret_cast<mrope_utils::fp8e4m3fnuz*>(k_cache.data_ptr()),
                            reinterpret_cast<mrope_utils::fp8e4m3fnuz*>(v_cache.data_ptr()),
                            reinterpret_cast<int64_t*>(slot_mapping.data_ptr()),
                            stream,
                            per_tensor_k_scale_,
                            per_tensor_v_scale_,
                            k_out_fp8_ptr,
                            v_out_fp8_ptr,
                            use_shuffle_layout,
                            block_size,
                            x,
                            rotary_dim,
                            k_cache_block_stride,
                            v_cache_block_stride);
                    }
                }
                else
                {
                    AITER_CHECK(false, "Unsupported KV cache dtype: ", kv_cache_dtype);
                }
            }
        });
}

void fused_qk_norm_rope_2way(aiter_tensor_t& q0,
                             aiter_tensor_t& k0,
                             aiter_tensor_t& q1,
                             aiter_tensor_t& k1,
                             aiter_tensor_t& w_q0,
                             aiter_tensor_t& w_k0,
                             aiter_tensor_t& w_q1,
                             aiter_tensor_t& w_k1,
                             aiter_tensor_t& cos_sin0,
                             aiter_tensor_t& cos_sin1,
                             int64_t batch_size,
                             int64_t num_tokens0,
                             int64_t num_tokens1,
                             int64_t num_heads_q,
                             int64_t num_heads_k,
                             int64_t head_size,
                             bool is_interleaved,
                             double eps,
                             aiter_tensor_t& out_q01,
                             aiter_tensor_t& out_k01)
{
    AITER_CHECK(q0.is_contiguous() && k0.is_contiguous() && q1.is_contiguous() &&
                k1.is_contiguous());
    AITER_CHECK(w_q0.is_contiguous() && w_k0.is_contiguous() && w_q1.is_contiguous() &&
                w_k1.is_contiguous());
    AITER_CHECK(cos_sin0.is_contiguous() && cos_sin1.is_contiguous());
    HipDeviceGuard device_guard(q0.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    VLLM_DISPATCH_FLOATING_TYPES_rmTorch(q0.dtype(), "fused_qk_norm_rope_2way", [&] {
        using T = scalar_t;
        fused_rope_rms_2way<T>(reinterpret_cast<T*>(q0.data_ptr()),
                               reinterpret_cast<T*>(k0.data_ptr()),
                               reinterpret_cast<T*>(q1.data_ptr()),
                               reinterpret_cast<T*>(k1.data_ptr()),
                               reinterpret_cast<T*>(w_q0.data_ptr()),
                               reinterpret_cast<T*>(w_k0.data_ptr()),
                               reinterpret_cast<T*>(w_q1.data_ptr()),
                               reinterpret_cast<T*>(w_k1.data_ptr()),
                               reinterpret_cast<T*>(cos_sin0.data_ptr()),
                               reinterpret_cast<T*>(cos_sin1.data_ptr()),
                               batch_size,
                               num_tokens0,
                               num_tokens1,
                               num_heads_q,
                               num_heads_k,
                               head_size,
                               is_interleaved,
                               eps,
                               reinterpret_cast<T*>(out_q01.data_ptr()),
                               reinterpret_cast<T*>(out_k01.data_ptr()),
                               stream);
    });
}

void fused_qk_norm_rope_1way(aiter_tensor_t& q,
                             aiter_tensor_t& k,
                             aiter_tensor_t& w_q,
                             aiter_tensor_t& w_k,
                             aiter_tensor_t& cos_sin,
                             int64_t batch_size,
                             int64_t num_tokens,
                             int64_t num_heads_q,
                             int64_t num_heads_k,
                             int64_t head_size,
                             bool is_interleaved,
                             double eps,
                             aiter_tensor_t& out_q,
                             aiter_tensor_t& out_k)
{
    AITER_CHECK(q.is_contiguous() && k.is_contiguous());
    AITER_CHECK(w_q.is_contiguous() && w_k.is_contiguous());
    AITER_CHECK(cos_sin.is_contiguous());
    AITER_CHECK(out_q.is_contiguous() && out_k.is_contiguous());
    AITER_CHECK(cos_sin.dtype() == AITER_DTYPE_fp32,
                "fused_qk_norm_rope_1way requires cos_sin in float32 (got ",
                AiterDtype_to_str(cos_sin.dtype()), ")");
    HipDeviceGuard device_guard(q.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    VLLM_DISPATCH_FLOATING_TYPES_rmTorch(q.dtype(), "fused_qk_norm_rope_1way", [&] {
        using T = scalar_t;
        fused_rope_rms_1way<T>(reinterpret_cast<T*>(q.data_ptr()),
                               reinterpret_cast<T*>(k.data_ptr()),
                               reinterpret_cast<T*>(w_q.data_ptr()),
                               reinterpret_cast<T*>(w_k.data_ptr()),
                               reinterpret_cast<float*>(cos_sin.data_ptr()),
                               batch_size,
                               num_tokens,
                               num_heads_q,
                               num_heads_k,
                               head_size,
                               is_interleaved,
                               eps,
                               reinterpret_cast<T*>(out_q.data_ptr()),
                               reinterpret_cast<T*>(out_k.data_ptr()),
                               stream);
    });
}

void fused_qk_norm_rope_cache_block_quant_shuffle(
    aiter_tensor_t& qkv,           // Combined QKV tensor [num_tokens,
                                   // (num_heads_q+num_heads_k+num_heads_v)*head_dim]
    int64_t num_heads_q,           // Number of query heads
    int64_t num_heads_k,           // Number of key heads
    int64_t num_heads_v,           // Number of value heads
    int64_t head_dim,              // Dimension per head
    double eps,                    // Epsilon for RMS normalization
    aiter_tensor_t& q_weight,      // RMSNorm weights for query [head_dim]
    aiter_tensor_t& k_weight,      // RMSNorm weights for key [head_dim]
    aiter_tensor_t& cos_sin_cache, // Cos/sin cache [max_position, head_dim]
    bool is_neox,                  // Whether RoPE is applied in Neox style
    aiter_tensor_t& position_ids,  // Position IDs for RoPE [num_tokens]
    aiter_tensor_t& k_cache,       // k cache
    aiter_tensor_t& v_cache,       // v cache
    aiter_tensor_t& slot_mapping,  // slot mapping
    aiter_tensor_t&
        cu_q_len, // cu q len tensor [0, batch0_seq_len, batch0_seq_len + batch1_seq_len, ...]
    const std::string& kv_cache_dtype,     // kv cache data type
    std::optional<aiter_tensor_t> k_scale, // k scale tensor for quantized k cache
    std::optional<aiter_tensor_t> v_scale, // v scale tensor for quantized v cache
    int64_t max_tokens_per_batch // max tokens in any single batch (0 = use avg, safe for uniform
                                 // distributions)
)
{
    // Input validation
    CHECK_INPUT(qkv);
    CHECK_INPUT(cu_q_len);
    CHECK_INPUT(position_ids);
    CHECK_INPUT(q_weight);
    CHECK_INPUT(k_weight);
    CHECK_INPUT(cos_sin_cache);
    CHECK_TYPE(position_ids, AITER_DTYPE_i64);

    AITER_CHECK(qkv.dim() == 2,
                "QKV tensor must be 2D: [num_tokens, "
                "(num_heads_q+num_heads_k+num_heads_v)*head_dim]");
    AITER_CHECK(position_ids.dim() == 1, "Position IDs must be 1D: [num_tokens]");
    AITER_CHECK(q_weight.dim() == 1, "Query weights must be 1D: [head_dim]");
    AITER_CHECK(k_weight.dim() == 1, "Key weights must be 1D: [head_dim]");
    AITER_CHECK(cos_sin_cache.dim() == 2, "Cos/sin cache must be 2D: [max_position, head_dim]");
    AITER_CHECK(q_weight.size(0) == head_dim, "Query weights size must match head dimension");
    AITER_CHECK(k_weight.size(0) == head_dim, "Key weights size must match head dimension");
    AITER_CHECK(cos_sin_cache.size(1) == head_dim, "Cos/sin cache dimension must match head_dim");
    AITER_CHECK(qkv.dtype() == q_weight.dtype() && qkv.dtype() == k_weight.dtype(),
                "qkv, q_weight and k_weight must have the same dtype");
    AITER_CHECK(head_dim % 32 == 0,
                "Head dimension must be multiple of 32 for fused QK Norm RoPE kernel");
    AITER_CHECK(
        num_heads_k <= 32,
        "Number of key heads must be less than or equal to 32 for fused QK Norm RoPE kernel");

    // cu_q_len format: [0, batch0_seq_len, batch0_seq_len + batch1_seq_len, ...]
    // batch_size = cu_q_len.size(0) - 1
    AITER_CHECK(cu_q_len.dim() == 1, "Cu Q len tensor must be 1D");
    int64_t batch_size = cu_q_len.size(0) - 1;
    AITER_CHECK(batch_size > 0, "Batch size must be greater than 0");

    int64_t num_tokens = qkv.size(0);
    int64_t page_size  = k_cache.size(-2);
    int64_t x          = k_cache.size(-1);
    AITER_CHECK(
        x > 0 && (x & (x - 1)) == 0, "KV cache tiling size (x) must be a power of two, got ", x);
    // vec_size is 8 for bf16/fp16, 4 for fp32; vec_per_x = x/vec_size requires x >= vec_size
    AITER_CHECK(x >= 4, "KV cache tiling size (x) must be >= 4 for vectorized access, got ", x);
    AITER_CHECK(position_ids.size(0) == num_tokens,
                "Number of tokens in position_ids must match QKV");

    int64_t total_heads = num_heads_q + num_heads_k + num_heads_v;
    AITER_CHECK(qkv.size(1) == total_heads * head_dim,
                "QKV tensor size must match total number of heads and head dimension");

    HipDeviceGuard device_guard_blk(qkv.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    DISPATCH_BY_KV_CACHE_DTYPE_OPUS_rmTorch(
        qkv.dtype(), kv_cache_dtype, CALL_QK_NORM_ROPE_CACHE_BLOCK_QUANT);
}

void fused_qk_norm_rope_2way_fp8_perhead_quant(aiter_tensor_t& q0,
                                               aiter_tensor_t& k0,
                                               aiter_tensor_t& q1,
                                               aiter_tensor_t& k1,
                                               aiter_tensor_t& w_q0,
                                               aiter_tensor_t& w_k0,
                                               aiter_tensor_t& w_q1,
                                               aiter_tensor_t& w_k1,
                                               aiter_tensor_t& cos_sin0,
                                               aiter_tensor_t& cos_sin1,
                                               int64_t batch_size,
                                               int64_t num_tokens0,
                                               int64_t num_tokens1,
                                               int64_t num_heads_q,
                                               int64_t num_heads_k,
                                               int64_t head_size,
                                               bool is_interleaved,
                                               double eps,
                                               aiter_tensor_t& q_fp8,
                                               aiter_tensor_t& k_fp8,
                                               aiter_tensor_t& q_descale,
                                               aiter_tensor_t& k_descale,
                                               aiter_tensor_t& q_unquantized,
                                               aiter_tensor_t& k_unquantized)
{
    AITER_CHECK(q0.is_contiguous() && k0.is_contiguous() && q1.is_contiguous() &&
                k1.is_contiguous());
    AITER_CHECK(w_q0.is_contiguous() && w_k0.is_contiguous() && w_q1.is_contiguous() &&
                w_k1.is_contiguous());
    AITER_CHECK(cos_sin0.is_contiguous() && cos_sin1.is_contiguous());
    AITER_CHECK(q_fp8.is_contiguous() && k_fp8.is_contiguous());
    AITER_CHECK(q_descale.is_contiguous() && k_descale.is_contiguous());
    AITER_CHECK(q_unquantized.is_contiguous() && k_unquantized.is_contiguous());
    AITER_CHECK(q0.dtype() == k0.dtype() && q0.dtype() == q1.dtype() && q0.dtype() == k1.dtype());
    AITER_CHECK(q0.dtype() == w_q0.dtype() && q0.dtype() == w_k0.dtype() &&
                q0.dtype() == w_q1.dtype() && q0.dtype() == w_k1.dtype());
    AITER_CHECK(q0.dtype() == q_unquantized.dtype() && k0.dtype() == k_unquantized.dtype());
    AITER_CHECK(q_fp8.dtype() == AITER_DTYPE_fp8 && k_fp8.dtype() == AITER_DTYPE_fp8);
    AITER_CHECK(q_descale.dtype() == AITER_DTYPE_fp32 && k_descale.dtype() == AITER_DTYPE_fp32);

    HipDeviceGuard device_guard(q0.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    int total_warps         = (num_tokens0 + num_tokens1) * (num_heads_q + num_heads_k);
    constexpr int block_size = 256;
    constexpr int warp_size  = 32;
    int num_warps_per_block  = block_size / warp_size;

    AiterTensor q_partial_amax =
        AiterTensor::empty({batch_size, total_warps}, AITER_DTYPE_fp32, q0.device_id, stream);
    AiterTensor k_partial_amax =
        AiterTensor::empty({batch_size, total_warps}, AITER_DTYPE_fp32, q0.device_id, stream);

    dim3 threadsPerBlock(block_size);
    dim3 numBlocks((total_warps + num_warps_per_block - 1) / num_warps_per_block, batch_size);

    AITER_DISPATCH_FLOATING16_TYPES_rmTorch(
        q0.dtype(), "fused_qk_norm_rope_2way_fp8_perhead_amax", [&] {
            using T = scalar_t;
            auto launch_amax = [&]<int HS, bool NEOX>() {
                fused_rope_rms_2way_amax_kernel<T, HS, NEOX>
                    <<<numBlocks, threadsPerBlock, 0, stream>>>(
                        reinterpret_cast<T*>(q0.data_ptr()),
                        reinterpret_cast<T*>(k0.data_ptr()),
                        reinterpret_cast<T*>(q1.data_ptr()),
                        reinterpret_cast<T*>(k1.data_ptr()),
                        reinterpret_cast<T*>(w_q0.data_ptr()),
                        reinterpret_cast<T*>(w_k0.data_ptr()),
                        reinterpret_cast<T*>(w_q1.data_ptr()),
                        reinterpret_cast<T*>(w_k1.data_ptr()),
                        reinterpret_cast<T*>(cos_sin0.data_ptr()),
                        reinterpret_cast<T*>(cos_sin1.data_ptr()),
                        (int)num_tokens0,
                        (int)num_tokens1,
                        (int)num_heads_q,
                        (int)num_heads_k,
                        (float)eps,
                        total_warps,
                        reinterpret_cast<T*>(q_unquantized.data_ptr()),
                        reinterpret_cast<T*>(k_unquantized.data_ptr()),
                        reinterpret_cast<float*>(q_partial_amax.data_ptr()),
                        reinterpret_cast<float*>(k_partial_amax.data_ptr()));
            };
            switch(head_size)
            {
            case 64:
                if(!is_interleaved) launch_amax.template operator()<64, true>();
                else                launch_amax.template operator()<64, false>();
                break;
            case 128:
                if(!is_interleaved) launch_amax.template operator()<128, true>();
                else                launch_amax.template operator()<128, false>();
                break;
            case 256:
                if(!is_interleaved) launch_amax.template operator()<256, true>();
                else                launch_amax.template operator()<256, false>();
                break;
            default:
                AITER_CHECK(false, "Unsupported head_size: ", head_size);
            }
        });

    {
        dim3 reduce_grid((unsigned)(num_heads_q + num_heads_k), (unsigned)batch_size);
        dim3 reduce_block(256);
        qk_partial_amax_to_perhead_scale_kernel<<<reduce_grid, reduce_block, 0, stream>>>(
            reinterpret_cast<float*>(q_partial_amax.data_ptr()),
            reinterpret_cast<float*>(k_partial_amax.data_ptr()),
            (int)num_tokens0,
            (int)num_tokens1,
            (int)num_heads_q,
            (int)num_heads_k,
            total_warps,
            reinterpret_cast<float*>(q_descale.data_ptr()),
            reinterpret_cast<float*>(k_descale.data_ptr()));
    }

    AITER_DISPATCH_FLOATING16_TYPES_rmTorch(
        q0.dtype(), "fused_qk_norm_rope_2way_fp8_perhead_quant", [&] {
            using T          = scalar_t;
            int total_tokens = num_tokens0 + num_tokens1;
            int64_t q_numel  = (int64_t)batch_size * total_tokens * num_heads_q * head_size;
            int64_t k_numel  = (int64_t)batch_size * total_tokens * num_heads_k * head_size;
            dim3 quant_block(256);
            dim3 q_grid((unsigned)((q_numel + quant_block.x - 1) / quant_block.x));
            dim3 k_grid((unsigned)((k_numel + quant_block.x - 1) / quant_block.x));
            static_fp8_quant_perhead_kernel<T><<<q_grid, quant_block, 0, stream>>>(
                reinterpret_cast<mrope_utils::fp8e4m3fnuz*>(q_fp8.data_ptr()),
                reinterpret_cast<T*>(q_unquantized.data_ptr()),
                reinterpret_cast<float*>(q_descale.data_ptr()),
                (int)batch_size,
                total_tokens,
                (int)num_heads_q,
                (int)head_size);
            static_fp8_quant_perhead_kernel<T><<<k_grid, quant_block, 0, stream>>>(
                reinterpret_cast<mrope_utils::fp8e4m3fnuz*>(k_fp8.data_ptr()),
                reinterpret_cast<T*>(k_unquantized.data_ptr()),
                reinterpret_cast<float*>(k_descale.data_ptr()),
                (int)batch_size,
                total_tokens,
                (int)num_heads_k,
                (int)head_size);
        });
}

// ---------- per-(batch, head) FP8 V quant (2-way, no bf16 cat) ----------

__device__ __forceinline__ void atomic_fmax_pos(float* addr, float val)
{
    int* iaddr = reinterpret_cast<int*>(addr);
    int ival   = __float_as_int(val);
    atomicMax(iaddr, ival);
}

__global__ void v_amax_to_descale_kernel(const float* __restrict__ v_amax,
                                         int num_heads,
                                         float* __restrict__ v_descale)
{
    int b   = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= num_heads) return;
    constexpr float fp8_max = 240.0f;
    v_descale[b * num_heads + idx] =
        fmaxf(v_amax[b * num_heads + idx], 1e-8f) / fp8_max;
}

template <typename T, int TILE_T, int HEAD_SIZE>
__global__ void __launch_bounds__(256) v_2way_per_head_amax_tiled_kernel(
    const T* __restrict__ v0_,
    const T* __restrict__ v1_,
    int num_tokens0,
    int num_tokens1,
    int num_heads,
    float* __restrict__ v_amax)
{
    constexpr int BT = 256;
    int b            = blockIdx.z;
    int h            = blockIdx.y;
    int tile         = blockIdx.x;
    int total_tokens = num_tokens0 + num_tokens1;
    int t_start      = tile * TILE_T;
    int t_end        = min(t_start + TILE_T, total_tokens);
    int slab_h_stride = num_heads * HEAD_SIZE;
    float local       = 0.0f;

    for(int idx = threadIdx.x; idx < (t_end - t_start) * HEAD_SIZE; idx += BT)
    {
        int local_t = idx / HEAD_SIZE;
        int d       = idx % HEAD_SIZE;
        int t       = t_start + local_t;
        float val;
        if(t < num_tokens0)
        {
            int64_t off = ((int64_t)b * num_tokens0 + t) * slab_h_stride +
                          (int64_t)h * HEAD_SIZE + d;
            val = (float)v0_[off];
        }
        else
        {
            int t1      = t - num_tokens0;
            int64_t off = ((int64_t)b * num_tokens1 + t1) * slab_h_stride +
                          (int64_t)h * HEAD_SIZE + d;
            val = (float)v1_[off];
        }
        local = fmaxf(local, fabsf(val));
    }

    __shared__ float sm[BT];
    sm[threadIdx.x] = local;
    __syncthreads();
#pragma unroll
    for(int s = BT / 2; s > 0; s >>= 1)
    {
        if(threadIdx.x < s) sm[threadIdx.x] = fmaxf(sm[threadIdx.x], sm[threadIdx.x + s]);
        __syncthreads();
    }
    if(threadIdx.x == 0) atomic_fmax_pos(v_amax + b * num_heads + h, sm[0]);
}

template <typename T, int TILE_T, int HEAD_SIZE>
__global__ void __launch_bounds__(256) v_2way_per_head_quant_tiled_kernel(
    const T* __restrict__ v0_,
    const T* __restrict__ v1_,
    int num_tokens0,
    int num_tokens1,
    int num_heads,
    mrope_utils::fp8e4m3fnuz* __restrict__ v_fp8_,
    const float* __restrict__ v_descale)
{
    constexpr int BT = 256;
    int b            = blockIdx.z;
    int h            = blockIdx.y;
    int tile         = blockIdx.x;
    int total_tokens = num_tokens0 + num_tokens1;
    int t_start      = tile * TILE_T;
    int t_end        = min(t_start + TILE_T, total_tokens);
    int slab_h_stride = num_heads * HEAD_SIZE;
    float inv         = 1.0f / v_descale[b * num_heads + h];

    for(int idx = threadIdx.x; idx < (t_end - t_start) * HEAD_SIZE; idx += BT)
    {
        int local_t = idx / HEAD_SIZE;
        int d       = idx % HEAD_SIZE;
        int t       = t_start + local_t;
        int64_t out_off = ((int64_t)b * total_tokens + t) * slab_h_stride +
                          (int64_t)h * HEAD_SIZE + d;
        float val;
        if(t < num_tokens0)
        {
            int64_t in_off = ((int64_t)b * num_tokens0 + t) * slab_h_stride +
                             (int64_t)h * HEAD_SIZE + d;
            val = (float)v0_[in_off];
        }
        else
        {
            int t1         = t - num_tokens0;
            int64_t in_off = ((int64_t)b * num_tokens1 + t1) * slab_h_stride +
                             (int64_t)h * HEAD_SIZE + d;
            val = (float)v1_[in_off];
        }
        v_fp8_[out_off] = mrope_utils::fp8e4m3fnuz(val * inv);
    }
}

void v_2way_per_head_fp8_quant(aiter_tensor_t& v0,
                               aiter_tensor_t& v1,
                               aiter_tensor_t& v_fp8,
                               aiter_tensor_t& v_descale)
{
    AITER_CHECK(v0.is_contiguous() && v1.is_contiguous());
    AITER_CHECK(v_fp8.is_contiguous() && v_descale.is_contiguous());
    AITER_CHECK(v0.ndim == 4 && v1.ndim == 4, "v0/v1 must be 4D [B, T, H, D]");
    int64_t batch_size  = v0.size(0);
    int64_t num_tokens0 = v0.size(1);
    int64_t num_tokens1 = v1.size(1);
    int64_t num_heads   = v0.size(2);
    int64_t head_size   = v0.size(3);
    AITER_CHECK(v1.size(0) == batch_size && v1.size(2) == num_heads &&
                    v1.size(3) == head_size,
                "v0/v1 must share B/H/D");
    AITER_CHECK(head_size == 128,
                "v_2way_per_head_fp8_quant currently only supports head_size=128");
    AITER_CHECK(v0.dtype() == v1.dtype(), "v0/v1 dtype must match");
    AITER_CHECK(v_fp8.dtype() == AITER_DTYPE_fp8, "v_fp8 must be fp8");
    AITER_CHECK(v_descale.dtype() == AITER_DTYPE_fp32, "v_descale must be fp32");

    HipDeviceGuard device_guard(v0.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    AiterTensor v_amax =
        AiterTensor::empty({batch_size, num_heads}, AITER_DTYPE_fp32, v0.device_id, stream);

    constexpr int TILE_T    = 128;
    constexpr int HEAD_SIZE = 128;
    int num_tiles           = (int)((num_tokens0 + num_tokens1 + TILE_T - 1) / TILE_T);
    dim3 grid((unsigned)num_tiles, (unsigned)num_heads, (unsigned)batch_size);
    dim3 block(256);

    AITER_DISPATCH_FLOATING16_TYPES_rmTorch(
        v0.dtype(), "v_2way_per_head_amax_tiled", [&] {
            v_2way_per_head_amax_tiled_kernel<scalar_t, TILE_T, HEAD_SIZE>
                <<<grid, block, 0, stream>>>(
                    reinterpret_cast<scalar_t*>(v0.data_ptr()),
                    reinterpret_cast<scalar_t*>(v1.data_ptr()),
                    (int)num_tokens0,
                    (int)num_tokens1,
                    (int)num_heads,
                    reinterpret_cast<float*>(v_amax.data_ptr()));
        });

    {
        dim3 fg((unsigned)((num_heads + 31) / 32), (unsigned)batch_size);
        dim3 fb(32);
        v_amax_to_descale_kernel<<<fg, fb, 0, stream>>>(
            reinterpret_cast<float*>(v_amax.data_ptr()),
            (int)num_heads,
            reinterpret_cast<float*>(v_descale.data_ptr()));
    }

    AITER_DISPATCH_FLOATING16_TYPES_rmTorch(
        v0.dtype(), "v_2way_per_head_quant_tiled", [&] {
            v_2way_per_head_quant_tiled_kernel<scalar_t, TILE_T, HEAD_SIZE>
                <<<grid, block, 0, stream>>>(
                    reinterpret_cast<scalar_t*>(v0.data_ptr()),
                    reinterpret_cast<scalar_t*>(v1.data_ptr()),
                    (int)num_tokens0,
                    (int)num_tokens1,
                    (int)num_heads,
                    reinterpret_cast<mrope_utils::fp8e4m3fnuz*>(v_fp8.data_ptr()),
                    reinterpret_cast<float*>(v_descale.data_ptr()));
        });
}

} // namespace aiter

// ============================================================================
// fused_qk_norm_rope_group_quant kernel (MLA group-quant path)
// Moved from cache_kernels.cu for better file organization.
// ============================================================================

namespace aiter {

    struct MlaKernelParams {
        // kv_cache is dense / token-contiguous: [num_tokens, num_kv_heads, entry].
        // token_stride = per-token stride (= kv_cache.stride(0)); kv_cache_stride_h =
        // per-kv-head stride (= kv_cache.stride(1)).
        int token_stride, kv_cache_stride_h;
        int q_stride_0, q_stride_1;
        int q_out_stride_0, q_out_stride_1;
        int q_rope_out_stride_0, q_rope_out_stride_1;
        int kv_stride_0, kv_stride_1;
        int k_pe_out_stride_0, k_pe_out_stride_1;
        // Q scale strides (used only when Q is fp8-quantised).
        // q_scale shape [num_tokens, num_heads, num_q_groups]; for typical contiguous layout:
        //   q_scale_stride_0 = num_heads * num_q_groups
        //   q_scale_stride_1 = num_q_groups
        int q_scale_stride_0, q_scale_stride_1;
        int num_tokens;
        int num_heads;  // V4 MQA: num_kv_heads is hardcoded to 1 (blockIdx.y==0 is the K wave)
        // --- K-only paged-cache write (fused_kv_norm_rope_group_quant) ---
        // When the K-only kernel writes into a paged cache via slot_mapping, the
        // dest row is block*block_stride + off*row_stride, with block = slot/page_size,
        // off = slot%page_size. Both the nope+scale cache and the rope cache are paged
        // [num_blocks, page_size, entry] (MQA, no NK dim); row strides are their .stride(1).
        // Unused by the QK kernel (token-contiguous).
        int page_size;
        int kcache_block_stride, kcache_row_stride;  // nope+scale paged cache
        int krope_block_stride, krope_row_stride;     // rope paged cache
    };

    // ============================================================================
    // K wave body (shared between fuse_qk_norm_rope_group_quant_cache_kernel_impl
    // and the K-only fuse_kv_norm_rope_group_quant_cache_kernel). RMSNorm over
    // the full head_dim, 1xG e8m0 group-quant on NoPE (writing nope fp8 +
    // duplicated inline scale into kv_cache, V4 nm asm reader layout), GPT-J/NeoX
    // RoPE on the PE tail (bf16, written to a separate k_pe_out buffer).
    //
    // Caller responsibilities:
    //   - One wave per token (WARP_SIZE lanes; vec_size = HEAD_DIM/WARP_SIZE). Caller must
    //     launch tokens_per_block * WARP_SIZE threads (wave_id = threadIdx.x/WARP_SIZE).
    //   - cos_ptr/sin_ptr are pre-offset by positions[token_idx]*(pe_dim/2) so the
    //     helper does NOT touch the positions tensor. Both QK and K-only callers
    //     compute the offset once; the helper just uses the resulting pointers.
    //   - kv shape: [num_tokens, (NK=1,) head_dim]; kv_cache shape matches; k_pe_out
    //     shape: [num_tokens, (NK=1,) pe_dim]. NK is hardcoded to 1 (MQA).
    //   - params.token_stride / kv_stride_0 / k_pe_out_stride_0 are used; other
    //     fields are ignored on this path.
    // ============================================================================
    template <typename scalar_t, typename cache_t,
              vllm::Fp8KVCacheDataType kv_dt, bool is_neox,
              int HEAD_DIM = 512, int PE_DIM = 64, int GROUP_SIZE = 64>
    __device__ inline void k_wave_norm_rope_group_quant_impl(
        const scalar_t* __restrict__ kv,         // [num_tokens, (NK=1,) head_dim]
        cache_t*        __restrict__ kv_cache,   // [num_tokens, (NK=1,) head_dim]
        scalar_t*       __restrict__ k_pe_out,   // [num_tokens, (NK=1,) pe_dim] bf16
        const scalar_t* __restrict__ k_weight,   // [head_dim]
        const scalar_t* __restrict__ cos_ptr,    // pre-offset by positions[token_idx]
        const scalar_t* __restrict__ sin_ptr,    // pre-offset by positions[token_idx]
        float eps,
        const MlaKernelParams& __restrict__ params,
        int32_t token_idx, int32_t tid,
        // Precomputed destination byte/elem offsets for the two output buffers.
        // QK passes token_idx*stride (token-contiguous); the K-only kernel passes
        // the paged slot offset (block*block_stride + off*row_stride) so the same
        // body writes either layout without branching on paging here.
        int64_t out_cache_offset, int64_t out_rope_offset)
    {
      // ---- Compile-time constants (collapsed by the compiler at every callsite) ----
      constexpr int32_t head_size = HEAD_DIM;
      constexpr int32_t pe_dim    = PE_DIM;
      constexpr int32_t nope_dim  = head_size - pe_dim;
      // Elements/lane to cover one head in a single wave = HEAD_DIM/WARP_SIZE
      // (wave64: 8; wave32: 16). WARP_SIZE is the device-resolved wavefront size; safe in
      // this __device__ constexpr (using it in __launch_bounds__/template args would trip
      // the host-pass ICE trap). fp32 is dead code; cap at 4 (no 32B opus float vector).
      constexpr int32_t kWave = WARP_SIZE;
      constexpr int32_t vec_size_i =
          std::is_same_v<scalar_t, float> ? 4 : ((HEAD_DIM / 8 <= kWave) ? 8 : 16);
      constexpr int32_t vec_size_o = vec_size_i;
      constexpr uint32_t nope_vec  = nope_dim / vec_size_o;
      constexpr uint32_t vec_stride = kWave;
      constexpr int32_t nope_offset = 0;            // V4 layout: nope-first
      constexpr int32_t pe_tid_start = nope_vec;
      constexpr int32_t pe_tid_end   = pe_tid_start + (pe_dim / vec_size_i);
      constexpr int32_t ooba_i = 4 / sizeof(scalar_t);
      constexpr int32_t ooba_o = 4 / sizeof(cache_t);
      constexpr int32_t oob_i  = (head_size + ooba_i - 1) / ooba_i * ooba_i;
      constexpr int32_t oob_o  = (head_size + ooba_o - 1) / ooba_o * ooba_o;
      constexpr int32_t reduce_thread_size = GROUP_SIZE / vec_size_i;
      // Streaming (read-once) NT/SLC|GLC loads: bypass L2 for the bf16/fp16 KV input.
      // Disabled for fp32 (worse cache-line utilization). See aiter_opus_plus.h::GROUP_NT.
      constexpr int32_t IN_LOAD_AUX = (sizeof(scalar_t) < 4) ? GROUP_NT : 0;

      // ---- Compile-time invariants on the dispatch-table template params (zero runtime
      // cost; user-shape validation lives in the host entry). Each lane owns vec_size_i
      // contiguous elems, one wave covers the head -> head_size/vec_size_i <= WARP_SIZE.
      // No upper-bound assert: fp32 (head<=256) is dead-code-instantiated by the macro.
      static_assert(head_size % vec_size_i == 0 && pe_dim % vec_size_i == 0
                    && GROUP_SIZE % vec_size_i == 0,
                    "head_dim / pe_dim / GROUP_SIZE must each be divisible by vec_size_i");
      static_assert(pe_dim > 0 && pe_dim < head_size && nope_dim % GROUP_SIZE == 0,
                    "need 0 < pe_dim < head_dim and (head_dim - pe_dim) % GROUP_SIZE == 0");
      static_assert(reduce_thread_size >= 1 && reduce_thread_size <= 64
                    && (reduce_thread_size & (reduce_thread_size - 1)) == 0,
                    "GROUP_SIZE/vec_size_i (within-group DPP reduce width) must be a pow2 in [1,64]");
      static_assert(pe_dim / vec_size_i >= 2,
                    "pe_dim/vec_size_i must be >= 2 for the NeoX __shfl_xor pairing");

      using opus_vec_i = opus::vector_t<scalar_t, vec_size_i>;
      using opus_vec_o = opus::vector_t<cache_t, vec_size_o>;
      // Raw opus buffer.load<N>/store<N> emit a single instruction (<=16B b128), so the
      // wave32 16-wide vec (32B) has no overload. load_vector_nbytes/store_vector split
      // into b128 chunks; pick the largest chunk dividing the per-lane byte count.
      constexpr int32_t in_chunk_bytes = (vec_size_i * sizeof(scalar_t)) % 16 == 0 ? 16 : 8;

      // ---- Pointers / buffer descriptors (single KV head: per-head strides vanish) ----
      // int64 base offsets: token_idx * stride can exceed INT32_MAX for large prefill
      // chunks (e.g. T=65536 / H=128 -> ~4.3e9 elements); strides stay 32-bit.
      const int64_t kv_cache_offset = out_cache_offset;  // dest row (token-contig or paged slot)
      const int64_t token_kv_base   = static_cast<int64_t>(token_idx) * params.kv_stride_0;

      const scalar_t* kv_ptr = kv + token_kv_base;
      auto* ptr_o = kv_cache + kv_cache_offset + nope_offset;
      auto buffer_kv = opus::make_gmem<scalar_t>(kv_ptr, oob_i * sizeof(scalar_t));
      auto buffer_o  = opus::make_gmem<cache_t>(ptr_o, oob_o * sizeof(cache_t));

      // Unified vec load: all 64 lanes load vec_size_i contiguous elements.
      // For head_dim < 64*vec_size_i (e.g. 128, 192, 256, 384) the trailing
      // lanes have tid*vec_size_i >= head_dim and would OOB on a raw pointer
      // load. We use opus buffer descriptors (which return 0 for OOB) so the
      // idle lanes' loaded values become zero -- benign because those lanes
      // are filtered out by `is_nope_thread` / pe_tid range later, and the
      // sum_sq wave-reduce sums in zeros from idle lanes (head_size still
      // matches the active lane count via head_size / vec_size_i, so the
      // mean(x^2) divisor is correct).
      const bool is_nope_thread = (tid < nope_vec);  // V4 nope-first

      opus_vec_i vec_kv =
          load_vector_nbytes<scalar_t, vec_size_i, in_chunk_bytes, IN_LOAD_AUX>(
              buffer_kv, tid * vec_size_i);
      // Bounds-safe k_weight load: use buffer descriptor (OOB returns 0) so head_dim
      // < 64*vec_size_i works (e.g. head_dim=128/192/256/384) without raw-pointer OOB
      // faults on the trailing idle lanes. NT/SLC|GLC NOT applied here -- k_weight
      // has heavy temporal reuse across tokens so we want it to live in L1.
      auto buffer_kw = opus::make_gmem<scalar_t>(k_weight, head_size * sizeof(scalar_t));
      opus_vec_i vec_k_weight =
          load_vector_nbytes<scalar_t, vec_size_i, in_chunk_bytes>(buffer_kw, tid * vec_size_i);

      // ---- Prefetch cos/sin for the PE lanes BEFORE the RMSNorm reduce ----
      // The cos/sin gather (indexed by positions[token]) is a scattered global
      // read issued by only the PE lanes. If loaded late (inside the rope block,
      // after the row-sum reduce) its latency lands on the post-reduce critical
      // path -- and since the wave can't retire until the PE lanes finish, that
      // latency directly inflates wave time. Issuing the load HERE lets it fly
      // during the reduce + norm + nope-quant so it has arrived by the time the
      // rope math needs it. Each PE lane reads a contiguous run in ONE vector op
      // (runs are aligned: cos_ptr is 64B-aligned via positions*(pe_dim/2), the
      // lane offset is a multiple of the vector width) and converts to float;
      // the rope math in Step 4 consumes these registers.
      const bool is_pe_lane = (tid >= pe_tid_start && tid < pe_tid_end);
      float pe_cos[vec_size_i];
      float pe_sin[vec_size_i];
      if (is_pe_lane) {
        const int32_t pe_local_tid = tid - pe_tid_start;
        if constexpr (is_neox) {
          // neox: each lane consumes cos/sin[base .. base+vec_size_i).
          constexpr int32_t half_pe_threads = pe_dim / vec_size_i / 2;
          const bool is_x_half = (pe_local_tid < half_pe_threads);
          const int32_t cos_base =
              (is_x_half ? pe_local_tid : (pe_local_tid - half_pe_threads)) * vec_size_i;
          const opus_vec_i cos_v = *reinterpret_cast<const opus_vec_i*>(cos_ptr + cos_base);
          const opus_vec_i sin_v = *reinterpret_cast<const opus_vec_i*>(sin_ptr + cos_base);
          #pragma unroll
          for (int i = 0; i < vec_size_i; i++) {
            pe_cos[i] = static_cast<float>(cos_v[i]);
            pe_sin[i] = static_cast<float>(sin_v[i]);
          }
        } else {
          // GPT-J: lane reads cos/sin[pe_local_tid*(vec_size_i/2) .. +vec_size_i/2),
          // i.e. one freq per adjacent (2k, 2k+1) pair.
          constexpr int32_t half_vec = vec_size_i / 2;
          using opus_vec_half = opus::vector_t<scalar_t, half_vec>;
          const int32_t cos_base = pe_local_tid * half_vec;
          const opus_vec_half cos_v = *reinterpret_cast<const opus_vec_half*>(cos_ptr + cos_base);
          const opus_vec_half sin_v = *reinterpret_cast<const opus_vec_half*>(sin_ptr + cos_base);
          #pragma unroll
          for (int j = 0; j < half_vec; j++) {
            pe_cos[j] = static_cast<float>(cos_v[j]);
            pe_sin[j] = static_cast<float>(sin_v[j]);
          }
        }
      }

      // ---- Step 1: row sum(x^2) -> rstd ----
      float sum_sq = 0.0f;
      #pragma unroll
      for (int i = 0; i < vec_size_i; ++i) {
        float val = static_cast<float>(vec_kv[i]);
        sum_sq += val * val;
      }
      auto sum_func = [](float a, float b) { return a + b; };
      float total_sum_sq =
          wave_reduce<float, decltype(sum_func), vec_stride, true>(sum_sq, sum_func);
      const float rms_scale = rsqrtf(total_sum_sq / static_cast<float>(head_size) + eps);

      // ---- Step 2: norm * weight (post-norm float buffer) ----
      float k_normed[vec_size_i];
      #pragma unroll
      for (int i = 0; i < vec_size_i; i++) {
        k_normed[i] = static_cast<float>(vec_kv[i]) * rms_scale
                    * static_cast<float>(vec_k_weight[i]);
      }

      // ---- Step 3 (nope threads only): group-amax -> e8m0 -> fp8 + cache write ----
      float thread_max = 0.0f;
      if (is_nope_thread) {
        if constexpr (kv_dt != vllm::Fp8KVCacheDataType::kAuto) {
          #pragma unroll
          for (int i = 0; i < vec_size_i; i++) {
            thread_max = fmaxf(thread_max, fabsf(k_normed[i]));
          }
        }
      }

      if constexpr (kv_dt != vllm::Fp8KVCacheDataType::kAuto) {
        thread_max = multithread_reduce_max_dpp<reduce_thread_size>(thread_max);

        float inv_scale;
        if constexpr (std::is_same_v<cache_t, opus::fp8_t>) {
          // E8M0 block scale via the shared MX helper, RoundUp mode (matches
          // V4-Pro / NV ROUND_UP / torchao RCEIL: dq_scale = ceil_pow2(amax / fp8_max)).
          constexpr MxDtype kMxDt = kHwFp8E4m3Dtype;
          const E8m0BlockScale s =
              fp_f32_to_e8m0_block_scale<MxScaleRoundMode::RoundUp, kMxDt>(thread_max);
          // v4 nm asm reader reads each tile scale TWICE consecutively
          // (s0,s0,s1,s1,...): write 2*num_nope_groups bytes at
          // [nope_dim : nope_dim+2*num_nope_groups). Trailing pad is zero-init by caller.
          if (tid % reduce_thread_size == 0 && is_nope_thread) {
            // group_id = (tid * vec_size_i) / GROUP_SIZE = tid / reduce_thread_size.
            // Compiler folds to a shift since GROUP_SIZE / vec_size_i is power-of-2.
            const int group_id = tid / reduce_thread_size;  // 0..num_nope_groups-1
            auto* tmp = reinterpret_cast<uint8_t*>(ptr_o + nope_dim);
            const uint16_t scale_pair =
                static_cast<uint16_t>(s.byte) | (static_cast<uint16_t>(s.byte) << 8);
            *reinterpret_cast<uint16_t*>(tmp + group_id * 2) = scale_pair;
          }
          inv_scale = is_nope_thread ? (1.0f / s.dq_scale) : 0.0f;
        } else {
          const float group_scale = thread_max / opus::finfo<cache_t>::max();
          inv_scale = is_nope_thread ? (1.0f / group_scale) : 0.0f;
        }

        if (is_nope_thread) {
          const uint32_t nope_out_offset = tid * vec_size_i;  // nope-first
          opus_vec_o vec_out;
          #pragma unroll
          for (int i = 0; i < vec_size_i; i++) {
            vec_out[i] = opus::cast<cache_t>(k_normed[i] * inv_scale);
          }
          store_vector<cache_t, cache_t, vec_size_o>(buffer_o, vec_out, nope_out_offset);
        }
      } else {
        // bf16/fp16/fp32 cache: just store the normed value (no quant).
        if (is_nope_thread) {
          const uint32_t nope_out_offset = tid * vec_size_i;  // nope-first
          opus_vec_o vec_out_k;
          #pragma unroll
          for (int i = 0; i < vec_size_i; i++) {
            vec_out_k[i] = static_cast<cache_t>(k_normed[i]);
          }
          store_vector<cache_t, cache_t, vec_size_o>(buffer_o, vec_out_k, nope_out_offset);
        }
      }

      // ---- Step 4 (pe threads only): RoPE on the normed bf16 -> separate k_pe_out ----
      scalar_t* k_out_rope = k_pe_out + out_rope_offset;  // dest row (token-contig or paged slot)
      if (is_pe_lane) {
        const int32_t pe_local_tid = tid - pe_tid_start;  // 0..(pe_dim/vec_size_i - 1)
        if constexpr (is_neox) {
          // neox: x-half = pe[0..pe_dim/2), y-half = pe[pe_dim/2..pe_dim).
          // __shfl_xor with mask=half_pe_threads pairs lanes across the halves.
          // cos/sin already prefetched into pe_cos/pe_sin above (latency hidden
          // behind the reduce) -- just consume the registers here.
          constexpr int32_t half_pe_threads = pe_dim / vec_size_i / 2;  // 4 for pe=64,vec=8
          const bool is_x_half = (pe_local_tid < half_pe_threads);
          #pragma unroll
          for (int i = 0; i < vec_size_i; i++) {
            float my_val   = k_normed[i];
            float pair_val = __shfl_xor(my_val, half_pe_threads, WARP_SIZE);
            float rot = is_x_half
                            ? (my_val * pe_cos[i] - pair_val * pe_sin[i])
                            : (my_val * pe_cos[i] + pair_val * pe_sin[i]);
            k_out_rope[pe_local_tid * vec_size_i + i] = static_cast<scalar_t>(rot);
          }
        } else {
          // GPT-J: pairs are adjacent (0,1), (2,3), ... within each thread's vec;
          // pe_cos[i>>1] is the prefetched freq for pair (i, i+1).
          #pragma unroll
          for (int i = 0; i < vec_size_i; i += 2) {
            float fkx = k_normed[i];
            float fky = k_normed[i + 1];
            float f32_cos = pe_cos[i >> 1];
            float f32_sin = pe_sin[i >> 1];
            k_out_rope[pe_local_tid * vec_size_i + i]     = static_cast<scalar_t>(fkx * f32_cos - fky * f32_sin);
            k_out_rope[pe_local_tid * vec_size_i + i + 1] = static_cast<scalar_t>(fky * f32_cos + fkx * f32_sin);
          }
        }
      }
    }

    template <typename scalar_t, typename cache_t, typename query_t, vllm::Fp8KVCacheDataType kv_dt, vllm::Fp8KVCacheDataType q_dt,
              bool is_neox,
              // --- NEW (flydsl-alignment) compile-time options ---
              int Q_GROUP_SIZE = 64, bool Q_SCALE_FP32 = false, bool HAS_Q_WEIGHT = false,
              int HEAD_DIM = 512, int TOKENS_PER_BLOCK = 1, int WAVE = 64>
    __device__ void fuse_qk_norm_rope_group_quant_cache_kernel_impl(
        const scalar_t* __restrict__ q,       // [num_tokens, num_heads, head_dim]
        const scalar_t* __restrict__ kv,      // [num_tokens, (k_num_heads,) head_dim]
        scalar_t* __restrict__ k_pe_out,      // [num_tokens, (k_num_heads,) pe_dim]
        const scalar_t* __restrict__ k_weight, // [head_dim]
        const scalar_t* __restrict__ q_weight, // [head_dim] (may be nullptr if !HAS_Q_WEIGHT)
        cache_t* __restrict__ kv_cache,
        query_t* __restrict__ q_out,          // bf16: [.,H,512]; fp8: q_nope_scale_buff [.,H,512]
        void* __restrict__ q_scale_raw,       // legacy separate Q scale (unused for fp8 inline path)
        scalar_t* __restrict__ q_rope_out,    // fp8-Q: rotated Q-PE bf16 [.,H,pe_dim] (separate)
        const int64_t* __restrict__ positions,
        const scalar_t *__restrict__ cos_cache,
        const scalar_t *__restrict__ sin_cache,
        float eps,
        const MlaKernelParams& __restrict__ params
    ) {
      // ---- All compile-time constants ----
      constexpr int32_t head_size = HEAD_DIM;
      constexpr int32_t pe_dim = 64;
      constexpr int32_t nope_dim = head_size - pe_dim;
      // Elements/lane to cover one head with a single wave = HEAD_DIM/WARP_SIZE
      // (wave64: 8 for head=512; wave32: 16). WARP_SIZE is the device-resolved wavefront
      // size; safe in this __device__ constexpr (matches the shared K-wave body). The trailing
      // WAVE template param is kept for API compatibility but is no longer consulted, so the
      // Q path is wave-generic without an extra per-wave-size instantiation.
      // fp32 is a dead-code instantiation; cap at 4 (no 32-byte opus float vector).
      constexpr int32_t kWave = WARP_SIZE;
      constexpr int32_t vec_size_i =
          std::is_same_v<scalar_t, float> ? 4 : ((HEAD_DIM / 8 <= kWave) ? 8 : 16);
      constexpr int32_t vec_size_o = vec_size_i;
      constexpr uint32_t nope_vec = nope_dim / vec_size_o;
      constexpr uint32_t vec_stride = 64;
      constexpr int32_t GROUP_SIZE = 64;
      // V4 layout is always nope-first: nope occupies [0:nope_dim), pe the tail.
      constexpr int32_t nope_offset = 0;
      constexpr int32_t pe_tid_start = nope_vec;
      constexpr int32_t pe_tid_end = pe_tid_start + (pe_dim / vec_size_i);
      constexpr int32_t ooba_i = 4 / sizeof(scalar_t);
      constexpr int32_t ooba_o = 4 / sizeof(cache_t);
      constexpr int32_t oob_i = (head_size + ooba_i - 1) / ooba_i * ooba_i;
      constexpr int32_t oob_o = (head_size + ooba_o - 1) / ooba_o * ooba_o;
      constexpr int32_t q_ooba_o = 4 / sizeof(query_t);
      constexpr int32_t q_oob_o = (head_size + q_ooba_o - 1) / q_ooba_o * q_ooba_o;
      constexpr int32_t reduce_thread_size = GROUP_SIZE / vec_size_i;
      // Non-temporal (streaming) load policy for the Q/KV inputs. Each input element is read
      // exactly once with no reuse, so marking the buffer loads SLC|GLC (GROUP_NT) bypasses L2
      // and avoids cache pollution, which helps the bandwidth-bound large-T regime (the Q read
      // alone is T*H*head_size*2 bytes, e.g. ~268 MB at T=4096/H=128). NT hurts fp32 inputs
      // (4 B/elem -> worse cache-line utilization), so gate on 2-byte dtypes only, matching the
      // activation-kernel convention (see GROUP_NT in aiter_opus_plus.h).
      constexpr int32_t IN_LOAD_AUX = (sizeof(scalar_t) < 4) ? GROUP_NT : 0;
      constexpr int32_t in_chunk_bytes = (vec_size_i * sizeof(scalar_t)) % 16 == 0 ? 16 : 8;
      // K nope_scale_buff layout (v4 nm asm reader, 512B entry for head_dim=512):
      //   [nope fp8 nope_dim B][e8m0 scale 2*(nope_dim/64) B, each tile-scale x2][pad -> 512].
      // The rotated K-PE goes to a SEPARATE rope_buff (k_pe_out), bf16.

      using opus_vec_i = opus::vector_t<scalar_t, vec_size_i>;
      using opus_vec_o = opus::vector_t<cache_t, vec_size_o>;
      using opus_vec_q = opus::vector_t<query_t, vec_size_o>;

      // ---- Wave-level indexing: each wave handles one token ----
      const uint32_t wave_id = threadIdx.x / WARP_SIZE;
      const uint32_t tid = threadIdx.x % WARP_SIZE;
      const int32_t token_idx = static_cast<int32_t>(blockIdx.x) * TOKENS_PER_BLOCK + wave_id;
      if (token_idx >= params.num_tokens) return;

      // ---- Grid layout (V4 MQA, num_kv_heads == 1): blockIdx.y == 0 → the single K wave,
      // blockIdx.y in [1, ..) → Q waves. ----
      const int32_t combined_head_idx = static_cast<int32_t>(blockIdx.y);
      const bool is_k_wave = (combined_head_idx == 0);

      // RoPE cos/sin pointers (shared between K and Q phases)
      const int32_t cos_sin_offset = static_cast<int32_t>(positions[token_idx]) * (pe_dim >> 1);
      const scalar_t *cos_ptr = cos_cache + cos_sin_offset;
      const scalar_t *sin_ptr = sin_cache + cos_sin_offset;

      // ============ K Processing: RMS Norm over full head_dim, group quant nope, RoPE pe ============
      if (is_k_wave) {
        // Delegate to the shared K-wave body (also used by the K-only kernel).
        // QK path is token-contiguous: dest row = token_idx * stride (no paging).
        const int64_t out_cache_offset =
            static_cast<int64_t>(token_idx) * params.token_stride;
        const int64_t out_rope_offset =
            static_cast<int64_t>(token_idx) * params.k_pe_out_stride_0;
        k_wave_norm_rope_group_quant_impl<scalar_t, cache_t, kv_dt, is_neox, HEAD_DIM>(
            kv, kv_cache, k_pe_out, k_weight, cos_ptr, sin_ptr, eps, params,
            token_idx, static_cast<int32_t>(tid), out_cache_offset, out_rope_offset);
      } else { // Q processing — multi-head loop
      // ============ Q Processing: RMS norm + optional q_weight + RoPE + optional fp8 group quant ============
      // Layout: every thread `tid` owns elements [tid*vec_size_i .. tid*vec_size_i+vec_size_i).
      // - nope threads: own non-rope elements; just write q_normed
      // - pe threads:   own rope elements; write q_normed AFTER RoPE rotation
      // When Q is fp8-quantised, the group-amax DPP reduce happens over Q_REDUCE threads at the END
      // (after RoPE), so we accumulate post-rope absolute max for every thread (whether nope or pe).
      const int32_t q_wave_idx = combined_head_idx - 1;  // blockIdx.y==0 is the K wave
      const int32_t num_q_waves = static_cast<int32_t>(gridDim.y) - 1;
      const int32_t q_heads_per_wave = (params.num_heads + num_q_waves - 1) / num_q_waves;
      const int32_t q_head_start = q_wave_idx * q_heads_per_wave;
      const int32_t q_head_end = min(q_head_start + q_heads_per_wave, params.num_heads);

      const int64_t token_q_base = static_cast<int64_t>(token_idx) * params.q_stride_0;
      const int64_t token_qout_base = static_cast<int64_t>(token_idx) * params.q_out_stride_0;

      // Q-quant constants. Q_REDUCE must be in {1,2,4,8,16,32,64} and Q_GROUP_SIZE must divide head_dim.
      constexpr int32_t Q_REDUCE = Q_GROUP_SIZE / vec_size_i;
      constexpr int32_t Q_NUM_GROUPS = head_size / Q_GROUP_SIZE;
      static_assert(head_size % Q_GROUP_SIZE == 0, "head_size must be divisible by Q_GROUP_SIZE");
      static_assert(Q_REDUCE >= 1 && Q_REDUCE <= 64 && (Q_REDUCE & (Q_REDUCE - 1)) == 0,
                    "Q_REDUCE (Q_GROUP_SIZE/vec_size_i) must be a power of 2 in [1,64]");

      // q_weight is loaded once per Q head (same across all heads since the weight is shared).
      // We could hoist this out of the head loop, but the cost is negligible (1 load / 16B / thread).
      opus_vec_i vec_q_weight;
      if constexpr (HAS_Q_WEIGHT) {
        vec_q_weight = *reinterpret_cast<const opus_vec_i*>(&q_weight[tid * vec_size_i]);
      }

      // Hoist RoPE cos/sin: cos_ptr[cos_i] is identical for every Q head of this token,
      // so load it once per wave (not once per head) and reuse across the HPW-head loop.
      const bool is_pe_thread = (tid >= pe_tid_start && tid < pe_tid_end);
      float pe_cos[vec_size_i], pe_sin[vec_size_i];
      if (is_pe_thread) {
        const int32_t pe_local_tid = tid - pe_tid_start;
        if constexpr (is_neox) {
          constexpr int32_t half_pe_threads = pe_dim / vec_size_i / 2;  // 4
          const bool is_x_half = (pe_local_tid < half_pe_threads);
          #pragma unroll
          for (int i = 0; i < vec_size_i; i++) {
            const int32_t cos_i = is_x_half ? (pe_local_tid * vec_size_i + i)
                                            : ((pe_local_tid - half_pe_threads) * vec_size_i + i);
            pe_cos[i] = static_cast<float>(cos_ptr[cos_i]);
            pe_sin[i] = static_cast<float>(sin_ptr[cos_i]);
          }
        } else {
          #pragma unroll
          for (int i = 0; i < vec_size_i; i += 2) {
            const int32_t cos_i = (pe_local_tid * vec_size_i + i) >> 1;
            pe_cos[i] = static_cast<float>(cos_ptr[cos_i]);
            pe_sin[i] = static_cast<float>(sin_ptr[cos_i]);
          }
        }
      }

      // Build the q buffer descriptor ONCE per wave (base = this token's q row); load each
      // head via a uniform per-head scalar offset (soffset) instead of rebuilding the SRD
      // (the make_gmem readfirstlane/saveexec pattern) for every head.
      const unsigned q_buf_bytes =
          static_cast<unsigned>(params.num_heads) * params.q_stride_1 * sizeof(scalar_t);
      auto q_buf = opus::make_gmem<scalar_t>(q + token_q_base, q_buf_bytes);

      for (int32_t q_head_idx = q_head_start; q_head_idx < q_head_end; q_head_idx++) {
        // Unified vec8 load: all 64 threads load 8 elements covering full head_dim.
        // (Tried vLLM-style 2-deep prefetch of the next head here -- measured neutral:
        // the compiler already pipelines the loop load, and the extra live vec_q_next
        // costs VGPR/occupancy, so it's a wash. Kept the simple single-buffer load.)
        opus_vec_i vec_q =
            load_vector_nbytes<scalar_t, vec_size_i, in_chunk_bytes, IN_LOAD_AUX>(
                q_buf, tid * vec_size_i + q_head_idx * params.q_stride_1);

        float sum_sq = 0.0f;
        #pragma unroll
        for (int i = 0; i < vec_size_i; i++) {
          float val = static_cast<float>(vec_q[i]);
          sum_sq += val * val;
        }

        auto sum_func = [](float a, float b) { return a + b; };
        float total_sum_sq = wave_reduce<float, decltype(sum_func), vec_stride, true>(sum_sq, sum_func);
        const float q_rms_scale = rsqrtf(total_sum_sq / static_cast<float>(head_size) + eps);

        // Step 1: per-thread normalized + (optional) q_weight
        float q_normed[vec_size_i];
        #pragma unroll
        for (int i = 0; i < vec_size_i; i++) {
          float v = static_cast<float>(vec_q[i]) * q_rms_scale;
          if constexpr (HAS_Q_WEIGHT) {
            v *= static_cast<float>(vec_q_weight[i]);
          }
          q_normed[i] = v;
        }

        // Step 2: RoPE on pe threads (hoisted pe_cos/pe_sin), identity on nope threads.
        float rotated[vec_size_i];
        if (is_pe_thread) {
          const int32_t pe_local_tid = tid - pe_tid_start;  // 0..7
          if constexpr (is_neox) {
            constexpr int32_t half_pe_threads = pe_dim / vec_size_i / 2;  // 4
            const bool is_x_half = (pe_local_tid < half_pe_threads);
            #pragma unroll
            for (int i = 0; i < vec_size_i; i++) {
              float my_val = q_normed[i];
              float pair_val = __shfl_xor(my_val, half_pe_threads, WARP_SIZE);
              rotated[i] = is_x_half ? (my_val * pe_cos[i] - pair_val * pe_sin[i])
                                     : (my_val * pe_cos[i] + pair_val * pe_sin[i]);
            }
          } else {
            #pragma unroll
            for (int i = 0; i < vec_size_i; i += 2) {
              float fqx = q_normed[i];
              float fqy = q_normed[i + 1];
              rotated[i]     = fqx * pe_cos[i] - fqy * pe_sin[i];
              rotated[i + 1] = fqy * pe_cos[i] + fqx * pe_sin[i];
            }
          }
        } else {
          #pragma unroll
          for (int i = 0; i < vec_size_i; i++) rotated[i] = q_normed[i];
        }

        // Step 3: write out. q_out base is the per-token-per-head row; every thread writes 8 elements
        // at offset tid*vec_size_i. For nope_first this puts nope in [0..nope_dim), pe in [nope_dim..head_size);
        // for !nope_first the same tid*vec_size_i mapping places pe threads (tid<8) at [0..pe_dim) and
        // nope threads (tid>=8) at [pe_dim..head_size). Either way, this is a fully coalesced 64-lane store.
        if constexpr (q_dt != vllm::Fp8KVCacheDataType::kAuto) {
          // FP8 Q mirrors the K layout: quantize NOPE only (1xGROUP_SIZE e8m0), write nope
          // fp8 + inline duplicated e8m0 scale into q_out (q_nope_scale_buff, 512B), and
          // write the rotated PE as bf16 into the separate q_rope_out (Q-PE NOT quantized).
          const bool is_nope_thr = (tid < nope_vec);  // nope-first
          float thread_max = 0.0f;
          #pragma unroll
          for (int i = 0; i < vec_size_i; i++) thread_max = fmaxf(thread_max, fabsf(rotated[i]));
          // Group-amax over the Q_REDUCE-lane group via __shfl_xor (DPP corrupts some Q
          // nope groups here). pe lanes reduce among themselves and are discarded.
          #pragma unroll
          for (int offset = Q_REDUCE / 2; offset > 0; offset >>= 1) {
            thread_max = fmaxf(thread_max, __shfl_xor(thread_max, offset, WARP_SIZE));
          }
          // E8M0 block scale via the shared MX helper, RoundUp mode (same as K).
          constexpr MxDtype kQMxDt = kHwFp8E4m3Dtype;
          const E8m0BlockScale qs_scale =
              fp_f32_to_e8m0_block_scale<MxScaleRoundMode::RoundUp, kQMxDt>(thread_max);
          const float inv_scale = is_nope_thr ? (1.0f / qs_scale.dq_scale) : 0.0f;

          query_t* q_out_head = q_out + token_qout_base + q_head_idx * params.q_out_stride_1;
          if (is_nope_thr) {
            // group-leader writes the e8m0 scale TWICE (s,s) at byte [nope_dim + 2*group_id).
            if (tid % Q_REDUCE == 0) {
              // group_id = (tid * vec_size_i) / Q_GROUP_SIZE = tid / Q_REDUCE; generic over
              // Q_GROUP_SIZE (the compiler folds to a shift since Q_REDUCE is a power of 2).
              const int group_id = tid / Q_REDUCE;  // 0..Q_NUM_GROUPS-1
              auto* qs = reinterpret_cast<uint8_t*>(q_out_head) + nope_dim;
              const uint16_t scale_pair =
                  static_cast<uint16_t>(qs_scale.byte) | (static_cast<uint16_t>(qs_scale.byte) << 8);
              *reinterpret_cast<uint16_t*>(qs + group_id * 2) = scale_pair;
            }
            const uint32_t nope_out_offset = tid * vec_size_i;  // nope-first
            opus_vec_q vec_out;
            #pragma unroll
            for (int i = 0; i < vec_size_i; i++) vec_out[i] = opus::cast<query_t>(rotated[i] * inv_scale);
            auto q_out_buf = opus::make_gmem<query_t>(q_out_head, q_oob_o * sizeof(query_t));
            q_out_buf.template store<vec_size_o>(vec_out, nope_out_offset);
          }
          if (tid >= pe_tid_start && tid < pe_tid_end) {
            const int32_t pe_local_tid = tid - pe_tid_start;
            scalar_t* q_rope_head = q_rope_out
                + static_cast<int64_t>(token_idx) * params.q_rope_out_stride_0
                + q_head_idx * params.q_rope_out_stride_1;
            opus_vec_i vrope;
            #pragma unroll
            for (int i = 0; i < vec_size_i; i++) vrope[i] = static_cast<scalar_t>(rotated[i]);
            *reinterpret_cast<opus_vec_i*>(&q_rope_head[pe_local_tid * vec_size_i]) = vrope;
          }
          (void)q_scale_raw;  // legacy separate-scale param unused on the inline path
        } else {
          // bf16 output — write rotated as scalar_t (no quant)
          opus_vec_i vec_out;
          #pragma unroll
          for (int i = 0; i < vec_size_i; i++) {
            vec_out[i] = static_cast<scalar_t>(rotated[i]);
          }
          scalar_t* q_out_head = reinterpret_cast<scalar_t*>(q_out) + token_qout_base + q_head_idx * params.q_out_stride_1;
          auto q_out_buf = opus::make_gmem<scalar_t>(q_out_head, q_oob_o * sizeof(scalar_t));
          q_out_buf.template store<vec_size_o>(vec_out, tid * vec_size_i);
        }
      } // end multi-head Q loop
      } // end Q processing (else branch of is_k_wave)
    }

    // GPT-J / neox RoPE for one PE thread's vec slice (fine-grained kernel helper).
    // Normalizes the raw input (x*rstd*[w]) then rotates; writes fp32 results to
    // ``out``. Must be called by ALL pe lanes (neox uses an intra-pe __shfl_xor).
    template <typename scalar_t, bool is_neox, int vec_size_i, int pe_dim, int pe_tid_start>
    __device__ inline void q_rope_gptj(
        const opus::vector_t<scalar_t, vec_size_i>& vec_q,
        const opus::vector_t<scalar_t, vec_size_i>& vec_w,
        float rms_scale, int tid,
        const scalar_t* __restrict__ cos_ptr, const scalar_t* __restrict__ sin_ptr,
        float (&out)[vec_size_i], bool has_weight) {
      float normed[vec_size_i];
      #pragma unroll
      for (int i = 0; i < vec_size_i; i++) {
        float w = has_weight ? static_cast<float>(vec_w[i]) : 1.0f;
        normed[i] = static_cast<float>(vec_q[i]) * rms_scale * w;
      }
      const int pe_local_tid = tid - pe_tid_start;
      if constexpr (is_neox) {
        constexpr int half_pe_threads = pe_dim / vec_size_i / 2;
        const bool is_x_half = (pe_local_tid < half_pe_threads);
        #pragma unroll
        for (int i = 0; i < vec_size_i; i++) {
          const int cos_i = is_x_half ? (pe_local_tid * vec_size_i + i)
                                      : ((pe_local_tid - half_pe_threads) * vec_size_i + i);
          float f32_cos = static_cast<float>(cos_ptr[cos_i]);
          float f32_sin = static_cast<float>(sin_ptr[cos_i]);
          float my_val = normed[i];
          float pair_val = __shfl_xor(my_val, half_pe_threads, WARP_SIZE);
          out[i] = is_x_half ? (my_val * f32_cos - pair_val * f32_sin)
                             : (my_val * f32_cos + pair_val * f32_sin);
        }
      } else {
        #pragma unroll
        for (int i = 0; i < vec_size_i; i += 2) {
          const int cos_i = (pe_local_tid * vec_size_i + i) >> 1;
          float f32_cos = static_cast<float>(cos_ptr[cos_i]);
          float f32_sin = static_cast<float>(sin_ptr[cos_i]);
          float fqx = normed[i];
          float fqy = normed[i + 1];
          out[i]     = fqx * f32_cos - fqy * f32_sin;
          out[i + 1] = fqy * f32_cos + fqx * f32_sin;
        }
      }
    }

    // ===========================================================================
    // Fine-grained variant (FlyDSL-style decomposition) -- auto-selected for the
    // xlarge prefill tier (T >= ~8k, num_tokens <= 65535); ~5-17% faster there.
    // ---------------------------------------------------------------------------
    // One block == one wave (64 threads) == exactly ONE (token, head) tile:
    //   grid.x = num_heads + 1 (head; 0 -> K, 1.. -> Q), grid.y = num_tokens.
    // Mirrors flydsl: no head loop, no tokens-per-block packing, head on the fast
    // grid dim (co-scheduled blocks read contiguous q[token,*,:] rows). It also
    // folds the quant: amax is taken over the RAW pre-norm input (so it fuses with
    // the row-sum butterfly), and rstd is folded into a single forward factor
    // applied to x_in directly -> fewer multiplies / live registers.
    //
    // MEASURED (MI355, gfx950): this MATCHES the coarse kernel, it does not beat it.
    // The coarse kernel already runs at the HW occupancy cap (32 VGPR -> ~8 waves/
    // SIMD), so "more, smaller waves" buys nothing -- the kernel is memory-traffic
    // bound, not occupancy bound. The residual gap to flydsl's wall-clock is mostly
    // that flydsl stores PE as fp8 (64 B) while the V4 nm asm layout requires PE as
    // bf16 (128 B), i.e. a format difference, not a schedule difference. Kept as a
    // documented, correct A/B baseline for future memory-layout experiments.
    // The per-tile math + v4 nm asm store layout are identical to the coarse kernel.
    // ===========================================================================
    template <typename scalar_t, typename cache_t, typename query_t,
              vllm::Fp8KVCacheDataType kv_dt, vllm::Fp8KVCacheDataType q_dt, bool is_neox,
              int Q_GROUP_SIZE = 64, bool Q_SCALE_FP32 = false, bool HAS_Q_WEIGHT = false,
              int HEAD_DIM = 512, int WAVE = 64>
    __device__ void fuse_qk_norm_rope_finegrained_impl(
        const scalar_t* __restrict__ q,
        const scalar_t* __restrict__ kv,
        scalar_t* __restrict__ k_pe_out,
        const scalar_t* __restrict__ k_weight,
        const scalar_t* __restrict__ q_weight,
        cache_t* __restrict__ kv_cache,
        query_t* __restrict__ q_out,
        void* __restrict__ q_scale_raw,
        scalar_t* __restrict__ q_rope_out,
        const int64_t* __restrict__ positions,
        const scalar_t* __restrict__ cos_cache,
        const scalar_t* __restrict__ sin_cache,
        float eps,
        const MlaKernelParams& __restrict__ params,
        // Per-tile coordinates (supplied by the launching global): one wave computes
        // exactly this (token_idx, combined_head_idx) tile. combined_head_idx==0 -> K,
        // 1.. -> Q head (combined-1). tid is the lane 0..63 within the wave.
        int32_t token_idx, int32_t combined_head_idx, int32_t tid
    ) {
      // ---- compile-time constants (identical to the coarse kernel) ----
      constexpr int32_t head_size = HEAD_DIM;
      constexpr int32_t pe_dim = 64;
      constexpr int32_t nope_dim = head_size - pe_dim;
      // Elements/lane to cover one head with a single wave = HEAD_DIM/WARP_SIZE
      // (wave64: 8 for head=512; wave32: 16). WARP_SIZE is the device-resolved wavefront
      // size; safe in this __device__ constexpr (matches the shared K-wave body). The trailing
      // WAVE template param is kept for API compatibility but is no longer consulted, so the
      // Q path is wave-generic without an extra per-wave-size instantiation.
      // fp32 is a dead-code instantiation; cap at 4 (no 32-byte opus float vector).
      constexpr int32_t kWave = WARP_SIZE;
      constexpr int32_t vec_size_i =
          std::is_same_v<scalar_t, float> ? 4 : ((HEAD_DIM / 8 <= kWave) ? 8 : 16);
      constexpr int32_t vec_size_o = vec_size_i;
      constexpr uint32_t nope_vec = nope_dim / vec_size_o;
      constexpr int32_t nope_offset = 0;            // V4 layout: nope-first
      constexpr int32_t pe_tid_start = nope_vec;
      constexpr int32_t pe_tid_end = pe_tid_start + (pe_dim / vec_size_i);
      constexpr int32_t ooba_i = 4 / sizeof(scalar_t);
      constexpr int32_t ooba_o = 4 / sizeof(cache_t);
      constexpr int32_t oob_i = (head_size + ooba_i - 1) / ooba_i * ooba_i;
      constexpr int32_t oob_o = (head_size + ooba_o - 1) / ooba_o * ooba_o;
      constexpr int32_t q_ooba_o = 4 / sizeof(query_t);
      constexpr int32_t q_oob_o = (head_size + q_ooba_o - 1) / q_ooba_o * q_ooba_o;
      constexpr int32_t GROUP_SIZE = 64;
      constexpr int32_t reduce_thread_size = GROUP_SIZE / vec_size_i;
      // Streaming (read-once) inputs -> NT/SLC|GLC to bypass L2 (same as the coarse
      // kernel). Measured neutral here vs plain cached loads, kept for consistency.
      constexpr int32_t IN_LOAD_AUX = (sizeof(scalar_t) < 4) ? GROUP_NT : 0;
      constexpr int32_t in_chunk_bytes = (vec_size_i * sizeof(scalar_t)) % 16 == 0 ? 16 : 8;

      using opus_vec_i = opus::vector_t<scalar_t, vec_size_i>;
      using opus_vec_o = opus::vector_t<cache_t, vec_size_o>;
      using opus_vec_q = opus::vector_t<query_t, vec_size_o>;

      // ---- this wave computes one (token, head) tile (coords passed by the global) ----
      if (token_idx >= params.num_tokens) return;
      const bool is_k_wave = (combined_head_idx == 0);  // V4 MQA: single K wave

      const int32_t cos_sin_offset = static_cast<int32_t>(positions[token_idx]) * (pe_dim >> 1);
      const scalar_t* cos_ptr = cos_cache + cos_sin_offset;
      const scalar_t* sin_ptr = sin_cache + cos_sin_offset;

      if (is_k_wave) {
        // ===== K: RMSNorm over head_dim, e8m0 group-quant nope, RoPE pe (bf16) =====
        const int64_t kv_cache_offset = static_cast<int64_t>(token_idx) * params.token_stride;
        const int64_t token_kv_base   = static_cast<int64_t>(token_idx) * params.kv_stride_0;
        const scalar_t* kv_ptr = kv + token_kv_base;            // single KV head
        auto* ptr_o = kv_cache + kv_cache_offset + nope_offset; // single KV head
        auto buffer_kv = opus::make_gmem<scalar_t>(kv_ptr, oob_i * sizeof(scalar_t));
        auto buffer_o  = opus::make_gmem<cache_t>(ptr_o, oob_o * sizeof(cache_t));

        const bool is_nope_thread = (tid < static_cast<int32_t>(nope_vec));
        constexpr bool K_QUANT = (kv_dt != vllm::Fp8KVCacheDataType::kAuto);

        opus_vec_i vec_kv =
            load_vector_nbytes<scalar_t, vec_size_i, in_chunk_bytes, IN_LOAD_AUX>(
                buffer_kv, tid * vec_size_i);
        opus_vec_i vec_k_weight = *reinterpret_cast<const opus_vec_i*>(&k_weight[tid * vec_size_i]);

        // Per-thread partials: sum(x^2) over the row, and (quant only) amax(|x*w|)
        // over this thread's slice -- both on the RAW input (pre-norm), so the
        // group-amax can fuse with the row-sum butterfly (rstd folds in later).
        float sum_sq = 0.0f;
        float amax_raw = 0.0f;
        #pragma unroll
        for (int i = 0; i < vec_size_i; ++i) {
          float val = static_cast<float>(vec_kv[i]);
          sum_sq += val * val;
          if constexpr (K_QUANT) {
            if (is_nope_thread) amax_raw = fmaxf(amax_raw, fabsf(val * static_cast<float>(vec_k_weight[i])));
          }
        }
        // Fused butterfly: row-sum over all 64 lanes; group-amax only over the
        // last log2(reduce_thread_size) steps (offset < group width) so the two
        // shuffle chains overlap (the LLVM scheduler hides one behind the other).
        #pragma unroll
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
          sum_sq += __shfl_xor(sum_sq, off, WARP_SIZE);
          if constexpr (K_QUANT) {
            if (off < reduce_thread_size) amax_raw = fmaxf(amax_raw, __shfl_xor(amax_raw, off, WARP_SIZE));
          }
        }
        const float rms_scale = rsqrtf(sum_sq / static_cast<float>(head_size) + eps);

        // nope: fold rstd + dequant-scale into a single forward factor applied to
        // x_in directly (out = x_in * w * rms_scale / dq_scale).
        if constexpr (K_QUANT) {
          const float amax_norm = amax_raw * rms_scale;  // == amax(|x_norm|) over the group
          float factor;
          if constexpr (std::is_same_v<cache_t, opus::fp8_t>) {
            constexpr MxDtype kMxDt = kHwFp8E4m3Dtype;
            const E8m0BlockScale s =
                fp_f32_to_e8m0_block_scale<MxScaleRoundMode::RoundUp, kMxDt>(amax_norm);
            if (is_nope_thread && (tid % reduce_thread_size) == 0) {
              // K NoPE is always group=64 (GROUP_SIZE hardcoded above for the asm reader's
              // 14-byte format); use the generic tid/reduce_thread_size to avoid a magic >>6.
              const int group_id = tid / reduce_thread_size;
              auto* tmp = reinterpret_cast<uint8_t*>(ptr_o + nope_dim);
              const uint16_t scale_pair =
                  static_cast<uint16_t>(s.byte) | (static_cast<uint16_t>(s.byte) << 8);
              *reinterpret_cast<uint16_t*>(tmp + group_id * 2) = scale_pair;
            }
            factor = rms_scale / s.dq_scale;
          } else {
            const float group_scale = amax_norm / opus::finfo<cache_t>::max();
            factor = rms_scale / group_scale;
          }
          if (is_nope_thread) {
            opus_vec_o vec_out;
            #pragma unroll
            for (int i = 0; i < vec_size_i; i++)
              vec_out[i] = opus::cast<cache_t>(static_cast<float>(vec_kv[i])
                              * static_cast<float>(vec_k_weight[i]) * factor);
            buffer_o.template store<vec_size_o>(vec_out, tid * vec_size_i);
          }
        } else if (is_nope_thread) {
          opus_vec_i vec_out_k;
          #pragma unroll
          for (int i = 0; i < vec_size_i; i++)
            vec_out_k[i] = static_cast<scalar_t>(static_cast<float>(vec_kv[i]) * rms_scale
                              * static_cast<float>(vec_k_weight[i]));
          buffer_o.template store<vec_size_o>(vec_out_k, tid * vec_size_i);
        }

        // RoPE pe -> separate rope_buff (bf16, not quantized). normed = x_in*rstd*w.
        scalar_t* k_out_rope = k_pe_out + static_cast<int64_t>(token_idx) * params.k_pe_out_stride_0;
        if (tid >= pe_tid_start && tid < pe_tid_end) {
          float k_normed[vec_size_i];
          #pragma unroll
          for (int i = 0; i < vec_size_i; i++)
            k_normed[i] = static_cast<float>(vec_kv[i]) * rms_scale * static_cast<float>(vec_k_weight[i]);
          const int32_t pe_local_tid = tid - pe_tid_start;
          if constexpr (is_neox) {
            constexpr int32_t half_pe_threads = pe_dim / vec_size_i / 2;
            const bool is_x_half = (pe_local_tid < half_pe_threads);
            #pragma unroll
            for (int i = 0; i < vec_size_i; i++) {
              float my_val = k_normed[i];
              float pair_val = __shfl_xor(my_val, half_pe_threads, WARP_SIZE);
              int32_t cos_i = is_x_half ? (pe_local_tid * vec_size_i + i)
                                        : ((pe_local_tid - half_pe_threads) * vec_size_i + i);
              float f32_cos = static_cast<float>(cos_ptr[cos_i]);
              float f32_sin = static_cast<float>(sin_ptr[cos_i]);
              float rot = is_x_half ? (my_val * f32_cos - pair_val * f32_sin)
                                    : (my_val * f32_cos + pair_val * f32_sin);
              k_out_rope[pe_local_tid * vec_size_i + i] = static_cast<scalar_t>(rot);
            }
          } else {
            #pragma unroll
            for (int i = 0; i < vec_size_i; i += 2) {
              float fkx = k_normed[i];
              float fky = k_normed[i + 1];
              int32_t cos_i = (pe_local_tid * vec_size_i + i) >> 1;
              float f32_cos = static_cast<float>(cos_ptr[cos_i]);
              float f32_sin = static_cast<float>(sin_ptr[cos_i]);
              k_out_rope[pe_local_tid * vec_size_i + i]     = static_cast<scalar_t>(fkx * f32_cos - fky * f32_sin);
              k_out_rope[pe_local_tid * vec_size_i + i + 1] = static_cast<scalar_t>(fky * f32_cos + fkx * f32_sin);
            }
          }
        }
        return;
      }

      // ===== Q (single head = blockIdx.y - 1): RMSNorm + opt q_weight + RoPE + opt fp8 quant =====
      const int32_t q_head_idx = combined_head_idx - 1;
      const int64_t token_q_base    = static_cast<int64_t>(token_idx) * params.q_stride_0;
      const int64_t token_qout_base = static_cast<int64_t>(token_idx) * params.q_out_stride_0;

      constexpr int32_t Q_REDUCE = Q_GROUP_SIZE / vec_size_i;
      static_assert(head_size % Q_GROUP_SIZE == 0, "head_size must be divisible by Q_GROUP_SIZE");
      static_assert(Q_REDUCE >= 1 && Q_REDUCE <= 64 && (Q_REDUCE & (Q_REDUCE - 1)) == 0,
                    "Q_REDUCE (Q_GROUP_SIZE/vec_size_i) must be a power of 2 in [1,64]");
      constexpr bool Q_QUANT = (q_dt != vllm::Fp8KVCacheDataType::kAuto);
      const bool is_pe_thread = (tid >= pe_tid_start && tid < pe_tid_end);
      const bool is_nope_thr  = (tid < static_cast<int32_t>(nope_vec));

      const scalar_t* q_ptr = q + token_q_base + q_head_idx * params.q_stride_1;
      auto q_buf = opus::make_gmem<scalar_t>(q_ptr, oob_i * sizeof(scalar_t));
      opus_vec_i vec_q =
          load_vector_nbytes<scalar_t, vec_size_i, in_chunk_bytes, IN_LOAD_AUX>(
              q_buf, tid * vec_size_i);

      opus_vec_i vec_q_weight;
      if constexpr (HAS_Q_WEIGHT) {
        vec_q_weight = *reinterpret_cast<const opus_vec_i*>(&q_weight[tid * vec_size_i]);
      }

      // Q quant only touches NOPE (PE stays bf16), so the group-amax is taken over
      // the RAW nope input (pre-norm, pre-RoPE) and fused with the row-sum butterfly.
      float sum_sq = 0.0f;
      float amax_raw = 0.0f;
      #pragma unroll
      for (int i = 0; i < vec_size_i; i++) {
        float val = static_cast<float>(vec_q[i]);
        sum_sq += val * val;
        if constexpr (Q_QUANT) {
          if (is_nope_thr) {
            float w = HAS_Q_WEIGHT ? static_cast<float>(vec_q_weight[i]) : 1.0f;
            amax_raw = fmaxf(amax_raw, fabsf(val * w));
          }
        }
      }
      #pragma unroll
      for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        sum_sq += __shfl_xor(sum_sq, off, WARP_SIZE);
        if constexpr (Q_QUANT) {
          if (off < Q_REDUCE) amax_raw = fmaxf(amax_raw, __shfl_xor(amax_raw, off, WARP_SIZE));
        }
      }
      const float q_rms_scale = rsqrtf(sum_sq / static_cast<float>(head_size) + eps);

      if constexpr (Q_QUANT) {
        const float amax_norm = amax_raw * q_rms_scale;  // == amax(|q_norm|) over the group
        constexpr MxDtype kQMxDt = kHwFp8E4m3Dtype;
        const E8m0BlockScale qs_scale =
            fp_f32_to_e8m0_block_scale<MxScaleRoundMode::RoundUp, kQMxDt>(amax_norm);
        const float factor = q_rms_scale / qs_scale.dq_scale;  // x_in -> fp8 (rstd folded)

        query_t* q_out_head = q_out + token_qout_base + q_head_idx * params.q_out_stride_1;
        if (is_nope_thr) {
          if ((tid % Q_REDUCE) == 0) {
            // group_id = (tid * vec_size_i) / Q_GROUP_SIZE = tid / Q_REDUCE; generic over
            // Q_GROUP_SIZE (folds to a shift since Q_REDUCE is a power of 2).
            const int group_id = tid / Q_REDUCE;
            auto* qs = reinterpret_cast<uint8_t*>(q_out_head) + nope_dim;
            const uint16_t scale_pair =
                static_cast<uint16_t>(qs_scale.byte) | (static_cast<uint16_t>(qs_scale.byte) << 8);
            *reinterpret_cast<uint16_t*>(qs + group_id * 2) = scale_pair;
          }
          opus_vec_q vec_out;
          #pragma unroll
          for (int i = 0; i < vec_size_i; i++) {
            float w = HAS_Q_WEIGHT ? static_cast<float>(vec_q_weight[i]) : 1.0f;
            vec_out[i] = opus::cast<query_t>(static_cast<float>(vec_q[i]) * w * factor);
          }
          auto q_out_buf = opus::make_gmem<query_t>(q_out_head, q_oob_o * sizeof(query_t));
          q_out_buf.template store<vec_size_o>(vec_out, tid * vec_size_i);
        }
        // PE: RoPE on the normed (bf16) values into the separate q_rope_out.
        if (is_pe_thread) {
          float rotated[vec_size_i];
          q_rope_gptj<scalar_t, is_neox, vec_size_i, pe_dim, pe_tid_start>(
              vec_q, vec_q_weight, q_rms_scale, tid, cos_ptr, sin_ptr, rotated,
              HAS_Q_WEIGHT);
          const int32_t pe_local_tid = tid - pe_tid_start;
          scalar_t* q_rope_head = q_rope_out
              + static_cast<int64_t>(token_idx) * params.q_rope_out_stride_0
              + q_head_idx * params.q_rope_out_stride_1;
          opus_vec_i vrope;
          #pragma unroll
          for (int i = 0; i < vec_size_i; i++) vrope[i] = static_cast<scalar_t>(rotated[i]);
          *reinterpret_cast<opus_vec_i*>(&q_rope_head[pe_local_tid * vec_size_i]) = vrope;
        }
        (void)q_scale_raw;
      } else {
        // bf16 Q: write the full rotated head (nope = normed, pe = RoPE(normed)).
        float out_f[vec_size_i];
        if (is_pe_thread) {
          q_rope_gptj<scalar_t, is_neox, vec_size_i, pe_dim, pe_tid_start>(
              vec_q, vec_q_weight, q_rms_scale, tid, cos_ptr, sin_ptr, out_f, HAS_Q_WEIGHT);
        } else {
          #pragma unroll
          for (int i = 0; i < vec_size_i; i++) {
            float w = HAS_Q_WEIGHT ? static_cast<float>(vec_q_weight[i]) : 1.0f;
            out_f[i] = static_cast<float>(vec_q[i]) * q_rms_scale * w;
          }
        }
        opus_vec_i vec_out;
        #pragma unroll
        for (int i = 0; i < vec_size_i; i++) vec_out[i] = static_cast<scalar_t>(out_f[i]);
        scalar_t* q_out_head =
            reinterpret_cast<scalar_t*>(q_out) + token_qout_base + q_head_idx * params.q_out_stride_1;
        auto q_out_buf = opus::make_gmem<scalar_t>(q_out_head, q_oob_o * sizeof(scalar_t));
        q_out_buf.template store<vec_size_o>(vec_out, tid * vec_size_i);
      }
    }

    // Fine-grained global. block = HEADS_PER_BLOCK waves (= the TOKENS_PER_BLOCK template
    // slot, reused so the coarse CALL macro works unchanged). grid.x = ceil((num_heads+1)
    // / HEADS_PER_BLOCK), grid.y = num_tokens. Each wave handles one (token, head) tile;
    // the HEADS_PER_BLOCK waves of a block share the SAME token and cover consecutive
    // heads, so a block reads the contiguous q[token, h..h+HPB, :] span. HEADS_PER_BLOCK=1
    // degenerates to the 1-wave-per-block variant (head on the fast grid dim).
    template <typename scalar_t, typename cache_t, typename query_t,
              vllm::Fp8KVCacheDataType kv_dt, vllm::Fp8KVCacheDataType q_dt,
              int Q_GROUP_SIZE = 64, bool Q_SCALE_FP32 = false, bool HAS_Q_WEIGHT = false,
              int HEAD_DIM = 512, int TOKENS_PER_BLOCK = 1 /*== HEADS_PER_BLOCK here*/>
    __global__ __launch_bounds__(TOKENS_PER_BLOCK * 64, 512 / (TOKENS_PER_BLOCK * 64))
    void fuse_qk_norm_rope_finegrained_kernel(
        const scalar_t* __restrict__ q,
        const scalar_t* __restrict__ kv,
        scalar_t* __restrict__ k_pe_out,
        const scalar_t* __restrict__ k_weight,
        const scalar_t* __restrict__ q_weight,
        cache_t* __restrict__ kv_cache,
        query_t* __restrict__ q_out,
        void* __restrict__ q_scale_raw,
        scalar_t* __restrict__ q_rope_out,
        const int64_t* __restrict__ positions,
        const scalar_t* __restrict__ cos_cache,
        const scalar_t* __restrict__ sin_cache,
        float eps,
        const MlaKernelParams params,
        bool is_neox
    ) {
      constexpr int HEADS_PER_BLOCK = TOKENS_PER_BLOCK;
      const int32_t wave_id = static_cast<int32_t>(threadIdx.x) / WARP_SIZE;
      const int32_t tid     = static_cast<int32_t>(threadIdx.x) % WARP_SIZE;
      const int32_t token_idx = static_cast<int32_t>(blockIdx.y);
      const int32_t combined_head_idx = static_cast<int32_t>(blockIdx.x) * HEADS_PER_BLOCK + wave_id;
      if (combined_head_idx > params.num_heads) return;  // last block may overhang
      #define DISPATCH_NEOX_FG(NEOX) \
        fuse_qk_norm_rope_finegrained_impl<scalar_t,cache_t,query_t, kv_dt, q_dt, NEOX, \
            Q_GROUP_SIZE, Q_SCALE_FP32, HAS_Q_WEIGHT, HEAD_DIM>( \
            q, kv, k_pe_out, k_weight, q_weight, kv_cache, q_out, q_scale_raw, q_rope_out, positions, \
            cos_cache, sin_cache, eps, params, token_idx, combined_head_idx, tid)
      if (is_neox) { DISPATCH_NEOX_FG(true); }
      else         { DISPATCH_NEOX_FG(false); }
      #undef DISPATCH_NEOX_FG
    }

    // Unified prefill kernel with RMS Norm and Group Quantization
    // TOKENS_PER_BLOCK=1: single-wave (decode/small prefill), TOKENS_PER_BLOCK>1: multi-wave
    template <typename scalar_t, typename cache_t, typename query_t, vllm::Fp8KVCacheDataType kv_dt, vllm::Fp8KVCacheDataType q_dt,
              int Q_GROUP_SIZE = 64, bool Q_SCALE_FP32 = false, bool HAS_Q_WEIGHT = false,
              int HEAD_DIM = 512, int TOKENS_PER_BLOCK = 1>
    __global__ __launch_bounds__(TOKENS_PER_BLOCK * 64, 512 / (TOKENS_PER_BLOCK * 64))
    void fuse_qk_norm_rope_group_quant_cache_kernel(
        const scalar_t* __restrict__ q,
        const scalar_t* __restrict__ kv,
        scalar_t* __restrict__ k_pe_out,
        const scalar_t* __restrict__ k_weight,
        const scalar_t* __restrict__ q_weight,
        cache_t* __restrict__ kv_cache,
        query_t* __restrict__ q_out,
        void* __restrict__ q_scale_raw,
        scalar_t* __restrict__ q_rope_out,
        const int64_t* __restrict__ positions,
        const scalar_t *__restrict__ cos_cache,
        const scalar_t *__restrict__ sin_cache,
        float eps,
        const MlaKernelParams params,
        bool is_neox
    ) {
      #define DISPATCH_NEOX(NEOX) \
        fuse_qk_norm_rope_group_quant_cache_kernel_impl<scalar_t,cache_t,query_t, kv_dt, q_dt, NEOX, \
            Q_GROUP_SIZE, Q_SCALE_FP32, HAS_Q_WEIGHT, HEAD_DIM, TOKENS_PER_BLOCK>( \
            q, kv, k_pe_out, k_weight, q_weight, kv_cache, q_out, q_scale_raw, q_rope_out, positions, \
            cos_cache, sin_cache, eps, params)

      if (is_neox) { DISPATCH_NEOX(true); }
      else         { DISPATCH_NEOX(false); }
      #undef DISPATCH_NEOX
    }

} // namespace aiter

// Unified macro for fused QK norm + RoPE + group quant + cache kernel
// Requires the following constexpr/locals in scope at the call site:
//   head_dim_val, tokens_per_block_val, q_group_size_val, q_scale_fp32_val, has_q_weight_val
//   q_weight_ptr (scalar_t*, may be nullptr), q_scale_ptr (void*, may be nullptr)
#define CALL_FUSED_QK_NORM_ROPE_GROUP_QUANT_CACHE(KV_T, CACHE_T, QUERY_T, KV_DTYPE, Q_DTYPE)   \
         aiter::fuse_qk_norm_rope_group_quant_cache_kernel<KV_T, CACHE_T, QUERY_T, KV_DTYPE, Q_DTYPE, \
                 q_group_size_val, q_scale_fp32_val, has_q_weight_val, head_dim_val, tokens_per_block_val> \
               <<<grid, block, 0, stream>>>(                                                             \
                 reinterpret_cast<const KV_T*>(q.data_ptr()),                                            \
                 reinterpret_cast<const KV_T*>(kv.data_ptr()),                                           \
                 reinterpret_cast<KV_T*>(k_rope_buff.data_ptr()),                                        \
                 reinterpret_cast<const KV_T*>(k_weight.data_ptr()),                                     \
                 reinterpret_cast<const KV_T*>(q_weight_ptr),                                            \
                 reinterpret_cast<CACHE_T*>(k_nope_scale_buff.data_ptr()),                               \
                 reinterpret_cast<QUERY_T*>(q_nope_scale_buff.data_ptr()),                               \
                 reinterpret_cast<void*>(q_scale_ptr),                                                   \
                 reinterpret_cast<KV_T*>(q_rope_out_ptr),                                                \
                 reinterpret_cast<const int64_t*>(positions.data_ptr()),                                 \
                 reinterpret_cast<const KV_T*>(cos_cache.data_ptr()),                                    \
                 reinterpret_cast<const KV_T*>(sin_cache.data_ptr()),                                    \
                 static_cast<float>(eps),                                                                \
                 mla_params,                                                                             \
                 is_neox);

// Fine-grained launcher (1 wave / (token,head); grid=(num_tokens,num_heads+1), block=64).
// Same arg list / scope requirements as the coarse macro above.
#define CALL_FUSED_QK_NORM_ROPE_FINEGRAINED(KV_T, CACHE_T, QUERY_T, KV_DTYPE, Q_DTYPE)   \
         aiter::fuse_qk_norm_rope_finegrained_kernel<KV_T, CACHE_T, QUERY_T, KV_DTYPE, Q_DTYPE, \
                 q_group_size_val, q_scale_fp32_val, has_q_weight_val, head_dim_val, tokens_per_block_val> \
               <<<grid, block, 0, stream>>>(                                                             \
                 reinterpret_cast<const KV_T*>(q.data_ptr()),                                            \
                 reinterpret_cast<const KV_T*>(kv.data_ptr()),                                           \
                 reinterpret_cast<KV_T*>(k_rope_buff.data_ptr()),                                        \
                 reinterpret_cast<const KV_T*>(k_weight.data_ptr()),                                     \
                 reinterpret_cast<const KV_T*>(q_weight_ptr),                                            \
                 reinterpret_cast<CACHE_T*>(k_nope_scale_buff.data_ptr()),                               \
                 reinterpret_cast<QUERY_T*>(q_nope_scale_buff.data_ptr()),                               \
                 reinterpret_cast<void*>(q_scale_ptr),                                                   \
                 reinterpret_cast<KV_T*>(q_rope_out_ptr),                                                \
                 reinterpret_cast<const int64_t*>(positions.data_ptr()),                                 \
                 reinterpret_cast<const KV_T*>(cos_cache.data_ptr()),                                    \
                 reinterpret_cast<const KV_T*>(sin_cache.data_ptr()),                                    \
                 static_cast<float>(eps),                                                                \
                 mla_params,                                                                             \
                 is_neox);

namespace aiter {

void fused_qk_norm_rope_group_quant(
    aiter_tensor_t& q,                  // [num_tokens, num_heads, head_dim]
    aiter_tensor_t& kv,                 // [num_tokens, (k_num_heads,) head_dim]
    aiter_tensor_t& k_rope_buff,        // [num_tokens, (k_num_heads,) pe_dim] bf16 (RoPE'd K-PE)
    aiter_tensor_t& k_weight,           // [head_dim] RMSNorm weights
    aiter_tensor_t& k_nope_scale_buff,  // [num_tokens, (num_kv_heads,) entry_bytes] K nope+scale
    aiter_tensor_t& q_nope_scale_buff,  // [num_tokens, num_heads, head_dim] bf16 (full Q) OR fp8 (nope+scale)
    aiter_tensor_t& positions,          // [num_tokens]
    aiter_tensor_t& cos_cache,          // [max_position, rot_dim//2]
    aiter_tensor_t& sin_cache,          // [max_position, rot_dim//2]
    double eps,                         // epsilon for RMS norm
    bool is_neox,
    std::optional<aiter_tensor_t> q_weight,
    std::optional<aiter_tensor_t> q_scale,
    int64_t quant_group_size,
    const std::string& scale_dtype,
    std::optional<aiter_tensor_t> q_rope_buff)
{
  int num_tokens = q.size(0);
  int head_dim   = kv.size(-1);
  int num_heads  = q.size(1);
  int rot_dim    = cos_cache.size(-1) * 2;

  AITER_CHECK(q.dim() == 3, "q must be 3D [num_tokens, num_heads, head_dim]");
  AITER_CHECK(q.size(-1) == head_dim, "q head_dim must equal kv head_dim");
  AITER_CHECK(q_nope_scale_buff.size(2) == head_dim, "q_nope_scale_buff last dim must match head_dim");
  AITER_CHECK(k_weight.size(0) == head_dim, "k_weight size must match head_dim");
  AITER_CHECK(kv.stride(-1) == 1, "kv stride(-1) must be equal to 1");

  // --- Validate Q-quant / q_weight options ---
  const bool has_q_weight = q_weight.has_value();
  if (has_q_weight) {
    AITER_CHECK(q_weight->size(0) == head_dim, "q_weight size must match head_dim");
    AITER_CHECK(q_weight->dtype() == q.dtype(),
                "q_weight dtype must match q dtype");
    AITER_CHECK(q_weight->stride(-1) == 1, "q_weight must be contiguous in last dim");
  }
  // q_out_type: "auto" = bf16/same-as-input, "fp8" = group-quantised fp8
  std::string q_out_type = "auto";
  if (q_nope_scale_buff.dtype() == AITER_DTYPE_fp8) {
    q_out_type = "fp8";
  }
  const bool q_is_fp8 = (q_out_type == "fp8");
  // fp8 Q mirrors K: q_nope_scale_buff holds nope fp8 + inline dup e8m0 scale, and the
  // rotated Q-PE goes to the separate q_rope_buff (bf16). The e8m0 scale is written
  // inline, so a separate q_scale tensor is no longer required.
  if (q_is_fp8) {
    // group=128 is excluded: the NoPE region (head_dim - rot_dim) is the only part that is
    // group-quantised, and for the V4 shape (512-64=448) 448 % 128 != 0. Only {32, 64} tile
    // the 448-wide NoPE evenly.
    AITER_CHECK(quant_group_size == 32 || quant_group_size == 64,
                "quant_group_size must be one of {32, 64}, got ", quant_group_size);
    // Validate the NoPE region (the quantised part), not the full head_dim: the trailing
    // rot_dim is RoPE'd bf16 and never quantised.
    AITER_CHECK((head_dim - rot_dim) % quant_group_size == 0,
                "NoPE size (head_dim - rot_dim) must be divisible by quant_group_size");
    AITER_CHECK(q_rope_buff.has_value(),
                "q_rope_buff (rotated Q-PE bf16 buffer) is required when Q is fp8");
  }

  // DeepSeek V4 is MQA: exactly one KV head. The dense kernel hardcodes this
  // (blockIdx.y == 0 is the single K wave), so reject any multi-head kv tensor.
  const int num_kv_heads = (kv.dim() == 3) ? static_cast<int>(kv.size(1)) : 1;
  AITER_CHECK(num_kv_heads == 1,
              "fused_qk_norm_rope_cache_quant requires num_kv_heads == 1 (MQA), got ",
              num_kv_heads);

  int q_stride_0          = q.stride(0);
  int q_stride_1          = q.stride(1);
  int q_out_stride_0      = q_nope_scale_buff.stride(0);
  int q_out_stride_1      = q_nope_scale_buff.stride(1);
  // fp8-Q rope buff (separate rotated Q-PE, bf16). 0 when absent (bf16 Q).
  const bool has_q_rope   = q_rope_buff.has_value();
  int q_rope_out_stride_0 = has_q_rope ? static_cast<int>(q_rope_buff->stride(0)) : 0;
  int q_rope_out_stride_1 = (has_q_rope && q_rope_buff->dim() == 3)
                            ? static_cast<int>(q_rope_buff->stride(1)) : 0;
  // Dense k_nope_scale_buff: [num_tokens, (num_kv_heads,) entry]. token_stride = per-token
  // stride; kv_cache_stride_h = per-kv-head stride (0 when num_kv_heads collapsed / 2D).
  int token_stride        = k_nope_scale_buff.stride(0);
  int kv_stride_0         = kv.stride(0);
  int kv_stride_1         = (kv.dim() == 3) ? kv.stride(1) : 0;
  int k_pe_out_stride_0   = k_rope_buff.stride(0);
  int k_pe_out_stride_1   = (k_rope_buff.dim() == 3) ? k_rope_buff.stride(1) : 0;
  int kv_cache_stride_h   = (k_nope_scale_buff.dim() >= 3) ? k_nope_scale_buff.stride(1) : 0;

  HipDeviceGuard device_guard(kv.device_id);
  const hipStream_t stream = aiter::getCurrentHIPStream();

  // Determine KV cache dtype from the k_nope_scale_buff tensor itself.
  std::string kv_cache_dtype = "auto";
  const AiterDtype cache_dt = k_nope_scale_buff.dtype();
  if (cache_dt == AITER_DTYPE_fp32 ||
      cache_dt == AITER_DTYPE_fp16 ||
      cache_dt == AITER_DTYPE_bf16) {
    kv_cache_dtype = "auto";
  } else if (cache_dt == AITER_DTYPE_fp8) {
    kv_cache_dtype = "fp8";
  } else {
    AITER_CHECK(false, "kv cache data type is not supported: ",
                AiterDtype_to_str(cache_dt));
  }

  constexpr int64_t OPTIMIZED_ROT_DIM = 64;
  const bool use_optimized = (rot_dim == OPTIMIZED_ROT_DIM && head_dim <= 512);

  aiter::MlaKernelParams mla_params{};  // zero-init: paged-cache fields are unused here
  mla_params.token_stride = token_stride;
  mla_params.kv_cache_stride_h = kv_cache_stride_h;
  mla_params.q_stride_0 = q_stride_0;
  mla_params.q_stride_1 = q_stride_1;
  mla_params.q_out_stride_0 = q_out_stride_0;
  mla_params.q_out_stride_1 = q_out_stride_1;
  mla_params.q_rope_out_stride_0 = q_rope_out_stride_0;
  mla_params.q_rope_out_stride_1 = q_rope_out_stride_1;
  mla_params.kv_stride_0 = kv_stride_0;
  mla_params.kv_stride_1 = kv_stride_1;
  mla_params.k_pe_out_stride_0 = k_pe_out_stride_0;
  mla_params.k_pe_out_stride_1 = k_pe_out_stride_1;
  // q_scale strides: legacy separate-scale tensor is unused on the fp8 inline path; keep
  // 0 unless an explicit q_scale is provided.
  if (q_scale.has_value()) {
    mla_params.q_scale_stride_0 = static_cast<int>(q_scale->stride(0));
    mla_params.q_scale_stride_1 = static_cast<int>(q_scale->stride(1));
  } else {
    mla_params.q_scale_stride_0 = 0;
    mla_params.q_scale_stride_1 = 0;
  }
  mla_params.num_tokens = num_tokens;
  mla_params.num_heads = num_heads;

  // --- Pointer locals used by CALL macro ---
  void* q_weight_ptr   = has_q_weight ? q_weight->data_ptr()   : nullptr;
  void* q_scale_ptr    = (q_is_fp8 && q_scale.has_value()) ? q_scale->data_ptr() : nullptr;
  void* q_rope_out_ptr = has_q_rope   ? q_rope_buff->data_ptr() : nullptr;

  int num_CUs;
  hipDeviceGetAttribute(&num_CUs, hipDeviceAttributeMultiprocessorCount, kv.device_id);
  // Device wavefront size (64 GFX9 / 32 else), queried at runtime for the block dim so the
  // launch matches the in-kernel wave_id = threadIdx.x / WARP_SIZE decomposition on wave32 too.
  const int warp_size = static_cast<int>(get_warp_size_func());

  // ---------------------------------------------------------------------------
  // Launch-config heuristic: pick Q heads-per-wave (HPW) from the prefill size.
  // grid.y = 1 (single K wave) + ceil(num_heads / HPW); the kernel is fully runtime-
  // driven by gridDim.y so HPW costs no template re-instantiation. All constants
  // are tuned (MI3xx, measured) and hardcoded -- intentionally NOT env-tunable.
  //
  // Four tiers, by prefill block count (= ceil(T/4) * (1 + q_waves_med)):
  //   decode : tiny T,  tokens_per_block=1, HPW=1   (max blocks to fill the CUs)
  //   med    : mid T,                       HPW=3   (~8-11% better than 4 here;
  //            the prefill mid-range is occupancy/latency-bound, so more, smaller
  //            blocks fill the CUs better)
  //   large  : T ~ 2k-4k,                   HPW=8
  //   xlarge : T >= ~8k,                    HPW=16  (largest prefill chunks, e.g.
  //            ATOM's 16384; HPW=16 is ~3-5% faster than 8 at T=8192/16384 for
  //            H=64/128, while T<=4096 stays on the large tier at 8).
  // Thresholds are in blocks/CU, so they scale with H (larger H reaches a tier at
  // smaller T) and with the device CU count -- matching the measured per-H crossovers.
  // ---------------------------------------------------------------------------
  constexpr int PREFILL_TOKENS_PER_BLOCK       = 4;
  constexpr int PREFILL_Q_HEADS_PER_WAVE_MED   = 3;
  constexpr int PREFILL_Q_HEADS_PER_WAVE_LRG   = 8;
  constexpr int PREFILL_Q_HEADS_PER_WAVE_XLRG  = 16;

  const int prefill_q_waves_med = (num_heads + PREFILL_Q_HEADS_PER_WAVE_MED - 1) / PREFILL_Q_HEADS_PER_WAVE_MED;
  const int prefill_blocks_med = ((num_tokens + PREFILL_TOKENS_PER_BLOCK - 1) / PREFILL_TOKENS_PER_BLOCK)
                                 * (1 + prefill_q_waves_med);

  constexpr int MIN_OVERSUBSCRIPTION     = 4;    // decode -> med
  constexpr int LARGE_PREFILL_THRESHOLD  = 48;   // med    -> large  (blocks/CU)
  constexpr int XLARGE_PREFILL_THRESHOLD = 300;  // large  -> xlarge (blocks/CU)

  const bool use_decode_path    = (prefill_blocks_med < MIN_OVERSUBSCRIPTION * num_CUs);
  const bool use_xlarge_prefill = !use_decode_path
                                  && (prefill_blocks_med > XLARGE_PREFILL_THRESHOLD * num_CUs);
  const bool use_large_prefill  = !use_decode_path && !use_xlarge_prefill
                                  && (prefill_blocks_med > LARGE_PREFILL_THRESHOLD * num_CUs);

  int q_heads_per_wave;
  if (use_decode_path) {
    q_heads_per_wave = 1;
  } else if (use_xlarge_prefill) {
    q_heads_per_wave = PREFILL_Q_HEADS_PER_WAVE_XLRG;
  } else if (use_large_prefill) {
    q_heads_per_wave = PREFILL_Q_HEADS_PER_WAVE_LRG;
  } else {
    q_heads_per_wave = PREFILL_Q_HEADS_PER_WAVE_MED;
  }
  const int num_q_waves = (num_heads + q_heads_per_wave - 1) / q_heads_per_wave;

  AITER_CHECK(use_optimized,
              "fused_qk_norm_rope_group_quant currently only supports "
              "head_dim<=512 and rot_dim=64. Got head_dim=", head_dim,
              " and rot_dim=", rot_dim);

  AITER_CHECK(head_dim == 512,
              "Unsupported head_dim=", head_dim, ". Supported: 512");

  // 4-level dispatch (HEAD_DIM, Q_GROUP_SIZE, Q_SCALE_FP32, HAS_Q_WEIGHT) collapsed into
  // a single generic lambda. The kernel templates instantiate one .co per combination.
  //   - 1 head_dim x 3 group sizes x 2 scale dtypes x 2 q_weight flags x 2 tokens_per_block x
  //     4 dtype combos = 96 instantiations per source dtype (bf16 typical → 96 ko).
  // Q_GROUP_SIZE / Q_SCALE_FP32 are only meaningful when q_out is fp8 (q_dt != kAuto);
  // for bf16 q_out we collapse onto (G=64, e8m0) — the kernel ignores them.
  // Fine-grained (FlyDSL-style) path: 1 wave per (token, head), block=64,
  // grid=(num_heads+1, num_tokens). Measured on gfx950 (MI355): ~5-17% faster
  // than the coarse HPW path at large prefill (T >= ~8k) for both bf16 and fp8 Q.
  // At mid T (256-2048) the coarse path's per-wave head aggregation (one cos/sin
  // gather reused across HPW heads) wins, so we only switch to FG for the xlarge
  // tier. grid.y == num_tokens, so cap at 65535 (larger T would need a Y-chunk loop).
  const bool use_finegrained = use_xlarge_prefill && (num_tokens <= 65535);
  auto launch_all = [&](auto group_size_tag, auto scale_fp32_tag, auto has_qw_tag) {
    constexpr int  head_dim_val      = 512;
    constexpr int  q_group_size_val  = decltype(group_size_tag)::value;
    constexpr bool q_scale_fp32_val  = decltype(scale_fp32_tag)::value;
    constexpr bool has_q_weight_val  = decltype(has_qw_tag)::value;
    if (use_finegrained) {
      // One (token, head) per block: grid.x = head (fast: 0=K, 1.. = Q), grid.y = token.
      // (num_tokens <= 65535 for V4 prefill chunks; larger T would need a Y-chunk loop
      // like flydsl's MAX_GRID_Y.) NB: packing multiple heads/token into one block
      // (HEADS_PER_BLOCK>1, via the kernel's TOKENS_PER_BLOCK slot) was measured to make
      // NO difference (1/2/4/8 identical) -- the MC already coalesces across co-resident
      // waves, so we keep the single-head block (no extra instantiations).
      constexpr int tokens_per_block_val = 1;  // == HEADS_PER_BLOCK for the FG kernel
      dim3 grid(static_cast<unsigned>(1 + num_heads), static_cast<unsigned>(num_tokens));
      dim3 block(static_cast<unsigned>(tokens_per_block_val * warp_size));
      DISPATCH_BY_KV_CACHE_QUERY_DTYPE_OPUS_rmTorch(kv.dtype(), kv_cache_dtype, q_out_type,
                                        CALL_FUSED_QK_NORM_ROPE_FINEGRAINED);
    } else if (use_decode_path) {
      constexpr int tokens_per_block_val = 1;
      dim3 grid((num_tokens + tokens_per_block_val - 1) / tokens_per_block_val, 1 + num_q_waves);
      dim3 block(tokens_per_block_val * warp_size);
      DISPATCH_BY_KV_CACHE_QUERY_DTYPE_OPUS_rmTorch(kv.dtype(), kv_cache_dtype, q_out_type,
                                        CALL_FUSED_QK_NORM_ROPE_GROUP_QUANT_CACHE);
    } else {
      constexpr int tokens_per_block_val = 4;
      dim3 grid((num_tokens + tokens_per_block_val - 1) / tokens_per_block_val, 1 + num_q_waves);
      dim3 block(tokens_per_block_val * warp_size);
      DISPATCH_BY_KV_CACHE_QUERY_DTYPE_OPUS_rmTorch(kv.dtype(), kv_cache_dtype, q_out_type,
                                        CALL_FUSED_QK_NORM_ROPE_GROUP_QUANT_CACHE);
    }
  };

  // Runtime -> compile-time dispatch: map the runtime quant config onto the matching
  // template instantiation. group_size / scale_is_fp32 are shared by Q and K. For bf16
  // q_out the quant params are ignored, so collapse onto (group=64, e8m0) to avoid
  // instantiating unused variants.
  const int  group_size    = q_is_fp8 ? static_cast<int>(quant_group_size) : 64;
  const bool scale_is_fp32 = q_is_fp8 ? (scale_dtype == "fp32") : false;

  // For a fixed (group_size, scale_fp32), branch on q_weight presence (compile-time flag)
  // and launch.
#define LAUNCH_FOR_CONFIG(GROUP_SIZE, SCALE_FP32)                                              \
    do {                                                                                        \
      if (has_q_weight)                                                                         \
        launch_all(std::integral_constant<int, (GROUP_SIZE)>{},                                 \
                   std::integral_constant<bool, (SCALE_FP32)>{}, std::true_type{});             \
      else                                                                                      \
        launch_all(std::integral_constant<int, (GROUP_SIZE)>{},                                 \
                   std::integral_constant<bool, (SCALE_FP32)>{}, std::false_type{});            \
    } while (0)

  if (group_size == 32) {
    if (scale_is_fp32) LAUNCH_FOR_CONFIG(32, true);  else LAUNCH_FOR_CONFIG(32, false);
  } else if (group_size == 64) {
    if (scale_is_fp32) LAUNCH_FOR_CONFIG(64, true);  else LAUNCH_FOR_CONFIG(64, false);
  } else if (group_size == 128) {
    if (scale_is_fp32) LAUNCH_FOR_CONFIG(128, true); else LAUNCH_FOR_CONFIG(128, false);
  } else {
    AITER_CHECK(false, "Unsupported quant_group_size=", group_size);
  }
#undef LAUNCH_FOR_CONFIG
}

// ============================================================================
// K-only kernel for V4-Pro Attention.forward (path A)
// ----------------------------------------------------------------------------
// Reuses k_wave_norm_rope_group_quant_impl (the shared K-wave body, also
// called from the K branch of fuse_qk_norm_rope_group_quant_cache_kernel_impl)
// so the algorithm/output layout stays bit-exact with the QK kernel. The only
// new code here is the launch decomposition (no Q grid.y, no Q register
// pressure) and the per-wave (token_idx, tid, cos/sin) prelude.
//
// Output layout (V4 nm asm sparse-attn reader, same as the K-side of QK):
//   kv_cache [num_tokens, NK=1, head_dim]        fp8:
//     [0       : nope_dim)               nope fp8                       (e.g. 448 B)
//     [nope_dim: nope_dim+2*nGroups)     e8m0 scale, each tile-scale x2 (e.g. 14 B)
//     [nope_dim+2*nGroups: head_dim)     pad, zero-initialised by caller
//   k_pe_out [num_tokens, NK=1, pe_dim]  bf16:    rotated K-PE (NOT quantized)
//
// Caller-side semantic mapping to V4-Pro's model.py Attention.forward (L682-686):
//   kv = self.wkv(x)                                                   -> kv arg
//   kv = self.kv_norm(kv)                                              -> RMSNorm + k_weight
//   apply_rotary_emb(kv[..., -rd:], freqs_cis)                         -> RoPE on PE
//   act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, inplace=True) -> nope fp8 + e8m0
// ============================================================================
template <typename scalar_t, typename cache_t, vllm::Fp8KVCacheDataType kv_dt,
          bool is_neox, int HEAD_DIM = 512, int PE_DIM = 64,
          int GROUP_SIZE = 64, int TOKENS_PER_BLOCK = 1>
__device__ void fuse_kv_norm_rope_group_quant_cache_kernel_impl(
    const scalar_t* __restrict__ kv,           // [num_tokens, (NK=1,) head_dim]
    scalar_t*       __restrict__ k_rope_buff,  // paged rope cache [num_blocks, page_size, pe_dim] bf16 (MQA)
    const scalar_t* __restrict__ k_weight,     // [head_dim]
    cache_t*        __restrict__ k_nope_scale_buff,  // paged nope+scale cache [num_blocks, page_size, head_dim] fp8 (MQA)
    const int64_t*  __restrict__ positions,    // [num_tokens]
    const int64_t*  __restrict__ slot_mapping, // [num_tokens] flat slot = block*page_size + offset
    const scalar_t* __restrict__ cos_cache,    // [max_position, pe_dim/2]
    const scalar_t* __restrict__ sin_cache,    // [max_position, pe_dim/2]
    float eps,
    const MlaKernelParams& __restrict__ params)
{
  // Wave-level indexing: each wave handles one token; no Q grid.y dimension.
  const uint32_t wave_id = threadIdx.x / WARP_SIZE;
  const int32_t  tid     = static_cast<int32_t>(threadIdx.x % WARP_SIZE);
  const int32_t  token_idx =
      static_cast<int32_t>(blockIdx.x) * TOKENS_PER_BLOCK + wave_id;
  if (token_idx >= params.num_tokens) return;

  // Paged destination: slot_mapping[token] is the flat slot into the paged
  // cache; split into (block, offset) so a padded block stride (kcache_block_stride
  // != page_size*row_stride) is honored. A negative slot marks a skipped token.
  const int64_t slot = slot_mapping[token_idx];
  if (slot < 0) return;
  const int64_t block  = slot / params.page_size;
  const int64_t offset = slot % params.page_size;
  const int64_t out_cache_offset =
      block * params.kcache_block_stride + offset * params.kcache_row_stride;
  const int64_t out_rope_offset =
      block * params.krope_block_stride + offset * params.krope_row_stride;

  // RoPE cos/sin pointers (per-token, shared by the wave). Tables are laid out
  // [max_pos, PE_DIM/2] (GPT-J reuse-front-part style), so each token-row is
  // (PE_DIM/2) elements wide -- match that exactly here regardless of PE_DIM.
  const int32_t cos_sin_offset =
      static_cast<int32_t>(positions[token_idx]) * (PE_DIM >> 1);
  const scalar_t* cos_ptr = cos_cache + cos_sin_offset;
  const scalar_t* sin_ptr = sin_cache + cos_sin_offset;

  k_wave_norm_rope_group_quant_impl<scalar_t, cache_t, kv_dt, is_neox,
                                    HEAD_DIM, PE_DIM, GROUP_SIZE>(
      kv, k_nope_scale_buff, k_rope_buff, k_weight, cos_ptr, sin_ptr, eps, params,
      token_idx, tid, out_cache_offset, out_rope_offset);
}

// __global__ wrapper: dispatches the runtime is_neox flag to a compile-time template arg.
// __launch_bounds__ uses the literal wave64 width as the MAX block size; on wave32 the
// host launches the smaller TOKENS_PER_BLOCK*32 block (still within [1, TPB*64]), so the
// bound holds for both. Literal because WARP_SIZE is not an ICE in the host compile pass.
template <typename scalar_t, typename cache_t, vllm::Fp8KVCacheDataType kv_dt,
          int HEAD_DIM = 512, int PE_DIM = 64, int GROUP_SIZE = 64,
          int TOKENS_PER_BLOCK = 1>
__global__ __launch_bounds__(TOKENS_PER_BLOCK * 64, 512 / (TOKENS_PER_BLOCK * 64))
void fuse_kv_norm_rope_group_quant_cache_kernel(
    const scalar_t* __restrict__ kv,
    scalar_t*       __restrict__ k_rope_buff,
    const scalar_t* __restrict__ k_weight,
    cache_t*        __restrict__ k_nope_scale_buff,
    const int64_t*  __restrict__ positions,
    const int64_t*  __restrict__ slot_mapping,
    const scalar_t* __restrict__ cos_cache,
    const scalar_t* __restrict__ sin_cache,
    float eps,
    const MlaKernelParams params,
    bool is_neox)
{
  if (is_neox) {
    fuse_kv_norm_rope_group_quant_cache_kernel_impl<scalar_t, cache_t, kv_dt, true,
                                                    HEAD_DIM, PE_DIM, GROUP_SIZE,
                                                    TOKENS_PER_BLOCK>(
        kv, k_rope_buff, k_weight, k_nope_scale_buff, positions, slot_mapping,
        cos_cache, sin_cache, eps, params);
  } else {
    fuse_kv_norm_rope_group_quant_cache_kernel_impl<scalar_t, cache_t, kv_dt, false,
                                                    HEAD_DIM, PE_DIM, GROUP_SIZE,
                                                    TOKENS_PER_BLOCK>(
        kv, k_rope_buff, k_weight, k_nope_scale_buff, positions, slot_mapping,
        cos_cache, sin_cache, eps, params);
  }
}

} // namespace aiter

// Launcher macro: shared by all KV-dtype dispatch arms. Mirrors the QK
// CALL_FUSED_QK_NORM_ROPE_GROUP_QUANT_CACHE macro shape (locals required at
// callsite: head_dim_val, pe_dim_val, group_size_val, tokens_per_block_val,
// grid, block, stream).
#define CALL_FUSED_KV_NORM_ROPE_GROUP_QUANT_CACHE(KV_T, CACHE_T, KV_DTYPE)                       \
        aiter::fuse_kv_norm_rope_group_quant_cache_kernel<KV_T, CACHE_T, KV_DTYPE,               \
                head_dim_val, pe_dim_val, group_size_val, tokens_per_block_val>                  \
              <<<grid, block, 0, stream>>>(                                                      \
                reinterpret_cast<const KV_T*>(kv.data_ptr()),                                    \
                reinterpret_cast<KV_T*>(k_rope_buff.data_ptr()),                                 \
                reinterpret_cast<const KV_T*>(k_weight.data_ptr()),                              \
                reinterpret_cast<CACHE_T*>(k_nope_scale_buff.data_ptr()),                        \
                reinterpret_cast<const int64_t*>(positions.data_ptr()),                          \
                reinterpret_cast<const int64_t*>(slot_mapping.data_ptr()),                       \
                reinterpret_cast<const KV_T*>(cos_cache.data_ptr()),                             \
                reinterpret_cast<const KV_T*>(sin_cache.data_ptr()),                             \
                static_cast<float>(eps),                                                         \
                mla_params,                                                                      \
                is_neox);

namespace aiter {

// Supported (head_dim, pe_dim, group_size) triples for the K-only kernel.
// vec_size_i = HEAD_DIM/WARP_SIZE elements/lane (8 wave64, 16 wave32; 4 for fp32).
// Constraints (enforced as static_asserts inside the K helper):
//   - head_dim % vec_size_i == 0  and  head_dim / vec_size_i <= WARP_SIZE
//   - pe_dim < head_dim and pe_dim % vec_size_i == 0
//   - (head_dim - pe_dim) % group_size == 0
//   - group_size / vec_size_i is a power-of-2 in [1, 64]
//
// We only instantiate combos that are actually used: V4-Pro (512/64/64) plus
// a handful of widely-used MLA / GQA shapes. Adding a new combo is one line
// in the dispatch ladder below.
#define KV_K_ONLY_DISPATCH_TABLE(X) \
    /* X(head_dim, pe_dim, group_size) -- all combos shipped today      */    \
    X(512, 64, 64)   /* DeepSeek V4-Pro (default)                       */    \
    X(192, 64, 64)   /* DeepSeek V2 / V3 MLA, default group             */    \
    X(384, 128, 64)  /* head_dim=384, rope=128 (Qwen-style)             */    


void fused_kv_norm_rope_group_quant(
    aiter_tensor_t& kv,                 // [num_tokens, (NK=1,) head_dim]
    aiter_tensor_t& k_rope_buff,        // paged rope cache [num_blocks, page_size, rot_dim] bf16 (MQA)
    aiter_tensor_t& k_weight,           // [head_dim] RMSNorm weights
    aiter_tensor_t& k_nope_scale_buff,  // paged nope+scale cache [num_blocks, page_size, head_dim] fp8 (MQA)
    aiter_tensor_t& positions,          // [num_tokens]
    aiter_tensor_t& slot_mapping,       // [num_tokens] int64 flat slot = block*page_size + offset
    aiter_tensor_t& cos_cache,          // [max_position, rot_dim//2]
    aiter_tensor_t& sin_cache,          // [max_position, rot_dim//2]
    double eps,
    bool is_neox,
    int64_t quant_group_size,
    const std::string& scale_dtype)
{
  // V4-Pro Attention.forward path A: KV-only fused RMSNorm + RoPE + 1xG e8m0
  // group-quant, scattered into a PAGED KV cache via slot_mapping. NK is
  // hardcoded to 1 (MLA latent KV).

  const int num_tokens = kv.size(0);
  const int head_dim   = kv.size(-1);
  const int rot_dim    = cos_cache.size(-1) * 2;
  const int group_size = static_cast<int>(quant_group_size);

  AITER_CHECK(k_weight.size(0) == head_dim, "k_weight size must match head_dim");
  AITER_CHECK(kv.stride(-1) == 1, "kv stride(-1) must be equal to 1");
  AITER_CHECK(scale_dtype == "e8m0",
              "fused_kv_norm_rope_group_quant currently only supports scale_dtype='e8m0', got ",
              scale_dtype);
  AITER_CHECK(slot_mapping.size(0) == num_tokens,
              "slot_mapping must have num_tokens entries; got ", slot_mapping.size(0),
              " vs num_tokens=", num_tokens);
  // Paged caches: [num_blocks, page_size, entry] (MQA, no NK dim). page_size = dim 1; both
  // caches must share it (indexed by the same slot). Row stride = stride(1).
  AITER_CHECK(k_nope_scale_buff.dim() >= 2 && k_rope_buff.dim() >= 2,
              "paged caches must be at least [num_blocks, page_size, ...]");
  const int page_size = static_cast<int>(k_nope_scale_buff.size(1));
  AITER_CHECK(k_rope_buff.size(1) == page_size,
              "k_rope_buff page_size (", k_rope_buff.size(1),
              ") must match k_nope_scale_buff page_size (", page_size, ")");

  const int num_kv_heads = (kv.dim() == 3) ? static_cast<int>(kv.size(1)) : 1;
  AITER_CHECK(num_kv_heads == 1,
              "fused_kv_norm_rope_group_quant requires num_kv_heads == 1 (MQA), got ",
              num_kv_heads);
  AITER_CHECK(rot_dim < head_dim && rot_dim > 0,
              "rot_dim must be in (0, head_dim); got rot_dim=", rot_dim, " head_dim=", head_dim);
  const int nope_dim = head_dim - rot_dim;
  AITER_CHECK(nope_dim % group_size == 0,
              "(head_dim - rot_dim) must be divisible by quant_group_size; got nope_dim=",
              nope_dim, " group_size=", group_size);

  // Strides. The K-only paged path uses kv_stride_* (input) + the paged-cache
  // block/row strides below; the rest of MlaKernelParams is zero-filled (kept
  // for struct-layout compat with the QK kernel).
  aiter::MlaKernelParams mla_params{};
  mla_params.kv_stride_0         = kv.stride(0);
  mla_params.kv_stride_1         = (kv.dim() == 3) ? kv.stride(1) : 0;
  mla_params.num_tokens          = num_tokens;
  mla_params.num_heads           = 0;  // K-only kernel: no Q heads
  mla_params.page_size           = page_size;
  mla_params.kcache_block_stride = k_nope_scale_buff.stride(0);
  mla_params.kcache_row_stride   = k_nope_scale_buff.stride(1);
  mla_params.krope_block_stride  = k_rope_buff.stride(0);
  mla_params.krope_row_stride    = k_rope_buff.stride(1);

  HipDeviceGuard device_guard(kv.device_id);
  const hipStream_t stream = aiter::getCurrentHIPStream();

  // KV cache dtype dispatch (auto / fp8). Same map as the QK entry.
  std::string kv_cache_dtype = "auto";
  const AiterDtype cache_dt = k_nope_scale_buff.dtype();
  if (cache_dt == AITER_DTYPE_fp32 ||
      cache_dt == AITER_DTYPE_fp16 ||
      cache_dt == AITER_DTYPE_bf16) {
    kv_cache_dtype = "auto";
  } else if (cache_dt == AITER_DTYPE_fp8) {
    kv_cache_dtype = "fp8";
  } else {
    AITER_CHECK(false, "kv cache data type is not supported: ",
                AiterDtype_to_str(cache_dt));
  }

  // Launch-config heuristic: TOKENS_PER_BLOCK=4 for prefill (matches the QK
  // coarse kernel), TOKENS_PER_BLOCK=1 for very small T (decode-style). A
  // single launch per token is enough since there is no Q wave to amortize.
  //
  // NOTE: a TPB sweep {1,2,4,8} x T {1k,4k,16k} on MI355 (rocprofv3 kernel
  // time) showed NO measurable effect -- all TPB land within ~3-5% of each
  // other and the ranking flips run-to-run (pure shared-box noise). The kernel
  // is HBM-bandwidth bound and occupancy-saturated (~8 waves/CU) at every TPB,
  // since __launch_bounds__'s min-blocks arg (512/(TPB*64)) scales inversely
  // with block size. So this stays at the simple decode=1 / prefill=4 split;
  // there is nothing to gain from a finer launch heuristic here.
  int num_CUs;
  hipDeviceGetAttribute(&num_CUs, hipDeviceAttributeMultiprocessorCount, kv.device_id);
  // Device wavefront size (64 GFX9 / 32 else), queried at runtime for the block dim.
  const int warp_size = static_cast<int>(get_warp_size_func());
  constexpr int PREFILL_TOKENS_PER_BLOCK = 4;
  constexpr int MIN_OVERSUBSCRIPTION     = 4;
  const int prefill_blocks =
      (num_tokens + PREFILL_TOKENS_PER_BLOCK - 1) / PREFILL_TOKENS_PER_BLOCK;
  const bool use_decode_path = (prefill_blocks < MIN_OVERSUBSCRIPTION * num_CUs);

  // Lambda factors out the (kv_dtype, kv_cache_dtype, tokens_per_block)
  // dispatch from the (head_dim, pe_dim, group_size) compile-time triple.
  // Captures `kv`, `kv_cache`, etc. for the launch macro. Each call to this
  // lambda triggers ONE template instantiation per (head_dim, pe_dim,
  // group_size, kv_dtype, cache_dtype, TPB, is_neox) -- the inner DISPATCH
  // expands the dtype combos.
  auto launch_for_shape = [&](auto head_dim_tag, auto pe_dim_tag, auto group_size_tag) {
    constexpr int head_dim_val   = decltype(head_dim_tag)::value;
    constexpr int pe_dim_val     = decltype(pe_dim_tag)::value;
    constexpr int group_size_val = decltype(group_size_tag)::value;
    // Block dim = tokens_per_block * (runtime) warp_size, matching the in-kernel
    // wave_id = threadIdx.x / WARP_SIZE decomposition on both wave64 and wave32.
    if (use_decode_path) {
      constexpr int tokens_per_block_val = 1;
      dim3 grid((num_tokens + tokens_per_block_val - 1) / tokens_per_block_val);
      dim3 block(tokens_per_block_val * warp_size);
      DISPATCH_BY_KV_CACHE_DTYPE_OPUS_rmTorch(kv.dtype(), kv_cache_dtype,
                                              CALL_FUSED_KV_NORM_ROPE_GROUP_QUANT_CACHE);
    } else {
      constexpr int tokens_per_block_val = PREFILL_TOKENS_PER_BLOCK;
      dim3 grid((num_tokens + tokens_per_block_val - 1) / tokens_per_block_val);
      dim3 block(tokens_per_block_val * warp_size);
      DISPATCH_BY_KV_CACHE_DTYPE_OPUS_rmTorch(kv.dtype(), kv_cache_dtype,
                                              CALL_FUSED_KV_NORM_ROPE_GROUP_QUANT_CACHE);
    }
  };

  // Runtime -> compile-time dispatch on (head_dim, rot_dim, group_size).
  // Adding a new shape: append one X(...) entry to KV_K_ONLY_DISPATCH_TABLE.
  bool dispatched = false;
  #define DISPATCH_ONE(HD, PD, GS)                                             \
    if (!dispatched && head_dim == (HD) && rot_dim == (PD) && group_size == (GS)) { \
      launch_for_shape(std::integral_constant<int, (HD)>{},                    \
                       std::integral_constant<int, (PD)>{},                    \
                       std::integral_constant<int, (GS)>{});                   \
      dispatched = true;                                                       \
    }
  KV_K_ONLY_DISPATCH_TABLE(DISPATCH_ONE)
  #undef DISPATCH_ONE

  AITER_CHECK(dispatched,
              "fused_kv_norm_rope_group_quant: unsupported shape (head_dim=",
              head_dim, ", rot_dim=", rot_dim, ", group_size=", group_size,
              "). See KV_K_ONLY_DISPATCH_TABLE in fused_qk_norm_rope_cache_quant.cu "
              "for the supported set; add a new X(...) entry to extend it.");
}

} // namespace aiter
