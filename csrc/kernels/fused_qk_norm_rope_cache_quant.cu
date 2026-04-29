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
#include <type_traits>

#include "aiter_hip_common.h"
#include "hip_reduce.h"
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

 // Minimum absmax used when computing FP8 KV scales to avoid division by zero when
 // activations are all zero (e.g. CUDA graph warmup, invalid slots, or padding).
 static constexpr float kFp8KvQuantAbsmaxFloorF32 = 1e-8f;
 
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
             float const warp_max_safe = fmaxf(warp_max, kFp8KvQuantAbsmaxFloorF32);
             k_scale_val                 = warp_max_safe / dtype_max;
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
             v_scale_val                 = warp_max_safe / dtype_max;
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
    scalar_t* qkv_void,            // Combined QKV tensor [num_tokens, (num_heads_q+num_heads_k+num_heads_v), head_dim]
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
    int64_t* slot_mapping,  // Slot mapping
    int64_t const* cu_q_len, // Cu Q len tensor [0, batch0_seq_len, batch0_seq_len + batch1_seq_len, ...]
    float* k_scale,        // Key scale for quantized key cache [num_blocks, num_heads_k]
    float* v_scale,        // Value scale for quantized value cache [num_blocks, num_heads_v]
    int const num_tokens,  // Number of tokens
    int const page_size,   // Page size for kv cache
    int const batch_size,  // Batch size
    int const blocks_per_batch // Uniform blocks per batch (>0: division mapping, 0: prefix-sum fallback)
)
{
    int const num_heads        = num_heads_q + num_heads_k + num_heads_v;
    int const localHeadIdx     = blockIdx.z;
    int const page_size_log2   = __builtin_ctz(page_size);
    int const page_mask        = page_size - 1;

    int batch_id = -1;
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
        batch_id = 0;
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
        int64_t prev_slot_idx = slot_mapping[prev_first_token_idx];
        int64_t preTg_block_idx = prev_slot_idx >> page_size_log2;
        int64_t last_token_idx = batch_end_idx - 1;
        slot_idx = slot_mapping[last_token_idx];
        block_idx = slot_idx >> page_size_log2;
        if(preTg_block_idx == block_idx)
        {
            return;
        }
        block_offset = slot_idx & page_mask;
    }
    else
    {
        slot_idx = slot_mapping[first_token_idx];
        block_idx = slot_idx >> page_size_log2;
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
            int64_t token_idx  = first_token_idx - (threadIdx.x + 1);
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
        block_offset    = slot_idx & page_mask;
    }
    // Each token should compute its own slot_id and block_offset
    int64_t actual_slot_id = -1;
    int64_t actual_block_offset = 0;
    int64_t actual_block_idx = -1;
    // Calculate the num_tokens that are in the same cache block (page)
    int tokens_in_block = 0;
    if(first_token_idx + threadIdx.x < batch_end_idx)
    {
        actual_slot_id = slot_mapping[first_token_idx + threadIdx.x];
        if(actual_slot_id >= 0)
        {
            actual_block_idx = actual_slot_id >> page_size_log2;
            actual_block_offset = actual_slot_id & page_mask;
            tokens_in_block = (actual_block_idx == block_idx) ? 1 : 0;
        }
    }
    auto sum               = [](float a, float b) { return a + b; };
    int numtokens_in_block = 0;
    numtokens_in_block = block_reduce<float, decltype(sum), wg_size, true>(static_cast<float>(tokens_in_block), sum);
    // Calculate tokenIdx for current thread
    int tokenIdx = first_token_idx + threadIdx.x;
    bool const isQ                  = localHeadIdx < num_heads_q;
    bool const isK                  = (localHeadIdx < num_heads_q + num_heads_k) & !isQ;
    bool const isV                  = !isQ & !isK;
    int const headIdx               = isV   ? localHeadIdx - num_heads_q - num_heads_k
                                     : isK ? localHeadIdx - num_heads_q
                                           : localHeadIdx;
    constexpr int numElemsPerThread = head_dim;
    constexpr int best_vec_size = sizeof(float4) / sizeof(scalar_t);
    constexpr int vec_size      = std::min(best_vec_size, numElemsPerThread);
    constexpr int load_loop_cnt = numElemsPerThread / vec_size;
    using ltype                 = ::vec_t<scalar_t, vec_size>;
    using kv_cache_ltype        = ::vec_t<kv_cache_scalar_t, vec_size>;
    ltype elements;
    ltype next_elements;
    float block_max = 0.0f;
    auto cur_element_offset = head_dim * threadIdx.x;
    auto f_absmax_f32 = [](float v_0_, float v_1_) {
        return __builtin_fmaxf(abs(v_0_), abs(v_1_));
    };
    // V: only valid tokens; Q/K: ALL threads must participate (avoids __syncthreads deadlock in block_reduce)
    if(isV)
    {
        int64_t total_elements = numtokens_in_block * head_dim;
        for(int idx = threadIdx.x; idx < total_elements / vec_size; idx += blockDim.x)
        {
            int token_idx          = first_token_idx + idx * vec_size / head_dim;
            int64_t offsetWarp = (token_idx * num_heads * head_dim + localHeadIdx * head_dim) /
                                vec_size;
            int vec_slot = idx % (head_dim / vec_size);
            elements = reinterpret_cast<ltype*>(qkv_void)[offsetWarp + vec_slot];
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
            int64_t total_elements = numtokens_in_block * head_dim;
            auto sum_op = [](float a, float b) { return a + b; };
            if constexpr(interleave) {
                for(int idx = threadIdx.x; idx < total_elements / vec_size; idx += blockDim.x)
                {
                    int token_local = idx / head_thread;
                    int vec_slot    = idx % head_thread;
                    int token_idx   = first_token_idx + token_local;
                    if(token_idx >= batch_end_idx) continue;
                    int64_t offsetWarp = (token_idx * num_heads * head_dim + localHeadIdx * head_dim) / vec_size;
                    elements = reinterpret_cast<ltype*>(qkv_void)[offsetWarp + vec_slot];
                    ltype weights;
                    scalar_t const* weight_ptr = isQ ? q_weight : k_weight;
                    weights = reinterpret_cast<const ltype*>(weight_ptr)[vec_slot];
                    float partial_sum = 0.0f;
                    #pragma unroll
                    for(int j = 0; j < vec_size; j++)
                        partial_sum += static_cast<float>(elements[j]) * static_cast<float>(elements[j]);
                    float sumOfSquares = wave_reduce<float, decltype(sum_op), head_thread, true>(partial_sum, sum_op);
                    float rms_rcp = rsqrtf(sumOfSquares / static_cast<float>(head_dim) + eps);
                    int64_t pos_id  = position_ids[token_idx];
                    scalar_t const* cache_ptr = cos_sin_cache + pos_id * head_dim;
                    scalar_t const* cos_ptr  = cache_ptr;
                    scalar_t const* sin_ptr  = cache_ptr + head_dim / 2;
                    int const base_idx = vec_slot * vec_size;
                    
                    using cos_sin_ltype = ::vec_t<scalar_t, vec_size/2>;
                    cos_sin_ltype cos;
                    cos = reinterpret_cast<const cos_sin_ltype*>(cos_ptr)[vec_slot];
                    cos_sin_ltype sin;
                    sin = reinterpret_cast<const cos_sin_ltype*>(sin_ptr)[vec_slot];
                    #pragma unroll
                    for(int k = 0; k < vec_size; k += 2)
                    {
                        int const local0   = base_idx + k;
                        int const local1   = base_idx + k + 1;
                        float weight0 = static_cast<float>(weights[k]);
                        float weight1 = static_cast<float>(weights[k + 1]);
                        int const half_dim = local0 / 2;
                        float cos_val = static_cast<float>(cos[k/2]);
                        float sin_val = static_cast<float>(sin[k/2]);
                        float const val0  = static_cast<float>(elements[k]) * rms_rcp * weight0;
                        float const val1  = static_cast<float>(elements[k + 1]) * rms_rcp * weight1;
                        elements[k]       = static_cast<scalar_t>(val0 * cos_val - val1 * sin_val);
                        elements[k + 1]   = static_cast<scalar_t>(val0 * sin_val + val1 * cos_val);
                        block_max          = f_absmax_f32(block_max, elements[k]);
                        block_max          = f_absmax_f32(block_max, elements[k + 1]);
                    }
                    reinterpret_cast<ltype*>(qkv_void)[offsetWarp + vec_slot] = elements;
                }
            } else {
                constexpr int64_t head_thread_half = head_dim / vec_size / 2;
                for(int idx = threadIdx.x; idx < total_elements / vec_size; idx += blockDim.x)
                {
                    int token_local = idx / head_thread;
                    int vec_slot    = idx % head_thread;
                    int token_idx   = first_token_idx + token_local;
                    if(token_idx >= batch_end_idx) continue;
                    if(vec_slot >= head_thread_half) continue;
                    int pair_slot   = vec_slot + head_thread_half;
                    int64_t offsetWarp = (token_idx * num_heads * head_dim + localHeadIdx * head_dim) / vec_size;
                    elements      = reinterpret_cast<ltype*>(qkv_void)[offsetWarp + vec_slot];
                    next_elements = reinterpret_cast<ltype*>(qkv_void)[offsetWarp + pair_slot];
                    ltype weights0, weights1;
                    scalar_t const* weight_ptr = isQ ? q_weight : k_weight;
                    weights0 = reinterpret_cast<const ltype*>(weight_ptr)[vec_slot];
                    weights1 = reinterpret_cast<const ltype*>(weight_ptr)[pair_slot];
                    int64_t pos_id = position_ids[token_idx];
                    scalar_t const* cache_ptr = cos_sin_cache + pos_id * head_dim;
                    scalar_t const* cos_ptr  = cache_ptr;
                    scalar_t const* sin_ptr  = cache_ptr + head_dim / 2;
                    float partial_sum = 0.0f;
                    #pragma unroll
                    for(int j = 0; j < vec_size; j++)
                        partial_sum += static_cast<float>(elements[j]) * static_cast<float>(elements[j])
                                     + static_cast<float>(next_elements[j]) * static_cast<float>(next_elements[j]);
                    float sumOfSquares = wave_reduce<float, decltype(sum_op), head_thread_half, true>(partial_sum, sum_op);
                    float rms_rcp = rsqrtf(sumOfSquares / static_cast<float>(head_dim) + eps);
                    using cos_sin_ltype = ::vec_t<scalar_t, vec_size>;
                    cos_sin_ltype cos;
                    cos = reinterpret_cast<const cos_sin_ltype*>(cos_ptr)[vec_slot];
                    cos_sin_ltype sin;
                    sin = reinterpret_cast<const cos_sin_ltype*>(sin_ptr)[vec_slot];
                    #pragma unroll                    
                    for(int j = 0; j < vec_size; j++)
                    {
                        int const idx0 = vec_slot * vec_size + j;
                        int const idx1 = pair_slot * vec_size + j;
                        float weight0 = static_cast<float>(weights0[j]);
                        float weight1 = static_cast<float>(weights1[j]);
                        float cos_val = static_cast<float>(cos[j]);
                        float sin_val = static_cast<float>(sin[j]);
                        float const val0 = static_cast<float>(elements[j]) * rms_rcp * weight0;
                        float const val1 = static_cast<float>(next_elements[j]) * rms_rcp * weight1;
                        float out0 = val0 * cos_val - val1 * sin_val;
                        float out1 = val1 * cos_val + val0 * sin_val;
                        block_max = f_absmax_f32(block_max, out0);
                        block_max = f_absmax_f32(block_max, out1);
                        elements[j]      = static_cast<scalar_t>(out0);
                        next_elements[j] = static_cast<scalar_t>(out1);
                    }
                    reinterpret_cast<ltype*>(qkv_void)[offsetWarp + vec_slot]  = elements;
                    reinterpret_cast<ltype*>(qkv_void)[offsetWarp + pair_slot]  = next_elements;
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
    auto f_max_f32 = [](float v_0_, float v_1_) { return __builtin_fmaxf(v_0_, v_1_); };
    if(kv_dt != vllm::Fp8KVCacheDataType::kAuto)
    {
        block_max = block_reduce<float, decltype(f_max_f32), wg_size, true>(block_max, f_max_f32);
    }
    if(isK)
    {
        float k_scale_val = 1.0f;
        float inv_scale_val = 1.0f;
        if constexpr(kv_dt != vllm::Fp8KVCacheDataType::kAuto)
        {
            float const block_max_safe = fmaxf(block_max, kFp8KvQuantAbsmaxFloorF32);
            k_scale_val                  = block_max_safe / dtype_max;
            inv_scale_val                = dtype_max / block_max_safe;
            int64_t scale_offset = block_idx * num_heads_k + headIdx;
            if(block_offset > 0)
            {
                float k_scale_global = k_scale[scale_offset];
                if(k_scale_global < k_scale_val)
                {
                    // k_cache layout: [num_blocks, num_heads_k, head_size//X, page_size, X]
                    int64_t cache_base = block_idx * page_size * num_heads_k * head_dim +
                                        headIdx * head_dim * page_size;
                    float rescale = k_scale_global * inv_scale_val;
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
                                kv_cache_ltype data = *reinterpret_cast<kv_cache_ltype*>(&k_cache[addr]);
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
        int64_t cache_offset = block_idx * page_size * num_heads_k * head_dim +
                               headIdx * head_dim * page_size;
        int64_t total_elements = numtokens_in_block * head_dim;
        for(int64_t idx = threadIdx.x; idx < total_elements / vec_size; idx += blockDim.x)
        {
            int token_idx          = first_token_idx + idx * vec_size / head_dim;
            int head_offset        = (idx * vec_size) % head_dim;
            int block_offset_local = (token_idx - first_token_idx + block_offset) & page_mask;
            int64_t offsetWarp = (token_idx * num_heads * head_dim + localHeadIdx * head_dim) / vec_size;
            elements = reinterpret_cast<ltype*>(qkv_void)[offsetWarp + head_offset / vec_size];
            int64_t vec_offset = cache_offset + (head_offset / X) * page_size * X + block_offset_local * X + head_offset % X;
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
        float v_scale_val = 1.0f;
        float inv_scale_val = 1.0f;
        if constexpr(kv_dt != vllm::Fp8KVCacheDataType::kAuto)
        {
            float const block_max_safe = fmaxf(block_max, kFp8KvQuantAbsmaxFloorF32);
            v_scale_val                  = block_max_safe / dtype_max;
            inv_scale_val                = dtype_max / block_max_safe;
            int64_t scale_offset = block_idx * num_heads_k + headIdx;
            if(block_offset > 0)
            {
                float v_scale_global = v_scale[scale_offset];
                if(v_scale_global < v_scale_val)
                {
                    // v_cache layout: [num_blocks, num_heads_k, page_size//X, head_size, X]
                    int64_t cache_base = block_idx * page_size * num_heads_v * head_dim +
                                        headIdx * head_dim * page_size;
                    float rescale = v_scale_global * inv_scale_val;
                    constexpr int vecs_per_bh = (X / vec_size) * head_dim;
                    int n_full_blocks   = block_offset / X;
                    int full_vecs       = n_full_blocks * vecs_per_bh;
                    for(int idx = threadIdx.x; idx < full_vecs; idx += blockDim.x)
                    {
                        kv_cache_ltype data =
                            *reinterpret_cast<kv_cache_ltype*>(v_cache + cache_base + idx * vec_size);
                        #pragma unroll
                        for(int j = 0; j < vec_size; j++)
                        {
                            data[j] = opus::cast<kv_cache_scalar_t>(
                                opus::cast<float>(data[j]) * rescale);
                        }
                        *reinterpret_cast<kv_cache_ltype*>(v_cache + cache_base + idx * vec_size) = data;
                    }
                    if((block_offset % X) != 0) {
                        int last_block_divX = (block_offset - 1) / X;
                        int last_x_idx      = (block_offset - 1) % X;
                        int last_full_vec   = (last_x_idx + 1) / vec_size;
                        int partial_vecs   = last_full_vec * head_dim;
                        for(int idx = threadIdx.x; idx < partial_vecs; idx += blockDim.x) {
                            int head_offset = idx / last_full_vec;
                            int vec_chunk   = idx % last_full_vec;
                            int64_t vec_off = cache_base + last_block_divX * head_dim * X +
                                              head_offset * X + vec_chunk * vec_size;
                            kv_cache_ltype data =
                                *reinterpret_cast<kv_cache_ltype*>(&v_cache[vec_off]);
                            #pragma unroll
                            for(int j = 0; j < vec_size; j++) {
                                data[j] = opus::cast<kv_cache_scalar_t>(
                                    opus::cast<float>(data[j]) * rescale);
                            }
                            *reinterpret_cast<kv_cache_ltype*>(&v_cache[vec_off]) = data;
                        }
                        int tail_count = (last_x_idx - last_full_vec * vec_size + 1) * head_dim;
                        for(int idx = threadIdx.x; idx < tail_count; idx += blockDim.x) {
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
        int64_t cache_offset = block_idx * page_size * num_heads_v * head_dim +
                               headIdx * head_dim * page_size;
        int64_t total_elements = numtokens_in_block * head_dim;
        for(int64_t idx = threadIdx.x; idx < total_elements / vec_size; idx += blockDim.x)
        {
            int token_idx          = first_token_idx + idx * vec_size / head_dim;
            int head_offset        = (idx * vec_size) % head_dim;
            int block_offset_local = (token_idx - first_token_idx + block_offset) & page_mask;
            int64_t v_base         = cache_offset + (block_offset_local / X) * head_dim * X + head_offset * X + block_offset_local % X;
            int64_t offsetWarp = (token_idx * num_heads * head_dim + localHeadIdx * head_dim) / vec_size;
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
    int max_tpb = max_tokens_per_batch > 0
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
        gridDimX = batch_size;
        gridSizeY = gridSizeY_decode;
    }
    else if(batch_size > 1)
    {
        // Mode B: uniform division — batch_id = blockIdx.y / blocks_per_batch
        // When max_tokens_per_batch provided: use actual max (exact).
        // When max_tokens_per_batch=0: use num_tokens as conservative upper bound
        // (safe for any distribution; may over-allocate Y-blocks for small batches).
        gridDimX = 1;
        blocks_per_batch_param = max_tokens_per_batch > 0
            ? gridSizeY_decode
            : ((num_tokens + page_size - 1) / page_size + 1);
        gridSizeY = batch_size * blocks_per_batch_param;
    }
    else
    {
        // batch_size <= 1: single batch, batch_id = 0
        gridDimX = 1;
        gridSizeY = (num_tokens + page_size - 1) / page_size + 1;
    }

    dim3 gridDim(gridDimX, gridSizeY, num_heads_q + num_heads_k + num_heads_v);
    dim3 blockDim(blockSize);

#define DISPATCH_X_VALUE(x_val, ...)                                            \
    if(x_val == 16) { constexpr int X_VAL = 16; __VA_ARGS__ }                  \
    else if(x_val == 8) { constexpr int X_VAL = 8; __VA_ARGS__ }               \
    else if(x_val == 4) { constexpr int X_VAL = 4; __VA_ARGS__ }               \
    else { TORCH_CHECK(false, "Unsupported x: ", x_val); }

#define DISPATCH_INTERLEAVE_BQ(interleave, ...)                                 \
    if(interleave) { const bool INTERLEAVE = true; __VA_ARGS__ }                \
    else           { const bool INTERLEAVE = false; __VA_ARGS__ }

#define LAUNCH_BLOCK_QUANT_ARGS                                                 \
                                                       num_heads_q,             \
                                                       num_heads_k,             \
                                                       num_heads_v,             \
                                                       eps,                     \
                                                       q_weight,                \
                                                       k_weight,                \
                                                       cos_sin_cache,           \
                                                       position_ids,            \
                                                       k_cache,                 \
                                                       v_cache,                 \
                                                       slot_mapping,            \
                                                       cu_q_len,                \
                                                       k_scale,                 \
                                                       v_scale,                 \
                                                       num_tokens,              \
                                                       page_size,               \
                                                       batch_size,              \
                                                       blocks_per_batch_param

#define LAUNCH_BLOCK_QUANT_KERNEL(HEAD_DIM, WG_SIZE)                            \
        DISPATCH_INTERLEAVE_BQ(interleave, {                                    \
            DISPATCH_X_VALUE(x, {                                               \
                fusedQKNormRopeBlockQuantCacheShuffleKernel<scalar_t,            \
                    kv_cache_scalar_t, HEAD_DIM, INTERLEAVE,                    \
                    X_VAL, WG_SIZE, kv_dt>                                      \
                    <<<gridDim, blockDim, 0, stream>>>(qkv,                     \
                        LAUNCH_BLOCK_QUANT_ARGS);                               \
            });                                                                 \
        });

#define DISPATCH_BLOCK_SIZE(HEAD_DIM)                                            \
    if(blockSize == 64) { LAUNCH_BLOCK_QUANT_KERNEL(HEAD_DIM, 64) }             \
    else if(blockSize == 128) { LAUNCH_BLOCK_QUANT_KERNEL(HEAD_DIM, 128) }      \
    else if(blockSize == 256) { LAUNCH_BLOCK_QUANT_KERNEL(HEAD_DIM, 256) }      \
    else { TORCH_CHECK(false, "Unsupported blockSize: ", blockSize); }

    switch(head_dim)
    {
    case 64:  DISPATCH_BLOCK_SIZE(64);  break;
    case 128: DISPATCH_BLOCK_SIZE(128); break;
    case 256: DISPATCH_BLOCK_SIZE(256); break;
        
#undef LAUNCH_BLOCK_QUANT_KERNEL
#undef DISPATCH_BLOCK_SIZE
#undef DISPATCH_X_VALUE
#undef DISPATCH_INTERLEAVE_BQ
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
 #define CALL_QK_NORM_ROPE_CACHE_BLOCK_QUANT(SRC_T, CACHE_T, KV_DTYPE)       \
         launchFusedQKNormRopeBlockQuantCacheShuffle<SRC_T, CACHE_T, KV_DTYPE>( \
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
             cu_q_len.data_ptr<int64_t>(),                                 \
             k_scale.has_value() ? k_scale->data_ptr<float>() : nullptr,   \
             v_scale.has_value() ? v_scale->data_ptr<float>() : nullptr,   \
             page_size,                                                    \
             x,                                                            \
             batch_size,                                                   \
             max_tokens_per_batch,                                         \
             stream);

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

template <typename T, int HEAD_SIZE, bool IS_NEOX>
__global__ void fused_rope_rms_1way_kernel(const T* q_,
                                           const T* k_,
                                           const T* w_q,
                                           const T* w_k,
                                           const T* cos_sin,
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

    vec_t<T, VEC_SIZE> w_vec, x_vec, cos_sin_vec, cos_vec, sin_vec;

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

    mrope_utils::warp_rms_norm_<T, VEC_SIZE>(x_vec, w_vec, HEAD_SIZE, eps);
    vec_t<T, VEC_SIZE> out_vec;

    if constexpr(IS_NEOX)
    {
        // ds_swizzle XOR-by-NEIGHBOR_XOR — replaces the prior runtime `lane + neighbor_offset`
        // path that lowered to ds_bpermute_b32. Same semantics as `__shfl(v, lane ^ NEIGHBOR_XOR, 32)`.
        auto nb_cos_sin_vec = mrope_utils::warp_shfl_xor_sync_vec<T, VEC_SIZE>(
            cos_sin_vec, opus::number<NEIGHBOR_XOR>{});
        auto nb_x_vec = mrope_utils::warp_shfl_xor_sync_vec<T, VEC_SIZE>(
            x_vec, opus::number<NEIGHBOR_XOR>{});
        if(is_lower_half)
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
                                                   const T* cos_sin,
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
    if(global_warp_id >= total_warps_quad)
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
    const bool is_q      = global_warp_id < warp_q_end;

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

    vec_t<T, VEC_PAIR> cos_sin_vec;
    vec_t<T, VEC_PAIR / 2> cos_vec_pair, sin_vec_pair;
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
            n0[i] = v0[i] * s_val0 * (float)w_vec[i];
            n1[i] = v1[i] * s_val1 * (float)w_vec[i];
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
        auto nb_cos_sin_vec = mrope_utils::warp_shfl_xor_sync_vec<T, VEC_PAIR>(
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
                         const T* cos_sin,
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
    TORCH_CHECK(head_size == 64 || head_size == 128 || head_size == 256);
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
    at::Tensor& k_cache,               // [num_blocks, num_kv_heads, head_dim//x, page_size, x]
    at::Tensor& v_cache,               // 4D [num_blocks, num_heads_v, head_dim, page_size] or 5D shuffle
                                       // [num_blocks, num_heads_v, page_size//x, head_dim, x]
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
    CHECK_INPUT(k_cache);
    CHECK_INPUT(v_cache);
    CHECK_INPUT(slot_mapping);
    CHECK_TYPE(position_ids, torch::kInt64);
    CHECK_TYPE(slot_mapping, torch::kInt64);

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
    TORCH_CHECK(position_ids.size(0) == num_tokens,
                "Number of tokens in position_ids must match QKV");

    TORCH_CHECK(k_cache.dim() == 5,
                "k_cache must be 5D [num_blocks, num_kv_heads, head_dim//x, page_size, x], got dim ",
                k_cache.dim());
    int64_t x            = k_cache.size(-1);
    int64_t page_size_k  = k_cache.size(-2);
    TORCH_CHECK(x > 0 && head_dim % x == 0,
                "head_dim (",
                head_dim,
                ") must be divisible by k_cache x (",
                x,
                ")");
    TORCH_CHECK(k_cache.size(2) == head_dim / x,
                "k_cache dim 2 must equal head_dim//x, got ",
                k_cache.size(2),
                " expected ",
                head_dim / x);
    TORCH_CHECK(k_cache.size(1) == num_heads_k,
                "k_cache dim 1 must equal num_heads_k, got ",
                k_cache.size(1));

    int64_t page_size;
    if(v_cache.dim() == 5)
    {
        // Shuffle layout: [num_blocks, num_heads_v, page_size//x, head_dim, x]
        TORCH_CHECK(v_cache.size(0) == k_cache.size(0),
                    "v_cache and k_cache num_blocks must match");
        TORCH_CHECK(v_cache.size(1) == num_heads_v,
                    "v_cache dim 1 must equal num_heads_v, got ",
                    v_cache.size(1));
        TORCH_CHECK(v_cache.size(-1) == x && v_cache.size(-2) == head_dim,
                    "v_cache trailing dims must be [head_dim, x], got [",
                    v_cache.size(-2),
                    ", ",
                    v_cache.size(-1),
                    "]");
        TORCH_CHECK(v_cache.size(-3) * x == page_size_k,
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
        TORCH_CHECK(v_cache.size(0) == k_cache.size(0),
                    "v_cache and k_cache num_blocks must match");
        TORCH_CHECK(v_cache.size(1) == num_heads_v,
                    "v_cache dim 1 must equal num_heads_v, got ",
                    v_cache.size(1));
        TORCH_CHECK(v_cache.size(2) == head_dim,
                    "v_cache dim 2 must equal head_dim, got ",
                    v_cache.size(2));
        page_size = v_cache.size(-1);
        TORCH_CHECK(page_size == page_size_k,
                    "v_cache page_size (last dim) must match k_cache page_size; got ",
                    page_size,
                    " vs ",
                    page_size_k);
        TORCH_CHECK(page_size % x == 0,
                    "page_size must be divisible by x for V cache layout; got page_size=",
                    page_size,
                    " x=",
                    x);
    }
    else
    {
        TORCH_CHECK(false,
                    "v_cache must be 4D [num_blocks, num_heads_v, head_dim, page_size] or 5D shuffle "
                    "[num_blocks, num_heads_v, page_size//x, head_dim, x], got dim ",
                    v_cache.dim());
    }

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
                                                int64_t x,
                                                int64_t rotary_dim)
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
                                                         x,
                                                         rotary_dim);
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
                        x,
                        rotary_dim);
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
                        x,
                        rotary_dim);
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

void fused_qk_norm_rope_1way(at::Tensor& q,
                             at::Tensor& k,
                             at::Tensor& w_q,
                             at::Tensor& w_k,
                             at::Tensor& cos_sin,
                             int64_t batch_size,
                             int64_t num_tokens,
                             int64_t num_heads_q,
                             int64_t num_heads_k,
                             int64_t head_size,
                             bool is_interleaved,
                             double eps,
                             at::Tensor& out_q,
                             at::Tensor& out_k)
{
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous());
    TORCH_CHECK(w_q.is_contiguous() && w_k.is_contiguous());
    TORCH_CHECK(cos_sin.is_contiguous());
    TORCH_CHECK(out_q.is_contiguous() && out_k.is_contiguous());
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(q));
    auto stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf, q.scalar_type(), "fused_qk_norm_rope_1way", [&] {
            using T = KernelElementType<scalar_t>::type;
            fused_rope_rms_1way<T>((T*)q.data_ptr<scalar_t>(),
                                   (T*)k.data_ptr<scalar_t>(),
                                   (T*)w_q.data_ptr<scalar_t>(),
                                   (T*)w_k.data_ptr<scalar_t>(),
                                   (T*)cos_sin.data_ptr<scalar_t>(),
                                   batch_size,
                                   num_tokens,
                                   num_heads_q,
                                   num_heads_k,
                                   head_size,
                                   is_interleaved,
                                   eps,
                                   (T*)out_q.data_ptr<scalar_t>(),
                                   (T*)out_k.data_ptr<scalar_t>(),
                                   stream);
        });
}

void fused_qk_norm_rope_cache_block_quant_shuffle(
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
    at::Tensor& cu_q_len,              // cu q len tensor [0, batch0_seq_len, batch0_seq_len + batch1_seq_len, ...]
    const std::string& kv_cache_dtype, // kv cache data type
    std::optional<at::Tensor> k_scale, // k scale tensor for quantized k cache
    std::optional<at::Tensor> v_scale, // v scale tensor for quantized v cache
    int64_t max_tokens_per_batch       // max tokens in any single batch (0 = use avg, safe for uniform distributions)
)
 {
     // Input validation
     CHECK_INPUT(qkv);
     CHECK_INPUT(cu_q_len);
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

     // cu_q_len format: [0, batch0_seq_len, batch0_seq_len + batch1_seq_len, ...]
     // batch_size = cu_q_len.size(0) - 1
     TORCH_CHECK(cu_q_len.dim() == 1, "Cu Q len tensor must be 1D");
     int64_t batch_size = cu_q_len.size(0) - 1;
     TORCH_CHECK(batch_size > 0, "Batch size must be greater than 0");
     
     int64_t num_tokens = qkv.size(0);
     int64_t page_size  = k_cache.size(-2);
     int64_t x          = k_cache.size(-1);
     TORCH_CHECK(x > 0 && (x & (x - 1)) == 0,
                 "KV cache tiling size (x) must be a power of two, got ", x);
     // vec_size is 8 for bf16/fp16, 4 for fp32; vec_per_x = x/vec_size requires x >= vec_size
     TORCH_CHECK(x >= 4,
                 "KV cache tiling size (x) must be >= 4 for vectorized access, got ", x);
     TORCH_CHECK(position_ids.size(0) == num_tokens,
                 "Number of tokens in position_ids must match QKV");

 
     int64_t total_heads = num_heads_q + num_heads_k + num_heads_v;
     TORCH_CHECK(qkv.size(1) == total_heads * head_dim,
                 "QKV tensor size must match total number of heads and head dimension");
 
     auto stream = at::hip::getCurrentHIPStream(qkv.get_device());
     DISPATCH_BY_KV_CACHE_DTYPE_OPUS(qkv.scalar_type(), kv_cache_dtype, CALL_QK_NORM_ROPE_CACHE_BLOCK_QUANT);
 }

} // namespace aiter
