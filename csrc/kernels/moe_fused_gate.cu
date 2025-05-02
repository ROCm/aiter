/*
 * Copyright © Advanced Micro Devices, Inc. All rights reserved.
 * Adapted from https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/moe/moe_fused_gate.cu
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <ck/ck.hpp>
#include <ck/utility/data_type.hpp>
#include <ck/utility/type_convert.hpp>
#include <ck/utility/array.hpp>
#include <stdio.h>
#include <torch/all.h>

#include <cfloat>
#include <type_traits>

#include <cub/util_type.cuh>
#include <cub/cub.cuh>
#include "hip_compat.h"


/// Aligned array type
template <
    typename T,
    /// Number of elements in the array
    int N,
    /// Alignment requirement in bytes
    int Alignment = sizeof(T) * N
>
class alignas(Alignment) AlignedArray {
    float data[N];
};

using bfloat16_t = ck::bhalf_t;
using float16_t = ck::half_t;
using float32_t = float;

// QQ NOTE: to handle the case for at::Half, error: more than one operator ">" matches these operands: built-in operator
// "arithmetic > arithmetic" function "operator>(const __half &, const __half &)"
template <typename T>
__device__ inline bool cmp_gt(const T& a, const T& b) {
  if constexpr (std::is_same<T, float16_t>::value || std::is_same<T, bfloat16_t>::value) {
    // at::Half (or float16_t in our native case) causes ambiguity, so we cast to float.
    return ck::type_convert<float>(a) > ck::type_convert<float>(b);
  } else {
    // For types like float, at::BFloat16, or cutlass::half_t / cutlass::bfloat16_t, assume operator> works as expected.
    return a > b;
  }
}

template <typename T>
__device__ inline bool cmp_eq(const T& a, const T& b) {
  if constexpr (std::is_same<T, float16_t>::value || std::is_same<T, bfloat16_t>::value) {
    return ck::type_convert<float>(a) == ck::type_convert<float>(b);
  } else {
    return a == b;
  }
}

// Fixed constants common to both dynamic and static template versions:
// static constexpr int WARP_SIZE = 32;
static constexpr int WARPS_PER_CTA = 1;
static constexpr int MAX_VPT = 32;  // maximum VPT we support, > params.VPT = num_expert / num_expert_group

// Create an alias for Array using AlignedArray
// template <typename T, int N>
// using Array = AlignedArray<T, N>;
// QQ: NOTE expression must have a constant value, this has to be > params.VPT
// template <typename T>
// using AccessType = AlignedArray<T, MAX_VPT>;

template <typename T, typename Params>
__device__ void moe_fused_gate_impl(
    void* input,
    void* bias,
    float* output_ptr,
    int32_t* indices_ptr,
    int64_t num_rows,
    int64_t topk_group,
    int64_t topk,
    int64_t n_share_experts_fusion,
    double routed_scaling_factor,
    Params params) {
  int tidx = threadIdx.x;
  int64_t thread_row =
      blockIdx.x * params.ROWS_PER_CTA + threadIdx.y * params.ROWS_PER_WARP + tidx / params.THREADS_PER_ROW;
  if (thread_row >= num_rows) {
    return;
  }

  // Calculate topk_excluding_share_expert_fusion from topk
  int64_t topk_excluding_share_expert_fusion = topk - (n_share_experts_fusion > 0 ? 1 : 0);

  // Cast pointers to type T:
  auto* input_ptr = reinterpret_cast<T*>(input);
  auto* bias_ptr = reinterpret_cast<T*>(bias);
  auto* thread_row_ptr = input_ptr + thread_row * params.NUM_EXPERTS;

  int thread_group_idx = tidx % params.THREADS_PER_ROW;
  int first_elt_read_by_thread = thread_group_idx * params.VPT;

  // Create local arrays for the row chunk and bias chunk and then reinterpret the address of row_chunk as a pointer to
  // AccessType.

  constexpr uint32_t vec_size = 32 / sizeof(T);
  using AccessType = ck::Array<T, vec_size>;

  T* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;
  float row_chunk[MAX_VPT];
  AccessType const* vec_thread_read_ptr = reinterpret_cast<AccessType const*>(thread_read_ptr);

  T* bias_thread_read_ptr = bias_ptr + first_elt_read_by_thread;
  float bias_chunk[MAX_VPT];
  AccessType const* vec_bias_thread_read_ptr = reinterpret_cast<AccessType const*>(bias_thread_read_ptr);

// QQ NOTE: doing the follow will be slower than loop assign and more importantly
// have misaligned address issue when params.VPT < 8 and mismatch with MAX_VPT
// AccessType<T>* row_chunk_vec_ptr = reinterpret_cast<AccessType<T>*>(&row_chunk);
  // row_chunk_vec_ptr[0] = vec_thread_read_ptr[0];
  // bias_chunk_vec_ptr[0] = vec_bias_thread_read_ptr[0];
// #pragma unroll
//   for (int ii = 0; ii < params.VPT; ++ii) {
//     row_chunk_vec_ptr[ii] = vec_thread_read_ptr[0][ii];
//     bias_chunk_vec_ptr[ii] = vec_bias_thread_read_ptr[0][ii];
//   }]

  #pragma unroll
  for (int ii = 0; ii < params.VPT / vec_size; ++ii) {
    AccessType row_chunk_vec = vec_thread_read_ptr[ii];
    AccessType bias_thread_read_vec = vec_bias_thread_read_ptr[ii];
    for (int jj = 0; jj < vec_size; ++jj) {
      row_chunk[ii * vec_size + jj] = ck::type_convert<float>(row_chunk_vec(jj));
      bias_chunk[ii * vec_size + jj] = ck::type_convert<float>(bias_thread_read_vec(jj));
    }
  }
  
  __syncthreads();

////////////////////// Sigmoid //////////////////////
#pragma unroll
  for (int ii = 0; ii < params.VPT; ++ii) {
    row_chunk[ii] = 1.0f / (1.0f + expf(-row_chunk[ii]));
  }
  __syncthreads();

////////////////////// Add Bias //////////////////////
#pragma unroll
  for (int ii = 0; ii < params.VPT; ++ii) {
    bias_chunk[ii] = row_chunk[ii] + bias_chunk[ii];
  }

  // local argmax
  float max_val = -FLT_MAX;
  float max_val_second = -FLT_MAX;
#pragma unroll
  for (int ii = 0; ii < params.VPT; ++ii) {
    float val = bias_chunk[ii];

    if (cmp_gt(val, max_val)) {
      max_val_second = max_val;
      max_val = val;
    } else if (cmp_gt(val, max_val_second)) {
      max_val_second = val;
    }
  }
  // QQ NOTE: currently fixed to pick top2 sigmoid weight value in each expert group and sum them as the group weight
  // to select expert groups
  max_val = max_val + max_val_second;

////////////////////// Exclude Groups //////////////////////
#pragma unroll
  for (int k_idx = 0; k_idx < params.THREADS_PER_ROW - topk_group;
       ++k_idx) {  // QQ NOTE Here params.THREADS_PER_ROW = num_expert_group
    int expert = first_elt_read_by_thread;
    float max_sum = max_val;
    
// argmin reduce
#pragma unroll
    for (int mask = params.THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      float other_max_sum =
          VLLM_SHFL_XOR_SYNC_WIDTH(max_sum, mask, params.THREADS_PER_ROW);
      int other_expert = VLLM_SHFL_XOR_SYNC_WIDTH(expert, mask, params.THREADS_PER_ROW);

      // higher indices win
      if (cmp_gt(max_sum, other_max_sum) || (cmp_eq(other_max_sum, max_sum) && other_expert > expert)) {
        max_sum = other_max_sum;
        expert = other_expert;
      }
    }

    // clear the max value in the thread
    if (k_idx < params.THREADS_PER_ROW - topk_group) {
      int const thread_to_clear_in_group = expert / params.VPT;

      if (thread_group_idx == thread_to_clear_in_group) {
        bias_chunk[0] = FLT_MAX;
        max_val = FLT_MAX;
      }
    }
  }

  __syncthreads();

  ////////////////////// Topk //////////////////////
  float output_sum = 0.0f;
  for (int k_idx = 0; k_idx < topk_excluding_share_expert_fusion; ++k_idx) {
    // local argmax
    float max_val = bias_chunk[0];
    int expert = first_elt_read_by_thread;

    if (!cmp_eq(max_val, FLT_MAX)) {
#pragma unroll
      for (int ii = 1; ii < params.VPT; ++ii) {
        float val = bias_chunk[ii];
        if (cmp_gt(val, max_val)) {
          max_val = val;
          expert = first_elt_read_by_thread + ii;
        }
      }
    } else {
      max_val = -FLT_MAX;
    }

    // argmax reduce
#pragma unroll
    for (int mask = params.THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      float other_max =
          VLLM_SHFL_XOR_SYNC_WIDTH(max_val, mask, params.THREADS_PER_ROW);
      int other_expert = VLLM_SHFL_XOR_SYNC_WIDTH(expert, mask, params.THREADS_PER_ROW);

      // lower indices to win
      if (cmp_gt(other_max, max_val) || (cmp_eq(other_max, max_val) && other_expert < expert)) {
        max_val = other_max;
        expert = other_expert;
      }
    }

    int thread_to_clear_in_group = expert / params.VPT;
    int64_t idx = topk * thread_row + k_idx;

    if (thread_group_idx == thread_to_clear_in_group) {
      int expert_to_clear_in_thread = expert % params.VPT;

      // clear the max value in the thread
      bias_chunk[expert_to_clear_in_thread] = -FLT_MAX;

      // store output
      output_ptr[idx] = row_chunk[expert_to_clear_in_thread];
      indices_ptr[idx] = ck::type_convert<int32_t>(expert);
    }

    // accumulate sum for all elements
    if (thread_group_idx == 0) {
      output_sum += output_ptr[idx];
    }

    __syncthreads();
  }

  if (thread_group_idx == 0 && n_share_experts_fusion > 0) {
    int64_t last_idx = topk * thread_row + topk_excluding_share_expert_fusion;

    // Use round-robin to select expert
    int64_t expert_offset = thread_row % n_share_experts_fusion;
    indices_ptr[last_idx] = ck::type_convert<int32_t>(params.NUM_EXPERTS + expert_offset);

    // Set the weight to the sum of all weights divided by routed_scaling_factor
    output_ptr[last_idx] = output_sum / routed_scaling_factor;
  }
  __syncthreads();

  ////////////////////// Rescale Output //////////////////////
  if (thread_group_idx == 0) {
#pragma unroll
    for (int ii = 0; ii < topk; ++ii) {
      int64_t const idx = topk * thread_row + ii;
      output_ptr[idx] = output_ptr[idx] / output_sum;
    }
  }
}

//------------------------------------------------------------------------------
// Templated Kernel Version (using compile-time constants)
//------------------------------------------------------------------------------
template <int VPT_, int NUM_EXPERTS_, int THREADS_PER_ROW_, int ROWS_PER_WARP_, int ROWS_PER_CTA_, int WARPS_PER_CTA_>
struct KernelParams {
  static constexpr int VPT = VPT_;
  static constexpr int NUM_EXPERTS = NUM_EXPERTS_;
  static constexpr int THREADS_PER_ROW = THREADS_PER_ROW_;
  static constexpr int ROWS_PER_WARP = ROWS_PER_WARP_;
  static constexpr int ROWS_PER_CTA = ROWS_PER_CTA_;
  static constexpr int WARPS_PER_CTA = WARPS_PER_CTA_;
};

template <
    typename T,
    int VPT,
    int NUM_EXPERTS,
    int THREADS_PER_ROW,
    int ROWS_PER_WARP,
    int ROWS_PER_CTA,
    int WARPS_PER_CTA>
__global__ void moe_fused_gate_kernel(
    void* input,
    void* bias,
    float* output_ptr,
    int32_t* indices_ptr,
    int64_t num_rows,
    int64_t topk_group,
    int64_t topk,
    int64_t n_share_experts_fusion,
    double routed_scaling_factor) {
  KernelParams<VPT, NUM_EXPERTS, THREADS_PER_ROW, ROWS_PER_WARP, ROWS_PER_CTA, WARPS_PER_CTA> params;
  moe_fused_gate_impl<T>(
      input,
      bias,
      output_ptr,
      indices_ptr,
      num_rows,
      topk_group,
      topk,
      n_share_experts_fusion,
      routed_scaling_factor,
      params);
}

// Macro to compute compile-time constants and launch the kernel.
#define LAUNCH_MOE_GATE_CONFIG(T, EXPERTS, EXPERT_GROUP)                                                 \
  do {                                                                                                   \
    constexpr int VPT = (EXPERTS) / (EXPERT_GROUP);                                                      \
    /* If EXPERT_GROUP > WARP_SIZE, fall back to 1 row per warp */                                       \
    constexpr int ROWS_PER_WARP = ((EXPERT_GROUP) <= WARP_SIZE) ? (WARP_SIZE / (EXPERT_GROUP)) : 1;      \
    constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;                                          \
    moe_fused_gate_kernel<T, VPT, (EXPERTS), (EXPERT_GROUP), ROWS_PER_WARP, ROWS_PER_CTA, WARPS_PER_CTA> \
        <<<num_blocks, block_dim, 0, stream>>>(                                                          \
            input.data_ptr(),                                                                            \
            bias.data_ptr(),                                                                             \
            output.data_ptr<float>(),                                                                    \
            indices.data_ptr<int32_t>(),                                                                 \
            num_rows,                                                                                    \
            topk_group,                                                                                  \
            topk,                                                                                        \
            n_share_experts_fusion,                                                                      \
            routed_scaling_factor);                                                                      \
    dispatched = true;                                                                                   \
  } while (0)

//------------------------------------------------------------------------------
// Dynamic Kernel Version (parameters computed at runtime)
//------------------------------------------------------------------------------
struct KernelParamsDynamic {
  int VPT;
  int NUM_EXPERTS;
  int THREADS_PER_ROW;
  int ROWS_PER_WARP;
  int ROWS_PER_CTA;
  int WARPS_PER_CTA;
};

template <typename T>
__global__ void moe_fused_gate_kernel_dynamic(
    void* input,
    void* bias,
    float* output_ptr,
    int32_t* indices_ptr,
    int64_t num_rows,
    int64_t num_experts,
    int64_t num_expert_group,
    int64_t topk_group,
    int64_t topk,
    int64_t n_share_experts_fusion,
    double routed_scaling_factor) {
  KernelParamsDynamic params;
  params.NUM_EXPERTS = num_experts;             // e.g, for deepseek v3, this is 256
  params.VPT = num_experts / num_expert_group;  // e.g., for deepseek v3, this is 256 / 8 = 32
  params.THREADS_PER_ROW = num_expert_group;    // fixed as num_expert_group, e.g., for deepseek v3, this is 8
  params.WARPS_PER_CTA = WARPS_PER_CTA;         // fixed as 6
  params.ROWS_PER_WARP = std::max<int64_t>(1, WARP_SIZE / num_expert_group);  // WARP_SIZE is fixed as 32
  params.ROWS_PER_CTA = params.WARPS_PER_CTA * params.ROWS_PER_WARP;

  moe_fused_gate_impl<T>(
      input,
      bias,
      output_ptr,
      indices_ptr,
      num_rows,
      topk_group,
      topk,
      n_share_experts_fusion,
      routed_scaling_factor,
      params);
}

//------------------------------------------------------------------------------
// Host Launcher Function
//------------------------------------------------------------------------------
std::vector<at::Tensor> moe_fused_gate(
    at::Tensor& input,
    at::Tensor& bias,
    int64_t num_expert_group,
    int64_t topk_group,
    int64_t topk,
    int64_t n_share_experts_fusion,
    double routed_scaling_factor) {
  int64_t num_rows = input.size(0);
  int32_t num_experts = input.size(1);
  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto output = torch::empty({num_rows, topk}, options);
  auto indices = torch::empty({num_rows, topk}, options.dtype(torch::kInt32));

  // Compute grid dimensions based on runtime value for num_expert_group.
  int64_t rows_per_warp = std::max<int64_t>(1, WARP_SIZE / num_expert_group);
  int64_t num_warps = (num_rows + rows_per_warp - 1) / rows_per_warp;
  int64_t num_blocks = (num_warps + WARPS_PER_CTA - 1) / WARPS_PER_CTA;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 block_dim(WARP_SIZE, WARPS_PER_CTA);

  // Check 1: Ensure that num_experts is a power of 2.
  TORCH_CHECK((num_experts & (num_experts - 1)) == 0, "num_experts must be a power of 2, but got ", num_experts);

  // Check 2: Ensure that num_experts is divisible by num_expert_group. (this also means num_expert_group is power of 2)
  TORCH_CHECK(
      num_experts % num_expert_group == 0,
      "num_experts must be divisible by num_expert_group, but got ",
      num_experts,
      " / ",
      num_expert_group);

  int computed_vpt = num_experts / num_expert_group;
  // Check 3: Ensure that num_experts/num_expert_group does not exceed MAX_VPT=32. Maximum VPT indicate max value per
  // threads we can process.
  TORCH_CHECK(
      computed_vpt <= MAX_VPT,
      "Per group experts: num_experts / num_expert_group = (",
      computed_vpt,
      ") exceeds the maximum supported (",
      MAX_VPT,
      ")");

  // Dispatch to templated kernel for known compile-time configurations.
  // We currently only support for:
  //   Case 1: 256 experts, with 8 or 16 groups.
  //   Case 2: 128 experts, with 4 or 8 groups.
  //   Case 3: other cases, require 8 <= num_experts / num_expert_group <= 32
  bool dispatched = false;
  switch (num_experts) {
    case 256:
      if (num_expert_group == 8)
        // This is deepseek v3 case. Here VPT = 256/8 = 32, ROWS_PER_WARP = 32/8 = 4, ROWS_PER_CTA = 6 * 4 = 24.
        if (input.scalar_type() == at::kBFloat16) {
          LAUNCH_MOE_GATE_CONFIG(bfloat16_t, 256, 8);
        } else if (input.scalar_type() == at::kHalf) {
          LAUNCH_MOE_GATE_CONFIG(float16_t, 256, 8);
        } else if (input.scalar_type() == at::kFloat) {
          LAUNCH_MOE_GATE_CONFIG(float32_t, 256, 8);
        } else if (num_expert_group == 16)
          // Here VPT = 256/16 = 16, ROWS_PER_WARP = 32/16 = 2, ROWS_PER_CTA = 6 * 2 = 12.
          if (input.scalar_type() == at::kBFloat16) {
            LAUNCH_MOE_GATE_CONFIG(bfloat16_t, 256, 16);
          } else if (input.scalar_type() == at::kHalf) {
            LAUNCH_MOE_GATE_CONFIG(float16_t, 256, 16);
          } else if (input.scalar_type() == at::kFloat) {
            LAUNCH_MOE_GATE_CONFIG(float32_t, 256, 16);
          }
      break;
    case 128:
      if (num_expert_group == 4)
        // VPT = 128/4 = 32, ROWS_PER_WARP = 32/16 = 2, ROWS_PER_CTA = 6 * 2 = 12.
        if (input.scalar_type() == at::kBFloat16) {
          LAUNCH_MOE_GATE_CONFIG(bfloat16_t, 128, 4);
        } else if (input.scalar_type() == at::kHalf) {
          LAUNCH_MOE_GATE_CONFIG(float16_t, 128, 4);
        } else if (input.scalar_type() == at::kFloat) {
          LAUNCH_MOE_GATE_CONFIG(float32_t, 128, 4);
        } else if (num_expert_group == 8)
          // VPT = 128/8 = 16, ROWS_PER_WARP = 32/8 = 4, ROWS_PER_CTA = 6 * 4 = 24.
          if (input.scalar_type() == at::kBFloat16) {
            LAUNCH_MOE_GATE_CONFIG(bfloat16_t, 128, 8);
          } else if (input.scalar_type() == at::kHalf) {
            LAUNCH_MOE_GATE_CONFIG(float16_t, 128, 8);
          } else if (input.scalar_type() == at::kFloat) {
            LAUNCH_MOE_GATE_CONFIG(float32_t, 128, 8);
          }
      break;
    default:
      break;
  }
  if (!dispatched) {
    // Fallback to the dynamic kernel if none of the supported combinations match.
    // currently only support num_experts / num_expert_group <= 32 for dynamic kernels
    if (input.scalar_type() == at::kBFloat16) {
      moe_fused_gate_kernel_dynamic<bfloat16_t><<<num_blocks, block_dim, 0, stream>>>(
          input.data_ptr(),
          bias.data_ptr(),
          output.data_ptr<float>(),
          indices.data_ptr<int32_t>(),
          num_rows,
          num_experts,
          num_expert_group,
          topk_group,
          topk,
          n_share_experts_fusion,
          routed_scaling_factor);
    } else if (input.scalar_type() == at::kHalf) {
      moe_fused_gate_kernel_dynamic<float16_t><<<num_blocks, block_dim, 0, stream>>>(
          input.data_ptr(),
          bias.data_ptr(),
          output.data_ptr<float>(),
          indices.data_ptr<int32_t>(),
          num_rows,
          num_experts,
          num_expert_group,
          topk_group,
          topk,
          n_share_experts_fusion,
          routed_scaling_factor);
    } else if (input.scalar_type() == at::kFloat) {
      moe_fused_gate_kernel_dynamic<float32_t><<<num_blocks, block_dim, 0, stream>>>(
          input.data_ptr(),
          bias.data_ptr(),
          output.data_ptr<float>(),
          indices.data_ptr<int32_t>(),
          num_rows,
          num_experts,
          num_expert_group,
          topk_group,
          topk,
          n_share_experts_fusion,
          routed_scaling_factor);
    } else {
      TORCH_CHECK(false, "Unsupported data type for moe_fused_gate");
    }
  }
  return {output, indices};
}
