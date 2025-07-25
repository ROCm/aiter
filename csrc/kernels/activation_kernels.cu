/*
 * Copyright © Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (C) 2024-2025, The vLLM team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>

#include "hip_compat.h"
#include "dispatch_utils.h"
#include "py_itfs_common.h"
#include "ck_tile/core.hpp"

using fp8_type = ck_tile::fp8_t;

namespace vllm
{

  // Activation and gating kernel template.
  template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t &)>
  __global__ void act_and_mul_kernel(
      scalar_t *__restrict__ out,         // [..., d]
      const scalar_t *__restrict__ input, // [..., 2, d]
      const int d)
  {
    const int64_t token_idx = blockIdx.x;
    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x)
    {
      const scalar_t x = VLLM_LDG(&input[token_idx * 2 * d + idx]);
      const scalar_t y = VLLM_LDG(&input[token_idx * 2 * d + d + idx]);
      out[token_idx * d + idx] = ACT_FN(x) * y;
    }
  }

  // Scaled activation and gating kernel template.
  #ifdef USE_ROCM
  template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&)>
  __global__ void scaled_act_and_mul_kernel(
    fp8_type* __restrict__ out,  // [..., d]
      const scalar_t* __restrict__ input,      // [..., 2, d]
      const int d, const float scale) {
    const int64_t token_idx = blockIdx.x;
    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
      const scalar_t x = VLLM_LDG(&input[token_idx * 2 * d + idx]);
      const scalar_t y = VLLM_LDG(&input[token_idx * 2 * d + d + idx]);
      float r = ACT_FN(x) * y * scale;
      out[token_idx * d + idx] = ck_tile::type_convert<fp8_type>(r);
    }
  }
  #endif

  template <typename T>
  __device__ __forceinline__ T silu_kernel(const T &x)
  {
    // x * sigmoid(x)
    return (T)(((float)x) / (1.0f + expf((float)-x)));
  }

  template <typename T>
  __device__ __forceinline__ T gelu_kernel(const T &x)
  {
    // Equivalent to PyTorch GELU with 'none' approximation.
    // Refer to:
    // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L36-L38
    const float f = (float)x;
    constexpr float ALPHA = M_SQRT1_2;
    return (T)(f * 0.5f * (1.0f + ::erf(f * ALPHA)));
  }

  template <typename T>
  __device__ __forceinline__ T gelu_tanh_kernel(const T &x)
  {
    // Equivalent to PyTorch GELU with 'tanh' approximation.
    // Refer to:
    // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L25-L30
    const float f = (float)x;
    constexpr float BETA = M_SQRT2 * M_2_SQRTPI * 0.5f;
    constexpr float KAPPA = 0.044715;
    float x_cube = f * f * f;
    float inner = BETA * (f + KAPPA * x_cube);
    return (T)(0.5f * f * (1.0f + ::tanhf(inner)));
  }

} // namespace vllm

// Launch activation and gating kernel.
#define LAUNCH_ACTIVATION_GATE_KERNEL(KERNEL)                                                                     \
  int d = input.size(-1) / 2;                                                                                     \
  int64_t num_tokens = input.numel() / input.size(-1);                                                            \
  dim3 grid(num_tokens);                                                                                          \
  dim3 block(std::min(d, 1024));                                                                                  \
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));                                               \
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                                   \
  VLLM_DISPATCH_FLOATING_TYPES(                                                                                   \
      input.scalar_type(), "act_and_mul_kernel", [&] { vllm::act_and_mul_kernel<scalar_t, KERNEL<scalar_t>>       \
                                                           <<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(), \
                                                                                        input.data_ptr<scalar_t>(), d); });
// Launch activation and gating kernel.
#ifdef USE_ROCM
  #define LAUNCH_SCALED_ACTIVATION_GATE_KERNEL(KERNEL)                \
    int d = input.size(-1) / 2;                                       \
    int64_t num_tokens = input.numel() / input.size(-1);              \
    dim3 grid(num_tokens);                                            \
    dim3 block(std::min(d, 1024));                                    \
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input)); \
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();     \
    VLLM_DISPATCH_FLOATING_TYPES(                                     \
        input.scalar_type(), "scaled_act_and_mul_kernel", [&] {       \
          vllm::scaled_act_and_mul_kernel<scalar_t, KERNEL<scalar_t>> \
              <<<grid, block, 0, stream>>>(                           \
                  reinterpret_cast<fp8_type*>(out.data_ptr()),               \
                  input.data_ptr<scalar_t>(), d,                      \
                  1.0 / (*scale.data_ptr<float>()));                  \
        });
#endif

namespace aiter {

void silu_and_mul(torch::Tensor &out,   // [..., d]
                  torch::Tensor &input) // [..., 2 * d]
{
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::silu_kernel);
}

void scaled_silu_and_mul(torch::Tensor &out,   // [..., d]
                  torch::Tensor &input, // [..., 2 * d]
		  torch::Tensor &scale)
{
  LAUNCH_SCALED_ACTIVATION_GATE_KERNEL(vllm::silu_kernel);
}

void gelu_and_mul(torch::Tensor &out,   // [..., d]
                  torch::Tensor &input) // [..., 2 * d]
{
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::gelu_kernel);
}

void gelu_tanh_and_mul(torch::Tensor &out,   // [..., d]
                       torch::Tensor &input) // [..., 2 * d]
{
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::gelu_tanh_kernel);
}

} // namespace aiter

namespace vllm
{

  // Element-wise activation kernel template.
  template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t &)>
  __global__ void activation_kernel(
      scalar_t *__restrict__ out,         // [..., d]
      const scalar_t *__restrict__ input, // [..., d]
      const int d)
  {
    const int64_t token_idx = blockIdx.x;
    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x)
    {
      const scalar_t x = VLLM_LDG(&input[token_idx * d + idx]);
      out[token_idx * d + idx] = ACT_FN(x);
    }
  }

} // namespace vllm

// Launch element-wise activation kernel.
#define LAUNCH_ACTIVATION_KERNEL(KERNEL)                                                                                                  \
  int d = input.size(-1);                                                                                                                 \
  int64_t num_tokens = input.numel() / d;                                                                                                 \
  dim3 grid(num_tokens);                                                                                                                  \
  dim3 block(std::min(d, 1024));                                                                                                          \
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));                                                                       \
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                                                           \
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "activation_kernel", [&] { vllm::activation_kernel<scalar_t, KERNEL<scalar_t>>        \
                                                                                   <<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(), \
                                                                                                                input.data_ptr<scalar_t>(), d); });

namespace vllm
{

  template <typename T>
  __device__ __forceinline__ T gelu_new_kernel(const T &x)
  {
    const float x3 = (float)(x * x * x);
    const T t = (T)tanhf((T)(0.79788456f * (float)(x + (T)(0.044715f * x3))));
    return ((T)0.5) * x * (((T)1.0) + t);
  }

  template <typename T>
  __device__ __forceinline__ T gelu_fast_kernel(const T &x)
  {
    const float f = (float)x;
    const T t =
        (T)tanhf(((T)(f * 0.79788456f)) * (((T)1.0) + (T)(0.044715f * f) * x));
    return ((T)0.5) * x * (((T)1.0) + t);
  }

} // namespace vllm

namespace aiter {

void gelu_new(torch::Tensor &out,   // [..., d]
              torch::Tensor &input) // [..., d]
{
  LAUNCH_ACTIVATION_KERNEL(vllm::gelu_new_kernel);
}

void gelu_fast(torch::Tensor &out,   // [..., d]
               torch::Tensor &input) // [..., d]
{
  LAUNCH_ACTIVATION_KERNEL(vllm::gelu_fast_kernel);
}

} // namespace aiter
