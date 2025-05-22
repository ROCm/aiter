#pragma once
/*
 * Copyright © Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) 2024, The vLLM team.
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
#include "vectorization.cuh"

#include <cmath>
#include "hip_float8.h"
#include "py_itfs_common.h"
using FP8_TYPE = ck_tile::fp8_t;

namespace vllm
{

  __device__ __forceinline__ float atomicMaxFloat(float *addr, float value)
  {
    float old;
    old = (value >= 0)
              ? __int_as_float(atomicMax((int *)addr, __float_as_int(value)))
              : __uint_as_float(
                    atomicMin((unsigned int *)addr, __float_as_uint(value)));

    return old;
  }

  template <bool is_scale_inverted>
  __device__ __forceinline__ FP8_TYPE scaled_fp8_conversion(float const val,
                                                            float const scale)
  {
    float x = 0.0f;
    if constexpr (is_scale_inverted)
    {
      x = val * scale;
    }
    else
    {
      x = val / scale;
    }

    return ck_tile::type_convert<FP8_TYPE, float>(x);
  }

  __global__ void initializeScale(float *d_data, int size, float value)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
      d_data[idx] = value;
    }
  }

  // Compute the absolute maximum m of the input tensor and store
  // m / float8_e4m3::max() in *scale. Each thread block performs a
  // reduction tree and the memory in scale is atomically updated.
  // So to get the right answer, *scale needs to be initialized to
  // a value <= 0.0 and we need to wait for all thread blocks to
  // finish before consuming *scale.
  template <typename scalar_t>
  __global__ void segmented_max_reduction(float *__restrict__ scale,
                                          const scalar_t *__restrict__ input,
                                          int64_t num_elems)
  {
    __shared__ float cache[1024];
    int64_t i = blockDim.x * blockIdx.x + threadIdx.x;

    // First store maximum for all values processes by
    // the current thread in cache[threadIdx.x]
    scalar_t tmp = 0.0;
    while (i < num_elems)
    {
      float x = static_cast<float>(input[i]);
      tmp = max(tmp, fabs(x));
      i += blockDim.x * gridDim.x;
    }
    cache[threadIdx.x] = tmp;

    __syncthreads();

    // Now perform parallel reduction within the thread block
    int ib = blockDim.x / 2;
    while (ib != 0)
    {
      if (threadIdx.x < ib && cache[threadIdx.x + ib] > cache[threadIdx.x])
      {
        cache[threadIdx.x] = cache[threadIdx.x + ib];
      }
      __syncthreads();
      ib /= 2;
    }
    // Finally, since cache[0] contains the maximum for this thread block,
    // atomically write the max to the target location
    if (threadIdx.x == 0)
    {
      atomicMaxFloat(scale, cache[0] / FP8_MAX);
    }
  }

  template <typename scalar_t>
  __device__ float thread_max_vec(scalar_t const *__restrict__ input,
                                  int64_t const num_elems, int const tid,
                                  int const step)
  {
    // Vectorized input/output to better utilize memory bandwidth.
    vec4_t<scalar_t> const *vectorized_in =
        reinterpret_cast<vec4_t<scalar_t> const *>(input);

    int64_t const num_vec_elems = num_elems >> 2;
    float absmax_val = 0.0f;

#pragma unroll 4
    for (int64_t i = tid; i < num_vec_elems; i += step)
    {
      vec4_t<scalar_t> in_vec = vectorized_in[i];
      absmax_val = max(absmax_val, fabs(in_vec.x));
      absmax_val = max(absmax_val, fabs(in_vec.y));
      absmax_val = max(absmax_val, fabs(in_vec.z));
      absmax_val = max(absmax_val, fabs(in_vec.w));
    }

    // Handle the remaining elements if num_elems is not divisible by 4
    for (int64_t i = num_vec_elems * 4 + tid; i < num_elems; i += step)
    {
      absmax_val = max(absmax_val, fabs(input[i]));
    }

    return absmax_val;
  }

  template <typename scalar_t, bool is_scale_inverted>
  __device__ void scaled_fp8_conversion_vec(FP8_TYPE *__restrict__ out,
                                            scalar_t const *__restrict__ input,
                                            float const scale,
                                            int64_t const num_elems,
                                            int const tid, int const step)
  {
    using float8x4_t = q8x4_t<FP8_TYPE>;
    // Vectorized input/output to better utilize memory bandwidth.
    auto const *vectorized_in = reinterpret_cast<vec4_t<scalar_t> const *>(input);
    auto *vectorized_out = reinterpret_cast<float8x4_t *>(out);

    int64_t const num_vec_elems = num_elems >> 2;

#pragma unroll 4
    for (int64_t i = tid; i < num_vec_elems; i += step)
    {
      vec4_t<scalar_t> in_vec = vectorized_in[i];
      float8x4_t out_vec;

      out_vec.x = scaled_fp8_conversion<is_scale_inverted>(
          static_cast<float>(in_vec.x), scale);
      out_vec.y = scaled_fp8_conversion<is_scale_inverted>(
          static_cast<float>(in_vec.y), scale);
      out_vec.z = scaled_fp8_conversion<is_scale_inverted>(
          static_cast<float>(in_vec.z), scale);
      out_vec.w = scaled_fp8_conversion<is_scale_inverted>(
          static_cast<float>(in_vec.w), scale);
      vectorized_out[i] = out_vec;
    }

    // Handle the remaining elements if num_elems is not divisible by 4
    for (int64_t i = num_vec_elems * 4 + tid; i < num_elems; i += step)
    {
      out[i] = scaled_fp8_conversion<is_scale_inverted>(
          static_cast<float>(input[i]), scale);
    }
  }

} // namespace vllm
