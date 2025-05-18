// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "quant_common.cuh"
#include "dispatch_utils.h"

#include <c10/cuda/CUDAGuard.h>

#include <hipcub/hipcub.hpp>
#include "vec_convert.h"

namespace aiter
{
  template <typename DTYPE_I, typename DTYPE_O>
  __device__ float data_to_per_row_scale(const DTYPE_I *__restrict__ input,
                                         int64_t cols)
  {
    static constexpr int32_t vec_size = 16 / sizeof(DTYPE_I);
    using vec_i = ck_tile::vec_t<DTYPE_I, vec_size>;
    using tb_i = ck_tile::thread_buffer<DTYPE_I, vec_size>;
    const float inverted_DTYPE_MAX = 1. / ck_tile::type_convert<float>(ck_tile::numeric<DTYPE_O>::max());

    const int32_t row_offset = blockIdx.x * cols;
    auto const *input_ptr = reinterpret_cast<DTYPE_I const *>(input);
    auto const *input_vecs = reinterpret_cast<vec_i const *>(input + blockIdx.x * cols);
    auto buffer_i = ck_tile::make_buffer_view<ck_tile::address_space_enum::global>(input_ptr, gridDim.x * cols);
    buffer_i.init_raw();

    // double load core loop start
    const int32_t num_elems_tail = cols % vec_size;
    const int32_t num_vecs = cols / vec_size;
    // const int32_t num_vecs = (cols + vec_size - 1) / vec_size * vec_size;
    vec_i vec_nxt;
    vec_i vec_cur;
    // size_t vec_idx = threadIdx.x * vec_size;
    // size_t vec_stride = blockDim.x * vec_size;
    size_t vec_idx = threadIdx.x;
    size_t vec_stride = blockDim.x;
    if (vec_idx < num_vecs)
    {
      // vec_cur = ck_tile::bit_cast<vec_i>(buffer_i.template get<tb_i>(vec_idx, row_offset, true));
      vec_cur = input_vecs[vec_idx];
    }

    float absMax = 0.f;
    for (vec_idx += vec_stride; vec_idx < num_vecs; vec_idx += vec_stride)
    {
      vec_nxt = input_vecs[vec_idx];
      // vec_nxt = ck_tile::bit_cast<vec_i>(buffer_i.template get<tb_i>(vec_idx, row_offset, true));
      for (size_t j = 0; j < vec_size; j++)
      {
        absMax = max(absMax, abs(ck_tile::type_convert<float>(vec_cur[j])));
      }
      vec_cur = vec_nxt;
    }
    if (vec_idx - vec_stride < num_vecs)
    {
      for (size_t j = 0; j < vec_size; j++)
      {
        absMax = max(absMax, abs(ck_tile::type_convert<float>(vec_cur[j])));
      }
    }
    // double load core loop end

    // tail elements
    if (num_elems_tail > 0)
    {
      auto *tmp_i = reinterpret_cast<DTYPE_I const *>(input_vecs + num_vecs);
      for (size_t j = threadIdx.x; j < num_elems_tail; j += blockDim.x)
      {
        absMax = max(absMax, abs(ck_tile::type_convert<float>(tmp_i[j])));
      }
    }

    using BlockReduce = hipcub::BlockReduce<float, 256>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    absMax = BlockReduce(temp_storage).Reduce(absMax, hipcub::Max());
    return absMax * inverted_DTYPE_MAX;
  }

  template <typename DTYPE_I, typename DTYPE_O>
  __global__ void data_to_scale_kernel(float *__restrict__ scale,
                                       const DTYPE_I *__restrict__ input,
                                       int64_t cols)
  {
    float row_scale = data_to_per_row_scale<DTYPE_I, DTYPE_O>(input, cols);
    if (threadIdx.x == 0)
    {
      vllm::atomicMaxFloat(scale, row_scale);
    }
  }

  template <typename DTYPE_I, typename DTYPE_O>
  __device__ void scaled_quant_impl(DTYPE_O *__restrict__ out,
                                    const DTYPE_I *__restrict__ input,
                                    const float *__restrict__ scale,
                                    int64_t cols)
  {
    const float inverted_scale = 1.0f / (*scale);
    static constexpr int32_t vec_size = 16 / sizeof(DTYPE_O);
    using vec_i = ck_tile::vec_t<DTYPE_I, vec_size>;
    using tb_i = ck_tile::thread_buffer<DTYPE_I, vec_size>;
    using vec_o = ck_tile::vec_t<DTYPE_O, vec_size>;
    using tb_o = ck_tile::thread_buffer<DTYPE_O, vec_size>;

    const int32_t row_offset = blockIdx.x * cols;
    auto const *input_ptr = reinterpret_cast<DTYPE_I const *>(input);
    auto const *input_vecs = reinterpret_cast<vec_i const *>(input + row_offset);
    auto *out_ptr = reinterpret_cast<DTYPE_O *>(out);
    auto *out_vecs = reinterpret_cast<vec_o *>(out + row_offset);

    // auto buffer_i = ck_tile::make_buffer_view<ck_tile::address_space_enum::global, ck_tile::amd_buffer_coherence_enum::glc>(input_ptr, gridDim.x * cols);
    // buffer_i.init_raw();
    // auto buffer_o = ck_tile::make_buffer_view<ck_tile::address_space_enum::global>(out_ptr, gridDim.x * cols);
    // buffer_o.init_raw();

    // double load core loop start
    // const int32_t num_vecs = (cols + vec_size - 1) / vec_size * vec_size;
    const int32_t num_elems_tail = cols % vec_size;
    const int32_t num_vecs = cols / vec_size;
    const int32_t tail_thread = num_vecs % blockDim.x;
    vec_i vec_nxt;
    vec_i vec_cur;
    // size_t vec_idx = threadIdx.x * vec_size;
    // size_t vec_stride = blockDim.x * vec_size;
    size_t vec_idx = threadIdx.x;
    size_t vec_stride = blockDim.x;
    if (vec_idx < num_vecs)
    {
      // vec_cur = ck_tile::bit_cast<vec_i>(buffer_i.template get<tb_i>(vec_idx, row_offset, true));
      vec_cur = input_vecs[vec_idx];
    }

    for (vec_idx += vec_stride; vec_idx < num_vecs; vec_idx += vec_stride)
    {
      // vec_nxt = ck_tile::bit_cast<vec_i>(buffer_i.template get<tb_i>(vec_idx, row_offset, true));
      // buffer_o.template set<tb_o>((vec_idx - vec_stride), row_offset, true, ck_tile::bit_cast<tb_o>(ck_tile::vec_convert<DTYPE_O, DTYPE_I, vec_size>(vec_cur, inverted_scale)));
      vec_nxt = input_vecs[vec_idx];
      out_vecs[vec_idx - vec_stride] = ck_tile::vec_convert<DTYPE_O, DTYPE_I, vec_size>(vec_cur, inverted_scale);
      vec_cur = vec_nxt;
    }

    if (vec_idx - vec_stride < num_vecs)
    {
      // buffer_o.template set<tb_o>((vec_idx - vec_stride), row_offset, true, ck_tile::bit_cast<tb_o>(ck_tile::vec_convert<DTYPE_O, DTYPE_I, vec_size>(vec_cur, inverted_scale)));
      out_vecs[vec_idx - vec_stride] = ck_tile::vec_convert<DTYPE_O, DTYPE_I, vec_size>(vec_cur, inverted_scale);
    }
    // double load core loop end

    // tail elements
    if (num_elems_tail > 0)
    {
      auto *out_ptr2 = (out + row_offset);
      auto *tmp_i = reinterpret_cast<DTYPE_I const *>(input_vecs + num_vecs);
      for (size_t j = threadIdx.x; j < num_elems_tail; j += blockDim.x)
      {
        out_ptr2[num_vecs * vec_size + j] =
            ck_tile::type_convert<DTYPE_O>(ck_tile::type_convert<float>(tmp_i[j]) * inverted_scale);
      }
    }
  }
  template <typename DTYPE_I, typename DTYPE_O>
  __global__ void scaled_quant_kernel(DTYPE_O *__restrict__ out,
                                      const DTYPE_I *__restrict__ input,
                                      const float *__restrict__ scale,
                                      int64_t cols)
  {
    scaled_quant_impl<DTYPE_I>(out, input, scale, cols);
  }

  template <typename DTYPE_I, typename DTYPE_O>
  __global__ void dynamic_per_token_scaled_quant_kernel(
      DTYPE_O *__restrict__ out, float *__restrict__ scale,
      DTYPE_I const *__restrict__ input, float const *__restrict__ scale_ub,
      const int32_t cols)
  {
    // float const min_scaling_factor = 1.0f / (FP8_MAX * 512.f);

    const int32_t token_idx = blockIdx.x;
    float row_scale = data_to_per_row_scale<DTYPE_I, DTYPE_O>(input, cols);

    __shared__ float token_scale;
    if (threadIdx.x == 0)
    {
      token_scale = row_scale;
      scale[token_idx] = token_scale;
    }
    __syncthreads();

    scaled_quant_impl<DTYPE_I>(out, input, &token_scale, cols);
  }

} // namespace aiter

void static_per_tensor_quant(torch::Tensor &out,         // [..., d]
                             torch::Tensor const &input, // [..., d]
                             torch::Tensor const &scale) // [1]
{
  int64_t cols = input.size(-1);
  int64_t rows = input.numel() / cols;
  dim3 grid(rows);
  dim3 block(256);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "scaled_quant_kernel", [&]
      {using input_dtype= typename t2ck<scalar_t>::type;
    aiter::scaled_quant_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<FP8_TYPE *>(out.data_ptr()),
        reinterpret_cast<input_dtype *>(input.data_ptr()),
        scale.data_ptr<float>(), cols); });
}

void dynamic_per_tensor_quant(torch::Tensor &out,         // [..., d]
                              torch::Tensor const &input, // [..., d]
                              torch::Tensor &scale)       // [1]
{
  int64_t cols = input.size(-1);
  int64_t rows = input.numel() / cols;
  dim3 grid(rows);
  dim3 block(256);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "scaled_quant_kernel", [&]
      {using input_dtype= typename t2ck<scalar_t>::type;
      vllm::initializeScale<<<dim3(1), dim3(64), 0, stream>>>(scale.data_ptr<float>(), 1, 0.0f);
      aiter::data_to_scale_kernel<input_dtype, FP8_TYPE><<<grid, block, 0, stream>>>(
          scale.data_ptr<float>(),
          reinterpret_cast<input_dtype *>(input.data_ptr()), cols);
      aiter::scaled_quant_kernel<<<grid, block, 0, stream>>>(
          reinterpret_cast<FP8_TYPE *>(out.data_ptr()),
          reinterpret_cast<input_dtype *>(input.data_ptr()), scale.data_ptr<float>(), cols); });
}

void dynamic_per_token_scaled_quant(
    torch::Tensor &out,         // [..., d]
    torch::Tensor const &input, // [..., d]
    torch::Tensor &scales, std::optional<at::Tensor> const &scale_ub)
{
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(out.is_contiguous());

  int32_t const cols = input.size(-1);
  int32_t const rows = input.numel() / cols;
  dim3 const grid(rows);
  dim3 const block(256);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (out.dtype() == torch_fp8)
  {
    VLLM_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "dynamic_per_token_scaled_quant_kernel", [&]
        { using input_dtype= typename t2ck<scalar_t>::type;
    aiter::dynamic_per_token_scaled_quant_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<FP8_TYPE *>(out.data_ptr()), scales.data_ptr<float>(),
            reinterpret_cast<input_dtype*>(input.data_ptr()),
            scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr,
            cols); });
  }
  else if (out.dtype() == torch::kInt8)
  {
    VLLM_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "dynamic_per_token_scaled_quant_kernel", [&]
        { using input_dtype= typename t2ck<scalar_t>::type;
    aiter::dynamic_per_token_scaled_quant_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<ck_tile::int8_t *>(out.data_ptr()), scales.data_ptr<float>(),
            reinterpret_cast<input_dtype *>(input.data_ptr()),
            scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr,
            cols); });
  }
  else
  {
    TORCH_CHECK(false, __func__, "Unsupported output type for dynamic_per_token_scaled_quant");
  }
}
