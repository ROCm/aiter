// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "quant_common.cuh"
#include "dispatch_utils.h"

#include <c10/cuda/CUDAGuard.h>

#include <hipcub/hipcub.hpp>
#include "vec_convert.h"
#include "rocprim/rocprim.hpp"

const int32_t BlockSize = 256;
const int32_t groupQuantBlockSize = 64;
const int32_t thread_data_size = 32;

namespace aiter
{
  template <typename T, typename F>
  __device__ constexpr T multithread_reduce(T data, F reduce_op, int stage)
  {
    if (stage == 1)
    {
      return data;
    }
    if (stage >= 2)
    {
      data = reduce_op(rocprim::detail::warp_move_dpp<T, 0xb1>(data), data);
    }
    if (stage == 4)
    {
      data = reduce_op(rocprim::detail::warp_move_dpp<T, 0x4e>(data), data);
    }
    return data;
  }

  template <typename DTYPE_I, typename DTYPE_O, int thread_data_size = 32>
  __global__ void dynamic_per_group_scaled_quant_kernel(
      DTYPE_O *__restrict__ out, float *__restrict__ scale,
      DTYPE_I const *__restrict__ input, float const *__restrict__ scale_ub,
      const int32_t rows, const int32_t cols)
  {
    int num_thread_per_group = cols / thread_data_size;
    int64_t row_offset = blockIdx.x * groupQuantBlockSize;
    int64_t groupId = (row_offset + threadIdx.x) / num_thread_per_group;
    if (groupId > rows)
      return;
    row_offset *= thread_data_size;
    using vec_i = ck_tile::vec_t<DTYPE_I, thread_data_size>;
    static constexpr int32_t vec_size_o = std::is_same_v<DTYPE_O, ck_tile::fp4x2_t> ? thread_data_size / 2 : thread_data_size;
    using vec_o = ck_tile::vec_t<DTYPE_O, vec_size_o>;
    const float inverted_DTYPE_MAX = std::is_same_v<DTYPE_O, ck_tile::fp4x2_t> ? 0.25 : (1. / ck_tile::type_convert<float>(ck_tile::numeric<DTYPE_O>::max()));

    auto const *input_vecs = reinterpret_cast<vec_i const *>(input + row_offset);
    vec_i thread_data = input_vecs[threadIdx.x];
    float absMax = 0.f;
    for (size_t j = 0; j < thread_data_size; j++)
    {
      absMax = max(absMax, abs(ck_tile::type_convert<float>(thread_data[j])));
    }
    absMax = multithread_reduce(absMax, hipcub::Max(), num_thread_per_group);

    auto fp4_scale = [](float tmp)
    {uint32_t u32= ck_tile::bit_cast<uint32_t>(tmp);
      uint32_t exponent = (u32 >> 23) & 0b11111111;
      if (exponent == 0b11111111)
      {
        return ck_tile::bit_cast<float>(exponent<<23);
      }
      if (((u32 & 0x400000)) && (((u32 & 0x200000)) || ((u32 & 0x1FFFFF)) || (exponent)))
        exponent+=1;
      return ck_tile::bit_cast<float>(exponent << 23); };
    float inverted_scale = std::is_same_v<DTYPE_O, ck_tile::fp4x2_t> ? fp4_scale(absMax) * inverted_DTYPE_MAX : absMax * inverted_DTYPE_MAX;

    if (threadIdx.x % num_thread_per_group == 0)
    {
      if constexpr (std::is_same_v<DTYPE_O, ck_tile::fp4x2_t>)
      {
        auto *tmp = reinterpret_cast<uint8_t *>(scale);
        uint8_t exponent = (ck_tile::bit_cast<uint32_t>(inverted_scale) >> 23) & 0b11111111;
        tmp[groupId] = exponent;
      }
      else
      {
        scale[groupId] = inverted_scale;
      }
    }
    inverted_scale = std::is_same_v<DTYPE_O, ck_tile::fp4x2_t> ? inverted_scale : 1.0f / inverted_scale;

    auto *out_ptr = reinterpret_cast<DTYPE_O *>(out);
    auto *out_vecs = std::is_same_v<DTYPE_O, ck_tile::fp4x2_t> ? reinterpret_cast<vec_o *>(out + row_offset / 2) : reinterpret_cast<vec_o *>(out + row_offset);

    out_vecs[threadIdx.x] = ck_tile::vec_convert<DTYPE_O, DTYPE_I, thread_data_size>(thread_data, inverted_scale);
  }

  template <typename DTYPE_I, typename DTYPE_O>
  __device__ float data_to_per_row_scale(const DTYPE_I *__restrict__ input,
                                         int cols)
  {
    static constexpr int32_t vec_size_i = 16 / sizeof(DTYPE_O);
    static constexpr int32_t vec_size_o = std::is_same_v<DTYPE_O, ck_tile::fp4x2_t> ? vec_size_i / 2 : vec_size_i;
    using vec_i = ck_tile::vec_t<DTYPE_I, vec_size_i>;
    using tb_i = ck_tile::thread_buffer<DTYPE_I, vec_size_o>;
    const float inverted_DTYPE_MAX = std::is_same_v<DTYPE_O, ck_tile::fp4x2_t> ? 0.25 : (1. / ck_tile::type_convert<float>(ck_tile::numeric<DTYPE_O>::max()));

    const int64_t row_offset = blockIdx.x * cols;
    auto const *input_ptr = reinterpret_cast<DTYPE_I const *>(input);
    auto const *input_vecs = reinterpret_cast<vec_i const *>(input + row_offset);
    auto buffer_i = ck_tile::make_buffer_view<ck_tile::address_space_enum::global>(input_ptr, gridDim.x * cols);
    buffer_i.init_raw();

    // double load core loop start
    const int32_t num_elems_tail = cols % vec_size_i;
    const int32_t num_vecs = cols / vec_size_i;
    // const int32_t num_vecs = (cols + vec_size_i - 1) / vec_size_i * vec_size_i;
    vec_i vec_nxt;
    vec_i vec_cur;
    // size_t vec_idx = threadIdx.x * vec_size_i;
    // size_t vec_stride = BlockSize * vec_size_i;
    size_t vec_idx = threadIdx.x;
    size_t vec_stride = BlockSize;
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
      for (size_t j = 0; j < vec_size_i; j++)
      {
        absMax = max(absMax, abs(ck_tile::type_convert<float>(vec_cur[j])));
      }
      vec_cur = vec_nxt;
    }
    if (vec_idx - vec_stride < num_vecs)
    {
      for (size_t j = 0; j < vec_size_i; j++)
      {
        absMax = max(absMax, abs(ck_tile::type_convert<float>(vec_cur[j])));
      }
    }
    // double load core loop end

    // tail elements
    if (num_elems_tail > 0)
    {
      auto *tmp_i = reinterpret_cast<DTYPE_I const *>(input_vecs + num_vecs);
      for (size_t j = threadIdx.x; j < num_elems_tail; j += BlockSize)
      {
        absMax = max(absMax, abs(ck_tile::type_convert<float>(tmp_i[j])));
      }
    }

    using BlockReduce = hipcub::BlockReduce<float, BlockSize>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    absMax = BlockReduce(temp_storage).Reduce(absMax, hipcub::Max());

    auto fp4_scale = [](float tmp)
    {uint32_t u32= ck_tile::bit_cast<uint32_t>(tmp);
      uint32_t exponent = (u32 >> 23) & 0b11111111;
      if (exponent == 0b11111111)
      {
        return ck_tile::bit_cast<float>(exponent<<23);
      }
      if (((u32 & 0x400000)) && (((u32 & 0x200000)) || ((u32 & 0x1FFFFF)) || (exponent)))
        exponent+=1;
      return ck_tile::bit_cast<float>(exponent << 23); };
    return std::is_same_v<DTYPE_O, ck_tile::fp4x2_t> ? fp4_scale(absMax) * inverted_DTYPE_MAX : absMax * inverted_DTYPE_MAX;
  }

  template <typename DTYPE_I, typename DTYPE_O>
  __global__ void data_to_scale_kernel(float *__restrict__ scale,
                                       const DTYPE_I *__restrict__ input,
                                       int cols)
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
                                    int cols)
  {

    const float inverted_scale = std::is_same_v<DTYPE_O, ck_tile::fp4x2_t> ? (*scale) : 1.0f / (*scale);
    static constexpr int32_t vec_size_i = 16 / sizeof(DTYPE_O);
    static constexpr int32_t vec_size_o = std::is_same_v<DTYPE_O, ck_tile::fp4x2_t> ? vec_size_i / 2 : vec_size_i;

    using vec_i = ck_tile::vec_t<DTYPE_I, vec_size_i>;
    using tb_i = ck_tile::thread_buffer<DTYPE_I, vec_size_i>;
    using vec_o = ck_tile::vec_t<DTYPE_O, vec_size_o>;
    using tb_o = ck_tile::thread_buffer<DTYPE_O, vec_size_o>;

    const int64_t row_offset = blockIdx.x * cols;
    auto const *input_ptr = reinterpret_cast<DTYPE_I const *>(input);
    auto const *input_vecs = reinterpret_cast<vec_i const *>(input + row_offset);
    auto *out_ptr = reinterpret_cast<DTYPE_O *>(out);
    auto *out_vecs = std::is_same_v<DTYPE_O, ck_tile::fp4x2_t> ? reinterpret_cast<vec_o *>(out + row_offset / 2) : reinterpret_cast<vec_o *>(out + row_offset);

    // auto buffer_i = ck_tile::make_buffer_view<ck_tile::address_space_enum::global, ck_tile::amd_buffer_coherence_enum::glc>(input_ptr, gridDim.x * cols);
    // buffer_i.init_raw();
    // auto buffer_o = ck_tile::make_buffer_view<ck_tile::address_space_enum::global>(out_ptr, gridDim.x * cols);
    // buffer_o.init_raw();

    // double load core loop start
    // const int32_t num_vecs = (cols + vec_size_i - 1) / vec_size_i * vec_size_i;
    const int32_t num_elems_tail = cols % vec_size_i;
    const int32_t num_vecs = cols / vec_size_i;
    const int32_t tail_thread = num_vecs % BlockSize;
    vec_i vec_nxt;
    vec_i vec_cur;
    // size_t vec_idx = threadIdx.x * vec_size_i;
    // size_t vec_stride = BlockSize * vec_size_i;
    size_t vec_idx = threadIdx.x;
    size_t vec_stride = BlockSize;
    if (vec_idx < num_vecs)
    {
      // vec_cur = ck_tile::bit_cast<vec_i>(buffer_i.template get<tb_i>(vec_idx, row_offset, true));
      vec_cur = input_vecs[vec_idx];
    }

    for (vec_idx += vec_stride; vec_idx < num_vecs; vec_idx += vec_stride)
    {
      // vec_nxt = ck_tile::bit_cast<vec_i>(buffer_i.template get<tb_i>(vec_idx, row_offset, true));
      // buffer_o.template set<tb_o>((vec_idx - vec_stride), row_offset, true, ck_tile::bit_cast<tb_o>(ck_tile::vec_convert<DTYPE_O, DTYPE_I, vec_size_i>(vec_cur, inverted_scale)));
      vec_nxt = input_vecs[vec_idx];
      out_vecs[vec_idx - vec_stride] = ck_tile::vec_convert<DTYPE_O, DTYPE_I, vec_size_i>(vec_cur, inverted_scale);
      vec_cur = vec_nxt;
    }

    if (vec_idx - vec_stride < num_vecs)
    {
      // buffer_o.template set<tb_o>((vec_idx - vec_stride), row_offset, true, ck_tile::bit_cast<tb_o>(ck_tile::vec_convert<DTYPE_O, DTYPE_I, vec_size_i>(vec_cur, inverted_scale)));
      out_vecs[vec_idx - vec_stride] = ck_tile::vec_convert<DTYPE_O, DTYPE_I, vec_size_i>(vec_cur, inverted_scale);
    }
    // double load core loop end

    // tail elements
    if (num_elems_tail > 0)
    {
      auto *out_ptr2 = (out + row_offset);
      auto *tmp_i = reinterpret_cast<DTYPE_I const *>(input_vecs + num_vecs);
      for (size_t j = threadIdx.x; j < num_elems_tail; j += BlockSize)
      {
        out_ptr2[num_vecs * vec_size_i + j] =
            ck_tile::type_convert<DTYPE_O>(ck_tile::type_convert<float>(tmp_i[j]) * inverted_scale);
      }
    }
  }
  template <typename DTYPE_I, typename DTYPE_O>
  __global__ void scaled_quant_kernel(DTYPE_O *__restrict__ out,
                                      const DTYPE_I *__restrict__ input,
                                      const float *__restrict__ scale,
                                      int cols)
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

    const int token_idx = blockIdx.x;
    float row_scale = data_to_per_row_scale<DTYPE_I, DTYPE_O>(input, cols);

    __shared__ float token_scale;
    if (threadIdx.x == 0)
    {
      token_scale = row_scale;
      if (std::is_same_v<DTYPE_O, ck_tile::fp4x2_t>)
      {
        // scale[token_idx] = token_scale;
        auto *tmp = reinterpret_cast<uint8_t *>(scale);
        uint8_t exponent = (ck_tile::bit_cast<uint32_t>(token_scale) >> 23) & 0b11111111;
        tmp[token_idx] = exponent;
      }
      else
      {
        scale[token_idx] = token_scale;
      }
    }
    __syncthreads();

    scaled_quant_impl<DTYPE_I>(out, input, &token_scale, cols);
  }

void static_per_tensor_quant(torch::Tensor &out,         // [..., d]
                             torch::Tensor const &input, // [..., d]
                             torch::Tensor const &scale) // [1]
{
  int cols = input.size(-1);
  int rows = input.numel() / cols;
  dim3 grid(rows);
  dim3 block(BlockSize);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AITER_DISPATCH_FLOATING16_TYPES(
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
  int cols = input.size(-1);
  int rows = input.numel() / cols;
  dim3 grid(rows);
  dim3 block(BlockSize);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AITER_DISPATCH_FLOATING16_TYPES(
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

  int const cols = input.size(-1);
  int const rows = input.numel() / cols;
  

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (cols == 32 || cols == 64 || cols == 128)
  {
    int num_thread_per_group = cols / thread_data_size;
    int num_group_per_tg = groupQuantBlockSize / num_thread_per_group;
    dim3 const grid((rows + num_group_per_tg -1)/num_group_per_tg);
    dim3 const block(groupQuantBlockSize);
    if (out.dtype() == torch_fp8)
    {
      AITER_DISPATCH_FLOATING16_TYPES(
          input.scalar_type(), "dynamic_per_group_scaled_quant_kernel", [&]
          { using input_dtype= typename t2ck<scalar_t>::type;
        aiter::dynamic_per_group_scaled_quant_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<FP8_TYPE *>(out.data_ptr()), scales.data_ptr<float>(),
                reinterpret_cast<input_dtype*>(input.data_ptr()),
                scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr,
                rows, cols); });
    }
    else if (out.dtype() == torch::kInt8)
    {
      AITER_DISPATCH_FLOATING16_TYPES(
          input.scalar_type(), "dynamic_per_group_scaled_quant_kernel", [&]
          { using input_dtype= typename t2ck<scalar_t>::type;
        aiter::dynamic_per_group_scaled_quant_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<ck_tile::int8_t *>(out.data_ptr()), scales.data_ptr<float>(),
                reinterpret_cast<input_dtype *>(input.data_ptr()),
                scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr,
                rows, cols); });
    }
#if defined(__Float4_e2m1fn_x2)
    else if (out.dtype() == torch::kFloat4_e2m1fn_x2 || out.dtype() == torch::kUInt8)
    {
      AITER_DISPATCH_FLOATING16_TYPES(
          input.scalar_type(), "dynamic_per_group_scaled_quant_kernel", [&]
          { using input_dtype= typename t2ck<scalar_t>::type;
        aiter::dynamic_per_group_scaled_quant_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<ck_tile::fp4x2_t *>(out.data_ptr()), 
            reinterpret_cast<float *>(scales.data_ptr()),
            reinterpret_cast<input_dtype *>(input.data_ptr()),
            scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr,
            rows, cols); });
    }
#endif
    else
    {
      TORCH_CHECK(false, __func__, " not support output type: ", out.dtype());
    }
  }
  else
  {
    dim3 const grid(rows);
    dim3 const block(BlockSize);
    if (out.dtype() == torch_fp8)
    {
      AITER_DISPATCH_FLOATING16_TYPES(
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
      AITER_DISPATCH_FLOATING16_TYPES(
          input.scalar_type(), "dynamic_per_token_scaled_quant_kernel", [&]
          { using input_dtype= typename t2ck<scalar_t>::type;
        aiter::dynamic_per_token_scaled_quant_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<ck_tile::int8_t *>(out.data_ptr()), scales.data_ptr<float>(),
                reinterpret_cast<input_dtype *>(input.data_ptr()),
                scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr,
                cols); });
    }
#if defined(__Float4_e2m1fn_x2)
    else if (out.dtype() == torch::kFloat4_e2m1fn_x2 || out.dtype() == torch::kUInt8)
    {
      AITER_DISPATCH_FLOATING16_TYPES(
          input.scalar_type(), "dynamic_per_token_scaled_quant_kernel", [&]
          { using input_dtype= typename t2ck<scalar_t>::type;
        aiter::dynamic_per_token_scaled_quant_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<ck_tile::fp4x2_t *>(out.data_ptr()), 
            reinterpret_cast<float *>(scales.data_ptr()),
            reinterpret_cast<input_dtype *>(input.data_ptr()),
            scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr,
            cols); });
    }
#endif
    else
    {
      TORCH_CHECK(false, __func__, " not support output type: ", out.dtype());
    }
  }
} 
} // namespace aiter

