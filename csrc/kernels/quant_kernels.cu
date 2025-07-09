// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "dispatch_utils.h"
#include "hip_reduce.h"
#include "quant_common.cuh"
#include "rocprim/rocprim.hpp"
#include "vec_convert.h"
#include <c10/cuda/CUDAGuard.h>
#include <hipcub/hipcub.hpp>

const int32_t BlockSize           = 256;
const int32_t groupQuantBlockSize = 64;

namespace aiter {
template <typename DTYPE_I, typename DTYPE_O, int thread_data_size = 32>
__global__ void dynamic_per_group_scaled_quant_kernel(DTYPE_O* __restrict__ out,
                                                      float* __restrict__ scale,
                                                      DTYPE_I const* __restrict__ input,
                                                      float const* __restrict__ scale_ub,
                                                      const int32_t group_size,
                                                      int32_t ori_rows,
                                                      int32_t ori_cols,
                                                      int32_t ori_row_stride,
                                                      bool shuffle_scale = true)
{
    auto fp4_scale_shuffle_id = [](int32_t scaleN_pad, int32_t x, int32_t y) {
        return (x / 32 * scaleN_pad) * 32 + (y / 8) * 256 + (y % 4) * 64 + (x % 16) * 4 +
               (y % 8) / 4 * 2 + (x % 32) / 16;
    };

    int num_thread_per_group = group_size / thread_data_size;
    int64_t row_offset       = blockIdx.x * groupQuantBlockSize;
    int64_t groupId          = (row_offset + threadIdx.x) / num_thread_per_group;
    int32_t scaleN     = ori_cols / group_size;
    int32_t scaleN_pad = (std::is_same_v<DTYPE_O, ck_tile::fp4x2_t> && shuffle_scale)
                             ? (((scaleN + 7) / 8) * 8)
                             : scaleN;
    int32_t x          = groupId / scaleN_pad;
    int32_t y          = groupId % scaleN_pad;
    if constexpr(std::is_same_v<DTYPE_O, ck_tile::fp4x2_t>)
    {
        if(x >= ori_rows || y >= scaleN)
        {
            // if (shuffle_scale && threadIdx.x % num_thread_per_group == 0)
            // {
            //   auto *tmp = reinterpret_cast<uint8_t *>(scale);
            //   groupId = fp4_scale_shuffle_id(scaleN_pad, x, y);
            //   tmp[groupId] = 0x7f;
            // }
            return;
        }
    }
    else
    {
        if(x >= ori_rows)
            return;
    }

    row_offset  = x * ori_row_stride + y * group_size;
    using vec_i = ck_tile::vec_t<DTYPE_I, thread_data_size>;
    static constexpr int32_t vec_size_o =
        std::is_same_v<DTYPE_O, ck_tile::fp4x2_t> ? thread_data_size / 2 : thread_data_size;
    using vec_o = ck_tile::vec_t<DTYPE_O, vec_size_o>;
    const float inverted_DTYPE_MAX =
        std::is_same_v<DTYPE_O, ck_tile::fp4x2_t>
            ? 0.25
            : (1. / ck_tile::type_convert<float>(ck_tile::numeric<DTYPE_O>::max()));

    // static constexpr int32_t ooba_i = 4 / sizeof(DTYPE_I);
    static constexpr int32_t ooba_o = 4 / sizeof(DTYPE_O);
    // const int32_t oob_i             = (cols + ooba_i - 1) / ooba_i * ooba_i;
    const int64_t oob_o = (ori_rows * ori_cols + ooba_o - 1) / ooba_o * ooba_o;
    // auto buffer_i = ck_tile::make_buffer_view<ck_tile::address_space_enum::global>(input +
    // row_offset, oob_i); buffer_i.init_raw();

    auto const* input_vecs = reinterpret_cast<vec_i const*>(input + row_offset);
    // vec_i thread_data      = buffer_i.template get<vec_i>(vec_idx * vec_size_i, 0, true);
    vec_i thread_data = input_vecs[threadIdx.x % num_thread_per_group];
    float absMax      = 0.f;
    for(size_t j = 0; j < thread_data_size; j++)
    {
        absMax = max(absMax, abs(ck_tile::type_convert<float>(thread_data[j])));
    }
    absMax = multithread_reduce(absMax, hipcub::Max(), num_thread_per_group);

    auto fp4_scale = [](float tmp) {
        uint32_t u32      = ck_tile::bit_cast<uint32_t>(tmp);
        uint32_t exponent = (u32 >> 23) & 0b11111111;
        if(exponent == 0b11111111)
        {
            return ck_tile::bit_cast<float>(exponent << 23);
        }
        if(((u32 & 0x400000)) && (((u32 & 0x200000)) || ((u32 & 0x1FFFFF)) || (exponent)))
            exponent += 1;
        return ck_tile::bit_cast<float>(exponent << 23);
    };
    float inverted_scale = std::is_same_v<DTYPE_O, ck_tile::fp4x2_t>
                               ? fp4_scale(absMax) * inverted_DTYPE_MAX
                               : absMax * inverted_DTYPE_MAX;
    row_offset =
        std::is_same_v<DTYPE_O, ck_tile::fp4x2_t>
            ? groupId * group_size / 2 + (threadIdx.x % num_thread_per_group) * vec_size_o
            : groupId * group_size + (threadIdx.x % num_thread_per_group) * vec_size_o;
    if(threadIdx.x % num_thread_per_group == 0)
    {
        if constexpr(std::is_same_v<DTYPE_O, ck_tile::fp4x2_t>)
        {
            auto* tmp        = reinterpret_cast<uint8_t*>(scale);
            uint8_t exponent = (ck_tile::bit_cast<uint32_t>(inverted_scale) >> 23) & 0b11111111;
            if(shuffle_scale)
            {
                groupId = fp4_scale_shuffle_id(scaleN_pad, x, y);
            }
            tmp[groupId] = exponent;
        }
        else
        {
            scale[groupId] = inverted_scale;
        }
    }
    inverted_scale =
        std::is_same_v<DTYPE_O, ck_tile::fp4x2_t> ? inverted_scale : 1.0f / inverted_scale;

    using DTYPE_STORE = typename ck_tile::vector_traits<DTYPE_O>::scalar_type;
    auto* out_ptr     = reinterpret_cast<DTYPE_STORE*>(out);
    auto buffer_o = ck_tile::make_buffer_view<ck_tile::address_space_enum::global, ck_tile::amd_buffer_coherence_enum::glc>(out_ptr, oob_o);
    buffer_o.init_raw();

    auto out_s =
        ck_tile::vec_convert<DTYPE_O, DTYPE_I, thread_data_size>(thread_data, inverted_scale)
            .template get_as<DTYPE_STORE>();
    if constexpr(thread_data_size <= 16)
    {
        buffer_o.template set(row_offset, 0, true, out_s);
    }
    else
    {
        static constexpr int32_t o_step = std::is_same_v<DTYPE_O, ck_tile::fp4x2_t> ? 8 : 16;
        assert(thread_data_size % 16 == 0);
        using vecT                        = ck_tile::vec_t<DTYPE_STORE, o_step>;
        auto vec                          = out_s.template get_as<vecT>();
        static constexpr int32_t num_iter = thread_data_size / 16;

        for(size_t j = 0; j < num_iter; j++)
        {
            buffer_o.template set(row_offset + j * o_step, 0, true, vec[j]);
        }
    }
}

template <typename DTYPE_I, typename DTYPE_O, int thread_data_size = 16>
__device__ std::tuple<float, DTYPE_I*> data_to_per_row_scale(const DTYPE_I* __restrict__ input,
                                                             const int32_t cols)
{
    static constexpr int32_t vec_size_i =
        thread_data_size == 0 ? 16 / sizeof(DTYPE_O) : thread_data_size;
    static constexpr int32_t vec_size_o =
        std::is_same_v<DTYPE_O, ck_tile::fp4x2_t> ? vec_size_i / 2 : vec_size_i;
    using vec_i = ck_tile::vec_t<DTYPE_I, vec_size_i>;
    const float inverted_DTYPE_MAX =
        std::is_same_v<DTYPE_O, ck_tile::fp4x2_t>
            ? 0.25
            : (1. / ck_tile::type_convert<float>(ck_tile::numeric<DTYPE_O>::max()));

    const int64_t row_offset        = blockIdx.x * cols;
    auto const* ptr_i               = reinterpret_cast<DTYPE_I const*>(input + row_offset);
    auto const* input_vecs          = reinterpret_cast<vec_i const*>(ptr_i);
    static constexpr int32_t ooba_i = 4 / sizeof(DTYPE_I);
    const int32_t oob_i             = (cols + ooba_i - 1) / ooba_i * ooba_i;
    auto buffer_i = ck_tile::make_buffer_view<ck_tile::address_space_enum::global>(ptr_i, oob_i);
    buffer_i.init_raw();

    // double load core loop start
    const int32_t num_elems_tail = cols % vec_size_i;
    const int32_t num_vecs       = (cols + vec_size_i - 1) / vec_size_i;

    vec_i vec_cur;
    size_t vec_idx    = threadIdx.x;
    size_t vec_stride = BlockSize;
    if(vec_idx < num_vecs)
    {
        vec_cur = buffer_i.template get<vec_i>(vec_idx * vec_size_i, 0, true);
    }

    float absMax = 0.f;
    if constexpr(thread_data_size == 0)
    {
        vec_i vec_nxt;
        for(vec_idx += vec_stride; vec_idx < num_vecs; vec_idx += vec_stride)
        {
            vec_nxt = buffer_i.template get<vec_i>(vec_idx * vec_size_i, 0, true);
            for(size_t j = 0; j < vec_size_i; j++)
            {
                absMax = max(absMax, abs(ck_tile::type_convert<float>(vec_cur[j])));
            }
            vec_cur = vec_nxt;
        }
        vec_idx -= vec_stride;
    }
    if(vec_idx < num_vecs)
    {
#pragma unroll
        for(size_t j = 0; j < vec_size_i; j++)
        {
            absMax = max(absMax, abs(ck_tile::type_convert<float>(vec_cur[j])));
        }
    }
    // double load core loop end

    // using BlockReduce = hipcub::BlockReduce<float, BlockSize>;
    // __shared__ typename BlockReduce::TempStorage temp_storage;
    // absMax = BlockReduce(temp_storage).Reduce(absMax, hipcub::Max());
    absMax = block_reduce<float, hipcub::Max, BlockSize, true>(absMax, hipcub::Max());

    auto fp4_scale = [](float tmp) {
        uint32_t u32      = ck_tile::bit_cast<uint32_t>(tmp);
        uint32_t exponent = (u32 >> 23) & 0b11111111;
        if(exponent == 0b11111111)
        {
            return ck_tile::bit_cast<float>(exponent << 23);
        }
        if(((u32 & 0x400000)) && (((u32 & 0x200000)) || ((u32 & 0x1FFFFF)) || (exponent)))
            exponent += 1;
        return ck_tile::bit_cast<float>(exponent << 23);
    };
    float row_scale = std::is_same_v<DTYPE_O, ck_tile::fp4x2_t>
                          ? fp4_scale(absMax) * inverted_DTYPE_MAX
                          : absMax * inverted_DTYPE_MAX;
    return std::make_tuple(row_scale, reinterpret_cast<DTYPE_I*>(&vec_cur));
}

template <typename DTYPE_I, typename DTYPE_O>
__global__ void
data_to_scale_kernel(float* __restrict__ scale, const DTYPE_I* __restrict__ input, const int cols)
{
    auto res        = data_to_per_row_scale<DTYPE_I, DTYPE_O, 0>(input, cols);
    float row_scale = std::get<0>(res);
    if(threadIdx.x == 0)
    {
        vllm::atomicMaxFloat(scale, row_scale);
    }
}

template <typename DTYPE_I, typename DTYPE_O>
__device__ void scaled_quant_impl(DTYPE_O* __restrict__ out,
                                  const DTYPE_I* __restrict__ input,
                                  const float* __restrict__ scale,
                                  const int32_t cols)
{

    const float inverted_scale =
        std::is_same_v<DTYPE_O, ck_tile::fp4x2_t> ? (*scale) : 1.0f / (*scale);
    static constexpr int32_t vec_size_i = 16 / sizeof(DTYPE_O);
    static constexpr int32_t vec_size_o =
        std::is_same_v<DTYPE_O, ck_tile::fp4x2_t> ? vec_size_i / 2 : vec_size_i;

    using vec_i       = ck_tile::vec_t<DTYPE_I, vec_size_i>;
    using vec_o       = ck_tile::vec_t<DTYPE_O, vec_size_o>;
    using DTYPE_STORE = typename ck_tile::vector_traits<DTYPE_O>::scalar_type;

    const int64_t row_offset        = blockIdx.x * cols;
    auto const* ptr_i               = reinterpret_cast<DTYPE_I const*>(input + row_offset);
    auto const* input_vecs          = reinterpret_cast<vec_i const*>(ptr_i);
    auto* ptr_o                     = std::is_same_v<DTYPE_O, ck_tile::fp4x2_t>
                                          ? reinterpret_cast<DTYPE_STORE*>(out + row_offset / 2)
                                          : reinterpret_cast<DTYPE_STORE*>(out + row_offset);
    auto* out_vecs                  = reinterpret_cast<vec_o*>(ptr_o);
    static constexpr int32_t ooba_i = 4 / sizeof(DTYPE_I);
    static constexpr int32_t ooba_o = 4 / sizeof(DTYPE_O);
    const int32_t oob_i             = (cols + ooba_i - 1) / ooba_i * ooba_i;
    const int32_t oob_o             = (cols + ooba_o - 1) / ooba_o * ooba_o;

    auto buffer_i = ck_tile::make_buffer_view<ck_tile::address_space_enum::global>(ptr_i, oob_i);
    buffer_i.init_raw();
    auto buffer_o = ck_tile::make_buffer_view<ck_tile::address_space_enum::global>(ptr_o, oob_o);
    buffer_o.init_raw();

    // double load core loop start
    const int32_t num_elems_tail = cols % vec_size_i;
    const int32_t num_vecs       = (cols + vec_size_i - 1) / vec_size_i;
    const int32_t tail_thread    = num_vecs % BlockSize;
    vec_i vec_nxt;
    vec_i vec_cur;
    // size_t vec_idx = threadIdx.x * vec_size_i;
    // size_t vec_stride = BlockSize * vec_size_i;
    size_t vec_idx    = threadIdx.x;
    size_t vec_stride = BlockSize;
    if(vec_idx < num_vecs)
    {
        vec_cur = buffer_i.template get<vec_i>(vec_idx * vec_size_i, 0, true);
    }

    for(vec_idx += vec_stride; vec_idx < num_vecs; vec_idx += vec_stride)
    {
        vec_nxt = buffer_i.template get<vec_i>(vec_idx * vec_size_i, 0, true);
        buffer_o.template set(
            (vec_idx - vec_stride) * vec_size_o,
            0,
            true,
            ck_tile::vec_convert<DTYPE_O, DTYPE_I, vec_size_i>(vec_cur, inverted_scale)
                .template get_as<DTYPE_STORE>());
        vec_cur = vec_nxt;
    }

    if(vec_idx - vec_stride < num_vecs)
    {
        buffer_o.template set(
            (vec_idx - vec_stride) * vec_size_o,
            0,
            true,
            ck_tile::vec_convert<DTYPE_O, DTYPE_I, vec_size_i>(vec_cur, inverted_scale)
                .template get_as<DTYPE_STORE>());
    }
    // double load core loop end
}

template <typename DTYPE_I, typename DTYPE_O, int thread_data_size = 16>
__device__ void scaled_quant_vgpr_impl(DTYPE_O* __restrict__ out,
                                       DTYPE_I* __restrict__ input,
                                       const float* __restrict__ scale,
                                       const int cols)
{

    const float inverted_scale =
        std::is_same_v<DTYPE_O, ck_tile::fp4x2_t> ? (*scale) : 1.0f / (*scale);
    static constexpr int32_t vec_size_i = thread_data_size;
    static constexpr int32_t vec_size_o =
        std::is_same_v<DTYPE_O, ck_tile::fp4x2_t> ? vec_size_i / 2 : vec_size_i;

    using vec_i       = ck_tile::vec_t<DTYPE_I, vec_size_i>;
    using vec_o       = ck_tile::vec_t<DTYPE_O, vec_size_o>;
    using DTYPE_STORE = typename ck_tile::vector_traits<DTYPE_O>::scalar_type;

    const int64_t row_offset        = blockIdx.x * cols;
    auto const* ptr_i               = reinterpret_cast<DTYPE_I const*>(input);
    auto const* input_vecs          = reinterpret_cast<vec_i const*>(ptr_i);
    auto* out_ptr                   = reinterpret_cast<DTYPE_O*>(out);
    auto* ptr_o                     = std::is_same_v<DTYPE_O, ck_tile::fp4x2_t>
                                          ? reinterpret_cast<DTYPE_STORE*>(out + row_offset / 2)
                                          : reinterpret_cast<DTYPE_STORE*>(out + row_offset);
    static constexpr int32_t ooba_i = 4 / sizeof(DTYPE_I);
    static constexpr int32_t ooba_o = 4 / sizeof(DTYPE_O);
    const int32_t oob_i             = (cols + ooba_i - 1) / ooba_i * ooba_i;
    const int32_t oob_o             = (cols + ooba_o - 1) / ooba_o * ooba_o;

    auto buffer_o = ck_tile::make_buffer_view<ck_tile::address_space_enum::global>(ptr_o, oob_o);
    buffer_o.init_raw();

    const int32_t num_vecs = (cols + vec_size_i - 1) / vec_size_i;

    if(threadIdx.x < num_vecs)
    {
        auto out = ck_tile::vec_convert<DTYPE_O, DTYPE_I, vec_size_i>(*input_vecs, inverted_scale)
                       .template get_as<DTYPE_STORE>();
        if constexpr(vec_size_i <= 16)
        {

            buffer_o.template set(threadIdx.x * vec_size_o, 0, true, out);
        }
        else
        {
            static constexpr int32_t o_step = std::is_same_v<DTYPE_O, ck_tile::fp4x2_t> ? 8 : 16;
            assert(vec_size_i % 16 == 0);
            using vecT                        = ck_tile::vec_t<DTYPE_STORE, o_step>;
            auto vec                          = out.template get_as<vecT>();
            static constexpr int32_t num_iter = vec_size_i / 16;

            for(size_t j = 0; j < num_iter; j++)
            {
                buffer_o.template set(threadIdx.x * vec_size_o + j * o_step, 0, true, vec[j]);
            }
        }
    }
}

template <typename DTYPE_I, typename DTYPE_O>
__global__ void scaled_quant_kernel(DTYPE_O* __restrict__ out,
                                    const DTYPE_I* __restrict__ input,
                                    const float* __restrict__ scale,
                                    const int cols)
{
    scaled_quant_impl<DTYPE_I>(out, input, scale, cols);
}

template <typename DTYPE_I, typename DTYPE_O, int thread_data_size = 16>
__global__ void dynamic_per_token_scaled_quant_kernel(DTYPE_O* __restrict__ out,
                                                      float* __restrict__ scale,
                                                      DTYPE_I* __restrict__ input,
                                                      float const* __restrict__ scale_ub,
                                                      const int32_t cols)
{
    const int token_idx = blockIdx.x;
    auto res            = data_to_per_row_scale<DTYPE_I, DTYPE_O, thread_data_size>(input, cols);
    float row_scale     = std::get<0>(res);
    DTYPE_I* vec_ptr    = std::get<1>(res);

    // __shared__ float token_scale;
    if(threadIdx.x == 0)
    {
        // token_scale = row_scale;
        if constexpr(std::is_same_v<DTYPE_O, ck_tile::fp4x2_t>)
        {
            // scale[token_idx] = token_scale;
            auto* tmp        = reinterpret_cast<uint8_t*>(scale);
            uint8_t exponent = (ck_tile::bit_cast<uint32_t>(row_scale) >> 23) & 0b11111111;
            tmp[token_idx]   = exponent;
        }
        else
        {
            scale[token_idx] = row_scale;
        }
    }
    // __syncthreads();

    if constexpr(thread_data_size == 0)
    {
        scaled_quant_impl<DTYPE_I>(out, input, &row_scale, cols);
    }
    else
    {
        scaled_quant_vgpr_impl<DTYPE_I, DTYPE_O, thread_data_size>(out, vec_ptr, &row_scale, cols);
    }
}

void static_per_tensor_quant(torch::Tensor& out,         // [..., d]
                             torch::Tensor const& input, // [..., d]
                             torch::Tensor const& scale) // [1]
{
    const int cols = input.size(-1);
    int rows       = input.numel() / cols;
    dim3 grid(rows);
    dim3 block(BlockSize);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if(out.dtype() == torch_fp8)
    {
        AITER_DISPATCH_FLOATING16_TYPES(input.scalar_type(), "scaled_quant_kernel", [&] {
            using input_dtype = typename t2ck<scalar_t>::type;
            aiter::scaled_quant_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<FP8_TYPE*>(out.data_ptr()),
                reinterpret_cast<input_dtype*>(input.data_ptr()),
                scale.data_ptr<float>(),
                cols);
        });
    }
    else if(out.dtype() == torch::kInt8)
    {
        AITER_DISPATCH_FLOATING16_TYPES(input.scalar_type(), "scaled_quant_kernel", [&] {
            using input_dtype = typename t2ck<scalar_t>::type;
            aiter::scaled_quant_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<ck_tile::int8_t*>(out.data_ptr()),
                reinterpret_cast<input_dtype*>(input.data_ptr()),
                scale.data_ptr<float>(),
                cols);
        });
    }
    else
    {
        TORCH_CHECK(false, __func__, " not support output type: ", out.dtype());
    }
}

#define DYNAMIC_PER_TOKEN_SCALED_QUANT_KERNEL_IMPL(quant_kernel, DTYPE_O, THREAD_DATA)      \
    AITER_DISPATCH_FLOATING16_TYPES(input.scalar_type(), "quant_kernel", [&] {              \
        using input_dtype = typename t2ck<scalar_t>::type;                                  \
        aiter::quant_kernel<input_dtype, DTYPE_O, THREAD_DATA><<<grid, block, 0, stream>>>( \
            reinterpret_cast<DTYPE_O*>(out.data_ptr()),                                     \
            scales.data_ptr<float>(),                                                       \
            reinterpret_cast<input_dtype*>(input.data_ptr()),                               \
            scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr,                   \
            cols);                                                                          \
    });

#define DYNAMIC_PER_TOKEN_SCALED_QUANT_KERNEL_DISPATCH(quant_kernel, DTYPE_O, cols) \
    if(cols <= 8 * BlockSize)                                                       \
    {                                                                               \
        DYNAMIC_PER_TOKEN_SCALED_QUANT_KERNEL_IMPL(quant_kernel, DTYPE_O, 8)        \
    }                                                                               \
    else if(cols <= 16 * BlockSize)                                                 \
    {                                                                               \
        DYNAMIC_PER_TOKEN_SCALED_QUANT_KERNEL_IMPL(quant_kernel, DTYPE_O, 16)       \
    }                                                                               \
    else if(cols <= 32 * BlockSize)                                                 \
    {                                                                               \
        DYNAMIC_PER_TOKEN_SCALED_QUANT_KERNEL_IMPL(quant_kernel, DTYPE_O, 32)       \
    }                                                                               \
    else                                                                            \
    {                                                                               \
        DYNAMIC_PER_TOKEN_SCALED_QUANT_KERNEL_IMPL(quant_kernel, DTYPE_O, 0)        \
    }

void dynamic_per_tensor_quant(torch::Tensor& out,         // [..., d]
                              torch::Tensor const& input, // [..., d]
                              torch::Tensor& scale)       // [1]
{
    const int cols = input.size(-1);
    int rows       = input.numel() / cols;
    dim3 grid(rows);
    dim3 block(BlockSize);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if(out.dtype() == torch_fp8)
    {
        AITER_DISPATCH_FLOATING16_TYPES(input.scalar_type(), "scaled_quant_kernel", [&] {
            using input_dtype = typename t2ck<scalar_t>::type;
            vllm::initializeScale<<<dim3(1), dim3(64), 0, stream>>>(
                scale.data_ptr<float>(), 1, 0.0f);
            aiter::data_to_scale_kernel<input_dtype, FP8_TYPE><<<grid, block, 0, stream>>>(
                scale.data_ptr<float>(), reinterpret_cast<input_dtype*>(input.data_ptr()), cols);
            aiter::scaled_quant_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<FP8_TYPE*>(out.data_ptr()),
                reinterpret_cast<input_dtype*>(input.data_ptr()),
                scale.data_ptr<float>(),
                cols);
        });
    }
    else if(out.dtype() == torch::kInt8)
    {
        AITER_DISPATCH_FLOATING16_TYPES(input.scalar_type(), "scaled_quant_kernel", [&] {
            using input_dtype = typename t2ck<scalar_t>::type;
            vllm::initializeScale<<<dim3(1), dim3(64), 0, stream>>>(
                scale.data_ptr<float>(), 1, 0.0f);
            aiter::data_to_scale_kernel<input_dtype, ck_tile::int8_t><<<grid, block, 0, stream>>>(
                scale.data_ptr<float>(), reinterpret_cast<input_dtype*>(input.data_ptr()), cols);
            aiter::scaled_quant_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<ck_tile::int8_t*>(out.data_ptr()),
                reinterpret_cast<input_dtype*>(input.data_ptr()),
                scale.data_ptr<float>(),
                cols);
        });
    }
    else
    {
        TORCH_CHECK(false, __func__, " not support output type: ", out.dtype());
    }
}

void dynamic_per_token_scaled_quant(torch::Tensor& out,         // [..., d]
                                    torch::Tensor const& input, // [..., d]
                                    torch::Tensor& scales,
                                    std::optional<at::Tensor> const& scale_ub,
                                    bool shuffle_scale = true)
{
    TORCH_CHECK(input.is_contiguous());
    TORCH_CHECK(out.is_contiguous());

    int const cols = input.size(-1);
    int const rows = input.numel() / cols;

    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if(cols == 32 || cols == 64 || cols == 128)
    {
        int group_size           = cols;
        int thread_data_size     = 32;
        int num_thread_per_group = group_size / thread_data_size;
        int num_group_per_tg     = groupQuantBlockSize / num_thread_per_group;
        if(out.dtype() == torch_fp8)
        {
            int ori_cols  = cols;
            int ori_rows  = rows;
            int num_group = rows;
            dim3 const grid((num_group + num_group_per_tg - 1) / num_group_per_tg);
            dim3 const block(groupQuantBlockSize);
            AITER_DISPATCH_FLOATING16_TYPES(
                input.scalar_type(), "dynamic_per_group_scaled_quant_kernel", [&] {
                    using input_dtype = typename t2ck<scalar_t>::type;
                    aiter::dynamic_per_group_scaled_quant_kernel<<<grid, block, 0, stream>>>(
                        reinterpret_cast<FP8_TYPE*>(out.data_ptr()),
                        scales.data_ptr<float>(),
                        reinterpret_cast<input_dtype*>(input.data_ptr()),
                        scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr,
                        group_size,
                        ori_rows,
                        ori_cols,
                        ori_cols,
                        shuffle_scale);
                });
        }
        else if(out.dtype() == torch::kInt8)
        {
            int ori_cols  = cols;
            int ori_rows  = rows;
            int num_group = rows;
            dim3 const grid((num_group + num_group_per_tg - 1) / num_group_per_tg);
            dim3 const block(groupQuantBlockSize);
            AITER_DISPATCH_FLOATING16_TYPES(
                input.scalar_type(), "dynamic_per_group_scaled_quant_kernel", [&] {
                    using input_dtype = typename t2ck<scalar_t>::type;
                    aiter::dynamic_per_group_scaled_quant_kernel<<<grid, block, 0, stream>>>(
                        reinterpret_cast<ck_tile::int8_t*>(out.data_ptr()),
                        scales.data_ptr<float>(),
                        reinterpret_cast<input_dtype*>(input.data_ptr()),
                        scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr,
                        group_size,
                        ori_rows,
                        ori_cols,
                        ori_cols,
                        shuffle_scale);
                });
        }
#if defined(__Float4_e2m1fn_x2)
        else if(out.dtype() == torch::kFloat4_e2m1fn_x2 || out.dtype() == torch::kUInt8)
        {
            int ori_cols  = out.size(-1) * 2;
            int scaleN    = ori_cols / cols;
            int ori_rows  = rows / scaleN;
            int num_group = shuffle_scale ? ori_rows * ((scaleN + 7) / 8 * 8) : rows;
            // int num_group = shuffle_scale ? ((ori_rows + 255) / 256 * 256) * ((scaleN + 7) / 8 *
            // 8) : rows;
            dim3 const grid((num_group + num_group_per_tg - 1) / num_group_per_tg);
            dim3 const block(groupQuantBlockSize);
            AITER_DISPATCH_FLOATING16_TYPES(
                input.scalar_type(), "dynamic_per_group_scaled_quant_kernel", [&] {
                    using input_dtype = typename t2ck<scalar_t>::type;
                    aiter::dynamic_per_group_scaled_quant_kernel<<<grid, block, 0, stream>>>(
                        reinterpret_cast<ck_tile::fp4x2_t*>(out.data_ptr()),
                        reinterpret_cast<float*>(scales.data_ptr()),
                        reinterpret_cast<input_dtype*>(input.data_ptr()),
                        scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr,
                        group_size,
                        ori_rows,
                        ori_cols,
                        ori_cols,
                        shuffle_scale);
                });
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
        if(out.dtype() == torch_fp8)
        {
            DYNAMIC_PER_TOKEN_SCALED_QUANT_KERNEL_DISPATCH(
                dynamic_per_token_scaled_quant_kernel, FP8_TYPE, cols);
        }
        else if(out.dtype() == torch::kInt8)
        {
            DYNAMIC_PER_TOKEN_SCALED_QUANT_KERNEL_DISPATCH(
                dynamic_per_token_scaled_quant_kernel, ck_tile::int8_t, cols);
        }
#if defined(__Float4_e2m1fn_x2)
        else if(out.dtype() == torch::kFloat4_e2m1fn_x2 || out.dtype() == torch::kUInt8)
        {
            DYNAMIC_PER_TOKEN_SCALED_QUANT_KERNEL_DISPATCH(
                dynamic_per_token_scaled_quant_kernel, ck_tile::fp4x2_t, cols);
        }
#endif
        else
        {
            TORCH_CHECK(false, __func__, " not support output type: ", out.dtype());
        }
    }
}

void dynamic_per_group_scaled_quant_fp4(torch::Tensor& out,         // [..., d]
                                        torch::Tensor const& input, // [..., d]
                                        torch::Tensor& scales,
                                        int group_size     = 32,
                                        bool shuffle_scale = true)
{
    TORCH_CHECK(group_size == 32 || group_size == 64 || group_size == 128,
                __func__,
                " only support group_size [32, 64 , 128]");
    TORCH_CHECK(out.is_contiguous());

    int const cols       = input.size(-1);
    int const rows       = input.numel() / cols;
    int const row_stride = input.stride(-2);

    TORCH_CHECK(cols % group_size == 0, __func__, " cols is not divisible by group_size");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int thread_data_size     = 32;
    int num_thread_per_group = group_size / thread_data_size;
    int num_group_per_tg     = groupQuantBlockSize / num_thread_per_group;

    int scaleN    = cols / group_size;
    int num_group = shuffle_scale ? rows * ((scaleN + 7) / 8 * 8) : rows * scaleN;
    // int num_group = shuffle_scale ? ((rows + 255) / 256 * 256) * ((scaleN + 7) / 8 * 8) : rows *
    // scaleN;
    dim3 const grid((num_group + num_group_per_tg - 1) / num_group_per_tg);
    dim3 const block(groupQuantBlockSize);

#if defined(__Float4_e2m1fn_x2)
    AITER_DISPATCH_FLOATING16_TYPES(
        input.scalar_type(), "dynamic_per_group_scaled_quant_kernel", [&] {
            using input_dtype = typename t2ck<scalar_t>::type;
            aiter::dynamic_per_group_scaled_quant_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<ck_tile::fp4x2_t*>(out.data_ptr()),
                reinterpret_cast<float*>(scales.data_ptr()),
                reinterpret_cast<input_dtype*>(input.data_ptr()),
                nullptr,
                group_size,
                rows,
                cols,
                row_stride,
                shuffle_scale);
        });
#else
    TORCH_CHECK(false, __func__, " device not support Float4_e2m1fn_x2 dtype");
#endif
}
} // namespace aiter
