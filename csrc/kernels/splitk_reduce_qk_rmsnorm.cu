// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Reduce of a six-way split-K BF16 GEMM, fused with the q/kv RMSNorm of the
// MLA input projection (hidden -> q_lora 2048 | kv_lora 512 | rope 64).
//
//   partial [6, M, 2624] fp32: plane p holds the partial sum over K range p
//   out     [M, 2624]    bf16: bf16(p0 + 0 + p1 + p2 + p3 + p4 + p5)
//   q_out   [M, 2048]    bf16: RMSNorm(out[:, 0:2048])
//   k_out   [M, 512]     bf16: RMSNorm(out[:, 2048:2560])
//
// The planes are added left to right in fp32 and rounded to bf16 once, so the
// result does not depend on scheduling. out[:, 2560:2624] (the rotary key) is
// written but not normalized. The norm is fused_qk_rmsnorm's for this shape:
// one 256-thread block per row and per q/kv half, 8 elements per thread,
// applied to the bf16 values of out.

#include "aiter_hip_common.h"
#include "aiter_opus_plus.h"
#include "aiter_stream.h"
#include "splitk_reduce_qk_rmsnorm.h"

#include <sstream>
#include <stdexcept>
#include <utility>

namespace aiter {
namespace {

constexpr int kPlanes    = 6;
constexpr int kN         = 2624;
constexpr int kQN        = 2048;
constexpr int kKN        = 512;
constexpr int kBlockSize = 256;
constexpr int kVec       = 8;

using element = opus::bf16_t;
using vec_i   = opus::vector_t<element, kVec>;
using vec_f   = opus::vector_t<float, kVec>;
using vec2_f  = opus::vector_t<float, 2>;

// fp32 -> bf16, round to nearest even. NaN stays NaN.
__device__ __forceinline__ element float_to_bf16_rne(float value)
{
    unsigned int bits;
    __builtin_memcpy(&bits, &value, sizeof(bits));
    if((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16) & 1u);
    else if(bits & 0xffffu)
        bits |= 0x10000u;
    unsigned short upper = static_cast<unsigned short>(bits >> 16);
    element result;
    __builtin_memcpy(&result, &upper, sizeof(upper));
    return result;
}

// Sum of the six planes at 8 consecutive elements, in plane order.
__device__ __forceinline__ vec_i reduce_planes(const float* partial,
                                               int64_t plane_stride,
                                               int64_t offset)
{
    vec_f planes[kPlanes];
#pragma unroll
    for(int p = 0; p < kPlanes; ++p)
        planes[p] = *reinterpret_cast<const vec_f*>(partial + p * plane_stride + offset);
    vec_i result;
#pragma unroll
    for(int i = 0; i < kVec; ++i)
    {
        float sum;
        asm("v_add_f32 %0, %1, 0" : "=v"(sum) : "v"(planes[0][i]));
#pragma unroll
        for(int p = 1; p < kPlanes; ++p)
            asm("v_add_f32 %0, %1, %2" : "=v"(sum) : "v"(sum), "v"(planes[p][i]));
        result[i] = float_to_bf16_rne(sum);
    }
    return result;
}

// grid (M, 2): blockIdx.y 0 reduces and normalizes the q columns, 1 the kv
// columns plus the unnormalized rope columns.
__global__ void splitk_reduce_qk_rmsnorm_kernel(const float* __restrict__ partial,
                                                element* __restrict__ out,
                                                element* __restrict__ q_out,
                                                element* __restrict__ k_out,
                                                const element* __restrict__ q_weight,
                                                const element* __restrict__ k_weight,
                                                float q_epsilon,
                                                float k_epsilon)
{
    const int64_t row          = blockIdx.x;
    const int64_t plane_stride = static_cast<int64_t>(gridDim.x) * kN;
    const bool is_q            = blockIdx.y == 0;
    const int n                = is_q ? kQN : kKN;
    const int offset           = is_q ? 0 : kQN;
    const int column           = threadIdx.x * kVec;
    const element* weight_ptr  = is_q ? q_weight : k_weight;
    element* norm_ptr          = (is_q ? q_out : k_out) + row * n;
    const float epsilon        = is_q ? q_epsilon : k_epsilon;

    vec_i thread_data_i{};
    if(column < n)
    {
        thread_data_i = reduce_planes(partial, plane_stride, row * kN + offset + column);
        *reinterpret_cast<vec_i*>(out + row * kN + offset + column) = thread_data_i;
    }
    if(!is_q && threadIdx.x < (kN - kQN - kKN) / kVec)
    {
        const int rope_column = kQN + kKN + threadIdx.x * kVec;
        *reinterpret_cast<vec_i*>(out + row * kN + rope_column) =
            reduce_planes(partial, plane_stride, row * kN + rope_column);
    }

    auto weight_buffer = opus::make_gmem<element>(weight_ptr, n * sizeof(element));
    vec_i thread_data_weight =
        load_vector_nbytes<element, kVec, 16, RT, true, WARP_SIZE>(weight_buffer, column);
    vec_f thread_data_float;
    vec2_f rcp;
    for(int i = 0; i < kVec; ++i)
        thread_data_float[i] = opus::cast<float>(thread_data_i[i]);

    float square_sum = 0.0f;
    for(int i = 0; i < kVec; ++i)
        square_sum += thread_data_float[i] * thread_data_float[i];
    auto sum_f = [](float a, float b) { return a + b; };
    rcp[0]     = block_reduce<float, decltype(sum_f), kBlockSize, true>(square_sum, sum_f);
    rcp[0]     = rsqrtf(rcp[0] / n + epsilon);
    rcp[1]     = rcp[0];

    vec2_f* thread_data_float2 = reinterpret_cast<vec2_f*>(&thread_data_float);
    for(int i = 0; i < kVec / 2; ++i)
        asm volatile("v_pk_mul_f32 %0, %1, %2"
                     : "=v"(thread_data_float2[i])
                     : "v"(thread_data_float2[i]), "v"(rcp));
    for(int i = 0; i < kVec / 2; ++i)
    {
        rcp[0] = static_cast<float>(thread_data_weight[2 * i]);
        rcp[1] = static_cast<float>(thread_data_weight[2 * i + 1]);
        asm volatile("v_pk_mul_f32 %0, %1, %2"
                     : "=v"(thread_data_float2[i])
                     : "v"(thread_data_float2[i]), "v"(rcp));
    }
    auto norm_buffer = opus::make_gmem<element>(norm_ptr, n * sizeof(element));
    store_vector<element, float, kVec, RT, true, WARP_SIZE, 1, element>(
        norm_buffer, thread_data_float, column);
}

// This is a pybind API: invalid operands must raise a Python exception.
// AITER_CHECK is fatal unless a separate entry point enables its thread-local
// throwing mode. Do not change that global mode for this operator.
template <typename... Args>
void check(bool condition, Args&&... args)
{
    if(!condition)
    {
        std::ostringstream message;
        (message << ... << std::forward<Args>(args));
        throw std::runtime_error(message.str());
    }
}

void check_device_and_alignment(const aiter_tensor_t& t,
                                int device_id,
                                size_t alignment,
                                const char* name)
{
    check(t.is_gpu(), name, " must be a GPU tensor");
    check(t.device_id == device_id, "all tensors must be on the same GPU");
    check(reinterpret_cast<uintptr_t>(t.data_ptr()) % alignment == 0,
          name,
          " must be ",
          alignment,
          "-byte aligned");
}

void check_no_overlap(const aiter_tensor_t& a, const aiter_tensor_t& b)
{
    const auto a_begin = reinterpret_cast<uintptr_t>(a.data_ptr());
    const auto b_begin = reinterpret_cast<uintptr_t>(b.data_ptr());
    const auto a_end   = a_begin + a.numel() * sizeof(element);
    const auto b_end =
        b_begin + b.numel() * (b.dtype() == AITER_DTYPE_fp32 ? sizeof(float) : sizeof(element));
    check(a_begin >= b_end || b_begin >= a_end, "outputs must not overlap inputs or other outputs");
}

void check_bf16_matrix(const aiter_tensor_t& t, int64_t rows, int64_t cols, const char* name)
{
    check(t.dtype() == AITER_DTYPE_bf16, name, " must be bf16");
    check(t.dim() == 2 && t.size(0) == rows && t.size(1) == cols,
          name,
          " must be [",
          rows,
          ", ",
          cols,
          "]");
    check(t.is_contiguous(), name, " must be contiguous");
}

void check_bf16_vector(const aiter_tensor_t& t, int64_t size, const char* name)
{
    check(t.dtype() == AITER_DTYPE_bf16, name, " must be bf16");
    check(t.dim() == 1 && t.size(0) == size, name, " must be [", size, "]");
    check(t.is_contiguous(), name, " must be contiguous");
}

} // namespace

void splitk_reduce_qk_rmsnorm(const aiter_tensor_t& partial,
                              aiter_tensor_t& out,
                              aiter_tensor_t& q_out,
                              aiter_tensor_t& k_out,
                              const aiter_tensor_t& q_weight,
                              const aiter_tensor_t& k_weight,
                              double q_eps,
                              double k_eps)
{
    check_device_and_alignment(partial, partial.device_id, 32, "partial");
    check_device_and_alignment(out, partial.device_id, 16, "out");
    check_device_and_alignment(q_out, partial.device_id, 16, "q_out");
    check_device_and_alignment(k_out, partial.device_id, 16, "k_out");
    check_device_and_alignment(q_weight, partial.device_id, 16, "q_weight");
    check_device_and_alignment(k_weight, partial.device_id, 16, "k_weight");
    check(partial.dtype() == AITER_DTYPE_fp32, "partial must be fp32");
    check(partial.dim() == 3 && partial.size(0) == kPlanes && partial.size(2) == kN,
          "partial must be [6, M, 2624]");
    check(partial.is_contiguous(), "partial must be contiguous");
    const int64_t m = partial.size(1);
    check_bf16_matrix(out, m, kN, "out");
    check_bf16_matrix(q_out, m, kQN, "q_out");
    check_bf16_matrix(k_out, m, kKN, "k_out");
    check_bf16_vector(q_weight, kQN, "q_weight");
    check_bf16_vector(k_weight, kKN, "k_weight");
    if(m == 0)
        return;
    for(const auto* output : {&out, &q_out, &k_out})
    {
        check_no_overlap(*output, partial);
        check_no_overlap(*output, q_weight);
        check_no_overlap(*output, k_weight);
    }
    check_no_overlap(out, q_out);
    check_no_overlap(out, k_out);
    check_no_overlap(q_out, k_out);

    HipDeviceGuard device_guard(partial.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    splitk_reduce_qk_rmsnorm_kernel<<<dim3(m, 2), dim3(kBlockSize), 0, stream>>>(
        static_cast<const float*>(partial.data_ptr()),
        static_cast<element*>(out.data_ptr()),
        static_cast<element*>(q_out.data_ptr()),
        static_cast<element*>(k_out.data_ptr()),
        static_cast<const element*>(q_weight.data_ptr()),
        static_cast<const element*>(k_weight.data_ptr()),
        static_cast<float>(q_eps),
        static_cast<float>(k_eps));
}

} // namespace aiter
