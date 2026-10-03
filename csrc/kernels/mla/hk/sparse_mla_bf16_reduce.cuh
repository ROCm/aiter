// SPDX-License-Identifier: MIT
// Fully unrolled, one-wave split-K reducer shared by the v1/v2 implementations.
#pragma once
#include <hip/hip_runtime.h>

namespace sparse_mla_bf16 {

template <int Heads, int Splits>
__global__ __launch_bounds__(64) void reduce(Params p)
{
    static_assert(Heads == 16 || Heads == 64);
    static_assert(Splits == 2 || Splits == 4 || Splits == 8 || Splits == 16 || Splits == 32);
    const uint32_t query = blockIdx.x, head = blockIdx.y;
    const uint32_t d     = threadIdx.x * 8u;
    const uint64_t first = uint64_t(query) * Splits * Heads + head;
    float maximum        = -INFINITY;
#pragma unroll
    for(int s = 0; s < Splits; ++s)
        maximum = fmaxf(maximum, p.partial_lse[first + uint64_t(s) * Heads]);
    float sum            = 0.0f;
    float accumulator[8] = {};
    // Every partial row is 2048 B; caller-provided workspace may be unaligned.
    // Retain a scalar-load fallback for callers of the raw C ABI.
    const bool aligned_partial = (reinterpret_cast<uintptr_t>(p.partial_o) & 15u) == 0;
#pragma unroll
    for(int s = 0; s < Splits; ++s)
    {
        const uint64_t row = first + uint64_t(s) * Heads;
        const float value  = p.partial_lse[row];
        // Preserve the baseline summation order and empty-partition semantics.
        const float weight = value == -INFINITY ? 0.0f : expf(value - maximum);
        sum += weight;
        const float* source = p.partial_o + row * 512u + d;
        float values[8];
        if(aligned_partial)
        {
            const float4 lo = *reinterpret_cast<const float4*>(source);
            const float4 hi = *reinterpret_cast<const float4*>(source + 4);
            values[0]       = lo.x;
            values[1]       = lo.y;
            values[2]       = lo.z;
            values[3]       = lo.w;
            values[4]       = hi.x;
            values[5]       = hi.y;
            values[6]       = hi.z;
            values[7]       = hi.w;
        }
        else
        {
#pragma unroll
            for(int i = 0; i < 8; ++i)
                values[i] = source[i];
        }
#pragma unroll
        for(int i = 0; i < 8; ++i)
            accumulator[i] += weight * values[i];
    }
    const float reciprocal = sum > 0.0f ? 1.0f / sum : 0.0f;
    uint32_t pairs[4];
#pragma unroll
    for(int i = 0; i < 4; ++i)
    {
        const float lo = accumulator[2 * i] * reciprocal;
        const float hi = accumulator[2 * i + 1] * reciprocal;
        asm volatile("v_cvt_pk_bf16_f32 %0, %1, %2" : "=v"(pairs[i]) : "v"(lo), "v"(hi));
    }
    uint16_t* destination = p.out + (uint64_t(query) * Heads + head) * 512u + d;
    if((reinterpret_cast<uintptr_t>(destination) & 15u) == 0)
    {
        const uint4 packed{pairs[0], pairs[1], pairs[2], pairs[3]};
        *reinterpret_cast<uint4*>(destination) = packed;
    }
    else
    {
        // The API permits an output storage offset of one BF16 (2 B alignment).
#pragma unroll
        for(int i = 0; i < 4; ++i)
        {
            destination[2 * i]     = uint16_t(pairs[i]);
            destination[2 * i + 1] = uint16_t(pairs[i] >> 16);
        }
    }
    if(threadIdx.x == 0 && p.lse)
        p.lse[uint64_t(query) * Heads + head] = sum > 0.0f ? maximum + logf(sum) : -INFINITY;
}

template <int Heads>
inline hipError_t launch_reduce64(Params p, hipStream_t stream)
{
#define SPARSE_MLA_REDUCE_CASE(S)                                                               \
    case S:                                                                                     \
        hipLaunchKernelGGL((reduce<Heads, S>), dim3(p.queries, Heads), dim3(64), 0, stream, p); \
        break
    switch(p.splits)
    {
        SPARSE_MLA_REDUCE_CASE(2);
        SPARSE_MLA_REDUCE_CASE(4);
        SPARSE_MLA_REDUCE_CASE(8);
        SPARSE_MLA_REDUCE_CASE(16);
        SPARSE_MLA_REDUCE_CASE(32);
    default: return hipErrorInvalidValue;
    }
#undef SPARSE_MLA_REDUCE_CASE
    return hipGetLastError();
}

} // namespace sparse_mla_bf16
