// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "fused_indexed_gated_residual.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <initializer_list>

namespace {

constexpr int THREADS = 256;

struct F32
{
    using raw = float;
    __device__ static float load(raw v) { return v; }
    __device__ static raw store(float f) { return f; }
};

struct F16
{
    using raw = __half;
    __device__ static float load(raw v) { return __half2float(v); }
    __device__ static raw store(float f) { return __float2half_rn(f); }
};

struct BF16
{
    using raw = uint16_t;
    __device__ static float load(raw v) { return __uint_as_float(static_cast<uint32_t>(v) << 16); }
    // Round to nearest even, NaN -> canonical quiet NaN, as torch does.
    __device__ static raw store(float f)
    {
        const uint32_t u = __float_as_uint(f);
        if((u & 0x7fffffffu) > 0x7f800000u)
            return 0x7fc0u;
        return static_cast<raw>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
    }
};

template <typename T, int VEC>
struct alignas(sizeof(T) * VEC) Vec
{
    T v[VEC];
};

// One thread per VEC consecutive columns of one row. No __restrict__: out may alias residual or x.
template <typename E, typename Index, int VEC, bool ROUND>
__global__ void __launch_bounds__(THREADS) gated_residual_kernel(const typename E::raw* residual,
                                                                 const typename E::raw* x,
                                                                 const typename E::raw* gate,
                                                                 const Index* indices,
                                                                 typename E::raw* out,
                                                                 int64_t rows,
                                                                 int64_t cols,
                                                                 int64_t residual_stride,
                                                                 int64_t x_stride,
                                                                 int64_t out_stride,
                                                                 int64_t gate_stride)
{
    using V               = Vec<typename E::raw, VEC>;
    const int64_t per_row = cols / VEC;
    const int64_t total   = rows * per_row;
    const int64_t step    = static_cast<int64_t>(gridDim.x) * THREADS;
    for(int64_t i = static_cast<int64_t>(blockIdx.x) * THREADS + threadIdx.x; i < total; i += step)
    {
        const int64_t r = i / per_row;
        const int64_t c = (i - r * per_row) * VEC;
        const V res     = *reinterpret_cast<const V*>(residual + r * residual_stride + c);
        const V branch  = *reinterpret_cast<const V*>(x + r * x_stride + c);
        const V g =
            *reinterpret_cast<const V*>(gate + static_cast<int64_t>(indices[r]) * gate_stride + c);
        V y;
#pragma unroll
        for(int k = 0; k < VEC; ++k)
        {
            float gated = E::load(g.v[k]) * E::load(branch.v[k]);
            if constexpr(ROUND)
                gated = E::load(E::store(gated));
            y.v[k] = E::store(E::load(res.v[k]) + gated);
        }
        *reinterpret_cast<V*>(out + r * out_stride + c) = y;
    }
}

template <typename E, typename Index, int VEC, bool ROUND>
void launch(const aiter_tensor_t& residual,
            const aiter_tensor_t& x,
            const aiter_tensor_t& gate,
            const aiter_tensor_t& indices,
            aiter_tensor_t& out,
            hipStream_t stream)
{
    using raw           = typename E::raw;
    const int64_t rows  = residual.size(0);
    const int64_t cols  = residual.size(1);
    const int64_t total = rows * (cols / VEC);
    // Grid-stride loop; enough blocks to fill any current GPU several times over.
    const int blocks = static_cast<int>(std::min<int64_t>((total + THREADS - 1) / THREADS, 65536));
    hipLaunchKernelGGL((gated_residual_kernel<E, Index, VEC, ROUND>),
                       dim3(blocks),
                       dim3(THREADS),
                       0,
                       stream,
                       static_cast<const raw*>(residual.data_ptr()),
                       static_cast<const raw*>(x.data_ptr()),
                       static_cast<const raw*>(gate.data_ptr()),
                       static_cast<const Index*>(indices.data_ptr()),
                       static_cast<raw*>(out.data_ptr()),
                       rows,
                       cols,
                       residual.stride(0),
                       x.stride(0),
                       out.stride(0),
                       gate.stride(0));
}

template <typename E, typename Index>
void dispatch(const aiter_tensor_t& residual,
              const aiter_tensor_t& x,
              const aiter_tensor_t& gate,
              const aiter_tensor_t& indices,
              aiter_tensor_t& out,
              bool round_intermediate,
              hipStream_t stream)
{
    constexpr int VEC = 16 / sizeof(typename E::raw);
    bool vectorized   = residual.size(1) % VEC == 0;
    for(const aiter_tensor_t* t :
        std::initializer_list<const aiter_tensor_t*>{&residual, &x, &gate, &out})
        vectorized = vectorized && (t->size(0) < 2 || t->stride(0) % VEC == 0) &&
                     reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0;
    if(vectorized)
        round_intermediate ? launch<E, Index, VEC, true>(residual, x, gate, indices, out, stream)
                           : launch<E, Index, VEC, false>(residual, x, gate, indices, out, stream);
    else
        round_intermediate ? launch<E, Index, 1, true>(residual, x, gate, indices, out, stream)
                           : launch<E, Index, 1, false>(residual, x, gate, indices, out, stream);
}

template <typename E>
void dispatch_index(const aiter_tensor_t& residual,
                    const aiter_tensor_t& x,
                    const aiter_tensor_t& gate,
                    const aiter_tensor_t& indices,
                    aiter_tensor_t& out,
                    bool round_intermediate,
                    hipStream_t stream)
{
    if(indices.dtype() == AITER_DTYPE_i64)
        dispatch<E, int64_t>(residual, x, gate, indices, out, round_intermediate, stream);
    else
        dispatch<E, int32_t>(residual, x, gate, indices, out, round_intermediate, stream);
}

} // namespace

void fused_indexed_gated_residual(const aiter_tensor_t& residual,
                                  const aiter_tensor_t& x,
                                  const aiter_tensor_t& gate,
                                  const aiter_tensor_t& indices,
                                  aiter_tensor_t& out,
                                  bool round_intermediate)
{
    AITER_CHECK(residual.dim() == 2, "residual must be 2-D");
    const int64_t rows = residual.size(0), cols = residual.size(1);
    for(const aiter_tensor_t* t : std::initializer_list<const aiter_tensor_t*>{&x, &out})
        AITER_CHECK(t->dim() == 2 && t->size(0) == rows && t->size(1) == cols,
                    "x and out must match residual's shape");
    AITER_CHECK(gate.dim() == 2 && gate.size(1) == cols, "gate table must be [G, N]");
    AITER_CHECK(indices.dim() == 1 && indices.size(0) == rows, "indices must be [M]");
    AITER_CHECK(indices.dtype() == AITER_DTYPE_i32 || indices.dtype() == AITER_DTYPE_i64,
                "indices must be int32 or int64, got ",
                AiterDtype_to_str(indices.dtype()));
    AITER_CHECK(rows < 2 || indices.stride(0) == 1, "indices must be contiguous");
    const AiterDtype dtype = residual.dtype();
    AITER_CHECK(dtype == AITER_DTYPE_fp32 || dtype == AITER_DTYPE_fp16 || dtype == AITER_DTYPE_bf16,
                "residual must be fp32, fp16 or bf16, got ",
                AiterDtype_to_str(dtype));
    for(const aiter_tensor_t* t : std::initializer_list<const aiter_tensor_t*>{&x, &gate, &out})
    {
        AITER_CHECK(t->dtype() == dtype, "residual, x, gate and out must share one dtype");
        AITER_CHECK(cols < 2 || t->stride(1) == 1, "all operands need unit stride along N");
    }
    AITER_CHECK(residual.stride(1) == 1 || cols < 2, "all operands need unit stride along N");
    if(rows == 0 || cols == 0)
        return;

    HipDeviceGuard device_guard(residual.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    if(dtype == AITER_DTYPE_bf16)
        dispatch_index<BF16>(residual, x, gate, indices, out, round_intermediate, stream);
    else if(dtype == AITER_DTYPE_fp16)
        dispatch_index<F16>(residual, x, gate, indices, out, round_intermediate, stream);
    else
        dispatch_index<F32>(residual, x, gate, indices, out, round_intermediate, stream);
}
