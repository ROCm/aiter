// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "aiter_hip_common.h"
#include "dispatch_utils.h"
#include "hip_float8.h"
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>

#ifdef __HIP_DEVICE_COMPILE__
#include "opus/opus.hpp"
#endif

// =====================================================================================================================
// Keyword Glossary
// =====================================================================================================================
//
//   1c / 2c              Number of input-output tensor pairs.
//                        1c = one input & one output channel;  2c = two inputs & two outputs channels.
//
//   Cached / Uncached    How cos/sin values are obtained.
//                        Cached  -- read pre-computed cos/sin from memory.
//                        Uncached -- compute cos/sin on the fly from theta (frequency) values.
//
//   ReuseFreqsFrontPart  When true, the kernel reuses the front portion of the freqs/cos/sin
//                        tensor for the repeated part, so the caller does not need to expand
//                        (repeat) the tensor beforehand.
//
//   sbhd / thd / 2d      Tensor layout conventions:
//                          sbhd -- [seq_len, batch, num_heads, head_dim]
//                          thd  -- [total_tokens, num_heads, head_dim]  (variable-length / packed)
//                          2d   -- 2-D positional encoding (image height x width)
//
//   NopeFirst            Controls which slice of the head dimension is rotated.
//                        false -- rotate [0, size_r), copy the remainder.
//                        true  -- copy [0, size_d - size_r), rotate the tail.
//

#define ROTATE_STYLE_NEOX 0
#define ROTATE_STYLE_GPTJ 1

namespace aiter {
// =====================================================================================================================
// Kernel Helper Functions
//

template <int32_t RotateStyle, bool IsForward, bool ReuseFreqsFrontPart, typename scalar_f_t>
__device__ __forceinline__ void load_cos_sin_uncached(float* p_cos_0,
                                                      float* p_sin_0,
                                                      float* p_cos_1,
                                                      float* p_sin_1,
                                                      const scalar_f_t* __restrict__ p_freqs,
                                                      const int32_t did,
                                                      const int32_t size_half_r)
{
    if constexpr(RotateStyle == ROTATE_STYLE_NEOX)
    {
        if constexpr(IsForward)
        {
            sincosf(float(p_freqs[did]), p_sin_0, p_cos_0);
            if constexpr(ReuseFreqsFrontPart)
            {
                *p_cos_1 = *p_cos_0;
                *p_sin_1 = *p_sin_0;
            }
            else
            {
                sincosf(float(p_freqs[did + size_half_r]), p_sin_1, p_cos_1);
            }
        }
        else
        {
            const float f_did_0 = float(p_freqs[did]);
            if constexpr(ReuseFreqsFrontPart)
            {
                *p_cos_0 = cosf(f_did_0);
                *p_sin_0 = sinf(f_did_0);
                *p_cos_1 = *p_cos_0;
                *p_sin_1 = *p_sin_0;
            }
            else
            {
                const float f_did_1 = p_freqs[did + size_half_r];
                *p_cos_0            = cosf(f_did_0);
                *p_sin_0            = sinf(f_did_1);
                *p_cos_1            = cosf(f_did_1);
                *p_sin_1            = sinf(f_did_0);
            }
        }
    }
    else if constexpr(RotateStyle == ROTATE_STYLE_GPTJ)
    {
        if constexpr(IsForward)
        {
            if constexpr(ReuseFreqsFrontPart)
            {
                sincosf(float(p_freqs[did]), p_sin_0, p_cos_0);
                *p_cos_1 = *p_cos_0;
                *p_sin_1 = *p_sin_0;
            }
            else
            {
                sincosf(float(p_freqs[did * 2]), p_sin_0, p_cos_0);
                sincosf(float(p_freqs[did * 2 + 1]), p_sin_1, p_cos_1);
            }
        }
        else
        {
            if constexpr(ReuseFreqsFrontPart)
            {
                sincosf(float(p_freqs[did]), p_sin_0, p_cos_0);
                *p_cos_1 = *p_cos_0;
                *p_sin_1 = *p_sin_0;
            }
            else
            {
                const float f_did_0 = float(p_freqs[did * 2]);
                const float f_did_1 = float(p_freqs[did * 2 + 1]);
                *p_cos_0            = cosf(f_did_0);
                *p_sin_0            = sinf(f_did_1);
                *p_cos_1            = cosf(f_did_1);
                *p_sin_1            = sinf(f_did_0);
            }
        }
    }
}

template <int32_t RotateStyle, bool IsForward, bool ReuseFreqsFrontPart, typename scalar_f_t>
__device__ __forceinline__ void load_cos_sin_cached(float* p_cos_0,
                                                    float* p_sin_0,
                                                    float* p_cos_1,
                                                    float* p_sin_1,
                                                    const scalar_f_t* __restrict__ p_cos,
                                                    const scalar_f_t* __restrict__ p_sin,
                                                    const int32_t did,
                                                    const int32_t size_half_r)
{
    if constexpr(RotateStyle == ROTATE_STYLE_NEOX)
    {
        if constexpr(IsForward)
        {
            *p_cos_0 = float(p_cos[did]);
            *p_sin_0 = float(p_sin[did]);
            if constexpr(ReuseFreqsFrontPart)
            {
                *p_cos_1 = *p_cos_0;
                *p_sin_1 = *p_sin_0;
            }
            else
            {
                *p_cos_1 = float(p_cos[did + size_half_r]);
                *p_sin_1 = float(p_sin[did + size_half_r]);
            }
        }
        else
        {
            if constexpr(ReuseFreqsFrontPart)
            {
                *p_cos_0 = float(p_cos[did]);
                *p_sin_0 = float(p_sin[did]);
                *p_cos_1 = *p_cos_0;
                *p_sin_1 = *p_sin_0;
            }
            else
            {
                *p_cos_0 = float(p_cos[did]);
                *p_sin_0 = float(p_sin[did + size_half_r]);
                *p_cos_1 = float(p_cos[did + size_half_r]);
                *p_sin_1 = float(p_sin[did]);
            }
        }
    }
    else if constexpr(RotateStyle == ROTATE_STYLE_GPTJ)
    {
        if constexpr(IsForward)
        {
            if constexpr(ReuseFreqsFrontPart)
            {
                *p_cos_0 = float(p_cos[did]);
                *p_sin_0 = float(p_sin[did]);
                *p_cos_1 = *p_cos_0;
                *p_sin_1 = *p_sin_0;
            }
            else
            {
                *p_cos_0 = float(p_cos[did * 2]);
                *p_sin_0 = float(p_sin[did * 2]);
                *p_cos_1 = float(p_cos[did * 2 + 1]);
                *p_sin_1 = float(p_sin[did * 2 + 1]);
            }
        }
        else
        {
            if constexpr(ReuseFreqsFrontPart)
            {
                *p_cos_0 = float(p_cos[did]);
                *p_sin_0 = float(p_sin[did]);
                *p_cos_1 = *p_cos_0;
                *p_sin_1 = *p_sin_0;
            }
            else
            {
                *p_cos_0 = float(p_cos[did * 2]);
                *p_sin_0 = float(p_sin[did * 2 + 1]);
                *p_cos_1 = float(p_cos[did * 2 + 1]);
                *p_sin_1 = float(p_sin[did * 2]);
            }
        }
    }
}

template <int32_t RotateStyle, bool StrideDEq1>
__device__ __forceinline__ void get_offset(int32_t* p_offset_0,
                                           int32_t* p_offset_1,
                                           const int32_t did,
                                           const int32_t hid,
                                           const int32_t stride_d,
                                           const int32_t stride_h,
                                           const int32_t size_half_r)
{
    const int32_t offset_h = hid * stride_h;

    if constexpr(RotateStyle == ROTATE_STYLE_NEOX)
    {
        *p_offset_0 = offset_h + did * stride_d;
        *p_offset_1 = *p_offset_0 + size_half_r * stride_d;
    }
    else if constexpr(RotateStyle == ROTATE_STYLE_GPTJ)
    {
        *p_offset_0 = offset_h + 2 * did * stride_d;
        if constexpr(StrideDEq1)
        {
            // Asking compiler to merge memory ops when accessing adjacent elements.
            *p_offset_1 = *p_offset_0 + 1;
        }
        else
        {
            *p_offset_1 = *p_offset_0 + stride_d;
        }
    }
}

template <int32_t RotateStyle, bool StrideDEq1, typename o_scalar_t, typename i_scalar_t>
__device__ __forceinline__ void load_payload(o_scalar_t* p_data_0,
                                             o_scalar_t* p_data_1,
                                             const i_scalar_t* p_buffer,
                                             const int32_t did,
                                             const int32_t hid,
                                             const int32_t stride_d,
                                             const int32_t stride_h,
                                             const int32_t size_half_r)
{
    int32_t offset_0, offset_1;
    get_offset<RotateStyle, StrideDEq1>(
        &offset_0, &offset_1, did, hid, stride_d, stride_h, size_half_r);

    *p_data_0 = o_scalar_t(p_buffer[offset_0]);
    *p_data_1 = o_scalar_t(p_buffer[offset_1]);
}

template <int32_t RotateStyle, bool StrideDEq1, typename o_scalar_t, typename i_scalar_t>
__device__ __forceinline__ void store_payload(o_scalar_t* p_buffer,
                                              const i_scalar_t data_0,
                                              const i_scalar_t data_1,
                                              const int32_t did,
                                              const int32_t hid,
                                              const int32_t stride_d,
                                              const int32_t stride_h,
                                              const int32_t size_half_r)
{
    int32_t offset_0, offset_1;
    get_offset<RotateStyle, StrideDEq1>(
        &offset_0, &offset_1, did, hid, stride_d, stride_h, size_half_r);

    p_buffer[offset_0] = o_scalar_t(data_0);
    p_buffer[offset_1] = o_scalar_t(data_1);
}

#ifdef __HIP_DEVICE_COMPILE__
// Map torch/c10 types to opus-compatible types for ext_vector_type
template <typename T>
struct opus_type_map
{
    using type = T;
};
template <>
struct opus_type_map<c10::Half>
{
    using type = opus::fp16_t;
};
template <>
struct opus_type_map<c10::BFloat16>
{
    using type = opus::bf16_t;
};
template <typename T>
using opus_type_t = typename opus_type_map<T>::type;

// Helper to create opus gmem accessor with automatic pointer cast from c10 types to opus types
template <typename T>
__device__ __forceinline__ auto opus_gmem(const T* ptr)
{
    return opus::make_gmem<opus_type_t<T>>(reinterpret_cast<const opus_type_t<T>*>(ptr));
}
template <typename T>
__device__ __forceinline__ auto opus_gmem(T* ptr)
{
    return opus::make_gmem<opus_type_t<T>>(reinterpret_cast<opus_type_t<T>*>(ptr));
}
#endif // __HIP_DEVICE_COMPILE__

template <int32_t VecPairs = 1, typename scalar_t, typename gmem_i_t, typename gmem_o_t>
__device__ __forceinline__ void
elementwise_copy(gmem_i_t& g_input,
                 gmem_o_t& g_output,
                 const int32_t offset_i, // per-thread (s,b) offset in elements
                 const int32_t offset_o,
                 scalar_t* __restrict__ p_output, // fallback for VP==1
                 const scalar_t* __restrict__ p_input,
                 const int32_t hid_end,
                 const int32_t did_start,
                 const int32_t did_end,
                 const int32_t stride_i_h,
                 const int32_t stride_i_d,
                 const int32_t stride_o_h,
                 const int32_t stride_o_d,
                 const int32_t my_did_offset,
                 const int32_t did_stride)
{
    if(did_end > did_start)
    {
        constexpr int32_t elem_bytes = sizeof(scalar_t);
        for(int32_t hid = 0; hid < hid_end; hid++)
        {
#ifdef __HIP_DEVICE_COMPILE__
            if constexpr(VecPairs > 1)
            {
                const int32_t byte_off_i = (offset_i + hid * stride_i_h) * elem_bytes;
                const int32_t byte_off_o = (offset_o + hid * stride_o_h) * elem_bytes;
                for(int32_t did = did_start + my_did_offset * VecPairs; did + VecPairs <= did_end;
                    did += did_stride * VecPairs)
                {
                    auto v = g_input.template _load<VecPairs>(byte_off_i + did * elem_bytes);
                    g_output.template _store<VecPairs>(v, byte_off_o + did * elem_bytes);
                }
            }
            else
            {
                const int32_t byte_off_i = (offset_i + hid * stride_i_h) * elem_bytes;
                const int32_t byte_off_o = (offset_o + hid * stride_o_h) * elem_bytes;
                for(int32_t did = did_start + my_did_offset; did < did_end; did += did_stride)
                {
                    auto v = g_input.template _load<1>(byte_off_i + did * stride_i_d * elem_bytes);
                    g_output.template _store<1>(v, byte_off_o + did * stride_o_d * elem_bytes);
                }
            }
#else
            {
                const int32_t off_i = offset_i + hid * stride_i_h;
                const int32_t off_o = offset_o + hid * stride_o_h;
                for(int32_t did = did_start + my_did_offset; did < did_end; did += did_stride)
                {
                    p_output[off_o + did * stride_o_d] = p_input[off_i + did * stride_i_d];
                }
            }
#endif
        }
    }
}

template <int32_t VecPairs = 1,
          typename scalar_t,
          typename gmem_ix_t,
          typename gmem_iy_t,
          typename gmem_ox_t,
          typename gmem_oy_t>
__device__ __forceinline__ void
elementwise_copy_2c(gmem_ix_t& g_ix,
                    gmem_iy_t& g_iy,
                    gmem_ox_t& g_ox,
                    gmem_oy_t& g_oy,
                    const int32_t off_ix, // per-thread (s,b) offset in elements
                    const int32_t off_iy,
                    const int32_t off_ox,
                    const int32_t off_oy,
                    scalar_t* __restrict__ p_output_x, // fallback for VP==1
                    scalar_t* __restrict__ p_output_y,
                    const scalar_t* __restrict__ p_input_x,
                    const scalar_t* __restrict__ p_input_y,
                    const int32_t hid_end_x,
                    const int32_t hid_end_y,
                    const int32_t did_start,
                    const int32_t did_end,
                    const int32_t stride_ix_h,
                    const int32_t stride_ix_d,
                    const int32_t stride_iy_h,
                    const int32_t stride_iy_d,
                    const int32_t stride_ox_h,
                    const int32_t stride_ox_d,
                    const int32_t stride_oy_h,
                    const int32_t stride_oy_d,
                    const int32_t my_did_offset,
                    const int32_t did_stride)
{
    if(did_end > did_start)
    {
        constexpr int32_t elem_bytes = sizeof(scalar_t);
        const int32_t hid_min_end    = hid_end_x < hid_end_y ? hid_end_x : hid_end_y;

        for(int32_t hid = 0; hid < hid_min_end; hid++)
        {
#ifdef __HIP_DEVICE_COMPILE__
            if constexpr(VecPairs > 1)
            {
                const int32_t bo_ix = (off_ix + hid * stride_ix_h) * elem_bytes;
                const int32_t bo_iy = (off_iy + hid * stride_iy_h) * elem_bytes;
                const int32_t bo_ox = (off_ox + hid * stride_ox_h) * elem_bytes;
                const int32_t bo_oy = (off_oy + hid * stride_oy_h) * elem_bytes;
                for(int32_t did = did_start + my_did_offset * VecPairs; did + VecPairs <= did_end;
                    did += did_stride * VecPairs)
                {
                    const int32_t db = did * elem_bytes;
                    auto vx          = g_ix.template _load<VecPairs>(bo_ix + db);
                    auto vy          = g_iy.template _load<VecPairs>(bo_iy + db);
                    g_ox.template _store<VecPairs>(vx, bo_ox + db);
                    g_oy.template _store<VecPairs>(vy, bo_oy + db);
                }
            }
            else
            {
                const int32_t bo_ix = (off_ix + hid * stride_ix_h) * elem_bytes;
                const int32_t bo_iy = (off_iy + hid * stride_iy_h) * elem_bytes;
                const int32_t bo_ox = (off_ox + hid * stride_ox_h) * elem_bytes;
                const int32_t bo_oy = (off_oy + hid * stride_oy_h) * elem_bytes;
                for(int32_t did = did_start + my_did_offset; did < did_end; did += did_stride)
                {
                    auto vx = g_ix.template _load<1>(bo_ix + did * stride_ix_d * elem_bytes);
                    auto vy = g_iy.template _load<1>(bo_iy + did * stride_iy_d * elem_bytes);
                    g_ox.template _store<1>(vx, bo_ox + did * stride_ox_d * elem_bytes);
                    g_oy.template _store<1>(vy, bo_oy + did * stride_oy_d * elem_bytes);
                }
            }
#else
            {
                const int32_t offset_ix = off_ix + hid * stride_ix_h;
                const int32_t offset_iy = off_iy + hid * stride_iy_h;
                const int32_t offset_ox = off_ox + hid * stride_ox_h;
                const int32_t offset_oy = off_oy + hid * stride_oy_h;
                for(int32_t did = did_start + my_did_offset; did < did_end; did += did_stride)
                {
                    p_output_x[offset_ox + did * stride_ox_d] =
                        p_input_x[offset_ix + did * stride_ix_d];
                    p_output_y[offset_oy + did * stride_oy_d] =
                        p_input_y[offset_iy + did * stride_iy_d];
                }
            }
#endif
        }

        for(int32_t hid = hid_min_end; hid < hid_end_x; hid++)
        {
#ifdef __HIP_DEVICE_COMPILE__
            if constexpr(VecPairs > 1)
            {
                const int32_t bo_ix = (off_ix + hid * stride_ix_h) * elem_bytes;
                const int32_t bo_ox = (off_ox + hid * stride_ox_h) * elem_bytes;
                for(int32_t did = did_start + my_did_offset * VecPairs; did + VecPairs <= did_end;
                    did += did_stride * VecPairs)
                {
                    const int32_t db = did * elem_bytes;
                    auto v           = g_ix.template _load<VecPairs>(bo_ix + db);
                    g_ox.template _store<VecPairs>(v, bo_ox + db);
                }
            }
            else
            {
                const int32_t bo_ix = (off_ix + hid * stride_ix_h) * elem_bytes;
                const int32_t bo_ox = (off_ox + hid * stride_ox_h) * elem_bytes;
                for(int32_t did = did_start + my_did_offset; did < did_end; did += did_stride)
                {
                    auto v = g_ix.template _load<1>(bo_ix + did * stride_ix_d * elem_bytes);
                    g_ox.template _store<1>(v, bo_ox + did * stride_ox_d * elem_bytes);
                }
            }
#else
            {
                const int32_t offset_ix = off_ix + hid * stride_ix_h;
                const int32_t offset_ox = off_ox + hid * stride_ox_h;
                for(int32_t did = did_start + my_did_offset; did < did_end; did += did_stride)
                {
                    p_output_x[offset_ox + did * stride_ox_d] =
                        p_input_x[offset_ix + did * stride_ix_d];
                }
            }
#endif
        }

        for(int32_t hid = hid_min_end; hid < hid_end_y; hid++)
        {
#ifdef __HIP_DEVICE_COMPILE__
            if constexpr(VecPairs > 1)
            {
                const int32_t bo_iy = (off_iy + hid * stride_iy_h) * elem_bytes;
                const int32_t bo_oy = (off_oy + hid * stride_oy_h) * elem_bytes;
                for(int32_t did = did_start + my_did_offset * VecPairs; did + VecPairs <= did_end;
                    did += did_stride * VecPairs)
                {
                    const int32_t db = did * elem_bytes;
                    auto v           = g_iy.template _load<VecPairs>(bo_iy + db);
                    g_oy.template _store<VecPairs>(v, bo_oy + db);
                }
            }
            else
            {
                const int32_t bo_iy = (off_iy + hid * stride_iy_h) * elem_bytes;
                const int32_t bo_oy = (off_oy + hid * stride_oy_h) * elem_bytes;
                for(int32_t did = did_start + my_did_offset; did < did_end; did += did_stride)
                {
                    auto v = g_iy.template _load<1>(bo_iy + did * stride_iy_d * elem_bytes);
                    g_oy.template _store<1>(v, bo_oy + did * stride_oy_d * elem_bytes);
                }
            }
#else
            {
                const int32_t offset_iy = off_iy + hid * stride_iy_h;
                const int32_t offset_oy = off_oy + hid * stride_oy_h;
                for(int32_t did = did_start + my_did_offset; did < did_end; did += did_stride)
                {
                    p_output_y[offset_oy + did * stride_oy_d] =
                        p_input_y[offset_iy + did * stride_iy_d];
                }
            }
#endif
        }
    }
}

// =====================================================================================================================
// Vectorized Helper Functions (using opus for buffer load/store)
//

// load_cos_sin_uncached_vec: loads freqs from pre-built gmem descriptor, computes sincos.
// g_freqs must be created from UNIFORM (kernel-arg) base pointer to avoid waterfall loops.
// byte_offset_f = offset_f * sizeof(scalar_f_t) -- the per-thread byte offset for this position.
template <int32_t RotateStyle,
          int32_t VecPairs,
          bool IsForward,
          bool ReuseFreqsFrontPart,
          typename gmem_t>
__device__ __forceinline__ void load_cos_sin_uncached_vec(float (&cos_0)[VecPairs],
                                                          float (&sin_0)[VecPairs],
                                                          float (&cos_1)[VecPairs],
                                                          float (&sin_1)[VecPairs],
                                                          gmem_t& g_freqs,
                                                          const int32_t byte_offset_f,
                                                          const int32_t did,
                                                          const int32_t size_half_r)
{
#ifdef __HIP_DEVICE_COMPILE__
    constexpr int32_t elem_bytes = sizeof(typename gmem_t::scalar_type);
    if constexpr(RotateStyle == ROTATE_STYLE_NEOX)
    {
        auto v_f0 = g_freqs.template _load<VecPairs>(byte_offset_f + did * elem_bytes);
        opus::static_for<VecPairs>(
            [&](auto i) { sincosf(float(v_f0[i.value]), &sin_0[i.value], &cos_0[i.value]); });

        if constexpr(ReuseFreqsFrontPart)
        {
            opus::static_for<VecPairs>([&](auto i) {
                cos_1[i.value] = cos_0[i.value];
                sin_1[i.value] = sin_0[i.value];
            });
        }
        else
        {
            auto v_f1 =
                g_freqs.template _load<VecPairs>(byte_offset_f + (did + size_half_r) * elem_bytes);
            opus::static_for<VecPairs>([&](auto i) {
                if constexpr(IsForward)
                {
                    sincosf(float(v_f1[i.value]), &sin_1[i.value], &cos_1[i.value]);
                }
                else
                {
                    sin_1[i.value] = sin_0[i.value]; // save sin(f0) from first loop
                    sincosf(float(v_f1[i.value]), &sin_0[i.value], &cos_1[i.value]);
                }
            });
        }
    }
    else if constexpr(RotateStyle == ROTATE_STYLE_GPTJ)
    {
        if constexpr(ReuseFreqsFrontPart)
        {
            auto v_f = g_freqs.template _load<VecPairs>(byte_offset_f + did * elem_bytes);
            opus::static_for<VecPairs>([&](auto i) {
                sincosf(float(v_f[i.value]), &sin_0[i.value], &cos_0[i.value]);
                cos_1[i.value] = cos_0[i.value];
                sin_1[i.value] = sin_0[i.value];
            });
        }
        else
        {
            // GPTJ non-reuse: freqs at did*2 and did*2+1, load 2*VecPairs contiguous
            if constexpr(VecPairs == 1)
            {
                auto v0  = g_freqs.template _load<1>(byte_offset_f + (did * 2) * elem_bytes);
                auto v1  = g_freqs.template _load<1>(byte_offset_f + (did * 2 + 1) * elem_bytes);
                float f0 = float(v0[0]), f1 = float(v1[0]);
                if constexpr(IsForward)
                {
                    sincosf(f0, &sin_0[0], &cos_0[0]);
                    sincosf(f1, &sin_1[0], &cos_1[0]);
                }
                else
                {
                    sincosf(f0, &sin_1[0], &cos_0[0]);
                    sincosf(f1, &sin_0[0], &cos_1[0]);
                }
            }
            else
            {
                auto v_lo =
                    g_freqs.template _load<VecPairs>(byte_offset_f + (did * 2) * elem_bytes);
                auto v_hi = g_freqs.template _load<VecPairs>(byte_offset_f +
                                                             (did * 2 + VecPairs) * elem_bytes);
                opus::static_for<VecPairs>([&](auto i) {
                    constexpr int idx0 = i.value * 2;
                    constexpr int idx1 = i.value * 2 + 1;
                    float f0, f1;
                    if constexpr(idx0 < VecPairs)
                    {
                        f0 = float(v_lo[idx0]);
                        f1 = float(v_lo[idx1]);
                    }
                    else
                    {
                        f0 = float(v_hi[idx0 - VecPairs]);
                        f1 = float(v_hi[idx1 - VecPairs]);
                    }
                    if constexpr(IsForward)
                    {
                        sincosf(f0, &sin_0[i.value], &cos_0[i.value]);
                        sincosf(f1, &sin_1[i.value], &cos_1[i.value]);
                    }
                    else
                    {
                        sincosf(f0, &sin_1[i.value], &cos_0[i.value]);
                        sincosf(f1, &sin_0[i.value], &cos_1[i.value]);
                    }
                });
            }
        }
    }
#endif
}

// load_cos_sin_cached_vec: loads cos/sin from pre-built gmem descriptors.
// g_cos/g_sin must be created from UNIFORM (kernel-arg) base pointers to avoid waterfall loops.
// byte_offset_f = offset_f * sizeof(scalar_f_t) -- the per-thread byte offset for this position.
template <int32_t RotateStyle,
          int32_t VecPairs,
          bool IsForward,
          bool ReuseFreqsFrontPart,
          typename gmem_t>
__device__ __forceinline__ void load_cos_sin_cached_vec(float (&cos_0)[VecPairs],
                                                        float (&sin_0)[VecPairs],
                                                        float (&cos_1)[VecPairs],
                                                        float (&sin_1)[VecPairs],
                                                        gmem_t& g_cos,
                                                        gmem_t& g_sin,
                                                        const int32_t byte_offset_f,
                                                        const int32_t did,
                                                        const int32_t size_half_r)
{
#ifdef __HIP_DEVICE_COMPILE__
    constexpr int32_t elem_bytes = sizeof(typename gmem_t::scalar_type);
    if constexpr(RotateStyle == ROTATE_STYLE_NEOX)
    {
        auto v_c0 = g_cos.template _load<VecPairs>(byte_offset_f + did * elem_bytes);
        auto v_s0 = g_sin.template _load<VecPairs>(byte_offset_f + did * elem_bytes);

        if constexpr(ReuseFreqsFrontPart)
        {
            opus::static_for<VecPairs>([&](auto i) {
                cos_0[i.value] = float(v_c0[i.value]);
                sin_0[i.value] = float(v_s0[i.value]);
                cos_1[i.value] = cos_0[i.value];
                sin_1[i.value] = sin_0[i.value];
            });
        }
        else
        {
            auto v_c1 =
                g_cos.template _load<VecPairs>(byte_offset_f + (did + size_half_r) * elem_bytes);
            auto v_s1 =
                g_sin.template _load<VecPairs>(byte_offset_f + (did + size_half_r) * elem_bytes);
            opus::static_for<VecPairs>([&](auto i) {
                if constexpr(IsForward)
                {
                    cos_0[i.value] = float(v_c0[i.value]);
                    sin_0[i.value] = float(v_s0[i.value]);
                    cos_1[i.value] = float(v_c1[i.value]);
                    sin_1[i.value] = float(v_s1[i.value]);
                }
                else
                {
                    cos_0[i.value] = float(v_c0[i.value]);
                    sin_0[i.value] = float(v_s1[i.value]);
                    cos_1[i.value] = float(v_c1[i.value]);
                    sin_1[i.value] = float(v_s0[i.value]);
                }
            });
        }
    }
    else if constexpr(RotateStyle == ROTATE_STYLE_GPTJ)
    {
        if constexpr(ReuseFreqsFrontPart)
        {
            auto v_c = g_cos.template _load<VecPairs>(byte_offset_f + did * elem_bytes);
            auto v_s = g_sin.template _load<VecPairs>(byte_offset_f + did * elem_bytes);
            opus::static_for<VecPairs>([&](auto i) {
                cos_0[i.value] = float(v_c[i.value]);
                sin_0[i.value] = float(v_s[i.value]);
                cos_1[i.value] = cos_0[i.value];
                sin_1[i.value] = sin_0[i.value];
            });
        }
        else
        {
            // GPTJ non-reuse: cos/sin at did*2 and did*2+1
            if constexpr(VecPairs == 1)
            {
                auto v_c0 = g_cos.template _load<1>(byte_offset_f + (did * 2) * elem_bytes);
                auto v_c1 = g_cos.template _load<1>(byte_offset_f + (did * 2 + 1) * elem_bytes);
                auto v_s0 = g_sin.template _load<1>(byte_offset_f + (did * 2) * elem_bytes);
                auto v_s1 = g_sin.template _load<1>(byte_offset_f + (did * 2 + 1) * elem_bytes);
                float c0 = float(v_c0[0]), c1 = float(v_c1[0]);
                float s0 = float(v_s0[0]), s1 = float(v_s1[0]);
                if constexpr(IsForward)
                {
                    cos_0[0] = c0;
                    sin_0[0] = s0;
                    cos_1[0] = c1;
                    sin_1[0] = s1;
                }
                else
                {
                    cos_0[0] = c0;
                    sin_0[0] = s1;
                    cos_1[0] = c1;
                    sin_1[0] = s0;
                }
            }
            else
            {
                auto v_c_lo =
                    g_cos.template _load<VecPairs>(byte_offset_f + (did * 2) * elem_bytes);
                auto v_c_hi = g_cos.template _load<VecPairs>(byte_offset_f +
                                                             (did * 2 + VecPairs) * elem_bytes);
                auto v_s_lo =
                    g_sin.template _load<VecPairs>(byte_offset_f + (did * 2) * elem_bytes);
                auto v_s_hi = g_sin.template _load<VecPairs>(byte_offset_f +
                                                             (did * 2 + VecPairs) * elem_bytes);
                opus::static_for<VecPairs>([&](auto i) {
                    constexpr int idx0 = i.value * 2;
                    constexpr int idx1 = i.value * 2 + 1;
                    float c0, c1, s0, s1;
                    if constexpr(idx0 < VecPairs)
                    {
                        c0 = float(v_c_lo[idx0]);
                        c1 = float(v_c_lo[idx1]);
                        s0 = float(v_s_lo[idx0]);
                        s1 = float(v_s_lo[idx1]);
                    }
                    else
                    {
                        c0 = float(v_c_hi[idx0 - VecPairs]);
                        c1 = float(v_c_hi[idx1 - VecPairs]);
                        s0 = float(v_s_hi[idx0 - VecPairs]);
                        s1 = float(v_s_hi[idx1 - VecPairs]);
                    }
                    if constexpr(IsForward)
                    {
                        cos_0[i.value] = c0;
                        sin_0[i.value] = s0;
                        cos_1[i.value] = c1;
                        sin_1[i.value] = s1;
                    }
                    else
                    {
                        cos_0[i.value] = c0;
                        sin_0[i.value] = s1;
                        cos_1[i.value] = c1;
                        sin_1[i.value] = s0;
                    }
                });
            }
        }
    }
#endif
}

// load_payload_vec: loads VecPairs element-pairs from a pre-built gmem descriptor.
// g_buffer must be created from a UNIFORM (kernel-arg) base pointer to avoid waterfall loops.
// byte_offset_base = (offset_sb + hid * stride_h) * sizeof(element) -- the per-thread byte offset.
template <int32_t RotateStyle, int32_t VecPairs, typename o_scalar_t, typename gmem_t>
__device__ __forceinline__ void
load_payload_vec(o_scalar_t (&data_0)[VecPairs],
                 o_scalar_t (&data_1)[VecPairs],
                 gmem_t& g_buffer,
                 const int32_t byte_offset_base,
                 const int32_t did_pair,  // 0-based pair index into rotary portion
                 const int32_t did_start, // physical start of rotary portion
                 const int32_t size_half_r)
{
#ifdef __HIP_DEVICE_COMPILE__
    constexpr int32_t elem_bytes = sizeof(typename gmem_t::scalar_type);
    if constexpr(RotateStyle == ROTATE_STYLE_NEOX)
    {
        const int32_t did = did_pair + did_start;
        auto v0           = g_buffer.template _load<VecPairs>(byte_offset_base + did * elem_bytes);
        auto v1 =
            g_buffer.template _load<VecPairs>(byte_offset_base + (did + size_half_r) * elem_bytes);
        opus::static_for<VecPairs>([&](auto i) {
            data_0[i.value] = o_scalar_t(v0[i.value]);
            data_1[i.value] = o_scalar_t(v1[i.value]);
        });
    }
    else if constexpr(RotateStyle == ROTATE_STYLE_GPTJ)
    {
        // GPTJ layout: (even0, odd0, even1, odd1, ...) interleaved within rotary portion
        // Physical position = did_start + 2 * did_pair
        const int32_t phys = did_start + 2 * did_pair;
        if constexpr(VecPairs == 1)
        {
            auto v0   = g_buffer.template _load<1>(byte_offset_base + phys * elem_bytes);
            auto v1   = g_buffer.template _load<1>(byte_offset_base + (phys + 1) * elem_bytes);
            data_0[0] = o_scalar_t(v0[0]);
            data_1[0] = o_scalar_t(v1[0]);
        }
        else
        {
            auto v_lo = g_buffer.template _load<VecPairs>(byte_offset_base + phys * elem_bytes);
            auto v_hi = g_buffer.template _load<VecPairs>(byte_offset_base +
                                                          (phys + VecPairs) * elem_bytes);
            opus::static_for<VecPairs>([&](auto i) {
                constexpr int idx0 = i.value * 2;
                constexpr int idx1 = i.value * 2 + 1;
                if constexpr(idx0 < VecPairs)
                {
                    data_0[i.value] = o_scalar_t(v_lo[idx0]);
                    data_1[i.value] = o_scalar_t(v_lo[idx1]);
                }
                else
                {
                    data_0[i.value] = o_scalar_t(v_hi[idx0 - VecPairs]);
                    data_1[i.value] = o_scalar_t(v_hi[idx1 - VecPairs]);
                }
            });
        }
    }
#endif
}

// store_payload_vec: stores VecPairs element-pairs via a pre-built gmem descriptor.
// g_buffer must be created from a UNIFORM (kernel-arg) base pointer to avoid waterfall loops.
// byte_offset_base = (offset_sb + hid * stride_h) * sizeof(element) -- the per-thread byte offset.
template <int32_t RotateStyle, int32_t VecPairs, typename gmem_t, typename i_scalar_t>
__device__ __forceinline__ void
store_payload_vec(gmem_t& g_buffer,
                  const i_scalar_t (&data_0)[VecPairs],
                  const i_scalar_t (&data_1)[VecPairs],
                  const int32_t byte_offset_base,
                  const int32_t did_pair,  // 0-based pair index into rotary portion
                  const int32_t did_start, // physical start of rotary portion
                  const int32_t size_half_r)
{
#ifdef __HIP_DEVICE_COMPILE__
    using store_scalar_t         = typename gmem_t::scalar_type;
    constexpr int32_t elem_bytes = sizeof(store_scalar_t);
    if constexpr(RotateStyle == ROTATE_STYLE_NEOX)
    {
        const int32_t did = did_pair + did_start;
        opus::vector_t<store_scalar_t, VecPairs> v0, v1;
        opus::static_for<VecPairs>([&](auto i) {
            v0[i.value] = store_scalar_t(data_0[i.value]);
            v1[i.value] = store_scalar_t(data_1[i.value]);
        });
        g_buffer.template _store<VecPairs>(v0, byte_offset_base + did * elem_bytes);
        g_buffer.template _store<VecPairs>(v1, byte_offset_base + (did + size_half_r) * elem_bytes);
    }
    else if constexpr(RotateStyle == ROTATE_STYLE_GPTJ)
    {
        const int32_t phys = did_start + 2 * did_pair;
        if constexpr(VecPairs == 1)
        {
            opus::vector_t<store_scalar_t, 1> v_lo, v_hi;
            v_lo[0] = store_scalar_t(data_0[0]);
            v_hi[0] = store_scalar_t(data_1[0]);
            g_buffer.template _store<1>(v_lo, byte_offset_base + phys * elem_bytes);
            g_buffer.template _store<1>(v_hi, byte_offset_base + (phys + 1) * elem_bytes);
        }
        else
        {
            opus::vector_t<store_scalar_t, VecPairs> v_lo, v_hi;
            opus::static_for<VecPairs>([&](auto i) {
                constexpr int idx0 = i.value * 2;
                constexpr int idx1 = i.value * 2 + 1;
                if constexpr(idx0 < VecPairs)
                {
                    v_lo[idx0] = store_scalar_t(data_0[i.value]);
                    v_lo[idx1] = store_scalar_t(data_1[i.value]);
                }
                else
                {
                    v_hi[idx0 - VecPairs] = store_scalar_t(data_0[i.value]);
                    v_hi[idx1 - VecPairs] = store_scalar_t(data_1[i.value]);
                }
            });
            g_buffer.template _store<VecPairs>(v_lo, byte_offset_base + phys * elem_bytes);
            g_buffer.template _store<VecPairs>(v_hi,
                                               byte_offset_base + (phys + VecPairs) * elem_bytes);
        }
    }
#endif
}

// =====================================================================================================================
// Kernel Functionalities
//

struct OpUncachedFwd
{
    template <int32_t RotateStyle,
              bool ReuseFreqsFrontPart,
              bool NopeFirst,
              bool Inplace,
              bool StrideDOutEq1,
              bool StrideDInEq1,
              int32_t VecPairs = 1,
              typename scalar_t,
              typename scalar_f_t>
    __device__ __forceinline__ static void
    apply_1c(const scalar_t* __restrict__ p_base_output,
             const scalar_t* __restrict__ p_base_input,
             const int32_t offset_o, // per-thread (s,b) offset in elements
             const int32_t offset_i,
             const scalar_f_t* __restrict__ p_base_freqs,
             const int32_t offset_f, // per-thread freq offset in elements
             const int32_t size_h,
             const int32_t size_d,
             const int32_t size_f,
             const int32_t stride_i_h,
             const int32_t stride_i_d,
             const int32_t stride_o_h,
             const int32_t stride_o_d,
             const int32_t d_chunk_idx,
             const int32_t threads_per_sb)
    {
        // rotate count
        const int32_t size_r      = ReuseFreqsFrontPart ? (size_f << 1) : size_f;
        const int32_t size_half_r = size_r >> 1;
        const int32_t did_start   = NopeFirst ? (size_d - size_r) : 0;
        const int32_t did_pair    = d_chunk_idx * VecPairs;
        const int32_t did         = did_pair + did_start;

#ifdef __HIP_DEVICE_COMPILE__
        // Create gmem descriptors ONCE from uniform base pointers (no waterfall)
        auto g_f                     = opus_gmem(p_base_freqs);
        auto g_i                     = opus_gmem(p_base_input);
        auto g_o                     = opus_gmem(p_base_output);
        constexpr int32_t elem_bytes = sizeof(scalar_t);
        const int32_t byte_offset_f  = offset_f * (int32_t)sizeof(scalar_f_t);

        // Load cos/sin once for this thread's VecPairs pairs
        float cos_0[VecPairs], sin_0[VecPairs], cos_1[VecPairs], sin_1[VecPairs];
        load_cos_sin_uncached_vec<RotateStyle, VecPairs, true, ReuseFreqsFrontPart>(
            cos_0, sin_0, cos_1, sin_1, g_f, byte_offset_f, did - did_start, size_half_r);

        // Loop over ALL heads
        for(int32_t hid = 0; hid < size_h; hid++)
        {
            float input_0[VecPairs], input_1[VecPairs];
            load_payload_vec<RotateStyle, VecPairs>(input_0,
                                                    input_1,
                                                    g_i,
                                                    (offset_i + hid * stride_i_h) * elem_bytes,
                                                    did_pair,
                                                    did_start,
                                                    size_half_r);

            float output_0[VecPairs], output_1[VecPairs];
            for(int32_t v = 0; v < VecPairs; v++)
            {
                output_0[v] = input_0[v] * cos_0[v] - input_1[v] * sin_0[v];
                output_1[v] = input_1[v] * cos_1[v] + input_0[v] * sin_1[v];
            }

            store_payload_vec<RotateStyle, VecPairs>(g_o,
                                                     output_0,
                                                     output_1,
                                                     (offset_o + hid * stride_o_h) * elem_bytes,
                                                     did_pair,
                                                     did_start,
                                                     size_half_r);
        }

        // the rest are just forwarded (nope copy, distributed round-robin)
        if constexpr(!Inplace)
        {
            const int32_t nope_start = NopeFirst ? 0 : size_r;
            const int32_t nope_end   = NopeFirst ? (size_d - size_r) : size_d;
            elementwise_copy<VecPairs>(g_i,
                                       g_o,
                                       offset_i,
                                       offset_o,
                                       const_cast<scalar_t*>(p_base_output),
                                       p_base_input,
                                       size_h,
                                       nope_start,
                                       nope_end,
                                       stride_i_h,
                                       stride_i_d,
                                       stride_o_h,
                                       stride_o_d,
                                       d_chunk_idx,
                                       threads_per_sb);
        }
#endif
    }

    template <int32_t RotateStyle,
              bool ReuseFreqsFrontPart,
              bool NopeFirst,
              bool Inplace,
              bool StrideDOutXEq1,
              bool StrideDOutYEq1,
              bool StrideDInXEq1,
              bool StrideDInYEq1,
              int32_t VecPairs = 1,
              typename scalar_t,
              typename scalar_f_t>
    __device__ __forceinline__ static void
    apply_2c(const scalar_t* __restrict__ p_base_output_x,
             const scalar_t* __restrict__ p_base_output_y,
             const scalar_t* __restrict__ p_base_input_x,
             const scalar_t* __restrict__ p_base_input_y,
             const int32_t offset_ox,
             const int32_t offset_oy,
             const int32_t offset_ix,
             const int32_t offset_iy,
             const scalar_f_t* __restrict__ p_base_freqs,
             const int32_t offset_f, // per-thread freq offset in elements
             const int32_t size_h_x,
             const int32_t size_h_y,
             const int32_t size_d,
             const int32_t size_f,
             const int32_t stride_ix_h,
             const int32_t stride_ix_d,
             const int32_t stride_iy_h,
             const int32_t stride_iy_d,
             const int32_t stride_ox_h,
             const int32_t stride_ox_d,
             const int32_t stride_oy_h,
             const int32_t stride_oy_d,
             const int32_t d_chunk_idx,
             const int32_t threads_per_sb)
    {
        // rotate count
        const int32_t size_r      = ReuseFreqsFrontPart ? (size_f << 1) : size_f;
        const int32_t size_half_r = size_r >> 1;
        const int32_t did_start   = NopeFirst ? (size_d - size_r) : 0;
        const int32_t size_min_h  = min(size_h_x, size_h_y);
        const int32_t did_pair    = d_chunk_idx * VecPairs;
        const int32_t did         = did_pair + did_start;

#ifdef __HIP_DEVICE_COMPILE__
        auto g_f                    = opus_gmem(p_base_freqs);
        auto g_ix                   = opus_gmem(p_base_input_x);
        auto g_iy                   = opus_gmem(p_base_input_y);
        auto g_ox                   = opus_gmem(p_base_output_x);
        auto g_oy                   = opus_gmem(p_base_output_y);
        const int32_t byte_offset_f = offset_f * (int32_t)sizeof(scalar_f_t);

        // Load cos/sin once for this thread's VecPairs pairs
        float cos_0[VecPairs], sin_0[VecPairs], cos_1[VecPairs], sin_1[VecPairs];
        load_cos_sin_uncached_vec<RotateStyle, VecPairs, true, ReuseFreqsFrontPart>(
            cos_0, sin_0, cos_1, sin_1, g_f, byte_offset_f, did - did_start, size_half_r);
        constexpr int32_t elem_bytes = sizeof(scalar_t);

        // Loop over shared heads (both x and y)
        for(int32_t hid = 0; hid < size_min_h; hid++)
        {
            const int32_t bo_ix = (offset_ix + hid * stride_ix_h) * elem_bytes;
            const int32_t bo_iy = (offset_iy + hid * stride_iy_h) * elem_bytes;

            float ix0[VecPairs], ix1[VecPairs], iy0[VecPairs], iy1[VecPairs];
            load_payload_vec<RotateStyle, VecPairs>(
                ix0, ix1, g_ix, bo_ix, did_pair, did_start, size_half_r);
            load_payload_vec<RotateStyle, VecPairs>(
                iy0, iy1, g_iy, bo_iy, did_pair, did_start, size_half_r);

            float ox0[VecPairs], ox1[VecPairs], oy0[VecPairs], oy1[VecPairs];
            for(int32_t v = 0; v < VecPairs; v++)
            {
                ox0[v] = ix0[v] * cos_0[v] - ix1[v] * sin_0[v];
                ox1[v] = ix1[v] * cos_1[v] + ix0[v] * sin_1[v];
                oy0[v] = iy0[v] * cos_0[v] - iy1[v] * sin_0[v];
                oy1[v] = iy1[v] * cos_1[v] + iy0[v] * sin_1[v];
            }

            store_payload_vec<RotateStyle, VecPairs>(g_ox,
                                                     ox0,
                                                     ox1,
                                                     (offset_ox + hid * stride_ox_h) * elem_bytes,
                                                     did_pair,
                                                     did_start,
                                                     size_half_r);
            store_payload_vec<RotateStyle, VecPairs>(g_oy,
                                                     oy0,
                                                     oy1,
                                                     (offset_oy + hid * stride_oy_h) * elem_bytes,
                                                     did_pair,
                                                     did_start,
                                                     size_half_r);
        }

        // Remaining x-only heads
        for(int32_t hid = size_min_h; hid < size_h_x; hid++)
        {
            float ix0[VecPairs], ix1[VecPairs];
            load_payload_vec<RotateStyle, VecPairs>(ix0,
                                                    ix1,
                                                    g_ix,
                                                    (offset_ix + hid * stride_ix_h) * elem_bytes,
                                                    did_pair,
                                                    did_start,
                                                    size_half_r);
            float ox0[VecPairs], ox1[VecPairs];
            for(int32_t v = 0; v < VecPairs; v++)
            {
                ox0[v] = ix0[v] * cos_0[v] - ix1[v] * sin_0[v];
                ox1[v] = ix1[v] * cos_1[v] + ix0[v] * sin_1[v];
            }
            store_payload_vec<RotateStyle, VecPairs>(g_ox,
                                                     ox0,
                                                     ox1,
                                                     (offset_ox + hid * stride_ox_h) * elem_bytes,
                                                     did_pair,
                                                     did_start,
                                                     size_half_r);
        }

        // Remaining y-only heads
        for(int32_t hid = size_min_h; hid < size_h_y; hid++)
        {
            float iy0[VecPairs], iy1[VecPairs];
            load_payload_vec<RotateStyle, VecPairs>(iy0,
                                                    iy1,
                                                    g_iy,
                                                    (offset_iy + hid * stride_iy_h) * elem_bytes,
                                                    did_pair,
                                                    did_start,
                                                    size_half_r);
            float oy0[VecPairs], oy1[VecPairs];
            for(int32_t v = 0; v < VecPairs; v++)
            {
                oy0[v] = iy0[v] * cos_0[v] - iy1[v] * sin_0[v];
                oy1[v] = iy1[v] * cos_1[v] + iy0[v] * sin_1[v];
            }
            store_payload_vec<RotateStyle, VecPairs>(g_oy,
                                                     oy0,
                                                     oy1,
                                                     (offset_oy + hid * stride_oy_h) * elem_bytes,
                                                     did_pair,
                                                     did_start,
                                                     size_half_r);
        }

        // the rest are just forwarded (nope copy, distributed round-robin)
        if constexpr(!Inplace)
        {
            const int32_t nope_start = NopeFirst ? 0 : size_r;
            const int32_t nope_end   = NopeFirst ? (size_d - size_r) : size_d;
            elementwise_copy_2c<VecPairs>(g_ix,
                                          g_iy,
                                          g_ox,
                                          g_oy,
                                          offset_ix,
                                          offset_iy,
                                          offset_ox,
                                          offset_oy,
                                          const_cast<scalar_t*>(p_base_output_x),
                                          const_cast<scalar_t*>(p_base_output_y),
                                          p_base_input_x,
                                          p_base_input_y,
                                          size_h_x,
                                          size_h_y,
                                          nope_start,
                                          nope_end,
                                          stride_ix_h,
                                          stride_ix_d,
                                          stride_iy_h,
                                          stride_iy_d,
                                          stride_ox_h,
                                          stride_ox_d,
                                          stride_oy_h,
                                          stride_oy_d,
                                          d_chunk_idx,
                                          threads_per_sb);
        }
#endif
    }
};

struct OpUncachedBwd
{
    template <int32_t RotateStyle,
              bool ReuseFreqsFrontPart,
              bool NopeFirst,
              bool Inplace,
              bool StrideDInGradsEq1,
              bool StrideDOutGradsEq1,
              int32_t VecPairs = 1,
              typename scalar_t,
              typename scalar_f_t>
    __device__ __forceinline__ static void
    apply_1c(const scalar_t* __restrict__ p_base_input_grads,
             const scalar_t* __restrict__ p_base_output_grads,
             const int32_t offset_ig,
             const int32_t offset_og,
             const scalar_f_t* __restrict__ p_base_freqs,
             const int32_t offset_f, // per-thread freq offset in elements
             const int32_t size_h,
             const int32_t size_d,
             const int32_t size_f,
             const int32_t stride_o_h,
             const int32_t stride_o_d,
             const int32_t stride_i_h,
             const int32_t stride_i_d,
             const int32_t d_chunk_idx,
             const int32_t threads_per_sb)
    {
        // rotate count
        const int32_t size_r      = ReuseFreqsFrontPart ? (size_f << 1) : size_f;
        const int32_t size_half_r = size_r >> 1;
        const int32_t did_start   = NopeFirst ? (size_d - size_r) : 0;
        const int32_t did_pair    = d_chunk_idx * VecPairs;
        const int32_t did         = did_pair + did_start;

#ifdef __HIP_DEVICE_COMPILE__
        auto g_f                     = opus_gmem(p_base_freqs);
        auto g_og                    = opus_gmem(p_base_output_grads);
        auto g_ig                    = opus_gmem(p_base_input_grads);
        constexpr int32_t elem_bytes = sizeof(scalar_t);
        const int32_t byte_offset_f  = offset_f * (int32_t)sizeof(scalar_f_t);

        // Load cos/sin once for this thread's VecPairs pairs
        float cos_0[VecPairs], sin_0[VecPairs], cos_1[VecPairs], sin_1[VecPairs];
        load_cos_sin_uncached_vec<RotateStyle, VecPairs, false, ReuseFreqsFrontPart>(
            cos_0, sin_0, cos_1, sin_1, g_f, byte_offset_f, did - did_start, size_half_r);

        // Loop over ALL heads
        for(int32_t hid = 0; hid < size_h; hid++)
        {
            float og0[VecPairs], og1[VecPairs];
            load_payload_vec<RotateStyle, VecPairs>(og0,
                                                    og1,
                                                    g_og,
                                                    (offset_og + hid * stride_o_h) * elem_bytes,
                                                    did_pair,
                                                    did_start,
                                                    size_half_r);

            float ig0[VecPairs], ig1[VecPairs];
            for(int32_t v = 0; v < VecPairs; v++)
            {
                ig0[v] = og0[v] * cos_0[v] + og1[v] * sin_0[v];
                ig1[v] = og1[v] * cos_1[v] - og0[v] * sin_1[v];
            }

            store_payload_vec<RotateStyle, VecPairs>(g_ig,
                                                     ig0,
                                                     ig1,
                                                     (offset_ig + hid * stride_i_h) * elem_bytes,
                                                     did_pair,
                                                     did_start,
                                                     size_half_r);
        }

        // the rest are just forwarded (nope copy, distributed round-robin)
        if constexpr(!Inplace)
        {
            const int32_t nope_start = NopeFirst ? 0 : size_r;
            const int32_t nope_end   = NopeFirst ? (size_d - size_r) : size_d;
            elementwise_copy<VecPairs>(g_og,
                                       g_ig,
                                       offset_og,
                                       offset_ig,
                                       const_cast<scalar_t*>(p_base_input_grads),
                                       p_base_output_grads,
                                       size_h,
                                       nope_start,
                                       nope_end,
                                       stride_o_h,
                                       stride_o_d,
                                       stride_i_h,
                                       stride_i_d,
                                       d_chunk_idx,
                                       threads_per_sb);
        }
#endif
    }

    template <int32_t RotateStyle,
              bool ReuseFreqsFrontPart,
              bool NopeFirst,
              bool Inplace,
              bool StrideDInGradsXEq1,
              bool StrideDInGradsYEq1,
              bool StrideDOutGradsXEq1,
              bool StrideDOutGradsYEq1,
              int32_t VecPairs = 1,
              typename scalar_t,
              typename scalar_f_t>
    __device__ __forceinline__ static void
    apply_2c(const scalar_t* __restrict__ p_base_input_grads_x,
             const scalar_t* __restrict__ p_base_input_grads_y,
             const scalar_t* __restrict__ p_base_output_grads_x,
             const scalar_t* __restrict__ p_base_output_grads_y,
             const int32_t offset_igx,
             const int32_t offset_igy,
             const int32_t offset_ogx,
             const int32_t offset_ogy,
             const scalar_f_t* __restrict__ p_base_freqs,
             const int32_t offset_f, // per-thread freq offset in elements
             const int32_t size_h_x,
             const int32_t size_h_y,
             const int32_t size_d,
             const int32_t size_f,
             const int32_t stride_ox_h,
             const int32_t stride_ox_d,
             const int32_t stride_oy_h,
             const int32_t stride_oy_d,
             const int32_t stride_ix_h,
             const int32_t stride_ix_d,
             const int32_t stride_iy_h,
             const int32_t stride_iy_d,
             const int32_t d_chunk_idx,
             const int32_t threads_per_sb)
    {
        // rotate count
        const int32_t size_r      = ReuseFreqsFrontPart ? (size_f << 1) : size_f;
        const int32_t size_half_r = size_r >> 1;
        const int32_t did_start   = NopeFirst ? (size_d - size_r) : 0;
        const int32_t size_min_h  = min(size_h_x, size_h_y);
        const int32_t did_pair    = d_chunk_idx * VecPairs;
        const int32_t did         = did_pair + did_start;

#ifdef __HIP_DEVICE_COMPILE__
        auto g_f                     = opus_gmem(p_base_freqs);
        auto g_ogx                   = opus_gmem(p_base_output_grads_x);
        auto g_ogy                   = opus_gmem(p_base_output_grads_y);
        auto g_igx                   = opus_gmem(p_base_input_grads_x);
        auto g_igy                   = opus_gmem(p_base_input_grads_y);
        constexpr int32_t elem_bytes = sizeof(scalar_t);
        const int32_t byte_offset_f  = offset_f * (int32_t)sizeof(scalar_f_t);

        // Load cos/sin once for this thread's VecPairs pairs
        float cos_0[VecPairs], sin_0[VecPairs], cos_1[VecPairs], sin_1[VecPairs];
        load_cos_sin_uncached_vec<RotateStyle, VecPairs, false, ReuseFreqsFrontPart>(
            cos_0, sin_0, cos_1, sin_1, g_f, byte_offset_f, did - did_start, size_half_r);

        // Loop over shared heads (both x and y)
        for(int32_t hid = 0; hid < size_min_h; hid++)
        {
            const int32_t bo_ogx = (offset_ogx + hid * stride_ox_h) * elem_bytes;
            const int32_t bo_ogy = (offset_ogy + hid * stride_oy_h) * elem_bytes;

            float ogx0[VecPairs], ogx1[VecPairs], ogy0[VecPairs], ogy1[VecPairs];
            load_payload_vec<RotateStyle, VecPairs>(
                ogx0, ogx1, g_ogx, bo_ogx, did_pair, did_start, size_half_r);
            load_payload_vec<RotateStyle, VecPairs>(
                ogy0, ogy1, g_ogy, bo_ogy, did_pair, did_start, size_half_r);

            float igx0[VecPairs], igx1[VecPairs], igy0[VecPairs], igy1[VecPairs];
            for(int32_t v = 0; v < VecPairs; v++)
            {
                igx0[v] = ogx0[v] * cos_0[v] + ogx1[v] * sin_0[v];
                igx1[v] = ogx1[v] * cos_1[v] - ogx0[v] * sin_1[v];
                igy0[v] = ogy0[v] * cos_0[v] + ogy1[v] * sin_0[v];
                igy1[v] = ogy1[v] * cos_1[v] - ogy0[v] * sin_1[v];
            }

            store_payload_vec<RotateStyle, VecPairs>(g_igx,
                                                     igx0,
                                                     igx1,
                                                     (offset_igx + hid * stride_ix_h) * elem_bytes,
                                                     did_pair,
                                                     did_start,
                                                     size_half_r);
            store_payload_vec<RotateStyle, VecPairs>(g_igy,
                                                     igy0,
                                                     igy1,
                                                     (offset_igy + hid * stride_iy_h) * elem_bytes,
                                                     did_pair,
                                                     did_start,
                                                     size_half_r);
        }

        // Remaining x-only heads
        for(int32_t hid = size_min_h; hid < size_h_x; hid++)
        {
            float ogx0[VecPairs], ogx1[VecPairs];
            load_payload_vec<RotateStyle, VecPairs>(ogx0,
                                                    ogx1,
                                                    g_ogx,
                                                    (offset_ogx + hid * stride_ox_h) * elem_bytes,
                                                    did_pair,
                                                    did_start,
                                                    size_half_r);
            float igx0[VecPairs], igx1[VecPairs];
            for(int32_t v = 0; v < VecPairs; v++)
            {
                igx0[v] = ogx0[v] * cos_0[v] + ogx1[v] * sin_0[v];
                igx1[v] = ogx1[v] * cos_1[v] - ogx0[v] * sin_1[v];
            }
            store_payload_vec<RotateStyle, VecPairs>(g_igx,
                                                     igx0,
                                                     igx1,
                                                     (offset_igx + hid * stride_ix_h) * elem_bytes,
                                                     did_pair,
                                                     did_start,
                                                     size_half_r);
        }

        // Remaining y-only heads
        for(int32_t hid = size_min_h; hid < size_h_y; hid++)
        {
            float ogy0[VecPairs], ogy1[VecPairs];
            load_payload_vec<RotateStyle, VecPairs>(ogy0,
                                                    ogy1,
                                                    g_ogy,
                                                    (offset_ogy + hid * stride_oy_h) * elem_bytes,
                                                    did_pair,
                                                    did_start,
                                                    size_half_r);
            float igy0[VecPairs], igy1[VecPairs];
            for(int32_t v = 0; v < VecPairs; v++)
            {
                igy0[v] = ogy0[v] * cos_0[v] + ogy1[v] * sin_0[v];
                igy1[v] = ogy1[v] * cos_1[v] - ogy0[v] * sin_1[v];
            }
            store_payload_vec<RotateStyle, VecPairs>(g_igy,
                                                     igy0,
                                                     igy1,
                                                     (offset_igy + hid * stride_iy_h) * elem_bytes,
                                                     did_pair,
                                                     did_start,
                                                     size_half_r);
        }

        // the rest are just forwarded (nope copy, distributed round-robin)
        if constexpr(!Inplace)
        {
            const int32_t nope_start = NopeFirst ? 0 : size_r;
            const int32_t nope_end   = NopeFirst ? (size_d - size_r) : size_d;
            elementwise_copy_2c<VecPairs>(g_ogx,
                                          g_ogy,
                                          g_igx,
                                          g_igy,
                                          offset_ogx,
                                          offset_ogy,
                                          offset_igx,
                                          offset_igy,
                                          const_cast<scalar_t*>(p_base_input_grads_x),
                                          const_cast<scalar_t*>(p_base_input_grads_y),
                                          p_base_output_grads_x,
                                          p_base_output_grads_y,
                                          size_h_x,
                                          size_h_y,
                                          nope_start,
                                          nope_end,
                                          stride_ox_h,
                                          stride_ox_d,
                                          stride_oy_h,
                                          stride_oy_d,
                                          stride_ix_h,
                                          stride_ix_d,
                                          stride_iy_h,
                                          stride_iy_d,
                                          d_chunk_idx,
                                          threads_per_sb);
        }
#endif
    }
};

struct OpCachedFwd
{
    template <int32_t RotateStyle,
              bool ReuseFreqsFrontPart,
              bool NopeFirst,
              bool Inplace,
              bool StrideDOutEq1,
              bool StrideDInEq1,
              int32_t VecPairs = 1,
              typename scalar_t,
              typename scalar_f_t>
    __device__ __forceinline__ static void
    apply_1c(const scalar_t* __restrict__ p_base_output,
             const scalar_t* __restrict__ p_base_input,
             const int32_t offset_o,
             const int32_t offset_i,
             const scalar_f_t* __restrict__ p_base_cos,
             const scalar_f_t* __restrict__ p_base_sin,
             const int32_t offset_f, // per-thread cos/sin offset in elements
             const int32_t size_h,
             const int32_t size_d,
             const int32_t size_f,
             const int32_t stride_i_h,
             const int32_t stride_i_d,
             const int32_t stride_o_h,
             const int32_t stride_o_d,
             const int32_t d_chunk_idx,
             const int32_t threads_per_sb)
    {
        // rotate count
        const int32_t size_r      = ReuseFreqsFrontPart ? (size_f << 1) : size_f;
        const int32_t size_half_r = size_r >> 1;
        const int32_t did_start   = NopeFirst ? (size_d - size_r) : 0;
        const int32_t did_pair    = d_chunk_idx * VecPairs;
        const int32_t did         = did_pair + did_start;

#ifdef __HIP_DEVICE_COMPILE__
        auto g_c                     = opus_gmem(p_base_cos);
        auto g_s                     = opus_gmem(p_base_sin);
        auto g_i                     = opus_gmem(p_base_input);
        auto g_o                     = opus_gmem(p_base_output);
        constexpr int32_t elem_bytes = sizeof(scalar_t);
        const int32_t byte_offset_f  = offset_f * (int32_t)sizeof(scalar_f_t);

        // Load cos/sin once for this thread's VecPairs pairs
        float cos_0[VecPairs], sin_0[VecPairs], cos_1[VecPairs], sin_1[VecPairs];
        load_cos_sin_cached_vec<RotateStyle, VecPairs, true, ReuseFreqsFrontPart>(
            cos_0, sin_0, cos_1, sin_1, g_c, g_s, byte_offset_f, did - did_start, size_half_r);

        // Loop over ALL heads
        for(int32_t hid = 0; hid < size_h; hid++)
        {
            float input_0[VecPairs], input_1[VecPairs];
            load_payload_vec<RotateStyle, VecPairs>(input_0,
                                                    input_1,
                                                    g_i,
                                                    (offset_i + hid * stride_i_h) * elem_bytes,
                                                    did_pair,
                                                    did_start,
                                                    size_half_r);

            float output_0[VecPairs], output_1[VecPairs];
            for(int32_t v = 0; v < VecPairs; v++)
            {
                output_0[v] = input_0[v] * cos_0[v] - input_1[v] * sin_0[v];
                output_1[v] = input_1[v] * cos_1[v] + input_0[v] * sin_1[v];
            }

            store_payload_vec<RotateStyle, VecPairs>(g_o,
                                                     output_0,
                                                     output_1,
                                                     (offset_o + hid * stride_o_h) * elem_bytes,
                                                     did_pair,
                                                     did_start,
                                                     size_half_r);
        }

        // the rest are just forwarded (nope copy, distributed round-robin)
        if constexpr(!Inplace)
        {
            const int32_t nope_start = NopeFirst ? 0 : size_r;
            const int32_t nope_end   = NopeFirst ? (size_d - size_r) : size_d;
            elementwise_copy<VecPairs>(g_i,
                                       g_o,
                                       offset_i,
                                       offset_o,
                                       const_cast<scalar_t*>(p_base_output),
                                       p_base_input,
                                       size_h,
                                       nope_start,
                                       nope_end,
                                       stride_i_h,
                                       stride_i_d,
                                       stride_o_h,
                                       stride_o_d,
                                       d_chunk_idx,
                                       threads_per_sb);
        }
#endif
    }

    template <int32_t RotateStyle,
              bool ReuseFreqsFrontPart,
              bool NopeFirst,
              bool Inplace,
              bool StrideDOutXEq1,
              bool StrideDOutYEq1,
              bool StrideDInXEq1,
              bool StrideDInYEq1,
              int32_t VecPairs = 1,
              typename scalar_t,
              typename scalar_f_t>
    __device__ __forceinline__ static void
    apply_2c(const scalar_t* __restrict__ p_base_output_x,
             const scalar_t* __restrict__ p_base_output_y,
             const scalar_t* __restrict__ p_base_input_x,
             const scalar_t* __restrict__ p_base_input_y,
             const int32_t offset_ox,
             const int32_t offset_oy,
             const int32_t offset_ix,
             const int32_t offset_iy,
             const scalar_f_t* __restrict__ p_base_cos,
             const scalar_f_t* __restrict__ p_base_sin,
             const int32_t offset_f, // per-thread cos/sin offset in elements
             const int32_t size_h_x,
             const int32_t size_h_y,
             const int32_t size_d,
             const int32_t size_f,
             const int32_t stride_ix_h,
             const int32_t stride_ix_d,
             const int32_t stride_iy_h,
             const int32_t stride_iy_d,
             const int32_t stride_ox_h,
             const int32_t stride_ox_d,
             const int32_t stride_oy_h,
             const int32_t stride_oy_d,
             const int32_t d_chunk_idx,
             const int32_t threads_per_sb)
    {
        // rotate count
        const int32_t size_r      = ReuseFreqsFrontPart ? (size_f << 1) : size_f;
        const int32_t size_half_r = size_r >> 1;
        const int32_t did_start   = NopeFirst ? (size_d - size_r) : 0;
        const int32_t size_min_h  = min(size_h_x, size_h_y);
        const int32_t did_pair    = d_chunk_idx * VecPairs;
        const int32_t did         = did_pair + did_start;

#ifdef __HIP_DEVICE_COMPILE__
        auto g_c                     = opus_gmem(p_base_cos);
        auto g_s                     = opus_gmem(p_base_sin);
        auto g_ix                    = opus_gmem(p_base_input_x);
        auto g_iy                    = opus_gmem(p_base_input_y);
        auto g_ox                    = opus_gmem(p_base_output_x);
        auto g_oy                    = opus_gmem(p_base_output_y);
        constexpr int32_t elem_bytes = sizeof(scalar_t);
        const int32_t byte_offset_f  = offset_f * (int32_t)sizeof(scalar_f_t);

        // Load cos/sin once for this thread's VecPairs pairs
        float cos_0[VecPairs], sin_0[VecPairs], cos_1[VecPairs], sin_1[VecPairs];
        load_cos_sin_cached_vec<RotateStyle, VecPairs, true, ReuseFreqsFrontPart>(
            cos_0, sin_0, cos_1, sin_1, g_c, g_s, byte_offset_f, did - did_start, size_half_r);

        // Loop over shared heads (both x and y)
        for(int32_t hid = 0; hid < size_min_h; hid++)
        {
            const int32_t bo_ix = (offset_ix + hid * stride_ix_h) * elem_bytes;
            const int32_t bo_iy = (offset_iy + hid * stride_iy_h) * elem_bytes;

            float ix0[VecPairs], ix1[VecPairs], iy0[VecPairs], iy1[VecPairs];
            load_payload_vec<RotateStyle, VecPairs>(
                ix0, ix1, g_ix, bo_ix, did_pair, did_start, size_half_r);
            load_payload_vec<RotateStyle, VecPairs>(
                iy0, iy1, g_iy, bo_iy, did_pair, did_start, size_half_r);

            float ox0[VecPairs], ox1[VecPairs], oy0[VecPairs], oy1[VecPairs];
            for(int32_t v = 0; v < VecPairs; v++)
            {
                ox0[v] = ix0[v] * cos_0[v] - ix1[v] * sin_0[v];
                ox1[v] = ix1[v] * cos_1[v] + ix0[v] * sin_1[v];
                oy0[v] = iy0[v] * cos_0[v] - iy1[v] * sin_0[v];
                oy1[v] = iy1[v] * cos_1[v] + iy0[v] * sin_1[v];
            }

            store_payload_vec<RotateStyle, VecPairs>(g_ox,
                                                     ox0,
                                                     ox1,
                                                     (offset_ox + hid * stride_ox_h) * elem_bytes,
                                                     did_pair,
                                                     did_start,
                                                     size_half_r);
            store_payload_vec<RotateStyle, VecPairs>(g_oy,
                                                     oy0,
                                                     oy1,
                                                     (offset_oy + hid * stride_oy_h) * elem_bytes,
                                                     did_pair,
                                                     did_start,
                                                     size_half_r);
        }

        // Remaining x-only heads
        for(int32_t hid = size_min_h; hid < size_h_x; hid++)
        {
            float ix0[VecPairs], ix1[VecPairs];
            load_payload_vec<RotateStyle, VecPairs>(ix0,
                                                    ix1,
                                                    g_ix,
                                                    (offset_ix + hid * stride_ix_h) * elem_bytes,
                                                    did_pair,
                                                    did_start,
                                                    size_half_r);
            float ox0[VecPairs], ox1[VecPairs];
            for(int32_t v = 0; v < VecPairs; v++)
            {
                ox0[v] = ix0[v] * cos_0[v] - ix1[v] * sin_0[v];
                ox1[v] = ix1[v] * cos_1[v] + ix0[v] * sin_1[v];
            }
            store_payload_vec<RotateStyle, VecPairs>(g_ox,
                                                     ox0,
                                                     ox1,
                                                     (offset_ox + hid * stride_ox_h) * elem_bytes,
                                                     did_pair,
                                                     did_start,
                                                     size_half_r);
        }

        // Remaining y-only heads
        for(int32_t hid = size_min_h; hid < size_h_y; hid++)
        {
            float iy0[VecPairs], iy1[VecPairs];
            load_payload_vec<RotateStyle, VecPairs>(iy0,
                                                    iy1,
                                                    g_iy,
                                                    (offset_iy + hid * stride_iy_h) * elem_bytes,
                                                    did_pair,
                                                    did_start,
                                                    size_half_r);
            float oy0[VecPairs], oy1[VecPairs];
            for(int32_t v = 0; v < VecPairs; v++)
            {
                oy0[v] = iy0[v] * cos_0[v] - iy1[v] * sin_0[v];
                oy1[v] = iy1[v] * cos_1[v] + iy0[v] * sin_1[v];
            }
            store_payload_vec<RotateStyle, VecPairs>(g_oy,
                                                     oy0,
                                                     oy1,
                                                     (offset_oy + hid * stride_oy_h) * elem_bytes,
                                                     did_pair,
                                                     did_start,
                                                     size_half_r);
        }

        // the rest are just forwarded (nope copy, distributed round-robin)
        if constexpr(!Inplace)
        {
            const int32_t nope_start = NopeFirst ? 0 : size_r;
            const int32_t nope_end   = NopeFirst ? (size_d - size_r) : size_d;
            elementwise_copy_2c<VecPairs>(g_ix,
                                          g_iy,
                                          g_ox,
                                          g_oy,
                                          offset_ix,
                                          offset_iy,
                                          offset_ox,
                                          offset_oy,
                                          const_cast<scalar_t*>(p_base_output_x),
                                          const_cast<scalar_t*>(p_base_output_y),
                                          p_base_input_x,
                                          p_base_input_y,
                                          size_h_x,
                                          size_h_y,
                                          nope_start,
                                          nope_end,
                                          stride_ix_h,
                                          stride_ix_d,
                                          stride_iy_h,
                                          stride_iy_d,
                                          stride_ox_h,
                                          stride_ox_d,
                                          stride_oy_h,
                                          stride_oy_d,
                                          d_chunk_idx,
                                          threads_per_sb);
        }
#endif
    }
};

struct OpCachedBwd
{
    template <int32_t RotateStyle,
              bool ReuseFreqsFrontPart,
              bool NopeFirst,
              bool Inplace,
              bool StrideDInGradsEq1,
              bool StrideDOutGradsEq1,
              int32_t VecPairs = 1,
              typename scalar_t,
              typename scalar_f_t>
    __device__ __forceinline__ static void
    apply_1c(const scalar_t* __restrict__ p_base_input_grads,
             const scalar_t* __restrict__ p_base_output_grads,
             const int32_t offset_ig,
             const int32_t offset_og,
             const scalar_f_t* __restrict__ p_base_cos,
             const scalar_f_t* __restrict__ p_base_sin,
             const int32_t offset_f, // per-thread cos/sin offset in elements
             const int32_t size_h,
             const int32_t size_d,
             const int32_t size_f,
             const int32_t stride_o_h,
             const int32_t stride_o_d,
             const int32_t stride_i_h,
             const int32_t stride_i_d,
             const int32_t d_chunk_idx,
             const int32_t threads_per_sb)
    {
        // rotate count
        const int32_t size_r      = ReuseFreqsFrontPart ? (size_f << 1) : size_f;
        const int32_t size_half_r = size_r >> 1;
        const int32_t did_start   = NopeFirst ? (size_d - size_r) : 0;
        const int32_t did_pair    = d_chunk_idx * VecPairs;
        const int32_t did         = did_pair + did_start;

#ifdef __HIP_DEVICE_COMPILE__
        auto g_c                     = opus_gmem(p_base_cos);
        auto g_s                     = opus_gmem(p_base_sin);
        auto g_og                    = opus_gmem(p_base_output_grads);
        auto g_ig                    = opus_gmem(p_base_input_grads);
        constexpr int32_t elem_bytes = sizeof(scalar_t);
        const int32_t byte_offset_f  = offset_f * (int32_t)sizeof(scalar_f_t);

        // Load cos/sin once for this thread's VecPairs pairs
        float cos_0[VecPairs], sin_0[VecPairs], cos_1[VecPairs], sin_1[VecPairs];
        load_cos_sin_cached_vec<RotateStyle, VecPairs, false, ReuseFreqsFrontPart>(
            cos_0, sin_0, cos_1, sin_1, g_c, g_s, byte_offset_f, did - did_start, size_half_r);

        // Loop over ALL heads
        for(int32_t hid = 0; hid < size_h; hid++)
        {
            float og0[VecPairs], og1[VecPairs];
            load_payload_vec<RotateStyle, VecPairs>(og0,
                                                    og1,
                                                    g_og,
                                                    (offset_og + hid * stride_o_h) * elem_bytes,
                                                    did_pair,
                                                    did_start,
                                                    size_half_r);

            float ig0[VecPairs], ig1[VecPairs];
            for(int32_t v = 0; v < VecPairs; v++)
            {
                ig0[v] = og0[v] * cos_0[v] + og1[v] * sin_0[v];
                ig1[v] = og1[v] * cos_1[v] - og0[v] * sin_1[v];
            }

            store_payload_vec<RotateStyle, VecPairs>(g_ig,
                                                     ig0,
                                                     ig1,
                                                     (offset_ig + hid * stride_i_h) * elem_bytes,
                                                     did_pair,
                                                     did_start,
                                                     size_half_r);
        }

        // the rest are just forwarded (nope copy, distributed round-robin)
        if constexpr(!Inplace)
        {
            const int32_t nope_start = NopeFirst ? 0 : size_r;
            const int32_t nope_end   = NopeFirst ? (size_d - size_r) : size_d;
            elementwise_copy<VecPairs>(g_og,
                                       g_ig,
                                       offset_og,
                                       offset_ig,
                                       const_cast<scalar_t*>(p_base_input_grads),
                                       p_base_output_grads,
                                       size_h,
                                       nope_start,
                                       nope_end,
                                       stride_o_h,
                                       stride_o_d,
                                       stride_i_h,
                                       stride_i_d,
                                       d_chunk_idx,
                                       threads_per_sb);
        }
#endif
    }

    template <int32_t RotateStyle,
              bool ReuseFreqsFrontPart,
              bool NopeFirst,
              bool Inplace,
              bool StrideDInGradsXEq1,
              bool StrideDInGradsYEq1,
              bool StrideDOutGradsXEq1,
              bool StrideDOutGradsYEq1,
              int32_t VecPairs = 1,
              typename scalar_t,
              typename scalar_f_t>
    __device__ __forceinline__ static void
    apply_2c(const scalar_t* __restrict__ p_base_input_grads_x,
             const scalar_t* __restrict__ p_base_input_grads_y,
             const scalar_t* __restrict__ p_base_output_grads_x,
             const scalar_t* __restrict__ p_base_output_grads_y,
             const int32_t offset_igx,
             const int32_t offset_igy,
             const int32_t offset_ogx,
             const int32_t offset_ogy,
             const scalar_f_t* __restrict__ p_base_cos,
             const scalar_f_t* __restrict__ p_base_sin,
             const int32_t offset_f, // per-thread cos/sin offset in elements
             const int32_t size_h_x,
             const int32_t size_h_y,
             const int32_t size_d,
             const int32_t size_f,
             const int32_t stride_ox_h,
             const int32_t stride_ox_d,
             const int32_t stride_oy_h,
             const int32_t stride_oy_d,
             const int32_t stride_ix_h,
             const int32_t stride_ix_d,
             const int32_t stride_iy_h,
             const int32_t stride_iy_d,
             const int32_t d_chunk_idx,
             const int32_t threads_per_sb)
    {
        // rotate count
        const int32_t size_r      = ReuseFreqsFrontPart ? (size_f << 1) : size_f;
        const int32_t size_half_r = size_r >> 1;
        const int32_t did_start   = NopeFirst ? (size_d - size_r) : 0;
        const int32_t size_min_h  = min(size_h_x, size_h_y);
        const int32_t did_pair    = d_chunk_idx * VecPairs;
        const int32_t did         = did_pair + did_start;

#ifdef __HIP_DEVICE_COMPILE__
        auto g_c                     = opus_gmem(p_base_cos);
        auto g_s                     = opus_gmem(p_base_sin);
        auto g_ogx                   = opus_gmem(p_base_output_grads_x);
        auto g_ogy                   = opus_gmem(p_base_output_grads_y);
        auto g_igx                   = opus_gmem(p_base_input_grads_x);
        auto g_igy                   = opus_gmem(p_base_input_grads_y);
        constexpr int32_t elem_bytes = sizeof(scalar_t);
        const int32_t byte_offset_f  = offset_f * (int32_t)sizeof(scalar_f_t);

        // Load cos/sin once for this thread's VecPairs pairs
        float cos_0[VecPairs], sin_0[VecPairs], cos_1[VecPairs], sin_1[VecPairs];
        load_cos_sin_cached_vec<RotateStyle, VecPairs, false, ReuseFreqsFrontPart>(
            cos_0, sin_0, cos_1, sin_1, g_c, g_s, byte_offset_f, did - did_start, size_half_r);

        // Loop over shared heads (both x and y)
        for(int32_t hid = 0; hid < size_min_h; hid++)
        {
            const int32_t bo_ogx = (offset_ogx + hid * stride_ox_h) * elem_bytes;
            const int32_t bo_ogy = (offset_ogy + hid * stride_oy_h) * elem_bytes;

            float ogx0[VecPairs], ogx1[VecPairs], ogy0[VecPairs], ogy1[VecPairs];
            load_payload_vec<RotateStyle, VecPairs>(
                ogx0, ogx1, g_ogx, bo_ogx, did_pair, did_start, size_half_r);
            load_payload_vec<RotateStyle, VecPairs>(
                ogy0, ogy1, g_ogy, bo_ogy, did_pair, did_start, size_half_r);

            float igx0[VecPairs], igx1[VecPairs], igy0[VecPairs], igy1[VecPairs];
            for(int32_t v = 0; v < VecPairs; v++)
            {
                igx0[v] = ogx0[v] * cos_0[v] + ogx1[v] * sin_0[v];
                igx1[v] = ogx1[v] * cos_1[v] - ogx0[v] * sin_1[v];
                igy0[v] = ogy0[v] * cos_0[v] + ogy1[v] * sin_0[v];
                igy1[v] = ogy1[v] * cos_1[v] - ogy0[v] * sin_1[v];
            }

            store_payload_vec<RotateStyle, VecPairs>(g_igx,
                                                     igx0,
                                                     igx1,
                                                     (offset_igx + hid * stride_ix_h) * elem_bytes,
                                                     did_pair,
                                                     did_start,
                                                     size_half_r);
            store_payload_vec<RotateStyle, VecPairs>(g_igy,
                                                     igy0,
                                                     igy1,
                                                     (offset_igy + hid * stride_iy_h) * elem_bytes,
                                                     did_pair,
                                                     did_start,
                                                     size_half_r);
        }

        // Remaining x-only heads
        for(int32_t hid = size_min_h; hid < size_h_x; hid++)
        {
            float ogx0[VecPairs], ogx1[VecPairs];
            load_payload_vec<RotateStyle, VecPairs>(ogx0,
                                                    ogx1,
                                                    g_ogx,
                                                    (offset_ogx + hid * stride_ox_h) * elem_bytes,
                                                    did_pair,
                                                    did_start,
                                                    size_half_r);
            float igx0[VecPairs], igx1[VecPairs];
            for(int32_t v = 0; v < VecPairs; v++)
            {
                igx0[v] = ogx0[v] * cos_0[v] + ogx1[v] * sin_0[v];
                igx1[v] = ogx1[v] * cos_1[v] - ogx0[v] * sin_1[v];
            }
            store_payload_vec<RotateStyle, VecPairs>(g_igx,
                                                     igx0,
                                                     igx1,
                                                     (offset_igx + hid * stride_ix_h) * elem_bytes,
                                                     did_pair,
                                                     did_start,
                                                     size_half_r);
        }

        // Remaining y-only heads
        for(int32_t hid = size_min_h; hid < size_h_y; hid++)
        {
            float ogy0[VecPairs], ogy1[VecPairs];
            load_payload_vec<RotateStyle, VecPairs>(ogy0,
                                                    ogy1,
                                                    g_ogy,
                                                    (offset_ogy + hid * stride_oy_h) * elem_bytes,
                                                    did_pair,
                                                    did_start,
                                                    size_half_r);
            float igy0[VecPairs], igy1[VecPairs];
            for(int32_t v = 0; v < VecPairs; v++)
            {
                igy0[v] = ogy0[v] * cos_0[v] + ogy1[v] * sin_0[v];
                igy1[v] = ogy1[v] * cos_1[v] - ogy0[v] * sin_1[v];
            }
            store_payload_vec<RotateStyle, VecPairs>(g_igy,
                                                     igy0,
                                                     igy1,
                                                     (offset_igy + hid * stride_iy_h) * elem_bytes,
                                                     did_pair,
                                                     did_start,
                                                     size_half_r);
        }

        // the rest are just forwarded (nope copy, distributed round-robin)
        if constexpr(!Inplace)
        {
            const int32_t nope_start = NopeFirst ? 0 : size_r;
            const int32_t nope_end   = NopeFirst ? (size_d - size_r) : size_d;
            elementwise_copy_2c<VecPairs>(g_ogx,
                                          g_ogy,
                                          g_igx,
                                          g_igy,
                                          offset_ogx,
                                          offset_ogy,
                                          offset_igx,
                                          offset_igy,
                                          const_cast<scalar_t*>(p_base_input_grads_x),
                                          const_cast<scalar_t*>(p_base_input_grads_y),
                                          p_base_output_grads_x,
                                          p_base_output_grads_y,
                                          size_h_x,
                                          size_h_y,
                                          nope_start,
                                          nope_end,
                                          stride_ox_h,
                                          stride_ox_d,
                                          stride_oy_h,
                                          stride_oy_d,
                                          stride_ix_h,
                                          stride_ix_d,
                                          stride_iy_h,
                                          stride_iy_d,
                                          d_chunk_idx,
                                          threads_per_sb);
        }
#endif
    }
};

// =====================================================================================================================
// Kernel Entries
//

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          bool StrideDOutEq1,
          bool StrideDInEq1,
          int32_t VecPairs = 1,
          typename scalar_t,
          typename scalar_f_t>
__launch_bounds__(256, 8) __global__
    void kn_entry_1c_sbhd_uncached(scalar_t* __restrict__ p_output,
                                   const scalar_t* __restrict__ p_input,
                                   const scalar_f_t* __restrict__ p_freqs,
                                   const int32_t size_h,
                                   const int32_t size_d,
                                   const int32_t size_f, // size of last dimension of freqs.
                                   const int32_t stride_i_s,
                                   const int32_t stride_i_b,
                                   const int32_t stride_i_h,
                                   const int32_t stride_i_d,
                                   const int32_t stride_o_s,
                                   const int32_t stride_o_b,
                                   const int32_t stride_o_h,
                                   const int32_t stride_o_d,
                                   const int32_t size_s,
                                   const int32_t threads_per_sb,
                                   const int32_t total_sb)
{
    const int32_t global_tid  = blockIdx.x * 256 + threadIdx.x;
    const int32_t sb_idx      = global_tid / threads_per_sb;
    const int32_t d_chunk_idx = global_tid % threads_per_sb;
    if(sb_idx >= total_sb)
        return;
    const int32_t sid = sb_idx % size_s;
    const int32_t bid = sb_idx / size_s;

    const int32_t offset_i  = sid * stride_i_s + bid * stride_i_b;
    const int32_t offset_o  = sid * stride_o_s + bid * stride_o_b;
    const uint64_t offset_f = sid * size_f;

    Op::template apply_1c<RotateStyle,
                          ReuseFreqsFrontPart,
                          NopeFirst,
                          false,
                          StrideDOutEq1,
                          StrideDInEq1,
                          VecPairs>(p_output,
                                    p_input,
                                    offset_o,
                                    offset_i,
                                    p_freqs,
                                    (int32_t)offset_f,
                                    size_h,
                                    size_d,
                                    size_f,
                                    stride_i_h,
                                    stride_i_d,
                                    stride_o_h,
                                    stride_o_d,
                                    d_chunk_idx,
                                    threads_per_sb);
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          bool StrideDEq1,
          int32_t VecPairs = 1,
          typename scalar_t,
          typename scalar_f_t>
__launch_bounds__(256, 8) __global__
    void kn_entry_1c_sbhd_uncached_inplace(scalar_t* __restrict__ p_inout,
                                           const scalar_f_t* __restrict__ p_freqs,
                                           const int32_t size_h,
                                           const int32_t size_d,
                                           const int32_t size_f, // size of last dimension of freqs.
                                           const int32_t stride_s,
                                           const int32_t stride_b,
                                           const int32_t stride_h,
                                           const int32_t stride_d,
                                           const int32_t size_s,
                                           const int32_t threads_per_sb,
                                           const int32_t total_sb)
{
    const int32_t global_tid  = blockIdx.x * 256 + threadIdx.x;
    const int32_t sb_idx      = global_tid / threads_per_sb;
    const int32_t d_chunk_idx = global_tid % threads_per_sb;
    if(sb_idx >= total_sb)
        return;
    const int32_t sid = sb_idx % size_s;
    const int32_t bid = sb_idx / size_s;

    const int32_t offset    = sid * stride_s + bid * stride_b;
    const uint64_t offset_f = sid * size_f;

    Op::template apply_1c<RotateStyle,
                          ReuseFreqsFrontPart,
                          NopeFirst,
                          true,
                          StrideDEq1,
                          StrideDEq1,
                          VecPairs>(p_inout,
                                    p_inout,
                                    offset,
                                    offset,
                                    p_freqs,
                                    (int32_t)offset_f,
                                    size_h,
                                    size_d,
                                    size_f,
                                    stride_h,
                                    stride_d,
                                    stride_h,
                                    stride_d,
                                    d_chunk_idx,
                                    threads_per_sb);
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          bool StrideDOutXEq1,
          bool StrideDOutYEq1,
          bool StrideDInXEq1,
          bool StrideDInYEq1,
          int32_t VecPairs = 1,
          typename scalar_t,
          typename scalar_f_t>
__launch_bounds__(256, 8) __global__
    void kn_entry_2c_sbhd_uncached(scalar_t* __restrict__ p_output_x,
                                   scalar_t* __restrict__ p_output_y,
                                   const scalar_t* __restrict__ p_input_x,
                                   const scalar_t* __restrict__ p_input_y,
                                   const scalar_f_t* __restrict__ p_freqs,
                                   const int32_t size_h_x,
                                   const int32_t size_h_y,
                                   const int32_t size_d,
                                   const int32_t size_f, // size of last dimension of freqs.
                                   const int32_t stride_ix_s,
                                   const int32_t stride_ix_b,
                                   const int32_t stride_ix_h,
                                   const int32_t stride_ix_d,
                                   const int32_t stride_iy_s,
                                   const int32_t stride_iy_b,
                                   const int32_t stride_iy_h,
                                   const int32_t stride_iy_d,
                                   const int32_t stride_ox_s,
                                   const int32_t stride_ox_b,
                                   const int32_t stride_ox_h,
                                   const int32_t stride_ox_d,
                                   const int32_t stride_oy_s,
                                   const int32_t stride_oy_b,
                                   const int32_t stride_oy_h,
                                   const int32_t stride_oy_d,
                                   const int32_t size_s,
                                   const int32_t threads_per_sb,
                                   const int32_t total_sb)
{
    const int32_t global_tid  = blockIdx.x * 256 + threadIdx.x;
    const int32_t sb_idx      = global_tid / threads_per_sb;
    const int32_t d_chunk_idx = global_tid % threads_per_sb;
    if(sb_idx >= total_sb)
        return;
    const int32_t sid = sb_idx % size_s;
    const int32_t bid = sb_idx / size_s;

    const int32_t offset_ix = sid * stride_ix_s + bid * stride_ix_b;
    const int32_t offset_iy = sid * stride_iy_s + bid * stride_iy_b;
    const int32_t offset_ox = sid * stride_ox_s + bid * stride_ox_b;
    const int32_t offset_oy = sid * stride_oy_s + bid * stride_oy_b;
    const uint64_t offset_f = sid * size_f;

    Op::template apply_2c<RotateStyle,
                          ReuseFreqsFrontPart,
                          NopeFirst,
                          false,
                          StrideDOutXEq1,
                          StrideDOutYEq1,
                          StrideDInXEq1,
                          StrideDInYEq1,
                          VecPairs>(p_output_x,
                                    p_output_y,
                                    p_input_x,
                                    p_input_y,
                                    offset_ox,
                                    offset_oy,
                                    offset_ix,
                                    offset_iy,
                                    p_freqs,
                                    (int32_t)offset_f,
                                    size_h_x,
                                    size_h_y,
                                    size_d,
                                    size_f,
                                    stride_ix_h,
                                    stride_ix_d,
                                    stride_iy_h,
                                    stride_iy_d,
                                    stride_ox_h,
                                    stride_ox_d,
                                    stride_oy_h,
                                    stride_oy_d,
                                    d_chunk_idx,
                                    threads_per_sb);
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          bool StrideDXEq1,
          bool StrideDYEq1,
          int32_t VecPairs = 1,
          typename scalar_t,
          typename scalar_f_t>
__launch_bounds__(256, 8) __global__
    void kn_entry_2c_sbhd_uncached_inplace(scalar_t* __restrict__ p_inout_x,
                                           scalar_t* __restrict__ p_inout_y,
                                           const scalar_f_t* __restrict__ p_freqs,
                                           const int32_t size_h_x,
                                           const int32_t size_h_y,
                                           const int32_t size_d,
                                           const int32_t size_f, // size of last dimension of freqs.
                                           const int32_t stride_x_s,
                                           const int32_t stride_x_b,
                                           const int32_t stride_x_h,
                                           const int32_t stride_x_d,
                                           const int32_t stride_y_s,
                                           const int32_t stride_y_b,
                                           const int32_t stride_y_h,
                                           const int32_t stride_y_d,
                                           const int32_t size_s,
                                           const int32_t threads_per_sb,
                                           const int32_t total_sb)
{
    const int32_t global_tid  = blockIdx.x * 256 + threadIdx.x;
    const int32_t sb_idx      = global_tid / threads_per_sb;
    const int32_t d_chunk_idx = global_tid % threads_per_sb;
    if(sb_idx >= total_sb)
        return;
    const int32_t sid = sb_idx % size_s;
    const int32_t bid = sb_idx / size_s;

    const int32_t offset_x  = sid * stride_x_s + bid * stride_x_b;
    const int32_t offset_y  = sid * stride_y_s + bid * stride_y_b;
    const uint64_t offset_f = sid * size_f;

    Op::template apply_2c<RotateStyle,
                          ReuseFreqsFrontPart,
                          NopeFirst,
                          true,
                          StrideDXEq1,
                          StrideDYEq1,
                          StrideDXEq1,
                          StrideDYEq1,
                          VecPairs>(p_inout_x,
                                    p_inout_y,
                                    p_inout_x,
                                    p_inout_y,
                                    offset_x,
                                    offset_y,
                                    offset_x,
                                    offset_y,
                                    p_freqs,
                                    (int32_t)offset_f,
                                    size_h_x,
                                    size_h_y,
                                    size_d,
                                    size_f,
                                    stride_x_h,
                                    stride_x_d,
                                    stride_y_h,
                                    stride_y_d,
                                    stride_x_h,
                                    stride_x_d,
                                    stride_y_h,
                                    stride_y_d,
                                    d_chunk_idx,
                                    threads_per_sb);
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          bool StrideDOutEq1,
          bool StrideDInEq1,
          int32_t VecPairs = 1,
          typename scalar_t,
          typename scalar_f_t>
__launch_bounds__(256, 8) __global__
    void kn_entry_1c_sbhd_cached(scalar_t* __restrict__ p_output,
                                 const scalar_t* __restrict__ p_input,
                                 const scalar_f_t* __restrict__ p_cos,
                                 const scalar_f_t* __restrict__ p_sin,
                                 const int32_t size_h,
                                 const int32_t size_d,
                                 const int32_t size_f, // size of last dimension of freqs.
                                 const int32_t stride_i_s,
                                 const int32_t stride_i_b,
                                 const int32_t stride_i_h,
                                 const int32_t stride_i_d,
                                 const int32_t stride_o_s,
                                 const int32_t stride_o_b,
                                 const int32_t stride_o_h,
                                 const int32_t stride_o_d,
                                 const int32_t size_s,
                                 const int32_t threads_per_sb,
                                 const int32_t total_sb)
{
    const int32_t global_tid  = blockIdx.x * 256 + threadIdx.x;
    const int32_t sb_idx      = global_tid / threads_per_sb;
    const int32_t d_chunk_idx = global_tid % threads_per_sb;
    if(sb_idx >= total_sb)
        return;
    const int32_t sid = sb_idx % size_s;
    const int32_t bid = sb_idx / size_s;

    const int32_t offset_i  = sid * stride_i_s + bid * stride_i_b;
    const int32_t offset_o  = sid * stride_o_s + bid * stride_o_b;
    const uint64_t offset_f = sid * size_f;

    Op::template apply_1c<RotateStyle,
                          ReuseFreqsFrontPart,
                          NopeFirst,
                          false,
                          StrideDOutEq1,
                          StrideDInEq1,
                          VecPairs>(p_output,
                                    p_input,
                                    (int32_t)offset_o,
                                    (int32_t)offset_i,
                                    p_cos,
                                    p_sin,
                                    (int32_t)offset_f,
                                    size_h,
                                    size_d,
                                    size_f,
                                    stride_i_h,
                                    stride_i_d,
                                    stride_o_h,
                                    stride_o_d,
                                    d_chunk_idx,
                                    threads_per_sb);
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          bool StrideDEq1,
          int32_t VecPairs = 1,
          typename scalar_t,
          typename scalar_f_t>
__launch_bounds__(256, 8) __global__
    void kn_entry_1c_sbhd_cached_inplace(scalar_t* __restrict__ p_inout,
                                         const scalar_f_t* __restrict__ p_cos,
                                         const scalar_f_t* __restrict__ p_sin,
                                         const int32_t size_h,
                                         const int32_t size_d,
                                         const int32_t size_f, // size of last dimension of freqs.
                                         const int32_t stride_s,
                                         const int32_t stride_b,
                                         const int32_t stride_h,
                                         const int32_t stride_d,
                                         const int32_t size_s,
                                         const int32_t threads_per_sb,
                                         const int32_t total_sb)
{
    const int32_t global_tid  = blockIdx.x * 256 + threadIdx.x;
    const int32_t sb_idx      = global_tid / threads_per_sb;
    const int32_t d_chunk_idx = global_tid % threads_per_sb;
    if(sb_idx >= total_sb)
        return;
    const int32_t sid = sb_idx % size_s;
    const int32_t bid = sb_idx / size_s;

    const int32_t offset    = sid * stride_s + bid * stride_b;
    const uint64_t offset_f = sid * size_f;

    Op::template apply_1c<RotateStyle,
                          ReuseFreqsFrontPart,
                          NopeFirst,
                          true,
                          StrideDEq1,
                          StrideDEq1,
                          VecPairs>(p_inout,
                                    p_inout,
                                    (int32_t)offset,
                                    (int32_t)offset,
                                    p_cos,
                                    p_sin,
                                    (int32_t)offset_f,
                                    size_h,
                                    size_d,
                                    size_f,
                                    stride_h,
                                    stride_d,
                                    stride_h,
                                    stride_d,
                                    d_chunk_idx,
                                    threads_per_sb);
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          bool StrideDOutXEq1,
          bool StrideDOutYEq1,
          bool StrideDInXEq1,
          bool StrideDInYEq1,
          int32_t VecPairs = 1,
          typename scalar_t,
          typename scalar_f_t>
__launch_bounds__(256, 8) __global__
    void kn_entry_2c_sbhd_cached(scalar_t* __restrict__ p_output_x,
                                 scalar_t* __restrict__ p_output_y,
                                 const scalar_t* __restrict__ p_input_x,
                                 const scalar_t* __restrict__ p_input_y,
                                 const scalar_f_t* __restrict__ p_cos,
                                 const scalar_f_t* __restrict__ p_sin,
                                 const int32_t size_h_x,
                                 const int32_t size_h_y,
                                 const int32_t size_d,
                                 const int32_t size_f, // size of last dimension of freqs.
                                 const int32_t stride_ix_s,
                                 const int32_t stride_ix_b,
                                 const int32_t stride_ix_h,
                                 const int32_t stride_ix_d,
                                 const int32_t stride_iy_s,
                                 const int32_t stride_iy_b,
                                 const int32_t stride_iy_h,
                                 const int32_t stride_iy_d,
                                 const int32_t stride_ox_s,
                                 const int32_t stride_ox_b,
                                 const int32_t stride_ox_h,
                                 const int32_t stride_ox_d,
                                 const int32_t stride_oy_s,
                                 const int32_t stride_oy_b,
                                 const int32_t stride_oy_h,
                                 const int32_t stride_oy_d,
                                 const int32_t size_s,
                                 const int32_t threads_per_sb,
                                 const int32_t total_sb)
{
    const int32_t global_tid  = blockIdx.x * 256 + threadIdx.x;
    const int32_t sb_idx      = global_tid / threads_per_sb;
    const int32_t d_chunk_idx = global_tid % threads_per_sb;
    if(sb_idx >= total_sb)
        return;
    const int32_t sid = sb_idx % size_s;
    const int32_t bid = sb_idx / size_s;

    const int32_t offset_ix = sid * stride_ix_s + bid * stride_ix_b;
    const int32_t offset_iy = sid * stride_iy_s + bid * stride_iy_b;
    const int32_t offset_ox = sid * stride_ox_s + bid * stride_ox_b;
    const int32_t offset_oy = sid * stride_oy_s + bid * stride_oy_b;
    const uint64_t offset_f = sid * size_f;

    Op::template apply_2c<RotateStyle,
                          ReuseFreqsFrontPart,
                          NopeFirst,
                          false,
                          StrideDOutXEq1,
                          StrideDOutYEq1,
                          StrideDInXEq1,
                          StrideDInYEq1,
                          VecPairs>(p_output_x,
                                    p_output_y,
                                    p_input_x,
                                    p_input_y,
                                    (int32_t)offset_ox,
                                    (int32_t)offset_oy,
                                    (int32_t)offset_ix,
                                    (int32_t)offset_iy,
                                    p_cos,
                                    p_sin,
                                    (int32_t)offset_f,
                                    size_h_x,
                                    size_h_y,
                                    size_d,
                                    size_f,
                                    stride_ix_h,
                                    stride_ix_d,
                                    stride_iy_h,
                                    stride_iy_d,
                                    stride_ox_h,
                                    stride_ox_d,
                                    stride_oy_h,
                                    stride_oy_d,
                                    d_chunk_idx,
                                    threads_per_sb);
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          bool StrideDXEq1,
          bool StrideDYEq1,
          int32_t VecPairs = 1,
          typename scalar_t,
          typename scalar_f_t>
__launch_bounds__(256, 8) __global__
    void kn_entry_2c_sbhd_cached_inplace(scalar_t* __restrict__ p_inout_x,
                                         scalar_t* __restrict__ p_inout_y,
                                         const scalar_f_t* __restrict__ p_cos,
                                         const scalar_f_t* __restrict__ p_sin,
                                         const int32_t size_h_x,
                                         const int32_t size_h_y,
                                         const int32_t size_d,
                                         const int32_t size_f, // size of last dimension of freqs.
                                         const int32_t stride_x_s,
                                         const int32_t stride_x_b,
                                         const int32_t stride_x_h,
                                         const int32_t stride_x_d,
                                         const int32_t stride_y_s,
                                         const int32_t stride_y_b,
                                         const int32_t stride_y_h,
                                         const int32_t stride_y_d,
                                         const int32_t size_s,
                                         const int32_t threads_per_sb,
                                         const int32_t total_sb)
{
    const int32_t global_tid  = blockIdx.x * 256 + threadIdx.x;
    const int32_t sb_idx      = global_tid / threads_per_sb;
    const int32_t d_chunk_idx = global_tid % threads_per_sb;
    if(sb_idx >= total_sb)
        return;
    const int32_t sid = sb_idx % size_s;
    const int32_t bid = sb_idx / size_s;

    const int32_t offset_x  = sid * stride_x_s + bid * stride_x_b;
    const int32_t offset_y  = sid * stride_y_s + bid * stride_y_b;
    const uint64_t offset_f = sid * size_f;

    Op::template apply_2c<RotateStyle,
                          ReuseFreqsFrontPart,
                          NopeFirst,
                          true,
                          StrideDXEq1,
                          StrideDYEq1,
                          StrideDXEq1,
                          StrideDYEq1,
                          VecPairs>(p_inout_x,
                                    p_inout_y,
                                    p_inout_x,
                                    p_inout_y,
                                    (int32_t)offset_x,
                                    (int32_t)offset_y,
                                    (int32_t)offset_x,
                                    (int32_t)offset_y,
                                    p_cos,
                                    p_sin,
                                    (int32_t)offset_f,
                                    size_h_x,
                                    size_h_y,
                                    size_d,
                                    size_f,
                                    stride_x_h,
                                    stride_x_d,
                                    stride_y_h,
                                    stride_y_d,
                                    stride_x_h,
                                    stride_x_d,
                                    stride_y_h,
                                    stride_y_d,
                                    d_chunk_idx,
                                    threads_per_sb);
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          bool StrideDOutEq1,
          bool StrideDInEq1,
          int32_t VecPairs = 1,
          typename scalar_t,
          typename scalar_f_t>
__launch_bounds__(256, 8) __global__
    void kn_entry_1c_sbhd_cached_indirect(scalar_t* __restrict__ p_output,
                                          const scalar_t* __restrict__ p_input,
                                          const scalar_f_t* __restrict__ p_cos,
                                          const scalar_f_t* __restrict__ p_sin,
                                          const int64_t* __restrict__ p_indirect_buffer,
                                          const int32_t max_position,
                                          const int32_t size_h,
                                          const int32_t size_d,
                                          const int32_t size_f, // size of last dimension of freqs.
                                          const int32_t stride_i_s,
                                          const int32_t stride_i_b,
                                          const int32_t stride_i_h,
                                          const int32_t stride_i_d,
                                          const int32_t stride_o_s,
                                          const int32_t stride_o_b,
                                          const int32_t stride_o_h,
                                          const int32_t stride_o_d,
                                          const int32_t size_s,
                                          const int32_t threads_per_sb,
                                          const int32_t total_sb)
{
    const int32_t global_tid  = blockIdx.x * 256 + threadIdx.x;
    const int32_t sb_idx      = global_tid / threads_per_sb;
    const int32_t d_chunk_idx = global_tid % threads_per_sb;
    if(sb_idx >= total_sb)
        return;
    const int32_t sid = sb_idx % size_s;
    const int32_t bid = sb_idx / size_s;

    const uint64_t ib_idx = sid * (total_sb / size_s) + bid;
    const int64_t pos     = p_indirect_buffer[ib_idx];

    if((pos >= 0) && (pos < max_position))
    {
        const int32_t offset_i = sid * stride_i_s + bid * stride_i_b;
        const int32_t offset_o = sid * stride_o_s + bid * stride_o_b;
        const int64_t offset_f = pos * size_f;

        Op::template apply_1c<RotateStyle,
                              ReuseFreqsFrontPart,
                              NopeFirst,
                              false,
                              StrideDOutEq1,
                              StrideDInEq1,
                              VecPairs>(p_output,
                                        p_input,
                                        (int32_t)offset_o,
                                        (int32_t)offset_i,
                                        p_cos,
                                        p_sin,
                                        (int32_t)offset_f,
                                        size_h,
                                        size_d,
                                        size_f,
                                        stride_i_h,
                                        stride_i_d,
                                        stride_o_h,
                                        stride_o_d,
                                        d_chunk_idx,
                                        threads_per_sb);
    }
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          bool StrideDOutXEq1,
          bool StrideDOutYEq1,
          bool StrideDInXEq1,
          bool StrideDInYEq1,
          int32_t VecPairs = 1,
          typename scalar_t,
          typename scalar_f_t>
__launch_bounds__(256, 8) __global__
    void kn_entry_2c_sbhd_cached_indirect(scalar_t* __restrict__ p_output_x,
                                          scalar_t* __restrict__ p_output_y,
                                          const scalar_t* __restrict__ p_input_x,
                                          const scalar_t* __restrict__ p_input_y,
                                          const scalar_f_t* __restrict__ p_cos,
                                          const scalar_f_t* __restrict__ p_sin,
                                          const int64_t* __restrict__ p_indirect_buffer,
                                          const int32_t max_position,
                                          const int32_t size_h_x,
                                          const int32_t size_h_y,
                                          const int32_t size_d,
                                          const int32_t size_f, // size of last dimension of freqs.
                                          const int32_t stride_ix_s,
                                          const int32_t stride_ix_b,
                                          const int32_t stride_ix_h,
                                          const int32_t stride_ix_d,
                                          const int32_t stride_iy_s,
                                          const int32_t stride_iy_b,
                                          const int32_t stride_iy_h,
                                          const int32_t stride_iy_d,
                                          const int32_t stride_ox_s,
                                          const int32_t stride_ox_b,
                                          const int32_t stride_ox_h,
                                          const int32_t stride_ox_d,
                                          const int32_t stride_oy_s,
                                          const int32_t stride_oy_b,
                                          const int32_t stride_oy_h,
                                          const int32_t stride_oy_d,
                                          const int32_t size_s,
                                          const int32_t threads_per_sb,
                                          const int32_t total_sb)
{
    const int32_t global_tid  = blockIdx.x * 256 + threadIdx.x;
    const int32_t sb_idx      = global_tid / threads_per_sb;
    const int32_t d_chunk_idx = global_tid % threads_per_sb;
    if(sb_idx >= total_sb)
        return;
    const int32_t sid = sb_idx % size_s;
    const int32_t bid = sb_idx / size_s;

    const uint64_t ib_idx = sid * (total_sb / size_s) + bid;
    const int64_t pos     = p_indirect_buffer[ib_idx];

    if((pos >= 0) && (pos < max_position))
    {
        const int32_t offset_ix = sid * stride_ix_s + bid * stride_ix_b;
        const int32_t offset_iy = sid * stride_iy_s + bid * stride_iy_b;
        const int32_t offset_ox = sid * stride_ox_s + bid * stride_ox_b;
        const int32_t offset_oy = sid * stride_oy_s + bid * stride_oy_b;
        const int64_t offset_f  = pos * size_f;

        Op::template apply_2c<RotateStyle,
                              ReuseFreqsFrontPart,
                              NopeFirst,
                              false,
                              StrideDOutXEq1,
                              StrideDOutYEq1,
                              StrideDInXEq1,
                              StrideDInYEq1,
                              VecPairs>(p_output_x,
                                        p_output_y,
                                        p_input_x,
                                        p_input_y,
                                        (int32_t)offset_ox,
                                        (int32_t)offset_oy,
                                        (int32_t)offset_ix,
                                        (int32_t)offset_iy,
                                        p_cos,
                                        p_sin,
                                        (int32_t)offset_f,
                                        size_h_x,
                                        size_h_y,
                                        size_d,
                                        size_f,
                                        stride_ix_h,
                                        stride_ix_d,
                                        stride_iy_h,
                                        stride_iy_d,
                                        stride_ox_h,
                                        stride_ox_d,
                                        stride_oy_h,
                                        stride_oy_d,
                                        d_chunk_idx,
                                        threads_per_sb);
    }
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          bool StrideDEq1,
          int32_t VecPairs = 1,
          typename scalar_t,
          typename scalar_f_t>
__launch_bounds__(256, 8) __global__ void kn_entry_1c_sbhd_cached_indirect_inplace(
    scalar_t* __restrict__ p_inout,
    const scalar_f_t* __restrict__ p_cos,
    const scalar_f_t* __restrict__ p_sin,
    const int64_t* __restrict__ p_indirect_buffer,
    const int32_t max_position,
    const int32_t size_h,
    const int32_t size_d,
    const int32_t size_f, // size of last dimension of freqs.
    const int32_t stride_s,
    const int32_t stride_b,
    const int32_t stride_h,
    const int32_t stride_d,
    const int32_t size_s,
    const int32_t threads_per_sb,
    const int32_t total_sb)
{
    const int32_t global_tid  = blockIdx.x * 256 + threadIdx.x;
    const int32_t sb_idx      = global_tid / threads_per_sb;
    const int32_t d_chunk_idx = global_tid % threads_per_sb;
    if(sb_idx >= total_sb)
        return;
    const int32_t sid = sb_idx % size_s;
    const int32_t bid = sb_idx / size_s;

    const uint64_t ib_idx = sid * (total_sb / size_s) + bid;
    const int64_t pos     = p_indirect_buffer[ib_idx];

    if((pos >= 0) && (pos < max_position))
    {
        const int32_t offset   = sid * stride_s + bid * stride_b;
        const int64_t offset_f = pos * size_f;

        Op::template apply_1c<RotateStyle,
                              ReuseFreqsFrontPart,
                              NopeFirst,
                              true,
                              StrideDEq1,
                              StrideDEq1,
                              VecPairs>(p_inout,
                                        p_inout,
                                        (int32_t)offset,
                                        (int32_t)offset,
                                        p_cos,
                                        p_sin,
                                        (int32_t)offset_f,
                                        size_h,
                                        size_d,
                                        size_f,
                                        stride_h,
                                        stride_d,
                                        stride_h,
                                        stride_d,
                                        d_chunk_idx,
                                        threads_per_sb);
    }
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          bool StrideDXEq1,
          bool StrideDYEq1,
          int32_t VecPairs = 1,
          typename scalar_t,
          typename scalar_f_t>
__launch_bounds__(256, 8) __global__ void kn_entry_2c_sbhd_cached_indirect_inplace(
    scalar_t* __restrict__ p_inout_x,
    scalar_t* __restrict__ p_inout_y,
    const scalar_f_t* __restrict__ p_cos,
    const scalar_f_t* __restrict__ p_sin,
    const int64_t* __restrict__ p_indirect_buffer,
    const int32_t max_position,
    const int32_t size_h_x,
    const int32_t size_h_y,
    const int32_t size_d,
    const int32_t size_f, // size of last dimension of freqs.
    const int32_t stride_x_s,
    const int32_t stride_x_b,
    const int32_t stride_x_h,
    const int32_t stride_x_d,
    const int32_t stride_y_s,
    const int32_t stride_y_b,
    const int32_t stride_y_h,
    const int32_t stride_y_d,
    const int32_t size_s,
    const int32_t threads_per_sb,
    const int32_t total_sb)
{
    const int32_t global_tid  = blockIdx.x * 256 + threadIdx.x;
    const int32_t sb_idx      = global_tid / threads_per_sb;
    const int32_t d_chunk_idx = global_tid % threads_per_sb;
    if(sb_idx >= total_sb)
        return;
    const int32_t sid = sb_idx % size_s;
    const int32_t bid = sb_idx / size_s;

    const uint64_t ib_idx = sid * (total_sb / size_s) + bid;
    const int64_t pos     = p_indirect_buffer[ib_idx];

    if((pos >= 0) && (pos < max_position))
    {
        const int32_t offset_x = sid * stride_x_s + bid * stride_x_b;
        const int32_t offset_y = sid * stride_y_s + bid * stride_y_b;
        const int64_t offset_f = pos * size_f;

        Op::template apply_2c<RotateStyle,
                              ReuseFreqsFrontPart,
                              NopeFirst,
                              true,
                              StrideDXEq1,
                              StrideDYEq1,
                              StrideDXEq1,
                              StrideDYEq1,
                              VecPairs>(p_inout_x,
                                        p_inout_y,
                                        p_inout_x,
                                        p_inout_y,
                                        (int32_t)offset_x,
                                        (int32_t)offset_y,
                                        (int32_t)offset_x,
                                        (int32_t)offset_y,
                                        p_cos,
                                        p_sin,
                                        (int32_t)offset_f,
                                        size_h_x,
                                        size_h_y,
                                        size_d,
                                        size_f,
                                        stride_x_h,
                                        stride_x_d,
                                        stride_y_h,
                                        stride_y_d,
                                        stride_x_h,
                                        stride_x_d,
                                        stride_y_h,
                                        stride_y_d,
                                        d_chunk_idx,
                                        threads_per_sb);
    }
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          bool StrideDOutEq1,
          bool StrideDInEq1,
          int32_t VecPairs = 1,
          typename scalar_t,
          typename scalar_f_t>
__launch_bounds__(256, 8) __global__
    void kn_entry_1c_sbhd_cached_indirect2(scalar_t* __restrict__ p_output,
                                           const scalar_t* __restrict__ p_input,
                                           const scalar_f_t* __restrict__ p_cos,
                                           const scalar_f_t* __restrict__ p_sin,
                                           const int64_t* __restrict__ p_indirect_buffer_0,
                                           const int64_t* __restrict__ p_indirect_buffer_1,
                                           const int32_t max_position,
                                           const int32_t size_h,
                                           const int32_t size_d,
                                           const int32_t size_f, // size of last dimension of freqs.
                                           const int32_t stride_i_s,
                                           const int32_t stride_i_b,
                                           const int32_t stride_i_h,
                                           const int32_t stride_i_d,
                                           const int32_t stride_o_s,
                                           const int32_t stride_o_b,
                                           const int32_t stride_o_h,
                                           const int32_t stride_o_d,
                                           const int32_t size_s,
                                           const int32_t threads_per_sb,
                                           const int32_t total_sb)
{
    const int32_t global_tid  = blockIdx.x * 256 + threadIdx.x;
    const int32_t sb_idx      = global_tid / threads_per_sb;
    const int32_t d_chunk_idx = global_tid % threads_per_sb;
    if(sb_idx >= total_sb)
        return;
    const int32_t sid = sb_idx % size_s;
    const int32_t bid = sb_idx / size_s;

    const uint64_t ib_idx = sid * (total_sb / size_s) + bid;
    const int64_t pos     = p_indirect_buffer_0[ib_idx] + p_indirect_buffer_1[ib_idx];

    if((pos >= 0) && (pos < max_position))
    {
        const int32_t offset_i = sid * stride_i_s + bid * stride_i_b;
        const int32_t offset_o = sid * stride_o_s + bid * stride_o_b;
        const int64_t offset_f = pos * size_f;

        Op::template apply_1c<RotateStyle,
                              ReuseFreqsFrontPart,
                              NopeFirst,
                              false,
                              StrideDOutEq1,
                              StrideDInEq1,
                              VecPairs>(p_output,
                                        p_input,
                                        (int32_t)offset_o,
                                        (int32_t)offset_i,
                                        p_cos,
                                        p_sin,
                                        (int32_t)offset_f,
                                        size_h,
                                        size_d,
                                        size_f,
                                        stride_i_h,
                                        stride_i_d,
                                        stride_o_h,
                                        stride_o_d,
                                        d_chunk_idx,
                                        threads_per_sb);
    }
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          bool StrideDOutXEq1,
          bool StrideDOutYEq1,
          bool StrideDInXEq1,
          bool StrideDInYEq1,
          int32_t VecPairs = 1,
          typename scalar_t,
          typename scalar_f_t>
__launch_bounds__(256, 8) __global__
    void kn_entry_2c_sbhd_cached_indirect2(scalar_t* __restrict__ p_output_x,
                                           scalar_t* __restrict__ p_output_y,
                                           const scalar_t* __restrict__ p_input_x,
                                           const scalar_t* __restrict__ p_input_y,
                                           const scalar_f_t* __restrict__ p_cos,
                                           const scalar_f_t* __restrict__ p_sin,
                                           const int64_t* __restrict__ p_indirect_buffer_0,
                                           const int64_t* __restrict__ p_indirect_buffer_1,
                                           const int32_t max_position,
                                           const int32_t size_h_x,
                                           const int32_t size_h_y,
                                           const int32_t size_d,
                                           const int32_t size_f, // size of last dimension of freqs.
                                           const int32_t stride_ix_s,
                                           const int32_t stride_ix_b,
                                           const int32_t stride_ix_h,
                                           const int32_t stride_ix_d,
                                           const int32_t stride_iy_s,
                                           const int32_t stride_iy_b,
                                           const int32_t stride_iy_h,
                                           const int32_t stride_iy_d,
                                           const int32_t stride_ox_s,
                                           const int32_t stride_ox_b,
                                           const int32_t stride_ox_h,
                                           const int32_t stride_ox_d,
                                           const int32_t stride_oy_s,
                                           const int32_t stride_oy_b,
                                           const int32_t stride_oy_h,
                                           const int32_t stride_oy_d,
                                           const int32_t size_s,
                                           const int32_t threads_per_sb,
                                           const int32_t total_sb)
{
    const int32_t global_tid  = blockIdx.x * 256 + threadIdx.x;
    const int32_t sb_idx      = global_tid / threads_per_sb;
    const int32_t d_chunk_idx = global_tid % threads_per_sb;
    if(sb_idx >= total_sb)
        return;
    const int32_t sid = sb_idx % size_s;
    const int32_t bid = sb_idx / size_s;

    const uint64_t ib_idx = sid * (total_sb / size_s) + bid;
    const int64_t pos     = p_indirect_buffer_0[ib_idx] + p_indirect_buffer_1[ib_idx];

    if((pos >= 0) && (pos < max_position))
    {
        const int32_t offset_ix = sid * stride_ix_s + bid * stride_ix_b;
        const int32_t offset_iy = sid * stride_iy_s + bid * stride_iy_b;
        const int32_t offset_ox = sid * stride_ox_s + bid * stride_ox_b;
        const int32_t offset_oy = sid * stride_oy_s + bid * stride_oy_b;
        const int64_t offset_f  = pos * size_f;

        Op::template apply_2c<RotateStyle,
                              ReuseFreqsFrontPart,
                              NopeFirst,
                              false,
                              StrideDOutXEq1,
                              StrideDOutYEq1,
                              StrideDInXEq1,
                              StrideDInYEq1,
                              VecPairs>(p_output_x,
                                        p_output_y,
                                        p_input_x,
                                        p_input_y,
                                        (int32_t)offset_ox,
                                        (int32_t)offset_oy,
                                        (int32_t)offset_ix,
                                        (int32_t)offset_iy,
                                        p_cos,
                                        p_sin,
                                        (int32_t)offset_f,
                                        size_h_x,
                                        size_h_y,
                                        size_d,
                                        size_f,
                                        stride_ix_h,
                                        stride_ix_d,
                                        stride_iy_h,
                                        stride_iy_d,
                                        stride_ox_h,
                                        stride_ox_d,
                                        stride_oy_h,
                                        stride_oy_d,
                                        d_chunk_idx,
                                        threads_per_sb);
    }
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          bool StrideDEq1,
          int32_t VecPairs = 1,
          typename scalar_t,
          typename scalar_f_t>
__launch_bounds__(256, 8) __global__ void kn_entry_1c_sbhd_cached_indirect2_inplace(
    scalar_t* __restrict__ p_inout,
    const scalar_f_t* __restrict__ p_cos,
    const scalar_f_t* __restrict__ p_sin,
    const int64_t* __restrict__ p_indirect_buffer_0,
    const int64_t* __restrict__ p_indirect_buffer_1,
    const int32_t max_position,
    const int32_t size_h,
    const int32_t size_d,
    const int32_t size_f, // size of last dimension of freqs.
    const int32_t stride_s,
    const int32_t stride_b,
    const int32_t stride_h,
    const int32_t stride_d,
    const int32_t size_s,
    const int32_t threads_per_sb,
    const int32_t total_sb)
{
    const int32_t global_tid  = blockIdx.x * 256 + threadIdx.x;
    const int32_t sb_idx      = global_tid / threads_per_sb;
    const int32_t d_chunk_idx = global_tid % threads_per_sb;
    if(sb_idx >= total_sb)
        return;
    const int32_t sid = sb_idx % size_s;
    const int32_t bid = sb_idx / size_s;

    const uint64_t ib_idx = sid * (total_sb / size_s) + bid;
    const int64_t pos     = p_indirect_buffer_0[ib_idx] + p_indirect_buffer_1[ib_idx];

    if((pos >= 0) && (pos < max_position))
    {
        const int32_t offset   = sid * stride_s + bid * stride_b;
        const int64_t offset_f = pos * size_f;

        Op::template apply_1c<RotateStyle,
                              ReuseFreqsFrontPart,
                              NopeFirst,
                              true,
                              StrideDEq1,
                              StrideDEq1,
                              VecPairs>(p_inout,
                                        p_inout,
                                        (int32_t)offset,
                                        (int32_t)offset,
                                        p_cos,
                                        p_sin,
                                        (int32_t)offset_f,
                                        size_h,
                                        size_d,
                                        size_f,
                                        stride_h,
                                        stride_d,
                                        stride_h,
                                        stride_d,
                                        d_chunk_idx,
                                        threads_per_sb);
    }
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          bool StrideDXEq1,
          bool StrideDYEq1,
          int32_t VecPairs = 1,
          typename scalar_t,
          typename scalar_f_t>
__launch_bounds__(256, 8) __global__ void kn_entry_2c_sbhd_cached_indirect2_inplace(
    scalar_t* __restrict__ p_inout_x,
    scalar_t* __restrict__ p_inout_y,
    const scalar_f_t* __restrict__ p_cos,
    const scalar_f_t* __restrict__ p_sin,
    const int64_t* __restrict__ p_indirect_buffer_0,
    const int64_t* __restrict__ p_indirect_buffer_1,
    const int32_t max_position,
    const int32_t size_h_x,
    const int32_t size_h_y,
    const int32_t size_d,
    const int32_t size_f, // size of last dimension of freqs.
    const int32_t stride_x_s,
    const int32_t stride_x_b,
    const int32_t stride_x_h,
    const int32_t stride_x_d,
    const int32_t stride_y_s,
    const int32_t stride_y_b,
    const int32_t stride_y_h,
    const int32_t stride_y_d,
    const int32_t size_s,
    const int32_t threads_per_sb,
    const int32_t total_sb)
{
    const int32_t global_tid  = blockIdx.x * 256 + threadIdx.x;
    const int32_t sb_idx      = global_tid / threads_per_sb;
    const int32_t d_chunk_idx = global_tid % threads_per_sb;
    if(sb_idx >= total_sb)
        return;
    const int32_t sid = sb_idx % size_s;
    const int32_t bid = sb_idx / size_s;

    const uint64_t ib_idx = sid * (total_sb / size_s) + bid;
    const int64_t pos     = p_indirect_buffer_0[ib_idx] + p_indirect_buffer_1[ib_idx];

    if((pos >= 0) && (pos < max_position))
    {
        const int32_t offset_x = sid * stride_x_s + bid * stride_x_b;
        const int32_t offset_y = sid * stride_y_s + bid * stride_y_b;
        const int64_t offset_f = pos * size_f;

        Op::template apply_2c<RotateStyle,
                              ReuseFreqsFrontPart,
                              NopeFirst,
                              true,
                              StrideDXEq1,
                              StrideDYEq1,
                              StrideDXEq1,
                              StrideDYEq1,
                              VecPairs>(p_inout_x,
                                        p_inout_y,
                                        p_inout_x,
                                        p_inout_y,
                                        (int32_t)offset_x,
                                        (int32_t)offset_y,
                                        (int32_t)offset_x,
                                        (int32_t)offset_y,
                                        p_cos,
                                        p_sin,
                                        (int32_t)offset_f,
                                        size_h_x,
                                        size_h_y,
                                        size_d,
                                        size_f,
                                        stride_x_h,
                                        stride_x_d,
                                        stride_y_h,
                                        stride_y_d,
                                        stride_x_h,
                                        stride_x_d,
                                        stride_y_h,
                                        stride_y_d,
                                        d_chunk_idx,
                                        threads_per_sb);
    }
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          bool StrideDOutEq1,
          bool StrideDInEq1,
          int32_t VecPairs = 1,
          typename scalar_t,
          typename scalar_f_t>
__launch_bounds__(256, 8) __global__
    void kn_entry_1c_thd_uncached(scalar_t* __restrict__ p_output,
                                  const scalar_t* __restrict__ p_input,
                                  const int32_t* __restrict__ p_cu_seqlens,
                                  const scalar_f_t* __restrict__ p_freqs,
                                  const int32_t size_h,
                                  const int32_t size_d,
                                  const int32_t size_f, // size of last dimension of freqs.
                                  const int32_t stride_i_t,
                                  const int32_t stride_i_h,
                                  const int32_t stride_i_d,
                                  const int32_t stride_o_t,
                                  const int32_t stride_o_h,
                                  const int32_t stride_o_d,
                                  const int32_t size_s,
                                  const int32_t threads_per_sb,
                                  const int32_t total_sb)
{
    const int32_t global_tid  = blockIdx.x * 256 + threadIdx.x;
    const int32_t sb_idx      = global_tid / threads_per_sb;
    const int32_t d_chunk_idx = global_tid % threads_per_sb;
    if(sb_idx >= total_sb)
        return;
    const int32_t sid  = sb_idx % size_s;
    const int32_t bid  = sb_idx / size_s;
    const uint64_t tid = sid + p_cu_seqlens[bid];
    if(tid >= p_cu_seqlens[bid + 1])
        return;

    const int32_t offset_i = tid * stride_i_t;
    const int32_t offset_o = tid * stride_o_t;
    const int32_t offset_f = sid * size_f;

    Op::template apply_1c<RotateStyle,
                          ReuseFreqsFrontPart,
                          NopeFirst,
                          false,
                          StrideDOutEq1,
                          StrideDInEq1,
                          VecPairs>(p_output,
                                    p_input,
                                    (int32_t)offset_o,
                                    (int32_t)offset_i,
                                    p_freqs,
                                    (int32_t)offset_f,
                                    size_h,
                                    size_d,
                                    size_f,
                                    stride_i_h,
                                    stride_i_d,
                                    stride_o_h,
                                    stride_o_d,
                                    d_chunk_idx,
                                    threads_per_sb);
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          bool StrideDEq1,
          int32_t VecPairs = 1,
          typename scalar_t,
          typename scalar_f_t>
__launch_bounds__(256, 8) __global__
    void kn_entry_1c_thd_uncached_inplace(scalar_t* __restrict__ p_inout,
                                          const int32_t* __restrict__ p_cu_seqlens,
                                          const scalar_f_t* __restrict__ p_freqs,
                                          const int32_t size_h,
                                          const int32_t size_d,
                                          const int32_t size_f, // size of last dimension of freqs.
                                          const int32_t stride_t,
                                          const int32_t stride_h,
                                          const int32_t stride_d,
                                          const int32_t size_s,
                                          const int32_t threads_per_sb,
                                          const int32_t total_sb)
{
    const int32_t global_tid  = blockIdx.x * 256 + threadIdx.x;
    const int32_t sb_idx      = global_tid / threads_per_sb;
    const int32_t d_chunk_idx = global_tid % threads_per_sb;
    if(sb_idx >= total_sb)
        return;
    const int32_t sid  = sb_idx % size_s;
    const int32_t bid  = sb_idx / size_s;
    const uint64_t tid = sid + p_cu_seqlens[bid];
    if(tid >= p_cu_seqlens[bid + 1])
        return;

    const int32_t offset   = tid * stride_t;
    const int32_t offset_f = sid * size_f;

    Op::template apply_1c<RotateStyle,
                          ReuseFreqsFrontPart,
                          NopeFirst,
                          true,
                          StrideDEq1,
                          StrideDEq1,
                          VecPairs>(p_inout,
                                    p_inout,
                                    (int32_t)offset,
                                    (int32_t)offset,
                                    p_freqs,
                                    (int32_t)offset_f,
                                    size_h,
                                    size_d,
                                    size_f,
                                    stride_h,
                                    stride_d,
                                    stride_h,
                                    stride_d,
                                    d_chunk_idx,
                                    threads_per_sb);
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          bool StrideDOutEq1,
          bool StrideDInEq1,
          int32_t VecPairs = 1,
          typename scalar_t,
          typename scalar_f_t>
__launch_bounds__(256, 8) __global__
    void kn_entry_1c_2d_cached(scalar_t* __restrict__ p_output,
                               const scalar_t* __restrict__ p_input,
                               const scalar_f_t* __restrict__ p_cos_h,
                               const scalar_f_t* __restrict__ p_sin_h,
                               const scalar_f_t* __restrict__ p_cos_w,
                               const scalar_f_t* __restrict__ p_sin_w,
                               const int32_t img_width,
                               const int32_t size_h,
                               const int32_t size_d,
                               const int32_t stride_i_b,
                               const int32_t stride_i_s,
                               const int32_t stride_i_h,
                               const int32_t stride_i_d,
                               const int32_t stride_o_b,
                               const int32_t stride_o_s,
                               const int32_t stride_o_h,
                               const int32_t stride_o_d,
                               const int32_t size_s_h,
                               const int32_t threads_per_sb,
                               const int32_t total_sb)
{
    const int32_t global_tid  = blockIdx.x * 256 + threadIdx.x;
    const int32_t sb_idx      = global_tid / threads_per_sb;
    const int32_t d_chunk_idx = global_tid % threads_per_sb;
    if(sb_idx >= total_sb)
        return;
    const int32_t bid    = sb_idx / (size_s_h * img_width);
    const int32_t hw_idx = sb_idx % (size_s_h * img_width);
    const int32_t Hid    = hw_idx / img_width;
    const int32_t Wid    = hw_idx % img_width;
    const uint64_t sid   = Hid * img_width + Wid;

    const uint64_t size_half_d = size_d >> 1;

    const int offset_h_i = bid * stride_i_b + sid * stride_i_s;
    const int offset_h_o = bid * stride_o_b + sid * stride_o_s;
    const int offset_h_f = Hid * size_half_d;
    Op::template apply_1c<RotateStyle,
                          ReuseFreqsFrontPart,
                          NopeFirst,
                          false,
                          StrideDOutEq1,
                          StrideDInEq1,
                          VecPairs>(p_output,
                                    p_input,
                                    (int32_t)offset_h_o,
                                    (int32_t)offset_h_i,
                                    p_cos_h + offset_h_f,
                                    p_sin_h + offset_h_f,
                                    0, // offset already baked into pointer
                                    size_h,
                                    size_half_d,
                                    size_half_d,
                                    stride_i_h,
                                    stride_i_d,
                                    stride_o_h,
                                    stride_o_d,
                                    d_chunk_idx,
                                    threads_per_sb);

    const int offset_w_i = offset_h_i + size_half_d * stride_i_d;
    const int offset_w_o = offset_h_o + size_half_d * stride_o_d;
    const int offset_w_f = Wid * size_half_d;
    Op::template apply_1c<RotateStyle,
                          ReuseFreqsFrontPart,
                          NopeFirst,
                          false,
                          StrideDOutEq1,
                          StrideDInEq1,
                          VecPairs>(p_output,
                                    p_input,
                                    (int32_t)offset_w_o,
                                    (int32_t)offset_w_i,
                                    p_cos_w + offset_w_f,
                                    p_sin_w + offset_w_f,
                                    0, // offset already baked into pointer
                                    size_h,
                                    size_half_d,
                                    size_half_d,
                                    stride_i_h,
                                    stride_i_d,
                                    stride_o_h,
                                    stride_o_d,
                                    d_chunk_idx,
                                    threads_per_sb);
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          bool StrideDEq1,
          int32_t VecPairs = 1,
          typename scalar_t,
          typename scalar_f_t>
__launch_bounds__(256, 8) __global__
    void kn_entry_1c_2d_cached_inplace(scalar_t* __restrict__ p_inout,
                                       const scalar_f_t* __restrict__ p_cos_h,
                                       const scalar_f_t* __restrict__ p_sin_h,
                                       const scalar_f_t* __restrict__ p_cos_w,
                                       const scalar_f_t* __restrict__ p_sin_w,
                                       const int32_t img_width,
                                       const int32_t size_h,
                                       const int32_t size_d,
                                       const int32_t stride_b,
                                       const int32_t stride_s,
                                       const int32_t stride_h,
                                       const int32_t stride_d,
                                       const int32_t size_s_h,
                                       const int32_t threads_per_sb,
                                       const int32_t total_sb)
{
    const int32_t global_tid  = blockIdx.x * 256 + threadIdx.x;
    const int32_t sb_idx      = global_tid / threads_per_sb;
    const int32_t d_chunk_idx = global_tid % threads_per_sb;
    if(sb_idx >= total_sb)
        return;
    const int32_t bid    = sb_idx / (size_s_h * img_width);
    const int32_t hw_idx = sb_idx % (size_s_h * img_width);
    const int32_t Hid    = hw_idx / img_width;
    const int32_t Wid    = hw_idx % img_width;
    const uint64_t sid   = Hid * img_width + Wid;

    const uint64_t size_half_d = size_d >> 1;

    const int offset_h   = bid * stride_b + sid * stride_s;
    const int offset_h_f = Hid * size_half_d;
    Op::template apply_1c<RotateStyle,
                          ReuseFreqsFrontPart,
                          NopeFirst,
                          true,
                          StrideDEq1,
                          StrideDEq1,
                          VecPairs>(p_inout,
                                    p_inout,
                                    (int32_t)offset_h,
                                    (int32_t)offset_h,
                                    p_cos_h + offset_h_f,
                                    p_sin_h + offset_h_f,
                                    0, // offset already baked into pointer
                                    size_h,
                                    size_half_d,
                                    size_half_d,
                                    stride_h,
                                    stride_d,
                                    stride_h,
                                    stride_d,
                                    d_chunk_idx,
                                    threads_per_sb);

    const int offset_w   = offset_h + size_half_d * stride_d;
    const int offset_w_f = Wid * size_half_d;
    Op::template apply_1c<RotateStyle,
                          ReuseFreqsFrontPart,
                          NopeFirst,
                          true,
                          StrideDEq1,
                          StrideDEq1,
                          VecPairs>(p_inout,
                                    p_inout,
                                    (int32_t)offset_w,
                                    (int32_t)offset_w,
                                    p_cos_w + offset_w_f,
                                    p_sin_w + offset_w_f,
                                    0, // offset already baked into pointer
                                    size_h,
                                    size_half_d,
                                    size_half_d,
                                    stride_h,
                                    stride_d,
                                    stride_h,
                                    stride_d,
                                    d_chunk_idx,
                                    threads_per_sb);
}

// =====================================================================================================================
// Dispatches
//

#define LAUNCH_KERNEL_STRIDE_EQUAL_1_1_STRIDES(ROTATE_STYLE, STRIDE_0, ...) \
    if((STRIDE_0) == 1)                                                     \
    {                                                                       \
        constexpr bool Stride0Eq1 = true;                                   \
        __VA_ARGS__;                                                        \
    }                                                                       \
    else                                                                    \
    {                                                                       \
        constexpr bool Stride0Eq1 = false;                                  \
        __VA_ARGS__;                                                        \
    }

#define LAUNCH_KERNEL_STRIDE_EQUAL_1_2_STRIDES(ROTATE_STYLE, STRIDE_0, STRIDE_1, ...) \
    if((STRIDE_0) == 1)                                                               \
    {                                                                                 \
        constexpr bool Stride0Eq1 = true;                                             \
        if((STRIDE_1) == 1)                                                           \
        {                                                                             \
            constexpr bool Stride1Eq1 = true;                                         \
            __VA_ARGS__;                                                              \
        }                                                                             \
        else                                                                          \
        {                                                                             \
            constexpr bool Stride1Eq1 = false;                                        \
            __VA_ARGS__;                                                              \
        }                                                                             \
    }                                                                                 \
    else                                                                              \
    {                                                                                 \
        constexpr bool Stride0Eq1 = false;                                            \
        if((STRIDE_1) == 1)                                                           \
        {                                                                             \
            constexpr bool Stride1Eq1 = true;                                         \
            __VA_ARGS__;                                                              \
        }                                                                             \
        else                                                                          \
        {                                                                             \
            constexpr bool Stride1Eq1 = false;                                        \
            __VA_ARGS__;                                                              \
        }                                                                             \
    }

#define LAUNCH_KERNEL_STRIDE_EQUAL_1_4_STRIDES(                \
    ROTATE_STYLE, STRIDE_0, STRIDE_1, STRIDE_2, STRIDE_3, ...) \
    if((STRIDE_0) == 1)                                        \
    {                                                          \
        constexpr bool Stride0Eq1 = true;                      \
        if((STRIDE_1) == 1)                                    \
        {                                                      \
            constexpr bool Stride1Eq1 = true;                  \
            if((STRIDE_2) == 1)                                \
            {                                                  \
                constexpr bool Stride2Eq1 = true;              \
                if((STRIDE_3) == 1)                            \
                {                                              \
                    constexpr bool Stride3Eq1 = true;          \
                    __VA_ARGS__;                               \
                }                                              \
                else                                           \
                {                                              \
                    constexpr bool Stride3Eq1 = false;         \
                    __VA_ARGS__;                               \
                }                                              \
            }                                                  \
            else                                               \
            {                                                  \
                constexpr bool Stride2Eq1 = false;             \
                if((STRIDE_3) == 1)                            \
                {                                              \
                    constexpr bool Stride3Eq1 = true;          \
                    __VA_ARGS__;                               \
                }                                              \
                else                                           \
                {                                              \
                    constexpr bool Stride3Eq1 = false;         \
                    __VA_ARGS__;                               \
                }                                              \
            }                                                  \
        }                                                      \
        else                                                   \
        {                                                      \
            constexpr bool Stride1Eq1 = false;                 \
            if((STRIDE_2) == 1)                                \
            {                                                  \
                constexpr bool Stride2Eq1 = true;              \
                if((STRIDE_3) == 1)                            \
                {                                              \
                    constexpr bool Stride3Eq1 = true;          \
                    __VA_ARGS__;                               \
                }                                              \
                else                                           \
                {                                              \
                    constexpr bool Stride3Eq1 = false;         \
                    __VA_ARGS__;                               \
                }                                              \
            }                                                  \
            else                                               \
            {                                                  \
                constexpr bool Stride2Eq1 = false;             \
                if((STRIDE_3) == 1)                            \
                {                                              \
                    constexpr bool Stride3Eq1 = true;          \
                    __VA_ARGS__;                               \
                }                                              \
                else                                           \
                {                                              \
                    constexpr bool Stride3Eq1 = false;         \
                    __VA_ARGS__;                               \
                }                                              \
            }                                                  \
        }                                                      \
    }                                                          \
    else                                                       \
    {                                                          \
        constexpr bool Stride0Eq1 = false;                     \
        if((STRIDE_1) == 1)                                    \
        {                                                      \
            constexpr bool Stride1Eq1 = true;                  \
            if((STRIDE_2) == 1)                                \
            {                                                  \
                constexpr bool Stride2Eq1 = true;              \
                if((STRIDE_3) == 1)                            \
                {                                              \
                    constexpr bool Stride3Eq1 = true;          \
                    __VA_ARGS__;                               \
                }                                              \
                else                                           \
                {                                              \
                    constexpr bool Stride3Eq1 = false;         \
                    __VA_ARGS__;                               \
                }                                              \
            }                                                  \
            else                                               \
            {                                                  \
                constexpr bool Stride2Eq1 = false;             \
                if((STRIDE_3) == 1)                            \
                {                                              \
                    constexpr bool Stride3Eq1 = true;          \
                    __VA_ARGS__;                               \
                }                                              \
                else                                           \
                {                                              \
                    constexpr bool Stride3Eq1 = false;         \
                    __VA_ARGS__;                               \
                }                                              \
            }                                                  \
        }                                                      \
        else                                                   \
        {                                                      \
            constexpr bool Stride1Eq1 = false;                 \
            if((STRIDE_2) == 1)                                \
            {                                                  \
                constexpr bool Stride2Eq1 = true;              \
                if((STRIDE_3) == 1)                            \
                {                                              \
                    constexpr bool Stride3Eq1 = true;          \
                    __VA_ARGS__;                               \
                }                                              \
                else                                           \
                {                                              \
                    constexpr bool Stride3Eq1 = false;         \
                    __VA_ARGS__;                               \
                }                                              \
            }                                                  \
            else                                               \
            {                                                  \
                constexpr bool Stride2Eq1 = false;             \
                if((STRIDE_3) == 1)                            \
                {                                              \
                    constexpr bool Stride3Eq1 = true;          \
                    __VA_ARGS__;                               \
                }                                              \
                else                                           \
                {                                              \
                    constexpr bool Stride3Eq1 = false;         \
                    __VA_ARGS__;                               \
                }                                              \
            }                                                  \
        }                                                      \
    }

template <int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool Is2D,
          typename scalar_t = ck_tile::fp16_t>
std::tuple<dim3, dim3, int32_t, int32_t> get_grid_config(const int32_t size_s_h,
                                                         const int32_t size_s_w,
                                                         const int32_t size_b,
                                                         const int32_t size_f,
                                                         const bool stride_d_eq_1 = true)
{
    constexpr int32_t num_threads      = 256; // 4 warps x 64 threads/warp
    constexpr int32_t kernel_occupancy = 8;   // __launch_bounds__(256, 8)
    constexpr float threshold          = 4.0f;

    const int32_t size_r      = ReuseFreqsFrontPart ? (size_f << 1) : size_f;
    const int32_t size_half_r = size_r >> 1;
    const int32_t total_sb    = size_s_h * size_s_w * size_b;

    // VP selection depends on rotate style (benchmarked on gfx942):
    //   NEOX: VP=2 always wins (8-15% over VP=1, VP=4 always worst)
    //   GPTJ: VP=4 wins at saturated workloads, VP=1 is catastrophically slow (2-3x)
    // VP>1 requires contiguous d-dimension (stride_d==1) for vectorized buffer ops.
    constexpr int32_t preferred_vp = (RotateStyle == 1) ? 4 : 2; // GPTJ=1 -> VP=4, NEOX=0 -> VP=2
    int32_t vec_pairs              = stride_d_eq_1 ? preferred_vp : 1;

    while(vec_pairs > 1 && (size_half_r % vec_pairs != 0))
        vec_pairs >>= 1;

    // Fall back to smaller VP if not enough waves to saturate the GPU.
    const int32_t gpu_capacity  = static_cast<int32_t>(get_num_cu_func() * kernel_occupancy);
    constexpr int32_t warp_size = 64;
    while(vec_pairs > 1)
    {
        const int32_t total_waves = total_sb * (size_half_r / vec_pairs) / warp_size;
        if(total_waves >= static_cast<int32_t>(gpu_capacity * threshold))
            break;
        vec_pairs >>= 1;
    }

    const int32_t threads_per_sb = size_half_r / vec_pairs;
    const int32_t total_threads  = total_sb * threads_per_sb;

    dim3 grid((total_threads + num_threads - 1) / num_threads);
    dim3 block(num_threads);

    return {grid, block, vec_pairs, threads_per_sb};
}

#define LAUNCH_KERNEL_VEC_PAIRS(VEC_PAIRS, ...) \
    switch(VEC_PAIRS)                           \
    {                                           \
    case 2: {                                   \
        constexpr int32_t VP = 2;               \
        __VA_ARGS__;                            \
        break;                                  \
    }                                           \
    case 4: {                                   \
        constexpr int32_t VP = 4;               \
        __VA_ARGS__;                            \
        break;                                  \
    }                                           \
    default: {                                  \
        constexpr int32_t VP = 1;               \
        __VA_ARGS__;                            \
        break;                                  \
    }                                           \
    }

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          bool AllStrideDEq1 = false,
          typename scalar_t,
          typename scalar_f_t>
void dispatch_1c_sbhd_uncached(scalar_t* __restrict__ p_output,
                               const scalar_t* __restrict__ p_input,
                               const scalar_f_t* __restrict__ p_freqs,
                               const int32_t size_s,
                               const int32_t size_b,
                               const int32_t size_h,
                               const int32_t size_d,
                               const int32_t size_f, // size of last dimension of freqs.
                               const int32_t stride_i_s,
                               const int32_t stride_i_b,
                               const int32_t stride_i_h,
                               const int32_t stride_i_d,
                               const int32_t stride_o_s,
                               const int32_t stride_o_b,
                               const int32_t stride_o_h,
                               const int32_t stride_o_d)
{
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    const bool all_stride_d_eq_1 = (stride_i_d == 1) && (stride_o_d == 1);
    auto [grid, block, vec_pairs, threads_per_sb] =
        get_grid_config<RotateStyle, ReuseFreqsFrontPart, false, scalar_t>(
            size_s, 1, size_b, size_f, all_stride_d_eq_1);
    const int32_t total_sb = size_s * size_b;

    LAUNCH_KERNEL_VEC_PAIRS(
        vec_pairs,
        if(p_output == p_input) {
            assert(stride_i_s == stride_o_s);
            assert(stride_i_b == stride_o_b);
            assert(stride_i_h == stride_o_h);
            assert(stride_i_d == stride_o_d);

            if constexpr(AllStrideDEq1)
            {
                constexpr bool Stride0Eq1 = true;
                kn_entry_1c_sbhd_uncached_inplace<Op,
                                                  RotateStyle,
                                                  ReuseFreqsFrontPart,
                                                  NopeFirst,
                                                  Stride0Eq1,
                                                  VP><<<grid, block, 0, stream>>>(p_output,
                                                                                  p_freqs,
                                                                                  size_h,
                                                                                  size_d,
                                                                                  size_f,
                                                                                  stride_i_s,
                                                                                  stride_i_b,
                                                                                  stride_i_h,
                                                                                  stride_i_d,
                                                                                  size_s,
                                                                                  threads_per_sb,
                                                                                  total_sb);
            }
            else
            {
                LAUNCH_KERNEL_STRIDE_EQUAL_1_1_STRIDES(
                    RotateStyle,
                    stride_i_d,
                    kn_entry_1c_sbhd_uncached_inplace<Op,
                                                      RotateStyle,
                                                      ReuseFreqsFrontPart,
                                                      NopeFirst,
                                                      Stride0Eq1,
                                                      VP>
                    <<<grid, block, 0, stream>>>(p_output,
                                                 p_freqs,
                                                 size_h,
                                                 size_d,
                                                 size_f,
                                                 stride_i_s,
                                                 stride_i_b,
                                                 stride_i_h,
                                                 stride_i_d,
                                                 size_s,
                                                 threads_per_sb,
                                                 total_sb););
            }
        } else {
            if constexpr(AllStrideDEq1)
            {
                constexpr bool Stride0Eq1 = true;
                constexpr bool Stride1Eq1 = true;
                kn_entry_1c_sbhd_uncached<Op,
                                          RotateStyle,
                                          ReuseFreqsFrontPart,
                                          NopeFirst,
                                          Stride0Eq1,
                                          Stride1Eq1,
                                          VP><<<grid, block, 0, stream>>>(p_output,
                                                                          p_input,
                                                                          p_freqs,
                                                                          size_h,
                                                                          size_d,
                                                                          size_f,
                                                                          stride_i_s,
                                                                          stride_i_b,
                                                                          stride_i_h,
                                                                          stride_i_d,
                                                                          stride_o_s,
                                                                          stride_o_b,
                                                                          stride_o_h,
                                                                          stride_o_d,
                                                                          size_s,
                                                                          threads_per_sb,
                                                                          total_sb);
            }
            else
            {
                LAUNCH_KERNEL_STRIDE_EQUAL_1_2_STRIDES(
                    RotateStyle,
                    stride_o_d,
                    stride_i_d,
                    kn_entry_1c_sbhd_uncached<Op,
                                              RotateStyle,
                                              ReuseFreqsFrontPart,
                                              NopeFirst,
                                              Stride0Eq1,
                                              Stride1Eq1,
                                              VP><<<grid, block, 0, stream>>>(p_output,
                                                                              p_input,
                                                                              p_freqs,
                                                                              size_h,
                                                                              size_d,
                                                                              size_f,
                                                                              stride_i_s,
                                                                              stride_i_b,
                                                                              stride_i_h,
                                                                              stride_i_d,
                                                                              stride_o_s,
                                                                              stride_o_b,
                                                                              stride_o_h,
                                                                              stride_o_d,
                                                                              size_s,
                                                                              threads_per_sb,
                                                                              total_sb););
            }
        });
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          bool AllStrideDEq1 = false,
          typename scalar_t,
          typename scalar_f_t>
void dispatch_2c_sbhd_uncached(scalar_t* __restrict__ p_output_x,
                               scalar_t* __restrict__ p_output_y,
                               const scalar_t* __restrict__ p_input_x,
                               const scalar_t* __restrict__ p_input_y,
                               const scalar_f_t* __restrict__ p_freqs,
                               const int32_t size_s,
                               const int32_t size_b,
                               const int32_t size_h_x,
                               const int32_t size_h_y,
                               const int32_t size_d,
                               const int32_t size_f, // size of last dimension of freqs.
                               const int32_t stride_ix_s,
                               const int32_t stride_ix_b,
                               const int32_t stride_ix_h,
                               const int32_t stride_ix_d,
                               const int32_t stride_iy_s,
                               const int32_t stride_iy_b,
                               const int32_t stride_iy_h,
                               const int32_t stride_iy_d,
                               const int32_t stride_ox_s,
                               const int32_t stride_ox_b,
                               const int32_t stride_ox_h,
                               const int32_t stride_ox_d,
                               const int32_t stride_oy_s,
                               const int32_t stride_oy_b,
                               const int32_t stride_oy_h,
                               const int32_t stride_oy_d)
{
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    const bool all_stride_d_eq_1 =
        (stride_ix_d == 1) && (stride_iy_d == 1) && (stride_ox_d == 1) && (stride_oy_d == 1);
    auto [grid, block, vec_pairs, threads_per_sb] =
        get_grid_config<RotateStyle, ReuseFreqsFrontPart, false, scalar_t>(
            size_s, 1, size_b, size_f, all_stride_d_eq_1);
    const int32_t total_sb = size_s * size_b;

    LAUNCH_KERNEL_VEC_PAIRS(
        vec_pairs,
        if((p_output_x == p_input_x) && (p_output_y == p_input_y)) {
            assert(stride_ix_s == stride_ox_s);
            assert(stride_ix_b == stride_ox_b);
            assert(stride_ix_h == stride_ox_h);
            assert(stride_ix_d == stride_ox_d);
            assert(stride_iy_s == stride_oy_s);
            assert(stride_iy_b == stride_oy_b);
            assert(stride_iy_h == stride_oy_h);
            assert(stride_iy_d == stride_oy_d);

            if constexpr(AllStrideDEq1)
            {
                constexpr bool Stride0Eq1 = true;
                constexpr bool Stride1Eq1 = true;
                kn_entry_2c_sbhd_uncached_inplace<Op,
                                                  RotateStyle,
                                                  ReuseFreqsFrontPart,
                                                  NopeFirst,
                                                  Stride0Eq1,
                                                  Stride1Eq1,
                                                  VP><<<grid, block, 0, stream>>>(p_output_x,
                                                                                  p_output_y,
                                                                                  p_freqs,
                                                                                  size_h_x,
                                                                                  size_h_y,
                                                                                  size_d,
                                                                                  size_f,
                                                                                  stride_ix_s,
                                                                                  stride_ix_b,
                                                                                  stride_ix_h,
                                                                                  stride_ix_d,
                                                                                  stride_iy_s,
                                                                                  stride_iy_b,
                                                                                  stride_iy_h,
                                                                                  stride_iy_d,
                                                                                  size_s,
                                                                                  threads_per_sb,
                                                                                  total_sb);
            }
            else
            {
                LAUNCH_KERNEL_STRIDE_EQUAL_1_2_STRIDES(
                    RotateStyle,
                    stride_ix_d,
                    stride_iy_d,
                    kn_entry_2c_sbhd_uncached_inplace<Op,
                                                      RotateStyle,
                                                      ReuseFreqsFrontPart,
                                                      NopeFirst,
                                                      Stride0Eq1,
                                                      Stride1Eq1,
                                                      VP>
                    <<<grid, block, 0, stream>>>(p_output_x,
                                                 p_output_y,
                                                 p_freqs,
                                                 size_h_x,
                                                 size_h_y,
                                                 size_d,
                                                 size_f,
                                                 stride_ix_s,
                                                 stride_ix_b,
                                                 stride_ix_h,
                                                 stride_ix_d,
                                                 stride_iy_s,
                                                 stride_iy_b,
                                                 stride_iy_h,
                                                 stride_iy_d,
                                                 size_s,
                                                 threads_per_sb,
                                                 total_sb););
            }
        } else {
            if constexpr(AllStrideDEq1)
            {
                constexpr bool Stride0Eq1 = true;
                constexpr bool Stride1Eq1 = true;
                constexpr bool Stride2Eq1 = true;
                constexpr bool Stride3Eq1 = true;
                kn_entry_2c_sbhd_uncached<Op,
                                          RotateStyle,
                                          ReuseFreqsFrontPart,
                                          NopeFirst,
                                          Stride0Eq1,
                                          Stride1Eq1,
                                          Stride2Eq1,
                                          Stride3Eq1,
                                          VP><<<grid, block, 0, stream>>>(p_output_x,
                                                                          p_output_y,
                                                                          p_input_x,
                                                                          p_input_y,
                                                                          p_freqs,
                                                                          size_h_x,
                                                                          size_h_y,
                                                                          size_d,
                                                                          size_f,
                                                                          stride_ix_s,
                                                                          stride_ix_b,
                                                                          stride_ix_h,
                                                                          stride_ix_d,
                                                                          stride_iy_s,
                                                                          stride_iy_b,
                                                                          stride_iy_h,
                                                                          stride_iy_d,
                                                                          stride_ox_s,
                                                                          stride_ox_b,
                                                                          stride_ox_h,
                                                                          stride_ox_d,
                                                                          stride_oy_s,
                                                                          stride_oy_b,
                                                                          stride_oy_h,
                                                                          stride_oy_d,
                                                                          size_s,
                                                                          threads_per_sb,
                                                                          total_sb);
            }
            else
            {
                LAUNCH_KERNEL_STRIDE_EQUAL_1_4_STRIDES(
                    RotateStyle,
                    stride_ox_d,
                    stride_oy_d,
                    stride_ix_d,
                    stride_iy_d,
                    kn_entry_2c_sbhd_uncached<Op,
                                              RotateStyle,
                                              ReuseFreqsFrontPart,
                                              NopeFirst,
                                              Stride0Eq1,
                                              Stride1Eq1,
                                              Stride2Eq1,
                                              Stride3Eq1,
                                              VP><<<grid, block, 0, stream>>>(p_output_x,
                                                                              p_output_y,
                                                                              p_input_x,
                                                                              p_input_y,
                                                                              p_freqs,
                                                                              size_h_x,
                                                                              size_h_y,
                                                                              size_d,
                                                                              size_f,
                                                                              stride_ix_s,
                                                                              stride_ix_b,
                                                                              stride_ix_h,
                                                                              stride_ix_d,
                                                                              stride_iy_s,
                                                                              stride_iy_b,
                                                                              stride_iy_h,
                                                                              stride_iy_d,
                                                                              stride_ox_s,
                                                                              stride_ox_b,
                                                                              stride_ox_h,
                                                                              stride_ox_d,
                                                                              stride_oy_s,
                                                                              stride_oy_b,
                                                                              stride_oy_h,
                                                                              stride_oy_d,
                                                                              size_s,
                                                                              threads_per_sb,
                                                                              total_sb););
            }
        });
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          bool AllStrideDEq1 = false,
          typename scalar_t,
          typename scalar_f_t>
void dispatch_1c_sbhd_cached(scalar_t* __restrict__ p_output,
                             const scalar_t* __restrict__ p_input,
                             const scalar_f_t* __restrict__ p_cos,
                             const scalar_f_t* __restrict__ p_sin,
                             const int32_t size_s,
                             const int32_t size_b,
                             const int32_t size_h,
                             const int32_t size_d,
                             const int32_t size_f, // size of last dimension of freqs.
                             const int32_t stride_i_s,
                             const int32_t stride_i_b,
                             const int32_t stride_i_h,
                             const int32_t stride_i_d,
                             const int32_t stride_o_s,
                             const int32_t stride_o_b,
                             const int32_t stride_o_h,
                             const int32_t stride_o_d)
{
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    const bool all_stride_d_eq_1 = (stride_i_d == 1) && (stride_o_d == 1);
    auto [grid, block, vec_pairs, threads_per_sb] =
        get_grid_config<RotateStyle, ReuseFreqsFrontPart, false, scalar_t>(
            size_s, 1, size_b, size_f, all_stride_d_eq_1);
    const int32_t total_sb = size_s * size_b;

    LAUNCH_KERNEL_VEC_PAIRS(
        vec_pairs,
        if(p_output == p_input) {
            assert(stride_i_s == stride_o_s);
            assert(stride_i_b == stride_o_b);
            assert(stride_i_h == stride_o_h);
            assert(stride_i_d == stride_o_d);

            if constexpr(AllStrideDEq1)
            {
                constexpr bool Stride0Eq1 = true;
                kn_entry_1c_sbhd_cached_inplace<Op,
                                                RotateStyle,
                                                ReuseFreqsFrontPart,
                                                NopeFirst,
                                                Stride0Eq1,
                                                VP><<<grid, block, 0, stream>>>(p_output,
                                                                                p_cos,
                                                                                p_sin,
                                                                                size_h,
                                                                                size_d,
                                                                                size_f,
                                                                                stride_i_s,
                                                                                stride_i_b,
                                                                                stride_i_h,
                                                                                stride_i_d,
                                                                                size_s,
                                                                                threads_per_sb,
                                                                                total_sb);
            }
            else
            {
                LAUNCH_KERNEL_STRIDE_EQUAL_1_1_STRIDES(
                    RotateStyle,
                    stride_i_d,
                    kn_entry_1c_sbhd_cached_inplace<Op,
                                                    RotateStyle,
                                                    ReuseFreqsFrontPart,
                                                    NopeFirst,
                                                    Stride0Eq1,
                                                    VP><<<grid, block, 0, stream>>>(p_output,
                                                                                    p_cos,
                                                                                    p_sin,
                                                                                    size_h,
                                                                                    size_d,
                                                                                    size_f,
                                                                                    stride_i_s,
                                                                                    stride_i_b,
                                                                                    stride_i_h,
                                                                                    stride_i_d,
                                                                                    size_s,
                                                                                    threads_per_sb,
                                                                                    total_sb););
            }
        } else {
            if constexpr(AllStrideDEq1)
            {
                constexpr bool Stride0Eq1 = true;
                constexpr bool Stride1Eq1 = true;
                kn_entry_1c_sbhd_cached<Op,
                                        RotateStyle,
                                        ReuseFreqsFrontPart,
                                        NopeFirst,
                                        Stride0Eq1,
                                        Stride1Eq1,
                                        VP><<<grid, block, 0, stream>>>(p_output,
                                                                        p_input,
                                                                        p_cos,
                                                                        p_sin,
                                                                        size_h,
                                                                        size_d,
                                                                        size_f,
                                                                        stride_i_s,
                                                                        stride_i_b,
                                                                        stride_i_h,
                                                                        stride_i_d,
                                                                        stride_o_s,
                                                                        stride_o_b,
                                                                        stride_o_h,
                                                                        stride_o_d,
                                                                        size_s,
                                                                        threads_per_sb,
                                                                        total_sb);
            }
            else
            {
                LAUNCH_KERNEL_STRIDE_EQUAL_1_2_STRIDES(RotateStyle,
                                                       stride_o_d,
                                                       stride_i_d,
                                                       kn_entry_1c_sbhd_cached<Op,
                                                                               RotateStyle,
                                                                               ReuseFreqsFrontPart,
                                                                               NopeFirst,
                                                                               Stride0Eq1,
                                                                               Stride1Eq1,
                                                                               VP>
                                                       <<<grid, block, 0, stream>>>(p_output,
                                                                                    p_input,
                                                                                    p_cos,
                                                                                    p_sin,
                                                                                    size_h,
                                                                                    size_d,
                                                                                    size_f,
                                                                                    stride_i_s,
                                                                                    stride_i_b,
                                                                                    stride_i_h,
                                                                                    stride_i_d,
                                                                                    stride_o_s,
                                                                                    stride_o_b,
                                                                                    stride_o_h,
                                                                                    stride_o_d,
                                                                                    size_s,
                                                                                    threads_per_sb,
                                                                                    total_sb););
            }
        });
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          bool AllStrideDEq1 = false,
          typename scalar_t,
          typename scalar_f_t>
void dispatch_2c_sbhd_cached(scalar_t* __restrict__ p_output_x,
                             scalar_t* __restrict__ p_output_y,
                             const scalar_t* __restrict__ p_input_x,
                             const scalar_t* __restrict__ p_input_y,
                             const scalar_f_t* __restrict__ p_cos,
                             const scalar_f_t* __restrict__ p_sin,
                             const int32_t size_s,
                             const int32_t size_b,
                             const int32_t size_h_x,
                             const int32_t size_h_y,
                             const int32_t size_d,
                             const int32_t size_f, // size of last dimension of freqs.
                             const int32_t stride_ix_s,
                             const int32_t stride_ix_b,
                             const int32_t stride_ix_h,
                             const int32_t stride_ix_d,
                             const int32_t stride_iy_s,
                             const int32_t stride_iy_b,
                             const int32_t stride_iy_h,
                             const int32_t stride_iy_d,
                             const int32_t stride_ox_s,
                             const int32_t stride_ox_b,
                             const int32_t stride_ox_h,
                             const int32_t stride_ox_d,
                             const int32_t stride_oy_s,
                             const int32_t stride_oy_b,
                             const int32_t stride_oy_h,
                             const int32_t stride_oy_d)
{
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    const bool all_stride_d_eq_1 =
        (stride_ix_d == 1) && (stride_iy_d == 1) && (stride_ox_d == 1) && (stride_oy_d == 1);
    auto [grid, block, vec_pairs, threads_per_sb] =
        get_grid_config<RotateStyle, ReuseFreqsFrontPart, false, scalar_t>(
            size_s, 1, size_b, size_f, all_stride_d_eq_1);
    const int32_t total_sb = size_s * size_b;

    LAUNCH_KERNEL_VEC_PAIRS(
        vec_pairs,
        if((p_output_x == p_input_x) && (p_output_y == p_input_y)) {
            assert(stride_ix_s == stride_ox_s);
            assert(stride_ix_b == stride_ox_b);
            assert(stride_ix_h == stride_ox_h);
            assert(stride_ix_d == stride_ox_d);
            assert(stride_iy_s == stride_oy_s);
            assert(stride_iy_b == stride_oy_b);
            assert(stride_iy_h == stride_oy_h);
            assert(stride_iy_d == stride_oy_d);

            if constexpr(AllStrideDEq1)
            {
                constexpr bool Stride0Eq1 = true;
                constexpr bool Stride1Eq1 = true;
                kn_entry_2c_sbhd_cached_inplace<Op,
                                                RotateStyle,
                                                ReuseFreqsFrontPart,
                                                NopeFirst,
                                                Stride0Eq1,
                                                Stride1Eq1,
                                                VP><<<grid, block, 0, stream>>>(p_output_x,
                                                                                p_output_y,
                                                                                p_cos,
                                                                                p_sin,
                                                                                size_h_x,
                                                                                size_h_y,
                                                                                size_d,
                                                                                size_f,
                                                                                stride_ix_s,
                                                                                stride_ix_b,
                                                                                stride_ix_h,
                                                                                stride_ix_d,
                                                                                stride_iy_s,
                                                                                stride_iy_b,
                                                                                stride_iy_h,
                                                                                stride_iy_d,
                                                                                size_s,
                                                                                threads_per_sb,
                                                                                total_sb);
            }
            else
            {
                LAUNCH_KERNEL_STRIDE_EQUAL_1_2_STRIDES(
                    RotateStyle,
                    stride_ix_d,
                    stride_iy_d,
                    kn_entry_2c_sbhd_cached_inplace<Op,
                                                    RotateStyle,
                                                    ReuseFreqsFrontPart,
                                                    NopeFirst,
                                                    Stride0Eq1,
                                                    Stride1Eq1,
                                                    VP><<<grid, block, 0, stream>>>(p_output_x,
                                                                                    p_output_y,
                                                                                    p_cos,
                                                                                    p_sin,
                                                                                    size_h_x,
                                                                                    size_h_y,
                                                                                    size_d,
                                                                                    size_f,
                                                                                    stride_ix_s,
                                                                                    stride_ix_b,
                                                                                    stride_ix_h,
                                                                                    stride_ix_d,
                                                                                    stride_iy_s,
                                                                                    stride_iy_b,
                                                                                    stride_iy_h,
                                                                                    stride_iy_d,
                                                                                    size_s,
                                                                                    threads_per_sb,
                                                                                    total_sb););
            }
        } else {
            if constexpr(AllStrideDEq1)
            {
                constexpr bool Stride0Eq1 = true;
                constexpr bool Stride1Eq1 = true;
                constexpr bool Stride2Eq1 = true;
                constexpr bool Stride3Eq1 = true;
                kn_entry_2c_sbhd_cached<Op,
                                        RotateStyle,
                                        ReuseFreqsFrontPart,
                                        NopeFirst,
                                        Stride0Eq1,
                                        Stride1Eq1,
                                        Stride2Eq1,
                                        Stride3Eq1,
                                        VP><<<grid, block, 0, stream>>>(p_output_x,
                                                                        p_output_y,
                                                                        p_input_x,
                                                                        p_input_y,
                                                                        p_cos,
                                                                        p_sin,
                                                                        size_h_x,
                                                                        size_h_y,
                                                                        size_d,
                                                                        size_f,
                                                                        stride_ix_s,
                                                                        stride_ix_b,
                                                                        stride_ix_h,
                                                                        stride_ix_d,
                                                                        stride_iy_s,
                                                                        stride_iy_b,
                                                                        stride_iy_h,
                                                                        stride_iy_d,
                                                                        stride_ox_s,
                                                                        stride_ox_b,
                                                                        stride_ox_h,
                                                                        stride_ox_d,
                                                                        stride_oy_s,
                                                                        stride_oy_b,
                                                                        stride_oy_h,
                                                                        stride_oy_d,
                                                                        size_s,
                                                                        threads_per_sb,
                                                                        total_sb);
            }
            else
            {
                LAUNCH_KERNEL_STRIDE_EQUAL_1_4_STRIDES(RotateStyle,
                                                       stride_ox_d,
                                                       stride_oy_d,
                                                       stride_ix_d,
                                                       stride_iy_d,
                                                       kn_entry_2c_sbhd_cached<Op,
                                                                               RotateStyle,
                                                                               ReuseFreqsFrontPart,
                                                                               NopeFirst,
                                                                               Stride0Eq1,
                                                                               Stride1Eq1,
                                                                               Stride2Eq1,
                                                                               Stride3Eq1,
                                                                               VP>
                                                       <<<grid, block, 0, stream>>>(p_output_x,
                                                                                    p_output_y,
                                                                                    p_input_x,
                                                                                    p_input_y,
                                                                                    p_cos,
                                                                                    p_sin,
                                                                                    size_h_x,
                                                                                    size_h_y,
                                                                                    size_d,
                                                                                    size_f,
                                                                                    stride_ix_s,
                                                                                    stride_ix_b,
                                                                                    stride_ix_h,
                                                                                    stride_ix_d,
                                                                                    stride_iy_s,
                                                                                    stride_iy_b,
                                                                                    stride_iy_h,
                                                                                    stride_iy_d,
                                                                                    stride_ox_s,
                                                                                    stride_ox_b,
                                                                                    stride_ox_h,
                                                                                    stride_ox_d,
                                                                                    stride_oy_s,
                                                                                    stride_oy_b,
                                                                                    stride_oy_h,
                                                                                    stride_oy_d,
                                                                                    size_s,
                                                                                    threads_per_sb,
                                                                                    total_sb););
            }
        });
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          typename scalar_t,
          typename scalar_f_t>
void dispatch_1c_sbhd_cached_indirect(scalar_t* __restrict__ p_output,
                                      const scalar_t* __restrict__ p_input,
                                      const scalar_f_t* __restrict__ p_cos,
                                      const scalar_f_t* __restrict__ p_sin,
                                      const int64_t* __restrict__ p_indirect_buffer,
                                      const int32_t max_position,
                                      const int32_t size_s,
                                      const int32_t size_b,
                                      const int32_t size_h,
                                      const int32_t size_d,
                                      const int32_t size_f, // size of last dimension of freqs.
                                      const int32_t stride_i_s,
                                      const int32_t stride_i_b,
                                      const int32_t stride_i_h,
                                      const int32_t stride_i_d,
                                      const int32_t stride_o_s,
                                      const int32_t stride_o_b,
                                      const int32_t stride_o_h,
                                      const int32_t stride_o_d)
{
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    const bool all_stride_d_eq_1 = (stride_i_d == 1) && (stride_o_d == 1);
    auto [grid, block, vec_pairs, threads_per_sb] =
        get_grid_config<RotateStyle, ReuseFreqsFrontPart, false, scalar_t>(
            size_s, 1, size_b, size_f, all_stride_d_eq_1);
    const int32_t total_sb = size_s * size_b;

    LAUNCH_KERNEL_VEC_PAIRS(
        vec_pairs,
        if(p_output == p_input) {
            assert(stride_i_s == stride_o_s);
            assert(stride_i_b == stride_o_b);
            assert(stride_i_h == stride_o_h);
            assert(stride_i_d == stride_o_d);

            LAUNCH_KERNEL_STRIDE_EQUAL_1_1_STRIDES(
                RotateStyle,
                stride_i_d,
                kn_entry_1c_sbhd_cached_indirect_inplace<Op,
                                                         RotateStyle,
                                                         ReuseFreqsFrontPart,
                                                         NopeFirst,
                                                         Stride0Eq1,
                                                         VP>
                <<<grid, block, 0, stream>>>(p_output,
                                             p_cos,
                                             p_sin,
                                             p_indirect_buffer,
                                             max_position,
                                             size_h,
                                             size_d,
                                             size_f,
                                             stride_i_s,
                                             stride_i_b,
                                             stride_i_h,
                                             stride_i_d,
                                             size_s,
                                             threads_per_sb,
                                             total_sb););
        } else {
            LAUNCH_KERNEL_STRIDE_EQUAL_1_2_STRIDES(
                RotateStyle,
                stride_o_d,
                stride_i_d,
                kn_entry_1c_sbhd_cached_indirect<Op,
                                                 RotateStyle,
                                                 ReuseFreqsFrontPart,
                                                 NopeFirst,
                                                 Stride0Eq1,
                                                 Stride1Eq1,
                                                 VP><<<grid, block, 0, stream>>>(p_output,
                                                                                 p_input,
                                                                                 p_cos,
                                                                                 p_sin,
                                                                                 p_indirect_buffer,
                                                                                 max_position,
                                                                                 size_h,
                                                                                 size_d,
                                                                                 size_f,
                                                                                 stride_i_s,
                                                                                 stride_i_b,
                                                                                 stride_i_h,
                                                                                 stride_i_d,
                                                                                 stride_o_s,
                                                                                 stride_o_b,
                                                                                 stride_o_h,
                                                                                 stride_o_d,
                                                                                 size_s,
                                                                                 threads_per_sb,
                                                                                 total_sb););
        });
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          typename scalar_t,
          typename scalar_f_t>
void dispatch_2c_sbhd_cached_indirect(scalar_t* __restrict__ p_output_x,
                                      scalar_t* __restrict__ p_output_y,
                                      const scalar_t* __restrict__ p_input_x,
                                      const scalar_t* __restrict__ p_input_y,
                                      const scalar_f_t* __restrict__ p_cos,
                                      const scalar_f_t* __restrict__ p_sin,
                                      const int64_t* __restrict__ p_indirect_buffer,
                                      const int32_t max_position,
                                      const int32_t size_s,
                                      const int32_t size_b,
                                      const int32_t size_h_x,
                                      const int32_t size_h_y,
                                      const int32_t size_d,
                                      const int32_t size_f, // size of last dimension of freqs.
                                      const int32_t stride_ix_s,
                                      const int32_t stride_ix_b,
                                      const int32_t stride_ix_h,
                                      const int32_t stride_ix_d,
                                      const int32_t stride_iy_s,
                                      const int32_t stride_iy_b,
                                      const int32_t stride_iy_h,
                                      const int32_t stride_iy_d,
                                      const int32_t stride_ox_s,
                                      const int32_t stride_ox_b,
                                      const int32_t stride_ox_h,
                                      const int32_t stride_ox_d,
                                      const int32_t stride_oy_s,
                                      const int32_t stride_oy_b,
                                      const int32_t stride_oy_h,
                                      const int32_t stride_oy_d)
{
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    const bool all_stride_d_eq_1 =
        (stride_ix_d == 1) && (stride_iy_d == 1) && (stride_ox_d == 1) && (stride_oy_d == 1);
    auto [grid, block, vec_pairs, threads_per_sb] =
        get_grid_config<RotateStyle, ReuseFreqsFrontPart, false, scalar_t>(
            size_s, 1, size_b, size_f, all_stride_d_eq_1);
    const int32_t total_sb = size_s * size_b;

    LAUNCH_KERNEL_VEC_PAIRS(
        vec_pairs,
        if((p_output_x == p_input_x) && (p_output_y == p_input_y)) {
            assert(stride_ix_s == stride_ox_s);
            assert(stride_ix_b == stride_ox_b);
            assert(stride_ix_h == stride_ox_h);
            assert(stride_ix_d == stride_ox_d);
            assert(stride_iy_s == stride_oy_s);
            assert(stride_iy_b == stride_oy_b);
            assert(stride_iy_h == stride_oy_h);
            assert(stride_iy_d == stride_oy_d);

            LAUNCH_KERNEL_STRIDE_EQUAL_1_2_STRIDES(
                RotateStyle,
                stride_ix_d,
                stride_iy_d,
                kn_entry_2c_sbhd_cached_indirect_inplace<Op,
                                                         RotateStyle,
                                                         ReuseFreqsFrontPart,
                                                         NopeFirst,
                                                         Stride0Eq1,
                                                         Stride1Eq1,
                                                         VP>
                <<<grid, block, 0, stream>>>(p_output_x,
                                             p_output_y,
                                             p_cos,
                                             p_sin,
                                             p_indirect_buffer,
                                             max_position,
                                             size_h_x,
                                             size_h_y,
                                             size_d,
                                             size_f,
                                             stride_ix_s,
                                             stride_ix_b,
                                             stride_ix_h,
                                             stride_ix_d,
                                             stride_iy_s,
                                             stride_iy_b,
                                             stride_iy_h,
                                             stride_iy_d,
                                             size_s,
                                             threads_per_sb,
                                             total_sb););
        } else {
            LAUNCH_KERNEL_STRIDE_EQUAL_1_4_STRIDES(
                RotateStyle,
                stride_ox_d,
                stride_oy_d,
                stride_ix_d,
                stride_iy_d,
                kn_entry_2c_sbhd_cached_indirect<Op,
                                                 RotateStyle,
                                                 ReuseFreqsFrontPart,
                                                 NopeFirst,
                                                 Stride0Eq1,
                                                 Stride1Eq1,
                                                 Stride2Eq1,
                                                 Stride3Eq1,
                                                 VP><<<grid, block, 0, stream>>>(p_output_x,
                                                                                 p_output_y,
                                                                                 p_input_x,
                                                                                 p_input_y,
                                                                                 p_cos,
                                                                                 p_sin,
                                                                                 p_indirect_buffer,
                                                                                 max_position,
                                                                                 size_h_x,
                                                                                 size_h_y,
                                                                                 size_d,
                                                                                 size_f,
                                                                                 stride_ix_s,
                                                                                 stride_ix_b,
                                                                                 stride_ix_h,
                                                                                 stride_ix_d,
                                                                                 stride_iy_s,
                                                                                 stride_iy_b,
                                                                                 stride_iy_h,
                                                                                 stride_iy_d,
                                                                                 stride_ox_s,
                                                                                 stride_ox_b,
                                                                                 stride_ox_h,
                                                                                 stride_ox_d,
                                                                                 stride_oy_s,
                                                                                 stride_oy_b,
                                                                                 stride_oy_h,
                                                                                 stride_oy_d,
                                                                                 size_s,
                                                                                 threads_per_sb,
                                                                                 total_sb););
        });
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          typename scalar_t,
          typename scalar_f_t>
void dispatch_1c_sbhd_cached_indirect2(scalar_t* __restrict__ p_output,
                                       const scalar_t* __restrict__ p_input,
                                       const scalar_f_t* __restrict__ p_cos,
                                       const scalar_f_t* __restrict__ p_sin,
                                       const int64_t* __restrict__ p_indirect_buffer_0,
                                       const int64_t* __restrict__ p_indirect_buffer_1,
                                       const int32_t max_position,
                                       const int32_t size_s,
                                       const int32_t size_b,
                                       const int32_t size_h,
                                       const int32_t size_d,
                                       const int32_t size_f, // size of last dimension of freqs.
                                       const int32_t stride_i_s,
                                       const int32_t stride_i_b,
                                       const int32_t stride_i_h,
                                       const int32_t stride_i_d,
                                       const int32_t stride_o_s,
                                       const int32_t stride_o_b,
                                       const int32_t stride_o_h,
                                       const int32_t stride_o_d)
{
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    const bool all_stride_d_eq_1 = (stride_i_d == 1) && (stride_o_d == 1);
    auto [grid, block, vec_pairs, threads_per_sb] =
        get_grid_config<RotateStyle, ReuseFreqsFrontPart, false, scalar_t>(
            size_s, 1, size_b, size_f, all_stride_d_eq_1);
    const int32_t total_sb = size_s * size_b;

    LAUNCH_KERNEL_VEC_PAIRS(
        vec_pairs,
        if(p_output == p_input) {
            assert(stride_i_s == stride_o_s);
            assert(stride_i_b == stride_o_b);
            assert(stride_i_h == stride_o_h);
            assert(stride_i_d == stride_o_d);

            LAUNCH_KERNEL_STRIDE_EQUAL_1_1_STRIDES(
                RotateStyle,
                stride_i_d,
                kn_entry_1c_sbhd_cached_indirect2_inplace<Op,
                                                          RotateStyle,
                                                          ReuseFreqsFrontPart,
                                                          NopeFirst,
                                                          Stride0Eq1,
                                                          VP>
                <<<grid, block, 0, stream>>>(p_output,
                                             p_cos,
                                             p_sin,
                                             p_indirect_buffer_0,
                                             p_indirect_buffer_1,
                                             max_position,
                                             size_h,
                                             size_d,
                                             size_f,
                                             stride_i_s,
                                             stride_i_b,
                                             stride_i_h,
                                             stride_i_d,
                                             size_s,
                                             threads_per_sb,
                                             total_sb););
        } else {
            LAUNCH_KERNEL_STRIDE_EQUAL_1_2_STRIDES(
                RotateStyle,
                stride_o_d,
                stride_i_d,
                kn_entry_1c_sbhd_cached_indirect2<Op,
                                                  RotateStyle,
                                                  ReuseFreqsFrontPart,
                                                  NopeFirst,
                                                  Stride0Eq1,
                                                  Stride1Eq1,
                                                  VP>
                <<<grid, block, 0, stream>>>(p_output,
                                             p_input,
                                             p_cos,
                                             p_sin,
                                             p_indirect_buffer_0,
                                             p_indirect_buffer_1,
                                             max_position,
                                             size_h,
                                             size_d,
                                             size_f,
                                             stride_i_s,
                                             stride_i_b,
                                             stride_i_h,
                                             stride_i_d,
                                             stride_o_s,
                                             stride_o_b,
                                             stride_o_h,
                                             stride_o_d,
                                             size_s,
                                             threads_per_sb,
                                             total_sb););
        });
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          typename scalar_t,
          typename scalar_f_t>
void dispatch_2c_sbhd_cached_indirect2(scalar_t* __restrict__ p_output_x,
                                       scalar_t* __restrict__ p_output_y,
                                       const scalar_t* __restrict__ p_input_x,
                                       const scalar_t* __restrict__ p_input_y,
                                       const scalar_f_t* __restrict__ p_cos,
                                       const scalar_f_t* __restrict__ p_sin,
                                       const int64_t* __restrict__ p_indirect_buffer_0,
                                       const int64_t* __restrict__ p_indirect_buffer_1,
                                       const int32_t max_position,
                                       const int32_t size_s,
                                       const int32_t size_b,
                                       const int32_t size_h_x,
                                       const int32_t size_h_y,
                                       const int32_t size_d,
                                       const int32_t size_f, // size of last dimension of freqs.
                                       const int32_t stride_ix_s,
                                       const int32_t stride_ix_b,
                                       const int32_t stride_ix_h,
                                       const int32_t stride_ix_d,
                                       const int32_t stride_iy_s,
                                       const int32_t stride_iy_b,
                                       const int32_t stride_iy_h,
                                       const int32_t stride_iy_d,
                                       const int32_t stride_ox_s,
                                       const int32_t stride_ox_b,
                                       const int32_t stride_ox_h,
                                       const int32_t stride_ox_d,
                                       const int32_t stride_oy_s,
                                       const int32_t stride_oy_b,
                                       const int32_t stride_oy_h,
                                       const int32_t stride_oy_d)
{
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    const bool all_stride_d_eq_1 =
        (stride_ix_d == 1) && (stride_iy_d == 1) && (stride_ox_d == 1) && (stride_oy_d == 1);
    auto [grid, block, vec_pairs, threads_per_sb] =
        get_grid_config<RotateStyle, ReuseFreqsFrontPart, false, scalar_t>(
            size_s, 1, size_b, size_f, all_stride_d_eq_1);
    const int32_t total_sb = size_s * size_b;

    LAUNCH_KERNEL_VEC_PAIRS(
        vec_pairs,
        if((p_output_x == p_input_x) && (p_output_y == p_input_y)) {
            assert(stride_ix_s == stride_ox_s);
            assert(stride_ix_b == stride_ox_b);
            assert(stride_ix_h == stride_ox_h);
            assert(stride_ix_d == stride_ox_d);
            assert(stride_iy_s == stride_oy_s);
            assert(stride_iy_b == stride_oy_b);
            assert(stride_iy_h == stride_oy_h);
            assert(stride_iy_d == stride_oy_d);

            LAUNCH_KERNEL_STRIDE_EQUAL_1_2_STRIDES(
                RotateStyle,
                stride_ix_d,
                stride_iy_d,
                kn_entry_2c_sbhd_cached_indirect2_inplace<Op,
                                                          RotateStyle,
                                                          ReuseFreqsFrontPart,
                                                          NopeFirst,
                                                          Stride0Eq1,
                                                          Stride1Eq1,
                                                          VP>
                <<<grid, block, 0, stream>>>(p_output_x,
                                             p_output_y,
                                             p_cos,
                                             p_sin,
                                             p_indirect_buffer_0,
                                             p_indirect_buffer_1,
                                             max_position,
                                             size_h_x,
                                             size_h_y,
                                             size_d,
                                             size_f,
                                             stride_ix_s,
                                             stride_ix_b,
                                             stride_ix_h,
                                             stride_ix_d,
                                             stride_iy_s,
                                             stride_iy_b,
                                             stride_iy_h,
                                             stride_iy_d,
                                             size_s,
                                             threads_per_sb,
                                             total_sb););
        } else {
            LAUNCH_KERNEL_STRIDE_EQUAL_1_4_STRIDES(
                RotateStyle,
                stride_ox_d,
                stride_oy_d,
                stride_ix_d,
                stride_iy_d,
                kn_entry_2c_sbhd_cached_indirect2<Op,
                                                  RotateStyle,
                                                  ReuseFreqsFrontPart,
                                                  NopeFirst,
                                                  Stride0Eq1,
                                                  Stride1Eq1,
                                                  Stride2Eq1,
                                                  Stride3Eq1,
                                                  VP>
                <<<grid, block, 0, stream>>>(p_output_x,
                                             p_output_y,
                                             p_input_x,
                                             p_input_y,
                                             p_cos,
                                             p_sin,
                                             p_indirect_buffer_0,
                                             p_indirect_buffer_1,
                                             max_position,
                                             size_h_x,
                                             size_h_y,
                                             size_d,
                                             size_f,
                                             stride_ix_s,
                                             stride_ix_b,
                                             stride_ix_h,
                                             stride_ix_d,
                                             stride_iy_s,
                                             stride_iy_b,
                                             stride_iy_h,
                                             stride_iy_d,
                                             stride_ox_s,
                                             stride_ox_b,
                                             stride_ox_h,
                                             stride_ox_d,
                                             stride_oy_s,
                                             stride_oy_b,
                                             stride_oy_h,
                                             stride_oy_d,
                                             size_s,
                                             threads_per_sb,
                                             total_sb););
        });
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          bool AllStrideDEq1 = false,
          typename scalar_t,
          typename scalar_f_t>
void dispatch_1c_thd_uncached(scalar_t* __restrict__ p_output,
                              const scalar_t* __restrict__ p_input,
                              const int32_t* __restrict__ p_cu_seqlens,
                              const scalar_f_t* __restrict__ p_freqs,
                              const int32_t size_max_s,
                              const int32_t size_b,
                              const int32_t size_h,
                              const int32_t size_d,
                              const int32_t size_f, // size of last dimension of freqs.
                              const int32_t stride_i_t,
                              const int32_t stride_i_h,
                              const int32_t stride_i_d,
                              const int32_t stride_o_t,
                              const int32_t stride_o_h,
                              const int32_t stride_o_d)
{
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    const bool all_stride_d_eq_1 = (stride_i_d == 1) && (stride_o_d == 1);
    auto [grid, block, vec_pairs, threads_per_sb] =
        get_grid_config<RotateStyle, ReuseFreqsFrontPart, false, scalar_t>(
            size_max_s, 1, size_b, size_f, all_stride_d_eq_1);
    const int32_t total_sb = size_max_s * size_b;

    LAUNCH_KERNEL_VEC_PAIRS(
        vec_pairs,
        if(p_output == p_input) {
            assert(stride_i_t == stride_o_t);
            assert(stride_i_h == stride_o_h);
            assert(stride_i_d == stride_o_d);

            if constexpr(AllStrideDEq1)
            {
                constexpr bool Stride0Eq1 = true;
                kn_entry_1c_thd_uncached_inplace<Op,
                                                 RotateStyle,
                                                 ReuseFreqsFrontPart,
                                                 NopeFirst,
                                                 Stride0Eq1,
                                                 VP><<<grid, block, 0, stream>>>(p_output,
                                                                                 p_cu_seqlens,
                                                                                 p_freqs,
                                                                                 size_h,
                                                                                 size_d,
                                                                                 size_f,
                                                                                 stride_i_t,
                                                                                 stride_i_h,
                                                                                 stride_i_d,
                                                                                 size_max_s,
                                                                                 threads_per_sb,
                                                                                 total_sb);
            }
            else
            {
                LAUNCH_KERNEL_STRIDE_EQUAL_1_1_STRIDES(
                    RotateStyle,
                    stride_i_d,
                    kn_entry_1c_thd_uncached_inplace<Op,
                                                     RotateStyle,
                                                     ReuseFreqsFrontPart,
                                                     NopeFirst,
                                                     Stride0Eq1,
                                                     VP><<<grid, block, 0, stream>>>(p_output,
                                                                                     p_cu_seqlens,
                                                                                     p_freqs,
                                                                                     size_h,
                                                                                     size_d,
                                                                                     size_f,
                                                                                     stride_i_t,
                                                                                     stride_i_h,
                                                                                     stride_i_d,
                                                                                     size_max_s,
                                                                                     threads_per_sb,
                                                                                     total_sb););
            }
        } else {
            if constexpr(AllStrideDEq1)
            {
                constexpr bool Stride0Eq1 = true;
                constexpr bool Stride1Eq1 = true;
                kn_entry_1c_thd_uncached<Op,
                                         RotateStyle,
                                         ReuseFreqsFrontPart,
                                         NopeFirst,
                                         Stride0Eq1,
                                         Stride1Eq1,
                                         VP><<<grid, block, 0, stream>>>(p_output,
                                                                         p_input,
                                                                         p_cu_seqlens,
                                                                         p_freqs,
                                                                         size_h,
                                                                         size_d,
                                                                         size_f,
                                                                         stride_i_t,
                                                                         stride_i_h,
                                                                         stride_i_d,
                                                                         stride_o_t,
                                                                         stride_o_h,
                                                                         stride_o_d,
                                                                         size_max_s,
                                                                         threads_per_sb,
                                                                         total_sb);
            }
            else
            {
                LAUNCH_KERNEL_STRIDE_EQUAL_1_2_STRIDES(RotateStyle,
                                                       stride_o_d,
                                                       stride_i_d,
                                                       kn_entry_1c_thd_uncached<Op,
                                                                                RotateStyle,
                                                                                ReuseFreqsFrontPart,
                                                                                NopeFirst,
                                                                                Stride0Eq1,
                                                                                Stride1Eq1,
                                                                                VP>
                                                       <<<grid, block, 0, stream>>>(p_output,
                                                                                    p_input,
                                                                                    p_cu_seqlens,
                                                                                    p_freqs,
                                                                                    size_h,
                                                                                    size_d,
                                                                                    size_f,
                                                                                    stride_i_t,
                                                                                    stride_i_h,
                                                                                    stride_i_d,
                                                                                    stride_o_t,
                                                                                    stride_o_h,
                                                                                    stride_o_d,
                                                                                    size_max_s,
                                                                                    threads_per_sb,
                                                                                    total_sb););
            }
        });
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          bool AllStrideDEq1 = false,
          typename scalar_t,
          typename scalar_f_t>
void dispatch_1c_2d_cached(scalar_t* __restrict__ p_output,
                           const scalar_t* __restrict__ p_input,
                           const scalar_f_t* __restrict__ p_cos_h,
                           const scalar_f_t* __restrict__ p_sin_h,
                           const scalar_f_t* __restrict__ p_cos_w,
                           const scalar_f_t* __restrict__ p_sin_w,
                           const int img_height,
                           const int img_width,
                           const int32_t size_b,
                           const int32_t size_h,
                           const int32_t size_d,
                           const int32_t stride_i_b,
                           const int32_t stride_i_s,
                           const int32_t stride_i_h,
                           const int32_t stride_i_d,
                           const int32_t stride_o_b,
                           const int32_t stride_o_s,
                           const int32_t stride_o_h,
                           const int32_t stride_o_d)
{
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    const bool all_stride_d_eq_1 = (stride_i_d == 1) && (stride_o_d == 1);
    auto [grid, block, vec_pairs, threads_per_sb] =
        get_grid_config<RotateStyle, ReuseFreqsFrontPart, true, scalar_t>(
            img_height, img_width, size_b, size_d >> 1, all_stride_d_eq_1);
    const int32_t total_sb = img_height * img_width * size_b;

    LAUNCH_KERNEL_VEC_PAIRS(
        vec_pairs,
        if(p_output == p_input) {
            assert(stride_i_s == stride_o_s);
            assert(stride_i_b == stride_o_b);
            assert(stride_i_h == stride_o_h);
            assert(stride_i_d == stride_o_d);

            if constexpr(AllStrideDEq1)
            {
                constexpr bool Stride0Eq1 = true;
                kn_entry_1c_2d_cached_inplace<Op,
                                              RotateStyle,
                                              ReuseFreqsFrontPart,
                                              NopeFirst,
                                              Stride0Eq1,
                                              VP><<<grid, block, 0, stream>>>(p_output,
                                                                              p_cos_h,
                                                                              p_sin_h,
                                                                              p_cos_w,
                                                                              p_sin_w,
                                                                              img_width,
                                                                              size_h,
                                                                              size_d,
                                                                              stride_i_b,
                                                                              stride_i_s,
                                                                              stride_i_h,
                                                                              stride_i_d,
                                                                              img_height,
                                                                              threads_per_sb,
                                                                              total_sb);
            }
            else
            {
                LAUNCH_KERNEL_STRIDE_EQUAL_1_1_STRIDES(
                    RotateStyle,
                    stride_i_d,
                    kn_entry_1c_2d_cached_inplace<Op,
                                                  RotateStyle,
                                                  ReuseFreqsFrontPart,
                                                  NopeFirst,
                                                  Stride0Eq1,
                                                  VP><<<grid, block, 0, stream>>>(p_output,
                                                                                  p_cos_h,
                                                                                  p_sin_h,
                                                                                  p_cos_w,
                                                                                  p_sin_w,
                                                                                  img_width,
                                                                                  size_h,
                                                                                  size_d,
                                                                                  stride_i_b,
                                                                                  stride_i_s,
                                                                                  stride_i_h,
                                                                                  stride_i_d,
                                                                                  img_height,
                                                                                  threads_per_sb,
                                                                                  total_sb););
            }
        } else {
            if constexpr(AllStrideDEq1)
            {
                constexpr bool Stride0Eq1 = true;
                constexpr bool Stride1Eq1 = true;
                kn_entry_1c_2d_cached<Op,
                                      RotateStyle,
                                      ReuseFreqsFrontPart,
                                      NopeFirst,
                                      Stride0Eq1,
                                      Stride1Eq1,
                                      VP><<<grid, block, 0, stream>>>(p_output,
                                                                      p_input,
                                                                      p_cos_h,
                                                                      p_sin_h,
                                                                      p_cos_w,
                                                                      p_sin_w,
                                                                      img_width,
                                                                      size_h,
                                                                      size_d,
                                                                      stride_i_b,
                                                                      stride_i_s,
                                                                      stride_i_h,
                                                                      stride_i_d,
                                                                      stride_o_b,
                                                                      stride_o_s,
                                                                      stride_o_h,
                                                                      stride_o_d,
                                                                      img_height,
                                                                      threads_per_sb,
                                                                      total_sb);
            }
            else
            {
                LAUNCH_KERNEL_STRIDE_EQUAL_1_2_STRIDES(RotateStyle,
                                                       stride_o_d,
                                                       stride_i_d,
                                                       kn_entry_1c_2d_cached<Op,
                                                                             RotateStyle,
                                                                             ReuseFreqsFrontPart,
                                                                             NopeFirst,
                                                                             Stride0Eq1,
                                                                             Stride1Eq1,
                                                                             VP>
                                                       <<<grid, block, 0, stream>>>(p_output,
                                                                                    p_input,
                                                                                    p_cos_h,
                                                                                    p_sin_h,
                                                                                    p_cos_w,
                                                                                    p_sin_w,
                                                                                    img_width,
                                                                                    size_h,
                                                                                    size_d,
                                                                                    stride_i_b,
                                                                                    stride_i_s,
                                                                                    stride_i_h,
                                                                                    stride_i_d,
                                                                                    stride_o_b,
                                                                                    stride_o_s,
                                                                                    stride_o_h,
                                                                                    stride_o_d,
                                                                                    img_height,
                                                                                    threads_per_sb,
                                                                                    total_sb););
            }
        });
}
} // namespace aiter

#define DISPATCH_ROPE_TYPES_PARAMS(                                               \
    TYPE0, TYPE1, ROTATE_STYLE, REUSE_FREQS_FRONT_PART, NOPE_FIRST, NAME, ...)    \
    switch((TYPE0))                                                               \
    {                                                                             \
    case at::ScalarType::Float: {                                                 \
        using scalar_t_0 = float;                                                 \
        switch((TYPE1))                                                           \
        {                                                                         \
        case at::ScalarType::Float: {                                             \
            using scalar_t_1 = float;                                             \
            if((REUSE_FREQS_FRONT_PART))                                          \
            {                                                                     \
                constexpr bool ReuseFreqsFrontPart = true;                        \
                if((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                           \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else if((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                      \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else                                                              \
                {                                                                 \
                    TORCH_CHECK(false,                                            \
                                NAME " does't support rotate type ",              \
                                std::to_string((ROTATE_STYLE)),                   \
                                ".");                                             \
                }                                                                 \
            }                                                                     \
            else                                                                  \
            {                                                                     \
                constexpr bool ReuseFreqsFrontPart = false;                       \
                if((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                           \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else if((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                      \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else                                                              \
                {                                                                 \
                    TORCH_CHECK(false,                                            \
                                NAME " does't support rotate type ",              \
                                std::to_string((ROTATE_STYLE)),                   \
                                ".");                                             \
                }                                                                 \
            }                                                                     \
            break;                                                                \
        }                                                                         \
        case at::ScalarType::Half: {                                              \
            using scalar_t_1 = at::Half;                                          \
            if((REUSE_FREQS_FRONT_PART))                                          \
            {                                                                     \
                constexpr bool ReuseFreqsFrontPart = true;                        \
                if((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                           \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else if((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                      \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else                                                              \
                {                                                                 \
                    TORCH_CHECK(false,                                            \
                                NAME " does't support rotate type ",              \
                                std::to_string((ROTATE_STYLE)),                   \
                                ".");                                             \
                }                                                                 \
            }                                                                     \
            else                                                                  \
            {                                                                     \
                constexpr bool ReuseFreqsFrontPart = false;                       \
                if((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                           \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else if((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                      \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else                                                              \
                {                                                                 \
                    TORCH_CHECK(false,                                            \
                                NAME " does't support rotate type ",              \
                                std::to_string((ROTATE_STYLE)),                   \
                                ".");                                             \
                }                                                                 \
            }                                                                     \
            break;                                                                \
        }                                                                         \
        case at::ScalarType::BFloat16: {                                          \
            using scalar_t_1 = at::BFloat16;                                      \
            if((REUSE_FREQS_FRONT_PART))                                          \
            {                                                                     \
                constexpr bool ReuseFreqsFrontPart = true;                        \
                if((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                           \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else if((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                      \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else                                                              \
                {                                                                 \
                    TORCH_CHECK(false,                                            \
                                NAME " does't support rotate type ",              \
                                std::to_string((ROTATE_STYLE)),                   \
                                ".");                                             \
                }                                                                 \
            }                                                                     \
            else                                                                  \
            {                                                                     \
                constexpr bool ReuseFreqsFrontPart = false;                       \
                if((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                           \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else if((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                      \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else                                                              \
                {                                                                 \
                    TORCH_CHECK(false,                                            \
                                NAME " does't support rotate type ",              \
                                std::to_string((ROTATE_STYLE)),                   \
                                ".");                                             \
                }                                                                 \
            }                                                                     \
            break;                                                                \
        }                                                                         \
        default:                                                                  \
            TORCH_CHECK(false,                                                    \
                        NAME " does't support ",                                  \
                        toString((TYPE0)),                                        \
                        " with ",                                                 \
                        toString((TYPE1)),                                        \
                        ".");                                                     \
        }                                                                         \
        break;                                                                    \
    }                                                                             \
    case at::ScalarType::Half: {                                                  \
        using scalar_t_0 = at::Half;                                              \
        switch((TYPE1))                                                           \
        {                                                                         \
        case at::ScalarType::Float: {                                             \
            using scalar_t_1 = float;                                             \
            if((REUSE_FREQS_FRONT_PART))                                          \
            {                                                                     \
                constexpr bool ReuseFreqsFrontPart = true;                        \
                if((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                           \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else if((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                      \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else                                                              \
                {                                                                 \
                    TORCH_CHECK(false,                                            \
                                NAME " does't support rotate type ",              \
                                std::to_string((ROTATE_STYLE)),                   \
                                ".");                                             \
                }                                                                 \
            }                                                                     \
            else                                                                  \
            {                                                                     \
                constexpr bool ReuseFreqsFrontPart = false;                       \
                if((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                           \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else if((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                      \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else                                                              \
                {                                                                 \
                    TORCH_CHECK(false,                                            \
                                NAME " does't support rotate type ",              \
                                std::to_string((ROTATE_STYLE)),                   \
                                ".");                                             \
                }                                                                 \
            }                                                                     \
            break;                                                                \
        }                                                                         \
        case at::ScalarType::Half: {                                              \
            using scalar_t_1 = at::Half;                                          \
            if((REUSE_FREQS_FRONT_PART))                                          \
            {                                                                     \
                constexpr bool ReuseFreqsFrontPart = true;                        \
                if((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                           \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else if((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                      \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else                                                              \
                {                                                                 \
                    TORCH_CHECK(false,                                            \
                                NAME " does't support rotate type ",              \
                                std::to_string((ROTATE_STYLE)),                   \
                                ".");                                             \
                }                                                                 \
            }                                                                     \
            else                                                                  \
            {                                                                     \
                constexpr bool ReuseFreqsFrontPart = false;                       \
                if((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                           \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else if((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                      \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else                                                              \
                {                                                                 \
                    TORCH_CHECK(false,                                            \
                                NAME " does't support rotate type ",              \
                                std::to_string((ROTATE_STYLE)),                   \
                                ".");                                             \
                }                                                                 \
            }                                                                     \
            break;                                                                \
        }                                                                         \
        case at::ScalarType::BFloat16: {                                          \
            using scalar_t_1 = at::BFloat16;                                      \
            if((REUSE_FREQS_FRONT_PART))                                          \
            {                                                                     \
                constexpr bool ReuseFreqsFrontPart = true;                        \
                if((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                           \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else if((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                      \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else                                                              \
                {                                                                 \
                    TORCH_CHECK(false,                                            \
                                NAME " does't support rotate type ",              \
                                std::to_string((ROTATE_STYLE)),                   \
                                ".");                                             \
                }                                                                 \
            }                                                                     \
            else                                                                  \
            {                                                                     \
                constexpr bool ReuseFreqsFrontPart = false;                       \
                if((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                           \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else if((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                      \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else                                                              \
                {                                                                 \
                    TORCH_CHECK(false,                                            \
                                NAME " does't support rotate type ",              \
                                std::to_string((ROTATE_STYLE)),                   \
                                ".");                                             \
                }                                                                 \
            }                                                                     \
            break;                                                                \
        }                                                                         \
        default:                                                                  \
            TORCH_CHECK(false,                                                    \
                        NAME " does't support ",                                  \
                        toString((TYPE0)),                                        \
                        " with ",                                                 \
                        toString((TYPE1)),                                        \
                        ".");                                                     \
        }                                                                         \
        break;                                                                    \
    }                                                                             \
    case at::ScalarType::BFloat16: {                                              \
        using scalar_t_0 = at::BFloat16;                                          \
        switch((TYPE1))                                                           \
        {                                                                         \
        case at::ScalarType::Float: {                                             \
            using scalar_t_1 = float;                                             \
            if((REUSE_FREQS_FRONT_PART))                                          \
            {                                                                     \
                constexpr bool ReuseFreqsFrontPart = true;                        \
                if((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                           \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else if((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                      \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else                                                              \
                {                                                                 \
                    TORCH_CHECK(false,                                            \
                                NAME " does't support rotate type ",              \
                                std::to_string((ROTATE_STYLE)),                   \
                                ".");                                             \
                }                                                                 \
            }                                                                     \
            else                                                                  \
            {                                                                     \
                constexpr bool ReuseFreqsFrontPart = false;                       \
                if((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                           \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else if((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                      \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else                                                              \
                {                                                                 \
                    TORCH_CHECK(false,                                            \
                                NAME " does't support rotate type ",              \
                                std::to_string((ROTATE_STYLE)),                   \
                                ".");                                             \
                }                                                                 \
            }                                                                     \
            break;                                                                \
        }                                                                         \
        case at::ScalarType::Half: {                                              \
            using scalar_t_1 = at::Half;                                          \
            if((REUSE_FREQS_FRONT_PART))                                          \
            {                                                                     \
                constexpr bool ReuseFreqsFrontPart = true;                        \
                if((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                           \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else if((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                      \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else                                                              \
                {                                                                 \
                    TORCH_CHECK(false,                                            \
                                NAME " does't support rotate type ",              \
                                std::to_string((ROTATE_STYLE)),                   \
                                ".");                                             \
                }                                                                 \
            }                                                                     \
            else                                                                  \
            {                                                                     \
                constexpr bool ReuseFreqsFrontPart = false;                       \
                if((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                           \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else if((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                      \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else                                                              \
                {                                                                 \
                    TORCH_CHECK(false,                                            \
                                NAME " does't support rotate type ",              \
                                std::to_string((ROTATE_STYLE)),                   \
                                ".");                                             \
                }                                                                 \
            }                                                                     \
            break;                                                                \
        }                                                                         \
        case at::ScalarType::BFloat16: {                                          \
            using scalar_t_1 = at::BFloat16;                                      \
            if((REUSE_FREQS_FRONT_PART))                                          \
            {                                                                     \
                constexpr bool ReuseFreqsFrontPart = true;                        \
                if((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                           \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else if((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                      \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else                                                              \
                {                                                                 \
                    TORCH_CHECK(false,                                            \
                                NAME " does't support rotate type ",              \
                                std::to_string((ROTATE_STYLE)),                   \
                                ".");                                             \
                }                                                                 \
            }                                                                     \
            else                                                                  \
            {                                                                     \
                constexpr bool ReuseFreqsFrontPart = false;                       \
                if((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                           \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else if((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                      \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else                                                              \
                {                                                                 \
                    TORCH_CHECK(false,                                            \
                                NAME " does't support rotate type ",              \
                                std::to_string((ROTATE_STYLE)),                   \
                                ".");                                             \
                }                                                                 \
            }                                                                     \
            break;                                                                \
        }                                                                         \
        default:                                                                  \
            TORCH_CHECK(false,                                                    \
                        NAME " does't support ",                                  \
                        toString((TYPE0)),                                        \
                        " with ",                                                 \
                        toString((TYPE1)),                                        \
                        ".");                                                     \
        }                                                                         \
        break;                                                                    \
    }                                                                             \
    default: TORCH_CHECK(false, NAME " does't support ", toString((TYPE0)), "."); \
    }

namespace mrope_utils {

static constexpr int kBytesPerAccess = 16;
static constexpr int WARP_SIZE       = 32;

namespace block_utils {

template <typename T>
__inline__ __device__ T warp_shfl_xor_sync(T val, int offset)
{
    return __shfl_xor(val, offset, 32);
}

template <typename T>
__inline__ __device__ T warp_reduce_sum(T val)
{
#pragma unroll
    for(int offset = 16; offset > 0; offset >>= 1)
        val += warp_shfl_xor_sync(val, offset);
    return val;
}

template <typename T>
__inline__ __device__ T warp_shfl_sync(T val, int src_id)
{
    return __shfl(val, src_id, 32);
}

} // namespace block_utils

template <typename T, int vec_size>
struct alignas(sizeof(T) * vec_size) vec_t
{
    T data[vec_size];
    __device__ __forceinline__ T& operator[](int i) { return data[i]; }
    __device__ __forceinline__ T const& operator[](int i) const { return data[i]; }
    __device__ __forceinline__ void load(const T* ptr)
    {
        *this = *reinterpret_cast<vec_t<T, vec_size>*>(const_cast<T*>(ptr));
    }
    __device__ __forceinline__ void loop_load(const T* ptr)
    {
#pragma unroll
        for(int i = 0; i < vec_size; ++i)
        {
            data[i] = ptr[i];
        }
    }
    __device__ __forceinline__ void store(T* ptr)
    {
        *reinterpret_cast<vec_t<T, vec_size>*>(ptr) = *this;
    }
    __device__ __forceinline__ void loop_store(T* ptr)
    {
#pragma unroll
        for(int i = 0; i < vec_size; ++i)
        {
            ptr[i] = data[i];
        }
    }
    __device__ __forceinline__ void nontemporal_load(const T* ptr)
    {
        constexpr int ITERS = vec_size * sizeof(T) / sizeof(uint32_t);
#pragma unroll
        for(int i = 0; i < ITERS; ++i)
        {
            reinterpret_cast<uint32_t*>(&data)[i] = __builtin_nontemporal_load((uint32_t*)ptr + i);
        }
    }
    __device__ __forceinline__ void nontemporal_store(T* ptr)
    {
        constexpr int ITERS = vec_size * sizeof(T) / sizeof(uint32_t);
#pragma unroll
        for(int i = 0; i < ITERS; ++i)
        {
            __builtin_nontemporal_store(reinterpret_cast<uint32_t*>(&data)[i], (uint32_t*)ptr + i);
        }
    }
    __device__ __forceinline__ void fill(T val)
    {
#pragma unroll
        for(int i = 0; i < vec_size; ++i)
        {
            data[i] = val;
        }
    }
    __device__ __forceinline__ void copy_(const vec_t<T, vec_size>& other)
    {
#pragma unroll
        for(int i = 0; i < vec_size; ++i)
        {
            data[i] = other.data[i];
        }
    }
    template <typename IT>
    __device__ __forceinline__ void from_(const vec_t<IT, vec_size>& src, float scale)
    {
#pragma unroll
        for(int i = 0; i < vec_size; ++i)
        {
            if constexpr(std::is_same_v<T, IT>)
            {
                data[i] = src[i];
            }
            else
            {
                data[i] = ck_tile::type_convert<T>(ck_tile::type_convert<float>(src[i]) / scale);
            }
        }
    }
};

template <typename T, int vec_size>
__inline__ __device__ vec_t<T, vec_size> warp_shfl_sync_vec(vec_t<T, vec_size>& val, int offset)
{
    constexpr int ITERS = vec_size * sizeof(T) / sizeof(uint32_t);
    vec_t<T, vec_size> out;
#pragma unroll
    for(int i = 0; i < ITERS; ++i)
    {
        uint32_t val_                        = reinterpret_cast<uint32_t*>(&val)[i];
        reinterpret_cast<uint32_t*>(&out)[i] = block_utils::warp_shfl_sync<uint32_t>(val_, offset);
    }
    return out;
}

template <typename T, int VEC_SIZE>
__device__ __forceinline__ void
warp_rms_norm_(vec_t<T, VEC_SIZE>& input, vec_t<T, VEC_SIZE>& gamma, float rms_dim, float rms_eps)
{
    vec_t<T, VEC_SIZE> norm_out;
    float acc = 0.f;
#pragma unroll
    for(int i = 0; i < VEC_SIZE; ++i)
    {
        float v = (float)input[i];
        acc += v * v;
    }
    int warp_id   = threadIdx.x / 32;
    int warp_t_id = threadIdx.x % 32;
    acc           = block_utils::warp_reduce_sum<float>(acc);
    acc           = block_utils::warp_shfl_sync<float>(acc, 0);
    auto s_val    = rsqrtf(acc / rms_dim + rms_eps);
#pragma unroll
    for(int i = 0; i < VEC_SIZE; ++i)
    {
        input[i] = static_cast<T>((float)input[i] * s_val * (float)gamma[i]);
    }
}

template <typename T, int VEC_SIZE, int HEAD_SIZE, bool IS_INTERLEAVED, int M>
__device__ __forceinline__ void mrope_load_cos_sin_vec(vec_t<T, VEC_SIZE>& out,
                                                       const T* cos_sin,
                                                       const int64_t* positions,
                                                       int64_t ps0,
                                                       int64_t ps1,
                                                       int64_t token_id,
                                                       int64_t num_tokens,
                                                       int access_id_in_head,
                                                       std::array<int64_t, M>& mrope_section,
                                                       int rotary_dim = 0)
{
    const int rd   = rotary_dim > 0 ? rotary_dim : HEAD_SIZE;
    const int half = rd / 2;
    if constexpr(IS_INTERLEAVED)
    {
        for(int i = 0; i < VEC_SIZE; ++i)
        {
            auto id   = access_id_in_head + i;
            auto id_  = (access_id_in_head < half) ? id : id - half;
            auto mid_ = id_ % M;
            if(mid_ >= 1 && id_ < mrope_section[mid_] * M)
            {
                auto p = positions[mid_ * ps0 + token_id * ps1];
                out[i] = cos_sin[p * rd + id];
            }
            else
            {
                out[i] = cos_sin[positions[token_id * ps1] * rd + id];
            }
        }
    }
    else
    {
        for(int i = 0; i < VEC_SIZE; ++i)
        {
            auto id  = access_id_in_head + i;
            auto id_ = (access_id_in_head < half) ? id : id - half;
            int mid;
            int end = 0;
            for(mid = 0; mid < M; ++mid)
            {
                end += mrope_section[mid];
                if(id_ < end)
                    break;
            }
            auto p = positions[mid * ps0 + token_id * ps1];
            out[i] = cos_sin[p * rd + id];
        }
    }
}

struct alignas(1) fp8e4m3fn
{
    struct from_bits_t
    {
    };
    __host__ __device__ static constexpr from_bits_t from_bits() { return from_bits_t(); }
    uint8_t data;

    fp8e4m3fn()                                               = default;
    __host__ __device__ constexpr fp8e4m3fn(const fp8e4m3fn&) = default;
    __host__ __device__ constexpr fp8e4m3fn(uint8_t v)        = delete;
    explicit __host__ __device__ constexpr fp8e4m3fn(uint8_t v, from_bits_t) : data(v) {}

    explicit __host__ __device__ fp8e4m3fn(float v)
    {
        data = hip_fp8_impl::to_float8<4, 3, float, false /*negative_zero_nan*/, true /*clip*/>(v);
    }

    explicit __host__ __device__ fp8e4m3fn(double v) : fp8e4m3fn(static_cast<float>(v)) {}

    explicit inline __host__ __device__ operator float() const
    {
        return hip_fp8_impl::from_float8<4, 3, float, false /*negative_zero_nan*/>(data);
    }
};

struct alignas(1) fp8e4m3fnuz
{
    struct from_bits_t
    {
    };
    __host__ __device__ static constexpr from_bits_t from_bits() { return from_bits_t(); }
    uint8_t data;

    fp8e4m3fnuz()                                                 = default;
    __host__ __device__ constexpr fp8e4m3fnuz(const fp8e4m3fnuz&) = default;
    __host__ __device__ constexpr fp8e4m3fnuz(uint8_t v)          = delete;
    explicit __host__ __device__ constexpr fp8e4m3fnuz(uint8_t v, from_bits_t) : data(v) {}

    explicit __host__ __device__ fp8e4m3fnuz(float v)
    {
        data = hip_fp8_impl::to_float8<4, 3, float, true /*negative_zero_nan*/, true /*clip*/>(v);
    }

    explicit __host__ __device__ fp8e4m3fnuz(double v) : fp8e4m3fnuz(static_cast<float>(v)) {}

    explicit inline __host__ __device__ operator float() const
    {
        return hip_fp8_impl::from_float8<4, 3, float, true /*negative_zero_nan*/>(data);
    }
};

template <int HEAD_SIZE>
__device__ __forceinline__ int64_t get_shuffle_layout_k_base(const int64_t slot_id,
                                                             const int block_size,
                                                             const int num_heads_k,
                                                             const int head_id_k,
                                                             const int access_id_in_head,
                                                             const int x)
{
    // Shuffle layout: [num_blocks, num_kv_heads, head_size // x, block_size, x]
    const int block_id      = static_cast<int>(slot_id / block_size);
    const int block_offset  = static_cast<int>(slot_id % block_size);
    const int k_head_stride = HEAD_SIZE * block_size;
    const int64_t dst_base =
        static_cast<int64_t>(block_id) * num_heads_k * k_head_stride + head_id_k * k_head_stride;
    // Pre-compute K base offset: since VEC_SIZE <= x, all elements are in the same
    // chunk
    const int chunk_id     = access_id_in_head / x;
    const int block_size_x = block_size * x;
    const int64_t k_base =
        dst_base + chunk_id * block_size_x + block_offset * x + (access_id_in_head % x);
    return k_base;
}

template <int HEAD_SIZE>
__device__ __forceinline__ int64_t get_shuffle_layout_v_base(const int64_t slot_id,
                                                             const int block_size,
                                                             const int num_heads_v,
                                                             const int head_id_v,
                                                             const int access_id_in_head,
                                                             const int x)
{
    // Shuffle layout: [num_blocks, num_kv_heads, block_size // x, head_size, x]
    const int block_id      = static_cast<int>(slot_id / block_size);
    const int block_offset  = static_cast<int>(slot_id % block_size);
    const int v_head_stride = (block_size / x) * HEAD_SIZE * x;
    const int64_t dst_base =
        static_cast<int64_t>(block_id) * num_heads_v * v_head_stride + head_id_v * v_head_stride;
    // Pre-compute V base offset (fixed for this token)
    const int v_slot_chunk    = block_offset / x;
    const int v_slot_in_chunk = block_offset % x;
    const int64_t v_base      = dst_base + v_slot_chunk * HEAD_SIZE * x + v_slot_in_chunk;
    return v_base;
}

template <typename T,
          int HEAD_SIZE,
          bool IS_NEOX,
          bool IS_MROPE,
          bool IS_INTERLEAVED,
          int M,
          typename KVT>
__global__ void fused_mrope_rms_kv_kernel(const T* qkv,
                                          const T* q_w,
                                          const T* k_w,
                                          const T* cos_sin,
                                          const int64_t* positions,
                                          int64_t positions_stride_0,
                                          int64_t positions_stride_1,
                                          int num_heads_q,
                                          int num_heads_k,
                                          int num_heads_v,
                                          double eps,
                                          std::array<int64_t, M> mrope_section,
                                          int num_tokens,
                                          int total_warps,
                                          T* q_out,
                                          KVT* k_cache,
                                          KVT* v_cache,
                                          const int64_t* slot_mapping,
                                          float per_tensor_k_scale = 1.0,
                                          float per_tensor_v_scale = 1.0,
                                          KVT* k_out               = nullptr,
                                          KVT* v_out               = nullptr,
                                          bool use_shuffle_layout  = false,
                                          int block_size           = 0,
                                          int x                    = 0,
                                          int rotary_dim           = 0)
{
    constexpr int VEC_SIZE        = HEAD_SIZE / WARP_SIZE;
    constexpr int HALF_HEAD_SIZE  = HEAD_SIZE / 2;
    const int warp_id             = threadIdx.x / WARP_SIZE;
    const int num_warps_per_block = blockDim.x / WARP_SIZE;
    const int global_warp_id      = blockIdx.x * num_warps_per_block + warp_id;

    if(global_warp_id >= total_warps)
        return;

    // Warp allocation: all Q first, then all K, then all V
    const int num_heads_qk   = num_heads_q + num_heads_k;
    const int num_heads      = num_heads_q + num_heads_k + num_heads_v;
    const int total_q_warps  = num_tokens * num_heads_q;
    const int total_k_warps  = num_tokens * num_heads_k;
    const int total_qk_warps = total_q_warps + total_k_warps;

    // Determine if current warp processes Q, K, or V
    const bool is_q = global_warp_id < total_q_warps;
    const bool is_k = !is_q && global_warp_id < total_qk_warps;
    const bool is_v = global_warp_id >= total_qk_warps;

    int token_id, head_id_in_token;

    if(is_q)
    {
        // Q warps: global_warp_id in range [0, total_q_warps)
        token_id         = global_warp_id / num_heads_q;
        head_id_in_token = global_warp_id % num_heads_q;
    }
    else if(is_k)
    {
        // K warps: global_warp_id in range [total_q_warps, total_qk_warps)
        const int k_warp_id = global_warp_id - total_q_warps;
        token_id            = k_warp_id / num_heads_k;
        head_id_in_token    = num_heads_q + (k_warp_id % num_heads_k);
    }
    else
    {
        // V warps: global_warp_id in range [total_qk_warps, total_warps)
        const int v_warp_id = global_warp_id - total_qk_warps;
        token_id            = v_warp_id / num_heads_v;
        head_id_in_token    = num_heads_qk + (v_warp_id % num_heads_v);
    }

    const int access_id_in_head = (threadIdx.x % WARP_SIZE) * VEC_SIZE;
    const int neighbor_offset =
        access_id_in_head < HALF_HEAD_SIZE ? HALF_HEAD_SIZE / VEC_SIZE : -HALF_HEAD_SIZE / VEC_SIZE;
    const T* qkv_ =
        &qkv[(static_cast<int64_t>(token_id) * num_heads + head_id_in_token) * HEAD_SIZE];

    if(!is_v)
    {
        vec_t<T, VEC_SIZE> w_vec;
        if(is_q)
        {
            w_vec.load(q_w + access_id_in_head);
        }
        else
        {
            w_vec.load(k_w + access_id_in_head);
        }
        vec_t<T, VEC_SIZE> x_vec;
        x_vec.load(qkv_ + access_id_in_head);
        vec_t<T, VEC_SIZE> out_vec;
        const int rotary_dim_ = rotary_dim > 0 ? rotary_dim : HEAD_SIZE;
        const int half_rotary = rotary_dim_ / 2;
        const bool in_rotary  = access_id_in_head < rotary_dim_;
        if constexpr(IS_NEOX)
        {
            vec_t<T, VEC_SIZE> cos_sin_vec;
            if constexpr(IS_MROPE)
            {
                if(in_rotary)
                {
                    mrope_load_cos_sin_vec<T, VEC_SIZE, HEAD_SIZE, IS_INTERLEAVED, M>(
                        cos_sin_vec,
                        cos_sin,
                        positions,
                        positions_stride_0,
                        positions_stride_1,
                        token_id,
                        num_tokens,
                        access_id_in_head,
                        mrope_section,
                        rotary_dim_);
                }
            }
            else
            {
                auto position_ = positions[token_id * positions_stride_1];
                if(in_rotary)
                {
                    cos_sin_vec.load(&cos_sin[position_ * rotary_dim_ + access_id_in_head]);
                }
            }
            warp_rms_norm_<T, VEC_SIZE>(x_vec, w_vec, HEAD_SIZE, eps);
            if(in_rotary)
            {
                const int rotary_neighbor_offset = access_id_in_head < half_rotary
                                                       ? half_rotary / VEC_SIZE
                                                       : -(half_rotary / VEC_SIZE);
                auto nb_cos_sin_vec              = warp_shfl_sync_vec<T, VEC_SIZE>(
                    cos_sin_vec, threadIdx.x + rotary_neighbor_offset);
                auto nb_x_vec =
                    warp_shfl_sync_vec<T, VEC_SIZE>(x_vec, threadIdx.x + rotary_neighbor_offset);
                if(access_id_in_head < half_rotary)
                {
#pragma unroll
                    for(int i = 0; i < VEC_SIZE; ++i)
                    {
                        out_vec[i] =
                            (float)x_vec[i] * (float)cos_sin_vec[i] -
                            (float)nb_x_vec[i] * (float)nb_cos_sin_vec[i]; // x0 * cos - x1 * sin
                    }
                }
                else
                {
#pragma unroll
                    for(int i = 0; i < VEC_SIZE; ++i)
                    {
                        out_vec[i] =
                            (float)x_vec[i] * (float)nb_cos_sin_vec[i] +
                            (float)nb_x_vec[i] * (float)cos_sin_vec[i]; // x1 * cos + x0 * sin
                    }
                }
            }
            else
            {
#pragma unroll
                for(int i = 0; i < VEC_SIZE; ++i)
                    out_vec[i] = x_vec[i];
            }
        }
        else
        {
            vec_t<T, VEC_SIZE> cos_vec, sin_vec;
            if constexpr(IS_MROPE)
            {
                if(in_rotary)
                {
                    mrope_load_cos_sin_vec<T, VEC_SIZE, HEAD_SIZE, IS_INTERLEAVED, M>(
                        cos_vec,
                        cos_sin,
                        positions,
                        positions_stride_0,
                        positions_stride_1,
                        token_id,
                        num_tokens,
                        access_id_in_head / 2,
                        mrope_section,
                        rotary_dim_);
                    mrope_load_cos_sin_vec<T, VEC_SIZE, HEAD_SIZE, IS_INTERLEAVED, M>(
                        sin_vec,
                        cos_sin,
                        positions,
                        positions_stride_0,
                        positions_stride_1,
                        token_id,
                        num_tokens,
                        access_id_in_head / 2 + half_rotary,
                        mrope_section,
                        rotary_dim_);
                }
            }
            else
            {
                auto position_ = positions[token_id * positions_stride_1];
                if(in_rotary)
                {
                    cos_vec.load(&cos_sin[position_ * rotary_dim_ + access_id_in_head / 2]);
                    sin_vec.load(
                        &cos_sin[position_ * rotary_dim_ + access_id_in_head / 2 + half_rotary]);
                }
            }
            warp_rms_norm_<T, VEC_SIZE>(x_vec, w_vec, HEAD_SIZE, eps);
            if(in_rotary)
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
            else
            {
#pragma unroll
                for(int i = 0; i < VEC_SIZE; ++i)
                    out_vec[i] = x_vec[i];
            }
        }

        if(is_q)
        {
            T* q_ = &q_out[(static_cast<int64_t>(token_id) * num_heads_q + head_id_in_token) *
                           HEAD_SIZE];
            out_vec.store(q_ + access_id_in_head);
        }
        else
        {
            vec_t<KVT, VEC_SIZE> out_kv_vec;
            out_kv_vec.from_(out_vec, per_tensor_k_scale);
            const int64_t slot_id = slot_mapping[token_id];
            if(slot_id < 0)
                return;
            const int head_id_k = head_id_in_token - num_heads_q;
            if(use_shuffle_layout)
            {
                int64_t k_base = get_shuffle_layout_k_base<HEAD_SIZE>(
                    slot_id, block_size, num_heads_k, head_id_k, access_id_in_head, x);
                out_kv_vec.store(k_cache + k_base);
            }
            else
            {
                const int64_t offset =
                    (slot_id * num_heads_k + head_id_k) * HEAD_SIZE + access_id_in_head;
                out_kv_vec.store(k_cache + offset);
            }
            if(k_out != nullptr)
            {
                const int64_t k_out_offset =
                    (static_cast<int64_t>(token_id) * num_heads_k + head_id_k) * HEAD_SIZE +
                    access_id_in_head;
                out_kv_vec.store(k_out + k_out_offset);
            }
        }
    }
    else
    {
        vec_t<T, VEC_SIZE> out_vec;
        vec_t<KVT, VEC_SIZE> out_kv_vec;
        out_vec.load(qkv_ + access_id_in_head);
        out_kv_vec.from_(out_vec, per_tensor_v_scale);
        const int64_t slot_id = slot_mapping[token_id];
        if(slot_id < 0)
            return;
        const int head_id_v = head_id_in_token - num_heads_qk;
        if(use_shuffle_layout)
        {
            int64_t v_base = get_shuffle_layout_v_base<HEAD_SIZE>(
                slot_id, block_size, num_heads_v, head_id_v, access_id_in_head, x);
#pragma unroll
            for(int i = 0; i < VEC_SIZE; ++i)
            {
                const int offset_in_head             = access_id_in_head + i;
                v_cache[v_base + offset_in_head * x] = out_kv_vec[i];
            }
        }
        else
        {
            const int64_t offset =
                (slot_id * num_heads_v + head_id_v) * HEAD_SIZE + access_id_in_head;
            out_kv_vec.store(v_cache + offset);
        }
        if(v_out != nullptr)
        {
            const int64_t v_out_offset =
                (static_cast<int64_t>(token_id) * num_heads_v + head_id_v) * HEAD_SIZE +
                access_id_in_head;
            out_kv_vec.store(v_out + v_out_offset);
        }
    }
}

template <typename T, int M, typename KVT>
void fused_mrope_rms_set_kv(const T* qkv,
                            const T* q_w,
                            const T* k_w,
                            const T* cos_sin,
                            const int64_t* positions,
                            int64_t positions_stride_0,
                            int64_t positions_stride_1,
                            int64_t num_tokens,
                            int64_t num_heads_q,
                            int64_t num_heads_k,
                            int64_t num_heads_v,
                            int64_t head_size,
                            bool is_neox_style,
                            double eps,
                            std::array<int64_t, M> mrope_section,
                            bool is_interleaved,
                            T* q_out,
                            KVT* k_cache,
                            KVT* v_cache,
                            const int64_t* slot_mapping,
                            hipStream_t stream,
                            float per_tensor_k_scale = 1.0,
                            float per_tensor_v_scale = 1.0,
                            KVT* k_out               = nullptr,
                            KVT* v_out               = nullptr,
                            bool use_shuffle_layout  = false,
                            int64_t block_size       = 0,
                            int64_t x                = 0,
                            int64_t rotary_dim       = 0)
{
    TORCH_CHECK(head_size == 64 || head_size == 128 || head_size == 256);
    auto dim           = std::accumulate(mrope_section.begin(), mrope_section.end(), 0);
    auto expected_half = rotary_dim > 0 ? rotary_dim / 2 : head_size / 2;
    TORCH_CHECK(dim == expected_half,
                "mrope_section sum (",
                dim,
                ") must equal rotary_dim/2 (",
                expected_half,
                ")");
    constexpr int THREAD_BLOCK_SIZE = 256;
    auto total_warps                = num_tokens * (num_heads_q + num_heads_k + num_heads_v);
    auto num_warps_per_block        = THREAD_BLOCK_SIZE / WARP_SIZE;
    dim3 threadsPerBlock(THREAD_BLOCK_SIZE);
    dim3 numBlocks((total_warps + num_warps_per_block - 1) / num_warps_per_block);

#define DISPATCH_NEOX(HEAD_SIZE, IS_INTERLEAVED)                                     \
    if(is_neox_style)                                                                \
    {                                                                                \
        fused_mrope_rms_kv_kernel<T, HEAD_SIZE, true, true, IS_INTERLEAVED, M, KVT>  \
            <<<numBlocks, threadsPerBlock, 0, stream>>>(qkv,                         \
                                                        q_w,                         \
                                                        k_w,                         \
                                                        cos_sin,                     \
                                                        positions,                   \
                                                        positions_stride_0,          \
                                                        positions_stride_1,          \
                                                        num_heads_q,                 \
                                                        num_heads_k,                 \
                                                        num_heads_v,                 \
                                                        eps,                         \
                                                        mrope_section,               \
                                                        num_tokens,                  \
                                                        total_warps,                 \
                                                        q_out,                       \
                                                        k_cache,                     \
                                                        v_cache,                     \
                                                        slot_mapping,                \
                                                        per_tensor_k_scale,          \
                                                        per_tensor_v_scale,          \
                                                        k_out,                       \
                                                        v_out,                       \
                                                        use_shuffle_layout,          \
                                                        block_size,                  \
                                                        x,                           \
                                                        (int)rotary_dim);            \
    }                                                                                \
    else                                                                             \
    {                                                                                \
        fused_mrope_rms_kv_kernel<T, HEAD_SIZE, false, true, IS_INTERLEAVED, M, KVT> \
            <<<numBlocks, threadsPerBlock, 0, stream>>>(qkv,                         \
                                                        q_w,                         \
                                                        k_w,                         \
                                                        cos_sin,                     \
                                                        positions,                   \
                                                        positions_stride_0,          \
                                                        positions_stride_1,          \
                                                        num_heads_q,                 \
                                                        num_heads_k,                 \
                                                        num_heads_v,                 \
                                                        eps,                         \
                                                        mrope_section,               \
                                                        num_tokens,                  \
                                                        total_warps,                 \
                                                        q_out,                       \
                                                        k_cache,                     \
                                                        v_cache,                     \
                                                        slot_mapping,                \
                                                        per_tensor_k_scale,          \
                                                        per_tensor_v_scale,          \
                                                        k_out,                       \
                                                        v_out,                       \
                                                        use_shuffle_layout,          \
                                                        block_size,                  \
                                                        x,                           \
                                                        (int)rotary_dim);            \
    }

    if(is_interleaved)
    {
        switch(head_size)
        {
        case 64: DISPATCH_NEOX(64, true) break;
        case 128: DISPATCH_NEOX(128, true) break;
        case 256: DISPATCH_NEOX(256, true) break;
        }
    }
    else
    {
        switch(head_size)
        {
        case 64: DISPATCH_NEOX(64, false) break;
        case 128: DISPATCH_NEOX(128, false) break;
        case 256: DISPATCH_NEOX(256, false) break;
        }
    }

#undef DISPATCH_NEOX
}

template <typename T, typename KVT>
void fused_rope_rms_set_kv(const T* qkv,
                           const T* q_w,
                           const T* k_w,
                           const T* cos_sin,
                           const int64_t* positions,
                           int64_t positions_stride_0,
                           int64_t positions_stride_1,
                           int64_t num_tokens,
                           int64_t num_heads_q,
                           int64_t num_heads_k,
                           int64_t num_heads_v,
                           int64_t head_size,
                           bool is_neox_style,
                           double eps,
                           T* q_out,
                           KVT* k_cache,
                           KVT* v_cache,
                           const int64_t* slot_mapping,
                           hipStream_t stream,
                           float per_tensor_k_scale = 1.0,
                           float per_tensor_v_scale = 1.0,
                           KVT* k_out               = nullptr,
                           KVT* v_out               = nullptr,
                           bool use_shuffle_layout  = false,
                           int64_t block_size       = 0,
                           int64_t x                = 0,
                           int64_t rotary_dim       = 0)
{
    TORCH_CHECK(head_size == 64 || head_size == 128 || head_size == 256);
    constexpr int THREAD_BLOCK_SIZE = 256;
    auto total_warps                = num_tokens * (num_heads_q + num_heads_k + num_heads_v);
    auto num_warps_per_block        = THREAD_BLOCK_SIZE / WARP_SIZE;
    dim3 threadsPerBlock(THREAD_BLOCK_SIZE);
    dim3 numBlocks((total_warps + num_warps_per_block - 1) / num_warps_per_block);
    std::array<int64_t, 1> mrope_section = {0};

#define DISPATCH_NEOX(HEAD_SIZE)                                             \
    if(is_neox_style)                                                        \
    {                                                                        \
        fused_mrope_rms_kv_kernel<T, HEAD_SIZE, true, false, false, 1, KVT>  \
            <<<numBlocks, threadsPerBlock, 0, stream>>>(qkv,                 \
                                                        q_w,                 \
                                                        k_w,                 \
                                                        cos_sin,             \
                                                        positions,           \
                                                        positions_stride_0,  \
                                                        positions_stride_1,  \
                                                        num_heads_q,         \
                                                        num_heads_k,         \
                                                        num_heads_v,         \
                                                        eps,                 \
                                                        mrope_section,       \
                                                        num_tokens,          \
                                                        total_warps,         \
                                                        q_out,               \
                                                        k_cache,             \
                                                        v_cache,             \
                                                        slot_mapping,        \
                                                        per_tensor_k_scale,  \
                                                        per_tensor_v_scale,  \
                                                        k_out,               \
                                                        v_out,               \
                                                        use_shuffle_layout,  \
                                                        block_size,          \
                                                        x,                   \
                                                        (int)rotary_dim);    \
    }                                                                        \
    else                                                                     \
    {                                                                        \
        fused_mrope_rms_kv_kernel<T, HEAD_SIZE, false, false, false, 1, KVT> \
            <<<numBlocks, threadsPerBlock, 0, stream>>>(qkv,                 \
                                                        q_w,                 \
                                                        k_w,                 \
                                                        cos_sin,             \
                                                        positions,           \
                                                        positions_stride_0,  \
                                                        positions_stride_1,  \
                                                        num_heads_q,         \
                                                        num_heads_k,         \
                                                        num_heads_v,         \
                                                        eps,                 \
                                                        mrope_section,       \
                                                        num_tokens,          \
                                                        total_warps,         \
                                                        q_out,               \
                                                        k_cache,             \
                                                        v_cache,             \
                                                        slot_mapping,        \
                                                        per_tensor_k_scale,  \
                                                        per_tensor_v_scale,  \
                                                        k_out,               \
                                                        v_out,               \
                                                        use_shuffle_layout,  \
                                                        block_size,          \
                                                        x,                   \
                                                        (int)rotary_dim);    \
    }

    switch(head_size)
    {
    case 64: DISPATCH_NEOX(64) break;
    case 128: DISPATCH_NEOX(128) break;
    case 256: DISPATCH_NEOX(256) break;
    }

#undef DISPATCH_NEOX
}

} // namespace mrope_utils
