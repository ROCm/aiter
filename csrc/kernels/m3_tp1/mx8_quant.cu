// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

// MXFP8 activation quant as a streaming kernel: a wave moves ITER x 1 KB of bf16 per load
// instruction, a 32-group is 4 lanes (DPP quad max).
#define AITER_NO_TORCH_TYPES
#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "m3_tp1_prefill.h"

namespace aiter {
namespace {

constexpr int kMx8QuantIter = 2; // 2 KB a wave in flight, the fastest measured

template <int CTRL>
__device__ __forceinline__ float dpp(float v)
{
    return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v), CTRL, 0xf, 0xf, false));
}

// code = ceil_pow2(max(amax, 1e-10) / 448), q = clamp(x * 2^(127 - code), +-448) -> e4m3 RNE.
// fmaxf / fminf: a NaN input leaves the group's amax alone and quantizes to -448.
template <int ITER>
__global__ void __launch_bounds__(256) m3_tp1_mx8_quant_kernel(const uint4* __restrict__ x,
                                                               uint2* __restrict__ q,
                                                               uint8_t* __restrict__ s,
                                                               int64_t n_chunks,
                                                               int groups_per_row,
                                                               int w4)
{
#if defined(__gfx950__)
    // chunk = 8 bf16 (16 B); wave w covers chunks [w * 64 * ITER, (w + 1) * 64 * ITER)
    const int64_t wave = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 64;
    const int lane     = threadIdx.x % 64;
    const int64_t c0   = wave * 64 * ITER + lane;
    uint4 raw[ITER];
#pragma unroll
    for(int k = 0; k < ITER; ++k)
        raw[k] = (c0 + k * 64 < n_chunks) ? x[c0 + k * 64] : make_uint4(0, 0, 0, 0);
#pragma unroll
    for(int k = 0; k < ITER; ++k)
    {
        const int64_t c   = c0 + k * 64;
        const uint32_t w[4] = {raw[k].x, raw[k].y, raw[k].z, raw[k].w};
        float v[8];
#pragma unroll
        for(int i = 0; i < 4; ++i)
        {
            v[2 * i]     = __builtin_bit_cast(float, w[i] << 16);
            v[2 * i + 1] = __builtin_bit_cast(float, w[i] & 0xffff0000u);
        }
        float am = 0.0f;
#pragma unroll
        for(int i = 0; i < 8; ++i)
            am = fmaxf(am, fabsf(v[i]));
        am = fmaxf(am, dpp<0xB1>(am)); // quad xor 1
        am = fmaxf(am, dpp<0x4E>(am)); // quad xor 2
        am = fmaxf(am, 1e-10f);
        const uint32_t bits = __builtin_bit_cast(uint32_t, __fmul_rn(am, 1.0f / 448.0f));
        const uint32_t code = (bits >> 23) + ((bits & 0x7FFFFFu) != 0u ? 1u : 0u);
        const int32_t rexp  = 254 - static_cast<int32_t>(code);
        const float recip   = __builtin_bit_cast(float, static_cast<uint32_t>(rexp > 0 ? rexp : 0) << 23);
        float y[8];
#pragma unroll
        for(int i = 0; i < 8; ++i)
            y[i] = fminf(fmaxf(__fmul_rn(v[i], recip), -448.0f), 448.0f);
        int lo = 0, hi = 0;
        lo = __builtin_amdgcn_cvt_pk_fp8_f32(y[0], y[1], lo, false);
        lo = __builtin_amdgcn_cvt_pk_fp8_f32(y[2], y[3], lo, true);
        hi = __builtin_amdgcn_cvt_pk_fp8_f32(y[4], y[5], hi, false);
        hi = __builtin_amdgcn_cvt_pk_fp8_f32(y[6], y[7], hi, true);
        if(c < n_chunks)
        {
            q[c] = make_uint2(static_cast<uint32_t>(lo), static_cast<uint32_t>(hi));
            if((lane & 3) == 0)
            {
                const int64_t gid = c / 4; // group index over the flattened [rows, K]
                int64_t si        = gid;
                if(w4)
                {
                    // shuffle_scale_w4: 256-byte block (row / 32, group / 8), word (g % 4) * 16 + row % 16, byte (g / 4) * 2 + row % 32 / 16
                    const int64_t row = gid / groups_per_row;
                    const int g       = static_cast<int>(gid - row * groups_per_row);
                    const int rr = static_cast<int>(row & 31), gg = g & 7;
                    si = ((row >> 5) * (groups_per_row >> 3) + (g >> 3)) * 256 + ((gg & 3) * 16 + (rr & 15)) * 4 +
                         (gg >> 2) * 2 + (rr >> 4);
                }
                s[si] = static_cast<uint8_t>(code);
            }
        }
    }
#else
    // gfx950-only (OCP e4m3 conversion): compiled out for the other archs of a multi-arch build;
    // the host wrapper never launches it there.
    __builtin_trap();
#endif
}

} // namespace

void m3_tp1_mx8_quant(const aiter_tensor_t& x, aiter_tensor_t& q, aiter_tensor_t& s, bool w4)
{
    aiter_detail::AiterThrowGuard throw_guard;

    AITER_CHECK(x.dtype() == AITER_DTYPE_bf16 && x.is_contiguous() && x.dim() >= 1,
                "m3_tp1_mx8_quant: x must be a contiguous bf16 tensor");
    const int64_t k     = x.size(-1);
    const int64_t numel = static_cast<int64_t>(x.numel());
    AITER_CHECK(k > 0 && k % 32 == 0 && (!w4 || k % 256 == 0),
                "m3_tp1_mx8_quant: K must be a multiple of 32 (256 with w4), got ", k);
    const int64_t rows = numel / k;
    AITER_CHECK(q.element_size() == 1 && q.is_contiguous() && q.numel() == x.numel(),
                "m3_tp1_mx8_quant: q must be a contiguous 1-byte tensor of x's size");
    const int64_t scale_bytes = (w4 ? (rows + 255) / 256 * 256 : rows) * (k / 32);
    AITER_CHECK(s.element_size() == 1 && s.is_contiguous() && s.numel() >= static_cast<size_t>(scale_bytes),
                "m3_tp1_mx8_quant: s must be a contiguous 1-byte tensor of at least ", scale_bytes, " elements");
    AITER_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(q.data_ptr()) % 8 == 0,
                "m3_tp1_mx8_quant: x must be 16-byte and q 8-byte aligned");
    if(numel == 0)
        return;

    const HipDeviceGuard guard(x.device_id);
    const hipStream_t stream = getCurrentHIPStream();
    const int64_t n_chunks   = numel / 8;
    const int64_t waves      = (n_chunks + 64 * kMx8QuantIter - 1) / (64 * kMx8QuantIter);
    m3_tp1_mx8_quant_kernel<kMx8QuantIter><<<dim3((waves + 3) / 4), dim3(256), 0, stream>>>(
        static_cast<const uint4*>(x.data_ptr()),
        static_cast<uint2*>(q.data_ptr()),
        static_cast<uint8_t*>(s.data_ptr()),
        n_chunks,
        static_cast<int>(k / 32),
        w4 ? 1 : 0);
    HIP_CALL(hipGetLastError());
}

} // namespace aiter
