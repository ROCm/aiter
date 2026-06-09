// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// HIP port of the Triton FP4 per-channel/kblock quantizer.
//
// Input:
//   v          (BH, T, D)        bf16 contiguous
// Outputs:
//   packed     (BH, T, D/2)      uint8, two E2M1 nibbles per byte
//   scale_byte (BH, T/KBLOCK, D) uint8, E8M0 biased exponent

#include "fp4_quant.h"

#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <hip/hip_runtime.h>
#include <torch/all.h>

#include <cstdint>

#ifndef TBPB
#define TBPB 1
#endif

#ifndef NTLOAD
#define NTLOAD 1
#endif

namespace {

__device__ __forceinline__ float bf16lo_to_f32(uint32_t pair)
{
    return __uint_as_float(pair << 16);
}

__device__ __forceinline__ float bf16hi_to_f32(uint32_t pair)
{
    return __uint_as_float(pair & 0xFFFF0000u);
}

__device__ __forceinline__ int clamp_i32(int x, int lo, int hi)
{
    return x < lo ? lo : (x > hi ? hi : x);
}

// Returns clamped E8M0 exponent in [-127, 127]. This matches
// ceil(log2(amax / 6)) for positive normal amax values.
__device__ __forceinline__ int e8m0_exp_clamped(float amax)
{
    int bits = __float_as_int(amax);
    int e = ((bits >> 23) & 0xFF) - 127;
    int mant = bits & 0x7FFFFF;
    e = e - 2 + (mant > 0x400000 ? 1 : 0); // 6 = 1.5 * 2^2
    return clamp_i32(e, -127, 127);
}

__device__ __forceinline__ uint8_t e2m1_nibble(float vn)
{
    float ax = fabsf(vn);
    int idx = (ax > 0.25f) + (ax > 0.75f) + (ax > 1.25f) + (ax > 1.75f) +
              (ax > 2.5f) + (ax > 3.5f) + (ax > 5.0f);
    int sign = (vn < 0.0f) ? 8 : 0;
    return static_cast<uint8_t>(idx | sign);
}

template <int KBLOCK, int TB_PER_BLK>
__global__ void __launch_bounds__(256)
fp4_pc_kblock_kernel(const uint32_t* __restrict__ Vu,
                     uint8_t* __restrict__ packed,
                     uint8_t* __restrict__ scaleb,
                     long BH,
                     long T,
                     int D)
{
    const int DHALF = D >> 1;
    const long num_tblocks = T / KBLOCK;
    const long global_tb = static_cast<long>(blockIdx.x) * TB_PER_BLK + threadIdx.y;
    const int p = threadIdx.x;
    if (p >= DHALF || global_tb >= BH * num_tblocks) {
        return;
    }

    const long bh = global_tb / num_tblocks;
    const long tb = global_tb % num_tblocks;
    const long t_start = tb * static_cast<long>(KBLOCK);

    const long v_base = (bh * T + t_start) * static_cast<long>(DHALF);
    const long v_stride = DHALF;

    uint32_t raw[KBLOCK];
    float amax_e = 1e-30f;
    float amax_o = 1e-30f;

#pragma unroll
    for (int t = 0; t < KBLOCK; ++t) {
#if NTLOAD
        uint32_t r = __builtin_nontemporal_load(Vu + v_base + static_cast<long>(t) * v_stride + p);
#else
        uint32_t r = Vu[v_base + static_cast<long>(t) * v_stride + p];
#endif
        raw[t] = r;
        amax_e = fmaxf(amax_e, fabsf(bf16lo_to_f32(r)));
        amax_o = fmaxf(amax_o, fabsf(bf16hi_to_f32(r)));
    }

    const int exp_e = e8m0_exp_clamped(amax_e);
    const int exp_o = e8m0_exp_clamped(amax_o);
    const int sb_e = exp_e + 127;
    const int sb_o = exp_o + 127;
    const float scale_e = __int_as_float(sb_e << 23);
    const float scale_o = __int_as_float(sb_o << 23);
    const float inv_e = 1.0f / scale_e;
    const float inv_o = 1.0f / scale_o;

    const long s_base = (bh * num_tblocks + tb) * static_cast<long>(D) + static_cast<long>(p << 1);
    uint16_t sword = static_cast<uint16_t>(clamp_i32(sb_e, 0, 255) |
                                           (clamp_i32(sb_o, 0, 255) << 8));
    __builtin_nontemporal_store(sword, reinterpret_cast<uint16_t*>(scaleb + s_base));

    const long p_base = (bh * T + t_start) * static_cast<long>(DHALF) + p;
#pragma unroll
    for (int t = 0; t < KBLOCK; ++t) {
        float ve = bf16lo_to_f32(raw[t]);
        float vo = bf16hi_to_f32(raw[t]);
        float vne = fminf(fmaxf(ve * inv_e, -6.0f), 6.0f);
        float vno = fminf(fmaxf(vo * inv_o, -6.0f), 6.0f);
        uint8_t byte = static_cast<uint8_t>((e2m1_nibble(vno) << 4) | e2m1_nibble(vne));
        __builtin_nontemporal_store(byte, packed + p_base + static_cast<long>(t) * v_stride);
    }
}

void launch_fp4_pc_kblock(const void* v,
                          void* packed,
                          void* scaleb,
                          long BH,
                          long T,
                          long D,
                          int KBLOCK,
                          hipStream_t stream)
{
    const int DHALF = static_cast<int>(D >> 1);
    const long num_tblocks = T / KBLOCK;
    const long groups = (BH * num_tblocks + TBPB - 1) / TBPB;
    dim3 grid(static_cast<unsigned>(groups));
    dim3 block(static_cast<unsigned>(DHALF), TBPB);

    auto* Vu = reinterpret_cast<const uint32_t*>(v);
    auto* P = reinterpret_cast<uint8_t*>(packed);
    auto* S = reinterpret_cast<uint8_t*>(scaleb);

    switch (KBLOCK) {
        case 16:
            fp4_pc_kblock_kernel<16, TBPB><<<grid, block, 0, stream>>>(Vu, P, S, BH, T, static_cast<int>(D));
            break;
        case 32:
            fp4_pc_kblock_kernel<32, TBPB><<<grid, block, 0, stream>>>(Vu, P, S, BH, T, static_cast<int>(D));
            break;
        case 64:
            fp4_pc_kblock_kernel<64, TBPB><<<grid, block, 0, stream>>>(Vu, P, S, BH, T, static_cast<int>(D));
            break;
        default:
            break;
    }
}

} // namespace

namespace aiter {

void quantize_fp4_e8m0_per_channel_kblock_hip(torch::Tensor& packed,
                                              torch::Tensor& scale_byte,
                                              const torch::Tensor& v,
                                              int64_t kblock_size)
{
    TORCH_CHECK(v.is_cuda(), "v must be a CUDA/HIP tensor");
    TORCH_CHECK(packed.is_cuda(), "packed must be a CUDA/HIP tensor");
    TORCH_CHECK(scale_byte.is_cuda(), "scale_byte must be a CUDA/HIP tensor");
    TORCH_CHECK(v.scalar_type() == torch::kBFloat16, "v must be bfloat16");
    TORCH_CHECK(packed.scalar_type() == torch::kUInt8, "packed must be uint8");
    TORCH_CHECK(scale_byte.scalar_type() == torch::kUInt8, "scale_byte must be uint8");
    TORCH_CHECK(v.dim() == 3, "v must have shape (BH, T, D)");
    TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
    TORCH_CHECK(packed.is_contiguous(), "packed must be contiguous");
    TORCH_CHECK(scale_byte.is_contiguous(), "scale_byte must be contiguous");
    TORCH_CHECK(kblock_size == 16 || kblock_size == 32 || kblock_size == 64,
                "kblock_size must be one of 16, 32, 64");

    const int64_t BH = v.size(0);
    const int64_t T = v.size(1);
    const int64_t D = v.size(2);
    TORCH_CHECK(D % 2 == 0, "D must be even");
    TORCH_CHECK(T % kblock_size == 0, "T must be divisible by kblock_size");
    TORCH_CHECK(packed.sizes() == torch::IntArrayRef({BH, T, D / 2}),
                "packed must have shape (BH, T, D/2)");
    TORCH_CHECK(scale_byte.sizes() == torch::IntArrayRef({BH, T / kblock_size, D}),
                "scale_byte must have shape (BH, T/kblock_size, D)");

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(v));
    const hipStream_t stream = at::hip::getCurrentHIPStream();
    launch_fp4_pc_kblock(v.data_ptr(),
                         packed.data_ptr(),
                         scale_byte.data_ptr(),
                         static_cast<long>(BH),
                         static_cast<long>(T),
                         static_cast<long>(D),
                         static_cast<int>(kblock_size),
                         stream);
}

} // namespace aiter
