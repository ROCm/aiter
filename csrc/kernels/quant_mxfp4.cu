// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "aiter_hip_common.h"
#include "aiter_dispatch.h"
#include "aiter_opus_plus.h"
#include "aiter_stream.h"
#include "quant.h"

namespace aiter {

// Even: e8m0 scale via even-rounding group max to nearest power-of-2.
//   gfx950: HW builtin (exact RNE); gfx942: SW fallback (round-half-away).
enum class MxFp4RoundMode : int { Even = 0 };

#define EVEN_ROUND_FP32_SIGN_EXP_MASK 0x7F800000u
#define EVEN_ROUND_VAL_TO_ADD         0x00200000u
#define EVEN_ROUND_FP4_EMAX           2

static constexpr int kGroupSize      = 32;
static constexpr int kPackedPerGroup = kGroupSize / 2;
static constexpr int kBlockThreads   = 256;

using packed_u16x8_t = vector_t<uint16_t, 8>;

#if defined(__gfx950__)
template <typename ftype, int sel>
__device__ __forceinline__ uint32_t cvt_fp4_pk(uint32_t src, uint32_t pair, float scale) {
    if constexpr (std::is_same_v<ftype, hip_bfloat16>)
        return __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(
            src, __builtin_bit_cast(bf16x2_t, pair), scale, sel);
    else
        return __builtin_amdgcn_cvt_scalef32_pk_fp4_f16(
            src, __builtin_bit_cast(fp16x2_t, pair), scale, sel);
}
#else
__device__ __forceinline__ uint8_t even_round_e2m1(float val) {
    float a = fabsf(val);
    uint8_t mag;
    if      (a >= 5.0f)  mag = 7;
    else if (a >= 3.5f)  mag = 6;
    else if (a >= 2.5f)  mag = 5;
    else if (a >= 1.75f) mag = 4;
    else if (a >= 1.25f) mag = 3;
    else if (a >= 0.75f) mag = 2;
    else if (a >= 0.25f) mag = 1;
    else                 mag = 0;
    uint8_t sign_bit = (val < 0.0f) ? 8u : 0u;
    return sign_bit | mag;
}
#endif

__device__ __forceinline__ int fp4_scale_shuffle_id(int scaleN_pad, int x, int y) {
    return (x / 32 * scaleN_pad) * 32 +
           (y / 8) * 256 + (y % 4) * 64 + (x % 16) * 4 +
           (y % 8) / 4 * 2 + (x % 32) / 16;
}

__device__ __forceinline__ int a16w4_shuffle_scale_id(
    int scaleN, int ori_rows, int x, int y, bool gate_up
) {
    int N1_idx, N_Pack_idx, N_Lane_idx;
    if (gate_up) {
        int half_rows = ori_rows / 2;
        N_Pack_idx = x / half_rows;
        int rem    = x % half_rows;
        N1_idx     = rem / 16;
        N_Lane_idx = rem % 16;
    } else {
        N1_idx     = x / 32;
        N_Pack_idx = (x % 32) / 16;
        N_Lane_idx = x % 16;
    }
    int K1_idx     = y / 8;
    int K_Pack_idx = (y % 8) / 4;
    int K_Lane_idx = y % 4;
    int k1_size    = scaleN / 8;
    return N1_idx * (k1_size * 256) +
           K1_idx * 256 + K_Lane_idx * 64 + N_Lane_idx * 4 +
           K_Pack_idx * 2 + N_Pack_idx;
}

template <typename float_type, MxFp4RoundMode rmode, bool e8m0_shuffle, bool a16w4_shuffle, bool shuffle_weight>
__global__ __launch_bounds__(kBlockThreads)
void quant_mxfp4_kernel(
    const float_type* __restrict__ inp,
    uint8_t* __restrict__ out_packed,
    float* __restrict__ out_scale,
    int64_t ori_rows, int32_t ori_cols,
    int32_t scaleN, int32_t scaleN_pad, bool gate_up
) {
    int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t x   = gid / scaleN_pad;
    int32_t y   = gid % scaleN_pad;

    if (x >= ori_rows || y >= scaleN) return;

    const packed_u16x8_t* vp = reinterpret_cast<const packed_u16x8_t*>(
        inp + x * ori_cols + y * kGroupSize
    );
    packed_u16x8_t chunks[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) chunks[i] = vp[i];

    const float_type* elems = reinterpret_cast<const float_type*>(chunks);

    float group_max = 0.f;
#if defined(__gfx950__)
    #pragma unroll
    for (int i = 0; i < kGroupSize; ++i)
        group_max = fmaxf(group_max, fabsf(static_cast<float>(elems[i])));
#else
    float vals[kGroupSize];
    #pragma unroll
    for (int i = 0; i < kGroupSize; ++i) {
        vals[i]   = static_cast<float>(elems[i]);
        group_max = fmaxf(group_max, fabsf(vals[i]));
    }
#endif

    uint32_t max_bits = __float_as_uint(group_max);
    float dequant_scale;
    uint8_t biased_exp;

    if constexpr (rmode == MxFp4RoundMode::Even) {
        max_bits          = (max_bits + EVEN_ROUND_VAL_TO_ADD) & EVEN_ROUND_FP32_SIGN_EXP_MASK;
        float max_rounded = __uint_as_float(max_bits);

        float scale_unbiased = floorf(log2f(max_rounded)) - EVEN_ROUND_FP4_EMAX;
        scale_unbiased       = fminf(fmaxf(scale_unbiased, -127.0f), 127.0f);
        dequant_scale        = exp2f(scale_unbiased);
        biased_exp           = (__float_as_uint(dequant_scale) >> 23) & 0xFF;
    }

    u8x16_t packed;

#if defined(__gfx950__)
    const uint32_t* pairs = reinterpret_cast<const uint32_t*>(chunks);
    u32x4_t pw;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint32_t* p = pairs + j * 4;
        uint32_t w = 0;
        w = cvt_fp4_pk<float_type, 0>(w, p[0], dequant_scale);
        w = cvt_fp4_pk<float_type, 1>(w, p[1], dequant_scale);
        w = cvt_fp4_pk<float_type, 2>(w, p[2], dequant_scale);
        w = cvt_fp4_pk<float_type, 3>(w, p[3], dequant_scale);
        pw[j] = w;
    }
    packed = __builtin_bit_cast(u8x16_t, pw);
#else
    float quant_scale = (dequant_scale == 0.0f) ? 0.0f : (1.0f / dequant_scale);
    #pragma unroll
    for (int i = 0; i < kPackedPerGroup; ++i) {
        packed[i] = even_round_e2m1(vals[i * 2] * quant_scale)
                  | (even_round_e2m1(vals[i * 2 + 1] * quant_scale) << 4);
    }
#endif

    int64_t w_base;
    if constexpr (shuffle_weight) {
        int K_pk = ori_cols / 2;
        if constexpr (e8m0_shuffle) {
            w_base = (int64_t)(x >> 4) * 16 * K_pk + (x & 15) * 16
                   + (y >> 1) * 512 + (y & 1) * 256;
        } else if constexpr (a16w4_shuffle) {
            int K0 = y >> 2, KLane = y & 3;
            if (gate_up) {
                int half_rows = (int)ori_rows >> 1;
                int N_Pack    = (int)x / half_rows;
                int rem       = (int)x % half_rows;
                int K0_size   = K_pk >> 6;
                w_base = (int64_t)(rem >> 4) * (2 * K0_size * 1024)
                       + (int64_t)N_Pack * (K0_size * 1024)
                       + K0 * 1024 + KLane * 256 + (rem & 15) * 16;
            } else {
                w_base = (int64_t)(x >> 4) * 16 * K_pk + (x & 15) * 16
                       + K0 * 1024 + KLane * 256;
            }
        }
    } else {
        w_base = x * (ori_cols / 2) + y * kPackedPerGroup;
    }
    *reinterpret_cast<u8x16_t*>(out_packed + w_base) = packed;

    int scale_idx;
    if constexpr (e8m0_shuffle) {
        scale_idx = fp4_scale_shuffle_id(scaleN_pad, (int)x, y);
    } else if constexpr (a16w4_shuffle) {
        scale_idx = a16w4_shuffle_scale_id(scaleN, (int)ori_rows, (int)x, y, gate_up);
    } else {
        scale_idx = (int)(x * scaleN + y);
    }
    reinterpret_cast<uint8_t*>(out_scale)[scale_idx] = biased_exp;
}

#define MXFP4_LAUNCH(ftype, rmode, ss, a16, sw)                              \
    quant_mxfp4_kernel<ftype, rmode, ss, a16, sw>                            \
        <<<(int)grid_size, kBlockThreads, 0, stream>>>(                      \
            reinterpret_cast<const ftype*>(inp.data_ptr()),                   \
            reinterpret_cast<uint8_t*>(out_packed.data_ptr()),               \
            reinterpret_cast<float*>(out_scale.data_ptr()),                  \
            ori_rows, ori_cols, scaleN, scaleN_pad, gate_up)

#define MXFP4_DISPATCH(ftype, rmode)                                         \
    if (e8m0_shuffle) {                                                      \
        if (shuffle_weight) { MXFP4_LAUNCH(ftype, rmode, true, false, true); }  \
        else                { MXFP4_LAUNCH(ftype, rmode, true, false, false); } \
    } else if (a16w4_shuffle) {                                              \
        if (shuffle_weight) { MXFP4_LAUNCH(ftype, rmode, false, true, true); }  \
        else                { MXFP4_LAUNCH(ftype, rmode, false, true, false); } \
    } else {                                                                 \
        MXFP4_LAUNCH(ftype, rmode, false, false, false);                     \
    }

void quant_mxfp4(
    const aiter_tensor_t& inp,
    aiter_tensor_t& out_packed,
    aiter_tensor_t& out_scale,
    int group_size,
    int round_mode,
    bool e8m0_shuffle,
    bool a16w4_shuffle,
    bool gate_up,
    bool shuffle_weight
) {
    AITER_CHECK(inp.is_contiguous(), __func__, " expected input to be contiguous");
    AITER_CHECK(inp.dim() == 2, __func__, " expected 2D input");
    AITER_CHECK(out_packed.is_contiguous(), __func__, " expected out_packed to be contiguous");
    AITER_CHECK(out_scale.is_contiguous(), __func__, " expected out_scale to be contiguous");
    AITER_CHECK(group_size == 32, __func__, " expected group_size=32");
    AITER_CHECK(round_mode == 0, __func__, " only Even round mode (0) is supported");
    AITER_CHECK(!(e8m0_shuffle && a16w4_shuffle),
                __func__, " e8m0_shuffle and a16w4_shuffle are mutually exclusive");
    AITER_CHECK(!shuffle_weight || e8m0_shuffle || a16w4_shuffle,
                __func__, " shuffle_weight requires e8m0_shuffle or a16w4_shuffle");

    const int64_t ori_rows = inp.size(0);
    const int32_t ori_cols = inp.size(1);
    AITER_CHECK(ori_cols % group_size == 0, __func__, " cols must be divisible by group_size");

    const int32_t scaleN     = ori_cols / group_size;
    const int32_t scaleN_pad = e8m0_shuffle ? ((scaleN + 7) / 8) * 8 : scaleN;

    if (a16w4_shuffle) {
        AITER_CHECK(ori_rows % 32 == 0, __func__, " a16w4 scale shuffle requires rows % 32 == 0");
        AITER_CHECK(scaleN % 8 == 0, __func__, " a16w4 scale shuffle requires scaleN % 8 == 0");
    }
    if (shuffle_weight) {
        AITER_CHECK(ori_rows % 16 == 0, __func__, " shuffle_weight requires rows % 16 == 0");
        int K_pk = ori_cols / 2;
        if (e8m0_shuffle) {
            AITER_CHECK(K_pk % 32 == 0, __func__, " e8m0 weight shuffle requires K_pk % 32 == 0");
        } else {
            AITER_CHECK(K_pk % 64 == 0, __func__, " a16w4 weight shuffle requires K_pk % 64 == 0");
        }
    }

    const int64_t total_groups = ori_rows * (int64_t)scaleN_pad;
    const int64_t grid_size    = (total_groups + kBlockThreads - 1) / kBlockThreads;
    AITER_CHECK(grid_size <= 2147483647LL, __func__, " grid size exceeds maximum");

    HipDeviceGuard device_guard(inp.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    AITER_DISPATCH_FLOATING16_TYPES_rmTorch(
        inp.dtype(), "quant_mxfp4_kernel", [&] {
            MXFP4_DISPATCH(scalar_t, MxFp4RoundMode::Even);
        });
}

#undef MXFP4_LAUNCH
#undef MXFP4_DISPATCH

} // namespace aiter
