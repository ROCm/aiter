// SPDX-License-Identifier: MIT
#pragma once
#include <cstdint>
// E4M3FN RNE, with signed zero, gradual underflow, and NaN on overflow.
// Same format semantics as torch.Tensor.to(torch.float8_e4m3fn); no saturation.
// Bit conversion follows the public c10 Float8_e4m3fn scalar algorithm.
namespace q_encode {
union FloatBits
{
    float value;
    uint32_t bits;
};
__host__ __device__ __forceinline__ float bf16_float(uint16_t x)
{
    FloatBits v;
    v.bits = uint32_t(x) << 16;
    return v.value;
}
__host__ __device__ __forceinline__ uint8_t from_fp32(float x)
{
    FloatBits v;
    v.value             = x;
    const uint32_t sign = v.bits & 0x80000000u;
    uint32_t bits       = v.bits ^ sign, result;
    if(bits >= (1087u << 20))
        result = 0x7f;
    else if(bits < (121u << 23))
    {
        v.bits = bits;
        FloatBits offset;
        offset.bits = 141u << 23;
        v.value     = v.value + offset.value;
        result      = v.bits - offset.bits;
    }
    else
    {
        const uint32_t odd = (bits >> 20) & 1;
        bits += uint32_t(7 - 127) * (1u << 23) + 0x7ffffu + odd;
        result = bits >> 20;
    }
    return uint8_t(result | (sign >> 24));
}
__device__ __forceinline__ uint32_t pack4(const uint16_t* src, float scale)
{
    uint32_t result = 0;
#pragma unroll
    for(int j = 0; j < 4; ++j)
    {
        result |= uint32_t(from_fp32(bf16_float(src[j]) / scale)) << (8 * j);
        // Bound scalar-conversion temporaries in the non-unit-scale path.
        // The empty asm preserves encoding while keeping later loads ordered.
        asm volatile("" : "+v"(result) : : "memory");
    }
    return result;
}
// A scale-one fast path: packed BF16 loads, direct packed BF16 -> FP8 RNE, then
// explicit signed NaN encoding for the non-saturating E4M3FN contract. Values
// with abs>464 overflow; the exact midpoint464 rounds to even448. This fixup
// also covers every Inf/NaN bit pattern and preserves its sign.
using Packed4 = uint32_t __attribute__((ext_vector_type(4)));
__host__ __device__ __forceinline__ uint32_t fast_pair(uint32_t bits)
{
    const uint16_t lo = uint16_t(bits), hi = uint16_t(bits >> 16);
    uint32_t packed;
#ifdef __HIP_DEVICE_COMPILE__
    // gfx950 packed conversion uses only the scale exponent; unity is exact.
    // Do not use this as a replacement for arbitrary FP32 division by q_scale.
    using BF16x2        = __bf16 __attribute__((ext_vector_type(2)));
    using I16x2         = short __attribute__((ext_vector_type(2)));
    const BF16x2 source = __builtin_bit_cast(BF16x2, bits);
    const I16x2 result  = __builtin_amdgcn_cvt_scalef32_pk_fp8_bf16(I16x2{}, source, 1.0f, false);
    packed              = __builtin_bit_cast(uint32_t, result);
#else
    // The host audit validates the fixup and boundary algorithm. Actual hardware
    packed = uint32_t(from_fp32(bf16_float(lo))) | (uint32_t(from_fp32(bf16_float(hi))) << 8);
#endif
    const uint32_t a = (lo & 0x7fffu) > 0x43e8u ? ((lo >> 8) & 0x80u) | 0x7fu : packed & 0xffu;
    const uint32_t b =
        (hi & 0x7fffu) > 0x43e8u ? ((hi >> 8) & 0x80u) | 0x7fu : (packed >> 8) & 0xffu;
    return a | (b << 8);
}
__host__ __device__ __forceinline__ Packed4 pack_loaded16(Packed4 lo, Packed4 hi)
{
    Packed4 result;
    result[0] = fast_pair(lo[0]) | (fast_pair(lo[1]) << 16);
    result[1] = fast_pair(lo[2]) | (fast_pair(lo[3]) << 16);
    result[2] = fast_pair(hi[0]) | (fast_pair(hi[1]) << 16);
    result[3] = fast_pair(hi[2]) | (fast_pair(hi[3]) << 16);
    return result;
}
template <bool Fast>
__device__ __forceinline__ Packed4 pack16(const uint16_t* src, float scale)
{
    Packed4 result;
    if constexpr(Fast)
    {
        const Packed4 lo = *reinterpret_cast<const Packed4*>(src);
        const Packed4 hi = *reinterpret_cast<const Packed4*>(src + 8);
        result           = pack_loaded16(lo, hi);
    }
    else
    {
#pragma unroll
        for(int j = 0; j < 4; ++j)
            result[j] = pack4(src + j * 4, scale);
    }
    return result;
}
} // namespace q_encode
