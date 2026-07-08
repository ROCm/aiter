// poc_kl f8_to_f32 / f32_to_f8 (FP8 E4M3, bias=8) — device port of common_mi300_fp8.h
#pragma once

#include <cstdint>

#include <hip/hip_runtime.h>

namespace pa_decode {

static constexpr uint32_t kFp8Fmt = 1u;  // FP8_FMT in poc_kl common.h
static constexpr uint32_t kBf8Fmt = 0u;  // BF8_FMT
static constexpr bool kFp8Bias = true;   // fp8_bias=true in pa.exe / gen_pa_buffers
static constexpr float kFp8QuantDenoBias8 = 240.0f;

// poc_kl quant() uses C abs(float) → int truncation (common_helper.h).
__device__ __forceinline__ float poc_kl_quant_row_abs(float x) {
    const int ix = static_cast<int>(x);
    return static_cast<float>(ix >= 0 ? ix : -ix);
}

__device__ __forceinline__ uint32_t fp8_sign(uint8_t x) { return (x >> 7) & 1u; }
__device__ __forceinline__ uint32_t fp8_exp_e4m3(uint8_t x) { return (x >> 3) & 0xfu; }
__device__ __forceinline__ uint32_t fp8_man_e4m3(uint8_t x) { return x & 0x7u; }

__device__ __forceinline__ uint32_t f32_sign(uint32_t x) { return x >> 31; }
__device__ __forceinline__ uint32_t f32_bexp(uint32_t x) { return (x >> 23) & 0xffu; }
__device__ __forceinline__ uint32_t f32_man(uint32_t x) { return x & 0x7fffffu; }

__device__ __forceinline__ uint32_t fp8_bias(uint32_t f8_src_fmt, bool f8_bias) {
    if (f8_src_fmt == kBf8Fmt) {
        return f8_bias ? 16u : 15u;
    }
    return f8_bias ? 8u : 7u;
}

__device__ __forceinline__ uint32_t fp8_mant_bits(uint32_t f8_src_fmt) {
    return (f8_src_fmt == kBf8Fmt) ? 2u : 3u;
}

__device__ __forceinline__ int32_t fp8_max_exp(uint32_t f8_src_fmt) {
    return (f8_src_fmt == kBf8Fmt) ? 0x1f : 0xf;
}

__device__ __forceinline__ uint8_t pack_fp8(uint32_t s, int32_t e, uint32_t m, uint32_t f8_src_fmt) {
    if (f8_src_fmt == kBf8Fmt) {
        return static_cast<uint8_t>((s << 7) | ((static_cast<uint32_t>(e) & 0x1fu) << 2) | (m & 0x3u));
    }
    return static_cast<uint8_t>((s << 7) | ((static_cast<uint32_t>(e) & 0xfu) << 3) | (m & 0x7u));
}

// round_ieee_8 — round to nearest even (poc_kl common_mi300_fp8.h)
__device__ __forceinline__ uint32_t round_ieee_8(uint32_t mant,
                                                 uint32_t guard,
                                                 uint32_t round,
                                                 uint32_t sticky,
                                                 uint32_t sign,
                                                 int32_t& exp,
                                                 uint32_t f8_fmt) {
    (void)sign;
    const uint32_t mant_bits = fp8_mant_bits(f8_fmt);
    const uint32_t max_mant = (1u << mant_bits) - 1u;
    uint32_t result = mant;

    if (exp == 1 && result == max_mant && guard && round == 0) {
        result = 0;
    }

    if (guard && ((mant & 1u) || round || sticky)) {
        result++;
    }

    const uint32_t shift = mant_bits + 1u;
    if (result >> shift) {
        result >>= 1;
        exp++;
    }

    if (exp <= 0 || result == 0u) {
        exp = 0;
        if (result || guard || round || sticky) {
            result = 0;
        }
    } else if (exp == 1 && (result >> mant_bits) == 0u) {
        exp = 0;
        result = 0;
    }

    return result;
}

__device__ __forceinline__ uint32_t fp8_to_f32_bits(uint8_t in,
                                                    uint32_t f8_src_fmt = kFp8Fmt,
                                                    bool f8_bias = kFp8Bias) {
    const uint32_t ss = fp8_sign(in);
    const uint32_t ee =
        (f8_src_fmt == kBf8Fmt) ? ((in >> 2) & 0x1fu) : fp8_exp_e4m3(in);
    uint32_t m = (f8_src_fmt == kBf8Fmt) ? (in & 0x3u) : fp8_man_e4m3(in);
    m <<= (f8_src_fmt == kBf8Fmt) ? 21u : 20u;

    uint32_t e = ee;
    const uint32_t b = fp8_bias(f8_src_fmt, f8_bias);
    e += (127u - b);
    const uint32_t s = ss;

    if (f8_bias) {
        if (ee == 0u) {
            if (m == 0u) {
                return (ss == 0u) ? 0u : 0x7fffffffu;
            }
            e++;
            while ((m & 0x800000u) == 0u) {
                m <<= 1;
                e--;
            }
            m &= 0x7fffffu;
        }
    } else {
        if ((ee == 0x1fu && f8_src_fmt == kBf8Fmt) ||
            (ee == 0xfu && f8_src_fmt == kFp8Fmt)) {
            return (m != 0u) ? 0x7fffffffu : (s ? 0xff800000u : 0x7f800000u);
        }
        if (ee == 0u) {
            if (m == 0u) {
                return s ? 0x80000000u : 0u;
            }
            e++;
            while ((m & 0x800000u) == 0u) {
                m <<= 1;
                e--;
            }
            m &= 0x7fffffu;
        }
    }
    m |= (e << 23);
    m |= (s << 31);
    return m;
}

__device__ __forceinline__ float fp8_to_float(uint8_t in,
                                              uint32_t f8_src_fmt = kFp8Fmt,
                                              bool f8_bias = kFp8Bias) {
    const uint32_t bits = fp8_to_f32_bits(in, f8_src_fmt, f8_bias);
    float out;
    __builtin_memcpy(&out, &bits, sizeof(out));
    return out;
}

__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t v) { return fp8_to_float(v); }

// f32_to_f8 — matches poc_kl common_mi300_fp8.h (non-stochastic).
__device__ __forceinline__ uint8_t f32_to_f8_bits(uint32_t src_a,
                                                  uint32_t f8_src_fmt = kFp8Fmt,
                                                  bool f8_bias = kFp8Bias) {
    bool sticky = false;
    const uint32_t ss = f32_sign(src_a);
    const uint32_t ee = f32_bexp(src_a);
    uint32_t mm = f32_man(src_a);

    if (ee == 0xffu) {
        return 0x80u;
    }
    if (ee == 0u && mm == 0u) {
        return 0u;
    }

    const uint32_t b = fp8_bias(f8_src_fmt, f8_bias);
    int32_t e = static_cast<int32_t>(ee) - 127 + static_cast<int32_t>(b);
    if (ee != 0u) {
        mm |= 0x800000u;
    }

    const uint32_t mbits = fp8_mant_bits(f8_src_fmt);
    const uint32_t mshift = 23u - mbits - 2u;
    uint32_t m = (mm >> mshift);
    const uint32_t mmask = (1u << mshift) - 1u;
    sticky = (mm & mmask) != 0u;

    const int32_t max_exp = fp8_max_exp(f8_src_fmt);
    if (e < 1) {
        int32_t shift = 1 - e;
        if (shift > max_exp) {
            shift = max_exp;
        }
        const uint32_t mask = (1u << shift) - 1u;
        sticky = sticky || ((m & mask) != 0u);
        m >>= shift;
        if (shift > max_exp) {
            m = 0;
        }
        e = 1;
    }

    m = round_ieee_8(m >> 2, m & 2u, m & 1u, sticky ? 1u : 0u, ss, e, f8_src_fmt);
    uint8_t result = pack_fp8(ss, e, m, f8_src_fmt);
    if (f8_src_fmt == kFp8Fmt && result == 0x80u) {
        result = 0u;
    }
    return result;
}

__device__ __forceinline__ uint8_t float_to_fp8_e4m3_bias8(float x) {
    const uint32_t bits = __float_as_uint(x);
    return f32_to_f8_bits(bits, kFp8Fmt, kFp8Bias);
}

}  // namespace pa_decode
