// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include <cstdint>

//
// Platform detection
//
#if defined(__HIPCC__)
#define AITER_MXFP8_PLATFORM_HIP 1
#else
#define AITER_MXFP8_PLATFORM_HIP 0
#endif

//
// Host/Device dispatch macros
//
#if AITER_MXFP8_PLATFORM_HIP
#define AITER_MXFP8_HOST_DEVICE inline __host__ __device__
#define AITER_MXFP8_DEVICE inline __device__
#else
#define AITER_MXFP8_HOST_DEVICE inline
#define AITER_MXFP8_DEVICE inline
#endif

// Compile-time device-side flag
#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__
#define AITER_MXFP8_DEVICE_COMPILE 1
#else
#define AITER_MXFP8_DEVICE_COMPILE 0
#endif

//
// Universal warp size constant (AMD = 64)
//
namespace aiter {
constexpr int THREADS_PER_WARP      = 64;
constexpr int MAX_THREADS_PER_BLOCK = 1024; // TODO: ?
} // namespace aiter
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include <cstdint>
#include <hip/hip_runtime.h>

namespace aiter {

enum class GPUArch { GFX942, GFX950, UNKNOWN };

inline GPUArch get_current_arch() {
    static GPUArch cached_arch = []() -> GPUArch {
        hipDeviceProp_t prop;
        hipError_t      err = hipGetDeviceProperties(&prop, 0);
        if (err != hipSuccess) {
            return GPUArch::UNKNOWN;
        }
        if (prop.major == 9 && prop.minor == 4)
            return GPUArch::GFX942;
        if (prop.major == 9 && prop.minor == 5)
            return GPUArch::GFX950;
        return GPUArch::UNKNOWN;
    }();
    return cached_arch;
}

inline bool is_gfx950() {
    return get_current_arch() == GPUArch::GFX950;
}

inline bool is_gfx942() {
    return get_current_arch() == GPUArch::GFX942;
}

} // namespace aiter
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.
#pragma once

#include <cstdint>
#include <cstring>

namespace aiter {

AITER_MXFP8_HOST_DEVICE float fp32_from_bits(uint32_t bits) {
#if AITER_MXFP8_DEVICE_COMPILE
    return __uint_as_float(bits);
#else
    float f;
    memcpy(&f, &bits, sizeof(float));
    return f;
#endif
}

AITER_MXFP8_HOST_DEVICE uint32_t fp32_to_bits(float f) {
#if AITER_MXFP8_DEVICE_COMPILE
    return __float_as_uint(f);
#else
    uint32_t bits;
    memcpy(&bits, &f, sizeof(uint32_t));
    return bits;
#endif
}

} // namespace aiter
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include <optional>
#include <stdexcept>
#include <string>

// #include <hip/hip_fp4.h>  // STUBBED: ROCm 7.2.3 fp4 header bug; fp4 unused by mxfp8 kernel

namespace aiter {

using float4x2_e2m1_t = unsigned char;  // STUBBED (fp4 unused)

} // namespace aiter
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include <optional>
#include <stdexcept>
#include <string>

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_version.h>

namespace aiter {

enum class Float8Format { FNUZ, OCP };

inline Float8Format current_fp8_format() {
#if AITER_MXFP8_DEVICE_COMPILE
    return Float8Format::FNUZ; // dummy
#else
    static Float8Format fmt = [] { return is_gfx950() ? Float8Format::OCP : Float8Format::FNUZ; }();
    return fmt;
#endif
}

AITER_MXFP8_HOST_DEVICE bool is_fp8_fnuz() {
#if AITER_MXFP8_DEVICE_COMPILE
#if defined(__gfx950__)
    return false; // gfx950 OCP
#else
    return true;
#endif
#else
    return current_fp8_format() == Float8Format::FNUZ;
#endif
}

struct float8_e4m3_t {

#if AITER_MXFP8_DEVICE_COMPILE
#if defined(__gfx950__)
    using storage_t = __hip_fp8_e4m3; // OCP on gfx950
#else
    using storage_t = __hip_fp8_e4m3_fnuz; // FNUZ on others
#endif
    storage_t val;
#else // host side – keep both encodings
    union {
        __hip_fp8_e4m3_fnuz fnuz;
        __hip_fp8_e4m3      ocp;
    } u{};
#endif

    AITER_MXFP8_HOST_DEVICE float8_e4m3_t() = default;
    //------------------------------------------------------------------
    // converters
    //------------------------------------------------------------------

    //---------------  from bits  -----------------
    AITER_MXFP8_HOST_DEVICE static float8_e4m3_t from_bits(uint8_t bits) {
        float8_e4m3_t x;
#if AITER_MXFP8_DEVICE_COMPILE
        *reinterpret_cast<uint8_t *>(&x.val) = bits;
#else
        if (is_fp8_fnuz())
            *reinterpret_cast<uint8_t *>(&x.u.fnuz) = bits;
        else
            *reinterpret_cast<uint8_t *>(&x.u.ocp) = bits;
#endif
        return x;
    }

    //---------------  float32  -----------------
    AITER_MXFP8_HOST_DEVICE float8_e4m3_t(float f) { *this = f; }

    AITER_MXFP8_HOST_DEVICE float8_e4m3_t &operator=(float f) {
#if AITER_MXFP8_DEVICE_COMPILE
        val = static_cast<storage_t>(f);
#else
        if (is_fp8_fnuz())
            u.fnuz = static_cast<__hip_fp8_e4m3_fnuz>(f);
        else
            u.ocp = static_cast<__hip_fp8_e4m3>(f);
#endif
        return *this;
    }

    AITER_MXFP8_HOST_DEVICE operator float() const {
#if AITER_MXFP8_DEVICE_COMPILE
        return static_cast<float>(val);
#else
        return is_fp8_fnuz() ? static_cast<float>(u.fnuz) : static_cast<float>(u.ocp);
#endif
    }

    //---------------  half  -----------------
    // TODO: Opt CVT
    AITER_MXFP8_HOST_DEVICE
    float8_e4m3_t(const half h) { *this = static_cast<float>(h); }

    AITER_MXFP8_HOST_DEVICE float8_e4m3_t &operator=(const half h) {
        *this = static_cast<float>(h);
        return *this;
    }

    AITER_MXFP8_HOST_DEVICE
    operator half() const { return half(float(*this)); }

    //---------------  bfloat16  -----------------
    // TODO: Opt CVT
    AITER_MXFP8_HOST_DEVICE
    float8_e4m3_t(const hip_bfloat16 bf) { *this = static_cast<float>(bf); }

    AITER_MXFP8_HOST_DEVICE
    float8_e4m3_t &operator=(const hip_bfloat16 bf) {
        *this = static_cast<float>(bf);
        return *this;
    }

    AITER_MXFP8_HOST_DEVICE
    operator hip_bfloat16() const { return hip_bfloat16(float(*this)); }

    //------------------------------------------------------------------
    // Basic arithmetic
    //------------------------------------------------------------------
    AITER_MXFP8_HOST_DEVICE friend float8_e4m3_t operator+(const float8_e4m3_t &lhs,
                                                            const float8_e4m3_t &rhs) {
        return float8_e4m3_t(float(lhs) + float(rhs));
    }

    AITER_MXFP8_HOST_DEVICE friend float8_e4m3_t operator-(const float8_e4m3_t &lhs,
                                                            const float8_e4m3_t &rhs) {
        return float8_e4m3_t(float(lhs) - float(rhs));
    }

    AITER_MXFP8_HOST_DEVICE friend float8_e4m3_t operator*(const float8_e4m3_t &lhs,
                                                            const float8_e4m3_t &rhs) {
        return float8_e4m3_t(float(lhs) * float(rhs));
    }

    AITER_MXFP8_HOST_DEVICE friend float8_e4m3_t operator/(const float8_e4m3_t &lhs,
                                                            const float8_e4m3_t &rhs) {
        return float8_e4m3_t(float(lhs) / float(rhs));
    }

    //------------------------------------------------------------------
    // In-place basic arithmetic
    //------------------------------------------------------------------
    AITER_MXFP8_HOST_DEVICE float8_e4m3_t &operator+=(const float8_e4m3_t &rhs) {
        *this = *this + rhs;
        return *this;
    }

    AITER_MXFP8_HOST_DEVICE float8_e4m3_t &operator-=(const float8_e4m3_t &rhs) {
        *this = *this - rhs;
        return *this;
    }

    AITER_MXFP8_HOST_DEVICE float8_e4m3_t &operator*=(const float8_e4m3_t &rhs) {
        *this = *this * rhs;
        return *this;
    }

    AITER_MXFP8_HOST_DEVICE float8_e4m3_t &operator/=(const float8_e4m3_t &rhs) {
        *this = *this / rhs;
        return *this;
    }
};
static_assert(sizeof(float8_e4m3_t) == 1, "float8_e4m3_t must be 1 byte");
static_assert(alignof(float8_e4m3_t) == 1);
static_assert(std::is_trivially_copyable_v<float8_e4m3_t>);

struct float8_e5m2_t {

#if AITER_MXFP8_DEVICE_COMPILE
#if defined(__gfx950__)
    using storage_t = __hip_fp8_e5m2; // OCP on gfx950
#else
    using storage_t = __hip_fp8_e5m2_fnuz; // FNUZ on others
#endif
    storage_t val;
#else // host side – keep both encodings
    union {
        __hip_fp8_e5m2_fnuz fnuz;
        __hip_fp8_e5m2      ocp;
    } u{};
#endif

    AITER_MXFP8_HOST_DEVICE float8_e5m2_t() = default;
    //------------------------------------------------------------------
    // converters
    //------------------------------------------------------------------

    //---------------  from bits  -----------------
    AITER_MXFP8_HOST_DEVICE static float8_e5m2_t from_bits(uint8_t bits) {
        float8_e5m2_t x;
#if AITER_MXFP8_DEVICE_COMPILE
        *reinterpret_cast<uint8_t *>(&x.val) = bits;
#else
        if (is_fp8_fnuz())
            *reinterpret_cast<uint8_t *>(&x.u.fnuz) = bits;
        else
            *reinterpret_cast<uint8_t *>(&x.u.ocp) = bits;
#endif
        return x;
    }

    //---------------  float32  -----------------
    AITER_MXFP8_HOST_DEVICE float8_e5m2_t(float f) { *this = f; }

    AITER_MXFP8_HOST_DEVICE float8_e5m2_t &operator=(float f) {
#if AITER_MXFP8_DEVICE_COMPILE
        val = static_cast<storage_t>(f);
#else
        if (is_fp8_fnuz())
            u.fnuz = static_cast<__hip_fp8_e5m2_fnuz>(f);
        else
            u.ocp = static_cast<__hip_fp8_e5m2>(f);
#endif
        return *this;
    }

    AITER_MXFP8_HOST_DEVICE operator float() const {
#if AITER_MXFP8_DEVICE_COMPILE
        return static_cast<float>(val);
#else
        return is_fp8_fnuz() ? static_cast<float>(u.fnuz) : static_cast<float>(u.ocp);
#endif
    }

    //---------------  half  -----------------
    // TODO: Opt CVT
    AITER_MXFP8_HOST_DEVICE
    float8_e5m2_t(const half h) { *this = static_cast<float>(h); }

    AITER_MXFP8_HOST_DEVICE float8_e5m2_t &operator=(const half h) {
        *this = static_cast<float>(h);
        return *this;
    }

    AITER_MXFP8_HOST_DEVICE
    operator half() const { return half(float(*this)); }

    //---------------  bfloat16  -----------------
    // TODO: Opt CVT
    AITER_MXFP8_HOST_DEVICE
    float8_e5m2_t(const hip_bfloat16 bf) { *this = static_cast<float>(bf); }

    AITER_MXFP8_HOST_DEVICE
    float8_e5m2_t &operator=(const hip_bfloat16 bf) {
        *this = static_cast<float>(bf);
        return *this;
    }

    AITER_MXFP8_HOST_DEVICE
    operator hip_bfloat16() const { return hip_bfloat16(float(*this)); }

    //------------------------------------------------------------------
    // Basic arithmetic
    //------------------------------------------------------------------
    AITER_MXFP8_HOST_DEVICE friend float8_e5m2_t operator+(const float8_e5m2_t &lhs,
                                                            const float8_e5m2_t &rhs) {
        return float8_e5m2_t(float(lhs) + float(rhs));
    }

    AITER_MXFP8_HOST_DEVICE friend float8_e5m2_t operator-(const float8_e5m2_t &lhs,
                                                            const float8_e5m2_t &rhs) {
        return float8_e5m2_t(float(lhs) - float(rhs));
    }

    AITER_MXFP8_HOST_DEVICE friend float8_e5m2_t operator*(const float8_e5m2_t &lhs,
                                                            const float8_e5m2_t &rhs) {
        return float8_e5m2_t(float(lhs) * float(rhs));
    }

    AITER_MXFP8_HOST_DEVICE friend float8_e5m2_t operator/(const float8_e5m2_t &lhs,
                                                            const float8_e5m2_t &rhs) {
        return float8_e5m2_t(float(lhs) / float(rhs));
    }

    //------------------------------------------------------------------
    // In-place basic arithmetic
    //------------------------------------------------------------------
    AITER_MXFP8_HOST_DEVICE float8_e5m2_t &operator+=(const float8_e5m2_t &rhs) {
        *this = *this + rhs;
        return *this;
    }

    AITER_MXFP8_HOST_DEVICE float8_e5m2_t &operator-=(const float8_e5m2_t &rhs) {
        *this = *this - rhs;
        return *this;
    }

    AITER_MXFP8_HOST_DEVICE float8_e5m2_t &operator*=(const float8_e5m2_t &rhs) {
        *this = *this * rhs;
        return *this;
    }

    AITER_MXFP8_HOST_DEVICE float8_e5m2_t &operator/=(const float8_e5m2_t &rhs) {
        *this = *this / rhs;
        return *this;
    }
};
static_assert(sizeof(float8_e5m2_t) == 1, "float8_e5m2_t must be 1 byte");
static_assert(alignof(float8_e5m2_t) == 1);
static_assert(std::is_trivially_copyable_v<float8_e5m2_t>);

struct float8_e8m0_t {
    using storage_t = uint8_t;
    storage_t val;

    AITER_MXFP8_HOST_DEVICE float8_e8m0_t() = default;

    //---------------  from bits  -----------------
    AITER_MXFP8_HOST_DEVICE static float8_e8m0_t from_bits(uint8_t bits) {
        float8_e8m0_t x;
        x.val = bits;
        return x;
    }

    //---------------  float32  -----------------
    AITER_MXFP8_HOST_DEVICE float8_e8m0_t(float f) { *this = f; }

    AITER_MXFP8_HOST_DEVICE float8_e8m0_t &operator=(float f) {
        uint32_t f_bits   = fp32_to_bits(f);
        uint32_t exponent = (f_bits >> 23) & 0b11111111;
        if (exponent == 0b11111111) { // NaN
            val = exponent;
        } else {
            // guard bit - bit 23, or 22 zero-indexed
            uint8_t g = (f_bits & 0x400000) > 0;
            // round bit - bit 22, or 21 zero-indexed
            uint8_t r = (f_bits & 0x200000) > 0;
            // sticky bit - bits 21 to 1, or 20 to 0 zero-indexed
            uint8_t s = (f_bits & 0x1FFFFF) > 0;
            // in casting to e8m0, LSB is the implied mantissa bit. It equals to 0 if the
            // original float32 is denormal, and to 1 if the original float32 is normal.
            uint8_t lsb = exponent > 0;

            bool round_up = false;
            // if g == 0, round down (no-op)
            if (g == 1) {
                if ((r == 1) || (s == 1)) {
                    round_up = true;
                } else {
                    if (lsb == 1) {
                        round_up = true;
                    }
                }
            }
            val = round_up ? exponent + 1 : exponent;
        }
        return *this;
    }

    AITER_MXFP8_HOST_DEVICE operator float() const {
        if (val == 0) {
            return fp32_from_bits(0x00400000);
        }
        if (val == 0b11111111) {
            return fp32_from_bits(0x7f800001);
        }
        uint32_t res = val << 23;
        return fp32_from_bits(res);
    }
};
static_assert(sizeof(float8_e8m0_t) == 1, "float8_e8m0_t must be 1 byte");
static_assert(alignof(float8_e8m0_t) == 1);
static_assert(std::is_trivially_copyable_v<float8_e8m0_t>);

} // namespace aiter

namespace std {

using aiter::float8_e4m3_t;
using aiter::float8_e5m2_t;

template <> class numeric_limits<float8_e4m3_t> {
public:
    static constexpr bool is_specialized = true;
    static constexpr bool has_infinity   = false;
    static constexpr bool has_quiet_NaN  = true;

    AITER_MXFP8_HOST_DEVICE static float8_e4m3_t min() { return float8_e4m3_t::from_bits(0x08); }

    AITER_MXFP8_HOST_DEVICE static float8_e4m3_t max() {
#if AITER_MXFP8_DEVICE_COMPILE
#if defined(__gfx950__)
        return float8_e4m3_t::from_bits(0x7E);
#else
        return float8_e4m3_t::from_bits(0x7F);
#endif
#else
        return aiter::is_fp8_fnuz() ? float8_e4m3_t::from_bits(0x7F)
                                           : float8_e4m3_t::from_bits(0x7E);
#endif
    }

    AITER_MXFP8_HOST_DEVICE static float8_e4m3_t lowest() { return float8_e4m3_t(-float(max())); }

    // E4M3 has no INF: by specification, has_infinity = false;
    // Here we defensively return max()
    AITER_MXFP8_HOST_DEVICE static float8_e4m3_t infinity() { return max(); }

    // NaN
    // OCP : 0x7F
    // FNUZ: 0x80
    AITER_MXFP8_HOST_DEVICE static float8_e4m3_t quiet_NaN() {
#if AITER_MXFP8_DEVICE_COMPILE
#if defined(__gfx950__)
        return float8_e4m3_t::from_bits(0x7F);
#else
        return float8_e4m3_t::from_bits(0x80);
#endif
#else
        return aiter::is_fp8_fnuz() ? float8_e4m3_t::from_bits(0x80)
                                           : float8_e4m3_t::from_bits(0x7F);
#endif
    }
};

template <> class numeric_limits<float8_e5m2_t> {
public:
    static constexpr bool is_specialized = true;
#if AITER_MXFP8_DEVICE_COMPILE
#if defined(__gfx950__)
    static constexpr bool has_infinity = true;
#else
    static constexpr bool has_infinity = false;
#endif
#else
    // Host: cannot determine at compile time whether this is OCP or FNUZ.
    // Therefore, conservatively set has_infinity = false.
    // Generic code should not rely on has_infinity; instead call infinity(),
    // which will return a true Inf in OCP and max() in FNUZ.
    static constexpr bool has_infinity = false;
#endif
    static constexpr bool has_quiet_NaN = true;

    AITER_MXFP8_HOST_DEVICE static float8_e5m2_t min() { return float8_e5m2_t::from_bits(0x04); }

    AITER_MXFP8_HOST_DEVICE static float8_e5m2_t max() {
#if AITER_MXFP8_DEVICE_COMPILE
#if defined(__gfx950__)
        return float8_e5m2_t::from_bits(0x7B);
#else
        return float8_e5m2_t::from_bits(0x7F);
#endif
#else
        return aiter::is_fp8_fnuz() ? float8_e5m2_t::from_bits(0x7F)
                                           : float8_e5m2_t::from_bits(0x7B);
#endif
    }

    AITER_MXFP8_HOST_DEVICE static float8_e5m2_t lowest() { return float8_e5m2_t(-float(max())); }

    AITER_MXFP8_HOST_DEVICE static float8_e5m2_t infinity() {
#if AITER_MXFP8_DEVICE_COMPILE
#if defined(__gfx950__)
        return float8_e5m2_t::from_bits(0x7C);
#else
        return max();
#endif
#else
        return aiter::is_fp8_fnuz() ? max() : float8_e5m2_t::from_bits(0x7C);
#endif
    }

    AITER_MXFP8_HOST_DEVICE static float8_e5m2_t quiet_NaN() {
#if AITER_MXFP8_DEVICE_COMPILE
#if defined(__gfx950__)
        return float8_e5m2_t::from_bits(0x7D);
#else
        return float8_e5m2_t::from_bits(0x80);
#endif
#else
        return aiter::is_fp8_fnuz() ? float8_e5m2_t::from_bits(0x80)
                                           : float8_e5m2_t::from_bits(0x7D);
#endif
    }
};

} // namespace std
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once
#include <cstdint>

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

// https://rocm.docs.amd.com/projects/HIP/en/docs-develop/reference/low_fp_types.html#
namespace aiter {

namespace dtype {

using float64       = double;
using float32       = float;
using float16       = half;
using bfloat16      = hip_bfloat16;
using float8_e4m3   = float8_e4m3_t;
using float8_e5m2   = float8_e5m2_t;
using float8_e8m0   = float8_e8m0_t;
using float4x2_e2m1 = float4x2_e2m1_t;

using int64 = int64_t;
using int32 = int32_t;
using int16 = int16_t;
using int8  = int8_t;

using uint64 = uint64_t;
using uint32 = uint32_t;
using uint16 = uint16_t;
using uint8  = uint8_t;

// Vector types (GCC vector extension, for inline asm / builtins)
using float32x4 = __attribute__((vector_size(16))) float;
using int32x4   = __attribute__((vector_size(16))) int;
using int32x8   = __attribute__((vector_size(32))) int;

} // namespace dtype

} // namespace aiter
