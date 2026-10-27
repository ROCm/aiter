// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

// The fp32 order the radix and the sampled top-k selectors rank by, as unsigned
// keys: -0.0 just below +0.0, and a NaN by its sign -- a negative one below
// -inf -- unless NAN_HIGH, which ranks a NaN of either sign above +inf, as
// topk_select does. Both selectors' keys live here, and the checks at the end
// hold them to one order, so a change to either reaches the other or fails to
// compile.

#include <cstdint>
#include <hip/hip_runtime.h>

// IEEE-754 fp32 -> monotone uint32, the "distort" trick from DeepSelect: a
// larger float gets a larger key, a positive NaN lands above +INF, a negative
// one below -INF.
__host__ __device__ __forceinline__ constexpr uint32_t fp32_to_sortable_bits(uint32_t u)
{
    return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}

__host__ __device__ __forceinline__ constexpr uint32_t sortable_to_fp32_bits(uint32_t s)
{
    return (s & 0x80000000u) ? (s ^ 0x80000000u) : ~s;
}

// Under fp32_to_sortable_bits the negative NaNs are exactly the keys
// [0, NAN_HIGH_SHIFT) and -inf is NAN_HIGH_SHIFT, so subtracting NAN_HIGH_SHIFT
// rotates them to the top of the key space, above the positive NaNs, which are
// already above +inf. Every other key moves down by the same amount, so their
// order is unchanged and the map stays a bijection with an exact inverse. One
// add per key: the sampled fallback's full-row passes are bound by VALU work per
// key, and a compare-and-select there cost 20% more phase_c on a falling-back
// row.
constexpr uint32_t NAN_HIGH_SHIFT = 0x007FFFFFu;

template <bool NAN_HIGH>
__host__ __device__ __forceinline__ constexpr uint32_t sortable_key_bits(uint32_t u)
{
    return fp32_to_sortable_bits(u) - (NAN_HIGH ? NAN_HIGH_SHIFT : 0u);
}

// The radix passes take the smallest key first, so they rank by the same order
// reversed. They also read a bound back as a float (radix_key_to_fp32_bits), so
// NAN_HIGH has to leave every other key where it is: a NaN takes its magnitude,
// the key of a positive NaN, rather than shifting the whole key space.
template <bool NAN_HIGH>
__host__ __device__ __forceinline__ constexpr uint32_t radix_key_bits(uint32_t u)
{
    if(NAN_HIGH)
    {
        uint32_t const magnitude = u & 0x7fffffffu;
        u                        = magnitude > 0x7f800000u ? magnitude : u;
    }
    uint32_t const mask = (u >> 31) ? 0u : 0x7fffffffu;
    return u ^ mask;
}

// The inverse of radix_key_bits<false>, which is also its own inverse.
__host__ __device__ __forceinline__ constexpr uint32_t radix_key_to_fp32_bits(uint32_t key)
{
    return (key >> 31) ? key : (key ^ 0x7fffffffu);
}

namespace topk_fp32_key_detail {
// Zeros, ones, infinities, the smallest normal and subnormal, and the largest
// finite value, of both signs; then quiet and signalling NaNs of both signs, a
// payload, and all-ones.
constexpr uint32_t kOrdered[] = {0x00000000u,
                                 0x80000000u,
                                 0x3f800000u,
                                 0xbf800000u,
                                 0x7f800000u,
                                 0xff800000u,
                                 0x00800000u,
                                 0x80800000u,
                                 0x00000001u,
                                 0x80000001u,
                                 0x7f7fffffu,
                                 0xff7fffffu};
constexpr uint32_t kNaNs[]    = {
    0x7fc00000u, 0xffc00000u, 0x7f800001u, 0xff800001u, 0x7fc00123u, 0xffffffffu};
constexpr uint32_t kPosInf = 0x7f800000u, kNegInf = 0xff800000u;

// Each key decodes back to the bits it came from, a NaN's payload included.
constexpr bool round_trips(uint32_t u)
{
    return radix_key_to_fp32_bits(radix_key_bits<false>(u)) == u &&
           radix_key_bits<false>(radix_key_bits<false>(u)) == u &&
           sortable_to_fp32_bits(fp32_to_sortable_bits(u)) == u &&
           sortable_key_bits<true>(u) + NAN_HIGH_SHIFT == fp32_to_sortable_bits(u);
}

constexpr bool keys_agree()
{
    for(uint32_t u : kOrdered)
    {
        // One order, reversed for radix; NAN_HIGH moves nothing but NaNs in it.
        if(radix_key_bits<false>(u) != static_cast<uint32_t>(~fp32_to_sortable_bits(u)) ||
           radix_key_bits<true>(u) != radix_key_bits<false>(u) || !round_trips(u))
            return false;
    }
    for(uint32_t u : kNaNs)
    {
        bool const negative = u >> 31;
        // NAN_HIGH: above +inf in both. Otherwise by sign, in both.
        if(!(radix_key_bits<true>(u) < radix_key_bits<true>(kPosInf)) ||
           !(sortable_key_bits<true>(u) > sortable_key_bits<true>(kPosInf)))
            return false;
        if(negative != (radix_key_bits<false>(u) > radix_key_bits<false>(kNegInf)) ||
           negative != (fp32_to_sortable_bits(u) < fp32_to_sortable_bits(kNegInf)) ||
           !round_trips(u))
            return false;
    }
    return true;
}

static_assert(keys_agree(), "the radix and sampled fp32 keys must rank by one order");
} // namespace topk_fp32_key_detail
