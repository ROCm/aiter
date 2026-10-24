// SPDX-License-Identifier: MIT
#pragma once
#include <cstdint>
#ifdef __HIPCC__
#define ORDER_INLINE __host__ __device__ __forceinline__
#else
#define ORDER_INLINE inline __attribute__((always_inline))
#endif
struct OrderIds
{
    uint32_t query, part, head;
};
template <unsigned S, unsigned Order, unsigned Heads = 2>
ORDER_INLINE OrderIds order_ids(uint32_t x, uint32_t y, uint32_t z, uint32_t q)
{
    static_assert(Order <= 3);
    if constexpr(Order == 0)
        return {x, y, z};
    uint32_t work = x + q * (y + S * z);
    if constexpr(Order >= 2)
    {
        const uint32_t total = q * S * Heads, per = (total + 7) / 8;
        const uint32_t tall = (total % 8) ? total % 8 : 8;
        const uint32_t xcd = work % 8, local = work / 8;
        work = xcd < tall ? xcd * per + local : tall * per + (xcd - tall) * (per - 1) + local;
    }
    if constexpr(Order == 3 && S > 1)
    {
        const uint32_t chunk = (q + 7) / 8, per = chunk * S * Heads;
        const uint32_t block = work / per, i = work % per;
        const uint32_t remaining = q - block * chunk;
        const uint32_t rows      = remaining < chunk ? remaining : chunk;
        const uint32_t r         = i % (rows * Heads);
        return {block * chunk + r / Heads, i / (rows * Heads), r % Heads};
    }
    return {work / (S * Heads), (work / Heads) % S, work % Heads};
}
#undef ORDER_INLINE
