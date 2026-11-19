// SPDX-License-Identifier: MIT
// Only the split partial storage changes; all arithmetic remains FP32.
using PF4 = float __attribute__((ext_vector_type(4)));
using PI2 = int __attribute__((ext_vector_type(2)));
template <bool NT = false>
__device__ __forceinline__ void partial_store(uint16_t* ptr, PF4 x)
{
    uint32_t lo, hi;
    asm("v_cvt_pk_bf16_f32 %0, %1, %2" : "=v"(lo) : "v"(x[0]), "v"(x[1]));
    asm("v_cvt_pk_bf16_f32 %0, %1, %2" : "=v"(hi) : "v"(x[2]), "v"(x[3]));
    PI2 v{int(lo), int(hi)};
    if constexpr(NT)
        __builtin_nontemporal_store(v, reinterpret_cast<PI2*>(ptr));
    else
        *reinterpret_cast<PI2*>(ptr) = v;
}
template <bool NT = false>
__device__ __forceinline__ PF4 partial_load(const uint16_t* ptr)
{
    const PI2* p = reinterpret_cast<const PI2*>(ptr);
    PI2 v;
    if constexpr(NT)
        v = __builtin_nontemporal_load(p);
    else
        v = *p;
    PF4 x;
#pragma unroll
    for(int i = 0; i < 4; ++i)
    {
        uint32_t bits = (uint32_t(v[i / 2]) >> ((i % 2) * 16)) << 16;
        x[i]          = __builtin_bit_cast(float, bits);
    }
    return x;
}
