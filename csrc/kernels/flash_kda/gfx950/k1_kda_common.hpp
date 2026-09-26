// Shared gfx950 K1 primitives.
//
// These helpers are deliberately architecture-private: the gfx942 and gfx950
// operators are compiled as separate translation units, while the host ABI
// stays in hip_common.hpp.
#pragma once

#include "mfma_gfx950.hpp"

namespace flashkda_hip::gfx950 {

// C = A @ B for row-major fp16 16x16 operands.  CDNA4 can transpose the B
// fragment while reading LDS, removing the sixteen-way strided access.  Keep
// the scalar fallback so the gfx950 TU can still be emitted into a fat binary.
__device__ __forceinline__ f32x4 gemm_std_f16_tr(
        const _Float16* __restrict__ a,
        const _Float16* __restrict__ b,
        int lane) {
    const int row = lane & 15;
    const int kb = (lane >> 4) * 4;
    f16x4 af;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        af[i] = a[row * 16 + kb + i];
#if defined(__gfx950__)
    using native_f16x4 = __fp16 __attribute__((ext_vector_type(4)));
    auto p = reinterpret_cast<__attribute__((address_space(3))) native_f16x4*>(
        (__attribute__((address_space(3))) __fp16*)(b + lane * 4));
    const native_f16x4 raw = __builtin_amdgcn_ds_read_tr16_b64_v4f16(p);
    f16x4 bf;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        bf[i] = raw[i];
#else
    f16x4 bf;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        bf[i] = b[(kb + i) * 16 + row];
#endif
    const f32x4 zero = {0.f, 0.f, 0.f, 0.f};
    return mfma_f16(af, bf, zero);
}

}  // namespace flashkda_hip::gfx950
