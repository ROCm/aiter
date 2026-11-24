// SPDX-License-Identifier: MIT
#pragma once
#ifdef __HIPCC__
#define LAYOUT_INLINE __host__ __device__ __forceinline__
#else
#define LAYOUT_INLINE inline __attribute__((always_inline))
#endif
template <int Rows, int Pad>
LAYOUT_INLINE int kv_address(int row, int col)
{
    static_assert(Pad >= 0 && Pad % 16 == 0);
    return ((col / 64) * (Rows / 16) + row / 16) * (1024 + Pad) + (row % 16) * 64 +
           ((col % 64) ^ ((row & 3) * 16));
}
#undef LAYOUT_INLINE
