// SPDX-License-Identifier: MIT
#ifndef KV_XOR
#define KV_XOR 0
#endif
#ifndef KV_PAD
#define KV_PAD 0
#endif
static_assert(KV_PAD >= 0 && KV_PAD % 16 == 0);
constexpr int KV_STRIDE = 1024 + KV_PAD;
__device__ __forceinline__ int kv_xor(int row)
{
    if constexpr(KV_XOR == 1)
        return ((row >> 2) & 3) * 16;
    if constexpr(KV_XOR == 2)
        return (((row >> 2) ^ (row >> 5)) & 3) * 16;
    return (row & 3) * 16;
}
template <int Rows>
__device__ __forceinline__ int address(int row, int col)
{
    constexpr int stride = Rows == 128 ? KV_STRIDE : 1024;
    int mask             = Rows == 128 ? kv_xor(row) : (row & 3) * 16;
    return ((col / 64) * (Rows / 16) + row / 16) * stride + (row % 16) * 64 + ((col % 64) ^ mask);
}
