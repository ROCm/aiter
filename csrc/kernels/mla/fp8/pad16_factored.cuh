// SPDX-License-Identifier: MIT
#pragma once
__host__ __device__ constexpr int padded_address(int row, int col)
{
    return ((col / 64) * 8 + row / 16) * 1040 + (row % 16) * 64 + ((col % 64) ^ ((row & 3) * 16));
}
__host__ __device__ constexpr int padded_pv_base(int lane, int wave, int tile)
{
    return (16 * wave + 2 * (lane / 16)) * 1040 + ((lane % 16) / 2) * 64 +
           ((8 * (lane & 1)) ^ (16 * (((lane % 16) / 2) & 3)) ^ (16 * (tile & 3)));
}
__host__ __device__ constexpr int padded_pv_offset(int tile, int piece)
{
    return (8 * (tile / 4) + piece / 2) * 1040 + 512 * (piece & 1);
}
__host__ __device__ constexpr int padded_qk_base(int lane, int wave, int vec)
{
    return (8 * ((lane / 16) / 2) + 2 * wave) * 1040 + (lane % 16) * 64 +
           ((((lane / 16) % 2) * 32 + vec) ^ (((lane % 16) & 3) * 16));
}
template <int K, int Half>
__device__ __forceinline__ I8 padded_k_fragment(const uint8_t* kvs, int lane, int wave)
{
    const uint8_t* a     = kvs + padded_qk_base(lane, wave, 0);
    const uint8_t* b     = kvs + padded_qk_base(lane, wave, 16);
    constexpr int Offset = (16 * K + Half) * 1040;
    U4 lo = *reinterpret_cast<const U4*>(a + Offset), hi = *reinterpret_cast<const U4*>(b + Offset);
    I8 out;
#pragma unroll
    for(int j = 0; j < 4; ++j)
    {
        out[j]     = lo[j];
        out[j + 4] = hi[j];
    }
    return out;
}
template <int K>
__device__ __forceinline__ void
padded_qk_step(const uint8_t* qs, const uint8_t* kvs, int head, int group, int wave, F4& a, F4& b)
{
    const I8 q     = fragment<16>(qs, head, K * 128 + group * 32);
    const int lane = threadIdx.x & 63;
    a              = mma(padded_k_fragment<K, 0>(kvs, lane, wave), q, a);
    b              = mma(padded_k_fragment<K, 1>(kvs, lane, wave), q, b);
    if constexpr(K < 3)
        padded_qk_step<K + 1>(qs, kvs, head, group, wave, a, b);
}
template <int Tile>
__device__ __forceinline__ I8 padded_v_fragment(const uint8_t* kvs, int wave)
{
    const int lane = threadIdx.x & 63;
    const uint32_t base =
        uint32_t(reinterpret_cast<uintptr_t>(kvs + padded_pv_base(lane, wave, Tile)));
    I2 r0, r1, r2, r3;
    asm volatile("ds_read_b64_tr_b8 %0, %1 offset:%2"
                 : "=v"(r0)
                 : "v"(base), "n"(padded_pv_offset(Tile, 0))
                 : "memory");
    asm volatile("ds_read_b64_tr_b8 %0, %1 offset:%2"
                 : "=v"(r1)
                 : "v"(base), "n"(padded_pv_offset(Tile, 1))
                 : "memory");
    asm volatile("ds_read_b64_tr_b8 %0, %1 offset:%2"
                 : "=v"(r2)
                 : "v"(base), "n"(padded_pv_offset(Tile, 2))
                 : "memory");
    asm volatile("ds_read_b64_tr_b8 %0, %1 offset:%2"
                 : "=v"(r3)
                 : "v"(base), "n"(padded_pv_offset(Tile, 3))
                 : "memory");
    I8 x = {r0[0], r0[1], r1[0], r1[1], r2[0], r2[1], r3[0], r3[1]};
    asm volatile("s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)" : "+v"(x) : : "memory");
    return x;
}
template <int Tile>
__device__ __forceinline__ void padded_pv_step(const uint8_t* kvs,
                                               I8 prob,
                                               int wave,
                                               int iteration,
                                               float alpha,
                                               float beta,
                                               float scale,
                                               F4 (&merged)[8])
{
    I8 v   = padded_v_fragment<Tile>(kvs, wave);
    F4 out = mma(v, prob, F4{});
    out *= scale;
    if(iteration == 0)
        merged[Tile] = out;
    else
    {
#pragma unroll
        for(int j = 0; j < 4; ++j)
            merged[Tile][j] = fmaf(out[j], beta, merged[Tile][j] * alpha);
    }
    if constexpr(Tile < 7)
        padded_pv_step<Tile + 1>(kvs, prob, wave, iteration, alpha, beta, scale, merged);
}

// FP32 partial normalization. This deliberately changes FP8 P rounding.
template <int Tile, bool PVVGPR>
__device__ __forceinline__ void
padded_online_pv_step(const uint8_t* kvs, I8 prob, int wave, float alpha, F4 (&merged)[8])
{
    I8 v         = padded_v_fragment<Tile>(kvs, wave);
    merged[Tile] = online_pv_mma<PVVGPR>(v, prob, merged[Tile] * alpha);
    if constexpr(Tile < 7)
        padded_online_pv_step<Tile + 1, PVVGPR>(kvs, prob, wave, alpha, merged);
}
