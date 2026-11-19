// SPDX-License-Identifier: MIT
#pragma once
#include "params.cuh"
#include "partial_io.cuh"
#include "q_encode.cuh"
#include "softmax_lane.cuh"
#define STREAM_ONLINE 1
#define STREAM_QK_VGPR 1
#define STREAM_PV_VGPR 1
#define STREAM_PV_IMMEDIATE 1
#define HEAD32_ORDER 2
#define KV_CHUNK_PAD 16
#define BANK_QK_ADDRESS 1
#define BANK_PV_ADDRESS 0
namespace sparse_mla_fp8::online {

// SPDX-License-Identifier: MIT
// H64 model, CTA32 N128 FP8 MLA. Two waves per 16 heads share QK N and PV D.
#include "layout.cuh"
#include "order.cuh"
#ifndef KV_CHUNK_PAD
#define KV_CHUNK_PAD 0
#endif
constexpr int KV_CHUNK_STRIDE = 1024 + KV_CHUNK_PAD;
#ifndef BANK_QK_ADDRESS
#define BANK_QK_ADDRESS 0
#endif
#ifndef BANK_PV_ADDRESS
#define BANK_PV_ADDRESS 0
#endif
#ifndef HEAD32_ORDER
#define HEAD32_ORDER 0
#endif
using I8     = int __attribute__((ext_vector_type(8)));
using U4     = uint32_t __attribute__((ext_vector_type(4)));
using F4     = float __attribute__((ext_vector_type(4)));
using Params = sparse_mla_fp8::Params;
#ifndef STREAM_QK_VGPR
#define STREAM_QK_VGPR 0
#endif
#ifndef STREAM_PV_VGPR
#define STREAM_PV_VGPR 0
#endif
constexpr float LOG2E = 1.4426950408889634f, LN2 = .6931471805599453f;
__device__ __forceinline__ F4 mma(I8 a, I8 b, F4 c)
{
    return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, c, 0, 0, 0, 127, 0, 127);
}
// Same scaled MFMA and unit E8M0 exponent as the intrinsic. Explicit
// read/write VGPR accumulator prevents unnecessary AGPR/VGPR duplication.
__device__ __forceinline__ F4 mma_vgpr(I8 a, I8 b, F4 c)
{
    int scale = 127;
    // Table 38 input RAW wait remains inside asm after compiler operand moves.
    asm volatile(
        "s_nop 1\n\tv_mfma_scale_f32_16x16x128_f8f6f4 %0, %1, %2, %0, %3, %3 op_sel_hi:[0,0,0]"
        : "+v"(c)
        : "v"(a), "v"(b), "v"(scale));
    return c;
}
__device__ __forceinline__ F4 mma_qk(I8 a, I8 b, F4 c)
{
    if constexpr(STREAM_QK_VGPR)
        return mma_vgpr(a, b, c);
    return mma(a, b, c);
}
__device__ __forceinline__ F4 mma_pv(I8 a, I8 b, F4 c)
{
    if constexpr(STREAM_PV_VGPR)
        return mma_vgpr(a, b, c);
    return mma(a, b, c);
}
__device__ __forceinline__ uint32_t pack4(F4 x)
{
#pragma unroll
    for(int j = 0; j < 4; ++j)
        x[j] = fminf(448.f, fmaxf(-448.f, x[j]));
    int w = __builtin_amdgcn_cvt_pk_fp8_f32(x[0], x[1], 0, 0);
    return __builtin_amdgcn_cvt_pk_fp8_f32(x[2], x[3], w, 1);
}
// Each 1 KiB chunk is 16 rows x 64 bytes. XOR preserves 16-byte vectors.
template <int Rows>
__device__ __forceinline__ int address(int row, int col)
{
    return kv_address<Rows, KV_CHUNK_PAD>(row, col);
}
template <int Rows>
__device__ __forceinline__ I8 fragment(const uint8_t* base, int row, int col)
{
    I8 x;
    const U4 a = *reinterpret_cast<const U4*>(base + address<Rows>(row, col));
    const U4 b = *reinterpret_cast<const U4*>(base + address<Rows>(row, col + 16));
#pragma unroll
    for(int j = 0; j < 4; ++j)
    {
        x[j]     = a[j];
        x[j + 4] = b[j];
    }
    return x;
}
// Constant tile offsets factored from the common per-lane base.
// This changes only integer address expressions, not the QK arithmetic tree.
template <int K, int N>
__device__ __forceinline__ I8 bank_qfragment(const uint8_t* base, int row, int col)
{
    constexpr int off = (16 * K + N) * KV_CHUNK_STRIDE;
    const U4 a        = *reinterpret_cast<const U4*>(base + address<128>(row, col) + off);
    const U4 b        = *reinterpret_cast<const U4*>(base + address<128>(row, col + 16) + off);
    I8 x;
#pragma unroll
    for(int j = 0; j < 4; ++j)
    {
        x[j]     = a[j];
        x[j + 4] = b[j];
    }
    return x;
}
template <int K, int Fragments>
__device__ __forceinline__ void
bank_qk_step(const uint8_t* base, int row, int col, I8 qr, F4 (&score)[Fragments])
{
    score[0] = mma_qk(bank_qfragment<K, 0>(base, row, col), qr, score[0]);
    score[1] = mma_qk(bank_qfragment<K, 1>(base, row, col), qr, score[1]);
    if constexpr(Fragments == 4)
    {
        score[2] = mma_qk(bank_qfragment<K, 2>(base, row, col), qr, score[2]);
        score[3] = mma_qk(bank_qfragment<K, 3>(base, row, col), qr, score[3]);
    }
}
__device__ __forceinline__ int p_address(int head, int key)
{
    return head * 128 + (key ^ ((head & 7) * 16));
}
__device__ __forceinline__ I8 p_fragment(const uint8_t* base, int head, int col)
{
    I8 x;
    U4 a = *reinterpret_cast<const U4*>(base + p_address(head, col));
    U4 b = *reinterpret_cast<const U4*>(base + p_address(head, col + 16));
#pragma unroll
    for(int j = 0; j < 4; ++j)
    {
        x[j]     = a[j];
        x[j + 4] = b[j];
    }
    return x;
}
__device__ __forceinline__ I8 v_fragment(const uint8_t* base, int col, int key)
{
    I8 x{};
#pragma unroll
    for(int j = 0; j < 8; ++j)
    {
        uint32_t w = 0;
#pragma unroll
        for(int b = 0; b < 4; ++b)
            w |= uint32_t(base[address<128>(key + j * 4 + b, col)]) << (8 * b);
        x[j] = w;
    }
    return x;
}
using I2 = int __attribute__((ext_vector_type(2)));
__device__ __forceinline__ I8 v_fragment_tr(const uint8_t* base, int col_base)
{
    const int lane = threadIdx.x & 63;
    I8 x;
#pragma unroll
    for(int i = 0; i < 4; ++i)
    {
        int row       = (lane / 16) * 32 + i * 8 + (lane % 16) / 2;
        int col       = col_base + (lane % 2) * 8;
        uint32_t addr = uint32_t(reinterpret_cast<uintptr_t>(base + address<128>(row, col)));
        I2 r;
        asm volatile("ds_read_b64_tr_b8 %0, %1" : "=v"(r) : "v"(addr) : "memory");
        x[i * 2]     = r[0];
        x[i * 2 + 1] = r[1];
    }
    // Tie the loaded operand to the wait. A memory-only barrier can move
    // after the last register-only MFMA when its results stay in registers.
    asm volatile("s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)" : "+v"(x) : : "memory");
    return x;
}
__device__ __forceinline__ void direct_load(uint8_t* dest, const uint8_t* src)
{
    uint32_t lds  = __builtin_amdgcn_readfirstlane(uint32_t(reinterpret_cast<uintptr_t>(dest)));
    uint64_t addr = reinterpret_cast<uintptr_t>(src);
    asm volatile("s_mov_b32 m0, %0\n\ts_nop 0\n\tglobal_load_lds_dwordx4 %1, off"
                 :
                 : "s"(lds), "v"(addr)
                 : "m0", "memory");
}
#ifndef HEAD32_FUSE_SUM_P
#define HEAD32_FUSE_SUM_P 0
#endif
#ifndef HEAD32_SUM_LOCAL
#define HEAD32_SUM_LOCAL 0
#endif
#ifndef STREAM_ONLINE
#define STREAM_ONLINE 0
#endif
#ifndef STREAM_PV_IMMEDIATE
#define STREAM_PV_IMMEDIATE 0
#endif
#ifndef STREAM_PV_FENCE
#define STREAM_PV_FENCE 0
#endif
#ifndef STREAM_BRANCHLESS
#define STREAM_BRANCHLESS 0
#endif

// Within the 8-row transpose piece, the XOR swizzle is unchanged.
// Fold the piece's row/chunk offset into the instruction immediate.
template <int Tile, int Piece>
__device__ __forceinline__ I2 bank_pv_read(const uint8_t* base)
{
    const int lane    = threadIdx.x & 63;
    int row           = (lane / 16) * 32 + (lane % 16) / 2;
    int col           = (Tile % 4) * 16 + (lane % 2) * 8;
    uint32_t addr     = uint32_t(reinterpret_cast<uintptr_t>(base + address<128>(row, col)));
    constexpr int off = ((Tile / 4) * 8 + Piece / 2) * KV_CHUNK_STRIDE + (Piece % 2) * 512;
    static_assert(off <= 65535);
    I2 x;
    asm volatile("ds_read_b64_tr_b8 %0, %1 offset:%2" : "=v"(x) : "v"(addr), "n"(off) : "memory");
    return x;
}
template <int Tile, bool BankPV>
__device__ __forceinline__ I8 stream_vfragment(const uint8_t* base)
{
    if constexpr(!STREAM_PV_IMMEDIATE)
        return v_fragment_tr(base, Tile * 16);
    else
    {
        if constexpr(BankPV)
        {
            I8 x;
            I2 a = bank_pv_read<Tile, 0>(base), b = bank_pv_read<Tile, 1>(base);
            I2 c = bank_pv_read<Tile, 2>(base), d = bank_pv_read<Tile, 3>(base);
            x[0] = a[0];
            x[1] = a[1];
            x[2] = b[0];
            x[3] = b[1];
            x[4] = c[0];
            x[5] = c[1];
            x[6] = d[0];
            x[7] = d[1];
            asm volatile("s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)" : "+v"(x) : : "memory");
            return x;
        }
        const int lane = threadIdx.x & 63;
        I8 x;
#pragma unroll
        for(int i = 0; i < 4; ++i)
        {
            int row       = (lane / 16) * 32 + i * 8 + (lane % 16) / 2;
            int col       = (Tile % 4) * 16 + (lane % 2) * 8;
            uint32_t addr = uint32_t(reinterpret_cast<uintptr_t>(base + address<128>(row, col)));
            I2 r;
            asm volatile("ds_read_b64_tr_b8 %0, %1 offset:%2"
                         : "=v"(r)
                         : "v"(addr), "n"((Tile / 4) * 8 * KV_CHUNK_STRIDE)
                         : "memory");
            x[i * 2]     = r[0];
            x[i * 2 + 1] = r[1];
        }
        asm volatile("s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)" : "+v"(x) : : "memory");
        return x;
    }
}

template <int Tile, int WavesPerHead, bool BankPV>
__device__ __forceinline__ void stream_pv_tile(const uint8_t* kvs,
                                               I8 prob,
                                               F4 (&merged)[32 / WavesPerHead],
                                               int iteration,
                                               float alpha,
                                               float beta,
                                               float scale,
                                               int half)
{
    I8 v = stream_vfragment<Tile, BankPV>(kvs + half * (64 / WavesPerHead) * KV_CHUNK_STRIDE);
    if constexpr(STREAM_ONLINE)
    {
        merged[Tile] = mma_pv(v, prob, merged[Tile] * alpha);
        if constexpr(STREAM_PV_FENCE)
            asm volatile("" : "+v"(merged[Tile]) : : "memory");
        return;
    }
    F4 out = mma_pv(v, prob, F4{});
    out *= scale;
    if constexpr(STREAM_BRANCHLESS)
    {
#pragma unroll
        for(int j = 0; j < 4; ++j)
            merged[Tile][j] = fmaf(out[j], iteration == 0 ? 1.f : beta, merged[Tile][j] * alpha);
    }
    else
    {
        if(iteration == 0)
            merged[Tile] = out;
        else
        {
#pragma unroll
            for(int j = 0; j < 4; ++j)
                merged[Tile][j] = fmaf(out[j], beta, merged[Tile][j] * alpha);
        }
    }
    if constexpr(STREAM_PV_FENCE)
        asm volatile("" : "+v"(merged[Tile]) : : "memory");
}

// Compile-time body choices. The head/K geometries change data reuse, not the
template <int BlockK>
__device__ __forceinline__ int p_address(int head, int key)
{
    return head * BlockK + (key ^ ((head & (BlockK / 16 - 1)) * 16));
}
template <int BlockK>
__device__ __forceinline__ I8 p_fragment(const uint8_t* ps, int head, int col)
{
    I8 x{};
    if(col < BlockK)
    {
        U4 a = *reinterpret_cast<const U4*>(ps + p_address<BlockK>(head, col));
        U4 b = *reinterpret_cast<const U4*>(ps + p_address<BlockK>(head, col + 16));
#pragma unroll
        for(int j = 0; j < 4; ++j)
        {
            x[j]     = a[j];
            x[j + 4] = b[j];
        }
    }
    return x;
}
template <int K, int N, int BlockK>
__device__ __forceinline__ I8 k_fragment(const uint8_t* base, int row, int col)
{
    constexpr int offset = (K * (BlockK / 8) + N) * KV_CHUNK_STRIDE;
    const U4 a           = *reinterpret_cast<const U4*>(base + address<BlockK>(row, col) + offset);
    const U4 b = *reinterpret_cast<const U4*>(base + address<BlockK>(row, col + 16) + offset);
    I8 x;
#pragma unroll
    for(int j = 0; j < 4; ++j)
    {
        x[j]     = a[j];
        x[j + 4] = b[j];
    }
    return x;
}
template <int K, int N, int Fragments, int BlockK>
__device__ __forceinline__ void
qk_step(const uint8_t* kvs, int row, int col, I8 q, F4 (&score)[Fragments])
{
    score[N] = mma_qk(k_fragment<K, N, BlockK>(kvs, row, col), q, score[N]);
    if constexpr(N + 1 < Fragments)
        qk_step<K, N + 1, Fragments, BlockK>(kvs, row, col, q, score);
}
template <bool BF16, bool Unit>
__device__ __forceinline__ I8 q_load(Params p, int64_t off)
{
    I8 x;
    U4 a, b;
    if constexpr(BF16)
    {
        const auto* src = reinterpret_cast<const uint16_t*>(p.q) + off;
        a               = q_encode::pack16<Unit>(src, p.q_scale);
        b               = q_encode::pack16<Unit>(src + 16, p.q_scale);
    }
    else
    {
        a = *reinterpret_cast<const U4*>(p.q + off);
        b = *reinterpret_cast<const U4*>(p.q + off + 16);
    }
#pragma unroll
    for(int j = 0; j < 4; ++j)
    {
        x[j]     = a[j];
        x[j + 4] = b[j];
    }
    return x;
}
template <int Mask, int K, int HeadRows>
__device__ __forceinline__ int q_address(int row, int col)
{
    constexpr int rank = __builtin_popcount(Mask & ((1 << K) - 1));
    return rank * HeadRows * 128 + kv_address<HeadRows, 0>(row, col);
}
template <int K, int Mask, int HeadRows, bool BF16, bool Unit>
__device__ __forceinline__ I8 q_fragment(Params p,
                                         const I8 (&retained)[4],
                                         uint8_t* qs,
                                         int64_t off,
                                         int iteration,
                                         int row,
                                         int col,
                                         int half)
{
    if constexpr(!(Mask & (1 << K)))
        return retained[K];
    else
    {
        const int low  = q_address<Mask, K, HeadRows>(row, col),
                  high = q_address<Mask, K, HeadRows>(row, col + 16);
        if(iteration == 0)
        {
            I8 x = q_load<BF16, Unit>(p, off + K * 128);
            if(__builtin_amdgcn_readfirstlane(half) == 0)
            {
                U4 a, b;
#pragma unroll
                for(int j = 0; j < 4; ++j)
                {
                    a[j] = x[j];
                    b[j] = x[j + 4];
                }
                *reinterpret_cast<U4*>(qs + low)  = a;
                *reinterpret_cast<U4*>(qs + high) = b;
            }
            return x;
        }
        U4 a, b;
        uint32_t addr0 = uint32_t(reinterpret_cast<uintptr_t>(qs + low)),
                 addr1 = uint32_t(reinterpret_cast<uintptr_t>(qs + high));
        asm volatile("ds_read_b128 %0, %2\n\tds_read_b128 %1, %3"
                     : "=&v"(a), "=&v"(b)
                     : "v"(addr0), "v"(addr1)
                     : "memory");
        // Tie the loaded operands to the wait; the old qreload failure showed
        // that a memory-only wait is insufficient for register-only MFMA use.
        asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(a), "+v"(b) : : "memory");
        I8 x;
#pragma unroll
        for(int j = 0; j < 4; ++j)
        {
            x[j]     = a[j];
            x[j + 4] = b[j];
        }
        return x;
    }
}
// No implicit wait: callers can batch or pipeline several four-read operands.
template <int Tile, int BlockK>
__device__ __forceinline__ I8 v_read(const uint8_t* base)
{
    const int lane = threadIdx.x & 63;
    I8 x{};
    if constexpr(BlockK == 128)
    {
#pragma unroll
        for(int i = 0; i < 4; ++i)
        {
            int row       = (lane / 16) * 32 + i * 8 + (lane % 16) / 2;
            int col       = (Tile % 4) * 16 + (lane % 2) * 8;
            uint32_t addr = uint32_t(reinterpret_cast<uintptr_t>(base + address<BlockK>(row, col)));
            I2 pair;
            asm volatile("ds_read_b64_tr_b8 %0, %1 offset:%2"
                         : "=v"(pair)
                         : "v"(addr), "n"((Tile / 4) * (BlockK / 16) * KV_CHUNK_STRIDE)
                         : "memory");
            x[i * 2]     = pair[0];
            x[i * 2 + 1] = pair[1];
        }
    }
    else if(lane < 32)
    {
#pragma unroll
        for(int i = 0; i < 4; ++i)
        {
            int row       = (lane / 16) * 32 + i * 8 + (lane % 16) / 2;
            int col       = (Tile % 4) * 16 + (lane % 2) * 8;
            uint32_t addr = uint32_t(reinterpret_cast<uintptr_t>(base + address<BlockK>(row, col)));
            I2 pair;
            asm volatile("ds_read_b64_tr_b8 %0, %1 offset:%2"
                         : "=v"(pair)
                         : "v"(addr), "n"((Tile / 4) * (BlockK / 16) * KV_CHUNK_STRIDE)
                         : "memory");
            x[i * 2]     = pair[0];
            x[i * 2 + 1] = pair[1];
        }
    }
    return x;
}
template <int Tile, int Tiles, int BlockK, int Mode>
__device__ __forceinline__ void pv(const uint8_t* kvs, I8 prob, F4 (&merged)[Tiles], float alpha)
{
    I8 a = v_read<Tile, BlockK>(kvs);
    if constexpr(Mode == 1)
    {
        static_assert(Tile + 1 < Tiles);
        I8 b = v_read<Tile + 1, BlockK>(kvs);
        asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(a), "+v"(b) : : "memory");
        merged[Tile]     = mma_pv(a, prob, merged[Tile] * alpha);
        merged[Tile + 1] = mma_pv(b, prob, merged[Tile + 1] * alpha);
        if constexpr(Tile + 2 < Tiles)
            pv<Tile + 2, Tiles, BlockK, Mode>(kvs, prob, merged, alpha);
    }
    else
    {
        asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(a) : : "memory");
        merged[Tile] = mma_pv(a, prob, merged[Tile] * alpha);
        if constexpr(Tile + 1 < Tiles)
            pv<Tile + 1, Tiles, BlockK, Mode>(kvs, prob, merged, alpha);
    }
}
template <int Tile, int Tiles>
__device__ __forceinline__ void
pv_pipeline(const uint8_t* kvs, I8 prob, F4 (&merged)[Tiles], float alpha, I8 current)
{
    if constexpr(Tile + 1 < Tiles)
    {
        I8 next = v_read<Tile + 1, 128>(kvs);
        // Only the next operand's four ordered LDS reads may remain pending.
        asm volatile("s_waitcnt lgkmcnt(4)" : "+v"(current) : : "memory");
        merged[Tile] = mma_pv(current, prob, merged[Tile] * alpha);
        pv_pipeline<Tile + 1, Tiles>(kvs, prob, merged, alpha, next);
    }
    else
    {
        asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(current) : : "memory");
        merged[Tile] = mma_pv(current, prob, merged[Tile] * alpha);
    }
}

// K64 PV consumes two real K32 operands; no zero-padded K128 work.
// Clang official signature is (int64_t,int64_t,F4,const int,const int,const int).
// Keep an actual builtin expression for compiler signature/type verification.
// build01 emitted this path; H32 selected costly AGPR transfer pairs, preserved.
__device__ __forceinline__ F4 mma32_signature(I2 a, I2 b, F4 accum)
{
    return __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(
        __builtin_bit_cast(int64_t, a), __builtin_bit_cast(int64_t, b), accum, 0, 0, 0);
}
__device__ __forceinline__ F4 mma32(I2 a, I2 b, F4 accum)
{
    asm volatile("v_mfma_f32_16x16x32_fp8_fp8 %0, %1, %2, %0" : "+v"(accum) : "v"(a), "v"(b));
    return accum;
}
template <int Sub>
__device__ __forceinline__ I2 p32(const uint8_t* ps, int head, int group)
{
    return *reinterpret_cast<const I2*>(ps + p_address<64>(head, Sub * 32 + group * 8));
}
template <int Tile, int Sub>
__device__ __forceinline__ I2 v32(const uint8_t* base)
{
    const int lane = threadIdx.x & 63;
    const int row  = Sub * 32 + (lane / 16) * 8 + (lane % 16) / 2;
    const int col  = (Tile % 4) * 16 + (lane % 2) * 8;
    uint32_t addr  = uint32_t(reinterpret_cast<uintptr_t>(base + address<64>(row, col)));
    I2 x;
    asm volatile("ds_read_b64_tr_b8 %0, %1 offset:%2"
                 : "=v"(x)
                 : "v"(addr), "n"((Tile / 4) * 4 * KV_CHUNK_STRIDE)
                 : "memory");
    return x;
}
template <int Tile, int Tiles, bool Pair>
__device__ __forceinline__ void
pv32(const uint8_t* kvs, I2 p0, I2 p1, F4 (&merged)[Tiles], float alpha)
{
    I2 a = v32<Tile, 0>(kvs), b = v32<Tile, 1>(kvs);
    if constexpr(Pair)
    {
        static_assert(Tile + 1 < Tiles);
        I2 c = v32<Tile + 1, 0>(kvs), d = v32<Tile + 1, 1>(kvs);
        asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(a), "+v"(b), "+v"(c), "+v"(d) : : "memory");
        merged[Tile]     = mma32(b, p1, mma32(a, p0, merged[Tile] * alpha));
        merged[Tile + 1] = mma32(d, p1, mma32(c, p0, merged[Tile + 1] * alpha));
        if constexpr(Tile + 2 < Tiles)
            pv32<Tile + 2, Tiles, Pair>(kvs, p0, p1, merged, alpha);
    }
    else
    {
        asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(a), "+v"(b) : : "memory");
        merged[Tile] = mma32(b, p1, mma32(a, p0, merged[Tile] * alpha));
        if constexpr(Tile + 1 < Tiles)
            pv32<Tile + 1, Tiles, Pair>(kvs, p0, p1, merged, alpha);
    }
}

// Stage one future K128 tile into its own KV and selected-ID buffers.
// Separate current/next masks survive until their matching tile is consumed.
// The caller waits for VMEM and all current readers before reusing a buffer.
template <int Waves, int BlockK>
__device__ __forceinline__ void
stage_async(Params p, uint8_t* kv, int* selected, int start, int end, int tile)
{
    static_assert((Waves == 8 || Waves == 4) && BlockK == 128);
    const int tid = threadIdx.x, lane = tid & 63, wave = tid / 64;
    if(tid < BlockK)
    {
        const int at  = start + tile * BlockK + tid;
        const int id  = at < end ? p.indices[at] : -1;
        selected[tid] = (id >= 0 && id < p.slots) ? id : -1;
    }
    __syncthreads();
    const int id0 = selected[(wave % (BlockK / 16)) * 16 + lane / 4];
    const int id1 = Waves == 4 ? selected[(wave + 4) * 16 + lane / 4] : -1;
#pragma unroll
    for(int i = 0; i < (BlockK / 2) / Waves; ++i)
    {
        const int chunk = wave + i * Waves, r = (chunk % (BlockK / 16)) * 16 + lane / 4;
        const int col = (chunk / (BlockK / 16)) * 64 + ((lane % 4) * 16 ^ ((r & 3) * 16));
        const int id  = (Waves == 4 && (i & 1)) ? id1 : id0;
        if(id >= 0)
            direct_load(kv + chunk * KV_CHUNK_STRIDE, p.kv + int64_t(id) * 512 + col);
        else
            *reinterpret_cast<U4*>(kv + chunk * KV_CHUNK_STRIDE + lane * 16) = U4{};
    }
}

template <bool Gradual>
__device__ __forceinline__ float domain_exp2(float x)
{
    if constexpr(Gradual)
    {
        const bool tiny     = x < -126.f;
        const float shifted = x + (tiny ? 64.f : 0.f);
        return __builtin_amdgcn_exp2f(shifted) * (tiny ? 0x1p-64f : 1.f);
    }
    else
        return __builtin_amdgcn_exp2f(x);
}
template <int Waves, int BlockK>
__device__ __forceinline__ void
stage_iddirect(Params p, uint8_t* kv, int* selected, int start, int end, int tile)
{
    static_assert(Waves == 8 && BlockK == 128);
    const int lane = threadIdx.x & 63, wave = threadIdx.x / 64;
    const int row = wave * 16 + lane / 4;
    const int at  = start + tile * BlockK + row;
    const int raw = at < end ? p.indices[at] : -1;
    const int id  = (raw >= 0 && raw < p.slots) ? raw : -1;
    if((lane & 3) == 0)
        selected[row] = id;
#pragma unroll
    for(int i = 0; i < (BlockK / 2) / Waves; ++i)
    {
        const int chunk = wave + i * Waves;
        const int col   = i * 64 + ((lane % 4) * 16 ^ ((row & 3) * 16));
        if(id >= 0)
            direct_load(kv + chunk * KV_CHUNK_STRIDE, p.kv + int64_t(id) * 512 + col);
        else
            *reinterpret_cast<U4*>(kv + chunk * KV_CHUNK_STRIDE + lane * 16) = U4{};
    }
}
} // namespace sparse_mla_fp8::online
