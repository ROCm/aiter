// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx1250 large-message all-reduce + residual add + RMSNorm (bf16, TP2/TP4/TP8),
// for prefill-sized inputs (tens of MiB) where the Lamport kernels run out of
// inbox and RCCL is 2x off the link.
//
//   phase 1  reduce-scatter, pushed: the rows are ngpus contiguous chunks of
//            R = ceil(rows / ngpus); every rank stores the rows of chunk c into
//            slot[rank] of rank c's RS inbox (remote stores, the faster
//            direction on MI455X: ~1.4 TB/s egress per GPU against ~1.15 TB/s
//            ingress for peer loads, all four GPUs at once).
//   phase 2  the owner sums its chunk in fp32 in rank order (the bits of the
//            Lamport kernels), rounds to bf16 once, pushes the sum into
//            slot[rank] of every peer's AG inbox, and adds the residual and
//            normalizes its own rows.
//   phase 3  every rank adds the residual and normalizes the peers' rows from
//            its AG inbox.
//
// Sync: block b owns rows b, b + G, ... of every chunk on every rank, so it
// only waits for block b of its peers: a flag per (phase, sender, block),
// 128 B apart, in cached IPC memory, written with a system-scope store once
// the block's remote data stores have completed (s_wait_storecnt; a
// system-scope release fence is a full L2 writeback on gfx1250 and costs
// milliseconds), polled with system-scope loads. Data is stored and loaded
// with system-scope 128-bit accesses, so no cache maintenance is needed.
// Epochs are a per-block counter in the rank's own flag buffer; every call of
// a given size launches the same grid on every rank, so the counters of block
// b agree across ranks even when the grid changes between calls.
//
// Reuse of the inboxes across calls: a rank enters call k+1 only after its
// call k finished, and its phase 3 waited for every owner's phase-2 flag, sent
// after that owner had read its RS inbox; an AG inbox is written in call k+1
// only after the writer saw the reader's call k+1 phase-1 flag. So no barrier
// is needed at either end, eager or in a CUDA graph.
//
// out / residual_out follow ar_rmsnorm_gfx1250 and the Lamport fused kernel
// step by step (round the sum to bf16, add the residual in fp32, 32-lane
// butterfly then a serial sum over the waves of the row, x * (rsqrt * w)).
#include "ar_2shot_rms_gfx1250.h"
#include "aiter_stream.h"
#include "mx_quant_utils.h"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace aiter {
namespace ar2s {

constexpr int kMaxBlocks = 1024;
constexpr int kMaxRanks  = 8;
constexpr int kFlagPad   = 32; // uint32 a flag: 128 B apart

struct Flags
{
    uint32_t rs[kMaxRanks][kMaxBlocks * kFlagPad];
    uint32_t ag[kMaxRanks][kMaxBlocks * kFlagPad];
    uint32_t counter[kMaxBlocks];
};

struct __align__(16) Peers
{
    char* data[kMaxRanks];
    Flags* flags[kMaxRanks];
};

using u32x4      = __attribute__((__vector_size__(4 * sizeof(uint32_t)))) uint32_t;
using u32x4_gptr = __attribute__((address_space(1))) u32x4*;

__device__ __forceinline__ void st_sys(void* p, u32x4 v)
{
    __builtin_amdgcn_global_store_b128((u32x4_gptr)p, v, "");
}
__device__ __forceinline__ u32x4 ld_sys(const void* p)
{
    return __builtin_amdgcn_global_load_b128((u32x4_gptr)p, "");
}
__device__ __forceinline__ uint32_t f2bf(float f)
{
    uint32_t u = __float_as_uint(f);
    if((u & 0x7fffffffu) > 0x7f800000u)
        return (u >> 16) | 0x40u; // quiet NaN
    u += 0x7fffu + ((u >> 16) & 1u);
    return u >> 16;
}
__device__ __forceinline__ void unpack(u32x4 v, float* x)
{
#pragma unroll
    for(int j = 0; j < 4; ++j)
    {
        x[2 * j]     = __uint_as_float(v[j] << 16);
        x[2 * j + 1] = __uint_as_float(v[j] & 0xffff0000u);
    }
}
__device__ __forceinline__ u32x4 pack(const float* x)
{
    u32x4 v;
#pragma unroll
    for(int j = 0; j < 4; ++j)
        v[j] = f2bf(x[2 * j]) | (f2bf(x[2 * j + 1]) << 16);
    return v;
}
__device__ __forceinline__ bool epoch_before(uint32_t observed, uint32_t target)
{
    return static_cast<int32_t>(observed - target) < 0;
}
__device__ __forceinline__ void wait_flag(const uint32_t* p, uint32_t target)
{
    while(epoch_before(__scoped_atomic_load_n(p, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM), target))
        ;
}
__device__ __forceinline__ void signal_flag(uint32_t* p, uint32_t v)
{
    __scoped_atomic_store_n(p, v, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
}

// sum over the block (blockDim = row packs, whole waves of 32), serial over waves
__device__ __forceinline__ float block_sum(float v, float (*partial)[32], int& buf)
{
    for(int off = 16; off > 0; off >>= 1)
        v += __shfl_xor(v, off);
    if(threadIdx.x % 32 == 0)
        partial[buf][threadIdx.x / 32] = v;
    __syncthreads();
    float r = partial[buf][0];
    for(int w = 1; w < (int)(blockDim.x / 32); ++w)
        r += partial[buf][w];
    buf ^= 1; // double-buffered: the next call's barrier orders this read
    return r;
}

// MXFP8 1x32 quant of the 8 bf16 values a thread holds (one quarter of a group of 32; the
// group is lanes 4g..4g+3 of a wave), bit for bit per_1x32_mx_quant_hip(fp8, e8m0) on gfx1250:
// amax over the bf16 values, the inf/NaN cap, the RoundUp E8M0 scale, the clamp of a
// degenerate group and v_cvt_scalef32_pk8_fp8_bf16 with the scale as divisor.
typedef __bf16 bf16x8_t __attribute__((ext_vector_type(8)));
typedef int i32x2_t __attribute__((ext_vector_type(2)));
__device__ __forceinline__ void mxfp8_quant8(u32x4 bf, uint32_t& lo, uint32_t& hi, uint8_t& sbyte)
{
    float v[8];
    unpack(bf, v);
    float amax = 0.f;
#pragma unroll
    for(int j = 0; j < 8; ++j)
        amax = fmaxf(amax, fabsf(v[j]));
    amax = fmaxf(amax, __shfl_xor(amax, 1));
    amax = fmaxf(amax, __shfl_xor(amax, 2));
    const bool degenerate = !(amax < __builtin_inff());
    amax                  = fminf(amax, 448.0f * 0x1.0p119f);
    float scale = aiter::fp_f32_to_e8m0_scale<aiter::kDefaultMxScaleRoundMode, aiter::MxDtype::FP8_E4M3>(amax);
    constexpr bool kScaleMayClip = aiter::kDefaultMxScaleRoundMode == aiter::MxScaleRoundMode::RoundDown ||
                                   aiter::kDefaultMxScaleRoundMode == aiter::MxScaleRoundMode::Even;
    if(kScaleMayClip || degenerate)
    {
        const float h      = 448.0f * scale;
        const uint32_t hb  = f2bf(h);
        const uint32_t nhb = f2bf(-h);
        uint32_t w[4] = {bf[0], bf[1], bf[2], bf[3]};
#pragma unroll
        for(int j = 0; j < 8; ++j)
        {
            uint32_t e = (w[j / 2] >> (16 * (j % 2))) & 0xffffu;
            if(v[j] > h)
                e = hb;
            if(v[j] < -h)
                e = nhb;
            w[j / 2] = (w[j / 2] & ~(0xffffu << (16 * (j % 2)))) | (e << (16 * (j % 2)));
        }
        bf = u32x4{w[0], w[1], w[2], w[3]};
    }
    const i32x2_t r = __builtin_amdgcn_cvt_scalef32_pk8_fp8_bf16(__builtin_bit_cast(bf16x8_t, bf), scale);
    lo              = (uint32_t)r[0];
    hi              = (uint32_t)r[1];
    sbyte           = (uint8_t)((__float_as_uint(scale) >> 23) & 0xffu);
}

// residual add + RMSNorm of the 8 fp32 values (sum + residual) a thread holds; returns the
// bf16 residual_out pack and the bf16 normed pack (the bits norm_store writes).
__device__ __forceinline__ void norm_packs(
    const float* x, const float* w, float eps, int hidden, float (*partial)[32], int& buf, u32x4& res_pack,
    u32x4& out_pack)
{
    float sq = 0.f;
#pragma unroll
    for(int j = 0; j < 8; ++j)
        sq += x[j] * x[j];
    const float inv_rms = rsqrtf(block_sum(sq, partial, buf) / hidden + eps);
    res_pack            = pack(x);
    float o[8];
#pragma unroll
    for(int j = 0; j < 8; ++j)
        o[j] = x[j] * (inv_rms * w[j]);
    out_pack = pack(o);
}

// The MXFP8 row a block just normalized, staged in LDS (hidden E4M3 bytes, then hidden / 32
// E8M0 bytes), as hidden / 16 + hidden / 512 16-byte packs. Callers sync before and after.
__device__ __forceinline__ void stage_mxfp8(u32x4 out_pack, uint8_t* stage, int hidden, int t)
{
    uint32_t lo, hi;
    uint8_t sb;
    mxfp8_quant8(out_pack, lo, hi, sb);
    *reinterpret_cast<uint2*>(stage + 8 * t) = make_uint2(lo, hi);
    if((t & 3) == 0)
        stage[hidden + t / 4] = sb;
}

// ar2s, templated on the output and on where the residual lives:
//   FP8   out is MXFP8: E4M3 [rows, hidden] in out_q, row-major E8M0 [rows, hidden / 32] in
//         out_s, the bytes per_1x32_mx_quant_hip(fp8, e8m0) gives for the bf16 out (which is
//         not written).
//   SP    sequence-parallel residual: each rank keeps the residual of its own chunk only
//         (rows [rank * R, rank * R + R)); res is read and res_out written there alone. The
//         owner normalizes (and quantizes) its rows and pushes the result instead of the
//         bf16 sum, so phase 3 is a copy. out is bit-identical to the replicated path's.
//   AUX   (with SP, bf16) no norm: the whole residual_out (sum + the owners' residual rows) on
//         every rank, the bits the replicated path writes to residual_out; out is not written.
//         It gathers a sequence-parallel residual stream (e.g. for a drafter's taps).
template <int NG, bool FP8, bool SP, bool AUX = false>
__global__ void __launch_bounds__(1024, 1) ar2s_rms_kernel(Peers peers,
                                                           int rank,
                                                           const u32x4* __restrict__ in,
                                                           const u32x4* __restrict__ res,
                                                           const u32x4* __restrict__ wgt,
                                                           u32x4* __restrict__ out,
                                                           u32x4* __restrict__ res_out,
                                                           u32x4* __restrict__ out_q,
                                                           u32x4* __restrict__ out_s,
                                                           int N,
                                                           int R,
                                                           int64_t slot_packs,
                                                           int64_t ag_off_packs,
                                                           float eps,
                                                           bool gemma)
{
    __shared__ float partial[2][32];
    __shared__ uint32_t s_epoch;
    __shared__ __align__(16) uint8_t stage[FP8 ? 8192 + 256 : 16];
    int buf          = 0;
    const int t      = threadIdx.x;
    const int rp     = blockDim.x; // 16B packs a row
    const int b      = blockIdx.x;
    const int G      = gridDim.x;
    const int hidden = rp * 8;
    const int qp     = hidden / 16;  // E4M3 packs a row (FP8)
    const int sp     = hidden / 512; // E8M0 packs a row (FP8)
    Flags* my_flags  = peers.flags[rank];
    u32x4* my_data   = reinterpret_cast<u32x4*>(peers.data[rank]);
    if(t == 0)
        s_epoch = my_flags->counter[b] + 1u;
    __syncthreads();
    const uint32_t epoch = s_epoch;

    // k = 0..NG-1 is rank (rank + k) % NG; read once, before any remote store
    u32x4* dst[NG];
    int chunk[NG];
    int rows_k[NG];
#pragma unroll
    for(int k = 0; k < NG; ++k)
    {
        chunk[k]  = (rank + k) % NG;
        dst[k]    = reinterpret_cast<u32x4*>(peers.data[chunk[k]]);
        int r     = N - chunk[k] * R;
        rows_k[k] = r < 0 ? 0 : (r > R ? R : r);
    }
    float w[8];
    unpack(wgt[t], w);
#pragma unroll
    for(int e = 0; e < 8; ++e)
        w[e] += gemma ? 1.f : 0.f;

    // ---- phase 1: reduce-scatter push
    for(int j = b; j < R; j += G)
    {
        u32x4 v[NG];
#pragma unroll
        for(int k = 1; k < NG; ++k)
            if(j < rows_k[k])
                v[k] = in[((int64_t)chunk[k] * R + j) * rp + t];
#pragma unroll
        for(int k = 1; k < NG; ++k)
            if(j < rows_k[k])
                st_sys(dst[k] + (int64_t)rank * slot_packs + (int64_t)j * rp + t, v[k]);
    }
    asm volatile("s_wait_storecnt 0x0" ::: "memory");
    __syncthreads();
    if(t < NG && t != rank)
        signal_flag(&peers.flags[t]->rs[rank][b * kFlagPad], epoch);

    // ---- phase 2: the own chunk: sum, add + norm (+ quant), push the sum (or, SP, the result)
    const int own_rows = rows_k[0];
    if(own_rows > b)
    {
        if(t < NG && t != rank)
            wait_flag(&my_flags->rs[t][b * kFlagPad], epoch);
        __syncthreads();
    }
    for(int j = b; j < own_rows; j += G)
    {
        const int64_t row = (int64_t)rank * R + j;
        const u32x4 mine  = in[row * rp + t];
        const u32x4 rv    = res[row * rp + t];
        u32x4 got[NG];
#pragma unroll
        for(int g = 0; g < NG; ++g)
            if(g != rank)
                got[g] = ld_sys(my_data + (int64_t)g * slot_packs + (int64_t)j * rp + t);
        float x[8] = {};
#pragma unroll
        for(int g = 0; g < NG; ++g)
        {
            float v[8];
            unpack(g == rank ? mine : got[g], v);
#pragma unroll
            for(int e = 0; e < 8; ++e)
                x[e] += v[e];
        }
        const u32x4 s = pack(x);
        const int64_t ag_row = ag_off_packs + (int64_t)rank * slot_packs + (int64_t)j * rp;
        if constexpr(!SP)
        {
#pragma unroll
            for(int k = 1; k < NG; ++k)
                st_sys(dst[k] + ag_row + t, s);
        }
        float r8[8];
        unpack(s, x);
        unpack(rv, r8);
#pragma unroll
        for(int e = 0; e < 8; ++e)
            x[e] += r8[e];
        if constexpr(AUX)
        {
            const u32x4 rpk       = pack(x);
            res_out[row * rp + t] = rpk;
#pragma unroll
            for(int k = 1; k < NG; ++k)
                st_sys(dst[k] + ag_row + t, rpk);
            continue;
        }
        u32x4 rpk, opk;
        norm_packs(x, w, eps, hidden, partial, buf, rpk, opk);
        res_out[row * rp + t] = rpk;
        if constexpr(FP8)
        {
            stage_mxfp8(opk, stage, hidden, t);
            __syncthreads();
            if(t < qp + sp)
            {
                const u32x4 q = reinterpret_cast<const u32x4*>(stage)[t];
                if(t < qp)
                    out_q[row * qp + t] = q;
                else
                    out_s[row * sp + (t - qp)] = q;
                if constexpr(SP)
                {
#pragma unroll
                    for(int k = 1; k < NG; ++k)
                        st_sys(dst[k] + ag_row + t, q);
                }
            }
            __syncthreads();
        }
        else
        {
            out[row * rp + t] = opk;
            if constexpr(SP)
            {
#pragma unroll
                for(int k = 1; k < NG; ++k)
                    st_sys(dst[k] + ag_row + t, opk);
            }
        }
    }
    asm volatile("s_wait_storecnt 0x0" ::: "memory");
    __syncthreads();
    if(t < NG && t != rank)
        signal_flag(&peers.flags[t]->ag[rank][b * kFlagPad], epoch);

    // ---- phase 3: the peers' rows from the AG inbox: add + norm (+ quant), or (SP) copy
#pragma unroll
    for(int k = 1; k < NG; ++k)
    {
        const int c  = chunk[k];
        const int rc = rows_k[k];
        if(rc <= b)
            continue;
        if(t == 0)
            wait_flag(&my_flags->ag[c][b * kFlagPad], epoch);
        __syncthreads();
        for(int j = b; j < rc; j += G)
        {
            const int64_t row    = (int64_t)c * R + j;
            const int64_t ag_row = ag_off_packs + (int64_t)c * slot_packs + (int64_t)j * rp;
            if constexpr(AUX)
            {
                res_out[row * rp + t] = ld_sys(my_data + ag_row + t);
            }
            else if constexpr(SP)
            {
                if constexpr(FP8)
                {
                    if(t < qp + sp)
                    {
                        const u32x4 q = ld_sys(my_data + ag_row + t);
                        if(t < qp)
                            out_q[row * qp + t] = q;
                        else
                            out_s[row * sp + (t - qp)] = q;
                    }
                }
                else
                {
                    out[row * rp + t] = ld_sys(my_data + ag_row + t);
                }
            }
            else
            {
                const u32x4 sv = ld_sys(my_data + ag_row + t);
                const u32x4 rv = res[row * rp + t];
                float x[8], r8[8];
                unpack(sv, x);
                unpack(rv, r8);
#pragma unroll
                for(int e = 0; e < 8; ++e)
                    x[e] += r8[e];
                u32x4 rpk, opk;
                norm_packs(x, w, eps, hidden, partial, buf, rpk, opk);
                res_out[row * rp + t] = rpk;
                if constexpr(FP8)
                {
                    stage_mxfp8(opk, stage, hidden, t);
                    __syncthreads();
                    if(t < qp + sp)
                    {
                        const u32x4 q = reinterpret_cast<const u32x4*>(stage)[t];
                        if(t < qp)
                            out_q[row * qp + t] = q;
                        else
                            out_s[row * sp + (t - qp)] = q;
                    }
                    __syncthreads();
                }
                else
                {
                    out[row * rp + t] = opk;
                }
            }
        }
    }
    if(t == 0)
        my_flags->counter[b] = epoch;
}

struct Ctx
{
    int rank, world, grid;
    int64_t max_bytes;
    Peers peers;
};

} // namespace ar2s

int64_t ar2s_rms_buffer_size(int64_t world_size, int64_t max_bytes)
{
    // RS inbox and AG inbox, world_size slots of a chunk each (slot = max_bytes / world)
    (void)world_size;
    return 2 * ((max_bytes + 4095) / 4096 * 4096);
}

int64_t ar2s_rms_flags_size() { return (int64_t)sizeof(ar2s::Flags); }

fptr_t init_ar2s_rms(int64_t rank,
                     const std::vector<int64_t>& data_ptrs,
                     const std::vector<int64_t>& flag_ptrs,
                     int64_t max_bytes,
                     int64_t grid)
{
    const int world = (int)data_ptrs.size();
    if(world < 2 || world > ar2s::kMaxRanks || world % 2 || (int)flag_ptrs.size() != world)
        throw std::invalid_argument("ar2s_rms: world size must be 2, 4, 6 or 8");
    auto* c      = new ar2s::Ctx();
    c->rank      = (int)rank;
    c->world     = world;
    c->max_bytes = (max_bytes + 4095) / 4096 * 4096;
    c->grid      = (int)std::min<int64_t>(std::max<int64_t>(grid, 1), ar2s::kMaxBlocks);
    for(int r = 0; r < world; ++r)
    {
        c->peers.data[r]  = reinterpret_cast<char*>(data_ptrs[r]);
        c->peers.flags[r] = reinterpret_cast<ar2s::Flags*>(flag_ptrs[r]);
    }
    // epochs and flags start at 0; the caller barriers before the first call
    if(hipMemset(c->peers.flags[rank], 0, sizeof(ar2s::Flags)) != hipSuccess ||
       hipDeviceSynchronize() != hipSuccess)
        throw std::runtime_error("ar2s_rms: cannot clear the flag buffer");
    return reinterpret_cast<fptr_t>(c);
}

int64_t ar2s_rms_max_rows(fptr_t ctx, int64_t hidden)
{
    auto* c = reinterpret_cast<ar2s::Ctx*>(ctx);
    return c->max_bytes / c->world / (hidden * 2) * c->world;
}

void ar2s_rms_ex(fptr_t ctx,
                 int64_t inp,
                 int64_t res_inp,
                 int64_t w,
                 int64_t out,
                 int64_t res_out,
                 int64_t out_q,
                 int64_t out_s,
                 int64_t rows,
                 int64_t hidden,
                 double eps,
                 bool gemma_norm,
                 bool fp8_out,
                 bool sp_residual,
                 bool aux_residual,
                 int64_t grid)
{
    auto* c      = reinterpret_cast<ar2s::Ctx*>(ctx);
    const int rp = (int)(hidden / 8);
    if(hidden % 256 != 0 || rp > 1024)
        throw std::invalid_argument("ar2s_rms: hidden must be a multiple of 256 and at most 8192");
    if(fp8_out && hidden % 512 != 0)
        throw std::invalid_argument("ar2s_rms: the MXFP8 output needs hidden a multiple of 512");
    if(aux_residual && (fp8_out || !sp_residual))
        throw std::invalid_argument("ar2s_rms: aux_residual needs sp_residual and a bf16 output");
    if(rows <= 0)
        return;
    const int64_t R          = (rows + c->world - 1) / c->world;
    const int64_t slot_bytes = c->max_bytes / c->world / 16 * 16;
    if(R * hidden * 2 > slot_bytes)
        throw std::invalid_argument("ar2s_rms: " + std::to_string(rows) +
                                    " rows exceed the inbox; split the call (ar2s_rms_max_rows)");
    if(((inp | res_inp | w | res_out | (aux_residual ? 0 : fp8_out ? (out_q | out_s) : out)) & 15) != 0)
        throw std::invalid_argument("ar2s_rms: operands must be 16-byte aligned");
    // the whole grid even if the last round of rows leaves blocks idle: fewer,
    // evenly loaded blocks measured slower (each block is latency bound)
    // grid > 0 caps the blocks (CUs) of this call, e.g. to leave CUs to a concurrent kernel; every rank
    // must pass the same value for a call (the per-block epochs then agree as for any grid change)
    const int G = (int)std::min<int64_t>(grid > 0 ? std::min<int64_t>(grid, ar2s::kMaxBlocks) : c->grid, R);
    const int64_t slot_packs  = slot_bytes / 16;
    const int64_t ag_off      = (int64_t)c->world * slot_packs;
    hipStream_t stream        = aiter::getCurrentHIPStream();
#define AR2S_LAUNCH(NG, F8, SPR, AX)                                                           \
    ar2s::ar2s_rms_kernel<NG, F8, SPR, AX><<<G, rp, 0, stream>>>(c->peers,                     \
                                                             c->rank,                          \
                                                             (const ar2s::u32x4*)inp,          \
                                                             (const ar2s::u32x4*)res_inp,      \
                                                             (const ar2s::u32x4*)w,            \
                                                             (ar2s::u32x4*)out,                \
                                                             (ar2s::u32x4*)res_out,            \
                                                             (ar2s::u32x4*)out_q,              \
                                                             (ar2s::u32x4*)out_s,              \
                                                             (int)rows,                        \
                                                             (int)R,                           \
                                                             slot_packs,                       \
                                                             ag_off,                           \
                                                             (float)eps,                       \
                                                             gemma_norm)
#define AR2S_MODES(NG)                                         \
    if(aux_residual)                                           \
        AR2S_LAUNCH(NG, false, true, true);                    \
    else if(fp8_out && sp_residual)                            \
        AR2S_LAUNCH(NG, true, true, false);                    \
    else if(fp8_out)                                           \
        AR2S_LAUNCH(NG, true, false, false);                   \
    else if(sp_residual)                                       \
        AR2S_LAUNCH(NG, false, true, false);                   \
    else                                                       \
        AR2S_LAUNCH(NG, false, false, false);
    switch(c->world)
    {
    case 2: AR2S_MODES(2); break;
    case 4: AR2S_MODES(4); break;
    case 6: AR2S_MODES(6); break;
    case 8: AR2S_MODES(8); break;
    default: throw std::invalid_argument("ar2s_rms: unsupported world size");
    }
#undef AR2S_MODES
#undef AR2S_LAUNCH
    hipError_t e = hipGetLastError();
    if(e != hipSuccess)
        throw std::runtime_error(std::string("ar2s_rms launch: ") + hipGetErrorString(e));
}

void ar2s_rms(fptr_t ctx,
              int64_t inp,
              int64_t res_inp,
              int64_t w,
              int64_t out,
              int64_t res_out,
              int64_t rows,
              int64_t hidden,
              double eps,
              bool gemma_norm)
{
    ar2s_rms_ex(ctx, inp, res_inp, w, out, res_out, 0, 0, rows, hidden, eps, gemma_norm, false, false, false, 0);
}

} // namespace aiter
