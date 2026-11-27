#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

// Deterministic split-K for the blockscale GEMMs, reduced inside the GEMM.
//
// K batch z of an output tile stores its FP32 partial to slice z of a
// [KBatch, M, N] workspace and then increments the tile's arrival counter. The
// workgroup that arrives last adds the KBatch partials of the tile in batch
// order, rounds once to the output type and resets the counter to zero. So the
// counters are allocated and zeroed once per stream, and there is neither a
// memset nor a second kernel per call.
//
// The atomic path rounds each FP32 partial to BF16/FP16 and atomically adds it
// into the zeroed output. With two partials that gives round(p0 + p1) in either
// order, so KBatch = 2 is deterministic and stays on it. With three or more,
// the intermediate roundings depend on the arrival order.
//
// Only gfx950 uses these paths: the fused kernel and the store specialization
// are compiled for gfx950 only, and the dispatch checks the device arch (the
// host code is shared by all GPU_ARCHS targets). Other archs keep the atomic
// path.
//
// The partials cross XCDs, whose L2 caches are not coherent with each other.
// They are stored and loaded with device-scope (sc1) instructions, the ones
// used for relaxed agent-scope atomic stores and loads, and every workgroup
// waits for its stores before it is counted. A release/acquire fence pair would
// write back and invalidate the whole L2 instead, which costs more than the GEMM.

#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <mutex>
#include <vector>

#include "gemm_dispatch_utils.h"

namespace aiter_blockscale {

inline bool splitk_deterministic_arch() { return get_device_gfx() == "gfx950"; }

// The K batch counts of the tuned split-K rows (splitK 2, 3) and 16.
inline bool use_splitk_fused(int KBatch)
{ return (KBatch == 4 || KBatch == 8 || KBatch == 16) && splitk_deterministic_arch(); }

// One counter per 128-byte cache line, so that tiles do not contend for lines.
constexpr int kSplitKCounterStride = 32;

inline torch::Tensor
splitk_workspace(const torch::Tensor& Y, int KBatch, int M, int N, bool zeroed = false)
{
    auto opts = Y.options().dtype(at::kFloat);
    return zeroed ? at::zeros({KBatch, M, N}, opts) : at::empty({KBatch, M, N}, opts);
}

struct SplitKCounters
{
    uint32_t* ptr;
    torch::Tensor holder; // Defined only for per-call counters.
};

// Zeroed counters for `tiles` tiles. Launches that can overlap must not share
// counters, and the counters reset themselves, so a region serves all later
// calls of its owner. The owner is the stream, or under stream capture the
// capture: PyTorch captures every graph on one stream, and the graphs can be
// replayed concurrently on different streams.
//
// Regions come from per-device chunks that are zeroed once and never freed
// (graphs keep pointers into them). HIP cannot allocate during capture, so a
// chunk is added outside of capture whenever less than half a chunk is free.
// Without a region (no free space under capture, the chunk budget used up, or
// more than kMaxTiles tiles), per-call counters are zeroed in the stream.
inline SplitKCounters splitk_counters(const torch::Tensor& Y, int64_t tiles, hipStream_t stream)
{
    constexpr int64_t kMaxTiles   = 4096;
    constexpr int64_t kChunkTiles = 65536; // 8 MB
    constexpr size_t kMaxChunks   = 32;
    struct Region
    {
        uint32_t* ptr = nullptr;
        int64_t tiles = 0;
    };
    struct Chunk
    {
        uint32_t* ptr;
        int64_t used;
    };
    struct Device
    {
        std::vector<Chunk> chunks;
        // Keyed by stream handle and by capture id.
        std::map<uint64_t, Region> streams;
        std::map<uint64_t, Region> captures;
    };
    static std::mutex mu;
    static std::map<int, Device> devices;

    hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
    unsigned long long capture_id  = 0;
    if(hipStreamGetCaptureInfo(stream, &capture, &capture_id) != hipSuccess)
    {
        (void)hipGetLastError();
    }
    else if(tiles <= kMaxTiles)
    {
        const bool capturing = capture != hipStreamCaptureStatusNone;
        std::lock_guard<std::mutex> lock(mu);
        Device& dev = devices[Y.get_device()];
        if(!capturing && dev.chunks.size() < kMaxChunks &&
           (dev.chunks.empty() || dev.chunks.back().used > kChunkTiles / 2))
        {
            const size_t bytes = kChunkTiles * kSplitKCounterStride * sizeof(uint32_t);
            void* p            = nullptr;
            if(hipMalloc(&p, bytes) == hipSuccess &&
               hipMemsetAsync(p, 0, bytes, stream) == hipSuccess &&
               hipStreamSynchronize(stream) == hipSuccess)
            {
                dev.chunks.push_back({static_cast<uint32_t*>(p), 0});
            }
            else
            {
                (void)hipGetLastError();
                (void)hipFree(p);
            }
        }
        auto& regions      = capturing ? dev.captures : dev.streams;
        const uint64_t key = capturing ? capture_id : reinterpret_cast<uintptr_t>(stream);
        const auto it      = regions.find(key);
        if(it != regions.end() && it->second.tiles >= tiles)
            return {it->second.ptr, {}};
        // A larger region replaces a smaller one, which stays reserved: launched
        // kernels and captured nodes still point into it.
        int64_t size = capturing ? 64 : 256;
        while(size < tiles)
            size *= 2;
        for(Chunk& chunk : dev.chunks)
        {
            if(chunk.used + size <= kChunkTiles)
            {
                uint32_t* ptr = chunk.ptr + chunk.used * kSplitKCounterStride;
                chunk.used += size;
                regions[key] = {ptr, size};
                return {ptr, {}};
            }
        }
    }
    auto holder = at::zeros({tiles * kSplitKCounterStride}, Y.options().dtype(at::kInt));
    return {reinterpret_cast<uint32_t*>(holder.data_ptr<int32_t>()), holder};
}

template <bool IsBF16>
__device__ __forceinline__ uint16_t splitk_round(float x)
{
    if constexpr(IsBF16)
    {
        const uint32_t u = __builtin_bit_cast(uint32_t, x);
        if((u & 0x7fffffffu) > 0x7f800000u)
            return static_cast<uint16_t>((u >> 16) | 0x40u);
        return static_cast<uint16_t>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
    }
    else
    {
        return __builtin_bit_cast(uint16_t, static_cast<_Float16>(x));
    }
}

// Device-scope (sc1) buffer loads.
constexpr int kSplitKSc1 = 16;

using splitk_f4 = float __attribute__((ext_vector_type(4)));

template <int Vec>
using splitk_vec = std::conditional_t<Vec == 4, splitk_f4, float>;

template <int Vec>
__device__ __forceinline__ splitk_vec<Vec> splitk_load(__amdgpu_buffer_rsrc_t r, uint32_t off)
{
    if constexpr(Vec == 4)
        return __builtin_amdgcn_raw_buffer_load_b128(r, off, 0, kSplitKSc1);
    else
        return __builtin_amdgcn_raw_buffer_load_b32(r, off, 0, kSplitKSc1);
}

template <int Vec, bool IsBF16>
__device__ __forceinline__ void splitk_store_out(uint16_t* dst, int64_t ldo, splitk_vec<Vec> acc)
{
    if constexpr(Vec == 4)
    {
        const uint64_t packed = static_cast<uint64_t>(splitk_round<IsBF16>(acc.x)) |
                                (static_cast<uint64_t>(splitk_round<IsBF16>(acc.y)) << 16) |
                                (static_cast<uint64_t>(splitk_round<IsBF16>(acc.z)) << 32) |
                                (static_cast<uint64_t>(splitk_round<IsBF16>(acc.w)) << 48);
        if(ldo % 4 == 0)
        {
            *reinterpret_cast<uint64_t*>(dst) = packed;
        }
        else
        {
#pragma unroll
            for(int v = 0; v < 4; ++v)
                dst[v] = static_cast<uint16_t>(packed >> (16 * v));
        }
    }
    else
    {
        *dst = splitk_round<IsBF16>(acc);
    }
}

// Each thread reduces Iters vectors of the tile, Chunk vectors at a time with
// the loads of all KB partials of a chunk in flight together: the reduction is
// bound by the latency of these loads.
template <int KB, int Vec, int TileM, int TileN, int BlockSize, bool IsBF16>
__device__ __forceinline__ void
splitk_sum_tile(const float* ws, uint16_t* out, int64_t ldo, int M, int N, int m0, int n0)
{
    constexpr int PerRow = TileN / Vec;
    constexpr int Iters  = (TileM * PerRow + BlockSize - 1) / BlockSize;
    constexpr int Chunk  = std::max(1, std::min(Iters, 16 * 4 / (KB * Vec)));
    const uint32_t slice = static_cast<uint32_t>(M) * N * sizeof(float);
    const auto rsrc =
        __builtin_amdgcn_make_buffer_rsrc(const_cast<float*>(ws), 0, slice * KB, 0x00020000);
    for(int it0 = 0; it0 < Iters; it0 += Chunk)
    {
        int m[Chunk], n[Chunk];
        bool valid[Chunk];
        splitk_vec<Vec> p[Chunk][KB];
#pragma unroll
        for(int c = 0; c < Chunk; ++c)
        {
            const int i = (it0 + c) * BlockSize + threadIdx.x;
            m[c]        = m0 + i / PerRow;
            n[c]        = n0 + (i % PerRow) * Vec;
            valid[c]    = (Iters * BlockSize == TileM * PerRow || i < TileM * PerRow) &&
                          (Iters % Chunk == 0 || it0 + c < Iters) && m[c] < M && n[c] < N;
            if(valid[c])
            {
                const uint32_t off = (static_cast<uint32_t>(m[c]) * N + n[c]) * sizeof(float);
#pragma unroll
                for(int s = 0; s < KB; ++s)
                    p[c][s] = splitk_load<Vec>(rsrc, off + s * slice);
            }
        }
#pragma unroll
        for(int c = 0; c < Chunk; ++c)
        {
            if(!valid[c])
                continue;
            splitk_vec<Vec> acc = p[c][0];
#pragma unroll
            for(int s = 1; s < KB; ++s)
                acc += p[c][s];
            splitk_store_out<Vec, IsBF16>(out + static_cast<int64_t>(m[c]) * ldo + n[c], ldo, acc);
        }
    }
}

template <int KB, int TileM, int TileN, int BlockSize, bool IsBF16>
__device__ __forceinline__ void
splitk_sum_tile(const float* ws, uint16_t* out, int64_t ldo, int M, int N, int m0, int n0)
{
    static_assert(TileN % 4 == 0);
    if(N % 4 == 0)
        splitk_sum_tile<KB, 4, TileM, TileN, BlockSize, IsBF16>(ws, out, ldo, M, N, m0, n0);
    else
        splitk_sum_tile<KB, 1, TileM, TileN, BlockSize, IsBF16>(ws, out, ldo, M, N, m0, n0);
}

// Called by every workgroup of K batch blockIdx.z of the tile at (m0, n0) after
// it has issued the device-scope stores of its partial to
// ws + blockIdx.z * M * N. The last of the kbatch workgroups to get here writes
// the rounded sum of all partials of the tile to out.
template <int TileM, int TileN, int BlockSize, bool IsBF16>
__device__ __forceinline__ void splitk_tile_fixup(const float* ws,
                                                  uint16_t* out,
                                                  int64_t ldo,
                                                  uint32_t* counter,
                                                  int M,
                                                  int N,
                                                  int m0,
                                                  int n0,
                                                  int kbatch)
{
    __shared__ bool last[1];
    // vmcnt(0), other counters unchanged: this wave's partial stores are complete.
    __builtin_amdgcn_s_waitcnt(0x0f70);
    __syncthreads();
    if(threadIdx.x == 0)
    {
        const uint32_t prev =
            __hip_atomic_fetch_add(counter, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        last[0] = prev == static_cast<uint32_t>(kbatch - 1);
        if(last[0])
            __hip_atomic_store(counter, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    __syncthreads();
    if(!last[0])
        return;
    if(kbatch == 4)
        splitk_sum_tile<4, TileM, TileN, BlockSize, IsBF16>(ws, out, ldo, M, N, m0, n0);
    else if(kbatch == 8)
        splitk_sum_tile<8, TileM, TileN, BlockSize, IsBF16>(ws, out, ldo, M, N, m0, n0);
    else
        splitk_sum_tile<16, TileM, TileN, BlockSize, IsBF16>(ws, out, ldo, M, N, m0, n0);
}

// CK-Tile path: the partials are reduced by a separate kernel.
//
// In the CK-Tile kernels the fused reduction above made the GEMM several times
// slower (an agent-scope store or atomic after the GEMM costs microseconds
// there, for reasons not yet understood), so CK-Tile writes the partials and
// launches this kernel instead.
inline bool use_splitk_partials(int KBatch) { return KBatch > 2 && splitk_deterministic_arch(); }

// One thread per Vec outputs. With a compile-time KBatch every partial load is
// issued before the first add; the adds run in batch order.
template <typename OutT, int Vec, int KBatchC>
__global__ void splitk_reduce_kernel(
    const float* __restrict__ ws, OutT* __restrict__ y, int M, int N, int64_t ldy, int KBatchR)
{
    using VecT         = std::conditional_t<Vec == 4, float4, float>;
    const int KBatch   = KBatchC > 0 ? KBatchC : KBatchR;
    const int64_t nvec = static_cast<int64_t>(N) / Vec;
    const int64_t idx  = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= nvec * M)
        return;
    const int64_t m  = idx / nvec;
    const int64_t n  = (idx - m * nvec) * Vec;
    const int64_t mn = static_cast<int64_t>(M) * N;
    const float* src = ws + m * N + n;

    float acc[Vec];
    if constexpr(KBatchC > 0)
    {
        VecT part[KBatchC];
#pragma unroll
        for(int s = 0; s < KBatchC; ++s)
            part[s] = *reinterpret_cast<const VecT*>(src + s * mn);
#pragma unroll
        for(int v = 0; v < Vec; ++v)
            acc[v] = reinterpret_cast<const float*>(&part[0])[v];
#pragma unroll
        for(int s = 1; s < KBatchC; ++s)
#pragma unroll
            for(int v = 0; v < Vec; ++v)
                acc[v] += reinterpret_cast<const float*>(&part[s])[v];
    }
    else
    {
#pragma unroll
        for(int v = 0; v < Vec; ++v)
            acc[v] = src[v];
        for(int s = 1; s < KBatch; ++s)
        {
            src += mn;
#pragma unroll
            for(int v = 0; v < Vec; ++v)
                acc[v] += src[v];
        }
    }
    OutT* dst = y + m * ldy + n;
#pragma unroll
    for(int v = 0; v < Vec; ++v)
        dst[v] = static_cast<OutT>(acc[v]);
}

template <typename OutT, int Vec>
inline void launch_splitk_reduce(
    const float* src, OutT* dst, int KBatch, int M, int N, int64_t ldy, hipStream_t stream)
{
    constexpr int threads = 256;
    const int64_t work    = static_cast<int64_t>(M) * (N / Vec);
    const int blocks      = static_cast<int>((work + threads - 1) / threads);
    const auto launch     = [&](auto kernel) {
        kernel<<<blocks, threads, 0, stream>>>(src, dst, M, N, ldy, KBatch);
    };
    switch(KBatch)
    {
    case 4: launch(splitk_reduce_kernel<OutT, Vec, 4>); break;
    case 8: launch(splitk_reduce_kernel<OutT, Vec, 8>); break;
    case 16: launch(splitk_reduce_kernel<OutT, Vec, 16>); break;
    default: launch(splitk_reduce_kernel<OutT, Vec, 0>);
    }
}

template <typename OutT>
inline void
launch_splitk_reduce(const torch::Tensor& ws, torch::Tensor& Y, int KBatch, int M, int N)
{
    const int64_t ldy = Y.stride(0);
    auto stream       = at::hip::getCurrentHIPStream();
    const float* src  = ws.data_ptr<float>();
    OutT* dst         = reinterpret_cast<OutT*>(Y.data_ptr());
    if(N % 4 == 0)
        launch_splitk_reduce<OutT, 4>(src, dst, KBatch, M, N, ldy, stream);
    else
        launch_splitk_reduce<OutT, 1>(src, dst, KBatch, M, N, ldy, stream);
}

// Y[m, n] = sum over s in order of ws[s, m, n], rounded once to Y's dtype.
inline void splitk_reduce(const torch::Tensor& ws, torch::Tensor& Y, int KBatch, int M, int N)
{
    TORCH_CHECK(Y.stride(-1) == 1, "split-K reduce: Y inner dim must be contiguous");
    if(Y.scalar_type() == at::kBFloat16)
        launch_splitk_reduce<c10::BFloat16>(ws, Y, KBatch, M, N);
    else if(Y.scalar_type() == at::kHalf)
        launch_splitk_reduce<c10::Half>(ws, Y, KBatch, M, N);
    else
        TORCH_CHECK(false, "split-K reduce: unsupported output dtype ", Y.scalar_type());
}

} // namespace aiter_blockscale
