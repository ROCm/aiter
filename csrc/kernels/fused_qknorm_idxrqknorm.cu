// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Horizontally fused Q/K RMSNorm + partial NeoX RoPE, with optional
// sparse-layer KV/index-cache insert. This ports the AME/vLLM operator into
// aiter's pybind/JIT extension style.

#include "fused_qknorm_idxrqknorm.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include "aiter_dispatch.h"
#include "aiter_stream.h"
#include "aiter_opus_plus.h"
#include "opus/opus.hpp"
#include "quant_utils.cuh"

namespace aiter {
namespace fused_qknorm_idxrqknorm_ops {

constexpr int kHeadDim = 128;
constexpr int kNumLanes = 32;
constexpr int kElemsPerLane = kHeadDim / kNumLanes;
constexpr int kBlockSize = 256; // every launch of the kernel
// A launch of kMultiRowMinRowsPerCu or more (token, slot) rows per CU runs faster with
// kMultiRows rows per warp (more rows in flight per wave) than with a warp per row (gfx1250).
constexpr int kMultiRows = 4;
constexpr int64_t kMultiRowMinRowsPerCu = 128;

// The multi-row kernels and the fused split-K partials load are built for gfx1250 only.
static bool isGfx1250()
{
    static const bool is_gfx1250 = get_gpu_arch() == "gfx1250"; // queries the driver
    return is_gfx1250;
}

// On gfx12 a 32-lane warp is a whole wave32 wave, so what depends only on the warp
// index is wave-uniform: readfirstlane keeps it in SGPRs (scalar loads of the
// per-token tables, scalar branches on the slot kind). A wave64 wave holds two warps.
__device__ __forceinline__ int warpUniform(int v)
{
#if defined(__GFX12__)
    return __builtin_amdgcn_readfirstlane(v);
#else
    return v;
#endif
}

// __shfl_xor(v, kMask, 32) for a warp whose lanes are all active.
template <int kMask>
__device__ __forceinline__ float shflXor(float v)
{
#if defined(__GFX12__)
    static_assert(kMask == 1 || kMask == 2 || kMask == 4 || kMask == 8 || kMask == 16);
    if constexpr(kMask == 16)
    {
        return warp_permlanex16(v); // lane L reads lane L of the other 16-lane row
    }
    else
    {
        // DPP row_xmask: lane L reads lane L ^ kMask of its row. No lane reads outside
        // its row, so bound_ctrl changes nothing; it lets the compiler fold the move
        // into the add or max that consumes it.
        return opus::mov_dpp(v, opus::number<0x160 | kMask>{});
    }
#else
    return __shfl_xor(v, kMask, 32);
#endif
}

__device__ __forceinline__ float warpReduceSum(float val)
{
    val += shflXor<16>(val);
    val += shflXor<8>(val);
    val += shflXor<4>(val);
    val += shflXor<2>(val);
    val += shflXor<1>(val);
    return val;
}

__device__ __forceinline__ float warpReduceMax(float val)
{
    val = fmaxf(val, shflXor<16>(val));
    val = fmaxf(val, shflXor<8>(val));
    val = fmaxf(val, shflXor<4>(val));
    val = fmaxf(val, shflXor<2>(val));
    val = fmaxf(val, shflXor<1>(val));
    return val;
}

// slot / block_size and slot % block_size for slot >= 0 (block_size, and x where the
// V layout splits by it, is a power of two in every cache layout in use).
__device__ __forceinline__ void splitSlot(int64_t slot, int block_size, int64_t& b, int64_t& t)
{
    if((block_size & (block_size - 1)) == 0)
    {
        b = slot >> __builtin_ctz(block_size);
        t = slot & (block_size - 1);
    }
    else
    {
        b = slot / block_size;
        t = slot % block_size;
    }
}

// Per-token dynamic quant scale = amax / fp8_max so the largest |value| maps to the
// fp8 max. The fp8 max is arch-dependent: OCP e4m3fn (448) on gfx950+, e4m3fnuz (240)
// on gfx942. opus::finfo<cache_t>::max() resolves this at compile time per arch and
// matches the opus::cast<fp8> hardware clamp range, so do NOT hardcode 448 here.
template <typename cache_t>
__device__ __forceinline__ float fp8Max()
{
    return static_cast<float>(opus::finfo<cache_t>::max());
}

template <typename scalar_t>
__device__ __forceinline__ void loadElems(const scalar_t* __restrict__ src,
                                          float (&elems)[kElemsPerLane])
{
#pragma unroll
    for(int i = 0; i < kElemsPerLane; ++i)
    {
        elems[i] = static_cast<float>(src[i]);
    }
}

// A lane's kElemsPerLane 16-bit elements as the 8 bytes they occupy in memory.
template <typename scalar_t>
__device__ __forceinline__ uint2 packRawElems(const scalar_t (&raw)[kElemsPerLane])
{
    static_assert(sizeof(scalar_t) == 2 && kElemsPerLane == 4, "four 16-bit elements");
    uint32_t w[kElemsPerLane];
#pragma unroll
    for(int i = 0; i < kElemsPerLane; ++i)
    {
        w[i] = __builtin_bit_cast(uint16_t, raw[i]);
    }
    return make_uint2(w[0] | (w[1] << 16), w[2] | (w[3] << 16));
}

// The inverse of packRawElems: four 16-bit elements from the 8 bytes they occupy.
template <typename scalar_t>
__device__ __forceinline__ void unpackRawElems(uint2 w, scalar_t (&raw)[kElemsPerLane])
{
    static_assert(sizeof(scalar_t) == 2 && kElemsPerLane == 4, "four 16-bit elements");
    const uint32_t h[kElemsPerLane] = {w.x & 0xffffu, w.x >> 16, w.y & 0xffffu, w.y >> 16};
#pragma unroll
    for(int i = 0; i < kElemsPerLane; ++i)
    {
        raw[i] = __builtin_bit_cast(scalar_t, static_cast<uint16_t>(h[i]));
    }
}

// Four packed 16-bit elements (one 8-byte load), widened to float.
template <typename scalar_t>
__device__ __forceinline__ void unpackElems(uint2 w, float (&elems)[kElemsPerLane])
{
    scalar_t raw[kElemsPerLane];
    unpackRawElems(w, raw);
    loadElems(raw, elems);
}

// static_cast<scalar_t> of each element (round to nearest even), as the 8 bytes the
// results occupy in memory.
template <typename scalar_t>
__device__ __forceinline__ uint2 roundPackElems(const float (&elems)[kElemsPerLane])
{
#if defined(__gfx1250__)
    if constexpr(std::is_same_v<scalar_t, hip_bfloat16>)
    {
        // hip_bfloat16 rounds in software; the hardware rounds two floats per instruction
        // to the same bits. Only a NaN can differ (in its payload): every float rounded
        // here is an FP32 result, which this module flushes, so none is subnormal.
        typedef float f32x2 __attribute__((ext_vector_type(2)));
        typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));
        const auto round2 = [](float lo, float hi) {
            return __builtin_bit_cast(uint32_t,
                                      __builtin_convertvector(f32x2{lo, hi}, bf16x2));
        };
        return make_uint2(round2(elems[0], elems[1]), round2(elems[2], elems[3]));
    }
#endif
    scalar_t v[kElemsPerLane];
#pragma unroll
    for(int i = 0; i < kElemsPerLane; ++i)
    {
        v[i] = static_cast<scalar_t>(elems[i]);
    }
    return packRawElems(v);
}

template <typename scalar_t, bool kSplitStore = false>
__device__ __forceinline__ void storeElems(scalar_t* __restrict__ dst,
                                           const float (&elems)[kElemsPerLane])
{
    const uint2 packed = roundPackElems<scalar_t>(elems);
#if defined(__gfx1250__)
    if constexpr(kSplitStore)
    {
        // Two 32-bit stores: with one 64-bit store per lane, the one-row non-insert kernel
        // runs 0.3-0.7 us longer per call on gfx1250 (HIP graph, M <= 64). The compiler merges
        // plain stores back into one 64-bit store; the empty asm, which takes the address
        // as an operand and so may read dst, keeps them apart.
        uint32_t* const words = reinterpret_cast<uint32_t*>(dst);
        words[0] = packed.x;
        asm volatile("" : : "v"(words) : "memory");
        words[1] = packed.y;
        return;
    }
#endif
    *reinterpret_cast<uint2*>(dst) = packed;
}

// FP32 split-K partials of a bf16 qkv that a split-K GEMM left unreduced:
// [num_splits, >= num_tokens, row], the splits split_stride elements apart.
// num_splits == 0 means qkv holds the row.
struct QkvPartials
{
    const float* ptr     = nullptr;
    int num_splits       = 0;
    int64_t split_stride = 0;
};

// The reduce kernel adds with f32 denormals; this module flushes them
// (-fgpu-flush-denormals-to-zero). A lane whose partials or running sums leave the
// normal range redoes its sum here exactly: each fp32 add is exact in double (both
// operands are fp32, and where they are 2^29 apart the smaller one cannot change
// the rounding), then rounded to fp32 with gradual underflow. It keeps the result
// bitwise for every input; besides subnormals it also runs for a sum that is exactly
// zero (a zero row, cancelling partials), which reads as |acc| < 2^-126.
__device__ __forceinline__ double exactF32ToDouble(float f)
{
    const uint32_t u   = __builtin_bit_cast(uint32_t, f);
    const uint32_t mag = u & 0x7fffffffu;
    if(mag != 0 && mag < 0x00800000u) // subnormal: the conversion would flush it
    {
        const double v = static_cast<double>(mag) * 0x1p-149;
        return (u >> 31) ? -v : v;
    }
    return static_cast<double>(f);
}

__device__ __forceinline__ double roundToF32(double x)
{
    if(fabs(x) < 0x1p-126) // fp32 subnormals are the multiples of 2^-149
    {
        return rint(x * 0x1p149) * 0x1p-149;
    }
    return static_cast<double>(static_cast<float>(x));
}

__device__ __forceinline__ uint16_t bf16BitsOfF32(double x)
{
    if(fabs(x) < 0x1p-126) // round the subnormal's fp32 bits to nearest even
    {
        const uint32_t mag = static_cast<uint32_t>(fabs(x) * 0x1p149);
        __builtin_assume(mag < 0x00800000u); // a subnormal: not inf/NaN, as the helper checks
        return opus::fp32_to_bf16_rtn_raw(
            __builtin_bit_cast(float, (signbit(x) ? 0x80000000u : 0u) | mag));
    }
    return __builtin_bit_cast(uint16_t, static_cast<__bf16>(static_cast<float>(x)));
}

__device__ __forceinline__ bool isSubnormalF32(float f)
{
    const uint32_t mag = __builtin_bit_cast(uint32_t, f) & 0x7fffffffu;
    return mag != 0 && mag < 0x00800000u;
}

// One lane's kElemsPerLane qkv elements reduced from kSplits partials as the gfx1250
// split-K GEMM's reduce kernel (gemm_a8w8_splitk_reduce) does it: every load first,
// an FP32 sum in split order,
// then one round-to-nearest-even to bf16 (the same fptrunc the reduce kernel
// emits), so the result matches the reduced qkv bit for bit.
template <int kSplits>
__device__ __forceinline__ void reducePartials(const float* __restrict__ src,
                                               int64_t split_stride,
                                               uint16_t (&raw)[kElemsPerLane])
{
    static_assert(kElemsPerLane == 4, "one float4 per split and lane");
    float4 parts[kSplits];
#pragma unroll
    for(int s = 0; s < kSplits; ++s)
    {
        parts[s] = *reinterpret_cast<const float4*>(src + s * split_stride);
    }
    const auto part = [&](int s, int i) {
        return i == 0 ? parts[s].x : i == 1 ? parts[s].y : i == 2 ? parts[s].z : parts[s].w;
    };
#pragma unroll
    for(int i = 0; i < kElemsPerLane; ++i)
    {
        float acc       = part(0, i);
        bool subnormals = isSubnormalF32(acc);
#pragma unroll
        for(int s = 1; s < kSplits; ++s)
        {
            acc += part(s, i);
            // A sum flushed to zero reads as |acc| < 2^-126 too (a NaN does not).
            subnormals |= isSubnormalF32(part(s, i)) || fabsf(acc) < 0x1p-126f;
        }
        raw[i] = __builtin_bit_cast(uint16_t, static_cast<__bf16>(acc));
        if(subnormals && isfinite(acc))
        {
            double exact = exactF32ToDouble(part(0, i));
#pragma unroll
            for(int s = 1; s < kSplits; ++s)
            {
                exact = roundToF32(exact + exactF32ToDouble(part(s, i)));
            }
            raw[i] = bf16BitsOfF32(exact);
        }
    }
}

// The lane's kElemsPerLane bf16 qkv elements, reduced from the split-K partials and
// written back to the row, so qkv ends up as the standalone reduce leaves it.
__device__ __forceinline__ void reduceRawElems(hip_bfloat16* __restrict__ row,
                                               const QkvPartials& partials,
                                               int64_t partial_off,
                                               hip_bfloat16 (&raw)[kElemsPerLane])
{
    uint16_t bits[kElemsPerLane];
    const float* src = partials.ptr + partial_off;
    switch(partials.num_splits)
    {
    case 2: reducePartials<2>(src, partials.split_stride, bits); break;
    case 4: reducePartials<4>(src, partials.split_stride, bits); break;
    default: reducePartials<8>(src, partials.split_stride, bits); break;
    }
#pragma unroll
    for(int i = 0; i < kElemsPerLane; ++i)
    {
        raw[i] = __builtin_bit_cast(hip_bfloat16, bits[i]);
    }
    *reinterpret_cast<uint2*>(row) = packRawElems(raw);
}

// A call whose mode does not sum the partials in its own pass (see
// fused_qknorm_idxrqknorm_impl) reduces them into qkv with this kernel first, the same
// sum, and then runs as on a reduced qkv. qkv is dense [num_tokens, row], the partials
// rows have the same stride.
__global__ void reduceQkvPartialsKernel(hip_bfloat16* __restrict__ qkv,
                                        QkvPartials partials,
                                        int64_t num_elems)
{
    const int64_t off =
        (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * kElemsPerLane;
    if(off < num_elems)
    {
        hip_bfloat16 raw[kElemsPerLane];
        reduceRawElems(qkv + off, partials, off, raw);
    }
}

// The lane's kElemsPerLane elements are contiguous in dst: one store.
template <typename scalar_t, typename cache_t, vllm::Fp8KVCacheDataType kv_dt>
__device__ __forceinline__ void storeCacheElems(cache_t* __restrict__ dst,
                                                const float (&elems)[kElemsPerLane],
                                                float scale)
{
    if constexpr(kv_dt == vllm::Fp8KVCacheDataType::kAuto)
    {
        storeElems(dst, elems);
    }
    else if constexpr(sizeof(cache_t) == 1 && sizeof(scalar_t) == 2 && kElemsPerLane == 4)
    {
        float rounded[kElemsPerLane];
        unpackElems<scalar_t>(roundPackElems<scalar_t>(elems), rounded);
        uint32_t packed = 0;
#pragma unroll
        for(int i = 0; i < kElemsPerLane; ++i)
        {
            const cache_t q = opus::cast<cache_t>(rounded[i] / scale);
            packed |= static_cast<uint32_t>(__builtin_bit_cast(uint8_t, q)) << (8 * i);
        }
        *reinterpret_cast<uint32_t*>(dst) = packed;
    }
    else
    {
#pragma unroll
        for(int i = 0; i < kElemsPerLane; ++i)
        {
            const scalar_t rounded = static_cast<scalar_t>(elems[i]);
            dst[i] = opus::cast<cache_t>(static_cast<float>(rounded) / scale);
        }
    }
}

// index_q gather: qkv dtype, or unit-scale e4m3 (4787's q_idx contract; no scale tensor).
// When neither gather buffer is provided, write the packed qkv index-Q row (inplace).
template <typename scalar_t, bool kSplitStore = false>
__device__ __forceinline__ void storeIndexQElems(scalar_t* __restrict__ index_q_out,
                                                 opus::fp8_t* __restrict__ index_q_fp8_out,
                                                 scalar_t* __restrict__ fallback_ptr,
                                                 const float (&elems)[kElemsPerLane],
                                                 int64_t token_idx,
                                                 int niq,
                                                 int iq_head,
                                                 int dim_base)
{
    const int64_t off = token_idx * static_cast<int64_t>(niq) * kHeadDim +
                        static_cast<int64_t>(iq_head) * kHeadDim + dim_base;
    if(index_q_fp8_out != nullptr)
    {
        storeCacheElems<scalar_t, opus::fp8_t, vllm::Fp8KVCacheDataType::kFp8E4M3>(
            index_q_fp8_out + off, elems, 1.0f);
    }
    else if(index_q_out != nullptr)
    {
        storeElems<scalar_t, kSplitStore>(index_q_out + off, elems);
    }
    else if(fallback_ptr != nullptr)
    {
        storeElems<scalar_t, kSplitStore>(fallback_ptr, elems);
    }
}

template <typename scalar_t, typename cache_t, vllm::Fp8KVCacheDataType kv_dt>
__device__ __forceinline__ void copyOrQuantizeCacheElems(const scalar_t (&src)[kElemsPerLane],
                                                         cache_t* __restrict__ dst,
                                                         float scale)
{
    if constexpr(kv_dt == vllm::Fp8KVCacheDataType::kAuto)
    {
        *reinterpret_cast<uint2*>(dst) = packRawElems(src);
    }
    else
    {
        float elems[kElemsPerLane];
        loadElems(src, elems);
        storeCacheElems<scalar_t, cache_t, kv_dt>(dst, elems, scale);
    }
}

// Strided V store for the SHUFFLE (asm) layout: the kElemsPerLane elements owned
// by a lane are `stride` apart in the destination cache (V layout interleaves the
// intra-page position x-remainder between consecutive head-dim elements). Source
// is the raw (no norm/rope) qkv row, contiguous.
template <typename scalar_t, typename cache_t, vllm::Fp8KVCacheDataType kv_dt>
__device__ __forceinline__ void storeCacheElemsStrided(const scalar_t* __restrict__ src,
                                                       cache_t* __restrict__ dst,
                                                       int64_t stride,
                                                       float scale)
{
#pragma unroll
    for(int i = 0; i < kElemsPerLane; ++i)
    {
        if constexpr(kv_dt == vllm::Fp8KVCacheDataType::kAuto)
        {
            dst[i * stride] = static_cast<cache_t>(src[i]);
        }
        else
        {
            dst[i * stride] = opus::cast<cache_t>(static_cast<float>(src[i]) / scale);
        }
    }
}

// What normAndRope reads besides the row: the lane's norm weights, cos and sin, four
// 16-bit values each. Loaded before the row is used, so their latency overlaps the row
// load; kept packed until normAndRope widens them.
struct NormRopeInputs
{
    uint2 w;
    uint2 c;
    uint2 s;
};

// The lane's cos and sin (dims dim0.. of the row's rotation); lanes past rotary_dim load
// element 0 and do not use it.
template <typename scalar_t>
__device__ __forceinline__ void loadCosSin(NormRopeInputs& in,
                                           int dim0,
                                           int rotary_dim,
                                           const scalar_t* __restrict__ cos_ptr)
{
    const int half   = rotary_dim / 2;
    const int i_base = dim0 >= rotary_dim ? 0 : dim0 < half ? dim0 : dim0 - half;
    in.c = *reinterpret_cast<const uint2*>(cos_ptr + i_base);
    in.s = *reinterpret_cast<const uint2*>(cos_ptr + half + i_base);
}

// NeoX rotation of the lane's elements (dims dim0..) on the lanes below rotary_dim;
// partner holds each element's rotate-half partner (dim +- rotary_dim / 2).
template <typename scalar_t>
__device__ __forceinline__ void rotateElems(float (&elems)[kElemsPerLane],
                                            const float (&partner)[kElemsPerLane],
                                            const NormRopeInputs& in,
                                            int dim0,
                                            int rotary_dim)
{
    float c[kElemsPerLane];
    float s[kElemsPerLane];
    unpackElems<scalar_t>(in.c, c);
    unpackElems<scalar_t>(in.s, s);
    const bool rotate     = dim0 < rotary_dim;
    const bool first_half = dim0 < rotary_dim / 2;
#pragma unroll
    for(int i = 0; i < kElemsPerLane; ++i)
    {
        const float rotated = first_half ? elems[i] * c[i] - partner[i] * s[i]
                                         : elems[i] * c[i] + partner[i] * s[i];
        elems[i] = rotate ? rotated : elems[i];
    }
}

// Rotary embedding of the lane's elements. The partner of dim0 lives rotary_dim / 8
// lanes away; every lane of the warp runs this.
template <typename scalar_t>
__device__ __forceinline__ void ropeElems(float (&elems)[kElemsPerLane],
                                          const NormRopeInputs& in,
                                          int lane_id,
                                          int rotary_dim)
{
    const int dim0 = lane_id * kElemsPerLane;
    float partner[kElemsPerLane];
#if defined(__GFX12__)
    // A power-of-two lane distance d pairs lane L < d with L + d == L ^ d, so the
    // partner is a cross-lane xor (shflXor) instead of an LDS permute (__shfl).
    switch(rotary_dim)
    {
#define ROPE_XOR(DIM, MASK)                                          \
    case DIM:                                                        \
        _Pragma("unroll") for(int i = 0; i < kElemsPerLane; ++i)     \
        {                                                            \
            partner[i] = shflXor<MASK>(elems[i]);                    \
        }                                                            \
        rotateElems<scalar_t>(elems, partner, in, dim0, DIM);        \
        return;
        ROPE_XOR(8, 1)
        ROPE_XOR(16, 2)
        ROPE_XOR(32, 4)
        ROPE_XOR(64, 8)
        ROPE_XOR(128, 16)
#undef ROPE_XOR
    default: break;
    }
#endif
    const int half         = rotary_dim / 2;
    const int partner_lane = dim0 < half ? (dim0 + half) / kElemsPerLane
                                         : (dim0 - half) / kElemsPerLane;
#pragma unroll
    for(int i = 0; i < kElemsPerLane; ++i)
    {
        partner[i] = __shfl(elems[i], partner_lane, 32);
    }
    rotateElems<scalar_t>(elems, partner, in, dim0, rotary_dim);
}

template <typename scalar_t>
__device__ __forceinline__ void normAndRope(float (&elems)[kElemsPerLane],
                                            int lane_id,
                                            float eps,
                                            const NormRopeInputs& in,
                                            bool do_rope,
                                            int rotary_dim)
{
    float w[kElemsPerLane];
    unpackElems<scalar_t>(in.w, w);
    float sumsq = 0.0f;
#pragma unroll
    for(int i = 0; i < kElemsPerLane; ++i)
    {
        sumsq += elems[i] * elems[i];
    }
    sumsq = warpReduceSum(sumsq);
    const float rms_rcp = rsqrtf(sumsq / static_cast<float>(kHeadDim) + eps);
#pragma unroll
    for(int i = 0; i < kElemsPerLane; ++i)
    {
        elems[i] = elems[i] * rms_rcp * (1.0f + w[i]);
    }

    if(do_rope)
    {
        ropeElems<scalar_t>(elems, in, lane_id, rotary_dim);
    }
}

// ---- MXFP4 row write (index K, main K/V) -----------------------------------------------
// One 128-token page is [128 x 64 B e2m1][128 x 4 B e8m0], 8704 B. Low nibble = even element,
// one e8m0 per 32 elements.
constexpr int kFp4PageTokens = 128;
constexpr int kFp4DataBytes  = kHeadDim / 2;  // packed e2m1 bytes per token
constexpr int kFp4ScaleBytes = kHeadDim / 32; // e8m0 bytes per token
constexpr int kFp4PageBytes  = kFp4PageTokens * (kFp4DataBytes + kFp4ScaleBytes);
// kElemsPerLane lanes of a 32-lane warp cover one head row; this many make one e8m0 group.
constexpr int kFp4LanesPerGroup = 32 / kElemsPerLane;

// ceil(log2(amax / 6)) + 127: copy of indexer_fp4_scale_e8m0 in cache_kernels.cu, pinned to
// per_1x32_f4_quant by the mxfp4_scale_rule mode of op_tests/test_fused_qknorm_idxrqknorm.py.
__device__ __forceinline__ uint8_t mxfp4ScaleE8m0(const float amax)
{
    constexpr float fp4_max     = static_cast<float>(opus::finfo<opus::fp4_t>::max());
    constexpr float inv_fp4_max = 1.0f / fp4_max;
    constexpr float eps_amax    = fp4_max * __builtin_bit_cast(float, 0x00800000u);

    const uint32_t bits = __builtin_bit_cast(uint32_t, fmaxf(amax, eps_amax) * inv_fp4_max);
    uint8_t exponent    = (bits >> 23) & 0xFF;
    if(exponent == 0xFF)
    {
        return exponent;
    }
    if(bits & 0x7FFFFF)
    {
        exponent += 1;
    }
    return exponent;
}

// Quantize one lane's kElemsPerLane elements of a head row into row `slot` of the MXFP4
// pages at `cache` (page = slot / 128 at s_block bytes, + head_off bytes). The elements are
// rounded to scalar_t first, so the bytes are the quantization of the row "auto" would store.
template <typename scalar_t>
__device__ __forceinline__ void storeMxfp4Row(uint8_t* __restrict__ cache,
                                              int64_t slot,
                                              int64_t s_block,
                                              int64_t head_off,
                                              const float (&elems)[kElemsPerLane],
                                              int lane_id)
{
    uint8_t* page = cache + (slot / kFp4PageTokens) * s_block + head_off;
    const int tok = static_cast<int>(slot % kFp4PageTokens);
    float v[kElemsPerLane];
    float local_amax = 0.0f;
#pragma unroll
    for(int i = 0; i < kElemsPerLane; ++i)
    {
        v[i]       = static_cast<float>(static_cast<scalar_t>(elems[i]));
        local_amax = fmaxf(local_amax, fabsf(v[i]));
    }
#pragma unroll
    for(int mask = 1; mask < kFp4LanesPerGroup; mask <<= 1)
    {
        local_amax = fmaxf(local_amax, __shfl_xor(local_amax, mask, 32));
    }
    const uint8_t e8m0 = mxfp4ScaleE8m0(local_amax);
    const float scale  = __builtin_bit_cast(float, static_cast<uint32_t>(e8m0) << 23);

    uint16_t packed = 0;
#pragma unroll
    for(int j = 0; j < kElemsPerLane / 2; ++j)
    {
        const opus::fp32x2_t pair = {v[2 * j], v[2 * j + 1]};
        const uint8_t byte = __builtin_bit_cast(uint8_t, scaled_cast<opus::fp4_t>(pair, scale));
        packed |= static_cast<uint16_t>(byte) << (8 * j);
    }
    *reinterpret_cast<uint16_t*>(page + tok * kFp4DataBytes + lane_id * (kElemsPerLane / 2)) =
        packed;
    if(lane_id % kFp4LanesPerGroup == 0)
    {
        page[kFp4PageTokens * kFp4DataBytes + tok * kFp4ScaleBytes + lane_id / kFp4LanesPerGroup] =
            e8m0;
    }
}

// ---- NVFP4 row write (index K, main K/V) -----------------------------------------------
// e2m1 data + one e4m3 scale per 16 elements (4 lanes), no per-token scale. The arithmetic is
// integer/compare arithmetic (see the torch reference in op_tests/test_fused_qknorm_nvfp4.py),
// not a hardware convert, so the bytes are the same on every arch. Pages:
//   index page, 9216 B : [128 tok x 64 B e2m1][128 tok x 8 B e4m3]
//   KV page, 2*nkv*9216 B : [K: nkv x 8192 B data | nkv x 1024 B scale][V: same]
//     K data  t*64 + d/2           K scale in MFMA lane order (nvKScaleOff)
//     V data  ((t/8)*128 + d)*4 B, token t%8 is nibble t%8 of that 32-bit word; V scale g*128 + t
constexpr int kNvLanesPerGroup = 16 / kElemsPerLane;
constexpr int kNvTokens        = 128;
constexpr int kNvHeadData      = kNvTokens * (kHeadDim / 2);  // 8192
constexpr int kNvHeadScale     = kNvTokens * (kHeadDim / 16); // 1024
constexpr int kNvHeadSide      = kNvHeadData + kNvHeadScale;  // 9216 == an index page
constexpr int kNvIndexPage     = kNvHeadSide;

// e4m3fn byte of x in [0, 448], round to nearest even.
__device__ __forceinline__ uint32_t nvE4m3Byte(float x)
{
    const uint32_t bits   = __builtin_bit_cast(uint32_t, x);
    const uint32_t normal = ((bits + 0x7FFFFu + ((bits >> 20) & 1u)) >> 20) - 960u;
    // subnormal (x < 2^-6): m * 2^-9, m = RNE(x * 512) by the 1.5 * 2^23 trick
    const uint32_t sub =
        static_cast<uint32_t>(static_cast<int>((x * 512.0f + 12582912.0f) - 12582912.0f));
    return x >= 0.015625f ? normal : sub;
}

__device__ __forceinline__ float nvE4m3Value(uint32_t byte)
{
    const uint32_t e = byte >> 3, m = byte & 7u;
    return e > 0 ? __builtin_bit_cast(float, ((e + 120u) << 23) | (m << 20))
                 : static_cast<float>(m) * 0.001953125f;
}

// e2m1 code (sign in bit 3): nearest grid value, ties to the even code.
__device__ __forceinline__ uint32_t nvE2m1Code(float y)
{
    const float a = fabsf(y);
    const int mag = (a > 0.25f) + (a >= 0.75f) + (a > 1.25f) + (a >= 1.75f) + (a > 2.5f) +
                    (a >= 3.5f) + (a > 5.0f);
    return static_cast<uint32_t>(mag) | ((__builtin_bit_cast(uint32_t, y) >> 31) << 3);
}

// One lane's kElemsPerLane elements (already scalar_t-rounded) -> codes + the group's scale byte.
__device__ __forceinline__ uint32_t nvQuantLane(const float (&v)[kElemsPerLane],
                                                float gs,
                                                uint32_t (&code)[kElemsPerLane])
{
    float amax = 0.0f;
#pragma unroll
    for(int i = 0; i < kElemsPerLane; ++i)
    {
        amax = fmaxf(amax, fabsf(v[i]));
    }
#pragma unroll
    for(int mask = 1; mask < kNvLanesPerGroup; mask <<= 1)
    {
        amax = fmaxf(amax, __shfl_xor(amax, mask, 32));
    }
    const float one_sixth = 0.1666666716337204f; // float32(1/6), as the torch reference
    const float sf        = fminf(amax * one_sixth * __fdiv_rn(1.0f, gs), 448.0f);
    const uint32_t sbyte  = nvE4m3Byte(sf);
    const float sv        = nvE4m3Value(sbyte);
    const float quant     = sv != 0.0f ? __fdiv_rn(1.0f, sv * gs) : 0.0f;
#pragma unroll
    for(int i = 0; i < kElemsPerLane; ++i)
    {
        code[i] = nvE2m1Code(v[i] * quant);
    }
    return sbyte;
}

__device__ __forceinline__ uint16_t nvPackBytes(const uint32_t (&code)[kElemsPerLane])
{
    return static_cast<uint16_t>(code[0] | (code[1] << 4) | (code[2] << 8) | (code[3] << 12));
}

// token-major data (low nibble = even element); the callers write the scale bytes.
__device__ __forceinline__ void
nvStoreRowData(uint8_t* __restrict__ data_row, int lane_id, const uint32_t (&code)[kElemsPerLane])
{
    *reinterpret_cast<uint16_t*>(data_row + lane_id * (kElemsPerLane / 2)) = nvPackBytes(code);
}

// K scale byte offset (MFMA lane order) of token pos, group g, inside a head's scale region.
__device__ __forceinline__ int nvKScaleOff(int pos, int g)
{
    const int r = pos % 32;
    const int n = 4 * (r / 8) + r % 4;
    return ((((pos / 32) * 4 + g / 2) * 16 + n) * 2 + (r % 8) / 4) * 2 + g % 2;
}

// Index-K row of token `pos` in a 9216 B page (main K uses storeNvfp4KRow).
template <typename scalar_t>
__device__ __forceinline__ void storeNvfp4IndexRow(uint8_t* __restrict__ page,
                                                   int pos,
                                                   const float (&elems)[kElemsPerLane],
                                                   int lane_id)
{
    float v[kElemsPerLane];
#pragma unroll
    for(int i = 0; i < kElemsPerLane; ++i)
    {
        v[i] = static_cast<float>(static_cast<scalar_t>(elems[i]));
    }
    uint32_t code[kElemsPerLane];
    const uint32_t sb = nvQuantLane(v, 1.0f, code);
    nvStoreRowData(page + pos * (kHeadDim / 2), lane_id, code);
    if(lane_id % kNvLanesPerGroup == 0)
    {
        page[kNvHeadData + pos * (kHeadDim / 16) + lane_id / kNvLanesPerGroup] = sb;
    }
}

template <typename scalar_t>
__device__ __forceinline__ void storeNvfp4KRow(uint8_t* __restrict__ page,
                                               int nkv,
                                               int head,
                                               int pos,
                                               float gs,
                                               const float (&elems)[kElemsPerLane],
                                               int lane_id)
{
    float v[kElemsPerLane];
#pragma unroll
    for(int i = 0; i < kElemsPerLane; ++i)
    {
        v[i] = static_cast<float>(static_cast<scalar_t>(elems[i]));
    }
    uint32_t code[kElemsPerLane];
    const uint32_t sb = nvQuantLane(v, gs, code);
    nvStoreRowData(page + head * kNvHeadData + pos * (kHeadDim / 2), lane_id, code);
    if(lane_id % kNvLanesPerGroup == 0)
    {
        page[nkv * kNvHeadData + head * kNvHeadScale +
             nvKScaleOff(pos, lane_id / kNvLanesPerGroup)] = sb;
    }
}

// V of token `pos` alone: its nibble of each element's word, atomic and/or (other tokens of the
// octet, in this or another warp, share the word).
template <typename scalar_t>
__device__ __forceinline__ void storeNvfp4VToken(uint8_t* __restrict__ page,
                                                 int nkv,
                                                 int head,
                                                 int pos,
                                                 float gs,
                                                 const scalar_t* __restrict__ row,
                                                 int lane_id)
{
    float v[kElemsPerLane];
    loadElems(row + lane_id * kElemsPerLane, v);
    uint32_t code[kElemsPerLane];
    const uint32_t sb = nvQuantLane(v, gs, code);
    uint32_t* words   = reinterpret_cast<uint32_t*>(page + nkv * kNvHeadSide + head * kNvHeadData +
                                                  (pos / 8) * 512) +
                      lane_id * kElemsPerLane;
    const int shift = ((pos % 8) / 2) * 8 + (pos % 2) * 4;
#pragma unroll
    for(int i = 0; i < kElemsPerLane; ++i)
    {
        atomicAnd(words + i, ~(0xFu << shift));
        atomicOr(words + i, code[i] << shift);
    }
    if(lane_id % kNvLanesPerGroup == 0)
    {
        page[nkv * (kNvHeadSide + kNvHeadData) + head * kNvHeadScale +
             (lane_id / kNvLanesPerGroup) * kNvTokens + pos] = sb;
    }
}

template <typename scalar_t,
          typename cache_t,
          typename index_cache_t,
          vllm::Fp8KVCacheDataType kv_dt,
          vllm::Fp8KVCacheDataType index_dt,
          bool kPerTokenKVScale,
          bool kStaticKVScale,
          bool kIsSparse,
          bool kProcessIndex,
          bool kInsertKV,
          bool kAsmLayout,
          bool kIndexFp4,
          bool kKvFp4,
          bool kNvfp4,
          bool kQFp8,
          bool kPartials,
          int kRows>
__global__ void fusedQKNormIdxrQKNormKernel(
    scalar_t* __restrict__ qkv,
    scalar_t* __restrict__ q_out,
    scalar_t* __restrict__ index_q_out,
    opus::fp8_t* __restrict__ index_q_fp8_out,
    const scalar_t* __restrict__ q_norm_w,
    const scalar_t* __restrict__ k_norm_w,
    const scalar_t* __restrict__ iq_norm_w,
    const scalar_t* __restrict__ ik_norm_w,
    const scalar_t* __restrict__ cos_sin_cache,
    const int64_t* __restrict__ positions,
    const int64_t* __restrict__ slot_mapping,
    const int64_t* __restrict__ index_slot_mapping,
    cache_t* __restrict__ kv_cache_k,
    cache_t* __restrict__ kv_cache_v,
    index_cache_t* __restrict__ index_cache,
    // Per-token dynamic-quant OUTPUT dequant scales (fp8 per-token path only).
    // Null for the SGLang unit-scale fp8 contract. Layout mirrors
    // reshape_and_cache_with_pertoken_quant:
    //   asm_layout : [num_blocks, num_kv_heads, block_size]
    //     idx = block_idx*(nkv*block_size) + head*block_size + block_offset
    //   page-128   : [num_kv_heads, max_kv_tokens]
    //     idx = head*max_kv_tokens + slot
    float* __restrict__ k_scale,
    float* __restrict__ v_scale,
    float eps,
    int rotary_dim,
    int num_tokens,
    int nq,
    int nkv,
    int niq,
    int block_size,
    int x,
    int max_kv_tokens,
    // page-128 (asm_layout=false) strides for the separate K/V caches
    // (each [num_blocks, block_size, num_kv_heads, head_dim]); unused when kAsmLayout.
    int64_t k_s_block,
    int64_t k_s_token,
    int64_t k_s_head,
    int64_t v_s_block,
    int64_t v_s_token,
    int64_t v_s_head,
    // Main Q as unit-scale e4m3 (kQFp8 only; replaces the q_out row).
    opus::fp8_t* __restrict__ q_fp8_out,
    // Unreduced split-K partials of qkv (kPartials only).
    QkvPartials partials)
{
    // NVFP4 is K, V and index-K pages together: never a fallback to the MXFP4 stores.
    static_assert(!kNvfp4 || (kKvFp4 && kIndexFp4), "kNvfp4 requires kKvFp4 and kIndexFp4");
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1250__)
    static_assert(kRows == 1 && !kPartials,
                  "several rows per warp and the partials load are compiled for gfx1250 only");
#endif
    constexpr int warps_per_block = kBlockSize / kNumLanes;
    const int lane_id = threadIdx.x % 32;
    const int global_warp_idx = blockIdx.x * warps_per_block + warpUniform(threadIdx.x / 32);

    const int v_slots = kInsertKV ? nkv : 0;
    const int idx_slots = kProcessIndex ? niq + 1 : 0;
    const int slots_per_token = nq + nkv + v_slots + idx_slots;

    // A warp takes kRows consecutive (token, slot) rows; a row past the last token
    // (last warp only) loads from the last token and is not processed.
    static_assert(kRows == 1 || !kPartials, "one row per warp with partials");
    int row_token[kRows];
    int row_slot[kRows];
    row_token[0] = global_warp_idx * kRows / slots_per_token;
    row_slot[0] = global_warp_idx * kRows % slots_per_token;
    if(row_token[0] >= num_tokens)
    {
        return;
    }
#pragma unroll
    for(int r = 1; r < kRows; ++r)
    {
        const bool wrap = row_slot[r - 1] + 1 == slots_per_token;
        row_token[r] = row_token[r - 1] + (wrap ? 1 : 0);
        row_slot[r] = wrap ? 0 : row_slot[r - 1] + 1;
    }

    const int k_begin = nq;
    const int v_begin = nq + nkv;
    const int iq_begin = nq + nkv + v_slots;
    const int ik_slot = iq_begin + niq;
    const int dim_base = lane_id * kElemsPerLane;
    const int qkv_row = (nq + 2 * nkv + (kIsSparse ? (niq + 1) : 0)) * kHeadDim;

    // Slots are [Q | K | V (insert) | IQ | IK (index)]; qkv rows hold [Q | K | V | IQ | IK]
    // heads, so a slot's head column is the slot itself, past the V heads it lacks.
    // No helper here selects between captured pointers: the compiler can turn that
    // into a load through a select of the captures' addresses, which keeps them in
    // scratch (gfx942, where the warp index is not wave-uniform).
    const auto is_ik_slot = [=](int slot) { return kProcessIndex && slot == ik_slot; };
    const auto row_of = [=](int token, int slot) {
        const int column = (kProcessIndex && slot >= iq_begin) ? slot + nkv - v_slots : slot;
        return qkv + static_cast<int64_t>(token) * qkv_row + column * kHeadDim;
    };

    // Every load of every row goes out before anything waits: the row, its norm weight
    // and its position, then the lane's cos/sin, then the cache slot. Scalar loads
    // return out of order, so the wait for the positions (which the cos/sin addresses
    // need) waits for every scalar load issued before it: the cache slot goes out after
    // the cos/sin loads, except with partials, whose split loads wait for kernel
    // arguments anyway. A V slot loads the norm inputs too, from valid addresses, and
    // uses none; a Q or IQ slot loads a cache slot (insert mode always has both
    // mappings) and stores nothing with it.
    int load_token[kRows];
    int64_t row_pos[kRows];
    scalar_t row_raw[kRows][kElemsPerLane];
    NormRopeInputs row_nri[kRows];
    int64_t row_cache_slot[kRows];
    const auto cache_slot_of = [](const int64_t* kv_map,
                                  const int64_t* index_map,
                                  bool ik,
                                  int token) -> int64_t {
        if constexpr(kInsertKV)
        {
            return (ik ? index_map : kv_map)[token];
        }
        return -1;
    };
#pragma unroll
    for(int r = 0; r < kRows; ++r)
    {
        load_token[r]  = r == 0 ? row_token[0] : min(row_token[r], num_tokens - 1);
        const int slot = row_slot[r];
        if constexpr(!kPartials)
        {
            unpackRawElems(*reinterpret_cast<const uint2*>(row_of(load_token[r], slot) + dim_base),
                           row_raw[r]);
        }
        // A V slot (v_begin <= slot < iq_begin) has no norm weight: it loads q's.
        const scalar_t* const norm_w = slot < k_begin    ? q_norm_w
                                       : slot < v_begin  ? k_norm_w
                                       : slot < iq_begin ? q_norm_w
                                       : slot == ik_slot ? ik_norm_w
                                                         : iq_norm_w;
        row_nri[r].w = *reinterpret_cast<const uint2*>(norm_w + dim_base);
        row_pos[r] = positions[load_token[r]];
        if constexpr(kPartials)
        {
            row_cache_slot[r] =
                cache_slot_of(slot_mapping, index_slot_mapping, is_ik_slot(slot), load_token[r]);
        }
    }
    // Left alone, the scheduler puts the loads above after the wait for the positions,
    // which only the cos/sin loads below need.
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for(int r = 0; r < kRows; ++r)
    {
        loadCosSin(row_nri[r], dim_base, rotary_dim, cos_sin_cache + row_pos[r] * rotary_dim);
    }
    if constexpr(kPartials)
    {
        // Last: the split loads branch on the split count, and their sum waits for them.
        scalar_t* row_ptr = row_of(row_token[0], row_slot[0]);
        reduceRawElems(row_ptr + dim_base, partials, (row_ptr - qkv) + dim_base, row_raw[0]);
    }
    else
    {
#pragma unroll
        for(int r = 0; r < kRows; ++r)
        {
            row_cache_slot[r] = cache_slot_of(
                slot_mapping, index_slot_mapping, is_ik_slot(row_slot[r]), load_token[r]);
        }
    }

    const auto process_row = [&](const int token_idx,
                                 const int slot,
                                 scalar_t (&raw)[kElemsPerLane],
                                 const NormRopeInputs& nri,
                                 const int64_t mapped_slot) {
        const bool is_q = slot < k_begin;
        const bool is_k = slot >= k_begin && slot < v_begin;
        const bool is_v = kInsertKV && slot >= v_begin && slot < v_begin + nkv;
        const bool is_iq = kProcessIndex && slot >= iq_begin && slot < ik_slot;
        const bool is_ik = is_ik_slot(slot);
        const bool do_rope = !is_v;
        const int head = is_k ? slot - k_begin : is_v ? slot - v_begin : 0;
        scalar_t* store_ptr = row_of(token_idx, slot);
        if(is_q && q_out != nullptr)
        {
            store_ptr = q_out + static_cast<int64_t>(token_idx) * nq * kHeadDim + slot * kHeadDim;
        }

        if constexpr(kInsertKV)
        {
            if(is_v)
            {
                if constexpr(kKvFp4)
                {
                    if constexpr(kNvfp4)
                    {
                        if(mapped_slot >= 0)
                        {
                            // NVFP4 main V: the raw row, its nibbles of the octet's words by atomic
                            // and/or (the other tokens of the octet, in any warp, share the words).
                            // Caller contract: no two tokens of one step write the same slot.
                            uint8_t* page = reinterpret_cast<uint8_t*>(kv_cache_v) +
                                            (mapped_slot / kNvTokens) * v_s_block;
                            storeNvfp4VToken<scalar_t>(page,
                                                       nkv,
                                                       head,
                                                       static_cast<int>(mapped_slot % kNvTokens),
                                                       *v_scale,
                                                       row_of(token_idx, slot),
                                                       lane_id);
                        }
                    }
                    else if(mapped_slot >= 0)
                    {
                        // MXFP4 main V: the raw row, one e8m0 per 32 elements (8 lanes).
                        float velems[kElemsPerLane];
                        loadElems(raw, velems);
                        storeMxfp4Row<scalar_t>(reinterpret_cast<uint8_t*>(kv_cache_v),
                                                mapped_slot,
                                                v_s_block,
                                                static_cast<int64_t>(head) * v_s_head,
                                                velems,
                                                lane_id);
                    }
                }
                else if(mapped_slot >= 0)
                {
                    // Per-token dynamic quant: amax over the lane's raw V elems, then a
                    // full-warp max -> the (token, head) head-dim amax. scale = amax/FP8_MAX.
                    float v_scale_val = 1.0f;
                    if constexpr(kv_dt != vllm::Fp8KVCacheDataType::kAuto &&
                                 kPerTokenKVScale)
                    {
                        float local_amax = 0.0f;
#pragma unroll
                        for(int i = 0; i < kElemsPerLane; ++i)
                        {
                            local_amax = fmaxf(local_amax, fabsf(static_cast<float>(raw[i])));
                        }
                        const float amax = warpReduceMax(local_amax);
                        v_scale_val = (amax > 0.0f) ? amax / fp8Max<cache_t>() : 1.0f;
                        if(lane_id == 0)
                        {
                            int64_t b, t;
                            splitSlot(mapped_slot, block_size, b, t);
                            const int64_t scale_idx =
                                kAsmLayout
                                    ? (b * (static_cast<int64_t>(nkv) * block_size) +
                                       static_cast<int64_t>(head) * block_size + t)
                                    : (static_cast<int64_t>(head) * max_kv_tokens + mapped_slot);
                            v_scale[scale_idx] = v_scale_val;
                        }
                    }
                    else if constexpr(kStaticKVScale)
                    {
                        v_scale_val = *v_scale;
                    }
                    if constexpr(kAsmLayout)
                    {
                        // V SHUFFLE: [num_blocks, num_kv_heads, block_size/x, head_dim, x]
                        //   off(d) = b*(nkv*hd*bs) + h*(hd*bs) + (t/x)*(hd*x) + d*x + (t%x)
                        // The kElemsPerLane elements of a lane (consecutive d) are x apart.
                        int64_t b, t, t_group, t_lane;
                        splitSlot(mapped_slot, block_size, b, t);
                        splitSlot(t, x, t_group, t_lane); // t / x, t % x
                        const int64_t v_base =
                            b * (static_cast<int64_t>(nkv) * kHeadDim * block_size) +
                            static_cast<int64_t>(head) * kHeadDim * block_size +
                            t_group * (static_cast<int64_t>(kHeadDim) * x) +
                            static_cast<int64_t>(dim_base) * x + t_lane;
                        storeCacheElemsStrided<scalar_t, cache_t, kv_dt>(
                            raw, kv_cache_v + v_base, x, v_scale_val);
                    }
                    else
                    {
                        // page-128 V cache [num_blocks, block_size, num_kv_heads, head_dim]
                        int64_t b, t;
                        splitSlot(mapped_slot, block_size, b, t);
                        const int64_t off =
                            b * v_s_block + t * v_s_token + head * v_s_head;
                        copyOrQuantizeCacheElems<scalar_t, cache_t, kv_dt>(
                            raw, kv_cache_v + off + dim_base, v_scale_val);
                    }
                }
                return;
            }
        }

        // Every slot past here has a norm weight (the host requires each one).
        float elems[kElemsPerLane];
        loadElems(raw, elems);
        normAndRope<scalar_t>(elems, lane_id, eps, nri, do_rope, rotary_dim);

        if constexpr(kInsertKV)
        {
            if(is_q)
            {
                if constexpr(kQFp8)
                {
                    // Saturate (+-inf too); the compares keep NaN a NaN.
                    const float fp8_max = fp8Max<opus::fp8_t>();
#pragma unroll
                    for(int i = 0; i < kElemsPerLane; ++i)
                    {
                        elems[i] = elems[i] > fp8_max    ? fp8_max
                                   : elems[i] < -fp8_max ? -fp8_max
                                                         : elems[i];
                    }
                    storeCacheElems<scalar_t, opus::fp8_t, vllm::Fp8KVCacheDataType::kFp8E4M3>(
                        q_fp8_out + static_cast<int64_t>(token_idx) * nq * kHeadDim +
                            slot * kHeadDim + dim_base,
                        elems,
                        1.0f);
                }
                else
                {
                    storeElems(store_ptr + dim_base, elems);
                }
            }
            if constexpr(kProcessIndex)
            {
                if(is_iq)
                {
                    storeIndexQElems(index_q_out,
                                     index_q_fp8_out,
                                     store_ptr + dim_base,
                                     elems,
                                     token_idx,
                                     niq,
                                     slot - iq_begin,
                                     dim_base);
                }
            }

            if(mapped_slot >= 0)
            {
                if constexpr(kProcessIndex)
                {
                    if(is_ik)
                    {
                        if constexpr(kNvfp4)
                        {
                            storeNvfp4IndexRow<scalar_t>(reinterpret_cast<uint8_t*>(index_cache) +
                                                             (mapped_slot / kNvTokens) * kNvIndexPage,
                                                         static_cast<int>(mapped_slot % kNvTokens),
                                                         elems,
                                                         lane_id);
                        }
                        else if constexpr(kIndexFp4)
                        {
                            storeMxfp4Row<scalar_t>(reinterpret_cast<uint8_t*>(index_cache),
                                                    mapped_slot,
                                                    kFp4PageBytes,
                                                    0,
                                                    elems,
                                                    lane_id);
                        }
                        else
                        {
                            storeCacheElems<scalar_t, index_cache_t, index_dt>(
                                index_cache + mapped_slot * kHeadDim + dim_base, elems, 1.0f);
                        }
                    }
                }
                if constexpr(kKvFp4)
                {
                    if(is_k)
                    {
                        if constexpr(kNvfp4)
                        {
                            storeNvfp4KRow<scalar_t>(reinterpret_cast<uint8_t*>(kv_cache_k) +
                                                         (mapped_slot / kNvTokens) * k_s_block,
                                                     nkv,
                                                     head,
                                                     static_cast<int>(mapped_slot % kNvTokens),
                                                     *k_scale,
                                                     elems,
                                                     lane_id);
                        }
                        else
                        {
                            storeMxfp4Row<scalar_t>(reinterpret_cast<uint8_t*>(kv_cache_k),
                                                    mapped_slot,
                                                    k_s_block,
                                                    static_cast<int64_t>(head) * k_s_head,
                                                    elems,
                                                    lane_id);
                        }
                    }
                }
                else if(is_k)
                {
                    // Per-token dynamic quant: amax over the lane's post-norm/rope K elems,
                    // then a full-warp max -> the (token, head) head-dim amax.
                    float k_scale_val = 1.0f;
                    if constexpr(kv_dt != vllm::Fp8KVCacheDataType::kAuto &&
                                 kPerTokenKVScale)
                    {
                        float local_amax = 0.0f;
#pragma unroll
                        for(int i = 0; i < kElemsPerLane; ++i)
                        {
                            local_amax = fmaxf(local_amax, fabsf(elems[i]));
                        }
                        const float amax = warpReduceMax(local_amax);
                        k_scale_val = (amax > 0.0f) ? amax / fp8Max<cache_t>() : 1.0f;
                        if(lane_id == 0)
                        {
                            int64_t b, t;
                            splitSlot(mapped_slot, block_size, b, t);
                            const int64_t scale_idx =
                                kAsmLayout
                                    ? (b * (static_cast<int64_t>(nkv) * block_size) +
                                       static_cast<int64_t>(head) * block_size + t)
                                    : (static_cast<int64_t>(head) * max_kv_tokens + mapped_slot);
                            k_scale[scale_idx] = k_scale_val;
                        }
                    }
                    else if constexpr(kStaticKVScale)
                    {
                        k_scale_val = *k_scale;
                    }
                    if constexpr(kAsmLayout)
                    {
                        // K SHUFFLE: [num_blocks, num_kv_heads, head_dim/x, block_size, x]
                        //   off(d) = b*(nkv*(hd/x)*bs*x) + h*((hd/x)*bs*x)
                        //            + (d/x)*(bs*x) + t*x + (d%x)
                        // kElemsPerLane <= x, so a lane's elements share d/x and (d%x) runs
                        // contiguously -> a contiguous run of kElemsPerLane in the cache.
                        int64_t b, t;
                        splitSlot(mapped_slot, block_size, b, t);
                        const int d0 = dim_base;  // = lane_id * kElemsPerLane
                        const int64_t k_off =
                            b * (static_cast<int64_t>(nkv) * (kHeadDim / x) * block_size * x) +
                            static_cast<int64_t>(head) * ((kHeadDim / x) * block_size * x) +
                            static_cast<int64_t>(d0 / x) * (static_cast<int64_t>(block_size) * x) +
                            t * x + (d0 % x);
                        storeCacheElems<scalar_t, cache_t, kv_dt>(
                            kv_cache_k + k_off, elems, k_scale_val);
                    }
                    else
                    {
                        // page-128 K cache [num_blocks, block_size, num_kv_heads, head_dim]
                        int64_t b, t;
                        splitSlot(mapped_slot, block_size, b, t);
                        const int64_t off = b * k_s_block + t * k_s_token + head * k_s_head;
                        storeCacheElems<scalar_t, cache_t, kv_dt>(
                            kv_cache_k + off + dim_base, elems, k_scale_val);
                    }
                }
            }
        }
        else
        {
            // No insert, one row per warp: two 32-bit row stores on gfx1250 (see storeElems).
            if constexpr(kProcessIndex)
            {
                if(is_iq)
                {
                    storeIndexQElems<scalar_t, kRows == 1>(index_q_out,
                                                           index_q_fp8_out,
                                                           store_ptr + dim_base,
                                                           elems,
                                                           token_idx,
                                                           niq,
                                                           slot - iq_begin,
                                                           dim_base);
                    return;
                }
            }
            storeElems<scalar_t, kRows == 1>(store_ptr + dim_base, elems);
        }
    };

    // One copy of the row code: process row 0, then shift the next row down.
    for(int r = 0;;)
    {
        process_row(row_token[0], row_slot[0], row_raw[0], row_nri[0], row_cache_slot[0]);
        if constexpr(kRows > 1)
        {
            if(++r < kRows && row_token[1] < num_tokens)
            {
#pragma unroll
                for(int i = 0; i + 1 < kRows; ++i)
                {
                    row_token[i]      = row_token[i + 1];
                    row_slot[i]       = row_slot[i + 1];
                    row_cache_slot[i] = row_cache_slot[i + 1];
                    row_nri[i]        = row_nri[i + 1];
#pragma unroll
                    for(int e = 0; e < kElemsPerLane; ++e)
                    {
                        row_raw[i][e] = row_raw[i + 1][e];
                    }
                }
                continue;
            }
        }
        break;
    }
}

template <typename scalar_t,
          typename cache_t,
          typename index_cache_t,
          vllm::Fp8KVCacheDataType kv_dt,
          vllm::Fp8KVCacheDataType index_dt,
          bool kIndexFp4 = false,
          bool kKvFp4 = false,
          bool kNvfp4 = false>
void launchFusedQKNormIdxrQKNorm(
    scalar_t* qkv,
    scalar_t* q_out,
    scalar_t* index_q_out,
    opus::fp8_t* index_q_fp8_out,
    const scalar_t* q_norm_w,
    const scalar_t* k_norm_w,
    const scalar_t* iq_norm_w,
    const scalar_t* ik_norm_w,
    const scalar_t* cos_sin_cache,
    const int64_t* positions,
    const int64_t* slot_mapping,
    const int64_t* index_slot_mapping,
    cache_t* kv_cache_k,
    cache_t* kv_cache_v,
    index_cache_t* index_cache,
    float* k_scale,
    float* v_scale,
    float eps,
    int rotary_dim,
    int num_tokens,
    int nq,
    int nkv,
    int niq,
    int block_size,
    int x,
    int max_kv_tokens,
    int64_t k_s_block,
    int64_t k_s_token,
    int64_t k_s_head,
    int64_t v_s_block,
    int64_t v_s_token,
    int64_t v_s_head,
    bool has_index,
    bool process_index,
    bool insert_kv,
    bool per_token_kv_scale,
    bool static_kv_scale,
    bool asm_layout,
    hipStream_t stream,
    opus::fp8_t* q_fp8_out = nullptr,
    QkvPartials partials   = {})
{
    const int v_slots = insert_kv ? nkv : 0;
    const int idx_slots = process_index ? niq + 1 : 0;
    const int slots_per_token = nq + nkv + v_slots + idx_slots;
    constexpr int kWarpsPerBlock = kBlockSize / 32;
    const int64_t total_rows = static_cast<int64_t>(num_tokens) * slots_per_token;
    // Several rows per warp, and the split-K partials load, on gfx1250 only. The fp4 caches,
    // which run on gfx950 only (the Python op refuses them elsewhere), have no multi-row
    // instantiation, nor does the split-K partials load. Only the host and gfx1250 passes
    // instantiate these kernels: the other device passes skip them, so other archs do not
    // build kernels that never run there (their host-side launch is not taken off gfx1250).
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1250__)
    constexpr int kMultiRowsCfg = 1;
    constexpr bool kPartialsCfg = false;
#else
    constexpr int kMultiRowsCfg = kKvFp4 || kIndexFp4 ? 1 : kMultiRows;
    constexpr bool kPartialsCfg = true;
#endif
    static const int64_t multi_row_min_rows = kMultiRowMinRowsPerCu * get_num_cu_func();
    const int rows_per_warp = kMultiRowsCfg > 1 && isGfx1250() && partials.num_splits == 0 &&
                                      total_rows >= multi_row_min_rows
                                  ? kMultiRowsCfg
                                  : 1;
    const int64_t total_warps = (total_rows + rows_per_warp - 1) / rows_per_warp;
    const int grid = static_cast<int>((total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock);
    if(grid == 0)
    {
        return;
    }

#define LAUNCH_Q(IS_SPARSE, PROCESS_INDEX, INSERT, ASM, PER_TOKEN, STATIC_SCALE, QFP8, PARTIALS, ROWS) \
    fusedQKNormIdxrQKNormKernel<scalar_t,                                                    \
                                cache_t,                                                     \
                                index_cache_t,                                               \
                                kv_dt,                                                       \
                                index_dt,                                                    \
                                PER_TOKEN,                                                   \
                                STATIC_SCALE,                                                \
                                IS_SPARSE,                                                   \
                                PROCESS_INDEX,                                               \
                                INSERT,                                                      \
                                ASM,                                                         \
                                kIndexFp4,                                                   \
                                kKvFp4,                                                      \
                                kNvfp4,                                                      \
                                QFP8,                                                        \
                                PARTIALS,                                                    \
                                ROWS>                                                        \
        <<<grid, kBlockSize, 0, stream>>>(qkv,                                               \
                                          q_out,                                             \
                                          index_q_out,                                       \
                                          index_q_fp8_out,                                   \
                                          q_norm_w,                                          \
                                          k_norm_w,                                          \
                                          iq_norm_w,                                         \
                                          ik_norm_w,                                         \
                                          cos_sin_cache,                                     \
                                          positions,                                         \
                                          slot_mapping,                                      \
                                          index_slot_mapping,                                \
                                          kv_cache_k,                                        \
                                          kv_cache_v,                                        \
                                          index_cache,                                       \
                                          k_scale,                                           \
                                          v_scale,                                           \
                                          eps,                                               \
                                          rotary_dim,                                        \
                                          num_tokens,                                        \
                                          nq,                                                \
                                          nkv,                                               \
                                          niq,                                               \
                                          block_size,                                        \
                                          x,                                                 \
                                          max_kv_tokens,                                     \
                                          k_s_block,                                         \
                                          k_s_token,                                         \
                                          k_s_head,                                          \
                                          v_s_block,                                         \
                                          v_s_token,                                         \
                                          v_s_head,                                          \
                                          q_fp8_out,                                         \
                                          partials)

#define LAUNCH_R(IS_SPARSE, PROCESS_INDEX, INSERT, ASM, PER_TOKEN, STATIC_SCALE, QFP8)      \
    do                                                                                       \
    {                                                                                        \
        if(rows_per_warp > 1)                                                                \
        {                                                                                    \
            LAUNCH_Q(IS_SPARSE, PROCESS_INDEX, INSERT, ASM, PER_TOKEN, STATIC_SCALE, QFP8, false, kMultiRowsCfg); \
        }                                                                                    \
        else                                                                                 \
        {                                                                                    \
            LAUNCH_Q(IS_SPARSE, PROCESS_INDEX, INSERT, ASM, PER_TOKEN, STATIC_SCALE, QFP8, false, 1); \
        }                                                                                    \
    } while(0)

// The e4m3 q_out store is compiled in only where it can run (insert mode), so the other
// instantiations are unchanged. So is the split-K partials load: gfx1250, bf16 insert modes
// that cover every qkv column, one row per warp (the host reduces the partials of any other
// call first).
#define LAUNCH(IS_SPARSE, PROCESS_INDEX, INSERT, ASM, PER_TOKEN, STATIC_SCALE)               \
    do                                                                                       \
    {                                                                                        \
        if constexpr(INSERT)                                                                 \
        {                                                                                    \
            if(q_fp8_out != nullptr)                                                         \
            {                                                                                \
                LAUNCH_R(IS_SPARSE, PROCESS_INDEX, INSERT, ASM, PER_TOKEN, STATIC_SCALE, true); \
            }                                                                                \
            else if(partials.num_splits > 0)                                                 \
            {                                                                                \
                if constexpr(kPartialsCfg && std::is_same_v<scalar_t, hip_bfloat16> &&       \
                             (!(IS_SPARSE) || (PROCESS_INDEX)) && !kKvFp4 && !kIndexFp4)     \
                {                                                                            \
                    LAUNCH_Q(IS_SPARSE, PROCESS_INDEX, INSERT, ASM, PER_TOKEN, STATIC_SCALE, false, true, 1); \
                }                                                                            \
            }                                                                                \
            else                                                                             \
            {                                                                                \
                LAUNCH_R(IS_SPARSE, PROCESS_INDEX, INSERT, ASM, PER_TOKEN, STATIC_SCALE, false); \
            }                                                                                \
        }                                                                                    \
        else                                                                                 \
        {                                                                                    \
            LAUNCH_R(IS_SPARSE, PROCESS_INDEX, INSERT, ASM, PER_TOKEN, STATIC_SCALE, false); \
        }                                                                                    \
    } while(0)

#define LAUNCH_SCALE(IS_SPARSE, PROCESS_INDEX, INSERT, ASM)     \
    do                                                          \
    {                                                           \
        if(per_token_kv_scale)                                  \
        {                                                       \
            LAUNCH(IS_SPARSE, PROCESS_INDEX, INSERT, ASM, true, false); \
        }                                                       \
        else if(static_kv_scale)                                 \
        {                                                       \
            LAUNCH(IS_SPARSE, PROCESS_INDEX, INSERT, ASM, false, true); \
        }                                                       \
        else                                                    \
        {                                                       \
            LAUNCH(IS_SPARSE, PROCESS_INDEX, INSERT, ASM, false, false); \
        }                                                       \
    } while(0)

#define LAUNCH_ASM(IS_SPARSE, PROCESS_INDEX, INSERT)             \
    do                                                          \
    {                                                           \
        if(asm_layout)                                          \
        {                                                       \
            LAUNCH_SCALE(IS_SPARSE, PROCESS_INDEX, INSERT, true); \
        }                                                       \
        else                                                    \
        {                                                       \
            LAUNCH_SCALE(IS_SPARSE, PROCESS_INDEX, INSERT, false); \
        }                                                       \
    } while(0)

    if constexpr(kKvFp4)
    {
        // FP4 (MXFP4/NVFP4) main KV: sparse insert only, pages are not shuffled and carry no
        // fp32 per-token scales.
        if(process_index)
        {
            LAUNCH(true, true, true, false, false, false);
        }
        else
        {
            LAUNCH(true, false, true, false, false, false);
        }
    }
    else if(has_index)
    {
        if(insert_kv)
        {
            if(process_index)
            {
                LAUNCH_ASM(true, true, true);
            }
            else
            {
                LAUNCH_ASM(true, false, true);
            }
        }
        else
        {
            if(process_index)
            {
                LAUNCH_ASM(true, true, false);
            }
            else
            {
                LAUNCH_ASM(true, false, false);
            }
        }
    }
    else
    {
        if(insert_kv)
        {
            LAUNCH_ASM(false, false, true);
        }
        else
        {
            LAUNCH_ASM(false, false, false);
        }
    }
#undef LAUNCH_ASM
#undef LAUNCH_SCALE
#undef LAUNCH
#undef LAUNCH_R
#undef LAUNCH_Q
}

} // namespace fused_qknorm_idxrqknorm_ops

static void fused_qknorm_idxrqknorm_impl(
    aiter_tensor_t& qkv,
    const aiter_tensor_t& q_norm_weight,
    const aiter_tensor_t& k_norm_weight,
    const aiter_tensor_t& cos_sin_cache,
    const aiter_tensor_t& positions,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t rotary_dim,
    double eps,
    std::optional<aiter_tensor_t> index_q_norm_weight,
    std::optional<aiter_tensor_t> index_k_norm_weight,
    int64_t num_index_heads,
    std::optional<aiter_tensor_t> slot_mapping,
    std::optional<aiter_tensor_t> kv_cache_k,
    std::optional<aiter_tensor_t> kv_cache_v,
    std::optional<aiter_tensor_t> index_cache,
    int64_t block_size,
    std::optional<aiter_tensor_t> q_out,
    std::optional<aiter_tensor_t> index_q_out,
    std::optional<aiter_tensor_t> index_slot_mapping,
    const std::string& kv_cache_dtype,
    const std::string& index_cache_dtype,
    std::optional<aiter_tensor_t> k_scale,
    std::optional<aiter_tensor_t> v_scale,
    bool asm_layout,
    bool skip_index_branch,
    std::optional<aiter_tensor_t> qkv_partials,
    bool check_only)
{
    using namespace fused_qknorm_idxrqknorm_ops;

    AITER_CHECK(qkv.is_gpu() && qkv.is_contiguous(), "qkv must be contiguous CUDA");
    AITER_CHECK(qkv.dtype() == AITER_DTYPE_bf16 || qkv.dtype() == AITER_DTYPE_fp16,
                "qkv must be bf16 or fp16");
    AITER_CHECK(positions.is_gpu() && positions.is_contiguous() &&
                    positions.dtype() == AITER_DTYPE_i64,
                "positions must be contiguous int64 CUDA");
    AITER_CHECK(cos_sin_cache.is_gpu() && cos_sin_cache.is_contiguous(),
                "cos_sin_cache must be contiguous CUDA");
    AITER_CHECK(cos_sin_cache.dtype() == qkv.dtype(),
                "cos_sin_cache dtype must match qkv");
    AITER_CHECK(cos_sin_cache.dim() == 2 && cos_sin_cache.size(1) == rotary_dim,
                "cos_sin_cache shape must be [max_pos, rotary_dim]");
    AITER_CHECK(q_norm_weight.is_gpu() && k_norm_weight.is_gpu() &&
                    q_norm_weight.is_contiguous() && k_norm_weight.is_contiguous(),
                "q/k norm weights must be contiguous CUDA tensors");
    AITER_CHECK(q_norm_weight.dtype() == qkv.dtype() &&
                    k_norm_weight.dtype() == qkv.dtype(),
                "q/k norm weight dtype must match qkv");
    AITER_CHECK(q_norm_weight.numel() == kHeadDim && k_norm_weight.numel() == kHeadDim,
                "q/k norm weight must have 128 elements");
    AITER_CHECK(rotary_dim > 0 && rotary_dim % 8 == 0 && rotary_dim <= kHeadDim,
                "rotary_dim must be a positive multiple of 8 and <= 128");

    const int num_tokens = static_cast<int>(qkv.size(0));
    const int nq = static_cast<int>(num_heads);
    const int nkv = static_cast<int>(num_kv_heads);
    const int niq = static_cast<int>(num_index_heads);
    AITER_CHECK(nq > 0 && nkv > 0 && niq >= 0,
                "num_heads/num_kv_heads must be positive, num_index_heads non-negative");
    const bool has_index = niq > 0;
    AITER_CHECK(!skip_index_branch || has_index,
                "skip_index_branch requires num_index_heads > 0");
    const bool process_index = has_index && !skip_index_branch;
    // Both layouts use the separate kv_cache_k / kv_cache_v caches; asm_layout only
    // selects the cache element addressing (page-16 SHUFFLE vs plain page-128).
    AITER_CHECK(kv_cache_k.has_value() == kv_cache_v.has_value(),
                "kv_cache_k and kv_cache_v must be provided together");
    const bool insert_kv = kv_cache_k.has_value();
    const bool nvfp4_kv_cache    = kv_cache_dtype == "nvfp4";
    const bool nvfp4_index_cache = index_cache_dtype == "nvfp4";
    // NVFP4 is one mode: K/V pages and index pages together (an index-less layer has no index).
    const bool fp4_kv_cache = kv_cache_dtype == "mxfp4" || nvfp4_kv_cache;
    AITER_CHECK(!fp4_kv_cache || (insert_kv && has_index),
                "kv_cache_dtype='mxfp4'/'nvfp4' requires the sparse insert mode "
                "(kv caches + index slots in qkv)");
    AITER_CHECK(nvfp4_kv_cache == nvfp4_index_cache || (nvfp4_kv_cache && skip_index_branch),
                "kv_cache_dtype='nvfp4' and index_cache_dtype='nvfp4' go together "
                "(index_cache_dtype may be 'auto' only with skip_index_branch)");
    const bool fp8_kv_cache =
        insert_kv && kv_cache_dtype.rfind("fp8", 0) == 0;
    const bool fp8_kv_per_token =
        fp8_kv_cache &&
        (kv_cache_dtype == "fp8" || kv_cache_dtype == "fp8_e4m3");
    const bool fp8_kv_static =
        fp8_kv_cache && kv_cache_dtype == "fp8_e4m3_static";
    const bool fp8_index_cache =
        insert_kv && process_index && index_cache_dtype.rfind("fp8", 0) == 0;
    const bool fp4_index_cache = index_cache_dtype == "mxfp4" || nvfp4_index_cache;
    AITER_CHECK(!fp4_index_cache || (insert_kv && process_index),
                "index_cache_dtype='mxfp4'/'nvfp4' requires the sparse insert mode "
                "(kv caches + index branch)");
    const int expected_row = (nq + 2 * nkv + (has_index ? niq + 1 : 0)) * kHeadDim;
    AITER_CHECK(positions.dim() == 1 && positions.size(0) >= num_tokens,
                "positions must be 1D with at least num_tokens elements");
    AITER_CHECK(qkv.dim() == 2 && qkv.size(1) == expected_row,
                "qkv must be [num_tokens, (num_heads + 2*num_kv_heads"
                " + optional index heads) * 128]");
    if(process_index)
    {
        AITER_CHECK(index_q_norm_weight.has_value() && index_k_norm_weight.has_value(),
                    "index branch requires both index norm weights");
        AITER_CHECK(index_q_norm_weight->is_gpu() && index_k_norm_weight->is_gpu() &&
                        index_q_norm_weight->is_contiguous() &&
                        index_k_norm_weight->is_contiguous(),
                    "index norm weights must be contiguous CUDA tensors");
        AITER_CHECK(index_q_norm_weight->dtype() == qkv.dtype() &&
                        index_k_norm_weight->dtype() == qkv.dtype(),
                    "index norm weights dtype must match qkv");
        AITER_CHECK(index_q_norm_weight->numel() == kHeadDim &&
                        index_k_norm_weight->numel() == kHeadDim,
                    "index norm weights must have 128 elements");
    }

    // page-128 K/V cache strides ([num_blocks, block_size, num_kv_heads, head_dim]);
    // unused in asm_layout (kernel computes SHUFFLE offsets directly).
    int64_t k_s_block = 0;
    int64_t k_s_token = 0;
    int64_t k_s_head = 0;
    int64_t v_s_block = 0;
    int64_t v_s_token = 0;
    int64_t v_s_head = 0;
    int x = 0;  // SHUFFLE interleave factor = 16 / cache_itemsize (asm_layout only)
    int max_kv_tokens = 0;  // page-128 per-token scale stride (scales: [nkv, max_kv_tokens])
    if(insert_kv)
    {
        AITER_CHECK(block_size > 0, "block_size must be positive in insert mode");
        AITER_CHECK(slot_mapping.has_value() && slot_mapping->is_gpu() &&
                        slot_mapping->is_contiguous() &&
                        slot_mapping->dtype() == AITER_DTYPE_i64,
                    "insert mode requires contiguous int64 CUDA slot_mapping");
        AITER_CHECK(slot_mapping->dim() == 1 && slot_mapping->size(0) >= num_tokens,
                    "slot_mapping must be 1D with at least num_tokens elements");
        if(process_index && !index_slot_mapping.has_value())
        {
            index_slot_mapping = slot_mapping;
        }
        if(process_index)
        {
            AITER_CHECK(index_slot_mapping.has_value() && index_slot_mapping->is_gpu() &&
                            index_slot_mapping->is_contiguous() &&
                            index_slot_mapping->dtype() == AITER_DTYPE_i64,
                        "sparse insert mode requires contiguous int64 CUDA index_slot_mapping");
            AITER_CHECK(index_slot_mapping->dim() == 1 &&
                            index_slot_mapping->size(0) >= num_tokens,
                        "index_slot_mapping must be 1D with at least num_tokens elements");
            AITER_CHECK(index_cache.has_value() && index_cache->is_gpu() &&
                            index_cache->is_contiguous(),
                        "sparse insert mode requires contiguous CUDA index_cache");
            AITER_CHECK(index_cache_dtype == "auto" || index_cache_dtype.rfind("fp8", 0) == 0 ||
                            fp4_index_cache,
                        "sparse insert mode expects index_cache_dtype='auto', fp8, mxfp4 or nvfp4");
            if(nvfp4_index_cache)
            {
                AITER_CHECK(index_cache->dtype() == AITER_DTYPE_u8 &&
                                index_cache->numel() % kNvIndexPage == 0,
                            "nvfp4 sparse insert mode requires a uint8 index_cache of whole "
                            "9216 B pages");
            }
            else if(fp4_index_cache)
            {
                AITER_CHECK(index_cache->dtype() == AITER_DTYPE_u8 &&
                                index_cache->numel() % kFp4PageBytes == 0,
                            "mxfp4 sparse insert mode requires a uint8 index_cache of whole "
                            "8704 B pages");
            }
            else if(fp8_index_cache)
            {
                AITER_CHECK(index_cache->dtype() == AITER_DTYPE_fp8 ||
                                index_cache->dtype() == AITER_DTYPE_u8,
                            "fp8 sparse insert mode requires fp8 or uint8 index_cache");
            }
            else
            {
                AITER_CHECK(index_cache->dtype() == qkv.dtype(),
                            "non-fp8 sparse insert mode requires matching index_cache");
            }
        }

        // K/V cache dtype + scale checks (common to both layouts). kv_cache_k is the
        // dtype reference for both the SHUFFLE and page-128 caches.
        const aiter_tensor_t& kc = *kv_cache_k;
        AITER_CHECK(kc.is_gpu(), "kv cache must be CUDA");
        if(fp8_kv_cache)
        {
            AITER_CHECK(kv_cache_dtype == "fp8" || kv_cache_dtype == "fp8_e4m3" ||
                            kv_cache_dtype == "fp8_e4m3_unit" ||
                            kv_cache_dtype == "fp8_e4m3_static",
                        "fused_qknorm_idxrqknorm fp8 cache insert supports "
                        "fp8/fp8_e4m3 per-token, fp8_e4m3_unit, or "
                        "fp8_e4m3_static only");
            AITER_CHECK(kc.dtype() == AITER_DTYPE_fp8 || kc.dtype() == AITER_DTYPE_u8,
                        "fp8 kv_cache must use float8_e4m3 or uint8 storage");
            AITER_CHECK(k_scale.has_value() == v_scale.has_value(),
                        "k_scale and v_scale must be provided together");
            if(fp8_kv_per_token)
            {
                // k_scale/v_scale are per-token dynamic-quant OUTPUT tensors (fp32).
                // Their exact shape is validated in the layout branches below.
                AITER_CHECK(k_scale.has_value() && k_scale->is_gpu() && v_scale->is_gpu() &&
                                k_scale->is_contiguous() && v_scale->is_contiguous() &&
                                k_scale->dtype() == AITER_DTYPE_fp32 &&
                                v_scale->dtype() == AITER_DTYPE_fp32,
                            "fp8 per-token insert requires contiguous float32 CUDA "
                            "k_scale/v_scale output tensors");
            }
            else if(fp8_kv_static)
            {
                AITER_CHECK(k_scale.has_value() && v_scale.has_value() &&
                                k_scale->is_gpu() && v_scale->is_gpu() &&
                                k_scale->is_contiguous() && v_scale->is_contiguous() &&
                                k_scale->dtype() == AITER_DTYPE_fp32 &&
                                v_scale->dtype() == AITER_DTYPE_fp32 &&
                                k_scale->numel() == 1 && v_scale->numel() == 1,
                            "fp8_e4m3_static requires single-element contiguous "
                            "float32 CUDA k_scale/v_scale input tensors");
            }
            else
            {
                AITER_CHECK(!k_scale.has_value(),
                            "fp8_e4m3_unit does not accept k_scale/v_scale tensors");
            }
        }
        else
        {
            AITER_CHECK(kv_cache_dtype == "auto" || fp4_kv_cache,
                        "non-fp8 fused_qknorm_idxrqknorm expects kv_cache_dtype='auto', 'mxfp4' "
                        "or 'nvfp4'");
            AITER_CHECK(fp4_kv_cache ? kc.dtype() == AITER_DTYPE_u8 : kc.dtype() == qkv.dtype(),
                        "kv_cache dtype must match qkv for a non-fp8 cache (uint8 for "
                        "mxfp4/nvfp4)");
            AITER_CHECK(nvfp4_kv_cache || !fp4_kv_cache ||
                            (!k_scale.has_value() && !v_scale.has_value()),
                        "kv_cache_dtype='mxfp4' keeps its scales in the pages: no k_scale/v_scale");
            AITER_CHECK(!nvfp4_kv_cache ||
                            (k_scale.has_value() && v_scale.has_value() && k_scale->is_gpu() &&
                             v_scale->is_gpu() && k_scale->dtype() == AITER_DTYPE_fp32 &&
                             v_scale->dtype() == AITER_DTYPE_fp32 && k_scale->numel() == 1 &&
                             v_scale->numel() == 1),
                        "kv_cache_dtype='nvfp4' takes the fp32 global k_scale/v_scale, one "
                        "element each");
        }

        // Whether the index cache holds `tokens` rows (whole pages: 8704 B mxfp4, 9216 B nvfp4).
        const auto index_holds = [&](int64_t tokens) {
            return nvfp4_index_cache ? index_cache->numel() / kNvIndexPage * kNvTokens >= tokens
                   : fp4_index_cache
                       ? index_cache->numel() / kFp4PageBytes * kFp4PageTokens >= tokens
                       : index_cache->numel() >= tokens * kHeadDim;
        };
        if(nvfp4_kv_cache)
        {
            // [num_pages, 2 * num_kv_heads * 9216] uint8, K side then V side; kv_cache_k and
            // kv_cache_v are the same pages. The page stride (bytes) may be wider; 4 B aligned
            // for the 32-bit V words.
            AITER_CHECK(block_size == kNvTokens,
                        "kv_cache_dtype='nvfp4' pages hold 128 tokens: block_size must be 128");
            AITER_CHECK(
                kv_cache_k->dtype() == AITER_DTYPE_u8 && kv_cache_v->dtype() == AITER_DTYPE_u8 &&
                    kv_cache_k->data_ptr() == kv_cache_v->data_ptr() && kv_cache_k->dim() == 2 &&
                    kv_cache_k->size(1) == 2 * nkv * kNvHeadSide && kv_cache_k->stride(1) == 1 &&
                    kv_cache_k->stride(0) % 4 == 0 && kv_cache_k->data_ptr() != nullptr &&
                    reinterpret_cast<uintptr_t>(kv_cache_k->data_ptr()) % 4 == 0,
                "nvfp4 kv_cache_k and kv_cache_v must be the same uint8 "
                "[num_pages, 2 * num_kv_heads * 9216] pages (4 B aligned)");
            // A page stride below the page size (e.g. an expanded view, stride 0) would alias
            // distinct pages.
            AITER_CHECK(kv_cache_k->size(0) <= 1 || kv_cache_k->stride(0) >= kv_cache_k->size(1),
                        "nvfp4 kv_cache page stride must be at least the page size");
            k_s_block = kv_cache_k->stride(0);
            v_s_block = k_s_block;
            if(process_index)
            {
                AITER_CHECK(index_holds(kv_cache_k->size(0) * block_size),
                            "index_cache must contain at least num_blocks * block_size rows");
            }
        }
        else if(fp4_kv_cache)
        {
            // [num_pages, num_kv_heads, 8704] uint8 per side: one page of kFp4PageTokens
            // tokens per (page, head), [128 x 64 B e2m1][128 x 4 B e8m0]; the strides are
            // the pages' own (bytes), so K and V may be views into one single-region pool.
            AITER_CHECK(block_size == kFp4PageTokens,
                        "kv_cache_dtype='mxfp4' pages hold 128 tokens: block_size must be 128");
            AITER_CHECK(kv_cache_k->is_gpu() && kv_cache_v->is_gpu() &&
                            kv_cache_k->dtype() == AITER_DTYPE_u8 &&
                            kv_cache_v->dtype() == AITER_DTYPE_u8,
                        "mxfp4 kv_cache_k/kv_cache_v must be uint8 CUDA tensors");
            AITER_CHECK(kv_cache_k->dim() == 3 && kv_cache_k->size(1) == nkv &&
                            kv_cache_k->size(2) == kFp4PageBytes && kv_cache_k->stride(2) == 1 &&
                            kv_cache_v->dim() == 3 && kv_cache_v->size(0) == kv_cache_k->size(0) &&
                            kv_cache_v->size(1) == nkv &&
                            kv_cache_v->size(2) == kFp4PageBytes && kv_cache_v->stride(2) == 1,
                        "mxfp4 kv_cache_k/kv_cache_v must be [num_pages, num_kv_heads, 8704] "
                        "with the same num_pages and a contiguous last dim");
            k_s_block = kv_cache_k->stride(0);
            k_s_head  = kv_cache_k->stride(1);
            v_s_block = kv_cache_v->stride(0);
            v_s_head  = kv_cache_v->stride(1);
            if(process_index)
            {
                AITER_CHECK(index_holds(kv_cache_k->size(0) * block_size),
                            "index_cache must contain at least num_blocks * block_size rows");
            }
        }
        else if(asm_layout)
        {
            // SHUFFLE (asm_layout) caches, matching reshape_and_cache(asm_layout=true):
            //   K [num_blocks, num_kv_heads, head_dim/x, block_size, x]
            //   V [num_blocks, num_kv_heads, block_size/x, head_dim, x]
            // with x = 16 / cache_itemsize and block_size the (physical) ASM page.
            const int itemsize = static_cast<int>(kv_cache_k->element_size());
            AITER_CHECK(itemsize > 0 && (16 % itemsize) == 0,
                        "asm_layout kv cache element size must divide 16");
            x = 16 / itemsize;
            AITER_CHECK(kHeadDim % x == 0 && block_size % x == 0,
                        "asm_layout requires x | head_dim and x | block_size");
            AITER_CHECK(kv_cache_k->is_gpu() && kv_cache_k->is_contiguous() &&
                            kv_cache_v->is_gpu() && kv_cache_v->is_contiguous(),
                        "asm_layout kv_cache_k/kv_cache_v must be contiguous CUDA");
            AITER_CHECK(kv_cache_k->dtype() == kv_cache_v->dtype(),
                        "asm_layout kv_cache_k/kv_cache_v dtype must match");
            AITER_CHECK(kv_cache_k->dim() == 5 && kv_cache_k->size(1) == nkv &&
                            kv_cache_k->size(2) == kHeadDim / x &&
                            kv_cache_k->size(3) == block_size && kv_cache_k->size(4) == x,
                        "kv_cache_k must be [num_blocks, num_kv_heads, head_dim/x, "
                        "block_size, x]");
            // The page spans are equal for separately allocated K/V caches. Packed
            // LBHNC storage instead exposes K from the beginning of each packed
            // block and V from an offset base pointer, so their contiguous tensor
            // spans intentionally have different size(0). Rebased slot mappings
            // still address the same logical page domain in both views.
            AITER_CHECK(kv_cache_v->dim() == 5 && kv_cache_v->size(1) == nkv &&
                            kv_cache_v->size(2) == block_size / x &&
                            kv_cache_v->size(3) == kHeadDim && kv_cache_v->size(4) == x,
                        "kv_cache_v must be [page_span, num_kv_heads, block_size/x, "
                        "head_dim, x]");
            if(process_index)
            {
                // Main-cache size(0) is a physical storage span in packed LBHNC
                // mode and includes gaps belonging to the opposite K/V side. It
                // therefore cannot define the independently mapped index-cache
                // capacity: no capacity check is possible here. Caller contract (as for
                // slot_mapping): every non-negative index_slot_mapping entry must be below
                // the rows index_cache holds (numel / 128; numel / 8704 * 128 for mxfp4).
                AITER_CHECK(index_cache->numel() % kHeadDim == 0,
                            "index_cache must contain contiguous 128-element rows");
            }
            // strides unused in asm_layout (kernel computes SHUFFLE offsets directly).
            if(fp8_kv_per_token)
            {
                // Per-token dequant scales: [num_blocks, num_kv_heads, block_size]
                // (mirrors reshape_and_cache_with_pertoken_quant asm_layout).
                // For packed views, the shorter span is the common page-id
                // domain reachable from both offset base pointers.
                const int64_t scale_page_span =
                    kv_cache_k->size(0) < kv_cache_v->size(0)
                        ? kv_cache_k->size(0)
                        : kv_cache_v->size(0);
                AITER_CHECK(k_scale->numel() >= scale_page_span * nkv * block_size &&
                                v_scale->numel() >= scale_page_span * nkv * block_size,
                            "asm_layout fp8 per-token k_scale/v_scale must have at least "
                            "common_page_span * num_kv_heads * block_size elements");
            }
        }
        else
        {
            // page-128 caches as separate K and V tensors
            //   [num_blocks, block_size, num_kv_heads, head_dim] (head_dim contiguous).
            // These are typically the key/value slices of a fused
            // [num_blocks, 2, block_size, num_kv_heads, head_dim] cache.
            AITER_CHECK(kv_cache_k->is_gpu() && kv_cache_v->is_gpu(),
                        "kv_cache_k/kv_cache_v must be CUDA");
            AITER_CHECK(kv_cache_k->dtype() == kv_cache_v->dtype(),
                        "kv_cache_k/kv_cache_v dtype must match");
            AITER_CHECK(kv_cache_k->dim() == 4 && kv_cache_k->size(1) == block_size &&
                            kv_cache_k->size(2) == nkv && kv_cache_k->size(3) == kHeadDim &&
                            kv_cache_k->stride(3) == 1,
                        "page-128 kv_cache_k must be [num_blocks, block_size, "
                        "num_kv_heads, 128] with contiguous head_dim");
            AITER_CHECK(kv_cache_v->dim() == 4 && kv_cache_v->size(0) == kv_cache_k->size(0) &&
                            kv_cache_v->size(1) == block_size && kv_cache_v->size(2) == nkv &&
                            kv_cache_v->size(3) == kHeadDim && kv_cache_v->stride(3) == 1,
                        "page-128 kv_cache_v must be [num_blocks, block_size, "
                        "num_kv_heads, 128] with contiguous head_dim");
            if(process_index)
            {
                AITER_CHECK(index_holds(kv_cache_k->size(0) * block_size),
                            "index_cache must contain at least num_blocks * block_size * 128 elements");
            }
            k_s_block = kv_cache_k->stride(0);
            k_s_token = kv_cache_k->stride(1);
            k_s_head = kv_cache_k->stride(2);
            v_s_block = kv_cache_v->stride(0);
            v_s_token = kv_cache_v->stride(1);
            v_s_head = kv_cache_v->stride(2);
            if(fp8_kv_per_token)
            {
                // Per-token dequant scales: [num_kv_heads, max_kv_tokens] with
                // idx = head * max_kv_tokens + slot (mirrors
                // reshape_and_cache_with_pertoken_quant page-128). max_kv_tokens is the
                // scale tensor's last-dim size (caller-sized to cover all slots).
                AITER_CHECK(k_scale->dim() == 2 && v_scale->dim() == 2 &&
                                k_scale->size(0) == nkv && v_scale->size(0) == nkv &&
                                k_scale->size(1) == v_scale->size(1),
                            "page-128 fp8 per-token k_scale/v_scale must be "
                            "[num_kv_heads, max_kv_tokens]");
                max_kv_tokens = static_cast<int>(k_scale->size(1));
                const int64_t num_blocks = kv_cache_k->size(0);
                AITER_CHECK(max_kv_tokens >= num_blocks * block_size,
                            "page-128 fp8 per-token scale max_kv_tokens must be >= "
                            "num_blocks * block_size");
            }
        }
    }

    const bool q_is_fp8 = q_out.has_value() && (q_out->dtype() == AITER_DTYPE_fp8 ||
                                                q_out->dtype() == AITER_DTYPE_u8);
    if(q_out.has_value())
    {
        AITER_CHECK(!q_is_fp8 || insert_kv, "fp8 q_out requires the insert mode (kv caches)");
        AITER_CHECK(q_out->is_gpu() && q_out->is_contiguous() &&
                        (q_out->dtype() == qkv.dtype() || q_is_fp8),
                    "q_out must be contiguous CUDA and match qkv dtype or fp8 e4m3");
        AITER_CHECK(q_out->numel() == static_cast<int64_t>(num_tokens) * nq * kHeadDim,
                    "q_out must have num_tokens * num_heads * 128 elements");
    }
    if(index_q_out.has_value())
    {
        AITER_CHECK(has_index, "index_q_out requires num_index_heads > 0");
        const bool index_q_is_fp8 = index_q_out->dtype() == AITER_DTYPE_fp8 ||
                                    index_q_out->dtype() == AITER_DTYPE_u8;
        AITER_CHECK(index_q_out->is_gpu() && index_q_out->is_contiguous() &&
                        (index_q_out->dtype() == qkv.dtype() || index_q_is_fp8),
                    "index_q_out must be contiguous CUDA and match qkv dtype or fp8 e4m3");
        AITER_CHECK(index_q_out->numel() == static_cast<int64_t>(num_tokens) * niq * kHeadDim,
                    "index_q_out must have num_tokens * num_index_heads * 128 elements");
    }

    // The launches run on qkv's device and read or write every tensor of the call they use
    // through a raw pointer.
    const auto on_qkv_device = [&](const std::optional<aiter_tensor_t>& t) {
        return !t.has_value() || t->device_id == qkv.device_id;
    };
    const bool kv_scaled = fp8_kv_per_token || fp8_kv_static || nvfp4_kv_cache;
    AITER_CHECK(q_norm_weight.device_id == qkv.device_id &&
                    k_norm_weight.device_id == qkv.device_id &&
                    cos_sin_cache.device_id == qkv.device_id &&
                    positions.device_id == qkv.device_id && on_qkv_device(q_out) &&
                    on_qkv_device(qkv_partials) &&
                    (!process_index ||
                     (on_qkv_device(index_q_out) && on_qkv_device(index_q_norm_weight) &&
                      on_qkv_device(index_k_norm_weight))) &&
                    (!insert_kv ||
                     (on_qkv_device(slot_mapping) && on_qkv_device(kv_cache_k) &&
                      on_qkv_device(kv_cache_v) &&
                      (!kv_scaled || (on_qkv_device(k_scale) && on_qkv_device(v_scale))) &&
                      (!process_index || (on_qkv_device(index_slot_mapping) &&
                                          on_qkv_device(index_cache))))),
                "every tensor the call reads or writes must be on qkv's device");

    QkvPartials partials{};
    if(qkv_partials.has_value())
    {
        AITER_CHECK(qkv.dtype() == AITER_DTYPE_bf16, "qkv_partials requires a bf16 qkv");
        const aiter_tensor_t& qp = *qkv_partials;
        AITER_CHECK(qp.is_gpu() && qp.dtype() == AITER_DTYPE_fp32 && qp.dim() == 3,
                    "qkv_partials must be a float32 CUDA [num_splits, rows, row] tensor");
        const int64_t splits = qp.size(0);
        AITER_CHECK(splits == 2 || splits == 4 || splits == 8,
                    "qkv_partials supports 2, 4 or 8 splits");
        AITER_CHECK(qp.size(1) >= num_tokens && qp.size(2) == expected_row &&
                        qp.stride(2) == 1 && qp.stride(1) == expected_row &&
                        qp.stride(0) >= qp.size(1) * expected_row,
                    "qkv_partials must hold num_tokens dense qkv rows per split");
        AITER_CHECK(reinterpret_cast<uintptr_t>(qp.data_ptr()) % 16 == 0 &&
                        qp.stride(0) % 4 == 0,
                    "qkv_partials must be 16-byte aligned per split");
        partials = {reinterpret_cast<const float*>(qp.data_ptr()),
                    static_cast<int>(splits),
                    qp.stride(0)};
    }

    // check_only: every check above passed, the call would run.
    if(check_only)
    {
        return;
    }

    HipDeviceGuard device_guard(qkv.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    // The main kernel sums the partials as it reads qkv where it writes every qkv column
    // back: on gfx1250, the insert mode with every slot processed, no fp4 cache and no
    // e4m3 q. Any other call reduces them into qkv first.
    const bool fuse_partials = isGfx1250() && insert_kv && (!has_index || process_index) &&
                               !fp4_kv_cache && !fp4_index_cache && !q_is_fp8;
    if(partials.num_splits > 0 && !fuse_partials)
    {
        // expected_row is a multiple of 128: whole lanes of kElemsPerLane elements
        const int64_t num_elems = static_cast<int64_t>(num_tokens) * expected_row;
        constexpr int kReduceBlock = 256;
        const int blocks = static_cast<int>(
            (num_elems / kElemsPerLane + kReduceBlock - 1) / kReduceBlock);
        if(blocks > 0)
        {
            reduceQkvPartialsKernel<<<blocks, kReduceBlock, 0, stream>>>(
                reinterpret_cast<hip_bfloat16*>(qkv.data_ptr()), partials, num_elems);
        }
        partials = {};
    }

    AITER_DISPATCH_FLOATING16_TYPES_rmTorch(qkv.dtype(), "fused_qknorm_idxrqknorm", [&] {
        using T = scalar_t;
        const bool index_q_is_fp8 =
            index_q_out.has_value() && (index_q_out->dtype() == AITER_DTYPE_fp8 ||
                                        index_q_out->dtype() == AITER_DTYPE_u8);
        T* index_q_ptr = (!index_q_is_fp8 && index_q_out.has_value())
                             ? reinterpret_cast<T*>(index_q_out->data_ptr())
                             : nullptr;
        opus::fp8_t* index_q_fp8_ptr =
            index_q_is_fp8 ? reinterpret_cast<opus::fp8_t*>(index_q_out->data_ptr())
                           : nullptr;
        T* q_ptr = (!q_is_fp8 && q_out.has_value()) ? reinterpret_cast<T*>(q_out->data_ptr())
                                                    : nullptr;
        opus::fp8_t* q_fp8_ptr =
            q_is_fp8 ? reinterpret_cast<opus::fp8_t*>(q_out->data_ptr()) : nullptr;
        if(fp4_kv_cache || fp4_index_cache)
        {
            // FP4 (MXFP4/NVFP4) main K/V and/or index K (sparse insert only); the other cache
            // keeps its type.
            auto launch_fp4 = [&]<typename CacheT,
                                  typename IndexT,
                                  vllm::Fp8KVCacheDataType kv_dt,
                                  vllm::Fp8KVCacheDataType index_dt,
                                  bool kIndexFp4,
                                  bool kKvFp4,
                                  bool kNv>() {
                launchFusedQKNormIdxrQKNorm<T, CacheT, IndexT, kv_dt, index_dt, kIndexFp4, kKvFp4,
                                            kNv>(
                    reinterpret_cast<T*>(qkv.data_ptr()),
                    q_ptr,
                    index_q_ptr,
                    index_q_fp8_ptr,
                    reinterpret_cast<const T*>(q_norm_weight.data_ptr()),
                    reinterpret_cast<const T*>(k_norm_weight.data_ptr()),
                    process_index ? reinterpret_cast<const T*>(index_q_norm_weight->data_ptr())
                                  : nullptr,
                    process_index ? reinterpret_cast<const T*>(index_k_norm_weight->data_ptr())
                                  : nullptr,
                    reinterpret_cast<const T*>(cos_sin_cache.data_ptr()),
                    reinterpret_cast<int64_t*>(positions.data_ptr()),
                    reinterpret_cast<int64_t*>(slot_mapping->data_ptr()),
                    process_index ? reinterpret_cast<int64_t*>(index_slot_mapping->data_ptr())
                                  : nullptr,
                    reinterpret_cast<CacheT*>(kv_cache_k->data_ptr()),
                    reinterpret_cast<CacheT*>(kv_cache_v->data_ptr()),
                    process_index ? reinterpret_cast<IndexT*>(index_cache->data_ptr()) : nullptr,
                    kv_scaled ? reinterpret_cast<float*>(k_scale->data_ptr()) : nullptr,
                    kv_scaled ? reinterpret_cast<float*>(v_scale->data_ptr()) : nullptr,
                    static_cast<float>(eps),
                    static_cast<int>(rotary_dim),
                    num_tokens,
                    nq,
                    nkv,
                    niq,
                    static_cast<int>(block_size),
                    x,
                    max_kv_tokens,
                    k_s_block,
                    k_s_token,
                    k_s_head,
                    v_s_block,
                    v_s_token,
                    v_s_head,
                    has_index,
                    process_index,
                    insert_kv,
                    fp8_kv_per_token,
                    fp8_kv_static,
                    asm_layout,
                    stream,
                    q_fp8_ptr,
                    partials);
            };
            using Dt = vllm::Fp8KVCacheDataType;
            if(nvfp4_kv_cache)
            {
                launch_fp4.template
                operator()<uint8_t, uint8_t, Dt::kAuto, Dt::kAuto, true, true, true>();
            }
            else if(fp4_kv_cache && fp4_index_cache)
            {
                launch_fp4.template
                operator()<uint8_t, uint8_t, Dt::kAuto, Dt::kAuto, true, true, false>();
            }
            else if(fp4_kv_cache && process_index && index_cache_dtype != "auto")
            {
                launch_fp4.template
                operator()<uint8_t, opus::fp8_t, Dt::kAuto, Dt::kFp8E4M3, false, true, false>();
            }
            else if(fp4_kv_cache)
            {
                launch_fp4
                    .template operator()<uint8_t, T, Dt::kAuto, Dt::kAuto, false, true, false>();
            }
            else if(fp8_kv_cache)
            {
                launch_fp4.template
                operator()<opus::fp8_t, uint8_t, Dt::kFp8E4M3, Dt::kAuto, true, false, false>();
            }
            else
            {
                launch_fp4
                    .template operator()<T, uint8_t, Dt::kAuto, Dt::kAuto, true, false, false>();
            }
            return;
        }
        if(fp8_kv_cache)
        {
            if(fp8_index_cache)
            {
                launchFusedQKNormIdxrQKNorm<T,
                                            opus::fp8_t,
                                            opus::fp8_t,
                                            vllm::Fp8KVCacheDataType::kFp8E4M3,
                                            vllm::Fp8KVCacheDataType::kFp8E4M3>(
                reinterpret_cast<T*>(qkv.data_ptr()),
                q_ptr,
                index_q_ptr,
                index_q_fp8_ptr,
                reinterpret_cast<const T*>(q_norm_weight.data_ptr()),
                reinterpret_cast<const T*>(k_norm_weight.data_ptr()),
                process_index ? reinterpret_cast<const T*>(index_q_norm_weight->data_ptr())
                              : nullptr,
                process_index ? reinterpret_cast<const T*>(index_k_norm_weight->data_ptr())
                              : nullptr,
                reinterpret_cast<const T*>(cos_sin_cache.data_ptr()),
                reinterpret_cast<int64_t*>(positions.data_ptr()),
                insert_kv ? reinterpret_cast<int64_t*>(slot_mapping->data_ptr()) : nullptr,
                (insert_kv && process_index)
                    ? reinterpret_cast<int64_t*>(index_slot_mapping->data_ptr())
                    : nullptr,
                insert_kv ? reinterpret_cast<opus::fp8_t*>(kv_cache_k->data_ptr()) : nullptr,
                insert_kv ? reinterpret_cast<opus::fp8_t*>(kv_cache_v->data_ptr()) : nullptr,
                (insert_kv && process_index)
                    ? reinterpret_cast<opus::fp8_t*>(index_cache->data_ptr())
                    : nullptr,
                (fp8_kv_per_token || fp8_kv_static)
                    ? reinterpret_cast<float*>(k_scale->data_ptr())
                    : nullptr,
                (fp8_kv_per_token || fp8_kv_static)
                    ? reinterpret_cast<float*>(v_scale->data_ptr())
                    : nullptr,
                static_cast<float>(eps),
                static_cast<int>(rotary_dim),
                num_tokens,
                nq,
                nkv,
                niq,
                static_cast<int>(block_size),
                x,
                max_kv_tokens,
                k_s_block,
                k_s_token,
                k_s_head,
                v_s_block,
                v_s_token,
                v_s_head,
                has_index,
                process_index,
                insert_kv,
                fp8_kv_per_token,
                fp8_kv_static,
                asm_layout,
                stream,
                q_fp8_ptr,
                partials);
            }
            else
            {
                launchFusedQKNormIdxrQKNorm<T,
                                            opus::fp8_t,
                                            T,
                                            vllm::Fp8KVCacheDataType::kFp8E4M3,
                                            vllm::Fp8KVCacheDataType::kAuto>(
                reinterpret_cast<T*>(qkv.data_ptr()),
                q_ptr,
                index_q_ptr,
                index_q_fp8_ptr,
                reinterpret_cast<const T*>(q_norm_weight.data_ptr()),
                reinterpret_cast<const T*>(k_norm_weight.data_ptr()),
                process_index ? reinterpret_cast<const T*>(index_q_norm_weight->data_ptr())
                              : nullptr,
                process_index ? reinterpret_cast<const T*>(index_k_norm_weight->data_ptr())
                              : nullptr,
                reinterpret_cast<const T*>(cos_sin_cache.data_ptr()),
                reinterpret_cast<int64_t*>(positions.data_ptr()),
                insert_kv ? reinterpret_cast<int64_t*>(slot_mapping->data_ptr()) : nullptr,
                (insert_kv && process_index)
                    ? reinterpret_cast<int64_t*>(index_slot_mapping->data_ptr())
                    : nullptr,
                insert_kv ? reinterpret_cast<opus::fp8_t*>(kv_cache_k->data_ptr()) : nullptr,
                insert_kv ? reinterpret_cast<opus::fp8_t*>(kv_cache_v->data_ptr()) : nullptr,
                (insert_kv && process_index)
                    ? reinterpret_cast<T*>(index_cache->data_ptr())
                    : nullptr,
                (fp8_kv_per_token || fp8_kv_static)
                    ? reinterpret_cast<float*>(k_scale->data_ptr())
                    : nullptr,
                (fp8_kv_per_token || fp8_kv_static)
                    ? reinterpret_cast<float*>(v_scale->data_ptr())
                    : nullptr,
                static_cast<float>(eps),
                static_cast<int>(rotary_dim),
                num_tokens,
                nq,
                nkv,
                niq,
                static_cast<int>(block_size),
                x,
                max_kv_tokens,
                k_s_block,
                k_s_token,
                k_s_head,
                v_s_block,
                v_s_token,
                v_s_head,
                has_index,
                process_index,
                insert_kv,
                fp8_kv_per_token,
                fp8_kv_static,
                asm_layout,
                stream,
                q_fp8_ptr,
                partials);
            }
        }
        else
        {
            if(fp8_index_cache)
            {
                launchFusedQKNormIdxrQKNorm<T,
                                            T,
                                            opus::fp8_t,
                                            vllm::Fp8KVCacheDataType::kAuto,
                                            vllm::Fp8KVCacheDataType::kFp8E4M3>(
                reinterpret_cast<T*>(qkv.data_ptr()),
                q_ptr,
                index_q_ptr,
                index_q_fp8_ptr,
                reinterpret_cast<const T*>(q_norm_weight.data_ptr()),
                reinterpret_cast<const T*>(k_norm_weight.data_ptr()),
                process_index ? reinterpret_cast<const T*>(index_q_norm_weight->data_ptr())
                              : nullptr,
                process_index ? reinterpret_cast<const T*>(index_k_norm_weight->data_ptr())
                              : nullptr,
                reinterpret_cast<const T*>(cos_sin_cache.data_ptr()),
                reinterpret_cast<int64_t*>(positions.data_ptr()),
                insert_kv ? reinterpret_cast<int64_t*>(slot_mapping->data_ptr()) : nullptr,
                (insert_kv && process_index)
                    ? reinterpret_cast<int64_t*>(index_slot_mapping->data_ptr())
                    : nullptr,
                insert_kv ? reinterpret_cast<T*>(kv_cache_k->data_ptr()) : nullptr,
                insert_kv ? reinterpret_cast<T*>(kv_cache_v->data_ptr()) : nullptr,
                (insert_kv && process_index)
                    ? reinterpret_cast<opus::fp8_t*>(index_cache->data_ptr())
                    : nullptr,
                nullptr,
                nullptr,
                static_cast<float>(eps),
                static_cast<int>(rotary_dim),
                num_tokens,
                nq,
                nkv,
                niq,
                static_cast<int>(block_size),
                x,
                max_kv_tokens,
                k_s_block,
                k_s_token,
                k_s_head,
                v_s_block,
                v_s_token,
                v_s_head,
                has_index,
                process_index,
                insert_kv,
                fp8_kv_per_token,
                fp8_kv_static,
                asm_layout,
                stream,
                q_fp8_ptr,
                partials);
            }
            else
            {
                launchFusedQKNormIdxrQKNorm<T,
                                            T,
                                            T,
                                            vllm::Fp8KVCacheDataType::kAuto,
                                            vllm::Fp8KVCacheDataType::kAuto>(
                reinterpret_cast<T*>(qkv.data_ptr()),
                q_ptr,
                index_q_ptr,
                index_q_fp8_ptr,
                reinterpret_cast<const T*>(q_norm_weight.data_ptr()),
                reinterpret_cast<const T*>(k_norm_weight.data_ptr()),
                process_index ? reinterpret_cast<const T*>(index_q_norm_weight->data_ptr())
                              : nullptr,
                process_index ? reinterpret_cast<const T*>(index_k_norm_weight->data_ptr())
                              : nullptr,
                reinterpret_cast<const T*>(cos_sin_cache.data_ptr()),
                reinterpret_cast<int64_t*>(positions.data_ptr()),
                insert_kv ? reinterpret_cast<int64_t*>(slot_mapping->data_ptr()) : nullptr,
                (insert_kv && process_index)
                    ? reinterpret_cast<int64_t*>(index_slot_mapping->data_ptr())
                    : nullptr,
                insert_kv ? reinterpret_cast<T*>(kv_cache_k->data_ptr()) : nullptr,
                insert_kv ? reinterpret_cast<T*>(kv_cache_v->data_ptr()) : nullptr,
                (insert_kv && process_index)
                    ? reinterpret_cast<T*>(index_cache->data_ptr())
                    : nullptr,
                nullptr,
                nullptr,
                static_cast<float>(eps),
                static_cast<int>(rotary_dim),
                num_tokens,
                nq,
                nkv,
                niq,
                static_cast<int>(block_size),
                x,
                max_kv_tokens,
                k_s_block,
                k_s_token,
                k_s_head,
                v_s_block,
                v_s_token,
                v_s_head,
                has_index,
                process_index,
                insert_kv,
                fp8_kv_per_token,
                fp8_kv_static,
                asm_layout,
                stream,
                q_fp8_ptr,
                partials);
            }
        }
    });
}

void fused_qknorm_idxrqknorm(
    aiter_tensor_t& qkv,
    const aiter_tensor_t& q_norm_weight,
    const aiter_tensor_t& k_norm_weight,
    const aiter_tensor_t& cos_sin_cache,
    const aiter_tensor_t& positions,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t rotary_dim,
    double eps,
    std::optional<aiter_tensor_t> index_q_norm_weight,
    std::optional<aiter_tensor_t> index_k_norm_weight,
    int64_t num_index_heads,
    std::optional<aiter_tensor_t> slot_mapping,
    std::optional<aiter_tensor_t> kv_cache_k,
    std::optional<aiter_tensor_t> kv_cache_v,
    std::optional<aiter_tensor_t> index_cache,
    int64_t block_size,
    std::optional<aiter_tensor_t> q_out,
    std::optional<aiter_tensor_t> index_q_out,
    std::optional<aiter_tensor_t> index_slot_mapping,
    const std::string& kv_cache_dtype,
    const std::string& index_cache_dtype,
    std::optional<aiter_tensor_t> k_scale,
    std::optional<aiter_tensor_t> v_scale,
    bool asm_layout,
    bool skip_index_branch,
    std::optional<aiter_tensor_t> qkv_partials,
    bool check_only)
{
    // A refused argument raises (RuntimeError in Python) before anything launches. Under
    // check_only a refusal is the answer, not an error: it is not printed, and it raises
    // std::invalid_argument (ValueError) so that the query can tell it from a build or
    // load failure. check_only runs the checks only, so any error here is a refusal.
    aiter_detail::AiterThrowGuard throw_guard(/*quiet=*/check_only);
    try
    {
        fused_qknorm_idxrqknorm_impl(qkv,
                                     q_norm_weight,
                                     k_norm_weight,
                                     cos_sin_cache,
                                     positions,
                                     num_heads,
                                     num_kv_heads,
                                     rotary_dim,
                                     eps,
                                     index_q_norm_weight,
                                     index_k_norm_weight,
                                     num_index_heads,
                                     slot_mapping,
                                     kv_cache_k,
                                     kv_cache_v,
                                     index_cache,
                                     block_size,
                                     q_out,
                                     index_q_out,
                                     index_slot_mapping,
                                     kv_cache_dtype,
                                     index_cache_dtype,
                                     k_scale,
                                     v_scale,
                                     asm_layout,
                                     skip_index_branch,
                                     qkv_partials,
                                     check_only);
    }
    catch(const std::runtime_error& e)
    {
        if(!check_only)
        {
            throw;
        }
        throw std::invalid_argument(e.what());
    }
}

} // namespace aiter
