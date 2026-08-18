// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
/*
 * Fused MiniMax-M3 attention pre-processing writing the page-16 SHUFFLE KV
 * layout that pa_decode_gluon consumes.
 *
 * The sparse layer's fused projection emits one packed row per token:
 *
 *     [ q (nq heads) | k (nkv) | v (nkv) | index_q (niq) | index_k (1) ]
 *
 * all with head_dim=128, all sharing the same partial-NeoX RoPE table.  The
 * four norms are Gemma-style RMSNorm, ``x * rsqrt(mean(x^2)+eps) * (1 + w)``,
 * with independent weights.  This kernel replaces the three-pass sequence
 *
 *     fused QK-norm/RoPE  ->  reshape_and_cache(asm_layout=True)
 *                         ->  index-cache scatter
 *
 * with a single pass: q and index_q are gathered into contiguous output
 * buffers, while k/v/index_k go straight from the projection output into their
 * caches and are never written back to ``qkv``.
 *
 * One warp (32 lanes) owns one (token, slot) pair; each lane owns 4 contiguous
 * head dims.  Slot order matches the packed row exactly, so a slot index
 * doubles as the head index inside the row:
 *
 *     [0, nq)                  Q  -> norm(q_w)  + RoPE -> q_out
 *     [nq, nq+nkv)             K  -> norm(k_w)  + RoPE -> k_cache (shuffle)
 *     [nq+nkv, nq+2*nkv)       V  -> verbatim          -> v_cache (shuffle)
 *     [nq+2*nkv, +niq)         IQ -> norm(iq_w) + RoPE -> index_q_out
 *     nq+2*nkv+niq             IK -> norm(ik_w) + RoPE -> index_cache
 *
 * Cache index math (page_size = k_cache.size(3), x = 16/elem_size):
 *
 *     page = slot / page_size,  off = slot % page_size
 *     k: page*k_block_stride + h*head_dim*page_size
 *          + (d/x)*page_size*x + off*x + (d%x)
 *     v: page*v_block_stride + h*head_dim*page_size
 *          + (off/x)*head_dim*x + d*x + (off%x)
 *
 * K keeps head_dim contiguous inside a 16-byte vector: x is a multiple of 4, so
 * a lane's 4 dims always land in one x group and the store stays a single
 * vector.  V transposes token against head_dim, which forces per-element stores
 * strided by x.
 */

#include <cmath>
#include <type_traits>

#include "aiter_dispatch.h"
#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "minimax_m3_fused_qknorm_rope_cache_shuffle.h"
#include "quant_utils.cuh"

#include "attention_dtypes.h"
#include "aiter_opus_plus.h"
#include "opus/opus.hpp"

#include <hip/hip_bf16.h>

namespace aiter {
namespace minimax_m3 {

constexpr int kHeadDim      = 128;
constexpr int kLanesPerHead = 32;
constexpr int kElemsPerLane = kHeadDim / kLanesPerHead; // 4
constexpr float kFp8Max     = 448.0f;

// Sum across the 32-lane group owning one head (a wave64 holds two groups).
__device__ __forceinline__ float headReduceSum(float v)
{
#pragma unroll
    for(int mask = kLanesPerHead / 2; mask > 0; mask >>= 1)
    {
        v += __shfl_xor(v, mask, kLanesPerHead);
    }
    return v;
}

// Gemma RMSNorm over the full head (skipped when ``weight == nullptr``) then
// partial NeoX RoPE over the leading ``rotary_dim`` dims.  Lane L owns dims
// [4L, 4L+4); ``half`` is a multiple of 4, so a lane lies wholly in one half of
// the rotary range and its RoPE partner sits ``half/4`` lanes away.
template <typename scalar_t>
__device__ __forceinline__ void normAndRope(float (&elems)[kElemsPerLane],
                                             const int lane,
                                             const float eps,
                                             const scalar_t* __restrict__ weight,
                                             const bool do_rope,
                                             const int rotary_dim,
                                             const scalar_t* __restrict__ cos_ptr)
{
    if(weight != nullptr)
    {
        float sumsq = 0.0f;
#pragma unroll
        for(int i = 0; i < kElemsPerLane; i++)
            sumsq += elems[i] * elems[i];
        sumsq               = headReduceSum(sumsq);
        const float rms_rcp = rsqrtf(sumsq / static_cast<float>(kHeadDim) + eps);
#pragma unroll
        for(int i = 0; i < kElemsPerLane; i++)
        {
            const float w = 1.0f + static_cast<float>(weight[lane * kElemsPerLane + i]);
            elems[i]      = elems[i] * rms_rcp * w;
        }
    }

    if(!do_rope)
        return;

    const int half     = rotary_dim / 2;
    const int dim0     = lane * kElemsPerLane;
    const int lane_xor = half / kElemsPerLane;
    float partner[kElemsPerLane];
#pragma unroll
    for(int i = 0; i < kElemsPerLane; i++)
        partner[i] = __shfl_xor(elems[i], lane_xor, kLanesPerHead);

    if(dim0 >= rotary_dim)
        return; // trailing dims pass through unrotated

    const bool first_half              = dim0 < half;
    const int i_base                   = first_half ? dim0 : (dim0 - half);
    const scalar_t* __restrict__ s_ptr = cos_ptr + half;
#pragma unroll
    for(int i = 0; i < kElemsPerLane; i++)
    {
        const float c = static_cast<float>(cos_ptr[i_base + i]);
        const float s = static_cast<float>(s_ptr[i_base + i]);
        elems[i]      = first_half ? (elems[i] * c - partner[i] * s)
                                    : (elems[i] * c + partner[i] * s);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Kernel
// ────────────────────────────────────────────────────────────────────────────
// idx_t is the index-K cache / index-Q output dtype: scalar_t, or fp8 e4m3 for
// the AITER fp8 score path.  kProcessIndex compiles away the index branch for
// skip-index-topk reuse layers (whose rows still carry the index sub-blocks).
template <typename scalar_t,
          typename cache_t,
          vllm::Fp8KVCacheDataType kv_dt,
          typename idx_t,
          bool kProcessIndex,
          bool kFp8Idx>
__global__ void minimaxM3QKNormRopeCacheShuffleInsertKernel(
    const scalar_t* __restrict__ qkv,
    scalar_t* __restrict__ q_out,
    idx_t* __restrict__ index_q_out,
    const scalar_t* __restrict__ q_norm_w,
    const scalar_t* __restrict__ k_norm_w,
    const scalar_t* __restrict__ iq_norm_w,
    const scalar_t* __restrict__ ik_norm_w,
    const scalar_t* __restrict__ cos_sin_cache,
    const int64_t* __restrict__ positions,
    const int64_t* __restrict__ slot_mapping,
    const int64_t* __restrict__ index_slot_mapping,
    cache_t* __restrict__ k_cache,
    cache_t* __restrict__ v_cache,
    idx_t* __restrict__ index_cache,
    const float* __restrict__ k_scale,
    const float* __restrict__ v_scale,
    const float eps,
    const int rotary_dim,
    const int num_tokens,
    const int nq,
    const int nkv,
    const int niq,
    const int page_size,
    const int x,
    const int64_t k_block_stride,
    const int64_t v_block_stride,
    const int64_t idx_page_stride,
    const int64_t idx_token_stride,
    const int idx_page_size)
{
    const int warps_per_block = blockDim.x / kLanesPerHead;
    const int lane            = threadIdx.x % kLanesPerHead;
    const int global_warp     = blockIdx.x * warps_per_block + (threadIdx.x / kLanesPerHead);

    const int v_begin  = nq + nkv;
    const int iq_begin = nq + 2 * nkv;
    const int ik_slot  = iq_begin + niq;
    const int slots    = kProcessIndex ? (ik_slot + 1) : iq_begin;

    const int token = global_warp / slots;
    const int slot  = global_warp % slots;
    if(token >= num_tokens)
        return;

    const bool isQ = slot < nq;
    const bool isK = slot >= nq && slot < v_begin;
    const bool isV = slot >= v_begin && slot < iq_begin;
    bool isIQ = false, isIK = false;
    if constexpr(kProcessIndex)
    {
        isIQ = slot >= iq_begin && slot < ik_slot;
        isIK = slot == ik_slot;
    }

    // Slot order mirrors the packed row, so ``slot`` is also the head index
    // inside the row -- one formula covers q/k/v/index_q/index_k.
    const int qkv_row  = (nq + 2 * nkv + niq + 1) * kHeadDim;
    const int dim_base = lane * kElemsPerLane;
    const scalar_t* __restrict__ row_ptr =
        qkv + static_cast<int64_t>(token) * qkv_row + static_cast<int64_t>(slot) * kHeadDim;

    const scalar_t* norm_w = nullptr;
    if(isQ)
        norm_w = q_norm_w;
    else if(isK)
        norm_w = k_norm_w;
    else if(isIQ)
        norm_w = iq_norm_w;
    else if(isIK)
        norm_w = ik_norm_w;

    float elems[kElemsPerLane];
#pragma unroll
    for(int i = 0; i < kElemsPerLane; i++)
        elems[i] = static_cast<float>(row_ptr[dim_base + i]);

    if(!isV)
    {
        const int64_t pos = positions[token];
        normAndRope<scalar_t>(elems,
                               lane,
                               eps,
                               norm_w,
                               /*do_rope=*/true,
                               rotary_dim,
                               cos_sin_cache + pos * rotary_dim);
    }

    // Round to the model dtype first: bf16/fp16 -> fp32 -> the same bf16/fp16 is
    // an exact round trip, so this reproduces the value the unfused path
    // materialized in ``qkv`` before quantizing.
    scalar_t rounded[kElemsPerLane];
#pragma unroll
    for(int i = 0; i < kElemsPerLane; i++)
        rounded[i] = opus::cast<scalar_t>(elems[i]);

    // ── Q / index_q gather into contiguous buffers ─────────────────────────
    if(isQ)
    {
        scalar_t* dst =
            q_out + static_cast<int64_t>(token) * nq * kHeadDim + slot * kHeadDim + dim_base;
#pragma unroll
        for(int i = 0; i < kElemsPerLane; i++)
            dst[i] = rounded[i];
        return;
    }
    if constexpr(kProcessIndex)
    {
        if(isIQ)
        {
            if(index_q_out != nullptr)
            {
                idx_t* dst = index_q_out + static_cast<int64_t>(token) * niq * kHeadDim +
                             (slot - iq_begin) * kHeadDim + dim_base;
#pragma unroll
                for(int i = 0; i < kElemsPerLane; i++)
                {
                    if constexpr(kFp8Idx)
                    {
                        // Straight from fp32, matching the unfused kernel's
                        // index_q store; going via bf16 would double-round.
                        const float v = fminf(fmaxf(elems[i], -kFp8Max), kFp8Max);
                        dst[i]        = opus::cast<idx_t>(v);
                    }
                    else
                    {
                        dst[i] = rounded[i];
                    }
                }
            }
            return;
        }
    }

    // ── Cache inserts ──────────────────────────────────────────────────────
    int64_t sm = -1;
    if(isK || isV)
        sm = slot_mapping[token];
    else if constexpr(kProcessIndex)
    {
        if(isIK)
            sm = index_slot_mapping[token];
    }
    if(sm < 0)
        return; // padded / unscheduled token

    if constexpr(kProcessIndex)
    {
        if(isIK)
        {
            if(index_cache != nullptr)
            {
                idx_t* dst = index_cache + (sm / idx_page_size) * idx_page_stride +
                             (sm % idx_page_size) * idx_token_stride + dim_base;
#pragma unroll
                for(int i = 0; i < kElemsPerLane; i++)
                {
                    // Deliberately quantized from the rounded scalar_t value:
                    // the path this replaces materialized index_k in ``qkv`` as
                    // bf16 before a separate insert cast it to fp8, so rounding
                    // here keeps the fused op bit-identical to it.
                    if constexpr(kFp8Idx)
                    {
                        const float v =
                            fminf(fmaxf(static_cast<float>(rounded[i]), -kFp8Max), kFp8Max);
                        dst[i] = opus::cast<idx_t>(v);
                    }
                    else
                    {
                        dst[i] = rounded[i];
                    }
                }
            }
            return;
        }
    }

    const int64_t page        = sm / page_size;
    const int off             = static_cast<int>(sm % page_size);
    const int64_t head_stride = static_cast<int64_t>(kHeadDim) * page_size;

    if(isK)
    {
        const int head = slot - nq;
        cache_t* dst   = k_cache + page * k_block_stride + head * head_stride +
                        static_cast<int64_t>(dim_base / x) * page_size * x +
                        static_cast<int64_t>(off) * x + (dim_base % x);
        const float inv = (k_scale == nullptr) ? 1.0f : 1.0f / (*k_scale);
#pragma unroll
        for(int i = 0; i < kElemsPerLane; i++)
        {
            if constexpr(kv_dt == vllm::Fp8KVCacheDataType::kAuto)
                dst[i] = rounded[i];
            else
                dst[i] = opus::cast<cache_t>(static_cast<float>(rounded[i]) * inv);
        }
    }
    else // isV
    {
        const int head = slot - v_begin;
        cache_t* dst   = v_cache + page * v_block_stride + head * head_stride +
                        static_cast<int64_t>(off / x) * kHeadDim * x +
                        static_cast<int64_t>(dim_base) * x + (off % x);
        const float inv = (v_scale == nullptr) ? 1.0f : 1.0f / (*v_scale);
#pragma unroll
        for(int i = 0; i < kElemsPerLane; i++)
        {
            if constexpr(kv_dt == vllm::Fp8KVCacheDataType::kAuto)
                dst[i * x] = rounded[i];
            else
                dst[i * x] = opus::cast<cache_t>(static_cast<float>(rounded[i]) * inv);
        }
    }
}

template <typename scalar_t, typename cache_t, vllm::Fp8KVCacheDataType kv_dt>
void launch(const scalar_t* qkv,
            scalar_t* q_out,
            void* index_q_out,
            const scalar_t* q_norm_w,
            const scalar_t* k_norm_w,
            const scalar_t* iq_norm_w,
            const scalar_t* ik_norm_w,
            const scalar_t* cos_sin_cache,
            const int64_t* positions,
            const int64_t* slot_mapping,
            const int64_t* index_slot_mapping,
            cache_t* k_cache,
            cache_t* v_cache,
            void* index_cache,
            const float* k_scale,
            const float* v_scale,
            float eps,
            int rotary_dim,
            int num_tokens,
            int nq,
            int nkv,
            int niq,
            int page_size,
            int x,
            int64_t k_block_stride,
            int64_t v_block_stride,
            int64_t idx_page_stride,
            int64_t idx_token_stride,
            int idx_page_size,
            bool process_index,
            bool fp8_idx,
            hipStream_t stream)
{
    constexpr int kBlockSize  = 256;
    const int slots           = process_index ? (nq + 2 * nkv + niq + 1) : (nq + 2 * nkv);
    const int64_t total_warps = static_cast<int64_t>(num_tokens) * slots;
    const int warps_per_block = kBlockSize / kLanesPerHead;
    const int grid = static_cast<int>((total_warps + warps_per_block - 1) / warps_per_block);
    if(grid == 0)
        return;

#define LAUNCH_M3_SHUFFLE(PROCESS_INDEX, FP8_IDX, IDX_T)                                   \
    minimaxM3QKNormRopeCacheShuffleInsertKernel<scalar_t,                                  \
                                                 cache_t,                                   \
                                                 kv_dt,                                     \
                                                 IDX_T,                                     \
                                                 PROCESS_INDEX,                             \
                                                 FP8_IDX><<<grid, kBlockSize, 0, stream>>>( \
        qkv,                                                                               \
        q_out,                                                                             \
        reinterpret_cast<IDX_T*>(index_q_out),                                             \
        q_norm_w,                                                                          \
        k_norm_w,                                                                          \
        iq_norm_w,                                                                         \
        ik_norm_w,                                                                         \
        cos_sin_cache,                                                                     \
        positions,                                                                         \
        slot_mapping,                                                                      \
        index_slot_mapping,                                                                \
        k_cache,                                                                           \
        v_cache,                                                                           \
        reinterpret_cast<IDX_T*>(index_cache),                                             \
        k_scale,                                                                           \
        v_scale,                                                                           \
        eps,                                                                               \
        rotary_dim,                                                                        \
        num_tokens,                                                                        \
        nq,                                                                                \
        nkv,                                                                                \
        niq,                                                                                \
        page_size,                                                                          \
        x,                                                                                  \
        k_block_stride,                                                                    \
        v_block_stride,                                                                    \
        idx_page_stride,                                                                    \
        idx_token_stride,                                                                   \
        idx_page_size)

    if(!process_index)
    {
        LAUNCH_M3_SHUFFLE(false, false, scalar_t);
    }
    else if(fp8_idx)
    {
        LAUNCH_M3_SHUFFLE(true, true, opus::fp8_t);
    }
    else
    {
        LAUNCH_M3_SHUFFLE(true, false, scalar_t);
    }
#undef LAUNCH_M3_SHUFFLE
}

} // namespace minimax_m3

namespace {
// Like is_contiguous() but ignoring dim 0, which permits an interleaved block
// stride (e.g. a vLLM [num_blocks, 2, ...] cache after unbind(1)).
inline bool contiguous_from_dim1(const aiter_tensor_t& t)
{
    if(t.numel() == 0)
        return true;
    int64_t expected = 1;
    for(int d = t.dim() - 1; d >= 1; --d)
    {
        if(t.size(d) != 1 && t.stride(d) != expected)
            return false;
        expected *= t.size(d);
    }
    return true;
}

// The scale is dereferenced on the device (as reshape_and_cache does), so a
// device scalar is required; nullptr means an identity scale.
inline const float* scale_ptr_of(const std::optional<aiter_tensor_t>& scale, const char* name)
{
    if(!scale.has_value() || scale->numel() == 0)
        return nullptr;
    AITER_CHECK(scale->dtype() == AITER_DTYPE_fp32 && scale->numel() == 1 && scale->is_gpu(),
                name,
                " must be a single-element fp32 GPU tensor");
    return reinterpret_cast<const float*>(scale->data_ptr());
}
} // namespace

void minimax_m3_qknorm_rope_cache_shuffle_insert(aiter_tensor_t& qkv,
                                                  aiter_tensor_t& q_norm_weight,
                                                  aiter_tensor_t& k_norm_weight,
                                                  aiter_tensor_t& cos_sin_cache,
                                                  aiter_tensor_t& positions,
                                                  int64_t num_heads,
                                                  int64_t num_kv_heads,
                                                  int64_t num_index_heads,
                                                  int64_t rotary_dim,
                                                  double eps,
                                                  aiter_tensor_t& slot_mapping,
                                                  aiter_tensor_t& k_cache,
                                                  aiter_tensor_t& v_cache,
                                                  aiter_tensor_t& q_out,
                                                  std::optional<aiter_tensor_t> index_q_norm_weight,
                                                  std::optional<aiter_tensor_t> index_k_norm_weight,
                                                  std::optional<aiter_tensor_t> index_slot_mapping,
                                                  std::optional<aiter_tensor_t> index_cache,
                                                  std::optional<aiter_tensor_t> index_q_out,
                                                  const std::string& kv_cache_dtype,
                                                  std::optional<aiter_tensor_t> k_scale,
                                                  std::optional<aiter_tensor_t> v_scale,
                                                  bool skip_index_branch)
{
    using namespace minimax_m3;

    const int nq  = static_cast<int>(num_heads);
    const int nkv = static_cast<int>(num_kv_heads);
    const int niq = static_cast<int>(num_index_heads);
    AITER_CHECK(nq > 0 && nkv > 0 && niq > 0,
                "minimax_m3 fused insert is the sparse-layer path: "
                "num_heads/num_kv_heads/num_index_heads must all be > 0");

    AITER_CHECK(qkv.is_gpu() && qkv.is_contiguous() && qkv.dim() == 2,
                "qkv must be a contiguous 2-D GPU tensor");
    AITER_CHECK(qkv.dtype() == AITER_DTYPE_bf16 || qkv.dtype() == AITER_DTYPE_fp16,
                "qkv must be bf16 or fp16");
    const int num_tokens = static_cast<int>(qkv.size(0));
    AITER_CHECK(qkv.size(1) == static_cast<int64_t>(nq + 2 * nkv + niq + 1) * kHeadDim,
                "qkv row must be (num_heads + 2*num_kv_heads + num_index_heads + 1) * 128, got ",
                qkv.size(1));

    AITER_CHECK(rotary_dim > 0 && rotary_dim % (2 * kElemsPerLane) == 0 && rotary_dim <= kHeadDim,
                "rotary_dim must be a positive multiple of 8 and <= 128, got ",
                rotary_dim);
    AITER_CHECK(cos_sin_cache.is_gpu() && cos_sin_cache.is_contiguous() &&
                    cos_sin_cache.dim() == 2 && cos_sin_cache.size(1) == rotary_dim &&
                    cos_sin_cache.dtype() == qkv.dtype(),
                "cos_sin_cache must be contiguous [max_pos, rotary_dim] matching qkv dtype");
    AITER_CHECK(positions.dtype() == AITER_DTYPE_i64 && positions.numel() >= num_tokens,
                "positions must be int64 with at least num_tokens elements");
    AITER_CHECK(slot_mapping.dtype() == AITER_DTYPE_i64 && slot_mapping.numel() >= num_tokens,
                "slot_mapping must be int64 with at least num_tokens elements");

    AITER_CHECK(q_norm_weight.numel() == kHeadDim && k_norm_weight.numel() == kHeadDim &&
                    q_norm_weight.dtype() == qkv.dtype() && k_norm_weight.dtype() == qkv.dtype(),
                "q/k norm weights must be 128 elements matching qkv dtype");

    AITER_CHECK(q_out.is_gpu() && q_out.is_contiguous() && q_out.dtype() == qkv.dtype() &&
                    q_out.numel() == static_cast<int64_t>(num_tokens) * nq * kHeadDim,
                "q_out must be contiguous [num_tokens, num_heads*128] matching qkv dtype");

    // Cache layout: k [pages, heads, D/x, page, x], v [pages, heads, page/x, D, x].
    AITER_CHECK(k_cache.dim() == 5 && v_cache.dim() == 5,
                "shuffle-layout k_cache/v_cache must be 5-D");
    AITER_CHECK(k_cache.dtype() == v_cache.dtype(), "k_cache/v_cache dtype must match");
    AITER_CHECK(contiguous_from_dim1(k_cache) && contiguous_from_dim1(v_cache),
                "k_cache/v_cache must be contiguous within a page (dims >= 1)");
    const int x         = static_cast<int>(k_cache.size(4));
    const int page_size = static_cast<int>(k_cache.size(3));
    AITER_CHECK(x == static_cast<int>(16 / k_cache.element_size()),
                "k_cache last dim must be 16 / element_size, got ",
                x);
    AITER_CHECK(x % kElemsPerLane == 0, "x must be a multiple of 4, got ", x);
    AITER_CHECK(page_size % x == 0, "page_size must be a multiple of x");
    AITER_CHECK(k_cache.size(1) == nkv && v_cache.size(1) == nkv,
                "k_cache/v_cache must have num_kv_heads in dim 1");
    AITER_CHECK(k_cache.size(2) == kHeadDim / x, "k_cache dim 2 must be head_dim/x");
    AITER_CHECK(v_cache.size(2) == page_size / x && v_cache.size(3) == kHeadDim &&
                    v_cache.size(4) == x,
                "v_cache must be [pages, heads, page_size/x, head_dim, x]");

    const bool process_index      = !skip_index_branch;
    const int64_t* index_slot_ptr = nullptr;
    void* index_cache_ptr         = nullptr;
    void* index_q_out_ptr         = nullptr;
    int64_t idx_page_stride = 0, idx_token_stride = 0;
    int idx_page_size       = 1;
    bool fp8_idx            = false;
    const void* iq_norm_ptr = nullptr;
    const void* ik_norm_ptr = nullptr;

    if(process_index)
    {
        AITER_CHECK(index_q_norm_weight.has_value() && index_k_norm_weight.has_value(),
                    "index branch requires both index norm weights");
        AITER_CHECK(index_q_norm_weight->numel() == kHeadDim &&
                        index_k_norm_weight->numel() == kHeadDim &&
                        index_q_norm_weight->dtype() == qkv.dtype() &&
                        index_k_norm_weight->dtype() == qkv.dtype(),
                    "index norm weights must be 128 elements matching qkv dtype");
        iq_norm_ptr = index_q_norm_weight->data_ptr();
        ik_norm_ptr = index_k_norm_weight->data_ptr();

        AITER_CHECK(index_cache.has_value(),
                    "index branch requires index_cache (pass skip_index_branch=True to disable)");
        AITER_CHECK(index_cache->dim() == 3 && index_cache->stride(2) == 1 &&
                        index_cache->size(2) == kHeadDim,
                    "index_cache must be [pages, page_size, 128] with contiguous head dim");
        AITER_CHECK(index_cache->dtype() == qkv.dtype() || index_cache->dtype() == AITER_DTYPE_fp8,
                    "index_cache must match qkv dtype or be fp8 e4m3");
        idx_page_stride  = index_cache->stride(0);
        idx_token_stride = index_cache->stride(1);
        idx_page_size    = static_cast<int>(index_cache->size(1));
        index_cache_ptr  = index_cache->data_ptr();
        fp8_idx          = index_cache->dtype() == AITER_DTYPE_fp8;

        AITER_CHECK(index_slot_mapping.has_value() &&
                        index_slot_mapping->dtype() == AITER_DTYPE_i64 &&
                        index_slot_mapping->numel() >= num_tokens,
                    "index branch requires an int64 index_slot_mapping");
        index_slot_ptr = reinterpret_cast<const int64_t*>(index_slot_mapping->data_ptr());

        if(index_q_out.has_value())
        {
            AITER_CHECK(index_q_out->is_contiguous() &&
                            index_q_out->numel() ==
                                static_cast<int64_t>(num_tokens) * niq * kHeadDim,
                        "index_q_out must be contiguous [num_tokens, niq*128]");
            AITER_CHECK(index_q_out->dtype() == index_cache->dtype(),
                        "index_q_out dtype must match index_cache dtype");
            index_q_out_ptr = index_q_out->data_ptr();
        }
    }

    const bool quantized     = kv_cache_dtype != "auto";
    const float* k_scale_ptr = quantized ? scale_ptr_of(k_scale, "k_scale") : nullptr;
    const float* v_scale_ptr = quantized ? scale_ptr_of(v_scale, "v_scale") : nullptr;
    AITER_CHECK(quantized ? (k_cache.dtype() == AITER_DTYPE_fp8)
                          : (k_cache.dtype() == qkv.dtype()),
                "k_cache must be fp8 for a quantized kv_cache_dtype and match qkv otherwise");

    HipDeviceGuard device_guard(qkv.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

#define CALL_M3_SHUFFLE_INSERT(SCALAR_T, CACHE_T, KV_DTYPE)                 \
    minimax_m3::launch<SCALAR_T, CACHE_T, KV_DTYPE>(                        \
        reinterpret_cast<const SCALAR_T*>(qkv.data_ptr()),                  \
        reinterpret_cast<SCALAR_T*>(q_out.data_ptr()),                      \
        index_q_out_ptr,                                                    \
        reinterpret_cast<const SCALAR_T*>(q_norm_weight.data_ptr()),        \
        reinterpret_cast<const SCALAR_T*>(k_norm_weight.data_ptr()),        \
        reinterpret_cast<const SCALAR_T*>(iq_norm_ptr),                     \
        reinterpret_cast<const SCALAR_T*>(ik_norm_ptr),                     \
        reinterpret_cast<const SCALAR_T*>(cos_sin_cache.data_ptr()),        \
        reinterpret_cast<const int64_t*>(positions.data_ptr()),             \
        reinterpret_cast<const int64_t*>(slot_mapping.data_ptr()),          \
        index_slot_ptr,                                                     \
        reinterpret_cast<CACHE_T*>(k_cache.data_ptr()),                     \
        reinterpret_cast<CACHE_T*>(v_cache.data_ptr()),                     \
        index_cache_ptr,                                                    \
        k_scale_ptr,                                                        \
        v_scale_ptr,                                                        \
        static_cast<float>(eps),                                            \
        static_cast<int>(rotary_dim),                                       \
        num_tokens,                                                         \
        nq,                                                                 \
        nkv,                                                                \
        niq,                                                                \
        page_size,                                                          \
        x,                                                                  \
        k_cache.stride(0),                                                  \
        v_cache.stride(0),                                                  \
        idx_page_stride,                                                    \
        idx_token_stride,                                                   \
        idx_page_size,                                                      \
        process_index,                                                      \
        fp8_idx,                                                            \
        stream)

    DISPATCH_BY_KV_CACHE_DTYPE_OPUS_rmTorch(qkv.dtype(), kv_cache_dtype, CALL_M3_SHUFFLE_INSERT)
#undef CALL_M3_SHUFFLE_INSERT
}

} // namespace aiter
