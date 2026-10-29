// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <cstdint>
#include <hip/hip_runtime.h>

// KV storage layouts, independent of query/head performance policy and tile size.
namespace fused_kv_layout {
constexpr int strided_fp8_rows     = -3;
constexpr int contiguous_bf16_rows = -2;
constexpr int legacy_tagged_fp8    = 0;
} // namespace fused_kv_layout

// FP8 uses either independent data/scale row views or legacy tagged byte offsets.
// Contiguous BF16 values are loaded directly by the cache-source helpers.
struct Fp8CacheLayout
{
    const uint8_t* scales   = nullptr;
    int64_t data_row_stride = 0, scale_row_stride = 0;
    int layout = fused_kv_layout::legacy_tagged_fp8, num_rows = 0;

    template <int KvLayout>
    __device__ __forceinline__ int64_t row(const void* indices, int i) const
    {
        if constexpr(KvLayout == fused_kv_layout::legacy_tagged_fp8)
            return static_cast<const int64_t*>(indices)[i];
        else
            return static_cast<const int*>(indices)[i];
    }
    template <int KvLayout>
    __device__ __forceinline__ bool valid(int64_t r, int64_t data_bytes) const
    {
        if constexpr(KvLayout == fused_kv_layout::legacy_tagged_fp8)
            return r >= 0 && (r & 7) == 1 && data_bytes >= 528 && (r >> 1) <= data_bytes - 528;
        else
            return r >= 0 && r < num_rows;
    }
    template <int KvLayout>
    __device__ __forceinline__ const uint8_t* data(const void* base, int64_t r) const
    {
        static_assert(KvLayout == fused_kv_layout::strided_fp8_rows ||
                      KvLayout == fused_kv_layout::legacy_tagged_fp8);
        if constexpr(KvLayout == fused_kv_layout::strided_fp8_rows)
            return static_cast<const uint8_t*>(base) + r * data_row_stride;
        else
            return static_cast<const uint8_t*>(base) + (r >> 1);
    }
    template <int KvLayout>
    __device__ __forceinline__ const uint8_t* scale(const void* base, int64_t r) const
    {
        static_assert(KvLayout == fused_kv_layout::strided_fp8_rows ||
                      KvLayout == fused_kv_layout::legacy_tagged_fp8);
        if constexpr(KvLayout == fused_kv_layout::strided_fp8_rows)
            return scales + r * scale_row_stride;
        else
            return static_cast<const uint8_t*>(base) + (r >> 1) + 512;
    }
};

#include <cstddef>
#include <hip/hip_runtime.h>
#include <opus/opus.hpp>
using B = opus::dtypes::bf16;
struct FusedNormRopeAttnRopeCastArgs
{
    const B* q;
    const uint8_t* q_scale;
    const uint8_t* pool;
    const void* prefix;
    const int* prefix_indptr;
    const B* extend_kv;
    const int *extend, *extend_indptr;
    const void* positions;
    const B *cos, *sin;
    const float* sink;
    uint8_t *out, *scales;
    float *lse, *max_logits, *debug_out;
    int tokens, heads, extend_rows, prefix_capacity, extend_capacity, positions64;
    int64_t pool_bytes;
    float softmax_scale;
    bool fp8_qk;
    bool q_norm;
    float norm_eps;
    Fp8CacheLayout cache;
    Fp8CacheLayout extend_cache;
    int num_splits;
    float* partial;
    template <int KvLayout>
    __device__ __forceinline__ const void* prefix_at(int begin) const
    {
        return KvLayout != fused_kv_layout::legacy_tagged_fp8
                   ? static_cast<const void*>(static_cast<const int*>(prefix) + begin)
                   : static_cast<const void*>(static_cast<const int64_t*>(prefix) + begin);
    }
};
hipError_t
launch_fused_norm_rope_attn_rope_cast(const FusedNormRopeAttnRopeCastArgs*, int, hipStream_t);

// Parameters for the 64-row and split-KV kernels. Keep this shared interface
// free of device pipeline definitions so the ctypes entry stays lightweight.
struct FusedNormRopeAttnRopeCastParams
{
    const void* __restrict__ q_ptr;                // [N, H, D] e4m3
    const void* __restrict__ q_scale_ptr;          // [N, H, D / 32] e8m0
    const void* __restrict__ unified_kv_ptr;       // [prefix_rows, D], prefix source
    const void* __restrict__ unified_kv_scale_ptr; // [prefix_rows, D / 32]
    const void* __restrict__ kv_ptr;               // [extend_rows, D], extend source
    const void* __restrict__ kv_scale_ptr;         // [extend_rows, D / 32]
    const void* __restrict__ attn_sink_ptr;        // [H] fp32, may be nullptr
    void* __restrict__ out_ptr;                    // [N, H, D] bf16
    void* __restrict__ lse_ptr;                    // [N, H] fp32, may be nullptr
    const int* __restrict__ kv_indptr_prefix;      // [N + 1]
    const int* __restrict__ kv_indices_prefix;     // [nnz_prefix]
    const int* __restrict__ kv_indptr_extend;      // [N + 1]
    const int* __restrict__ kv_indices_extend;     // [nnz_extend]
    int N;
    int H;
    int prefix_rows;
    int extend_rows;
    int stride_q_n;
    int stride_q_h;
    int stride_q_scale_n;
    int stride_q_scale_h;
    int stride_o_n;
    int stride_o_h;
    int stride_lse_n;
    int stride_kv_prefix;       // prefix data row stride in bytes
    int stride_kv_scale_prefix; // prefix scale row stride in bytes
    float softmax_scale;

    float* max_ptr;
    unsigned char* output_q_ptr;
    unsigned char* output_s_ptr;
    const void* cos_ptr;
    const void* sin_ptr;
    const void* positions_ptr;
    const void* raw_q_ptr;
    float* debug_ptr;
    int positions64;
    int q_is_rotated;
    int cache_mode; // 0: strided FP8 rows, 1: legacy tagged FP8
    int prefix_capacity;
    long long pool_bytes;
    bool q_norm;
    float norm_eps;
    int extend_capacity;
    int stride_kv_extend;       // independent extend data row stride in bytes
    int stride_kv_scale_extend; // independent extend scale row stride in bytes
    float* partial_ptr;
    int num_splits;
};

static_assert(sizeof(FusedNormRopeAttnRopeCastParams) == 288);
static_assert(offsetof(FusedNormRopeAttnRopeCastParams, lse_ptr) == 64);
static_assert(offsetof(FusedNormRopeAttnRopeCastParams, kv_indices_extend) == 96);
static_assert(offsetof(FusedNormRopeAttnRopeCastParams, N) == 104);
static_assert(offsetof(FusedNormRopeAttnRopeCastParams, stride_kv_prefix) == 148);
static_assert(offsetof(FusedNormRopeAttnRopeCastParams, softmax_scale) == 156);

hipError_t run_fused_norm_rope_attn_rope_cast_fwd(const FusedNormRopeAttnRopeCastParams& args,
                                                  bool raw_q,
                                                  hipStream_t stream);

bool use_fused_split_reduce(const FusedNormRopeAttnRopeCastParams& args);
hipError_t launch_fused_split_reduce(const FusedNormRopeAttnRopeCastParams& args, hipStream_t stream);

hipError_t run_fused_norm_rope_attn_rope_cast_fwd_small_heads(
    const FusedNormRopeAttnRopeCastParams& args, bool raw_q, hipStream_t stream);

hipError_t launch_small_head_output_reduce(const FusedNormRopeAttnRopeCastParams& args,
                                         hipStream_t stream);
