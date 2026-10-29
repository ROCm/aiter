// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Split-K for the opus MLA sparse prefill kernels. grid.z cuts each token's prefix rows, then its
// extend rows, into num_splits chunks of whole KV tiles. A split writes its O normalized by its own
// sum, its max (natural log units of the scaled scores) and that sum; pa_decode_ps_reduce_hip_kernel
// folds the splits and the sink.

#pragma once

#include <cfloat>
#include <cstdint>

struct opus_mla_prefill_split_args
{
    float* __restrict__ partial_o;   // [N, num_splits, H, 512]
    float* __restrict__ partial_max; // [N, num_splits, H]
    float* __restrict__ partial_sum; // [N, num_splits, H]
    int num_splits;
};

template <class Kargs>
struct opus_mla_prefill_split_kargs
{
    Kargs base;
    opus_mla_prefill_split_args split;
};

template <class Kargs>
__host__ __device__ inline const Kargs& opus_mla_prefill_base(const Kargs& kargs)
{ return kargs; }

template <class Kargs>
__host__ __device__ inline const Kargs&
opus_mla_prefill_base(const opus_mla_prefill_split_kargs<Kargs>& kargs)
{ return kargs.base; }

struct opus_mla_prefill_split_rows
{ int prefix_begin, prefix_len, extend_begin, extend_len; };

template <int KV_TILE>
__device__ inline opus_mla_prefill_split_rows
opus_mla_prefill_split_rows_of(int prefix_len, int extend_len, int num_splits, int split)
{
    auto clamp = [](int x, int hi) { return x < 0 ? 0 : (x > hi ? hi : x); };
    const int chunk =
        ((prefix_len + extend_len + num_splits - 1) / num_splits + KV_TILE - 1) / KV_TILE * KV_TILE;
    const int lo = split * chunk;
    const int hi = lo + chunk;
    const int pb = clamp(lo, prefix_len);
    const int eb = clamp(lo - prefix_len, extend_len);
    return {pb, clamp(hi, prefix_len) - pb, eb, clamp(hi - prefix_len, extend_len) - eb};
}

__device__ inline int64_t opus_mla_prefill_split_row(
    const opus_mla_prefill_split_args& split, int token, int split_idx, int H, int head)
{ return (static_cast<int64_t>(token) * split.num_splits + split_idx) * H + head; }

// m in log2 units. A split without keys stores -FLT_MAX, which the reduce skips; -inf would not
// survive -ffast-math.
__device__ inline void opus_mla_prefill_split_store_ml(const opus_mla_prefill_split_args& split,
                                                       int64_t row,
                                                       float m,
                                                       float l)
{
    split.partial_max[row] = l > 0.0f ? m * 0.69314718056f : -FLT_MAX;
    split.partial_sum[row] = l;
}
