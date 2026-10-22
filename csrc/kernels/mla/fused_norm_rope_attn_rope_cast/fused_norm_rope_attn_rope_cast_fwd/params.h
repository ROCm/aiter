// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Parameters shared by the fused attention dispatcher and device kernel.

#pragma once

#include <cstddef>

struct FusedNormRopeAttnRopeCastParams
{
    const void* __restrict__ q_ptr;                // [N, H, D] e4m3
    const void* __restrict__ q_scale_ptr;          // [N, H, D / 32] e8m0
    const void* __restrict__ unified_kv_ptr;       // [total_pages, D], prefix source
    const void* __restrict__ unified_kv_scale_ptr; // [total_pages, D / 32]
    const void* __restrict__ kv_ptr;               // [total_tokens, D], extend source
    const void* __restrict__ kv_scale_ptr;         // [total_tokens, D / 32]
    const void* __restrict__ attn_sink_ptr;        // [H] fp32, may be nullptr
    void* __restrict__ out_ptr;                    // [N, H, D] bf16
    void* __restrict__ lse_ptr;                    // [N, H] fp32, may be nullptr
    const int* __restrict__ kv_indptr_prefix;      // [N + 1]
    const int* __restrict__ kv_indices_prefix;     // [nnz_prefix]
    const int* __restrict__ kv_indptr_extend;      // [N + 1]
    const int* __restrict__ kv_indices_extend;     // [nnz_extend]
    int N;
    int H;
    int total_pages;
    int total_tokens;
    int stride_q_n;
    int stride_q_h;
    int stride_q_scale_n;
    int stride_q_scale_h;
    int stride_o_n;
    int stride_o_h;
    int stride_lse_n;
    int stride_kv_page;       // shared by unified_kv and kv
    int stride_kv_scale_page; // shared by unified_kv_scale and kv_scale
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
    int cache_mode;
    int page_size;
    int cache_rows;
    int prefix_capacity;
    long long pool_bytes;
    long long scale_bytes;
    long long data_page_stride;
    long long scale_page_stride;
    bool q_norm;
    float norm_eps;
};

static_assert(sizeof(FusedNormRopeAttnRopeCastParams) == 288);
static_assert(offsetof(FusedNormRopeAttnRopeCastParams, lse_ptr) == 64);
static_assert(offsetof(FusedNormRopeAttnRopeCastParams, kv_indices_extend) == 96);
static_assert(offsetof(FusedNormRopeAttnRopeCastParams, N) == 104);
static_assert(offsetof(FusedNormRopeAttnRopeCastParams, stride_kv_page) == 148);
static_assert(offsetof(FusedNormRopeAttnRopeCastParams, softmax_scale) == 156);
