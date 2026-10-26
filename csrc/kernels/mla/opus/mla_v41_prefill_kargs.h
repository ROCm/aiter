// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Kernel argument ABI of the prebuilt MLA v4.1 sparse prefill code object; a field change
// requires rebuilding it.

#pragma once

#include <cstddef>

#include "mla_prefill_split.h"

struct opus_mla_v41_prefill_kargs
{
    const void* __restrict__ q_ptr;                // [N, H, D] e4m3
    const void* __restrict__ q_scale_ptr;          // [N, H, D / 32] e8m0
    const void* __restrict__ unified_kv_ptr;       // [total_pages, D], prefix source
    const void* __restrict__ unified_kv_scale_ptr; // [total_pages, D / 32]
    const void* __restrict__ kv_ptr;               // [total_tokens, D], extend source
    const void* __restrict__ kv_scale_ptr;         // [total_tokens, D / 32]
    const void* __restrict__ attn_sink_ptr;        // [H] fp32, may be nullptr
    void* __restrict__ out_ptr;                    // [N, H, D] bf16
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
    int stride_kv_page;       // shared by unified_kv and kv
    int stride_kv_scale_page; // shared by unified_kv_scale and kv_scale
    float softmax_scale;
};

static_assert(sizeof(opus_mla_v41_prefill_kargs) == 152);
static_assert(offsetof(opus_mla_v41_prefill_kargs, kv_indices_extend) == 88);
static_assert(offsetof(opus_mla_v41_prefill_kargs, N) == 96);
static_assert(offsetof(opus_mla_v41_prefill_kargs, stride_kv_page) == 136);
static_assert(offsetof(opus_mla_v41_prefill_kargs, softmax_scale) == 144);
static_assert(sizeof(opus_mla_prefill_split_kargs<opus_mla_v41_prefill_kargs>) == 184);
