// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Kernel arguments of the MLA v4 sparse prefill kernels. The gfx1250 code objects are built
// against them; opus_mla_prefill_fwd.cu pins their layout.

#pragma once

#include "mla_prefill_split.h"

// Kernel arguments.
struct opus_mla_v4_prefill_kargs
{
    const void* __restrict__ q_ptr;          // [N, H, D]
    const void* __restrict__ unified_kv_ptr; // [total_pages, D], prefix source
    const void* __restrict__ kv_ptr;         // [total_tokens, D], extend source
    const void* __restrict__ attn_sink_ptr;  // [H], softmax denominator sink, may be nullptr
    void* __restrict__ out_ptr;              // [N, H, D]
    const int* __restrict__ kv_indptr_prefix;  // [N+1]
    const int* __restrict__ kv_indices_prefix; // [nnz_prefix]
    const int* __restrict__ kv_indptr_extend;  // [N+1]
    const int* __restrict__ kv_indices_extend; // [nnz_extend]
    int N;
    int H;
    int D;
    int total_pages;
    int total_tokens;
    int stride_qo_n;
    int stride_qo_h;
    int stride_kv_page;
    float softmax_scale;
};

// Kernel arguments for the split-precision (NoPE fp8 / RoPE bf16) DSA prefill.
struct opus_mla_v4_prefill_fp8_kargs
{
    const void* __restrict__ q_nope_ptr;          // [N, H, D_NOPE_PADDED] fp8
    const void* __restrict__ q_rope_ptr;          // [N, H, D_ROPE]        bf16
    const void* __restrict__ unified_kv_nope_ptr; // [total_pages, D_NOPE_PADDED] fp8
    const void* __restrict__ unified_kv_rope_ptr; // [total_pages, D_ROPE]        bf16
    const void* __restrict__ kv_nope_ptr;         // [total_tokens, D_NOPE_PADDED] fp8
    const void* __restrict__ kv_rope_ptr;         // [total_tokens, D_ROPE]        bf16
    const void* __restrict__ attn_sink_ptr;       // [H], may be nullptr
    void* __restrict__ out_ptr;                   // [N, H, D_HEAD] bf16
    const int* __restrict__ kv_indptr_prefix;     // [N+1]
    const int* __restrict__ kv_indices_prefix;    // [nnz_prefix]
    const int* __restrict__ kv_indptr_extend;     // [N+1]
    const int* __restrict__ kv_indices_extend;    // [nnz_extend]
    int N;
    int H;
    int total_pages;
    int total_tokens;
    int stride_q_nope_n;
    int stride_q_nope_h;
    int stride_q_rope_n;
    int stride_q_rope_h;
    int stride_o_n;
    int stride_o_h;
    int stride_kv_nope_page;
    int stride_kv_rope_page;
    float softmax_scale;
};
