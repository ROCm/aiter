// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Kernel argument ABI of the opus MLA sparse prefill kernels. The prebuilt code objects are compiled
// against it, so a layout change requires rebuilding them.

#pragma once

#include <cstddef>

#include "mla_prefill_split.h"

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

// NoPE fp8, RoPE bf16.
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

struct opus_mla_v41_prefill_kargs
{
    const void* __restrict__ q_ptr;                // [N, H, D] e4m3
    const void* __restrict__ q_scale_ptr;          // [N, H, D / 32] e8m0
    const void* __restrict__ unified_kv_ptr;       // [total_pages, D] e4m3, or [total_pages, D / 2] e2m1 pairs (a8w4)
    const void* __restrict__ unified_kv_scale_ptr; // [total_pages, D / 32] e8m0, or [total_pages, D / 16] e4m3 (a8w4)
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
    int stride_unified_kv_page;       // bytes
    int stride_unified_kv_scale_page; // bytes
    int stride_kv_page;               // bytes
    int stride_kv_scale_page;         // bytes
    float softmax_scale;
};

static_assert(sizeof(opus_mla_v4_prefill_kargs) == 112);
static_assert(offsetof(opus_mla_v4_prefill_kargs, q_ptr) == 0);
static_assert(offsetof(opus_mla_v4_prefill_kargs, unified_kv_ptr) == 8);
static_assert(offsetof(opus_mla_v4_prefill_kargs, kv_ptr) == 16);
static_assert(offsetof(opus_mla_v4_prefill_kargs, attn_sink_ptr) == 24);
static_assert(offsetof(opus_mla_v4_prefill_kargs, out_ptr) == 32);
static_assert(offsetof(opus_mla_v4_prefill_kargs, kv_indptr_prefix) == 40);
static_assert(offsetof(opus_mla_v4_prefill_kargs, kv_indices_prefix) == 48);
static_assert(offsetof(opus_mla_v4_prefill_kargs, kv_indptr_extend) == 56);
static_assert(offsetof(opus_mla_v4_prefill_kargs, kv_indices_extend) == 64);
static_assert(offsetof(opus_mla_v4_prefill_kargs, N) == 72);
static_assert(offsetof(opus_mla_v4_prefill_kargs, H) == 76);
static_assert(offsetof(opus_mla_v4_prefill_kargs, D) == 80);
static_assert(offsetof(opus_mla_v4_prefill_kargs, total_pages) == 84);
static_assert(offsetof(opus_mla_v4_prefill_kargs, total_tokens) == 88);
static_assert(offsetof(opus_mla_v4_prefill_kargs, stride_qo_n) == 92);
static_assert(offsetof(opus_mla_v4_prefill_kargs, stride_qo_h) == 96);
static_assert(offsetof(opus_mla_v4_prefill_kargs, stride_kv_page) == 100);
static_assert(offsetof(opus_mla_v4_prefill_kargs, softmax_scale) == 104);

static_assert(sizeof(opus_mla_v4_prefill_fp8_kargs) == 152);
static_assert(offsetof(opus_mla_v4_prefill_fp8_kargs, q_nope_ptr) == 0);
static_assert(offsetof(opus_mla_v4_prefill_fp8_kargs, q_rope_ptr) == 8);
static_assert(offsetof(opus_mla_v4_prefill_fp8_kargs, unified_kv_nope_ptr) == 16);
static_assert(offsetof(opus_mla_v4_prefill_fp8_kargs, unified_kv_rope_ptr) == 24);
static_assert(offsetof(opus_mla_v4_prefill_fp8_kargs, kv_nope_ptr) == 32);
static_assert(offsetof(opus_mla_v4_prefill_fp8_kargs, kv_rope_ptr) == 40);
static_assert(offsetof(opus_mla_v4_prefill_fp8_kargs, attn_sink_ptr) == 48);
static_assert(offsetof(opus_mla_v4_prefill_fp8_kargs, out_ptr) == 56);
static_assert(offsetof(opus_mla_v4_prefill_fp8_kargs, kv_indptr_prefix) == 64);
static_assert(offsetof(opus_mla_v4_prefill_fp8_kargs, kv_indices_prefix) == 72);
static_assert(offsetof(opus_mla_v4_prefill_fp8_kargs, kv_indptr_extend) == 80);
static_assert(offsetof(opus_mla_v4_prefill_fp8_kargs, kv_indices_extend) == 88);
static_assert(offsetof(opus_mla_v4_prefill_fp8_kargs, N) == 96);
static_assert(offsetof(opus_mla_v4_prefill_fp8_kargs, H) == 100);
static_assert(offsetof(opus_mla_v4_prefill_fp8_kargs, total_pages) == 104);
static_assert(offsetof(opus_mla_v4_prefill_fp8_kargs, total_tokens) == 108);
static_assert(offsetof(opus_mla_v4_prefill_fp8_kargs, stride_q_nope_n) == 112);
static_assert(offsetof(opus_mla_v4_prefill_fp8_kargs, stride_q_nope_h) == 116);
static_assert(offsetof(opus_mla_v4_prefill_fp8_kargs, stride_q_rope_n) == 120);
static_assert(offsetof(opus_mla_v4_prefill_fp8_kargs, stride_q_rope_h) == 124);
static_assert(offsetof(opus_mla_v4_prefill_fp8_kargs, stride_o_n) == 128);
static_assert(offsetof(opus_mla_v4_prefill_fp8_kargs, stride_o_h) == 132);
static_assert(offsetof(opus_mla_v4_prefill_fp8_kargs, stride_kv_nope_page) == 136);
static_assert(offsetof(opus_mla_v4_prefill_fp8_kargs, stride_kv_rope_page) == 140);
static_assert(offsetof(opus_mla_v4_prefill_fp8_kargs, softmax_scale) == 144);

static_assert(sizeof(opus_mla_v41_prefill_kargs) == 160);
static_assert(offsetof(opus_mla_v41_prefill_kargs, kv_indices_extend) == 88);
static_assert(offsetof(opus_mla_v41_prefill_kargs, N) == 96);
static_assert(offsetof(opus_mla_v41_prefill_kargs, stride_unified_kv_page) == 136);
static_assert(offsetof(opus_mla_v41_prefill_kargs, stride_kv_page) == 144);
static_assert(offsetof(opus_mla_v41_prefill_kargs, softmax_scale) == 152);

static_assert(sizeof(opus_mla_prefill_split_args) == 32);
static_assert(offsetof(opus_mla_prefill_split_args, partial_o) == 0);
static_assert(offsetof(opus_mla_prefill_split_args, partial_max) == 8);
static_assert(offsetof(opus_mla_prefill_split_args, partial_sum) == 16);
static_assert(offsetof(opus_mla_prefill_split_args, num_splits) == 24);
static_assert(sizeof(opus_mla_prefill_split_kargs<opus_mla_v4_prefill_kargs>) == 144);
static_assert(offsetof(opus_mla_prefill_split_kargs<opus_mla_v4_prefill_kargs>, split) == 112);
static_assert(sizeof(opus_mla_prefill_split_kargs<opus_mla_v4_prefill_fp8_kargs>) == 184);
static_assert(offsetof(opus_mla_prefill_split_kargs<opus_mla_v4_prefill_fp8_kargs>, split) == 152);
static_assert(sizeof(opus_mla_prefill_split_kargs<opus_mla_v41_prefill_kargs>) == 192);
