// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Kernel argument ABI for the prebuilt split-KV MLA decode code objects. The
// layout is frozen: the .co files are compiled against these structs, so any
// field change here requires rebuilding them.

#pragma once

#include <cstddef>

// One entry of the scheduler's work queue, matching aiter's MlaWorkInfo.
struct opus_mla_decode_work_info
{
    int batch_idx;
    int partial_slot;
    int qo_start;
    int qo_end;
    int kv_start;
    int kv_end;
    int kv_offset;
    int _pad;
};

struct opus_mla_decode_kargs
{
    const void* __restrict__ q_ptr;
    const void* __restrict__ kv_ptr;
    void* __restrict__ out_ptr;
    void* __restrict__ lse_ptr;
    void* __restrict__ o_accum;
    void* __restrict__ lse_accum;

    const int* __restrict__ q_indptr;
    const int* __restrict__ kv_indptr;
    const int* __restrict__ kv_indices;
    const int* __restrict__ work_indptr;
    const opus_mla_decode_work_info* __restrict__ work_info_set;

    int H;
    int total_tokens;
    int stride_q_b;
    int stride_q_h;
    int stride_o_b;
    int stride_o_h;
    int stride_kv_page;
    float softmax_scale;
};

struct opus_mla_decode_mxfp8_kargs
{
    const void* __restrict__ q_nope_ptr;
    const void* __restrict__ q_scale_ptr;
    const void* __restrict__ q_rope_ptr;
    const void* __restrict__ kv_nope_ptr;
    const void* __restrict__ kv_scale_ptr;
    const void* __restrict__ kv_rope_ptr;
    void* __restrict__ out_ptr;
    void* __restrict__ lse_ptr;
    void* __restrict__ o_accum;
    void* __restrict__ lse_accum;

    const int* __restrict__ q_indptr;
    const int* __restrict__ kv_indptr;
    const int* __restrict__ kv_indices;
    const int* __restrict__ work_indptr;
    const opus_mla_decode_work_info* __restrict__ work_info_set;

    int H;
    int total_tokens;
    int stride_q_nope_b;
    int stride_q_nope_h;
    int stride_q_scale_b;
    int stride_q_scale_h;
    int stride_q_rope_b;
    int stride_q_rope_h;
    int stride_o_b;
    int stride_o_h;
    int stride_kv_nope_page;
    int stride_kv_scale_page;
    int stride_kv_rope_page;
    float softmax_scale;
};

static_assert(sizeof(opus_mla_decode_work_info) == 32);
static_assert(sizeof(opus_mla_decode_kargs) == 120);
static_assert(offsetof(opus_mla_decode_kargs, work_info_set) == 80);
static_assert(offsetof(opus_mla_decode_kargs, softmax_scale) == 116);
static_assert(sizeof(opus_mla_decode_mxfp8_kargs) == 176);
static_assert(offsetof(opus_mla_decode_mxfp8_kargs, work_info_set) == 112);
static_assert(offsetof(opus_mla_decode_mxfp8_kargs, softmax_scale) == 172);
