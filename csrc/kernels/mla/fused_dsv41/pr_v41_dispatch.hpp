// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include "aiter_hip_common.h"
#include "pr_v41/kernel_args.h"

// The code objects preserve FP32 max/LSE and fuse Q preparation and output IR/quant.
// Their first CTA phase validates indices before choosing checked/unchecked tile loops.
inline bool launch_fused_dsv41_pr_v41(const FusedDsv41Args& a, bool rotate_q, hipStream_t stream)
{
    // Measured crossover: short KV lists lose; T512 wide-cache gains are marginal.
    if(!a.fp8_qk || a.heads != 128 || a.tokens == 0 || a.tokens > 320 ||
       a.prefix_capacity < int64_t(a.tokens)*1024 || a.extend_indptr ||
       (a.cache.page_size != 0 && a.cache.page_size != 128))
        return false;
    opus_mla_v41_prefill_kargs k{};
    k.q_ptr=a.q; k.q_scale_ptr=a.q_scale; k.raw_q_ptr=a.q;
    k.unified_kv_ptr=a.pool; k.unified_kv_scale_ptr=a.cache.scales;
    k.kv_indptr_prefix=a.prefix_indptr;
    k.kv_indices_prefix=static_cast<const int*>(a.prefix);
    k.attn_sink_ptr=a.sink; k.lse_ptr=a.lse; k.max_ptr=a.max_logits;
    k.output_q_ptr=a.out; k.output_s_ptr=a.scales; k.debug_ptr=a.debug_out;
    k.cos_ptr=a.cos; k.sin_ptr=a.sin; k.positions_ptr=a.positions;
    k.N=a.tokens; k.H=a.heads;
    k.stride_q_n=a.heads*512; k.stride_q_h=512;
    k.stride_q_scale_n=a.heads*16; k.stride_q_scale_h=16;
    k.stride_o_n=a.heads*512; k.stride_o_h=512; k.stride_lse_n=a.heads;
    k.softmax_scale=a.softmax_scale; k.positions64=a.positions64; k.q_is_rotated=!rotate_q;
    k.cache_mode=a.cache.page_size ? 2 : 1; k.page_size=a.cache.page_size;
    // num_rows was bounded by BOTH actual byte views by the public wrapper.
    k.cache_rows=a.cache.num_rows; k.prefix_capacity=a.prefix_capacity;
    k.pool_bytes=a.pool_bytes;
    k.data_page_stride=a.cache.data_page_stride; k.scale_page_stride=a.cache.scale_page_stride;
    constexpr const char* symbol="opus_mla_v41_prefill_a8w8_16mx4_64nx1_kernel";
    size_t bytes=sizeof(k);
    const AiterAsmKernelArgs launch{&k,&bytes,a.tokens,2,1,256,1,1,stream};
    if(a.cache.page_size == 0) {
        if(a.q_scale) {
            static AiterAsmKernel kernel(symbol,"fused_dsv41/legacy_check_prepared.co");
            kernel.launch_kernel(launch);
        } else {
            static AiterAsmKernel kernel(symbol,"fused_dsv41/legacy_check_raw.co");
            kernel.launch_kernel(launch);
        }
    } else {
        if(a.q_scale) {
            static AiterAsmKernel kernel(symbol,"fused_dsv41/paged128_check_prepared.co");
            kernel.launch_kernel(launch);
        } else {
            static AiterAsmKernel kernel(symbol,"fused_dsv41/paged128_check_raw.co");
            kernel.launch_kernel(launch);
        }
    }
    return true;
}
