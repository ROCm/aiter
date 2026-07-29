// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// MXFP8 paged MQA logits (gfx950) — host launcher.
// Thin wrapper over the device kernel template in `pa_mqa_logits_mxfp8_opus.h`
// (single-header, IMPL-guarded). Q/KV must already be MXFP8-quantized and
// preshuffled into the kernel ABI; `cta_info` must already be built.

#define PA_MQA_LOGITS_MXFP8_IMPL
#include "pa_mqa_logits_mxfp8_opus.h"

#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "aiter_tensor.h"

// Compiled configuration: 4-wave MXFP8 variant (block_k=256, PAGE=64, D=128, H=64).
using mqa_logits_traits_4wave = opus_mqa_logits_traits<256, 64, 128, 64, 4>;

void pa_mqa_logits_mxfp8_fwd(aiter_tensor_t& q,
                          aiter_tensor_t& q_scale,
                          aiter_tensor_t& kv_cache,
                          aiter_tensor_t& kv_scale,
                          aiter_tensor_t& block_tables,
                          aiter_tensor_t& weights,
                          aiter_tensor_t& cta_info,
                          aiter_tensor_t& out,
                          int n_ctas,
                          float weight_scale,
                          int block_k,
                          int kv_block_size,
                          int max_seq_len)
{
    using Traits = mqa_logits_traits_4wave;

    // ---- Shape / dtype validation (single compiled config) ----------------
    AITER_CHECK(q.dim() == 3, "q must be 3-D [T, H, D], got ndim=", q.dim());
    AITER_CHECK(weights.dim() == 2, "weights must be 2-D [T, H], got ndim=", weights.dim());
    AITER_CHECK(block_tables.dim() == 2, "block_tables must be 2-D [batch, max_blocks_per_seq]");
    AITER_CHECK(cta_info.dim() == 2 && cta_info.size(1) == CTA_INFO_WIDTH,
                "cta_info must be 2-D [n_ctas, 6]");
    AITER_CHECK(out.dim() == 2, "out must be 2-D [T, max_seq_len], got ndim=", out.dim());

    const int T = static_cast<int>(q.size(0));
    const int H = static_cast<int>(q.size(1));
    const int D = static_cast<int>(q.size(2));
    AITER_CHECK(H == Traits::N_HEADS, "compiled for H=", (int)Traits::N_HEADS, ", got H=", H);
    AITER_CHECK(D == Traits::HEAD_DIM, "compiled for D=", (int)Traits::HEAD_DIM, ", got D=", D);
    AITER_CHECK(block_k == Traits::KV_TILE_SIZE,
                "compiled for block_k=", (int)Traits::KV_TILE_SIZE, ", got ", block_k);
    AITER_CHECK(kv_block_size == Traits::PAGE_SIZE,
                "compiled for kv_block_size=", (int)Traits::PAGE_SIZE, ", got ", kv_block_size);

    // Q/KV are read as raw fp8 (E4M3) bytes; accept fp8 or u8 byte buffers.
    AITER_CHECK(q.dtype() == AITER_DTYPE_fp8 || q.dtype() == AITER_DTYPE_u8,
                "q must be fp8 (E4M3) or u8 bytes");
    AITER_CHECK(kv_cache.dtype() == AITER_DTYPE_fp8 || kv_cache.dtype() == AITER_DTYPE_u8,
                "kv_cache must be fp8 (E4M3) or u8 bytes");
    AITER_CHECK(q_scale.dtype() == AITER_DTYPE_u8 || q_scale.dtype() == AITER_DTYPE_fp8_e8m0,
                "q_scale must be u8 / fp8_e8m0 (E8M0 bytes)");
    AITER_CHECK(kv_scale.dtype() == AITER_DTYPE_u8 || kv_scale.dtype() == AITER_DTYPE_fp8_e8m0,
                "kv_scale must be u8 / fp8_e8m0 (E8M0 bytes)");
    AITER_CHECK(weights.dtype() == AITER_DTYPE_bf16, "weights must be bf16");
    AITER_CHECK(block_tables.dtype() == AITER_DTYPE_i32, "block_tables must be int32");
    AITER_CHECK(cta_info.dtype() == AITER_DTYPE_i32, "cta_info must be int32");
    AITER_CHECK(out.dtype() == AITER_DTYPE_fp32, "out must be fp32");

    AITER_CHECK(q.stride(2) == 1 && weights.stride(1) == 1 && out.stride(1) == 1,
                "q / weights / out must be contiguous along their last dim");
    AITER_CHECK(cta_info.is_contiguous(), "cta_info must be contiguous");

    AITER_CHECK(weights.size(0) == T && weights.size(1) == H, "weights shape must be [T, H]");
    AITER_CHECK(out.size(0) == T, "out row count must equal T");

    if(n_ctas <= 0 || T == 0)
        return;

    // ---- Build kernel args -----------------------------------------------
    opus_mqa_logits_kargs kargs{};
    kargs.ptr_q             = q.data_ptr();
    kargs.ptr_q_scale       = q_scale.data_ptr();
    kargs.ptr_kv            = kv_cache.data_ptr();
    kargs.ptr_kv_scale      = kv_scale.data_ptr();
    kargs.ptr_block_tables  = reinterpret_cast<const int*>(block_tables.data_ptr());
    kargs.ptr_weights       = weights.data_ptr();
    kargs.ptr_cta_info      = reinterpret_cast<const int*>(cta_info.data_ptr());
    kargs.ptr_out           = reinterpret_cast<float*>(out.data_ptr());
    kargs.max_seq_len       = max_seq_len;
    kargs.stride_out_row    = static_cast<int>(out.stride(0));
    kargs.weight_scale      = weight_scale;
    kargs.block_k           = block_k;
    kargs.kv_block_size     = kv_block_size;
    kargs.max_blocks_per_seq = static_cast<int>(block_tables.size(1));

    // ---- Launch ----------------------------------------------------------
    HipDeviceGuard guard(q.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    dim3 grid(n_ctas);
    dim3 block(Traits::BLOCK_SIZE);
    opus_logits::pa_mqa_logits_mxfp8_kernel<Traits><<<grid, block, 0, stream>>>(kargs);
    HIP_CALL_LAUNCH(hipGetLastError());
}
