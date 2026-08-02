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

// Two compiled configurations, dispatched by block_k at launch time:
//   block_k=256 -> 4-wave variant (BLOCK=256, PAGE=64): use for DECODE (varqlen).
//   block_k=64  -> 1-wave variant (BLOCK=64,  PAGE=64, smaller kv split): use for PREFILL.
// Both share the same compute core, ABI and preshuffled KV cache; they differ only in the KV
// split granularity (block_k) and the async-Q LDS fill (see opus_mqa_logits_traits::Q_LDS_WARPS).
using mqa_logits_traits_4wave = opus_mqa_logits_traits<256, 64, 128, 64, 4>;
using mqa_logits_traits_1wave = opus_mqa_logits_traits<64, 64, 128, 64, 1, /*Q_LDS_WARPS=*/4>;

// Validation + launch for one compiled Traits. block_k / kv_block_size must match the Traits.
template<class Traits>
static void pa_mqa_logits_mxfp8_launch(aiter_tensor_t& q,
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

// ── Schedule-free (direct) launch: no cta_info; per-CTA assignment derived
//    in-kernel from blockIdx + per-row window arrays + split_kv. ──────────────
template<class Traits>
static void pa_mqa_logits_mxfp8_launch_direct(aiter_tensor_t& q,
                          aiter_tensor_t& q_scale,
                          aiter_tensor_t& kv_cache,
                          aiter_tensor_t& kv_scale,
                          aiter_tensor_t& block_tables,
                          aiter_tensor_t& weights,
                          aiter_tensor_t& row_to_batch,
                          aiter_tensor_t& local_starts,
                          aiter_tensor_t& local_ends,
                          aiter_tensor_t& out,
                          int num_rows,
                          int split_kv,
                          float weight_scale,
                          int block_k,
                          int kv_block_size,
                          int max_seq_len)
{
    AITER_CHECK(q.dim() == 3, "q must be 3-D [T, H, D], got ndim=", q.dim());
    AITER_CHECK(weights.dim() == 2, "weights must be 2-D [T, H], got ndim=", weights.dim());
    AITER_CHECK(block_tables.dim() == 2, "block_tables must be 2-D [batch, max_blocks_per_seq]");
    AITER_CHECK(out.dim() == 2, "out must be 2-D [T, max_seq_len], got ndim=", out.dim());

    const int H = static_cast<int>(q.size(1));
    const int D = static_cast<int>(q.size(2));
    AITER_CHECK(H == Traits::N_HEADS, "compiled for H=", (int)Traits::N_HEADS, ", got H=", H);
    AITER_CHECK(D == Traits::HEAD_DIM, "compiled for D=", (int)Traits::HEAD_DIM, ", got D=", D);
    AITER_CHECK(block_k == Traits::KV_TILE_SIZE,
                "compiled for block_k=", (int)Traits::KV_TILE_SIZE, ", got ", block_k);
    AITER_CHECK(kv_block_size == Traits::PAGE_SIZE,
                "compiled for kv_block_size=", (int)Traits::PAGE_SIZE, ", got ", kv_block_size);

    AITER_CHECK(q.dtype() == AITER_DTYPE_fp8 || q.dtype() == AITER_DTYPE_u8,
                "q must be fp8 (E4M3) or u8 bytes");
    AITER_CHECK(kv_cache.dtype() == AITER_DTYPE_fp8 || kv_cache.dtype() == AITER_DTYPE_u8,
                "kv_cache must be fp8 (E4M3) or u8 bytes");
    AITER_CHECK(weights.dtype() == AITER_DTYPE_bf16, "weights must be bf16");
    AITER_CHECK(block_tables.dtype() == AITER_DTYPE_i32, "block_tables must be int32");
    AITER_CHECK(out.dtype() == AITER_DTYPE_fp32, "out must be fp32");
    AITER_CHECK(row_to_batch.dtype() == AITER_DTYPE_i32 && local_starts.dtype() == AITER_DTYPE_i32 &&
                    local_ends.dtype() == AITER_DTYPE_i32,
                "row_to_batch / local_starts / local_ends must be int32");
    AITER_CHECK(q.stride(2) == 1 && weights.stride(1) == 1 && out.stride(1) == 1,
                "q / weights / out must be contiguous along their last dim");

    if(num_rows <= 0 || split_kv <= 0)
        return;
    // 2D grid (split_kv, num_rows): the kernel reads split=blockIdx.x, row=blockIdx.y,
    // avoiding the pid/split_kv div+mod. grid.y is capped at 65535 by HW.
    AITER_CHECK(num_rows <= 65535,
                "direct launch: num_rows=", num_rows, " exceeds grid.y limit (65535)");

    opus_mqa_logits_kargs kargs{};
    kargs.ptr_q             = q.data_ptr();
    kargs.ptr_q_scale       = q_scale.data_ptr();
    kargs.ptr_kv            = kv_cache.data_ptr();
    kargs.ptr_kv_scale      = kv_scale.data_ptr();
    kargs.ptr_block_tables  = reinterpret_cast<const int*>(block_tables.data_ptr());
    kargs.ptr_weights       = weights.data_ptr();
    kargs.ptr_cta_info      = nullptr;
    kargs.ptr_out           = reinterpret_cast<float*>(out.data_ptr());
    kargs.ptr_row_to_batch  = reinterpret_cast<const int*>(row_to_batch.data_ptr());
    kargs.ptr_local_starts  = reinterpret_cast<const int*>(local_starts.data_ptr());
    kargs.ptr_local_ends    = reinterpret_cast<const int*>(local_ends.data_ptr());
    kargs.split_kv          = split_kv;
    kargs.num_rows          = num_rows;
    kargs.max_seq_len       = max_seq_len;
    kargs.stride_out_row    = static_cast<int>(out.stride(0));
    kargs.weight_scale      = weight_scale;
    kargs.block_k           = block_k;
    kargs.kv_block_size     = kv_block_size;
    kargs.max_blocks_per_seq = static_cast<int>(block_tables.size(1));

    HipDeviceGuard guard(q.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    dim3 grid(static_cast<unsigned>(split_kv), static_cast<unsigned>(num_rows));
    dim3 block(Traits::BLOCK_SIZE);
    opus_logits::pa_mqa_logits_mxfp8_kernel<Traits, /*SCHED=prefill-direct*/1><<<grid, block, 0, stream>>>(kargs);
    HIP_CALL_LAUNCH(hipGetLastError());
}

// ── Schedule-free DECODE launch: 3D grid (split_kv, next_n_max, batch); windows inline. ──
template<class Traits>
static void pa_mqa_logits_mxfp8_launch_decode(aiter_tensor_t& q,
                          aiter_tensor_t& q_scale,
                          aiter_tensor_t& kv_cache,
                          aiter_tensor_t& kv_scale,
                          aiter_tensor_t& block_tables,
                          aiter_tensor_t& weights,
                          aiter_tensor_t& cu_seq_q,
                          aiter_tensor_t& context_lens,
                          aiter_tensor_t& out,
                          int batch,
                          int next_n_max,
                          int split_kv,
                          int axis,
                          float weight_scale,
                          int block_k,
                          int kv_block_size,
                          int max_seq_len)
{
    AITER_CHECK(q.dim() == 3, "q must be 3-D [total_q, H, D], got ndim=", q.dim());
    AITER_CHECK(weights.dim() == 2, "weights must be 2-D [total_q, H]");
    AITER_CHECK(block_tables.dim() == 2, "block_tables must be 2-D [batch, max_blocks_per_seq]");
    AITER_CHECK(out.dim() == 2, "out must be 2-D [total_q, max_seq_len]");
    const int H = static_cast<int>(q.size(1));
    const int D = static_cast<int>(q.size(2));
    AITER_CHECK(H == Traits::N_HEADS, "compiled for H=", (int)Traits::N_HEADS, ", got H=", H);
    AITER_CHECK(D == Traits::HEAD_DIM, "compiled for D=", (int)Traits::HEAD_DIM, ", got D=", D);
    AITER_CHECK(block_k == Traits::KV_TILE_SIZE,
                "compiled for block_k=", (int)Traits::KV_TILE_SIZE, ", got ", block_k);
    AITER_CHECK(kv_block_size == Traits::PAGE_SIZE,
                "compiled for kv_block_size=", (int)Traits::PAGE_SIZE, ", got ", kv_block_size);
    AITER_CHECK(q.dtype() == AITER_DTYPE_fp8 || q.dtype() == AITER_DTYPE_u8, "q must be fp8/u8");
    AITER_CHECK(kv_cache.dtype() == AITER_DTYPE_fp8 || kv_cache.dtype() == AITER_DTYPE_u8, "kv_cache must be fp8/u8");
    AITER_CHECK(weights.dtype() == AITER_DTYPE_bf16, "weights must be bf16");
    AITER_CHECK(block_tables.dtype() == AITER_DTYPE_i32, "block_tables must be int32");
    AITER_CHECK(out.dtype() == AITER_DTYPE_fp32, "out must be fp32");
    AITER_CHECK(cu_seq_q.dtype() == AITER_DTYPE_i32 && context_lens.dtype() == AITER_DTYPE_i32,
                "cu_seq_q / context_lens must be int32");
    AITER_CHECK(cu_seq_q.size(0) == batch + 1, "cu_seq_q must have length batch+1");
    AITER_CHECK(q.stride(2) == 1 && weights.stride(1) == 1 && out.stride(1) == 1,
                "q / weights / out must be contiguous along their last dim");

    if(batch <= 0 || next_n_max <= 0 || split_kv <= 0)
        return;
    AITER_CHECK(batch <= 65535 && next_n_max <= 65535,
                "decode launch: batch / next_n_max exceed grid.z/.y limit (65535)");

    opus_mqa_logits_kargs kargs{};
    kargs.ptr_q             = q.data_ptr();
    kargs.ptr_q_scale       = q_scale.data_ptr();
    kargs.ptr_kv            = kv_cache.data_ptr();
    kargs.ptr_kv_scale      = kv_scale.data_ptr();
    kargs.ptr_block_tables  = reinterpret_cast<const int*>(block_tables.data_ptr());
    kargs.ptr_weights       = weights.data_ptr();
    kargs.ptr_cta_info      = nullptr;
    kargs.ptr_out           = reinterpret_cast<float*>(out.data_ptr());
    kargs.ptr_cu_seq_q      = reinterpret_cast<const int*>(cu_seq_q.data_ptr());
    kargs.ptr_context_lens  = reinterpret_cast<const int*>(context_lens.data_ptr());
    kargs.split_kv          = split_kv;
    kargs.max_seq_len       = max_seq_len;
    kargs.stride_out_row    = static_cast<int>(out.stride(0));
    kargs.weight_scale      = weight_scale;
    kargs.block_k           = block_k;
    kargs.kv_block_size     = kv_block_size;
    kargs.max_blocks_per_seq = static_cast<int>(block_tables.size(1));

    HipDeviceGuard guard(q.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    // AXIS chooses the blockIdx -> (split, n, batch) mapping; grid dims are ordered
    // so blockIdx.x = 1st dim. batch on the slowest (z) axis clusters a batch's CTAs
    // (shared KV) unless AXIS moves it. (L2-locality sweep knob.)
    const unsigned S = static_cast<unsigned>(split_kv);
    const unsigned N = static_cast<unsigned>(next_n_max);
    const unsigned B = static_cast<unsigned>(batch);
    dim3 block(Traits::BLOCK_SIZE);
    switch(axis) {
    case 0: { dim3 g(S, N, B); opus_logits::pa_mqa_logits_mxfp8_kernel<Traits, 2, 0><<<g, block, 0, stream>>>(kargs); break; }
    case 1: { dim3 g(N, S, B); opus_logits::pa_mqa_logits_mxfp8_kernel<Traits, 2, 1><<<g, block, 0, stream>>>(kargs); break; }
    case 2: { dim3 g(S, B, N); opus_logits::pa_mqa_logits_mxfp8_kernel<Traits, 2, 2><<<g, block, 0, stream>>>(kargs); break; }
    default:{ dim3 g(B, N, S); opus_logits::pa_mqa_logits_mxfp8_kernel<Traits, 2, 3><<<g, block, 0, stream>>>(kargs); break; }
    }
    HIP_CALL_LAUNCH(hipGetLastError());
}

void pa_mqa_logits_mxfp8_fwd_decode(aiter_tensor_t& q,
                          aiter_tensor_t& q_scale,
                          aiter_tensor_t& kv_cache,
                          aiter_tensor_t& kv_scale,
                          aiter_tensor_t& block_tables,
                          aiter_tensor_t& weights,
                          aiter_tensor_t& cu_seq_q,
                          aiter_tensor_t& context_lens,
                          aiter_tensor_t& out,
                          int batch,
                          int next_n_max,
                          int split_kv,
                          int axis,
                          float weight_scale,
                          int block_k,
                          int kv_block_size,
                          int max_seq_len)
{
    if(block_k == mqa_logits_traits_4wave::KV_TILE_SIZE) {
        pa_mqa_logits_mxfp8_launch_decode<mqa_logits_traits_4wave>(
            q, q_scale, kv_cache, kv_scale, block_tables, weights, cu_seq_q, context_lens, out,
            batch, next_n_max, split_kv, axis, weight_scale, block_k, kv_block_size, max_seq_len);
    } else if(block_k == mqa_logits_traits_1wave::KV_TILE_SIZE) {
        pa_mqa_logits_mxfp8_launch_decode<mqa_logits_traits_1wave>(
            q, q_scale, kv_cache, kv_scale, block_tables, weights, cu_seq_q, context_lens, out,
            batch, next_n_max, split_kv, axis, weight_scale, block_k, kv_block_size, max_seq_len);
    } else {
        AITER_CHECK(false, "block_k must be 256 (4-wave) or 64 (1-wave), got ", block_k);
    }
}

void pa_mqa_logits_mxfp8_fwd_direct(aiter_tensor_t& q,
                          aiter_tensor_t& q_scale,
                          aiter_tensor_t& kv_cache,
                          aiter_tensor_t& kv_scale,
                          aiter_tensor_t& block_tables,
                          aiter_tensor_t& weights,
                          aiter_tensor_t& row_to_batch,
                          aiter_tensor_t& local_starts,
                          aiter_tensor_t& local_ends,
                          aiter_tensor_t& out,
                          int num_rows,
                          int split_kv,
                          float weight_scale,
                          int block_k,
                          int kv_block_size,
                          int max_seq_len)
{
    if(block_k == mqa_logits_traits_4wave::KV_TILE_SIZE) {
        pa_mqa_logits_mxfp8_launch_direct<mqa_logits_traits_4wave>(
            q, q_scale, kv_cache, kv_scale, block_tables, weights,
            row_to_batch, local_starts, local_ends, out,
            num_rows, split_kv, weight_scale, block_k, kv_block_size, max_seq_len);
    } else if(block_k == mqa_logits_traits_1wave::KV_TILE_SIZE) {
        pa_mqa_logits_mxfp8_launch_direct<mqa_logits_traits_1wave>(
            q, q_scale, kv_cache, kv_scale, block_tables, weights,
            row_to_batch, local_starts, local_ends, out,
            num_rows, split_kv, weight_scale, block_k, kv_block_size, max_seq_len);
    } else {
        AITER_CHECK(false,
                    "block_k must be 256 (4-wave) or 64 (1-wave), got ", block_k);
    }
}

// Public entry: dispatch to the compiled variant by block_k (256 -> 4-wave decode,
// 64 -> 1-wave prefill). kv_block_size (PAGE) is 64 for both.
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
    if(block_k == mqa_logits_traits_4wave::KV_TILE_SIZE) {
        pa_mqa_logits_mxfp8_launch<mqa_logits_traits_4wave>(
            q, q_scale, kv_cache, kv_scale, block_tables, weights, cta_info, out,
            n_ctas, weight_scale, block_k, kv_block_size, max_seq_len);
    } else if(block_k == mqa_logits_traits_1wave::KV_TILE_SIZE) {
        pa_mqa_logits_mxfp8_launch<mqa_logits_traits_1wave>(
            q, q_scale, kv_cache, kv_scale, block_tables, weights, cta_info, out,
            n_ctas, weight_scale, block_k, kv_block_size, max_seq_len);
    } else {
        AITER_CHECK(false,
                    "block_k must be 256 (4-wave decode) or 64 (1-wave prefill), got ", block_k);
    }
}
