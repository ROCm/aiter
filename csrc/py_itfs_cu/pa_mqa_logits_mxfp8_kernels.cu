// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// MXFP8 paged MQA logits (gfx950) — host launchers + prefill window build.
// Thin wrapper over the device kernel template in `pa_mqa_logits_mxfp8_opus.h`
// (single-header, IMPL-guarded). Q/KV must already be MXFP8-quantized and
// preshuffled into the kernel ABI; the per-CTA assignment is derived in-kernel
// from blockIdx (schedule-free): prefill via 1D grid + per-row window arrays,
// decode via 3D grid + inline MTP windows. Also hosts the device-side per-row
// window builder (`pa_mqa_logits_mxfp8_prefill_windows`) the prefill path consumes.

#define PA_MQA_LOGITS_MXFP8_IMPL
#include "pa_mqa_logits_mxfp8_opus.h"

#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "aiter_tensor.h"

// Two compiled configs dispatched by block_k (same compute core / ABI / preshuffled KV;
// they differ only in KV split granularity + async-Q LDS fill): 256 -> 4-wave, 64 -> 1-wave.
using mqa_logits_traits_4wave = opus_mqa_logits_traits<256, 64, 128, 64, 4>;
using mqa_logits_traits_1wave = opus_mqa_logits_traits<64, 64, 128, 64, 1, /*Q_LDS_WARPS=*/4>;

// ── Prefill (direct) launch: 1D grid (num_rows); per-CTA assignment derived
//    in-kernel from blockIdx.x + per-row window arrays (one CTA per row). ───────
template<class Traits>
static void pa_mqa_logits_mxfp8_launch_prefill(aiter_tensor_t& q,
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

    if(num_rows <= 0)
        return;

    opus_mqa_logits_kargs kargs{};
    kargs.ptr_q             = q.data_ptr();
    kargs.ptr_q_scale       = q_scale.data_ptr();
    kargs.ptr_kv            = kv_cache.data_ptr();
    kargs.ptr_kv_scale      = kv_scale.data_ptr();
    kargs.ptr_block_tables  = reinterpret_cast<const int*>(block_tables.data_ptr());
    kargs.ptr_weights       = weights.data_ptr();
    kargs.ptr_out           = reinterpret_cast<float*>(out.data_ptr());
    kargs.ptr_row_to_batch  = reinterpret_cast<const int*>(row_to_batch.data_ptr());
    kargs.ptr_local_starts  = reinterpret_cast<const int*>(local_starts.data_ptr());
    kargs.ptr_local_ends    = reinterpret_cast<const int*>(local_ends.data_ptr());
    kargs.num_rows          = num_rows;
    kargs.max_seq_len       = max_seq_len;
    kargs.stride_out_row    = static_cast<int>(out.stride(0));
    kargs.weight_scale      = weight_scale;
    kargs.block_k           = block_k;
    kargs.kv_block_size     = kv_block_size;
    kargs.max_blocks_per_seq = static_cast<int>(block_tables.size(1));

    HipDeviceGuard guard(q.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    dim3 grid(static_cast<unsigned>(num_rows));   // 1D: one CTA per query row
    dim3 block(Traits::BLOCK_SIZE);
    opus_logits::pa_mqa_logits_mxfp8_kernel<Traits, opus_logits::mqa_logits_sched::Prefill><<<grid, block, 0, stream>>>(kargs);
    HIP_CALL_LAUNCH(hipGetLastError());
}

// ── Schedule-free DECODE launch: 3D grid (batch, next_n_max, split_kv); windows inline. ──
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
                "decode launch: batch / next_n_max exceed grid.x/.y limit (65535)");

    opus_mqa_logits_kargs kargs{};
    kargs.ptr_q             = q.data_ptr();
    kargs.ptr_q_scale       = q_scale.data_ptr();
    kargs.ptr_kv            = kv_cache.data_ptr();
    kargs.ptr_kv_scale      = kv_scale.data_ptr();
    kargs.ptr_block_tables  = reinterpret_cast<const int*>(block_tables.data_ptr());
    kargs.ptr_weights       = weights.data_ptr();
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

    // grid (batch, next_n_max, split_kv): batch on the fast x-axis clusters a batch's
    // split CTAs (shared KV) -> best L2 locality (min-of-N sweep winner; the only mapping kept).
    dim3 grid(static_cast<unsigned>(batch),
              static_cast<unsigned>(next_n_max),
              static_cast<unsigned>(split_kv));
    dim3 block(Traits::BLOCK_SIZE);
    opus_logits::pa_mqa_logits_mxfp8_kernel<Traits, opus_logits::mqa_logits_sched::Decode><<<grid, block, 0, stream>>>(kargs);
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
                          float weight_scale,
                          int block_k,
                          int kv_block_size,
                          int max_seq_len)
{
    if(block_k == mqa_logits_traits_4wave::KV_TILE_SIZE) {
        pa_mqa_logits_mxfp8_launch_decode<mqa_logits_traits_4wave>(
            q, q_scale, kv_cache, kv_scale, block_tables, weights, cu_seq_q, context_lens, out,
            batch, next_n_max, split_kv, weight_scale, block_k, kv_block_size, max_seq_len);
    } else if(block_k == mqa_logits_traits_1wave::KV_TILE_SIZE) {
        pa_mqa_logits_mxfp8_launch_decode<mqa_logits_traits_1wave>(
            q, q_scale, kv_cache, kv_scale, block_tables, weights, cu_seq_q, context_lens, out,
            batch, next_n_max, split_kv, weight_scale, block_k, kv_block_size, max_seq_len);
    } else {
        AITER_CHECK(false, "block_k must be 256 (4-wave) or 64 (1-wave), got ", block_k);
    }
}

void pa_mqa_logits_mxfp8_fwd_prefill(aiter_tensor_t& q,
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
                          float weight_scale,
                          int block_k,
                          int kv_block_size,
                          int max_seq_len)
{
    if(block_k == mqa_logits_traits_4wave::KV_TILE_SIZE) {
        pa_mqa_logits_mxfp8_launch_prefill<mqa_logits_traits_4wave>(
            q, q_scale, kv_cache, kv_scale, block_tables, weights,
            row_to_batch, local_starts, local_ends, out,
            num_rows, weight_scale, block_k, kv_block_size, max_seq_len);
    } else if(block_k == mqa_logits_traits_1wave::KV_TILE_SIZE) {
        pa_mqa_logits_mxfp8_launch_prefill<mqa_logits_traits_1wave>(
            q, q_scale, kv_cache, kv_scale, block_tables, weights,
            row_to_batch, local_starts, local_ends, out,
            num_rows, weight_scale, block_k, kv_block_size, max_seq_len);
    } else {
        AITER_CHECK(false,
                    "block_k must be 256 (4-wave) or 64 (1-wave), got ", block_k);
    }
}

// ── Prefill per-row window build (device, cudagraph-safe) ───────────────────
// Builds the per-row [local_start, local_end) window arrays that the PREFILL
// launch consumes, from cu_seq_q + context_lens (MTP tail-causal; for qlen == ctx
// this is plain causal). Decode does NOT use this -- the decode kernel derives its
// window inline. All inputs/outputs are caller-allocated device buffers (no
// hipMalloc / no host<->device sync); the launch grid is the static shape (total_q).
namespace {

constexpr int WINDOW_BUILD_BLOCK = 256;

// Per-row window build. MTP tail-causal: batch b's n-th query token (n in [0, qlen))
// sees [0, context_len[b] - (qlen - 1 - n)); rows past cu[B] get an empty window.
__global__ void mqa_logits_prefill_windows_kernel(const int* __restrict__ cu,
                                                  const int* __restrict__ ctx,
                                                  int* __restrict__ row_to_batch,
                                                  int* __restrict__ local_starts,
                                                  int* __restrict__ local_ends,
                                                  int total_q, int B)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if(r >= total_q)
        return;

    // searchsorted(cu[1:], r, right=True) = count(cu[1..B] <= r).
    int lo = 0, hi = B;
    while(lo < hi)
    {
        int mid    = (lo + hi) >> 1;
        int cu_mid = cu[1 + (mid < (B - 1) ? mid : (B - 1))];
        if(cu_mid <= r)
            lo = mid + 1;
        else
            hi = mid;
    }
    const int b = (lo < (B - 1)) ? lo : (B - 1);

    const int cu_b  = cu[b];
    const int cu_b1 = cu[b + 1];
    const int ctx_b = ctx[b];
    const int n     = r - cu_b;
    const int qlen  = cu_b1 - cu_b;
    int le = ctx_b - qlen + n + 1;
    le = le > 0 ? le : 0;
    // Rows beyond the real total (cu[B]) are flat tail-padding -> empty window.
    const int real_total = cu[B];
    if(r >= real_total)
        le = 0;

    row_to_batch[r] = b;
    local_starts[r] = 0;
    local_ends[r]   = le;
}

} // namespace

void pa_mqa_logits_mxfp8_prefill_windows(aiter_tensor_t& cu_seq_q,
                                      aiter_tensor_t& context_lens,
                                      aiter_tensor_t& row_to_batch,
                                      aiter_tensor_t& local_starts,
                                      aiter_tensor_t& local_ends,
                                      int total_q)
{
    const int B = static_cast<int>(context_lens.size(0));
    AITER_CHECK(cu_seq_q.dtype() == AITER_DTYPE_i32 && context_lens.dtype() == AITER_DTYPE_i32,
                "cu_seq_q / context_lens must be int32");
    AITER_CHECK(cu_seq_q.size(0) == B + 1, "cu_seq_q must have length B+1");

    if(total_q <= 0 || B <= 0)
        return;

    HipDeviceGuard guard(context_lens.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    const int grid = (total_q + WINDOW_BUILD_BLOCK - 1) / WINDOW_BUILD_BLOCK;
    mqa_logits_prefill_windows_kernel<<<grid, WINDOW_BUILD_BLOCK, 0, stream>>>(
        reinterpret_cast<const int*>(cu_seq_q.data_ptr()),
        reinterpret_cast<const int*>(context_lens.data_ptr()),
        reinterpret_cast<int*>(row_to_batch.data_ptr()),
        reinterpret_cast<int*>(local_starts.data_ptr()),
        reinterpret_cast<int*>(local_ends.data_ptr()),
        total_q, B);
    HIP_CALL_LAUNCH(hipGetLastError());
}
