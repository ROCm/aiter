// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// MXFP4 paged MQA logits (gfx950) — the schedule builder and the launch it feeds.
// Thin wrapper over the device kernel in `pa_mqa_logits_mxfp4_opus.h`, which documents the
// input ABI. Prefill and decode are ONE launch over a per-row table. Cudagraph-safe: the
// builder syncs with nothing and reads nothing back, and the grid is a caller-held constant.

#define PA_MQA_LOGITS_MXFP4_IMPL
#include "pa_mqa_logits_mxfp4_opus.h"

#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "aiter_tensor.h"

// Two compiled configs dispatched by block_k: 256 -> 4-wave, 64 -> 1-wave. Same compute
// core, ABI and scale layout; they differ only in KV split granularity and CTA width.
using mqa_logits_fp4_traits_4wave = opus_mqa_logits_fp4_traits<256, 64, 128, 64, 4>;
using mqa_logits_fp4_traits_1wave = opus_mqa_logits_fp4_traits<64, 64, 128, 64, 1>;


// Shape validation for the launch.
//
// Note `Q_ROW_BYTES` is the WHOLE query row (H * D/2), not the per-head count, so q.size(2)
// is compared against HEAD_DIM * ELEM_BITS / 8 instead.
//
// The scale arrays get a byte-count check because they are the only buffers with a
// tile-specific layout, and passing a differently-permuted one is SILENT: every fp4 scale
// layout has the same byte count. The check catches a wrong array, not a wrong permutation.
template<class Traits>
static void pa_mqa_logits_mxfp4_check_shapes(aiter_tensor_t& q,
                                             aiter_tensor_t& q_scale,
                                             aiter_tensor_t& kv_cache,
                                             aiter_tensor_t& kv_scale,
                                             aiter_tensor_t& block_tables,
                                             aiter_tensor_t& weights,
                                             aiter_tensor_t& out,
                                             int block_k,
                                             int kv_block_size,
                                             int max_seq_len)
{
    AITER_CHECK(q.dim() == 3, "q must be 3-D [T, H, D/2], got ndim=", q.dim());
    AITER_CHECK(weights.dim() == 2, "weights must be 2-D [T, H], got ndim=", weights.dim());
    AITER_CHECK(block_tables.dim() == 2, "block_tables must be 2-D [batch, max_blocks_per_seq]");
    AITER_CHECK(out.dim() == 2, "out must be 2-D [T, max_seq_len], got ndim=", out.dim());
    AITER_CHECK(weights.size(0) >= q.size(0),
                "weights is per query row; need at least ", q.size(0), " rows, got ",
                weights.size(0));
    // The kernel bounds its store by the WINDOW, not by max_seq_len -- `max_seq_len` is
    // carried in kargs and never read, only `stride_out_row` is. So these two are the only
    // place an undersized `out` can be caught at all; a `local_ends` entry past out.size(1)
    // is the caller's contract (see the wrapper docstring).
    AITER_CHECK(out.size(0) >= q.size(0),
                "out is [T, max_seq_len]; need at least ", q.size(0), " rows, got ",
                out.size(0));
    AITER_CHECK(out.size(1) >= max_seq_len,
                "out is [T, max_seq_len]; need at least ", max_seq_len, " columns, got ",
                out.size(1));

    constexpr int HEAD_BYTES = Traits::HEAD_DIM * Traits::ELEM_BITS / 8;  // 64: D/2 per head
    const int H       = static_cast<int>(q.size(1));
    const int D_BYTES = static_cast<int>(q.size(2));
    AITER_CHECK(H == Traits::N_HEADS, "compiled for H=", (int)Traits::N_HEADS, ", got H=", H);
    AITER_CHECK(D_BYTES == HEAD_BYTES,
                "q last dim is D/2 packed bytes; compiled for ", HEAD_BYTES,
                " (D=", (int)Traits::HEAD_DIM, "), got ", D_BYTES);
    static_assert(Traits::N_HEADS * HEAD_BYTES == Traits::Q_ROW_BYTES,
                  "the kernel strides q rows by Q_ROW_BYTES; it must equal H * D/2");
    AITER_CHECK(block_k == Traits::KV_TILE_SIZE,
                "compiled for block_k=", (int)Traits::KV_TILE_SIZE, ", got ", block_k);
    AITER_CHECK(kv_block_size == Traits::PAGE_SIZE,
                "compiled for kv_block_size=", (int)Traits::PAGE_SIZE, ", got ", kv_block_size);

    AITER_CHECK(q.dtype() == AITER_DTYPE_fp4x2 || q.dtype() == AITER_DTYPE_u8,
                "q must be fp4x2 (E2M1, 2/byte) or u8 bytes");
    AITER_CHECK(kv_cache.dtype() == AITER_DTYPE_fp4x2 || kv_cache.dtype() == AITER_DTYPE_u8,
                "kv_cache must be fp4x2 (E2M1, 2/byte) or u8 bytes");
    AITER_CHECK(q_scale.dtype() == AITER_DTYPE_u8 && kv_scale.dtype() == AITER_DTYPE_u8,
                "q_scale / kv_scale are E8M0 bytes and must be u8");
    AITER_CHECK(weights.dtype() == AITER_DTYPE_bf16, "weights must be bf16");
    AITER_CHECK(block_tables.dtype() == AITER_DTYPE_i32, "block_tables must be int32");
    AITER_CHECK(out.dtype() == AITER_DTYPE_fp32, "out must be fp32");

    // Every input is strided by a COMPILE-TIME constant in the kernel -- q by Q_ROW_BYTES,
    // weights by W_ROW_ELEMS, q_scale / kv_scale / kv_cache by their row or page strides,
    // block_tables by max_blocks_per_seq -- and no runtime stride is ever read. A padded or
    // permuted input is therefore SILENTLY wrong, so require the layout the kernel assumes.
    // `out` is the one exception: its row stride is passed through as stride_out_row, so it
    // only needs its last dim packed.
    AITER_CHECK(q.is_contiguous() && q_scale.is_contiguous() && kv_cache.is_contiguous() &&
                    kv_scale.is_contiguous() && block_tables.is_contiguous() &&
                    weights.is_contiguous(),
                "q / q_scale / kv_cache / kv_scale / block_tables / weights must be contiguous");
    AITER_CHECK(out.stride(1) == 1, "out must be contiguous along its last dim");

    AITER_CHECK(q_scale.numel() == (int64_t)q.size(0) * Traits::QS_ROW_BYTES,
                "q_scale must hold ", (int)Traits::QS_ROW_BYTES, " bytes per query row "
                "([T, K_CHUNKS, MFMA_N, QS_BYTES]), got numel=", q_scale.numel(),
                " for T=", q.size(0));
    AITER_CHECK(kv_cache.numel() % Traits::stride_kv_block == 0,
                "kv_cache must be a whole number of ", (int)Traits::stride_kv_block,
                "-byte pages, got numel=", kv_cache.numel());
    const int64_t num_blocks = kv_cache.numel() / Traits::stride_kv_block;
    AITER_CHECK(kv_scale.numel() == num_blocks * Traits::stride_kvs_block,
                "kv_scale must hold ", (int)Traits::stride_kvs_block, " bytes per page "
                "([num_blocks, K_CHUNKS, MFMA_N, KVS_BYTES]), got numel=", kv_scale.numel(),
                " for num_blocks=", num_blocks);
}

// ── SCHED Table: 1D grid over a per-row schedule, for BOTH entry points ───────
// `cta_info` is reused in place, so a captured graph replays from one address. The builder
// writes every slot, so a previous forward's schedule cannot leak into this one.
template<class Traits>
static void pa_mqa_logits_mxfp4_launch_sched(aiter_tensor_t& q,
                          aiter_tensor_t& q_scale,
                          aiter_tensor_t& kv_cache,
                          aiter_tensor_t& kv_scale,
                          aiter_tensor_t& block_tables,
                          aiter_tensor_t& weights,
                          aiter_tensor_t& cta_info,
                          aiter_tensor_t& out,
                          int num_ctas,
                          float weight_scale,
                          int block_k,
                          int kv_block_size,
                          int max_seq_len)
{
    pa_mqa_logits_mxfp4_check_shapes<Traits>(q, q_scale, kv_cache, kv_scale, block_tables,
                                             weights, out, block_k, kv_block_size, max_seq_len);
    // No `local_ends` here, and that is not an oversight: each record carries its own row's
    // window, so the array the SCHEDULE was built from is not an input to the launch.
    AITER_CHECK(cta_info.dtype() == AITER_DTYPE_i32 && cta_info.is_contiguous(),
                "cta_info must be contiguous int32");
    AITER_CHECK(num_ctas > 0, "num_ctas must be >= 1, got ", num_ctas);
    AITER_CHECK(cta_info.numel() >= (int64_t)num_ctas * 8,
                "cta_info holds 8 int32 per CTA slot; need ", (int64_t)num_ctas * 8,
                " for num_ctas=", num_ctas, ", got numel=", cta_info.numel());
    AITER_CHECK(num_ctas <= 2147483647, "num_ctas exceeds the grid limit");

    opus_mqa_logits_kargs kargs{};
    kargs.ptr_q             = q.data_ptr();
    kargs.ptr_q_scale       = q_scale.data_ptr();
    kargs.ptr_kv            = kv_cache.data_ptr();
    kargs.ptr_kv_scale      = kv_scale.data_ptr();
    kargs.ptr_block_tables  = reinterpret_cast<const int*>(block_tables.data_ptr());
    kargs.ptr_weights       = weights.data_ptr();
    kargs.ptr_out           = reinterpret_cast<float*>(out.data_ptr());
    kargs.ptr_cta_info      =
        reinterpret_cast<const opus_mqa_cta_record*>(cta_info.data_ptr());
    kargs.num_ctas          = num_ctas;
    kargs.num_rows          = static_cast<int>(q.size(0));
    kargs.max_seq_len       = max_seq_len;
    kargs.stride_out_row    = static_cast<int>(out.stride(0));
    kargs.weight_scale      = weight_scale;
    kargs.block_k           = block_k;
    kargs.kv_block_size     = kv_block_size;
    kargs.max_blocks_per_seq = static_cast<int>(block_tables.size(1));

    HipDeviceGuard guard(q.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    // No XCD padding and no `num_batches` early-out: the grid is the table.
    opus_logits::pa_mqa_logits_mxfp4_kernel<Traits, opus_logits::mqa_logits_sched::Table>
        <<<dim3(static_cast<unsigned>(num_ctas)), dim3(Traits::BLOCK_SIZE), 0, stream>>>(kargs);
    HIP_CALL_LAUNCH(hipGetLastError());
}

// Fill `cta_info`. No host<->device sync and nothing read back, so it is capture-safe.
void pa_mqa_logits_mxfp4_build_sched(aiter_tensor_t& local_starts,
                          aiter_tensor_t& local_ends,
                          aiter_tensor_t& row_to_batch,
                          aiter_tensor_t& cta_info,
                          int num_rows,
                          int num_ctas,
                          int block_k,
                          int cta_target)
{
    aiter_detail::g_aiter_can_throw = true;
    AITER_CHECK(local_ends.dtype() == AITER_DTYPE_i32 && local_ends.is_contiguous(),
                "local_ends must be contiguous int32");
    AITER_CHECK(cta_info.dtype() == AITER_DTYPE_i32 && cta_info.is_contiguous(),
                "cta_info must be contiguous int32");
    AITER_CHECK(num_rows >= 0, "num_rows must be >= 0, got ", num_rows);
    AITER_CHECK(local_ends.numel() >= num_rows,
                "local_ends is per query row; need ", num_rows, ", got ", local_ends.numel());
    // Below this a row could get no CTA at all and its logits would keep whatever the caller
    // pre-filled -- silently, since every other row would still be right.
    AITER_CHECK(num_ctas >= num_rows,
                "num_ctas (", num_ctas, ") must be >= num_rows (", num_rows,
                "); use aiter.ops.opus.pa_mqa_logits_mxfp4_sched_slots()");
    // The BUFFER is `sched_buffer_records(num_ctas)` while the GRID stays `num_ctas`: the
    // multi-workgroup emit's per-block partials sit past the slots.
    AITER_CHECK(cta_info.numel() >=
                    (int64_t)opus_logits::sched_buffer_records(num_ctas) * 8,
                "cta_info holds 8 int32 per CTA slot plus a ",
                opus_logits::SCHED_SCRATCH_RECORDS, "-record build scratch; need ",
                (int64_t)opus_logits::sched_buffer_records(num_ctas) * 8,
                ", got numel=", cta_info.numel(),
                ". Size it with aiter.ops.opus.pa_mqa_logits_mxfp4_sched_buffer_ints()");
    AITER_CHECK(cta_target >= 1, "cta_target must be >= 1, got ", cta_target);
    AITER_CHECK(block_k > 0, "block_k must be >= 1, got ", block_k);

    // Empty is legal and must not launch a builder that would index local_ends[0].
    if(num_rows == 0 && num_ctas == 0)
        return;

    const int* p_ls = local_starts.numel() > 0
                        ? reinterpret_cast<const int*>(local_starts.data_ptr())
                        : nullptr;
    if(p_ls)
        AITER_CHECK(local_starts.dtype() == AITER_DTYPE_i32 && local_starts.is_contiguous() &&
                        local_starts.numel() >= num_rows,
                    "local_starts, when given, must be contiguous int32 with one entry per row");

    const int* p_rb = row_to_batch.numel() > 0
                        ? reinterpret_cast<const int*>(row_to_batch.data_ptr())
                        : nullptr;
    if(p_rb)
        AITER_CHECK(row_to_batch.dtype() == AITER_DTYPE_i32 && row_to_batch.is_contiguous() &&
                        row_to_batch.numel() >= num_rows,
                    "row_to_batch, when given, must be contiguous int32 with one entry per row");

    HipDeviceGuard guard(local_ends.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    const int* p_le = reinterpret_cast<const int*>(local_ends.data_ptr());
    auto* p_cta     = reinterpret_cast<opus_mqa_cta_record*>(cta_info.data_ptr());

    // `sched_plan` is the policy and lives in the header, so this launcher and opus-ops'
    // standalone host cannot drift on it; only the dispatch is per-host.
    namespace ol   = opus_logits;
    const auto plan = ol::sched_plan(num_rows, num_ctas);
    if(plan.blocks == 1)
    {
        if(plan.block == ol::SCHED_BUILD_BLOCK)
            ol::mqa_logits_build_sched<ol::SCHED_BUILD_BLOCK>
                <<<dim3(1), dim3(ol::SCHED_BUILD_BLOCK), 0, stream>>>(
                    p_ls, p_le, p_rb, p_cta, num_rows, num_ctas, block_k, cta_target);
        else
            ol::mqa_logits_build_sched<ol::SCHED_BUILD_BLOCK_WIDE>
                <<<dim3(1), dim3(ol::SCHED_BUILD_BLOCK_WIDE), 0, stream>>>(
                    p_ls, p_le, p_rb, p_cta, num_rows, num_ctas, block_k, cta_target);
    }
    else
    {
        int* scratch = reinterpret_cast<int*>(p_cta + num_ctas);
        ol::mqa_logits_build_sched_emit<ol::SCHED_BUILD_BLOCK>
            <<<dim3(plan.blocks), dim3(ol::SCHED_BUILD_BLOCK), 0, stream>>>(
                p_ls, p_le, p_rb, p_cta, scratch, num_rows, num_ctas, block_k, plan.blocks);
        ol::mqa_logits_build_sched_finish<ol::SCHED_BUILD_BLOCK_WIDE>
            <<<dim3(1), dim3(ol::SCHED_BUILD_BLOCK_WIDE), 0, stream>>>(
                p_ls, p_le, p_rb, p_cta, scratch, num_rows, num_ctas, block_k, cta_target,
                plan.blocks);
    }
    HIP_CALL_LAUNCH(hipGetLastError());
}

void pa_mqa_logits_mxfp4_fwd_sched(aiter_tensor_t& q,
                          aiter_tensor_t& q_scale,
                          aiter_tensor_t& kv_cache,
                          aiter_tensor_t& kv_scale,
                          aiter_tensor_t& block_tables,
                          aiter_tensor_t& weights,
                          aiter_tensor_t& cta_info,
                          aiter_tensor_t& out,
                          int num_ctas,
                          float weight_scale,
                          int block_k,
                          int kv_block_size,
                          int max_seq_len)
{
    aiter_detail::g_aiter_can_throw = true;
    if(block_k == mqa_logits_fp4_traits_4wave::KV_TILE_SIZE) {
        pa_mqa_logits_mxfp4_launch_sched<mqa_logits_fp4_traits_4wave>(
            q, q_scale, kv_cache, kv_scale, block_tables, weights, cta_info, out, num_ctas, weight_scale, block_k, kv_block_size, max_seq_len);
    } else if(block_k == mqa_logits_fp4_traits_1wave::KV_TILE_SIZE) {
        pa_mqa_logits_mxfp4_launch_sched<mqa_logits_fp4_traits_1wave>(
            q, q_scale, kv_cache, kv_scale, block_tables, weights, cta_info, out, num_ctas, weight_scale, block_k, kv_block_size, max_seq_len);
    } else {
        AITER_CHECK(false, "block_k must be 256 (4-wave) or 64 (1-wave), got ", block_k);
    }
}
