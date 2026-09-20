// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// MXFP4 paged MQA logits (gfx1250) -- host launcher plus the two device-side builders the
// caller needs. The input ABI and the three conditions a qshare caller owes are documented in
// pa_mqa_logits_mxfp4_gfx1250.h.

#define PA_MQA_LOGITS_MXFP4_GFX1250_IMPL
#include "pa_mqa_logits_mxfp4_gfx1250_opus.h"

#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "aiter_tensor.h"

// The single compiled config: 4 query rows per CTA (4 waves of 32), KV tile 64 = 1 page,
// D = 128, H = 64, page 64.
using mqa_logits_fp4_gfx1250_traits = logits_fp4_qshare_traits_4q;

// Every input is strided by a COMPILE-TIME constant in the kernel and no runtime stride is ever
// read, so a padded or permuted input is silently wrong rather than rejected -- hence the
// layout is required, not adapted to. `out` is the one exception: its row stride is passed
// through as stride_out_row.
//
// The two scale arrays get a byte-count check because they are the only buffers whose layout is
// specific to this matrix instruction. It catches a wrong ARRAY, never a wrong PERMUTATION:
// every fp4 scale layout has the same byte count, gfx950's included.
template <class Traits>
static void pa_mqa_logits_mxfp4_gfx1250_check_shapes(aiter_tensor_t& q,
                                                     aiter_tensor_t& q_scale,
                                                     aiter_tensor_t& kv_cache,
                                                     aiter_tensor_t& kv_scale,
                                                     aiter_tensor_t& block_tables,
                                                     aiter_tensor_t& weights,
                                                     aiter_tensor_t& out,
                                                     int kv_block_size,
                                                     int max_seq_len)
{
    AITER_CHECK(q.dim() == 3, "q must be 3-D [T, H, D/2], got ndim=", q.dim());
    AITER_CHECK(weights.dim() == 2, "weights must be 2-D [T, H], got ndim=", weights.dim());
    AITER_CHECK(block_tables.dim() == 2, "block_tables must be 2-D [batch, max_blocks_per_seq]");
    AITER_CHECK(out.dim() == 2, "out must be 2-D [T, max_seq_len], got ndim=", out.dim());
    AITER_CHECK(weights.size(0) >= q.size(0),
                "weights is per query row; need at least ",
                q.size(0),
                " rows, got ",
                weights.size(0));
    // The kernel bounds its store by the WINDOW, not by max_seq_len (header, condition 3), so
    // these two are the only place an undersized `out` can be caught at all.
    AITER_CHECK(out.size(0) >= q.size(0),
                "out is [T, max_seq_len]; need at least ",
                q.size(0),
                " rows, got ",
                out.size(0));
    AITER_CHECK(out.size(1) >= max_seq_len,
                "out is [T, max_seq_len]; need at least ",
                max_seq_len,
                " columns, got ",
                out.size(1));

    constexpr int HEAD_BYTES = Traits::HEAD_DIM * Traits::ELEM_BITS / 8; // 64: D/2 per head
    const int H              = static_cast<int>(q.size(1));
    const int D_BYTES        = static_cast<int>(q.size(2));
    AITER_CHECK(H == Traits::N_HEADS, "compiled for H=", (int)Traits::N_HEADS, ", got H=", H);
    AITER_CHECK(weights.size(1) == Traits::W_ROW_ELEMS,
                "weights is [T, H] and the kernel strides it by a compile-time H=",
                (int)Traits::W_ROW_ELEMS, ": got weights.size(1)=", weights.size(1));
    AITER_CHECK(D_BYTES == HEAD_BYTES,
                "q last dim is D/2 packed bytes; compiled for ",
                HEAD_BYTES,
                " (D=",
                (int)Traits::HEAD_DIM,
                "), got ",
                D_BYTES);
    static_assert(Traits::N_HEADS * HEAD_BYTES == Traits::Q_ROW_BYTES,
                  "the kernel strides q rows by Q_ROW_BYTES; it must equal H * D/2");
    AITER_CHECK(kv_block_size == Traits::PAGE_SIZE,
                "compiled for kv_block_size=",
                (int)Traits::PAGE_SIZE,
                ", got ",
                kv_block_size);
    // block_tables is sized in KV TILES and not in pages: a CTA rounds its window up to a whole
    // tile and reads the table at EVERY page of the last one, including where the window stops
    // inside it. `max_blocks_per_seq` is just this size and the kernel uses it as a row stride,
    // so nothing else bounds the page index. `max_seq_len` bounds every window -- it is the
    // store's own bound -- which makes this the tightest size the host can know.
    const int64_t bt_tiles = (max_seq_len + Traits::KV_TILE_SIZE - 1) / Traits::KV_TILE_SIZE;
    const int64_t bt_cols  = bt_tiles * Traits::PAGES_PER_TILE;
    AITER_CHECK(block_tables.size(1) >= bt_cols,
                "block_tables is sized in KV TILES of ", (int)Traits::KV_TILE_SIZE,
                " tokens, not pages of ", (int)Traits::PAGE_SIZE, ": max_seq_len=", max_seq_len,
                " needs ", bt_cols, " columns, got ", block_tables.size(1));

    AITER_CHECK(q.dtype() == AITER_DTYPE_fp4x2 || q.dtype() == AITER_DTYPE_u8,
                "q must be fp4x2 (E2M1, 2/byte) or u8 bytes");
    AITER_CHECK(kv_cache.dtype() == AITER_DTYPE_fp4x2 || kv_cache.dtype() == AITER_DTYPE_u8,
                "kv_cache must be fp4x2 (E2M1, 2/byte) or u8 bytes");
    AITER_CHECK(q_scale.dtype() == AITER_DTYPE_u8 && kv_scale.dtype() == AITER_DTYPE_u8,
                "q_scale / kv_scale are E8M0 bytes and must be u8");
    AITER_CHECK(weights.dtype() == AITER_DTYPE_bf16, "weights must be bf16");
    AITER_CHECK(block_tables.dtype() == AITER_DTYPE_i32, "block_tables must be int32");
    AITER_CHECK(out.dtype() == AITER_DTYPE_fp32, "out must be fp32");

    AITER_CHECK(q.is_contiguous() && q_scale.is_contiguous() && kv_cache.is_contiguous() &&
                    kv_scale.is_contiguous() && block_tables.is_contiguous() &&
                    weights.is_contiguous(),
                "q / q_scale / kv_cache / kv_scale / block_tables / weights must be contiguous");
    AITER_CHECK(out.stride(1) == 1, "out must be contiguous along its last dim");

    AITER_CHECK(q_scale.numel() == (int64_t)q.size(0) * Traits::QS_ROW_BYTES,
                "q_scale is natural [T, H, ",
                (int)Traits::SCALE_BYTES_PER_DWORD,
                "] and must hold ",
                (int)Traits::QS_ROW_BYTES,
                " bytes per query row, got numel=",
                q_scale.numel(),
                " for T=",
                q.size(0));
    AITER_CHECK(kv_cache.numel() % Traits::KV_PAGE_BYTES == 0,
                "kv_cache must be a whole number of ",
                (int)Traits::KV_PAGE_BYTES,
                "-byte pages, got numel=",
                kv_cache.numel());
    const int64_t num_blocks = kv_cache.numel() / Traits::KV_PAGE_BYTES;
    AITER_CHECK(kv_scale.numel() == num_blocks * Traits::KVS_PAGE_BYTES,
                "kv_scale is natural [num_blocks, PAGE, ",
                (int)Traits::SCALE_BYTES_PER_DWORD,
                "] and must hold ",
                (int)Traits::KVS_PAGE_BYTES,
                " bytes per page, got numel=",
                kv_scale.numel(),
                " for num_blocks=",
                num_blocks);
}

// ── the launch: 1D grid over the schedule table ───────────────────────────────
// `cta_info` is reused in place, so a captured graph replays from one address. The builder
// writes every slot, so a previous forward's schedule cannot leak into this one.
template <class Traits>
static void pa_mqa_logits_mxfp4_gfx1250_launch_sched(aiter_tensor_t& q,
                                                     aiter_tensor_t& q_scale,
                                                     aiter_tensor_t& kv_cache,
                                                     aiter_tensor_t& kv_scale,
                                                     aiter_tensor_t& block_tables,
                                                     aiter_tensor_t& weights,
                                                     aiter_tensor_t& local_starts,
                                                     aiter_tensor_t& local_ends,
                                                     aiter_tensor_t& cta_info,
                                                     aiter_tensor_t& out,
                                                     int num_rows,
                                                     int num_ctas,
                                                     float weight_scale,
                                                     int kv_block_size,
                                                     int max_seq_len)
{
    pa_mqa_logits_mxfp4_gfx1250_check_shapes<Traits>(
        q, q_scale, kv_cache, kv_scale, block_tables, weights, out, kv_block_size, max_seq_len);
    AITER_CHECK(local_ends.dtype() == AITER_DTYPE_i32 && local_ends.is_contiguous(),
                "local_ends must be contiguous int32");
    AITER_CHECK(cta_info.dtype() == AITER_DTYPE_i32 && cta_info.is_contiguous(),
                "cta_info must be contiguous int32");
    AITER_CHECK(num_rows <= q.size(0),
                "num_rows exceeds the query rows in q: ", num_rows, " > ", q.size(0));
    // The store's upper bound is read PER ROW, so this array is the only thing between a short
    // allocation and a CTA reading a neighbouring one.
    AITER_CHECK(static_cast<int64_t>(local_ends.numel()) >= num_rows,
                "local_ends is per query row; need at least ", num_rows, " entries, got ",
                local_ends.numel());
    AITER_CHECK(num_ctas > 0, "num_ctas must be >= 1, got ", num_ctas);
    AITER_CHECK(static_cast<int64_t>(cta_info.numel()) >= (int64_t)num_ctas * 8,
                "cta_info holds 8 int32 per CTA slot; need ", (int64_t)num_ctas * 8,
                " for num_ctas=", num_ctas, ", got numel=", cta_info.numel(),
                ". Let aiter.ops.opus.pa_mqa_logits_mxfp4.pa_mqa_logits_mxfp4_plan() allocate "
                "it, or hand back a previous plan's cta_info");

    const int* p_ls = nullptr;
    if(local_starts.numel() > 0)
    {
        AITER_CHECK(local_starts.dtype() == AITER_DTYPE_i32 && local_starts.is_contiguous() &&
                        static_cast<int64_t>(local_starts.numel()) >= num_rows,
                    "local_starts, when given, must be contiguous int32 with one entry per row");
        p_ls = reinterpret_cast<const int*>(local_starts.data_ptr());
    }

    if(num_rows <= 0)
        return;

    opus_mqa_logits_kargs kargs{};
    kargs.ptr_q            = q.data_ptr();
    kargs.ptr_q_scale      = q_scale.data_ptr();
    kargs.ptr_kv           = kv_cache.data_ptr();
    kargs.ptr_kv_scale     = kv_scale.data_ptr();
    kargs.ptr_block_tables = reinterpret_cast<const int*>(block_tables.data_ptr());
    kargs.ptr_weights      = weights.data_ptr();
    kargs.ptr_out          = reinterpret_cast<float*>(out.data_ptr());
    // The per-row STORE MASK. The loop bound comes from the record's union instead.
    kargs.ptr_local_starts = p_ls;
    kargs.ptr_local_ends   = reinterpret_cast<const int*>(local_ends.data_ptr());
    kargs.ptr_cta_info = reinterpret_cast<const opus_mqa_cta_record*>(cta_info.data_ptr());
    kargs.num_ctas           = num_ctas;
    kargs.num_rows           = num_rows;
    kargs.max_seq_len        = max_seq_len;
    kargs.stride_out_row     = static_cast<int>(out.stride(0));
    kargs.weight_scale       = weight_scale;
    kargs.block_k            = Traits::KV_TILE_SIZE;
    kargs.kv_block_size      = kv_block_size;
    kargs.max_blocks_per_seq = static_cast<int>(block_tables.size(1));

    HipDeviceGuard guard(q.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    dim3 grid(static_cast<unsigned>(num_ctas)); // one CTA per schedule slot
    dim3 block(Traits::BLOCK_SIZE);
    opus_logits::qshare::
        mqa_logits_mxfp4_32x16x128_qshare_kernel<Traits, opus_logits::mqa_logits_sched::Table>
        <<<grid, block, 0, stream>>>(kargs);
    HIP_CALL_LAUNCH(hipGetLastError());
}

void pa_mqa_logits_mxfp4_gfx1250_fwd_sched(aiter_tensor_t& q,
                                           aiter_tensor_t& q_scale,
                                           aiter_tensor_t& kv_cache,
                                           aiter_tensor_t& kv_scale,
                                           aiter_tensor_t& block_tables,
                                           aiter_tensor_t& weights,
                                           aiter_tensor_t& local_starts,
                                           aiter_tensor_t& local_ends,
                                           aiter_tensor_t& cta_info,
                                           aiter_tensor_t& out,
                                           int num_rows,
                                           int num_ctas,
                                           float weight_scale,
                                           int kv_block_size,
                                           int max_seq_len)
{
    // pybind path: make the shape checks throw a Python RuntimeError instead of abort()ing the
    // interpreter. Same convention as opus_gemm.cu / gradlib.
    aiter_detail::g_aiter_can_throw = true;
    pa_mqa_logits_mxfp4_gfx1250_launch_sched<mqa_logits_fp4_gfx1250_traits>(
        q, q_scale, kv_cache, kv_scale, block_tables, weights, local_starts, local_ends,
        cta_info, out, num_rows, num_ctas, weight_scale, kv_block_size, max_seq_len);
}

// Both builders take and return caller-allocated device buffers: no hipMalloc, no host<->device
// sync, and a grid that is a function of the static shapes. Both are PER-FORWARD quantities
// while the kernel runs PER LAYER.
void pa_mqa_logits_mxfp4_gfx1250_build_tiles(aiter_tensor_t& cu_seq_q,
                                             aiter_tensor_t& cu_tiles,
                                             int total_q,
                                             int max_tiles)
{
    aiter_detail::g_aiter_can_throw = true;
    constexpr int QPB               = mqa_logits_fp4_gfx1250_traits::Q_PER_BLOCK;
    const int B                     = static_cast<int>(cu_seq_q.size(0)) - 1;
    AITER_CHECK(cu_seq_q.dtype() == AITER_DTYPE_i32 && cu_tiles.dtype() == AITER_DTYPE_i32,
                "cu_seq_q / cu_tiles must be int32");
    AITER_CHECK(cu_seq_q.is_contiguous() && cu_tiles.is_contiguous(),
                "cu_seq_q / cu_tiles must be contiguous");
    AITER_CHECK(B >= 1, "cu_seq_q must have length batch+1 with batch >= 1, got ", B + 1);
    // The scan is serial on one lane and the emit binary-searches it per tile, so the cost grows
    // with the batch: 3.4 us at B <= 128, 8.9 at B = 512, 32.9 at the cap. Fine for a decode
    // batch of 128; hundreds of sequences would want a parallel scan first.
    AITER_CHECK(B <= opus_logits::GROUPS_BUILD_MAX_BATCH,
                "the tile cut scans the batch prefix in LDS and is capped at ",
                opus_logits::GROUPS_BUILD_MAX_BATCH,
                " batches, got ",
                B);
    // The kernel writes every t in [0, max_tiles], so the array is max_tiles + 1 long.
    AITER_CHECK(static_cast<int64_t>(cu_tiles.numel()) >= (int64_t)max_tiles + 1,
                "cu_tiles needs max_tiles + 1 = ",
                max_tiles + 1,
                " entries, got ",
                cu_tiles.numel());
    // A correctness bound, not a tuning one: a max_tiles below the real count silently DROPS the
    // tail tiles, and their rows are then never written.
    AITER_CHECK((int64_t)max_tiles >= ((int64_t)total_q + QPB - 1) / QPB,
                "max_tiles must cover every row; at Q_PER_BLOCK=",
                (int)QPB,
                " and total_q=",
                total_q,
                " it must be at least ",
                (total_q + QPB - 1) / QPB,
                ", got ",
                max_tiles);

    if(max_tiles <= 0)
        return;

    HipDeviceGuard guard(cu_seq_q.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    opus_logits::mqa_logits_build_tiles<<<1, opus_logits::GROUPS_BUILD_BLOCK, 0, stream>>>(
        reinterpret_cast<const int*>(cu_seq_q.data_ptr()),
        reinterpret_cast<int*>(cu_tiles.data_ptr()),
        B,
        max_tiles,
        QPB);
    HIP_CALL_LAUNCH(hipGetLastError());
}

// Fill `cta_info`. No host<->device sync and nothing read back, so it is capture-safe.
//
// `sched_plan` is the policy and lives in the header, so this launcher and the opus-ops
// standalone host cannot drift on it; only the dispatch is per-host.
void pa_mqa_logits_mxfp4_gfx1250_build_sched(aiter_tensor_t& cu_tiles,
                                             aiter_tensor_t& local_starts,
                                             aiter_tensor_t& local_ends,
                                             aiter_tensor_t& row_to_batch,
                                             aiter_tensor_t& cta_info,
                                             int num_tiles,
                                             int num_ctas,
                                             int cta_resident)
{
    aiter_detail::g_aiter_can_throw = true;
    namespace ol                    = opus_logits;
    AITER_CHECK(cu_tiles.dtype() == AITER_DTYPE_i32 && cu_tiles.is_contiguous(),
                "cu_tiles must be contiguous int32");
    AITER_CHECK(local_ends.dtype() == AITER_DTYPE_i32 && local_ends.is_contiguous(),
                "local_ends must be contiguous int32");
    AITER_CHECK(cta_info.dtype() == AITER_DTYPE_i32 && cta_info.is_contiguous(),
                "cta_info must be contiguous int32");
    AITER_CHECK(num_tiles >= 0, "num_tiles must be >= 0, got ", num_tiles);
    // Tile t reads BOTH cu_tiles[t] and cu_tiles[t + 1].
    AITER_CHECK(static_cast<int64_t>(cu_tiles.numel()) >= (int64_t)num_tiles + 1,
                "cu_tiles holds one boundary per tile PLUS a terminator; need ",
                num_tiles + 1,
                ", got ",
                cu_tiles.numel());
    // Below this a tile could get no CTA at all and its rows would keep whatever the caller
    // pre-filled -- silently, since every other row would still be right.
    AITER_CHECK(num_ctas >= num_tiles,
                "num_ctas (",
                num_ctas,
                ") must be >= num_tiles (",
                num_tiles,
                "); aiter.ops.opus.pa_mqa_logits_mxfp4.pa_mqa_logits_mxfp4_plan() is what "
                "guarantees it");
    // The BUFFER is `sched_buffer_records(num_ctas)` while the GRID stays `num_ctas`: the
    // multi-workgroup emit's per-block partials sit past the slots.
    AITER_CHECK(static_cast<int64_t>(cta_info.numel()) >=
                    (int64_t)ol::sched_buffer_records(num_ctas) * 8,
                "cta_info holds 8 int32 per CTA slot plus a ",
                ol::SCHED_SCRATCH_RECORDS,
                "-record build scratch; need ",
                (int64_t)ol::sched_buffer_records(num_ctas) * 8,
                ", got numel=",
                cta_info.numel(),
                ". Let aiter.ops.opus.pa_mqa_logits_mxfp4.pa_mqa_logits_mxfp4_plan() allocate "
                "it, or hand back a previous plan's cta_info");

    if(num_tiles == 0 && num_ctas == 0)
        return;

    const int* p_ls = nullptr;
    if(local_starts.numel() > 0)
    {
        AITER_CHECK(local_starts.dtype() == AITER_DTYPE_i32 && local_starts.is_contiguous(),
                    "local_starts, when given, must be contiguous int32");
        p_ls = reinterpret_cast<const int*>(local_starts.data_ptr());
    }
    const int* p_rb = nullptr;
    if(row_to_batch.numel() > 0)
    {
        AITER_CHECK(row_to_batch.dtype() == AITER_DTYPE_i32 && row_to_batch.is_contiguous(),
                    "row_to_batch, when given, must be contiguous int32");
        p_rb = reinterpret_cast<const int*>(row_to_batch.data_ptr());
    }

    HipDeviceGuard guard(local_ends.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    const int* p_cut         = reinterpret_cast<const int*>(cu_tiles.data_ptr());
    const int* p_le          = reinterpret_cast<const int*>(local_ends.data_ptr());
    auto* p_cta              = reinterpret_cast<opus_mqa_cta_record*>(cta_info.data_ptr());
    const int block_k        = mqa_logits_fp4_gfx1250_traits::KV_TILE_SIZE;

    const auto plan = ol::sched_plan(num_tiles, num_ctas);
    if(plan.blocks == 1)
    {
        if(plan.block == ol::SCHED_BUILD_BLOCK)
            ol::mqa_logits_build_sched<ol::SCHED_BUILD_BLOCK>
                <<<1, ol::SCHED_BUILD_BLOCK, 0, stream>>>(
                    p_cut, p_ls, p_le, p_rb, p_cta, num_tiles, num_ctas, block_k, cta_resident);
        else
            ol::mqa_logits_build_sched<ol::SCHED_BUILD_BLOCK_WIDE>
                <<<1, ol::SCHED_BUILD_BLOCK_WIDE, 0, stream>>>(
                    p_cut, p_ls, p_le, p_rb, p_cta, num_tiles, num_ctas, block_k, cta_resident);
    }
    else
    {
        int* scratch = reinterpret_cast<int*>(p_cta + num_ctas);
        ol::mqa_logits_build_sched_emit<ol::SCHED_BUILD_BLOCK>
            <<<plan.blocks, ol::SCHED_BUILD_BLOCK, 0, stream>>>(p_cut,
                                                                p_ls,
                                                                p_le,
                                                                p_rb,
                                                                p_cta,
                                                                scratch,
                                                                num_tiles,
                                                                num_ctas,
                                                                block_k,
                                                                plan.blocks);
        ol::mqa_logits_build_sched_finish<ol::SCHED_BUILD_BLOCK_WIDE>
            <<<1, ol::SCHED_BUILD_BLOCK_WIDE, 0, stream>>>(p_cut,
                                                           p_ls,
                                                           p_le,
                                                           p_rb,
                                                           p_cta,
                                                           scratch,
                                                           num_tiles,
                                                           num_ctas,
                                                           block_k,
                                                           cta_resident,
                                                           plan.blocks);
    }
    HIP_CALL_LAUNCH(hipGetLastError());
}
