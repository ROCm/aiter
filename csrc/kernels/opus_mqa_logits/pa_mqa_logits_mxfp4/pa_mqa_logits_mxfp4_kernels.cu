// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// MXFP4 paged MQA logits (OPUS): one launcher for gfx950 (MFMA) and gfx1250 (WMMA) over one
// per-tile schedule. build_tiles / build_sched are arch-agnostic; fwd_sched dispatches on the
// runtime arch, then (q_per_block, block_k). Input ABI: pa_mqa_logits_mxfp4_opus.h.

#define PA_MQA_LOGITS_MXFP4_IMPL
#include "pa_mqa_logits_mxfp4_opus.h"

#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "aiter_tensor.h"
#include <string>

// Compiled configs. gfx950: KV tile 256 (4-wave) or 64 (1-wave), q_per_block 1.
// gfx1250: q_per_block 4 or 1, KV tile 64 (one page).
using mqa_logits_fp4_traits_4wave =
    opus_logits::gfx950::pa_mqa_logits_mxfp4_traits<256, 64, 128, 64, 4>;
using mqa_logits_fp4_traits_1wave =
    opus_logits::gfx950::pa_mqa_logits_mxfp4_traits<64, 64, 128, 64, 1>;
using mqa_logits_traits_qlen4_kv64 = opus_logits::gfx1250::pa_mqa_logits_mxfp4_traits<4, 2, 64>;
using mqa_logits_traits_qlen1_kv64 = opus_logits::gfx1250::pa_mqa_logits_mxfp4_traits<1, 2, 64>;

// ══ shape check + launch, one template for both arches ══════════════════════════════════════
// The kernel strides every input by compile-time constants and reads no runtime stride, so a
// padded or permuted input would be silently wrong: the layout is required. Only `out` passes
// its row stride, so it needs just its last dim packed.
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
    // The store is bounded by the window, not max_seq_len: these are the only guard on `out`.
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
    // weights rows are strided by the compile-time W_ROW_ELEMS; a narrower row would overread.
    AITER_CHECK(weights.size(1) == Traits::W_ROW_ELEMS,
                "weights is [T, H] and the kernel strides it by a compile-time H=",
                (int)Traits::W_ROW_ELEMS, ": got weights.size(1)=", weights.size(1));
    AITER_CHECK(D_BYTES == HEAD_BYTES,
                "q last dim is D/2 packed bytes; compiled for ", HEAD_BYTES,
                " (D=", (int)Traits::HEAD_DIM, "), got ", D_BYTES);
    static_assert(Traits::N_HEADS * HEAD_BYTES == Traits::Q_ROW_BYTES,
                  "the kernel strides q rows by Q_ROW_BYTES; it must equal H * D/2");
    AITER_CHECK(block_k == Traits::KV_TILE_SIZE,
                "compiled for block_k=", (int)Traits::KV_TILE_SIZE, ", got ", block_k);
    AITER_CHECK(kv_block_size == Traits::PAGE_SIZE,
                "compiled for kv_block_size=", (int)Traits::PAGE_SIZE, ", got ", kv_block_size);

    // block_tables is sized in KV tiles, not pages: a CTA reads every page of its last tile,
    // past where the window ends. Nothing else bounds the page index.
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

    // Scale byte counts catch a wrong array, not a wrong permutation: every fp4 scale layout,
    // either arch's, has the same size.
    AITER_CHECK(q_scale.numel() == (int64_t)q.size(0) * Traits::QS_ROW_BYTES,
                "q_scale must hold ", (int)Traits::QS_ROW_BYTES,
                " bytes per query row in this arch's layout, got numel=", q_scale.numel(),
                " for T=", q.size(0));
    AITER_CHECK(kv_cache.numel() % Traits::KV_PAGE_BYTES == 0,
                "kv_cache must be a whole number of ", (int)Traits::KV_PAGE_BYTES,
                "-byte pages, got numel=", kv_cache.numel());
    const int64_t num_blocks = kv_cache.numel() / Traits::KV_PAGE_BYTES;
    AITER_CHECK(kv_scale.numel() == num_blocks * Traits::KVS_PAGE_BYTES,
                "kv_scale must hold ", (int)Traits::KVS_PAGE_BYTES,
                " bytes per page in this arch's layout, got numel=", kv_scale.numel(),
                " for num_blocks=", num_blocks);
}

// ── the launch: 1D grid over the schedule table ───────────────────────────────
// The caller (fwd_sched) holds the device guard.
template<class Traits>
static void pa_mqa_logits_mxfp4_launch_sched(aiter_tensor_t& q,
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
                                             int block_k,
                                             int kv_block_size,
                                             int max_seq_len)
{
    pa_mqa_logits_mxfp4_check_shapes<Traits>(q, q_scale, kv_cache, kv_scale, block_tables,
                                             weights, out, block_k, kv_block_size, max_seq_len);
    AITER_CHECK(cta_info.dtype() == AITER_DTYPE_i32 && cta_info.is_contiguous(),
                "cta_info must be contiguous int32");
    // num_rows is the schedule's row count; the kernels take rows off the records, so this
    // bounds q / weights / out.
    AITER_CHECK(num_rows >= 0 && num_rows <= q.size(0),
                "the schedule was built for num_rows=", num_rows, " query rows and q holds ",
                q.size(0), "; build the plan with total_q <= q.shape[0]");
    AITER_CHECK(num_ctas > 0, "num_ctas must be >= 1, got ", num_ctas);
    AITER_CHECK(static_cast<int64_t>(cta_info.numel()) >= (int64_t)num_ctas * 8,
                "cta_info holds 8 int32 per CTA slot; need ", (int64_t)num_ctas * 8,
                " for num_ctas=", num_ctas, ", got numel=", cta_info.numel(),
                ". Let aiter.ops.opus.pa_mqa_logits_mxfp4.pa_mqa_logits_mxfp4_plan() allocate "
                "it, or hand back a previous plan's cta_info");

    // Per-row window arrays are read only on gfx1250; gfx950 takes the window off the record.
    const int* p_ls = nullptr;
    const int* p_le = nullptr;
    if constexpr(Traits::READS_ROW_WINDOWS)
    {
        AITER_CHECK(local_ends.dtype() == AITER_DTYPE_i32 && local_ends.is_contiguous(),
                    "local_ends must be contiguous int32");
        AITER_CHECK(static_cast<int64_t>(local_ends.numel()) >= num_rows,
                    "local_ends is per query row; need at least ", num_rows, " entries, got ",
                    local_ends.numel());
        p_le = reinterpret_cast<const int*>(local_ends.data_ptr());
        if(local_starts.numel() > 0)
        {
            AITER_CHECK(local_starts.dtype() == AITER_DTYPE_i32 && local_starts.is_contiguous() &&
                            static_cast<int64_t>(local_starts.numel()) >= num_rows,
                        "local_starts, when given, must be contiguous int32 with one entry per row");
            p_ls = reinterpret_cast<const int*>(local_starts.data_ptr());
        }
    }

    if(num_rows <= 0)
        return;

    opus_mqa_logits_kargs kargs{};
    kargs.ptr_q              = q.data_ptr();
    kargs.ptr_q_scale        = q_scale.data_ptr();
    kargs.ptr_kv             = kv_cache.data_ptr();
    kargs.ptr_kv_scale       = kv_scale.data_ptr();
    kargs.ptr_block_tables   = reinterpret_cast<const int*>(block_tables.data_ptr());
    kargs.ptr_weights        = weights.data_ptr();
    kargs.ptr_out            = reinterpret_cast<float*>(out.data_ptr());
    kargs.ptr_local_starts   = p_ls;
    kargs.ptr_local_ends     = p_le;
    kargs.ptr_cta_info = reinterpret_cast<const opus_mqa_cta_record*>(cta_info.data_ptr());
    kargs.num_ctas           = num_ctas;
    kargs.num_rows           = num_rows;
    kargs.stride_out_row     = static_cast<int>(out.stride(0));
    kargs.weight_scale       = weight_scale;
    kargs.block_k            = Traits::KV_TILE_SIZE;
    kargs.max_blocks_per_seq = static_cast<int>(block_tables.size(1));

    const hipStream_t stream = aiter::getCurrentHIPStream();
    const dim3 grid(static_cast<unsigned>(num_ctas));   // one CTA per schedule slot
    const dim3 block(Traits::BLOCK_SIZE);
    if constexpr(Traits::READS_ROW_WINDOWS)
        opus_logits::gfx1250::
            pa_mqa_logits_mxfp4_kernel<Traits, opus_logits::mqa_logits_sched::Table>
            <<<grid, block, 0, stream>>>(kargs);
    else
        opus_logits::gfx950::
            pa_mqa_logits_mxfp4_kernel<Traits, opus_logits::mqa_logits_sched::Table>
            <<<grid, block, 0, stream>>>(kargs);
    HIP_CALL_LAUNCH(hipGetLastError());
}

// Both builders use caller-allocated buffers, no host sync (capture-safe), and run once per
// forward while the kernel runs per layer.
void pa_mqa_logits_mxfp4_build_tiles(aiter_tensor_t& cu_seq_q,
                                             aiter_tensor_t& cu_tiles,
                                             int total_q,
                                             int max_tiles,
                                             int q_per_block)
{
    aiter_detail::g_aiter_can_throw = true;
    // Must match the launch's q_per_block.
    const int QPB                   = q_per_block;
    AITER_CHECK(QPB >= 1, "q_per_block must be >= 1, got ", QPB);
    const int B                     = static_cast<int>(cu_seq_q.size(0)) - 1;
    AITER_CHECK(cu_seq_q.dtype() == AITER_DTYPE_i32 && cu_tiles.dtype() == AITER_DTYPE_i32,
                "cu_seq_q / cu_tiles must be int32");
    AITER_CHECK(cu_seq_q.is_contiguous() && cu_tiles.is_contiguous(),
                "cu_seq_q / cu_tiles must be contiguous");
    AITER_CHECK(B >= 1, "cu_seq_q must have length batch+1 with batch >= 1, got ", B + 1);
    // The prefix scan is serial on one lane, so cost grows with the batch.
    AITER_CHECK(B <= opus_logits::GROUPS_BUILD_MAX_BATCH,
                "the tile cut scans the batch prefix in LDS and is capped at ",
                opus_logits::GROUPS_BUILD_MAX_BATCH,
                " batches, got ",
                B);
    AITER_CHECK(static_cast<int64_t>(cu_tiles.numel()) >= (int64_t)max_tiles + 1,
                "cu_tiles needs max_tiles + 1 = ",
                max_tiles + 1,
                " entries, got ",
                cu_tiles.numel());
    // Too small a max_tiles silently drops the tail tiles' rows.
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

// Fill `cta_info`. The sched_plan policy lives in the header; only the dispatch is here.
void pa_mqa_logits_mxfp4_build_sched(aiter_tensor_t& cu_tiles,
                                             aiter_tensor_t& local_starts,
                                             aiter_tensor_t& local_ends,
                                             aiter_tensor_t& row_to_batch,
                                             aiter_tensor_t& cta_info,
                                             int num_tiles,
                                             int num_rows,
                                             int num_ctas,
                                             int cta_resident,
                                             int block_k,
                                             int q_per_block)
{
    aiter_detail::g_aiter_can_throw = true;
    AITER_CHECK(q_per_block >= 1, "q_per_block must be >= 1, got ", q_per_block);
    const bool identity_cut = (q_per_block == 1);
    // cta_resident follows the kernel's occupancy, which only the caller knows; a wrong value
    // idles the part silently. block_k must be the launch's KV tile.
    AITER_CHECK(cta_resident >= 1, "cta_resident must be >= 1, got ", cta_resident);
    AITER_CHECK(block_k >= 1, "block_k must be >= 1, got ", block_k);
    // Per-row arrays are checked against num_rows, not num_tiles (an upper bound in tiles);
    // at the identity cut the two must agree.
    AITER_CHECK(num_rows >= 0, "num_rows must be >= 0, got ", num_rows);
    if(identity_cut)
        AITER_CHECK(num_tiles == num_rows,
                    "at q_per_block == 1 a tile is a row, so num_tiles (", num_tiles,
                    ") must equal num_rows (", num_rows, ")");
    namespace ol                    = opus_logits;
    if(!identity_cut)
        AITER_CHECK(cu_tiles.dtype() == AITER_DTYPE_i32 && cu_tiles.is_contiguous(),
                    "cu_tiles must be contiguous int32");
    AITER_CHECK(local_ends.dtype() == AITER_DTYPE_i32 && local_ends.is_contiguous(),
                "local_ends must be contiguous int32");
    AITER_CHECK(static_cast<int64_t>(local_ends.numel()) >= num_rows,
                "local_ends is per row; need at least num_rows=", num_rows, " entries, got ",
                local_ends.numel());
    AITER_CHECK(cta_info.dtype() == AITER_DTYPE_i32 && cta_info.is_contiguous(),
                "cta_info must be contiguous int32");
    AITER_CHECK(num_tiles >= 0, "num_tiles must be >= 0, got ", num_tiles);
    if(!identity_cut)
        AITER_CHECK(static_cast<int64_t>(cu_tiles.numel()) >= (int64_t)num_tiles + 1,
                "cu_tiles holds one boundary per tile PLUS a terminator; need ",
                    num_tiles + 1,
                    ", got ",
                    cu_tiles.numel());
    // Otherwise a tile may get no CTA and its rows are silently never written.
    AITER_CHECK(num_ctas >= num_tiles,
                "num_ctas (",
                num_ctas,
                ") must be >= num_tiles (",
                num_tiles,
                "); aiter.ops.opus.pa_mqa_logits_mxfp4.pa_mqa_logits_mxfp4_plan() is what "
                "guarantees it");
    // The buffer also holds the multi-block emit's scratch past the num_ctas slots.
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
        AITER_CHECK(local_starts.dtype() == AITER_DTYPE_i32 && local_starts.is_contiguous() &&
                        static_cast<int64_t>(local_starts.numel()) >= num_rows,
                    "local_starts, when given, must be contiguous int32 with at least num_rows=",
                    num_rows, " entries, got ", local_starts.numel());
        p_ls = reinterpret_cast<const int*>(local_starts.data_ptr());
    }
    const int* p_rb = nullptr;
    if(row_to_batch.numel() > 0)
    {
        AITER_CHECK(row_to_batch.dtype() == AITER_DTYPE_i32 && row_to_batch.is_contiguous() &&
                        static_cast<int64_t>(row_to_batch.numel()) >= num_rows,
                    "row_to_batch, when given, must be contiguous int32 with at least num_rows=",
                    num_rows, " entries, got ", row_to_batch.numel());
        p_rb = reinterpret_cast<const int*>(row_to_batch.data_ptr());
    }

    HipDeviceGuard guard(local_ends.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    const int* p_cut =
        identity_cut ? nullptr : reinterpret_cast<const int*>(cu_tiles.data_ptr());
    const int* p_le          = reinterpret_cast<const int*>(local_ends.data_ptr());
    auto* p_cta              = reinterpret_cast<opus_mqa_cta_record*>(cta_info.data_ptr());

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

// ══ the public fwd op: dispatch on runtime arch, then (q_per_block, block_k) ═════════════════
// The other arch's kernel compiles to an empty stub. An unmatched config must raise, never fall
// back to a default.
void pa_mqa_logits_mxfp4_fwd_sched(aiter_tensor_t& q,
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
                                   int max_seq_len,
                                   int q_per_block,
                                   int block_k)
{
    aiter_detail::g_aiter_can_throw = true;
    // Launch on q's device, not the current one.
    HipDeviceGuard guard(q.device_id);
    // Probed once per process, on the first call's device (fwd_sched runs every layer). Assumes
    // one arch per process: mixed-arch nodes are not supported.
    static const std::string arch = get_gpu_arch();

#define PA_MQA_LOGITS_MXFP4_LAUNCH(TRAITS)                                                       \
    pa_mqa_logits_mxfp4_launch_sched<TRAITS>(q, q_scale, kv_cache, kv_scale, block_tables,     \
                                             weights, local_starts, local_ends, cta_info, out, \
                                             num_rows, num_ctas, weight_scale, block_k,        \
                                             kv_block_size, max_seq_len)
    if(arch == "gfx950")
    {
        AITER_CHECK(q_per_block == 1,
                    "gfx950 runs one row per tile (q_per_block == 1), got ", q_per_block);
        if(block_k == mqa_logits_fp4_traits_4wave::KV_TILE_SIZE)
            PA_MQA_LOGITS_MXFP4_LAUNCH(mqa_logits_fp4_traits_4wave);
        else if(block_k == mqa_logits_fp4_traits_1wave::KV_TILE_SIZE)
            PA_MQA_LOGITS_MXFP4_LAUNCH(mqa_logits_fp4_traits_1wave);
        else
            AITER_CHECK(false, "gfx950: block_k must be 256 (4-wave) or 64 (1-wave), got ",
                        block_k);
    }
    else if(arch == "gfx1250")
    {
        if(q_per_block == 4 && block_k == 64)
            PA_MQA_LOGITS_MXFP4_LAUNCH(mqa_logits_traits_qlen4_kv64);
        else if(q_per_block == 1 && block_k == 64)
            PA_MQA_LOGITS_MXFP4_LAUNCH(mqa_logits_traits_qlen1_kv64);
        else
            AITER_CHECK(false, "gfx1250: no kernel instance for q_per_block=", q_per_block,
                        " block_k=", block_k, "; this module has (4, 64) and (1, 64)");
    }
    else
    {
        AITER_CHECK(false, "pa_mqa_logits_mxfp4 supports gfx950 and gfx1250, got ", arch);
    }
#undef PA_MQA_LOGITS_MXFP4_LAUNCH
}
