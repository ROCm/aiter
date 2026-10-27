// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// MXFP4 paged MQA logits (OPUS): the registry's host side, the Table schedule builder and the
// launch. ABI: pa_mqa_logits_mxfp4.h. The kernels live in their arch's TU
// (pa_mqa_logits_mxfp4_<arch>.cu) and are reached only through their registry entries.
#include "pa_mqa_logits_mxfp4.h"
#include "pa_mqa_logits_mxfp4_sched.cuh"

#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include <algorithm>
#include <string>
#include <vector>

namespace {

namespace ol = opus_logits;

using registry_fn = const ol::mqa_logits_instance* (*)(int*);
// Every arch's instances; an instance's id is its index in this concatenation.
constexpr registry_fn REGISTRIES[] = {ol::mqa_logits_gfx950_instances,
                                       ol::mqa_logits_gfx1250_instances};

const ol::mqa_logits_instance& instance_at(int id) {
    AITER_CHECK(id >= 0, "no MXFP4 MQA-logits instance ", id);
    int base = 0;
    for (registry_fn fn : REGISTRIES) {
        int n = 0;
        const ol::mqa_logits_instance* table = fn(&n);
        if (id < base + n) return table[id - base];
        base += n;
    }
    AITER_CHECK(false, "no MXFP4 MQA-logits instance ", id, "; there are ", base);
    return *REGISTRIES[0](&base);   // unreachable
}

std::string device_arch(int device_id) {
    hipDeviceProp_t prop;
    HIP_CALL(hipGetDeviceProperties(&prop, device_id));
    std::string arch = prop.gcnArchName;
    return arch.substr(0, arch.find(':'));
}

// Tiles the Table builder cuts `rows` rows of `seqs` sequences into, at most: one a row, or up
// to rows_per_cta rows of one sequence (each sequence's last tile may be short).
int max_tiles_for(int rows, int seqs, int qpb) {
    return qpb == 1 ? rows : (rows + seqs * (qpb - 1)) / qpb;
}

void check_i32_vector(const aiter_tensor_t& t, const char* name) {
    AITER_CHECK(t.dtype() == AITER_DTYPE_i32 && t.dim() == 1 && t.is_contiguous(),
                name, " must be a contiguous 1-D int32 tensor");
}

}  // namespace

// The record layout of mqa_logits_mxfp4_instances (aiter/ops/opus/pa_mqa_logits_mxfp4.py reads it
// by the leading field count).
#define MQA_LOGITS_INSTANCE_FIELDS                                                              \
    X(id) X(heads) X(rows_per_wave) X(waves) X(stages) X(scale_tdm) X(gather) X(entry_rows_min) \
    X(entry_rows_max) X(tile_keys) X(modes) X(flat_out) X(resident_table) X(resident_free)      \
    X(resident_prefill)

int mqa_logits_mxfp4_instances(int device_id, aiter_tensor_t& out)
{
    aiter_detail::AiterThrowGuard throw_guard;
    AITER_CHECK(out.is_cpu() && out.dtype() == AITER_DTYPE_i32 && out.is_contiguous(),
                "out must be a contiguous host int32 tensor");
    HipDeviceGuard guard(device_id);
    const std::string arch = device_arch(device_id);
    int cus = 0;
    HIP_CALL(hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, device_id));

#define X(f) +1
    constexpr int fields = 0 MQA_LOGITS_INSTANCE_FIELDS;
#undef X
    std::vector<int> recs{fields};
    int id = 0;
    for (registry_fn fn : REGISTRIES) {
        int n = 0;
        const ol::mqa_logits_instance* table = fn(&n);
        for (int i = 0; i < n; ++i, ++id) {
            const ol::mqa_logits_instance& in = table[i];
            if (arch != in.arch) continue;
            // CTAs resident at once in each compiled mode: occupancy x CUs, measured, not
            // hard-coded (a stale figure silently under-fills the part).
            int resident[ol::SCHED_MODES] = {};
            for (int m = 0; m < ol::SCHED_MODES; ++m) {
                if (!(in.modes & (1 << m))) continue;
                int per_cu = 0;
                HIP_CALL(in.occupancy(m, &per_cu));
                resident[m] = per_cu * cus;
            }
            const int row[] = {id, in.heads, in.rows_per_wave, in.waves, in.stages, in.scale_tdm,
                               in.gather, in.entry_rows_min, in.entry_rows_max, in.tile_keys,
                               in.modes, in.flat_out, resident[ol::SCHED_TABLE],
                               resident[ol::SCHED_FREE], resident[ol::SCHED_PREFILL]};
            static_assert(sizeof(row) / sizeof(row[0]) == fields);
            recs.insert(recs.end(), row, row + fields);
        }
    }
    const int n = static_cast<int>(recs.size());
    if (static_cast<int64_t>(out.numel()) >= n)
        std::copy(recs.begin(), recs.end(), reinterpret_cast<int*>(out.data_ptr()));
    return n;
}

// Both builders use caller-held buffers, read no device data on the host (capture-safe), and run
// once per forward while the kernel runs per layer.
void mqa_logits_mxfp4_build_schedule(aiter_tensor_t& query_start_loc,
                                     aiter_tensor_t& row_ends,
                                     aiter_tensor_t& cu_tiles,
                                     aiter_tensor_t& cta_info,
                                     int instance,
                                     int num_ctas,
                                     int cta_resident)
{
    aiter_detail::AiterThrowGuard throw_guard;
    const ol::mqa_logits_instance& in = instance_at(instance);
    AITER_CHECK(in.modes & (1 << ol::SCHED_TABLE), "instance ", instance, " has no Table mode");
    check_i32_vector(query_start_loc, "query_start_loc");
    check_i32_vector(row_ends, "row_ends");
    AITER_CHECK(cta_info.dtype() == AITER_DTYPE_i32 && cta_info.is_contiguous(),
                "cta_info must be contiguous int32");
    const int qpb      = in.rows_per_wave * in.waves;
    const int num_seqs = static_cast<int>(query_start_loc.numel()) - 1;
    const int num_rows = static_cast<int>(row_ends.numel());
    AITER_CHECK(num_seqs >= 1, "query_start_loc must hold num_seqs + 1 >= 2 entries");
    // cta_resident follows the kernel's occupancy; a wrong value idles the part silently.
    AITER_CHECK(cta_resident >= 1, "cta_resident must be >= 1, got ", cta_resident);
    const int num_tiles = max_tiles_for(num_rows, num_seqs, qpb);
    // Otherwise a tile may get no CTA and its rows are silently never written.
    AITER_CHECK(num_ctas >= num_tiles, "num_ctas (", num_ctas, ") must be >= the tiles (",
                num_tiles, ") of ", num_rows, " rows in ", num_seqs, " sequences");
    // The buffer also holds the multi-block emit's scratch past the num_ctas slots.
    AITER_CHECK(static_cast<int64_t>(cta_info.numel()) >= (int64_t)ol::sched_buffer_records(num_ctas) * 8,
                "cta_info holds 8 int32 a CTA slot plus a ", ol::SCHED_SCRATCH_RECORDS,
                "-record build scratch; need ", (int64_t)ol::sched_buffer_records(num_ctas) * 8,
                ", got ", cta_info.numel(), ". Take it from cfg.schedule_buffers()");
    if (qpb > 1) {
        // the cut [num_tiles + 1], then each tile's sequence [num_tiles + 1]
        AITER_CHECK(cu_tiles.dtype() == AITER_DTYPE_i32 && cu_tiles.is_contiguous() &&
                        static_cast<int64_t>(cu_tiles.numel()) >= 2 * ((int64_t)num_tiles + 1),
                    "cu_tiles must be contiguous int32 with ", 2 * (num_tiles + 1), " entries");
        // The tile cut stages the sequences' tile offsets in one workgroup's LDS.
        AITER_CHECK(num_seqs <= ol::GROUPS_BUILD_MAX_BATCH, "the Table builder cuts at most ",
                    ol::GROUPS_BUILD_MAX_BATCH, " sequences into tiles, got ", num_seqs);
    }
    if (num_tiles == 0 && num_ctas == 0) return;

    HipDeviceGuard guard(row_ends.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    const int* p_qsl = reinterpret_cast<const int*>(query_start_loc.data_ptr());
    const int* p_le  = reinterpret_cast<const int*>(row_ends.data_ptr());
    auto* p_cta      = reinterpret_cast<opus_mqa_cta_record*>(cta_info.data_ptr());
    const int* p_cut = nullptr;
    const int* p_seq = nullptr;
    if (qpb > 1 && num_tiles > 0) {
        int* cut = reinterpret_cast<int*>(cu_tiles.data_ptr());
        ol::mqa_logits_build_tiles<<<1, ol::GROUPS_BUILD_BLOCK, 0, stream>>>(
            p_qsl, cut, cut + num_tiles + 1, num_seqs, num_tiles, qpb);
        HIP_CALL_LAUNCH(hipGetLastError());
        p_cut = cut;
        p_seq = cut + num_tiles + 1;
    }
    const int block_k = in.tile_keys;
    const auto plan   = ol::sched_plan(num_tiles, num_ctas);
    if (plan.blocks == 1) {
        if (plan.block == ol::SCHED_BUILD_BLOCK)
            ol::mqa_logits_build_sched<ol::SCHED_BUILD_BLOCK><<<1, ol::SCHED_BUILD_BLOCK, 0, stream>>>(
                p_cut, p_seq, p_qsl, p_le, p_cta, num_seqs, num_tiles, num_ctas, block_k, cta_resident);
        else
            ol::mqa_logits_build_sched<ol::SCHED_BUILD_BLOCK_WIDE><<<1, ol::SCHED_BUILD_BLOCK_WIDE, 0, stream>>>(
                p_cut, p_seq, p_qsl, p_le, p_cta, num_seqs, num_tiles, num_ctas, block_k, cta_resident);
    } else {
        int* scratch = reinterpret_cast<int*>(p_cta + num_ctas);
        ol::mqa_logits_build_sched_emit<ol::SCHED_BUILD_BLOCK><<<plan.blocks, ol::SCHED_BUILD_BLOCK, 0, stream>>>(
            p_cut, p_seq, p_qsl, p_le, p_cta, scratch, num_seqs, num_tiles, num_ctas, block_k, plan.blocks);
        ol::mqa_logits_build_sched_finish<ol::SCHED_BUILD_BLOCK_WIDE><<<1, ol::SCHED_BUILD_BLOCK_WIDE, 0, stream>>>(
            p_cut, p_seq, p_qsl, p_le, p_cta, scratch, num_seqs, num_tiles, num_ctas, block_k,
            cta_resident, plan.blocks);
    }
    HIP_CALL_LAUNCH(hipGetLastError());
}

// The kernels stride every input by compile-time constants and read no runtime stride but the
// table's and out's, so a padded or permuted input would be silently wrong: layouts are required.
// Byte counts catch a wrong array, not a wrong permutation (the Python wrapper checks the ndim).
void mqa_logits_mxfp4_launch(int instance,
                             int mode,
                             aiter_tensor_t& q,
                             aiter_tensor_t& q_scale,
                             aiter_tensor_t& kv_cache,
                             aiter_tensor_t& kv_scale,
                             aiter_tensor_t& weights,
                             aiter_tensor_t& block_tables,
                             aiter_tensor_t& query_start_loc,
                             aiter_tensor_t& row_ends,
                             aiter_tensor_t& cta_info,
                             aiter_tensor_t& out,
                             aiter_tensor_t& out_offsets,
                             float weight_scale,
                             int tpb_shift,
                             int cta_target,
                             int prefill_parts,
                             int grid)
{
    aiter_detail::AiterThrowGuard throw_guard;
    const ol::mqa_logits_instance& in = instance_at(instance);
    AITER_CHECK(mode >= 0 && mode < ol::SCHED_MODES && (in.modes & (1 << mode)),
                "instance ", instance, " has no schedule mode ", mode);

    constexpr int TOKEN_BYTES = 64;   // D = 128 fp4
    constexpr int SCALE_BYTES = 4;    // D / 32 E8M0
    const int64_t rows = q.dim() == 3 ? q.size(0) : -1;
    AITER_CHECK(rows >= 0 && q.size(1) == in.heads && q.size(2) == TOKEN_BYTES,
                "q must be [rows, ", in.heads, ", ", TOKEN_BYTES, "] (H x D/2 packed fp4)");
    AITER_CHECK(q.dtype() == AITER_DTYPE_fp4x2 || q.dtype() == AITER_DTYPE_u8,
                "q must be fp4x2 (E2M1, 2/byte) or u8 bytes");
    AITER_CHECK(kv_cache.dtype() == AITER_DTYPE_fp4x2 || kv_cache.dtype() == AITER_DTYPE_u8,
                "kv_cache must be fp4x2 (E2M1, 2/byte) or u8 bytes");
    AITER_CHECK(q_scale.dtype() == AITER_DTYPE_u8 && kv_scale.dtype() == AITER_DTYPE_u8,
                "q_scale / kv_scale are E8M0 bytes and must be u8");
    AITER_CHECK(weights.dtype() == AITER_DTYPE_bf16 && weights.dim() == 2 &&
                    weights.size(0) >= rows && weights.size(1) == in.heads,
                "weights must be bf16 [rows, ", in.heads, "]");
    AITER_CHECK(q.is_contiguous() && q_scale.is_contiguous() && kv_cache.is_contiguous() &&
                    kv_scale.is_contiguous() && weights.is_contiguous(),
                "q / q_scale / kv_cache / kv_scale / weights must be contiguous");
    AITER_CHECK(static_cast<int64_t>(q_scale.numel()) == rows * in.heads * SCALE_BYTES,
                "q_scale must hold ", in.heads * SCALE_BYTES, " bytes a row, got ", q_scale.numel(),
                " for ", rows, " rows");
    AITER_CHECK(kv_cache.numel() % TOKEN_BYTES == 0 &&
                    kv_scale.numel() == kv_cache.numel() / TOKEN_BYTES * SCALE_BYTES,
                "kv_cache must be whole ", TOKEN_BYTES, "-byte keys and kv_scale ", SCALE_BYTES,
                " bytes a key; got ", kv_cache.numel(), " and ", kv_scale.numel());
    AITER_CHECK(block_tables.dtype() == AITER_DTYPE_i32 && block_tables.dim() == 2 &&
                    block_tables.stride(1) == 1,
                "block_tables must be int32 [num_seqs, width] with unit column stride");
    check_i32_vector(query_start_loc, "query_start_loc");
    check_i32_vector(row_ends, "row_ends");
    const int num_seqs = static_cast<int>(query_start_loc.numel()) - 1;
    AITER_CHECK(num_seqs >= 1 && block_tables.size(0) >= num_seqs,
                "query_start_loc must hold num_seqs + 1 >= 2 entries and block_tables a row each");
    AITER_CHECK(static_cast<int64_t>(row_ends.numel()) >= rows, "row_ends needs ", rows, " entries");

    AITER_CHECK(out.dtype() == AITER_DTYPE_fp32, "out must be fp32");
    const bool flat = out_offsets.numel() > 0;
    if (flat) {
        AITER_CHECK(in.flat_out, "instance ", instance, " writes [rows, width] only, not out_offsets");
        AITER_CHECK(out_offsets.dtype() == AITER_DTYPE_i64 && out_offsets.is_contiguous() &&
                        static_cast<int64_t>(out_offsets.numel()) >= rows,
                    "out_offsets must be contiguous int64 with an entry a row");
        AITER_CHECK(out.is_contiguous(), "a flat out must be contiguous");
    } else {
        AITER_CHECK(out.dim() == 2 && out.size(0) >= rows && out.stride(1) == 1,
                    "out must be [rows, width] with unit column stride");
    }
    if (mode == ol::SCHED_TABLE)
        AITER_CHECK(cta_info.dtype() == AITER_DTYPE_i32 && cta_info.is_contiguous() &&
                        static_cast<int64_t>(cta_info.numel()) >= (int64_t)grid * 8,
                    "a Table launch over ", grid, " slots needs that many records");
    else
        AITER_CHECK(cta_target >= 1 && prefill_parts >= 1, "cta_target / prefill_parts must be >= 1");
    AITER_CHECK(grid >= 0 && tpb_shift >= 0 && tpb_shift < 16, "bad grid / tpb_shift");
    if (rows == 0 || grid == 0) return;

    mqa_logits_mxfp4_kargs k{};
    k.ptr_q               = q.data_ptr();
    k.ptr_q_scale         = q_scale.data_ptr();
    k.ptr_kv              = kv_cache.data_ptr();
    k.ptr_kv_scale        = kv_scale.data_ptr();
    k.ptr_block_tables    = reinterpret_cast<const int*>(block_tables.data_ptr());
    k.ptr_weights         = weights.data_ptr();
    k.ptr_out             = reinterpret_cast<float*>(out.data_ptr());
    k.ptr_out_offsets     = flat ? reinterpret_cast<const long long*>(out_offsets.data_ptr()) : nullptr;
    k.ptr_query_start_loc = reinterpret_cast<const int*>(query_start_loc.data_ptr());
    k.ptr_row_ends        = reinterpret_cast<const int*>(row_ends.data_ptr());
    k.ptr_cta_info        = mode == ol::SCHED_TABLE
                                ? reinterpret_cast<const opus_mqa_cta_record*>(cta_info.data_ptr())
                                : nullptr;
    k.bt_stride           = static_cast<int>(block_tables.stride(0));
    k.tpb_shift           = tpb_shift;
    k.out_stride          = flat ? 0 : static_cast<int>(out.stride(0));
    k.num_rows            = static_cast<int>(rows);
    k.num_seqs            = num_seqs;
    k.weight_scale        = weight_scale;
    k.cta_target          = cta_target;
    k.prefill_parts       = prefill_parts;

    HipDeviceGuard guard(q.device_id);
    HIP_CALL_LAUNCH(in.launch(mode, k, grid, aiter::getCurrentHIPStream()));
}
