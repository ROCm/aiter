// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx950 MoE token permute / unpermute: host checks and launches. Device code:
// include/gfx950/opus_moe_token_reorder_pipeline_gfx950.cuh.

#include "opus_moe_token_reorder.h"

#include "gfx950/opus_moe_token_reorder_pipeline_gfx950.cuh"

#include "aiter_hip_common.h"
#include "aiter_stream.h"

#include <climits>

namespace {

using namespace opus_moe_token_reorder;

constexpr int64_t VEC = OPUS_MOE_TOKEN_REORDER_VEC;

inline bool is_rows(const aiter_tensor_t& t) {
    // a row is addressed with 32-bit byte offsets, a buffer [rows, H] of any size with one 64-bit base per row
    return t.dim() == 2 && t.dtype() == AITER_DTYPE_bf16 && t.is_contiguous() && t.size(1) % VEC == 0 &&
           t.size(1) < (1ll << 30) && t.size(0) <= INT_MAX;
}
// A per-token / per-row table: [n] or [n, 1], contiguous.
inline bool is_table(const aiter_tensor_t& t, int dtype, int64_t n) {
    return t.dtype() == dtype && t.is_contiguous() && t.numel() == n &&
           (t.dim() == 1 || (t.dim() == 2 && t.size(1) == 1));
}
inline bool is_i32(const aiter_tensor_t& t, int64_t n) { return is_table(t, AITER_DTYPE_i32, n); }
inline bool is_fp32(const aiter_tensor_t& t, int64_t n) { return is_table(t, AITER_DTYPE_fp32, n); }

template <typename Kernel, typename Kargs>
inline void launch(Kernel kernel, int64_t workgroups, const Kargs& kargs, hipStream_t stream) {
    if (workgroups == 0)
        return;
    hipLaunchKernelGGL(kernel, dim3(static_cast<uint32_t>(workgroups)), dim3(THREADS), 0, stream, kargs);
}

// A gather source is read once per call: non-temporal when it does not fit the last-level cache.
inline int streams(const aiter_tensor_t& src) {
    return src.size(0) * src.size(1) * 2 > OPUS_MOE_TOKEN_REORDER_LLC_BYTES;
}

// K2 / K3 / K4: dst[b] = src[index[b]] (times weights[b]), one workgroup per row of dst.
template <typename Kernel>
inline void move_rows(Kernel kernel, const char* what, aiter_tensor_t& src, aiter_tensor_t& index, aiter_tensor_t& dst,
                      aiter_tensor_t* weights = nullptr) {
    AITER_CHECK(is_rows(src) && is_rows(dst) && src.size(1) == dst.size(1), what,
                ": the rows must be contiguous bf16 [rows, H], H a multiple of 8");
    AITER_CHECK(is_i32(index, dst.size(0)), what, ": the index must be contiguous int32, one per destination row");
    AITER_CHECK(!weights || is_fp32(*weights, dst.size(0)), what, ": weights must be contiguous fp32 [N]");
    AITER_CHECK(src.device_id == dst.device_id && index.device_id == dst.device_id &&
                    (!weights || weights->device_id == dst.device_id),
                what, ": one device");
    HipDeviceGuard device_guard(dst.device_id);
    opus_moe_permute_rows_kargs k{};
    k.src = static_cast<const char*>(src.data_ptr());
    k.dst = static_cast<char*>(dst.data_ptr());
    k.index = static_cast<const int*>(index.data_ptr());
    k.weights = weights ? static_cast<const float*>(weights->data_ptr()) : nullptr;
    k.chunks = static_cast<int>(dst.size(1) / VEC);
    k.stream = streams(src);
    launch(kernel, dst.size(0), k, aiter::getCurrentHIPStream());
}

} // namespace

void opus_moe_sort(aiter_tensor_t& expert_ids,
                           aiter_tensor_t& counts,
                           aiter_tensor_t& reverse_sorted,
                           aiter_tensor_t& sorted_token_ids,
                           aiter_tensor_t& grouped_layout,
                           aiter_tensor_t& group_rows,
                           aiter_tensor_t& num_valid_ids) {
    const int64_t N = expert_ids.numel(), E = group_rows.numel(), R = sorted_token_ids.numel();
    AITER_CHECK(is_i32(expert_ids, N) && is_i32(reverse_sorted, N),
                "moe_sort: expert_ids / reverse_sorted must be contiguous int32 [N]");
    AITER_CHECK(is_i32(group_rows, E) && E > 0 && E <= OPUS_MOE_SORT_EXPERTS,
                "moe_sort: group_rows must be contiguous int32 [E], E up to ", OPUS_MOE_SORT_EXPERTS);
    // R: the most rows N tokens can take: N rounded up to GROUP_ALIGN, plus GROUP_ALIGN rows for every expert
    // but one (a multiple of GROUP_ALIGN). The rows in use are a multiple of GROUP_ALIGN and at least N, so fewer
    // than GROUP_ALIGN E rows are left past them: what the fill workgroups cover. The kernels write the two [R]
    // tables through views with 32-bit byte offsets.
    const int64_t align = OPUS_MOE_TOKEN_REORDER_GROUP_ALIGN;
    const int64_t n_aligned = (N + align - 1) / align * align;
    AITER_CHECK(is_i32(sorted_token_ids, R) && is_i32(grouped_layout, R) &&
                    R == n_aligned + align * (E - 1) && R < (1ll << 30),
                "moe_sort: sorted_token_ids / grouped_layout must be contiguous int32 [N rounded up to 128, + 128 (E - 1)], below 2^30 rows");
    const int64_t blocks = (N + OPUS_MOE_SORT_TOKENS - 1) / OPUS_MOE_SORT_TOKENS;
    const int64_t block_stride =
        (blocks + OPUS_MOE_SORT_COUNT_BATCH - 1) / OPUS_MOE_SORT_COUNT_BATCH * OPUS_MOE_SORT_COUNT_BATCH;
    AITER_CHECK(counts.dim() == 1 && is_i32(counts, counts.numel()) && counts.numel() >= block_stride * E,
                "moe_sort: counts must be contiguous int32 [E * blocks rounded up to ", OPUS_MOE_SORT_COUNT_BATCH, "]");
    AITER_CHECK(is_i32(num_valid_ids, 1), "moe_sort: num_valid_ids must be int32 [1]");
    for (const aiter_tensor_t* t : {&counts, &reverse_sorted, &sorted_token_ids, &grouped_layout, &group_rows, &num_valid_ids})
        AITER_CHECK(t->device_id == expert_ids.device_id, "moe_sort: all tensors must be on one device");

    HipDeviceGuard device_guard(expert_ids.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    opus_moe_sort_kargs k{};
    k.expert_ids = static_cast<const int*>(expert_ids.data_ptr());
    k.counts = static_cast<int*>(counts.data_ptr());
    k.reverse_sorted = static_cast<int*>(reverse_sorted.data_ptr());
    k.sorted_token_ids = static_cast<int*>(sorted_token_ids.data_ptr());
    k.grouped_layout = static_cast<int*>(grouped_layout.data_ptr());
    k.group_rows = static_cast<int*>(group_rows.data_ptr());
    k.num_valid_ids = static_cast<int*>(num_valid_ids.data_ptr());
    k.tokens = static_cast<int>(N), k.experts = static_cast<int>(E), k.rows = static_cast<int>(R);
    k.blocks = static_cast<int>(blocks);
    // rows launch: the token blocks, then the fill workgroups
    const int64_t rows_grid = blocks + (E + OPUS_MOE_SORT_FILL_EXPERTS - 1) / OPUS_MOE_SORT_FILL_EXPERTS;
    launch(moe_sort_count, blocks, k, stream);
    launch(moe_sort_rows, rows_grid, k, stream);
}

void opus_moe_permute_fwd(aiter_tensor_t& x, aiter_tensor_t& sorted_token_ids, aiter_tensor_t& x_perm) {
    move_rows(moe_permute_fwd, "moe_permute", x, sorted_token_ids, x_perm);
}

void opus_moe_permute_bwd(aiter_tensor_t& d_perm, aiter_tensor_t& reverse_sorted, aiter_tensor_t& dx) {
    move_rows(moe_permute_bwd, "moe_permute backward", d_perm, reverse_sorted, dx);
}

void opus_moe_unpermute_fwd(aiter_tensor_t& y_perm,
                            aiter_tensor_t& reverse_sorted,
                            aiter_tensor_t& weights,
                            aiter_tensor_t& out) {
    move_rows(moe_unpermute_fwd, "moe_unpermute", y_perm, reverse_sorted, out, &weights);
}

void opus_moe_unpermute_bwd(aiter_tensor_t& d_out,
                            aiter_tensor_t& y_perm,
                            aiter_tensor_t& weights,
                            aiter_tensor_t& sorted_token_ids,
                            aiter_tensor_t& d_perm,
                            aiter_tensor_t& d_weights) {
    AITER_CHECK(is_rows(d_out) && is_rows(y_perm) && is_rows(d_perm) && d_out.size(1) == y_perm.size(1) &&
                    d_perm.size(0) == y_perm.size(0) && d_perm.size(1) == y_perm.size(1),
                "moe_unpermute backward: d_out [N, H], y_perm / d_perm [R, H] must be contiguous bf16, H a multiple of 8");
    AITER_CHECK(is_i32(sorted_token_ids, d_perm.size(0)), "moe_unpermute backward: sorted_token_ids must be contiguous int32 [R]");
    AITER_CHECK(is_fp32(weights, d_out.size(0)) && is_fp32(d_weights, d_out.size(0)),
                "moe_unpermute backward: weights / d_weights must be contiguous fp32 [N]");
    for (const aiter_tensor_t* t : {&y_perm, &weights, &sorted_token_ids, &d_perm, &d_weights})
        AITER_CHECK(t->device_id == d_out.device_id, "moe_unpermute backward: one device");
    HipDeviceGuard device_guard(d_out.device_id);

    opus_moe_unpermute_bwd_kargs k{};
    k.d_out = static_cast<const char*>(d_out.data_ptr());
    k.y_perm = static_cast<const char*>(y_perm.data_ptr());
    k.weights = static_cast<const float*>(weights.data_ptr());
    k.sorted_token_ids = static_cast<const int*>(sorted_token_ids.data_ptr());
    k.d_perm = static_cast<char*>(d_perm.data_ptr());
    k.d_weights = static_cast<float*>(d_weights.data_ptr());
    k.chunks = static_cast<int>(d_out.size(1) / VEC);
    k.stream = streams(y_perm);
    launch(moe_unpermute_bwd, d_perm.size(0), k, aiter::getCurrentHIPStream());
}
