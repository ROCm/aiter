// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx950 fallback: the rows select could not finish from their pools (fewer candidates than the
// top-k, or more than the pool holds), rescored densely and selected exactly. Persistent: each
// workgroup takes FB_ROWS listed rows at a time, its waves splitting the tiles of their windows,
// into its own scratch slice, then selects them one by one. Device pass on gfx950 only.
#pragma once
#include "mqa_topk_bf16_gfx950_scan.cuh"
#include "mqa_topk_bf16_select.cuh"

namespace opus_mqa_topk::gfx950 {

template<class T>
__global__ __launch_bounds__(T::BLOCK, 2) void fallback_kernel(mqa_topk_bf16_kargs k) {
    static_assert(T::BLOCK == SEL_BLOCK && FB_ROWS == R,
                  "a workgroup selects; a wave holds the rows");
    __shared__ select_smem sm;
    __shared__ int s_slot;
    const int tid  = opus::thread_id_x();
    const int lane = lane_id();
    const int wave = __builtin_amdgcn_readfirstlane(tid / WAVE);
    const int i = lane % 16, g = lane / 16;
    const int keys = k.num_keys;
    const mqa_topk_ws ws = ws_layout(k.num_rows, keys, k.cap_row, k.fb_slots);
    char* base        = static_cast<char*>(k.ptr_workspace);
    int* ctrl         = reinterpret_cast<int*>(base + ws.fb_ctrl);
    const int* list   = reinterpret_cast<const int*>(base + ws.fb_list);
    float* scratch    = reinterpret_cast<float*>(base + ws.fb_scratch) +
                     (size_t)opus::block_id_x() * FB_ROWS * keys;
    const int count   = ctrl[0];
    const bool values = (k.flags & MQA_TOPK_VALUES) != 0;

    while (true) {
        if (tid == 0) s_slot = atomicAdd(ctrl + 1, FB_ROWS);
        __syncthreads();
        const int s0 = s_slot;
        __syncthreads();
        if (s0 >= count) break;
        int rows[R];
        opus::static_for<R>([&](auto rc) {
            const int at   = s0 + rc.value < count ? s0 + rc.value : count - 1;
            rows[rc.value] = __builtin_amdgcn_readfirstlane(list[at]);
        });

        wave_rows w;
        w.template load<true>(k, rows, lane);
        if (w.ulo < w.uhi) {
            const int kb = w.ulo & ~(TILE - 1);
            const int nt = (w.uhi - kb + TILE - 1) / TILE;
            const int nw = (nt - wave + T::WAVES - 1) / T::WAVES;
            auto g_k = opus::make_gmem(reinterpret_cast<const bf16_t*>(k.ptr_k) +
                                           (size_t)kb * HEAD_DIM,
                                       (unsigned)(keys - kb) * (HEAD_DIM * 2u));
            float* mine = scratch + (size_t)g * keys - w.lo;
            walk(
                w, g_k, nw, lane,
                [&](int t) { return ((wave + T::WAVES * t) * TILE + i) * HEAD_DIM; },
                [&](int t, int, float s) {
                    const int key = kb + (wave + T::WAVES * t) * TILE + i;
                    if (key >= w.lo && key < w.hi) mine[key] = s;
                },
                [](int) {});
        }
        __syncthreads();
        opus::static_for<R>([&](auto rc) {
            constexpr int r = rc.value;
            if constexpr (r > 0) {
                if (rows[r] == rows[r - 1]) return;
            }
            const int row  = rows[r];
            const window wd = clamp_window(k.ptr_row_starts[row], k.ptr_row_ends[row], keys);
            const dense_source src{scratch + (size_t)r * keys, wd.lo, wd.hi - wd.lo, tid};
            int* out_i   = k.ptr_indices + (size_t)row * k.width;
            float* out_v = values ? k.ptr_values + (size_t)row * k.width : nullptr;
            select_row(src, wd.hi - wd.lo, k.width, k.width, wd, out_i, out_v, sm, tid);
            __syncthreads();
        });
    }
}

}  // namespace opus_mqa_topk::gfx950
