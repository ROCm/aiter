// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Compiles the MiniMax-M3 NVFP4 MoE kernels (moe_nvfp4_kernels.hip) and wraps
// their struct-based launchers in pointer/scalar entry points, so the
// aiter_tensor_t layer (moe_nvfp4.cu) never needs the kernel file's Params /
// PrepParams layouts. This TU includes no aiter or ck_tile header on purpose: the
// kernel file defines short macros (DEV, KERNEL, ...) and vector typedefs that are not
// meant to share a TU with them.
#include "moe_nvfp4_kernels.hip"

#include "moe_nvfp4_entry.h"

int moe_nvfp4_entry_num_cfgs() { return moe_nvfp4_kernels_num_cfgs(); }

const char* moe_nvfp4_entry_cfg_name(int i) { return moe_nvfp4_kernels_cfg_name(i); }

void moe_nvfp4_entry_cfg_info(int i, int* out) { moe_nvfp4_kernels_cfg_info(i, out); }

int moe_nvfp4_entry_sort(const void* x,
                         void* x_f16,
                         void* out,
                         const void* ids,
                         const void* w,
                         void* sorted_ids,
                         void* sorted_w,
                         void* sorted_eids,
                         void* num_valid,
                         int T,
                         int H,
                         int topk,
                         int E,
                         int BM,
                         int nblk,
                         hipStream_t stream)
{
    PrepParams p{};
    p.x           = (const uint16_t*)x;
    p.x_f16       = (uint8_t*)x_f16;
    p.out         = (uint16_t*)out;
    p.ids         = (const int*)ids;
    p.w           = (const float*)w;
    p.sorted_ids  = (int*)sorted_ids;
    p.sorted_w    = (float*)sorted_w;
    p.sorted_eids = (int*)sorted_eids;
    p.num_valid   = (int*)num_valid;
    p.T           = T;
    p.H           = H;
    p.topk        = topk;
    p.E           = E;
    p.BM          = BM;
    p.nblk        = nblk;
    return moe_nvfp4_kernels_sort(&p, stream);
}

int moe_nvfp4_entry_gemm(int cfg,
                         const void* x,
                         const void* w,
                         const void* w_sc,
                         const void* inv_gs,
                         const void* sorted_ids,
                         const void* sorted_w,
                         const void* sorted_eids,
                         const void* num_valid,
                         void* out,
                         int T,
                         int topk,
                         int K,
                         int ntiles,
                         int n_out,
                         int grid_x,
                         int grid_y,
                         hipStream_t stream)
{
    // Same field values as the reference ctypes host code: no zeroing.
    Params p{};
    p.x           = (const uint8_t*)x;
    p.w           = (const uint8_t*)w;
    p.w_sc        = (const uint8_t*)w_sc;
    p.inv_gs      = (const float*)inv_gs;
    p.sorted_ids  = (const int*)sorted_ids;
    p.sorted_w    = (const float*)sorted_w;
    p.sorted_eids = (const int*)sorted_eids;
    p.num_valid   = (const int*)num_valid;
    p.out         = (uint8_t*)out;
    p.T           = T;
    p.topk        = topk;
    p.K           = K;
    p.ntiles      = ntiles;
    p.n_out       = n_out;
    return moe_nvfp4_kernels_launch(cfg, &p, grid_x, grid_y, stream);
}
