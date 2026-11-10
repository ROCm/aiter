// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

// Pointer/scalar entry points of the NVFP4 MoE kernels, defined in
// moe_nvfp4_entry.cu. Return values are hipError_t codes, or the kernel
// file's own codes: -1 kernel index out of range, -2 sort shape over its
// limits, -3 kernel built for a different K.

#include <hip/hip_runtime.h>

// Fields of a kernel's Cfg filled by moe_nvfp4_entry_cfg_info, by index.
enum CfgField
{
    CFG_S1,
    CFG_M_TILES,
    CFG_N_TILES,
    CFG_WAVES,
    CFG_GS,
    CFG_INBF16,
    CFG_OUTBF16,
    CFG_NUM_FIELDS
};

int moe_nvfp4_entry_num_cfgs();
const char* moe_nvfp4_entry_cfg_name(int i);
// out: CFG_NUM_FIELDS ints
void moe_nvfp4_entry_cfg_info(int i, int* out);
int moe_nvfp4_entry_cfg_fits(int i, int K);
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
                         hipStream_t stream);
int moe_nvfp4_entry_gemm(int kernel,
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
                         hipStream_t stream);
