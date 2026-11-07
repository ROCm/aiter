// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "moe_nvfp4.h"
#include "moe_nvfp4_entry.h"

namespace aiter {

int64_t moe_nvfp4_num_cfgs() { return moe_nvfp4_entry_num_cfgs(); }

std::string moe_nvfp4_cfg_name(int64_t i)
{
    AITER_CHECK(i >= 0 && i < moe_nvfp4_entry_num_cfgs(), "cfg index out of range");
    return moe_nvfp4_entry_cfg_name((int)i);
}

int64_t moe_nvfp4_cfg_field(int64_t i, int64_t k)
{
    AITER_CHECK(i >= 0 && i < moe_nvfp4_entry_num_cfgs(), "cfg index out of range");
    AITER_CHECK(k >= 0 && k < 6, "cfg field out of range");
    int v[6];
    moe_nvfp4_entry_cfg_info((int)i, v);
    return v[k];
}

void moe_nvfp4_sort(const aiter_tensor_t& x,
                    aiter_tensor_t& x_f16,
                    aiter_tensor_t& out,
                    const aiter_tensor_t& topk_ids,
                    const aiter_tensor_t& topk_w,
                    aiter_tensor_t& sorted_ids,
                    aiter_tensor_t& sorted_w,
                    aiter_tensor_t& sorted_eids,
                    aiter_tensor_t& num_valid,
                    int64_t num_experts,
                    int64_t block_m,
                    int64_t nblk)
{
    AITER_CHECK(x.dtype() == AITER_DTYPE_bf16 && x.dim() == 2 && x.is_contiguous(),
                "x must be contiguous bf16 [T, H]");
    AITER_CHECK(x_f16.dtype() == AITER_DTYPE_fp16 &&
                    (x_f16.numel() == x.numel() || x_f16.numel() == 0),
                "x_f16 must be f16 [T, H] (empty: no cast)");
    AITER_CHECK(out.dtype() == AITER_DTYPE_bf16 && out.numel() == x.numel(),
                "out must be bf16 [T, H]");
    AITER_CHECK(topk_ids.dtype() == AITER_DTYPE_i32 && topk_ids.dim() == 2 &&
                    topk_ids.is_contiguous() && topk_ids.size(0) == x.size(0),
                "topk_ids must be contiguous int32 [T, topk]");
    AITER_CHECK(topk_w.dtype() == AITER_DTYPE_fp32 && topk_w.numel() == topk_ids.numel(),
                "topk_w must be fp32 [T, topk]");
    AITER_CHECK(sorted_ids.numel() >= (size_t)(nblk * block_m) &&
                    sorted_w.numel() >= (size_t)(nblk * block_m) &&
                    sorted_eids.numel() >= (size_t)nblk && num_valid.numel() >= 1,
                "sort buffers too small");
    HipDeviceGuard device_guard(x.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    const int rc             = moe_nvfp4_entry_sort(x.data_ptr(),
                                                    x_f16.numel() ? x_f16.data_ptr() : nullptr,
                                                    out.data_ptr(),
                                                    topk_ids.data_ptr(),
                                                    topk_w.data_ptr(),
                                                    sorted_ids.data_ptr(),
                                                    sorted_w.data_ptr(),
                                                    sorted_eids.data_ptr(),
                                                    num_valid.data_ptr(),
                                                    (int)x.size(0),
                                                    (int)x.size(1),
                                                    (int)topk_ids.size(1),
                                                    (int)num_experts,
                                                    (int)block_m,
                                                    (int)nblk,
                                                    stream);
    AITER_CHECK(rc == 0, "moe_nvfp4 sort failed: ", rc);
}

void moe_nvfp4_gemm(int64_t cfg,
                    const aiter_tensor_t& x,
                    const aiter_tensor_t& w,
                    const aiter_tensor_t& w_sc,
                    const aiter_tensor_t& inv_gs,
                    const aiter_tensor_t& sorted_ids,
                    const aiter_tensor_t& sorted_w,
                    const aiter_tensor_t& sorted_eids,
                    const aiter_tensor_t& num_valid,
                    aiter_tensor_t& out,
                    int64_t T,
                    int64_t topk,
                    int64_t K,
                    int64_t ntiles,
                    int64_t n_out,
                    int64_t grid_x,
                    int64_t grid_y)
{
    AITER_CHECK(x.is_contiguous() && w.is_contiguous() && w_sc.is_contiguous() &&
                    out.is_contiguous(),
                "x, w, w_sc and out must be contiguous");
    AITER_CHECK(inv_gs.dtype() == AITER_DTYPE_fp32 && inv_gs.is_contiguous(),
                "inv_gs must be contiguous fp32");
    HipDeviceGuard device_guard(x.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    const int rc             = moe_nvfp4_entry_gemm((int)cfg,
                                                    x.data_ptr(),
                                                    w.data_ptr(),
                                                    w_sc.data_ptr(),
                                                    inv_gs.data_ptr(),
                                                    sorted_ids.data_ptr(),
                                                    sorted_w.data_ptr(),
                                                    sorted_eids.data_ptr(),
                                                    num_valid.data_ptr(),
                                                    out.data_ptr(),
                                                    (int)T,
                                                    (int)topk,
                                                    (int)K,
                                                    (int)ntiles,
                                                    (int)n_out,
                                                    (int)grid_x,
                                                    (int)grid_y,
                                                    stream);
    AITER_CHECK(rc == 0, "moe_nvfp4 gemm ", moe_nvfp4_cfg_name(cfg), " failed: ", rc);
}

} // namespace aiter
