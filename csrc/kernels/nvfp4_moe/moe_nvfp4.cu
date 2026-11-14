// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// ctypes entry points of an NVFP4 MoE JIT module (aiter/ops/moe_nvfp4.py): the
// kernel-table queries and the sort, reduce and GEMM launches on aiter_tensor_t.
#include "aiter_tensor.h"

#include "aiter_ctypes_error.h"

#include "aiter_hip_common.h"
#include "moe_nvfp4_entry.h"
#include <cstring>

AITER_CTYPES_ERROR_DEF

static void check_kernel(int64_t i)
{
    AITER_CHECK(i >= 0 && i < moe_nvfp4_entry_num_cfgs(), "kernel index out of range");
}

static int cfg_field(int64_t i, int k)
{
    int v[CFG_NUM_FIELDS];
    moe_nvfp4_entry_cfg_info((int)i, v);
    return v[k];
}

// Index of kernel name in this module's kernel table, -1 if it has none
AITER_CTYPES_DEFINE_ENTRYPOINT(moe_nvfp4_cfg_index,
                               (const char* name, hipStream_t stream),
                               (name, stream))
{
    (void)stream;
    for(int i = 0; i < moe_nvfp4_entry_num_cfgs(); ++i)
        if(std::strcmp(moe_nvfp4_entry_cfg_name(i), name) == 0)
            return i;
    return -1;
}

AITER_CTYPES_DEFINE_ENTRYPOINT(moe_nvfp4_cfg_field,
                               (int64_t i, int64_t k, hipStream_t stream),
                               (i, k, stream))
{
    (void)stream;
    check_kernel(i);
    AITER_CHECK(k >= 0 && k < CFG_NUM_FIELDS, "kernel field out of range");
    return cfg_field(i, (int)k);
}

AITER_CTYPES_DEFINE_ENTRYPOINT(moe_nvfp4_cfg_fits,
                               (int64_t i, int64_t K, hipStream_t stream),
                               (i, K, stream))
{
    (void)stream;
    check_kernel(i);
    return moe_nvfp4_entry_cfg_fits((int)i, (int)K) != 0;
}

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(moe_nvfp4_sort,
                                    (aiter_tensor_t * x,
                                     aiter_tensor_t* x_f16,
                                     aiter_tensor_t* out,
                                     aiter_tensor_t* topk_ids,
                                     aiter_tensor_t* topk_w,
                                     aiter_tensor_t* sorted_ids,
                                     aiter_tensor_t* sorted_w,
                                     aiter_tensor_t* sorted_eids,
                                     aiter_tensor_t* num_valid,
                                     int64_t num_experts,
                                     int64_t block_m,
                                     int64_t nblk,
                                     hipStream_t stream),
                                    (x,
                                     x_f16,
                                     out,
                                     topk_ids,
                                     topk_w,
                                     sorted_ids,
                                     sorted_w,
                                     sorted_eids,
                                     num_valid,
                                     num_experts,
                                     block_m,
                                     nblk,
                                     stream))
{
    const bool bf16 = x->dtype() == AITER_DTYPE_bf16;
    AITER_CHECK((bf16 || x->dtype() == AITER_DTYPE_fp16) && x->dim() == 2 && x->is_contiguous(),
                "x must be contiguous bf16 or f16 [T, H]");
    AITER_CHECK(x_f16->dtype() == AITER_DTYPE_fp16 &&
                    ((bf16 && x_f16->numel() == x->numel()) || x_f16->numel() == 0),
                "x_f16 must be f16 [T, H] for bf16 x (empty: no cast)");
    AITER_CHECK(out->is_contiguous() && (out->numel() == 0 || (out->numel() == x->numel() &&
                                                               (out->dtype() == x->dtype() ||
                                                                out->dtype() == AITER_DTYPE_fp32))),
                "out must be empty or [T, H] of the x dtype or fp32");
    AITER_CHECK(topk_ids->dtype() == AITER_DTYPE_i32 && topk_ids->dim() == 2 &&
                    topk_ids->is_contiguous() && topk_ids->size(0) == x->size(0),
                "topk_ids must be contiguous int32 [T, topk]");
    AITER_CHECK(topk_w->dtype() == AITER_DTYPE_fp32 && topk_w->numel() == topk_ids->numel(),
                "topk_w must be fp32 [T, topk]");
    AITER_CHECK(sorted_ids->numel() >= (size_t)(nblk * block_m) &&
                    sorted_w->numel() >= (size_t)(nblk * block_m) &&
                    sorted_eids->numel() >= (size_t)nblk && num_valid->numel() >= 1,
                "sort buffers too small");
    HipDeviceGuard device_guard(x->device_id);
    const int rc = moe_nvfp4_entry_sort(x->data_ptr(),
                                        x_f16->numel() ? x_f16->data_ptr() : nullptr,
                                        out->data_ptr(),
                                        (int)(out->numel() * out->element_size() / 16),
                                        topk_ids->data_ptr(),
                                        topk_w->data_ptr(),
                                        sorted_ids->data_ptr(),
                                        sorted_w->data_ptr(),
                                        sorted_eids->data_ptr(),
                                        num_valid->data_ptr(),
                                        (int)x->size(0),
                                        (int)x->size(1),
                                        (int)topk_ids->size(1),
                                        (int)num_experts,
                                        (int)block_m,
                                        (int)nblk,
                                        stream);
    AITER_CHECK(rc == 0, "moe_nvfp4 sort failed: ", rc);
}

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    moe_nvfp4_reduce,
    (aiter_tensor_t * ws, aiter_tensor_t* out, int64_t topk, hipStream_t stream),
    (ws, out, topk, stream))
{
    const bool bf16 = out->dtype() == AITER_DTYPE_bf16;
    AITER_CHECK((bf16 || out->dtype() == AITER_DTYPE_fp16) && out->dim() == 2 &&
                    out->is_contiguous() && out->size(1) % 8 == 0,
                "out must be contiguous bf16 or f16 [T, H], H a multiple of 8");
    AITER_CHECK(ws->dtype() == AITER_DTYPE_fp32 && ws->is_contiguous() && topk > 0 &&
                    ws->numel() == out->numel() * (size_t)topk,
                "ws must be contiguous fp32 [T * topk, H]");
    HipDeviceGuard device_guard(out->device_id);
    const int rc = moe_nvfp4_entry_reduce(ws->data_ptr(),
                                          out->data_ptr(),
                                          (int)out->size(0),
                                          (int)out->size(1),
                                          (int)topk,
                                          bf16,
                                          stream);
    AITER_CHECK(rc == 0, "moe_nvfp4 reduce failed: ", rc);
}

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(moe_nvfp4_gemm,
                                    (int64_t kernel,
                                     aiter_tensor_t* x,
                                     aiter_tensor_t* w,
                                     aiter_tensor_t* w_sc,
                                     aiter_tensor_t* inv_gs,
                                     aiter_tensor_t* sorted_ids,
                                     aiter_tensor_t* sorted_w,
                                     aiter_tensor_t* sorted_eids,
                                     aiter_tensor_t* num_valid,
                                     aiter_tensor_t* out,
                                     int64_t T,
                                     int64_t topk,
                                     int64_t K,
                                     int64_t ntiles,
                                     int64_t n_out,
                                     int64_t grid_x,
                                     int64_t grid_y,
                                     hipStream_t stream),
                                    (kernel,
                                     x,
                                     w,
                                     w_sc,
                                     inv_gs,
                                     sorted_ids,
                                     sorted_w,
                                     sorted_eids,
                                     num_valid,
                                     out,
                                     T,
                                     topk,
                                     K,
                                     ntiles,
                                     n_out,
                                     grid_x,
                                     grid_y,
                                     stream))
{
    check_kernel(kernel);
    AITER_CHECK(x->is_contiguous() && w->is_contiguous() && w_sc->is_contiguous() &&
                    out->is_contiguous(),
                "x, w, w_sc and out must be contiguous");
    AITER_CHECK(inv_gs->dtype() == AITER_DTYPE_fp32 && inv_gs->is_contiguous(),
                "inv_gs must be contiguous fp32");
    // stage 1 reads bf16 x iff _inbf16 (else f16), stage 2 writes bf16 out iff _outbf16 (else
    // f32: the workspace of f16 output, or the [T * topk, H] rows of _reduce)
    const bool s1            = cfg_field(kernel, CFG_S1) != 0;
    const aiter_tensor_t* io = s1 ? x : out;
    const bool io_bf16       = cfg_field(kernel, s1 ? CFG_INBF16 : CFG_OUTBF16) != 0;
    const auto io_other      = s1 ? AITER_DTYPE_fp16 : AITER_DTYPE_fp32;
    const char* name         = moe_nvfp4_entry_cfg_name((int)kernel);
    AITER_CHECK(io->dtype() == (io_bf16 ? AITER_DTYPE_bf16 : io_other),
                s1 ? "x" : "out",
                " dtype does not match kernel ",
                name);
    HipDeviceGuard device_guard(x->device_id);
    const int rc = moe_nvfp4_entry_gemm((int)kernel,
                                        x->data_ptr(),
                                        w->data_ptr(),
                                        w_sc->data_ptr(),
                                        inv_gs->data_ptr(),
                                        sorted_ids->data_ptr(),
                                        sorted_w->data_ptr(),
                                        sorted_eids->data_ptr(),
                                        num_valid->data_ptr(),
                                        out->data_ptr(),
                                        (int)T,
                                        (int)topk,
                                        (int)K,
                                        (int)ntiles,
                                        (int)n_out,
                                        (int)grid_x,
                                        (int)grid_y,
                                        stream);
    AITER_CHECK(rc == 0, "moe_nvfp4 gemm ", name, " failed: ", rc);
}
