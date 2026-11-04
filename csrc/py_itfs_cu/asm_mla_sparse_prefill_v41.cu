// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// gfx1250 H64 MLA V4.1: FP8 Q/SWA, NVFP4 CSA, fused split-K reduction.

#include "aiter_tensor.h"

// Must follow aiter_tensor.h for the error-bridge definitions.
#include "aiter_ctypes_error.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>

AITER_CTYPES_ERROR_DEF

namespace {

struct __attribute__((packed)) MlaV41FusedArgs
{
    const void* q;
    const void* q_scale;
    const void* unified_kv;
    const void* unified_kv_scale;
    const void* kv;
    const void* kv_scale;
    const void* attn_sink;
    void* out;
    const void* kv_indptr_prefix;
    const void* kv_indices_prefix;
    const void* kv_indptr_extend;
    const void* kv_indices_extend;
    float softmax_scale;
    uint32_t passes;
    void* partial_r;
    void* partial_lse;
};

static_assert(sizeof(MlaV41FusedArgs) == 120, "fused kernarg size drift");
#define CHECK_OFFSET(field, offset) \
    static_assert(offsetof(MlaV41FusedArgs, field) == offset, "fused kernarg offset drift")
CHECK_OFFSET(q, 0);
CHECK_OFFSET(q_scale, 8);
CHECK_OFFSET(unified_kv, 16);
CHECK_OFFSET(unified_kv_scale, 24);
CHECK_OFFSET(kv, 32);
CHECK_OFFSET(kv_scale, 40);
CHECK_OFFSET(attn_sink, 48);
CHECK_OFFSET(out, 56);
CHECK_OFFSET(kv_indptr_prefix, 64);
CHECK_OFFSET(kv_indices_prefix, 72);
CHECK_OFFSET(kv_indptr_extend, 80);
CHECK_OFFSET(kv_indices_extend, 88);
CHECK_OFFSET(softmax_scale, 96);
CHECK_OFFSET(passes, 100);
CHECK_OFFSET(partial_r, 104);
CHECK_OFFSET(partial_lse, 112);
#undef CHECK_OFFSET

void check_tensor(const aiter_tensor_t* tensor,
                  const char* name,
                  int device,
                  std::initializer_list<AiterDtype> dtypes)
{
    AITER_CHECK(tensor != nullptr, name, " must be a tensor");
    AITER_CHECK(
        tensor->is_gpu() && tensor->device_id == device, name, " must be on the query HIP device");
    AITER_CHECK(
        tensor->data_ptr() != nullptr || tensor->numel() == 0, name, " has a null data pointer");
    AITER_CHECK(tensor->is_contiguous(),
                name,
                " must be contiguous; prepare separate CSA data/scale buffers before calling");
    AITER_CHECK(std::find(dtypes.begin(), dtypes.end(), tensor->dtype()) != dtypes.end(),
                name,
                " has unsupported dtype ",
                AiterDtype_to_str(tensor->dtype()));
}

void check_shape(const aiter_tensor_t* tensor,
                 const char* name,
                 std::initializer_list<int64_t> shape)
{
    AITER_CHECK(tensor->dim() == static_cast<int>(shape.size()), name, " has wrong rank");
    int i = 0;
    for(auto size : shape)
        AITER_CHECK(tensor->size(i++) == size, name, " has wrong shape");
}

void check_csr(const aiter_tensor_t* indices, const aiter_tensor_t* indptr, int64_t n, int device)
{
    check_tensor(indices, "CSR indices", device, {AITER_DTYPE_i32});
    check_tensor(indptr, "CSR indptr", device, {AITER_DTYPE_i32});
    AITER_CHECK(indices->dim() == 1, "CSR indices must be 1-D");
    check_shape(indptr, "CSR indptr", {n + 1});
    AITER_CHECK(indices->numel() <= 0x7fffffffULL, "CSR indices exceed int32 capacity");
    // The caller validates CSR values; no device-to-host synchronization.
}

} // namespace

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(asm_mla_sparse_prefill_v41_fwd,
                                    (aiter_tensor_t * q,
                                     aiter_tensor_t* q_scale,
                                     aiter_tensor_t* unified_kv,
                                     aiter_tensor_t* unified_kv_scale,
                                     aiter_tensor_t* kv_indices_prefix,
                                     aiter_tensor_t* kv_indptr_prefix,
                                     aiter_tensor_t* kv,
                                     aiter_tensor_t* kv_scale,
                                     aiter_tensor_t* kv_indices_extend,
                                     aiter_tensor_t* kv_indptr_extend,
                                     aiter_tensor_t* attn_sink,
                                     aiter_tensor_t* out,
                                     float softmax_scale,
                                     int64_t num_splits,
                                     aiter_tensor_t* partial_r,
                                     aiter_tensor_t* partial_lse,
                                     hipStream_t stream),
                                    (q,
                                     q_scale,
                                     unified_kv,
                                     unified_kv_scale,
                                     kv_indices_prefix,
                                     kv_indptr_prefix,
                                     kv,
                                     kv_scale,
                                     kv_indices_extend,
                                     kv_indptr_extend,
                                     attn_sink,
                                     out,
                                     softmax_scale,
                                     num_splits,
                                     partial_r,
                                     partial_lse,
                                     stream))
{
    AITER_CHECK(q != nullptr && q->is_gpu(), "q must be a HIP tensor");
    HipDeviceGuard guard(q->device_id);
    AITER_CHECK(get_gpu_arch() == "gfx1250", "V4.1 fused asm requires gfx1250");
    const int device = q->device_id;
    check_tensor(q, "q", device, {AITER_DTYPE_fp8});
    AITER_CHECK(q->dim() == 3, "q must have shape [N,64,512]");
    const int64_t n = q->size(0);
    AITER_CHECK(n >= 0 && n <= 65536, "N must be in [0,65536]");
    AITER_CHECK(num_splits >= 1 && num_splits <= 16, "num_splits must be in [1,16]");
    AITER_CHECK(softmax_scale == static_cast<float>(1.0 / std::sqrt(512.0)),
                "V4.1 fused asm requires softmax_scale=1/sqrt(512)");
    check_shape(q, "q", {n, 64, 512});
    check_tensor(q_scale, "q_scale", device, {AITER_DTYPE_u8, AITER_DTYPE_fp8_e8m0});
    check_shape(q_scale, "q_scale", {n, 64, 16});
    check_tensor(unified_kv, "unified_kv", device, {AITER_DTYPE_u8, AITER_DTYPE_fp4x2});
    AITER_CHECK(unified_kv->dim() == 2, "unified_kv must have shape [Pc,256]");
    check_shape(unified_kv, "unified_kv", {unified_kv->size(0), 256});
    check_tensor(unified_kv_scale, "unified_kv_scale", device, {AITER_DTYPE_u8, AITER_DTYPE_fp8});
    check_shape(unified_kv_scale, "unified_kv_scale", {unified_kv->size(0), 32});
    check_tensor(kv, "kv", device, {AITER_DTYPE_fp8});
    AITER_CHECK(kv->dim() == 2, "kv must have shape [Ps,512]");
    check_shape(kv, "kv", {kv->size(0), 512});
    check_tensor(kv_scale, "kv_scale", device, {AITER_DTYPE_u8, AITER_DTYPE_fp8_e8m0});
    check_shape(kv_scale, "kv_scale", {kv->size(0), 16});
    check_csr(kv_indices_prefix, kv_indptr_prefix, n, device);
    check_csr(kv_indices_extend, kv_indptr_extend, n, device);
    check_tensor(out, "out", device, {AITER_DTYPE_bf16});
    check_shape(out, "out", {n, 64, 512});
    if(n == 0)
        return;
    check_tensor(attn_sink, "attn_sink", device, {AITER_DTYPE_fp32});
    check_shape(attn_sink, "attn_sink", {64});
    if(num_splits > 1)
    {
        check_tensor(partial_r, "partial_r", device, {AITER_DTYPE_fp32});
        check_tensor(partial_lse, "partial_lse", device, {AITER_DTYPE_fp32});
        check_shape(partial_r, "partial_r", {n, num_splits, 65, 512});
        check_shape(partial_lse, "partial_lse", {n, num_splits, 64});
    }

    MlaV41FusedArgs args{q->data_ptr(),
                         q_scale->data_ptr(),
                         unified_kv->data_ptr(),
                         unified_kv_scale->data_ptr(),
                         kv->data_ptr(),
                         kv_scale->data_ptr(),
                         attn_sink->data_ptr(),
                         out->data_ptr(),
                         kv_indptr_prefix->data_ptr(),
                         kv_indices_prefix->data_ptr(),
                         kv_indptr_extend->data_ptr(),
                         kv_indices_extend->data_ptr(),
                         softmax_scale,
                         static_cast<uint32_t>(num_splits),
                         num_splits > 1 ? partial_r->data_ptr() : nullptr,
                         num_splits > 1 ? partial_lse->data_ptr() : nullptr};
    size_t arg_size = sizeof(args);
    static AiterAsmKernel kernel("mla_v41_fused_cs_v1", "mla_v41/mla_v41_fused_cs_v1.co");
    const int s = static_cast<int>(num_splits);
    // Query is grid.y even for S1. S>1 merges within one cluster and one launch.
    kernel.launch_kernel({&args, &arg_size, s, static_cast<int>(n), 1, 128, 1, 1, stream, s, 1, 1});
}
