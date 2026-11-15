// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// The error bridge requires the tensor header's AITER_CHECK/aiter_detail definitions.
// clang-format off
#include "aiter_tensor.h"
#include "aiter_ctypes_error.h"
// clang-format on
#include "../kernels/mla/nvfp4/quantize.cuh"
#include <cstddef>
#include <initializer_list>

AITER_CTYPES_ERROR_DEF

namespace {
struct MlaNvfp4Args
{
    const void *qn, *qa, *cn, *ca, *indptr, *indices, *last;
    void *out, *lse;
    const void* params;
};
static_assert(sizeof(MlaNvfp4Args) == 80);
#define OFFSET(n, x) static_assert(offsetof(MlaNvfp4Args, n) == x)
OFFSET(qn, 0);
OFFSET(qa, 8);
OFFSET(cn, 16);
OFFSET(ca, 24);
OFFSET(indptr, 32);
OFFSET(indices, 40);
OFFSET(last, 48);
OFFSET(out, 56);
OFFSET(lse, 64);
OFFSET(params, 72);
#undef OFFSET

void tensor(const aiter_tensor_t* t, const char* name, AiterDtype dtype, int dev, size_t align = 1)
{
    AITER_CHECK(t && t->is_gpu() && t->device_id == dev, name, " must be on the query device");
    AITER_CHECK(t->dtype() == dtype, name, " has unsupported dtype");
    AITER_CHECK(t->is_contiguous(), name, " must be contiguous");
    AITER_CHECK(t->numel() == 0 || (t->data_ptr() && uintptr_t(t->data_ptr()) % align == 0),
                name,
                " requires ",
                align,
                "-byte pointer alignment");
}
void shape(const aiter_tensor_t* t, const char* name, std::initializer_list<int64_t> dims)
{
    AITER_CHECK(t->dim() == int(dims.size()), name, " has wrong rank");
    int i = 0;
    for(auto dim : dims)
        AITER_CHECK(t->size(i++) == dim, name, " has wrong shape");
}
int device(const aiter_tensor_t* q)
{
    AITER_CHECK(q && q->is_gpu(), "query must be a GPU tensor");
    return q->device_id;
}
void arch()
{
    AITER_CHECK(get_gpu_arch() == "gfx1250", "R1 NVFP4 MLA requires gfx1250 B0+");
    require_gfx1250_asm_or_throw("mla_nvfp4_n128");
}
int64_t cache(const aiter_tensor_t* cn, const aiter_tensor_t* ca, int dev)
{
    tensor(cn, "kv_nope", AITER_DTYPE_u8, dev, 256);
    tensor(ca, "kv_aux", AITER_DTYPE_u8, dev, 256);
    AITER_CHECK(cn->dim() == 2 && cn->size(0) > 0 && cn->size(0) <= INT32_MAX,
                "cache needs 1..INT32_MAX physical pages");
    int64_t pages = cn->size(0);
    shape(cn, "kv_nope", {pages, 16384});
    shape(ca, "kv_aux", {pages, 8192});
    return pages;
}
} // namespace

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    mla_nvfp4_asm_fwd,
    (aiter_tensor_t * qn,
     aiter_tensor_t* qa,
     aiter_tensor_t* cn,
     aiter_tensor_t* ca,
     aiter_tensor_t* indptr,
     aiter_tensor_t* indices,
     aiter_tensor_t* last,
     aiter_tensor_t* out,
     aiter_tensor_t* lse,
     aiter_tensor_t* params,
     hipStream_t stream),
    (qn, qa, cn, ca, indptr, indices, last, out, lse, params, stream))
{
    const int dev = device(qn);
    const HipDeviceGuard guard(dev);
    arch();
    tensor(qn, "q_nope", AITER_DTYPE_u8, dev, 16);
    tensor(qa, "q_aux", AITER_DTYPE_u8, dev, 16);
    AITER_CHECK(qn->dim() == 3, "q_nope must be [B,128,256]");
    const int64_t b = qn->size(0);
    AITER_CHECK(b < (1LL << 25), "batch exceeds uint32 head indexing");
    shape(qn, "q_nope", {b, 128, 256});
    shape(qa, "q_aux", {b, 128, 128});
    cache(cn, ca, dev);
    tensor(indptr, "kv_indptr", AITER_DTYPE_i32, dev, 4);
    shape(indptr, "kv_indptr", {b + 1});
    tensor(last, "kv_last_page_lens", AITER_DTYPE_i32, dev, 4);
    shape(last, "kv_last_page_lens", {b});
    tensor(indices, "kv_indices", AITER_DTYPE_i32, dev, 4);
    AITER_CHECK(indices->dim() == 1 && indices->numel() <= INT32_MAX,
                "indices require int32 capacity");
    tensor(out, "out", AITER_DTYPE_bf16, dev, 16);
    shape(out, "out", {b, 128, 512});
    tensor(lse, "lse", AITER_DTYPE_fp32, dev, 4);
    shape(lse, "lse", {b, 128});
    tensor(params, "params", AITER_DTYPE_i32, dev, 16);
    shape(params, "params", {4});
    // CSR/params contents are a producer contract; never synchronize to read them here.
    if(b == 0)
        return;
    static SynchronizedCache<int, AiterAsmKernel> kernels;
    auto& kernel = kernels.get_or_create(
        dev, [] { return AiterAsmKernel("mla_nvfp4_n128", "mla_nvfp4/mla_nvfp4_n128.co"); });
    MlaNvfp4Args args{qn->data_ptr(),
                      qa->data_ptr(),
                      cn->data_ptr(),
                      ca->data_ptr(),
                      indptr->data_ptr(),
                      indices->data_ptr(),
                      last->data_ptr(),
                      out->data_ptr(),
                      lse->data_ptr(),
                      params->data_ptr()};
    size_t bytes = sizeof(args);
    kernel.launch_kernel({&args, &bytes, int(b), 1, 1, 128, 1, 1, stream});
}

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    _mla_nvfp4_quantize_q,
    (aiter_tensor_t * q, aiter_tensor_t* qn, aiter_tensor_t* qa, hipStream_t stream),
    (q, qn, qa, stream))
{
    const int dev = device(q);
    const HipDeviceGuard guard(dev);
    arch();
    tensor(q, "q", AITER_DTYPE_bf16, dev, 2);
    AITER_CHECK(q->dim() == 3, "q must be [B,128,576]");
    int64_t b = q->size(0);
    AITER_CHECK(b < (1LL << 25), "batch exceeds uint32 head indexing");
    shape(q, "q", {b, 128, 576});
    tensor(qn, "q_nope", AITER_DTYPE_u8, dev, 16);
    shape(qn, "q_nope", {b, 128, 256});
    tensor(qa, "q_aux", AITER_DTYPE_u8, dev, 16);
    shape(qa, "q_aux", {b, 128, 128});
    if(!b)
        return;
    mla_nvfp4_quantize_q_kernel<<<b * 128, 64, 0, stream>>>(
        static_cast<const uint16_t*>(q->data_ptr()),
        static_cast<uint8_t*>(qn->data_ptr()),
        static_cast<uint8_t*>(qa->data_ptr()));
    HIP_CALL_LAUNCH(hipGetLastError());
}

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(_mla_nvfp4_update_cache,
                                    (aiter_tensor_t * tiles,
                                     aiter_tensor_t* pages,
                                     aiter_tensor_t* groups,
                                     aiter_tensor_t* lengths,
                                     aiter_tensor_t* cn,
                                     aiter_tensor_t* ca,
                                     hipStream_t stream),
                                    (tiles, pages, groups, lengths, cn, ca, stream))
{
    const int dev = device(tiles);
    const HipDeviceGuard guard(dev);
    arch();
    tensor(tiles, "tiles", AITER_DTYPE_bf16, dev, 2);
    AITER_CHECK(tiles->dim() == 3 && tiles->size(0) <= INT32_MAX, "tiles must be [T,16,576]");
    const int64_t n = tiles->size(0), npages = cache(cn, ca, dev);
    shape(tiles, "tiles", {n, 16, 576});
    tensor(pages, "page_ids", AITER_DTYPE_i32, dev, 4);
    shape(pages, "page_ids", {n});
    tensor(groups, "tile_ids", AITER_DTYPE_i32, dev, 4);
    shape(groups, "tile_ids", {n});
    tensor(lengths, "valid_tokens", AITER_DTYPE_i32, dev, 4);
    shape(lengths, "valid_tokens", {n});
    if(!n)
        return;
    mla_nvfp4_update_cache_kernel<<<n, 64, 0, stream>>>(
        static_cast<const uint16_t*>(tiles->data_ptr()),
        static_cast<const int32_t*>(pages->data_ptr()),
        static_cast<const int32_t*>(groups->data_ptr()),
        static_cast<const int32_t*>(lengths->data_ptr()),
        static_cast<uint8_t*>(cn->data_ptr()),
        static_cast<uint8_t*>(ca->data_ptr()),
        npages);
    HIP_CALL_LAUNCH(hipGetLastError());
}
