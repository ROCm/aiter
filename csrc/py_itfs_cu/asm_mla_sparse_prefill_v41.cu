// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// gfx1250 H64 BF16-Q RoPE/quant + mixed MLA + FP8-O inverse-RoPE/quant.
#include "aiter_tensor.h"
#include "aiter_ctypes_error.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <utility>

AITER_CTYPES_ERROR_DEF

namespace {
struct __attribute__((packed)) MlaV41FusedArgs
{
    const void* q_nope;
    const void* cos;
    const void* sin;
    const void* positions;
    const void* unified_kv;
    const void* unified_kv_scale;
    const void* kv;
    const void* kv_scale;
    const void* kv_indptr_prefix;
    const void* kv_indices_prefix;
    const void* kv_indptr_extend;
    const void* kv_indices_extend;
    const void* attn_sink;
    float softmax_scale;
    uint32_t passes;
    void* out;
    void* out_scale;
    void* partial_r;
    void* partial_lse;
};
static_assert(sizeof(void*) == 8, "64-bit pointer ABI required");
static_assert(sizeof(MlaV41FusedArgs) == 144, "Q/O kernarg size drift");
static_assert(offsetof(MlaV41FusedArgs, q_nope) == 0, "q_nope offset drift");
static_assert(offsetof(MlaV41FusedArgs, cos) == 8, "cos offset drift");
static_assert(offsetof(MlaV41FusedArgs, sin) == 16, "sin offset drift");
static_assert(offsetof(MlaV41FusedArgs, positions) == 24, "positions offset drift");
static_assert(offsetof(MlaV41FusedArgs, unified_kv) == 32, "unified_kv offset drift");
static_assert(offsetof(MlaV41FusedArgs, unified_kv_scale) == 40, "unified_kv_scale offset drift");
static_assert(offsetof(MlaV41FusedArgs, kv) == 48, "kv offset drift");
static_assert(offsetof(MlaV41FusedArgs, kv_scale) == 56, "kv_scale offset drift");
static_assert(offsetof(MlaV41FusedArgs, kv_indptr_prefix) == 64, "kv_indptr_prefix offset drift");
static_assert(offsetof(MlaV41FusedArgs, kv_indices_prefix) == 72, "kv_indices_prefix offset drift");
static_assert(offsetof(MlaV41FusedArgs, kv_indptr_extend) == 80, "kv_indptr_extend offset drift");
static_assert(offsetof(MlaV41FusedArgs, kv_indices_extend) == 88, "kv_indices_extend offset drift");
static_assert(offsetof(MlaV41FusedArgs, attn_sink) == 96, "attn_sink offset drift");
static_assert(offsetof(MlaV41FusedArgs, softmax_scale) == 104, "softmax_scale offset drift");
static_assert(offsetof(MlaV41FusedArgs, passes) == 108, "passes offset drift");
static_assert(offsetof(MlaV41FusedArgs, out) == 112, "out offset drift");
static_assert(offsetof(MlaV41FusedArgs, out_scale) == 120, "out_scale offset drift");
static_assert(offsetof(MlaV41FusedArgs, partial_r) == 128, "partial_r offset drift");
static_assert(offsetof(MlaV41FusedArgs, partial_lse) == 136, "partial_lse offset drift");

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


bool overlaps(const aiter_tensor_t* a, const aiter_tensor_t* b)
{
    if(a == nullptr || b == nullptr || a->numel() == 0 || b->numel() == 0)
        return false;
    const auto ab = reinterpret_cast<uintptr_t>(a->data_ptr());
    const auto bb = reinterpret_cast<uintptr_t>(b->data_ptr());
    return ab < bb + b->numel() * b->element_size() &&
           bb < ab + a->numel() * a->element_size();
}

using NamedTensor = std::pair<const char*, const aiter_tensor_t*>;
void check_disjoint(std::initializer_list<NamedTensor> reads,
                    std::initializer_list<NamedTensor> writes)
{
    for(auto w = writes.begin(); w != writes.end(); ++w)
    {
        for(const auto& r : reads)
            AITER_CHECK(!overlaps(w->second, r.second), w->first, " must not overlap ", r.first);
        for(auto other = writes.begin(); other != w; ++other)
            AITER_CHECK(!overlaps(w->second, other->second), w->first, " must not overlap ", other->first);
    }
}
} // namespace

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(asm_mla_sparse_prefill_v41_fwd,
    (aiter_tensor_t* q_nope,
     aiter_tensor_t* cos,
     aiter_tensor_t* sin,
     aiter_tensor_t* positions,
     aiter_tensor_t* unified_kv,
     aiter_tensor_t* unified_kv_scale,
     aiter_tensor_t* kv,
     aiter_tensor_t* kv_scale,
     aiter_tensor_t* kv_indptr_prefix,
     aiter_tensor_t* kv_indices_prefix,
     aiter_tensor_t* kv_indptr_extend,
     aiter_tensor_t* kv_indices_extend,
     aiter_tensor_t* attn_sink,
     float softmax_scale,
     int64_t num_splits,
     aiter_tensor_t* out,
     aiter_tensor_t* out_scale,
     aiter_tensor_t* partial_r,
     aiter_tensor_t* partial_lse,
     hipStream_t stream),
    (q_nope, cos, sin, positions, unified_kv, unified_kv_scale, kv, kv_scale, kv_indptr_prefix, kv_indices_prefix, kv_indptr_extend, kv_indices_extend, attn_sink, softmax_scale, num_splits, out, out_scale, partial_r, partial_lse, stream))
{
    AITER_CHECK(q_nope != nullptr && q_nope->is_gpu(), "q_nope must be a HIP tensor");
    HipDeviceGuard guard(q_nope->device_id);
    AITER_CHECK(get_gpu_arch() == "gfx1250", "V4.1 Q/O fused asm requires gfx1250");
    const int device = q_nope->device_id;
    check_tensor(q_nope, "q_nope", device, {AITER_DTYPE_bf16});
    AITER_CHECK(q_nope->dim() == 3, "q_nope must have shape [N,64,512]");
    const int64_t n = q_nope->size(0);
    AITER_CHECK(n >= 0 && n <= 65535, "N must be in [0,65535]");
    AITER_CHECK(num_splits >= 1 && num_splits <= 16, "num_splits must be in [1,16]");
    AITER_CHECK(softmax_scale == static_cast<float>(1.0 / std::sqrt(512.0)),
                "Q/O fused asm requires softmax_scale=1/sqrt(512)");
    check_shape(q_nope, "q_nope", {n,64,512});
    check_tensor(cos, "cos", device, {AITER_DTYPE_bf16});
    check_tensor(sin, "sin", device, {AITER_DTYPE_bf16});
    AITER_CHECK(cos->dim() == 2, "cos must have shape [P,32]");
    check_shape(cos, "cos", {cos->size(0),32});
    check_shape(sin, "sin", {cos->size(0),32});
    AITER_CHECK(n == 0 || cos->size(0) > 0, "nonempty Q requires a coefficient table");
    check_tensor(positions, "positions", device, {AITER_DTYPE_i64});
    check_shape(positions, "positions", {n});
    // Caller validates position and CSR values; no device-to-host reads here.
    check_tensor(unified_kv, "unified_kv", device, {AITER_DTYPE_u8,AITER_DTYPE_fp4x2});
    AITER_CHECK(unified_kv->dim() == 2, "unified_kv must have shape [Pc,256]");
    check_shape(unified_kv, "unified_kv", {unified_kv->size(0),256});
    check_tensor(unified_kv_scale, "unified_kv_scale", device, {AITER_DTYPE_u8,AITER_DTYPE_fp8});
    check_shape(unified_kv_scale, "unified_kv_scale", {unified_kv->size(0),32});
    check_tensor(kv, "kv", device, {AITER_DTYPE_fp8});
    AITER_CHECK(kv->dim() == 2, "kv must have shape [Ps,512]");
    check_shape(kv, "kv", {kv->size(0),512});
    check_tensor(kv_scale, "kv_scale", device, {AITER_DTYPE_u8,AITER_DTYPE_fp8_e8m0});
    check_shape(kv_scale, "kv_scale", {kv->size(0),16});
    check_csr(kv_indices_prefix, kv_indptr_prefix, n, device);
    check_csr(kv_indices_extend, kv_indptr_extend, n, device);
    check_tensor(out, "out", device, {AITER_DTYPE_fp8});
    check_shape(out, "out", {n,64,512});
    check_tensor(out_scale, "out_scale", device, {AITER_DTYPE_u8});
    check_shape(out_scale, "out_scale", {n,64,16});
    if(n == 0)
        return;
    check_tensor(attn_sink, "attn_sink", device, {AITER_DTYPE_fp32});
    check_shape(attn_sink, "attn_sink", {64});
    if(num_splits > 1)
    {
        check_tensor(partial_r, "partial_r", device, {AITER_DTYPE_fp32});
        check_tensor(partial_lse, "partial_lse", device, {AITER_DTYPE_fp32});
        check_shape(partial_r, "partial_r", {n,num_splits,65,512});
        check_shape(partial_lse, "partial_lse", {n,num_splits,64});
    }
    check_disjoint(
        {{"q_nope",q_nope},{"cos",cos},{"sin",sin},{"positions",positions},
         {"unified_kv",unified_kv},{"unified_kv_scale",unified_kv_scale},
         {"kv",kv},{"kv_scale",kv_scale},
         {"kv_indptr_prefix",kv_indptr_prefix},{"kv_indices_prefix",kv_indices_prefix},
         {"kv_indptr_extend",kv_indptr_extend},{"kv_indices_extend",kv_indices_extend},
         {"attn_sink",attn_sink}},
        {{"out",out},{"out_scale",out_scale},
         {"partial_r",num_splits > 1 ? partial_r : nullptr},
         {"partial_lse",num_splits > 1 ? partial_lse : nullptr}});

    MlaV41FusedArgs args{
        q_nope->data_ptr(),
        cos->data_ptr(),
        sin->data_ptr(),
        positions->data_ptr(),
        unified_kv->data_ptr(),
        unified_kv_scale->data_ptr(),
        kv->data_ptr(),
        kv_scale->data_ptr(),
        kv_indptr_prefix->data_ptr(),
        kv_indices_prefix->data_ptr(),
        kv_indptr_extend->data_ptr(),
        kv_indices_extend->data_ptr(),
        attn_sink->data_ptr(),
        softmax_scale,
        static_cast<uint32_t>(num_splits),
        out->data_ptr(),
        out_scale->data_ptr(),
        num_splits > 1 ? partial_r->data_ptr() : nullptr,
        num_splits > 1 ? partial_lse->data_ptr() : nullptr,
    };
    size_t arg_size = sizeof(args);
    // Exact adopted CO SHA256: 89624d57a011ce9d9b88ef3ff51025f51eacf1949274afbbf331643f6bd535df.
    // Keep the packaged path; the new ELF export is intentionally different.
    static AiterAsmKernel kernel("mla_v41_prologue_cleanup_fallback_merge_v1",
        "mla_v41/fused_mla_a8w4_qh64_1tg_16mx4_64nx1_sparse_np_v41.co");
    const int s = static_cast<int>(num_splits);
    kernel.launch_kernel({&args,&arg_size,s,static_cast<int>(n),1,128,1,1,stream,s,1,1});
}
