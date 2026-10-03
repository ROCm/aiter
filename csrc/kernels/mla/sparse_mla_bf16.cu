// SPDX-License-Identifier: MIT
// BF16, D512 sparse MLA for gfx950. v1: full LDS; v2: hybrid + unrolled reducer.
// aiter_ctypes_error.h requires the common declarations supplied by aiter_tensor.h.
// clang-format off
#include "aiter_tensor.h"
#include "aiter_ctypes_error.h"
#include "hk/sparse_mla_bf16_wave4.cuh"
#include "hk/sparse_mla_bf16_headtiles.cuh"
#include "hk/sparse_mla_bf16_reduce.cuh"
// clang-format on
#include <climits>
#include <cmath>

AITER_CTYPES_ERROR_DEF

namespace sparse_mla_bf16 {
#include "hk/sparse_mla_bf16_full_lds.cuh"

template <int Heads, bool Split>
__global__ __launch_bounds__(Heads == 64 ? 512 : 64)
    __attribute__((amdgpu_num_vgpr(40))) void sparse_mla_v1(Params p)
{
    if constexpr(Heads == 64)
        full_lds_h64<Split>(p);
    else
        sparse_mla_legacy<Heads, Split>(p);
}

template <int Heads, bool Split>
hipError_t launch_main(Params p, int version, hipStream_t stream)
{
    const dim3 grid(p.queries, p.splits);
    if(version == 1)
    {
        constexpr int threads = Heads == 64 ? 512 : 64;
        constexpr int lds     = Heads == 64 ? FullLdsTraits::kLdsBytes : Traits<16>::kLdsBytes;
        hipLaunchKernelGGL((sparse_mla_v1<Heads, Split>), grid, dim3(threads), lds, stream, p);
    }
    else if(Heads == 64 && uint64_t(p.queries) * p.splits <= 128u)
    {
        hipLaunchKernelGGL((sparse_mla_headtiles<64, Split>),
                           dim3(p.queries, p.splits, 4),
                           dim3(256),
                           headtiles::Traits::kLdsBytes,
                           stream,
                           p);
    }
    else
    {
        constexpr int lds = Heads == 16 ? wave4::Traits::kLdsBytes : Traits<64>::kLdsBytes;
        hipLaunchKernelGGL((sparse_mla_v2<Heads, Split>), grid, dim3(256), lds, stream, p);
    }
    return hipGetLastError();
}

template <int Heads>
hipError_t launch(Params p, int version, hipStream_t stream)
{
    if(p.splits == 1)
        return launch_main<Heads, false>(p, version, stream);
    hipError_t status = launch_main<Heads, true>(p, version, stream);
    if(status != hipSuccess)
        return status;
    if constexpr(Heads == 16)
    {
        if(version == 1)
        {
            hipLaunchKernelGGL(reduce, dim3(p.queries, Heads), dim3(256), 0, stream, p, Heads);
            return hipGetLastError();
        }
    }
    return launch_reduce64<Heads>(p, stream);
}
} // namespace sparse_mla_bf16

namespace {
void check_tensor(const aiter_tensor_t* t, const char* name, AiterDtype dtype, int device)
{
    AITER_CHECK(t && t->is_gpu() && t->device_id == device, name, ": expected the query device");
    AITER_CHECK(t->dtype() == dtype, name, ": unexpected dtype");
    AITER_CHECK(t->numel() == 0 || t->data_ptr(), name, ": null storage");
}

void check_shape(const aiter_tensor_t* t, const char* name, std::initializer_list<int64_t> shape)
{
    AITER_CHECK(t->dim() == int(shape.size()), name, ": unexpected rank");
    int d = 0;
    for(int64_t extent : shape)
        AITER_CHECK(t->size(d++) == extent, name, ": unexpected shape");
}

// Conservatively reject overlapping storage spans. Inputs can alias each other;
// writable tensors must not alias inputs or another writable tensor.
uintptr_t end_address(const aiter_tensor_t* t)
{
    if(!t || !t->numel())
        return 0;
    size_t span = 1;
    for(int d = 0; d < t->dim(); ++d)
        span += (t->size(d) - 1) * t->stride(d);
    return reinterpret_cast<uintptr_t>(t->data_ptr()) + span * t->element_size();
}
void check_no_overlap(const aiter_tensor_t* a, const aiter_tensor_t* b)
{
    if(!a || !b || !a->numel() || !b->numel())
        return;
    AITER_CHECK(reinterpret_cast<uintptr_t>(a->data_ptr()) >= end_address(b) ||
                    reinterpret_cast<uintptr_t>(b->data_ptr()) >= end_address(a),
                "sparse_mla_bf16: output/workspace storage must not overlap");
}
} // namespace

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(sparse_mla_bf16_fwd_out,
                                    (aiter_tensor_t * q,
                                     aiter_tensor_t* kv_buffer,
                                     aiter_tensor_t* kv_indptr,
                                     aiter_tensor_t* kv_indices,
                                     aiter_tensor_t* out,
                                     aiter_tensor_t* lse,
                                     aiter_tensor_t* partial_o,
                                     aiter_tensor_t* partial_lse,
                                     float softmax_scale,
                                     int64_t kv_splits,
                                     int64_t version,
                                     hipStream_t stream),
                                    (q,
                                     kv_buffer,
                                     kv_indptr,
                                     kv_indices,
                                     out,
                                     lse,
                                     partial_o,
                                     partial_lse,
                                     softmax_scale,
                                     kv_splits,
                                     version,
                                     stream))
{
    AITER_CHECK(q && q->is_gpu(), "q must be on a HIP device");
    const int device = q->device_id;
    HipDeviceGuard guard(device);
    AITER_CHECK(get_gpu_arch() == "gfx950", "sparse_mla_bf16 requires gfx950");
    AITER_CHECK(version == 1 || version == 2, "version must be 1 or 2");
    AITER_CHECK(kv_splits >= 1 && kv_splits <= 32 && (kv_splits & (kv_splits - 1)) == 0,
                "kv_splits must be one of 1, 2, 4, 8, 16, 32");
    AITER_CHECK(std::isfinite(softmax_scale) && softmax_scale > 0, "invalid softmax_scale");
    check_tensor(q, "q", AITER_DTYPE_bf16, device);
    AITER_CHECK(q->dim() == 3 && q->size(2) == 512 && (q->size(1) == 16 || q->size(1) == 64),
                "expected q=[Q,16|64,512]");
    const int64_t queries = q->size(0), heads = q->size(1);
    AITER_CHECK(queries <= INT_MAX, "too many query rows");
    AITER_CHECK(q->stride(2) == 1 && q->stride(1) >= 512 &&
                    q->stride(0) >= (heads - 1) * q->stride(1) + 512,
                "q must have non-overlapping, unit-inner-stride rows");
    check_tensor(kv_buffer, "kv_buffer", AITER_DTYPE_bf16, device);
    AITER_CHECK(kv_buffer->dim() == 2 && kv_buffer->size(1) == 512 && kv_buffer->stride(1) == 1 &&
                    kv_buffer->stride(0) >= 512,
                "expected kv_buffer=[slots,512] with unit inner stride");
    check_tensor(kv_indptr, "kv_indptr", AITER_DTYPE_i32, device);
    check_tensor(kv_indices, "kv_indices", AITER_DTYPE_i32, device);
    check_shape(kv_indptr, "kv_indptr", {queries + 1});
    AITER_CHECK(kv_indices->dim() == 1 && kv_indptr->is_contiguous() && kv_indices->is_contiguous(),
                "CSR tensors must be contiguous vectors");
    AITER_CHECK(kv_indices->numel() <= INT_MAX, "CSR offsets must fit int32");
    check_tensor(out, "out", AITER_DTYPE_bf16, device);
    check_shape(out, "out", {queries, heads, 512});
    AITER_CHECK(out->is_contiguous(), "out must be contiguous");
    if(lse)
    {
        check_tensor(lse, "lse", AITER_DTYPE_fp32, device);
        check_shape(lse, "lse", {queries, heads});
        AITER_CHECK(lse->is_contiguous(), "lse must be contiguous");
    }
    if(kv_splits > 1)
    {
        check_tensor(partial_o, "partial_o", AITER_DTYPE_fp32, device);
        check_tensor(partial_lse, "partial_lse", AITER_DTYPE_fp32, device);
        check_shape(partial_o, "partial_o", {queries, kv_splits, heads, 512});
        check_shape(partial_lse, "partial_lse", {queries, kv_splits, heads});
        AITER_CHECK(partial_o->is_contiguous() && partial_lse->is_contiguous(),
                    "workspace must be contiguous");
    }
    else
    {
        partial_o = partial_lse = nullptr;
    }
    const aiter_tensor_t* writes[] = {out, lse, partial_o, partial_lse};
    for(int i = 0; i < 4; ++i)
    {
        for(auto input : {q, kv_buffer, kv_indptr, kv_indices})
            check_no_overlap(writes[i], input);
        for(int j = 0; j < i; ++j)
            check_no_overlap(writes[i], writes[j]);
    }
    // CSR offsets are caller-owned device data: monotone, starting at zero and
    // ending at indices.numel(). No D2H reads or synchronization during capture.
    // Out-of-range slot ids (including -1) are masked in the kernel.
    if(!queries)
        return;
    sparse_mla_bf16::Params p{static_cast<const uint16_t*>(q->data_ptr()),
                              static_cast<const uint16_t*>(kv_buffer->data_ptr()),
                              static_cast<const int32_t*>(kv_indptr->data_ptr()),
                              static_cast<const int32_t*>(kv_indices->data_ptr()),
                              static_cast<uint16_t*>(out->data_ptr()),
                              lse ? static_cast<float*>(lse->data_ptr()) : nullptr,
                              partial_o ? static_cast<float*>(partial_o->data_ptr()) : nullptr,
                              partial_lse ? static_cast<float*>(partial_lse->data_ptr()) : nullptr,
                              int32_t(queries),
                              int32_t(kv_splits),
                              kv_buffer->size(0),
                              q->stride(0),
                              q->stride(1),
                              kv_buffer->stride(0),
                              softmax_scale};
    const auto status = heads == 16 ? sparse_mla_bf16::launch<16>(p, version, stream)
                                    : sparse_mla_bf16::launch<64>(p, version, stream);
    HIP_CALL_LAUNCH(status);
}
