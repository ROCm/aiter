// SPDX-License-Identifier: MIT
// FP8 sparse MLA with fused BF16 query conversion and BF16 output.
// clang-format off
#include "aiter_tensor.h"
#include "aiter_ctypes_error.h"
#include "fp8/params.cuh"
// clang-format on
#include <climits>
#include <cmath>

AITER_CTYPES_ERROR_DEF

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
void check_aligned(const aiter_tensor_t* t, const char* name, size_t alignment)
{
    AITER_CHECK(!t->numel() || reinterpret_cast<uintptr_t>(t->data_ptr()) % alignment == 0,
                name,
                ": unaligned storage");
}
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
                "sparse_mla_fp8: output/workspace storage must not overlap");
}
} // namespace

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(sparse_mla_fp8_fwd_out,
                                    (aiter_tensor_t * q,
                                     aiter_tensor_t* kv_buffer,
                                     aiter_tensor_t* kv_indices,
                                     aiter_tensor_t* out,
                                     aiter_tensor_t* lse,
                                     aiter_tensor_t* partial_o,
                                     aiter_tensor_t* partial_lse,
                                     float softmax_scale,
                                     float q_scale,
                                     float kv_scale,
                                     hipStream_t stream),
                                    (q,
                                     kv_buffer,
                                     kv_indices,
                                     out,
                                     lse,
                                     partial_o,
                                     partial_lse,
                                     softmax_scale,
                                     q_scale,
                                     kv_scale,
                                     stream))
{
    AITER_CHECK(q && q->is_gpu(), "q must be on a HIP device");
    const int device = q->device_id;
    HipDeviceGuard guard(device);
    AITER_CHECK(get_gpu_arch() == "gfx950", "sparse_mla_fp8 requires gfx950");
    for(float scale : {softmax_scale, q_scale, kv_scale})
        AITER_CHECK(std::isfinite(scale) && scale > 0, "scales must be finite and positive");
    AITER_CHECK(q->dtype() == AITER_DTYPE_bf16 || q->dtype() == AITER_DTYPE_fp8,
                "q must be BF16 or FP8");
    check_tensor(q, "q", q->dtype(), device);
    AITER_CHECK(q->dim() == 3 && q->size(1) == 64 && q->size(2) == 512, "expected q=[Q,64,512]");
    const int64_t queries = q->size(0);
    AITER_CHECK(queries <= 2048, "Q must be <= 2048");
    AITER_CHECK(q->stride(2) == 1 && q->stride(1) >= 512 && q->stride(0) >= 63 * q->stride(1) + 512,
                "q must have non-overlapping, unit-inner-stride rows");
    const int bytes = q->element_size();
    AITER_CHECK((q->stride(0) * bytes) % 16 == 0 && (q->stride(1) * bytes) % 16 == 0,
                "q rows must be aligned to 16 bytes");
    check_aligned(q, "q", 16);
    check_tensor(kv_buffer, "kv_buffer", AITER_DTYPE_fp8, device);
    AITER_CHECK(kv_buffer->dim() == 2 && kv_buffer->size(1) == 512 && kv_buffer->size(0) > 0 &&
                    kv_buffer->size(0) <= INT_MAX && kv_buffer->is_contiguous(),
                "expected contiguous kv_buffer=[slots,512]");
    check_aligned(kv_buffer, "kv_buffer", 16);
    check_tensor(kv_indices, "kv_indices", AITER_DTYPE_i32, device);
    AITER_CHECK(kv_indices->dim() == 2 && kv_indices->size(0) == queries &&
                    kv_indices->size(1) <= 2176 && kv_indices->is_contiguous(),
                "expected contiguous kv_indices=[Q,K], K <= 2176");
    check_aligned(kv_indices, "kv_indices", 4);
    check_tensor(out, "out", AITER_DTYPE_bf16, device);
    check_shape(out, "out", {queries, 64, 512});
    AITER_CHECK(out->is_contiguous(), "out must be contiguous");
    check_aligned(out, "out", 16);
    if(lse)
    {
        check_tensor(lse, "lse", AITER_DTYPE_fp32, device);
        check_shape(lse, "lse", {queries, 64});
        AITER_CHECK(lse->is_contiguous(), "lse must be contiguous");
        check_aligned(lse, "lse", 4);
    }
    const int splits   = sparse_mla_fp8::split_count(queries);
    const int64_t rows = splits > 1 ? queries * splits * 64 : 0;
    check_tensor(partial_o, "partial_o", AITER_DTYPE_bf16, device);
    check_tensor(partial_lse, "partial_lse", AITER_DTYPE_fp32, device);
    check_shape(partial_o, "partial_o", {rows * 512});
    check_shape(partial_lse, "partial_lse", {rows});
    AITER_CHECK(partial_o->is_contiguous() && partial_lse->is_contiguous(),
                "workspace must be contiguous");
    check_aligned(partial_o, "partial_o", 16);
    check_aligned(partial_lse, "partial_lse", 4);
    const aiter_tensor_t* writes[] = {out, lse, partial_o, partial_lse};
    for(int i = 0; i < 4; ++i)
    {
        for(auto input : {q, kv_buffer, kv_indices})
            check_no_overlap(writes[i], input);
        for(int j = 0; j < i; ++j)
            check_no_overlap(writes[i], writes[j]);
    }
    if(!queries)
        return;
    sparse_mla_fp8::Params p{};
    p.q               = static_cast<const uint8_t*>(q->data_ptr());
    p.kv              = static_cast<const uint8_t*>(kv_buffer->data_ptr());
    p.indices         = static_cast<const int32_t*>(kv_indices->data_ptr());
    p.partial_o       = static_cast<uint16_t*>(partial_o->data_ptr());
    p.partial_lse     = static_cast<float*>(partial_lse->data_ptr());
    p.out             = static_cast<uint8_t*>(out->data_ptr());
    p.lse             = lse ? static_cast<float*>(lse->data_ptr()) : nullptr;
    p.queries         = int(queries);
    p.slots           = int(kv_buffer->size(0));
    p.capacity        = int(kv_indices->size(1));
    p.q_scale         = q_scale;
    p.kv_scale        = kv_scale;
    p.attention_scale = softmax_scale;
    p.q_stride_row    = q->stride(0);
    p.q_stride_head   = q->stride(1);
    p.bf16_q          = q->dtype() == AITER_DTYPE_bf16;
    hipError_t status;
    if(queries <= 6)
        status = sparse_mla_fp8::launch_small(p, stream);
    else if(queries <= 12)
        status = sparse_mla_fp8::launch_prefetch(p, stream);
    else if(queries <= 64)
        status = sparse_mla_fp8::launch_rotating(p, stream);
    else
        status = sparse_mla_fp8::launch_streaming(p, stream);
    HIP_CALL_LAUNCH(status);
}
