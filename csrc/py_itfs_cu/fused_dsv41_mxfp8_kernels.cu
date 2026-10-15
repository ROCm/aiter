// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_stream.h"
#include "fused_dsv41/kernel_args.hpp"
#include "fused_dsv41_mxfp8.h"
#include <climits>
#include <cmath>
#include <cstring>

namespace {
void check_tensor(const aiter_tensor_t& x,
                  const char* name,
                  AiterDtype dtype,
                  std::initializer_list<int64_t> shape,
                  int device,
                  size_t alignment = 1)
{
    AITER_CHECK(x.is_gpu() && x.device_id == device, name, ": wrong device");
    AITER_CHECK(x.dtype() == dtype, name, ": wrong dtype");
    AITER_CHECK(x.is_contiguous() && x.dim() == shape.size(), name, ": wrong shape or strides");
    int i = 0;
    for(auto dim : shape)
        AITER_CHECK(x.size(i++) == dim, name, ": wrong shape");
    AITER_CHECK(x.numel() == 0 || (x.data_ptr() != nullptr &&
                                   reinterpret_cast<uintptr_t>(x.data_ptr()) % alignment == 0),
                name,
                ": unaligned pointer");
}
} // namespace

void fused_dsv41_mxfp8_fwd(const aiter_tensor_t& q,
                           const aiter_tensor_t& pool,
                           const aiter_tensor_t& prefix,
                           const aiter_tensor_t& prefix_indptr,
                           const aiter_tensor_t& cos,
                           const aiter_tensor_t& sin,
                           const aiter_tensor_t& positions,
                           const aiter_tensor_t& out,
                           const aiter_tensor_t& out_scale,
                           const aiter_tensor_t& max_logits,
                           const aiter_tensor_t& lse,
                           const std::optional<aiter_tensor_t>& extend_kv,
                           const std::optional<aiter_tensor_t>& extend,
                           const std::optional<aiter_tensor_t>& extend_indptr,
                           const std::optional<aiter_tensor_t>& attn_sink,
                           const std::optional<aiter_tensor_t>& debug_out,
                           float softmax_scale,
                           bool q_is_rotated)
{
    aiter_detail::AiterThrowGuard errors;
    AITER_CHECK(q.is_gpu() && q.dim() == 3, "q must be a GPU tensor [T,H,512]");
    const auto t = q.size(0), h = q.size(1);
    const int device = q.device_id;
    AITER_CHECK(h == 8 || h == 16 || h == 32 || h == 64 || h == 128, "unsupported head count");
    AITER_CHECK(t <= INT_MAX && prefix.numel() <= INT_MAX, "dimensions must fit int32");
    check_tensor(q, "q", AITER_DTYPE_bf16, {t, h, 512}, device, 16);
    AITER_CHECK(pool.dim() == 1 || (pool.dim() == 2 && pool.size(1) == 1),
                "pool must be [bytes] or [bytes,1]");
    AITER_CHECK(pool.is_gpu() && pool.device_id == device && pool.is_contiguous() &&
                    pool.dtype() == AITER_DTYPE_u8 &&
                    reinterpret_cast<uintptr_t>(pool.data_ptr()) % 4 == 0,
                "pool must be contiguous uint8 and 4-byte aligned on q.device");
    check_tensor(
        prefix, "prefix", AITER_DTYPE_i64, {static_cast<int64_t>(prefix.numel())}, device, 8);
    check_tensor(prefix_indptr, "prefix_indptr", AITER_DTYPE_i32, {t + 1}, device, 4);
    AITER_CHECK(positions.dtype() == AITER_DTYPE_i32 || positions.dtype() == AITER_DTYPE_i64,
                "positions must be int32 or int64");
    check_tensor(positions, "positions", positions.dtype(), {t}, device, positions.element_size());
    AITER_CHECK(cos.dim() == 2 && (!t || cos.size(0) > 0), "cos must be [P,32]");
    check_tensor(cos, "cos", AITER_DTYPE_bf16, {cos.size(0), 32}, device, 2);
    check_tensor(sin, "sin", AITER_DTYPE_bf16, {cos.size(0), 32}, device, 2);
    check_tensor(out, "out", AITER_DTYPE_fp8, {t, h / 8, 4096}, device, 16);
    check_tensor(out_scale, "out_scale", AITER_DTYPE_fp8_e8m0, {t, h / 8, 128}, device);
    check_tensor(max_logits, "max_logits", AITER_DTYPE_fp32, {t, h}, device, 4);
    check_tensor(lse, "lse", AITER_DTYPE_fp32, {t, h}, device, 4);
    if(attn_sink)
        check_tensor(*attn_sink, "attn_sink", AITER_DTYPE_fp32, {h}, device, 4);
    if(debug_out)
        check_tensor(*debug_out, "debug_out", AITER_DTYPE_fp32, {t, h, 512}, device, 16);
    int erows = 0, ecap = 0;
    AITER_CHECK(bool(extend_kv) == bool(extend) && bool(extend_kv) == bool(extend_indptr),
                "extend_kv, extend and extend_indptr must be provided together");
    if(extend_kv)
    {
        AITER_CHECK(extend_kv->dim() == 2 && extend_kv->size(0) <= INT_MAX &&
                        extend->numel() <= INT_MAX,
                    "extend dimensions must fit int32");
        erows = extend_kv->size(0);
        ecap  = extend->numel();
        check_tensor(*extend_kv, "extend_kv", AITER_DTYPE_bf16, {erows, 512}, device, 16);
        check_tensor(*extend, "extend", AITER_DTYPE_i32, {ecap}, device, 4);
        check_tensor(*extend_indptr, "extend_indptr", AITER_DTYPE_i32, {t + 1}, device, 4);
    }
    AITER_CHECK(std::isfinite(softmax_scale) && softmax_scale > 0,
                "softmax_scale must be positive and finite");
    HipDeviceGuard guard(device);
    hipDeviceProp_t properties;
    HIP_CALL(hipGetDeviceProperties(&properties, device));
    AITER_CHECK(std::strncmp(properties.gcnArchName, "gfx950", 6) == 0,
                "fused_dsv41_mxfp8 requires gfx950");
    FusedDsv41Args a{};
    a.q             = static_cast<const B*>(q.data_ptr());
    a.pool          = static_cast<const uint8_t*>(pool.data_ptr());
    a.prefix        = static_cast<const int64_t*>(prefix.data_ptr());
    a.prefix_indptr = static_cast<const int*>(prefix_indptr.data_ptr());
    a.cos           = static_cast<const B*>(cos.data_ptr());
    a.sin           = static_cast<const B*>(sin.data_ptr());
    a.positions     = positions.data_ptr();
    if(extend_kv)
    {
        a.extend_kv     = static_cast<const B*>(extend_kv->data_ptr());
        a.extend        = static_cast<const int*>(extend->data_ptr());
        a.extend_indptr = static_cast<const int*>(extend_indptr->data_ptr());
    }
    a.sink            = attn_sink ? static_cast<const float*>(attn_sink->data_ptr()) : nullptr;
    a.debug_out       = debug_out ? static_cast<float*>(debug_out->data_ptr()) : nullptr;
    a.out             = static_cast<uint8_t*>(out.data_ptr());
    a.scales          = static_cast<uint8_t*>(out_scale.data_ptr());
    a.max_logits      = static_cast<float*>(max_logits.data_ptr());
    a.lse             = static_cast<float*>(lse.data_ptr());
    a.tokens          = t;
    a.heads           = h;
    a.extend_rows     = erows;
    a.extend_capacity = ecap;
    a.prefix_capacity = prefix.numel();
    a.pool_bytes      = pool.numel();
    a.positions64     = positions.dtype() == AITER_DTYPE_i64;
    a.softmax_scale   = softmax_scale;
    HIP_CALL(launch_fused_dsv41_mxfp8(&a, !q_is_rotated, aiter::getCurrentHIPStream()));
}
