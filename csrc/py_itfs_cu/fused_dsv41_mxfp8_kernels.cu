// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_stream.h"
#include "fused_dsv41/kernel_args.hpp"
#include "fused_dsv41_mxfp8.h"
#include <algorithm>
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
// Number of complete rows addressable through a possibly truncated view.
// Clamp before multiplication so even enormous strides/capacities cannot overflow.
int backed_rows(int64_t bytes, int64_t stride, int64_t block, int64_t width, int64_t requested)
{
    const int64_t pages = bytes / stride;
    if(pages >= (requested + block - 1) / block)
        return requested;
    return std::min(requested, pages * block + std::min(block, (bytes % stride) / width));
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
                           bool q_is_rotated,
                           const std::optional<aiter_tensor_t>& cache_scales,
                           int64_t cache_page_size,
                           int64_t cache_num_rows,
                           int64_t cache_data_page_stride,
                           int64_t cache_scale_page_stride,
                           bool fp8_qk,
                           const std::optional<aiter_tensor_t>& q_scale)
{
    aiter_detail::AiterThrowGuard errors;
    AITER_CHECK(q.is_gpu() && q.dim() == 3, "q must be a GPU tensor [T,H,512]");
    const auto t = q.size(0), h = q.size(1);
    const int device = q.device_id;
    AITER_CHECK(h == 8 || h == 16 || h == 32 || h == 64 || h == 128, "unsupported head count");
    AITER_CHECK(t <= INT_MAX && prefix.numel() <= INT_MAX, "dimensions must fit int32");
    if(q_scale)
    {
        AITER_CHECK(fp8_qk && q_is_rotated,
                    "prequantized Q requires fp8_qk=True and q_is_rotated=True");
        check_tensor(q, "q", AITER_DTYPE_fp8, {t, h, 512}, device, 32);
        check_tensor(*q_scale, "q_scale", AITER_DTYPE_u8, {t, h, 16}, device);
    }
    else
        check_tensor(q, "q", AITER_DTYPE_bf16, {t, h, 512}, device, 16);
    AITER_CHECK(!fp8_qk || ((h == 64 || h == 128) && pool.dtype() != AITER_DTYPE_bf16 &&
                            !extend_kv && !extend && !extend_indptr),
                "FP8 QK requires H64/H128, FP8 KV, and no BF16 extend source");
    const bool bf16_cache = pool.dtype() == AITER_DTYPE_bf16;
    if(bf16_cache)
    {
        AITER_CHECK(pool.dim() == 2 && pool.size(0) <= INT_MAX,
                    "BF16 pool must be [S,512] with S fitting int32");
        check_tensor(pool, "pool", AITER_DTYPE_bf16, {pool.size(0), 512}, device, 16);
        AITER_CHECK(cache_page_size == 0 && !cache_scales,
                    "BF16 pool does not accept FP8 page metadata");
    }
    else
    {
        AITER_CHECK(pool.dim() == 1 || (pool.dim() == 2 && pool.size(1) == 1),
                    "pool must be [bytes] or [bytes,1]");
        AITER_CHECK(pool.is_gpu() && pool.device_id == device && pool.is_contiguous() &&
                        pool.dtype() == AITER_DTYPE_u8 &&
                        reinterpret_cast<uintptr_t>(pool.data_ptr()) % 4 == 0,
                    "pool must be contiguous uint8 and 4-byte aligned on q.device");
    }
    AITER_CHECK(cache_page_size >= 0 && cache_page_size <= INT_MAX && cache_num_rows >= 0 &&
                    cache_num_rows <= INT_MAX,
                "cache page size and row count must fit nonnegative int32");
    const bool paged = cache_page_size != 0;
    AITER_CHECK(paged == bool(cache_scales), "paged cache requires cache_scales and page size");
    if(paged)
    {
        check_tensor(*cache_scales,
                     "cache_scales",
                     AITER_DTYPE_u8,
                     {static_cast<int64_t>(cache_scales->numel())},
                     device,
                     4);
        AITER_CHECK(cache_data_page_stride >= cache_page_size * 512 &&
                        cache_scale_page_stride >= cache_page_size * 16 &&
                        cache_data_page_stride % 4 == 0 && cache_scale_page_stride % 4 == 0,
                    "cache page strides must cover rows and be 4-byte aligned");
    }
    else
        AITER_CHECK(cache_num_rows == 0 && cache_data_page_stride == 0 &&
                        cache_scale_page_stride == 0,
                    "cache metadata requires paged mode");
    check_tensor(prefix,
                 "prefix",
                 (paged || bf16_cache) ? AITER_DTYPE_i32 : AITER_DTYPE_i64,
                 {static_cast<int64_t>(prefix.numel())},
                 device,
                 (paged || bf16_cache) ? 4 : 8);
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
    a.q               = static_cast<const B*>(q.data_ptr());
    a.pool            = static_cast<const uint8_t*>(pool.data_ptr());
    a.prefix          = prefix.data_ptr();
    a.cache.scales    = paged ? static_cast<const uint8_t*>(cache_scales->data_ptr()) : nullptr;
    a.cache.page_size = bf16_cache ? -2 : cache_page_size;
    a.cache.num_rows =
        bf16_cache ? pool.size(0)
        : paged
            ? std::min(
                  backed_rows(
                      pool.numel(), cache_data_page_stride, cache_page_size, 512, cache_num_rows),
                  backed_rows(cache_scales->numel(),
                              cache_scale_page_stride,
                              cache_page_size,
                              16,
                              cache_num_rows))
            : 0;
    a.cache.data_page_stride  = cache_data_page_stride;
    a.cache.scale_page_stride = cache_scale_page_stride;
    a.prefix_indptr           = static_cast<const int*>(prefix_indptr.data_ptr());
    a.cos                     = static_cast<const B*>(cos.data_ptr());
    a.sin                     = static_cast<const B*>(sin.data_ptr());
    a.positions               = positions.data_ptr();
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
    a.fp8_qk = fp8_qk;
    a.q_scale = q_scale ? static_cast<const uint8_t*>(q_scale->data_ptr()) : nullptr;
    a.softmax_scale   = softmax_scale;
    HIP_CALL(launch_fused_dsv41_mxfp8(&a, !q_is_rotated, aiter::getCurrentHIPStream()));
}
