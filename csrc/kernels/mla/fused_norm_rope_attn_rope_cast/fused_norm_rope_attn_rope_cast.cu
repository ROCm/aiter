// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// Torch-free ctypes entry: validate tensors, construct kernel arguments, and dispatch.
// Device implementations are in attention_32rows.cu and attention_64rows_h*.cu.
#include "aiter_tensor.h"

// The error bridge requires aiter_hip_common.h, included by aiter_tensor.h.
#include "aiter_ctypes_error.h"
#include "dispatch.hpp"
#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>

// Try the 64-row KV, 32-head tile, or split-KV FP8 path with fused Q/output transforms.
// Returns false when this dispatcher does not select a kernel; the caller must fall back.
// Their first CTA phase validates indices before choosing checked/unchecked tile loops.
inline bool try_launch_fused_fp8_attention(const FusedNormRopeAttnRopeCastArgs& a,
                                           bool apply_q_rope,
                                           hipStream_t stream)
{
    const bool uses_strided_fp8_rows = a.cache.layout == fused_kv_layout::strided_fp8_rows;
    if(!a.fp8_qk || a.tokens == 0 || (!uses_strided_fp8_rows && a.extend_indptr) ||
       (!uses_strided_fp8_rows && a.cache.layout != fused_kv_layout::legacy_tagged_fp8))
        return false;

    // Host capacities estimate rows/query without synchronizing the device CSR.
    const int64_t index_capacity =
        int64_t(a.prefix_capacity) + (uses_strided_fp8_rows ? a.extend_capacity : 0);
    namespace policy                 = fused_norm_rope_attn_rope_cast_policy;
    const bool has_prefix_and_extend = a.prefix_capacity > 0 && a.extend_capacity > 0;
    const bool use_shared_kv         = a.heads == 128 && uses_strided_fp8_rows && !a.q_norm &&
                               (a.q_scale || apply_q_rope) &&
                               policy::prefer_h128_shared_kv(a.tokens, index_capacity);
    const bool use_64row_kv_tile =
        (a.heads == 64 && uses_strided_fp8_rows &&
         policy::prefer_h64_64row(a.tokens, index_capacity, has_prefix_and_extend)) ||
        (a.heads == 128 && !use_shared_kv && policy::prefer_h128_64row(a.tokens, index_capacity));
    const bool use_32head_tile = uses_strided_fp8_rows && has_prefix_and_extend &&
                                 policy::prefer_32head_tile(a.tokens, a.heads, index_capacity);
    if(!(uses_strided_fp8_rows && a.heads < 64) && !use_64row_kv_tile && !use_32head_tile &&
       !(uses_strided_fp8_rows && a.num_splits > 1))
        return false;
    FusedNormRopeAttnRopeCastParams k{};
    k.partial_ptr          = a.partial;
    k.num_splits           = a.num_splits;
    k.q_ptr                = a.q;
    k.q_scale_ptr          = a.q_scale;
    k.raw_q_ptr            = a.q;
    k.unified_kv_ptr       = a.pool;
    k.unified_kv_scale_ptr = a.cache.scales;
    k.kv_indptr_prefix     = a.prefix_indptr;
    k.kv_indices_prefix    = static_cast<const int*>(a.prefix);
    k.attn_sink_ptr        = a.sink;
    k.lse_ptr              = a.lse;
    k.max_ptr              = a.max_logits;
    k.output_q_ptr         = a.out;
    k.output_s_ptr         = a.scales;
    k.debug_ptr            = a.debug_out;
    k.cos_ptr              = a.cos;
    k.sin_ptr              = a.sin;
    k.positions_ptr        = a.positions;
    k.N                    = a.tokens;
    k.H                    = a.heads;
    k.stride_q_n           = a.heads * 512;
    k.stride_q_h           = 512;
    k.stride_q_scale_n     = a.heads * 16;
    k.stride_q_scale_h     = 16;
    k.stride_o_n           = a.heads * 512;
    k.stride_o_h           = 512;
    k.stride_lse_n         = a.heads;
    k.softmax_scale        = a.softmax_scale;
    k.positions64          = a.positions64;
    k.q_is_rotated         = !apply_q_rope;
    k.q_norm               = a.q_norm;
    k.norm_eps             = a.norm_eps;
    k.cache_mode           = uses_strided_fp8_rows ? 0 : 1;
    k.prefix_capacity      = a.prefix_capacity;
    k.pool_bytes           = a.pool_bytes;
    if(uses_strided_fp8_rows)
    {
        k.prefix_rows            = a.cache.num_rows;
        k.extend_rows            = a.extend_rows;
        k.stride_kv_prefix       = a.cache.data_row_stride;
        k.stride_kv_scale_prefix = a.cache.scale_row_stride;
        k.kv_ptr                 = a.extend_kv;
        k.kv_scale_ptr           = a.extend_cache.scales;
        k.kv_indptr_extend       = a.extend_indptr;
        k.kv_indices_extend      = a.extend;
        k.extend_capacity        = a.extend_capacity;
        k.stride_kv_extend       = a.extend_cache.data_row_stride;
        k.stride_kv_scale_extend = a.extend_cache.scale_row_stride;
    }
    HIP_CALL(run_fused_norm_rope_attn_rope_cast_fwd(k, a.q_scale == nullptr, stream));
    return true;
}

AITER_CTYPES_ERROR_DEF

namespace {
// Bound combined tile/prefetch offsets and the 32-bit index-buffer byte range.
constexpr int max_csr_capacity = INT_MAX / 2 - 256;

void check_tensor(const aiter_tensor_t* x,
                  const char* name,
                  AiterDtype dtype,
                  std::initializer_list<int64_t> shape,
                  int device,
                  size_t alignment = 1)
{
    AITER_CHECK(x != nullptr, name, ": must not be null");
    AITER_CHECK(x->is_gpu() && x->device_id == device, name, ": wrong device");
    AITER_CHECK(x->dtype() == dtype, name, ": wrong dtype");
    AITER_CHECK(x->is_contiguous() && x->dim() == shape.size(), name, ": wrong shape or strides");
    int i = 0;
    for(auto dim : shape)
        AITER_CHECK(x->size(i++) == dim, name, ": wrong shape");
    AITER_CHECK(x->numel() == 0 || (x->data_ptr() != nullptr &&
                                    reinterpret_cast<uintptr_t>(x->data_ptr()) % alignment == 0),
                name,
                ": unaligned pointer");
}
void check_row_view(const aiter_tensor_t* x, const char* name, int width, int device, bool scale)
{
    AITER_CHECK(x != nullptr, name, ": must not be null");
    AITER_CHECK(x->is_gpu() && x->device_id == device, name, ": wrong device");
    AITER_CHECK(x->dim() == 2 && x->size(1) == width && x->size(0) <= INT_MAX,
                name,
                ": expected a row matrix");
    AITER_CHECK(scale ? (x->dtype() == AITER_DTYPE_u8 || x->dtype() == AITER_DTYPE_fp8_e8m0)
                      : x->dtype() == AITER_DTYPE_fp8,
                name,
                ": expected FP8 values or E8M0/uint8 scales");
    AITER_CHECK(x->stride(1) == 1 && x->stride(0) >= width && x->stride(0) <= INT_MAX &&
                    x->stride(0) % 4 == 0,
                name,
                ": require unit channel stride and 4-byte aligned row stride");
    AITER_CHECK(x->numel() == 0 || (x->data_ptr() != nullptr &&
                                    reinterpret_cast<uintptr_t>(x->data_ptr()) % 4 == 0),
                name,
                ": unaligned row base");
}

bool overlaps(const aiter_tensor_t* x, const aiter_tensor_t* y)
{
    if(!x || !y || !x->numel() || !y->numel())
        return false;
    auto bytes = [](const aiter_tensor_t& a) {
        uint64_t last = 0;
        for(size_t i = 0; i < a.dim(); ++i)
            last += uint64_t(a.size(i) - 1) * a.stride(i);
        return (last + 1) * a.element_size();
    };
    const auto px = reinterpret_cast<uintptr_t>(x->data_ptr());
    const auto py = reinterpret_cast<uintptr_t>(y->data_ptr());
    return px < py + bytes(*y) && py < px + bytes(*x);
}
} // namespace

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(_fused_norm_rope_attn_rope_cast_legacy_fwd,
                                    (const aiter_tensor_t* q,
                                     const aiter_tensor_t* pool,
                                     const aiter_tensor_t* prefix,
                                     const aiter_tensor_t* prefix_indptr,
                                     const aiter_tensor_t* cos,
                                     const aiter_tensor_t* sin,
                                     const aiter_tensor_t* positions,
                                     const aiter_tensor_t* out,
                                     const aiter_tensor_t* out_scale,
                                     const aiter_tensor_t* max_logits,
                                     const aiter_tensor_t* lse,
                                     const aiter_tensor_t* extend_kv,
                                     const aiter_tensor_t* extend,
                                     const aiter_tensor_t* extend_indptr,
                                     const aiter_tensor_t* attn_sink,
                                     const aiter_tensor_t* debug_out,
                                     float softmax_scale,
                                     int q_is_rotated,
                                     int fp8_qk,
                                     const aiter_tensor_t* q_scale,
                                     int q_norm,
                                     float norm_eps,
                                     hipStream_t stream),
                                    (q,
                                     pool,
                                     prefix,
                                     prefix_indptr,
                                     cos,
                                     sin,
                                     positions,
                                     out,
                                     out_scale,
                                     max_logits,
                                     lse,
                                     extend_kv,
                                     extend,
                                     extend_indptr,
                                     attn_sink,
                                     debug_out,
                                     softmax_scale,
                                     q_is_rotated,
                                     fp8_qk,
                                     q_scale,
                                     q_norm,
                                     norm_eps,
                                     stream))
{
    AITER_CHECK(q && pool && prefix && prefix_indptr && cos && sin && positions && out &&
                    out_scale && max_logits && lse,
                "required tensors must not be null");
    AITER_CHECK(q->is_gpu() && q->dim() == 3, "q must be a GPU tensor [T,H,512]");
    const auto t = q->size(0), h = q->size(1);
    const int device = q->device_id;
    AITER_CHECK(h == 8 || h == 16 || h == 32 || h == 64 || h == 128, "unsupported head count");
    AITER_CHECK(t <= INT_MAX, "dimensions must fit int32");
    AITER_CHECK(prefix->numel() <= max_csr_capacity, "CSR capacity exceeds supported limit");
    AITER_CHECK(std::isfinite(norm_eps) && norm_eps >= std::numeric_limits<float>::min(),
                "norm_eps must be positive, finite and normal in float32");
    AITER_CHECK(!q_norm || (!q_is_rotated && !q_scale),
                "q_norm requires raw BF16 Q and q_is_rotated=False");
    if(q_scale)
    {
        AITER_CHECK(fp8_qk && q_is_rotated,
                    "prequantized Q requires fp8_qk=True and q_is_rotated=True");
        check_tensor(q, "q", AITER_DTYPE_fp8, {t, h, 512}, device, 32);
        check_tensor(q_scale, "q_scale", AITER_DTYPE_u8, {t, h, 16}, device);
    }
    else
        check_tensor(q, "q", AITER_DTYPE_bf16, {t, h, 512}, device, 16);
    AITER_CHECK(!fp8_qk || ((h == 64 || h == 128) && pool->dtype() != AITER_DTYPE_bf16 &&
                            !extend_kv && !extend && !extend_indptr),
                "FP8 QK requires H64/H128, FP8 KV, and no BF16 extend source");
    const bool bf16_cache = pool->dtype() == AITER_DTYPE_bf16;
    if(bf16_cache)
    {
        AITER_CHECK(pool->dim() == 2 && pool->size(0) <= INT_MAX,
                    "BF16 pool must be [S,512] with S fitting int32");
        check_tensor(pool, "pool", AITER_DTYPE_bf16, {pool->size(0), 512}, device, 16);
    }
    else
    {
        AITER_CHECK(pool->dim() == 1 || (pool->dim() == 2 && pool->size(1) == 1),
                    "pool must be [bytes] or [bytes,1]");
        AITER_CHECK(pool->is_gpu() && pool->device_id == device && pool->is_contiguous() &&
                        pool->dtype() == AITER_DTYPE_u8 &&
                        reinterpret_cast<uintptr_t>(pool->data_ptr()) % 4 == 0,
                    "pool must be contiguous uint8 and 4-byte aligned on q.device");
    }
    check_tensor(prefix,
                 "prefix",
                 bf16_cache ? AITER_DTYPE_i32 : AITER_DTYPE_i64,
                 {static_cast<int64_t>(prefix->numel())},
                 device,
                 bf16_cache ? 4 : 8);
    check_tensor(prefix_indptr, "prefix_indptr", AITER_DTYPE_i32, {t + 1}, device, 4);
    AITER_CHECK(positions->dtype() == AITER_DTYPE_i32 || positions->dtype() == AITER_DTYPE_i64,
                "positions must be int32 or int64");
    check_tensor(
        positions, "positions", positions->dtype(), {t}, device, positions->element_size());
    AITER_CHECK(cos->dim() == 2 && (!t || cos->size(0) > 0), "cos must be [P,32]");
    check_tensor(cos, "cos", AITER_DTYPE_bf16, {cos->size(0), 32}, device, 2);
    check_tensor(sin, "sin", AITER_DTYPE_bf16, {cos->size(0), 32}, device, 2);
    check_tensor(out, "out", AITER_DTYPE_fp8, {t, h / 8, 4096}, device, 16);
    check_tensor(out_scale, "out_scale", AITER_DTYPE_fp8_e8m0, {t, h / 8, 128}, device);
    check_tensor(max_logits, "max_logits", AITER_DTYPE_fp32, {t, h}, device, 4);
    check_tensor(lse, "lse", AITER_DTYPE_fp32, {t, h}, device, 4);
    if(attn_sink)
        check_tensor(attn_sink, "attn_sink", AITER_DTYPE_fp32, {h}, device, 4);
    if(debug_out)
        check_tensor(debug_out, "debug_out", AITER_DTYPE_fp32, {t, h, 512}, device, 16);
    int erows = 0, ecap = 0;
    AITER_CHECK(bool(extend_kv) == bool(extend) && bool(extend_kv) == bool(extend_indptr),
                "extend_kv, extend and extend_indptr must be provided together");
    if(extend_kv)
    {
        AITER_CHECK(extend_kv->dim() == 2 && extend_kv->size(0) <= INT_MAX,
                    "extend dimensions must fit int32");
        AITER_CHECK(extend->numel() <= max_csr_capacity, "CSR capacity exceeds supported limit");
        erows = extend_kv->size(0);
        ecap  = extend->numel();
        check_tensor(extend_kv, "extend_kv", AITER_DTYPE_bf16, {erows, 512}, device, 16);
        check_tensor(extend, "extend", AITER_DTYPE_i32, {ecap}, device, 4);
        check_tensor(extend_indptr, "extend_indptr", AITER_DTYPE_i32, {t + 1}, device, 4);
    }
    AITER_CHECK(std::isfinite(softmax_scale) && softmax_scale > 0 &&
                    std::isfinite(softmax_scale * 1.4426950408889634f),
                "softmax_scale must be positive and finite after conversion to base-2 units");
    HipDeviceGuard guard(device);
    AITER_CHECK(get_gpu_arch() == "gfx950",
                "fused_norm_rope_attn_rope_cast requires gfx950");
    FusedNormRopeAttnRopeCastArgs a{};
    a.q      = static_cast<const B*>(q->data_ptr());
    a.pool   = static_cast<const uint8_t*>(pool->data_ptr());
    a.prefix = prefix->data_ptr();
    a.cache.layout =
        bf16_cache ? fused_kv_layout::contiguous_bf16_rows : fused_kv_layout::legacy_tagged_fp8;
    a.cache.num_rows = bf16_cache ? pool->size(0) : 0;
    a.prefix_indptr  = static_cast<const int*>(prefix_indptr->data_ptr());
    a.cos            = static_cast<const B*>(cos->data_ptr());
    a.sin            = static_cast<const B*>(sin->data_ptr());
    a.positions      = positions->data_ptr();
    if(extend_kv)
    {
        a.extend_kv     = static_cast<const B*>(extend_kv->data_ptr());
        a.extend        = static_cast<const int*>(extend->data_ptr());
        a.extend_indptr = static_cast<const int*>(extend_indptr->data_ptr());
    }
    a.sink            = attn_sink ? static_cast<const float*>(attn_sink->data_ptr()) : nullptr;
    a.debug_out       = debug_out ? static_cast<float*>(debug_out->data_ptr()) : nullptr;
    a.out             = static_cast<uint8_t*>(out->data_ptr());
    a.scales          = static_cast<uint8_t*>(out_scale->data_ptr());
    a.max_logits      = static_cast<float*>(max_logits->data_ptr());
    a.lse             = static_cast<float*>(lse->data_ptr());
    a.tokens          = t;
    a.heads           = h;
    a.extend_rows     = erows;
    a.extend_capacity = ecap;
    a.prefix_capacity = prefix->numel();
    a.pool_bytes      = pool->numel();
    a.positions64     = positions->dtype() == AITER_DTYPE_i64;
    a.fp8_qk          = fp8_qk;
    a.q_scale         = q_scale ? static_cast<const uint8_t*>(q_scale->data_ptr()) : nullptr;
    a.softmax_scale   = softmax_scale;
    a.q_norm          = q_norm;
    a.norm_eps        = norm_eps;
    if(!try_launch_fused_fp8_attention(a, !q_is_rotated, stream))
        HIP_CALL(launch_fused_norm_rope_attn_rope_cast(&a, !q_is_rotated, stream));
}

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(_fused_norm_rope_attn_rope_cast_fwd,
                                    (const aiter_tensor_t* q,
                                     const aiter_tensor_t* q_scale,
                                     const aiter_tensor_t* unified_kv,
                                     const aiter_tensor_t* unified_kv_scale,
                                     const aiter_tensor_t* kv_indices_prefix,
                                     const aiter_tensor_t* kv_indptr_prefix,
                                     const aiter_tensor_t* kv,
                                     const aiter_tensor_t* kv_scale,
                                     const aiter_tensor_t* kv_indices_extend,
                                     const aiter_tensor_t* kv_indptr_extend,
                                     const aiter_tensor_t* attn_sink,
                                     float softmax_scale,
                                     const aiter_tensor_t* cos,
                                     const aiter_tensor_t* sin,
                                     const aiter_tensor_t* positions,
                                     const aiter_tensor_t* out,
                                     const aiter_tensor_t* out_scale,
                                     const aiter_tensor_t* max_logits,
                                     const aiter_tensor_t* lse,
                                     const aiter_tensor_t* debug_out,
                                     int q_is_rotated,
                                     int q_norm,
                                     float norm_eps,
                                     int64_t num_splits,
                                     const aiter_tensor_t* workspace,
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
                                     softmax_scale,
                                     cos,
                                     sin,
                                     positions,
                                     out,
                                     out_scale,
                                     max_logits,
                                     lse,
                                     debug_out,
                                     q_is_rotated,
                                     q_norm,
                                     norm_eps,
                                     num_splits,
                                     workspace,
                                     stream))
{
    AITER_CHECK(q && unified_kv && unified_kv_scale && kv_indices_prefix && kv_indptr_prefix &&
                    cos && sin && positions && out && out_scale,
                "required tensors must not be null");
    AITER_CHECK(q->is_gpu() && q->dim() == 3 && q->size(0) <= INT_MAX,
                "q must be a GPU tensor [T,H,512]");
    const int t = q->size(0), h = q->size(1), device = q->device_id;
    AITER_CHECK(h == 16 || h == 32 || h == 64 || h == 128,
                "two-source FP8 attention requires H16/H32/H64/H128");
    AITER_CHECK(std::isfinite(norm_eps) && norm_eps >= std::numeric_limits<float>::min(),
                "norm_eps must be positive, finite and normal in float32");
    AITER_CHECK(!q_norm || (!q_is_rotated && !q_scale),
                "q_norm requires raw BF16 Q and q_is_rotated=False");
    if(q_scale)
    {
        AITER_CHECK(q_is_rotated, "prepared FP8 Q must already be rotated");
        check_tensor(q, "q", AITER_DTYPE_fp8, {t, h, 512}, device, 32);
        AITER_CHECK(q_scale->dtype() == AITER_DTYPE_u8 || q_scale->dtype() == AITER_DTYPE_fp8_e8m0,
                    "q_scale must be E8M0 or uint8");
        check_tensor(q_scale, "q_scale", q_scale->dtype(), {t, h, 16}, device);
    }
    else
        check_tensor(q, "q", AITER_DTYPE_bf16, {t, h, 512}, device, 16);
    check_row_view(unified_kv, "unified_kv", 512, device, false);
    check_row_view(unified_kv_scale, "unified_kv_scale", 16, device, true);
    AITER_CHECK(unified_kv->size(0) == unified_kv_scale->size(0),
                "prefix data/scale row counts differ");
    AITER_CHECK(kv_indices_prefix->numel() <= max_csr_capacity,
                    "CSR capacity exceeds supported limit");
    check_tensor(kv_indices_prefix,
                 "kv_indices_prefix",
                 AITER_DTYPE_i32,
                 {static_cast<int64_t>(kv_indices_prefix->numel())},
                 device,
                 4);
    check_tensor(kv_indptr_prefix, "kv_indptr_prefix", AITER_DTYPE_i32, {t + 1}, device, 4);
    AITER_CHECK(bool(kv) == bool(kv_scale) && bool(kv) == bool(kv_indices_extend) &&
                    bool(kv) == bool(kv_indptr_extend),
                "kv, kv_scale and both extend CSR tensors must be supplied together");
    if(kv)
    {
        check_row_view(kv, "kv", 512, device, false);
        check_row_view(kv_scale, "kv_scale", 16, device, true);
        AITER_CHECK(kv->size(0) == kv_scale->size(0), "extend data/scale row counts differ");
        AITER_CHECK(kv_indices_extend->numel() <= max_csr_capacity,
                    "CSR capacity exceeds supported limit");
        check_tensor(kv_indices_extend,
                     "kv_indices_extend",
                     AITER_DTYPE_i32,
                     {static_cast<int64_t>(kv_indices_extend->numel())},
                     device,
                     4);
        check_tensor(kv_indptr_extend, "kv_indptr_extend", AITER_DTYPE_i32, {t + 1}, device, 4);
    }
    AITER_CHECK(positions->dtype() == AITER_DTYPE_i32 || positions->dtype() == AITER_DTYPE_i64,
                "positions must be int32 or int64");
    check_tensor(
        positions, "positions", positions->dtype(), {t}, device, positions->element_size());
    AITER_CHECK(cos->dim() == 2 && (!t || cos->size(0) > 0), "cos must be [P,32]");
    check_tensor(cos, "cos", AITER_DTYPE_bf16, {cos->size(0), 32}, device, 2);
    check_tensor(sin, "sin", AITER_DTYPE_bf16, {cos->size(0), 32}, device, 2);
    check_tensor(out, "out", AITER_DTYPE_fp8, {t, h / 8, 4096}, device, 16);
    check_tensor(out_scale, "out_scale", AITER_DTYPE_fp8_e8m0, {t, h / 8, 128}, device);
    if(max_logits)
        check_tensor(max_logits, "max_logits", AITER_DTYPE_fp32, {t, h}, device, 4);
    if(lse)
        check_tensor(lse, "lse", AITER_DTYPE_fp32, {t, h}, device, 4);
    if(attn_sink)
        check_tensor(attn_sink, "attn_sink", AITER_DTYPE_fp32, {h}, device, 4);
    if(debug_out)
        check_tensor(debug_out, "debug_out", AITER_DTYPE_fp32, {t, h, 512}, device, 16);
    AITER_CHECK(num_splits >= 1 && num_splits <= 32, "num_splits must be in [1,32]");
    AITER_CHECK(bool(workspace) == (num_splits > 1),
                "split decode requires workspace, single split must omit it");
    if(workspace)
        check_tensor(workspace, "workspace", AITER_DTYPE_fp32, {t, num_splits, h, 516}, device, 16);
    const aiter_tensor_t* outputs[] = {workspace, out, out_scale, max_logits, lse, debug_out};
    const aiter_tensor_t* inputs[]  = {q,
                                       q_scale,
                                       unified_kv,
                                       unified_kv_scale,
                                       kv_indices_prefix,
                                       kv_indptr_prefix,
                                       kv,
                                       kv_scale,
                                       kv_indices_extend,
                                       kv_indptr_extend,
                                       cos,
                                       sin,
                                       positions,
                                       attn_sink};
    for(size_t i = 0; i < 6; ++i)
    {
        for(const auto* input : inputs)
            AITER_CHECK(!overlaps(outputs[i], input), "outputs must not overlap inputs");
        for(size_t j = 0; j < i; ++j)
            AITER_CHECK(!overlaps(outputs[i], outputs[j]), "outputs must not overlap one another");
    }
    AITER_CHECK(std::isfinite(softmax_scale) && softmax_scale > 0 &&
                    std::isfinite(softmax_scale * 1.4426950408889634f),
                "softmax_scale must be positive and finite after conversion to base-2 units");
    HipDeviceGuard guard(device);
    AITER_CHECK(get_gpu_arch() == "gfx950",
                "fused_norm_rope_attn_rope_cast requires gfx950");
    FusedNormRopeAttnRopeCastArgs a{};
    a.q               = static_cast<const B*>(q->data_ptr());
    a.q_scale         = q_scale ? static_cast<const uint8_t*>(q_scale->data_ptr()) : nullptr;
    a.pool            = static_cast<const uint8_t*>(unified_kv->data_ptr());
    a.prefix          = kv_indices_prefix->data_ptr();
    a.prefix_indptr   = static_cast<const int*>(kv_indptr_prefix->data_ptr());
    a.prefix_capacity = kv_indices_prefix->numel();
    a.cache           = {static_cast<const uint8_t*>(unified_kv_scale->data_ptr()),
                         unified_kv->stride(0),
                         unified_kv_scale->stride(0),
                         fused_kv_layout::strided_fp8_rows,
                         int(unified_kv->size(0))};
    if(kv)
    {
        a.extend_kv       = static_cast<const B*>(kv->data_ptr());
        a.extend          = static_cast<const int*>(kv_indices_extend->data_ptr());
        a.extend_indptr   = static_cast<const int*>(kv_indptr_extend->data_ptr());
        a.extend_capacity = kv_indices_extend->numel();
        a.extend_rows     = kv->size(0);
        a.extend_cache    = {static_cast<const uint8_t*>(kv_scale->data_ptr()),
                             kv->stride(0),
                             kv_scale->stride(0),
                             fused_kv_layout::strided_fp8_rows,
                             int(kv->size(0))};
    }
    a.cos           = static_cast<const B*>(cos->data_ptr());
    a.sin           = static_cast<const B*>(sin->data_ptr());
    a.positions     = positions->data_ptr();
    a.positions64   = positions->dtype() == AITER_DTYPE_i64;
    a.out           = static_cast<uint8_t*>(out->data_ptr());
    a.scales        = static_cast<uint8_t*>(out_scale->data_ptr());
    a.max_logits    = max_logits ? static_cast<float*>(max_logits->data_ptr()) : nullptr;
    a.lse           = lse ? static_cast<float*>(lse->data_ptr()) : nullptr;
    a.debug_out     = debug_out ? static_cast<float*>(debug_out->data_ptr()) : nullptr;
    a.sink          = attn_sink ? static_cast<const float*>(attn_sink->data_ptr()) : nullptr;
    a.tokens        = t;
    a.heads         = h;
    a.fp8_qk        = true;
    a.softmax_scale = softmax_scale;
    a.q_norm        = q_norm;
    a.norm_eps      = norm_eps;
    a.num_splits    = num_splits;
    a.partial       = workspace ? static_cast<float*>(workspace->data_ptr()) : nullptr;
    if(!try_launch_fused_fp8_attention(a, !q_is_rotated, stream))
        HIP_CALL(launch_fused_norm_rope_attn_rope_cast(&a, !q_is_rotated, stream));
}
