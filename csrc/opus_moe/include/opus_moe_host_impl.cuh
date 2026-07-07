// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "opus_moe.h"

#include "aiter_hip_common.h"
#include "aiter_stream.h"

#include "opus_moe_arch.cuh"
#include "gfx950/opus_moe_arch_gfx950.cuh"
#include "gfx950/a8w4/opus_moe_stage2_a8w4_decode_dispatch_gfx950.cuh"
#include "opus_moe_common.cuh"

#include <cstdint>
#include <cstdlib>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

namespace {
void check_contiguous_last_dim(const aiter_tensor_t& t, const char* name)
{
    AITER_CHECK(t.dim() > 0, name, " must have at least one dimension");
    AITER_CHECK(t.stride(-1) == 1, name, " last dimension must be contiguous");
}

void check_tensor(const aiter_tensor_t& t,
                  const char* name,
                  int expected_dim,
                  const char* expected_shape,
                  AiterDtype expected_dtype,
                  const char* expected_dtype_name)
{
    AITER_CHECK(t.dim() == expected_dim,
                name,
                " must be ",
                expected_dim,
                "-D ",
                expected_shape,
                ", got ndim=",
                t.dim());
    AITER_CHECK(t.dtype() == expected_dtype,
                name,
                " must be ",
                expected_dtype_name,
                ", got ",
                AiterDtype_to_str(t.dtype()));
    check_contiguous_last_dim(t, name);
}

void check_i32_metadata(const aiter_tensor_t& t, const char* name, bool non_empty)
{
    AITER_CHECK(t.dim() == 1, name, " must be 1-D, got ndim=", t.dim());
    AITER_CHECK(t.dtype() == AITER_DTYPE_i32,
                name,
                " must be int32, got ",
                AiterDtype_to_str(t.dtype()));
    AITER_CHECK(t.is_contiguous(), name, " must be contiguous");
    if(non_empty)
        AITER_CHECK(t.size(0) > 0, name, " must be non-empty");
}

void check_same_device(const aiter_tensor_t& ref,
                       const char* ref_name,
                       const aiter_tensor_t& t,
                       const char* name)
{
    AITER_CHECK(t.device_id == ref.device_id,
                name,
                " must be on the same device as ",
                ref_name);
}

void check_sorted_weights(const std::optional<aiter_tensor_t>& sorted_weights)
{
    if(!sorted_weights.has_value())
        return;
    AITER_CHECK(sorted_weights->dtype() == AITER_DTYPE_fp32,
                "sorted_weights must be fp32 when provided, got ",
                AiterDtype_to_str(sorted_weights->dtype()));
    AITER_CHECK(sorted_weights->is_contiguous(), "sorted_weights must be contiguous");
}

int select_a8w4_kernel_id(int requested_kernel_id,
                          int block_m,
                          int logical_inter_dim,
                          int inter_dim_pad)
{
    int selected_kernel_id = requested_kernel_id;
    if(selected_kernel_id == opus_moe::kStage2KidAuto)
    {
        // Auto selects direct-atomic; route-out kernels must be requested explicitly.
        selected_kernel_id = opus_moe::stage2_a8w4_auto_direct_atomic_kid(
            logical_inter_dim, inter_dim_pad, block_m);
    }
    AITER_CHECK(opus_moe::stage2_a8w4_kid_is_valid(selected_kernel_id),
                "opus_moe_stage2_a8w4_decode_fwd got unsupported kernel_id=",
                selected_kernel_id,
                " (",
                opus_moe::stage2_a8w4_kid_name(selected_kernel_id),
                ")");
    // Validate that the caller sorted with the block_m required by the selected kid.
    AITER_CHECK(opus_moe::stage2_a8w4_kid_block_m(selected_kernel_id) == block_m,
                "kernel_id=",
                selected_kernel_id,
                " (",
                opus_moe::stage2_a8w4_kid_name(selected_kernel_id),
                ") requires block_m=",
                opus_moe::stage2_a8w4_kid_block_m(selected_kernel_id),
                ", got ",
                block_m);
    return selected_kernel_id;
}

void check_a8w4_output_layout(const aiter_tensor_t& out,
                              int selected_kernel_id,
                              int token_num,
                              int actual_topk,
                              int model_dim)
{
    const bool route_out_mode =
        opus_moe::stage2_a8w4_kid_uses_route_out(selected_kernel_id);
    const bool route_out_fp8 =
        opus_moe::stage2_a8w4_kid_route_fp8(selected_kernel_id);
    const int expected_output_rows = route_out_mode ? token_num * actual_topk : token_num;
    if(route_out_fp8)
    {
        check_tensor(out,
                     "out",
                     2,
                     "[token * topk, model_dim + model_dim / 8]",
                     AITER_DTYPE_u8,
                     "uint8");
        AITER_CHECK(out.size(0) == expected_output_rows &&
                        out.size(1) == model_dim + model_dim / 8,
                    "MXFP8 route-out shape must be [",
                    expected_output_rows,
                    ", ",
                    model_dim + model_dim / 8,
                    "] for kernel_id=",
                    selected_kernel_id,
                    " (",
                    opus_moe::stage2_a8w4_kid_name(selected_kernel_id),
                    ")");
    }
    else if(route_out_mode)
    {
        check_tensor(out,
                     "out",
                     2,
                     "[token * topk, model_dim]",
                     AITER_DTYPE_bf16,
                     "bf16");
        AITER_CHECK(out.size(0) == expected_output_rows && out.size(1) == model_dim,
                    "BF16 route-out shape must be [",
                    expected_output_rows,
                    ", ",
                    model_dim,
                    "] for kernel_id=",
                    selected_kernel_id,
                    " (",
                    opus_moe::stage2_a8w4_kid_name(selected_kernel_id),
                    ")");
    }
    else
    {
        check_tensor(out, "out", 2, "[token, model_dim]", AITER_DTYPE_bf16, "bf16");
        AITER_CHECK(out.size(0) == expected_output_rows && out.size(1) == model_dim,
                    "out shape must be [",
                    expected_output_rows,
                    ", ",
                    model_dim,
                    "] for kernel_id=",
                    selected_kernel_id,
                    " (",
                    opus_moe::stage2_a8w4_kid_name(selected_kernel_id),
                    ")");
    }
}

} // namespace

void opus_moe_stage2_a8w4_decode_fwd(
    aiter_tensor_t& inter_states,
    aiter_tensor_t& w2,
    aiter_tensor_t& a2_scale,
    aiter_tensor_t& w2_scale,
    aiter_tensor_t& sorted_token_ids,
    std::optional<aiter_tensor_t> sorted_weights,
    aiter_tensor_t& sorted_expert_ids,
    aiter_tensor_t& num_valid_ids,
    aiter_tensor_t& out,
    int block_m,
    int kernel_id,
    int inter_dim_pad)
{
    check_tensor(inter_states,
                 "inter_states",
                 3,
                 "[token, topk, packed_inter_dim]",
                 AITER_DTYPE_fp8,
                 "fp8");
    check_tensor(
        w2, "w2", 3, "[expert, model_dim, packed_inter_dim]", AITER_DTYPE_fp4x2, "fp4x2");
    check_tensor(
        a2_scale, "a2_scale", 2, "[route, scale_cols]", AITER_DTYPE_fp8_e8m0, "fp8_e8m0");
    check_tensor(w2_scale,
                 "w2_scale",
                 2,
                 "[expert * model_dim, scale_cols]",
                 AITER_DTYPE_fp8_e8m0,
                 "fp8_e8m0");
    check_i32_metadata(sorted_token_ids, "sorted_token_ids", false);
    check_i32_metadata(sorted_expert_ids, "sorted_expert_ids", true);
    check_i32_metadata(num_valid_ids, "num_valid_ids", true);
    check_sorted_weights(sorted_weights);
    check_same_device(inter_states, "inter_states", w2, "w2");
    check_same_device(inter_states, "inter_states", a2_scale, "a2_scale");
    check_same_device(inter_states, "inter_states", w2_scale, "w2_scale");
    check_same_device(inter_states, "inter_states", sorted_token_ids, "sorted_token_ids");
    check_same_device(inter_states, "inter_states", sorted_expert_ids, "sorted_expert_ids");
    check_same_device(inter_states, "inter_states", num_valid_ids, "num_valid_ids");
    check_same_device(inter_states, "inter_states", out, "out");
    if(sorted_weights.has_value())
        check_same_device(inter_states, "inter_states", *sorted_weights, "sorted_weights");

    const int token_num = static_cast<int>(inter_states.size(0));
    const int actual_topk = static_cast<int>(inter_states.size(1));
    const int logical_inter_dim = static_cast<int>(inter_states.size(2));
    const int effective_inter_dim = logical_inter_dim - inter_dim_pad;
    const int num_experts = static_cast<int>(w2.size(0));
    const int model_dim = static_cast<int>(w2.size(1));
    const int packed_inter_dim = static_cast<int>(w2.size(2));
    const int sorted_blocks = static_cast<int>(sorted_expert_ids.size(0));
    const int scale_cols =
        logical_inter_dim / opus_moe::kStage2A8W4DecodeScaleGroupLogicalK;

    AITER_CHECK(opus_moe::stage2_a8w4_block_m_is_valid(block_m),
                "Opus A8W4 stage2 has no generated kid for block_m=",
                block_m);

    const int selected_kernel_id =
        select_a8w4_kernel_id(
            kernel_id, block_m, logical_inter_dim, inter_dim_pad);
    const int kernel_block_n = opus_moe::stage2_a8w4_kid_block_n(selected_kernel_id);
    const int expected_logical_inter_dim =
        opus_moe::stage2_a8w4_kid_logical_inter_dim(selected_kernel_id);
    const int expected_inter_dim_pad =
        opus_moe::stage2_a8w4_kid_inter_dim_pad(selected_kernel_id);
    const int expected_effective_inter_dim =
        opus_moe::stage2_a8w4_kid_effective_inter_dim(selected_kernel_id);
    const int expected_k_tiles =
        opus_moe::stage2_a8w4_kid_k_tiles(selected_kernel_id);

    AITER_CHECK(actual_topk > 0,
                "Opus A8W4 stage2 requires positive topk, got ",
                actual_topk);
    AITER_CHECK(model_dim > 0,
                "Opus A8W4 stage2 requires positive model_dim, got ",
                model_dim);
    AITER_CHECK(num_experts > 0,
                "Opus A8W4 stage2 requires positive experts, got ",
                num_experts);
    AITER_CHECK(kernel_block_n > 0 && model_dim % kernel_block_n == 0,
                "Opus A8W4 stage2 kernel_id=",
                selected_kernel_id,
                " (",
                opus_moe::stage2_a8w4_kid_name(selected_kernel_id),
                ") requires model_dim to be a multiple of block_n=",
                kernel_block_n,
                ", got ",
                model_dim);
    AITER_CHECK(logical_inter_dim == expected_logical_inter_dim &&
                    inter_dim_pad == expected_inter_dim_pad &&
                    effective_inter_dim == expected_effective_inter_dim,
                "Opus A8W4 stage2 family expects logical/effective inter_dim "
                "logical=",
                expected_logical_inter_dim,
                " effective=",
                expected_effective_inter_dim,
                " inter_dim_pad=",
                expected_inter_dim_pad,
                " k_tiles=",
                expected_k_tiles);
    AITER_CHECK(packed_inter_dim ==
                    logical_inter_dim / opus_moe::kStage2A8W4DecodeFp4ValuesPerByte,
                "w2 packed inter_dim mismatch, expected ",
                logical_inter_dim / opus_moe::kStage2A8W4DecodeFp4ValuesPerByte,
                ", got ",
                packed_inter_dim);
    AITER_CHECK(a2_scale.size(0) >= sorted_token_ids.size(0) &&
                    a2_scale.size(1) >= scale_cols,
                "a2_scale shape must cover sorted route rows and logical_inter_dim / ",
                opus_moe::kStage2A8W4DecodeScaleGroupLogicalK);
    AITER_CHECK(w2_scale.size(0) >= num_experts * model_dim &&
                    w2_scale.size(1) >= scale_cols,
                "w2_scale shape must be at least [expert * model_dim, logical_inter_dim / ",
                opus_moe::kStage2A8W4DecodeScaleGroupLogicalK,
                "]");

    const bool route_out_fp8 =
        opus_moe::stage2_a8w4_kid_route_fp8(selected_kernel_id);
    AITER_CHECK(!route_out_fp8 || model_dim % 8 == 0,
                "MXFP8 route-out requires model_dim to be a multiple of 8, got ",
                model_dim);
    check_a8w4_output_layout(out, selected_kernel_id, token_num, actual_topk, model_dim);
    AITER_CHECK(out.stride(1) == 1,
                "Opus A8W4 stage2 expects contiguous columns in out, got stride(1)=",
                out.stride(1));
    if(token_num == 0 || model_dim == 0 || logical_inter_dim == 0)
        return;

    opus_moe_stage2_a8w4_kargs kargs{};
    kargs.inter_states_fp8 = reinterpret_cast<const uint8_t*>(inter_states.data_ptr());
    kargs.w2_fp4 = reinterpret_cast<const uint8_t*>(w2.data_ptr());
    kargs.a2_scale_e8m0 = reinterpret_cast<const uint8_t*>(a2_scale.data_ptr());
    kargs.w2_scale_e8m0 = reinterpret_cast<const uint8_t*>(w2_scale.data_ptr());
    kargs.sorted_token_ids = reinterpret_cast<const int32_t*>(sorted_token_ids.data_ptr());
    kargs.sorted_weights = sorted_weights.has_value()
                               ? reinterpret_cast<const float*>(sorted_weights->data_ptr())
                               : nullptr;
    kargs.sorted_expert_ids =
        reinterpret_cast<const int32_t*>(sorted_expert_ids.data_ptr());
    kargs.num_valid_ids = reinterpret_cast<const int32_t*>(num_valid_ids.data_ptr());
    kargs.out_bf16 = reinterpret_cast<hip_bfloat16*>(out.data_ptr());
    kargs.stride_a_t = inter_states.stride(0);
    kargs.stride_a_k = inter_states.stride(1);
    kargs.stride_w_e = w2.stride(0);
    kargs.stride_a_scale_route = a2_scale.stride(0);
    kargs.stride_w_scale_row = w2_scale.stride(0);
    kargs.stride_o_t = route_out_fp8 ? 0 : out.stride(0);
    kargs.token_num = token_num;
    kargs.topk = actual_topk;
    kargs.num_experts = num_experts;
    kargs.model_dim = model_dim;
    kargs.sorted_blocks = sorted_blocks;
    // Keep runtime route-out guard for more stable MXFP8 codegen.
    kargs.route_out_fp8 = route_out_fp8 ? 1 : 0;
    kargs.route_out_row_bytes = route_out_fp8 ? out.stride(0) : 0;

    HipDeviceGuard guard(inter_states.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    opus_moe_stage2_a8w4_decode_dispatch_gfx950(selected_kernel_id, kargs, stream);
    HIP_CALL_LAUNCH(hipGetLastError());
}

void opus_moe_stage2_reduce_token_slot_route_output_fwd(aiter_tensor_t& route_out,
                                                        aiter_tensor_t& out,
                                                        int topk,
                                                        int block_n)
{
    // Derive fp8/bf16 route_out mode from dtype to stay in sync with the decode kid.
    const int route_out_fp8 = (route_out.dtype() == AITER_DTYPE_u8) ? 1 : 0;
    check_tensor(out, "out", 2, "[token, model_dim]", AITER_DTYPE_bf16, "bf16");
    check_same_device(route_out, "route_out", out, "out");
    AITER_CHECK(topk > 0, "route_out reduce requires positive topk, got ", topk);
    const int token_num = static_cast<int>(out.size(0));
    const int actual_topk = topk;
    const int model_dim = static_cast<int>(out.size(1));
    if(!route_out_fp8)
    {
        check_tensor(route_out, "route_out", 3, "[token, topk, model_dim]",
                     AITER_DTYPE_bf16, "bf16");
        AITER_CHECK(route_out.size(0) == token_num && route_out.size(1) == actual_topk &&
                        route_out.size(2) == model_dim,
                    "bf16 route_out shape must be [",
                    token_num,
                    ", ",
                    actual_topk,
                    ", ",
                    model_dim,
                    "]");
        AITER_CHECK(route_out.stride(0) == route_out.stride(1) * topk,
                    "route_out must be contiguous over [token, topk] rows");
    }
    else
    {
        check_tensor(route_out,
                     "route_out",
                     2,
                     "[token * topk, model_dim + model_dim / 8]",
                     AITER_DTYPE_u8,
                     "uint8");
        AITER_CHECK(model_dim % 8 == 0,
                    "MXFP8 route_out reduce requires model_dim to be a multiple of 8, got ",
                    model_dim);
        const int64_t expected_rows = static_cast<int64_t>(token_num) * actual_topk;
        const int64_t expected_cols = static_cast<int64_t>(model_dim) + model_dim / 8;
        AITER_CHECK(route_out.size(0) == expected_rows && route_out.size(1) == expected_cols,
                    "MXFP8 route_out shape must be [",
                    expected_rows,
                    ", ",
                    expected_cols,
                    "], got [",
                    route_out.size(0),
                    ", ",
                    route_out.size(1),
                    "]");
        AITER_CHECK(route_out.stride(0) >= expected_cols,
                    "MXFP8 route_out row stride must cover model_dim + model_dim / 8 bytes");
    }
    if(token_num == 0 || model_dim == 0)
        return;

    opus_moe_stage2_route_reduce_kargs kargs{};
    kargs.route_out = reinterpret_cast<const uint8_t*>(route_out.data_ptr());
    kargs.out_bf16 = reinterpret_cast<hip_bfloat16*>(out.data_ptr());
    kargs.token_num = token_num;
    kargs.topk = actual_topk;
    kargs.model_dim = model_dim;
    kargs.stride_route_out_t = route_out_fp8 ? 0 : route_out.stride(1);
    kargs.stride_o_t = out.stride(0);
    kargs.route_out_fp8 = route_out_fp8;
    kargs.route_out_row_bytes = route_out_fp8 ? route_out.stride(0) : 0;

    HipDeviceGuard guard(route_out.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    opus_moe_stage2_reduce_token_slot_route_output_launch_gfx950(
        kargs, stream, block_n);
    HIP_CALL_LAUNCH(hipGetLastError());
}
