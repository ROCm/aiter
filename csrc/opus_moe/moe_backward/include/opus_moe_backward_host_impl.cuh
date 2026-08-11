// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "opus_moe_backward.h"
#include "opus_moe_backward_arch.cuh"
#include "gfx950/opus_moe_backward_arch_gfx950.cuh"

#include "aiter_hip_common.h"
#include "aiter_stream.h"

#include <cstdint>

namespace opus_moe_backward
{
namespace detail
{

inline void check_route_metadata(const RouteMetadata& route,
                                 Family family,
                                 bool needs_sorted,
                                 bool needs_reverse,
                                 bool needs_expert_offsets)
{
    AITER_CHECK(route.token_num > 0,
                family_name(family),
                ": token_num must be positive, got ",
                route.token_num);
    AITER_CHECK(route.token_num <= static_cast<int>(kPackedTokenMask),
                family_name(family),
                ": token_num does not fit the 24-bit sorted-token encoding");
    AITER_CHECK(route.topk > 0 && route.topk <= kMaxPackedTopk,
                family_name(family),
                ": topk must be in [1, ",
                kMaxPackedTopk,
                "], got ",
                route.topk);
    if(needs_sorted || needs_expert_offsets)
        AITER_CHECK(route.num_experts > 0,
                    family_name(family),
                    ": num_experts must be positive");

    if(needs_sorted)
    {
        AITER_CHECK(route.sorted_token_ids != nullptr,
                    family_name(family),
                    ": sorted_token_ids must not be null");
        AITER_CHECK(route.sorted_expert_ids != nullptr,
                    family_name(family),
                    ": sorted_expert_ids must not be null");
        AITER_CHECK(route.num_valid_ids != nullptr,
                    family_name(family),
                    ": num_valid_ids must not be null");
        AITER_CHECK(route.sort_block_m > 0,
                    family_name(family),
                    ": sort_block_m must be positive");
        AITER_CHECK(route.sorted_capacity > 0 && route.sorted_block_capacity > 0,
                    family_name(family),
                    ": sorted metadata capacities must be positive");
    }
    if(needs_reverse)
        AITER_CHECK(route.reverse_sorted != nullptr,
                    family_name(family),
                    ": reverse_sorted must not be null");
    if(needs_expert_offsets)
        AITER_CHECK(route.expert_offsets != nullptr,
                    family_name(family),
                    ": expert_offsets must not be null");
}

inline void check_problem_dims(int model_dim, int inter_dim, Family family)
{
    AITER_CHECK(model_dim > 0,
                family_name(family),
                ": model_dim must be positive, got ",
                model_dim);
    AITER_CHECK(inter_dim > 0,
                family_name(family),
                ": inter_dim must be positive, got ",
                inter_dim);
}

inline void check_stride(int64_t stride, const char* name, Family family)
{
    AITER_CHECK(stride > 0,
                family_name(family),
                ": ",
                name,
                " must be a positive element stride, got ",
                stride);
}

inline void check_gfx950_or_fail()
{
    if(opus_get_gfx_arch() == OpusGfxArch::Gfx950)
        return;
    const auto& info = opus_get_arch_info();
    AITER_CHECK(false,
                "opus_moe_backward: only gfx950 is implemented; current device ",
                info.dev,
                " has gcnArchName='",
                info.name,
                "'");
}

template<typename Launcher, typename Kargs>
inline void invoke(Launcher launcher, const Kargs& kargs, hipStream_t stream, Family family)
{
    AITER_CHECK(launcher != nullptr,
                "opus_moe_backward: null launcher for family '",
                family_name(family),
                "'");
    if(launcher != nullptr)
        launcher(kargs, stream);
}

} // namespace detail

void launch_down_bwd_bf16(const DownBwdKargs& kargs,
                          int kernel_id,
                          hipStream_t stream)
{
    constexpr Family family = Family::DownBwd;
    detail::check_route_metadata(kargs.route, family, true, false, false);
    detail::check_problem_dims(kargs.model_dim, kargs.inter_dim, family);
    AITER_CHECK(kargs.d_out != nullptr && kargs.z != nullptr && kargs.w2 != nullptr &&
                    kargs.scores != nullptr && kargs.d_z != nullptr &&
                    kargs.a_scaled != nullptr && kargs.d_scores != nullptr,
                "down_bwd: required input/output pointer is null");
    AITER_CHECK(kargs.d_scores_parts > 0,
                "down_bwd: d_scores_parts must be positive");
    if(kargs.d_scores_parts > 1)
        AITER_CHECK(kargs.d_scores_workspace != nullptr,
                    "down_bwd: d_scores_workspace is required for multipart reduction");
    detail::check_stride(kargs.stride_do_t, "stride_do_t", family);
    detail::check_stride(kargs.stride_z_r, "stride_z_r", family);
    detail::check_stride(kargs.stride_w2_e, "stride_w2_e", family);
    detail::check_stride(kargs.stride_w2_d, "stride_w2_d", family);
    detail::check_stride(kargs.stride_score_t, "stride_score_t", family);
    detail::check_stride(kargs.stride_dz_r, "stride_dz_r", family);
    detail::check_stride(kargs.stride_a_scaled_r, "stride_a_scaled_r", family);
    detail::check_stride(kargs.stride_ds_t, "stride_ds_t", family);
    if(kargs.d_scores_parts > 1)
        detail::check_stride(
            kargs.stride_ds_workspace_r, "stride_ds_workspace_r", family);

    detail::check_gfx950_or_fail();
    detail::invoke(gfx950::dispatch_down_bwd(kernel_id), kargs, stream, family);
}

void launch_route_dx_bf16(const RouteDxKargs& kargs,
                          int kernel_id,
                          hipStream_t stream)
{
    constexpr Family family = Family::RouteDx;
    detail::check_route_metadata(kargs.route, family, true, false, false);
    detail::check_problem_dims(kargs.model_dim, kargs.inter_dim, family);
    AITER_CHECK(kargs.d_z != nullptr && kargs.w1 != nullptr && kargs.d_x_route != nullptr,
                "route_dx: required input/output pointer is null");
    detail::check_stride(kargs.stride_dz_r, "stride_dz_r", family);
    detail::check_stride(kargs.stride_w1_e, "stride_w1_e", family);
    detail::check_stride(kargs.stride_w1_i, "stride_w1_i", family);
    detail::check_stride(kargs.stride_dx_route_r, "stride_dx_route_r", family);

    detail::check_gfx950_or_fail();
    detail::invoke(gfx950::dispatch_route_dx(kernel_id), kargs, stream, family);
}

void launch_route_reduce_bf16(const RouteReduceKargs& kargs,
                              int kernel_id,
                              hipStream_t stream)
{
    constexpr Family family = Family::RouteReduce;
    detail::check_route_metadata(kargs.route, family, false, false, false);
    AITER_CHECK(kargs.model_dim > 0,
                "route_reduce: model_dim must be positive, got ",
                kargs.model_dim);
    AITER_CHECK(kargs.d_x_route != nullptr && kargs.d_x != nullptr,
                "route_reduce: required input/output pointer is null");
    detail::check_stride(kargs.stride_dx_route_r, "stride_dx_route_r", family);
    detail::check_stride(kargs.stride_dx_t, "stride_dx_t", family);

    detail::check_gfx950_or_fail();
    detail::invoke(gfx950::dispatch_route_reduce(kernel_id), kargs, stream, family);
}

void launch_router_bwd_fp32(const RouterBwdKargs& kargs,
                            int kernel_id,
                            hipStream_t stream)
{
    constexpr Family family = Family::RouterBwd;
    AITER_CHECK(kargs.token_num > 0,
                "router_bwd: token_num must be positive, got ",
                kargs.token_num);
    AITER_CHECK(kargs.topk == 1 || kargs.topk == 2 ||
                    kargs.topk == 4 || kargs.topk == 8,
                "router_bwd: topk must be in {1,2,4,8}");
    AITER_CHECK(kargs.num_experts >= kargs.topk,
                "router_bwd: num_experts must be at least topk");
    AITER_CHECK(kargs.d_scores != nullptr && kargs.scores != nullptr &&
                    kargs.topk_ids != nullptr && kargs.d_logits != nullptr,
                "router_bwd: required input/output pointer is null");
    detail::check_stride(kargs.stride_ds_t, "stride_ds_t", family);
    detail::check_stride(kargs.stride_score_t, "stride_score_t", family);
    detail::check_stride(kargs.stride_topk_id_t, "stride_topk_id_t", family);
    detail::check_stride(kargs.stride_dl_t, "stride_dl_t", family);

    detail::check_gfx950_or_fail();
    detail::invoke(gfx950::dispatch_router_bwd(kernel_id), kargs, stream, family);
}

void launch_bias_bwd_bf16(const BiasBwdKargs& kargs,
                          int kernel_id,
                          hipStream_t stream)
{
    constexpr Family family = Family::BiasBwd;
    AITER_CHECK(kargs.compute_dscore || kargs.compute_db1 || kargs.compute_db2,
                "bias_bwd: at least one output must be requested");
    AITER_CHECK(kargs.route.token_num > 0 && kargs.route.num_experts > 0,
                "bias_bwd: token and expert counts must be positive");
    AITER_CHECK(kargs.route.topk == 1 || kargs.route.topk == 2 ||
                    kargs.route.topk == 4 || kargs.route.topk == 8,
                "bias_bwd: topk must be in {1,2,4,8}");
    AITER_CHECK(kargs.route.sorted_token_ids != nullptr &&
                    kargs.route.num_valid_ids != nullptr,
                "bias_bwd: sorted_token_ids and num_valid_ids are required");
    if(kargs.compute_dscore)
        AITER_CHECK(kargs.route.sorted_expert_ids != nullptr &&
                        kargs.d_out != nullptr && kargs.b2 != nullptr &&
                        kargs.d_scores != nullptr,
                    "bias_bwd: dscore inputs/outputs are required");
    if(kargs.compute_db1)
        AITER_CHECK(kargs.route.expert_offsets != nullptr &&
                        kargs.d_z != nullptr && kargs.d_b1 != nullptr,
                    "bias_bwd: db1 inputs/outputs are required");
    if(kargs.compute_db2)
        AITER_CHECK(kargs.route.expert_offsets != nullptr &&
                        kargs.d_out != nullptr && kargs.scores != nullptr &&
                        kargs.d_b2 != nullptr,
                    "bias_bwd: db2 inputs/outputs are required");
    detail::check_problem_dims(kargs.model_dim, kargs.inter_dim, family);
    if(kargs.compute_dscore || kargs.compute_db2)
    {
        detail::check_stride(kargs.stride_do_t, "stride_do_t", family);
        detail::check_stride(kargs.stride_score_t, "stride_score_t", family);
    }
    if(kargs.compute_dscore)
    {
        detail::check_stride(kargs.stride_b2_e, "stride_b2_e", family);
        detail::check_stride(kargs.stride_ds_t, "stride_ds_t", family);
    }
    if(kargs.compute_db1)
    {
        detail::check_stride(kargs.stride_dz_r, "stride_dz_r", family);
        detail::check_stride(kargs.stride_db1_e, "stride_db1_e", family);
    }
    if(kargs.compute_db2)
        detail::check_stride(kargs.stride_db2_e, "stride_db2_e", family);

    detail::check_gfx950_or_fail();
    detail::invoke(gfx950::dispatch_bias_bwd(kernel_id), kargs, stream, family);
}

void launch_dw1_bf16(const Dw1Kargs& kargs,
                     int kernel_id,
                     hipStream_t stream)
{
    constexpr Family family = Family::Dw1;
    detail::check_route_metadata(kargs.route, family, false, false, true);
    AITER_CHECK(kargs.route.sorted_token_ids != nullptr &&
                    kargs.route.num_valid_ids != nullptr,
                "dw1: sorted_token_ids and num_valid_ids must not be null");
    detail::check_problem_dims(kargs.model_dim, kargs.inter_dim, family);
    AITER_CHECK(kargs.x != nullptr && kargs.d_z != nullptr && kargs.d_w1 != nullptr,
                "dw1: required input/output pointer is null");
    AITER_CHECK(kargs.split_k > 0, "dw1: split_k must be positive");
    if(kargs.split_k > 1)
        AITER_CHECK(kargs.workspace != nullptr,
                    "dw1: FP32 workspace is required when split_k > 1");
    detail::check_stride(kargs.stride_x_t, "stride_x_t", family);
    detail::check_stride(kargs.stride_dz_r, "stride_dz_r", family);
    detail::check_stride(kargs.stride_dw1_e, "stride_dw1_e", family);
    detail::check_stride(kargs.stride_dw1_i, "stride_dw1_i", family);
    if(kargs.split_k > 1)
        detail::check_stride(
            kargs.stride_workspace_split, "stride_workspace_split", family);

    detail::check_gfx950_or_fail();
    detail::invoke(gfx950::dispatch_dw1(kernel_id), kargs, stream, family);
}

void launch_dw2_bf16(const Dw2Kargs& kargs,
                     int kernel_id,
                     hipStream_t stream)
{
    constexpr Family family = Family::Dw2;
    detail::check_route_metadata(kargs.route, family, false, false, true);
    AITER_CHECK(kargs.route.sorted_token_ids != nullptr &&
                    kargs.route.num_valid_ids != nullptr,
                "dw2: sorted_token_ids and num_valid_ids must not be null");
    detail::check_problem_dims(kargs.model_dim, kargs.inter_dim, family);
    AITER_CHECK(kargs.d_out != nullptr && kargs.a_scaled != nullptr &&
                    kargs.d_w2 != nullptr,
                "dw2: required input/output pointer is null");
    AITER_CHECK(kargs.split_k > 0, "dw2: split_k must be positive");
    if(kargs.split_k > 1)
        AITER_CHECK(kargs.workspace != nullptr,
                    "dw2: FP32 workspace is required when split_k > 1");
    detail::check_stride(kargs.stride_do_t, "stride_do_t", family);
    detail::check_stride(kargs.stride_a_scaled_r, "stride_a_scaled_r", family);
    detail::check_stride(kargs.stride_dw2_e, "stride_dw2_e", family);
    detail::check_stride(kargs.stride_dw2_d, "stride_dw2_d", family);
    if(kargs.split_k > 1)
        detail::check_stride(
            kargs.stride_workspace_split, "stride_workspace_split", family);

    detail::check_gfx950_or_fail();
    detail::invoke(gfx950::dispatch_dw2(kernel_id), kargs, stream, family);
}

namespace detail
{

// Keep the production fixed-top-k pipeline visible as one flat K1--K5
// sequence. Validation and kargs construction remain at the public boundary.
inline void launch_fixed_pipeline(const DownBwdKargs& down,
                                  const RouteDxKargs& route_dx,
                                  const RouteReduceKargs& route_reduce,
                                  const Dw1Kargs& dw1,
                                  const Dw2Kargs& dw2,
                                  int down_kernel_id,
                                  int route_dx_kernel_id,
                                  int route_reduce_kernel_id,
                                  int dw1_kernel_id,
                                  int dw2_kernel_id,
                                  hipStream_t stream)
{
    check_gfx950_or_fail();
    invoke(gfx950::dispatch_down_bwd(down_kernel_id),
           down,
           stream,
           Family::DownBwd);
    invoke(gfx950::dispatch_route_dx(route_dx_kernel_id),
           route_dx,
           stream,
           Family::RouteDx);
    invoke(gfx950::dispatch_route_reduce(route_reduce_kernel_id),
           route_reduce,
           stream,
           Family::RouteReduce);
    invoke(gfx950::dispatch_dw1(dw1_kernel_id), dw1, stream, Family::Dw1);
    invoke(gfx950::dispatch_dw2(dw2_kernel_id), dw2, stream, Family::Dw2);
}

} // namespace detail

} // namespace opus_moe_backward

namespace
{

void check_down_tensor(const aiter_tensor_t& tensor,
                       const char* name,
                       int dimensions,
                       AiterDtype dtype,
                       const char* dtype_name)
{
    AITER_CHECK(tensor.is_gpu(), name, " must be a GPU tensor");
    AITER_CHECK(tensor.dim() == dimensions,
                name,
                " must have ",
                dimensions,
                " dimensions, got ",
                tensor.dim());
    AITER_CHECK(tensor.dtype() == dtype,
                name,
                " must have dtype ",
                dtype_name);
    AITER_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_down_same_device(const aiter_tensor_t& reference,
                            const aiter_tensor_t& tensor,
                            const char* name)
{
    AITER_CHECK(tensor.device_id == reference.device_id,
                name,
                " must be on GPU ",
                reference.device_id,
                ", got GPU ",
                tensor.device_id);
}

} // namespace

void opus_moe_router_bwd(aiter_tensor_t& d_scores,
                         aiter_tensor_t& scores,
                         aiter_tensor_t& topk_ids,
                         aiter_tensor_t& d_logits,
                         int kernel_id)
{
    check_down_tensor(d_scores, "d_scores", 2, AITER_DTYPE_fp32, "float32");
    check_down_tensor(scores, "scores", 2, AITER_DTYPE_fp32, "float32");
    check_down_tensor(topk_ids, "topk_ids", 2, AITER_DTYPE_i32, "int32");
    check_down_tensor(d_logits, "d_logits", 2, AITER_DTYPE_fp32, "float32");

    check_down_same_device(d_scores, scores, "scores");
    check_down_same_device(d_scores, topk_ids, "topk_ids");
    check_down_same_device(d_scores, d_logits, "d_logits");

    const int token_num = static_cast<int>(scores.size(0));
    const int topk = static_cast<int>(scores.size(1));
    const int num_experts = static_cast<int>(d_logits.size(1));
    AITER_CHECK(token_num > 0, "router_bwd: scores must contain tokens");
    AITER_CHECK(topk == 1 || topk == 2 || topk == 4 || topk == 8,
                "router_bwd: topk must be in {1,2,4,8}");
    AITER_CHECK(num_experts >= topk,
                "router_bwd: d_logits expert dimension must be at least topk");
    AITER_CHECK(d_scores.size(0) == token_num &&
                    d_scores.size(1) == topk,
                "router_bwd: d_scores must have the same [T,K] shape as scores");
    AITER_CHECK(topk_ids.size(0) == token_num && topk_ids.size(1) == topk,
                "router_bwd: topk_ids must have the same [T,K] shape as scores");
    AITER_CHECK(d_logits.size(0) == token_num,
                "router_bwd: d_logits must have shape [T,E]");

    opus_moe_backward::RouterBwdKargs args{};
    args.d_scores = reinterpret_cast<const float*>(d_scores.data_ptr());
    args.scores = reinterpret_cast<const float*>(scores.data_ptr());
    args.topk_ids = reinterpret_cast<const int32_t*>(topk_ids.data_ptr());
    args.d_logits = reinterpret_cast<float*>(d_logits.data_ptr());
    args.token_num = token_num;
    args.topk = topk;
    args.num_experts = num_experts;
    args.stride_ds_t = d_scores.stride(0);
    args.stride_score_t = scores.stride(0);
    args.stride_topk_id_t = topk_ids.stride(0);
    args.stride_dl_t = d_logits.stride(0);

    HipDeviceGuard guard(d_scores.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    opus_moe_backward::launch_router_bwd_fp32(args, kernel_id, stream);
    HIP_CALL_LAUNCH(hipGetLastError());
}

void opus_moe_varlen_router_bwd(aiter_tensor_t& d_scores,
                                aiter_tensor_t& scores,
                                aiter_tensor_t& route_expert_ids,
                                aiter_tensor_t& token_route_offsets,
                                aiter_tensor_t& d_logits,
                                int kernel_id)
{
    check_down_tensor(d_scores, "d_scores", 1, AITER_DTYPE_fp32, "float32");
    check_down_tensor(scores, "scores", 1, AITER_DTYPE_fp32, "float32");
    check_down_tensor(route_expert_ids,
                      "route_expert_ids",
                      1,
                      AITER_DTYPE_i32,
                      "int32");
    check_down_tensor(token_route_offsets,
                      "token_route_offsets",
                      1,
                      AITER_DTYPE_i32,
                      "int32");
    check_down_tensor(d_logits, "d_logits", 2, AITER_DTYPE_fp32, "float32");
    check_down_same_device(d_scores, scores, "scores");
    check_down_same_device(d_scores, route_expert_ids, "route_expert_ids");
    check_down_same_device(
        d_scores, token_route_offsets, "token_route_offsets");
    check_down_same_device(d_scores, d_logits, "d_logits");

    const int route_count = static_cast<int>(scores.numel());
    const int token_num = static_cast<int>(d_logits.size(0));
    const int num_experts = static_cast<int>(d_logits.size(1));
    AITER_CHECK(d_scores.numel() == route_count &&
                    route_expert_ids.numel() == route_count,
                "varlen router: route tensors must have shape [R]");
    AITER_CHECK(token_route_offsets.numel() == token_num + 1,
                "varlen router: token_route_offsets must have T+1 entries");
    AITER_CHECK(token_num > 0 && num_experts > 0,
                "varlen router: d_logits must have positive [T,E]");

    opus_moe_backward::RouterBwdKargs args{};
    args.d_scores = reinterpret_cast<const float*>(d_scores.data_ptr());
    args.scores = reinterpret_cast<const float*>(scores.data_ptr());
    args.topk_ids =
        reinterpret_cast<const int32_t*>(route_expert_ids.data_ptr());
    args.token_route_offsets =
        reinterpret_cast<const int32_t*>(token_route_offsets.data_ptr());
    args.d_logits = reinterpret_cast<float*>(d_logits.data_ptr());
    args.token_num = token_num;
    args.topk = route_count;
    args.num_experts = num_experts;
    args.stride_ds_t = 1;
    args.stride_score_t = 1;
    args.stride_topk_id_t = 1;
    args.stride_dl_t = d_logits.stride(0);

    HipDeviceGuard guard(d_scores.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    opus_moe_backward::detail::check_gfx950_or_fail();
    constexpr int varlen_kid = 100;
    opus_moe_backward::detail::invoke(
        opus_moe_backward::gfx950::dispatch_router_bwd(
            kernel_id == opus_moe_backward::kKernelAuto ? varlen_kid
                                                        : kernel_id),
        args,
        stream,
        opus_moe_backward::Family::RouterBwd);
    HIP_CALL_LAUNCH(hipGetLastError());
}

void opus_moe_db1_bwd(aiter_tensor_t& d_z,
                      aiter_tensor_t& sorted_token_ids,
                      aiter_tensor_t& num_valid_ids,
                      aiter_tensor_t& expert_padded_offsets,
                      aiter_tensor_t& d_b1,
                      int token_num,
                      int topk,
                      int block_m,
                      int kernel_id)
{
    check_down_tensor(d_z, "d_z", 2, AITER_DTYPE_bf16, "bfloat16");
    check_down_tensor(
        sorted_token_ids, "sorted_token_ids", 1, AITER_DTYPE_i32, "int32");
    check_down_tensor(
        num_valid_ids, "num_valid_ids", 1, AITER_DTYPE_i32, "int32");
    check_down_tensor(expert_padded_offsets,
                      "expert_padded_offsets",
                      1,
                      AITER_DTYPE_i32,
                      "int32");
    check_down_tensor(d_b1, "d_b1", 2, AITER_DTYPE_bf16, "bfloat16");
    check_down_same_device(d_z, sorted_token_ids, "sorted_token_ids");
    check_down_same_device(d_z, num_valid_ids, "num_valid_ids");
    check_down_same_device(d_z, expert_padded_offsets, "expert_padded_offsets");
    check_down_same_device(d_z, d_b1, "d_b1");

    const int num_experts = static_cast<int>(d_b1.size(0));
    const int gate_up_dim = static_cast<int>(d_b1.size(1));
    AITER_CHECK(token_num > 0, "db1: token_num must be positive");
    AITER_CHECK(topk == 1 || topk == 2 || topk == 4 || topk == 8,
                "db1: topk must be in {1,2,4,8}");
    AITER_CHECK(gate_up_dim > 0 && gate_up_dim % 2 == 0,
                "db1: d_b1 must have shape [E,2I]");
    AITER_CHECK(d_z.size(0) == sorted_token_ids.size(0) &&
                    d_z.size(1) == gate_up_dim,
                "db1: d_z must have shape [sorted_capacity,2I]");
    AITER_CHECK(expert_padded_offsets.numel() == num_experts + 1,
                "db1: expert_padded_offsets must contain E+1 entries");
    AITER_CHECK(num_valid_ids.numel() >= 1,
                "db1: num_valid_ids must not be empty");
    AITER_CHECK(block_m == 32, "db1: first kernel requires block_m=32");

    opus_moe_backward::BiasBwdKargs args{};
    args.d_z = reinterpret_cast<const hip_bfloat16*>(d_z.data_ptr());
    args.route.sorted_token_ids =
        reinterpret_cast<const int32_t*>(sorted_token_ids.data_ptr());
    args.route.num_valid_ids =
        reinterpret_cast<const int32_t*>(num_valid_ids.data_ptr());
    args.route.expert_offsets =
        reinterpret_cast<const int32_t*>(expert_padded_offsets.data_ptr());
    args.route.token_num = token_num;
    args.route.topk = topk;
    args.route.num_experts = num_experts;
    args.route.sort_block_m = block_m;
    args.route.sorted_capacity = static_cast<int>(sorted_token_ids.size(0));
    args.route.sorted_block_capacity =
        (args.route.sorted_capacity + block_m - 1) / block_m;
    args.d_b1 = reinterpret_cast<hip_bfloat16*>(d_b1.data_ptr());
    args.model_dim = 1;
    args.inter_dim = gate_up_dim / 2;
    args.compute_db1 = true;
    args.stride_dz_r = d_z.stride(0);
    args.stride_db1_e = d_b1.stride(0);

    HipDeviceGuard guard(d_z.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    opus_moe_backward::launch_bias_bwd_bf16(args, kernel_id, stream);
    HIP_CALL_LAUNCH(hipGetLastError());
}

void opus_moe_bias_down_bwd(aiter_tensor_t& d_out,
                            aiter_tensor_t& scores,
                            aiter_tensor_t& b2,
                            aiter_tensor_t& sorted_token_ids,
                            aiter_tensor_t& sorted_expert_ids,
                            aiter_tensor_t& num_valid_ids,
                            aiter_tensor_t& expert_padded_offsets,
                            aiter_tensor_t& d_scores,
                            aiter_tensor_t& d_b2,
                            int block_m,
                            int kernel_id)
{
    check_down_tensor(d_out, "d_out", 2, AITER_DTYPE_bf16, "bfloat16");
    check_down_tensor(scores, "scores", 2, AITER_DTYPE_fp32, "float32");
    check_down_tensor(b2, "b2", 2, AITER_DTYPE_bf16, "bfloat16");
    check_down_tensor(
        sorted_token_ids, "sorted_token_ids", 1, AITER_DTYPE_i32, "int32");
    check_down_tensor(sorted_expert_ids,
                      "sorted_expert_ids",
                      1,
                      AITER_DTYPE_i32,
                      "int32");
    check_down_tensor(
        num_valid_ids, "num_valid_ids", 1, AITER_DTYPE_i32, "int32");
    check_down_tensor(expert_padded_offsets,
                      "expert_padded_offsets",
                      1,
                      AITER_DTYPE_i32,
                      "int32");
    check_down_tensor(d_scores, "d_scores", 2, AITER_DTYPE_fp32, "float32");
    check_down_tensor(d_b2, "d_b2", 2, AITER_DTYPE_bf16, "bfloat16");
    check_down_same_device(d_out, scores, "scores");
    check_down_same_device(d_out, b2, "b2");
    check_down_same_device(d_out, sorted_token_ids, "sorted_token_ids");
    check_down_same_device(d_out, sorted_expert_ids, "sorted_expert_ids");
    check_down_same_device(d_out, num_valid_ids, "num_valid_ids");
    check_down_same_device(
        d_out, expert_padded_offsets, "expert_padded_offsets");
    check_down_same_device(d_out, d_scores, "d_scores");
    check_down_same_device(d_out, d_b2, "d_b2");

    const int token_num = static_cast<int>(d_out.size(0));
    const int model_dim = static_cast<int>(d_out.size(1));
    const int num_experts = static_cast<int>(b2.size(0));
    const int topk = static_cast<int>(scores.size(1));
    const int sorted_capacity = static_cast<int>(sorted_token_ids.size(0));
    AITER_CHECK(scores.size(0) == token_num &&
                    d_scores.size(0) == token_num &&
                    d_scores.size(1) == topk,
                "bias_down: scores and d_scores must have shape [T,K]");
    AITER_CHECK(topk == 1 || topk == 2 || topk == 4 || topk == 8,
                "bias_down: topk must be in {1,2,4,8}");
    AITER_CHECK(b2.size(1) == model_dim &&
                    d_b2.size(0) == num_experts &&
                    d_b2.size(1) == model_dim,
                "bias_down: b2 and d_b2 must have shape [E,D]");
    AITER_CHECK(expert_padded_offsets.numel() == num_experts + 1,
                "bias_down: expert_padded_offsets must contain E+1 entries");
    AITER_CHECK(num_valid_ids.numel() >= 1,
                "bias_down: num_valid_ids must not be empty");
    AITER_CHECK(sorted_expert_ids.size(0) * block_m >= sorted_capacity,
                "bias_down: sorted_expert_ids does not cover sorted capacity");
    AITER_CHECK(block_m == 32,
                "bias_down: first kernel requires block_m=32");

    opus_moe_backward::BiasBwdKargs args{};
    args.d_out = reinterpret_cast<const hip_bfloat16*>(d_out.data_ptr());
    args.scores = reinterpret_cast<const float*>(scores.data_ptr());
    args.b2 = reinterpret_cast<const hip_bfloat16*>(b2.data_ptr());
    args.route.sorted_token_ids =
        reinterpret_cast<const int32_t*>(sorted_token_ids.data_ptr());
    args.route.sorted_expert_ids =
        reinterpret_cast<const int32_t*>(sorted_expert_ids.data_ptr());
    args.route.num_valid_ids =
        reinterpret_cast<const int32_t*>(num_valid_ids.data_ptr());
    args.route.expert_offsets =
        reinterpret_cast<const int32_t*>(expert_padded_offsets.data_ptr());
    args.route.token_num = token_num;
    args.route.topk = topk;
    args.route.num_experts = num_experts;
    args.route.sort_block_m = block_m;
    args.route.sorted_capacity = sorted_capacity;
    args.route.sorted_block_capacity =
        static_cast<int>(sorted_expert_ids.size(0));
    args.d_scores = reinterpret_cast<float*>(d_scores.data_ptr());
    args.d_b2 = reinterpret_cast<hip_bfloat16*>(d_b2.data_ptr());
    args.model_dim = model_dim;
    args.inter_dim = 1;
    args.compute_dscore = true;
    args.compute_db2 = true;
    args.stride_do_t = d_out.stride(0);
    args.stride_score_t = scores.stride(0);
    args.stride_b2_e = b2.stride(0);
    args.stride_ds_t = d_scores.stride(0);
    args.stride_db2_e = d_b2.stride(0);

    HipDeviceGuard guard(d_out.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    opus_moe_backward::launch_bias_bwd_bf16(args, kernel_id, stream);
    HIP_CALL_LAUNCH(hipGetLastError());
}

void opus_moe_varlen_down_bwd(aiter_tensor_t& d_out,
                              aiter_tensor_t& z,
                              aiter_tensor_t& w2,
                              aiter_tensor_t& scores,
                              aiter_tensor_t& b2,
                              aiter_tensor_t& sorted_route_ids,
                              aiter_tensor_t& sorted_expert_ids,
                              aiter_tensor_t& num_valid_ids,
                              aiter_tensor_t& route_to_token,
                              aiter_tensor_t& expert_padded_offsets,
                              aiter_tensor_t& d_scores_workspace,
                              aiter_tensor_t& d_z,
                              aiter_tensor_t& a_scaled,
                              aiter_tensor_t& d_scores,
                              aiter_tensor_t& d_w2,
                              aiter_tensor_t& d_b2,
                              int block_m,
                              bool has_bias,
                              bool compute_dz,
                              bool compute_dw2,
                              bool compute_dscore,
                              bool compute_db2,
                              int down_kernel_id,
                              int dw2_kernel_id,
                              int bias_kernel_id)
{
    check_down_tensor(d_out, "d_out", 2, AITER_DTYPE_bf16, "bfloat16");
    check_down_tensor(z, "z", 2, AITER_DTYPE_bf16, "bfloat16");
    check_down_tensor(w2, "w2", 3, AITER_DTYPE_bf16, "bfloat16");
    check_down_tensor(scores, "scores", 1, AITER_DTYPE_fp32, "float32");
    check_down_tensor(b2, "b2", 2, AITER_DTYPE_bf16, "bfloat16");
    check_down_tensor(
        sorted_route_ids, "sorted_route_ids", 1, AITER_DTYPE_i32, "int32");
    check_down_tensor(sorted_expert_ids,
                      "sorted_expert_ids",
                      1,
                      AITER_DTYPE_i32,
                      "int32");
    check_down_tensor(
        num_valid_ids, "num_valid_ids", 1, AITER_DTYPE_i32, "int32");
    check_down_tensor(
        route_to_token, "route_to_token", 1, AITER_DTYPE_i32, "int32");
    check_down_tensor(expert_padded_offsets,
                      "expert_padded_offsets",
                      1,
                      AITER_DTYPE_i32,
                      "int32");
    check_down_tensor(d_scores_workspace,
                      "d_scores_workspace",
                      2,
                      AITER_DTYPE_fp32,
                      "float32");
    check_down_tensor(d_z, "d_z", 2, AITER_DTYPE_bf16, "bfloat16");
    check_down_tensor(a_scaled, "a_scaled", 2, AITER_DTYPE_bf16, "bfloat16");
    check_down_tensor(d_scores, "d_scores", 1, AITER_DTYPE_fp32, "float32");
    check_down_tensor(d_w2, "d_w2", 3, AITER_DTYPE_bf16, "bfloat16");
    check_down_tensor(d_b2, "d_b2", 2, AITER_DTYPE_bf16, "bfloat16");
    for(const auto* item : {&z,
                            &w2,
                            &scores,
                            &b2,
                            &sorted_route_ids,
                            &sorted_expert_ids,
                            &num_valid_ids,
                            &route_to_token,
                            &expert_padded_offsets,
                            &d_scores_workspace,
                            &d_z,
                            &a_scaled,
                            &d_scores,
                            &d_w2,
                            &d_b2})
        AITER_CHECK(item->device_id == d_out.device_id,
                    "varlen down tensors must share a GPU device");

    const int token_num = static_cast<int>(d_out.size(0));
    const int model_dim = static_cast<int>(d_out.size(1));
    const int num_experts = static_cast<int>(w2.size(0));
    const int inter_dim = static_cast<int>(w2.size(2));
    const int route_count = static_cast<int>(scores.numel());
    const int sorted_capacity = static_cast<int>(sorted_route_ids.numel());
    const int dscore_parts = (inter_dim + 127) / 128;
    AITER_CHECK(compute_dz || compute_dw2 || compute_dscore || compute_db2,
                "varlen down: at least one gradient must be requested");
    AITER_CHECK(!compute_db2 || has_bias,
                "varlen down: db2 requires a forward bias tensor");
    AITER_CHECK(token_num > 0 && model_dim > 0 && num_experts > 0 &&
                    inter_dim > 0,
                "varlen down dimensions must be positive");
    AITER_CHECK(w2.size(1) == model_dim,
                "varlen down: w2 must have shape [E,D,I]");
    AITER_CHECK(route_to_token.numel() == route_count &&
                    d_scores.numel() == route_count,
                "varlen down: scores/route_to_token/d_scores must have [R]");
    AITER_CHECK(z.size(0) == sorted_capacity && z.size(1) == 2 * inter_dim &&
                    d_z.size(0) == sorted_capacity &&
                    d_z.size(1) == 2 * inter_dim &&
                    a_scaled.size(0) == sorted_capacity &&
                    a_scaled.size(1) == inter_dim,
                "varlen down: sorted tensors have inconsistent shapes");
    AITER_CHECK(d_w2.size(0) == num_experts &&
                    d_w2.size(1) == model_dim &&
                    d_w2.size(2) == inter_dim,
                "varlen down: d_w2 must have shape [E,D,I]");
    AITER_CHECK(expert_padded_offsets.numel() == num_experts + 1 &&
                    num_valid_ids.numel() >= 1,
                "varlen down: invalid expert offsets/num_valid_ids");
    AITER_CHECK(block_m == 32,
                "varlen down: first kernel requires block_m=32");
    AITER_CHECK(sorted_expert_ids.size(0) * block_m >= sorted_capacity,
                "varlen down: sorted_expert_ids does not cover capacity");
    AITER_CHECK(d_scores_workspace.size(0) == route_count &&
                    d_scores_workspace.size(1) == dscore_parts,
                "varlen down: workspace must have shape [R,ceil(I/128)]");
    if(has_bias)
        AITER_CHECK(b2.size(0) == num_experts && b2.size(1) == model_dim &&
                        d_b2.size(0) == num_experts &&
                        d_b2.size(1) == model_dim,
                    "varlen down: b2/d_b2 must have shape [E,D]");

    opus_moe_backward::RouteMetadata route{};
    route.sorted_token_ids =
        reinterpret_cast<const int32_t*>(sorted_route_ids.data_ptr());
    route.sorted_expert_ids =
        reinterpret_cast<const int32_t*>(sorted_expert_ids.data_ptr());
    route.num_valid_ids =
        reinterpret_cast<const int32_t*>(num_valid_ids.data_ptr());
    route.route_to_token =
        reinterpret_cast<const int32_t*>(route_to_token.data_ptr());
    route.expert_offsets =
        reinterpret_cast<const int32_t*>(expert_padded_offsets.data_ptr());
    route.token_num = token_num;
    route.route_count = route_count;
    route.num_experts = num_experts;
    route.sort_block_m = block_m;
    route.sorted_capacity = sorted_capacity;
    route.sorted_block_capacity = static_cast<int>(sorted_expert_ids.numel());

    opus_moe_backward::DownBwdKargs down{};
    down.d_out = reinterpret_cast<const hip_bfloat16*>(d_out.data_ptr());
    down.z = reinterpret_cast<const hip_bfloat16*>(z.data_ptr());
    down.w2 = reinterpret_cast<const hip_bfloat16*>(w2.data_ptr());
    down.scores = reinterpret_cast<const float*>(scores.data_ptr());
    down.route = route;
    down.d_z = reinterpret_cast<hip_bfloat16*>(d_z.data_ptr());
    down.a_scaled = reinterpret_cast<hip_bfloat16*>(a_scaled.data_ptr());
    down.d_scores = reinterpret_cast<float*>(d_scores.data_ptr());
    down.d_scores_workspace = dscore_parts > 1
                                  ? reinterpret_cast<float*>(
                                        d_scores_workspace.data_ptr())
                                  : nullptr;
    down.model_dim = model_dim;
    down.inter_dim = inter_dim;
    down.d_scores_parts = dscore_parts;
    down.stride_do_t = d_out.stride(0);
    down.stride_z_r = z.stride(0);
    down.stride_w2_e = w2.stride(0);
    down.stride_w2_d = w2.stride(1);
    down.stride_score_t = 1;
    down.stride_dz_r = d_z.stride(0);
    down.stride_a_scaled_r = a_scaled.stride(0);
    down.stride_ds_t = 1;
    down.stride_ds_workspace_r = d_scores_workspace.stride(0);

    opus_moe_backward::Dw2Kargs weight{};
    weight.d_out = down.d_out;
    weight.a_scaled = down.a_scaled;
    weight.route = route;
    weight.d_w2 = reinterpret_cast<hip_bfloat16*>(d_w2.data_ptr());
    weight.model_dim = model_dim;
    weight.inter_dim = inter_dim;
    weight.split_k = 1;
    weight.stride_do_t = d_out.stride(0);
    weight.stride_a_scaled_r = a_scaled.stride(0);
    weight.stride_dw2_e = d_w2.stride(0);
    weight.stride_dw2_d = d_w2.stride(1);

    HipDeviceGuard guard(d_out.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    opus_moe_backward::detail::check_gfx950_or_fail();
    constexpr int varlen_kid = 100;
    const bool compute_down = compute_dz || compute_dw2 || compute_dscore;
    if(compute_down)
        opus_moe_backward::detail::invoke(
            opus_moe_backward::gfx950::dispatch_down_bwd(
                down_kernel_id == opus_moe_backward::kKernelAuto
                    ? varlen_kid
                    : down_kernel_id),
            down,
            stream,
            opus_moe_backward::Family::DownBwd);
    if(compute_dw2)
        opus_moe_backward::detail::invoke(
            opus_moe_backward::gfx950::dispatch_dw2(
                dw2_kernel_id == opus_moe_backward::kKernelAuto
                    ? varlen_kid
                    : dw2_kernel_id),
            weight,
            stream,
            opus_moe_backward::Family::Dw2);

    if(has_bias && (compute_dscore || compute_db2))
    {
        opus_moe_backward::BiasBwdKargs bias{};
        bias.d_out = down.d_out;
        bias.scores = down.scores;
        bias.b2 = reinterpret_cast<const hip_bfloat16*>(b2.data_ptr());
        bias.route = route;
        bias.d_scores = down.d_scores;
        bias.d_b2 = reinterpret_cast<hip_bfloat16*>(d_b2.data_ptr());
        bias.model_dim = model_dim;
        bias.inter_dim = inter_dim;
        bias.compute_dscore = compute_dscore;
        bias.compute_db2 = compute_db2;
        bias.stride_do_t = d_out.stride(0);
        bias.stride_score_t = 1;
        bias.stride_b2_e = b2.stride(0);
        bias.stride_ds_t = 1;
        bias.stride_db2_e = d_b2.stride(0);
        opus_moe_backward::detail::invoke(
            opus_moe_backward::gfx950::dispatch_bias_bwd(
                bias_kernel_id == opus_moe_backward::kKernelAuto
                    ? varlen_kid
                    : bias_kernel_id),
            bias,
            stream,
            opus_moe_backward::Family::BiasBwd);
    }
    HIP_CALL_LAUNCH(hipGetLastError());
}

void opus_moe_down_bwd(aiter_tensor_t& d_out,
                       aiter_tensor_t& z,
                       aiter_tensor_t& w2,
                       aiter_tensor_t& scores,
                       aiter_tensor_t& sorted_token_ids,
                       aiter_tensor_t& sorted_expert_ids,
                       aiter_tensor_t& num_valid_ids,
                       aiter_tensor_t& d_scores_workspace,
                       aiter_tensor_t& d_z,
                       aiter_tensor_t& a_scaled,
                       aiter_tensor_t& d_scores,
                       int block_m,
                       int kernel_id)
{
    check_down_tensor(d_out, "d_out", 2, AITER_DTYPE_bf16, "bfloat16");
    check_down_tensor(z, "z", 2, AITER_DTYPE_bf16, "bfloat16");
    check_down_tensor(w2, "w2", 3, AITER_DTYPE_bf16, "bfloat16");
    check_down_tensor(scores, "scores", 2, AITER_DTYPE_fp32, "float32");
    check_down_tensor(
        sorted_token_ids, "sorted_token_ids", 1, AITER_DTYPE_i32, "int32");
    check_down_tensor(
        sorted_expert_ids, "sorted_expert_ids", 1, AITER_DTYPE_i32, "int32");
    check_down_tensor(num_valid_ids, "num_valid_ids", 1, AITER_DTYPE_i32, "int32");
    check_down_tensor(d_scores_workspace,
                      "d_scores_workspace",
                      2,
                      AITER_DTYPE_fp32,
                      "float32");
    check_down_tensor(d_z, "d_z", 2, AITER_DTYPE_bf16, "bfloat16");
    check_down_tensor(a_scaled, "a_scaled", 2, AITER_DTYPE_bf16, "bfloat16");
    check_down_tensor(d_scores, "d_scores", 2, AITER_DTYPE_fp32, "float32");

    check_down_same_device(d_out, z, "z");
    check_down_same_device(d_out, w2, "w2");
    check_down_same_device(d_out, scores, "scores");
    check_down_same_device(d_out, sorted_token_ids, "sorted_token_ids");
    check_down_same_device(d_out, sorted_expert_ids, "sorted_expert_ids");
    check_down_same_device(d_out, num_valid_ids, "num_valid_ids");
    check_down_same_device(d_out, d_scores_workspace, "d_scores_workspace");
    check_down_same_device(d_out, d_z, "d_z");
    check_down_same_device(d_out, a_scaled, "a_scaled");
    check_down_same_device(d_out, d_scores, "d_scores");

    const int token_num = static_cast<int>(d_out.size(0));
    const int model_dim = static_cast<int>(d_out.size(1));
    const int num_experts = static_cast<int>(w2.size(0));
    const int inter_dim = static_cast<int>(w2.size(2));
    const int topk = static_cast<int>(scores.size(1));
    const int sorted_capacity = static_cast<int>(sorted_token_ids.size(0));
    constexpr int kFirstDownBlockN = 128;
    const int d_scores_parts =
        (inter_dim + kFirstDownBlockN - 1) / kFirstDownBlockN;

    AITER_CHECK(token_num > 0 && model_dim > 0,
                "d_out must have positive [T,D] dimensions");
    AITER_CHECK(num_experts > 0 && inter_dim > 0,
                "w2 must have positive [E,D,I] dimensions");
    AITER_CHECK(w2.size(1) == model_dim,
                "w2 must have shape [E,D,I] matching d_out D");
    AITER_CHECK(scores.size(0) == token_num && topk > 0,
                "scores must have shape [T,K] matching d_out T");
    AITER_CHECK(z.size(0) == sorted_capacity && z.size(1) == 2 * inter_dim,
                "z must have shape [sorted_capacity,2I]");
    AITER_CHECK(d_z.size(0) == z.size(0) && d_z.size(1) == z.size(1),
                "d_z must have the same shape as z");
    AITER_CHECK(a_scaled.size(0) == sorted_capacity &&
                    a_scaled.size(1) == inter_dim,
                "a_scaled must have shape [sorted_capacity,I]");
    AITER_CHECK(d_scores.size(0) == scores.size(0) &&
                    d_scores.size(1) == scores.size(1),
                "d_scores must have the same shape as scores");
    AITER_CHECK(num_valid_ids.numel() >= 1,
                "num_valid_ids must contain at least one element");
    AITER_CHECK(block_m == 32,
                "the first down_bwd kernel requires block_m=32, got ",
                block_m);
    AITER_CHECK(sorted_expert_ids.size(0) * block_m >= sorted_capacity,
                "sorted_expert_ids does not cover sorted_token_ids capacity");
    const int64_t route_count = static_cast<int64_t>(token_num) * topk;
    if(d_scores_parts > 1)
    {
        AITER_CHECK(d_scores_workspace.size(0) == route_count &&
                        d_scores_workspace.size(1) == d_scores_parts,
                    "d_scores_workspace must have shape [T*K,ceil(I/128)]");
    }

    opus_moe_backward::DownBwdKargs kargs{};
    kargs.d_out = reinterpret_cast<const hip_bfloat16*>(d_out.data_ptr());
    kargs.z = reinterpret_cast<const hip_bfloat16*>(z.data_ptr());
    kargs.w2 = reinterpret_cast<const hip_bfloat16*>(w2.data_ptr());
    kargs.scores = reinterpret_cast<const float*>(scores.data_ptr());
    kargs.route.sorted_token_ids =
        reinterpret_cast<const int32_t*>(sorted_token_ids.data_ptr());
    kargs.route.sorted_expert_ids =
        reinterpret_cast<const int32_t*>(sorted_expert_ids.data_ptr());
    kargs.route.num_valid_ids =
        reinterpret_cast<const int32_t*>(num_valid_ids.data_ptr());
    kargs.route.token_num = token_num;
    kargs.route.topk = topk;
    kargs.route.num_experts = num_experts;
    kargs.route.sort_block_m = block_m;
    kargs.route.sorted_capacity = sorted_capacity;
    kargs.route.sorted_block_capacity =
        static_cast<int>(sorted_expert_ids.size(0));
    kargs.d_z = reinterpret_cast<hip_bfloat16*>(d_z.data_ptr());
    kargs.a_scaled = reinterpret_cast<hip_bfloat16*>(a_scaled.data_ptr());
    kargs.d_scores = reinterpret_cast<float*>(d_scores.data_ptr());
    kargs.d_scores_workspace = d_scores_parts > 1
                                         ? reinterpret_cast<float*>(
                                               d_scores_workspace.data_ptr())
                                         : nullptr;
    kargs.model_dim = model_dim;
    kargs.inter_dim = inter_dim;
    kargs.d_scores_parts = d_scores_parts;
    kargs.stride_do_t = d_out.stride(0);
    kargs.stride_z_r = z.stride(0);
    kargs.stride_w2_e = w2.stride(0);
    kargs.stride_w2_d = w2.stride(1);
    kargs.stride_score_t = scores.stride(0);
    kargs.stride_dz_r = d_z.stride(0);
    kargs.stride_a_scaled_r = a_scaled.stride(0);
    kargs.stride_ds_t = d_scores.stride(0);
    kargs.stride_ds_workspace_r =
        d_scores_parts > 1 ? d_scores_workspace.stride(0) : 1;

    HipDeviceGuard guard(d_out.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    opus_moe_backward::launch_down_bwd_bf16(kargs, kernel_id, stream);
    HIP_CALL_LAUNCH(hipGetLastError());
}

void opus_moe_varlen_up_bwd(aiter_tensor_t& d_z,
                            aiter_tensor_t& x,
                            aiter_tensor_t& w1,
                            aiter_tensor_t& sorted_route_ids,
                            aiter_tensor_t& sorted_expert_ids,
                            aiter_tensor_t& num_valid_ids,
                            aiter_tensor_t& route_to_token,
                            aiter_tensor_t& token_route_offsets,
                            aiter_tensor_t& expert_padded_offsets,
                            aiter_tensor_t& d_x_route,
                            aiter_tensor_t& d_x,
                            aiter_tensor_t& d_w1,
                            aiter_tensor_t& d_b1,
                            int block_m,
                            bool compute_dx,
                            bool compute_dw1,
                            bool compute_db1,
                            int route_dx_kernel_id,
                            int route_reduce_kernel_id,
                            int dw1_kernel_id,
                            int bias_kernel_id)
{
    check_down_tensor(d_z, "d_z", 2, AITER_DTYPE_bf16, "bfloat16");
    check_down_tensor(x, "x", 2, AITER_DTYPE_bf16, "bfloat16");
    check_down_tensor(w1, "w1", 3, AITER_DTYPE_bf16, "bfloat16");
    check_down_tensor(
        sorted_route_ids, "sorted_route_ids", 1, AITER_DTYPE_i32, "int32");
    check_down_tensor(sorted_expert_ids,
                      "sorted_expert_ids",
                      1,
                      AITER_DTYPE_i32,
                      "int32");
    check_down_tensor(
        num_valid_ids, "num_valid_ids", 1, AITER_DTYPE_i32, "int32");
    check_down_tensor(
        route_to_token, "route_to_token", 1, AITER_DTYPE_i32, "int32");
    check_down_tensor(token_route_offsets,
                      "token_route_offsets",
                      1,
                      AITER_DTYPE_i32,
                      "int32");
    check_down_tensor(expert_padded_offsets,
                      "expert_padded_offsets",
                      1,
                      AITER_DTYPE_i32,
                      "int32");
    check_down_tensor(
        d_x_route, "d_x_route", 2, AITER_DTYPE_bf16, "bfloat16");
    check_down_tensor(d_x, "d_x", 2, AITER_DTYPE_bf16, "bfloat16");
    check_down_tensor(d_w1, "d_w1", 3, AITER_DTYPE_bf16, "bfloat16");
    check_down_tensor(d_b1, "d_b1", 2, AITER_DTYPE_bf16, "bfloat16");
    for(const auto* item : {&x,
                            &w1,
                            &sorted_route_ids,
                            &sorted_expert_ids,
                            &num_valid_ids,
                            &route_to_token,
                            &token_route_offsets,
                            &expert_padded_offsets,
                            &d_x_route,
                            &d_x,
                            &d_w1,
                            &d_b1})
        AITER_CHECK(item->device_id == d_z.device_id,
                    "varlen up tensors must share a GPU device");

    const int token_num = static_cast<int>(x.size(0));
    const int model_dim = static_cast<int>(x.size(1));
    const int num_experts = static_cast<int>(w1.size(0));
    const int gate_up_dim = static_cast<int>(w1.size(1));
    const int inter_dim = gate_up_dim / 2;
    const int route_count = static_cast<int>(route_to_token.numel());
    const int sorted_capacity = static_cast<int>(sorted_route_ids.numel());
    AITER_CHECK(compute_dx || compute_dw1 || compute_db1,
                "varlen up: at least one gradient must be requested");
    AITER_CHECK(token_num > 0 && model_dim > 0 && num_experts > 0 &&
                    gate_up_dim > 0 && gate_up_dim % 2 == 0,
                "varlen up dimensions must be positive and W1 must contain 2I");
    AITER_CHECK(w1.size(2) == model_dim,
                "varlen up: w1 must have shape [E,2I,D]");
    AITER_CHECK(d_z.size(0) == sorted_capacity &&
                    d_z.size(1) == gate_up_dim,
                "varlen up: d_z must have shape [sorted_capacity,2I]");
    AITER_CHECK(token_route_offsets.numel() == token_num + 1,
                "varlen up: token_route_offsets must have T+1 entries");
    AITER_CHECK(expert_padded_offsets.numel() == num_experts + 1 &&
                    num_valid_ids.numel() >= 1,
                "varlen up: invalid expert offsets/num_valid_ids");
    AITER_CHECK(d_x_route.size(0) == route_count &&
                    d_x_route.size(1) == model_dim &&
                    d_x.size(0) == token_num && d_x.size(1) == model_dim,
                "varlen up: d_x_route/d_x have inconsistent shapes");
    AITER_CHECK(d_w1.size(0) == num_experts &&
                    d_w1.size(1) == gate_up_dim &&
                    d_w1.size(2) == model_dim,
                "varlen up: d_w1 must have shape [E,2I,D]");
    if(compute_db1)
        AITER_CHECK(d_b1.size(0) == num_experts &&
                        d_b1.size(1) == gate_up_dim,
                    "varlen up: d_b1 must have shape [E,2I]");
    AITER_CHECK(block_m == 32,
                "varlen up: first kernel requires block_m=32");
    AITER_CHECK(sorted_expert_ids.size(0) * block_m >= sorted_capacity,
                "varlen up: sorted_expert_ids does not cover capacity");

    opus_moe_backward::RouteMetadata route{};
    route.sorted_token_ids =
        reinterpret_cast<const int32_t*>(sorted_route_ids.data_ptr());
    route.sorted_expert_ids =
        reinterpret_cast<const int32_t*>(sorted_expert_ids.data_ptr());
    route.num_valid_ids =
        reinterpret_cast<const int32_t*>(num_valid_ids.data_ptr());
    route.route_to_token =
        reinterpret_cast<const int32_t*>(route_to_token.data_ptr());
    route.expert_offsets =
        reinterpret_cast<const int32_t*>(expert_padded_offsets.data_ptr());
    route.token_num = token_num;
    route.route_count = route_count;
    route.num_experts = num_experts;
    route.sort_block_m = block_m;
    route.sorted_capacity = sorted_capacity;
    route.sorted_block_capacity = static_cast<int>(sorted_expert_ids.numel());

    opus_moe_backward::RouteDxKargs route_dx{};
    route_dx.d_z = reinterpret_cast<const hip_bfloat16*>(d_z.data_ptr());
    route_dx.w1 = reinterpret_cast<const hip_bfloat16*>(w1.data_ptr());
    route_dx.route = route;
    route_dx.d_x_route =
        reinterpret_cast<hip_bfloat16*>(d_x_route.data_ptr());
    route_dx.model_dim = model_dim;
    route_dx.inter_dim = inter_dim;
    route_dx.stride_dz_r = d_z.stride(0);
    route_dx.stride_w1_e = w1.stride(0);
    route_dx.stride_w1_i = w1.stride(1);
    route_dx.stride_dx_route_r = d_x_route.stride(0);

    opus_moe_backward::RouteReduceKargs reduce{};
    reduce.d_x_route =
        reinterpret_cast<const hip_bfloat16*>(d_x_route.data_ptr());
    reduce.route = route;
    reduce.route.token_route_offsets =
        reinterpret_cast<const int32_t*>(token_route_offsets.data_ptr());
    reduce.d_x = reinterpret_cast<hip_bfloat16*>(d_x.data_ptr());
    reduce.model_dim = model_dim;
    reduce.stride_dx_route_r = d_x_route.stride(0);
    reduce.stride_dx_t = d_x.stride(0);

    opus_moe_backward::Dw1Kargs weight{};
    weight.x = reinterpret_cast<const hip_bfloat16*>(x.data_ptr());
    weight.d_z = route_dx.d_z;
    weight.route = route;
    weight.d_w1 = reinterpret_cast<hip_bfloat16*>(d_w1.data_ptr());
    weight.model_dim = model_dim;
    weight.inter_dim = inter_dim;
    weight.split_k = 1;
    weight.stride_x_t = x.stride(0);
    weight.stride_dz_r = d_z.stride(0);
    weight.stride_dw1_e = d_w1.stride(0);
    weight.stride_dw1_i = d_w1.stride(1);

    HipDeviceGuard guard(d_z.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    opus_moe_backward::detail::check_gfx950_or_fail();
    constexpr int varlen_kid = 100;
    if(compute_dx)
    {
        opus_moe_backward::detail::invoke(
            opus_moe_backward::gfx950::dispatch_route_dx(
                route_dx_kernel_id == opus_moe_backward::kKernelAuto
                    ? varlen_kid
                    : route_dx_kernel_id),
            route_dx,
            stream,
            opus_moe_backward::Family::RouteDx);
        opus_moe_backward::detail::invoke(
            opus_moe_backward::gfx950::dispatch_route_reduce(
                route_reduce_kernel_id == opus_moe_backward::kKernelAuto
                    ? varlen_kid
                    : route_reduce_kernel_id),
            reduce,
            stream,
            opus_moe_backward::Family::RouteReduce);
    }
    if(compute_dw1)
        opus_moe_backward::detail::invoke(
            opus_moe_backward::gfx950::dispatch_dw1(
                dw1_kernel_id == opus_moe_backward::kKernelAuto
                    ? varlen_kid
                    : dw1_kernel_id),
            weight,
            stream,
            opus_moe_backward::Family::Dw1);

    if(compute_db1)
    {
        opus_moe_backward::BiasBwdKargs bias{};
        bias.d_z = route_dx.d_z;
        bias.route = route;
        bias.d_b1 = reinterpret_cast<hip_bfloat16*>(d_b1.data_ptr());
        bias.model_dim = model_dim;
        bias.inter_dim = inter_dim;
        bias.compute_db1 = true;
        bias.stride_dz_r = d_z.stride(0);
        bias.stride_db1_e = d_b1.stride(0);
        opus_moe_backward::detail::invoke(
            opus_moe_backward::gfx950::dispatch_bias_bwd(
                bias_kernel_id == opus_moe_backward::kKernelAuto
                    ? varlen_kid
                    : bias_kernel_id),
            bias,
            stream,
            opus_moe_backward::Family::BiasBwd);
    }
    HIP_CALL_LAUNCH(hipGetLastError());
}

void opus_moe_route_bwd(aiter_tensor_t& d_z,
                        aiter_tensor_t& w1,
                        aiter_tensor_t& sorted_token_ids,
                        aiter_tensor_t& sorted_expert_ids,
                        aiter_tensor_t& num_valid_ids,
                        aiter_tensor_t& expert_padded_offsets,
                        aiter_tensor_t& reverse_sorted,
                        aiter_tensor_t& d_x_route,
                        aiter_tensor_t& d_x,
                        int block_m,
                        int route_dx_kernel_id,
                        int route_reduce_kernel_id)
{
    check_down_tensor(d_z, "d_z", 2, AITER_DTYPE_bf16, "bfloat16");
    check_down_tensor(w1, "w1", 3, AITER_DTYPE_bf16, "bfloat16");
    check_down_tensor(
        sorted_token_ids, "sorted_token_ids", 1, AITER_DTYPE_i32, "int32");
    check_down_tensor(
        sorted_expert_ids, "sorted_expert_ids", 1, AITER_DTYPE_i32, "int32");
    check_down_tensor(num_valid_ids, "num_valid_ids", 1, AITER_DTYPE_i32, "int32");
    check_down_tensor(expert_padded_offsets,
                      "expert_padded_offsets",
                      1,
                      AITER_DTYPE_i32,
                      "int32");
    check_down_tensor(
        reverse_sorted, "reverse_sorted", 1, AITER_DTYPE_i32, "int32");
    check_down_tensor(
        d_x_route, "d_x_route", 2, AITER_DTYPE_bf16, "bfloat16");
    check_down_tensor(d_x, "d_x", 2, AITER_DTYPE_bf16, "bfloat16");

    check_down_same_device(d_z, w1, "w1");
    check_down_same_device(d_z, sorted_token_ids, "sorted_token_ids");
    check_down_same_device(d_z, sorted_expert_ids, "sorted_expert_ids");
    check_down_same_device(d_z, num_valid_ids, "num_valid_ids");
    check_down_same_device(d_z, expert_padded_offsets, "expert_padded_offsets");
    check_down_same_device(d_z, reverse_sorted, "reverse_sorted");
    check_down_same_device(d_z, d_x_route, "d_x_route");
    check_down_same_device(d_z, d_x, "d_x");

    const int sorted_capacity = static_cast<int>(sorted_token_ids.size(0));
    const int num_experts = static_cast<int>(w1.size(0));
    const int gate_up_dim = static_cast<int>(w1.size(1));
    const int model_dim = static_cast<int>(w1.size(2));
    const int token_num = static_cast<int>(d_x.size(0));
    AITER_CHECK(gate_up_dim > 0 && gate_up_dim % 2 == 0,
                "w1 gate/up dimension must be positive and even");
    const int inter_dim = gate_up_dim / 2;
    AITER_CHECK(token_num > 0 && reverse_sorted.size(0) % token_num == 0,
                "reverse_sorted must contain T*K entries");
    const int topk = static_cast<int>(reverse_sorted.size(0) / token_num);
    AITER_CHECK(topk == 1 || topk == 2 || topk == 4 || topk == 8,
                "the first route reduce supports topk in {1,2,4,8}");
    AITER_CHECK(d_z.size(0) == sorted_capacity &&
                    d_z.size(1) == gate_up_dim,
                "d_z must have shape [sorted_capacity,2I]");
    AITER_CHECK(d_x_route.size(0) == sorted_capacity &&
                    d_x_route.size(1) == model_dim,
                "d_x_route must have shape [sorted_capacity,D]");
    AITER_CHECK(d_x.size(1) == model_dim,
                "d_x must have shape [T,D]");
    AITER_CHECK(num_experts > 0, "w1 must contain at least one expert");
    AITER_CHECK(expert_padded_offsets.numel() == num_experts + 1,
                "expert_padded_offsets must have shape [E+1]");
    AITER_CHECK(num_valid_ids.numel() >= 1,
                "num_valid_ids must contain at least one element");
    AITER_CHECK(block_m == 32,
                "the first route_dx kernel requires block_m=32");
    AITER_CHECK(sorted_expert_ids.size(0) * block_m >= sorted_capacity,
                "sorted_expert_ids does not cover sorted capacity");

    opus_moe_backward::RouteMetadata route{};
    route.sorted_token_ids =
        reinterpret_cast<const int32_t*>(sorted_token_ids.data_ptr());
    route.sorted_expert_ids =
        reinterpret_cast<const int32_t*>(sorted_expert_ids.data_ptr());
    route.num_valid_ids =
        reinterpret_cast<const int32_t*>(num_valid_ids.data_ptr());
    route.expert_offsets = reinterpret_cast<const int32_t*>(
        expert_padded_offsets.data_ptr());
    route.reverse_sorted =
        reinterpret_cast<const int32_t*>(reverse_sorted.data_ptr());
    route.token_num = token_num;
    route.topk = topk;
    route.num_experts = num_experts;
    route.sort_block_m = block_m;
    route.sorted_capacity = sorted_capacity;
    route.sorted_block_capacity =
        static_cast<int>(sorted_expert_ids.size(0));

    opus_moe_backward::RouteDxKargs route_dx_args{};
    route_dx_args.d_z =
        reinterpret_cast<const hip_bfloat16*>(d_z.data_ptr());
    route_dx_args.w1 =
        reinterpret_cast<const hip_bfloat16*>(w1.data_ptr());
    route_dx_args.route = route;
    route_dx_args.d_x_route =
        reinterpret_cast<hip_bfloat16*>(d_x_route.data_ptr());
    route_dx_args.model_dim = model_dim;
    route_dx_args.inter_dim = inter_dim;
    route_dx_args.stride_dz_r = d_z.stride(0);
    route_dx_args.stride_w1_e = w1.stride(0);
    route_dx_args.stride_w1_i = w1.stride(1);
    route_dx_args.stride_dx_route_r = d_x_route.stride(0);

    opus_moe_backward::RouteReduceKargs reduce_args{};
    reduce_args.d_x_route =
        reinterpret_cast<const hip_bfloat16*>(d_x_route.data_ptr());
    reduce_args.route = route;
    reduce_args.d_x = reinterpret_cast<hip_bfloat16*>(d_x.data_ptr());
    reduce_args.model_dim = model_dim;
    reduce_args.stride_dx_route_r = d_x_route.stride(0);
    reduce_args.stride_dx_t = d_x.stride(0);

    HipDeviceGuard guard(d_z.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    opus_moe_backward::launch_route_dx_bf16(
        route_dx_args, route_dx_kernel_id, stream);
    opus_moe_backward::launch_route_reduce_bf16(
        reduce_args, route_reduce_kernel_id, stream);
    HIP_CALL_LAUNCH(hipGetLastError());
}

void opus_moe_weight_bwd(aiter_tensor_t& x,
                         aiter_tensor_t& d_out,
                         aiter_tensor_t& d_z,
                         aiter_tensor_t& a_scaled,
                         aiter_tensor_t& sorted_token_ids,
                         aiter_tensor_t& num_valid_ids,
                         aiter_tensor_t& expert_padded_offsets,
                         aiter_tensor_t& d_w1,
                         aiter_tensor_t& d_w2,
                         int topk,
                         int block_m,
                         int dw1_kernel_id,
                         int dw2_kernel_id)
{
    check_down_tensor(x, "x", 2, AITER_DTYPE_bf16, "bfloat16");
    check_down_tensor(d_out, "d_out", 2, AITER_DTYPE_bf16, "bfloat16");
    check_down_tensor(d_z, "d_z", 2, AITER_DTYPE_bf16, "bfloat16");
    check_down_tensor(
        a_scaled, "a_scaled", 2, AITER_DTYPE_bf16, "bfloat16");
    check_down_tensor(
        sorted_token_ids, "sorted_token_ids", 1, AITER_DTYPE_i32, "int32");
    check_down_tensor(num_valid_ids, "num_valid_ids", 1, AITER_DTYPE_i32, "int32");
    check_down_tensor(expert_padded_offsets,
                      "expert_padded_offsets",
                      1,
                      AITER_DTYPE_i32,
                      "int32");
    check_down_tensor(d_w1, "d_w1", 3, AITER_DTYPE_bf16, "bfloat16");
    check_down_tensor(d_w2, "d_w2", 3, AITER_DTYPE_bf16, "bfloat16");

    check_down_same_device(x, d_out, "d_out");
    check_down_same_device(x, d_z, "d_z");
    check_down_same_device(x, a_scaled, "a_scaled");
    check_down_same_device(x, sorted_token_ids, "sorted_token_ids");
    check_down_same_device(x, num_valid_ids, "num_valid_ids");
    check_down_same_device(x, expert_padded_offsets, "expert_padded_offsets");
    check_down_same_device(x, d_w1, "d_w1");
    check_down_same_device(x, d_w2, "d_w2");

    const int token_num = static_cast<int>(x.size(0));
    const int model_dim = static_cast<int>(x.size(1));
    const int sorted_capacity = static_cast<int>(sorted_token_ids.size(0));
    const int gate_up_dim = static_cast<int>(d_z.size(1));
    AITER_CHECK(gate_up_dim > 0 && gate_up_dim % 2 == 0,
                "d_z gate/up dimension must be positive and even");
    const int inter_dim = gate_up_dim / 2;
    AITER_CHECK(expert_padded_offsets.numel() >= 2,
                "expert_padded_offsets must contain E+1 entries");
    const int num_experts =
        static_cast<int>(expert_padded_offsets.numel() - 1);

    AITER_CHECK(token_num > 0 && model_dim > 0,
                "x must have positive [T,D] dimensions");
    AITER_CHECK(d_out.size(0) == token_num && d_out.size(1) == model_dim,
                "d_out must have the same [T,D] shape as x");
    AITER_CHECK(d_z.size(0) == sorted_capacity,
                "d_z must have shape [sorted_capacity,2I]");
    AITER_CHECK(a_scaled.size(0) == sorted_capacity &&
                    a_scaled.size(1) == inter_dim,
                "a_scaled must have shape [sorted_capacity,I]");
    AITER_CHECK(d_w1.size(0) == num_experts &&
                    d_w1.size(1) == gate_up_dim &&
                    d_w1.size(2) == model_dim,
                "d_w1 must have shape [E,2I,D]");
    AITER_CHECK(d_w2.size(0) == num_experts &&
                    d_w2.size(1) == model_dim &&
                    d_w2.size(2) == inter_dim,
                "d_w2 must have shape [E,D,I]");
    AITER_CHECK(num_valid_ids.numel() >= 1,
                "num_valid_ids must contain at least one element");
    AITER_CHECK(topk > 0 && topk <= opus_moe_backward::kMaxPackedTopk,
                "topk must be in [1,256]");
    AITER_CHECK(block_m == 32,
                "the first K4/K5 kernels require block_m=32");

    opus_moe_backward::RouteMetadata route{};
    route.sorted_token_ids =
        reinterpret_cast<const int32_t*>(sorted_token_ids.data_ptr());
    route.num_valid_ids =
        reinterpret_cast<const int32_t*>(num_valid_ids.data_ptr());
    route.expert_offsets =
        reinterpret_cast<const int32_t*>(expert_padded_offsets.data_ptr());
    route.token_num = token_num;
    route.topk = topk;
    route.num_experts = num_experts;
    route.sort_block_m = block_m;
    route.sorted_capacity = sorted_capacity;
    route.sorted_block_capacity =
        (sorted_capacity + block_m - 1) / block_m;

    opus_moe_backward::Dw1Kargs dw1_args{};
    dw1_args.x = reinterpret_cast<const hip_bfloat16*>(x.data_ptr());
    dw1_args.d_z = reinterpret_cast<const hip_bfloat16*>(d_z.data_ptr());
    dw1_args.route = route;
    dw1_args.d_w1 = reinterpret_cast<hip_bfloat16*>(d_w1.data_ptr());
    dw1_args.model_dim = model_dim;
    dw1_args.inter_dim = inter_dim;
    dw1_args.split_k = 1;
    dw1_args.stride_x_t = x.stride(0);
    dw1_args.stride_dz_r = d_z.stride(0);
    dw1_args.stride_dw1_e = d_w1.stride(0);
    dw1_args.stride_dw1_i = d_w1.stride(1);

    opus_moe_backward::Dw2Kargs dw2_args{};
    dw2_args.d_out =
        reinterpret_cast<const hip_bfloat16*>(d_out.data_ptr());
    dw2_args.a_scaled =
        reinterpret_cast<const hip_bfloat16*>(a_scaled.data_ptr());
    dw2_args.route = route;
    dw2_args.d_w2 = reinterpret_cast<hip_bfloat16*>(d_w2.data_ptr());
    dw2_args.model_dim = model_dim;
    dw2_args.inter_dim = inter_dim;
    dw2_args.split_k = 1;
    dw2_args.stride_do_t = d_out.stride(0);
    dw2_args.stride_a_scaled_r = a_scaled.stride(0);
    dw2_args.stride_dw2_e = d_w2.stride(0);
    dw2_args.stride_dw2_d = d_w2.stride(1);

    HipDeviceGuard guard(x.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    opus_moe_backward::launch_dw1_bf16(dw1_args, dw1_kernel_id, stream);
    HIP_CALL_LAUNCH(hipGetLastError());
    opus_moe_backward::launch_dw2_bf16(dw2_args, dw2_kernel_id, stream);
    HIP_CALL_LAUNCH(hipGetLastError());
}

void opus_moe_dw1_bwd(aiter_tensor_t& x,
                      aiter_tensor_t& d_z,
                      aiter_tensor_t& sorted_token_ids,
                      aiter_tensor_t& num_valid_ids,
                      aiter_tensor_t& expert_padded_offsets,
                      aiter_tensor_t& d_w1,
                      int topk,
                      int block_m,
                      int kernel_id)
{
    check_down_tensor(x, "x", 2, AITER_DTYPE_bf16, "bfloat16");
    check_down_tensor(d_z, "d_z", 2, AITER_DTYPE_bf16, "bfloat16");
    check_down_tensor(
        sorted_token_ids, "sorted_token_ids", 1, AITER_DTYPE_i32, "int32");
    check_down_tensor(num_valid_ids, "num_valid_ids", 1, AITER_DTYPE_i32, "int32");
    check_down_tensor(expert_padded_offsets,
                      "expert_padded_offsets",
                      1,
                      AITER_DTYPE_i32,
                      "int32");
    check_down_tensor(d_w1, "d_w1", 3, AITER_DTYPE_bf16, "bfloat16");

    check_down_same_device(x, d_z, "d_z");
    check_down_same_device(x, sorted_token_ids, "sorted_token_ids");
    check_down_same_device(x, num_valid_ids, "num_valid_ids");
    check_down_same_device(x, expert_padded_offsets, "expert_padded_offsets");
    check_down_same_device(x, d_w1, "d_w1");

    const int token_num = static_cast<int>(x.size(0));
    const int model_dim = static_cast<int>(x.size(1));
    const int sorted_capacity = static_cast<int>(sorted_token_ids.size(0));
    const int gate_up_dim = static_cast<int>(d_z.size(1));
    AITER_CHECK(gate_up_dim > 0 && gate_up_dim % 2 == 0,
                "d_z gate/up dimension must be positive and even");
    const int inter_dim = gate_up_dim / 2;
    AITER_CHECK(expert_padded_offsets.numel() >= 2,
                "expert_padded_offsets must contain E+1 entries");
    const int num_experts =
        static_cast<int>(expert_padded_offsets.numel() - 1);
    AITER_CHECK(token_num > 0 && model_dim > 0,
                "x must have positive [T,D] dimensions");
    AITER_CHECK(d_z.size(0) == sorted_capacity,
                "d_z must have shape [sorted_capacity,2I]");
    AITER_CHECK(d_w1.size(0) == num_experts &&
                    d_w1.size(1) == gate_up_dim &&
                    d_w1.size(2) == model_dim,
                "d_w1 must have shape [E,2I,D]");
    AITER_CHECK(num_valid_ids.numel() >= 1,
                "num_valid_ids must contain at least one element");
    AITER_CHECK(topk > 0 && topk <= opus_moe_backward::kMaxPackedTopk,
                "topk must be in [1,256]");
    AITER_CHECK(block_m == 32,
                "the first K4 kernel requires block_m=32");

    opus_moe_backward::RouteMetadata route{};
    route.sorted_token_ids =
        reinterpret_cast<const int32_t*>(sorted_token_ids.data_ptr());
    route.num_valid_ids =
        reinterpret_cast<const int32_t*>(num_valid_ids.data_ptr());
    route.expert_offsets =
        reinterpret_cast<const int32_t*>(expert_padded_offsets.data_ptr());
    route.token_num = token_num;
    route.topk = topk;
    route.num_experts = num_experts;
    route.sort_block_m = block_m;
    route.sorted_capacity = sorted_capacity;
    route.sorted_block_capacity =
        (sorted_capacity + block_m - 1) / block_m;

    opus_moe_backward::Dw1Kargs args{};
    args.x = reinterpret_cast<const hip_bfloat16*>(x.data_ptr());
    args.d_z = reinterpret_cast<const hip_bfloat16*>(d_z.data_ptr());
    args.route = route;
    args.d_w1 = reinterpret_cast<hip_bfloat16*>(d_w1.data_ptr());
    args.model_dim = model_dim;
    args.inter_dim = inter_dim;
    args.split_k = 1;
    args.stride_x_t = x.stride(0);
    args.stride_dz_r = d_z.stride(0);
    args.stride_dw1_e = d_w1.stride(0);
    args.stride_dw1_i = d_w1.stride(1);

    HipDeviceGuard guard(x.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    opus_moe_backward::launch_dw1_bf16(args, kernel_id, stream);
    HIP_CALL_LAUNCH(hipGetLastError());
}

void opus_moe_dw2_bwd(aiter_tensor_t& d_out,
                      aiter_tensor_t& a_scaled,
                      aiter_tensor_t& sorted_token_ids,
                      aiter_tensor_t& num_valid_ids,
                      aiter_tensor_t& expert_padded_offsets,
                      aiter_tensor_t& d_w2,
                      int topk,
                      int block_m,
                      int kernel_id)
{
    check_down_tensor(d_out, "d_out", 2, AITER_DTYPE_bf16, "bfloat16");
    check_down_tensor(
        a_scaled, "a_scaled", 2, AITER_DTYPE_bf16, "bfloat16");
    check_down_tensor(
        sorted_token_ids, "sorted_token_ids", 1, AITER_DTYPE_i32, "int32");
    check_down_tensor(num_valid_ids, "num_valid_ids", 1, AITER_DTYPE_i32, "int32");
    check_down_tensor(expert_padded_offsets,
                      "expert_padded_offsets",
                      1,
                      AITER_DTYPE_i32,
                      "int32");
    check_down_tensor(d_w2, "d_w2", 3, AITER_DTYPE_bf16, "bfloat16");

    check_down_same_device(d_out, a_scaled, "a_scaled");
    check_down_same_device(d_out, sorted_token_ids, "sorted_token_ids");
    check_down_same_device(d_out, num_valid_ids, "num_valid_ids");
    check_down_same_device(
        d_out, expert_padded_offsets, "expert_padded_offsets");
    check_down_same_device(d_out, d_w2, "d_w2");

    const int token_num = static_cast<int>(d_out.size(0));
    const int model_dim = static_cast<int>(d_out.size(1));
    const int sorted_capacity = static_cast<int>(sorted_token_ids.size(0));
    const int inter_dim = static_cast<int>(a_scaled.size(1));
    AITER_CHECK(expert_padded_offsets.numel() >= 2,
                "expert_padded_offsets must contain E+1 entries");
    const int num_experts =
        static_cast<int>(expert_padded_offsets.numel() - 1);
    AITER_CHECK(token_num > 0 && model_dim > 0,
                "d_out must have positive [T,D] dimensions");
    AITER_CHECK(a_scaled.size(0) == sorted_capacity && inter_dim > 0,
                "a_scaled must have shape [sorted_capacity,I]");
    AITER_CHECK(d_w2.size(0) == num_experts &&
                    d_w2.size(1) == model_dim &&
                    d_w2.size(2) == inter_dim,
                "d_w2 must have shape [E,D,I]");
    AITER_CHECK(num_valid_ids.numel() >= 1,
                "num_valid_ids must contain at least one element");
    AITER_CHECK(topk > 0 && topk <= opus_moe_backward::kMaxPackedTopk,
                "topk must be in [1,256]");
    AITER_CHECK(block_m == 32,
                "the first K5 kernel requires block_m=32");

    opus_moe_backward::RouteMetadata route{};
    route.sorted_token_ids =
        reinterpret_cast<const int32_t*>(sorted_token_ids.data_ptr());
    route.num_valid_ids =
        reinterpret_cast<const int32_t*>(num_valid_ids.data_ptr());
    route.expert_offsets =
        reinterpret_cast<const int32_t*>(expert_padded_offsets.data_ptr());
    route.token_num = token_num;
    route.topk = topk;
    route.num_experts = num_experts;
    route.sort_block_m = block_m;
    route.sorted_capacity = sorted_capacity;
    route.sorted_block_capacity =
        (sorted_capacity + block_m - 1) / block_m;

    opus_moe_backward::Dw2Kargs args{};
    args.d_out =
        reinterpret_cast<const hip_bfloat16*>(d_out.data_ptr());
    args.a_scaled =
        reinterpret_cast<const hip_bfloat16*>(a_scaled.data_ptr());
    args.route = route;
    args.d_w2 = reinterpret_cast<hip_bfloat16*>(d_w2.data_ptr());
    args.model_dim = model_dim;
    args.inter_dim = inter_dim;
    args.split_k = 1;
    args.stride_do_t = d_out.stride(0);
    args.stride_a_scaled_r = a_scaled.stride(0);
    args.stride_dw2_e = d_w2.stride(0);
    args.stride_dw2_d = d_w2.stride(1);

    HipDeviceGuard guard(d_out.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    opus_moe_backward::launch_dw2_bf16(args, kernel_id, stream);
    HIP_CALL_LAUNCH(hipGetLastError());
}

template<bool Validate>
void opus_moe_full_bwd_impl(aiter_tensor_t& d_out,
                            aiter_tensor_t& x,
                            aiter_tensor_t& z,
                            aiter_tensor_t& w1,
                            aiter_tensor_t& w2,
                            aiter_tensor_t& scores,
                            aiter_tensor_t& sorted_token_ids,
                            aiter_tensor_t& sorted_expert_ids,
                            aiter_tensor_t& num_valid_ids,
                            aiter_tensor_t& reverse_sorted,
                            aiter_tensor_t& expert_padded_offsets,
                            aiter_tensor_t& d_scores_workspace,
                            aiter_tensor_t& d_z,
                            aiter_tensor_t& a_scaled,
                            aiter_tensor_t& d_scores,
                            aiter_tensor_t& d_x_route,
                            aiter_tensor_t& d_x,
                            aiter_tensor_t& d_w1,
                            aiter_tensor_t& d_w2,
                            int block_m,
                            int down_kernel_id,
                            int route_dx_kernel_id,
                            int route_reduce_kernel_id,
                            int dw1_kernel_id,
                            int dw2_kernel_id)
{
    if constexpr(Validate)
    {
        check_down_tensor(d_out, "d_out", 2, AITER_DTYPE_bf16, "bfloat16");
        check_down_tensor(x, "x", 2, AITER_DTYPE_bf16, "bfloat16");
        check_down_tensor(z, "z", 2, AITER_DTYPE_bf16, "bfloat16");
        check_down_tensor(w1, "w1", 3, AITER_DTYPE_bf16, "bfloat16");
        check_down_tensor(w2, "w2", 3, AITER_DTYPE_bf16, "bfloat16");
        check_down_tensor(scores, "scores", 2, AITER_DTYPE_fp32, "float32");
        check_down_tensor(
            sorted_token_ids, "sorted_token_ids", 1, AITER_DTYPE_i32, "int32");
        check_down_tensor(sorted_expert_ids,
                          "sorted_expert_ids",
                          1,
                          AITER_DTYPE_i32,
                          "int32");
        check_down_tensor(
            num_valid_ids, "num_valid_ids", 1, AITER_DTYPE_i32, "int32");
        check_down_tensor(
            reverse_sorted, "reverse_sorted", 1, AITER_DTYPE_i32, "int32");
        check_down_tensor(expert_padded_offsets,
                          "expert_padded_offsets",
                          1,
                          AITER_DTYPE_i32,
                          "int32");
        check_down_tensor(d_scores_workspace,
                          "d_scores_workspace",
                          2,
                          AITER_DTYPE_fp32,
                          "float32");
        check_down_tensor(d_z, "d_z", 2, AITER_DTYPE_bf16, "bfloat16");
        check_down_tensor(
            a_scaled, "a_scaled", 2, AITER_DTYPE_bf16, "bfloat16");
        check_down_tensor(
            d_scores, "d_scores", 2, AITER_DTYPE_fp32, "float32");
        check_down_tensor(
            d_x_route, "d_x_route", 2, AITER_DTYPE_bf16, "bfloat16");
        check_down_tensor(d_x, "d_x", 2, AITER_DTYPE_bf16, "bfloat16");
        check_down_tensor(d_w1, "d_w1", 3, AITER_DTYPE_bf16, "bfloat16");
        check_down_tensor(d_w2, "d_w2", 3, AITER_DTYPE_bf16, "bfloat16");

        check_down_same_device(d_out, x, "x");
        check_down_same_device(d_out, z, "z");
        check_down_same_device(d_out, w1, "w1");
        check_down_same_device(d_out, w2, "w2");
        check_down_same_device(d_out, scores, "scores");
        check_down_same_device(d_out, sorted_token_ids, "sorted_token_ids");
        check_down_same_device(d_out, sorted_expert_ids, "sorted_expert_ids");
        check_down_same_device(d_out, num_valid_ids, "num_valid_ids");
        check_down_same_device(d_out, reverse_sorted, "reverse_sorted");
        check_down_same_device(
            d_out, expert_padded_offsets, "expert_padded_offsets");
        check_down_same_device(
            d_out, d_scores_workspace, "d_scores_workspace");
        check_down_same_device(d_out, d_z, "d_z");
        check_down_same_device(d_out, a_scaled, "a_scaled");
        check_down_same_device(d_out, d_scores, "d_scores");
        check_down_same_device(d_out, d_x_route, "d_x_route");
        check_down_same_device(d_out, d_x, "d_x");
        check_down_same_device(d_out, d_w1, "d_w1");
        check_down_same_device(d_out, d_w2, "d_w2");
    }

    const int token_num = static_cast<int>(x.size(0));
    const int model_dim = static_cast<int>(x.size(1));
    const int num_experts = static_cast<int>(w1.size(0));
    const int gate_up_dim = static_cast<int>(w1.size(1));
    const int inter_dim = gate_up_dim / 2;
    const int topk = static_cast<int>(scores.size(1));
    const int sorted_capacity = static_cast<int>(sorted_token_ids.size(0));
    constexpr int kFirstDownBlockN = 128;
    const int d_scores_parts =
        (inter_dim + kFirstDownBlockN - 1) / kFirstDownBlockN;

    if constexpr(Validate)
    {
        AITER_CHECK(gate_up_dim > 0 && gate_up_dim % 2 == 0,
                    "w1 gate/up dimension must be positive and even");
        AITER_CHECK(token_num > 0 && model_dim > 0,
                    "x must have positive [T,D] dimensions");
        AITER_CHECK(d_out.size(0) == token_num &&
                        d_out.size(1) == model_dim,
                    "d_out must have the same [T,D] shape as x");
        AITER_CHECK(w1.size(2) == model_dim,
                    "w1 must have shape [E,2I,D]");
        AITER_CHECK(w2.size(0) == num_experts &&
                        w2.size(1) == model_dim &&
                        w2.size(2) == inter_dim,
                    "w2 must have shape [E,D,I] matching w1");
        AITER_CHECK(scores.size(0) == token_num && topk > 0 &&
                        topk <= opus_moe_backward::kMaxPackedTopk,
                    "scores must have shape [T,K] with K in [1,256]");
        AITER_CHECK(z.size(0) == sorted_capacity &&
                        z.size(1) == gate_up_dim,
                    "z must have shape [sorted_capacity,2I]");
        AITER_CHECK(d_z.size(0) == sorted_capacity &&
                        d_z.size(1) == gate_up_dim,
                    "d_z must have shape [sorted_capacity,2I]");
        AITER_CHECK(a_scaled.size(0) == sorted_capacity &&
                        a_scaled.size(1) == inter_dim,
                    "a_scaled must have shape [sorted_capacity,I]");
        AITER_CHECK(d_scores.size(0) == token_num &&
                        d_scores.size(1) == topk,
                    "d_scores must have shape [T,K]");
        AITER_CHECK(d_x_route.size(0) == sorted_capacity &&
                        d_x_route.size(1) == model_dim,
                    "d_x_route must have shape [sorted_capacity,D]");
        AITER_CHECK(d_x.size(0) == token_num && d_x.size(1) == model_dim,
                    "d_x must have shape [T,D]");
        AITER_CHECK(d_w1.size(0) == num_experts &&
                        d_w1.size(1) == gate_up_dim &&
                        d_w1.size(2) == model_dim,
                    "d_w1 must have shape [E,2I,D]");
        AITER_CHECK(d_w2.size(0) == num_experts &&
                        d_w2.size(1) == model_dim &&
                        d_w2.size(2) == inter_dim,
                    "d_w2 must have shape [E,D,I]");
        AITER_CHECK(reverse_sorted.numel() ==
                        static_cast<int64_t>(token_num) * topk,
                    "reverse_sorted must contain T*K entries");
        AITER_CHECK(expert_padded_offsets.numel() == num_experts + 1,
                    "expert_padded_offsets must contain E+1 entries");
        AITER_CHECK(num_valid_ids.numel() >= 1,
                    "num_valid_ids must contain at least one element");
        AITER_CHECK(block_m == 32,
                    "the first full backward path requires block_m=32");
        AITER_CHECK(sorted_expert_ids.size(0) * block_m >= sorted_capacity,
                    "sorted_expert_ids does not cover sorted capacity");
        if(d_scores_parts > 1)
        {
            AITER_CHECK(d_scores_workspace.size(0) ==
                            static_cast<int64_t>(token_num) * topk &&
                            d_scores_workspace.size(1) == d_scores_parts,
                        "d_scores_workspace must have shape "
                        "[T*K,ceil(I/128)]");
        }
    }

    opus_moe_backward::RouteMetadata route{};
    route.sorted_token_ids =
        reinterpret_cast<const int32_t*>(sorted_token_ids.data_ptr());
    route.sorted_expert_ids =
        reinterpret_cast<const int32_t*>(sorted_expert_ids.data_ptr());
    route.num_valid_ids =
        reinterpret_cast<const int32_t*>(num_valid_ids.data_ptr());
    route.reverse_sorted =
        reinterpret_cast<const int32_t*>(reverse_sorted.data_ptr());
    route.expert_offsets =
        reinterpret_cast<const int32_t*>(expert_padded_offsets.data_ptr());
    route.token_num = token_num;
    route.topk = topk;
    route.num_experts = num_experts;
    route.sort_block_m = block_m;
    route.sorted_capacity = sorted_capacity;
    route.sorted_block_capacity = static_cast<int>(sorted_expert_ids.size(0));

    opus_moe_backward::DownBwdKargs down_args{};
    down_args.d_out =
        reinterpret_cast<const hip_bfloat16*>(d_out.data_ptr());
    down_args.z = reinterpret_cast<const hip_bfloat16*>(z.data_ptr());
    down_args.w2 = reinterpret_cast<const hip_bfloat16*>(w2.data_ptr());
    down_args.scores = reinterpret_cast<const float*>(scores.data_ptr());
    down_args.route = route;
    down_args.d_z = reinterpret_cast<hip_bfloat16*>(d_z.data_ptr());
    down_args.a_scaled =
        reinterpret_cast<hip_bfloat16*>(a_scaled.data_ptr());
    down_args.d_scores = reinterpret_cast<float*>(d_scores.data_ptr());
    down_args.d_scores_workspace =
        d_scores_parts > 1
            ? reinterpret_cast<float*>(d_scores_workspace.data_ptr())
            : nullptr;
    down_args.model_dim = model_dim;
    down_args.inter_dim = inter_dim;
    down_args.d_scores_parts = d_scores_parts;
    down_args.stride_do_t = d_out.stride(0);
    down_args.stride_z_r = z.stride(0);
    down_args.stride_w2_e = w2.stride(0);
    down_args.stride_w2_d = w2.stride(1);
    down_args.stride_score_t = scores.stride(0);
    down_args.stride_dz_r = d_z.stride(0);
    down_args.stride_a_scaled_r = a_scaled.stride(0);
    down_args.stride_ds_t = d_scores.stride(0);
    down_args.stride_ds_workspace_r =
        d_scores_parts > 1 ? d_scores_workspace.stride(0) : 1;

    opus_moe_backward::RouteDxKargs route_dx_args{};
    route_dx_args.d_z =
        reinterpret_cast<const hip_bfloat16*>(d_z.data_ptr());
    route_dx_args.w1 =
        reinterpret_cast<const hip_bfloat16*>(w1.data_ptr());
    route_dx_args.route = route;
    route_dx_args.d_x_route =
        reinterpret_cast<hip_bfloat16*>(d_x_route.data_ptr());
    route_dx_args.model_dim = model_dim;
    route_dx_args.inter_dim = inter_dim;
    route_dx_args.stride_dz_r = d_z.stride(0);
    route_dx_args.stride_w1_e = w1.stride(0);
    route_dx_args.stride_w1_i = w1.stride(1);
    route_dx_args.stride_dx_route_r = d_x_route.stride(0);

    opus_moe_backward::RouteReduceKargs reduce_args{};
    reduce_args.d_x_route =
        reinterpret_cast<const hip_bfloat16*>(d_x_route.data_ptr());
    reduce_args.route = route;
    reduce_args.d_x = reinterpret_cast<hip_bfloat16*>(d_x.data_ptr());
    reduce_args.model_dim = model_dim;
    reduce_args.stride_dx_route_r = d_x_route.stride(0);
    reduce_args.stride_dx_t = d_x.stride(0);

    opus_moe_backward::Dw1Kargs dw1_args{};
    dw1_args.x = reinterpret_cast<const hip_bfloat16*>(x.data_ptr());
    dw1_args.d_z = reinterpret_cast<const hip_bfloat16*>(d_z.data_ptr());
    dw1_args.route = route;
    dw1_args.d_w1 = reinterpret_cast<hip_bfloat16*>(d_w1.data_ptr());
    dw1_args.model_dim = model_dim;
    dw1_args.inter_dim = inter_dim;
    dw1_args.split_k = 1;
    dw1_args.stride_x_t = x.stride(0);
    dw1_args.stride_dz_r = d_z.stride(0);
    dw1_args.stride_dw1_e = d_w1.stride(0);
    dw1_args.stride_dw1_i = d_w1.stride(1);

    opus_moe_backward::Dw2Kargs dw2_args{};
    dw2_args.d_out =
        reinterpret_cast<const hip_bfloat16*>(d_out.data_ptr());
    dw2_args.a_scaled =
        reinterpret_cast<const hip_bfloat16*>(a_scaled.data_ptr());
    dw2_args.route = route;
    dw2_args.d_w2 = reinterpret_cast<hip_bfloat16*>(d_w2.data_ptr());
    dw2_args.model_dim = model_dim;
    dw2_args.inter_dim = inter_dim;
    dw2_args.split_k = 1;
    dw2_args.stride_do_t = d_out.stride(0);
    dw2_args.stride_a_scaled_r = a_scaled.stride(0);
    dw2_args.stride_dw2_e = d_w2.stride(0);
    dw2_args.stride_dw2_d = d_w2.stride(1);

    const hipStream_t stream = aiter::getCurrentHIPStream();
    if constexpr(Validate)
    {
        HipDeviceGuard guard(d_out.device_id);
        opus_moe_backward::detail::launch_fixed_pipeline(
            down_args,
            route_dx_args,
            reduce_args,
            dw1_args,
            dw2_args,
            down_kernel_id,
            route_dx_kernel_id,
            route_reduce_kernel_id,
            dw1_kernel_id,
            dw2_kernel_id,
            stream);
        HIP_CALL_LAUNCH(hipGetLastError());
    }
    else
    {
        // Internal training call sites have already established the current
        // device and validated the reusable tensor contract once.  Avoid two
        // hipSetDevice calls and a hot-path hipGetLastError query.
        opus_moe_backward::detail::launch_fixed_pipeline(
            down_args,
            route_dx_args,
            reduce_args,
            dw1_args,
            dw2_args,
            down_kernel_id,
            route_dx_kernel_id,
            route_reduce_kernel_id,
            dw1_kernel_id,
            dw2_kernel_id,
            stream);
    }
}

void opus_moe_full_bwd(aiter_tensor_t& d_out,
                       aiter_tensor_t& x,
                       aiter_tensor_t& z,
                       aiter_tensor_t& w1,
                       aiter_tensor_t& w2,
                       aiter_tensor_t& scores,
                       aiter_tensor_t& sorted_token_ids,
                       aiter_tensor_t& sorted_expert_ids,
                       aiter_tensor_t& num_valid_ids,
                       aiter_tensor_t& reverse_sorted,
                       aiter_tensor_t& expert_padded_offsets,
                       aiter_tensor_t& d_scores_workspace,
                       aiter_tensor_t& d_z,
                       aiter_tensor_t& a_scaled,
                       aiter_tensor_t& d_scores,
                       aiter_tensor_t& d_x_route,
                       aiter_tensor_t& d_x,
                       aiter_tensor_t& d_w1,
                       aiter_tensor_t& d_w2,
                       int block_m,
                       int down_kernel_id,
                       int route_dx_kernel_id,
                       int route_reduce_kernel_id,
                       int dw1_kernel_id,
                       int dw2_kernel_id)
{
    opus_moe_full_bwd_impl<true>(d_out,
                                 x,
                                 z,
                                 w1,
                                 w2,
                                 scores,
                                 sorted_token_ids,
                                 sorted_expert_ids,
                                 num_valid_ids,
                                 reverse_sorted,
                                 expert_padded_offsets,
                                 d_scores_workspace,
                                 d_z,
                                 a_scaled,
                                 d_scores,
                                 d_x_route,
                                 d_x,
                                 d_w1,
                                 d_w2,
                                 block_m,
                                 down_kernel_id,
                                 route_dx_kernel_id,
                                 route_reduce_kernel_id,
                                 dw1_kernel_id,
                                 dw2_kernel_id);
}

void opus_moe_full_bwd_trusted(aiter_tensor_t& d_out,
                               aiter_tensor_t& x,
                               aiter_tensor_t& z,
                               aiter_tensor_t& w1,
                               aiter_tensor_t& w2,
                               aiter_tensor_t& scores,
                               aiter_tensor_t& sorted_token_ids,
                               aiter_tensor_t& sorted_expert_ids,
                               aiter_tensor_t& num_valid_ids,
                               aiter_tensor_t& reverse_sorted,
                               aiter_tensor_t& expert_padded_offsets,
                               aiter_tensor_t& d_scores_workspace,
                               aiter_tensor_t& d_z,
                               aiter_tensor_t& a_scaled,
                               aiter_tensor_t& d_scores,
                               aiter_tensor_t& d_x_route,
                               aiter_tensor_t& d_x,
                               aiter_tensor_t& d_w1,
                               aiter_tensor_t& d_w2,
                               int block_m,
                               int down_kernel_id,
                               int route_dx_kernel_id,
                               int route_reduce_kernel_id,
                               int dw1_kernel_id,
                               int dw2_kernel_id)
{
    opus_moe_full_bwd_impl<false>(d_out,
                                  x,
                                  z,
                                  w1,
                                  w2,
                                  scores,
                                  sorted_token_ids,
                                  sorted_expert_ids,
                                  num_valid_ids,
                                  reverse_sorted,
                                  expert_padded_offsets,
                                  d_scores_workspace,
                                  d_z,
                                  a_scaled,
                                  d_scores,
                                  d_x_route,
                                  d_x,
                                  d_w1,
                                  d_w2,
                                  block_m,
                                  down_kernel_id,
                                  route_dx_kernel_id,
                                  route_reduce_kernel_id,
                                  dw1_kernel_id,
                                  dw2_kernel_id);
}
