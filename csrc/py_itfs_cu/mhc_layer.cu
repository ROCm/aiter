// SPDX-License-Identifier: MIT
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include "py_itfs_common.h"
#include "mhc_layer.h"
#include "mhc_layer.cuh"

namespace aiter {

static void check_device_match(const torch::Tensor &a, const torch::Tensor &b, const char *name) {
    TORCH_CHECK(a.device() == b.device(), "mhc_layer_fwd: ", name, " must be on same device");
}

static void check_debug_tensor(const torch::Tensor &t,
                               const torch::Tensor &ref,
                               const char *name) {
    TORCH_CHECK(t.is_cuda(), "mhc_layer_fwd_debug: ", name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), "mhc_layer_fwd_debug: ", name, " must be contiguous");
    TORCH_CHECK(t.device() == ref.device(),
                "mhc_layer_fwd_debug: ",
                name,
                " must be on same device as input");
}

void mhc_layer_fwd(torch::Tensor &out,
                   torch::Tensor &x_expanded,
                   torch::Tensor &rmsnorm_weight,
                   torch::Tensor &phi_pre,
                   torch::Tensor &phi_post,
                   torch::Tensor &phi_res,
                   torch::Tensor &b_pre,
                   torch::Tensor &b_post,
                   torch::Tensor &b_res,
                   double alpha_pre,
                   double alpha_post,
                   double alpha_res,
                   int64_t sinkhorn_iters,
                   double eps,
                   bool use_pdl)
{
    TORCH_CHECK(x_expanded.is_cuda(), "mhc_layer_fwd: x_expanded must be a CUDA tensor");
    TORCH_CHECK(out.is_cuda(), "mhc_layer_fwd: out must be a CUDA tensor");
    TORCH_CHECK(x_expanded.scalar_type() == torch::kFloat32,
                "mhc_layer_fwd: x_expanded must be float32");
    TORCH_CHECK(out.scalar_type() == torch::kFloat32, "mhc_layer_fwd: out must be float32");
    TORCH_CHECK(rmsnorm_weight.scalar_type() == torch::kBFloat16,
                "mhc_layer_fwd: rmsnorm_weight must be bfloat16");
    TORCH_CHECK(b_pre.scalar_type() == torch::kFloat32, "mhc_layer_fwd: b_pre must be float32");
    TORCH_CHECK(b_post.scalar_type() == torch::kFloat32, "mhc_layer_fwd: b_post must be float32");
    TORCH_CHECK(b_res.scalar_type() == torch::kFloat32, "mhc_layer_fwd: b_res must be float32");

    TORCH_CHECK(x_expanded.dim() == 3, "mhc_layer_fwd: x_expanded must be [B, n, C]");
    TORCH_CHECK(out.sizes() == x_expanded.sizes(),
                "mhc_layer_fwd: out must have same shape as x_expanded");

    TORCH_CHECK(x_expanded.is_contiguous(), "mhc_layer_fwd: x_expanded must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "mhc_layer_fwd: out must be contiguous");
    TORCH_CHECK(rmsnorm_weight.is_contiguous(), "mhc_layer_fwd: rmsnorm_weight must be contiguous");
    TORCH_CHECK(b_pre.is_contiguous(), "mhc_layer_fwd: b_pre must be contiguous");
    TORCH_CHECK(b_post.is_contiguous(), "mhc_layer_fwd: b_post must be contiguous");
    TORCH_CHECK(b_res.is_contiguous(), "mhc_layer_fwd: b_res must be contiguous");

    check_device_match(x_expanded, out, "out");
    check_device_match(x_expanded, rmsnorm_weight, "rmsnorm_weight");
    check_device_match(x_expanded, b_pre, "b_pre");
    check_device_match(x_expanded, b_post, "b_post");
    check_device_match(x_expanded, b_res, "b_res");

    const int64_t B = x_expanded.size(0);
    const int64_t n = x_expanded.size(1);
    const int64_t C = x_expanded.size(2);
    const int64_t nC = n * C;
    const int64_t n2 = n * n;

    TORCH_CHECK(rmsnorm_weight.numel() == C, "mhc_layer_fwd: rmsnorm_weight shape mismatch");
    TORCH_CHECK(b_pre.numel() == n, "mhc_layer_fwd: b_pre shape mismatch");
    TORCH_CHECK(b_post.numel() == n, "mhc_layer_fwd: b_post shape mismatch");
    TORCH_CHECK(b_res.numel() == n2, "mhc_layer_fwd: b_res shape mismatch");

    TORCH_CHECK(phi_pre.is_cuda() && phi_post.is_cuda() && phi_res.is_cuda(),
                "mhc_layer_fwd: phi tensors must be CUDA tensors");
    TORCH_CHECK(phi_pre.scalar_type() == torch::kBFloat16 &&
                    phi_post.scalar_type() == torch::kBFloat16 &&
                    phi_res.scalar_type() == torch::kBFloat16,
                "mhc_layer_fwd: phi tensors must be bfloat16");
    TORCH_CHECK(phi_pre.is_contiguous() && phi_post.is_contiguous() && phi_res.is_contiguous(),
                "mhc_layer_fwd: phi tensors must be contiguous");
    check_device_match(x_expanded, phi_pre, "phi_pre");
    check_device_match(x_expanded, phi_post, "phi_post");
    check_device_match(x_expanded, phi_res, "phi_res");

    TORCH_CHECK(phi_pre.dim() == 2 && phi_post.dim() == 2 && phi_res.dim() == 2,
                "mhc_layer_fwd: phi tensors must be 2D");
    TORCH_CHECK(phi_pre.size(0) == n && phi_pre.size(1) == nC,
                "mhc_layer_fwd: phi_pre shape mismatch");
    TORCH_CHECK(phi_post.size(0) == n && phi_post.size(1) == nC,
                "mhc_layer_fwd: phi_post shape mismatch");
    TORCH_CHECK(phi_res.size(0) == n2 && phi_res.size(1) == nC,
                "mhc_layer_fwd: phi_res shape mismatch");

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(x_expanded));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    mhc::MHCLayerConfig cfg;
    cfg.batch_size = static_cast<int>(B);
    cfg.hidden_dim = static_cast<int>(C);
    cfg.expansion_rate = static_cast<int>(n);
    cfg.sinkhorn_iters = static_cast<int>(sinkhorn_iters);
    cfg.eps = static_cast<float>(eps);
    cfg.use_pdl = use_pdl;
    cfg.use_dynamic_h = true;
    cfg.alpha_init = static_cast<float>(alpha_pre);

    mhc::MHCLayer layer;
    layer.init(cfg, stream);

    const size_t C_bytes = static_cast<size_t>(C) * sizeof(__hip_bfloat16);
    HIP_CALL(hipMemcpyAsync(layer.weights.rmsnorm_weight,
                               rmsnorm_weight.data_ptr(),
                               C_bytes,
                               hipMemcpyDeviceToDevice,
                               stream));

    const size_t phi_pre_bytes = static_cast<size_t>(nC) * n * sizeof(__hip_bfloat16);
    const size_t phi_post_bytes = static_cast<size_t>(nC) * n * sizeof(__hip_bfloat16);
    const size_t phi_res_bytes = static_cast<size_t>(nC) * n2 * sizeof(__hip_bfloat16);

    HIP_CALL(hipMemcpyAsync(layer.weights.phi_pre,
                               phi_pre.data_ptr(),
                               phi_pre_bytes,
                               hipMemcpyDeviceToDevice,
                               stream));
    HIP_CALL(hipMemcpyAsync(layer.weights.phi_post,
                               phi_post.data_ptr(),
                               phi_post_bytes,
                               hipMemcpyDeviceToDevice,
                               stream));
    HIP_CALL(hipMemcpyAsync(layer.weights.phi_res,
                               phi_res.data_ptr(),
                               phi_res_bytes,
                               hipMemcpyDeviceToDevice,
                               stream));

    HIP_CALL(hipMemcpyAsync(layer.weights.b_pre,
                               b_pre.data_ptr(),
                               static_cast<size_t>(n) * sizeof(float),
                               hipMemcpyDeviceToDevice,
                               stream));
    HIP_CALL(hipMemcpyAsync(layer.weights.b_post,
                               b_post.data_ptr(),
                               static_cast<size_t>(n) * sizeof(float),
                               hipMemcpyDeviceToDevice,
                               stream));
    HIP_CALL(hipMemcpyAsync(layer.weights.b_res,
                               b_res.data_ptr(),
                               static_cast<size_t>(n2) * sizeof(float),
                               hipMemcpyDeviceToDevice,
                               stream));

    layer.weights.alpha_pre = static_cast<float>(alpha_pre);
    layer.weights.alpha_post = static_cast<float>(alpha_post);
    layer.weights.alpha_res = static_cast<float>(alpha_res);

    layer.forward_device(x_expanded.data_ptr<float>());

    const size_t out_bytes = static_cast<size_t>(B) * n * C * sizeof(float);
    HIP_CALL(hipMemcpyAsync(out.data_ptr<float>(),
                               layer.get_output(),
                               out_bytes,
                               hipMemcpyDeviceToDevice,
                               stream));

    layer.sync();
    layer.destroy();
}

void mhc_layer_fwd_debug(torch::Tensor &out,
                         torch::Tensor &x_expanded,
                         torch::Tensor &rmsnorm_weight,
                         torch::Tensor &phi_pre,
                         torch::Tensor &phi_post,
                         torch::Tensor &phi_res,
                         torch::Tensor &b_pre,
                         torch::Tensor &b_post,
                         torch::Tensor &b_res,
                         double alpha_pre,
                         double alpha_post,
                         double alpha_res,
                         int64_t sinkhorn_iters,
                         double eps,
                         torch::Tensor &H_proj_raw,
                         torch::Tensor &H_pre,
                         torch::Tensor &H_post,
                         torch::Tensor &M,
                         torch::Tensor &x_agg_bf16,
                         torch::Tensor &layer_out_bf16,
                         torch::Tensor &rms_values,
                         bool use_pdl)
{
    const int64_t B = x_expanded.size(0);
    const int64_t n = x_expanded.size(1);
    const int64_t C = x_expanded.size(2);
    const int64_t nC = n * C;
    const int64_t n2 = n * n;
    const int64_t total_H_dim = n + n + n2;

    check_debug_tensor(H_proj_raw, x_expanded, "H_proj_raw");
    check_debug_tensor(H_pre, x_expanded, "H_pre");
    check_debug_tensor(H_post, x_expanded, "H_post");
    check_debug_tensor(M, x_expanded, "M");
    check_debug_tensor(x_agg_bf16, x_expanded, "x_agg_bf16");
    check_debug_tensor(layer_out_bf16, x_expanded, "layer_out_bf16");
    check_debug_tensor(rms_values, x_expanded, "rms_values");

    TORCH_CHECK(H_proj_raw.scalar_type() == torch::kFloat32,
                "mhc_layer_fwd_debug: H_proj_raw must be float32");
    TORCH_CHECK(H_pre.scalar_type() == torch::kFloat32,
                "mhc_layer_fwd_debug: H_pre must be float32");
    TORCH_CHECK(H_post.scalar_type() == torch::kFloat32,
                "mhc_layer_fwd_debug: H_post must be float32");
    TORCH_CHECK(M.scalar_type() == torch::kFloat32, "mhc_layer_fwd_debug: M must be float32");
    TORCH_CHECK(x_agg_bf16.scalar_type() == torch::kBFloat16,
                "mhc_layer_fwd_debug: x_agg_bf16 must be bfloat16");
    TORCH_CHECK(layer_out_bf16.scalar_type() == torch::kBFloat16,
                "mhc_layer_fwd_debug: layer_out_bf16 must be bfloat16");
    TORCH_CHECK(rms_values.scalar_type() == torch::kFloat32,
                "mhc_layer_fwd_debug: rms_values must be float32");

    TORCH_CHECK(H_proj_raw.numel() == B * total_H_dim,
                "mhc_layer_fwd_debug: H_proj_raw shape mismatch");
    TORCH_CHECK(H_pre.numel() == B * n, "mhc_layer_fwd_debug: H_pre shape mismatch");
    TORCH_CHECK(H_post.numel() == B * n, "mhc_layer_fwd_debug: H_post shape mismatch");
    TORCH_CHECK(M.numel() == B * n2, "mhc_layer_fwd_debug: M shape mismatch");
    TORCH_CHECK(x_agg_bf16.numel() == B * C, "mhc_layer_fwd_debug: x_agg_bf16 shape mismatch");
    TORCH_CHECK(layer_out_bf16.numel() == B * C,
                "mhc_layer_fwd_debug: layer_out_bf16 shape mismatch");
    TORCH_CHECK(rms_values.numel() == B, "mhc_layer_fwd_debug: rms_values shape mismatch");

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(x_expanded));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    mhc::MHCLayerConfig cfg;
    cfg.batch_size = static_cast<int>(B);
    cfg.hidden_dim = static_cast<int>(C);
    cfg.expansion_rate = static_cast<int>(n);
    cfg.sinkhorn_iters = static_cast<int>(sinkhorn_iters);
    cfg.eps = static_cast<float>(eps);
    cfg.use_pdl = use_pdl;
    cfg.use_dynamic_h = true;
    cfg.alpha_init = static_cast<float>(alpha_pre);

    mhc::MHCLayer layer;
    layer.init(cfg, stream);

    const size_t C_bytes = static_cast<size_t>(C) * sizeof(__hip_bfloat16);
    HIP_CALL(hipMemcpyAsync(layer.weights.rmsnorm_weight,
                               rmsnorm_weight.data_ptr(),
                               C_bytes,
                               hipMemcpyDeviceToDevice,
                               stream));

    const size_t phi_pre_bytes = static_cast<size_t>(nC) * n * sizeof(__hip_bfloat16);
    const size_t phi_post_bytes = static_cast<size_t>(nC) * n * sizeof(__hip_bfloat16);
    const size_t phi_res_bytes = static_cast<size_t>(nC) * n2 * sizeof(__hip_bfloat16);

    HIP_CALL(hipMemcpyAsync(layer.weights.phi_pre,
                               phi_pre.data_ptr(),
                               phi_pre_bytes,
                               hipMemcpyDeviceToDevice,
                               stream));
    HIP_CALL(hipMemcpyAsync(layer.weights.phi_post,
                               phi_post.data_ptr(),
                               phi_post_bytes,
                               hipMemcpyDeviceToDevice,
                               stream));
    HIP_CALL(hipMemcpyAsync(layer.weights.phi_res,
                               phi_res.data_ptr(),
                               phi_res_bytes,
                               hipMemcpyDeviceToDevice,
                               stream));

    HIP_CALL(hipMemcpyAsync(layer.weights.b_pre,
                               b_pre.data_ptr(),
                               static_cast<size_t>(n) * sizeof(float),
                               hipMemcpyDeviceToDevice,
                               stream));
    HIP_CALL(hipMemcpyAsync(layer.weights.b_post,
                               b_post.data_ptr(),
                               static_cast<size_t>(n) * sizeof(float),
                               hipMemcpyDeviceToDevice,
                               stream));
    HIP_CALL(hipMemcpyAsync(layer.weights.b_res,
                               b_res.data_ptr(),
                               static_cast<size_t>(n2) * sizeof(float),
                               hipMemcpyDeviceToDevice,
                               stream));

    layer.weights.alpha_pre = static_cast<float>(alpha_pre);
    layer.weights.alpha_post = static_cast<float>(alpha_post);
    layer.weights.alpha_res = static_cast<float>(alpha_res);

    layer.forward_device(x_expanded.data_ptr<float>());

    const size_t out_bytes = static_cast<size_t>(B) * n * C * sizeof(float);
    HIP_CALL(hipMemcpyAsync(out.data_ptr<float>(),
                               layer.get_output(),
                               out_bytes,
                               hipMemcpyDeviceToDevice,
                               stream));

    HIP_CALL(hipMemcpyAsync(H_proj_raw.data_ptr<float>(),
                               layer.buffers.H_proj_raw,
                               static_cast<size_t>(B) * total_H_dim * sizeof(float),
                               hipMemcpyDeviceToDevice,
                               stream));
    HIP_CALL(hipMemcpyAsync(H_pre.data_ptr<float>(),
                               layer.buffers.H_pre_activated,
                               static_cast<size_t>(B) * n * sizeof(float),
                               hipMemcpyDeviceToDevice,
                               stream));
    HIP_CALL(hipMemcpyAsync(H_post.data_ptr<float>(),
                               layer.buffers.H_post_activated,
                               static_cast<size_t>(B) * n * sizeof(float),
                               hipMemcpyDeviceToDevice,
                               stream));
    HIP_CALL(hipMemcpyAsync(M.data_ptr<float>(),
                               layer.buffers.sinkhorn_M,
                               static_cast<size_t>(B) * n2 * sizeof(float),
                               hipMemcpyDeviceToDevice,
                               stream));
    HIP_CALL(hipMemcpyAsync(x_agg_bf16.data_ptr(),
                               layer.buffers.x_aggregated_bf16,
                               static_cast<size_t>(B) * C * sizeof(__hip_bfloat16),
                               hipMemcpyDeviceToDevice,
                               stream));
    HIP_CALL(hipMemcpyAsync(layer_out_bf16.data_ptr(),
                               layer.buffers.layer_out_bf16,
                               static_cast<size_t>(B) * C * sizeof(__hip_bfloat16),
                               hipMemcpyDeviceToDevice,
                               stream));
    HIP_CALL(hipMemcpyAsync(rms_values.data_ptr<float>(),
                               layer.buffers.rms_values,
                               static_cast<size_t>(B) * sizeof(float),
                               hipMemcpyDeviceToDevice,
                               stream));

    layer.sync();
    layer.destroy();
}

} // namespace aiter
