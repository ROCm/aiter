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

    const size_t C_bytes = static_cast<size_t>(C) * sizeof(mhc::floatX);
    CHECK_CUDA(cudaMemcpyAsync(layer.weights.rmsnorm_weight,
                               rmsnorm_weight.data_ptr(),
                               C_bytes,
                               cudaMemcpyDeviceToDevice,
                               stream));

    const size_t phi_pre_bytes = static_cast<size_t>(nC) * n * sizeof(mhc::floatX);
    const size_t phi_post_bytes = static_cast<size_t>(nC) * n * sizeof(mhc::floatX);
    const size_t phi_res_bytes = static_cast<size_t>(nC) * n2 * sizeof(mhc::floatX);

    CHECK_CUDA(cudaMemcpyAsync(layer.weights.phi_pre,
                               phi_pre.data_ptr(),
                               phi_pre_bytes,
                               cudaMemcpyDeviceToDevice,
                               stream));
    CHECK_CUDA(cudaMemcpyAsync(layer.weights.phi_post,
                               phi_post.data_ptr(),
                               phi_post_bytes,
                               cudaMemcpyDeviceToDevice,
                               stream));
    CHECK_CUDA(cudaMemcpyAsync(layer.weights.phi_res,
                               phi_res.data_ptr(),
                               phi_res_bytes,
                               cudaMemcpyDeviceToDevice,
                               stream));

    CHECK_CUDA(cudaMemcpyAsync(layer.weights.b_pre,
                               b_pre.data_ptr(),
                               static_cast<size_t>(n) * sizeof(float),
                               cudaMemcpyDeviceToDevice,
                               stream));
    CHECK_CUDA(cudaMemcpyAsync(layer.weights.b_post,
                               b_post.data_ptr(),
                               static_cast<size_t>(n) * sizeof(float),
                               cudaMemcpyDeviceToDevice,
                               stream));
    CHECK_CUDA(cudaMemcpyAsync(layer.weights.b_res,
                               b_res.data_ptr(),
                               static_cast<size_t>(n2) * sizeof(float),
                               cudaMemcpyDeviceToDevice,
                               stream));

    layer.weights.alpha_pre = static_cast<float>(alpha_pre);
    layer.weights.alpha_post = static_cast<float>(alpha_post);
    layer.weights.alpha_res = static_cast<float>(alpha_res);

    layer.forward_device(x_expanded.data_ptr<float>());

    const size_t out_bytes = static_cast<size_t>(B) * n * C * sizeof(float);
    CHECK_CUDA(cudaMemcpyAsync(out.data_ptr<float>(),
                               layer.get_output(),
                               out_bytes,
                               cudaMemcpyDeviceToDevice,
                               stream));

    layer.sync();
    layer.destroy();
}

} // namespace aiter
