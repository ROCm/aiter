#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
// MHC (Multi-Head Channel) Layer implementation for aiter

#include <torch/extension.h>

/**
 * MHC Layer Forward Pass
 *
 * Performs the MHC layer forward computation with:
 * 1. Stream aggregation with H_pre weighting
 * 2. RMSNorm on aggregated input
 * 3. Sinkhorn-Knopp normalization on H_res
 * 4. Stream distribution with H_post scaling and mixing
 *
 * @param x_expanded Input tensor [B, n, C] - expanded representation
 * @param rmsnorm_weight RMSNorm weight [C]
 * @param H_pre Pre-aggregation weights [n] (will be sigmoid activated)
 * @param H_post Post-distribution weights [n] (will be 2*sigmoid activated)
 * @param H_res Residual mixing weights [n, n] (will be exp + Sinkhorn-Knopp)
 * @param eps Epsilon for numerical stability
 * @param sinkhorn_iters Number of Sinkhorn-Knopp iterations
 * @return Output tensor [B, n, C]
 */
torch::Tensor mhc_layer_forward(
    torch::Tensor& x_expanded,
    torch::Tensor& rmsnorm_weight,
    torch::Tensor& H_pre,
    torch::Tensor& H_post,
    torch::Tensor& H_res,
    float eps,
    int sinkhorn_iters);

/**
 * MHC Layer Forward Pass (Dynamic H)
 *
 * Performs the MHC layer forward computation with dynamic H parameters:
 * H_pre, H_post, H_res are computed from x_expanded via projection matrices
 *
 * @param x_expanded Input tensor [B, n, C] - expanded representation
 * @param rmsnorm_weight RMSNorm weight [C]
 * @param phi_pre Projection matrix for H_pre [n, n*C]
 * @param phi_post Projection matrix for H_post [n, n*C]
 * @param phi_res Projection matrix for H_res [n*n, n*C]
 * @param b_pre Bias for H_pre [n]
 * @param b_post Bias for H_post [n]
 * @param b_res Bias for H_res [n, n]
 * @param alpha_pre Scaling for H_pre projection
 * @param alpha_post Scaling for H_post projection
 * @param alpha_res Scaling for H_res projection
 * @param eps Epsilon for numerical stability
 * @param sinkhorn_iters Number of Sinkhorn-Knopp iterations
 * @return Output tensor [B, n, C]
 */
torch::Tensor mhc_layer_forward_dynamic(
    torch::Tensor& x_expanded,
    torch::Tensor& rmsnorm_weight,
    torch::Tensor& phi_pre,
    torch::Tensor& phi_post,
    torch::Tensor& phi_res,
    torch::Tensor& b_pre,
    torch::Tensor& b_post,
    torch::Tensor& b_res,
    float alpha_pre,
    float alpha_post,
    float alpha_res,
    float eps,
    int sinkhorn_iters);

/**
 * Sinkhorn-Knopp Normalization Forward
 *
 * Applies Sinkhorn-Knopp algorithm to normalize a matrix to be doubly stochastic
 *
 * @param inp Input matrix [M, N] or batch [B, M, N]
 * @param num_iters Number of iterations
 * @param eps Epsilon for numerical stability
 * @return Doubly stochastic matrix
 */
torch::Tensor sinkhorn_knopp_forward(
    torch::Tensor& inp,
    int num_iters,
    float eps);

/**
 * Stream Aggregate Forward
 *
 * Aggregates streams with weighted sum: out[b,c] = sum_i(H_pre[i] * x[b,i,c])
 *
 * @param x_expanded Input tensor [B, n, C]
 * @param H_pre Weights [n] or [B, n] for dynamic
 * @return Aggregated tensor [B, C]
 */
torch::Tensor stream_aggregate_forward(
    torch::Tensor& x_expanded,
    torch::Tensor& H_pre);

/**
 * Stream Distribute Mix Add Forward
 *
 * Distributes and mixes: out[b,i,c] = H_post[i] * y[b,c] + sum_j(M[i,j] * x[b,j,c])
 *
 * @param x_expanded Input tensor [B, n, C]
 * @param y RMSNorm output [B, C]
 * @param H_post Distribution weights [n] or [B, n]
 * @param M Mixing matrix [n, n] or [B, n, n] (doubly stochastic)
 * @return Output tensor [B, n, C]
 */
torch::Tensor stream_distribute_mix_add_forward(
    torch::Tensor& x_expanded,
    torch::Tensor& y,
    torch::Tensor& H_post,
    torch::Tensor& M);
