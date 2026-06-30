// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include <torch/extension.h>

// Fused intra-chunk GDN prefill preparation:
//   chunk_local_cumsum + chunk_scaled_dot_kkt_fwd + solve_tril + recompute_w_u_fwd
void gdn_chunk_prepare_fwd(
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor g,
    torch::Tensor beta,
    torch::Tensor w_bar,
    torch::Tensor u_bar,
    torch::Tensor g_cumsum);
