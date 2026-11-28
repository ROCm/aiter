// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "aiter_tensor.h"
#include <hip/hip_runtime.h>
#include <optional>

namespace aiter {
void mhc_pre_gemm_sqrsum(aiter_tensor_t& out,    // (split_k, m, hc_mult3) / (m, hc_mult3)
                         aiter_tensor_t& sqrsum, // (split_k, m) / (m)
                         aiter_tensor_t& x,      // (m, hc_hidden_size)
                         aiter_tensor_t& fn,     // (hc_mult3, hc_hidden_size) fp32; packed int32 when w_preshuffle_bf16
                         int tile_k = 128,
                         int w_preshuffle_bf16 = 0);
void mhc_pre_big_fuse(aiter_tensor_t& post_mix,        // (m, hc_mult)
                      aiter_tensor_t& comb_mix,        // (m, hc_mult * hc_mult)
                      aiter_tensor_t& layer_input,     // (m, hidden_size)
                      aiter_tensor_t& gemm_out_mul,    // (split_k, m, hc_mult3)
                      aiter_tensor_t& gemm_out_sqrsum, // (split_k, m)
                      aiter_tensor_t& hc_scale,        // (3)
                      aiter_tensor_t& hc_base,         // (hc_mult3)
                      aiter_tensor_t& residual,        // (m, hc_mult, hidden_size)
                      float rms_eps            = 1e-6,
                      float hc_pre_eps         = 1e-6,
                      float hc_sinkhorn_eps    = 1e-6,
                      float hc_post_mult_value = 1.0,
                      int sinkhorn_repeat      = 20,
                      // 1: residual is in the pre-shuffled resS[k/KS][head][row][k%KS]
                      // layout written by mhc_fused_post_pre_gemm_sqrsum.
                      int res_preshuffle       = 0);
void mhc_pre_big_fuse_rmsnorm(aiter_tensor_t& post_mix,        // (m, hc_mult)
                              aiter_tensor_t& comb_mix,        // (m, hc_mult * hc_mult)
                              aiter_tensor_t& out,             // (m, hidden_size)
                              aiter_tensor_t& gemm_out_mul,    // (split_k, m, hc_mult3)
                              aiter_tensor_t& gemm_out_sqrsum, // (split_k, m)
                              aiter_tensor_t& hc_scale,        // (3)
                              aiter_tensor_t& hc_base,         // (hc_mult3)
                              aiter_tensor_t& residual,        // (m, hc_mult, hidden_size)
                              aiter_tensor_t& norm_weight,     // (hidden_size)
                              float rms_eps            = 1e-6,
                              float hc_pre_eps         = 1e-6,
                              float hc_sinkhorn_eps    = 1e-6,
                              float norm_eps           = 1e-6,
                              float hc_post_mult_value = 1.0,
                              int sinkhorn_repeat      = 20,
                              // 1: residual is in the pre-shuffled
                              // resS[k/KS][head][row][k%KS] layout.
                              int res_preshuffle       = 0);
void mhc_post(aiter_tensor_t& out,            // (m, hc_mult, hidden_size)
              aiter_tensor_t& x,              // (m, hidden_size)
              aiter_tensor_t& residual,       // (m, hc_mult, hidden_size)
              aiter_tensor_t& post_layer_mix, // (m, hc_mult)
              aiter_tensor_t& comb_res_mix,   // (m, hc_mult, hc_mult)
              int store_nt                    = -1);
// mhc_post with its hc_mult output streams reduced in the epilogue, into out
// (m, hidden_size): each stream exactly as mhc_post writes it, then summed in
// FP32 with h ascending -- weighted by pre_mix (products and sums rounded
// apart), or with no pre_mix scaled by 1 / hc_mult (the mean). Bit-identical
// to mhc_post followed by that reduction. residual is the plain layout (no
// res_preshuffle). Coverage: mhc_post_reduce_supported in aiter/ops/mhc.py (the
// tiles of mhc_post_tile without the 256-column one).
void mhc_post_reduce(aiter_tensor_t& out,                   // (m, hidden_size)
                     aiter_tensor_t& x,                     // (m, hidden_size)
                     aiter_tensor_t& residual,              // (m, hc_mult, hidden_size)
                     aiter_tensor_t& post_layer_mix,        // (m, hc_mult)
                     aiter_tensor_t& comb_res_mix,          // (m, hc_mult, hc_mult)
                     std::optional<aiter_tensor_t> pre_mix, // (m, hc_mult) fp32, or none
                     int store_nt = -1);
// Optimized mhc_post launch on raw device pointers (used by fused AR+MHC split epilogue).
void launch_mhc_post_raw(hipStream_t stream,
                         AiterDtype dtype,
                         void* out,
                         void* x,
                         void* residual,
                         void* post_layer_mix,
                         void* comb_res_mix,
                         int m,
                         int hidden_size,
                         int x_stride,
                         int residual_stride,
                         int store_nt = -1);
void mhc_fused_post_pre_gemm_sqrsum(
    aiter_tensor_t& gemm_out_mul,    // (split_k, m, hc_mult3)
    aiter_tensor_t& gemm_out_sqrsum, // (split_k, m)
    aiter_tensor_t& next_residual,   // (m, hc_mult, hidden_size)
    aiter_tensor_t& layer_input,     // (m, hidden_size)
    aiter_tensor_t& residual_in,     // (m, hc_mult, hidden_size)
    aiter_tensor_t& post_layer_mix,  // (m, hc_mult)
    aiter_tensor_t& comb_res_mix,    // (m, hc_mult, hc_mult)
    aiter_tensor_t& fn,              // (hc_mult3, hc_mult * hidden_size) fp32; packed int32
                                     // when w_preshuffle_bf16
    int tile_m                       = 16,
    int tile_n                       = 32,
    int tile_k                       = 32,
    // Independent compute and residual-layout controls.
    int w_preshuffle_bf16           = 0,
    int res_preshuffle              = 0); // shuffled residuals require gfx1250
// gfx950 decode route of mhc_fused_post_pre with fused RMSNorm (mhc_post_pre_gram.cu):
// four BF16 residual streams, 20 Sinkhorn iterations. partials is scratch; its first
// dimension is the number of hidden slabs of the producer.
void mhc_post_pre_gram_gfx950(aiter_tensor_t& post,          // (m, 4, 1)
                              aiter_tensor_t& comb,          // (m, 4, 4)
                              aiter_tensor_t& out,           // (m, hidden_size)
                              aiter_tensor_t& next_residual, // (m, 4, hidden_size)
                              aiter_tensor_t& partials,      // (split_k, m, 48)
                              aiter_tensor_t& layer_input,   // (m, hidden_size)
                              aiter_tensor_t& residual_in,   // (m, 4, hidden_size)
                              aiter_tensor_t& post_mix,      // (m, 4)
                              aiter_tensor_t& comb_mix,      // (m, 4, 4)
                              aiter_tensor_t& fn,            // (24, 4 * hidden_size) fp32 or packed
                              aiter_tensor_t& hc_scale,      // (3)
                              aiter_tensor_t& hc_base,       // (24)
                              aiter_tensor_t& weight,        // (hidden_size)
                              float rms_eps,
                              float pre_eps,
                              float sink_eps,
                              float norm_eps,
                              float post_mult,
                              int parts = 4); // hidden segments of a row in the finisher
// gfx950 packed prefill route of mhc_fused_post_pre (mhc_post_pre_gfx950.cu). All
// buffers are owned by the current call.
void mhc_post_project_gfx950(aiter_tensor_t& projection,
                             aiter_tensor_t& sq,
                             aiter_tensor_t& next_residual,
                             aiter_tensor_t& x,
                             aiter_tensor_t& residual,
                             aiter_tensor_t& packed_fn,
                             aiter_tensor_t& post_mix,
                             aiter_tensor_t& comb_mix);
void mhc_finalize_gfx950(aiter_tensor_t& post,
                         aiter_tensor_t& comb,
                         aiter_tensor_t& out,
                         aiter_tensor_t& projection,
                         aiter_tensor_t& sq,
                         aiter_tensor_t& scale,
                         aiter_tensor_t& base,
                         aiter_tensor_t& residual,
                         aiter_tensor_t& weight,
                         float rms_eps,
                         float pre_eps,
                         float sink_eps,
                         float norm_eps,
                         float post_mult,
                         int repeats);
} // namespace aiter
