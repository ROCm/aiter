// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_tensor.h"

// Generic K-grouped weight-gradient (BF16 operands -> FP32 accumulate).
//   dW[e] = sum_{m in [offs[e]:offs[e+1])} dy[m,:]^T (x) a[m,:]
//   dy : [M, P] bf16 ; a : [M, Q] bf16 ; expert_offsets : [E+1] int32
//   dW : [E, P, Q] fp32
// Rows must be grouped contiguously by expert (as produced by moe sorting /
// make_routing_offs). Serves both stage2 (dy,h->dW2) and stage1 (d_act,x->dW1).
// M1 scaffold: naive one-thread-per-output correctness kernel (no MFMA).
void opus_moe_wgrad_bf16(aiter_tensor_t& dy,
                         aiter_tensor_t& a,
                         aiter_tensor_t& expert_offsets,
                         aiter_tensor_t& dW);

// Generic M-grouped data-gradient (BF16). Each row m uses its expert's weight:
//   dh[m,:] = dy[m,:] @ w[row_expert[m]]        (contract over K = w dim1)
//   dy : [M,K] bf16 ; w : [E,K,N] bf16 ; row_expert : [M] int32 ; dh : [M,N] bf16
// Serves stage2 (dy,w2->dh) and stage1 (d_act,w1->dA). M1-style naive kernel.
void opus_moe_dgrad_bf16(aiter_tensor_t& dy,
                         aiter_tensor_t& w,
                         aiter_tensor_t& row_expert,
                         aiter_tensor_t& dh);

// Fused opus-MFMA grouped dgrad (BF16), COMPACT (unpadded) layout.
//   dy [M,K] bf16, w [E,N,K] bf16, sorted_expert_ids/block_m_start/block_m_end
//   [num_blocks] i32, dh [M,N] bf16
void opus_moe_dgrad_mfma_bf16(aiter_tensor_t& dy,
                              aiter_tensor_t& w,
                              aiter_tensor_t& sorted_expert_ids,
                              aiter_tensor_t& block_m_start,
                              aiter_tensor_t& block_m_end,
                              aiter_tensor_t& dh);

// Equal-routes stage-2 dgrad with a standard SwiGLU-backward epilogue.
// dy [E*M,K], w [E,N,K], act_input/d_act_input [E*M,2N], all BF16.
void opus_moe_dgrad_swiglu_bf16(aiter_tensor_t& dy,
                                aiter_tensor_t& w,
                                aiter_tensor_t& act_input,
                                aiter_tensor_t& d_act_input,
                                int uniform_m);

// Equal-routes stage-2 dgrad/SwiGLU plus per-route, per-N-tile dscore partials.
void opus_moe_dgrad_swiglu_dscore_bf16(aiter_tensor_t& dy,
                                       aiter_tensor_t& w,
                                       aiter_tensor_t& act_input,
                                       aiter_tensor_t& d_act_input,
                                       aiter_tensor_t& dscore_partials,
                                       int uniform_m);

// Build target-shape compact 192-row dgrad tile metadata entirely on GPU.
// tile_metadata stores E+1 prefix sums followed by packed tile descriptors.
void opus_moe_build_dgrad_meta_i32(aiter_tensor_t& expert_offsets,
                                   aiter_tensor_t& tile_metadata);

// Ragged compact-route variant of the fused stage-2 dgrad/SwiGLU/dscore path.
// expert_offsets [E+1] delimit operand rows; tile_offsets [E+1] delimit each
// expert's compact 192-row tile interval and num_tiles is its final value.
void opus_moe_dgrad_swiglu_dscore_ragged_bf16(
    aiter_tensor_t& dy,
    aiter_tensor_t& w,
    aiter_tensor_t& act_input,
    aiter_tensor_t& d_act_input,
    aiter_tensor_t& dscore_partials,
    aiter_tensor_t& expert_offsets,
    aiter_tensor_t& tile_offsets,
    int num_tiles,
    int max_m);

// Exact full-MoE variant: dscore partials are scattered from expert-grouped
// route rows into original (token, top-k rank) order via route_to_flat [M].
void opus_moe_dgrad_swiglu_dscore_ragged_flat_bf16(
    aiter_tensor_t& dy,
    aiter_tensor_t& w,
    aiter_tensor_t& act_input,
    aiter_tensor_t& d_act_input,
    aiter_tensor_t& dscore_partials,
    aiter_tensor_t& expert_offsets,
    aiter_tensor_t& tile_offsets,
    aiter_tensor_t& route_to_flat,
    int num_tiles,
    int max_m);

// Ragged compact-route plain dgrad through the 192x256x64 mono mainloop.
void opus_moe_dgrad_mono_ragged_bf16(aiter_tensor_t& dy,
                                     aiter_tensor_t& w,
                                     aiter_tensor_t& out,
                                     aiter_tensor_t& expert_offsets,
                                     aiter_tensor_t& tile_offsets,
                                     int num_tiles,
                                     int max_m);

// Fused opus-MFMA grouped wgrad (BF16->FP32), transposed+padded inputs.
//   dyT [P,Mp] bf16, aT [Q,Mp] bf16, pad_offs [E+1] i32, dW [E,P,Q] fp32
void opus_moe_wgrad_mfma_bf16(aiter_tensor_t& dyT,
                              aiter_tensor_t& aT,
                              aiter_tensor_t& pad_offs,
                              aiter_tensor_t& dW);

// Full-TN grouped wgrad (BF16->FP32): dW[e]=dy_e^T@a_e from natural compact
// dy [M,P] / a [M,Q] (no transpose/padding), offs [E+1] i32, dW [E,P,Q] fp32.
void opus_moe_wgrad_tn_bf16(aiter_tensor_t& dy,
                            aiter_tensor_t& a,
                            aiter_tensor_t& offs,
                            aiter_tensor_t& dW,
                            int uniform_m);

// Fused compact->feature-major pad+transpose (BF16). dst[F,Mp] = padded
// transpose of src[M,F]; col_to_m[col] = compact row of padded column col
// (-1 = padding). One pass, coalesced writes (replaces torch scatter+transpose).
void opus_moe_transpose_pad_bf16(aiter_tensor_t& src,
                                 aiter_tensor_t& col_to_m,
                                 aiter_tensor_t& dst);

// Elementwise activation backward (g1u1). d[gate;up] from dh (grad wrt post-act
// h) + pre-act gate/up. act: 0=Silu 1=Gelu 2=Swiglu(+1 bias) 3=SiTUv2(beta=lb=1).
//   dh : [M,I] bf16 ; act_input : [M,2I] bf16 ; d_act_input : [M,2I] bf16
void opus_moe_act_bwd_bf16(aiter_tensor_t& dh,
                           aiter_tensor_t& act_input,
                           aiter_tensor_t& d_act_input,
                           int act,
                           float swiglu_limit);

// Combine backward (M5). dy[m,:] = p[m]*dout[gather[m],:]; dp[m] =
// <dout[gather[m],:], y[m,:]>. dout [T,H] bf16, gather [M] i32, p [M] fp32,
// y [M,H] bf16 -> dy [M,H] bf16, dp [M] fp32.
void opus_moe_combine_bwd_bf16(aiter_tensor_t& dout,
                               aiter_tensor_t& gather,
                               aiter_tensor_t& p,
                               aiter_tensor_t& y,
                               aiter_tensor_t& dy,
                               aiter_tensor_t& dp);

// Exact Sonic-parity token-major combine. token_routes[t,k] is the compact
// route row for token t's kth selection. One block shares dout[t,:] across all
// eight routes, avoiding the route-major kernel's eight identical dout reads.
void opus_moe_combine_bwd_token8_h2048_bf16(aiter_tensor_t& dout,
                                             aiter_tensor_t& token_routes,
                                             aiter_tensor_t& p,
                                             aiter_tensor_t& y,
                                             aiter_tensor_t& dy,
                                             aiter_tensor_t& dp);

void opus_moe_combine_scale_token8_h2048_bf16(
    aiter_tensor_t& dout,
    aiter_tensor_t& token_routes,
    aiter_tensor_t& p,
    aiter_tensor_t& dy);

void opus_moe_dscore_router_bwd_token8_e64_bf16(
    aiter_tensor_t& token_routes,
    aiter_tensor_t& p,
    aiter_tensor_t& partials,
    aiter_tensor_t& order,
    aiter_tensor_t& topk_ids,
    aiter_tensor_t& dlogits);

// Exact Sonic-parity combine + softmax-router backward. In addition to sharing
// dout[t,:] across the token's eight routes, the route-dot reductions are
// consumed in the same CTA to produce the BF16 [T,64] router-logit gradient.
void opus_moe_combine_router_bwd_token8_h2048_e64_bf16(
    aiter_tensor_t& dout,
    aiter_tensor_t& token_routes,
    aiter_tensor_t& p,
    aiter_tensor_t& y,
    aiter_tensor_t& order,
    aiter_tensor_t& topk_ids,
    aiter_tensor_t& dy,
    aiter_tensor_t& dlogits);

// dx scatter-add (M5). dst[gather[m],:] += src[m,:] (topk routes -> token).
// src [M,H] bf16, gather [M] i32, dst [T,H] fp32 (pre-zeroed).
void opus_moe_scatter_add_bf16(aiter_tensor_t& src,
                               aiter_tensor_t& gather,
                               aiter_tensor_t& dst);

// Deterministic dx gather-sum (fixed top-k, no atomics). One block per token
// sums its topk route rows. src [M,H] bf16, token_routes [T,topk] i32,
// dst [T,H] bf16 (FP32 accumulation). Requires H % 8 == 0.
void opus_moe_gather_sum_bf16(aiter_tensor_t& src,
                              aiter_tensor_t& token_routes,
                              aiter_tensor_t& dst);

void opus_moe_gather_sum_dscore_router_token8_h2048_e64_bf16(
    aiter_tensor_t& src,
    aiter_tensor_t& token_routes,
    aiter_tensor_t& route_scores,
    aiter_tensor_t& partials,
    aiter_tensor_t& order,
    aiter_tensor_t& topk_ids,
    aiter_tensor_t& dst,
    aiter_tensor_t& dlogits);

void opus_moe_gather_sum_dscore_router_dx_token8_h2048_e64_bf16(
    aiter_tensor_t& src,
    aiter_tensor_t& token_routes,
    aiter_tensor_t& route_scores,
    aiter_tensor_t& partials,
    aiter_tensor_t& order,
    aiter_tensor_t& topk_ids,
    aiter_tensor_t& router_w,
    aiter_tensor_t& dst,
    aiter_tensor_t& dlogits);

// Keep expert-grouped route gradients, but consume dscore metadata in flat
// [T,topk] order so the sparse router tail needs no order indirection.
void opus_moe_gather_sum_dscore_router_dx_flat_meta_token8_h2048_e64_bf16(
    aiter_tensor_t& src,
    aiter_tensor_t& token_routes,
    aiter_tensor_t& route_scores,
    aiter_tensor_t& partials,
    aiter_tensor_t& topk_ids,
    aiter_tensor_t& router_w,
    aiter_tensor_t& dst,
    aiter_tensor_t& dlogits);

// Router backward (M5 R7): softmax-over-topk Jacobian.
// dp [T,topk] fp32, topk_w [T,topk] fp32, topk_ids [T,topk] i32 ->
// dlogits [T,E] fp32 (pre-zeroed). dlogits[t,ids[t,k]] = pw*(dp - Σ dp·pw).
void opus_moe_router_bwd_bf16(aiter_tensor_t& dp,
                              aiter_tensor_t& topk_w,
                              aiter_tensor_t& topk_ids,
                              aiter_tensor_t& dlogits);

// Router backward, sigmoid scoring + optional renorm (DeepSeek/Kimi).
// dp [T,topk] fp32, logits [T,E] fp32, topk_ids [T,topk] i32 -> dlogits [T,E]
// fp32 (pre-zeroed). renorm!=0: w=s/Σs Jacobian; else dl=dp·s(1-s).
void opus_moe_router_bwd_sigmoid_bf16(aiter_tensor_t& dp,
                                      aiter_tensor_t& logits,
                                      aiter_tensor_t& topk_ids,
                                      aiter_tensor_t& dlogits,
                                      int renorm);
