// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// OPUS attention backward on gfx950, bf16, D_QK = 192 / D_V = 128, of
//   out = attention(q, k, v) * sigmoid(gate)   (or of the attention alone).
#pragma once
#include <torch/extension.h>

// Pre-pass of the backward, per (token, head) row of 128:
//   s = sigmoid(gate)            (the forward epilogue's expression)
//   d_o   = dout * s             [total, H, 128] bf16, the attention output gradient
//   dgate = dout * out * (1 - s) [total, H, 128] bf16, the gate gradient
//   d     = rowsum(dout * out)   [H, total] fp32, equal to rowsum(dO * O) of the ungated O
// `out` is the gated forward output. All bf16 tensors are contiguous along the head dim with
// 16-byte aligned rows and int32 strides; d is contiguous along total. d_o, dgate and d are
// written, the other tensors only read.
void fmha_bwd_bf16_opus_gate(at::Tensor& dout,
                             at::Tensor& out,
                             at::Tensor& gate,
                             at::Tensor& d_o,
                             at::Tensor& dgate,
                             at::Tensor& d);

// The pre-pass of an attention without the gate: d = rowsum(dout * out) alone, dout then being
// the attention output gradient as it is.
void fmha_bwd_bf16_opus_dot(at::Tensor& dout, at::Tensor& out, at::Tensor& d);

// The attention's backward, group / varlen: the dK / dV pass and the dQ pass as workgroups of one
// kernel, in one launch.
//   q   [total_q, H, 192]   k [total_k, H_k, 192]   v [total_k, H_k, 128]   bf16
//   d_o [total_q, H, 128] bf16   lse, d [H, total_q] fp32
// H is a multiple of H_k: head h reads the K and V of head h / (H / H_k). The inputs' token
// strides cover a row (no broadcast or overlapping rows) and stay within 1 << 24 elements.
// Outputs, bf16, all written (not accumulated), and the only tensors written: dq [total_q, H,
// 192], dk [total_k, H, 192] and dv [total_k, H, 128], a row per query head h. With H > H_k, dk
// and dv are the per-query-head partials: the gradient of a K / V head is the sum of its query
// heads' rows, which the caller forms. dq and dk include softmax_scale. max_seqlen_q and
// max_seqlen_k are upper bounds of the sequences' lengths and size the grid.
void fmha_bwd_bf16_opus_bwd(at::Tensor& q,
                            at::Tensor& k,
                            at::Tensor& v,
                            at::Tensor& d_o,
                            at::Tensor& lse,
                            at::Tensor& d,
                            at::Tensor& dq,
                            at::Tensor& dk,
                            at::Tensor& dv,
                            at::Tensor& seqstart_q,
                            at::Tensor& seqstart_k,
                            int max_seqlen_q,
                            int max_seqlen_k,
                            bool causal,
                            float softmax_scale);
