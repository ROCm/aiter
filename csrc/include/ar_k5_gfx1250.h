// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx1250 TP4 one-shot all-reduce + residual add + RMSNorm (+ native group-32 MXFP8) for
// decode sizes, hidden 6144 (see ar_k5_gfx1250.cuh). Every rank sets aside k5_ar_buffer_size
// bytes of an IPC buffer that every rank maps (the custom all-reduce carves them out of its
// Lamport inbox allocation, so k5 adds no allocation or IPC mapping), passes the address of
// that region on every rank to init_k5_ar once (it arms its own inbox and clears its counters,
// which follow the inbox in its own region) and barriers before the first call.
#pragma once
#include <cstdint>
#include <vector>

using fptr_t = int64_t;

namespace aiter {
int64_t k5_ar_buffer_size(int64_t world_size, int64_t max_rows, int64_t hidden);
fptr_t init_k5_ar(int64_t rank, const std::vector<int64_t>& inbox_ptrs, int64_t max_rows, int64_t hidden);
// bf16 [rows, hidden] operands, 16-byte aligned, on the current stream. lamport_order: the
// bits of lamport_ar_rmsnorm_gfx1250 (+ per_1x32_mx_quant_hip), else of the Gluon one-shot
// kernels. fp8_out: E4M3 [rows, hidden] at out, row-major E8M0 [rows, hidden / 32] at scale.
// maxit bounds every poll; a timeout counts in k5_ar_errors.
void k5_ar_rms(fptr_t ctx,
               int64_t inp,
               int64_t res_inp,
               int64_t w,
               int64_t out,
               int64_t res_out,
               int64_t scale,
               int64_t rows,
               int64_t hidden,
               double eps,
               bool gemma_norm,
               bool lamport_order,
               bool fp8_out,
               int64_t maxit);
// polls that timed out so far (synchronizes the device)
int64_t k5_ar_errors(fptr_t ctx);
} // namespace aiter
