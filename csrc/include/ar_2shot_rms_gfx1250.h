// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx1250 pushed two-shot all-reduce + residual add + RMSNorm for large
// (prefill) messages, see ar_2shot_rms_gfx1250.cu. Every rank allocates a data
// buffer of ar2s_rms_buffer_size bytes and a flag buffer of ar2s_rms_flags_size
// bytes (both zeroed, cached, IPC-exported), maps the peers' buffers, calls
// init_ar2s_rms once and barriers before the first call.
#pragma once
#include <cstdint>
#include <vector>

using fptr_t = int64_t;

namespace aiter {
int64_t ar2s_rms_buffer_size(int64_t world_size, int64_t max_bytes);
int64_t ar2s_rms_flags_size();
fptr_t init_ar2s_rms(int64_t rank,
                     const std::vector<int64_t>& data_ptrs,
                     const std::vector<int64_t>& flag_ptrs,
                     int64_t max_bytes,
                     int64_t grid);
// rows a single ar2s_rms call takes at this hidden size
int64_t ar2s_rms_max_rows(fptr_t ctx, int64_t hidden);
// bf16 [rows, hidden] operands, 16-byte aligned, on the current stream
void ar2s_rms(fptr_t ctx,
              int64_t inp,
              int64_t res_inp,
              int64_t w,
              int64_t out,
              int64_t res_out,
              int64_t rows,
              int64_t hidden,
              double eps,
              bool gemma_norm);
// ar2s_rms with options: fp8_out writes MXFP8 (E4M3 [rows, hidden] at out_q, row-major E8M0
// [rows, hidden / 32] at out_s; out is not written); sp_residual keeps the residual of each
// rank's own chunk only (rows [rank * R, rank * R + R), R = ceil(rows / world)): res_inp is
// read and res_out written there alone, while out (or out_q / out_s) is whole on every rank.
// aux_residual (with sp_residual, bf16): no norm, the whole residual_out on every rank (out unused).
void ar2s_rms_ex(fptr_t ctx,
                 int64_t inp,
                 int64_t res_inp,
                 int64_t w,
                 int64_t out,
                 int64_t res_out,
                 int64_t out_q,
                 int64_t out_s,
                 int64_t rows,
                 int64_t hidden,
                 double eps,
                 bool gemma_norm,
                 bool fp8_out,
                 bool sp_residual,
                 bool aux_residual,
                 int64_t grid); // > 0: cap the blocks of this call (same on every rank)
// MiniMax-M3 MoE front (sequence-parallel residual, TP2 / TP4, hidden % 512 == 0): the
// all-reduce + residual add + RMSNorm of ar2s_rms_ex(sp_residual) whose owner quantizes its rows
// to the MoE layer's inputs and all-gathers those instead of the bf16 rows. Whole on every rank:
// q8 [rows, hidden] E4M3 + s8 [rows, hidden / 32] E8M0 (the shared expert's MXFP8 input, the bytes
// of ATOM quantize_fp8: amax floored at 1e-10) and q4 [rows, hidden / 2] + s4 [rows, hidden / 32]
// (the routed experts' MXFP4 A1, the bytes of the FlyDSL MoE A1 quant; E2M1 element 2i in the low
// nibble). Own rows only: res_out (as sp_residual) and out_own [own rows, hidden] bf16, the
// normalized own rows, compact (row j of the own chunk at row j; null when the chunk is empty).
void ar2s_rms_moe(fptr_t ctx,
                  int64_t inp,
                  int64_t res_inp,
                  int64_t w,
                  int64_t out_own,
                  int64_t res_out,
                  int64_t q8,
                  int64_t s8,
                  int64_t q4,
                  int64_t s4,
                  int64_t rows,
                  int64_t hidden,
                  double eps,
                  bool gemma_norm,
                  int64_t grid);
} // namespace aiter
