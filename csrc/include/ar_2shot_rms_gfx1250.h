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
} // namespace aiter
