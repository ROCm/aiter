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
} // namespace aiter
