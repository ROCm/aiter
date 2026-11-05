// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_tensor.h"
#include <cstdint>
#include <optional>

namespace aiter {

// Gather + dequantize the DeepSeek V4 paged K record (584 B per token:
// 448 fp8 e4m3 NoPE dims, 64 bf16 RoPE dims, then 7 UE8M0 scales of 64 dims
// plus a pad byte) into a bf16 workspace.
//
//   out          [num_reqs, max_num_tokens, head_size] bf16, 512 dims written
//   k_cache      [num_blocks, block_size, 584] uint8
//   seq_lens     [num_reqs] int32
//   gather_lens  [num_reqs] int32, optional (nullopt gathers the whole seq)
//   block_table  [num_reqs, max_blocks_per_seq] int32
//   block_size   tokens per paged block
//   offset       first output column written
//   use_fnuz     true if the fp8 payload is e4m3fnuz, false for OCP e4m3
void dsv4_dequantize_and_gather_k(aiter_tensor_t& out,
                                  const aiter_tensor_t& k_cache,
                                  const aiter_tensor_t& seq_lens,
                                  std::optional<aiter_tensor_t> gather_lens,
                                  const aiter_tensor_t& block_table,
                                  int64_t block_size,
                                  int64_t offset,
                                  bool use_fnuz);

} // namespace aiter
