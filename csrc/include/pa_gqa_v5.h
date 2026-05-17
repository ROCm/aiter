// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// pa_gqa v5 paged-attention decode (AMD MI300 / gfx942) — drives the asm
// kernel pair shipped in hsa/gfx942/pa_gqa_v5/asm_pa_gqa_v5.co via
// AiterAsmKernel.  Hard-specialised for:
//
//   bf16 Q/K/V, num_kv_heads=1, num_heads=8 (GQA=8), head=128, block=16,
//   mtp=2, partition_size=256, max ctx <= 524288.
//
// All workspace tensors must be allocated by the caller.  Bit-identical to
// the in-tree HIP backend (`paged_attention_rocm`); ~1.5x-2.7x faster on
// the supported config.
#pragma once

#include <torch/extension.h>

void pa_gqa_v5_decode(torch::Tensor& out,            // [num_seqs*MTP, num_heads, head_size]
                      torch::Tensor& exp_sums,       // [num_seqs*MTP, num_heads, max_num_partitions]
                      torch::Tensor& max_logits,     // [num_seqs*MTP, num_heads, max_num_partitions]
                      torch::Tensor& tmp_out,        // [num_seqs*MTP, num_heads, max_num_partitions, head_size]
                      const torch::Tensor& query,    // [num_seqs*MTP, num_heads, head_size]
                      const torch::Tensor& key_cache,
                      const torch::Tensor& value_cache,
                      const torch::Tensor& block_tables,
                      const torch::Tensor& context_lens,
                      int64_t num_kv_heads,
                      double scale,
                      int64_t block_size,
                      int64_t max_context_len,
                      int64_t partition_size,
                      int64_t mtp);
