# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Smoke-test for pa_decode_gluon_aot (Python path).
# Inputs match test_pa_gluon.cpp exactly (seed=42, same tensor creation order).

import torch
import math

num_seqs = 2
query_length = 3
num_query_heads = 8
num_kv_heads = 1
head_size = 128
kv_block_size = 64
max_num_blocks_per_seq = 4
num_blocks = num_seqs * max_num_blocks_per_seq
max_context_partition_num = 4
context_partition_size = 256
query_group_size = num_query_heads // num_kv_heads
eq_qgs = query_length * query_group_size
kv_elem_per_vec = 16 // torch.bfloat16.itemsize

torch.manual_seed(42)

query = torch.randn(num_seqs * query_length, num_query_heads, head_size, dtype=torch.bfloat16, device="cuda")
key_cache = torch.randn(num_blocks, num_kv_heads, head_size // kv_elem_per_vec, kv_block_size, kv_elem_per_vec, dtype=torch.bfloat16, device="cuda")
value_cache = torch.randn(num_blocks, num_kv_heads, head_size, kv_block_size, dtype=torch.bfloat16, device="cuda")
output = torch.zeros(num_seqs * query_length, num_query_heads, head_size, dtype=torch.bfloat16, device="cuda")
context_lengths = torch.full((num_seqs,), 64, dtype=torch.int32, device="cuda")
block_tables = torch.arange(num_seqs * max_num_blocks_per_seq, dtype=torch.int32, device="cuda").reshape(num_seqs, max_num_blocks_per_seq)
exp_sums = torch.zeros(num_seqs, num_kv_heads, max_context_partition_num, eq_qgs, dtype=torch.float32, device="cuda")
max_logits = torch.full((num_seqs, num_kv_heads, max_context_partition_num, eq_qgs), float("-inf"), dtype=torch.float32, device="cuda")
temporary_output = torch.zeros(num_seqs, num_kv_heads, max_context_partition_num, eq_qgs, head_size, dtype=torch.bfloat16, device="cuda")
softmax_scale = 1.0 / math.sqrt(head_size)

from csrc.cpp_itfs.pa_gluon_aot.pa_decode_gluon_aot import pa_decode_gluon_aot

pa_decode_gluon_aot(
    output, query, key_cache, value_cache,
    context_lengths, block_tables,
    softmax_scale, query_length,
    max_context_partition_num, context_partition_size,
    torch.bfloat16,
    None, None, None,
    exp_sums, max_logits, temporary_output,
    alibi_slopes=None, run_compiled_kernel=True, sinks=None,
)
torch.cuda.synchronize()
print(f"Python output sum = {output.to(torch.float32).sum().item()}")
