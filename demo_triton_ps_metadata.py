#!/usr/bin/env python
"""
Demonstration of the new Triton GPU-native implementation of get_ps_metadata_v1.

This script shows that all tensors can remain on GPU throughout the entire
metadata generation process, eliminating CPU transfers.
"""

import sys
import torch
import aiter
from aiter.jit.utils.chip_info import get_gfx

if get_gfx() == "gfx942":
    print("Skipping: only supported on gfx950")
    sys.exit(0)

torch.set_default_device("cuda")

print("=" * 70)
print("Triton GPU-Native Implementation of get_ps_metadata_v1")
print("=" * 70)
print()

# Configuration
device = 'cuda:0'
batch_size = 4
seq_lens_qo = [64, 128, 192, 256]
seq_lens_kv = [64, 128, 192, 256]
gqa_ratio = 8
num_heads_k = 2
block_size = 16
qlen_granularity = 256
kvlen_granularity = 128

print(f"Configuration:")
print(f"  Device: {device}")
print(f"  Batch size: {batch_size}")
print(f"  Sequence lengths (QO): {seq_lens_qo}")
print(f"  Sequence lengths (KV): {seq_lens_kv}")
print(f"  GQA ratio: {gqa_ratio}")
print(f"  Num KV heads: {num_heads_k}")
print()

# Create GPU tensors
qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
context_lens = torch.tensor(seq_lens_kv, dtype=torch.int32, device=device)

qo_indptr[1:] = torch.cumsum(torch.tensor(seq_lens_qo, dtype=torch.int32, device=device), dim=0)
actual_blocks = [(kv_len + block_size - 1) // block_size for kv_len in seq_lens_kv]
kv_indptr[1:] = torch.cumsum(torch.tensor(actual_blocks, dtype=torch.int32, device=device), dim=0)

max_qlen = max(seq_lens_qo)

# Get metadata buffer sizes
result = aiter.get_ps_metadata_info_v1(
    batch_size=batch_size,
    num_head_k=num_heads_k,
    max_qlen=max_qlen,
    qlen_granularity=qlen_granularity,
)

# Allocate output buffers on GPU
work_metadata_ptrs = torch.empty(result[0][0], dtype=result[0][1], device=device)
work_indptr = torch.empty(result[1][0], dtype=result[1][1], device=device)
work_info = torch.empty(result[2][0], dtype=result[2][1], device=device)
reduce_indptr = torch.empty(result[3][0], dtype=result[3][1], device=device)
reduce_final_map = torch.empty(result[4][0], dtype=result[4][1], device=device)
reduce_partial_map = torch.empty(result[5][0], dtype=result[5][1], device=device)

print("Input tensors:")
print(f"  qo_indptr: shape={qo_indptr.shape}, device={qo_indptr.device}, dtype={qo_indptr.dtype}")
print(f"  kv_indptr: shape={kv_indptr.shape}, device={kv_indptr.device}, dtype={kv_indptr.dtype}")
print(f"  context_lens: shape={context_lens.shape}, device={context_lens.device}, dtype={context_lens.dtype}")
print()

print("Output buffer allocation:")
print(f"  work_indptr: shape={work_indptr.shape}, device={work_indptr.device}")
print(f"  work_info: shape={work_info.shape}, device={work_info.device}")
print(f"  reduce_indptr: shape={reduce_indptr.shape}, device={reduce_indptr.device}")
print(f"  reduce_final_map: shape={reduce_final_map.shape}, device={reduce_final_map.device}")
print(f"  reduce_partial_map: shape={reduce_partial_map.shape}, device={reduce_partial_map.device}")
print()

# Generate metadata entirely on GPU
print("🚀 Calling get_ps_metadata_v1 (Triton GPU implementation)...")
print("   No CPU transfers required!")
print()

aiter.get_ps_metadata_v1(
    qo_indptr,
    kv_indptr,
    context_lens,
    gqa_ratio=gqa_ratio,
    num_heads_k=num_heads_k,
    work_metadata_ptrs=work_metadata_ptrs,
    work_indptr=work_indptr,
    work_info=work_info,
    reduce_indptr=reduce_indptr,
    reduce_final_map=reduce_final_map,
    reduce_partial_map=reduce_partial_map,
    qhead_granularity=gqa_ratio,
    qlen_granularity=qlen_granularity,
    kvlen_granularity=kvlen_granularity,
    block_size=block_size,
    is_causal=True,
)

print("✅ Metadata generation complete!")
print()

# Display results
total_works = work_indptr[-1].item()
print(f"Results:")
print(f"  Total work items generated: {total_works}")
print(f"  Work items per thread group (first 10): {work_indptr[:11].cpu().tolist()}")
print()

# Verify all outputs are still on GPU
print("Output verification:")
print(f"  work_indptr device: {work_indptr.device} ✓")
print(f"  work_info device: {work_info.device} ✓")
print(f"  reduce_indptr device: {reduce_indptr.device} ✓")
print()

# Show a few work items
print(f"Sample work items (first 3):")
for i in range(min(3, total_works)):
    work = work_info[i].cpu()
    print(f"  Work {i}: batch={work[0]}, partial={work[1]}, "
          f"qo=[{work[2]},{work[3]}), kv=[{work[4]},{work[5]})")
print()

print("=" * 70)
print("Key Benefits:")
print("=" * 70)
print("✓ No CPU tensor transfers (eliminates PCIe overhead)")
print("✓ No synchronization points (better pipelining)")
print("✓ Lower latency for inference serving")
print("✓ Maintains full compatibility with existing API")
print("=" * 70)
