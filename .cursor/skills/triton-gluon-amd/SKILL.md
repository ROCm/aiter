---
name: triton-gluon-amd
description: Write Triton Gluon kernels for AMD GPUs (CDNA3/gfx942, CDNA4/gfx950, GFX1250). Covers layouts (AMDMFMALayout, AMDWMMALayout, BlockedLayout, PaddedSharedLayout), async operations (TDM, async_copy, mbarrier), load/store (buffer_load/store), MFMA/WMMA matrix core instructions, warp specialization, and warp pipelines. Use when writing or debugging Gluon kernels targeting AMD hardware, or when the user mentions gluon, MFMA, WMMA, TDM, cdna3, cdna4, gfx1250, or AMD matrix cores.
---

# Triton Gluon for AMD GPUs

Gluon is a low-level IR and Python frontend in `triton.experimental.gluon` for writing GPU kernels with explicit control over layouts, memory, and compute instructions.

Repo: [triton-lang/triton @ 2b1031d](https://github.com/triton-lang/triton/tree/2b1031d66167eb7f0b55d464935b69aa76b3aff4)
Source: [`python/triton/experimental/gluon/`](https://github.com/triton-lang/triton/tree/2b1031d66167eb7f0b55d464935b69aa76b3aff4/python/triton/experimental/gluon)

## Architecture Mapping

| Name | Arch | MFMA/WMMA | Warp Size | Layout Class | Version |
|------|------|-----------|-----------|--------------|---------|
| CDNA3 | gfx942 | MFMA | 64 | `AMDMFMALayout` | 3 |
| CDNA4 | gfx950 | MFMA | 64 | `AMDMFMALayout` | 4 |
| GFX1250 | gfx1250 | WMMA | 32 | `AMDWMMALayout` | 3 |

## Import Pattern

```python
from triton.experimental import gluon
import triton.experimental.gluon.language as ttgl

# AMD-specific
# GFX1250: ttgl.amd.gfx1250.*
# CDNA4:   ttgl.amd.cdna4.*
# CDNA3:   ttgl.amd.cdna3.* (also imported by cdna4)
```

## Quick Reference: Layouts

### Distributed Layouts (register-level)

| Layout | Usage |
|--------|-------|
| `BlockedLayout(spt, tpw, wpc, order)` | General-purpose load/store layout |
| `AMDMFMALayout(ver, shape, trans, wpc)` | MFMA accumulator (CDNA3/4) |
| `AMDWMMALayout(ver, trans, warp_bases)` | WMMA accumulator (GFX1250) |
| `DotOperandLayout(idx, parent, k_width)` | Dot operand (idx: 0=LHS, 1=RHS) |
| `SliceLayout(dim, parent)` | Slice of parent layout |
| `AutoLayout()` | Compiler-inferred layout |
| `CoalescedLayout()` | Compiler-chosen coalesced layout |

### Shared Memory Layouts

| Layout | Usage |
|--------|-------|
| `PaddedSharedLayout(pairs, bases, cga, shape)` | Padded shared mem (AMD preferred) |
| `PaddedSharedLayout.with_identity_for(pairs, shape, order)` | Convenience constructor |
| `SwizzledSharedLayout(vec, per_phase, max_phase, order)` | Swizzled shared mem |
| `SharedLinearLayout(offset_bases, block_bases)` | Explicit linear layout |

## Quick Reference: Operations by Architecture

### CDNA3 (gfx942)

```python
ttgl.amd.cdna3.buffer_load(ptr, offsets, mask, other, cache)
ttgl.amd.cdna3.buffer_store(value, ptr, offsets, mask, cache)
ttgl.amd.cdna3.mfma(a, b, acc)
ttgl.amd.cdna3.buffer_atomic_{max,min,add,and,or,xor,xchg}(ptr, offsets, value, mask, sem, scope)
```

### CDNA4 (gfx950) — inherits CDNA3 + adds:

```python
ttgl.amd.cdna4.mfma_scaled(a, a_scale, a_format, b, b_scale, b_format, acc)
ttgl.amd.cdna4.get_mfma_scale_layout(dot_operand_layout, shape)
ttgl.amd.cdna4.async_copy.global_load_to_shared(dest, ptr, mask, other)
ttgl.amd.cdna4.async_copy.buffer_load_to_shared(dest, ptr, offsets, mask, other)
ttgl.amd.cdna4.async_copy.commit_group()
ttgl.amd.cdna4.async_copy.wait_group(num_outstanding)
ttgl.amd.cdna4.async_copy.load_shared_relaxed(smem, layout)
```

### GFX1250

```python
# WMMA
ttgl.amd.gfx1250.wmma(a, b, acc)
ttgl.amd.gfx1250.wmma_scaled(a, a_scale, a_format, b, b_scale, b_format, acc)
ttgl.amd.gfx1250.get_wmma_scale_layout(dot_operand_layout, shape)

# TDM (Tensor Data Movement) — descriptor-based async
ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base, shape, strides, block_shape, layout)
ttgl.amd.gfx1250.tdm.async_load(src_desc, offsets, dest_smem, pred, mbarrier)
ttgl.amd.gfx1250.tdm.async_store(dest_desc, offsets, src_smem, mbarrier)
ttgl.amd.gfx1250.tdm.async_scatter(desc, row_indices, col_offset, src_smem, mbarrier)
ttgl.amd.gfx1250.tdm.async_gather(desc, row_indices, col_offset, dst_smem, mbarrier)
ttgl.amd.gfx1250.tdm.async_wait(num_outstanding)
ttgl.amd.gfx1250.tdm.prefetch(src_desc, offsets, pred, speculative)

# Pointer-based async copy
ttgl.amd.gfx1250.async_copy.global_to_shared(smem, pointer, mask, other)
ttgl.amd.gfx1250.async_copy.shared_to_global(pointer, smem, mask)
ttgl.amd.gfx1250.async_copy.commit_group()
ttgl.amd.gfx1250.async_copy.wait_group(num_outstanding)
ttgl.amd.gfx1250.async_copy.mbarrier_arrive(mbarrier)

# MBarrier (LDS barriers)
ttgl.amd.gfx1250.mbarrier.init(mbarrier, count)
ttgl.amd.gfx1250.mbarrier.wait(mbarrier, phase)
ttgl.amd.gfx1250.mbarrier.arrive(mbarrier, count=1)

# Cluster synchronization
ttgl.amd.gfx1250.cluster.arrive()
ttgl.amd.gfx1250.cluster.wait()

# Buffer ops (inherited from cdna3)
ttgl.amd.gfx1250.buffer_load(ptr, offsets, mask, other, cache)
ttgl.amd.gfx1250.buffer_store(value, ptr, offsets, mask, cache)
```

## Warp Pipeline (CDNA3/4)

```python
from triton.experimental.gluon.language.amd.warp_pipeline import warp_pipeline_stage

with amd.warp_pipeline_stage("load", priority=3):
    # load ops
with amd.warp_pipeline_stage("compute", priority=0):
    # MFMA ops
```

Priority: 0 (lowest) to 3 (highest). Lowers to `s_setprio`.

## Warp Specialization (GFX1250)

```python
ttgl.warp_specialize([
    (default_partition_fn, (args,)),      # default partition
    (partition1_fn, (args,)),             # specialized partition
    (partition2_fn, (args,)),             # specialized partition
], [num_warps_partition1, num_warps_partition2])
```

## Shared Memory

```python
# Allocate
buf = ttgl.allocate_shared_memory(dtype, shape, layout)
# Multi-buffer
buf = ttgl.allocate_shared_memory(dtype, [NUM_BUFFERS] + block_shape, layout)
# Index into buffer
tile = buf.index(i)
# Load from shared to registers
tensor = tile.load(layout=OPERAND_LAYOUT)
# Store registers to shared
tile.store(tensor)
# Permute (e.g. for transposed B)
transposed = tile.permute([1, 0])
```

## Additional Resources

- For detailed layout reference, see [layouts-reference.md](layouts-reference.md)
- For async operations and memory reference, see [async-ops-reference.md](async-ops-reference.md)
- For MFMA/WMMA matrix core reference, see [mfma-wmma-reference.md](mfma-wmma-reference.md)
- For complete GEMM examples, see [examples-reference.md](examples-reference.md)
