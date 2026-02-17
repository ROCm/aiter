# Coalescing Reference (AMD)

## Overview

The coalescing pass (`tritongpu-coalesce`) runs early in the pipeline to optimize memory access patterns. It replaces the default blocked layout on each memory op with a layout that maximizes vectorized, coalesced access.

Source: [`lib/Dialect/TritonGPU/Transforms/Coalesce.cpp`](https://github.com/triton-lang/triton/blob/2b1031d66167eb7f0b55d464935b69aa76b3aff4/lib/Dialect/TritonGPU/Transforms/Coalesce.cpp), [`CoalesceUtils.cpp`](https://github.com/triton-lang/triton/blob/2b1031d66167eb7f0b55d464935b69aa76b3aff4/lib/Dialect/TritonGPU/Transforms/CoalesceUtils.cpp)

## AxisInfo Analysis

`AxisInfo` tracks three properties per dimension for each tensor value:

| Property | Meaning |
|----------|---------|
| `contiguity[d]` | Number of contiguous elements along dim `d` |
| `divisibility[d]` | Pointer alignment in bytes along dim `d` |
| `constancy[d]` | Number of elements with constant value along dim `d` |

These are computed by `ModuleAxisInfoAnalysis` and propagated through the IR.

## `buildCoalescedEncoding` Algorithm

```
Input: memory op, AxisInfo, numWarps, threadsPerWarp, shapePerCTA
Output: BlockedEncodingAttr

1. Get contiguity of the pointer operand
2. order = getOrderFromContiguity(contiguity)
   // sorts dimensions by descending contiguity → fastest dim first
3. Compute perThread = getNumElementsPerThread(op, order, axisInfo, shape):
   a. maxMultipleBytes = divisibility[order[0]]
   b. maxMultiple = maxMultipleBytes / elemNumBytes
   c. maxContig = min(contiguity[order[0]], shape[order[0]])
   d. alignment = min(maxMultiple, maxContig)
   e. perThread = min(alignment, 128 / elemNumBits)  // cap at 128-bit vector
4. Multi-root: for all same-order memory accesses in the slice, take max(perThread)
5. Cap: perThread = min(perThread, numElems / numThreads)
6. For stores: additionally cap perThread to avoid warp-level gaps
7. sizePerThread[order[0]] = perThread; all other dims = 1
8. Return BlockedEncodingAttr::get(shape, sizePerThread, order, numWarps, ...)
```

## Descriptor Load/Store Heuristic

For TMA/descriptor-based ops (`DescriptorOpInterface`), a separate simpler heuristic:

```
vectorSize = min(numElems / numThreads, 128 / elemBitWidth)
sizePerThread = [1, ..., 1, vectorSize]  // vectorize last dim
order = row-major
```

## Key Properties

- **Order matters most**: the fastest-varying dimension determines memory access pattern
- **128-bit cap**: per-thread elements never exceed 128 bits (16 bytes) — the widest vector load
- **Store strictness**: stores don't get the load benefit of L1 cache hiding gaps
- **Multi-root sharing**: all memory ops with the same shape and access order share the same coalesced layout, taking the best alignment across all of them

## AMD-Specific: Direct-to-LDS Async Copy Coalescing

For CDNA3/4, `AsyncCopyGlobalToLocalOp` (buffer_load to shared) requires coalesced writes to LDS. The `tritonamdgpu-coalesce-async-copy` pass:

1. Computes contiguity from `AxisInfo` (including mask alignment)
2. Restricts by register-to-shared layout mapping: `regLayout.invertAndCompose(sharedLayout).getNumConsecutiveInOut()`
3. Fits to valid direct-to-LDS vector sizes via `fitToValidDirectToLdsVecSize`
4. For swizzled shared: adjusts `sizePerThread` of the blocked encoding
5. For padded shared: constructs a `LinearEncodingAttr` from the shared memory `linear_component` offset bases, assigning them to register/lane/warp bases to ensure consecutive LDS offsets

This is CDNA3/4-specific because GFX9 `buffer_load` directly to LDS has strict alignment requirements.
