---
name: triton-amd-layout-heuristics
description: Understand and debug Triton compiler layout heuristics for AMD GPUs (CDNA1-4, RDNA3/4, GFX1250). Covers the layout decision pipeline from default blocked encoding through coalescing, MFMA/WMMA acceleration, layout propagation, and async copy coalescing. Use when analyzing Triton IR layouts on AMD, debugging layout conversions, understanding why a particular encoding was chosen, optimizing memory coalescing for AMD, or when the user mentions AMD layout heuristics, coalescing, AccelerateMatmul, RemoveLayoutConversions, warpsPerTile, kPack, tilesPerWarp, or isTransposed.
---

# Triton AMD Layout Heuristics

How the Triton compiler decides tensor layouts for AMD GPUs. Layouts flow through a multi-phase pipeline, each pass making increasingly specialized decisions.

Repo: [triton-lang/triton @ 2b1031d](https://github.com/triton-lang/triton/tree/2b1031d66167eb7f0b55d464935b69aa76b3aff4)

## Layout Decision Pipeline (AMD)

```
Phase 1: TritonToTritonGPU        → default BlockedEncoding (sizePerThread=1, row-major order)
Phase 2: tritongpu-coalesce       → AxisInfo-driven coalesced BlockedEncoding for memory ops
Phase 3: tritongpu-remove-layout  → forward propagation from anchors, conflict resolution
Phase 4: tritonamdgpu-accelerate  → blocked → AMDMfmaEncoding or AMDWmmaEncoding for dots
Phase 5: tritongpu-remove-layout  → cleanup after matmul acceleration
Phase 6: AMD-specific passes      → optimize-epilogue, optimize-dot-operands, hoist/sink
Phase 7: pipelining + async copy  → coalesce-async-copy (CDNA3/4 direct-to-LDS)
Phase 8: tritongpu-remove-layout  → final cleanup
```

## Key Files

| File | Role |
|------|------|
| [`lib/Dialect/TritonGPU/IR/Dialect.cpp`](https://github.com/triton-lang/triton/blob/2b1031d66167eb7f0b55d464935b69aa76b3aff4/lib/Dialect/TritonGPU/IR/Dialect.cpp) | `getDefaultBlockedEncoding` |
| [`lib/Dialect/TritonGPU/Transforms/Coalesce.cpp`](https://github.com/triton-lang/triton/blob/2b1031d66167eb7f0b55d464935b69aa76b3aff4/lib/Dialect/TritonGPU/Transforms/Coalesce.cpp) | Coalescing pass |
| [`lib/Dialect/TritonGPU/Transforms/CoalesceUtils.cpp`](https://github.com/triton-lang/triton/blob/2b1031d66167eb7f0b55d464935b69aa76b3aff4/lib/Dialect/TritonGPU/Transforms/CoalesceUtils.cpp) | `buildCoalescedEncoding` |
| [`lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp`](https://github.com/triton-lang/triton/blob/2b1031d66167eb7f0b55d464935b69aa76b3aff4/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp) | Propagation + remat |
| [`third_party/amd/lib/TritonAMDGPUTransforms/AccelerateAMDMatmul.cpp`](https://github.com/triton-lang/triton/blob/2b1031d66167eb7f0b55d464935b69aa76b3aff4/third_party/amd/lib/TritonAMDGPUTransforms/AccelerateAMDMatmul.cpp) | MFMA/WMMA selection |
| [`third_party/amd/lib/TritonAMDGPUTransforms/CoalesceAsyncCopy.cpp`](https://github.com/triton-lang/triton/blob/2b1031d66167eb7f0b55d464935b69aa76b3aff4/third_party/amd/lib/TritonAMDGPUTransforms/CoalesceAsyncCopy.cpp) | Direct-to-LDS coalescing |
| [`third_party/amd/lib/TritonAMDGPUTransforms/HoistLayoutConversions.cpp`](https://github.com/triton-lang/triton/blob/2b1031d66167eb7f0b55d464935b69aa76b3aff4/third_party/amd/lib/TritonAMDGPUTransforms/HoistLayoutConversions.cpp) | Hoist dot operand cvts |
| [`third_party/amd/lib/TritonAMDGPUTransforms/SinkLayoutConversions.cpp`](https://github.com/triton-lang/triton/blob/2b1031d66167eb7f0b55d464935b69aa76b3aff4/third_party/amd/lib/TritonAMDGPUTransforms/SinkLayoutConversions.cpp) | Sink cvts past deallocs |
| [`third_party/amd/backend/compiler.py`](https://github.com/triton-lang/triton/blob/2b1031d66167eb7f0b55d464935b69aa76b3aff4/third_party/amd/backend/compiler.py) | Pass pipeline order |

## Phase 1: Default Blocked Layout

Every tensor without an encoding gets `BlockedEncodingAttr` with `sizePerThread=[1,1,...]` and row-major order. The `BlockedEncodingAttr::get` constructor distributes threads/warps/CTAs.

**Special case for DotOp**: gets `sizePerThread` up to `[4,4]` based on elements-per-thread ratio.

## Phase 2: Coalescing

For each memory op (load/store/atomic), uses `AxisInfo` analysis:

1. **Order from contiguity**: `getOrderFromContiguity` sorts dims by descending contiguity
2. **Per-thread elements**: `min(alignment, 128bits / elemBits)`, capped at 16B vectors
3. **Multi-root optimization**: takes max `perThread` across same-order memory accesses in the slice
4. **Store restriction**: stores cap `perThread` to avoid warp-level gaps (no L1 cache)

## Phase 4: AMD Matmul Acceleration

### ISA Family → Pattern Selection

| ISA Family | Patterns (by priority) |
|------------|----------------------|
| CDNA4 | `ScaledBlockedToScaledMFMAF8F6F4(4)` → `DecomposeAMDScaledBlocked(3)` → `BlockedToMFMA(2)` + `ScaledBlockedToMFMA(2)` |
| CDNA1-3 | `BlockedToMFMA(2)` + `ScaledBlockedToMFMA(2)` |
| GFX1250 | `ScaledBlockedToScaledWMMAF8F6F4(4)` → `DecomposeAMDScaledBlocked(3)` → `BlockedToWMMA(2)` |
| RDNA3/4 | `DecomposeScaledBlocked(3)` → `BlockedToWMMA(2)` |

After matrix core patterns, `AccelerateBlocked(1)` handles FMA fallback (V_DOT on supported archs).

### MFMA Tile Selection (`chooseMfmaInstruction`)

- `min(M,N) >= 32` → 32x32 tile (16x16 for f64)
- `min(M,N) >= 16` → 16x16 tile
- `min(M,N) >= 4` → 64x4 or 4x64 tile
- User override via `matrix_instr_nonkdim`

### Warp Distribution (`warpsPerTile`)

- **Batched matmul**: `{numWarps, 1, 1}`
- **Chain-dot head** (1st dot in FA): `{numWarps, 1}` — eliminates inter-warp softmax reduction
- **Chain-dot tail** (2nd dot in FA): fill M first up to `ceil(shape[0]/mDim)`, remainder to N
- **Regular**: balance M and N tiles, biased toward the dimension with more remaining tiles

### Key AMD-Specific Layout Knobs

- **`isTransposed`**: almost always `true` (better global store vectorization); `false` only for 4x64 MFMA
- **`kPack`**: multiplies `kWidth` (1 or 2); chain-dot tail restricted to `kPack=1`
- **`tilesPerWarp`**: CDNA4 16x16 chain-dot: `{2,1}` or `{1,2}` for intra-warp conversion
- **`kWidth`**: chain-dot tail with f16 forced to 4 for no-op mma→dotOp conversion

## Phase 3/5: Layout Propagation

**Anchors** (layouts preserved): expensive loads/stores, dots, atomics, TMA ops, permuting reshapes.

**Conflict resolution**: loads/stores prefer `BlockedEncoding`; compute ops prefer `MmaEncoding`.

**Backward rematerialization**: cost model compares shared-memory round-trip cost (`32 * bytes`) vs recomputation cost (arithmetic + loads + reduces).

## AMD-Specific Passes

- **HoistLayoutConversions**: hoists `DotOperandEncoding` converts out of loops when source is loop-invariant (Q tensor in flash attention)
- **SinkLayoutConversions**: sinks converts past `LocalDeallocOp` to free LDS earlier
- **CoalesceAsyncCopy** (CDNA3/4): ensures direct-to-LDS writes are coalesced by adjusting `sizePerThread` or constructing `LinearEncodingAttr` matching the shared memory linear component

## Additional Resources

- For coalescing details, see [coalescing-reference.md](coalescing-reference.md)
- For MFMA/WMMA acceleration details, see [matmul-acceleration-reference.md](matmul-acceleration-reference.md)
- For layout propagation details, see [layout-propagation-reference.md](layout-propagation-reference.md)
- For full AMD pass pipeline, see [pipeline-reference.md](pipeline-reference.md)
