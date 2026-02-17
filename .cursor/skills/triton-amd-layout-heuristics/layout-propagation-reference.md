# Layout Propagation & Removal Reference

## Overview

The `tritongpu-remove-layout-conversions` pass is the primary layout optimizer. It runs multiple times in the pipeline to eliminate unnecessary `ConvertLayoutOp`s. The algorithm has four sub-phases.

Source: [`lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp`](https://github.com/triton-lang/triton/blob/2b1031d66167eb7f0b55d464935b69aa76b3aff4/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp)

## Sub-Phase 1: Forward Propagation from Anchors

### What is an anchor?

An anchor is an op whose layout should be preserved. The pass propagates layouts forward from anchors.

```
Anchors:
- DescriptorOpInterface (TMA loads/stores)
- LoadOp, StoreOp (only if "expensive" — i.e. not scalar/small)
- DotOp, DotScaledOp, WarpGroupDotOp
- AtomicRMWOp, AtomicCASOp
- TMEMLoadOp
- GatherOp (if efficientLayout is set)
- ReshapeOp (if allowReorder — permuting reshape)
- Function arguments
```

### Propagation rules

Layouts propagate through:
- Elementwise ops, `SameOperandsAndResultEncoding` ops
- `ReduceOp`, `ExpandDimsOp`, `ReshapeOp`, `TransOp`, `JoinOp`, `SplitOp`
- `ConvertLayoutOp` (tries to eliminate by matching src encoding)
- `scf.for`, `scf.while`, `scf.if` (through iter args, yields, conditions)

Propagation uses `inferDstEncoding(op, srcEncoding)` to compute the encoding of results given an input encoding.

### Data structure

Each value maps to a `LayoutInfo` containing a set of candidate encodings. The propagation is a worklist algorithm that iterates until fixpoint.

## Sub-Phase 2: Conflict Resolution

When a value has multiple candidate encodings from different anchors:

```cpp
// Heuristic:
// - Load/store/atomic ops prefer BlockedEncoding (for coalescing)
// - All other ops prefer MmaEncoding (for compute)
for (Attribute e : info.encodings) {
    if ((isLoadOrStore && isa<BlockedEncodingAttr>(e)) ||
        (!isLoadOrStore && isa<MmaEncodingTrait>(e))) {
        encoding = e;
        break;
    }
}
```

This is acknowledged as "hacky" in the source code. The tension between coalesced memory layout and MMA compute layout is the central optimization problem.

## Sub-Phase 3: Backward Rematerialization

For remaining `ConvertLayoutOp`s, tries to rematerialize (re-compute) the producer chain in the target layout, eliminating the conversion.

### Cost model

```
convertLayoutCost = 32 * getByteCount(src, minElems=32, minBits=32)
// Pessimistic: assumes shared memory round-trip (store + load + sync overhead)

rematerializationCost = sum of:
  - arith.constant: 0
  - LoadOp/LocalLoadOp: 8 * byteCount (L1 optimistic)
  - Cheap math (add, mul, etc.): 1 * byteCount
  - Expensive math (div, exp, sin, sqrt, etc.): 8 * byteCount
  - ReduceOp: intraWarpSize + 8 * interWarpSize
  + cost of any new ConvertLayoutOps introduced

Remat applied when: convertLayoutCost >= rematerializationCost
```

### What can be rematerialized?

```
canBeRemat(op):
  - NOT: expensive loads/stores, atomics, DotOp, WhileOp, ConditionOp
  - NOT: GatherOp with efficientLayout
  - YES: everything else (cheap loads, elementwise, views, etc.)
```

The backward slice stops when it hits an op that cannot be rematerialized.

### Iterative application

Backward remat runs in a loop until no more conversions are removed, with canonicalization cleanup between iterations.

## Sub-Phase 4: Hoisting Converts

Three hoisting strategies, applied in order:

### 4a. Hoist over type extensions/broadcasts

If a `ConvertLayoutOp` follows an `ExtFOp`/`ExtSIOp`/`BroadcastOp`/`ExpandDimsOp`, the convert is moved before the extension so it operates on smaller data.

Cost check: `convertLayoutCost(original) >= rematerializationCost(new) + newCvtCost`

### 4b. Hoist into conditionals

For `ConvertLayoutOp` after `scf.if`:
- Takes backward slice, stopping at `scf.if` ops
- For each conditional, tries to propagate remat through both branches
- If one branch succeeds and the other doesn't, hoists the convert into the failing branch
- Only hoists inside loops (assumes if-inside-loop executes fewer iterations)

### 4c. Hoist DotOperand converts to loads

For `ConvertLayoutOp` targeting `DotOperandEncoding` inside loops:
- Moves the convert next to the load that produces the data
- This enables the software pipeliner to pipeline the load+convert together
- Only applies if there's a dot op in the loop that post-dominates the convert

## AMD-Specific Layout Passes

### HoistLayoutConversions (`tritonamdgpu-hoist-layout-conversions`)

Targeted at flash attention: hoists `ConvertLayoutOp` with `DotOperandEncoding` destination **out of loops** when the source is defined outside the loop.

Use case: keeps Q tensor in registers across loop iterations instead of reloading.

```
Before: for i in range(N): cvt = convert_layout(Q, #dot_op); dot(cvt, K, acc)
After:  cvt = convert_layout(Q, #dot_op); for i in range(N): dot(cvt, K, acc)
```

### SinkLayoutConversions (`tritonamdgpu-sink-layout-conversions`)

Sinks `ConvertLayoutOp` past `LocalDeallocOp` (but before first use) to free shared memory earlier. Reduces peak LDS usage.

### OptimizeDotOperands (`tritonamdgpu-optimize-dot-operands`)

For CDNA4: allocates LDS for scale operands used in `ScaledUpcastFp4Op`/`ScaledUpcastFp8Op`. Inserts `LocalAlloc` + `LocalLoad` between the global load and the upcast op to enable pipelining.
