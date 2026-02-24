# Memory, Barrier, and Async Lowering Reference

Detailed reference for memory operations, barrier lowering, async copy, warp pipeline/specialization, and TDM in the Triton AMD LLVM lowering.

Source: [`MemoryOpToLLVM.cpp`](https://github.com/triton-lang/triton/blob/main/third_party/amd/lib/TritonAMDGPUToLLVM/MemoryOpToLLVM.cpp), [`BarrierOpToLLVM.cpp`](https://github.com/triton-lang/triton/blob/main/third_party/amd/lib/TritonAMDGPUToLLVM/BarrierOpToLLVM.cpp), [`AsyncUtility.cpp`](https://github.com/triton-lang/triton/blob/main/third_party/amd/lib/TritonAMDGPUToLLVM/AsyncUtility.cpp), [`TDMUtility.cpp`](https://github.com/triton-lang/triton/blob/main/third_party/amd/lib/TritonAMDGPUToLLVM/TDMUtility.cpp), [`ConvertWarpPipeline.cpp`](https://github.com/triton-lang/triton/blob/main/third_party/amd/lib/TritonAMDGPUToLLVM/ConvertWarpPipeline.cpp), [`ConvertWarpSpecializeToLLVM.cpp`](https://github.com/triton-lang/triton/blob/main/third_party/amd/lib/TritonAMDGPUToLLVM/ConvertWarpSpecializeToLLVM.cpp)

## LDS Transpose Read (`TransLocalLoadOpConversion`)

Cooperative transposed loads from LDS for efficient MFMA/WMMA operand preparation.

### Supported Architectures

| Architecture | 16-bit (f16/bf16) | 8-bit (fp8/bf8) | 4-bit (fp4) |
|-------------|-------------------|------------------|-------------|
| CDNA4 (gfx950) | `ds_read_tr16_b64` | `ds_read_tr8_b64` | `ds_read_tr4_b64` |
| GFX1250 | `ds_load_tr16_b128` | `ds_load_tr8_b64` | `ds_load_tr4_b64` |

### How It Works

`ds_read_tr*` performs a cooperative transposed load across lanes in a shuffle group (typically 16 lanes):

```
Input tile in LDS (N×16):         Output per lane (after transpose):
K0  K1  ... K15                   R0  R1  R2  R3
M0[ ............... ]      =>     T0 [ .   .   .   . ]
M1[ ............... ]             T1 [ .   .   .   . ]
M2[ ............... ]             ...
M3[ ............... ]             T15[ .   .   .   . ]
```

Each lane loads 64 contiguous bits from LDS. After the transpose, lane i receives column i from the input tile.

### LDS Transpose Load Parameters

Queried via `targetInfo.queryLDSTransLoadParams(bitWidth)`:

| Parameter | Description |
|-----------|-------------|
| `numLanesInShuffleGroup` | Lanes cooperating (typically 16) |
| `instBitWidth` | Bits per lane per instruction (64 or 96) |
| `tileSize` | Contiguous elements needed in LDS |
| `needsDoubleB8Contiguity` | Whether B8 types need double contiguity |

### Lowering Flow

1. **Compute `cvtDstLL`**: `toLinearLayout(dstTy).invertAndCompose(sharedLL)` maps register positions to LDS offsets
2. **Verify tile divisibility**: `divideLeft(cvt, tile)` ensures the conversion divides cleanly by the transpose tile
3. **Compute repetitions**: `reps = zerosLike(tile) * quotient`
4. **Additive stride optimization**: `actionAdditiveStrides` identifies strides that can use `add` instead of `xor` for address computation
5. **Emit instructions**: for each tile repetition, compute LDS address and emit `ds_read_tr*`/`ds_load_tr*`
6. **Permute results**: apply inverse permutations from stride optimization

## Packed Transposed Local Load

`LocalLoadPackedTransposedOpConversion` handles `triton::amdgpu::LocalLoadPackedTransposedOp`:
- Specifically for FP4 (represented as i8) packed along K
- Uses `ds_read_tr4_b64` (CDNA4 only, not supported on GFX1250 for packed M/N)
- Computes a custom `dsReadTrLayout` via `chooseDsReadTrLayout` for the FP4 packing

## General BarrierOp Lowering

`BarrierOpConversion` in `MemoryOpToLLVM.cpp` lowers `triton::gpu::BarrierOp`:

```
BarrierOp(addrSpace) → MemoryCounterWaitOp + s_barrier
```

The wait op specification:
- `load` counter: set to 0 if `hasGlobalRead()`
- `store` counter: set to 0 if `hasGlobalWrite()`
- `ds` counter: set to 0 if `hasLocal()`

### MemoryCounterWaitOp Lowering

`MemoryCounterWaitOpConversion` lowers `amdgpu::MemoryCounterWaitOp`:

**GFX12+ (GFX1250)**:
```
ds present   → ROCDL::WaitDscntOp(value)
load present → ROCDL::WaitLoadcntOp(value)
store present → ROCDL::WaitStorecntOp(value)
```

**Pre-GFX12 (GFX9/10/11)**:
```
encode vmcnt + lgkmcnt → s_waitcnt <encoded_value>
```

Encoding bitpacking varies by ISA version:
- GFX9: `vmcnt[3:0]` | `(vmcnt[5:4] << 14)` | `(expcnt << 4)` | `(lgkmcnt << 8)`
- GFX10: same as GFX9 but `lgkmcnt` up to 63
- GFX11: `(vmcnt << 10)` | `expcnt` | `(lgkmcnt << 4)`

## MBarrier Operations (GFX1250)

LDS-based barriers using `ds_atomic_barrier_arrive_rtn_b64` intrinsics.

### InitBarrierOp

```
InitBarrierOp(alloc, count) →
  if (threadId == 0):
    val = ((count-1) << 32) | (count-1)
    store val to LDS barrier address
  s_barrier  // sync CTA
```

The count is decremented by 1 because phase changes on underflow (pending count becomes negative), not on reaching zero.

Barrier state is a 64-bit LDS value:
- Bits [28:0]: pending count
- Bits [31:29]: phase bits (only parity used, mask = 1)
- Bits [63:32]: initial count for reset

### ArriveBarrierOp

```
ArriveBarrierOp(alloc, count) →
  priorState = ds_atomic_barrier_arrive_rtn_b64(ldsAddr, count)
  priorPhase = (priorState >> 29) & 1
  return priorPhase
```

### WaitBarrierOp

Spin-wait loop:
```
WaitBarrierOp(alloc, phase) →
  loop:
    s_sleep 1          // sleep 64 clocks (SIMM16[6:0] = 1 → 64*1)
    curState = load i64 from LDS barrier
    curPhase = (curState >> 29) & 1
    if (curPhase != phase) break
```

## Cluster Barrier (GFX1250)

For multi-CTA synchronization within a cluster.

### ClusterBarrierArriveOp

```
ClusterBarrierArriveOp →
  if (warpId == 0):            // only first warp signals
    rocdl.barrier_signal(-3)   // barrier ID -3 = cluster barrier
```

### ClusterBarrierWaitOp

```
ClusterBarrierWaitOp →
  rocdl.barrier_wait(-3)
```

## Async Copy Utilities

### GFX9 Alias Annotation

`annotateLocalLoadsSyncedViaAsyncWait()` in `AsyncUtility.cpp`:

GFX9 architectures (CDNA3) require alias metadata to prevent LLVM from inserting conservative waits between direct-to-LDS writes and subsequent LDS reads. The function:

1. Walks the module for `LocalLoadOp` preceded by `AsyncWaitOp`
2. Annotates these loads with alias scope metadata
3. Ensures synchronization is handled explicitly via `ttg.async_wait` rather than LLVM wait counts

This is only needed when `targetInfo.requiresAliasInfoForAsyncOps()` returns true (GFX9 family).

### TargetInfo Async Capabilities

| Method | CDNA3 | CDNA4 | GFX1250 |
|--------|-------|-------|---------|
| `requiresAliasInfoForAsyncOps()` | yes | no | no |
| `supportsDirectToLDSScattering()` | no | yes | yes |
| `supportsBufferLoadToLocal()` | yes | yes | yes |
| `supportsDirectFromLdsStoreBitWidth()` | none | none | 32/64/128 |
| `supportsTDM()` | no | no | yes |

## Warp Pipeline (CDNA3/4)

`ConvertWarpPipeline.cpp` lowers warp pipeline stage annotations to `s_setprio` instructions:

```python
with warp_pipeline_stage("load", priority=3):
    # load ops
with warp_pipeline_stage("compute", priority=0):
    # MFMA ops
```

Lowered to:
```
s_setprio 3    // before load stage
<load ops>
s_setprio 0    // before compute stage
<compute ops>
```

Priority levels: 0 (lowest) to 3 (highest). Higher-priority wavefronts get scheduling preference.

## Warp Specialization (GFX1250)

`ConvertWarpSpecializeToLLVM.cpp` lowers warp specialization constructs:

Warps in a CTA are partitioned into groups, each running different code. Uses `warpId` to branch into the appropriate partition:

```
warp_specialize([
    (default_fn, args),      // default partition
    (partition1_fn, args),   // specialized partition 1
    (partition2_fn, args),   // specialized partition 2
], [num_warps_p1, num_warps_p2])
```

Lowered to conditional branches based on `warpId` ranges. Warp groups are isolated from above (`makeAllWarpGroupsIsolatedFromAbove`).

## TDM — Tensor Data Movement (GFX1250)

`TDMUtility.cpp` provides helpers for descriptor-based async memory operations:

- **Tensor descriptors**: encode base pointer, shape, strides, block shape
- **Async loads/stores**: descriptor-based, go through TDM engine
- **Async gather/scatter**: indexed variants for irregular access patterns
- **Wait**: `tdm.async_wait(num_outstanding)` for completion

TDM operations bypass the standard load/store path and use a dedicated hardware unit.

## Shared Memory Management

### Allocation

`ModuleAllocation` analysis (run before conversion) assigns LDS offsets to all shared memory allocations.

### Shared Memory Object

`LLVM::getSharedMemoryObjectFromStruct` extracts the base pointer and offset information from an LLVM struct representing a `MemDescType`:
- `getBase()`: pointer to the start of the allocation in LDS
- `getShmemOffset()`: affine offset for the current slice (from `extract_slice` operations)
- `getMaskSpanOffsets()`: span offset for padding-aware addressing

### Global Shared Memory Symbol

The pass creates `global_smem` — an external LLVM global with:
- Type: `[0 x i8]` (zero-size, dynamic allocation)
- Address space: 3 (shared/LDS)
- Alignment: 16 bytes
- Linkage: external (size determined at kernel launch)

## SPMD Operations

`SPMDOpToLLVM.cpp` lowers:

| Op | Lowering |
|----|----------|
| `GetNumProgramsOp` | `gpu::GridDimOp` → grid dimension query |
| `CondBarrierOp` | Conditional `s_barrier` (only threads with `pred=true` execute the barrier) |

`WarpIdOpToLLVM.cpp` provides warp ID computation:
- Standard: `threadId / warpSize`
- GFX1250: can use dedicated wave ID hardware query (`supportsWaveId()`)

## Elementwise Op Lowering

`ElementwiseOpToLLVM.cpp` handles AMD-specific elementwise operations:

Key patterns:
- **FP8/BF8 conversions**: `cvt_pk_fp8_f32`, `cvt_pk_bf8_f32`, and reverse (CDNA4/GFX1250)
- **FP4 conversions**: specialized packing/unpacking (via `Fp4ToFpOpToLLVM.cpp`)
- **FTZ (flush-to-zero)**: controlled by pass parameter, affects denormal handling
- **Packed operations**: 2×f16 packed arithmetic when `supportBitwidth16Elementwise()` returns true
- **MXFP upcast**: `UpcastMXFPToLLVM.cpp` handles microscaling FP format conversion
