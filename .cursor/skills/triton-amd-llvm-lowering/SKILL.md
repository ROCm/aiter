---
name: triton-amd-llvm-lowering
description: Understand and debug the Triton AMD GPU to LLVM lowering pass for AMD GPUs (CDNA1-4, RDNA3/4, GFX1250). Covers the ConvertTritonAMDGPUToLLVM pass including load/store lowering (global, buffer, direct-to-LDS), dot operation lowering (MFMA, WMMA, FMA), layout conversions, elementwise ops, barrier/async lowering, buffer resource descriptors, warp pipeline/specialization, and TDM. Use when analyzing how Triton IR becomes LLVM IR for AMD, debugging lowering failures, understanding buffer ops emission, investigating load vectorization, or when the user mentions TritonAMDGPUToLLVM, BufferEmitter, direct-to-LDS, buffer_load, LoadStoreOpToLLVM, DotOpToLLVM, ConvertLayoutOp, TargetInfo, or AMD lowering patterns.
---

# Triton AMD GPU to LLVM Lowering

How the Triton compiler lowers TritonGPU dialect ops to LLVM dialect IR for AMD GPUs. Uses MLIR's dialect conversion framework with pattern-based rewriting.

Repo: [triton-lang/triton](https://github.com/triton-lang/triton/tree/main/third_party/amd/lib/TritonAMDGPUToLLVM)
Common (shared with NVIDIA): [`lib/Conversion/TritonGPUToLLVM/`](https://github.com/triton-lang/triton/tree/main/lib/Conversion/TritonGPUToLLVM)

## Pass Overview

The `ConvertTritonAMDGPUToLLVM` pass (`TritonGPUToLLVM.cpp`) runs in these stages:

```
1. TargetInfo construction      → ISA family detection from arch string (gfx942, gfx950, gfx1250, ...)
2. Type converter setup         → TritonAMDGPUToLLVMTypeConverter (index width = 32)
3. Shared memory allocation     → ModuleAllocation analysis, async alias annotation
4. Memory barrier analysis      → ModuleMembarAnalysis with AMD-specific membar filter
5. Function lowering            → FuncOp → LLVM func with ABI, kernel metadata
6. Shared memory init           → global_smem external symbol (dynamic LDS, 16B aligned)
7. Pattern-based conversion     → All TritonGPU/TritonAMDGPU ops → LLVM+ROCDL
8. Post-processing              → Mode register adjustment, loop annotation fix, warp group isolation
```

## Pattern Priority

AMD-specific patterns get higher benefit than common patterns to ensure they match first:

| Benefit | Patterns |
|---------|----------|
| `commonBenefit + 1` (AMD) | ConvertLayout, DotOp, Elementwise, LoadStore, Memory, Barrier, SPMD, TensorPtr, UpcastMXFP, Fp4ToFp |
| `commonBenefit` (shared) | ConvertLayout (common), Reduce, Scan, View, Histogram, Gather, MakeRange, Assert, ControlFlow, Print |

## Key Files

| File | Role |
|------|------|
| [`TritonGPUToLLVM.cpp`](https://github.com/triton-lang/triton/blob/main/third_party/amd/lib/TritonAMDGPUToLLVM/TritonGPUToLLVM.cpp) | Pass entry point, pattern registration, shared memory init |
| [`LoadStoreOpToLLVM.cpp`](https://github.com/triton-lang/triton/blob/main/third_party/amd/lib/TritonAMDGPUToLLVM/LoadStoreOpToLLVM.cpp) | Global/buffer loads and stores, atomics, direct-to-LDS |
| [`DotOpToLLVM.cpp`](https://github.com/triton-lang/triton/blob/main/third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM.cpp) | Dot op dispatch (MFMA/WMMA/FMA) |
| [`DotOpToLLVM/MFMA.cpp`](https://github.com/triton-lang/triton/blob/main/third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/MFMA.cpp) | MFMA instruction emission (CDNA1-4) |
| [`DotOpToLLVM/WMMA.cpp`](https://github.com/triton-lang/triton/blob/main/third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/WMMA.cpp) | WMMA instruction emission (GFX1250) |
| [`DotOpToLLVM/FMA.cpp`](https://github.com/triton-lang/triton/blob/main/third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/FMA.cpp) | FMA fallback (BlockedEncoding dots) |
| [`ConvertLayoutOpToLLVM.cpp`](https://github.com/triton-lang/triton/blob/main/third_party/amd/lib/TritonAMDGPUToLLVM/ConvertLayoutOpToLLVM.cpp) | Register↔shared memory layout conversions |
| [`ElementwiseOpToLLVM.cpp`](https://github.com/triton-lang/triton/blob/main/third_party/amd/lib/TritonAMDGPUToLLVM/ElementwiseOpToLLVM.cpp) | Type conversions, math ops, FP8/BF8 casts |
| [`MemoryOpToLLVM.cpp`](https://github.com/triton-lang/triton/blob/main/third_party/amd/lib/TritonAMDGPUToLLVM/MemoryOpToLLVM.cpp) | LocalLoad (LDS transpose reads), BarrierOp, MemoryCounterWait |
| [`BufferOpsEmitter.cpp`](https://github.com/triton-lang/triton/blob/main/third_party/amd/lib/TritonAMDGPUToLLVM/BufferOpsEmitter.cpp) | Buffer resource descriptor creation, buffer load/store/atomic emission |
| [`AtomicRMWOpsEmitter.cpp`](https://github.com/triton-lang/triton/blob/main/third_party/amd/lib/TritonAMDGPUToLLVM/AtomicRMWOpsEmitter.cpp) | Atomic RMW lowering (global and buffer) |
| [`BarrierOpToLLVM.cpp`](https://github.com/triton-lang/triton/blob/main/third_party/amd/lib/TritonAMDGPUToLLVM/BarrierOpToLLVM.cpp) | MBarrier init/arrive/wait, cluster barriers |
| [`SPMDOpToLLVM.cpp`](https://github.com/triton-lang/triton/blob/main/third_party/amd/lib/TritonAMDGPUToLLVM/SPMDOpToLLVM.cpp) | GetNumPrograms, conditional barrier |
| [`AsyncUtility.cpp`](https://github.com/triton-lang/triton/blob/main/third_party/amd/lib/TritonAMDGPUToLLVM/AsyncUtility.cpp) | Async copy alias annotation for GFX9 |
| [`TDMUtility.cpp`](https://github.com/triton-lang/triton/blob/main/third_party/amd/lib/TritonAMDGPUToLLVM/TDMUtility.cpp) | Tensor Data Movement (GFX1250) |
| [`ConvertWarpPipeline.cpp`](https://github.com/triton-lang/triton/blob/main/third_party/amd/lib/TritonAMDGPUToLLVM/ConvertWarpPipeline.cpp) | Warp pipeline lowering (s_setprio) |
| [`ConvertWarpSpecializeToLLVM.cpp`](https://github.com/triton-lang/triton/blob/main/third_party/amd/lib/TritonAMDGPUToLLVM/ConvertWarpSpecializeToLLVM.cpp) | Warp specialization lowering (GFX1250) |
| [`UpcastMXFPToLLVM.cpp`](https://github.com/triton-lang/triton/blob/main/third_party/amd/lib/TritonAMDGPUToLLVM/UpcastMXFPToLLVM.cpp) | MXFP upcast (microscaling FP) |
| [`Fp4ToFpOpToLLVM.cpp`](https://github.com/triton-lang/triton/blob/main/third_party/amd/lib/TritonAMDGPUToLLVM/Fp4ToFpOpToLLVM.cpp) | FP4 → FP conversion |
| [`TargetInfo.cpp`](https://github.com/triton-lang/triton/blob/main/third_party/amd/lib/TritonAMDGPUToLLVM/TargetInfo.cpp) | Architecture-specific queries (warp size, LDS size, shuffles, etc.) |
| [`TargetUtils.cpp`](https://github.com/triton-lang/triton/blob/main/third_party/amd/lib/TritonAMDGPUToLLVM/TargetUtils.cpp) | ISA family deduction from arch string |
| [`Utility.cpp`](https://github.com/triton-lang/triton/blob/main/third_party/amd/lib/TritonAMDGPUToLLVM/Utility.cpp) | AMD-specific LLVM emission helpers (llLoad, llStore, vector size) |
| [`GCNAsmFormat.cpp`](https://github.com/triton-lang/triton/blob/main/third_party/amd/lib/TritonAMDGPUToLLVM/GCNAsmFormat.cpp) | GCN inline assembly formatting |
| [`SchedInstructions.cpp`](https://github.com/triton-lang/triton/blob/main/third_party/amd/lib/TritonAMDGPUToLLVM/SchedInstructions.cpp) | Scheduling hint insertion |
| [`ScalarizePackedFOps.cpp`](https://github.com/triton-lang/triton/blob/main/third_party/amd/lib/TritonAMDGPUToLLVM/ScalarizePackedFOps.cpp) | Scalarize packed float ops when needed |

## ISA Family Detection

`TargetInfo` wraps the arch string and provides architecture-specific queries. `deduceISAFamily()` maps arch → `ISAFamily` enum:

| ISAFamily | Arch | Warp Size | Matrix Core | Key Capabilities |
|-----------|------|-----------|-------------|-----------------|
| CDNA1 | gfx908 | 64 | MFMA | Basic MFMA |
| CDNA2 | gfx90a | 64 | MFMA | AGPR aliasing |
| CDNA3 | gfx942 | 64 | MFMA | Direct-to-LDS (no scattering) |
| CDNA4 | gfx950 | 64 | MFMA + Scaled | Direct-to-LDS (with scattering), ds_read_tr, permlane swap |
| RDNA3 | gfx1100–1153 | 32 | WMMA (limited) | — |
| RDNA4 | gfx1200–1201 | 32 | — | — |
| GFX1250 | gfx1250 | 32 | WMMA 256b/128b | TDM, mbarrier, cluster, ds_load_tr, warp specialization |

## Core Lowering Patterns

### Load/Store Lowering

Two paths for loads: **global pointer loads** and **buffer loads**.

**Global loads** (`LoadOpConversion`): unpack per-thread pointers → vectorized `llLoad` with mask/other.
Vectorization: `getVectorSize(ptr, axisAnalysis)` returns max contiguous elements that can be loaded as a single vector.

**Buffer loads** (`BufferLoadOpConversion`): create buffer resource descriptor (V# in SGPRs) + per-thread offsets (VGPRs) → `rocdl.raw.ptr.buffer.load`. Key advantage: scalar base pointer + 32-bit offsets save VGPR pressure.

**Direct-to-LDS loads** (`BufferLoadToLocalOpConversion`, `GlobalLoadToLocalOpConversion`): bypass VGPRs, data goes straight to LDS. Uses `LinearLayout` to compute shared memory addresses. Swizzling handled by shuffling source pointers across lanes. GFX9 (CDNA3) lacks per-lane scattering — uses warp-level coalesced writes.

**Stores** follow similar patterns: `StoreOpConversion` (global), `BufferStoreOpConversion` (buffer).

### Dot Op Lowering

Dispatched by accumulator encoding:

| Encoding | Handler | Instruction |
|----------|---------|------------|
| `AMDMfmaEncodingAttr` | `convertMFMA` | `rocdl.mfma.*` intrinsics |
| `AMDWmmaEncodingAttr` | `convertWMMA` | WMMA intrinsics |
| `BlockedEncodingAttr` | `convertAMDFMADot` | Scalar FMA loops |

`DotScaledOp` dispatches similarly to `convertScaledMFMA` / `convertScaledWMMA`.

### Buffer Operations Model

`BufferEmitter` creates ROCDL buffer intrinsics with:

1. **Resource descriptor** (V#): 128-bit `v4i32` in SGPRs — base pointer + size + format flags
2. **Offset**: per-thread VGPR offset (32-bit, non-negative)
3. **Out-of-bounds masking**: set offset to `max_int32 + 1` for masked-off threads (OOB → nop)

Requirements: scalar base pointer, 32-bit non-negative offsets. Violation of pointer uniformity → scalarized loop.

### Memory/Barrier Lowering

**LDS Transpose Reads** (`TransLocalLoadOpConversion`): `ds_read_tr*_b64` (CDNA4) / `ds_load_tr*` (GFX1250) for cooperative transposed loads from LDS. 16 lanes cooperate to transpose an N×16 tile.

**MBarrier** (GFX1250): `InitBarrierOp` → LDS atomic init, `ArriveBarrierOp` → `ds_atomic_barrier_arrive_rtn_b64`, `WaitBarrierOp` → spin-wait loop with `s_sleep`.

**Cluster Barrier** (GFX1250): `ClusterBarrierArriveOp` → `rocdl.barrier_signal(-3)`, `ClusterBarrierWaitOp` → `rocdl.barrier_wait(-3)`.

**BarrierOp** (general): lowered to `MemoryCounterWaitOp` + `s_barrier`. The wait op encodes `s_waitcnt` (GFX9) or `s_wait_loadcnt/storecnt/dscnt` (GFX12+).

### Async Copy & Memory Fencing

Memory ordering for atomics uses `LLVM::FenceOp` with AMDHSA sync scopes (`agent`, `workgroup`, or default system). Release fence emitted before atomics, acquire fence after. GFX9 uses `buffer_wbl2 sc1=1` + `s_waitcnt vmcnt(0)` for agent-scope release.

GFX9 async ops (CDNA3) require alias info annotation (`annotateLocalLoadsSyncedViaAsyncWait`) so LLVM doesn't insert conservative waits between direct-to-LDS writes and LDS reads.

### Layout Conversion Lowering

AMD-specific `ConvertLayoutOp` patterns avoid shared memory round-trips:

**Permlane Swap** (`ConvertLayoutOpPermlaneSwap`, CDNA4): uses `v_permlane16_swap` / `v_permlane32_swap` for intra-warp layout conversions. Decomposes the conversion into lane-register bit transpositions `(r_i, l4)` or `(r_i, l5)`. Handles both simple transpositions and 3-cycles `(r_i, l4, l5)`. Used for MFMA→DotOp conversions and epilogue store vectorization.

**In-Thread VPerm** (`ConvertLayoutOpInThreadSwap`): uses `v_perm` for intra-thread byte shuffles (8-bit element types only). Multi-stage algorithm: 1-way deps (copy/perm single register), 2-way deps (perm 2 registers), 4-way deps (pair→quad byte assembly with temporary registers). Each `v_perm` copies 4 bytes from two source registers with arbitrary byte ordering.

### Elementwise Op Lowering

`ElementwiseOpToLLVM.cpp` handles AMD-specific type conversions with two paths:

| Conversion | CDNA3 (SW) | CDNA4+ (HW) |
|------------|-----------|------------|
| FP8 E4M3FN ↔ FP32/FP16 | Bit manipulation + LUT rounding | `cvt_scale_pk` / `cvt_pk_scale_pk8` intrinsics |
| BF8 E5M2 ↔ FP32/FP16 | Bit manipulation + rounding | `cvt_scale_pk` intrinsics |
| FP8/BF8 FNUZ variants | Software-only | Software-only |

SW path: full subnormal handling with LUT-based halfway-point rounding, saturation, NaN preservation. HW path: packed 4-element or 8-element conversion via ROCDL intrinsics with scale factor.

### Instruction Scheduling Hints

`SchedInstructions.cpp` provides two passes for controlling the AMDGPU backend scheduler:

**InsertInstructionSchedHints**: walks `scf.for` loops, inserts `InstructionSchedHint` at the start of loops containing chain dots (flash attention).

**LowerInstructionSchedHints**: converts hints to ROCDL intrinsics:
- `SchedHint::attention` → `ROCDL::IglpOpt(2)` (instruction group level parallelism) with `ROCDL::SchedBarrier(none)` at block boundaries to prevent instruction motion across block edges.

### Scalarize Packed Float Ops

`ScalarizePackedFOps.cpp` is a post-LLVM-optimization LLVM IR pass that:
- Finds basic blocks containing MFMA/WMMA intrinsic calls
- Scalarizes vector `fmul`/`fadd`/`fsub` in those blocks
- Eliminates `v_pk_mul_f32` / `v_pk_add_f32` which cannot be *issued* in parallel with matrix core instructions
- Converts to scalar `v_mul_f32_e32` / `v_add_f32_e32` that interleave better with MFMA/WMMA
- These packed ops are introduced by LLVM's `VectorCombine::foldPermuteOfBinops` during optimization, not by Triton itself

### Mode Register

`adjustModeRegister()` post-processes the module to set wavefront execution mode (rounding, FTZ, etc.) via `s_setreg_b32`.

## Architecture-Specific Capabilities (TargetInfo)

| Capability | CDNA3 | CDNA4 | GFX1250 |
|------------|-------|-------|---------|
| Direct-to-LDS scattering | no | yes | yes |
| Buffer load to local | yes | yes | yes |
| Direct from LDS store | no | no | yes |
| LDS transpose reads | no | ds_read_tr{4,8,16}_b64 | ds_load_tr{4,6,8,16} |
| Multi-CTA launch | no | no | yes |
| TDM (tensor data movement) | no | no | yes |
| Permlane swap | no | yes | no |
| Warp specialization | no | no | yes |
| Wave ID query | no | no | yes |
| DPP broadcast | yes | yes | no |
| Vectorized atomics | yes | yes | yes |

## Additional Resources

- For detailed load/store lowering, see [load-store-reference.md](load-store-reference.md)
- For dot operation lowering details, see [dot-ops-reference.md](dot-ops-reference.md)
- For memory, barrier, and async lowering, see [memory-barrier-reference.md](memory-barrier-reference.md)
- For layout conversion, elementwise, scheduling, and packed ops, see [valu-layout-sched-reference.md](valu-layout-sched-reference.md)
