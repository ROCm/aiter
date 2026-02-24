---
name: llvm-amdgpu-backend
description: Understand and debug the LLVM AMDGPU backend lowering pipeline for AMD GPUs (CDNA3/gfx942, CDNA4/gfx950, GFX1250). Covers the full codegen pass pipeline, instruction selection (SelectionDAG and GlobalISel), register model (SGPR/VGPR/AGPR), instruction formats (MFMA, WMMA, buffer, flat, DS), wait count insertion, hazard recognition, memory ordering, and cache control. Use when analyzing LLVM IR lowering to AMD machine code, debugging ISel failures, understanding register pressure, investigating wait counts or hazards, reading AMDGPU assembly, or when the user mentions LLVM AMDGPU backend, ISel, waitcnt, s_waitcnt, hazard, SGPR, VGPR, AGPR, buffer_load, global_load, ds_read, s_wait_loadcnt, GCNSchedStrategy, or SIISelLowering.
---

# LLVM AMDGPU Backend Lowering

How LLVM lowers IR to machine code for AMD GPUs. Focused on CDNA3 (gfx942), CDNA4 (gfx950), and GFX1250.

Source: [`llvm/lib/Target/AMDGPU/`](https://github.com/llvm/llvm-project/tree/main/llvm/lib/Target/AMDGPU)

## Architecture Quick Reference

| Property | CDNA3 (gfx942) | CDNA4 (gfx950) | GFX1250 |
|----------|----------------|----------------|---------|
| ISA generation | GFX9 | GFX9 | GFX12 |
| Sched model | `SIDPGFX942FullSpeedModel` | `SIDPGFX950FullSpeedModel` | `GFX1250SpeedModel` |
| Warp size | 64 | 64 | 32 |
| Matrix core | MFMA | MFMA + SMFMAC + MFMA_SCALE | WMMA (256b/128b) + scaled WMMA |
| SGPRs | 106 (s0–s105) | 106 | 106 |
| VGPRs | 1024 (v0–v1023) | 1024 | 1024 |
| AGPRs | 256 (a0–a255) | 256 | 256 |
| Wait model | vmcnt/lgkmcnt/expcnt/vscnt | same | Extended (loadcnt/storecnt/dscnt/kmcnt/samplecnt/bvhcnt/xcnt) |
| Cache control | GLC/SLC/DLC/SCC | GLC/SLC/DLC/SCC | TH/Scope |
| MTBUF/formatted MUBUF | yes | yes | no |
| Export insts | no | no | no |
| WGP mode | yes | yes | no |

## Pass Pipeline

Defined in [`AMDGPUTargetMachine.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/AMDGPUTargetMachine.cpp):

```
IR Passes:
  UnifyDivergentExitNodes → AlwaysInline → RemoveIncompatibleFunctions
  → LowerModuleLDS → LowerBufferFatPointers → LowerIntrinsics
  → AtomicOptimizer → LowerKernelArguments → PromoteKernelArguments
  → LowerKernelAttributes → Attributor → AnnotateUniformValues
  → UniformIntrinsicCombine → CodeGenPrepare → LateCodeGenPrepare → PromoteAlloca

ISel (GlobalISel path):
  IRTranslator → Legalizer → RegBankSelect (AMDGPURegBankSelect) → InstructionSelect

ISel (SelectionDAG path):
  AMDGPUISelLowering → SIISelLowering → AMDGPUISelDAGToDAG

Pre-RA MIR:
  LowerI1Copies → LowerWWMCopies → LowerSGPRSpills → FixSGPRCopies → FixVGPRCopies
  → FoldOperands → PeepholeSDWA → ShrinkInstructions → OptimizeExecMaskingPreRA
  → OptimizeVGPRLiveRange → LoadStoreOptimizer → DPPCombine → PrepareAGPRAlloc

Register Allocation:
  SGPR (Greedy/Fast) → WWM → VGPR (Greedy/Fast)

Post-RA:
  OptimizeExecMasking → FormMemoryClauses → PostRABundler
  → CreateVOPD (if hasVOPD3()) → RewriteAGPRCopyMFMA

Pre-Emit:
  InsertHardClauses → InsertWaitcnts → MemoryLegalizer
  → InsertDelayAlu (GFX11+) → WaitSGPRHazards → HazardRecognizer
  → LowerVGPREncoding → PreloadKernArgProlog
```

## Key Files

| File | Role |
|------|------|
| [`AMDGPUTargetMachine.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/AMDGPUTargetMachine.cpp) | Pass pipeline construction |
| [`SIISelLowering.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/SIISelLowering.cpp) | Custom DAG lowering (loads/stores, intrinsics, address casts) |
| [`AMDGPUISelDAGToDAG.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/AMDGPUISelDAGToDAG.cpp) | DAG pattern selection, complex addressing patterns |
| [`AMDGPUInstructionSelector.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/AMDGPUInstructionSelector.cpp) | GlobalISel instruction selection |
| [`AMDGPULegalizerInfo.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/AMDGPULegalizerInfo.cpp) | GlobalISel legalization rules |
| [`AMDGPURegisterBankInfo.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/AMDGPURegisterBankInfo.cpp) | Register bank selection (SGPR/VGPR/AGPR/VCC) |
| [`GCNSchedStrategy.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/GCNSchedStrategy.cpp) | Scheduling strategy (occupancy/ILP/memory clause) |
| [`SIInsertWaitcnts.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/SIInsertWaitcnts.cpp) | Wait count insertion, counter tracking |
| [`GCNHazardRecognizer.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/GCNHazardRecognizer.cpp) | Hazard detection and NOP insertion |
| [`SIMemoryLegalizer.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/SIMemoryLegalizer.cpp) | Memory ordering enforcement, cache control |
| [`SILoadStoreOptimizer.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/SILoadStoreOptimizer.cpp) | Load/store merging |
| [`SIInstrInfo.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/SIInstrInfo.cpp) | Instruction queries and transforms |
| [`GCNSubtarget.h`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/GCNSubtarget.h) | Feature flag queries (`has*()` methods) |
| [`GCNProcessors.td`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/GCNProcessors.td) | Processor definitions (gfx942/950/1250) |
| [`VOP3PInstructions.td`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/VOP3PInstructions.td) | MFMA, WMMA, packed VOP3P instruction defs |
| [`BUFInstructions.td`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/BUFInstructions.td) | Buffer load/store instruction defs |
| [`FLATInstructions.td`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/FLATInstructions.td) | Flat/global/scratch instruction defs |
| [`DSInstructions.td`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/DSInstructions.td) | LDS/GDS instruction defs |
| [`SIRegisterInfo.td`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/SIRegisterInfo.td) | Register class definitions |
| [`SIDefines.h`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/SIDefines.h) | Counter types, CPol bits, wait count enums |

## Address Spaces

| AS | Name | Instruction class | Notes |
|----|------|-------------------|-------|
| 0 | Flat | `FLAT_*` (flat_load/store) | 64-bit VGPR address |
| 1 | Global | `GLOBAL_*` (global_load/store) | SADDR+VGPR or VGPR-only |
| 2 | Region (GDS) | `DS_*` with gds=1 | |
| 3 | Local (LDS) | `DS_*` (ds_read/write) | M0 init on GFX9 |
| 4 | Constant | `S_LOAD_*` (scalar) or `GLOBAL_*` | Scalar if uniform |
| 5 | Private (scratch) | `SCRATCH_*` (scratch_load/store) | SADDR/SVS/ST variants |
| 7 | Buffer fat ptr | — | 160-bit, lowered by `LowerBufferFatPointers` before ISel |
| 8 | Buffer resource | `BUFFER_*` via intrinsics | 128-bit V# descriptor (v4i32) |

## Register Model

Defined in [`SIRegisterInfo.td`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/SIRegisterInfo.td):

**SGPR** — scalar, uniform across wavefront. 106 registers (s0–s105). Tuples: 64/96/128/256/512/1024-bit, aligned to 4 units (128-bit).

**VGPR** — vector, per-lane. 1024 registers (v0–v1023). Tuples: 64–1024-bit, stride 1. `VGPR_32_Lo128` subset for VOP1/2/C encoding.

**AGPR** — accumulator, 256 registers (a0–a255). Used for MFMA/WMMA `vdst`/`src2`. Transfer via `v_accvgpr_read`/`v_accvgpr_write`. On GFX90A+, even-aligned tuples required (`RequiresAlignVGPR`).

**Special**: VCC (64-bit), EXEC (64-bit), M0, FLAT_SCR, TTMP0–TTMP15.

### Register Banks (GlobalISel)

Defined in [`AMDGPURegisterBankInfo.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/AMDGPURegisterBankInfo.cpp):

| Bank | Usage | Copy to SGPR |
|------|-------|-------------|
| SGPRRegBank | Uniform scalars, addresses, constants | — |
| VGPRRegBank | Divergent values, load data | cost=∞ |
| AGPRRegBank | MFMA/WMMA accumulators | cost=∞ |
| VCCRegBank | s1 condition codes (VCC register) | cost=∞ |

Divergence drives selection: uniform → SGPR, divergent → VGPR. AGPR ↔ VGPR costs 4 (goes through `v_accvgpr_read/write`). AGPR ↔ AGPR also costs 4 (via VGPR intermediary).

## Instruction Selection

Two paths: **SelectionDAG** and **GlobalISel**.

### Custom DAG Lowering (`SITargetLowering::LowerOperation`)

Key custom-lowered operations from [`SIISelLowering.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/SIISelLowering.cpp):

| Operation | Handler | Notes |
|-----------|---------|-------|
| LOAD/STORE | `LowerLOAD/LowerSTORE` | v2–v16i32, v32i32, i1, i16 |
| INTRINSIC_W_CHAIN | `LowerINTRINSIC_W_CHAIN` | Buffer loads, atomics |
| INTRINSIC_VOID | `LowerINTRINSIC_VOID` | Buffer stores |
| ADDRSPACECAST | `LowerADDRSPACECAST` | AS conversions, buffer fat ptrs |
| ATOMIC_CMP_SWAP | `LowerATOMIC_CMP_SWAP` | i32/i64 buffer/flat atomics |
| SELECT | `LowerSELECT` | i64, v2i32 |
| BRCOND | `LowerBRCOND` | Branch on condition |
| GlobalAddress | `LowerGlobalAddress` | Global symbol references |
| FSQRT/FDIV/FSIN/FCOS | custom | f32, f64 |

### Buffer Intrinsic Lowering

`llvm.amdgcn.raw.buffer.load/store` → `LowerINTRINSIC_W_CHAIN`:

1. `bufferRsrcPtrToVector()` — ptr addrspace(8) → v4i32 resource descriptor
2. `splitBufferOffsets()` — offset → `{voffset (VGPR), soffset (SGPR), immoffset (12-bit imm)}`
3. `buildRSRC()` — constructs V# from base_ptr + RsrcDword1 + RsrcDword2And3
4. Selects `BUFFER_LOAD_*` / `BUFFER_STORE_*` with appropriate BUFAddrKind

V# (buffer resource descriptor) layout: 128-bit v4i32 — dword0/1: base address, dword2: stride+size, dword3: format/type flags.

### DAG-to-DAG Selection

[`AMDGPUISelDAGToDAG.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/AMDGPUISelDAGToDAG.cpp) `Select()` handles:

- `SelectINTRINSIC_W_CHAIN/WO_CHAIN/VOID` for buffer/image intrinsics
- `SelectBuildVector` for 16-bit packed or REG_SEQUENCE
- `SelectMAD_64_32`, `SelectMUL_LOHI` for wide multiply
- `glueCopyToM0LDSInit` for M0 init on GFX9 LDS access
- `matchLoadD16FromBuildVector` for D16 load merging
- Complex patterns: `SelectMUBUFAddr64`, `SelectFlatOffset`, `SelectGlobalOffset`

### GlobalISel Legalization

From [`AMDGPULegalizerInfo.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/AMDGPULegalizerInfo.cpp):

| Address Space | Max Load | Max Store | Notes |
|---------------|----------|-----------|-------|
| Private (5) | 128-bit | 32-bit | 128 if flat scratch enabled |
| Local (3) | 128-bit | 64-bit | 128 if `useDS128` |
| Global/Constant (1/4) | 512-bit | 128-bit | |
| Flat (0) | 128-bit | 32-bit | |

Buffer rsrc pointers (addrspace 8): `castBufferRsrcToV4I32()` before MUBUF selection.

## Scheduling

Defined in [`GCNSchedStrategy.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/GCNSchedStrategy.cpp):

### Strategies

| Strategy | Flag | Usage |
|----------|------|-------|
| `GCNMaxOccupancySchedStrategy` | `gcn-max-occupancy` (default) | Balance occupancy vs ILP |
| `GCNMaxILPSchedStrategy` | `gcn-max-ilp` | Maximize ILP |
| `GCNMaxMemoryClauseSchedStrategy` | `gcn-max-memory-clause` | Maximize memory clause grouping |

### MaxOccupancy Stages

1. **OccInitialSchedule** — occupancy-based initial schedule
2. **RewriteMFMAForm** — MFMA instruction layout rewriting
3. **UnclusteredHighRPReschedule** — reduce register pressure in high-RP regions
4. **ClusteredLowOccupancyReschedule** — improve ILP under low occupancy
5. **PreRARematerialize** — rematerialization before RA

### Register Pressure

- `SGPRExcessLimit` / `VGPRExcessLimit`: from allocatable register count
- `SGPRCriticalLimit` / `VGPRCriticalLimit`: `min(MaxRegsForOccupancy, ExcessLimit) - bias`
- Occupancy target: `ST.getMaxNumSGPRs(TargetOccupancy)` / `ST.getMaxNumVGPRs(TargetOccupancy)`
- Bias: `amdgpu-schedule-metric-bias` (default 10)

### DAG Mutations

`LoadCluster`, `StoreCluster`, `IGroupLPDAGMutation`, `AMDGPUMacroFusion`, `AMDGPUExportClustering`, `AMDGPUBarrierLatency`, `AMDGPUHazardLatency`.

MFMA padding: `amdgpu-mfma-padding-ratio` (0–100) fills a percentage of MFMA-to-MFMA latency with `s_nop`.

## Target-Specific Features

Feature queries in [`GCNSubtarget.h`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/GCNSubtarget.h):

### CDNA4 (gfx950) additions over CDNA3

| Feature | Method |
|---------|--------|
| Scaled MFMA | `hasGFX950Insts()` — `v_smfmac_*`, `v_mfma_scale_f32_*_f8f6f4` |
| FP8/BF8/FP4/FP6/BF6 convert | `hasFP8ConversionScaleInsts()`, `hasFP4ConversionScaleInsts()` etc. |
| Permlane16/32 swap | `hasPermlane16Swap()`, `hasPermlane32Swap()` |
| LDS B96/B128 loads | `hasLDSLoadB96_B128()` |
| DWORDX3/X4 LDS buffer | `hasGFX950Insts()` |
| DS transpose reads | `ds_read_b64_tr_b4/b8/b16`, `ds_read_b96_tr_b6` |
| Packed add-min/max | `V_PK_ADD_MAX/MIN_I16/U16` |
| FP8 dot products | `hasDot11Insts()` |
| CvtScale fwd hazard | `hasCvtScaleForwardingHazard()` |
| Loop head split | `hasLoopHeadInstSplitSensitivity()` |

### GFX1250 additions

| Feature | Method |
|---------|--------|
| WMMA 256b/128b | `hasWMMAInsts()`, `hasWMMA256bInsts()`, `hasWMMA128bInsts()` |
| Scaled WMMA | `hasGFX1250Insts()` |
| Extended wait counts | `hasExtendedWaitCounts()` — loadcnt/storecnt/dscnt/kmcnt/samplecnt/bvhcnt/xcnt |
| Expert scheduling | `hasExpertSchedulingMode()` — VA_VDST/VM_VSRC |
| TH/Scope cache control | `SIGfx12CacheControl` replaces GLC/SLC/DLC |
| VOPD3 dual-issue | `hasVOPD3()` |
| 64-bit vector multiply | `hasVectorMulU64()` |
| AddPC64 | `hasAddPC64Inst()` |
| Scale offset | `hasScaleOffset()` |
| Signed GVS offset | `hasSignedGVSOffset()` |
| TDM async loads | Tensor Data Movement |
| DS transpose loads | `ds_load_tr4_b64`, `ds_load_tr6_b96`, `ds_load_tr8_b64`, `ds_load_tr16_b128` |
| Async global→LDS | `global_load_lds_dword/x2/x3/x4` |
| MBarrier sync | `ds_atomic_async_barrier_arrive_b64`, `ds_atomic_barrier_arrive_rtn_b64` |
| FMA F64 variants | `hasFmaakFmamkF64Insts()` |
| MovB64 | `hasMovB64()` |
| No MTBUF/VINTERP | `hasMTBUFInsts()` = false, `hasVINTERPEncoding()` = false |

## Additional Resources

- For instruction categories and operand details, see [instructions-reference.md](instructions-reference.md)
- For wait counts, hazards, and memory model, see [memory-sync-reference.md](memory-sync-reference.md)
