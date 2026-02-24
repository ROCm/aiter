# AMDGPU Memory Model, Wait Counts, and Hazard Reference

Detailed coverage of synchronization, memory ordering, wait count insertion, and hazard handling for CDNA3 (gfx942), CDNA4 (gfx950), and GFX1250.

Source: [`llvm/lib/Target/AMDGPU/`](https://github.com/llvm/llvm-project/tree/main/llvm/lib/Target/AMDGPU)

## Wait Count Model

### Pre-GFX12 (CDNA3/gfx942, CDNA4/gfx950)

From [`SIInsertWaitcnts.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/SIInsertWaitcnts.cpp), `WaitcntGeneratorPreGFX12`:

| Counter | Events tracked | Wait Instruction |
|---------|---------------|-----------------|
| vmcnt (LOAD_CNT) | VMEM_ACCESS, VMEM_SAMPLER_READ, VMEM_BVH_READ | `s_waitcnt vmcnt(N)` |
| lgkmcnt (DS_CNT) | SMEM_ACCESS, LDS_ACCESS, GDS_ACCESS, SQ_MESSAGE | `s_waitcnt lgkmcnt(N)` |
| expcnt (EXP_CNT) | EXP_GPR_LOCK, GDS_GPR_LOCK, VMW_GPR_LOCK, EXP_PARAM/POS/LDS | `s_waitcnt expcnt(N)` |
| vscnt (STORE_CNT) | VMEM_WRITE_ACCESS, SCRATCH_WRITE_ACCESS | `s_waitcnt_vscnt null, N` |

Combined: `s_waitcnt vmcnt(X) lgkmcnt(Y) expcnt(Z)`.

SAMPLE_CNT, BVH_CNT, KM_CNT, X_CNT are empty on pre-GFX12.

### GFX12+ (GFX1250)

From [`SIInsertWaitcnts.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/SIInsertWaitcnts.cpp), `WaitcntGeneratorGFX12Plus`:

| Counter | Events tracked | Wait Instruction |
|---------|---------------|-----------------|
| loadcnt (LOAD_CNT) | VMEM_ACCESS, GLOBAL_INV_ACCESS | `s_wait_loadcnt N` |
| storecnt (STORE_CNT) | VMEM_WRITE_ACCESS, SCRATCH_WRITE_ACCESS | `s_wait_storecnt N` |
| samplecnt (SAMPLE_CNT) | VMEM_SAMPLER_READ_ACCESS | `s_wait_samplecnt N` |
| bvhcnt (BVH_CNT) | VMEM_BVH_READ_ACCESS | `s_wait_bvhcnt N` |
| dscnt (DS_CNT) | LDS_ACCESS, GDS_ACCESS | `s_wait_dscnt N` |
| kmcnt (KM_CNT) | SMEM_ACCESS, SQ_MESSAGE, SCC_WRITE | `s_wait_kmcnt N` |
| expcnt (EXP_CNT) | EXP_GPR_LOCK, GDS_GPR_LOCK, VMW_GPR_LOCK, EXP_PARAM/POS/LDS | `s_wait_expcnt N` |
| xcnt (X_CNT) | VMEM_GROUP, SMEM_GROUP | `s_wait_xcnt N` |

#### Expert Scheduling Mode (GFX1250)

Additional pseudo-counters (`hasExpertSchedulingMode()` = true):

| Counter | Events tracked | Purpose |
|---------|---------------|---------|
| va_vdst (VA_VDST) | VGPR_CSMACC_WRITE, VGPR_DPMACC_WRITE, VGPR_TRANS_WRITE, VGPR_XDL_WRITE | VGPR writes from matrix/transcendental |
| vm_vsrc (VM_VSRC) | VGPR_LDS_READ, VGPR_FLAT_READ, VGPR_VMEM_READ | VGPR reads from memory ops |

### Instruction → Counter Mapping Summary

| Instruction class | Pre-GFX12 counter | GFX12+ counter |
|-------------------|-------------------|----------------|
| VMEM loads | vmcnt | loadcnt |
| VMEM stores | vscnt | storecnt |
| Sampler reads | vmcnt | samplecnt |
| BVH reads | vmcnt | bvhcnt |
| LDS/GDS | lgkmcnt | dscnt |
| SMEM / SQ_MSG | lgkmcnt | kmcnt |
| Exports | expcnt | expcnt |
| Scratch writes | vscnt | storecnt |
| GLOBAL_INV | — | loadcnt |

### Wait Count Insertion Logic

`SIInsertWaitcnts` runs pre-emit:

1. **Scoreboard tracking**: `WaitcntBrackets` tracks outstanding events per register — which counter type, what score value.
2. **Lazy insertion**: waits only inserted when a dependent instruction is about to execute and its source register has an outstanding event.
3. **Minimal waits**: inserts the smallest count that resolves all dependencies (`ScoreLB` tracking).
4. **Soft vs hard waits**: soft waits (`promoteSoftWaitCnt`) can be strengthened later when a stronger constraint is found.
5. **Cross-BB propagation**: `BlockInfos` per-BB, predecessor brackets merged via `merge()`/`mergeScore()`/`mergeAsyncMarks()`. Loop preheaders can force `FlushVmCnt`/`FlushDsCnt`.
6. **Wait simplification**: `simplifyWaitcnt()` avoids redundant waits when multiple event types share a counter.

## Memory Ordering

### Atomic Scopes

From [`SIMemoryLegalizer.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/SIMemoryLegalizer.cpp):

```
SIAtomicScope: NONE → SINGLETHREAD → WAVEFRONT → WORKGROUP → CLUSTER → AGENT → SYSTEM
```

Scope determines cache flush/invalidation depth:
- **SINGLETHREAD**: no cache ops needed
- **WAVEFRONT**: no cache ops (all lanes see same data)
- **WORKGROUP**: L1 cache ops (LDS naturally coherent)
- **AGENT**: L2 cache ops (device-wide coherence)
- **SYSTEM**: system-level coherence (host-visible)

### Address Space → Scope Mapping

| Address space | Max required scope |
|---------------|--------------------|
| Scratch (5) | SINGLETHREAD |
| LDS (3) + Scratch | WORKGROUP |
| Global (1) / GDS (2) | AGENT or SYSTEM |
| Flat (0) | depends on runtime address |

### Acquire / Release Semantics

For **acquire** (before consuming load): invalidate caches at or above the scope level.
For **release** (after producing store): write-back caches at or above the scope level.

### Cache Control Implementation

#### Pre-GFX12 (CDNA3/4) — `SIGfx940CacheControl`

From [`SIDefines.h`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/SIDefines.h) CPol namespace:

| Bit | Name | Value | Effect |
|-----|------|-------|--------|
| GLC | Globally coherent | 1 | Bypass L1 cache, coherent access |
| SLC | System-level coherent | 2 | Bypass L2, system-level coherent |
| DLC | Device-level coherent | 4 | Device-level cache coherency |
| SCC | Scalar cache coherent | 16 | Bypass scalar cache |

`enableLoadCacheBypass()` / `enableStoreCacheBypass()` / `enableRMWCacheBypass()` set these bits based on scope and ordering.

Cache flush/invalidate instructions:
- **Agent scope**: `BUFFER_WBINVL1_VOL` (L1 writeback + invalidate)
- **Workgroup scope**: `BUFFER_GL1_INV` + `BUFFER_GL0_INV`
- **System scope**: `BUFFER_WBINVL1_VOL` + system-level fence

#### GFX12+ (GFX1250) — `SIGfx12CacheControl`

Replaces GLC/SLC/DLC with TH (Temporal Hint) and Scope fields:

| Field | Values | Effect |
|-------|--------|--------|
| TH | RT (regular temporal), NT (non-temporal), HT (high temporal), LU (last use), WB (write-back) | Cache residency hint |
| Scope | CU, SE, DEV, SYS | Coherence domain |
| NV | — | Non-volatile (persistence hint) |

`setTH()` and `setScope()` called on CPol operands via `SIGfx12CacheControl::setAtomicScope()`.

Cache flush/invalidate:
- `GLOBAL_INV` with scope operand (CU/SE/DEV/SYS)
- `insertWaitsBeforeSystemScopeStore()` before system-scope stores
- `handleCooperativeAtomic()` for cooperative atomics
- `finalizeStore()` for store finalization

## Hazard Recognition

From [`GCNHazardRecognizer.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/GCNHazardRecognizer.cpp).

### MFMA Hazards (CDNA3/4)

#### MFMA → MFMA Pipeline Latency

- Source: `getMFMAPipelineWaitStates(MI)` reads `ReleaseAtCycle` from the schedule model (`TSchedModel.getWriteProcResBegin(SC)->ReleaseAtCycle`)
- NOPs needed: `WaitStatesNeeded = getMFMAPipelineWaitStates(MI) - getWaitStatesSince(IsMAI, Limit)`
- MFMA padding: `amdgpu-mfma-padding-ratio` (0–100) fills a percentage of MFMA-to-MFMA latency with `s_nop`
- Max lookahead: **19 cycles** when AGPRs are in use, **5 cycles** otherwise

#### MFMA → VALU (`checkMAIVALUHazards`)

- Scope: any VALU, VMEM, DS, or EXP instruction that reads an AGPR written by a preceding MFMA
- Uses `getWaitStatesSinceDef()` to count cycles since the MFMA that produced the AGPR
- Mitigation: insert `s_nop` to reach the required wait states

#### MFMA → LdSt (`checkMAILdStHazards`)

- Scope: VMEM and DS instructions reading AGPRs (e.g. buffer stores from AGPR on GFX90A+)
- Same NOP-based mitigation as MFMA → VALU

#### s_nop Encoding

- `s_nop N` waits for N+1 cycles (max `s_nop 7` = 8 cycles)
- Multiple NOPs chained for longer waits
- `PreEmitNoopsCommon()` takes the maximum wait needed across all hazard checks

### WMMA Hazards (GFX1250)

From `checkWMMACoexecutionHazards()`:

- **Cannot be mitigated with S_NOPs** — returns `Hazard` (not `NoopHazard`)
- Must be resolved by the instruction scheduler or by wait count insertion
- Co-execution with certain other instructions is illegal

### CDNA4 (gfx950) Specific Hazards

| Hazard | Check function | Description |
|--------|----------------|-------------|
| Permlane | `checkPermlaneHazards()` | `v_permlane16_swap`, `v_permlane32_swap` instructions |
| CvtScale forwarding | `hasCvtScaleForwardingHazard()` | Scale conversion result forwarding hazard |
| Loop head split | `hasLoopHeadInstSplitSensitivity()` | First instruction of loop body alignment sensitivity |
| VALU wait states | `VALUWaitStates = 2` | 2 NOP wait states (vs 1 on other GFX9) |
| Trans forwarding | `hasTransForwardingHazard()` | Transcendental unit result forwarding |
| DstSel forwarding | `hasDstSelForwardingHazard()` | Destination selection forwarding |
| DOT op_sel | `hasDOTOpSelHazard()` | DOT instruction op_sel field hazard |
| VDec co-exec | `hasVDecCoExecHazard()` | VDEC co-execution hazard |

### GFX1250 Specific Hazards

| Hazard | Check / Pass | Description |
|--------|-------------|-------------|
| INVWBL2 wait | `hasINVWBL2WaitCntRequirement()` | Must wait before L2 invalidate+writeback |
| setRegMode VNOPs | `setRegModeNeedsVNOPs()` | Insert VNOP before mode register changes |
| SGPR RAW | [`AMDGPUWaitSGPRHazards.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/AMDGPUWaitSGPRHazards.cpp) | SGPR read-after-write hazards |
| DelayAlu | [`AMDGPUInsertDelayAlu.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/AMDGPUInsertDelayAlu.cpp) | `s_delay_alu` for VALU/trans/SALU dependency encoding |
| WMMA co-exec | `checkWMMACoexecutionHazards()` | Cannot be fixed with NOPs |

### Hazard Pass Ordering

1. **During scheduling** (pre-RA): `GCNHazardRecognizer` participates in scheduling decisions via [`GCNSchedStrategy.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/GCNSchedStrategy.cpp)
2. **Wait count insertion** (pre-emit): [`SIInsertWaitcnts.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/SIInsertWaitcnts.cpp) handles most synchronization
3. **Final NOP insertion** (pre-emit): `GCNHazardRecognizer::PreEmitNoopsCommon()` inserts final NOPs
4. **DelayAlu** (GFX11+, pre-emit): [`AMDGPUInsertDelayAlu.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/AMDGPUInsertDelayAlu.cpp) inserts `s_delay_alu` dependency encoding
5. **SGPR hazards** (pre-emit): [`AMDGPUWaitSGPRHazards.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/AMDGPUWaitSGPRHazards.cpp) handles SGPR-specific RAW

## Load/Store Optimization

From [`SILoadStoreOptimizer.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/SILoadStoreOptimizer.cpp):

### Instruction Classes for Merging

DS_READ, DS_WRITE, S_BUFFER_LOAD_IMM, S_LOAD_IMM, BUFFER_LOAD, BUFFER_STORE, MIMG, TBUFFER_LOAD, TBUFFER_STORE, GLOBAL_LOAD, GLOBAL_STORE, FLAT_LOAD, FLAT_STORE.

### Merge Patterns

| Source instructions | Merged to | Constraint |
|--------------------|-----------|-----------|
| 2× `ds_read_b32` | `ds_read2_b32` | Same base, stride ≤ 255 dwords |
| 2× `ds_read_b64` | `ds_read2_b64` | Same base, stride ≤ 255 qwords |
| 2× `ds_write_b32` | `ds_write2_b32` | Same base, stride ≤ 255 dwords |
| 2× `buffer_load_dword` | `buffer_load_dwordx2` | Consecutive offsets |
| 2–4× `global_load_dword` | `global_load_dwordx2/x3/x4` | Consecutive offsets |
| 2–4× `flat_load_dword` | `flat_load_dwordx2/x3/x4` | Consecutive offsets |
| 2–8× `s_buffer_load_dword` | `s_buffer_load_dwordx2/x4/x8` | Consecutive offsets |
| 2–4× `s_load_dword` | `s_load_dwordx2/x4` | Consecutive offsets |
| MIMG with partial dmask | Combined dmask | Compatible image ops |

### Merge Constraints

- Same base register
- Compatible cache policy (`CPol` must match)
- Offsets must be consecutive or constant-stride
- No intervening aliasing stores between the merged ops
- Same instruction subclass (e.g. both must be `GLOBAL_LOAD`)

### Constant Offset Promotion

`promoteConstantOffsetToImm()`: folds `base_reg + constant_reg` into the instruction's immediate offset field. Offset sizes:
- Flat/global: 13-bit signed (–4096 to +4095)
- Buffer: 12-bit unsigned (0–4095)
- GFX1250 VFLAT: 24-bit offset

## Memory Barrier Instructions

### Pre-GFX12 (CDNA3/4)

```
S_BARRIER                         workgroup barrier
S_WAITCNT vmcnt(0) lgkmcnt(0)    full memory fence
BUFFER_WBINVL1_VOL               L1 writeback + invalidate (agent scope)
BUFFER_GL1_INV                    GL1 cache invalidate
BUFFER_GL0_INV                    GL0 cache invalidate
S_DCACHE_INV                      scalar data cache invalidate
```

### GFX12+ (GFX1250)

```
S_BARRIER_SIGNAL -1 / S_BARRIER_WAIT -1   split barrier (signal then wait)
S_WAIT_LOADCNT 0                           wait for all pending loads
S_WAIT_STORECNT 0                          wait for all pending stores
S_WAIT_DSCNT 0                             wait for all pending LDS ops
S_WAIT_KMCNT 0                             wait for all pending scalar loads
GLOBAL_INV scope:CU/SE/DEV/SYS            cache invalidate with scope
```

## Key Source Files

| File | Role |
|------|------|
| [`SIInsertWaitcnts.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/SIInsertWaitcnts.cpp) | Wait count insertion, counter tracking, cross-BB propagation |
| [`GCNHazardRecognizer.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/GCNHazardRecognizer.cpp) | Hazard detection, NOP/wait insertion, MFMA/WMMA hazards |
| [`SIMemoryLegalizer.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/SIMemoryLegalizer.cpp) | Memory ordering enforcement, acquire/release fences, cache control |
| [`SILoadStoreOptimizer.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/SILoadStoreOptimizer.cpp) | Load/store merging, offset promotion |
| [`SIDefines.h`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/SIDefines.h) | Counter type enums, CPol bits (GLC/SLC/DLC/TH/Scope) |
| [`AMDGPUWaitSGPRHazards.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/AMDGPUWaitSGPRHazards.cpp) | SGPR read-after-write hazard handling |
| [`AMDGPUInsertDelayAlu.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/AMDGPUInsertDelayAlu.cpp) | GFX11+ `s_delay_alu` dependency insertion |
| [`AMDGPUMachineFunction.h`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/AMDGPUMachineFunction.h) | Per-function machine state, occupancy tracking |
| [`GCNSchedStrategy.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/GCNSchedStrategy.cpp) | Scheduling with hazard-awareness |
