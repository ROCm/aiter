---
name: llvm-amdgpu-sched-heuristics
description: Understand and debug instruction scheduling heuristics in the LLVM AMDGPU backend for AMD GPUs (CDNA3/gfx942, CDNA4/gfx950, GFX1250). Covers scheduling strategies (MaxOccupancy, MaxILP, MaxMemoryClause), multi-stage scheduling pipeline, register pressure tracking (GCNRegPressure), DAG mutations (IGroupLP, MacroFusion, BarrierLatency, ExportClustering), candidate selection heuristics, iterative schedulers (MinReg, ILP), scheduling models (SISchedule.td), MFMA padding, rematerialization, and wave priority. Use when analyzing scheduling decisions, debugging register pressure issues, understanding why instructions are reordered, tuning occupancy vs ILP trade-offs, or when the user mentions GCNSchedStrategy, GCNScheduleDAGMILive, IGroupLP, scheduling stages, occupancy, register pressure, MFMA interleaving, memory clauses, rematerialization, or amdgpu-schedule-metric-bias.
---

# LLVM AMDGPU Instruction Scheduling Heuristics

How LLVM schedules instructions for AMD GPUs. The scheduler balances occupancy (wave parallelism) against ILP (latency hiding within a single wave).

Source: [`llvm/lib/Target/AMDGPU/`](https://github.com/llvm/llvm-project/tree/main/llvm/lib/Target/AMDGPU)

## Key Files

| File | Role |
|------|------|
| [`GCNSchedStrategy.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/GCNSchedStrategy.cpp) | Main scheduling strategy and multi-stage pipeline |
| [`GCNSchedStrategy.h`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/GCNSchedStrategy.h) | Strategy, stage, and DAG class declarations |
| [`GCNRegPressure.h`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/GCNRegPressure.h) | Register pressure tracking (SGPR/VGPR/AGPR/AVGPR) |
| [`GCNRegPressure.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/GCNRegPressure.cpp) | RP calculation, comparison, live reg tracking |
| [`AMDGPUIGroupLP.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/AMDGPUIGroupLP.cpp) | IGLP DAG mutation (SCHED_GROUP_BARRIER, IGLP_OPT) |
| [`GCNIterativeScheduler.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/GCNIterativeScheduler.cpp) | Alternative iterative scheduling framework |
| [`GCNILPSched.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/GCNILPSched.cpp) | ILP scheduler (Sethi-Ullman based) |
| [`GCNMinRegStrategy.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/GCNMinRegStrategy.cpp) | Minimum register pressure scheduler |
| [`SISchedule.td`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/SISchedule.td) | Scheduling model latencies & resources |
| [`AMDGPUBarrierLatency.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/AMDGPUBarrierLatency.cpp) | Barrier/fence latency DAG mutation |
| [`AMDGPUHazardLatency.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/AMDGPUHazardLatency.cpp) | Hazard latency DAG mutation (VALU mask writes) |
| [`AMDGPUMacroFusion.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/AMDGPUMacroFusion.cpp) | Macro fusion (VCC defs → uses clustering) |
| [`AMDGPUExportClustering.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/AMDGPUExportClustering.cpp) | Export instruction clustering |
| [`AMDGPUSetWavePriority.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/AMDGPUSetWavePriority.cpp) | Wave priority management (s_setprio) |

## Scheduling Architecture Overview

```
GCNScheduleDAGMILive : ScheduleDAGMILive
  ├── schedule()          — records regions, computes RP
  ├── finalizeSchedule()  — calls runSchedStages()
  └── runSchedStages()    — iterates through GCNSchedStage instances

GCNSchedStrategy : GenericScheduler
  ├── GCNMaxOccupancySchedStrategy  — default, balance occupancy/ILP
  ├── GCNMaxILPSchedStrategy        — maximize ILP
  └── GCNMaxMemoryClauseSchedStrategy — maximize memory clauses
```

The scheduler operates in a **multi-stage pipeline** — it schedules all regions of the function in each stage, then re-schedules them with different constraints in subsequent stages. This handles the kernel-wide effect of register pressure on occupancy: usually only a few regions limit occupancy, so constraints can be relaxed in other regions to improve ILP.

## Scheduling Strategies

| Strategy | CLI flag | Stages |
|----------|----------|--------|
| `GCNMaxOccupancySchedStrategy` | `gcn-max-occupancy` (default) | OccInitialSchedule → RewriteMFMAForm → UnclusteredHighRPReschedule → ClusteredLowOccupancyReschedule → PreRARematerialize |
| `GCNMaxILPSchedStrategy` | `gcn-max-ilp` | ILPInitialSchedule |
| `GCNMaxMemoryClauseSchedStrategy` | `gcn-max-memory-clause` | MemoryClauseInitialSchedule |

### GCNMaxOccupancy Stages (Default)

**Stage 0: OccInitialSchedule** — initial schedule targeting maximum occupancy. Reverts if the new schedule decreases occupancy or produces a worse schedule (more bubbles) without improving RP.

**Stage 1: RewriteMFMAForm** — rewrites MFMA instructions to use different register forms (AGPR↔VGPR) when it reduces register pressure. Uses a cost model: `copyCost` vs `rpSavings`. Only active on targets with AGPRs.

**Stage 2: UnclusteredHighRPReschedule** — re-schedules only regions with high register pressure, this time **without** load/store clustering. Removes `LoadCluster`/`StoreCluster` mutations. Temporarily drops the occupancy target to allow more freedom in high-RP regions. Reverts if occupancy drops or the schedule metric (bubbles ratio) worsens.

**Stage 3: ClusteredLowOccupancyReschedule** — if the actual occupancy discovered after stages 0–2 is lower than the initial target, re-schedules all regions with the real (lower) occupancy target. This gives low-RP regions more ILP freedom when they don't constrain occupancy.

**Stage 4: PreRARematerialize** — sinks rematerializable instructions from their defining region to their using region to reduce live ranges and RP. Targets registers with a single def and single user in different regions. Scoring heuristic: `MaxFreq > FreqDiff > RegionImpact`. Can increase occupancy or reduce spilling.

### GCNMaxILP Heuristic Order

1. RegExcess (avoid spilling)
2. PhysReg bias
3. Stall (latency stall cycles)
4. ResourceReduce / ResourceDemand
5. Latency (unconditional)
6. Weak edges (clustering)
7. Cluster (keep clustered nodes together)
8. RegCritical (occupancy-affecting pressure)
9. RegMax (overall pressure)
10. NodeOrder (original program order)

### GCNMaxMemoryClause Heuristic Order

1. PhysReg bias
2. RegExcess / RegCritical
3. **Cluster (prioritized early)**
4. **IsLongLatency (prefer memory loads)**
5. Stall, ResourceReduce/Demand
6. Latency
7. RegMax
8. NodeOrder

## Register Pressure Tracking

`GCNRegPressure` tracks 4 register kinds: SGPR, VGPR (ArchVGPR), AGPR, AVGPR (pseudo registers that can be either).

### Pressure Limits

```
SGPRExcessLimit  = allocatable SGPR_32 count
VGPRExcessLimit  = allocatable VGPR_32 count
SGPRCriticalLimit = min(maxSGPRs(targetOccupancy), SGPRExcessLimit) - bias - ErrorMargin
VGPRCriticalLimit = min(maxVGPRs(targetOccupancy), VGPRExcessLimit) - bias - ErrorMargin
```

- `ErrorMargin` = 3 (default, compensates for RP tracker imprecision)
- `HighRPSGPRBias` / `HighRPVGPRBias` = 7 (under high RP, bias limits down)
- `amdgpu-schedule-metric-bias` = 10 (default, weight occupancy vs latency)

### Pressure Comparison

`GCNRegPressure::less()` compares pressures in tiered order:
1. Better occupancy wins
2. Less spilling (VGPR spills first, then SGPR)
3. Less tuple register weight (VGPR tuples if SGPR pressure is low)
4. Less raw register count

### Candidate Pressure Initialization

`initCandidate()` sets `RPDelta.Excess` and `RPDelta.CriticalMax`:

- **Excess**: triggered when pressure >= ExcessLimit. Only tracks VGPRs **or** SGPRs (prefers VGPR tracking when VGPR pressure + 16 >= VGPRExcessLimit).
- **CriticalMax**: triggered when pressure >= CriticalLimit. Reports whichever register kind has a larger delta over its limit.

### Fast Path: PressureDiffs

For bottom-up scheduling, the scheduler uses cached `PressureDiff` arrays (~80% of SUnits) instead of expensive `RegPressureTracker` queries. Fallback to full LIS queries for SUnits with physical register operands or subregister defs.

### AMDGPU-specific RP Trackers

`-amdgpu-use-amdgpu-trackers` enables `GCNDownwardRPTracker`/`GCNUpwardRPTracker` — AMDGPU-specific trackers with better precision than the generic `RegPressureTracker`. They directly track `GCNRegPressure` values (SGPR/VGPR/AGPR/AVGPR counts).

## DAG Mutations

Applied during DAG construction to add/modify scheduling edges.

### IGroupLP (`AMDGPUIGroupLP.cpp`)

Overrides default scheduling using `SCHED_GROUP_BARRIER` / `IGLP_OPT` intrinsics. Groups instructions into `SchedGroup`s by type (VALU, MFMA, VMEM_READ, VMEM_WRITE, DS_READ, DS_WRITE, SALU, TRANS) and adds ordering edges.

**Pipeline solver**: assigns SUnits to SchedGroups when there are ambiguities (an instruction fits multiple groups). Two algorithms:
- **Greedy** (default, polynomial time): assigns each SU to lowest-cost group
- **Exact** (`-amdgpu-igrouplp-exact-solver`, exponential time): explores full search tree with pruning

**Built-in strategies** (selected by `IGLP_OPT` immediate):
- `MFMASmallGemmOpt` (ID 0) — interleaves DS reads with MFMAs (2 DS per MFMA)
- `MFMASmallGemmSingleWaveOpt` (ID 1)
- `MFMAExpInterleaveOpt` (ID 2) — interleaves TRANS/EXP instructions with MFMA chains
- `MFMAExpSimpleInterleaveOpt` (ID 3)

**SchedGroupMask** bit definitions:
```
ALU=0x001, VALU=0x002, SALU=0x004, MFMA=0x008,
VMEM=0x010, VMEM_READ=0x020, VMEM_WRITE=0x040,
DS=0x080, DS_READ=0x100, DS_WRITE=0x200, TRANS=0x400
```

### MacroFusion (`AMDGPUMacroFusion.cpp`)

Fuses VCC-producing instructions with their consumers (`V_ADDC_U32_e64`, `V_SUBB_U32_e64`, `V_CNDMASK_B32_e64`). Clustering VCC defs with uses improves VOP2 encoding shrinkage opportunities.

### BarrierLatency (`AMDGPUBarrierLatency.cpp`)

Adds synthetic latency to edges:
- `ATOMIC_FENCE` predecessors: +2000 cycles for memory loads (encourages prefetching before fences)
- `S_BARRIER_SIGNAL` → `S_BARRIER_WAIT`: +16 cycles (`-amdgpu-barrier-signal-wait-latency`, encourages independent work between signal/wait)

### HazardLatency (`AMDGPUHazardLatency.cpp`)

On targets with `VALUMaskWriteHazard` (Wave64): boosts latency ×3 on VALU→VALU edges that write SGPRs, reducing VALU pipeline stalls.

### ExportClustering (`AMDGPUExportClustering.cpp`)

Clusters export instructions together, position exports first. Removes default inter-export barrier edges and rebuilds them as a chain.

## Iterative Scheduler (`GCNIterativeScheduler`)

Alternative to the stage-based scheduler, selected via `-misched`:

| Strategy | `-misched` value | Description |
|----------|-----------------|-------------|
| MinReg | `gcn-iterative-minreg` | Minimize RP using top-down Sethi-Ullman |
| MinRegForced | `gcn-iterative-minreg-forced` | Same but force for all regions |
| LegacyMaxOccupancy | `gcn-iterative-max-occupancy-experimental` | MinReg + MaxOccupancy two-pass |
| ILP | `gcn-iterative-ilp` | Maximize ILP bottom-up |

### MinReg Scheduler (`GCNMinRegStrategy`)

Top-down scheduler prioritizing minimum register consumption:
1. **Priority** (bumped for predecessors of non-ready successors)
2. **Min non-ready successors** (fewer future dependencies)
3. **Max ready successors** (most successors become schedulable)
4. **Program order** (tiebreaker)

### ILP Scheduler (`GCNILPSched`)

Bottom-up scheduler using Sethi-Ullman numbers:
1. **Critical path** depth (within MaxReorderWindow=6)
2. **Height** (within MaxReorderWindow)
3. **Sethi-Ullman priority** (register needs)
4. **Closest successor** (minimize live ranges)
5. **Scratch count** (data dependencies)
6. **Latency comparison** (height → depth → latency)

## Scheduling Models (`SISchedule.td`)

| Model | Target | Key Latencies |
|-------|--------|---------------|
| `SIFullSpeedModel` | GFX6–7 | VALU=5, Trans=10, F64=22 |
| `SIQuarterSpeedModel` | GFX8 | Trans=16, F64=22, MFMA 2/8/16-pass |
| `SIDPFullSpeedModel` | GFX908 | FMA=5, Trans=10, MFMA+DGEMM |
| `SIDPGFX942FullSpeedModel` | gfx942 | MFMA 2/4/8/16-pass, SMFMAC 4/8-pass |
| `SIDPGFX950FullSpeedModel` | gfx950 | + MFMA_SCALE (f8→2x cycles), DGEMM 16-pass |
| `GFX10SpeedModel` | GFX10 | VALU=7, Trans=10, F64=22 |
| `GFX11SpeedModel` | GFX11 | VALU=7, Trans=10, SFPU=4 |
| `GFX12SpeedModel` | GFX12 | VALU=5, Trans=10 |
| `GFX1250SpeedModel` | GFX1250 | WMMA XDL 2/4-pass, PseudoScalarTrans=4 |

**Common resource units**: `HWBranch`, `HWExport`, `HWLGKM`, `HWSALU`, `HWVMEM`, `HWVALU`, `HWTransVALU`, `HWRC`, `HWXDL` (all BufferSize=1 except HWXDL=0).

All models: `MicroOpBufferSize=1` (instructions added to ready queue immediately), `IssueWidth=1`, `PostRAScheduler=1`, `MispredictPenalty=20`.

**MFMA pass counts** (latency = passes × cycle-per-pass):
- 2-pass: 4x4x* variants
- 4-pass: 16x16x8/16/32 variants (gfx942+)
- 8-pass: 16x16x1/4, 32x32x4/8/16, SMFMAC 32x32, DGEMM 16x16
- 16-pass: 32x32x1/2/4, DGEMM 16x16 (gfx950)

## Schedule Metrics and Revert Decisions

The scheduler computes a **schedule metric** to decide whether to keep or revert a scheduling attempt:

```
Metric = (BubbleCycles × 100) / ScheduleLength
```

`BubbleCycles` counts cycles where the VALU pipeline is idle. Lower metric = better schedule.

**Revert conditions** (vary by stage):
- **OccInitialSchedule**: revert if occupancy drops, or if RP didn't improve but metric worsened
- **UnclusteredHighRPReschedule**: revert if occupancy drops below initial, or if high-RP region metric worsened
- **ClusteredLowOccupancy**: revert if occupancy drops below target
- **ILPInitialSchedule/MemoryClause**: revert if would cause spilling (`mayCauseSpilling`)

`mayCauseSpilling`: returns true if `WavesAfter` is less than function occupancy AND (not already in an excess-RP region OR RP increased).

## Pending Queue Inspection

When `MicroOpBufferSize > 0` and `(Available + Pending) ≤ 256` (configurable via `-amdgpu-scheduler-pending-queue-limit`), the scheduler also evaluates instructions in the **pending queue** using a reduced heuristic set:
1. PhysReg bias
2. RegExcess
3. RegCritical
4. ResourceReduce / ResourceDemand

This allows the scheduler to "reach ahead" and pick a not-yet-ready instruction when it significantly reduces register pressure, at the cost of inserting idle cycles.

## Wave Priority (`AMDGPUSetWavePriority`)

Inserts `s_setprio 3` before first VALU in entry block and `s_setprio 0` after the last VMEM load that has ≥100 VALU instructions (`-amdgpu-set-wave-priority-valu-insts-threshold`) following it. This temporarily raises wave priority to allow younger waves to also issue VMEM loads.

## MFMA Padding

`-amdgpu-mfma-padding-ratio=N` (0–100): inserts `s_nop` instructions to fill N% of MFMA-to-MFMA latency. Used to reduce inter-wave MFMA contention at the cost of single-wave ILP.

## Useful CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `-amdgpu-schedule-metric-bias` | 10 | Weight occupancy vs latency (100 = occupancy only) |
| `-amdgpu-schedule-relaxed-occupancy` | false | Allow lower occupancy for memory/wave-limited kernels |
| `-amdgpu-use-amdgpu-trackers` | false | Use AMDGPU-specific RP trackers |
| `-amdgpu-scheduler-pending-queue-limit` | 256 | Max queue size for pending queue inspection (0 disables) |
| `-amdgpu-disable-unclustered-high-rp-reschedule` | false | Skip stage 2 |
| `-amdgpu-disable-clustered-low-occupancy-reschedule` | false | Skip stage 3 |
| `-amdgpu-disable-rewrite-mfma-form-sched-stage` | false | Skip stage 1 |
| `-amdgpu-mfma-padding-ratio` | 0 | MFMA-to-MFMA NOP padding percentage |
| `-amdgpu-igrouplp-exact-solver` | false | Use exact IGLP solver |
| `-amdgpu-igrouplp-exact-solver-cutoff` | 0 | Max conflicts for exact solver |
| `-amdgpu-set-wave-priority-valu-insts-threshold` | 100 | VALU count for wave priority |
| `-amdgpu-barrier-signal-wait-latency` | 16 | Synthetic barrier signal→wait latency |
