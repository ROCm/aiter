# Identifying the CK Unified-Attention FP8 Bottleneck with `rocprofv3`

This note documents how to profile the CK Unified Attention kernel on
gfx950 (MI355) using `rocprofv3`, how to translate the raw numbers into a
clear bottleneck story, and how to map hot instructions back to the C++
source in `unified_attention_pipeline.hpp`.

Reference shape used throughout: decode `b=4 sq=8 sk=4096 hq=64 hk=8
d=128 block=32 dtype=fp8 --no-splitkv`. CK takes ~190us per dispatch,
Triton 3D takes ~38us → a **~5x gap** that we want to explain.

All commands below are reproducible via:

```bash
cd ua-test-scripts/rocprof_analysis
./run_profile.sh -b 4 -sq 8 -sk 4096 -hq 64 -hk 8 -d 128 \
                 --block-size 32 --dtype fp8 --no-splitkv
```

The wrapper runs three phases (kernel-trace, PMC, PC sampling) and
prints aggregated tables; the rest of this document explains each phase
and the conclusions.

---

## 1. Tooling on this box

```bash
$ rocprofv3 --version
1.1.0  gfx950
```

Output formats we use:

| Format | Used for |
| --- | --- |
| `csv` | per-dispatch timing + counter rows (easy to parse) |
| `json` | PC sampling (the host_trap samples carry full `hw_id` / `pc.offset`) |
| `rocpd` | sqlite snapshot of everything (useful for ad-hoc queries) |

Performance counter names live in
`/opt/rocm/share/rocprofiler-sdk/counter_defs.yaml`; there are 629
counters defined for gfx950. The ones we need are listed below.

---

## 2. Phase A — kernel trace + resource fingerprint

Goal: find the dominant kernel and check launch resources (workgroup
size, VGPR, LDS) — the cheap "what am I even looking at" pass.

```bash
rocprofv3 --kernel-trace --stats \
  -d phase_a -o trace -f csv \
  -- python3 ../test_single_shape.py <shape args> --warmup 5 --iters 50
```

Outputs:

| File | Contents |
| --- | --- |
| `trace_kernel_trace.csv` | one row per dispatch: timestamps, VGPR/LDS, grid/block |
| `trace_kernel_stats.csv` | aggregate min/max/avg duration per kernel |
| `trace_domain_stats.csv` | per-domain (kernel dispatch, HIP API, …) totals |

`parse_kernel_trace.py --warmup N` is the existing helper that compares
CK vs Triton timings while skipping warmup dispatches.

### What we observed for decode-d128-fp8

| Resource | CK | Triton 3D | Notes |
| --- | --- | --- | --- |
| Workgroup size | **512 (8 warps)** | **128 (2 warps)** | 4x larger blocks in CK |
| Grid | 20480 x 1 x 1 | 2560 x 8 x 8 | Triton: 8x more blocks |
| VGPR / thread | 100 | 88 | similar |
| SGPR / thread | 80 | 112 | similar |
| LDS / block | **32 KB** | **0 KB** | CK uses LDS; Triton is register-resident |
| Dispatches measured | 26 | 26 | same |
| Avg duration | **192 us** | **35 us (+ 4 us reduce)** | 5x wall-clock gap |

Already two big architectural differences pop out:

1. **CK uses LDS heavily, Triton not at all.** Anything that follows
   from LDS (barriers, bank conflicts, async copies) is a CK-only cost.
2. **Triton launches 8x more blocks of 4x fewer warps.** That means
   the SQ has many more concurrent waves to schedule around stalls.

These are hints, not conclusions — Phase B explains *why* this matters.

---

## 3. Phase B — PMC counters

Hardware counters give us a quantitative split of "where did the cycles
go". On gfx950 each counter sample is one number per kernel dispatch;
we run small benchmarks (3 warmup + 6 iters) since the GPU has to
multiplex counters and is slowed slightly.

### 3.1 Counter group 1 — compute mix

```bash
rocprofv3 --pmc \
    SQ_INSTS_VALU SQ_INSTS_MFMA SQ_INSTS_LDS SQ_INSTS_SALU SQ_INSTS_VMEM \
    SQ_INSTS_VALU_CVT GRBM_GUI_ACTIVE SQ_WAVES \
  -d phase_b1 -o pmc -f csv -- python3 ../test_single_shape.py <args>
```

`aggregate_pmc.py phase_b1/pmc_counter_collection.csv --warmup 3`
aggregates the long-format CSV by kernel/counter and prints derived
ratios.

What we got:

```
Counter            |        CK |    Triton | CK/Triton
GRBM_GUI_ACTIVE    | 4,153,778 |   805,548 | 5.16x    ← matches wall-clock gap
SQ_WAVES           |       320 |     2,560 | 0.125x   ← Triton has 8x more waves
SQ_INSTS_MFMA      |   524,288 |   196,608 | 2.67x    ← CK does 2.67x more MFMA
SQ_INSTS_VALU      | 8,672,064 | 3,985,536 | 2.18x
SQ_INSTS_LDS       |   590,848 |   399,360 | 1.48x
SQ_INSTS_VMEM      |   269,824 |   193,664 | 1.39x
SQ_INSTS_VALU_CVT  |   274,048 |   131,072 | 2.09x

%CVT / %VALU       |     3.16% |     3.29% | ≈equal   ← FP8 cast isn't the gap
%MFMA / total inst |     4.70% |     3.59% |
```

Two big takeaways:

- **Triton runs ~8x more waves concurrently** (`SQ_WAVES` 2560 vs 320).
  Waves are the unit the SQ schedules around stalls; more waves =
  better latency hiding.
- **CK retires 2.18-2.67x *more* instructions** for the same math.
  Same problem, same flop budget, so CK is doing more *redundant or
  wider* work per useful op. This is consistent with 8 warps/block
  sub-tiling the same KV problem with more bookkeeping per warp.
- **`%CVT/VALU` is the same in both backends.** Our FP8 cast +
  `ds_bpermute_b32` lane swap (attempt 1) is *not* the gap; the cast
  fraction is ~3% on both sides.

### 3.2 Counter group 2 — stalls and memory

```bash
rocprofv3 --pmc \
    SQ_WAIT_INST_LDS SQ_WAIT_INST_ANY SQ_WAIT_ANY GRBM_GUI_ACTIVE \
    TCP_PENDING_STALL_CYCLES_sum TA_BUSY_avr TCC_BUSY_avr SQC_TC_STALL \
  -d phase_b2 ...
```

```
Counter                       |        CK |    Triton
SQ_WAIT_INST_ANY              | 8,342,731 | 1,339,577   ← CK has 6.2x more issue stalls
SQ_WAIT_INST_LDS              |   553,450 |   161,661
SQ_WAIT_ANY (sum over waves)  |11,120,502 |14,467,350   ← similar total
TCP_PENDING_STALL_CYCLES_sum  | 7,717,571 | 7,772,167   ← equal! VMEM behaviour is the same
TA_BUSY_avr                   |    10,543 |    13,546   ← TA slightly busier on Triton
TCC_BUSY_avr                  |   262,239 |    57,269   ← CK pushes L2 4.6x harder per dispatch
```

How to read it:

- **VMEM stall (`TCP_PENDING_STALL_CYCLES`) is equal** between the two.
  Both backends issue the same global memory traffic at the same
  latency. CK is *not* memory-throughput bound.
- **`SQ_WAIT_INST_ANY` is 6.2x worse for CK** — these are cycles where
  waves were stalled at the issue stage. With 8x fewer waves in flight
  (Phase B.1), each wave eats the stall instead of being scheduled
  around it.
- **`SQ_WAIT_ANY` is similar in absolute terms.** Triton has more total
  wait cycles, but distributes them across 8x more waves, so the
  *fraction of any wave's lifetime spent waiting* is much smaller.
- **`TCC_BUSY_avr` 4.6x higher in CK** is just a side-effect of CK
  finishing 5x slower — the L2 sees the same traffic spread over more
  cycles.

So the bottleneck is *not* memory throughput nor instruction count; it
is **the lack of concurrent waves to hide intra-kernel stalls**.

That points us at synchronization. Phase C confirms which
synchronization.

---

## 4. Phase C — PC sampling

PMC counters tell you *how much* you spent on each kind of work; PC
sampling tells you *where in the instruction stream* the kernel was
parked at sample time. On gfx950 only `host_trap` PC sampling is
supported in `time` units (stochastic is available too but only on
power-of-2 cycle intervals).

```bash
rocprofv3 --pc-sampling-beta-enabled \
    --pc-sampling-unit time --pc-sampling-interval 100 \
    --pc-sampling-method host_trap --kernel-trace \
  -d phase_c -o pcsamp -f rocpd csv json \
  -- python3 ../test_single_shape.py <args> --warmup 2 --iters 200
```

The CSV gives `Sample_Timestamp, Exec_Mask, Dispatch_Id, Instruction,
Instruction_Comment, Correlation_Id` but **no PC offset**. To get the
PC you must read the JSON file's
`buffer_records.pc_sample_host_trap[].record.pc.{code_object_id,code_object_offset}`.
The `pc_hotspots.py` helper does exactly that.

### 4.1 Category breakdown (CK kernel, 2671 recognized samples)

```
BARRIER  (s_barrier)           402  15.05%   ← largest single category
SALU                           330  12.35%
VALU_OTHER                     984  36.84%
VMUL (v_pk_mul_f32/v_mul_f32)  181   6.78%
VEXP (softmax exp)             166   6.21%
MFMA (f32_32x32x16_fp8_fp8)    126   4.72%
VMEM                           126   4.72%
WAIT (s_waitcnt)               118   4.42%
FMA                             96   3.59%
LDS_OTHER                       56   2.10%
CVT                             45   1.68%
LDS_TR (ds_read_b64_tr_b8)      41   1.54%
```

Compare to Triton (recognized samples in `kernel_unified_attention_3d`):

```
WAIT (s_waitcnt)              1788  53.48%   ← parallel waves parked waiting
VALU                           749  22.41%
SALU                           306   9.15%
BARRIER                        265   7.93%
LDS                             92   2.75%
VMEM                            80   2.39%
MFMA (16x16x32_fp8_fp8)         40   1.20%
CVT                             23   0.69%
```

The contrast is the whole story:

- **CK spends 15% of GPU time at `s_barrier` instructions.** Triton
  spends 8%.
- **Triton's bottleneck is `s_waitcnt` (53%).** That isn't bad — it
  means Triton has so many waves in flight that several are always
  parked on `s_waitcnt vmcnt(*)` waiting for outstanding loads while
  *other* waves are issuing real work. The SQ trivially schedules
  around it. CK doesn't have enough waves to amortise that.
- **MFMA is only 4.7% of CK time and 1.2% of Triton time.** Neither
  kernel is compute-bound on MFMA throughput. The MFMA shape difference
  (32x32x16 vs 16x16x32) is not the gap by itself.

### 4.2 Hottest individual PCs

`pc_hotspots.py phase_c/pcsamp_results.json --top 8 --context 4`:

```
HOT [cobj=12 off=0x806c]  hits=78 (2.92%)   s_barrier
    pre:  v_pk_mul_f32 v[42:43], v[42:43], v[130:131]    ← 16 packed muls
          v_pk_mul_f32 v[52:53], v[52:53], v[130:131]       (softmax rescale by alpha)
          v_pk_mul_f32 v[54:55], v[54:55], v[130:131]
          v_pk_mul_f32 v[58:59], v[58:59], v[130:131]
          v_pk_mul_f32 v[64:65], v[64:65], v[130:131]
    ==>   s_barrier                                       ← phase boundary
    post: s_mov_b64 / v_readfirstlane / s_lshr / s_mulk_i32  ← LDS write addr calc
          buffer_load_dword v66, s[28:31], 0 offen lds    ← async V tile -> LDS

HOT [cobj=12 off=0x7940]  hits=73 (2.73%)   s_barrier
    pre:  ds_bpermute_b32 v167, v179, v110               ← FP8 lane-pair swap
          v_cndmask_b32_e64
          ds_bpermute_b32 v168, v179, v110
          v_mul_f32_e32                                  ← softmax rescale
    ==>   s_barrier
    post: ... (same async V/K LDS prefetch pattern)

HOT [cobj=12 off=0x83f0]  hits=70 (2.62%)   s_barrier
    pre:  same ds_bpermute_b32 + v_mul_f32 pattern
HOT [cobj=12 off=0x75b4]  hits=69 (2.58%)   s_barrier
```

All four hottest PCs are the **same kind of barrier**: between the
post-softmax rescale + (optional fused FP8 swap) and the next async
global→LDS K/V tile prefetch.

### 4.3 Mapping PC offset → C++ source

The four hot offsets sit inside `_ZN7ck_tile6kentry..UnifiedAttentionKernel..`
in `module_unified_attention.so.16.hipv4-amdgcn-amd-amdhsa--gfx950`
(this is the gfx950 ELF whose `load_size=39976` matches `code_object_id=12`
in the PC-sampling JSON).

To resolve a PC to the source:

```bash
# 1. find the gfx950 ELF embedded in the .so (one per shape variant)
ls /root/aiter/aiter/jit/module_unified_attention.so.*-amd-amdhsa--gfx950

# 2. match by size to the rocprof code object
python3 -c "
import json
data = json.load(open('phase_c/pcsamp_results.json'))['rocprofiler-sdk-tool'][0]
for c in data['code_objects']:
    print(c['code_object_id'], c.get('load_size'), c.get('uri'))
"

# 3. disassemble and search for the hex address (PC offset)
/opt/rocm/llvm/bin/llvm-objdump -d --arch-name=amdgcn --mcpu=gfx950 \
    module_unified_attention.so.16.hipv4-amdgcn-amd-amdhsa--gfx950 \
    > kernel.s
grep -nE '00000000(806C|7940|83F0|75B4)' kernel.s
```

For the CK FP8 path, the disassembly at offset `0x806c` is preceded by
exactly 16 consecutive `v_pk_mul_f32 v[X:X], v[X:X], v[130:131]`
(softmax-rescale of the PV-C accumulator by the alpha scaling broadcast
in `v[130:131]`) and followed by a `buffer_load_dword … offen lds`
(async V-tile load into LDS). That is the **`__builtin_amdgcn_s_barrier()`
in the 4-phase ping-pong loop body** in
`composable_kernel/include/ck_tile/ops/unified_attention/pipeline/unified_attention_pipeline.hpp`
at line ~1276 (and its three siblings at ~1227, ~1310, ~1337). Each of
the 4 ping-pong phases ends with an explicit `s_barrier` to fence the
LDS write done by half of the warps against the LDS read about to be
issued by the other half.

If you want exact line numbers instead of "approximate", rebuild the
relevant kernel TU with `-gline-tables-only` (only the line-number
DWARF section, no full debug info — keeps the binary small) and use
`llvm-objdump -d --source` or `llvm-addr2line -e kernel.elf 0x806c`.
The build system invocation lives in `aiter/jit/core.py`; passing
`extra_cuda_cflags=['-gline-tables-only']` to the JIT compile step is
enough.

---

## 5. The bottleneck, in one paragraph

The CK kernel is **synchronization-bound, not compute or memory bound**.
It runs 8 warps per block sharing a 32 KB LDS buffer in a 4-phase
ping-pong: one half of the warps stores K/V into LDS while the other
half consumes the previously prefetched tile via QK→softmax→PV. Each
phase ends with an `__builtin_amdgcn_s_barrier()` so the LDS write
finishes before the next LDS read. With 8 warps per block, the
slowest warp's outstanding VMEM ops hold the barrier, exposing
`SQ_WAIT_INST_ANY` cycles. Triton sidesteps the problem by going
**register-resident**: 0 LDS, 2 warps/block, 8x more blocks → the SQ
trivially schedules around stalls, so even though `s_waitcnt` is 53%
of Triton's PC samples, those parked waves cost no wall time. The
~5x gap is the price of CK's LDS-cooperative pipeline at this
sequence length.

This also explains why **attempt 2** (removing the FP8 lane-pair swap
by using a custom non-pack-Double layout) didn't help: the swap was
fused into the cast in attempt 1 already, and the barrier wait is *not*
caused by the swap — it's caused by the slowest of 8 warps reaching the
phase boundary. Removing the swap saves a few cycles inside one warp,
which then arrives at the barrier earlier and waits longer.

## 6. Where to optimise next

Listed by likely yield, with the supporting metric:

1. **Reduce warps per block** (from 8 to 4) — directly halves the
   number of waves the barrier must serialise on. Requires policy and
   tile-shape changes and may force more dispatches per problem. The
   `SQ_WAVES` ratio (8x for Triton) is the headline number predicting
   the gain.
2. **Persistent kernel with multiple KV tiles in flight per block** —
   more outstanding async LDS loads → cover more memory latency without
   widening the LDS footprint. Watch `TCP_PENDING_STALL_CYCLES`; if it
   is already saturated, this gives less than expected.
3. **Eliminate or reduce one of the 4 phase barriers** by extending
   the LDS double-buffer to triple-buffer (one prefetch ahead). Costs
   1.5x LDS, may force VGPR pressure up — check `LDS_Block_Size` and
   `VGPR_Count` in the Phase A trace before committing.
4. **`async_load_tile_raw_long` → `global_load_lds_dwordx8`** for V/K
   prefetch — fewer instructions to enqueue, possibly fewer barriers
   needed. Already alluded to in source comments around line 651-693.
5. **Preshuffle V on the host (attempt 3 originally proposed)** — only
   helps the `LDS_TR` 1.54% category; will not close a 5x gap on its
   own and adds significant integration cost.

The Phase A/B/C numbers should be re-collected after each attempt and
compared (per-counter ratios in the table at section 3.1, hot-PC
distribution in section 4.2). The `aggregate_pmc.py` + `pc_hotspots.py`
helpers were written so each follow-up experiment can be summarised in
under a minute.

---

## 7. Files in this directory

| Path | Purpose |
| --- | --- |
| `run_profile.sh` | runs the three phases for one shape |
| `aggregate_pmc.py` | aggregates `pmc_counter_collection.csv` per kernel + counter; computes derived ratios |
| `pc_hotspots.py` | parses the rocprof JSON, buckets PC samples per offset, prints hot PCs with neighbour disassembly context |
| `BOTTLENECK_ANALYSIS.md` | this document |

The wrapper script writes everything under `runs/<timestamp>/`. The
result directories are intentionally not committed.

---

## 8. Follow-up: FP8 long-context decode (`b=128 sq=1 sk=128000 d=64`)

The prior sections used a small prefill shape (`b=4 sq=8`). A separate
investigation against a realistic long-context decode shape uncovered a
second, totally different bottleneck — **FP8 was issuing 2x more
async-load instructions than BF16 for the same byte volume.**

Reproduce:

```bash
cd ua-test-scripts/rocprof_analysis
OUT=runs/bf16_d64 ./run_profile.sh -b 128 -sq 1 -sk 128000 -hq 64 -hk 8 -d 64 \
                    --num-blocks 12000 --block-size 32 --dtype bf16
OUT=runs/fp8_d64 ./run_profile.sh -b 128 -sq 1 -sk 128000 -hq 64 -hk 8 -d 64 \
                    --num-blocks 12000 --block-size 32 --dtype fp8
```

### 8.1 Symptom

CK BF16 was *faster* than CK FP8 on this shape, opposite of Triton. Per
`SQ_INSTS_*` (median over 6 post-warmup dispatches):

| Counter | BF16 | FP8 (before fix) | Δ FP8 vs BF16 |
|---|---:|---:|---:|
| GRBM_GUI_ACTIVE | 116.5 M | 144.3 M | +24% |
| SQ_INSTS_VMEM | 65.5 M | **131.1 M** | **+100%** |
| SQ_INSTS_VALU | 1.34 B | **2.21 B** | **+65%** |
| SQ_INSTS_MFMA | 32.8 M | 32.8 M | same |
| TCC_BUSY_avr (L2) | 13.1 M | 17.2 M | +32% |

PC-sample category shift: BF16 spends 27% in WAIT and 45% in
VALU_OTHER; FP8 spends 15% in WAIT and **55%** in VALU_OTHER. The FP8
kernel is **doing more work** rather than waiting more.

### 8.2 Root cause: the FP8 `GetAlignmentK`/`GetAlignmentV` blanket cap

`GetAlignmentK` previously returned `4 B/lane` (one `dword`) for every
FP8 tile, regardless of `kBlockSize`, citing the static_assert in
`amd_buffer_addressing_builtins` (only `dword`/`dwordx3`/`dwordx4` are
supported on gfx950 LDS-direct loads) and a NumIssues=0.5 case in the
8-warp prefill variants. That reasoning over-fitted to prefill.

The actual constraint is per-tile:

```
NumIssues = (kPageBlockSize * kHeadDim) / (kBlockSize * KVector_elems)
```

`KVector_elems = 16` (`dwordx4` for FP8) works whenever `tile_bytes`
is a multiple of `kBlockSize * 16`. For the seven decode/prefill
variants currently compiled:

| Variant | `kBlockSize` | `tile_elems` | NumIssues @ 16 B | Decision |
|---|---:|---:|---:|:---|
| `prefill_d128` | 512 | 32x128 = 4096 | 0.5 | fall back to dword |
| `prefill_d64`  | 512 | 64x64 = 4096 | 0.5 | fall back to dword |
| `decode_d128_m128` | 256 | 4096 | 1 | **use dwordx4** |
| `decode_d128_m32`  |  64 | 4096 | 16 | **use dwordx4** |
| `decode_d128_m16`  |  64 | 4096 | 16 | **use dwordx4** |
| `decode_d64_m128`  | 256 | 4096 | 1 | **use dwordx4** |
| `decode_d64_m64`   | 128 | 4096 | 2 | **use dwordx4** |
| `decode_d64_m16`   |  64 | 4096 | 16 | **use dwordx4** |

The fix in `unified_attention_pipeline_default_policy.hpp`
(`GetKVAlignmentBytes<>`) picks `dwordx4` whenever it tiles cleanly and
falls back to `dword` for the prefill tier. BF16/FP16 paths are
unchanged.

### 8.3 Result on the reference shape

PMC re-collected with the same script after the fix:

| Counter | FP8 (before) | FP8 (after) | Δ |
|---|---:|---:|---:|
| GRBM_GUI_ACTIVE | 144.3 M | **87.2 M** | **-40%** |
| SQ_INSTS_VMEM | 131.1 M | **32.8 M** | **-75%** (4x fewer issues) |
| SQ_INSTS_VALU | 2.21 B | **0.74 B** | -67% |
| SQ_INSTS_SALU | 146.7 M | 73.0 M | -50% |
| TCC_BUSY_avr | 17.2 M | 10.1 M | -41% |
| SQ_WAIT_INST_ANY | 441.6 M | 203.3 M | -54% |

Wallclock on the reference shape (b=128 sq=1 sk=128000 d=64):

| | CK FP8 (before) | CK FP8 (after) | CK BF16 | Triton FP8 |
|---|---:|---:|---:|---:|
| ms | 7.17 | **4.57** | 6.06 | 7.44 |

CK FP8 is now **35% faster than before** and the BF16/FP8 ordering
matches expectation (FP8 < BF16, since BF16 still uses the same 16 B
loads but moves 2x the bytes per element). On the broader decode sweep
the speedup ranges 36-69% across `b ∈ {4, 128, 256}`, `d ∈ {64, 128}`,
and prefill (which keeps the `dword` fall-back) is unchanged.

### 8.4 Takeaways for future profiling

- **Always sanity-check the issue count, not just bytes-moved.**
  `SQ_INSTS_VMEM` and `SQ_INSTS_VALU` were the headline counters here;
  the load-width misconfiguration was invisible from `TA_BUSY` or
  `TCC_BUSY` alone.
- **Compare FP8 against BF16 explicitly** on every shape — the 2x ratio
  was an immediate red flag because FP8 should issue *fewer*, not more,
  load instructions for the same data.
- The original "FP8 needs dword on gfx950" comment was correct *for the
  prefill tier* but over-applied. Per-variant analysis at compile time
  is the right granularity, not per-dtype.

---

## 9. Follow-up: prefill `d=128 fp8` page-table optimisations

Reference shape: `b=1 sq=sk=75600 hq=64 hk=8 d=128 page_size=32 fp8`.
This is the long-context single-batch prefill that lands on the 8-warp
`prefill_d128` tier with `KY0_step_N=16`. CK baseline: **142.0 ms**.

### 9.1 Tier 0 — scalar-promote `block_tables[]` lookup

`refresh_*_offsets(tile_idx)` originally emitted 64 redundant per-lane
`global_load_dword`s per warp per K/V tile to materialise the *same*
`block_tables[block_table_offset + logical_page]` — `logical_page` is
uniform across the warp under the existing `Y0_step_N | page_size`
precondition. Wrapping the index in `__builtin_amdgcn_readfirstlane`
collapses those 64 VMEM loads to a single per-warp `s_load_dword`
through the scalar L1.

Gated constexpr-only on
`(kBlockSize >= 8*WarpSize) && (KNRepeat >= 2) && (KY0_step_N <= 16)`
to keep correctness on the prefill tiers where the per-iter stride
*could* exceed runtime `page_size` (a runtime gate was tried and
regressed 30% due to dual-path code bloat).

Result on the reference shape: **142.0 → 125.4 ms (-11.5%)**. No
regressions on the decode sweep or on the BF16/d=64 prefill variants
that gate out.

### 9.2 Tier 2 — LDS-resident page-table cache

After Tier 0 the `s_load_dword` is one per K/V tile per warp, but the
PC-sampling profile (`runs/tier0_prefill_d128_fp8/`) shows it still
contributes ~8.5% of total samples through `s_waitcnt vmcnt(0)`
stalls. Tier 2 folds every per-tile load into a single cooperative
bulk load at kernel entry, staging this CTA's `block_tables` slice
into a dedicated LDS region:

- **Capacity**: `kPageTableLdsEntries = 4096` × 4 B = 16 KiB. With
  prefill_d128 fp8 at ~32 KiB current LDS and 80 KiB/block budget at
  `kBlockPerCu=2`, this fits without forcing occupancy down (verified
  via embedded ELF metadata: post-Tier-2 kernel reports 48 KiB
  `group_segment_fixed_size`, 196 VGPRs — both well inside the
  budget). At `page_size ∈ {16, 32, 64}` this covers
  `sk ≤ {64, 128, 256} K` tokens — comfortably above any realistic
  prefill. Larger sequences trip an `assert` at kernel entry; the
  runtime fallback was tried and regresses 30% for the same reason
  Tier 0 needed compile-time gating.
- **Populate**: cooperative `for (i = tid; i < num_pages; i += kBlockSize)`
  followed by a single `s_barrier`. Coalesced within a warp; overlaps
  with the Q tile load at kernel entry.
- **Read**: `refresh_*_offsets` does
  `int32_t phys_page = block_tables_lds[i_base_page];`. **No outer
  `readfirstlane` on the LDS read result** — the ds_read with uniform
  address is already a broadcast, and keeping `phys_page` in VGPRs
  lets the downstream `phys_page * page_size + within_page` chain
  stay per-lane VALU instead of forcing it through a scalar
  bottleneck. (Adding `readfirstlane` here was tried first and made
  Tier 2 a 2% net **regression** vs Tier 0; removing it produced the
  -5% win below.)
- **Gating**: same constexpr predicate as Tier 0 — Tier 2 only
  matters when refresh_*_offsets actually uses the scalar-promoted
  path; otherwise the LDS read would just slot into the same per-lane
  critical chain that Tier 0 sidesteps.

Result on the reference shape: **125.4 → 119.0 ms** (additional
**-5.1%**), so cumulative **142.0 → 119.0 ms = -16.2%** with both
tiers. PC-sampling category breakdown shifts:

| category | Tier 0 only | Tier 0+2 | Δ |
| --- | ---: | ---: | ---: |
| WAIT | 8.54% | **4.49%** | -4.0 pp (s_load_dword waits gone) |
| LDS_OTHER | 2.93% | 4.65% | +1.7 pp (new ds_read_b32 reads) |
| BARRIER | 19.72% | 19.36% | unchanged (cross-warp sync) |
| SALU | 21.57% | 22.28% | +0.7 pp (extra address comp) |
| VALU_OTHER | 22.51% | 24.39% | +1.9 pp (per-lane phys_page math) |

The net 2.3 pp drop in stalls maps to the measured ~5% runtime win.
Barriers (~19%) and the address-comp chain (~46%) now dominate; both
are intrinsic to the cross-warp pipelining structure and won't be
moved without restructuring the pipeline phases themselves.

### 9.3 Why Triton can't reach Tier 2

Triton's tensor model treats LDS as a statically-shaped tile and does
not expose dynamic per-thread indexing into it. The Tier 2 fast path
needs a per-lane index expression that lowers to `ds_read_b32` with a
broadcasted scalar address — only available at the HIP/CK abstraction
level. This is genuinely a CK-only optimisation, and it is the
largest contributor to closing the prefill gap with Triton-3D
(76.9 ms Triton vs 119.0 ms CK; before tier 0+2 it was 76.9 vs 142.0).

### 9.4 Tier 3 — constexpr page_size

After Tier 0+2 was enabled the runtime `page_size` argument was still
threaded through the per-tile arithmetic (`/ page_size`, `* page_size`,
`% page_size`). With a fully runtime value the compiler can't strength-
reduce those, and the Tier-0 / Tier-2 gate had to use a conservative
`KY0_step_N <= 16` hedge (the smallest production page size) rather
than the *real* precondition `KY0_step_N <= page_size`, leaving three
of the four prefill instances on the slow page-index path.

Tier 3 makes `page_size` a non-type template parameter (`kPageSize_`)
on `UnifiedAttentionPipeline` and threads it through
`unified_attention_kernel_traits` and `dispatch_variant<V>`. The host
dispatcher switches on `args.page_blk_size` ∈ {16, 32, 64} and routes
to a constexpr-pinned instance; values outside that menu (or any
decode variant) fall back to the existing `kPageSize_=0` runtime-page-
size instance. 36 new prefill instances are compiled (6 dtype-mask
combinations × 2 head dims × 3 page sizes).

Two wins fold together:

1. **Strength-reduction**. Every `/ ps`, `* ps`, `% ps` collapses to a
   literal-folded shift / multiply-by-magic (e.g. `/ 32` → `shr 5`).
2. **Wider gate**. The Tier-0 / Tier-2 predicate is now `KY0_step_N <=
   kPageSize` at compile time, which fires on three additional prefill
   instances at their natural page sizes.

Measured impact (sq=sk=75600, MI355, n=30 iters, mask=false, GQA-8):

| variant            | KY0_step_N | page_size | before | after  | Δ       | notes                                 |
|--------------------|-----------:|----------:|-------:|-------:|--------:|---------------------------------------|
| prefill_d128 fp8   | 16         | 32        | 119.0  | 111.5  | -6.3 %  | Tier 0+2 was already on; strength-red |
| prefill_d128 bf16  | 32         | 32        | 132.7  | 130.3  | -1.8 %  | Tier 0+2 newly on                     |
| prefill_d64  fp8   | 32         | 32        |  80.9  |  68.1  | -15.8 % | Tier 0+2 newly on — biggest win       |
| prefill_d64  bf16  | 64         | 64        |  74.4  |  73.4  | -1.3 %  | Tier 0+2 newly on                     |

The big d=64 fp8 win is because the slow PageSize=0 path on
`KY0_step_N=32` was both (a) doing per-lane page-index loads (no Tier
0) *and* (b) doing runtime divs/muls on `page_size` — Tier 3
collapses both at once. d=128 bf16 sees a smaller gain than the d=64
fp8 case because it is more compute-bound (more FLOPs per byte
loaded), so address-comp / page-table-load savings matter
proportionally less.

Decode variants stay on `kPageSize_=0` instances — the Tier-0 / Tier-2
gate gates out anyway (< 8 warps) and the binary-size cost isn't
justified.

### 9.5 Closing the Triton gap (status)

Post-Tier-0+2+3 CK numbers vs the same Triton baseline:

|                    | Triton | CK before | CK after | gap         |
|--------------------|-------:|----------:|---------:|------------:|
| prefill_d128 fp8   | 77     |  142      | 111.5    | -45 %       |
| prefill_d64  fp8   | 61     |   91      |  68.1    | gap closed  |
| prefill_d128 bf16  | 119    |  133      | 130.0    | -9 %        |
| prefill_d64  bf16  | 74     |   75      |  73.4    | gap closed  |

d=64 prefill (both dtypes) is at or below Triton. d=128 still trails;
the remaining gap is mostly inside the address-comp / barrier chain
that section 9.2 already flagged — not the page-table fetch.

### 9.6 Tier 4 — optional paging (contiguous / THD K/V layout)

Tier 0–3 attack the paged-KV-cache fast path. There is a separate
class of callers (pretraining, flash-attention-style inference without
a shared cache, microbenchmarks) that pass K/V as a flat
`[num_kv_tokens, num_kv_heads, head_dim]` tensor and have no
`block_tables` at all. For that case the entire per-tile page-table
fetch chain — `block_tables_ptr_[block_table_offset + logical_page]`,
the `/ % page_size` arithmetic, the Tier-0 scalar-promote, and the
Tier-2 LDS-cache populate — is dead weight.

Tier 4 adds a `bool kEnablePaging_` non-type template parameter on
`UnifiedAttentionPipeline` (default `true` to preserve the paged
behaviour) and a `bool args.kv_contiguous` runtime selector. The host
dispatcher routes the contiguous request to a `kEnablePaging_ = false`
instance whose `refresh_*_offsets` collapses to a single per-row
`logical_token * row_stride` `imad`. Twelve new prefill instances are
compiled (`prefill_d{64,128}` × `{fp16, bf16, fp8}` × `{mask, nmask}`)
— decode variants don't need it (callers without a KV cache don't have
decode workloads).

Measured impact on the same physical memory (sq=1×4096, sk varied,
page_size=32 paged baseline, causal, MI355, n=30 iters):

| variant           | sk    | paged (ms) | contig (ms) | Δ        |
|-------------------|------:|-----------:|------------:|---------:|
| prefill_d64  bf16 |  4096 |     0.274  |     0.227   | -17.1 %  |
| prefill_d64  bf16 | 16384 |     1.529  |     1.198   | -21.6 %  |
| prefill_d64  bf16 | 32768 |     3.218  |     2.505   | -22.1 %  |
| prefill_d64  fp8  |  4096 |     0.299  |     0.235   | -21.4 %  |
| prefill_d64  fp8  | 16384 |     1.489  |     1.150   | -22.7 %  |
| prefill_d64  fp8  | 32768 |     3.054  |     2.386   | -21.9 %  |
| prefill_d128 bf16 |  4096 |     0.493  |     0.397   | -19.3 %  |
| prefill_d128 bf16 | 16384 |     2.638  |     2.224   | -15.7 %  |
| prefill_d128 bf16 | 32768 |     5.731  |     4.598   | -19.8 %  |
| prefill_d128 fp8  |  4096 |     0.476  |     0.341   | -28.3 %  |
| prefill_d128 fp8  | 16384 |     2.416  |     1.792   | -25.8 %  |
| prefill_d128 fp8  | 32768 |     4.973  |     3.727   | -25.0 %  |

`prefill_d128 fp8 -28 %` is the biggest single CK-UA optimisation
ever measured on this kernel: it eclipses Tier 0 (-12 %), Tier 2 (-5 %)
and the d=64 fp8 Tier-3 win (-16 %) in *relative* terms, and beats the
Tier-3 d=64 fp8 win in absolute ms too on equal-sk shapes.

The contiguous path is opt-in via `kv_contiguous=True`; callers that
need a shared paged KV cache (vLLM/SGLang) keep using the paged
instances and pay nothing. Correctness validated by bit-exact
comparison against the paged instance with page_size=32 and an identity
block_tables on 48 shape × dtype × mask combinations.
