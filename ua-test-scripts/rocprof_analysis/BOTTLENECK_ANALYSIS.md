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
