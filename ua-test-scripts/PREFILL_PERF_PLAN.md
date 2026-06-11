# CK Unified Attention — Prefill Performance Plan

> **Living document.** Update the *Progress log* and *Baseline measurements*
> sections as work lands. The agent should re-read this file at the start of
> each prefill-perf work session and keep the status markers current.

Last updated: 2026-06-09

---

## 0. Purpose

Track the prefill-path performance work for the CK unified-attention (UA)
kernel. Two concrete goals (below), a fixed canonical shape, and a recorded
baseline so we can see movement run-over-run.

---

## 1. Canonical shape (use for ALL further prefill tests)

```
batch (b)        = 1
seqlen_q (sq)    = 75600
seqlen_k (sk)    = 75600
num_q_heads      = 16
num_kv_heads     = 2      (GQA-8)
head_size (d)    = 128
block_size       = 64
dtype            = bf16
num_blocks       = auto   (= b * ceil(sk/block_size) * 2)
```

Hardware of record: **gfx950 (MI355X)**, 256 CUs, 294 GB HBM.

### Repro commands

CK UA, paged (default leg):
```
python3 op_tests/test_unified_attention_ck.py \
  -b 1 -sq 75600 -sk 75600 --num-heads 16,2 --head-size 128 \
  --block-size 64 --dtype bf16 --num-blocks auto --no-triton --no-reference
```

CK UA, contiguous (`--contiguous` flips the single CK leg to is_paged=False):
```
python3 op_tests/test_unified_attention_ck.py \
  -b 1 -sq 75600 -sk 75600 --num-heads 16,2 --head-size 128 \
  --block-size 64 --dtype bf16 --num-blocks auto --contiguous --no-triton --no-reference
```

CK contiguous + Sage v1 (non-causal), the Goal-2 head-to-head:
```
python3 op_tests/test_unified_attention_ck.py \
  -b 1 -sq 75600 -sk 75600 --num-heads 16,2 --head-size 128 \
  --block-size 64 --dtype bf16 --num-blocks auto \
  --contiguous --sagev1 --no-triton --no-reference --mask-type 0
```

Triton UA (paged) comparison — **causal only** (its kernel asserts `causal`):
```
python3 op_tests/test_unified_attention_ck.py \
  -b 1 -sq 75600 -sk 75600 --num-heads 16,2 --head-size 128 \
  --block-size 64 --dtype bf16 --num-blocks auto --triton --no-reference
```

> Reference (torch) is skipped at this size — it needs a ~23 GB per-head score
> matrix. Correctness is validated separately at smaller shapes (≤4096).

---

## 2. Baseline measurements

> Median of `@perftest` iters. TFLOPs uses the harness FLOPs model (causal =
> ~½ the dense pair count, non-causal = full n²), so **causal and non-causal
> numbers are NOT directly comparable** — compare within a column.

### Causal (mask_type=2) — 2026-06-09

| Backend            | KV tensors  | Time (ms) | TFLOPs | Notes                    |
|--------------------|-------------|-----------|--------|--------------------------|
| CK UA              | contiguous  | 27.20     | 861    | single-launch            |
| Triton UA          | paged       | 30.62     | 764    | causal-only kernel       |
| CK UA              | paged       | 34.52     | 678    | num_splits=1             |
| Sage v1            | contiguous  | —         | —      | **crashes** (see §6)     |

Derived:
- **Paging overhead** = CK paged / CK contiguous = **1.27×** (~7.3 ms, ≈21 % of
  the paged runtime is address-computation / page-table work).
- **CK contiguous vs Triton paged** = **1.13×** (CK wins). This is the key
  signal: strip our paging and we already beat Triton — so the paged address
  computation is the prime suspect for why paged CK loses to Triton.

### Non-causal (mask_type=0) — 2026-06-09

> **Honest comparison is fp8.** Sage v1 is effectively an 8-bit kernel (Int8 QK,
> FP8 V), so the apples-to-apples CK leg is **fp8**, not bf16. The bf16 row is
> kept only to show the dtype-mismatched figure.

| Backend            | KV tensors  | dtype          | Time (ms) | TFLOPs | Notes                 |
|--------------------|-------------|----------------|-----------|--------|-----------------------|
| Sage v1            | contiguous  | Int8 QK / FP8 V| 32.76     | 1429   | 8-bit                 |
| CK UA              | contiguous  | **fp8**        | 40.71     | 1150   | **honest comparison** |
| CK UA              | contiguous  | bf16           | 49.24     | 951    | dtype-mismatched      |
| Triton UA          | paged       | —              | —         | —      | skipped (causal-only) |

Derived:
- **CK fp8 contiguous vs Sage** = **0.805×** → Sage **1.24×** faster. This is the
  honest **Goal-2 gap** to close (40.71 ms → ~32.8 ms).
- (For context: bf16 CK vs Sage = 0.666× / 1.50×, but that compares 16-bit
  compute against 8-bit and overstates the gap.)

---

## 3. Goals

### Goal 1 — Paged ≈ Contiguous (paging / address-computation optimization)
**Target:** paged CK UA reaches near contiguous CK UA perf (close the 1.27×
gap), which would also put paged CK ahead of Triton UA paged.

- **Rationale / signal:** with contiguous tensors we already beat Triton UA
  (1.13×, causal). The only thing we removed is the per-tile page-table fetch +
  physical-address computation. So address computation is very likely the paged
  bottleneck.
- **Status: DEFERRED.** We optimize on the contiguous path first (Goal 2), then
  return here once the non-paging bottlenecks are fixed.

### Goal 2 — Contiguous ≈ Sage v1 (non-causal) — **PRIMARY FOCUS NOW**
**Target:** CK UA contiguous reaches Sage v1 perf on non-causal. Honest 8-bit
comparison is **fp8 vs Sage**: close the **1.24× gap (40.71 ms → ~32.8 ms)**.
(bf16 CK is 1.50× off, but that's a 16-bit-vs-8-bit comparison and overstates
the real gap — track fp8 as the primary metric.)

- **Rationale:** the contiguous path has no paging overhead, so whatever is left
  between us and Sage is "the other bottlenecks" (compute scheduling, MFMA
  utilization, softmax/rescale, K/V load pipelining, occupancy, etc.). Fixing
  these on the clean contiguous path isolates them from paging.
- These same fixes are expected to lift the paged path too, making Goal 1
  easier when we return to it.

---

## 4. Strategy / sequencing

1. **Profile the contiguous path** at the canonical shape (non-causal for the
   Sage head-to-head; causal also useful). Use the contiguous kernel so paging
   overhead is out of the picture and we see the real compute/scheduling
   bottlenecks.
2. Identify + fix the top non-paging bottleneck(s) on contiguous. Re-measure
   against Sage each iteration; record in the Progress log.
3. When contiguous ≈ Sage (Goal 2 met or plateaued), **return to Goal 1**:
   optimize the paged address computation so paged ≈ contiguous.
4. Re-validate correctness at smaller shapes after each kernel change
   (`--swa-fixtures regression`, plus causal/non-causal contiguous & paged at
   ≤4096).

---

## 5. Profiling notes

Tooling: ATT (Advanced Thread Trace) via `rocprof_att_prefill.sh` +
the headless `att_analysis/` overlay (RCV-equivalent). `rocprofv3` defaults to
contiguous / fp8 / non-causal / canonical shape now; `att_analysis/run.sh` adds
single-SIMD + slim-build + minimal-iters for fast iteration.

### First contiguous-fp8 overlay (source-mapped, SE0/SIMD0, 4 inner-loop iters)

`runs/att_lt_d128_fp8_b1_sq8192_sk8192/att_analysis/overlap_simd0.png`
(shape sq=sk=8192 is steady-state-identical to canonical — same inner loop, fewer tiles).

- **Warp-group balance is healthy.** WG0 = exec 54% / stall 24% / wait 22%;
  WG1 = exec 51% / stall 24% / wait 25%. `same-phase collision = 0%`.
- **The inefficiency is latency-hiding, not contention.** `MATRIX//SOFTMAX
  overlap = 29%` — the ping-pong only hides MFMA behind the other group's
  softmax 29% of the window.
- Per-phase stall attribution:
  - `matrix` (MFMA): **78–84% stall** — MFMA waits on its operand feed, not on
    the other WG (collision 0%).
  - `memwait`: **100% stall, ~15% of cycles** — exposed global/DRAM load
    latency; the largest *memory* cost.
  - `barrier`: 100% stall, 5–8% — warpgroup sync bubbles.
  - `softmax`: large busy, 18–32% stall — VALU relatively healthy.
  - `lds`: small (20–42% stall on a few-% cycle count) — not LDS-bound.
  - `addr`: negligible — **confirms the contiguous path has no paging cost**
    (the thing Goal-1 must recover on the paged path).
- **Reading:** contiguous fp8 is DRAM/global-latency-bound with MFMA starved on
  operands, and the two-warpgroup ping-pong is under-overlapping (29%). Next
  kernel ideas: deeper KV-tile prefetch / software pipelining to hide memwait,
  and restructure so one WG's MATRIX overlaps the other's SOFTMAX more.

### Post-decoupling overlay (per-WG K/V split + early-V, CK c11722bf3) — fp8 noncausal sq8192

After committing the per-warp-group K/V decoupling (WG0->V, WG1->K), early-V
read, and per-WG offset gating, the standalone overlay shows the bottleneck
**moved, not removed**:

| state | WG0 (loads V) | WG1 (loads K) |
|---|---|---|
| `s_waitcnt` (memwait) | **836 cyc / 100% stall (~18%)** | 184 cyc |
| `s_barrier`           | 76 cyc | **1084 cyc / 100% stall (~24%)** |

- `MATRIX//SOFTMAX overlap` fell **29% -> 24%**. `same-phase collision = 0%`.
- **V-load is now the critical path; WG1 idles at the shared barrier waiting.**
  Root cause = prefetch-distance asymmetry in `iteration(pi)`: V is issued in
  slot-A `prefetch()` and consumed one phase later in slot-B `fa4_vload`
  (~1 phase of hiding), while K is consumed a full iteration later (~2 phases).
  Concentrating V onto WG0's 4 waves made the 1-phase window too short to hide
  V's DRAM latency -> exposed 836-cyc V memwait on WG0, ~1084-cyc barrier idle
  on WG1.
- **Matrix vs softmax (fp8) confirmed imbalanced:** `matrix` 84-86% stall
  (~160 cyc real MFMA) vs `softmax` 19-34% stall (~664 cyc real VALU) ->
  softmax does **~4x** the real compute. The matrix stall IS the exposed K/V
  latency (operand starvation via `lgkmcnt`), i.e. same root cause as memwait.

### Resource footprint / occupancy (gfx950, measured from the code object)

`vgpr=181  sgpr=88  lds=49152 B (48 KB)  scratch=0 (no spills)  block=8 waves`

- **Decoupling freed ZERO static resources** — A/B'd decoupled vs cooperative
  (flags off): identical 181/88/49152. The load gating is a *runtime*
  `if(k_load_active)`, not `if constexpr`, so every wave still compiles in both
  K and V address machinery and the allocator reserves for both. The +6.4%
  came from less *executed* work + DRAM balance + early-V, not a smaller footprint.
- **One block already caps the CU**: LDS 48 of 64 KB (a 2nd block needs 96 KB),
  and 181 VGPR caps ~2 waves/SIMD. A 2nd resident block needs **<=32 KB LDS AND
  <=128 VGPR simultaneously** -> not close; treat "fit a 2nd workgroup" as a
  separate large effort, not a near-term win.
- `addr` on the contiguous path stays negligible (~156-176 cyc, ~3.5%); the real
  address-compute cost is the paged path (Goal 1).

### Next-step candidates for the WG0/WG1 asymmetry (the top item)

1. **Deepen V prefetch** to match K (issue V a full iteration ahead). Cleanest,
   but needs a 3rd V buffer -> LDS 48->60 KB (near the 64 KB cap, worse occupancy).
2. **Asymmetric DRAM split**: give hard-to-hide V more loading waves and
   well-hidden K fewer (e.g. WG1 helps load V). Targets the bottleneck; softens
   the clean decoupling.
3. **Reschedule V earlier within the existing 2 buffers** (hoist `V_mem_load`
   into the prior iteration if buffer lifetime allows) before spending LDS.

### Collection-cost learnings (how to profile fast)

- **Iteration count is irrelevant to trace time.** ATT traces ONE dispatch
  (`--att-consecutive-kernels 1`); the first UA launch under rocprofv3 is
  captured. Cut `@perftest` to `AITER_PERF_WARMUP=1` / `AITER_PERF_ITERS=2`
  (floor: stats path asserts `num_iters>1`) via env — saves nothing on trace
  time but avoids 99 wasted launches.
- **Slim build** (`AITER_UA_TRACE_INSTANCES=...`, default in `att_analysis/run.sh`
  with `SLIM=1`): stubs out the other 121 UA instances (real kernel only for the
  traced one). Module `.so` 50MB→1.3MB, **compile minutes→~30s**. Big win for
  *kernel-iteration* turnaround. NOTE: a slim `.so` only has that one instance —
  other shapes fail "no matching kernel"; rebuild `SLIM=0 LINETABLES=1` to restore.
- **Trace wall-time floor ≈ 2.5 min (torch harness) is set by PyTorch, not us.**
  rocprofv3 ATT disassembles *every* loaded code object; PyTorch's HIP lib
  (rocprim/`at::cuda`, ~26MB) loads at torch CUDA-init and costs ~110s to
  disassemble. `rocprofv3` has no flag to skip non-traced code objects
  (`--att-serialize-all` is the opposite).
- **Single SIMD** (`SIMD_MASK=0x1`): trace blob 1.1GB→8.7MB, decode <1s — keep.
- Keep a paged profile too for Goal-1, to quantify address-compute at ISA level.

### Fast path: torch-free standalone driver — `standalone/` (RECOMMENDED)

To escape PyTorch's disassembly floor, trace a standalone executable that calls
the kernel with raw HIP allocations (no libtorch in the process). rocprofv3 ATT
then disassembles ~1 kernel instead of torch's 26MB lib.

- `standalone/ua_trace_main.cpp` — fills `unified_attention_args` for the
  fp8 / contiguous (`is_paged=false`) / non-causal|causal prefill instance
  exactly as the production glue does, with **random (normal) fp8 inputs** and a
  bf16 output buffer.
- `standalone/build.sh` — compiles driver + dispatcher + all instances into an
  executable, with the same codegen `-mllvm` flags as the JIT build (ISA is
  representative) + `-gline-tables-only`; every non-target instance gets
  `-DUA_STUB_INSTANCE` (links, but no device kernel). Build ≈ **18s**, exe ~600K.
- `standalone/run.sh [sq] [hq] [hk] [d] [mask] [iters]` — build + rocprofv3 ATT
  + render overlay.
- `standalone/check.sh [sq] [hq] [hk] [d] [mask]` — **accuracy check** (below).
- `standalone/perf.sh [sq] [hq] [hk] [d] [mask]` — **perf measurement** (below).

**Measured: full collect+render in ~2.0s** (app ran 0.42s under rocprofv3) vs
~156s via the torch harness — ~75× faster, same source-mapped overlay.

```
GPU=2 bash ua-test-scripts/standalone/run.sh 8192 16 2 128 0 3   # trace+overlay
GPU=2 bash ua-test-scripts/standalone/check.sh 512 16 2 128 0    # accuracy check
GPU=2 bash ua-test-scripts/standalone/perf.sh 8192 16 2 128 0    # latency/TFLOPs
GPU=2 SWEEP="2048 4096 8192 16384" bash ua-test-scripts/standalone/perf.sh  # sweep sq
```

#### Perf path — the JIT-free, always-fresh replacement for the Python harness

`PERF=1` runs warmup + a GPU-event-timed steady-state loop and reports
`lat (us) / TFLOP/s / TB/s` using the SAME cost model as the Python harness
(`flops = 4·valid_pairs·d·hq`; `mem = Q+K+V@fp8 + O@bf16`). `ROTATE=N` cycles N
independent Q/K/V copies to defeat L2 reuse (mirrors `@perftest`'s
`num_rotate_args`; use for small shapes, `1` is fine for large prefill).

**No "is the JIT stale?" ambiguity:** `build.sh` self-guards — it rebuilds iff
`REBUILD=1`, the stamp `(arch,dtype,d,mask)` mismatches, **or any `ck_tile`
header / UA source / the driver is newer than the exe** (`find -newer`).
Otherwise it no-ops in ~10ms. So `run/check/perf.sh` call it unconditionally and
can never silently measure a stale binary. The instance is selected at compile
time only by `(dtype,d,mask)`; `sq/hq/hk/batch` are runtime, so one build sweeps
all sequence lengths and only a dtype/d/mask change forces a rebuild.

**Cross-validated** against the Python harness (same kernel, same shape
sq8192/b1/16·2/d128/fp8/non-causal/contiguous): standalone **534.6 µs** (1028
TFLOP/s) vs Python `@perftest` **540.5 µs** — within ~1%.

#### Accuracy check (validates the bench setup is faithful)

`CHECK=1` recomputes `O = softmax(scale·Q·Kᵀ [+ causal mask])·V` on the host in
float, reading the SAME fp8 bytes the kernel reads, and compares. Tolerances +
pass rule mirror `test_unified_attention_ck.py`'s `checkAllclose` for fp8
(`atol=rtol=0.15`, PASS if <5% of elements exceed it). Verified **PASS, 0%
mismatch** for non-causal (sq=256/512) and causal (sq=384/512); a tight-tolerance
probe (1e-3) shows 67% mismatch, confirming the check is genuinely sensitive
(catches setup bugs) and the ~4-5% per-element gap is just the expected fp8
(e4m3 + internal P-quant) approximation. Reference is `O(hq·sq²·d)` on CPU → use
a small `sq`.

**fp8 encoding gotcha (fixed):** `ck_tile::fp8_t` is `_BitInt(8)`
(`CK_TILE_USE_CUSTOM_DATA_TYPE=0`), so float↔fp8 MUST use `ck_tile::type_convert`
(a `static_cast` truncates to int → all-zero output). Also `CK_TILE_USE_OCP_FP8`
auto-resolves to 1 only on the **device** pass for gfx950 and 0 on the **host**
pass — so `build.sh` forces `-DCK_TILE_USE_OCP_FP8=1` to make the host encode the
same e4m3fn bytes the kernel decodes. (Zeroed inputs masked both bugs; this is
exactly why the random+reference check was worth adding.)

Caveats: overlap% can differ slightly from a torch-harness trace due to window
selection — use a torch-harness trace at canonical shape if you need the exact
production window. The standalone path is independent of the JIT
`module_unified_attention.so` (it builds its own exe), so it is unaffected by the
SLIM JIT-module knob.

---

## 6. Known issues / blockers

- **Sage v1 causal crashes at long context.** A direct call to
  `fav3_sage_wrapper_func(..., causal=True, layout="bshd")` at the canonical
  shape aborts during Triton compilation:
  `llvm/ADT/Sequence.h:275: iota_range<unsigned int>: Assertion 'Begin <= End'`.
  Causal works at sq=sk ≤ 4096; non-causal works at 75600. Root cause is an
  unsigned-int underflow in the sage causal block-range math (kernel bug, not
  our wiring — reproduced with a bare direct call). `bench_fav3_sage.py` can't
  catch it because it hardcodes `causal=False`. ⇒ Goal-2 head-to-head is
  **non-causal only** for now.
- **Triton UA is causal-only** (`assert causal, "Only causal attention is
  supported"`), so it has no non-causal leg.

---

## 7. Progress log

| Date       | Change                                              | Canonical-shape result                         |
|------------|-----------------------------------------------------|------------------------------------------------|
| 2026-06-09 | Baseline recorded; plan created. Merged aiter main. | CK-ctg 27.20 ms (causal) / 49.24 ms (non-causal); Sage 32.80 ms (non-causal); Triton 30.62 ms (causal, paged) |
| 2026-06-09 | Honest fp8 (8-bit) non-causal comparison vs Sage.   | CK-ctg **fp8** 40.71 ms / 1150 TFLOPs vs Sage 32.76 ms / 1429 TFLOPs → **0.805× (Sage 1.24×)**. Gap shrinks from 1.50× (bf16) to 1.24× (fp8). |
| 2026-06-09 | First contiguous-fp8 ATT overlay + bottleneck read. | DRAM/global-latency-bound: memwait ~15% @100% stall, MFMA 78–84% stall (operand-starved), MATRIX//SOFTMAX overlap only 29%, WG balance healthy, addr negligible. Tooling: single-SIMD + slim build (50MB→1.3MB, compile→30s); trace wall-time ~2.5min floored by PyTorch code-object disassembly. |
| 2026-06-09 | Torch-free standalone trace driver (`standalone/`).  | Built a no-libtorch executable that calls the kernel directly → rocprofv3 ATT disassembles ~1 kernel, not torch's 26MB lib. Full collect+render **~2.0s** (vs ~156s), source-mapped. `standalone/run.sh 8192 16 2 128 0 3`. |
| 2026-06-09 | Random inputs + C++ accuracy check (`check.sh`).     | Reference `softmax(scale·QKᵀ[+mask])·V` from same fp8 bytes; PASS 0% mismatch (non-causal sq256/512, causal sq384/512) at fp8 tol (0.15, <5%). Caught+fixed 2 bench bugs: fp8 needs `type_convert` (not static_cast) and host needs `-DCK_TILE_USE_OCP_FP8=1` to match device e4m3fn. Bench setup validated faithful. |
| 2026-06-09 | Standalone perf path (`perf.sh`) = JIT-free perf testing. | GPU-event timed loop + TFLOPs/TB-s (Python cost model). `build.sh` self-guards (rebuilds iff stamp mismatch or any ck_tile/UA source newer than exe; else ~10ms no-op) → no stale-binary ambiguity. Cross-validated: standalone 534.6us vs Python @perftest 540.5us (~1%) on sq8192/fp8/non-causal/ctg. One build sweeps all seq lengths. |
| 2026-06-09 | Per-WG K/V decoupling + early-V + offset gating (CK c11722bf3, aiter 46196d647). | sq8192 **500.3 us / 1098 TFLOP/s** vs 534.6/1028 baseline = **+6.4%** (sweep: sq2048 67.8us, sq4096 147.8us, sq16384 1804us/1219). Overlay: memwait NOT gone — moved to WG0 V-load (836 cyc/18% exposed) while WG1 idles 1084 cyc/24% at barrier; overlap 29%->24%. matrix 84-86% stall vs softmax ~4x real compute. Resources unchanged (vgpr181/sgpr88/lds48KB, A/B identical) → decoupling freed no static footprint; still 1 block/CU (LDS- and VGPR-capped). Top next item: WG0/WG1 V-load-vs-barrier asymmetry (= matrix operand starvation). |
| 2026-06-09 | K/V load-width fix: thread per-WG load-thread count into `GetAlignmentK/V` selector (was `kBlockSize`=512 → always dword; now 4-warp group's 256 thr → 4 KB tile tiles cleanly at dwordx4). | Both load paths widen 4×: `global_load_lds_dword` 36→**9 dwordx4**, `buffer_load_dword` 36→**9 dwordx4**. VGPR **181→173** (fewer address regs), LDS/SGPR unchanged. Accuracy PASS 0% (non-causal + causal sq512). Perf **latency-neutral**: sq8192 nc ~502–511 us (vs 500.3), causal 339.6 us. Confirms kernel is memory-**latency**-bound (exposed V-load wait), not load-issue/instruction-count bound — instr-count & VGPR win is real but doesn't move wall-clock. Clean keeper; not the lever. Next lever: 32x32x16→32x32x64 fp8 MFMA (halve matrix phase) or prefetch distance. |

<!-- Append new rows above this line as work lands. Keep the §2 tables in sync
     with the latest numbers and note the delta vs the previous entry. -->
