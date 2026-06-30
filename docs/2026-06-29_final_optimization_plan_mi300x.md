# Jagged-Dense BMM Backward — Final Optimization Plan (2026-06-29, MI300X)

Planning doc spun off from the MI325X plan `2026-06-26_final_optimizations.md`,
re-grounded on the **MI300X** box (gfx942, `HIP_VISIBLE_DEVICES=6`) with **working
rocprofiler PMC counters** via `flydsl_venv` (rocprof-compute 3.4.0). It (1) reads the
fresh MI300X roofline + full counters, (2) re-confirms the **layout-algebra** audit
against the current source, and (3) proposes a phased plan tuned to the MI300X numbers.
Phase numbers here are local to this doc — **do not reference them in source** (same
rule as the other logs). Each phase has a measurement gate; re-run the configs in §7
and append a dated EXP block to the profiling log on every gate. Correctness gate
everywhere: `example_jagged_dense_bmm_bwd.py --dim {256,512}` cosine > 0.999,
**uniform + skew**, no regression at the repo default.

Kernels: `aiter/ops/flydsl/kernels/jagged_dense_bmm_bwd.py`
(`grad_jagged`; fused `grad_dense_bias` = `grad_dense_partials_kernel` [dDense+dBias
MFMA partials] + `grad_dense_reduce_kernel` + `grad_bias_reduce_kernel`).

**Source note (2026-06-29):** the in-tree shape is now `N = K = 256`
(`jagged_dense_bmm.py:29-30`), so **D=256 is the current default star** and
`SPLIT = 2 if K<=256 else 1` → **SPLIT = 2 at D=256, SPLIT = 1 at D=512**. This changes
the cost structure of the reduce passes versus the MI325X/D=512 plan (see Phase B).

---

## 1. Baseline this plan is built on (MI300X, gfx942, B=1024, Mi=7680, uniform)

From `workloads/bwd_full_d{512,256}_b1024_m7680_uniform/` (rocprof-compute 3.4.0 full
counters + empirical roofline, captured on GPU 6; per-kernel `analyze_kernel_{0,1}.txt`).

**Empirical ceilings (this box, both runs agree):** HBM **4.16 TB/s** · MALL 6.40 TB/s ·
L2 23.4 TB/s · L1 30.9 TB/s · LDS 62.8 TB/s · **MFMA bf16 473 TF/s** · FP16 836 · F8 1709 ·
VALU fp32 118 TF/s. bf16 **ridge point = 473/4.16 ≈ 114 FLOP/byte**.

> MI300X vs the MI325X baseline: lower bf16 MFMA ceiling (473 vs 591 TF/s) and HBM
> (4.16 vs 4.61 TB/s), so absolute times here are ~25–30% higher than the MI325X plan's
> for the same shape — the **shape of the bottleneck is unchanged**, only the ceilings.

### 1a. D=512 (matches the MI325X plan's North-Star shape; SPLIT=1)

| kernel | µs/call | TF/s | % MFMA peak (473) | MFMA util | occ | VGPR/AGPR | AI_hbm | % HBM BW | dominant stall |
|---|--:|--:|--:|--:|--:|---|--:|--:|---|
| `grad_dense_partials` (dDense+dBias) | **19423** | 212.4 | **44.9%** | 28.3% | **30.4%** | 96 / 128 | 82.9 | 61.5% | Issue-wait 38.7% ≈ Dep-wait 34.6% |
| `grad_jagged` (dJagged) | **15468** | 266.8 | **56.4%** | 36.0% | **8.8%** | 36 / 132 | 96.6 | 66.4% | Issue-wait 52.8% > Dep-wait 31.7% |
| `grad_dense_reduce` (fp32→bf16) | 824 | — | — | 0 | — | 4 / 4 | ~0.2 | mem-bound | mem tail |
| `grad_bias_reduce` | 3.5 | — | — | 0 | — | 4 / 4 | ~0.2 | low | negligible |

Full backward (sum) ≈ **35.7 ms**: partials **54%**, jagged **43%**, reduces ~2.3%.

### 1b. D=256 (current in-tree default; SPLIT=2)

| kernel | µs/call | TF/s | % MFMA peak (473) | MFMA util | occ | VGPR/AGPR | AI_hbm | % HBM BW | dominant stall |
|---|--:|--:|--:|--:|--:|---|--:|--:|---|
| `grad_dense_partials` (dDense+dBias) | **5236** | 197.5 | **41.7%** | 26.2% | **27.7%** | 96 / 128 | 61.9 | 76.5% | Issue-wait 37.0% ≈ Dep-wait 37.0% |
| `grad_jagged` (dJagged) | **4818** | 214.1 | **45.3%** | 25.7% | **6.4%** | 40 / 128 | 79.1 | 64.9% | Issue-wait 54.5% > Dep-wait 30.1% |
| `grad_dense_reduce` (fp32→bf16) | 250 | — | — | 0 | — | 4 / 4 | ~0.2 | mem-bound | mem tail |
| `grad_bias_reduce` | 4.1 | — | — | 0 | — | 4 / 4 | ~0.2 | low | negligible |

Full backward (sum) ≈ **10.3 ms**: partials **51%**, jagged **47%**, reduces ~2.4%.

**Roofline reading (the important part).** With *measured* HBM traffic (TCC/L2-Fabric
counters), both MFMA kernels sit **left of the ridge** (AI 62–97 < 114) — i.e. in the
**HBM-leaning region**, achieving only **42–56% of the bf16 MFMA peak** and **62–77% of
HBM BW**. They are near-balanced at the knee, not deep compute-bound. Halving D
(512→256) pushes both kernels **further left** (lower AI: partials 82.9→61.9, jagged
96.6→79.1) because the per-tile fp32 partials round-trip and dOut re-reads scale weaker
than the D² compute — so the **HBM-traffic levers matter more at D=256**, the default
shape. There is headroom on **both** axes; the biggest wins **cut HBM traffic** (push
right, toward compute) and **raise MFMA overlap / occupancy** (push up).

Occupancy limiters (from the PMC "Insufficient …" stall reasons, consistent across D):
- `grad_dense_partials`: **VGPR-capped** (~72–76% "Insufficient SIMD VGPRs", 96 VGPR +
  128 AGPR/thread) **and** LDS-capped (~71–77% "Insufficient CU LDS", 32 KB/WG → 2 WG/CU).
- `grad_jagged`: purely **LDS-capped** (~60–75% "Insufficient CU LDS"; VGPR fine at
  36–40), plus dispatch latency (Scheduler-Pipe Stall ~22%, many tiny 8-K-step WGs,
  Issue-Wait dominant at 53–55%, IPC ~0.29).

---

## 2. Layout-algebra audit (still valid against current source)

FlyDSL exposes a CuTe-style **layout algebra**: `make_layout` / `make_ordered_layout`,
`flat_divide` / `logical_divide`, `make_composed_layout` (swizzle), `partition_S/D`,
`make_fragment_{A,B,C}`, `retile`, `thr_slice`, and **`make_tiled_copy` (vectorized,
thread-mapped copies)**. Using it lets the compiler vectorize, swizzle, and map
threads→data correctly; hand-rolling linear indices + scalar `buffer_load`/`memref_store`
bypasses all of that. Line numbers below are current as of the 618-line source.

| kernel / region | layout algebra? | evidence |
|---|---|---|
| `grad_jagged_kernel` (whole) | **Yes — faithful** | `flat_divide` tiling (221-223), `partition_S/D` (240-243), `make_fragment_*`+`retile` (245-252), swizzled LDS `make_composed_layout` (234-238), vectorized `tiled_copy_g2s_A` = `UniversalCopy128b` (343-347). Only manual bit: int64 row rebasing (208-213), idiomatic. |
| `grad_dense_partials` MFMA core + epilogue | **Yes** | `make_fragment_{A,B,C}`+`retile` (463-467), `make_tiled_copy_C`+`partition_S` fp32 store (520-525), swizzled `sJ/sD` (435-446). |
| **`grad_dense_partials` global→LDS staging** | **NO — hand-rolled scalar** | **lines 483-499**: per-thread `lin/m_local/k_local` index math + `buffer_load(..., vec_width=1, dtype=bf16)` (488, 496) → **32 + 32 scalar 2-byte loads per thread per m-tile**, then scalar `memref_store` scatter into LDS (489, 497). No `TiledCopy`/`partition_*`. |
| **`grad_dense_reduce` / `grad_bias_reduce`** | **Partial — scalar granularity** | `_load_scalar`/`_store_scalar` (145-159) = `fx.slice` + `logical_divide(make_layout(1,1))` + **1-element** copy_atom; one thread per column, scalar fp32 loads over SPLIT (377, 571) and scalar bf16 store (380, 575). |

**Verdict.** The two GEMM *cores* respect the layout algebra; the **memory-movement
edges of the dDense path do not** — the partials staging and the reduces are
hand-indexed scalar copies. The staging loop is loop-coalesced across a wavefront
(consecutive lanes → consecutive addresses, so DRAM bytes are fine), but it emits
**~64 separate scalar load instructions + address arithmetic per thread per m-tile**
instead of 4+4 vectorized 128-bit loads — directly feeding the partials kernel's high
issue-/dependency-wait (combined ~73–74%) and its 26–28% MFMA utilization. This is the
intersection of "respect the layout algebra" and "the #1 measured perf lever", and it
matters **more at D=256** (SPLIT=2 doubles the partials work + the reduce traffic).

---

## 3. Opportunities, ranked by measured leverage

1. **Vectorize the partials staging via a `TiledCopy` (layout algebra).** Biggest kernel
   (51% @ D=256, 54% @ D=512); scalar loads today → issue/dep-wait bound (~73%) at
   26–28% MFMA util. `grad_jagged` already shows the target pattern (`tiled_copy_g2s_A`,
   128-bit). Helps **both** stars.
2. **`grad_jagged` K-column operand reuse.** dOut is re-streamed once per `KOUT_BLOCKS`
   (= K/BLOCK_N) K-output tile; one WG computing multiple K-tiles from a single dOut
   load cuts that traffic and amortizes the short (NRED_TILES-step) MFMA pipeline.
   jagged is Issue-Wait-bound (53–55%) at the lowest occupancy (6–9%).
3. **`SPLIT==1` direct-bf16 dDense fast path (D=512 only).** At D=512 `SPLIT=1`, so
   `grad_dense_reduce` does **no reduction** — it is a pure fp32→bf16 HBM round-trip
   (824 µs). Eliminating it also lets the partials kernel write **bf16 instead of fp32**,
   halving its output traffic. **Does not apply at D=256 (SPLIT=2)** where the reduce is
   a genuine cross-split reduction.
4. **Lift `grad_dense_partials` occupancy** (VGPR+LDS capped at ~2 WG/CU at both D):
   smaller output sub-tile / AGPR footprint sweep.
5. **Vectorize the reduce kernels + retune the transpose-store swizzle.** At D=256
   SPLIT=2 the reduce (250 µs) is a real reduction and the staging swizzle shows LDS
   bank conflicts; scalar reduces also matter more under skew.

---

## 4. Phased plan

### Phase A — Vectorize the dDense partials staging with a TiledCopy (layout algebra)
- **Why:** lines 488/496 do `vec_width=1` bf16 loads (32+32/thread/m-tile) with manual
  indices; the kernel is issue-/dependency-wait bound (~73% combined) at only 26–28%
  MFMA util and 62–77% HBM BW. `grad_jagged` already shows the target pattern
  (`tiled_copy_g2s_A`, 128-bit). **Top lever at both D=256 and D=512.**
- **Do:** express the global→LDS transpose-staging as `make_tiled_copy` + `partition_S`
  (global, vectorized 128-bit / 8×bf16 along the contiguous axis — k for J, n for dOut)
  → `partition_D` (LDS). The transpose stays on the LDS store (gfx942 has no
  `ds_read_transpose`); evaluate (a) a swizzled **vectorized** LDS store vs (b) a
  vectorized load + scalar scatter store. **Rework the fused-dBias accumulation:** a
  vectorized dOut load makes each thread own `vec_width` consecutive columns `n`, so the
  per-thread `bias_acc` becomes a small vector and the end-of-kernel LDS combine widens
  accordingly (it currently assumes 1 column/thread via `tid % DDENSE_BN`).
- **Target metrics:** VMEM-issued instrs ↓; Issue-Wait + Dependency-Wait ↓; MFMA
  Utilization ↑ (toward `grad_jagged`'s 36%); partials µs ↓.
- **Gate:** ≥1.15× on `grad_dense_partials` µs (rocprof) at **both** D=256 and D=512,
  correctness green.
- [x] vectorized staging  [x] dBias-fusion rework  [x] re-profile + EXP block.
- **RESULT (EXP-2026-06-29a): GATE FAILED — reverted.** Vectorizing the staging to
  128-bit loads is correct (cos 0.999999, both D × both regimes) but **regresses**
  `grad_dense_partials` 5425 → 8383 µs (**0.65×**) at D=256 uniform; every `dense_bias`
  bench config is slower (J-only vectorization also regressed). Root cause (ISA + A/B
  isolation): the kernel is **occupancy-starved (~2 waves/SIMD: 224 VGPR + 32 KB dynamic
  LDS) and memory-latency-bound**, so the baseline's many small loads supply the MLP that
  hides HBM latency; few wide loads lose it, and the transpose keeps the LDS store scalar
  regardless (no `ds_read_transpose` on gfx942). The "issue-/dependency-wait bound, cut
  VMEM instrs" premise (from the MI325X box) does **not** transfer here. **Recommend
  re-sequencing: run Phase D (occupancy) first**, then re-attempt vectorized staging once
  there are enough waves to tolerate wide-load latency.

### Phase B — `grad_jagged` K-column operand reuse (cut dOut re-streaming)
- **Why:** dJagged reads dOut once per K-output tile (`KOUT_BLOCKS = K/BLOCK_N`). dOut
  traffic ≈ `L*N*2 * KOUT_BLOCKS` dominates the kernel's bytes; jagged is Issue-Wait-bound
  (53–55%), LDS-occupancy-capped (6–9% occ), with many tiny WGs. It is the #2 cost at
  both D and the lowest-occupancy kernel.
- **Do:** make one WG compute several `BLOCK_N` K-output tiles from a **single** staged
  dOut A-fragment (a K-coarsening analogous to the existing `COARSEN_M`, but over the
  output-K axis). The dOut g2s + s2r feed is loaded once and reused across the K-tiles'
  Dense B-fragments. **Re-sweep `COARSEN_M` at the current D=256 default** and at D=512
  (the value may differ per shape).
- **Target:** dOut HBM bytes ↓ (→ AI ↑, push toward compute), Issue-Wait ↓, fewer WGs.
- **Gate:** measurable `grad_jagged` µs/TF-s gain at both D, no correctness regression.
- [x] K-coarsening  [x] re-sweep COARSEN_M @ D=256 and D=512  [x] re-profile + EXP block.
- **RESULT (EXP-2026-06-29b): GATE FAILED — reverted.** K-coarsening is correct (cos
  0.999999, both D × both regimes) but gives **no robust gain**: at D=256 every coarsened
  config regresses (M2K2 5237 vs baseline 4946 µs; fastest is *no* coarsening), and at
  D=512 the best point (M1K2 15257) is only ~0.9% under baseline (15400) = noise;
  `COARSEN_K=4` **spills** (21–26 ms). Root cause = same as Phase A: `grad_jagged` is
  **occupancy/latency-bound (6.4% occ, Issue-Wait 54.5%), not dOut-bandwidth-bound**, so
  the extra fp32 accumulator (VGPR 36→268) cuts occupancy faster than the dOut-reuse
  helps. The COARSEN_M re-sweep also showed the shipped **COARSEN_M=2 is itself neutral**
  on this box. **Recommend Phase D (occupancy) first**, then re-attempt — register
  headroom would let the reuse convert to time.

### Phase C — `SPLIT==1` fast path: write bf16 dDense directly, skip the reduce (D=512)
- **Why:** at D=512 `SPLIT = 2 if K<=256 else 1` → **1**. With one split, each output
  `(K,N)` tile is fully reduced inside the partials kernel, so `grad_dense_reduce`
  (824 µs uniform; a larger share under skew) only casts fp32→bf16 over the scratch —
  pure HBM round-trip — and the partials kernel writes fp32 (2×) only to feed it.
  **At D=256, SPLIT=2, so this phase is N/A there** (the reduce is a real reduction).
- **Do:** when `SPLIT == 1` (compile-time), have `grad_dense_partials_kernel` truncate
  its fp32 accumulator to **bf16** and store straight to `dDense` (the bounded
  `(n_groups*K, N)` view), skipping `partials` scratch and the `grad_dense_reduce`
  launch (3 launches → 2). Keep the SPLIT≥2 path (D=256) as-is. Do the same for the tiny
  dBias when SPLIT==1 (write `dBias` directly, drop `grad_bias_reduce`).
- **Target:** remove ~824 µs reduce + halve the partials write traffic (fp32→bf16) at
  D=512; partials HBM-BW% and AI ↑.
- **Gate:** D=512 `dense_bias` end-to-end ↓ (expect ≥1.05× uniform, more under skew),
  D=256 (SPLIT=2) **unchanged**, correctness green both stars.
- [x] SPLIT==1 direct-write  [x] drop reduce launches  [x] re-profile + EXP block.
- **RESULT (EXP-2026-06-29c): GATE PASSED ✅ — SHIPPED.** D=512 `dense_bias` **1.04×
  uniform / 1.30× skew** (kernel-sum 19841→19077 / 4310→3307 µs; both reduce launches
  dropped, 3→1), D=256 unchanged, correctness green (cos 0.999999, both regimes). Finding:
  the win is entirely the **reduce-launch removal** (a fixed D-bound `n_groups·K·N`
  round-trip → small under uniform, large under skew); **halving the partials write traffic
  did nothing** because `grad_dense_partials` is read-dominated (streams ~16 GB J+dOut vs
  ~0.5 GB partials write), so the partials kernel time is unchanged. First kept win of the
  three phases; also simplifies the D=512 schedule and drops the fp32 scratch from its
  critical path.

### Phase D — Lift `grad_dense_partials` occupancy (register/LDS pressure)
- **Why:** ~28–30% wavefront occupancy, capped by VGPR (~72–76%) + LDS (~71–77% →
  2 WG/CU) at both D. More residency would hide the load/MFMA latency that Phases A/C
  don't fully remove.
- **Do:** sweep the output sub-tile / accumulator footprint — e.g. `DDENSE_BK×DDENSE_BN`
  64×128 or 64×64 to shrink the AGPR C-fragment and the 32 KB LDS staging (currently
  `(128·64 + 128·64)·2`), trading a larger grid for >2 WG/CU. Re-confirm `DDENSE_BM=64`
  (min for the (4,4,2) K-fragment) and audit VGPR temporaries introduced by Phase A.
- **Gate:** occupancy ↑ **and** partials µs ↓ (occupancy alone is not the goal — it must
  convert to time given the kernel is already 62–77% HBM-BW).
- [ ] tile/footprint sweep  [ ] re-profile + EXP block.

### Phase E — Vectorize the reduce kernels + swizzle polish (matters most at D=256)
- **Why:** reduces are scalar per-column (`_load_scalar`/`_store_scalar`). At **D=256
  SPLIT=2** the dense reduce (250 µs) is a genuine cross-split reduction, not a pure
  cast, so it carries more weight than at D=512; the partials transpose-store also shows
  LDS bank conflicts.
- **Do:** (1) widen the reduce to a vectorized per-thread column strip (128-bit fp32 =
  4 cols) via a `TiledCopy` instead of 1 element/thread — relevant wherever SPLIT≥2
  (D=256). (2) Retune the `sJ/sD` `SwizzleType.get(3,3,3)` for the transposed
  `(out_dim, m)` store to drive LDS conflicts → 0.
- **Gate:** reduce µs ↓ (esp. skew), LDS conflicts/access → ~0, no regression.
- [ ] vectorized reduce (D=256)  [ ] swizzle retune  [ ] re-profile + EXP block.

---

## 5. Suggested order & expected payoff

A → B, then C (D=512-specific) and D/E as polish. **A attacks the partials kernel
(51–54% of the backward)** and helps both stars; it is the single highest-leverage,
shape-independent change. **B is the jagged (43–47%) lever** and the lowest-occupancy
kernel. **C is the cheapest large win for the D=512 star only** (delete the ~0.8 ms
reduce + halve partials write traffic) and is a no-op at the D=256 default, so sequence
it after A/B unless D=512 is the immediate target. D/E are second-order. Re-baseline
**both stars (D=256 default and D=512) and both regimes** after each phase — they tune
differently (SPLIT, and likely COARSEN/footprint), exactly as the main log warns.

## 6. Out of scope here (tracked in the main log's Backlog)
- Non-square `K ≠ N` (`D ≠ Kout`) support.
- gfx950/CDNA4 `ds_read_transpose` (would remove the LDS transpose-store entirely;
  N/A on this gfx942 MI300X).
- Shape-adaptive / `n_groups`-aware `SPLIT` for small launch-bound shapes.

## 7. Reproduce (MI300X, GPU 6)
```bash
cd /workspaces/aiter
export FLYDSL_RUNTIME_ENABLE_CACHE=1
RC=/opt/rocm/libexec/rocprofiler-compute/rocprof-compute
PY=flydsl_venv/bin/python   # flydsl_venv already has pandas 2.2.3, dash, etc.

# full counters + empirical roofline (one pass), D=256 (current source default):
HIP_VISIBLE_DEVICES=6 $PY $RC profile -n bwd_full_d256_b1024_m7680_uniform \
  -p workloads/bwd_full_d256_b1024_m7680_uniform -- \
  $PY aiter/ops/flydsl/kernels/profile_jagged_dense_bmm_bwd.py --mode profile --only all \
  -d 256 -b 1024 -m 7680 --regime uniform --iters 8 --warmup 3

# D=512 star (SPLIT=1):
HIP_VISIBLE_DEVICES=6 $PY $RC profile -n bwd_full_d512_b1024_m7680_uniform \
  -p workloads/bwd_full_d512_b1024_m7680_uniform -- \
  $PY aiter/ops/flydsl/kernels/profile_jagged_dense_bmm_bwd.py --mode profile --only all \
  -d 512 -b 1024 -m 7680 --regime uniform --iters 8 --warmup 3

# per-kernel counters (index 0=dense_partials, 1=jagged, 3=dense_reduce, 6=bias_reduce):
$PY $RC analyze -p workloads/bwd_full_d256_b1024_m7680_uniform -k 0
```
On this MI300X box there is **no `profile_roofline.sh` / no separate `rocprof_venv`**:
the torch wheel here (2.7.1+rocm7.2.2, hip 7.2.53211) bundles no clashing rocprofiler
libs, so there is no double-registration clash, and `flydsl_venv` already carries every
rocprof-compute dependency (pandas pinned 2.2.3). PMC works on this gfx942 MI300X.
`HIP_VISIBLE_DEVICES=6` pins all work to GPU 6 (verified: the analyze Dispatch List
reports `GPU_ID = 6` for every dispatch).


## EXP-2026-06-29a — Phase A: vectorize the dDense partials staging (TiledCopy/128-bit) — GATE FAILED (regression)

**Box / tooling.** MI300X (gfx942), `HIP_VISIBLE_DEVICES=6`, `flydsl_venv`
(torch 2.10.0+rocm7.2.4). Per-kernel GPU time via `rocprofv3 --kernel-trace`
(mean/call, iters=20/warmup=5); end-to-end via `example/profile_jagged_dense_bmm_bwd.py
--bench`. This is **Phase A of the 2026-06-29 MI300X plan** (vectorize the
`grad_dense_partials` global→LDS transpose-staging, top-ranked lever there).

**Scope / implementation.** Replaced the per-thread `vec_width=1` bf16 staging loads
(32 J + 32 dOut per m-tile, manual indices) with **128-bit / 8×bf16 vectorized
`buffer_load`s** along the contiguous global axis (k for J, n for dOut), then scatter
the 8 elements into the transposed swizzled LDS (`sJ(k,m)`/`sD(n,m)`, m contiguous).
Reworked the fused dBias accordingly: a vectorized dOut load makes each thread own
`_VEC=8` consecutive columns, so the per-thread `bias_acc` became a `_VEC`-wide fp32
loop iter-arg and the end-of-kernel LDS combine widened to `[col*_BIAS_ROW_GROUPS+r]`.
The transpose stays on the LDS store (gfx942 has no `ds_read_transpose`).

**Correctness: green.** cos 0.999999 (dDense + dBias) at **D∈{256,512} × {uniform,skew}**
— the vectorized staging + widened dBias combine are correct. The change is purely a
perf regression, not a bug.

### Result: vectorizing the staging makes the partials kernel *slower* at both D.
`grad_dense_partials` mean/call (rocprofv3 `--kernel-trace`, D=256 uniform, B=1024, Mi=7680):
| variant | partials µs | vs baseline |
|---|--:|--:|
| **baseline** (scalar `vec_width=1` staging) | **5425** | 1.00× |
| Phase A (J **and** dOut vectorized + dBias rework) | **8383** | **0.65× (−55% slower)** |

End-to-end `dense_bias` bench (wall-clock µs/iter; partials is ~95% of it):
| D / regime | baseline | Phase A | J-only vectorized |
|---|--:|--:|--:|
| 256 uniform | **5843** | 8796 | 7422 |
| 512 uniform | **20494** | 33394 | — |
| 256 skew | **1333** | 1945 | — |
| 512 skew | **4332** | 6653 | — |

Every config regresses. An **A/B isolation** (vectorize *only* the J load, leaving the
dOut staging + single-column dBias exactly as baseline) **also** regressed
(7422 vs 5843 µs end-to-end) → the regressor is the **load width itself**, not the
dBias rework.

### Why (ISA + mechanism)
ASM dump (`FLYDSL_DUMP_IR=1`), `grad_dense_partials_kernel`, D=256:
| metric | baseline | Phase A |
|---|--:|--:|
| `buffer_load_ushort` (per m-tile) | **64** | 0 |
| `buffer_load_dwordx4` | 0 | **8** |
| `ds_write_b16` (LDS scatter) | 64 | **64 (unchanged)** |
| `v_lshrrev`/`v_and` (bf16 unpack) | ~0 | **25** |
| `s_waitcnt` (whole module) | 73 | 27 |
| VGPR / AGPR | **224 / 0** | 212 / 0 |

The vectorization **worked** (64 scalar loads → 8 `dwordx4`), but:
1. **The transpose forces the LDS store to stay scalar** (`ds_write_b16` stays 64;
   gfx942 has no `ds_read_transpose`), so only one side of the staging vectorizes.
2. **The kernel is occupancy-starved → memory-latency-bound, not issue-instruction-
   bound.** It carries 224 VGPR + a 32 KB *dynamic* LDS staging buffer → only ~2
   wavefronts/SIMD. At that occupancy the baseline's **32 small in-flight loads per
   operand provide the memory-level parallelism (MLP)** that hides HBM latency; folding
   them into 4 wide `dwordx4` loads **cuts MLP** and exposes the latency the few waves
   cannot cover. The inserted `load → unpack(25 ALU) → scalar-scatter` dependency chain
   makes it worse. VGPRs even went *down* (224→212), so this is **not** register
   pressure — it is MLP/latency.

So the MI325X-era premise ("issue-/dependency-wait bound at ~73%, cut VMEM instrs to
win") **does not transfer to this MI300X box**: `grad_dense_partials` here is
latency/occupancy-bound, and fewer-but-wider loads lose the MLP that was hiding the
latency.

### Gate: **FAILED** (target ≥1.15× on `grad_dense_partials` µs at both D; got 0.65×).
**Reverted** — baseline staging restored, **no source change shipped** (`git diff`
clean). Correctness re-confirmed on the restored baseline implicitly (identical to HEAD).

### Recommendation (re-sequence the plan for this box)
- **Do Phase D (occupancy) before Phase A.** The partials kernel must first get more
  waves/SIMD (shrink the 32 KB LDS staging and/or the VGPR/AGPR footprint — e.g. a
  64×64 output sub-tile) so it can tolerate wide-load latency. Vectorized staging
  (and any deeper m-tile pipelining, which would *2×* the LDS and hurt occupancy
  further) is only likely to pay **after** occupancy is lifted.
- A TiledCopy (`make_tiled_copy` + `partition_S/D`) phrasing was **not** pursued:
  it issues the same 128-bit loads, so it shares the MLP-loss root cause; the problem
  is architectural (occupancy/latency), not a manual-codegen artifact.
- The bytes staged are unchanged by Phase A, so even a perfectly vectorized staging
  has limited ceiling here; **cutting HBM traffic** (Phase C at D=512; lighter
  partials/reduce) and **lifting occupancy** (Phase D) are the higher-value levers.

Artifacts: `/tmp/kt_base`, `/tmp/kt_pA` (kernel traces); ASM dumps under
`~/.flydsl/debug/grad_dense_bias/` (regenerable with `FLYDSL_DUMP_IR=1`).

## EXP-2026-06-29b — Phase B: `grad_jagged` K-column operand reuse (K-coarsening) — GATE FAILED (no robust gain)

**Box / tooling.** MI300X (gfx942), `HIP_VISIBLE_DEVICES=6`, `flydsl_venv`. Per-kernel
GPU time via `rocprofv3 --kernel-trace` (mean/call, iters=20/warmup=5). **Phase B of the
2026-06-29 MI300X plan** (cut dOut re-streaming in `grad_jagged`).

**Scope / implementation.** Added **`COARSEN_K`**: one WG now computes `COARSEN_K`
consecutive `BLOCK_N` K-output tiles from a **single** staged dOut A-fragment — the
dOut g2s + s2r feed is loaded once and reused across the K-tiles' Dense B-fragments and
fp32 accumulators (analogous to the existing `COARSEN_M`, but over the output-K axis).
The grid's K dim shrank to `KOUT_KGROUPS = KOUT_BLOCKS // COARSEN_K`; `COARSEN_K` is
clamped to a divisor of `KOUT_BLOCKS` (so D=128/`KOUT_BLOCKS=1` disables it). dOut HBM
reads drop by `COARSEN_K`; LDS (only dOut is staged) is unchanged. Both `COARSEN_M` and
`COARSEN_K` were made env-overridable for the sweep, then **reverted**.

**Correctness: green.** cos 0.999999 at D∈{256,512} × {uniform,skew}.

### Result: K-coarsening gives no measurable gain; it regresses at D=256.
`grad_jagged` mean/call (rocprofv3 `--kernel-trace`, uniform, B=1024, Mi=7680), sweeping
(COARSEN_M, COARSEN_K):
| D | M1K1 | **M2K1 (baseline)** | M4K1 | M1K2 | M2K2 | M4K2 | M1K4 | M2K4 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 256 (KOUT=2) | 4938 | **4946** | 5022 | 5015 | 5237 | 5271 | — | — |
| 512 (KOUT=4) | — | **15400** | 15689 | **15257** | 15565 | — | 21061 | 25914 |

- **D=256:** every K-coarsened config is **slower** than baseline (M2K2 5237 vs 4946);
  the fastest point is *no* coarsening at all (M1K1 4938 ≈ M2K1 4946).
- **D=512:** the best K-coarsened point (M1K2 15257) is only **~0.9%** under baseline
  (15400) — within run-to-run noise, not a robust gain. **`COARSEN_K=4` spills** and is
  catastrophic (21–26 ms, +37–68%).
- **Aside:** the *shipped* `COARSEN_M=2` is itself **neutral** on this box (M1K1≈M2K1 at
  D=256) — the cross-box value from EXP-2026-06-24b doesn't help here either. Left at 2
  (no regression).

### Why (mechanism)
`grad_jagged` is **occupancy/latency-bound, not dOut-bandwidth-bound**, so reducing dOut
traffic does not move the wall: per the 06-29 plan §1b (this box, PMC) it runs at **6.4%
occupancy** (D=256), **Issue-Wait 54.5%**, IPC 0.29 — many tiny WGs that can't hide
latency. K-coarsening adds a **second fp32 accumulator** (and a second Dense B stream),
pushing `grad_jagged` from ~36 VGPR (accumulator in AGPR) to **268 VGPR** at `COARSEN_K=2`
(no spill) and into **spills** at `COARSEN_K=4` — i.e. it **cuts occupancy further**,
offsetting any dOut-reuse benefit. Same root cause as Phase A (EXP-2026-06-29a):
**occupancy is the binding constraint**, and both levers that trade registers for reuse
lose more to occupancy than they gain. (The EXP-2026-06-26a roofline already had
`grad_jagged` at only ~28% HBM BW at D=512, consistent with "not bandwidth-bound".)

### Gate: **FAILED** (target: measurable `grad_jagged` µs gain at **both** D; got a
regression at D=256 and ~noise at D=512). **Reverted** — `grad_jagged` restored to
baseline (`git checkout`, diff clean), `COARSEN_M` kept at 2. Correctness re-confirmed
on the restored baseline (cos 0.999999).

### Recommendation
Same as Phase A: **lift occupancy first (Phase D)**, then re-attempt K-coarsening — with
register headroom the extra accumulator(s) would not crater occupancy, and the dOut-reuse
could then convert to time. As-is, neither M- nor K-coarsening is a lever on this box.

## EXP-2026-06-29c — Phase C: `SPLIT==1` fast path — write bf16 dDense/dBias directly, skip the reduce (D=512) — GATE PASSED ✅ (SHIPPED)

**Box / tooling.** MI300X (gfx942), `HIP_VISIBLE_DEVICES=6`, `flydsl_venv`. Per-kernel GPU
time via `rocprofv3 --kernel-trace` (mean/call, iters=20/warmup=5); end-to-end via
`example --bench`. **Phase C of the 2026-06-29 MI300X plan.**

**Scope / implementation.** When `SPLIT == 1` (compile-time; D=512 → `2 if K<=256 else 1`),
each `(K,N)` output tile is fully reduced inside one workgroup, so the separate reduce
passes are pure fp32→bf16 round-trips. Added a compile-time `if fx.const_expr(SPLIT == 1)`
branch to `grad_dense_partials_kernel`'s epilogue: truncate the fp32 MFMA accumulator to
bf16 and store it **straight to dDense** (`BufferCopy16b`), and likewise write the fused
dBias column sums straight to bf16 `dBias`. The launcher passes `dDense`/`dBias` as the
`PARTIALS`/`BIAS_PARTIALS` args and launches **1 kernel instead of 3** when `SPLIT==1`
(`part_off` collapses to the dDense element offset because `off_s==0`). The `SPLIT>=2`
path (D=256) is byte-identical to before (fp32 scratch + two reduces). `make_fragment_C`
still yields an fp32 accumulator for a bf16 output view (same as `grad_jagged`).

**Correctness: green.** cos 0.999999 (dDense + dBias) at D∈{256,512} × {uniform,skew}.

### Result: removes both reduce launches at D=512; clear win, big under skew.
Per-kernel GPU time (`rocprofv3 --kernel-trace`, D=512, B=1024, Mi=7680, mean/call):
| kernel | base uniform | **Phase C uniform** | base skew | **Phase C skew** |
|---|--:|--:|--:|--:|
| `grad_dense_partials` | 18974 | 19077 | 3448 | 3307 |
| `grad_dense_reduce` | 861 | **— (dropped)** | 857 | **— (dropped)** |
| `grad_bias_reduce` | 5.5 | **— (dropped)** | 5.5 | **— (dropped)** |
| **dense_bias (sum)** | **19841** | **19077 (1.04×)** | **4310** | **3307 (1.30×)** |

End-to-end `dense_bias` bench (wall-clock µs/iter): D=512 uniform 20494→**20073** (1.02×),
skew 4332→**3342 (1.30×)**. **D=256 unchanged** (SPLIT=2 path untouched): rocprof still
shows 3 launches (partials 5455 + reduce 243 + bias_reduce 5.9); bench 5843→5903 / 1333→1342
(noise).

### Finding: the win is the reduce-launch removal, not the partials write-traffic halving.
The plan expected "remove ~824 µs reduce **+** halve the partials write traffic
(fp32→bf16)". The reduce removal landed (−866 µs uniform); the **partials kernel itself is
unchanged** (18974→19077 µs, noise) — halving its *write* bytes does ~nothing because the
partials kernel is **read-dominated** (it streams ~16 GB of J+dOut vs ~0.5 GB fp32 partials
write at D=512). So the realized win = the deleted reduce pass, which is a **fixed
D-bound cost** (`n_groups·K·N` round-trip) → a small 1.04× under uniform (where it's ~4% of
a 19.8 ms job) but a large **1.30×** under skew (where the small ~3.3 ms partials makes the
fixed ~0.86 ms reduce a big fraction). D=256 (SPLIT=2, a genuine cross-split reduction) is
correctly left alone.

### Gate: **PASS** — D=512 `dense_bias` end-to-end ↓ (1.04× uniform, **1.30× skew**),
D=256 **unchanged**, correctness green at both stars. (Uniform is just under the 1.05×
estimate only because the write-traffic-halving half of the hypothesis didn't materialize;
the reduce-removal alone is a robust, zero-downside win.) **SHIPPED** — first kept change of
the three phases. It also simplifies the D=512 schedule (3 launches → 1) and removes the
fp32 dDense/dBias scratch from the D=512 critical path.

## EXP-2026-06-29d — Phase D: lift `grad_dense_partials` occupancy (tile/footprint sweep) — GATE FAILED + it reframes Phase A

**Box / tooling.** MI300X (gfx942), `HIP_VISIBLE_DEVICES=6`, `flydsl_venv`. Per-kernel GPU
time via `rocprofv3 --kernel-trace` (mean/call, iters=20/warmup=5); VGPR/LDS from
`FLYDSL_DUMP_IR=1` ISA. **Phase D of the 2026-06-29 MI300X plan** + the requested **Phase A
revisit** (does higher occupancy rescue the vectorized staging?).

**Scope.** Made the partials output sub-tile `DDENSE_BK × DDENSE_BN` env-overridable and
swept {128×128 (baseline), 64×128, 128×64, 64×64}; `DDENSE_BM=64` fixed. Smaller tiles
shrink the LDS staging (`(BK+BN)·BM·2`) and the per-thread fp32 C-fragment
(`BK·BN/256`), raising waves/CU — at the cost of a bigger grid and more operand re-reads
(J ×`N/BN`, dOut ×`K/BK`). Fixed the SPLIT==1 epilogue's hardcoded `[64]` C-fragment to
the computed `_DDENSE_CFRAG` so the small tiles are correct.

### Result: occupancy rises a lot; partials time does NOT move (gate fails).
`grad_dense_partials` mean/call (rocprofv3, uniform, B=1024, Mi=7680):
| tile (BK×BN) | D=256 µs | D=512 µs | VGPR | LDS | ~occupancy |
|---|--:|--:|--:|--:|--:|
| **128×128 (baseline)** | **5462** | **19032** | 224 | 32 KB | ~2 WG/CU |
| 64×128 | 5487 | 19040 | — | 24 KB | — |
| 128×64 | 5455 | 19090 | — | 24 KB | — |
| 64×64 | 5465 | 19044 | **98** | **16 KB** | ~4 WG/CU |

`64×64` more than **halves VGPR (224→98)** and **halves LDS (32→16 KB)** → roughly
**doubles occupancy** (~25%→~50%), yet partials time is **identical** (5465 vs 5462 µs,
noise). It also **doubles** the J+dOut HBM re-reads and the LDS staging volume — also with
**no time change**.

### Conclusion: `grad_dense_partials` is MFMA-pipeline-bound, not occupancy/traffic/LDS-bound.
The only quantity invariant across the whole sweep is the **MFMA instruction count**
(`2·L·K·N` is fixed; tiling just redistributes the same MFMAs). Occupancy ↑, HBM traffic
↑/↓, LDS volume ↑/↓ all leave the time flat → the kernel sits at its **matrix-core
issue/dependency limit**, with the load/LDS/feed already fully overlapped behind the MFMA
pipeline. The "26–28% MFMA util" from §1 is therefore **issue/dependency-latency in the
MFMA feed**, not a throughput headroom that occupancy can recover.

### Gate: **FAILED** — occupancy ↑ ~2× but partials µs flat (does not convert to time, the
gate's explicit requirement). **Reverted** to the shipped Phase-C state (128×128, fixed
constants). The smaller tile is a *free* occupancy lever (same time) but ships **2× more
HBM traffic + 4× the grid**, so it is strictly worse system-wide — not kept.

### Phase A revisit (the asked-for "does Phase D rescue Phase A?")
Added an env-guarded **vectorized-J staging** (128-bit / 8×bf16 loads) and A/B'd it against
the scalar baseline at both tiles (rocprof partials, D=256 uniform):
| tile | scalar | vectorized-J |
|---|--:|--:|
| 128×128 | 5444 | **5443** |
| 64×64 | 5476 | **5449** |

**Vectorizing the loads is NEUTRAL at every tile** (also end-to-end: J-vec dense_bias 5862
vs baseline 5843 µs, cos 0.999999) — exactly what "MFMA-bound, loads already hidden"
predicts. **This corrects EXP-2026-06-29a:** its claim that "J-only vectorization also
regressed (7422 µs end-to-end)" was a bad measurement — clean rocprof + bench show
J-staging vectorization is **neutral**. The full Phase-A regression (partials 5425→8383 µs)
therefore came from the **dOut-vectorization + dBias-fusion rework** (the vec8 bias
iter-args + per-element extract/convert/add in the dOut loop), **not** the load
vectorization. So:
- Higher occupancy *is* available at zero time cost (Phase D headroom), and at that
  headroom vectorized loads don't regress — **but it doesn't matter**, because staging was
  never the bottleneck. Phase A has **no path to a win** on this kernel: loads are already
  hidden (vectorizing = neutral) and the bias rework is pure overhead.
- **The real lever for `grad_dense_partials` is the MFMA pipeline itself** — the MMA atom
  (try `16×16×32` / `32×32×8`), the `(4,4,2)` K-fragment, `traversal_order`, and breaking
  the accumulator dependency chain (more independent C sub-tiles per wave) to lift the
  26–28% MFMA util — **not** staging, occupancy, LDS, or HBM traffic.

Artifacts: `/tmp/dsw`, `/tmp/vsw` (kernel traces); ISA via `FLYDSL_DUMP_IR=1`.

## EXP-2026-06-29e — Phase D revisited: SHIP the 64×64 footprint as a register/LDS-headroom state change (perf-neutral)

**Decision (supersedes EXP-2026-06-29d's "reverted").** EXP-29d showed the 64×64 sub-tile
is *time-neutral* but reverted it because it ships more HBM traffic. Reconsidered on a
"state-machine" view: a perf-neutral change that **halves register + LDS pressure** is a
real state change worth banking — it is exactly the headroom the next lever (MFMA-feed ILP:
more independent accumulator chains / a wider MMA atom) needs, and 128×128's **224 VGPR /
~2 WG/CU** had none. So `grad_dense_partials` now uses **`DDENSE_BK = DDENSE_BN = 64`**
(was 128), plus a computed `_DDENSE_CFRAG` for the SPLIT==1 bf16 epilogue.

**Footprint (ISA, D=256):** partials **VGPR 224 → 98** (no spill), **LDS 32 KB → 16 KB**,
fp32 C-fragment 64 → 16 elems/thread → occupancy ~2×.

**Correctness: green** — cos 0.999999 (dDense + dBias) at D∈{256,512} × {uniform,skew}.

**Performance: neutral everywhere** (end-to-end `dense_bias` bench, **back-to-back**
128×128 vs 64×64, µs/iter):
| D / regime | 128×128 | 64×64 |
|---|--:|--:|
| 256 uniform | 5867 | 5908 |
| 512 uniform | 19990 | 19974 |
| 256 skew | 1335 | 1336 |
| 512 skew | 3305 | 3321 |
All within ~0.7% (noise). Also re-verified per-kernel (rocprof) flat as in EXP-29d.

**Variance caveat (shared box).** An *initial* one-off D=512-skew run read 3887 µs for
64×64 (looked like a 16% regression); a clean back-to-back re-measure (all four tiles
3305–3327 µs) showed it was **transient contention from another user on the server**, not a
real regression. Lesson: always compare tiles back-to-back in one run here, and distrust a
single off reading.

**Status: SHIPPED** (kept). It does not reduce time on its own (the kernel is MFMA-bound,
EXP-29d), but it **banks the register/LDS headroom** for the MFMA-feed ILP work, which is
the actual next lever. The traffic/grid cost (J ×N/64, dOut ×K/64; 4× WGs) is free while
MFMA-bound — to be watched if a later change makes the kernel memory-bound.

## Current status (2026-06-29)

**Phase C SHIPPED ✅; Phase D footprint (64×64) SHIPPED as a perf-neutral register/LDS
headroom change (EXP-2026-06-29e); Phases A & B gate-failed.** The day's mechanistic
keystone (EXP-29d): `grad_dense_partials` is **MFMA-pipeline-bound**: a
128×128→64×64 tile sweep halves VGPR (224→98) and LDS (32→16 KB) and ~doubles occupancy
with **zero** change in kernel time, and also 2×'s HBM traffic with zero change — so
occupancy, LDS, and bandwidth are all already hidden behind the matrix-core feed. This
**explains and corrects Phase A**: vectorizing the staging *loads* is **neutral** (not the
regressor EXP-29a thought); the Phase-A regression came from the dOut+dBias-fusion rework,
and no staging/occupancy change can speed an MFMA-bound kernel. **Next lever is the MFMA
pipeline (atom/fragment/traversal/accumulator-ILP), not memory or occupancy.** For
`grad_jagged` the same occupancy-isn't-the-lever story holds (Phase B). **Shipped: Phase C**
(D=512 reduce-launch removal, 1.04× uniform / 1.30× skew) **+ the Phase D 64×64 footprint**
(EXP-29e) — perf-neutral but banks VGPR 224→98 / LDS 32→16 KB as headroom for the MFMA-feed
ILP work, which is the next lever to try.

## Current status (2026-06-29-C)

**Phase C SHIPPED ✅ (EXP-2026-06-29c); Phases A & B gate-failed and reverted
(EXP-2026-06-29a/b).** Net for the day on GPU 6:
- **C (kept):** `SPLIT==1` (D=512) now writes bf16 dDense/dBias straight from the partials
  kernel and drops both reduce launches (3→1). **D=512 `dense_bias` 1.04× uniform / 1.30×
  skew**, D=256 unchanged, correctness green. The win is the reduce-launch removal (a fixed
  D-bound cost, hence huge under skew); halving the partials *write* traffic did nothing
  because that kernel is read-dominated.
- **A & B (reverted):** both were register-for-reuse levers on the two MFMA kernels and both
  lost to **occupancy**, the binding constraint here — (A) vectorizing `grad_dense_partials`
  staging regressed it 5425→8383 µs (latency/MLP-bound at ~2 waves); (B) `grad_jagged`
  K-coarsening gave no robust gain (extra accumulator drove VGPR 36→268, cutting the 6.4%
  occupancy). Even the shipped `COARSEN_M=2` is neutral on this box.
- **Takeaway / next:** the two wins available without an occupancy fix are *structural*
  (delete work / traffic), like C. The compute-kernel micro-opts (A/B, and likely the
  staging part of D's siblings) are gated on **lifting occupancy first → Phase D** (shrink
  the 32 KB LDS / register footprint of `grad_dense_partials`), after which A/B may convert.
`grad_jagged` baselines (this box, GPU 6, rocprofv3): D=256 uniform 4953 µs, D=512 uniform
15384 µs; bench djagged D=256/512 uniform 5035 / 15844 µs, skew 953 / 2779 µs.

## Current status (2026-06-29-A)

**Phase A (2026-06-29 MI300X plan) attempted and GATE-FAILED on GPU 6 (EXP-2026-06-29a).**
Vectorizing the `grad_dense_partials` global→LDS transpose-staging to 128-bit loads is
**correct** (cos 0.999999, D∈{256,512} × {uniform,skew}) but **regresses** the partials
kernel **5425 → 8383 µs (0.65×)** at D=256 uniform, and every `dense_bias` bench config
is slower. Root cause (ISA + A/B isolation): the kernel is **occupancy-starved (~2
waves/SIMD: 224 VGPR + 32 KB dynamic LDS) and memory-latency-bound**, so the baseline's
many small loads give the MLP that hides HBM latency; few wide loads lose it, and the
transpose keeps the LDS store scalar regardless (no `ds_read_transpose` on gfx942).
**Reverted to baseline; nothing shipped.** Next: **re-order to do Phase D (occupancy)
first**, then re-attempt vectorized staging once there are enough waves to tolerate
wide-load latency. Baselines for reference (this box, GPU 6, rocprofv3 kernel-trace,
B=1024/Mi=7680): partials D=256 uniform **5425 µs**, end-to-end `dense_bias` bench
D=256/512 uniform **5843 / 20494 µs**, skew **1333 / 4332 µs**.
