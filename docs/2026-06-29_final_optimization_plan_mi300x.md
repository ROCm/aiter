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
