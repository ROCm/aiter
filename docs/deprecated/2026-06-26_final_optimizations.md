# Jagged-Dense BMM Backward — Final Optimization Plan (2026-06-26)

Planning doc spun off from the profiling log `2026-06-22_jagged_bmm_backward_opt.md`
(EXP-2026-06-26b, the first run with **working rocprofiler PMC counters**, on the
**MI325X** box). It (1) reads the new roofline + counters, (2) audits how faithfully
each kernel uses FlyDSL's **layout algebra** (vs hand-rolled index math), and (3)
proposes a phased plan. Phase numbers here are local to this doc — **do not reference
them in source** (same rule as the other logs). Each phase has a measurement gate;
re-run the EXP-2026-06-26b config and append a dated EXP block to the profiling log on
every gate. Correctness gate everywhere: `example_jagged_dense_bmm_bwd.py --dim {256,512}`
cosine > 0.999, **uniform + skew**, no regression at the repo default D=128.

Kernels: `aiter/ops/flydsl/kernels/jagged_dense_bmm_bwd.py`
(`grad_jagged`; fused `grad_dense_bias` = `grad_dense_partials_kernel` [dDense+dBias
MFMA partials] + `grad_dense_reduce_kernel` + `grad_bias_reduce_kernel`).

---

## 1. Baseline this plan is built on (MI325X, gfx942, D=512, B=1024, Mi=7680, uniform)

From `workloads/bwd_full_d512_b1024_m7680_uniform/` (rocprof-compute 3.4.0 full
counters + empirical roofline; `analysis/report_k{0,1,3,6}.txt`).

**Empirical ceilings (this box):** HBM **4.61 TB/s** · L2 28.0 TB/s · L1 37.6 TB/s ·
LDS 62.8 TB/s · **MFMA bf16 591 TF/s** · FP16 1069 · F8 2178 · VALU fp32 146 TF/s.
bf16 **ridge point = 591/4.61 ≈ 128 FLOP/byte**.

| kernel | µs/call | TF/s | % MFMA peak (591) | MFMA util | occ | VGPR/AGPR | AI_hbm | % HBM BW | dominant stall |
|---|--:|--:|--:|--:|--:|---|--:|--:|---|
| `grad_dense_partials` (dDense+dBias) | **14964** | 276.6 | **46.8%** | 27.5% | **5.15%** | 96 / 128 | 83.6 | 71.8% | Dep-wait 38.5% ≈ Issue-wait 35.8% |
| `grad_jagged` (dJagged) | **12488** | 330.2 | **55.9%** | 32.4% | **2.75%** | 36 / 132 | 97.0 | 73.9% | Issue-wait 52% > Dep-wait 33% |
| `grad_dense_reduce` (fp32→bf16) | 618 | 0.43 | — | 0 | 3.8% | 4 / 4 | 0.2 | 56.6% | mem tail |
| `grad_bias_reduce` | 2.9 | — | — | 0 | 1.6% | 4 / 4 | 0.2 | 23% | negligible |

Full backward (sum) ≈ **28.07 ms**: partials **53%**, jagged **44%**, reduces 2.2%.

**Roofline reading (the important part).** With *measured* HBM traffic (TCC
counters), both MFMA kernels sit **left of the ridge** (AI 83.6 / 97.0 < 128) — i.e.
they are in the **HBM-leaning region**, achieving **~72–74% of HBM BW** *and only*
**47–56% of the bf16 MFMA peak**. They are near-balanced at the knee, not "deep
compute-bound" (that earlier claim used an analytic-AI upper bound that ignored the
fp32 partials round-trip + dOut re-reads). So there is headroom on **both** axes; the
biggest wins **cut HBM traffic** (push right, toward compute) and **raise MFMA overlap
/ occupancy** (push up).

Occupancy limiters are now known from the counters (were TODO before PMC worked):
- `grad_dense_partials`: **VGPR-capped** (75.6% "Insufficient SIMD VGPRs", 96 VGPR +
  128 AGPR/thread) **and** LDS-capped (75.9% "Insufficient CU LDS", 32 KB/WG → 2 WG/CU).
- `grad_jagged`: purely **LDS-capped** (70.2% "Insufficient CU LDS"; VGPR fine at 36),
  plus dispatch latency (Scheduler-Pipe Stall 21%, ~123k tiny WGs, Issue-Wait dominant).

---

## 2. Layout-algebra audit (the user's question)

FlyDSL exposes a CuTe-style **layout algebra**: `make_layout` / `make_ordered_layout`,
`flat_divide` / `logical_divide`, `make_composed_layout` (swizzle), `partition_S/D`,
`make_fragment_{A,B,C}`, `retile`, `thr_slice`, and **`make_tiled_copy` (vectorized,
thread-mapped copies)**. Using it lets the compiler vectorize, swizzle, and map
threads→data correctly; hand-rolling linear indices + scalar `buffer_load`/`memref_store`
bypasses all of that.

| kernel / region | layout algebra? | evidence |
|---|---|---|
| `grad_jagged_kernel` (whole) | **Yes — faithful** | `flat_divide` tiling (221-223), `partition_S/D` (240-243), `make_fragment_*`+`retile` (245-252), swizzled LDS `make_composed_layout` (234-238), vectorized `tiled_copy_g2s_A` = `UniversalCopy128b` (343-347). Only manual bit: int64 row rebasing (208-213), which is idiomatic. |
| `grad_dense_partials` MFMA core + epilogue | **Yes** | `make_fragment_{A,B,C}`+`retile` (463-467), `make_tiled_copy_C`+`partition_S` fp32 store (520-525), swizzled `sJ/sD` (435-446). |
| **`grad_dense_partials` global→LDS staging** | **NO — hand-rolled scalar** | **lines 483-499**: per-thread `lin/m_local/k_local` index math + `buffer_load(..., vec_width=1, dtype=bf16)` (488, 496) → **32 + 32 scalar 2-byte loads per thread per m-tile**, then scalar `memref_store` scatter into LDS (489, 497). No `TiledCopy`/`partition_*`. |
| **`grad_dense_reduce` / `grad_bias_reduce`** | **Partial — scalar granularity** | `_load_scalar`/`_store_scalar` (145-159) = `fx.slice` + `logical_divide(make_layout(1,1))` + **1-element** copy_atom; one thread per column, scalar fp32 loads over SPLIT (377, 571) and scalar bf16 store (380, 575). |

**Verdict.** The two GEMM *cores* respect the layout algebra; the **memory-movement
edges of the dDense path do not** — the partials staging and the reduces are
hand-indexed scalar copies. The staging loop is loop-coalesced across a wavefront
(consecutive lanes → consecutive addresses, so DRAM bytes are fine), but it emits
**~64 separate scalar load instructions + address-arithmetic per thread per m-tile**
instead of 4+4 vectorized 128-bit loads — directly feeding the partials kernel's high
issue-/dependency-wait and its 27.5% MFMA utilization. This is exactly the
intersection of "respect the layout algebra" and "the #1 measured perf lever".

---

## 3. Opportunities, ranked by measured leverage

1. **Vectorize the partials staging via a `TiledCopy` (layout algebra).** Biggest
   kernel (53%), scalar loads today → issue/dep-wait bound at 27.5% MFMA util.
2. **`SPLIT==1` direct-bf16 dDense (skip the fp32 partials scratch + reduce).** At
   D=512 `SPLIT=1`, so `grad_dense_reduce` does **no reduction** — it is a pure
   ~1 GB fp32→bf16 HBM round-trip. Eliminating it also lets the partials kernel write
   **bf16 (0.5 GB) instead of fp32 (1 GB)**, cutting its output traffic in half.
3. **`grad_jagged` K-column operand reuse.** dOut is re-streamed once per `KOUT_BLOCKS`
   (=4 @ D=512) K-output tile; one WG computing multiple K-tiles from a single dOut
   load cuts that traffic and amortizes the short (8-step) MFMA pipeline.
4. **Lift `grad_dense_partials` occupancy** (VGPR+LDS capped at 2 WG/CU): smaller
   output sub-tile / AGPR footprint sweep.
5. **Vectorize the reduce kernels + retune the transpose-store swizzle** (LDS conflicts
   1.33/access in partials; scalar reduces matter under skew).

---

## 4. Phased plan

### Phase A — Vectorize the dDense partials staging with a TiledCopy (layout algebra)
- **Why:** lines 488/496 do `vec_width=1` bf16 loads (32+32/thread/m-tile) with manual
  indices; the kernel is issue-/dependency-wait bound (74% combined) at only 27.5% MFMA
  util and 71.8% HBM BW. `grad_jagged` already shows the target pattern
  (`tiled_copy_g2s_A`, 128-bit).
- **Do:** express the global→LDS transpose-staging as `make_tiled_copy` + `partition_S`
  (global, vectorized 128-bit / 8×bf16 along the contiguous axis — k for J, n for dOut)
  → `partition_D` (LDS). The transpose stays on the LDS store (gfx942 has no
  `ds_read_transpose`); evaluate (a) a swizzled **vectorized** LDS store vs (b) a
  vectorized load + scalar scatter store. **Rework the fused-dBias accumulation**: a
  vectorized dOut load makes each thread own `vec_width` consecutive columns `n`, so the
  per-thread `bias_acc` becomes a small vector and the end-of-kernel LDS combine widens
  accordingly (it currently assumes 1 column/thread via `tid % DDENSE_BN`).
- **Target metrics:** VMEM-issued instrs ↓; Issue-Wait + Dependency-Wait ↓; MFMA
  Utilization ↑ (toward `grad_jagged`'s 32%+); partials µs ↓.
- **Gate:** ≥1.15× on `grad_dense_partials` µs (rocprof), correctness green.
- [ ] vectorized staging  [ ] dBias-fusion rework  [ ] re-profile + EXP block.

### Phase B — `SPLIT==1` fast path: write bf16 dDense directly, skip the reduce
- **Why:** at D=512 `SPLIT = 2 if K<=256 else 1` → **1**. With one split, each output
  `(K,N)` tile is fully reduced inside the partials kernel, so `grad_dense_reduce`
  (618 µs uniform; **~18% of dense_bias under skew**) only casts fp32→bf16 over a
  ~1 GB scratch — pure HBM round-trip. And the partials kernel writes fp32 (2×) only to
  feed that reduce.
- **Do:** when `SPLIT == 1` (compile-time), have `grad_dense_partials_kernel` truncate
  its fp32 accumulator to **bf16** and store straight to `dDense` (the bounded
  `(n_groups*K, N)` view), skipping `partials` scratch and the `grad_dense_reduce`
  launch (3 launches → 2). Keep the SPLIT≥2 path (D=256) as-is. Do the same for the
  tiny dBias when SPLIT==1 (write `dBias` directly, drop `grad_bias_reduce`).
- **Target:** remove ~618 µs reduce + halve the partials write traffic (fp32→bf16);
  partials HBM-BW% and AI ↑.
- **Gate:** D=512 `dense_bias` end-to-end ↓ (expect ≥1.05× uniform, more under skew),
  D=256 (SPLIT=2) **unchanged**, correctness green both stars.
- [ ] SPLIT==1 direct-write  [ ] drop reduce launches  [ ] re-profile + EXP block.

### Phase C — `grad_jagged` K-column operand reuse (cut dOut re-streaming)
- **Why:** dJagged reads dOut once per K-output tile (`KOUT_BLOCKS = K/BLOCK_N = 4` @
  D=512). dOut traffic ≈ `L*N*2 * KOUT_BLOCKS` dominates the kernel's bytes; jagged is
  Issue-Wait-bound (52%), LDS-occupancy-capped, ~123k tiny 8-K-step WGs.
- **Do:** make one WG compute several `BLOCK_N` K-output tiles from a **single** staged
  dOut A-fragment (a K-coarsening analogous to the existing `COARSEN_M`, but over the
  output-K axis). The dOut g2s + s2r feed is loaded once and reused across the K-tiles'
  Dense B-fragments. Re-sweep `COARSEN_M` at **D=512** (the COARSEN_M=2 pick was made at
  D=256, EXP-2026-06-24b).
- **Target:** dOut HBM bytes ↓ (→ AI ↑, push toward compute), Issue-Wait ↓, fewer WGs.
- **Gate:** measurable `grad_jagged` µs/TF-s gain, both stars, no correctness regression.
- [ ] K-coarsening  [ ] re-sweep COARSEN_M @ D=512  [ ] re-profile + EXP block.

### Phase D — Lift `grad_dense_partials` occupancy (register/LDS pressure)
- **Why:** 5.15% occupancy, capped by VGPR (75.6%) + LDS (75.9% → 2 WG/CU). More
  residency would hide the load/MFMA latency that Phases A/B don't fully remove.
- **Do:** sweep the output sub-tile / accumulator footprint — e.g. `DDENSE_BK×DDENSE_BN`
  64×128 or 64×64 to shrink the AGPR C-fragment and the 32 KB LDS staging (currently
  `(128·64 + 128·64)·2`), trading a larger grid (already saturated at 16 384 WGs) for
  >2 WG/CU. Re-confirm `DDENSE_BM=64` (min for the (4,4,2) K-fragment) and audit VGPR
  temporaries introduced by Phase A.
- **Gate:** occupancy ↑ **and** partials µs ↓ (occupancy alone is not the goal — it must
  convert to time given the kernel is already 72% HBM-BW).
- [ ] tile/footprint sweep  [ ] re-profile + EXP block.

### Phase E — Vectorize the reduce kernels + swizzle polish
- **Why:** reduces are scalar per-column (`_load_scalar`/`_store_scalar`); negligible at
  D=512 uniform but a real share under skew, and the partials transpose-store shows 1.33
  LDS bank-conflicts/access.
- **Do:** (1) widen the reduce to a vectorized per-thread column strip (128-bit fp32 =
  4 cols) via a `TiledCopy` instead of 1 element/thread — only relevant where SPLIT≥2
  survives Phase B (D=256). (2) Retune the `sJ/sD` `SwizzleType.get(3,3,3)` for the
  transposed `(out_dim, m)` store to drive LDS conflicts → 0.
- **Gate:** reduce µs ↓ (skew), LDS conflicts/access → ~0, no regression.
- [ ] vectorized reduce (D=256)  [ ] swizzle retune  [ ] re-profile + EXP block.

---

## 5. Suggested order & expected payoff

A → B → C, then D/E as polish. A and B both attack the **partials kernel (53% of the
backward)** and are largely independent; B is the cheapest large win for the **D=512
star** specifically (delete a ~0.6 ms kernel + halve partials write traffic). C is the
**jagged (44%)** lever. D/E are second-order. Re-baseline both stars (D=256 **and**
D=512) and both regimes after each phase — the two stars tune differently (SPLIT, and
likely COARSEN/footprint), exactly as the main log warns.

## 6. Out of scope here (tracked in the main log's Backlog)
- Non-square `K ≠ N` (`D ≠ Kout`) support.
- gfx950/CDNA4 `ds_read_transpose` (would remove the LDS transpose-store entirely;
  N/A on this gfx942 MI325X).
- Shape-adaptive / `n_groups`-aware `SPLIT` for small launch-bound shapes.

## 7. Reproduce
```bash
cd ~/git/meta/aiter
export FLYDSL_RUNTIME_ENABLE_CACHE=1
RC=/opt/rocm/libexec/rocprofiler-compute/rocprof-compute
PY=~/git/meta/flydsl_venv/bin/python   # flydsl_venv already has pandas 2.2.3, dash, etc.
# full counters + empirical roofline (one pass):
$PY $RC profile -n bwd_full_d512_b1024_m7680_uniform -p workloads/bwd_full_d512_b1024_m7680_uniform -- \
  $PY aiter/ops/flydsl/kernels/profile_jagged_dense_bmm_bwd.py --mode profile --only all \
  -d 512 -b 1024 -m 7680 --regime uniform --iters 8 --warmup 3
# per-kernel counters (0=dense_partials,1=jagged,3=dense_reduce,6=bias_reduce):
$PY $RC analyze -p workloads/bwd_full_d512_b1024_m7680_uniform -k 0
# roofline table + PNG (bf16/HBM, grad_-filtered):
$PY aiter/ops/flydsl/kernels/roofline_report.py workloads/bwd_full_d512_b1024_m7680_uniform \
  --filter grad_ --png workloads/bwd_full_d512_b1024_m7680_uniform/roofline_d512_uniform.png
```
(No `profile_roofline.sh` / no separate `rocprof_venv` needed on this box: torch
2.10.0+rocm7.2.2 bundles no rocprofiler libs, so there is no double-registration clash,
and `flydsl_venv` already carries every rocprof-compute dependency. PMC works — the
prior box's unsupported-iGPU blocker is absent here.)
