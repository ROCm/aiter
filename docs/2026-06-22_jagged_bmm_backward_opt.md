# Jagged-Dense BMM Backward — Profiling & Optimization Log

Living document. Append dated experiments; **never silently overwrite old results**.
Each experiment records the exact shape/tiles/commit so stale numbers are obvious
once kernels or shapes change. Mark superseded results ~~struck~~ or move them to
"Archived". Date = day the data was collected.

Kernels under study: `aiter/aiter/ops/flydsl/kernels/jagged_dense_bmm_bwd.py`
(`grad_jagged`, `grad_dense` = partials+reduce, `grad_bias` = partials+reduce).
Backward math/plan: `docs/2026-06-18_backward_bmm.md`.

---

## Environment (pinned 2026-06-22)

- GPU: **MI300X_A1 (gfx942)**, 304 CUs, 9728 wave slots, 192 GB HBM3, ROCm **7.2.0**.
- `flydsl_venv`: torch **2.12.1+rocm7.2** (hip 7.2.53211). **Do not modify.**
- Profiler: **rocprof-compute 3.4.0** (`rocprofiler-compute` apt pkg).

## Tooling (how to reproduce)

Three helpers live next to the kernels:
- `profile_jagged_dense_bmm_bwd.py` — minimal driver. Builds inputs + compiles
  once, then loops only the kernel under test. `--mode bench` (wall-clock
  TFLOP/s) or `--mode profile` (for rocprof).
- `profile_roofline.sh` — clash-safe wrapper around rocprof-compute.
- `roofline_report.py` — turns a workload into a per-kernel table + roofline PNG.

Two practical problems were solved to get here:
1. **Separate driver venv.** rocprof-compute needs pandas/dash/matplotlib; these
   are installed in a dedicated `rocprof_venv` (pandas **pinned 2.2.3** — pandas
   3.x breaks the CSV merge). The profiled app is launched with `flydsl_venv`'s
   python, so `flydsl_venv` stays pristine.
2. **PyTorch rocprofiler clash.** The torch wheel bundles its own
   `librocprofiler-register.so`, `librocprofiler-sdk.so`, `libroctracer64.so`
   (loaded via `RPATH=$ORIGIN`, which beats `LD_*PATH`). With rocprofv3 active
   this double-registers → abort *"error code 16 … outside of valid rocprofiler
   configuration period"*. Fix (in `profile_roofline.sh`): move those 3 libs
   aside for the run + `LD_LIBRARY_PATH=/opt/rocm/lib` so torch uses the single
   system stack (identical 7.2.70200 version), then always restore. Net-zero venv
   change.

```bash
# roofline (lightweight, ~3 counter passes):
bash aiter/aiter/ops/flydsl/kernels/profile_roofline.sh --only all -b 64 -m 512
# full counters (~13 passes) for the detailed report:
bash aiter/aiter/ops/flydsl/kernels/profile_roofline.sh --only all --full \
  -b 64 -m 512 --iters 10 --warmup 3 --name bwd_full_all
# per-kernel report (kernel ids: 0=dense_partials 1=bias_partials 2=jagged
#                                 3=dense_reduce 5=bias_reduce):
rocprof_venv/bin/python /opt/rocm/libexec/rocprofiler-compute/rocprof-compute \
  analyze -p workloads/bwd_full_all -k 0 --kernel-verbose 1
# roofline table + PNG:
rocprof_venv/bin/python aiter/aiter/ops/flydsl/kernels/roofline_report.py \
  workloads/bwd_full_all --png workloads/bwd_full_all/roofline_bwd.png
```

Tile constants at time of writing (from `jagged_dense_bmm_bwd.py`):
`BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, K=N=128, SPLIT=4`; dDense partials use an
LDS-staged scalar-FMA reduction `DDENSE_BM=64, 16×16 threads, KPT=NPT=8` (no MFMA).

---

## EXP-2026-06-22a — Baseline roofline + full profile

- Shape: `n_groups=64, max_seq_len=512, uniform`, `L=32768`, `K=N=128`. bf16 in/out, fp32 accum.
- Kernels as of 2026-06-22 (pre-optimization baseline).
- Artifacts: `workloads/bwd_full_all/` (counters, PDFs, `analysis/report_k*.txt`).

### Empirical MI300X ceilings (this box)
HBM **4.21 TB/s** · L2 23.7 TB/s · VALU fp32 **121 TFLOP/s** (theo. 81.7) ·
MFMA bf16 **479 TFLOP/s** (theo. 1307). (`analyze` %-of-peak uses the *theoretical*
peaks; `roofline_report.py` uses the *empirical* ones — hence different %.)

### Per-kernel placement
| kernel | µs/call | TFLOP/s | engine | AI(HBM) | % of roofline @ AI |
|---|--:|--:|---|--:|--:|
| `grad_jagged` (GEMM) | 11.8 | **91.1** | MFMA bf16 | 42.6 | **~51%** (HBM-bound region) |
| `grad_dense_partials` | 79.3 | **6.8** | VALU fp32 | 15.8 | **~10%** |
| `grad_dense_reduce` | 5.7 | 0.7 | VALU | 0.2 | mem/serial tail |
| `grad_bias_partials` | 31.7 | 0.13 | VALU | 0.5 | mem tail |
| `grad_bias_reduce` | 2.0 | 0.02 | VALU | 0.2 | mem tail |

### Diagnosis (from full report)
**`grad_dense_partials` (dominant cost).** Not memory-bound (HBM 2.5%), not LDS-bound
(0 bank conflicts), not at compute peak (VALU 8.3%). Bound by, in order:
- **No MFMA** (`MFMA Instr=0`): scalar fp32 VALU path, ceiling 16× below bf16 MFMA.
- **MUL+ADD not fused** (`F32-FMA=0`, F32-MUL=F32-ADD=4.19M): ~2× VALU instrs.
- **Latency-bound**: Dependency-Wait = **58%** of cycles (loop-carried fp32 acc),
  with only **3.6% occupancy** (1024 waves; 128 VGPR + 256 AGPR/thread; 256 WGs).

**`grad_jagged` (healthiest).** Uses MFMA (bf16), VALU FLOPs=0. But **1.06%
occupancy** (103 waves, 139/304 CUs active), IPC 0.19, MFMA 6.7% of theo. peak —
~12 µs kernel that's launch/latency/tail-bound; grid (256 WGs) underfills the GPU.

**Cross-cutting.** At this size both compute kernels **under-occupy** the MI300X;
low %-of-roofline is as much occupancy/parallelism as algorithm. Reduce kernels
are negligible tails.

### Actions ranked (data-backed)
1. **MFMA-ize `grad_dense_partials`** (plan §8) — biggest lever (16× ceiling, frees
   VALU, breaks the dependency stall).
2. **Raise occupancy / grid** for partials + jagged (more `m`-splits / tile output
   over more CUs).
3. **FMA fusion** in the dense_partials inner loop (moot once MFMA lands).

---

## EXP-2026-06-23a — Phase 0: occupancy vs. algorithm (shape × regime sweep)

- **No kernel changes** (same source as EXP-2026-06-22a). Goal: decide whether the
  low %-of-roofline is "too little work" (occupancy) or "bad kernel" (algorithm)
  before investing. Profiled the full sweep `{b64/m512, b128/m1024, b256/m2048} ×
  {uniform, skew}`, full counters, `--iters 10 --warmup 3`, via the clean harness.
- Run in `flydsl_venv` directly (rocprof-compute 3.4.0, torch 2.7.1+rocm7.2.2,
  pandas 2.2.3 all present); empirical `roofline.csv` was reused across workloads.
- Artifacts (`aiter/ops/flydsl/kernels/workloads/`):
  - `bwd_full_all/` (b64/m512 uniform = baseline), `p0_b128_m1024_uniform/`,
    `p0_b256_m2048_uniform/`, `p0_b64_m512_skew/`, `p0_b128_m1024_skew/`,
    `p0_b256_m2048_skew/` — each `…/MI300X_A1/` has counters + roofline PDFs.
  - Combined comparison reports (multi-`--path` analyze): `phase0_uniform_sweep_report.txt`
    (current vs new), `phase0_full_sweep_report.txt` (+skew), and per-kernel
    `phase0_uniform_k0_grad_dense_partials.txt`, `phase0_uniform_k2_grad_jagged.txt`.

### Uniform size sweep (profiled µs/call, TFLOP/s = 2·L·K·N, occupancy, MFMA util)
| kernel | metric | b64/m512 | b128/m1024 | b256/m2048 |
|---|---|--:|--:|--:|
| `grad_dense_partials` | µs/call | 79.3 | 283.0 | 1233.7 |
|  | TFLOP/s | 13.5 | 15.2 | 13.9 |
|  | occupancy | 3.59% | 4.08% | 3.39% |
|  | VALU FLOP %peak | 8.3% | 9.3% | 8.5% |
|  | MFMA util | 0% | 0% | 0% |
|  | grid (WGs) / active CUs | 256 / 73% | 512 / 81% | 1024 / 85% |
| `grad_jagged` | µs/call | 12.6 | 39.5 | 125.0 |
|  | TFLOP/s | 85.0 | 108.6 | 137.5 |
|  | occupancy | 1.06% | 1.77% | 2.97% |
|  | MFMA bf16 util | 4.6% | 8.2% | 15.9% |
|  | grid (WGs) | 256 | 1024 | 4096 |

(`L` per shape: 32768 / 131072 / 524288. Skew shapes have much smaller `L`
— 6130 / 21008 / 86945 — so every kernel is even more launch/latency-bound there;
see `phase0_full_sweep_report.txt`. The doc's earlier 6.8 TF/s for dense_partials
used a MAC count without the ×2; numbers here use the harness `2·L·K·N` convention.)

### Gate decision → **algorithm-bound; go to Phase 2 (MFMA) next**
- **`grad_dense_partials` (dominant, 58–80% of GPU time): does NOT scale with size.**
  16× more work (b64→b256) costs 15.6× time; TFLOP/s is flat ~14, VALU FLOPs flat
  ~8% of peak, MFMA=0. Occupancy is **stuck at ~3.5%** even when the grid grows to
  1024 WGs and 85% of CUs are active — it is **register-capped** (128 VGPR + 256
  AGPR/thread), not work-starved. Bigger shapes do not help → algorithm-bound.
  Per the gate ("stays low → prioritize MFMA"), **Phase 2 is the lever.**
- **`grad_jagged` (already MFMA): is partly occupancy-bound.** With size, occupancy
  rises 1.06%→2.97%, MFMA util 4.6%→15.9%, throughput 85→137 TF/s (1.6×). It is
  launch/tail-limited at small shapes and improves a lot when the GPU fills — but
  even GPU-full it sits at ~29% of the empirical bf16 MFMA roofline. → Phase 3
  (grid/occupancy) is a real but **secondary** win (small share of total time).
- **Reorder note:** Phase 1 (FMA fusion on the scalar path) is now low value since
  the path is being replaced; Phase 0 says it "may reorder everything" — **skip
  straight to Phase 2 (MFMA-ize `grad_dense_partials`)**, then revisit Phase 3 for
  `grad_jagged` occupancy. Phase 1b register/occupancy work folds into Phase 2.

---

## EXP-2026-06-23b — Required production shapes B=1024, D=256/512

**Mapping the request onto the kernels.** "B" = `n_groups` (number of jagged
groups/sequences); "D" = the dense-weight feature/head dimension, which in these
kernels is `K == N` (per-group `Dense[b]` is `(K, N)`, jagged is `(L, K)`, dOut is
`(L, N)`). `D` is a **compile-time constant** (`N`, `K` in `jagged_dense_bmm.py`),
so to benchmark D∈{256,512} I temporarily set `K=N=D`, ran, and reverted with git
(net-zero source change). Seq length was **not** specified in the request, so I used
`max_seq_len=512` (the established baseline) for both `uniform` and `skew`; `L` is
D-independent (uniform `L=524288`, skew `L=79790`). Numbers are `--mode bench`
wall-clock (warmup 10/iters 50; fewer for slow `ddense`), TF/s = `2·L·K·N / t`.

### Results (B=1024, m=512; bench wall-clock)
| grad | regime | D=256 µs/iter | D=256 TF/s | D=512 µs/iter | D=512 TF/s |
|---|---|--:|--:|--:|--:|
| `grad_jagged` (GEMM, MFMA) | uniform | 389.1 | **176.6** | 1169.0 | **235.1** |
| `grad_jagged` | skew | 157.8 | 66.3 | 564.2 | 74.1 |
| `grad_bias` (partials+reduce) | uniform | 104.0 | 1.29 | — | **FAILS** |
| `grad_bias` | skew | 65.0 | 0.31 | — | **FAILS** |
| `grad_dense` (partials+reduce) | uniform | 5396.1 | 12.7 | — | **FAILS** |
| `grad_dense` | skew | 2305.0 | 4.54 | — | **FAILS** |

(Empirical ceilings, this box: bf16 MFMA **479 TF/s**, HBM **4.21 TB/s**. So
`grad_jagged` reaches ~37% of the MFMA roofline at D=256 uniform and **~49% at
D=512 uniform** — it scales up nicely with D and the GPU-filling B=1024.
`grad_bias` is memory-bound (D=256 uniform 2668 GB/s ≈ 63% of HBM).)

### What works and what doesn't at these shapes
- **D=256: all three gradients run.** `grad_jagged` is healthy (MFMA, 177→235
  TF/s). `grad_dense` runs but is **catastrophically slow (12.7 TF/s, 5.4 ms)** and
  the same flat ~13 TF/s seen in Phase 0 — *worse* now because its per-thread
  accumulator is `(K/16)·(N/16) = 16·16 = 256` fp32 regs/thread at D=256 (vs 64 at
  D=128), so it heavily **spills VGPRs**; its LDS is `2·64·256·2 = 64 KB` (right at
  the MI300 limit). This is the scalar-FMA kernel Phase 0 already flagged as
  algorithm-bound.
- **D=512: only `grad_jagged` runs.** `grad_bias`/`grad_dense` **fail to launch**:
  - `grad_bias_partials` and `grad_dense_reduce` launch `block=(N,1,1)` → 512
    threads > AMDGPU default `max_flat_workgroup_size` of **256**
    (`ValueError: ... Add known_block_size=[512,1,1]`).
  - `grad_dense_partials` additionally needs `2·64·512·2 = 128 KB` LDS > the **64 KB**
    MI300 LDS limit, and `(32·32)=1024` fp32 acc regs/thread (VGPR cap is 256).

### Implications for the plan
- These shapes **harden the Phase-0 verdict**: the dominant `grad_dense` path is the
  blocker at scale. It is not just slow at D=256, it is structurally **unbuildable at
  D=512**. **Phase 2 (MFMA-ize `grad_dense_partials`)** is now a hard requirement,
  not just an optimization — the MFMA rewrite removes the `D²` per-thread register
  block and the `O(D)` LDS staging that break D=512.
- Cheap enabling fixes needed for D=512 regardless of Phase 2 (no algorithm change):
  add `known_block_size` / cap threads ≤256 (tile the N axis) in `grad_bias_partials`,
  `grad_bias_reduce`, and `grad_dense_reduce` so `block ≤ 256` at D≥512.
- `grad_jagged` already meets these shapes well (235 TF/s @ D=512) — no action beyond
  the secondary Phase-3 occupancy work.

---

## Current status (2026-06-23)
Tooling works end-to-end; baseline + Phase-0 sweep + required production shapes
(B=1024, D=256/512) characterized. No kernel changes yet. Phase-0 gate + EXP-…b both
say the dominant `grad_dense` path is the blocker: algorithm-bound at D≤256 (flat
~13 TF/s, occupancy ~3.5%) and **structurally unbuildable at D=512** (LDS 128 KB >
64 KB, 1024 acc VGPRs, 512-thread reduce block). `grad_jagged` is healthy and scales
to 235 TF/s @ D=512. Next action is **Phase 2 (MFMA-ize `grad_dense_partials`)**,
plus the cheap `block ≤ 256` enabling fixes for `grad_bias`/`grad_dense_reduce` at
D≥512; `grad_jagged` occupancy (Phase 3) remains a secondary win.

---

# Optimization Plan (phased)

Phases are an internal evaluation cadence — **re-run EXP-2026-06-22a config at
each gate and append a dated EXP block**. Do **not** reference these phase
numbers in source code (same rule as `2026-06-18_backward_bmm.md` §0). Stop after
each gate to decide whether to proceed, reorder, or drop a phase. Order targets
the biggest measured loss first, but Phase 0 may reorder everything.

Baseline to beat (EXP-2026-06-22a, b64/m512/uniform): `grad_dense_partials`
**6.8 TF/s / 79 µs**, `grad_jagged` **91 TF/s / 12 µs**.

### Phase 0 — Separate occupancy from algorithm (no kernel changes)
- **Why:** at b64/m512 both kernels under-occupy (dense 3.6%, jagged 1.06%). Must
  know how much gap is "too little work" vs. "bad kernel" before investing.
- **Do:** profile a GPU-filling shape (b=256, m=2048) + a mid point; also skew.
- **Gate:** if %-of-roofline rises a lot with size → prioritize occupancy
  (Phase 1b/3); if it stays low → algorithm-bound, prioritize MFMA (Phase 2).
- [x] EXP-2026-06-23a: shape sweep {b64/m512, b128/m1024, b256/m2048} × {uniform,
  skew}. **Gate result:** dominant `grad_dense_partials` is algorithm-bound
  (occupancy flat ~3.5% with size, MFMA=0) → **go to Phase 2**; `grad_jagged`
  scales with size (occupancy 1.06%→2.97%) → secondary Phase 3. See EXP block above.

### Phase 1 — `grad_dense_partials` cheap wins (FMA + occupancy)
- **Why:** `F32-FMA=0` (MUL+ADD unfused → 2× VALU instrs); 58% dependency-stall
  at 3.6% occupancy; 128 VGPR + **256 AGPR**/thread caps residency.
- **Do (1a):** make the inner `acc += j*d` emit `v_fma_f32` (restructure the
  rmem accumulator so the DSL fuses; avoid the per-`m` load/store round-trip).
- **Do (1b):** grow parallelism — tile the (K,N)=128² output across multiple
  workgroups and/or raise `SPLIT`; cut register pressure to lift occupancy.
- **Target metrics:** F32-FMA>0 & VALU instrs ~halved; Dependency-Wait ↓;
  occupancy ↑; µs ↓. **Gate:** ≥1.5× on `grad_dense_partials` TF/s.
- [ ] 1a FMA fusion  [ ] 1b output-tiling/SPLIT + regs  [ ] re-profile + EXP block

### Phase 2 — MFMA-ize `grad_dense_partials` (the big lever)
- **Why:** `MFMA Instr=0`; scalar fp32 ceiling (81.7 TF/s) is 16× below bf16 MFMA
  (1307). This is `2026-06-18` §8 item #1.
- **Do:** replace the scalar-FMA reduction with bf16 MFMA on the transposed GEMM
  `C[k,n]=Σ_m J[m,k]·dOut[m,n]`, feeding fragments from LDS-staged tiles
  (CDNA4 LDS-read-transpose). Keep fp32 accumulate + the split-reduction skeleton.
- **Target:** MFMA Utilization > 0, large TF/s jump toward `grad_jagged`-class.
  **Gate:** `grad_dense_partials` within ~2× of `grad_jagged` TF/s; correctness
  (cosine > 0.999, uniform+skew) still passes via `example_…_bwd.py`.
- [ ] MFMA partials  [ ] validate correctness  [ ] re-profile + EXP block

### Phase 3 — `grad_jagged` throughput / occupancy
- **Why:** MFMA already, but 1.06% occupancy, 139/304 CUs, ~12 µs → launch/tail
  bound; grid only 256 WGs.
- **Do:** more work per launch / better CU fill (e.g. finer N or K tiling to
  raise WG count, or persistent-block scheme); confirm the forward double-buffer
  pipeline is actually overlapping here.
- **Target:** occupancy ↑, CUs active ↑, TF/s ↑. **Gate:** measurable TF/s gain
  without correctness regression. (May be size-limited — Phase 0 informs this.)
- [ ] grid/tiling change  [ ] re-profile + EXP block

### Phase 4 — Fuse dBias into dDense partials + reduce-tail cleanup
- **Why:** `grad_bias_partials`/reduce are tiny tails reducing over the same `m`
  axis as dDense (`2026-06-18` §8 item #2). Shared `dOut` loads.
- **Do:** fold bias partial-sums into the dDense partials kernel; revisit a
  2-level reduction tree if `SPLIT` grew in Phase 1b.
- **Gate:** fewer kernels/launches, no regression; correctness holds.
- [ ] fuse bias  [ ] reduce tree (if needed)  [ ] re-profile + EXP block

### Phase 5 — Autotune & integrate
- **Do:** autotune `SPLIT` / block sizes over the seq-length distribution; wire a
  timing+TFLOPs summary into `example_jagged_dense_bmm_bwd.py` (`2026-06-18`
  Phase 4); run the style gate.
- **Gate:** best config picked per regime; both `uniform`+`skew` green.
- [ ] autotune  [ ] example timing  [ ] style gate

## Backlog (not yet scheduled)
- Tall `dDense` `(n_groups*N, K)` layout (revisit `2026-06-18` A2) if it removes a
  host transpose on the hot path.
- Port `dJagged` tweaks back to the forward kernel if shared.

## Notes on staleness
Numbers above are tied to the 2026-06-22 kernel source + the b64/m512/uniform
shape. Any kernel edit or shape change invalidates the table — re-run and add a
new dated EXP block rather than editing this one.
