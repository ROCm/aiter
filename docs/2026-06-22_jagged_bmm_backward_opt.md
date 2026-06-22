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

## Current status (2026-06-22)
Tooling works end-to-end; baseline characterized. No kernel changes yet.
Optimization plan below; start at Phase 0.

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
- [ ] EXP-…: shape sweep {b64/m512, b128/m1024, b256/m2048} × {uniform, skew}.

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
