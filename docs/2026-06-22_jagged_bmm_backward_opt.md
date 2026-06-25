# Jagged-Dense BMM Backward — Profiling & Optimization Log

Living document. Append dated experiments; **never silently overwrite old results**.
Each experiment records the exact shape/tiles/commit so stale numbers are obvious
once kernels or shapes change. Mark superseded results ~~struck~~ or move them to
"Archived". Date = day the data was collected.

Kernels under study: `aiter/aiter/ops/flydsl/kernels/jagged_dense_bmm_bwd.py`
(`grad_jagged` = GEMM; `grad_dense_bias` = one fused MFMA partials pass writing both
dDense and dBias partials + two reduce passes — see EXP-2026-06-25a; `grad_dense`/
`grad_bias` were separate before Phase 4).
Backward math/plan: `docs/2026-06-18_backward_bmm.md`.

---

## Environment (pinned 2026-06-22)

- GPU: **MI300X_A1 (gfx942)**, 304 CUs, 9728 wave slots, 192 GB HBM3, ROCm **7.2.0**.
- `flydsl_venv`: torch **2.12.1+rocm7.2** (hip 7.2.53211). **Do not modify.**
- Profiler: **rocprof-compute 3.4.0** (`rocprofiler-compute` apt pkg).

## North Star shape (production target — fully specified)

The optimization target is the production HSTU-class shape:

| axis | value | kernel/driver knob | HSTU bench name |
|---|--:|---|---|
| groups (batch) | **B = 1024** | `n_groups` (`-b 1024`) | `B` |
| dense feature dim | **D = 256** (`K = N = 256`) | `K`, `N` in `jagged_dense_bmm.py` | `D` (= `Kout`, square) |
| max sequence length | **Mi = 7680** | `max_seq_len` (`-m 7680`) | `Mi` |

`Mi = 7680` is fixed to match the **reference HSTU Triton forward benchmark default**
(`op_tests/flydsl_tests/bench_jagged_dense_bmm_perf.py`, `-mi 7680`), so our
backward numbers are directly comparable to that baseline's envelope.

> **Caveat on existing results (append-only policy).** The EXP blocks dated
> 2026-06-23b/c/d were collected **before** `Mi` was pinned, at the placeholder
> **`max_seq_len = 512`** (see EXP-2026-06-23b). Their absolute µs/TFLOP/s therefore
> reflect a much smaller, more launch-bound envelope than the now-canonical
> `Mi = 7680`; the *relative* speedups (e.g. Phase 2's 4.60×) still hold, but the
> headline numbers must be **re-measured at `Mi = 7680`** before being quoted as
> the fully-specified North Star. Those blocks are left intact (numbers tagged
> `Mi=512`) rather than overwritten; a fresh EXP block should capture `Mi = 7680`.

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
# per-kernel report (kernel ids shift after Phase 4 dropped grad_bias_partials;
#  now: grad_dense_partials [fused dDense+dBias], grad_jagged, grad_dense_reduce,
#  grad_bias_reduce — confirm ids with `analyze --list-kernels`):
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

## EXP-2026-06-23c — Phase 1 cheap wins on `grad_dense_partials` (B=1024, D=256, Mi=512 — pre-spec North Star)

**Scope.** Phase 1 (1a FMA fusion + 1b occupancy/register/LDS) on the dominant
`grad_dense_partials` kernel, evaluated at the **production shape B=1024, D=256**
(D=K=N; `max_seq_len=512`). Kernel edits live in `jagged_dense_bmm_bwd.py`; D was
set to 256 for measurement then reverted (repo default stays D=128). All T-1↔T deltas
are from `rocprof-compute analyze` with **multiple `--path`** (raw counters), kernel
`-k 0` = `grad_dense_partials`.

**Changes (all kept; correctness `cos≈1.0`, uniform+skew, D=128 and D=256):**
- **1a — FMA fusion.** Inner contraction now uses explicit `fx.math.fma(j, d, acc)`
  instead of `acc + j*d`, which was lowering to unfused mul+add. → `F32-FMA` 0 →
  268M, VALU instrs −31%.
- **1b — output-tiling.** Each workgroup now owns one `DDENSE_BK×DDENSE_BN = 128×128`
  output sub-tile of the `(K,N)` block (grid gains a `NK·NN` factor), so per-thread
  accumulators stay fixed at `8×8` and **LDS is fixed at 32 KB regardless of D**
  (was 64 KB @ D=256, 128 KB @ D=512). This already removes the *partials* kernel's
  D=512 LDS blocker (the bias/reduce 256-thread blocker is separate, still open).
- **1b — loop-carried accumulators.** Removed the per-m-tile `acc` rmem load/store
  round-trip; accumulators are carried through the dynamic m-tile loop. Cleaner,
  perf-neutral (VGPR pinned at 128 either way → kernel is not VGPR-alloc-bound).

### `grad_dense_partials` results (rocprof mean µs/call)
| shape | T-1 | T(1a) | T(1b, final) | speedup | artifacts |
|---|--:|--:|--:|--:|---|
| **B=1024, D=256, Mi=512** | 5263.6 | 4537.0 | **3930.3** | **1.34×** | `phase1_t0/t1a/t1b_b1024_d256/`, `phase1_cmp_*` |
| B=64, D=128 (repo default) | 79.3 | — | **65.8** | **1.21×** | `bwd_full_all` vs `phase1_t1b_b64_d128/`, `phase1_cmp_d128_*` |

Supporting counters @ D=256 (T-1 → T1b): VALU instrs 881M → **516M (−41%)**;
VALU FLOPs 8.0% → **10.7%** of peak; `F32-FMA` 0 → 268M; Dependency-Wait **+20%**;
Wavefront occupancy 4.38% → **2.68%**; VGPR 128 → 128.

### Gate: **1.34× < 1.5× target → cheap wins do not clear the bar → go to Phase 2.**
The diagnosis is now unambiguous and matches the Phase-0 verdict: with the MULs fused
and registers/LDS bounded, the kernel is **dependency-wait / latency-bound** on the
loop-carried *scalar fp32* accumulation (Dependency-Wait rose to dominate; occupancy
fell yet runtime improved → not occupancy-bound, not VALU-throughput-bound). The only
remaining lever is replacing the scalar FMA reduction with **bf16 MFMA (Phase 2)**,
which both raises the compute ceiling 16× and breaks the dependency chain. Phase 1 is
banked as a strict improvement (1.2–1.34×, and it pays down the D=512 LDS blocker);
**proceed to Phase 2.**

---

## EXP-2026-06-23d — Phase 2: MFMA-ize `grad_dense_partials` (B=1024, D=256, Mi=512 — pre-spec North Star)

**Scope.** Replace the scalar fp32 LDS-FMA reduction in `grad_dense_partials` with
a **bf16 MFMA** transposed GEMM, keeping the fp32 accumulate + two-pass
split-reduction skeleton. Only `grad_dense_partials_kernel` (+ its launcher's
`tiled_mma`) changed for the perf win; the reduce/bias kernels got an unrelated
`block ≤ 256` col-tiling fix (see below). Evaluated at **B=1024, D=256** (D=K=N,
`max_seq_len=512`); D set to 256 for measurement, repo default restored to 128.

**Implementation.** `dDense[k,n] = Σ_m J[m,k]·dOut[m,n]` is the MFMA atom form
`C[i,j]=Σ_l A[i,l]·B[j,l]` with `i=k, j=n, l=m`, so `A = J.T (k,m)` and
`B = dOut.T (n,m)` — **both operands carry the reduction axis `m` as their
contiguous fragment (K) axis.** This box is **gfx942 (CDNA3)**, which has **no**
`ds_read` hardware transpose (that is gfx950/CDNA4), so the transpose is done **on
the global→LDS store**: each m-tile is read coalesced along the contiguous global
axis (k for J, n for dOut) and stored into LDS as `sJ(k,m)` / `sD(n,m)` (m
contiguous), reusing the forward kernel's swizzle. The s2r feed + `fx.gemm` are the
forward's `tiled_mma` (MFMA 16×16×16 bf16, atom-layout (1,4,1), (4,4,2)
K-fragment). One fp32 MFMA C-fragment per thread is accumulated **in place across
the dynamic, split-strided m-tile loop** (AGPR accumulate, no SSA carry). Output
tiling (128×128 per WG) keeps LDS at 32 KB and the accumulator fixed at any D.

### `grad_dense` results (bench wall-clock, partials+reduce; TF/s = 2·L·K·N / t)
| shape | regime | T-1 (Phase-1b) | T (Phase-2 MFMA) | speedup |
|---|---|--:|--:|--:|
| **B=1024, D=256, Mi=512** | uniform | 4252.7 µs / 16.16 TF/s | **924.0 µs / 74.37 TF/s** | **4.60×** |
| B=1024, D=256, Mi=512 | skew | 1780.6 µs / 5.87 TF/s | **686.8 µs / 15.23 TF/s** | **2.59×** |
| B=64, D=128 (repo default) | uniform | 83.1 µs / 12.92 TF/s | 86.8 µs / 12.37 TF/s | ~1.0× (launch-bound) |

(At B=64/D=128 the kernel is launch/occupancy-bound — only 256 WGs — so MFMA is
neutral there, exactly as Phase 0 predicted. The win shows up once the GPU fills.)

### `grad_dense_partials` counter deltas (rocprof `analyze` multi-`--path`, `-k 0`)
Artifacts: `workloads/p2_t0_b1024_d256/` (T-1), `workloads/p2_t1_b1024_d256/` (T).
| metric | T-1 | T (Phase-2) | Δ |
|---|--:|--:|--:|
| partials kernel duration | — | — | **−84.5% (≈6.4×)** |
| MFMA Utilization | 0% | **15.95%** | MFMA on (gate ✓) |
| MFMA bf16 instrs / kernel | 0 | **8.39 M** | — |
| VALU FLOPs (F32) | 8634 Gflop/s | **0** | −100% (scalar path gone) |
| Dependency-Wait cycles | 6.58 B | 0.45 B | **−93.1%** (Phase-1 bottleneck collapsed) |
| Issue-Wait cycles | 0.21 B | 1.12 B | +434% (now memory/issue-bound) |
| Wavefront occupancy | 2.64% | 4.07% | +54% |
| LDS instrs / kernel | 25.2 M | 11.0 M | −56% |
| LDS bank conflicts/access | 0 | 1.33 | transpose-store conflicts (minor; optimizable) |

### Gate: **PASS → MFMA on, big TF/s jump, correctness green.**
- **MFMA Utilization > 0** ✓ (0 → 15.95%); **correctness** cosine 0.999999, uniform
  + skew, D=128 and D=256 ✓.
- **Within ~2× of `grad_jagged`** ✓ at the kernel level: the MFMA partials kernel is
  ~6.4× faster (rocprof), i.e. ~110 TF/s vs `grad_jagged` 176.6 TF/s (≈1.6×). The
  *end-to-end* `grad_dense` bench (74 TF/s) is now dragged by the **reduce tail**,
  which became the dominant fraction once partials sped up 6×. → **Phase 4** (fuse
  bias into dDense partials / lighter reduce) is now the next lever, not the GEMM.
- The kernel is now **HBM/issue-bound** (uniform 3050 GB/s ≈ 72% of the 4.21 TB/s
  HBM ceiling), not compute- or dependency-bound. Further partials gains would come
  from vectorizing the transpose-staging g2s and cutting fp32 partials traffic.

### Side fix (enables D=512, was a separate blocker — not validated yet)
`grad_bias_partials`, `grad_bias_reduce`, `grad_dense_reduce` launched `block=(N,1,1)`
→ 512 threads > AMDGPU's 256 cap at D=512. Generalized them to a **col-tiled** grid
(`NRED_COL_TILES` blocks of `NRED_BLK=min(N,256)` columns; `col = col_tile·NRED_BLK +
tid`). No-regression verified at D=128 + D=256 (uniform+skew). Combined with the
Phase-2 partials kernel (LDS now fixed 32 KB, no D² register block), this removes all
known D=512 build blockers — but **D=512 end-to-end validation is deferred** per the
earlier decision to keep D=256 as the North Star; greenlight to validate D=512.

---

## EXP-2026-06-24a — North-Star head-to-head vs Triton (B=1024, D=256, Mi=7680) + int32 overflow fix

**Scope.** First fully-specified North-Star measurement (`B=1024, D=256, Kout=256,
Mi=7680`) and the first *head-to-head* FlyDSL-vs-Triton run for the backward. Two
infra changes plus one correctness fix:
- **Wired the FlyDSL provider into `bench_jagged_dense_bmm_bwd_perf.py`** (was
  Triton-only). The bench now scores `flydsl` and `triton` side-by-side via
  `triton.testing.do_bench` (CUDA-event, L2-flushed). `--flydsl-only` /
  `--triton-only` added; the FlyDSL path asserts the requested `D,Kout` match the
  compile-time `K,N`.
- **Set `K=N=256`** in `jagged_dense_bmm.py` for the measurement (North-Star D;
  repo default is 128 — restore after the session, same temp-edit practice as the
  earlier EXP blocks).
- **Correctness fix (`grad_dense_partials_kernel`): int32 → int64 buffer
  `base_byte_offset`.** The group-rebased J/dOut buffer descriptors computed
  `base_byte_offset = seq_start * K * 2` in **int32**. At the North Star `seq_start`
  reaches ≈ `L = 1024·7680 = 7.86M` rows, so the byte offset ≈ **4.0 GB overflows
  int32** and silently wraps the descriptor base → `grad_dense` garbage. This was
  invisible before because every prior EXP ran at `Mi=512` (offset ≈ 268 MB) and
  skew has a much smaller `L`. Fixed by computing the product in int64
  (`fx.Int64(seq_start) * fx.Int64(K*2)`); `num_records_bytes` stays int32 (per-group,
  `≤ Mi`). Validated `cos=1.0000` at the North Star, both regimes.
  - **Perf impact of int64: none (verified, not assumed).** Concern was added VGPR
    pressure / hot-loop cost. A/B at `B=1024, D=256, Mi=512` (where int32 does *not*
    overflow, so both are correct and timing is comparable), `grad_dense` bench
    wall-clock: uniform **int64 905.9 µs vs int32 904.3 µs (+0.17%, in noise)**; skew
    **int64 673.2 µs vs int32 678.8 µs (int64 marginally faster)**. Reason: `seq_start`
    is `readfirstlane`'d → **uniform/scalar**, so the i64 offset multiply is a
    once-per-workgroup **SGPR** op in the prologue; it never enters the per-thread
    VGPR accumulator fragment or the inner m-tile loop. No VGPR-pressure regression.

### Head-to-head (do_bench wall-clock; TF/s = 2·L·D·N / t). Empirical ceilings: bf16 MFMA 479 TF/s, HBM 4.21 TB/s.
| component | regime | FlyDSL | Triton | result |
|---|---|--:|--:|---|
| `jagged` (dJagged) | uniform | **4.97 ms / 207.9 TF/s** | 6.26 ms / 165.0 TF/s | **FlyDSL 1.26×** |
| `jagged` | skew | **0.99 ms / 156.5 TF/s** | 1.28 ms / 123.4 TF/s | **FlyDSL 1.27×** |
| `dense_bias` (dDense+dBias) | uniform | **7.53 ms / 133.1 TF/s** | 8.76 ms / 117.1 TF/s | **FlyDSL 1.16×** |
| `dense_bias` | skew | 2.24 ms / 69.6 TF/s | **1.56 ms / 101.2 TF/s** | **Triton 1.45× (FlyDSL loses)** |

(`L`: uniform = 7,864,320; skew ≈ 1.5M. Component mapping: FlyDSL `grad_jagged` vs
Triton `bwd_jagged`; FlyDSL `grad_dense`+`grad_bias` vs Triton `bwd_dense_bias`.)

### Findings that reframe the plan
- **FlyDSL already beats Triton in 3 of 4 cases at the North Star.** `grad_jagged`
  wins by ~1.27× in both regimes; `dense_bias` wins by 1.16× uniform.
- **Phase 3's premise (grad_jagged occupancy) is obsolete at the North Star.** The
  Phase-3 motivation was a 256-WG grid at `b64/m512` (1.06% occupancy, launch/tail
  bound). At `B=1024, Mi=7680` the `grad_jagged` grid is `ceil(7680/128)·(K/128)·B =
  60·2·1024 = 122,880 WGs` — the GPU is saturated. The kernel runs at **207.9 TF/s
  ≈ 43% of the 479 TF/s MFMA roofline** (uniform); remaining headroom is
  memory/MFMA-tiling, **not** occupancy. So "raise grid/occupancy" no longer applies
  here; any further `grad_jagged` win is a roofline-push (g2s vectorization, tiling).
- **The only Triton loss is `dense_bias` skew (Triton 1.45× faster).** This is the
  real gap-to-North-Star. Likely the split-reduction partials/reduce tail behaves
  badly under skew (many empty/short groups → wasted `SPLIT` blocks + launch
  overhead on the reduce pass), where Triton's dense_bias path is leaner.

## EXP-2026-06-24b — Phase 3: `grad_jagged` profile + M-coarsening (B=1024, D=256, Mi=7680)

**Scope.** Phase 3 on `grad_jagged` at the *fully-specified* North Star. Note this
**re-scopes Phase 3**: the plan text (raise WG count / occupancy) was written for the
old `b64/m512` regime where the grid was launch-*starved* (256 WGs, 1.06% occ). At
the North Star `grad_jagged` already beats Triton (EXP-2026-06-24a) and the grid is
huge, so the lever is the **opposite** of "raise WG count".

**Profile (rocprof-compute full counters, `workloads/p3_djagged_b1024_m7680_uniform/`).**
`grad_jagged` at the North Star is **dispatch/latency-bound, not occupancy-resource-bound**:
- Achieved **Wavefront Occupancy 4.32%** (420/9728) — far below even the LDS-capped
  ceiling (32 KB/WG → 2 WG/CU ≈ 25%). Resource limiters are all clear: VGPR 36 +
  AGPR 74, "Insufficient SIMD VGPRs/SGPRs" 0%, "Reached CU Workgroup Limit" 0%.
- **Workgroup-Manager (SPI) utilization 81.8%**, Scheduler-pipe not-scheduled 15.8%
  → the SPI struggles to dispatch the ~123k *tiny* WGs.
- Each WG does only `NRED_TILES = N/BLOCK_K = 4` MFMA K-steps, so prologue/epilogue
  dominate; **Issue-Wait 6.26 B cycles > Dependency-Wait 3.54 B** (latency unhidden).
- (Profiler rates are serialized/counter-inflated — MFMA util shows 12.6%; trust the
  clean `do_bench` wall-clock for absolute speed, the counters for *ratios*.)

**Change: M-coarsening (`COARSEN_M`).** One workgroup now processes `COARSEN_M`
consecutive `BLOCK_M` output row-tiles (same K-column/Dense slice); the launch grid
M-dim shrinks by `COARSEN_M`. WGs live longer (amortize prologue/epilogue), the SPI
dispatches fewer WGs. Implemented as a `range_constexpr(COARSEN_M)` wrap of the
existing per-tile body (group resolution recomputed per sub-tile — cheap SGPR work;
needed because the `scf.if` rewriter downcasts copy-slice objects hoisted across the
branch). `COARSEN_M=1` reproduces the original kernel/grid exactly.

### `grad_jagged` sweep (do_bench, FlyDSL-only, ms; lower is better)
| COARSEN_M | uniform | skew |
|---|--:|--:|
| 1 (orig) | 4.976 | 1.001 |
| **2 (chosen)** | **4.813** | **0.996** |
| 3 | 4.957 | 1.102 |
| 4 | 4.994 | 1.000 |

### Result (chosen `COARSEN_M=2`, validated head-to-head, cos=1.0000)
| regime | FlyDSL (was→now) | Triton | speedup vs Triton |
|---|--:|--:|--:|
| uniform | 4.97 → **4.81 ms** (−3.3%) | 6.27 ms | 1.26× → **1.31×** |
| skew | 0.99 → **0.98 ms** | 1.28 ms | 1.29× → **1.31×** |

**Gate: PASS (measurable TF/s gain, no correctness regression).** Modest (~3%
uniform) — `grad_jagged` was already well-tuned at this shape; coarsening claws back
dispatch overhead but the kernel is not *severely* dispatch-bound at the wall-clock
level (the profiler's serialized SPI pressure overstates it). `COARSEN_M>2` loses to
reduced parallelism (skew especially). No-regression re-validated at the repo-default
**D=128** (uniform+skew, all three gradients cos=0.999999). Further `grad_jagged`
headroom (still ~44% of MFMA roofline) would need bigger structural changes
(merge the two K-column output tiles to reuse the shared dOut A-fragment; vectorize
staging) — diminishing returns vs the `dense_bias`-skew deficit (see Backlog).

## EXP-2026-06-25a — Phase 4: fuse dBias into the dDense partials pass (B=1024, D=256, Mi=7680)

**Scope.** Phase 4 at the *fully-specified* North Star. `dDense[b][k,n]=Σ_m J[m,k]·dOut[m,n]`
and `dBias[b][n]=Σ_m dOut[m,n]` both reduce over the dynamic sequence axis `m`, and the
dDense partials kernel already streams **every** dOut element through LDS. Before Phase 4,
`grad_bias` was an independent two-pass kernel that **re-read all of dOut from HBM** purely
to sum it — pure memory traffic with no reuse of the dDense pass.

**Profiled motivation (pre-Phase-4 breakdown, profile `--mode bench`, `dense_bias` split
into `ddense` vs `dbias`):**
| regime | ddense (partials+reduce) | dbias (partials+reduce) | dbias share | dbias BW / AI |
|---|--:|--:|--:|--:|
| uniform | 5265 µs | **1721 µs** | **25%** | 2344 GB/s · AI 0.50 |
| skew | 1366 µs | **754 µs** | **36%** | 828 GB/s · AI 0.49 |

`dbias` is almost entirely the `grad_bias_partials` re-read of dOut (`L·N·2` ≈ 4.0 GB uniform;
the reduce is a few µs). This is exactly the standing North-Star deficit (`dense_bias` skew,
EXP-2026-06-24a) — skew spends an even larger fraction in the bias re-read.

**Change.** Folded the dBias column-sums into `grad_dense_partials_kernel` and dropped
`grad_bias_partials_kernel` entirely; the launcher is now a single fused `grad_dense_bias`
(one MFMA partials pass + `grad_dense_reduce` + `grad_bias_reduce`), **4 launches → 3**.
- In the dOut global→LDS transpose-staging the kernel already loads, each thread sums the
  dOut elements it loads into one fp32 register carried as a loop iter_arg (the tail rows
  beyond `M_b` zero-fill via the bounded descriptor, so they add 0). In that staging map
  thread `tid` always owns output column `n = tid % DDENSE_BN` and each column is co-owned
  by `DDENSE_THREADS/DDENSE_BN = 2` threads, so the per-thread partials are combined once
  through LDS at the end (reusing the now-dead sJ/sD staging region as fp32 scratch — **no
  extra smem, occupancy unchanged at 32 KB/WG**). Only the k-tile-0 workgroups (`k_off==0`)
  emit a column's bias partial, so each N-tile is written exactly once regardless of K-tiling.
- `dBias` partials scratch layout `(n_groups*SPLIT, N)` and `grad_bias_reduce_kernel` are
  unchanged; the split-`s` partition of rows over the m-tile loop differs from the old
  row-strided one but the per-(group) sum over all SPLIT partials is identical.

### `dense_bias` results (do_bench wall-clock; TF/s = 2·L·D·N / t)
| regime | T-1 (Phase-3 tree) | T (Phase-4 fused) | speedup | profile µs (was→now) |
|---|--:|--:|--:|--:|
| **uniform** | 6.94 ms / 148.6 TF/s | **5.52 ms / 186.7 TF/s** | **1.26×** | 6986 → **5592** (−20%) |
| **skew** | 2.10 ms / 74.9 TF/s | **1.45 ms / 108.5 TF/s** | **1.45×** | 2120 → **1455** (−31%) |

The fused partials pass costs only **+327 µs uniform / +89 µs skew** (the bias adds + the
one LDS combine) while removing the **1721 µs / 754 µs** dOut re-read — a net 1.25–1.46×.

### Head-to-head vs Triton (same run, `-test` cos=1.0000 both regimes)
| regime | FlyDSL (was→now) | Triton | result |
|---|--:|--:|---|
| uniform | 6.94 → **5.54 ms** | 7.76 ms | **FlyDSL 1.40×** (was 1.26×) |
| skew | 2.24 → **~1.45 ms** | ~1.41 ms | **near-tie** (Triton ~1.03×, was Triton **1.45×**) |

(Skew is stable across 3 samples: FlyDSL 1.450–1.468 ms vs Triton 1.402–1.433 ms. The one
North-Star case FlyDSL was *losing* by 1.45× is now within ~3% of Triton — effectively closed.
Triton's own `dense_bias` numbers here (uniform 7.76, skew ~1.42) differ a little from
EXP-2026-06-24a's 8.76/1.56 — run-to-run / machine-state variance — so the headline is the
same-run delta, not the cross-session absolute.)

### Gate: **PASS — fewer launches (4→3), no regression, correctness green.**
- **Correctness** cosine 0.999999 (example) / 1.0000 (bench `-test`) at **D=128 and D=256,
  uniform + skew** (empty/short skew groups → exact-0 bias via the never-entered m-loop).
- **dense_bias** 1.26×/1.45× end-to-end vs the Phase-3 tree; **uniform 1.40× vs Triton**,
  **skew now a near-tie** (the standing deficit). `grad_jagged` untouched.
- Kernel is still HBM-bound (uniform 1850 GB/s on the combined traffic; the fused pass trades
  a separate 2.3 TB/s dOut streaming pass for ~0 extra reads). Remaining `dense_bias` headroom
  is the fp32 partials round-trip through HBM (the `grad_dense_reduce` 1 GB read/write tail),
  now the dominant non-GEMM cost → a length-aware / smaller-`SPLIT` reduce is the next lever
  (Backlog), not the bias.

(Repo default restored to **D=128** after the session; `K=N` was temporarily 256 in
`jagged_dense_bmm.py` for the North-Star measurement, same temp-edit practice as prior EXP
blocks. The forward kernel's int64 `a_row_off/c_row_off` guard from EXP-2026-06-24a remains
in the working tree.)

## Current status (2026-06-25)

**Phase 4 done (EXP-2026-06-25a): dBias fused into the dDense partials pass.** The separate
`grad_bias_partials` kernel (a full HBM re-read of dOut) is gone; a single `grad_dense_bias`
launcher now runs one bf16-MFMA partials pass — writing **both** the fp32 dDense and dBias
partials by summing the dOut it already streams — plus two light reduce passes (**4 launches
→ 3**). **`dense_bias` 1.26× (uniform) / 1.45× (skew)** vs the Phase-3 build; vs Triton this is
now **1.40× uniform** and a **near-tie on skew** (~1.03× Triton), closing the one North-Star
case FlyDSL was losing (was Triton 1.45×, EXP-2026-06-24a). Correctness green at D=128 + D=256,
uniform + skew. The new dominant non-GEMM cost is the `grad_dense_reduce` fp32 partials
round-trip (≈1 GB). **Next: Phase 5** (autotune `SPLIT` / a length-aware reduce to cut the
partials tail — also the remaining `dense_bias`-skew lever in the Backlog — then example timing
+ style gate). (`K=N` restored to 128; D=256 is the measurement North Star.)

## Current status (2026-06-24)

Phase 3 done (EXP-2026-06-24b): `grad_jagged` M-coarsening (`COARSEN_M=2`) →
**1.31× vs Triton, both regimes** (was 1.26×/1.29×). North-Star head-to-head
established (EXP-2026-06-24a). **FlyDSL beats Triton on
`grad_jagged` (1.27×, both regimes) and on `dense_bias` uniform (1.16×); it loses
`dense_bias` skew (Triton 1.45×).** A latent **int32 byte-offset overflow** in
`grad_dense_partials` (only triggered at the North Star's `L≈7.86M`) was found and
fixed. The FlyDSL provider is now wired into the backward bench. **Reframing:**
Phase 3 (grad_jagged occupancy) was written for the old `b64/m512` launch-bound
regime — at the North Star the grad_jagged grid is ~123k WGs and the GPU is full,
so grad_jagged is already winning and is roofline-bound (~43% MFMA), not
occupancy-bound. The standing North-Star deficit is **`dense_bias` under skew**.
(`K=N` temporarily 256 in `jagged_dense_bmm.py` for measurement; restore to 128.)

## Current status (2026-06-23)
Tooling works end-to-end; baseline + Phase-0 sweep + production shapes characterized;
**Phase 1 done** (EXP-2026-06-23c, 1.34× @ D=256) and **Phase 2 done**
(EXP-2026-06-23d). `grad_dense_partials` is now a **bf16 MFMA** transposed GEMM
(`A=J.T, B=dOut.T`, contraction `m`, transpose-on-LDS-store since gfx942 has no HW
transpose, fp32 MFMA accum across the dynamic split m-loop): **4.60× @ D=256 uniform
/ 2.59× skew** end-to-end, **≈6.4×** on the partials kernel itself (MFMA util
0→16%, VALU-F32→0, Dependency-Wait −93%), correctness green (uniform+skew, D=128 &
D=256). The kernel is now **HBM/issue-bound** and the **reduce pass is the new
dominant tail**. The `block ≤ 256` col-tiling fix for `grad_bias`/`grad_dense_reduce`
is in (no D=128/256 regression); with the 32 KB-fixed MFMA partials it removes the
known D=512 blockers, but **D=512 end-to-end validation is deferred** per the D=256
North-Star decision. Repo default restored to D=128. **North Star now fully specified:
B=1024, D=256, Mi=7680** (matching the reference HSTU bench default; see "North Star
shape" section). All Phase-0…2 numbers above were collected at the pre-spec
`Mi=512` — the **next EXP block should re-measure the Phase-2 kernel at Mi=7680**
before quoting fully-specified North-Star figures. **Next: Phase 4** (fuse dBias
into dDense partials + lighter reduce tail — now the biggest remaining share), then
Phase 3 (`grad_jagged` occupancy). Optional: vectorize the partials transpose-staging
g2s (currently scalar) to push toward the HBM roofline.

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
- [x] 1a FMA fusion  [x] 1b output-tiling + loop-carried acc (fixed regs/LDS)
  [x] re-profile + EXP block (EXP-2026-06-23c). **Result: 1.34× @ D=256 / 1.21×
  @ D=128 — below the 1.5× gate.** F32-FMA achieved & VALU −41%, but Dependency-Wait
  *rose* and occupancy *fell* while runtime dropped → confirmed **latency/dep-bound on
  the scalar fp32 chain**, not VALU/occupancy. Cheap wins exhausted → **go to Phase 2**.
  (Side benefit: output-tiling bounds partials LDS to 32 KB → D=512 LDS blocker gone.)

### Phase 2 — MFMA-ize `grad_dense_partials` (the big lever)
- **Why:** `MFMA Instr=0`; scalar fp32 ceiling (81.7 TF/s) is 16× below bf16 MFMA
  (1307). This is `2026-06-18` §8 item #1.
- **Do:** replace the scalar-FMA reduction with bf16 MFMA on the transposed GEMM
  `C[k,n]=Σ_m J[m,k]·dOut[m,n]`, feeding fragments from LDS-staged tiles
  (CDNA4 LDS-read-transpose). Keep fp32 accumulate + the split-reduction skeleton.
- **Target:** MFMA Utilization > 0, large TF/s jump toward `grad_jagged`-class.
  **Gate:** `grad_dense_partials` within ~2× of `grad_jagged` TF/s; correctness
  (cosine > 0.999, uniform+skew) still passes via `example_…_bwd.py`.
- [x] MFMA partials  [x] validate correctness  [x] re-profile + EXP block
  (EXP-2026-06-23d). **Result: GATE PASS.** Transposed GEMM `A=J.T, B=dOut.T`,
  contraction `m` contiguous, transpose-on-LDS-store (gfx942 has no HW transpose),
  fp32 MFMA accum in place over the dynamic split m-loop. **4.60× @ D=256 uniform /
  2.59× skew end-to-end**; partials kernel **≈6.4×** (MFMA util 0→16%, VALU-F32→0,
  Dependency-Wait −93%). Now HBM/issue-bound; the **reduce tail dominates** → Phase 4
  is next. (Side fix: col-tiled bias/reduce kernels remove the D=512 thread-cap
  blocker; D=512 validation deferred per the D=256 North-Star decision.)

### Phase 3 — `grad_jagged` throughput / occupancy
- **Why:** MFMA already, but 1.06% occupancy, 139/304 CUs, ~12 µs → launch/tail
  bound; grid only 256 WGs.
- **Do:** more work per launch / better CU fill (e.g. finer N or K tiling to
  raise WG count, or persistent-block scheme); confirm the forward double-buffer
  pipeline is actually overlapping here.
- **Target:** occupancy ↑, CUs active ↑, TF/s ↑. **Gate:** measurable TF/s gain
  without correctness regression. (May be size-limited — Phase 0 informs this.)
- [x] grid/tiling change  [x] re-profile + EXP block (EXP-2026-06-24b).
  **Re-scoped at the North Star:** the original "raise WG count" premise was for the
  old launch-starved `b64/m512`; at `B=1024, Mi=7680` `grad_jagged` already beats
  Triton and is **dispatch/latency-bound** (4.3% occ, SPI 82%, ~123k tiny WGs), so
  the fix was the opposite — **M-coarsening** (`COARSEN_M=2`, fewer/longer WGs).
  **Result: 1.26×→1.31× vs Triton (uniform), 1.29×→1.31× (skew); GATE PASS**, no
  D=128 regression. Diminishing returns beyond `COARSEN_M=2`.

### Phase 4 — Fuse dBias into dDense partials + reduce-tail cleanup
- **Why:** `grad_bias_partials`/reduce are tiny tails reducing over the same `m`
  axis as dDense (`2026-06-18` §8 item #2). Shared `dOut` loads.
- **Do:** fold bias partial-sums into the dDense partials kernel; revisit a
  2-level reduction tree if `SPLIT` grew in Phase 1b.
- **Gate:** fewer kernels/launches, no regression; correctness holds.
- [x] fuse bias  [x] reduce tree (not needed — `SPLIT` stayed 4)  [x] re-profile + EXP
  block (EXP-2026-06-25a). **Result: GATE PASS.** dBias column-sums folded into
  `grad_dense_partials_kernel` (each thread sums the dOut it already stages; per-thread
  column partials combined once through LDS, k-tile-0 WGs only); `grad_bias_partials`
  removed; single fused `grad_dense_bias` launcher (**4→3 launches**). **dense_bias
  1.26× uniform / 1.45× skew** vs the Phase-3 build; **1.40× vs Triton uniform** and a
  **near-tie on skew** (closes the EXP-2026-06-24a deficit). Correctness cos 0.999999 at
  D=128 + D=256, uniform + skew. New dominant tail = the `grad_dense_reduce` fp32
  partials round-trip → Phase 5 / Backlog.

### Phase 5 — Autotune & integrate
- **Do:** autotune `SPLIT` / block sizes over the seq-length distribution; wire a
  timing+TFLOPs summary into `example_jagged_dense_bmm_bwd.py` (`2026-06-18`
  Phase 4); run the style gate.
- **Gate:** best config picked per regime; both `uniform`+`skew` green.
- [ ] autotune  [ ] example timing  [ ] style gate

## Backlog (not yet scheduled)
- **`dense_bias` under skew — MOSTLY CLOSED by Phase 4 (EXP-2026-06-25a).** Was the one
  North-Star case FlyDSL lost (Triton 1.45× — EXP-2026-06-24a). Fusing dBias into the
  dDense partials pass removed the redundant dOut re-read and brought skew to a **near-tie**
  (FlyDSL ~1.45 ms vs Triton ~1.41 ms, ~1.03× Triton). The *remaining* gap is the fixed
  `SPLIT=4 × n_groups` partials/reduce grid: skew's ~20% empty + many short groups still
  launch full `SPLIT` blocks and round-trip ~1 GB of fp32 dDense partials through HBM
  (`grad_dense_reduce`) on tiny work. **Do (ideas):** length-aware / dynamic `SPLIT` (fewer
  splits for short groups), skip empty groups, or a 2-level / fused reduce to cut the
  partials round-trip. **Gate:** beat Triton at `dense_bias` skew without regressing uniform
  or `jagged`. Measure with `bench_jagged_dense_bmm_bwd_perf.py --component dense_bias
  --regime skew -b 1024 -d 256 -kout 256 -mi 7680`.
- **Relax the `K == N == D` constraint (support non-square `D != Kout`).** The
  backward kernels currently collapse the dense reduction dim and output dim into a
  single compile-time constant (`K == N`), so they only cover *square* shapes. The
  reference HSTU bench (`bench_jagged_dense_bmm_perf.py`) separates `D` (= reduction
  `K`) from `Kout` (= output `N`) and can drive non-square shapes; our backward
  cannot express those. **Do:** split the single `D` constant into independent `K`
  (reduction) and `N` (output) in `jagged_dense_bmm_bwd.py` (and the example/profile
  harnesses), audit every tile/grid/LDS size that assumed `K == N` (the MFMA partials
  output-tiling `NK_TILES`/`NN_TILES`, the col-tiled reduce/bias `NRED_*`, scratch
  layout `(n_groups*SPLIT*K, N)`), and validate `K != N` (e.g. `D=256, Kout=512`)
  in `uniform`+`skew`. **Gate:** correctness (cosine > 0.999) at a non-square shape
  with no regression at the square North Star. Enables apples-to-apples coverage of
  the full HSTU bench config space, not just the square headline shapes.
- Tall `dDense` `(n_groups*N, K)` layout (revisit `2026-06-18` A2) if it removes a
  host transpose on the hot path.
- Port `dJagged` tweaks back to the forward kernel if shared.

## Notes on staleness
Numbers above are tied to the 2026-06-22 kernel source + the b64/m512/uniform
shape. Any kernel edit or shape change invalidates the table — re-run and add a
new dated EXP block rather than editing this one.
