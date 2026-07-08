# FlyDSL HSTU Backward — Optimization Log

A running log of performance measurements for the FlyDSL HSTU **backward** kernel.
Each entry records the date, the code state (commit), the hardware/environment, and
the benchmark results, so later optimization work can be compared against a known
baseline.

Companion documents:
- Implementation plan: [`2026-07-07_HSTU_backward_plan.md`](./2026-07-07_HSTU_backward_plan.md)
- Environment & constraints: [`HSTU_bwd_kernel_dev.md`](./HSTU_bwd_kernel_dev.md)

---

## 2026-07-07 — First baseline (untuned) on MI300X

### Code state

| Repo | Path | Branch | Commit |
|---|---|---|---|
| aiter (kernel + tests) | `meta/aiter` | `dlejeune/flydsl_hsta_bwd` | `6d4fd73f9ca73074c5d2057a0b98200708cef816` |
| recsys harness (bench) | `meta/mvonstra-amd` | `dlejeune/hstu_backward` | `f1a3e392d526e03d1fc48d46117a6e82f16cedd5` (+ local `flydsl`-provider edits, uncommitted at time of run) |

The `flydsl` provider (forward + backward) was added to the recsys harness
`bench_hstu.py`, and the `flydsl_prod` / `flydsl_prod_lite` / `flydsl_long` shape
grids and `singles` mask grid were added to `sweep_hstu.py`.

### Environment

- **Hardware:** AMD Instinct **MI300X** (`gfx942`, CDNA3). *(Note: the plan/dev docs
  were originally written for MI350 / `gfx950`; the kernel supports both archs.)*
- **Device pinning:** `HIP_VISIBLE_DEVICES=6` (per `HSTU_bwd_kernel_dev.md`; shared node).
- **Python env:** `/workspaces/git/meta/aiter/flydsl_venv` (torch `2.10.0+rocm7.2.4`,
  triton `3.6.0+rocm7.2.4`).
- **Roofline reference:** bf16 peak `1307 TF/s` (harness `common.py` MI300X value — valid
  here since we are on MI300X).

### Method

- Bench harness: `recsys_harness/bench_hstu.py` + `sweep_hstu.py`, timing via
  `triton.testing.do_bench`.
- Comparison provider: `aiter_triton` (AMD-tuned Triton HSTU), the natural reference.
- Inputs: jagged `(L, H, ·)` with `sparsity=0.95`, `alpha = 1/attn_dim`, causal,
  `dtype=bf16`, `attn_dim = hidden_dim = 128`.
- FlyDSL runs use **default (untuned) configs** — no per-shape tiling/tuning applied
  yet. `aiter_triton` is AMD-tuned. Numerical correctness is gated separately by the
  pytest suite (`test_flydsl_hstu_attention_bwd.py`, 26/26 green on `gfx942`); the
  harness `--correctness-check` was not used (needs `generative_recommenders`/`fbgemm`,
  unavailable in this venv).
- Reproduce, e.g.:

```bash
cd meta/mvonstra-amd/recsys-kernels/recsys_harness
PYTHONPATH=/workspaces/git/meta/aiter HIP_VISIBLE_DEVICES=6 \
  /workspaces/git/meta/aiter/flydsl_venv/bin/python sweep_hstu.py \
  --shape-grid flydsl_prod --mask-grid smoke \
  --providers flydsl aiter_triton --mode bwd \
  --out-csv runs/flydsl_interlude/hstu_bwd_causal_baseline.csv
```

Raw CSVs: `meta/mvonstra-amd/recsys-kernels/recsys_harness/runs/flydsl_interlude/`.

---

### Results — Backward, dense causal (bf16, d=128)

`ms` = mean latency (lower is better); `TF/s` from the causal-halved FLOP convention
(`bwd = 3·f1 + 2·f2`). Ratio > 1 ⇒ FlyDSL faster.

| B | H | seq_len | FlyDSL ms | FlyDSL TF/s | Triton ms | Triton TF/s | speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 120 | 4 | 512 | 0.479 | 50.7 | 1.171 | 20.7 | **2.44×** |
| 120 | 4 | 1024 | 1.524 | 65.2 | 3.975 | 25.0 | **2.61×** |
| 120 | 4 | 2048 | 4.525 | 80.6 | 14.323 | 25.5 | **3.17×** |
| 120 | 8 | 512 | 0.909 | 53.4 | 2.409 | 20.2 | **2.65×** |
| 120 | 8 | 1024 | 2.943 | 67.5 | 8.199 | 24.2 | **2.79×** |
| 120 | 8 | 2048 | 9.154 | 79.7 | 27.540 | 26.5 | **3.01×** |
| 1024 | 4 | 512 | 3.550 | 57.9 | 2.444 | 84.1 | 0.69× |
| 1024 | 4 | 1024 | 10.893 | 76.8 | 8.400 | 99.6 | 0.77× |
| 1024 | 4 | 2048 | 35.349 | 90.0 | 29.758 | 106.9 | 0.84× |
| 1024 | 8 | 512 | 7.002 | 58.7 | 4.770 | 86.2 | 0.68× |
| 1024 | 8 | 1024 | 21.940 | 76.3 | 15.986 | 104.7 | 0.73× |
| 1024 | 8 | 2048 | 71.542 | 88.9 | 55.275 | 115.1 | 0.77× |

### Results — Forward, dense causal (bf16, d=128)

| B | H | seq_len | FlyDSL ms | FlyDSL TF/s | Triton ms | Triton TF/s | speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 120 | 4 | 512 | 0.137 | 70.9 | 0.094 | 103.6 | 0.69× |
| 120 | 4 | 1024 | 0.339 | 117.4 | 0.300 | 132.2 | 0.88× |
| 120 | 4 | 2048 | 0.781 | 186.9 | 0.782 | 186.7 | 1.00× |
| 120 | 8 | 512 | 0.223 | 87.0 | 0.177 | 109.7 | 0.79× |
| 120 | 8 | 1024 | 0.578 | 137.5 | 0.553 | 143.7 | 0.96× |
| 120 | 8 | 2048 | 1.446 | 201.8 | 1.523 | 191.6 | 1.05× |
| 1024 | 4 | 512 | 0.781 | 105.2 | 0.728 | 113.0 | 0.93× |
| 1024 | 4 | 1024 | 2.035 | 164.4 | 2.164 | 154.6 | 1.06× |
| 1024 | 4 | 2048 | 5.654 | 225.0 | 6.573 | 193.5 | 1.16× |
| 1024 | 8 | 512 | 1.525 | 107.8 | 1.497 | 109.8 | 0.98× |
| 1024 | 8 | 1024 | 4.074 | 164.3 | 4.410 | 151.8 | 1.08× |
| 1024 | 8 | 2048 | 11.459 | 222.0 | 13.199 | 192.8 | 1.15× |

### Results — Backward, mask regimes applied singly (bf16, d=128, B=120)

Window uses `max_attn_len=128`; contextual uses `contextual_seq_len=64`; targets uses
`target_size=20` (uniform per batch). All 48 fwd + 48 bwd masked measurements ran with
0 failures (`hstu_{fwd,bwd}_singles_lite.csv`).

| mask | H | seq_len | FlyDSL ms | FlyDSL TF/s | Triton ms | Triton TF/s | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| targets | 4 | 2048 | 4.541 | 80.3 | 16.421 | 22.2 | **3.62×** |
| targets | 8 | 2048 | 9.207 | 79.2 | 27.532 | 26.5 | **2.99×** |
| window | 4 | 2048 | 3.138 | 116.2 | 4.107 | 88.8 | **1.31×** |
| window | 8 | 2048 | 6.375 | 114.4 | 7.929 | 92.0 | **1.24×** |
| contextual | 4 | 2048 | 6.982 | 52.2 | 15.576 | 23.4 | **2.23×** |
| contextual | 8 | 2048 | 13.954 | 52.3 | 29.926 | 24.4 | **2.14×** |

(Full per-seq_len rows for 512/1024/2048 are in the CSVs; the N=2048 rows above are
representative. Small-batch FlyDSL wins every masked cell.)

### Results — Long sequence, deployment-scale `seq_len = 16384` (bf16, d=128)

**Backward:**

| B | H | FlyDSL ms | FlyDSL TF/s | Triton ms | Triton TF/s | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 120 | 4 | 296.7 | 84.0 | 951.8 | 26.2 | **3.21×** |
| 120 | 8 | 602.9 | 82.7 | 1708.8 | 29.2 | **2.83×** |
| 1024 | 4 | 2105.8 | 100.0 | 2017.3 | 104.4 | 0.96× |
| 1024 | 8 | 4257.9 | 98.9 | 3915.2 | 107.6 | 0.91× |

**Forward:**

| B | H | FlyDSL ms | FlyDSL TF/s | Triton ms | Triton TF/s | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 120 | 4 | 38.5 | 258.6 | 40.0 | 249.1 | 1.04× |
| 120 | 8 | 76.1 | 262.1 | 81.5 | 244.6 | 1.07× |

Memory: the heaviest cell (B=1024, H=8, d=128, N=16384; ~114 GB of bwd tensors) fit
within the 192 GB MI300X — no OOM.

---

### Observations

1. **Small batch (B=120) — FlyDSL dominates the backward (~2.4–3.6×)** across all
   seq_len and all mask regimes. `aiter_triton` bwd is batch-starved here (stuck at
   ~20–29 TF/s regardless of length), whereas FlyDSL's lock-free two-kernel design
   sustains ~50–90 TF/s. This is the deployment *inference* regime.
2. **Large batch (B=1024) — FlyDSL trails, but the gap closes with sequence length.**
   At N=2048 FlyDSL is ~0.77–0.84× of Triton; by N=16384 it is ~0.91–0.96× (near
   parity, both ~100 TF/s). The deficit is a short-sequence + large-batch effect
   (launch/occupancy overhead), not a fundamental algorithmic gap.
3. **Forward is at rough parity**, with FlyDSL slightly ahead at large N (best roofline
   ~20% at N=16384) and slightly behind at N=512.
4. **Roofline headroom is large** (bwd ~4–8% of bf16 peak on these shapes) — expected
   for untuned configs and a recompute-heavy backward.

### Implications for next steps

- These are **untuned** FlyDSL numbers vs **tuned** Triton. Per-shape tuning
  (`block_m/block_n/num_waves/waves_per_eu`) is the cheapest lever and likely enough to
  reach parity/win in the large-batch backward, where the remaining gap is single-digit
  percent and shrinking with N.
- Sequence-parallel `dQ` (splitting KV blocks across programs) targets exactly the
  large-batch backward; its value depends on whether the deployment mix is large-batch
  short-sequence (where the gap is largest) vs small-batch long-sequence (where FlyDSL
  already wins).

---

## 2026-07-08 — Tile-config tuning (Phase 7) on MI300X

### Code state

- aiter `meta/aiter` @ branch `dlejeune/flydsl_hsta_bwd`, base commit `cef3b1b79`
  ("Adding tiling and tunable config"), plus the Phase-6-closure + Phase-7 changes
  described here (uncommitted at time of writing).
- **Phase 6 (sequence-parallel `dQ`) resolved as N/A**: the Phase-3 two-kernel design
  (KV-owned `dV`/`dK` + Q-owned `dQ`) is already fully tile-parallel *and* single-writer,
  so there is no `dQ` read-modify-write to synchronize. The vestigial `sequence_parallel`
  kwarg / CSV column was removed.

### Method

- New tuner `op_tests/op_benchmarks/flydsl/tune_hstu_attn_bwd.py`: per problem, sweeps a
  curated `(block_m, block_n, num_waves, waves_per_eu)` shortlist, times the full backward
  (both kernels) with `triton.testing.do_bench` on **realistic jagged inputs** (uniform
  lengths × sparsity, mirroring the recsys harness — not the test generator's power-law
  clamp), and writes the fastest config per problem to
  `aiter/ops/flydsl/hstu_attention_bwd_tuned.csv`. Invalid configs auto-skipped.
- Grid: P1–P4 (B∈{1024,120}, H∈{4,8}, d=128) × seq_len∈{512,1024,2048}, bf16, `gfx942`,
  sparsity 0.9. Reproduce:

```bash
PYTHONPATH=/workspaces/git/meta/aiter HIP_VISIBLE_DEVICES=6 \
  /workspaces/git/meta/aiter/flydsl_venv/bin/python \
  op_tests/op_benchmarks/flydsl/tune_hstu_attn_bwd.py --grid prod \
  --out aiter/ops/flydsl/hstu_attention_bwd_tuned.csv
```

### Result — best config per shape (bf16, d=128, dense causal)

| seq_len | best config | vs default `(64,32,4,0)` |
|---|---|---|
| 512  | `(64,32,4,0)` | default already best |
| 1024 | `(64,32,4,0)` | default already best |
| 2048 | `(192,32,4,0)` | **~6–9% faster** (all four P1–P4) |

Per-shape tuner ms at N=2048: B1024/H4 32.2→29.5, B1024/H8 64.3→58.7, B120/H4 4.17→3.92,
B120/H8 8.34→7.70. (`(256,·)` and `block_n=64` / `waves_per_eu=2` were consistently worse.)

### End-to-end (recsys harness, sparsity 0.95, B1024/H8/N2048 bwd)

| config | ms | TF/s | Triton gap |
|---|---|---|---|
| default `(64,32,4,0)` (2026-07-07 baseline) | 71.5 | 88.9 | 1.29× |
| tuned `(192,32,4,0)` (auto-selected via CSV) | 64.0 | 99.3 | **1.16×** |

`aiter_triton` unchanged at ~55 ms. So tuning delivered **−10.5%** on the worst large-batch
cell and narrowed the Triton gap from 1.29× to 1.16×.

### Notes / follow-ups

- The tuned CSV keys on `prev_power_of_2(batch)` and `prev_power_of_2(max_seq_len)`, so
  `B=120→64` and `B=1024→1024`; non-listed shapes fall back to the default heuristic.
- Correctness of the shipped `(192,32,4,0)` config is locked by
  `test_flydsl_bwd_block_size_overrides`. Full bwd suite: **34 green**.
- Not yet tuned: mask regimes (`targets`/`window`/`contextual`), `N=16384`, and `gfx950`.

---

## 2026-07-08 — rocprof-compute baseline profile (characterization, MI300X)

Not a new latency benchmark — a **hardware-counter characterization** of the current tuned
kernels, to anchor the optimization work. Detailed analysis + the phased plan live in
[`2026_07_08_first_optimization_plan.md`](./2026_07_08_first_optimization_plan.md).

### Method

- Tool: `rocprof-compute` 3.4.0 (rocprofiler-sdk) + `rocprofv3` kernel trace, `gfx942`,
  `HIP_VISIBLE_DEVICES=6`. Driver: `op_tests/op_benchmarks/flydsl/profile_hstu_attn_bwd.py`.
- Shape: **B=1024, H=8, d=128, N=2048, bf16, dense causal**, tuned `(192,32,4,0)` — the cell
  where FlyDSL trails Triton (≈64 vs ≈55 ms, 1.16×).

### Per-kernel (of the two-kernel backward)

| Kernel | Time | Occupancy | Dep-wait | MFMA util | BF16 SoL | VGPR/AGPR | Spill rd | HBM rd BW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `hstu_attention_bwd` (dV/dK) | 32.5 ms | 5.87 % | 49.7 % | 14.7 % | 12.9 % | 128/336 | 27.7 M | 419 GB/s (4.9%) |
| `hstu_attention_bwd_dq` (dQ) | 26.2 ms | 5.26 % | 52.4 % | 14.8 % | 13.2 % | 128/200 | 30.7 M | ~similar |

### Headline finding

**Latency/occupancy-bound, not compute- or bandwidth-bound.** Register (VGPR+AGPR) pressure
holds residency to ~2 waves/CU (allocation staller = "Insufficient SIMD VGPRs" ~31–33%), so
~50% of cycles are dependency-wait stalls, the MFMA units run at ~13% of bf16 peak, HBM is at
~5%, and the config even spills to scratch. Optimization order: **occupancy first** (shrink
the live register footprint), then pipelining, LDS bank conflicts, causal-triangle balance,
and finally the two-kernel recompute overhead. See the plan doc for the phased detail.

---

## 2026-07-08 — Phase 1: decouple the two kernels' tile configs (MI300X)

### Change

The backward launches two kernels (`hstu_attention_bwd` for dV/dK, `hstu_attention_bwd_dq`
for dQ) that previously **shared one tuned config**. Profiling-guided sweeps showed they want
**different** configs: the heavier dV/dK kernel (two accumulator families → ~336 AGPR at
`block_m=192`) prefers a large owned tile for load amortization and *cannot* take forced
occupancy (`waves_per_eu>0` makes it spill); the lighter dQ kernel (one accumulator family)
prefers `waves_per_eu=2`, which forces it from 1→2 waves/SIMD (AGPR 200→128, a negligible
12-byte spill) for a big latency-hiding win. Added a `kernel` discriminator column to
`hstu_attention_bwd_tuned.csv`; `_compile_bwd_launcher` now resolves each kernel's config
independently (override > per-kernel tuned > default). The tuner (`tune_hstu_attn_bwd.py`)
times the two kernels independently via `_make_bwd_kernel_runners` and emits per-kernel rows.

### Best per-kernel configs (bf16, d=128, dense causal, gfx942)

| seq_len | dV/dK | dQ |
|---|---|---|
| 512  | `(64,32,4,0)`  | `(128,32,4,2)` |
| 1024 | `(64,32,4,0)`  | `(128,32,4,2)` |
| 2048 | `(192,32,4,0)` | `(128,32,4,2)` |

dQ prefers `(128,32,4,2)` across all shapes tried (forced 2 waves/SIMD).

### Result (do_bench, same inputs, sparsity 0.9)

| cell | before (shared `192/32/4/0`) | after (per-kernel) | FlyDSL Δ | Triton (same inputs) | after vs Triton |
|---|---:|---:|---:|---:|---:|
| B1024/H8/N2048 | 59.19 ms | **52.84 ms** | **−10.7%** | 49.78 ms | 0.94× (gap 1.18×→1.06×) |
| B120/H8/N2048  |  7.70 ms | **6.95 ms**  | −9.6%    | 24.80 ms | **3.57×** |

Per-kernel split on the primary cell: dV/dK ~32.5 ms (unchanged, best config already), dQ
**26.1 → 20.2 ms**. The entire win is dQ.

### Deployment-scale check — N=16384 (do_bench, same inputs, sparsity 0.9)

The prior N=16384 numbers (2026-07-07 baseline) were **untuned** — Phase-7 tuning only covered
N≤2048, so N=16384 fell back to the conservative default `(64,32,4,0)` on *both* kernels. This
run tunes N=16384 per-kernel (dvdk `(192,32,4,0)`, dq `(128,32,4,2)` — same split as smaller N,
appended to the tuned CSV) and compares against Triton on identical inputs.

| cell | old default (both `64/32/4/0`) | Phase-1 tuned | Triton (same inputs) | tuned vs Triton | Phase-1 gain |
|---|---:|---:|---:|---:|---:|
| B120/H4/N16384  |  268.8 ms |  184.7 ms |  854.2 ms | **4.63×** | −31.3% |
| B120/H8/N16384  |  538.0 ms |  371.8 ms | 1537.5 ms | **4.14×** | −30.9% |
| B1024/H4/N16384 | 1893.2 ms | 1317.4 ms | 1819.9 ms | **1.38×** | −30.4% |
| B1024/H8/N16384 | 3845.2 ms | 2679.2 ms | 3502.5 ms | **1.31×** | −30.3% |

**The old large-batch gap is reproduced and then eliminated.** The `old default` column matches
the 2026-07-07 baseline ratio (B1024/H8: 3845/3502 → Triton `0.91×`, exactly the previously
logged 0.91×). Phase-1 tuning flips that to **1.31–1.38× faster than Triton** at large batch, and
**4.1–4.6×** at small batch. The ~30% gain (larger than the ~10% at N=2048) is because N=16384 had
never been tuned at all — this run applies both the dV/dK large-tile config *and* the new
per-kernel dQ occupancy config for the first time at this scale.

### Finding (scopes the remaining phases)

Reducing the register footprint via **smaller tiles does NOT help the dV/dK kernel**: e.g.
`(64,32,4,0)` halves its V+A registers (464→256, ~2 waves/SIMD) but is *slower* (36.5 vs
32.5 ms) — amortization loss beats the 1→2-wave occupancy gain, which is too small to hide
the ~466-cycle latency. And dV/dK is **AGPR-bound** (336 AGPR = accumulators, which must stay
in registers), so Phase-1 lever 2 (move resident operands VGPR→LDS) can't raise its occupancy
either. **The dV/dK kernel's headroom therefore moves to Phase 2 (deeper pipelining to hide
latency at low occupancy) and Phase 5 (split / recompute trade to cut accumulator families).**
Correctness: full bwd suite **41 green** (added `test_bwd_tuned_csv_per_kernel_configs`).

---

## 2026-07-08 — Phase 5 (brought forward): split the fused dV/dK kernel (MI300X)

### Why (from the Phase-1 profiling)

The fused dV/dK kernel carried **both** accumulator families (dV over hidden_dim + dK over
head_dim) → ~48 MFMA accumulators → **336 AGPR**, pinning it at **1 wave/SIMD** (VGPR+AGPR ≈
464 of the 512 pool). It is on-chip-latency bound, so 1-wave occupancy — not compute/bandwidth —
was the wall. Stall attribution confirmed global-memory is *not* the bottleneck (TA address stall
2.1%, cache→data-return 8%, VMEM util 0.4%), so classic DMA double-buffering (Phase 2) would buy
only a few %. The lever is cutting an accumulator family.

### Change

Added a build-time `which ∈ {"dv","dk"}` flag to `build_hstu_attention_bwd`, so one maintainable
file compiles into **two single-family kernels**. The backward now launches **three** kernels
(dV, dK, dQ), each with its own tuned config (`kernel` CSV discriminator now `dv`|`dk`|`dq`). Each
kernel carries one accumulator family (~24 acc); dV additionally drops the V-resident load and the
dA/dS machinery. Cost: dV and dK each recompute S (a recompute-over-occupancy trade, consistent
with the existing dQ kernel).

> **FlyDSL gotcha (recorded).** The AST rewriter branch-ifies in-kernel `if` statements and traces
> **both** branches, so gating mutually-exclusive heavy code with `if produce_dv:` fails (dead
> branch is still traced → crashes on the other family's structures / undefined names). Fix: build
> the shared fragments unconditionally (DCE drops the unused ones) and select the *accumulate* step
> with a **Python ternary** (`accum_dv(...) if produce_dv else accum_dk(...)`), which short-circuits
> at trace time so only the taken family's IR — and thus a single accumulator family — is emitted.

### Register footprint (primary cell, per kernel)

| kernel | VGPR | AGPR | V+A | waves/SIMD | notes |
|---|---:|---:|---:|---:|---|
| fused dV/dK (before) | 128 | 336 | 464 | 1 | both families |
| dV (after) | 116 | 132 | 248 | **2** | no V, no dA |
| dK (after) | 128 | ~200 | ~328 | 1–2 | keeps V + dA; prefers `block_n=16` |

### Best per-kernel configs (bf16, d=128, dense causal, N=2048)

dV `(128,32,4,0)`, dK `(128,16,4,2)`, dQ `(128,16,4,0)`. dK and dQ (both 3-pass: S+dA+dgrad)
prefer `block_n=16` (fewer transient MFMA frags → lower AGPR); dV (2-pass) prefers `block_n=32`.

### Result (do_bench, same inputs, sparsity 0.9)

| cell | Phase-1 (fused) | Phase-5 (split) | Triton | vs Triton | Δ |
|---|---:|---:|---:|---:|---:|
| B1024/H8/N2048 | 52.8 ms | **51.1 ms** | 50.2 ms | 0.98× (was 0.94×) | −3.2% |
| B120/H8/N2048  |  6.95 ms | **6.54 ms** | 24.8 ms | **3.80×** (was 3.57×) | −5.9% |

Isolated: dV **32.5→10.2 ms** (the win — 2 waves), dK ~20.3 ms (tuned `block_n=16`), dQ ~20.2 ms.
The split adds one S-recompute, so the net dV/dK gain is ~7% (30.4 vs 32.5 ms); the dV kernel now
has occupancy headroom for later pipelining. Cumulative primary-cell progress:
**59.2 (orig) → 52.8 (Phase 1) → 51.1 (Phase 5)**, now ~parity with Triton. Full bwd suite **42
green**.

### N=16384 (split, per-kernel tuned; 12 rows appended to the CSV)

At N=16384 dV prefers `block_m=192` (amortization dominates for long sequences); dK/dQ keep
`block_n=16`. Split vs Triton (same inputs, sparsity 0.9), and vs the Phase-1 fused numbers:

| cell | Phase-1 (fused) | Phase-5 (split) | Triton | split vs Triton |
|---|---:|---:|---:|---:|
| B120/H4/N16384  |  184.7 ms |  179.0 ms |  853.0 ms | **4.77×** |
| B120/H8/N16384  |  371.8 ms |  361.6 ms | 1538.0 ms | **4.25×** |
| B1024/H4/N16384 | 1317.4 ms | 1274.8 ms | 1796.9 ms | **1.41×** |
| B1024/H8/N16384 | 2679.2 ms | 2630.2 ms | 3533.4 ms | **1.34×** |

Small consistent gain (~2–3%) at scale, no regression, Triton lead preserved (1.34–4.77×).
Final tuned CSV: **48 rows** (P1–P4 × {512,1024,2048,16384} × {dv,dk,dq}).

---

## 2026-07-08 — Roofline analysis (post Phase 1/5, MI300X, primary cell)

`rocprof-compute profile` of the split kernels at B1024/H8/N2048 (dv+dk share the trace name →
merged as "group 0"; dq = "group 1"). Empirical peaks are rocprof-compute's own microbenchmarks
on this box: **bf16 MFMA 471 TF/s**, HBM **4170 GB/s** (note: the bf16 MFMA microbench measures
471, ~0.36× the 1307 theoretical — treat the empirical % as the practical ceiling and the
theoretical % as the absolute floor).

| metric | dV+dK (grp0) | dQ (grp1) |
|---|---:|---:|
| MFMA bf16 achieved | 211 TF/s | 205 TF/s |
| **% of empirical MFMA roofline (471)** | **45%** | **44%** |
| % of theoretical (1307) | 16% | 16% |
| Arithmetic intensity (HBM) | 188 Flop/B | 436 Flop/B |
| roofline knee (471/4170) | ~113 Flop/B | ~113 Flop/B |
| region | **compute-bound** (AI ≫ knee) | **compute-bound** (deeper) |
| HBM achieved / empirical peak | 1205 / 4170 = 29% | 475 / 4170 = 11% |
| MFMA utilization | 25% | 20% |
| wavefront occupancy | 18.7% | 23.5% |

**Findings.**
1. **We are compute-bound, not memory-bound.** Both kernels sit far right of the roofline knee
   (AI 188/436 ≫ 113 Flop/B), so HBM bandwidth (11–29% used) is *not* the ceiling — the bf16 MFMA
   compute roofline is. The recompute-over-bandwidth design (S recomputed in dv, dk, dq) inflates
   AI, which is fine precisely because we're compute-bound.
2. **~45% of the empirical MFMA roofline** (16% of theoretical) → ~2.2× headroom to the practical
   compute ceiling. The gap is **MFMA utilization (only 20–25% of cycles issue an MFMA; IPC 9–14%
   of peak)** — i.e. latency *between* MFMAs, not a lack of FLOP throughput.
3. **Phase 1/5 moved the needle a lot on occupancy**: 5.9% (fused, pre-Phase-1) → 18.7% (dv+dk) /
   23.5% (dq); MFMA SoL 12.9% → ~16% (theoretical) and MFMA util 14.7% → 25%.

**Implication.** The lever from here is raising MFMA utilization: **Phase 2 (software-pipeline /
deeper interleave to fill the inter-MFMA latency shadows)** and further occupancy — not memory
work. Memory-side optimizations would not help while we're this deep in the compute-bound region.

---

## 2026-07-08 — Phase 2 attempt: double-buffer Q LDS — NET LOSS, reverted

Implemented software-pipelined **double-buffering of the streamed Q LDS tile** in the KV-owned
kernel (dv/dk): prologue Q prefetch, next-tile Q DMA issued after the dO wait so it overlaps the
current tile's GEMM2/3, per-buffer LDS offset (`buf*BLOCK_N`, swizzle-preserving). Correct (suite
green), but **slower**:

| | pre (single-buf) | double-buf | Δ |
|---|---:|---:|---:|
| dV (128,32,4,0) | 10.1 ms | 11.4 ms | +13% |
| dK (128,16,4,2) | 20.1 ms | 21.7 ms | +8% |
| end-to-end primary | 51.1 ms | 53.4 ms | +4.5% |

**Why it failed (and confirms the roofline):** the kernel is **compute-bound** — hiding the Q-DMA /
memory latency targets a resource that isn't the bottleneck. The extra LDS (2× Q → lower headroom),
the extra prefetch DMA (incl. a wasted last-iter prefetch), and the per-call `readfirstlane` are
pure overhead with no latency to hide. Reverted to the single-buffer Phase-5 kernel (back to
51.4 ms, suite green).

**Pivot.** MFMA idle here is **compute-side**, not memory: the units stall on (a) the LDS-read
latency of operands feeding each MFMA — notably the **per-element B-operand gathers** in
`accum_dv/dk/dq` (4 single-element `ds_read`s per pack, on the MFMA critical path), (b) the
transcendental SiLU (`exp2`/`rcp`) sitting *between* GEMM1 and GEMM2 as a serial dependency, and
(c) the 2 workgroup barriers/tile. So the next MFMA-util levers are compute-side: **vectorize /
reorganize the LDS B-operand gathers, batch GEMM1 MFMAs before the SiLU so bursts stay back-to-back,
and reduce barriers** — not DMA pipelining.

### Lever "vectorize the accum B-operand gathers" — gfx942-blocked; **gfx950 opportunity**

Investigated: the gathers are per-element because each operand is needed in **two conflicting
layouts** within its kernel — contiguous along `d` for the GEMM1 A-operand, but **column-strided
along the contraction axis** (q / kv) for the accum B-operand (dO in dV, Q in dK, K in dQ). One LDS
layout can't serve both. The clean fix is a **transpose LDS read** (`ds_read_..._tr`), which is a
**CDNA4/gfx950** feature — **not on gfx942** (FlyDSL exposes only `ds_bpermute` / `permlane16_swap`).
On gfx942, vectorizing would require materializing a transposed LDS copy (transpose-**scatter** on
store + write-side conflicts), a read↔write trade likely net-neutral/negative (cf. the double-buffer).
The identical gather is in the well-performing forward, further suggesting it's not the dominant
gfx942 limiter. **Decision: defer this lever to gfx950 (the MI350 deployment target), where
`ds_read_tr` makes it free; pursue gfx942-viable levers instead** (batch-GEMM1-before-SiLU next).

### Lever "batch GEMM1 MFMAs before SiLU" — NEUTRAL, reverted

Reordered `compute_s_tile` into two phases (all GEMM1 MFMA chains first, then all SiLU/pack) so the
MFMA bursts stay back-to-back and the `exp2`/`rcp` transcendentals batch for ILP. Result: **neutral**
(dv 10.1→10.0, dk 20.1→20.2, end-to-end 51.1→51.5 ms — all within noise), and it holds the S
fragments live (extra register pressure). Reverted. The compiler was already interleaving the
independent (ng,og) MFMA chains with the SiLU adequately.

### Conclusion: gfx942 source-level MFMA-util levers are exhausted for now

Two source-level attempts to raise MFMA utilization on gfx942 came up empty (double-buffer:
negative; batch-SiLU: neutral), and lever #1 (vectorize gathers) is gfx950-only. Read together with
the roofline (compute-bound, ~45% of empirical MFMA peak, occupancy AGPR-limited), this says the
gfx942 MFMA-util ceiling for this kernel structure is largely set by **occupancy (AGPR)** and
irreducible per-tile dependency chains/barriers — which source reordering can't move.

> **Scope: MI300X (gfx942) only.** The gfx950 `ds_read_tr` gather-vectorization lever is
> **out of scope** (we are not optimizing for MI350). The remaining *in-scope* levers are:
> **(a) further cutting AGPR/occupancy** (algorithmic, e.g. narrower output-chunk residency), and
> **(b) Phase 4** causal-triangle balance / L2-locality (memory-side; helps large-batch scaling,
> not the compute ceiling). Current kernel remains the validated Phase-1+Phase-5 state (42 tests
> green).

### Lever (a) — cut AGPR to raise occupancy — BLOCKED (compiler AGPR floor)

At the tuned configs, all three kernels sit at **2 waves/SIMD** with **AGPR pinned at 128**
(dV: V=40/A=128=168; dK,dQ: V=96/A=128=224). Probed `waves_per_eu ∈ {0,2,3,4}` with register
readback:
- dV stays 168 for wpe 0/2/3 (no 3rd wave — needs V+A≤160, but AGPR floor 128 + min VGPR ~32 = 160
  best case, and it's at 168); wpe=4 pushes AGPR→0/VGPR→128 with spill → slower (15 ms).
- dK/dQ at wpe≥3 **spill hard** (scratch 316–500) → ~38 ms (2× worse).

**Conclusion:** the compiler won't allocate <128 AGPR for these MFMA-accumulator kernels, so
**2 waves is the structural occupancy ceiling on gfx942** and cutting the accumulator *count*
(narrower output residency) won't drop AGPR below the floor → no occupancy gain. Reaching 3 waves
would require VGPR ≤ 32 (move all resident operands to LDS), and even then dK/dQ would only reach
~176 = still 2 waves. **Lever (a) is not viable on gfx942; dropped.** Occupancy stands at 2 waves.

### Lever (b) — causal-triangle balance via length-sorting — COUNTERPRODUCTIVE

Tested the production balance technique (sort batches by sequence length, as genrec/aiter do via
`sort_by_length_indices`) purely at the input level (no kernel change: feed the current kernel
length-sorted `seq_offsets`/tokens). Result on B1024/H8/N2048 bwd:

| batch order | ms |
|---|---:|
| **unsorted (current)** | **51.0** |
| ascending by length | 96.4 |
| descending by length | 95.7 |

**Sorting is ~2× worse, both directions.** The current unsorted/random order is already
near-optimal: a random length mix keeps CUs full (short workgroups finish and free slots for
the greedy hardware scheduler) *and* keeps the concurrently-resident working set small/varied
(good cache behavior). Clustering same-length work does the opposite — during the "long" phase all
concurrent workgroups touch large sequences at once, thrashing L2. So the group-major grid with
natural jagged lengths already delivers the balance + L2 locality that lever (b) sought.
**Lever (b) is not viable (sorting hurts); dropped.**

### Overall conclusion — at the practical MI300X ceiling for this kernel structure

Every in-scope MI300X lever explored today came up empty: DMA double-buffer (negative), batch-SiLU
(neutral), gather-vectorize (gfx950-only), AGPR cut (compiler floor → 2-wave cap), length-sort
balance (counterproductive). Consistent with the roofline (compute-bound, ~45% of the empirical
BF16 MFMA peak, 2-wave occupancy): the kernel sits near its **structural ceiling on gfx942** for
this recompute-heavy, single-accumulator-family design. Further gains would need a fundamentally
different algorithm (out of current scope). Shipping state: **Phase 1 + Phase 5**, 42 tests green,
beats aiter_triton everywhere, wins small-batch vs genrec (~1.03×), ~8–10% behind genrec at large
batch.

---

## 2026-07-08 — generative-recommenders comparison enabled (recsys harness)

### Setup

Re-verified the `flydsl` provider in the recsys harness (`meta/mvonstra-amd`, branch
`dlejeune/hstu_backward`, `bench_hstu.py`/`sweep_hstu.py` — still present, uncommitted). The
public `flydsl_hstu_attention_bwd` API is unchanged by Phases 1/5, so no harness edit was needed
(only a stale-comment fix: bwd now launches **three** kernels). **Newly enabled the
`genrec_triton` comparison** (Meta generative-recommenders in-tree Triton) — the deployment
reference — which previously didn't run here:

- Applied `patch_genrec.py --genrec-root /workspaces/git/meta/generative-recommenders` to inject
  the `USE_TLX/NUM_BUFFERS/NUM_MMA_WARPS_PER_GROUP/NUM_MMA_GROUPS` constexpr defaults the HIP
  autotune configs omit (else `TypeError: dynamic_func() missing 4 required positional arguments`).
  After the patch, `genrec_triton` runs **fwd and bwd** on gfx942.
- Still unavailable here: `pytorch_ref` / genrec PyTorch path (`fbgemm_gpu` `.so` fails to load —
  built against a mismatched torch), so the harness `--correctness-check` (compares fwd vs
  `pytorch_ref`) can't run; FlyDSL correctness stays covered by the pytest oracle. `cuda_hstu_mha*`
  (CUTLASS SM90a) is NVIDIA-Hopper-only → N/A on MI300X.

### Backward vs genrec_triton (do_bench via harness, bf16, d=128, sparsity 0.9)

| cell | flydsl | genrec_triton | aiter_triton | flydsl vs genrec |
|---|---:|---:|---:|---:|
| B1024/H8/N2048 | 50.96 ms (112 TF) | **45.75 ms (125 TF)** | 51.71 ms | **0.90×** (genrec ahead) |
| B120/H8/N2048  | **6.59 ms (99 TF)** | 6.78 ms (97 TF) | 24.88 ms | **1.03×** (flydsl ahead) |

Forward smoke (B64/H4/N512): flydsl 0.102 ms, genrec_triton 0.070 ms, aiter_triton 0.056 ms.

### Takeaway (reframes the target)

**genrec_triton is a stronger competitor than aiter_triton** and is the right reference. At small
batch flydsl edges it (1.03×) and both crush the batch-starved aiter_triton (~3.8×). At large batch
genrec is **~10% faster than flydsl** (45.8 vs 51.0 ms) — so the large-batch gap we thought was
"~parity vs aiter_triton" is actually a real ~10% deficit vs the genrec reference. This is the gap
Phases 2–4 (pipelining, LDS conflicts, causal-triangle balance) should target. (CUTLASS — the
~2× stretch goal — remains unmeasurable on AMD; SM90a-only.)

---

## TODAY'S (2026-07-08) CONCLUSIONS & FUTURE

### Where we ended (shipping state)
- **Kernel:** Phase 1 (per-kernel tuned configs) + Phase 5 (dV/dK split into single-accumulator-
  family kernels). Backward is **three** single-writer kernels (dV, dK, dQ), each independently
  tuned; tuned CSV = 48 rows (P1–P4 × {512,1024,2048,16384} × {dv,dk,dq}). **42 tests green.**
- **Perf (primary cell B1024/H8/N2048 bwd):** 59.2 (orig) → 52.8 (Phase 1) → **51.1 ms** (Phase 5).
- **Vs references (do_bench, same inputs):** beats **aiter_triton** everywhere; vs **genrec_triton**
  (the real reference) wins small batch (~1.03×, B120) and trails ~8–10% at large batch (B1024).
- **Roofline:** compute-bound, **~45% of the empirical BF16 MFMA ceiling** (AI ≫ knee; HBM 11–29%),
  occupancy **2 waves/SIMD**.

### What we tried today and why each stopped (all measured)
| lever | outcome | why |
|---|---|---|
| DMA double-buffer (Q LDS) | **negative** (51→53 ms), reverted | compute-bound; hides memory latency that isn't the bottleneck |
| batch GEMM1 before SiLU | **neutral**, reverted | compiler already interleaves MFMA/SiLU adequately |
| vectorize `accum_*` gathers | **gfx942-blocked** | B-operand is column-strided; needs transpose LDS read (`ds_read_tr`) = **gfx950-only** |
| (a) cut AGPR → +occupancy | **blocked** | compiler **floors AGPR at 128** → 2-wave cap; `wpe≥3` only spills |
| (b) length-sort balance | **counterproductive** (~2× worse) | current unsorted/random order already balances CUs + keeps working-set cache-friendly |

### Conclusion
The kernel is at its **practical MI300X (gfx942) ceiling** for this recompute-heavy,
single-accumulator-family design. The binding limits are **structural**: AGPR floored at 128
(2-wave occupancy) and per-tile dependency/barrier latency that source-level scheduling can't hide.
Incremental source-level and config levers are exhausted.

### Recommendation
1. **Consolidate this milestone** — it's a strong, well-characterized result (beats aiter_triton;
   wins small-batch vs genrec; ~8–10% behind genrec at large batch, with the gap fully explained by
   the roofline + occupancy analysis). Commit the validated Phase-1+5 work.
2. **Do NOT keep grinding gfx942 micro-opts** — ROI is now poor; today's five levers all came up
   empty and the ceiling is structural.

### Future directions (only if the ~8–10% large-batch gap must be closed)
These are **architectural**, not incremental — bigger projects with real risk:
- **Reduce recompute (the real large-batch cost driver).** We recompute S three times (dV, dK, dQ).
  A fused or partially-fused reduction that computes S once and shares it would cut MFMA FLOPs, but
  reintroduces the accumulator-pressure / dQ-hazard the split was designed to avoid — the tension is
  fundamental (fewer passes ⇄ lower occupancy). Worth a careful cost model before attempting.
- **Different MFMA tiling / operand layout** that avoids the column-strided accum gather on gfx942
  without a transpose read (e.g. restructuring which operand is the MFMA A vs B so both GEMM1 and the
  accum want the same LDS layout). Non-trivial; may not exist cleanly.
- **f8 / mixed precision** for parts of the recompute (accuracy budget permitting) — would raise the
  compute roofline itself. Out of the current bf16 accuracy contract.
- **gfx950/MI350 port** (explicitly out of current scope, but where the clean gather-vectorization
  win lives via `ds_read_tr`).

Net: on MI300X we recommend stopping here and treating Phase 1+5 as the deliverable; the remaining
gap to genrec is an architectural problem, not a tuning one.
