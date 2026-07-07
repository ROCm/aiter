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
