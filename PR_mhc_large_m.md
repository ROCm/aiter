## Summary

Add **gfx950 large-M additive kernels** for `mhc_fused_post_pre` on top of upstream PR #3623. When `force_fused=True` and **`M > 1024`**, dispatch uses a new HIP entry point `mhc_post_pre_large_m` (post → gemm → tuned big_fuse). **PR #3623 kernels are not modified**; all tuning lives in newly added symbols at the end of `mhc_kernels.cu`.

> **Naming note:** tables use `fuse_rmsnorm` (= `fuse_msnorm` in older logs). All rows here are `fuse_rmsnorm=False`.

## Technical idea

### Background (upstream PR #3623)

PR #3623 fuses `mhc_post` + `mhc_pre_gemm_sqrsum` into `mhc_fused_post_pre_gemm_sqrsum`, then calls upstream `mhc_pre_big_fuse`. On gfx950, production already falls back to unfused `mhc_post` + `mhc_pre` at `M >= 1024`. Benchmarks use `force_fused=True` to always exercise the fused kernel.

### Problem at large M

Forcing PR #3623 fused path at large M on gfx950:

1. **Performance:** monolithic fused post+GEMM misses L2-friendly post→gemm sequencing.
2. **Accuracy:** `hip_fused_err` can reach **0.3%–3.2%** on several `(M, hidden_size)` points.

### Additive large-M path (`mhc_large_m`)

**Dispatch** (`aiter/ops/mhc.py`): `force_fused=True` + gfx950 + **`m > 1024`** → `mhc_fused_post_pre_large_m()`.

**New HIP symbols only** (appended after PR #3623 code in `mhc_kernels.cu`):

```
mhc_post_pre_large_m
  ├─ mhc_post_large_m        (store_nt override)
  ├─ mhc_pre_gemm_sqrsum     (existing, unchanged)
  └─ mhc_pre_big_fuse_large_m (new tuned big_fuse templates)
```

| Knob | When | Effect |
|------|------|--------|
| `post_store_nt=0` | `M > 8 * CU` | RT store on post output for L2-hot GEMM read |
| `use_nt=2` on big_fuse_large_m | `M == 8 * CU` | Layer NT + residual RT in fused chain |
| `get_mhc_pre_splitk_large_m` | `M >= 8192` | Prefer `(split_k=8, tile_k=64)` on gfx950 |
| Split cache policies | large M | Independent `gemm_load_nt / residual_nt / layer_nt` in **new** big_fuse kernel |

**Unchanged:** `M <= 1024` (incl. M=1024) still uses upstream PR #3623 fused path. Production default (`force_fused=False`) unchanged.

---

## Test setup

| Item | Value |
|------|-------|
| GPU | AMD Instinct **MI355X** (gfx950, **256 CU**) |
| Container | `rocm/atom:gfx950_latest` |
| HIP device | `HIP_VISIBLE_DEVICES=0` (idle GPU) |
| Dtype | bf16 activations, fp32 mixes / GEMM acc |
| `hc_mult` | 4 |
| `fuse_rmsnorm` | False |
| Benchmark | `run_perftest`, warmup=2, iters=101 |
| Unfused column | `mhc_post` + `mhc_pre` |
| Fused column | `mhc_fused_post_pre(..., force_fused=True)` |
| Error metric | `hip_*_err` = `checkAllclose(layer_input)` bad-element **ratio** |

Reproduce:

```bash
# A/B sweep: upstream vs mhc_large_m (post_pre segment only)
python3 /tmp/bench_mhc_post_pre_large_m_idle.py upstream   # ROCm/aiter main
python3 /tmp/bench_mhc_post_pre_large_m_idle.py hybrid     # this branch

# large-M dedicated test (M > 1024 only)
python3 op_tests/test_mhc_large_m.py -m 2048 8192 65536 -n 4096 7168

# full upstream-style sweep (M=1024 still PR #3623 on this branch)
python3 op_tests/test_mhc.py -m 1024 2048 8192 65536 -n 4096 7168
```

Baseline repo: `/home/yinfeliu/aiter-upstream-main` (ROCm/aiter main).  
This branch: `/home/yinfeliu/aiter-mhc_large_m`.

---

## Performance (µs, lower is better)

*Updated **2026-06-10** after additive-kernel refactor, idle GPU, `rocm/atom:gfx950_latest`.*

### Upstream PR #3623 (`force_fused=True`, all M)

| m | hidden_size | unfused_us | fused_us | hip_fused_err | triton_us |
|--:|------------:|-----------:|---------:|--------------:|----------:|
| 1024 | 4096 | 43.78 | 41.16 | 2.38e-07 | 331.35 |
| 2048 | 4096 | 69.56 | 69.25 | **0.02452** | 781.42 |
| 8192 | 4096 | 245.89 | 253.79 | **0.00333** | — |
| 65536 | 4096 | 1857.03 | 1856.44 | **0.03250** | — |
| 1024 | 7168 | 61.92 | 62.84 | **0.02589** | 560.02 |
| 2048 | 7168 | 107.98 | 116.25 | **0.00884** | 1469.49 |
| 8192 | 7168 | 417.09 | 409.75 | **0.02583** | — |

### This branch (`mhc_large_m`, `force_fused=True`)

| m | hidden_size | unfused_us | fused_us | hip_fused_err | triton_us | path @ fused |
|--:|------------:|-----------:|---------:|--------------:|----------:|:-------------|
| 1024 | 4096 | 43.47 | 41.25 | 0.01954 | 331.61 | PR #3623 (M≤1024) |
| 2048 | 4096 | 70.60 | **69.57** | 4.77e-07 | 781.88 | **large_m** |
| 8192 | 4096 | 244.46 | **248.50** | 4.17e-07 | — | **large_m** |
| 65536 | 4096 | 1863.17 | **1844.54** | 3.09e-07 | — | **large_m** |
| 1024 | 7168 | 62.04 | 63.85 | 0.00851 | 560.02 | PR #3623 (M≤1024) |
| 2048 | 7168 | 111.04 | **110.91** | 2.04e-07 | 1472.46 | **large_m** |
| 8192 | 7168 | 414.14 | **401.32** | 4.77e-07 | — | **large_m** |

### Fused speedup vs upstream (Δ = upstream − this branch, positive = this branch faster)

| m | hidden_size | upstream fused | this branch fused | Δ (µs) |
|--:|------------:|---------------:|------------------:|-------:|
| 1024 | 4096 | 41.16 | 41.25 | −0.1 (tie) |
| 2048 | 4096 | 69.25 | 69.57 | −0.3 (tie) |
| 8192 | 4096 | 253.79 | 248.50 | **+5.3** |
| 65536 | 4096 | 1856.44 | 1844.54 | **+11.9** |
| 1024 | 7168 | 62.84 | 63.85 | −1.0 (upstream) |
| 2048 | 7168 | 116.25 | 110.91 | **+5.3** |
| 8192 | 7168 | 409.75 | 401.32 | **+8.4** |

**Highlights**

- **M > 1024:** new kernel gives **~1e-7** `hip_fused_err` vs upstream **0.3%–3.2%** on the same shapes.
- **Performance:** wins at **8192+** (both hidden sizes) and **2048/7168**; **65536/4096** wins in sweep (+11.9 µs).
- **M = 1024:** still upstream PR #3623 path; perf within noise, accuracy similar to upstream forced-fused behavior.

### Isolated benchmark: `M=65536, hidden_size=4096`

Fresh container, single case only:

| stack | unfused_us | fused_us | hip_fused_err |
|:------|----------:|---------:|--------------:|
| upstream | 1853.00 | 1862.48 | **0.00196** |
| this branch | 1856.33 | **1841.79** | **2.76e-07** |
| **Δ (upstream − branch)** | −3.3 | **+20.7** | — |

---

## Accuracy summary (`hip_fused_err`)

| region | upstream PR #3623 | this branch |
|--------|-------------------|-------------|
| M = 1024 | 0%–2.6% bad elements | similar (still PR #3623 kernel) |
| **M > 1024** | **0.3%–3.2%** | **~1e-7** on all tested points |

---

## `test_mhc_large_m.py` columns explained

`op_tests/test_mhc_large_m.py` (M > 1024 only) reports three latency columns:

| column | meaning |
|--------|---------|
| `large_m_us` | Direct call to **`mhc_fused_post_pre_large_m()`** (new API) |
| `dispatch_us` | Call via **`mhc_fused_post_pre(..., force_fused=True)`** — should route to the same kernel when `m > 1024` |
| `unfused_us` | Baseline `mhc_post` + `mhc_pre` |

Example from quick sanity run (2048/4096): `large_m_us=69.46`, `dispatch_us=70.65` — within ~1 µs, confirms dispatch wiring. Full sweep numbers above use the idle-GPU A/B script and are the authoritative PR table.

---

## Test plan

- [x] PR #3623 kernels left unchanged (additive symbols only)
- [x] gfx950 large-M perf sweep vs upstream (`M ∈ {1024, 2048, 8192, 65536}`)
- [x] `hip_fused_err` ~1e-7 for **M > 1024** fused path
- [x] `op_tests/test_mhc_large_m.py` dedicated large-M test
- [x] Idle-GPU re-bench in `rocm/atom:gfx950_latest` (2026-06-10, post-refactor)
- [ ] CI gfx950 runner (if upstream adds)
