## Summary

Add a **gfx950 large-M hybrid path** (`M >= 1024`) for `mhc_fused_post_pre`, building on upstream PR #3623 with minimal kernel changes. When `force_fused=True` on gfx950 at large M, dispatch switches from the monolithic `mhc_fused_post_pre_gemm_sqrsum` kernel to a tuned **post → gemm → big_fuse** chain that preserves numerical accuracy while improving latency on medium/large batch sizes.

> **Naming note:** tables below use `fuse_rmsnorm` (same meaning as `fuse_msnorm` in earlier EvoKernel logs). All rows here are `fuse_rmsnorm=False`.

## Technical idea

### Background (upstream PR #3623)

PR #3623 fuses `mhc_post` + `mhc_pre_gemm_sqrsum` into `mhc_fused_post_pre_gemm_sqrsum`, then calls `mhc_pre_big_fuse`. On gfx950, production dispatch already falls back to unfused `mhc_post` + `mhc_pre` at `M >= 1024` because the fused GEMM kernel is slower at large M. Benchmarks use `force_fused=True` to always exercise the fused kernel.

### Problem at large M

At `M >= 1024`, forcing the PR #3623 fused kernel has two issues on gfx950:

1. **Performance:** the single fused post+GEMM kernel does not match the L2-friendly behavior of a separate post write followed by GEMM read on the same stream.
2. **Accuracy:** `hip_fused_err` (bad-element ratio vs. reference on `layer_input`) can reach **0.6%–2.5%** on several `(M, hidden_size)` points, while the unfused path stays at ~1e-7.

### Hybrid approach (`mhc_large_m`)

For `gfx950` and `M >= 1024` with `force_fused=True`, use a **hybrid chain** instead of replacing upstream wholesale:

```
mhc_post (store_nt override)
  → mhc_pre_gemm_sqrsum (split-K from get_mhc_pre_splitk)
    → mhc_pre_big_fuse (use_nt cache-policy tuning)
```

Key tuning knobs (ported from EvoKernel, applied as small upstream patches):

| Knob | When | Effect |
|------|------|--------|
| `post_store_nt=0` | `M > 8 * CU` | Force RT store on post output so GEMM reads L2-hot residual |
| `use_nt=2` on big_fuse | `M == 8 * CU` (e.g. 2048 @ 256 CU) | Layer NT + residual RT in the fused chain |
| `get_mhc_pre_splitk` | `M >= 8192` | Prefer `(split_k=8, tile_k=64)` on gfx950 |
| Split big_fuse cache policies | all large M | Independent `gemm_load_nt / residual_nt / layer_nt` template dispatch |

Small-M paths (`M < 1024`) and gfx942 behavior are unchanged.

### What we intentionally did **not** change

- No replacement of upstream `mhc_fused_post_pre_gemm_sqrsum` for `M < 1024`
- No new fused MFMA post+GEMM kernel for large M
- Production default (`force_fused=False`, unfused at `M >= 1024`) unchanged

---

## Test setup

| Item | Value |
|------|-------|
| GPU | AMD Instinct **MI355X** (gfx950, **256 CU**) |
| Container | `rocm/atom:gfx950_latest` |
| HIP device | `HIP_VISIBLE_DEVICES=0` |
| Dtype | bf16 activations, fp32 mixes / GEMM acc |
| `hc_mult` | 4 |
| `fuse_rmsnorm` | False |
| Benchmark | `run_perftest`, warmup=2, iters=101 |
| Unfused column | `mhc_post` + `mhc_pre` |
| Fused column | `mhc_fused_post_pre(..., force_fused=True)` |
| Error metric | `hip_*_err` = `checkAllclose(layer_input)` bad-element **ratio** (0 = exact pass) |

Reproduce:

```bash
# upstream baseline (PR #3623 style)
docker run ... -v aiter-upstream-main:/workspace/aiter \
  python3 run_upstream_pr3623_test.py -m 1024 2048 8192 65536 -n 4096 7168

# this branch
REPO=aiter-mhc_large_m docs/EvoKernel/scripts/run_mhc_large_m_atom.sh
```

---

## Performance (µs, lower is better)

### Upstream PR #3623 (`mhc_fused_post_pre`, `force_fused=True`)

| m | hidden_size | unfused_us | fused_us | triton_us |
|--:|------------:|-----------:|---------:|----------:|
| 1024 | 4096 | 44.28 | 42.33 | 331.83 |
| 2048 | 4096 | 69.63 | 68.61 | 782.49 |
| 8192 | 4096 | 243.94 | 258.29 | — |
| 65536 | 4096 | 1851.91 | 1861.68 | — |
| 1024 | 7168 | 62.24 | 63.64 | 558.42 |
| 2048 | 7168 | 109.44 | 115.98 | 1472.85 |
| 8192 | 7168 | 415.74 | 412.25 | — |

### This branch (`mhc_large_m` hybrid, `force_fused=True`)

| m | hidden_size | unfused_us | fused_us | triton_us |
|--:|------------:|-----------:|---------:|----------:|
| 1024 | 4096 | 43.12 | 42.86 | 331.67 |
| 2048 | 4096 | 69.79 | 70.03 | 782.34 |
| 8192 | 4096 | 246.62 | 251.54 | — |
| 65536 | 4096 | 1833.30 | 1840.16 | — |
| 1024 | 7168 | 62.15 | 62.35 | 558.66 |
| 2048 | 7168 | 108.75 | 112.43 | 1470.90 |
| 8192 | 7168 | 413.40 | **396.97** | — |

### Fused speedup vs upstream (Δ = upstream − hybrid)

| m | hidden_size | upstream fused | hybrid fused | Δ (µs) |
|--:|------------:|---------------:|-------------:|-------:|
| 1024 | 4096 | 42.33 | 42.86 | −0.5 (tie) |
| 2048 | 4096 | 68.61 | 70.03 | −1.4 (tie) |
| 8192 | 4096 | 258.29 | 251.54 | **+6.8** |
| 65536 | 4096 | 1861.68 | 1840.16 | +21.5 |
| 1024 | 7168 | 63.64 | 62.35 | +1.3 |
| 2048 | 7168 | 115.98 | 112.43 | **+3.6** |
| 8192 | 7168 | 412.25 | 396.97 | **+15.3** |

**Highlights**

- Largest win: **hidden_size=7168, M=8192** — hybrid fused **397 µs** vs upstream **412 µs** (~3.7% faster).
- Unfused baselines are within ~1–4 µs (noise-level) across configs.
- `triton_us` unavailable at `M >= 8192` on both stacks (same as upstream).
- `M=65536, hidden_size=7168` OOMs in the combined sweep; `65536/4096` OK.

---

## Accuracy (`hip_*_err`, bad-element ratio vs reference)

### Upstream PR #3623

| m | hidden_size | hip_unfused_err | hip_fused_err |
|--:|------------:|----------------:|--------------:|
| 1024 | 4096 | 7.15e-07 | **0.00639** |
| 2048 | 4096 | 2.38e-07 | **0.02493** |
| 8192 | 4096 | 2.98e-07 | **0.00329** |
| 65536 | 4096 | 2.50e-07 | **0.00581** |
| 1024 | 7168 | 4.09e-07 | **0.00856** |
| 2048 | 7168 | 2.04e-07 | **0.01135** |
| 8192 | 7168 | 1.53e-07 | 1.53e-07 |

### This branch (`mhc_large_m`)

| m | hidden_size | hip_unfused_err | hip_fused_err |
|--:|------------:|----------------:|--------------:|
| 1024 | 4096 | 2.38e-07 | 2.38e-07 |
| 2048 | 4096 | 2.38e-07 | 2.38e-07 |
| 8192 | 4096 | 2.98e-07 | 2.98e-07 |
| 65536 | 4096 | 2.57e-07 | 2.57e-07 |
| 1024 | 7168 | 8.17e-07 | 8.17e-07 |
| 2048 | 7168 | 2.72e-07 | 2.72e-07 |
| 8192 | 7168 | 2.72e-07 | 2.72e-07 |

**Takeaway:** hybrid fused path matches reference at ~1e-7 bad-element ratio on all tested large-M points. Upstream forced-fused path shows elevated `hip_fused_err` on most 4096 configs and several 7168 configs.

---

## Full tables (with Triton)

<details>
<summary>Upstream PR #3623 full results</summary>

| m | hidden_size | hc_mult | fuse_rmsnorm | unfused_us | hip_unfused_err | fused_us | hip_fused_err | triton_us | triton_fused_err |
|--:|------------:|--------:|:-------------:|-----------:|----------------:|---------:|--------------:|----------:|-----------------:|
| 1024 | 4096 | 4 | False | 44.2814 | 7.15e-07 | 42.3292 | 0.00639486 | 331.831 | 0.00372982 |
| 2048 | 4096 | 4 | False | 69.6253 | 2.38e-07 | 68.6099 | 0.0249293 | 782.486 | 0.0143178 |
| 8192 | 4096 | 4 | False | 243.943 | 2.98e-07 | 258.286 | 0.00328824 | nan | nan |
| 65536 | 4096 | 4 | False | 1851.91 | 2.50e-07 | 1861.68 | 0.00580556 | nan | nan |
| 1024 | 7168 | 4 | False | 62.24 | 4.09e-07 | 63.6434 | 0.00855609 | 558.416 | 0.00441306 |
| 2048 | 7168 | 4 | False | 109.436 | 2.04e-07 | 115.979 | 0.0113461 | 1472.85 | 0.0051196 |
| 8192 | 7168 | 4 | False | 415.742 | 1.53e-07 | 412.25 | 1.53e-07 | nan | nan |

</details>

<details>
<summary>mhc_large_m full results</summary>

| m | hidden_size | hc_mult | fuse_rmsnorm | unfused_us | hip_unfused_err | fused_us | hip_fused_err | triton_us | triton_fused_err |
|--:|------------:|--------:|:-------------:|-----------:|----------------:|---------:|--------------:|----------:|-----------------:|
| 1024 | 4096 | 4 | False | 43.1174 | 2.38e-07 | 42.8608 | 2.38e-07 | 331.669 | 0.00657678 |
| 2048 | 4096 | 4 | False | 69.7913 | 2.38e-07 | 70.0307 | 2.38e-07 | 782.343 | 8.70e-06 |
| 8192 | 4096 | 4 | False | 246.619 | 2.98e-07 | 251.541 | 2.98e-07 | nan | nan |
| 65536 | 4096 | 4 | False | 1833.30 | 2.57e-07 | 1840.16 | 2.57e-07 | nan | nan |
| 1024 | 7168 | 4 | False | 62.1488 | 8.17e-07 | 62.3508 | 8.17e-07 | 558.664 | 0.00945813 |
| 2048 | 7168 | 4 | False | 108.755 | 2.72e-07 | 112.426 | 2.72e-07 | 1470.90 | 0.0136537 |
| 8192 | 7168 | 4 | False | 413.399 | 2.72e-07 | 396.973 | 2.72e-07 | nan | nan |

</details>

---

## Test plan

- [x] gfx950 large-M perf sweep (`M ∈ {1024, 2048, 8192, 65536}`, `hidden_size ∈ {4096, 7168}`)
- [x] Numerical check: `hip_fused_err` at ~1e-7 on all hybrid fused points
- [x] Compare against upstream PR #3623 baseline on same docker / GPU
- [ ] CI `op_tests/test_mhc.py` (if upstream adds gfx950 runner)
- [ ] Isolated `M=65536, hidden_size=7168` run after GPU reset
