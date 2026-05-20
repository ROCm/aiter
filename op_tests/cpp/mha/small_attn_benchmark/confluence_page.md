# Benchmark CK_PR_6764

Small-sequence MHA forward/backward: **ck_pr_6764** (CK tile, `fwd_v3=0`, `bwd_v3=0`) vs **jax_unfused** (reference einsum; scenarios 1–2 fwd use THD + `cu_seqlens`, same packing as CK group mode). Times are mean step latency in **ms** (warmup 5, repeat 25).

---

## Test environment

| Item | Value |
|------|--------|
| **Host** | `smci355-ccs-aus-m07-05` (MI355 class system) |
| **Benchmark harness** | [aiter `op_tests/cpp/mha/small_attn_benchmark`](https://github.com/ROCm/aiter/tree/veergopu/check_ck_small_seq/op_tests/cpp/mha/small_attn_benchmark) |
| **CK run** | `cd op_tests/cpp/mha && bash build_mha.sh` then `cd small_attn_benchmark && ./run_all.sh` |
| **JAX run** | `python3 run_jax_benchmark.py all` (after CK CSVs exist; adds `jax_unfused(ms)` column) |
| **Graphs** | Paste screenshots under each **Graph** heading; optional PNGs from `python3 gen_confluence_assets.py` |
| **Results** | `results/scenario*/{fwd,bwd}.csv` |

### Composable Kernel (CK)

| Item | Value |
|------|--------|
| **Repository** | [ROCm/composable_kernel](https://github.com/ROCm/composable_kernel) |
| **Branch** | [veergopu/add_6764](https://github.com/ROCm/composable_kernel/tree/veergopu/add_6764) |
| **Commit** | [1eafdc8](https://github.com/ROCm/composable_kernel/commit/1eafdc8bd705bdc201563905d505331e39eb17cb) |

### aiter

| Item | Value |
|------|--------|
| **Repository** | [ROCm/aiter](https://github.com/ROCm/aiter) |
| **Branch** | [veergopu/check_ck_small_seq](https://github.com/ROCm/aiter/tree/veergopu/check_ck_small_seq) |
| **Commit** | [88e129697a6d](https://github.com/ROCm/aiter/commit/88e129697a6db9763da5ff5cb60d242c70dfa86e) |

---

## Common configuration

| Parameter | CK | JAX unfused |
|-----------|-----|-------------|
| Precision | bf16 | bf16 |
| Layout | BHSD (`iperm=1`, `operm=1`) | BSHD (einsum); varlen fwd via THD + `cu_seqlens` |
| Heads / dim | h=32, d=128 | same |
| Mask / bias / dropout | non-causal, no bias, p_drop=0 | non-causal |
| Scale | `scale_s=0` → 1/√128 | 1/√128 |
| Batch sizes | 2048, 4096 | same (from CSV rows) |
| Warmup / repeat | 5 / 25 | 5 / 25 |
| Length RNG (scen 1–2 fwd) | seed 6764, i.i.d. uniform {2,…,P} per row | same lists |

**Table columns:** `jax/ck` = JAX time ÷ CK time (**>1** means CK is faster).

---

## Customer requests vs what we ran

| # | Customer request | Included? | How we ran it | Gap |
|---|------------------|-----------|---------------|-----|
| **1** | sq, skv ≤ 16, padding + varlen | Partial | Fwd: CK group packed; JAX THD+cu_seqlens. Bwd: uniform batch P. | No physical padding; bwd not per-row varlen. |
| **2** | sq=1, skv ≤ 16 varlen | Partial | Fwd: CK group; JAX cross + THD KV. Bwd: batch s_q=1, s_kv=P. | Same padding/bwd limits. |
| **3** | sq=skv=16 fixed | Yes | Row in scenario_3_4 sweep. | — |
| **4** | sq=skv=17 fixed (priority) | Yes | Row in scenario_3_4 sweep. | CK bwd outlier at s=17 (see below). |

---

## Scenario 1 — sq, skv ≤ 16 (packed varlen Q and KV)

**Script:** `scenario_1.sh` · **CK fwd:** `-mode=1` · **CK bwd:** `-mode=0`, uniform `s_q=s_kv=P` · **JAX fwd:** THD + `cu_seqlens` · **JAX bwd:** dense `(B,P,…)`.

### Forward

#### Table — forward (P (max logical length) sweep)

| batch | s_q | s_kv | ck_pr_6764 (ms) | jax_unfused (ms) | jax/ck |
| --- | --- | --- | --- | --- | --- |
| 2048 | 2 | 2 | 0.099 | 0.263 | 2.66 |
| 2048 | 3 | 3 | 0.102 | 0.292 | 2.86 |
| 2048 | 4 | 4 | 0.102 | 0.316 | 3.10 |
| 2048 | 5 | 5 | 0.107 | 0.364 | 3.40 |
| 2048 | 6 | 6 | 0.109 | 0.388 | 3.56 |
| 2048 | 7 | 7 | 0.109 | 0.419 | 3.84 |
| 2048 | 8 | 8 | 0.111 | 0.443 | 3.99 |
| 2048 | 9 | 9 | 0.120 | 0.541 | 4.51 |
| 2048 | 10 | 10 | 0.123 | 0.538 | 4.37 |
| 2048 | 11 | 11 | 0.129 | 0.611 | 4.74 |
| 2048 | 12 | 12 | 0.133 | 0.605 | 4.55 |
| 2048 | 13 | 13 | 0.133 | 0.655 | 4.92 |
| 2048 | 14 | 14 | 0.146 | 0.669 | 4.58 |
| 2048 | 15 | 15 | 0.142 | 0.717 | 5.05 |
| 2048 | 16 | 16 | 0.168 | 0.705 | 4.20 |
| 4096 | 2 | 2 | 0.192 | 0.429 | 2.23 |
| 4096 | 3 | 3 | 0.194 | 0.503 | 2.59 |
| 4096 | 4 | 4 | 0.216 | 0.561 | 2.60 |
| 4096 | 5 | 5 | 0.222 | 0.645 | 2.91 |
| 4096 | 6 | 6 | 0.228 | 0.688 | 3.02 |
| 4096 | 7 | 7 | 0.242 | 0.751 | 3.10 |
| 4096 | 8 | 8 | 0.241 | 0.779 | 3.23 |
| 4096 | 9 | 9 | 0.247 | 0.957 | 3.87 |
| 4096 | 10 | 10 | 0.257 | 0.968 | 3.77 |
| 4096 | 11 | 11 | 0.260 | 1.086 | 4.18 |
| 4096 | 12 | 12 | 0.271 | 1.076 | 3.97 |
| 4096 | 13 | 13 | 0.275 | 1.196 | 4.35 |
| 4096 | 14 | 14 | 0.278 | 1.222 | 4.40 |
| 4096 | 15 | 15 | 0.279 | 1.320 | 4.73 |
| 4096 | 16 | 16 | 0.294 | 1.290 | 4.39 |

#### Graph — Scenario 1 forward — CK vs JAX vs P (batch 2048 & 4096)

*(Screenshot: paste below. Optional reference PNG: `confluence_assets/scenario1_fwd.png` from `python3 gen_confluence_assets.py`.)*

---




### Backward

#### Table — backward (P (max logical length) sweep)

| batch | s_q | s_kv | ck_pr_6764 (ms) | jax_unfused (ms) | jax/ck |
| --- | --- | --- | --- | --- | --- |
| 2048 | 2 | 2 | 0.217 | 0.394 | 1.82 |
| 2048 | 3 | 3 | 0.257 | 0.443 | 1.72 |
| 2048 | 4 | 4 | 0.276 | 0.501 | 1.82 |
| 2048 | 5 | 5 | 0.294 | 0.542 | 1.84 |
| 2048 | 6 | 6 | 0.309 | 0.648 | 2.10 |
| 2048 | 7 | 7 | 0.329 | 0.647 | 1.97 |
| 2048 | 8 | 8 | 0.347 | 1.016 | 2.93 |
| 2048 | 9 | 9 | 0.358 | 0.815 | 2.28 |
| 2048 | 10 | 10 | 0.386 | 0.753 | 1.95 |
| 2048 | 11 | 11 | 0.405 | 0.922 | 2.28 |
| 2048 | 12 | 12 | 0.429 | 0.916 | 2.14 |
| 2048 | 13 | 13 | 0.445 | 1.017 | 2.29 |
| 2048 | 14 | 14 | 0.479 | 0.959 | 2.00 |
| 2048 | 15 | 15 | 0.490 | 1.221 | 2.49 |
| 2048 | 16 | 16 | 0.514 | 1.226 | 2.39 |
| 4096 | 2 | 2 | 0.468 | 0.641 | 1.37 |
| 4096 | 3 | 3 | 0.509 | 0.784 | 1.54 |
| 4096 | 4 | 4 | 0.533 | 0.888 | 1.67 |
| 4096 | 5 | 5 | 0.563 | 0.989 | 1.76 |
| 4096 | 6 | 6 | 0.594 | 1.009 | 1.70 |
| 4096 | 7 | 7 | 0.624 | 1.458 | 2.34 |
| 4096 | 8 | 8 | 0.663 | 1.652 | 2.49 |
| 4096 | 9 | 9 | 0.705 | 1.487 | 2.11 |
| 4096 | 10 | 10 | 0.748 | 1.373 | 1.84 |
| 4096 | 11 | 11 | 0.781 | 1.703 | 2.18 |
| 4096 | 12 | 12 | 0.834 | 1.706 | 2.05 |
| 4096 | 13 | 13 | 0.871 | 1.876 | 2.15 |
| 4096 | 14 | 14 | 0.946 | 1.768 | 1.87 |
| 4096 | 15 | 15 | 0.965 | 4.810 | 4.98 |
| 4096 | 16 | 16 | 1.003 | 2.319 | 2.31 |

#### Graph — Scenario 1 backward — uniform P

*(Screenshot: paste below. Optional reference PNG: `confluence_assets/scenario1_bwd.png` from `python3 gen_confluence_assets.py`.)*

---




## Scenario 2 — sq = 1, skv ≤ 16 (cross-attn / packed KV)

**Script:** `scenario_2.sh` · **CK fwd:** group, sq=1 · **JAX fwd:** Q `(B,1,H,D)`, KV THD packed.

### Forward

#### Table — forward (P (max KV length) sweep)

| batch | s_q | s_kv | ck_pr_6764 (ms) | jax_unfused (ms) | jax/ck |
| --- | --- | --- | --- | --- | --- |
| 2048 | 1 | 2 | 0.095 | 0.067 | 0.71 |
| 2048 | 1 | 3 | 0.095 | 0.075 | 0.79 |
| 2048 | 1 | 4 | 0.095 | 0.128 | 1.35 |
| 2048 | 1 | 5 | 0.098 | 0.135 | 1.38 |
| 2048 | 1 | 6 | 0.095 | 0.149 | 1.57 |
| 2048 | 1 | 7 | 0.096 | 0.158 | 1.65 |
| 2048 | 1 | 8 | 0.097 | 0.171 | 1.76 |
| 2048 | 1 | 9 | 0.096 | 0.168 | 1.75 |
| 2048 | 1 | 10 | 0.097 | 0.176 | 1.81 |
| 2048 | 1 | 11 | 0.097 | 0.182 | 1.88 |
| 2048 | 1 | 12 | 0.098 | 0.196 | 2.00 |
| 2048 | 1 | 13 | 0.105 | 0.211 | 2.01 |
| 2048 | 1 | 14 | 0.105 | 0.220 | 2.10 |
| 2048 | 1 | 15 | 0.112 | 0.229 | 2.04 |
| 2048 | 1 | 16 | 0.121 | 0.479 | 3.96 |
| 4096 | 1 | 2 | 0.184 | 0.141 | 0.77 |
| 4096 | 1 | 3 | 0.186 | 0.143 | 0.77 |
| 4096 | 1 | 4 | 0.185 | 0.167 | 0.90 |
| 4096 | 1 | 5 | 0.187 | 0.200 | 1.07 |
| 4096 | 1 | 6 | 0.207 | 0.211 | 1.02 |
| 4096 | 1 | 7 | 0.211 | 0.228 | 1.08 |
| 4096 | 1 | 8 | 0.216 | 0.479 | 2.22 |
| 4096 | 1 | 9 | 0.217 | 0.260 | 1.20 |
| 4096 | 1 | 10 | 0.215 | 0.278 | 1.29 |
| 4096 | 1 | 11 | 0.223 | 0.287 | 1.29 |
| 4096 | 1 | 12 | 0.224 | 0.303 | 1.35 |
| 4096 | 1 | 13 | 0.221 | 0.321 | 1.45 |
| 4096 | 1 | 14 | 0.231 | 0.336 | 1.45 |
| 4096 | 1 | 15 | 0.230 | 0.355 | 1.54 |
| 4096 | 1 | 16 | 0.231 | 0.861 | 3.73 |

#### Graph — Scenario 2 forward — sq=1, CK vs JAX vs P

*(Screenshot: paste below. Optional reference PNG: `confluence_assets/scenario2_fwd.png` from `python3 gen_confluence_assets.py`.)*

---




### Backward

#### Table — backward (P (max KV length) sweep)

| batch | s_q | s_kv | ck_pr_6764 (ms) | jax_unfused (ms) | jax/ck |
| --- | --- | --- | --- | --- | --- |
| 2048 | 1 | 2 | 0.209 | 0.172 | 0.82 |
| 2048 | 1 | 3 | 0.221 | 0.252 | 1.14 |
| 2048 | 1 | 4 | 0.256 | 0.198 | 0.77 |
| 2048 | 1 | 5 | 0.259 | 0.227 | 0.88 |
| 2048 | 1 | 6 | 0.270 | 0.253 | 0.94 |
| 2048 | 1 | 7 | 0.287 | 0.263 | 0.92 |
| 2048 | 1 | 8 | 0.294 | 0.296 | 1.01 |
| 2048 | 1 | 9 | 0.301 | 0.310 | 1.03 |
| 2048 | 1 | 10 | 0.312 | 0.319 | 1.02 |
| 2048 | 1 | 11 | 0.325 | 0.347 | 1.07 |
| 2048 | 1 | 12 | 0.331 | 0.478 | 1.44 |
| 2048 | 1 | 13 | 0.335 | 0.387 | 1.16 |
| 2048 | 1 | 14 | 0.347 | 0.401 | 1.16 |
| 2048 | 1 | 15 | 0.356 | 0.439 | 1.23 |
| 2048 | 1 | 16 | 0.362 | 0.450 | 1.24 |
| 4096 | 1 | 2 | 0.454 | 0.225 | 0.50 |
| 4096 | 1 | 3 | 0.477 | 0.240 | 0.50 |
| 4096 | 1 | 4 | 0.496 | 0.292 | 0.59 |
| 4096 | 1 | 5 | 0.513 | 0.322 | 0.63 |
| 4096 | 1 | 6 | 0.537 | 0.363 | 0.68 |
| 4096 | 1 | 7 | 0.556 | 0.401 | 0.72 |
| 4096 | 1 | 8 | 0.566 | 0.446 | 0.79 |
| 4096 | 1 | 9 | 0.583 | 0.474 | 0.81 |
| 4096 | 1 | 10 | 0.603 | 0.511 | 0.85 |
| 4096 | 1 | 11 | 0.620 | 0.555 | 0.90 |
| 4096 | 1 | 12 | 0.640 | 1.110 | 1.73 |
| 4096 | 1 | 13 | 0.649 | 1.088 | 1.68 |
| 4096 | 1 | 14 | 0.674 | 0.679 | 1.01 |
| 4096 | 1 | 15 | 0.689 | 0.728 | 1.06 |
| 4096 | 1 | 16 | 0.698 | 0.775 | 1.11 |

#### Graph — Scenario 2 backward — s_q=1, s_kv=P

*(Screenshot: paste below. Optional reference PNG: `confluence_assets/scenario2_bwd.png` from `python3 gen_confluence_assets.py`.)*

---




## Scenarios 3 & 4 — fixed self-attention (sq = skv, 2…17)

**Script:** `scenario_3_4.sh` · Customer **#3** = s=16, **#4** = s=17 rows in tables below.

### Forward

#### Table — forward (seq len (sq = skv) sweep)

| batch | s_q | s_kv | ck_pr_6764 (ms) | jax_unfused (ms) | jax/ck |
| --- | --- | --- | --- | --- | --- |
| 2048 | 2 | 2 | 0.093 | 0.260 | 2.80 |
| 2048 | 3 | 3 | 0.098 | 0.280 | 2.86 |
| 2048 | 4 | 4 | 0.099 | 0.316 | 3.19 |
| 2048 | 5 | 5 | 0.104 | 0.348 | 3.35 |
| 2048 | 6 | 6 | 0.118 | 0.381 | 3.23 |
| 2048 | 7 | 7 | 0.129 | 0.412 | 3.19 |
| 2048 | 8 | 8 | 0.132 | 0.441 | 3.34 |
| 2048 | 9 | 9 | 0.137 | 0.518 | 3.78 |
| 2048 | 10 | 10 | 0.156 | 0.540 | 3.46 |
| 2048 | 11 | 11 | 0.158 | 0.583 | 3.69 |
| 2048 | 12 | 12 | 0.170 | 0.603 | 3.55 |
| 2048 | 13 | 13 | 0.179 | 0.651 | 3.64 |
| 2048 | 14 | 14 | 0.194 | 0.681 | 3.51 |
| 2048 | 15 | 15 | 0.200 | 0.716 | 3.58 |
| 2048 | 16 | 16 | 0.206 | 0.717 | 3.48 |
| 2048 | 17 | 17 | 0.268 | 0.827 | 3.09 |
| 4096 | 2 | 2 | 0.180 | 0.424 | 2.36 |
| 4096 | 3 | 3 | 0.207 | 0.490 | 2.37 |
| 4096 | 4 | 4 | 0.217 | 0.541 | 2.49 |
| 4096 | 5 | 5 | 0.230 | 0.652 | 2.83 |
| 4096 | 6 | 6 | 0.243 | 0.694 | 2.86 |
| 4096 | 7 | 7 | 0.256 | 0.749 | 2.93 |
| 4096 | 8 | 8 | 0.268 | 0.790 | 2.95 |
| 4096 | 9 | 9 | 0.280 | 0.943 | 3.37 |
| 4096 | 10 | 10 | 0.301 | 0.973 | 3.23 |
| 4096 | 11 | 11 | 0.314 | 1.041 | 3.32 |
| 4096 | 12 | 12 | 0.337 | 1.085 | 3.22 |
| 4096 | 13 | 13 | 0.353 | 1.193 | 3.38 |
| 4096 | 14 | 14 | 0.384 | 1.244 | 3.24 |
| 4096 | 15 | 15 | 0.398 | 1.323 | 3.32 |
| 4096 | 16 | 16 | 0.406 | 1.296 | 3.19 |
| 4096 | 17 | 17 | 0.519 | 1.622 | 3.13 |

#### Graph — Scenario 3+4 forward — fixed self-attn vs seq len

*(Screenshot: paste below. Optional reference PNG: `confluence_assets/scenario3_4_fwd.png` from `python3 gen_confluence_assets.py`.)*

---




### Backward

#### Table — backward (seq len (sq = skv) sweep)

| batch | s_q | s_kv | ck_pr_6764 (ms) | jax_unfused (ms) | jax/ck |
| --- | --- | --- | --- | --- | --- |
| 2048 | 2 | 2 | 0.215 | 0.382 | 1.78 |
| 2048 | 3 | 3 | 0.255 | 0.449 | 1.76 |
| 2048 | 4 | 4 | 0.275 | 0.501 | 1.82 |
| 2048 | 5 | 5 | 0.290 | 0.535 | 1.84 |
| 2048 | 6 | 6 | 0.309 | 0.563 | 1.82 |
| 2048 | 7 | 7 | 0.325 | 0.636 | 1.96 |
| 2048 | 8 | 8 | 0.345 | 0.869 | 2.52 |
| 2048 | 9 | 9 | 0.365 | 0.815 | 2.23 |
| 2048 | 10 | 10 | 0.385 | 0.762 | 1.98 |
| 2048 | 11 | 11 | 0.405 | 0.918 | 2.27 |
| 2048 | 12 | 12 | 0.431 | 0.919 | 2.13 |
| 2048 | 13 | 13 | 0.446 | 1.011 | 2.27 |
| 2048 | 14 | 14 | 0.479 | 0.950 | 1.98 |
| 2048 | 15 | 15 | 0.495 | 1.216 | 2.46 |
| 2048 | 16 | 16 | 0.517 | 1.210 | 2.34 |
| 2048 | 17 | 17 | 2.183 | 1.273 | 0.58 |
| 4096 | 2 | 2 | 0.465 | 0.637 | 1.37 |
| 4096 | 3 | 3 | 0.508 | 0.766 | 1.51 |
| 4096 | 4 | 4 | 0.533 | 0.868 | 1.63 |
| 4096 | 5 | 5 | 0.562 | 0.976 | 1.74 |
| 4096 | 6 | 6 | 0.592 | 1.005 | 1.70 |
| 4096 | 7 | 7 | 0.626 | 1.163 | 1.86 |
| 4096 | 8 | 8 | 0.659 | 1.625 | 2.47 |
| 4096 | 9 | 9 | 0.692 | 1.470 | 2.12 |
| 4096 | 10 | 10 | 0.748 | 1.374 | 1.84 |
| 4096 | 11 | 11 | 0.784 | 1.662 | 2.12 |
| 4096 | 12 | 12 | 0.836 | 1.678 | 2.01 |
| 4096 | 13 | 13 | 0.867 | 1.878 | 2.17 |
| 4096 | 14 | 14 | 0.948 | 1.727 | 1.82 |
| 4096 | 15 | 15 | 0.967 | 2.066 | 2.14 |
| 4096 | 16 | 16 | 1.001 | 2.273 | 2.27 |
| 4096 | 17 | 17 | 4.361 | 2.397 | 0.55 |

#### Graph — Scenario 3+4 backward — fixed self-attn vs seq len

*(Screenshot: paste below. Optional reference PNG: `confluence_assets/scenario3_4_bwd.png` from `python3 gen_confluence_assets.py`.)*

---




---

## Summary (CK vs JAX)

| Scenario | Direction | Point | jax/ck (B=2048) | Notes |
|----------|-----------|-------|-----------------|-------|
| 1 | fwd | P=16 | 4.20 | CK faster |
| 1 | bwd | P=16 | 2.39 | CK faster |
| 2 | fwd | P=16 (sq=1) | 3.96 | CK faster |
| 2 | bwd | P=16 | 1.24 | CK faster |
| 3+4 | fwd | s=16 | 3.48 | CK faster |
| 3+4 | bwd | s=16 | 2.34 | CK faster |

### CK bwd outlier (scenario 3+4, seq = 17)

| batch | s_q | s_kv | ck_pr_6764 (ms) | jax_unfused (ms) | jax/ck |
|-------|-----|------|-----------------|------------------|--------|
| 2048 | 17 | 17 | **2.183** | 1.273 | 0.58 |
| 4096 | 17 | 17 | **4.361** | 2.397 | 0.55 |

At seq=17 backward, JAX is faster than CK — opposite of other rows. Worth CK/asm follow-up (4×4 vs 16×16 tile paths).

---

## Reproduce

```bash
cd op_tests/cpp/mha && bash build_mha.sh
cd small_attn_benchmark
./run_all.sh
python3 run_jax_benchmark.py all
python3 gen_confluence_assets.py   # PNGs for Confluence paste
python3 build_confluence_md.py     # refresh this page
```
