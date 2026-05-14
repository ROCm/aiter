# Cross-attention: CK vs JAX unfused

**Layout:** CK **BHSD** (`benchmark_mha_fwd.cpp`, `benchmark_mha_bwd.cpp`: `-iperm=1 -operm=1`). JAX unfused **BSHD** (`jax_unfused_attention.py`).

**Files & run:**

- `run_mha_performance_comparison_cross_attn.sh` — CK + optional JAX, then `write_mha_performance_comparison_md.py --kind cross_attn`. From repo root: `cd op_tests/cpp/mha && ./run_mha_performance_comparison_cross_attn.sh`.
- `run_jax_unfused_cross_attn_benchmark.py` — JAX timings only (env `JAX_UNFUSED_*` set by the shell).

`jax/ck` = jax_time / ck_time (>1 ⇒ CK faster).

Each **Configuration *n*** is one batch size **B**; numbering follows the order of batch blocks in the timing CSV (same as `CROSS_ATTN_BATCHES` in `run_mha_performance_comparison_cross_attn.sh` when that script produced the CSV).

## Configuration 1 — B=2048

### Forward

**CK:** `benchmark_mha_fwd`, bf16, BHSD, **B** as in section title, `s_q=1`, `s_kv=P`, `h=32`, `d=128`, `mask=0`, `fwd_v3=0`; warmup/repeat from `run_mha_performance_comparison_cross_attn.sh`.

**JAX:** `run_jax_unfused_cross_attn_benchmark.py` + `jax_unfused_attention.py`, BSHD, same **B** / `s_q` / `s_kv` / `h` / `d`; non-causal; softmax `1/sqrt(d)` when `JAX_UNFUSED_SM_SCALE=ck`.

| s_kv (P) | ck fwd (ms) | jax unfused fwd (ms) | jax/ck fwd | group fwd |
|---------:|--------------:|---------------------:|-----------:|:---------:|
| 2 | 0.089 | 0.064 | 0.72x | no |
| 3 | 0.089 | 0.088 | 0.99x | no |
| 4 | 0.090 | 0.097 | 1.08x | no |
| 5 | 0.089 | 0.107 | 1.20x | no |
| 6 | 0.090 | 0.121 | 1.34x | no |
| 7 | 0.090 | 0.147 | 1.63x | no |
| 8 | 0.100 | 0.165 | 1.65x | no |
| 9 | 0.108 | 0.171 | 1.58x | no |
| 10 | 0.105 | 0.175 | 1.67x | no |
| 11 | 0.108 | 0.186 | 1.72x | no |
| 12 | 0.111 | 0.196 | 1.77x | no |
| 13 | 0.113 | 0.206 | 1.82x | no |
| 14 | 0.118 | 0.216 | 1.83x | no |
| 15 | 0.119 | 0.222 | 1.87x | no |
| 16 | 0.122 | 0.292 | 2.39x | no |

### Backward

**CK:** `benchmark_mha_bwd`, bf16, BHSD, **B** as in section title, `s_q=1`, `s_kv=P`, `h=32`, `d=128`, `mask=0`, `bwd_v3=0`, `mode=0`; warmup/repeat from same shell.

**JAX:** same script as forward; `vjp` on unfused forward, then `jit(pullback)` timed for `do`.

| s_kv (P) | ck bwd (ms) | jax unfused bwd (ms) | jax/ck bwd | group fwd |
|---------:|--------------:|---------------------:|-----------:|:---------:|
| 2 | 0.209 | 0.148 | 0.71x | no |
| 3 | 0.219 | 0.149 | 0.68x | no |
| 4 | 0.254 | 0.168 | 0.66x | no |
| 5 | 0.259 | 0.195 | 0.75x | no |
| 6 | 0.270 | 0.236 | 0.87x | no |
| 7 | 0.282 | 0.246 | 0.87x | no |
| 8 | 0.291 | 0.278 | 0.96x | no |
| 9 | 0.297 | 0.290 | 0.98x | no |
| 10 | 0.305 | 0.300 | 0.98x | no |
| 11 | 0.318 | 0.315 | 0.99x | no |
| 12 | 0.329 | 0.335 | 1.02x | no |
| 13 | 0.335 | 0.352 | 1.05x | no |
| 14 | 0.341 | 0.383 | 1.12x | no |
| 15 | 0.351 | 0.399 | 1.14x | no |
| 16 | 0.360 | 0.418 | 1.16x | no |

## Configuration 2 — B=4096

### Forward

**CK:** `benchmark_mha_fwd`, bf16, BHSD, **B** as in section title, `s_q=1`, `s_kv=P`, `h=32`, `d=128`, `mask=0`, `fwd_v3=0`; warmup/repeat from `run_mha_performance_comparison_cross_attn.sh`.

**JAX:** `run_jax_unfused_cross_attn_benchmark.py` + `jax_unfused_attention.py`, BSHD, same **B** / `s_q` / `s_kv` / `h` / `d`; non-causal; softmax `1/sqrt(d)` when `JAX_UNFUSED_SM_SCALE=ck`.

| s_kv (P) | ck fwd (ms) | jax unfused fwd (ms) | jax/ck fwd | group fwd |
|---------:|--------------:|---------------------:|-----------:|:---------:|
| 2 | 0.169 | 0.124 | 0.73x | no |
| 3 | 0.169 | 0.144 | 0.85x | no |
| 4 | 0.189 | 0.169 | 0.89x | no |
| 5 | 0.196 | 0.204 | 1.04x | no |
| 6 | 0.197 | 0.221 | 1.12x | no |
| 7 | 0.201 | 0.241 | 1.20x | no |
| 8 | 0.206 | 0.239 | 1.16x | no |
| 9 | 0.213 | 0.255 | 1.20x | no |
| 10 | 0.216 | 0.270 | 1.25x | no |
| 11 | 0.223 | 0.288 | 1.29x | no |
| 12 | 0.230 | 0.308 | 1.34x | no |
| 13 | 0.232 | 0.325 | 1.40x | no |
| 14 | 0.238 | 0.541 | 2.27x | no |
| 15 | 0.245 | 0.694 | 2.83x | no |
| 16 | 0.254 | 0.523 | 2.06x | no |

### Backward

**CK:** `benchmark_mha_bwd`, bf16, BHSD, **B** as in section title, `s_q=1`, `s_kv=P`, `h=32`, `d=128`, `mask=0`, `bwd_v3=0`, `mode=0`; warmup/repeat from same shell.

**JAX:** same script as forward; `vjp` on unfused forward, then `jit(pullback)` timed for `do`.

| s_kv (P) | ck bwd (ms) | jax unfused bwd (ms) | jax/ck bwd | group fwd |
|---------:|--------------:|---------------------:|-----------:|:---------:|
| 2 | 0.457 | 0.220 | 0.48x | no |
| 3 | 0.476 | 0.240 | 0.50x | no |
| 4 | 0.498 | 0.290 | 0.58x | no |
| 5 | 0.511 | 0.313 | 0.61x | no |
| 6 | 0.531 | 0.353 | 0.66x | no |
| 7 | 0.551 | 0.388 | 0.70x | no |
| 8 | 0.573 | 0.447 | 0.78x | no |
| 9 | 0.576 | 0.470 | 0.82x | no |
| 10 | 0.595 | 0.514 | 0.86x | no |
| 11 | 0.613 | 0.547 | 0.89x | no |
| 12 | 0.639 | 0.589 | 0.92x | no |
| 13 | 0.653 | 0.628 | 0.96x | no |
| 14 | 0.670 | 0.660 | 0.99x | no |
| 15 | 0.684 | 0.723 | 1.06x | no |
| 16 | 0.704 | 0.768 | 1.09x | no |
