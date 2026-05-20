# Small-sequence attention benchmarks (ck_pr_6764 + JAX unfused)

## Workflow

**1. CK only** (build once from parent `mha/`):

```bash
cd op_tests/cpp/mha && bash build_mha.sh
cd small_attn_benchmark
./run_all.sh          # or ./scenario_1.sh etc.
```

Writes `results/scenario{N}/fwd.csv` and `bwd.csv` with column `ck_pr_6764(ms)`.

**2. JAX later** (adds one column to those CSVs):

```bash
python3 run_jax_benchmark.py all    # or 1 | 2 | 3_4
```

Adds `jax_unfused(ms)` to each existing row. Requires `jax` / `jaxlib`.

## Scenarios

| Script | Shape | CK forward | CK backward | JAX forward | JAX backward |
|--------|--------|------------|-------------|-------------|--------------|
| `scenario_1.sh` | varlen ≤ P | group packed `-s` lists | batch uniform P | THD + `cu_seqlens` | dense `(B,P,…)` |
| `scenario_2.sh` | sq=1, kv ≤ P | group packed KV | batch `s_q=1`, `s_kv=P` | THD KV + `cu_seqlens_k` | dense `(B,1,P,…)` |
| `scenario_3_4.sh` | fixed self-attn s∈[2,17] | batch | batch | dense `(B,s,…)` | dense |

Lengths for scenarios 1–2: `gen_small_attn_lengths.py`, seed **6764**.

## CSV columns

`batch`, `s_q`, `s_kv`, `ck_pr_6764(ms)`, `jax_unfused(ms)`

JAX reads the CK CSV rows (no duplicate sweep config). Env: `WARMUP`, `REPEAT`, `RESULTS_DIR`, `NHEADS`, `HDIM`.
