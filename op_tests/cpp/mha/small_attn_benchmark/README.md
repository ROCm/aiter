# Small-sequence attention benchmarks (ck_pr_6764)

CK forward/backward via `benchmark_mha_fwd` / `benchmark_mha_bwd` in the parent `mha/` folder (`fwd_v3=0`, `bwd_v3=0`, seed **6764** length lists for scenarios 1–2).

## Scenarios

| Script | Shape | CK forward | CK backward |
|--------|--------|------------|-------------|
| `scenario_1.sh` | `sq, skv ≤ 16`, packed varlen | group `mode=1`, comma `-s` / `-s_k` | batch `mode=0`, uniform `s_q=s_kv=P` |
| `scenario_2.sh` | `sq=1`, `skv ≤ 16`, packed KV varlen | group `mode=1` | batch `mode=0`, `s_q=1`, `s_kv=P` |
| `scenario_3_4.sh` | fixed self-attn `sq=skv` ∈ [`SEQ_MIN`, `SEQ_MAX`] | batch `mode=0` | batch `mode=0` |

Default sweeps:

- Scenarios **1–2:** `PAD_MIN`…`PAD_MAX` = **2–16** (varlen upper bound **P** per CSV row).
- **3+4:** `SEQ_MIN`…`SEQ_MAX` = **2–17** (fixed length per row).

**Length RNG (scenarios 1–2, forward):** `gen_small_attn_lengths.py` — discrete uniform in `{2,…,P}` per batch row, seed **6764**.

> `benchmark_mha_bwd` does not accept comma-separated length lists; backward for scenarios 1–2 uses uniform batch shapes.

## Results layout

```
results/
  scenario1/    fwd.csv  bwd.csv   (all BATCHES as rows)
  scenario2/    fwd.csv  bwd.csv
  scenario3_4/  fwd.csv  bwd.csv   (seq 2..17 × each batch)
```

**Columns:** `batch`, `s_q`, `s_kv`, `ck_pr_6764(ms)`

## Quick start

```bash
cd op_tests/cpp/mha && bash build_mha.sh
cd small_attn_benchmark
./scenario_3_4.sh
./run_all.sh
```

**Env:** `BATCHES`, `PAD_MIN`, `PAD_MAX`, `SEQ_MIN`, `SEQ_MAX`, `WARMUP`, `REPEAT`, `RESULTS_DIR`.

CK binaries stay in `../` (parent `mha/`).
