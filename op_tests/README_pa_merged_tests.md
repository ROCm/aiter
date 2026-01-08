# Paged Attention HIP vs ASM comparison tests

This repo contains multiple PA (Paged Attention) implementations. The file
`op_tests/test_pa_merged.py` is a **focused** test module whose only goal is to:

- **Compare correctness**: HIP (`paged_attention_v1_core`) vs ASM (`pa_fwd_asm`)
- **Use ASM-compatible KV-cache layouts for BOTH paths**
  - HIP is exercised through the **5D-cache** codepath that can consume these layouts
  - ASM is exercised directly via `pa_fwd_asm` on the same layouts
- Optionally **measure performance** using AITER’s standard `@perftest()` harness

The tests are **opt-in** (they skip by default) to avoid running long GPU workloads in CI.

## What’s inside `op_tests/test_pa_merged.py`

- **Repro test (`-k repro`)**
  - Uses a fixed set of shapes (from an EngineCore log)
  - Compares `paged_attention_v1_core` vs `pa_fwd_asm`
  - Optional perf: runs `@perftest(num_iters=200)` for HIP and ASM and prints a winner line

- **Stress test (`-k stress`)**
  - Randomly generates ASM-compatible KV-cache inputs
  - Compares `paged_attention_v1_core` vs `pa_fwd_asm`

## Environment variables (“defines”) you may set

### Required (choose one, otherwise tests skip)

- **`AITER_RUN_REPRO_SHAPES=1`**
  - Enables the repro test (fixed shapes)
- **`AITER_RUN_STRESS=1`**
  - Enables the stress test (random trials)

### Optional knobs

- **`AITER_REPRO_PERF=1`** (default in the test is `"1"`)
  - If enabled, the repro test runs perf for HIP and ASM and prints:
    - `HIP(paged_attention_v1_core) avg_us/iter=...`
    - `ASM(pa_fwd_asm) avg_us/iter=...`
    - `winner=...`
  - Set **`AITER_REPRO_PERF=0`** to run correctness only.

- **`AITER_STRESS_TRIALS=N`** (default: `25`)
  - Number of randomized stress trials.

- **`AITER_LOG_MORE=1`**
  - Makes AITER logging more verbose and enables profiler table printing inside `@perftest()`.
  - Under pytest you often want `-s` (or `-o log_cli=true`) to see logs.

## How to run

Run from repo root.

### Repro (fixed shapes) — correctness only

```bash
AITER_RUN_REPRO_SHAPES=1 AITER_REPRO_PERF=0 pytest -q op_tests/test_pa_merged.py -k repro
```

### Repro (fixed shapes) — correctness + perf (prints who is faster)

```bash
AITER_RUN_REPRO_SHAPES=1 AITER_REPRO_PERF=1 pytest -q op_tests/test_pa_merged.py -k repro -s
```

### Stress (random ASM-layout inputs) — 1 quick trial

```bash
AITER_RUN_STRESS=1 AITER_STRESS_TRIALS=1 pytest -q op_tests/test_pa_merged.py -k stress
```

### Stress (random ASM-layout inputs) — more trials

```bash
AITER_RUN_STRESS=1 AITER_STRESS_TRIALS=25 pytest -q op_tests/test_pa_merged.py -k stress
```

## Notes

- If you see `ss` in pytest output, that means **both tests were skipped** because you didn’t set
  `AITER_RUN_REPRO_SHAPES=1` and/or `AITER_RUN_STRESS=1`.
- `@perftest()` uses `torch.profiler` to attribute time to kernels and reports an average at the end.
- The ASM path (`pa_fwd_asm`) has tighter supported-config constraints (e.g., typically `bf16`, `head_size=128`,
  `block_size=16`); this module intentionally sticks to those constraints.

