# Plan: Add run_config / compare Benchmark to Tuners

## Context
The tuner scripts (GEMM/MOE) can find optimal kernel configs, but there's no way to:
1. Measure baseline performance before tuning (using the production operator path)
2. Verify that tuned configs actually work correctly at runtime after tuning
3. Compare before/after performance to quantify the improvement ratio

The existing `run_config()` method in `gemm_a8w8_bpreshuffle_tune.py:448` was a partial/incomplete implementation. We built this into a general framework in `base_tuner.py` that all tuners can use.

> **Full implementation documentation:** see [`docs/run_config_benchmark.md`](run_config_benchmark.md)

## Approach

### 1. Add `run_config` / `compare` infrastructure to `base_tuner.py`

**File: `aiter/utility/base_tuner.py`**

Add to `TunerCommon`:
- `--run_config` CLI flag — when set, runs benchmark only and exits (no tuning)
- `--compare` CLI flag — runs pre-tune benchmark, performs tuning, runs post-tune benchmark, then prints comparison table with speedup ratios
- Default method `def run_config(self, args)` — each subclass overrides this to call the actual production operator for each shape, returning a list of `{"shape": str, "us": float, "status": str}` dicts
- `config_env_name` in `ARG_DEFAULTS` — each subclass declares its config env var name (e.g., `"AITER_CONFIG_GEMM_A8W8"`) for post-tune config switching
- `_clear_op_caches(self)` — each subclass overrides to clear its own operator config caches (e.g., `@lru_cache` or `file_cache`)
- `_set_config_env_for_run_config(args)` — sets config env var to the `-o` output file, calls `_clear_op_caches()`, clears JIT/module caches, enables `AITER_REBUILD = 2`
- `_restore_config_env(env_name, old_val)` — restores config env var and `AITER_REBUILD` to original values
- `_print_benchmark_results(label, results)` — prints a formatted table of benchmark timings
- `_print_comparison(pre_results, post_results)` — prints a comparison table with speedup ratios

### 2. Modify `run()` flow in `base_tuner.py`

```
run(args):
    1. pre_process(args)

    2. if --run_config:
           results = run_config(args)              # Benchmark only (uses current/default config)
           print Benchmark table
           return (no tuning)

    3. if --compare:
           pre_tune_results = run_config(args)     # Pre-tune benchmark
           print Pre-tune table

    4. ... normal tuning loop (batched) ...

    5. finally:
           tune_summary(tuning_status)             # May sys.exit(1)
           if --compare:
               _set_config_env_for_run_config(args)    # Switch to tuned output file
               post_tune_results = run_config(args)    # Post-tune benchmark
               print Post-tune table
               print Comparison table (pre vs post)
               _restore_config_env(env_name, old_val)  # Restore original config
```

The post-tune benchmark runs inside a `finally` block with special `SystemExit` handling, ensuring it always executes even if `tune_summary()` calls `sys.exit(1)` due to failed/untuned shapes.

### 3. Implement `run_config` for each tuner subclass

Each subclass overrides `run_config()` to:
- Read shapes from `self.untunedf` (the untuned CSV after `pre_process` filtering)
- For each shape, generate test data, call the production operator (from `aiter/ops/`), benchmark with `run_perftest`
- Verify correctness against a reference implementation using `checkAllclose`
- Return timing results as a list of dicts: `[{"shape": str, "us": float, "status": "ok"/"mismatch"/"error:..."}]`

**Files modified:**
- `csrc/ck_gemm_a8w8/gemm_a8w8_tune.py` — `run_config()` calling `gemm_a8w8`
- `csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py` — `run_config()` calling `gemm_a8w8_bpreshuffle` / `gemm_a8w8_ASM`
- `csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py` — `run_config()` calling `gemm_a8w8_blockscale`
- `csrc/ck_gemm_a8w8_blockscale_bpreshuffle/gemm_a8w8_blockscale_bpreshuffle_tune.py` — `run_config()` calling `gemm_a8w8_blockscale_bpreshuffle`
- `csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_tune.py` — `run_config()` calling `gemm_a4w4`
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py` — `run_config()` calling `fused_moe`
- `csrc/ck_batched_gemm_a8w8/batched_gemm_a8w8_tune.py` — `run_config()` calling `batched_gemm_a8w8`
- `csrc/ck_batched_gemm_bf16/batched_gemm_bf16_tune.py` — `run_config()` calling `batched_gemm_bf16`
- `gradlib/gradlib/GemmTuner.py` — `run_config()` calling `gemm_a16w16`

### 4. Post-tune config switching

The post-tune benchmark must use the **newly tuned config file** (the `-o` output), not the default config that was used during pre-tune. This requires each tuner to declare `config_env_name` in `ARG_DEFAULTS` and override `_clear_op_caches()`:

| Tuner | `config_env_name` | Cache to clear |
|-------|-------------------|----------------|
| A8W8 BPreShuffle | `AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE` | `get_GEMM_config_with_quant_type.file_cache` |
| A8W8 | `AITER_CONFIG_GEMM_A8W8` | `get_GEMM_config_with_quant_type.file_cache` |
| A8W8 BlockScale | `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE` | `get_GEMM_config_with_quant_type.file_cache` |
| A8W8 BlockScale BPreShuffle | `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE` | `get_GEMM_config_with_quant_type.file_cache` |
| A4W4 BlockScale | `AITER_CONFIG_GEMM_A4W4` | `get_GEMM_config.file_cache` |
| MOE 2-Stage | `AITER_CONFIG_FMOE` | `cfg_2stages`, `_flydsl_fallback_cache` |
| Batched A8W8 | `AITER_CONFIG_A8W8_BATCHED_GEMM` | `get_CKBatchedGEMM_config.cache_clear()`, `ck_batched_gemm_dict` |
| Batched BF16 | `AITER_CONFIG_BF16_BATCHED_GEMM` | `get_CKBatchedGEMM_config.cache_clear()`, `ck_batched_gemm_dict` |
| BF16 (GemmTuner) | `AITER_CONFIG_GEMM_BF16` | `get_GEMM_A16W16_config_.cache_clear()`, `get_GEMM_A16W16_config.cache_clear()` |

### 5. Comparison table output format

```
============= Tune Performance Comparison =============
Shape                                    |  Pre-tune(us) |  Post-tune(us) |  Speedup |   Status
-----------------------------------------------------------------------------------------------
(64, 1536, 5120, dtypes.fp8)             |         42.35 |          38.12 |    1.11x |       OK
(128, 1536, 5120, dtypes.fp8)            |         58.21 |          45.67 |    1.27x |       OK
```

## Key Files Modified
1. `aiter/utility/base_tuner.py` — `--run_config` + `--compare` flags, default `run_config()`, `_clear_op_caches()`, `_set_config_env_for_run_config()`, `_restore_config_env()`, comparison logic, modified `run()`
2. `csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py` — `run_config()`, `_clear_op_caches()`, `config_env_name`
3. `csrc/ck_gemm_a8w8/gemm_a8w8_tune.py` — `run_config()`, `_clear_op_caches()`, `config_env_name`
4. `csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py` — `run_config()`, `_clear_op_caches()`, `config_env_name`
5. `csrc/ck_gemm_a8w8_blockscale_bpreshuffle/gemm_a8w8_blockscale_bpreshuffle_tune.py` — `run_config()`, `_clear_op_caches()`, `config_env_name`
6. `csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_tune.py` — `run_config()`, `_clear_op_caches()`, `config_env_name`
7. `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py` — `run_config()`, `_clear_op_caches()`, `config_env_name`
8. `csrc/ck_batched_gemm_a8w8/batched_gemm_a8w8_tune.py` — `run_config()`, `_clear_op_caches()`, `config_env_name`
9. `csrc/ck_batched_gemm_bf16/batched_gemm_bf16_tune.py` — `run_config()`, `_clear_op_caches()`, `config_env_name`
10. `gradlib/gradlib/GemmTuner.py` — `run_config()`, `_clear_op_caches()`, `config_env_name`

## Key Utilities Reused
- `aiter.test_common.run_perftest` — for benchmarking timing
- `aiter.test_common.checkAllclose` — for correctness verification
- Each tuner's existing `generate_data()` functions — for test data generation
- Production ops from `aiter/ops/gemm_op_a8w8.py`, `aiter/tuned_gemm.py`, etc. — the actual dispatch functions

## Verification

All verification must be run on actual GPU hardware (AMD MI series).

- `--run_config`: Run benchmark only and exit, e.g.:
  ```bash
  python3 csrc/ck_gemm_a8w8/gemm_a8w8_tune.py --run_config -i aiter/configs/a8w8_untuned_gemm.csv -o aiter/configs/a8w8_tuned_gemm.csv
  python3 gradlib/gradlib/GemmTuner.py --run_config -i aiter/configs/bf16_untuned_gemm.csv -o aiter/configs/bf16_tuned_gemm.csv
  ```
  Expected: prints benchmark table for all shapes and exits (no tuning).

- `--compare`: Pre-tune benchmark, tune, post-tune benchmark, comparison table:
  ```bash
  python3 csrc/ck_gemm_a8w8/gemm_a8w8_tune.py --compare -i aiter/configs/a8w8_untuned_gemm.csv -o aiter/configs/a8w8_tuned_gemm.csv
  python3 gradlib/gradlib/GemmTuner.py --compare -i aiter/configs/bf16_untuned_gemm.csv -o aiter/configs/bf16_tuned_gemm.csv
  ```
  Expected: prints pre-tune table, runs tuning, prints post-tune table, prints comparison table with speedup ratios.

- Check that all shapes report `ok` status (no `mismatch` or `error`).
- Check that post-tune speedup >= 1.0x for tuned shapes.
