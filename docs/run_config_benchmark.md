# Production Operator Benchmark (run_config / compare)

## Overview

Two flags enable production operator benchmarking:

- **`--run_config`**: Run the production operator benchmark for all shapes and exit immediately (no tuning). Useful for quick performance checks or verifying existing configs.
- **`--compare`**: Run benchmark before tuning, perform tuning, then run benchmark again and show a comparison table with speedup ratios.

## Usage

### `--run_config` (benchmark only, no tuning)

```bash
# Just benchmark current performance and exit
python3 csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py \
    --run_config \
    -i aiter/configs/a8w8_bpreshuffle_untuned_gemm.csv \
    -o aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv
```

### `--compare` (pre-tune + tune + post-tune comparison)

```bash
# Benchmark before tuning, tune, benchmark after tuning, show comparison
python3 csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py \
    --compare \
    -i aiter/configs/a8w8_bpreshuffle_untuned_gemm.csv \
    -o aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv

# Also works with other flags
python3 csrc/ck_gemm_a8w8/gemm_a8w8_tune.py \
    --compare --splitK --verbose \
    -i aiter/configs/a8w8_untuned_gemm.csv \
    -o aiter/configs/a8w8_tuned_gemm.csv
```

If no shapes need tuning (all already tuned), `--compare` still runs the benchmark to verify existing configs.

## Output Format

### `--run_config` Output

```
============= Benchmark Results =============
Shape                                    |   Time(us) |   Status
------------------------------------------------------------------------
(64, 1536, 5120, dtypes.fp8)             |      42.35 |       ok
(128, 1536, 5120, dtypes.fp8)            |      58.21 |       ok
(256, 3072, 5120, dtypes.i8)             |     125.10 |       ok
```

### `--compare` Output

Pre-tune and post-tune benchmark tables, followed by a comparison table:

```
============= Pre-tune Benchmark Results =============
Shape                                    |   Time(us) |   Status
------------------------------------------------------------------------
(64, 1536, 5120, dtypes.fp8)             |      42.35 |       ok
...

============= Post-tune Benchmark Results =============
Shape                                    |   Time(us) |   Status
------------------------------------------------------------------------
(64, 1536, 5120, dtypes.fp8)             |      38.12 |       ok
...

============= Tune Performance Comparison =============
Shape                                    |  Pre-tune(us) |  Post-tune(us) |  Speedup |   Status
-----------------------------------------------------------------------------------------------
(64, 1536, 5120, dtypes.fp8)             |         42.35 |          38.12 |    1.11x |       OK
(128, 1536, 5120, dtypes.fp8)            |         58.21 |          45.67 |    1.27x |       OK
(256, 3072, 5120, dtypes.i8)             |        125.10 |          98.40 |    1.27x |       OK
```

- **Speedup** = Pre-tune(us) / Post-tune(us). Values > 1.0 indicate improvement.
- **Status**: `OK` if post-tune passes correctness check; `ERROR` otherwise. Pre-tune errors are acceptable (e.g., before tuning, some shapes may not have a valid kernel config).
- **Correctness**: Each shape's output is verified against a reference (torch) implementation using `checkAllclose` from `aiter/test_common.py`. Error ratio is compared against `--errRatio` (default 0.05).

## Architecture

### Base Infrastructure (`aiter/utility/base_tuner.py`)

The following components were added to `TunerCommon`:

| Component | Description |
|-----------|-------------|
| `--run_config` CLI flag | Run benchmark only, no tuning (exit immediately after) |
| `--compare` CLI flag | Run pre-tune benchmark, tune, post-tune benchmark, show comparison |
| `run_config(self, args)` | Default method returning empty list. Subclasses override to call production operators. |
| `_clear_op_caches(self)` | Clear operator-specific config caches. Each tuner overrides for its own caches. |
| `_set_config_env_for_run_config(args)` | Sets config env var to `-o` output file, calls `_clear_op_caches()`, enables `AITER_REBUILD`. |
| `_restore_config_env(env_name, old_val)` | Restores config env var and `AITER_REBUILD` to original values. |
| `_print_benchmark_results(label, results)` | Prints a formatted table of benchmark timings. |
| `_print_comparison(pre_results, post_results)` | Prints a comparison table with speedup ratios. |

### Execution Flow in `run()`

```
run(args):
    1. pre_process(args)

    2. if --run_config:
           results = run_config(args)                  # Benchmark only (uses current/default config)
           print Benchmark table
           return (no tuning)

    3. if --compare:
           pre_tune_results = run_config(args)         # Pre-tune benchmark
           print Pre-tune table

    4. ... normal tuning loop (batched) ...

    5. finally:
           tune_summary(tuning_status)                 # May sys.exit(1)
           if --compare:
               _set_config_env_for_run_config(args)    # Switch to tuned output file
               post_tune_results = run_config(args)    # Post-tune benchmark (uses new tuned config)
               print Post-tune table
               print Comparison table (pre vs post)
               _restore_config_env(env_name, old_val)  # Restore original config
```

The post-tune benchmark runs inside a `finally` block with special `SystemExit` handling, ensuring it always executes even if `tune_summary()` calls `sys.exit(1)` due to failed/untuned shapes.

### Post-tune Config Switching (`_set_config_env_for_run_config`)

The post-tune benchmark must use the **newly tuned config file** (the `-o` output), not the default config that was used during pre-tune. This requires:

1. **Set config env var**: Each tuner declares a `config_env_name` in `ARG_DEFAULTS` (e.g., `"AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE"`). The env var is set to the `-o` output file path.

2. **Clear operator-specific caches** via `self._clear_op_caches()`: Each tuner overrides this to clear its own config caches:
   - A8W8 tuners: `get_GEMM_config_with_quant_type.file_cache` (CSV data cache in `gemm_op_a8w8.py`)
   - A4W4 tuner: `get_GEMM_config.file_cache` (CSV data cache in `gemm_op_a4w4.py`)
   - MOE tuner: `cfg_2stages`, `_flydsl_fallback_cache`
   - Batched gemm tuners: `get_CKBatchedGEMM_config.cache_clear()`, `ck_batched_gemm_dict`

3. **Clear JIT/module caches** (done in base `_set_config_env_for_run_config`):
   - `AITER_CONFIGS.get_config_file` `@lru_cache` (resolves env var to file path)
   - `get_module` `@lru_cache` (loaded JIT modules)
   - `__mds` dict (loaded module objects)
   - `rebuilded_list` (tracks which modules have been rebuilt)

4. **Enable `AITER_REBUILD`**: Set `jit_core.AITER_REBUILD = 2` so that JIT-compiled operators (`.so` files) are rebuilt with the new config. Level 2 removes `.so` only but keeps the build cache for faster incremental rebuild (level 1 also clears cmake build dir, which can cause first-shape failures).

### `config_env_name` in Tuner ARG_DEFAULTS

Each tuner must declare its config env var name:

```python
class MyTuner(GemmCommonTuner):
    ARG_DEFAULTS = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": f"{MY_CONFIG_FILE}",
        "config_env_name": "AITER_CONFIG_MY_KERNEL",  # Required for post-tune switching
    }
```

| Tuner | `config_env_name` |
|-------|-------------------|
| A8W8 BPreShuffle | `AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE` |
| A8W8 | `AITER_CONFIG_GEMM_A8W8` |
| A8W8 BlockScale | `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE` |
| A8W8 BlockScale BPreShuffle | `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE` |
| A4W4 BlockScale | `AITER_CONFIG_GEMM_A4W4` |
| MOE 2-Stage | `AITER_CONFIG_FMOE` |
| Batched A8W8 | `AITER_CONFIG_A8W8_BATCHED_GEMM` |
| Batched BF16 | `AITER_CONFIG_BF16_BATCHED_GEMM` |

### Debugging

Set `AITER_LOG_TUNED_CONFIG=1` to see which kernel config is selected at runtime:

```bash
AITER_LOG_TUNED_CONFIG=1 python3 csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py \
    --run_config -i aiter/configs/a8w8_bpreshuffle_untuned_gemm.csv \
    -o aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv
```

This will print messages like:
```
[aiter] shape M:64, N:1536, K:5120 q_dtype_w:torch.int8, found padded_M: 64, N:1536, K:5120 is tuned, in aiter/configs/a8w8_bpreshuffle_tuned_gemm3193.csv! libtype is asm!
```

### `run_config()` Return Format

Each subclass returns a list of dicts:

```python
[
    {"shape": "(M, N, K, ...)", "us": 42.35, "status": "ok"},
    {"shape": "(M, N, K, ...)", "us": -1,    "status": "error:..."},
    {"shape": "(M, N, K, ...)", "us": 58.21, "status": "mismatch"},
]
```

- `"ok"`: Output passes correctness check (`err_ratio <= args.errRatio`)
- `"mismatch"`: Operator runs but output diverges from reference
- `"error:..."`: Exception occurred (with message)

## Supported Tuners

### 1. GEMM A8W8 BPreShuffle (`ck_gemm_a8w8_bpreshuffle`)

| Item | Detail |
|------|--------|
| **Class** | `GemmA8W8BpreShuffleTuner` |
| **File** | `csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py` |
| **Production Op** | `gemm_a8w8_bpreshuffle` (fp8) / `gemm_a8w8_ASM` (i8) |
| **Reference** | `run_torch()` |
| **Shape Keys** | M, N, K, q_dtype_w |
| **Special** | Dual path: uses ASM operator when `q_dtype_w == dtypes.i8`, CK operator otherwise |

```bash
python3 csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py \
    --run_config \
    -i aiter/configs/a8w8_bpreshuffle_untuned_gemm.csv \
    -o aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv
```

### 2. GEMM A8W8 (`ck_gemm_a8w8`)

| Item | Detail |
|------|--------|
| **Class** | `GemmA8W8Tuner` |
| **File** | `csrc/ck_gemm_a8w8/gemm_a8w8_tune.py` |
| **Production Op** | `gemm_a8w8` |
| **Reference** | `gemm_a8w8_ref()` |
| **Shape Keys** | M, N, K, q_dtype_w |

```bash
python3 csrc/ck_gemm_a8w8/gemm_a8w8_tune.py \
    --run_config \
    -i aiter/configs/a8w8_untuned_gemm.csv \
    -o aiter/configs/a8w8_tuned_gemm.csv
```

### 3. GEMM A8W8 BlockScale (`ck_gemm_a8w8_blockscale`)

| Item | Detail |
|------|--------|
| **Class** | `GemmA8W8BlockScaleTuner` |
| **File** | `csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py` |
| **Production Op** | `gemm_a8w8_blockscale` |
| **Reference** | `run_torch()` |
| **Shape Keys** | M, N, K |

```bash
python3 csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py \
    --run_config \
    -i aiter/configs/a8w8_blockscale_untuned_gemm.csv \
    -o aiter/configs/a8w8_blockscale_tuned_gemm.csv
```

### 4. GEMM A8W8 BlockScale BPreShuffle (`ck_gemm_a8w8_blockscale_bpreshuffle`)

| Item | Detail |
|------|--------|
| **Class** | `Gemma8W8BlockScaleBPreShuffleTuner` |
| **File** | `csrc/ck_gemm_a8w8_blockscale_bpreshuffle/gemm_a8w8_blockscale_bpreshuffle_tune.py` |
| **Production Op** | `gemm_a8w8_blockscale_bpreshuffle` |
| **Reference** | `self.run_torch()` (static method) |
| **Shape Keys** | M, N, K |

```bash
python3 csrc/ck_gemm_a8w8_blockscale_bpreshuffle/gemm_a8w8_blockscale_bpreshuffle_tune.py \
    --run_config \
    -i aiter/configs/a8w8_blockscale_bpreshuffle_untuned_gemm.csv \
    -o aiter/configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv
```

### 5. GEMM A4W4 BlockScale (`ck_gemm_a4w4_blockscale`)

| Item | Detail |
|------|--------|
| **Class** | `GemmA4W4BlockScaleTuner` |
| **File** | `csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_tune.py` |
| **Production Op** | `gemm_a4w4` |
| **Reference** | `run_torch()` |
| **Shape Keys** | M, N, K |
| **Special** | Output is padded; uses `out[:M]` slice for correctness comparison |

```bash
python3 csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_tune.py \
    --run_config \
    -i aiter/configs/a4w4_blockscale_untuned_gemm.csv \
    -o aiter/configs/a4w4_blockscale_tuned_gemm.csv
```

### 6. MOE 2-Stage (`ck_gemm_moe_2stages_codegen`)

| Item | Detail |
|------|--------|
| **Class** | `GemmMoe2stageTuner` |
| **File** | `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py` |
| **Production Op** | `fused_moe` (from `aiter.fused_moe`) |
| **Reference** | `self.torch_moe_2stages()` |
| **Shape Keys** | token, model_dim, inter_dim, expert, topk, act_type, dtype, q_dtype_a, q_dtype_w, q_type, use_g1u1, doweight_stage1 |
| **Special** | Handles multiple quant types (per_token, per_1x128, per_1x32), g1u1, activation types; generates quantized weights and activations inline |

```bash
python3 csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py \
    --run_config \
    -i aiter/configs/moe_2stages_untuned.csv \
    -o aiter/configs/moe_2stages_tuned.csv
```

### 7. Batched GEMM A8W8 (`ck_batched_gemm_a8w8`)

| Item | Detail |
|------|--------|
| **Class** | `BatchedGemma8W8Tuner` |
| **File** | `csrc/ck_batched_gemm_a8w8/batched_gemm_a8w8_tune.py` |
| **Production Op** | `batched_gemm_a8w8` |
| **Reference** | `run_torch()` |
| **Shape Keys** | B, M, N, K |

```bash
python3 csrc/ck_batched_gemm_a8w8/batched_gemm_a8w8_tune.py \
    --run_config \
    -i aiter/configs/a8w8_untuned_batched_gemm.csv \
    -o aiter/configs/a8w8_tuned_batched_gemm.csv
```

### 8. Batched GEMM BF16 (`ck_batched_gemm_bf16`)

| Item | Detail |
|------|--------|
| **Class** | `BatchedGemmBf16Tuner` |
| **File** | `csrc/ck_batched_gemm_bf16/batched_gemm_bf16_tune.py` |
| **Production Op** | `batched_gemm_bf16` |
| **Reference** | `run_torch()` |
| **Shape Keys** | B, M, N, K |

```bash
python3 csrc/ck_batched_gemm_bf16/batched_gemm_bf16_tune.py \
    --run_config \
    -i aiter/configs/bf16_untuned_batched_gemm.csv \
    -o aiter/configs/bf16_tuned_batched_gemm.csv
```

## Related CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--run_config` | off | Run benchmark only and exit (no tuning) |
| `--compare` | off | Run pre-tune benchmark, tune, post-tune benchmark, show comparison |
| `--errRatio` | 0.05 | Tolerable error ratio for correctness checking |
| `--warmup` | 5 | Warmup iterations for benchmarking |
| `--iters` | 101 | Benchmark iterations |
| `-i` / `--untune_file` | (per tuner) | Input: untuned shapes CSV |
| `-o` / `--tune_file` | (per tuner) | Output: tuned results CSV |

## Key Dependencies

- `aiter.test_common.run_perftest` -- Returns `(output_data, avg_time_us)`. Used for timing.
- `aiter.test_common.checkAllclose` -- Returns error ratio (float). 0 = perfect match; compared against `--errRatio` for pass/fail.
- Production operators in `aiter.ops.gemm_op_a8w8`, `aiter.ops.gemm_op_a4w4`, `aiter.ops.batched_gemm_op_a8w8`, `aiter.ops.batched_gemm_op_bf16`, `aiter.fused_moe` -- The actual dispatch functions that read tuned config CSVs internally.

## Adding `run_config` to a New Tuner

To add `run_config` support to a new tuner subclass:

1. Add `"config_env_name": "AITER_CONFIG_YOUR_KERNEL"` to `ARG_DEFAULTS` (required for post-tune config switching).
2. Override `_clear_op_caches(self)` to clear your operator's specific config caches.
3. Override `run_config(self, args)` in your tuner class.
4. Read shapes from `self.untunedf` (the truly untuned shapes after `pre_process` filtering).
5. For each shape:
   - Generate test data (reuse your tuner's `generate_data()` function).
   - Call the **production operator** (from `aiter/ops/` or `aiter/fused_moe`), not the tuning-specific one.
   - Benchmark with `run_perftest(op, *args, num_warmup=args.warmup, num_iters=args.iters)`.
   - Verify correctness with `checkAllclose(output, reference)` and compare against `args.errRatio`.
6. Return a list of `{"shape": str, "us": float, "status": str}` dicts.

Template:

```python
class MyTuner(GemmCommonTuner):
    ARG_DEFAULTS = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": f"{MY_CONFIG_FILE}",
        "config_env_name": "AITER_CONFIG_MY_KERNEL",
    }

    def _clear_op_caches(self):
        from aiter.ops.your_op import get_config_func
        if hasattr(get_config_func, "file_cache"):
            get_config_func.file_cache.clear()

    def run_config(self, args):
        from aiter.ops.your_op import your_production_op
        from aiter.test_common import run_perftest, checkAllclose

        untunedf = self.untunedf
        results = []
        for i in range(len(untunedf)):
            M = int(untunedf.loc[i, "M"])
            N = int(untunedf.loc[i, "N"])
            K = int(untunedf.loc[i, "K"])
            shape_str = f"({M}, {N}, {K})"
            try:
                # Generate data
                data = generate_data(M, N, K, seed=0)
                # Run production operator
                out, us = run_perftest(
                    your_production_op, *data,
                    num_warmup=args.warmup, num_iters=args.iters,
                )
                # Verify correctness
                ref = your_reference_fn(*data)
                err_ratio = checkAllclose(out, ref, msg=f"run_config {shape_str}")
                status = "ok" if err_ratio <= args.errRatio else "mismatch"
                results.append({"shape": shape_str, "us": us, "status": status})
            except Exception as e:
                results.append({"shape": shape_str, "us": -1, "status": f"error:{e}"})
        return results
```
