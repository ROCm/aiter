# Skill: Add run_config/compare to a New Tuner

When adding a new tuning operator, follow these steps to support `--run_config` (benchmark only), `--compare` (pre/post tune comparison), and `--update_improved` (conditionally write improved results back to the final tuned CSV).

## Prerequisites

- The tuner inherits from `GemmCommonTuner` or `TunerCommon` (in `aiter/utility/base_tuner.py`)
- The base class already provides `--run_config`, `--compare`, and `--update_improved` CLI flags and the `run()` flow
- You only need to implement 3 things in the subclass

## Step 1: Add `config_env_name` to `ARG_DEFAULTS`

Find the env var name that controls which config CSV the production operator reads. It's defined in `aiter/jit/core.py` as `AITER_CONFIGS` attributes (e.g., `AITER_CONFIG_GEMM_A8W8`).

```python
class MyTuner(GemmCommonTuner):
    ARG_DEFAULTS = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": f"{MY_CONFIG_FILE}",
        "untune_file": "aiter/configs/my_untuned.csv",
        "config_env_name": "AITER_CONFIG_MY_KERNEL",  # <-- ADD THIS
    }
```

**How to find the right env var name:** grep for your config file name in `aiter/jit/core.py`:
```bash
grep -n "MY_KERNEL" aiter/jit/core.py
```

### Existing mappings

| Tuner | config_env_name |
|-------|-----------------|
| A8W8 | `AITER_CONFIG_GEMM_A8W8` |
| A8W8 BPreShuffle | `AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE` |
| A8W8 BlockScale | `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE` |
| A8W8 BlockScale BPreShuffle | `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE` |
| A4W4 BlockScale | `AITER_CONFIG_GEMM_A4W4` |
| BF16 (GemmTuner) | `AITER_CONFIG_GEMM_BF16` |
| MOE 2-Stage | `AITER_CONFIG_FMOE` |
| Batched A8W8 | `AITER_CONFIG_A8W8_BATCHED_GEMM` |
| Batched BF16 | `AITER_CONFIG_BF16_BATCHED_GEMM` |

## Step 2: Override `_clear_op_caches(self)`

Clear the operator's config caches so post-tune benchmark picks up new configs. Find the cache by looking at how the production operator loads configs.

**Cache types by operator pattern:**

| Pattern | How to clear | Example |
|---------|-------------|---------|
| `@lru_cache` function | `func.cache_clear()` | `get_GEMM_A16W16_config_.cache_clear()` |
| `file_cache` dict on function | `func.file_cache.clear()` | `get_GEMM_config_with_quant_type.file_cache.clear()` |
| Module-level dict | `the_dict.clear()` | `ck_batched_gemm_dict.clear()` |

**How to find caches:** look in the production operator's module (e.g., `aiter/ops/gemm_op_a8w8.py` or `aiter/tuned_gemm.py`) for `@lru_cache`, `@functools.lru_cache`, or dict-based caches that store loaded CSV data.

### Example: lru_cache (BF16 GemmTuner)
```python
def _clear_op_caches(self):
    from aiter.tuned_gemm import get_GEMM_A16W16_config_, get_GEMM_A16W16_config
    get_GEMM_A16W16_config_.cache_clear()
    get_GEMM_A16W16_config.cache_clear()
```

### Example: file_cache (A8W8 tuners)
```python
def _clear_op_caches(self):
    from aiter.ops.gemm_op_a8w8 import get_GEMM_config_with_quant_type
    if hasattr(get_GEMM_config_with_quant_type, "file_cache"):
        get_GEMM_config_with_quant_type.file_cache.clear()
```

## Step 3: Override `run_config(self, args)`

This method benchmarks the **production operator** (not the tuning kernels) on all shapes.

### Pattern

```python
def run_config(self, args):
    from aiter.ops.your_op import your_production_op    # production operator
    from aiter.test_common import run_perftest, checkAllclose

    untunedf = self.untunedf
    results = []
    for i in range(len(untunedf)):
        # 1. Read shape from CSV
        M = int(untunedf.loc[i, "M"])
        N = int(untunedf.loc[i, "N"])
        K = int(untunedf.loc[i, "K"])
        # ... read other shape-specific columns
        shape_str = f"({M}, {N}, {K})"
        try:
            # 2. Generate data (reuse existing generate_data in same file)
            data = generate_data(M, N, K, ...)

            # 3. Benchmark production operator
            out, us = run_perftest(
                your_production_op, *positional_args,
                keyword=keyword_args,
                num_warmup=args.warmup, num_iters=args.iters,
            )

            # 4. Verify correctness against reference
            # Use the SAME rtol/atol as the tuning code — check the tuner's
            # Gemm/kernel class for how it sets self.atol/self.rtol per dtype.
            ref = your_reference_fn(...)
            err_ratio = checkAllclose(out, ref, atol=_atol, rtol=_rtol,
                                      msg=f"run_config {shape_str}")
            status = "ok" if err_ratio <= args.errRatio else "mismatch"
            results.append({"shape": shape_str, "us": us, "status": status})
        except Exception as e:
            results.append({"shape": shape_str, "us": -1, "status": f"error:{e}"})
    return results
```

### Key points

1. **Import production op, not tuning op** — e.g., `from aiter.ops.gemm_op_a8w8 import gemm_a8w8`, not the CK kernel list
2. **Use `self.untunedf`** — already filtered by `pre_process()`, contains only shapes to benchmark
3. **Use `run_perftest`** — returns `(output_tensor, avg_time_us)`, pass `num_warmup=args.warmup, num_iters=args.iters`
4. **Use `checkAllclose`** — returns error ratio (float), 0 = perfect match
5. **Match tuning rtol/atol** — `checkAllclose` defaults to `atol=0.01, rtol=0.01`, but the tuning code may use different tolerances per dtype (e.g., `atol=0.05, rtol=0.05` for bf16). Check the tuner's kernel class for `self.atol`/`self.rtol` and pass the same values to `checkAllclose` in `run_config`, otherwise post-tune verification may report false mismatches.
6. **Return format** — list of `{"shape": str, "us": float, "status": str}` where status is `"ok"`, `"mismatch"`, or `"error:msg"`
7. **Wrap in try/except** — catch all exceptions, report as error status

## Verification

After implementing, test on actual GPU. Use the entry-point script (e.g., `gemm_tuner.py`) rather than the class file (`GemmTuner.py`) directly — the entry point wraps tuning in `mp.Process` with automatic retry on GPU memory faults/coredumps.

```bash
# Benchmark only (no tuning)
python3 your_entry.py --run_config -i untuned.csv -o tuned.csv

# Full comparison (pre-tune → tune → post-tune → speedup table, no final CSV update)
python3 your_entry.py --compare -i untuned.csv -o tuned.csv

# Compare and apply improved shapes to the final tuned CSV
python3 your_entry.py --compare --update_improved -i untuned.csv -o tuned.csv

# Retune all shapes (including already-tuned ones) with comparison and gated update
python3 your_entry.py --compare --update_improved --all -i untuned.csv -o tuned.csv
```

**GPU memory faults:** Tuning may trigger GPU memory access faults (VM_L2_PROTECTION_FAULT/coredump). The entry-point script catches these (negative exit code) and retries automatically. If faults persist, re-run with `--timeout <seconds>` to set a per-shape timeout — the value depends on the shapes being tuned.

Expected output:
- `--run_config`: benchmark table with all shapes showing time and status, then exit
- `--compare`: pre-tune table → tuning → post-tune table → comparison table with speedup ratios, plus a retained `*.candidate.csv`
- `--compare --update_improved`: same compare flow, and write shapes meeting `--min_improvement_pct` back to the final tuned CSV
- `--compare --update_improved --all`: same as above, but retunes every shape instead of only untuned ones

## Reference implementations

| Tuner | File | Production Op | Cache Type |
|-------|------|--------------|------------|
| A8W8 | `csrc/ck_gemm_a8w8/gemm_a8w8_tune.py:122` | `gemm_a8w8` | `file_cache` |
| BF16 | `gradlib/gradlib/GemmTuner.py:770` | `gemm_a16w16` | `lru_cache` |
| A8W8 BPreShuffle | `csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py` | `gemm_a8w8_bpreshuffle`/`gemm_a8w8_ASM` | `file_cache` |
| MOE 2-Stage | `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py` | `fused_moe` | module-level dicts |

