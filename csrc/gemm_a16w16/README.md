# GEMM A16W16 (BF16) Tune

Multi-backend bf16 GEMM tuner. Searches across asm, opus, flydsl, triton, skinny, and torch backends by default. hipblaslt can be included with `--with-hipblaslt`.

For hipblaslt-only tuning, use `gradlib/gradlib/gemm_tuner.py` instead.

1. Install aiter:
`cd $aiter_path`
`python3 setup.py develop`

2. Add GEMM shapes in `aiter/configs/bf16_untuned_gemm.csv`
    |**M**|**N**|**K**|**bias**|**dtype**|**outdtype**|**scaleAB**|**bpreshuffle**|
    |-----|-----|-----|--------|---------|------------|-----------|---------------|
    |1    |7168 |2048 |False   |torch.bfloat16|torch.bfloat16|False  |False          |

   Or capture shapes automatically by running your workload with `AITER_TUNE_GEMM=1`.

3. Start tuning:

There are two entry points:

- **`gemm_a16w16_tune.py`** — runs the tuner directly. Use this when tuning
  non-hipblaslt backends only (asm, opus, flydsl, triton, skinny, torch).
- **`gemm_tuner.py`** — runs the tuner inside a subprocess with automatic
  retry on GPU crashes (SIGABRT, SIGSEGV). Use this when `--with-hipblaslt`
  is enabled, because hipblaslt is a third-party library whose GPU faults
  cannot be fixed locally. The subprocess wrapper catches these crashes and
  retries automatically (up to 30 times).

```bash
# Tune non-hipblaslt backends (default, no subprocess wrapper needed):
python3 csrc/gemm_a16w16/gemm_a16w16_tune.py \
  --input_file aiter/configs/bf16_untuned_gemm.csv

# Tune all backends including hipblaslt (use subprocess wrapper):
python3 csrc/gemm_a16w16/gemm_tuner.py \
  --input_file aiter/configs/bf16_untuned_gemm.csv \
  --with-hipblaslt

# Tune a specific backend only:
python3 csrc/gemm_a16w16/gemm_a16w16_tune.py \
  --input_file aiter/configs/bf16_untuned_gemm.csv \
  --libtype asm
```

Results are written to `aiter/configs/bf16_tuned_gemm.csv`:
    |**gfx**|**cu_num**|**M**|**N**|**K**|**bias**|**dtype**|**outdtype**|**scaleAB**|**bpreshuffle**|**libtype**|**solidx**|**splitK**|**us**|**kernelName**|**err_ratio**|**tflops**|**bw**|
    |-------|----------|-----|-----|-----|--------|---------|------------|-----------|---------------|-----------|----------|----------|------|--------------|-------------|----------|------|
    |gfx942 |304       |1    |7168 |2048 |False   |torch.bfloat16|torch.bfloat16|False|False       |asm        |1         |1         |12.5  |bf16gemm_...  |0.001        |2.35      |34.1  |

4. Build tuned kernels and test:
```bash
python3 op_tests/test_gemm.py
```
If you have built kernels before tuning, add `AITER_REBUILD=1` to rebuild with new configs.

## Tuner-Specific Options

### `--libtype`
- **Type**: Comma-separated string
- **Default**: `all`
- **Choices**: `all`, `asm`, `hipblaslt`, `triton`, `flydsl`, `torch`, `skinny`, `opus`
- **Description**: Choose which backends to tune. hipblaslt requires **both** `--libtype all` (or `--libtype hipblaslt`) **and** `--with-hipblaslt` to run.

**Example**:
```bash
# tune asm and opus only
--libtype asm,opus

# tune all backends including hipblaslt
--with-hipblaslt

# tune hipblaslt only
--libtype hipblaslt --with-hipblaslt
```

### `--with-hipblaslt`
- **Type**: Flag
- **Default**: disabled
- **Description**: Enable hipblaslt backend (imports from gradlib). This is a **gate switch** — hipblaslt is never run without it, regardless of `--libtype`. With the default `--libtype all`, adding `--with-hipblaslt` is sufficient to include hipblaslt alongside all other backends. When using this flag, it is recommended to run via `gemm_tuner.py` (the subprocess wrapper) so that GPU-level crashes from hipblaslt are retried automatically.

### `--indtype` / `--outdtype`
- **Choices**: `f32`, `f16`, `bf16`, `fp8`
- **Description**: Override input/output dtype for all shapes.

### `--all_bias`
- **Type**: Flag
- **Description**: Tune for both bias and non-bias cases regardless of what was used to collect the shapes.

## Common Options

### `--run_config [TUNED_CSV]`
Run production-operator benchmark only (no tuning).
```bash
# Benchmark tuned kernels:
python3 csrc/gemm_a16w16/gemm_a16w16_tune.py \
  --run_config aiter/configs/bf16_tuned_gemm.csv

# Benchmark default kernels:
python3 csrc/gemm_a16w16/gemm_a16w16_tune.py \
  -i aiter/configs/bf16_untuned_gemm.csv --run_config
```

### `--compare` / `--update_improved`
Run pre-tune and post-tune benchmark, print comparison. With `--update_improved`, update the tuned CSV for shapes improved by at least `--min_improvement_pct` (default 3%).

### `--mp`
Number of GPUs to use for parallel tuning. Default: all available.

### `--errRatio`
Tolerable error ratio threshold (default 0.05).

### `-o2, --profile_file`
Save all candidate results (not just the best) to this file.

### `-v, --verbose`
Enable detailed logging output.
