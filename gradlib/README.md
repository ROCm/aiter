```
                      _ _ _ _
   __ _ _ __ __ _  __| | (_) |__
  / _` | '__/ _` |/ _` | | | '_ \
 | (_| | | | (_| | (_| | | | |_) |
  \__, |_|  \__,_|\__,_|_|_|_.__/
  |___/
```

## What Is gradlib
`gradlib` is a vLLM-derived tuning toolkit for GEMM kernels. It helps you find the best kernel parameters for your current hardware to improve model inference performance.

## Quick Start

### 1) Capture Untuned GEMM Shapes
Replace `F.linear` with `tgemm.mm` in `aiter/tuned_gemm.py`, then run your workload:

```bash
AITER_TUNE_GEMM=1 python {workload_tests}
```

Captured shapes are written to `aiter/configs/bf16_untuned_gemm.csv`.

### 2) Tune GEMMs

The multi-backend tuner lives at `csrc/gemm_a16w16/gemm_a16w16_tune.py` and supports
asm, opus, flydsl, triton, skinny, torch, and hipblaslt (opt-in) backends:

```bash
python3 csrc/gemm_a16w16/gemm_a16w16_tune.py \
  --tuned_file aiter/configs/bf16_tuned_gemm.csv \
  --input_file aiter/configs/bf16_untuned_gemm.csv
```

To tune with a specific backend:

```bash
# opus only
python3 csrc/gemm_a16w16/gemm_a16w16_tune.py \
  --input_file aiter/configs/bf16_untuned_gemm.csv \
  --libtype opus

# all backends including hipblaslt (opt-in, requires gradlib)
python3 csrc/gemm_a16w16/gemm_a16w16_tune.py \
  --input_file aiter/configs/bf16_untuned_gemm.csv \
  --with-hipblaslt

# hipblaslt only
python3 csrc/gemm_a16w16/gemm_a16w16_tune.py \
  --input_file aiter/configs/bf16_untuned_gemm.csv \
  --libtype hipblaslt --with-hipblaslt
```

The legacy hipblaslt-only tuner is still available at `gradlib/gradlib/gemm_tuner.py`.

Tuned results are saved to `aiter/configs/bf16_tuned_gemm.csv`.

Example columns:

|**cu_num**|**M**|**N**|**K**|**bias**|**dtype**|**outdtype**|**scaleAB**|**bpreshuffle**|**libtype**|**solidx**|**splitK**|**us**|**kernelName**|**tflops**|**bw**|
|----------|-----|-----|-----|--------|---------|-----------|-----------|---------------|-----------|----------|----------|------|--------------|----------|------|
|80|128|1536|7168|False|torch.bfloat16|torch.float32|False|False|asm|5|2|10.6|bf16gemm_fp32bf16_tn_64x64_splitk_clean|xx|xx|

Notes:
- `cu_num`: compute units for current GPU.
- `bpreshuffle`: whether weight is shuffled.
- `dtype`: input dtype (`hipblaslt` supports fp8/bf16/fp16; asm/triton supports bf16/fp16).
- `libtype`: kernel backend (`asm` / `opus` / `flydsl` / `triton` / `skinny` / `torch` / `hipblaslt`).
- `splitK`: split-K factor (backend-dependent).
- `tflops`: throughput in TFLOPS.
- `bw`: bandwidth in GB/s.

### 3) Run Your Workload Normally
After tuning, run your model/tests as usual.

## More Features

#### `-o2, --profile_file`
- **Type**: String
- **Default**: `""` (empty string)
- **Required**: No
- **Description**: Optional output file storing **all** tuning candidates (not only the best).

**Example**:
```bash
--profile_file /path/to/all_results.csv
```

#### `--mp`
- **Type**: Integer
- **Default**: `torch.cuda.device_count()`
- **Description**: Number of parallel processes / GPUs used for tuning.

**Example**:
```bash
--mp 1
```

### Tuning Configuration

#### `--libtype`
- **Type**: String (comma-separated list)
- **Default**: `all`
- **Description**: Choose which backends to tune: `all`, `asm`, `opus`, `flydsl`, `triton`, `skinny`, `torch`, `hipblaslt`. `hipblaslt` requires `--with-hipblaslt`.

**Example**:
```bash
--libtype asm
--libtype opus,flydsl
--libtype hipblaslt --with-hipblaslt
```

#### `--with-hipblaslt`
- **Type**: Flag (boolean)
- **Default**: `False`
- **Description**: Include hipblaslt in tuning (disabled by default). hipblaslt tuning is also available standalone via `gradlib/gradlib/gemm_tuner.py`.

#### `--outdtype`
- **Type**: String
- **Default**: None (uses CSV column value or bf16)
- **Choices**: `f32`, `f16`, `bf16`, `fp8`
- **Description**: Output dtype override.

#### `--errRatio`
- **Type**: Float
- **Default**: `0.05` (5%)
- **Description**: Max tolerable error ratio for valid kernels.

**Example**:
```bash
--errRatio 0.01
--errRatio 0.10
```

#### `--sort`
- **Type**: Flag (boolean)
- **Default**: `False`
- **Description**: Sort output by key columns.

**Example**:
```bash
--sort
```

#### `--all`
- **Type**: Flag (boolean)
- **Default**: `False`
- **Description**: Retune all shapes based on file relationship.
  - If `tune_file == untune_file`: retune all shapes in tune file.
  - If `tune_file != untune_file`: retune shapes that exist in untuned file.

**Example**:
```bash
--all
```

#### `--run_config [TUNED_CSV]`
- **Type**: Optional argument
- **Default**: disabled
- **Description**: Run production benchmark only and exit (no tuning).
  - `--run_config /path/to/tuned.csv`: read shapes from that tuned CSV and run tuned kernels from that file.
  - `--run_config` (no path): read shapes from `--input_file` (or auto-generated shapes when `--input_file` is omitted) and run default kernels.

**Examples**:
```bash
# benchmark tuned kernels from specified tuned config
python3 csrc/gemm_a16w16/gemm_a16w16_tune.py \
  --input_file aiter/configs/bf16_untuned_gemm.csv \
  --run_config aiter/configs/bf16_tuned_gemm.csv

# benchmark default kernels using shapes from --input_file
python3 csrc/gemm_a16w16/gemm_a16w16_tune.py \
  --input_file aiter/configs/bf16_untuned_gemm.csv \
  --run_config
```

#### `--compare`
- **Type**: Flag (boolean)
- **Default**: `False`
- **Description**: Run pre-tune and post-tune production benchmark, print compare results, and keep a compare candidate CSV.
  - Pre-tune reads shapes from `--input_file` (or auto-generated shapes).
  - Post-tune uses configs written to `<tuned_file>.candidate.csv` during the compare run.
  - The final tuned CSV is only updated when `--update_improved` is also set.
  - Shapes with no valid pre-run baseline can still update when the post-tune benchmark passes.

**Example**:
```bash
--compare
```

#### `--update_improved`
- **Type**: Flag (boolean)
- **Default**: `False`
- **Description**: With `--compare`, update the final tuned CSV for shapes improved by at least `--min_improvement_pct`, or for shapes with no valid pre-run baseline when the post-tune benchmark passes.

**Example**:
```bash
--compare --update_improved
```

#### `--min_improvement_pct`
- **Type**: Float
- **Default**: `3.0`
- **Description**: With `--compare --update_improved`, the minimum percentage improvement required before a compared result replaces the final tuned CSV entry when both pre/post benchmarks are valid. Shapes with no valid pre-run baseline but passing post-tune are still allowed to update.

### Debugging and Verbose Output

#### `-v, --verbose`
- **Type**: Flag (boolean)
- **Default**: `False`
- **Description**: Enable verbose logs.

**Example**:
```bash
--verbose
-v
```

## hipBLASLt Online Tuning

Enable hipBLASLt online tuning:

```bash
export HIP_ONLINE_TUNING=1
```

The one-time overhead can take several minutes. Results are saved to `hip_online_tuning_res.csv`.
