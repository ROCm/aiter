# CK Batched GEMM A8W8 Tune

1. Install aiter:
`cd $aiter_path`
`python3 setup.py develop`

2. Add GEMM shapes in `aiter/configs/a8w8_untuned_batched_gemm.csv`
    |**B**|**M**|**N**|**K**|
    |-----|-----|-----|-----|
    |16   |128  |1536 |7168 |

3. Start tuning:
Run the following cmd to start tuning, please wait a few minutes as it will build batched_gemm_a8w8_tune via jit:
`python3 csrc/ck_batched_gemm_a8w8/batched_gemm_a8w8_tune.py -i aiter/configs/a8w8_untuned_batched_gemm.csv -o aiter/configs/a8w8_tuned_batched_gemm.csv`
You can find the results of the tuning in `aiter/configs/a8w8_tuned_batched_gemm.csv`, like this:
    |**gfx**  |**cu_num**|**B**|**M**|**N**|**K**|**kernelId**|**splitK**|**us**|**kernelName**|**tflops**|**bw**|**errRatio**|
    |---------|----------|-----|-----|-----|-----|------------|----------|------|--------------|----------|------|------------|
    |gfx942   |80        |16   |128  |1536 |7168 |23          |0         |32.99 |xxxxxxxx      |125.4     |89.5  |0.01        |

    `gfx` identifies the GPU architecture (e.g. `gfx942`, `gfx950`). `cu_num` is the number of compute units and distinguishes partitioned or binned variants of the same architecture (e.g. MI308X vs MI300X both use `gfx942`).

4. Build tuned kernels and test:
Test the performance, modify the test instance in `op_tests/test_batched_gemm_a8w8.py` and run it, please wait a few minutes as it will build batched_gemm_a8w8 tuned kernels in `aiter/configs/a8w8_tuned_batched_gemm.csv` via jit:
`python3 op_tests/test_batched_gemm_a8w8.py`
If you have built batched_gemm_a8w8 kernels before tuning new GEMM shapes, please add `AITER_REBUILD=1` before your test cmd, such as `AITER_REBUILD=1 python3 op_tests/test_batched_gemm_a8w8.py`. It will rebuild kernels from `AITER_CONFIG_A8W8_BATCHED_GEMM`, the default one will be results merged from `aiter/configs/a8w8_tuned_batched_gemm.csv` and tuned fmoe csv under `aiter/configs/model_configs/xx_a8w8_tuned_batched_gemm_xx.csv`, the merged result is store in `/tmp/aiter_configs/a8w8_tuned_batched_gemm.csv`.

## More Options

### Output Configuration

#### `-o2, --profile_file`
- **Type**: String
- **Default**: `""` (empty string)
- **Description**: Optional output file to store **all** tuning results (not just the best ones). Useful for profiling and analyzing all kernel candidates.

**Example**:
```bash
--profile_file aiter/configs/profile_a8w8_batched_all.csv
```

#### `--sort`
- **Type**: Boolean (True/False)
- **Default**: `True` (enabled by default for GEMM tuners)
- **Description**: Sort the output file according to the key columns (e.g., `cu_num`, `N`, `M`, `K` for GEMM). Useful for maintaining consistent ordering in result files. The flag is enabled by default to ensure results are always sorted.


**Example**:
```bash
--sort True   # Enable sorting (default)
--sort False  # Disable sorting
```

### Tuning Configuration

#### `--errRatio`
- **Type**: Float
- **Default**: `0.05` (5%)
- **Description**: Tolerable error ratio threshold. Only kernels with error ratios below this threshold will be considered valid candidates.

**Example**:
```bash
--errRatio 0.01
```

#### `--mp`
- **Type**: Integer
- **Default**: Number of available GPUs
- **Description**: Number of parallel processes to use for tuning across multiple GPUs.

**Example**:
```bash
--mp 4
```

#### `--batch`
- **Type**: Integer
- **Default**: `100`
- **Description**: Number of shapes to tune in each batch.

**Example**:
```bash
--batch 50
```

#### `-k, --splitK`
- **Type**: Flag (boolean)
- **Default**: `False`
- **Description**: Enable split-K optimization for GEMM kernels. Split-K divides the K dimension across multiple workgroups to improve parallelism and performance for certain shapes.

**Example**:
```bash
-k
--splitK
```

#### `--all`
- **Type**: Flag (boolean)
- **Default**: `False`
- **Description**: Retune all shapes based on file relationship.
- If `tune_file` == `untune_file`: Retune all shapes in the tune file
- If `tune_file` != `untune_file`: Retune shapes that exist in untuned file


**Example**:
```bash
--all
```

#### `--run_config [TUNED_CSV]`
- **Type**: Optional argument
- **Default**: disabled
- **Description**: Run production-operator benchmark only and exit (no tuning).
  - `--run_config /path/to/tuned.csv`: read shapes from that tuned CSV and run tuned kernels from that file.
  - `--run_config` (no path): read shapes from `-i/--untune_file` and run default kernels.

**Examples**:
```bash
# benchmark tuned kernels from specified tuned config
python3 csrc/ck_batched_gemm_a8w8/batched_gemm_a8w8_tune.py \
  --run_config aiter/configs/a8w8_tuned_batched_gemm.csv

# benchmark default kernels using shapes from -i
python3 csrc/ck_batched_gemm_a8w8/batched_gemm_a8w8_tune.py \
  -i aiter/configs/a8w8_untuned_batched_gemm.csv --run_config
```

#### `--compare`
- **Type**: Flag (boolean)
- **Default**: `False`
- **Description**: Run pre-tune and post-tune production benchmark, print compare results, and keep a compare candidate CSV.
  - Pre-tune reads shapes from `-i/--untune_file`.
  - Post-tune uses configs written to `<tune_file>.candidate.csv` during the compare run.
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

### Profiling Configuration

#### `--warmup`
- **Type**: Integer
- **Default**: `5`
- **Description**: Number of warmup iterations before profiling.

**Example**:
```bash
--warmup 10
```

#### `--iters`
- **Type**: Integer
- **Default**: `101`
- **Description**: Number of profiling iterations to run for performance measurement.

**Example**:
```bash
--iters 200
```

#### `--timeout`
- **Type**: Integer
- **Default**: `None`
- **Description**: Timeout in seconds for each task group.

**Example**:
```bash
--timeout 300
```

### Debugging and Verbose Output

#### `-v, --verbose`
- **Type**: Flag (boolean)
- **Default**: `False`
- **Description**: Enable verbose output with detailed logging information.

**Example**:
```bash
-v
```
## Notes
If you use flag `PREBUILD_KERNELS=1` when you install aiter, it will build batched_gemm_a8w8 kernels in tuned gemm csv by default. If you want to use the new result of batched_gemm_a8w8_tune, please remove `build` and `*.so` in `aiter/jit` first, then re-install aiter after finishing tune. This can take a lot of time and is not recommended.
