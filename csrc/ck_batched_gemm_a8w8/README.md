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
    |**cu_num**|**B**|**M**|**N**|**K**|**kernelId**|**splitK**|**us**|**kernelName**|**tflops**|**bw**|**errRatio**|
    |----------|-----|-----|-----|-----|------------|----------|------|--------------|----------|------|------------|
    |80        |16   |128  |1536 |7168 |23          |0         |32.99 |xxxxxxxx      |125.4     |89.5  |0.01        |

    `cu_num` means the number of compute units, and it is used to distinguish between graphics.

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
- **Type**: Flag (boolean)
- **Default**: `False`
- **Description**: Sort the output file according to the key columns (e.g., `cu_num`, `N`, `M`, `K` for GEMM). Useful for maintaining consistent ordering in result files.


**Example**:
```bash
--sort
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
