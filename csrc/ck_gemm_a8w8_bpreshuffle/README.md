# CK GEMM A8W8 BPreshuffle Tune

1. Install aiter:
`cd $aiter_path`
`python3 setup.py develop`

2. Add GEMM shapes in `aiter/configs/a8w8_bpreshuffle_untuned_gemm.csv`
    |**M**|**N**|**K**|******q_dtype_w******|
    |-----|-----|-----|---------------------|
    |128  |1536 |7168 |torch.float8_e4m3fnuz|


3. Start tuning:
Run the following cmd to start tuning, please wait a few minutes as it will build gemm_a8w8_bpreshuffle_tune via jit:
`python3 csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py -i aiter/configs/a8w8_bpreshuffle_untuned_gemm.csv -o aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv`
You can find the results of this tuning in `aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv`, like this:
    |**cu_num**|**M**|**N**|**K**|**q_dtype_w**        |**libtype**|**kernelId**|**splitK**|**us**|**kernelName**|**tflops**|**bw**|**errRatio**|
    |----------|-----|-----|-----|---------------------|-----------|------------|----------|------|--------------|----------|------|------------|
    |80        |128  |1536 |7168 |torch.float8_e4m3fnuz|    ck     | 23         |0         |32.99 |xxxxxxxx      |125.4     |89.5  |0.01        |

    `cu_num` means the number of compute units, and it is used to distinguish between graphics.
    `q_dtype_w` means the quantization data type of weight, and it is used to distinguish between different quantization data types. support torch.int8 and fp8

4. Build tuned kernels and test:
Test the performance, modify the test instance in `op_tests/test_gemm_a8w8.py` and run it, please wait a few minutes as it will build gemm_a8w8 tuned kernels in `aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv` via jit:
`python3 op_tests/test_gemm_a8w8.py`
If you have built gemm_a8w8_bpreshuffle kernels before tuning new GEMM shapes, please add `AITER_REBUILD=1` before your test cmd, such as `AITER_REBUILD=1 python3 op_tests/test_gemm_a8w8.py`. It will rebuild kernels from `AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE`, the default one will be results merged from `aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv` and tuned fmoe csv under `aiter/configs/model_configs/xx_a8w8_bpreshuffle_tuned_gemm_xx.csv`, the merged result is store in `/tmp/aiter_configs/a8w8_bpreshuffle_tuned_gemm.csv`.

## More Options

### Library Selection

#### `--libtype`
- **Type**: List of strings
- **Default**: `["all"]`
- **Choices**: `all`, `asm`, `ck`, `cktile`
- **Description**: Choose which library implementations to tune. You can specify one or multiple library types to compare their performance.
  - `all`: Tune all available library implementations
  - `asm`: Use assembly optimized kernels
  - `ck`: Use Composable Kernel library
  - `cktile`: Use CK Tile library

**Example**:
```bash
--libtype all
--libtype ck,cktile
--libtype cktile
```

### Output Configuration

#### `-o2, --profile_file`
- **Type**: String
- **Default**: `""` (empty string)
- **Description**: Optional output file to store **all** tuning results (not just the best ones). Useful for profiling and analyzing all kernel candidates.

**Example**:
```bash
--profile_file aiter/configs/profile_a8w8_bpreshuffle_all.csv
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

## More
If you use flag `PREBUILD_KERNELS=1` when you install aiter, it will build gemm a8w8 bpreshuffle kernels in tuned bpreshuffle gemm csv by default. If you want to use the new result of gemm_a8w8_bpreshuffle_tune, please remove `build` and `*.so` in `aiter/jit` first, then re-intall aiter after finishing tune. This can take a lot of time and is not recommended.
