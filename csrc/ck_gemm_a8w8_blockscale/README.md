# CK GEMM A8W8 Blockscale Tune

1. Install aiter:
`cd $aiter_path`
`python3 setup.py develop`

2. Add GEMM shapes in `aiter/configs/a8w8_blockscale_untuned_gemm.csv`
    |**M**|**N**|**K**|
    |-----|-----|-----|
    |128  |1536 |7168 |

3. Start tuning:
Run one of the following cmds to start tuning, please wait a few minutes as it will build `gemm_a8w8_blockscale_tune` via jit:

Tune standard blockscale CK + CKTile candidates:
`python3 csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py -i aiter/configs/a8w8_blockscale_untuned_gemm.csv -o aiter/configs/a8w8_blockscale_tuned_gemm.csv --libtype both`

Tune preshuffled-B candidates, including ASM when available:
`python3 csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py -i aiter/configs/a8w8_blockscale_untuned_gemm.csv -o aiter/configs/a8w8_blockscale_tuned_gemm.csv --libtype asm --preshuffle`

Exhaustively tune CK, CKTile and ASM across both standard and preshuffleB modes:
`python3 csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py -i aiter/configs/a8w8_blockscale_untuned_gemm.csv -o aiter/configs/a8w8_blockscale_tuned_gemm.csv --libtype all`

`--libtype` supports `ck`, `cktile`, `asm`, `both` and `all`:
- `ck`: tune CK legacy kernels under the current `--preshuffle` setting
- `cktile`: tune CKTile kernels under the current `--preshuffle` setting
- `asm`: tune ASM kernels under the current `--preshuffle` setting
- `both`: tune `ck + cktile` under the current `--preshuffle` setting
- `all`: tune `ck + cktile + asm` across both standard and preshuffleB modes

`--preshuffle` enables B-matrix preshuffle for candidate generation. When it is enabled, the tuner feeds preshuffled weights and transposed `x_scale`; ASM kernel discovery is also filtered by the `bpreshuffle` column in `fp8gemm_bf16_blockscale.csv`.

You can find the results of the tuning in `aiter/configs/a8w8_blockscale_tuned_gemm.csv`, like this:
    |**cu_num**|**M**|**N**|**K**|**libtype**|**kernelId**|**splitK**|**us**|**kernelName**|**tflops**|**bw**|**errRatio**|
    |----------|-----|-----|-----|-----------|------------|----------|------|--------------|----------|------|------------|
    |80        |128  |1536 |7168 |cktile     |23          |0         |32.99 |xxxxxxxx      |125.4     |89.5  |0.01        |

    `cu_num` means the number of compute units, and it is used to distinguish between graphics. `libtype` records which backend produced the result. `asm` entries are mainly useful for profiling and comparison; actual runtime selection still depends on the corresponding operator path and target gfx.

4. Build tuned kernels and test:
Test the performance, modify the test instance in `op_tests/test_gemm_a8w8_blockscale.py` and run it, please wait a few minutes as it will build gemm_a8w8_blockscale tuned kernels in `aiter/configs/a8w8_blockscale_tuned_gemm.csv` via jit:
`python3 op_tests/test_gemm_a8w8_blockscale.py`
If you want to validate the preshuffled-B path, enable the corresponding preshuffle test flow in `op_tests/test_gemm_a8w8_blockscale.py` before running the test.
If you have built gemm_a8w8 kernels before tuning new GEMM shapes, please add `AITER_REBUILD=1` before your test cmd, such as `AITER_REBUILD=1 python3 op_tests/test_gemm_a8w8_blockscale.py`. It will rebuild kernels from `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE`, the default one will be results merged from `aiter/configs/a8w8_blockscale_tuned_gemm.csv` and tuned fmoe csv under `aiter/configs/model_configs/xx_a8w8_blockscale_tuned_gemm_xx.csv`, the merged result is store in `/tmp/aiter_configs/a8w8_blockscale_tuned_gemm.csv`.

## More Options

### Output Configuration

#### `-o2, --profile_file`
- **Type**: String
- **Default**: `""` (empty string)
- **Description**: Optional output file to store **all** tuning results (not just the best ones). Useful for profiling and analyzing all kernel candidates.

**Example**:
```bash
--profile_file aiter/configs/profile_a8w8_blockscale_all.csv
```

#### `--sort`
- **Type**: Boolean (True/False)
- **Default**: `True` (enabled by default for GEMM tuners)
- **Description**: Sort the output file according to the key columns(e.g., `cu_num`, `N`, `M`, `K` for GEMM). Useful for maintaining consistent ordering in result files. The flag is enabled by default to ensure results are always sorted.

**Example**:
```bash
--sort True   # Enable sorting (default)
--sort False  # Disable sorting
```

### Tuning Configuration

#### `--libtype`
- **Type**: String
- **Default**: `both`
- **Choices**: `ck`, `cktile`, `asm`, `both`, `all`
- **Description**: Select which backend candidates to tune. `both` tunes `ck + cktile` under the current `--preshuffle` setting. `all` expands the search to `ck`, `cktile` and `asm` across both standard and preshuffleB modes.

**Example**:
```bash
--libtype ck
--libtype asm --preshuffle
--libtype all
```

#### `--preshuffle`
- **Type**: Flag (boolean)
- **Default**: `False`
- **Description**: Enable B-matrix preshuffle candidate generation. This switches CK/CKTile tuning to preshuffled weights and transposed `x_scale`, and also filters ASM candidates with `bpreshuffle=1` from `fp8gemm_bf16_blockscale.csv`.

**Example**:
```bash
--preshuffle
```

#### `--blockPerCu`
- **Type**: Integer list
- **Default**: `1 ... BLOCK_PER_CU_MAX`
- **Description**: Limit CKTile candidates to selected `BlockPerCu` values. This option only affects `cktile` candidates and is ignored by `ck` and `asm`.

**Example**:
```bash
--blockPerCu 1 2 4
```

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
--mp 1
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
- **Description**: Enable split-K optimization for GEMM kernels. CK/CKTile candidates use `compute_gemm_SplitK(...)` to enumerate valid split-K values. ASM candidates sweep `splitK=1..8` only when the selected CSV entry marks split-K support; otherwise only `splitK=1` is tested.

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
## Notes
If you use flag `PREBUILD_KERNELS=1` when you install aiter, it will build gemm a8w8 blockscale kernels in tuned gemm csv by default. If you want to use the new result of gemm_a8w8_blockscale_tune, please remove `build` and `*.so` in `aiter/jit` first, then re-install aiter after finishing tune. This can take a lot of time and is not recommended.
