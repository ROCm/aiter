# CK GEMM A4W4 Blockscale Tune

1. Install aiter:
`cd $aiter_path`
`python3 setup.py develop`

2. Add GEMM shapes in `aiter/configs/a4w4_blockscale_untuned_gemm.csv`
    |**M**|**N**|**K**|
    |-----|-----|-----|
    |128  |1536 |7168 |

3. Start tuning:
Run the following cmd to start tuning, please wait a few minutes as it will build gemm_a4w4_blockscale_tune via jit:
`GEMM_A4W4_BLOCKWISE_HIP_CLANG_PATH=/work/llvm-project/build/bin/ python3 csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_tune.py -i aiter/configs/a4w4_blockscale_untuned_gemm.csv -o aiter/configs/a4w4_blockscale_tuned_gemm.csv`
You can find the results of the tuning in `aiter/configs/a4w4_blockscale_tuned_gemm.csv`, like this:
    |**gfx**  |**cu_num**|**M**|**N**|**K**|**libtype**|**kernelId**|**splitK**|**us**|**kernelName**|**tflops**|**bw**|**errRatio**|
    |---------|----------|-----|-----|-----|-----------|------------|----------|------|--------------|----------|------|------------|
    |gfx950   |256       |128  |1536 |7168 |ck         |23          |0         |xx.xx |CK kernel name|...       |...   |0.01        |
    |gfx950   |256       |128  |512  |7168 |triton     |0           |0         |xx.xx |triton        |...       |...   |0.01        |

    The rows above illustrate legacy rows, not measured winners. This tuner supports gfx950. `gfx` identifies the GPU architecture, and `cu_num` is the number of compute units. New results also include `bpreshuffle` and `scale_preshuffle` in the shape key. `libtype` names the winning backend (`ck`, `asm` or `triton`): `gemm_a4w4` selects the raw or preshuffled Triton kernel according to `bpreshuffle` and keeps CK/ASM rows on `kernelName`. Tables without `libtype` still dispatch on `kernelName` alone. Missing or blank layout columns mean `1,1`, preserving the historical contract. CK code generation excludes Triton and ASM rows and retains CK winners from each layout.

4. Build tuned kernels and test:
Test the performance, modify the test instance in `op_tests/test_gemm_a4w4_blockscale.py` and run it, please wait a few minutes as it will build gemm_a4w4_blockscale tuned kernels in `aiter/configs/a4w4_blockscale_tuned_gemm.csv` via jit:
`GEMM_A4W4_BLOCKWISE_HIP_CLANG_PATH=/work/llvm-project/build/bin/ python3 op_tests/test_gemm_a4w4_blockscale.py`
If you have built gemm_a4w4 kernels before tuning new GEMM shapes, please add `AITER_REBUILD=1` before your test cmd, such as `AITER_REBUILD=1 python3 op_tests/test_gemm_a4w4_blockscale.py`. It will rebuild kernels from `AITER_CONFIG_GEMM_A4W4`, the default one will be results merged from `aiter/configs/a4w4_blockscale_tuned_gemm.csv` and tuned fmoe csv under `aiter/configs/model_configs/xx_a4w4_blockscale_tuned_gemm_xx.csv`, the merged result is store in `/tmp/aiter_configs/a4w4_blockscale_tuned_gemm.csv`

## More Options

**Note**: All commands require setting `GEMM_A4W4_BLOCKWISE_HIP_CLANG_PATH=/work/llvm-project/build/bin/` environment variable.

### Output Configuration

#### `-o2, --profile_file`
- **Type**: String
- **Default**: `""` (empty string)
- **Description**: Optional output file to store **all** tuning results (not just the best ones). Useful for profiling and analyzing all kernel candidates.

**Example**:
```bash
--profile_file aiter/configs/profile_a4w4_blockscale_all.csv
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

#### `--libtype`
- **Type**: String
- **Default**: `all`
- **Choices**: `ck`, `asm`, `triton`, `all`
- **Description**: Backends timed for each shape and input layout. `all` (the default) compares compatible CK, ASM and Triton candidates. CK requires preshuffled B. ASM candidates are filtered by B layout. Triton uses its raw kernel for raw B (`M,N,K > 0`, `K % 32 == 0`) and its preshuffled kernel for shuffled B (`M >= 32`, `N % 16 == 0`, `K % 256 == 0`). The winner is written to `libtype` and read by `gemm_a4w4`.

**Examples**:
```bash
--libtype all      # default: CK + asm + triton
--libtype triton   # triton candidates only
```

The Triton candidate uses the existing per-shape Triton configuration; this option
compares backends rather than searching new Triton tile parameters. Runtime
dispatch requires contiguous packed operands, scales matching the declared layout,
and bf16 output, without bias, activation preshuffling or non-default scaling.
Other contracts use the existing ASM fallback.

#### `--bpreshuffle` and `--scale_preshuffle`

Each option accepts `0` (raw), `1` (shuffled) or `both`. Without an override,
the input CSV supplies that flag; missing or blank flags default to `1`.
`both` expands each input shape into separate layout candidates. New CSV rows
are keyed by `gfx,cu_num,M,N,K,bpreshuffle,scale_preshuffle`, so tuning one layout
does not replace another layout's winner.

| B layout | Scale layout | Candidate behavior |
|---|---|---|
| `1` | `1` | CK, preshuffled ASM and preshuffled Triton consume the inputs directly |
| `0` | `0` | Plain Triton consumes the inputs directly; noB ASM shuffles scales |
| `0` | `1` | noB ASM consumes the inputs directly; plain Triton unshuffles scales |
| `1` | `0` | CK, preshuffled ASM and preshuffled Triton shuffle scales |

All scale conversions are included in candidate timing and production dispatch.
No pointer cache is used. B is never silently reshuffled. `scale_preshuffle=False`
is currently supported only on gfx950. Raw e8m0 scales have shape `[M or N, K/32]`;
shuffled scales use `shuffle_scale`'s padded 32x4 layout.

```python
# Raw packed B and raw e8m0 scales; a matching Triton CSV row calls plain Triton.
y = aiter.gemm_a4w4(a, b, a_scale, b_scale,
                   bpreshuffle=False, scale_preshuffle=False)
```

```bash
# Compare all four input layouts independently.
python3 csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_tune.py \
  -i shapes.csv -o layouts.csv --libtype all \
  --bpreshuffle both --scale_preshuffle both
python3 csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_tune.py \
  --run_config layouts.csv
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
python3 csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_tune.py \
  --run_config aiter/configs/a4w4_blockscale_tuned_gemm.csv

# benchmark default kernels using shapes from -i
python3 csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_tune.py \
  -i aiter/configs/a4w4_blockscale_untuned_gemm.csv --run_config
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
If you use flag `PREBUILD_KERNELS=1` when you install aiter, it will build gemm a4w4 kernels in tuned gemm csv by default. If you want to use the new result of gemm_a4w4_blockscale_tune, please remove `build` and `*.so` in `aiter/jit` first, then re-install aiter after finishing tune. This can take a lot of time and is not recommended.
