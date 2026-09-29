# CK GEMM A8W8 Blockscale Tune

1. Install aiter:
`cd $aiter_path`
`python3 setup.py develop`

2. Add GEMM shapes in `aiter/configs/a8w8_blockscale_untuned_gemm.csv`
    |**M**|**N**|**K**|
    |-----|-----|-----|
    |128  |1536 |7168 |

3. Start tuning:
Run the following cmd to start tuning, please wait a few minutes as it will build gemm_a8w8_blockscale_tune via jit:
`python3 csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py -i aiter/configs/a8w8_blockscale_untuned_gemm.csv -o aiter/configs/a8w8_blockscale_tuned_gemm.csv --libtype both`
libtype can be `ck`, `cktile` or `both`. We recommend to tune together by setting `--libtype both` to get both ck legacy and tile implementations, then choose the best one, this will take more time but help to get better performance. You can find the results of the tuning in `aiter/configs/a8w8_blockscale_tuned_gemm.csv`, like this:
    |**gfx**  |**cu_num**|**M**|**N**|**K**|**kernelId**|**splitK**|**us**|**kernelName**|**tflops**|**bw**|**errRatio**|
    |---------|----------|-----|-----|-----|------------|----------|------|--------------|----------|------|------------|
    |gfx942   |80        |128  |1536 |7168 |23          |0         |32.99 |xxxxxxxx      |125.4     |89.5  |0.01        |

    `gfx` identifies the GPU architecture (e.g. `gfx942`, `gfx950`). `cu_num` is the number of compute units and distinguishes partitioned or binned variants of the same architecture (e.g. MI308X vs MI300X both use `gfx942`).

4. Build tuned kernels and test:
Test the performance, modify the test instance in `op_tests/test_gemm_a8w8_blockscale.py` and run it, please wait a few minutes as it will build gemm_a8w8_blockscale tuned kernels in `aiter/configs/a8w8_blockscale_tuned_gemm.csv` via jit:
`python3 op_tests/test_gemm_a8w8_blockscale.py`
If you have built gemm_a8w8 kernels before tuning new GEMM shapes, please add `AITER_REBUILD=1` before your test cmd, such as `AITER_REBUILD=1 python3 op_tests/test_gemm_a8w8_blockscale.py`. It will rebuild kernels from `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE`, the default one will be results merged from `aiter/configs/a8w8_blockscale_tuned_gemm.csv` and tuned fmoe csv under `aiter/configs/model_configs/xx_a8w8_blockscale_tuned_gemm_xx.csv`, the merged result is store in `/tmp/aiter_configs/a8w8_blockscale_tuned_gemm.csv`.

## More Options

### FlyDSL FP8 blockscale (gfx950)

The existing tuner also accepts `--libtype flydsl`. Use `--libtype all` to race
FlyDSL against the existing supported CK, CKTile, ASM and Opus candidates;
`--libtype both` retains its original CK/CKTile-only meaning. Add `--preshuffle`
for the B-preshuffled operator and select its output CSV with `-o`.

- The public `gemm_a8w8_blockscale` and `gemm_a8w8_blockscale_bpreshuffle` APIs
  keep their signatures and default backends. A winning row with
  `libtype=flydsl` and a `flydsl_blockscale_8w_...` name selects the new backend.
  No FlyDSL default or tuned rows are installed by this integration.
- The candidate table fixes the 256x256x128 tile, half-M pipeline and raw DMA,
  with one candidate per B layout. The surviving IDs/names stay stable:
  ID 2 ends in `ps0_sm1_tdma0`; ID 6 ends in `ps1_sm1_tdma0`.
  `splitK` remains zero. Full-M and tiled-DMA names are no longer accepted;
  retune external CSVs selecting those removed modes rather than aliasing them
  to a different implementation.
- Supported calls use gfx950, FP8 E4M3FN operands, FP32 block scales and BF16
  output, with positive M/N, K >= 256, K divisible by 256, and N divisible by
  8 (plain B) or 16 (preshuffled B). LDS, workgroup-local signed-i32 spans and
  unrebased scale/grid limits are checked before launch. A/B/C descriptor bases
  are rebased per workgroup with i64 arithmetic, so whole matrices may exceed
  4 GiB; matrix arguments keep their two-dimensional launch ABI. M/N tile tails
  are supported. When a tuned row selects FlyDSL,
  unsupported output types/shapes/devices or non-FP32 scales raise an assertion.
  Import failures also raise `AssertionError`, preserving the original exception
  as the cause. No CK fallback is taken for a selected FlyDSL row; the original
  no-config default path is unchanged. Other compile/runtime errors propagate.
- Plain B uses row-major `x_scale[M,K/128]`; its transpose cost is included in
  tuning. For FlyDSL, a prepared scale tensor with the same shape may set
  `is_transposed=True` to skip that transpose; the flag is trusted without
  layout inference. Set it on the final tensor (views/clones do not inherit
  Python attributes). Other backends' scale contracts are unchanged.
  Preshuffled B consumes the existing `(16,16)` weight shuffle and
  column-major scale storage, either packed back into shape `[M,K/128]` or a
  strided column-major view. Both use `w_scale[ceil(N/128),K/128]`. The
  preshuffle API continues to honor a caller-supplied `out`.
- The kernel is ported from pyhip commit
  `a3a94c5a34fc525c118649418b76221cd6d91579`, preserving its half-M blockscale
  compute and raw-DMA scheduling. The compiler accepts only `TILE_M`, `TILE_N`,
  `TILE_K`, `N`, `K`, `pid_swizzle`, `permlane_epilogue`, and `preshuffle_b`.
  `with_scale`, `split_m`, and `useTileDMA` parameters and alternative branches
  are removed; scales/half-M are mandatory and tiled DMA is unsupported.
  Aiter does not acquire a pyhip runtime dependency. The tensor adapter and tune
  table remain separate. The existing gfx1250 MXFP8_128 path is unaffected.

Use `-o2` to retain every candidate result, and `--run_config` with the resulting
CSV (plus `--preshuffle` for that layout) to validate the production dispatch.
Prefer scratch output CSVs for experiments rather than overwriting existing
model configurations. Before promoting winners, check for duplicate shape keys
across the canonical and model-specific config files.

The dedicated correctness/performance sweep is
[op_tests/test_gemm_a8w8_blockscale_flydsl.py](../../op_tests/test_gemm_a8w8_blockscale_flydsl.py);
it covers both public APIs, signed/random data, M/N tails, and packed/strided
scales. CPU routing/tuner regressions are in
[op_tests/tuning_tests/test_flydsl_blockscale.py](../../op_tests/tuning_tests/test_flydsl_blockscale.py).

#### Recorded performance

See [MI355X backend comparison and FlyDSL regression calibration (2026-09-28)](perf_gfx950_20260928.md)
for same-GPU CK/CKTile/ASM/Triton/FlyDSL results, graph and event timing scopes,
historical screenshot calibration, source fingerprints, and validation limits.
These measurements do not install or change tuned dispatch configurations.

The [model Q/KV projection comparison (2026-09-28)](perf_model_qkv_gfx950_20260928.md)
covers Qwen3.8, Qwen3.5, Kimi-K3 and DeepSeek V4 at 8192–65536 local tokens and
TP 1/2/4/8: 80 unique shapes, both B layouts, 688 validated backend results,
per-model summaries and large-output availability limits. Model quantization
caveats and the correct gate/KV-replication/low-rank TP rules are included.

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
python3 csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py \
  --run_config aiter/configs/a8w8_blockscale_tuned_gemm.csv

# benchmark default kernels using shapes from -i
python3 csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py \
  -i aiter/configs/a8w8_blockscale_untuned_gemm.csv --run_config
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
If you use flag `PREBUILD_KERNELS=1` when you install aiter, it will build gemm a8w8 blockscale kernels in tuned gemm csv by default. If you want to use the new result of gemm_a8w8_blockscale_tune, please remove `build` and `*.so` in `aiter/jit` first, then re-install aiter after finishing tune. This can take a lot of time and is not recommended.
