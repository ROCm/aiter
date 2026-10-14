# CK GEMM MoE 2-Stages Codegen or asm 1-stage Tune

1. Install aiter:
`cd $aiter_path`
`python3 setup.py develop`

2. Add MoE shapes in `aiter/configs/untuned_fmoe.csv`
    |**token**|**model_dim**|**inter_dim**|**expert**|**topk**|**act_type**|**dtype**|**q_dtype_a**|**q_dtype_w**|**q_type**|**use_g1u1**|**doweight_stage1**|
    |---------|-------------|-------------|----------|--------|------------|---------|-------------|-------------|----------|------------|-------------------|
    |1024     |4096         |14336        |8         |2       |ActivationType.Silu|dtypes.bf16|dtypes.fp8|dtypes.fp8|QuantType.per_Token|True|True|


3. Start tuning:
Run the following cmd to start tuning, please wait a few minutes as it will build moe 2-stages kernels via jit:
`python3 csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py -i aiter/configs/untuned_fmoe.csv -o aiter/configs/tuned_fmoe.csv`
You can find the results of this tuning in `aiter/configs/tuned_fmoe.csv`, like this:
    |**cu_num**|**token**|**model_dim**|**inter_dim**|**expert**|**topk**|**act_type**|**dtype**|**q_dtype_a**|**q_dtype_w**|**q_type**|**use_g1u1**|**doweight_stage1**|**block_m**|**ksplit**|**us1**|**kernelName1**|**err1**|**us2**|**kernelName2**|**err2**|**us**|**run_1stage**|**tflops**|**bw**|
    |----------|---------|-------------|-------------|----------|--------|------------|---------|-------------|-------------|----------|------------|-------------------|-----------|----------|-------|---------------|--------|-------|---------------|--------|------|--------------|----------|------|
    |80        |1024     |4096         |14336        |8         |2       |ActivationType.Silu|dtypes.bf16|dtypes.fp8|dtypes.fp8|QuantType.per_Token|True|True|64|0|45.23|kernel_stage1|0.5%|38.67|kernel_stage2|0.3%|83.90|0|125.4|89.5|

    `cu_num` means the number of compute units, and it is used to distinguish between graphics.
    `run_1stage` indicates whether to run fused 1-stage kernel (1) or 2-stages kernels (0).

4. Build tuned kernels and test:
Test the performance, modify the test instance in `op_tests/test_moe.py` or `python3 op_tests/test_moe_2stage.py` and run it, please wait a few minutes as it will build moe tuned kernels in `aiter/configs/tuned_fmoe.csv` via jit:
`python3 op_tests/test_moe.py` or `python3 op_tests/test_moe_2stage.py`
If you have built moe kernels before tuning new MoE shapes, please add `AITER_REBUILD=1` before your test cmd, such as `AITER_REBUILD=1 python3 op_tests/test_moe.py`. It will rebuild kernels from `AITER_CONFIG_FMOE`, the default one will be results merged from `aiter/configs/tuned_fmoe.csv` and tuned fmoe csv under `aiter/configs/model_configs/xx_tuned_fmoe_xx.csv`, the merged result is stored in `/tmp/aiter_configs/tuned_fmoe.csv`.

## More Options

### Tuning Scope

#### `--mxfp4-flydsl` and `--mxfp4-search-mode`

The dedicated gfx950 A4W4 MXMOE tuner prunes legal GEMM1 candidates by default
using `M_est = ceil(token * topk / expert)`, before generating routing data.
It keeps overlapping BM families: BM16 for `M_est < 16`, BM32 for
`4 <= M_est <= 128`, BM64 for `M_est >= 16`, and BM128 for `M_est >= 64`.
Above `M_est = 32`, BM32 keeps only 4-wave, `k_wave=1`, non-NT candidates.
Above `M_est = 64`, BM64 keeps only non-NT candidates. All legal BN,
`xcd_swizzle`, and BM16 `prefetch_hidden` choices remain available.

`--mxfp4-search-mode {prune,full}` selects the GEMM1 search mode. It defaults
to `prune`; `full` enumerates every statically supported GEMM1 candidate:

```bash
python3 csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py \
  -i <untuned.csv> -o <tuned.csv> --mxfp4-flydsl --mxfp4-search-mode prune

python3 csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py \
  -i <untuned.csv> -o <tuned.csv> --mxfp4-flydsl --mxfp4-search-mode full
```

Switching the mode alone does not re-tune shapes already written to the output
CSV: `pre_process` skips them by shape key, and the search mode does not change
that dedup rule. To re-check tuned shapes with full search, either add `--all`
to re-tune and overwrite them, or point `-o` at a new file to keep the pruned
results for comparison:

```bash
python3 csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py \
  -i <untuned.csv> -o <tuned.csv> --mxfp4-flydsl --mxfp4-search-mode full --all

python3 csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py \
  -i <untuned.csv> -o <tuned_full.csv> --mxfp4-flydsl --mxfp4-search-mode full
```

An explicit `--mxfp4-search-mode` requires `--mxfp4-flydsl` and cannot be
combined with `--grouped-gemm`. Both serial and multi-GPU shape workers use the same choice.
When pruning removes candidates, the tuner prints one summary containing the
pid, the shape fields that identify the row, `M_est`, the legal G1 count, and
the retained G1 count, even without `--verbose`. The pid and shape fields keep
the line attributable when `--mp` workers interleave their stdout.
Full search and shapes with no removed candidates omit that summary.

Pruning reduces candidate evaluation work and is expected to shorten tuning
wall time. The actual wall time of `full` and `prune` has not been compared, so
no measured speedup is reported. The performance of the selected kernels and
winner retention on unseen shapes require separate validation; use full search
to investigate those shapes. GEMM2 enumeration and the same-BM pairing rule are
unchanged, but dropping a GEMM1 BM family also drops every pipeline pair under
it.

See [the pruning method and tuner changes](../../docs/a4w4_mxmoe_gemm1_pruning.md)
for a worked example and static validation results.

#### `--last`
- **Type**: Flag (boolean)
- **Default**: `False`
- **Description**: Only tune the last kernel in the CSV file. Useful for quickly testing newly added shapes.

**Example**:
```bash
--last
```

### Output Configuration

#### `-o2, --profile_file`
- **Type**: String
- **Default**: `""` (empty string)
- **Description**: Optional output file to store **all** tuning results (not just the best ones). Useful for profiling and analyzing all kernel candidates.

**Example**:
```bash
--profile_file aiter/configs/profile_fmoe_all.csv
```

### Tuning Configuration

#### `--errRatio`
- **Type**: Float
- **Default**: `0.5` (50%)
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
--mp 8
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
#### `--sort`
- **Type**: Boolean (True/False)
- **Default**: `False` 
- **Description**: Sort the output file according to the key columns

**Example**:
```bash
--sort True
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
python3 csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py \
  --run_config aiter/configs/tuned_fmoe.csv

# benchmark default kernels using shapes from -i
python3 csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py \
  -i aiter/configs/untuned_fmoe.csv --run_config
```

#### Fast MXFP4 FlyDSL scan

Use `--fast-scan` with `--mxfp4-flydsl` to scan each legal coupled G1/G2
candidate using the profiler-summed active GPU-kernel time, excluding launch
overhead and idle gaps. The scan prepares one input fixture per shape and
builds one shared torch reference. Candidates that exceed
`--errRatio`, return a non-finite error, or fail to launch are rejected
individually, and the remaining candidates continue. A shared
`--fast-scan-warmup-accuracy-checks` control (default 1) runs an untimed launch
and validates its output before the coarse stage. The remaining candidates are
profiled for `--fast-scan-iters` launches (default 10, for 11 coarse launches
including warm-up; the value must be at least 2). The five lowest-latency
`--fast-scan-finalists` candidates (default 5), always including the coarse
global fastest candidate, plus the most accurate candidate not already in that
performance set are revalidated and retimed for `--fast-scan-final-iters`
profiled launches (default 100; the value must be at least 2) without another
warm-up. If a finalist fails final accuracy validation, the next-lowest-latency
unselected coarse survivor is retimed as a backfill until the target number of
successful final retimes is reached or the candidate pool is exhausted. The
accuracy-bucket ordering is diagnostic only: it ranks the printed summary but
never reorders the latency-ordered backfill queue. The accuracy promotion
itself is not diagnostic, and picks the most accurate unselected coarse
survivor by the unrounded error, with latency breaking an exact tie. If the
globally most accurate candidate is already a performance finalist, the
next-most-accurate unselected candidate is promoted instead, so one independent
accuracy candidate always reaches the final stage when one is available. Every
requested profiler sample counts toward the active-kernel average except the
warm-up iterations and, once the sample count exceeds 30, the IQR outliers that
`get_trace_perf` drops. With `AITER_MXFP4_TUNE_SWIGLU_LIMIT` set, the limit is
applied to the shared reference, to every candidate launch, and to the parallel
precompile, so the scan neither compares different clamps nor JITs during
scoring. The final measured output is checked after timing as an additional
stability guard. Final-winner selection finds the globally fastest finalist,
includes every finalist within 1% of that latency, and selects the lowest
worst-case cosine error in that global performance-equivalent set. Latency
breaks an exact accuracy tie. A shape is marked failed only when no candidate
survives its accuracy and launch checks; otherwise the best valid final retime
is emitted.

The funnel checks kernel-level correctness at every stage. Still validate the
selected rows through the production operator before using or committing the
table, because that also covers runtime dispatch and integration:

```bash
python3 csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py \
  -i aiter/configs/model_configs/model_fp4_untuned_fmoe.csv \
  -o aiter/configs/model_configs/model_fp4_tuned_fmoe.csv \
  --mxfp4-flydsl --fast-scan --mp 1 --batch 1

python3 csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py \
  --mxfp4-flydsl \
  --run_config aiter/configs/model_configs/model_fp4_tuned_fmoe.csv
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
- This tuner supports both 1-stage fused MoE kernels and 2-stages MoE kernels (stage1 and stage2)
- The tuner will automatically select the best kernel configuration based on performance
- Only G1U1 (gate-up fused) MoE configurations are currently supported for tuning
- Supported quantization types include: per_Token, per_1x128 (blockscale), per_1x32 (MXFP4, gfx950 only)
- If you use flag `PREBUILD_KERNELS=1` when you install aiter, it will build moe kernels in tuned csv by default. If you want to use the new result of moe tuning, please remove `build` and `*.so` in `aiter/jit` first, then re-install aiter after finishing tune. This can take a lot of time and is not recommended.

