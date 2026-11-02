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

Use `--mxfp4-flydsl` for gfx950 A4W4 MoE tuning. Choose a search mode with
`--mxfp4-search-mode`:

- `staged` (default): prune GEMM1 by `M_est = ceil(token * topk / expert)`,
  screen both stages, and validate shortlisted pairs end to end.
  `--mxfp4-screen-topk` keeps the fastest candidates per group (default 2).
- `full`: validate all compatible GEMM1/GEMM2 pairs without M_est pruning.

Staged search validates all pruned pairs when `M_est < 4`. Otherwise, a group
falls back to a full search of its pruned pairs if no shortlist is available
or every shortlisted pair fails. The fastest pair passing `--errRatio` wins.
Each shape runs in a fresh process, including with `--mp 1`.

Input (`-i`) and output (`-o`) must refer to different files. Existing configured
shapes are skipped; use `--all` to retune them or choose a new output file.

```bash
python3 csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py \
  -i <untuned.csv> -o <tuned.csv> --mxfp4-flydsl

python3 csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py \
  -i <untuned.csv> -o <tuned.csv> --mxfp4-flydsl --mxfp4-search-mode full --all
```

With `--fast-scan`, an omitted mode scans every pruned pair; `full` scans all
pairs. `--fast-scan` cannot be combined with explicit `staged` search.

Staged screening can miss a faster pair. Use `full` to check the selected
kernel, and keep activation limits and stage2 settings consistent across
tuning, AOT and production; see [MXFP4 settings](../../aiter/aot/flydsl/README.md#mxfp4-activation-and-stage2-settings).

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

This option applies to ordinary fMoE tuning.

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

Use `--fast-scan` with `--mxfp4-flydsl` to scan coupled G1/G2 candidates
(pruned by default, or all with `--mxfp4-search-mode full`) using the
profiler-summed active GPU-kernel time, excluding launch
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
- **Default**: `1800`
- **Description**: Per-task watchdog for ordinary tuners. MXFP4 uses a
  best-effort SIGALRM timeout per operation; a permanent GPU hang may wait
  indefinitely.

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
