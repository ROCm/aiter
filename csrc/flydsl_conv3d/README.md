# FlyDSL conv3d (BF16) Tile Tune

Offline tile tuner for `flydsl_conv_implicit`, the implicit-GEMM convolution. It
reads shapes from an untuned CSV, sweeps the launch configs
`aiter/ops/flydsl/conv3d_policy.py` enumerates for each one, and writes the
winner to a checked-in tuned CSV that `conv_kernels._lookup_tuned_tile` reads
at runtime. A shape with no tuned row falls back to the heuristic tile ladder,
so tuning is an optimization rather than a prerequisite.

Single backend, unlike the GEMM tuners: there is no asm/CK/triton alternative
for this kernel, so there is no `--libtype` and no `gemm_tuner.py`-style
subprocess wrapper -- that one exists to retry hipBLASLt's GPU faults, which are
not fixable locally. Here a crash is this repo's own bug and should surface.
The launch config is stored as five explicit integer columns rather than a
`solidx`, so reordering the candidate list cannot silently invalidate a
checked-in CSV.

1. Install aiter:

```bash
cd $aiter_path
python3 setup.py develop
```

2. Add conv shapes to a per-model untuned table under
   `aiter/configs/model_configs/`. The header is the 20-column problem key --
   the same columns `conv_kernels.TUNED_KEY_COLUMNS` looks up on:

    |**N**|**C**|**D**|**H**|**W**|**K**|**kT**|**kH**|**kW**|**stride_d**|**stride_h**|**stride_w**|**pad_d**|**pad_h**|**pad_w**|**dil_d**|**dil_h**|**dil_w**|**groups**|**bias**|
    |-----|-----|-----|-----|-----|-----|------|------|------|------------|------------|------------|---------|---------|---------|---------|---------|---------|----------|--------|
    |1    |3    |1    |1024 |1024 |96   |1     |3     |3     |1           |1           |1           |0        |1        |1        |1        |1        |1        |1         |True    |

   Tables are per input resolution, because a VAE's shapes are derived from it:
   `qwenimage_vae_1024x1024`, `qwenimage_vae_1328x1328`, `wan21_vae_368x544`,
   `wan21_vae_480x832`.

   Pass `-i` and `-o` explicitly. The defaults are the canonical pair, which
   ships header-only, so a run without them finds no shapes and exits rather
   than tuning a table you did not mean.

3. Tune into the matching per-model tuned table:

```bash
python3 csrc/flydsl_conv3d/conv3d_tune.py \
  -i aiter/configs/model_configs/qwenimage_vae_1024x1024_bf16_untuned_conv3d.csv \
  -o aiter/configs/model_configs/qwenimage_vae_1024x1024_bf16_tuned_conv3d.csv
```

   Write winners into the per-model file, never into
   `aiter/configs/bf16_tuned_conv3d.csv`. That one and its untuned sibling ship
   header-only and back the merge `AITER_CONFIG_CONV3D_BF16` performs:

   - `bf16_tuned_conv3d.csv` -- merge anchor, and the path `get_config_file`
     returns as-is when no per-model table is present, so it has to stay a
     readable csv.
   - `bf16_untuned_conv3d.csv` -- supplies the duplicate-detection keys, so two
     tables claiming the same shape fail the merge instead of silently
     coexisting. Its columns must stay equal to the tuner's `SHAPE_KEYS`.

   Results carry the tuning device plus the chosen config:

    |**gfx**|**cu_num**|*(the 20 key columns)*|**tile_m**|**tile_n**|**wave_m**|**wave_n**|**wgm**|**splitK**|**us**|**kernelName**|**err_ratio**|**tflops**|**bw**|
    |-------|----------|----------------------|----------|----------|----------|----------|-------|----------|------|--------------|-------------|----------|------|
    |gfx950 |256       |...                   |96        |96        |2         |3         |1      |1         |137.3024|conv3d_implicit_t96x96_w2x3_g1|0.0|39.59|1512.16|

4. Check the result and its coverage:

```bash
# Correctness and per-shape timings
python3 op_tests/test_flydsl_conv_implicit.py

# No two config tables claim the same shape
python3 -m pytest op_tests/tuning_tests/test_config_shape_collision.py
```

   The AOT pass (`aiter/aot/flydsl/conv.py`, run from `setup.py` at build time)
   compiles exactly what the tuned CSV holds, so new rows widen AOT coverage and
   removed rows narrow it. Its compile keys are derived separately from the
   runtime's, and a disagreement is silent -- the shape just falls back to JIT.
   Nothing checks that automatically; `aiter.aot.flydsl.common.run_only_env()`
   makes FlyDSL raise on a JIT rather than fall back, which is how to verify a
   row by hand after changing the padding, channel-padding or split-K rules.

   To see which tile a given conv actually picked, run with
   `AITER_LOG_TUNED_CONFIG=1`. A shape that falls back to the heuristic says so
   without the switch, and names the devices it *was* tuned for if the table has
   the shape under a different `gfx`/`cu_num`.

## Tuner-Specific Options

### `--max_configs`
- **Type**: int
- **Default**: 96
- **Description**: Cap on enumerated candidates per shape. The kernel's own
  candidate table and heuristic ladder are unioned in on top of the cap, so the
  tuned pick can never come out worse than the shipped default.

## Common Options

### `--run_config [TUNED_CSV]`
Benchmark the production operator only, no tuning. Each shape is read
`RUN_CONFIG_REPS` (3) times and the fastest is kept, because the compare gate
decides on 3%. Pass `-i` as well: the run still loads the untuned table first,
and the default one is empty (see step 2).

```bash
python3 csrc/flydsl_conv3d/conv3d_tune.py \
  -i aiter/configs/model_configs/wan21_vae_480x832_bf16_untuned_conv3d.csv \
  --run_config aiter/configs/model_configs/wan21_vae_480x832_bf16_tuned_conv3d.csv
```

### `--compare` / `--update_improved`
Benchmark before and after tuning and print the comparison. With
`--update_improved`, only shapes improved by at least `--min_improvement_pct`
(default 3%) are written back. The tuner holds the GPU busy for a couple of
seconds before each half so both are measured at settled clocks -- without that
the same config on the same shape has been observed to differ by 2.1x.

### `--mp`
Number of GPUs for parallel tuning. Default: all available.

### `--errRatio`
Tolerable error ratio against the `torch.nn.functional.conv3d` reference
(default 0.05). The reference is bf16 rather than fp32 on purpose: the tuner
needs to catch a config that computes the wrong thing, not to measure bf16
rounding.

### `--timeout`
Per-task watchdog in seconds (default 1800). A worker killed by a GPU
memory-access fault leaves its task unresolvable; the watchdog drops it and
restarts the pool instead of hanging the run.

### `-o2, --profile_file`
Save every candidate's result, not just the winner.

### `-v, --verbose`
Detailed logging.
