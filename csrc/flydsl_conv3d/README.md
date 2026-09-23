# FlyDSL conv3d (BF16) Tile Tune

Offline tile tuner for `flydsl_conv_implicit`, the implicit-GEMM convolution. It
reads shapes from an untuned CSV, sweeps the launch configs
`aiter/ops/flydsl/conv3d_policy.py` enumerates for each one, and writes the
winner to a checked-in tuned CSV that `conv_kernels._lookup_tuned_tile` reads
at runtime. A shape with no tuned row of its own borrows the nearest usable
tuned resolution of the same layer, and falls back to the heuristic tile ladder
when there is not one, so tuning is an optimization rather than a prerequisite.

One layout only: rows are tuned NCDHW in and out, and an NDHWC-output call
reuses that tile even though `out_ndhwc` is a compile-time parameter that
changes the epilogue (it gives up the vectorised store). Measured rather than
assumed -- re-sweeping the full candidate set of 8 Qwen-Image rows under
`output_layout="NDHWC"` on gfx950 left the tuned tile the winner on 5 of them
and within 0.2% on a 6th. It drifts only where the GEMM's K axis is tiny and
the epilogue is most of the kernel: `3->96` (K=27) came out 4.0% off the
NDHWC-best and `16->384` (K=144) 8.3%, the latter 19.8us against 18.3us. That
is why there is one row per shape rather than one per layout; a layout column
would double every table for those two cases.

Single backend, unlike the GEMM tuners: there is no asm/CK/triton alternative
for this kernel, so there is no `--libtype` flag and no `gemm_tuner.py`-style
subprocess wrapper -- that one exists to retry hipBLASLt's GPU faults, which are
not fixable locally. Here a crash is this repo's own bug and should surface.
(The tuned CSV does carry a `libtype` column, as the GEMM tables do; this tuner
always writes `flydsl` into it.)
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

   Tables are per model: `qwenimage_vae`, `wan21_vae`. A VAE's shapes are derived
   from its input resolution, so a table holds one block of rows per resolution
   it was tuned at -- Qwen-Image covers 1024² and 1328², Wan2.1 368×544 and
   480×832.

   Unlisted resolutions borrow the nearest same-layer tuned tile (npq within
   4x, and only while that tile still fills the device at the npq being
   served), then fall back to `_pick_tile`. `AITER_CONV3D_DYN_HW` (on by
   default) shares one *artifact* across resolutions of a layer; tile is still
   per row.

   Borrowing is a fallback, not a substitute for a row. Measured on gfx950 over
   93 off-table resolutions of the two VAEs: with the fills-device bar in place
   it runs 2.6% faster than the heuristic on the Qwen-Image layers and 3.6%
   faster on the Wan2.1 ones, worst case 1.22x slower. Without that bar the
   same sweep was a 1.5% net *loss* on Qwen-Image with a worst case of 2.02x,
   all of it on layers whose M had shrunk enough that the borrowed tile no
   longer filled the device. The residual cases are not separable by npq
   distance, so tune a resolution that matters rather than leaving it to this.

   Pass `-i` and `-o` explicitly. The defaults are the canonical pair, which
   ships header-only, so a run without them finds no shapes and exits rather
   than tuning a table you did not mean.

3. Tune into the matching per-model tuned table:

```bash
python3 csrc/flydsl_conv3d/conv3d_tune.py \
  -i aiter/configs/model_configs/qwenimage_vae_bf16_untuned_conv3d.csv \
  -o aiter/configs/model_configs/qwenimage_vae_bf16_tuned_conv3d.csv
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

    |**gfx**|**cu_num**|*(the 20 key columns)*|**libtype**|**tile_m**|**tile_n**|**wave_m**|**wave_n**|**wgm**|**splitK**|**us**|**kernelName**|**err_ratio**|**tflops**|**bw**|
    |-------|----------|----------------------|-----------|----------|----------|----------|----------|-------|----------|------|--------------|-------------|----------|------|
    |gfx950 |256       |...                   |flydsl     |96        |96        |2         |3         |1      |1         |137.3024|conv3d_implicit_t96x96_w2x3_g1|0.0|39.59|1512.16|

   `us`, `tflops` and `bw` are end-to-end: the entry point runs the
   NCDHW->NDHWC pre-transpose and the weight repack before the kernel, and the
   tuner times it the way the model calls it. That is a constant per shape, so
   the ranking is unaffected, but the three columns are a floor rather than a
   kernel figure and are not comparable with another implementation's
   kernel-only numbers.

   `splitK` is **not** swept: `_resolve_splitk` derives it per candidate and the
   tuner pins that value so the column records what ran. Every row of both VAE
   tables is `splitK=1`, and structurally so rather than by default -- a split
   only buys occupancy where the M/N grid alone cannot fill the device, and 75
   of the 76 rows are already at or above the heuristic's own 3/4*CU block bar,
   where a split can do nothing but add an fp32 staging buffer and an atomic
   reduce. The exception is Wan's `conv_out` at 368x544 (98 blocks), which
   `_resolve_splitk` declines on `npq < 4096`.

   Note that split-K cannot currently be combined with a tile that leaves a
   tail: the masked element is dropped by routing the store to the OOB
   sentinel, which needs a buffer descriptor, and the split-K atomic has none.
   `make_output_scatter_plan` asserts on it. The heuristic cannot reach that
   combination (it requires `npq % tile_m == 0` and `kg % tile_n == 0` before
   splitting), so it is only reachable through an explicit `splitk=` or a
   hand-written row.

   `libtype` names the implementation the rest of the row configures, as it does
   in the GEMM tables. It is a result rather than part of the key: where a tuner
   has several backends to choose between, it records whose config won, so one
   shape still owns one row. FlyDSL is the only conv3d backend
   today, so this tuner always writes `flydsl`; the runtime and the AOT pass skip
   rows naming anything else, and read a table without the column -- or with the
   cell empty -- as all-FlyDSL.

4. Check the result and its coverage:

```bash
# Correctness and per-shape timings
python3 op_tests/test_flydsl_conv_implicit.py

# No two config tables claim the same shape
python3 -m pytest op_tests/tuning_tests/test_config_shape_collision.py
```

   The AOT pass (`aiter/aot/flydsl/conv.py`, run from `setup.py` at build time)
   compiles exactly what the tuned CSV holds, so new rows widen AOT coverage and
   removed rows narrow it. `AITER_CONV3D_DYN_HW` (on by default) widens what
   each of those artifacts then serves -- one covers a layer at any resolution
   instead of the one it was compiled for -- but it is part of the compile key,
   so a build and the runtime reading its cache have to agree on it. Tile lookup
   is unaffected: it still needs an exact 20-column match, and a resolution
   without one borrows the nearest tuned row of the same layer rather than
   getting its own tuned tile -- it just no longer pays a JIT for the artifact.
   Both the AOT pass and the runtime build the compile key through
   `conv_kernels._implicit_param_from_problem`, so channel padding and field
   order cannot drift between them. `splitK` is the remaining coupling: the
   runtime freezes the tuned row's value instead of re-deriving it, so a row
   written without that column falls back to the CU-count heuristic and can miss
   the AOT cache. `aiter.aot.flydsl.common.run_only_env()` makes FlyDSL raise on
   a JIT rather than fall back, which is how to verify a row by hand after
   changing the padding, channel-padding or split-K rules.

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
  -i aiter/configs/model_configs/wan21_vae_bf16_untuned_conv3d.csv \
  --run_config aiter/configs/model_configs/wan21_vae_bf16_tuned_conv3d.csv
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
(default 0.0, not the usual 0.05). The reference is bf16 rather than fp32 on
purpose: the tuner needs to catch a config that computes the wrong thing, not to
measure bf16 rounding, and a matching rounding regime puts every correct
candidate at exactly 0. Raise it only to investigate a specific failure --
`op_tests/test_flydsl_conv_implicit.py` fails on any mismatched element, so a
row tuned under a looser bar is a row that test will reject.

### `--timeout`
Per-task watchdog in seconds (default 1800). A worker killed by a GPU
memory-access fault leaves its task unresolvable; the watchdog drops it and
restarts the pool instead of hanging the run.

### `-o2, --profile_file`
Save every candidate's result, not just the winner.

### `-v, --verbose`
Detailed logging.
