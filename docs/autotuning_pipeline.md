# Autotuning Pipelines in Aiter CI

## What is the tuning pipeline workflow?

An automated tuning system that ingests and benchmarks a volume of inputs, then records the best operator for each input in a database based on test results, so that future identical inputs can directly return the optimal operator.

## Implementation

In the Aiter repository, there are tuning scripts designed for various shapes, such as `aiter/csrc/ck_batched_gemm_a8w8` (see: [ROCm/aiter](https://github.com/ROCm/aiter)).

Running these scripts generates tuned results, which are stored in the `aiter/configs` directory, for example: `aiter/configs/a8w8_tuned_batched_gemm.csv`. These CSV files are compiled during the Aiter installation process and are referenced when using Aiter operators.

Based on this, we provide two CI paths: one for generating tuned CSVs on demand, and one for validating the tuning infrastructure on demand.

- [Manual Pipeline](https://github.com/ROCm/aiter/actions/workflows/operators-tuning.yaml): Uses the current untuned CSV inputs to generate refreshed tuned CSV artifacts. This is the workflow to run when you want to benchmark operators, inspect CSV diffs, and decide whether to update tracked configs.

    1. Navigate to the Autotuning Pipelines GitHub Actions workflow page: https://github.com/ROCm/aiter/actions/workflows/operators-tuning.yaml
    
    2. To trigger the workflow, click the `Run workflow` button at the top right corner of the Actions page. By default, this will run the tuning process for all shapes available in the `aiter/configs` directory. If you wish to tune only specific shapes, enter a comma-separated list of shape names in the `List of shape names to run` field, for example: `ck_gemm_a8w8, ck_gemm_a8w8_blockscale, ck_gemm_a8w8_blockscale_bpreshuffle, ck_gemm_a8w8_bpreshuffle`. If additional arguments are needed for the tuning script, you can provide them in the `Additional arguments for the tuning script` field. A full list of supported arguments can be found in the [base_tuner.py script](https://github.com/ROCm/aiter/blob/main/aiter/utility/base_tuner.py#L70).

        ![Aiter Autotuning CI Pipeline - 1](https://raw.githubusercontent.com/ROCm/aiter/main/docs/images/autotuning_ci_pipeline_1.jpeg)

    3. During the workflow execution, the following steps will be performed:
        - Run performance tests before tuning.
        - Execute the tuning process for the selected operators.
        - Display the differences in the CSV files after tuning.
        - Run performance tests again after tuning to compare results.
        - Upload the tuned CSV files as GitHub workflow artifacts.
        - You can download the tuned CSV artifacts and upload them to the Aiter repository as needed.

    4. If you wish to upload your own untuned CSV files, please create a new branch and update the relevant untuned CSV files in the `aiter/configs` directory. Then, trigger the workflow on your branch to proceed with tuning.

        ![Aiter Autotuning CI Pipeline - 2](https://raw.githubusercontent.com/ROCm/aiter/main/docs/images/autotuning_ci_pipeline_2.jpeg)

- [Manual Validation Pipeline](https://github.com/ROCm/aiter/actions/workflows/tuning-tests.yaml): Runs the `op_tests/tuning_tests` suite from `README.md` without rewriting repository CSVs.

    1. The workflow is started with `workflow_dispatch` and lets you choose which README command to run:
        - `all`
        - `level01`
        - `tune_pipeline`
        - `run_config`

    2. It mirrors the tuning test plan in `op_tests/tuning_tests/README.md`:
        - Level 0+1:
          - `test_csv_validation.py`
          - `test_tuner_infra.py`
          - `test_mp_tuner_logic.py`
        - Level 2 pipeline:
          - `test_tune_pipeline.py`
        - Level 2 run_config:
          - `test_run_config.py`

    3. The workflow uploads unittest logs and `/tmp/tuning_test_reports/` as artifacts so manual failures can be diagnosed without regenerating tuned CSVs.

    4. Unlike the manual tuning pipeline, this workflow does not call `op_tune.sh`, does not mutate tracked CSV files, and is intended only to verify that the tuning stack and existing tuned configs remain healthy in CI.

## Tuning a second SKU of an architecture

Some architectures ship in more than one size. MI350X/MI355X and MI350P are all `gfx950`, but
the first two have 256 compute units and MI350P has 128. Tuned rows are keyed on
`(gfx, cu_num, ...)`, so a tuned catalog for one does not apply to the other, and a build for
the smaller part finds no rows at all.

Because `cu_num` is part of the key, rows for a second SKU **cannot collide** with the existing
ones — `(gfx950, 128)` and `(gfx950, 256)` are distinct keys for the same shape. Adding a
catalog for a new SKU is additive.

### 1. Tune on hardware that reports the target CU count

The tuner takes the CU count from the live device (`multi_processor_count`) and the
architecture from `rocminfo`, and this is deliberate: `CU_NUM` selects which rows a *build*
uses, but a measurement has to come from real silicon. There is no way to produce 128-CU rows
on a device that reports 256. (The opus tuner does fall back to `chip_info.get_cu_num`, which
honours `CU_NUM`, when torch's device enumeration fails — leave `CU_NUM` unset while tuning so
that fallback cannot mislabel a row.)

Two ways to get a device that reports the target count:

- The part itself.
- A compute partition of a larger part that exposes the same count — for `gfx950`, a
  DPX-partitioned MI355X exposes 128-CU devices. `multi_processor_count` is reported per
  visible device, so the tuner stamps `cu_num=128` with no configuration.

Note that a partition of a larger part is not identical to the smaller part: CUs per XCD and
all per-CU resources match, but cache and memory topology need not. Prefer the real part when
one is available.

### 2. Run the tuners

Nothing changes here — run them exactly as described above, whether through the CI pipeline or
directly, for example:

```bash
python3 csrc/ck_gemm_a8w8/gemm_a8w8_tune.py -i aiter/configs/a8w8_untuned_gemm.csv \
                                            -o aiter/configs/a8w8_tuned_gemm.csv
```

The rows come out stamped with the `gfx` and `cu_num` of the machine they ran on.

### 3. Check the merge

New rows for a new `cu_num` cannot collide with existing ones, but the tuning run may also have
refreshed shapes that already exist. Run the collision guard before pushing:

```bash
python3 -m unittest op_tests.tuning_tests.test_config_shape_collision -v
```

If it reports duplicates, resolve them with the built-in dedup rather than by hand:

```bash
python3 op_tests/tuning_tests/test_config_shape_collision.py --fix
```

Keep the `gfx` column the tuner wrote. Architectures that share a `cu_num` are only
distinguishable by that column. For backward compatibility, legacy AOT rows without `gfx`
still use the historical CU-to-arch inference and emit a warning; re-tune them to remove the
ambiguity.

### 4. Build for both SKUs

Name every target the build should serve. `AITER_GPU_TARGETS` takes a `;`- or `,`-separated
list of `gfx` or `gfx:cu_num` entries. Ordering and duplicate entries do not change the target
set or its cache identity:

```bash
AITER_GPU_TARGETS="gfx950:128;gfx950:256" pip install -e .
```

A bare `gfx950` entry uses the default CU count for that architecture, so name both explicitly
when you want both baked. With the variable unset, `GPU_ARCHS` and `CU_NUM` behave as before.

When `AITER_GPU_TARGETS` is set it is the authority for the whole build: its arch set is what
reaches `--offload-arch`, so `GPU_ARCHS` does not need to be set alongside it, and if both are
set and disagree the build warns and follows `AITER_GPU_TARGETS`.

`GPU_ARCHS` cannot carry a `:cu_num` suffix and rejects one; use `CU_NUM` for its single global
CU override. Without `CU_NUM`, the count comes from the live device when it matches the named
arch, and from `GFX_CU_NUM_MAP` otherwise.

### 5. Rebuild after adding rows

Which kernels a module *contains* is decided at **build** time: config CSVs are filtered by
`(gfx, cu_num)` before codegen, so a library built before the new rows existed holds only the
default kernel for that SKU. The tuned kernels are not merely unselected, they were never
compiled. A requested target with no rows emits a warning and retains its existing runtime
fallback. Adding rows therefore requires a rebuild, not just a config update.

Some ops read the tuned CSV again at run time and index it on `(gfx, cu_num, M, N, K)` —
`get_CKGEMM_config` in `aiter/ops/gemm_op_a8w8.py` and the MoE equivalent in
`aiter/fused_moe.py` do. There the key has to match too: setting `CU_NUM` in a serving
environment changes the runtime key. If that row was not baked, the wrapper either takes its
documented default or the C++ registry rejects the unavailable kernel name; do not use
`CU_NUM` to impersonate hardware in serving.
