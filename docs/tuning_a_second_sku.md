# Tuning a second SKU of an architecture

A single architecture can cover several devices with different compute-unit counts. On `gfx950`, for example, MI355X reports 256 CUs and MI350P reports 128. Tuned rows are keyed on architecture and CU count together, so each device needs its own tuning run, and the results for both can live in one catalog.[^key] You then name the pairs that a build should cover with `AITER_GPU_TARGETS`. If a build target has no matching rows, every shape for that target uses the default kernel.

To tune and package a second SKU, follow these steps:

1. Tune as described in [Autotuning Pipelines in Aiter CI](autotuning_pipeline.md), either through the manual pipeline or by invoking the tuner directly. Nothing about the procedure changes for a second SKU, but the visible device has to report the target architecture and CU count. Leave `CU_NUM` unset.[^cunum] Both an MI350P and a DPX-partitioned MI355X can produce the 128-CU rows.[^partition]

2. Run the collision guard before you commit the new rows. Rows for a new CU count cannot collide with the existing ones, but the same run may have refreshed shapes that were already tuned:

    ```bash
    python3 -m unittest op_tests.tuning_tests.test_config_shape_collision -v
    ```

    If it reports duplicates, its write-back mode keeps the row with the lowest `us` value for each duplicate key:

    ```bash
    python3 op_tests/tuning_tests/test_config_shape_collision.py --fix
    ```

    The `--fix` option is honored only when the file runs as a script; the unittest invocation is always read-only. Keep the `gfx` column in the rewritten files, because architectures that share a CU count cannot otherwise be told apart.[^gfx]

3. Run the Level 2 `run_config` validation on each SKU. `TUNE_TEST_FAMILY` must name one of the `TUNER_FAMILIES` keys defined in [test_run_config.py](https://github.com/ROCm/aiter/blob/main/op_tests/tuning_tests/test_run_config.py). For example, to validate the `a8w8` family:

    ```bash
    TUNE_TEST_FAMILY=a8w8 \
    python3 -m unittest op_tests.tuning_tests.test_run_config.TestRunConfigCustom -v
    ```

    This test uses whichever GPU is visible; it does not take its target from `AITER_GPU_TARGETS`. When `TUNE_TEST_CONFIG` is unset, it locates the tuned CSV through `AITER_CONFIGS`, in the same way as the runtime operator. To validate one particular file instead, set `TUNE_TEST_CONFIG=aiter/configs/a8w8_tuned_gemm.csv`.

4. Build the package with every architecture and CU-count pair named in `AITER_GPU_TARGETS`.[^codegen] For an inference package covering both `gfx950` SKUs:

    ```bash
    PREBUILD_KERNELS=2 \
    AITER_GPU_TARGETS="gfx950:128;gfx950:256" \
    python3 setup.py install
    ```

    `PREBUILD_KERNELS` selects how much is prebuilt, and its default prebuilds nothing at all.[^prebuild] When `AITER_GPU_TARGETS` is set, it is authoritative for the build.[^archs] Watch the build log for `The tuned config CSV has no rows for build target(s) ...`, which names any target whose rows are missing.

    To exercise a SKU without building a package, run an op test under JIT on the target device and force the affected module to rebuild:[^jitcache]

    ```bash
    AITER_GPU_TARGETS=gfx950:128 \
    AITER_REBUILD=1 \
    python3 op_tests/test_gemm_a8w8.py
    ```

`CU_NUM` is a build-time flag, and a running kernel should take its CU count from the device it is on. Not every operator honors that split yet, so a `CU_NUM` left set in a serving environment can still change which kernel is selected.[^serving]

[^key]: When `update_config_files` creates the key for a tuned row, it uses `(gfx, cu_num, <shape columns>)`.

[^partition]: A DPX-partitioned MI355X exposes devices with 128 CUs. Because PyTorch reports `multi_processor_count` for each visible device, the tuner records `cu_num=128` without additional configuration. Such a partition has the same number of CUs per XCD and the same per-CU resources as the smaller part, but its cache and memory topology need not match,

[^cunum]: Most tuners read the live `multi_processor_count` reported by PyTorch and ignore `CU_NUM` entirely. The opus tuner can fall back to the override if device enumeration fails, which would silently label a row with the wrong CU count.

[^gfx]: During a merge, a missing `gfx` value is silently inferred from `cu_num`. At run time, `get_CKGEMM_config` instead emits a warning and falls back to a key based only on `cu_num`, which is ambiguous for two architectures that report the same CU count. Rows that still lack a `gfx` value should be re-tuned.

[^prebuild]: `PREBUILD_KERNELS` is an integer mode: `0` is the default JIT-only mode and prebuilds nothing, `1` prebuilds core kernels, `2` prebuilds inference kernels, and `3` prebuilds MHA kernels only. A plain `pip install -e .` therefore prebuilds no kernels.

[^archs]: If `GPU_ARCHS` specifies a conflicting value, the build emits a warning, ignores that value, and follows `AITER_GPU_TARGETS`.

[^codegen]: Config CSV files are filtered by `(gfx, cu_num)` before code generation, so a package built before the new rows were added did not compile their kernels. Updating only the config cannot enable them.

[^jitcache]: A cached module is not rebuilt automatically when you move between GPUs that share an architecture but have different CU counts, so the rebuild has to be forced.

[^serving]: `get_CKGEMM_config` and the fused MoE equivalent build their run-time key from `chip_info.get_cu_num`, which reads `CU_NUM` when it is set, so the lookup can miss or resolve to a kernel the package does not contain. The same value also sizes grids and split factors on several paths. Operators that read the count from the device instead are unaffected. The value is cached on first use and not re-read for the life of the process.
