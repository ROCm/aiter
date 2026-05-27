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
