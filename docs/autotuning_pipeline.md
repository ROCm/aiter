# Autotuning Pipelines in Aiter CI

## What is the tuning pipeline workflow?

An automated tuning system that ingests and benchmarks a volume of inputs, then records the best operator for each input in a database based on test results, so that future identical inputs can directly return the optimal operator.

## Implementation

In the Aiter repository, there are tuning scripts designed for various shapes, such as `aiter/csrc/ck_batched_gemm_a8w8` (see: [ROCm/aiter](https://github.com/ROCm/aiter)).

Running these scripts generates tuned results, which are stored in the `aiter/configs` directory, for example: `aiter/configs/a8w8_tuned_batched_gemm.csv`. These CSV files are compiled during the Aiter installation process and are referenced when using Aiter operators.

Based on this, we provide CI pipelines to generate and use these tuned CSV files:

- [Manual Pipeline](https://github.com/ROCm/aiter/actions/workflows/operators-tuning.yaml): Allows users to select specific shapes to tune and choose whether to upload the results to the Aiter repository.

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

- Scheduled Pipeline: Runs nightly or weekly to generate all tuned CSV files and automatically upload the results to the Aiter repository.
