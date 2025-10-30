# Autotuning Pipelines in Aiter CI

## What is the tuning pipeline workflow?

An automated tuning system that ingests and benchmarks a volume of inputs, then records the best operator for each input in a database based on test results, so that future identical inputs can directly return the optimal operator.

## Implementation

In the Aiter repository, there are tuning scripts designed for various shapes, such as `aiter/csrc/ck_batched_gemm_a8w8` (see: [ROCm/aiter](https://github.com/ROCm/aiter)).

Running these scripts generates tuned results, which are stored in the `aiter/configs` directory, for example: `aiter/configs/a8w8_tuned_batched_gemm.csv`. These CSV files are compiled during the Aiter installation process and are referenced when using Aiter operators.

Based on this, we provide CI pipelines to generate and use these tuned CSV files:

- [Manual Pipeline](https://github.com/ROCm/aiter/actions/workflows/operators-tuning.yaml): Allows users to select specific shapes to tune and choose whether to upload the results to the Aiter repository.

- Scheduled Pipeline: Runs nightly or weekly to generate all tuned CSV files and automatically upload the results to the Aiter repository.
