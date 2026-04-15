# Tuning Tests

Minimal test suite for validating the aiter tuning infrastructure.

## Structure

| File | Level | GPU | What it tests |
|------|-------|-----|---------------|
| `test_csv_validation.py` | 0 | No | Tuned CSV integrity: duplicates, invalid times, errRatio, git conflicts |
| `test_tuner_infra.py` | 1 | No | `base_tuner` utilities: CSV I/O, merge, dedup, calculate, post_process topk |
| `test_mp_tuner_logic.py` | 1 | No | `mp_tuner` polling: timeout, AcceleratorError, KeyError, pool restart |
| `test_tune_pipeline.py` | 2 | Yes | End-to-end: run each tuner on small shapes, verify output CSV |
| `test_run_config.py` | 2 | Yes | Run --run_config on ALL existing tuned CSVs (configs + model_configs) |

## Running

```bash
# Level 0+1 only (no GPU, <10s)
python3 -m unittest op_tests.tuning_tests.test_csv_validation \
  op_tests.tuning_tests.test_tuner_infra \
  op_tests.tuning_tests.test_mp_tuner_logic -v

# Level 2: pipeline smoke (~10min)
python3 -m unittest op_tests.tuning_tests.test_tune_pipeline -v

# Level 2: run_config validation (~20min, all tuned CSVs)
python3 -m unittest op_tests.tuning_tests.test_run_config -v

# Everything
python3 -m unittest discover -s op_tests/tuning_tests -v
```

## Reproducing with custom config

Use `TUNE_TEST_FAMILY` and `TUNE_TEST_CONFIG` to run `--run_config` on a specific tuned CSV:

```bash
# Single config (relative path from aiter root)
TUNE_TEST_FAMILY=a8w8_blockscale \
TUNE_TEST_CONFIG="aiter/configs/a8w8_blockscale_tuned_gemm.csv" \
python3 -m unittest op_tests.tuning_tests.test_run_config.TestRunConfigCustom -v

# Merge multiple configs (pathsep separated, same as AITER_CONFIG_* env)
TUNE_TEST_FAMILY=a8w8_blockscale \
TUNE_TEST_CONFIG="aiter/configs/a8w8_blockscale_tuned_gemm.csv:aiter/configs/model_configs/a8w8_blockscale_tuned_gemm_ds_v3.csv" \
python3 -m unittest op_tests.tuning_tests.test_run_config.TestRunConfigCustom -v

# Reproduce fmoe issues
TUNE_TEST_FAMILY=fmoe \
TUNE_TEST_CONFIG="aiter/configs/tuned_fmoe.csv" \
python3 -m unittest op_tests.tuning_tests.test_run_config.TestRunConfigCustom -v

# bpreshuffle with model-specific config
TUNE_TEST_FAMILY=a8w8_bpreshuffle \
TUNE_TEST_CONFIG="aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv:aiter/configs/a8w8_bpreshuffle_tuned_gemm1.csv" \
python3 -m unittest op_tests.tuning_tests.test_run_config.TestRunConfigCustom -v
```

Available families: `a8w8`, `a8w8_bpreshuffle`, `a8w8_blockscale`, `batched_a8w8`, `batched_bf16`, `fmoe`

The test checks both **exit code** and **per-shape status** — shapes with `ERROR` (kernel crash) or `MISMATCH` (accuracy exceeded errRatio) will fail the test.
