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

## Tuner family coverage

| Family | Tuner script | Tuned CSVs validated | run_config | pipeline |
|--------|-------------|---------------------|------------|----------|
| `a8w8` | `csrc/ck_gemm_a8w8/gemm_a8w8_tune.py` | `a8w8_tuned_gemm.csv` | ✓ | ✓ (int8+fp8) |
| `a8w8_bpreshuffle` | `csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py` | `a8w8_bpreshuffle_tuned_gemm*.csv` | ✓ | ✓ (int8+fp8) |
| `a8w8_blockscale` | `csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py` | `a8w8_blockscale_tuned_gemm*.csv` | ✓ | ✓ + shape_grouped |
| `a8w8_blockscale_bpreshuffle` | same + `--preshuffle` | `a8w8_blockscale_bpreshuffle_tuned_gemm*.csv` | ✓ | — |
| `a4w4_blockscale` | `csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_tune.py` | `a4w4_blockscale_tuned_gemm*.csv` | ✓ | — |
| `batched_a8w8` | `csrc/ck_batched_gemm_a8w8/batched_gemm_a8w8_tune.py` | `a8w8_tuned_batched_gemm.csv` | ✓ | ✓ |
| `batched_bf16` | `csrc/ck_batched_gemm_bf16/batched_gemm_bf16_tune.py` | `bf16_tuned_batched_gemm.csv` | ✓ | ✓ + shape_grouped |
| `fmoe` | `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py` | `tuned_fmoe.csv` + model_configs | ✓ | ✓ (bf16/fp8/int8/gelu) |
| `gradlib_bf16` | `gradlib/gradlib/gemm_tuner.py` | `bf16_tuned_gemm.csv` | — | ✓ (hipBLASLt/ASM/FlyDSL) |

`test_run_config` auto-discovers tuned CSVs from both `aiter/configs/` and `aiter/configs/model_configs/`, merges them per family.

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

# blockscale with preshuffle
TUNE_TEST_FAMILY=a8w8_blockscale_bpreshuffle \
TUNE_TEST_CONFIG="aiter/configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv" \
python3 -m unittest op_tests.tuning_tests.test_run_config.TestRunConfigCustom -v
```

Available families: `a8w8`, `a8w8_bpreshuffle`, `a8w8_blockscale`, `a8w8_blockscale_bpreshuffle`, `a4w4_blockscale`, `batched_a8w8`, `batched_bf16`, `fmoe`

The test checks both **exit code** and **per-shape status** — shapes with `ERROR` (kernel crash) or `MISMATCH` (accuracy exceeded errRatio) will fail the test.
