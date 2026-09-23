# Tuning Tests

Minimal test suite for validating the aiter tuning infrastructure.

## Structure

| File | Level | GPU | What it tests |
|------|-------|-----|---------------|
| `test_csv_validation.py` | 0 | No | Tuned CSV integrity: duplicates (all families), invalid times, errRatio, git conflicts |
| `test_tuner_infra.py` | 1 | No | `base_tuner` utilities: CSV I/O, merge, dedup, calculate, post_process topk, update_config_files |
| `test_compare_logic.py` | 1 | No | Compare/update_improved: `_build_compare_update_plan`, `_merge_compare_filtered_results` |
| `test_mp_tuner_logic.py` | 1 | No | `mp_tuner` polling: timeout, AcceleratorError, KeyError, pool restart |
| `test_mha_tuner_logic.py` | 1 | No | MHA forward tuner: problem keys, candidate enumeration, gating, evidence, selection proof |
| `test_block_race.py` | 1 | No | Interleaved elimination race: incumbent protection, resume, tie-breaking, from synthetic latencies |
| `test_mha_search_and_promotion.py` | 1 | No | MHA forward candidate sample, `--backends` restriction, and the gate against the incumbent |
| `test_mha_store_agreement.py` | 1 | No | `aiter.ops.mha` and the Triton entry point resolve the same tile from one runtime CSV |
| `test_online_tune.py` | 1 | No | `AITER_ONLINE_TUNE` decision logic, `mp_lock` synchronization, MainFunc CSV write, cfg_2stages reload |
| `test_tune_pipeline.py` | 2 | Yes | End-to-end: run each tuner on small shapes (mp=1 + mp=default), verify output CSV; `--compare --update_improved`; `AITER_ONLINE_TUNE` e2e |
| `test_asm_splitk_guard.py` | 1 | No | `GemmTuner.asm_gemm_all_solutions` SplitK semaphore grid guard |
| `test_run_config.py` | 2 | Yes | Run --run_config on ALL existing tuned CSVs (configs + model_configs) |

## CI coverage

Fast deterministic tuning validation is part of the normal pull-request test
sharding through `.github/scripts/split_tests.sh`. That gate includes CSV
validation, shape-collision checks, and the mixed-MXFP tuner unit tests.

The full Level 0+1 workflow repeats the broader CPU suite on its daily/manual
schedule. GPU tuner-pipeline and all-shape `--run_config` validation remain
scheduled/manual because they require gfx950 and have substantially longer
runtime budgets.

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
| `gradlib_bf16` | `gradlib/gradlib/gemm_tuner.py` | `bf16_tuned_gemm.csv` | ✓ | ✓ (hipBLASLt/ASM/FlyDSL) |
| `gdn_k5_opt` | `csrc/gdn_k5/chunk_gdn_h_opt_tune.py` | `model_configs/*_chunk_gdn_h_opt_tuned.csv` | ✓ | ✓ (shape-only varlen smoke) |
| `mha_fwd` | `op_tests/tuners/tune_mha_fwd.py` | `tuned_mha_fwd.csv` | skipped until the CSV has rows for the GPU | ✓ (one varlen shape, one sampled Triton tile) |

Mixed-MXFP coverage is provided by
`csrc/gemm_a6w4/gemm_a6w4_tune.py` and
`csrc/gemm_a4w6/gemm_a4w6_tune.py`. Both families have static CSV validation,
runtime-config validation, and single-/multi-GPU pipeline smoke tests.

## Config resolution

`test_run_config` resolves tuned config files through `AITER_CONFIGS` in `aiter/jit/core.py` — the same path used by production operators at runtime. This validates that:

1. The `AITER_CONFIG_*` env var names and default file paths in `core.py` are correct
2. Model-specific configs under `aiter/configs/model_configs/` are properly discovered and merged
3. The merged config works with every tuned shape

If `AITER_CONFIGS` is unavailable (e.g. aiter not installed), the test falls back to filesystem scanning of `aiter/configs/` and `aiter/configs/model_configs/`.

## Running

```bash
# Level 0+1 only (no GPU, <10s)
python3 -m unittest op_tests.tuning_tests.test_csv_validation \
  op_tests.tuning_tests.test_tuner_infra \
  op_tests.tuning_tests.test_mp_tuner_logic \
  op_tests.tuning_tests.test_mixed_mxfp_tuning \
  op_tests.tuning_tests.test_online_tune -v

# Level 2: pipeline smoke (~10min)
python3 -m unittest op_tests.tuning_tests.test_tune_pipeline -v

# Level 2: run_config validation (~20min, all tuned CSVs)
python3 -m unittest op_tests.tuning_tests.test_run_config -v

# Everything
python3 -m unittest discover -s op_tests/tuning_tests -v
```

### Running individual tuner tests

Each tuner in `test_tune_pipeline.py` has two variants: `_mp1` (single GPU) and `_mp_default` (all GPUs).

```bash
# Run a specific tuner (both mp1 and mp_default)
python3 -m pytest op_tests/tuning_tests/test_tune_pipeline.py -k "gradlib_bf16" -v

# Run only the single-GPU variant
python3 -m pytest op_tests/tuning_tests/test_tune_pipeline.py -k "gradlib_bf16_mp1" -v

# Run only the multi-GPU variant
python3 -m pytest op_tests/tuning_tests/test_tune_pipeline.py -k "gradlib_bf16_mp_default" -v

# Run a specific tuner with unittest
python3 -m unittest op_tests.tuning_tests.test_tune_pipeline.TestTunePipeline.test_a8w8_blockscale_mp1 -v
```

## Reproducing with custom config

Use `TUNE_TEST_FAMILY` to run `--run_config` for a specific family. Config is resolved via `AITER_CONFIGS` automatically:

```bash
# Use production config resolution (recommended)
TUNE_TEST_FAMILY=a8w8_blockscale \
python3 -m unittest op_tests.tuning_tests.test_run_config.TestRunConfigCustom -v

# blockscale with preshuffle (--preshuffle is auto-applied)
TUNE_TEST_FAMILY=a8w8_blockscale_bpreshuffle \
python3 -m unittest op_tests.tuning_tests.test_run_config.TestRunConfigCustom -v
```

Optionally set `TUNE_TEST_CONFIG` to override with explicit CSV paths:

```bash
# Single config (relative path from aiter root)
TUNE_TEST_FAMILY=a8w8_blockscale \
TUNE_TEST_CONFIG="aiter/configs/a8w8_blockscale_tuned_gemm.csv" \
python3 -m unittest op_tests.tuning_tests.test_run_config.TestRunConfigCustom -v

# Merge multiple configs (pathsep separated, same as AITER_CONFIG_* env)
TUNE_TEST_FAMILY=a8w8_blockscale \
TUNE_TEST_CONFIG="aiter/configs/a8w8_blockscale_tuned_gemm.csv:aiter/configs/model_configs/a8w8_blockscale_tuned_gemm_ds_v3.csv" \
python3 -m unittest op_tests.tuning_tests.test_run_config.TestRunConfigCustom -v
```

Available families include `a8w8`, `a8w8_bpreshuffle`,
`a8w8_blockscale`, `a8w8_blockscale_bpreshuffle`, `a4w4_blockscale`,
`a6w4_asm`, `a4w6_asm`, `batched_a8w8`, `batched_bf16`,
`fmoe`, `gradlib_bf16`, and `gdn_k5_opt`.

The test checks both **exit code** and **per-shape status** — shapes with `ERROR` (kernel crash) or `MISMATCH` (accuracy exceeded errRatio) will fail the test.
