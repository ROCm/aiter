# MXFP4 tuning selection validation

This report compares the default configuration before PR #7538 with the corrected
default and the opt-in Qwen3 FSDP profile. No kernel code or source CSV was changed.

## Revision and environment

- Baseline: ROCm/aiter `ff745fb09d669f40cf81d15d3e7598b61fe2a61c`.
- PR before correction: `636354c5b1c0a93ce24b5bc2587b019e41411ae2`.
- Validation host: `smci355-ccs-aus-n01-13`, one AMD Instinct MI355X, gfx950,
  256 CU as reported by PyTorch. No `CU_NUM` override.
- Image: `rocm/sgl-dev:v0.5.23-rocm724-mi35x-20261030`.
- Python 3.12, PyTorch 2.11.0+rocm7.2, HIP 7.2.26015, Triton 3.7.0.
- GPU measurements use latency selection. Power overlay compatibility was tested with isolated CSV fixtures.

## Default compatibility

The baseline includes 1,464 stock keys and 1,363 additional keys in seven
pre-existing model tables. The stock-only version of this PR dropped those 1,363
keys. The corrected default loads all eight tables from an explicit manifest.
The new Lumen tables remain opt-in.

All 2,827 baseline `(gfx, cu_num, M, N, K)` keys retain `kernelName` and `splitK`.
The real `get_GEMM_config` lookup was compared for 9,482 queries: every baseline
and new profile key, plus positive M-31, M-1 and M+1 neighbours, deduplicated.
The default has zero lookup differences, including padded-M lookup.

The audit uses the real compiled `get_padded_m`. It substitutes gfx/CU metadata
to inspect every table's key space; it does not claim GPU execution on those
other architectures or CU counts. Numerical execution below uses gfx950/256 CU.

| Profile | Profile keys | Effective keys | Changed lookups | Changes outside profile lookup coverage |
|---|---:|---:|---:|---:|
| llama2_7b | 9 | 2836 | 28 | 0 |
| llama31_8b | 7 | 2834 | 21 | 0 |
| qwen3_8b | 8 | 2835 | 25 | 0 |
| qwen3_small | 13 | 2839 | 41 | 0 |
| qwen3_fsdp | 21 | 2847 | 66 | 0 |

"Outside profile lookup coverage" means none of the exact-M, padded-M level 0,
or padded-M level 1 candidate keys belongs to the profile. Absence of an exact
shape from the profile CSV is not sufficient: for Qwen3 FSDP, 45 of the 66
changed queries are neighbours that reach a profile row through padded M.
These are changes within the existing GEMM bucket contract.

## GPU correctness and latency

The public `gemm_a4w4` ran on the same quantized inputs, seeded with 7538, through
the baseline, corrected default, and Qwen3 FSDP paths. All 18 cases passed the
repository's dequantized FP32 GEMM reference (`atol=rtol=0.01`). All six default
outputs are bit-identical to baseline outputs, and profiler kernel names match.
This does not establish bit identity for every backend or every workload.

Latency is the median of 50 CUDA-event samples after 10 warmups. Each sample
contains five public GEMM calls. Quantization, reference GEMM and profiling are
outside timing. This is a microbenchmark, not an end-to-end training result.

| M x N x K | Baseline us | Corrected default us | Qwen3 FSDP us | Baseline kernel | Profile kernel |
|---|---:|---:|---:|---|---|
| 1 x 512 x 4096 | 22.152 | 22.004 | 22.224 | 32x128 | 32x128 |
| 128 x 7168 x 4608 | 22.288 | 22.204 | 21.776 | 32x128 | 32x128 |
| 1024 x 1024 x 8192 | 24.328 | 24.092 | 22.888 | 192x128 | 32x128 |
| 1023 x 1024 x 8192 | 24.632 | 24.240 | 22.592 | 192x128 | 32x128 |
| 8192 x 4096 x 1024 | 36.156 | 36.208 | 35.816 | 256x256 | 128x512 |
| 8191 x 4096 x 1024 | 36.264 | 35.964 | 35.701 | 256x256 | 128x512 |

Small timing differences between identical default kernels are not an
optimization claim. The JSON evidence records configured rows, actual profiler
kernel names and output SHA-256 values.

## Reproduce

Prepare a baseline table from the unmodified upstream revision above. For
example, export its `aiter/configs` tree to an isolated directory, point
`aiter.jit.core.AITER_ROOT_DIR` at that directory, and call the unchanged
`AITER_CONFIGS.get_config_file("AITER_CONFIG_GEMM_A4W4", stock_path,
"a4w4_blockscale_tuned_gemm")` with the override unset. Copy its result to
`baseline.csv`. This uses the actual old merger and excludes the new Lumen files.
Do not run that merger against a dirty checkout: its duplicate guard may rewrite
sources when conflicts exist.

From the corrected checkout, run:

```bash
python op_tests/test_mxfp4_tuning_config.py
PYTHONPATH=. python -m pytest -q op_tests/tuning_tests/test_config_shape_collision.py op_tests/tuning_tests/test_csv_validation.py
PYTHONPATH=. python docs/mxfp4_tuning_validation/audit_lookup.py --baseline /path/to/baseline.csv --output-dir /path/to/audit-output
PYTHONPATH=. python docs/mxfp4_tuning_validation/gpu_check.py old --baseline /path/to/baseline.csv
PYTHONPATH=. python docs/mxfp4_tuning_validation/gpu_check.py default
PYTHONPATH=. python docs/mxfp4_tuning_validation/gpu_check.py qwen3_fsdp
```

Results: 21 configuration tests and 52 adjacent tests passed (73 total, 55 subtests). The adjacent
suite emitted two existing Cython/distutils deprecation warnings. Its A4W4
branch now exercises the actual new resolver in an isolated fixture, instead
of scanning and implicitly merging the opt-in Lumen tables.

## Review follow-ups

Regression tests cover power-row overlay and uncovered latency fallback, explicit-path bypass, frozen power policy, and read-only rejection of conflicting power tables. The A4W4 collision tool uses the production resolver even under `--fix`, preserving opt-in source tables. A private tuner-only reset lets pre/post comparisons load different tables while application configuration remains frozen after first lookup.

An additional repository CK GEMM sample at (8192,1024,1024) passed with the Qwen3 FSDP profile and the matching CK revision.

## Packaging and limits

An isolated source copy passed `setup.py sdist bdist_wheel` with
`ENABLE_CK=0 PREBUILD_KERNELS=0 AITER_USE_SYSTEM_TRITON=1`. The archives contain
the profile manifest, default-source manifest, and CSV resources. The wheel was
installed with `pip --no-deps --target` into a separate directory; importing it
from outside the checkout with a new `AITER_JIT_DIR` compiled core from scratch
and resolved 2,847 Qwen3 FSDP keys. No warm-cache source `.so` was copied in.

The first disk-backed attempt returned nonzero while the root filesystem was
nearly full; its full command is not counted as passed. The successful retry
used a fresh memory-backed directory to avoid consuming root-disk space. The review-fix rebuild also passed. Its first cold import found that the temporary copy script had excluded the source package `aiter/dist`; restoring that package, rebuilding, and installing into another empty target passed cold import and resolved 2,847 keys. The failed import is not counted as passed.

Full wheel kernel prebuild, the full GPU suite, other GPU architectures,
power-mode GPU measurements, and end-to-end training were not run. GitHub baseline
workflows required maintainer approval when this supplement was prepared;
local results do not establish a green GitHub CI run.
