# SILOTIGER-722: dense MiniMax-M3 MXFP8 FlyDSL GEMM

## Scope

This commit adds a FlyDSL MX-scale preshuffle GEMM, exact-signature dispatch,
tuning support, and AOT compilation for MiniMax-M3 dense projections on
`gfx950`.

The model table contains 147 exact `(M,N,K)` signatures:

- decode graph buckets: `1,2,4,8,12,16,24,32,40,48,56,64`;
- prefill values: `128,256,512,1024,2048,4096,8192,8320,16384`;
- seven TP4 weight shapes: QKV, fused QKV/index, attention output, dense
  gate/up, dense down, shared gate/up, and shared down.

Unknown signatures remain on the owning framework's canonical fallback.

## Build

FlyDSL `0.3.0` is required.

```bash
git submodule update --init --recursive
pip install -r requirements.txt
PREBUILD_KERNELS=1 GPU_ARCHS=gfx950 python3 setup.py build_ext --inplace
GPU_ARCHS=gfx950 pip install --config-settings editable_mode=compat -e .
```

## Required AOT step

The tuned CSV is runtime data; compiling the source tree alone does not
materialize all selected FlyDSL kernels. Precompile the exact deployment table:

```bash
PYTHONPATH=. python3 -m aiter.aot.flydsl.mxscale_preshuffle \
  --csv aiter/configs/model_configs/minimax_m3_dense_mxfp8_mxscale_preshuffle_tuned_gemm.csv
```

The command must finish with all 147 signatures compiled and zero failures.
Use the same FlyDSL, ROCm, Python, and target architecture as the serving image.

## Runtime configuration

The default merged table is:

```text
aiter/configs/model_configs/minimax_m3_dense_mxfp8_mxscale_preshuffle_tuned_gemm.csv
```

Override it for isolated experiments with:

```bash
export AITER_CONFIG_GEMM_MXSCALE_PRESHUFFLE=/path/to/tuned.csv
```

Do not switch dense backends inside a running server. Weight preshuffle happens
at model load, so restart the process when changing the backend or AITER
revision.

## Retuning

```bash
HIP_VISIBLE_DEVICES=0 PYTHONPATH=. \
python3 -m aiter.ops.flydsl.gemm_tune.tune_mxscale_preshuffle \
  --untune_file aiter/configs/mxscale_preshuffle_untuned_gemm.csv \
  --tune_file /tmp/mxscale_preshuffle_candidate.csv
```

Promote only complete quant-plus-GEMM winners that pass unchanged correctness
and the predeclared regression threshold.

## Validation

```bash
pytest -q op_tests/flydsl_tests/test_flydsl_mxscale_preshuffle.py
pytest -q op_tests/tuning_tests/test_mxscale_preshuffle.py
pytest -q op_tests/tuning_tests/test_csv_validation.py
```

The paired SGLang runtime must use BF16, `SGLANG_USE_AITER=1`, and
`--fp8-gemm-backend aiter`.
