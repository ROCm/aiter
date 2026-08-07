# SILOTIGER-785: MiniMax-M3 EP4 native MXFP8 MoE

## Scope

This commit adds the MiniMax-M3 TP4/EP4 shape contract and makes AITER's FMoE
configuration lookup distinguish two EP conventions:

- standard SGLang EP passes the four routed top-k columns unchanged;
- legacy callers may append one always-masked fake slot.

Standard SGLang must call `fused_moe(..., has_fake_topk_slot=False)`. Omitting
the argument preserves AITER's legacy fake-slot lookup first.

Target contract:

- architecture: `gfx950`, 256 CUs;
- model/intermediate dimensions: `6144/3072`;
- local experts/top-k: `32/4`;
- activation/output: BF16 clamped SwiGLU;
- activation/weight quantization: FP8 E4M3 with per-1x32 E8M0 scales;
- gate/up layout: interleaved.

## Paired SGLang requirement

Deploy this commit with the SGLang SILOTIGER-785 commit from the paired
`minimax-m3-mxfp8-v0.5.16` branch. Stock AITER and stock SGLang use different
top-k conventions and must not be mixed with only one side patched.

## Build

```bash
git submodule update --init --recursive
pip install -r requirements.txt
AITER_USE_SYSTEM_TRITON=1 python3 setup.py develop
```

The ROCm SGLang image uses the stronger prebuild form:

```bash
PREBUILD_KERNELS=1 GPU_ARCHS=gfx950 python3 setup.py build_ext --inplace
GPU_ARCHS=gfx950 pip install --config-settings editable_mode=compat -e .
```

FlyDSL `0.3.0` is required by the post2 branch.

## Configuration

The shape source is:

```text
aiter/configs/model_configs/minimax_m3_ep4_mxfp8_untuned_fmoe.csv
```

`AITER_CONFIG_FMOE` may point at an isolated tuned CSV during experiments.
Normally AITER merges model-specific tuned CSVs automatically.

No model-specific AOT kernels are added by this commit alone. SILOTIGER-792
adds the accepted fused-quant rows and its required AOT step.

## Validation

```bash
pytest -q op_tests/test_fused_moe_config_lookup.py
pytest -q op_tests/tuning_tests/test_csv_validation.py
```

For serving, use `SGLANG_USE_AITER=1`, `--quantization mxfp8`,
`--moe-runner-backend aiter`, `--tp 4`, and `--ep-size 4`.
