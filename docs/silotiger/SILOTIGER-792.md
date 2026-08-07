# SILOTIGER-792: fused-quant MiniMax-M3 EP4 MoE

## Scope

This commit deploys correctness-gated stage-one kernels that emit the
intermediate FP8 values and E8M0 scales directly. It removes the separate
intermediate quantization helper from the covered two-stage FMoE path.

The tuned keys are `T=1,2,4,8,16,32,64,8192,16384` for the SILOTIGER-785
EP4 contract. Runtime token counts use AITER's padded-M lookup, so intermediate
decode buckets map to the corresponding power-of-two key and `M=8320` maps to
the `16384` MoE key.

The optimization changes the intermediate rounding boundary. It passed the
existing AITER oracle/tolerance policy, but still requires model-level accuracy
before deployment.

## Configuration

The deployment table is:

```text
aiter/configs/model_configs/minimax_m3_ep4_mxfp8_tuned_fmoe.csv
```

AITER merges this file automatically. For an isolated comparison:

```bash
export AITER_CONFIG_FMOE=$PWD/aiter/configs/model_configs/minimax_m3_ep4_mxfp8_tuned_fmoe.csv
```

Keep the default Opus sorting backend. CK sorting won the sorting microbenchmark
but did not produce a consistent complete-FMoE improvement, so this commit does
not force or add a sorting override.

## Required AOT step

```bash
PYTHONPATH=. python3 -m aiter.aot.flydsl.moe \
  --csv aiter/configs/model_configs/minimax_m3_ep4_mxfp8_tuned_fmoe.csv
```

The expected result is 18 successful jobs: nine stage-one and nine stage-two
exact signatures, with zero failures.

## Validation

```bash
pytest -q op_tests/test_fused_moe_config_lookup.py
pytest -q op_tests/tuning_tests/test_mxscale_preshuffle.py

HIP_VISIBLE_DEVICES=0 PYTHONPATH=. \
python3 op_tests/test_moe_2stage.py \
  -d bf16 -dim 6144,3072 \
  -t 1 2 4 8 16 32 64 8192 16384 \
  -q 9 -a swiglu -s f -e 32 -k 4 -p t -hip 0,0 \
  --no-flydsl-csv --kernel
```

Before deployment, run the paired SGLang TP4/EP4 serving sweep and fixed-seed
GSM8K. A generated or AOT-compiled table is not by itself an E2E performance
claim.
