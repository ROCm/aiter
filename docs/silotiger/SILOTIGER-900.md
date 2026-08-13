# SILOTIGER-900: exhaustive MiniMax-M3 EP1 fused-quant MXFP8 MoE

## Contract

- architecture: gfx950, 256 CUs;
- model/intermediate dimensions: 6144/768;
- experts/top-k: 128/4;
- activation/output: BF16 clamped SwiGLU;
- activation/weight quantization: FP8 E4M3 with per-1x32 E8M0 scales;
- gate/up layout: interleaved.

This commit combines the accepted config-key normalization and fused-quant
stage-one behavior from the EP4 campaign with an exhaustive EP1 worklist.
Existing EP4 files remain available for audit; they are not dispatchable at
EP1.

## Exhaustive token buckets

The EP1 worklist and tuned table cover every power-of-two bucket from 1 through
131072:

```text
1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
8192, 16384, 32768, 65536, 131072
```

The configuration files are:

```text
aiter/configs/model_configs/minimax_m3_ep1_mxfp8_untuned_fmoe.csv
aiter/configs/model_configs/minimax_m3_ep1_mxfp8_tuned_fmoe.csv
```

The checked-in EP1 table is the older measured 768-wide configuration re-keyed
to the correct `128/4` contract. It is tagged
`ep1_paired_validated_20260813` after seven alternating paired rounds against
an exhaustive eight-GPU retune. The retuned treatment was 0.726% faster by
unweighted geometric mean, below the predeclared 1% noise threshold, and
regressed five buckets. It was rejected as inconclusive, so the checked-in
table remains selected.

All 18 buckets passed every production replay and every stage-one kernel emits
intermediate FP8 values and E8M0 scales directly (`kernelName1` uses the
fused-quant `_fp8` family). This is microbenchmark validation, not a
serving-level performance claim.

## AOT

```bash
PYTHONPATH=. python3 -m aiter.aot.flydsl.moe \
  --csv aiter/configs/model_configs/minimax_m3_ep1_mxfp8_tuned_fmoe.csv
```

The expected result is 36 successful jobs: 18 stage-one and 18 stage-two exact
signatures, with zero failures.

## Validation

```bash
pytest -q op_tests/test_fused_moe_config_lookup.py
pytest -q op_tests/tuning_tests/test_mxscale_preshuffle.py
pytest -q op_tests/tuning_tests/test_csv_validation.py
```

Run the two-stage correctness oracle for all 18 tokens with
`-dim 6144,768 -e 128 -k 4 -q 9 -a swiglu`. Generated or AOT-compiled rows are
not a performance claim; deployment requires the paired TP4/EP1 serving sweep
and fixed-seed GSM8K.
