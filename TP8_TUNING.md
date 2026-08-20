# MiniMax-M3 TP8 FP8 W8A8 FlyDSL tuning

Shape: `model_dim=6144, inter_dim=384, experts=129, topk=5` on MI325X/gfx942.

## Method

- Swept legal stage-1 and stage-2 tiles at 16 token buckets from M=1 to M=32768.
- Used realistic imbalanced routing and HIP-graph replay timing.
- Compared each stage against the vLLM Triton MoE output and rejected NaN/Inf or mean relative delta above 0.08.
- Re-ran the selected composed pipeline from the AITER branch CSV.
- Ran the AITER model-CSV correctness path against its torch reference at all 16 buckets.

## Composed FlyDSL versus current vLLM Triton fallback

| M | Stage 1 tile | Stage 2 tile | FlyDSL us | Triton us | Speedup | Mean rel. delta |
|---:|---|---|---:|---:|---:|---:|
| 1 | 16:64:512 | 16:128:128 | 31.12 | 45.39 | 1.458x | 0.0554 |
| 2 | 16:64:512 | 16:128:128 | 35.96 | 49.47 | 1.376x | 0.0584 |
| 4 | 16:64:512 | 16:128:128 | 43.60 | 54.43 | 1.248x | 0.0577 |
| 8 | 16:64:256 | 16:128:128 | 62.37 | 79.43 | 1.273x | 0.0577 |
| 16 | 16:64:128 | 16:128:128 | 89.47 | 100.88 | 1.127x | 0.0572 |
| 32 | 16:128:256 | 16:128:128 | 142.55 | 157.53 | 1.105x | 0.0569 |
| 64 | 16:192:256 | 16:128:128 | 181.64 | 283.96 | 1.563x | 0.0562 |
| 128 | 16:192:512 | 16:128:128 | 213.51 | 317.97 | 1.489x | 0.0562 |
| 256 | 32:192:128 | 16:128:128 | 241.78 | 332.40 | 1.375x | 0.0563 |
| 512 | 32:192:64 | 32:256:128 | 263.68 | 358.14 | 1.358x | 0.0563 |
| 1024 | 96:192:256 | 32:256:128 | 323.37 | 451.53 | 1.396x | 0.0561 |
| 2048 | 96:128:128 | 32:256:128 | 419.30 | 572.87 | 1.366x | 0.0561 |
| 4096 | 96:384:64 | 32:256:128 | 593.73 | 887.43 | 1.495x | 0.0561 |
| 8192 | 128:128:64 | 32:256:128 | 1101.30 | 1589.35 | 1.443x | 0.0561 |
| 16384 | 64:192:128 | 32:256:128 | 1921.76 | 3000.83 | 1.562x | 0.0561 |
| 32768 | 64:192:128 | 32:256:128 | 3781.13 | 6004.11 | 1.588x | 0.0561 |

Geometric-mean speedup is 1.376x; M=8 decode is 1.273x and M=32768 prefill is 1.588x.

The Triton side has no `E=129,N=384` tuned file and uses its default fallback, so these ratios are evidence for the FlyDSL TP8 path but are not a final comparison against tuned Triton.

## Validation

- `op_tests/flydsl_tests/test_flydsl_moe_fp8_w8a8.py`: 10 passed.
- AITER `test_moe_2stage.py` model-CSV run: all 16 TP8 rows executed with strict accuracy enabled.
- Logits difference versus the torch reference stayed between `6.78e-06` and `1.59e-05`.
- Selected outputs had no NaN or Inf.
- Independent stage-2 sorting was exercised whenever stage tile-M differed.
- HIP graph replay with padded `topk_ids=-1` slots passed at M=8, 512, and 32768 after neutralizing invalid slots as the vLLM wire-up does. Graph output-norm spread was at most `2.57e-05`, and mean relative error remained below `0.0560`.
