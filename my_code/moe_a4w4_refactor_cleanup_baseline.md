# MoE A4W4 kernel cleanup baseline

## Scope

This document records the performance baseline before consolidating the tuned
gfx1250 A4W4 MoE kernels and removing obsolete experimental code.

- Machine: `a07-3`
- Container: `hyg_fyd_e2e`
- Repository: `/app/aiter`
- Branch: `hyg/moe_a4w4_pr_refactor`
- Commit: `f5ab14d6ce534da83576ee6e9439feb3659c990a`
- Data: `const0`
- Rounds per shape: `3`
- Timing: `testGraph=False`, `use_cuda_event=False`, `num_warmup=5`,
  `num_iters=20`, torch profiler `get_trace_perf(...).device_time_sum`
- GPU/KFD: checked idle before and after every shape

## Reproduction command

```bash
ROUNDS=3 bash my_code/moe_perf_compare.sh --curr
```

## Pre-refactor performance

| Shape | GEMM1 samples (us) | GEMM1 median (us) | GEMM2 samples (us) | GEMM2 median (us) | MoE e2e samples (us) | MoE median (us) | Correctness |
|---|---|---:|---|---:|---|---:|---|
| E64/T1536/topk8/M7168/I2048 | 72.057, 75.044, 77.207 | **75.044** | 47.590, 56.165, 54.040 | **54.040** | 175.84, 190.48, 189.14 | **189.14** | bitwise exact |
| E64/T16384/topk8/M7168/I2048 | 523.864, 519.042, 521.774 | **521.774** | 381.256, 382.414, 381.233 | **381.256** | 1159.37, 1156.06, 1155.40 | **1156.06** | bitwise exact |
| E256/T512/topk8/M7168/I2048 | 208.507, 207.940, 208.775 | **208.507** | 115.332, 115.572, 116.268 | **115.572** | 343.64, 343.25, 344.80 | **343.64** | bitwise exact |
| E256/T16384/topk8/M7168/I2048 | 543.883, 547.381, 546.593 | **546.593** | 401.763, 401.034, 400.127 | **401.034** | 1166.76, 1170.84, 1167.80 | **1167.80** | bitwise exact |
| E96/T512/topk6/M7168/I3072 | 120.396, 121.107, 120.944 | **120.944** | 65.238, 65.640, 66.777 | **65.640** | 204.65, 205.87, 206.98 | **205.87** | bitwise exact |
| E96/T16384/topk6/M7168/I3072 | 577.083, 574.555, 578.687 | **577.083** | 355.742, 358.456, 358.089 | **358.089** | 1126.26, 1126.72, 1128.90 | **1126.72** | bitwise exact |

Every run reported `logits_diff=0`, `rel_l2=0`, and matching GEMM1, GEMM2,
and final MoE output hashes.

The raw benchmark output is stored at:

```text
/app/aiter/my_code/moe_prefill_yadai_compare_runs/20261004T133754Z
```

## Post-refactor validation before commit

The same command was rerun after the file consolidation and removal of the
obsolete A/B paths. All cases remained bitwise exact.

| Shape | GEMM1 samples (us) | GEMM1 median (us) | GEMM2 samples (us) | GEMM2 median (us) | MoE e2e samples (us) | MoE median (us) | Correctness |
|---|---|---:|---|---:|---|---:|---|
| E64/T1536/topk8/M7168/I2048 | 72.659, 77.317, 77.427 | **77.317** | 49.351, 55.166, 55.771 | **55.166** | 179.62, 192.27, 190.15 | **190.15** | bitwise exact |
| E64/T16384/topk8/M7168/I2048 | 512.360, 524.926, 517.658 | **517.658** | 381.790, 381.627, 380.967 | **381.627** | 1146.49, 1160.16, 1151.64 | **1151.64** | bitwise exact |
| E256/T512/topk8/M7168/I2048 | 208.880, 207.253, 208.296 | **208.296** | 115.439, 115.933, 116.512 | **115.933** | 344.37, 342.00, 344.88 | **344.37** | bitwise exact |
| E256/T16384/topk8/M7168/I2048 | 554.218, 555.680, 548.467 | **554.218** | 399.699, 399.947, 400.295 | **399.947** | 1175.86, 1178.13, 1171.14 | **1175.86** | bitwise exact |
| E96/T512/topk6/M7168/I3072 | 121.646, 120.845, 120.681 | **120.845** | 66.218, 65.816, 65.759 | **65.816** | 206.76, 203.51, 205.47 | **205.47** | bitwise exact |
| E96/T16384/topk6/M7168/I3072 | 581.592, 574.818, 575.354 | **575.354** | 358.433, 357.048, 356.778 | **357.048** | 1132.45, 1124.64, 1125.14 | **1125.14** | bitwise exact |

Median changes relative to the pre-refactor run are within normal run-to-run
variation. For the two explicitly requested small-token shapes:

- `E256/T512/topk8/M7168/I2048`: GEMM1 `-0.10%`, GEMM2 `+0.31%`, MoE `+0.21%`.
- `E96/T512/topk6/M7168/I3072`: GEMM1 `-0.08%`, GEMM2 `+0.27%`, MoE `-0.19%`.

The raw post-refactor output is stored at:

```text
/app/aiter/my_code/moe_prefill_yadai_compare_runs/20261004T141529Z
```

## Refactor acceptance criteria

The refactor must preserve the selected kernel family, numerical behavior, and
performance for all six shapes above. The `E256/T512/topk8/M7168/I2048` and
`E96/T512/topk6/M7168/I3072` cases are the explicitly requested small-token
coverage. Dispatch predicates for specialized kernels must remain narrow so
other ROCm/main shapes continue to use their existing code paths.
