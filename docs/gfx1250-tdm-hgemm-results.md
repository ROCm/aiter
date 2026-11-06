# gfx1250 TDM BF16 tuning

38 rows (19 shapes, bias off/on); source-space candidates checked and winners
validated via AITER public GEMM/out APIs and changed-input HIP graph replay.

Tuning: FlyDSL 0.4.1. Final public-dispatch validation: isolated FlyDSL 0.4.2.
Fixed-address graph-event GPU times on a shared gfx1250 machine; not a speedup claim.

| M | N | K | Bias | us | TFLOPS | Correct / candidates |
|---:|---:|---:|:---:|---:|---:|---:|
| 8 | 4096 | 4096 | False | 6.113 | 43.91 | 781/781 |
| 8 | 4096 | 4096 | True | 7.101 | 37.80 | 781/781 |
| 16 | 4096 | 4096 | False | 5.701 | 94.18 | 781/781 |
| 16 | 4096 | 4096 | True | 6.436 | 83.42 | 781/781 |
| 32 | 4096 | 4096 | False | 8.362 | 128.40 | 2144/2144 |
| 32 | 4096 | 4096 | True | 9.975 | 107.64 | 2144/2144 |
| 64 | 4096 | 4096 | False | 11.986 | 179.17 | 6113/6113 |
| 64 | 4096 | 4096 | True | 11.495 | 186.82 | 6113/6113 |
| 128 | 4096 | 4096 | False | 22.249 | 193.04 | 6604/6604 |
| 128 | 4096 | 4096 | True | 14.201 | 302.44 | 6604/6604 |
| 256 | 4096 | 4096 | False | 15.413 | 557.31 | 5863/5863 |
| 256 | 4096 | 4096 | True | 24.693 | 347.87 | 5863/5863 |
| 512 | 4096 | 4096 | False | 19.123 | 898.40 | 4411/4411 |
| 512 | 4096 | 4096 | True | 25.999 | 660.79 | 4411/4411 |
| 1024 | 4096 | 4096 | False | 33.658 | 1020.84 | 2952/2952 |
| 1024 | 4096 | 4096 | True | 36.467 | 942.23 | 2952/2952 |
| 2048 | 4096 | 4096 | False | 41.003 | 1675.95 | 1636/1636 |
| 2048 | 4096 | 4096 | True | 43.775 | 1569.82 | 1636/1636 |
| 1024 | 1024 | 1024 | False | 32.965 | 65.14 | 5232/5232 |
| 1024 | 1024 | 1024 | True | 37.953 | 56.58 | 5232/5232 |
| 2048 | 2048 | 2048 | False | 27.377 | 627.53 | 2952/2952 |
| 2048 | 2048 | 2048 | True | 38.051 | 451.50 | 2952/2952 |
| 4096 | 4096 | 4096 | False | 69.394 | 1980.57 | 650/650 |
| 4096 | 4096 | 4096 | True | 69.995 | 1963.56 | 650/650 |
| 4096 | 4096 | 8192 | False | 110.535 | 2486.80 | 650/650 |
| 4096 | 4096 | 8192 | True | 115.196 | 2386.18 | 650/650 |
| 8192 | 8192 | 8192 | False | 455.053 | 2416.23 | 650/650 |
| 8192 | 8192 | 8192 | True | 469.799 | 2340.39 | 650/650 |
| 8 | 7168 | 2048 | False | 27.716 | 8.47 | 599/599 |
| 8 | 7168 | 2048 | True | 32.589 | 7.21 | 599/599 |
| 32 | 384 | 7168 | False | 29.336 | 6.00 | 2734/2734 |
| 32 | 384 | 7168 | True | 33.498 | 5.26 | 2734/2734 |
| 32 | 14336 | 4096 | False | 24.543 | 153.13 | 1496/1496 |
| 32 | 14336 | 4096 | True | 28.000 | 134.22 | 1496/1496 |
| 16 | 28672 | 4096 | False | 44.753 | 83.97 | 440/440 |
| 16 | 28672 | 4096 | True | 46.718 | 80.44 | 440/440 |
| 4096 | 256 | 4096 | False | 47.281 | 181.68 | 5863/5863 |
| 4096 | 256 | 4096 | True | 47.697 | 180.09 | 5863/5863 |

## Validation and provenance

- Source: `xytpai/flydsl-examples@5017f4c16fc46badb5fce58c485e9bd090ffe325`.
- AITER base: `2cc7c97ad4c9f5c9594201ac730c781ae390a770`.
- Kernel factory and validator AST are identical to the source.
- 42 source/port ordered candidate-space comparisons passed.
- **105,102/105,102 candidate checks passed**, with zero compile or validation
  failures across the 38 shape/bias combinations.
- The top eight candidates per combination were remeasured; all 38 selected
  configurations passed an additional fresh normal-random check.
- All 38 CSV rows passed public `gemm_a16w16`, `gemm_a16w16_out`, and HIP graph
  replay after changing input and bias values, using FlyDSL 0.4.2.
- 11 catalog/device regression tests passed on gfx1250 with FlyDSL 0.4.2.
- 32 additional old `ft`/`ht` cases passed: FP16/BF16 inputs, matching/FP32
  outputs, split-K 1/2 and full/boundary M/N shapes.
- Extended CPU tuner/CSV regression: 50 passed, 2 errors. Both errors reproduced
  on the unmodified AITER base above: `TestGroupedFmoeResultToCsv` tests
  `test_production_layout_gets_no_new_blank_cell` and
  `test_production_layout_keeps_its_columns`, due to missing `a_preshuffle`
  blank-cell handling. The affected grouped-MoE code is not changed by this PR.

Environment: PyTorch `2.11.0+rocm10.1.0a20260811-bkc.20260820`, HIP `7.16.26315`,
gfx1250 with 256 CUs. Tuning used the existing FlyDSL 0.4.1 installation;
a separate 0.4.2 installation was used for final correctness/dispatch validation.
The CSV timings were **not** remeasured with 0.4.2.

These are TDM-only selections from the imported source space, not a claim that
TDM beats every existing AITER backend. Shared-device load was not controlled;
in particular, timings across rows should not be treated as bias-overhead or
shape-scaling comparisons. Fixed-address HIP graph timing is not equivalent to
the examples' rotating-input benchmark or end-to-end application latency.

## Load the shipped opt-in table

From the repository root:

    export AITER_CONFIG_GEMM_BF16="$PWD/aiter/configs/gfx1250_tdm_bf16_tuned_gemm.csv"
    ROCBLAS_USE_HIPBLASLT=0 python -m csrc.gemm_a16w16.validate_tdm_csv

This table is not automatically merged into the default model tables.
See [usage and tuning methodology](gfx1250-tdm-hgemm.md) for kernel constraints,
source rounding semantics and reproduction commands.
Compact per-row candidate counts and finalist samples are recorded in
[the tuning summary](assets/gfx1250_tdm_tuning_summary.json).
