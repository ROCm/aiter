# MiniMax M3 BF16 on gfx1250: partial tuning results

**287 of 300 source cases passed tuning and public-dispatch validation.**
This publishes the successful subset for inspection, not a completed model tuning
or a recommendation to replace every existing backend.

## Scope and loading

- Source: `aiter/configs/model_configs/minimax_m3_mxfp4_bf16_untuned_gemm.csv`
  at `83acc38bf4cb9c9cbc547916941bb16b778d72da`. Original per-row bias settings are retained.
- Main-model BF16 GEMMs only; the separate Eagle draft-model list is not included.
- Full imported TDM candidate space with split-K enabled, no extra pruning.
- Requires the `tdm` implementation in [PR #7548](https://github.com/ROCm/aiter/pull/7548).
  This branch contains that prerequisite; #7548 remains separate.
- Opt-in CSV: `aiter/configs/minimax_m3_gfx1250_tdm_bf16_partial_tuned_gemm.csv`.
  It is deliberately outside `model_configs/` and does not replace the default
  BF16 table. No default dispatch code or model table is changed.
- Failed cases are omitted, not filled with an unvalidated winner or zero time.
  Normal AITER lookup/fallback rules still apply when a row is absent.

Set the table before importing AITER:

    export AITER_CONFIG_GEMM_BF16="$PWD/aiter/configs/minimax_m3_gfx1250_tdm_bf16_partial_tuned_gemm.csv"

For correctness-only replay of the published table:

    ROCBLAS_USE_HIPBLASLT=0 python -m csrc.gemm_a16w16.validate_tdm_csv

This command reruns correctness checks; the publication checks below used the
saved per-case validations and audited all evidence. It is not a claim of an
additional 287-row GPU rerun at publication time.

## What passed

- **1,299,446/1,299,446 candidate checks**
  in the 287 published cases, zero recorded compile/candidate-validation errors.
- Every configuration list matches its frozen ordered source-space SHA256.
- Candidates were checked against nonzero quantized random source-order references;
  the fastest eight were remeasured with nine graph-event samples of twenty calls.
- Each winner also passed fresh normal-random validation (`atol=rtol=0.05`).
- Every published row passed AITER `gemm_a16w16`, `gemm_a16w16_out`, and graph
  replay after input/bias mutation. These are saved per-case GPU validations.
- Both tuning and validation used isolated FlyDSL **0.4.2**, gfx1250 with 256 CUs,
  PyTorch `2.11.0+rocm10.1.0a20260811-bkc.20260820`, HIP `7.16.26315`.

## Comparison results and limits

**286** cases have same-device baseline comparisons: **240** measured faster
with TDM, **46** did not. All 287 correct TDM winners are included for inspection,
including the 46 without a speed win. Do not deploy the entire opt-in table on
the assumption that every row is faster.

Case **164**, `(256,6144,1024)`, bias off, passed tuning and public validation,
but its comparison subprocess aborted (`-6`) without a result. Its speedup is
left blank, not treated as a win.

The baseline is the actual existing gfx1250 AITER selection at the parent
commit, not a gfx950 recorded time. Retiming alternated TDM and baseline order.
The tuning CSV `us` is the tuning-winner median; independent comparison medians
are reported separately and are not substituted into the tuning evidence.

All timings are **fixed-address HIP graph GPU event medians on a shared machine**.
Competing load/clocks were uncontrolled. These observations are not confidence
bounds, cold-cache/rotating-input results, or model-level end-to-end speedups.

TDM preserves source rounding: cast each split partial, accumulate partials in
FP32, cast the sum, then add bias and cast again. Even without splitting, output
is cast before bias. Other backends may fuse bias in FP32. `err_ratio=0` refers
to the quantized source-reference check, not bitwise agreement with arbitrary
fused-FP32 results. See [TDM documentation](gfx1250-tdm-hgemm.md).

## Excluded failures (13)

All were observed on device 1 with GPU memory access faults, at different stages
including reference-library and public-dispatch validation. This correlation
does **not** establish a hardware or TDM root cause. They remain unresolved
and are excluded regardless of whether a preliminary winner CSV exists.

| Case | M | N | K | Bias |
|---:|---:|---:|---:|:---:|
| 66 | 112 | 6144 | 4096 | False |
| 96 | 160 | 6144 | 1536 | False |
| 110 | 176 | 12288 | 6144 | False |
| 138 | 224 | 4992 | 6144 | False |
| 146 | 240 | 1536 | 6144 | False |
| 149 | 240 | 4992 | 6144 | False |
| 162 | 256 | 4992 | 6144 | False |
| 174 | 324 | 5120 | 1280 | True |
| 185 | 512 | 5120 | 1280 | True |
| 203 | 999 | 1280 | 5120 | True |
| 210 | 1296 | 1280 | 1280 | True |
| 247 | 7992 | 6144 | 6144 | True |
| 262 | 12483 | 6144 | 6144 | True |

## Audit data

- [Per-case candidate counts, medians, comparison status and evidence fingerprints](assets/minimax_m3_gfx1250_bf16_partial_comparison.csv)
- [Input/source fingerprints, aggregate counts and exclusions](assets/minimax_m3_gfx1250_bf16_partial_manifest.json)

Raw per-candidate JSON, timing samples and process logs remain in the recorded
benchmark run; they are not all included in this small results-only commit.
Remaining work is to diagnose/retry the 13 exclusions and recover the missing
comparison. No failure has been relabeled as a successful case.
