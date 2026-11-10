# gfx1250 TDM HGEMM

This ports the BF16/FP16 TDM kernel and candidate space from
[xytpai/flydsl-examples](https://github.com/xytpai/flydsl-examples/tree/5017f4c16fc46badb5fce58c485e9bd090ffe325)
at commit `5017f4c16fc46badb5fce58c485e9bd090ffe325`.
It is a new FlyDSL `tdm` policy, not a replacement for `ft`, `ht`, or `sk`.
The source kernel factory, wave32 TDM pipeline, block swizzle, cluster split-K,
and numerical rounding order are preserved. This port does not include the
separate experimental FP8/PTPC implementation.

## Interface and limits

    from aiter.ops.flydsl.gemm_kernels import flydsl_hgemm

    # A[M,K], W[N,K]; no weight copy or preshuffle.
    out = flydsl_hgemm(
        a, w, bias=bias, policy="tdm",
        block_m=64, block_n=128, block_k=128,
        stages=3, m_waves=2, n_waves=2, group_m=0, split_k=1,
    )

- gfx1250 only, including AITER's `gfx1250-strict` architecture label.
- BF16 or FP16 inputs, with matching output and bias dtype; FP32 accumulation.
- A is K-contiguous. W may be K-contiguous (source NT) or N-contiguous
  (source NN). The wrapper uses a transpose view, not a data copy.
- M/N boundary tiles are masked. K must be divisible by `split_k * block_k`;
  there is no K-tail path.
- Pointers and leading strides require 16-byte alignment; aligned row padding
  is supported. Output/input overlap is rejected.
- The original tile/wave/LDS and cluster constraints are validated before launch.
- Explicit streams and HIP graph replay are supported. For an explicit stream,
  callers must establish dependencies and retain their input/output tensors.
  The wrapper records the internal split-K workspace on the launch stream.

**Numerical semantics:** the non-split result is rounded to the input dtype
before bias addition. Cluster split-K rounds each partial to the input dtype,
sums partials in FP32, rounds the sum, then adds bias and rounds again. This is
the source behavior, not the rounding behavior of a single FP32 `linear` with
fused bias. The policy does not support FP32 output, scaling or preshuffled W.
Existing policies continue to serve their supported cases.

## Tune through AITER's standard GEMM tuner

The new `tdm` candidates are part of the existing `GemmA16W16Tuner`, alongside
the existing backends. No separate TDM tuner or additional tuning CSV is shipped.

For the existing MiniMax M3 gfx1250 BF16 entries, run from the repository root:

    config=aiter/configs/model_configs/minimax_m3_eagle_bf16_tuned_gemm.csv
    python -m csrc.gemm_a16w16.gemm_a16w16_tune \
      --input_file "$config" --tuned_file "$config" \
      --all --libtype all --splitK --mp 3 --shape_grouped --batch 3 \
      --compare --update_improved --min_improvement_pct 3

Use the same input/output path: the standard tuner's `--all` selects existing
rows for the current gfx/CU identity, retaining other architectures.
The file name includes `eagle`, but this is the existing MiniMax BF16 table
containing the gfx1250 entries. The separate `minimax_m3_mxfp4` table has no
gfx1250 rows at the source base and is not extended by this retune.

`--compare --update_improved` uses the standard production-op before/after
comparison and only updates qualifying rows. Failed candidates retain the
standard tuner's rejection rules; there is no custom reference or timing path.
`--libtype all` includes the backends available in the environment; hipBLASLt
remains opt-in via the existing `--with-hipblaslt` option.

To verify the resulting table through the standard production-op benchmark:

    python -m csrc.gemm_a16w16.gemm_a16w16_tune \
      --run_config "$config" --input_file "$config" --tuned_file "$config"

Tune and benchmark on representative hardware/load before interpreting results
as application-level gains. The source TDM rounding semantics above still apply.

## Regression tests

    python -m unittest \
      op_tests.tuning_tests.test_a16w16_tuned_config_lookup \
      op_tests.flydsl_tests.test_gemm_a16w16_tdm -v

The device tests cover BF16/FP16, NT/NN weights, padded input/output strides,
masked M/N, small M, group swizzle, bias rounding, cluster split-K (1/2/3/9),
non-default streams, graph replay with changed inputs, and tuned-name dispatch.
Device cases require gfx1250; the catalog tests do not launch GPU kernels.

## MiniMax M3 normal-TDM retune (2026-11-10)

This continuation focuses on **normal TDM and cluster split-K**, not HTI.
The original TDM kernel, candidate catalog and rounding contract above are
unchanged. The standard tuner now also builds FlyDSL candidates using the
actual FP16/BF16 input dtype, rather than hard-coding BF16.

All **122 existing gfx1250/256-CU BF16 rows** in
`minimax_m3_eagle_bf16_tuned_gemm.csv` were considered. This was a restricted
family sweep, not an exhaustive/all-backend retune: normal TDM only;
M tiles 16/64/128/256, N tiles 32/64/128/256, K tiles 128/256,
stages 2/3/4 and wave layouts 1x2/2x2/4x4 in the combinations selected by the
command below and the unchanged source pruner. Legal cluster factors were
1,2,3,4,6,8,9.

- 5679 standard-tuner candidates, all with positive final latency and
  passing the unchanged accuracy gate.
- Standard `--compare --update_improved` initially accepted 35 rows.
- Independent same-GPU production runs in before/after/after/before order
  retained only 24 rows with >=3% latency improvement in **both** comparisons.
- Only those 24 existing rows were updated; the other 98 gfx1250 rows and all
  163 other-architecture rows remain byte-for-byte unchanged.
- 21 updates use cluster split-K; three are non-split TDM at long K.
- Updated-row latency reduction: median 11.35%, range 3.92–36.91%, versus the
  previous production configs on the same GPU. This is not whole-model speedup.

Representative production latencies (mean of the two independent runs, us):

| M,N,K | split-K | previous config | normal TDM | reduction |
|---|---:|---:|---:|---:|
| 16,2304,6144 | 8 | 10.635 | 6.710 | 36.91% |
| 512,2304,6144 | 6 | 25.125 | 20.275 | 19.30% |
| 128,6144,18432 | 9 | 51.800 | 38.315 | 26.03% |
| 256,6144,18432 | 8 | 79.140 | 58.845 | 25.64% |
| 128,9216,6144 | 6 | 29.395 | 23.220 | 21.01% |
| 8192,6144,18432 | 1 | 1150.925 | 939.650 | 18.36% |

For the 21 cluster updates, production controls with the **same tile, waves
and stages**, changing only split-K to 1, were 1.59–3.84x slower (median 1.89x).
This supports the benefit of the complete K-parallel cluster reduction path;
it does not isolate the cost/benefit of `cluster_barrier` alone. Split partials
still round to input dtype, unlike a single FP32 sum. Winning `err_ratio`
values are 0–0.0142, below the unchanged 0.05 standard fraction threshold.
There is no tolerance relaxation or bitwise-equality claim.

A real tuning issue was also fixed: `post_process` accepted zero/non-finite
latencies as winners. It now requires finite positive time; invalid samples
remain available in raw profiles, with zero derived throughput metrics.
The fix has a regression test and does not change the timing method.

### Reproduction and verification

Start from an unmodified copy of the existing model table for the baseline;
use another full copy for the compare-gated output. Select only its 122
gfx1250/256-CU rows as the input. No new production model table is required.

    export ROCBLAS_USE_HIPBLASLT=0 PYTORCH_BLAS_PREFER_HIPBLASLT=0
    export AITER_CONFIG_GEMM_BF16=/absolute/path/to/baseline.csv
    python -m csrc.gemm_a16w16.gemm_a16w16_tune \
      --input_file all_gfx1250.csv --tuned_file gated.csv \
      --all --libtype flydsl --flydsl-policies tdm \
      --flydsl-kernel-filter 't(?:16x(?:32|64|128)x256x[234]|64x(?:32|64|128)x256x[23]|128x128x128x2|256x256x128x2)_.*_w(?:1x2|2x2|4x4)x1_' \
      --splitK --mp 4 --shape_grouped --batch 12 --timeout 1800 \
      --compare --update_improved --min_improvement_pct 3 \
      --profile_file profile.csv --verbose

The optional policy/name filters restrict only the standard candidate list;
the default catalog, reference, tolerances, timing and CSV selection remain
AITER's. Use `--run_config <csv>` for the repeated production checks.

The final updated table passed standard production validation for all 122
gfx1250 rows. The first attempt aborted in a BLAS reference kernel near an
unchanged wide-N row; the successful retry explicitly selected
`torch.backends.cuda.preferred_blas_library("cublas")` (rocBLAS on ROCm),
in addition to the environment above, before invoking the same tuner module.
This is an SDK workaround, not a change to the reference mathematics or
evidence that the library fault is fixed.

38 related tests passed, including normal BF16/FP16 device cases, tuned lookup,
standard half task creation, invalid-time rejection and compare logic.
No FP16 model rows were added by this BF16-table retune.

Local audit artifacts (not another shipped tuner/table) are in
`temp/minimax_normal/`: `tune_all.sh`, `all_profile.csv`,
`all_compare.txt`, `verified_improvements.csv`, `final_updates.csv`,
`same_tile_sk1.log`, `final_model_run_config_rocblas.log`,
and `normal_results.json`.
