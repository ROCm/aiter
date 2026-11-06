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
  \
      --input_file "$config" --tuned_file "$config" \
  \
      --all --libtype all --splitK --mp 3 --shape_grouped --batch 3 \
  \
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
  \
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
