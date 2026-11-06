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

## Reproduce tuning

From the AITER repository in an environment with its FlyDSL dependency installed:

    ROCBLAS_USE_HIPBLASLT=0 \
    python -m csrc.gemm_a16w16.tune_gfx1250_tdm \
      --all-bias --compile-workers 16 \
      --output /tmp/gfx1250_tdm.csv --details /tmp/gfx1250_tdm_details

Existing output files are not overwritten. To continue a stopped run, pass
`--resume` with the same output, details path, seed, split mode and environment.
Completed rows are reused only if their per-candidate JSON evidence and current
source space match; interrupted shapes are rerun from the beginning.

This covers the source `tune_all` list of 19 `(M,N,K)` shapes, independently for
bias off/on. `--shape M N K` selects one shape. `--no-split-k` restricts enumeration
to the source non-split mode; otherwise the source's valid factors from 1 to 9
and original pruning are used. No candidate-space reduction is added.
`--shard I --shards N` divides the shape list; give each shard a different output
and details path, and run GPU timing shards on different devices.

The normal AITER GEMM tuner also sees `tdm` candidates on gfx1250 for matching
input/output dtypes, alongside its existing policies. The dedicated script above
isolates the imported source space and checks its split-partial rounding semantics.

Each candidate is compiled, run through AITER's `flydsl_hgemm`, and checked
against a source-order reference on nonzero quantized random data.
Compile/validation failures are recorded, not included as winners.
Initial timing uses three graph samples of eight calls. The fastest eight
candidates are remeasured with nine graph samples of twenty calls each; selection
uses the median. A selected winner additionally passes a fresh normal-random
input check with `atol=rtol=0.05`.

The CSV uses AITER's existing BF16 GEMM schema; `kernelName` carries `ptdm`,
the full tile/wave/group/split configuration and architecture. `solidx` is the
source-space index and is informational for FlyDSL dispatch.
`err_ratio=0` refers to exact equality on the quantized-data check, not bitwise
equality to a fused FP32 reference for arbitrary inputs.
JSON evidence includes every candidate and the finalist samples.

**Timing scope:** graph-event GPU time at fixed input addresses, including the
kernel's bias or cluster-reduction work. CPU dispatch, compilation, tensor
allocation and input/reference generation are outside the timed graph.
This is not an end-to-end application, cold-cache or rotating-input benchmark.
Shared-device load and clocks can affect selection; rerun on the deployment
device before treating a table as a performance baseline.

## Load and verify a table

The shipped opt-in table is
`aiter/configs/gfx1250_tdm_bf16_tuned_gemm.csv` (19 shapes, bias off/on).
See [measured results and validation](gfx1250-tdm-hgemm-results.md).

Default AITER model tables are not replaced. Load a generated or shipped table explicitly,
before importing AITER:

    export AITER_CONFIG_GEMM_BF16=/tmp/gfx1250_tdm.csv
    ROCBLAS_USE_HIPBLASLT=0 \
    python -m csrc.gemm_a16w16.validate_tdm_csv

The validator checks each row through `aiter.tuned_gemm.gemm_a16w16` and
`gemm_a16w16_out`, asserts that dispatch retains `policy="tdm"`, and replays a
captured graph after changing the inputs.

Application usage after setting that environment variable:

    from aiter.tuned_gemm import gemm_a16w16
    out = gemm_a16w16(a, w, bias)

## Regression tests

    python -m unittest \
      op_tests.tuning_tests.test_a16w16_tuned_config_lookup \
      op_tests.flydsl_tests.test_gemm_a16w16_tdm -v

The device tests cover BF16/FP16, NT/NN weights, padded input/output strides,
masked M/N, small M, group swizzle, bias rounding, cluster split-K (1/2/3/9),
non-default streams, graph replay with changed inputs, and tuned-name dispatch.
Device cases require gfx1250; the catalog tests do not launch GPU kernels.
