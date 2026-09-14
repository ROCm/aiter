# FlyDSL MXFP8 GEMM (gfx950)

The layout-dynamic kernel is ported from `flydsl-examples/kernels/scaled_gemm_gfx950.py`.
It uses scaled MFMA, full-tile or half-tile-interleaved (HTI) pipelines, split-K,
and workgroup-local slice-K. FP32 output and slice-K cshuffle retain FP32
intermediates; BF16 split-K still uses BF16 atomic accumulation.

## Operand contracts

All data tensors are `torch.float8_e4m3fn`. Scale tensors are
`torch.float8_e8m0fnu` or bit-identical `torch.uint8` E8M0 bytes.

| Mode | A | B | A scale | B scale |
|---|---|---|---|---|
| Native MXFP8, `scale_block=32` | `[M,K]` | `[N,K]` | `[M,K/32]` | `[N,K/32]` |
| E8M0 blockscale, `scale_block=128` | `[M,K]` | `[N,K]` | `[M,K/128]` | `[ceil(N/128),K/128]` |

Output is BF16 (default) or FP32; optional bias has the output dtype and shape
`[N]`. A caller-owned contiguous `out=[M,N]` is supported.

`bpreshuffle=True` consumes `shuffle_weight(B, layout=(16,16))` **directly**.
No inverse shuffle or scale expansion runs per GEMM. Scales remain unshuffled:
do **not** pass MoE `shuffle_scale`/`e8m0_shuffle` or PR #4254's compact shuffled
scale buffers to this API. The PR was used as a reference for isolating the
microscale tuned table; its separate kernel/packed-scale ABI is not imported.

Supported dimensions/configs are validated before launch:

- Positive M/N/K; vector-aligned N (normally a multiple of 8 for BF16,
  4 for FP32). Preshuffled B requires N divisible by 16.
- K divisible by the scale block and the selected tile/partition alignment.
  No K-tail; a split partition must have enough tiles for `stages`.
- HTI requires two stages, two M waves, no slice-K, and an even K-tile count
  per split partition.
- Only gfx950 and the above FP8/scale/output types are supported by this backend.
  Existing gfx1250 kernels and ordinary FP32-scale blockscale dispatch are unchanged.

## Runtime

Native 1x32 MXFP8 through the existing tuned GEMM interface:

    from aiter.tuned_gemm import tgemm
    from aiter.ops.shuffle import shuffle_weight

    w_shuffled = shuffle_weight(w, layout=(16, 16))
    y = tgemm.mm(a, w_shuffled, scale_a=sa, scale_b=sw, bpreshuffle=True)

Use explicit `bpreshuffle` under `torch.compile`: Python tensor attributes
are not an operator-schema layout contract. Eager calls still recognize
`is_shuffled`. Batched leading dimensions on A/scales are flattened/restored.

The existing DSv4 model call needs no new packed-scale preprocessing:

    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_bpreshuffle
    # a, sa = per_group_quant_hip(..., group_size=128,
    #         scale_type=dtypes.fp8_e8m0, transpose_scale=True)
    y = gemm_a8w8_blockscale_bpreshuffle(a, w_shuffled, sa, sw, out=out)

On gfx950, BF16-output calls with **both** scales E8M0/uint8 use this backend.
The MXFP8 Python entry uses separate functional and output-writing custom ops
for inductor. It views E8M0 scales as uint8 without copying before a mutating
custom-op boundary, avoiding an E8M0 functionalization failure in some PyTorch
versions. The Python API is unchanged; both variants share the existing backend dispatch.

`sa` has the existing byte-transposed layout emitted by
`per_group_quant_hip(transpose_scale=True)`; `sw` is row-major block128 scale.

For explicit tile policies or logical (rather than byte-packed) scales:

    from aiter.ops.flydsl import flydsl_mxfp8_gemm
    y = flydsl_mxfp8_gemm(a, w, sa, sw, config=tile, out=out,
                          scale_block=32, bpreshuffle=False)

## Tuning

A separate table avoids collisions with tensor/per-token FP8 and FP32 blockscale:
`aiter/configs/mxfp8_tuned_gemm.csv`. Override it with
`AITER_CONFIG_GEMM_MXFP8=/path/to/tuned.csv`.

Keys include architecture, CU count, M/N/K, output dtype, bias, scale block,
weight preshuffle, and A-scale transpose. Stable `flydsl_mxfp8_*` kernel names
include tile/MMA/pipeline/layout parameters. `ks1` selects the non-split
kernel; `ksd` selects the dynamic split kernel. The actual partition count is
stored **only in the CSV `splitK` column**, and runtime/AOT both pass it as an
Int32 launch argument. Names and compile signatures are identical for split
counts 2, 4, 7, etc.; a name/splitK mismatch is rejected.

Invalid or mismatched rows
fall back only to a validated MXFP8 config, never ordinary unscaled GEMM.

    python csrc/gemm_mxfp8/gemm_mxfp8_tune.py \
      -i aiter/configs/mxfp8_untuned_gemm.csv -o /tmp/mxfp8_tuned.csv

    python csrc/gemm_mxfp8/gemm_mxfp8_tune.py \
      --run_config /tmp/mxfp8_tuned.csv

The tuner uses `GemmCommonTuner`/`mp_tuner`, with accuracy checks, dirty output
buffers, full-tile/HTI, split-K/slice-K candidates and standard profiling,
retuning and compare options. It currently searches the FlyDSL backend only.
DSv4 shapes and results are in `aiter/configs/model_configs/`:
`dsv4_mxfp8_untuned_gemm.csv` and `dsv4_mxfp8_tuned_gemm.csv`. They cover all
204 shapes in the existing `dsv4_a8w8_blockscale_untuned_gemm.csv`: 68 token
counts from M=1 to M=32768, with (N,K)=(768,7168), (2048,7168), (7168,384).
All use BF16 output, block128 E8M0, native B preshuffle, and byte-transposed A
scales. This is an operator sweep, not a full-model inference run.

Reproduce the tuning sweep (shape-grouped on one gfx950 GPU):

    python csrc/gemm_mxfp8/gemm_mxfp8_tune.py \
      -i aiter/configs/model_configs/dsv4_mxfp8_untuned_gemm.csv \
      -o /tmp/dsv4_mxfp8_tuned.csv --mp 1 --shape_grouped \
      --screen-topk 8 --batch 4 --warmup 2 --iters 11

The catalog shares HGEMM's search axes: M tiles 16/32/48/64/80/96/128/256,
N tiles 16/32/64/80/96/128/256, stages 2..9, M/N waves 1/2/4, group_m 0/4,
FT/HTI. MXFP8 adapts K tiles to 128/256/512 and slice-K (`k_waves`) to 1/2/4.
Split-K searches 1 and divisors of K from 2..9 (including 7 for K=7168 and 3 for
K=384), then applies partition/alignment/LDS/grid checks. Slice choices are not
discarded merely for equal estimated occupancy.

`--screen-topk 8` evaluates the entire legal space using graph-event timing,
then sends up to eight candidates **per split/slice regime** to the standard
rotating-buffer `mp_tuner` profiler. Zero disables screening and profiles the
whole space. Only standard profiler timings go into the CSV. BF16 split-K
finalists also undergo 16 repeated accuracy checks outside timing to reject
order-sensitive atomic reductions. Successful rows are resumable.

For the 204 DSv4 shapes the expanded space has 362036 shape/config candidates
(4518 unique compile configurations), including 152782 split-K and 100028
slice-K candidates. The original M=1 singleton-scale regression remains covered.


## AOT and tests

The default GEMM AOT job set includes the new tuned table (including
model-specific `*mxfp8_tuned_gemm*.csv` merges). Runtime and AOT share the same
layout-dynamic launch argument builder. AOT allocates tiny CPU tensors, not
model-sized GPU buffers.

    AITER_AOT_IMPORT=1 GPU_ARCHS=gfx950 HIP_VISIBLE_DEVICES='' \
      FLYDSL_RUNTIME_CACHE_DIR=/tmp/mxfp8_cache \
      python -m aiter.aot.flydsl.gemm --csv /tmp/mxfp8_tuned.csv

    FLYDSL_RUNTIME_CACHE_DIR=/tmp/mxfp8_cache FLYDSL_RUNTIME_RUN_ONLY=1 \
      AITER_CONFIG_GEMM_MXFP8=/tmp/mxfp8_tuned.csv \
      python csrc/gemm_mxfp8/gemm_mxfp8_tune.py --run_config /tmp/mxfp8_tuned.csv

    pytest -q op_tests/flydsl_tests/test_mxfp8_integration.py
    python op_tests/test_flydsl_mxfp8.py -s 1,768,7168 32,2048,7168 4096,7168,384 \
      -l preshuffle --scale-block 128

AOT coverage is the selected tuned policies. For deployment, tune/include all
required shapes; run-only mode intentionally fails for missing artifacts.
The top-level op test follows the standard `@benchmark` / candidate dictionary /
`run_perftest` / `checkAllclose` / markdown summary structure. Kernel, tuner,
inductor and AOT regressions are separate in
`op_tests/flydsl_tests/test_mxfp8_integration.py`.

Tests cover native/preshuffled weights, actual quantizer outputs, scale dtype
and layout, tails, stage/scale-chunk wrap, bias, output reuse, stream/graph,
`torch.compile`, config isolation, tuner CSV roundtrip, CPU-only AOT and fresh
process run-only execution, including negative cache-miss checks.

Dense random FP8 inputs with tiny products need a small absolute FP32 tolerance
against dequantized GEMM. A separate binary-exact-input FP64-reference test
checks that FP32 output is not truncated through BF16 cshuffle.


## Previous restricted-space baseline (gfx950, 2026-09-14)

All 204 shapes were tuned and then verified through the model entry with
preallocated output and `FLYDSL_RUNTIME_RUN_ONLY=1`: all errors were zero.
CPU-only AOT compiled 204 jobs successfully. This run used 11 timing iterations
during tuning; the standard op-test verification used its normal 101 iterations.
The table below is the raw verification subset for M=1/32/256/4096, not the
tuner's shorter timing pass.

The sweep exposed and fixed a singleton scale-stride issue at M=1. A BF16
split-K=4 candidate also failed repeat validation and was replaced by a stable
split-K=2 candidate after retuning; the test tolerance was not widened.

|    m |    n |    k | dtype          | layout     |   scale_block | gfx    |   flydsl us |   flydsl TFLOPS |   flydsl TB/s |   flydsl err |
|-----:|-----:|-----:|:---------------|:-----------|--------------:|:-------|------------:|----------------:|--------------:|-------------:|
|    1 |  768 | 7168 | torch.bfloat16 | preshuffle |           128 | gfx950 |    12.303   |        0.894906 |      0.448192 |            0 |
|   32 |  768 | 7168 | torch.bfloat16 | preshuffle |           128 | gfx950 |    18.558   |       18.9849   |      0.311762 |            0 |
|  256 |  768 | 7168 | torch.bfloat16 | preshuffle |           128 | gfx950 |    24.182   |      116.556    |      0.3204   |            0 |
| 4096 |  768 | 7168 | torch.bfloat16 | preshuffle |           128 | gfx950 |    62.3906  |      722.82     |      0.663342 |            0 |
|    1 | 2048 | 7168 | torch.bfloat16 | preshuffle |           128 | gfx950 |    18.4547  |        1.59093  |      0.796129 |            0 |
|   32 | 2048 | 7168 | torch.bfloat16 | preshuffle |           128 | gfx950 |    18.7208  |       50.1861   |      0.803555 |            0 |
|  256 | 2048 | 7168 | torch.bfloat16 | preshuffle |           128 | gfx950 |    29.7902  |      252.304    |      0.590089 |            0 |
| 4096 | 2048 | 7168 | torch.bfloat16 | preshuffle |           128 | gfx950 |    84.372   |     1425.34     |      0.723554 |            0 |
|    1 | 7168 |  384 | torch.bfloat16 | preshuffle |           128 | gfx950 |     4.54151 |        1.21216  |      0.609357 |            0 |
|   32 | 7168 |  384 | torch.bfloat16 | preshuffle |           128 | gfx950 |     4.785   |       36.8152   |      0.673734 |            0 |
|  256 | 7168 |  384 | torch.bfloat16 | preshuffle |           128 | gfx950 |     6.63331 |      212.456    |      0.983185 |            0 |
| 4096 | 7168 |  384 | torch.bfloat16 | preshuffle |           128 | gfx950 |    30.5038  |      739.206    |      2.06722  |            0 |
