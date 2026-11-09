# Orientation-aware A16W16 GEMM

The `gemm_a16w16_oriented` wrapper computes `X[M,K] @ W[N,K].T`.
It can instead compute `W @ X.T` and swap the output strides. Neither input
nor output is copied to transpose it. This changes the MFMA tile layout.

The wrapper uses the existing Triton GEMM and shared split-K reducer.
Partials use FP32. Bias is added once, on the final result.
The new row-bias and grouped split-K options default off in the shared kernel.

## Scope

- gfx950, FP16/BF16 inputs, same-dtype or FP32 output.
- Positive-stride two-dimensional inputs; contiguous output.
- `out=` writes into the supplied buffer without a copy.
- Configs use the standard `GEMM-A16W16-ORIENTED` JSON family.
- The normal Triton API and its default config selection do not change.

The multi-backend tuner tests this family alongside `triton/auto` for
`8 <= M <= 256` on gfx950. It stores the name `triton_a16w16_oriented`.
The name selects the family; tile values remain in JSON, not in the CSV.
Existing CSV rows are not changed by this PR.

## Run

    python op_tests/test_gemm_a16w16_oriented.py \
      -s 64,896,7168 128,896,7168 256,4480,7168 65,97,385 \
      --outdtype bf16 fp32 --layout contiguous strided --bias 0 1

This test uses a Torch FP32 reference without timing it.
Each candidate uses `run_perftest` and `checkAllclose`.
The table reports raw latency, TFLOPS, TB/s, and error.

    python op_tests/op_benchmarks/triton/bench_gemm_a16w16_oriented.py \
      --m 64 128 256 --n 896 --k 7168

The separate benchmark uses `RotatingBufferBench`.
The op test and rotating-buffer benchmark have different cache conditions.
Do not combine their absolute times.

## Evidence and limits

On gfx950 with 256 CUs, PyTorch 2.10.0+rocm7.2.4 and Triton 3.8.0,
the rotating-buffer benchmark measured:

| M | N | K | Existing Triton (us) | Oriented (us) |
|---:|---:|---:|---:|---:|
| 64 | 896 | 7168 | 22.8885 | 11.4266 |
| 128 | 896 | 7168 | 33.4679 | 9.56048 |
| 256 | 896 | 7168 | 46.0322 | 12.6070 |

These compare the default Triton path and the new family on the same device.
They do not compare all AITER backends or claim model-level speedup.
The earlier tuning study used an older branch. Its numbers are not evidence
of speedup over the current AITER main. No B300 comparison is claimed here.
Other GPU architectures have not been tested for this family.
