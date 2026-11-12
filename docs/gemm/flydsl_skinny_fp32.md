# FP32 output from FlyDSL skinny HGEMM

The gfx950 `sk` policy computes BF16 `A[M,K] @ B[N,K].T`.
It now writes BF16 or FP32 output. The default is still BF16.
If `out_dtype` is absent, a supplied `out` buffer sets the output dtype.

The kernel already sums its wave-private partials in FP32. The new path
stores that sum without BF16 rounding. It needs no extra cast or reduce
kernel, semaphore, or persistent scratch.

Both epilogues support FP32:
- `block_m <= 16`: one four-element FP32 store.
- `block_m > 16`: two four-element FP32 stores, not one 256-bit store.

The output descriptor uses the output element size. The compile key and
kernel symbol distinguish FP32 from BF16. Input dtype, bias, shape, and
architecture restrictions do not change.

The public FlyDSL name parser, tuner catalog and production dispatch pass
the output dtype through. FP32 skinny is a candidate, not a new default.
No tuned CSV is added. Existing BF16 names keep their spelling.
AOT replays the public wrapper and therefore needs no separate signature.

## Validation

Run with the repository's pinned FlyDSL 0.4.2:

    python op_tests/test_flydsl_hgemm.py \
      -s 1,896,7168 7,896,7168 16,896,7168 23,896,7168 32,896,7168 \
         5,4104,2048 17,6144,2048 8,4096,4096 --out-dtype bf16 fp32
    pytest -q op_tests/flydsl_tests/test_hgemm_skinny_output.py

The existing op test keeps its candidate loop, `run_perftest`,
`checkAllclose`, and markdown table. For FP32, Torch is an untimed
reference. Tests cover output guards, masked M rows, both epilogues, dtype
cache separation, inferred output dtype, changed-input graph replay,
streams, tuning selection and production `gemm_a16w16_out`.

FP32 checks use `atol=5e-4`, `rtol=2e-4`, zero bad elements, and normalized
RMSE below `2e-5`. Existing BF16 tolerances are not widened.
The >2 GiB weight and existing strided-input checks remain enabled.

## Measured scope

MI355X/gfx950, 256 CUs, PyTorch 2.10.0+rocm7.2.4, FlyDSL 0.4.2.
Three runs of the op test, median `run_perftest` latency, FP32 output:

| M | N | K | FlyDSL FT (us) | FlyDSL skinny (us) |
|---:|---:|---:|---:|---:|
| 1 | 896 | 7168 | 5.37990 | 4.52225 |
| 7 | 896 | 7168 | 5.46107 | 4.58813 |
| 16 | 896 | 7168 | 5.52864 | 5.41385 |
| 23 | 896 | 7168 | 5.84342 | 7.04266 |
| 32 | 896 | 7168 | 5.80348 | 10.63629 |

The compared policies and their tiles are in `test_flydsl_hgemm.py`.
Both compute the full FP32 result into a preallocated output.
Skinny wins only for part of this range. The tuner must retain FT too.

These are same-device candidate comparisons, not all-backend or model
end-to-end speedups. There is no B300 claim. Other architectures are
unverified. Five BF16 shapes were also checked against the base revision:
outputs were bitwise equal; sequential timing is not an ISA-equivalence claim.
