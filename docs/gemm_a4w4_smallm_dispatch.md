# Dense MXFP4 small-M / narrow-N dispatch candidate

`gemm_a4w4` can now select the existing gfx950 Triton **preshuffle** GEMM.
It consumes the same 16x16-shuffled weight and shuffled E8M0 scales as
the original CK/ASM route. Only tensor views are created; activation
quantization and weight conversion are unchanged.

This candidate was selected from standalone measurements on October 30,
2026, then validated against the branch's parent main commit on November 2.
The measurements below cover this dispatch change, not a new kernel or
an end-to-end model speedup.

## Selection

| Input / configuration | Selected route |
| --- | --- |
| gfx950, 256 CUs; M=32..64, N=256..768, K=256..1024; N/K multiples of 256 | Existing preshuffle Triton config selection |
| Exact entries in `a4w4_triton_preshuffle_dispatch.csv` | Preshuffle Triton with the recorded tile/launch parameters |
| Other shapes, architectures, CU counts or incompatible operands | Original CK/ASM tuned selection |

All Triton routes require BF16 output, row-major packed A, preshuffled B,
shuffled per-1x32 E8M0 scales, contiguous buffers, M>=32, N divisible by 32
and K divisible by 256. Bias, nontrivial alpha/beta, preshuffled A and
NVFP4 global scales retain the original route. Missing Triton also retains
the original route; kernel failures are surfaced rather than silently hidden.

The exact configuration table initially contains M=32/64/128/256 with
N=512 and K=7168. These are exact M matches: padding buckets do not expand
the new routes to neighboring M values. The original CK/ASM table remains
available for every fallback, including calls with unsupported epilogues.

The table uses the existing config-file mechanism. To supply a different
table, set `AITER_CONFIG_GEMM_A4W4_TRITON_PRESHUFFLE` before importing AITER.
Restart before changing configurations, as with the original GEMM cache.
Launch dictionaries are copied before passing them to the Triton wrapper,
which mutates its config when computing split-K parameters.

## Evidence used to select this candidate

The historical runtime was MI355X / gfx950 / 256 CUs, AITER
`c90842add033fb3f9dbe0c6c53187a023faff1ee`, with BF16 output and HIP
per-1x32 MXFP4 quantization. Data was saved in the workspace under
`aiter_narrow_n_20261030/range_validation/evidence/sweep_v2_worker*.jsonl`.

For the short-K preshuffle range, the discovery sweep covered 36 shapes:
M=32/33/48/64, N=256/512/768, K=256/512/1024. All passed the independent
dequantized FP32 reference and matched valid default output bytes. The
minimum observed speedups were 1.178x for cold GEMM, 1.107x for dynamic
quantization plus cold GEMM, and 1.509x for hot GEMM. This discovery sweep
does not cover every integer M or K=768; the fresh full-range test does.

For N=512, K=7168, historical cold GEMM medians were:

| M | Default CK, us | Preshuffle Triton, us | Ratio |
| ---: | ---: | ---: | ---: |
| 32 | 24.03 | 21.52 | 1.12x |
| 64 | 24.12 | 20.65 | 1.17x |
| 128 | 22.96 | 15.55 | 1.48x |
| 256 | 22.67 | 16.88 | 1.34x |

The reported approximately 2.2x at M=128 came from **plain**, non-shuffled
Triton, which uses a different weight/scale layout. This patch does not
substitute plain Triton for preshuffled weights. Broader preshuffle
switches regressed at other K values, so longer K uses exact entries only.
These numbers describe the old discovery run, not a benchmark of this patch.

## Measured branch validation (November 2, 2026)

Environment: MI355X / gfx950 / 256 CUs, BF16 output, baseline main commit
`f7374b5d4caeab87b741b6977c2db93ca5dea0a6`, PyTorch
`2.10.0+rocm7.2.4.git3d3aa833`, HIP `7.2.53211`, Triton `3.8.0`.
The container image digest was
`sha256:9333a39842ab4364b75418832471b8f5cbcbf1fc6449928a5200b02503ae71d6`.
Core, quantization, CK and ASM extensions were compiled from the new
checkout with a fresh JIT cache. The dispatch and config SHA256 values
recorded by every row matched the source being published.

The complete run contains 406 shapes with two independent random seeds:
396 shapes covering the entire short-K selection range, four exact
N=512/K=7168 entries, and six fallback controls. All 812 shape/seed checks
passed finite-output, relative L2 < 0.01, bit-exact comparison of valid
output bytes with the original route, and graph-output validation. The
maximum observed relative L2 was 0.001781. Profiling confirmed the
expected routes for every shape; the baseline did not execute the
candidate Triton kernel. CPU routing tests also passed (31 tests), as did
the new config family's real config-merge/collision test.

Short-K results across both rounds:

| Mode | Geometric mean speedup | Minimum speedup |
| --- | ---: | ---: |
| Cold GEMM | 1.458x | 1.164x |
| Dynamic quantization + cold GEMM | 1.229x | 1.071x |
| Hot GEMM | 1.673x | 1.508x |

Cold GEMM for N=512, K=7168 (baseline / candidate medians):

| M | Round 1, us | Round 2, us | Speedup, rounds 1 / 2 |
| ---: | ---: | ---: | ---: |
| 32 | 24.030 / 21.575 | 24.069 / 21.464 | 1.114x / 1.121x |
| 64 | 24.055 / 20.731 | 24.143 / 20.794 | 1.160x / 1.161x |
| 128 | 23.036 / 15.588 | 22.931 / 15.585 | 1.478x / 1.471x |
| 256 | 22.584 / 16.893 | 22.545 / 17.035 | 1.337x / 1.323x |

For M=128, profiling identified the baseline as CK
`a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3`
and the candidate as `_gemm_afp4wfp4_preshuffle_kernel` with the config
recorded in this branch. Dynamic quantization plus cold GEMM measured
25.303 / 18.121 us and 25.318 / 18.119 us (about 1.40x).

All 400 shapes that changed routes improved in both rounds and all three
modes. Baseline-before/after drift stayed below 1% throughout the run.
Fallback controls retained the original GPU kernel names; their cold and
hot GEMM ratios stayed within 0.31% of parity.

One fallback control still needs investigation: M=512, N=512, K=7168 in
dynamic quantization plus cold GEMM measured 25.733 / 26.897 us and
25.656 / 26.907 us, ratios of 0.9567x / 0.9535x. Although the GEMM route
did not change, this is a repeatable 4-5% difference between separately
captured graphs in this run, not an established noise explanation or a
passed regression check. A follow-up comparison of identical graphs and
shared versus independent graph pools was prepared but did not execute
after the GPU allocation ended. This remains a limitation of the
candidate's regression evidence.

Raw results and environment/source manifests are retained in the workspace
under `aiter_narrow_n_20261030/dispatch_smallm_20261102/`.
`full_range_v1.jsonl` SHA256:
`ebe181c40a708fe4dd563d4d7881b80d5e457a833210023c21f7bffa35fe0f8b`.

## Reproducing validation

CPU routing coverage:

```bash
python -m pytest -q tests/test_gemm_a4w4_smallm_dispatch.py
```

Fresh standalone GPU coverage, using an installed checkout of this branch:

```bash
python op_tests/bench_gemm_a4w4_smallm_dispatch.py \
  --full-range --rounds 2 --samples 10 \
  --output smallm-dispatch-new-run.jsonl
```

The full range covers all integer M=32..64, all N=256/512/768 and all
K=256/512/768/1024, plus the four exact long-K entries and six controls.
It runs two independent seeds and at least ten samples per arm and mode.
The baseline temporarily disables only the new Python dispatch helper;
the original public `gemm_a4w4` and its CK/ASM config selection are preserved.
Both arms receive identical quantized inputs and layouts. Independent
profiling records the actual GPU kernels outside the timed region.

Measurements exclude JIT and weight preparation. Cold modes evict 512 MiB
before each sample and visit distinct weight addresses once per replay;
hot mode reuses one weight. The graph contains up to 64 calls. Dynamic
quantization is included only in `quant_gemm_cold`. Baseline/candidate/
baseline samples retain all raw latencies and both baseline medians.
Numerical checks require finite output, relative L2 < 0.01 against the
dequantized FP32 reference, valid-output byte equality with the original
route, and graph-output equality after timing.

Review both rounds for baseline drift and regressions, including the
unresolved quantization-plus-GEMM control above. M=31/65, larger N, K
outside the short range and M=512 retained the original route in this run.
Other GPU architectures, FP16, current ATOM nightly, the full ATOM Linear
wrapper and full-model accuracy/performance have not been validated by
these standalone measurements.
