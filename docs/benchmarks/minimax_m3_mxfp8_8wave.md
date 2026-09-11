# MiniMax M3 MXFP8 eight-wave MoE prefill

The native AITER path improves the four MiniMax M3 16k/32k prefill cases by
**1.14–1.31x on the same inputs**. TP8 (H=6144, I=384, E=129, topk=5) at 32k
improves from **2851.6 to 2347.5 us**, including GPU sorting. These are measured
GPU kernel durations, not serving wall-clock latency.

| Tokens | I per rank | Original (us) | Eight-wave (us) | Speedup | GEMM1/GEMM2 swizzle |
| ---: | ---: | ---: | ---: | ---: | --- |
| 16384 | 384 | 1567.5 | 1314.3 | 1.193x | 3 / 0 |
| 32768 | 384 | 2851.6 | 2347.5 | 1.215x | 1 / 3 |
| 16384 | 768 | 2182.1 | 1908.2 | 1.144x | 3 / 0 |
| 32768 | 768 | 4505.6 | 3437.1 | 1.311x | 1 / 3 |

Measured on 2026-09-11, gfx950 with 256 CUs, ROCm 7.2.4, PyTorch
2.10.0+rocm7.2.4.git3d3aa833. AITER base: `8f5cb975f`. The source dense eight-wave
pipeline is from ROCm/FlyDSL `98c47b58`, following dense merge `578cfc7a`.

## Implementation and supported contract

The new `flydsl_moe{1,2}_mxfp8_8w_t256x256_xcd{S}` family uses the standard
AITER sorting ABI. GEMM1 quantizes each BF16 source token once, scatters E8M0
scales into expert order, gathers FP8 activations, and fuses MiniMax clamped
SwiGLU (alpha 1.702; configurable clamp, default 7). It returns sorted FP8
intermediate activations and scales. GEMM2 writes sorted BF16 partials, then
reduces routing-weighted top-k in FP32 into the caller's BF16 output.

Weights retain AITER's G1U1 16-row gate/up interleave and 16x64 preshuffle.
GEMM2 uses A stride 512, physical **B stride 384**, and only three K128 compute
iterations for TP8. No per-call repacking or host readback of the GPU's valid
sorted row count is required. Upper-bound workspaces and a uniform CTA guard
support changing routing inside a captured graph. Both stages sort in blocks
of 256. The 128x512 GEMM1 candidate is available to the tuner but loses here.

The CSV dispatch is limited to BF16 input/output, FP8 E4M3 A/W with per-32 E8M0
scales, interleaved G1U1 Swiglu, and stage-2 routing weights. It requires
H divisible by 256, I divisible by 128 and at least 256, no model/intermediate
padding, and no expert bias. Unsupported explicitly selected configurations
raise an error. The changed CSV rows are only T=16384/32768, I=384/768.
Other MiniMax rows retain their existing kernels.

The grouped pipeline keeps the validated `vmcnt(2)` allowance before LDS
rotation and the epilogue wave-group barrier. Loosening the wait to the dense
pipeline's `vmcnt(6)` previously caused races around routed/padded rows.

## Reproduction and numerical checks

The [sweep results](minimax_m3_mxfp8_8wave/integrated_sweep.json) record all 72
candidates, input seed, selected wrappers, total times, and full-output
comparisons. Each of the four shapes sweeps GEMM1 tiles 256x256 and 128x512 and
swizzles 0/1/2/3/4/8; GEMM2 sweeps the same swizzles with tile 256x256. Routing
is random; `--balanced` enables an additional cyclic expert distribution.
All candidates compare against the original kernel on the same tensors, and
the selected paths produce identical full outputs on three repeated launches.
Full-output normalized differences from the original are 1.18e-5–1.49e-5.

```bash
# Use the saved original CSV as the baseline; omit --write to inspect first.
python csrc/ck_gemm_moe_2stages_codegen/tune_mxfp8_8wave.py \
  --csv docs/benchmarks/minimax_m3_mxfp8_8wave/aiter_original.csv \
  --tokens 16384 32768 --output /tmp/minimax_sweep.json

AITER_CONFIG_FMOE=aiter/configs/model_configs/minimax_m3_mxfp8_tuned_fmoe.csv \
  python op_tests/test_moe_2stage.py --no-legacy --csv-filter mxfp8_8w

python -m pytest op_tests/flydsl_tests/test_mxfp8_moe_8wave.py -q
```

The tuning script's `--write` updates the selected rows in its input CSV.
CSV `us1` and `us2` time the complete stage wrappers, including stage-1
quantization and stage-2 reduction; `us` separately measures the full pipeline
including sorting. Independently timed wrappers can differ from their summed
pipeline time due to cache state. Historical CSV microseconds were measured
elsewhere and must not substitute for a reproduced baseline on this device.
Allocation/CPU dispatch, router top-k, compilation and weight packing are not
included in the sum of GPU durations. Inputs and weights are fixed during timing.

`test_moe_2stage.py` now quantizes MXFP8 activations at both reference stages,
matching the runtime instead of passing BF16 through. Its existing accuracy
thresholds are unchanged. All four final CSV cases passed: normalized error
3.998e-6–4.015e-6. Elementwise `checkAllclose` still reports differences from the
Torch reduction/rounding convention; the test's existing normalized-error
acceptance passes. The [original test timings](minimax_m3_mxfp8_8wave/aiter_before_bench.csv)
and [final test timings](minimax_m3_mxfp8_8wave/aiter_final_bench.csv) reproduce
TP8 32k at 2906.9 → 2453.8 us and I768 32k at 4547.6 → 3526.9 us.
The seed-42 sweep above provides the controlled identical-input comparison.

Additional tests cover both GEMM1 tiles, K384 weight stride, changing GPU
valid-row counts, unused experts, nondefault clamp, full-output repeatability,
and graph replay after routing changes. The shared FlyDSL kernel suite passes
41 tests (25 MoE, 16 dense). Eight AITER native stages were precompiled into a
fresh independent cache and successfully loaded with
`FLYDSL_RUNTIME_RUN_ONLY=1` (all four shapes, both stages); the normal sorter
was prepared outside that stage-only check.

```bash
python -m aiter.aot.flydsl.moe \
  --csv aiter/configs/model_configs/minimax_m3_mxfp8_tuned_fmoe.csv
# After AOT compilation, verify one selected native stage:
python csrc/ck_gemm_moe_2stages_codegen/profile_mxfp8_8wave.py \
  --csv aiter/configs/model_configs/minimax_m3_mxfp8_tuned_fmoe.csv \
  --tokens 32768 --inter-dim 384 --stage 2 --run-only-stages
```

## Equivalent expert GEMM and activation controls

For TP8 32k balance mode, each of the 129 experts receives 1270 or 1271 routed
rows (1280 after padding). GEMM1 has **N=2I=768, K=6144**, followed by SwiGLU
reducing to I=384. Expert count is a batch dimension: concatenating expert
weights into N=129*768 computes unrouted products and changes the useful work.

The shared FlyDSL core gives these controls (useful FLOPs exclude padded rows
and K; activation/quantization/reduction are excluded from the numerator):

| Control | Time (us) | Useful TFLOPS |
| --- | ---: | ---: |
| Grouped gathered GEMM1 + SwiGLU, 256x256 | 728.2 | 2123.2 |
| Same expert weights/routes, without activation | 710.0 | 2177.7 |
| Same grouped GEMM, sorted contiguous A, no activation | 761.0 | 2031.9 |
| Dense same total rows/N/K, sharing one expert's weight | 719.7 | 2148.3 |
| Single expert dense M1280/N768/K6144 | 52.5 | 228.1 |

The no-activation kernel stores twice as many BF16 columns, so its 18.2 us
difference is not a pure isolated activation latency. Single-expert dense
launches just 15 CTAs on 256 CUs and cannot predict batched throughput. The tall
dense control has different weight reuse, but confirms that the grouped path
is close to dense throughput for this matrix geometry. See
[equivalent_tp8.json](minimax_m3_mxfp8_8wave/equivalent_tp8.json).

128x512 is slower: TP8 balance GEMM1 is 1252.8 us (1234 TFLOPS), versus 727.0 us
(2127 TFLOPS) for 256x256 in the direct tile comparison. Projection N768 takes
two N512 tiles, computing 1024 columns; LDS also rises from 128 to 160 KiB.
Even when I768 makes N1536 divisible by 512, 128x512 is slower (1848.7 versus
1370.8 us). It is retained as an explicit measured candidate, not the default.

## Native AITER thread trace and PMC

The installed `flyprof` capture/analyze/report workflow captured each native
stage independently, with ATT and PMC in separate passes. It selected the real
`kernel_gemm_0`, with scaled FP8 MFMA instructions and AITER source mapping.
[Compact evidence](minimax_m3_mxfp8_8wave/native_profile.json) includes the
selected dispatch, counter ratios, stall taxonomy, and waitcnt producer edges.

| Evidence | GEMM1 | GEMM2 |
| --- | ---: | ---: |
| Source mapping | 100.0% rounded | 99.9% |
| L2 hit rate | 66.2% | 60.4% |
| Barrier share of attributed stall cycles | 30.82% | 29.39% |
| VM-wait share of attributed stall cycles | 17.43% | 16.82% |
| MFMA share of attributed stall cycles | 28.16% | 15.18% |
| Allocated VGPRs/thread | 256 | 251 |
| LDS/CTA | 128 KiB | 128 KiB |

The main remaining opportunities are:

1. Reduce GEMM2 partial-output/reduction traffic. The 32k TP8 path materializes
   over 2 GB of sorted BF16 partials, then gathers them for reduction. Reduction
   alone is roughly 0.46–0.49 ms in the prepared benchmark. A different output
   layout or fused reduction could improve the whole pipeline; prior BF16
   atomic-output experiments were slower, so fusion alone is not sufficient.
2. Design a short-K pipeline with less LDS/register state. GEMM2 does only three
   K128 iterations, so startup and barriers are expensive relative to MFMA.
   The current 128 KiB LDS allocation permits one CTA per CU. Smaller footprints
   may improve latency hiding, but must be checked against extra operand traffic.
3. Revisit overlap of scale and payload loads while preserving LDS lifetimes.
   The trace's initial `vmcnt(4)` waits on six scale loads and the initial
   global-to-LDS loads; `vmcnt(2)` and barriers still contribute substantially.
   A barrier percentage is not proof that a barrier can safely be removed.

These are directions supported by the profile, not promised speedups. Stall
attribution ratios are not percentages of wall-clock time saved by deletion.
Several debug locations collapse to the enclosing `if active`; use the ISA
and producer edges rather than interpreting that Python predicate as the cost.
The tool incorrectly infers BF16 compute from output, so its automatic roofline
is excluded. EA0 bandwidth is an uncalibrated channel lower bound, and no valid
LDS bank-conflict rate was captured. GEMM2 capture uses a synthetic sorted
intermediate; full pipeline timings and correctness use real GEMM1 output.

To capture the native path with a FlyDSL checkout available:

```bash
flyprof capture mxfp8_moe_8wave --worktree /path/to/FlyDSL \
  --invocation 'python /path/to/aiter/csrc/ck_gemm_moe_2stages_codegen/profile_mxfp8_8wave.py --csv /path/to/aiter/aiter/configs/model_configs/minimax_m3_mxfp8_tuned_fmoe.csv --stage 1' \
  --tag big --with-pmc --gpu 0 --bundle /tmp/m3_native_s1 --timeout 600 -f json
flyprof counters --bundle /tmp/m3_native_s1 -f json
flyprof bubbles --bundle /tmp/m3_native_s1 -f json
flyprof map --bundle /tmp/m3_native_s1 -f json
```

Repeat stage 2 in a separate bundle. Full artifacts from this run are under
`/tmp/flydsl_mxfp8_moe/profile_native_stage{1,2}` on the benchmark machine.
