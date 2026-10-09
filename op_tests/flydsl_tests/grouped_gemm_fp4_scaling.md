# Grouped GEMM FP4 block scales on gfx1250

Measured on 2026-10-09, gfx1250 with 256 CUs, PyTorch 2.11.0 + ROCm 10.1.
The problem is 96 experts, 512/2048/16384 tokens, topk 6, model dimension 7168,
intermediate dimension 3072, SiLU, no bias, and balanced routing.

All modes use the same existing CSV setting: **64 × 512 × 128** for both
GEMMs, waves 1 × 4, two buffers, cluster 1, two waves per TDM tensor, and
no next-stage prefetch. The CSV has not been modified. Scale size and format
do not select different tiles.

## Performance

The first tables cover 512 tokens; larger token counts follow below.

Median of five interleaved rounds. Each profiler measurement uses the same
`run_perftest(..., num_warmup=5, num_iters=101, testGraph=False)` as the
requested `--scenario kernel` command.

| Mode | GEMM1 (µs) | GEMM2 (µs) | Sum (µs) | Sum vs scale32 |
|---|---:|---:|---:|---:|
| Scale32, E8M0 | 158.86 | 93.28 | 252.14 | baseline |
| Real scale16, E8M0 | 157.77 | 96.57 | 254.34 | +0.87% |
| NVFP4, scale16 E4M3 with float scales | 159.44 | 96.72 | 256.16 | +1.59% |

The sum is the two isolated GEMM medians; it excludes routing, quantization,
and gather/reduce. In this profiler measurement, scale16 GEMM1 is 0.68%
faster and GEMM2 is 3.52% slower than scale32.

Graph/event timing gives a second view of device performance. Each graph has
16 repeated launches, three warmup replays, and five event samples; the table
uses the median across five interleaved rounds.

| Mode | GEMM1 (µs) | GEMM2 (µs) | Sum (µs) | Sum vs scale32 |
|---|---:|---:|---:|---:|
| Scale32, E8M0 | 151.57 | 79.23 | 230.81 | baseline |
| Real scale16, E8M0 | 158.76 | 82.14 | 240.90 | +4.37% |
| NVFP4, scale16 E4M3 with float scales | 159.14 | 84.25 | 243.39 | +5.45% |

Profiler timings are close, but graph timing still shows overhead with the
same tiles. These results do not establish exact equality. All samples and
launch settings are in [grouped_gemm_fp4_scaling_results.json](grouped_gemm_fp4_scaling_results.json).

Separate CLI invocations varied substantially: scale32 GEMM1 ranged from
159.07 to 184.11 µs, and scale16 from 176.24 to 186.78 µs in the observed
runs. The interleaved medians above compare modes within one prepared
process; individual CLI timings can differ from them.

## Larger token counts

The 2,048 and 16,384-token cases also use **64 × 512 × 128** for both
GEMMs, as specified by their existing a4w4 CSV rows. All three scale modes
use identical tiles, wave counts, buffers, and scheduling settings. Measurements
use the same five-round interleaved procedure as the 512-token case.

Profiler timing:

| Tokens | Mode | GEMM1 (µs) | GEMM2 (µs) | Sum (µs) | Sum vs scale32 |
|---:|---|---:|---:|---:|---:|
| 2048 | Scale32 E8M0 | 253.65 | 151.27 | 404.93 | baseline |
| 2048 | Real scale16 E8M0 | 269.64 | 150.77 | 420.41 | +3.82% |
| 2048 | NVFP4 + float scales | 276.20 | 150.25 | 426.45 | +5.32% |
| 16384 | Scale32 E8M0 | 1485.17 | 808.78 | 2293.95 | baseline |
| 16384 | Real scale16 E8M0 | 1604.70 | 842.39 | 2447.09 | +6.68% |
| 16384 | NVFP4 + float scales | 1637.34 | 851.80 | 2489.14 | +8.51% |

Graph/event timing:

| Tokens | Mode | GEMM1 (µs) | GEMM2 (µs) | Sum (µs) | Sum vs scale32 |
|---:|---|---:|---:|---:|---:|
| 2048 | Scale32 E8M0 | 256.17 | 142.02 | 398.18 | baseline |
| 2048 | Real scale16 E8M0 | 275.26 | 148.07 | 423.32 | +6.31% |
| 2048 | NVFP4 + float scales | 276.79 | 149.37 | 426.16 | +7.03% |
| 16384 | Scale32 E8M0 | 1488.85 | 812.88 | 2301.73 | baseline |
| 16384 | Real scale16 E8M0 | 1607.44 | 846.63 | 2454.07 | +6.62% |
| 16384 | NVFP4 + float scales | 1640.65 | 854.89 | 2495.54 | +8.42% |

At 16,384 tokens, profiler and graph timing agree closely. Scale16 GEMM1
has about 8% overhead and GEMM2 about 4%; NVFP4 has about 10% and 5%,
respectively. The 512-token profiler alignment does not persist at larger
token counts with these fixed tiles.

All six larger cases pass correctness. Scale32 `logits_diff` is about
3.40e-6, scale16 at most 3.14e-11, and NVFP4 at most 1.78e-5, versus
the existing 0.01 gate. NVFP4 uses the same float scales as the 512-token
comparison: A1=0.7, A2=1.3, W1=0.8, W2=1.1.

Raw samples and launch settings:
[2,048-token results](grouped_gemm_fp4_scaling_tokens2048_results.json) and
[16,384-token results](grouped_gemm_fp4_scaling_tokens16384_results.json).

## Correctness and representation

Scale16 stores **eight independent scale bytes per K128 row** and loads them
as a 64-bit WMMA scale operand. It emits
`v_wmma_scale16_f32_32x16x128_f4`. Adjacent K16 scales can differ; the
implementation does not duplicate each K32 scale.

Scale format 0 selects E8M0, 1 selects unsigned E5M3, and 2 selects E4M3.
The ISA matrix A format uses NEG and matrix B uses NEG_HI. The kernel's
transposed WMMA order maps logical weights to ISA matrix A and logical
activations to ISA matrix B, so it swaps the logical A/B format parameters
when emitting WMMA. Disassembly confirms SCALE16 and E4M3 on both operands
for NVFP4.

Each operand also has a float tensor scale. GEMM1 multiplies its accumulator
by `global_scale_a1 * global_scale_w1` before bias and activation; GEMM2 uses
`global_scale_a2 * global_scale_w2`. NVFP4 quantization normalizes in f32
before FP4 packing so E4M3 mantissas and float scales are honored.

| Mode | logits_diff vs reference |
|---|---:|
| Scale32 E8M0 | 3.40e-6 |
| Scale16 E8M0 | 4.54e-12 |
| NVFP4 with float scales | 1.79e-5 |

All pass the existing `logits_diff < 0.01` gate. All 25 scale tests pass,
covering mixed A/B formats, bias, exact quantized scale/payload bytes, zero inputs, and a
bit-exact scale32/scale16 GEMM comparison with deliberately repeated scales.
Existing FP8 activation and SiLU/Swiglu/SiTUv2 regressions pass. Custom FP4
scales currently apply to the gfx1250 grouped INTERLEAVE path. AOT wiring
covers auxiliary kernels; the existing GEMM AOT wiring remains incomplete.

## Reproduce

Run the original command once per scale block size:

```bash
for block in 32 16; do
  AITER_MOE_EXPERT_BALANCE=true python3 op_tests/flydsl_tests/test_flydsl_grouped_gemm.py \
    --scenario kernel --data-format a4w4 --experts 96 --tokens 512 --topk 6 \
    --model-dim 7168 --inter-dim 3072 --act silu --no-bias --no-check-aot-cache \
    --scale-block-size "$block"
done
```

For NVFP4, use the same command with `--nvfp4` and the tested float scales:
`--global-scale-a1 0.7 --global-scale-a2 1.3 --global-scale-w1 0.8 --global-scale-w2 1.1`.
Independent formats are selectable with `--scale-format-a` and
`--scale-format-b` (`e8m0`, `e5m3`, or `e4m3`). Float scales default to 1.0.

The controlled comparison checks that all modes launch with identical settings:

```bash
python3 op_tests/flydsl_tests/benchmark_grouped_gemm_fp4_scales.py \
  --rounds 5 --output /tmp/fp4_scaling_results.json

for tokens in 2048 16384; do
  python3 op_tests/flydsl_tests/benchmark_grouped_gemm_fp4_scales.py \
    --tokens "$tokens" --rounds 5 --output "/tmp/fp4_scaling_tokens${tokens}.json"
done
```

Unset tile/scheduling environment overrides to reproduce the CSV defaults.
