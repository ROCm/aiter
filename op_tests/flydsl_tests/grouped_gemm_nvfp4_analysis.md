# NVFP4 performance investigation on gfx1250

All production comparisons retain the existing CSV **64 × 512 × 128** tiles,
waves 1 × 4, two buffers, cluster 1, two waves per TDM tensor, and no
next-stage prefetch. The shape is 96 experts, topk 6, model dimension 7168,
intermediate dimension 3072, balanced routing, SiLU, and no bias. Measurements
are from 2026-10-09 on gfx1250 with 256 CUs. Experiments below use 16,384
tokens and five interleaved graph timing rounds unless specified otherwise.

The retained change corrects the FP4 scheduler's weight-read count from two
to four b128 reads per fragment. It improves absolute performance slightly,
but does not establish scale16/scale32 parity. Final production results and
the commands for 512, 2048, and 16384 tokens are in
[the scaling report](grouped_gemm_fp4_scaling.md).

## What the timing includes

GEMM1 includes fused SiLU and intermediate FP4 quantization when bias is absent.
GEMM2 consumes that quantized intermediate. Input quantization, routing, and
gather/reduce are outside the isolated GEMM measurements. Consequently,
changing the activation scale format affects both the main loop and GEMM1's
output quantizer. Changing the weight scale format affects the main loop.

The final graph medians after the scheduler correction are:

| Tokens | Scale32 sum (µs) | Scale16 sum (µs) | NVFP4 + float sum (µs) | Scale16 overhead | NVFP4 overhead |
|---:|---:|---:|---:|---:|---:|
| 512 | 228.51 | 238.69 | 242.47 | 4.45% | 6.11% |
| 2048 | 397.57 | 422.18 | 428.79 | 6.19% | 7.85% |
| 16384 | 2275.32 | 2442.34 | 2478.60 | 7.34% | 8.93% |

At 16k, profiler timing gives 7.28% and 9.06% overhead, respectively. The
small-token profiler results are closer, but do not imply equality at larger
token counts. These sums are the two GEMM medians, not end-to-end MoE latency.

## Scale16 adds movement, without changing matrix arithmetic

One K128 row needs four encoded scale bytes for scale32 and eight for scale16.
For a 64 × 512 × 128 tile, activation and weight payloads total 36,864 bytes.
Scale data grows from 2304 to 4608 bytes: total input bytes grow by 5.88%.
This is a byte accounting result, not a measured bandwidth attribution.

The generated main loop keeps the same b128 payload loads and WMMA count.
Scale32 uses `ds_load_2addr_b32`; scale16 uses `ds_load_2addr_b64`. Each
scale16 WMMA reads two-register scale operands instead of one-register
operands. In the disassembled loop, scale readiness participates in the
`s_wait_dscnt` dependencies before WMMA. Wider transfers, register reads, and
the resulting instruction schedule can therefore cost time even when the
matrix instruction's intrinsic throughput is unchanged.

Resource checks show no spills or register-count occupancy cliff:

| Mode | GEMM1 VGPRs | GEMM2 VGPRs | LDS per workgroup | VGPR/SGPR spills |
|---|---:|---:|---:|---:|
| Scale32 E8M0 | 376 | 374 | 80,896 B | 0 / 0 |
| Scale16 E8M0 | 376 | 374 | 84,992 B | 0 / 0 |
| NVFP4 E4M3 + float | 376 | 374 | 84,992 B | 0 / 0 |

These are the final binaries after the scheduler correction. Before that
correction the counts were 360/358 for every scale mode. The changed schedule
adds sixteen VGPRs uniformly and still gives the small timing improvement
shown below.

Forcing scale32 to reserve 84,992 B leaves it near 1487/814 µs, versus
scale16 at 1610/846 µs in that experiment. Extra LDS allocation alone does
not explain the gap.

A one-wave probe holds payloads and correctly encoded unit scales in
registers, performs 1024 WMMA iterations, and verifies the result. Different
accumulator initial values keep all sixteen independent chains observable.

| Scale mode | One dependent accumulator (µs) | Sixteen independent accumulators (µs) |
|---|---:|---:|
| Scale32 E8M0 | 10.188 | 62.231 |
| Scale16 E8M0 | 10.213 | 61.930 |
| Scale16 unsigned E5M3 | 10.138 | 61.888 |
| Scale16 E4M3 | 10.133 | 62.063 |

Within this synthetic register-resident probe, SCALE16 and the format bits
show essentially equal throughput. It does not measure the GEMM's exact
register-bank arrangement or mixed LDS/WMMA schedule.

## E4M3 quantization costs more than the final float multiply

The following format ablation rebuilds and validates inputs for every mode;
it does not reinterpret E4M3 bytes as E8M0. All entries use real scale16.
Float scales, when enabled, are A1=0.7, A2=1.3, W1=0.8, W2=1.1.

| Format/global-scale configuration | GEMM1 (µs) | GEMM2 (µs) |
|---|---:|---:|
| E8M0, unit float scales | 1607.34 | 847.59 |
| E4M3, unit float scales | 1637.55 | 849.98 |
| E4M3, non-unit float scales | 1642.33 | 855.29 |
| Activation E4M3, weight E8M0, unit floats | 1631.32 | 848.09 |
| Activation E8M0, weight E4M3, unit floats | 1618.67 | 850.69 |

E4M3's main additional cost appears in GEMM1. Its fused output quantizer must
round and encode a mantissa-bearing block scale, decode the rounded value,
normalize values in f32 with the block and tensor scales, and pack FP4. The
E8M0 path uses a simpler power-of-two scale and bf16 pack path. Disassembly
shows f32 FP4 conversion and additional reciprocal/normalization work for
NVFP4. This is consistent with the activation-only ablation, but does not
assign every microsecond to an individual instruction.

Replacing GEMM1's fused epilogue with a raw bf16 store gives:

| Mode | Original fused GEMM1 (µs) | GEMM1 with raw store (µs) |
|---|---:|---:|
| Scale32 E8M0 | 1488.70 | 1475.36 |
| Scale16 E8M0 | 1606.64 | 1598.21 |
| NVFP4 + float | 1643.47 | 1616.70 |

The scale16 gap persists in this probe: about 8.3% in the raw-store kernels.
The NVFP4-versus-scale16 difference falls from 36.83 to 18.50 µs. Raw bf16
output has twice as many columns as activated output and stores more bytes
than FP4, so subtracting the columns does not measure exact quantizer cost.

A separate same-input probe forces only the final accumulator float product
to 1.0. GEMM1 changes from 1643.94 to 1643.34 µs and GEMM2 from 855.81 to
854.09 µs. This deliberately changes numerical output and is used only to
measure that multiplier. It leaves quantization and block bytes unchanged.
The final float multiplication is a small part of the observed slowdown.

## Preload and LDS experiments

| Experiment | Baseline GEMM1/GEMM2 (µs) | Probe GEMM1/GEMM2 (µs) | Result |
|---|---:|---:|---|
| NVFP4 whole next-stage A/B/scale prefetch | 1643.94 / 855.81 | 2231.59 / 1145.76 | Slower; reverted |
| NVFP4 direct global scale loads at tile boundary | 1642.05 / 855.76 | 1994.98 / 1047.92 | Slower; reverted |
| NVFP4 direct global scales preloaded one tile ahead | 1632.48 / 848.43 | 1901.99 / 999.03 | Slower; reverted |
| FP4 scheduler read-count correction, scale32 | 1489.01 / 810.53 | 1474.19 / 803.69 | Retained |
| FP4 scheduler read-count correction, scale16 | 1608.69 / 847.99 | 1603.40 / 839.13 | Retained |
| FP4 scheduler read-count correction, NVFP4 | 1643.04 / 857.42 | 1633.63 / 850.00 | Retained |

Direct global scale experiments remove scale TDM jobs and load the correctly
indexed preshuffled bytes into VGPRs. The preload version carries only scales
across tiles while payloads retain the CSV pipeline. Both pass correctness.
They duplicate some activation-scale reads across waves, add global address
work, and expose VMEM dependencies; preloading reduces the direct-load cost
but does not beat the existing TDM/LDS path. The LDS arena was kept unchanged
to isolate the transfer path. These probes apply to the fixed K128 pipeline;
they are not production alternatives for every tile configuration.

Loading scales first with a scheduler barrier does not improve on the
read-count correction. The retained correction follows the actual FP4
`load_b`: two N16 halves, each with two b128 reads. It changes the scheduling
hint consistently for scale32, scale16, and NVFP4, and leaves CSV selection
and launch settings unchanged.

A one-wave LDS latency probe issues 1024 loads with an immediate
`s_wait_dscnt 0`, consumes the result, and verifies all lanes:

| Read / lane stride | Median (µs) |
|---|---:|
| b32 / 4 B | 33.573 |
| b64 / 8 B, current contiguous scale pattern | 33.385 |
| b64 / 12 B | 33.806 |
| b64 / 16 B | 33.630 |

Padding does not reduce latency in this probe. An attempted TDM scale-padding
layout failed correctness and was discarded without using its timing. The
microbenchmark does not reproduce two-address loads or contention with
payload LDS traffic, so it cannot rule out all bank effects in the full GEMM.
Hardware counter collection failed with `aqlprofile API table load failed`;
no bank-conflict counter values were obtained.

## Evidence and remaining limits

The experiments support scale movement/scheduling as the main scale16 cost,
plus fused E4M3 encoding and normalization for NVFP4. They do not establish a
single measured bottleneck such as HBM saturation or LDS bank conflicts.
Register-resident WMMA throughput, the allocation control, and the float-only
probe narrow the explanation; none closes the fixed-tile performance gap.

The 25 scale tests, eight JIT activation/FP8 regressions, and all nine final
token/scale cases pass. Existing AOT-only checks require caches absent in this
workspace; JIT validation follows the requested `--no-check-aot-cache` mode.
Auxiliary AOT metadata is supported, while TDM GEMM AOT coverage remains
incomplete.

Raw interleaved samples are in
[grouped_gemm_nvfp4_analysis_results.json](grouped_gemm_nvfp4_analysis_results.json).
Each experiment has its own baseline; comparing absolute values across
experiments is less reliable than the interleaved comparison within one.

Reproduce the instruction probes on an otherwise idle gfx1250:

```bash
python3 op_tests/flydsl_tests/benchmark_fp4_scale_instructions.py \
  --rounds 5 --output /tmp/fp4_instructions.json
```
