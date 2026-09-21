# FP4 Decode Pipeline Experiment

Base: ROCm/aiter PR 5707, head `3ab6dfc2a82aa8d200f0b5029b0b584c62f7e0a4`.

Worktree: `/home/jiacao/aiter-fp4-decode-lds`.
Remote: `smci355-ccs-aus-m15-17`, MI355X GPU 0.
Container: `fp4-lds-prefill`, with per-command `HIP_VISIBLE_DEVICES=0`.
Remote source mirror: `/tmp/aiter-decode-pipeline-20260921`.
Container source mirror: `/host-tmp/aiter-decode-pipeline-20260921`.

The requested 2x latency improvement across ordinary decode is NOT achieved.
Do not interpret internal compute-only speedups as complete kernel speedups.

## Measurement

- Identical quantized inputs, page layouts, and weights for every variant.
- Original 256-token/4-wave and 64-token/1-wave kernels are explicit baselines.
- Independent CUDA Graphs contain 64 scorer launches.
- Each sample replays a graph 10 times; variant order is randomized per round.
- The final v11 run reports the median of nine rounds and retains raw samples.
- No graph construction, allocations, schedule building, or reference computation
  is included in scorer timing.
- Empty kernel dispatches use the same graph/event timing method.

## Progress

- Independent decode supports cooperative head waves, LDS KV/scales stages,
  in-CTA head reduction, external schedules, and direct device-side intervals.
- No decode-to-prefill routing or next_n divisibility specialization remains
  in the public API.
- 59 correctness tests pass: next_n 1/2/3/4/5/8; H16..128; D128/256;
  page64/128; shuffled and strided pages; strided outputs; empty contexts;
  single-/multi-chunk schedules; graph replay with changing context lengths.
- The selected implementation uses two cooperative head waves for H64, four
  for H128, and scalar weight loads once the direct grid reaches 768 CTAs.
- Final same-run speedups against the original PR-default decode path are
  1.046x to 1.237x. Comparing against the older timings quoted in PR 5707
  gives 1.18x to 1.63x, but that is not an identical-run comparison.
- Empty dispatch timing is 1.585-1.610 us. Seven of the ten quoted original
  scorer medians cannot reach 2x even with a zero-work kernel under this
  measurement method.

## Final Same-Run Results

| Shape | Original default (us) | Selected decode (us) | Speedup |
|---|---:|---:|---:|
| 2 x 1 x 512, H64 | 2.532 | 2.088 | 1.213x |
| 3 x 1 x 1K, H64 | 2.311 | 2.161 | 1.069x |
| 4 x 1 x 2K, H64 | 2.338 | 2.173 | 1.076x |
| 16 x 1 x 4352, H64 | 3.017 | 2.681 | 1.125x |
| 8 x 1 x 8K, H64 | 2.672 | 2.514 | 1.063x |
| 1 x 1 x 32K, H64 | 2.482 | 2.374 | 1.046x |
| 1 x 1 x 64K, H64 | 2.710 | 2.471 | 1.097x |
| 1 x 1 x 128K, H64 | 3.334 | 2.872 | 1.161x |
| 2 x 2 x 512, H64 | 2.312 | 2.120 | 1.091x |
| 2 x 1 x 768, H128 | 2.734 | 2.211 | 1.237x |

## Pipeline Findings

- Register staging now lets next-chunk buffer-to-LDS DMA overlap current
  MFMA/reduction; ISA inspection confirms the overlap.
- The default direct grid assigns one 64-token chunk per CTA, so ordinary
  shapes do not execute the cross-chunk pipeline loop.
- Reducing the 128K grid from 2048 CTAs to 1536/1024/512 CTAs regresses the
  selected kernel from 2.872 us to approximately 3.123/3.131/4.090 us.
- Grouping both rows of `next_n=2` into one CTA reduces duplicate KV reads but
  regresses 2 x 2 x 512 from 2.120 us to 2.22-2.29 us because parallelism is
  more important than KV bandwidth at that size. The grouped path is not used.
