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
- The final v13 run reports the median of nine rounds and retains raw samples.
- Raw v13 data is in `decode-pipeline-v13-final.json`.
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
- The selected D128 direct path bypasses KV/scales LDS staging. It uses a
  32-token, two-wave CTA while the full grid is at most 1024 CTAs, then a
  64-token CTA for larger grids. Static page addressing and a 2D launch remove
  integer address work from the hot path; large H64 grids also retain wave 0's
  partial sum in registers and put only the other wave's partial in LDS.
- Final same-run speedups against pushed commit `6da2fbd27` are 1.013x to
  1.102x (1.064x geometric mean). Against the original 256-token kernel in the
  same run they are 1.128x to 1.635x (1.315x geometric mean).
- Comparing against the initial MXFP4 timings quoted in PR 5707 gives 1.225x
  to 1.776x, with a 1.440x geometric mean. This last comparison is across runs.
- Empty dispatch timing is 1.585-1.610 us. Seven of the ten quoted original
  scorer medians cannot reach 2x even with a zero-work kernel under this
  measurement method.

## Final Results

`Pushed` is the exact configuration selected by commit `6da2fbd27`. `Initial`
is the older MXFP4 timing quoted in PR 5707. The optimized and pushed columns
are paired in the same v13 run; the initial column is included for historical
comparison.

| Shape | Pushed (us) | Optimized (us) | vs pushed | Initial (us) | vs initial |
|---|---:|---:|---:|---:|---:|
| 2 x 1 x 512, H64 | 2.064 | 1.926 | 1.071x | 2.766 | 1.436x |
| 3 x 1 x 1K, H64 | 2.102 | 1.992 | 1.055x | 2.825 | 1.418x |
| 4 x 1 x 2K, H64 | 2.152 | 2.017 | 1.067x | 2.832 | 1.404x |
| 16 x 1 x 4352, H64 | 2.554 | 2.366 | 1.079x | 3.512 | 1.484x |
| 8 x 1 x 8K, H64 | 2.455 | 2.424 | 1.013x | 2.970 | 1.225x |
| 1 x 1 x 32K, H64 | 2.193 | 2.052 | 1.069x | 2.918 | 1.422x |
| 1 x 1 x 64K, H64 | 2.403 | 2.267 | 1.060x | 3.016 | 1.330x |
| 1 x 1 x 128K, H64 | 2.462 | 2.379 | 1.035x | 3.590 | 1.509x |
| 2 x 2 x 512, H64 | 2.146 | 1.960 | 1.095x | 2.840 | 1.449x |
| 2 x 1 x 768, H128 | 2.231 | 2.023 | 1.102x | 3.594 | 1.776x |

## Pipeline Findings

- Register staging lets next-chunk buffer-to-LDS DMA overlap current
  MFMA/reduction, but the ordinary direct grid assigns one chunk per CTA.
  Those shapes never execute the cross-chunk loop, so adding LDS pipeline
  stages does not create overlap or additional waves for them.
- Splitting a 64-token CTA into two 32-token CTAs creates the missing grid-level
  parallelism and shortens each CTA's critical path. It wins through 1024 CTAs;
  beyond that point dispatch and inactive-CTA cost outweigh the shorter tile.
- Direct global KV/scales reads avoid a DMA-to-LDS round trip. The remaining
  LDS use is only the cross-wave partial reduction, not a KV pipeline.
- In the earlier v11 shared-LDS experiment, reducing the 128K grid from 2048
  CTAs to 1536/1024/512 CTAs regressed latency from 2.872 us to approximately
  3.123/3.131/4.090 us.
- Grouping both rows of `next_n=2` into one CTA reduces duplicate KV reads but
  regresses 2 x 2 x 512 from 2.120 us to 2.22-2.29 us because parallelism is
  more important than KV bandwidth at that size. The grouped path is not used.
