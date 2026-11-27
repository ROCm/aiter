# Pull all-reduce + Gemma RMSNorm on MiniMax-M3.1 shapes: measured routing table

Measured on MI355X, TP4 (one node, xGMI), hidden 6144 bf16, Gemma RMSNorm (`gemma_norm=True`, eps 1e-6),
through the real interface (`tensor_model_parallel_fused_allreduce_rmsnorm` inside aiter's `graph_capture()`).
Reproduce with `op_tests/multigpu_tests/bench_fused_ar_rms_gemma.py`.

Protocol: graph of 64 calls on distinct inputs (16 above 1024 rows), 21 ABAB rounds (the candidate order flips
every round), median of the rounds after two warm-up rounds, maximum over the four ranks, microseconds per call.
Numerics of every cell: the last call's `residual_out` is bit-exact against the fp32 two-stage reference
(chunk owner first, one bf16 rounding) and `out` is within 1 bf16 ulp of it (431 measurements, 0 residual
mismatches, max 1 ulp).

Candidates: `iface` = the interface as configured (pull first where it accepts; custom fused AR+RMS above);
`iface_nopull` = the same interface without the pull engine (what aiter did before); `custom_1stage` /
`custom_2stage` = `CustomAllreduce.custom_fused_ar_rms`; `unfused` = `all_reduce` + `rmsnorm2d_fwd_with_add`.
The FlyDSL one-shot and quick-reduce fused ops decline `gemma_norm` (they take a plain weight), so they cannot
serve this call.

## Winner per row count (the shipped table)

| rows | bytes | tier of the shipped ladder | iface (pull) | iface_nopull | custom 1-stage | custom 2-stage | unfused | gain vs best other |
|---|---|---|---|---|---|---|---|---|
| 1 | 12288 | oneshot | 5.16 | 7.27 | 7.23 | 8.04 | 6.42 | +24.4% |
| 2 | 24576 | oneshot | 5.41 | 7.39 | 7.46 | 8.04 | 6.68 | +23.3% |
| 4 | 49152 | oneshot | 5.56 | 7.65 | 7.58 | 8.31 | 7.19 | +29.2% |
| 8 | 98304 | oneshot | 6.49 | 8.45 | 8.29 | 8.53 | 8.09 | +24.6% |
| 16 | 196608 | pull block | 7.35 | 10.30 | 10.26 | 9.43 | 9.02 | +22.7% |
| 32 | 393216 | pull block | 8.79 | 13.93 | 13.84 | 11.44 | 10.96 | +24.6% |
| 64 | 786432 | pull wave | 12.06 | 15.59 | 21.68 | 15.23 | 14.85 | +23.1% |
| 96 | 1179648 | pull wave | 15.64 | 21.47 | 29.08 | 21.45 | 19.44 | +24.2% |
| 128 | 1572864 | pull block | 19.23 | 26.68 | 36.40 | 26.58 | 23.08 | +20.0% |
| 192 | 2359296 | pull block | 26.34 | 34.65 | 51.54 | 34.70 | 30.33 | +15.2% |
| 256 | 3145728 | pull block | 33.82 | 41.81 | 67.03 | 41.77 | 38.18 | +12.9% |
| 257 | 3158016 | stream r2 hoist | 34.67 | 41.85 | 66.91 | 41.88 | 38.55 | +11.2% |
| 288 | 3538944 | stream r2 hoist | 38.25 | 46.75 | 73.52 | 46.68 | 42.68 | +11.6% |
| 384 | 4718592 | stream r2 hoist | 49.22 | 58.66 | 95.49 | 58.53 | 54.24 | +10.2% |
| 512 | 6291456 | stream r2 hoist | 64.09 | 73.89 | 125.46 | 73.91 | 69.06 | +7.8% |
| 513 | 6303744 | stream r4 hoist | 65.07 | 74.03 | 125.70 | 73.95 | 70.35 | +8.1% |
| 640 | 7864320 | stream r4 hoist | 79.71 | 89.74 | 154.17 | 89.60 | 85.28 | +7.0% |
| 768 | 9437184 | stream r4 hoist | 95.62 | 105.85 | 184.09 | 105.52 | 101.74 | +6.4% |
| 896 | 11010048 | stream r4 hoist | 111.29 | 122.74 | 214.45 | 122.61 | 118.32 | +6.3% |
| 1024 | 12582912 | stream r4 hoist | 127.41 | 138.26 | 243.11 | 137.93 | 130.22 | +2.2% |

Beyond the window: 2048 rows iface 263.8 / custom 2-stage 263.7 / unfused 252.9 us, 4096 rows 511.4 / 512.9 / 493.5
(graph, 16 calls); eager prefill calls (no graph): 1024 rows iface 147.6 / unfused 140.3 us, 2048 275.2 / 264.7, 4096
537.2 / 511.2. The unfused pair is 3.6-5% faster than the custom fused kernel there (and 2.5-15% faster at every row count of
the table above); the custom fused kernel is never the fastest candidate for Gemma at hidden 6144 on this platform.

## Fit of the pull ladder (each kernel forced on every row, same protocol)

| rows | pf_oneshot | pf_pull_block | pf_pull_wave | pf_stream_r2 | pf_stream_r2_nohoist | pf_stream_r4 | pf_stream_r4_nohoist | fastest |
|---|---|---|---|---|---|---|---|---|
| 1 | 5.14 | 6.50 | 7.44 | - | - | - | - | pf_oneshot |
| 2 | 5.65 | 6.88 | 7.86 | - | - | - | - | pf_oneshot |
| 4 | 5.59 | 6.69 | 7.85 | - | - | - | - | pf_oneshot |
| 8 | 6.35 | 7.07 | 8.04 | - | - | - | - | pf_oneshot |
| 11 | 7.08 | 7.20 | 8.19 | - | - | - | - | pf_oneshot |
| 12 | 7.31 | 7.17 | 8.18 | - | - | - | - | pf_pull_block |
| 16 | 8.35 | 7.42 | 8.18 | 11.42 | 11.70 | 18.27 | 19.38 | pf_pull_block |
| 32 | - | 9.10 | 9.21 | 12.04 | 12.24 | 19.01 | 19.76 | pf_pull_block |
| 36 | - | 9.34 | 9.20 | 12.02 | - | 18.88 | - | pf_pull_wave |
| 37 | - | 9.72 | 9.59 | 12.28 | - | 19.00 | - | pf_pull_wave |
| 64 | - | 12.43 | 12.09 | 13.85 | 13.86 | 19.77 | 20.86 | pf_pull_wave |
| 96 | - | 15.88 | 15.76 | 17.19 | 17.34 | 20.79 | 21.72 | pf_pull_wave |
| 97 | - | 16.04 | 16.00 | 17.42 | - | 20.81 | - | pf_pull_wave |
| 128 | - | 19.35 | 19.82 | 20.75 | 21.08 | 22.57 | 23.89 | pf_pull_block |
| 192 | - | 26.32 | 27.09 | 27.12 | 27.96 | 28.64 | 29.89 | pf_pull_block |
| 256 | - | 33.96 | 34.90 | 34.29 | 34.91 | 35.40 | 36.71 | pf_pull_block |
| 257 | - | 34.92 | 34.88 | 34.48 | - | 36.37 | - | pf_stream_r2 |
| 287 | - | 39.49 | 39.45 | 37.86 | - | 39.43 | - | pf_stream_r2 |
| 288 | - | 39.42 | 39.69 | 38.04 | - | 39.79 | - | pf_stream_r2 |
| 384 | - | 56.81 | 53.19 | 48.79 | 49.94 | 50.31 | 51.15 | pf_stream_r2 |
| 448 | - | 63.71 | 60.89 | 56.46 | - | 58.50 | - | pf_stream_r2 |
| 480 | - | 67.77 | 65.17 | 60.07 | - | 62.29 | - | pf_stream_r2 |
| 481 | - | 68.33 | 65.24 | 60.48 | - | 61.16 | - | pf_stream_r2 |
| 512 | - | 71.89 | 69.55 | 64.32 | 64.98 | 65.21 | 66.36 | pf_stream_r2 |
| 513 | - | - | - | 65.73 | - | 65.53 | - | pf_stream_r4 |
| 528 | - | - | - | 67.76 | - | 66.84 | - | pf_stream_r4 |
| 544 | - | - | - | 70.07 | - | 68.67 | - | pf_stream_r4 |
| 560 | - | - | - | 72.07 | - | 70.42 | - | pf_stream_r4 |
| 576 | - | 74.70 | 76.11 | 74.09 | - | 72.17 | - | pf_stream_r4 |
| 640 | - | 84.49 | 87.05 | 84.42 | 86.48 | 79.73 | 80.37 | pf_stream_r4 |
| 641 | - | - | - | 86.32 | - | 80.87 | - | pf_stream_r4 |
| 704 | - | - | - | 95.55 | - | 87.17 | - | pf_stream_r4 |
| 768 | - | - | - | 103.61 | - | 95.09 | - | pf_stream_r4 |
| 896 | - | - | - | 121.32 | - | 110.97 | - | pf_stream_r4 |
| 960 | - | - | - | 130.98 | - | 119.23 | - | pf_stream_r4 |
| 1024 | - | - | - | 145.99 | - | 126.28 | - | pf_stream_r4 |

The one-row-per-workgroup kernels (`pf_pull_*`) hang at 768 rows and above (a 300 s timeout on each of 768 and
1024): every workgroup must be resident, so the engine refuses a ladder that reaches them above 640 rows. The
shipped ladder switches where a kernel overtakes its neighbour by more than the 1-2% run-to-run difference: the
streamed kernel at 2 rows per workgroup from 257 rows (1-4% ahead of the one-row kernels from 257 on), at 4 rows per
workgroup from 513 (1.4-2.3% ahead of 2 rows from 528 on, 1.4% behind at 512). Edges inside noise (one-shot vs
pull at 11-16 rows, block vs wave hand-offs at 36-37, 96-97) keep the tiers of the PR as they were.

## Rows 641-1024 (new window; three repeats, microseconds)

| rows | pf_stream_r4 (3 runs) | pf_stream_r2 | unfused (3 runs) | custom 2-stage | gain r4 vs unfused |
|---|---|---|---|---|---|
| 641 | 80.9/81.2/80.8 | 86.3 | 85.7/85.6/85.5 | 89.6 | +5.8% |
| 704 | 87.2/87.1/87.2 | 95.5 | 92.8/93.0/92.8 | 98.3 | +6.5% |
| 768 | 95.1/94.9/95.2 | 103.6 | 101.5/101.4/101.5 | 105.3 | +6.7% |
| 896 | 111.2/111.0/110.9 | 121.3 | 118.3/117.9/118.0 | 122.6 | +6.3% |
| 960 | 119.2/119.5/119.2 | 131.0 | 123.4/123.4/123.4 | 130.1 | +3.5% |
| 1024 | 126.4/126.3/126.2 | 146.0 | 130.3/130.3/130.3 | 137.9 | +3.2% |
