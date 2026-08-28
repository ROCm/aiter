# MegaMoE Tile Stage2 performance study (2026-08-24)

The finalized 2026-08-26 epilogue-pipeline ablation and small-op comparison is
in [STAGE2_EPILOGUE_OPT_20260826.md](STAGE2_EPILOGUE_OPT_20260826.md).

## Locked benchmark

All headline measurements use EP16 on two MI355 nodes, 128 tokens/rank,
H=7168, I=3072, E=896, TopK=16, SiLU, and the
`paired-rank-half-remote` routing fixture. Reported Stage2 numbers first take
the maximum across all 16 ranks for each iteration, then summarize the tail.
The breakdown driver freezes one physical Stage1 H1/scale/route arena and
replays Stage2 with generations of the same parity.

Driver:

```text
op_tests/multigpu_tests/bench_megamoe_tile_ep16_stage2_breakdown.py
```

## Baseline decomposition

The original direct-FP32 Stage2 measured approximately:

| Path | rank-max mean |
|---|---:|
| common init/clear | 0.311 ms |
| zero-GMM direct FP32 atomic + scoreboard | 2.171 ms |
| real GMM2 + direct FP32 atomic + scoreboard | 2.373 ms |
| return + final combine, including common init | 2.922 ms |
| full Stage2 | 5.564 ms |

The exact selected `fused_moe` Stage2 callable was captured before its first
execution. Its 100-iteration reference measurements were:

| Path | rank-max mean | all-rank mean |
|---|---:|---:|
| fused_moe GMM2 | 0.260 ms | 0.235 ms |
| fused_moe GMM2 + MORI combine | 1.365 ms | 1.257 ms |
| MORI combine component | 1.184 ms | 1.022 ms |

The primary original cost was therefore the scalar FP32 peer-LSA atomic
epilogue, not the matrix contraction.

## Retained optimizations

Production Stage2 now uses:

- MI355 `buffer_atomic_pk_add_bf16`, two output values per atomic;
- a system release (`buffer_wbl2 sc0 sc1` in emitted gfx950 ISA) before the
  existing system-scope done/ready publication chain;
- direct CCO reads from the BF16 node accumulator, eliminating the separate
  FP32-to-BF16 transmit staging pass;
- 14 persistent dynamic final-combine CTAs instead of 7;
- 8 contiguous tokens (112 KiB) per return WQE, four QPs and four batches,
  with one wave serializing the four doorbells in each batch.

Previous tile-ready Stage2, 20 warmup + 100 timed + tail 50:

```text
rank-max mean  2.087 ms
rank-max P50   2.078 ms
rank-max P95   2.183 ms
all-rank mean  1.952 ms
```

The retained diagnostic modes further separate the optimized compute path:

| Path | rank-max mean | all-rank mean |
|---|---:|---:|
| common init/clear | 0.516 ms | 0.421 ms |
| private GMM2 + checksum sink | 0.802 ms | 0.673 ms |
| private GMM2 + packed-BF16 peer atomic | 1.536 ms | 1.248 ms |

Subtracting the common init gives an approximate private GMM2 cost of 0.286
ms, close to the 0.260 ms generic GMM2 kernel. The remaining compute-side gap
is therefore the peer-atomic epilogue rather than MFMA throughput.

Final communication/final-combine-only, same steady-state policy:

```text
rank-max mean  1.638 ms
rank-max P50   1.637 ms
rank-max P95   1.771 ms
all-rank mean  1.504 ms
```

The final full end-to-end comparison (quant+dispatch+GMM1+SiLU+GMM2+combine)
was:

| Metric | MORI + fused_moe | fused Stage1 + Stage2 |
|---|---:|---:|
| rank-max mean | 1.780 ms | 2.885 ms |
| rank-max P50 | 1.772 ms | 2.876 ms |
| rank-max P95 | 1.803 ms | 3.025 ms |
| all-rank mean | 1.703 ms | 2.745 ms |

The optimized candidate is about 48.0% faster than the previous 5.552 ms
candidate, but remains about 62.1% slower than the small-op baseline.

## Accuracy and ordering checks

- Candidate versus MORI rank-max relative L2: `0.0352540` (`< 0.05`).
- Across the prime and the final sample after 120 additional generations:
  - max rank relative L2: `0.0065285`;
  - max per-token relative L2: `0.0070982`;
  - max absolute difference: `12`;
  - all protocol error counters were zero.
- Fused Stage1 GMM1 remains bitwise identical to its standalone replay; this
  Stage2 change does not alter H1.

Packed-BF16 accumulation is order-dependent, unlike the old FP32 accumulator.
The accuracy threshold and repeat-difference metrics must therefore remain
visible in production validation.

## Rejected experiments

The following were measured but are not production defaults:

| Experiment | observed Stage2 | reason rejected |
|---|---:|---|
| 28 fixed N-tile final CTAs | 9.35 ms | head-of-line waits and barriers |
| GMM CTAs rejoin final queue | 8.15 ms | workers claim unready work and spin |
| 56 KiB group WQE, 32 flushes | 3.59 ms | larger WQE without fewer doorbells |
| 2 x 64-token return chunks | 4.61 ms | coarse readiness/large-transfer tail |
| 32 WQE producer CTAs, 192 grid | 4.49 ms | posting/ready contention |
| 32 producers, 8-token batches | 10.01 ms | queue/doorbell contention |
| 256-CTA direct-node grid | 2.86 ms smoke | excess peer-atomic pressure |
| 256-CTA two-phase final fold | 2.99 ms smoke | lost overlap |
| rank-local partial + LSA scan | 14.26 ms | excessive LSA scan/clear overhead |
| BN128 | 2.76 ms | twice as many GMM/scoreboard jobs |
| static-strided GMM queue | 2.61 ms | no full-path gain |
| explicit global system-scope BF16 pair atomic | 2.50 ms | slower than buffer atomic |
| defer four QP request waits | 2.75 ms | larger tail and less stable progress |

These results show that occupying all 256 CUs is not itself beneficial for
the direct-node design. The 160-CTA grid is bandwidth/atomic constrained; more
GMM producers increase contention. A future 256-CTA design must use ready-aware
phase work rather than merely launch more producer CTAs.

### Follow-up RAIL producer sweep (2026-08-25)

A smaller 2/4/8-CTA RAIL producer design was also tested.  Block 0 remained
the only doorbell coordinator, the grid grew as `159 + P`, and the GMM2 pool
therefore remained fixed at 145 CTAs.  Each producer split the existing
8-token/QP batch into a contiguous sub-WQE and release-published completion
before block 0 appended the ready WQE and serially flushed QP0 through QP3.

The 10-warmup/30-iteration screens were:

| RAIL CTAs | grid | full Stage2 rank-max mean | all-rank mean |
|---:|---:|---:|---:|
| 1 | 160 | 2.083 ms | 1.911 ms |
| 2 | 161 | 2.185 ms | 1.911 ms |
| 4 | 163 | 2.160 ms | 1.961 ms |
| 8 | 167 | 2.269 ms | 2.076 ms |

All protocol and scoreboard checks passed.  A reverse-order 20-warmup/100-
iteration comparison measured P4 at 2.165 ms rank-max / 1.987 ms all-rank and
P1 at 2.198 ms rank-max / 1.919 ms all-rank.  The isolated rank-max movement
was not accompanied by an all-rank gain and did not beat the locked 2.087 ms
production baseline, so the multi-CTA path was not retained.

Two single-CTA protocol changes were screened as well:

- flattening the 8-token by 28-tile ready scan across all wave lanes regressed
  return-only rank-max from 1.424 ms to 3.715 ms;
- removing the per-batch reciprocal remote-ready wait measured 2.083 ms
  rank-max / 1.969 ms all-rank over the final 50 samples, effectively flat in
  rank-max and slightly worse in all-rank throughput versus production.

The production RAIL allocation consequently remains one four-wave CTA with
8-token WQEs and reciprocal batch progress.

### Whole-token node readiness (2026-08-25)

The retained follow-up replaces the 28 consumer-visible tile-ready words per
token with a two-level completion tree. Per-tile `node_done` remains unchanged;
the last route for each tile performs one acq-rel increment of a 64-byte-padded
`node_token_done`, and the 28th completed tile publishes one
`node_token_ready`. RAIL now polls one ready word per token. This changes no
payload bytes and preserves the direct node reduction.

The final reverse-order 20-warmup/100-iteration measurements were:

| Path, rank-max mean | 28 tile flags/token | one token flag | change |
|---|---:|---:|---:|
| GMM2 + peer atomic | 1.118 ms | 1.165 ms | +4.2% |
| return + final combine | 1.635 ms | 1.610 ms | -1.5% |
| full Stage2 | 2.156 ms | 2.128 ms | -1.3% |

The full-Stage2 distribution was:

| Readiness | rank-max mean | P50 | P95 | all-rank mean |
|---|---:|---:|---:|---:|
| 28 tile flags/token | 2.156 ms | 2.117 ms | 2.438 ms | 1.956 ms |
| one token flag | **2.128 ms** | 2.129 ms | **2.243 ms** | **1.889 ms** |

The final token-only kernel measured 2.067 ms rank-max mean / 2.067 ms P50 /
2.164 ms P95 / 1.902 ms all-rank mean. The token path passed all route-count
targets from 0 through 16, arbitrary
Top-K duplicate routes, two-parity poison/reuse, and all Stage2 protocol
counters. The final 100-iteration end-to-end run measured 2.888 ms rank-max
mean / 2.741 ms all-rank mean, with candidate-versus-MORI and all-checks
rank-max relative L2 `0.035257` (`< 0.05`). Whole-token readiness is therefore
the production default.

### Device timeline to the first Stage2 RAIL return (2026-08-25)

A compile-time-only timeline build records `s_memrealtime` on each GPU and
converts ticks with the queried 100000-kHz wall-clock rate. Every duration is
formed within one GPU clock domain, then each iteration takes the EP16 rank
maximum. The run used 20 warmups, 100 samples and the final 50 samples. Because
node 46 was occupied at 91% VRAM by an unrelated idle process, the timeline
driver used memory-light prepacked constant weights; shapes, kernel work and
communication volume were unchanged.

The requested interval is the Stage1 ticket-0 device entry through the first
Stage2 return `flush_async`. The actual Ionic doorbell is bracketed by the two
timestamps around that call:

| Interval | rank-max mean | P50 | P95 | all-rank mean |
|---|---:|---:|---:|---:|
| Stage1 entry -> return doorbell, lower bound | 3.194 ms | 3.073 ms | 4.243 ms | 2.645 ms |
| Stage1 entry -> return doorbell, upper bound | 3.204 ms | 3.083 ms | 4.245 ms | 2.652 ms |
| Stage1 entry -> Stage1 dispatch doorbell, lower | 0.157 ms | 0.137 ms | 0.264 ms | 0.075 ms |
| Stage1 entry -> Stage1 dispatch doorbell, upper | 0.181 ms | 0.161 ms | 0.286 ms | 0.089 ms |

The Stage2-return lower-bound decomposition is:

| Segment | rank-max mean | P50 | P95 | all-rank mean |
|---|---:|---:|---:|---:|
| Stage1 entry -> Stage1 done publish | 1.828 ms | 1.741 ms | 2.669 ms | 1.520 ms |
| Stage1 done -> Stage2 entry | 0.297 ms | 0.240 ms | 0.681 ms | 0.093 ms |
| Stage2 entry -> all local Stage1 gates | 0.138 ms | 0.120 ms | 0.311 ms | 0.047 ms |
| local Stage1 gates -> Stage2 init gate | 0.339 ms | 0.282 ms | 0.631 ms | 0.192 ms |
| Stage2 init gate -> first 32-token batch ready | 1.027 ms | 0.952 ms | 1.705 ms | 0.770 ms |
| first batch ready -> first return flush | 0.058 ms | 0.049 ms | 0.087 ms | 0.022 ms |

The four first-batch QP groups become ready after the Stage2 init gate at:

| Group | rank-max mean | P50 | P95 | all-rank mean |
|---|---:|---:|---:|---:|
| QP0, tokens 0..7 | 0.835 ms | 0.729 ms | 1.474 ms | 0.641 ms |
| QP1, tokens 8..15 | 0.849 ms | 0.719 ms | 1.580 ms | 0.641 ms |
| QP2, tokens 16..23 | 0.972 ms | 0.868 ms | 1.668 ms | 0.760 ms |
| QP3, tokens 24..31 | 0.998 ms | 0.926 ms | 1.668 ms | 0.752 ms |

Consequently QP0 waits another 0.222 ms rank-max mean / 0.129 ms all-rank
mean at the CTA-wide 32-token barrier before it may post and flush. Once all
32 tokens are ready, payload-WQE posting plus the remaining barrier/control
takes only 0.058 ms rank-max mean. The first `flush_async` itself costs 0.025
ms rank-max mean, and its returned request completes another 0.063 ms later.

The timeline instrumentation is compiled out of production kernels. ROCprof
cannot expose this boundary because CCO PUT/flush is inlined device code rather
than a separate HIP API or GPU kernel.

#### Same-GPU first-doorbell versus all-GMM2 completion

Each of the 145 local GMM2 worker CTAs records one timestamp after leaving its
persistent work queue. The host takes the maximum on that same GPU, then
subtracts that GPU's first-return-doorbell upper-bound timestamp. Positive
values mean RAIL starts while local GMM2 is still executing; negative values
mean local GMM2 completed before the first doorbell. No absolute timestamps
are compared across GPUs.

Two completed 100-iteration runs gave nearly identical critical overlap:

```text
run 1: rank-max mean 263.53 us, P50 268.49 us, P95 301.31 us,
       all-rank mean 151.47 us
run 2: rank-max mean 262.22 us, P50 268.19 us, P95 310.76 us,
       all-rank mean 136.61 us
```

The absolute Stage1-entry-to-doorbell interval changed as the unrelated node46
load eased (the later run was 1.778--1.784 ms rank-max mean), while the signed
same-GPU GMM2 overlap remained stable across both runs. The overlap delta is
therefore the more reliable answer to the GMM-versus-doorbell question.

Per-rank tail-50 means from run 2 are:

| rank | first doorbell lower/upper from S1 entry | all GMM2 done from S1 entry | GMM2 remaining after doorbell | overlap samples |
|---:|---:|---:|---:|---:|
| 0 | 1624.76 / 1630.00 us | 1798.82 us | 168.82 us | 98% |
| 1 | 1608.08 / 1613.42 us | 1722.07 us | 108.65 us | 96% |
| 2 | 1633.17 / 1639.41 us | 1752.19 us | 112.78 us | 96% |
| 3 | 1655.79 / 1662.08 us | 1794.21 us | 132.13 us | 98% |
| 4 | 1658.34 / 1664.93 us | 1843.20 us | 178.27 us | 98% |
| 5 | 1656.39 / 1661.47 us | 1777.01 us | 115.54 us | 98% |
| 6 | 1642.98 / 1649.70 us | 1782.86 us | 133.16 us | 98% |
| 7 | 1669.94 / 1676.63 us | 1799.79 us | 123.16 us | 98% |
| 8 | 1621.10 / 1625.71 us | 1776.93 us | 151.22 us | 98% |
| 9 | 1555.43 / 1559.38 us | 1694.53 us | 135.15 us | 96% |
| 10 | 1570.57 / 1575.55 us | 1702.12 us | 126.57 us | 94% |
| 11 | 1631.53 / 1636.64 us | 1798.36 us | 161.73 us | 96% |
| 12 | 1603.70 / 1608.07 us | 1760.56 us | 152.49 us | 98% |
| 13 | 1593.50 / 1597.76 us | 1720.48 us | 122.73 us | 98% |
| 14 | 1583.53 / 1588.19 us | 1703.36 us | 115.17 us | 92% |
| 15 | 1702.42 / 1706.63 us | 1854.78 us | 148.15 us | 98% |

Thus the current kernel does overlap its first return with the final local
GMM2 tail, but only by about 0.14 ms on an average rank. Across ranks, 92--98%
of tail samples started RAIL before local GMM2 was fully quiescent. Occasional
negative samples show that a rank can finish all GMM2 work before its first
doorbell when another QP group controls the first-batch barrier.

The emitted gfx950 ISA for the retained buffer-atomic path contains
`buffer_atomic_pk_add_bf16`, followed before done/ready publication by
`buffer_wbl2 sc0 sc1`. The 120-generation end-to-end run completed with zero
protocol errors; prime-to-final rank-max relative L2 was 0.00650.

## Logs

All logs are under:

```text
/home/hzm/logs/megamoe_stage2_breakdown_20260824/
/home/hzm/logs/megamoe_ep16_direct_20260821/
```

Key files:

```text
stage2_candidate_init_bf16_steady_v1_node{0,1}.log
stage2_candidate_gmm2pure_steady_v1_node{0,1}.log
stage2_candidate_bf16_gmm2atomic_steady_v1_node{0,1}.log
stage2_final_returnonly_ret8_steady_v1_node{0,1}.log
stage2_final_bf16_fc14_ret8_steady_v1_node{0,1}.log
stage2_mori_gmm2_combine_steady_v1_node{0,1}.log
full_e2e_fusedmoe_ab_stage2_final_bf16_fc14_ret8_e2e_steady_v1_node{0,1}.log
tileready_full_steady_reverse_20260825_v2_node{0,1}.log
tokenready_full_steady_reverse_20260825_v2_node{0,1}.log
full_e2e_fusedmoe_ab_tokenready_e2e_steady_20260825_v1_node{0,1}.log
tokenready_final_stage2_steady_20260825_v3_node{0,1}.log
full_e2e_fusedmoe_ab_tokenready_final_e2e_steady_20260825_v2_node{0,1}.log
stage2_expected_0to16_stress_node{0,1}.log
device_timeline_steady_20260825_v3_node{0,1}.log
device_timeline_gmmdone_perrank_20260825_v3_node{0,1}.log
```
