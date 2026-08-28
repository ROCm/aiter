# TPR128_TopK16_E896_H7168_I3072_EP16_A4W4

## Measurement contract

- Hardware: 2 x 8-GPU MI355 nodes, EP16.
- Routing: `paired-rank-half-remote`.
- Data path: A4W4, SiLU, BF16 node accumulator, packed-BF16 peer atomic.
- Long runs: 20 warmups, 100 timed generations, final 50 generations used
  for each mean.
- Every sample first takes the maximum over all 16 ranks.  `all-rank mean` is
  the mean over every rank and retained generation.
- Forward/reverse entries below are averaged so process order is not confused
  with an optimization benefit.
- No percentile is used as a headline or case identifier.

## Root cause in the old grouped epilogue

For each rank the direct epilogue processes about 2048 routes.  Every route
emits `7168 / 2 = 3584` packed-BF16 peer atomics, or about 7.34 million packed
atomic lane operations/rank.  Each output tile also stages 32 KiB of FP32
accumulators through LDS, applies the route weight, converts to BF16, drains
the VM pipeline, performs one system release, and publishes the scoreboard.

The generated gfx950 ISA exposed the specific lost-overlap point in the old
two-N-tile grouping:

```text
atomic0 final buffer_atomic_pk_add_bf16
  -> A1 direct-to-LDS loads
  -> expert-id global load
  -> s_waitcnt vmcnt(0)
  -> first GEMM1 MFMA
```

Thus atomic0 was completely drained before GEMM1 contraction began.  The
selected implementation:

1. caches route metadata/weights once per M group;
2. loads the shared expert ID in the group prologue;
3. groups two adjacent N tiles under one final drain/release;
4. allocates a disjoint A1 LDS slab and completes its initial DMA before
   issuing atomic0.

The selected ISA reaches GEMM1 MFMA with `vmcnt(17)`, rather than
`vmcnt(0)`, so outstanding atomic work really overlaps GEMM1.  Resource use
changes from 181 VGPR / 104 SGPR / 33,088 B LDS to
183 VGPR / 104 SGPR / 45,376 B LDS.

## Stage2 optimization ablation

The following values average a forward-order and reverse-order long run.
Changes are relative to the old production path.

| Stage2 implementation | 16-rank max mean | all-rank mean | rank-max change | all-rank change |
|---|---:|---:|---:|---:|
| old production: lane32, one N tile | 2226.19 us | 2008.39 us | +0.00% | +0.00% |
| metadata cache, one N tile | 2219.54 us | 1996.96 us | -0.30% | -0.57% |
| metadata cache, two-N-tile drain | 2153.33 us | 1966.55 us | -3.27% | -2.08% |
| two N tiles + expert/metadata hoist | 2128.05 us | 1959.82 us | -4.41% | -2.42% |
| two N tiles + A-LDS double buffer (selected) | **2084.82 us** | 1969.76 us | **-6.35%** | -1.92% |
| two N tiles + QP pre-post | 2141.61 us | 1976.50 us | -3.80% | -1.59% |
| hoist + QP pre-post | 2106.16 us | 1973.19 us | -5.39% | -1.75% |
| A-LDS double buffer + QP pre-post | 2187.43 us | 1972.31 us | -1.74% | -1.80% |

The final default-only rerun, after synchronizing compiler/script defaults,
measured **2105.03 us / 1960.01 us**.  This is consistent with the paired A/B
average above.

The experiment that delayed the post-atomic CTA barrier was not retained:

| A-LDS schedule | 16-rank max mean | all-rank mean |
|---|---:|---:|
| conservative barrier | 2136.08 us | 1994.15 us |
| barrier delayed to GEMM1 | 2152.74 us | 1967.92 us |

It improved all-rank throughput by 1.31%, but regressed the primary rank-max
mean by 0.78%.

## GMM2 and epilogue decomposition

Candidate diagnostic kernels include the same Stage2 initialization.  The
`epilogue increment` rows are therefore the paired difference between
`GMM2 + atomic` and `GMM2 + checksum sink`; they are a useful delta, not a
standalone kernel duration.

| Path | 16-rank max mean | all-rank mean |
|---|---:|---:|
| standalone fused_moe GMM2 | 284.78 us | 241.42 us |
| selected Stage2 init only | 583.67 us | 477.39 us |
| old schedule: GMM2 + checksum sink | 826.07 us | 714.33 us |
| old schedule: GMM2 + peer atomic | 1164.26 us | 1033.14 us |
| old schedule: epilogue increment | 338.19 us | 318.81 us |
| selected ADB: GMM2 + checksum sink | 820.21 us | 705.48 us |
| selected ADB: GMM2 + peer atomic | 1095.78 us | 990.49 us |
| selected ADB: epilogue increment | **275.57 us** | **285.01 us** |

The selected pipeline reduces the epilogue increment by approximately 18.52%
on the 16-rank critical path and 10.60% over all ranks.  Pure GMM2 work is
effectively unchanged; the gain comes from scheduling and overlap.

## Equivalent Stage2 comparison

| Path | 16-rank max mean | all-rank mean | rank-max change vs MORI path |
|---|---:|---:|---:|
| standalone fused_moe GMM2 | 284.78 us | 241.42 us | n/a |
| MORI combine | 1174.78 us | 1030.60 us | n/a |
| fused_moe GMM2 + MORI combine | **1382.87 us** | **1272.02 us** | +0.00% |
| old fused Stage2 | 2226.19 us | 2008.39 us | +60.98% |
| selected fused Stage2 | 2105.03 us | 1960.01 us | +52.22% |

The optimization closes part of the gap, but direct per-route peer atomics,
common initialization/clear, and final communication remain substantially
more expensive than the existing GMM2 + MORI combine path.

## Complete end-to-end comparison

The two execution orders agree within 0.7%.  Values below average both orders.

| End-to-end path | 16-rank max mean | all-rank mean | rank-max change vs small-op |
|---|---:|---:|---:|
| BF16->A4 quant + MORI A4 dispatch + fused_moe GMM1/SiLU/GMM2 + MORI combine | **1803.90 us** | **1715.63 us** | +0.00% |
| fused Stage1 + selected fused Stage2 | 2911.20 us | 2766.83 us | **+61.38%** |

This is the honest current result: the selected epilogue pipeline improves
isolated Stage2, but the two fused stages are still about 61% slower than the
existing small-op chain for this shape.

## Experiments not selected

- `wave64_meta`: removes the two-destination waterfall, but its row loop and
  scheduling cost increased full Stage2 time.
- `preload_pairs`: more live values and register pressure; no atomic issue-rate
  gain.
- four-wave scoreboard publication: substantially slower.
- runtime two-tile loop: halves static MFMA/atomic instances in the ISA but
  does not reduce VGPR count and is unstable in rank-max time.
- QP-independent and QP pre-post return: first doorbell can move earlier, but
  the improvement does not reliably reach the Stage2 critical tail.
- delayed post-atomic barrier: better all-rank throughput, worse rank-max mean.
- non-default waves-per-EU hints: either regress isolated atomic work or the
  complete Stage2 path.

## Correctness and stability

- Full local contract suite: `144 passed`.
- Actual production GMM2/epilogue E2E: prime plus 20 warmups plus 100 timed
  generations, ending at generation 121 in both execution orders.
- Maximum candidate-versus-MORI relative L2: `0.03525781`, below `0.05`.
- No Stage1, Stage2, protocol, expected/done, or token-ready mismatch.
- 16-rank protocol stress passed 32 consecutive generations/rank, including:
  arbitrary Top-K with repeated destination ranks and repeated experts;
  local expected counts 0 through 16; both parity buffers repeatedly poisoned
  and reused; zero-payload clearing; and two single-hot-rank cases.

## Production defaults

The canonical operator and the Stage2 compiler/compile scripts now agree on:

```text
accumulator_dtype      = bf16
bf16_atomic_kind       = buffer
rail_return_schedule   = lockstep
return_chunk_tokens    = 8
epilogue_schedule      = lane32_meta
n_tile_group           = 2
group_pipeline_schedule= a_double_buffer
scoreboard_schedule    = wave0
atomic_issue_schedule  = interleaved
waves_per_eu           = 2
```

## Raw artifacts

```text
/home/hzm/logs/megamoe_stage2_breakdown_20260824/
  epilogue_final_20260825_v1_forward_*_node{0,1}.log
  epilogue_final_20260825_v1_reverse_*_node{0,1}.log
  epilogue_diag_20260825_v1_forward_*_node{0,1}.log
  epilogue_diag_20260825_v1_reverse_*_node{0,1}.log
  production_default_final_20260826_v1_node{0,1}.log
  production_init_only_final_20260826_v1_node{0,1}.log
  mori_stage2_final_20260825_v1_node{0,1}.log

/home/hzm/logs/megamoe_final_comparison/
  TPR128_TopK16_E896_H7168_I3072_EP16_A4W4_
    production_adb_baseline_first_20260825_v1_e2e_node{0,1}.log
  TPR128_TopK16_E896_H7168_I3072_EP16_A4W4_
    production_adb_candidate_first_20260825_v1_e2e_node{0,1}.log

/home/hzm/logs/megamoe_validation_20260826/
  expected_sweep_poison_v2_node{0,1}.log
```
