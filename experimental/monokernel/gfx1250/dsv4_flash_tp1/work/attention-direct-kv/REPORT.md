# Direct current-token KV results

C1 Flash region microbenchmark, not serving throughput. All values are round 2 median GPU-event time across 40 samples; each graph has 128 region invocations. Alternating A/B order. L2 prefetch, TDM, WMMA, and split-KV reduction retained.

## Reused weights

| Tokens | Splits | Region | Cache barrier (us) | Direct KV (us) | Time reduction |
|---:|---:|---|---:|---:|---:|
| 128 | 1 | attention | 6.419 | 7.408 | -15.40% |
| 128 | 1 | two | 18.505 | 19.318 | -4.40% |
| 128 | 1 | fused | 19.982 | 20.025 | -0.22% |
| 128 | 4 | attention | 11.215 | 10.732 | 4.30% |
| 128 | 4 | two | 21.358 | 20.927 | 2.02% |
| 128 | 4 | fused | 21.300 | 20.516 | 3.68% |
| 1024 | 1 | attention | 35.180 | 45.226 | -28.56% |
| 1024 | 1 | two | 46.832 | 57.350 | -22.46% |
| 1024 | 1 | fused | 50.609 | 60.029 | -18.61% |
| 1024 | 4 | attention | 17.596 | 21.741 | -23.56% |
| 1024 | 4 | two | 32.038 | 33.875 | -5.73% |
| 1024 | 4 | fused | 28.657 | 30.986 | -8.13% |

## Sixteen rotating weight copies, T=128

| Splits | Launches | Cache barrier (us) | Direct KV (us) | Time reduction |
|---:|---:|---:|---:|---:|
| 1 | 2 | 20.341 | 21.172 | -4.09% |
| 1 | 1 | 21.529 | 21.971 | -2.06% |
| 4 | 2 | 25.381 | 24.598 | 3.08% |
| 4 | 1 | 24.116 | 22.677 | 5.96% |

## Validation

- Existing FP64 attention and chain validation passed at unchanged 0.001 relative RMS tolerance.
- 61,440 changing-input graph stress checks passed, with cache rows deliberately stale.
- 120 edge cases passed, including slot 0, duplicate/missing indices, all invalid, partial tiles, and one-row fallback. Persistent cache contents verified.
- All measured specializations have zero spills.
- Profiler confirms one launch for fused and two for hybrid.
- Split-KV reduction remains inside each measured attention invocation.

## Exploration

The initial whole-tile LDS overlay and rejected register-substitution experiment are preserved as lds_overlay.py / lds-overlay-timings.json and register_overlay.py / register-overlay-{timings,resources}.json. Register substitution expanded register use and was substantially slower. Current source uses row-only LDS scatter. These are not serving-level speedup claims.

## Next focus

Per user direction, prioritize split 4 and higher for low-concurrency monokernels; split-1 results are historical controls. Split 4 at T=128 improves, including rotating weights; T=1024 split 4 still regresses and needs attention-pipeline optimization. Do not assume the direct path wins across contexts. No split >4 has been measured here.
