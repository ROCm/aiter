# Current-port MXFP4 kernel-duration-sum calibration

All27 prescribed initial searches completed with valid winners on physicalGPU7 (`GPU-5d8a72a332aa124c`, PCI`0000:f5:00.0`, gfx950/256CUs). All12 prune/staged paired points are retained; prune-valid/staged-invalid count is0. This is the current port's acceptance evidence, independent of historical experiments. Every search used a separate output and an initially empty FlyDSL cache, with common warmed AITER auxiliary builds, generatedseed0 data/routes, stable invocation flags/limit, warmup5 and iters101.

The saved winner metric is main-compatible device kernel-duration sum for complete end-to-end execution. It is not serving latency or synchronized invocation elapsed time. Search process wall includes import, preparation, compilation/cache, correctness, warmup, profiling, screening and failure costs; controller idle waits remain outside that process sum. Winner deployment and matched elapsed/span diagnostics are separate pending records.

| Matrix | Prune validations | Staged validations | Prune search wall | Staged search wall | Prune/staged wall ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| MiniMax seven default rows | 4392 | 120 | 2507.705s | 908.925s | 2.759x |
| GPT-OSS five environments | 4560 | 120 | 2464.419s | 1042.376s | 2.364x |

The27 independent search tasks span15:52:59–18:35:50UTC, with0 overlaps verified from recorded process start/exit times. All27 process wall sums total9327.387s; all searches returned13296 validated pipeline pairs. Staged recorded405G1/772G2 screens and0 inferred/explicit fallback work groups. Static counts remain separate from actual compilation/numerical evidence.

| MiniMax tokens | Full winner sum μs | Prune winner sum μs | Staged winner sum μs | Full→prune | Prune→staged | Full→staged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 76.9861 | 75.8671 | 76.0348 | -1.454% | +0.221% | -1.236% |
| 128 | unavailable | 84.5815 | 84.2686 | unavailable | -0.370% | unavailable |
| 512 | unavailable | 97.7542 | 101.2191 | unavailable | +3.545% | unavailable |
| 1024 | unavailable | 117.0330 | 117.6557 | unavailable | +0.532% | unavailable |
| 2048 | 173.7718 | 174.7378 | 179.6055 | +0.556% | +2.786% | +3.357% |
| 4096 | unavailable | 248.1153 | 248.8316 | unavailable | +0.289% | unavailable |
| 8192 | 404.8570 | 406.7968 | 406.5586 | +0.479% | -0.059% | +0.420% |

Full→prune isolates the existingM_est universe restriction. Prune→staged isolates screening within that pruned universe. Full→staged combines both effects. Independent measured minima can vary with noise; full64's negative subset comparison motivates alternating selected-pair replay, not an assertion that pruning creates a faster mathematical optimum.

| GPT-OSS environment | Prune winner sum μs | Staged winner sum μs | Prune→staged | Prune winner staged ranks G1/G2 | Retained K2 |
| --- | ---: | ---: | ---: | --- | --- |
| Default | 115.6761 | 121.5918 | +5.114% | 2/3 atomic | no |
| FP8 | 101.8640 | 110.5011 | +8.479% | 2/7 non-atomic | no |
| FP4 | 111.0863 | 110.3552 | -0.658% | 2/2 non-atomic | yes |
| Limit5 | 117.0535 | 122.0186 | +4.242% | 2/3 atomic | no |
| Combined5 | 101.9686 | 110.4887 | +8.356% | 1/7 non-atomic | no |

No GPT-OSS full references were prescribed or run, so full-relative regret is unavailable. Default seven-row wall totals remain separate from these environments. FP4 selected exactly the same pair in prune and staged; its negative difference is a repeated measurement of that identity, not stage screening improvement.

MiniMax512/2048 prune winners were excluded at G1 ranks4/5. GPTdefault/limit5 prune winners lost atomicG2 rank3, while FP8/combined5 lost non-atomicG2 rank7. MiniMax128's small negative difference accompanies G2 rank3 exclusion; alternating replay checks that noisy ordering. MiniMax1024 and8192 prune winners were retained and validated. These rank records explain shortlist loss without treating isolated stage sums as end-to-end performance or correctness.

Ten concrete two-winner checks are retained in `targeted-replay-plan.json`: full/prune64, prune/staged128/512/2048/8192 and GPTdefault/FP8/FP4/limit5/combined5. Each proposes three balanced alternating cycles on one fixed generated fixture, with normal main kernel-sum timing, finite Torch gate0.1 and median/dispersion. They add no tuning invocation. Their measured outcomes and the81 exact winner production/AOT tasks are pending root execution; this report does not mark them passed. The retained8192 and same-pairFP4 negative differences also receive targeted alternating checks. Smaller positive differences remain independently measured observations unless later evidence requires another focused check. There is no hard regret/speedup threshold or automaticK/group/policy change.

Full anchors returned1368 validations each. Their process wall totals were699.554s (token64),807.809s (2048),896.597s (8192). Each retained its pre-start compilation/preparation/warmup/profiling/failure estimate and uncertainty, progress records and actual end record. Token2048 actual807.809s fell slightly below its pre-start842–14280s range; token8192 actual896.597s fell below its conservative pre-start1732–27960s estimate; the extrapolated lower planning hypothesis was not a guaranteed bound. Cost estimates never stopped or reduced required search work.

The complete names, errors, settings, counts, times and screening ranks are in `calibration-summary.json` and `calibration-summary.md`. Original task/runtime identity, idle samples, full compressed logs, outputs and cost/progress evidence remain under `gpu/<search-id>/`. Final production/AOT, targeted repeat results and Standards/Spec review remain required before closing ticket04.
