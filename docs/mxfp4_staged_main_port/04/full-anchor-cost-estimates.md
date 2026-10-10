# Full-anchor cost planning before calibration

Current prescribed-pair evidence supplies elapsed overhead, not winner kernel latency. Ten MiniMax pair harnesses ranged 4.484–26.675 s including imports, preparation, compilation/cache, correctness, warmup, profiling and extra synchronized diagnostics. Setup ranged 0.748–0.789 s. The full harness is a conservative comparator; it includes extra work beyond each normal tuner pair attempt.

| Full anchor | Static pairs | Attempt hypothesis s | Projected wall range |
| --- | ---: | --- | --- |
| minimax-64-full | 1368 | [0.35, 8] | 9.5–192.4 min |
| minimax-2048-full | 1368 | [0.55, 10] | 14.0–238.0 min |
| minimax-8192-full | 1368 | [1.2, 20] | 28.9–466.0 min |

The optimistic lower rates assume warmed imports and mostly cached compilation; they are hypotheses rather than directly measured normal-attempt values. The upper rates allow cold compile keys, extra preparation/reference costs and failures. Large token8192 has not been measured by the focused pair fixtures, so the range explicitly scales that uncertainty. The dedicated search cache begins empty and remains shared within each recorded configuration. Main warmup5/iters101, data seed0/routes seed0, toolchain and physical card stay fixed.

Each anchor records this estimate before startup and its actual process elapsed/attempt count after exit. Progress logs let the coordinator refine expectations. Soft SIGALRM is not a hard bound; a permanently live hang remains unbounded. No required search is skipped or stopped on a cost estimate.

Every source record, compilation-cache path and assumption is in full-anchor-cost-estimates.json.
