# MXFP4 integrated calibration and acceptance

Current-port evidence follows the approved01→02→03→04 graph and fixed882ea79dd7 base. The root coordinator owns tracker status and all realGPU tasks.

- [Usage](usage.md): modes/K, device mapping, stable settings, CSV collision and matching AOT/production examples.
- [Offline enumeration](candidate-counts.md) and [full-anchor cost planning](full-anchor-cost-estimates.md): all27 initial searches, grouped counts and before/after cost assumptions.
- [Calibration analysis](calibration-analysis.md), [complete summary](calibration-summary.md) and [JSON](calibration-summary.json):27 valid winners,12 complete paired rows, separate universe/screening regret and seven-row/extended wall sums.
- [Winner deployment](deployment-summary.md) and [JSON](deployment-summary.json):81 successful exact public/AOT/run-only tasks, matching settings/route hashes and complete job keys.
- [Task sequence](search-and-deployment-task-sequence.json):108 search/winner-deployment processes, prescribed UUID/PCI,0 overlaps.
- [Omitted Situv2 defaults](situv2-public-defaults.md): corrected4/25 production and matching run-only retries, with the original failed harness assertion retained.
- [Targeted plan](targeted-replay-plan.json): ten concrete alternating winner comparisons, no additional tuning searches.
- [Combined checks](combined-checks.log.txt), [Situv2 boundary checks](situv2-boundary-checks.log.txt) and [artifact checks](offline-calibration-checks.log.txt): controlled offline behavior, including actual production parameter forwarding.

Original records live in `gpu/<search-id>/`: per-process idle/runtime identity/environment/commit/times, complete compressed tuning logs, winner CSVs, production traces, AOT job keys and run-only outcomes. Targeted repeat results live under `gpu/targeted/`. Source scripts remain adjacent for reproduction; runtime compiler caches are retained externally under the recorded paths rather than committed.

The finalStandards/Spec review and any resulting fixes remain required before root resolves ticket04 and the spec. Kernel-duration sum, synchronized companion span/elapsed and tuning process wall are separately labelled throughout.
