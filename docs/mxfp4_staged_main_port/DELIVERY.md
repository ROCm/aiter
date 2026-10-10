# MXFP4 staged main-port delivery

Integration branch: `integration/mxfp4-staged-main-port`.
Integration worktree: `/root/workspace/tmp_dev/aiter-mxfp4-integration`.
Fixed main base: `882ea79dd7cc7091f859a5bedac914523ac77bbd`.
Final reviewed implementation: `9ffeb6a747fd6989728cc64c8311bba76e5dab97`; the subsequent signed delivery commit updates only final status/evidence pointers. Run `git rev-parse integration/mxfp4-staged-main-port` for the final branch tip.

All four local tickets are resolved and contain Answers with implementation, merge and acceptance evidence. The canonical tracker is `.scratch/mxfp4-staged-main-port/` in this integration worktree. Original source worktree `/root/workspace/tmp_dev/aiter` remains on `back/worktree-mxmoe-gemm1` at `f68e08528cb85f7f3f7211d3f8a1664660bb7f50`; its pre-existing files and prior experiments are preserved. No push or PR was performed.

| Ticket | Status | Integration merge | Evidence |
| --- | --- | --- | --- |
| 01 lifecycle | resolved | `47cb9fb46d8ea532beea98d6dc7d920475f7ece6` | [lifecycle](01/lifecycle.md), [ordinary tuner](01/ordinary-regression.md) |
| 02 default staged | resolved | `00f9c6073ff3d078ee5ead9688f7dc6804a337e6` | [implementation](02/implementation.md), [small GPU/production/AOT](02/gpu/README.md) |
| 03 configured paths/AOT | resolved | `1e901d768dc246dd13d9238ff711d68e6dfc16bf` | [implementation](03/implementation.md), [specified pairs and clamp](03/gpu/README.md) |
| 04 acceptance/calibration | resolved | `d1ee155922468a02b41e9ea781782dc5d8e9a2bb` | [calibration](04/calibration-summary.md), [deployment](04/deployment-summary.md), [targeted replay](04/targeted-summary.md) |

Final review fixes merged as `ff4e6db25cb98bab1959fffc67d75526fd017e2c`, with annotation-only follow-ups `65a3ff92ff589755c490ec9bcffdbe6daa3a3681` and `9ffeb6a747fd6989728cc64c8311bba76e5dab97`. [Standards clearance](final-review/standards-clearance.md) and [Spec verification](final-review/spec-verification.md) report zero remaining actionable findings. New fix/final commits carry sign-off; original implementation/evidence history is preserved.

## Implementation and checks

The CLI keeps default `prune`, preserves `full`, and adds opt-in `staged` with screening top-K2. Staged projects the shared M_est-pruned layout-v2 universe, screens by compatible contexts, validates the actual shortlisted pipelines, and exhausts admitted group pairs when reference/screening/shortlist failures require fallback. Scalar returned outcomes are reused by pair identity. Fresh private-Pipe workers and operation SIGALRM ownership implement the agreed shape recovery/resource contract. Runtime device preflight, input metadata, valid-winner CSV persistence and stable-key summary repairs apply without replacing prune/full scheduling.

Configured limit/FP8/FP4 settings agree through workers, direct/reference/pair execution, production and AOT. The caller still chooses the matching CSV/environment/limit; no sidecar, key extension or automatic binding was introduced. [Usage](04/usage.md) describes modes/K, file reuse hazards, profile/timing/timeout boundaries and physical device selection.

Final repair suite:294 controlled checks passed in104.08 seconds with48 existing toolchain/Pandas warnings. Earlier focused ordinary-tuner lifecycle, policy, supervisor, parameter and AOT checks are retained in each ticket record. Black, Ruff, Python compilation and full diff checks passed on affected files. Annotation-only changes retain identical executable AST;22 named fixture outputs match prior configurations. Two root-coordinated exact run-only default/configured replay checks passed after shared helper extraction with errors0.0111744/0.0112815. [Repair evidence](final-review/fixes/README.md) retains complete checks and commands.

## Numerical and deployment acceptance

All compared GPU work uses physical GPU7 selected by `ROCR_VISIBLE_DEVICES=GPU-5d8a72a332aa124c` and `HIP_VISIBLE_DEVICES=0`, with verified PCI `0000:f5:00.0`, gfx950/256 CUs. Before each independent task, idle samples and UUID/PCI are checked; the task exits before the next starts. Toolchain versions and exact environments are recorded in [toolchain](implementation/toolchain.json) and per-task records.

The prescribed small staged smoke passed with8 validations and canonical winner15.3478 microseconds, then exact A4W4 production, fresh AOT and run-only replay. Ticket03 retained22 specified pairs with88 deployment/numerical tasks plus2 independent active-clamp controls; pair errors0.00755293–0.01815661 passed the unchanged Torch finite cosine-diff gate<=0.1. FP8 exactly-once nonuniform route weights, FP4 N128/N256 payload/scales/quantized scatter, combined output flags, unchanged atomic handling, custom limit5 and Situv2 are covered.

All27 calibration winners passed81 exact public replay/AOT/run-only tasks;108 AOT jobs compiled. Production error range is0.01116842–0.01814995. The omitted MXFP4 public Situv2 parameters resolve to the established4/25 defaults; corrected production/run-only errors are0.01113366/0.01113105. The initial harness's incorrect1/1 expectation and failure remain preserved as diagnosis evidence.

## Calibration results

All27 initial searches produced valid winners:17 default MiniMax searches and10 GPT-OSS configuration searches.13296 total pair validations; staged fallback0 in this matrix; prune-valid/staged-invalid outcomes0. Every paired outcome row is retained. Each initial search starts with an empty dedicated FlyDSL cache and shared warmed auxiliary infrastructure; data/routes/weights/iterations match across comparisons.

| Matrix | Prune validations | Staged validations | Prune wall seconds | Staged wall seconds | Measured tuning-time ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default MiniMax seven shapes | 4392 | 120 | 2507.705315 | 908.925484 | 2.759x |
| GPT-OSS five configurations | 4560 | 120 | 2464.419399 | 1042.375843 | 2.364x |

The three exhaustive full anchors each validated1368 admitted pairs. Actual process wall times were699.554425 seconds(token64),807.808870(token2048) and896.597213(token8192), compared with their preserved pre-start projections569–11544,842–14280 and1732–27960 seconds. Conservative lower-range misses and compilation/reference extrapolation uncertainty are disclosed. The estimates are not hard deadlines.

[Calibration analysis](04/calibration-analysis.md) reports full→prune, prune→staged and full→staged separately where matching full data exists. Single-search staged/prune kernel-sum differences range from about−0.658% to+8.479%. Ten targeted alternating replay tasks retain60 samples with median/dispersion/rank evidence: some signs reverse or shrink, while measured shortlist losses remain in MiniMax2048 and GPT default/FP8/limit5/combined5. No additional search, adaptiveK or policy change was introduced to improve numbers. Kernel-duration sum is not serving latency; synchronized elapsed and trace-span diagnostics are separately labelled.

## Retained limits

Results describe the recorded physical card, data, toolchain and cache/measurement contracts; there is no universal-optimum, regret or speedup guarantee. Soft SIGALRM cannot bound a permanently live GPU hang. Winner-file/configuration consistency remains the caller's responsibility.

[Ticket01 incident](01/incident.md) records early uncontrolled test-development GPU execution. Those results were excluded from acceptance; complete raw logs were not saved and physical/UTC identity is unknown. Final controlled tests use empty visibility/fail-closed guards, and all formal GPU work follows the coordinator's identity/idle/sequential discipline. Original unsigned implementation commits remain as provenance; final repairs/adoption are signed without rewriting evidence.

Complete source scripts, configuration CSVs, scalar results, runtime/idle/process records and lossless compressed logs/traces are preserved under this delivery directory. Only newly created merged implementer worktrees are removed; integration, external runtime caches and original/prior experiment worktrees remain.
