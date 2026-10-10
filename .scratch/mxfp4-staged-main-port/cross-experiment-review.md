# Crossed-pair experiment outcome and plan update

Status: resolved

Completed experiment branch: `experiments/mxfp4-cross-measurement`, data commit `96e2b5545` and deterministic-analysis tip `4977e927d`, based on main `882ea79dd7cc7091f859a5bedac914523ac77bbd`. Artifacts are retained in the branch and the current workspace under [docs/mxfp4_cross_measurement/20261010](../../docs/mxfp4_cross_measurement/20261010/README.md).

The user requested single-card measurement, pre-start idle checks, one independent process per measurement task, sequential GPU ownership, comparisons on the same physical card, and subsequently selected GPU7. The formal experiment fulfilled those constraints with PCI/UUID checks, three-round task logs and 81 consistent results. On this host, monitoring GPU7 required UUID-based ROCR visibility plus logical HIP0. The first ordinal7 import-only attempt was rejected on PCI mismatch before timing; it is not part of the formal data.

The formal BM128 sample contained 3 G1 configurations and 6 G2 configurations (18 pipeline pairs) at M_est64, default Silu A4W4/BF16, balanced fixed routes, stage2 FP4/FP8 switches off. All measured configurations were numerically valid, with maximum error 0.011222. The kernel-sum winner was G1[2]/G2[1], 121.466 µs median of three task means. Separate mode K2 (8 pairs) and merged G2 K2 (4 pairs) both retained it in three of three rounds. Independent G1 rank was stable in the formal matrix, but G2 independent/pipeline rankings differed.

Trace kernel first-to-last span selected G1[1]/G2[3] instead. Its 294.614 µs median was retained by the separate-mode shortlist; the merged list discarded atomic and had a 295.840 µs best span (0.416% descriptive sample loss). No significance threshold or performance target was applied. Trace span includes profiler-execution gaps and is not serving wall time.

A focused same-process nine-pair interleaving changed the kernel-sum G1 winner while retaining it in K2. That companion pre-bound metadata, used fewer annotations and a different sampling estimator; it does not isolate a cache cause or measure every separated-shortlist pair. A separate temporary single-row CSV public dispatch check selected the formal winner's exact kernel names and passed the numerical gate. No graph or AOT acceptance was performed; the experiment used recorded older Torch/Triton versions with target FlyDSL0.3.4.1.

Independent scientific review checked the report against the JSON/raw evidence, including task identity, real kernel attribution, zero placement, numerical checks and summary scope. No unresolved substantive reporting error remains. Raw traces/logs were validated then losslessly compressed; scripts and analysis pass syntax, artifact-integrity and documentation-link checks.

Plan changes: keep the current BM/accumulation compatibility partition and normal end-to-end pair authority; do not infer generally safe merging or exact additivity. Explicitly label the main-compatible kernel-sum metric, include matched elapsed/span diagnostics for deployment-latency interpretation, and verify actual production input/quantization and final-output paths. The agreed 27-search integrated-port calibration, non-default support work, numerical gate and lack of hard performance goals are unchanged. Implementation tickets 02/07 and the spec/plan/acceptance now reference these findings.
