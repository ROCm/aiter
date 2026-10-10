# Exact calibration-winner production and precompiled acceptance

All27 calibration winners passed separate public production replay, matching AOT compilation and later run-only replay:81 independent tasks, all exit0. The27 initial search plus81 winner-deployment processes have0 overlaps on physicalGPU7 UUID`GPU-5d8a72a332aa124c`/PCI`0000:f5:00.0`. The earlier failed Situv2 harness-assertion task is retained separately and is not counted among these81 successes.

Every replay selected the exact CSVG1/G2 names with explicit A4W4 input, the intended complete stage2 path and finite unchanged Torch cosine metric≤0.1. Every AOT job compiled and run-only replay used its matching cache. Input/routes/weight fingerprints match between ordinary and run-only replay, between search modes, and across allGPT configurations.

| Winner | Search kernel sum μs | Production error | Run-only error | AOT jobs | Production matched elapsed μs | Production matched trace span μs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| minimax-64-full | 76.9861 | 0.01149866 | 0.01149741 | 4 | 231.565 | 236.118 |
| minimax-64-prune | 75.8671 | 0.01149842 | 0.01149701 | 4 | 227.905 | 230.777 |
| minimax-64-staged | 76.0348 | 0.01149840 | 0.01149761 | 4 | 227.035 | 228.763 |
| minimax-128-prune | 84.5815 | 0.01132695 | 0.01132705 | 4 | 234.915 | 235.809 |
| minimax-128-staged | 84.2686 | 0.01132622 | 0.01132800 | 4 | 236.315 | 237.403 |
| minimax-512-prune | 97.7542 | 0.01129155 | 0.01129130 | 4 | 270.854 | 273.312 |
| minimax-512-staged | 101.2191 | 0.01129138 | 0.01129140 | 4 | 270.574 | 273.417 |
| minimax-1024-prune | 117.0330 | 0.01120686 | 0.01120686 | 4 | 295.045 | 297.163 |
| minimax-1024-staged | 117.6557 | 0.01120686 | 0.01120686 | 4 | 292.554 | 295.668 |
| minimax-2048-full | 173.7718 | 0.01117292 | 0.01117292 | 4 | 370.800 | 372.244 |
| minimax-2048-prune | 174.7378 | 0.01117292 | 0.01117292 | 4 | 366.632 | 368.647 |
| minimax-2048-staged | 179.6055 | 0.01117292 | 0.01117292 | 4 | 371.642 | 372.945 |
| minimax-4096-prune | 248.1153 | 0.01118485 | 0.01118485 | 4 | 435.910 | 437.799 |
| minimax-4096-staged | 248.8316 | 0.01118485 | 0.01118485 | 4 | 438.331 | 439.962 |
| minimax-8192-full | 404.8570 | 0.01116842 | 0.01116842 | 4 | 585.108 | 586.190 |
| minimax-8192-prune | 406.7968 | 0.01116842 | 0.01116842 | 4 | 577.797 | 580.006 |
| minimax-8192-staged | 406.5586 | 0.01116842 | 0.01116842 | 4 | 580.107 | 582.422 |
| gpt-2048-default-prune | 115.6761 | 0.01146977 | 0.01146979 | 4 | 302.623 | 305.114 |
| gpt-2048-default-staged | 121.5918 | 0.01146779 | 0.01146779 | 4 | 311.783 | 313.868 |
| gpt-2048-fp8-prune | 101.8640 | 0.01181910 | 0.01181910 | 4 | 296.114 | 299.104 |
| gpt-2048-fp8-staged | 110.5011 | 0.01181910 | 0.01181910 | 4 | 302.163 | 304.894 |
| gpt-2048-fp4-prune | 111.0863 | 0.01814995 | 0.01814995 | 4 | 303.664 | 306.081 |
| gpt-2048-fp4-staged | 110.3552 | 0.01814995 | 0.01814995 | 4 | 299.993 | 303.102 |
| gpt-2048-limit5-prune | 117.0535 | 0.01146980 | 0.01146979 | 4 | 298.823 | 300.942 |
| gpt-2048-limit5-staged | 122.0186 | 0.01146779 | 0.01146779 | 4 | 313.343 | 318.876 |
| gpt-2048-combined5-prune | 101.9686 | 0.01181910 | 0.01181910 | 4 | 296.723 | 299.645 |
| gpt-2048-combined5-staged | 110.4887 | 0.01181910 | 0.01181910 | 4 | 307.393 | 309.487 |

Search kernel-duration sum is the CSV selection metric. The matched elapsed and real-kernel trace-span columns come from synchronized production calls; they are companion diagnostics, not replacements for the search metric. Synthetic GPU annotations are excluded by the replay harness. AOT compile jobs retain complete effective dtype, persistence, activation parameters and buffer/scale keys in deployment-summary.json and each original aot.json.

ConfiguredFP8 reduce winners record unweighted packedFP8 routes and final exactly-once route weighting; configuredFP4 scatter winners record packed payload/scales and complete quantized scatter. Both-flags reduce winners retainFP8 output. Required scatter/atomic combinations and active limit5 clamp fixtures remain separately covered by ticket03; winner replay supplies the same configured limit as its search.

Corrected omitted-argumentSituv2 production/run-only retries also passed with actual4/25 defaults, retained separately in situv2-public-defaults.md. Targeted alternating pair checks and finalStandards/Spec review are separate remaining gates; this completed winner-deployment record does not close ticket04 before them.
