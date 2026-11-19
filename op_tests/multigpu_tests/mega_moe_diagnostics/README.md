# Mega-MoE diagnostics

See [the handoff README](../../../README_dispatch_pull.md) for layout invariants,
validation status, exact workload, reproduction commands and next actions.

`run.sh affinity LABEL` uses the label prefix to choose the operation:

- `check_src*`: the repository's source-rank shuffle correctness test, Flash shape.
- `events*`: native E2E, then explicit HIP graph event nodes for plan/ready/GEMM1/GEMM2.
- `check*`: existing pull correctness with fine gather-cycle probes.
- Other labels: native timing with fine gather-cycle probes, then trace export.

Cycle modes generate ignored module overlays from current repository source using
`build_overlay.py`; exact replacement anchors intentionally fail if the source
has changed incompatibly. They require external-ready, M64 flags, one-TG
producers, M256, H5120/I2304, and the original gather path. Packed modes use the
repository code directly, not that cycle overlay.

Do not regenerate an overlay while a process is importing/JIT-compiling it.
Every run records its source and environment under `ipf*_p*_LABEL`. Use unique
labels, and compare like timing scopes. See REVIEW.md for cycle probe ordering
and instrumentation caveats. Source-shuffle prelaunch reviews are also recorded
in the main kernel README. GPU validation of the newly committed source-shuffle
path is pending.
