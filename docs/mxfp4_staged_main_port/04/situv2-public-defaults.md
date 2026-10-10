# Omitted public Situv2 parameters on the actual MXFP4 path

Two independent retry tasks passed exact public production and later matching precompiled replay of the prescribedA4W4 Situv2 pair. Both omit public `beta`/`linear_beta` kwargs, capture actualG1 beta4/linear_beta25, compare against the same ordinary Torch4/25 reference, verify the exact selectedG1/G2 names and A4W4 input, and retain finite numerical outputs. Production cosine_diff was0.01113366; run-only replay was0.01113105, both below0.1. The run-only process inherited`FLYDSL_RUNTIME_RUN_ONLY=1` and used ticket03's matchingAOT cache, whose jobs retain the4/25 variant.

The first task under `gpu/situv2-public-defaults/replay-task/` failed a harness assertion expecting1/1. The generic activation normalization uses1/1, but `fused_moe_2stages` explicitly substitutes the establishedMXFP4 defaults4/25 when API beta arguments are omitted. The runtime was correct; the acceptance harness and documentation were corrected in476af59e0. The failure and traceback are retained. A CPU production-boundary check in eacf9177e separately proves omitted4/25 and explicit1/1 forwarding.

Retries use fresh evidence paths `replay-r1-task/` and `precompiled-r1-task/`; no original task record was overwritten. The prescribed-pair CSV retains its independently measured4/25 tuning timing, and retry results label synchronized kernel-sum/span/elapsed diagnostics separately. AOT's explicit1/1 job remains compatible but is not claimed as the omittedAPI execution.

Together with ticket03's explicit4/25 specified-pair checks, this supplies the required defaultSituv2 production path acceptance. It adds no tuning search or configurableSituv2 behavior.
