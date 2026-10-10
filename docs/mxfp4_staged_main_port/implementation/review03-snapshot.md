# Ticket 03 snapshot and routing increment review

Reviewed fixed diff `9c035b435...2308ce0bf` in
`/root/workspace/tmp_dev/aiter-mxfp4-03`, against current
`.scratch/mxfp4-staged-main-port/spec.md`,
`issues/03-configured-execution-and-aot.md`, `environment-options.md`, root
`CONTEXT.md`, and ADR 0001. Followed `karpathy-guidelines`.

No concrete actionable bug was found in this increment.

Inspected configuration capture at CLI parsing, preprocessing, tuning invocation,
and reconstructed worker shape entry; the args-carried snapshot is reused by
pair validation and direct G1/reference/bootstrap screening. Invalid nonpositive
or nonfinite limits fail before shape work. Configured Situv2 is classified
unsupported before already-tuned dedup and at shape execution, matching the
production named-dispatch restriction. Omitted/empty defaults retain production
normalization (Swiglu7, other supported default activations unclamped).

Inspected pair and G2 screening forwarding of original `topk_weights` through
`_mxfp4_a4w4_stage2_fw` into the existing layout-v2 wrapper. The existing wrapper
uses the same stable environment interpretation as the captured switch flags:
FP8 reduce defers route weighting to final reduction; eligible FP4 scatter uses
packed values/scales and quantized scatter; atomic/default paths retain their
existing behavior. Candidate enumeration/grouping, search defaults, fallback,
process ownership, winner schema and timing assignment were not changed in this
increment.

The committed evidence `03/01-snapshot-checks.txt` records32 controlled checks
passing with the retained system Triton warning. Reviewed the tests, including
spawned CPU worker snapshot/inherited-production-settings parity and real
production-wrapper branch observation with fake tensors. These are controlled
behavior checks, not GPU numerical, clamp, exact deployment or AOT evidence.

No tests or GPU tasks were run for this read-only review. Uncommitted AOT and
configured-replay work was expressly excluded. This review covers only the
snapshot/routing increment; final integration Standards/Spec review and remaining
ticket03 AOT/numerical acceptance remain required.
