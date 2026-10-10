# Execution-environment support work

Status: ready-for-agent

The [spec.md](spec.md) includes FP8 reduce, FP4 scatter and non-default swiglu_limit with independent path verification. Reviewer decisions are incorporated; the spec and approved [four tickets](issues/README.md) are ready for implementation. No implementation has started; the [work map](map.md) tracks ticket state. This document retains verified execution/AOT gaps and follows the spec's scope.

All facts below refer to main `882ea79dd7`. These are existing execution paths to align with staged tuning, rather than new mathematical operators to implement.

| Option | Confirmed gap | Work to support it | Meaningful acceptance |
| --- | --- | --- | --- |
| `AITER_FLYDSL_STAGE2_FP8=1` | Production sends `topk_weights` and defers route weighting to the FP8 reduction. The tuner omits it and thus disables FP8 route output even when the environment flag is enabled. | Pass production-equivalent route weights in pair validation and G2 screening; keep the final reduction inside G2 timing. Preserve reference semantics and check the existing AOT FP8-reduce job. | Prove enabled tuning and production select the FP8-reduce path; check routing weights apply exactly once, finite correctness and matching AOT/cache behavior. |
| Non-default `AITER_MXFP4_TUNE_SWIGLU_LIMIT` | Main pair validation/reference carry the configured limit, but old staged direct G1 screening/bootstrap do not. AOT generates fixed Swiglu 7 / other activations infinity. | Normalize the limit consistently with production, propagate it through G1 screening and bootstrap/reference, and supply it to AOT without adding winner CSV columns. Keep production's supported activation/limit combinations. | Check default and a non-default positive limit, actual clamp behavior, screening/bootstrap/final validation consistency and a matching AOT compile key. Situv2 plus an unsupported limit must not tune as a deployable pair. |
| `AITER_MXFP4_INTERMEDIATE=1` | Layout-v2 scatter runtime emits packed FP4 values and E8M0 scales, then quantized scatter. Current layout-v2 AOT emits BF16 scatter output only. | Reuse the production wrapper for G2 screening/validation; generate the effective FP4 scatter AOT job and its packed output/scale buffers with production-equivalent eligibility, including existing bias handling. Align environment gate interpretation. | Explicitly exercise a BM128 scatter pair, including output/scales and quantized scatter; verify E2E error against the agreed reference, matching compiled key and production replay. Measure enabled and disabled configurations separately. |

FP8-reduce and custom-limit changes are primarily parameter propagation and AOT integration. FP4 scatter additionally requires output/scales preparation and separate lossy-path numerics. No wall-time estimate has been validated yet.

Normal inference may JIT a kernel missing from AOT; the consequence is missing precompilation and first-call compilation, not an automatic numerical failure. A deployment requiring only precompiled entries cannot rely on that fallback. AOT completion alone does not establish the same effective compile key as production.

These settings can change the best pipeline pair or numerical behavior while leaving the CSV names unchanged. Tune, calibrate, compile and deploy under the same effective settings, and record them in the independent validation artifacts. Supporting both stage2 switches requires testing their combined configuration: reduce takes the FP8 branch while scatter takes the FP4 branch.

The tuner uses `AITER_MXFP4_TUNE_SWIGLU_LIMIT`, resolved with stage2 switches once per invocation; AOT receives the corresponding limit and production receives its API value. The user assigns winner-file/settings consistency to the caller. No configuration sidecar, automatic binding or mismatch rejection is introduced; independent validation/calibration records retain their settings and separate outputs as already required. See [reviewer resolution](reviewer-comments.md) for this explicit scope decision.

Production reads `AITER_FLYDSL_STAGE2_FP8` and `AITER_MXFP4_INTERMEDIATE` on each wrapper call. Keep those variables and the tuning-limit variable unchanged in the parent and workers from snapshot capture until the tuning run ends; verify each worker's inherited effective environment and actual output path agree with the snapshot. Configure separate AOT/replay invocations before they start. This contract requires stable invocation settings rather than a new runtime mutation detector.

The caller must account for existing rows: a limit 5 invocation targeting a CSV with a same-key default-limit result can silently skip it without `--all`, or replace it with a valid retuned winner under `--all`. Output-switch differences have the same key collision. Choose the intended configuration's CSV and avoid incompatible reuse; `--all` does not provide configuration isolation.

Implement all three within the dedicated parameter/AOT slice, with separate acceptance per execution path. Preserve the public search CLI, main-compatible defaults, candidate scope and group/fallback policy already agreed in [spec.md](spec.md).
