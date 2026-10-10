# 03: Support configured output paths and matching AOT

**What to build:** Custom activation limits, FP8 route reduction, FP4 scatter and their supported combinations work consistently through screening, references, validation, winner CSV, exact production replay and precompiled execution.

**Blocked by:** 02: Deliver complete default-environment staged search.

**Status:** claimed

- [ ] Read the existing tuner AITER_MXFP4_TUNE_SWIGLU_LIMIT and stage2 settings once per invocation, normalize defaults/configured production-valid values and forward the fixed snapshot through workers, direct G1, bootstrap, Torch references and pair validation in every search mode; retain default Situv2 parameters and reject unsupported activation/limit combinations.
- [ ] Keep the parent and workers' AITER_MXFP4_TUNE_SWIGLU_LIMIT, AITER_FLYDSL_STAGE2_FP8 and AITER_MXFP4_INTERMEDIATE environment unchanged from snapshot capture until the tuning run ends. Production retains its per-call stage2 environment reads, so worker inherited effective settings must match the snapshot. Test default/unset/empty and enabled/combined settings through spawned-worker and production-path behavior; verify the path actually selected agrees with the snapshot. No new runtime binding or mutation-detection mechanism is required.
- [ ] Add MXFP4 AOT --swiglu-limit with compatible omitted callers and matching standalone/batch configuration; preserve CSV columns and production's explicit activation parameter interface.
- [ ] Document that the caller associates the correct winner CSV with tuning/AOT/production limits and output switches and sets the environment before each invocation. A limit 5 run can silently skip a same-key default-limit row without --all, or replace it with a valid winner under --all; output-switch changes have the same hazard. Add no sidecar, automatic file binding or mismatch-rejection mechanism. Acceptance and calibration provide explicitly matching settings and separate result files.
- [ ] Demonstrate actual clamp behavior for limit5 and matching reference, production names and compiled entries; defaults remain compatible and effective-configuration scores/cache are not shared.
- [ ] FP8 reduce receives topk_weights where needed, actually emits FP8 payload/scales and applies routing weights exactly once during complete final reduction. Screening and validation match production rather than measuring BF16 fallback.
- [ ] Enabled FP4 scatter uses eligible BM128 packed values/E8M0 scales and quantized scatter to final BF16; complete output work stays timed and reduce/scatter remain in the agreed non-atomic group.
- [ ] Generate matching FP4-scatter AOT buffers/dtype/persistence/eligibility/keys, retaining shared BF16/FP8 and bias behavior. Preserve disabled/default and atomic paths.
- [ ] Both stage2 flags select FP8 reduce versus FP4 scatter correctly with default and custom limit; actual A4W4 dtype/path remains explicit for Swiglu/Situv2 replay.
- [ ] Focused forwarding/job/behavior tests and single-card numerical checks cover exactly-once nonuniform route weights, clamp behavior, FP4 N128/N256 payload/scales, negative configurations and matching precompiled entries.
- [ ] Each extension and their combinations provide real exact named production/CSV/AOT evidence, passing the unchanged Torch numerical gate and positive finite timing. A fallback OK or new JIT key is not precompiled acceptance.
- [ ] Deliver reproducible matching environment/limit commands, relevant checks and current implementation evidence under the [specification](../spec.md); retain normal fallback/process/CSV behavior while adding these paths.
