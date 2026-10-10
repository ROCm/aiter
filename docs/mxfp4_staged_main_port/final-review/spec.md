# Final Spec-axis review

Pinned review HEAD: `5b09efe5f2843edd0ce104531c55f2fe9e41e3e5`.
Fixed base: `882ea79dd7cc7091f859a5bedac914523ac77bbd`.
Diff: `git diff 882ea79dd7cc7091f859a5bedac914523ac77bbd...5b09efe5f2843edd0ce104531c55f2fe9e41e3e5`.
The supplied final-review-commits.txt and final-review-files.txt were inspected. Review followed karpathy-guidelines. Read-only code/evidence review; no GPU work and no code or tracker modifications.

## Finding

One low-priority documentation correction remains for final closure. Ticket 04 line 18 requires updated help/docs, and spec line 223 requires accurate current-port acceptance distinct from historical evidence. `.scratch/mxfp4-staged-main-port/acceptance.md:113` still says **“No measured total wall time exists yet.”** Current calibration artifacts record all 27 wall times and complete default/extended totals. Replace the obsolete sentence with a link to those observed results, retaining the distinction between estimates and measured time. Separately, `docs/mxfp4_staged_main_port/04/summarize_deployment.py:180` still generates a final paragraph calling completed targeted replays and corrected Situv2 checks “remaining gates,” although the checked-in deployment-summary.md line 41 accurately states those are complete and only final review remains. Update the generator so documented offline reproduction does not restore obsolete status. Final closure can then update the remaining review gate normally. No runtime change or additional GPU search is required for this correction.

No missing/incorrect functional requirement and no unrequested runtime scope found.

## Code audit

- Existing selector accepts prune/full/staged, defaults to prune. Explicit screening top-K requires explicit staged, accepts positive integers, defaults to 2. Existing grouped/tuner checks are retained.
- Candidate enumeration `_candidate_rows` and `_g1_variants` retain fixed-base order and legality. Staged uses `full_search=False` through that exact source. Native/re-tile GEMM2 remain excluded; I384 is admitted by layout-v2 K128. Kernel name identities retain all launch distinctions. Sorting/G1/G2 BM and stage dtype/activation/inline-quant contracts are checked before grouping. BM16 derived native scales match production.
- G1 scores are computed once per BM and reused across atomic/non-atomic groups; G2 scores distinguish accumulation mode, combining reduce/scatter. Stable ties use admitted order and never expand K. Shortlist selection walks the original pair order.
- References come only from admitted same-BM producers/atomic consumers, preferred configurations then remaining original-order candidates. Consumers self-test on independent Torch-produced sorted payload/scales; producer/consumer pairs receive untimed numerical bootstrap. Bootstrap outcomes do not enter pair cache or become winners.
- Unusable bootstrap, empty successful screens and invalid/non-finite shortlist results trigger exhaustive admitted group fallback. It includes returned screening failures and exhausts all untested pairs before choosing fastest fallback. Pair cache stores scalar returned time/failure text, with no exception traceback or intermediate retention. Every admitted group stays eligible. AcceleratorError aborts the shape; normal returned failures remain recoverable. Actual pair `_run_candidate` checks finite configured cosine gate and positive finite time, writing full-pipeline us to us1/us and leaving us2 zero.
- G1 direct screening launch times its actual kernel after setup. Setup matches BM16 inline and BM32/64/128 fused-prequantization, complete tuning parameters and activation limit. Each G2 attempt allocates its own producer intermediate during untimed setup; timed wrapper includes atomic zeroing or full non-atomic reduction/scatter. Both screening and pair paths forward original topk_weights for exactly-once deferred FP8 route weighting.
- Effective configuration captures unset/empty/default/custom limits and exact opt-in flags once, passes snapshot through args to spawned workers, and preserves the stated caller-owned stable environment. No runtime mutation detector, CSV sidecar/binding/key expansion or collision rejection was added.
- AOT reads call-time flags, normalizes default/custom limit, retains omitted entry compatibility and shared bias behavior, creates correct FP4 packed target/scales and dtype keys, and forwards persistence/other stage launch settings. Situv2 configured limit is rejected; existing default variants remain. Actual public MXFP4 omitted beta override in fused_moe.py lines 4315 onward is 4/25; corrected replay checks that boundary rather than generic 1/1 normalization.
- Runtime preflight checks participating devices before work and derives gfx/CU from each assigned torch runtime device. Prune/full single-slot uses current GPU and parallel path first selected devices; staged first visible devices and existing cap normalization are retained. Unused unsupported visible devices do not reject. Other tuners avoid MXFP4 architecture restriction.
- Staged private Pipe/process supervisor keeps GPU owned after result arrival until process exit/join/exit status/close; EOF/exception/fatal exit handling, bounded active resources, cancellation cleanup and owned SIGALRM restoration are covered. No parent hard deadline or prune/full Queue redesign was introduced.
- Input-only gate metadata survives retune selection, and unsupported rows are checked before already-tuned dedup. Shared normalization preserves explicit gfx/legacy identity and ordinary nt/numeric/stage timing semantics; tagged fallback filtering occurs before projection. Summary uses stable shape keys and retains nonzero failed/unfinished exit with later valid persistence.

## Independent evidence checks

Read manifest, counts, full-anchor estimates, calibration-analysis/summary, targeted-summary, required fixture records, deployment-summary, and exact controller/replay/AOT scripts. Parsed every search/winner replay/AOT/run-only artifact structurally, plus sampled compressed original search logs and real kernel traces.

- 27 unique manifest IDs, correct 17 MiniMax + 10 GPT configuration/mode searches. All 27 valid finite positive winners; 13,296 logged pair validations.
- 22 required pair fixture records pass, covering I384 BM16/32/64/128 atomic/reduce, BM128 scatter, GPT K128/K256 FP8, FP4 BN128/BN256, both flags default/custom, atomic unchanged, active clamp checks and Situv2.
- All 27 winners have exact CSV G1/G2 names in public replay and run-only replay, finite ordinary Torch cosine <=0.1, explicit FP4 input contract, matching replay/run-only fingerprints and AOT cache. All AOT jobs return non-null compile time; RUN_ONLY is 1 in later replay.
- 121 ticket04 GPU task records checked for strictly serial nonoverlapping start/exit ordering and recorded UUID selector/logical map/idle sample/PCI. 120 succeeded; the 1 diagnosed original Situv2 harness assertion is preserved. Required GPU identity is GPU-5d8a72a332aa124c / 0000:f5:00.0 throughout. This review adds no GPU tasks.
- Three full anchors each record estimate timestamp before process startup and 1,368 returned validations. Actual wall times include process/preparation/compile/warmup/profile costs; conservative lower estimate misses are disclosed honestly.
- Sampled compressed staged logs contain positive screens, expected validation markers and final summaries. Sampled FP8/FP4/default/combined replay traces retain the exact declared count of real kernel events, excluding synthetic annotations.
- 10 targeted tasks retain 60 normal kernel-sum profiles, medians/dispersion/ranks and concrete shortlist losses/noisy sign reversals; there is no unrequested re-search or policy/K change.
- Calibration-summary's unavailable elapsed fields concern search measurement; separate deployment-summary supplies matched synchronized elapsed/span diagnostics, correctly distinguished from continuous tuner kernel-duration sums.
- Runtime code files have no diff from the ticket03 tested implementation commit `4e5d8bdc213880549e144a28d017a208c4661488` or first-calibration `6c69bb11beae1414e7d2e8468d737019b1c15bff` to pinned review HEAD, supporting reuse of their retained GPU evidence.

## Conclusion

Spec axis: 1 low-priority documentation reproduction/status correction, 0 actionable runtime findings, 0 scope-creep findings. Update stale prose/generator, then complete final tracker closure after both review axes and any required fixes. Do not rerun the 27 GPU searches for documentation-only changes.
