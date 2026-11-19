# A03/A09 mono review and acceptance

## Current 1 x 256 result: 2026-11-19

Tested commit: `ce7b69057f6da66c3291ec89361cdc202253034d`. Only ASM kernels must avoid s104/s105.
HIP/FlyDSL/Triton have no such requirement. A03/A09 use Triton/Gluon and
no longer impose the 104-SGPR allocation limit. Grid and task batches remain
`(1,256)`, 128 threads/TG, and `task = tg + j*256`.

- [x] All 108 UTs pass with zero skips; frozen native hashes pass.
- [x] Actual 336 PS dispatches use 1 x 256 TGs, 128 threads/TG and wave32.
- [x] Recorded actual PS JIT variants keep DEV activation loads and DEV + HT
  output stores. Metadata confirms no explicit SGPR cap.
- [x] Four-profile timing: 22 unique cases, 15 trials each, hot/rotating,
  paired native/PS pools and Graph method, no profiling during timing.
- [ ] Full performance acceptance. Decisions: {'regression': 19, 'inconclusive': 3}.
- [ ] Main-kernel call/join, simultaneous residency, common fused cluster and
  model path. Standalone results do not prove these.

Primary c16: a03 hot regression (+6.71%); a03 rotating regression (+3.28%); a09 hot inconclusive (+0.63%); a09 rotating inconclusive (+0.94%); a09-qat hot inconclusive (+0.27%); a09-qat rotating regression (+0.62%).

[Raw results and JIT evidence](mono-persistent-results/2026-11-19-grid1x256-unrestricted-sgpr) record this revision.
Earlier capped candidates and 16 x 16 results below are historical.

## Previous grid-only candidate: 2026-11-19

The requirement is fixed 256 TGs. The selected A03/A09 PS grid is now
`(1,256)`, with the TG id from `program_id(1)`. Each TG has 128 threads.
Batch selection and memory scope are unchanged. The earlier 16 x 16 checklist
below records that tested revision only. New GPU validation is pending.

Candidate `2dc6f7c35` passes Black, Ruff, frozen native hashes and 20 offline
builds. Static checks of 12 PS variants pass. The GPU attempt could not acquire
the shared lock, with four earlier waiters, and stopped before any GPU launch.
The [manifest](mono-persistent-results/2026-11-19-grid1x256/manifest.json)
marks UT, actual dispatch and timing as pending. Resume the saved runner in
`/home/weizhu15/workspace/mono-grid1x256-20261119` in the next exclusive window.

## Previous 16 x 16 acceptance checklist: 2026-11-19

[Published raw results and JIT evidence](mono-persistent-results/2026-11-19)
include all four target profiles and the rejected weight-load trial.

Tested code: `9a294c8c1661c5c69e8b75ea8c8ce7fdb7d977c8`.
The [current contract](mono-persistent-validation.md#current-acceptance-contract-2026-11-19)
supersedes the historical scope. The full acceptance item does not pass.

- [ ] Complete A03/A09 mono acceptance.
- [x] Both ops keep one original task per iteration at task count <= 256.
- [x] Real standalone physical 16 x 16 grid, 256 TGs, 128 threads/TG, wave32.
- [x] Original task ownership `task = tg + j*256`, including batched paths.
- [x] All 108 UTs pass with no skips; prior tests remain. Native bytes,
  input/weight integrity, guards and Graph replay pass in the covered cases.
- [x] All 22 recorded real PS JIT variants avoid s104/s105 operands.
- [x] Actual op activation loads use DEV and output payload/scale stores
  use DEV + HT. Frozen native sources are unchanged.
- [x] Final 22 unique target-shape timing cases, 15 trials each, hot/rotating,
  common native/PS input pools and Graph method, no profiling during timing.
- [ ] c16 no-regression: A03 fails (+7.18% hot, +2.83% rotating). A09 FP8
  is inconclusive; QAT improves. Inconclusive does not establish equivalence.
- [ ] c64/c128/c256 secondary acceptance: multiple regressions remain.
- [ ] Every possible source-supported shape. The finite suite is coverage,
  not a proof; wide-stride fallback builds have no GPU coverage yet.
- [ ] All 256 TGs can reside together; idle TGs return to the caller's join.
- [ ] All ops use the same physical 1 x 1 cluster in the final fused launch.
- [ ] Actual call/return preserves the main kernel's s104/s105 values.
- [ ] Consumer loads follow the required join. Flag store/load scope and
  writer completion, single-lane publication and reader convergence pass.
- [ ] Actual TP4/EP4 model shape/Graph trace and model-path performance.

Keep the measured candidate. A03 counts rows; A09 counts ceil(numel/1024).
Do not force every task count into a four-task batch. Only target-model shapes
are performance priorities; other shapes retain correctness coverage.

The ordinary-weight-load trial `e055dedfb` passed correctness but slowed A03.
It is rejected by `9a294c8c1`, which restores the earlier measured source tree.
The restored tree was tested again in full. No untested rollback is accepted.
The final timing matrix is from one complete run, not best samples from
multiple runs. All 22 cases have no foreign GPU PID in the 100 ms samples.

The recipe/native baseline is confirmed: c16/c64 TP4 + DSpark5,
c128/c256 EP4 + DP attention. Current measurements use source-derived full,
balanced verify shapes. A03 c256 shares the measured c64 shape. One visible
GPU cannot certify four-GPU model execution; the op data remain valid as
isolated measurements. The main-kernel integration is still absent.

Full report and evidence: `/home/weizhu15/workspace/mono-criteria-20261119`.
See `REPORT.zh.md`, `final/performance-table.md`, `final/all-real-jit-audit.json`,
`final/launch-current-gate.json`, `final/logs/`, and `round2/` for the rejected
trial. The validation guide contains the full final performance table.

## Adaptive PS result: 2026-11-18

Tested code: `96826c2c50f69d50484d8e64a250d7793268ce9d`. This result supersedes the fixed-four-task
experiment below. It does not claim a universal performance win.

A03 counts rows. It selects one task through 767 rows and four tasks above
that bound. The two-row main batch regressed in measured cases and is not
selected. A09 counts `ceil(numel/1024)`: one task through 256, two tasks
through 512, and four tasks above 512. Thus A09 width-2048 rows 96/257/513
select 1/4/4. The two-task path balances one/two-task workers. The four-task
path keeps the direct cyclic loop; splitting its tail into smaller batches
did not give a general win. Configs use the existing JSON config loaders.

All 89 GPU tests passed with zero skips. All original 52 tests remain, with
37 new threshold, tail and Graph cases. All 12 frozen native hashes and
20 offline compile variants passed. Native source and byte assertions did
not change. The final Graph trace recorded 336 PS dispatches across eight
kernel names. All had grid `(32768,1,1)` and workgroup `(128,1,1)`.
Actual four-task ISA instruction sequences match the prior PS4 kernels.
All selected PS paths have zero scratch spill.

Against old single-task PS, the 108 four-provider cases have 77 improvements,
31 inconclusive results and no regressions under the stated rule. Inconclusive
does not prove equivalence. Four cases still regress against fixed PS4.

The original paired matrix has 18 cases. A separate four-provider matrix has
108 cases, including threshold boundaries and larger rows. Each selected case
has 15 trials and no foreign GPU PID observed by 100 ms sampling. Timing uses
the shared GPU lock, common input pools and graph timing without profiling.
Values below are median / P10 / P90 in microseconds.

| Op | Rows | Cache | Native median / P10 / P90 (us) | Adaptive median / P10 / P90 (us) | Adaptive/native | Decision |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| a03 | 96 | hot | 1.577154 / 1.565078 / 1.594461 | 1.598902 / 1.566974 / 1.611840 | 1.013790 | inconclusive |
| a03 | 96 | rotating | 1.919952 / 1.915746 / 1.925917 | 1.915368 / 1.913960 / 1.923626 | 0.997613 | inconclusive |
| a03 | 257 | hot | 1.692482 / 1.673074 / 1.709228 | 2.177104 / 1.895471 / 2.196804 | 1.286338 | regression |
| a03 | 257 | rotating | 2.034134 / 2.023184 / 2.046160 | 2.299303 / 2.244865 / 2.446154 | 1.130360 | regression |
| a03 | 513 | hot | 1.817402 / 1.810269 / 1.835534 | 2.592026 / 2.585223 / 2.606715 | 1.426226 | regression |
| a03 | 513 | rotating | 2.242839 / 2.235572 / 2.246908 | 3.085473 / 3.058017 / 3.127181 | 1.375700 | regression |
| a09 | 96 | hot | 1.352658 / 1.327280 / 1.358627 | 1.369643 / 1.367575 / 1.371885 | 1.012557 | inconclusive |
| a09 | 96 | rotating | 1.686201 / 1.675495 / 1.697029 | 1.670057 / 1.655876 / 1.683546 | 0.990426 | inconclusive |
| a09 | 257 | hot | 1.498188 / 1.490096 / 1.509130 | 1.327011 / 1.323278 / 1.328867 | 0.885744 | improvement |
| a09 | 257 | rotating | 2.042094 / 2.034590 / 2.092444 | 1.837972 / 1.825438 / 1.844247 | 0.900043 | improvement |
| a09 | 513 | hot | 1.636515 / 1.630898 / 1.643407 | 1.612555 / 1.604425 / 1.620328 | 0.985359 | improvement |
| a09 | 513 | rotating | 2.737604 / 2.733202 / 2.743499 | 2.007082 / 1.996708 / 2.020106 | 0.733153 | improvement |
| a09-qat | 96 | hot | 1.250873 / 1.245912 / 1.252150 | 1.227627 / 1.226595 / 1.228271 | 0.981416 | improvement |
| a09-qat | 96 | rotating | 1.600152 / 1.598866 / 1.604137 | 1.579173 / 1.577714 / 1.580304 | 0.986889 | improvement |
| a09-qat | 257 | hot | 1.307610 / 1.305376 / 1.308629 | 1.332011 / 1.329461 / 1.334294 | 1.018660 | regression |
| a09-qat | 257 | rotating | 1.691141 / 1.689636 / 1.692849 | 1.751627 / 1.750559 / 1.754123 | 1.035767 | regression |
| a09-qat | 513 | hot | 1.419213 / 1.418548 / 1.424220 | 1.721550 / 1.686713 / 1.747807 | 1.213031 | regression |
| a09-qat | 513 | rotating | 1.794043 / 1.792317 / 1.795352 | 1.856036 / 1.849132 / 1.859300 | 1.034555 | regression |

Some cases still regress against native, so standalone performance acceptance
does not pass. Task-count dispatch alone cannot promise zero regression for
every valid shape. Other widths have correctness coverage, not a broad
performance guarantee. Full mono ABI, DEV/HT, reserved-register, joins and
model integration remain outside this change.

Raw evidence: `/home/weizhu15/workspace/mono-adaptive-20261118`. See `REPORT.zh.md`,
`performance-four-way-table.md`, `performance-accepted.json`,
`performance-selections.json`, `launch-final-gate.json`, `jit-final-review.json`
and `logs/`. Earlier strategies and their rejected results remain in that
directory; they are not mixed into the final timing matrix.

## Four-task PS experiment: 2026-11-18

This section records the current user-requested four-task experiment. The
sections below record the earlier one-task implementation and its handoff.
The older static resource limits describe that earlier selection. The new
experiment deliberately increases per-wave work and does not meet those old
VGPR/code-size limits. It is not a successful performance acceptance.

Tested code: `25c1092044557ab3e0c5f90374cb2e4b06cc6142`. Baseline PS1: `6f85fc6d8dbc4c01cf4d40a6e48f1990c7952056`.
A03 handles four adjacent rows per TG iteration. A09 handles four adjacent
1024-element tasks (two rows at width 2048). Both keep 256 TGs and 128 threads
per TG. The actual grid is one-dimensional, with work-item grid `(32768,1,1)`.

All 52 GPU tests passed with zero skips. These retain all original 45 cases
and add seven A03 cases at the new 1024-row loop boundary. Frozen native hashes
and all 12 compile variants passed. The first A03 attempt failed six byte
checks because its automatic reduction layout changed. Explicit Gluon layouts
fixed these cases; no native source or tolerance changed. Non-PS A03 is unchanged.

Actual Graph tracing recorded 48 dispatches for each of A03 PS, A09 FP8 PS and
A09 QAT PS. All used 256 TGs x 128 threads. Actual ISA issues four A09 loads
before the first load-result wait. A03 issues four rows of Q/K loads before
that wait. PS4 VGPR/shared/scratch are A03 168/64/0, FP8 76/0/0 and QAT 60/0/0.
These resource fields do not measure residency.

Timing used the shared GPU lock and idle checks, without profiling. Each
selected case has 15 trials and no foreign GPU PID in its 100 ms process
samples. Two contaminated paired cases were discarded and repeated in full.
Native/PS4 use one input pool and alternating order. Values are microseconds.

| Op | Rows | Cache | Native median / P10 / P90 (us) | PS4 median / P10 / P90 (us) | PS4/native | Decision |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| a03 | 96 | hot | 1.556748 / 1.553161 / 1.562145 | 2.892640 / 2.878984 / 2.896413 | 1.858130 | regression |
| a03 | 96 | rotating | 1.918209 / 1.915257 / 1.920165 | 3.276872 / 3.269959 / 3.365023 | 1.708298 | regression |
| a03 | 257 | hot | 1.685472 / 1.661242 / 1.691207 | 2.963542 / 2.950575 / 2.971481 | 1.758286 | regression |
| a03 | 257 | rotating | 2.036438 / 2.026278 / 2.053536 | 3.508891 / 3.504066 / 3.515869 | 1.723053 | regression |
| a03 | 513 | hot | 1.840280 / 1.787212 / 1.846371 | 3.001283 / 2.998181 / 3.013520 | 1.630883 | regression |
| a03 | 513 | rotating | 2.242006 / 2.234880 / 2.245511 | 3.675697 / 3.672838 / 3.678854 | 1.639468 | regression |
| a09 | 96 | hot | 1.399276 / 1.392892 / 1.408441 | 1.444631 / 1.323942 / 1.454359 | 1.032414 | inconclusive |
| a09 | 96 | rotating | 1.732855 / 1.725823 / 1.741467 | 1.719751 / 1.717479 / 1.722751 | 0.992438 | inconclusive |
| a09 | 257 | hot | 1.538324 / 1.492725 / 1.540745 | 1.507490 / 1.500089 / 1.512898 | 0.979956 | inconclusive |
| a09 | 257 | rotating | 2.002499 / 1.984849 / 2.009294 | 1.739041 / 1.737788 / 1.740148 | 0.868435 | improvement |
| a09 | 513 | hot | 1.637156 / 1.631263 / 1.647279 | 1.452247 / 1.431756 / 1.456366 | 0.887055 | improvement |
| a09 | 513 | rotating | 2.707783 / 2.699110 / 2.715536 | 2.026919 / 2.015178 / 2.044917 | 0.748553 | improvement |
| a09-qat | 96 | hot | 1.257856 / 1.255068 / 1.258948 | 1.302819 / 1.301787 / 1.307620 | 1.035746 | regression |
| a09-qat | 96 | rotating | 1.614831 / 1.613323 / 1.616599 | 1.679771 / 1.678194 / 1.681111 | 1.040215 | regression |
| a09-qat | 257 | hot | 1.314621 / 1.313751 / 1.316103 | 1.341153 / 1.339088 / 1.343135 | 1.020182 | regression |
| a09-qat | 257 | rotating | 1.692847 / 1.691781 / 1.693656 | 1.754295 / 1.753258 / 1.756638 | 1.036299 | regression |
| a09-qat | 513 | hot | 1.427337 / 1.425034 / 1.431802 | 1.728108 / 1.712330 / 1.747755 | 1.210722 | regression |
| a09-qat | 513 | rotating | 1.808997 / 1.805118 / 1.811625 | 2.019332 / 1.999071 / 2.059459 | 1.116272 | regression |

A03 regresses. A09 FP8 improves some cases but does not pass all six cases.
A09 QAT remains slower than native. The separate three-provider comparison
shows that four-task PS helps A09 at 257/513 rows relative to old PS. That gain
does not override a native regression. A03 gains no benefit from this mapping.
For A03, useful TG counts at 96/257/513 rows fall to 24/65/129; registers and
per-TG work increase. A09 issues more independent loads and reduces loop trips.
PS4 ATT, per-WGP residency and exact residual cycle attribution remain untested.

Raw evidence and full before/after statistics: `/home/weizhu15/workspace/mono-ps4-20261118`.
See `REPORT.zh.md`, `performance-accepted.json`, `performance-selections.json`,
`performance-load-audit.json`, `trace-jit.jsonl`, `jit-review.json`, and `logs/`.
No full mono ABI, DEV/HT, reserved-register, join or model integration was added.

Both success conditions remain required: fixed P256 execution and performance
better than, or non-regressing against, frozen original native. The selected
candidate passes worker checks and the fixed outer acceptance. Claude Opus 5.5
independent review returned `approve` with no blockers. The recovered MI400 run
at delivery
`5446016c426910d96e2b13996586dfe875863da0` passed all 45 GPU tests and actual
A09 JIT/P256 checks. Both FP8 and QAT failed performance acceptance. See the
[measured matrix](mono-persistent-validation.md#mi400-result-2026-11-18).

## User-requested follow-up result

A03 was retested at delivery `5446016c`: 20 tests and P256 Graph tracing passed.
Its six performance rows have four regressions and two inconclusive results.
The [follow-up evidence](mono-persistent-validation.md#follow-up-a03-timing-and-a09-diagnosis)
contains the measured table and A09 controlled comparisons.

A09's main issue is now supported by actual ATT data: fewer independent wave
contexts and serial load-result waits in the current loop. At the sampled
SIMD, native and the same loop body with one TG per task have four overlapping
waves; P256 has one wave processing four tasks. P256 QAT load-result stalls
are 1682 GFX cycles, versus 233 XACK-wait cycles in that sampled wave.
Removing descriptor reuse waits alone does not recover QAT performance.
P512/P1024/P1026 comparisons keep the same instruction sequence except four
stride immediates and reduce the slowdown. A clean 512-row QAT case still
regresses with exactly four tasks per worker, so the extra tail is not required.
These facts revise the earlier unmeasured scheduling hypothesis. Exact cache
traffic and whole-GPU cycle attribution remain open.

The diagnostic wrappers passed 64 strict cases plus four boundary cases.
Contaminated timing cases and zero-valued SQ wait/cycle counters are marked
unusable. No production kernel changed; standalone acceptance still fails.

## 1. Which arithmetic change is selected, and is it exact?

Select the exponent-code reciprocal in the shared Gluon tile helper. For
example, code 128 gives scale 2 and exact reciprocal 0.5. For finite BF16 group
maxima with floor 1e-10, codes are 85..247. Scale is `2^(code-127)` and its
reciprocal is `2^(127-code)`; both are normal FP32 numbers. Construct reciprocal
bits as `max(254-code, 0) << 23`, with `0xff800000` for code 256. Replace only
`x / scale` with `x * reciprocal_scale`. This is an exact power of two, not an
approximate reciprocal. The original scale calculation and E8M0 code store
remain unchanged. Keep maximum(-448), minimum(448), then FP8 conversion. QAT
still converts FP8 to FP32, multiplies by the original scale and converts to BF16.

BF16 to FP32 is exact. Division and multiplication have the same exact result
before rounding for these finite scales. The actual compiler keeps FP32
nearest-even and gradual underflow: LLIR has `denormal-fp-math-f32=ieee`,
metadata has `allow_flush_denorm=false`, and ISA has round mode 0 and denorm
mode 3. Native and candidate MODE writes do not change those FP fields.
MI400 Shader Programming Guide section 4.6.2.4 defines these modes. Thus tiny
results, signed zero and overflow retain the same rounding. The new multiply
has no fast-math flags. No compiler or flush mode was changed.

Special values are part of the contract:

| Code / scale / reciprocal | Division and multiplication before clamp |
| --- | --- |
| 85..247 / positive finite / positive finite | Same rounded finite values and signed zeros; signed Inf stays Inf; NaNs remain NaN |
| 255 / +Inf / +0 | Finite inputs give signed zero; Inf and NaN inputs give NaN |
| 256 / -0 / -Inf | Nonzero finite or Inf inputs give opposite-sign Inf; zero and NaN inputs give NaN |

Code 256 defensively covers a surviving NaN in scale coding. It remains even
though GPU generation of that code is unproven. CPU fmax can propagate a
signalling NaN. Code bytes remain 255 and 0 for codes 255 and 256. Gluon's
actual default is `PropagateNan.NONE`. LLIR uses maxnum/minnum and ISA uses
med3_num with the original operand order. Arithmetic quiets signalling NaNs;
the lower clamp selects -448 for an arithmetic NaN. Pre-clamp payload changes
therefore cannot enter FP8. QAT then sees identical FP8 and original-scale
operands, including Inf and signed zero. GPU byte comparisons remain required.

The worker ran a byte-identical private copy of the independent CPU checker:
65,536 BF16 patterns x 165 codes = 10,813,440 pairs. Pre-clamp bits match with
NaNs compared as both NaN; final FP8 and QAT have zero byte mismatches. This is
CPU algebra/reference evidence, not GPU proof. The existing boundary node now
adds deterministic GPU bit fixtures with all BF16 patterns and mixed groups.
It checks both PS/non-PS paths against frozen native with exact FP8/E8M0/QAT
bytes, XOR-prefilled outputs, unchanged input and input/output guards. All
finite-only oracle assertions remain; the new nonfinite data skips that oracle.

## 2. Does the static reduction justify this candidate?

Yes for local selection. The later hardware run failed the performance gate.
The actual selected source, hashes, production HEAD and patch baseline are in the
[validation guide](mono-persistent-validation.md#source-and-local-status).
Selected code is `0d54cbf3e7b16ac7563c34df6e5d744945f6a495`; its independent
review is run `20261118-155450-63f32955`. The earlier candidate is historical.

Real-binder builds with Triton 3.9.0, gfx1250/wave32, four waves and
`[513,2048]` give these results. Instruction counts are static ISA instructions,
including scheduling instructions; a dual instruction counts once.

| Path | Frozen native bytes / VGPR / instructions | Previous PS bytes / VGPR / instructions | Selected PS bytes / VGPR / instructions | Selected non-PS bytes / VGPR / instructions |
| --- | ---: | ---: | ---: | ---: |
| FP8 | 1388 / 49 / 205 | 1436 / 53 / 221 | 892 / 21 / 138 | 780 / 14 / 112 |
| QAT | 1264 / 44 / 181 | 1420 / 47 / 209 | 876 / 18 / 131 | 772 / 15 / 114 |

The same selected instruction bodies and resources were built for `[1,32]`
and `[513,2080]` tails. All selected paths have 68 SGPR, shared=0 and scratch=0.
PS VGPRBlocks are 1/1 versus native 3/2; these do not measure occupancy.
Eight general FP32 divide sequences per lane become four packed FP32 multiply
instructions. Both paths now use fewer code bytes and VGPRs than native.
No v_div_* or v_rcp_f32 remains. Static rejection gates are FP8 code >= 1388 B
or VGPR > 49, QAT code >= 1264 B or VGPR > 44, any LDS/spill/scratch,
memory-mask EXEC branch or lost vector width. All selected builds pass.

The group32 layout, grid `(256,)`, four wave32 waves, 1024 elements/task,
`task = pid0 + 256*k`, buffer range and 64-bit tile rebasing remain fixed.
Buffer loads stay 128-bit; FP8 stores stay 64-bit plus a byte scale store;
QAT stores stay 128-bit. No speculative address variant or runtime switch was
added. The installed `amd.cdna3` buffer API lowers to gfx1250 opcodes. The
namespace suggestion N1 remains a nonblocking naming issue outside this patch.

A full FP8 task still reads 2048 B and writes 1024 B plus 32 B of scales:
3104 B/task. QAT reads and writes 2048 B each: 4096 B/task. Wave w writes scales
8w..8w+7 through lanes 0,4,...,28. Each wave issues one scale store with OOB
non-owner offsets. This does not establish a fourfold change in issued stores.
The explicit layout removed the old PS's 32 B dynamic LDS and two TG syncs;
frozen FP8 native has one such sync. QAT already has no LDS-removal gain.
At 96 rows there are only 192 tasks, so no worker amortizes multiple iterations.
The arithmetic reduction is concrete. Request combining, 8 B/wave service
cost and cache traffic remain unmeasured. Runtime performance is measured in
the validation guide; the smaller code did not prevent native regressions.

The new ISA was reviewed for global masks, zero filling, single-writer masks,
SCC/VCC, descriptor SGPRs and source VGPRs. FP8 PS preserves the owner predicate
in VCC and uses a separate scalar mask for code 256. Non-PS consumes ownership
into its scale offset before VCC is reused. QAT reuses VCC only after load zero
filling. SCC remains live from the loop comparison to its backedge. Descriptor
rewrites have XACK waits; cross-iteration source reuse follows the protected
PC change. MI400 section 6.9.7.2 requires translation or completion before
source reuse; VM_VSRC alone is insufficient. Tasks do not consume preceding
outputs, so no unconditional per-iteration store-completion wait is added.
No wait or barrier was manually removed. RETRY_DISABLE is unknown;
MODE.REPLAY_MODE does not establish runtime retry state. Private artifacts
retain the line-by-line review of this build, not the previous line numbers.

## 3. Are correctness and review complete?

- [x] Implementation and Codex code/algebra review, including special values,
  default NaN behavior, denorm modes and the final FP8/QAT conversion order.
- [x] CPU exhaustive check: 10,813,440 pairs; final bytes match. No GPU ran.
- [x] Worker local builds: 12 native/candidate AST variants and all 12 frozen
  symbol hashes; 22 real-binder builds include 12 selected paths and tails,
  six native references and four previous-code comparisons. QAT uses `S=None`.
- [x] Collection-only harness: exactly the same 45 IDs. Original assertions,
  finite oracles, five-iteration/tail/ownership/API tests and both A09 Graph
  nodes remain. The Graph nodes retain the three-output tuple and separate
  FP8/scale and QAT input-change assertions for 20 replays.
- [x] Black per-file checks, Ruff, protected-file hashes and diff whitespace.
- [x] Previous findings A1/A2/A3: new dependency review, quantitative static
  gates on both paths, and real `S=None` builds. N2's Graph tuple fix remains.
- [x] A4/A5 and A09-R1/R2/R3: mandatory FP8/QAT timing, explicit row decisions,
  numerical tests without profiling, then two Graph traces with explicit
  profiler branches. The guide states that benchmark defaults exclude A03.
- [x] Outer runner: unchanged fixed acceptance passed, exit 0.
- [x] Claude Opus 5.5 independent arithmetic/code review: approve, no blockers.
  This certifies bounded local readiness only. R1 special-value failure triage
  is in the guide; R2 line wrapping is fixed. Kernel/test bytes remain exact.
- [x] GPU: all 45 tests passed, zero skips, byte/guard mismatches or GPU errors.
  `[513,2048]` has 1026 tasks; two workers run five iterations.
  `[513,2080]` has 1043 tasks; its last task has one valid group.
- [x] GPU: actual wrapper JIT saved. Both A09 Graph trace nodes passed.
  All 48 FP8 and 48 QAT PS dispatches used 256 TGs x 128 threads.
  Actual PS FP8 is 808 B / 21 VGPR; QAT is 844 B / 20 VGPR.
  Both use 68 SGPR, shared=0 and scratch=0. Actual ISA gates passed.
- [ ] GPU performance acceptance failed. FP8 has four regressions and two
  inconclusive rows. QAT has four regressions and two improvements.
  Each path must pass all six rows. The one 15-trial matrix is complete.
  No automatic rerun or tuning followed. No upstream push was made.

A03 `7108dbec079595026f76b7b1b8e082fe1669796d`, its wrapper, A09's wrapper,
benchmark, compiler, frozen native/manifest and submodule declarations remain
unchanged. Public base and frozen-native identities are in the validation guide.
The [historical hardware report](mono-results/2026-11-18-gfx1250/README.md)
belongs to `d5930aff35d67fa0a2e74ba425909747f3ccef9f`: 36 tests passed,
eight timing rows regressed, four were inconclusive. It cannot validate this
candidate or its QAT changes.

## 4. What is the one hardware handoff?

The handoff below was executed once after machine recovery. Its full matrix
and environment are in [the validation guide](mono-persistent-validation.md).
The failed performance gate is final for this run. Do not repeat the matrix
automatically. The outer local acceptance and Claude review are complete.
Use the supplied source and existing ROCm environment. Run all 45 numerical
tests without profiling. Then trace
only `test_graph_replay_changes_inputs[a09]` and `[a09-five-iterations]`.
A failed or unavailable profiler leaves launch acceptance pending, preserves
numerical results and stops timing. Save actual JIT and check P256 first.

Then compare only frozen native and selected PS. The default benchmark includes
A09 FP8/QAT; this handoff also selects them explicitly:

```bash
python -m op_tests.op_benchmarks.triton.bench_mono_persistent \
  --ops a09 a09-qat --rows 96 257 513 --cache both --trials 15 \
  --output "$MONO_RESULTS/performance.json"
```

Retain RotatingBufferBench, the common input pool, alternating provider order
and each round's latency, delta and timestamps. Hot/rotating describes input
reuse. Outputs are allocated by wrappers during graph capture.

```text
delta = PS median - native median
noise = native P90 - native P10
regression = delta > noise and PS P10 > native P90
improvement = -delta > noise and PS P90 < native P10
otherwise = inconclusive
```

All six rows for each path must be improvements, with correctness and P256
passing, before that path passes. A regression fails; equal medians, overlap
or insufficient separation remain inconclusive and do not prove equivalence.
There is no positive slowdown allowance. No automatic rerun or tuning sweep
follows a failure or inconclusive result. Return the evidence and separate
FP8/QAT statuses. A gain against old PS cannot offset a native regression.
