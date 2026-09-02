# Fused Stage2 CU optimization checkpoint (2026-09-01)

## Resume identity and scope

- Codex session: `01a041cb-107c-7612-a6df-08eb5323c56a`.
- Resume command: `cd /home/zihuang/work && codex --yolo resume 01a041cb-107c-7612-a6df-08eb5323c56a`.
- Repository: `/home/zihuang/work/aiter-mega-tile-pr`.
- Current branch: `dev/stage2_compute_pipeline_v1`.
- Current HEAD: `a69a16420 optimizate combine`.
- Baseline parent: `a69a16420 optimizate combine`.
- Remote worktrees: `mi355-gpu-{46,50}:hzm_work:/home/hzm/aiter`.
- Fixed optimization case: `TPR128 / TopK16 / E896 / H7168 / I3072 / EP16 / A4W4`.
- Generalization envelope requested by the user: BS/TPR `1..4096`, hidden
  `1024..10240` initially (original upper request was 16384), experts
  `16..1024`. Up to roughly 4 GiB extra arena storage is acceptable; each GPU
  has 288 GiB.
- End goal is not merely parity with the rank-local baseline: fused Stage2
  must beat the separate MORI GMM2+combine small operators and demonstrate a
  real fusion benefit.

The branch began clean at the baseline; the worker has implemented the
opt-in `tail_claim` path in nine tracked files. Do not reset those concurrent
changes. Also do not delete or overwrite the untracked checkpoint, profile
wrapper, or `trace_data/`. The kernel files must not be edited by the
checkpoint/planner role.

## Git and saved variants

```text
branch: dev/stage2_compute_pipeline_v1
HEAD:   a69a16420
stash@{0}: wip/rank-init-single-owner-20260901
stash@{1}: wip/stage1-preclear-lowticket-nocounter-20260901
stash@{2}: wip/consumer-zero-remote-reset-20260901
stash@{3}: wip/cu-pull-scan-final-cap-20260901
stash@{4}: wip/cu-aware-producer-push-20260831
```

`stash@{0}` contains the rank-init single-owner experiment from the previous
baseline-next branch. `stash@{1}` contains the rejected Stage1-preclear
low-ticket/no-counter delta on top of committed experiment `e4f01f852`.
`stash@{2}` contains the rejected/hanging consumer-zero remote-reset WIP.
`stash@{3}` contains the
validated but slower baseline-atomic + token-ready scan + bounded-final-rejoin
implementation and its harness/tests. `stash@{4}` is the rejected full-payload
producer-push implementation. Older snapshots outside Git are:

```text
/home/zihuang/work/profile_multi_cu_att_20260831.patch
/home/zihuang/work/cta_reduce_wip_20260828.patch
/home/zihuang/work/cta_reduce_wip_20260828.untracked.tar.gz
```

Current untracked artifacts which must be preserved:

```text
scripts/megamoe_tile/STAGE2_CU_OPT_CHECKPOINT_20260901.md
scripts/megamoe_tile/run_stage2_cu_pull_profile_ep16.sh
trace_data/
```

## Stable baseline

The usable baseline is rank-local atomic aggregation, not the staged-ring or
producer-push experiments:

```text
GMM CTA packed-BF16 atomic-adds into its local rank_accumulator
last local contributor decrements/completes rank_pending
completion queue exposes a ready token
node reducer peer-pulls each participating rank_accumulator row
node partial is returned through compact RAIL
final CTA combines local/remote node partials and writes output
```

Best historical rank-local performance is approximately:

| Configuration | Rank-max | All-rank mean |
|---|---:|---:|
| grid224 / reducer16 / final4 / compact16 / vec8 / load-first | 1.507--1.529 ms | 1.393--1.405 ms |
| shared-node grid176 dynamic-base | about 1.645 ms | not sign-off quality |
| MORI GMM2 + combine clean reference | about 1.365--1.383 ms | about 1.272 ms |

The last local checkpoint contains the full baseline history:
`scripts/megamoe_tile/STAGE2_RANK_LOCAL_CHECKPOINT_20260827.md`.

## Experiments completed

### 1. Full producer-push: rejected

The first CU-aware design copied the complete H7168 BF16 row from the winning
GMM CTA into a per-producer proxy slot, then published readiness. The grid was
256 CTAs, with four QP owners, progress/reducer roles, GMM-first workers, and
post-GMM final work.

Mode isolation quantified the regression:

| Mode | Rank-max |
|---|---:|
| init-only | 0.866 ms |
| GMM2-only | 1.060 ms |
| GMM2 + local atomic + full-row producer push | 3.401 ms |
| GMM2 + push + progress reducer | about 3.360 ms |
| return/final-only | about 1.159 ms |
| full producer-push | 3.233 ms |

The approximately 2.34 ms jump from GMM2-only to route-store-only showed that
the dominant regression was the last-contributor full-row copy/publish in the
GMM epilogue, not the reducer or RAIL. Vec16 copy and four-wave variants did
not repair it. The implementation is saved in `stash@{4}` and must not be
restored over the current worktree without first saving the current branch.

Primary logs:

```text
/home/hzm/logs/megamoe_stage2_breakdown_20260824/cuaware_initonly_grid256_v1_node{0,1}.log
/home/hzm/logs/megamoe_stage2_breakdown_20260824/cuaware_gmm2only_grid256_v1_node{0,1}.log
/home/hzm/logs/megamoe_stage2_breakdown_20260824/cuaware_pushonly_ready_v5_node{0,1}.log
/home/hzm/logs/megamoe_stage2_breakdown_20260824/cuaware_reduceonly_v6_node{0,1}.log
/home/hzm/logs/megamoe_stage2_breakdown_20260824/cuaware_returnonly_v7_node{0,1}.log
/home/hzm/logs/megamoe_stage2_breakdown_20260824/cuaware_push_qp4_perf_v2_node{0,1}.log
```

### 2. Completion/peer-ready scanning: correct but slower

The next design kept payloads in peer rank accumulators and assigned more
reducer CTAs to scan readiness. Several representations were tested: direct
peer-ready scan, ready-mask aggregation, faster scan, and bounded peer scan.

| Variant | Reducer CTAs | Rank-max mean | Conclusion |
|---|---:|---:|---|
| baseline rank-local | 16 | about 1.661 ms in same noisy period | reference |
| direct scan | 12 | 1.773 ms | slower |
| direct scan | 24 | 1.816 ms | slower |
| ready-mask scan | 12 | about 1.863 ms | slower |
| ready-mask scan | 8 | about 1.863 ms | no recovery |
| bounded peer scan | 12 | 1.974 ms | slower and noisy |

The baseline completion queue already exposes completed tokens in useful
order, so scanning added flag loads, claim atomics, cache-line traffic, and
reducer/GMM resource interference without removing a measured head-of-line
bottleneck.

Primary logs:

```text
/home/hzm/logs/megamoe_stage2_breakdown_20260824/cupullscan_r12_perf_packed_v2_node{0,1}.log
/home/hzm/logs/megamoe_stage2_breakdown_20260824/cupullscan_r24_perf_packed_v1_node{0,1}.log
/home/hzm/logs/megamoe_stage2_breakdown_20260824/cupullmask_r12_perf_v2_node{0,1}.log
/home/hzm/logs/megamoe_stage2_breakdown_20260824/cupullmask_r8_perf_v1_node{0,1}.log
/home/hzm/logs/megamoe_stage2_breakdown_20260824/cupullmask_fastscan_r12_perf_v3_node{0,1}.log
/home/hzm/logs/megamoe_stage2_breakdown_20260824/cupull_peer_bounded_r12_perf_v4_node{0,1}.log
```

### 3. Independent ready-slot design: abandoned

The attempted protocol was:

```text
GMM still atomic-aggregates only into its local rank_accumulator
last contributor stores a tiny generation flag into a distinct slot on proxy
reducer waits until all producer slots required by one token are ready
only then peer-pulls all participating payload rows and reduces once
```

This avoids both full-row push and multi-producer atomic-add on one ready word.
The attempted ABI appended `rank_pull_ready_slots` with shape:

```text
parity x producer_local_rank x source_node/plane x token x 8*i64
```

Each logical generation word is padded to a 64-byte cache line. The current
kernel uses system-scope atomic exchange to publish and agent-scope exchange
to claim. The intended final version should use a release store for a truly
single-writer slot, but only after the address/producer ownership is proven.

Its role geometry was:

```text
CTA 0..3                          one compact-return QP owner each
CTA 4..(4 + reduce_blocks - 1)   ready-slot scan + peer-pull reducer
remaining CTAs to grid 256       GMM-first
only cu_final_rejoin_blocks      allowed to drain final work after GMM
```

There is no hardware guarantee that block N maps to CU N, but at one resident
CTA per CU a fully resident 256-CTA grid approximately occupies all 256 CUs.

The producer-slot diagnostics were not correct. Missing-ready patterns
remain producer-dependent even after switching publication to system-scope
exchange:

```text
cupull_slots_producer_diag_v2: ordinary-store experiment missed producer-only
                              columns, 128..896 slots depending on rank
cupull_slots_xchg_diag_v1:     system-scope xchg still missed producer-only
                              columns, 128..896 slots depending on rank
```

Examples showed entire producer columns missing for every even token. This was
more consistent with wrong peer/base/producer-slot addressing or incomplete
producer ownership than with weak visibility. This route has now been
abandoned and the append-only independent-slot ABI region has been removed
from the current worktree.

Diagnostic logs:

```text
/home/hzm/logs/megamoe_stage2_breakdown_20260824/cupull_slots_producer_diag_v1_node{0,1}.log
/home/hzm/logs/megamoe_stage2_breakdown_20260824/cupull_slots_producer_diag_v2_node{0,1}.log
/home/hzm/logs/megamoe_stage2_breakdown_20260824/cupull_slots_xchg_diag_v1_node{0,1}.log
```

## Correctness status

- The historical atomic baseline passed the full Stage2 contract suite (135
  tests at the August checkpoint; later local suite count was approximately
  140 after new CU contracts were added).
- Earlier CU scan/ready-mask variants passed paired and arbitrary EP16
  validation. Representative arbitrary result: protocol errors 0 and maximum
  relative L2 `0.0036925`.
- Validation logs:

```text
/home/hzm/logs/megamoe_route_store_validation_20260826/cuaware_push_qp4_paired_v1_rank_paired-rank-half-remote_node{0,1}.log
/home/hzm/logs/megamoe_route_store_validation_20260826/cuaware_push_qp4_arbitrary_v1_rank_permuted-arbitrary-topk_node{0,1}.log
```

- Passing Python contract tests proves ABI/factory/control-flow contracts; it
  does not prove the 16-rank runtime protocol. The latest independent-slot
  implementation failed the runtime missing-ready diagnostic and was dropped.

## Validated scan implementation now saved in stash@{3}

The implementation now saved in `stash@{3}` returned to baseline local payload
aggregation and removed producer-specific ready slots. It retained CU-aware
scheduling with simplified readiness:

```text
each GMM contribution
  -> packed BF16 atomic_add into local rank_accumulator
  -> atomic_add/decrement the existing partial_done aggregation counter

last producer for one token
  -> direct release-store generation to rank_reduce_queue_ready[token]
  -> no FIFO tail reservation and no token record enqueue

reducer CTA
  -> scans token-indexed rank_reduce_queue_ready entries
  -> xchg-claims rank_reduce_queue[token]
  -> peer-pulls all required ranks and reduces once after token is complete
  -> increments rank_reduce_queue_tail only as a completed-work count

bounded GMM workers
  -> after GMM, at most cu_final_rejoin_blocks drain final work
```

This deliberately separates three meanings which were previously conflated:

- `rank_reduce_queue_ready[token]`: generation-tagged token completion;
- `rank_reduce_queue[token]`: one-winner reducer claim state in CU mode;
- `rank_reduce_queue_tail`: number of reductions completed, not FIFO publish
  position and not a producer reservation counter.

The design keeps the baseline payload path and its aggregation semantics. Its
only producer-side addition is one tiny token-ready store by the last
producer; it neither copies an H-wide row nor maintains per-producer slots.
The direct token-indexed ready array also removes FIFO head-of-line ordering.
This revision has now passed the first full validation gates:

- 141 local Stage2 contract tests passed;
- cold single-process full compile completed in 128 seconds, below the
  mandatory 180-second gate;
- paired-rank-half-remote EP16 validation passed;
- permuted-arbitrary-topk EP16 validation passed.

These results validate the atomic aggregation + token-ready protocol for the
tested TPR128 case. They do not yet establish a performance win or the full
BS/hidden/expert generalization envelope.

## Compile-time gate

Historical full single-process compile times:

| Version | Time |
|---|---:|
| atomic rank-local baseline | about 136 s |
| staged-ring/large-CFG experiment | about 369--396 s |
| staged-ring 8-process compile | about 459 s |

Any modified full kernel taking more than 180 seconds in a clean
single-process compile is considered a compiler/code-generation regression.
Stop before 16-rank validation and bisect with `init_only`, `gmm2_only`,
`route_store_only`, `gmm2_atomic_only`, `return_only`, and `full`. Check for
duplicated/inlined final bodies, large cross-role CFGs, dynamic polling loops,
and cross-role register liveness. Runtime cache reuse is not a substitute for
passing the cold compile gate.

The atomic-aggregation + token-ready revision now in `stash@{3}` cold-compiled
in 128 s. This is close to the historical 136 s atomic baseline and
comfortably below the 180 s regression gate. The new consumer-reset branch has
not yet been compiled; continue to measure cold compile after any material
control-flow change.

## Atomic aggregation + token-ready CTA screen, stash@{3} (2026-09-01)

The first CTA screen is complete. Times below are rank-max mean / all-rank
mean for the fixed TPR128 case:

| Reducer CTAs | Final rejoin CTAs | Run type | Rank-max | All-rank |
|---:|---:|---|---:|---:|
| 12 | 16 | reference screen | 1.752 ms | 1.687 ms |
| 12 | 4 | screen | 1.826 ms | 1.712 ms |
| 12 | 8 | screen | 1.821 ms | 1.744 ms |
| 12 | 32 | screen | 1.796 ms | 1.719 ms |
| 8 | 16 | screen | 1.751 ms | 1.663 ms |
| 8 | 16 | long | 1.804 ms | 1.733 ms |
| 16 | 16 | screen | **1.730 ms** | 1.669 ms |
| 16 | 16 | long | 1.804 ms | **1.723 ms** |
| 24 | 16 | screen | 1.731 ms | **1.657 ms** |
| 24 | 16 | long | 1.824 ms | 1.742 ms |

Reference targets:

| Path | Rank-max | All-rank |
|---|---:|---:|
| same-period rank-local baseline grid176 | 1.661 ms | 1.528 ms |
| historical MORI GMM2 + combine | about 1.365 ms | not directly remeasured |

Conclusions:

- The token-ready path remains about 4--10% slower than the same-period
  grid176 baseline depending on the run and statistic.
- Final rejoin 16 is clearly better than 4/8/32 at reducer12; final queue
  under-service hurts F4/F8 and extra rejoin contention/resource use hurts F32.
- Reducer 8/16/24 screens are very close, and longer runs converge around
  1.80--1.82 ms rank-max. There is no CTA-count-only performance win.
- The gap to the 1.365 ms small-op target is still material. Do not promote
  this CU scheduler as the default based on correctness alone.

An ATT profile wrapper has been added at
`scripts/megamoe_tile/run_stage2_cu_pull_profile_ep16.sh` to support the next
attribution step. It is currently untracked. The exact environment, target
CUs, decoder output, and kernel hash must be appended here when the first
captures complete.

## First multi-CU ATT results

ATT capture has started for the exact grid256 fused candidate kernel. Recorded
resource metadata:

```text
megamoe_tile_ep16_stage2_direct_node_a4w4_r0_h7168_i3072_e896_bm32_bn256_
bk256_q4_direct_v6_gridrelease_accbf16_fc0_gspersistent_queue_rt16_nrtoken_
babuffer_rrcompact_epilane32_meta_ng2_gpa_double_buffer_narank_local_
ramatomic_cupull12f16_nr16v8tokenload_first_nrwdynamic_head_nrr0_
rlaexpanded_sbwave0_aiinterleaved
```

```text
grid:          256 CTAs
LDS:           29,184 B
scratch:       1,024 B
VGPR:          92 in profiler CSV; 183 in emitted ISA/resource report
SGPR:          112
```

The VGPR discrepancy must be preserved rather than silently normalized: the
profiler CSV and compiler/ISA report use different accounting conventions
(likely allocation granularity or per-wave representation). Quote the source
with every resource number.

Initial capture directories cover target CUs `0`, `4`, `8`, `12`, and `16`
across multiple shader engines:

```text
/home/hzm/profiles/cupull_atomic_att_v1_cu0
/home/hzm/profiles/cupull_atomic_att_v1_cu4
/home/hzm/profiles/cupull_atomic_att_v1_cu8
/home/hzm/profiles/cupull_atomic_att_v1_cu12
/home/hzm/profiles/cupull_atomic_att_v1_cu16
/home/hzm/profiles/cupull_atomic_att_se1_v1_cu8
/home/hzm/profiles/cupull_atomic_att_se2_v1_cu8
/home/hzm/profiles/cupull_reduce_att_se1_v1_cu0
/home/hzm/profiles/cupull_reduce_att_se2_v1_cu8
/home/hzm/profiles/cupull_reduce_atomiconly_att_cu0
/home/hzm/profiles/cupull_atomic_r12_f16_stats_v1_cu1
```

Decoded directories use names such as
`ui_output_agent_<pid>_dispatch_92`; raw profiler CSVs and decoded JSON/CSV
must both be retained.

Observed pipeline statistics so far:

| Capture | Dominant stalls/instructions |
|---|---|
| CU0/CU4/CU8, several SEs | dominated by MORI CCO wait/progress path; not representative of GMM throughput |
| full fused, SE1 CU8 GMM/final trace | barrier 38.1%, VMEM wait 36.9%, VMEM load 18.8% |
| `gmm2_atomic`, SE1 CU8 | barrier 30.9%, VMEM wait 28.8%, VMEM load 24.6%, VMEM store 11.9% |
| `atomic_only`, SE0 CU0 reducer trace | barrier 47.7%, VMEM wait 33.7%, VMEM store 16.8% |

Interpretation is preliminary:

- CCO-wait-dominated traces identify QP/progress ownership but cannot be used
  to infer GMM critical-path latency.
- The representative GMM/final trace shows barrier plus VMEM-wait pressure,
  not an obvious scalar/ALU bottleneck.
- Comparing full versus `gmm2_atomic`, full raises barrier and VMEM-wait share
  while lowering load/store instruction share. This is consistent with extra
  cross-role synchronization/idle intervals, but percentages alone do not
  prove causality or elapsed-time contribution.
- The target0 `atomic_only` reducer capture is complete. Within it, the main
  attributable regions are initialization grid barrier 22.0%, rank-accumulator
  clear stores 16.8%, scan barriers 15.3% + 5.1%, and peer-pull wait about 17%.
  This attribution caused the consumer-reset and Stage1-preclear pivots; both
  subsequent implementation routes hung and are rejected as described below.

ATT is strictly for pipeline attribution. Do not compare ATT-instrumented
kernel duration against benchmark latency. Role assignment is by logical CTA;
the hardware scheduler does not guarantee block-to-CU-number identity, so
each decoded trace must first be classified from its executed PC/role path.

Local downloaded trace roots:

```text
trace_data/cupull_atomic_att_v1_cu0_scp/
trace_data/cupull_atomic_att_v1_cu4_scp/
trace_data/cupull_atomic_att_v1_cu8_scp/
trace_data/cupull_atomic_att_v1_cu12_scp/
trace_data/cupull_atomic_att_se1_v1_cu8_scp/
trace_data/cupull_atomic_att_se2_v1_cu8_scp/
trace_data/cupull_reduce_att_se1_v1_cu0_scp/
trace_data/cupull_reduce_att_se2_v1_cu8_scp/
trace_data/cupull_reduce_atomiconly_att_cu0_scp/
```

## Rejected pivot: consumer-zero accumulator, stash@{2}

Reducer `atomic-only` ATT exposed a larger, more direct target than further
ready-scan CTA tuning:

| Region in reducer/atomic-only trace | Approximate share |
|---|---:|
| initialization grid barrier | 22.0% |
| rank-accumulator clear stores | 16.8% |
| scan barrier component A | 15.3% |
| scan barrier component B | 5.1% |
| peer-pull wait | about 17% |

The initialization barrier plus rank-accumulator clear alone account for a
large visible fraction of the traced pipeline. This motivated stopping work on
the slower CU-pull scan branch and creating `dev/stage2_consumer_reset_v1`
directly from baseline commit `a69a16420`.

The attempted target was **consumer-zero accumulator**: remove the launch-time
full rank-accumulator clear and make the consumer establish a physical zero
before peer-pull. It was implemented as a WIP and tested at both grid176 and
grid160. Both configurations hung. The failure is attributed to the reducer
performing remote LSA zero stores as part of the reset protocol; reducing the
grid did not resolve it, so this is not merely a 256-CTA residency problem.
The WIP is saved as `stash@{2}` and is no longer the active direction.

The invariants which made this route risky remain useful context:

- no stale value from a previous parity/generation can participate;
- the first writer cannot race later packed-BF16 atomic additions;
- reset ownership must be unique at the required token/tile granularity;
- publication cannot occur until zeroing and all valid contributions finish;
- empty-route tokens and parity reuse remain correct;
- added reset metadata/work must cost materially less than clearing the whole
  accumulator plus the initialization barrier.

## Rejected pivot: Stage1 local preclear, stash@{1}

The branch `dev/stage2_stage1_preclear_v1` tested moving Stage2
rank-accumulator initialization into Stage1, using local stores rather than
reducer-issued remote LSA stores:

```text
Stage1 local CTAs
  -> cooperatively clear this rank's next-generation Stage2 rank_accumulator
  -> overlap clear traffic with useful Stage1 work
  -> atomically/countably publish clear_done

Stage1 finisher
  -> waits until all required local preclear work is clear_done
  -> release-publishes stage1_done

Stage2
  -> acquires/waits stage1_done as it already does
  -> skips its rank-accumulator clear and associated init barrier
  -> runs the original baseline atomic aggregation/reducer protocol
```

The key happens-before chain must be explicit:

```text
all local accumulator zero stores
  -> release clear_done
  -> Stage1 finisher acquire/wait
  -> release stage1_done
  -> Stage2 acquire/wait
  -> first Stage2 atomic accumulation
```

Two variants failed:

1. Initial high-ticket preclear plus `clear_done` counter and finisher wait
   hung in paired execution.
2. A lower-ticket partition with the counter and finisher wait removed also
   hung in paired execution.

Because removing the synchronization protocol did not remove the hang, the
working conclusion is that adding preclear work to the 256-CTA persistent
Stage1 kernel breaks residency/forward progress or starves its existing role
schedule. This is not currently considered a viable way to hide the clear.

The committed first attempt is `e4f01f852`; the low-ticket/no-counter delta is
saved in `stash@{1}`. Do not spend additional distributed runs on this route
unless a static occupancy/residency proof or a reduced-grid reproducer changes
the diagnosis.

## Archived experiment: baseline init/metadata single-owner, stash@{0}

The `dev/stage2_baseline_next_v1` experiment stayed inside the rank-local
Stage2 baseline. Based on the ATT attribution (init barrier 22.0%, accumulator
clear stores 16.8%), it investigated:

- merge or remove redundant initialization barriers where the existing
  producer/consumer ordering already provides the necessary happens-before;
- assign Stage2 metadata initialization to one explicit owner CTA/wave rather
  than redundantly executing it across roles;
- retain local accumulator clearing in Stage2 unless a safe local overlap is
  proven;
- retain the baseline rank-local atomic aggregation, completion queue,
  peer-pull reducer, RAIL, and final-combine dataflow.

Its static-strided baseline grid176 full run hung and was terminated. The WIP
is archived as `stash@{0}`; it is not the active direction. As with the other
resident-grid hangs, do not interpret the absence of a result as a performance
measurement.

## Current direction: persistent GMM queue `tail_claim`

The current branch is `dev/stage2_compute_pipeline_v1` at `a69a16420`, with an
uncommitted opt-in implementation in progress. The experiment targets the persistent GMM work queue rather than
initialization or communication. ATT attributes approximately 7.7% of the
representative GMM pipeline to the per-work queue barrier.

Baseline persistent queue iteration effectively uses two CTA barriers around
work acquisition/broadcast. The proposed `gmm_queue_sync=tail_claim` schedule
restructures the tail of the current work item and the claim/broadcast of the
next item so each work item pays one barrier instead of two.

Required invariants:

- exactly one wave/lane performs each global queue atomic claim;
- the claimed work ID is visible uniformly to all waves before use;
- no wave starts the next work while another wave still consumes shared LDS
  from the previous work;
- the terminal/no-work result is broadcast without deadlock;
- static-strided behavior and the default persistent schedule remain unchanged;
- no new cross-iteration register liveness materially increases VGPR/scratch.

## Planner state audit and immediate next steps (2026-09-01)

The repository was audited after resuming the session. The handoff and current
worktree agree on the active experiment:

- branch/HEAD are `dev/stage2_compute_pipeline_v1` / `a69a16420`;
- nine tracked files contain the uncommitted, opt-in `tail_claim` plumbing and
  implementation; the default remains `two_barrier`;
- `git diff --check` passes;
- the validation wrapper accepts `gmm_queue_sync` as positional argument 13;
- the breakdown wrapper accepts it as positional argument 33 (not 13);
- all five named experimental stashes and the untracked checkpoint/profile/
  trace artifacts remain present.

The resumed handoff reports, but this planner audit did not independently
rerun, the following completed gates for the exact current diff:

```text
local contract suite:             136 passed
cold single-process full compile: 135 s
```

Fresh local static validation in the resumed session passed `py_compile` for
the changed Python entry points and `git diff --check`. The host shell does not
provide `pytest` (`pytest: command not found`), so no fresh contract suite was
run there; this is an environment/tooling limitation, not a test failure. Run
the suite in the established container/environment before final promotion.

Thus the reported cold compile satisfies the mandatory <=180 s gate. The
implementation remains fused: the change only restructures work acquisition
inside the persistent GMM path; it does not split GMM2 and combine into
separate kernels. It introduces no shape-specific indexing or constants, so
the code structure remains compatible with the requested BS/hidden/expert
generalization envelope, though runtime coverage of that envelope is still
outstanding.

Next actions, in order:

1. Run paired-rank-half-remote EP16 correctness on nodes 46/50 with positional
   argument 13 set to `tail_claim`.
2. Run permuted-arbitrary-topk EP16 correctness the same way. Add poison/stress
   coverage before promotion if the existing harness supports it.
3. Run isolated `gmm2_only` same-run A/B for `two_barrier` versus `tail_claim`.
4. Run full fused same-run baseline/candidate/baseline performance with packed
   weights, `warmup=5`, `iters=30`, `tail=20`, grid176/final4/rank-local/
   reducer16/vec8/load-first. In the breakdown wrapper, remember that queue
   sync is positional argument 33.
5. Reject the experiment if it has no stable >=1% win. If it regresses or is
   ambiguous, capture representative GMM and reducer/final CTA roles with ATT
   across multiple target CUs and classify each trace by executed role/PC.

The theoretical target is only the ATT-observed ~7.7% work-queue barrier
region; realizable gain may be smaller because barrier samples include wave
convergence. A successful final project result still must close the fused
`1.661 -> 1.365 ms` gap rather than merely pass correctness.

### Remote recovery and paired validation launch

Remote execution was restored and preflight confirmed both node 46 and node
50 at 0% GPU utilization with no `torch`, benchmark, validation, or `sglang`
processes. The same nine tracked `tail_claim` files were synchronized to both
remote worktrees.

Contract results after synchronization:

| Node | Result | Interpretation |
|---|---:|---|
| 50 | 136 passed | clean contract pass |
| 46 | 134 passed, 2 failed | pre-existing remote-worktree drift, not a `tail_claim` assertion failure |

The two node-46 failures concern the Stage2 ABI-region count and the public
`return_chunk_tokens` default source string. They must be reconciled before a
final clean-tree sign-off, but they do not currently invalidate the queue-sync
experiment because neither failure exercises the new `tail_claim` contract.

Paired 16-rank `tail_claim` validation was launched concurrently:

```text
tag:        cuaware_push_qp4_paired_tailclaim_v1
node 46:    exec session 71299
node 50:    exec session 31712
route:      paired-rank-half-remote
queue sync: tail_claim
```

The first launch exited with status 1 on both nodes during validation
preflight. The new tag's reference directory did not contain the required 16
`direct_*.pt` files. Execution never entered the `tail_claim` kernel, so this
result is neither a correctness failure nor a compile failure of the candidate.

Recovery sequence for the same tag:

1. Run `direct` mode on both nodes to generate all 16 reference tensors under
   the tag's reference directory.
2. Confirm the complete reference set exists.
3. Rerun `rank` mode with `gmm_queue_sync=tail_claim`.

Next checkpoint event: record direct-reference exit status/count, followed by
both rank-mode exit statuses, protocol-error counts, maximum relative L2,
generation coverage, and log paths. Do not begin performance attribution
unless both rank-mode processes complete successfully.

### Tail-claim runtime failure and LDS lifetime root cause

The recovery sequence generated references successfully: `direct` mode exited
0 on both nodes. The subsequent 16-rank `rank` run with `tail_claim` exited 1
on both nodes. Every rank reported a HIP illegal-memory-access failure during
the first forward/synchronize, so there is no numerical result to evaluate.

Root cause analysis found a real `tail_claim` correctness bug. The experiment
reuses `work_ptr = lds_base`. Inside `run_gmm_work`, the last CTA barrier occurs
before the epilogue fence/publication sequence. After tx0 returns from the
helper it may immediately claim the next item and overwrite `work_ptr`, while
other waves still use aliased LDS for epilogue metadata or publication. The
baseline two-barrier schedule's next-loop leading barrier protected this LDS
lifetime; removing it made the alias unsafe. This invalidates the earlier
handoff assumption that `run_gmm_work` fully reconverged before the tail claim.

Required repair:

- allocate a dedicated, non-aliasing LDS slot for the tail-claim work ID;
- permit tx0's global claim and store to overlap the remaining publication;
- retain the tail barrier as both epilogue convergence and next-work
  publication/broadcast;
- do not "fix" the issue by merely restoring an extra barrier, because that
  would erase the intended synchronization reduction;
- re-run static contracts and the <=180 s cold-compile gate before distributed
  correctness, and check the extra LDS allocation/resource report.

Current verdict: `tail_claim` v1 is correctness-failing and must not be
benchmarked. The fused-kernel and shape-generality constraints remain in force.

### Tail-claim v2 dedicated-LDS repair and compile gate

The LDS lifetime bug was repaired by adding a dedicated
`gmm_queue_work_ptr` for `tail_claim`. Only that opt-in schedule receives the
additional 16 bytes of LDS; the legacy queue schedule and final-work path keep
using the original `work_ptr`. This preserves the intended fused kernel and
does not add shape-specific indexing.

Fresh gates:

```text
py_compile:                         pass
git diff --check:                   pass
node 50 single-process cache=0:     exit 0
cold compile wall time:             136.015 s
reported LDS:                       29,008 bytes
mandatory compile threshold:        <=180 s, pass
```

The compile log includes `hipErrorInvalidValue` from the script's intentional
zero/invalid dummy launch, but the script emitted the compiled kernel and its
resource report and exited 0. This message is not a candidate runtime
correctness failure and must not be conflated with the earlier 16-rank illegal
memory access.

The repaired nine-file worktree has been synchronized to nodes 46 and 50.
Next action is to reuse the already successful paired direct references and
rerun paired `rank` validation with `tail_claim`. Performance testing remains
blocked until both nodes pass runtime correctness.

The first paired rerun of v2 again showed all-rank HIP illegal memory access,
but the run is **invalid due to stale-cache contamination** and is not evidence
that the dedicated-LDS repair failed. Both v1 and v2 used the same kernel-name
component `_gqstail_claim`, while the rank wrapper enables
`FLYDSL_RUNTIME_ENABLE_CACHE=1`. The preceding cache-disabled cold compile
validated/generated the new source but did not replace the artifact subsequently
selected by the cache-enabled distributed run, which could therefore load v1.

Cache-isolation recovery:

1. Add a distinct `_qslot` or version-2 suffix to the repaired tail-claim
   kernel name/cache key, without changing the legacy name.
2. Repeat the cold compile/resource gate for that exact named artifact.
3. Synchronize both nodes and rerun paired rank validation using the new key.

Do not count this contaminated rerun as either a v2 pass or failure. A runtime
verdict requires a uniquely keyed artifact generated from the repaired source.

### Tail-claim v2 definitive failure; pivot to safe chunk claim

The repaired implementation was rebuilt and rerun with a unique `_qslotv2`
kernel/cache key, eliminating the stale-artifact ambiguity. Paired rank mode
still exited 1 on both nodes with HIP illegal memory access on every rank.
Therefore dedicated non-alias LDS alone is insufficient: v2 is a genuine
runtime correctness failure.

The direction which overlaps the next global queue claim with the current
work item's epilogue publication is now closed. Do not attempt to benchmark or
revive it by further cache or CTA-count changes.

New experiment: **safe chunk claim**.

```text
tx0 atomically claims CHUNK consecutive tickets for the CTA's shard
  -> CTA barrier broadcasts the chunk base
  -> process each valid work item in that chunk
  -> after every work item, CTA barrier converges epilogue publication and LDS reuse
  -> only after the complete chunk, tx0 claims the next chunk
  -> CTA barrier broadcasts the next chunk base
```

This design does not overlap a queue claim with an unfinished epilogue. Its
potential benefit is limited to amortizing the global queue atomic and chunk
base broadcast across multiple work items; it intentionally retains the
per-work safety barrier. Required checks include terminal partial chunks,
shard-stride ticket mapping, no duplicate/missing work, uniform barrier
participation, unchanged legacy default, and a shape-derived compile-time
chunk parameter. The new variant requires a unique cache key, static tests,
resource comparison, <=180 s cold compile, then paired/arbitrary runtime
correctness before any performance A/B.

### Safe chunk-claim v3 compile gate

The first implementation uses `CHUNK=4` and a unique kernel suffix
`_chunk4v3`. Review found no correctness blocker, and static checks passed.
Node 50 then compiled the exact uniquely named artifact with runtime cache
disabled:

```text
single-process cold compile: exit 0
wall time:                   133.862 s
reported LDS:               28,992 bytes
kernel suffix:              _chunk4v3
```

This passes the mandatory <=180 s gate and introduces no LDS growth relative
to the pre-tail-claim resource footprint. The next gate is paired 16-rank
runtime correctness using the existing direct references. Do not infer a
performance win from compile/resource results; arbitrary routing and same-run
A/B remain required after paired correctness.

### GMM queue optimization route closed

The uniquely named `_chunk4v3` paired rank run exited 1 on both nodes. Every
rank failed with HIP illegal memory access during the first generation, so
safe chunk claim is also a runtime correctness failure despite passing review,
static checks, resource checks, and cold compilation.

Three queue variants have now failed distributed runtime correctness:

1. tail claim overlapping epilogue publication;
2. tail claim with a dedicated non-alias LDS queue slot (`_qslotv2`);
3. per-work convergence with four-ticket chunked queue claims (`_chunk4v3`).

The persistent GMM queue optimization route is therefore closed. Its ATT
attribution was only about 7.7% of the representative GMM pipeline; even an
unrealistic complete removal would not by itself close the approximately
`1.661 -> 1.365 ms` fused-to-small-operator gap. Further correctness risk on
this component is not justified.

Recovery and next direction:

1. Save all tracked queue-optimization WIP in a clearly named stash, preserving
   this checkpoint and other untracked artifacts.
2. Restore the usable `a69a16420` rank-local atomic baseline and verify the
   worktree/source identity before testing.
3. Use existing or fresh multi-role ATT to select a larger measured bottleneck.
   Captures must include representative QP/RAIL, GMM/epilogue, rank reducer,
   and final/unpermute CTA paths across multiple target CUs; classify by
   executed PC/role rather than assumed CU number.
4. Continue to treat ATT as pipeline attribution only, not latency measurement,
   and preserve the fused single-kernel requirement.

### Baseline rerun infrastructure correction

The first restored-baseline benchmark attempt was invalid. The breakdown
wrapper unconditionally exported `FLYDSL_RUNTIME_ENABLE_CACHE=0`, causing all
16 distributed ranks to compile independently. After more than 377 seconds
the processes were still consuming 100% CPU; this explains one source of the
historically anomalous multi-rank compile times and is not a kernel runtime
performance regression. Only processes matching the exact benchmark tag were
terminated on nodes 46 and 50.

The wrapper was corrected to:

```bash
export FLYDSL_RUNTIME_ENABLE_CACHE="${FLYDSL_RUNTIME_ENABLE_CACHE:-1}"
```

Distributed benchmarks now reuse compiled artifacts by default. A caller can
still request a deliberate cold compile by explicitly setting the variable to
0. This is test-infrastructure behavior only and does not change generated
kernel logic or runtime performance. The aborted >377-second run must not be
reported as a baseline latency or normal single-process compile time.

### Baseline ATT re-analysis and rank-specialized cache issue

Existing ATT captures were rechecked using the FlyDSL capture and analysis
skill procedures. Valid role-classified results are:

| Trace/role | Barrier | VMEM wait | VMEM load/store | Principal hotspots |
|---|---:|---:|---:|---|
| full/GMM, SE2 CU8 | 38.1% | 36.9% | load 18.8% | `gemm2.py:401`, `:636`, `:91`; historical `stage2.py:3988`, `:956`, `:5384` |
| atomic-only reducer | 47.7% | 33.7% | store 16.8% | init grid barrier 22.0%; rank-accumulator clear 16.79%; reducer barriers 15.38% + 5.22%; peer-pull wait 17.27% |

CU0 and CU4 captures were dominated by the QP/communication role's
`cco/ops.py:31` wait (98.9% and 91%, respectively), so they are useful for
role identification but not for GMM throughput attribution. Some CU0, CU12,
and SE1 captures are damaged or decode with `code=null`; exclude them from
quantitative conclusions. A clean baseline capture dedicated to the
final/unpermute role is still missing.

The baseline source has been synchronized to both remote nodes. Two baseline
performance attempts remain invalid:

1. The wrapper-forced cache-disabled run made all 16 ranks compile and was
   terminated after >377 seconds.
2. After fixing the wrapper default to cache-enabled, the rank-specialized
   artifacts were not yet present. All 16 ranks independently missed cache and
   again compiled concurrently for roughly >300 seconds without producing a
   benchmark result. Exact-tag processes were terminated.

Neither attempt is a performance datapoint. Cache enablement only permits
reuse; it cannot create the distinct rank-specialized artifacts on first use.

Current repository state after closing the queue experiment:

```text
tracked implementation WIP:  saved in stash@{0}
active tracked change:       breakdown wrapper cache-default correction only
remote source:               baseline synchronized
```

Required next sequence:

1. Precompile each required rank-specialized baseline artifact in a controlled,
   serialized or low-concurrency process on each node.
2. Verify artifact presence and demonstrate cache-hit startup before launching
   the 16-rank benchmark.
3. Run the baseline performance measurement only after cache hits are proven;
   retain logs that separate compile/setup time from timed kernel iterations.
4. Capture a clean baseline final/unpermute CTA trace. Sample multiple target
   CUs as needed and classify the executed role from PCs/source lines rather
   than CU number.
5. Use the combined QP, GMM/epilogue, reducer, and final-role evidence to choose
   a bottleneck larger than the closed ~7.7% queue region.

### Fresh rank-local baseline ATT launch

At the user's request, the FlyDSL profiling wrapper was updated from the
stashed CU-pull candidate to the current usable rank-local atomic baseline:

```text
grid/workers:             176
final combine CTAs:       4
node accumulation:        rank_local
rank accumulation:        atomic
node reducer CTAs:        16
reducer vector bytes:     8
reducer load schedule:    load_first
reducer work schedule:    dynamic_head
RAIL return schedule:     compact
return chunk tokens:      8
```

Obsolete CU-pull parameters were removed from the wrapper. Kernel selection now
requires regex components `narank_local` and `ramatomic`, preventing capture of
the archived CU-pull implementation.

Nodes 46 and 50 passed the idle preflight. A fresh ATT run was launched with:

```text
tag:                ranklocal_baseline_full_v1
master port:        29716
target CU:          8
shader-engine mask: 0xf
profile rank:       global rank 0 only
node 46 session:    75211
node 50 session:    92524
```

The objective is to recover representative GMM/epilogue, reducer, RAIL/QP, and
final/unpermute execution timing from the baseline. A target CU does not imply
a logical CTA role; classify each decoded trace using executed PCs/source
locations. If CU8 does not sample every role, perform additional targeted-CU
captures rather than inferring missing roles.

The `ranklocal_baseline_full_v1` capture is invalid. After more than ten
minutes it had not reached the target dispatch. The profiling launcher had
propagated `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1` to all 16 ranks, forcing every
rank-specialized large kernel through DWARF-enabled recompilation. Exact-tag
processes were terminated; no ATT pipeline conclusion may be drawn from v1.

Profiling infrastructure fix for v2:

- the distributed run wrapper sets debug info to 0 globally;
- `profile_rank0_worker` sets `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1` only inside the
  selected `PROFILE_GLOBAL_RANK` branch;
- non-profile peer ranks retain normal cache behavior and participate only as
  required for distributed progress;
- use a new v2 tag so partial v1 output cannot be mistaken for a valid capture.

This change affects profiling setup, not baseline kernel performance. Confirm
the v2 run reaches and decodes the intended dispatch before interpreting
stall percentages or CTA roles.

The v2 capture is also invalid. With shader-engine mask `0xf`, it ran for about
13 minutes at 100% GPU utilization without producing a `ui` dispatch output.
This indicates severe multi-SE ATT perturbation/loss of forward progress for
the resident-grid workload, compounded by possible inherited debug-info state.
Exact-tag processes were terminated. The run provides no timing or pipeline
evidence.

The minimal-forward-progress v3 setup is:

```text
tag:          ranklocal_baseline_full_v3
SE mask:      0x1 (single shader engine)
target CU:    8
profile rank: selected global rank only
```

Before `exec`, both the parent and every non-profile rank explicitly unset
`FLYDSL_DEBUG_ENABLE_DEBUG_INFO`; only the selected profile-rank branch sets it
to 1. Start with one SE/CU to prove dispatch progress and successful decode,
then cover other SE/CU combinations in separate runs. Do not return to a
multi-SE mask until a low-perturbation configuration is demonstrated.

The v3 single-SE attempt is invalid as well. After roughly 14 minutes it still
had no `ui` dispatch output. Process telemetry on node 46 showed rank 0 at
about 198% CPU and each of the other seven ranks at about 95% CPU. Thus the
non-profile peers were still building missing rank-specialized artifacts; the
run had not reached ATT trace execution, and reducing the SE mask could not
solve first-build cache misses. Exact-tag processes were terminated and both
nodes released.

There are no new ATT files, kernel timings, or stall conclusions from v3. Keep
the profiling-infrastructure edits, but do not launch another direct
`torchrun` capture until this hard prerequisite is satisfied:

1. Without rocprof/ATT, generate all 16 exact-configuration, rank-specialized
   cache artifacts in a controlled serialized or low-concurrency workflow.
2. Run a non-profile distributed smoke check and prove all peer ranks hit cache
   and reach the target kernel promptly.
3. Only then launch ATT, allowing the selected rank 0 debug artifact to rebuild
   while all non-profile peers reuse their verified cached artifacts.

This sequencing is required both for a valid baseline benchmark and a valid
ATT capture; merely enabling cache does not populate missing per-rank keys.

### Controlled rank-specialized baseline precompile

A reusable precompile driver was added:

```text
scripts/megamoe_tile/precompile_stage2_rank_local_baseline.sh
```

It compiles the exact rank-local baseline configuration with cache enabled and
debug info unset. Work is strictly serialized within each node so only one
rank artifact compiles there at a time; the two nodes may proceed in parallel.
The active mapping is:

```text
node 46: global ranks 0..7  -> local devices 0..7, exec session 24603
node 50: global ranks 8..15 -> local devices 0..7, exec session 82548
```

An earlier inline SSH-loop attempt expanded/escaped variables incorrectly and
exited in zero seconds without occupying a GPU or producing a compile result.
It has no performance or correctness significance. The standalone script is
the corrected active workflow.

For each rank, preserve exit status, wall time, kernel/cache identity, and
resource output. After all 16 complete, perform a second cache-hit smoke pass;
do not assume successful first compilation alone proves that the distributed
launcher resolves the same cache keys.

Initial controlled-precompile results reveal a node-specific divergence:

| Node/rank | Result | Wall time | Resource/interpretation |
|---|---:|---:|---|
| 50 / global rank 8 | exit 0 | 136 s | LDS 28,992 B; normal compile gate |
| 50 / global rank 9 | exit 0 | 133 s | LDS 28,992 B; normal compile gate |
| 50 / global rank 10 | in progress at milestone | — | serialized continuation |
| 46 / global rank 0 | inner rc 143 after TERM | >352 s | abnormal single-process compile stall |

Node 46 rank 0 ran without compile concurrency or competing rocprof/benchmark
processes. Its main thread remained near 100% CPU while the other threads
waited in futex. It was terminated after 352 seconds and the node-46 batch was
stopped; do not proceed with ranks 1..7 until the node's environment/cache/
compiler difference is isolated. Because the command piped output through
`tee`, the outer SSH command reported exit 0; the logged inner rc 143 is the
authoritative status. The precompile script should preserve pipeline status
explicitly (for example with `pipefail`) before relying on its aggregate exit.

Follow-up narrowed this diagnosis: node 46 global rank 1 on device 1 compiled
successfully in 135 seconds with LDS 28,992 B. Thus node 46 is not generally
slow; the anomaly is specific to global rank 0/device 0 or its existing
profiling/cache state. Node 46 ranks 2..7 are now compiling serially in exec
session 38123, while rank 0 is deferred for isolated cleanup/rebuild.

Node 50 subsequently completed global rank 10 in 134 seconds and rank 11 in
136 seconds; rank 12 is in progress under the original exec session 82548.
Together with ranks 8 and 9, these results establish a stable 133--136 second
single-rank compile range on node 50.

The >352-second result must therefore be labeled rank0/device0-specific, not a
node-wide compiler issue. It is not evidence that baseline kernel generation
normally takes >352 seconds and is not a runtime performance measurement.

Controlled precompilation is now complete for all 16 rank-specialized
artifacts:

```text
node 50, global ranks 8..15: all exit 0, 133--137 s each
node 46, global ranks 1..7:  all exit 0, 133--136 s each
node 46, global rank 0:      exit 0 in 137 s when retried on device 1
```

The successful rank-0/device-1 retry proves the earlier >352-second event was
device-0 residual profiling/cache state rather than rank-0-specific generated
code. Cache artifacts are complete; the next technical gate is the planned
16-rank cache-hit smoke, followed by baseline performance and ATT.

Execution is currently blocked by external occupancy on node 46. Eight
`ATOM::DP*` processes owned by UID `14517920` occupy approximately 255--256 GiB
per GPU at 100% utilization and had run for roughly 29 minutes at observation.
They are not `hzm_work` processes and must not be terminated. Node 50 is idle.
Status: **cache artifacts complete; blocked on node-46 external occupancy**.

After node 46 became available, a 16-rank cache-hit smoke was launched:

```text
tag:             baseline_cachehit_smoke_20260901
master port:     29719
node 46 session: 62835
node 50 session: 23315
```

At about 90 seconds all 16 workers were still near 98% CPU with GPUs at 0%, so
the run was still compiling and had no runtime result. This does not invalidate
the completed Stage2 cache artifacts: the controlled precompile script invokes
`compile_ep16_stage2_fused.py` and populates Stage2 only, while construction of
the complete fused operator also requires rank-specialized Stage1 artifacts.
The smoke is currently filling that missing Stage1 cache layer.

The run will be allowed to finish rather than repeatedly discarding first-build
work. Until it reaches GPU execution and exits, do not label it a cache-hit
success, runtime hang, correctness result, or performance datapoint. After
completion, distinguish Stage1 build time from subsequent warm cache startup.

The smoke ultimately became a valid **runtime-hang** observation. At
approximately 807 seconds all 16 workers were still
alive with roughly 100% CPU usage and all 16 GPUs across both nodes at 100%
utilization. Logs remained at Gloo connection establishment with no warmup,
iteration, or status output. The early GPU-idle interval was compilation, but
the later all-GPU-active interval proves the processes entered a persistent
device flow and failed to make forward progress; this is not explainable as a
remaining cache miss alone.

Termination required a correction: the first tag-based `pkill` matched
nothing because the benchmark tag was not present in the Python argv, and the
processes were still running on inspection. They were subsequently terminated
using `master-port=29719` and the explicit torchrun PIDs. A final check on both
nodes confirmed no benchmark/torch processes and 0% GPU utilization. Only this
second termination is considered successful.

Preserve logs for tag `baseline_cachehit_smoke_20260901`. Current gate status:

```text
Stage2 rank-specialized artifacts: 16/16 complete
full baseline distributed runtime: hang / not validated
performance result:                none
ATT permission:                    blocked until runtime is healthy
```

The local tracked `stage2.py` difference is only a user-requested
`# agent:` Chinese comment. It was not synchronized to the remote nodes and is
not a possible source of this hang. Do not discard it when restoring or
cleaning test-infrastructure changes.

### Baseline runtime-hang isolation

SHA256 checksums for eight critical files match exactly between nodes 46 and
50: Stage1, Stage1 ABI, Stage2, Stage2 ABI, the operator, rank-local factory,
breakdown benchmark, and run wrapper. This rules out cross-node source/ABI
drift for the observed hang.

A 16-rank `init_only` diagnostic was launched to isolate initialization,
accumulator clearing, and the grid barrier:

```text
tag:             diag_initonly_20260901
master port:     29720
node 46 session: 94860
node 50 session: 74484
configuration:   grid176 / reducer16 / final4 / rank-local atomic /
                 dynamic-head / compact return
```

Interpretation gate: if `init_only` hangs after reaching GPU execution, focus
on launch/residency, initialization ownership, clear traffic, and grid-barrier
progress before investigating GMM/reducer/final paths. If it completes, advance
through narrowly scoped modes to identify the first stage that loses progress.

`init_only` completed successfully on both nodes with status 0
(`warmup=1`, `iters=1`):

```text
rank-max:      501.043 us
all-rank mean: 285.680 us
```

This confirms forward progress through initialization, rank-accumulator clear,
metadata setup, and the resident-grid barrier. These numbers are diagnostic
mode timings, not fused-performance results.

The next 16-rank isolation run is active:

```text
mode/tag:        gmm2_only / diag_gmm2only_20260901
master port:     29721
node 46 session: 39657
node 50 session: 35874
```

If this mode hangs, the first suspect region becomes persistent GMM work
dispatch/computation or its diagnostic sink. If it passes, continue toward the
atomic epilogue, reducer/return, and final paths independently.

`gmm2_only` completed successfully on both nodes with status 0. Its single
diagnostic sample reported:

```text
rank-max:      728.528 us
all-rank mean: 513.487 us
```

This rules out an independent forward-progress failure in the persistent GMM
queue, main GMM computation, or its CTA-local synchronization. The timing is
diagnostic only and must not be treated as production GMM latency.

The next 16-rank isolation run is:

```text
mode/tag:        atomic_only / diag_atomiconly_20260901
master port:     29722
node 46 session: 77026
node 50 session: 78605
```

This mode isolates the rank-local epilogue/atomic publication, reducer, and
communication path. A hang here would narrow the failure to the interaction
introduced after standalone GMM progress rather than initialization or the
GMM body itself.

`atomic_only` reached 100% GPU utilization on both nodes but produced no result
after approximately 260 seconds. It was classified as a runtime hang and
terminated by master port 29722. This mode enables compute with a zero-valued
GMM epilogue and enables reduction while disabling the ordinary role path, so
the failing region still contains both producer publication and reducer
consumption. It does not yet prove that either side fails independently.

The next producer/consumer discriminator was launched:

```text
mode/tag:        route_store_only / diag_routestoreonly_20260901
master port:     29723
node 46 session: 53029
node 50 session: 31272
```

`route_store_only` runs real GMM plus rank-local atomic accumulation and
publication with reducer, RAIL, and final processing disabled. A pass will
establish independent producer-side progress and move the investigation to
the reducer/consumer interaction. A hang will implicate the producer epilogue
or publication protocol before communication.

`route_store_only` returned from the kernel rather than hanging, but all 16
ranks failed its postcondition. Enhanced snapshots showed:

```text
rank_local_active_tokens = 1024
pending_nonzero          = 1024
ready_missing            = 1024
stage2_error             = 0
```

Every active token retained its initial pending value and lacked ready
publication. Thus `publish_rank_group` had no observable effect at all; this
is not a sparse missed-work case. Stage1 canonical metadata was healthy
(`num_valid=2816`, `tile_alloc=88`, `compute_done=2112`), which shifts the
investigation away from Stage1 route construction and into the Stage2 producer
pipeline/publication control.

A single-variable producer-side discriminator is now running: change only
`group_pipeline_schedule` from `a_double_buffer` to `baseline`, leaving the
rest of the rank-local configuration unchanged.

```text
tag:             diag_route_baselinepipe_20260901
master port:     29725
node 46 session: 51370
node 50 session: 41161
```

If this passes, the likely fault is double-buffer metadata/publication
lifetime. If all pending/ready state remains untouched, inspect persistent GMM
publication control and diagnostic-mode gating shared by both pipeline modes.

### Rank-pending initialization root cause and repair WIP

Enhanced snapshot evidence established that the pending counters were not
merely left at a small expected initial value; they had been massively
over-counted:

```text
active tokens:   1024
nonzero pending: 1024
pending min:     98
pending max:     4900
pending sum:     4,523,554
```

The root cause is in rank-local pending initialization. The loop resides in the
rank-local branch but is not restricted to `bx == 0`; nevertheless it used
`tx/THREADS` as if only one CTA executed it. All 176 CTAs therefore repeated
atomic increments into the same route-derived counters. Producers could not
drain the inflated counts to zero, so ready publication never occurred.
`static_strided` reproduced the same all-undrained state, independently ruling
out the persistent GMM queue as the cause.

The WIP repair changes initialization to a unique whole-grid partition using
`global_tx/grid_threads`. Review also found a required happens-before edge:
bx0 clears pending storage, so a grid-wide synchronization must occur after
that clear and before distributed CTAs begin their unique increments. The WIP
therefore adds this grid sync and retains the existing post-increment grid sync
before producer consumption.

Current compile gate:

```text
node:                 50
mode:                 single-process full, cache=0
exec session:         23004
mandatory threshold: <=180 s
```

Do not run distributed correctness if this exact repaired artifact exceeds the
compile threshold. If it passes, first rerun `route_store_only` and verify that
pending drains and ready publication occurs before testing reducer/full modes.

The repaired full kernel passed the cold compile gate on node 50, global rank
8, with runtime cache disabled:

```text
exit status: 0
wall time:   135.438 s
LDS:         28,992 bytes
threshold:   <=180 s, pass
```

The Stage2 repair was synchronized to node 46. Because the modified source
produces new rank-specialized artifacts, a controlled serialized rebuild is in
progress on both nodes:

```text
node 46 ranks 0..7:  exec session 25859
node 50 ranks 8..15: exec session 80371
concurrency:         one compile per node at a time
```

After all 16 repaired artifacts complete, rerun `route_store_only` first and
inspect pending/ready snapshots. Only after producer publication is verified
should the full reducer/communication path be exercised.

### Expanded final performance acceptance matrix

After repaired correctness passes, the user requires non-ATT performance
comparisons at BS/tokens-per-rank `512`, `1024`, and `4096` between:

```text
fused rank-local Stage2
vs.
standalone MORI gmm2_combine
```

All other target dimensions remain TopK16 / E896 / H7168 / I3072 / EP16 /
A4W4. Unless the user specifies otherwise, report BS here as per-rank token
count (TPR) and state that convention explicitly in every result table.

The current breakdown harness `_shape()` and one preparation/check message are
hard-coded to 128 tokens. Before running this matrix, add a `--tokens` option
and derive allocations, inputs, checks, and labels from `shape.tokens`; do not
introduce separate 512/1024/4096 constants into kernel indexing or protocol
logic. Validate the harness change at the existing 128-token case first.

For every shape/path preserve:

- exact command and environment, including cache state;
- per-rank timing samples;
- rank-max and all-rank aggregates;
- output/performance file and log paths;
- correctness status before interpreting timing.

ATT must be disabled for these acceptance measurements. The goal remains a
real fused advantage over MORI `gmm2_combine`, not merely improvement over a
slower fused revision.

### Pending-init repair producer validation

All 16 repaired rank-specialized artifacts completed rebuilding in 132--138
seconds each. The repaired `route_store_only` run then passed on both nodes
with status 0, including the pending/ready validator. Its single diagnostic
sample was:

```text
rank-max:      1272.133 us
all-rank mean: 913.221 us
```

This confirms the unique whole-grid pending initialization and publication
repair restores producer-side counter drain/ready behavior. The timing is
diagnostic and is not a production fused performance result.

A full 16-rank smoke is now running to validate reducer, RAIL, and final
integration:

```text
tag:             pendingfix_full_smoke_20260901
master port:     29729
node 46 session: 53163
node 50 session: 18770
```

Do not start the expanded performance matrix or ATT until this full path exits
successfully and its protocol/correctness status is inspected.

The repaired full 16-rank smoke completed successfully on both nodes with
status 0 and no hang. Its single-sample timing was:

```text
rank-max:      1717.976 us
all-rank mean: 1580.610 us
```

This is a forward-progress smoke result only, not final performance evidence.
It confirms the pending-initialization repair restores the integrated
GMM/producer/reducer/RAIL/final path sufficiently to proceed to numerical
validation.

Paired four-generation validation has begun by generating direct references:

```text
tag:             pendingfix_paired_v1
mode:            direct, then rank with the same references
generations:     4
master port:     29730
node 46 session: 15891
node 50 session: 88562
```

Record both direct and rank exit statuses, protocol errors, generation
coverage, and maximum relative L2. Expanded BS performance and ATT remain
blocked until the rank-mode numerical check passes.

Paired four-generation validation completed successfully:

```text
direct mode:      both nodes status 0
rank mode:        both nodes status 0
generations:      4
protocol errors:  0
rank max abs:     0.1875
rank max rel L2:  0.0056015821
required rel L2:  <0.05, pass
```

The pending-init repair therefore passes the paired route numerical and
protocol gate. The next correctness stage is the same direct-then-rank process
with `permuted-arbitrary-topk`. Do not promote the repair or begin final
performance claims until arbitrary routing also passes.

Permuted-arbitrary four-generation rank validation also completed successfully
on both nodes:

```text
rank mode:        both nodes exit 0
generations:      4
protocol errors:  0
rank max abs:     0.0625
rank max rel L2:  0.00236463081
required rel L2:  <0.05, pass
```

Both nodes were clean afterward with no remaining test processes and 0% GPU
utilization. The relatively long wall time came from separate full 16-rank
direct-reference and rank launches, fixture setup, allocation/preparation of
the 40 GiB MORI arena and Stage1 state, and first-time generation of
validation-specific artifacts. It was not a runtime kernel hang.

The pending-init repair has now passed the required paired and arbitrary
distributed correctness gates. Work may proceed to the shape-derived
`--tokens` harness change, its 128-token regression, and the non-ATT
512/1024/4096 fused-versus-MORI performance matrix.

### Token-count harness generalization

The breakdown benchmark now accepts `--tokens` in the inclusive range
1..4096. `_shape(tokens)` derives the problem size, the old fixed-128 check was
removed, and case labels are generated dynamically. The shell wrapper exposes
tokens as positional argument 33. This supersedes the earlier temporary use of
position 33 for the now-stashed queue-sync experiment; commands must follow the
current wrapper definition.

Validation of the harness change:

```text
py_compile:       pass
bash -n:          pass
git diff --check: pass
sync to 46/50:    complete
TPR128 full CLI:  both nodes exit 0
```

The unused local `COMPARISON_CASE_LABEL` import may be cleaned up later but has
no functional or performance impact. The harness is now cleared to begin the
non-ATT TPR512 fused measurement, followed by the matching MORI
`gmm2_combine` run and then TPR1024/4096.

### Kernel-level token generalization plan

The user authorized removal of the kernel's fixed token-count limitation. For
this work, BS is interpreted as tokens per rank (TPR) unless the user corrects
that convention. Required target values are 128 regression plus 512, 1024, and
4096; TopK16 / E896 / H7168 / I3072 / EP16 / A4W4 remain fixed for the first
performance matrix.

Implementation plan and gates:

1. **Shape-derived capacity.** Replace fixed `MAX_TOKENS` assumptions with a
   compile-time capacity derived from the requested shape. Keep bounds explicit
   and reject unsupported sizes rather than silently truncating. Kernel names
   and cache keys must encode every capacity-affecting compile-time choice.
2. **Bounded Stage1 scheduling.** Do not launch producer CTAs proportional to
   token count. Convert token/tile production to a bounded persistent work
   schedule with a shape-derived total-work count and a fixed, occupancy-safe
   CTA budget. Prove unique work ownership, uniform CTA barriers, terminal
   handling, and generation reuse.
3. **ABI and storage audit.** Trace every token-indexed region and offset through
   Stage1 ABI, Stage2 ABI, workspace/arena sizing, route metadata, rank-local
   accumulators and pending/ready queues, completion/final queues, RAIL payload
   and source/destination encoding. Check integer widths and alignment at TPR4096.
   Up to 4 GiB of additional workspace is permitted, but actual per-region and
   total byte growth must be reported.
4. **Host/harness propagation.** Pass `shape.tokens` consistently through the
   public operator, factories, compile scripts, validation, benchmark, cache
   identity, tensor allocation, reference generation, and labels. No target
   value may be embedded as a special-case kernel index.
5. **Static and compile gates.** Run Python/shell syntax checks, contract tests,
   source/ABI assertions, and resource reports. Enforce <=180 seconds for each
   representative single-process cold full compile; investigate any violation
   before 16-rank execution.
6. **Correctness ladder.** First rerun TPR128 paired and arbitrary checks to
   prove no regression. Then for TPR512, 1024, and 4096 run direct references
   followed by fused rank validation, multiple generations, protocol-error
   checks, pending/ready drain checks, and numerical thresholds. Use smaller
   diagnostic modes first if a full shape fails.
7. **Performance matrix.** With ATT disabled and warm cache verified, collect
   fused and MORI `gmm2_combine` results for every target TPR. Preserve exact
   commands/environment, per-rank samples, rank-max/all-rank aggregates, and
   perf/log files. Use same-run ordering or sandwiches to control node noise.
8. **ATT only on a validated regression.** If fused does not beat MORI, collect
   role-complete traces at that shape: GMM/epilogue, rank reducer, RAIL/QP, and
   final/unpermute. Profile one rank/SE/CU at a time after all peer artifacts are
   warm, classify traces by PC/source, and do not use ATT duration as latency.

No implementation is promoted unless it preserves the fused GMM2+combine
kernel, passes the 128 regression and target-shape correctness, stays within
the compile-time and memory budgets, and demonstrates measured fusion benefit.

### Planner repository audit before kernel token work

The current repository state was re-audited against this handoff:

```text
branch: dev/stage2_compute_pipeline_v1
HEAD:   a69a16420
git diff --check: pass
```

Tracked changes currently comprise five files:

- `stage2.py`: the validated pending-init partition/synchronization repair plus
  user-requested explanatory `# agent:` comments;
- `mega_moe_tile_a4w4.py`: pending-counter diagnostic snapshot fields;
- the breakdown benchmark and wrapper: shape-derived CLI `--tokens`, dynamic
  labels, diagnostics, and cache-default correction;
- `profile_rank0_worker.sh`: debug-info isolation to the profiled rank.

The failed GMM queue work is preserved in `stash@{0}` with the explicit name
`persistent GMM queue tail-claim and chunk-claim failed WIP`; older rejected
experiments remain in stashes 1..5. The checkpoint, precompile/profile scripts,
and trace data remain untracked and must be preserved.

The handoff is consistent with the code: kernel token generalization has **not
yet started**. `stage1.py` still defines `MAX_TOKENS = 128` and
`PRODUCER_CTAS = MAX_TOKENS`, and the public operator still requires
`run_tokens == self.mtpr` with a fixed-128 error message. Therefore the next
worker must begin with the Stage1 scheduling/ABI audit rather than attempting
TPR512 execution from the already-generalized benchmark CLI alone.

### Token generalization compile milestone

Host/ABI contract tests passed on node 50 (`46 passed`). A real cold compile of
both fused-operator kernels established the TPR128 reference:

```text
TPR128 two-kernel cold compile: exit 0
wall time:                      136.408 s
Stage1 LDS:                     32,768 bytes
Stage2 LDS:                     28,992 bytes
arena:                          229,187,584 bytes
```

The same cache-disabled script at TPR512 timed out after 400.09 seconds without
producing kernel/resource output. This violates the mandatory <=180-second
compile gate. TPR512 must not proceed to 16-rank execution.

Next diagnostic step: compile Stage1 and Stage2 independently for the same
TPR512 configuration and record per-kernel wall time/resource output. Inspect
the failing side for token-capacity-dependent Python/DSL static loops, unrolled
branches, replicated CTA-role bodies, and capacity-sized constant control flow.
Do not treat the 400-second timeout as runtime performance, and do not mask the
regression through cache reuse.

The TPR512 compile-time root cause was isolated to a staged-ring branch which
is runtime-unreachable for the active rank-local atomic configuration but is
still built into IR. It used `range_constexpr(SOURCE_CAPACITY)`, expanding 8192
source iterations at TPR512. Converting this capacity-scaled static expansion
to a runtime loop removed the compiler explosion.

Fresh node-50 cache-disabled results for the repaired exact TPR512 artifact:

```text
init_only compile:  12.557 s
full compile:       13.814 s
two-kernel compile: 17.762 s
Stage1 LDS:         32,768 bytes
Stage2 LDS:         28,992 bytes
Stage1 arena:       571,559,936 bytes
Stage2 arena:       324,947,968 bytes
total arena:        896,507,904 bytes
```

Compilation now passes the <=180-second gate and the total arena remains below
the allowed 4 GiB budget. This is compile/resource evidence only; TPR512 still
requires distributed correctness before performance measurement.

Two review blockers were also repaired:

- restore the exact `run_tokens == capacity` public gate, preventing a caller
  from launching a capacity-specialized kernel with a different token count;
- encode return-phase generation as
  `generation * (batch_count + 1) + batch + 1`, preventing phase reuse/collision
  when the number of return batches grows with token capacity.

TPR512 may now enter controlled 16-rank validation. Rebuild/verify all required
rank-specialized artifacts, then use direct reference followed by rank mode;
inspect protocol errors, generation coverage, pending/ready drain, and numerical
thresholds before any timing claim.

### TPR512 RDMA registration blocker

The first TPR512 16-rank full attempt did not enter the kernel. MORI failed
while registering the single combined arena:

```text
combined arena: 896,507,904 bytes
Stage1 region:  571,559,936 bytes
Stage2 region:  324,947,968 bytes
error:          RegisterRdmaMemoryRegionDmabufIova0 errno 22
                Allgather failed
```

This is neither a compile failure nor a generated-kernel/runtime hang. Although
the allocation is below the user-approved 4 GiB workspace budget and device
memory is sufficient, the active MORI/DMABUF backend has a smaller or otherwise
constrained single-registration-window limit.

TPR512 correctness and performance are blocked at memory registration. Do not
continue the performance matrix until one of these is implemented and validated:

1. reduce worst-case token/route storage through a compact, shape-derived
   layout; or
2. split Stage1/Stage2 and/or parity planes into multiple separately registered
   windows while preserving every ABI address, remote offset, and generation
   lifetime.

Any redesign must report individual registered-region sizes, not only total
workspace, and must re-run host/ABI contracts, the <=180-second cold compile
gate, and TPR128 regression before TPR512 distributed correctness.

Region audit identified the dominant TPR512 capacity costs:

```text
Stage1 total:             571,559,936 bytes
  h1_output_q:            408,158,208 bytes
Stage2 total:             324,947,968 bytes
  rank_accumulator:       234,881,024 bytes
combined registration:   896,507,904 bytes
```

`h1_output_q` is sized for the worst case in which all TopK routes of every
source token land on the same rank. The proposed compact-capacity contract adds
a shape-derived compile-time `max_routes_per_token_per_rank`:

- default remains `topk` (16) for backward-compatible unrestricted routing;
- performance cases may select cap 1 only when the actual input routing obeys
  at most one destination expert on any single rank per source token;
- host/preflight code must compute route multiplicity and reject overflow with
  a clear error before launch;
- no route may be silently truncated, overwritten, or wrapped;
- the cap must participate in ABI/layout contracts, cache/kernel identity,
  arena sizing, reference validation, and result metadata.

Stage1 route capacity then becomes `source_capacity * cap` rather than
`source_capacity * topk`. The current estimate is that TPR512 cap1 reduces the
combined arena below roughly 512 MiB, but the exact per-region sizes and MORI
registration must be measured. This is an explicit input-capacity contract,
not a hidden fixed assumption. Cap1 cannot be used for arbitrary-route tests
whose multiplicity exceeds one; those must retain a sufficient cap or fail
preflight intentionally.

### Formal TPR512 performance: fused cap2

The first formal non-ATT TPR512 fused run completed with status 0 on both
nodes. BS is interpreted as 512 tokens per rank. Configuration used the
explicit per-token/per-rank route cap 2 and warm cache:

```text
warmup / iterations / tail: 5 / 30 / 20
rank-max tail mean:          4375.971 us
rank-max P50:                4327.685 us
rank-max P95:                4805.569 us
all-rank tail mean:          3970.303 us
ATT:                         disabled
```

Logs:

```text
/home/hzm/logs/megamoe_stage2_breakdown_20260824/perf_tpr512_fused_cap2_v1_node0.log
/home/hzm/logs/megamoe_stage2_breakdown_20260824/perf_tpr512_fused_cap2_v1_node1.log
```

This establishes the fused datapoint only. A matching MORI `gmm2_combine` run
is in progress; no fusion-benefit conclusion is valid until its rank-max and
all-rank statistics are recorded under the same shape and timing protocol.

The matching MORI TPR512 run completed with status 0 using an 8 GiB MORI heap,
ATT disabled, and the same `warmup=5 / iterations=30 / tail=20` protocol:

| Path/metric | Rank-max mean | P50 | P95 | All-rank mean |
|---|---:|---:|---:|---:|
| fused cap2 full | 4375.971 us | 4327.685 us | 4805.569 us | 3970.303 us |
| MORI GMM2+combine | 6019.678 us | 4682.267 us | 8724.527 us | 5094.844 us |
| MORI GMM2 component | 1066.860 us | 955.889 us | — | — |
| MORI combine component | 5109.135 us | 3757.238 us | — | — |

On this run fused is faster by 27.3% using rank-max mean, 7.6% using rank-max
P50, and 22.1% using all-rank mean. The large MORI P95 and the gap between its
mean and median show substantial outliers, primarily in combine, so these are
promising first-run fusion gains rather than final stable percentages. Repeat
the matched comparison and preferably use baseline/candidate ordering or a
sandwich before sign-off.

MORI logs:

```text
/home/hzm/logs/megamoe_stage2_breakdown_20260824/perf_tpr512_mori_gmm2combine_heap8g_v1_node0.log
/home/hzm/logs/megamoe_stage2_breakdown_20260824/perf_tpr512_mori_gmm2combine_heap8g_v1_node1.log
```

The earlier 40 GiB heap OOM must not be attributed to summing memory across
GPUs: each GPU has its own 288 GiB address space. An 8 GiB heap works, but the
specific cause of the 40 GiB failure remains unproven and should be described
as unresolved rather than a device-capacity conclusion.

### TPR512 MORI FP8 direct-cast performance

A formal MORI FP8 direct-cast run completed with status 0 using heap 8 GiB,
ATT disabled, and `warmup=5 / iterations=30 / tail=20`:

| Component | Rank-max mean | P50 | P95 | Rank-min mean | All-rank mean |
|---|---:|---:|---:|---:|---:|
| GMM2 | 1007.858 us | 1007.408 us | 1023.248 us | 894.416 us | 952.018 us |
| FP8 combine | 2090.592 us | 2089.962 us | 2145.062 us | 1663.833 us | 1871.290 us |
| total | 3000.456 us | 2996.313 us | 3058.752 us | 2648.078 us | 2823.308 us |

Logs:

```text
/home/hzm/logs/megamoe_stage2_breakdown_20260824/perf_tpr512_mori_fp8_rankrange_heap8g_v1_node0.log
/home/hzm/logs/megamoe_stage2_breakdown_20260824/perf_tpr512_mori_fp8_rankrange_heap8g_v1_node1.log
```

Against the corresponding non-FP8 MORI total measurement
`5606.746 / 5592.065 us`, FP8 direct-cast is about 46.5% faster. The combine
rank-max mean falls from 5109.135 to 2090.592 us, a 59.1% improvement. Against
the same-period fused result of 4302.642 us, FP8 MORI total is about 30.3%
faster; equivalently fused is about 43.4% slower relative to FP8 MORI.

Critical qualification: this FP8 path has only passed a finite-output check.
It has not yet been numerically compared against the BF16/reference result or
shown to satisfy the fused operator's accuracy contract. Treat it as a
performance candidate/upper-bound comparison, not an accuracy-equivalent
small-operator baseline. Before it can become the final target, record max abs,
relative L2, routing/protocol equivalence, and the exact cast/scaling semantics.

### Proposed tile-granular Stage2 overlap before blockwise FP8

The next optimization direction is tile-granular readiness and communication
overlap, implemented before attempting blockwise FP8. It must be opt-in; the
validated token-granular rank-local mode remains the default and rollback path.

Protocol geometry:

```text
tile width:                  BN=256
H7168 tiles/token:           28
initial hidden envelope:     H<=10240 => <=40 tiles
ready group tiles:           n_tile_group=2
H7168 ready groups/token:    14
readiness representation:    uint64 group-generation/mask contract
GMM n_tile_group:            2 (unchanged)
```

A uint64 mask covers the requested hidden envelope at GMM-work group
granularity. Define `hidden_tiles=ceil(H/BN)`,
`ready_group_tiles=n_tile_group`, and
`ready_group_count=ceil(hidden_tiles/ready_group_tiles)`. Bits at or above
`ready_group_count` must remain zero and must never be interpreted as ready
payload.

Proposed flow:

1. **N-major GMM work swizzle.** Keep two N tiles per GMM work, but order work
   so a fixed N-group is processed across M work before moving to the next
   N-group. Each completed work publishes completion for its two output tiles.
2. **Tile producer state.** Replace/augment token-wide pending state with
   `rank_tile_pending` and generation-safe `rank_tile_ready`. A tile becomes
   ready only after all local route contributors for that token/tile finish.
3. **Tile node reduction.** The source-proxy reducer claims `(token, tile)` as
   soon as peer ranks required for that tile are ready, peer-pulls the BN-wide
   slices, and performs one FP32 node reduction. It does not wait for all tiles
   of the token.
4. **Node-ready mask.** Completed node partial tiles atomically/generation-safely
   update a uint64 `node_ready_mask` for the source token.
5. **RAIL batching.** bx0 consumes ready groups and aggregates 4, 8, or 12
   groups per WQE/flush (8/16/24 tiles when n_tile_group=2). Each message carries
   the uint64 group mask so the destination
   can place sparse/tail batches correctly. A generation-ending partial batch
   must flush even when below the configured batch size.
6. **Final tile combine.** Final work unpacks mask-selected tiles and accumulates
   local/remote node partials directly into the corresponding output slices;
   it must not wait for whole-token readiness. If the destination is node-local,
   the local tile-ready path bypasses unnecessary RAIL while preserving the
   same final ownership semantics.

No new scheduler CTA is planned. Reuse the existing persistent GMM queue,
dynamic reducer work queue, bx0 RAIL/progress role, and final queue. CTA budgets
must remain bounded and shape-derived.

Implementation/measurement phases:

1. **Swizzle only:** N-major versus current ordering, using `gmm2_only` A/B;
   require identical work coverage and no compile/resource regression.
2. **Tile reducer without RAIL:** validate producer pending/ready masks and
   node-local FP32 tile reduction, including local-destination flow.
3. **Tile RAIL:** test batch sizes 4, 8, and 12 with identical correctness and
   measure WQE count, flush count, reducer-to-RAIL latency, and full fused time.
4. **Final tile path:** validate sparse masks, out-of-order tile arrival, and
   final output coverage; compare against token-granular baseline.
5. **Only after tile overlap wins:** layer blockwise FP8 transport/combine on
   the validated tile protocol and perform independent numerical qualification.

Mandatory correctness and liveness invariants:

- exactly one decrement per valid local `(route, tile)` contribution and no
  tile publication before its pending count reaches zero;
- exactly one reducer claim per `(generation, token, tile)` and no duplicate
  peer-pull or final accumulation;
- release publication/acquire consumption for payload-before-ready ordering;
- generation-safe mask reset/reuse: no stale bit may satisfy the next
  generation, including parity wrap;
- all required tiles eventually reach final output; padding bits remain zero;
- tail groups include only the remaining valid tiles; a partial final tile uses
  element bounds/masks and never accesses beyond H;
- RAIL partial batches flush at generation/end-of-token progress boundaries;
- sparse/out-of-order masks cannot deadlock a FIFO head, final queue, or quiet
  protocol;
- node-local destinations do not wait for a nonexistent remote message;
- errors/overflow are explicit; no mask bit, queue item, or tile is silently
  dropped;
- multi-generation paired and arbitrary routes preserve the existing relative
  L2 threshold and protocol-error count of zero.

Each phase requires a unique cache key, static/contract review, <=180-second
cold compile, TPR128 correctness first, then target-shape correctness and a
same-run performance A/B. If a phase loses forward progress, isolate producer,
tile reducer, RAIL, and final modes before adding the next layer.

Implementation is now active under this staged gate table:

| Phase | Scope | Current status | Promotion gate |
|---|---|---|---|
| 1 | opt-in N-major/windowed GMM work swizzle | in progress | `gmm2_only` correctness/coverage, <=180 s compile, same-run A/B win |
| 2 | tile pending/ready plus tile reducer; existing BF16 RAIL unchanged | pending | producer/reducer diagnostics, paired+arbitrary correctness, full A/B |
| 3 | uint64 tile-mask RAIL and tile final; batch 4/8/12 | pending | tail/sparse/multi-generation correctness and best stable batch timing |
| 4 | node-reduced `fp8_blockwise` | pending | explicit scaling/accuracy contract plus fused-vs-MORI performance |

The default token-granular baseline must remain source- and runtime-selectable
throughout. Phase-1 results must be measured in `gmm2_only`; they are not
evidence for communication/final overlap. Conversely, phase 2 keeps the
existing BF16 RAIL path so tile producer/reducer behavior can be validated
without simultaneously changing transport representation.

### Phase 1 TPR512 swizzle screen

The first cap2 `gmm2_only` comparison at TPR512 produced:

| Schedule | Rank-max mean | P50 | P95 | Rank-min mean | All-rank mean |
|---|---:|---:|---:|---:|---:|
| token-major baseline | 1330.711 us | 1309.031 us | 1415.254 us | 999.783 us | 1191.864 us |
| strict N-major window W1 | 1203.569 us | 1145.072 us | 1458.772 us | 989.808 us | 1081.281 us |

W1 is faster by 9.55% on rank-max mean, 12.53% on rank-max P50, and 9.28%
on all-rank mean. However, its P95 is slightly worse (1458.772 versus
1415.254 us), so the schedule is not yet stable enough for promotion. Continue
the same screen with W2 and W4 and retain per-rank distributions/outlier ranks.

Logs:

```text
/home/hzm/logs/megamoe_stage2_breakdown_20260824/swizzle_tpr512_tokenmajor_v2_node0.log
/home/hzm/logs/megamoe_stage2_breakdown_20260824/swizzle_tpr512_tokenmajor_v2_node1.log
/home/hzm/logs/megamoe_stage2_breakdown_20260824/swizzle_tpr512_nmajor_w1_v1_node0.log
/home/hzm/logs/megamoe_stage2_breakdown_20260824/swizzle_tpr512_nmajor_w1_v1_node1.log
```

This remains a Phase-1 GMM scheduling result only; it does not demonstrate
full fused or tile-communication overlap benefit.

W2 subsequently passed its `gmm2_only`/full smoke and produced a stable full
fused TPR512 performance result:

| Full schedule | Rank-max mean | P50 | P95 | Rank-min mean | All-rank mean |
|---|---:|---:|---:|---:|---:|
| token-major repeat | 4302.642 us | 4294.924 us | — | — | 3949.901 us |
| N-major W2 | 3916.470 us | 3918.115 us | 3962.159 us | 3607.870 us | 3769.115 us |

W2 improves rank-max mean by 8.98%, rank-max P50 by 8.77%, and all-rank mean
by 4.58% relative to the token-major repeat. Unlike the W1 screen, the W2 full
P95 is close to its mean/P50 and does not show the same visible long-tail
spread. W2 is promoted as the Phase-2 scheduling base, while remaining opt-in
until the complete tile protocol passes correctness.

Logs use:

```text
swizzle_tpr512_nmajor_w2_full_smoke_node{0,1}.log
swizzle_tpr512_nmajor_w2_full_perf_v1_node{0,1}.log
```

Phase 2 is now active at the ABI/tile-metadata stage. Keep the existing BF16
RAIL representation unchanged while introducing and validating tile
pending/ready plus tile reducer semantics.

Phase-2 host/ABI plumbing is complete. `ready_granularity` accepts `token` or
`tile` and defaults to `token`, preserving the validated baseline. Tile mode
appends these metadata regions without reinterpreting the token-mode layout:

```text
rank_tile_pending
rank_tile_ready
node_tile_arrived
node_tile_ready
node_ready_mask
tile_reduce_queue
tile_reduce_queue_ready
tile_reduce_queue_head
tile_reduce_queue_tail
```

The option is propagated through the operator, benchmark, factory/compiler,
kernel/cache key, launcher metadata, and shell wrapper (positional argument
40). Static validation passed `py_compile`, `bash -n`, and `git diff --check`.

The tile reducer kernel behavior is not yet implemented. Therefore `tile`
mode must not be launched or treated as a correctness/performance candidate;
only the default `token` mode remains executable at this milestone. The next
gate is implementation/review of unique tile pending initialization,
producer publication, reducer claim/pull, and node-ready mask publication
before any runtime test.

### Hard environment deadline and final-run priority

The environment has a hard cutoff at 10:00 today, with approximately three
hours remaining at this milestone. Execution priority is now:

1. Complete the formal token-major versus N-major W2 matrix for per-rank TPR
   64, 128, 256, and 512 (eight total runs).
2. Use identical settings for every run:

   ```text
   rail quantization: none
   ready granularity: token
   route cap:         2
   warmup/iters/tail: 5 / 30 / 20
   ATT:               disabled
   ```

3. Preserve both-node logs and summarize per-rank samples, rank-max mean/P50/
   P95, rank-min mean, all-rank mean, and relative W2 improvement for each TPR.
4. Only if all eight measurements and log checks finish with time remaining may
   Phase-2 tile-reducer implementation continue.

Do not start ATT, FP8, or other long experiments before the deadline. If a run
requires a first-time compile, distinguish compile/setup wall time from timed
iterations and do not relax the <=180-second cold-compile rule.

### Immediate full handoff snapshot and corrected deadline priority

This section supersedes the eight-run swizzle-matrix priority above. The user
explicitly changed the remaining-window objective: **finish and validate the
complete unquantized tile-overlap pipeline first**, then measure only end-to-end
fused Stage2 at TPR64/128/256/512. W2 is the base schedule, not the final
optimization claim. The 10:00 hard environment cutoff remains in force.

Current repository identity/state:

```text
branch: dev/stage2_compute_pipeline_v1
HEAD:   a69a16420
tracked modified files: 12
git diff --check: pass at snapshot
```

Tracked modified files:

```text
aiter/ops/flydsl/kernels/megamoe_tile/mega_moe_tile_a4w4.py
aiter/ops/flydsl/kernels/megamoe_tile/stage1.py
aiter/ops/flydsl/kernels/megamoe_tile/stage1_abi.py
aiter/ops/flydsl/kernels/megamoe_tile/stage2.py
aiter/ops/flydsl/kernels/megamoe_tile/stage2_abi.py
op_tests/multigpu_tests/bench_megamoe_tile_ep16_stage2_breakdown.py
op_tests/multigpu_tests/bench_megamoe_tile_ep16_two_kernel.py
op_tests/test_megamoe_tile_a4w4_public_contract.py
scripts/megamoe_tile/compile_ep16_stage2_fused.py
scripts/megamoe_tile/compile_ep16_two_kernel.py
scripts/megamoe_tile/profile_rank0_worker.sh
scripts/megamoe_tile/run_stage2_breakdown_ep16.sh
```

Untracked artifacts to preserve:

```text
scripts/megamoe_tile/STAGE2_CU_OPT_CHECKPOINT_20260901.md
scripts/megamoe_tile/precompile_stage2_rank_local_baseline.sh
scripts/megamoe_tile/run_stage2_cu_pull_profile_ep16.sh
trace_data/
```

Saved rejected experiments:

```text
stash@{0}: persistent GMM queue tail-claim and chunk-claim failed WIP
stash@{1}: rank-init single-owner WIP
stash@{2}: Stage1 preclear WIP
stash@{3}: consumer-zero remote reset WIP
stash@{4}: CU-pull scan/final-cap WIP
stash@{5}: full producer-push WIP
```

Remote synchronization boundary: token-capacity/route-cap work, the pending-init
repair, and P1 W2 were exercised on nodes 46/50, so those tested revisions were
present remotely. The newest P2 ABI/host plumbing has only a recorded local
static pass; no remote tile-mode run is valid, and synchronization of the exact
current 12-file diff is not yet confirmed. Before the next remote compile/run,
sync the required files to both `/home/hzm/aiter` worktrees and compare hashes.

#### Completed foundation

- Token capacity is shape-derived and the Stage1 producer schedule is bounded;
  TPR512 compile explosion from staged-ring
  `range_constexpr(SOURCE_CAPACITY)` was removed with a runtime loop.
- `max_routes_per_token_per_rank` is an explicit compile/layout contract.
  Cap2 is used by the TPR512 performance fixture and route multiplicity is
  checked before launch; overflow is rejected, never silently truncated.
- MORI single-window registration failed at 896,507,904 bytes. Route-capacity
  compression brought the tested configuration into a registrable range; the
  backend single-window constraint remains distinct from the 4 GiB workspace
  budget.
- The rank-pending initialization bug is fixed: use
  `global_tx/grid_threads`, with a grid sync after bx0 clear and another after
  distributed increments. Paired and arbitrary four-generation validation
  both passed with protocol errors 0.
- Exact TPR512 repaired compile reference: two kernels 17.762 s; Stage1 LDS
  32,768 B; Stage2 LDS 28,992 B; cap-era sizes must always be read from the
  selected current layout rather than the older 896 MB unrestricted layout.

#### Preserved TPR512 performance references

```text
fused cap2 token-major formal:
  rank-max mean/P50/P95 = 4375.971 / 4327.685 / 4805.569 us
  all-rank mean         = 3970.303 us

same-period token-major repeat used for W2 A/B:
  rank-max mean/P50     = 4302.642 / 4294.924 us
  all-rank mean         = 3949.901 us

MORI BF16 GMM2+combine first formal run:
  rank-max mean/P50/P95 = 6019.678 / 4682.267 / 8724.527 us
  all-rank mean         = 5094.844 us
  note: large combine outliers; repeat required for stable claim

MORI FP8 direct-cast:
  rank-max mean/P50/P95 = 3000.456 / 2996.313 / 3058.752 us
  all-rank mean         = 2823.308 us
  note: finite-only; no BF16/reference numerical qualification

N-major W2 full fused:
  rank-max mean/P50/P95 = 3916.470 / 3918.115 / 3962.159 us
  rank-min mean         = 3607.870 us
  all-rank mean         = 3769.115 us
  versus token-major repeat: mean -8.98%, P50 -8.77%, all-rank -4.58%
```

Existing log families to preserve:

```text
perf_tpr512_fused_cap2_v1_node{0,1}.log
perf_tpr512_mori_gmm2combine_heap8g_v1_node{0,1}.log
perf_tpr512_mori_fp8_rankrange_heap8g_v1_node{0,1}.log
swizzle_tpr512_tokenmajor_v2_node{0,1}.log
swizzle_tpr512_nmajor_w1_v1_node{0,1}.log
swizzle_tpr512_nmajor_w2_full_{smoke,perf_v1}_node{0,1}.log
```

#### Current P1/P2 implementation boundary

- P1 is implemented: opt-in `gmm_work_swizzle=n_major_window`, W2 promoted as
  the scheduling base. The mapping logic is in `stage2.py` inside
  `run_gmm_work` near the current source line 5158; the N-major/window mapping
  begins near line 5164. Default remains `token_major`.
- P2 plumbing is implemented but kernel semantics are not: `ready_granularity`
  is `token|tile`, default token, and participates in the kernel key/launcher.
  Shell wrapper positional argument 40 is `candidate_ready_granularity`.
- Tile ABI append regions are constructed in `stage2_abi.py` near line 683:
  `rank_tile_pending`, `rank_tile_ready`, `node_tile_arrived`,
  `node_tile_ready`, `node_ready_mask`, `tile_reduce_queue`,
  `tile_reduce_queue_ready`, `tile_reduce_queue_tail`, and
  `tile_reduce_queue_head`.
- Do not run tile mode yet: none of pending initialization, producer tile
  publication, tile reducer peer-pull, mask RAIL, or tile final is complete.

#### Next concrete Stage2 code entry points

Implement unquantized tile mode behind
`ready_granularity == "tile"`, leaving the token branches untouched:

1. Rank-local init around the existing accumulator/pending metadata setup near
   `stage2.py:1000..1370`: initialize per `(source,tile)` pending and generation
   state with unique whole-grid partitioning and clear/increment barriers.
2. Producer completion in `publish_rank_group` near `stage2.py:4172`: for the
   two N tiles of a GMM work, decrement tile pending and publish tile-ready only
   when that tile's local contributors are complete.
3. Reducer path beginning at the rank-local peer-pull section near
   `stage2.py:3099`: claim `(token,tile)`, wait/pull only the BN=256 slices from
   participating peer ranks, FP32-reduce once, write node partial, and set the
   generation-safe `node_ready_mask` bit.
4. Compact bx0 RAIL role near `stage2.py:2032`: consume ready mask bits and
   aggregate 4/8/12 ready tiles per WQE/flush; carry a uint64 mask and flush a
   final partial batch. First implementation is BF16/unquantized.
5. Final role near `stage2.py:3827`: accept sparse/out-of-order tile masks,
   accumulate only selected local/remote tile slices, and publish output tiles
   without a whole-token wait. Node-local destinations must not wait for RAIL.
6. Extend diagnostic snapshot/validator before full mode: pending nonzero,
   ready/mask missing, duplicate claims, queue head/tail, padding-mask bits,
   and per-tile final coverage.

#### Quantization sequencing

`rail_quant_type` is already a compile-time option (`none|fp8_blockwise`) and
is represented in host/ABI/kernel key plumbing, including optional scale
storage. This is preparation only; it is not a validated FP8 tile transport.
Per the user's latest priority, keep `rail_quant_type=none` until the complete
BF16 tile-ready -> peer-pull reduce -> mask RAIL -> tile-final path is correct
and faster end to end. Only then implement/validate node-reduced
`fp8_blockwise`, including scale semantics and numerical error.

#### Remaining-time execution priority (supersedes prior matrix plan)

1. Finish the complete unquantized tile-overlap kernel path above.
2. Static review and <=180 s cold compile; do not bypass the gate.
3. Run producer/reducer/RAIL/final diagnostics, then paired/arbitrary
   multi-generation correctness. Do not launch full before the narrower modes
   prove pending/masks/queues drain.
4. Measure **only full fused Stage2 end-to-end** at TPR64/128/256/512 against
   the validated token-granular baseline. Use W2 as the common base schedule;
   do not present swizzle-only benefit as tile-overlap benefit.
5. No ATT and no FP8 experiment before the 10:00 cutoff unless the complete
   unquantized tile pipeline and four-shape end-to-end measurements are done.

For every last-window run retain the exact command/environment, both-node logs,
status, correctness summary, and rank-max/P50/P95/all-rank timing. If time
expires, stop at a compilable named WIP boundary and append the precise missing
function/phase here rather than launching an unreviewed long run.

### Emergency P2 implementation checkpoint

The unquantized tile producer/reducer kernel path is now materially implemented
in `stage2.py`:

- tile ABI offsets and pointers are resolved for the active parity/generation;
- initialization builds `rank_tile_pending` from every valid route across all
  28 H7168/BN256 tiles;
- `publish_rank_group` handles the two tiles in each N-group, decrements their
  local tile-pending counters, and on last-local completion publishes to the
  remote source proxy's `node_tile_arrived` plus tile-reduce queue;
- the tile reducer claims tile queue items, currently uses wave 0 with vec4
  loads, peer-pulls/reduces the BN-wide tile, and sets the corresponding
  `node_ready_mask` bit;
- completion of the last required tile also publishes token-level
  `partial_ready` for compatibility with the still-token-granular downstream
  return/final path;
- the reducer role branches to the tile queue when
  `ready_granularity == "tile"`.

Compile and diagnostic gates achieved at TPR128 tile mode:

```text
cold full compile:               13.163 s, pass <=180 s
producer-only:
  rank_tile_pending nonzero:     0
  tile queue tail:               3584 (=128*28)
  node_tile_arrived:             3584
  stage2 error:                  0
gmm2_atomic_only + tile reducer:
  tile queue processed:          3584
  full node_ready_mask tokens:   128
  partial_ready tokens:          128
  stage2 error:                  0
```

These results prove producer tile publication and the first tile peer-pull
reducer implementation make progress for the diagnostic case. They do not yet
validate numerical output, multi-generation reuse, RAIL, or final combine.

The first full tile-mode run hung. The unresolved failure boundary is therefore
after tile reduction, in the compatibility transition to compact RAIL and/or
the token-level final path. Do not optimize the reducer vector width or add
mask-batched RAIL until this liveness issue is isolated.

Latest local debug instrumentation reads and reports:

```text
tile_partial_ready_planes
rank_return_counts
node_ready_mask_full_count
```

The concrete local debug edits are in:

```text
aiter/ops/flydsl/kernels/megamoe_tile/mega_moe_tile_a4w4.py
op_tests/multigpu_tests/bench_megamoe_tile_ep16_stage2_breakdown.py
```

They were not synchronized to node 46 because SSH authentication began failing
with `publickey`; node 50 remained reachable. Treat any remote result until
this is fixed as running the prior snapshot. After SSH recovery, synchronize
these two files to both nodes, verify hashes, and rerun a narrow full/return
diagnostic that snapshots partial planes and `rank_return_count` rather than a
long performance job.

Next liveness decision tree:

1. If all expected `partial_ready` planes are populated but
   `rank_return_counts` do not advance, inspect bx0 compact-return consumption,
   generation/plane addressing, and quiet/flush conditions.
2. If return counts advance but final does not complete, inspect remote
   `partial_ready`, final queue publication/claim, and node-local destination
   bypass semantics.
3. If only one plane is populated, inspect local/remote node identity and the
   last-tile compatibility publication target before implementing mask RAIL.
4. Once full compatibility mode passes, replace token-wide compatibility RAIL
   incrementally with uint64 mask batches 4/8/12 and tile final; retain a
   switchable compatibility path for bisection.

### Final paused state: tile pipeline WIP

A dedicated implementation/resume document now exists and is the primary
short-form handoff for this direction:

```text
scripts/megamoe_tile/STAGE2_TILE_PIPELINE_WIP_20260901.md
```

It records the complete objective, current code state, P1/P2/P3/P4 ordering,
measurements, invariants, and concrete source entry points. Preserve it with
this main historical checkpoint.

P2 progressed beyond the earlier RAIL/final hang:

1. A token-owner/group-counter compatibility implementation passed full smoke
   on both nodes with status 0. Formal TPR128 performance was:

   ```text
   rank-max mean: 3105.814 us
   rank-max P50:  3070.557 us
   rank-max P95:  3571.781 us
   all-rank mean: 3005.524 us
   ```

   This version was functionally progressing but remained a performance
   regression and was not promoted.

2. The current worktree replaces it with a lower-overhead
   group-arrival + token-owner scheme. This latest version passed full smoke on
   both nodes with status 0. Its exact cold compile completed in 14.387 seconds
   with Stage2 LDS 28,992 bytes. Formal performance for this newest version has
   **not** been rerun because the environment window ended; do not quote the
   older 3105.814-us result as the latest scheme's performance.

All final local static gates pass:

```text
py_compile:       pass
bash -n:          pass
git diff --check: pass
```

Pause/environment state:

- no benchmark, validation, torchrun, or profiling process from this task
  remains on either remote node;
- node 50 is at 0% GPU utilization;
- node 46 shows approximately 47--52% GPU utilization from an external task;
  it is not owned by this work and must not be terminated;
- the latest low-overhead P2 formal performance run is the first action after
  both nodes are available and source hashes are confirmed.

Resume order:

1. Read `STAGE2_TILE_PIPELINE_WIP_20260901.md` in full and preserve the dirty
   worktree/untracked artifacts.
2. Verify the exact current 12 tracked source/script files are synchronized to
   both nodes; do not assume the last external-occupancy interval preserved
   remote state.
3. Re-run a short full smoke for the current group-arrival + token-owner
   artifact, then formal TPR128 `warmup=5/iters=30/tail=20`, no ATT/no FP8.
4. Compare only against the matching W2 token-ready baseline. If the current
   P2 still regresses, quantify the remaining counter/atomic/queue overhead
   before implementing P3 mask RAIL.
5. If P2 wins and remains correct, proceed to P3 tile-mask RAIL/final and only
   later P4 `fp8_blockwise` as specified in the dedicated handoff.

Status: **paused cleanly at a compilable, smoke-passing P2 WIP; latest formal
performance pending**.

### 2026-09-02 resume audit

Both this historical checkpoint and
`STAGE2_TILE_PIPELINE_WIP_20260901.md` were reread in full after environment
recovery. The current worktree still matches the final paused handoff:

```text
branch / HEAD: dev/stage2_compute_pipeline_v1 / a69a16420
tracked modified files: 12
untracked handoff/scripts/traces: preserved
saved rejected stashes: 0..5 preserved
git diff --check: pass
```

The authoritative current state is the latest low-overhead P2
group-arrival + token-owner source, not the older per-tile-task reducer and not
the measured 3.106-ms token-owner/group-counter revision. The current source
has these verified gates from the prior environment:

```text
TPR128 full smoke: both nodes exit 0
cold compile:      14.387 s
Stage2 LDS:        28,992 bytes
formal perf:       pending for the exact current source
```

P3 uint64 mask RAIL/tile-final and P4 blockwise FP8 remain designs/plumbing,
not validated kernel implementations. `rail_quant_type=none` remains required.

Remote synchronization is not trusted across the environment transition. The
last recorded state says P1 and earlier P2 revisions ran remotely, while some
late diagnostics/source cleanup may be newer locally. Before any runtime test:

1. confirm node 46/50 availability and container execution (including the
   documented node46 crun symlink workaround if needed);
2. synchronize the exact current 12 tracked files to both remote worktrees;
3. compare local/node46/node50 SHA256 for every synced file;
4. confirm no unrelated GPU/process occupancy and never terminate external
   work;
5. run local/container static contracts before launching the unique current
   artifact.

The next promotion gate is deliberately narrow:

1. TPR128 current-source short full smoke, no ATT/no FP8;
2. if smoke passes, formal `warmup=5 / iters=30 / tail=20` performance;
3. compare against the matching W2 token-ready baseline, not against the older
   regressing P2 measurement;
4. if P2 is still slower, quantify group-arrival counters, token-owner queue,
   and per-tile peer-ready wait overhead before P3; if non-regressing, proceed
   to P3 mask RAIL/tile final with separate correctness gates.

Historical sections earlier in this file intentionally preserve the experiment
timeline and may describe superseded in-progress states. For resume decisions,
this audit plus `STAGE2_TILE_PIPELINE_WIP_20260901.md` are authoritative.

### 2026-09-02 P2 review and performance update

Additional tile-mode review gates are now enforced:

- `n_tile_group == 2` in tile mode;
- `hidden_tiles % 4 == 0` for the four-wave token reducer partition;
- arrival-counter overflow reports an explicit Stage2 error rather than
  wrapping or silently publishing readiness.

The per-tile-task family is rejected by formal measurements:

| Variant | TPR | Rank-max mean | Verdict |
|---|---:|---:|---|
| original per-tile tasks | 128 | 3150.698 us | reject |
| wave-independent per-tile tasks | 128 | 3091.406 us | reject |
| wave-independent per-tile tasks | 512 | 10092.620 us | reject; poor scaling |

The current authoritative implementation is the lower-overhead
group-arrival + token-owner design:

```text
rank_pending:                         route count
rank_tile_pending[source,n_group]:    arrival count for 14 two-tile groups
last group arrival:                   publish two tile-ready generations
group 0 only:                         node arrival + enqueue one token owner
token reducer:                        four waves, per-tile peer-ready waits
TPR128 queue size:                    128 token tasks
```

Its latest full smoke passed on both nodes. The newest exact cold compile is
14.146 seconds with LDS 28,992 bytes, superseding the earlier 14.387-second
record. The latest exact formal performance run remains pending.

Diagnostics/validators now interpret group arrival as
`arrival == route_count` and tile readiness as a generation value; do not apply
the older decrement-to-zero per-tile diagnostic semantics to this source.
Node50 contract tests pass (`113 passed`).

The current execution blocker is external occupancy on node46: a V4-Pro TP8
server and serving benchmark are using the node. They do not belong to this
task and must not be terminated. Wait for node46 release, then verify both-node
source hashes and run the exact current TPR128 formal performance comparison.

### Exact P2 formal gate and direction change

The exact current group-arrival + token-owner P2 was formally measured at
TPR128 against the matching W2 token-ready baseline:

| Path | Rank-max mean | P50 | P95 | All-rank mean |
|---|---:|---:|---:|---:|
| P2 group-arrival/token-owner | 2842.748 us | 2795.720 us | 3117.879 us | 2758.828 us |
| matched W2 token baseline | 1666.154 us | 1645.591 us | 1781.581 us | 1566.334 us |

P2 is 70.6% slower by rank-max mean and is rejected by the explicit promotion
gate. It must remain opt-in and must not become the default. Do not proceed to
P3 mask RAIL/final on top of this implementation.

The root cause is structural: each token-owner reducer still processes 28
tiles and performs peer-ready acquire/polling for every token/tile combination.
The 128-item token queue removed per-tile claims but did not remove the
token x tile x peer wait/setup cost.

The new optimization direction is **rank-group watermark**, using the existing
N-major W2 schedule:

```text
each rank/n_group completes all of its M work
  -> one release-published generation watermark
reducer claims one n_group
  -> waits once for the eight required peer-rank watermarks
  -> batch-reduces both tiles of that group for all active tokens
```

The active watermark path should eliminate per-route tile arrival,
per-token tile queueing, and per-token x tile polling while retaining overlap
between completed N-groups and later GMM work. The old P2 remains an opt-in
diagnostic/correctness reference.

Before implementation promotion, prove exact rank/group work counts, unique
watermark publication, generation/parity safety, tail-group correctness,
payload-before-watermark ordering, unique reducer group claim, and exactly-once
coverage of all active tokens. Re-run <=180-second compile, narrow producer and
batch-reducer diagnostics, TPR128 numerical correctness, and same-run full A/B
against W2 token-ready baseline. Only a non-regressing watermark result may
reopen P3.

### Planned reducer allocation experiment: peer2_stage_last

An additional opt-in reducer schedule will be evaluated alongside rank-group
watermark:

```text
node_reduce_work_schedule=peer2_stage_last
16 reducer CTAs -> 8 peer ranks x 2 CTAs per peer
4 waves/CTA may independently claim/copy peer tasks
```

Each peer has an independent work head. Its two CTAs split token/tile copy
tasks and write unique local BF16 staging rows indexed by peer,
parity/generation, token, and tile. Completion publishes an arrival counter;
the last required peer winner acquires the earlier stores, reads all eight
staging rows in fixed peer order, performs local FP32 reduction, writes the node
partial, and publishes the ready mask. This avoids direct atomic accumulation
into node partial and preserves the current fixed-peer numerical reduction
order.

This schedule is not presumed faster. It adds eight local staging writes and
eight local reads per reduced tile, plus arrival coordination. The current
`load_first` reducer already issues/overlaps eight LSA peer loads. Potential
benefit exists only when peer-specific CTA parallelism and ready-skew hiding
outweigh that local traffic and counter overhead.

Keep it compile-time opt-in and validate in this order: minimal staging/arrival
diagnostic, <=180-second compile, paired/arbitrary correctness, then same-run
full A/B against rank-group watermark and the W2 token baseline. Enforce unique
peer row writers, independent-head no-skip/no-duplicate behavior,
payload-before-arrival ordering, one last-arrival reducer, absent-peer handling,
generation-safe reuse, and exactly-once tile-mask publication.

### Ready flag/mask unit correction

The authoritative P2/P3 unit is one GMM N-group:

```text
hidden_tiles      = ceil(H / BN)
ready_group_tiles = n_tile_group
ready_group_count = ceil(hidden_tiles / ready_group_tiles)
```

For H7168/BN256/n_tile_group2, readiness uses 14 flags/mask bits, not 28.
Each bit represents two consecutive BN tiles / 512 hidden elements. Producers
publish only after all valid tiles in that group are complete; reducers and
final consumers use the same group index. A tail group with fewer valid tiles
must not wait for an absent tile, and the final partial tile must mask elements
beyond H. Bits above `ready_group_count` stay zero across all generations.
RAIL batch 4/8/12 means groups (8/16/24 tiles here); every performance report
must label the unit explicitly.

`atomic_only` reached 100% GPU utilization on both nodes but produced no result
after roughly 260 seconds. It was classified as a hang and terminated using
master port 29722. This mode has compute enabled with a zero-GMM epilogue,
reduction enabled, and the ordinary role path disabled; therefore the failure
still spans both producer publication and reducer consumption and cannot yet
be assigned to either side.

The discriminator now running is:

```text
mode/tag:        route_store_only / diag_routestoreonly_20260901
master port:     29723
node 46 session: 53029
node 50 session: 31272
```

`route_store_only` performs real GMM plus rank-local atomic accumulation and
publication while disabling reducer, RAIL, and final processing. If it passes,
the producer path can make progress independently and attention moves to the
consumer/reducer interaction. If it hangs, investigate producer epilogue and
publication before any communication analysis.

Planned CTA performance matrix:

| Reducer CTAs | Final rejoin CTAs |
|---:|---:|
| 8 | 4, 8, 16 |
| 12 | 4, 8, 16, 32 |
| 16 | 4, 8, 16, 32 |
| 24 | 8, 16, 32 |
| 32 | 8, 16, 32 |

The initial matrix has been screened. Final16 is retained; reducer8/16/24 long
runs are effectively tied within noise and all lose to baseline. Further broad
CTA sweeps are lower priority than multi-role ATT attribution.

## Safe resume commands

Static gate:

```bash
cd /home/zihuang/work/aiter-mega-tile-pr
python3 -m py_compile \
  aiter/ops/flydsl/kernels/megamoe_tile/comm_ops.py \
  aiter/ops/flydsl/kernels/megamoe_tile/stage2.py \
  aiter/ops/flydsl/kernels/megamoe_tile/stage2_abi.py \
  aiter/ops/flydsl/kernels/megamoe_tile/mega_moe_tile_a4w4.py \
  op_tests/multigpu_tests/megamoe_tile_rank_local_factory.py \
  op_tests/multigpu_tests/validate_megamoe_tile_route_store_ep16.py \
  op_tests/multigpu_tests/bench_megamoe_tile_ep16_stage2_breakdown.py \
  scripts/megamoe_tile/compile_ep16_stage2_fused.py
git diff --check
```

Historical Stage2 contract suite:

```bash
pytest -q \
  op_tests/test_megamoe_tile_rank_local_contract.py \
  op_tests/test_megamoe_tile_stage2_abi.py \
  op_tests/test_megamoe_tile_route_store_contract.py \
  op_tests/test_megamoe_tile_direct_tile_contract.py \
  op_tests/test_megamoe_tile_comm_probe_contract.py \
  op_tests/test_megamoe_tile_a4w4_public_contract.py
```

Always verify nodes 46 and 50 are idle before resident-grid tests. Keep new
scheduler/protocol modes opt-in until cold compile, paired/arbitrary/stress
correctness, and same-run performance all pass.

### Latest group-flag source boundary (authoritative tail)

The ready-group correction is merged locally. The active source derives
`ready_groups=ceil(hidden_tiles/n_tile_group)` (14 for the current shape), and
Stage2 init, publish, and wait indices are group-based. One group publishes one
generation flag after both valid constituent tiles complete. ABI, host,
harness, compile scripts, and contracts were updated. Node50 contracts pass
(`61 passed`).

Authoritative local SHA256 boundary:

```text
3468183379c644a619f05247ae91a0d8771ca36e78913f0567da3aeb8e38c919  mega_moe_tile_a4w4.py
dc7c37cd6de400346f55d8820085922f78ccf9b141cc76caf133bf00f2bf2953  stage1.py
2a4b0236dcdf958d9be87c983c005e4f282806cac4297cf7de63fbc75177eab1  stage1_abi.py
c0ed2cf9bdb3871301e0ef9d5e33694e5fa619e855762fb8b81ea0c9c37d835d  stage2.py
5c0d077f3fe6b6b44b8086c44bd0b155d3ee915640270b172fcc0319c47e5375  stage2_abi.py
be7d9095357aa1d539197724280d8dc1a037839c5044ec4f6c4dd2ea276d3039  bench_megamoe_tile_ep16_stage2_breakdown.py
5204e25e5f734f243371a20846e0ba7a96284ec7b02eae737bd4bfdd9b512155  bench_megamoe_tile_ep16_two_kernel.py
c2f8ab4c8717a9072afd5b03c6f505478482edf3e96697666191da8ebd34cbf6  test_megamoe_tile_a4w4_public_contract.py
63e719aacf480342a2558613bcd71f2501813e234fccd7ffdfe371d08a6f294c  compile_ep16_stage2_fused.py
68801a47942c786d7d6542d6d7b28102392fd22cf28bb6c38e4b8dd8636df001  compile_ep16_two_kernel.py
ed21ae9ab8ca7b3c79a803772fdb54e13f428925a3b9e6997a65c99a844b32e9  profile_rank0_worker.sh
2d169c74e05610848a8b10a7ce42fa73e3e556b2f00d20d788dc6f414d86996f  run_stage2_breakdown_ep16.sh
tracked diff 9137bd8158c63c8387f0eccb5250b11ba9fcfb42a78ee8c32d9c4f283a042777
```

Node46 remains externally occupied, so this exact source has no two-node
smoke/formal result. On release: sync all 12 files, verify hashes across local/
46/50, rerun static/contracts, cold compile under <=180 s, then producer/reducer
diagnostics and full smoke. Earlier compile/smoke results belong to a preceding
source boundary and are not proof for this version.

### Local branch snapshot request

Save the current group-ready WIP on a new local branch:

```text
dev/stage2_tile_group_ready_wip_20260902
```

Create a local commit only; do not push it to any remote. The snapshot must
include:

- all currently tracked implementation/test/script changes;
- `scripts/megamoe_tile/STAGE2_CU_OPT_CHECKPOINT_20260901.md`;
- `scripts/megamoe_tile/STAGE2_TILE_PIPELINE_WIP_20260901.md`;
- the untracked reusable helper scripts under `scripts/megamoe_tile/` that
  belong to this work (`precompile_stage2_rank_local_baseline.sh` and
  `run_stage2_cu_pull_profile_ep16.sh`).

Do **not** add or commit `trace_data/`; it remains a local profiling artifact.
Before committing, run `git diff --check` and inspect the staged file list to
prove no unrelated files or traces are included. After checkout/commit, append
the resulting local commit hash here and retain the branch without pushing.
# Local implementation snapshot (2026-09-02)

The complete tile-group WIP implementation was preserved on local branch
`dev/stage2_tile_group_ready_wip_20260902` in commit
`85cd0e78359fe8d59befc1eb7d46ca71c2e2ebda`.  The follow-up documentation
commit only records this immutable implementation hash.  Nothing was pushed.
`trace_data/` remains intentionally untracked.
