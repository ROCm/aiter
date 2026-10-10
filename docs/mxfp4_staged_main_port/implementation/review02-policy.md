# Ticket 02 planning and policy review

Reviewed fixed diff `119212da2e...60cbce1f6` in
`/root/workspace/tmp_dev/aiter-mxfp4-02`, covering commits `65a4eba35` and
`60cbce1f6`. Applied karpathy-guidelines. This is a focused review of completed
02.1/02.2 planning and controlled search policy, not final ticket acceptance.
Uncommitted 02.3 supervision, future real screening/bootstrap implementations,
GPU, production and AOT were excluded. No GPU process or kernel was launched.

## Result

No new actionable defect found in the reviewed planning/reference-choice/
ranking/fallback/cache scope. The existing CLI remains a single selector with
prune default and positive explicit screening top-K restricted to staged.
Planning uses the shared pruned enumerator, verifies the fixed shape and
derived BM/dtype/activation contracts, retains complete stage identities and
original pair order, and keeps atomic/non-atomic contexts separate while
sharing non-atomic reduce/scatter ranking and BM-specific G1 ranking.

Reference choices and returned-failure retries remain in the admitted BM set.
Bootstrap contributes no validation result. Missing references, empty stage
shortlists and entirely invalid shortlist products exhaust the group's
admitted normal pairs, including screening-failed candidates. Expansion
continues through the final remaining pair, does not repeat returned pair
identities, and keeps cached failure records scalar. Existing normal pair
execution still owns the finite numerical gate and canonical winner timing.

Evidence inspected: `docs/mxfp4_staged_main_port/02/01-planning-checks.txt`,
`02-policy-checks.txt` (115 controlled planning/policy/lifecycle checks passed),
the corresponding shape-level behavioral fixtures, current spec, ticket 02
and pipeline grouping audit. `git diff --check 119212da2...60cbce1f6` passed.
These fake outcomes establish search decisions, not real GPU correctness.

## Already assigned follow-up

The committed 02.2 `_staged_timeout` restores the prior handler but cancels an
existing SIGALRM timer. Root confirmed timer ownership/restoration belongs to
the active 02.3 implementation, which is adding getitimer/setitimer restoration
and `test_mxfp4_staged_timeout.py`. This note does not duplicate that work or
claim the unreviewed correction passed.

## Acceptance controller correction reread

Reread the updated integration `run_gpu_task.py` and its five offline tests.
Unavailable AMD-SMI occupancy now uses the remaining idle evidence; child
tasks own a new session; SIGTERM enters cancellation cleanup; termination
targets the task's process group including descendants; interruption records
the exit outcome. No additional controller finding in the requested scope.
Evidence: `/root/workspace/tmp_dev/mxfp4-port-notes/controller-final.txt`
(5 passed; harmless CPU descendant test). No controller or tracker edits.
