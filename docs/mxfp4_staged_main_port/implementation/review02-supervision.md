# Ticket 02 staged supervision review

Read-only review of fixed diff `60cbce1f6...2c902bf08` in
`/root/workspace/tmp_dev/aiter-mxfp4-02`, applying karpathy-guidelines against
the current spec, ticket 02 and acceptance. Active 02.4 edits were excluded.
No tests, workers, GPU tasks or kernels were launched; no repository or tracker
files were changed.

## Finding: required exit-before-read ordering is not established

`op_tests/tuning_tests/test_mxfp4_staged_supervision.py` coordinates
result-before-exit with an event and cross-device overlap with a barrier.
Its fast-result/fatal tests can race into either arrival/exit ordering, but
none deterministically forces the worker to exit before the parent reads its
Pipe. `acceptance.md:17` explicitly requires exit-before-read coverage.

Minimal correction: add a CPU-only synchronized worker/wait wrapper that
confirms the worker sentinel before exposing readiness to the supervisor, then
assert the queued result is retained, exit/join/close happen before GPU reuse,
and the subsequent shape continues. This is a coverage requirement, not an
observed implementation failure. Sent to root and implement_02 immediately.

## Code assessment

No concrete functional defect found in the reviewed implementation. One
worker-private one-way Pipe is owned per fresh shape, including single-slot
runs. The parent closes its writer, separates result arrival from process
exit, receives queued output on exit, joins and inspects exitcode, closes the
Process and only then releases the GPU. Clean EOF and nonzero exits return
shape failure; a result preceding nonzero exit cannot win. Returned Python
shape errors are converted at the existing worker boundary. Outstanding
Process/Pipe handles are bounded by active slots.

Start/receive/deserialization/wait/cancellation errors stop new launches and
finally cleans still-owned workers and receive endpoints. Cleanup catches its
own failures so the original supervisor exception is preserved. Empty work
does not spawn, normalization caps slots by visible devices and rows, staged
preflight selects the first participating visible logical devices, and
prune/full keep their existing paths. SIGALRM restores the prior handler and
timer and documents its soft timeout contract without a parent/shape deadline.

Evidence inspected: `03-supervision-checks.txt` reports 139 controlled combined
checks passing (including 19 CPU supervision and 5 timeout cases); the source
and behavioral fixtures were reviewed, not rerun. `git diff --check
60cbce1f6...2c902bf08` passed. These checks provide no real GPU evidence and this
review does not close ticket 02.
