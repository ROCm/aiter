# Default-environment staged search implementation

Authoritative ticket: `.scratch/mxfp4-staged-main-port/issues/02-default-staged-search.md`.
Implementation starts from integration `119212da2e`; fixed main base remains
`882ea79dd7cc7091f859a5bedac914523ac77bbd`. Tracker state is maintained only in
the integration worktree.

## CLI and candidate planning

The existing `--mxfp4-search-mode` accepts `prune`, `full` and `staged`, with
`prune` retained as default. Explicit positive `--mxfp4-screen-topk` requires
explicit staged mode; omission selects 2. Routing top-k remains a shape field.
The staged plan projects `_candidate_rows(row, full_search=False)` and retains
complete kernel names, first-seen stage order and pair order. It introduces no
candidate registry, native GEMM2 aliases, or re-tile pairs. Input contracts are
validated before grouping. GEMM1 contexts are BM; pipeline/GEMM2 contexts are
BM plus atomic versus non-atomic accumulation. Reduce/scatter share the latter.

Focused checks use the real imported tuner with offline synthetic import
metadata and empty ROCm/HIP/CUDA visibility. Tests forbid GPU tensor preparation,
real measurements and real shape launches. The I384 fixture calls the real
shared static enumerator; planning itself performs no pair execution. Import
metadata is synthetic and supplies no runtime-device acceptance evidence.

TDD evidence: the first CLI test failed on `staged` as an invalid choice before
the selector change; its minimal implementation passed. The next shape-planning
test failed because `_staged_candidate_plan` was absent; projection then passed
against real pruned I384 rows and preserved full/prune candidate boundaries.

Reproduce controlled checks with the dedicated environment, external core JIT
cache, and all visibility variables empty:

```bash
ROCR_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' CUDA_VISIBLE_DEVICES='' \
GPU_ARCHS=gfx950 CU_NUM=256 AITER_USE_SYSTEM_TRITON=1 \
CK_DIR=/root/workspace/tmp_dev/aiter-mxfp4-integration/3rdparty/composable_kernel \
AITER_JIT_DIR=/root/workspace/tmp_dev/mxfp4-port-notes/isolated-01-jit \
PYTHONPATH="$PWD" /root/workspace/tmp_dev/mxfp4-port-env/bin/python -m pytest \
op_tests/tuning_tests/test_mxfp4_staged_search.py \
op_tests/tuning_tests/test_mxfp4_tuner_lifecycle.py -q
```

The environment's existing system Triton3.4 warning is retained; these controlled
checks execute no Triton kernel. GPU, numerical, production and AOT evidence are
recorded separately after the tested policy and supervisor are connected.

## Controlled screening and pair selection

Shape-level staged policy selects deterministic admitted atomic reference
consumers and G1 producers, retries returned reference failures in the same
admitted BM set, and keeps bootstrap untimed. G1 ranks are shared across
accumulation groups. Screening retains at most K positive finite scores in
stable admitted order. Selected products use ordinary pair validation; only
that validation supplies the canonical winner timing/error fields.

A missing reference, empty successful stage ranking or no valid shortlisted
pair enters exhaustive group-local validation over the original pruned pairs.
Every returned pair outcome is stored as scalar time/failure text and its full
pair identity; expansion never repeats it. Screening failures remain eligible
for pair fallback, and fallback continues to its last admitted pair before
selecting a winner. Cache and rankings belong to one shape invocation.

TDD evidence: the first controlled staged-shape test observed nine normal pair
attempts when K1 required one ranked product; after the policy implementation it
observed only that product and its separately supplied end-to-end timing. The
focused fixtures cover reference/screen failures, invalid shortlist expansion,
nonfinite outcomes, returned soft timeout/exception outcomes, stable ties and
shape-context reset. The fake outcomes verify policy independently of GPU
compilation and numerical behavior; those remain part of later real acceptance.

## Staged shape ownership and timeout

Staged retains main's parallelism normalization/caps but assigns the first
participating visible devices, even with one slot. Every shape owns a fresh
process and private one-way result Pipe. A result is retained until the worker
exits, joins and closes; a nonzero exit makes the shape fail even if a result
arrived first. Completed handles close before GPU reassignment. Parent/IPC
failures stop new launches and clean owned processes/endpoints, preserving the
original exception. The existing direct/Queue prune/full schedules remain.

The operation SIGALRM timer restores its previous handler and previous timer.
The help reports best-effort recovery and the absence of a parent/shape hard
deadline; a still-live GPU operation that never returns may wait indefinitely.
The first timer test failed because the existing timer was lost before the
owned `setitimer` implementation. The first supervisor test failed because the
private-Pipe entry point was absent before fresh-worker implementation.

CPU supervisor fixtures allow only explicit module-level fake worker targets,
with GPU visibility empty and all real GPU/measurement functions forbidden.
Events/barriers coordinate result-before-exit and cross-device overlap; no fixed
sleep establishes execution ordering. Fake multi-device coverage does not
constitute physical multi-GPU measurement evidence.
