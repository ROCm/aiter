---
name: validate-kernel-pr
description: Reproducible validation executor for kernel PRs. Applies an explicit base-to-head patch in an isolated worktree, runs it on a verified-idle GPU, compares the same targets against base, policy-checks the test diff, and emits a head-bound validation_report.json. Missing environment evidence is INCONCLUSIVE, never PASS.
argument-hint: --repo <worktree> --target <script file or pytest target>
---

# validate-kernel-pr

`review-pr` never builds and never runs. It is a static reviewer, and a good one — but three
failure modes are invisible to it, and this skill exists for exactly those three:

1. **The PR's own tests pass while the kernel is wrong.** A suite whose non-aligned shapes are
   commented out reports green on an out-of-bounds tail store.
2. **A green suite that cannot fail.** Loosening a comparison tolerance leaves every test
   passing and the kernel unguarded.
3. **Defects that only exist at runtime.** LDS over-allocation on one arch, an accuracy gate
   failing against the reference, a JIT path that no-ops on cache miss.

Output is `validation_report.json`: deterministic execution evidence kept separate from
`review-pr`'s advisory judgement. A review may consume it only when `repo.head` matches the exact
PR head; a review written without one must mark validation `NOT RUN`.

---

## Invocation

The caller supplies a clean base checkout, the base-to-head patch, the exact head OID, and the
test target; this script does not fetch PRs itself (see
[Not implemented yet](#not-implemented-yet)).

```bash
# Example for a FlyDSL softmax PR.
REPO=ROCm/FlyDSL
PR="${PR:?set PR to the open softmax PR number}"

# 1. pin the PR identity and put its base in an isolated worktree
BASE_REF=$(gh pr view "$PR" --repo "$REPO" --json baseRefName --jq .baseRefName)
BASE_REF_PATH=$(python3 -c \
  'import sys,urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' \
  "$BASE_REF")
BASE=$(gh api "repos/$REPO/branches/$BASE_REF_PATH" --jq .commit.sha)
HEAD=$(gh pr view "$PR" --repo "$REPO" --json headRefOid --jq .headRefOid)
git worktree add --detach "/tmp/pr-$PR" "$BASE"
gh pr diff "$PR" --repo "$REPO" > "/tmp/pr-$PR.patch"

# 2. validate base and head under the same runner and GPU lock
.claude/skills/validate-kernel-pr/validate_pr.sh \
    --repo "/tmp/pr-$PR" \
    --patch "/tmp/pr-$PR.patch" \
    --head-sha "$HEAD" \
    --target tests/kernels/test_softmax.py \
    --expected-route kernels.softmax_kernel:build_softmax_module \
    --shape-vars M,N,dtype_str \
    --shape-env ROCDSL_SOFTMAX_SHAPES \
    --grid "64,2048,f32;64,2000,f32" \
    --tol-table "f32=1e-5,f16=2e-3,bf16=1e-2" \
    --out validation_report.json
```

For a local candidate with no remote head, omit `--head-sha`. The report then records
`repo.head: null`; it remains useful locally but `review-pr` will reject it as PR evidence.

| flag | meaning |
|---|---|
| `--repo` | worktree to validate (required) |
| `--target` | script file or pytest node/file the PR ships (`--tests` remains an alias) |
| `--patch` | patch to apply first; conflict is a blocker |
| `--head-sha` | exact remote PR head represented by the patch |
| `--expected-route` | exact `module:function` route the validator-owned profiler must observe |
| `--shape-vars` | comma-separated local names captured from each route call, in grid order |
| `--shape-env` `--grid` | env var and shape list for the S1-owned grid |
| `--tol-table` | reference tolerances, e.g. `f32=1e-5,f16=2e-3,bf16=1e-2` |
| `--label` `--out` | run name and report path (default `./validation_report.json`) |

Environment knobs: `PYLIB` (runtime modules outside the checkout), `PYTHON_BIN` (one interpreter
used for pytest and script targets), `PICKER` (override the shipped `pick-idle-gpu.py`), and
`TIMEOUT` (per-target budget, default 1800s). The executor
overrides `AITER_JIT_DIR` with separate fresh base/head directories and sets
`PYTHONDONTWRITEBYTECODE=1`, so repository JIT output cannot cross phases or dirty the worktree.

---

## Stages

Each stage writes its own status into the report. A stage that cannot run says `skip` with a
reason — it never reports `pass` for work it did not do.

### 1 — `merge_sim`

Apply the PR head on top of the current base. A conflict is a blocker and short-circuits: no
number produced downstream would describe the merged code. Known collision surfaces worth a
second look because they are edited by many PRs at once: tuning CSVs (duplicate shape rows),
`csrc/include/rocm_ops.hpp`, `aiter/jit/optCompilerConfig.json`.

The supplied worktree must be clean. The report records the base commit, patch SHA-256, and the
caller-supplied head OID. A direct head checkout without a patch can run diagnostics, but cannot
prove mergeability or base attribution and therefore cannot produce `PASS`.

The patch is reverted when the process exits, including on interrupt and on every degraded path,
so the worktree is handed back in the state it was supplied. Consecutive runs in the same worktree
are therefore supported; a run that left the patch applied would make the next one report
`not isolated-clean` and blame the caller.

### 2 — `gpu_claim`

Claim a GPU over a **sampling window**, not one instantaneous reading, and acquire a non-blocking
lock immediately after selection. Hold that file descriptor for the whole run:

```bash
PICK=$(python3 .claude/skills/validate-kernel-pr/pick-idle-gpu.py \
    --samples 10 --interval 1 --quiet)
flock /tmp/gpu-$PICK.lock <command>
```

The report records host, HIP index, matching AMD SMI index, BDF, market name, architecture, and
GFX activity before the run. `pick-idle-gpu.py` emits the **translated HIP index**; the validator
maps it back through AMD SMI enumeration instead of incorrectly using it as an AMD SMI index.

`amdsmi_get_gpu_activity` is not available everywhere — some driver and amd-smi combinations fail
it outright or report `N/A` while enumeration, BDF, ASIC and VRAM queries all work. Activity is
therefore treated as optional, and `gpu_claim.idleness_basis` names the evidence the claim rests
on: `activity+vram` when busy percentages were measured, `vram-only` when only resident VRAM
separated the devices. In the `vram-only` case `gfx_activity_before_pct` is `null`, which means
unknown, not zero — an unavailable metric is never reported as an observed idle GPU.

If no GPU stays idle, `gpu_claim` is `skip`, `degraded_mode` is `NO_GPU`, both correctness stages
are `skip`, and the verdict is `INCONCLUSIVE`. The script performs no architecture-specific
compile in this branch, so it does not call the result `compile-only`. That skip distinguishes two
different facts: GPUs present but none idle is an environment fact, whereas AMD SMI being
unqueryable is a portability gap in the validator and says nothing about the GPUs.

### 3 — `runtime_compat`

Does the repository's own package import from the supplied checkout against the runtime that is
actually installed? The probe is repository-aware: Aiter resolves `aiter` from the checkout;
FlyDSL resolves the pinned package from `PYLIB` (when supplied) and compares its version with the
checkout's `python/flydsl`. This keeps compiled `_mlir` bindings available without pretending an
unrelated FlyDSL install validates an Aiter checkout. A pinned prebuilt runtime can drift behind
the tree, and the resulting `ImportError` looks exactly like a defect in the PR. A mismatch is an
environment fact: `runtime_compat` and correctness are skipped, the verdict is `INCONCLUSIVE`,
and nothing is attributed to the author.

The report records the Python executable/version, resolved package path/version, and SHA-256
identities for native libraries loaded by the runtime probe.
If a FlyDSL PR changes Python, C++/MLIR bindings, headers, CMake, or packaging inputs, a prebuilt
`PYLIB` is not accepted: trusted build-system provenance is not implemented, and caller-authored
metadata cannot prove which source produced a binary. Such runs return `INCONCLUSIVE` instead of
testing a stale package.

This matters most for FlyDSL kernels: the Python kernels import symbols from a compiled runtime,
so "one fresh container per PR" would mean rebuilding MLIR/LLVM per PR. The workable shape is a
pinned prebuilt image plus this compatibility gate.

### 4 — `test_policy` — run **before** the suite

A suite that cannot fail is worse than no suite, because it produces a green report. Two checks:

- **Tolerance, compared head-vs-base.** Repos legitimately differ per kernel; the question is
  whether *this change* loosened what was there. A test-only widening is a deterministic blocker.
  If kernel code changed too, the widening is `NEEDS_WORK` pending numerical justification rather
  than a false deterministic block.
- **Commented-out shape rows, compared head-vs-base.** Existing rows are recorded as coverage
  context; only rows newly disabled by the change produce `NEEDS_WORK`. The independent grid
  remains visible either way.

### 5 — `correctness` — the repo's tests, then a grid the repo does not run

Runner selection is structural, not assumed:

- an explicit `path::node`, or a file defining `test*`/`Test*`, uses pytest;
- otherwise a file with an `if __name__ == "__main__"` guard runs as `python <file>`;
- a file with neither is `skip`, never a test failure.

The report records `test_selection.runner` and `runner_reason`. Script targets can establish that
their real repository entry point succeeds or fails, but cannot currently use the pytest route
profiler, so even a successful script run tops out at `INCONCLUSIVE`.

Both, and they are reported separately, because the interesting case is when they disagree.
Pytest runs emit JUnit XML and a zero-executed/all-skipped target is `skip`, never `pass`.
Script runs record their process exit and whether they produced output under an explicit
`script-*` evidence basis; those counters are not described as JUnit.

For a patch run, the validator reverses the exact patch to create the baseline, verifies that the
worktree is clean, runs both targets under base-only caches, and reapplies the patch before a
head run with separate caches. This removes new files too; a PR-added failing test is therefore
`target-not-present` on base, not falsely classified as a pre-existing failure. Any worktree
artifact, reverse/reapply failure, or cache-isolation failure aborts the head run and produces
`INCONCLUSIVE`.

The S1-owned grid must cover three classes the PR's own tests routinely miss:

| class | why |
|---|---|
| non-toy | `M=1` / `M=16` only is the standard agent-generated test |
| boundary / odd | odd N, N not a multiple of the tile — where tail masks fail |
| long-context / large M | where 32-bit index arithmetic wraps |

The grid stage runs only when the selected target source references the configured
`--shape-env`; otherwise it is `skip` and the verdict is `INCONCLUSIVE`. This is a positive
control against reporting the same default test run twice under different stage names.

When the kernel exposes no shape override, the report says `repo-default-only` rather than
claiming coverage it does not have.

### 6 — `execution_receipt`

The validator loads its own pytest profiling plugin before test collection. The caller names an
exact Python `module:function` route and the route's shape-local variable names; the plugin
records actual calls and writes:

```json
{
  "schema_version": 1,
  "route": "aiter.ops.flydsl.kernels.moe_2stage_a16wmix:flydsl_a16w4_gemm1",
  "kernel_symbols": ["aiter.ops.flydsl.kernels.moe_2stage_a16wmix:flydsl_a16w4_gemm1"],
  "executed_shapes": ["1,3584,384", "128,3584,384"]
}
```

`PASS` requires the observed route to equal `--expected-route`, at least one observed route
symbol, and every shape named by `--grid`. The tested PR cannot obtain credit merely by writing
its own receipt; `validate-kernel-pr.validation_probe` owns the receipt producer.

### 7 — `index_width_scan` (informational)

Runs `scan_index_width.py` over the diff and records the count of index×stride multiplies that
carry no 64-bit widening. Candidates, not verdicts — the reviewer judges each. See
[Why this stage exists](#why-the-index-width-scan-is-a-separate-stage).

### 8 — verdict

`BLOCK` if a reproducible candidate defect fired, `NEEDS_WORK` if a deterministic policy concern
fired, `INCONCLUSIVE` if any required stage did not complete, else `PASS`. `PASS` therefore means
the merge simulation, GPU claim, repo-aware runtime probe, policy comparison, baseline control,
both correctness targets, execution receipt, and index scan all ran.

Process exit codes match the verdict: `PASS=0`, `BLOCK/NEEDS_WORK=1`, and `INCONCLUSIVE=2`.

---

## Honesty rules the report enforces

These are fields, not prose, so a report cannot overclaim by omission:

- **`arch_coverage`** — per architecture, `runtime`, `compile-only`, or `not-covered`.
  A GPU claim alone earns no runtime coverage; `runtime` is added only after a selected head
  correctness test is collected and executed with that device visible. `compile-only` requires
  an actual architecture-specific compile.
- **`isolation`** — the real level. Where no container runtime is available it is
  `git-worktree + private caches`, and the report says `container: false`.
- **`degraded_mode`** — `NO_GPU` when no device was claimable; required stages then make the
  verdict `INCONCLUSIVE`.
- **Every declared stage exists.** A stage that did not run is an object with `status: skip` and
  a reason; it never disappears and never becomes a JSON string.
- **`test_selection`** — the exact target, selected runner, and independent grid. A
  verdict applies only to those named inputs.
- **`runtime_identity`** — resolved package, interpreter, source SHA, and native artifact hashes.
- **`execution_receipt`** — observed route, kernel symbols, and exact shapes emitted by the test.
- **Every perf number keeps its provenance.** A number in the PR description that does not
  reproduce is tagged `[unreproducible]`; it is not quietly dropped.

---

## Why the index-width scan is a separate stage

Rule `D9` in `review-pr` covers 32-bit overflow in pointer arithmetic. Its original trigger was
a list of variable names (`token_id`, `seq_start`, `batch_offset`, `total_tokens`). Real defects
used other names, so the rule stayed silent:

| defect | expression | consequence |
|---|---|---|
| aiter#1674 | `stride_out_batch` not `tl.int64` | output offset wraps at large MTP batch; tail rows keep stale sparse-KV indices |
| aiter#1674 | `block_id * stride` with no `.to(int64)` | every block past INT32_MAX returns logits of exactly 0.0, silently |
| aiter#3541 | `ArithValue(physical_block) * stride` | wraps on a ~150M-row KV pool; the wrapped offset still lands inside the allocation, so no fault |

The scan is D9's structural pre-filter instead — an index-shaped value multiplied by a
stride-shaped value on a line with no widening — and is deliberately noisy in the safe
direction. Its candidate count is deterministic and informational; production scale still
determines whether a candidate is a defect.

```bash
.claude/skills/validate-kernel-pr/scan_index_width.py ROCm/aiter 1674
.claude/skills/validate-kernel-pr/scan_index_width.py --diff /tmp/pr.diff
```

---

## Not implemented yet

Deliberately absent rather than half-built — everything shipped here has been observed failing on
a seeded defect, and these have not been:

- **PR fetch orchestration.** There is no `--pr N`. The caller creates the worktree, as above.
  Choosing the right `--target` from a diff is the unsolved part; an irrelevant target can
  still produce `PASS`. The report names the target so a reviewer can reject that evidence, but
  the executor cannot decide relevance itself.
- **External grid adapters.** A script-only PR target may lack a shape hook. The validator does
  not yet accept an independently hashed `--extra-target`, because that harness must be bound
  without changing the PR diff hash or live-base identity. Such runs remain `INCONCLUSIVE`.
- **Cross-architecture compilation.** `arch_coverage: compile-only` is reserved for a future
  stage that actually invokes an architecture-specific compiler. No-GPU mode does not claim it.
- **`perf` and `claims` stages.** The schema reserves both — median-of-N against a baseline on the
  same locked GPU, and reproducing the numbers in the PR description — and the script emits
  neither. A report today carries no performance evidence, and a review must not read the absence
  of a `perf` stage as "no regression".
- **Adversarial route attestation.** The validator-owned profiler prevents accidental and
  worktree-shadowed receipts, but arbitrary Python running in the same process can still spoof a
  matching frame. A hostile-code gate needs an out-of-process HIP/rocprof trace.

## What this skill does not do

- It does not replace `review-pr`. It produces evidence; the judgement stays there.
- It does not write findings about design, style, or API shape.
- It does not perform a merge or publish a decision. A `BLOCK` is reproducible executor evidence;
  `review-pr` keeps its separate advisory verdict.
- It does not validate an architecture it has no device for.

---

## Regression assets

Fast synthetic tests cover the report contract, no-GPU behavior, repo-aware runtime probing,
new-file baseline attribution, tolerance widening, missing pytest, and deterministic scanner
counts:

```bash
python -m pytest .claude/skills/validate-kernel-pr/tests/test_validator.py -q
```

The original FlyDSL softmax evidence is committed under `tests/mutants/`, pinned to
`ROCm/FlyDSL@421935cc6f09fd9b27d5d5ae52e0960e18834bd5`. It includes a behavior-neutral control
and the three distinct mutants from the PR table. Replay it on a checkout-matched runtime and a
verified-idle GPU:

```bash
PYLIB=/path/to/flydsl-runtime \
  bash .claude/skills/validate-kernel-pr/tests/replay_mutants.sh /path/to/FlyDSL
```

The replay fails unless the control is `PASS`, the tail-mask and vector-index mutants block in
`correctness`, and the tolerance mutant blocks in `test_policy`.

---

## Adding a stage

A new stage must be able to **fail on a seeded defect**. Before adding one, seed the defect it is
meant to catch, confirm the stage goes red and the clean baseline stays green, and record both in
the PR. A stage that has never been observed failing is not a check — it is decoration.
