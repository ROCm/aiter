---
name: validate-kernel-pr
description: Reproducible validation executor for kernel PRs. Applies an explicit base-to-head patch in an isolated worktree, runs it on a verified-idle GPU, compares the same targets against base, policy-checks the test diff, and emits a head-bound validation_report.json. Missing environment evidence is INCONCLUSIVE, never PASS.
argument-hint: --repo <worktree> --tests <pytest target>
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
# 1. pin the PR identity and put its base in an isolated worktree
BASE=$(gh pr view 4394 --repo ROCm/aiter --json baseRefOid --jq .baseRefOid)
HEAD=$(gh pr view 4394 --repo ROCm/aiter --json headRefOid --jq .headRefOid)
git worktree add --detach /tmp/pr-4394 "$BASE"
gh pr diff 4394 --repo ROCm/aiter > /tmp/pr-4394.patch

# 2. validate base and head under the same runner and GPU lock
.claude/skills/validate-kernel-pr/validate_pr.sh \
    --repo /tmp/pr-4394 \
    --patch /tmp/pr-4394.patch \
    --head-sha "$HEAD" \
    --tests op_tests/test_moe_2stage.py \
    --shape-env AITER_TEST_SHAPES \
    --grid "1,4096,f32;7,2000,bf16;16384,8192,bf16" \
    --tol-table "f32=1e-5,f16=2e-3,bf16=1e-2" \
    --out validation_report.json
```

For a local candidate with no remote head, omit `--head-sha`. The report then records
`repo.head: null`; it remains useful locally but `review-pr` will reject it as PR evidence.

| flag | meaning |
|---|---|
| `--repo` | worktree to validate (required) |
| `--tests` | pytest target the PR ships |
| `--patch` | patch to apply first; conflict is a blocker |
| `--head-sha` | exact remote PR head represented by the patch |
| `--shape-env` `--grid` | env var and shape list for the S1-owned grid |
| `--tol-table` | reference tolerances, e.g. `f32=1e-5,f16=2e-3,bf16=1e-2` |
| `--label` `--out` | run name and report path (default `./validation_report.json`) |

Environment knobs: `PYLIB` (runtime modules outside the checkout), `PICKER` (path to
`pick-idle-gpu.py`), `TIMEOUT` (per-pytest budget, default 1800s).

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

### 2 — `gpu_claim`

Claim a GPU over a **sampling window**, not one instantaneous reading, and acquire a non-blocking
lock immediately after selection. Hold that file descriptor for the whole run:

```bash
PICK=$(pick-idle-gpu.py --samples 10 --interval 1 --quiet)
flock /tmp/gpu-$PICK.lock <command>
```

The report records host, HIP index, matching AMD SMI index, BDF, market name, architecture, and
GFX activity before the run. `pick-idle-gpu.py` emits the **translated HIP index**; the validator
maps it back through AMD SMI enumeration instead of incorrectly using it as an AMD SMI index.

If no GPU stays idle, `gpu_claim` is `skip`, `degraded_mode` is `NO_GPU`, both correctness stages
are `skip`, and the verdict is `INCONCLUSIVE`. The script performs no architecture-specific
compile in this branch, so it does not call the result `compile-only`.

### 3 — `runtime_compat`

Does the repository's own package import from the supplied checkout against the runtime that is
actually installed? The probe is repository-aware: Aiter resolves `aiter` from the checkout;
FlyDSL resolves the pinned package from `PYLIB` (when supplied) and compares its version with the
checkout's `python/flydsl`. This keeps compiled `_mlir` bindings available without pretending an
unrelated FlyDSL install validates an Aiter checkout. A pinned prebuilt runtime can drift behind
the tree, and the resulting `ImportError` looks exactly like a defect in the PR. A mismatch is an
environment fact: `runtime_compat` and correctness are skipped, the verdict is `INCONCLUSIVE`,
and nothing is attributed to the author.

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

Both, and they are reported separately, because the interesting case is when they disagree.

For a patch run, the validator reverses the exact patch to create the baseline, verifies that the
worktree is clean, runs both targets, and reapplies the patch before the head run. This removes
new files too; a PR-added failing test is therefore `target-not-present` on base, not falsely
classified as a pre-existing failure. Any reverse/reapply failure aborts attribution and produces
`INCONCLUSIVE`.

The S1-owned grid must cover three classes the PR's own tests routinely miss:

| class | why |
|---|---|
| non-toy | `M=1` / `M=16` only is the standard agent-generated test |
| boundary / odd | odd N, N not a multiple of the tile — where tail masks fail |
| long-context / large M | where 32-bit index arithmetic wraps |

When the kernel exposes no shape override, the report says `repo-default-only` rather than
claiming coverage it does not have.

### 6 — `index_width_scan` (informational)

Runs `scan_index_width.py` over the diff and records the count of index×stride multiplies that
carry no 64-bit widening. Candidates, not verdicts — the reviewer judges each. See
[Why this stage exists](#why-the-index-width-scan-is-a-separate-stage).

### 7 — verdict

`BLOCK` if a reproducible candidate defect fired, `NEEDS_WORK` if a deterministic policy concern
fired, `INCONCLUSIVE` if any required stage did not complete, else `PASS`. `PASS` therefore means
the merge simulation, GPU claim, repo-aware runtime probe, policy comparison, baseline control,
both correctness targets, and index scan all ran.

---

## Honesty rules the report enforces

These are fields, not prose, so a report cannot overclaim by omission:

- **`arch_coverage`** — per architecture, `runtime`, `compile-only`, or `not-covered`.
  `compile-only` requires an actual architecture-specific compile; a gfx950 host does not earn
  gfx942 coverage merely by lacking a gfx942 device.
- **`isolation`** — the real level. Where no container runtime is available it is
  `git-worktree + private caches`, and the report says `container: false`.
- **`degraded_mode`** — `NO_GPU` when no device was claimable; required stages then make the
  verdict `INCONCLUSIVE`.
- **Every declared stage exists.** A stage that did not run is an object with `status: skip` and
  a reason; it never disappears and never becomes a JSON string.
- **`test_selection`** — the exact pytest target and independent grid chosen by the caller. A
  verdict applies only to those named inputs.
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
  Choosing the right `--tests` target from a diff is the unsolved part; an irrelevant target can
  still produce `PASS`. The report names the target so a reviewer can reject that evidence, but
  the executor cannot decide relevance itself.
- **Cross-architecture compilation.** `arch_coverage: compile-only` is reserved for a future
  stage that actually invokes an architecture-specific compiler. No-GPU mode does not claim it.
- **`perf` and `claims` stages.** The schema reserves both — median-of-N against a baseline on the
  same locked GPU, and reproducing the numbers in the PR description — and the script emits
  neither. A report today carries no performance evidence, and a review must not read the absence
  of a `perf` stage as "no regression".

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
