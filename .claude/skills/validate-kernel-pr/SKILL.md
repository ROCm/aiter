---
name: validate-kernel-pr
description: Deterministic validation layer for kernel PRs. Builds and runs the PR in an isolated worktree on a claimed GPU, runs a shape grid the PR's own tests may not cover, policy-checks the test diff, and emits validation_report.json. Use before review-pr on any kernel PR; review-pr consumes the report as evidence.
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

Output is `validation_report.json`: the evidence base that review findings hang on. A review
written without it must mark itself `[static-only review]`.

---

## Invocation

The caller supplies the checkout and the test target; this script does not fetch PRs itself
(see [Not implemented yet](#not-implemented-yet)).

```bash
# 1. put the PR head in its own worktree
git worktree add --detach /tmp/pr-4394 origin/main
git -C /tmp/pr-4394 fetch origin pull/4394/head && git -C /tmp/pr-4394 checkout FETCH_HEAD

# 2. validate it
.claude/skills/validate-kernel-pr/validate_pr.sh \
    --repo /tmp/pr-4394 \
    --tests op_tests/test_moe_2stage.py \
    --shape-env AITER_TEST_SHAPES \
    --grid "1,4096,f32;7,2000,bf16;16384,8192,bf16" \
    --tol-table "f32=1e-5,f16=2e-3,bf16=1e-2" \
    --out validation_report.json
```

To validate a candidate patch against a fixed base instead, add `--patch /tmp/candidate.patch`;
the `merge_sim` stage applies it and blocks on conflict.

| flag | meaning |
|---|---|
| `--repo` | worktree to validate (required) |
| `--tests` | pytest target the PR ships |
| `--patch` | patch to apply first; conflict is a blocker |
| `--shape-env` `--grid` | env var and shape list for the S1-owned grid |
| `--tol-table` | reference tolerances, e.g. `f32=1e-5,f16=2e-3,bf16=1e-2` |
| `--label` `--out` | run name and report path (default `./validation_report.json`) |

Environment knobs: `PYLIB` (prepended to `PYTHONPATH` when the runtime lives outside the
checkout), `PICKER` (path to `pick-idle-gpu.py`), `TIMEOUT` (per-pytest budget, default 1800s).

---

## Stages

Each stage writes its own status into the report. A stage that cannot run says `skip` with a
reason — it never reports `pass` for work it did not do.

### 1 — `merge_sim`

Apply the PR head on top of the current base. A conflict is a blocker and short-circuits: no
number produced downstream would describe the merged code. Known collision surfaces worth a
second look because they are edited by many PRs at once: tuning CSVs (duplicate shape rows),
`csrc/include/rocm_ops.hpp`, `aiter/jit/optCompilerConfig.json`.

### 2 — `gpu_claim`

Claim a GPU over a **sampling window**, not one instantaneous reading, and hold a lock for the
whole run:

```bash
PICK=$(pick-idle-gpu.py --samples 10 --interval 1 --quiet)
flock /tmp/gpu-$PICK.lock <command>
```

The report records host, HIP index, BDF, market name, architecture, and GFX activity before the
run. `pick-idle-gpu.py` emits the **translated HIP index**; `amd-smi` numbering can differ.

If no GPU stays idle, the run degrades to `COMPILE_ONLY` and every downstream claim is tagged —
it does not silently proceed on a contended device.

### 3 — `runtime_compat`

Does the checkout import against the runtime that is actually installed? A pinned prebuilt
runtime drifts behind the tree, and the resulting `ImportError` looks exactly like a defect in
the PR. When they disagree the report says `runtime_mismatch` and stops attributing the failure
to the author.

This matters most for FlyDSL kernels: the Python kernels import symbols from a compiled runtime,
so "one fresh container per PR" would mean rebuilding MLIR/LLVM per PR. The workable shape is a
pinned prebuilt image plus this compatibility gate.

### 4 — `test_policy` — run **before** the suite

A suite that cannot fail is worse than no suite, because it produces a green report. Two checks:

- **Tolerance, compared head-vs-base, not against an absolute table.** Repos legitimately differ
  per kernel; the question is whether *this change* loosened what was there. A loosened tolerance
  on an unchanged kernel path is a blocker.
- **Commented-out shape rows** in the test config. The suite then exercises fewer paths than it
  appears to, and the reviewer should know the grid below is doing that work instead.

### 5 — `correctness` — the repo's tests, then a grid the repo does not run

Both, and they are reported separately, because the interesting case is when they disagree.

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

`BLOCK` if any blocker fired, `NEEDS WORK` if any should-fix fired, else `PASS`. The verdict
covers **only what ran**; `arch_coverage` and `isolation` state the rest.

---

## Honesty rules the report enforces

These are fields, not prose, so a report cannot overclaim by omission:

- **`arch_coverage`** — per architecture, `runtime` or `compile-only`. A gfx950 host cannot
  validate a gfx942 claim; the report says so rather than implying a matrix it did not run.
- **`isolation`** — the real level. Where no container runtime is available it is
  `git-worktree + private caches`, and the report says `container: false`.
- **`degraded_mode`** — `COMPILE_ONLY` when no GPU was claimable.
- **Every perf number keeps its provenance.** A number in the PR description that does not
  reproduce is tagged `[unreproducible]`; it is not quietly dropped.

---

## Why the index-width scan is a separate stage

Rule `D9` in `review-pr` covers 32-bit overflow in pointer arithmetic, and its trigger is a list
of variable names (`token_id`, `seq_start`, `batch_offset`, `total_tokens`). Real defects use
other names, so the rule stays silent:

| defect | expression | consequence |
|---|---|---|
| aiter#1674 | `stride_out_batch` not `tl.int64` | output offset wraps at large MTP batch; tail rows keep stale sparse-KV indices |
| aiter#1674 | `block_id * stride` with no `.to(int64)` | every block past INT32_MAX returns logits of exactly 0.0, silently |
| aiter#3541 | `ArithValue(physical_block) * stride` | wraps on a ~150M-row KV pool; the wrapped offset still lands inside the allocation, so no fault |

The scan is structural instead — an index-shaped value multiplied by a stride-shaped value on a
line with no widening — and is deliberately noisy in the safe direction. A candidate checked and
dismissed costs one line of reasoning; a missed one costs silent wrong output.

```bash
.claude/skills/validate-kernel-pr/scan_index_width.py ROCm/aiter 1674
.claude/skills/validate-kernel-pr/scan_index_width.py --diff /tmp/pr.diff
```

---

## Not implemented yet

Deliberately absent rather than half-built — everything shipped here has been observed failing on
a seeded defect, and these have not been:

- **PR fetch orchestration.** There is no `--pr N`. The caller creates the worktree, as above.
  Choosing the right `--tests` target from a diff is the unsolved part; a wrong target produces a
  confident green.
- **`perf` and `claims` stages.** The schema reserves both — median-of-N against a baseline on the
  same locked GPU, and reproducing the numbers in the PR description — and the script emits
  neither. A report today carries no performance evidence, and a review must not read the absence
  of a `perf` stage as "no regression".

## What this skill does not do

- It does not replace `review-pr`. It produces evidence; the judgement stays there.
- It does not write findings about design, style, or API shape.
- It does not decide merge. A `PASS` means the stages that ran passed on the arch that ran.
- It does not validate an architecture it has no device for.

---

## Adding a stage

A new stage must be able to **fail on a seeded defect**. Before adding one, seed the defect it is
meant to catch, confirm the stage goes red and the clean baseline stays green, and record both in
the PR. A stage that has never been observed failing is not a check — it is decoration.
