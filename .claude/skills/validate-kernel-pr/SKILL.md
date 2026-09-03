---
name: validate-kernel-pr
description: Run a kernel PR's tests as evidence rather than as a claim. Apply the base-to-head patch in an isolated worktree, run it on a locked idle GPU, compare the same targets against base, and record every result in a head-bound validation_report.json. You supply the judgement — which target exercises the diff, how it takes its shapes, whether a tolerance was loosened; the ledger and the verdict are computed by tools you cannot narrate around. Missing environment evidence is INCONCLUSIVE, never PASS. Use when validating, reproducing, or gathering runtime evidence for a kernel PR in aiter or FlyDSL.
argument-hint: <pr number or owner/repo#N>
---

# validate-kernel-pr

`review-pr` reads the diff; it does not build and it does not run. It is a static reviewer, and a
good one — but three failure modes are invisible to it, and this skill exists for exactly those
three:

1. **The PR's own tests pass while the kernel is wrong.** A suite whose non-aligned shapes are
   commented out reports green on an out-of-bounds tail store.
2. **A green suite that cannot fail.** Loosening a comparison tolerance leaves every test
   passing and the kernel unguarded.
3. **Defects that only exist at runtime.** LDS over-allocation on one arch, an accuracy gate
   failing against the reference, a JIT path that no-ops on cache miss.

Output is `validation_report.json`: deterministic execution evidence kept separate from
`review-pr`'s advisory judgement. A review may consume it only when `repo.head` matches the exact
PR head; a review written without one must mark validation `NOT RUN`.

The two skills stay split at judgement, not at invocation. `review-pr` triages whether a PR has
runtime surface at all and, when it does and the PR ships a single target, runs this script itself
rather than asking a human to. Everything below is still produced here and merely consumed there:
the executor never writes an advisory verdict, and `review-pr` never manufactures evidence it did
not get from a report.

---

## You may judge; you may not keep the books

Most of the work below is judgement, and it is yours. Which target actually exercises the diff.
How that target takes its shapes. Whether a changed tolerance was loosened or merely moved.
Whether a grid cell is anything other than the default the target would have run anyway. These
were once an AST scanner and a nineteen-flag command line, and the encoding was the mistake: a
scanner that guesses wrong is wrong silently, whereas you can read the target and say why.

The ledger is not yours. Whether a stage ran, what it exited with, which GPU held the lock, which
route the profiler observed, and what verdict follows — those are written by the tools below and
read back out of the report. The reason is narrow and non-negotiable: a model asked to report on
its own execution will report success it did not observe, and that is the single failure this
skill exists to prevent. So every fact that could clear a PR is recorded by a process separate
from the one narrating it.

The practical rule: **reason in prose, record through a command.** If a claim ends up in
`validation_report.json`, a command put it there.

## The command surface

These are the only ways a fact reaches the report. (They are being carved out of the current
`validate_pr.sh`, which is still the shipped entry point; the responsibilities below are the
boundary, whatever the file layout.)

| command | what it owns |
|---|---|
| `report.py init \| set \| stage \| finding \| finish` | the report itself. `finish` computes the verdict from the stages actually recorded, validates against `report_schema.json`, and is the **sole** writer of `verdict` and of the process exit code. |
| `pick-idle-gpu.py` | the sampling window that decides a GPU is idle, and the `idleness-basis:` line saying how it knows. |
| `gpu_probe.py` | which device the run actually holds — arch, BDF, activity — asked of amd-smi, never turning an unreadable reading into an idle one. |
| `run-target.sh` | one target, one phase. Patch state, private caches, the constructed environment, the receipt probe, and the exit code plus JUnit counts that come back. |
| `scrape_perf.py` | parsing a benchmark's rows into comparable numbers. |
| `scan_index_width.py` | the 32-bit index-width scan. |

The GPU **lock** is deliberately not on this list. It is a `flock` on a file descriptor held open
by the entry point for the whole run, and a descriptor dies with the process that opened it — so a
child that claimed the device would release it on exit, and every concurrent validator would then
pick the same one. The lock stays with whoever runs the tests, not with a tool that can be called.

You choose what to run and you explain why. You do not hand-write a stage result, and you do not
compute the verdict — `finish` does, from what is on record.

## Establishing the run's identity

Four facts pin a run, and a report missing any of them is evidence about nothing in particular:
the **base commit** it was compared against, the **patch** and its SHA-256, the exact **head OID**
the patch represents, and the **target**.

```bash
REPO=ROCm/FlyDSL
PR="${PR:?set PR to the PR number}"

BASE_REF=$(gh pr view "$PR" --repo "$REPO" --json baseRefName --jq .baseRefName)
BASE_REF_PATH=$(python3 -c \
  'import sys,urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' \
  "$BASE_REF")
BASE=$(gh api "repos/$REPO/branches/$BASE_REF_PATH" --jq .commit.sha)
HEAD=$(gh pr view "$PR" --repo "$REPO" --json headRefOid --jq .headRefOid)

git worktree add --detach "/tmp/pr-$PR" "$BASE"
gh pr diff "$PR" --repo "$REPO" > "/tmp/pr-$PR.patch"
```

Take the base from the **branch tip**, not from `baseRefOid`. `baseRefOid` is where the branch
stood when the PR was opened; comparing against it attributes every intervening merge on the base
branch to this PR's author.

Validate in a worktree you created, and leave it as you found it — the patch is applied and
reverted around each phase, and a run that dies mid-phase must still restore the base. A dirty
worktree makes the next phase's result unattributable.

If there is no remote head — a local candidate — record `repo.head: null` rather than inventing
one. The report stays useful locally, and `review-pr` correctly refuses it as PR evidence.

Choose the target by reading the diff, not by pattern-matching a filename. A target that does not
touch the changed code can return `PASS` on evidence about something else entirely, and that reads
to a reviewer as clearance — worse than no report at all.

## Host settings

These stay environment variables because they describe the **host**, not the PR under test — a
caller validating many PRs on one machine sets them once and never thinks about them again:

| env | meaning |
|---|---|
| `PYLIB` | runtime modules living outside the checkout |
| `PYTHON_BIN` | the one interpreter used for both pytest and script targets |
| `PICKER` | override the shipped `pick-idle-gpu.py`; unset, the **shipped** picker is used, and only then one found on `PATH` |
| `TIMEOUT` | per-target budget, default 1800s |
| `PERF_TIMEOUT` | the timing stage's own budget, defaulting to `TIMEOUT`, because a bench sweep is legitimately longer than a correctness run |
| `PERF_REPEAT` | runs per side, default 3 |
| `PERF_THRESHOLD` | head/base ratio that counts as a regression, default 0.95 |
| `PERF_MIN_ROWS` | matched rows required before any ratio is reported, default 3 |

`run-target.sh` additionally gives each phase a fresh private `AITER_JIT_DIR` and sets
`PYTHONDONTWRITEBYTECODE=1`, so JIT output cannot cross between base and head or dirty the
worktree.

---

## Stages

Every stage records its own status. A stage that could not run says `skip` and names the fact the
skip rests on; it never says `pass` for work it did not do. Record each one as you finish it —
`finish` builds the verdict from what is on record, so a stage you reasoned about but never wrote
down counts as not run.

### 1 — `merge_sim`

Apply the head patch onto the base you pinned. **A conflict is a blocker and stops the run**: no
number produced after it would describe the merged code, so continuing would generate evidence
about a tree nobody will ship.

The worktree must be clean going in, and the patch must be reverted on the way out — on interrupt
and on every degraded path, not just the happy one. Otherwise the next run in that worktree sees a
dirty tree and reports it against the wrong author.

Look twice at files many PRs edit at once, because that is where silent semantic collisions live
even when git merges cleanly: tuning CSVs (duplicate shape rows), `csrc/include/rocm_ops.hpp`,
`aiter/jit/optCompilerConfig.json`.

A head checkout with no patch can still run diagnostics, but it proves neither mergeability nor
base attribution, and so cannot reach `PASS`.

### 2 — `gpu_claim`

Claim a GPU over a **sampling window** rather than one instantaneous reading, then take a
non-blocking lock and hold that file descriptor for the entire run — a GPU that was idle when you
looked is not the same as a GPU nobody else takes while you measure.

Two things about the recording are easy to get wrong, and both were:

- The picker emits a **translated HIP index**, which is not an AMD SMI index. Using it as one
  identifies the wrong device in the report.
- `amdsmi_get_gpu_activity` is unavailable on some driver/amd-smi combinations — it fails or
  returns `N/A` while enumeration, BDF, ASIC and VRAM all answer fine. So the claim records
  *what it rests on*: `activity+vram` when busy percentages were really measured, `vram-only` when
  only resident VRAM separated the devices. In the second case activity is `null`, meaning
  unknown — **never 0**, which would read as an observed idle GPU.

If nothing stays idle, `gpu_claim` is `skip` with `degraded_mode: NO_GPU`. Say which fact the skip
rests on: no GPUs on this host, or GPUs present but all busy, are facts about the environment;
AMD SMI being unqueryable says nothing about the GPUs at all, and the three must not collapse into
one message.

**Then ask whether the target even needed one.** Run it once with no visible device.
`gpu_requirement` is `not-required` only if it passes **and executes at least one test** — a suite
guarded by `skipif(not torch.cuda.is_available())` also exits 0 while proving nothing, so the
executed count is what makes this evidence rather than an assumption.

Resist the urge to decide this from the diff instead. A Python-level dispatch change reroutes
kernels without touching kernel source, and ROCm/aiter#5089 decided whether 34 gfx950 kernels
compile from a seven-line helper — no rule over changed paths would have settled either.

A `not-required` target runs its correctness stages rather than skipping them; that is the evidence
a CPU-only fix can honestly supply. It earns nothing more: `arch_coverage` stays empty, because
only a passing `gpu_claim` credits an architecture, and `PASS` still requires one.

### 3 — `runtime_compat`

Does the repository's own package import, from *this* checkout, against the runtime actually
installed here?

Resolve it the way the repository does. Aiter resolves `aiter` from the checkout. FlyDSL resolves
the pinned package from `PYLIB` and compares its version against the checkout's `python/flydsl` —
which keeps compiled `_mlir` bindings reachable without pretending an unrelated FlyDSL install
validates an Aiter checkout.

The reason this is a gate and not a footnote: a pinned prebuilt runtime drifts behind the tree, and
the `ImportError` that follows is indistinguishable from a defect in the PR. So a mismatch is
recorded as an **environment fact** — `runtime_compat` and correctness skip, verdict `INCONCLUSIVE`,
nothing attributed to the author.

One refusal is absolute: if a FlyDSL PR touches Python, C++/MLIR bindings, headers, CMake, or
packaging inputs, **a prebuilt `PYLIB` is not accepted**. Trusted build provenance does not exist
here, and caller-authored metadata cannot prove which source produced a binary. Return
`INCONCLUSIVE` rather than test a stale package and call it the PR's.

This is why the whole gate exists: FlyDSL kernels import symbols from a compiled runtime, so "one
fresh container per PR" would mean rebuilding MLIR/LLVM per PR. A pinned image plus this gate is
the workable shape.

### 4 — `test_policy` — reason about this **before** you trust the suite

A suite that cannot fail is worse than no suite, because it returns green. So establish that this
suite can still fail before its result means anything — afterwards, a green run has already made
the question feel answered.

This stage is judgement, and it is yours. Read the test files' head-vs-base diff and ask two
questions.

**Was a tolerance loosened?** Not "is this tolerance loose" — repositories legitimately differ per
kernel, and an absolute threshold would flag half the tree. The question is whether *this change*
widened what was already there. Note that a comparison can be weakened without any number moving:
switching `assert_close` for `allclose`, dropping a `rtol=` argument so the default applies, or
comparing against a value the kernel itself produced. Then attribute it:

- **test-only change** → blocker. Nothing else in the PR explains the widening.
- **kernel code changed too** → `NEEDS_WORK` pending numerical justification, not a blocker. The
  looser tolerance may be the honest consequence of a new algorithm, and calling that a
  deterministic failure was a false block worth avoiding.

**Were shape rows disabled?** Compare head against base again. Rows already commented out in base
are coverage context — record them, they tell a reviewer what this suite never covered — but only
rows *this change* disabled produce `NEEDS_WORK`. The distinction matters because the pre-existing
ones are usually numerous and would drown the one that is actually the PR's doing.

Either way, the independent grid in the next stage stays visible: it is the answer to a disabled
row, not a substitute for noticing one.

### 5 — `correctness` — the repo's tests, then a grid the repo does not run

Two runs, reported separately, because the interesting case is when they disagree: what the PR's
own suite says, and what a grid the PR never runs says.

#### Choosing how to run the target

**You decide this, and you must say so: `--runner pytest|script` with `--runner-reason`.** The
validator no longer guesses, and with nothing declared it runs nothing and says the runner was
never declared. A wrong guess here is not a wrong guess about style — it publishes *"the PR's own
test fails on head"* against the author, on a target that is green.

Read the file. A file defining `test*`/`Test*` is usually pytest; a file with an
`if __name__ == "__main__"` guard runs as a script; a file with neither is `skip`, never a test
failure. Only `path::node` decides itself — nothing can run that string as a script.

Two traps, both of which have already produced that false blocker:

- **"Defines a `test*` function" is not "pytest can collect it."** aiter's dominant `op_tests`
  convention is a *script* whose worker happens to be named `test_<op>(m, d, dtype)` and is called
  from `main()` with real arguments. pytest collects it, cannot supply the parameters, and errors.
  So look at the parameters: **required positional parameters, and no `parametrize`/`fixture`/
  `usefixtures` decorator, means pytest cannot run it** — a required positional can be a fixture,
  but not one the file neither defines nor imports. When in doubt call it a script; being wrong in
  that direction costs a run, being wrong in the other direction blames a person.
  (ROCm/aiter#5081.)
- **A module that parses argv in its body cannot be collected**, even if it does define real test
  nodes: pytest imports it during collection with pytest's own argv, and argparse exits the
  process. The same file is green as a script. (ROCm/aiter#5172.)

So when a run executes nothing, say that in those words. *"Red on both sides"* is an attribution,
not an explanation, and a reader who is not told otherwise concludes the code is broken when the
runner choice was. The report records `runner_basis` — whether the runner was your declaration or
a fact about the target — so that a reader can tell your claim from a measurement.

Both runners are profiled identically: the probe is installed by a validator-owned wrapper that
then executes the file under `runpy` with `run_name="__main__"`, because nothing about
`sys.setprofile` ever needed pytest; pytest was only where the hook was convenient to install.

#### What counts as having run

Pytest emits JUnit XML, and a zero-executed or all-skipped target is `skip`, never `pass`.

A script publishes no per-case count, so its `executed` is a **liveness signal only** — the
process ran. It must never stand in for work. aiter#4538's target returns silently with exit 0 and
a log line when the arch is unsupported or an optional package is missing; that produced exactly
the same `executed: 1` as the run which graded 56 cases, and earned the same `arch_coverage` on a
basis that described the process rather than the kernel. Hence:

- `observed_work` is the only number backed by evidence — route calls counted in **that run's
  own** receipt. It is `null` when no route was named, because nothing was watched.
- `arch_coverage` credits nothing when `observed_work` is `0`: a route was watched for and never
  reached the device. The basis line prints the count and where it came from, rather than
  implying a measurement.

#### The baseline must be the same tree minus the patch

Reverse the exact patch, confirm the worktree is clean, run base under base-only caches, reapply,
run head under separate caches.

Reversing removes **new files too**, which is the point: a test the PR added is `target-not-present`
on base, not a pre-existing failure. Any leftover artifact, any reverse/reapply failure, any cache
bleed aborts the head run into `INCONCLUSIVE` — a baseline you cannot trust makes the comparison
worthless in the direction that clears the PR.

#### The grid

Cover three classes the PR's own tests routinely miss:

| class | why |
|---|---|
| non-toy | `M=1` / `M=16` only is the standard agent-generated test |
| boundary / odd | odd N, N not a multiple of the tile — where tail masks fail |
| long-context / large M | where 32-bit index arithmetic wraps |

Then decide how the grid reaches the target. These are alternatives — pick the one the target
actually has, and say what in the source told you so:

| the target takes its shapes from | what to look for |
|---|---|
| an environment variable | the source reads that name via `os.getenv` / `os.environ` |
| its own CLI flag | the source passes that flag literal to `add_argument` |
| `@pytest.mark.parametrize` literals | the source binds those names as test parameters; the shipped plugin replaces them |

The third channel exists because the first two require the target to have been *written* for a
validator. Zero of the seven files in aiter's `op_tests/flydsl_tests/` expose an env var or a shape
flag; every one declares shapes as parametrize literals. Four consecutive real FlyDSL kernel PRs
reached `INCONCLUSIVE` for that reason alone, and the skip text blamed the kernel for a limit that
belonged to the injector.

**Reading the source is never enough to credit the channel.** Re-run the target with a
deliberately invalid grid and require it to **fail**. A target that passes with garbage shapes is
not consuming the grid — whatever the source looked like — so the stage is `skip`, never credited.
This one probe is what makes every channel equally trustworthy, so do not skip it because the AST
evidence looked convincing.

With no channel at all the stage is `skip` and the verdict `INCONCLUSIVE`. That is a positive
control: without it, the same default test run gets reported twice under two stage names. When the
kernel exposes no shape override at all, say `repo-default-only` rather than claim coverage that
does not exist.

#### A proven channel is not the same as added coverage

A target can consume the grid faithfully and still be handed cells it already runs by default. On
ROCm/aiter#4538 all three requested shapes were in the target's own default list, so the
"independent" grid re-ran a strict subset of the repository run and the stage reported `pass` —
exactly the duplication this stage exists to prevent, and invisible in the report.

So compare the cells against the target's own declared default and record which case holds:

| value | meaning |
|---|---|
| `adds-coverage` | at least one cell is outside the target's own default |
| `duplicates-target-defaults` | every cell is already a default. A **passing** run is downgraded to `skip` and the verdict to `INCONCLUSIVE`, because it proves only what the repo-tests stage already said |
| `unknown` | there is no literal default to compare against — every env-var channel, and any flag whose default is computed |

A duplicate grid that **fails** keeps its `fail`. The finding is real; what a duplicate cannot do
is earn a pass.

And say *which* of the several ways the comparison can be skipped actually applied. Defaulting to
a claim about the target — "the channel exposes no declared defaults" — asserts something the run
never established.

#### Axes: when the failing configuration is not a shape

A grid is one ordered tuple on one channel, which is all a shape flag accepts. A target whose
remaining knobs are separate flags — head counts, dtypes, window modes — cannot be gridded over
them at all, so entire configurations stay unreachable however the grid is spelled. That is not a
missing shape; it is a missing axis.

aiter#4538 again: the shape flag carries `(seq_len, seq_len_kv)` while `--num-heads` is its own
flag defaulting to `64 128`, and the public API asserts at `num_heads=16` — a real blocker no grid
could have requested.

An axis is a name, a flag, and its values, and it carries the same burden of proof as a shape
channel, in two steps. Record which step it reached:

| state | meaning |
|---|---|
| `none` / `unusable` | none requested, or the target is not a script — argv reaches script targets only |
| `hook-not-found` | the source declares no `add_argument` for that flag |
| `hook-not-consumed` | the flag was declared but accepted a deliberately invalid value; the axis is **dropped and named**, never dropped quietly |
| `proven` | every axis flag refused an invalid value, and its values rode the grid run's argv |

Each axis is judged for independence against its flag's declared default on the same terms as the
shape cells — and a proven axis asking for values outside the default makes the run independent
even when the shape cells duplicate.

A requested axis is recorded **whatever becomes of it**, including when the run never got far
enough to look for the flag. Dropping the request itself is precisely the silently narrowed test
space this is here to make visible.

### 6 — `execution_receipt`

Name the exact Python `module:function` route the diff is supposed to make execute, and the local
variable names inside it that carry the shapes. A validator-owned profiler loads before collection
and records what actually got called:

```json
{
  "schema_version": 1,
  "route": "aiter.ops.flydsl.kernels.moe_2stage_a16wmix:flydsl_a16w4_gemm1",
  "kernel_symbols": ["aiter.ops.flydsl.kernels.moe_2stage_a16wmix:flydsl_a16w4_gemm1"],
  "executed_shapes": ["1,3584,384", "128,3584,384"]
}
```

`PASS` requires the observed route to equal the one you named, at least one observed route symbol,
and every shape the grid asked for. The tested PR cannot earn credit by writing its own receipt:
the producer is validator-owned, and the script runner calls that producer's hooks rather than
re-implementing them — a re-implementation would be a second thing the PR's tree could influence.

Validate a receipt whenever a route was named, **including** when no grid was configured or its
channel could not be established. With no grid it attests route execution and nothing about
shapes, which is all it is then entitled to claim; abandoning it alongside the grid would throw
away evidence already collected.

**One receipt per run, not per phase.** The repo-tests run and the grid run both execute inside
the head phase. Sharing one receipt path meant the second erased the first — and with the grid
cells a subset of the target's defaults, a receipt written by *either* run satisfied the grid's
requirement, which made the grid's own evidence unfalsifiable. One file per run, read the grid
run's own file when a grid ran, and record which run the published receipt describes.

**Name the op a reviewer cares about, not the wrapper it runs through.** The probe resolves the
declared route to a code object and walks the `__wrapped__` chain, so decoration does not have to
be worked around by hand — aiter's entire `@compile_ops` family executes as `aiter.jit.core:wrapper`
under a plain string match, and naming the actual op matched nothing.

**Shape capture is a separate question, and it constrains which route to name.** The shape locals
have to be bound in that route's own frame. A dispatch wrapper declared `(*args, **kwargs)` binds
none of them, so a route through one attests execution and nothing about shapes, and the receipt
says *missing required shapes* rather than passing. Choosing a route whose frame actually carries
those names is your judgement to make before the run, not a defect to diagnose after it.

**A route is not a variant.** The receipt records that
`…mqa_logits.fp8_mqa_logits:flydsl_fp8_mqa_logits` was entered and with which shape-locals.
It says nothing about which of that module's 30 registered gfx950 kernel variants the call
selected, so no variant-coverage claim is supportable from a receipt. Getting one needs the
variant to be a declared axis or a captured shape-local; see [Not implemented yet](#not-implemented-yet).

### 7 — `index_width_scan` (informational)

Runs `scan_index_width.py` over the diff and records the count of index×stride multiplies that
carry no 64-bit widening. Candidates, not verdicts — see
[Reading the index-width candidates](#reading-the-index-width-candidates) for how to judge one,
and do not let the count stand in for that judgement in either direction.

### 8 — `perf`

The cost of a kernel change, measured rather than assumed. Base and head are timed on the same
locked GPU, back to back, in the same worktree — the baseline is this PR's own base with the patch
reversed, not whatever machine the PR's table was produced on. A head-only number reproduces the
PR's own comparison and cannot show a regression.

**A PR that adds its own benchmark target is not a PR with no baseline.** "The PR adds this
target, so base has nothing to time against" ended the stage on aiter#4538 — whose entire
motivation is being faster than the kernel it replaces, and which was reported `PASS` with no
number at all. The file being new does not make the code it drives new: when the target only
exercises an entry point that already exists on base, dropping that exact file into the base
tree times the pre-PR implementation through the same harness. `perf.baseline_method` says
which baseline was used, `patch-reversed-same-worktree` or `target-transplant`.

A transplant spans two trees, so it carries one extra burden: name a **control column** — a timing
column the patch does not touch, typically a reference implementation the target times alongside
the kernel under test — and `skip` unless that column reproduces within `PERF_CONTROL_TOL`
(default 10 %) across the two runs. Two trees can differ in ways that have nothing to do with the
kernel, and the control is what distinguishes "the kernel got slower" from "these are different
machines wearing the same name". With no control, decline the transplant rather than guess, and
say the control was missing instead of claiming there was nothing to measure.

Note that the headline ratio is the *worst* column, so an unchanged reference column sitting at
1.0 caps it; read the per-column numbers for the kernel's own movement.

Time by default. The regression this stage catches is the one nobody suspected, and an opt-in
switch is only ever flipped by someone who already suspects. Detect the entry point from the
target — a bench scenario, a `perftest` or `@benchmark` harness — and name it explicitly when
detection would decline.

Rows are matched across the two sides by their identity columns. A header carrying no
recognised unit is treated as an identity column — but aiter bench tables routinely print
unlabeled *measurements* (`flydsl rel`, `triton err`, `speedup`), which differ between base
and head by construction, so every row got a unique name on each side and `matched_rows` was
`0` on 56 perfectly comparable rows. The strict key is still tried first; only if it matches
nothing is a relaxed key used, in which identity columns whose cells are non-integral numbers
are dropped (shape and count columns are integral and stay). `row_key_basis` records which
key produced the match.

Each side runs `PERF_REPEAT` times and each cell is reduced to its **best** sample, which is what
makes a threshold as tight as 0.95 usable. Minimum is the correct estimator because contention,
clock ramp and scheduling only ever add time. `repeats` is recorded in the report, because the
threshold is only defensible if a reader can see N.

The stage is `skip` — never `fail` — on every path that is not "both sides ran clean and the
numbers disagree": no harness, a timeout, a nonzero exit on either side (a truncated log compares
whatever printed before the crash), or fewer than `PERF_MIN_ROWS` matched rows. A false regression
blocks a good PR and would get the stage switched off within a week.

A measured regression appends a `should-fix` finding, which makes the verdict `NEEDS_WORK` and the
exit code 1, and it ships its own reproducer: both logs, both exit codes, and the command. `perf`
is deliberately **not** a required stage — a run that could not be timed downgrades nothing, and a
`PASS` stays a `PASS`.

Because a bench harness routinely writes results next to the code (aiter targets drop a
`tuned_op_bench.csv` in the repo root), the timing run snapshots and restores the worktree, touching
only paths whose git status changed across it. Without that, the baseline cleanliness check would
fail and skip the entire head correctness phase — a perf stage that silently disables correctness
validation is far worse than no perf stage.

### 9 — verdict

`BLOCK` if a reproducible candidate defect fired, `NEEDS_WORK` if a deterministic policy concern
fired, `INCONCLUSIVE` if any required stage did not complete, else `PASS`. `PASS` therefore means
the merge simulation, GPU claim, repo-aware runtime probe, policy comparison, baseline control,
both correctness targets, execution receipt, and index scan all ran. It does **not** mean a timing
comparison was made: read `stages.perf` for that, and read a `skip` there as "not measured", not as
"no regression".

Process exit codes match the verdict: `PASS=0`, `BLOCK/NEEDS_WORK=1`, and `INCONCLUSIVE=2`.

You do not write this. `finish` derives it from the stages on record and is the only writer of
both the verdict and the exit code — which is what makes the list above a consequence of what ran
rather than a summary of it.

---

## Honesty rules the report enforces

These are fields, not prose, so a report cannot overclaim by omission:

- **`arch_coverage`** — per architecture, `runtime`, `compile-only`, or `not-covered`.
  A GPU claim alone earns no runtime coverage; `runtime` is added only after a selected head
  correctness test is collected and executed with that device visible. `compile-only` requires
  an actual architecture-specific compile.
- **`isolation`** — the real level. Where no container runtime is available it is
  `git-worktree + private caches`, and the report says `container: false`.
- **`isolation.target_environment`** — the target is unmerged third-party code, so it runs in
  a **constructed** environment (`env -i` plus a name/prefix allowlist, minus a
  secret-shaped denylist), not the reviewer's. `env VAR=… cmd` *adds* to the inherited
  environment; before this, every token in the calling shell was readable from `os.environ`
  inside the code under review — and reached a stage log the moment a target printed its
  environment. The report lists the variable names that were passed through.
- **The process exit code comes from this run.** It is read from a verdict file inside the run's
  own working directory, and the report path is deleted at startup. Deriving the exit code by
  re-reading the report made a previous run's file a fallback source of truth: a run that died
  before finishing exited on the *earlier* run's verdict.
- **`degraded_mode`** — `NO_GPU` when no device was claimable; required stages then make the
  verdict `INCONCLUSIVE`.
- **Every declared stage exists.** A stage that did not run is an object with `status: skip` and
  a reason; it never disappears and never becomes a JSON string.
- **`test_selection`** — the exact target, selected runner, and independent grid. A
  verdict applies only to those named inputs.
- **`runtime_identity`** — resolved package, interpreter, source SHA, and native artifact hashes.
- **`execution_receipt`** — observed route, kernel symbols, and exact shapes emitted by the test.
- **Every perf number keeps its provenance.** `stages.perf` carries the baseline it was measured
  against, the command, the harness it was detected from, the repeat count and reduction, the
  threshold, the matched-row count, and both logs. A ratio without those is not reportable, and a
  stage that could not measure says `skip` with a reason rather than reporting an empty comparison
  as agreement.

---

## Reading the index-width candidates

The scan reports index×stride multiplies added by the diff that carry no 64-bit widening. They are
**candidates, not verdicts**, and the count alone means nothing — the same expression is a defect
at one deployment scale and correct at another.

So judge each one against `production_scale.md`, which holds the numbers the diff does not contain.
A candidate is a defect only if you can name a shape that reaches it and show the product exceeding
2^31 there. If the scale table cannot settle it, say the candidate is unresolved rather than
guessing in either direction — and note that the table's first three rows are in-sample, drawn from
the problem statements of the fix PRs that supplied the defect labels, so they demonstrate the
arithmetic is decidable without establishing that it generalizes.

`scan_index_width.py`'s docstring carries why the scanner exists and why its trigger is structural
rather than a list of variable names.

---

## Not implemented yet

Deliberately absent rather than half-built — everything shipped here has been observed failing on
a seeded defect, and these have not been:

- **Target relevance.** Nothing checks that the target you chose exercises the diff. An irrelevant
  target can still produce `PASS`, and the only guard is that the report names it so a reviewer can
  reject the evidence. This is the load-bearing judgement in the whole skill and it is entirely
  yours.
- **External grid adapters.** The three channels reach the great majority of aiter's `op_tests`;
  what stays unreachable is a target whose shapes are none of them — a parametrized case whose
  parameter is a **dict or object** rather than scalar cells, and a target taking shapes from a
  file or a fixture. Supplying a separate harness of our own is not the answer either: it would
  have to be bound without changing the PR's diff hash or its live-base identity. Such runs stay
  `INCONCLUSIVE`, and the reason says which case applied.
- **Axes on the env-var and pytest channels.** An axis rides argv, so it reaches script targets
  only. A pytest target's extra knobs are `parametrize` argnames, needing a different injector,
  and an env-var channel has no per-axis spelling to prove against. Requesting an axis on either
  is `unusable` **with the reason**, never a silent drop.
- **Variant attestation.** A receipt proves the route ran; it cannot say which kernel variant
  the route selected. Until a variant is either a declared axis or a captured shape-local, a
  report covering a module with N registered variants covers the ones its inputs happen to
  select, and says so rather than implying N.
- **Naming the external runtime that executes the kernel.** `runtime_identity` resolves the
  repository's own package. A FlyDSL kernel reviewed as an aiter PR executes inside an external
  `flydsl` wheel whose version and hash the report does not record — the most load-bearing
  artifact in the run is the one it cannot identify.
- **Cross-architecture compilation.** `arch_coverage: compile-only` is reserved for a future
  stage that actually invokes an architecture-specific compiler. No-GPU mode does not claim it.
- **Reproducing the PR's stated numbers.** The `perf` stage measures base against head on this
  box; it does not attempt to reproduce the specific figures a PR description claims, so a number
  in the description that nobody can reproduce is not flagged as such. That check was previously
  reserved in the schema as a `claims` stage, which has been removed rather than left standing as
  a contract nothing satisfies.
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

A new stage must be able to **fail on a seeded defect**. Seed the defect it is meant to catch,
confirm the stage goes red *and* that the clean baseline stays green, and put both runs in the PR.
Only the pair is evidence: red on a seeded defect could be a stage that is red on everything. A
stage never observed failing is not a check, it is decoration.

Then decide where it belongs, which is the same question this file keeps asking. If the stage is a
**judgement** — reading a diff, weighing whether something is a defect at this scale — it is prose
here, and its output is a finding you record with a reason. If it is a **ledger entry** — did this
run, what did it exit with — it is a command, and it must be declared so that `finish` requires it.
An undeclared stage is one nobody notices the absence of, which is exactly how a report comes to
overclaim by omission.
