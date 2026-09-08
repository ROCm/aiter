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
Whether the target's own shapes leave a class of input untested. These were once an AST scanner
and a nineteen-flag command line, and the encoding was the mistake: a scanner that guesses wrong
is wrong silently, whereas you can read the target and say why.

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
| `report.py init \| set \| stage \| finding \| coverage \| finish` | the report itself. `finish` computes the verdict from the stages actually recorded, validates against `report_schema.json`, and is the **sole** writer of `verdict` and of the process exit code. |
| `pick-idle-gpu.py` | the sampling window that decides a GPU is idle, and the `idleness-basis:` line saying how it knows. |
| `gpu_probe.py` | which device the run actually holds — arch, BDF, activity — asked of amd-smi, never turning an unreadable reading into an idle one. |
| `target_run.py` | the decisions around one target run: what goes into the receipt probe, how a grid cell becomes a Python value, what a script target's exit code may be counted as, and which environment variables the target is allowed to see. |
| `scrape_perf.py` | everything around a timing run except the run: which harness the target exposes, how a benchmark's rows become comparable numbers, whether a cross-tree difference is attributable to the patch, and what a timing run is allowed to have left in the worktree. |
| `scan_index_width.py` | the 32-bit index-width scan. |

Two things are deliberately **not** on this list, for the same reason: they cannot survive being a
separate process.

The GPU **lock** is a `flock` on a file descriptor held open by the entry point for the whole run,
and a descriptor dies with the process that opened it — so a child that claimed the device would
release it on exit, and every concurrent validator would then pick the same one.

**Launching the target** needs that locked device, the private cache roots, and the constructed
environment assembled together at `env -i` time; handing all of it to a child would just move the
assembly, not isolate it. So the entry point spawns the target — for correctness and for timing
alike — and `target_run.py` and `scrape_perf.py` own everything that had to be *decided* around
those launches.

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

.claude/skills/validate-kernel-pr/validate_pr.sh \
    --repo "/tmp/pr-$PR" \
    --patch "/tmp/pr-$PR.patch" \
    --head-sha "$HEAD" \
    --target tests/kernels/test_softmax.py \
    --runner pytest --runner-reason "the file is a pytest module the PR ships" \
    --expected-route kernels.softmax_kernel:build_softmax_module \
    --shape-vars M,N,dtype_str \
    --shape-env ROCDSL_SOFTMAX_SHAPES \
    --grid "64,2048,f32;64,2000,f32" \
    --tol-table "f32=1e-5,f16=2e-3,bf16=1e-2" \
    --out validation_report.json
```

Everything that describes the PR is a flag; everything that describes the host is an environment
variable (next section).

| flag | meaning |
|---|---|
| `--repo` | worktree to validate (required) |
| `--patch` | patch to apply first; a conflict is a blocker, not a skip |
| `--head-sha` | the exact remote head this patch represents; omit only for a local candidate |
| `--target` | the script file or pytest node that exercises the change (`--tests` is an alias) |
| `--no-target` | the reason you looked and found none. Publishes a **blocker**, not a skip — see below |
| `--runner` `--runner-reason` | `pytest`, `script`, or `none`, and **why** — you read the target, the validator does not check you. `none` means you have decided nothing here is runnable |
| `--expected-route` | the `module:function` the profiler must observe; without it there is no receipt and no observed work |
| `--shape-vars` | local names captured at each route call, in grid order |
| `--shape-env` \| `--shape-arg` \| `--shape-argnames` | how the grid reaches the target: an env var it reads, its own CLI flag, or the `parametrize` names to replace |
| `--grid` | optional extra cells the target's own shapes miss. A passing grid earns no verdict; only a failing one is a finding |
| `--axis` | repeatable `NAME=--flag:v1;v2` — an independent axis that is not a shape |
| `--tol-table` | tolerances recorded alongside the comparison |
| `--perf-args` \| `--no-perf` | force the timing entry point, or skip timing entirely |
| `--perf-target` | the file to **time**, when that is not the file to run. Defaults to `--target`'s file |
| `--perf-control-column` | a column the patch does not touch; **required** before a transplanted baseline is believed |
| `--label` `--out` | run name and report path (default `./validation_report.json`) |

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

### Two places a target comes from, and the third case

A target is either **already in the repository** or **shipped by the PR**, and the difference is
recorded rather than flattened. `test_selection.test_provenance` reads it out of the patch:
`pre-existing` when the patch does not touch the target, `pr-added` or `pr-modified` when it does,
and `unknown` with no `--patch` to compare against — a checkout validated directly cannot tell a
test the author wrote from one they did not, and answering `pre-existing` there would assert an
independence nobody established. `none` is the fourth case, below.

Both are worth running. Only one is independent. A pre-existing test was not written to make this
PR pass; a test that arrives with the patch was written by the same hand as the code it grades and
can pass vacuously without anyone noticing — which is why the execution receipt matters most in
exactly that case. Whether the test is any *good* is a reading, and it stays with you; the report's
job is only to stop a reviewer mistaking one for the other.

**And when neither exists, say so with `--no-target "<what you looked for and did not find>"`.**
That is a `blocker` and a `BLOCK` verdict: a change with runtime surface arrived with nothing that
runs it. It is not a skip — a skip says the validator could not establish something, and here it
established exactly what it says. You never reach this for a PR with no runtime surface at all;
`review-pr` reports that as N/A and does not invoke the validator.

Supplying **neither** flag stays a usage error, and that asymmetry is deliberate. A forgotten
`--target` must not read as "there is no test", because that would publish your slip as a finding
against the author. The absence has to be declared, with a reason, the same way the runner is.

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
| `PERF_CONTROL_TOL` | how far `--perf-control-column` may move across two trees before the comparison is refused as unattributable, default 0.10 |

Each phase additionally gets a fresh private `AITER_JIT_DIR` and sets
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

A disabled row is also the clearest case for reaching past the target's own shapes with `--grid`:
the row tells you exactly which input stopped being covered. That is an answer to a disabled row,
not a substitute for noticing one.

### 5 — `correctness` — the target, and optionally shapes it does not run

The target's own run is the evidence. An extra shape grid can be layered on top of it, reported
separately, because the interesting case is when the two disagree — but only the target's run
can earn a verdict.

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
runner choice was. The report records `runner_basis` — `declared-by-caller` when it was your
declaration, `explicit-node-selector` when the target string carried a `::` node and settled it,
`target-missing` when there was no file to run, `undeclared` when nobody said — so that a reader
can tell your claim from a measurement.

Both runners are profiled by the same `sys.setprofile` hook, so the receipt means the same thing
either way; only the delivery differs, because pytest owns its own startup and a script does not.
A pytest target loads the probe as a plugin; a script target is executed by a validator-owned
wrapper under `runpy` with `run_name="__main__"`, which is what makes a `__main__` guard fire.

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
reached `INCONCLUSIVE` for that reason alone — back when a missing grid capped the verdict — and
the skip text blamed the kernel for a limit that belonged to the injector. The verdict no longer
turns on it, but the diagnostic still has to name the injector rather than the target.

**You name the channel; the validator does not read the file to check you.** It records
`grid_channel_basis: declared-by-caller`, because your naming it is a claim, not a measurement.

**Reading the source is never enough to credit the channel.** Re-run the target with a
deliberately invalid grid and require it to **fail**. A target that passes with garbage shapes is
not consuming the grid — whatever the source looked like — so the stage is `skip`, never credited.
This one probe is what makes every channel equally trustworthy, and it is the *only* thing that
credits one, so it is where a channel you named wrongly gets caught.

It catches the quiet case: a name the target ignores changes nothing, the invalid grid passes, and
the stage skips. The loud case it cannot resolve — a flag the target does not define makes argparse
exit non-zero, which looks exactly like a grid that found a shape the kernel crashes on. Nothing in
the evidence tells those apart, so the run does not guess: the grid stays `fail`, the note names
both possibilities, and **no blocker is charged**, because a receipt that observed no call to the
routed work never saw the author's code fail at all. Check the flag yourself before you name it;
the run will not do it for you.

With no channel at all the stage is `skip`, and that is now just a skip: the verdict is unaffected,
because the grid is not a required stage. Say `repo-default-only` rather than claim coverage that
does not exist — the report should show that the target's own shapes were all that ran.

#### The grid earns nothing, and that is what makes it safe

`correctness_s1_grid` is **not a required stage**, and a passing grid cannot complete a verdict.
Only a *failing* one moves anything, and a failure is a real defect whether or not the cells were
novel.

That asymmetry replaced a much larger apparatus, and the history is worth keeping. The grid used
to be required, so a run without one topped out at `INCONCLUSIVE` — which meant every caller had
to supply a grid whether or not they had anything to say with it. On ROCm/aiter#4538 all three
requested shapes were already in the target's own default list: the "independent" grid re-ran a
strict subset of the repository run and the stage reported `pass`. The answer at the time was to
make the caller *declare* what their cells covered (`--grid-novelty`) and to refuse a pass without
it — more bookkeeping around a grid nobody wanted to supply.

Making the grid optional removes the pressure that produced that grid in the first place, and with
it the reason to police duplication at all: a duplicate grid that passes now proves nothing and
claims nothing, which is the correct amount. The `grid_independence` vocabulary is gone rather
than fixed, because the error it guarded against is no longer reachable.

So supply `--grid` when you have read the diff and can say what the target's own shapes miss —
a tail path against a suite of powers of two, long-context against a suite of toys. Supply nothing
when you cannot. Neither choice costs you a verdict.

#### Axes: when the failing configuration is not a shape

A grid is one ordered tuple on one channel, which is all a shape flag accepts. A target whose
remaining knobs are separate flags — head counts, dtypes, window modes — cannot be gridded over
them at all, so entire configurations stay unreachable however the grid is spelled. That is not a
missing shape; it is a missing axis.

aiter#4538 again: the shape flag carries `(seq_len, seq_len_kv)` while `--num-heads` is its own
flag defaulting to `64 128`, and the public API asserts at `num_heads=16` — a real blocker no grid
could have requested.

An axis is a name, a flag, and its values: `--axis num_heads=--num-heads:16;32`. **You read the
target to find the flag; the validator does not check that it exists.** If you name a flag the
target does not take, the grid run dies at argument parsing — see the shape-channel section above
for why that failure is reported without charging anyone.

What the validator does enforce is the same burden of proof a shape channel carries: every axis
flag is fed `__VALIDATOR_INVALID_AXIS__` and **must fail**. A flag that is declared but ignored,
or whose value is silently clamped, would otherwise let the report claim coverage of head counts
that never reached the kernel.

| state | meaning |
|---|---|
| `none` / `unusable` | none requested, or the target is not a script — argv reaches script targets only |
| `malformed-spec` | the `name=--flag:v1;v2` spelling does not parse; nothing is guessed from it |
| `declared` | parsed and usable, but the probe has not reported yet. A finished report carrying this means the run died before the axis was proven, so believe nothing about the axis |
| `hook-not-consumed` | a flag accepted the invalid value; the axis is **dropped and named**, never dropped quietly |
| `proven` | every axis flag refused the invalid value, and its values rode the grid run's argv |

Each axis also carries `hook_proof`, the probe's verdict for that one flag. It is the only
evidence that an axis reached the kernel; a flag existing in the source is not.

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

**One receipt per run, not per phase.** The repo-tests run and a grid run both execute inside the
head phase. Sharing one receipt path meant the second erased the first — and with the grid cells a
subset of the target's defaults, a receipt written by *either* run satisfied the grid's
requirement, which made the grid's own evidence unfalsifiable. One file per run, read the grid
run's own file when a grid ran, and record which run the published receipt describes.

With no grid the receipt's `required_shapes` is empty, and that is a legal `pass`. It was
unreachable while the grid was mandatory, which is why the schema demanded a non-empty list there;
`executed_shapes` still has to be non-empty, so a receipt never passes without having watched
something happen.

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

**The file that is timed need not be the file that is run.** A timing run executes a file; a
correctness run may select cases inside one with a pytest node id. They were the same file for as
long as perf had no target of its own, and `--perf-target` is how you say otherwise — an aiter
kernel's unit test and its bench are routinely two different files. `perf.target` names what was
timed on every status including `skip`, and `perf.target_basis` says whose choice it was:
`declared-by-caller` when someone who read the diff named it, `discovered-pr-shipped` when the
patch brought a bench along and the validator took it, `same-as-correctness-target` when it fell
back. The fallback is an inference, not a reading of the change, and the report does not let the
two look alike.

**Two places a perf target comes from, and neither of them is a filename.** When no
`--perf-target` is given, both are searched:

1. **A bench the PR ships** — `discovered-pr-shipped`. A PR that means to be faster usually says
   so by bringing one along.
2. **A bench the repository already has** — `discovered-repo-bench`. Every `.py` in the worktree
   that carries a harness *and* imports a module the patch changed under `aiter/`.
   `op_benchmarks/triton/bench_gemm_a8w8.py` imports `aiter.ops.triton.gemm.basic.gemm_a8w8`;
   that import is an edge that can be checked, where a matching filename is only a resemblance.
   The imports are **parsed, not matched** — aiter spells the same edge three ways, including
   `from aiter.ops.triton.attention import extend_attention` where the imported name is itself a
   module, and a regex loose enough to catch that also matches `gemm_a8w8_preshuffle` when the
   patch touched `gemm_a8w8`.

The repository bench wins, and mechanically rather than as a preference: it is on **both sides**
of the patch, so the baseline is this worktree with the patch reversed. A bench the PR adds is
absent from base and forces the cross-tree transplant below, which needs `--perf-control-column`
before it means anything.

Every other outcome is the fallback, and that asymmetry is the entire safety argument: a target
declined costs a measurement, while a target chosen *wrong* spends a `should-fix` on an author
whose code may be innocent — and nothing downstream can tell those apart, because `run_perf`
injects no probe and no evidence exists that the bench executed the changed line rather than
merely importing near it. **The edge proves reference, not execution.** That is the residual risk,
and the refusals below are what bound it.

So more than one candidate is **named, not chosen between**. `perf.candidates` lists everything
considered and `perf.target_basis_reason` says why the fallback stood, because a reader told only
that it stood cannot distinguish an empty search from one that found three benches and refused.
Which of several benches measures a given change is a reading of the diff, not a fact about it;
settle it with `--perf-target`. This is the rule `--runner` already established.

One tie-break comes before that refusal, and only among candidates the import edge already
proved: if exactly one of them lives under `op_tests/op_benchmarks/`, that is the benchmark. A
file's place in the directory the project set aside for timing is a fact about how aiter is
organised, not the resemblance the import edge exists to replace. Measured on 40 random
`aiter/ops/triton` modules, it turns 10 resolutions into 12 and leaves the genuine ties alone —
a change to the shared `aiter.ops.triton.utils.types` still declines, because 13 of its 14
candidates live there and a tie is still a tie.

A note on what this searches: **every** `.py` in the worktree, not a curated list of directories.
aiter keeps benches in `op_tests/op_benchmarks/`, but 119 files elsewhere under `op_tests/` carry
a timing harness too, and a hardcoded directory would quietly decide those are not perf tests.
Measured at 1530 files and 0.3 s.

`perf.target_provenance` then asks of that file the same question `test_provenance` asks of the
correctness target, using the same function against the same post-apply snapshot:
`pre-existing`, `pr-added`, `pr-modified`, or `unknown` when no patch was supplied. It is not
bookkeeping — it decides the baseline. Only a `pre-existing` target sits on both sides of the
patch and can be timed by reversing it; anything the patch wrote needs the transplant below.

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
the correctness target, execution receipt, and index scan all ran. It does **not** mean an extra
shape grid was supplied — that stage is optional and completes nothing either way. Nor does it mean
a timing comparison was made: read `stages.perf` for that, and read a `skip` there as "not
measured", not as "no regression".

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
- **`test_selection`** — the exact target, where it came from relative to the patch
  (`test_provenance`), the selected runner, and any extra grid. A verdict applies only to
  those named inputs.
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
  have to be bound without changing the PR's diff hash or its live-base identity. Such a target
  simply gets no grid, which costs it nothing now, and the reason says which case applied.
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
