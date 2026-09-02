---
name: validate-kernel-pr
description: Reproducible validation executor for kernel PRs. Applies an explicit base-to-head patch in an isolated worktree, runs it on a verified-idle GPU, compares the same targets against base, policy-checks the test diff, and emits a head-bound validation_report.json. Missing environment evidence is INCONCLUSIVE, never PASS.
argument-hint: --repo <worktree> --target <script file or pytest target>
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
| `--shape-arg` | the target's own CLI flag that accepts shapes, for script targets that read no env var |
| `--shape-argnames` | the pytest parameter names the grid should replace, for targets whose shapes are literals inside `@pytest.mark.parametrize` |
| `--axis` | repeatable `NAME=FLAG:v1;v2;…` — an extra independent test axis on its own CLI flag (see [`axes`](#axes-when-the-failing-configuration-is-not-a-shape)) |
| `--runner` | force `pytest` or `script` when the structural classifier gets it wrong; the report records both the forced choice and what selection had said |
| `--perf-control-column` | a timing column the patch does not touch; required before a transplanted baseline is believed (see [`perf`](#8--perf)) |
| `--tol-table` | reference tolerances recorded alongside the head-vs-base comparison (see [`test_policy`](#4--test_policy--run-before-the-suite)) |
| `--perf-args` | benchmark entry point for the timing stage; also forces perf on when detection would decline |
| `--no-perf` | skip the timing stage entirely |
| `--label` `--out` | run name and report path (default `./validation_report.json`) |

Several settings are environment variables rather than flags, because they describe the **host**
rather than the PR under test, and a caller validating many PRs on one machine sets them once:

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

Everything that describes the PR is a flag. The executor also overrides `AITER_JIT_DIR` with
separate fresh base/head directories and sets `PYTHONDONTWRITEBYTECODE=1`, so repository JIT
output cannot cross phases or dirty the worktree.

### Which shape channel to name

The three shape flags are alternatives, not a sequence; supply the one the target actually has.
The report says which channel was established in `test_selection.grid_channel`, and when none
was, `test_selection.grid_channel_reason` names each channel tried and what was found in the
target — so a failed guess costs one run, not a reading of this file.

| the target takes its shapes from | flag | what is checked before the channel is credited |
|---|---|---|
| an environment variable it reads | `--shape-env` | the source reads that name via `os.getenv` / `os.environ` |
| its own CLI flag | `--shape-arg` | the source passes that flag literal to `add_argument` |
| `@pytest.mark.parametrize` literals | `--shape-argnames` | the source binds all those names as test parameters |

All three are then held to the same proof: a deliberately unusable grid must make the target
**fail**. A target that ignores the grid produces a skip, never a pass.

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

If no GPU stays idle, `gpu_claim` is `skip`, `degraded_mode` is `NO_GPU`, and the script performs no
architecture-specific compile in this branch, so it does not call the result `compile-only`. That
skip names which fact it rests on: no GPUs on this host (picker exit 3) and GPUs present but none
idle (exit 1) are environment facts, while AMD SMI being unqueryable (exit 2) says nothing about the
GPUs at all.

When no GPU was claimed, the target is then asked whether it needs one: it is run once with no
visible device, and `test_selection.gpu_requirement` becomes `not-required` only if it **passes and
executes at least one test**. The executed count is what makes this evidence — a suite guarded by
`skipif(not torch.cuda.is_available())` also exits 0 while proving nothing. This is deliberately an
observation rather than a judgement about the diff: a Python-level dispatch change reroutes kernels
without touching kernel source, and ROCm/aiter#5089 decides whether 34 gfx950 kernels compile from a
seven-line helper, so no static rule over changed paths could settle it.

A `not-required` target runs its correctness stages instead of skipping them, which is the evidence
a CPU-only fix is able to supply. It still claims nothing further: `arch_coverage` stays empty
because only a passing `gpu_claim` credits an architecture, and `PASS` is unchanged — it continues
to require `gpu_claim: pass`, so this path cannot produce a clearance that was previously
unreachable.

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

Selection can still pick a runner the target cannot survive. A file that defines `test*` nodes
**and** parses argv in its module body is collected by pytest, which imports it with pytest's
own argv — and argparse exits the process, while the same file is green run as a script.
`test_selection.runner_risk` names that structurally, and a run that executes nothing under
the selected runner says so: *"red on both sides"* is an attribution, not an explanation, and a
reader who is not told otherwise concludes the code is broken when the runner choice is.

The report records `test_selection.runner` and `runner_reason`. A script target is profiled the
same way a pytest target is: the probe is installed by a validator-owned runner that then executes
the file under `runpy` with `run_name="__main__"`, so `execution_receipt` is reachable for both.
Nothing about `sys.setprofile` needed pytest; pytest was only where the hook was installed.

Both, and they are reported separately, because the interesting case is when they disagree.
Pytest runs emit JUnit XML and a zero-executed/all-skipped target is `skip`, never `pass`.

A script target publishes no per-case count, so its `executed` is a liveness signal and
nothing more — it says the process ran. What it must not do is stand in for work. A target
that returns silently with exit 0 and a log line (aiter#4538's does, when the arch is
unsupported or an optional package is missing) produced exactly the same `executed: 1` as
the run that graded 56 cases, and earned the same `arch_coverage: runtime` on basis
`script-exit-zero-with-output` — a statement about the process, not the kernel. So:

- `stats.observed_work` carries the only number backed by evidence: route calls counted in
  **that run's own** execution receipt. It is `null` when no route was named, because
  nothing was watched;
- `arch_coverage` credits nothing when `observed_work` is `0` — a route was watched for and
  never reached the device — and `arch_coverage_basis` prints the count and its provenance
  rather than implying a measurement.

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

The grid reaches the target through whichever channel that target actually reads, and the channel
must be proven structurally before it is used:

| channel | flag | proof the hook exists |
|---|---|---|
| environment variable | `--shape-env` | the source references `os.getenv(VAR)` / `os.environ[VAR]` |
| the target's own CLI flag | `--shape-arg` | the source passes that flag literal to `add_argument` |

Injecting through an env var only would have made this stage permanently inert for repositories
whose tests take shapes on the command line — a limit of the injector, not of the target. The flag
is named by the caller rather than guessed, because a wrong guess appends argv the target silently
ignores. Neither channel is trusted on the strength of the AST scan alone: the stage re-runs the
target with a deliberately invalid grid value and requires it to fail. A target that passes with
garbage shapes is not consuming the grid, so the stage is `skip`, never credited.

With no channel configured the stage is `skip` and the verdict is `INCONCLUSIVE`. This is a
positive control against reporting the same default test run twice under different stage names.

When the kernel exposes no shape override, the report says `repo-default-only` rather than
claiming coverage it does not have.

**A proven channel is not the same as added coverage.** A target that consumes the grid can
still be handed cells it already runs by default. On ROCm/aiter#4538 all three requested
shapes were in the target's own `--shapes` default list, so the "independent" grid re-ran a
strict subset of the repository run and the stage reported `pass` — the exact duplication
this stage exists to prevent, invisible in the report. The grid cells are therefore compared
against the target's own declared default for the same flag, and
`test_selection.grid_independence` is one of:

| value | meaning |
|---|---|
| `adds-coverage` | at least one cell is outside the target's own default |
| `duplicates-target-defaults` | every cell is already a default; a passing run is downgraded to `skip` and the verdict to `INCONCLUSIVE`, because a passing duplicate proves only what `correctness_repo_tests` already said |
| `unknown` | the channel exposes no literal default to compare against (every env-var channel, and a flag whose default is computed) |

A duplicate grid that **fails** keeps its `fail`. The finding is real; what a duplicate
cannot do is earn a pass.

#### Axes: when the failing configuration is not a shape

`--grid` is one ordered tuple on one channel, which is all a target's shape flag accepts. A
target whose remaining knobs are separate flags — head counts, dtypes, window modes — could
not be gridded over them at all, so entire configurations were unreachable however the grid
was spelled. That is not a missing shape; it is a missing axis.

aiter#4538 is the case: `--shapes` carries `(seq_len, seq_len_kv)` while `--num-heads` is its
own flag defaulting to `64 128`, and the public API asserts at `num_heads=16` — a real
blocker the validator had no way to request.

```
--axis 'num_heads=--num-heads:16;32'
```

Axes obey the same burden of proof as `--shape-arg`, in two steps, and
`test_selection.axis_state` names which one it reached:

| state | meaning |
|---|---|
| `none` / `unusable` | none requested, or the target is not a script (argv reaches script targets only) |
| `hook-not-found` | the target's source declares no `add_argument` for that flag |
| `hook-not-consumed` | the flag was declared but accepted `__VALIDATOR_INVALID_AXIS__`; the axis is **dropped and named**, never dropped quietly |
| `proven` | every axis flag refused an invalid value, and its values rode the grid run's argv |

Each axis records its own `independence` against the flag's declared default, on the same
terms as the shape cells. A proven axis that asks for values outside the default makes the
run independent even when the shape cells duplicate.

A requested axis appears in `test_selection.axes` **whatever** becomes of it, with
`hook_proof: not-evaluated` when the run never got far enough to scan for the flag. An empty
`axes` beside a non-`none` `axis_state` would lose the request itself — which is the silently
narrowed test space these fields exist to make visible. For the same reason
`grid_independence_reason` names which of the several ways the comparison can be skipped
actually applied, rather than defaulting to a claim about the target ("the channel exposes no
declared defaults") that the run never established.

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
its own receipt; `validate-kernel-pr.validation_probe` owns the receipt producer, and the script
runner calls that producer's own hooks rather than re-implementing them.

A receipt is validated whenever a route was named, including when no grid was configured or the
grid channel could not be established. With no grid it asserts route execution and nothing about
shapes, which is all it is then entitled to claim. Abandoning the receipt along with the grid
would discard evidence that was already collected.

**One receipt per run, not per phase.** `head-repo` and `head-grid` both execute inside the
head phase. They shared a receipt path, so the second erased the first, and with the grid
cells a subset of the target's defaults a receipt written by *either* run satisfied `--grid`
— which made the grid's own evidence unfalsifiable. Each run now writes
`execution-receipt-<label>.json`, the stage reads the grid run's own file when a grid ran,
and `execution_receipt.receipt_scope` names which run the published receipt describes.

**A route is resolved to a code object, not matched as a string.** A frame's identity is
`f_globals["__name__"] + ":" + f_code.co_name`, which stops being the declared route the
moment the function is wrapped: `functools.wraps` copies `__name__` onto the wrapper object
and leaves `co_name` alone, so aiter's entire `@compile_ops` family executes as
`aiter.jit.core:wrapper` and naming the op a reviewer cares about matched nothing. The probe
imports the declared module, resolves the attribute, walks its `__wrapped__` chain and matches
on the resulting code objects; the string match remains as a fallback, so a route into a module
that cannot be imported behaves exactly as before.

**Shape capture still needs the route's own frame to bind the shape locals.** A dispatch
wrapper declared `(*args, **kwargs)` binds none of them, so a route through one attests
execution and nothing about shapes; the receipt then says *"missing required shapes"* rather
than passing. Naming a route whose frame does carry the shape names is the caller's move.

**A route is not a variant.** The receipt records that
`…mqa_logits.fp8_mqa_logits:flydsl_fp8_mqa_logits` was entered and with which shape-locals.
It says nothing about which of that module's 30 registered gfx950 kernel variants the call
selected, so no variant-coverage claim is supportable from a receipt. Getting one needs the
variant to be a declared axis or a captured shape-local; see [Not implemented yet](#not-implemented-yet).

### 7 — `index_width_scan` (informational)

Runs `scan_index_width.py` over the diff and records the count of index×stride multiplies that
carry no 64-bit widening. Candidates, not verdicts — the reviewer judges each. See
[Why this stage exists](#why-the-index-width-scan-is-a-separate-stage).

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

A transplant spans two trees, so it carries one extra burden: `--perf-control-column` names a
timing column the patch does not touch — typically a reference implementation the target times
alongside the kernel under test — and the stage `skip`s unless that column reproduces within
`PERF_CONTROL_TOL` (default 10 %) across the two runs. Without a control the transplant is
declined rather than guessed at, and the reason names the missing flag instead of claiming
there was nothing to measure. Note that `median_ratio` remains the *worst* column, so an
unchanged reference column sitting at 1.0 caps the reported ratio; read `columns` for the
kernel's own movement.

On by default, because the regression this stage exists to catch is the one nobody suspected and an
opt-in flag is only ever set by someone who already suspects. The entry point is detected from the
target (`--scenario bench`, or a `perftest`/`@benchmark` harness); `--perf-args` names it explicitly
and `--no-perf` turns the stage off.

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
- **The process exit code comes from this run.** It is read from a verdict file written
  inside the run's own `$WORK`, and `--out` is deleted at startup. Deriving it by re-reading
  `--out` made a previous run's report a fallback source of truth: a run that died before
  `finish_report` exited on the *earlier* run's verdict.
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
- **External grid adapters.** Three channels now exist — `--shape-env`, `--shape-arg`, and
  `--shape-argnames` for pytest parametrization — and between them they reach the great majority
  of aiter's `op_tests`. What remains unreachable is a target whose shapes are none of those: a
  parametrized case whose single parameter is a **dict or object** rather than scalar cells, and
  a target that takes its shapes from a file or a fixture. The validator still does not accept an
  independently hashed `--extra-target`, because that harness would have to be bound without
  changing the PR diff hash or live-base identity. Such runs remain `INCONCLUSIVE`, and
  `test_selection.grid_channel_reason` says which of these applies.
- **External grid adapters.** A target that exposes no shape channel at all — neither an env var
  for `--shape-env`, a CLI flag for `--shape-arg`, nor a flag for `--axis` — still cannot be given
  a grid. The validator
  does not accept an independently hashed `--extra-target`, because that harness must be bound
  without changing the PR diff hash or live-base identity. Such runs remain `INCONCLUSIVE`.
- **Axes on the env-var and pytest channels.** `--axis` reaches script targets through argv only.
  A pytest target's extra knobs are `parametrize` argnames, which needs a different injector, and
  an env-var channel has no per-axis spelling to prove. Requesting an axis on either is
  `axis_state: unusable` with the reason, never a silent drop.
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

A new stage must be able to **fail on a seeded defect**. Before adding one, seed the defect it is
meant to catch, confirm the stage goes red and the clean baseline stays green, and record both in
the PR. A stage that has never been observed failing is not a check — it is decoration.
