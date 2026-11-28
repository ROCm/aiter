# Defects found in the skills this loop drives

Four reproducible defects in `.claude/skills/validate-kernel-pr/`, each observed while validating a
real PR. They matter more than ordinary bugs because **every one of them produces a signal that
reads as a defect in the PR under review**, and a reviewer who trusts the tool will report it to
the author.

Each entry states what was observed, the mechanism at the cited lines, and how it was confirmed.
Line numbers are from the skill as of the commit in `../VALIDATED-AGAINST.md`.

---

## 1. The validator dirties the worktree it is checking, then blames the dirt on the PR

**Observed.** A validation run reported `baseline_control: skip` with the note *"base run did not
leave a clean worktree; head tests were not run"*, which cascaded into `correctness_repo_tests`,
`correctness_s1_grid`, `execution_receipt` and `perf` all skipping. The verdict was `NEEDS_WORK`
with no correctness evidence at all — indistinguishable, to a reader, from a PR that broke isolation.

**Mechanism.** `validate_pr.sh` captures the ignored-file set at startup and compares it after the
base phase; a new ignored entry means the base run was not isolated. The only new entry was
`.claude/skills/validate-kernel-pr/__pycache__/pick-idle-gpu.cpython-312.pyc`, written by the
validator's own GPU picker: lines 446 and 629 import `pick-idle-gpu.py` through
`importlib.spec_from_file_location` + `exec_module` **without** `PYTHONDONTWRITEBYTECODE=1`, while
the script does set that variable on its own test invocations (lines 731, 765, 815, 824).

**Confirmed by.** Injecting `PYTHONDONTWRITEBYTECODE=1` into the environment of the run — with no
other change — made every stage execute normally. `AITER_JIT_DIR` was ruled out first: it is
already redirected to `$WORK` (lines 200, 730, 814, 1767, 2095), so JIT output was never the cause.

**Fix.** Set `PYTHONDONTWRITEBYTECODE=1` (or `PYTHONPYCACHEPREFIX`) around those two imports.
Editing `validate_pr.sh` from outside is not a workaround: the file is tracked, so touching it
dirties the very worktree the run is about to check.

---

## 2. `test_policy` compares tolerances by position after discarding their names

**Observed.** `test_policy: fail — comparison tolerance widened [[0.1, 0.5]] while kernel code also
changed`. Reported as a `should-fix`, it made the run `NEEDS_WORK`. It reached a review card as a
🔴 before being withdrawn.

**Mechanism.** Lines 897–918 extract tolerances with
`re.findall(r'(?:atol|rtol)\s*=\s*([0-9.eE+-]+)', src)` — the `atol=`/`rtol=` keyword is captured
and then **thrown away**. Lines 922–932 zip the base and head lists positionally and keep any pair
where the head value is larger. Nothing matches by name.

The PR under test kept three call sites written `rtol=, atol=` and added a helper written
`atol=, rtol=`. Position 5 therefore compared the base's **rtol** `0.1` against the head's **atol**
`0.5` and declared a widening. Like for like, every tolerance was unchanged or tighter: bf16 rtol
`0.1 → 0.03`, fp32 rtol `0.1 → 1e-3`, atol unchanged at `0.5` for bf16 and `0.5 → 0.05` for fp32.

**Confirmed by.** Reading all six literals on each side with their owning call, and checking the
arithmetic: bf16 unit roundoff is `2^-8 ≈ 0.0039`, so the head's `rtol=0.03` is ~7.7× the
single-rounding bound and 3.3× *tighter* than the base's `0.1`.

**Fix.** Pair by name (`(name, value)`), not by position. Note also that the comparison only runs
when the two lists have equal length (lines 923–927), so today it silently passes whenever a diff
changes the number of tolerance literals — a second reason to key by name.

---

## 3. The same regex sees only the first literal of a conditional tolerance

**Observed.** In the same run, the head's fp32 tolerances never entered the comparison at all.

**Mechanism.** The extractor takes one numeric literal per `atol=`/`rtol=` occurrence. For
`atol = 0.5 if actual.dtype == torch.bfloat16 else 0.05` it captures `0.5` and never sees `0.05`;
likewise `rtol = 0.03 if ... else 1e-3` yields only `0.03`. A file that expresses tolerances as
conditionals is therefore compared on a subset of its own values.

**Fix.** Parse the assignment with `ast` rather than a regex, or capture every numeric literal in
the right-hand side.

---

## 4. `scrape_perf.py` reports "no common timing column" on logs that do have one

**Observed.** A perf stage note: *"14 row(s) matched but no timing column is common to both logs"*,
on a run whose base and head logs both carried timings.

**Mechanism.** `scrape_perf.py:383-386` emits that message when its column-intersection step comes
up empty. Row identity and column identity are derived from header text; bench tables in this repo
print unlabeled measurement columns, which the relaxed row key already has to work around (the
`row_key_basis` field exists for exactly that reason). The column side has no equivalent fallback.

**Confirmed by.** An independent reading of both logs, which contained comparable timing columns
under headers the intersection did not match.

**Fix.** Apply the same relaxation on the column axis that `row_key_basis` applies on the row axis,
and report the column names from each side in the note so the failure is diagnosable from the
report alone.

---

## Two adjacent observations (not defects, but they mislead)

**`-x` under-reports the failure surface.** The correctness stage runs pytest with `-x`, so a
failure hides every case that would have run after it. One run reported 64 executed with 1 failure;
the same file without `-x` executed 70 with 4 failures, and the extra three changed the diagnosis
(the failure was specific to one kernel id, not to one parameter). A report should either say that
execution stopped early, or the stage should re-run without `-x` to characterise a failure.

**The C++ launch check cannot distinguish "not a workspace kid" from "not compiled in this build".**
`opus_gemm.cu`'s non-workspace branch raises `non-workspace kid N requires workspace=None` for a kid
that is simply absent from the generated table for that build. Checking the non-workspace table
before raising would turn a misleading message into an accurate one — this cost a day of root-cause
work on a real failure.
