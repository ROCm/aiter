---
name: review-pr
description: Advisory AI code review for aiter and FlyDSL PRs. Catches perf regressions, silent correctness bugs, dispatch gate holes, and AI-generated code patterns, but never acts as a merge gate. Invoke with a PR number (optionally owner/repo#N) and, when one exists, a validation report path. Step 1 triages whether the PR changes runtime surface at all and, when it does and the PR ships a single test target, runs validate-kernel-pr itself; a PR with no runtime surface is reported N/A rather than unvalidated. That run also times the target on base and head back to back on one locked GPU, so a kernel PR's latency is measured rather than assumed. The review line stays advisory; deterministic correctness and perf results are judged only from a head-matched report.
argument-hint: <PR number> [owner/repo] [validation-report]
---

# aiter PR Review — advisory tier

This skill supplies hints to a human reviewer. Its judgement is stochastic and never blocks a
merge. Only a reproducible blocker from an explicitly supplied, head-matched
`validation_report.json` may be used as a deterministic gate.

## Promotion bar

This skill is advisory now and stays advisory until both conditions below hold. Neither holds today,
so no part of it may gate a merge.

- **False clearance is measured and near zero for every family that raises a red verdict.** The
  number that matters is not recall, and not a spot check: it is the rate at which the tool reports
  nothing wrong when something is wrong. No committed replay corpus establishes it, so that number
  does not currently exist.
- **The judgement relied on is not an LLM's.** An LLM judgement never gates a merge, whatever its
  measured accuracy. Only a reproducible blocker carried by a head-matched `validation_report.json`
  may gate, because the report ships its reproducer with it.

Until then `🔴 HIGH RISK` requests human attention and nothing more. Whoever proposes a rule edit as
an improvement, or proposes letting this tool gate a merge, owns building the corpus and measuring
against it. This bar lives in the header rather than in an issue because a header is read on every
use and an issue sinks.

---

## Step 1 — Fetch

```bash
set -euo pipefail

# Per-invocation scratch dir. Fixed /tmp paths collide: two reviews running at once
# overwrite each other's pr.diff between the write and the read, and the second review
# silently analyses the first one's diff under its own PR number. Observed.
WORK=$(mktemp -d /tmp/review-pr-XXXXXX)
PROJECT_ROOT=$(git rev-parse --show-toplevel) || {
  echo "review-pr must run inside the repository that owns .claude/skills" >&2
  exit 1
}
SKILLS_ROOT="$PROJECT_ROOT/.claude/skills"

PR=$1  # PR number from skill argument
# Second argument, or a PR given as owner/repo#N, selects the repository. FlyDSL kernels
# are reviewed from their own repo, so this must not be hard-coded to aiter.
REPO="${2:-ROCm/aiter}"
VALIDATION_REPORT="${3:-}"
case "$1" in
  */*#*)
    REPO="${1%#*}"
    PR="${1##*#}"
    VALIDATION_REPORT="${2:-}"
    ;;
esac

# Full metadata
gh pr view "$PR" --repo "$REPO" \
  --json title,body,number,labels,files,author,reviews,comments,baseRefName,headRefOid \
  > "$WORK/pr_meta.json"

# Diff
gh pr diff "$PR" --repo "$REPO" > "$WORK/pr.diff"

# Current base branch tip. PR metadata's base OID can remain the historical merge base after
# main advances, so it is not sufficient for stale merge-simulation detection.
BASE_REF=$(python3 -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["baseRefName"])' \
  "$WORK/pr_meta.json")
BASE_REF_PATH=$(python3 -c \
  'import sys,urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' \
  "$BASE_REF")
gh api "repos/$REPO/branches/$BASE_REF_PATH" --jq .commit.sha > "$WORK/base_head.txt"

# Linked issue (extract from body "fix: #NNN" or "close #NNN")
ISSUE=$(cat "$WORK/pr_meta.json" | python3 -c "
import json,re,sys
body = json.load(sys.stdin).get('body','') or ''
m = re.search(r'(?:fix|close|resolve)[s]?[: ]*#(\d+)', body, re.I)
print(m.group(1) if m else '')
")
[ -n "$ISSUE" ] && gh issue view "$ISSUE" --repo "$REPO" --json title,body > "$WORK/pr_issue.json"

# Prior reviewer comments (top-level)
cat "$WORK/pr_meta.json" | python3 -c "
import json,sys
d = json.load(sys.stdin)
for r in d.get('reviews',[]):
    b = (r.get('body','') or '').strip()
    if b: print(f'[REVIEW {r[\"author\"][\"login\"]}] {b[:200]}')
for c in d.get('comments',[]):
    b = (c.get('body','') or '').strip()
    if b: print(f'[COMMENT {c[\"author\"][\"login\"]}] {b[:200]}')
"

# Mechanical pre-filter for rule D9 (index x stride with no 64-bit widening).
# It runs HERE, inside the fetch step, and not where D9 is described in Step 5. A scan that
# Step 5 asks for mid-checklist does not happen: in a 14-PR controlled run the revised D9 text
# caught 0 of 3 known overflow defects and no run invoked the scanner at all. Put the candidate
# list in context before the rule pass instead of relying on the reviewer to remember it.
SCAN="$SKILLS_ROOT/validate-kernel-pr/scan_index_width.py"
if [ ! -x "$SCAN" ]; then
  echo "required scanner is missing or not executable: $SCAN" >&2
  exit 1
fi
# The scan is an AST pass, so it needs each changed file's post image. The diff names the
# post-image blob of every hunk, and fetching the PR head puts those blobs in the local object
# store, where `git cat-file` reaches them without a second worktree. Without this the scan
# still runs, but reports the files as NOT SCANNED rather than silently reporting no findings.
git fetch -q origin "refs/pull/$PR/head" 2>/dev/null || \
  echo "note: could not fetch the PR head; the index-width scan will report unscanned files" >&2
if ! "$SCAN" --diff "$WORK/pr.diff"; then
  echo "required index-width scan failed; do not report an empty candidate list" >&2
  exit 1
fi
# The candidates above cannot be judged without deployment scale, and the scale facts are
# useless 400 lines away from them -- print both together.
SCALE="$SKILLS_ROOT/validate-kernel-pr/production_scale.md"
if [ ! -r "$SCALE" ]; then
  echo "required production-scale evidence is missing: $SCALE" >&2
  exit 1
fi
cat "$SCALE"
SCHEMA="$SKILLS_ROOT/validate-kernel-pr/report_schema.json"
if [ ! -r "$SCHEMA" ]; then
  echo "required validation report schema is missing: $SCHEMA" >&2
  exit 1
fi

# Triage: decide whether this PR needs runtime evidence at all, and if so, what could carry it.
# This runs HERE, executable, for the same reason the D9 scan does: a judgement the checklist
# asks for mid-review is a judgement that does not get made. Leaving it to the reviewer also
# gave the verdict line only two states, so a README fix reported the same
# "NOT RUN" as an unvalidated kernel rewrite -- which teaches a reader to skip the line.
python3 - "$WORK/pr_meta.json" "$WORK/pr.diff" "$WORK/validation_requirement.json" \
  "$PROJECT_ROOT" <<'PY'
import json
import pathlib
import sys

meta_path, diff_path, out_path, project_root = (pathlib.Path(a) for a in sys.argv[1:5])
meta = json.loads(meta_path.read_text())
paths = [f["path"] for f in meta.get("files", [])]

# Checked in this order: a .md under csrc/ is documentation, and op_tests/ is neither
# runtime nor documentation.
TEST_PREFIXES = ("op_tests/",)
STATIC_PREFIXES = (".github/", ".claude/", ".cursor/", "bin/", "docs/")
STATIC_SUFFIXES = (".md", ".rst", ".txt", ".toml", ".cfg", ".ini", ".gitignore")
RUNTIME_PREFIXES = ("csrc/", "hsa/", "aiter/", "gradlib/")
RUNTIME_SUFFIXES = (".cu", ".cuh", ".hip", ".cpp", ".cc", ".c", ".h", ".hpp", ".s", ".asm")


def classify(path):
    if path.startswith(TEST_PREFIXES):
        return "test"
    if path.startswith(STATIC_PREFIXES) or path.endswith(STATIC_SUFFIXES):
        return "static"
    if path.startswith(RUNTIME_PREFIXES) or path.endswith(RUNTIME_SUFFIXES):
        return "runtime"
    # Unclassified counts as runtime, deliberately. The error to avoid is clearing a kernel
    # change because nobody put its directory on a list; demanding evidence that turns out
    # to be unnecessary only costs a run. Note this makes `aiter/configs/*.csv` tuned-shape
    # edits and `aiter/jit/**` build config runtime, which is correct: both reroute kernels.
    return "runtime"


runtime_paths = sorted(p for p in paths if classify(p) == "runtime")
required = bool(runtime_paths)

# Added files come from the diff, not from the file list: `gh pr view --json files` reports
# additions/deletions per path but not status, and "deletions == 0" also matches a file that
# was only appended to.
added = set()
current = None
for line in diff_path.read_text(errors="replace").splitlines():
    if line.startswith("diff --git ") and " b/" in line:
        current = line.split(" b/", 1)[1]
    elif line.startswith("new file mode") and current:
        added.add(current)


def is_candidate_target(path):
    name = pathlib.PurePosixPath(path).name
    # op_benchmarks/ holds bench_*.py perf harnesses. They are excluded because they are
    # not correctness targets; the validator's perf stage times the correctness target it
    # already selected, so it does not need one of these either.
    return (
        path.startswith("op_tests/")
        and "/op_benchmarks/" not in f"/{path}"
        and name.startswith("test_")
        and name.endswith(".py")
    )


candidates = sorted(p for p in paths if is_candidate_target(p))
added_candidates = [p for p in candidates if p in added]
target = None
basis = None
blocker = None
if not required:
    blocker = "no runtime surface changed"
elif len(candidates) == 1:
    target, basis = candidates[0], "the one test target this PR touches"
elif len(added_candidates) == 1:
    target, basis = added_candidates[0], "the one test target this PR adds"
elif candidates:
    blocker = (
        f"{len(candidates)} candidate targets and no unique added one; "
        "name the target explicitly rather than letting the tool pick"
    )
else:
    blocker = "the PR changes runtime code but ships no test target"

# ---- Perf triage. A kernel change can be correct and still be a regression --
# correctness_repo_tests passing says nothing about latency. So the same runtime surface that
# makes correctness evidence REQUIRED makes perf evidence REQUIRED too, and this decides it
# here for the reason everything else in this step is executable: in a controlled run the
# reviewer with a perf-shaped PR in front of it shipped a card with no numbers at all and a
# finding that stopped at "reviewer should ask" (aiter#4538).
#
# The measurement itself belongs to the validator's `perf` stage, which times base and head
# back to back on one locked GPU and gates on the result. What is computed HERE is only the
# fallback: which command a human would run if that stage could not. Keep the detection below
# in step with perf_detect() in validate-kernel-pr/validate_pr.sh -- if the two disagree, this
# step prints a recipe for a harness the validator declined to use, or vice versa.
#
# `perf_claimed` only separates two reports ("the PR's own claim is unverified" vs "no claim
# was made, but a kernel moved"). It does NOT gate the requirement: a refactor that claims
# nothing is exactly where an unnoticed regression lands.
PERF_WORDS = (
    "perf", "optimiz", "optimis", "faster", "speedup", "speed-up", "latency",
    "throughput", "tflops", "fuse", "regression", "us/", "µs",
)
_blurb = f"{meta.get('title', '')}\n{meta.get('body', '') or ''}".lower()
perf_claimed = any(w in _blurb for w in PERF_WORDS)

# A perf harness cannot be inferred from the diff alone, so look for aiter's three timing
# conventions in the target's own text: `--scenario bench` (argparse sweep), the
# @benchmark/run_perftest pair the aiter-op-test skill mandates, and the older bare `perftest`
# decorator. A new target's body is in the diff as `+` lines; an
# existing one is read from the checkout, whose revision may differ but whose entry points do
# not. Anything else means the PR ships no runnable perf harness -- which is itself the
# finding, not a reason to stay silent.
def perf_command(path):
    text = ""
    local = project_root / path
    if local.is_file():
        text = local.read_text(errors="replace")
    if not text:
        want = f" b/{path}"
        keep = False
        for line in diff_path.read_text(errors="replace").splitlines():
            if line.startswith("diff --git "):
                keep = line.endswith(want)
            elif keep and line.startswith("+") and not line.startswith("+++"):
                text += line[1:] + "\n"
    if not text:
        return None, "the target's contents could not be read"
    if "--scenario" in text and "bench" in text:
        return f"python3 {path} --scenario bench", "target exposes --scenario bench"
    # `perftest`, not `run_perftest`. aiter carries three timing conventions and the bare
    # `perftest` decorator is one of them; `perftest` also subsumes `run_perftest` as a
    # substring, so the shorter test is strictly wider. Measured on the 123 targets in
    # op_tests/: matching `run_perftest` detects 85 and reports 38 as having no harness;
    # matching `perftest` detects 97 and reports 26. 11 of the 12 recovered targets have
    # live `perftest` usage -- test_moe.py, test_pa_v1.py, test_batch_prefill.py,
    # test_rope.py, test_layernorm2d.py among them. Reporting those as "no benchmark entry
    # point" reads as "there was nothing to measure" when the truth was that the detector
    # was too narrow, which is the exact silence this rule exists to break.
    #
    # The 12th (test_aiter_sigmoid.py) matches only a commented-out import, because this is
    # a substring test and not a parse. That direction is the safe one: an over-eager match
    # runs the target, finds no timing table, and the perf stage reports `skip` -- the same
    # outcome as not matching, one wasted run later. The opposite error stays silent.
    # Keep this in step with perf_detect() in validate-kernel-pr/validate_pr.sh.
    if "perftest" in text or "@benchmark" in text:
        return f"python3 {path}", "target uses the perftest/@benchmark harness"
    return None, (
        "target exposes no benchmark entry point "
        "(no --scenario bench, no perftest/@benchmark harness)"
    )


perf_cmd = perf_reason = None
if target:
    perf_cmd, perf_reason = perf_command(target)
elif required:
    perf_reason = f"no perf target for the same reason there is no test target: {blocker}"
else:
    perf_reason = "no runtime surface changed"

# A target is never inferred from the *kernel* being changed, only from a test the PR itself
# touches. The validator cannot judge whether a target exercises the diff, so an invented
# target can return PASS on evidence about unrelated code -- worse than no report, because
# it reads as clearance.
out_path.write_text(
    json.dumps(
        {
            "required": required,
            "families": sorted({classify(p) for p in paths}),
            "runtime_paths": runtime_paths[:20],
            "runtime_path_count": len(runtime_paths),
            "target": target,
            "target_basis": basis,
            "candidates": candidates,
            "blocking_reason": blocker,
            "perf_required": required,
            "perf_claimed": perf_claimed,
            "perf_command": perf_cmd,
            "perf_basis": perf_reason,
        },
        indent=2,
    )
    + "\n"
)
verdict = "REQUIRED" if required else "NOT REQUIRED"
print(f"validation triage: {verdict} ({', '.join(sorted({classify(p) for p in paths}))})")
if runtime_paths:
    print(f"  runtime surface: {len(runtime_paths)} path(s), e.g. {', '.join(runtime_paths[:3])}")
print(f"  target: {target} — {basis}" if target else f"  no auto target: {blocker}")
print(f"perf triage: {verdict}" + (" (PR claims perf)" if perf_claimed else " (no perf claim)"))
if perf_cmd:
    print(f"  harness: {perf_reason}")
    print("  the validator's perf stage runs this on both sides automatically; the form below")
    print("  is the manual fallback for when stages.perf comes back absent or skip. Run BOTH")
    print("  sides on this box, same GPU, back to back -- head alone only reproduces the PR's")
    print("  own comparison and cannot show a regression:")
    print("    git worktree add --detach $WORK/perf_base $(cat $WORK/base_head.txt)")
    print(f"    (cd $WORK/perf_base && {perf_cmd})                              # base")
    print(f"    (cd $WORK/perf_base && git apply $WORK/pr.diff && {perf_cmd})   # head")
else:
    print(f"  no perf run possible: {perf_reason}")
PY

# Validation is opt-in in the sense that matters: a report is never adopted because it happens
# to be lying in the working directory. A stale report from another PR is worse than none, so
# every report -- supplied by the caller or produced by the auto-run below -- goes through the
# same identity gate, and reports that do not name this exact head are rejected.
#
# Auto-running the validator is not the same act as trusting a found file. The run below binds
# its own report to this head, writes it inside this invocation's scratch dir, and then faces
# the unmodified gate. Set REVIEW_AUTO_VALIDATE=0 to skip it.
#
# Each way the auto-run can give up records why, because "required but not run" is only useful
# to a reader if it names the reason. Triage's own blocker covers the cases where no run was
# ever attempted; this file covers the ones where it was.
if [ "${REVIEW_AUTO_VALIDATE:-1}" != 1 ]; then
  echo "auto-validation is disabled (REVIEW_AUTO_VALIDATE=0)" \
    >"$WORK/auto_validation_outcome.txt"
fi
if [ -z "$VALIDATION_REPORT" ] \
  && [ "${REVIEW_AUTO_VALIDATE:-1}" = 1 ] \
  && python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r["required"] and r["target"] else 1)' \
    "$WORK/validation_requirement.json"; then
  AUTO_TARGET=$(python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["target"])' \
    "$WORK/validation_requirement.json")
  # The validator is invoked directly, with the base and head this step already holds. A
  # PR-number front end that re-fetched the diff and re-resolved the base tip would reopen the
  # window the gate below closes: main can advance between the two `gh api` calls, and the report
  # would then name a base this same review rejects as stale. Calling it in place also keeps the
  # dependency inside .claude/skills, alongside the scanner and the schema above.
  VALIDATOR="$SKILLS_ROOT/validate-kernel-pr/validate_pr.sh"
  if [ ! -x "$VALIDATOR" ]; then
    echo "required validator is missing or not executable: $VALIDATOR" >&2
    exit 1
  fi
  # repo.base in the report is `git rev-parse HEAD` inside the worktree handed over, so that
  # worktree has to sit on the base recorded above for the two to agree.
  AUTO_BASE=$(cat "$WORK/base_head.txt")
  AUTO_HEAD=$(python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["headRefOid"])' \
    "$WORK/pr_meta.json")
  AUTO_WT="$WORK/base_repo"
  # A PR reviewed from another repository resolves a base this checkout has never seen, so
  # failing to materialise it leaves the review static-only rather than aborting it.
  set +e
  git -C "$PROJECT_ROOT" cat-file -e "${AUTO_BASE}^{commit}" 2>/dev/null \
    || git -C "$PROJECT_ROOT" fetch -q origin "$AUTO_BASE" 2>/dev/null
  git -C "$PROJECT_ROOT" worktree add --detach "$AUTO_WT" "$AUTO_BASE" >/dev/null 2>&1
  AUTO_WT_RC=$?
  set -e
  if [ "$AUTO_WT_RC" -ne 0 ]; then
    echo "base $AUTO_BASE cannot be checked out in $PROJECT_ROOT, which is expected for a PR" \
      "reviewed from another repository" >"$WORK/auto_validation_outcome.txt"
    echo "auto-validation skipped: $(cat "$WORK/auto_validation_outcome.txt")" >&2
  else
    # Removal is on a trap because a run interrupted mid-validation would otherwise leave the
    # worktree registered. The validator reverts its candidate patch on every exit path, so
    # this disposes of the review's own scratch checkout and is not a dirty-tree repair.
    remove_auto_worktree() {
      git -C "$PROJECT_ROOT" worktree remove --force "$AUTO_WT" >/dev/null 2>&1 || true
      git -C "$PROJECT_ROOT" worktree prune
    }
    trap remove_auto_worktree EXIT
    # Route and shape knowledge cannot be derived from a diff, so without these the receipt
    # and grid stages skip and the run tops out at INCONCLUSIVE by construction. That is a
    # limit of what a diff tells you, not a defect in the PR -- Step 8 must say so.
    # When a grid is supplied and no channel carries it, the report's
    # test_selection.grid_channel_reason names each channel tried, what was found in the
    # target, and which channels the target does offer -- so a wrong guess costs one run
    # rather than a reading of the target's source.
    AUTO_ARGS=()
    [ -n "${REVIEW_EXPECTED_ROUTE:-}" ] && AUTO_ARGS+=(--expected-route "$REVIEW_EXPECTED_ROUTE")
    [ -n "${REVIEW_SHAPE_VARS:-}" ] && AUTO_ARGS+=(--shape-vars "$REVIEW_SHAPE_VARS")
    [ -n "${REVIEW_SHAPE_ENV:-}" ] && AUTO_ARGS+=(--shape-env "$REVIEW_SHAPE_ENV")
    [ -n "${REVIEW_SHAPE_ARG:-}" ] && AUTO_ARGS+=(--shape-arg "$REVIEW_SHAPE_ARG")
    # The pytest-parametrization channel reaches targets neither of the other two can: none of
    # the seven files in op_tests/flydsl_tests/ reads a shape env var or parses a shape flag,
    # and all of them declare shapes as literals in @pytest.mark.parametrize. Without this the
    # channel exists but no auto-validated review can use it.
    [ -n "${REVIEW_SHAPE_ARGNAMES:-}" ] \
      && AUTO_ARGS+=(--shape-argnames "$REVIEW_SHAPE_ARGNAMES")
    [ -n "${REVIEW_GRID:-}" ] && AUTO_ARGS+=(--grid "$REVIEW_GRID")
    echo "auto-validation: running $AUTO_TARGET for PR #$PR (minutes, needs an idle GPU)"
    # BLOCK, NEEDS_WORK and INCONCLUSIVE all still write a report worth consuming, so the
    # exit code must not abort the review; only a missing file means there is nothing to read.
    set +e
    "$VALIDATOR" --repo "$AUTO_WT" --patch "$WORK/pr.diff" --head-sha "$AUTO_HEAD" \
      --target "$AUTO_TARGET" --label "review-pr-auto" \
      --out "$WORK/auto_validation_report.json" "${AUTO_ARGS[@]}"
    AUTO_RC=$?
    set -e
    if [ -r "$WORK/auto_validation_report.json" ]; then
      VALIDATION_REPORT="$WORK/auto_validation_report.json"
      echo "auto-validation exited $AUTO_RC; consuming its report through the standard gate"
    else
      echo "the validator exited $AUTO_RC without writing a report" \
        >"$WORK/auto_validation_outcome.txt"
      echo "auto-validation exited $AUTO_RC and wrote no report; the review stays static-only" >&2
    fi
  fi
fi

if [ -n "$VALIDATION_REPORT" ]; then
  python3 - "$WORK/pr_meta.json" "$WORK/base_head.txt" "$WORK/pr.diff" "$SCHEMA" \
    "$VALIDATION_REPORT" "$WORK/validation_report.json" <<'PY'
import hashlib
import json
import pathlib
import sys

meta_path, base_path, diff_path, schema_path, report_path, out_path = map(
    pathlib.Path, sys.argv[1:]
)
meta = json.loads(meta_path.read_text())
report = json.loads(report_path.read_text())
try:
    import jsonschema
except ImportError as error:
    raise SystemExit(f"jsonschema is required to consume validation evidence: {error}")
jsonschema.validate(report, json.loads(schema_path.read_text()))
expected_head = meta["headRefOid"]
actual_head = report.get("repo", {}).get("head")
if actual_head != expected_head:
    raise SystemExit(
        "validation report is stale or for another checkout: "
        f"expected head {expected_head}, got {actual_head}"
    )
expected_base = base_path.read_text().strip()
actual_base = report.get("repo", {}).get("base")
if actual_base != expected_base:
    raise SystemExit(
        "validation report used a stale merge base: "
        f"expected {expected_base}, got {actual_base}"
    )
expected_patch = hashlib.sha256(diff_path.read_bytes()).hexdigest()
actual_patch = report.get("repo", {}).get("patch_sha256")
if actual_patch != expected_patch:
    raise SystemExit(
        "validation report patch does not match the current PR diff: "
        f"expected {expected_patch}, got {actual_patch}"
    )
required_stages = {
    "merge_sim",
    "gpu_claim",
    "runtime_compat",
    "test_policy",
    "baseline_control",
    "correctness_repo_tests",
    "correctness_s1_grid",
    "execution_receipt",
    "index_width_scan",
}
missing = required_stages - report.get("stages", {}).keys()
if missing:
    raise SystemExit(f"validation report omits required stages: {sorted(missing)}")
for name in required_stages:
    stage = report["stages"][name]
    if not isinstance(stage, dict) or stage.get("status") not in {
        "pass",
        "fail",
        "skip",
        "info",
    }:
        raise SystemExit(f"validation stage {name} is malformed: {stage!r}")
if report.get("verdict") not in {"PASS", "NEEDS_WORK", "BLOCK", "INCONCLUSIVE"}:
    raise SystemExit(f"validation report has an invalid verdict: {report.get('verdict')!r}")
findings = report.get("findings")
if not isinstance(findings, list):
    raise SystemExit("validation report findings must be a list")
for finding in findings:
    if (
        not isinstance(finding, dict)
        or finding.get("severity") not in {"blocker", "should-fix", "note"}
        or not finding.get("stage")
        or not finding.get("detail")
    ):
        raise SystemExit(f"validation report has a malformed finding: {finding!r}")
selection = report.get("test_selection", {})
if not selection.get("target"):
    raise SystemExit("validation report does not name the test target it selected")
if not selection.get("expected_route"):
    raise SystemExit("validation report does not name the expected kernel route")
if not selection.get("shape_vars"):
    raise SystemExit("validation report does not name the route-call shape variables")
if selection.get("runner") not in {"pytest", "script", "none", "unresolved"}:
    raise SystemExit("validation report has no supported target runner")
if not selection.get("runner_reason"):
    raise SystemExit("validation report does not explain its runner selection")
identity = report.get("runtime_identity")
if (
    not isinstance(identity, dict)
    or not identity.get("module_path")
    or not identity.get("python_executable")
    or not isinstance(identity.get("native_artifacts"), list)
):
    raise SystemExit("validation report has no runtime build identity")
coverage = report.get("arch_coverage", {})
coverage_basis = report.get("arch_coverage_basis", {})
if any(arch not in coverage_basis for arch, level in coverage.items() if level == "runtime"):
    raise SystemExit("runtime architecture coverage has no evidence basis")
gpu_arch = report["stages"]["gpu_claim"].get("arch")
if coverage and coverage != {gpu_arch: "runtime"}:
    raise SystemExit("runtime architecture coverage does not match the claimed GPU")
if set(coverage_basis) != set(coverage):
    raise SystemExit("architecture coverage and evidence-basis keys differ")
if coverage:
    basis = coverage_basis[gpu_arch]
    if selection["runner"] == "pytest":
        grid_stats = report["stages"]["correctness_s1_grid"].get("stats", {})
        basis_stage = (
            report["stages"]["correctness_s1_grid"]
            if grid_stats.get("executed", 0) > 0
            else report["stages"]["correctness_repo_tests"]
        )
        expected_basis = (
            f"pytest-junit-executed:{basis_stage.get('stats', {}).get('executed', 0)}"
        )
        if basis != expected_basis:
            raise SystemExit("pytest architecture coverage basis is inconsistent")
    elif selection["runner"] == "script" and not basis.startswith("script-"):
        raise SystemExit("script architecture coverage basis is inconsistent")
for stage_name in ("correctness_repo_tests", "correctness_s1_grid"):
    stage = report["stages"][stage_name]
    stats = stage.get("stats")
    if stats is not None:
        stat_keys = ("tests", "failures", "errors", "skipped", "executed")
        if any(type(stats.get(key)) is not int for key in stat_keys):
            raise SystemExit(f"{stage_name} has malformed execution counters")
        # errors are collection and fixture failures: the test body never ran, so they are
        # not executed tests. The two sides of this equality disagreed only when errors > 0 --
        # which is exactly the shape of a report carrying a real runtime blocker, so a report
        # that had correctly found one was rejected here and could not be used.
        if stats["executed"] != stats["tests"] - stats["skipped"] - stats["errors"]:
            raise SystemExit(f"{stage_name} has inconsistent execution counters")
    elif stage["status"] == "pass":
        raise SystemExit(f"{stage_name} passed without execution counters")
    if stage["status"] == "pass" and (
        stage.get("exit") != 0
        or stats.get("executed", 0) < 1
        or stats.get("failures", 0) != 0
        or stats.get("errors", 0) != 0
    ):
        raise SystemExit(f"{stage_name} has a hollow or contradictory pass")
receipt = report["stages"]["execution_receipt"]
# Only a grid that was actually DELIVERED imposes required shapes. A grid the caller supplied
# for a target with no channel to receive it was still being turned into a requirement here,
# so the receipt's honest empty list read as a contradiction and the report was rejected --
# discarding exactly the runs that carried the accurate "no channel" diagnostic.
required_shapes = (
    [shape.strip() for shape in selection.get("grid", "").split(";") if shape.strip()]
    if selection.get("grid_channel")
    else []
)
if receipt.get("status") == "pass" and (
    receipt.get("producer") != "validate-kernel-pr.validation_probe"
    or receipt.get("route") != selection["expected_route"]
    or selection["expected_route"] not in receipt.get("kernel_symbols", [])
    or sorted(set(receipt.get("required_shapes", []))) != sorted(set(required_shapes))
    or (
        selection.get("grid_channel") != "pytest"
        and not set(required_shapes).issubset(set(receipt.get("executed_shapes", [])))
    )
):
    raise SystemExit("execution receipt contradicts the selected route/grid")
severities = {
    finding.get("severity")
    for finding in findings
}
complete = (
    selection["runner"] in {"pytest", "script"}
    and bool(coverage)
    and bool(coverage_basis)
    and report["stages"]["merge_sim"]["status"] == "pass"
    and report["stages"]["gpu_claim"]["status"] == "pass"
    and report["stages"]["runtime_compat"]["status"] == "pass"
    and report["stages"]["test_policy"]["status"] == "pass"
    and report["stages"]["baseline_control"]["status"] == "pass"
    and report["stages"]["correctness_repo_tests"]["status"] == "pass"
    and report["stages"]["correctness_s1_grid"]["status"] == "pass"
    and report["stages"]["execution_receipt"]["status"] == "pass"
    and report["stages"]["index_width_scan"]["status"] == "info"
)
expected_verdict = (
    "BLOCK"
    if "blocker" in severities
    else "NEEDS_WORK"
    if "should-fix" in severities
    else "PASS"
    if complete
    else "INCONCLUSIVE"
)
if report["verdict"] != expected_verdict:
    raise SystemExit(
        "validation verdict contradicts its stages/findings: "
        f"expected {expected_verdict}, got {report['verdict']}"
    )
expected_exit = 0 if expected_verdict == "PASS" else (
    2 if expected_verdict == "INCONCLUSIVE" else 1
)
if report.get("process_exit_code") != expected_exit:
    raise SystemExit(
        "validation report exit-code contract is inconsistent: "
        f"expected {expected_exit}, got {report.get('process_exit_code')}"
    )
# perf is optional -- a report without it is valid, and older reports have none. But a perf
# stage that ASSERTS an outcome has to carry the number it was drawn from, or the card would
# print "NO REGRESSION" backed by nothing and a reader could not tell the difference.
perf = report["stages"].get("perf")
if perf is not None:
    if perf["status"] in {"pass", "fail"}:
        if not isinstance(perf.get("median_ratio"), (int, float)):
            raise SystemExit("perf stage claims a result without a median_ratio")
        if not isinstance(perf.get("matched_rows"), int) or perf["matched_rows"] < 1:
            raise SystemExit("perf stage claims a result with no matched rows")
        if not perf.get("baseline"):
            raise SystemExit("perf stage claims a result without naming its baseline")
        # The status has to agree with the number it is standing on. Without this a report
        # can carry median_ratio 0.80 against threshold 0.95 and still say `pass`, and the
        # card would print "NO REGRESSION" over the top of a measured 20% regression --
        # every other check here passes, because each field is individually well-formed.
        threshold = perf.get("threshold")
        if isinstance(threshold, (int, float)):
            regressed = perf["median_ratio"] < threshold
            if regressed != (perf["status"] == "fail"):
                raise SystemExit(
                    "perf stage status contradicts its own numbers: "
                    f"median_ratio {perf['median_ratio']} vs threshold {threshold} "
                    f"but status is {perf['status']}"
                )
    if perf["status"] == "fail" and not any(
        item.get("stage") == "perf" and item.get("severity") == "should-fix"
        for item in findings
    ):
        raise SystemExit("perf stage failed but no should-fix finding was recorded")
    if perf["status"] == "skip" and "median_ratio" in perf:
        raise SystemExit("perf stage was skipped but still reports a median_ratio")
out_path.write_text(json.dumps(report, indent=2) + "\n")
print(
    f"validation report accepted for head {expected_head}; "
    f"target={selection['target']}; "
    f"grid={selection.get('grid') or 'not configured'}"
)
# Printed separately and unconditionally, because Step 8 must state a perf line either way:
# a silent absence here is what produced a card with no numbers on aiter#4538.
if perf is None:
    print("perf stage: absent — the card's perf line is advisory (see P6)")
elif perf["status"] in {"pass", "fail"}:
    print(
        f"perf stage: {perf['status']} — median_ratio {perf['median_ratio']} on "
        f"{perf.get('worst_column')} over {perf['matched_rows']} row(s), "
        f"threshold {perf.get('threshold')}"
    )
else:
    print(f"perf stage: skip — {perf.get('note', 'no reason recorded')}")
PY
else
  # Distinguish the reasons there is no report. "Not applicable" and "required but missing"
  # are different facts about the PR, and collapsing them into one sentence is what made the
  # old verdict line uninformative. Triage's blocker explains the runs never attempted; the
  # outcome file explains the ones that were, and one of the two is always present.
  python3 - "$WORK/validation_requirement.json" "$WORK/auto_validation_outcome.txt" <<'PY'
import json
import pathlib
import sys

req = json.loads(pathlib.Path(sys.argv[1]).read_text())
outcome = pathlib.Path(sys.argv[2])
if not req["required"]:
    print(
        "validation not applicable: no runtime surface changed "
        f"({', '.join(req['families'])}); a static-only review is complete here, "
        "not deficient"
    )
else:
    reason = req["blocking_reason"]
    if not reason and outcome.is_file():
        reason = outcome.read_text().strip()
    print(
        "validation REQUIRED but not run: "
        f"{reason or 'reason unrecorded, which is itself a defect in this step'}. "
        "Report it as a gap in the evidence, and if the reason is a missing test target, "
        "as a finding about the PR."
    )
PY
fi

# Inline review comments (line-level code comments — often more specific than top-level)
gh api "repos/$REPO/pulls/$PR/comments" | python3 -c "
import json,sys
comments = json.load(sys.stdin)
for c in comments:
    author = c.get('user',{}).get('login','')
    body = (c.get('body','') or '').strip()
    path = c.get('path','')
    line = c.get('line') or c.get('original_line','')
    if body and 'copilot' not in author.lower() and 'bot' not in author.lower():
        print(f'[INLINE {author}] {path}:{line}')
        print(f'  {body[:250]}')
" 2>/dev/null
```

Read the diff and PR body before proceeding.

**Cross-file verification — do this before reporting any kernel/dispatch finding.** The diff shows changed lines, not the whole story. Grep the *entire* symbol family, not just files in the diff:
- Sync/fence/atomics, or the "other half" of a scatter, may live in a `.cuh`/`.h` rather than the `.cu` in the diff. Grep `.cu` + `.cuh` + `.h` together before claiming "no synchronization" or "no bounds check." (aiter#3802: a "kernel has no sync" finding was a false positive — the signal barrier was in the `.cuh`.)
- Read the actual function in the head file, not just the diff hunk, before claiming a branch/else is missing. (aiter#4098: a "compares raw uint8 vs float" finding was false — the reader had a conditional `maybe_view_fp8()` the diff never showed.)

**Classify every CI failure before blaming the PR.** A red check is not automatically the PR's fault:
- Read the failed *step*. `check-signal` / "Wait for Checks" timeouts, "Expected exactly one wheel artifact", and dep-resolver noise are **infra flakes**, not code failures (aiter#3593, #4171).
- Compare against main: if main fails the same shard in the same window, it's baseline/flaky, not a regression introduced here.
- Expired logs (`HTTP 410 Gone`) on old runs mean the failure is months-stale and meaningless against today's main — ask for a rebase + fresh run instead of quoting it (aiter#2565).

---

## Step 2 — Semantic Understanding (answer all 5 before rules)

Work through these by reading the diff, not the description alone.

**Q1 — What specifically changed computationally?**
Not "improves perf" — what algorithm/formula/data flow changed?
_Answer:_

**Q2 — Hardware scope: which arch(es), precision(s), execution phase(s)?**
gfx942 / gfx950 / gfx1250? fp16/bf16/fp8? decode / prefill / both?
_Answer:_

**Q3 — Does this change any public aiter API?**
New symbol in `aiter/ops/*.py`, new kwarg on existing op, change to `aiter/__init__.py`?
_Answer:_

**Q4 — Performance claim: what is the mechanism?**
Not "faster" — WHY is it faster? (fewer memory round-trips, fewer kernel launches, better tiling?)
_Answer:_

**Q5 — Does the description explain WHY or only WHAT?**
"Fuses kernels for speedup" = surface. "Eliminates intermediate HBM write between rmsnorm and quant" = understanding.
If surface-level only → treat as elevated AI-code risk.
_Answer:_

---

## Step 3 — PR Type Classification

Check which type(s) apply; these determine which Step 5 categories are mandatory.

- [ ] **New kernel / new Triton op** → B1 (dispatch gate), B2 (tl.load mask), B4 (new routing value unhandled?), A1 (sibling variants), D1 (atomic zero-init), D8 (contiguous check), HK6 (UT), P6 (measure it base-vs-head)
- [ ] **New constexpr / routing flag / new dtype or arch value added** → B4 (do ALL dispatch branches handle the new value, or assert on it?), C4 (new arch string literal?)
- [ ] **Tuning config (CSV / YAML)** → D3 (hipblaslt), HK4 (kpack:1)
- [ ] **Dispatch logic change** → B1 (silent bypass), B3 (string normalization), B4 (new value unhandled?), A3 (scope too broad)
- [ ] **Replaces existing kernel as default** → D2 (rollback env-var), P6 (measure it base-vs-head — a default swap is the case where an unmeasured regression reaches everyone)
- [ ] **Core file change** (see Tier table below) → full Step 4 risk assessment
- [ ] **API signature change** (param added / removed / renamed, default changed, return dtype or arity changed) → B6 (propagation to all receivers), E1 (linked consumer PR?), E5 (owner sign-off if stable core API)
- [ ] **Refactor / rename** → HK2 (unrelated files), variable name mismatch check, B6 (rename breaks all importers)
- [ ] **FP8 / quantization path** → C1 (fnuz by dtype), C2 (fp8_max hardcoded), D1 (atomic zero-init)
- [ ] **Perf / benchmark PR** → P1 (numbers with units), P5 (setup cost excluded?), P2 (production shapes), P3 (reproducible), P6 (re-measure here — P1–P5 only grade the PR's own table)
- [ ] **Test / benchmark only** → P2 (production shapes), HK6 (aiter-op-test format)
- [ ] **Async / multi-stream** → G1 (stream sync missing), G1b (blocking queue.get without timeout in serving code)
- [ ] **FlyDSL kernel** → D10 (compile result called?), D10b (arith.unwrap() before arith.bitcast?). A FlyDSL kernel change is runtime surface, so Step 1's triage marks it `REQUIRED` and, when the PR ships exactly one test target, has already run the validator against it. Use whatever report Step 1 accepted as the evidence; where it reached no report, mark the result `[static-only advisory review]` (see Step 8) and make no runtime clearance claim. Absence of a report is not itself a blocker. Two target classes cannot reach `PASS` by construction, so their `INCONCLUSIVE` is the expected output and not a deficiency: a CPU-only target claims no GPU and therefore no architecture, and a bugfix with no shape dimension has no grid for `correctness_s1_grid` to consume. Never ask such an author for a passing report.
- [ ] **New if/elif dispatch with variable assignment** → D1b (UnboundLocalError on uninitialized path)
- [ ] **Change to behavior/dispatch of a downstream-consumed op** (mla / fused_moe / attention / mha / quant / gemm_op_a8w8 / moe_op / jit-core) → E4 (is downstream CI triggered or skipped?), E5 (stable-API owner sign-off)
- [ ] **New `@compile_ops` / `torch.library.custom_op`, or change to an op's return dtype/arity** → D7 (fake/abstract impl exists?), D6 (fake dtype/shape matches real op?)
- [ ] **Kernel launcher / buffer-offset or index arithmetic (long-context or large-batch path)** → D9 (int32 overflow at production scale)
- [ ] **Removes or reverses a zero-init / assert / `.contiguous()` / documented invariant** → D4 (invariant reversal cited?), B7 (assert may block valid shapes)
- [ ] **New weight attribute / weight transform** → F1 (double HBM pin)

---

## Step 4 — Core File Risk Assessment

**What makes a file "backbone"?** Apply these three questions to any file in the diff — including new files not in the table below.

```
Q1 — Tier 1 test: If this file has a Python syntax error or fails to import,
     does `import aiter` still succeed?
     → NO  → Tier 1 (system-critical: aiter itself breaks)

Q2 — Tier 2 test: Does this file contain the Python dispatch logic that
     selects which kernel to run for an op class,
     AND is that op used by >1 production model family (DSv3, Kimi, MiniMax…)?
     → YES → Tier 2 (op-class critical: wrong result for ALL users of that op)

Q3 — Tier 2 alt: Is this file the public aiter API for an op
     (`from aiter import X` imports from here)?
     → YES → Tier 2 (signature change silently breaks all consumers)

Otherwise → Tier 3 (individual kernel or model-specific code).
```

The table below is the current snapshot — use it to confirm, but Q1/Q2/Q3 to classify new files.

Backbone files ranked by git commit frequency (2025–2026) and blast radius:

| Tier | File | Git commits | Blast radius | Failure mode |
|------|------|-------------|-------------|--------------|
| **1** | `aiter/jit/core.py` | 182 | **ALL ops** — JIT compilation engine | Any import of aiter fails; zero ops load |
| **1** | `aiter/__init__.py` | 52 | **ALL** vLLM/SGLang/ATOM users | `ImportError` or silent namespace truncation below broken import |
| **1** | `aiter/ops/*.py` (any) | varies | All consumers of that op | `AttributeError` at call time in downstream |
| **2** | `aiter/fused_moe.py` | 119 | All MoE models (DeepSeek, Kimi, MiniMax) | Wrong expert routing, silent accuracy drop |
| **2** | `aiter/ops/mha.py` | 89 | All MHA attention paths | Wrong attention output, crash |
| **2** | `aiter/ops/attention.py` | 66 | MLA/paged attention dispatch | Wrong KV, accuracy drop |
| **2** | `aiter/ops/gemm_op_a8w8.py` | 59 | All FP8 quantized GEMM | Wrong matmul result, silent accuracy drop |
| **2** | `aiter/mla.py` | 57 | All MLA decode/prefill (DSv3/Kimi) | Wrong KV, accuracy drop, crash |
| **2** | `aiter/tuned_gemm.py` | 52 | All GEMM-backed ops | `assert False` crash or silent fallback to slow path |
| **2** | `aiter/ops/moe_op.py` | 51 | MoE op dispatch table | Wrong dispatch, wrong expert weights |
| **2** | `aiter/ops/quant.py` | 49 | All quantization paths | Wrong scale, silent accuracy drop |
| **3** | Individual kernel `.py`/`.cu` | — | Ops using that kernel | Depends on kernel type |

**`aiter/__init__.py` special rule**: The import block must NOT be wrapped in try/except.
Any new import added here → check the imported module for bare `ImportError` paths that
could silently truncate the namespace.

**`aiter/jit/core.py` special rule**: This file bootstraps the entire JIT compilation pipeline.
A syntax error, wrong default, or broken env-var handling here means zero aiter ops load.
Changes here require e2e smoke test across all GPU arch targets.

**Mandatory backbone checks — must be answered before writing the verdict:**

For **Tier 1** files (jit/core.py, __init__.py, ops/*.py):
- [ ] List every public symbol changed. Grep for all callers across aiter itself: `grep -rn '<symbol>' aiter/`. If a caller is not covered by the PR's test, flag it.
- [ ] For `__init__.py`: does the new import have a bare `ImportError` path that could silently truncate the namespace?
- [ ] For `jit/core.py`: is there an e2e smoke test that loads all kernels on gfx942 AND gfx950 after this change?
- [ ] State explicitly: if this change is wrong, what breaks and how would it be detected? (all ops fail / one op family fails / silent wrong value)

For **Tier 2** files (fused_moe, mha, attention, gemm, mla, tuned_gemm, quant):
- [ ] Which model families (DSv3, Kimi, MiniMax, GLM…) use this op? Is at least one from each family in the test?
- [ ] Are production shapes tested? At minimum: decode (M=1, TP=4/TP=8) AND prefill (ISL=4096, TP=4/TP=8).
- [ ] Does the change affect gfx942 only, gfx950 only, or both? If both, are both arch paths tested?

**AI code red flag — verbatim duplication across backbone files:** Same algorithm copy-pasted into 2+ backbone files with only variable names changed. See D5.

---

## Step 5 — Rule Checklist

Six failure categories — work all six in order. Advisory severity per finding:
🔴 high risk / ⚠️ should fix / 📝 note. These labels prioritize human attention; they do not
themselves gate a merge.

**🔴 evidence threshold — before firing any 🔴, write down the concrete input that triggers it.** Name the specific shape / scale / dtype / arch / value that makes the finding fire (e.g. "at `token_id` > 16M with H=32, D=128 the int32 product exceeds 2^31", or "when `arch=='gfx1250'` with fp4 input the branch assumes fp8"). If you cannot state a concrete triggering case, the 🔴 is unproven — **downgrade to ⚠️ ("worth checking") or drop it.** A 🔴 that reads as a definite defect but names no demonstrable triggering input is exactly how a false positive lands on a maintainer's PR. This threshold applies to every rule below — including those whose own text omits an explicit FP self-check (e.g. D9): the same index expression is safe in a capped/small-batch kernel and unsafe only at a scale you must actually exhibit.

| Category | Core question | Key triggers |
|---|---|---|
| **A. Coverage gaps** | Same bug elsewhere? Same code other configs? | `_opt`, `_prefill_opt`, `_v2`; shared path; broad `if` condition |
| **B. Silent bypass** | Does every input reach the right branch? | gated-off param; string alias; non-aligned dim; proxy metric |
| **C. Hardcoded arch/dtype** | Does the constant break on another GPU or fp8 flavor? | `240.0`, `448.0`; arch name for fnuz; `bf16` fixed |
| **D. Uninitialized state** | Is the buffer clean before atomic/kernel launch? | `::empty()`+`atomic_fmax`; `fill_(0)` missing |
| **E. Cross-repo sync** | Does the consumer know about this change? | new aiter symbol; default-preserving new param; plugin bridge |
| **F. Resource duplication** | Does the change double GPU memory silently? | new `_preshuffled`/`_quantized` weight alongside original |

---

### A — Coverage Gaps
_"Fixed one path; the same bug lives in a sibling."_

**A1 — Sibling kernel not fixed** ⚠️ (🔴 if in Tier-1/2 backbone)
Fix changes address calc, bounds check, type widening, or data layout in a CUDA/HIP kernel:
scan the same file for variants named `_opt`, `_prefill`, `_decode`, `_prefill_opt`, `_v2`, `_fast`.
Real example (PR#3841): strided q_nope OOB fix applied to decode kernel; `_prefill_opt` in the same file had the same bug unfixed.
→ `⚠️ A1: same bug may exist in [variant] — check kernel family in this file`

**A2 — Shared path, no cross-model validation** ⚠️
Changed code shared across model families (not model-specific): validated on all?
Real example (PR#3891): valarLip: "please make sure e2e CI passes before changes to common part."
→ `⚠️ A2: change touches shared path — e2e or cross-model validation needed`

**A3 — Activation condition broader than validated scope** ⚠️
New dispatch condition (e.g., `if is_deepseek():`) enables a kernel for more archs/models than tested.
Real example (vLLM#16435): FusedMoE activated for wrong model families → follow-up restrict PR needed.
→ `⚠️ A3: activation condition [X] enables more than validated scope [Y]`

---

### B — Silent Bypass
_"The code looks complete but certain inputs silently take the wrong path."_

**B1 — Dispatch gate with unchecked parameter** 🔴
New `if/elif/else` branch: for each parameter gated off — is it **asserted** (None/zero) or **forwarded**?
If neither: wrong results, no crash, no error.
Trigger: `dropout_p`, `window_size`, `block_table`, `logits_soft_cap`, `alibi_slopes`, `is_causal`.
Real example (PR#3576): `block_table is not None` False-branch computed dense attention silently.
Real example (PR#3390): `is_causal=True` not forwarded → "fake causal" fmha passed all CI.
→ `🔴 B1: [param] silently ignored in [branch] — assert or forward`

**B2 — Triton tl.load / tl.store without mask** 🔴
Unmasked load when dim is not a multiple of BLOCK_SIZE → silent garbage read, no segfault.
Common non-aligned dims: `seqlen`, `vocab_size`, `hidden_dim`, `num_heads`, `head_dim`, `kv_lora_rank`.
FP self-check (do this before firing): confirm the loaded dim is NOT guaranteed a multiple of BLOCK_SIZE — i.e. it is not padded/rounded up at allocation, not `tl.cdiv`-tiled with a masked tail elsewhere, and not already guarded by an enclosing mask or a caller-side pad. An unmasked load on a provably-aligned dim is safe; do not fire. Name the concrete non-aligned dim value that triggers the OOB.
→ `🔴 B2: tl.load at [line] missing mask= — silent OOB on non-aligned inputs`

**B3 — String dispatch without normalization** ⚠️
`quant_type == "per_token"` before normalizing: aliases `"fp8_per_token"`, `"per-token"`, `QuantType.per_Token` silently miss the branch.
Real example (PR#3981): raw string compare in `parallel_state.py` — alias callers missed torch-compile fast path.
→ `⚠️ B3: string dispatch [cond] without normalization — aliases fall through to slow path`

**B4 — New dispatch value not handled by all paths, no warning** ⚠️/🔴
When a PR introduces a new routing value to a multi-way dispatch — a new dtype string (`'fp4'`), a new arch string (`'gfx1201'`), a new layout flag (`SWIGLU_INTERLEAVED`), a new constexpr enum value — every reachable dispatch branch must either (a) handle it explicitly, (b) fall through to a documented safe default, or (c) assert/warn before the wrong branch is reached. If any reachable branch silently falls through to behavior that is wrong for the new value, flag it.
Severity: 🔴 if the wrong path produces incorrect output silently (wrong layout, wrong kernel, wrong scale). ⚠️ if the wrong path is a safe-but-suboptimal default (e.g., generic tile depths instead of tuned fp4 depths).
Exception: an upstream assert/raise/isinstance check that prevents the bad value from entering the branch → not B4. A runtime assert that fires for the dangerous combo → not B4.
FP self-check: Is the uncovered branch actually reachable with the new value? Is there a caller contract (documented or asserted) guaranteeing the bad combo never occurs?
Real examples: GGUU flag not wired into gfx950 Triton path — runtime assert guards the explicitly dangerous combo but the remaining gap is silent (aiter#4169); fp4 silently falls through to `in_dtype in ('fp8','int8')` tile table, uses generic preload depths (aiter#3941); cross-attention + mt=1 on gfx1250 falls through to get_heuristic_kernel with no gfx1250 kernel compiled for that combo (aiter#3939).
→ `🔴/⚠️ B4: [new value] reaches [branch] which assumes [old value] — [what wrong thing happens] — add assert or explicit handling`

**B5 — Triton `tl.constexpr` safety check disabled without invariant proof** ⚠️
A `tl.constexpr` bool that gates a validity check (e.g., `CHECK_NEG_ONE_SENTINEL`, `CHECK_BOUNDS`) can be set `False` by a caller to skip the check. If the invariant the check enforces is not independently guaranteed on that path, illegal memory access or silent wrong values result.
Trigger: new `tl.constexpr` bool in a Triton kernel that disables a bounds/sentinel/validity check; caller comment says "X path can disable this" without documenting what guarantees the invariant holds on that path.
Real example (ATOM#1498): `CHECK_NEG_ONE_SENTINEL=False` disables the -1 slot filter in the paged prefill kernel; illegal access if any -1 slot appears without the check.
→ `⚠️ B5: [constexpr] disables [check] — document which caller invariant guarantees no [invalid value] on that path`

**B6 — API propagation incompleteness** 🔴/⚠️
When an API surface changes in dimension X, all downstream receivers (Y) must be updated. Unhandled propagation silently falls through to wrong behavior (Z).

| Sub-type | X (what changed) | Y (downstream not updated) | Z (failure) | Sev |
|----------|-----------------|---------------------------|-------------|-----|
| param-discard | new param in signature | function body | value accepted but never used | ⚠️/🔴 |
| param-removed | param removed from signature | all call sites (cross-repo if public) | TypeError at call time | 🔴 |
| repr-key | new Gluon constexpr | kernel repr key list | stale JIT binary served | 🔴 |
| arch-discard | arch-specific kwarg | non-target-arch path | kwarg silently discarded | ⚠️ |
| dispatch-silent | multi-backend fallback | caller logging | backend switch with no diagnostic | ⚠️ |
| rename | public symbol renamed | all importers (cross-repo if public) | AttributeError at import/call time | 🔴 |

Severity (param-discard): 🔴 if param controls output correctness (`expert_mask`, `q_scale`, `kv_scale`); ⚠️ for performance knobs or optional features with working defaults.
**Public-API scope:** if the changed symbol is a public op (`from aiter import X`, or lives in `aiter/ops/*.py` / `aiter/__init__.py`), param-removed and rename break cross-repo consumers (ATOM / SGLang / vLLM), not just same-file call sites — the downstream to check is every repo that imports it. Also apply E1 (is a linked consumer PR mentioned?) and E5 (owner sign-off for a stable core-API contract).
Exception: method override where base class forces the signature but subclass legitimately ignores the param — flag as 📝 (structural discard, not a bug).
FP self-check (rename / param-removed): before firing, confirm the old symbol is NOT preserved by a compatibility shim added in the same PR — a new same-named `def` wrapper (keeps `from aiter import old_name` resolving), an alias, or a binding pin (`@compile_ops(..., fc_name='old_name')` keeps the C++ symbol even when the Python fn is renamed). A rename/removal behind such a shim is backward-compatible — do not fire. Real non-example (aiter#4227): `get_mla_metadata_v1` renamed to `_impl`, but a same-named wrapper + `fc_name='get_mla_metadata_v1'` preserved both the Python and C++ symbols → not B6.
Real examples (param-discard): `expert_mask` accepted but `# return None` commented out → TP expert-parallel callers silently routed wrong; `v_scale` strides never computed — `sc_off` indexes v_scale_ptr using k_scale strides, wrong scale on non-contiguous tensors (aiter#3959); `gate_up` discarded when `is_guinterleave=False` (aiter#4167).
→ `🔴/⚠️ B6-[sub-type]: [what changed] — [downstream not updated] — [failure]`

**B7 — Over-conservative assert blocks valid shapes** ⚠️
`assert M % tileM == 0` when the kernel pads internally and handles non-aligned M.
Real example (PR#3998): wrapper asserted alignment; asm kernel padded — valid small-M shapes rejected at the Python layer.
FP self-check: Does the kernel actually handle non-aligned inputs, or does the assert reflect a real hardware requirement?
→ `⚠️ B7: assert [constraint] may be unnecessary — verify kernel handles non-aligned inputs before removing`

---

### C — Hardcoded Arch / Dtype Assumptions
_"The constant is correct for gfx942/fnuz; it silently breaks on gfx950 or OCP e4m3."_

**C1 — FP8 fnuz check uses arch name** ⚠️
`if "gfx942" in arch: treat_as_fnuz()` — wrong. Same arch can have both fn and fnuz in flight.
Check IS fnuz: `tensor.dtype == fp8_fnuz`. Gate CONVERSION by arch is OK; inspection must use dtype.
Real example (PR#4073): valarLip: "check _is_fnuz by tensor's DType instead of arch."
→ `⚠️ C1: fnuz check uses arch name — use tensor.dtype comparison`

**C2 — FP8 scale bound hardcoded** ⚠️
`fp8_max = 240.0` → correct for fnuz (e4m3fnuz max=240), wrong for OCP e4m3 (max=448).
Use `get_dtype_max(dtype)` to derive; add a runtime guard if gfx942-only.
FP self-check: if the constant sits on a path already runtime-guarded to a single dtype/arch (e.g. inside an `if arch == 'gfx942':` block), the hardcode is safe there — do not fire; fire only when the path handles multiple fp8 flavors.
Real example (PR#4015): yzhou103: "would break for OCP e4m3 (max=448)."
→ `⚠️ C2: fp8_max hardcoded to [value] — use get_dtype_max(dtype)`

**C3 — Dtype hardcoded without checking actual tensor** ⚠️
Fixed `bf16`, `fp8_e8m0`, or similar in a forward path that handles multiple configs.
FP self-check first: search the unchanged lines of this file for the same hardcoded dtype — if it already appears pre-existing on the same path, this is not a new violation (do not fire as new). Fire only when the hardcode is newly introduced, or the path newly handles more than one dtype/config.
Real examples: ATOM#1423 "not always bf16"; ATOM#1458 "hard code to fp8_e8m0?"
→ `⚠️ C3: dtype hardcoded to [type] — should derive from actual tensor/config`

**C4 — New GPU arch string literal in dispatch condition** ⚠️
**FP self-check first (do this before deciding to fire):** Search the unchanged lines of this file for the same arch string (e.g., `'gfx1250'`). If that string already appears on an unchanged line → **do not fire** (pre-existing style, not a new violation). Only proceed if the arch string is genuinely new to this file.
Trigger (only after self-check passes): a new `+` line introduces an arch string literal in a dispatch condition (`if arch == 'gfx1250':`, `if 'gfx950' in arch_name:`), rather than routing through the central kernel registry or a named constant.
Also exempt: arch strings used only in comments, docstrings, or directory path strings; arch strings imported from a central registry module; arch strings used as **capability guards inside a kernel-specific wrapper function** (not in the centralized dispatch layer) — e.g., `get_gfx() == 'gfx1250'` inside `flydsl_flash_attn_batch_func` determines whether the FlyDSL variant is available; that check belongs in the wrapper, not in the central registry, and does not trigger C4 (aiter#3870).
Real examples: `'gfx1250'` new to `fused_mxfp4_quant.py` dispatch logic where no prior arch literals existed (aiter#3937 → fire C4); `'gfx1201'` added to `unified_attention.py` where `'gfx1250'` was already on line 79 (aiter#3956 → skip, pre-existing style); `get_gfx() == "gfx1250"` inside FlyDSL wrapper `flydsl_flash_attn_batch_func` (aiter#3870 → skip, capability guard not centralized dispatch).
→ `⚠️ C4: new arch string '[gfxNNNN]' hardcoded in dispatch — route through arch registry or named constant`

---

### D — Uninitialized / Boundary State
_"The code writes or reads memory that was never properly initialized."_

**D1 — Atomic reduction on uninitialized buffer** 🔴
`atomic_fmax(*ptr, val)` = `*ptr = max(*ptr, val)`. If `*ptr` is uninitialized (from `::empty()`),
garbage dominates the max → corrupted amax → corrupted FP8 descale → silent wrong quantization.
Trigger: `atomic_fmax` / `atomic_max` + `::empty()` or non-zeroed allocation near it.
Severity: 🔴 for atomic accumulation (atomic_fmax, atomicAdd) — garbage propagates into every output element. ⚠️ for partial-sum buffers where a zero-weight coefficient mathematically cancels the contribution (e.g., online softmax with empty batch: `exp(-inf) × garbage = 0`); still flag because `0.0 × NaN = NaN` on IEEE hardware if the allocator returns dirty pages.
Real example (PR#4015): yzhou103: "AiterTensor::empty does not zero-initialize... garbage in v_amax silently corrupts descale."
→ `🔴 D1: [buffer] passed to atomic_fmax not zero-initialized — use ::zero() not ::empty()`

**D1b — Python-side UnboundLocalError from conditional assignment** 🔴
A variable is assigned inside an `if/elif` branch but referenced unconditionally after the block. Python does not detect this statically — `UnboundLocalError` or `NameError` fires only at runtime when the skipped branch is exercised. Silent in test environments that never hit the uninitialized path.
Trigger: new `if/elif` gate assigns a variable (`result = ...`) on some branches; a later line references it without a pre-block default. Check: is there a `var = None` or `var = default_val` before the if-block?
Exception: if there is a definitive `else` branch that also assigns the variable, or if the variable is only ever used inside the branch that assigns it.
Real example (ATOM#860): `needs_independent_noise` returned from `prepare_model()` tuple but assigned only in one branch of `prefill_forward` — other branch paths raised `NameError` when the sampler tried to use it.
→ `🔴 D1b: [var] assigned only inside [branch] but referenced unconditionally — add [var = default] before the if-block`

**D2 — New default path without rollback env-var** ⚠️
New implementation replaces existing default before wide validation: is there an env var to revert?
Scope: D2 is about a **temporary rollback kill-switch** for a risky default swap (meant to be removed once validated) — NOT a permanent feature-flag knob. aiter maintainers generally reject new *permanent* env vars (see HK9): a MoE activation knob added in #3593 was reverted in #4225. If the safe path can be auto-derived from dtype/arch/shape instead of an env var, prefer that; reserve the env var for a genuine short-lived rollback.
Real example (PR#3266): flydsl sort replaced opus sort; reviewer: "gate flydsl behind env var until validated on broader workloads."
→ `⚠️ D2: new default path needs rollback env-var for safe rollout`

**D3 — hipblaslt in CSV/YAML tuning config** 🔴
Any `+` line with `hipblaslt` in a tuning file. Not persistent across Docker; causes hangs.
→ `🔴 D3: hipblaslt config must not be committed`

**D4 — Invariant reversal without citation** 🔴
A documented safety invariant is reversed: old comment says "must X because Y" → new code removes X claiming "X not needed" but no spec/asm/test is cited to prove Y no longer holds.
Trigger: `::zeros() → ::empty()` / `torch.zeros → torch.empty` where old comment mentions "must" / "required" / "read back as zero"; assert deletion without explanation; `.contiguous()` removal; zero-init removal with contradicting justification.
Real example (aiter#4043): old: "trailing pad must read back as zero for the asm reader, so zero-initialise it here" → new: "trailing pad is never read by the asm reader, so no zero-init is needed" — two comments directly contradict; PR cites no spec. Human reviewers missed this, only saw the profiling screenshot.
→ `🔴 D4: [operation] reverses a documented safety invariant — cite the spec/asm/test proving new assumption is safe`

**D5 — Verbatim duplication across backbone files** ⚠️
The same fix is copy-pasted into 2+ Tier 1/2 backbone files with trivial name substitution (different variable names, identical algorithm and comments). AI code signature: changes look symmetric but each file's invariants may differ and were not independently verified.
Trigger: nearly identical `+` blocks appearing in two backbone files in the same PR diff; same formula / same comment structure / same magic constants, only variable names differ.
Real example (ATOM#1493): chunked indexer loop copy-pasted verbatim between `deepseek_v2.py` and `deepseek_v4.py` — same `(budget_rows // 128) * 128` formula, same `bit_length() - 1` fallback, same comment block, only variable names changed.
→ `⚠️ D5: identical algorithm in [file_a] and [file_b] — was correctness verified independently in each context, or copy-pasted?`

**D6 — Fake / meta function dtype or shape mismatch** 🔴
When a `gen_fake` / `_fake` / `abstract_impl` function is added or modified, its return tensor dtypes and shapes must match the real op exactly. torch.compile uses the fake to infer output types; a wrong dtype compiles cleanly but causes a dtype assertion or silent wrong values at runtime.
Trigger (1): diff contains a `_fake` / `gen_fake` function alongside the real op; compare each return tensor's dtype and shape against the real op's actual output.
Trigger (2): real op's return dtype or arity changes in the diff but no corresponding `_fake` / `gen_fake` change appears — the existing fake is now stale and will produce wrong types.
Real example (aiter#4110): `fused_allreduce_rmsnorm_quant_fake` returned `torch.empty_like(res_inp)` (bf16) as first element, but real op returns fp8 — wrong dtype for torch.compile's dtype checks. Human reviewers missed this entirely.
→ `🔴 [fake_fn] return [N] dtype is [X] but real op returns [Y] — torch.compile will assert or silently miscompute`

**D7 — New compile_op without fake function** 🔴
A new `@compile_ops` / `torch.library.custom_op` is added but has no corresponding `_fake` / `gen_fake` / `abstract_impl`. torch.compile traces the graph using fake tensors; without a fake, the op is a black box → runtime crash or silent fallback to eager inside a compiled region.
Trigger: diff adds a new function decorated with `@compile_ops` or `torch.library.custom_op`; grep for a `_fake` or `gen_fake` function with the same op name — if absent, flag.
→ `🔴 D7: [op_name] has no fake/abstract implementation — torch.compile will crash or silently fall back to eager`

**D8 — Kernel wrapper missing contiguous check** ⚠️
Python wrapper passes tensor to C++ / HIP kernel but doesn't assert `.is_contiguous()` or call `.contiguous()`. If the caller passes a strided tensor (slice, `.T`, output of non-contiguous `view()`), the kernel reads from wrong addresses — completely silent wrong result.
Trigger: new Python wrapper that calls a `@compile_ops` or C-extension kernel; check that non-trivially-shaped inputs (anything other than a freshly allocated `torch.empty`) are either asserted contiguous or explicitly made contiguous before the call.
→ `⚠️ D8: [tensor] passed to [kernel] without contiguous check — add .contiguous() or assert .is_contiguous()`

**D9 — INT32 overflow in GPU pointer arithmetic** 🔴
C++ kernel launcher or Python wrapper computes a buffer offset, record count, or index in `int32` (or Python `torch.int32`) when the product of dimensions can exceed 2^31 (~2 billion) at production scale.
Common patterns: `token_id * (num_heads * head_dim)` overflows at token_id > 16M with H=32, D=128; `seq_start * K` overflows for long-context at seq_start > 256K with K=8192; gfx1250 TDM block descriptor count fields computed as Python int default to int64 — a missing `.to(torch.int32)` cast silently produces wrong offsets.
Trigger (structural, NOT a name list): a multiplication that feeds pointer or index arithmetic, where at least one operand derives from a **non-`constexpr` parameter of the enclosing kernel** — a value supplied at runtime, which is the only kind that can grow past 2^31 — and no operand is widened to 64 bits, counting a widening applied on an earlier line and carried in through a local name. `constexpr` tile constants bound the product at compile time and are excluded. Also fires on a TDM descriptor field feeding block offset computation without an explicit int32 cast.
**Why the trigger is structural, and why saying so was not enough:** an earlier version of this rule listed the names `token_id`, `seq_start`, `batch_offset`, `total_tokens`. Three real defects used none of them — `stride_out_batch`, `block_id`, `physical_block`, `context_kv_idx` — and the rule stayed silent on all three (aiter#1674 ×2, aiter#3541). The rule text was then rewritten to say "structural" while still defining index-shaped and stride-shaped by name, and the scanner behind it matched two name lists against operand text. Measured on aiter#4978, the PR that introduced the `moe_wgrad` overflow later fixed by #5132: **0 of the real defect lines were reported**, the one `moe_wgrad` candidate emitted was an already-`int64` line, and 390 candidates were produced overall. The scanner is now an AST pass with no name lists at all; on the same diff it reports 4 of 4, and on #5132 it reports none. Do not narrow it back to a name list.
**Production scale.** Step 1 printed `validate-kernel-pr/production_scale.md` directly beneath the candidate list: pool sizes, batch limits and stride semantics that the diff does not contain. Use those numbers to name the triggering case the 🔴 gate requires; if none of them puts the product past 2^31, clear the candidate and say so.

**The candidate list is already in context.** Step 1 ran `scan_index_width.py` over the diff and printed, per file, every distinct index×stride expression reaching pointer arithmetic with no 64-bit widening. Work that list: clear each candidate, and fire D9 only where you can name the production scale at which the product exceeds 2^31. If the list is empty, say so rather than skipping the category silently. **If the scan printed a `NOT SCANNED` section, D9 cannot be cleared** — those files were never examined, and an empty candidate list that excluded them is not evidence of absence. Report the unscanned files instead of reporting no candidates.
Real examples: `out_base = token_id * num_heads * head_dim` in int32 overflows at scale (PR#3844); forward kernel uses `Int32(seq_start) * Int32(K)` while the backward kernel correctly uses int64 (PR#4113).
→ `🔴 D9: [index expr] in int32 — widen [index operand] to int64 before multiplying by [stride], overflows at [concrete production scale]`

**D10 — FlyDSL compile result stored but never called** 🔴
`flyc.compile(exe, *args)` on a cache-miss path compiles and stores the `CompiledFunction` object (`exe._cf = cf`) but does NOT call it — `cf(*args)` is absent. Every first-invocation of a new (shape, arch, dtype) combination silently no-ops the entire kernel launch and returns the uninitialized `torch.empty` output to the caller with no error.
Trigger: a cache-miss branch in a `_run_compiled`-style function that calls `flyc.compile(...)` and then returns without executing the compiled result.
Note: `flyc.compile()` ONLY compiles; it does NOT execute. The compiled result must be explicitly called with `cf(*args)` on the same branch. Do not confuse this with Triton's `@triton.jit` which auto-executes on first call.
Real example (aiter#3987): `tensor_shim.py` — cold-start on any new shape returns garbage output; all `_launch()` call sites through `fused_moe_gfx942.py` inherit this behavior.
→ `🔴 D10: [fn] compiles on cache-miss but does not call the result — add cf(*args) on the same branch`

**D10b — FlyDSL arith.bitcast requires arith.unwrap() on operand** 🔴
Inside a FlyDSL kernel, passing a raw DSL value directly to `arith.bitcast(val, target_type)` causes a type error at JIT-compile time — DSL values must be unwrapped with `arith.unwrap(val)` first. This fails silently in Python (no static type error) and only crashes at kernel JIT time when the shape/dtype combo is first encountered.
Trigger: any `arith.bitcast(...)` call in a FlyDSL kernel where the first argument is a DSL expression (result of an arithmetic op, a load, or a `const_expr`) rather than a plain Python literal. Check: is `arith.unwrap(...)` wrapping the value?
Real example (aiter#3944): `arith.bitcast(val, ...)` inside a bf16/f16 output path without `arith.unwrap()` — JIT type error on first invocation of that dtype branch.
→ `🔴 D10b: [expr] passed to arith.bitcast without arith.unwrap() — wrap as arith.unwrap([expr]) first`

---

### E — Cross-Repo Sync
_"The change is incomplete without a matching update in another repo."_

**E1 — New aiter symbol or kwarg without linked aiter PR** ⚠️
New `from aiter import X`, new kwargs on aiter calls, new aiter usage: PR description links an aiter PR?
New kwargs may require an aiter version not yet released.
Real example (ATOM#1494): `emit_bf16=True` kwarg added → needed aiter PR first.
→ `⚠️ E1: new aiter usage — corresponding aiter PR not mentioned`

**E2 — New param with backward-compatible default is dead code** 📝
New param added with default that preserves old behavior: the fix only activates when a consumer passes non-default. Who updates the consumer?
Real example (PR#3773): `max_seqlen=-1` added in aiter; fix never activated until ATOM passed actual value.
→ `📝 E2: new API param needs consumer-side update to activate — follow-up tracked?`

**E3 — Plugin bridge not updated** ⚠️
PR changes KV layout, function signature, or data structure that `deepseek_v4_bridge.py` / `sglang_bridge.py` read directly.
Real example (ATOM#1423): paged-SWA layout changed; bridge still used old layout.
→ `⚠️ E3: [structure] changed — plugin bridge sync needed`

**E4 — Downstream CI skipped on a change downstream consumes** 🔴
aiter's downstream tests (ATOM, SGLang, vLLM) are SKIPPED BY DEFAULT and only run when a label is added — `ci:atom` (DeepSeek-R1-0528, GPT-OSS-120B), `ci:sglang` (DeepSeek-R1-MXFP4, Qwen 3.5), `ci:vllm` (GPT-OSS-120B, DeepSeek-R1-0528, Kimi-K2.5), `ci:all` (all of the above), or `ci:atom_full` (ATOM accuracy suite; only for FlyDSL/Triton upgrades). A PR that changes an op a downstream consumes can pass every *aiter* check with the downstream job skipped, merge green, and break the downstream silently — visible only after merge.
**Staleness guard:** the label→model mapping here is a snapshot. Before quoting a specific model for a label, confirm the current `ci:*` definitions in `.github/workflows/*.yaml` — the model roster rotates, and a stale mapping produces a confidently-wrong label recommendation.
Which label (be precise, do not reflexively pick `ci:all`):
1. **Dispatch reachability** — is the changed/new kernel wired into a default dispatch path? A pure-additive, arch-gated kernel not in any default path is unreachable by downstream → exempt, no label needed.
2. **Map activation → model** — if reachable, read the branch's activation condition (arch × dtype × shape × model gate) and map it to a model (e.g. 128-head fp8 MLA decode on gfx950 → DeepSeek-V4).
3. **Minimal label set** — DeepSeek is exercised by atom+sglang+vllm (≈ `ci:all`); Qwen 3.5 → `ci:sglang` only; Kimi-K2.5 → `ci:vllm` only; GPT-OSS-120B → `ci:atom`+`ci:vllm`.
Fallback: if you cannot trace the activation to a specific model but the diff changes the behavior of an mla/fused_moe/attention/quant/gemm/jit-core path, default to `ci:all` — a wasted CI run is far cheaper than a broken downstream.
Check: in the PR's statusCheckRollup, if `Atom Test` / `Kimi Downstream` / `Sglang Downstream` is `skipped` AND the diff touches a downstream-consumed op, coverage is missing.
Real example (aiter#3459): a DeepSeek-V4 128-head MLA decode kernel passed Aiter Test (success) with Atom Test SKIPPED and no `ci:*` label; after merge the Atom Test went red — the MLA change broke ATOM, invisible pre-merge.
→ `🔴 E4: [op] is consumed by [ATOM/SGLang/vLLM] but its downstream CI is skipped — add ci:all (or the minimal ci:atom/sglang/vllm) and require it green before merge`

**E5 — Stable core-API change needs owner sign-off, not just CI + one approve** 🔴
Modifying the BEHAVIOR / SIGNATURE / DEFAULT DISPATCH / NUMERIC SEMANTICS of a long-lived, widely-consumed API — `fused_moe.py`, `mla.py`, `ops/attention.py`, `ops/mha.py`, `ops/quant.py`, `gemm_op_a8w8.py`, `moe_op.py`, `jit/core.py` — must not be self-merged by a contributor or landed on a single reviewer's approval. These are downstream contracts; green CI (even `ci:all`) only covers the models/shapes it knows, not every downstream version or call path — necessary but not sufficient.
Trigger: diff changes the behavior/signature/default-path of a Step-4 Tier-1/Tier-2 file — NOT a pure-additive, arch-gated, behavior-preserving change (those are exempt; see E4 step 1).
Who signs off: aiter has **no CODEOWNERS file**, so ownership is de-facto — the top committer of the path is the effective gatekeeper:
`git log --format='%an' -- <file> | sort | uniq -c | sort -rn | head`
For `fused_moe.py` / core MoE dispatch this is currently @valarLip (top committer, and the maintainer who reverted #3593 and gates MoE PRs). Re-derive per file — MLA / attention / quant may have a different top committer.
**The reviewer must proactively notify the owner — do not wait for them to notice the PR.** Post a PR comment that @-mentions them (e.g. `@valarLip`) with a one-line summary of the contract change and an explicit request to approve before merge. Passive "someone should sign off" is not enough; the finding is not resolved until the owner has been actively pinged and has responded. Do not settle for a revert after merge.
Real example (aiter#3593): a `fused_moe.py` env knob merged on CI + one approval, then reverted by a maintainer within the hour — it should have had owner sign-off before merge.
→ `🔴 E5: [file] is a stable downstream-facing contract — do NOT self-merge. **Reviewer must @-mention the de-facto owner (git top-committer; @valarLip for fused_moe) in a PR comment requesting explicit sign-off** before merge, on top of ci:all`

---

### F — Resource Duplication
_"The change pins the same data twice on GPU without freeing the original."_

**F1 — New weight variant alongside original** ⚠️
New `w13_weight_preshuffled` / `w_quantized` stored as a new attribute alongside `w13_weight`: both pinned simultaneously → double HBM for that weight.
Real example (ATOM#1469): valarLip: "this will make us pin double weight."
Check: is the original freed after the new variant is created?
→ `⚠️ F1: [new_attr] stored alongside [original] — doubles HBM; is original freed?`

---

### G — Multi-Stream Synchronization
_"Written on stream A, consumed on stream B — no sync between them."_

**G1 — Missing HIP/CUDA stream synchronization** 🔴
HIP/CUDA streams execute concurrently by default. A tensor produced on stream A and consumed by a kernel on stream B without an explicit sync between them causes the consumer to read garbage — no crash, no error, silent wrong output.
Trigger: diff introduces a non-default `torch.cuda.Stream`, passes an explicit `stream=` argument to a kernel, or prepares buffers/weights on a side stream that are later consumed during forward pass on the compute stream. Check: is there `stream.synchronize()`, `stream.wait_stream(other)`, `hipEventRecord` + `hipStreamWaitEvent`, or `torch.cuda.current_stream().wait_stream(other)` between the last write on stream A and the first read on stream B?
→ `🔴 G1: [tensor] written on [stream A] consumed on [stream B] without sync — add stream.wait_stream() or hipStreamWaitEvent`

**G1b — Blocking queue.get() without timeout in production serving code** ⚠️
`queue.get()` without `timeout=` in a worker or service thread that depends on an external producer (decode loop, stream consumer, request handler). If the producer exits abnormally, the worker blocks forever — no crash, no log, hung process.
Trigger: `queue.get()` or `asyncio.Queue.get()` inside a `while True:` worker loop in production serving paths (entrypoints, engine loop, scheduler) without `timeout=` and without a corresponding `except queue.Empty` / `asyncio.TimeoutError` handler or a `done` flag.
Exception: test code, CLI tools, or one-shot scripts where a hang is detectable (CI timeout, interactive TTY).
→ `⚠️ G1b: [worker] blocks on queue.get() without timeout — add timeout= and handle Empty/TimeoutError to survive producer failure`

---

### Performance Evidence (always check)

**P1 — Perf PR without benchmark numbers** ⚠️
Trigger words: perf, optimize, fuse, faster, improve, +X%, replace kernel, OOM fix that changes algo.
Description must have numbers with units (ms, tokens/s, TFLOPS, %, speedup). Screenshots ≠ numbers.
Exception: PRs adding benchmarks/tests for existing ops without claiming improvement.
→ `⚠️ P1: perf claimed — no benchmark numbers with units`

**P2 — Benchmark covers only toy shapes** ⚠️
Numbers exist but only for M≤256, only 1 token, or one model.
Production: DSv4 E=385/topk=7, GPT-OSS 120B, Kimi-K2.5; token range 1→16384.
Staleness guard: the production config list is a snapshot — verify current E/topk/hidden and the model roster from the model registry or a recent benchmark before asserting what counts as "production".
→ `⚠️ P2: benchmark missing production shapes — [what's absent]`

**P3 — Perf claim not reproducible** ⚠️
Missing: test script, ROCm version, GPU model, TP config, model checkpoint.
→ `⚠️ P3: perf claim missing reproduction info — [what's absent]`

**P4 — TP split shapes not covered** ⚠️
New attention / norm kernel tested only at full head count (TP=1 equivalent). At TP=4/8, `num_heads_q` / `num_heads_k` per device is divided by TP. A kernel that passes at H=128 may OOB at H=32 (TP=4) if shape math doesn't account for the split.
Trigger: new kernel taking `num_heads_q` / `num_heads_k`; PR test shows only one head count without a TP=4 or TP=8 variant.
→ `⚠️ P4: test covers only TP=1 head count — verify at num_heads÷TP=4 (e.g., [128→32])`

**P5 — Benchmark hides a cost real users pay on every call or cold start** ⚠️
The perf claim is measured with the timing window drawn so a *recurring* production cost is excluded: a first-call JIT compile on a path that is NOT cached across calls, or a setup step that runs on the live stream inside the timed region on every cold start. If that cost is real and recurring, omitting it can turn a net regression into an apparent speedup.
Do NOT fire on a genuinely one-time, amortizable setup that production pays once at model init — excluding weight shuffle/preshuffle, model weight loading, or a first-call JIT whose result is cached forever from steady-state per-call latency is CORRECT methodology, not deception. `warmup_iters` before a steady-state loop is standard and by itself is not P5.
FP self-check (do this before firing): is the excluded cost paid **once per deployment** (amortizable → do NOT fire) or **again on every call / every cold start / inside the timed stream** (→ fire)? If you cannot show it recurs, do not fire.
Counter-example (does NOT trigger P5): aiter#4166 preshuffles the static weight once outside the timing loop and honestly reports a geomean 0.69x result — a correct steady-state benchmark, not a hidden cost. Charging that one-time shuffle against a single call to manufacture a "regression" is itself the false positive this rule must avoid.
→ `⚠️ P5: timing window excludes [cost] that recurs per call / per cold start — re-run including it, or confirm it is one-time amortizable`

**P6 — Kernel change whose cost nobody measured** ⚠️
P1–P5 all grade the numbers *the PR supplies*. None of them produces a number, so a kernel PR can clear the whole Performance block on the strength of a table nobody re-ran. Correctness evidence does not cover this: `correctness_repo_tests: pass` means the kernel computes the right values and says nothing about how long it takes.

**Where the measurement now comes from.** The validator has a `perf` stage. When it runs, it times the target on base and on head back to back on the same locked GPU — base with the patch reversed out on the same worktree — and reduces the pair to `median_ratio`, the head speedup over base, oriented so `<1` is always a regression. That is a deterministic result, not an advisory one: below `threshold` (default 0.95, over ≥3 matched rows, both sides exiting cleanly) it writes a `should-fix` finding and the report's verdict becomes `NEEDS_WORK`. Read it out of `stages.perf`; do not re-derive it.

Trigger: Step 1's triage printed `perf triage: REQUIRED` — i.e. the PR changes runtime surface — **and** the head-matched report carries no usable `stages.perf` (absent, or `status: skip`). That is the gap P6 exists to name. If `stages.perf` is `pass` or `fail`, the measurement happened; report the number and do not fire.

Why the stage skips, and what each case means for the card:
- **no benchmark entry point in the target** — the honest common case (26 of aiter's 123 `op_tests/` targets have no timing harness at all). Fire P6, and say the target cannot be timed as written.
- **the PR adds the target** — base has nothing to compare against. Fire P6 with that reason; a head-only number is not a comparison.
- **nonzero exit on either side** — deliberately never a regression, because a truncated log yields a meaningless ratio. Fire P6 and treat the crash as the more interesting finding.
- **fewer than 3 matched rows / no shared timing column** — the two sides measured different things. Fire P6.

**The measurement that counts is base vs head, on this box, back to back.** Running only head against whatever baseline the PR chose reproduces the PR's own comparison; it cannot show a regression, and it silently inherits any staleness in that baseline. When the stage could not run, Step 1 prints the exact two-command manual form for the triaged target.
What to report: the shapes measured, both sides' numbers with units, and the delta. If nothing ran, say why — no idle GPU, no benchmark entry point, arch not available here — and mark every perf statement in the card `[inferred]`.
FP self-check: do NOT fire when the triage says NOT REQUIRED (no runtime surface), when `stages.perf` is `pass`/`fail`, when a manual measurement was taken and reported, or when the PR's own harness is the thing under test and has no steady state to measure. A single sample on a shared box is weak evidence — report the spread or the sample count rather than one number.
Real example (aiter#4538): a FlyDSL kernel whose entire justification was perf was reviewed to `Validation: PASS` with zero timing data; the perf finding stopped at "reviewer should ask", and a maintainer had already posted on the PR that a competing kernel beat it. Measuring afterwards took two `--scenario bench` runs and showed the fused path is where the PR's gain comes from — which the review should have carried in the first place. That gap is what the `perf` stage now closes automatically.
→ `⚠️ P6: kernel changed and no base-vs-head timing exists — [why the perf stage could not run]; measure [target] on both sides, or mark the perf findings [inferred]`

---

### Housekeeping (quick scan)

| Check | Trigger | Flag |
|---|---|---|
| Temp script committed | `.sh`, `runperf*.py`, `test_local_*.py` in diff | `⚠️ HK1: [file] looks temporary — remove before merge` |
| Unrelated files | Files with no connection to PR purpose | `⚠️ HK2: [file] appears unrelated` |
| `sys.path` at module level | `sys.path.insert(` / `sys.path.append(` in non-test `.py` | `⚠️ HK3: sys.path mutation — use relative imports` |
| kpack:1 in gfx950 config | `kpack: 1` in added YAML/CSV for gfx950 | `📝 HK4: kpack:1 on gfx950 is anti-pattern` |
| N-th op variant | 3rd+ variant of same op family | `📝 HK5: consider unified API — [N]th variant of [op]` |
| No UT for new op | New Triton/HIP op, no `op_tests/test_*.py` | `📝 HK6: new op needs UT following aiter-op-test format` |
| TODO/stub in new path | `# TODO`, `# FIXME`, `raise NotImplementedError`, lone `pass` on a `+` line inside a new branch | `⚠️ HK7: [location] — incomplete implementation in new code path` |
| `develop=True` on new op | `@compile_ops(..., develop=True)` in added code | `⚠️ HK8: develop=True bypasses JIT cache — remove before op leaves experimental` |
| New permanent env-var knob | `os.environ.get("AITER_...` on a `+` line that adds a lasting behavioral flag | `⚠️ HK9: new env-var knob [NAME] — aiter generally rejects permanent env flags (AITER_MOE_FORCE_BF16_ACT in #3593 was reverted by #4225); prefer auto-deriving from dtype/arch/shape. Acceptable only as a temporary rollback kill-switch (see D2), and then must be documented.` |
| Test reference dtype promotion | New test reference impl uses Python float literal (`1.0 + weight`, `0.5 * x.float()`) or explicit upcast (`.to(torch.float32)`, `.double()`) promoting to fp32 while kernel runs in bf16/fp8 — comparison calibrated against wrong-precision baseline | `⚠️ HK10: reference [fn] promotes to fp32 — cast back to [kernel dtype] before comparison` |
| New third-party dependency | New package in `requirements*.txt`, `setup.py`, `pyproject.toml`; or new top-level `import [pkg]` not already a project dep. Exception: ROCm system packages (`amdsmi`, `hip`, `rccl`) are intentionally not on PyPI — flag only if there is no `try/except ImportError` guard AND no comment explaining the ROCm-only dependency | `📝 HK11: new dependency [pkg] — add to requirements, or add try/except ImportError with a comment for ROCm system packages` |

---

## Step 6 — AI Code Diagnostic

For each question below, note if the answer is a warning sign:

| Question | Warning sign |
|----------|-------------|
| Does description explain mechanism (WHY) or just action (WHAT)? | Only WHAT → elevated risk |
| Are perf numbers suspiciously clean? (exact 2.0x, 1.5x, 3.0x) | Could be cherry-picked or fabricated |
| Are perf claims only trace screenshots with no numeric values? | Screenshots ≠ numbers; reviewer will ask |
| Does the test only cover M=1 or M=16? | AI defaults to toy shapes |
| Are gated-off parameters asserted or silently ignored? | Silent → B1 violation |
| Does code introduce `sys.path`, `os.environ` mutations at module level? | Global state leak → HK3 |
| Were unrelated files committed alongside the actual change? | AI commit artifact → HK2 |
| Is the new default path revertible? | No env-var gate → D2 violation |
| Is "Test Plan" / "Test Result" section left as template comment? | Empty = untested, AI-generated description |
| PR description footer says "🤖 Generated with Claude Code" or similar AI attribution? | Author may not understand the change — elevated manual review priority |

**Structural verification — the table above is a cheap pre-filter; a clean description does not make the code correct.** When the diff touches code, AI fails in specific *structural* ways. Run these checks and report each as a finding tagged `[verified]`/`[inferred]`, ending in an action verb (per the finding format below):

1. **Hallucinated-symbol sweep.** List every symbol NEW to this diff — function name, kwarg, enum/constant, attribute, import — and grep each against its real definition. AI invents plausible names and signatures that do not exist or do not match. Any symbol you cannot locate in source/API is a defect until proven real.
   → `🔴 [symbol] on [line] not found in [module] / signature mismatch — confirm it exists or it is a hallucinated API`
2. **Twin divergence (copy-paste half-adapted).** Identify mirrored code — fwd/bwd, v2/v3, prefill/decode, gfx942/gfx950. Compare field by field; any asymmetry (one side int64 the other int32, one masked the other not, one stride order flipped) is an unfinished copy. This is the signature AI kernel bug (cf. D9's fwd-int32/bwd-int64 case).
   → `🔴/⚠️ [detail] differs between [twin A] and [twin B] — copy-paste left [side] unadapted`
3. **Claim/comment ↔ code, and number provenance.** Does the code actually enforce the invariant the description or a comment asserts? Then take the single most impressive number in the PR and trace it to its source (script output / log line). A number or PR/issue citation you cannot trace is `[unverified]` — never repeat it as fact. (This skill's own P5 once shipped a fabricated "1.14x" for aiter#4166 that the PR never claimed — verify, do not trust.)
   → `⚠️ [claim/comment] asserted but code does [X]; or [number] not traceable to any output — mark [unverified] and ask for the source`
4. **Safety theater.** For each new `if`/`try`/`assert` guard: is it reachable, will it ever fire, does `except: pass` swallow a real error? AI adds defensive code that is unreachable or silently hides failures.
   → `⚠️ guard on [line] is [unreachable / swallows errors] — remove it or make it actually enforce the invariant`
5. **Test calibrated to pass, not to falsify.** Is the reference impl structurally a twin of the kernel (the same bug lives in both, so they always agree)? Is `atol`/`rtol` loosened with no justification? Does it assert against the kernel's own output? AI writes tests that pass rather than tests that could catch a regression.
   → `⚠️ test [name] cannot fail because [mirrored ref / loose tol / self-comparison] — replace with an independent oracle`
6. **Magic constant without derivation.** A new tile size / threshold / epsilon / literal — is there a stated derivation or tuning basis, or does it merely look plausible?
   → `📝 constant [value] on [line] has no stated derivation — ask for the tuning/source basis`

If 3+ table signs OR any structural check above fires: note "elevated AI code risk — recommend thorough manual verification of the dispatch logic and test coverage." Regardless of the table count, when the diff changes code the structural checks are mandatory — a clean, well-written description is itself something AI produces easily.

---

## Step 7 — Free-Form Review

After the rule checklist, read the diff as a domain expert:
- Does the approach make sense given the hardware constraints?
- Are there correctness concerns not caught by the rules above?
- LDS limits: gfx942 = 64KB, gfx950 = 64KB per CU. gfx1250 (RDNA4) has 320KB LDS per CU but `ds_read`/`ds_write` immediate offset is only 16-bit (max 65535 = 64KB). If LDS allocation exceeds 64KB on gfx1250, the compiler uses VGPRs for the LDS address → VGPR spill → perf regression or compile failure. Real example (PR#4031): reviewer caught OPUS kernel on gfx1250 would hit this.
- For new Triton kernels: BLOCK_SIZE choices, num_warps, num_stages — are they reasonable for MI300X? Large BLOCK_SIZE can push LDS over limit causing test failures (Real example: PR#3808, 10 LDS-exhaustion failures in Triton batched GEMM configs).
- `.contiguous()` before kernel calls when tensor may have non-standard strides?
- For mixed FP8 dtype paths (fn vs fnuz): gfx942 KV cache is fnuz by default, but Q quantization may emit fn (e.g., DSv4 Flash fused indexer). A kernel handling mixed fn/fnuz inputs needs explicit dtype dispatch — silent dtype mismatch compiles but produces wrong values. Real example (PR#3913): reviewer asked "why is there a mixed FN/FNUZ path?" and asked for `if arch == "gfx942":` guard on the fnuz *conversion* path.
- For FlyDSL/assembly kernels: hardware tile size constants (MFMA M=16, N=16, K=32 for MI300X FP8) should be named constants, not raw magic numbers (16, 32) scattered across the kernel. Real example (PR#3913): vpietila asked "add named constants MFMA_M=16, MFMA_N=16, MFMA_K=32 and use them throughout."

---

## Step 7.5 — Blind-Spot Check

Before writing the verdict, answer this one question in full:

**"Is there any correctness risk, resource hazard, or behavioral edge case in this diff that none of Steps 1–7 above caught?"**

If the answer is yes, add it to the findings. If the answer is no, proceed.

---

## Step 8 — Verdict

**Output rules (strictly enforced):**
- Run Steps 1–7 internally. Do NOT narrate steps, do NOT show checklists, do NOT show which rules fired.
- Output ONLY the card below. Nothing before it, nothing after it.
- If there are no findings, the findings section is omitted entirely.
- "What it does" must be one sentence, written for a reviewer who hasn't read the diff.
- **At most 5 findings, ordered most-severe first.** Rank by (severity, then blast radius), keep the top 5, and drop the rest — do not append them as a tail. This is a readability limit, not a measured recall claim; no committed replay corpus currently establishes recall@5.
- **State the validation evidence** on the line under the verdict, using the state Step 1's triage
  actually reached. The three no-report states are different facts and must not be merged:
  - with an accepted exact-head report: `Validation (deterministic): <verdict>` plus selected target/runner, runtime arch, and failed/skipped stages. Say when the report came from the auto-run, because its ceiling is lower: with no route supplied the receipt and grid stages skip, so `INCONCLUSIVE` there describes what a diff can tell you and is not a finding against the PR.
  - triage said not required: `Validation (deterministic): N/A — no runtime surface changed`. Do not write `NOT RUN`; there is no gap to report, and a docs or tooling PR carrying an alarming evidence line is what makes the line ignorable.
  - required, but no target existed to run: `Validation (deterministic): NOT RUN — <triage reason>`. A runtime change shipping no test target is also a finding in its own right.
  - required and a target existed, but the run could not happen (no idle GPU, validator missing, `REVIEW_AUTO_VALIDATE=0`): `Validation (deterministic): NOT RUN — <reason>`. This is an environment gap, not a PR defect.
  - In every `NOT RUN` and `N/A` state, no finding may assert runtime behaviour (perf, accuracy, launch failure) as fact; such findings are `[inferred]` and phrased as questions.
- **State the perf evidence on its own line, always.** A `Validation` verdict covers correctness
  only, so a card that carries just that line reads as clearance for a kernel whose latency
  nobody measured. The line goes in the header block, never as a finding — the 5-finding cap
  must not be able to evict it. Two tiers, and the label says which one you are in:
  - **the report carries `stages.perf` with status `pass` or `fail`** — this is deterministic
    evidence, from base vs head on one locked GPU with the patch reversed for base, and it
    ships its own reproducer. Write `Perf (deterministic): <verdict> — median_ratio <n> on
    <worst_column> over <matched_rows> rows, threshold <t>`, where the verdict is `REGRESSION`
    for `fail` and `NO REGRESSION` for `pass`. `median_ratio` is the head speedup over base,
    so `<1` is slower. Quote `worst_column`: the ratio is the minimum across columns, so
    naming the column that moved is what makes the number checkable.
    A `fail` here already produced a `should-fix` finding and put the report at `NEEDS_WORK`;
    report that as the deterministic result it is, not as a suggestion.
  - **no usable `stages.perf`** — anything you measured by hand is advisory. Write
    `Perf (advisory): ...` and use the state Step 1's perf triage reached (see P6):
    - measured by hand: `Perf (advisory): MEASURED — <shapes>, base <n> vs head <n> <units>, <delta>`. Base and head, same box, back to back. Say how many samples, and say if only head was run — head-only reproduces the PR's own comparison and cannot show a regression.
    - triage said not required: `Perf (advisory): N/A — no runtime surface changed`.
    - required, but the target ships no benchmark entry point: `Perf (advisory): NOT RUN — <triage reason>`. A kernel PR with no runnable perf harness is also a finding in its own right.
    - required and a harness existed, but the run could not happen (no idle GPU, wrong arch, nonzero exit, out of time): `Perf (advisory): NOT RUN — <reason>`. An environment gap, not a PR defect.
    - In this tier the line is advisory in both directions: a slower hand-measured number is not a gate, and `MEASURED` is not clearance — one run on a shared box is weak evidence, so report the sample count with it.
  - Never label a hand-run number `(deterministic)`, and never soften a `stages.perf` `fail`
    into `(advisory)`. The label is the reader's only signal for whether a reproducer exists.
- The review line is always advisory. `🔴 HIGH RISK` requests human attention; it is not a merge gate. A deterministic `Validation: BLOCK` may gate because its reproducer is in the report.

```
## [repo] PR #NNN — [title]

**[One sentence: what this PR does, in plain terms.]**

Review (advisory): [✅ NO FINDINGS | ⚠️ NEEDS WORK | 🔴 HIGH RISK]
Validation (deterministic): [PASS/NEEDS_WORK/BLOCK/INCONCLUSIVE — target, exact runtime, and skipped-stage evidence | N/A — no runtime surface changed | NOT RUN — reason]
Perf (deterministic): [REGRESSION | NO REGRESSION — median_ratio N on <worst_column> over N rows, threshold T]
  ...or, when the report carries no usable stages.perf, this line instead:
Perf (advisory): [MEASURED — shapes, base vs head with units, delta, sample count | N/A — no runtime surface changed | NOT RUN — reason]

🔴 [specific finding — what, where, why it matters]
⚠️ [specific finding]
📝 [note]
```

Each finding must have **three parts**:
1. **Problem** — what exactly is wrong, with file/line if relevant
2. **Impact** — what goes wrong at runtime if this is not fixed (wrong output / crash / perf regression)
3. **Action** — end with a verb phrase: "**Author must** [do X]" or "**Reviewer should ask** [Y]" — no verb = incomplete finding, do not include

**Tag every finding [verified] or [inferred], and never ship a root cause you only inferred.**
- `[verified]` — traced to the actual code/evidence chain (aiter#4029: fp4 auto-K-split confirmed by following `_is_csa_indexer_fp4` → the auto branch → no gate rejects it).
- `[inferred]` — plausible but unconfirmed; say so and downgrade to "worth checking," do not assert it as the cause (aiter#2565: "w1/w2 shuffle asymmetry is the MI35X root cause" was inferred and likely wrong — it may be a legitimate stage1/stage2 layout difference).
A finding that stops at "likely / probably the root cause" without an evidence chain is not shippable — either trace it to [verified] or label it [inferred] and frame it as a question.

Do NOT use rule codes (P1, D4, A1…) in output — they are internal labels only.

Examples of good findings:
- `🔴 fused_qk_norm_rope_cache_quant.py:463 changes torch.zeros → torch.empty, but the old comment says "trailing pad must be zero for asm reader" and the new comment claims "never read" — if padding IS read, every quantized output is corrupted. **Author must** cite the asm spec or a test proving padding is not read.`
- `⚠️ PR claims fp8 latency is now 1.3–1.5x better, but the benchmark starts timing after shuffle_weight() completes — users pay that cost on every cold start. **Author must** re-run with shuffle_weight included in the timing window and confirm the result is still positive.`
- `⚠️ Chunked indexer logic is copy-pasted verbatim into deepseek_v2.py and deepseek_v4.py. If v4's variable semantics differ, the formula silently produces wrong KV offsets for v4 callers. **Author must** confirm correctness was verified independently under v4's variable layout.`
- `📝 No corresponding ATOM consumer PR mentioned. **Reviewer should ask** who will pass emit_bf16=True to activate this path.`

Examples of bad findings (too vague, no action verb):
- `⚠️ Missing perf numbers` — no impact stated, no action
- `🔴 D4 violation` — rule code means nothing to a reviewer
- `⚠️ The benchmark may not include setup cost` — no "Author must" conclusion

---

## Adding New Rules

When a human reviewer catches something real that this skill missed:
1. Add it to Step 5 with a real PR example as evidence
2. Increment the rule number
3. Commit with message: `review-pr: add R[N] from PR#[NNN] — [one line description]`

The skill grows from real review history, not hypothetical patterns.
