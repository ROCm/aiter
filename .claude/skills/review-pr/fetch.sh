#!/usr/bin/env bash
# Step 1 + 1b of the review-pr skill, as a program rather than as 785 lines of shell
# inside a document. It was in SKILL.md, where a reviewer had to read past it to reach the
# 299 lines that are actually instructions -- and where it made the entry file grow 487 ->
# 632 -> 1372 lines while the instruction content barely moved.
#
# usage: fetch.sh <PR|owner/repo#N> [owner/repo] [validation-report]
# Everything it prints is meant to be read. It ends by printing the scratch dir; every later
# step of the skill names artifacts inside it.

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

# Every git fetch below names this URL, never the remote called `origin`. A clone whose
# origin is a fork -- the normal state for anyone who has ever pushed a branch -- resolves
# `origin refs/pull/N/head` against the FORK's pull refs, silently fetching a different PR
# or none at all. The repo under review is $REPO and nothing else.
REPO_URL="https://github.com/$REPO"

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
git fetch -q "$REPO_URL" "refs/pull/$PR/head" 2>/dev/null || \
  echo "note: could not fetch the PR head; the index-width scan will report unscanned files" >&2

# Both SHAs, from the API rather than from local refs, and defined ONCE here so every later
# step names the same two commits. Step 1b referenced $BASE_SHA/$HEAD_SHA without either ever
# being assigned: under `set -u` that aborts the whole block, so the cross-file evidence
# collector never ran for anybody, and neither did anything after it.
BASE_SHA=$(cat "$WORK/base_head.txt")
HEAD_SHA=$(python3 -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["headRefOid"])' \
  "$WORK/pr_meta.json")
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
    || git -C "$PROJECT_ROOT" fetch -q "$REPO_URL" "$AUTO_BASE" 2>/dev/null
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

# Which rule families this diff actually triggers. Step 3 used to be a prose
# checklist the model ticked itself; measured over 597 open aiter PRs that meant
# reading all 44 rules every time. Derived structurally it is 12 at the median.
PR_TITLE=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('title') or '')" \
           "$WORK/pr_meta.json")
"$SKILLS_ROOT/review-pr/triage.py" rules "$WORK/pr.diff" "$PR_TITLE" | tee "$WORK/rules.txt"

# The full text of exactly those rules, and nothing else. The rule bodies are 383 lines in
# rules.md and the median PR needs 103 of them; shipping all of them in the skill and telling
# the reviewer to ignore 73% is not the same as not shipping them.
"$SKILLS_ROOT/review-pr/triage.py" expand "$WORK/rules.txt" \
  "$SKILLS_ROOT/review-pr/rules.md" > "$WORK/rules_expanded.txt" || {
  echo "rule bodies could not be expanded; rules.md has drifted from the deriver" >&2
  exit 1
}

# For the families that turn on cross-file context, fetch that context NOW rather
# than reminding the reviewer to go and read it. A deleted guard is judged by how
# the symbol is handled on head, not by the hunk that removed it.
if grep -q "invariant-removed\|api-signature" "$WORK/rules.txt"; then
  # Materialise the PR's own head images under $WORK. Prefixing the changed paths with
  # $PROJECT_ROOT read the reviewer's WORKING TREE instead -- whatever branch they happen to
  # have checked out, with whatever uncommitted edits -- so the same PR produced different
  # evidence on two machines, and evidence for a PR that was never checked out at all.
  mkdir -p "$WORK/head"
  HEAD_FILES=""
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    dst="$WORK/head/$f"
    mkdir -p "$(dirname "$dst")"
    if git -C "$PROJECT_ROOT" cat-file -e "$HEAD_SHA:$f" 2>/dev/null \
       && git -C "$PROJECT_ROOT" show "$HEAD_SHA:$f" > "$dst" 2>/dev/null; then
      HEAD_FILES="$HEAD_FILES $dst"
    else
      echo "note: $f unavailable at head $HEAD_SHA; not in the evidence set" >&2
    fi
  done < <(git -C "$PROJECT_ROOT" diff --name-only "$BASE_SHA...$HEAD_SHA" 2>/dev/null \
           || grep '^+++ b/' "$WORK/pr.diff" | sed 's|^+++ b/||')
  if [ -n "$HEAD_FILES" ]; then
    "$SKILLS_ROOT/review-pr/triage.py" evidence "$WORK/pr.diff" $HEAD_FILES \
      | tee "$WORK/evidence.txt"
  else
    echo "SKIPPED: no head images available; cross-file evidence not collected" \
      | tee "$WORK/evidence.txt"
  fi
fi

# Every first-party import the diff ADDS, resolved against the branch this PR merges
# INTO -- fetched fresh, not the PR base and not whatever is on disk. A stale root
# makes this check silently pass; that is the whole failure mode it exists to catch.
# $BASE_SHA is the base branch tip read from the API in Step 1, not a local ref and not the
# PR's historical merge base -- the two differ by exactly the window in which this class of
# breakage happens.
git -C "$PROJECT_ROOT" cat-file -e "${BASE_SHA}^{commit}" 2>/dev/null \
  || git -C "$PROJECT_ROOT" fetch -q "$REPO_URL" "$BASE_SHA" 2>/dev/null || true
TARGET_WT="$WORK/merge-target"
if git -C "$PROJECT_ROOT" worktree add --detach "$TARGET_WT" "$BASE_SHA" >/dev/null 2>&1; then
  "$SKILLS_ROOT/review-pr/triage.py" symbols "$WORK/pr.diff" "$TARGET_WT" \
    | tee "$WORK/symbols.txt"
  git -C "$PROJECT_ROOT" worktree remove --force "$TARGET_WT" >/dev/null 2>&1 || true
else
  echo "SKIPPED: could not check out base $BASE_SHA; symbol sweep not run" \
    | tee "$WORK/symbols.txt"
fi

echo
echo "WORK=$WORK"
echo "artifacts: pr.diff pr_meta.json base_head.txt rules.txt rules_expanded.txt \
evidence.txt symbols.txt validation_requirement.json"
