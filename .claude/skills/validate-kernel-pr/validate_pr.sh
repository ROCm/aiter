#!/usr/bin/env bash
# S1 validate-kernel-pr -- deterministic validation layer for kernel PRs.
#
# Produces validation_report.json: the evidence base every review finding must hang on.
# Design rules it enforces (each learned from a real failure mode):
#   * isolation is REPORTED, never assumed  -- no docker here, so: worktree + private caches
#   * arch coverage is REPORTED, never implied -- a gfx950 box cannot validate a gfx942 claim
#   * a test is run and where it CAME FROM is reported -- a test the PR ships was written by
#     the same hand as the code it grades, and a change nothing runs at all is a blocker
#   * an extra shape grid is OPTIONAL and earns nothing -- it can only ever add a failure.
#     While it was required, every caller supplied one whether or not it covered anything the
#     target did not already run, and aiter#4538's duplicate was credited as new coverage
#   * a green pytest with loosened tolerances is not a pass -- tolerances are policy-checked
#   * GPU is claimed over a sampling window and locked (kernel-profiling-optimization skill)
#
#   * correctness is not performance -- a kernel PR can compute the right values and still
#     be a regression, so base and head are also timed, on the same locked GPU, back to back
#
# usage: validate_pr.sh --repo <worktree> (--target <test file or node> | --no-target <reason>)
#                       [--patch p.patch]
#                       [--head-sha <expected PR head>] [--shape-env VAR]
#                       [--grid "M,N,dt;..."] [--tol-table f32=1e-5,...]
#                       [--perf-args "--scenario bench"] [--no-perf]
#                       --expected-route NAME [--label NAME] [--out report.json]
set -uo pipefail

REPO_WT=""
TESTS=""
PATCHF=""
HEAD_SHA=""
SHAPE_ENV=""
GRID=""
EXPECTED_ROUTE=""
SHAPE_VARS=""
SHAPE_ARG=""
SHAPE_ARGNAMES=""
# Extra independent test axes, each `NAME=FLAG:v1;v2;...`. The shape grid is one ordered
# tuple on one channel, which is the whole of what a target's shape flag accepts; a target
# whose remaining knobs are separate flags -- head counts, dtypes, window modes -- could not
# be gridded over them at all, so entire failing configurations were unreachable however the
# grid was spelled. On ROCm/aiter#4538 that is `--num-heads`, whose default is `64 128`, and
# the public API asserts at num_heads=16 in a configuration the validator could not request.
# Force the runner instead of inferring it. The classifier is structural and can be
# wrong in both directions; when it is, a caller who can see the target should be able
# to say so rather than having a runner-selection artefact charged to the PR author.
RUNNER_OVERRIDE=""
RUNNER_REASON=""
# The caller's declaration that they looked for a test exercising this change and found none.
NO_TARGET_REASON=""
AXES=()
AXIS_CLI=()
AXIS_CLI_OVERRIDE=()
AXIS_REPORT="[]"
TOL_TABLE=""
LABEL="run"
OUT=""
PYLIB="${PYLIB:-}"
TIMEOUT="${TIMEOUT:-1800}"
# Perf measurement is on by default. It has to be: the regression this stage exists to catch
# is the one nobody suspected, and an opt-in flag is only ever set by someone who already
# suspects. --no-perf turns it off for the cases where it genuinely cannot work.
PERF_ENABLED=1
PERF_ARGS=""
PERF_ARGS_SET=0
PERF_BASIS=""
# A bench sweep is legitimately longer than a correctness run, so it gets its own budget.
PERF_TIMEOUT="${PERF_TIMEOUT:-$TIMEOUT}"
# Each side is run PERF_REPEAT times and each cell reduced to its best sample. This is not
# belt-and-braces, it is what makes a 0.95 threshold usable. Measured here: five warm repeat
# runs of an unchanged op_tests/test_layernorm2d.py gave `ck avg` of
# 13.10 20.98 20.70 13.28 13.17 us -- bimodal, 1.60x spread, on code that did not change,
# while the untouched `torch avg` reference column held to 1.03x. One run per side would put
# the ratio anywhere in [0.62, 1.60] and fire a false regression about half the time.
# Minimum-over-three collapses that same data to 1.014x. Minimum is the correct estimator
# because contention, clock ramp and scheduling only ever ADD time.
PERF_REPEAT="${PERF_REPEAT:-3}"
# 0.95 is the user-facing sensitivity knob and holds only because of the reduction above.
# Both guards below exist because the threshold is tight -- it fires on >= PERF_MIN_ROWS
# matched rows and never on a nonzero exit, so a crashed or truncated run reports `skip`.
PERF_THRESHOLD="${PERF_THRESHOLD:-0.95}"
PERF_MIN_ROWS="${PERF_MIN_ROWS:-3}"
# Name of a timing column the patch does not touch -- typically a reference implementation
# the target times alongside the kernel under test. Required before a TRANSPLANTED baseline
# is believed, because that comparison spans two trees and this column is the only evidence
# that the two runs are comparable at all. Unused for the ordinary same-worktree baseline.
PERF_CONTROL_COLUMN=""
PERF_CONTROL_TOL="${PERF_CONTROL_TOL:-0.10}"
PERF_BASELINE_METHOD="patch-reversed-same-worktree"
# The file the timing runs execute. It defaults to the correctness target, and until now it
# WAS the correctness target -- run_perf simply reused $TEST_FILE. Naming it separately
# changes nothing by itself; what it does is force the places that ask "is the target on
# base?" to say WHICH target. Four of them read a state computed from the correctness
# target, two write to the worktree, and one deletes from it.
PERF_TARGET=""
# How the validator came to be timing that file. A caller who names it has read the diff;
# falling back to the correctness target is the validator's own inference, and the two are
# not the same kind of evidence -- the same distinction runner_basis and test_provenance
# already draw. Absent any measurement, this is what tells a reader whose choice it was.
PERF_TARGET_BASIS="same-as-correctness-target"
PERF_TARGET_PROVENANCE="unknown"
PERF_TARGET_PROVENANCE_REASON=""
# Discovery can return two targets -- the repository's bench and the one the PR ships -- and
# both get timed. Everything above describes ONE target and stays that way: PERF_TARGET is
# what perf_detect and run_perf act on, and the loops below set it per iteration. What the
# loops accumulate lives in these index-aligned arrays instead. Parallel arrays are a poor
# record type, and bash offers no better one; the arrays exist only to carry results to the
# manifest, where scrape_perf.py assembles them back into objects.
PERF_TARGETS=()
PERF_TARGET_BASES=()
PERF_TARGET_BASIS_REASONS=()
PERF_TARGET_PROVENANCES=()
PERF_TARGET_PROVENANCE_REASONS=()
TARGET_PYTHON="${PYTHON_BIN:-$(command -v python3 || command -v python || true)}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

need_value() {
  if [ "$#" -lt 2 ]; then
    echo "missing value for $1" >&2
    exit 2
  fi
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --repo) need_value "$@"; REPO_WT="$2"; shift 2;;
    --target) need_value "$@"; TESTS="$2"; shift 2;;
    --tests) need_value "$@"; TESTS="$2"; shift 2;;
    --no-target) need_value "$@"; NO_TARGET_REASON="$2"; shift 2;;
    --patch) need_value "$@"; PATCHF="$2"; shift 2;;
    --head-sha) need_value "$@"; HEAD_SHA="$2"; shift 2;;
    --shape-env) need_value "$@"; SHAPE_ENV="$2"; shift 2;;
    --grid) need_value "$@"; GRID="$2"; shift 2;;
    --expected-route) need_value "$@"; EXPECTED_ROUTE="$2"; shift 2;;
    --shape-vars) need_value "$@"; SHAPE_VARS="$2"; shift 2;;
    --shape-arg) need_value "$@"; SHAPE_ARG="$2"; shift 2;;
    --shape-argnames) need_value "$@"; SHAPE_ARGNAMES="$2"; shift 2;;
    --axis) need_value "$@"; AXES+=("$2"); shift 2;;
    --runner) need_value "$@"; RUNNER_OVERRIDE="$2"; shift 2;;
    --runner-reason) need_value "$@"; RUNNER_REASON="$2"; shift 2;;
    --tol-table) need_value "$@"; TOL_TABLE="$2"; shift 2;;
    --label) need_value "$@"; LABEL="$2"; shift 2;;
    --out) need_value "$@"; OUT="$2"; shift 2;;
    --perf-args) need_value "$@"; PERF_ARGS="$2"; PERF_ARGS_SET=1; shift 2;;
    --perf-target) need_value "$@"; PERF_TARGET="$2"; PERF_TARGET_BASIS="declared-by-caller"; shift 2;;
    --perf-control-column) need_value "$@"; PERF_CONTROL_COLUMN="$2"; shift 2;;
    --no-perf) PERF_ENABLED=0; shift;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac
done

if [ -z "$REPO_WT" ]; then
  echo "--repo is required" >&2
  exit 2
fi
# A PR with runtime surface and no test that exercises it is a finding about the PR, and it used
# to be a usage error: the caller who looked and found nothing had no way to say so, so the run
# died before writing a report and the PR went unvalidated rather than red.
#
# It stays an error to supply NEITHER, because a forgotten --target must not read as "there is no
# test" -- that would publish a caller's slip as a blocker against the author, which is the one
# mistake this file spends the most lines avoiding. The absence has to be DECLARED, with a reason,
# exactly like the runner is.
if [ -n "$TESTS" ] && [ -n "$NO_TARGET_REASON" ]; then
  echo "--target and --no-target contradict each other; supply one" >&2
  exit 2
fi
if [ -z "$TESTS" ] && [ -z "$NO_TARGET_REASON" ]; then
  echo "one of --target or --no-target <reason> is required" >&2
  exit 2
fi
if ! git -C "$REPO_WT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "--repo is not a git worktree: $REPO_WT" >&2
  exit 2
fi
if [ -n "$PATCHF" ] && [ ! -r "$PATCHF" ]; then
  echo "--patch is not readable: $PATCHF" >&2
  exit 2
fi
# Resolve before anything changes directory. A relative --patch used to be read from the
# caller's cwd in one place and from the worktree in another; the mismatch surfaced as
# "patch does not apply to the current base" and a BLOCK verdict -- a fact about the
# invocation, published as a reproducible defect against the PR author.
if [ -n "$PATCHF" ]; then
  PATCHF=$(cd -- "$(dirname -- "$PATCHF")" && pwd)/$(basename -- "$PATCHF")
fi
if [ -n "$OUT" ]; then
  OUT=$(cd -- "$(dirname -- "$OUT")" && pwd)/$(basename -- "$OUT")
fi
if [ -n "$SHAPE_ARGNAMES" ] && [ -n "$GRID" ]; then
  if ! python3 - "$SHAPE_ARGNAMES" "$GRID" <<'PY'
import sys

names = [part.strip() for part in sys.argv[1].split(",") if part.strip()]
rows = [row for row in sys.argv[2].split(";") if row.strip()]
bad = [row for row in rows if len(row.split(",")) != len(names)]
if bad:
    print(
        f"--grid rows must have {len(names)} cells to match --shape-argnames "
        f"{','.join(names)}; offending rows: {bad}",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
  then
    # Checked here and not inside a phase. The arity check used to live in the plugin
    # generator, whose exit status run_pytest never read: the stale plugin from the previous
    # phase survived, head-grid re-ran the invalid-grid sentinel, and its failure was
    # published as "the PR adds this target and its independent shape grid fails".
    echo "--grid does not match --shape-argnames" >&2
    exit 2
  fi
fi
if [ -n "$HEAD_SHA" ] && [[ ! "$HEAD_SHA" =~ ^[0-9a-fA-F]{40}$ ]]; then
  echo "--head-sha must be a full 40-character commit OID" >&2
  exit 2
fi
if [[ ! "$TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
  echo "TIMEOUT must be a positive integer" >&2
  exit 2
fi
if [[ ! "$PERF_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
  echo "PERF_TIMEOUT must be a positive integer" >&2
  exit 2
fi
if [[ ! "$PERF_MIN_ROWS" =~ ^[1-9][0-9]*$ ]]; then
  echo "PERF_MIN_ROWS must be a positive integer" >&2
  exit 2
fi
if [[ ! "$PERF_REPEAT" =~ ^[1-9][0-9]*$ ]]; then
  echo "PERF_REPEAT must be a positive integer" >&2
  exit 2
fi
if ! python3 -c 'import sys; v=float(sys.argv[1]); sys.exit(0 if 0 < v <= 2 else 1)' \
    "$PERF_THRESHOLD" 2>/dev/null; then
  echo "PERF_THRESHOLD must be a number in (0, 2]" >&2
  exit 2
fi
if [ -z "$TARGET_PYTHON" ] || [ ! -x "$TARGET_PYTHON" ]; then
  echo "no executable target Python interpreter; set PYTHON_BIN" >&2
  exit 2
fi

: "${OUT:=$PWD/validation_report.json}"
mkdir -p "$(dirname "$OUT")"
# A run OWNS its output path. Leaving a previous run's report in place made `--out` a
# fallback source of truth: the process exit code was read back out of that file, so if this
# run died before finish_report copied its own report over it, the shell exited on the
# PREVIOUS run's verdict -- a stale `PASS` published as this PR's result.
rm -f "$OUT"
WORK=$(mktemp -d "/tmp/validate-kernel-pr-XXXXXX")
JSON="$WORK/report.json"
PROBE_DIR="$WORK/probe"
PROBE_MODULE="validation_probe_${RANDOM}_${RANDOM}"
mkdir -p "$PROBE_DIR"
REPORT_TOOL="$SCRIPT_DIR/report.py"
TARGET_TOOL="$SCRIPT_DIR/target_run.py"
python3 "$REPORT_TOOL" init "$JSON" "$LABEL"

# Every write to the report goes through report.py, which is its only writer. These wrappers
# keep the call sites unchanged; `--` guards values that begin with a dash.
jset_json() {
  python3 "$REPORT_TOOL" set --json -- "$JSON" "$1" "$2"
}

jset_string() {
  python3 "$REPORT_TOOL" set -- "$JSON" "$1" "$2"
}

stage_note() {
  python3 "$REPORT_TOOL" stage -- "$JSON" "$1" "$2" "$3"
}

# Reads one field out of any JSON blob a tool handed back. It sat 400 lines below its first
# caller until perf target discovery acquired an earlier one -- bash resolves a function only
# from definitions it has already executed, so the call failed with `command not found` and,
# with no `set -e`, the run carried on with an empty variable. Helpers live up here with the
# other helpers.
stats_field() {
  python3 "$TARGET_TOOL" stats-field "$1" "$2"
}

finding() {
  python3 "$REPORT_TOOL" finding -- "$JSON" "$1" "$2" "$3"
}

log_excerpt() {
  python3 - "$1" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.exists():
    print("log unavailable")
else:
    text = " ".join(path.read_text(errors="replace").splitlines()[-4:])
    print(text[:220])
PY
}

mark_runtime_coverage() {
  python3 "$REPORT_TOOL" coverage -- "$JSON" "$1" "$2" "$3"
}

finish_report() {
  python3 "$REPORT_TOOL" finish "$JSON" "$OUT"
}

# Two independent facts about the supplied worktree:
#   BASE_ACTIVE=1    the patch is currently reversed out, i.e. we are mid-baseline-run
#   PATCH_APPLIED=1  this process applied the patch and still owes the caller a revert
BASE_ACTIVE=0
PATCH_APPLIED=0
PATCH_STATUS=""
restore_head() {
  if [ "$BASE_ACTIVE" -eq 0 ]; then
    return 0
  fi
  if git -C "$REPO_WT" apply --check "$PATCHF" >/dev/null 2>&1 \
      && git -C "$REPO_WT" apply "$PATCHF" >/dev/null 2>&1; then
    BASE_ACTIVE=0
    return 0
  fi
  return 1
}
cleanup() {
  if [ "$PATCH_APPLIED" -eq 0 ]; then
    return
  fi
  if [ "$BASE_ACTIVE" -eq 1 ]; then
    # The baseline run already reversed the patch out, which is the state the
    # caller handed us; re-applying it here is what used to leave residue.
    PATCH_APPLIED=0
    return
  fi
  if git -C "$REPO_WT" apply -R --check "$PATCHF" >/dev/null 2>&1 \
      && git -C "$REPO_WT" apply -R "$PATCHF" >/dev/null 2>&1; then
    PATCH_APPLIED=0
  else
    echo "failed to revert the candidate patch in $REPO_WT; it is left applied" >&2
  fi
}
trap cleanup EXIT

record_gpu_activity_after() {
  if [ -z "$PICK" ]; then
    return
  fi
  ACTIVITY_AFTER=$(python3 "$SCRIPT_DIR/gpu_probe.py" activity "$PICK")
  if [[ "$ACTIVITY_AFTER" =~ ^[0-9]+$ ]]; then
    jset_json "stages.gpu_claim.gfx_activity_after_pct" "$ACTIVITY_AFTER"
  elif [ "$ACTIVITY_AFTER" = "unavailable" ]; then
    jset_string "stages.gpu_claim.post_run_note" \
      "post-run GFX activity is not reported by the activity API on this host"
  else
    jset_string "stages.gpu_claim.post_run_note" \
      "post-run GFX activity could not be recorded"
  fi
}

echo "=== validate-kernel-pr [$LABEL] ==="
jset_string "started_utc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
jset_json "isolation" \
  '{"level":"git-worktree + private caches","container":false,"reason":"tests run in the supplied worktree with private HOME and compiler caches"}'
jset_json "arch_coverage" '{}'
jset_json "arch_coverage_basis" '{}'
jset_json "degraded_mode" 'null'
jset_json "runtime_identity" 'null'
if [ -n "$TESTS" ]; then
  jset_string "test_selection.target" "$TESTS"
else
  # `null`, not `""`. An empty string is what an unset variable also produces, and the one thing
  # this field must distinguish is "nobody named a target" from "the caller looked and there is
  # none". The second is a declaration, and it is paired with test_provenance: none below.
  jset_json "test_selection.target" 'null'
fi
jset_string "test_selection.shape_env" "$SHAPE_ENV"
jset_string "test_selection.grid" "$GRID"
jset_string "test_selection.shape_arg" "$SHAPE_ARG"
jset_string "test_selection.shape_argnames" "$SHAPE_ARGNAMES"
jset_string "test_selection.expected_route" "$EXPECTED_ROUTE"
jset_string "test_selection.shape_vars" "$SHAPE_VARS"
jset_string "test_selection.runner" "unresolved"
jset_string "test_selection.runner_reason" "merge simulation has not completed"
# Provisional, for the same reason the runner above is: a run that exits before the patch is
# applied still owes the reader a legal report, and a required field that is simply absent
# reads as an oversight rather than as a stage that never got to run.
jset_string "test_selection.test_provenance" "unknown"
jset_string "test_selection.test_provenance_reason" \
  "merge simulation has not completed, so the target has not been compared against the patch"

# ---------- stage 1: merge simulation ----------
BASE_SHA=$(git -C "$REPO_WT" rev-parse HEAD)
INITIAL_IGNORED=$(git -C "$REPO_WT" status --porcelain \
  --ignored --untracked-files=all | awk '$1 == "!!"')
jset_string "repo.worktree" "$REPO_WT"
jset_string "repo.base" "$BASE_SHA"
if [ -f "$REPO_WT/aiter/__init__.py" ]; then
  REPO_KIND="aiter"
elif [ -f "$REPO_WT/python/flydsl/__init__.py" ]; then
  REPO_KIND="flydsl"
else
  REPO_KIND="unknown"
fi
jset_string "repo.kind" "$REPO_KIND"

if [ -n "$PATCHF" ]; then
  DIRTY=$(git -C "$REPO_WT" status --porcelain --untracked-files=all)
  if [ -n "$DIRTY" ] || [ -n "$INITIAL_IGNORED" ]; then
    stage_note "merge_sim" "skip" \
      "supplied worktree has tracked, untracked, or ignored artifacts; patch was not applied"
    finding "note" "merge_sim" \
      "worktree is not isolated-clean, so merge simulation is inconclusive"
    jset_json "repo.head" 'null'
    finish_report
    exit 2
  fi
  if git -C "$REPO_WT" apply --check "$PATCHF" >/dev/null 2>&1 \
      && git -C "$REPO_WT" apply "$PATCHF" >/dev/null 2>&1; then
    PATCH_APPLIED=1
    # Taken here and nowhere else: the worktree was verified clean four lines up, so this is
    # the only moment at which `git status` describes the patch and nothing else. Later stages
    # write caches, receipts and bench artifacts into the same tree.
    PATCH_STATUS=$(git -C "$REPO_WT" status --porcelain --untracked-files=all)
    stage_note "merge_sim" "pass" "patch applies cleanly to the recorded base"
    jset_string "repo.patch_sha256" "$(sha256sum "$PATCHF" | awk '{print $1}')"
    if [ -n "$HEAD_SHA" ]; then
      jset_string "repo.head" "$HEAD_SHA"
    else
      jset_json "repo.head" 'null'
      jset_string "stages.merge_sim.identity_note" \
        "no --head-sha supplied; report cannot be matched to a remote PR head"
    fi
  else
    stage_note "merge_sim" "fail" "patch does not apply to the recorded base"
    finding "blocker" "merge_sim" "patch/PR does not apply to the current base"
    jset_json "repo.head" 'null'
    finish_report
    exit 1
  fi
else
  jset_string "repo.head" "$BASE_SHA"
  stage_note "merge_sim" "skip" \
    "checkout validated directly; no base-to-head patch was supplied, so merge and attribution were not tested"
fi

# ---------- no target: the PR ships no executable evidence ----------
#
# Placed AFTER the merge simulation and not before it, because a report review-pr cannot bind to
# the PR head is not evidence about that PR. The blocker is only worth publishing once `repo.base`
# and `repo.head` pin what it is a blocker about.
#
# A blocker, not a skip. A skip says "the validator could not establish this"; here the validator
# established something, and what it established is that a change with runtime surface arrived
# with nothing that runs it. review-pr reports a PR with no runtime surface as N/A and never gets
# here, so reaching this line means the caller judged there IS surface -- and then found nothing
# exercising it.
if [ -z "$TESTS" ]; then
  jset_string "test_selection.runner" "none"
  jset_string "test_selection.runner_reason" \
    "no target was declared, so there was nothing to choose a runner for"
  jset_string "test_selection.runner_basis" "declared-by-caller"
  jset_string "test_selection.test_provenance" "none"
  jset_string "test_selection.test_provenance_reason" "$NO_TARGET_REASON"
  stage_note "correctness_repo_tests" "skip" \
    "the caller declared that no test exercises this change: $NO_TARGET_REASON"
  finding "blocker" "correctness" \
    "this change has runtime surface and no test exercises it, so nothing about its behaviour was run: $NO_TARGET_REASON"
  finish_report
  exit 1
fi

# ---------- stage 2: GPU claim (sampling window + whole-run lock) ----------
# Resolution order: an explicit PICKER wins, then the picker this skill SHIPS, and only then
# whatever is on PATH. The shipped picker is part of the evidence contract -- it is the thing
# that prints `idleness-basis:`. Preferring a PATH copy silently substituted a picker that
# omits that line, and the report then published `idleness_basis: "unknown"` next to a
# concrete `gfx_activity_before_pct: 0`, presenting an unavailable reading as a measured idle
# one. Observed on two independent runs.
if [ -z "${PICKER:-}" ]; then
  for candidate in "$SCRIPT_DIR/pick-idle-gpu.py" \
                   "$(command -v pick-idle-gpu.py || true)" \
                   "$HOME/.local/bin/pick-idle-gpu.py" \
                   /usr/local/bin/pick-idle-gpu.py /opt/bin/pick-idle-gpu.py; do
    [ -n "$candidate" ] || continue
    if [ -x "$candidate" ] || {
      [ "$candidate" = "$SCRIPT_DIR/pick-idle-gpu.py" ] && [ -r "$candidate" ]
    }; then
      PICKER="$candidate"
      break
    fi
  done
fi

PICK=""
GPU_LOCK_FD=""
if [ -z "$PICKER" ] || { [ ! -x "$PICKER" ] && [ ! -r "$PICKER" ]; }; then
  stage_note "gpu_claim" "skip" "pick-idle-gpu.py is unavailable"
  jset_string "degraded_mode" "NO_GPU"
  finding "note" "gpu_claim" "GPU idleness could not be established; no runtime correctness claim is made"
else
  PICKER_CMD=("$PICKER")
  [ -x "$PICKER" ] || PICKER_CMD=(python3 "$PICKER")
  # Ask for the whole eligible ranking, not just the winner, so a contended lock on the first
  # choice falls through to the next. The picker is deterministic, so concurrent validators all
  # select the same device and all but one reported "no idle GPU" while the other seven sat
  # idle. Older pickers do not know --all; their single line is a ranking of one.
  PICK_CANDIDATES=$("${PICKER_CMD[@]}" --samples 10 --interval 1 --quiet --all \
    2>"$WORK/gpu-picker.log")
  PICK_RC=$?
  if [ "$PICK_RC" -ne 0 ] && grep -q -- "--all" "$WORK/gpu-picker.log" 2>/dev/null; then
    PICK_CANDIDATES=$("${PICKER_CMD[@]}" --samples 10 --interval 1 --quiet \
      2>"$WORK/gpu-picker.log")
    PICK_RC=$?
  fi
  PICK=$(printf '%s\n' "$PICK_CANDIDATES" | head -1)
  if [ "$PICK_RC" -ne 0 ] || [[ ! "$PICK" =~ ^[0-9]+$ ]]; then
    PICK=""
    # An environment fact and a validator portability gap are different things
    # and must not share one message.
    case "$PICK_RC" in
      1) CLAIM_NOTE="GPUs are present but none stayed below the idleness thresholds across the sampling window" ;;
      2) CLAIM_NOTE="AMD SMI could not be queried on this host, so idleness could not be established; see the picker log for whether AMD SMI is absent or failing" ;;
      3) CLAIM_NOTE="this host reports no GPUs" ;;
      *) CLAIM_NOTE="no verified-idle GPU was claimable (picker exit $PICK_RC)" ;;
    esac
    stage_note "gpu_claim" "skip" "$CLAIM_NOTE"
    jset_string "degraded_mode" "NO_GPU"
    finding "note" "gpu_claim" "$CLAIM_NOTE; no runtime correctness claim is made"
  else
    GPU_LOCK_TRIED=0
    GPU_LOCK_HELD=0
    while read -r _cand; do
      [[ "$_cand" =~ ^[0-9]+$ ]] || continue
      GPU_LOCK_TRIED=$((GPU_LOCK_TRIED + 1))
      exec {GPU_LOCK_FD}>"/tmp/gpu-$_cand.lock"
      if flock -n "$GPU_LOCK_FD"; then
        PICK="$_cand"
        GPU_LOCK_HELD=1
        break
      fi
      exec {GPU_LOCK_FD}>&-
      GPU_LOCK_FD=""
    done <<< "$PICK_CANDIDATES"
    if [ "$GPU_LOCK_HELD" -ne 1 ]; then
      PICK=""
      stage_note "gpu_claim" "skip" \
        "all $GPU_LOCK_TRIED verified-idle GPUs were already locked by another validator"
      jset_string "degraded_mode" "NO_GPU"
      finding "note" "gpu_claim" "GPU claim raced with another process; no runtime correctness claim is made"
    else
      GPU_INFO=$(python3 "$SCRIPT_DIR/gpu_probe.py" identify "$PICK")
      GPU_INFO_RC=$?
      if [ "$GPU_INFO_RC" -ne 0 ]; then
        flock -u "$GPU_LOCK_FD"
        PICK=""
        stage_note "gpu_claim" "skip" \
          "selected HIP index could not be mapped back to amd-smi metadata"
        jset_string "degraded_mode" "NO_GPU"
        finding "note" "gpu_claim" "GPU identity could not be verified; no runtime correctness claim is made"
      else
        jset_json "stages.gpu_claim" "$GPU_INFO"
        IDLENESS_BASIS=$(sed -n 's/^idleness-basis: //p' "$WORK/gpu-picker.log" | tail -1)
        jset_string "stages.gpu_claim.idleness_basis" "${IDLENESS_BASIS:-unknown}"
        if [ "$IDLENESS_BASIS" = "vram-only" ]; then
          finding "note" "gpu_claim" \
            "GPU activity is unavailable on this host; idleness was established from resident VRAM alone"
        fi
      fi
    fi
  fi
fi

# ---------- stage 3: repo-aware runtime compatibility ----------
RUNTIME_OK=0
RUNTIME_SOURCE_CHANGED=0
RC_OUT=""
RC=0
mkdir -p "$WORK/head/aiter-jit"
if [ -n "$PATCHF" ]; then
  RUNTIME_SOURCE_CHANGED=$(python3 - "$PATCHF" <<'PY'
import re
import sys

diff = open(sys.argv[1], encoding="utf-8").read()
paths = re.findall(
    r"^(?:--- a/|\+\+\+ b/|rename (?:from|to) )(.+)$",
    diff,
    re.MULTILINE,
)
runtime_prefixes = (
    "python/flydsl/",
    "python/mlir_flydsl/",
    "lib/",
    "include/",
    "cmake/",
    "thirdparty/",
    "tools/",
)
runtime_files = {"CMakeLists.txt", "MANIFEST.in", "setup.py", "pyproject.toml"}
print(int(any(path.startswith(runtime_prefixes) or path in runtime_files for path in paths)))
PY
)
fi
case "$REPO_KIND" in
  aiter)
    PROBE_PATH="$REPO_WT${PYLIB:+:$PYLIB}"
    RC_OUT=$(
      cd "$REPO_WT" \
        && AITER_TRITON_ONLY=1 AITER_JIT_DIR="$WORK/head/aiter-jit" \
          PYTHONDONTWRITEBYTECODE=1 \
          PYTHONPATH="$PROBE_PATH" timeout 300 \
          "$TARGET_PYTHON" - "$REPO_WT" 2>&1 <<'PY'
import importlib
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
module = importlib.import_module("aiter")
module_path = pathlib.Path(module.__file__).resolve()
if root not in module_path.parents:
    raise RuntimeError(f"aiter resolved outside checkout: {module_path}")
print(f"aiter {getattr(module, '__version__', '?')} from {module_path}")
PY
    )
    RC=$?
    ;;
  flydsl)
    if [ "$RUNTIME_SOURCE_CHANGED" -eq 1 ] && [ -n "$PYLIB" ]; then
      RC=2
      RC_OUT="patch changes FlyDSL runtime/build inputs; trusted build provenance is not implemented, so PYLIB cannot validate this patch"
    elif [ -n "$PYLIB" ]; then
      PROBE_PATH="$PYLIB:$REPO_WT/python"
      EXPECTED_FLYDSL_ROOT="$PYLIB"
    elif [ "$RUNTIME_SOURCE_CHANGED" -eq 1 ]; then
      PROBE_PATH="$REPO_WT/python"
      EXPECTED_FLYDSL_ROOT="$REPO_WT/python"
    else
      PROBE_PATH="$REPO_WT/python"
      EXPECTED_FLYDSL_ROOT="$REPO_WT/python"
    fi
    if [ "$RC_OUT" = "" ]; then
      RC_OUT=$(
        cd "$REPO_WT" \
          && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PROBE_PATH" timeout 300 \
            "$TARGET_PYTHON" - "$REPO_WT/python/flydsl/__init__.py" \
              "$EXPECTED_FLYDSL_ROOT" 2>&1 <<'PY'
import importlib
import pathlib
import re
import sys

source_init = pathlib.Path(sys.argv[1]).resolve()
expected_root = pathlib.Path(sys.argv[2]).resolve()
module = importlib.import_module("flydsl")
module_path = pathlib.Path(module.__file__).resolve()
if expected_root not in module_path.parents:
    raise RuntimeError(f"flydsl resolved outside expected runtime: {module_path}")
match = re.search(
    r"""__version__\s*=\s*["']([^"']+)["']""",
    source_init.read_text(),
)
source_version = match.group(1) if match else None
runtime_version = getattr(module, "__version__", None)
if source_version and runtime_version != source_version:
    raise RuntimeError(
        f"FlyDSL source/runtime version mismatch: {source_version} != {runtime_version}"
    )
print(f"flydsl {runtime_version or '?'} from {module_path}")
PY
      )
      RC=$?
    fi
    ;;
  *)
    RC=2
    RC_OUT="unsupported repository layout; expected aiter/ or python/flydsl/"
    ;;
esac

RC_DETAIL=$(python3 - "$RC_OUT" <<'PY'
import sys

print(" ".join(sys.argv[1].splitlines()[-3:])[:300])
PY
)
if [ "$RC" -eq 0 ]; then
  IDENTITY_FILE="$WORK/runtime-identity.json"
  if [ "$REPO_KIND" = "aiter" ]; then
    DEPENDENCY_ARGS=()
    [ -n "$PYLIB" ] && DEPENDENCY_ARGS=(--dependency-root "$PYLIB")
    (
      cd "$REPO_WT" \
        && AITER_TRITON_ONLY=1 AITER_JIT_DIR="$WORK/head/aiter-jit" \
          PYTHONDONTWRITEBYTECODE=1 \
          PYTHONPATH="$PROBE_PATH" timeout 300 \
          "$TARGET_PYTHON" "$SCRIPT_DIR/validate_evidence.py" runtime aiter "$REPO_WT" \
          "${DEPENDENCY_ARGS[@]}" --output "$IDENTITY_FILE"
    ) >"$WORK/runtime-identity.log" 2>&1
    IDENTITY_RC=$?
  else
    (
      cd "$REPO_WT" \
        && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PROBE_PATH" \
          timeout 300 "$TARGET_PYTHON" "$SCRIPT_DIR/validate_evidence.py" runtime flydsl \
          "$EXPECTED_FLYDSL_ROOT" --output "$IDENTITY_FILE"
    ) >"$WORK/runtime-identity.log" 2>&1
    IDENTITY_RC=$?
  fi
  if [ "$IDENTITY_RC" -eq 0 ] && [ -s "$IDENTITY_FILE" ]; then
    RUNTIME_IDENTITY=$(<"$IDENTITY_FILE")
    if python3 -c 'import json,sys; json.load(open(sys.argv[1]))' "$IDENTITY_FILE" \
        && jset_json "runtime_identity" "$RUNTIME_IDENTITY"; then
      stage_note "runtime_compat" "pass" "$RC_DETAIL"
      RUNTIME_OK=1
    else
      stage_note "runtime_compat" "skip" \
        "runtime identity output was not valid JSON"
      finding "note" "runtime_compat" \
        "runtime build identity could not be parsed; correctness is not trusted"
    fi
  else
    stage_note "runtime_compat" "skip" \
      "runtime imported but build identity collection failed"
    finding "note" "runtime_compat" \
      "runtime build identity could not be recorded; correctness is not trusted"
  fi
else
  stage_note "runtime_compat" "skip" "$RC_DETAIL"
  jset_string "stages.runtime_compat.reason" "runtime_mismatch"
  finding "note" "runtime_compat" \
    "checkout/runtime compatibility was not established; correctness stages are skipped rather than blamed on the PR"
fi

# ---------- stage 4: test policy (before execution) ----------
if [ -n "$PATCHF" ]; then
  if ! python3 - "$JSON" "$REPO_WT" "$TESTS" "$TOL_TABLE" <<'PY'
import json
import os
import re
import subprocess
import sys

report_path, worktree, tests, table = sys.argv[1:5]
relative_test = tests.split("::", 1)[0]
head_path = os.path.join(worktree, relative_test)
head = open(head_path).read() if os.path.exists(head_path) else ""
base_result = subprocess.run(
    ["git", "-C", worktree, "show", f"HEAD:{relative_test}"],
    capture_output=True,
    text=True,
)
base = base_result.stdout if base_result.returncode == 0 else ""
changed = subprocess.run(
    ["git", "-C", worktree, "diff", "--name-only", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
).stdout.splitlines()

def tolerances(source):
    # A tolerance declared as a named constant -- DEFAULT_REL_TOL = 2e-2, TOL = 1e-3 -- was
    # invisible to the old pattern, so a file that uses them reported ZERO tolerances and this
    # policy check passed on an empty list. Loosening a named constant is exactly the m2
    # mutant the stage exists to catch, and it was undetectable. Observed on a real PR
    # declaring DEFAULT_REL_TOL / DEFAULT_TILE_REL_TOL, which reported tolerances_head: [].
    assignments = [
        float(direct or named)
        for direct, named in re.findall(
            r"(?:atol|rtol)\s*=\s*([0-9.eE+-]+)"
            r"|^[ \t]*[A-Za-z_][A-Za-z0-9_]*(?:TOL|tol)[A-Za-z0-9_]*[ \t]*=[ \t]*([0-9.eE+-]+)",
            source,
            re.MULTILINE,
        )
        if (direct or named)
    ]
    # A tolerance passed by NAME (atol=DEFAULT_TOL) resolves to no literal here. Record that
    # the file has tolerances the checker cannot value, so an empty list is never read as
    # "this suite declares no tolerances".
    indirect = re.findall(r"(?:atol|rtol)\s*=\s*([A-Za-z_][A-Za-z0-9_.]*)", source)
    mappings = [
        float(value)
        for value in re.findall(
            r"""["'](?:f32|f16|bf16)["']\s*:\s*([0-9.eE+-]+)""",
            source,
        )
    ]
    return assignments + mappings, sorted(set(indirect))

head_tolerances, head_indirect = tolerances(head)
base_tolerances, base_indirect = tolerances(base)
loosened = []
if (
    base_tolerances
    and head_tolerances
    and len(base_tolerances) == len(head_tolerances)
):
    loosened = [
        [before, after]
        for before, after in zip(base_tolerances, head_tolerances)
        if after > before
    ]

commented_pattern = (
    r"""^\s*#\s*\(\s*\d+\s*,\s*\d+\s*,\s*["']"""
    r"""(?:f32|f16|bf16)["']\s*\)"""
)
commented_base = len(re.findall(commented_pattern, base, re.MULTILINE))
commented_head = len(re.findall(commented_pattern, head, re.MULTILINE))
commented_added = max(0, commented_head - commented_base)
reference = {}
for item in filter(None, table.split(",")):
    name, value = item.split("=", 1)
    reference[name] = float(value)

# --tol-table was parsed, published, and compared against nothing: a documented flag with a
# validated argument and no effect on any verdict. The comparison the caller is entitled to is
# whether the suite accepts error beyond anything they declared acceptable. Which literal
# belongs to which dtype is not recoverable from the source, so the bound is the loosest
# declared reference -- a tolerance above that is loose under every reading.
exceeds_reference = []
if reference and head_tolerances:
    ceiling = max(reference.values())
    exceeds_reference = sorted(
        value for value in set(head_tolerances) if value > ceiling
    )

kernel_suffixes = (".py", ".cu", ".cuh", ".h", ".hpp", ".cpp")
kernel_changed = any(
    path.endswith(kernel_suffixes)
    and not path.startswith(("tests/", "op_tests/"))
    for path in changed
)
data = json.load(open(report_path))
stage = {
    "status": "fail" if loosened else "pass",
    "tolerances_base": base_tolerances,
    "tolerances_head": head_tolerances,
    "tolerances_head_by_name": head_indirect,
    "tolerances_base_by_name": base_indirect,
    "reference_tolerances": reference,
    "exceeds_reference": exceeds_reference,
    "commented_out_shape_rows_base": commented_base,
    "commented_out_shape_rows": commented_head,
    "commented_out_shape_rows_added": commented_added,
    "kernel_files_changed": kernel_changed,
}
if exceeds_reference:
    data["findings"].append(
        {
            "severity": "should-fix",
            "stage": "test_policy",
            "detail": (
                f"the suite accepts error up to {max(exceeds_reference)}, above the loosest "
                f"reference tolerance supplied ({max(reference.values())}); a kernel defect "
                f"smaller than that gap cannot make these tests red"
            ),
        }
    )
if loosened:
    stage["loosened"] = loosened
    if kernel_changed:
        data["findings"].append(
            {
                "severity": "should-fix",
                "stage": "test_policy",
                "detail": (
                    f"comparison tolerance widened {loosened} while kernel code also "
                    "changed; require a numerical justification instead of treating "
                    "the green suite as clearance"
                ),
            }
        )
    else:
        data["findings"].append(
            {
                "severity": "blocker",
                "stage": "test_policy",
                "detail": (
                    f"test-only change widens comparison tolerance {loosened} "
                    "(base -> head), so the suite can no longer enforce its prior bound"
                ),
            }
        )
if commented_added:
    data["findings"].append(
        {
            "severity": "should-fix",
            "stage": "test_policy",
            "detail": (
                f"this change comments out {commented_added} additional shape rows; "
                "independent boundary-grid coverage must remain explicit"
            ),
        }
    )
data["stages"]["test_policy"] = stage
json.dump(data, open(report_path, "w"), indent=2)
PY
  then
    stage_note "test_policy" "skip" "test-policy analyzer failed"
    finding "note" "test_policy" "test-policy analysis failed; validation is inconclusive"
  fi
else
  stage_note "test_policy" "skip" \
    "no patch supplied; base-to-head tolerance and test-shape policy cannot be compared"
fi

# ---------- stage 5: correctness with an exact baseline control ----------
if [ "$REPO_KIND" = "flydsl" ] && [ -n "$PYLIB" ]; then
  TEST_PYTHONPATH="$PYLIB:$REPO_WT:$REPO_WT/python"
else
  TEST_PYTHONPATH="$REPO_WT/python:$REPO_WT${PYLIB:+:$PYLIB}"
fi
TEST_PYTHONPATH="$PROBE_DIR:$SCRIPT_DIR:$TEST_PYTHONPATH"
TEST_FILE=${TESTS%%::*}
TARGET_PATH="$REPO_WT/$TEST_FILE"
# A pytest node id selects cases within a file; a timing run executes the file. So where the
# perf target falls back to the correctness target, it falls back to its FILE, never its node
# id. Discovery below may replace this; a caller who passed --perf-target has settled it.
if [ -z "$PERF_TARGET" ]; then
  PERF_TARGET="$TEST_FILE"
fi
# The runner is DECLARED by the caller, not derived here. Reading the target and deciding
# whether pytest can collect it is judgement, and judgement belongs in the prompt -- but a
# declaration is not a measurement, so the report records which of the two it got. A reader who
# cannot tell "the validator determined this" from "the caller asserted this" cannot weigh a
# runner-caused failure correctly, and that failure gets charged to the PR author.
TARGET_RUNNER="none"
TARGET_RUNNER_REASON=""
TARGET_RUNNER_BASIS=""
if [ ! -f "$TARGET_PATH" ]; then
  TARGET_RUNNER="none"
  TARGET_RUNNER_REASON="target file does not exist on head"
  TARGET_RUNNER_BASIS="target-missing"
elif [ "$TESTS" != "${TESTS%%::*}" ]; then
  # A `path::node` selector is not a judgement about the file; nothing can run that string as a
  # script. This one fact stays here because it is syntax, not analysis.
  TARGET_RUNNER="pytest"
  TARGET_RUNNER_REASON="explicit pytest node selector"
  TARGET_RUNNER_BASIS="explicit-node-selector"
elif [ -n "$RUNNER_OVERRIDE" ]; then
  case "$RUNNER_OVERRIDE" in
    pytest|script|none) TARGET_RUNNER="$RUNNER_OVERRIDE" ;;
    *) echo "--runner must be pytest, script, or none" >&2; exit 2;;
  esac
  TARGET_RUNNER_REASON="${RUNNER_REASON:-declared by the caller, which stated no reason}"
  TARGET_RUNNER_BASIS="declared-by-caller"
else
  # Refusing to guess is the whole point. A default of "pytest" here is what published a
  # collection error as the PR's test failing on head.
  TARGET_RUNNER="none"
  TARGET_RUNNER_REASON="no --runner was declared for this target, so it was not run"
  TARGET_RUNNER_BASIS="undeclared"
fi
jset_string "test_selection.runner" "$TARGET_RUNNER"
jset_string "test_selection.runner_reason" "$TARGET_RUNNER_REASON"
jset_string "test_selection.runner_basis" "$TARGET_RUNNER_BASIS"

# Which of the two kinds of evidence this run is about to gather. The status was captured at
# stage 1, when it still described the patch alone.
# `-n "$PATCHF"` and not PATCH_APPLIED: the latter tracks a liability -- whether this process
# still owes the caller a revert -- and it is toggled by every phase switch. The question here is
# the immutable one, whether a patch was supplied at all.
if [ -n "$PATCHF" ]; then
  PROVENANCE_OUT=$(printf '%s' "$PATCH_STATUS" \
    | "$SCRIPT_DIR/target_run.py" provenance "$TEST_FILE" --patch-supplied)
else
  PROVENANCE_OUT=$("$SCRIPT_DIR/target_run.py" provenance "$TEST_FILE" </dev/null)
fi
TEST_PROVENANCE=${PROVENANCE_OUT%%$'\n'*}
TEST_PROVENANCE_REASON=${PROVENANCE_OUT#*$'\n'}
jset_string "test_selection.test_provenance" "$TEST_PROVENANCE"
jset_string "test_selection.test_provenance_reason" "$TEST_PROVENANCE_REASON"
# Not a defect, and deliberately not a should-fix: a PR is entitled to bring its own test, and
# whether that test is a good one is a reading, which belongs to review-pr. What the report owes
# the reader is the fact that the evidence and the code under review came from the same hand.
if [ "$TEST_PROVENANCE" = "pr-added" ] || [ "$TEST_PROVENANCE" = "pr-modified" ]; then
  finding "note" "correctness" \
    "the evidence is not independent of the change: $TEST_PROVENANCE_REASON"
fi

# ---------- which file gets timed ----------
# Two places a perf target comes from, and this commit implements the first: a bench the PR
# itself ships. Run here, and not later beside the timing runs, because here the patch is
# still applied -- the base phase reverses it out further down, and a bench the PR adds is
# not in the tree once that happens.
#
# `--no-perf` skips the search outright: reading every file the patch touched to answer a
# question nobody asked is work, and the answer would go into a stage that reports `skip`.
PERF_TARGET_BASIS_REASON=""
PERF_CANDIDATES="[]"
# What the patch changed, as discovery saw it. Carried to the stage so "nothing timed this"
# can be told apart from "there was nothing here to time": the first is a gap the author can
# close, the second is not a finding at all.
PERF_KERNEL_MODULES="[]"
PERF_NATIVE_PATHS="[]"
PERF_TARGETS=("$PERF_TARGET")
PERF_TARGET_BASES=("$PERF_TARGET_BASIS")
PERF_TARGET_BASIS_REASONS=("")
if [ "$PERF_TARGET_BASIS" != "declared-by-caller" ] && [ "$PERF_ENABLED" -eq 1 ]; then
  PERF_DISCOVERY=$(printf '%s' "$PATCH_STATUS" \
    | "$SCRIPT_DIR/scrape_perf.py" discover \
      --root "$REPO_WT" --correctness-target "$TEST_FILE" --patch "$PATCHF")
  if [ -n "$PERF_DISCOVERY" ]; then
    PERF_TARGETS=()
    PERF_TARGET_BASES=()
    PERF_TARGET_BASIS_REASONS=()
    # Both paths are timed when both resolve. The count comes from the discovery blob rather
    # than from reading indices until one is empty, which cannot tell "past the end" from
    # "this field is blank".
    PERF_TARGET_COUNT=$(stats_field "$PERF_DISCOVERY" target_count)
    for ((_i = 0; _i < PERF_TARGET_COUNT; _i++)); do
      PERF_TARGETS+=("$(stats_field "$PERF_DISCOVERY" "targets.$_i.target")")
      PERF_TARGET_BASES+=("$(stats_field "$PERF_DISCOVERY" "targets.$_i.basis")")
      PERF_TARGET_BASIS_REASONS+=("$(stats_field "$PERF_DISCOVERY" "targets.$_i.reason")")
    done
    PERF_TARGET="${PERF_TARGETS[0]}"
    PERF_TARGET_BASIS="${PERF_TARGET_BASES[0]}"
    PERF_TARGET_BASIS_REASON="${PERF_TARGET_BASIS_REASONS[0]}"
    PERF_CANDIDATES=$(python3 "$TARGET_TOOL" stats-field --json "$PERF_DISCOVERY" candidates)
    PERF_KERNEL_MODULES=$(python3 "$TARGET_TOOL" \
      stats-field --json "$PERF_DISCOVERY" kernel_modules)
    PERF_NATIVE_PATHS=$(python3 "$TARGET_TOOL" \
      stats-field --json "$PERF_DISCOVERY" native_paths)
  fi
fi

# The same question test_provenance asks, asked of every file the timing runs will execute. It
# is the same pure function against the same snapshot -- a perf target is a target, and "did
# the patch write this?" has one answer however the file is used. Reusing it also inherits the
# honesty that no patch means `unknown` rather than a cheerful `pre-existing`.
PERF_TARGET_PROVENANCES=()
PERF_TARGET_PROVENANCE_REASONS=()
for _target in "${PERF_TARGETS[@]}"; do
  if [ "$_target" = "$TEST_FILE" ]; then
    PERF_TARGET_PROVENANCES+=("$TEST_PROVENANCE")
    PERF_TARGET_PROVENANCE_REASONS+=("$TEST_PROVENANCE_REASON")
    continue
  fi
  if [ -n "$PATCHF" ]; then
    PERF_PROVENANCE_OUT=$(printf '%s' "$PATCH_STATUS" \
      | "$SCRIPT_DIR/target_run.py" provenance "$_target" --patch-supplied)
  else
    PERF_PROVENANCE_OUT=$("$SCRIPT_DIR/target_run.py" provenance "$_target" </dev/null)
  fi
  PERF_TARGET_PROVENANCES+=("${PERF_PROVENANCE_OUT%%$'\n'*}")
  PERF_TARGET_PROVENANCE_REASONS+=("${PERF_PROVENANCE_OUT#*$'\n'}")
done
PERF_TARGET_PROVENANCE="${PERF_TARGET_PROVENANCES[0]}"
PERF_TARGET_PROVENANCE_REASON="${PERF_TARGET_PROVENANCE_REASONS[0]}"

# Two independent channels can carry the S1 grid: the target's own CLI flag (--shape-arg)
# and an environment variable it reads (--shape-env). They are probed separately and the
# results are combined, because a caller who supplies both is describing one target that has
# both -- and an earlier version let the env probe's result overwrite the CLI probe's
# unconditionally, so supplying both DISCARDED a working CLI channel and then reported the
# env channel's absence as the reason no grid ran.
# Which channel carries the grid is settled by which flag the CALLER named, not by reading the
# target. Reading it was static analysis of a claim the caller had already made, and the claim is
# proved at runtime a few hundred lines below: the target is run once with a deliberately invalid
# grid and must FAIL. A channel that does not exist cannot fail that way, so the proof stands on
# its own and the AST pre-check only ever decided how good the error message was.
GRID_HOOK_OK=0
GRID_CHANNEL=""
GRID_CHANNEL_BASIS=""
if [ -n "$GRID" ]; then
  # CLI first when several are named: the caller who spelled out a flag was most specific, and
  # the wiring for the other two is narrower.
  if [ -n "$SHAPE_ARG" ] && [ "$TARGET_RUNNER" = "script" ]; then
    GRID_HOOK_OK=1
    GRID_CHANNEL="cli"
  elif [ -n "$SHAPE_ENV" ]; then
    GRID_HOOK_OK=1
    GRID_CHANNEL="env"
  elif [ -n "$SHAPE_ARGNAMES" ] && [ "$TARGET_RUNNER" = "pytest" ]; then
    GRID_HOOK_OK=1
    GRID_CHANNEL="pytest"
  fi
  [ "$GRID_HOOK_OK" -eq 1 ] && GRID_CHANNEL_BASIS="declared-by-caller"
fi
jset_string "test_selection.grid_channel" "$GRID_CHANNEL"
jset_string "test_selection.grid_channel_basis" "$GRID_CHANNEL_BASIS"
# When no channel could even be named, say which of the several ways that happened applied. A
# caller must be able to tell "I named nothing" from "I named something this runner cannot
# deliver" -- and a limit of the validator must never be published as a property of the target.
GRID_CHANNEL_REASON=""
if [ -n "$GRID" ] && [ "$GRID_HOOK_OK" -ne 1 ]; then
  if [ -z "$SHAPE_ARG" ] && [ -z "$SHAPE_ENV" ] && [ -z "$SHAPE_ARGNAMES" ]; then
    GRID_CHANNEL_REASON="a grid was supplied but neither --shape-arg nor --shape-env named a channel to deliver it through"
  else
    GRID_CHANNEL_REASON="grid channel not established:"
    if [ -n "$SHAPE_ARG" ] && [ "$TARGET_RUNNER" != "script" ]; then
      GRID_CHANNEL_REASON="$GRID_CHANNEL_REASON --shape-arg '"'"'$SHAPE_ARG'"'"' was ignored because the target runs under $TARGET_RUNNER and the CLI channel is wired only for script targets (a validator limit, not a target property);"
    fi
    if [ -n "$SHAPE_ARGNAMES" ] && [ "$TARGET_RUNNER" != "pytest" ]; then
      GRID_CHANNEL_REASON="$GRID_CHANNEL_REASON --shape-argnames needs a pytest target and this one runs under $TARGET_RUNNER;"
    fi
  fi
fi
jset_string "test_selection.grid_channel_reason" "$GRID_CHANNEL_REASON"
# The grid used to be REQUIRED -- correctness_s1_grid was a required stage, so a run without one
# topped out at INCONCLUSIVE. That is what made callers supply a grid whether or not they had
# anything to say with it, and on ROCm/aiter#4538 the three requested shapes were all already in
# the target's own default list: the "independent" grid re-ran a strict subset of the repository
# run and the report credited it as new coverage.
#
# The answer at the time was to make the caller DECLARE what their cells covered and to refuse a
# pass without the declaration -- more bookkeeping around a grid nobody wanted to supply. The
# answer now is that the grid earns nothing. It is optional, it is not a required stage, and a
# passing grid cannot move the verdict; only a FAILING one can, and a failure is a real defect
# whether or not the cells were novel. A duplicate grid is then harmless rather than policed,
# which is why the whole grid_independence vocabulary is gone: the error it guarded against is
# no longer reachable.
#
# ---- Extra axes.
#
# A grid is one ordered tuple on one flag. A target whose remaining knobs are separate flags
# -- head counts, dtypes, window modes -- cannot be gridded over them at all, so on aiter#4538
# the kernel's assert at num_heads=16 was unreachable however the shape grid was spelled. An
# axis is a name, a flag, and its values, and the caller supplies all three.
#
# Whether the target declares that flag is a reading of its source and is left to the caller.
# What is NOT left to the caller is the proof: further down, every axis flag must be observed
# REFUSING a deliberately invalid value before its values are allowed onto the grid run's
# argv. A flag the target declares but ignores, or silently clamps, would otherwise let the
# report claim coverage of head counts that never reached the kernel.
AXIS_STATE="none"
AXIS_STATE_REASON="no extra axes were requested"
if [ "${#AXES[@]}" -gt 0 ]; then
  # Record what was ASKED FOR before deciding whether it can be honoured. An empty `axes`
  # beside a non-`none` axis_state loses the request itself: a reader could not see that a
  # head-count axis had been requested and dropped -- which is precisely the silently
  # narrowed test space this stage exists to make visible.
  AXIS_REPORT=$(python3 - "${AXES[@]}" <<'PY'
import json
import sys

axes = []
for spec in sys.argv[1:]:
    name, _, rest = spec.partition("=")
    flag, _, values = rest.partition(":")
    cells = [cell.strip() for cell in values.split(";") if cell.strip()]
    entry = {
        "name": name.strip(),
        "flag": flag.strip(),
        "values": cells,
        # Filled in by the runtime refusal probe, which is the only thing that decides an
        # axis is consumed. Until it runs, nothing here may be read as proof.
        "hook_proof": "not-evaluated",
    }
    if not entry["name"] or not entry["flag"] or not cells:
        entry["hook_proof"] = "malformed-axis-spec"
    axes.append(entry)
print(json.dumps(axes))
PY
)
  AXIS_MALFORMED=$(python3 -c '
import json
import sys

print(",".join(a["name"] or "(unnamed)" for a in json.loads(sys.argv[1])
                if a["hook_proof"] == "malformed-axis-spec"))
' "$AXIS_REPORT")
  if [ -n "$AXIS_MALFORMED" ]; then
    AXIS_STATE="malformed-spec"
    AXIS_STATE_REASON="--axis wants name=--flag:v1;v2, and these do not parse: $AXIS_MALFORMED"
  elif [ "$TARGET_RUNNER" != "script" ]; then
    AXIS_STATE="unusable"
    AXIS_STATE_REASON="extra axes ride argv, which reaches script targets only (this target runs under $TARGET_RUNNER)"
  else
    AXIS_STATE="declared"
    AXIS_STATE_REASON="${#AXES[@]} axis/axes requested"
    while IFS= read -r token; do
      [ -n "$token" ] && AXIS_CLI+=("$token")
    done < <(python3 -c '
import json
import sys

for axis in json.loads(sys.argv[1]):
    print(axis["flag"])
    for value in axis["values"]:
        print(value)
' "$AXIS_REPORT")
  fi
fi
jset_json "test_selection.axes" "$AXIS_REPORT"
jset_string "test_selection.axis_state" "$AXIS_STATE"
jset_string "test_selection.axis_state_reason" "$AXIS_STATE_REASON"

# Runs the selected target once, whatever its runner is -- the name predates script targets.
# The second argument is the grid VALUE, not an env assignment: the channel is decided by
# the channel that probed positive. It used to take "$SHAPE_ENV=$GRID" and re-split on the
# first `=`, which
# worked for a CLI-only run only because an unset SHAPE_ENV left a leading `=` that the split
# then removed -- the shapes were travelling inside a string shaped like the channel they were
# not using.
run_pytest() {
  local label="$1"
  local grid_value="$2"
  local log="$WORK/$TARGET_RUNNER-$label.log"
  local phase=${label%%-*}
  local cache_root="$WORK/$phase"
  local junit="$cache_root/junit-$label.xml"
  # Per LABEL, not per phase. head-repo and head-grid share a phase directory, so a grid run
  # that died during collection overwrote the receipt head-repo had already written and the
  # report claimed the route never executed -- erasing evidence that had been collected.
  local receipt="$cache_root/execution-receipt-$label.json"
  mkdir -p "$cache_root/home" "$cache_root/xdg-cache" \
    "$cache_root/flydsl-cache" "$cache_root/triton-cache" \
    "$cache_root/torch-extensions" "$cache_root/pytest-cache" \
    "$cache_root/aiter-jit"
  rm -f "$junit" "$receipt"
  if [ "$TARGET_RUNNER" = "pytest" ] || [ -n "$EXPECTED_ROUTE" ]; then
    python3 "$TARGET_TOOL" probe-module "$SCRIPT_DIR/validation_probe.py" \
      "$PROBE_DIR/$PROBE_MODULE.py" "$EXPECTED_ROUTE" "$SHAPE_VARS" "$receipt"
  fi
  local -a environment=(
    "HIP_VISIBLE_DEVICES=$PICK"
    "PYTHONPATH=$TEST_PYTHONPATH"
    "PYTHONDONTWRITEBYTECODE=1"
    "HOME=$cache_root/home"
    "XDG_CACHE_HOME=$cache_root/xdg-cache"
    "FLYDSL_CACHE_DIR=$cache_root/flydsl-cache"
    "FLYDSL_RUNTIME_CACHE_DIR=$cache_root/flydsl-cache"
    "TRITON_CACHE_DIR=$cache_root/triton-cache"
    "TORCH_EXTENSIONS_DIR=$cache_root/torch-extensions"
    "AITER_JIT_DIR=$cache_root/aiter-jit"
    "VALIDATION_PHASE=$label"
  )
  # The pytest channel needs its plugin generated per run, with the grid baked in, so the
  # tested PR can neither read nor forge it.
  local -a shape_plugin=()
  if [ "$GRID_CHANNEL" = "pytest" ] && [ -n "$grid_value" ]; then
    local _grid_value="$grid_value"
    # Remove first: a generator that fails must not leave the PREVIOUS phase's plugin in
    # place, or the next phase silently re-runs the grid it was carrying.
    rm -f "$PROBE_DIR/${PROBE_MODULE}_shapes.py"
    python3 "$TARGET_TOOL" shape-plugin "$SCRIPT_DIR/shape_grid_plugin.py" \
      "$PROBE_DIR/${PROBE_MODULE}_shapes.py" "$SHAPE_ARGNAMES" "$_grid_value"
    if [ ! -s "$PROBE_DIR/${PROBE_MODULE}_shapes.py" ]; then
      echo "shape plugin generation failed for $label" >&2
      printf '%s|%s\n' 2 "$log"
      return 0
    fi
    shape_plugin=(-p "${PROBE_MODULE}_shapes")
    # The plugin now carries the grid; nothing may also send it on argv or in the env.
    grid_value=""
  fi
  local -a shape_cli=()
  # Dispatch on the channel that actually probed positive, not on "--shape-arg was supplied".
  # With both flags given and only the env channel real, the old condition still routed the
  # grid through the CLI flag the target does not parse.
  if [ -n "$grid_value" ] && [ "$GRID_CHANNEL" = "cli" ]; then
    shape_cli=("$SHAPE_ARG")
    local _old_ifs="$IFS"
    IFS=';'
    for _shape in $grid_value; do
      [ -n "$_shape" ] && shape_cli+=("$_shape")
    done
    IFS="$_old_ifs"
  elif [ -n "$grid_value" ] && [ "$GRID_CHANNEL" = "env" ]; then
    environment+=("$SHAPE_ENV=$grid_value")
  fi
  # Extra axes ride on the same argv as the shape grid, so the run that carries the grid is
  # the run that carries the axes and one receipt describes both. AXIS_CLI_OVERRIDE exists
  # only for the per-axis refusal probes, which must send one deliberately invalid value and
  # nothing else.
  if [ "${#AXIS_CLI_OVERRIDE[@]}" -gt 0 ]; then
    shape_cli+=("${AXIS_CLI_OVERRIDE[@]}")
  elif [ -n "$grid_value" ] && [ "${#AXIS_CLI[@]}" -gt 0 ] \
      && [ "$GRID_CHANNEL" = "cli" ]; then
    shape_cli+=("${AXIS_CLI[@]}")
  fi
  if [ "$TARGET_RUNNER" = "pytest" ]; then
    (
      cd "$REPO_WT" \
        && env -i "${TARGET_BASE_ENV[@]}" "${environment[@]}" timeout "$TIMEOUT" \
          "$TARGET_PYTHON" -m pytest -p "$PROBE_MODULE" "${shape_plugin[@]}" \
            "$TESTS" -x -q \
            --junitxml="$junit" -o "cache_dir=$cache_root/pytest-cache"
    ) >"$log" 2>&1
  elif [ -n "$EXPECTED_ROUTE" ]; then
    (
      cd "$REPO_WT" \
        && env -i "${TARGET_BASE_ENV[@]}" "${environment[@]}" timeout "$TIMEOUT" \
          "$TARGET_PYTHON" "$SCRIPT_DIR/run_script_with_probe.py" \
            "$PROBE_MODULE" "$TEST_FILE" "${shape_cli[@]}"
    ) >"$log" 2>&1
  else
    (
      cd "$REPO_WT" \
        && env -i "${TARGET_BASE_ENV[@]}" "${environment[@]}" timeout "$TIMEOUT" \
          "$TARGET_PYTHON" "$TEST_FILE" "${shape_cli[@]}"
    ) >"$log" 2>&1
  fi
  local result=$?
  echo "$result|$log"
}

# Decide whether the target can be timed at all, and with what arguments. Which harness a
# target has is a reading of its source, and it lives in scrape_perf.py with the reasons for
# it; what stays here is only whether there is a file to read.
perf_detect() {
  local file="$REPO_WT/$PERF_TARGET"
  if [ ! -f "$file" ]; then
    PERF_BASIS="the perf target $PERF_TARGET is not present in this checkout"
    return 1
  fi
  local harness
  if ! harness=$("$SCRIPT_DIR/scrape_perf.py" detect "$file"); then
    PERF_BASIS="the target exposes no benchmark entry point (no --scenario bench, no perftest/@benchmark harness)"
    return 1
  fi
  PERF_ARGS=${harness%%$'\n'*}
  PERF_BASIS=${harness#*$'\n'}
  return 0
}

# The timing counterpart to run_pytest. Three things differ, and each difference is the point:
#   * no probe module is injected. The receipt probe wraps every route call to record shapes;
#     a traced kernel is not the kernel whose latency we are about to report.
#   * the phase cache root is shared with that phase's correctness run, so the JIT cache is
#     already warm and the table measures the kernel rather than a compile.
#   * PERF_TIMEOUT is separate from TIMEOUT, because silently killing a legitimately long
#     sweep would produce an empty log -- indistinguishable from "this target has no harness".
#
# A timing run leaves artifacts behind -- aiter targets drop a tuned_op_bench.csv in the repo
# root -- and the baseline phase asserts a clean worktree, so the run has to leave the tree as
# it found it. These two record the tree before and put it back after; which paths may be
# touched, and which must only be reported, is decided in scrape_perf.py.
perf_snapshot() {
  git -C "$REPO_WT" status --porcelain --untracked-files=all \
    >"$WORK/perf-worktree-$1.txt" 2>/dev/null || : >"$WORK/perf-worktree-$1.txt"
}

perf_restore() {
  local before="$WORK/perf-worktree-$1.txt"
  [ -r "$before" ] || return 0
  "$SCRIPT_DIR/scrape_perf.py" restore-worktree "$REPO_WT" "$before"
}

# Run one side PERF_REPEAT times. Results go to globals rather than a packed string because
# the log list is variable-length and re-splitting it on the caller side is how paths with
# awkward characters get mangled.
PERF_RUN_LOGS=()
PERF_RUN_RC=0
run_perf_repeats() {
  local phase="$1"
  # Which target this side is timing. Part of the log name because two targets are timed in
  # the same phase and a shared name would have the second overwrite the first -- silently,
  # since both sides would still find a readable log at the path they recorded.
  local slot="$2"
  local index result rc
  PERF_RUN_LOGS=()
  PERF_RUN_RC=0
  for ((index = 1; index <= PERF_REPEAT; index++)); do
    result=$(run_perf "$phase-perf-t$slot-$index")
    rc=${result%%|*}
    PERF_RUN_LOGS+=("${result##*|}")
    # Any failed repeat poisons the side: the reduction takes a minimum, so one truncated
    # run could contribute an impossibly fast sample and manufacture a regression on the
    # other side. Report the failure instead.
    if [ "$rc" -ne 0 ]; then
      PERF_RUN_RC=$rc
    fi
  done
}

run_perf() {
  local label="$1"
  local log="$WORK/perf-$label.log"
  local phase=${label%%-*}
  local cache_root="$WORK/$phase"
  mkdir -p "$cache_root/home" "$cache_root/xdg-cache" \
    "$cache_root/flydsl-cache" "$cache_root/triton-cache" \
    "$cache_root/torch-extensions" "$cache_root/aiter-jit"
  local -a environment=(
    "HIP_VISIBLE_DEVICES=$PICK"
    "PYTHONPATH=$TEST_PYTHONPATH"
    "PYTHONDONTWRITEBYTECODE=1"
    "HOME=$cache_root/home"
    "XDG_CACHE_HOME=$cache_root/xdg-cache"
    "FLYDSL_CACHE_DIR=$cache_root/flydsl-cache"
    "FLYDSL_RUNTIME_CACHE_DIR=$cache_root/flydsl-cache"
    "TRITON_CACHE_DIR=$cache_root/triton-cache"
    "TORCH_EXTENSIONS_DIR=$cache_root/torch-extensions"
    "AITER_JIT_DIR=$cache_root/aiter-jit"
    "VALIDATION_PHASE=$label"
  )
  local -a extra=()
  local _old_ifs="$IFS"
  IFS=' '
  local _word
  for _word in $PERF_ARGS; do
    [ -n "$_word" ] && extra+=("$_word")
  done
  IFS="$_old_ifs"
  (
    cd "$REPO_WT" \
      && env -i "${TARGET_BASE_ENV[@]}" "${environment[@]}" timeout "$PERF_TIMEOUT" \
        "$TARGET_PYTHON" "$PERF_TARGET" "${extra[@]}"
  ) >"$log" 2>&1
  local result=$?
  echo "$result|$log"
}

target_stats() {
  local label="$1"
  local result="$2"
  local phase=${label%%-*}
  local junit="$WORK/$phase/junit-$label.xml"
  if [ "$TARGET_RUNNER" = "script" ]; then
    # A script target publishes no per-case count, so "executed" used to be hard-coded to 1
    # and stood for "the process ran". That is the number a silently-returning target also
    # produces -- aiter#4538's own target returns with exit 0 and log output when the arch is
    # unsupported or an optional package is missing -- so a run that graded 56 cases and a run
    # that graded none were indistinguishable, and both credited runtime architecture
    # coverage. When a route was named, the run's own receipt carries observable work, and
    # that count is used instead. With no route named there is still nothing to observe, and
    # the basis says so rather than implying a case count.
    python3 "$TARGET_TOOL" script-stats "$result" \
      "$WORK/$phase/execution-receipt-$label.json" "$EXPECTED_ROUTE"
  elif [ -f "$junit" ]; then
    python3 "$SCRIPT_DIR/validate_evidence.py" pytest-stats "$junit"
  else
    printf '%s\n' \
      '{"tests":0,"failures":0,"errors":1,"skipped":0,"executed":0,"note":"JUnit XML missing"}'
  fi
}

# The grid receipt is preferred when it proves the route, because it is the run that
# exercised the injected shapes; otherwise the repository run's receipt stands. A phase that
# observed nothing never speaks over one that observed something.
head_receipt() {
  python3 "$TARGET_TOOL" pick-receipt \
    "$WORK/head/execution-receipt-head-grid.json" \
    "$WORK/head/execution-receipt-head-repo.json"
}

# ---------- credential-free execution isolation ----------
#
# The allowlist, the secret-shaped denylist, and the reasons for both live in target_run.py.
# What stays here is only the handoff: the pairs are read once, and every launch below runs
# under `env -i` with exactly this set and nothing inherited.
TARGET_BASE_ENV=()
mapfile -d '' -t TARGET_BASE_ENV < <(python3 "$TARGET_TOOL" env)
jset_json "isolation.target_environment" \
  "$(python3 "$TARGET_TOOL" env-summary "${TARGET_BASE_ENV[@]}")"


# ---------- does this target actually need a GPU? ----------
# Asked of the target, not inferred from the diff. A diff heuristic cannot settle this:
# a Python-level dispatch change reroutes kernels without touching kernel source, and
# ROCm/aiter#5089 decides whether 34 gfx950 kernels compile from a 7-line helper. Here
# PICK is empty, so run_pytest already exports HIP_VISIBLE_DEVICES="" and the target
# runs with no visible device -- passing there is an observation, not a guess.
GPU_REQUIREMENT="required"
GPU_REQUIREMENT_BASIS="a GPU was claimed, so whether the target can run without one was never probed; 'required' here is the conservative default, not an observation"
if [ -z "$PICK" ]; then
  if [ "$RUNTIME_OK" -eq 1 ] && [ "$TARGET_RUNNER" != "none" ]; then
    GPUFREE_RESULT=$(run_pytest "gpufree-probe" "")
    GPUFREE_RC=${GPUFREE_RESULT%%|*}
    GPUFREE_LOG=${GPUFREE_RESULT##*|}
    GPUFREE_STATS=$(target_stats "gpufree-probe" "$GPUFREE_RC")
    GPUFREE_EXECUTED=$(stats_field "$GPUFREE_STATS" executed)
    if [ "$GPUFREE_RC" -eq 0 ] && [ "$GPUFREE_EXECUTED" -ge 1 ]; then
      # executed>=1 carries this test: a suite guarded by
      # skipif(not torch.cuda.is_available()) also exits 0, having proved nothing.
      GPU_REQUIREMENT="not-required"
      GPU_REQUIREMENT_BASIS="target passed with no visible GPU, executing $GPUFREE_EXECUTED test(s)"
    else
      GPU_REQUIREMENT_BASIS="target did not pass with no visible GPU (exit $GPUFREE_RC, executed $GPUFREE_EXECUTED)"
    fi
  else
    GPU_REQUIREMENT_BASIS="the target could not be probed without a GPU"
  fi
fi
jset_string "test_selection.gpu_requirement" "$GPU_REQUIREMENT"
jset_string "test_selection.gpu_requirement_basis" "$GPU_REQUIREMENT_BASIS"
if [ "$GPU_REQUIREMENT" = "not-required" ]; then
  # gpu_claim stays skip: no device was claimed, which remains the fact. What changes is
  # that the absence no longer suppresses the correctness stages. arch_coverage is left
  # empty because mark_runtime_coverage credits only a passing claim, so a run in this
  # mode cannot assert that any architecture was exercised.
  jset_string "stages.gpu_claim.requirement_note" \
    "the target does not require a GPU: $GPU_REQUIREMENT_BASIS"
  finding "note" "gpu_claim" \
    "the target ran with no visible GPU; correctness was checked but no runtime architecture coverage is claimed"
fi

CAN_TEST=1
SKIP_REASON=""
PERF_BASE_RC=""
PERF_BASE_LOGS=()
PERF_HEAD_LOGS=()
PERF_SKIP_REASON=""
# Per-target results, index-aligned with PERF_TARGETS. Pre-filled so that a phase that never
# ran leaves every slot with an answer -- an unset slot under `set -u` is a crash, and the
# report a crash produces is no report at all.
PERF_SKIP_REASONS=()
PERF_BASELINE_METHODS=()
PERF_ARGS_LIST=()
PERF_BASIS_LIST=()
PERF_BASE_RCS=()
PERF_BASE_LOGS_JOINED=()
PERF_HEAD_RCS=()
PERF_HEAD_LOGS_JOINED=()
for _slot in "${!PERF_TARGETS[@]}"; do
  PERF_SKIP_REASONS+=("")
  PERF_BASELINE_METHODS+=("patch-reversed-same-worktree")
  PERF_ARGS_LIST+=("")
  PERF_BASIS_LIST+=("")
  PERF_BASE_RCS+=("")
  PERF_BASE_LOGS_JOINED+=("")
  PERF_HEAD_RCS+=("")
  PERF_HEAD_LOGS_JOINED+=("")
done
if [ -z "$PICK" ] && [ "$GPU_REQUIREMENT" != "not-required" ]; then
  CAN_TEST=0
  SKIP_REASON="no verified-idle GPU was claimed"
elif [ "$RUNTIME_OK" -ne 1 ]; then
  CAN_TEST=0
  SKIP_REASON="runtime compatibility was not established"
elif [ "$TARGET_RUNNER" = "none" ]; then
  CAN_TEST=0
  SKIP_REASON="$TARGET_RUNNER_REASON"
elif [ "$TARGET_RUNNER" = "pytest" ] && ! (
  cd "$REPO_WT" \
    && PYTHONPATH="$TEST_PYTHONPATH" "$TARGET_PYTHON" -m pytest --version
) >/dev/null 2>&1; then
  CAN_TEST=0
  SKIP_REASON="python -m pytest is not runnable in this environment"
fi

BASE_REPO_STATE="not-run"
BASE_REPO_RC=""
BASE_REPO_LOG=""
BASE_GRID_STATE="not-run"
BASE_GRID_RC=""
BASE_GRID_LOG=""

if [ "$CAN_TEST" -eq 0 ]; then
  stage_note "baseline_control" "skip" "$SKIP_REASON"
  stage_note "correctness_repo_tests" "skip" "$SKIP_REASON"
  stage_note "correctness_s1_grid" "skip" "$SKIP_REASON"
  stage_note "execution_receipt" "skip" "$SKIP_REASON"
  finding "note" "correctness" "$SKIP_REASON; this report makes no correctness claim"
else
  BASE_READY=0
  # Keep a copy of the head target before the patch is reversed. A PR that ADDS its target
  # leaves base with nothing to time, which used to end the perf stage outright -- on
  # aiter#4538, a PR whose entire motivation is being faster than the kernel it replaces.
  # But "the file is new" is not the same as "the code it exercises is new": when the target
  # only drives an entry point that already exists on base, dropping this exact file into the
  # base tree times the OLD implementation through the SAME harness. That transplant is a
  # cross-tree comparison and is only attributable if something the patch does not touch
  # reproduces across it, which is what --perf-control-column requires below.
  PERF_TRANSPLANT_SRCS=()
  for _slot in "${!PERF_TARGETS[@]}"; do
    PERF_TRANSPLANT_SRCS+=("")
    if [ -n "$PATCHF" ] && [ "$PERF_ENABLED" -eq 1 ] \
        && [ -f "$REPO_WT/${PERF_TARGETS[$_slot]}" ]; then
      PERF_TRANSPLANT_SRCS[$_slot]="$WORK/transplant-target-$_slot"
      cp "$REPO_WT/${PERF_TARGETS[$_slot]}" "${PERF_TRANSPLANT_SRCS[$_slot]}"
    fi
  done
  if [ -n "$PATCHF" ]; then
    if git -C "$REPO_WT" apply -R --check "$PATCHF" >/dev/null 2>&1 \
        && git -C "$REPO_WT" apply -R "$PATCHF" >/dev/null 2>&1; then
      BASE_ACTIVE=1
      if [ -z "$(git -C "$REPO_WT" status --porcelain --untracked-files=all)" ]; then
        BASE_READY=1
      fi
    fi

    if [ "$BASE_READY" -eq 1 ]; then
      if [ -f "$REPO_WT/$TEST_FILE" ]; then
        BASE_RESULT=$(run_pytest "base-repo" "")
        BASE_REPO_RC=${BASE_RESULT%%|*}
        BASE_REPO_LOG=${BASE_RESULT##*|}
        BASE_REPO_STATS=$(target_stats "base-repo" "$BASE_REPO_RC")
        if [ "$BASE_REPO_RC" -eq 0 ] \
            && [ "$(stats_field "$BASE_REPO_STATS" executed)" -eq 0 ]; then
          BASE_REPO_STATE="all-skipped"
        else
          BASE_REPO_STATE="ran"
        fi
      else
        BASE_REPO_STATE="target-not-present"
      fi
      # Time the baseline HERE, inside the base phase. This is the only window in which the
      # patch is reversed out on this worktree, and the same locked GPU is still held. A base
      # number taken later, or on another box, or from the PR description, reintroduces
      # exactly the variance a 0.95 threshold is too tight to absorb.
      if [ "$PERF_ENABLED" -eq 1 ]; then
        # One pass per discovered target. Each keeps its own harness, its own baseline method
        # and its own skip reason, because a target that could not be timed says nothing
        # about the one beside it: the repository's bench takes an ordinary reversed-patch
        # baseline while the PR's own bench needs a transplant, and either may fail alone.
        for PERF_SLOT in "${!PERF_TARGETS[@]}"; do
          PERF_TARGET="${PERF_TARGETS[$PERF_SLOT]}"
          PERF_BASELINE_METHOD="patch-reversed-same-worktree"
          PERF_SKIP_REASON=""
          PERF_BASE_RC=""
          PERF_BASE_LOGS=()
          [ "$PERF_ARGS_SET" -eq 1 ] || PERF_ARGS=""
          PERF_BASIS=""
          PERF_TRANSPLANT_SRC="${PERF_TRANSPLANT_SRCS[$PERF_SLOT]}"
          # Whether the PERF target survives the reverse-apply is its own question. It used to
          # be answered with BASE_REPO_STATE, which describes the CORRECTNESS target -- the
          # same file, back when perf had no target of its own. Once the two can differ that
          # is a category error in both directions: a pre-existing bench alongside a PR-added
          # unit test would be refused a baseline it could trivially have taken, and a
          # PR-added bench alongside a pre-existing unit test would fall through to the
          # ordinary branch and transplant nothing. Ask about the file about to be executed.
          PERF_BASE_STATE="present"
          [ -f "$REPO_WT/$PERF_TARGET" ] || PERF_BASE_STATE="target-not-present"
          if [ "$PERF_BASE_STATE" = "target-not-present" ] \
              && [ -z "$PERF_CONTROL_COLUMN" ]; then
            PERF_SKIP_REASON="the PR adds this target, so a base timing requires transplanting it into the base tree; that comparison spans two trees and is only attributable when a column the patch does not touch reproduces across it, so --perf-control-column is required and was not supplied"
          elif [ "$PERF_BASE_STATE" = "target-not-present" ] \
              && [ -n "$PERF_TRANSPLANT_SRC" ] && [ -r "$PERF_TRANSPLANT_SRC" ]; then
            mkdir -p "$(dirname "$REPO_WT/$PERF_TARGET")"
            cp "$PERF_TRANSPLANT_SRC" "$REPO_WT/$PERF_TARGET"
            PERF_BASELINE_METHOD="target-transplant"
            if [ "$PERF_ARGS_SET" -eq 1 ] || perf_detect; then
              perf_snapshot base
              run_perf_repeats base "$PERF_SLOT"
              PERF_BASE_RC=$PERF_RUN_RC
              PERF_BASE_LOGS=("${PERF_RUN_LOGS[@]}")
              perf_restore base
            else
              PERF_SKIP_REASON="$PERF_BASIS"
            fi
            # The transplanted file is not part of the base tree and must not be left in it:
            # the cleanliness check that guards the head phase would otherwise fail and take
            # the whole correctness phase down with it. Reached only when the perf target was
            # absent from base a moment ago, so this deletes what the two lines above wrote
            # and nothing else. Spelled with $TEST_FILE it would delete a TRACKED base file
            # whenever the two targets differ -- dirtying the tree it keeps clean.
            rm -f "$REPO_WT/$PERF_TARGET"
          elif [ "$PERF_BASE_STATE" = "target-not-present" ]; then
            PERF_SKIP_REASON="the PR adds this target and no copy of it was available to transplant onto base"
          elif [ "$PERF_ARGS_SET" -eq 1 ] || perf_detect; then
            perf_snapshot base
            run_perf_repeats base "$PERF_SLOT"
            PERF_BASE_RC=$PERF_RUN_RC
            PERF_BASE_LOGS=("${PERF_RUN_LOGS[@]}")
            perf_restore base
          else
            PERF_SKIP_REASON="$PERF_BASIS"
          fi
          PERF_SKIP_REASONS[$PERF_SLOT]="$PERF_SKIP_REASON"
          PERF_BASELINE_METHODS[$PERF_SLOT]="$PERF_BASELINE_METHOD"
          PERF_ARGS_LIST[$PERF_SLOT]="$PERF_ARGS"
          PERF_BASIS_LIST[$PERF_SLOT]="$PERF_BASIS"
          PERF_BASE_RCS[$PERF_SLOT]="$PERF_BASE_RC"
          # Log lists are variable-length, and bash has no array of arrays. They are joined on
          # newlines here and split back on newlines at the point of use; every path is one
          # this script generated under $WORK, so none of them can contain one.
          PERF_BASE_LOGS_JOINED[$PERF_SLOT]=$(printf '%s\n' "${PERF_BASE_LOGS[@]}")
        done
        PERF_TARGET="${PERF_TARGETS[0]}"
      fi
      if [ "$GRID_HOOK_OK" -eq 1 ]; then
        if [ -f "$REPO_WT/$TEST_FILE" ]; then
          BASE_PROBE_RESULT=$(run_pytest \
            "base-grid-probe" "__VALIDATOR_INVALID_GRID__")
          BASE_PROBE_RC=${BASE_PROBE_RESULT%%|*}
          BASE_PROBE_LOG=${BASE_PROBE_RESULT##*|}
          # A non-zero probe exit is only evidence that the GRID was consumed when the same
          # target succeeds without it. On a held-out PR whose module could not be imported at
          # all, the probe failed for that reason and the channel was credited although no
          # shape ever reached the kernel. Require the unpoisoned baseline run to have passed.
          if [ "$BASE_PROBE_RC" -eq 0 ] || [ "${BASE_REPO_RC:-1}" -ne 0 ]; then
            BASE_GRID_STATE="hook-not-consumed"
          else
            BASE_GRID_RESULT=$(run_pytest "base-grid" "$GRID")
            BASE_GRID_RC=${BASE_GRID_RESULT%%|*}
            BASE_GRID_LOG=${BASE_GRID_RESULT##*|}
            BASE_GRID_STATS=$(target_stats "base-grid" "$BASE_GRID_RC")
            if [ "$BASE_GRID_RC" -eq 0 ] \
                && [ "$(stats_field "$BASE_GRID_STATS" executed)" -eq 0 ]; then
              BASE_GRID_STATE="all-skipped"
            else
              BASE_GRID_STATE="ran"
            fi
          fi
        else
          BASE_GRID_STATE="target-not-present"
        fi
      elif [ -n "$GRID" ]; then
        # Not "hook-not-found": nothing was looked for. A grid was requested and no channel
        # was declared to carry it.
        BASE_GRID_STATE="no-channel-declared"
      else
        BASE_GRID_STATE="not-configured"
      fi
      if [ -n "$(git -C "$REPO_WT" status --porcelain --untracked-files=all)" ]; then
        BASE_READY=0
      fi
      CURRENT_IGNORED=$(git -C "$REPO_WT" status --porcelain \
        --ignored --untracked-files=all | awk '$1 == "!!"')
      if [ "$CURRENT_IGNORED" != "$INITIAL_IGNORED" ]; then
        BASE_READY=0
      fi
    fi

    if ! restore_head; then
      BASE_READY=0
      CAN_TEST=0
      stage_note "baseline_control" "skip" \
        "candidate patch could not be restored after the baseline run"
      finding "note" "baseline_control" \
        "failed to restore the candidate patch; head tests were not run"
    elif [ "$BASE_READY" -ne 1 ]; then
      CAN_TEST=0
      stage_note "baseline_control" "skip" \
        "base run did not leave a clean worktree; head tests were not run"
      finding "note" "baseline_control" \
        "base isolation failed or produced worktree artifacts; attribution is inconclusive"
    else
      [ -n "${BASE_REPO_STATS:-}" ] || \
        BASE_REPO_STATS='{"tests":0,"failures":0,"errors":0,"skipped":0,"executed":0}'
      [ -n "${BASE_GRID_STATS:-}" ] || \
        BASE_GRID_STATS='{"tests":0,"failures":0,"errors":0,"skipped":0,"executed":0}'
      python3 - "$JSON" "$BASE_REPO_STATE" "${BASE_REPO_RC:-}" \
        "$BASE_REPO_LOG" "$BASE_REPO_STATS" "$BASE_GRID_STATE" \
        "${BASE_GRID_RC:-}" "$BASE_GRID_LOG" "$BASE_GRID_STATS" \
        "${BASE_PROBE_RC:-}" "${BASE_PROBE_LOG:-}" <<'PY'
import json
import sys

(
    path,
    repo_state,
    repo_exit,
    repo_log,
    repo_stats,
    grid_state,
    grid_exit,
    grid_log,
    grid_stats,
    probe_exit,
    probe_log,
) = sys.argv[1:12]
stage = {
    "status": "pass",
    "repo_tests": {"state": repo_state, "stats": json.loads(repo_stats)},
    "s1_grid": {"state": grid_state, "stats": json.loads(grid_stats)},
}
if repo_exit:
    stage["repo_tests"]["exit"] = int(repo_exit)
    stage["repo_tests"]["log"] = repo_log
if grid_exit:
    stage["s1_grid"]["exit"] = int(grid_exit)
    stage["s1_grid"]["log"] = grid_log
if probe_exit:
    stage["s1_grid"]["hook_probe_exit"] = int(probe_exit)
    stage["s1_grid"]["hook_probe_log"] = probe_log
data = json.load(open(path))
data["stages"]["baseline_control"] = stage
json.dump(data, open(path, "w"), indent=2)
PY
    fi
  else
    stage_note "baseline_control" "skip" \
      "no patch supplied; failures on this checkout cannot be attributed against a base control"
  fi

  if [ "$CAN_TEST" -eq 1 ]; then
    HEAD_RESULT=$(run_pytest "head-repo" "")
    HEAD_RC=${HEAD_RESULT%%|*}
    HEAD_LOG=${HEAD_RESULT##*|}
    HEAD_STATS=$(target_stats "head-repo" "$HEAD_RC")
    HEAD_EXECUTED=$(stats_field "$HEAD_STATS" executed)
    python3 - "$JSON" "$HEAD_RC" "$HEAD_LOG" "$HEAD_STATS" <<'PY'
import json
import sys

path, exit_code, log, raw_stats = sys.argv[1:5]
data = json.load(open(path))
stats = json.loads(raw_stats)
status = "fail" if int(exit_code) else ("pass" if stats["executed"] else "skip")
data["stages"]["correctness_repo_tests"] = {
    "status": status,
    "exit": int(exit_code),
    "log": log,
    "stats": stats,
}
if status == "skip":
    data["stages"]["correctness_repo_tests"]["note"] = (
        "target completed with no executed tests"
    )
json.dump(data, open(path, "w"), indent=2)
PY
    mark_runtime_coverage "$HEAD_STATS" "$TARGET_RUNNER" "$HEAD_LOG"
    if [ "$HEAD_RC" -eq 0 ] && [ "$HEAD_EXECUTED" -eq 0 ]; then
      finding "note" "correctness" \
        "repository target executed no tests; no correctness claim is made"
    elif [ "$HEAD_RC" -ne 0 ]; then
      HEAD_EXCERPT=$(log_excerpt "$HEAD_LOG")
      if [ -z "$PATCHF" ]; then
        finding "blocker" "correctness" \
          "the supplied head checkout's test target fails: $HEAD_EXCERPT"
      elif [ "$BASE_REPO_STATE" = "target-not-present" ]; then
        finding "blocker" "correctness" \
          "the PR adds this test target and it fails on head: $HEAD_EXCERPT"
      elif [ "$BASE_REPO_STATE" = "ran" ] && [ "$BASE_REPO_RC" -eq 0 ]; then
        finding "blocker" "correctness" \
          "the test target passes on base and fails on head: $HEAD_EXCERPT"
      else
        finding "note" "correctness" \
          "the test target is red on both baseline and head; the failure is not attributed without matching failure evidence"
      fi
      # "Red on both sides" is an attribution, not an explanation. A reader who is not told
      # that the runner could be the cause concludes the code is broken when the choice was.
      if [ "$HEAD_EXECUTED" -eq 0 ]; then
        finding "note" "correctness" \
          "the target executed nothing under the $TARGET_RUNNER runner ($TARGET_RUNNER_BASIS); a runner that cannot collect or execute this target produces exactly this result, so the runner selection is a candidate cause and the code is not the only one"
      fi
    fi

    # Head's timing run pairs with the base one and is skipped outright when base produced
    # nothing: a head-only number reproduces the PR's own comparison and cannot show a
    # regression, which is the single thing this stage is for.
    if [ "$PERF_ENABLED" -eq 1 ]; then
      for PERF_SLOT in "${!PERF_TARGETS[@]}"; do
        [ -n "${PERF_BASE_LOGS_JOINED[$PERF_SLOT]:-}" ] || continue
        PERF_TARGET="${PERF_TARGETS[$PERF_SLOT]}"
        PERF_ARGS="${PERF_ARGS_LIST[$PERF_SLOT]}"
        perf_snapshot head
        run_perf_repeats head "$PERF_SLOT"
        PERF_HEAD_RCS[$PERF_SLOT]=$PERF_RUN_RC
        PERF_HEAD_LOGS_JOINED[$PERF_SLOT]=$(printf '%s\n' "${PERF_RUN_LOGS[@]}")
        # Symmetric with base: the grid run and the caller's worktree both follow this point,
        # and neither should inherit a results file the timing run happened to drop.
        perf_restore head
      done
      PERF_TARGET="${PERF_TARGETS[0]}"
    fi


    # Same causality requirement as the base side: a probe that fails because the target
    # is broken proves nothing about the grid. On a held-out PR whose module could not be
    # imported at all, the probe's non-zero exit credited the channel although no shape
    # ever reached the kernel. Require the unpoisoned head run to have passed first.
    if [ "$GRID_HOOK_OK" -eq 1 ] && [ "${HEAD_RC:-1}" -eq 0 ]; then
      HEAD_PROBE_RESULT=$(run_pytest \
        "head-grid-probe" "__VALIDATOR_INVALID_GRID__")
      HEAD_PROBE_RC=${HEAD_PROBE_RESULT%%|*}
      HEAD_PROBE_LOG=${HEAD_PROBE_RESULT##*|}
      if [ "$HEAD_PROBE_RC" -eq 0 ]; then
        # State what was observed, and do not assign it to a party. The target ran unchanged
        # with a deliberately invalid grid, which happens both when the caller named a channel
        # this target does not have and when the target has it and ignores it. Naming the
        # target as the one that "ignores" the channel publishes a caller's mistake as a
        # property of someone's code.
        stage_note "correctness_s1_grid" "skip" \
          "the declared $GRID_CHANNEL channel was not consumed: the target ran unchanged with a deliberately invalid grid, so either it does not read what --shape-$GRID_CHANNEL named or it ignores it; either way no shape reached the kernel"
        stage_note "execution_receipt" "skip" \
          "shape-grid runtime handshake failed on $GRID_CHANNEL"
        jset_json "stages.correctness_s1_grid.hook_probe_exit" "$HEAD_PROBE_RC"
        jset_string "stages.correctness_s1_grid.hook_probe_log" "$HEAD_PROBE_LOG"
        finding "note" "correctness" \
          "the selected target passes an invalid shape-grid probe, so grid consumption is unproven"
      else
        # Every requested axis must be observed REFUSING an invalid value before its values
        # are allowed onto the grid run's argv. Without this an axis flag the target declares
        # but ignores -- or one whose value it silently clamps -- would let the report claim
        # coverage of head counts or dtypes that never reached the kernel. An axis that fails
        # the probe is dropped from the run and named in the report; it is never dropped
        # quietly, because a silently narrowed test space is the failure this stage exists to
        # prevent.
        if [ "$AXIS_STATE" = "declared" ]; then
          AXIS_REFUSED_OK=1
          AXIS_PROBE_FAILED=""
          for _axis_flag in $(python3 -c '
import json
import sys

for axis in json.loads(sys.argv[1]):
    print(axis["flag"])
' "$AXIS_REPORT"); do
            AXIS_CLI_OVERRIDE=("$_axis_flag" "__VALIDATOR_INVALID_AXIS__")
            AXIS_PROBE_RESULT=$(run_pytest "head-axisprobe" "")
            AXIS_CLI_OVERRIDE=()
            if [ "${AXIS_PROBE_RESULT%%|*}" -eq 0 ]; then
              AXIS_REFUSED_OK=0
              AXIS_PROBE_FAILED="$AXIS_PROBE_FAILED $_axis_flag"
            fi
          done
          if [ "$AXIS_REFUSED_OK" -eq 1 ]; then
            AXIS_STATE="proven"
            AXIS_STATE_REASON="every axis flag rejected a deliberately invalid value"
          else
            AXIS_STATE="hook-not-consumed"
            AXIS_STATE_REASON="these axis flags accepted a deliberately invalid value, so the target does not consume them:$AXIS_PROBE_FAILED"
            AXIS_CLI=()
            finding "note" "correctness" \
              "requested test axes were dropped: $AXIS_STATE_REASON"
          fi
          # `hook_proof` carries the probe's own verdict per axis. It used to hold a
          # structural reading of the target's source, which said nothing about whether the
          # value arrived; the only evidence that ever counted is this refusal.
          AXIS_REPORT=$(python3 -c '
import json
import sys

failed = set(sys.argv[2].split())
axes = json.loads(sys.argv[1])
for axis in axes:
    axis["hook_proof"] = (
        "accepted-invalid-value" if axis["flag"] in failed else "refused-invalid-value"
    )
print(json.dumps(axes))
' "$AXIS_REPORT" "$AXIS_PROBE_FAILED")
          jset_json "test_selection.axes" "$AXIS_REPORT"
          jset_string "test_selection.axis_state" "$AXIS_STATE"
          jset_string "test_selection.axis_state_reason" "$AXIS_STATE_REASON"
        fi
        HEAD_GRID_RESULT=$(run_pytest "head-grid" "$GRID")
        HEAD_GRID_RC=${HEAD_GRID_RESULT%%|*}
        HEAD_GRID_LOG=${HEAD_GRID_RESULT##*|}
        HEAD_GRID_STATS=$(target_stats "head-grid" "$HEAD_GRID_RC")
        HEAD_GRID_EXECUTED=$(stats_field "$HEAD_GRID_STATS" executed)
        python3 - "$JSON" "$HEAD_GRID_RC" "$GRID" "$HEAD_GRID_LOG" \
          "$HEAD_GRID_STATS" "$HEAD_PROBE_RC" "$HEAD_PROBE_LOG" \
          "$GRID_CHANNEL" "${HEAD_RC:-1}" <<'PY'
import json
import sys

(
    path,
    exit_code,
    grid,
    log,
    raw_stats,
    probe_exit,
    probe_log,
    channel,
    repo_exit_code,
) = sys.argv[1:10]
data = json.load(open(path))
stats = json.loads(raw_stats)
status = "fail" if int(exit_code) else ("pass" if stats["executed"] else "skip")
note = ""
if status == "skip":
    note = "shape-grid target completed with no executed tests"
# A grid run that fails without its receipt observing ANY routed work never reached the code
# under test. That happens when the declared channel does not exist -- an unknown flag, a
# parametrization naming an argument no test takes -- and it also happens when the grid asked for
# a shape that crashes before the kernel runs, which is the grid doing its job. Nothing in the
# evidence separates those two, so the status stays "fail" and the report says both are possible;
# what does NOT happen is charging it to the author as a blocker, below.
if status == "fail" and stats.get("observed_work") == 0 and int(repo_exit_code) == 0:
    note = (
        f"the grid run failed without its execution receipt observing any call to the routed "
        f"work that the repository run reached, so the kernel was never seen failing on these "
        f"shapes; a declared '{channel}' channel this target does not have and a shape that "
        f"crashes before the route both look exactly like this"
    )
# A "pass" here is recorded and earns nothing: this stage is not required, so its status cannot
# complete a verdict, and the only thing a duplicate grid's pass would prove is that the
# repository run passed -- which correctness_repo_tests already said. A "fail" is the direction
# that carries weight, and it carries it whatever the cells were.
data["stages"]["correctness_s1_grid"] = {
    "status": status,
    "exit": int(exit_code),
    "grid": grid,
    "log": log,
    "stats": stats,
    "hook_probe_exit": int(probe_exit),
    "hook_probe_log": probe_log,
}
if note:
    data["stages"]["correctness_s1_grid"]["note"] = note
json.dump(data, open(path, "w"), indent=2)
PY
        mark_runtime_coverage "$HEAD_GRID_STATS" "$TARGET_RUNNER" "$HEAD_GRID_LOG"
        if [ "$HEAD_GRID_RC" -eq 0 ] && [ "$HEAD_GRID_EXECUTED" -eq 0 ]; then
          finding "note" "correctness" \
            "shape-grid target executed no tests; no grid claim is made"
        elif [ "$HEAD_GRID_RC" -ne 0 ]; then
          GRID_EXCERPT=$(log_excerpt "$HEAD_GRID_LOG")
          # A blocker says "this PR is broken". When the grid run's own receipt observed no
          # call to the routed work that the repository run DID reach, the kernel was never
          # seen failing on these shapes -- the run stopped before it got there. That is not
          # enough to charge anyone, and a mistyped --shape-arg reaches this line looking
          # exactly like a real defect.
          if [ "$(stats_field "$HEAD_GRID_STATS" observed_work)" = "0" ] \
              && [ "$(stats_field "$HEAD_STATS" observed_work)" != "0" ]; then
            finding "note" "correctness" \
              "the independent shape grid failed without reaching the routed work the repository run reached, so no defect is attributed: $GRID_EXCERPT"
          elif [ -z "$PATCHF" ]; then
            finding "blocker" "correctness" \
              "the independent shape grid fails on the supplied head checkout: $GRID_EXCERPT"
          elif [ "$BASE_GRID_STATE" = "target-not-present" ]; then
            finding "blocker" "correctness" \
              "the PR adds this target and its independent shape grid fails: $GRID_EXCERPT"
          elif [ "$BASE_GRID_STATE" = "ran" ] && [ "$BASE_GRID_RC" -eq 0 ]; then
            finding "blocker" "correctness" \
              "the independent shape grid passes on base and fails on head: $GRID_EXCERPT"
          else
            finding "note" "correctness" \
              "the independent grid is red on both baseline and head; attribution is inconclusive"
          fi
        fi
        # The grid's OWN receipt, not whichever head run wrote last. The grid exists to be a
        # positive control against re-reporting the repo-default run under a second stage
        # name; reading a shared receipt made that control unfalsifiable, because a receipt
        # written by the default run satisfies --grid whenever the grid shapes are a subset
        # of the target's own defaults.
        RECEIPT_JSON=$(
          python3 "$SCRIPT_DIR/validate_evidence.py" receipt \
            "$(head_receipt)" \
            --expected-route "$EXPECTED_ROUTE" --grid "$GRID" \
            --grid-channel "$GRID_CHANNEL"
        )
        jset_json "stages.execution_receipt" "$RECEIPT_JSON"
        jset_string "stages.execution_receipt.receipt_scope" \
          "the head-grid run only; the head-repo run has its own receipt"
        RECEIPT_STATUS=$(python3 - "$RECEIPT_JSON" <<'PY'
import json
import sys

print(json.loads(sys.argv[1])["status"])
PY
)
        if [ "$RECEIPT_STATUS" != "pass" ]; then
          finding "note" "execution_receipt" \
            "route/shape execution receipt was not established; PASS is not permitted"
        fi
      fi
    elif [ -n "$SHAPE_ENV" ] && [ -n "$GRID" ]; then
      stage_note "correctness_s1_grid" "skip" \
        "configured shape environment variable is not referenced by the target"
      if [ -n "$EXPECTED_ROUTE" ] && [ -f "$(head_receipt)" ]; then
        RECEIPT_JSON=$(
          python3 "$SCRIPT_DIR/validate_evidence.py" receipt \
            "$(head_receipt)" \
            --expected-route "$EXPECTED_ROUTE" --grid "" --grid-channel ""
        )
        jset_json "stages.execution_receipt" "$RECEIPT_JSON"
        jset_string "stages.execution_receipt.receipt_scope" \
          "the head-repo run only; no grid run took place"
        RECEIPT_STATUS=$(python3 -c \
          'import json,sys; print(json.loads(sys.argv[1])["status"])' "$RECEIPT_JSON")
        if [ "$RECEIPT_STATUS" != "pass" ]; then
          finding "note" "execution_receipt" \
            "route execution receipt was not established; PASS is not permitted"
        fi
      else
        stage_note "execution_receipt" "skip" \
          "shape-grid hook was not established and no route was supplied"
      fi
      finding "note" "correctness" \
        "the selected target does not consume the configured shape-grid hook"
    else
      # A skip must describe what was actually found. "kernel exposes no configured shape
      # override" reads as a property of the target even when the real cause is that the
      # validator ignored the channel the caller named -- which is a capability gap wearing a
      # skip's costume, the exact failure this skill exists to prevent.
      if [ -n "$GRID_CHANNEL_REASON" ]; then
        stage_note "correctness_s1_grid" "skip" \
          "$GRID_CHANNEL_REASON; coverage is repo-default-only"
      elif [ -z "$GRID" ]; then
        stage_note "correctness_s1_grid" "skip" \
          "no --grid was supplied, so no independent shape coverage was attempted; coverage is repo-default-only"
      else
        stage_note "correctness_s1_grid" "skip" \
          "kernel exposes no configured shape override; coverage is repo-default-only"
      fi
      if [ -n "$EXPECTED_ROUTE" ] && [ -f "$(head_receipt)" ]; then
        RECEIPT_JSON=$(
          python3 "$SCRIPT_DIR/validate_evidence.py" receipt \
            "$(head_receipt)" \
            --expected-route "$EXPECTED_ROUTE" --grid "" --grid-channel ""
        )
        jset_json "stages.execution_receipt" "$RECEIPT_JSON"
        RECEIPT_STATUS=$(python3 -c \
          'import json,sys; print(json.loads(sys.argv[1])["status"])' "$RECEIPT_JSON")
        if [ "$RECEIPT_STATUS" != "pass" ]; then
          finding "note" "execution_receipt" \
            "route execution receipt was not established; PASS is not permitted"
        fi
      else
        stage_note "execution_receipt" "skip" \
          "no shape grid was configured and no route was supplied"
      fi
      finding "note" "correctness" \
        "no independent shape-grid hook was configured; coverage is limited to repository defaults"
    fi
  else
    stage_note "correctness_repo_tests" "skip" \
      "candidate patch was not restored after baseline control"
    stage_note "correctness_s1_grid" "skip" \
      "candidate patch was not restored after baseline control"
    stage_note "execution_receipt" "skip" \
      "candidate patch was not restored after baseline control"
  fi
fi

# ---------- stage 6: index-width scan (informational) ----------
SCANNER="$SCRIPT_DIR/scan_index_width.py"
if [ -z "$PATCHF" ]; then
  stage_note "index_width_scan" "skip" \
    "no patch supplied; there is no base-to-head diff to scan"
elif [ ! -x "$SCANNER" ]; then
  stage_note "index_width_scan" "skip" \
    "required scan_index_width.py is missing or not executable"
  finding "note" "index_width_scan" \
    "required index-width scan did not run; do not interpret this as an empty candidate list"
else
  # The scan is an AST pass and needs each changed file's POST image. Reading it from the
  # worktree while the patch is applied is exact and free; without it the scanner falls back
  # to the diff's index blobs, which are absent unless the PR head was fetched, and every
  # MODIFIED file lands in `unscanned`. A held-out PR ran with three of its files unexamined
  # while the stage still reported a candidate count.
  SCAN_ARGS=(--diff "$PATCHF" --json)
  if [ "$PATCH_APPLIED" -eq 1 ] && [ "$BASE_ACTIVE" -eq 0 ]; then
    SCAN_ARGS+=(--source-root "$REPO_WT")
  fi
  SCAN_JSON=$("$SCANNER" "${SCAN_ARGS[@]}" 2>"$WORK/index-width-scan.log")
  SCAN_RC=$?
  if [ "$SCAN_RC" -ne 0 ]; then
    stage_note "index_width_scan" "skip" "index-width scanner failed"
    finding "note" "index_width_scan" \
      "index-width scan failed; do not interpret this as an empty candidate list"
  else
    python3 - "$JSON" "$SCAN_JSON" <<'PY'
import json
import sys

path, raw = sys.argv[1:3]
data = json.load(open(path))
stage = json.loads(raw)
stage["status"] = "info"
stage["note"] = (
    "index x stride with no 64-bit widening; candidates require scale-aware review"
)
data["stages"]["index_width_scan"] = stage
json.dump(data, open(path, "w"), indent=2)
PY
    SCAN_COUNT=$(python3 - "$SCAN_JSON" <<'PY'
import json
import sys

print(json.loads(sys.argv[1])["total_candidates"])
PY
)
    if [ "$SCAN_COUNT" -gt 0 ]; then
      finding "note" "index_width_scan" \
        "$SCAN_COUNT index/stride candidates carry no explicit 64-bit widening; review each against production scale"
    fi
  fi
fi

# ---- perf stage.
#
# Emitted last because it is the only stage needing results from both the baseline phase and
# the head phase. It is deliberately NOT in finish_report's required-stage set -- the same
# footing the shape grid now stands on: `complete` is computed from the required stages alone,
# so a perf run that could not happen downgrades nothing and a PASS stays a PASS. What it can
# do is append a should-fix finding, which
# finish_report turns into NEEDS_WORK and exit 1 -- a measured regression is a real result,
# not an advisory note, and the whole point of putting it in the deterministic layer is that
# it ships its own reproducer (both logs, both exit codes, the command) with it.
#
# Every path that is not "both sides ran clean and the numbers disagree" reports `skip`.
# A timeout, a crash, a missing harness and a one-row table must never be able to look like
# a regression, because a false regression here blocks a good PR and would get the stage
# switched off within a week.
#
# Every discovered target contributes one measurement, whether or not it produced a number,
# and the stage mirrors whichever of them gates. Which that is, is decided in scrape_perf.py
# from the measurements; nothing here ranks anything.
PERF_STAGE_COMPOSED=0
if [ "$PERF_ENABLED" -ne 1 ]; then
  stage_note "perf" "skip" "perf measurement was disabled with --no-perf"
else
  PERF_MANIFEST="$WORK/perf-measurements.json"
  rm -f "$PERF_MANIFEST"
  for PERF_SLOT in "${!PERF_TARGETS[@]}"; do
    PERF_WHY="${PERF_SKIP_REASONS[$PERF_SLOT]}"
    PERF_JSON=""
    PERF_SLOT_BASE_LOGS=()
    PERF_SLOT_HEAD_LOGS=()
    [ -n "${PERF_BASE_LOGS_JOINED[$PERF_SLOT]}" ] \
      && mapfile -t PERF_SLOT_BASE_LOGS <<<"${PERF_BASE_LOGS_JOINED[$PERF_SLOT]}"
    [ -n "${PERF_HEAD_LOGS_JOINED[$PERF_SLOT]}" ] \
      && mapfile -t PERF_SLOT_HEAD_LOGS <<<"${PERF_HEAD_LOGS_JOINED[$PERF_SLOT]}"
    if [ -n "$PERF_WHY" ]; then
      :
    elif [ "${#PERF_SLOT_BASE_LOGS[@]}" -eq 0 ] \
        || [ "${#PERF_SLOT_HEAD_LOGS[@]}" -eq 0 ]; then
      PERF_WHY="the run did not reach both a baseline and a head phase"
      [ "$CAN_TEST" -eq 0 ] && PERF_WHY="${SKIP_REASON:-$PERF_WHY}"
    elif [ "${PERF_BASE_RCS[$PERF_SLOT]}" -ne 0 ] \
        || [ "${PERF_HEAD_RCS[$PERF_SLOT]}" -ne 0 ]; then
      # Deliberately not a regression. A nonzero exit means the log is truncated at an unknown
      # point, so any ratio drawn from it compares whatever printed before the crash.
      PERF_WHY="benchmark run exited nonzero (base=${PERF_BASE_RCS[$PERF_SLOT]} head=${PERF_HEAD_RCS[$PERF_SLOT]}); timings from a truncated run are not comparable"
    else
      PERF_JSON="$WORK/perf-compare-$PERF_SLOT.json"
      "$SCRIPT_DIR/scrape_perf.py" \
        --base "${PERF_SLOT_BASE_LOGS[@]}" --head "${PERF_SLOT_HEAD_LOGS[@]}" \
        --threshold "$PERF_THRESHOLD" --min-rows "$PERF_MIN_ROWS" \
        --out "$PERF_JSON" >/dev/null 2>"$WORK/perf-compare-$PERF_SLOT.err"
      PERF_CMP_RC=$?
      if [ "$PERF_CMP_RC" -ne 0 ] || [ ! -r "$PERF_JSON" ]; then
        PERF_WHY="the benchmark comparison failed: $(log_excerpt "$WORK/perf-compare-$PERF_SLOT.err")"
        PERF_JSON=""
      fi
    fi
    "$SCRIPT_DIR/scrape_perf.py" measure \
      --manifest "$PERF_MANIFEST" \
      --target "${PERF_TARGETS[$PERF_SLOT]}" \
      --target-basis "${PERF_TARGET_BASES[$PERF_SLOT]}" \
      --target-basis-reason "${PERF_TARGET_BASIS_REASONS[$PERF_SLOT]}" \
      --target-provenance "${PERF_TARGET_PROVENANCES[$PERF_SLOT]}" \
      --target-provenance-reason "${PERF_TARGET_PROVENANCE_REASONS[$PERF_SLOT]}" \
      --skip-reason "$PERF_WHY" --compare "$PERF_JSON" \
      --base-log "${PERF_SLOT_BASE_LOGS[0]:-}" \
      --head-log "${PERF_SLOT_HEAD_LOGS[0]:-}" \
      --base-sha "$BASE_SHA" --command "${PERF_ARGS_LIST[$PERF_SLOT]}" \
      --basis "${PERF_BASIS_LIST[$PERF_SLOT]}" \
      --baseline-method "${PERF_BASELINE_METHODS[$PERF_SLOT]}" \
      --control-column "$PERF_CONTROL_COLUMN" --control-tol "$PERF_CONTROL_TOL"
  done
  # A run that never got both phases cannot say a benchmark was missing -- it could not have
  # run one either way, and its verdict is INCONCLUSIVE, which a should-fix would overwrite
  # with the more confident NEEDS_WORK.
  # BASE_READY is only assigned inside the branch CAN_TEST=1 takes, so it is read through a
  # default: under `set -u` an unset name is a crash, and crashing here would cost the whole
  # report to answer a question whose answer is already "no".
  PERF_PHASES_REACHED=0
  [ "$CAN_TEST" -eq 1 ] && [ "${BASE_READY:-0}" -eq 1 ] && PERF_PHASES_REACHED=1
  "$SCRIPT_DIR/scrape_perf.py" stage --report "$JSON" --manifest "$PERF_MANIFEST" \
    --kernel-modules "$PERF_KERNEL_MODULES" --native-paths "$PERF_NATIVE_PATHS" \
    --candidates "$PERF_CANDIDATES" --phases-reached "$PERF_PHASES_REACHED"
  PERF_STAGE_COMPOSED=1
fi

# After the chain, not inside it. stage_note REPLACES stages.perf wholesale, so a field
# written by the branch above would survive on some paths and vanish on others -- and the
# reader most in need of knowing which file was timed is the one reading a `skip`. Skipped
# when the measurements composed the stage, because there the target fields describe the
# measurement that gates, and this would overwrite that answer with the first target's.
if [ "$PERF_STAGE_COMPOSED" -ne 1 ]; then
  jset_string "stages.perf.target" "$PERF_TARGET"
  jset_string "stages.perf.target_basis" "$PERF_TARGET_BASIS"
  jset_string "stages.perf.target_provenance" "$PERF_TARGET_PROVENANCE"
  jset_string "stages.perf.target_provenance_reason" "$PERF_TARGET_PROVENANCE_REASON"
  jset_string "stages.perf.target_basis_reason" "$PERF_TARGET_BASIS_REASON"
fi
# Reported even where discovery declined, and especially there: a reader who is told only
# "the fallback stood" cannot tell an empty search from a search that found three benches and
# refused to pick between them. The second of those is a question for the caller.
jset_json "stages.perf.candidates" "$PERF_CANDIDATES"

record_gpu_activity_after
finish_report
# `$WORK/verdict` is written by finish_report in the same breath as the report it
# describes, and `$WORK` belongs to this process alone. If it is absent, finish_report did
# not complete and there is no verdict to report -- which is INCONCLUSIVE, not whatever
# `--out` happens to hold.
FINAL_VERDICT=""
if [ -r "$WORK/verdict" ]; then
  FINAL_VERDICT=$(cat "$WORK/verdict")
else
  echo "validator internal error: no verdict was recorded for this run" >&2
fi
case "$FINAL_VERDICT" in
  PASS) exit 0;;
  BLOCK|NEEDS_WORK) exit 1;;
  INCONCLUSIVE) exit 2;;
  *) exit 2;;
esac
