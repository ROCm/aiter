#!/usr/bin/env bash
# S1 validate-kernel-pr -- deterministic validation layer for kernel PRs.
#
# Produces validation_report.json: the evidence base every review finding must hang on.
# Design rules it enforces (each learned from a real failure mode):
#   * isolation is REPORTED, never assumed  -- no docker here, so: worktree + private caches
#   * arch coverage is REPORTED, never implied -- a gfx950 box cannot validate a gfx942 claim
#   * the repo's own tests are NOT trusted as coverage -- S1 runs its own shape grid, because
#     a suite whose odd/unaligned shapes are commented out passes while the tail path is broken
#   * a green pytest with loosened tolerances is not a pass -- tolerances are policy-checked
#   * GPU is claimed over a sampling window and locked (kernel-profiling-optimization skill)
#
#   * correctness is not performance -- a kernel PR can compute the right values and still
#     be a regression, so base and head are also timed, on the same locked GPU, back to back
#
# usage: validate_pr.sh --repo <worktree> --target <test file or pytest node> [--patch p.patch]
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
    --patch) need_value "$@"; PATCHF="$2"; shift 2;;
    --head-sha) need_value "$@"; HEAD_SHA="$2"; shift 2;;
    --shape-env) need_value "$@"; SHAPE_ENV="$2"; shift 2;;
    --grid) need_value "$@"; GRID="$2"; shift 2;;
    --expected-route) need_value "$@"; EXPECTED_ROUTE="$2"; shift 2;;
    --shape-vars) need_value "$@"; SHAPE_VARS="$2"; shift 2;;
    --shape-arg) need_value "$@"; SHAPE_ARG="$2"; shift 2;;
    --tol-table) need_value "$@"; TOL_TABLE="$2"; shift 2;;
    --label) need_value "$@"; LABEL="$2"; shift 2;;
    --out) need_value "$@"; OUT="$2"; shift 2;;
    --perf-args) need_value "$@"; PERF_ARGS="$2"; PERF_ARGS_SET=1; shift 2;;
    --no-perf) PERF_ENABLED=0; shift;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac
done

if [ -z "$REPO_WT" ] || [ -z "$TESTS" ]; then
  echo "--repo and --target are required" >&2
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
WORK=$(mktemp -d "/tmp/validate-kernel-pr-XXXXXX")
JSON="$WORK/report.json"
PROBE_DIR="$WORK/probe"
PROBE_MODULE="validation_probe_${RANDOM}_${RANDOM}"
mkdir -p "$PROBE_DIR"
python3 - "$LABEL" "$JSON" <<'PY'
import json
import sys

json.dump(
    {"label": sys.argv[1], "stages": {}, "findings": []},
    open(sys.argv[2], "w"),
    indent=2,
)
PY

jset_json() {
  python3 - "$JSON" "$1" "$2" <<'PY'
import json
import sys

path, key, raw = sys.argv[1:4]
data = json.load(open(path))
current = data
parts = key.split(".")
for part in parts[:-1]:
    current = current.setdefault(part, {})
current[parts[-1]] = json.loads(raw)
json.dump(data, open(path, "w"), indent=2)
PY
}

jset_string() {
  python3 - "$JSON" "$1" "$2" <<'PY'
import json
import sys

path, key, value = sys.argv[1:4]
data = json.load(open(path))
current = data
parts = key.split(".")
for part in parts[:-1]:
    current = current.setdefault(part, {})
current[parts[-1]] = value
json.dump(data, open(path, "w"), indent=2)
PY
}

stage_note() {
  python3 - "$JSON" "$1" "$2" "$3" <<'PY'
import json
import sys

path, name, status, note = sys.argv[1:5]
data = json.load(open(path))
data["stages"][name] = {"status": status, "note": note}
json.dump(data, open(path, "w"), indent=2)
PY
}

finding() {
  python3 - "$JSON" "$1" "$2" "$3" <<'PY'
import json
import sys

path, severity, stage, detail = sys.argv[1:5]
data = json.load(open(path))
data["findings"].append(
    {"severity": severity, "stage": stage, "detail": detail}
)
json.dump(data, open(path, "w"), indent=2)
PY
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
  python3 - "$JSON" "$1" "$2" "$3" <<'PY'
import json
import pathlib
import sys

report_path, raw_stats, runner, log_path = sys.argv[1:5]
stats = json.loads(raw_stats)
if stats["executed"] < 1:
    raise SystemExit(0)
if runner == "script" and pathlib.Path(log_path).stat().st_size == 0:
    raise SystemExit(0)
data = json.load(open(report_path))
gpu = data["stages"].get("gpu_claim", {})
arch = gpu.get("arch")
if gpu.get("status") == "pass" and arch:
    data["arch_coverage"][arch] = "runtime"
    data.setdefault("arch_coverage_basis", {})[arch] = (
        f"pytest-junit-executed:{stats['executed']}"
        if runner == "pytest"
        else (
            "script-exit-zero-with-output"
            if stats["failures"] == 0
            else "script-nonzero-with-output"
        )
    )
    json.dump(data, open(report_path, "w"), indent=2)
PY
}

finish_report() {
  python3 - "$JSON" "$OUT" <<'PY'
import datetime
import json
import shutil
import sys

source, output = sys.argv[1:3]
data = json.load(open(source))
required_stages = (
    "merge_sim",
    "gpu_claim",
    "runtime_compat",
    "test_policy",
    "baseline_control",
    "correctness_repo_tests",
    "correctness_s1_grid",
    "execution_receipt",
    "index_width_scan",
)
for name in required_stages:
    if name not in data["stages"]:
        data["stages"][name] = {
            "status": "skip",
            "note": "validator internal error: stage did not record a result",
        }
        data["findings"].append(
            {
                "severity": "note",
                "stage": name,
                "detail": "stage result was missing; validation is inconclusive",
            }
        )

severities = {finding["severity"] for finding in data["findings"]}
complete = (
    isinstance(data.get("runtime_identity"), dict)
    and bool(data["runtime_identity"].get("module_path"))
    and data["stages"]["merge_sim"]["status"] == "pass"
    and data["stages"]["gpu_claim"]["status"] == "pass"
    and data["stages"]["runtime_compat"]["status"] == "pass"
    and data["stages"]["test_policy"]["status"] == "pass"
    and data["stages"]["baseline_control"]["status"] == "pass"
    and data["stages"]["correctness_repo_tests"]["status"] == "pass"
    and data["stages"]["correctness_s1_grid"]["status"] == "pass"
    and data["stages"]["execution_receipt"]["status"] == "pass"
    and data["stages"]["index_width_scan"]["status"] == "info"
)
if "blocker" in severities:
    verdict = "BLOCK"
elif "should-fix" in severities:
    verdict = "NEEDS_WORK"
elif not complete:
    verdict = "INCONCLUSIVE"
else:
    verdict = "PASS"
data["verdict"] = verdict
data["process_exit_code"] = (
    0 if verdict == "PASS" else (2 if verdict == "INCONCLUSIVE" else 1)
)
data["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime(
    "%Y-%m-%dT%H:%M:%SZ"
)
json.dump(data, open(source, "w"), indent=2)
shutil.copyfile(source, output)
print(f"verdict={verdict}  findings={len(data['findings'])}  -> {output}")
for item in data["findings"]:
    print(f"  [{item['severity']}] {item['stage']}: {item['detail'][:150]}")
PY
}

# Two independent facts about the supplied worktree:
#   BASE_ACTIVE=1    the patch is currently reversed out, i.e. we are mid-baseline-run
#   PATCH_APPLIED=1  this process applied the patch and still owes the caller a revert
BASE_ACTIVE=0
PATCH_APPLIED=0
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
  ACTIVITY_AFTER=$(HIP_ID="$PICK" python3 - "$SCRIPT_DIR/pick-idle-gpu.py" <<'PY'
import importlib.util
import os
import sys

spec = importlib.util.spec_from_file_location("validation_gpu_picker", sys.argv[1])
picker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(picker)
amdsmi = picker.import_amdsmi()

requested = int(os.environ["HIP_ID"])
amdsmi.amdsmi_init()
try:
    for handle in amdsmi.amdsmi_get_processor_handles():
        if amdsmi.amdsmi_get_gpu_enumeration_info(handle).get("hip_id") == requested:
            gfx, _ = picker.read_activity(amdsmi, handle)
            print("unavailable" if gfx is None else gfx)
            break
    else:
        raise RuntimeError(f"HIP index {requested} has no amd-smi mapping")
finally:
    amdsmi.amdsmi_shut_down()
PY
  )
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
jset_string "test_selection.target" "$TESTS"
jset_string "test_selection.shape_env" "$SHAPE_ENV"
jset_string "test_selection.grid" "$GRID"
jset_string "test_selection.shape_arg" "$SHAPE_ARG"
jset_string "test_selection.expected_route" "$EXPECTED_ROUTE"
jset_string "test_selection.shape_vars" "$SHAPE_VARS"
jset_string "test_selection.runner" "unresolved"
jset_string "test_selection.runner_reason" "merge simulation has not completed"

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

# ---------- stage 2: GPU claim (sampling window + whole-run lock) ----------
PICKER="${PICKER:-$(command -v pick-idle-gpu.py || true)}"
if [ -z "$PICKER" ]; then
  for candidate in "$SCRIPT_DIR/pick-idle-gpu.py" \
                   "$HOME/.local/bin/pick-idle-gpu.py" \
                   /usr/local/bin/pick-idle-gpu.py /opt/bin/pick-idle-gpu.py; do
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
  PICK=$("${PICKER_CMD[@]}" --samples 10 --interval 1 --quiet 2>"$WORK/gpu-picker.log")
  PICK_RC=$?
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
    exec {GPU_LOCK_FD}>"/tmp/gpu-$PICK.lock"
    if ! flock -n "$GPU_LOCK_FD"; then
      PICK=""
      stage_note "gpu_claim" "skip" \
        "selected GPU was claimed by another process before the lock was acquired"
      jset_string "degraded_mode" "NO_GPU"
      finding "note" "gpu_claim" "GPU claim raced with another process; no runtime correctness claim is made"
    else
      GPU_INFO=$(HIP_ID="$PICK" python3 - "$SCRIPT_DIR/pick-idle-gpu.py" <<'PY'
import importlib.util
import json
import os
import socket
import sys

spec = importlib.util.spec_from_file_location("validation_gpu_picker", sys.argv[1])
picker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(picker)
amdsmi = picker.import_amdsmi()

requested = int(os.environ["HIP_ID"])
amdsmi.amdsmi_init()
try:
    match = None
    for smi_index, handle in enumerate(amdsmi.amdsmi_get_processor_handles()):
        enumeration = amdsmi.amdsmi_get_gpu_enumeration_info(handle)
        if enumeration.get("hip_id") == requested:
            match = (smi_index, handle)
            break
    if match is None:
        raise RuntimeError(f"HIP index {requested} has no amd-smi mapping")
    smi_index, handle = match
    asic = amdsmi.amdsmi_get_gpu_asic_info(handle)
    gfx_activity, _ = picker.read_activity(amdsmi, handle)
    print(
        json.dumps(
            {
                "status": "pass",
                "hip_index": requested,
                "amd_smi_index": smi_index,
                "model": asic.get("market_name", "unknown"),
                "arch": asic.get("target_graphics_version", "unknown"),
                "bdf": amdsmi.amdsmi_get_gpu_device_bdf(handle),
                "gfx_activity_before_pct": gfx_activity,
                "host": socket.gethostname(),
            }
        )
    )
finally:
    amdsmi.amdsmi_shut_down()
PY
)
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
    assignments = [
        float(value)
        for value in re.findall(
            r"(?:atol|rtol)\s*=\s*([0-9.eE+-]+)", source
        )
    ]
    mappings = [
        float(value)
        for value in re.findall(
            r"""["'](?:f32|f16|bf16)["']\s*:\s*([0-9.eE+-]+)""",
            source,
        )
    ]
    return assignments + mappings

head_tolerances = tolerances(head)
base_tolerances = tolerances(base)
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
    "reference_tolerances": reference,
    "commented_out_shape_rows_base": commented_base,
    "commented_out_shape_rows": commented_head,
    "commented_out_shape_rows_added": commented_added,
    "kernel_files_changed": kernel_changed,
}
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
RUNNER_JSON=$(python3 - "$TARGET_PATH" "$TESTS" <<'PY'
import ast
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
selector = sys.argv[2]
if "::" in selector:
    result = {"runner": "pytest", "reason": "explicit pytest node selector"}
elif not path.is_file():
    result = {"runner": "none", "reason": "target file does not exist on head"}
else:
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError) as error:
        result = {"runner": "none", "reason": f"target AST is not readable: {error}"}
    else:
        has_pytest = any(
            (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name.startswith("test")
            )
            or (isinstance(node, ast.ClassDef) and node.name.startswith("Test"))
            for node in tree.body
        )
        has_main = any(
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
            and len(node.test.ops) == 1
            and isinstance(node.test.ops[0], ast.Eq)
            and len(node.test.comparators) == 1
            and isinstance(node.test.comparators[0], ast.Constant)
            and node.test.comparators[0].value == "__main__"
            for node in tree.body
        )
        if has_pytest:
            result = {"runner": "pytest", "reason": "target defines pytest test nodes"}
        elif has_main:
            result = {"runner": "script", "reason": "target has a __main__ entry point"}
        else:
            result = {"runner": "none", "reason": "target has no pytest nodes or __main__ entry point"}
print(json.dumps(result))
PY
)
TARGET_RUNNER=$(python3 - "$RUNNER_JSON" <<'PY'
import json
import sys

print(json.loads(sys.argv[1])["runner"])
PY
)
TARGET_RUNNER_REASON=$(python3 - "$RUNNER_JSON" <<'PY'
import json
import sys

print(json.loads(sys.argv[1])["reason"])
PY
)
jset_string "test_selection.runner" "$TARGET_RUNNER"
jset_string "test_selection.runner_reason" "$TARGET_RUNNER_REASON"
GRID_HOOK_OK=0
if [ -n "$SHAPE_ARG" ] && [ -n "$GRID" ] && [ "$TARGET_RUNNER" = "script" ] \
    && [ -f "$REPO_WT/$TEST_FILE" ]; then
  GRID_HOOK_OK=$(python3 - "$REPO_WT/$TEST_FILE" "$SHAPE_ARG" <<'PY'
import ast
import sys

tree = ast.parse(open(sys.argv[1], encoding='utf-8').read())
flag = sys.argv[2]
found = False
for node in ast.walk(tree):
    if not isinstance(node, ast.Call):
        continue
    if getattr(node.func, "attr", "") != "add_argument":
        continue
    for arg in node.args:
        if isinstance(arg, ast.Constant) and arg.value == flag:
            found = True
print(int(found))
PY
)
fi
if [ -n "$SHAPE_ENV" ] && [ -n "$GRID" ] \
    && [ -f "$REPO_WT/$TEST_FILE" ]; then
  GRID_HOOK_OK=$(python3 - "$REPO_WT/$TEST_FILE" "$SHAPE_ENV" <<'PY'
import ast
import sys

tree = ast.parse(open(sys.argv[1], encoding="utf-8").read())
name = sys.argv[2]

def attr_path(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return tuple(reversed(parts))

found = False
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and node.args:
        path = attr_path(node.func)
        key = node.args[0]
        if (
            path in {("os", "getenv"), ("os", "environ", "get")}
            and isinstance(key, ast.Constant)
            and key.value == name
        ):
            found = True
            break
    if isinstance(node, ast.Subscript) and attr_path(node.value) == ("os", "environ"):
        key = node.slice
        if isinstance(key, ast.Constant) and key.value == name:
            found = True
            break
print(int(found))
PY
)
fi

run_pytest() {
  local label="$1"
  local shape_assignment="$2"
  local log="$WORK/$TARGET_RUNNER-$label.log"
  local phase=${label%%-*}
  local cache_root="$WORK/$phase"
  local junit="$cache_root/junit-$label.xml"
  local receipt="$cache_root/execution-receipt.json"
  mkdir -p "$cache_root/home" "$cache_root/xdg-cache" \
    "$cache_root/flydsl-cache" "$cache_root/triton-cache" \
    "$cache_root/torch-extensions" "$cache_root/pytest-cache" \
    "$cache_root/aiter-jit"
  rm -f "$junit" "$receipt"
  if [ "$TARGET_RUNNER" = "pytest" ] || [ -n "$EXPECTED_ROUTE" ]; then
    python3 - "$SCRIPT_DIR/validation_probe.py" \
      "$PROBE_DIR/$PROBE_MODULE.py" "$EXPECTED_ROUTE" "$SHAPE_VARS" "$receipt" <<'PY'
import pathlib
import sys

source, output, route, shape_vars, receipt = sys.argv[1:6]
text = pathlib.Path(source).read_text()
text += (
    f"\n_VALIDATION_EXPECTED_ROUTE = {route!r}\n"
    f"_VALIDATION_SHAPE_VARS = {shape_vars!r}\n"
    f"_VALIDATION_RECEIPT_PATH = {receipt!r}\n"
)
pathlib.Path(output).write_text(text)
PY
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
  local -a shape_cli=()
  if [ -n "$shape_assignment" ] && [ -n "$SHAPE_ARG" ] \
      && [ "$TARGET_RUNNER" = "script" ]; then
    shape_cli=("$SHAPE_ARG")
    local _grid_value="${shape_assignment#*=}"
    local _old_ifs="$IFS"
    IFS=';'
    for _shape in $_grid_value; do
      [ -n "$_shape" ] && shape_cli+=("$_shape")
    done
    IFS="$_old_ifs"
    shape_assignment=""
  fi
  if [ -n "$shape_assignment" ]; then
    environment+=("$shape_assignment")
  fi
  if [ "$TARGET_RUNNER" = "pytest" ]; then
    (
      cd "$REPO_WT" \
        && env "${environment[@]}" timeout "$TIMEOUT" \
          "$TARGET_PYTHON" -m pytest -p "$PROBE_MODULE" "$TESTS" -x -q \
            --junitxml="$junit" -o "cache_dir=$cache_root/pytest-cache"
    ) >"$log" 2>&1
  elif [ -n "$EXPECTED_ROUTE" ]; then
    (
      cd "$REPO_WT" \
        && env "${environment[@]}" timeout "$TIMEOUT" \
          "$TARGET_PYTHON" "$SCRIPT_DIR/run_script_with_probe.py" \
            "$PROBE_MODULE" "$TEST_FILE" "${shape_cli[@]}"
    ) >"$log" 2>&1
  else
    (
      cd "$REPO_WT" \
        && env "${environment[@]}" timeout "$TIMEOUT" \
          "$TARGET_PYTHON" "$TEST_FILE" "${shape_cli[@]}"
    ) >"$log" 2>&1
  fi
  local result=$?
  echo "$result|$log"
}

# Decide whether the target can be timed at all, and with what arguments. A perf harness
# cannot be inferred from the diff, only from the target's own text, and aiter carries three
# conventions for it.
perf_detect() {
  local file="$REPO_WT/$TEST_FILE"
  if [ ! -f "$file" ]; then
    PERF_BASIS="the target file is not present in this checkout"
    return 1
  fi
  local detected
  detected=$(python3 - "$file" <<'PY'
import pathlib
import sys

text = pathlib.Path(sys.argv[1]).read_text(errors="replace")
if "--scenario" in text and "bench" in text:
    print("--scenario bench")
elif "perftest" in text or "@benchmark" in text:
    # `perftest`, not `run_perftest`. aiter has three timing conventions and the bare
    # `perftest` decorator is one of them; matching only the longer name misses 12 of the
    # 123 targets in op_tests/, 11 of which have live `perftest` usage. Reporting those as
    # "no benchmark entry point" reads as "there was nothing to measure" when the truth is
    # that the detector was too narrow -- the failure mode this whole stage exists to avoid.
    # This is a substring test, not a parse, so it also matches a commented-out import (the
    # 12th target). That error is the safe one: the run finds no timing table and the stage
    # reports `skip`, which is where it would have landed anyway.
    print("")
else:
    raise SystemExit(3)
PY
  )
  if [ $? -ne 0 ]; then
    PERF_BASIS="the target exposes no benchmark entry point (no --scenario bench, no perftest/@benchmark harness)"
    return 1
  fi
  PERF_ARGS="$detected"
  if [ -n "$detected" ]; then
    PERF_BASIS="target exposes --scenario bench"
  else
    PERF_BASIS="target uses the perftest/@benchmark harness"
  fi
  return 0
}

# The timing counterpart to run_pytest. Three things differ, and each difference is the point:
#   * no probe module is injected. The receipt probe wraps every route call to record shapes;
#     a traced kernel is not the kernel whose latency we are about to report.
#   * the phase cache root is shared with that phase's correctness run, so the JIT cache is
#     already warm and the table measures the kernel rather than a compile.
#   * PERF_TIMEOUT is separate from TIMEOUT, because silently killing a legitimately long
#     sweep would produce an empty log -- indistinguishable from "this target has no harness".
# A bench harness routinely writes its results next to the code -- aiter targets drop a
# tuned_op_bench.csv in the repo root. The baseline phase asserts a CLEAN worktree after the
# base runs, so an artifact left by the timing run sets BASE_READY=0 and skips the entire head
# correctness phase: measured, the same target went PASS with --no-perf and INCONCLUSIVE with
# perf on, with head correctness never executed. A perf stage that silently disables
# correctness validation is far worse than no perf stage, so the timing run has to leave the
# worktree exactly as it found it.
#
# Scoped deliberately: only paths whose git status CHANGED across the timing run are touched.
# Anything already dirty beforehand is somebody else's and is left alone.
perf_snapshot() {
  git -C "$REPO_WT" status --porcelain --untracked-files=all \
    >"$WORK/perf-worktree-$1.txt" 2>/dev/null || : >"$WORK/perf-worktree-$1.txt"
}

perf_restore() {
  local before="$WORK/perf-worktree-$1.txt"
  [ -r "$before" ] || return 0
  python3 - "$REPO_WT" "$before" <<'PY'
import pathlib
import shutil
import subprocess
import sys

root = pathlib.Path(sys.argv[1]).resolve()


def parse(text):
    entries = {}
    for line in text.splitlines():
        if len(line) > 3:
            entries[line[3:]] = line[:2]
    return entries


before = parse(pathlib.Path(sys.argv[2]).read_text(errors="replace"))
current = parse(
    subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
)

removed, reverted, skipped = [], [], []
for path, code in current.items():
    if before.get(path) == code:
        continue
    # git quotes paths with unusual characters. Un-quoting them correctly is fiddly and
    # this code deletes files, so refuse to guess and report instead.
    if path.startswith('"'):
        skipped.append(path)
        continue
    target = root / path
    try:
        resolved = target.resolve()
    except OSError:
        skipped.append(path)
        continue
    if root != resolved and root not in resolved.parents:
        skipped.append(path)
        continue
    if code == "??":
        if resolved.is_dir() and not resolved.is_symlink():
            shutil.rmtree(resolved, ignore_errors=True)
        else:
            try:
                resolved.unlink()
            except OSError:
                skipped.append(path)
                continue
        removed.append(path)
    else:
        subprocess.run(
            ["git", "-C", str(root), "checkout", "--", path],
            capture_output=True,
            check=False,
        )
        reverted.append(path)

if removed or reverted or skipped:
    print(
        f"timing run artifacts cleaned: removed={removed} "
        f"reverted={reverted} skipped={skipped}"
    )
PY
}

# Run one side PERF_REPEAT times. Results go to globals rather than a packed string because
# the log list is variable-length and re-splitting it on the caller side is how paths with
# awkward characters get mangled.
PERF_RUN_LOGS=()
PERF_RUN_RC=0
run_perf_repeats() {
  local phase="$1"
  local index result rc
  PERF_RUN_LOGS=()
  PERF_RUN_RC=0
  for ((index = 1; index <= PERF_REPEAT; index++)); do
    result=$(run_perf "$phase-perf-$index")
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
      && env "${environment[@]}" timeout "$PERF_TIMEOUT" \
        "$TARGET_PYTHON" "$TEST_FILE" "${extra[@]}"
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
    python3 - "$result" <<'PY'
import json
import sys

result = int(sys.argv[1])
print(json.dumps({
    "tests": 1,
    "failures": int(result != 0),
    "errors": 0,
    "skipped": 0,
    "executed": 1,
    "basis": "script process exit",
}))
PY
  elif [ -f "$junit" ]; then
    python3 "$SCRIPT_DIR/validate_evidence.py" pytest-stats "$junit"
  else
    printf '%s\n' \
      '{"tests":0,"failures":0,"errors":1,"skipped":0,"executed":0,"note":"JUnit XML missing"}'
  fi
}

stats_field() {
  python3 - "$1" "$2" <<'PY'
import json
import sys

print(json.loads(sys.argv[1])[sys.argv[2]])
PY
}

# ---------- does this target actually need a GPU? ----------
# Asked of the target, not inferred from the diff. A diff heuristic cannot settle this:
# a Python-level dispatch change reroutes kernels without touching kernel source, and
# ROCm/aiter#5089 decides whether 34 gfx950 kernels compile from a 7-line helper. Here
# PICK is empty, so run_pytest already exports HIP_VISIBLE_DEVICES="" and the target
# runs with no visible device -- passing there is an observation, not a guess.
GPU_REQUIREMENT="required"
GPU_REQUIREMENT_BASIS="a GPU was claimed, so the requirement was not probed"
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
PERF_BASE_LOG=""
PERF_BASE_LOGS=()
PERF_HEAD_RC=""
PERF_HEAD_LOG=""
PERF_HEAD_LOGS=()
PERF_SKIP_REASON=""
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
        if [ "$BASE_REPO_STATE" = "target-not-present" ]; then
          PERF_SKIP_REASON="the PR adds this target, so base has nothing to time against"
        elif [ "$PERF_ARGS_SET" -eq 1 ] || perf_detect; then
          perf_snapshot base
          run_perf_repeats base
          PERF_BASE_RC=$PERF_RUN_RC
          PERF_BASE_LOGS=("${PERF_RUN_LOGS[@]}")
          PERF_BASE_LOG="${PERF_BASE_LOGS[0]}"
          perf_restore base
        else
          PERF_SKIP_REASON="$PERF_BASIS"
        fi
      fi
      if [ "$GRID_HOOK_OK" -eq 1 ]; then
        if [ -f "$REPO_WT/$TEST_FILE" ]; then
          BASE_PROBE_RESULT=$(run_pytest \
            "base-grid-probe" "$SHAPE_ENV=__VALIDATOR_INVALID_GRID__")
          BASE_PROBE_RC=${BASE_PROBE_RESULT%%|*}
          BASE_PROBE_LOG=${BASE_PROBE_RESULT##*|}
          if [ "$BASE_PROBE_RC" -eq 0 ]; then
            BASE_GRID_STATE="hook-not-consumed"
          else
            BASE_GRID_RESULT=$(run_pytest "base-grid" "$SHAPE_ENV=$GRID")
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
      elif [ -n "$SHAPE_ENV" ] && [ -n "$GRID" ]; then
        BASE_GRID_STATE="hook-not-found"
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
    fi

    # Head's timing run pairs with the base one and is skipped outright when base produced
    # nothing: a head-only number reproduces the PR's own comparison and cannot show a
    # regression, which is the single thing this stage is for.
    if [ "$PERF_ENABLED" -eq 1 ] && [ -n "$PERF_BASE_LOG" ]; then
      perf_snapshot head
      run_perf_repeats head
      PERF_HEAD_RC=$PERF_RUN_RC
      PERF_HEAD_LOGS=("${PERF_RUN_LOGS[@]}")
      PERF_HEAD_LOG="${PERF_HEAD_LOGS[0]}"
      # Symmetric with base: the grid run and the caller's worktree both follow this point,
      # and neither should inherit a results file the timing run happened to drop.
      perf_restore head
    fi

    if [ "$GRID_HOOK_OK" -eq 1 ]; then
      HEAD_PROBE_RESULT=$(run_pytest \
        "head-grid-probe" "$SHAPE_ENV=__VALIDATOR_INVALID_GRID__")
      HEAD_PROBE_RC=${HEAD_PROBE_RESULT%%|*}
      HEAD_PROBE_LOG=${HEAD_PROBE_RESULT##*|}
      if [ "$HEAD_PROBE_RC" -eq 0 ]; then
        stage_note "correctness_s1_grid" "skip" \
          "target ignores the shape environment variable at runtime"
        stage_note "execution_receipt" "skip" \
          "shape environment runtime handshake failed"
        jset_json "stages.correctness_s1_grid.hook_probe_exit" "$HEAD_PROBE_RC"
        jset_string "stages.correctness_s1_grid.hook_probe_log" "$HEAD_PROBE_LOG"
        finding "note" "correctness" \
          "the selected target passes an invalid shape-grid probe, so grid consumption is unproven"
      else
        HEAD_GRID_RESULT=$(run_pytest "head-grid" "$SHAPE_ENV=$GRID")
        HEAD_GRID_RC=${HEAD_GRID_RESULT%%|*}
        HEAD_GRID_LOG=${HEAD_GRID_RESULT##*|}
        HEAD_GRID_STATS=$(target_stats "head-grid" "$HEAD_GRID_RC")
        HEAD_GRID_EXECUTED=$(stats_field "$HEAD_GRID_STATS" executed)
        python3 - "$JSON" "$HEAD_GRID_RC" "$GRID" "$HEAD_GRID_LOG" \
          "$HEAD_GRID_STATS" "$HEAD_PROBE_RC" "$HEAD_PROBE_LOG" <<'PY'
import json
import sys

path, exit_code, grid, log, raw_stats, probe_exit, probe_log = sys.argv[1:8]
data = json.load(open(path))
stats = json.loads(raw_stats)
status = "fail" if int(exit_code) else ("pass" if stats["executed"] else "skip")
data["stages"]["correctness_s1_grid"] = {
    "status": status,
    "exit": int(exit_code),
    "grid": grid,
    "log": log,
    "stats": stats,
    "hook_probe_exit": int(probe_exit),
    "hook_probe_log": probe_log,
}
if status == "skip":
    data["stages"]["correctness_s1_grid"]["note"] = (
        "shape-grid target completed with no executed tests"
    )
json.dump(data, open(path, "w"), indent=2)
PY
        mark_runtime_coverage "$HEAD_GRID_STATS" "$TARGET_RUNNER" "$HEAD_GRID_LOG"
        if [ "$HEAD_GRID_RC" -eq 0 ] && [ "$HEAD_GRID_EXECUTED" -eq 0 ]; then
          finding "note" "correctness" \
            "shape-grid target executed no tests; no grid claim is made"
        elif [ "$HEAD_GRID_RC" -ne 0 ]; then
          GRID_EXCERPT=$(log_excerpt "$HEAD_GRID_LOG")
          if [ -z "$PATCHF" ]; then
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
        RECEIPT_JSON=$(
          python3 "$SCRIPT_DIR/validate_evidence.py" receipt \
            "$WORK/head/execution-receipt.json" \
            --expected-route "$EXPECTED_ROUTE" --grid "$GRID"
        )
        jset_json "stages.execution_receipt" "$RECEIPT_JSON"
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
      if [ -n "$EXPECTED_ROUTE" ] && [ -f "$WORK/head/execution-receipt.json" ]; then
        RECEIPT_JSON=$(
          python3 "$SCRIPT_DIR/validate_evidence.py" receipt \
            "$WORK/head/execution-receipt.json" \
            --expected-route "$EXPECTED_ROUTE" --grid ""
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
          "shape-grid hook was not established and no route was supplied"
      fi
      finding "note" "correctness" \
        "the selected target does not consume the configured shape-grid hook"
    else
      stage_note "correctness_s1_grid" "skip" \
        "kernel exposes no configured shape override; coverage is repo-default-only"
      if [ -n "$EXPECTED_ROUTE" ] && [ -f "$WORK/head/execution-receipt.json" ]; then
        RECEIPT_JSON=$(
          python3 "$SCRIPT_DIR/validate_evidence.py" receipt \
            "$WORK/head/execution-receipt.json" \
            --expected-route "$EXPECTED_ROUTE" --grid ""
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
  SCAN_JSON=$("$SCANNER" --diff "$PATCHF" --json 2>"$WORK/index-width-scan.log")
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
# the head phase. It is deliberately NOT in finish_report's required-stage set: `complete` is
# computed from the nine correctness stages, so a perf run that could not happen downgrades
# nothing and a PASS stays a PASS. What it can do is append a should-fix finding, which
# finish_report turns into NEEDS_WORK and exit 1 -- a measured regression is a real result,
# not an advisory note, and the whole point of putting it in the deterministic layer is that
# it ships its own reproducer (both logs, both exit codes, the command) with it.
#
# Every path that is not "both sides ran clean and the numbers disagree" reports `skip`.
# A timeout, a crash, a missing harness and a one-row table must never be able to look like
# a regression, because a false regression here blocks a good PR and would get the stage
# switched off within a week.
if [ "$PERF_ENABLED" -ne 1 ]; then
  stage_note "perf" "skip" "perf measurement was disabled with --no-perf"
elif [ -n "$PERF_SKIP_REASON" ]; then
  stage_note "perf" "skip" "$PERF_SKIP_REASON"
  finding "note" "perf" \
    "no base-vs-head timing was taken: $PERF_SKIP_REASON"
elif [ -z "$PERF_BASE_LOG" ] || [ -z "$PERF_HEAD_LOG" ]; then
  PERF_WHY="the run did not reach both a baseline and a head phase"
  [ "$CAN_TEST" -eq 0 ] && PERF_WHY="${SKIP_REASON:-$PERF_WHY}"
  stage_note "perf" "skip" "$PERF_WHY"
  finding "note" "perf" "no base-vs-head timing was taken: $PERF_WHY"
elif [ "$PERF_BASE_RC" -ne 0 ] || [ "$PERF_HEAD_RC" -ne 0 ]; then
  # Deliberately not a regression. A nonzero exit means the log is truncated at an unknown
  # point, so any ratio drawn from it compares whatever happened to print before the crash.
  PERF_WHY="benchmark run exited nonzero (base=$PERF_BASE_RC head=$PERF_HEAD_RC); timings from a truncated run are not comparable"
  stage_note "perf" "skip" "$PERF_WHY"
  jset_string "stages.perf.base_log" "$PERF_BASE_LOG"
  jset_string "stages.perf.head_log" "$PERF_HEAD_LOG"
  finding "note" "perf" "$PERF_WHY"
else
  PERF_JSON="$WORK/perf-compare.json"
  "$SCRIPT_DIR/scrape_perf.py" \
    --base "${PERF_BASE_LOGS[@]}" --head "${PERF_HEAD_LOGS[@]}" \
    --threshold "$PERF_THRESHOLD" --min-rows "$PERF_MIN_ROWS" \
    --out "$PERF_JSON" >/dev/null 2>"$WORK/perf-compare.err"
  PERF_CMP_RC=$?
  if [ "$PERF_CMP_RC" -ne 0 ] || [ ! -r "$PERF_JSON" ]; then
    stage_note "perf" "skip" \
      "the benchmark comparison failed: $(log_excerpt "$WORK/perf-compare.err")"
    finding "note" "perf" \
      "base and head both produced benchmark logs, but they could not be compared"
  else
    python3 - "$JSON" "$PERF_JSON" "$PERF_BASE_LOG" "$PERF_HEAD_LOG" \
      "$BASE_SHA" "$PERF_ARGS" "$PERF_BASIS" <<'PY'
import json
import sys

report_path, compare_path, base_log, head_log, base_sha, command, basis = sys.argv[1:8]
data = json.load(open(report_path))
result = json.load(open(compare_path))

stage = {
    "status": {"regression": "fail", "ok": "pass"}.get(result["status"], "skip"),
    "baseline": f"{base_sha} with the candidate patch reversed, same worktree and GPU",
    "command": command or "(target's default entry point)",
    "harness": basis,
    "threshold": result.get("threshold"),
    "matched_rows": result.get("matched_rows", 0),
    "columns": result.get("columns", {}),
    # Repeat count is part of the claim, not trivia: the threshold is only defensible
    # because each cell is a best-of-N, so a reader has to be able to see N.
    "repeats": {
        "base": result.get("base_runs", 1),
        "head": result.get("head_runs", 1),
        "reduction": "best sample per cell (min latency / max throughput)",
    },
    "base_log": base_log,
    "head_log": head_log,
    "note": result.get("reason") or "",
}
# median_ratio is omitted, never nulled, when there is no measurement: report_schema.json
# types it as a number, and a null would fail validation at review-pr's identity gate --
# turning "we could not measure" into "this report is malformed".
if result.get("median_ratio") is not None:
    stage["median_ratio"] = result["median_ratio"]
if result.get("worst_column"):
    stage["worst_column"] = result["worst_column"]
if result.get("regressed_rows"):
    stage["regressed_rows"] = result["regressed_rows"]
data["stages"]["perf"] = stage

if result["status"] == "regression":
    rows = ", ".join(
        f"{row['row']}: {row['base']:g} -> {row['head']:g}"
        for row in result.get("regressed_rows", [])[:3]
    )
    data["findings"].append(
        {
            "severity": "should-fix",
            "stage": "perf",
            "detail": (
                "head is slower than base on the same locked GPU -- "
                + result["reason"]
                + (f"; worst rows: {rows}" if rows else "")
            ),
        }
    )
elif result["status"] == "insufficient":
    data["findings"].append(
        {
            "severity": "note",
            "stage": "perf",
            "detail": f"no perf comparison was made: {result['reason']}",
        }
    )
json.dump(data, open(report_path, "w"), indent=2)
PY
  fi
fi

record_gpu_activity_after
finish_report
FINAL_VERDICT=$(python3 - "$OUT" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1]))["verdict"])
PY
)
case "$FINAL_VERDICT" in
  PASS) exit 0;;
  BLOCK|NEEDS_WORK) exit 1;;
  INCONCLUSIVE) exit 2;;
  *) exit 2;;
esac
