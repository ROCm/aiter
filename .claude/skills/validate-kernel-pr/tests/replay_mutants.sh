#!/usr/bin/env bash
# Replays the pinned FlyDSL softmax control and three seeded defects.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SKILL_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
MANIFEST="$SCRIPT_DIR/mutants/manifest.json"
FLYDSL_REPO="${1:-${FLYDSL_REPO:-}}"
OUT_DIR="${OUT_DIR:-$PWD/validation-mutant-reports}"
CASE_FILTER="${CASE_FILTER:-}"

if [ -z "$FLYDSL_REPO" ]; then
  echo "usage: bash replay_mutants.sh /path/to/FlyDSL" >&2
  exit 2
fi
if ! git -C "$FLYDSL_REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "not a FlyDSL git checkout: $FLYDSL_REPO" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
BASE=$(python3 - "$MANIFEST" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1]))["base"])
PY
)
mapfile -t CASES < <(python3 - "$MANIFEST" <<'PY'
import json
import sys

for case in json.load(open(sys.argv[1]))["cases"]:
    print(f"{case['id']}|{case['patch']}")
PY
)
TESTS=$(python3 - "$MANIFEST" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1]))["tests"])
PY
)
SHAPE_ENV=$(python3 - "$MANIFEST" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1]))["shape_env"])
PY
)
GRID=$(python3 - "$MANIFEST" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1]))["grid"])
PY
)
TOLERANCES=$(python3 - "$MANIFEST" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1]))["tolerance_table"])
PY
)

ACTIVE_WORKTREE=""
ACTIVE_PATCH=""
cleanup() {
  if [ -n "$ACTIVE_WORKTREE" ] && [ -d "$ACTIVE_WORKTREE" ]; then
    if [ -n "$ACTIVE_PATCH" ]; then
      git -C "$ACTIVE_WORKTREE" apply -R "$ACTIVE_PATCH" >/dev/null 2>&1 || true
    fi
    git -C "$FLYDSL_REPO" worktree remove "$ACTIVE_WORKTREE" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

for entry in "${CASES[@]}"; do
  case_id=${entry%%|*}
  patch_name=${entry##*|}
  if [ -n "$CASE_FILTER" ] && [ "$case_id" != "$CASE_FILTER" ]; then
    continue
  fi
  ACTIVE_PATCH="$SCRIPT_DIR/mutants/$patch_name"
  case_root=$(mktemp -d "/tmp/validate-mutant-$case_id-XXXXXX")
  ACTIVE_WORKTREE="$case_root/worktree"
  git -C "$FLYDSL_REPO" worktree add --detach "$ACTIVE_WORKTREE" "$BASE" >/dev/null

  echo "== $case_id =="
  rm -f "$OUT_DIR/$case_id.json" "$OUT_DIR/$case_id.exit"
  set +e
  "$SKILL_DIR/validate_pr.sh" \
    --repo "$ACTIVE_WORKTREE" \
    --patch "$ACTIVE_PATCH" \
    --target "$TESTS" \
    --expected-route kernels.norm.softmax_kernel:build_softmax_module \
    --shape-vars M,N,dtype_str \
    --shape-env "$SHAPE_ENV" \
    --grid "$GRID" \
    --tol-table "$TOLERANCES" \
    --label "$case_id" \
    --out "$OUT_DIR/$case_id.json"
  validator_rc=$?
  set -e
  printf '%s\n' "$validator_rc" > "$OUT_DIR/$case_id.exit"

  # validate_pr.sh hands the worktree back in the state it was given, patch already
  # reversed. An unconditional reverse-apply here therefore fails with "patch does not
  # apply" on EVERY case, and under `set -e` that aborts the driver before the summary
  # block that checks the verdicts -- so the mutant suite could never report itself green,
  # however well the mutants were discriminated. Reverse only if the patch is still applied.
  if git -C "$ACTIVE_WORKTREE" apply -R --check "$ACTIVE_PATCH" >/dev/null 2>&1; then
    git -C "$ACTIVE_WORKTREE" apply -R "$ACTIVE_PATCH"
  fi
  ACTIVE_PATCH=""
  git -C "$FLYDSL_REPO" worktree remove "$ACTIVE_WORKTREE"
  ACTIVE_WORKTREE=""
  rmdir "$case_root"
  echo "validator_exit=$validator_rc"
done

python3 - "$MANIFEST" "$OUT_DIR" "$CASE_FILTER" <<'PY'
import json
import pathlib
import sys

manifest = json.load(open(sys.argv[1]))
report_dir = pathlib.Path(sys.argv[2])
case_filter = sys.argv[3]
failed = []
for case in manifest["cases"]:
    if case_filter and case["id"] != case_filter:
        continue
    report = json.loads((report_dir / f"{case['id']}.json").read_text())
    expected_exit = 0 if report["verdict"] == "PASS" else (2 if report["verdict"] == "INCONCLUSIVE" else 1)
    actual_exit = int((report_dir / f"{case['id']}.exit").read_text())
    if actual_exit != expected_exit or report["process_exit_code"] != expected_exit:
        failed.append(
            f"{case['id']}: actual/report exits {actual_exit}/{report['process_exit_code']} != {expected_exit}"
        )
    if report["verdict"] != case["expected_verdict"]:
        failed.append(
            f"{case['id']}: expected {case['expected_verdict']}, "
            f"got {report['verdict']}"
        )
        continue
    expected_stage = case["expected_stage"]
    if expected_stage and not any(
        finding["stage"] == expected_stage
        and finding["severity"] == "blocker"
        for finding in report["findings"]
    ):
        failed.append(f"{case['id']}: no blocker from {expected_stage}")
if failed:
    raise SystemExit("\n".join(failed))
print("mutant replay matched all expected verdicts")
PY
