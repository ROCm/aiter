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
# usage: validate_pr.sh --repo <worktree> --tests <pytest target> [--patch p.patch]
#                       [--shape-env VAR] [--grid "M,N,dt;..."] [--tol-table f32=1e-5,...]
#                       [--label NAME] [--out report.json]
set -u
REPO_WT=""; TESTS=""; PATCHF=""; SHAPE_ENV=""; GRID=""; TOL_TABLE=""; LABEL="run"; OUT=""
PYLIB="${PYLIB:-}"
while [ $# -gt 0 ]; do
  case "$1" in
    --repo) REPO_WT="$2"; shift 2;;
    --tests) TESTS="$2"; shift 2;;
    --patch) PATCHF="$2"; shift 2;;
    --shape-env) SHAPE_ENV="$2"; shift 2;;
    --grid) GRID="$2"; shift 2;;
    --tol-table) TOL_TABLE="$2"; shift 2;;
    --label) LABEL="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac
done
: "${OUT:=$PWD/validation_report.json}"
mkdir -p "$(dirname "$OUT")"
WORK=$(mktemp -d "/tmp/s1-$LABEL-XXXX")
JSON="$WORK/r.json"
python3 -c "import json,sys;json.dump({'label':sys.argv[1],'stages':{},'findings':[]},open(sys.argv[2],'w'))" "$LABEL" "$JSON"

jset() { python3 - "$JSON" "$1" "$2" <<'PY'
import json,sys
p,k,v=sys.argv[1],sys.argv[2],sys.argv[3]
d=json.load(open(p)); cur=d
ks=k.split(".")
for kk in ks[:-1]: cur=cur.setdefault(kk,{})
try: v=json.loads(v)
except Exception: pass
cur[ks[-1]]=v
json.dump(d,open(p,"w"),indent=1)
PY
}
finding() { python3 - "$JSON" "$1" "$2" "$3" <<'PY'
import json,sys
p,sev,stage,msg=sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]
d=json.load(open(p)); d["findings"].append({"severity":sev,"stage":stage,"detail":msg})
json.dump(d,open(p,"w"),indent=1)
PY
}

echo "=== S1 validate-kernel-pr [$LABEL] ==="
jset "started_utc" "\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\""
jset "isolation" '{"level":"git-worktree + private caches","container":false,"reason":"no docker CLI available inside this container"}'

# ---------- stage 1: merge simulation ----------
BASE_SHA=$(git -C "$REPO_WT" rev-parse HEAD 2>/dev/null)
jset "repo.worktree" "\"$REPO_WT\""; jset "repo.head" "\"$BASE_SHA\""
if [ -n "$PATCHF" ]; then
  if git -C "$REPO_WT" apply --check "$PATCHF" 2>/dev/null; then
    git -C "$REPO_WT" apply "$PATCHF"; jset "stages.merge_sim" '{"status":"pass","note":"patch applies cleanly"}'
  else
    jset "stages.merge_sim" '{"status":"fail","note":"patch does not apply"}'
    finding "blocker" "merge_sim" "patch/PR does not apply to the current base"
    cp "$JSON" "$OUT"; echo "MERGE CONFLICT -> $OUT"; exit 1
  fi
else
  jset "stages.merge_sim" '{"status":"skip","note":"no patch supplied (base validation run)"}'
fi

# ---------- stage 2: GPU claim (sampling window + lock) ----------
PICKER="${PICKER:-$(command -v pick-idle-gpu.py || true)}"
if [ -z "$PICKER" ]; then
  for c in "$HOME/.local/bin/pick-idle-gpu.py" /usr/local/bin/pick-idle-gpu.py /opt/bin/pick-idle-gpu.py; do
    [ -x "$c" ] && PICKER="$c" && break
  done
fi
  PICK=$([ -n "$PICKER" ] && "$PICKER" --samples 10 --interval 1 --quiet 2>/dev/null || true)
if [ -z "$PICK" ]; then
  jset "stages.gpu_claim" '{"status":"fail","note":"no idle GPU after sampling window"}'
  jset "degraded_mode" '"COMPILE_ONLY"'
  finding "note" "gpu_claim" "no idle GPU -- correctness/perf degraded to compile-only"
else
  MKT=$(amd-smi static -g "$PICK" 2>/dev/null | grep -m1 MARKET_NAME | awk '{print $2, $3, $4}')
  ARCH=$(amd-smi static -g "$PICK" 2>/dev/null | grep -m1 TARGET_GRAPHICS_VERSION | awk '{print $2}')
  BDF=$(amd-smi static -g "$PICK" 2>/dev/null | grep -m1 BDF | awk '{print $2}')
  ACT0=$(amd-smi metric -g "$PICK" 2>/dev/null | grep -m1 GFX_ACTIVITY | awk '{print $2}')
  jset "stages.gpu_claim" "{\"status\":\"pass\",\"hip_index\":$PICK,\"model\":\"$MKT\",\"arch\":\"$ARCH\",\"bdf\":\"$BDF\",\"gfx_activity_before_pct\":\"$ACT0\",\"host\":\"$(hostname)\"}"
  # arch coverage is a fact about this host, not an aspiration
  jset "arch_coverage" "{\"$ARCH\":\"runtime\",\"gfx942\":\"compile-only (no gfx942 device on this host)\"}"
fi

# ---------- stage 3: runtime compatibility ----------
# A pinned prebuilt runtime can be older than the checkout. Say so instead of
# reporting the resulting ImportError as a code failure.
RC_OUT=$(cd "$REPO_WT" && PYTHONPATH="$PYLIB:$REPO_WT" timeout 300 python3 -c "
import importlib,sys
sys.path.insert(0,'.')
import flydsl; print('runtime', getattr(flydsl,'__version__','?'))
" 2>&1 | tail -2)
if echo "$RC_OUT" | grep -qi "error\|Traceback"; then
  jset "stages.runtime_compat" "{\"status\":\"fail\",\"detail\":\"$(echo "$RC_OUT" | tail -1 | tr -d '"' | cut -c1-180)\"}"
  finding "blocker" "runtime_compat" "checkout does not import against the pinned runtime -- rebuild the runtime image before trusting any test result"
else
  jset "stages.runtime_compat" "{\"status\":\"pass\",\"detail\":\"$(echo "$RC_OUT" | tail -1 | tr -d '"')\"}"
fi

# ---------- stage 4: test-policy check (runs BEFORE the suite) ----------
# A suite that cannot fail is worse than no suite: it produces a green report.
if [ -n "$TOL_TABLE" ]; then
  python3 - "$JSON" "$REPO_WT" "$TESTS" "$TOL_TABLE" <<'PY'
import json,re,subprocess,sys,os
jp,wt,tests,tbl=sys.argv[1:5]
rel=tests.split("::")[0]
path=os.path.join(wt,rel)
cur=open(path).read() if os.path.exists(path) else ""
# The question is not "is this tolerance small enough in the abstract" (repos
# legitimately differ per kernel) but "did THIS change loosen what was there".
base=subprocess.run(["git","-C",wt,"show",f"HEAD:{rel}"],capture_output=True,text=True).stdout
def tols(src):
    a=[float(m) for m in re.findall(r"(?:atol|rtol)\s*=\s*([0-9.eE+-]+)", src)]
    b=[float(m) for m in re.findall(r'"(?:f32|f16|bf16)"\s*:\s*([0-9.eE+-]+)', src)]
    return a+b
cur_t, base_t = tols(cur), tols(base)
loosened=[]
if base_t and cur_t and len(cur_t)==len(base_t):
    loosened=[(b,c) for b,c in zip(base_t,cur_t) if c>b]
commented=len(re.findall(r'^\s*#\s*\(\s*\d+\s*,\s*\d+\s*,\s*"(?:f32|f16|bf16)"\s*\)', cur, re.M))
d=json.load(open(jp))
st={"status":"pass","tolerances_base":base_t,"tolerances_head":cur_t,"commented_out_shape_rows":commented}
if loosened:
    st["status"]="fail"; st["loosened"]=loosened
    d["findings"].append({"severity":"blocker","stage":"test_policy",
      "detail":f"this change loosens comparison tolerance {loosened} (base -> head) while leaving the kernel path it guards unchanged -- the suite can no longer fail on the regression it was written to catch"})
if commented:
    d["findings"].append({"severity":"should-fix","stage":"test_policy",
      "detail":f"{commented} shape rows are commented out in the test config -- the suite exercises fewer paths than it appears to; the S1 shape grid covers them instead"})
d["stages"]["test_policy"]=st
json.dump(d,open(jp,"w"),indent=1)
PY
fi

# ---------- stage 5: correctness ----------
run_pytest() { # $1 = label, $2 = extra env assignment
  local lbl="$1"
  local envassign="$2"
  local log="$WORK/pytest-$lbl.log"
  local rc
  ( cd "$REPO_WT" && env HIP_VISIBLE_DEVICES="${PICK:-0}" PYTHONPATH="$PYLIB" \
      HOME="$WORK/home" FLYDSL_CACHE_DIR="$WORK/flydsl-cache" $envassign \
      timeout 1800 flock "/tmp/gpu-${PICK:-0}.lock" python -m pytest $TESTS -x -q ) > "$log" 2>&1
  rc=$?
  echo "$rc|$log"
}
mkdir -p "$WORK/home" "$WORK/flydsl-cache"
# An unusable test runner is an environment fact, not a defect in the PR. Say so, rather
# than reporting "the PR's own tests fail" for a missing pytest.
if ! ( cd "$REPO_WT" && PYTHONPATH="$PYLIB" python -m pytest --version ) >/dev/null 2>&1; then
  jset "stages.correctness_repo_tests" '"'"'{"status":"skip","note":"pytest not runnable in this environment - correctness not attempted"}'"'"'
  jset "stages.correctness_s1_grid" '"'"'{"status":"skip","note":"pytest not runnable in this environment"}'"'"'
  finding "note" "correctness" "test runner unavailable (python -m pytest failed to start) - this report makes no correctness claim"
  PICK=""
fi
if [ -n "${PICK:-}" ]; then
  # Baseline control: with a patch applied, a failure only belongs to the PR if the SAME
  # target passes without it. Without this the harness charges pre-existing red to the
  # author -- the exact misattribution this skill exists to prevent. Learned from an
  # end-to-end run where base and head failed the same unrelated shape.
  BASE_RC=""
  if [ -n "$PATCHF" ]; then
    git -C "$REPO_WT" stash -q 2>/dev/null
    R0=$(run_pytest base ""); BASE_RC=${R0%%|*}; BASE_LOG=${R0##*|}
    git -C "$REPO_WT" stash pop -q 2>/dev/null
    jset "stages.baseline_control" "{\"status\":\"$([ "$BASE_RC" -eq 0 ] && echo clean || echo "pre-existing-failures")\",\"exit\":$BASE_RC,\"log\":\"$BASE_LOG\"}"
  else
    jset "stages.baseline_control" '"'"'{"status":"skip","note":"no patch: this run IS a base measurement, failures are not attributable to any PR"}'"'"'
  fi

  R=$(run_pytest repo ""); RC=${R%%|*}; LOG=${R##*|}
  jset "stages.correctness_repo_tests" "{\"status\":\"$([ $RC -eq 0 ] && echo pass || echo fail)\",\"exit\":$RC,\"log\":\"$LOG\"}"
  if [ $RC -ne 0 ]; then
    if [ -n "$PATCHF" ] && [ "${BASE_RC:-0}" -ne 0 ]; then
      finding "note" "correctness" "test target is red BOTH with and without this change -- pre-existing, not attributable to the PR: $(tail -3 "$LOG" | tr -d '"' | tr '\n' ' ' | cut -c1-150)"
    else
      finding "blocker" "correctness" "the PR's own test target fails (base is clean): $(tail -3 "$LOG" | tr -d '"' | tr '\n' ' ' | cut -c1-180)"
    fi
  fi

  # S1-owned shape grid: non-toy, boundary/odd, large-M. Independent of what the PR chose to test.
  if [ -n "$SHAPE_ENV" ] && [ -n "$GRID" ]; then
    R=$(run_pytest grid "$SHAPE_ENV=$GRID"); RC2=${R%%|*}; LOG2=${R##*|}
    jset "stages.correctness_s1_grid" "{\"status\":\"$([ $RC2 -eq 0 ] && echo pass || echo fail)\",\"exit\":$RC2,\"grid\":\"$GRID\",\"log\":\"$LOG2\"}"
    [ $RC2 -ne 0 ] && finding "blocker" "correctness" "S1 shape grid fails where the PR's own tests pass -- grid: $GRID; $(grep -iE 'mismatch|fail|error' "$LOG2" | head -2 | tr -d '"' | tr '\n' ' ' | cut -c1-200)"
  else
    jset "stages.correctness_s1_grid" '{"status":"skip","note":"kernel exposes no shape override; coverage is repo-default-only"}'
    finding "note" "correctness" "no shape-grid hook for this kernel -- coverage claim limited to the repo default shapes"
  fi
else
  jset "stages.correctness_repo_tests" '{"status":"skip","note":"COMPILE_ONLY: no GPU claimed"}'
fi

# ---------- stage 6: index-width scan (informational; feeds the reviewer, not the verdict) ----------
# Rule D9's name-list trigger misses stride/offset multiplies whose operands are not on that
# list. This records the structural candidates so the reviewer has a "check these" list.
if [ -n "$PATCHF" ] && [ -x "$(dirname "$0")/scan_index_width.py" ]; then
  SCAN=$("$(dirname "$0")/scan_index_width.py" --diff "$PATCHF" 2>/dev/null | grep -c "^  " || true)
  jset "stages.index_width_scan" "{\"status\":\"info\",\"candidates\":${SCAN:-0},\"note\":\"index x stride with no 64-bit widening; reviewer must judge each\"}"
  [ "${SCAN:-0}" -gt 0 ] && finding "note" "index_width_scan" "$SCAN index/stride multiply sites carry no 64-bit widening - confirm each cannot exceed 2^31 at production scale"
fi

# ---------- verdict ----------
python3 - "$JSON" "$OUT" <<'PY'
import json,sys,shutil
d=json.load(open(sys.argv[1]))
sev=[f["severity"] for f in d["findings"]]
d["verdict"]="BLOCK" if "blocker" in sev else ("NEEDS WORK" if "should-fix" in sev else "PASS")
d["finished_utc"]=__import__("datetime").datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
json.dump(d,open(sys.argv[1],"w"),indent=1); shutil.copy(sys.argv[1],sys.argv[2])
print(f"verdict={d['verdict']}  findings={len(d['findings'])}  -> {sys.argv[2]}")
for f in d["findings"]: print(f"  [{f['severity']}] {f['stage']}: {f['detail'][:150]}")
PY
