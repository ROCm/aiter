#!/bin/bash
# Run one aiter Triton benchmark with internal and with upstream Triton, alternating on one GPU, and print
# internal vs upstream for every shape and metric. "+" means internal is faster.
#
#   bash op_tests/op_benchmarks/triton/repro_upstream_triton_regression.sh <bench_script.py> [bench args...]
#
# Env: GPU (default 0), REPS (default 3), OUT (default a new /tmp dir),
#      INTERNAL_PY (default python3), UPSTREAM_PY (default /opt/triton-upstream/bin/python3)
set -u
[ $# -ge 1 ] || { sed -n '2,8p' "$0"; exit 2; }
HERE=$(cd "$(dirname "$0")" && pwd)
AITER=$(cd "$HERE/../../.." && pwd)
SCRIPT=$1
shift
[ -f "$HERE/$SCRIPT" ] || { echo "no such benchmark: $HERE/$SCRIPT"; exit 2; }
GPU=${GPU:-0} REPS=${REPS:-3}
INTERNAL_PY=${INTERNAL_PY:-python3}
UPSTREAM_PY=${UPSTREAM_PY:-/opt/triton-upstream/bin/python3}
OUT=${OUT:-$(mktemp -d /tmp/triton_regression.XXXXXX)}
mkdir -p "$OUT"

for v in internal upstream; do
  py=$INTERNAL_PY
  [ $v = upstream ] && py=$UPSTREAM_PY
  echo "$v Triton: $(cd /tmp && $py -c 'import os, triton; print(triton.__version__, os.path.dirname(triton.__file__))' 2>&1 | tail -1)"
done

cat > "$OUT/wrap.py" <<'PY'
# Run a benchmark script the way `python <script> [args]` would, saving each triton.testing.perf_report table as CSV.
import os, runpy, sys
out, script, args = sys.argv[1], os.path.abspath(sys.argv[2]), sys.argv[3:]
os.makedirs(out, exist_ok=True)
os.chdir(out)
import triton.testing as tt
orig, n = tt.Mark._run, [0]
def _run(self, bench, *a, **k):
    df = orig(self, bench, *a, **k)
    n[0] += 1
    df.to_csv(f"table_{n[0]:02d}.csv", index=False)
    return df
tt.Mark._run = _run
sys.argv = [script, *args]
sys.path[0] = os.path.dirname(script)
runpy.run_path(script, run_name="__main__")
PY

for i in $(seq 1 "$REPS"); do
  order="internal upstream"
  [ $((i % 2)) = 0 ] && order="upstream internal"
  for v in $order; do
    py=$INTERNAL_PY
    [ $v = upstream ] && py=$UPSTREAM_PY
    echo "repeat $i: $v"
    env -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=$GPU TRITON_HIP_USE_EXPERT_SCHEDULING=1 \
      TRITON_HIP_USE_COEXEC_SCHEDULER=1 TRITON_CACHE_DIR=$OUT/triton_cache_$v PYTHONPATH=$AITER${PYTHONPATH:+:$PYTHONPATH} \
      $py "$OUT/wrap.py" "$OUT/rep${i}_$v" "$HERE/$SCRIPT" "$@" > "$OUT/rep${i}_$v.log" 2>&1 \
      || { echo "  failed:"; tail -3 "$OUT/rep${i}_$v.log"; }
  done
done

python3 - "$OUT" "$REPS" <<'PY'
import csv, glob, math, os, re, sys
out, reps = sys.argv[1], sys.argv[2]
HIGHER = re.compile(r"tflop|gb/s|tb/s|bandwidth|throughput", re.I)
LOWER = re.compile(r"time|latency|\bms\b|\(ms\)|\bus\b|\(us\)", re.I)
def direction(col):
    m = re.match(r"^(.*) \((.*)\)$", col)
    for part in (m.groups() if m else (col,)):
        if HIGHER.search(part):
            return 1
        if LOWER.search(part):
            return -1
    return 0
data = {}  # (table, shape, metric) -> {variant: {repeat: value}}
for d in glob.glob(os.path.join(out, "rep*_*")):
    if not os.path.isdir(d):
        continue
    rep, v = os.path.basename(d)[3:].split("_")
    for f in sorted(glob.glob(os.path.join(d, "table_*.csv"))):
        rows = list(csv.reader(open(f)))
        if not rows:
            continue
        hdr = rows[0]
        metrics = [i for i, c in enumerate(hdr) if re.search(r" \(.*\)$", c) and direction(c)]
        for r in rows[1:]:
            shape = " ".join(f"{hdr[i]}={r[i]}" for i in range(len(hdr)) if i not in metrics and i < len(r))
            for i in metrics:
                try:
                    x = float(r[i])
                except (ValueError, IndexError):
                    continue
                if x > 0:
                    data.setdefault((os.path.basename(f), shape, hdr[i]), {}).setdefault(v, {})[rep] = x
if not data:
    sys.exit(f"no results: see the rep*_*.log files in {out}")
print(f"\n{'diff':>7s} {'min':>7s} {'max':>7s} {'internal':>11s} {'upstream':>11s}  metric | shape"
      "   (+ = internal faster; min/max over the repeats)")
for (table, shape, metric), byv in sorted(data.items()):
    a, c = byv.get("internal", {}), byv.get("upstream", {})
    both = sorted(set(a) & set(c))
    if not both:
        continue
    d = direction(metric)
    ratios = [a[k] / c[k] if d > 0 else c[k] / a[k] for k in both]
    diff = (math.exp(sum(map(math.log, ratios)) / len(ratios)) - 1) * 100
    per = [(r - 1) * 100 for r in ratios]
    print(f"{diff:+6.1f}% {min(per):+6.1f}% {max(per):+6.1f}% {sum(a[k] for k in both) / len(both):11.4g} "
          f"{sum(c[k] for k in both) / len(both):11.4g}  {metric} | {shape}")
print(f"\n{reps} repeat(s) per Triton; raw results in {out}")
PY
