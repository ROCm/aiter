"""Aggregate rocprofv3 per-XCC counter CSV into per-XCD balance metrics."""
import csv
import glob
import sys
from collections import defaultdict

path = sys.argv[1]
csvf = glob.glob(f"{path}/**/*counter_collection.csv", recursive=True)
if not csvf:
    print(f"no counter CSV under {path}")
    sys.exit(1)

# counter -> list of per-dispatch values
vals = defaultdict(list)
with open(csvf[0]) as f:
    for row in csv.DictReader(f):
        vals[row["Counter_Name"]].append(float(row["Counter_Value"]))


def per_xcd(prefix):
    out = []
    for x in range(8):
        v = vals.get(f"{prefix}{x}", [])
        out.append(sum(v) / len(v) if v else 0.0)
    return out


def stats(name, arr):
    mx, mn = max(arr), min(arr)
    avg = sum(arr) / len(arr)
    cv = (sum((a - avg) ** 2 for a in arr) / len(arr)) ** 0.5 / avg * 100 if avg else 0
    print(f"\n{name} per XCD:")
    print("  " + "  ".join(f"X{i}={a:,.0f}" for i, a in enumerate(arr)))
    print(f"  min={mn:,.0f} max={mx:,.0f} avg={avg:,.0f}")
    print(f"  max/min={mx / mn:.3f}x   spread(max-min)/max={100 * (mx - mn) / mx:.1f}%   CV={cv:.1f}%")
    return mx, mn, avg


busy = per_xcd("SQ_BUSY_XCC")
waves = per_xcd("SQ_WAVES_XCC")
print(f"=== {path} ===")
bmx, bmn, bavg = stats("SQ_BUSY_CYCLES (wall time an XCD had >=1 active wave)", busy)
stats("SQ_WAVES (workgroups dispatched)", waves)
# Tail model: if every XCD started together, the run ends when the busiest XCD
# finishes. Wasted = how much sooner a perfectly balanced run could finish.
print(f"\n  -> busy-time imbalance tail: busiest XCD runs {100 * (bmx - bavg) / bavg:.1f}% "
      f"longer than the average XCD")
