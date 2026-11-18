import json, glob, sys
n_ = sys.argv[1]; tot = None
for f in sorted(glob.glob("/mnt/data/vigokhal/qwen35_runs/%s.acc.[0-9].tile.json" % n_)):
    a = json.load(open(f))
    if tot is None: tot = a
    else:
        for k, v in a.items():
            tot[k] = max(tot[k], v) if k == "maxel" else ([x + y for x, y in zip(tot[k], v)] if isinstance(v, list) else tot[k] + v)
acc = None
for f in sorted(glob.glob("/mnt/data/vigokhal/qwen35_runs/%s.acc.[0-9].json" % n_)):
    a = json.load(open(f)); acc = a if acc is None else [[x + y for x, y in zip(r, s)] for r, s in zip(acc, a)]
rows = sum(r[3] for r in acc[:40]); calls = sum(r[2] for r in acc[:40])
print("rows %d, calls %d | bad tiles %d in %d rows, %d calls (%.4f%% of calls) | worst element error / row max %.3f" % (rows, calls, tot["n"], tot["rows_with"], tot["calls_with"], 100 * tot["calls_with"] / calls, tot["maxel"]))
t = tot["tile"]; print("bad tiles by tile index: top", sorted(((int(v), i) for i, v in enumerate(t) if v), reverse=True)[:8], "| by rows a call:", {i: int(v) for i, v in enumerate(tot["T"]) if v})
