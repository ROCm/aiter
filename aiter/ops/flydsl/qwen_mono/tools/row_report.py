import json, glob, math, sys
n_ = sys.argv[1]; tot = None; acc = None
for f in sorted(glob.glob("/mnt/data/vigokhal/qwen35_runs/%s.acc.[0-9].tile.json" % n_)):
    a = json.load(open(f))
    if tot is None: tot = a
    else:
        for k, v in a.items(): tot[k] = max(tot[k], v) if k == "maxel" else ([x + y for x, y in zip(tot[k], v)] if isinstance(v, list) else tot[k] + v)
for f in sorted(glob.glob("/mnt/data/vigokhal/qwen35_runs/%s.acc.[0-9].json" % n_)):
    a = json.load(open(f)); acc = a if acc is None else [[x + y for x, y in zip(r, s)] for r, s in zip(acc, a)]
L = acc[:40]; A = lambda k: sum(r[k] for r in L); q = lambda a, b: math.sqrt(a / b) if b and a == a else float("nan")
calls = A(2)
print("%s: calls %d, rows %d | total rel %.3f routed %.3f shared %.3f | rows off > 50%% %d, > 200%% %d | routed rows off > 100%% %d | bad tiles %d in %d rows, %d calls (%.2f per 100k calls) | top tiles %s | by rows a call %s" % (
    n_, calls, A(3), q(A(0), A(1)), q(A(12), A(13)), q(A(14), A(15)), A(8), sum(acc[40 + T][9] for T in range(1, 17)), sum(acc[40 + T][11] for T in range(1, 17)),
    tot["n"], tot["rows_with"], tot["calls_with"], 1e5 * tot["calls_with"] / calls, sorted(((int(v), i) for i, v in enumerate(tot["tile"]) if v), reverse=True)[:4], {i: int(v) for i, v in enumerate(tot["T"]) if v}))
