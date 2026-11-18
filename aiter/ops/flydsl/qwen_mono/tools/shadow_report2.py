import json, glob, math, sys
n_ = sys.argv[1]; tot = None
for f in sorted(glob.glob("/mnt/data/vigokhal/qwen35_runs/%s.acc.[0-9].json" % n_)):
    a = json.load(open(f))
    tot = a if tot is None else [[x + y for x, y in zip(r, s)] for r, s in zip(tot, a)]
L = tot[:40]; A = lambda k: sum(r[k] for r in L); q = lambda a, b: math.sqrt(a / b) if b else float("nan")
print("all: calls a layer %d, rows %d, total rel %.3f, routed %.3f, shared %.3f, gate %.4f, non-finite kernel rows %d" % (L[0][2], A(3), q(A(0), A(1)), q(A(12), A(13)), q(A(14), A(15)), q(A(16), A(17)), A(5)))
h = [A(20 + i) for i in range(6)]; n = sum(h)
print("routed rel error of a row: <2%% %.2f%% | 2-5%% %.2f%% | 5-10%% %.2f%% | 10-20%% %.3f%% | 20-50%% %.4f%% | >50%% %.4f%%  (rows %d)" % tuple([100 * x / n for x in h] + [n]))
print("by rows a call:  T: calls | total rel | routed rows off > 20% | > 100% | total rows off > 50% | > 200% | non-finite")
for T in range(1, 17):
    r = tot[40 + T]
    if r[2]: print("   %2d: %7d | %.3f | %6d | %5d | %6d | %5d | %d" % (T, r[2], q(r[0], r[1]), r[10], r[11], r[8], r[9], r[5]))
print("routed rel by layer:", " ".join("%d:%.3f" % (i, q(r[12], r[13])) for i, r in enumerate(L)))
