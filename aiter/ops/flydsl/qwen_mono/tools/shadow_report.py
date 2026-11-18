import json, glob, math, sys
tot = None
for f in sorted(glob.glob("/mnt/data/vigokhal/qwen35_runs/%s.acc.*.json" % sys.argv[1])):
    a = json.load(open(f))
    tot = a if tot is None else [[x + y for x, y in zip(r, s)] for r, s in zip(tot, a)]
L = tot[:40]; q = lambda a, b: math.sqrt(a / b) if b else float("nan"); A = lambda k: sum(r[k] for r in L)
print("calls a layer %d, rows a call %.1f, rows compared %d, off by > 50%% %d, non-finite kernel out %d" % (L[0][2], L[0][3] / max(L[0][2], 1), A(7), A(8), A(5)))
print("layer: total | routed | shared | gate | shared share of the sum")
for i in (0, 1, 5, 10, 17, 24, 32, 39):
    r = L[i]; print("  %2d: %.3f | %.3f | %.3f | %.4f | %.2f" % (i, q(r[0], r[1]), q(r[12], r[13]), q(r[14], r[15]), q(r[16], r[17]), q(r[18], r[1])))
print("all layers: total %.3f | routed %.3f | shared %.3f | gate %.4f" % (q(A(0), A(1)), q(A(12), A(13)), q(A(14), A(15)), q(A(16), A(17))))
