"""Hand-off latencies, FlyDSL vs Gluon (rank 0, layer 1, TRACE=3-style stamps):
producer's last stamp (max over CTAs) -> consumer's landed stamp (median).
python3 bench/tl_hops.py [rows]"""
import sys
import torch

rows = int(sys.argv[1]) if len(sys.argv) > 1 else 1
# hop: (fly producer slot, fly consumer slot, gluon producer slot, gluon consumer slot)
HOPS = [("split -> o parts", 1, 3, 2, 32), ("o -> router a,h", 4, 22, 3, 17),
        ("router scores -> routed", 23, 21, 19, 41), ("up/gate -> mids", 8, 10, 8, 23),
        ("qkv gemv -> heads/split q", 29, 12, 1, 13)]


def pts(path):
    t = torch.load(path)[1].double()
    t0 = t[:, 0][t[:, 0] > 0].min()

    def get(k, stat):
        v = t[:, k]
        v = (v[v > 0] - t0) / 100
        return float(v.max() if stat == "max" else v.median()) if len(v) else float("nan")
    return get


f = pts(f"bench/logs/tl_flydsl_r0_rows{rows}_ctx1536.pt")
g = pts(f"bench/logs/tl_gluon_r0_rows{rows}_ctx1536.pt")
print(f"{'hop':28s} {'fly prod':>8s} {'cons':>6s} {'hop':>5s} | {'glu prod':>8s} {'cons':>6s} {'hop':>5s} | gap")
for name, fp, fc, gp, gc in HOPS:
    a, b = f(fp, "max"), f(fc, "med")
    c, d = g(gp, "max"), g(gc, "med")
    print(f"{name:28s} {a:8.2f} {b:6.2f} {b - a:5.2f} | {c:8.2f} {d:6.2f} {d - c:5.2f} | {(d - c) - (b - a):+.2f}")
