"""Attention hand-off latencies per layer (us after layer t0), Gluon vs FlyDSL, S = 4.
    python3 tl_handoff.py GLUON.pt FLYDSL.pt"""
import sys
import torch


def rel(x):
    t0 = x[:, 0][x[:, 0] > 0].min()
    return lambda b, k: ((x[b, k] - t0) / 100).item() if x[b, k] > 0 else float("nan")


def gl_layer(x):
    f = rel(x)
    sp = max(f(255 - (8 * k + p), 16) for k in range(4) for p in range(8))  # last split's puts start
    mg = [f(192 + k, 31) for k in range(4)]
    ml = [f(192 + k, 33) for k in range(4)]
    mp = [f(192 + k, 3) for k in range(4)]
    oi = sorted(f(b, 33) for b in range(192))
    oe = sorted(f(b, 29) for b in range(192))
    return [sp, max(mg), max(ml), max(mp), oi[96], oe[96], oe[-1]]


def fly_layer(x):
    f = rel(x)
    sp = max(f(b, 15) for b in range(32))
    ml = [f(32 + k, 3) for k in range(4)]
    mp = [f(32 + k, 4) for k in range(4)]
    oi = sorted(f(b, 3) for b in range(36, 228))
    oe = sorted(f(b, 4) for b in range(36, 228))
    return [sp, float("nan"), max(ml), max(mp), oi[96], oe[96], oe[-1]]


g = torch.load(sys.argv[1]).double()
fl = torch.load(sys.argv[2]).double()
cols = ["split put", "merge gate", "merge land", "merge end", "o input(med)", "o end(med)", "o end(max)"]
for name, t, fn in (("gluon", g, gl_layer), ("flydsl", fl, fly_layer)):
    rows = torch.tensor([fn(t[L]) for L in range(1, t.shape[0])])
    m = rows.nanmean(0) if hasattr(rows, "nanmean") else rows.mean(0)
    print(f"{name:7s} " + "  ".join(f"{c} {v:6.2f}" for c, v in zip(cols, m.tolist())))
    d = m - m[0]
    print(f"{'':7s} " + "  ".join(f"{c} {v:+6.2f}" for c, v in zip(cols, d.tolist())))
