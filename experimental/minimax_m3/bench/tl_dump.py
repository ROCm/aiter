"""Per stamp point averaged over layers (selecting / reusing): median and max us after layer start."""
import sys, torch
sys.path.insert(0, "/home/vigokhal/m3_monokernel/bench")
tl = torch.load(sys.argv[1]).double()
L = int(sum(bool((tl[l, :, 0] > 0).any()) for l in range(tl.shape[0])))
t0 = torch.stack([tl[l, :, 0][tl[l, :, 0] > 0].min() for l in range(L)])
for kind, ls in (("select", [l for l in range(1, L) if l % 4 == 0]), ("reuse", [l for l in range(1, L) if l % 4])):
    print(f"== {kind} ({len(ls)} layers); layer period {((t0[1:] - t0[:-1]) / 100)[[l for l in ls if l < L - 1]].mean():.2f} us")
    rows = []
    for k in range(tl.shape[2]):
        meds, mxs, ns = [], [], []
        for l in ls:
            v = tl[l, :, k]; v = v[v > 0]
            if len(v):
                us = (v - t0[l]) / 100; meds.append(us.median()); mxs.append(us.max()); ns.append(len(v))
        if meds:
            rows.append((sum(meds) / len(meds), sum(mxs) / len(mxs), sum(ns) / len(ns), k))
    for md, mx, n, k in sorted(rows):
        print(f"  p{k:2d} med {md:6.2f} max {mx:6.2f} n {n:5.1f}")
