"""Absolute stamp times (us after the layer's first start): per slot min / median / max over
the CTAs reaching it. python3 bench/tl_abs.py TL.pt layer slots..."""
import sys
import torch

tl = torch.load(sys.argv[1]).double()
t = tl[int(sys.argv[2])]
t0 = t[:, 0][t[:, 0] > 0].min()
for k in (int(x) for x in sys.argv[3:]):
    v = t[:, k]
    v = (v[v > 0] - t0) / 100.0
    print(f"  slot {k:2d}: n={len(v):3d} min {v.min().item():6.2f} med {v.median().item():6.2f} max {v.max().item():6.2f}")
