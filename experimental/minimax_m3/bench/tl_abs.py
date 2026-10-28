"""Absolute stage times of one layer of a saved timeline: per slot, median / max over CTAs of
(stamp - the layer's first start), us. python3 bench/tl_abs.py TL.pt LAYER s0,s1,..."""
import sys
import torch

tl = torch.load(sys.argv[1])
li = int(sys.argv[2])
t = tl[li].double()
t0 = t[:, 0][t[:, 0] > 0].min()
for s in [int(x) for x in sys.argv[3].split(",")]:
    v = t[:, s]
    v = (v[v > 0] - t0) / 100.0
    if v.numel():
        print(f"  slot {s:2d}: n={v.numel():3d}  median {v.median():6.2f}  max {v.max():6.2f}")
