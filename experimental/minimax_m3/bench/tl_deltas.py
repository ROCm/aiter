"""Per-CTA deltas between Gluon timeline stamps (median over the CTAs having both).
python3 bench/tl_deltas.py TL.pt layer a,b b,c ..."""
import sys
import torch

tl = torch.load(sys.argv[1]).double()
li = int(sys.argv[2])
t = tl[li]
t0 = t[:, 0][t[:, 0] > 0].min()
for pair in sys.argv[3:]:
    a, b = (int(x) for x in pair.split(","))
    ok = (t[:, a] > 0) & (t[:, b] > 0)
    d = (t[ok, b] - t[ok, a]) / 100.0
    if len(d):
        print(f"  {a:2d} -> {b:2d}: n={int(ok.sum()):3d} median {d.median().item():6.2f} max {d.max().item():6.2f} us"
              f"   (start of {a}: median {((t[ok, a] - t0) / 100).median().item():6.2f})")
