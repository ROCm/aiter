"""Per-CTA stamp deltas of a saved timeline (bench_layer_tp --timeline): for each consecutive
pair of the given slots, the median / p90 over the CTAs that stamped both.
python3 bench/tl_cta.py bench/logs/tl_gluon_r0_rows8_ctx1536.pt LAYER s0,s1,s2,..."""
import sys
import torch

tl = torch.load(sys.argv[1])
li = int(sys.argv[2])
slots = [int(x) for x in sys.argv[3].split(",")]
t = tl[li].double()  # [CTA, slots] ns (s_memrealtime: 100 MHz ticks -> us below)
base = t[:, 0][t[:, 0] > 0].min()
for a, b in zip(slots, slots[1:]):
    m = (t[:, a] > 0) & (t[:, b] > 0)
    d = (t[m, b] - t[m, a]) / 100.0
    if d.numel():
        q = torch.quantile(d, torch.tensor([0.5, 0.9], dtype=torch.float64))
        print(f"  {a:2d} -> {b:2d}: n={int(m.sum()):3d}  median {q[0]:6.2f} us  p90 {q[1]:6.2f} us")
