"""Front of the layer (input norm -> qkv -> own key -> split end) per CTA role, median/max over
reuse layers, us after the layer's t0.  python3 tl_front.py A.pt [B.pt ..]"""
import sys, statistics as st
import torch
for path in sys.argv[1:]:
    t = torch.load(path).double()
    n = int(sum(bool((t[l, :, 0] > 0).any()) for l in range(t.shape[0]))); t = t[:n]
    Ls = [l for l in range(1, n - 1) if l % 4]
    def col(bids, k):
        v = []
        for L in Ls:
            x = t[L]; t0 = x[:, 0][x[:, 0] > 0].min()
            for b in bids:
                if x[b, k] > 0: v.append(((x[b, k] - t0) / 100).item())
        return v
    print(f"== {path} ({len(Ls)} reuse layers)")
    rows = [("norm 165-168", range(165, 169), [0, 49, 50, 53, 54, 55, 51]), ("qkv 0-143", range(0, 144), [0, 20, 21, 1]),
            ("own 161-164", range(161, 165), [34, 35, 36]), ("split 224..255", range(224, 256), [16, 2]),
            ("layer end (all)", range(256), [5])]
    for name, bids, ks in rows:
        out = []
        for k in ks:
            v = col(bids, k)
            if v: out.append(f"p{k} {st.median(v):5.2f}/{max(v):5.2f}")
        print(f"  {name:16s} " + "  ".join(out))
