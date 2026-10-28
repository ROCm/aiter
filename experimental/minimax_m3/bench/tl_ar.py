"""All-reduce milestones from saved TRACE=3 timelines (bench/logs/tl{tag}_gluon_r*_rows{os.environ.get('ROWS', '48')}_ctx1536.pt):
per (rank, layer), us after the layer's first stamp, of the last CTA reaching each milestone;
averaged over ranks and layers 1 .. L-2.
python bench/tl_ar.py tag [tag ...]   (ROWS=60: other widths)"""
import os, sys, glob, statistics as st, torch

MS = [  # (name, stamp, CTA range)
    ("qkv tasks end", 1, (0, 160)),
    ("split end", 2, (0, 256)),
    ("ffn puts issued", 38, (0, 192)),
    ("ffn puts acked", 37, (0, 192)),
    ("ffn AR done (far CTAs)", 24, (0, 192)),
    ("norm input landed", 50, (192, 256)),
    ("x8s put", 51, (192, 256)),
    ("merged", 11, (0, 256)),
    ("o puts issued", 25, (0, 192)),
    ("attn AR landed", 29, (0, 192)),
    ("o end", 3, (0, 192)),
    ("router end", 4, (0, 256)),
    ("launch end", 5, (0, 256)),
]
for tag in sys.argv[1:]:
    agg = {m[0]: [] for m in MS}
    first = {m[0]: [] for m in MS}
    for f in sorted(glob.glob(f"bench/logs/tl{tag}_gluon_r*_rows{os.environ.get('ROWS', '48')}_ctx1536.pt")):
        tl = torch.load(f).double()
        for li in range(1, tl.shape[0] - 1):
            t = tl[li]
            t0 = t[:, 0][t[:, 0] > 0].min()
            for n, s, (a, b) in MS:
                v = t[a:b, s]
                v = v[v > 0]
                if v.numel():
                    agg[n].append(float((v.max() - t0) / 100.0))
                    first[n].append(float((v.min() - t0) / 100.0))
    print(f"[{tag}] last CTA (first CTA), us after layer start:")
    for n, _, _ in MS:
        if agg[n]:
            print(f"  {n:24s} {st.mean(agg[n]):7.2f}  ({st.mean(first[n]):6.2f})")
