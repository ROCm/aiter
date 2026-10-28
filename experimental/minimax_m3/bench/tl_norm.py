"""Input-norm critical path from saved TRACE=3 timelines (bench/logs/tl_gluon_r*_rows48_ctx1536.pt):
per (rank, layer) the last norm input landed -> last x8s put -> first qkv task past its gate,
and the per-CTA phase medians; averaged over ranks and layers 1 .. L-2.
python bench/tl_norm.py [tag]"""
import sys, glob, statistics as st, torch
PH = [(50, 53, "rstd"), (53, 54, "x_scale"), (54, 55, "words issued"), (55, 51, "x8s put")]
agg = {k: [] for k in ["far->land", "far->qkv", "land->x8s", "x8s->qkv", "land->qkv", "norm task (med CTA)"] + [p[2] for p in PH]}
tag = sys.argv[1] if len(sys.argv) > 1 else ""
for f in sorted(glob.glob(f"bench/logs/tl{tag}_gluon_r*_rows48_ctx1536.pt")):
    tl = torch.load(f).double()
    for li in range(1, tl.shape[0] - 1):
        t = tl[li]
        if not (t[:, 49] > 0).any():
            continue
        nt = t[t[:, 49] > 0] / 100.0
        q = t[:, 20][t[:, 20] > 0] / 100.0
        land, x8s = nt[:, 50].max(), nt[:, 51].max()
        far = (t[:, 24][t[:, 24] > 0] / 100.0).max()
        agg["far->land"].append(land - far)
        agg["far->qkv"].append(q.min() - far)
        agg["land->x8s"].append(x8s - land)
        agg["x8s->qkv"].append(q.min() - x8s)
        agg["land->qkv"].append(q.min() - land)
        agg["norm task (med CTA)"].append((nt[:, 51] - nt[:, 50]).median())
        for a, b, n in PH:
            if (nt[:, a] > 0).all() and (nt[:, b] > 0).all():
                agg[n].append((nt[:, b] - nt[:, a]).median())
print(tag or "-", "  ".join(f"{k} {st.mean(float(x) for x in v):.2f}" for k, v in agg.items() if v))
