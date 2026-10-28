"""o pipeline milestones (TRACE=3): last / median CTA, us after layer start, layers 1 .. L-2, all ranks.
python bench/tl_o.py TAG ROWS"""
import sys, glob, statistics as st, torch
tag, rows = sys.argv[1], sys.argv[2]
MS = [("split end", 2, (0, 256)), ("merge start (31)", 31, (0, 256)), ("merge end (33)", 33, (0, 256)),
      ("o g0 input landed", 30, (0, 192)), ("o last group landed", 56, (0, 192)), ("o puts issued", 25, (0, 192)),
      ("attn AR landed", 29, (0, 192))]
agg = {m[0]: ([], []) for m in MS}
for f in sorted(glob.glob(f"bench/logs/tl{tag}_gluon_r*_rows{rows}_ctx1536.pt")):
    tl = torch.load(f).double()
    for li in range(1, tl.shape[0] - 1):
        t = tl[li]; t0 = t[:, 0][t[:, 0] > 0].min()
        for n, s, (a, b) in MS:
            v = t[a:b, s]; v = v[v > 0]
            if v.numel():
                agg[n][0].append(float((v.max() - t0) / 100)); agg[n][1].append(float((v.median() - t0) / 100))
print(f"[{tag} rows {rows}] last (median)")
for n, _, _ in MS:
    if agg[n][0]:
        print(f"  {n:22s} {st.mean(agg[n][0]):7.2f} ({st.mean(agg[n][1]):6.2f})")
