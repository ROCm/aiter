"""o group-2 phase stamps (VLLM_M3_GLUON_O_TST=1, TRACE=3): per-CTA deltas, median / p90 over o CTAs,
layers 1 .. L-2, all ranks.  python bench/tl_og.py TAG ROWS"""
import sys, glob, torch
tag, rows = sys.argv[1], sys.argv[2]
SEQ = [(14, "top: peeks issued"), (15, "x built + LDS stored"), (16, "scale wait"), (26, "GEMM (o_mma4)"),
       (27, "puts"), (28, "gate wait + next words issued"), (34, "owner fin"), (35, "land fin")]
d = {i: [] for i in range(len(SEQ) - 1)}
for f in sorted(glob.glob(f"bench/logs/tl{tag}_gluon_r*_rows{rows}_ctx1536.pt")):
    tl = torch.load(f).double()
    for li in range(1, tl.shape[0] - 1):
        t = tl[li, :192]
        for i in range(len(SEQ) - 1):
            a, b = t[:, SEQ[i][0]], t[:, SEQ[i + 1][0]]
            ok = (a > 0) & (b > 0)
            d[i].append((b[ok] - a[ok]) / 100)
for i in range(len(SEQ) - 1):
    v = torch.cat(d[i])
    print(f"  {SEQ[i][1]:28s} -> {SEQ[i + 1][1]:30s} med {v.median():6.2f}  p90 {v.quantile(.9):6.2f} us")
