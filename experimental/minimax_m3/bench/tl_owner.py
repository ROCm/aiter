import glob, torch, statistics as st, sys
for tag in sys.argv[1:]:
    ps = {q: [] for q in (10, 50, 90, 100)}
    order = []
    for f in sorted(glob.glob(f"/home/vigokhal/m3_monokernel/bench/logs/tl{tag}_gluon_r*_rows48_ctx1536.pt")):
        rank = int(f.split("_r")[1].split("_")[0])
        tl = torch.load(f).double()
        for li in range(1, tl.shape[0] - 1):
            t = tl[li]
            t0 = t[:, 0][t[:, 0] > 0].min()
            own = torch.tensor([c for c in range(192) if c % 4 == rank])
            v = ((t[own, 24] - t0) / 100.0).sort().values
            for q in ps:
                ps[q].append(float(v[min(len(v) - 1, int(q / 100 * len(v)))]))
            order.append(own[((t[own, 24] - t0) / 100.0).argsort()[-3:]].tolist())
    print(tag, "owner done p10/p50/p90/max:", " ".join(f"{st.mean(ps[q]):.2f}" for q in ps), " slowest CTAs (last 6 samples):", order[-6:])
