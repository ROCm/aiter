import sys, torch
D = torch.load(sys.argv[1], map_location="cpu")
for d in D[:int(sys.argv[2])]:
    kr, nr, x = d["kr"].float(), d["nr"].float(), d["x"].float(); T = kr.shape[0]
    E = ((kr[:, None, :] - nr[None, :, :]).norm(dim=-1) / nr.norm(dim=-1)[None, :])        # [kernel row i][native row j]
    best = E.argmin(0)
    cosx = torch.nn.functional.cosine_similarity(x[:, None, :], x[None, :, :], dim=-1)
    print("layer %d T %d: diag rel %s" % (d["layer"], T, " ".join("%.2f" % float(E[i, i]) for i in range(T))))
    print("   best kernel row for each native row: %s  (rel %s)" % (best.tolist(), " ".join("%.2f" % float(E[best[j], j]) for j in range(T))))
    print("   x row cosine to row 0: %s" % " ".join("%.2f" % float(cosx[0, j]) for j in range(T)))
    print("   |nr| rows %s" % " ".join("%.2f" % float(v) for v in nr.norm(dim=1)), "| |kr| rows %s" % " ".join("%.2f" % float(v) for v in kr.norm(dim=1)))
    ns, ks = d["ns"].float(), d["ks"].float()
    print("   shared diag rel %s" % " ".join("%.2f" % float((ks[i] - ns[i]).norm() / ns[i].norm()) for i in range(T)))
