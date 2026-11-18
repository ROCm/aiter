import sys, torch
torch.set_printoptions(linewidth=250, precision=4, sci_mode=False)
for p in sys.argv[1:]:
    d = torch.load(p, map_location="cpu"); LY, T = int(d["meta"][0]), int(d["meta"][1])
    out, kr, nr, ks, ns = d["out"][:T], d["kr"][:T], d["nr"][:T], d["ks"][:T], d["ns"][:T]
    te = (kr - nr).view(T, 128, 16).pow(2).sum(-1).sqrt()
    tn = nr.view(T, 128, 16).pow(2).sum(-1).sqrt()
    r, tl = divmod(int((te / tn.clamp_min(1e-9)).argmax()), 128)
    ts = (ks - ns).view(T, 128, 16).pow(2).sum(-1).sqrt(); rs_, tls = divmod(int(ts.argmax()), 128)
    print("%s: layer %d T %d, worst tile err %.2f | routed: worst tile %d of row %d (err %.4f, native norm %.4f) | shared: worst tile %d of row %d (err %.4f, native norm %.4f)" % (
        p.split("/")[-1], LY, T, float(d["err"]), tl, r, float(te[r, tl]), float(tn[r, tl]), tls, rs_, float(ts[rs_, tls]), float(ns.view(T, 128, 16)[rs_, tls].norm())))
    c = slice(79 * 16, 80 * 16)
    print("   row %d cols 1264..1279:" % r)
    print("     kernel routed", kr[r, c]); print("     native routed", nr[r, c]); print("     kernel shared", ks[r, c]); print("     native shared", ns[r, c]); print("     kernel out   ", out[r, c])
    print("   routed tile 79 err by row:", [round(float(v), 3) for v in te[:, 79]], "| neighbours 78, 80 of row %d: %.4f %.4f" % (r, float(te[r, 78]), float(te[r, 80])))
    print("   |x| max of row %d: %.2f; kernel top-8 %s" % (r, float(d["x"][r].abs().max()), d["sel"][r].long().tolist()))
