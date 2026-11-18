import os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
import lab as L
names = ["CTA barrier", "handoff (incl. start skew)", "handoff, aligned, no fences", "handoff with fences", "cluster barrier + fences",
         "cold burst from HBM -> LDS", "(read LDS, dev-scope store issue)", "store drain + handoff", "L2 load of other CTA's data + store", "64 KB x staging L2 -> LDS + barrier"]
for KB in [int(a) for a in sys.argv[1:]] or [8]:
    W = torch.randint(0, 2**31, (2048 * KB * 1024 // 4 * 64,), dtype=torch.int32, device="cuda")
    xg = torch.zeros(65536 * 4 + 65536, dtype=torch.int32, device="cuda")
    cnt = torch.zeros(256, dtype=torch.int32, device="cuda"); st = torch.zeros(256, L.NST, dtype=torch.int64, device="cuda")
    ln = L.build(KB)
    rows = []
    for i in range(80):
        cnt.zero_(); torch.cuda.synchronize()
        off = (i % 60) * 2048 * KB * 1024          # a fresh (cold) region each launch
        ln(W.data_ptr() + off, xg.data_ptr(), cnt.data_ptr(), st.data_ptr(), stream=torch.cuda.current_stream()); torch.cuda.synchronize()
        d = (st[:, 1:11] - st[:, 0:10]).float() / 100
        rows.append(torch.stack([d.median(0).values, d.max(0).values]))
    r = torch.stack(rows[10:]).median(0).values
    print("burst %d KB/wave (%.0f MB total)" % (KB, 2048 * KB / 1024))
    for n, a, b in zip(names, r[0].tolist(), r[1].tolist()):
        print("  %-40s median CTA %.2f us   slowest CTA %.2f us" % (n, a, b))
