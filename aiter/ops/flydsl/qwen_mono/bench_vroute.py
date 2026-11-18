"""vroute on real rows: is the selection every wave holds right, and when does each step happen."""
import os, sys, glob, torch
os.environ["STAMPS"] = "1"; os.environ["ALLOW_STAMPS"] = "1"
sys.path.insert(0, os.path.dirname(__file__))
import vmoe_atom as VA, vmoe2 as V, vroute as RT
dev = "cuda"; R = int(os.environ.get("R", 16)); NL = int(os.environ.get("NL", 8)); REPS = int(os.environ.get("REPS", 80))
vm = VA.VMoE("/data/Qwen3.6-35B-A3B-MXFP4/", torch.device(dev))
for i in range(NL): vm.load_layer(i)
ln = RT.build()
REAL = [torch.load(f)["x"][:R].to(dev).to(torch.bfloat16) for f in sorted(glob.glob("/runs/*.worst.pt")) if int(torch.load(f)["meta"][1]) == R]
xs = [torch.zeros(R, 2048, dtype=torch.bfloat16, device=dev) for _ in range(NL)]
out = torch.zeros(16, 2048, dtype=torch.bfloat16, device=dev)
Wg = {}
from safetensors import safe_open
for i in range(NL):
    with safe_open("/data/Qwen3.6-35B-A3B-MXFP4/layers-%d.safetensors" % i, "pt") as f: Wg[i] = f.get_tensor("model.language_model.layers.%d.mlp.gate.weight" % i).to(dev).float()
gs = []
for k in range(NL):
    g_ = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_):
        if not V.BANK: vm.cnt.zero_()
        ln(vm.layers[k][0].data_ptr(), vm.cnt.data_ptr(), vm.st.data_ptr(), xs[k].data_ptr(), out.data_ptr(), R, stream=torch.cuda.current_stream())
    gs.append(g_)
names = {0: "entry", 20: "router: x in LDS", 21: "router: weights in", 22: "router: GEMM + reduce", 6: "router: logits out", 16: "rank: logit in hand", 17: "rank: after CTA barrier",
         18: "rank: rank known", 7: "rank: published", 28: "collector: counter seen", 19: "collector: done", 27: "signal seen (after CTA barrier)", 10: "wave holds the selection", 5: "exit"}
roles = {"routers 0..15": (0, 16), "consumers 16..255": (16, 256)}
acc = {(r, k): [] for r in roles for k in names}; wrong = 0; tot = 0; slow = 0
for rep in range(REPS):
    for k in range(NL):
        xs[k].copy_(REAL[(k * 7 + rep) % len(REAL)])
        vm.st.zero_(); gs[k].replay(); torch.cuda.synchronize()
        st = vm.st; t0 = st[:, 0][st[:, 0] > 0].min()
        lg = (xs[k].float() @ Wg[k].T).to(torch.bfloat16).float()
        sel = torch.sort(-lg, dim=1, stable=True).indices[:, :8]
        U = len(set(sel.flatten().tolist())); w2 = 0
        for e in set(sel.flatten().tolist()):
            if 64 <= e < 96: w2 |= 1 << (e - 64)
        w2 = w2 - (1 << 32) if w2 >= (1 << 31) else w2
        ok = (st[:, 24] == U) & (st[:, 25] == w2)
        tot += 1; wrong += int((~ok).any()); slow += int((st[:, 5].max() - t0) > 100000)
        if not bool(ok.all()) and wrong <= 3: print("  rep %d layer %d: %d waves hold a wrong selection (U %s vs %d)" % (rep, k, int((~ok).sum()), sorted(set(st[:, 24].tolist()))[:6], U))
        if rep < 5: continue
        for r, (lo, hi) in roles.items():
            for kk in names:
                v = st[lo * 8:hi * 8, kk]; v = v[v > 0]
                if len(v): acc[(r, kk)].append(((v - t0).double() / 100).cpu())
print("vroute R %d, %d layers, RT3 %s MB %s: launches with a wrong selection %d of %d; launches over 1 ms: %d" % (R, NL, V.RT3, V.MB, wrong, tot, slow))
print("us after the first entry, over waves x launches: p10 | p50 | p90 | max-of-launch median")
for r in roles:
    print(r)
    for kk in names:
        a = acc[(r, kk)]
        if not a: continue
        allv = torch.cat(a).sort().values; n = len(allv); mx = torch.stack([x.max() for x in a]).sort().values
        print("   %-26s %5.1f | %5.1f | %5.1f | %5.1f" % (names[kk], allv[n // 10], allv[n // 2], allv[9 * n // 10], mx[len(mx) // 2]))
