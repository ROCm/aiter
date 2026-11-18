"""Stamps of the LAST launch of an NL-layer back-to-back graph (STAMPS=1 ALLOW_STAMPS=1): where the time goes when the
weights of 40 layers cycle, vs one layer (NL=1)."""
import os, sys, torch
os.environ["STAMPS"] = "1"; os.environ["ALLOW_STAMPS"] = "1"
sys.path.insert(0, os.path.dirname(__file__))
import vmoe_atom as VA, vmoe2 as V
torch.manual_seed(0); dev = "cuda"
R = int(os.environ.get("R", 16)); NL = int(os.environ.get("NL", 40)); REPS = int(os.environ.get("REPS", 200))
vm = VA.VMoE("/data/Qwen3.6-35B-A3B-MXFP4/", torch.device(dev))
for i in range(NL): vm.load_layer(i)
xs = [torch.zeros(R, 2048, dtype=torch.bfloat16, device=dev) for _ in range(NL)]
outs = [torch.zeros(16, 2048, dtype=torch.bfloat16, device=dev) for _ in range(NL)]
REAL = []
if os.environ.get("REALX") == "1":                 # real served rows (the worst-call dumps of the shadow runs) of R rows a call
    import glob
    for f in sorted(glob.glob("/runs/*.worst.pt")):
        d = torch.load(f)
        if int(d["meta"][1]) == R: REAL.append(d["x"][:R].to(dev).to(torch.bfloat16))
    print("real x: %d calls of %d rows" % (len(REAL), R))
UN = []
gph = torch.cuda.CUDAGraph()
with torch.cuda.graph(gph):
    for k in range(NL):
        vm.cnt.zero_()
        vm.ln(vm.layers[k][0].data_ptr(), vm.cnt.data_ptr(), vm.st.data_ptr(), xs[k].data_ptr(), outs[k].data_ptr(), R, stream=torch.cuda.current_stream())
names = {0: "entry", 9: "prestage (pointers, first loads issued)", 23: "router: reached its stage", 20: "router: x in LDS", 21: "router: weights in LDS", 22: "router: GEMM + reduce", 6: "router: logits out", 16: "rank: logit in hand", 17: "rank: after CTA barrier", 18: "rank: rank known", 7: "rank published", 19: "collector: all words tagged", 10: "all-token tag seen", 13: "owner: own expert flagged", 8: "owner unit done", 14: "helper unit done", 3: "gate/up all done (wave)", 12: "down: its expert's activations ready", 4: "down done", 15: "combine wait over"}
roles = {"rankers 0..15": (0, 16), "shared 16..32": (16, 33), "experts 33..255": (33, 256)}
acc = {(r, k): [] for r in roles for k in names}
for rep in range(REPS):
    for k in range(NL):
        if REAL: xs[k].copy_(REAL[(k * 7 + rep) % len(REAL)]); continue
        h = torch.randn(R, 2048, device=dev); xs[k].copy_((h * torch.rsqrt(h.pow(2).mean(1, keepdim=True) + 1e-6)).to(torch.bfloat16))
    vm.st.zero_(); gph.replay(); torch.cuda.synchronize()
    if rep < 10: continue
    st = vm.st.double(); t0 = st[:, 0][st[:, 0] > 0].min()
    for r, (lo, hi) in roles.items():
        for k in names:
            v = st[lo * 8:hi * 8, k]; v = v[v > 0]
            if len(v): acc[(r, k)].append(((v - t0) / 100).cpu())
print("R %d, %d layers, real rows %s: us after the first entry, over waves x replays: p10 | p50 | p90 | p99 | max-of-launch median  (waves a launch)" % (R, NL, bool(REAL)))
for r in roles:
    print(r)
    for k in names:
        a = acc[(r, k)]
        if not a: continue
        allv = torch.cat(a).sort().values; n = len(allv); mx = torch.stack([x.max() for x in a]).sort().values
        print("   %-38s %5.1f | %5.1f | %5.1f | %5.1f | %5.1f   (%d)" % (names[k], allv[n // 10], allv[n // 2], allv[9 * n // 10], allv[99 * n // 100], mx[len(mx) // 2], n // len(a)))
