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
names = {13: "owner: own expert flagged", 1: "router: x in LDS", 5: "router: weights in", 10: "router: GEMM + reduce", 12: "router: logits stored", 6: "router: cluster barrier", 0: "entry", 9: "prestage done", 7: "rank published", 11: "tag seen", 2: "assigned", 8: "own unit done", 14: "helper unit done", 3: "gate/up done", 4: "down done", 15: "combine wait over"}
acc = {k: [] for k in names}; accm = {k: [] for k in names}
for rep in range(REPS):
    for k in range(NL):
        if REAL: xs[k].copy_(REAL[(k * 7 + rep) % len(REAL)]); continue
        h = torch.randn(R, 2048, device=dev) * (1 + k % 3); xs[k].copy_((h * torch.rsqrt(h.pow(2).mean(1, keepdim=True) + 1e-6)).to(torch.bfloat16))
    vm.st.zero_(); gph.replay(); torch.cuda.synchronize()
    UN.append(len(set((((vm.buf['selm'][:R] >> 16) & 255) if os.environ.get('MB') == '1' else vm.buf['sel'][:R]).flatten().tolist())))
    if rep < 10: continue
    st = vm.st.double(); t0 = st[:, 0][st[:, 0] > 0].min()
    for k in names:
        v = st[:, k]; v = v[v > 0]
        if len(v): acc[k].append(float((v.median() - t0)) / 100); accm[k].append(float((v.max() - t0)) / 100)
print("R %d, %d layers cycling: stamp (us after the first wave's entry; median over waves | last wave), mean of %d replays" % (R, NL, REPS - 10))
for k in sorted(names, key=lambda k_: sum(accm[k_]) / max(len(accm[k_]), 1)):
    if acc[k]: print("  %-18s %6.1f | %6.1f" % (names[k], sum(acc[k]) / len(acc[k]), sum(accm[k]) / len(accm[k])))
print("distinct experts of the last launch: mean %.0f" % (sum(UN) / len(UN)))
