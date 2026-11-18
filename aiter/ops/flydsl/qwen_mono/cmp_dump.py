"""The kernel on dumped server rows against ATOM's own parts (routed sum nr, shared ns, gate ng) from the same dump."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
import vmoe_atom as VA
dev = "cuda"; vm = VA.VMoE("/data/Qwen3.6-35B-A3B-MXFP4/", torch.device(dev))
print("A4 =", VA.V.A4)
rel = lambda a, b: float((a.float() - b.float()).norm() / b.float().norm())
seen = set()
for p in sys.argv[1:]:
    for d in torch.load(p, map_location="cpu"):
        LY, T = d["layer"], d["x"].shape[0]
        if (LY, T) in seen: continue
        seen.add((LY, T)); vm.load_layer(LY)
        out = vm.run(LY, d["x"].to(dev).contiguous()).clone(); torch.cuda.synchronize()
        kr = vm.buf["slot"][:T].sum(1).cpu(); ks = vm.buf["sdn"][:T].cpu(); kg = vm.buf["sgate"][:T].cpu().unsqueeze(1)
        nat = d["nr"] + d["ng"] * d["ns"]
        rows = ((kr - d["nr"]).norm(dim=1) / d["nr"].norm(dim=1)).tolist()
        print("layer %2d T %2d: routed vs ATOM %.3f (rows %.3f..%.3f) | shared %.3f | gate %.4f | total %.3f" % (LY, T, rel(kr, d["nr"]), min(rows), max(rows), rel(ks, d["ns"]), rel(kg, d["ng"]), rel(out.cpu(), nat)))
