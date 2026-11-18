"""40 layers back to back in one graph, a NEW random x every replay (cold expert weights, as in serving). No reference:
a launch is bad when a real row has |out| > THR or a non-finite value (the kernel's normal outputs are below ~10)."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
import vmoe_atom as VA
torch.manual_seed(int(os.environ.get("SEED", 0))); dev = "cuda"
D = "/data/Qwen3.6-35B-A3B-MXFP4/"; R = int(os.environ.get("R", 16)); NL = int(os.environ.get("NL", 40)); REPS = int(os.environ.get("REPS", 500)); THR = 100.0
vm = VA.VMoE(D, torch.device(dev))
for i in range(NL): vm.load_layer(i)
xs = [torch.zeros(R, 2048, dtype=torch.bfloat16, device=dev) for _ in range(NL)]
outs = [torch.zeros(16, 2048, dtype=torch.bfloat16, device=dev) for _ in range(NL)]
keys = ("sel", "topv", "sgate", "xqs8", "slot", "sdn")
gs = []
for k in range(NL):
    g_ = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_):
        vm.cnt.zero_()
        vm.ln(vm.layers[k][0].data_ptr(), vm.cnt.data_ptr(), vm.st.data_ptr(), xs[k].data_ptr(), outs[k].data_ptr(), R, stream=torch.cuda.current_stream())
    gs.append(g_)
nbad = 0; shown = 0; cause = {}
for rep in range(REPS):
    for k in range(NL):
        h = torch.randn(R, 2048, device=dev) * (1 + k % 3); xs[k].copy_((h * torch.rsqrt(h.pow(2).mean(1, keepdim=True) + 1e-6)).to(torch.bfloat16))
    torch.cuda.synchronize()
    for k in range(NL):
        gs[k].replay(); torch.cuda.synchronize()
        o = outs[k][:R].float()
        bad = ((~torch.isfinite(o)) | (o.abs() > THR)).any(1)
        if bool(bad.any()):
            for r in bad.nonzero().flatten().tolist():
                nbad += 1
                sl = vm.buf["slot"][r].abs().amax(1); tv = vm.buf["topv"][r]; sd = float(vm.buf["sdn"][r].abs().max()); sg = float(vm.buf["sgate"][r])
                why = []
                if bool((sl > THR).any()) or not bool(torch.isfinite(sl).all()): why.append("slot ranks %s" % (sl > THR).nonzero().flatten().tolist())
                if sd > THR or sd != sd: why.append("sdn")
                if not (0 <= sg <= 1): why.append("sgate")
                if bool((tv.abs() > 1).any()): why.append("topv")
                why = "; ".join(why) or "none of slot/sdn/sgate/topv"
                cause[why.split(" ranks")[0]] = cause.get(why.split(" ranks")[0], 0) + 1
                if shown < 10:
                    shown += 1
                    nb = (vm.buf["slot"][r].abs() > THR)
                    print("rep %d layer %d row %d: %s | topv %s sel %s | slot max/rank %s | bad cols per rank %s first %s | sdn max %.3g sgate %.3g | experts %d" % (
                        rep, k, r, why, [float("%.3f" % v) for v in tv.tolist()], vm.buf["sel"][r].tolist(), [float("%.3g" % v) for v in sl.tolist()], nb.sum(1).tolist(),
                        [int(x_.nonzero()[0]) if bool(x_.any()) else -1 for x_ in nb], sd, sg, len(set(vm.buf["sel"][:R].flatten().tolist()))))
print("R %d, %d layers x %d replays: bad rows %d, causes %s" % (R, NL, REPS, nbad, cause))
