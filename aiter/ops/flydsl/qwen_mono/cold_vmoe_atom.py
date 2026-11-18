"""40 layers back to back in one graph, a NEW random x every replay (cold expert weights, as in serving). No reference:
a launch is bad when a real row has |out| > THR or a non-finite value (the kernel's normal outputs are below ~10)."""
import os, sys, time, torch
sys.path.insert(0, os.path.dirname(__file__))
import vmoe_atom as VA
torch.manual_seed(int(os.environ.get("SEED", 0))); dev = "cuda"
D = "/data/Qwen3.6-35B-A3B-MXFP4/"; R = int(os.environ.get("R", 16)); NL = int(os.environ.get("NL", 40)); REPS = int(os.environ.get("REPS", 500)); THR = 100.0
vm = VA.VMoE(D, torch.device(dev))
for i in range(NL): vm.load_layer(i)
xs = [torch.zeros(R, 2048, dtype=torch.bfloat16, device=dev) for _ in range(NL)]
outs = [torch.zeros(16, 2048, dtype=torch.bfloat16, device=dev) for _ in range(NL)]
keys = ("sel", "topv", "sgate", "xqs8", "slot", "sdn")
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
        if not VA.V.BANK: vm.cnt.zero_()
        vm.ln(vm.layers[k][0].data_ptr(), vm.cnt.data_ptr(), vm.st.data_ptr(), xs[k].data_ptr(), outs[k].data_ptr(), R, stream=torch.cuda.current_stream())
TT = []; FIXED = os.environ.get('FIXED', '0') == '1'
nbad = 0; byl = [0] * NL; byr = [0] * 16; shown = 0
for rep in range(REPS):
    for k in range(NL if (rep == 0 or not FIXED) else 0):
        if REAL: xs[k].copy_(REAL[(k * 7 + rep) % len(REAL)]); continue
        h = torch.randn(R, 2048, device=dev) * (1 + k % 3); xs[k].copy_((h * torch.rsqrt(h.pow(2).mean(1, keepdim=True) + 1e-6)).to(torch.bfloat16))
    torch.cuda.synchronize(); t0_ = time.perf_counter(); gph.replay(); torch.cuda.synchronize(); TT.append(time.perf_counter() - t0_)
    UN.append(len(set((((vm.buf['selm'][:R] >> 16) & 255) if os.environ.get('MB') == '1' else vm.buf['sel'][:R]).flatten().tolist())))
    if int(vm.cnt[512]) or rep == 1:
        uu = vm.st[:, 24]; ww = vm.st[:, 25]
        vals, cn = torch.unique(uu, return_counts=True); v2, c2 = torch.unique(ww, return_counts=True)
        print("  rep %d: U seen by the waves %s | bitmap word 2 variants %d %s | true union %d" % (rep, dict(zip(vals.tolist(), cn.tolist())), len(v2), c2.tolist(), len(set(vm.buf["sel"][:R].flatten().tolist()))))
        bad = (uu != vals[cn.argmax()]).nonzero().flatten()
        if len(bad): print("     waves with another U: CTAs", sorted(set((bad // 8).tolist()))[:40], "waves", sorted(set((bad % 8).tolist())))
    if int(vm.cnt[512]): print('  rep %d: poll timeouts %d, last site %d | gen %d | selm tags %s' % (rep, int(vm.cnt[512]), int(vm.cnt[513]), int(vm.buf['gen'][0]), sorted(set((vm.buf['selm'] >> 32).flatten().tolist()))))
    if os.environ.get("DBG") == "1":
        d = vm.st.view(-1).view(torch.int32)[:48].view(6, 8).tolist()
        if any(r_[0] for r_ in d):
            for kind, nm in ((1, "weight scale junk"), (2, "x scale junk"), (3, "huge sum, sane scales"), (4, "junk activation scale")):
                if d[kind][0]: print("  rep %d dbg %-22s: v0 %d v1 0x%08x cta.wave %d.%d count %d" % (rep, nm, d[kind][1], d[kind][2] & 0xffffffff, d[kind][3] // 8, d[kind][3] % 8, d[kind][4]))
            vm.st.zero_()
    o = torch.stack([o_[:R] for o_ in outs]).float()
    bad = (~torch.isfinite(o)) | (o.abs() > THR)
    if bool(bad.any()):
        ks, rs = bad.any(2).nonzero(as_tuple=True)
        for k, r in zip(ks.tolist(), rs.tolist()):
            nbad += 1; byl[k] += 1; byr[r] += 1
            if shown < 8:
                shown += 1; b = bad[k, r]
                print("rep %d layer %d row %d: %d bad columns (first %d last %d), max |out| %.3g, nonfinite %d" % (rep, k, r, int(b.sum()), int(b.nonzero()[0]), int(b.nonzero()[-1]), float(torch.nan_to_num(o[k, r], nan=0, posinf=0, neginf=0).abs().max()), int((~torch.isfinite(o[k, r])).sum())))
                if k == NL - 1:
                    print("   last launch: topv", vm.buf["topv"][r].tolist(), "sel", vm.buf["sel"][r].tolist(), "sgate %.4g" % float(vm.buf["sgate"][r]),
                          "\n   slot max per rank", [float("%.3g" % v) for v in vm.buf["slot"][r].abs().amax(1).tolist()], "sdn max %.3g" % float(vm.buf["sdn"][r].abs().max()), "xqs8 max code", int(vm.buf["xqs8"].max()))
print("R %d, %d layers x %d replays (%d launches): bad rows %d | by layer %s | by row %s" % (R, NL, REPS, NL * REPS, nbad, {k: v for k, v in enumerate(byl) if v}, {k: v for k, v in enumerate(byr) if v}))
TT = sorted(TT[min(20, len(TT) // 2):]); print("time a launch (incl. 1.3 us counter zero): median %.1f us, p10 %.1f, p90 %.1f (%s inputs)" % (1e6 * TT[len(TT) // 2] / NL, 1e6 * TT[len(TT) // 10] / NL, 1e6 * TT[9 * len(TT) // 10] / NL, "fixed" if FIXED else "fresh"))
print("distinct experts of the last launch: mean %.0f" % (sum(UN) / len(UN)))
