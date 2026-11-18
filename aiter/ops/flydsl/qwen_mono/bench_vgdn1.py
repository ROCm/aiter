"""vgdn1 (in_proj, stand-in verify, out_proj) on layer LAYER's real weights: check vs torch, time in a graph, stamps."""
import os, sys, time, torch
sys.path.insert(0, os.path.dirname(__file__))
import vmoe_atom as VA
import vgdn1 as G
from safetensors import safe_open
torch.manual_seed(0); dev = "cuda"
D = "/data/Qwen3.6-35B-A3B-MXFP4/"; LY = int(os.environ.get("LAYER", 0)); REPS = int(os.environ.get("REPS", 300))
FP8 = torch.float8_e4m3fn
def q8(v):
    g = v.float().reshape(*v.shape[:-1], -1, 32)
    am = g.abs().amax(-1).clamp(1e-10, 448.0 * 2.0 ** 119)
    mant, ex = torch.frexp(am / 448.0); e = torch.where(mant > 0.5, ex, ex - 1)
    sc = torch.pow(2.0, e.float()).unsqueeze(-1)
    q = (g / sc).clamp(-448, 448).to(FP8)
    return q.reshape(v.shape), (e + 127).to(torch.uint8), (q.float() * sc).reshape(v.shape)
with safe_open(D + "layers-%d.safetensors" % LY, "pt") as f:
    g = lambda k: f.get_tensor("model.language_model.layers.%d.linear_attn.%s" % (LY, k)).to(dev)
    Win = torch.cat([g("in_proj_qkv.weight"), g("in_proj_z.weight"), g("in_proj_b.weight"), g("in_proj_a.weight")])
    Wout = g("out_proj.weight")
wi8, wis, wid = q8(Win); wo8, wos, wod = q8(Wout)
h = torch.randn(16, 2048, device=dev); x = (h * torch.rsqrt(h.pow(2).mean(1, keepdim=True) + 1e-6)).to(torch.bfloat16)
x8, xs, xd = q8(x)
h0 = (torch.randn(16, 2048, device=dev) * 0.5).to(torch.bfloat16)
z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=dev)
pk = VA._pk8
B = {}
B["Win"], B["WinS"] = pk(wi8, wis); B["Wout"], B["WoutS"] = pk(wo8, wos); B["xq8"], B["xqs8"] = pk(x8, xs)
B["proj"] = z(16, G.NPROJ, dt=torch.bfloat16); B["oq8"] = z(32 * 2048 + 1024, dt=torch.uint8); B["oqs8"] = z(32 * 64 + 1024, dt=torch.uint8)
B["h0"] = h0; B["h1"] = z(16, 2048, dt=torch.bfloat16)
B = {k: v.contiguous() for k, v in B.items()}
tab = torch.tensor([B[k].data_ptr() for k in G.TAB], dtype=torch.int64, device=dev)
cnt = torch.zeros(4096, dtype=torch.int32, device=dev); st = torch.zeros(2048, G.NSTAMP, dtype=torch.int64, device=dev)
ln = G.build()
ln(tab.data_ptr(), cnt.data_ptr(), st.data_ptr(), stream=torch.cuda.current_stream()); torch.cuda.synchronize()
bf = lambda v: v.to(torch.bfloat16).float()
pref = bf(xd @ wid.T)
o = bf(pref[:, 4096:8192] * torch.sigmoid(pref[:, 8192:12288]))
href = bf(h0.float() + bf(q8(o)[2] @ wod.T))
rel = lambda a, r: float((a.float() - r).norm() / r.norm())
print("layer %d: proj rel %.2e (q|k|v|z %.2e, b|a %.2e) | h1 rel %.2e, out part rel %.2e" % (
    LY, rel(B["proj"], pref), rel(B["proj"][:, :12288], pref[:, :12288]), rel(B["proj"][:, 12288:], pref[:, 12288:]),
    rel(B["h1"], href), rel(B["h1"].float() - h0.float(), href - h0.float())))
gph = torch.cuda.CUDAGraph()
with torch.cuda.graph(gph):
    cnt.zero_()
    ln(tab.data_ptr(), cnt.data_ptr(), st.data_ptr(), stream=torch.cuda.current_stream())
K = 20; TT = []
for rep in range(REPS):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(K): gph.replay()
    torch.cuda.synchronize(); TT.append((time.perf_counter() - t0) / K)
TT = sorted(TT[10:]); print("in a graph: %.1f us a launch (incl. counter zero), p10 %.1f p90 %.1f" % (1e6 * TT[len(TT) // 2], 1e6 * TT[len(TT) // 10], 1e6 * TT[9 * len(TT) // 10]))
names = {0: "entry", 1: "in_proj: x in LDS", 2: "in_proj: weights in", 3: "in_proj: stored + counted", 4: "verify stub: saw qdone", 5: "verify stub: stored", 6: "out_proj: saw odone", 7: "out_proj: x in", 8: "out_proj: h1 stored", 15: "exit"}
acc = {k: [] for k in names}; accm = {k: [] for k in names}
for rep in range(100):
    st.zero_(); gph.replay(); torch.cuda.synchronize()
    s = st.double(); t0 = s[:, 0][s[:, 0] > 0].min()
    for k in names:
        v = s[:, k]; v = v[v > 0]
        if len(v): acc[k].append(float(v.median() - t0) / 100); accm[k].append(float(v.max() - t0) / 100)
print("stamps, us after the first entry (median wave | last wave):")
for k in names:
    if acc[k]: print("  %-26s %6.1f | %6.1f" % (names[k], sum(acc[k]) / len(acc[k]), sum(accm[k]) / len(accm[k])))
print("counters:", cnt[0].item(), cnt[16].item(), cnt[32].item(), "expected", G.QC, G.VCN, G.OCN)
