import os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
import vmoe as V, stage as S, draft_mono as DM
from safetensors import safe_open
torch.manual_seed(0); dev = "cuda"; R = int(os.environ.get("R", 12)); LY = int(os.environ.get("LAYER", 0))
D = "/data/Qwen3.6-35B-A3B-MXFP4-experts-MXFP8-gemm-lmhead/"
with safe_open(D + "layers-%d.safetensors" % LY, "pt") as f:
    g = lambda k: f.get_tensor("model.language_model.layers.%d.%s" % (LY, k)).to(dev)
    Wg, seg, w1 = g("mlp.gate.weight"), g("mlp.shared_expert_gate.weight"), g("post_attention_layernorm.weight").float()
    SH = {n: (g("mlp.shared_expert.%s_proj.weight" % n), g("mlp.shared_expert.%s_proj.weight_scale" % n)) for n in ("gate", "up", "down")}
    EX = {n: (torch.stack([g("mlp.experts.%d.%s_proj.weight" % (e, n)) for e in range(256)]), torch.stack([g("mlp.experts.%d.%s_proj.weight_scale" % (e, n)) for e in range(256)])) for n in ("gate", "up", "down")}
with safe_open(D + "outside.safetensors", "pt") as f:
    emb = f.get_tensor("model.language_model.embed_tokens.weight").to(dev)
bf = lambda x: x.to(torch.bfloat16); fl = lambda x: x.float()
z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=dev)
TAB4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device=dev)
d4 = lambda q, s: TAB4[torch.stack([q & 15, q >> 4], -1).reshape(*q.shape[:-1], -1).long()] * torch.pow(2.0, s.float() - 127).repeat_interleave(32, -1)
d8 = lambda w, s: w.float() * torch.pow(2.0, s.float() - 127).repeat_interleave(32, -1)
def pk8(w8, ws):                                   # [N][K] fp8, [N][K/32] -> [N/16][K/128][lane][64 B], [N/16][K/128][16][4]
    n, k = w8.shape
    return (w8.view(torch.uint8).view(n // 16, 16, k // 128, 4, 2, 16).permute(0, 2, 4, 1, 3, 5).contiguous(),
            torch.cat([ws.view(n // 16, 16, k // 128, 4).permute(0, 2, 1, 3).reshape(-1), torch.zeros(1024, dtype=torch.uint8, device=dev)]))
SW, SWS = pk8(torch.cat([SH["gate"][0], SH["up"][0]]), torch.cat([SH["gate"][1], SH["up"][1]])); SD, SDS = pk8(*SH["down"])
EW, ES = DM._pk4(torch.cat([EX["gate"][0], EX["up"][0]], 1), torch.cat([EX["gate"][1], EX["up"][1]], 1)); ED, EDS = DM._pk4(*EX["down"])
h1 = z(16, 2048, dt=torch.bfloat16); hr = bf(emb[torch.randint(0, 248320, (R,))].float() * 20 + torch.randn(R, 2048, device=dev) * 0.2)
h1[:R] = hr; h1[R:] = hr[R - 1]
xg = bf(fl(h1) * (1 + w1)); part = z(16, 256); part[:, 0] = fl(h1).pow(2).sum(1)
B = dict(Wg=S.pack_w(Wg), Wsg=S.pack_w(torch.cat([seg, z(15, 2048, dt=torch.bfloat16)])), SW=SW, SWS=SWS, EW=EW, ES=ES, ED=ED, EDS=EDS, SD=SD, SDS=SDS,
         h1=h1, xg1=S.pack_x(xg), xg1p=xg.contiguous(), part1=part, lg=z(16, 256), sgate=z(16), sel=z(16, 8, dt=torch.int32), wt=z(16, 8),
         GE=z(128, 4, 32, 64, dt=torch.uint8), GS=z(128, 4, 32, 4, dt=torch.int32), GSU=z(4, 32, 64, dt=torch.uint8), GSS=z(4, 32, 4, dt=torch.int32),
         h2=z(16, 2048, dt=torch.bfloat16))
tab = torch.tensor([B[k].data_ptr() for k in V.TAB], dtype=torch.int64, device=dev)
cnt = torch.zeros(4096, dtype=torch.int32, device=dev); st = torch.zeros(256, V.NSTAMP, dtype=torch.int64, device=dev)
ln = V.build()
def run():
    cnt.zero_(); ln(tab.data_ptr(), cnt.data_ptr(), st.data_ptr(), stream=torch.cuda.current_stream()); torch.cuda.synchronize()
run()
# reference with the kernel's rounding points
def mx8(v, blk=32):
    am = v.view(*v.shape[:-1], -1, blk).abs().amax(-1).clamp_min(1e-30); code = (torch.floor(torch.log2(am)) - 8 + 127).clamp(1, 254)
    sc = torch.pow(2.0, code - 127).repeat_interleave(blk, -1); return (v / sc).clamp(-448, 448).to(torch.float8_e4m3fn).float() * sc
silu = torch.nn.functional.silu
m = fl(xg); rs = torch.rsqrt(fl(h1).pow(2).mean(1, keepdim=True) + 1e-6); mq = mx8(m * rs)
lg = rs * (m @ fl(Wg).T); p = torch.softmax(lg, 1); wt, sel = p.topk(8, 1); wt = wt / wt.sum(1, keepdim=True)
gate = torch.sigmoid(rs * (m @ fl(seg).T))
sh = mx8(silu(fl(bf(mq @ d8(*SH["gate"]).T))) * fl(bf(mq @ d8(*SH["up"]).T))) @ d8(*SH["down"]).T
moe = torch.zeros(16, 2048, device=dev)
for t in range(R):
    for j in range(8):
        e = int(sel[t, j]); ge = fl(bf(mq[t] @ d4(*[x[e] for x in EX["gate"]]).T)); ue = fl(bf(mq[t] @ d4(*[x[e] for x in EX["up"]]).T))
        moe[t] += mx8(wt[t, j] * silu(ge) * ue) @ d4(*[x[e] for x in EX["down"]]).T
h2 = bf(fl(h1) + moe + gate * sh)
rel = lambda a, b: float((fl(a) - fl(b)).abs().max() / fl(b).abs().max().clamp_min(1e-30))
selk = B["sel"][:R].long(); dm = (fl(B["h2"][:R]) - fl(h1[:R])) - (moe + gate * sh)[:R]
print("layer %d R %d: logits %.1e  top-8 sets equal %d/%d  weights %.1e  h2 %.1e  MoE part rel err %.1e  U %d" % (LY, R, rel(B["lg"][:R], lg[:R]),
      sum(set(selk[t].tolist()) == set(sel[t].tolist()) for t in range(R)), R, rel(B["wt"][:R], wt[:R]), rel(B["h2"][:R], h2[:R]),
      float(dm.norm() / (moe + gate * sh)[:R].norm()), len(set(selk.flatten().tolist()))))
rows = []
for _ in range(150):
    run(); rows.append((st[:, 1:5].max(0).values - st[:, 0].min()).float() / 100)
r = torch.stack(rows[20:]).median(0).values.tolist()
print("us: start %.1f | F1 router+top-8, shared up, quant %.1f | F2 experts gate/up %.1f | F3 down+combine %.1f | F1-F3 %.1f" % (r[0], r[1] - r[0], r[2] - r[1], r[3] - r[2], r[3] - r[0]))
