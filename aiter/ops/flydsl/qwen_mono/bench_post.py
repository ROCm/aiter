import os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
import post as P, stage as S, head8 as H8, draft_mono as DM
torch.manual_seed(0); dev = "cuda"
R = int(os.environ.get("R", 16)); D = "/data/Qwen3.6-35B-A3B-MXFP4-experts-MXFP8-gemm-lmhead/"
m_ = DM.DraftMono("/data/Qwen3.6-35B-A3B-MXFP4", D, stamps=True); B = m_.B; st = m_.st
Wo, Wg, sgp, sup, sdn, seg, w1, wf = (m_.ref[k] for k in "Wo Wg sgp sup sdn seg w1 wf".split()); EG, EU, EDn = m_.ref["E"]; w8, ws = m_.ref["w8"], m_.ref["ws"]
from safetensors import safe_open
with safe_open(D + "outside.safetensors", "pt") as f_:
    emb = f_.get_tensor("model.language_model.embed_tokens.weight").to(dev)
bf = lambda x: x.to(torch.bfloat16)
TAB4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device=dev)
def dq(w):
    q, s = DM._q4(w); return TAB4[torch.stack([q & 15, q >> 4], -1).reshape(w.shape).long()] * torch.pow(2.0, s.float() - 127).repeat_interleave(32, -1)
EGd, EUd, EDd = dq(EG), dq(EU), dq(EDn)
# inputs as ATOM gives them: R gated attention rows, a residual with more rows and a row list
xin = bf(torch.randn(R, 4096, device=dev) * 0.5); rall = bf(emb[torch.randint(0, 248320, (4 * R,))].float() * 20)
rowl = (torch.arange(R, device=dev) * 4 + 3).contiguous()
x0 = torch.zeros(16, 4096, dtype=torch.bfloat16, device=dev); x0[:R] = xin; res = torch.zeros(16, 2048, dtype=torch.bfloat16, device=dev); res[:R] = rall[rowl]
def run():
    global hout, ids
    hout, ids = m_.run(xin, rall, rowl); torch.cuda.synchronize()
run()
# ---- reference with the kernel's rounding points
f = lambda x: x.float()
rel = lambda a, b: float((f(a) - f(b)).abs().max() / f(b).abs().max().clamp_min(1e-30))
h1 = bf(f(bf(f(x0) @ f(Wo).T)) + f(res)); rs1 = torch.rsqrt(f(h1).pow(2).mean(1, keepdim=True) + 1e-6)
m = f(bf(f(h1) * (1 + w1)))
lg = rs1 * (m @ f(Wg).T); sg = f(bf(rs1 * (m @ f(sgp).T))); su = f(bf(rs1 * (m @ f(sup).T))); gate = torch.sigmoid(rs1 * (m @ f(seg).T))
p = torch.softmax(lg, 1); wt, sel = p.topk(8, 1); wt = wt / wt.sum(1, keepdim=True)
moe = f(bf(gate * f(bf(torch.nn.functional.silu(sg) * su)))) @ f(sdn).T
def mx8(v, blk):
    am = v.view(*v.shape[:-1], -1, blk).abs().amax(-1).clamp_min(1e-30); code = (torch.floor(torch.log2(am)) - 8 + 127).clamp(1, 254)
    sc = torch.pow(2.0, code - 127).repeat_interleave(blk, -1); return (v / sc).clamp(-448, 448).to(torch.float8_e4m3fn).float() * sc
mq = mx8(f(bf(f(h1) * (1 + w1))) * rs1, 32)
for t in range(R):
    for j in range(8):
        e = int(sel[t, j]); ge = f(bf(mq[t] @ EGd[e].T)); ue = f(bf(mq[t] @ EUd[e].T))
        moe[t] += mx8(wt[t, j] * torch.nn.functional.silu(ge) * ue, 32) @ EDd[e].T
h2 = bf(f(h1) + moe); rs2 = torch.rsqrt(f(h2).pow(2).mean(1, keepdim=True) + 1e-6)
xf = f(bf(f(h2) * (1 + wf))); v = xf * rs2
am = v.view(16, 64, 32).abs().amax(2).clamp_min(1e-30); code = (torch.floor(torch.log2(am)) - 8 + 127).clamp(1, 254)
sc = torch.pow(2.0, code - 127).repeat_interleave(32, 1); vq = (v / sc).clamp(-448, 448).to(torch.float8_e4m3fn).float() * sc
Wf = w8.float() * torch.pow(2.0, ws.float() - 127).repeat_interleave(32, 1)
ref_tok = bf(vq @ Wf.T).float().argmax(1); tok_unq = (f(bf(f(h2) * (1 + wf) * rs2)) @ Wf.T).argmax(1)
tok = ids.clone(); tok = torch.cat([tok, torch.zeros(16 - R, dtype=torch.int64, device=dev)])
print("hidden out vs bf16(xf rstd): %.1e" % rel(hout, v[:R]))
selk = B["sel"][:R].long()
print("R %d  h1 %.1e  logits %.1e  top-8 sets equal %d/%d  weights %.1e  xf %.1e  U %d" % (R, rel(B["h1"][:R], h1[:R]), rel(B["lg"][:R], lg[:R]),
      sum(set(selk[t].tolist()) == set(sel[t].tolist()) for t in range(R)), R, rel(B["wt"][:R], wt[:R]), rel(B["xf"][:R], xf[:R]), len(set(selk.flatten().tolist()))))  # (rows past R repeat the last row)
print("tokens vs reference %d/%d (vs unquantized head input %d/%d)" % (int((tok[:R] == ref_tok[:R]).sum()), R, int((tok[:R] == tok_unq[:R]).sum()), R), tok[:4].tolist(), ref_tok[:4].tolist())
rows = []
for _ in range(120):
    run(); rows.append((st[:, 1:9].max(0).values - st[:, 0].min()).float() / 100)
r = torch.stack(rows[20:]).median(0).values.tolist()
names = ["start+handoff", "o_proj+res+norm", "router+shared up", "top-8", "experts gate/up", "down+combine+norm", "rstd+quant x", "head+argmax"]
print("us: " + " | ".join("%s %.1f" % (n, r[i] - (r[i - 1] if i else 0)) for i, n in enumerate(names)) + " | total %.1f" % r[-1])
