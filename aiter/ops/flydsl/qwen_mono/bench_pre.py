import os, sys, json, torch
sys.path.insert(0, os.path.dirname(__file__))
import pre as P, stage as S
from safetensors import safe_open
torch.manual_seed(0); dev = "cuda"
R = int(os.environ.get("R", 16)); D = "/data/Qwen3.6-35B-A3B-MXFP4-experts-MXFP8-gemm-lmhead/"
cfg = json.load(open(D + "config.json")); tc = cfg.get("text_config", cfg)
prf = tc.get("partial_rotary_factor", 1.0) or 1.0; H = int(256 * prf) // 2
print("partial_rotary_factor", prf, "-> rotary pairs (i, i + %d)" % H, "rope_theta", tc.get("rope_theta"))
with safe_open(D + "mtp.safetensors", "pt") as f:
    g = lambda k: f.get_tensor("mtp." + k).to(dev)
    L = "layers.0."
    Wfc = g("fc.weight"); Wq, Wk, Wv = (g(L + "self_attn.%s_proj.weight" % n) for n in "qkv")
    ge, gh, gin = (1 + g(n).float() for n in ("pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight", L + "input_layernorm.weight"))
    gq, gk = (1 + g(L + "self_attn.%s_norm.weight" % n).float() for n in "qk")
with safe_open(D + "outside.safetensors", "pt") as f:
    emb = f.get_tensor("model.language_model.embed_tokens.weight").to(dev)
bf = lambda x: x.to(torch.bfloat16); fl = lambda x: x.float()
z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=dev)
ids = torch.randint(0, 248320, (16,), dtype=torch.int32, device=dev)
hid = bf(emb[torch.randint(0, 248320, (16,))].float() * 20 + torch.randn(16, 2048, device=dev) * 0.3)
pos = torch.randint(1000, 70000, (16,), device=dev).float()
ang = pos[:, None] * (float(tc.get("rope_theta") or 10000.0) ** (-torch.arange(H, device=dev).float() / H))[None]
Wqkv = torch.cat([Wq, Wk, Wv])
B = dict(ids=ids, emb=emb, hid=hid, ge=ge, gh=gh, Wfc=S.pack_w(Wfc), Wqkv=S.pack_w(Wqkv), gin=gin, gq=gq, gk=gk, cos=torch.cos(ang).contiguous(), sin=torch.sin(ang).contiguous(),
         xfc=z(128, 32, 16, dt=torch.bfloat16), resid=z(16, 2048, dt=torch.bfloat16), xg=z(64, 32, 16, dt=torch.bfloat16), part=z(16, 256),
         qkv=z(16, 9216, dt=torch.bfloat16), qo=z(16, 16, 256, dt=torch.bfloat16), ko=z(16, 2, 256, dt=torch.bfloat16))
tab = torch.tensor([B[k].data_ptr() for k in P.TAB], dtype=torch.int64, device=dev)
cnt = torch.zeros(4096, dtype=torch.int32, device=dev); st = torch.zeros(256, P.NSTAMP, dtype=torch.int64, device=dev)
ln = P.build(H)
def run():
    cnt.zero_(); ln(tab.data_ptr(), cnt.data_ptr(), st.data_ptr(), stream=torch.cuda.current_stream()); torch.cuda.synchronize()
run()
from atom.mono.runtime.compile import launcher_resources
r = launcher_resources(ln)[0]; print("vgpr %d sgpr %d scratch %d spills %s/%s" % (r["vgpr"], r["sgpr"], r["scratch"], r["vgpr_spill"], r["sgpr_spill"]))
rms = lambda x, gw: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * gw
rel = lambda a, b: float((fl(a) - fl(b)).abs().max() / fl(b).abs().max())
x = torch.cat([fl(bf(rms(fl(emb[ids.long()]), ge))), fl(bf(rms(fl(hid), gh)))], 1)
res = bf(x @ fl(Wfc).T); rs = torch.rsqrt(fl(res).pow(2).mean(1, keepdim=True) + 1e-6)
qkv = bf(rs * (fl(bf(fl(res) * gin)) @ fl(Wqkv).T))
def rope(v, gw):                                   # [16][heads][256]
    n = rms(fl(v), gw); o = n.clone(); c, s = B["cos"][:, None], B["sin"][:, None]
    o[..., :H] = n[..., :H] * c - n[..., H:2 * H] * s; o[..., H:2 * H] = n[..., H:2 * H] * c + n[..., :H] * s
    return bf(o)
qo = rope(qkv[:, :8192].view(16, 16, 512)[..., :256], gq); ko = rope(qkv[:, 8192:8704].view(16, 2, 256), gk)
print("fc input %.1e  residual %.1e  qkv %.1e  q (norm+rotary) %.1e  k %.1e" % (rel(S.unpack_x(B["xfc"], 4096), x), rel(B["resid"], res), rel(B["qkv"], qkv), rel(B["qo"], qo), rel(B["ko"], ko)))
rows = []
for _ in range(120):
    run(); rows.append((st[:, 1:5].max(0).values - st[:, 0].min()).float() / 100)
r = torch.stack(rows[20:]).median(0).values.tolist()
print("us: " + " | ".join("%s %.1f" % (n, r[i] - (r[i - 1] if i else 0)) for i, n in enumerate(["embed + pre-fc norms", "fc", "q/k/v", "q,k norm + rotary"])) + " | total %.1f" % r[-1])
