import os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
import draft_mono as DM
from safetensors import safe_open
torch.manual_seed(0); dev = "cuda"; T = int(os.environ.get("T", 12)); D = "/data/Qwen3.6-35B-A3B-MXFP4/"
with safe_open(D + "mtp.safetensors", "pt") as f:
    g = lambda k: f.get_tensor("mtp." + k).to(dev); L = "layers.0."
    Wfc = g("fc.weight"); Wqkv = torch.cat([g(L + "self_attn.%s_proj.weight" % n) for n in "qkv"])
    we, wh, win = g("pre_fc_norm_embedding.weight"), g("pre_fc_norm_hidden.weight"), g(L + "input_layernorm.weight")
with safe_open(D + "outside.safetensors", "pt") as f:
    emb = f.get_tensor("model.language_model.embed_tokens.weight").to(dev)
m = DM.DraftPre(emb, we, wh, Wfc, win, Wqkv, dev)
ids = torch.randint(0, 248320, (T,), device=dev).to(torch.int32 if os.environ.get('I32') else torch.int64); hid = (emb[torch.randint(0, 248320, (T,))].float() * 20).to(torch.bfloat16)
bf = lambda x: x.to(torch.bfloat16); fl = lambda x: x.float()
rms = lambda x, w: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * (1 + w.float())
res, qkv = m.run(ids, hid); torch.cuda.synchronize()
x = torch.cat([fl(bf(rms(fl(emb[ids.long()]), we))), fl(bf(rms(fl(hid), wh)))], 1)
r = bf(x @ fl(Wfc).T); rs = torch.rsqrt(fl(r).pow(2).mean(1, keepdim=True) + 1e-6); q = bf(rs * (fl(bf(fl(r) * (1 + win.float()))) @ fl(Wqkv).T))
rel = lambda a, b: float((fl(a) - fl(b)).abs().max() / fl(b).abs().max())
print("T %d  residual %.1e  qkv %.1e" % (T, rel(res, r), rel(qkv, q)))
rows = []
for _ in range(120):
    m.run(ids, hid); torch.cuda.synchronize(); rows.append((m.st[:, 1:4].max(0).values - m.st[:, 0].min()).float() / 100)
t = torch.stack(rows[20:]).median(0).values.tolist()
print("us: embed+norms %.1f | fc %.1f | q/k/v %.1f | total %.1f" % (t[0], t[1] - t[0], t[2] - t[1], t[2]))
