import sys, torch
sys.path.insert(0, "/home/vigokhal/m3_monokernel/bench")
from m3_inputs import make_layers
dev = torch.device("cuda")
lw = make_layers(1, 64, 0, dev)[0]
d = torch.load("bench/logs/sav/moein.pt")
hm = torch.load("bench/logs/sav/pure_L1_rows16.pt")["gluon"][1].to(dev).float()
x = d["x"].to(dev).float(); ids = d["ids"].to(dev); w = d["w"].to(dev)
rstd = torch.rsqrt((hm * hm).mean(-1, keepdim=True) + 1e-6)
xr = (hm * rstd * (1 + lw.g_post.float())).bfloat16().float()
print("xn nrmse %.2e max %.3e" % (((x - xr).norm() / xr.norm()).item(), (x - xr).abs().max().item()))
logit = (x @ lw.gate.float().T).bfloat16().float()
sig = torch.sigmoid(logit)
key = sig + lw.bias
top = key.topk(4, dim=-1)
ki, ks = top.indices, top.values
print("ids routed match:", torch.equal(ids[:, :4].sort(-1).values.long(), ki.sort(-1).values), "shared col", ids[:, 4].unique().tolist(), w[:, 4].unique().tolist())
sw = sig.gather(1, ki)
wr = sw / sw.sum(-1, keepdim=True) * 2.0
o = ki.argsort(-1)
mine = w[:, :4].gather(1, ids[:, :4].long().argsort(-1))
print("weights max abs diff", (mine - wr.gather(1, o)).abs().max().item(), "4th/5th key gap min", (top.values[:, 3] - key.topk(5, -1).values[:, 4]).min().item())
