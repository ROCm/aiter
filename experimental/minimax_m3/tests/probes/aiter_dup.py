import sys, torch
sys.path.insert(0, "/home/vigokhal/m3_monokernel/bench")
from m3_inputs import make_layers
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
dev = torch.device("cuda")
lw = make_layers(1, 64, 0, dev)[0]
d = torch.load("bench/logs/sav/moein48.pt")
x, ids, w = d["x"].to(dev), d["ids"].to(dev), d["w"].to(dev)
def fp4(t):
    v = t.view(torch.float4_e2m1fn_x2); v.is_shuffled = True; return v
w13, w2 = fp4(lw.w13), fp4(lw.w2)
outs = []
for i in range(3):
    o = fused_moe(x, w13, w2, w, ids, None, ActivationType.Swiglu, QuantType.per_1x32, False, lw.s13, lw.s2, None, None,
                  dtype=torch.bfloat16, swiglu_limit=7.0)
    outs.append(o.float())
o = outs[0]
for g in (1, 2):
    y, z = o[16 * g:16 * g + 16], o[:16]
    print("group", g, "exact", torch.equal(y, z), "nrmse %.2e" % ((y - z).norm() / z.norm()).item())
print("run-to-run exact", torch.equal(outs[0], outs[1]), torch.equal(outs[1], outs[2]),
      "nrmse %.2e" % ((outs[0] - outs[1]).norm() / outs[0].norm()).item())
