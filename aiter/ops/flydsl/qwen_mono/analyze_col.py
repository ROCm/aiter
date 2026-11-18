"""One captured row: the routed experts' output at the massive column under different FP4 rules, against ATOM's value."""
import sys, torch, itertools
from safetensors import safe_open
d = torch.load(sys.argv[1], map_location="cpu"); LY, T = int(d["meta"][0]), int(d["meta"][1]); r = int(sys.argv[2]); COL = 1269
fl = lambda x: x.float()
TAB4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])
d4 = lambda q, s: TAB4[torch.stack([q & 15, q >> 4], -1).reshape(*q.shape[:-1], -1).long()] * torch.pow(2.0, s.float() - 127).repeat_interleave(32, -1)
G = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6])
def q4(v, scale_rule="up", rnd="rne", blk=32):
    vv = v.reshape(*v.shape[:-1], -1, blk); am = vv.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    if scale_rule == "up":
        mant, ex = torch.frexp(am / 6.0); e = torch.where(mant > 0.5, ex, ex - 1).float()
    elif scale_rule == "floor":                            # floor(log2(amax)) - 2: amax / scale in [4, 8): values above 6 saturate
        e = torch.floor(torch.log2(am)) - 2
    elif scale_rule == "even":                             # aiter _mxfp4_quant_op: amax rounded at mantissa 1.75 first
        ab = ((am.view(torch.int32) + 0x200000) & -8388608).view(torch.float32); e = torch.floor(torch.log2(ab)) - 2
    y = vv / torch.pow(2.0, e); ay = y.abs().clamp(max=6.0)
    if rnd == "rne":
        inv = torch.where(ay < 2, torch.full_like(ay, 2.0), torch.where(ay < 4, torch.ones_like(ay), torch.full_like(ay, 0.5))); q = (torch.round(ay * inv) / inv).clamp(max=6.0)
    elif rnd == "rha":                                     # ties away from zero
        inv = torch.where(ay < 2, torch.full_like(ay, 2.0), torch.where(ay < 4, torch.ones_like(ay), torch.full_like(ay, 0.5))); q = (torch.floor(ay * inv + 0.5) / inv).clamp(max=6.0)
    elif rnd == "rhd":                                     # ties toward zero
        inv = torch.where(ay < 2, torch.full_like(ay, 2.0), torch.where(ay < 4, torch.ones_like(ay), torch.full_like(ay, 0.5))); q = (torch.ceil(ay * inv - 0.5) / inv).clamp(max=6.0)
    elif rnd in ("hup", "hdn", "odd", "evenval"):
        inv = torch.where(ay < 2, torch.full_like(ay, 2.0), torch.where(ay < 4, torch.ones_like(ay), torch.full_like(ay, 0.5)))
        z = ay * inv; lo = torch.floor(z); tie = (z - lo) == 0.5
        up_ = lo + 1
        if rnd == "hup": pick_up = y > 0                   # ties toward +inf
        elif rnd == "hdn": pick_up = y < 0                 # ties toward -inf
        elif rnd == "odd": pick_up = (lo % 2) == 0         # ties to the odd step
        else:                                              # ties to the value with an even e2m1 code (0 .5 1 1.5 2 3 4 6 = codes 0..7)
            code_lo = torch.searchsorted(G, (lo / inv).clamp(max=6.0).contiguous()); pick_up = (code_lo % 2) == 1
        q = (torch.where(tie, torch.where(pick_up, up_, lo), torch.round(z)) / inv).clamp(max=6.0)
    elif rnd == "trunc":
        q = G[(ay.unsqueeze(-1) >= G).sum(-1) - 1]
    elif rnd == "up":
        q = G[((ay.unsqueeze(-1) > G).sum(-1)).clamp(max=7)]
    return (q * y.sign() * torch.pow(2.0, e)).reshape(v.shape), (y.abs().amax(-1)).reshape(-1)
with safe_open("/data/Qwen3.6-35B-A3B-MXFP4/layers-%d.safetensors" % LY, "pt") as f:
    g = lambda k: f.get_tensor("model.language_model.layers.%d.%s" % (LY, k))
    Wg = g("mlp.gate.weight").float(); x = d["x"][r]
    lg = (x @ Wg.T).to(torch.bfloat16).float(); order = torch.sort(-lg, stable=True).indices[:8]; p = torch.softmax(lg, 0); wt = p[order] / p[order].sum()
    EX = {n: [(g("mlp.experts.%d.%s_proj.weight" % (int(e), n)), g("mlp.experts.%d.%s_proj.weight_scale" % (int(e), n))) for e in order] for n in ("gate", "up", "down")}
print("layer %d row %d: ATOM routed[%d] = %.4f, kernel = %.4f; kernel top-8 %s, torch %s" % (LY, r, COL, float(d["nr"][r, COL]), float(d["kr"][r, COL]), sorted(d["sel"][r].long().tolist()), sorted(order.tolist())))
silu = torch.nn.functional.silu
xq0, _ = q4(x, "up", "rne"); yy = (x.view(-1, 32) / (xq0.view(-1, 32).abs().amax(-1, keepdim=True).clamp_min(1e-30))).abs()
print("  x: %d of 2048 elements differ between ties-to-even and ties-away" % int((q4(x, "up", "rne")[0] != q4(x, "up", "rha")[0]).sum()))
for xs, xr, as_, ar in list(itertools.product(("up",), ("rne", "rha", "rhd", "hup", "hdn", "odd", "evenval"), ("up",), ("rne",))):
    xq, _ = q4(x, xs, xr); tot = 0.0; parts = []
    for j in range(8):
        ge = xq @ d4(*EX["gate"][j]).T; ue = xq @ d4(*EX["up"][j]).T
        a = silu(ge) * ue; aq, ratio = q4(a, as_, ar)
        dn = d4(*EX["down"][j])[COL]; c = float(wt[j] * (aq @ dn)); tot += c; parts.append(c)
    print("  x %-5s %-5s act %-5s %-5s: routed[%d] = %8.4f   by expert %s" % (xs, xr, as_, ar, COL, tot, " ".join("%.3f" % c for c in parts)))
# the unquantized-activation value, and where the column's mass comes from
xq, _ = q4(x, "up", "rne"); j = max(range(8), key=lambda j_: abs(float(wt[j_] * ((silu(xq @ d4(*EX["gate"][j_]).T) * (xq @ d4(*EX["up"][j_]).T)) @ d4(*EX["down"][j_])[COL]))))
a = silu(xq @ d4(*EX["gate"][j]).T) * (xq @ d4(*EX["up"][j]).T); dn = d4(*EX["down"][j])[COL]; contrib = a * dn; k = int(contrib.abs().argmax())
print("  dominant expert rank %d: act f32 gives %.4f; top activation index %d (value %.3f, block max %.3f, weight %.4f, contributes %.4f)" % (j, float(wt[j] * contrib.sum()), k, float(a[k]), float(a.view(-1, 32)[k // 32].abs().max()), float(dn[k]), float(wt[j] * contrib[k])))
