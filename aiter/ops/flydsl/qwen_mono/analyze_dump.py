"""Which arithmetic does ATOM's routed-expert path use? Dumped server rows (x, native routed sum nr, kernel routed sum kr)
against torch references built under different hypotheses."""
import sys, torch
from safetensors import safe_open
D = torch.load(sys.argv[1], map_location="cpu")
CK = "/data/Qwen3.6-35B-A3B-MXFP4/"
fl = lambda x: x.float(); bf = lambda x: x.to(torch.bfloat16)
TAB4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])
d4 = lambda q, s: TAB4[torch.stack([q & 15, q >> 4], -1).reshape(*q.shape[:-1], -1).long()] * torch.pow(2.0, s.float() - 127).repeat_interleave(32, -1)
def mx8(v, blk=32, rule="up"):
    am = v.reshape(*v.shape[:-1], -1, blk).abs().amax(-1).clamp(1e-10, 448.0 * 6.646139978924579e35)
    if rule == "up":
        mant, ex = torch.frexp(am / 448.0); code = torch.where(mant > 0.5, ex, ex - 1).float()
    else:
        code = torch.floor(torch.log2(am)) - 8
    sc = torch.pow(2.0, code).repeat_interleave(blk, -1); return (v / sc).clamp(-448, 448).to(torch.float8_e4m3fn).float() * sc
silu = torch.nn.functional.silu
rel = lambda a, b: float((a - b).norm() / b.norm())
W = {}
for d in D[:int(sys.argv[2]) if len(sys.argv) > 2 else 6]:
    LY = d["layer"]
    if LY not in W:
        with safe_open(CK + "layers-%d.safetensors" % LY, "pt") as f:
            g = lambda k: f.get_tensor("model.language_model.layers.%d.%s" % (LY, k))
            W[LY] = dict(Wg=g("mlp.gate.weight"), EX={n: [(g("mlp.experts.%d.%s_proj.weight" % (e, n)), g("mlp.experts.%d.%s_proj.weight_scale" % (e, n))) for e in range(256)] for n in ("gate", "up", "down")})
    w = W[LY]; x = fl(d["x"]); T = x.shape[0]; nr, kr = d["nr"], d["kr"]
    lg = x @ fl(w["Wg"]).T; p = torch.softmax(lg, 1); wt, sel = p.topk(8, 1); wtn = wt / wt.sum(1, keepdim=True)
    def routed(xq, actq, weight_first=True, round_gu=True, renorm=True, wd=None):
        out = torch.zeros(T, 2048)
        for t in range(T):
            for j in range(8):
                e = int(sel[t, j]); wgt = (wtn if renorm else wt)[t, j]
                ge = xq[t] @ d4(*w["EX"]["gate"][e]).T; ue = xq[t] @ d4(*w["EX"]["up"][e]).T
                if round_gu: ge, ue = fl(bf(ge)), fl(bf(ue))
                a = silu(ge) * ue
                if weight_first: a = wgt * a
                a = actq(a)
                o = a @ d4(*w["EX"]["down"][e]).T
                out[t] += o if weight_first else wgt * o
        return out
    idn = lambda v: v
    def rmx4(v, blk=32):                                   # FlyDSL kernels: e8m0 = RoundUp(amax / 6), elements to e2m1 round-to-nearest-even, saturating
        vv = v.reshape(*v.shape[:-1], -1, blk)
        am = vv.abs().amax(-1, keepdim=True).clamp_min(1e-30)
        mant, ex = torch.frexp(am / 6.0); e = torch.where(mant > 0.5, ex, ex - 1).float()
        y = vv / torch.pow(2.0, e); ay = y.abs().clamp(max=6.0)
        G = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6]); dist = (ay.unsqueeze(-1) - G).abs()
        i0 = dist.argmin(-1); d0 = dist.gather(-1, i0.unsqueeze(-1)).squeeze(-1)
        i1 = (i0 + 1).clamp(max=7); tie = (dist.gather(-1, i1.unsqueeze(-1)).squeeze(-1) == d0) & (i1 != i0)
        idx = torch.where(tie & (i0 % 2 == 1), i1, i0)
        return (G[idx] * y.sign() * torch.pow(2.0, e)).reshape(v.shape)
    def amx4(v, blk=32):                                   # aiter's _mxfp4_quant_op: amax rounded at mantissa 1.75, floor(log2) - 2, RNE, saturate at 6
        vv = v.reshape(*v.shape[:-1], -1, blk)
        am = vv.abs().amax(-1, keepdim=True)
        ab = ((am.view(torch.int32) + 0x200000) & -8388608).view(torch.float32)
        e = (torch.floor(torch.log2(ab.clamp_min(1e-38))) - 2).clamp(-127, 127)
        y = vv / torch.pow(2.0, e); ay = y.abs()
        G = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6]); dist = (ay.unsqueeze(-1) - G).abs()
        i0 = dist.argmin(-1); d0 = dist.gather(-1, i0.unsqueeze(-1)).squeeze(-1)
        i1 = (i0 + 1).clamp(max=7); tie = (dist.gather(-1, i1.unsqueeze(-1)).squeeze(-1) == d0) & (i1 != i0)
        idx = torch.where(tie & (i0 % 2 == 1), i1, i0)       # ties to the even code
        q = torch.where(ay >= 6, torch.full_like(ay, 6.0), G[idx]) * y.sign()
        return (q * torch.pow(2.0, e)).reshape(v.shape)
    GRID = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6])
    def mx4(v, blk=32, shift=2, up=False):                 # MXFP4 1x32: e2m1 values, one power-of-two scale a group
        am = v.reshape(*v.shape[:-1], -1, blk).abs().amax(-1).clamp_min(1e-30)
        if up:
            mant, ex = torch.frexp(am / 6.0); code = torch.where(mant > 0.5, ex, ex - 1).float()
        else:
            code = torch.floor(torch.log2(am)) - shift
        sc = torch.pow(2.0, code).repeat_interleave(blk, -1); y = (v / sc).clamp(-6, 6)
        q = GRID[(y.abs().unsqueeze(-1) - GRID).abs().argmin(-1)] * y.sign()
        return q * sc
    H = {"kernel math (x MXFP8 up, act MXFP8 up of w*silu*u)": routed(mx8(x), mx8),
         "x bf16 (not quantized)": routed(x, mx8),
         "activation not quantized": routed(mx8(x), idn),
         "x bf16 and activation not quantized": routed(x, idn),
         "activation quantized before the route weight": routed(mx8(x), mx8, weight_first=False),
         "floor scale rule on x and act": routed(mx8(x, rule="floor"), lambda v: mx8(v, rule="floor")),
         "no top-8 renormalisation": routed(mx8(x), mx8, renorm=False),
         "x MXFP4 (floor - 2), act MXFP8": routed(mx4(x), mx8),
         "x MXFP4 (floor - 2), act MXFP4": routed(mx4(x), mx4),
         "x MXFP4 (round up), act MXFP8": routed(mx4(x, up=True), mx8),
         "x MXFP4 (round up), act MXFP4 (round up)": routed(mx4(x, up=True), lambda v: mx4(v, up=True)),
         "x MXFP8, act MXFP4 (floor - 2)": routed(mx8(x), mx4),
         "FlyDSL rule: x FP4 RoundUp RNE, act FP4 RoundUp RNE unweighted, f32 sums": routed(rmx4(x), rmx4, weight_first=False, round_gu=False),
         "same, gate/up rounded to bf16": routed(rmx4(x), rmx4, weight_first=False, round_gu=True),
         "same, f32 sums, act not quantized": routed(rmx4(x), idn, weight_first=False, round_gu=False),
         "aiter MXFP4 x, act f32": routed(amx4(x), idn),
         "aiter MXFP4 x, aiter MXFP4 act (weighted)": routed(amx4(x), amx4),
         "aiter MXFP4 x, aiter MXFP4 act, weight after down": routed(amx4(x), amx4, weight_first=False),
         "aiter MXFP4 x, MXFP8 act, weight after down": routed(amx4(x), mx8, weight_first=False),
         "aiter MXFP4 x, aiter MXFP4 act, weight after, no bf16 round": routed(amx4(x), amx4, weight_first=False, round_gu=False)}
    print("layer %d, %d rows: kernel vs native %.3f | finite x %s | native |nr| %.3e, kernel |kr| %.3e" % (LY, T, rel(kr, nr), bool(torch.isfinite(x).all()), float(nr.norm()), float(kr.norm())))
    for k, v in H.items():
        print("   %-48s vs native %.3f   vs kernel %.3f" % (k, rel(v, nr), rel(v, kr)))
    same = [set(sel[t].tolist()) == set(d["sel"][t].tolist()) for t in range(T)]
    print("   top-8 sets: torch vs kernel equal %d/%d; x abs max %.1f, x row norms %s" % (sum(same), T, float(x.abs().max()), [round(float(v), 1) for v in x.norm(dim=1)[:4]]))
