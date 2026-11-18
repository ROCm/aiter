"""vmoe_atom.VMoE against a torch reference on random normed rows, layers LAYERS, rows ROWS (as ATOM would call it)."""
import os, sys, time, torch
sys.path.insert(0, os.path.dirname(__file__))
import vmoe_atom as VA
from safetensors import safe_open
torch.manual_seed(0); dev = "cuda"
D = os.environ.get("CKPT", "/data/Qwen3.6-35B-A3B-MXFP4/")
vm = VA.VMoE(D, torch.device(dev))
fl = lambda x: x.float(); bf = lambda x: x.to(torch.bfloat16)
TAB4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device=dev)
d4 = lambda q, s: TAB4[torch.stack([q & 15, q >> 4], -1).reshape(*q.shape[:-1], -1).long()] * torch.pow(2.0, s.float() - 127).repeat_interleave(32, -1)
d8 = lambda w, s: w.float() * torch.pow(2.0, s.float() - 127).repeat_interleave(32, -1)
def mx8(v, blk=32):                                # ATOM's MXFP8 1x32 rule: scale = 2^ceil(log2(amax / 448)), nothing clamped
    am = v.view(*v.shape[:-1], -1, blk).abs().amax(-1).clamp(1e-10, 448.0 * 6.646139978924579e35)
    mant, ex = torch.frexp(am / 448.0); code = torch.where(mant > 0.5, ex, ex - 1).float() + 127
    sc = torch.pow(2.0, code - 127).repeat_interleave(blk, -1); return (v / sc).clamp(-448, 448).to(torch.float8_e4m3fn).float() * sc
def rmx4(v, blk=32):                               # ATOM's A4: e8m0 = RoundUp(amax / 6), e2m1 values, round to nearest even, saturating
    vv = v.reshape(*v.shape[:-1], -1, blk)
    am = vv.abs().amax(-1, keepdim=True).clamp(1e-10, 6.0 * 6.646139978924579e35)
    mant, ex = torch.frexp(am / 6.0); e = torch.where(mant > 0.5, ex, ex - 1).float()
    y = vv / torch.pow(2.0, e); ay = y.abs().clamp(max=6.0)
    inv = torch.where(ay < 2, torch.full_like(ay, 2.0), torch.where(ay < 4, torch.ones_like(ay), torch.full_like(ay, 0.5)))
    q = (torch.round(ay * inv) / inv).clamp(max=6.0)        # torch.round: half to even
    return (q * y.sign() * torch.pow(2.0, e)).reshape(v.shape)
def top8(lg):                                       # ATOM: the bf16 router output, largest first, ties to the lower expert index
    lg16 = lg.to(torch.bfloat16).float()
    order = torch.sort(-lg16, dim=1, stable=True).indices[:, :8]
    p = torch.softmax(lg16, 1); wt = p.gather(1, order)
    return lg16, p, wt / wt.sum(1, keepdim=True), order
silu = torch.nn.functional.silu
with safe_open(D + "outside.safetensors", "pt") as f:
    emb = f.get_tensor("model.language_model.embed_tokens.weight").to(dev)
SB = {}
for LY in [int(v) for v in os.environ.get("LAYERS", "0,17").split(",")]:
    with safe_open("/data/Qwen3.6-35B-A3B-MXFP4/layers-%d.safetensors" % LY, "pt") as f:
        SB[LY] = {n: f.get_tensor("model.language_model.layers.%d.mlp.shared_expert.%s_proj.weight" % (LY, n)).to(dev) for n in ("gate", "up", "down")}
for LY in [int(v) for v in os.environ.get("LAYERS", "0,17").split(",")]:
    t = time.time(); vm.load_layer(LY); t = time.time() - t
    with safe_open(D + "layers-%d.safetensors" % LY, "pt") as f:
        g = lambda k: f.get_tensor("model.language_model.layers.%d.%s" % (LY, k)).to(dev)
        Wg, seg = g("mlp.gate.weight"), g("mlp.shared_expert_gate.weight")
        SH = {n: (g("mlp.shared_expert.%s_proj.weight" % n), None if VA.V.SHB else g("mlp.shared_expert.%s_proj.weight_scale" % n)) for n in ("gate", "up", "down")}
        EX = {n: (torch.stack([g("mlp.experts.%d.%s_proj.weight" % (e, n)) for e in range(256)]), torch.stack([g("mlp.experts.%d.%s_proj.weight_scale" % (e, n)) for e in range(256)])) for n in ("gate", "up", "down")}
    for R in [int(v) for v in os.environ.get("ROWS", "1,3,12,16").split(",")]:
        h = emb[torch.randint(0, 248320, (R,))].float() * 20 + torch.randn(R, 2048, device=dev) * 0.2
        x = bf(h * torch.rsqrt(h.pow(2).mean(1, keepdim=True) + 1e-6)).contiguous()
        out = vm.run(LY, x).clone(); torch.cuda.synchronize()
        if R >= 3 and os.environ.get("NANROW", "1") == "1":       # a pad row of NaN / inf must not disturb the other rows (nor stall the launch)
            import time as _t
            xb = x.clone(); xb[R - 1] = float("nan"); xb[R - 1, ::2] = float("inf"); t0 = _t.time()
            ob = vm.run(LY, xb).clone(); torch.cuda.synchronize()
            print("   NaN/inf pad row: other rows equal %s, finite %s, launch %.3f s" % (bool((ob[:R - 1] == out[:R - 1]).all()), bool(torch.isfinite(ob.float()).all()), _t.time() - t0))
        m = fl(x); mq = mx8(m); m4 = rmx4(m); lg = m @ fl(Wg).T; p = torch.softmax(lg, 1); wt, sel = p.topk(8, 1); wt = wt / wt.sum(1, keepdim=True)
        if VA.V.A4: lg, p, wt, sel = top8(lg)
        gate = torch.sigmoid(m @ fl(seg).T)
        if VA.V.SHB:
            sh = fl(bf(silu(m @ fl(SH["gate"][0]).T) * (m @ fl(SH["up"][0]).T))) @ fl(SH["down"][0]).T
        else:
            sh = mx8(silu(fl(bf(mq @ d8(*SH["gate"]).T))) * fl(bf(mq @ d8(*SH["up"]).T))) @ d8(*SH["down"]).T
        moe = torch.zeros(R, 2048, device=dev)
        for tk in range(R):
            for j in range(8):
                e = int(sel[tk, j])
                if VA.V.A4:                            # ATOM's A4W4 experts
                    ge = m4[tk] @ d4(*[v[e] for v in EX["gate"]]).T; ue = m4[tk] @ d4(*[v[e] for v in EX["up"]]).T
                    moe[tk] += wt[tk, j] * (rmx4(silu(ge) * ue) @ d4(*[v[e] for v in EX["down"]]).T)
                else:
                    ge = fl(bf(mq[tk] @ d4(*[v[e] for v in EX["gate"]]).T)); ue = fl(bf(mq[tk] @ d4(*[v[e] for v in EX["up"]]).T))
                    moe[tk] += mx8(wt[tk, j] * silu(ge) * ue) @ d4(*[v[e] for v in EX["down"]]).T
        ref = moe + gate * sh
        if LY in SB:                                   # ATOM's chain as served: the bf16 shared expert on the bf16 x
            shb = (silu(m @ fl(SB[LY]["gate"]).T) * (m @ fl(SB[LY]["up"]).T)) @ fl(SB[LY]["down"]).T
            refn = moe + gate * shb
            print("   vs the served (bf16) shared expert: rel %.1e | shared part / total norm %.2f | MXFP8 shared vs bf16 shared rel %.1e" % (
                float((fl(out) - refn).norm() / refn.norm()), float((gate * shb).norm() / refn.norm()), float((sh - shb).norm() / shb.norm())))
        print("layer %2d rows %2d: out rel err (norm) %.1e  max abs err / max %.1e  (load %.1f s)" % (LY, R, float((fl(out) - ref).norm() / ref.norm()), float((fl(out) - ref).abs().max() / ref.abs().max()), t))
    del EX
print("GPU memory for %d layers: %.1f GB" % (len(vm.layers), torch.cuda.memory_allocated() / 2**30))
