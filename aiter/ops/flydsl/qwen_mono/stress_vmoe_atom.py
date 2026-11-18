"""Stress: many row sets with widely varying expert unions (duplicated / near-duplicate rows of dumped server inputs and
synthetic rows) through vmoe_atom.VMoE; every output is compared with a torch reference and every launch is timed
(a missed handoff shows as a 2 s poll timeout)."""
import os, sys, time, torch
sys.path.insert(0, os.path.dirname(__file__))
import vmoe_atom as VA
from safetensors import safe_open
torch.manual_seed(int(os.environ.get("SEED", 0))); dev = "cuda"
D = "/data/Qwen3.6-35B-A3B-MXFP4/"; LY = int(os.environ.get("LAYER", 17)); N = int(os.environ.get("N", 300))
vm = VA.VMoE(D, torch.device(dev)); vm.load_layer(LY)
fl = lambda x: x.float(); bf = lambda x: x.to(torch.bfloat16)
TAB4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device=dev)
d4 = lambda q, s: TAB4[torch.stack([q & 15, q >> 4], -1).reshape(*q.shape[:-1], -1).long()] * torch.pow(2.0, s.float() - 127).repeat_interleave(32, -1)
d8 = lambda w, s: w.float() * torch.pow(2.0, s.float() - 127).repeat_interleave(32, -1)
def mx8(v, blk=32):
    am = v.view(*v.shape[:-1], -1, blk).abs().amax(-1).clamp(1e-10, 448.0 * 6.646139978924579e35)
    mant, ex = torch.frexp(am / 448.0); code = torch.where(mant > 0.5, ex, ex - 1).float()
    sc = torch.pow(2.0, code).repeat_interleave(blk, -1); return (v / sc).clamp(-448, 448).to(torch.float8_e4m3fn).float() * sc
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
with safe_open(D + "layers-%d.safetensors" % LY, "pt") as f:
    g = lambda k: f.get_tensor("model.language_model.layers.%d.%s" % (LY, k)).to(dev)
    Wg, seg = g("mlp.gate.weight"), g("mlp.shared_expert_gate.weight")
    SH = {n: fl(g("mlp.shared_expert.%s_proj.weight" % n)) for n in ("gate", "up", "down")}
    EXd = {n: torch.stack([d4(g("mlp.experts.%d.%s_proj.weight" % (e, n)), g("mlp.experts.%d.%s_proj.weight_scale" % (e, n))) for e in range(256)]) for n in ("gate", "up", "down")}
if os.environ.get("SYNROUTER", "0") == "1":             # a synthetic router: expert e listens to dimension 8 e, so a row picks exactly the experts it is built from
    import stage as S_
    Wg = torch.zeros(256, 2048, device=dev); Wg[torch.arange(256), torch.arange(256) * 8] = 4.0; Wg = bf(Wg)
    tab_, B_ = vm.layers[LY]; B_["Wg"] = S_.pack_w(Wg); import vmoe2 as V_
    tab_[V_.TAB.index("Wg")] = B_["Wg"].data_ptr()
PINV = torch.linalg.pinv((fl(Wg) - fl(Wg).mean(0, keepdim=True)).double()).float()      # x = PINV[:, e] raises expert e's logit alone
pool = []
for p in sys.argv[1:]:
    pool += [d["x"] for d in torch.load(p, map_location="cpu") if d["layer"] == LY]
pool = torch.cat(pool).to(dev) if pool else None
def ref(x, ksel=None):
    m = fl(x); mq = mx8(m); lg = m @ fl(Wg).T; p = torch.softmax(lg, 1); wt, sel = p.topk(8, 1)
    if VA.V.A4: lg, p, wt, sel = top8(lg)
    if ksel is not None:                               # the kernel's own top-8 (a near-tie may pick the other expert): its weights
        sel = ksel; wt = p.gather(1, sel)
    wt = wt / wt.sum(1, keepdim=True)
    gate = torch.sigmoid(m @ fl(seg).T)
    sh = fl(bf(silu(m @ SH["gate"].T) * (m @ SH["up"].T))) @ SH["down"].T
    if VA.V.A4:
        m4 = rmx4(m); ge = torch.einsum("tk,tjnk->tjn", m4, EXd["gate"][sel]); ue = torch.einsum("tk,tjnk->tjn", m4, EXd["up"][sel])
        moe = (wt.unsqueeze(-1) * torch.einsum("tjk,tjnk->tjn", rmx4(silu(ge) * ue), EXd["down"][sel])).sum(1)
    else:
        ge = fl(bf(torch.einsum("tk,tjnk->tjn", mq, EXd["gate"][sel]))); ue = fl(bf(torch.einsum("tk,tjnk->tjn", mq, EXd["up"][sel])))
        a = mx8(wt.unsqueeze(-1) * silu(ge) * ue)
        moe = torch.einsum("tjk,tjnk->tn", a, EXd["down"][sel])
    gap = (p.topk(9, 1).values[:, 7] - p.topk(9, 1).values[:, 8]) / p.topk(9, 1).values[:, 7]
    return moe + gate * sh, sel, gap
worst, slow, bad, nonfin, us = 0.0, 0, 0, 0, {}
for it in range(N):
    R = int(torch.randint(int(os.environ.get('RMIN', 1)), 17, (1,)))
    mode = it % 4 if os.environ.get('CRAFT', '0') == '0' else 4
    if mode == 0 or pool is None:
        h = torch.randn(R, 2048, device=dev); x = bf(h * torch.rsqrt(h.pow(2).mean(1, keepdim=True) + 1e-6))
    elif mode == 1:                                    # distinct dumped rows
        x = pool[torch.randperm(pool.shape[0], device=dev)[:R]].clone()
    elif mode == 2:                                    # a few dumped rows repeated (small unions)
        k = int(torch.randint(1, 4, (1,))); x = pool[torch.randint(0, pool.shape[0], (k,), device=dev)][torch.randint(0, k, (R,), device=dev)].clone()
    else:                                              # dumped rows with a little noise (near-ties)
        x = bf(fl(pool[torch.randint(0, pool.shape[0], (R,), device=dev)]) * (1 + 0.01 * torch.randn(R, 2048, device=dev)))
    if os.environ.get("SYNROUTER", "0") == "1":
        perm = torch.randperm(256, device=dev); nd = max(1, R - int(torch.randint(0, 4, (1,)))) if it % 3 else int(torch.randint(1, R + 1, (1,)))
        h = torch.randn(R, 2048, device=dev) * 0.05
        for t_ in range(R):
            ex_ = perm[(t_ % nd) * 8:(t_ % nd) * 8 + 8]; h[t_, ex_ * 8] += 3.0 + torch.rand(8, device=dev)
        x = bf(h * torch.rsqrt(h.pow(2).mean(1, keepdim=True) + 1e-6)); mode = 5
    if mode == 4:                                      # rows steered to chosen (mostly disjoint) expert sets: unions up to 8 R
        perm = torch.randperm(256, device=dev); nd = int(torch.randint(1, R + 1, (1,)))
        rows_ = []
        for t_ in range(R):
            g_ = t_ % nd; ex_ = perm[g_ * 8:g_ * 8 + 8]
            h = (PINV[:, ex_] * (1 + 0.3 * torch.rand(8, device=dev))).sum(1) * float(torch.rand(1) * 2 + 0.5) + torch.randn(2048, device=dev) * PINV.std() * 0.05
            rows_.append(h)
        h = torch.stack(rows_); x = bf(h * torch.rsqrt(h.pow(2).mean(1, keepdim=True) + 1e-6))
    x = x.contiguous(); torch.cuda.synchronize(); t = time.perf_counter()
    out = vm.run(LY, x).clone(); torch.cuda.synchronize(); dt = time.perf_counter() - t
    ks = vm.buf["sel"][:R].long().clone()
    r, sel, gap = ref(x, ks if os.environ.get("KSEL", "1") == "1" else None); U = len(set(sel.flatten().tolist()))
    if os.environ.get("KSEL", "1") == "1":
        tsel = top8(fl(x) @ fl(Wg).T)[3] if VA.V.A4 else torch.softmax(fl(x) @ fl(Wg).T, 1).topk(8, 1).indices
        gap = torch.where(torch.tensor([set(a) == set(b) for a, b in zip(ks.tolist(), tsel.tolist())], device=dev), torch.ones(R, device=dev), gap)   # sets equal: not a tie case
        selbad = sum(len(set(a)) != 8 for a in ks.tolist())
        if selbad: print("  case %d: %d rows whose kernel top-8 has a repeated expert" % (it, selbad))
        gap = torch.ones(R, device=dev)                # with the kernel's own selection every row is checked
    fin = bool(torch.isfinite(out.float()).all())
    e = float((fl(out) - r).norm() / r.norm()) if fin else float("inf")
    tie = float(gap.min()) < 1e-3                      # a near-tie at the top-8 boundary may pick the other expert
    nonfin += not fin; slow += dt > 0.5; us[U] = us.get(U, 0) + 1
    if fin and not tie: worst = max(worst, e)
    if (not fin) or dt > 0.5 or (e > 3e-2 and not tie):
        bad += 1
        c16 = len(set(v for v in sel.flatten().tolist() if v < 16)); X = len(set(v for v in sel.flatten().tolist() if 16 <= v < 33))
        rowe = ((fl(out) - r).norm(dim=1) / r.norm(dim=1)).tolist() if fin else []
        if bad <= 12: print("  case %d mode %d R %d U %d c16 %d X %d: rel %.3e finite %s launch %.3f s tie %s rows %s" % (it, mode, R, U, c16, X, e, fin, dt, tie, " ".join("%.2f" % v for v in rowe)))
        if bad <= 3: torch.save(dict(x=x.cpu(), out=out.cpu(), ref=r.cpu(), sel=sel.cpu()), "/work/mono/dk/_bad_case_%d.pt" % bad)
print("layer %d, %d cases: non-finite %d, launches > 0.5 s %d, flagged %d, worst rel (no near-tie) %.2e | union sizes seen %d..%d" % (LY, N, nonfin, slow, bad, worst, min(us), max(us)))
