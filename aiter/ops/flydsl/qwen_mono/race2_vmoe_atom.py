"""Back-to-back launches with DIFFERENT inputs inside one captured graph (as ATOM's 40 layers run): every launch's output
is kept in its own buffer and compared with a torch reference. A read that returns the previous launch's data shows here
(the benches replay one input, where stale data equals fresh data)."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
import vmoe_atom as VA
from safetensors import safe_open
torch.manual_seed(0); dev = "cuda"
D = "/data/Qwen3.6-35B-A3B-MXFP4/"; LY = int(os.environ.get("LAYER", 17)); K = int(os.environ.get("K", 24)); R = int(os.environ.get("R", 12))
vm = VA.VMoE(D, torch.device(dev)); 
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
LYS = [int(v) for v in os.environ.get("LAYERS", "17,18").split(",")]
W = {}
for LY in LYS:
    vm.load_layer(LY)
    with safe_open(D + "layers-%d.safetensors" % LY, "pt") as f:
        g = lambda k: f.get_tensor("model.language_model.layers.%d.%s" % (LY, k)).to(dev)
        W[LY] = (g("mlp.gate.weight"), g("mlp.shared_expert_gate.weight"), {n: fl(g("mlp.shared_expert.%s_proj.weight" % n)) for n in ("gate", "up", "down")},
                 {n: torch.stack([d4(g("mlp.experts.%d.%s_proj.weight" % (e, n)), g("mlp.experts.%d.%s_proj.weight_scale" % (e, n))) for e in range(256)]) for n in ("gate", "up", "down")})
def ref(x, LY):
    Wg, seg, SH, EXd = W[LY]
    m = fl(x); lg = m @ fl(Wg).T
    lg, p, wt, sel = top8(lg)
    gate = torch.sigmoid(m @ fl(seg).T)
    sh = fl(bf(silu(m @ SH["gate"].T) * (m @ SH["up"].T))) @ SH["down"].T
    m4 = rmx4(m); ge = torch.einsum("tk,tjnk->tjn", m4, EXd["gate"][sel]); ue = torch.einsum("tk,tjnk->tjn", m4, EXd["up"][sel])
    return (wt.unsqueeze(-1) * torch.einsum("tjk,tjnk->tjn", rmx4(silu(ge) * ue), EXd["down"][sel])).sum(1) + gate * sh
xs = []
for k in range(K):
    h = torch.randn(R, 2048, device=dev) * (1 + k % 3); xs.append(bf(h * torch.rsqrt(h.pow(2).mean(1, keepdim=True) + 1e-6)).contiguous())
SAMEX = os.environ.get('SAMEX','0')=='1'
if SAMEX: xs = [xs[0]] * K
refs = [ref(xs[k], LYS[k % len(LYS)]) for k in range(K)]
G = os.environ.get("GARB", "")
if G:                                               # x is the first R rows of a 16-row buffer whose other rows hold garbage
    big = []
    for k in range(K):
        b_ = torch.zeros(16, 2048, dtype=torch.bfloat16, device=dev)
        if G == "big": b_.fill_(3e30)
        elif G == "bits": b_.view(torch.int16).copy_(torch.randint(-32768, 32767, (16, 2048), device=dev, dtype=torch.int16))
        elif G == "nan": b_.fill_(float("nan"))
        elif G == "inf": b_.fill_(float("inf"))
        elif G == "mid": b_.copy_(bf(torch.randn(16, 2048, device=dev) * 300))
        b_[:R] = xs[k]; big.append(b_)
    xs = [b_[:R] for b_ in big]
outs = [torch.zeros(16, 2048, dtype=torch.bfloat16, device=dev) for _ in range(K)]
gph = torch.cuda.CUDAGraph()
with torch.cuda.graph(gph):
    for k in range(K):
        vm.cnt.zero_()
        vm.ln(vm.layers[LYS[k % len(LYS)]][0].data_ptr(), vm.cnt.data_ptr(), vm.st.data_ptr(), xs[k].data_ptr(), outs[k].data_ptr(), R, stream=torch.cuda.current_stream())
REPS = int(os.environ.get("REPS", 200)); gold = None; nbad = 0
keys = ("sel", "topv", "sgate", "lg", "xq8", "xqs8", "slot", "sdn", "GS")
snap = lambda: {k_: vm.buf[k_].clone() for k_ in keys}
for rep in range(REPS):
    gph.replay(); torch.cuda.synchronize()
    cur = outs[K - 1][:R].clone()
    if rep < 10 and os.environ.get("WARM", "0") == "1": continue
    if gold is None:
        e = (fl(cur) - refs[K - 1]).norm(dim=1) / refs[K - 1].norm(dim=1)
        if float(e.max()) > 0.05: print("rep %d: bad vs reference, rows" % rep, (e > 0.05).nonzero().flatten().tolist()); nbad += 1; bs = snap(); bo = cur; continue
        gold, gs = cur, snap(); continue
    df = (cur.view(torch.int16) != gold.view(torch.int16))
    if bool(df.any()):
        nbad += 1; bs = snap(); bo = cur
        print("rep %d: rows %s differ" % (rep, df.any(1).nonzero().flatten().tolist()))
print("R %d K %d: %d replays, bad last launches %d" % (R, K, REPS, nbad))
if nbad and gold is not None:
    rows = (bo.view(torch.int16) != gold.view(torch.int16)).any(1).nonzero().flatten().tolist()
    print("last bad: rows", rows, "cols differing per row", [(int((bo[r].view(torch.int16) != gold[r].view(torch.int16)).sum())) for r in rows])
    for k_ in keys:
        d = (bs[k_] != gs[k_])
        if bool(d.any()):
            print("  scratch %-5s differs: %d elements, index sample %s" % (k_, int(d.sum()), d.nonzero()[:6].tolist()))
            if k_ in ("sel", "topv", "sgate"): print("     bad", bs[k_][rows[0]].tolist() if bs[k_].dim() > 1 else bs[k_].tolist(), "\n     good", gs[k_][rows[0]].tolist() if gs[k_].dim() > 1 else gs[k_].tolist())
    r = rows[0]; sb, sg_ = bs["slot"][r], gs["slot"][r]
    print("  slot row %d: per rank max|bad| %s  max|good| %s  elements differing %s" % (r, [float("%.3g" % v) for v in sb.abs().amax(1).tolist()], [float("%.3g" % v) for v in sg_.abs().amax(1).tolist()], (sb != sg_).sum(1).tolist()))
    print("  sdn row %d differing %d; out bad max %.3g" % (r, int((bs["sdn"][r] != gs["sdn"][r]).sum()), float(fl(bo[r]).abs().max())))
