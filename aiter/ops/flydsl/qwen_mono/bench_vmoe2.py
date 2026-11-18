import os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
import vmoe2 as V, stage as S, draft_mono as DM
from safetensors import safe_open
torch.manual_seed(0); dev = "cuda"; R = int(os.environ.get("R", 12)); LY = int(os.environ.get("LAYER", 0))
D = "/data/Qwen3.6-35B-A3B-MXFP4-experts-MXFP8-gemm-lmhead/"
with safe_open(D + "layers-%d.safetensors" % LY, "pt") as f:
    g = lambda k: f.get_tensor("model.language_model.layers.%d.%s" % (LY, k)).to(dev)
    Wg, seg, w1 = g("mlp.gate.weight"), g("mlp.shared_expert_gate.weight"), g("post_attention_layernorm.weight").float()
    SH = {n: (g("mlp.shared_expert.%s_proj.weight" % n), g("mlp.shared_expert.%s_proj.weight_scale" % n)) for n in ("gate", "up", "down")}
    EX = {n: (torch.stack([g("mlp.experts.%d.%s_proj.weight" % (e, n)) for e in range(256)]), torch.stack([g("mlp.experts.%d.%s_proj.weight_scale" % (e, n)) for e in range(256)])) for n in ("gate", "up", "down")}
with safe_open(D + "outside.safetensors", "pt") as f:
    emb = f.get_tensor("model.language_model.embed_tokens.weight").to(dev)
bf = lambda x: x.to(torch.bfloat16); fl = lambda x: x.float()
z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=dev)
TAB4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device=dev)
d4 = lambda q, s: TAB4[torch.stack([q & 15, q >> 4], -1).reshape(*q.shape[:-1], -1).long()] * torch.pow(2.0, s.float() - 127).repeat_interleave(32, -1)
d8 = lambda w, s: w.float() * torch.pow(2.0, s.float() - 127).repeat_interleave(32, -1)
def pk8(w8, ws):                                   # [N][K] fp8, [N][K/32] -> [N/16][K/128][lane][64 B], [N/16][K/128][16][4]
    n, k = w8.shape
    return (w8.view(torch.uint8).view(n // 16, 16, k // 128, 4, 2, 16).permute(0, 2, 4, 1, 3, 5).contiguous(),
            torch.cat([ws.view(n // 16, 16, k // 128, 4).permute(0, 2, 1, 3).reshape(-1), torch.zeros(1024, dtype=torch.uint8, device=dev)]))
SW, SWS = pk8(torch.cat([SH["gate"][0], SH["up"][0]]), torch.cat([SH["gate"][1], SH["up"][1]])); SD, SDS = pk8(*SH["down"])
EW, ES = DM._pk4(torch.cat([EX["gate"][0], EX["up"][0]], 1), torch.cat([EX["gate"][1], EX["up"][1]], 1)); ED, EDS = DM._pk4(*EX["down"])
h1 = z(16, 2048, dt=torch.bfloat16); hr = bf(emb[torch.randint(0, 248320, (R,))].float() * 20 + torch.randn(R, 2048, device=dev) * 0.2)
h1[:R] = hr; h1[R:] = hr[R - 1]
xg = bf(fl(h1) * (1 + w1)); part = z(16, 256); part[:, 0] = fl(h1).pow(2).sum(1)
B = dict(Wg=S.pack_w(Wg), Wsg=S.pack_w(torch.cat([seg, z(15, 2048, dt=torch.bfloat16)])), SW=SW, SWS=SWS, EW=EW, ES=ES, ED=ED, EDS=EDS, SD=SD, SDS=SDS,
         h1=h1, xg1=S.pack_x(xg), xg1p=xg.contiguous(), part1=part, lg=z(16, 256), sgate=z(16), sel=z(16, 8, dt=torch.int32), topv=z(16, 8), sdn=z(16, 2048), slot=z(16, 8, 2048),
         GE=z(256, 4, 32, 64, dt=torch.uint8), GS=z(256, 4, 32, 4, dt=torch.int32), GSU=z(4, 32, 64, dt=torch.uint8), GSS=z(4, 32, 4, dt=torch.int32),
         h2=z(16, 2048, dt=torch.bfloat16))
def q8(v):
    am = v.view(16, 64, 32).abs().amax(-1).clamp_min(1e-30); code = (torch.floor(torch.log2(am)) - 8 + 127).clamp(1, 254)
    return (v / torch.pow(2.0, code - 127).repeat_interleave(32, -1)).clamp(-448, 448).to(torch.float8_e4m3fn), code.to(torch.uint8)
import head8 as H8
xq_, xc_ = q8(fl(xg)); B["xq8"], xs_ = H8.pack_x(xq_.view(torch.uint8), xc_); B["xqs8"] = torch.cat([xs_.reshape(-1), torch.zeros(1024, dtype=torch.uint8, device=dev)])
tab = torch.tensor([B[k].data_ptr() for k in V.TAB], dtype=torch.int64, device=dev)
cnt = torch.zeros(4096, dtype=torch.int32, device=dev); st = torch.zeros(2048, V.NSTAMP, dtype=torch.int64, device=dev)
ln = V.build()
AIO = V.ATOMIO
xin = bf(fl(xg) * torch.rsqrt(fl(h1).pow(2).mean(1, keepdim=True) + 1e-6))[:R].contiguous()      # ATOM's normed x, R rows only
out = z(16, 2048, dt=torch.bfloat16)
def run():
    cnt.zero_(); st.zero_(); ln(tab.data_ptr(), cnt.data_ptr(), st.data_ptr(), xin.data_ptr(), out.data_ptr(), R, stream=torch.cuda.current_stream()); torch.cuda.synchronize()
run()
# reference with the kernel's rounding points
def mx8(v, blk=32):                                # ATOM's MXFP8 1x32 rule: scale = 2^ceil(log2(amax / 448)), nothing clamped
    am = v.view(*v.shape[:-1], -1, blk).abs().amax(-1).clamp(1e-10, 448.0 * 6.646139978924579e35)
    mant, ex = torch.frexp(am / 448.0); code = torch.where(mant > 0.5, ex, ex - 1).float() + 127
    sc = torch.pow(2.0, code - 127).repeat_interleave(blk, -1); return (v / sc).clamp(-448, 448).to(torch.float8_e4m3fn).float() * sc
silu = torch.nn.functional.silu
m = fl(xg); rs = torch.rsqrt(fl(h1).pow(2).mean(1, keepdim=True) + 1e-6)
if AIO:                                            # ATOM order: the normed bf16 x is what gets quantized; rstd = 1
    m = torch.zeros(16, 2048, device=dev); m[:R] = fl(xin); m[R:] = fl(xin[R - 1]); rs = torch.ones(16, 1, device=dev)
mq = mx8(m) * rs if os.environ.get('REFQ', 'pre') == 'pre' else mx8(m * rs)
lg = rs * (m @ fl(Wg).T); p = torch.softmax(lg, 1); wt, sel = p.topk(8, 1); wt = wt / wt.sum(1, keepdim=True)
gate = torch.sigmoid(rs * (m @ fl(seg).T))
sh = mx8(silu(fl(bf(mq @ d8(*SH["gate"]).T))) * fl(bf(mq @ d8(*SH["up"]).T))) @ d8(*SH["down"]).T
moe = torch.zeros(16, 2048, device=dev)
for t in range(R):
    for j in range(8):
        e = int(sel[t, j]); ge = fl(bf(mq[t] @ d4(*[x[e] for x in EX["gate"]]).T)); ue = fl(bf(mq[t] @ d4(*[x[e] for x in EX["up"]]).T))
        moe[t] += mx8(wt[t, j] * silu(ge) * ue) @ d4(*[x[e] for x in EX["down"]]).T
h2 = bf(fl(h1) + moe + gate * sh)
if AIO:
    h2 = bf(moe + gate * sh); B["h2"] = out; h1 = torch.zeros_like(h1)
rel = lambda a, b: float((fl(a) - fl(b)).abs().max() / fl(b).abs().max().clamp_min(1e-30))
selk = B["sel"][:R].long(); dm = (fl(B["h2"][:R]) - fl(h1[:R])) - (moe + gate * sh)[:R]
print("layer %d R %d: logits %.1e  top-8 sets equal %d/%d  h2 %.1e  MoE part rel err %.1e  U %d  X %d" % (LY, R, rel(B["lg"][:R] * rs[:R], lg[:R]),
      sum(set(selk[t].tolist()) == set(sel[t].tolist()) for t in range(R)), R, rel(B["h2"][:R], h2[:R]),
      float(dm.norm() / (moe + gate * sh)[:R].norm()), len(set(selk.flatten().tolist())), len([e for e in set(selk.flatten().tolist()) if 16 <= e < 33])))
if os.environ.get("TIME", "1") == "1":
    rows = []
    def ev(k, m=None, f="max"):
        v = st[:, k].double(); ok = v > 0
        if m is not None: ok &= m
        v = v[ok]
        if v.numel() == 0: return float("nan")
        return float({"max": v.max, "min": v.min, "med": v.median}[f]())
    cta = torch.arange(2048, device=dev) // 8
    for _ in range(150):
        run(); t0 = ev(9, f="min")
        rows.append([(x - t0) / 100 for x in (ev(6), ev(7), ev(1, cta >= 33), ev(8, cta >= 33, "med"), ev(8, cta >= 33), ev(2), ev(3, None, "med"), ev(3), ev(4, None, "med"), ev(4), ev(5))] + [(t0 - ev(0, f="min")) / 100])
    r = torch.tensor(rows[20:], dtype=torch.float64).nanmedian(0).values.tolist()
    print("us after entry: logits %.1f, tag published %.1f | owners: prologue %.1f, own unit med %.1f max %.1f | tag seen (max) %.1f | gate/up done med %.1f max %.1f | down done med %.1f max %.1f | end %.1f   (start %.1f)" % tuple(r))

    run(); t0 = ev(9, f="min"); S = set(selk.flatten().tolist())
    selm = torch.tensor([(i // 8) in S for i in range(2048)], device=dev)
    own = selm & (cta >= 33); hlp = (~selm) & ((cta < 16) | (cta >= 33))
    f = lambda k, m, how="med": (ev(k, m, how) - t0) / 100
    print("selected owner wave: chunk 0/2/4/6 ready %.1f %.1f %.1f %.1f, unit done med %.1f max %.1f" % (f(10, own), f(11, own), f(12, own), f(13, own), f(8, own), f(8, own, "max")))
    print("helper wave: tag seen %.1f, chunk 0/2/4/6 ready %.1f %.1f %.1f %.1f, WMMAs done med %.1f max %.1f, all done %.1f" % (f(2, hlp), f(10, hlp), f(11, hlp), f(12, hlp), f(13, hlp), f(14, hlp), f(14, hlp, "max"), f(3, hlp)))
    print("down: start med %.1f, done med %.1f max %.1f" % (f(3, None), f(4, None), f(4, None, "max")))
if os.environ.get("DIST", "0") == "1":
    S = set(selk.flatten().tolist()); cta = torch.arange(2048, device=dev) // 8; wv = torch.arange(2048, device=dev) % 8
    own = torch.tensor([(i // 8) in S for i in range(2048)], device=dev) & (cta >= 33)
    M = []
    for _ in range(80):
        run(); v = st[:, 9].double(); t0 = v[v > 0].min(); M.append((st[:, 8].double() - t0) / 100)
    M = torch.stack(M[10:])[:, own]; c2 = cta[own]; w2 = wv[own]
    med = M.median(0).values; q = lambda x, p: float(torch.quantile(x, p))
    print("owner waves %d: per-wave median of unit-done: p5 %.1f p50 %.1f p95 %.1f max %.1f" % (med.numel(), q(med, .05), q(med, .5), q(med, .95), float(med.max())))
    print("spread across waves (std of per-wave medians) %.2f | spread within a wave across runs (median std) %.2f" % (float(med.std()), float(M.std(0).median())))
    print("by wave in CTA:", " ".join("%.1f" % float(med[w2 == k].median()) for k in range(8)))
    print("by CTA %% 16:  ", " ".join("%.1f" % (float(med[c2 % 16 == k].median()) if (c2 % 16 == k).any() else float("nan")) for k in range(16)))
    print("by cluster:   ", " ".join("%.1f" % (float(med[c2 // 16 == k].median()) if (c2 // 16 == k).any() else float("nan")) for k in range(16)))
    pc = sorted(((float(med[c2 == e].median()), float(med[c2 == e].min()), float(med[c2 == e].max()), e) for e in sorted(set(c2.tolist()))))
    print("fastest CTAs (med,min,max,cta):", ["%.1f/%.1f/%.1f@%d" % x for x in pc[:6]]); print("slowest CTAs:", ["%.1f/%.1f/%.1f@%d" % x for x in pc[-10:]])
if os.environ.get("DIST", "0") == "2":
    S = set(selk.flatten().tolist()); wid = torch.arange(2048, device=dev); cta = wid // 8; U = len(S)
    selm = torch.tensor([(i // 8) in S for i in range(2048)], device=dev)
    own = selm & (cta >= 33); hlp = (~selm) & ((cta < 16) | (cta >= 33))
    A = {k: [] for k in (2, 3, 4, 8, 14)}
    for _ in range(80):
        run(); v = st[:, 9].double(); t0 = v[v > 0].min()
        for k in A: A[k].append(torch.where(st[:, k] > 0, (st[:, k].double() - t0) / 100, torch.full((2048,), float("nan"), dtype=torch.float64, device=dev)))
    A = {k: torch.stack(v[10:]).nanmedian(0).values for k, v in A.items()}
    def qs(x):
        x = x[~x.isnan()]
        return "n %4d  p5 %.1f p25 %.1f p50 %.1f p75 %.1f p95 %.1f max %.1f" % ((x.numel(),) + tuple(float(torch.quantile(x, p)) for p in (.05, .25, .5, .75, .95, 1.0))) if x.numel() else "n 0"
    print("U %d  helper waves %d  helper units %d" % (U, int(hlp.sum()), (U + len([e for e in S if 16 <= e < 33])) * 8 if False else -1))
    print("owner: unit done        ", qs(A[8][own]))
    print("helper: tag seen        ", qs(A[2][hlp]))
    print("helper: last unit done  ", qs(A[14][hlp]))
    print("helper: units time      ", qs((A[14] - A[2])[hlp]))
    print("all: leave gate/up      ", qs(A[3]))
    two = wid < U * 32 - 2048; one = (~two) & (wid < U * 32)
    print("down (1 unit) start     ", qs(A[3][one])); print("down (1 unit) duration  ", qs((A[4] - A[3])[one]))
    print("down (2 units) start    ", qs(A[3][two])); print("down (2 units) duration ", qs((A[4] - A[3])[two]))
    print("down done, all          ", qs(A[4]))
    for nm, m in (("ranker CTAs 0..15", cta < 16), ("shared CTAs 16..32", (cta >= 16) & (cta < 33)), ("owners sel", own), ("helpers", hlp & (cta >= 33))):
        print("%-18s leave gate/up %s | down done %s" % (nm, "%.1f/%.1f" % (float(A[3][m].nanmedian()), float(A[3][m][~A[3][m].isnan()].max())), "%.1f/%.1f" % (float(A[4][m].nanmedian()), float(A[4][m][~A[4][m].isnan()].max()))))
if os.environ.get("DIST", "0") == "3":
    KS = (9, 1, 10, 8, 11, 2, 14, 3, 4, 15, 5); nan = float("nan"); X = []; TO = []
    for _ in range(int(os.environ.get("NRUN", 90))):
        run(); v = st[:, 9].double(); t0 = v[v > 0].min(); TO.append((int(cnt[3072]), int(cnt[3073])))
        X.append(torch.where(st[:, list(KS)] > 0, (st[:, list(KS)].double() - t0) / 100, torch.full((2048, len(KS)), nan, dtype=torch.float64, device=dev)))
    X = torch.stack(X[10:]); med = X.nanmedian(0).values; L = X - med               # [run][wave][k] lateness vs the wave's own median
    nm = dict(zip(KS, ("start", "roles", "wait1", "ownunit", "wait2", "assigned", "helpunit", "leave-gu", "downdone", "ddone-seen", "end")))
    print("timeouts (count, last site) in runs:", [(i, t) for i, t in enumerate(TO[10:]) if t[0]])
    print("per-wave-median max:", " ".join("%s %.1f" % (nm[k], float(med[:, i][~med[:, i].isnan()].max())) for i, k in enumerate(KS)))
    mx = lambda a: torch.where(a.isnan(), torch.full_like(a, -1e9), a)
    pr = torch.stack([mx(X[:, :, i]).max(1).values for i in range(len(KS))], 1)
    print("per-run max, median over runs:", " ".join("%s %.1f" % (nm[k], float(pr[:, i].median())) for i, k in enumerate(KS)))
    print("per-run max, p10/p90 of end: %.1f / %.1f" % (float(torch.quantile(pr[:, -1], .1)), float(torch.quantile(pr[:, -1], .9))))
    for thr in (1.5, 3.0):
        print("waves later than their own median by > %.1f us, mean per run:" % thr, " ".join("%s %.1f" % (nm[k], float((mx(L[:, :, i]) > thr).sum(1).double().mean())) for i, k in enumerate(KS)))
    order = pr[:, -1].argsort()
    for r in [int(order[0]), int(order[len(order) // 2]), int(order[-2]), int(order[-1])]:
        print("run %d: end %.1f" % (r, float(pr[r, -1])))
        for i, k in enumerate(KS):
            l = mx(L[r, :, i]); bad = (l > 1.5).nonzero().flatten()
            if bad.numel():
                ct = sorted(set((bad // 8).tolist()))
                print("   %-11s late waves %4d in %3d CTAs, worst +%.1f (wave %d, CTA %d), CTAs %s" % (nm[k], bad.numel(), len(ct), float(l.max()), int(l.argmax()), int(l.argmax()) // 8, ct[:14]))

if os.environ.get("DIST", "0") == "4":
    N = int(os.environ.get("NRUN", 500)); bad = 0; slow = 0; ends = []; worst = 0.0
    for _ in range(N):
        B["h2"].zero_(); run(); v = st[:, 9].double(); t0 = v[v > 0].min(); e = float((st[:, 5].double().max() - t0) / 100); ends.append(e)
        er = rel(B["h2"][:R], h2[:R]); worst = max(worst, er); bad += er > 4e-3; slow += e > 1000
    ends = torch.tensor(sorted(ends))
    print("R %d, %d runs: end p10 %.1f p50 %.1f p90 %.1f p99 %.1f | runs > 1 ms: %d | runs with h2 err > 4e-3: %d (worst %.1e)" % (R, N, ends[N // 10], ends[N // 2], ends[N * 9 // 10], ends[N * 99 // 100], slow, bad, worst))
if os.environ.get("DIST", "0") == "5":
    S = set(selk.flatten().tolist()); wid = torch.arange(2048, device=dev); cta = wid // 8
    selm = torch.tensor([(i // 8) in S for i in range(2048)], device=dev)
    KS = (7, 1, 10, 8, 14, 2, 3, 4, 15, 5); names = ("publish", "roles", "wait1", "ownunit", "helpunit", "assigned", "leave-gu", "downdone", "ddone-seen", "exit")
    X = []
    for _ in range(80):
        run(); v = st[:, 9].double(); t0 = v[v > 0].min()
        X.append(torch.where(st[:, list(KS)] > 0, (st[:, list(KS)].double() - t0) / 100, torch.full((2048, len(KS)), float("nan"), dtype=torch.float64, device=dev)))
    M = torch.stack(X[10:]).nanmedian(0).values
    for nm, m in (("ranker sel", (cta < 16) & selm), ("ranker unsel", (cta < 16) & ~selm), ("shared 16..32", (cta >= 16) & (cta < 33)), ("owner sel", (cta >= 33) & selm), ("helper", (cta >= 33) & ~selm)):
        row = []
        for i, k in enumerate(names):
            v = M[m, i]; v = v[~v.isnan()]
            row.append("%s %s" % (k, "%.1f/%.1f" % (float(v.median()), float(v.max())) if v.numel() else "-"))
        print("%-14s (%4d waves) %s" % (nm, int(m.sum()), "  ".join(row)))
if os.environ.get("DIST", "0") == "6":             # host wall time per call of (zero the counters + launch), back to back, as ATOM would issue them
    import time
    N = 3000
    def loop(fn):
        for _ in range(200): fn()
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(N): fn()
        torch.cuda.synchronize(); return (time.perf_counter() - t) / N * 1e6
    s_ = torch.cuda.current_stream()
    a = loop(lambda: (cnt.zero_(), ln(tab.data_ptr(), cnt.data_ptr(), st.data_ptr(), xin.data_ptr(), out.data_ptr(), R, stream=s_)))
    b_ = loop(lambda: cnt.zero_())
    print("R %d: zero + launch %.1f us a call (host-driven, includes launch overhead); zero alone %.1f us" % (R, a, b_))
    print("out rel err after the loop: %.1e" % rel(B["h2"][:R], h2[:R]))
if os.environ.get("DIST", "0") == "7":             # GPU time per call inside a captured graph (as ATOM replays its decode steps)
    import time
    K = 40
    def cap(fn):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(K): fn()
        for _ in range(5): g.replay()
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(50): g.replay()
        torch.cuda.synchronize(); return (time.perf_counter() - t) / (50 * K) * 1e6
    full = cap(lambda: (cnt.zero_(), ln(tab.data_ptr(), cnt.data_ptr(), st.data_ptr(), xin.data_ptr(), out.data_ptr(), R, stream=torch.cuda.current_stream())))
    zero = cap(lambda: cnt.zero_())
    print("R %d in a graph: zero + kernel %.1f us a call; zero alone %.1f us; kernel %.1f us" % (R, full, zero, full - zero))
    print("out rel err after the replays: %.1e" % rel(B["h2"][:R], h2[:R]))
