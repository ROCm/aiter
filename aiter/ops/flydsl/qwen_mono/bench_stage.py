"""usage: bench_stage.py <name> N,K N,K ...   (x of stage i+1 = the first K columns of stage i's output, zero-extended)"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
import stage as S
torch.manual_seed(0)
name = sys.argv[1]; stages = [tuple(int(v) for v in a.split(",")) for a in sys.argv[2:]]
dev = "cuda"
Ws = [(torch.randn(n, k, device=dev) / k ** 0.5).to(torch.bfloat16) for n, k in stages]
Wp = [S.pack_w(w) for w in Ws]
x0 = torch.randn(16, stages[0][1], device=dev).to(torch.bfloat16)
widths = [stages[0][1]] + [max(n, stages[i + 1][1] if i + 1 < len(stages) else n) for i, (n, k) in enumerate(stages)]
Xs = [torch.zeros(wd // 32, 32, 16, dtype=torch.bfloat16, device=dev) for wd in widths]
Xs[0].view(-1).copy_(S.pack_x(x0).reshape(-1))
tab = torch.tensor([w.data_ptr() for w in Wp] + [x.data_ptr() for x in Xs], dtype=torch.int64, device=dev)
cnt = torch.zeros(4096, dtype=torch.int32, device=dev); st = torch.zeros(256, S.NSTAMP, dtype=torch.int64, device=dev)
ln = S.build(stages)
def run():
    cnt.zero_()
    ln(tab.data_ptr(), cnt.data_ptr(), st.data_ptr(), stream=torch.cuda.current_stream()); torch.cuda.synchronize()
run()
try:
    from atom.mono.runtime.compile import launcher_resources
    r = launcher_resources(ln)[0]; print("vgpr %d sgpr %d scratch %d" % (r["vgpr"], r["sgpr"], r["scratch"]))
except Exception as e:
    print("resources:", repr(e)[:120])
# reference chain
x = x0.float(); errs = []
for i, (n, k) in enumerate(stages):
    xi = torch.zeros(16, k, device=dev); m = min(k, x.shape[1]); xi[:, :m] = x[:, :m]
    ref = (xi @ Ws[i].float().T).to(torch.bfloat16).float()
    got = S.unpack_x(Xs[i + 1], widths[i + 1])[:, :n].float()
    errs.append(float((got - ref).abs().max() / ref.abs().max()))
    x = got
print(name, "max rel err per stage:", " ".join("%.1e" % e for e in errs))
rows = []
for _ in range(150):
    run(); rows.append((st[:, 1:len(stages) + 2].max(0).values - st[:, 0].min()).float() / 100)
r = torch.stack(rows[20:]).median(0).values.tolist()
print(name, "us: start handoff %.2f | " % r[0] + " | ".join("%s %.2f" % ("%dx%d" % s, r[i + 1] - r[i]) for i, s in enumerate(stages)) + " | total %.2f (stages only %.2f)" % (r[-1], r[-1] - r[0]))
if os.environ.get("DBG"):
    x = x0.float()
    for i, (n, k) in enumerate(stages):
        xi = torch.zeros(16, k, device=dev); m = min(k, x.shape[1]); xi[:, :m] = x[:, :m]
        ref = (xi @ Ws[i].float().T).to(torch.bfloat16).float()
        got = S.unpack_x(Xs[i + 1], widths[i + 1])[:, :n].float()
        bad = ((got - ref).abs() > 0.05 * ref.abs().max()) | got.isnan()
        cols = bad.any(0).nonzero().flatten()
        print(i, "bad cols %d of %d" % (len(cols), n), "tiles:", sorted(set((cols // 16).tolist()))[:12], "...", "nan", int(got.isnan().sum()), "ref nan", int(ref.isnan().sum()))
        x = ref
