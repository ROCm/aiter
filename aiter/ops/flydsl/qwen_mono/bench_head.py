import os, sys, time, torch
sys.path.insert(0, os.path.dirname(__file__))
import head as H
torch.manual_seed(0)
E = lambda k, d: int(os.environ.get(k, d))
N, K, TW, D, GW, math, nt = E("N", 248320), 2048, E("TW", 4), E("DD", 4), E("GW", 2), E("MATH", 1) == 1, E("NT", 1) == 1
W = (torch.randn(N, K, device="cuda") * 0.02).to(torch.bfloat16)
x = torch.randn(16, K, device="cuda").to(torch.bfloat16)
out = torch.empty(16, dtype=torch.int64, device="cuda")
st = torch.zeros(256, 4, dtype=torch.int64, device="cuda")
U = E("U", 1); ln = H.build(N, K, D, math, nt, TW, GW, U); Wp = H.pack(W, TW)
def run():
    out.fill_(H.IMIN)
    ln(x.data_ptr(), Wp.data_ptr(), out.data_ptr(), st.data_ptr(), stream=torch.cuda.current_stream())
run(); torch.cuda.synchronize()
try:
    from atom.mono.runtime.compile import launcher_resources
    print("resources", launcher_resources(ln))
except Exception as e:
    print("resources: ", repr(e)[:200])
ri = (x.float() @ W.float().T).to(torch.bfloat16).float().argmax(1)
tok = 0x7FFFFFFF - (out & 0xFFFFFFFF)
ts = []; ph = []
for _ in range(200):
    run(); torch.cuda.synchronize()
    ts.append(int(st[:, 1].max() - st[:, 0].min())); a = st[:242]; t0 = st[:, 0].min()
    ph.append([float((a[:, 0].max() - t0)), float((a[:, 2] - t0).float().median()), float((a[:, 3] - t0).float().median()), float((a[:, 3] - t0).max()), float((a[:, 1] - t0).max())])
ts = torch.tensor(ts[20:]).float() / 100.0
print("U %d LDS %d KB " % (U, H.lds_bytes(K, TW, D) // 1024), end=""); print("N %d TW %d GW %d B %d math %d nt %d: match %d/16, in-kernel us median %.1f min %.1f p90 %.1f" % (N, TW, GW, D, math, nt, int((tok == ri).sum()), ts.median(), ts.min(), ts.quantile(0.9)))
p = torch.tensor(ph[20:]).median(0).values / 100
print("  us from first start: last CTA start %.1f | loop start (median CTA) %.1f | loop end median %.1f, last %.1f | end %.1f" % tuple(p.tolist()))
d = ((st[:242, 3] - st[:242, 2]).float() / 100)
print("  loop us per CTA: min %.1f p25 %.1f med %.1f p75 %.1f p90 %.1f max %.1f" % (d.min(), d.quantile(.25), d.median(), d.quantile(.75), d.quantile(.9), d.max()))
print("  by 16-CTA cluster (mean):", " ".join("%.0f" % v for v in d[:240].view(15, 16).mean(1).tolist()))
print("  by CTA %% 16 (mean):", " ".join("%.0f" % v for v in d[:240].view(15, 16).mean(0).tolist()))
