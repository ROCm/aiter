"""AITER fused_moe time vs distinct experts (same call, weights, M), CUDA-graph timed."""
import sys, torch
sys.path.insert(0, "/home/vigokhal/m3_monokernel/bench")
from m3_inputs import make_layers
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
dev = torch.device("cuda")
lw = make_layers(1, 64, 0, dev)[0]
def fp4(t):
    v = t.view(torch.float4_e2m1fn_x2); v.is_shuffled = True; return v
w13, w2 = fp4(lw.w13), fp4(lw.w2)
d = torch.load("bench/logs/sav/moein48.pt")  # the hybrid's real layer-0 MoE inputs (48 rows, dup rows)
x = d["x"].to(dev)
def run(ids, w, M):
    out = torch.empty(M, 6144, dtype=torch.bfloat16, device=dev)
    f = lambda: fused_moe(x[:M], w13, w2, w, ids, None, ActivationType.Swiglu, QuantType.per_1x32, False, lw.s13,
                          lw.s2, None, None, dtype=torch.bfloat16, swiglu_limit=7.0, output=out)
    f(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.cuda.graph(g, stream=s):
        for _ in range(20): f()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    ts = []
    for _ in range(15):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1000 / 20)
    return sorted(ts)[7]
gen = torch.Generator(device="cpu").manual_seed(0)
for M in (16, 48):
    w = torch.full((M, 5), 0.5, device=dev); w[:, 4] = 1.0
    for nd in (8, 16, 32, 48, 64, 96, 128):
        pool = torch.randperm(128, generator=gen)[:nd]
        ids = torch.stack([pool[torch.randperm(nd, generator=gen)[:4]] for _ in range(M)])
        ids = torch.cat([ids, torch.full((M, 1), 128)], 1).int().to(dev)
        print(f"M={M} distinct routed experts {len(ids[:, :4].unique()):3d}: {run(ids, w, M):6.1f} us", flush=True)
    if M == 48:
        ids = d["ids"].to(dev); print(f"M=48 hybrid's synthetic routing ({len(ids[:, :4].unique())} distinct): {run(ids, d['w'].to(dev), 48):6.1f} us")
