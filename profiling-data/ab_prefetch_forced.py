"""Untracked: forced-split A/B. usage: page batch S(auto|N) reps"""
import sys, statistics, math, copy
import torch
from torch.profiler import profile, ProfilerActivity
import ab_prefetch as ab
from ab_prefetch import bench, make_case

def kt(case, fs, iters=101):
    out = torch.empty(case['q'].shape, device='cuda', dtype=torch.bfloat16)
    call, pos = bench._flydsl_call(case, out, fs)
    fp = sum(t.numel() * t.element_size() for t in pos)
    copies = max(2, math.ceil(bench.CACHE_TARGET_BYTES / fp))
    rot = [copy.deepcopy(pos) for _ in range(copies - 1)] + [pos]
    for i in range(3): call(*rot[i % copies])
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for i in range(iters): call(*rot[i % copies])
        torch.cuda.synchronize()
    sp, cb = [], []
    for e in prof.events():
        if e.device_type != torch.autograd.DeviceType.CUDA: continue
        n = e.name.lower()
        if 'combine' in n: cb.append(e.device_time)
        elif 'elementwise' not in n and 'fill' not in n and 'copy' not in n: sp.append(e.device_time)
    return statistics.median(sp), (statistics.median(cb) if cb else 0.0)

page, b, S, reps = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
fs = None if S == 'auto' else int(S)
mods = ab.load_variants()
spec = next((s for s in bench.sliding_specs(page_sizes=(page,)) if s['name'] == f'decode-b{b}-context32768'), None)
if spec is None:
    spec = dict(name=f'decode-b{b}-context32768', mode='decode', batch=b, query_lens=[1]*b,
                kv_lens=[32768]*b, context=32768, page_size=page)
case = make_case(spec, False); case['sink'] = False
s = {'A': [], 'B': []}
for rep in range(reps):
    for v in (('A','B') if rep % 2 == 0 else ('B','A')):
        ab.select_builder(mods, v)
        s[v].append(bench._timing_row(case, 'flydsl', v, 101, fs)['us'])
k = {}
for v in 'AB':
    ab.select_builder(mods, v); k[v] = kt(case, fs)
a, bb = statistics.median(s['A']), statistics.median(s['B'])
print(f"RESULT page={page} B={b} S={S} A={a:.2f} B={bb:.2f} BA={bb/a:.3f} "
      f"Asplit={k['A'][0]:.2f} Acomb={k['A'][1]:.2f} Bsplit={k['B'][0]:.2f} Bcomb={k['B'][1]:.2f} "
      f"Arng={min(s['A']):.1f}-{max(s['A']):.1f} Brng={min(s['B']):.1f}-{max(s['B']):.1f}", flush=True)
