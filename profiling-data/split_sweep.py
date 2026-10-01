"""Split-count sweep: FlyDSL D256 split decode vs Triton Tier B. Kernel split/combine via torch.profiler."""
import argparse, csv, statistics as st, torch
from torch.profiler import profile, ProfilerActivity
from rescale_hit_rate import make_case
from scripts import bench_unified_attention_gemma4 as bench
from aiter.ops.flydsl.kernels.flash_attn_fp8_gfx942 import _decode_splits

p = argparse.ArgumentParser()
p.add_argument('--page', type=int, required=True)
p.add_argument('--cases', required=True, help='batch:S,S,..;batch:S,..  (S=0 means auto)')
p.add_argument('--context', type=int, default=32768)
p.add_argument('--rounds', type=int, default=5)
p.add_argument('--iters', type=int, default=101)
p.add_argument('--output', required=True)
a = p.parse_args()
bench._select_triton_tier('B')
cu = torch.cuda.get_device_properties(0).multi_processor_count

def kern_times(case, splits):
    out = torch.empty(case['q'].shape, device='cuda', dtype=torch.bfloat16)
    call, pos = bench._flydsl_call(case, out, splits)
    import math
    fp = sum(t.numel() * t.element_size() for t in pos)
    copies = max(2, math.ceil(bench.CACHE_TARGET_BYTES / fp))
    import copy
    rot = [copy.deepcopy(pos) for _ in range(copies - 1)] + [pos]
    for i in range(3): call(*rot[i % copies])
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for i in range(a.iters): call(*rot[i % copies])
        torch.cuda.synchronize()
    sp, cb = [], []
    for e in prof.events():
        if e.device_type != torch.autograd.DeviceType.CUDA: continue
        n = e.name.lower()
        if 'combine' in n: cb.append(e.device_time)
        elif 'elementwise' not in n and 'fill' not in n and 'copy' not in n: sp.append(e.device_time)
    return sp, cb, sorted({e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA})

rows = []
for cspec in a.cases.split(';'):
    b, ss = cspec.split(':')
    b = int(b); ss = [int(x) for x in ss.split(',')]
    spec = next(s for s in bench.sliding_specs(page_sizes=(a.page,))
                if s['mode'] == 'decode' and s['batch'] == b and s['context'] == a.context)
    case = make_case(spec, False)
    auto = _decode_splits(b, 1, a.context, a.page, cu, None)
    # correctness vs auto
    outs = {}
    valid = []
    for S in ss:
        try:
            o = torch.empty(case['q'].shape, device='cuda', dtype=torch.bfloat16)
            c, pos = bench._flydsl_call(case, o, S); c(*pos); torch.cuda.synchronize()
            outs[S] = o.float().clone(); valid.append(S)
        except Exception as ex:
            print(f'B{b} p{a.page} S={S} INVALID: {type(ex).__name__}: {str(ex)[:120]}', flush=True)
    o = torch.empty(case['q'].shape, device='cuda', dtype=torch.bfloat16)
    c, pos = bench._flydsl_call(case, o, None); c(*pos); torch.cuda.synchronize()
    ref = o.float()
    diff = {S: (outs[S] - ref).abs().max().item() for S in valid}
    print(f'B{b} p{a.page} auto={auto} maxdiff={diff}', flush=True)
    tot = {S: [] for S in valid}; tri = []; spl = {S: [] for S in valid}; cmb = {S: [] for S in valid}
    names = None
    for r in range(a.rounds):
        order = list(valid) + ['T']
        if r % 2: order.reverse()
        for v in order:
            if v == 'T':
                tri.append(bench._timing_row(case, 'triton', 'B', a.iters)['us'])
            else:
                tot[v].append(bench._timing_row(case, 'flydsl', f'S{v}', a.iters, v)['us'])
                sp, cb, names = kern_times(case, v)
                spl[v].append(st.median(sp) if sp else float('nan'))
                cmb[v].append(st.median(cb) if cb else 0.0)
    if names: print('kernel names', names[:6], flush=True)
    t = st.median(tri)
    for S in valid:
        rows.append(dict(batch=b, page=a.page, S=S, auto=auto, is_auto=int(S == auto), maxdiff=diff[S],
            split_us=st.median(spl[S]), combine_us=st.median(cmb[S]), total_us=st.median(tot[S]),
            triton_us=t, ratio=st.median(tot[S]) / t, rounds=a.rounds))
        print(rows[-1], flush=True)
    del case; torch.cuda.empty_cache()
with open(a.output, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
