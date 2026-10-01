"""Untracked: auto-mode A(HEAD 3db73864f) vs B(working tree) vs Triton, split heuristic retune."""
import csv, statistics, math
import torch
import ab_prefetch as ab
from ab_prefetch import bench, make_case

ab.OUTPUT = ab.Path('/tmp/ua_repro/ab_costmodel.csv')
mods = ab.load_variants()
chosen = {}
for n, m in mods.items():
    orig = m._decode_splits
    def wrap(*a, _o=orig, _n=n, **k):
        r = _o(*a, **k); chosen[_n] = r; return r
    m._decode_splits = wrap
bench._select_triton_tier('B')
BATCHES = (1, 4, 5, 8, 12, 16, 20, 24, 32, 40, 44, 48, 56, 64, 80, 96, 128)
cases = [(p, b, False, 32768) for p in (32, 64) for b in BATCHES] + [(64, 8, True, 32768), (64, 24, True, 32768)] + [(p, b, False, 600) for p in (32, 64) for b in (16, 64)]
ab.OUTPUT.parent.mkdir(parents=True, exist_ok=True)
w = None
with ab.OUTPUT.open('w', newline='') as f:
    for page, b, sink, ctx in cases:
        name = f'decode-b{b}-context{ctx}'
        spec = next((s for s in bench.sliding_specs(page_sizes=(page,)) if s['name'] == name), None)
        if spec is None:
            spec = dict(name=name, mode='decode', batch=b, query_lens=[1]*b,
                        kv_lens=[ctx]*b, context=ctx, page_size=page)
        case = make_case(spec, sink); case['sink'] = sink
        outs = {}
        for v in 'AB':
            ab.select_builder(mods, v)
            out = torch.empty_like(case['q'], dtype=torch.bfloat16)
            call, pos = bench._flydsl_call(case, out); call(*pos)
            torch.cuda.synchronize(); outs[v] = out
        S = {}
        for v in 'AB':
            ab.select_builder(mods, v)
            out = torch.empty_like(case['q'], dtype=torch.bfloat16)
            call, pos = bench._flydsl_call(case, out); chosen.pop(v, None); call(*pos)
            S[v] = chosen.get(v)
        md = (outs['A'].float() - outs['B'].float()).abs().max().item()
        samples = {'A': [], 'B': [], 'T': []}
        for rep in range(5):
            order = ('A', 'B', 'T') if rep % 2 == 0 else ('T', 'B', 'A')
            for v in order:
                if v != 'T': ab.select_builder(mods, v)
                samples[v].append(bench._timing_row(case, 'triton' if v == 'T' else 'flydsl', v, 101)['us'])
        a, bb, t = (statistics.median(samples[v]) for v in 'ABT')
        row = dict(page=page, ctx=ctx, input='sink' if sink else 'random', batch=b, S_A=S['A'], S_B=S['B'],
                   A_us=round(a, 2), B_us=round(bb, 2), BA=round(bb/a, 3), T_us=round(t, 2),
                   B_over_T=round(bb/t, 3), A_over_T=round(a/t, 3),
                   A_spread=round(max(samples['A'])/min(samples['A']), 3),
                   B_spread=round(max(samples['B'])/min(samples['B']), 3), maxdiff=round(md, 5))
        if w is None: w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader()
        w.writerow(row); f.flush(); print('ROW', row, flush=True)
        del case; torch.cuda.empty_cache()
