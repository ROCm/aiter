"""Untracked: same-session FlyDSL vs Triton Tier A/B remeasure (tier-cache fix applied locally)."""
import argparse, csv, statistics, sys
import torch
from torch.profiler import profile, ProfilerActivity
from rescale_hit_rate import make_case
from scripts import bench_unified_attention_gemma4 as bench
from aiter.ops.triton.utils import unified_attention_utils as uau


def select_tier(t):
    # local fix: also drop the tier-agnostic config lru_cache
    bench._select_triton_tier(t)
    uau._get_unified_attention_config_cached.cache_clear()


def kernel_names(case, tier):
    select_tier(tier)
    out = torch.empty(case['q'].shape, device='cuda', dtype=torch.bfloat16)
    call, pos = bench._triton_call(case, out)
    call(*pos); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        call(*pos); torch.cuda.synchronize()
    return [e.name for e in prof.events() if 'unified_attention' in e.name]


def correctness(case):
    outs = {}
    for key in ('A', 'B', 'F'):
        out = torch.empty(case['q'].shape, device='cuda', dtype=torch.bfloat16)
        if key == 'F':
            call, pos = bench._flydsl_call(case, out)
        else:
            select_tier(key)
            call, pos = bench._triton_call(case, out)
        call(*pos); torch.cuda.synchronize()
        outs[key] = out
    mx = outs['A'].float().abs().max().item()
    d = lambda a, b: (outs[a].float() - outs[b].float()).abs().max().item()
    return mx, d('F', 'A'), d('F', 'B'), d('A', 'B')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', required=True)
    p.add_argument('--page', type=int, required=True)
    p.add_argument('--contexts', required=True)
    p.add_argument('--batches', default='1')
    p.add_argument('--inputs', default='random')
    p.add_argument('--rounds', type=int, default=5)
    p.add_argument('--check', action='store_true')
    p.add_argument('--names', action='store_true')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    f = open(a.output, 'w', newline=''); w = None
    for ctx in map(int, a.contexts.split(',')):
        for b in (map(int, a.batches.split(',')) if a.mode == 'decode' else [1]):
            for inp in a.inputs.split(','):
                if a.mode == 'prefill':
                    spec = dict(name=f'prefill-{ctx}', mode='prefill', batch=1, query_lens=[ctx],
                                kv_lens=[ctx], context=ctx, page_size=a.page)
                else:
                    spec = dict(name=f'decode-b{b}-context{ctx}', mode='decode', batch=b,
                                query_lens=[1]*b, kv_lens=[ctx]*b, context=ctx, page_size=a.page)
                case = make_case(spec, inp == 'sink')
                row = dict(mode=a.mode, page=a.page, ctx=ctx, batch=b, input=inp)
                if a.names:
                    row['nameA'] = kernel_names(case, 'A')[:1]; row['nameB'] = kernel_names(case, 'B')[:1]
                if a.check:
                    row['maxref'], row['F-A'], row['F-B'], row['A-B'] = correctness(case)
                keys = ['F', 'A', 'B']
                s = {k: [] for k in keys}
                for rep in range(a.rounds + 1):
                    for k in (keys if rep % 2 == 0 else keys[::-1]):
                        if k == 'F':
                            r = bench._timing_row(case, 'flydsl', 'F', 101)
                        else:
                            select_tier(k)
                            r = bench._timing_row(case, 'triton', k, 101)
                        if rep: s[k].append(r['us'])
                med = {k: statistics.median(v) for k, v in s.items()}
                row.update(F_us=med['F'], A_us=med['A'], B_us=med['B'],
                           F_A=med['F']/med['A'], F_B=med['F']/med['B'],
                           spread_F=(max(s['F'])-min(s['F']))/med['F'])
                if w is None:
                    w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader()
                w.writerow(row); f.flush(); print('ROW', row, flush=True)
                del case
                torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
