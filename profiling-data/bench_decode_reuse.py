"""Untracked: interleaved single-LDS scheduling comparison, with rotated inputs."""
import argparse
import copy
import csv
import math
from pathlib import Path
import statistics

import torch
from torch.profiler import profile, ProfilerActivity
import ab_prefetch as ab


def split_times(case, split):
    out = torch.empty_like(case['q'], dtype=torch.bfloat16)
    call, pos = ab.bench._flydsl_call(case, out, split)
    footprint = sum(t.numel() * t.element_size() for t in pos)
    copies = max(2, math.ceil(ab.bench.CACHE_TARGET_BYTES / footprint))
    rot = [copy.deepcopy(pos) for _ in range(copies - 1)] + [pos]
    for i in range(10):
        call(*rot[i % copies])
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for i in range(21):
            call(*rot[i % copies])
        torch.cuda.synchronize()
    times = [e.device_time for e in prof.events()
             if e.device_type == torch.autograd.DeviceType.CUDA and 'decode' in e.name.lower()]
    assert times
    return statistics.median(times)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--page', type=int, required=True)
    p.add_argument('--batches', default='64')
    p.add_argument('--context', type=int, default=32768)
    p.add_argument('--splits', default='2,4')
    p.add_argument('--auto', action='store_true')
    p.add_argument('--variants', default='both,k,v,none')
    p.add_argument('--base', default='c4c571f12')
    p.add_argument('--rounds', type=int, default=5)
    p.add_argument('--kernel-times', action='store_true')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    names = args.variants.split(',')
    paths = ab.ROOT / 'profiling-data' / 'decode-reuse-variants'
    ab.COMMITS = {'A': args.base, 'B': paths / f'{names[0]}.py'}
    ab.COMMITS.update({name: paths / f'{name}.py' for name in names[1:]})
    ab.TEMP = Path('/tmp/ua_repro/reuse_ab_modules')
    mods = ab.load_variants()
    names[0] = 'B'
    ab.bench._select_triton_tier('B')
    cu = torch.cuda.get_device_properties(0).multi_processor_count
    print(torch.cuda.get_device_properties(0), flush=True)
    with open(args.output, 'w', newline='') as f:
        writer = None
        for b in map(int, args.batches.split(',')):
            spec = dict(name=f'decode-b{b}-context{args.context}', mode='decode', batch=b,
                        query_lens=[1]*b, kv_lens=[args.context]*b,
                        context=args.context, page_size=args.page)
            case = ab.make_case(spec, False)
            ss = [None] if args.auto else list(map(int, args.splits.split(',')))
            keys = [(v, s) for s in ss for v in mods] + [('T', None)]
            samples = {key: [] for key in keys}
            maxdiff = {}
            for s in ss:
                outs = []
                for v in mods:
                    ab.select_builder(mods, v)
                    out = torch.empty_like(case['q'], dtype=torch.bfloat16)
                    call, pos = ab.bench._flydsl_call(case, out, s)
                    for _ in range(10):
                        call(*pos)
                    outs.append(out)
                torch.cuda.synchronize()
                for v, out in zip(mods, outs):
                    maxdiff[v, s] = (outs[0].float() - out.float()).abs().max().item()
                    if s is not None:
                        assert torch.equal(outs[0], out), (b, args.page, v, s, maxdiff[v, s])
            for rep in range(args.rounds + 1):
                order = keys if rep % 2 == 0 else keys[::-1]
                for v, s in order:
                    if v != 'T':
                        ab.select_builder(mods, v)
                    us = ab.bench._timing_row(case, 'triton' if v == 'T' else 'flydsl',
                                              'B' if v == 'T' else v, 101, s)['us']
                    if rep:
                        samples[v, s].append(us)
            kt = {}
            if args.kernel_times:
                for s in ss:
                    for v in mods:
                        ab.select_builder(mods, v)
                        kt[v, s] = split_times(case, s)
            t = statistics.median(samples['T', None])
            for s in ss:
                a = statistics.median(samples['A', s])
                for v in names:
                    bb = statistics.median(samples[v, s])
                    row = dict(variant=args.variants.split(',')[0] if v == 'B' else v,
                               page=args.page, ctx=args.context, batch=b, S=s or 0,
                               S_A=mods['A']._decode_splits(b, 1, args.context, args.page, cu, s),
                               S_B=mods[v]._decode_splits(b, 1, args.context, args.page, cu, s),
                               A_us=a, B_us=bb, T_us=t, BA=bb/a, B_over_T=bb/t,
                               A_split=kt.get(('A', s)), B_split=kt.get((v, s)),
                               maxdiff=maxdiff[v, s], rounds=args.rounds)
                    if writer is None:
                        writer = csv.DictWriter(f, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    print('ROW', row, flush=True)
            f.flush()
            del case, outs, pos, out, call
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
