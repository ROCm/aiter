"""Untracked: interleaved compact-LDS A/B against c2298cbba and Triton Tier B."""
import argparse
import csv
import statistics

import torch
import ab_prefetch as ab


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--page', type=int, required=True)
    p.add_argument('--batches', default='16,24,32,40,44,48,56,64,80,96,128,256')
    p.add_argument('--context', type=int, default=32768)
    p.add_argument('--splits', default='1,2,3,4,5,6,8')
    p.add_argument('--auto', action='store_true')
    p.add_argument('--rounds', type=int, default=5)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    ab.COMMITS['A'] = 'c2298cbba'
    ab.TEMP = ab.Path('/tmp/ua_repro/lds_ab_modules')
    mods = ab.load_variants()
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
            if not args.auto:
                for mod in mods.values():
                    s = mod._decode_splits(b, 1, args.context, args.page, cu)
                    if s not in ss:
                        ss.append(s)
            samples = {(v, s): [] for s in ss for v in 'AB'}
            samples[('T', None)] = []
            maxdiff = {}
            for s in ss:
                outs = []
                for v in 'AB':
                    ab.select_builder(mods, v)
                    out = torch.empty_like(case['q'], dtype=torch.bfloat16)
                    call, pos = ab.bench._flydsl_call(case, out, s)
                    for _ in range(10):
                        call(*pos)
                    outs.append(out)
                torch.cuda.synchronize()
                maxdiff[s] = (outs[0].float()-outs[1].float()).abs().max().item()
                if s is not None:
                    assert torch.equal(*outs), (b, args.page, s, maxdiff[s])
            keys = list(samples)
            # One discarded timing round also warms Triton and the rotated buffers.
            for rep in range(args.rounds + 1):
                order = keys if rep % 2 == 0 else keys[::-1]
                for v, s in order:
                    if v != 'T':
                        ab.select_builder(mods, v)
                    us = ab.bench._timing_row(case, 'triton' if v == 'T' else 'flydsl',
                                              'B' if v == 'T' else v, 101, s)['us']
                    if rep:
                        samples[v, s].append(us)
            t = statistics.median(samples['T', None])
            for s in ss:
                a, bb = (statistics.median(samples[v, s]) for v in 'AB')
                row = dict(page=args.page, ctx=args.context, batch=b, S=s or 0,
                           S_A=mods['A']._decode_splits(b, 1, args.context, args.page, cu, s),
                           S_B=mods['B']._decode_splits(b, 1, args.context, args.page, cu, s),
                           A_us=a, B_us=bb, T_us=t, BA=bb/a, B_over_T=bb/t,
                           maxdiff=maxdiff[s], rounds=args.rounds)
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
