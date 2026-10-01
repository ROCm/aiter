"""Untracked: cache-defeated, interleaved prefill comparisons and decode guards."""
import argparse
import csv
import torch
import prefill_baseline as baseline
from rescale_hit_rate import make_case
from scripts import bench_unified_attention_gemma4 as bench
from aiter.ops.flydsl.kernels import flash_attn_fp8_gfx942 as current

parser = argparse.ArgumentParser()
parser.add_argument('--output')
parser.add_argument('--quick', action='store_true')
parser.add_argument('--decode', action='store_true')
parser.add_argument('--sink', action='store_true')
parser.add_argument('--reps', type=int, default=3)
args = parser.parse_args()
bench._select_triton_tier('B')
original = current.build_flash_attn_fp8_gfx942
if args.decode:
    for page in (32, 64):
        for batch in (1, 16, 64):
            spec = next(s for s in bench.sliding_specs(page_sizes=(page,))
                        if s['name'] == f'decode-b{batch}-context32768')
            case = make_case(spec, args.sink)
            for splits in (None, 1, 2, 5, 16):
                outputs = []
                for builder in (baseline.build_flash_attn_fp8_gfx942, original):
                    current.build_flash_attn_fp8_gfx942 = builder
                    out = torch.empty_like(case['q'], dtype=torch.bfloat16)
                    call, positional = bench._flydsl_call(case, out, splits)
                    call(*positional)
                    outputs.append(out)
                torch.cuda.synchronize()
                assert torch.equal(*outputs), (page, batch, splits)
                print('BITWISE PASS', page, batch, splits, flush=True)
            del case, outputs, positional, out, call
            torch.cuda.empty_cache()
else:
    with open(args.output, 'w') as file:
        writer = None
        for page in ((32,) if args.quick else (32, 64)):
            for context in ((32768,) if args.quick else (1024, 4096, 16384, 32768)):
                spec = next(s for s in bench.sliding_specs(page_sizes=(page,))
                            if s['name'] == f'prefill-{context}')
                case = make_case(spec, args.sink)
                for rep in range(args.reps):
                    names = ('baseline', 'current', 'triton')
                    if rep % 2:
                        names = names[::-1]
                    for name in names:
                        current.build_flash_attn_fp8_gfx942 = (
                            baseline.build_flash_attn_fp8_gfx942 if name == 'baseline' else original)
                        row = bench._timing_row(case, 'triton' if name == 'triton' else 'flydsl', name, 101)
                        row['rep'] = rep
                        if writer is None:
                            writer = csv.DictWriter(file, fieldnames=list(row))
                            writer.writeheader()
                        writer.writerow(row)
                        file.flush()
                        print(row, flush=True)
                del case
                torch.cuda.empty_cache()
current.build_flash_attn_fp8_gfx942 = original
