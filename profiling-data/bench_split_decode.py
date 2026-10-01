"""Untracked: same-process split/unsplit/Triton benchmark using the suite helpers."""
import argparse
import csv
import torch
from rescale_hit_rate import make_case
from scripts import bench_unified_attention_gemma4 as bench
from aiter.ops.flydsl.kernels.flash_attn_fp8_gfx942 import _decode_splits

parser = argparse.ArgumentParser()
parser.add_argument('--page', type=int, required=True)
parser.add_argument('--context', type=int, required=True)
parser.add_argument('--batches', type=int, nargs='+', default=[1, 4, 16, 64])
parser.add_argument('--reps', type=int, default=3)
parser.add_argument('--sink', action='store_true')
parser.add_argument('--output', required=True)
args = parser.parse_args()
bench._select_triton_tier('B')
with open(args.output, 'w') as f:
    writer = None
    for batch in args.batches:
        spec = next(s for s in bench.sliding_specs(page_sizes=(args.page,))
                    if s['mode'] == 'decode' and s['batch'] == batch and s['context'] == args.context)
        case = make_case(spec, args.sink)
        bench._check_flydsl(case)
        for rep in range(args.reps):
            variants = [('unsplit', 'flydsl', 1), ('split', 'flydsl', None), ('triton-B', 'triton', None)]
            if rep % 2:
                variants.reverse()
            for name, backend, forced in variants:
                row = bench._timing_row(case, backend, name, 101, forced)
                row['rep'] = rep
                row['splits'] = _decode_splits(batch, 1, args.context, args.page,
                    torch.cuda.get_device_properties(0).multi_processor_count, forced) if backend == 'flydsl' else 0
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                f.flush()
                print(row, flush=True)
        del case
        torch.cuda.empty_cache()
