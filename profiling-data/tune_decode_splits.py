"""Untracked: split-count sweep for the third decode tuning iteration."""
import argparse
import csv
import torch
from scripts import bench_unified_attention_gemma4 as bench

parser = argparse.ArgumentParser()
parser.add_argument('--page', type=int, required=True)
parser.add_argument('--context', type=int, required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
with open(args.output, 'w') as file:
    writer = None
    for batch in (4, 16, 64):
        spec = next(s for s in bench.sliding_specs(page_sizes=(args.page,))
                    if s['mode'] == 'decode' and s['batch'] == batch and s['context'] == args.context)
        case = bench._make_case(spec)
        for rep in range(3):
            splits = (1, 2, 4, 8, 16) if rep % 2 == 0 else (16, 8, 4, 2, 1)
            for split in splits:
                row = bench._timing_row(case, 'flydsl', 'tune', 101, split)
                row.update(rep=rep, splits=split)
                if writer is None:
                    writer = csv.DictWriter(file, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                file.flush()
                print(row, flush=True)
        del case
        torch.cuda.empty_cache()
