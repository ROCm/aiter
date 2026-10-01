"""Untracked: final decode specialization resource capture."""
import torch
from scripts import bench_unified_attention_gemma4 as bench
spec = next(s for s in bench.sliding_specs(page_sizes=(32,)) if s['name'] == 'decode-b1-context32768')
case = bench._make_case(spec)
out = torch.empty_like(case['q'], dtype=torch.bfloat16)
for splits in (1, 2, 4, 8, 16):
    call, positional = bench._flydsl_call(case, out, splits)
    call(*positional)
    torch.cuda.synchronize()
    print('SPECIALIZATION', splits, flush=True)
