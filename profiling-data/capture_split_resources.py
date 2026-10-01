"""Untracked: capture compiler resources for the remaining decode specializations."""
import torch
from scripts import bench_unified_attention_gemma4 as bench

for splits in (1, 2, 5):
    spec = next(s for s in bench.sliding_specs(page_sizes=(32,))
                if s['name'] == 'decode-b1-context32768')
    case = bench._make_case(spec)
    out = torch.empty_like(case['q'], dtype=torch.bfloat16)
    call, args = bench._flydsl_call(case, out, splits)
    call(*args)
    torch.cuda.synchronize()
    print(f'resource specialization S={splits}', flush=True)
