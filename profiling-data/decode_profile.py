"""Untracked: standalone driver for rocprofv3 runs of FlyDSL split decode vs Triton Tier B."""
import argparse, torch
from rescale_hit_rate import make_case
from scripts import bench_unified_attention_gemma4 as bench
from aiter.ops.flydsl.kernels.flash_attn_fp8_gfx942 import _decode_splits

p = argparse.ArgumentParser()
p.add_argument('--backend', choices=['flydsl', 'triton'], required=True)
p.add_argument('--page', type=int, required=True)
p.add_argument('--context', type=int, default=32768)
p.add_argument('--batch', type=int, required=True)
p.add_argument('--warmup', type=int, default=5)
p.add_argument('--iters', type=int, default=10)
a = p.parse_args()
bench._select_triton_tier('B')
spec = next(s for s in bench.sliding_specs(page_sizes=(a.page,))
            if s['mode'] == 'decode' and s['batch'] == a.batch and s['context'] == a.context)
case = make_case(spec, False)
out = torch.empty(case['q'].shape, device='cuda', dtype=torch.bfloat16)
call, pos = (bench._flydsl_call if a.backend == 'flydsl' else bench._triton_call)(case, out)
if a.backend == 'flydsl':
    print('SPLITS', _decode_splits(a.batch, 1, a.context, a.page,
          torch.cuda.get_device_properties(0).multi_processor_count, None), flush=True)
for _ in range(a.warmup):
    call(*pos)
torch.cuda.synchronize()
for _ in range(a.iters):
    call(*pos)
torch.cuda.synchronize()
print('DONE', out.abs().max().item())
