"""Untracked: isolate one scheduling variant for ISA resource capture."""
import argparse
from pathlib import Path
import torch
import ab_prefetch as ab

p = argparse.ArgumentParser()
p.add_argument('--variant', required=True)
p.add_argument('--page', type=int, required=True)
a = p.parse_args()
ab.COMMITS['A'] = 'c4c571f12'
ab.COMMITS['B'] = ab.ROOT / 'profiling-data' / 'decode-reuse-variants' / f'{a.variant}.py'
ab.TEMP = Path('/tmp/ua_repro/reuse_resource_modules')
mods = ab.load_variants()
ab.select_builder(mods, 'B')
spec = dict(name='resources', mode='decode', batch=1, query_lens=[1],
            kv_lens=[32768], context=32768, page_size=a.page)
case = ab.make_case(spec, False)
out = torch.empty_like(case['q'], dtype=torch.bfloat16)
call, pos = ab.bench._flydsl_call(case, out, 4)
call(*pos)
torch.cuda.synchronize()
assert torch.isfinite(out).all()
print('PASS', a.variant, a.page)
