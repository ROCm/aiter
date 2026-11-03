# SPDX-License-Identifier: MIT
"""ROUND-NATIVE64 real production dispatch -> real stock ISA, no devices or launches.

Derived from compile_nosync_dispatch_cpu.py; generic over every launch layout
(WMMA bf16/fp8 and the scalar fallback). Specializations are selected with the
usual import-time QSA_* environment, set by the caller. ``--meta i32`` passes
int32 query_positions, ``--meta qp64`` int64 ones (golden/ATOM callers).

Fresh process per case. Only _run and CPU stream lookup are intercepted;
production wrapper, routing, builder and compiler remain real.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
p.add_argument('--results', type=Path, required=True)
p.add_argument('--case', required=True)
p.add_argument('--dtype', choices=('bf16', 'fp8'), required=True)
p.add_argument('--shape', nargs=3, type=int, required=True, metavar=('ROWS', 'REQUESTS', 'COLUMNS'))
p.add_argument('--meta', choices=('i32', 'qp64'), default='i32')
a = p.parse_args()
assert not Path('/dev/kfd').exists() and not Path('/dev/dri').exists()
os.environ['COMPILE_ONLY'] = '1'
os.environ['FLYDSL_GPU_ARCH'] = 'gfx1250'
os.environ['FLYDSL_DUMP_IR'] = '1'
os.environ['FLYDSL_DEBUG_DUMP_ASM'] = '1'
os.environ.pop('QSA_PAD_BUCKET', None)
dest = a.results / a.case
dest.mkdir(parents=True, exist_ok=True)
os.environ['FLYDSL_DUMP_DIR'] = str(dest / 'isa')
os.environ['FLYDSL_RUNTIME_CACHE_DIR'] = str(dest / 'cache')
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx


def forbidden(*args, **kwargs):
    raise AssertionError('GPU discovery/runtime forbidden by ROUND-NATIVE64 no-device verification')


torch.cuda.current_stream = lambda *args, **kwargs: None
for name in ('is_available', 'device_count', 'init', '_lazy_init', 'synchronize'):
    setattr(torch.cuda, name, forbidden)
torch.set_num_threads(1)
spec = importlib.util.spec_from_file_location('qsa_native64_real_source', a.source)
K = importlib.util.module_from_spec(spec)
spec.loader.exec_module(K)
rows, requests, columns = a.shape
width = columns // 4
q = torch.zeros((rows, 4, 128), dtype=torch.bfloat16)
size = 8 * 4 * 128 * (1 if a.dtype == 'fp8' else 2)
backing = torch.zeros(size + 128, dtype=torch.uint8)
offset = (-backing.data_ptr()) % 128
cache = backing[offset:offset + size].view(
    torch.float8_e4m3fn if a.dtype == 'fp8' else torch.bfloat16).view(8, 4, 1, 128)
scale = torch.ones((8, 4, 1), dtype=torch.float32) if a.dtype == 'fp8' else None
table = torch.zeros((requests, width), dtype=torch.int32)
t2r = torch.arange(rows, dtype=torch.int32) % requests
qp = torch.full((rows,), columns * 4 - 1, dtype=torch.int64 if a.meta == 'qp64' else torch.int32)
ctx = torch.full((requests,), columns * 4, dtype=torch.int32)
logits = torch.empty((rows, columns), dtype=torch.float32)
vis = torch.empty(rows, dtype=torch.int32)
rst = torch.empty(rows, dtype=torch.int32)
env = {k: v for k, v in sorted(os.environ.items()) if k.startswith('QSA_')}
rec = {'case': a.case, 'dtype': a.dtype, 'meta': a.meta, 'env': env,
       'shape': {'rows': rows, 'requests': requests, 'columns': columns, 'width': width},
       'source_sha256': hashlib.sha256(a.source.read_bytes()).hexdigest(),
       'scope': 'Real production wrapper and stock FlyDSL compiler; only _run and stream '
                'lookup intercepted; no shader/numerical/GPU execution'}
captures = []


def compile_only(launch, *args):
    tensors = [t for t in args if isinstance(t, torch.Tensor)]
    assert all(t.device.type == 'cpu' for t in tensors)
    flyc.compile(launch, *args[:-1], fx.Stream(None))
    qp_arg = next(t for t in tensors if t.data_ptr() == qp.data_ptr())
    captures.append({'kernel_name': launch.kernel_name,
                     'scalars': [x for x in args[:-1] if not isinstance(x, torch.Tensor)],
                     'qp_launch': {'dtype': str(qp_arg.dtype), 'shape': list(qp_arg.shape),
                                   'zero_copy': qp_arg.data_ptr() == qp.data_ptr()}})
    for out in (logits, vis, rst):
        out.fill_(-math.inf if out.dtype == torch.float32 else 0)


K._run = compile_only
K.qsa_paged_mqa_logits_page4(q, cache, table, t2r, qp, ctx, max_columns=columns,
                             compressed_k_scale=scale, logits=logits, visible_groups=vis,
                             row_starts=rst)
assert len(captures) == 1, captures
rec.update(captures[0])
paths = sorted((dest / 'isa').glob('*/21_final_isa.s'))
assert len(paths) == 1, paths
isa = paths[0]
text = isa.read_text()
body = [ln.strip() for ln in text.splitlines()]
instr = [ln for ln in body if ln and re.match(r'^[a-z_][a-z0-9_.]*(\s|$)', ln)
         and not ln.startswith('.')]
loc = sum(1 for ln in body if ln.startswith('.loc') or ln.startswith('.file'))
normal = '\n'.join(ln for ln in body if ln and not ln.startswith(('.loc', '.file', ';', '//')))
rec['isa'] = {'path': str(isa), 'sha256': hashlib.sha256(isa.read_bytes()).hexdigest(),
              'normalized_sha256': hashlib.sha256(normal.encode()).hexdigest(),
              'bytes': isa.stat().st_size, 'instructions': len(instr), 'loc_lines': loc,
              'mnemonics': {}}
for ln in instr:
    m = ln.split()[0]
    rec['isa']['mnemonics'][m] = rec['isa']['mnemonics'].get(m, 0) + 1
rec['resources'] = {key: int(re.search(r'\.amdhsa_' + key + r'\s+(\d+)', text)[1])
                    for key in ('next_free_vgpr', 'next_free_sgpr', 'group_segment_fixed_size',
                                'private_segment_fixed_size')}
rec['gpu_devices_absent'] = True
(dest / 'dispatch.json').write_text(json.dumps(rec, indent=2) + '\n')
print('NATIVE64_DISPATCH_PASS', a.meta, a.case, rec['kernel_name'], rec['isa']['sha256'], flush=True)
