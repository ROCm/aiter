# SPDX-License-Identifier: MIT
"""ROUND-PAD2 real production dispatch -> real stock ISA, no devices or launches.

Fresh process per shape/env. Only _run and CPU stream lookup are intercepted;
production wrapper, selection, allocation, builder and compiler remain real.
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
p.add_argument('--shape', nargs=4, type=int, required=True, metavar=('ROWS','REQUESTS','COLUMNS','WIDTH'))
p.add_argument('--pad', choices=('unset','0','1'), required=True)
a = p.parse_args()
assert not Path('/dev/kfd').exists() and not Path('/dev/dri').exists()
os.environ['COMPILE_ONLY'] = '1'
os.environ['FLYDSL_GPU_ARCH'] = 'gfx1250'
os.environ['FLYDSL_DUMP_IR'] = '1'
os.environ['FLYDSL_DEBUG_DUMP_ASM'] = '1'
if a.pad == 'unset':
    os.environ.pop('QSA_PAD_BUCKET', None)
else:
    os.environ['QSA_PAD_BUCKET'] = a.pad
dest = a.results / a.case
dest.mkdir(parents=True, exist_ok=True)
os.environ['FLYDSL_DUMP_DIR'] = str(dest / 'isa')
os.environ['FLYDSL_RUNTIME_CACHE_DIR'] = str(dest / 'cache')
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx

def forbidden(*args, **kwargs):
    raise AssertionError('GPU discovery/runtime forbidden by ROUND-PAD2 no-device verification')
torch.cuda.current_stream = lambda *args, **kwargs: None
for name in ('is_available', 'device_count', 'init', '_lazy_init', 'synchronize'):
    setattr(torch.cuda, name, forbidden)
torch.set_num_threads(1)
spec = importlib.util.spec_from_file_location('qsa_pad2_real_source', a.source)
K = importlib.util.module_from_spec(spec)
spec.loader.exec_module(K)
rows, requests, columns, width = a.shape
q = torch.zeros((rows, 4, 128), dtype=torch.bfloat16)
size = 8 * 4 * 128 * (1 if a.dtype == 'fp8' else 2)
backing = torch.zeros(size + 128, dtype=torch.uint8)
offset = (-backing.data_ptr()) % 128
cache = backing[offset:offset+size].view(torch.float8_e4m3fn if a.dtype == 'fp8' else torch.bfloat16).view(8,4,1,128)
assert cache.data_ptr() % 128 == 0
scale = torch.ones((8,4,1), dtype=torch.float32) if a.dtype == 'fp8' else None
table = torch.zeros((requests,width), dtype=torch.int32)
t2r = torch.arange(rows, dtype=torch.int32) % requests
qp = torch.full((rows,), columns * 4 - 1, dtype=torch.int32)
ctx = torch.full((requests,), columns * 4, dtype=torch.int32)
logits = torch.empty((rows,columns), dtype=torch.float32)
vis = torch.empty(rows, dtype=torch.int32)
rst = torch.empty(rows, dtype=torch.int32)
original = (q, cache, scale, table, t2r, qp, ctx, logits, vis, rst)
rec = {'case':a.case, 'dtype':a.dtype, 'pad_env':a.pad,
       'true_dimensions':{'rows':rows,'columns':columns,'cache_pages':8,'requests':requests,'width':width},
       'source_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),
       'scope':'Real production wrapper and stock FlyDSL compiler; only _run and stream lookup intercepted; no shader/numerical/GPU execution'}
captures = []

def compile_only(launch, *args):
    tensors = args[:-8]
    gx, gy, r, c, pages, n, w = args[-8:-1]
    assert all(t.device.type == 'cpu' for t in tensors)
    cf = flyc.compile(launch, *args[:-1], fx.Stream(None))
    indexes = (0,1,3,4,5,6,7,8,9) if a.dtype == 'fp8' else (0,1,2,3,4,5,6,7,8)
    refs = (q,cache,table,t2r,qp,ctx,logits,vis,rst)
    reused = {name: tensors[i].data_ptr() == ref.data_ptr() for name,i,ref in
              zip(('q','cache','page_table','token_to_request','query_positions','context_lens','logits','visible','row_starts'), indexes, refs)}
    if a.dtype == 'fp8':
        reused['compressed_k_scale'] = tensors[2].data_ptr() == scale.data_ptr()
    captures.append({'kernel_name':launch.kernel_name, 'grid':[gx,gy],
       'launch_dimensions':{'rows':r,'columns':c,'cache_pages':pages,'requests':n,'width':w},
       'tensor_shapes':[list(t.shape) for t in tensors], 'pointer_reuse':reused,
       'compiled':True,'compiled_function':type(cf).__name__})
    # Initialize CPU outputs solely to let any real padding copyback complete.
    for out in tensors[-3:]:
        out.fill_(-math.inf if out.dtype == torch.float32 else 0)
K._run = compile_only
K.qsa_paged_mqa_logits_page4(q,cache,table,t2r,qp,ctx,max_columns=columns,
    compressed_k_scale=scale,logits=logits,visible_groups=vis,row_starts=rst)
assert len(captures) == 1, captures
rec.update(captures[0])
paths = sorted((dest/'isa').glob('*/21_final_isa.s'))
assert len(paths) == 1, paths
isa = paths[0]
text = isa.read_text()
rec['isa'] = {'path':str(isa), 'sha256':hashlib.sha256(isa.read_bytes()).hexdigest(), 'bytes':isa.stat().st_size}
rec['resources'] = {key:int(re.search(r'\.amdhsa_'+key+r'\s+(\d+)',text)[1])
                   for key in ('next_free_vgpr','next_free_sgpr','group_segment_fixed_size','private_segment_fixed_size')}
rec['gpu_devices_absent'] = True
(dest/'dispatch.json').write_text(json.dumps(rec,indent=2)+'\n')
print('PAD2_REAL_DISPATCH_PASS',a.pad,a.case,rec['kernel_name'],rec['isa']['sha256'],flush=True)
