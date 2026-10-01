"""Wide split-count sweep on committed e038cb3b6 kernel (independent module) vs Triton Tier B."""
import argparse, csv, copy, importlib.util, math, statistics as st, subprocess, sys, hashlib
from pathlib import Path
import torch
from torch.profiler import profile, ProfilerActivity
ROOT = Path(__file__).resolve().parent.parent
REL = 'aiter/ops/flydsl/kernels/flash_attn_fp8_gfx942.py'
COMMIT = 'e038cb3b6'
src = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{COMMIT}:{REL}'], text=True)
tmp = Path('/tmp/ua_wide'); tmp.mkdir(exist_ok=True)
path = tmp / 'flash_attn_fp8_gfx942_e038cb3b6.py'; path.write_text(src)
assert hashlib.sha256(path.read_bytes()).hexdigest() == hashlib.sha256(src.encode()).hexdigest()
FULL = 'aiter.ops.flydsl.kernels.flash_attn_fp8_gfx942'
import aiter.ops.flydsl.kernels  # parent package
spec = importlib.util.spec_from_file_location(FULL, path)
mod = importlib.util.module_from_spec(spec); sys.modules[FULL] = mod; spec.loader.exec_module(mod)
setattr(sys.modules['aiter.ops.flydsl.kernels'], 'flash_attn_fp8_gfx942', mod)
from rescale_hit_rate import make_case
from scripts import bench_unified_attention_gemma4 as bench
import aiter.ops.flydsl.kernels.flash_attn_fp8_gfx942 as chk
assert chk is mod and chk.__file__ == str(path)
print('loaded', path, 'sha256', hashlib.sha256(src.encode()).hexdigest()[:16], flush=True)

p = argparse.ArgumentParser()
p.add_argument('--page', type=int, required=True)
p.add_argument('--batches', default='1,2,4,8,16,24,32,48,64,128')
p.add_argument('--contexts', default='1024,4096,32768')
p.add_argument('--splits', default='1,2,3,4,6,8')
p.add_argument('--rounds', type=int, default=3)
p.add_argument('--iters', type=int, default=101)
p.add_argument('--output', required=True)
a = p.parse_args()
bench._select_triton_tier('B')
cu = torch.cuda.get_device_properties(0).multi_processor_count

def kern_times(case, S):
    out = torch.empty(case['q'].shape, device='cuda', dtype=torch.bfloat16)
    call, pos = bench._flydsl_call(case, out, S)
    fp = sum(t.numel() * t.element_size() for t in pos)
    copies = max(2, math.ceil(bench.CACHE_TARGET_BYTES / fp))
    rot = [copy.deepcopy(pos) for _ in range(copies - 1)] + [pos]
    for i in range(3): call(*rot[i % copies])
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for i in range(a.iters): call(*rot[i % copies])
        torch.cuda.synchronize()
    sp, cb = [], []
    for e in prof.events():
        if e.device_type != torch.autograd.DeviceType.CUDA: continue
        n = e.name.lower()
        if 'combine' in n: cb.append(e.device_time)
        elif 'elementwise' not in n and 'fill' not in n and 'copy' not in n: sp.append(e.device_time)
    del rot
    return (st.median(sp) if sp else float('nan')), (st.median(cb) if cb else 0.0)

fields = ['page','batch','context','S','is_auto','split_us','combine_us','total_us','triton_us','ratio','maxdiff','rounds']
f = open(a.output, 'w', newline=''); w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
for ctx in map(int, a.contexts.split(',')):
  for b in map(int, a.batches.split(',')):
    sp_ = dict(name=f'decode-b{b}-context{ctx}', mode='decode', batch=b, query_lens=[1]*b,
               kv_lens=[ctx]*b, context=ctx, page_size=a.page)
    case = make_case(sp_, False)
    auto = mod._decode_splits(b, 1, ctx, a.page, cu, None)
    ss = [int(x) for x in a.splits.split(',')]
    if auto not in ss: ss.append(auto)
    valid, outs = [], {}
    for S in ss:
        try:
            o = torch.empty(case['q'].shape, device='cuda', dtype=torch.bfloat16)
            c, pos = bench._flydsl_call(case, o, S); c(*pos); torch.cuda.synchronize()
            outs[S] = o.float().clone(); valid.append(S)
        except Exception as ex:
            print(f'SKIP p{a.page} b{b} c{ctx} S={S}: {type(ex).__name__}: {str(ex)[:100]}', flush=True)
    o = torch.empty(case['q'].shape, device='cuda', dtype=torch.bfloat16)
    c, pos = bench._flydsl_call(case, o, None); c(*pos); torch.cuda.synchronize()
    ref = o.float()
    diff = {S: (outs[S] - ref).abs().max().item() for S in valid}
    tot = {S: [] for S in valid}; spl = {S: [] for S in valid}; cmb = {S: [] for S in valid}; tri = []
    for r in range(a.rounds):
        order = valid + ['T']
        if r % 2: order.reverse()
        for v in order:
            if v == 'T': tri.append(bench._timing_row(case, 'triton', 'B', a.iters)['us'])
            else:
                tot[v].append(bench._timing_row(case, 'flydsl', f'S{v}', a.iters, v)['us'])
                s_, c_ = kern_times(case, v); spl[v].append(s_); cmb[v].append(c_)
    t = st.median(tri)
    for S in valid:
        m = st.median(tot[S])
        w.writerow(dict(page=a.page, batch=b, context=ctx, S=S, is_auto=int(S == auto), split_us=st.median(spl[S]),
            combine_us=st.median(cmb[S]), total_us=m, triton_us=t, ratio=m/t, maxdiff=diff[S], rounds=a.rounds))
    f.flush()
    print(f'p{a.page} b{b} c{ctx} auto={auto} tri={t:.1f} ' + ' '.join(f'S{S}={st.median(tot[S]):.1f}' for S in valid), flush=True)
    del case; torch.cuda.empty_cache()
