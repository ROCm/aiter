"""Untracked: same-process A/B of maxnum and wave-uniform rescale skip."""
import csv
import importlib.util
from pathlib import Path
import statistics
import subprocess

import torch, copy, math
from torch.profiler import profile, ProfilerActivity

from rescale_hit_rate import make_case
from scripts import bench_unified_attention_gemma4 as bench
from aiter.ops.flydsl.kernels import flash_attn_fp8_gfx942 as current


ROOT = Path(__file__).resolve().parent.parent
TEMP = Path('/tmp/ua_ab')
OUTPUT = Path('/tmp/ua_repro/ab_prefetch.csv')
COMMITS = {'A': '3db73864f', 'B': ROOT / 'aiter/ops/flydsl/kernels/flash_attn_fp8_gfx942.py'}


def load_variants():
    TEMP.mkdir(parents=True, exist_ok=True)
    modules = {}
    for name, source_ref in COMMITS.items():
        source = (source_ref.read_text() if isinstance(source_ref, Path)
                  else subprocess.check_output([
                      'git', '-C', str(ROOT), 'show',
                      f'{source_ref}:aiter/ops/flydsl/kernels/flash_attn_fp8_gfx942.py',
                  ], text=True))
        path = TEMP / f'flash_attn_fp8_gfx942_{name}.py'
        path.write_text(source)
        spec = importlib.util.spec_from_file_location(f'ua_ab_{name}', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[name] = module
    assert (TEMP / 'flash_attn_fp8_gfx942_B.py').read_text().count('rescale = fx.Int64(rocdl.ballot(') == 1
    assert modules['A'].build_flash_attn_fp8_gfx942 is not modules['B'].build_flash_attn_fp8_gfx942
    print("Independent A/B modules loaded", flush=True)
    return modules


def select_builder(modules, name):
    # The suite imports this binding inside _flydsl_call; each module's recursive
    # build calls still resolve its own globals when auto split-KV is selected.
    current.build_flash_attn_fp8_gfx942 = modules[name].build_flash_attn_fp8_gfx942


def check_bitwise(case, modules):
    outputs = []
    for name in ('A', 'B'):
        select_builder(modules, name)
        out = torch.empty_like(case['q'], dtype=torch.bfloat16)
        call, positional = bench._flydsl_call(case, out)
        call(*positional)
        outputs.append(out)
    torch.cuda.synchronize()
    passed = torch.equal(*outputs)
    print(f"BITWISE {case['spec']['name']} page={case['spec']['page_size']} "
          f"{'sink' if case['sink'] else 'random'}: {passed}", flush=True)
    if not passed:
        raise AssertionError('A/B output differs bitwise')



def kern_times(case, iters=101):
    out = torch.empty(case['q'].shape, device='cuda', dtype=torch.bfloat16)
    call, pos = bench._flydsl_call(case, out)
    fp = sum(t.numel() * t.element_size() for t in pos)
    copies = max(2, math.ceil(bench.CACHE_TARGET_BYTES / fp))
    rot = [copy.deepcopy(pos) for _ in range(copies - 1)] + [pos]
    for i in range(3): call(*rot[i % copies])
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for i in range(iters): call(*rot[i % copies])
        torch.cuda.synchronize()
    sp, cb = [], []
    for e in prof.events():
        if e.device_type != torch.autograd.DeviceType.CUDA: continue
        n = e.name.lower()
        if 'combine' in n: cb.append(e.device_time)
        elif 'elementwise' not in n and 'fill' not in n and 'copy' not in n: sp.append(e.device_time)
    del rot
    return (statistics.median(sp) if sp else float('nan')), (statistics.median(cb) if cb else 0.0)


BATCHES = (1, 4, 8, 16, 24, 32, 64, 128)


def main():
    modules = load_variants()
    bench._select_triton_tier('B')
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    original = current.build_flash_attn_fp8_gfx942
    todo = [('prefill-4096', False)]
    try:
        with OUTPUT.open('w', newline='') as file:
            writer = None
            for page in (32, 64):
                cases = [(f'decode-b{b}-context32768', False) for b in BATCHES]
                cases += [(f'decode-b{b}-context32768', True) for b in (16, 64)]
                if page == 64:
                    cases = [('prefill-4096', False)] + cases
                for name, sink in cases:
                    if name.startswith('prefill') and page != 64:
                        continue
                    if name.startswith('decode') and sink and page != 64:
                        continue
                    spec = next((s for s in bench.sliding_specs(page_sizes=(page,)) if s['name'] == name), None)
                    if spec is None:
                        b = int(name.split('-b')[1].split('-')[0])
                        spec = dict(name=name, mode='decode', batch=b, query_lens=[1]*b,
                                    kv_lens=[32768]*b, context=32768, page_size=page)
                    case = make_case(spec, sink)
                    case['sink'] = sink
                    if name.startswith('decode'):
                        check_bitwise(case, modules)
                    for attempt in range(2):
                        samples = {'A': [], 'B': [], 'triton-B': []}
                        for rep in range(5):
                            order = ('A', 'B', 'triton-B')
                            if rep % 2:
                                order = order[::-1]
                            for variant in order:
                                if variant != 'triton-B':
                                    select_builder(modules, variant)
                                row = bench._timing_row(
                                    case, 'triton' if variant == 'triton-B' else 'flydsl',
                                    variant, 101)
                                samples[variant].append(row['us'])
                        a, b_, t = (statistics.median(samples[v]) for v in samples)
                        if abs(b_ / a - 1) > 0.02 or attempt:
                            break
                        print(f'RERUN {name} p{page} B/A={b_/a:.3f}', flush=True)
                    ks = {}
                    for v in ('A', 'B'):
                        select_builder(modules, v)
                        ks[v] = kern_times(case)
                    row = dict(page=page, case=name, input='sink' if sink else 'random',
                               A_us=a, B_us=b_, BA=b_/a, triton_us=t, B_over_triton=b_/t,
                               A_split=ks['A'][0], B_split=ks['B'][0], A_comb=ks['A'][1], B_comb=ks['B'][1],
                               A_rng=f"{min(samples['A']):.1f}-{max(samples['A']):.1f}",
                               B_rng=f"{min(samples['B']):.1f}-{max(samples['B']):.1f}", reran=attempt)
                    if writer is None:
                        writer = csv.DictWriter(file, fieldnames=list(row)); writer.writeheader()
                    writer.writerow(row); file.flush()
                    print('SUMMARY', row, flush=True)
                    del case
                    torch.cuda.empty_cache()
    finally:
        current.build_flash_attn_fp8_gfx942 = original
    print(f'CSV: {OUTPUT}', flush=True)


if __name__ == '__main__':
    main()
