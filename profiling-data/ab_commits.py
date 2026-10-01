"""Untracked: same-process A/B of maxnum and wave-uniform rescale skip."""
import csv
import importlib.util
from pathlib import Path
import statistics
import subprocess

import torch

from rescale_hit_rate import make_case
from scripts import bench_unified_attention_gemma4 as bench
from aiter.ops.flydsl.kernels import flash_attn_fp8_gfx942 as current


ROOT = Path(__file__).resolve().parent.parent
TEMP = Path('/tmp/ua_ab')
OUTPUT = Path('/tmp/ua_repro/ab_commits.csv')
COMMITS = {'A': 'ae58abba7', 'B': ROOT / 'aiter/ops/flydsl/kernels/flash_attn_fp8_gfx942.py'}


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
    assert 'rescale = fx.Int64(rocdl.ballot(' not in (TEMP / 'flash_attn_fp8_gfx942_A.py').read_text()
    assert (TEMP / 'flash_attn_fp8_gfx942_B.py').read_text().count('rescale = fx.Int64(rocdl.ballot(') == 1
    assert modules['A'].build_flash_attn_fp8_gfx942 is not modules['B'].build_flash_attn_fp8_gfx942
    print('Independent A/B modules: A has 0 rescale ballots; B has 1', flush=True)
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


def main():
    modules = load_variants()
    bench._select_triton_tier('B')
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    original = current.build_flash_attn_fp8_gfx942
    try:
        with OUTPUT.open('w', newline='') as file:
            writer = None
            for page in (32, 64):
                specs = bench.sliding_specs(page_sizes=(page,))
                for name in ('prefill-4096', 'prefill-32768',
                             'decode-b16-context32768', 'decode-b64-context32768'):
                    spec = next(s for s in specs if s['name'] == name)
                    for sink in (False, True):
                        case = make_case(spec, sink)
                        case['sink'] = sink
                        if name in ('prefill-4096', 'decode-b16-context32768'):
                            check_bitwise(case, modules)
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
                                    variant, 101,
                                )
                                row.update(input='sink' if sink else 'random', rep=rep)
                                if writer is None:
                                    writer = csv.DictWriter(file, fieldnames=list(row))
                                    writer.writeheader()
                                writer.writerow(row)
                                file.flush()
                                samples[variant].append(row['us'])
                                print(row, flush=True)
                        a, b, t = (statistics.median(samples[v]) for v in samples)
                        spread = ' '.join(
                            f'{v}=[{min(samples[v]):.1f},{max(samples[v]):.1f}]'
                            for v in samples
                        )
                        print(f"SUMMARY {name} page={page} {'sink' if sink else 'random'}: "
                              f'A={a:.1f} B={b:.1f} B/A={b/a:.3f} '
                              f'Triton={t:.1f} B/Triton={b/t:.3f} {spread}', flush=True)
                        del case
                        torch.cuda.empty_cache()
    finally:
        current.build_flash_attn_fp8_gfx942 = original
    print(f'CSV: {OUTPUT}', flush=True)


if __name__ == '__main__':
    main()
