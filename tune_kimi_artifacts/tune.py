"""Compare main with two N widths copied from matching DSV non-EP rows."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time


ROOT = Path('/app/aiter')
ARTIFACTS = Path(__file__).resolve().parent
PYTHON = Path('/workspace/trace/dsv41_flash_ep8_ep16_20261119/venv/bin/python')
CONFIG = ROOT / 'aiter/configs/model_configs/kimik3tp8_tuned_grouped_fmoe.csv'


def read_csv(path):
    """Read tuning rows without importing GPU modules."""
    with path.open() as source:
        reader = csv.DictReader(source)
        return reader.fieldnames, list(reader)


def write_csv(path, fields, rows):
    """Write a deterministic LF-separated CSV."""
    with path.open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def prepare():
    """Keep shape/M fixed and copy one matching non-EP schedule per row."""
    fields, baseline = read_csv(ARTIFACTS / 'main.csv')
    _, donors = read_csv(ARTIFACTS / 'donor.csv')
    donors = [row for row in donors if row['topk'] == '6'
              and row['expert'] == '96'
              and row['q_dtype_a'] == 'torch.float4_e2m1fn_x2']
    schedule_fields = fields[fields.index('tile_m'):fields.index('us')]
    mapping = []
    candidates = {256: [], 512: []}
    for row in baseline:
        matches = [donor for donor in donors
                   if donor['tile_m'] == row['tile_m']
                   and donor['tile_m2'] == row['tile_m2']]
        density = int(row['token']) * int(row['topk']) / int(row['expert'])
        donor = min(matches, key=lambda source: abs(math.log2(
            (int(source['token']) * 6 / 96) / density)))
        mapping.append({'token': int(row['token']),
                        'tile_m': int(row['tile_m']),
                        'donor_token': int(donor['token']),
                        'donor_experts': 96, 'donor_topk': 6})
        for width in candidates:
            candidate = row.copy()
            candidate.update({key: donor[key] for key in schedule_fields})
            candidate.update(tile_n='256', tile_n2=str(width),
                             us='0', tflops='0', bw='0')
            candidates[width].append(candidate)
    for width, rows in candidates.items():
        write_csv(ARTIFACTS / f'n{width}.csv', fields, rows)
    (ARTIFACTS / 'mapping.json').write_text(json.dumps(mapping, indent=2)+'\n')
    metadata = {
        'main_revision': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'flydsl_version': subprocess.check_output(
            [str(PYTHON), '-c', 'import importlib.metadata; '
             'print(importlib.metadata.version("flydsl"))'], text=True).strip(),
        'scope': 'Kimi3 TP8 A4W4, SiTUv2, all 15 CSV token buckets',
        'mapping': mapping,
        'main_csv_sha256': hashlib.sha256(
            (ARTIFACTS / 'main.csv').read_bytes()).hexdigest(),
    }
    (ARTIFACTS / 'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata, indent=2), flush=True)


def run(config, tokens, name):
    """Run native isolated-stage timing and existing correctness checks."""
    command = [str(PYTHON), '-u',
               'op_tests/flydsl_tests/test_flydsl_grouped_gemm.py',
               '--scenario', 'kernel', '--data-format', 'a4w4',
               '--experts', '896', '--tokens', *map(str, tokens),
               '--topk', '16', '--model-dim', '3584', '--inter-dim', '384',
               '--act', 'situv2', '--no-bias', '--iters', '101',
               '--no-check-aot-cache', '--data-init', 'uniform',
               '--scale-init', 'uniform', '--balanced-routing', '--seed', '0']
    overrides = {
        'PATH': str(PYTHON.parent) + ':' + os.environ['PATH'],
        'PYTHONPATH': str(ROOT), 'AITER_RUNTIME_GPU_ARCH': 'gfx1250',
        'AITER_USE_SYSTEM_TRITON': '1', 'ENABLE_CK': '0',
        'LLVM_GFX1250_ANY_XNACK_CODEGEN': '0', 'AITER_GROUPED_DEBUG': '0',
        'AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE': '1',
        'AITER_FLYDSL_TOKEN_MULTIDEST_QUANT': '1',
        'AITER_FLYDSL_MXFP4_WMMA_B2B': '1', 'AITER_LOG_MORE': '0',
        'FLYDSL_DUMP_IR': '0', 'AITER_MOE_EXPERT_BALANCE': 'true',
        'AITER_CONFIG_GROUPED_FMOE': str(ARTIFACTS / f'{config}.csv'),
    }
    # Remove incidental tuning overrides so main and candidates share defaults.
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('AITER_', 'FLYDSL_', 'LLVM_GFX1250'))}
    env.update(overrides)
    record = {'command': command, 'env': overrides, 'config': config,
              'tokens': tokens, 'name': name,
              'shared_lock': os.environ.get(
                  'KIMI_GPU_LOCK_PATH', '/var/lib/gpu-lock/amd-gpu.lock')}
    (ARTIFACTS / f'{name}.json').write_text(json.dumps(record, indent=2)+'\n')
    wrapper = Path('/usr/local/bin/gpu-lock')
    locked = wrapper.is_file() and os.access(wrapper, os.X_OK)
    launch = ['python3', str(ARTIFACTS / 'idle_exec.py'), *command]
    if locked:
        launch.insert(0, str(wrapper))
    record['launch_command'] = launch
    record['gpu_lock_installed'] = locked
    (ARTIFACTS / f'{name}.json').write_text(
        json.dumps(record, indent=2)+'\n')
    with (ARTIFACTS / f'{name}.log').open('w') as log:
        while True:
            print('Starting '+name+f' (GPU lock installed: {locked})', flush=True)
            result = subprocess.run(launch, cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT)
            if result.returncode != 75:
                break
            print('GPU busy; waiting before retrying '+name, flush=True)
            time.sleep(10)
    log = (ARTIFACTS / f'{name}.log').read_text()
    record['returncode'] = result.returncode
    if result.returncode:
        print(log[-5000:], flush=True)
        raise RuntimeError('Benchmark failed: '+name)
    assert log.count('| True ') == len(tokens), 'Missing correctness pass'
    assert 'config miss' not in log.lower(), 'Missing CSV lookup'
    assert 'no grouped CSV config matched' not in log, 'Missing CSV lookup'
    values = []
    for stage in ('gemm1', 'gemm2'):
        pattern = (rf'\[kernel-bench.*?{stage}: us = ([\d.]+|nan), '
                   rf'FLOPS = ([\d.]+), TFLOPS = ([\d.]+|nan), '
                   rf'Bandwidth = ([\d.]+|nan) GB/s')
        matches = re.findall(pattern, log)
        assert len(matches) == len(tokens), (name, stage, matches)
        values.append(matches)
    rows = []
    for index, token in enumerate(tokens):
        row = {'token': token, 'config': config, 'log': name,
               'correctness': 'PASS'}
        for stage, matches in zip(('gemm1', 'gemm2'), values):
            us, flops, tflops, bw = map(float, matches[index])
            row[stage+'_us'] = us if math.isfinite(us) else None
            row[stage+'_tflops'] = tflops if math.isfinite(tflops) else None
            row[stage+'_bw_gbs'] = bw if math.isfinite(bw) else None
        rows.append(row)
    rows = refine_native_table(rows, log)
    (ARTIFACTS / f'{name}_results.json').write_text(
        json.dumps(rows, indent=2, allow_nan=False)+'\n')
    print('Finished '+name, flush=True)
    return rows


def refine_native_table(rows, log):
    """Use the benchmark's native summary precision for stage medians."""
    header = next(line for line in log.splitlines()
                  if line.startswith('| data_format'))
    keys = [item.strip() for item in header.strip('|').split('|')]
    table = {}
    for line in log.splitlines():
        if not line.startswith('| a4w4'):
            continue
        record = dict(zip(keys, [item.strip()
                                for item in line.strip('|').split('|')]))
        table[int(record['tokens'])] = record
    for row in rows:
        record = table[row['token']]
        for stage in ('gemm1', 'gemm2'):
            for output, source in (('_us', '_us'), ('_tflops', '_TFLOPS'),
                                   ('_bw_gbs', '_Bandwidth (GB/s)')):
                value = float(record[stage+source])
                row[stage+output] = value if math.isfinite(value) else None
    return rows


def main():
    """Generate candidate files or run selected configurations and tokens."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--configs', nargs='+', default=['main','n256','n512'])
    parser.add_argument('--tokens', nargs='+', type=int)
    parser.add_argument('--tag', default='screen')
    args = parser.parse_args()
    if args.prepare:
        prepare()
        return
    _, rows = read_csv(ARTIFACTS / 'main.csv')
    tokens = args.tokens or [int(row['token']) for row in rows]
    for config in args.configs:
        run(config, tokens, f'{args.tag}_{config}')


if __name__ == '__main__':
    main()
