# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Compare audited code bodies, descriptors and metadata across a migration."""
import argparse
import hashlib
import json
from pathlib import Path

from aiter.ops.mono_standard.elf import Elf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-suite', type=Path, required=True)
    parser.add_argument('--candidate-suite', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for stage in ('a05', 'a07'):
        reports = [json.loads(Path(str(suite) + '-audit-' + stage, 'audit.json').read_text())
                   for suite in (args.baseline_suite, args.candidate_suite)]
        if not all(report['passed'] and len(report['objects']) == 1 for report in reports):
            raise ValueError('Expected one audited code object per stage')
        objects = [report['objects'][0] for report in reports]
        old, new = [Elf(obj['binary']) for obj in objects]
        if any(hashlib.sha256(elf.data).hexdigest() != obj['sha256']
               for elf, obj in zip((old, new), objects)):
            raise ValueError('Audited binary hash changed')
        if objects[0]['kernels'] != objects[1]['kernels']:
            raise ValueError('Kernel argument/resource metadata changed')
        for kernel in objects[0]['kernels']:
            symbol = kernel['.name']
            body_equal = old.symbol_bytes(symbol) == new.symbol_bytes(symbol)
            before, after = old.symbol_bytes(symbol + '.kd'), new.symbol_bytes(symbol + '.kd')
            # Bytes 16:24 are the descriptor-relative code address. Relocation
            # may move that address without changing the argument or mode ABI.
            descriptor_equal = before[:16] + before[24:] == after[:16] + after[24:]
            if not body_equal or not descriptor_equal:
                raise ValueError(f'Machine code or descriptor changed: {symbol}')
            rows.append({'stage': stage.upper(), 'symbol': symbol,
                         'body_bytes': len(old.symbol_bytes(symbol)),
                         'body_sha256': hashlib.sha256(old.symbol_bytes(symbol)).hexdigest(),
                         'body_equal': body_equal, 'descriptor_equal': descriptor_equal,
                         'metadata_equal': True})
        if stage == 'a07':
            names = {name for name in old.symbols if 'inverse_rope_group_quant' in name
                     and 'persistent' not in name and not name.endswith('.kd')}
            new_names = {name for name in new.symbols if 'inverse_rope_group_quant' in name
                         and 'persistent' not in name and not name.endswith('.kd')}
            if names != new_names or any(old.symbol_bytes(name) != new.symbol_bytes(name) for name in names):
                raise ValueError('A07 fallback body changed')
            fallback_count = len(names)
    report = {'passed': True, 'persistent_bodies': rows, 'identical_a07_fallback_bodies': fallback_count}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
