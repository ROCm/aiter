# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Replay frozen A05/A07 fixtures using this checkout and archive actual ISA.

Requires the external dsv41_single_layer recipe and its verified tensor fixture.
Run inside the shared GPU lock. No model weights or ATOM installation are needed;
--source supplies the frozen ATOM source used by the replay reference.
"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--fixture-root', type=Path, required=True)
    parser.add_argument('--fixture-id', required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ranks', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--rounds', type=int, default=3)
    args = parser.parse_args()
    from dsv41_single_layer.runtime import configure, require_lock
    require_lock()
    configure(args.source, args.cache_root)
    # The external recipe inserts its bundled AITER path. Select this checkout
    # before the first AITER import; later configure() calls retain that module.
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    import aiter
    if Path(aiter.__file__).resolve() != root / 'aiter/__init__.py':
        raise RuntimeError(f'Replay selected another AITER: {aiter.__file__}')
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['MONO_EVIDENCE_DIR'] = str(args.output / 'artifacts')
    from dsv41_single_layer.__main__ import main as replay
    rows = []
    base = ['--fixture-root', str(args.fixture_root), '--fixture-id', args.fixture_id]
    commands = [('verify', ['verify', *base])]
    for rank in args.ranks:
        for stage in ('A05', 'A07'):
            commands.append((f'{stage}-r{rank}', [
                'replay', *base, '--source', str(args.source),
                '--cache-root', str(args.cache_root), '--kernel', stage,
                '--rank', str(rank), '--rounds', str(args.rounds),
                '--candidate', 'op_tests.mono_rope.replay_candidate',
            ]))
    for name, command in commands:
        with (args.output / f'{name}.log').open('w') as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                status = replay([*command, '--output', str(args.output / name)])
        rows.append({'name': name, 'exit_code': status})
        report = {'complete': False, 'aiter': aiter.__file__, 'runs': rows}
        (args.output / 'suite.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(rows[-1]), flush=True)
        if status:
            raise SystemExit(status)
    report.update(complete=True, passed=True, physical_gpu=os.environ['GPU_LOCK_GPUS'])
    (args.output / 'suite.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
