"""Untracked: generate single-LDS decode scheduling experiments from the trim base."""
import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent
SOURCE = 'aiter/ops/flydsl/kernels/flash_attn_fp8_gfx942.py'
p = argparse.ArgumentParser()
p.add_argument('variant', choices=['both', 'both-late', 'k', 'k-late-v', 'v', 'none', 'none-fenced', 'k-fenced'])
args = p.parse_args()
s = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'c4c571f12:{SOURCE}'], text=True)
start = s.index('    @fx.struct\n    class DecodeStorage:')
end = s.index('    @flyc.jit\n    def launch_decode(', start)
x = s[start:end]
x = x.replace('        v: fx.Array[fx.Int8, 256 * 32, 16]\n', '')
x = x.replace('lds.v.ptr', 'lds.k.ptr')
# Split staging at the score/PV boundary; both views now alias the same allocation.
a = x.index('            for i in range_constexpr(8):', x.index('        for block, state'))
b = x.index('            gpu.barrier()\n', a)
stage = x[a:b]
kstage = stage[:stage.index('                vval =')] + stage[stage.index('                # XOR'):stage.index('                for j in')]
vstage = stage[:stage.index('                vval =')] + '                vval = fx.Vector(current_v[i])\n' + stage[stage.index('                for j in'):]
x = x[:a] + kstage + x[b:]
insert = x.index('            for shift in (16, 32):')
x = x[:insert] + '            gpu.barrier()\n' + vstage + '            gpu.barrier()\n' + x[insert:]
if args.variant.startswith('both'):
    x = x.replace('            scores = []', '            current_v = [state[19 + i * 2] for i in range(8)]\n            scores = []')
    if args.variant == 'both-late':
        x = x.replace('            next_tiles = prefetch(block + 1)\n', '')
        x = x.replace('            denom = fx.Float32(state[1])', '            next_tiles = prefetch(block + 1)\n            rocdl.sched_barrier(0)\n            denom = fx.Float32(state[1])')
else:
    # Each helper returns only one operand, preserving the guarded page lookup.
    x = x.replace('def prefetch(block):', 'def prefetch(block, ptr):')
    x = x.replace('for _ in range(16)]\n            # Guard', 'for _ in range(8)]\n            # Guard')
    x = x.replace('tiles.append(fx.Vector(_load(kp, src, fx.Vector.make_type(4, fx.Int32), 16)))\n                    tiles.append(fx.Vector(_load(vp, src, fx.Vector.make_type(4, fx.Int32), 16)))', 'tiles.append(fx.Vector(_load(ptr, src, fx.Vector.make_type(4, fx.Int32), 16)))')
    x = x.replace('state[18 + i * 2]', 'current_k[i]')
    x = x.replace('            next_tiles = prefetch(block + 1)\n', '')
    if args.variant.startswith('k'):
        x = x.replace('init = init + prefetch(start)', 'init = init + prefetch(start, kp)')
        x = x.replace('            block = fx.Int32(block)', '            block = fx.Int32(block)\n            current_k = state[18:]')
        if args.variant == 'k-late-v':
            x = x.replace('            gpu.barrier()\n' + vstage, '            current_v = prefetch(block, vp)\n            gpu.barrier()\n' + vstage)
        else:
            x = x.replace('            scores = []', '            current_v = prefetch(block, vp)\n            scores = []')
        x = x.replace('            denom = fx.Float32(state[1])', '            next_tiles = prefetch(block + 1, kp)\n            rocdl.sched_barrier(0)\n            denom = fx.Float32(state[1])')
    elif args.variant == 'v':
        x = x.replace('init = init + prefetch(start)', 'init = init + prefetch(start, vp)')
        x = x.replace('            block = fx.Int32(block)', '            block = fx.Int32(block)\n            current_k = prefetch(block, kp)\n            current_v = state[18:]')
        x = x.replace('            denom = fx.Float32(state[1])', '            next_tiles = prefetch(block + 1, vp)\n            rocdl.sched_barrier(0)\n            denom = fx.Float32(state[1])')
    else:
        x = x.replace('        init = init + prefetch(start)\n', '')
        x = x.replace('            block = fx.Int32(block)', '            block = fx.Int32(block)\n            current_k = prefetch(block, kp)')
        x = x.replace('            scores = []', '            current_v = prefetch(block, vp)\n            scores = []')
        x = x.replace(' + accum + next_tiles', ' + accum')
    if args.variant.endswith('fenced'):
        x = x.replace('            current_v = prefetch(block, vp)', '            rocdl.sched_barrier(0)\n            current_v = prefetch(block, vp)')
        x = x.replace('            gpu.barrier()\n' + vstage, '            gpu.barrier()\n            rocdl.sched_barrier(0)\n' + vstage)
        x = x.replace('            pfrag =', '            rocdl.sched_barrier(0)\n            pfrag =')
s = s[:start] + x + s[end:]
s = s.replace('# A one-wave WG uses 16 KiB LDS, allowing four resident WGs per CU.\n_DECODE_WGS_PER_CU = 4', '# One wave reuses 8 KiB LDS for K then V, allowing eight resident WGs per CU.\n_DECODE_WGS_PER_CU = 8')
(ROOT / SOURCE).write_text(s)
archive = ROOT / 'profiling-data' / 'decode-reuse-variants'
archive.mkdir(exist_ok=True)
(archive / f'{args.variant}.py').write_text(s)
print(args.variant)
