"""Untracked: correctness-only HEAD/prefetch comparison; never measures latency."""
import argparse
import importlib.util
from pathlib import Path
import subprocess

import torch

from scripts import bench_unified_attention_gemma4 as bench

ROOT = Path(__file__).resolve().parent.parent
SOURCE = 'aiter/ops/flydsl/kernels/flash_attn_fp8_gfx942.py'
TEMP = Path('/tmp/ua_repro/decode_prefetch_modules')


def variants():
    TEMP.mkdir(parents=True, exist_ok=True)
    modules = []
    for name, source in (
        ('head', subprocess.check_output(
            ['git', '-C', str(ROOT), 'show', f'e038cb3b6:{SOURCE}'], text=True)),
        ('prefetch', (ROOT / SOURCE).read_text()),
    ):
        path = TEMP / f'{name}.py'
        path.write_text(source)
        spec = importlib.util.spec_from_file_location(f'ua_prefetch_{name}', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    before = (TEMP / 'head.py').read_text()
    after = (TEMP / 'prefetch.py').read_text()
    assert before.split('    @fx.struct\n    class DecodeStorage:')[0] == after.split('    @fx.struct\n    class DecodeStorage:')[0]
    assert before.split('    @flyc.jit\n    def launch_decode(')[1] == after.split('    @flyc.jit\n    def launch_decode(')[1]
    print('Prefill, combine, and dispatch source byte-identical to HEAD', flush=True)
    return modules


def compare(case, modules, forced):
    outputs = []
    for module in modules:
        out = torch.empty_like(case['q'], dtype=torch.bfloat16)
        module.build_flash_attn_fp8_gfx942(case['spec']['page_size'])(
            case['q'], case['k'], case['v'], out,
            cu_seqlens_q=case['cu_q'], seqused_k=case['used_k'],
            max_seqlen_q=case['max_q'], max_seqlen_k=case['max_k'],
            block_table=case['block_table'], softmax_scale=256 ** -0.5,
            q_descale=case['scales'][0], k_descale=case['scales'][1],
            v_descale=case['scales'][2], _force_splits=forced,
        )
        outputs.append(out)
    torch.cuda.synchronize()
    assert torch.isfinite(outputs[1]).all()
    error = (outputs[0].float() - outputs[1].float()).abs().max().item()
    equal = torch.equal(*outputs)
    splits = modules[1]._decode_splits(
        case['spec']['batch'], 1, case['max_k'], case['spec']['page_size'],
        torch.cuda.get_device_properties(0).multi_processor_count, forced)
    print(f"CHECK B={case['spec']['batch']} page={case['spec']['page_size']} "
          f"lengths={case['kv_lens']} sink={case['sink']} "
          f"split={'auto' if forced is None else forced} actual={splits} "
          f"bitwise={equal} max_abs={error}", flush=True)
    assert equal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--page', type=int, default=32)
    parser.add_argument('--context', type=int, nargs='+', default=[100, 32768])
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--ragged', action='store_true')
    args = parser.parse_args()
    print(torch.cuda.get_device_properties(0), flush=True)
    modules = variants()
    checks = 0
    for context in args.context:
        lengths = ([1, 31, 32, 33, 63, 64, 65, 100, 1023, 1024, 1025, 32767]
                   if args.ragged else [context] * args.batch)
        spec = dict(name='prefetch-check', mode='decode', batch=len(lengths),
                    query_lens=[1] * len(lengths), kv_lens=lengths,
                    context=max(lengths), page_size=args.page)
        case = bench._make_case(spec)
        if args.ragged:
            # Permute valid page IDs and poison padding to expose tail overreads.
            for row, length in enumerate(lengths):
                count = (length + args.page - 1) // args.page
                case['block_table'][row, :count] = case['block_table'][row, :count].flip(0)
                case['block_table'][row, count:] = 0x7fffffff
        for sink in ([False] if args.smoke else [False, True]):
            case['sink'] = sink
            if sink:
                q = case['q'].float()
                q[..., :16] += 2.0
                case['q'] = q.to(torch.float8_e4m3fnuz)
                del q
                # Recurrent sink pages, vectorized to keep validation short.
                target = case['k'][::128 // args.page, :4, :, :16]
                target.copy_((target.float() + 5.0).to(case['k'].dtype))
            for forced in ([None] if args.smoke else [None, 3, 16]):
                compare(case, modules, forced)
                checks += 1
        del case
        torch.cuda.empty_cache()
    print(f'PASS {checks} bitwise HEAD comparisons', flush=True)


if __name__ == '__main__':
    main()
