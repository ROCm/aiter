"""Untracked: bitwise baseline comparison and interleaved prefill timing."""
import csv
import torch
import decode16_baseline as baseline
from scripts import bench_unified_attention_gemma4 as bench

with open('/home/jograner/projects/aiter/unified-attention-gemma4/profiling-data/decode16-prefill.csv', 'w') as file:
    writer = None
    for page in (32, 64):
        for name in ('mixed', 'prefill-32768'):
            spec = next(s for s in bench.sliding_specs(page_sizes=(page,)) if s['name'] == name)
            case = bench._make_case(spec)
            old = torch.empty_like(case['q'], dtype=torch.bfloat16)
            new = torch.empty_like(old)
            call, positional = bench._flydsl_call(case, new)
            call(*positional)
            baseline.build_flash_attn_fp8_gfx942(page)(
                case['q'], case['k'], case['v'], old,
                cu_seqlens_q=case['cu_q'], seqused_k=case['used_k'], max_seqlen_q=case['max_q'],
                max_seqlen_k=case['max_k'], block_table=case['block_table'],
                softmax_scale=0.0625, q_descale=case['scales'][0],
                k_descale=case['scales'][1], v_descale=case['scales'][2])
            torch.cuda.synchronize()
            assert torch.equal(old, new), (page, name)
            print('BITWISE PASS', page, name, flush=True)
            if name == 'prefill-32768':
                for rep in range(3):
                    bench._select_triton_tier('B')
                    for backend in (('flydsl', 'triton') if rep % 2 == 0 else ('triton', 'flydsl')):
                        row = bench._timing_row(case, backend, backend, 101)
                        row['rep'] = rep
                        if writer is None:
                            writer = csv.DictWriter(file, fieldnames=list(row))
                            writer.writeheader()
                        writer.writerow(row)
                        file.flush()
                        print(row, flush=True)
            del case, old, new, positional, call
            torch.cuda.empty_cache()
