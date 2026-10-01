"""Single-dispatch profiling driver; isolate each backend in a process."""
import argparse
import torch
from rescale_hit_rate import make_case
from scripts import bench_unified_attention_gemma4 as bench
from aiter.ops.flydsl.kernels.flash_attn_fp8_gfx942 import _decode_splits


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--backend', choices=['flydsl', 'triton'], required=True)
    p.add_argument('--page', type=int, required=True)
    p.add_argument('--batch', type=int, required=True)
    p.add_argument('--context', type=int, default=32768)
    p.add_argument('--iters', type=int, default=1)
    a = p.parse_args()
    bench._select_triton_tier('B')
    spec = dict(name=f'decode-b{a.batch}-context{a.context}', mode='decode',
                batch=a.batch, query_lens=[1] * a.batch, kv_lens=[a.context] * a.batch,
                context=a.context, page_size=a.page)
    case = make_case(spec, False)
    out = torch.empty_like(case['q'], dtype=torch.bfloat16)
    call, pos = (bench._flydsl_call if a.backend == 'flydsl' else bench._triton_call)(case, out)
    if a.backend == 'flydsl':
        print('SPLITS', _decode_splits(a.batch, 1, a.context, a.page,
              torch.cuda.get_device_properties(0).multi_processor_count, None), flush=True)
    for _ in range(a.iters):
        call(*pos)
    torch.cuda.synchronize()
    print('DONE', out.abs().max().item(), flush=True)


if __name__ == '__main__':
    main()
