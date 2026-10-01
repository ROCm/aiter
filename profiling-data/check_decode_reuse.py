"""Untracked: 86-case numerical parity check against the committed 16 KiB trim."""
import torch
import ab_prefetch as ab
from check_decode_lds import compare

ab.COMMITS['A'] = 'c4c571f12'
ab.COMMITS['B'] = ab.ROOT / 'aiter/ops/flydsl/kernels/flash_attn_fp8_gfx942.py'
ab.TEMP = ab.Path('/tmp/ua_repro/reuse_check_modules')
mods = list(ab.load_variants().values())
checks = 0
for page in (32, 64):
    cells = [(b, ctx, False) for b in (1, 16, 64) for ctx in (100, 32768)]
    cells.append((12, 32767, True))
    for b, ctx, ragged in cells:
        lengths = ([1,31,32,33,63,64,65,100,1023,1024,1025,32767]
                   if ragged else [ctx]*b)
        spec = dict(name='reuse-check', mode='decode', batch=b, query_lens=[1]*b,
                    kv_lens=lengths, context=max(lengths), page_size=page)
        case = ab.bench._make_case(spec)
        if ragged:
            for row, length in enumerate(lengths):
                count = (length + page - 1) // page
                case['block_table'][row, :count] = case['block_table'][row, :count].flip(0)
                case['block_table'][row, count:] = 0x7fffffff
        for sink in (False, True):
            case['sink'] = sink
            if sink:
                q = case['q'].float()
                q[..., :16] += 2.0
                case['q'] = q.to(torch.float8_e4m3fnuz)
                del q
                target = case['k'][::128 // page, :4, :, :16]
                target.copy_((target.float()+5.0).to(case['k'].dtype))
            for s in (None, 3, 16):
                compare(case, mods, s)
                checks += 1
        del case
        torch.cuda.empty_cache()
    spec = dict(name='reuse-unsplit', mode='decode', batch=1, query_lens=[1],
                kv_lens=[32768], context=32768, page_size=page)
    case = ab.bench._make_case(spec)
    case['sink'] = False
    compare(case, mods, 1)
    checks += 1
    del case
    torch.cuda.empty_cache()
print(f'PASS {checks}/86 bit-exact comparisons against c4c571f12', flush=True)
