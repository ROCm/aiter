"""Measure running-max no-op rates without launching the attention kernel.

The synthetic sink variant adds a common Q direction and four matching K keys
per 128-token segment, so a sliding 1024-token window always contains sinks.
"""
import torch

from scripts import bench_unified_attention_gemma4 as bench
from aiter.ops.flydsl.kernels.flash_attn_fp8_gfx942 import _decode_splits


DEVICE = 'cuda'
SCALE = (256 ** -0.5) * 1.4426950408889634


def make_case(spec, sink):
    case = bench._make_case(spec)
    if sink:
        # Quantize modified activations just as the benchmark quantizes its random inputs.
        q = case['q'].float()
        q[..., :16] += 2.0
        case['q'] = q.to(torch.float8_e4m3fnuz)
        del q
        k = case['k']
        page = spec['page_size']
        # Recurrent sink keys keep late sliding windows representative of the skew.
        for seq, length in enumerate(spec['kv_lens']):
            page_base = int(case['block_table'][seq, 0].item())
            for pos in range(0, length, 128):
                target = k[page_base + pos // page, pos % page:pos % page + 4, :, :16]
                target.copy_((target.float() + 5.0).to(k.dtype))
    return case


def tally(scores, positions, blocks, totals, mode):
    # [heads, rows, tiles, 32]; all comparisons use scaled log2-space FP32 maxima.
    heads, rows, _ = scores.shape
    count = len(blocks)
    key_pos = torch.arange(blocks[0] * 32, (blocks[-1] + 1) * 32, device=DEVICE)
    valid = (key_pos[None, :] <= positions[:, None]) & (key_pos[None, :] > positions[:, None] - 1024)
    if mode == 'decode':
        valid &= key_pos[None, :] < 32768
    tile_max = scores.masked_fill(~valid[None, :, :], float('-inf')).reshape(
        heads, rows, count, 32).amax(-1)
    running = tile_max.cummax(-1).values
    # Kernel initializes max to -1e30; tile zero is never a skippable rescale.
    unchanged = running[:, :, 1:] == running[:, :, :-1]
    eligible = torch.ones_like(unchanged, dtype=torch.bool)
    if mode == 'prefill':
        wave_hit = unchanged.reshape(heads, 2, 32, count - 1).all(2)
        wave_eligible = eligible.reshape(heads, 2, 32, count - 1).all(2)
        wg_hit = wave_hit.all(1)
        wg_eligible = wave_eligible.all(1)
    else:
        # 16 MFMA rows repeat the two GQA heads eight times (row % 2).
        wave_hit = unchanged.all(1, keepdim=True)
        wave_eligible = eligible.all(1, keepdim=True)
        wg_hit = wave_hit[:, 0]
        wg_eligible = wave_eligible[:, 0]
    totals[:6] += torch.stack((
        unchanged.sum(), eligible.sum(), wave_hit.sum(), wave_eligible.sum(),
        wg_hit.sum(), wg_eligible.sum()))
    index = torch.arange(count - 1, device=DEVICE) / max(1, count - 2)
    for i, mask in enumerate((index <= .1, (index > .1) & (index < .9), index >= .9)):
        totals[6 + 2 * i] += wave_hit[..., mask].sum()
        totals[7 + 2 * i] += wave_eligible[..., mask].sum()


def measure_prefill(case, totals):
    length = case['spec']['context']
    q = case['q']
    k = case['k'].view(length, 16, 256)
    for qbase in range(0, length, 32):
        start = max(0, qbase - 1023) // 32
        end = (min(qbase + 32, length) + 31) // 32
        blocks = range(start, end)
        query = q[qbase:qbase + 32].float().reshape(32, 16, 2, 256)
        query = query.permute(1, 0, 2, 3).reshape(16, 64, 256)
        keys = k[start * 32:end * 32].float().permute(1, 2, 0)
        scores = torch.bmm(query, keys) * SCALE
        positions = (qbase + torch.arange(32, device=DEVICE)).repeat_interleave(2)
        tally(scores, positions, blocks, totals, 'prefill')


def measure_decode(case, totals):
    spec = case['spec']
    length = spec['context']
    splits = _decode_splits(spec['batch'], 1, length, 32,
                            torch.cuda.get_device_properties(0).multi_processor_count)
    start, end = max(0, length - 1024) // 32, (length + 31) // 32
    pages = end - start
    for seq in range(spec['batch']):
        page_base = int(case['block_table'][seq, 0].item())
        query = case['q'][seq].float().reshape(16, 2, 256)
        for split in range(splits):
            lo = max(start, start + pages * split // splits)
            hi = min(end, start + pages * (split + 1) // splits)
            blocks = range(lo, hi)
            keys = case['k'][page_base + lo:page_base + hi].reshape(-1, 16, 256)
            scores = torch.bmm(query, keys.float().permute(1, 2, 0)) * SCALE
            tally(scores, torch.full((2,), length - 1, device=DEVICE), blocks, totals, 'decode')
    return splits


def main():
    specs = bench.sliding_specs(page_sizes=(32,))
    for name in ('prefill-4096', 'prefill-32768',
                 'decode-b16-context32768', 'decode-b64-context32768'):
        spec = next(s for s in specs if s['name'] == name)
        for sink in (False, True):
            case = make_case(spec, sink)
            totals = torch.zeros(12, dtype=torch.int64, device=DEVICE)
            if spec['mode'] == 'decode':
                splits = measure_decode(case, totals)
            else:
                measure_prefill(case, totals)
                splits = 1
            values = totals.tolist()
            rate = lambda i: values[i] / values[i + 1] if values[i + 1] else float('nan')
            print(f'{name} {"sink" if sink else "random"} splits={splits} '
                  f'row={rate(0):.4%} wave={rate(2):.4%} wg={rate(4):.4%} '
                  f'wave_first10={rate(6):.4%} wave_middle={rate(8):.4%} '
                  f'wave_last10={rate(10):.4%} counts={values}', flush=True)
            del case
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
