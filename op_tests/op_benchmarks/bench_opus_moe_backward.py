# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Reproducible Opus/Triton backward screen; run in a ROCm torch environment."""
import argparse
import gc
import json
import statistics
import subprocess
from pathlib import Path
import aiter

import torch
from aiter.ops.opus import moe_backward as opus
from aiter.ops.triton.moe import moe_backward as tri


def make_case(shape):
    t, d, i, e, k = shape
    torch.manual_seed(20261003)
    ids = torch.rand(t, e).argsort(dim=1)[:, :k].contiguous()
    flat = ids.flatten()
    counts = torch.bincount(flat, minlength=e)
    padded = ((counts + 31) // 32) * 32
    offsets = torch.cat((torch.zeros(1, dtype=torch.int64), padded.cumsum(0)))
    compact = flat.argsort(stable=True)
    capacity = int(padded.sum())
    sorted_ids = torch.full((capacity,), t, dtype=torch.int32)
    sorted_experts = torch.zeros(capacity // 32, dtype=torch.int32)
    reverse = torch.empty(t * k, dtype=torch.int32)
    start = 0
    for expert in range(e):
        n = int(counts[expert])
        base = int(offsets[expert])
        routes = compact[start:start + n]
        sorted_ids[base:base + n] = ((routes // k) | ((routes % k) << 24)).int()
        reverse[routes] = torch.arange(base, base + n, dtype=torch.int32)
        sorted_experts[base // 32:int(offsets[expert + 1]) // 32] = expert
        start += n
    gpu = lambda value: value.cuda().contiguous()
    scores = torch.softmax(torch.randn(t, k, device='cuda'), dim=-1)
    md = opus.OpusMoeFixedMetadata(gpu(sorted_ids), gpu(sorted_experts),
        gpu(torch.tensor([capacity, capacity // 32], dtype=torch.int32)),
        gpu(reverse), gpu(offsets.int()), 32)
    tm = tri.TritonMoeBackwardMetadata(md.sorted_token_ids, torch.zeros(capacity, device='cuda'),
        md.sorted_expert_ids, md.num_valid_ids, md.reverse_sorted, gpu(compact.int()),
        gpu(torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0))).int()),
        md.expert_padded_offsets, 32)
    rand = lambda *s: torch.randn(*s, device='cuda', dtype=torch.bfloat16)
    x, dout, w1, w2 = rand(t, d), rand(t, d), rand(e, 2 * i, d), rand(e, d, i)
    w1.mul_(d ** -0.5)
    w2.mul_(i ** -0.5)
    z = rand(t * k, 2 * i)
    zs = torch.zeros(capacity, 2 * i, device='cuda', dtype=torch.bfloat16)
    zs[md.reverse_sorted.long()] = z
    return x, dout, w1, w2, scores, z, zs, md, tm, ids


def reference(case):
    x, dout, w1, w2, scores, z, zs, md, tm, ids = case
    t, d = x.shape
    e, _, i = w2.shape
    k = scores.shape[1]
    dx_routes = torch.empty(t * k, d, device='cuda', dtype=torch.bfloat16)
    dw1, dw2 = torch.zeros_like(w1), torch.zeros_like(w2)
    ds = torch.zeros_like(scores).flatten()
    for expert in range(e):
        routes = (ids.flatten() == expert).nonzero().flatten().cuda()
        tokens = routes // k
        gate, up = z[routes].float().chunk(2, dim=-1)
        sigmoid = gate.sigmoid()
        silu = gate * sigmoid
        a = silu * up
        acc = dout[tokens].float() @ w2[expert].float()
        q = acc * scores.flatten()[routes, None]
        dz = torch.cat((q * up * sigmoid * (1 + gate * (1 - sigmoid)), q * silu), dim=1).bfloat16()
        scaled = (a * scores.flatten()[routes, None]).bfloat16()
        ds[routes] = (acc * a).sum(dim=1)
        dx_routes[routes] = (dz.float() @ w1[expert].float()).bfloat16()
        dw1[expert] = (dz.float().T @ x[tokens].float()).bfloat16()
        dw2[expert] = (dout[tokens].float().T @ scaled.float()).bfloat16()
    return dx_routes.reshape(t, k, d).float().sum(1).bfloat16(), dw1, dw2, ds.reshape(t, k)


def errors(actual, expected):
    result = {}
    for name, a, b in zip(('dX', 'dW1', 'dW2', 'dScores'), actual, expected):
        af, bf = a.float(), b.float()
        rel = ((af - bf).norm() / bf.norm().clamp_min(1e-12)).item()
        assert torch.isfinite(af).all() and rel < 0.02, (name, rel)
        result[name] = {'relative_l2': rel, 'max_abs': (af - bf).abs().max().item()}
    return result


def elapsed(fn, iterations):
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000 / iterations


def measure(fn, rounds, iterations):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        held = fn()
    direct, captured = [], []
    for round_id in range(rounds):
        if round_id % 2:
            captured.append(elapsed(graph.replay, iterations))
            direct.append(elapsed(fn, iterations))
        else:
            direct.append(elapsed(fn, iterations))
            captured.append(elapsed(graph.replay, iterations))
    return {'direct_us': statistics.median(direct), 'graph_us': statistics.median(captured),
            'direct_rounds_us': direct, 'graph_rounds_us': captured}


def run(shape, rounds, iterations):
    case = make_case(shape)
    x, dout, w1, w2, scores, z, zs, md, tm, ids = case
    plain = lambda: opus.opus_moe_backward(dout, x, zs, w1, w2, scores, md)
    print('WARMUP', shape, flush=True)
    output = plain()
    torch.cuda.synchronize()
    ws = tri.allocate_moe_backward_workspace(x, w1, w2, scores)
    baseline = lambda: tri.triton_moe_backward_out(dout, x, z, w1, w2, scores, tm, ws)
    baseline()
    torch.cuda.synchronize()
    expected = reference(case) if shape[0] <= 1024 else (ws.d_x, ws.d_w1, ws.d_w2, ws.d_s)
    result = {'shape_T_D_I_E_K': shape, 'sorted_capacity': md.sorted_token_ids.numel(),
        'reference': 'FP32 PyTorch equations with BF16 intermediate rounding' if shape[0] <= 1024 else 'branch Triton baseline',
        'plain_errors': errors((output.d_x, output.d_w1, output.d_w2, output.d_scores), expected)}
    if shape[0] <= 1024:
        result['triton_errors'] = errors((ws.d_x, ws.d_w1, ws.d_w2, ws.d_s), expected)
    x_saved = opus.opus_moe_gather_x_blocked_g2(x, md.sorted_token_ids, md.num_valid_ids, block_m=32)
    cached = lambda: opus.opus_moe_backward(dout, x, zs, w1, w2, scores, md,
        saved_a_scaled=output.a_scaled, saved_x_sorted=x_saved, saved_x_sorted_blocked_g2=True)
    cache_output = cached()
    result['cached_errors'] = errors((cache_output.d_x, cache_output.d_w1, cache_output.d_w2, cache_output.d_scores), expected)
    print('CORRECTNESS', json.dumps(result), flush=True)
    for name, fn in [('opus_plain', plain), ('opus_forward_cached_blocked_g2', cached), ('triton', baseline)]:
        result[name] = measure(fn, rounds, iterations)
        flops = 12 * shape[0] * shape[4] * shape[1] * shape[2]
        result[name]['graph_gemm_tflops'] = flops / result[name]['graph_us'] / 1e6
        print('TIMING', name, json.dumps(result[name]), flush=True)
    result['cache_contract'] = 'sorting and forward cache production excluded; cached path receives zero-padded S*SwiGLU(Z) and blocked sorted X'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shapes', default='1024,512,256,8,4;16384,2048,1024,64,8;32768,2048,1024,64,8')
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=10)
    args = parser.parse_args()
    info = {'torch': torch.__version__, 'rocm': torch.version.hip, 'gpu': torch.cuda.get_device_name(),
        'commit': subprocess.check_output(['git', '-c', f'safe.directory={Path(aiter.__file__).resolve().parents[1]}', '-C', str(Path(aiter.__file__).resolve().parents[1]), 'rev-parse', 'HEAD'], text=True).strip(),
        'direct_contract': 'public Opus wrapper includes allocation/validation; Triton uses preallocated workspace',
        'graph_contract': 'captured backward only, allocation excluded for all paths', 'rounds': args.rounds, 'iterations': args.iterations}
    print('ENV', json.dumps(info), flush=True)
    results = []
    for shape in args.shapes.split(';'):
        results.append(run(tuple(map(int, shape.split(','))), args.rounds, args.iterations))
        gc.collect()
        torch.cuda.empty_cache()
    print('RESULT_JSON', json.dumps({'environment': info, 'results': results}), flush=True)
