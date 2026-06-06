"""Fuzz flydsl vs triton MHA varlen with random inputs.

Records failing seeds for reproducibility.

Usage:
    cd /app/aiter && python op_tests/fuzz_flydsl_vs_triton.py [--iters 500] [--seed0 0]
"""
import argparse
import math
import os
import sys
import random

os.environ["ARCH"] = "gfx1250"
os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"
os.environ["ENABLE_CK"] = "0"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from aiter.ops.flydsl.kernels.mha_1250.fmha_kernel_gfx1250 import (
    flash_attn_varlen_d192_gfx1250,
)
from aiter.ops.triton.attention.mha import (
    flash_attn_varlen_func as flash_attn_varlen_func_triton,
)

HEAD_QK = 192
HEAD_V = 128
RTOL, ATOL = 5e-3, 5e-3


DATA_MODES = ['randn', 'spike', 'uniform_wide', 'mixed']


def _make_tensor(shape, dtype, device, mode, gen):
    """Generate tensor with different data distributions."""
    if mode == 'randn':
        return torch.randn(shape, dtype=dtype, device=device, generator=gen)
    elif mode == 'spike':
        t = torch.randn(shape, dtype=dtype, device=device, generator=gen)
        # Sprinkle large spikes (~1% of elements)
        mask = torch.rand(shape, device=device, generator=gen) < 0.01
        spikes = torch.randn(shape, dtype=dtype, device=device, generator=gen) * 50.0
        t[mask] = spikes[mask]
        return t
    elif mode == 'uniform_wide':
        return (torch.rand(shape, device=device, generator=gen) * 20.0 - 10.0).to(dtype)
    elif mode == 'mixed':
        # Half normal, half near-zero — triggers softmax saturation
        t = torch.randn(shape, dtype=dtype, device=device, generator=gen)
        mask = torch.rand(shape, device=device, generator=gen) < 0.5
        t[mask] = t[mask] * 0.001
        return t
    raise ValueError(f"unknown mode: {mode}")


def run_one(seed, seqs_q, seqs_k, H, causal, data_mode='randn'):
    """Returns None on pass, or dict with mismatch details (including seed)."""
    torch.manual_seed(seed)
    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)
    B = len(seqs_q)
    total_q = sum(seqs_q)
    total_k = sum(seqs_k)
    max_sq = max(seqs_q)
    max_sk = max(seqs_k)
    scale = 1.0 / math.sqrt(HEAD_QK)

    q = _make_tensor((total_q, H, HEAD_QK), torch.bfloat16, 'cuda', data_mode, gen)
    k = _make_tensor((total_k, H, HEAD_QK), torch.bfloat16, 'cuda', data_mode, gen)
    v = _make_tensor((total_k, H, HEAD_V), torch.bfloat16, 'cuda', data_mode, gen)

    cu_q = torch.zeros(B + 1, dtype=torch.int32, device='cuda')
    cu_k = torch.zeros(B + 1, dtype=torch.int32, device='cuda')
    for i in range(B):
        cu_q[i + 1] = cu_q[i] + seqs_q[i]
        cu_k[i + 1] = cu_k[i] + seqs_k[i]

    out_fly = flash_attn_varlen_d192_gfx1250(
        q, k, v, cu_q, cu_k,
        max_seqlen_q=max_sq, max_seqlen_k=max_sk,
        softmax_scale=scale, causal=causal,
    )

    out_tri = flash_attn_varlen_func_triton(
        q=q, k=k, v=v,
        cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
        max_seqlen_q=max_sq, max_seqlen_k=max_sk,
        softmax_scale=scale, causal=causal,
    )
    if isinstance(out_tri, tuple):
        out_tri = out_tri[0]
    torch.cuda.synchronize()

    fly_f = out_fly.cpu().float()
    tri_f = out_tri.cpu().float()
    close = torch.isclose(fly_f, tri_f, rtol=RTOL, atol=ATOL)
    if close.all():
        return None

    diff = (fly_f - tri_f).abs()
    bad = (~close).sum().item()
    total = close.numel()
    return {
        'seed': seed,
        'seqs_q': seqs_q,
        'seqs_k': seqs_k,
        'H': H,
        'causal': causal,
        'data_mode': data_mode,
        'max_err': diff.max().item(),
        'bad': bad,
        'total': total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--seed0', type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed0)
    failures = []
    n_pass = 0

    print(f"Fuzzing flydsl vs triton: {args.iters} iters, master_seed={args.seed0}")
    print(f"Thresholds: rtol={RTOL} atol={ATOL}")
    print("=" * 70)

    for i in range(args.iters):
        seed = rng.randint(0, 2**31 - 1)

        # Focused on log pattern: sq==sk, single batch, H=128, causal
        causal = True
        H = 128
        sq = rng.randint(400, 900)
        seqs_q = [sq]
        seqs_k = [sq]
        data_mode = rng.choice(DATA_MODES)

        tag = f"[{i+1}/{args.iters}] seed={seed} sq={sq} mode={data_mode}"

        try:
            result = run_one(seed, seqs_q, seqs_k, H, causal, data_mode=data_mode)
        except Exception as e:
            print(f"{tag}  ERROR: {e}")
            failures.append({'seed': seed, 'seqs_q': seqs_q, 'seqs_k': seqs_k,
                             'H': H, 'causal': causal, 'data_mode': data_mode,
                             'error': str(e)})
            continue

        if result is None:
            n_pass += 1
        else:
            pct = result['bad'] / result['total']
            print(f"{tag}  MISMATCH max_err={result['max_err']:.6f} "
                  f"bad={result['bad']}/{result['total']} ({pct:.4%})")
            failures.append(result)

    print()
    print("=" * 70)
    print(f"Results: {n_pass} passed, {len(failures)} failed out of {args.iters}")
    print("=" * 70)

    if failures:
        print("\nFailing seeds for reproduction:")
        for f in failures:
            mode = f.get('data_mode', '?')
            if 'error' in f:
                print(
                    f"  seed={f['seed']}  "
                    f"sq={f['seqs_q']} sk={f['seqs_k']} "
                    f"H={f['H']} causal={f['causal']} "
                    f"mode={mode}  ERROR: {f['error']}"
                )
            else:
                print(f"  seed={f['seed']}  sq={f['seqs_q']} sk={f['seqs_k']} "
                      f"H={f['H']} causal={f['causal']} mode={mode}  "
                      f"max_err={f['max_err']:.6f} bad={f['bad']}/{f['total']}")


if __name__ == "__main__":
    main()
