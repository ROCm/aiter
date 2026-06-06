"""Reproduce flydsl vs triton from dumped tensors.

Usage:
    python3 repro_flydsl_dump.py /tmp/flydsl_mismatch_XXXX
"""

import os
import sys
import torch

_AITER_ROOT = '/app/aiter'
if _AITER_ROOT not in sys.path:
    sys.path.insert(0, _AITER_ROOT)
os.environ['ARCH'] = 'gfx1250'
os.environ['FLYDSL_RUNTIME_ENABLE_CACHE'] = '0'
os.environ['ENABLE_CK'] = '0'


def load_dump(d):
    tensors = {}
    for name in ['q', 'k', 'v', 'cu_seqlens_q', 'cu_seqlens_k',
                  'seqlens', 'softmax_scale', 'causal',
                  'out_flydsl', 'out_triton',
                  'dropout_p', 'deterministic', 'return_flags', 'window_size',
                  'bias', 'alibi_slopes', 'block_table', 'sink_ptr', 'out_pre']:
        p = os.path.join(d, f'{name}.pt')
        if os.path.exists(p):
            tensors[name] = torch.load(p, map_location='cpu')
    return tensors


def run_repro(dump_dir):
    t = load_dump(dump_dir)

    q = t['q'].cuda()
    k = t['k'].cuda()
    v = t['v'].cuda()
    cu_q = t['cu_seqlens_q'].cuda()
    cu_k = t['cu_seqlens_k'].cuda()
    seqlens = t['seqlens']
    max_sq = seqlens[0].item()
    max_sk = seqlens[1].item()
    scale = t['softmax_scale'].item()
    causal = bool(t['causal'].item())

    S, H = q.shape[0], q.shape[1]
    Dv = v.shape[-1]

    print(f'q: {q.shape} {q.dtype}')
    print(f'k: {k.shape} {k.dtype}')
    print(f'v: {v.shape} {v.dtype}')
    print(f'cu_q: {cu_q.tolist()}')
    print(f'cu_k: {cu_k.tolist()}')
    print(f'max_seqlen_q={max_sq} max_seqlen_k={max_sk}')
    print(f'scale={scale:.6f} causal={causal}')
    print()

    # Run flydsl
    from aiter.ops.flydsl.kernels.mha_1250.fmha_kernel_gfx1250 import (
        flash_attn_varlen_d192_gfx1250,
    )
    out_fly = torch.empty(S, H, Dv, dtype=torch.bfloat16, device='cuda')
    flash_attn_varlen_d192_gfx1250(q, k, v, cu_q, cu_k, max_sq, max_sk,
                             softmax_scale=scale, causal=causal, out=out_fly)
    torch.cuda.synchronize()

    # Run triton
    from aiter.ops.triton.attention.mha import (
        flash_attn_varlen_func as flash_attn_varlen_func_triton,
    )
    out_tri = flash_attn_varlen_func_triton(
        q=q, k=k, v=v,
        cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
        max_seqlen_q=max_sq, max_seqlen_k=max_sk,
        softmax_scale=scale, causal=causal,
    )
    torch.cuda.synchronize()

    # fp32 reference
    B = cu_q.shape[0] - 1
    cu_q_cpu = cu_q.cpu().tolist()
    cu_k_cpu = cu_k.cpu().tolist()
    q_f = q.cpu().float()
    k_f = k.cpu().float()
    v_f = v.cpu().float()
    ref_parts = []
    for b in range(B):
        sq = cu_q_cpu[b+1] - cu_q_cpu[b]
        sk = cu_k_cpu[b+1] - cu_k_cpu[b]
        qb = q_f[cu_q_cpu[b]:cu_q_cpu[b+1]]
        kb = k_f[cu_k_cpu[b]:cu_k_cpu[b+1]]
        vb = v_f[cu_k_cpu[b]:cu_k_cpu[b+1]]
        ref_b = torch.zeros(sq, H, Dv)
        for h in range(H):
            qk = (qb[:, h, :] @ kb[:, h, :].T) * scale
            if causal:
                mask = torch.triu(torch.ones(sq, sk, dtype=torch.bool), diagonal=1)
                qk = qk.masked_fill(mask, float('-inf'))
            p = torch.softmax(qk, dim=-1)
            ref_b[:, h, :] = p @ vb[:, h, :]
        ref_parts.append(ref_b)
    ref = torch.cat(ref_parts, dim=0)

    out_fly_f = out_fly.cpu().float()
    out_tri_f = out_tri.cpu().float()

    print('=== vs fp32 ref ===')
    fly_err = (out_fly_f - ref).abs()
    tri_err = (out_tri_f - ref).abs()
    print(f'flydsl vs ref:  max={fly_err.max():.6f}  mean={fly_err.mean():.6f}')
    print(f'triton vs ref:  max={tri_err.max():.6f}  mean={tri_err.mean():.6f}')

    print()
    print('=== flydsl vs triton (reproduced) ===')
    ft_diff = (out_fly_f - out_tri_f).abs()
    bad_ft = (~torch.isclose(out_fly_f, out_tri_f, rtol=1e-2, atol=1e-2)).sum().item()
    n = out_fly_f.numel()
    pct = bad_ft / n * 100
    print(
        f'max={ft_diff.max():.6f}  '
        f'mean={ft_diff.mean():.6f}  '
        f'bad={bad_ft}/{n} ({pct:.1f}%)'
    )

    # Compare with original dump
    if 'out_flydsl' in t and 'out_triton' in t:
        orig_fly = t['out_flydsl'].float()
        orig_tri = t['out_triton'].float()
        print()
        print('=== vs original dump ===')
        d1 = (out_fly_f - orig_fly).abs().max()
        d2 = (out_tri_f - orig_tri).abs().max()
        d3 = (out_fly_f - orig_tri).abs().max()
        print(
            f'flydsl repro vs dump_flydsl: '
            f'max_diff={d1:.6f}'
        )
        print(
            f'triton repro vs dump_triton: '
            f'max_diff={d2:.6f}'
        )
        print(
            f'flydsl repro vs dump_triton: '
            f'max_diff={d3:.6f}'
        )

        bad_orig = (~torch.isclose(
            orig_fly, orig_tri, rtol=1e-2, atol=1e-2
        )).sum().item()
        n_orig = orig_fly.numel()
        pct_orig = bad_orig / n_orig * 100
        print(
            f'dump_flydsl vs dump_triton:  '
            f'bad={bad_orig}/{n_orig} ({pct_orig:.1f}%)'
        )


if __name__ == '__main__':
    if len(sys.argv) < 2:
        # default: use latest dump
        import glob
        dirs = sorted(glob.glob('/tmp/flydsl_mismatch_*'), key=os.path.getmtime)
        if not dirs:
            print('No dump dirs found in /tmp/flydsl_mismatch_*')
            sys.exit(1)
        dump_dir = dirs[-1]
    else:
        dump_dir = sys.argv[1]

    print(f'Using dump: {dump_dir}')
    print('=' * 60)
    run_repro(dump_dir)
