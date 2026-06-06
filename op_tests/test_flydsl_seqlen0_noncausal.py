"""Test flydsl MHA kernel: seqlen_k==0 zero-fill and non-causal mode.

Usage:
    cd /app/aiter && python op_tests/test_flydsl_seqlen0_noncausal.py
"""
import math
import os
import sys

os.environ["ARCH"] = "gfx1250"
os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"
os.environ["ENABLE_CK"] = "0"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from aiter.ops.flydsl.mha_flydsl import flash_attn_varlen_flydsl

HEAD_QK = 192
HEAD_V = 128


def ref_varlen(q, k, v, cu_q, cu_k, scale, causal):
    B = cu_q.shape[0] - 1
    cu_q_cpu = cu_q.cpu().tolist()
    cu_k_cpu = cu_k.cpu().tolist()
    q_f = q.cpu().float()
    k_f = k.cpu().float()
    v_f = v.cpu().float()
    H = q.shape[1]
    Dv = v.shape[-1]
    parts = []
    for b in range(B):
        sq = cu_q_cpu[b + 1] - cu_q_cpu[b]
        sk = cu_k_cpu[b + 1] - cu_k_cpu[b]
        qb = q_f[cu_q_cpu[b]:cu_q_cpu[b + 1]]
        kb = k_f[cu_k_cpu[b]:cu_k_cpu[b + 1]]
        vb = v_f[cu_k_cpu[b]:cu_k_cpu[b + 1]]
        ref_b = torch.zeros(sq, H, Dv)
        for h in range(H):
            if sk == 0:
                pass  # ref_b stays zero
            else:
                qk = (qb[:, h, :] @ kb[:, h, :].T) * scale
                if causal:
                    mask = torch.triu(torch.ones(sq, sk, dtype=torch.bool),
                                      diagonal=sk - sq + 1)
                    qk = qk.masked_fill(mask, float('-inf'))
                p = torch.softmax(qk, dim=-1)
                ref_b[:, h, :] = p @ vb[:, h, :]
        parts.append(ref_b)
    return torch.cat(parts, dim=0)


def check(name, out, ref, rtol=1e-2, atol=1e-2):
    out_f = out.cpu().float()
    ref_f = ref.cpu().float()
    close = torch.isclose(out_f, ref_f, rtol=rtol, atol=atol)
    if close.all():
        print(f"  [{name}] \033[32mPASSED\033[0m")
        return True
    bad = (~close).sum().item()
    total = close.numel()
    diff = (out_f - ref_f).abs()
    print(f"  [{name}] \033[31mFAIL\033[0m  "
          f"max_err={diff.max():.6f}  bad={bad}/{total} ({bad/total:.1%})")
    return False


def test_seqlen_k_zero():
    """When seqlen_k == 0 for all batches, output must be all zeros."""
    print("=== seqlen_k == 0 tests ===")
    results = []
    for B, H, SQ in [(1, 1, 128), (1, 2, 256), (2, 1, 128), (2, 4, 300)]:
        scale = 1.0 / math.sqrt(HEAD_QK)
        torch.manual_seed(0)
        total_q = B * SQ
        q = torch.randn(total_q, H, HEAD_QK, dtype=torch.bfloat16, device='cuda')
        k = torch.empty(0, H, HEAD_QK, dtype=torch.bfloat16, device='cuda')
        v = torch.empty(0, H, HEAD_V, dtype=torch.bfloat16, device='cuda')
        cu_q = torch.arange(0, (B + 1) * SQ, SQ, dtype=torch.int32, device='cuda')
        cu_k = torch.zeros(B + 1, dtype=torch.int32, device='cuda')  # all zeros

        out = flash_attn_varlen_flydsl(
            q, k, v, cu_q, cu_k,
            max_seqlen_q=SQ, max_seqlen_k=0,
            softmax_scale=scale, causal=True,
        )
        torch.cuda.synchronize()
        ref = torch.zeros_like(out)
        results.append(check(f"sk=0 B={B} H={H} SQ={SQ}", out, ref))
    return results


def test_causal_sq_small():
    """Causal attention with sq << sk (decode-like)."""
    print("=== causal sq << sk tests ===")
    results = []
    for seqs_q, seqs_k, H in [
        ([72], [600], 1),
        ([72], [600], 2),
        ([1], [512], 1),
        ([1], [512], 2),
        ([16], [1024], 2),
        ([72, 72], [600, 600], 2),
        ([1, 128], [512, 1024], 2),
        ([72, 1], [600, 256], 4),
    ]:
        B = len(seqs_q)
        scale = 1.0 / math.sqrt(HEAD_QK)
        torch.manual_seed(42)
        total_q = sum(seqs_q)
        total_k = sum(seqs_k)
        max_sq = max(seqs_q)
        max_sk = max(seqs_k)

        q = torch.randn(total_q, H, HEAD_QK, dtype=torch.bfloat16, device='cuda')
        k = torch.randn(total_k, H, HEAD_QK, dtype=torch.bfloat16, device='cuda')
        v = torch.randn(total_k, H, HEAD_V, dtype=torch.bfloat16, device='cuda')

        cu_q = torch.zeros(B + 1, dtype=torch.int32, device='cuda')
        cu_k = torch.zeros(B + 1, dtype=torch.int32, device='cuda')
        for i in range(B):
            cu_q[i + 1] = cu_q[i] + seqs_q[i]
            cu_k[i + 1] = cu_k[i] + seqs_k[i]

        out = flash_attn_varlen_flydsl(
            q, k, v, cu_q, cu_k,
            max_seqlen_q=max_sq, max_seqlen_k=max_sk,
            softmax_scale=scale, causal=True,
        )
        torch.cuda.synchronize()
        ref = ref_varlen(q, k, v, cu_q, cu_k, scale, causal=True)
        tag = f"causal sq={seqs_q} sk={seqs_k} H={H}"
        results.append(check(tag, out, ref))
    return results


def test_seqlen_k_zero_mixed():
    """Mixed batch: some seqlen_k == 0, some > 0."""
    print("=== seqlen_k == 0 mixed batch tests ===")
    results = []
    for seqs_q, seqs_k, H, causal in [
        ([128, 128], [0, 128], 1, True),
        ([128, 256], [128, 0], 2, True),
        ([128, 128, 128], [0, 0, 128], 1, True),
    ]:
        B = len(seqs_q)
        scale = 1.0 / math.sqrt(HEAD_QK)
        torch.manual_seed(42)
        total_q = sum(seqs_q)
        total_k = sum(seqs_k)
        max_sq = max(seqs_q)
        max_sk = max(seqs_k)

        q = torch.randn(total_q, H, HEAD_QK, dtype=torch.bfloat16, device='cuda')
        k = torch.randn(total_k, H, HEAD_QK, dtype=torch.bfloat16, device='cuda')
        v = torch.randn(total_k, H, HEAD_V, dtype=torch.bfloat16, device='cuda')

        cu_q = torch.zeros(B + 1, dtype=torch.int32, device='cuda')
        cu_k = torch.zeros(B + 1, dtype=torch.int32, device='cuda')
        for i in range(B):
            cu_q[i + 1] = cu_q[i] + seqs_q[i]
            cu_k[i + 1] = cu_k[i] + seqs_k[i]

        out = flash_attn_varlen_flydsl(
            q, k, v, cu_q, cu_k,
            max_seqlen_q=max_sq, max_seqlen_k=max_sk,
            softmax_scale=scale, causal=causal,
        )
        torch.cuda.synchronize()
        ref = ref_varlen(q, k, v, cu_q, cu_k, scale, causal)
        tag = f"mixed sq={seqs_q} sk={seqs_k} H={H}"
        results.append(check(tag, out, ref))
    return results


def test_noncausal():
    """Non-causal attention: every query attends to all keys."""
    print("=== non-causal tests ===")
    results = []
    for seqs_q, seqs_k, H in [
        ([128], [128], 1),
        ([128], [256], 1),
        ([256], [128], 2),
        ([128, 256], [128, 256], 1),
        ([128, 256], [256, 384], 2),
        ([300], [300], 2),
        ([128, 128], [256, 256], 4),
        # sq << sk
        ([72], [600], 1),
        ([72], [600], 2),
        ([1], [512], 1),
        ([1], [512], 2),
        ([16], [1024], 2),
        ([72, 72], [600, 600], 2),
        ([1, 128], [512, 1024], 2),
        ([72, 1], [600, 256], 4),
    ]:
        B = len(seqs_q)
        scale = 1.0 / math.sqrt(HEAD_QK)
        torch.manual_seed(42)
        total_q = sum(seqs_q)
        total_k = sum(seqs_k)
        max_sq = max(seqs_q)
        max_sk = max(seqs_k)

        q = torch.randn(total_q, H, HEAD_QK, dtype=torch.bfloat16, device='cuda')
        k = torch.randn(total_k, H, HEAD_QK, dtype=torch.bfloat16, device='cuda')
        v = torch.randn(total_k, H, HEAD_V, dtype=torch.bfloat16, device='cuda')

        cu_q = torch.zeros(B + 1, dtype=torch.int32, device='cuda')
        cu_k = torch.zeros(B + 1, dtype=torch.int32, device='cuda')
        for i in range(B):
            cu_q[i + 1] = cu_q[i] + seqs_q[i]
            cu_k[i + 1] = cu_k[i] + seqs_k[i]

        out = flash_attn_varlen_flydsl(
            q, k, v, cu_q, cu_k,
            max_seqlen_q=max_sq, max_seqlen_k=max_sk,
            softmax_scale=scale, causal=False,
        )
        torch.cuda.synchronize()
        ref = ref_varlen(q, k, v, cu_q, cu_k, scale, causal=False)
        tag = f"noncausal sq={seqs_q} sk={seqs_k} H={H}"
        results.append(check(tag, out, ref))
    return results


if __name__ == "__main__":
    all_results = []
    all_results.extend(test_seqlen_k_zero())
    all_results.extend(test_causal_sq_small())
    all_results.extend(test_seqlen_k_zero_mixed())
    all_results.extend(test_noncausal())
    passed = sum(all_results)
    total = len(all_results)
    print(f"\n{'='*60}")
    print(f"Total: {passed}/{total} passed")
    print(f"{'='*60}")
