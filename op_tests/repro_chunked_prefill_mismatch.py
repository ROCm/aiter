"""Reproduce flydsl vs triton MISMATCH from 1.log line 2696-2697.

The failing case is chunked-prefill where cu_seqlens_q != cu_seqlens_k
(last batch has sq=461 but sk=701).

Usage:
    cd /app/aiter && python op_tests/repro_chunked_prefill_mismatch.py
"""
import math
import os
import sys

os.environ["ARCH"] = "gfx1250"
os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"
os.environ["ENABLE_CK"] = "0"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from aiter.ops.flydsl.kernels.mha_1250.fmha_kernel_gfx1250 import flash_attn_varlen_d192_gfx1250

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
                pass
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

    # Per-batch breakdown
    cu_q_cpu = cu_q.cpu().tolist()
    B = len(cu_q_cpu) - 1
    for b in range(B):
        ob = out_f[cu_q_cpu[b]:cu_q_cpu[b + 1]]
        rb = ref_f[cu_q_cpu[b]:cu_q_cpu[b + 1]]
        bc = torch.isclose(ob, rb, rtol=rtol, atol=atol)
        bbad = (~bc).sum().item()
        if bbad > 0:
            bd = (ob - rb).abs()
            print(f"    batch {b}: bad={bbad}/{bc.numel()} max_err={bd.max():.6f}")
    return False


# Exact case from 1.log line 2696-2697
seqs_q = [693, 692, 461]
seqs_k = [693, 692, 701]

print("=" * 60)
print("Reproduce chunked-prefill MISMATCH (1.log:2696)")
print(f"  seqs_q={seqs_q}  seqs_k={seqs_k}")
print(f"  cu_seqlens_q=[0, 693, 1385, 1846]")
print(f"  cu_seqlens_k=[0, 693, 1385, 2086]")
print("=" * 60)

B = len(seqs_q)
H = 128
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

print(f"\ncu_q={cu_q.cpu().tolist()}  cu_k={cu_k.cpu().tolist()}")
print(f"max_sq={max_sq}  max_sk={max_sk}  H={H}  causal=True\n")

out = flash_attn_varlen_d192_gfx1250(
    q, k, v, cu_q, cu_k,
    max_seqlen_q=max_sq, max_seqlen_k=max_sk,
    softmax_scale=scale, causal=True,
)
torch.cuda.synchronize()

ref = ref_varlen(q, k, v, cu_q, cu_k, scale, causal=True)
check("chunked-prefill sq!=sk", out, ref)

# Minimal repro: single batch with sq < sk
print("\n" + "=" * 60)
print("Minimal repro: single batch sq=461 sk=701 causal=True")
print("=" * 60)

torch.manual_seed(42)
sq_single, sk_single = 461, 701
q1 = torch.randn(sq_single, H, HEAD_QK, dtype=torch.bfloat16, device='cuda')
k1 = torch.randn(sk_single, H, HEAD_QK, dtype=torch.bfloat16, device='cuda')
v1 = torch.randn(sk_single, H, HEAD_V, dtype=torch.bfloat16, device='cuda')
cu_q1 = torch.tensor([0, sq_single], dtype=torch.int32, device='cuda')
cu_k1 = torch.tensor([0, sk_single], dtype=torch.int32, device='cuda')

out1 = flash_attn_varlen_d192_gfx1250(
    q1, k1, v1, cu_q1, cu_k1,
    max_seqlen_q=sq_single, max_seqlen_k=sk_single,
    softmax_scale=scale, causal=True,
)
torch.cuda.synchronize()

ref1 = ref_varlen(q1, k1, v1, cu_q1, cu_k1, scale, causal=True)
check("single-batch sq<sk", out1, ref1)
