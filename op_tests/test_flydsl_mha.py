#!/usr/bin/env python3
"""Unit test for FlyDSL MHA kernel (gfx1250 varlen flash attention)."""
import torch

torch.set_default_device("cuda")


def ref_attention(q, k, v, sm_scale):
    """Per-batch non-causal attention reference: q(SQ,H,D), k(SK,H,D), v(SK,H,Dv)."""
    q_h = q.float().permute(1, 0, 2)
    k_h = k.float().permute(1, 0, 2)
    v_h = v.float().permute(1, 0, 2)
    attn = torch.matmul(q_h, k_h.transpose(-2, -1)) * sm_scale
    attn = torch.softmax(attn, dim=-1)
    out = torch.matmul(attn, v_h)
    return out.permute(1, 0, 2)


def ref_varlen_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, sm_scale):
    batch = cu_seqlens_q.shape[0] - 1
    outs = []
    for b in range(batch):
        qs = cu_seqlens_q[b].item()
        qe = cu_seqlens_q[b + 1].item()
        ks = cu_seqlens_k[b].item()
        ke = cu_seqlens_k[b + 1].item()
        o_b = ref_attention(q[qs:qe], k[ks:ke], v[ks:ke], sm_scale)
        outs.append(o_b)
    return torch.cat(outs, dim=0)


def run_test(batch, nheads, seqlen_q, seqlen_k):
    from aiter.ops.flydsl.mha_flydsl import flash_attn_varlen_flydsl

    HEAD_QK = 192
    HEAD_V = 128
    sm_scale = 1.0 / (HEAD_QK ** 0.5)

    torch.manual_seed(42)
    total_q = batch * seqlen_q
    total_k = batch * seqlen_k
    q = torch.randn((total_q, nheads, HEAD_QK), dtype=torch.bfloat16)
    k = torch.randn((total_k, nheads, HEAD_QK), dtype=torch.bfloat16)
    v = torch.randn((total_k, nheads, HEAD_V), dtype=torch.bfloat16)
    cu_seqlens_q = torch.tensor([i * seqlen_q for i in range(batch + 1)], dtype=torch.int32)
    cu_seqlens_k = torch.tensor([i * seqlen_k for i in range(batch + 1)], dtype=torch.int32)

    ref = ref_varlen_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, sm_scale)

    out = flash_attn_varlen_flydsl(
        q, k, v, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q=seqlen_q,
        max_seqlen_k=seqlen_k,
        softmax_scale=sm_scale,
    )

    out_f = out.float().cpu()
    ref_f = ref.float().cpu()
    max_err = (out_f - ref_f).abs().max().item()
    print(f"  B={batch} H={nheads} SQ={seqlen_q} SK={seqlen_k}: max_err={max_err:.6f}")
    try:
        torch.testing.assert_close(out_f, ref_f, rtol=0.02, atol=0.015)
        print("  PASS")
    except AssertionError as e:
        print(f"  FAIL: {str(e)[:200]}")


if __name__ == "__main__":
    cases = [
        (1, 1, 128, 512),
        (1, 1, 128, 256),
        (2, 1, 128, 512),
        (1, 2, 128, 512),
        (2, 2, 128, 256),
    ]
    for batch, nheads, sq, sk in cases:
        print(f"\n--- test B={batch} H={nheads} SQ={sq} SK={sk} ---")
        run_test(batch, nheads, sq, sk)
