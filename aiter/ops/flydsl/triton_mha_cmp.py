"""Triton MHA comparison helper for FlyDSL kernel validation."""

import torch


def compare_with_triton(
    flydsl_out, q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    **triton_kwargs,
):
    from ..triton.attention.mha import (
        flash_attn_varlen_func as flash_attn_varlen_func_triton,
    )
    tri_result = flash_attn_varlen_func_triton(
        q=q, k=k, v=v,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
        **triton_kwargs,
    )
    o_cmp = tri_result[0] if isinstance(tri_result, tuple) else tri_result
    close = torch.isclose(flydsl_out, o_cmp, rtol=5e-3, atol=5e-3)
    if close.all():
        print("[flydsl vs triton] MATCH")
    else:
        diff = (flydsl_out.float() - o_cmp.float()).abs()
        num_bad = (~close).sum().item()
        print(
            f"[flydsl vs triton] MISMATCH  "
            f"max_err={diff.max():.6f}  "
            f"bad={num_bad}/{flydsl_out.numel()}"
        )
