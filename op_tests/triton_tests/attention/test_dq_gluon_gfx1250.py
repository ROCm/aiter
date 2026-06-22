"""Isolated correctness for the gfx1250 gluon dQ backward kernel (M1).

Compares the gluon dQ pass against a CPU torch-autograd reference (GPU torch GEMM
is unreliable on the gfx1250 A0 stack, so the reference is computed on CPU).
Multi-chunk (TOPK=128, R_CHUNK=64) to exercise the cross-chunk dQ RMW fold.

Run on a gfx1250 box:  python test_dq_gluon_gfx1250.py
"""
import math
import torch

from aiter.ops.triton._gluon_kernels.gfx1250.attention.dsa_bwd_dq import (
    sparse_mla_bwd_dq_gl_gfx1250,
)
from aiter.ops.triton._triton_kernels.attention.deepseek_sparse_attention_v4 import (
    sparse_mla_fwd_v4,
)


def _ref_dq_cpu(q, kv, topk_indices, do, kv_lora_rank, scale):
    """dQ via torch autograd on CPU (trusted)."""
    total, num_heads, d_qk = q.shape
    d_v = kv_lora_rank
    q_f = q.detach().float().cpu().requires_grad_(True)
    kv_f = kv.detach().float().cpu()
    idx = topk_indices.cpu()
    invalid = (idx < 0) | (idx >= total)
    safe = idx.clamp(0, total - 1).long()
    gathered = kv_f.squeeze(1)[safe]                       # [T, TOPK, D_QK]
    S = torch.einsum("thd,tkd->thk", q_f, gathered) * scale
    S = S.masked_fill(invalid[:, None, :], float("-inf"))
    lse = torch.logsumexp(S, dim=-1)
    P = torch.exp(S - lse[:, :, None])
    P = torch.where(invalid[:, None, :], torch.zeros_like(P), P)
    O = torch.einsum("thk,tkd->thd", P, gathered[..., :d_v])
    (O * do.float().cpu()).sum().backward()
    return q_f.grad


def main():
    dev = "cuda"
    T, H, D_V, D_ROPE = 128, 64, 512, 64
    D_QK = D_V + D_ROPE
    TOPK, R_CHUNK, TILE_K = 128, 64, 32       # 2 chunks -> RMW fold
    scale = 1.0 / math.sqrt(D_QK)

    torch.manual_seed(0)
    q = torch.randn(T, H, D_QK, dtype=torch.bfloat16, device=dev)
    kv = torch.randn(T, 1, D_QK, dtype=torch.bfloat16, device=dev)
    do = torch.randn(T, H, D_V, dtype=torch.bfloat16, device=dev)
    topk = torch.randint(0, T, (T, TOPK), dtype=torch.int32, device=dev)

    # lse from the Triton fwd (correct; tl.dot bypasses the broken torch GEMM); delta=rowsum(O*dO)
    o, lse = sparse_mla_fwd_v4(q, kv, topk, attn_sink=None, kv_lora_rank=D_V)
    delta = (o.float() * do.float()).sum(-1).contiguous()

    dq_g, _, _ = sparse_mla_bwd_dq_gl_gfx1250(
        q, kv, do, topk, lse, delta,
        R_CHUNK=R_CHUNK, topk=TOPK, kv_lora_rank=D_V, scale=scale,
        BLOCK_H=64, TILE_K=TILE_K,
    )
    torch.cuda.synchronize()

    dq_ref = _ref_dq_cpu(q, kv, topk, do, D_V, scale)
    g = dq_g.float().cpu(); r = dq_ref.float()
    diff = (g - r).abs()
    mask = r.abs() > 1e-2
    med_rel = (diff[mask] / r.abs()[mask]).median().item() if mask.any() else 0.0
    a, b = g.flatten(), r.flatten()
    cos_err = 1.0 - (a @ b / (a.norm() * b.norm() + 1e-30)).item()
    print(f"dQ: max_abs={diff.max():.3e} median_rel={med_rel:.3e} cos_err={cos_err:.3e}")
    assert med_rel < 2e-2 and abs(cos_err) < 1e-3, "gfx1250 gluon dQ mismatch"
    print("PASS")


if __name__ == "__main__":
    main()
