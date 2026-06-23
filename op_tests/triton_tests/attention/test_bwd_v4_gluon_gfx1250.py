"""End-to-end gfx1250 gluon backward (M4) vs CPU torch-autograd reference.

Validates `sparse_mla_bwd_v4(backend="gluon")` on gfx1250 (routes to the gfx1250
gluon kernels) — dQ, dKV, dSink — at no-sink / sink=0 / sink=+1. CPU reference
because GPU torch GEMM is unreliable on the gfx1250 A0 stack.

Run on a gfx1250 box:  python test_bwd_v4_gluon_gfx1250.py
"""
import math
import torch

from aiter.ops.triton._triton_kernels.attention.deepseek_sparse_attention_v4 import (
    sparse_mla_fwd_v4, sparse_mla_bwd_v4,
)


def _ref_grads_cpu(q, kv, topk_indices, attn_sink, do, kv_lora_rank, scale):
    """dq, dkv, d_sink via torch autograd on CPU."""
    total, num_heads, d_qk = q.shape
    d_v = kv_lora_rank
    qf = q.detach().float().cpu().requires_grad_(True)
    kvf = kv.detach().float().cpu().requires_grad_(True)
    sinkf = attn_sink.detach().float().cpu().requires_grad_(True) if attn_sink is not None else None
    idx = topk_indices.cpu()
    invalid = (idx < 0) | (idx >= total)
    safe = idx.clamp(0, total - 1).long()
    gathered = kvf.squeeze(1)[safe]
    S = torch.einsum("thd,tkd->thk", qf, gathered) * scale
    S = S.masked_fill(invalid[:, None, :], float("-inf"))
    if sinkf is not None:
        col = sinkf.view(1, num_heads, 1).expand(total, num_heads, 1)
        lse = torch.logsumexp(torch.cat([S, col], dim=-1), dim=-1)
    else:
        lse = torch.logsumexp(S, dim=-1)
    Pm = torch.exp(S - lse[:, :, None])
    Pm = torch.where(invalid[:, None, :], torch.zeros_like(Pm), Pm)
    O = torch.einsum("thk,tkd->thd", Pm, gathered[..., :d_v])
    (O * do.float().cpu()).sum().backward()
    return qf.grad, kvf.grad, (sinkf.grad if sinkf is not None else None)


def _chk(name, ours, ref, abs_tol, med_tol, cos_tol):
    o = ours.detach().float().cpu(); r = ref.detach().float().cpu()
    d = (o - r).abs(); m = r.abs() > 1e-2
    med = (d[m] / r.abs()[m]).median().item() if m.any() else 0.0
    a, b = o.flatten(), r.flatten()
    ce = 1.0 - (a @ b / (a.norm() * b.norm() + 1e-30)).item()
    ok = (d.max() < abs_tol) and (med < med_tol) and (abs(ce) < cos_tol)
    print(f"   [{'OK' if ok else 'XX'}] {name}: max_abs={d.max():.3e} median_rel={med:.3e} cos_err={ce:.3e}")
    assert ok, f"{name} mismatch"


def main():
    T, H, D_V, D_ROPE = 256, 64, 512, 64
    D_QK = D_V + D_ROPE; TOPK = 512; scale = 1.0 / math.sqrt(D_QK)
    for sv, tag in [(None, "no-sink"), (0.0, "sink=0"), (1.0, "sink=+1")]:
        torch.manual_seed(0)
        q = torch.randn(T, H, D_QK, dtype=torch.bfloat16, device="cuda")
        kv = torch.randn(T, 1, D_QK, dtype=torch.bfloat16, device="cuda")
        do = torch.randn(T, H, D_V, dtype=torch.bfloat16, device="cuda")
        topk = torch.randint(0, T, (T, TOPK), dtype=torch.int32, device="cuda")
        sink = None if sv is None else torch.full((H,), sv, dtype=torch.float32, device="cuda")
        o, lse = sparse_mla_fwd_v4(q, kv, topk, attn_sink=sink, kv_lora_rank=D_V)
        dq, dkv, ds = sparse_mla_bwd_v4(q, kv, o, do, topk, lse, attn_sink=sink,
                                        kv_lora_rank=D_V, backend="gluon")
        dq_r, dkv_r, ds_r = _ref_grads_cpu(q, kv, topk, sink, do, D_V, scale)
        print(f" [{tag}]")
        _chk("dQ ", dq, dq_r, 5e-2, 2e-2, 1e-3)
        _chk("dKV", dkv, dkv_r.view_as(dkv), 2e-1, 2e-2, 1e-3)
        if sink is not None:
            _chk("dSink", ds, ds_r, 5e-1, 5e-2, 1e-3)
    print("PASS")


if __name__ == "__main__":
    main()
