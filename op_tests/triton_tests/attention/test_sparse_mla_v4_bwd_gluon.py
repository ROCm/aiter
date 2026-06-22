"""
Correctness test for the Gluon backend of `sparse_mla_bwd_v4` (gfx950).

Checks `sparse_mla_bwd_v4(..., backend="gluon")` against:
  [1] the Triton backend (regression — same gradients within bf16 noise), and
  [2] torch autograd of the V4 reference forward.
"""
import math
import torch

from aiter.ops.triton._triton_kernels.attention.deepseek_sparse_attention_v4 import (
    sparse_mla_fwd_v4, sparse_mla_bwd_v4,
)

D_V, D_ROPE = 512, 64
D_QK = D_V + D_ROPE


def _ref_fwd(q, kv, topk, sink, scale):
    T, H, _ = q.shape
    inv = (topk < 0) | (topk >= T)
    idx = topk.clamp(0, T - 1).long()
    gk = kv.squeeze(1)[idx]
    S = torch.einsum("thd,tkd->thk", q, gk) * scale
    S = S.masked_fill(inv[:, None, :], float("-inf"))
    if sink is not None:
        sc = sink.view(1, H, 1).expand(T, H, 1)
        lse = torch.logsumexp(torch.cat([S, sc], dim=-1), dim=-1)
    else:
        lse = torch.logsumexp(S, dim=-1)
    P = torch.exp(S - lse[:, :, None])
    P = torch.where(inv[:, None, :], torch.zeros_like(P), P)
    return torch.einsum("thk,tkd->thd", P, gk[..., :D_V])


def _ref_grads(q, kv, topk, sink, do, scale):
    qf = q.detach().float().requires_grad_(True)
    kf = kv.detach().float().requires_grad_(True)
    sf = sink.detach().float().requires_grad_(True) if sink is not None else None
    o = _ref_fwd(qf, kf, topk, sf, scale)
    (o * do.float()).sum().backward()
    return qf.grad, kf.grad, (sf.grad if sf is not None else None)


def _chk(name, a, b, abs_tol, sig=1e-2, med=None, cos=None):
    d = (a.float() - b.float()).abs()
    m = b.float().abs() > sig
    rel = d[m] / b.float().abs()[m] if m.any() else d.new_zeros(0)
    mr = rel.median().item() if rel.numel() else 0.0
    av, bv = a.float().flatten(), b.float().flatten()
    ce = 1.0 - (av @ bv / (av.norm() * bv.norm() + 1e-30)).item()
    print(f"    {name}: max_abs={d.max().item():.3e} median_rel={mr:.3e} cos_err={ce:.3e}")
    assert d.max().item() < abs_tol, f"{name} abs {d.max().item()} > {abs_tol}"
    if med is not None and rel.numel():
        assert mr < med, f"{name} median_rel {mr} > {med}"
    if cos is not None:
        assert ce < cos, f"{name} cos_err {ce} > {cos}"


def _case(S, H, TOPK, sink_init):
    torch.manual_seed(0)
    q = torch.randn(S, H, D_QK, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(S, 1, D_QK, dtype=torch.bfloat16, device="cuda")
    do = torch.randn(S, H, D_V, dtype=torch.bfloat16, device="cuda")
    topk = torch.randint(0, S, (S, TOPK), dtype=torch.int32, device="cuda")
    sink = torch.full((H,), sink_init, dtype=torch.float32, device="cuda") if sink_init is not None else None
    scale = 1.0 / math.sqrt(D_QK)

    o, lse = sparse_mla_fwd_v4(q, kv, topk, attn_sink=sink, kv_lora_rank=D_V)
    dq_t, dkv_t, dsk_t = sparse_mla_bwd_v4(q, kv, o, do, topk, lse, attn_sink=sink, kv_lora_rank=D_V, backend="triton")
    dq_g, dkv_g, dsk_g = sparse_mla_bwd_v4(q, kv, o, do, topk, lse, attn_sink=sink, kv_lora_rank=D_V, backend="gluon")
    dq_r, dkv_r, dsk_r = _ref_grads(q, kv, topk, sink, do, scale)

    # gluon vs triton: same algorithm, differ only in bf16 rounding of the cross-chunk
    # RMW (gluon folds at store-time bf16, triton accumulates fp32). median_rel/cos_err
    # are the real signals; max_abs sits at the bf16 noise floor.
    print(f"  [S={S} H={H} TOPK={TOPK} sink={sink_init}] gluon vs triton")
    _chk("dQ ", dq_g, dq_t, 2e-2, sig=1e-3, med=5e-3, cos=1e-4)
    _chk("dKV", dkv_g, dkv_t, 5e-2, sig=1e-3, med=5e-3, cos=1e-4)
    print(f"  [S={S} H={H} TOPK={TOPK} sink={sink_init}] gluon vs autograd")
    _chk("dQ ", dq_g, dq_r, 5e-2, med=2e-2, cos=1e-3)
    _chk("dKV", dkv_g, dkv_r.view_as(dkv_g), 2e-1, med=2e-2, cos=1e-3)
    if sink is not None:
        _chk("dSk", dsk_g, dsk_r, 5e-1, med=5e-2, cos=1e-3)


def main():
    print(f"Arch: {torch.cuda.get_device_properties(0).gcnArchName}")
    _case(256, 64, 128, None)
    _case(256, 128, 256, 1.0)
    _case(512, 128, 640, -1.0)
    print("All gluon-backend tests PASSED.")


if __name__ == "__main__":
    main()
