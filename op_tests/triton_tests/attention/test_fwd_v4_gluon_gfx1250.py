"""gfx1250 gluon fwd (M-fwd) vs CPU reference. no-sink / sink=0 / sink=+1; O and LSE.
CPU ref because GPU torch GEMM is unreliable on the gfx1250 A0 stack.

Run on a gfx1250 box:  python test_fwd_v4_gluon_gfx1250.py
"""
import math
import torch

from aiter.ops.triton._gluon_kernels.gfx1250.attention.dsa_fwd_v4_gluon import (
    sparse_mla_fwd_v4_gluon_gfx1250,
)


def _ref_fwd_cpu(q, kv, topk, attn_sink, d_v, scale):
    total, H, d_qk = q.shape
    qf = q.float().cpu(); kvf = kv.float().cpu().squeeze(1); idx = topk.cpu()
    invalid = (idx < 0) | (idx >= total); safe = idx.clamp(0, total - 1).long()
    g = kvf[safe]
    S = torch.einsum("thd,tkd->thk", qf, g) * scale
    S = S.masked_fill(invalid[:, None, :], float("-inf"))
    if attn_sink is not None:
        col = attn_sink.float().cpu().view(1, H, 1).expand(total, H, 1)
        lse = torch.logsumexp(torch.cat([S, col], -1), -1)
    else:
        lse = torch.logsumexp(S, -1)
    P = torch.exp(S - lse[:, :, None])
    P = torch.where(invalid[:, None, :], torch.zeros_like(P), P)
    O = torch.einsum("thk,tkd->thd", P, g[..., :d_v])
    return O, lse


def _chk(name, a, b, atol, mtol, ctol):
    a = a.float().cpu(); b = b.float().cpu(); d = (a - b).abs(); m = b.abs() > 1e-2
    med = (d[m] / b.abs()[m]).median().item() if m.any() else 0.0
    fa, fb = a.flatten(), b.flatten(); ce = 1 - (fa @ fb / (fa.norm() * fb.norm() + 1e-30)).item()
    ok = d.max() < atol and med < mtol and abs(ce) < ctol
    print(f"   [{'OK' if ok else 'XX'}] {name}: max_abs={d.max():.3e} median_rel={med:.3e} cos_err={ce:.3e}")
    assert ok, f"{name} mismatch"


def main():
    T, H, D_V, D_ROPE = 256, 64, 512, 64
    D_QK = D_V + D_ROPE; TOPK = 512; scale = 1.0 / math.sqrt(D_QK)
    for sv, tag in [(None, "no-sink"), (0.0, "sink=0"), (1.0, "sink=+1")]:
        torch.manual_seed(0)
        q = torch.randn(T, H, D_QK, dtype=torch.bfloat16, device="cuda")
        kv = torch.randn(T, 1, D_QK, dtype=torch.bfloat16, device="cuda")
        topk = torch.randint(0, T, (T, TOPK), dtype=torch.int32, device="cuda")
        sink = None if sv is None else torch.full((H,), sv, dtype=torch.float32, device="cuda")
        o, lse = sparse_mla_fwd_v4_gluon_gfx1250(q, kv, topk, attn_sink=sink, kv_lora_rank=D_V)
        o_r, lse_r = _ref_fwd_cpu(q, kv, topk, sink, D_V, scale)
        print(f" [{tag}]")
        _chk("O  ", o, o_r, 5e-2, 1e-2, 1e-3)
        _chk("LSE", lse, lse_r, 1e-2, 1e-3, 1e-5)
    print("PASS")


if __name__ == "__main__":
    main()
