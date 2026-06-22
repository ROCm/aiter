"""
Correctness test for the Gluon backend of `sparse_mla_fwd_v4` (gfx950).

Checks `sparse_mla_fwd_v4(..., backend="gluon")` against the Triton backend (O and LSE),
with and without attention sink.
"""
import math
import torch

from aiter.ops.triton._triton_kernels.attention.deepseek_sparse_attention_v4 import (
    sparse_mla_fwd_v4,
)

D_V, D_ROPE = 512, 64
D_QK = D_V + D_ROPE


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
    topk = torch.randint(0, S, (S, TOPK), dtype=torch.int32, device="cuda")
    sink = torch.full((H,), sink_init, dtype=torch.float32, device="cuda") if sink_init is not None else None

    o_t, lse_t = sparse_mla_fwd_v4(q, kv, topk, attn_sink=sink, kv_lora_rank=D_V, backend="triton")
    o_g, lse_g = sparse_mla_fwd_v4(q, kv, topk, attn_sink=sink, kv_lora_rank=D_V, backend="gluon")

    print(f"  [S={S} H={H} TOPK={TOPK} sink={sink_init}] gluon vs triton")
    _chk("O  ", o_g, o_t, 2e-2, sig=1e-2, med=1e-2, cos=1e-4)
    _chk("LSE", lse_g, lse_t, 5e-3, sig=1e-2, med=1e-4, cos=1e-6)


def main():
    print(f"Arch: {torch.cuda.get_device_properties(0).gcnArchName}")
    _case(256, 64, 128, None)
    _case(256, 128, 256, 1.0)
    _case(512, 128, 640, -1.0)
    _case(256, 64, 128, -1e3)   # sink dominates
    print("All gluon-fwd tests PASSED.")


if __name__ == "__main__":
    main()
