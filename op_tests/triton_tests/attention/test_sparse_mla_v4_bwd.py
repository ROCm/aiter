"""
Correctness test for V4 backward kernel `sparse_mla_bwd_v4`.

Compares dQ, dKV, dSink against PyTorch autograd of the reference V4 forward.

Two scenarios:
  1. sink=None — V4 bwd must match V3.2 chunked_gather bwd exactly.
  2. sink set — gradients (dQ, dKV, dSink) must match torch autograd on the V4
     reference forward.
"""
import math

import torch

from aiter.ops.triton._triton_kernels.attention.deepseek_sparse_attention import (
    sparse_mla_bwd as sparse_mla_bwd_v3,
)
from aiter.ops.triton._triton_kernels.attention.deepseek_sparse_attention_v4 import (
    sparse_mla_fwd_v4,
    sparse_mla_bwd_v4,
)


# ---------------------------------------------------------------------
# Reference V4 forward built from differentiable torch ops
# ---------------------------------------------------------------------
def ref_sparse_mla_fwd_v4_diff(q, kv, topk_indices, attn_sink, kv_lora_rank, scale):
    """
    Same math as `ref_sparse_mla_fwd_v4` in test_sparse_mla_v4_fwd.py, but written
    so torch autograd can differentiate through it w.r.t. q, kv, attn_sink.

    Returns O [T, H, D_V] fp32.
    """
    total_tokens, num_heads, d_qk = q.shape
    topk = topk_indices.shape[1]
    d_v = kv_lora_rank

    invalid = (topk_indices < 0) | (topk_indices >= total_tokens)
    safe_idx = topk_indices.clamp(0, total_tokens - 1).long()

    kv_flat = kv.squeeze(1)                          # [T, D_QK]
    gathered_kv = kv_flat[safe_idx]                  # [T, TOPK, D_QK]

    S = torch.einsum("thd,tkd->thk", q, gathered_kv) * scale  # [T, H, TOPK]
    S = S.masked_fill(invalid[:, None, :], float("-inf"))

    if attn_sink is not None:
        sink_col = attn_sink.view(1, num_heads, 1).expand(total_tokens, num_heads, 1)
        S_ext = torch.cat([S, sink_col], dim=-1)
        lse_total = torch.logsumexp(S_ext, dim=-1)
    else:
        lse_total = torch.logsumexp(S, dim=-1)

    P = torch.exp(S - lse_total[:, :, None])
    P = torch.where(invalid[:, None, :], torch.zeros_like(P), P)
    V = gathered_kv[..., :d_v]
    O = torch.einsum("thk,tkd->thd", P, V)
    return O


# ---------------------------------------------------------------------
# Compute reference gradients via torch autograd
# ---------------------------------------------------------------------
def ref_grads_v4(q, kv, topk_indices, attn_sink, do, kv_lora_rank, scale):
    """Returns (dq, dkv, d_sink_or_None) computed by torch autograd."""
    q_f = q.detach().float().requires_grad_(True)
    kv_f = kv.detach().float().requires_grad_(True)
    if attn_sink is not None:
        sink_f = attn_sink.detach().float().requires_grad_(True)
    else:
        sink_f = None

    o_ref = ref_sparse_mla_fwd_v4_diff(q_f, kv_f, topk_indices, sink_f, kv_lora_rank, scale)
    loss = (o_ref * do.float()).sum()
    if sink_f is not None:
        loss.backward()
        return q_f.grad, kv_f.grad, sink_f.grad
    else:
        loss.backward()
        return q_f.grad, kv_f.grad, None


# ---------------------------------------------------------------------
# Compare helper
# ---------------------------------------------------------------------
def _check(name, ours, ref, abs_tol, rel_tol=None, sig_threshold=1e-2,
           median_rel_tol=None, cos_tol=None):
    """
    Multi-metric correctness check (same pattern as fwd test):
      - max_abs catches gross bugs.
      - median_rel + cos_err are the real signals (bulk drift + direction).
      - max_rel is reported but rarely asserted (dominated by bf16 noise tail).
    """
    diff = (ours.float() - ref.float()).abs()
    sig = ref.float().abs() > sig_threshold
    max_abs = diff.max().item()
    rel = diff[sig] / ref.float().abs()[sig] if sig.any() else diff.new_zeros(0)
    max_rel = rel.max().item() if rel.numel() else 0.0
    median_rel = rel.median().item() if rel.numel() else 0.0
    a, b = ours.float().flatten(), ref.float().flatten()
    # Skip cos_err when both vectors are essentially zero (cosine is undefined);
    # max_abs already covers that case.
    both_zero = (a.norm() < 1e-6) and (b.norm() < 1e-6)
    cos_sim = (a @ b / (a.norm() * b.norm() + 1e-30)).item()
    cos_err = 0.0 if both_zero else (1.0 - cos_sim)
    zero_tag = " (both-zero)" if both_zero else ""
    print(f"    {name}: max_abs={max_abs:.4e}  max_rel={max_rel:.4e}  "
          f"median_rel={median_rel:.4e}  cos_err={cos_err:.4e}{zero_tag}")
    assert max_abs < abs_tol, f"{name} abs error {max_abs} > {abs_tol}"
    if median_rel_tol is not None and rel.numel():
        assert median_rel < median_rel_tol, \
            f"{name} median rel error {median_rel} > {median_rel_tol}"
    if cos_tol is not None and not both_zero:
        assert cos_err < cos_tol, f"{name} cos error {cos_err} > {cos_tol}"
    if rel_tol is not None and rel.numel():
        assert max_rel < rel_tol, f"{name} max rel error {max_rel} > {rel_tol}"


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------
def test_v3_regression(B, S, H, D_V, D_ROPE, TOPK, device):
    """V4 bwd with sink=None must equal V3.2 chunked_gather bwd."""
    torch.manual_seed(0)
    D_QK = D_V + D_ROPE
    total = B * S
    q = torch.randn(total, H, D_QK, dtype=torch.bfloat16, device=device)
    kv = torch.randn(total, 1, D_QK, dtype=torch.bfloat16, device=device)
    topk_indices = torch.randint(0, total, (total, TOPK), dtype=torch.int32, device=device)
    do = torch.randn(total, H, D_V, dtype=torch.bfloat16, device=device)

    o_v4, lse_v4 = sparse_mla_fwd_v4(q, kv, topk_indices, attn_sink=None, kv_lora_rank=D_V)
    dq_v4, dkv_v4, d_sink_v4 = sparse_mla_bwd_v4(
        q, kv, o_v4, do, topk_indices, lse_v4, attn_sink=None, kv_lora_rank=D_V,
    )

    dq_v3, dkv_v3 = sparse_mla_bwd_v3(
        q, kv, o_v4, do, topk_indices, lse_v4, kv_lora_rank=D_V, method="chunked_gather",
    )

    print(f"  [B={B} S={S} H={H} D_V={D_V} TOPK={TOPK}]  V4(sink=None) bwd vs V3.2 chunked_gather")
    # V3.2 regression: V4(sink=None) compiles to same kernel as V3.2.
    _check("dQ ", dq_v4, dq_v3, abs_tol=1e-4, sig_threshold=1e-3,
           median_rel_tol=1e-4, cos_tol=1e-6)
    _check("dKV", dkv_v4, dkv_v3, abs_tol=1e-4, sig_threshold=1e-3,
           median_rel_tol=1e-4, cos_tol=1e-6)
    assert d_sink_v4 is None, "d_sink should be None when attn_sink is None"


def test_v4_with_sink(B, S, H, D_V, D_ROPE, TOPK, sink_init, device):
    """V4 bwd with non-trivial sink must match torch autograd of V4 fwd."""
    torch.manual_seed(0)
    D_QK = D_V + D_ROPE
    total = B * S
    q = torch.randn(total, H, D_QK, dtype=torch.bfloat16, device=device)
    kv = torch.randn(total, 1, D_QK, dtype=torch.bfloat16, device=device)
    topk_indices = torch.randint(0, total, (total, TOPK), dtype=torch.int32, device=device)
    do = torch.randn(total, H, D_V, dtype=torch.bfloat16, device=device)
    attn_sink = torch.full((H,), sink_init, dtype=torch.float32, device=device)

    scale = 1.0 / math.sqrt(D_QK)

    # Our V4 fwd + bwd
    o_v4, lse_v4 = sparse_mla_fwd_v4(q, kv, topk_indices, attn_sink=attn_sink, kv_lora_rank=D_V)
    dq_v4, dkv_v4, d_sink_v4 = sparse_mla_bwd_v4(
        q, kv, o_v4, do, topk_indices, lse_v4, attn_sink=attn_sink, kv_lora_rank=D_V,
    )

    # Torch autograd reference
    dq_ref, dkv_ref, d_sink_ref = ref_grads_v4(
        q, kv, topk_indices, attn_sink, do, D_V, scale,
    )
    # dkv_ref shape: [T, 1, D_QK]; ours: [T, 1, D_QK]
    dkv_ref = dkv_ref.view_as(dkv_v4)

    print(f"  [B={B} S={S} H={H} D_V={D_V} TOPK={TOPK}]  V4(sink={sink_init:+.1f}) bwd vs autograd")
    # bf16 accumulator + chunked reduction → looser tolerance
    # bf16 + chunked reduction noise; rely on median + cos_err
    _check("dQ   ", dq_v4, dq_ref, abs_tol=5e-2, sig_threshold=1e-2,
           median_rel_tol=2e-2, cos_tol=1e-3)
    _check("dKV  ", dkv_v4, dkv_ref, abs_tol=2e-1, sig_threshold=1e-2,
           median_rel_tol=2e-2, cos_tol=1e-3)
    _check("dSink", d_sink_v4, d_sink_ref, abs_tol=5e-1, sig_threshold=1e-2,
           median_rel_tol=5e-2, cos_tol=1e-3)


def test_swa_only_bwd(B, S, H, D_V, D_ROPE, device):
    torch.manual_seed(0)
    D_QK = D_V + D_ROPE
    total = B * S
    TOPK = 128
    q = torch.randn(total, H, D_QK, dtype=torch.bfloat16, device=device)
    kv = torch.randn(total, 1, D_QK, dtype=torch.bfloat16, device=device)
    do = torch.randn(total, H, D_V, dtype=torch.bfloat16, device=device)

    topk_indices = torch.full((total, TOPK), -1, dtype=torch.int32, device=device)
    for t in range(total):
        lo = max(0, t - TOPK + 1)
        n = t - lo + 1
        topk_indices[t, :n] = torch.arange(lo, t + 1, dtype=torch.int32, device=device)

    attn_sink = torch.full((H,), -1.0, dtype=torch.float32, device=device)
    scale = 1.0 / math.sqrt(D_QK)

    o_v4, lse_v4 = sparse_mla_fwd_v4(q, kv, topk_indices, attn_sink=attn_sink, kv_lora_rank=D_V)
    dq_v4, dkv_v4, d_sink_v4 = sparse_mla_bwd_v4(
        q, kv, o_v4, do, topk_indices, lse_v4, attn_sink=attn_sink, kv_lora_rank=D_V,
    )

    dq_ref, dkv_ref, d_sink_ref = ref_grads_v4(
        q, kv, topk_indices, attn_sink, do, D_V, scale,
    )
    dkv_ref = dkv_ref.view_as(dkv_v4)

    print(f"  [B={B} S={S} H={H} D_V={D_V} SWA-only TOPK=128]  V4 bwd vs autograd")
    # bf16 + chunked reduction noise; rely on median + cos_err
    _check("dQ   ", dq_v4, dq_ref, abs_tol=5e-2, sig_threshold=1e-2,
           median_rel_tol=2e-2, cos_tol=1e-3)
    _check("dKV  ", dkv_v4, dkv_ref, abs_tol=2e-1, sig_threshold=1e-2,
           median_rel_tol=2e-2, cos_tol=1e-3)
    _check("dSink", d_sink_v4, d_sink_ref, abs_tol=5e-1, sig_threshold=1e-2,
           median_rel_tol=5e-2, cos_tol=1e-3)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    device = "cuda"
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Arch:   {torch.cuda.get_device_properties(device).gcnArchName}")
    print()

    print("=" * 60)
    print("[1] V3.2 regression: V4 bwd(sink=None) == V3.2 chunked_gather bwd")
    print("=" * 60)
    test_v3_regression(B=1, S=128, H=16,  D_V=256, D_ROPE=64, TOPK=64,   device=device)
    test_v3_regression(B=1, S=256, H=32,  D_V=512, D_ROPE=64, TOPK=128,  device=device)
    test_v3_regression(B=1, S=512, H=128, D_V=512, D_ROPE=64, TOPK=640,  device=device)

    print()
    print("=" * 60)
    print("[2] V4 bwd with sink vs torch autograd")
    print("=" * 60)
    for sink_init in [-1e3, -1.0, 0.0, 1.0]:
        test_v4_with_sink(B=1, S=128, H=16,  D_V=256, D_ROPE=64, TOPK=128,
                          sink_init=sink_init, device=device)
        test_v4_with_sink(B=1, S=256, H=128, D_V=512, D_ROPE=64, TOPK=640,
                          sink_init=sink_init, device=device)

    print()
    print("=" * 60)
    print("[3] SWA-only layer bwd")
    print("=" * 60)
    test_swa_only_bwd(B=1, S=256, H=16,  D_V=256, D_ROPE=64, device=device)
    test_swa_only_bwd(B=1, S=512, H=128, D_V=512, D_ROPE=64, device=device)

    print()
    print("All V4 bwd tests PASSED.")


if __name__ == "__main__":
    main()
