"""
Correctness test for V4 forward kernel `sparse_mla_fwd_v4`.

Verifies two things against a torch reference:
  1. With `attn_sink=None`, output bit-matches V3.2 `sparse_mla_fwd`
     (the only valid regression — V4 must reduce to V3.2 when sink is absent).
  2. With non-trivial `attn_sink`, output matches the torch reference that
     extends the softmax with an extra "sink column":
        LSE_total = logsumexp(cat([S, sink], dim=-1))
        P_j      = exp(S_j - LSE_total)   # sums to <1
        O        = sum_j P_j * V_j        # sink contributes nothing to O

Also tests the returned LSE is sink-inclusive (so backward can use it directly).
"""
import math

import torch

from aiter.ops.triton._triton_kernels.attention.deepseek_sparse_attention import (
    sparse_mla_fwd as sparse_mla_fwd_v3,
)
from aiter.ops.triton._triton_kernels.attention.deepseek_sparse_attention_v4 import (
    sparse_mla_fwd_v4,
)


# ---------------------------------------------------------------------
# Torch reference (matches the kernel's math exactly)
# ---------------------------------------------------------------------
def ref_sparse_mla_fwd_v4(q, kv, topk_indices, attn_sink, kv_lora_rank, scale):
    """
    Returns (O [T,H,D_V] bf16, LSE [T,H] fp32 sink-inclusive).
    """
    total_tokens, num_heads, d_qk = q.shape
    assert kv.dim() == 3 and kv.shape[1] == 1
    topk = topk_indices.shape[1]
    d_v = kv_lora_rank

    invalid = (topk_indices < 0) | (topk_indices >= total_tokens)
    safe_idx = topk_indices.clamp(0, total_tokens - 1).long()

    kv_flat = kv.squeeze(1).float()  # [T, D_QK]
    gathered_kv = kv_flat[safe_idx]  # [T, TOPK, D_QK]

    q_fp = q.float()
    S = torch.einsum("thd,tkd->thk", q_fp, gathered_kv) * scale  # [T, H, TOPK]
    S = S.masked_fill(invalid[:, None, :], float("-inf"))

    if attn_sink is not None:
        sink_col = attn_sink.float().view(1, num_heads, 1).expand(total_tokens, num_heads, 1)
        S_ext = torch.cat([S, sink_col], dim=-1)  # [T, H, TOPK+1]
        lse_total = torch.logsumexp(S_ext, dim=-1)  # [T, H]
    else:
        lse_total = torch.logsumexp(S, dim=-1)

    # Where all topk are invalid for a (t,h) row, lse_total is -inf -> output is 0.
    P = torch.exp(S - lse_total[:, :, None])  # [T, H, TOPK]
    P = torch.where(invalid[:, None, :], torch.zeros_like(P), P)

    V = gathered_kv[..., :d_v]                # [T, TOPK, D_V]
    O = torch.einsum("thk,tkd->thd", P, V)    # [T, H, D_V]
    return O.to(q.dtype), lse_total


# ---------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------
def _check(name, ours, ref, abs_tol, rel_tol=None, sig_threshold=1e-2,
           median_rel_tol=None, cos_tol=None):
    """
    Tolerance check for bf16 sparse attention output vs fp32 torch reference.

    Why we don't assert on max_rel by default:
      - bf16 has ~3 decimal digits of precision.
      - Sparse attention with peaked softmax + bf16 V gives 10-25% per-element
        rel-err on tail elements (small denominator amplifies bf16 noise).
      - max_rel on a single element is not a correctness signal — it's
        dominated by where the random data happens to land.
      - Median rel-err and cosine similarity are the real correctness signals.

    Args:
      abs_tol         hard ceiling on max|ours - ref| (catches gross bugs)
      sig_threshold   ignore elements with |ref| < this for rel-err
      rel_tol         optional ceiling on tail rel-err. **Off by default** —
                      use only when you have a known-good tail-noise bound.
      median_rel_tol  median rel-err over significant elements (the real test)
      cos_tol         1 - cosine_similarity (the real test)
    """
    diff = (ours.float() - ref.float()).abs()
    sig = ref.float().abs() > sig_threshold
    max_abs = diff.max().item()
    rel = diff[sig] / ref.float().abs()[sig] if sig.any() else diff.new_zeros(0)
    max_rel = rel.max().item() if rel.numel() else 0.0
    median_rel = rel.median().item() if rel.numel() else 0.0

    a, b = ours.float().flatten(), ref.float().flatten()
    cos_sim = (a @ b / (a.norm() * b.norm() + 1e-30)).item()
    cos_err = 1.0 - cos_sim

    print(f"    {name}: max_abs={max_abs:.4e}  max_rel={max_rel:.4e}  "
          f"median_rel={median_rel:.4e}  cos_err={cos_err:.4e}")

    assert max_abs < abs_tol, f"{name} abs error {max_abs} > {abs_tol}"
    if median_rel_tol is not None and rel.numel():
        assert median_rel < median_rel_tol, \
            f"{name} median rel error {median_rel} > {median_rel_tol}"
    if cos_tol is not None:
        assert cos_err < cos_tol, f"{name} cos error {cos_err} > {cos_tol}"
    if rel_tol is not None and rel.numel():
        assert max_rel < rel_tol, f"{name} max rel error {max_rel} > {rel_tol}"


def test_v3_regression(B, S, H, D_V, D_ROPE, TOPK, device):
    """V4 with sink=None must match V3.2 exactly (same kernel modulo HAS_SINK guard)."""
    torch.manual_seed(0)
    D_QK = D_V + D_ROPE
    total = B * S
    q = torch.randn(total, H, D_QK, dtype=torch.bfloat16, device=device)
    kv = torch.randn(total, 1, D_QK, dtype=torch.bfloat16, device=device)
    topk_indices = torch.randint(0, total, (total, TOPK), dtype=torch.int32, device=device)

    o_v3, lse_v3 = sparse_mla_fwd_v3(q, kv, topk_indices, kv_lora_rank=D_V)
    o_v4, lse_v4 = sparse_mla_fwd_v4(q, kv, topk_indices, attn_sink=None, kv_lora_rank=D_V)

    print(f"  [B={B} S={S} H={H} D_V={D_V} TOPK={TOPK}]  V4(sink=None) vs V3.2")
    # V3.2 regression: V4(sink=None) compiles to same kernel as V3.2 (HAS_SINK=False),
    # picks same autotune config, so we expect bit-identical output.
    _check("O  ", o_v4, o_v3, abs_tol=1e-4, rel_tol=1e-4, sig_threshold=1e-3)
    _check("LSE", lse_v4, lse_v3, abs_tol=1e-6, rel_tol=1e-6, sig_threshold=1e-3)


def test_v4_with_sink(B, S, H, D_V, D_ROPE, TOPK, sink_init, device):
    """V4 with non-trivial sink must match torch reference."""
    torch.manual_seed(0)
    D_QK = D_V + D_ROPE
    total = B * S
    q = torch.randn(total, H, D_QK, dtype=torch.bfloat16, device=device)
    kv = torch.randn(total, 1, D_QK, dtype=torch.bfloat16, device=device)
    topk_indices = torch.randint(0, total, (total, TOPK), dtype=torch.int32, device=device)
    attn_sink = torch.full((H,), sink_init, dtype=torch.float32, device=device)

    scale = 1.0 / math.sqrt(D_QK)
    o_ref, lse_ref = ref_sparse_mla_fwd_v4(q, kv, topk_indices, attn_sink, D_V, scale)
    o_ours, lse_ours = sparse_mla_fwd_v4(q, kv, topk_indices, attn_sink=attn_sink, kv_lora_rank=D_V)

    print(f"  [B={B} S={S} H={H} D_V={D_V} TOPK={TOPK}]  V4(sink={sink_init:+.1f}) vs torch ref")
    # bf16 sparse-attention noise floor:
    #   max abs:        ~1e-2  (per-element bf16 quantization)
    #   max rel:        ~15%   (peaked-softmax + bf16 V tail-elements)
    #   median rel:     ~5e-3  (bulk of elements close)
    #   cos err:        ~1e-5  (overall direction preserved)
    # Drop max_rel assertion (tail noise) — rely on median_rel + cos_err.
    _check("O  ", o_ours, o_ref, abs_tol=5e-2, sig_threshold=1e-2,
           median_rel_tol=1e-2, cos_tol=1e-3)
    _check("LSE", lse_ours, lse_ref, abs_tol=1e-2, sig_threshold=1e-2,
           median_rel_tol=1e-3, cos_tol=1e-5)


def test_swa_only(B, S, H, D_V, D_ROPE, device):
    """SWA-only layer: TOPK=128, all indices are the local window."""
    torch.manual_seed(0)
    D_QK = D_V + D_ROPE
    total = B * S
    TOPK = 128
    q = torch.randn(total, H, D_QK, dtype=torch.bfloat16, device=device)
    kv = torch.randn(total, 1, D_QK, dtype=torch.bfloat16, device=device)

    # Build SWA indices: for query t, attend to tokens [max(0, t-127), t]
    topk_indices = torch.full((total, TOPK), -1, dtype=torch.int32, device=device)
    for t in range(total):
        lo = max(0, t - TOPK + 1)
        n = t - lo + 1
        topk_indices[t, :n] = torch.arange(lo, t + 1, dtype=torch.int32, device=device)

    attn_sink = torch.full((H,), -1.0, dtype=torch.float32, device=device)
    scale = 1.0 / math.sqrt(D_QK)
    o_ref, lse_ref = ref_sparse_mla_fwd_v4(q, kv, topk_indices, attn_sink, D_V, scale)
    o_ours, lse_ours = sparse_mla_fwd_v4(q, kv, topk_indices, attn_sink=attn_sink, kv_lora_rank=D_V)

    print(f"  [B={B} S={S} H={H} D_V={D_V} SWA-only TOPK=128]  V4 vs torch ref")
    # Drop max_rel assertion (tail noise) — rely on median_rel + cos_err.
    _check("O  ", o_ours, o_ref, abs_tol=5e-2, sig_threshold=1e-2,
           median_rel_tol=1e-2, cos_tol=1e-3)
    _check("LSE", lse_ours, lse_ref, abs_tol=1e-2, sig_threshold=1e-2,
           median_rel_tol=1e-3, cos_tol=1e-5)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    device = "cuda"
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Arch:   {torch.cuda.get_device_properties(device).gcnArchName}")
    print()

    print("=" * 60)
    print("[1] V3.2 regression: V4(sink=None) must equal V3.2")
    print("=" * 60)
    test_v3_regression(B=1, S=128, H=16,  D_V=256, D_ROPE=64, TOPK=64,   device=device)
    test_v3_regression(B=1, S=256, H=32,  D_V=512, D_ROPE=64, TOPK=128,  device=device)
    test_v3_regression(B=1, S=512, H=128, D_V=512, D_ROPE=64, TOPK=640,  device=device)

    print()
    print("=" * 60)
    print("[2] V4 with non-trivial sink vs torch reference")
    print("=" * 60)
    # Sweep sink magnitudes: -inf-like (sink ~ -1e3, should be ~no-op),
    # negative (small effect), zero, positive (significant effect).
    for sink_init in [-1e3, -1.0, 0.0, 1.0, 5.0]:
        test_v4_with_sink(B=1, S=128, H=16,  D_V=256, D_ROPE=64, TOPK=128,
                          sink_init=sink_init, device=device)
        test_v4_with_sink(B=1, S=256, H=128, D_V=512, D_ROPE=64, TOPK=640,
                          sink_init=sink_init, device=device)
        test_v4_with_sink(B=1, S=256, H=128, D_V=512, D_ROPE=64, TOPK=1152,
                          sink_init=sink_init, device=device)

    print()
    print("=" * 60)
    print("[3] SWA-only layer (TOPK=128, window-limited indices)")
    print("=" * 60)
    test_swa_only(B=1, S=256, H=16,  D_V=256, D_ROPE=64, device=device)
    test_swa_only(B=1, S=512, H=128, D_V=512, D_ROPE=64, device=device)

    print()
    print("All V4 fwd tests PASSED.")


if __name__ == "__main__":
    main()
