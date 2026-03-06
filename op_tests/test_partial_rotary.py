"""Test partial rotary embedding support in fused_qk_norm_rope_cache_pts_quant_shuffle."""
import torch
from torch import Tensor
import aiter
from aiter.test_common import checkAllclose


def rms_norm_forward(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x.to(weight.dtype)


def apply_rotary_emb_neox(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat((o1, o2), dim=-1)


def apply_rotary_emb_interleaved(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.stack((o1, o2), dim=-1).flatten(-2)


def test_partial_rotary(
    dtype, num_tokens, num_heads_q, num_heads_k, num_heads_v,
    head_size, rotary_dim, is_neox_style, eps=1e-6,
):
    max_pos = 4096
    num_slots = num_tokens + 64

    qkv = torch.randn(
        (num_tokens, (num_heads_q + num_heads_k + num_heads_v) * head_size),
        dtype=dtype, device="cuda",
    )
    qw = torch.randn(head_size, dtype=dtype, device="cuda")
    kw = torch.randn(head_size, dtype=dtype, device="cuda")
    cos_sin = torch.randn((max_pos, rotary_dim), dtype=dtype, device="cuda")
    positions = torch.randint(0, max_pos, (num_tokens,), dtype=torch.int64, device="cuda")
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device="cuda")
    k_cache = torch.zeros((num_slots, num_heads_k, head_size), dtype=dtype, device="cuda")
    v_cache = torch.zeros((num_slots, num_heads_v, head_size), dtype=dtype, device="cuda")
    per_tensor_k_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    per_tensor_v_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    q_out = torch.empty((num_tokens, num_heads_q, head_size), dtype=dtype, device="cuda")
    k_out = torch.empty((num_tokens, num_heads_k, head_size), dtype=dtype, device="cuda")
    v_out = torch.empty((num_tokens, num_heads_v, head_size), dtype=dtype, device="cuda")

    # Reference
    q_size = num_heads_q * head_size
    k_size = num_heads_k * head_size
    v_size = num_heads_v * head_size
    qr, kr, vr = qkv.split([q_size, k_size, v_size], dim=-1)
    qr = rms_norm_forward(qr.view(num_tokens, num_heads_q, head_size), qw, eps)
    kr = rms_norm_forward(kr.view(num_tokens, num_heads_k, head_size), kw, eps)
    vr = vr.view(num_tokens, num_heads_v, head_size)

    indexed = cos_sin[positions]
    cos, sin = indexed.chunk(2, dim=-1)

    apply_fn = apply_rotary_emb_neox if is_neox_style else apply_rotary_emb_interleaved
    q_rot = apply_fn(qr[..., :rotary_dim], cos, sin)
    q_ref = torch.cat([q_rot, qr[..., rotary_dim:]], dim=-1)
    k_rot = apply_fn(kr[..., :rotary_dim], cos, sin)
    k_ref = torch.cat([k_rot, kr[..., rotary_dim:]], dim=-1)
    v_ref = vr

    # Fused kernel
    aiter.fused_qk_norm_rope_cache_pts_quant_shuffle(
        qkv.clone(), qw, kw, cos_sin, positions,
        num_tokens, num_heads_q, num_heads_k, num_heads_v, head_size,
        is_neox_style, eps,
        q_out, k_cache, v_cache, slot_mapping,
        per_tensor_k_scale, per_tensor_v_scale,
        k_out, v_out, True,
        False, 0, 0,
        rotary_dim,
    )

    tag = (
        f"dtype={dtype}, tokens={num_tokens}, Hq={num_heads_q}, Hk={num_heads_k}, "
        f"D={head_size}, rotary_dim={rotary_dim}, neox={is_neox_style}"
    )
    q_diff = (q_ref.reshape(num_tokens, -1) - q_out.reshape(num_tokens, -1)).abs().max().item()
    k_diff = (k_ref.reshape(num_tokens, -1) - k_out.reshape(num_tokens, -1)).abs().max().item()
    v_diff = (v_ref.reshape(num_tokens, -1) - v_out.reshape(num_tokens, -1)).abs().max().item()
    print(f"  q_diff={q_diff:.6f}, k_diff={k_diff:.6f}, v_diff={v_diff:.6f}")

    checkAllclose(q_ref.reshape(num_tokens, -1), q_out.reshape(num_tokens, -1),
                  msg=f"q  {tag}", rtol=1e-2, atol=0.05)
    checkAllclose(k_ref.reshape(num_tokens, -1), k_out.reshape(num_tokens, -1),
                  msg=f"k  {tag}", rtol=1e-2, atol=0.05)
    checkAllclose(v_ref.reshape(num_tokens, -1), v_out.reshape(num_tokens, -1),
                  msg=f"v  {tag}", rtol=1e-2, atol=0.05)
    print(f"[PASS] {tag}")


def test_full_rotary(
    dtype, num_tokens, num_heads_q, num_heads_k, num_heads_v,
    head_size, is_neox_style, eps=1e-6,
):
    """Verify full rotary (rotary_dim == head_size) still works with the new param."""
    test_partial_rotary(
        dtype, num_tokens, num_heads_q, num_heads_k, num_heads_v,
        head_size, head_size, is_neox_style, eps,
    )


if __name__ == "__main__":
    print("=== Full Rotary (regression) ===")
    for hs in [128, 256]:
        test_full_rotary(torch.bfloat16, 32, 8, 2, 2, hs, True)
        test_full_rotary(torch.bfloat16, 32, 8, 2, 2, hs, False)

    print("\n=== Partial Rotary (Qwen3.5-style) ===")
    for nt in [3, 32, 127, 512]:
        for (hq, hk) in [(32, 4), (8, 2)]:
            for (hs, rd) in [(256, 64), (128, 32)]:
                for neox in [True]:
                    test_partial_rotary(
                        torch.bfloat16, nt, hq, hk, hk,
                        hs, rd, neox,
                    )

    print("\n=== ALL TESTS PASSED ===")
