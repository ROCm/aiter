"""
Test fused_qkv_split_qk_norm_rope_cache with Qwen3Next-80B-A3B parameters.

Qwen3Next uses partial Rotary Position Embeddings (RoPE):
  head_dim = 256, partial_rotary_factor = 0.25 => rotary_dim = 64

This means cos/sin have shape [..., rotary_dim // 2] = [..., 32], but
head_dim // 2 = 128.  The existing kernel sets BLOCK_D_HALF = head_dim // 2
and reads cos/sin at offsets 0..BLOCK_D_HALF-1, which is an out-of-bounds
read when cos only has 32 elements.  This causes either:
  - GPU memory access fault (for larger batch sizes), or
  - Silently incorrect rotation (garbage cos/sin values from adjacent memory)

The kernel needs a ROTARY_DIM_HALF parameter so it can:
  1. Load cos/sin only for the first ROTARY_DIM_HALF dimensions
  2. Pass through (identity) the remaining head_dim - rotary_dim dimensions

All existing tests have rotary_dim == head_dim, so this was never caught.

Config reference (Qwen3Next-80B-A3B-Instruct-FP8):
  num_attention_heads = 16, num_key_value_heads = 2  (GQA ratio 8:1)
  head_dim = 256
  partial_rotary_factor = 0.25  =>  rotary_dim = 64, cos_dim = 32
  attn_output_gate = True  (interleaved Q||gate in QKV projection)
  rope_theta = 1000000.0
  rope_type = "default"  (NeoX style)
  rms_norm_eps = 1e-6
"""

import torch
import pytest
from aiter.ops.triton.rope.fused_qkv_split_qk_norm_rope_cache import (
    fused_qkv_split_qk_norm_rope_cache,
)
from op_tests.triton_tests.fusions.test_fused_qk_concat import (
    generate_rope_cached_freqs,
)
from op_tests.test_rope import ref_rope_sbhd_fwd, RotateStyle


def rms_norm(x, w, eps):
    orig_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * (1.0 + w.float())
    return x.to(orig_dtype)


def run_torch_reference(
    qkv,
    q_weight,
    k_weight,
    QH_PER_KH,
    KH,
    D,
    attn_output_gate,
    ref_freqs,
    rotate_style,
    reuse_freqs_front_part,
    eps,
    slot_mapping,
    num_blocks,
    block_size,
    k_scale,
    v_scale,
):
    """PyTorch reference matching run_torch_with_cache (interleaved Q||gate when gated)."""
    q_size = QH_PER_KH * KH * D
    kv_size = KH * D

    QH = QH_PER_KH * KH
    if attn_output_gate:
        qg_len = 2 * q_size
        qg = qkv[:, :qg_len].reshape(-1, QH, 2, D)
        q = qg[:, :, 0, :].contiguous()
        gate = qg[:, :, 1, :].contiguous()
        k, v = qkv[:, qg_len:].split([kv_size, kv_size], dim=-1)
    else:
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    q = q.reshape(-1, QH, D).contiguous()
    k = k.view(-1, KH, D).contiguous()
    v = v.view(-1, KH, D).contiguous()

    q = rms_norm(q, q_weight, eps)
    k = rms_norm(k, k_weight, eps)

    q = ref_rope_sbhd_fwd(
        q,
        ref_freqs,
        rotate_style=rotate_style,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=False,
    )
    k = ref_rope_sbhd_fwd(
        k,
        ref_freqs,
        rotate_style=rotate_style,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=False,
    )

    if k_scale is None:
        k_scale = 1
    if v_scale is None:
        v_scale = 1

    k_descaled = k * (1 / k_scale)
    v_descaled = v * (1 / v_scale)

    k_cache = torch.zeros(
        (num_blocks, KH, block_size, D), dtype=qkv.dtype, device="cuda"
    )
    v_cache = torch.zeros(
        (num_blocks, KH, block_size, D), dtype=qkv.dtype, device="cuda"
    )

    for i in range(qkv.shape[0]):
        slot = slot_mapping[i].item()
        if slot >= 0:
            b = slot // block_size
            s = slot % block_size
            k_cache[b, :, s, :] = k_descaled[i]
            v_cache[b, :, s, :] = v_descaled[i]

    if attn_output_gate:
        return q, gate, k, v, k_cache, v_cache
    else:
        return q, k, v, k_cache, v_cache


@pytest.mark.parametrize("B", [1, 32])
@pytest.mark.parametrize("attn_output_gate", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_qwen3next_partial_rope(B, attn_output_gate, dtype):
    """
    Qwen3Next-80B-A3B:
      head_dim=256, rotary_dim=64 (e.g. partial_rotary_factor 0.25 in config)
      num_attention_heads=16, num_key_value_heads=2
      attn_output_gate=True, is_neox=True, reuse_freqs_front_part=True
      rms_norm_eps=1e-6

    Critical: cos has rotary_dim//2=32 elements, but head_dim=256.
    The existing kernel sets BLOCK_D_HALF=128 and reads cos at offsets 0..127.
    """
    D = 256
    rotary_dim = 64
    QH_PER_KH = 8
    KH = 2
    QH = QH_PER_KH * KH
    block_size = 16
    eps = 1e-6
    max_embed_positions = 131072

    torch.manual_seed(42)

    q_size = QH * D
    kv_size = KH * D
    if attn_output_gate:
        q_part = torch.randn((B, QH, D), dtype=dtype, device="cuda")
        gate_part = torch.randn((B, QH, D), dtype=dtype, device="cuda")
        qg = torch.stack([q_part, gate_part], dim=2).reshape(B, 2 * q_size)
        k_part = torch.randn((B, KH, D), dtype=dtype, device="cuda").reshape(B, kv_size)
        v_part = torch.randn((B, KH, D), dtype=dtype, device="cuda").reshape(B, kv_size)
        qkv = torch.cat([qg, k_part, v_part], dim=-1)
    else:
        qkv_dim = q_size + 2 * kv_size
        qkv = torch.randn((B, qkv_dim), dtype=dtype, device="cuda")
    q_weight = torch.randn((D,), dtype=dtype, device="cuda")
    k_weight = torch.randn((D,), dtype=dtype, device="cuda")

    # cos/sin with rotary_dim//2 = 32 elements
    freqs_dim = rotary_dim // 2  # 32
    pos, freqs, cos, sin = generate_rope_cached_freqs(
        B, max_embed_positions, freqs_dim, dtype
    )
    ref_freqs = freqs[pos].squeeze(-2)

    num_blocks = (B + block_size - 1) // block_size + 2
    k_cache = torch.zeros((num_blocks, KH, block_size, D), dtype=dtype, device="cuda")
    v_cache = torch.zeros((num_blocks, KH, block_size, D), dtype=dtype, device="cuda")
    slot_mapping = torch.randperm(num_blocks * block_size)[:B].to(torch.int32).cuda()

    # Pass full cos/sin table — kernel indexes by position internally
    tri_result = fused_qkv_split_qk_norm_rope_cache(
        qkv,
        q_weight,
        k_weight,
        cos,
        sin,
        pos,
        k_cache,
        v_cache,
        slot_mapping,
        QH,
        KH,
        D,
        is_neox=True,
        offsets=None,
        reuse_freqs_front_part=True,
        eps=eps,
        attn_output_gate=attn_output_gate,
    )

    if attn_output_gate:
        q_tri, gate_tri, k_tri, v_tri = tri_result
    else:
        q_tri, k_tri, v_tri = tri_result

    ref_result = run_torch_reference(
        qkv,
        q_weight,
        k_weight,
        QH_PER_KH,
        KH,
        D,
        attn_output_gate,
        ref_freqs,
        RotateStyle.NEOX,
        True,
        eps,
        slot_mapping,
        num_blocks,
        block_size,
        None,
        None,
    )

    # BF16 fused kernel vs float ref: allow ~2 ULP-scale error on occasional elements
    atol, rtol = 2e-2, 2e-2

    if attn_output_gate:
        q_ref, gate_ref, k_ref, v_ref, k_cache_ref, v_cache_ref = ref_result
        torch.testing.assert_close(gate_tri, gate_ref, atol=atol, rtol=rtol)
    else:
        q_ref, k_ref, v_ref, k_cache_ref, v_cache_ref = ref_result

    torch.testing.assert_close(q_tri, q_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(k_tri, k_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(v_tri, v_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(k_cache, k_cache_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(v_cache, v_cache_ref, atol=atol, rtol=rtol)
