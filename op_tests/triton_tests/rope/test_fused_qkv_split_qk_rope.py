import torch
import pytest
from aiter.ops.triton.rope.fused_qkv_split_qk_rope import fused_qkv_split_qk_rope
from op_tests.triton_tests.fusions.test_fused_qk_concat import (
    generate_rope_cached_freqs,
)
from op_tests.test_rope import ref_rope_sbhd_fwd, RotateStyle


def generate_qkv_inputs(
    B: int, QH_PER_KH: int, KH: int, D: int, nope: bool, nope_first: bool,  attn_output_gate: bool, dtype
):
    qkv = torch.randn(
        (B, (QH_PER_KH * KH * (2 if attn_output_gate else 1) + 2 * KH) * (D * (2 if nope else 1))),
        dtype=dtype,
        device="cuda",
    )
    return qkv


def run_torch(
    qkv,
    QH_PER_KH,
    KH,
    D,
    ref_freqs,
    reuse_freqs_front_part,
    nope,
    nope_first,
    rotate_style,
    eps=1e-5,
    rms_norm_qk=False,
    attn_output_gate=False,
):
    q_size = QH_PER_KH * KH * D
    kv_size = KH * D
    if attn_output_gate:
        q, gate, k, v = qkv.split([q_size, q_size, kv_size, kv_size], dim=-1)
        gate = gate.view((-1, QH_PER_KH * KH, D))
    else:
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

    q = q.view(-1, QH_PER_KH * KH, D).contiguous()
    k = k.view(-1, KH, D).contiguous()
    v = v.view(-1, KH, D).contiguous()

    if rms_norm_qk:
        q = q.to(torch.float32) * torch.rsqrt(q.to(torch.float32).pow(2).mean(-1, keepdim=True) + eps)
        k = k.to(torch.float32) * torch.rsqrt(k.to(torch.float32).pow(2).mean(-1, keepdim=True) + eps)
        q = q.to(v.dtype)
        k = k.to(v.dtype)

    q = ref_rope_sbhd_fwd(
        q,
        ref_freqs,
        rotate_style=rotate_style,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=nope_first,
    )
    k = ref_rope_sbhd_fwd(
        k,
        ref_freqs,
        rotate_style=rotate_style,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=nope_first,
    )

    if attn_output_gate:
        return q, gate, k, v
    else:
        return q, k , v

# @pytest.mark.parametrize("B", [32])
# @pytest.mark.parametrize("QH_PER_KH", [8])
# @pytest.mark.parametrize("KH", [8])
# @pytest.mark.parametrize("D", [64])
@pytest.mark.parametrize("B", [1, 4, 8, 16, 32])
@pytest.mark.parametrize("QH_PER_KH", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("KH", [1, 4])
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("rotate_style", [RotateStyle.GPTJ, RotateStyle.NEOX])
@pytest.mark.parametrize("max_embed_positions", [131072])
@pytest.mark.parametrize(
    "nope, nope_first, rms_norm_qk", [(False, False, False), (False, False, True), (True, False, False), (True, True, False)]
)
@pytest.mark.parametrize("attn_output_gate", [False, True])
@pytest.mark.parametrize("reuse_freqs_front_part", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fused_qkv_split_qk_rope(
    B: int,
    QH_PER_KH: int,
    KH: int,
    D: int,
    rotate_style: int,
    max_embed_positions: int,
    nope: bool,
    nope_first: bool,
    rms_norm_qk: bool,
    attn_output_gate: bool,
    reuse_freqs_front_part: bool,
    dtype: torch.dtype,
):

    qkv = generate_qkv_inputs(B, QH_PER_KH, KH, D, nope, nope_first, attn_output_gate, dtype)

    pos, freqs, cos, sin = generate_rope_cached_freqs(
        B, max_embed_positions, (D // 2) if reuse_freqs_front_part else D, dtype
    )
    ref_freqs = freqs[pos].squeeze(-2)

    eps=0.00001

    triton_result = fused_qkv_split_qk_rope(
        qkv,
        cos,
        sin,
        pos,
        QH_PER_KH * KH,
        KH,
        (D * (2 if nope else 1)),
        is_neox=(rotate_style == RotateStyle.NEOX),
        offsets=None,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=nope_first,
        attn_output_gate=attn_output_gate,
        rms_norm_qk=rms_norm_qk,
        eps=eps
    )

    if attn_output_gate:
        q_triton, gate_triton, k_triton, v_triton = triton_result
    else:
        q_triton, k_triton, v_triton = triton_result

    torch_result = run_torch(
        qkv,
        QH_PER_KH,
        KH,
        (D * (2 if nope else 1)),
        ref_freqs,
        reuse_freqs_front_part,
        nope,
        nope_first,
        rotate_style,
        eps=eps,
        rms_norm_qk=rms_norm_qk,
        attn_output_gate=attn_output_gate
    )

    if attn_output_gate:
        q_torch, gate_torch, k_torch, v_torch = torch_result
    else:
        q_torch, k_torch, v_torch = torch_result

    if attn_output_gate:
        torch.testing.assert_close(gate_torch, gate_triton)

    torch.testing.assert_close(q_torch, q_triton, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(k_torch, k_triton, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(v_torch, v_triton)
