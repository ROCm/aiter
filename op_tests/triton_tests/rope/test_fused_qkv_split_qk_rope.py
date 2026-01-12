import torch
import pytest
from aiter.ops.triton.fused_qkv_split_qk_rope import fused_qkv_split_qk_rope
from op_tests.triton_tests.fusions.test_fused_qk_concat import (
    generate_rope_cached_freqs,
)
from op_tests.test_rope import ref_rope_sbhd_fwd, RotateStyle


def generate_qkv_inputs(
    B: int, QH_PER_KH: int, KH: int, D: int, nope: bool, nope_first: bool, dtype
):
    qkv = torch.randn(
        (B, (QH_PER_KH * KH + 2 * KH) * (D * (2 if nope else 1))),
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
):
    q_size = QH_PER_KH * KH * D
    kv_size = KH * D
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    q = q.view(-1, QH_PER_KH * KH, D).contiguous()
    k = k.view(-1, KH, D).contiguous()
    v = v.view(-1, KH, D).contiguous()

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

    return q, k, v


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
    "nope, nope_first", [(False, False), (True, False), (True, True)]
)
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
    reuse_freqs_front_part: bool,
    dtype: torch.dtype,
):

    qkv = generate_qkv_inputs(B, QH_PER_KH, KH, D, nope, nope_first, dtype)

    pos, freqs, cos, sin = generate_rope_cached_freqs(
        B, max_embed_positions, (D // 2) if reuse_freqs_front_part else D, dtype
    )
    ref_freqs = freqs[pos].squeeze(-2)

    q_triton, k_triton, v_triton = fused_qkv_split_qk_rope(
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
    )
    q_torch, k_torch, v_torch = run_torch(
        qkv,
        QH_PER_KH,
        KH,
        (D * (2 if nope else 1)),
        ref_freqs,
        reuse_freqs_front_part,
        nope,
        nope_first,
        rotate_style,
    )

    torch.testing.assert_close(q_torch, q_triton)
    torch.testing.assert_close(k_torch, k_triton)
    torch.testing.assert_close(v_torch, v_triton)
