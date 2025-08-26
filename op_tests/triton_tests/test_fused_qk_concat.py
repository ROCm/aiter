import torch
import pytest
from aiter.ops.triton.fused_qk_concat import fused_qk_cat, fused_qk_rope_cat
from op_tests.test_rope import ref_rope_sbhd_fwd, RotateStyle


def generate_qk_inputs(B: int, QH_PER_KH: int, KH: int, D_nope: int, D_pe: int, dtype):
    q_nope = torch.randn((B, QH_PER_KH * KH, D_nope), dtype=dtype, device="cuda")
    q_pe = torch.randn((B, QH_PER_KH * KH, D_pe), dtype=dtype, device="cuda")
    k_nope = torch.randn((B, KH, D_nope), dtype=dtype, device="cuda")
    k_pe = torch.randn((B, KH, D_pe), dtype=dtype, device="cuda")

    return q_nope, q_pe, k_nope, k_pe


def generate_rope_cached_freqs(B: int, max_embed_positions: int, freqs_D: int, dtype):
    pos = torch.randint(0, max_embed_positions, (B,), device="cuda")
    # freqs = torch.randn((max_embed_positions, 1, 1, freqs_D), dtype=dtype, device="cuda")
    freqs = torch.randn(
        (max_embed_positions, 1, 1, freqs_D), dtype=dtype, device="cuda"
    )
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    cos_sin = torch.cat((cos, sin), dim=-1)
    cos, sin = torch.chunk(cos_sin, 2, dim=-1)
    return pos, freqs, cos, sin


def ref_qk_cat(q_nope, q_pe, k_nope, k_pe):
    return torch.cat((q_nope, q_pe), dim=-1), torch.cat((k_nope, k_pe), dim=-1)


def ref_qk_rope_cat(
    q_nope, q_pe, k_nope, k_pe, ref_freqs, reuse_freqs_front_part, rotate_style
):
    q_pe_out = ref_rope_sbhd_fwd(
        q_pe,
        ref_freqs,
        rotate_style=rotate_style,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=False,
    )
    k_pe_out = ref_rope_sbhd_fwd(
        k_pe,
        ref_freqs,
        rotate_style=rotate_style,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=False,
    )
    return torch.cat((q_nope, q_pe_out), dim=-1), torch.cat((k_nope, k_pe_out), dim=-1)


@pytest.mark.parametrize("B", [1, 4, 8, 16, 32])
@pytest.mark.parametrize("QH_PER_KH", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("KH", [1, 4])
@pytest.mark.parametrize("D_nope", [512])
@pytest.mark.parametrize("D_pe", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_qk_cat(B: int, QH_PER_KH: int, KH: int, D_nope: int, D_pe: int, dtype):

    q_nope, q_pe, k_nope, k_pe = generate_qk_inputs(
        B, QH_PER_KH, KH, D_nope, D_pe, dtype
    )

    q_torch, k_torch = ref_qk_cat(q_nope, q_pe, k_nope, k_pe)
    q_triton, k_triton = fused_qk_cat(q_nope, q_pe, k_nope, k_pe)

    torch.testing.assert_close(q_torch, q_triton)
    torch.testing.assert_close(k_torch, k_triton)


@pytest.mark.parametrize("B", [1, 4, 8, 16, 32])
@pytest.mark.parametrize("QH_PER_KH", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("KH", [1, 4])
@pytest.mark.parametrize("D_nope", [512])
@pytest.mark.parametrize("D_pe", [64, 128])
@pytest.mark.parametrize("max_embed_positions", [131072])
@pytest.mark.parametrize("reuse_freqs_front_part", [True, False])
@pytest.mark.parametrize("rotate_style", [RotateStyle.GPTJ, RotateStyle.NEOX])
# @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16]) # TODO fp16 results in ~0.6 error rate
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_qk_rope_cat(
    B: int,
    QH_PER_KH: int,
    KH: int,
    D_nope: int,
    D_pe: int,
    max_embed_positions: int,
    reuse_freqs_front_part: bool,
    rotate_style: RotateStyle,
    dtype,
):

    torch.cuda.empty_cache()  # Helps avoid hangs in large tests

    q_nope, q_pe, k_nope, k_pe = generate_qk_inputs(
        B, QH_PER_KH, KH, D_nope, D_pe, dtype
    )

    pos, freqs, cos, sin = generate_rope_cached_freqs(
        B, max_embed_positions, (D_pe // 2) if reuse_freqs_front_part else D_pe, dtype
    )
    ref_freqs = freqs[pos].squeeze(-2)

    q_torch, k_torch = ref_qk_rope_cat(
        q_nope, q_pe, k_nope, k_pe, ref_freqs, reuse_freqs_front_part, rotate_style
    )
    q_triton, k_triton = fused_qk_rope_cat(
        q_nope, q_pe, k_nope, k_pe, pos, cos, sin, (rotate_style == RotateStyle.NEOX)
    )

    torch.testing.assert_close(q_torch, q_triton)
    torch.testing.assert_close(k_torch, k_triton)
