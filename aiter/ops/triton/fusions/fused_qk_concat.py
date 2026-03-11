import torch
from aiter.ops.triton._triton_kernels.fusions.fused_qk_concat import (
    _qk_rope_cat_kernel,
    _qk_cat_kernel,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def fused_qk_cat(
    q1: torch.Tensor,
    q2: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
):
    """
    Concat q1 with q2 and k1 with k2 along the last dimension

    Key parameters:
    - q1: Matrix X with shape (B, QH, D1).
    - q2: Matrix W with shape (B, QH, D2).
    - k1: Matrix X with shape (B, KH, D1).
    - k2: Matrix W with shape (B, KH, D2).

    QH must be multiple of KH

    Returns:
    - q_out: The output matrix with shape (B, QH, D1+D2).
    - k_out: The output matrix with shape (B, KH, D1+D2).
    """
    _LOGGER.info(
        f"FUSED_QK_CAT: q1={tuple(q1.shape)} q2={tuple(q2.shape)} k1={tuple(k1.shape)} k2={tuple(k2.shape)} "
    )
    b, qh, d1 = q1.shape
    b2, qh2, d2 = q2.shape
    bk, kh, dk1 = k1.shape
    bk2, kh2, dk2 = k2.shape
    assert (
        b == b2 == bk == bk2
    ), "q1 batch dimension should be identical across all inputs"
    assert qh == qh2, "Q head should be identical"
    assert kh == kh2, "K head should be identical"
    assert d1 == dk1, "D dimension of q1 and k1 should be identical"
    assert d2 == dk2, "D dimension of q2 and k2 should be identical"
    assert qh % kh == 0, "Number of Q heads must be multiple of number H heads"

    q_out = torch.empty((b, qh, d1 + d2), dtype=q1.dtype, device=q1.device)
    k_out = torch.empty((b, kh, d1 + d2), dtype=q1.dtype, device=q1.device)

    grid = (b, qh, 1)

    _qk_cat_kernel[grid](
        q1,
        q2,
        k1,
        k2,
        q_out,
        k_out,
        *q1.stride(),
        *q2.stride(),
        *k1.stride(),
        *k2.stride(),
        *q_out.stride(),
        *k_out.stride(),
        QH_PER_KH=qh // kh,
        BLOCK_D1=d1,
        BLOCK_D2=d2,
    )

    return q_out, k_out


def fused_qk_rope_cat(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    pos: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox: bool,
):
    """
    Perform RoPE on q_pe and k_pe and concat q_nope with q_pe and k_nope with k_pe along the last dimension

    Key parameters:
    - q_nope: Matrix X with shape (B, QH, D1).
    - q_pe: Matrix W with shape (B, QH, D2).
    - k_nope: Matrix X with shape (B, KH, D1).
    - k_pe: Matrix W with shape (B, KH, D2).

    QH must be multiple of KH

    Returns:
    - q_out: The output matrix with shape (B, QH, D1+D2).
    - k_out: The output matrix with shape (B, KH, D1+D2).
    """
    _LOGGER.info(
        f"FUSED_QK_ROPE_CAT: q_nope={tuple(q_nope.shape)} q_pe={tuple(q_pe.shape)} k_nope={tuple(k_nope.shape)} k_pe={tuple(k_pe.shape)} "
        + f"pos={tuple(pos.shape)} cos={tuple(cos.shape)} sin={tuple(sin.shape)}"
    )
    b, qh, d_nope = q_nope.shape
    b2, qh2, d_pe = q_pe.shape
    bk, kh, dk1 = k_nope.shape
    bk2, kh2, dk2 = k_pe.shape

    assert (
        b == b2 == bk == bk2
    ), "q1 batch dimension should be identical across all inputs"
    assert qh == qh2, "Q head should be identical"
    assert kh == kh2, "K head should be identical"
    assert d_nope == dk1, "D dimension of q_nope and k_nope should be identical"
    assert d_pe == dk2, "D dimension of q_pe and k_pe should be identical"
    assert qh % kh == 0, "Q heads must be multiple of H heads"
    d_freq = cos.shape[-1]
    assert (d_freq == d_pe // 2) or (
        d_freq == d_pe
    ), "cos/sin last dim should be the same or half of the qk last dim"
    reuse_freqs_front_part = d_freq == d_pe // 2

    q_out = torch.empty(
        (b, qh, d_nope + d_pe), dtype=q_nope.dtype, device=q_nope.device
    )
    k_out = torch.empty(
        (b, kh, d_nope + d_pe), dtype=q_nope.dtype, device=q_nope.device
    )

    grid = (b, qh, 1)

    _qk_rope_cat_kernel[grid](
        q_nope,
        q_pe,
        k_nope,
        k_pe,
        pos,
        cos,
        sin,
        q_out,
        k_out,
        *q_nope.stride(),
        *q_pe.stride(),
        *k_nope.stride(),
        *k_pe.stride(),
        pos.stride(0),
        cos.stride(0),
        cos.stride(-1),
        *q_out.stride(),
        *k_out.stride(),
        QH_PER_KH=qh // kh,
        REUSE_FREQS_FRONT_PART=reuse_freqs_front_part,
        IS_NEOX=is_neox,
        BLOCK_D_nope=d_nope,
        BLOCK_D_pe=d_pe,
        BLOCK_D_HALF_pe=d_pe // 2,
    )

    return q_out, k_out
