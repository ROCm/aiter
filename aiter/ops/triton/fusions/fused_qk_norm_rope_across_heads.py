# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.fusions.fused_qk_norm_rope_across_heads import (
    _fused_qk_norm_rope_across_heads_kernel,
    _get_config,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

__all__ = ["fused_qk_norm_rope_across_heads"]


def _rows_view(t: torch.Tensor, name: str) -> torch.Tensor:
    """(B, S, N) view with a dense last dim; a 2-D (S, N) input gets B = 1."""
    if t.dim() == 2:
        t = t.unsqueeze(0)
    assert t.dim() == 3, f"{name} must be (B, S, N) or (S, N), got {tuple(t.shape)}"
    assert t.stride(-1) == 1, f"{name} must have a contiguous last dim"
    return t


def fused_qk_norm_rope_across_heads(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    head_dim: int,
    cos: torch.Tensor | None = None,
    sin: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    RMS norm of q and k over all heads of a token, then RoPE on each head, in
    one kernel.

    For each token row of N = heads * head_dim values, computes
    ``F.rms_norm(row, (N,), weight, eps)``, then, if ``cos`` and ``sin`` are
    given, rotates the pairs (2i, 2i + 1) of each head:
    ``(a * cos[2i] - b * sin[2i + 1], a * sin[2i + 1] + b * cos[2i])``. This is
    ``norm_q`` / ``norm_k`` with ``qk_norm="rms_norm_across_heads"`` and the
    ``apply_rotary_emb`` of the diffusers Wan transformer.

    Args:
        q (torch.Tensor): (B, S_q, N) or (S_q, N), fp32, fp16 or bf16. The rows
            can have a stride (for example a chunk of a fused QKV output); the
            last dim must be contiguous.
        k (torch.Tensor): (B, S_k, N) or (S_k, N), same dtype and N as q.
        q_weight (torch.Tensor): RMS norm weight of q, N elements.
        k_weight (torch.Tensor): RMS norm weight of k, N elements.
        eps (float): RMS norm epsilon.
        head_dim (int): Elements per head. N must be a multiple of it.
        cos (Optional[torch.Tensor]): RoPE table with S_q * head_dim elements,
            one row of head_dim per position, for example (1, S_q, 1, head_dim).
            Must be contiguous. Without cos and sin, only the norm runs.
        sin (Optional[torch.Tensor]): Same shape and dtype as cos.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Contiguous q and k outputs with the
        shapes and dtype of q and k. View them as (B, S, heads, head_dim).

    Limits: with RoPE, S_k must equal S_q and the batch shares one table. The
    norm is in fp32 and rounds once to the input dtype before the RoPE, as in
    eager PyTorch; the RoPE is in fp32 and rounds once.
    """
    q3 = _rows_view(q, "q")
    k3 = _rows_view(k, "k")
    N = q3.shape[-1]
    assert (
        k3.shape[-1] == N
    ), f"q and k must have the same N, got {N} and {k3.shape[-1]}"
    assert q.dtype == k.dtype, f"dtype mismatch: {q.dtype} vs {k.dtype}"
    assert q.dtype in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ), f"unsupported dtype: {q.dtype}"
    assert k.device == q.device, "q and k must be on the same device"
    for w, name in ((q_weight, "q_weight"), (k_weight, "k_weight")):
        assert w.numel() == N, f"{name} must have {N} elements, got {w.numel()}"
        assert w.device == q.device, f"{name} must be on the device of q"
    q_weight = q_weight.contiguous()
    k_weight = k_weight.contiguous()
    assert (
        N % head_dim == 0 and head_dim % 2 == 0
    ), f"N={N} must be a multiple of an even head_dim, got head_dim={head_dim}"

    use_rope = cos is not None
    assert use_rope == (sin is not None), "pass both cos and sin, or neither"
    S_q, S_k = q3.shape[1], k3.shape[1]
    if use_rope:
        assert S_k == S_q, f"RoPE needs S_k == S_q, got {S_k} and {S_q}"
        assert cos.shape == sin.shape and cos.dtype == sin.dtype, "cos and sin differ"
        assert (
            cos.is_contiguous() and sin.is_contiguous()
        ), "cos and sin must be contiguous"
        assert (
            cos.numel() == S_q * head_dim
        ), f"cos must have S_q * head_dim = {S_q * head_dim} elements, got {cos.numel()}"
        assert cos.device == q.device, "cos and sin must be on the device of q"

    _LOGGER.info(
        "FUSED_QK_NORM_ROPE_ACROSS_HEADS: q=%s k=%s head_dim=%d rope=%s",
        tuple(q.shape),
        tuple(k.shape),
        head_dim,
        use_rope,
    )

    q_out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    k_out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    M_q, M_k = q3.shape[0] * S_q, k3.shape[0] * S_k
    if max(M_q, M_k) == 0:
        return q_out, k_out

    config = _get_config(N)
    BLOCK_M = config.pop("BLOCK_M")
    # The row is done in chunks of BLOCK_N; a chunk never needs to exceed the row.
    BLOCK_N = min(config.pop("BLOCK_N"), triton.next_power_of_2(N))
    num_blocks_q = triton.cdiv(M_q, BLOCK_M)
    grid = (num_blocks_q + triton.cdiv(M_k, BLOCK_M),)
    _fused_qk_norm_rope_across_heads_kernel[grid](
        q3,
        k3,
        q_weight,
        k_weight,
        cos if use_rope else q_weight,
        sin if use_rope else q_weight,
        q_out,
        k_out,
        max(S_q, 1),
        max(S_k, 1),
        M_q,
        M_k,
        num_blocks_q,
        q3.stride(0),
        q3.stride(1),
        k3.stride(0),
        k3.stride(1),
        head_dim,
        eps,
        N=N,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        USE_ROPE=use_rope,
        **config,
    )
    return q_out, k_out
