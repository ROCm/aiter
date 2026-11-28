# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Sparse MLA training backward with a wide Q/K and a narrow V (OPUS, gfx950).

Work in progress. This revision sets the interface and checks the inputs. The OPUS kernels come in
later commits. Until then, ``sparse_mla_bwd`` raises ``NotImplementedError`` after the checks.

The op is the absorbed MLA of DeepSeek-V3.2, GLM-5.x and Kimi-K3.1 DSA. All heads share one KV
row per token: ``kv = [latent | rope]``. K is the full row. V is the latent only. For GLM-5.3 and
Kimi-K3.1, Q/K are 576 wide and V/O are 512 wide. The sparse MLA backwards on ``main``
(``sparse_mla_bwd_dsv4``, ``sparse_mla_dsv4_train``) use one width for Q, KV, O and dO, so they
cannot run this op.

    S      = q @ kv^T * scale                                 D_qk = 576
    P      = exp(S - lse)
    O      = P @ kv[:, :D_v]                                  D_v  = 512
    delta  = rowsum(O * dO)
    dP     = dO @ kv[:, :D_v]^T
    dS     = P * (dP - delta) * scale
    dQ     = dS @ kv                                          [T, H, D_qk]
    dKV    = scatter_add over top-k of sum_h(dS*Q + [P*dO | 0])     [num_kv, D_qk]
    d_sink = -sum_t exp(sink - lse) * delta                   [H]

The forward is ``sparse_mla_fwd(..., return_lse=True)`` on a bf16 ``[num_kv, 576]`` pool. Its
LSE uses the natural log and includes the sink. This backward uses the same convention.
"""

import torch

from aiter.jit.utils.chip_info import get_gfx

_NAME = "sparse_mla_bwd"


def _check_inputs(
    q,
    kv,
    o,
    do,
    lse,
    topk_indices,
    kv_lora_rank,
    qk_rope_head_dim,
    attn_sink,
    dkv_out,
):
    """Check one call against the contract of ``sparse_mla_bwd``.

    Returns ``(T, H, D_qk, D_v, TOPK, num_kv)``.
    """
    if kv_lora_rank <= 0 or qk_rope_head_dim < 0:
        raise ValueError(
            f"{_NAME}: need kv_lora_rank > 0 and qk_rope_head_dim >= 0, got "
            f"{kv_lora_rank} / {qk_rope_head_dim}"
        )
    d_qk = kv_lora_rank + qk_rope_head_dim
    d_v = kv_lora_rank

    tensors = [
        ("q", q),
        ("kv", kv),
        ("o", o),
        ("do", do),
        ("lse", lse),
        ("topk_indices", topk_indices),
    ]
    if attn_sink is not None:
        tensors.append(("attn_sink", attn_sink))
    if dkv_out is not None:
        tensors.append(("dkv_out", dkv_out))
    for name, t in tensors:
        if not t.is_cuda:
            raise ValueError(f"{_NAME}: {name} must be on the GPU, got {t.device}")

    for name, t in (("q", q), ("kv", kv), ("o", o), ("do", do)):
        if t.dtype != torch.bfloat16:
            raise ValueError(f"{_NAME}: {name} must be bf16, got {t.dtype}")
    if lse.dtype != torch.float32:
        raise ValueError(f"{_NAME}: lse must be float32, got {lse.dtype}")
    if topk_indices.dtype != torch.int32:
        raise ValueError(
            f"{_NAME}: topk_indices must be int32, got {topk_indices.dtype}"
        )

    if q.ndim != 3:
        raise ValueError(f"{_NAME}: q must be [T, H, D_qk], got {tuple(q.shape)}")
    T, H, _ = q.shape
    if q.shape[2] != d_qk:
        raise ValueError(
            f"{_NAME}: q width {q.shape[2]} != kv_lora_rank + qk_rope_head_dim = {d_qk}"
        )
    if kv.ndim != 2 or kv.shape[1] != d_qk:
        raise ValueError(f"{_NAME}: kv must be [num_kv, {d_qk}], got {tuple(kv.shape)}")
    num_kv = kv.shape[0]
    for name, t in (("o", o), ("do", do)):
        if tuple(t.shape) != (T, H, d_v):
            raise ValueError(
                f"{_NAME}: {name} must be [{T}, {H}, {d_v}] (V is the latent only), got "
                f"{tuple(t.shape)}"
            )
    if tuple(lse.shape) != (T, H):
        raise ValueError(f"{_NAME}: lse must be [{T}, {H}], got {tuple(lse.shape)}")
    # The grid size comes from q. A shorter index tensor would be read past its end.
    if topk_indices.ndim != 2 or topk_indices.shape[0] != T:
        raise ValueError(
            f"{_NAME}: topk_indices must be [T={T}, TOPK], got {tuple(topk_indices.shape)}"
        )
    TOPK = topk_indices.shape[1]

    if attn_sink is not None and (
        tuple(attn_sink.shape) != (H,) or attn_sink.dtype != torch.float32
    ):
        raise ValueError(
            f"{_NAME}: attn_sink must be [{H}] float32, got {tuple(attn_sink.shape)} "
            f"{attn_sink.dtype}"
        )
    if dkv_out is not None and (
        tuple(dkv_out.shape) != (num_kv, d_qk)
        or dkv_out.dtype != torch.float32
        or not dkv_out.is_contiguous()
    ):
        raise ValueError(
            f"{_NAME}: dkv_out must be a contiguous [{num_kv}, {d_qk}] float32 tensor, got "
            f"{tuple(dkv_out.shape)} {dkv_out.dtype}"
        )
    return T, H, d_qk, d_v, TOPK, num_kv


def sparse_mla_bwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    lse: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    *,
    kv_lora_rank: int = 512,
    qk_rope_head_dim: int = 64,
    attn_sink: torch.Tensor | None = None,
    dkv_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Backward of sparse (top-k gathered) MLA attention where V is narrower than Q/K.

    The arguments after ``softmax_scale`` are keyword-only, as in ``sparse_mla_fwd``.

    Args:
        q: [T, H, kv_lora_rank + qk_rope_head_dim] bf16. The queries of the forward.
        kv: [num_kv, kv_lora_rank + qk_rope_head_dim] bf16. The latent | rope rows. All heads
            share them. ``topk_indices`` gives global row ids into ``kv``. With context
            parallelism, ``kv`` is the gathered (global) KV and ``q`` is the local slice.
        o: [T, H, kv_lora_rank] bf16. The output of the forward.
        do: [T, H, kv_lora_rank] bf16. The gradient of ``o``.
        lse: [T, H] float32. The natural-log LSE of the forward
            (``sparse_mla_fwd(..., return_lse=True)``). It includes the sink when
            ``attn_sink`` is set. A row with only -1 indices has -inf.
        topk_indices: [T, TOPK] int32. Global row ids into ``kv``. -1 is a masked slot. Each
            slot is a separate key, as in the forward.
        softmax_scale: The scale of the forward. It is required. For absorbed MLA it comes
            from the unabsorbed head width (for example ``1/sqrt(192 + 64)``, times a YaRN
            mscale if any), not from the 576-wide row.
        kv_lora_rank: The width of V, and of the first part of K. 512 for GLM-5.x and
            Kimi-K3.1.
        qk_rope_head_dim: The width of the rope part of K. 64. Use 0 for a rope-free model.
        attn_sink: Optional [H] float32 sink per head, as in the forward.
        dkv_out: Optional contiguous [num_kv, D_qk] float32 accumulator. If set, dKV is added
            to it. Use it to sum chunk or context-parallel parts in fp32 before one cast.

    Returns:
        ``(dq, dkv, d_sink)``. ``dq`` is [T, H, D_qk] bf16. ``dkv`` is [num_kv, D_qk] float32
        (``dkv_out`` if set). ``d_sink`` is [H] float32, or None without a sink.

    Limits:
        gfx950 only. bf16 only.
    """
    arch = get_gfx()
    if arch != "gfx950":
        raise RuntimeError(f"{_NAME} requires gfx950, got {arch}")
    _check_inputs(
        q,
        kv,
        o,
        do,
        lse,
        topk_indices,
        kv_lora_rank,
        qk_rope_head_dim,
        attn_sink,
        dkv_out,
    )
    raise NotImplementedError(
        f"{_NAME}: the OPUS kernels are not in this revision yet (draft)"
    )
