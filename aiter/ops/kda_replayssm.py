# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ReplaySSM decode / speculative-verify step of KDA (Kimi Delta Attention), HIP.

Drop-in for the Triton ``replayssm_sigmoid_gating_delta_rule`` of ATOM's ``fla_ops/replayssm.py`` (same arguments,
same in-place updates of the checkpoint and the (k, u, g) record buffers) for the shapes it is specialised to:
K = V = 128, a verify window of at most 8 tokens and a record buffer of at most 16 rows, bf16 activations, an fp32
checkpoint, bf16 k/u records and fp32 or bf16 g records.  Use ``kda_replayssm_decode_supported`` to route.
"""

import torch

from ..jit.core import compile_ops

KDA_REPLAYSSM_HEAD_DIM = 128
KDA_REPLAYSSM_MAX_QUERY_LEN = 8
KDA_REPLAYSSM_MAX_CACHE_LEN = 16


@compile_ops(
    "module_kda_replayssm_decode", fc_name="kda_replayssm_decode", develop=True
)
def _kda_replayssm_decode(
    o: torch.Tensor,
    ckpt: torch.Tensor,
    buf_k: torch.Tensor,
    buf_u: torch.Tensor,
    buf_g: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    write_pos: torch.Tensor,
    slot_idx: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_query_len: int,
    scale: float,
    use_qk_l2norm: bool,
    use_lower_bound: bool,
    lower_bound: float,
) -> None:
    """KDA ReplaySSM decode step (in place on ckpt and the record buffers; o is written)."""


def _aligned(t: torch.Tensor) -> bool:
    return t.data_ptr() % 16 == 0


def kda_replayssm_decode_supported(
    q: torch.Tensor,
    v: torch.Tensor,
    ckpt: torch.Tensor,
    buf_k: torch.Tensor,
    buf_u: torch.Tensor,
    buf_g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    max_query_len: int,
) -> bool:
    """Whether ``kda_replayssm_decode`` handles this call (else use the Triton kernel)."""
    K, V = q.shape[-1], v.shape[-1]
    H, HV = q.shape[-2], v.shape[-2]
    cap = buf_k.shape[2]
    return (
        K == KDA_REPLAYSSM_HEAD_DIM
        and V == KDA_REPLAYSSM_HEAD_DIM
        and H > 0
        and HV % H == 0
        and 1 <= max_query_len <= KDA_REPLAYSSM_MAX_QUERY_LEN
        and 2 * max_query_len <= cap <= KDA_REPLAYSSM_MAX_CACHE_LEN
        and q.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
        and ckpt.dtype == torch.float32
        and A_log.dtype == torch.float32
        and dt_bias.dtype == torch.float32
        and buf_k.dtype == torch.bfloat16
        and buf_u.dtype == torch.bfloat16
        and buf_g.dtype in (torch.float32, torch.bfloat16)
        and ckpt.stride(-1) == 1
        and ckpt.stride(-2) == K
        and ckpt.stride(-3) == V * K
        and all(
            t.stride(-1) == 1 and t.stride(-2) % 8 == 0 for t in (buf_k, buf_u, buf_g)
        )
        and all(_aligned(t) for t in (ckpt, buf_k, buf_u, buf_g, dt_bias))
    )


def kda_replayssm_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ckpt: torch.Tensor,
    buf_k: torch.Tensor,
    buf_u: torch.Tensor,
    buf_g: torch.Tensor,
    write_pos: torch.Tensor,
    slot_idx: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_query_len: int,
    o: torch.Tensor | None = None,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    lower_bound: float | None = None,
) -> torch.Tensor:
    """ReplaySSM KDA decode step; arguments and layouts as ``replayssm_sigmoid_gating_delta_rule``.

    q, k ``[1, T, H, K]``, v ``[1, T, HV, V]``, a ``[1, T, HV, K]`` (gate input), b ``[1, T, HV]`` (beta logits),
    ckpt ``[slot, HV, V, K]``, records ``[slot, HV, cache_len, K|V]``.  Returns ``[1, T, HV, V]`` (``o`` if given,
    as ``[T, HV, V]``).
    """
    assert q.shape[0] == 1, "varlen layout expected (B == 1)"
    _, T_tot, _, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    if scale is None:
        scale = K**-0.5
    q, k, v, a, b = (x.contiguous() for x in (q, k, v, a, b))
    out = q.new_empty(1, T_tot, HV, V) if o is None else o.unsqueeze(0)
    _kda_replayssm_decode(
        out,
        ckpt,
        buf_k,
        buf_u,
        buf_g,
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        write_pos,
        slot_idx,
        cu_seqlens,
        int(max_query_len),
        float(scale),
        bool(use_qk_l2norm_in_kernel),
        lower_bound is not None,
        float(0.0 if lower_bound is None else lower_bound),
    )
    return out
