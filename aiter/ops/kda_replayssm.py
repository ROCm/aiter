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
    as ``[T, HV, V]``).  ``a`` and ``b`` may be column views of a wider buffer (read by token stride, no copy) as
    long as each token's row is dense.
    """
    assert q.shape[0] == 1, "varlen layout expected (B == 1)"
    _, T_tot, _, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    if scale is None:
        scale = K**-0.5
    q, k, v = (x.contiguous() for x in (q, k, v))
    if (
        a.stride(-1) != 1
        or a.stride(-2) != K
        or (a.stride(-3) * a.element_size()) % 16
        or a.data_ptr() % 16
    ):
        a = a.contiguous()
    if b.stride(-1) != 1:
        b = b.contiguous()
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


@compile_ops(
    "module_kda_replayssm_decode", fc_name="kda_replayssm_decode_conv", develop=True
)
def _kda_replayssm_decode_conv(
    o: torch.Tensor,
    ckpt: torch.Tensor,
    buf_k: torch.Tensor,
    buf_u: torch.Tensor,
    buf_g: torch.Tensor,
    conv_state: torch.Tensor,
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state_indices: torch.Tensor,
    conv_state_indices_in: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    write_pos: torch.Tensor,
    slot_idx: torch.Tensor,
    cu_seqlens: torch.Tensor,
    H: int,
    HV: int,
    max_query_len: int,
    scale: float,
    use_qk_l2norm: bool,
    use_lower_bound: bool,
    lower_bound: float,
) -> None:
    """Short conv + KDA ReplaySSM decode step (in place on conv_state, ckpt and the records; o is written)."""


def kda_replayssm_decode_conv_supported(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    num_heads: int,
    ckpt: torch.Tensor,
    buf_k: torch.Tensor,
    buf_u: torch.Tensor,
    buf_g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    max_query_len: int,
) -> bool:
    """Whether ``kda_replayssm_decode_conv`` handles this call: as ``kda_replayssm_decode_supported`` with H == HV
    = ``num_heads``, plus a bf16 [T, 3 * H * 128] input, a bf16 [lines, dim, >= 2 + window] conv state and contiguous
    fp32 [dim, 4] conv weights."""
    D = KDA_REPLAYSSM_HEAD_DIM
    dim = 3 * num_heads * D
    cap = buf_k.shape[2]
    return (
        num_heads > 0
        and x.dim() == 2
        and x.shape[1] == dim
        and x.dtype == torch.bfloat16
        and x.stride(-1) == 1
        and x.stride(0) % 4 == 0
        and x.data_ptr() % 8 == 0
        and conv_weight.dtype == torch.float32
        and tuple(conv_weight.shape) == (dim, 4)
        and conv_weight.is_contiguous()
        and _aligned(conv_weight)
        and conv_state.dtype == torch.bfloat16
        and conv_state.dim() == 3
        and conv_state.shape[1] == dim
        and conv_state.shape[2] >= 2 + max_query_len
        and 1 <= max_query_len <= KDA_REPLAYSSM_MAX_QUERY_LEN
        and 2 * max_query_len <= cap <= KDA_REPLAYSSM_MAX_CACHE_LEN
        and ckpt.dtype == torch.float32
        and A_log.dtype == torch.float32
        and dt_bias.dtype == torch.float32
        and buf_k.dtype == torch.bfloat16
        and buf_u.dtype == torch.bfloat16
        and buf_g.dtype in (torch.float32, torch.bfloat16)
        and ckpt.stride(-1) == 1
        and ckpt.stride(-2) == D
        and ckpt.stride(-3) == D * D
        and all(
            t.stride(-1) == 1 and t.stride(-2) % 8 == 0 for t in (buf_k, buf_u, buf_g)
        )
        and all(_aligned(t) for t in (ckpt, buf_k, buf_u, buf_g, dt_bias))
    )


def kda_replayssm_decode_conv(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
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
    num_heads: int,
    conv_state_indices_in: torch.Tensor | None = None,
    o: torch.Tensor | None = None,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    lower_bound: float | None = None,
) -> torch.Tensor:
    """Speculative-decoding short convolution (width 4, SiLU; the ``causal_conv1d_update`` call with
    ``num_accepted_tokens``) fused with the KDA ReplaySSM step.

    x ``[T, 3 * H * 128]`` is the mixed q | k | v input, conv_state ``[lines, dim, state_len]`` the speculative
    window (read at ``conv_state_indices_in`` / ``num_accepted_tokens - 1``, rewritten at ``conv_state_indices``),
    conv_weight ``[dim, 4]`` fp32; the rest as ``kda_replayssm_decode`` with ``H == HV == num_heads``.  Returns
    ``[T, HV, 128]`` (``o`` if given)."""
    D = KDA_REPLAYSSM_HEAD_DIM
    T_tot = x.shape[0]
    if scale is None:
        scale = D**-0.5
    if conv_state_indices_in is None:
        conv_state_indices_in = conv_state_indices
    if (
        a.stride(-1) != 1
        or a.stride(-2) != D
        or (a.stride(-3) * a.element_size()) % 16
        or a.data_ptr() % 16
    ):
        a = a.contiguous()
    if b.stride(-1) != 1:
        b = b.contiguous()
    out = x.new_empty(T_tot, num_heads, D) if o is None else o
    _kda_replayssm_decode_conv(
        out,
        ckpt,
        buf_k,
        buf_u,
        buf_g,
        conv_state,
        x,
        conv_weight,
        conv_state_indices,
        conv_state_indices_in,
        num_accepted_tokens,
        a,
        b,
        A_log,
        dt_bias,
        write_pos,
        slot_idx,
        cu_seqlens,
        int(num_heads),
        int(num_heads),
        int(max_query_len),
        float(scale),
        bool(use_qk_l2norm_in_kernel),
        lower_bound is not None,
        float(0.0 if lower_bound is None else lower_bound),
    )
    return out
