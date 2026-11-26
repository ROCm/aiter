# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gated MHA training backward with Q/K 192 and V 128 (OPUS, gfx950).

The op is the attention of the Kimi-K3 gated MLA layers on the expanded heads, with the output
gate fused. Sequences are packed, the mask is causal or none:

    S      = q @ k^T * scale                                  D_qk = 192
    P      = exp(S - lse)
    O      = P @ v                                            D_v  = 128
    out    = O * sigmoid(gate)
    dO     = dout * sigmoid(gate)
    dgate  = dout * out * (1 - sigmoid(gate))                 [total_q, H, D_v]
    delta  = rowsum(dout * out)                               = rowsum(dO * O)
    dP     = dO @ v^T
    dS     = P * (dP - delta) * scale
    dQ     = dS @ k                                           [total_q, H, D_qk]
    dK     = dS^T @ q                                         [total_k, H_k, D_qk]
    dV     = P^T @ dO                                         [total_k, H_k, D_v]

Without a gate, ``out = O`` and ``dO = dout``: the backward of the attention alone.

Two kernels: a pre-pass (dO, dgate and delta), then the dK / dV pass and the dQ pass as
workgroups of one kernel. No output is accumulated with atomics, so repeated runs give the
same bits. Around them, in torch: copies of inputs the kernels cannot read in place, and with
fewer KV heads (GQA / MQA) the sum of the per-query-head dK and dV over each KV head's query
heads into dk and dv.

The three kernel calls are torch ops whose schemas mark only the tensors they write; under
torch.compile they are opaque, and the op around them traces with fullgraph=True.
"""

import torch
from torch import Tensor

from ...jit.core import compile_ops
from ...jit.utils.chip_info import get_gfx_runtime_for_device
from ...jit.utils.torch_guard import torch_compile_guard

_NAME = "gated_mha_bwd"
_MD_NAME = "module_fmha_bwd_bf16_opus"
# The attention kernel's byte offsets within a tile are 32-bit (fmha_bwd_bf16_opus_kernels.cu).
_MAX_ROW_STRIDE = 1 << 24
_INT32_MAX = 2**31 - 1


# The module's bindings. They are listed in NONE_WRAPPED_OP: the torch ops below wrap them with
# schemas that mark the tensors they write, where compile_ops' would mark every tensor.
@compile_ops(_MD_NAME, fc_name="fmha_bwd_bf16_opus_gate")
def _fmha_bwd_bf16_opus_gate(
    dout: Tensor, out: Tensor, gate: Tensor, d_o: Tensor, dgate: Tensor, d: Tensor
) -> None: ...


@compile_ops(_MD_NAME, fc_name="fmha_bwd_bf16_opus_dot")
def _fmha_bwd_bf16_opus_dot(dout: Tensor, out: Tensor, d: Tensor) -> None: ...


@compile_ops(_MD_NAME, fc_name="fmha_bwd_bf16_opus_bwd")
def _fmha_bwd_bf16_opus_bwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    d_o: Tensor,
    lse: Tensor,
    d: Tensor,
    dq: Tensor,
    dk: Tensor,
    dv: Tensor,
    seqstart_q: Tensor,
    seqstart_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool,
    softmax_scale: float,
) -> None: ...


def _check_arch(x: Tensor) -> None:
    """Built for another architecture the kernels are empty: raise instead of returning the
    outputs they would leave unwritten. The architecture is that of x's device."""
    arch = get_gfx_runtime_for_device(x.device)
    if arch != "gfx950":
        raise RuntimeError(f"{_NAME} requires gfx950, got {arch} on {x.device}")


def _rows(x: Tensor) -> Tensor:
    """`x` [total, heads, D] if the kernels can read it in place, else a copy with dense strides.

    In place: rows of 16-byte aligned 8-element accesses, a token stride that covers a row (no
    broadcast or overlapping rows) within the attention kernel's 32-bit offsets, and strides
    that fit int32. `contiguous()` would keep the stride of a dim of size 1, a broadcast one
    included, so the copy is made with fresh strides.
    """
    sn, sh, sd = x.stride()
    ok = sd == 1 and x.data_ptr() % 16 == 0 and sn % 8 == 0 and sh % 8 == 0
    ok = ok and x.size(-1) <= sn <= _MAX_ROW_STRIDE and sh <= _INT32_MAX
    return x if ok else x.new_empty(x.shape).copy_(x)


def _writes_nothing_else(*args, **kwargs) -> None:
    return None


# The pre-pass with a gate: writes d_o, dgate and d = rowsum(dout * out).
@torch_compile_guard(mutates_args=["d_o", "dgate", "d"], gen_fake=_writes_nothing_else)
def _gated_mha_bwd_pre_gate(
    dout: Tensor, out: Tensor, gate: Tensor, d_o: Tensor, dgate: Tensor, d: Tensor
) -> None:
    _check_arch(dout)
    _fmha_bwd_bf16_opus_gate(_rows(dout), _rows(out), _rows(gate), d_o, dgate, d)


# The pre-pass without a gate: writes d = rowsum(dout * out).
@torch_compile_guard(mutates_args=["d"], gen_fake=_writes_nothing_else)
def _gated_mha_bwd_pre(dout: Tensor, out: Tensor, d: Tensor) -> None:
    _check_arch(dout)
    _fmha_bwd_bf16_opus_dot(_rows(dout), _rows(out), d)


# The dK / dV pass and the dQ pass in one launch: writes dq, dk and dv.
@torch_compile_guard(mutates_args=["dq", "dk", "dv"], gen_fake=_writes_nothing_else)
def _gated_mha_bwd_attn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    d_o: Tensor,
    lse: Tensor,
    d: Tensor,
    dq: Tensor,
    dk: Tensor,
    dv: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool,
    softmax_scale: float,
) -> None:
    _check_arch(q)
    # A query that sees no key (LSE -inf, a zero output) must get P = exp(S - LSE) = 0,
    # which LSE +inf gives. One cu_seqlens for both bounds has no such query; comparing
    # values would need a device sync.
    if cu_seqlens_q.data_ptr() != cu_seqlens_k.data_ptr():
        lse = lse.masked_fill(lse == float("-inf"), float("inf"))
    _fmha_bwd_bf16_opus_bwd(
        _rows(q),
        _rows(k),
        _rows(v),
        _rows(d_o),
        lse.contiguous(),
        d,
        dq,
        dk,
        dv,
        cu_seqlens_q.contiguous(),
        cu_seqlens_k.contiguous(),
        max_seqlen_q,
        max_seqlen_k,
        causal,
        softmax_scale,
    )


def gated_mha_bwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    dout: Tensor,
    lse: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    gate: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
    """Backward of ``out = attention(q, k, v) * sigmoid(gate)`` on packed sequences.

    Args:
        q: [total_q, H, 192] bf16. The queries of the forward.
        k: [total_k, H_k, 192] bf16. The keys. H is a multiple of H_k (GQA / MQA).
        v: [total_k, H_k, 128] bf16. The values.
        out: [total_q, H, 128] bf16. The output of the forward: the gated one, with a gate.
        dout: The gradient of ``out``, bf16, [total_q, H, 128] or [total_q, H * 128].
        lse: [H, total_q] float32. The natural-log LSE of the attention's forward. A query
            that sees no key has -inf.
        cu_seqlens_q: [num_seqs + 1] int32. The cumulative lengths of the query sequences.
        cu_seqlens_k: [num_seqs + 1] int32. The same for the keys. With a causal mask and
            another length than the queries', the mask is aligned to the bottom right.
        max_seqlen_q: An upper bound of the query sequences' lengths; 0 only without queries.
        max_seqlen_k: The same for the keys.
        softmax_scale: The scale of the forward.
        causal: The mask of the forward, causal or none.
        gate: Optional gate logits before the sigmoid, bf16, [total_q, H, 128] or
            [total_q, H * 128]. None is the attention alone.

    Returns:
        ``(dq, dk, dv, dgate)`` of the shapes of q, k, v and gate, bf16. ``dq`` and ``dk``
        include ``softmax_scale``. ``dgate`` is None without a gate. Keys that no query sees
        get zero gradients.

    Inputs of any strides are accepted: views with 16-byte aligned rows, strides that are
    multiples of 8 elements and a token stride of at least a row and at most 2^24 elements
    are read in place, others (broadcast or overlapping views among them) are copied.

    Limits:
        gfx950 only (the device of q). bf16 only. No dropout, bias, window or sink.
    """
    for name, cu in (("cu_seqlens_q", cu_seqlens_q), ("cu_seqlens_k", cu_seqlens_k)):
        if cu.dtype != torch.int32:
            raise ValueError(f"{_NAME}: {name} must be int32, got {cu.dtype}")
    (total_q, hq), (total_k, hk) = q.shape[:2], k.shape[:2]
    if hk == 0 or hq % hk != 0:
        raise ValueError(f"{_NAME}: q's {hq} heads must be a multiple of k's {hk}")
    dq, dk, dv = (
        torch.empty(x.shape, dtype=x.dtype, device=x.device) for x in (q, k, v)
    )
    dout = dout.reshape(out.shape)
    d = torch.empty(hq, total_q, dtype=torch.float32, device=q.device)
    dgate = None
    if gate is None:
        _gated_mha_bwd_pre(dout, out, d)
        d_o = dout
    else:
        d_o = torch.empty(out.shape, dtype=out.dtype, device=out.device)
        dgate = torch.empty(out.shape, dtype=out.dtype, device=out.device)
        _gated_mha_bwd_pre_gate(dout, out, gate.reshape(out.shape), d_o, dgate, d)
        dgate = dgate.view(gate.shape)
    # The kernel writes dk and dv a row per query head.
    ratio = hq // hk
    dk_q, dv_q = dk, dv
    if ratio > 1:
        dk_q = torch.empty(total_k, hq, k.size(-1), dtype=k.dtype, device=k.device)
        dv_q = torch.empty(total_k, hq, v.size(-1), dtype=v.dtype, device=v.device)
    _gated_mha_bwd_attn(
        q,
        k,
        v,
        d_o,
        lse,
        d,
        dq,
        dk_q,
        dv_q,
        cu_seqlens_q,
        cu_seqlens_k,
        int(max_seqlen_q),
        int(max_seqlen_k),
        bool(causal),
        float(softmax_scale),
    )
    if ratio > 1:
        dk.copy_(dk_q.view(total_k, hk, ratio, -1).sum(2, dtype=torch.float32))
        dv.copy_(dv_q.view(total_k, hk, ratio, -1).sum(2, dtype=torch.float32))
    return dq, dk, dv, dgate
