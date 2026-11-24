# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host side of the FlyDSL MHA backward for gfx1250, head dim 128 (batch mode, BSHD).

Launches the three kernels of ``kernels/fmha_bwd_gfx1250/fmha_bwd_hd128.py`` the way aiter's
asm path drives the gfx1250 asm backward:

  odo     D = rowsum(dO * O) -> softmax_d, and zeroes the fp32 dQ accumulator
  dqdkdv  persistent: one 128-thread workgroup per CU walks the work items (128-key block [a pair of
          blocks for causal: the kb fold], head, batch) in XCD-swizzled order
  convert dQ = bf16(dQ_acc * softmax_scale)

GQA / MQA follow the asm path too: dK / dV are produced per query head and summed over the
group afterwards. dQ is accumulated with atomics, so dQ is not bitwise deterministic (dK / dV
are).

Contract (``flash_attn_bwd_hd128_gfx1250_supported`` checks it):
  * gfx1250, bf16, head dim 128 for q / k / v, nhead_q % nhead_k == 0
  * q / k / v / out / dout are [b, s, h, 128] views with unit stride on the last dim and every
    other stride a multiple of 8 elements (TDM rows are copied 16 B at a time)
  * softmax_lse is fp32 [b, nhead_q, seqlen_q] with unit stride on the last dim, natural log,
    softmax scale folded in: P = exp(scale * Q K^T - lse)
  * no mask, or causal aligned to the bottom-right corner (aiter's convention); no dropout,
    alibi, bias, sliding window or sink
"""

from __future__ import annotations

import torch

from .kernels.fmha_bwd_gfx1250.fmha_bwd_hd128 import (
    HDIM,
    LOG2E,
    SEG,
    SUB_QO,
    build_convert,
    build_dqdkdv,
    build_odo,
)
from .kernels.tensor_shim import _run_compiled

__all__ = [
    "flash_attn_bwd_hd128_gfx1250",
    "flash_attn_bwd_hd128_gfx1250_supported",
]


def _strides_ok(t: torch.Tensor) -> bool:
    return (
        t.dim() == 4 and t.stride(-1) == 1 and all(s % 8 == 0 for s in t.stride()[:3])
    )


def flash_attn_bwd_hd128_gfx1250_supported(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    *,
    dropout_p: float = 0.0,
    window_size_left: int = -1,
    window_size_right: int = -1,
    causal: bool = False,
    alibi_slopes=None,
    bias=None,
    deterministic: bool = False,
    sink=None,
) -> bool:
    """True when ``flash_attn_bwd_hd128_gfx1250`` serves this call exactly."""
    from ...jit.utils.chip_info import get_gfx

    if get_gfx() != "gfx1250":
        return False
    if any(x is None for x in (dout, q, k, v, out, softmax_lse)):
        return False
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        return False
    b, sq, hq, d = q.shape
    _, sk, hk, _ = k.shape
    if d != HDIM or k.shape[-1] != HDIM or v.shape[-1] != HDIM:
        return False
    if tuple(v.shape) != (b, sk, hk, HDIM) or tuple(k.shape[:1]) != (b,):
        return False
    if tuple(out.shape) != (b, sq, hq, HDIM) or tuple(dout.shape) != (b, sq, hq, HDIM):
        return False
    if hk == 0 or hq % hk != 0:
        return False
    # dK / dV leave the kernel through buffer stores with 32-bit byte offsets: one head's rows of the
    # contiguous [b, sk, hq, d] fallback must fit
    if sk * hq * HDIM * 2 >= 2**31:
        return False
    if any(x.dtype != torch.bfloat16 for x in (dout, q, k, v, out)):
        return False
    if not all(_strides_ok(x) for x in (dout, q, k, v, out)):
        return False
    if softmax_lse.dtype != torch.float32 or tuple(softmax_lse.shape) != (b, hq, sq):
        return False
    if sq > 0 and softmax_lse.stride(-1) != 1:
        return False
    if (
        dropout_p > 0.0
        or alibi_slopes is not None
        or bias is not None
        or sink is not None
    ):
        return False
    if deterministic:
        return False
    wl = -1 if window_size_left >= sk else window_size_left
    wr = -1 if window_size_right >= sk else window_size_right
    if causal:
        return wl < 0
    return wl < 0 and wr < 0


def flash_attn_bwd_hd128_gfx1250(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    dq: torch.Tensor | None = None,
    dk: torch.Tensor | None = None,
    dv: torch.Tensor | None = None,
):
    """Batch-mode BSHD backward. Returns ``(dq, dk, dv, softmax_d)``.

    ``dq`` / ``dk`` / ``dv`` are written in place when given (any stride with a unit last dim and
    the other strides multiples of 8) and allocated otherwise. ``softmax_d`` is the fp32
    ``[b, nhead_q, seqlen_q]`` rowsum(dO * O) that the CK and asm backwards also return.
    """
    batch, seqlen_q, nhead, hdim = q.shape
    seqlen_k, nhead_k = k.shape[1], k.shape[2]
    assert hdim == HDIM and k.shape[-1] == HDIM and v.shape[-1] == HDIM
    assert nhead % nhead_k == 0
    ratio = nhead // nhead_k
    dev = q.device
    if dq is None:
        dq = torch.empty_like(q)
    if dk is None:
        dk = torch.empty_like(k)
    if dv is None:
        dv = torch.empty_like(v)
    softmax_d = torch.empty((batch, nhead, seqlen_q), dtype=torch.float32, device=dev)
    if batch == 0 or nhead == 0:
        return dq, dk, dv, softmax_d
    if seqlen_q == 0 or seqlen_k == 0:
        dq.zero_()
        dk.zero_()
        dv.zero_()
        softmax_d.zero_()
        return dq, dk, dv, softmax_d
    for t in (dq, dk, dv):
        assert _strides_ok(
            t
        ), "dq / dk / dv need a unit last stride and 16 B aligned rows"

    def fits(t):  # one head's rows within the kernel's 32-bit buffer offsets
        return seqlen_k * t.stride(1) * 2 < 2**31

    if ratio > 1 or not (fits(dk) and fits(dv)):
        dk_x = torch.empty((batch, seqlen_k, nhead, HDIM), dtype=q.dtype, device=dev)
        dv_x = torch.empty_like(dk_x)
    else:
        dk_x, dv_x = dk, dv
    sq_pad = (
        (seqlen_q + SUB_QO - 1) // SUB_QO * SUB_QO
    )  # dq_acc rows are written as whole tiles
    dq_acc = torch.empty((batch, nhead, sq_pad, HDIM), dtype=torch.float32, device=dev)
    stream = torch.cuda.current_stream(dev)

    _run_compiled(
        build_odo(),
        out,
        dout,
        softmax_d,
        dq_acc,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        dout.stride(0),
        dout.stride(1),
        dout.stride(2),
        softmax_d.stride(0),
        softmax_d.stride(1),
        seqlen_q,
        sq_pad,
        nhead,
        batch,
        stream,
    )
    _run_compiled(
        build_dqdkdv(bool(causal)),
        q,
        k,
        v,
        dout,
        softmax_lse,
        softmax_d,
        dq_acc,
        dk_x,
        dv_x,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        dout.stride(0),
        dout.stride(1),
        dout.stride(2),
        dk_x.stride(0),
        dk_x.stride(1),
        dk_x.stride(2),
        dv_x.stride(0),
        dv_x.stride(1),
        dv_x.stride(2),
        softmax_lse.stride(0),
        softmax_lse.stride(1),
        softmax_d.stride(0),
        softmax_d.stride(1),
        seqlen_q,
        seqlen_k,
        sq_pad,
        nhead,
        ratio,
        float(softmax_scale) * LOG2E,
        float(softmax_scale),
        SEG,
        batch,
        stream,
    )
    _run_compiled(
        build_convert(),
        dq_acc,
        dq,
        dq.stride(0),
        dq.stride(1),
        dq.stride(2),
        seqlen_q,
        sq_pad,
        nhead,
        batch,
        float(softmax_scale),
        stream,
    )
    if ratio > 1:
        torch.sum(dk_x.view(batch, seqlen_k, nhead_k, ratio, HDIM), dim=3, out=dk)
        torch.sum(dv_x.view(batch, seqlen_k, nhead_k, ratio, HDIM), dim=3, out=dv)
    elif dk_x is not dk:
        dk.copy_(dk_x)
        dv.copy_(dv_x)
    return dq, dk, dv, softmax_d
