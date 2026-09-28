# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host side of the FlyDSL varlen FMHA backward (d_qk=192, d_v=128, causal, bf16, gfx942).

Grid planning plus cached ``CompiledFunction`` dispatch for the kernels in
``kernels/fmha_bwd_gfx942/fmha_bwd_core.py``: a ``k_delta`` pre-pass (D = rowsum(dO*O)) and one
fused ``k_bwd`` that carries all five backward GEMMs.  The device code stays under ``kernels/``
and everything host-side lives here.  Unlike the CK and ASM backwards this needs neither a
separate ``FmhaBwdOGradDotOKernel`` nor a ``FmhaBwdConvertQGradKernel``, and it does no work on
zeros -- d_v = 128 is native, so v/out/dout/dv are never padded to 192.

The launcher never synchronises with the device: the grid is sized purely from tensor shapes and
``max_seqlen_*`` (both host-side), and the per-sequence bounds are read from ``cu_seqlens`` on
the GPU inside the kernels.

Contract (asserted by the caller in ``aiter/ops/flydsl/fmha_kernels.py``):
  * causal, bf16, ``d_qk = 192``, ``d_v = 128``
  * THD varlen self-attention -- one ``cu_seqlens`` for both q and k, ``nhead_q == nhead_k``
  * q, k contiguous ``[T, H, 192]``; v, out, dout contiguous ``[T, H, 128]``;
    lse contiguous ``[H, T]`` fp32; cu_seqlens ``[B+1]`` int32
  * ``P = exp(softmax_scale * Q@K^T - LSE)`` (natural log, scale folded in, causal j <= i)
"""

from __future__ import annotations

import functools

import torch

from ...jit.utils.chip_info import get_cu_num
from .kernels.fmha_bwd_gfx942.fmha_bwd_core import (
    DQK,
    DV,
    KEYS_PER_WG,
    QUERIES_PER_WG,
    RED_THREADS,
    RED_VEC,
    ROWS_DELTA,
    build,
)
from .kernels.tensor_shim import _run_compiled

__all__ = ["flash_attn_varlen_bwd_d192_gfx942"]

# Split-K factor used on workloads whose whole grid is co-resident (see `_split`).
NSPLIT = 3


@functools.lru_cache(maxsize=1)
def _num_cu() -> int:
    try:
        return int(get_cu_num())
    except (RuntimeError, AssertionError, ValueError):
        return int(
            torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count
        )


def _interleave(t: int, h: int, n_seqs: int, n_dkdv_blocks: int, ncu: int) -> int:
    """Pick the k_bwd grid-decode mode: 1 = merged-LPT interleave of the dK/dV and dQ job lists,
    0 = the "all dK/dV blocks, then all dQ blocks" concatenation.
    """
    est_wgs = 2 * h * min(t // KEYS_PER_WG + n_seqs, n_seqs * n_dkdv_blocks)
    return 1 if est_wgs <= 2 * ncu else 0


def _split(interleave: int) -> int:
    """Split-K factor along the STREAMED index.  Gated on exactly the same host-side predicate as
    `_interleave()`, so both 32K cases take the untouched n_split == 1 kernel.
    """
    return NSPLIT if interleave else 1


@functools.lru_cache(maxsize=128)
def _plan(t: int, h: int, n_seqs: int, max_seqlen_q: int, max_seqlen_k: int, ncu: int):
    n_dkdv_blocks = (max_seqlen_k + KEYS_PER_WG - 1) // KEYS_PER_WG
    n_dq_blocks = (max_seqlen_q + QUERIES_PER_WG - 1) // QUERIES_PER_WG
    interleave = _interleave(t, h, n_seqs, n_dkdv_blocks, ncu)
    n_split = _split(interleave)
    elems_per_red_blk = RED_THREADS * RED_VEC
    return (
        (t * h + ROWS_DELTA - 1) // ROWS_DELTA,  # 0: k_delta grid
        # 1: dK/dV list length -- scaled by n_split so the split index rides the low digit
        n_dkdv_blocks * n_split,
        (n_dkdv_blocks + n_dq_blocks) * n_split,  # 2: total list length
        interleave,  # 3: grid-decode mode
        n_split,  # 4: split-K factor
        # 5, 6: reduction grids for the dq/dk and dv workspaces
        (t * h * DQK + elems_per_red_blk - 1) // elems_per_red_blk,
        (t * h * DV + elems_per_red_blk - 1) // elems_per_red_blk,
    )


def flash_attn_varlen_bwd_d192_gfx942(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    dq: torch.Tensor | None = None,
    dk: torch.Tensor | None = None,
    dv: torch.Tensor | None = None,
):
    """Run the FlyDSL varlen causal backward.  Returns ``(dq, dk, dv, softmax_d)``.

    ``dq`` / ``dk`` / ``dv`` are written in place when supplied and allocated otherwise.
    ``softmax_d`` is the ``[H, T]`` fp32 ``rowsum(dO*O)`` the CK and ASM backwards also return.
    """
    t, h, d_qk = q.shape
    assert d_qk == DQK and v.shape[-1] == DV, (
        f"FlyDSL gfx942 backward is specialised for d_qk={DQK}, d_v={DV}, "
        f"got {d_qk} / {v.shape[-1]}"
    )
    # The kernels size the LSE buffer resource as `nrow * 4` bytes and issue fp32 loads, so a
    # narrower dtype or a transposed layout would read out of bounds and silently corrupt P.
    assert softmax_lse.dtype == torch.float32 and tuple(softmax_lse.shape) == (h, t), (
        f"lse must be fp32 [{h}, {t}], "
        f"got {softmax_lse.dtype} {tuple(softmax_lse.shape)}"
    )
    device = q.device
    n_seqs = cu_seqlens.numel() - 1
    (
        delta_blocks,
        n_dkdv_blocks,
        n_blocks,
        interleave,
        n_split,
        red_blocks_qk,
        red_blocks_v,
    ) = _plan(t, h, n_seqs, int(max_seqlen_q), int(max_seqlen_k), _num_cu())
    ws_bytes = n_split * t * h * DQK * 2
    assert ws_bytes < (1 << 31), (
        f"FlyDSL gfx942 backward addresses tensors through 32-bit buffer descriptors; "
        f"T={t} x H={h} needs a {ws_bytes / 2**30:.2f} GiB dq/dk workspace (>= 2 GiB). "
        f"Split the batch so T*H < {(1 << 31) // (n_split * DQK * 2)}."
    )

    if dq is None:
        dq = torch.empty_like(q)
    if dk is None:
        dk = torch.empty_like(k)
    if dv is None:
        dv = torch.empty_like(v)
    if not softmax_lse.is_contiguous():
        softmax_lse = softmax_lse.contiguous()

    stream = torch.cuda.current_stream(device)
    delta = torch.empty((h, t), dtype=torch.float32, device=device)

    # Only the selected variant is compiled: `k_delta` does not depend on `n_split`, so the
    # split-K build supplies an identical `launch_delta` and the n_split == 1 kernel set is never
    # built on a shape that will not use it.
    kernels = build(n_split)
    launch_delta, launch_bwd = kernels[0], kernels[1]

    # k_delta goes out FIRST, before k_bwd's argument marshalling: the GPU is idle until the
    # first packet lands, so every microsecond of host work moved after this launch overlaps
    # k_delta's ~10 us of device time instead of adding to the latency.
    _run_compiled(launch_delta, dout, out, delta, t, h, delta_blocks, stream)

    if n_split > 1:
        ws_dq = torch.empty(n_split * dq.numel(), dtype=dq.dtype, device=device)
        ws_dk = torch.empty(n_split * dk.numel(), dtype=dk.dtype, device=device)
        ws_dv = torch.empty(n_split * dv.numel(), dtype=dv.dtype, device=device)
        launch_red = kernels[2]
        _run_compiled(
            launch_bwd,
            q,
            k,
            v,
            dout,
            softmax_lse,
            delta,
            cu_seqlens,
            ws_dq,
            ws_dk,
            ws_dv,
            t,
            h,
            float(softmax_scale),
            n_dkdv_blocks,
            n_blocks,
            n_seqs,
            interleave,
            stream,
        )
        _run_compiled(launch_red, ws_dq, dq, dq.numel(), red_blocks_qk, stream)
        _run_compiled(launch_red, ws_dk, dk, dk.numel(), red_blocks_qk, stream)
        _run_compiled(launch_red, ws_dv, dv, dv.numel(), red_blocks_v, stream)
        return dq, dk, dv, delta

    _run_compiled(
        launch_bwd,
        q,
        k,
        v,
        dout,
        softmax_lse,
        delta,
        cu_seqlens,
        dq,
        dk,
        dv,
        t,
        h,
        float(softmax_scale),
        n_dkdv_blocks,
        n_blocks,
        n_seqs,
        interleave,
        stream,
    )
    return dq, dk, dv, delta
