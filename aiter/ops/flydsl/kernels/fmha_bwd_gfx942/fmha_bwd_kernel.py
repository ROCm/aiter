# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host side of the FlyDSL varlen FMHA backward (d_qk=192, d_v=128, causal, bf16, gfx942).

Grid planning plus cached ``CompiledFunction`` dispatch for the kernels in
``fmha_bwd_core.py``: a ``k_delta`` pre-pass (D = rowsum(dO*O)) and one fused ``k_bwd`` that
carries all five backward GEMMs.  Unlike the CK and ASM backwards this needs neither a separate
``FmhaBwdOGradDotOKernel`` nor a ``FmhaBwdConvertQGradKernel``, and it does no work on zeros --
d_v = 128 is native, so v/out/dout/dv are never padded to 192.

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

from ..tensor_shim import _run_compiled
from .fmha_bwd_core import (
    BM2,
    BN1,
    DQK,
    DV,
    NT_RED,
    ROWS_DELTA,
    VEC_RED,
    build,
)

__all__ = ["flash_attn_varlen_bwd_d192_gfx942"]

# Split-K factor used on workloads whose whole grid is co-resident (see `_split`).  Swept on a
# T=8192 / H=2 / 5-sequence case (candidate us, 2-3 runs each):
#     nsp = 1  204.4 | 2  163.3/167.9/170.1 | 3  160.7/166.1/166.1 | 4  181.2 | 5  188.1
# 3 wins narrowly over 2 and both are far past 1; beyond 3 the extra partial-slab traffic and
# the per-split prologue/epilogue overhead outrun the shrinking critical path.
NSPLIT = 3

_FALLBACK_NUM_CU = 304  # MI300X/MI325X


@functools.lru_cache(maxsize=8)
def _num_cu(device_index: int) -> int:
    try:
        return int(torch.cuda.get_device_properties(device_index).multi_processor_count)
    except Exception:  # noqa: BLE001
        return _FALLBACK_NUM_CU


def _interleave(t: int, h: int, n_seqs: int, nb1: int, ncu: int) -> int:
    """Pick the k_bwd grid-decode mode: 1 = merged-LPT interleave of the dK/dV and dQ job lists,
    0 = the "all dK/dV blocks, then all dQ blocks" concatenation.

    ``cu_seqlens`` lives on the GPU, so the launcher cannot know the exact per-sequence tile
    counts without a device sync.  It does not need to: the number of NON-EMPTY workgroups is
    bounded host-side by the shapes alone, because sum_s ceil(len_s / BN1) <= T//BN1 + n_seqs
    (and is also <= n_seqs * nb1).  That bound is tight -- 269 vs the true 264 on the main case,
    69 vs 68 on the small one -- and needs no sync.

    Interleaving only pays when that whole work list is resident at once (<= ~2 dispatch rounds
    of one workgroup per CU, which is what the 44.3 KB LDS arena allows).  There the dispatch
    order decides which workgroups are CO-RESIDENT, and pairing the longest dK/dV job with the
    longest dQ job lets the two job types' causal tails overlap.  Deeper than that the kernel is
    work-bound and mixing the job types only widens the working set.

    Measured (forcing the mode on every case):
        case                     concat     interleave
        main_T32768_H2_13seq     691.7 us   688.8 us   (1076 WGs, 3.5 rounds -- noise)
        uniform_T32768_H2_8seq   724.5 us   738.4 us   (1056 WGs, 3.5 rounds -- REGRESSES)
        small_T8192_H2_5seq      217.0 us   205.8 us   ( 276 WGs, 0.9 rounds -- +5.4 %)
    """
    est_wgs = 2 * h * min(t // BN1 + n_seqs, n_seqs * nb1)
    return 1 if est_wgs <= 2 * ncu else 0


def _split(ilv: int) -> int:
    """Split-K factor along the STREAMED index.  Gated on exactly the same host-side predicate
    as `_interleave()`, so both 32K cases take the untouched nsp == 1 kernel.

    THE MEASUREMENT THAT MOTIVATES IT (T=8192, H=2, lens 900/1200/1700/2200/2192): 272 non-empty
    workgroups on 304 CUs, so the ENTIRE grid is co-resident and the kernel's makespan is the
    longest single workgroup, not the total work.  Job A's block 0 of the 2200-token sequence
    streams ceil(2200/32) = 69 query tiles; the work-balanced average is
    sum(len^2)/8192 * 2 heads * 2 job types / 304 CUs = 23.8 tiles.  204.7 us / 69 tiles =
    2.97 us per tile, i.e. the measured time IS the critical path to within a percent.  Job B is
    the same shape (its last query block streams all 69 key tiles), which is why "job A alone"
    was already 204.8 us of the 207.1 us fused kernel -- the two critical paths run concurrently
    and neither hides the other.

    No tile-size change can touch this: the workgroup that owns the FIRST keys must contract over
    every query in the sequence whatever its width, so BN1 128 -> 64 leaves block 0 at 69 tiles
    and merely doubles the workgroup count.  Only cutting the contraction shortens it.  With
    nsp = 2 the critical path drops to 35 tiles while the balanced average stays 23.8, so the
    makespan becomes max(23.8, 35) instead of 69.
    """
    return NSPLIT if ilv else 1


@functools.lru_cache(maxsize=128)
def _plan(t: int, h: int, n_seqs: int, max_seqlen_q: int, max_seqlen_k: int, ncu: int):
    """Everything about a call that depends only on its SHAPES, computed once per signature.

    Worth hoisting because the GPU is idle for all of it: the host work sits inside the measured
    device window, and on the main case it used to be 21.7 us of a 695.5 us call.
    """
    nb1 = (max_seqlen_k + BN1 - 1) // BN1
    nb2 = (max_seqlen_q + BM2 - 1) // BM2
    ilv = _interleave(t, h, n_seqs, nb1, ncu)
    nsp = _split(ilv)
    red_per_blk = NT_RED * VEC_RED
    return (
        (t * h + ROWS_DELTA - 1) // ROWS_DELTA,  # 0: k_delta grid
        nb1 * nsp,  # 1: job-A list length (the split index rides the low digit)
        (nb1 + nb2) * nsp,  # 2: total list length
        ilv,  # 3: grid-decode mode
        nsp,  # 4: split-K factor
        (t * h * DQK + red_per_blk - 1) // red_per_blk,  # 5: dq / dk reduction grid
        (t * h * DV + red_per_blk - 1) // red_per_blk,  # 6: dv reduction grid
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
    dev_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    ndblk, nb1, nbtot, ilv, nsp, nrblk_q, nrblk_v = _plan(
        t, h, n_seqs, int(max_seqlen_q), int(max_seqlen_k), _num_cu(dev_index)
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

    # Only the selected variant is compiled: `k_delta` does not depend on `nsp`, so the split-K
    # build supplies an identical `launch_delta` and the nsp == 1 kernel set is never built on a
    # shape that will not use it.
    kernels = build(nsp)
    launch_delta, launch_bwd = kernels[0], kernels[1]

    # k_delta goes out FIRST, before k_bwd's argument marshalling: the GPU is idle until the
    # first packet lands, so every microsecond of host work moved after this launch overlaps
    # k_delta's ~10 us of device time instead of adding to the latency.  (Merging the two
    # launches into a single @flyc.jit body was tried and is 3-5 us SLOWER for exactly this
    # reason -- it delays k_delta until k_bwd's operands have been marshalled too.)
    #
    # Tensors go in with their NATIVE rank: FlyDSL's buffer resource only needs the base pointer
    # plus the explicit `num_records` byte count the kernel computes from (Tlen, Hn), so
    # flattening them first would just add ATen dispatch to the critical path.
    _run_compiled(launch_delta, dout, out, delta, t, h, ndblk, stream)

    # ONE launch for both halves of the backward.  The two job lists share the grid's y axis so
    # the dQ jobs slot straight into the dK/dV causal drain; see `k_bwd`'s grid-decode comment.
    if nsp > 1:
        # bf16 partial workspaces, nsp slabs each.  No zeroing: every (token, d, split) slot is
        # written by exactly one workgroup.
        ws_dq = torch.empty(nsp * dq.numel(), dtype=dq.dtype, device=device)
        ws_dk = torch.empty(nsp * dk.numel(), dtype=dk.dtype, device=device)
        ws_dv = torch.empty(nsp * dv.numel(), dtype=dv.dtype, device=device)
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
            nb1,
            nbtot,
            n_seqs,
            ilv,
            stream,
        )
        _run_compiled(launch_red, ws_dq, dq, dq.numel(), nrblk_q, stream)
        _run_compiled(launch_red, ws_dk, dk, dk.numel(), nrblk_q, stream)
        _run_compiled(launch_red, ws_dv, dv, dv.numel(), nrblk_v, stream)
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
        nb1,
        nbtot,
        n_seqs,
        ilv,
        stream,
    )
    return dq, dk, dv, delta
