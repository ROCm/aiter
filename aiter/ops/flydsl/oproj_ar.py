# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host wrapper of the o_proj GEMV + k5 one-shot all-reduce + residual + RMSNorm launch
(``kernels/oproj_ar_gfx1250.py``)."""

from __future__ import annotations

import torch

from .kernels.tensor_shim import _preload_compiled, _run_compiled, current_stream_arg, ptr_arg

# the o_proj decode GEMM's table row for <= 64 rows (mxfp8_decode_gemm_gfx1250._TABLE)
_CFG = (32, 3, 16, 1, 1, 1, 1)  # tile_m, nb, kw, wn, wdepth, wcpol, atdm
_SYNCED = set()  # specializations compiled and synced across the TP group


def flydsl_oproj_ar(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    res: torch.Tensor,
    norm_w: torch.Tensor,
    eps: float,
    inbox: list[int],
    ctr: int,
    slot: int,
    rank: int,
    *,
    maxit: int = 20000000,
    max_rows: int = 256,
    out: torch.Tensor | None = None,
    res_out: torch.Tensor | None = None,
    timeline: torch.Tensor | None = None,
    stream=None,
):
    """x [m, K] e4m3 (+ row-major e8m0 x_scale), w / w_scale the o_proj weight in the decode GEMM's layout;
    res [m, N] bf16, norm_w [N] bf16 (Gemma: the kernel adds 1). ``inbox``: the four ranks' k5 inbox
    addresses, ``ctr`` this rank's k5 counter address, ``slot`` = max_rows * 768. Returns (out, res_out)."""
    import flydsl.expr as fx

    from .kernels.oproj_ar_gfx1250 import launch_oproj_ar

    m, k = x.shape
    n = w.shape[0]
    tile_m, nb, kw, wn, wdepth, wcpol, atdm = _CFG
    assert 0 < m <= tile_m and n == 6144 and len(inbox) == 4
    assert x.stride(1) == 1 and x.stride(0) % 16 == 0 and x_scale.is_contiguous()
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    if res_out is None:
        res_out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    if stream is None:
        stream = current_stream_arg()
    tl = 0 if timeline is None else 1
    spec = (n, k, tile_m, nb, kw, wn, wdepth, wcpol, atdm, max_rows, tl)
    args = (
        ptr_arg(x), ptr_arg(x_scale), ptr_arg(w), ptr_arg(w_scale), ptr_arg(res), ptr_arg(norm_w),
        ptr_arg(out), ptr_arg(res_out),
        __import__("flydsl.compiler", fromlist=["from_c_void_p"]).from_c_void_p(fx.Int32, int(ctr)),
        int(inbox[0]), int(inbox[1]), int(inbox[2]), int(inbox[3]),
        int(m), int(x.stride(0)), int(slot), int(rank), int(maxit), float(eps),
        stream,
        *spec,
        ptr_arg(timeline if timeline is not None else x, fx.Int32),
    )  # fmt: skip
    if spec not in _SYNCED:
        # first launch of this build: compile without launching, then barrier the TP group, so the peers' bounded
        # all-reduce polls never wait out a slower rank's JIT compile (no barrier inside a graph capture)
        _preload_compiled(launch_oproj_ar, *args)
        if torch.distributed.is_initialized() and not torch.cuda.is_current_stream_capturing():
            torch.distributed.barrier()
        _SYNCED.add(spec)
    _run_compiled(launch_oproj_ar, *args, specialization_key=("oproj_ar",) + spec)
    return out, res_out
